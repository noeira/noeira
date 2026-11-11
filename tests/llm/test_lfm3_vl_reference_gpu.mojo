"""LFM3-VL's vision tower (`noeira.llm.lfm3.Lfm3Vision`) and d1 with pictures,
against d1-2B's own code (docs/NN_MODEL_PORTS_PLAN.md §12.5, step 6).

    build on the Orin: build-jetson tests/llm/test_lfm3_vl_reference_gpu.mojo [-D LFM3_BF16=1]
    ./test_lfm3_vl_reference_gpu [d1 dir] [ref dir]
      defaults: ~/.cache/noeira/local-ai/d1/d1-2B and ~/.cache/noeira/d1/ref_vl

The reference is `tools/d1/dump_d1_vl_reference.py` (d1's System One in fp32,
pictures in `media`). For each case:

1. PATCHES: the pixel values made here from the raw picture equal the
   processor's (bicubic resize to the same size, rescale, normalise, merge-block
   patch order), max |d|.
2. TOWER: the embedding, each of the 14 layers and the projector's output, for
   each picture's slice — max |d| / max |reference|.
3. PROMPT: the rows d1 packs WITH the pictures' markup — ids, groups, state
   length, slots — exactly.
4. BACKBONE: the embedding with the features in place, each of the 24 layers,
   the final norm and the slots' log-softmax.
5. ANSWERS: `system_one` with the pictures: the same picks, every probability
   within the bound.

Every vision tensor is used. Falsification on `vl_one` (non-square: 30 x 40
patches): patches in raster order, the conv's neighbours in token order, the
RoPE's axes swapped — each must move the projector's output ten times past the
bound; and the features left out (the text embeddings of `<|image|>` instead)
must move the slots ten times past theirs.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.json import load_json, parse_json
from noeira.io.npy import load_npy_f64
from noeira.llm.d1 import D1
from noeira.llm.lfm3 import Picture
from noeira.llm.lfm3.model import LF_BF16
from noeira.llm.lfm3.vision import VIS_BF16, LF_VIS_TF32
from noeira.nn.constants import DT


comptime PIXEL_BOUND: Float64 = 1e-6
comptime VIS_BOUND: Float64 = 0.2 if VIS_BF16 else (7e-2 if LF_BF16 else 1e-4)
"""The tower's layers and the projector. fp32: measured <= 2.9e-5 (its last
layer adds an update 30x the stream). bf16 (the bf16 build's default): measured
0.13 / 0.089 — the same order as TORCH'S OWN bf16 tower against its fp32 (0.068
/ 0.118 at the last layer, 0.049 / 0.101 at the projector, these pictures, the
Orin 2026-11-11): the drift d1 has as shipped. `-D LFM3_VIS_TF32=1` (fp32
activations, TF32 GEMMs): measured 0.023 / 0.033."""
comptime STAGE_BOUND: Float64 = 3e-2 if LF_BF16 else 1e-4
"""The backbone's TEXT rows (every row a question reads: slots sit there).
bf16: measured <= 1.8e-2."""
comptime IMAGE_ROW_BOUND: Float64 = 2e-3
"""The backbone's `<|image|>` rows, fp32 only, max |d| / max |reference| of
the whole stage. Their INPUT is the tower's output, already ~3e-5 off: the
tower's last layer adds an update 30x the stream (absolute max 30 against 1.2,
the reference's own), whose rounding is 30x larger too; 24 layers then amplify
it on those rows (measured: 1.3e-4 at layer 23, 8.4e-4 after the final norm,
against <= 3.2e-5 on text rows; the Orin, 2026-11-11). The bound is twice that.
The slots stay at ~1e-6: the questions read through it. In the bf16 builds
these rows carry the tower's drift (above) and are not gated."""
comptime SLOT_BOUND: Float64 = 5e-3 if LF_BF16 else 1e-4
"""bf16: measured <= 1.1e-3."""
comptime PROB_BOUND: Float64 = 5e-3 if LF_BF16 else 1e-4
"""bf16: measured <= 2.6e-3 (TF32 tower: 1.1e-3)."""
comptime FALSIFY_FACTOR: Float64 = 10.0
comptime VL_ONE_STATE = "The robot's head camera sees this."
comptime VL_ONE_Q = """{"door_open": {"type": "noul", "instructions": "Is there an open door in the picture?"}, "furniture": {"type": "choice", "instructions": "Which of these is in the picture?", "criteria": {"sofa": "a sofa", "bed": "a bed", "fridge": "a fridge", "car": "a car"}}, "light": {"type": "score", "instructions": "How bright is the room?", "criteria": ["dark", "dim", "bright"]}}"""
comptime VL_TWO_STATE = """{"robot": {"at": "door"}, "heard": "go to the bed", "pictures": "first: ahead, second: behind"}"""
comptime VL_TWO_Q = """{"bed_first": {"type": "noul", "instructions": "Is there a bed in the first picture?"}, "where": {"type": "choice", "instructions": "Where should the robot walk?", "criteria": {"ahead": "walk ahead", "behind": "turn around and walk", "stay": "do not walk anywhere"}}}"""


def rel_err(got: List[Float32], want: List[Float64], off: Int = 0, n: Int = -1) raises -> Float64:
    var count = len(want) - off if n < 0 else n
    if len(got) != count:
        raise Error("size " + String(len(got)) + " against the reference's " + String(count))
    var d = 0.0
    var m = 0.0
    for i in range(count):
        d = max(d, abs(Float64(got[i]) - want[off + i]))
        m = max(m, abs(want[off + i]))
    return d / m if m > 0.0 else d


def max_abs(got: List[Float32], want: List[Float64], off: Int, n: Int) raises -> Float64:
    if len(got) != n:
        raise Error("size " + String(len(got)) + " against " + String(n))
    var d = 0.0
    for i in range(n):
        d = max(d, abs(Float64(got[i]) - want[off + i]))
    return d


def split_err(got: List[Float32], want: List[Float64], ids: List[Int], image_token: Int, dim: Int) raises -> Tuple[Float64, Float64, Float64]:
    """max |d| / max |reference| over the image-token rows and over the others,
    and the worst row-relative error (|d| / that row's own max) on the image rows."""
    var di = 0.0
    var dt = 0.0
    var m = 0.0
    var row_rel = 0.0
    for r in range(len(ids)):
        var rd = 0.0
        var rm = 0.0
        for j in range(dim):
            var e = abs(Float64(got[r * dim + j]) - want[r * dim + j])
            rd = max(rd, e)
            rm = max(rm, abs(want[r * dim + j]))
        m = max(m, rm)
        if ids[r] == image_token:
            di = max(di, rd)
            row_rel = max(row_rel, rd / rm if rm > 0.0 else rd)
        else:
            dt = max(dt, rd)
    return (di / m, dt / m, row_rel)


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for x in s.as_bytes():
        b.append(x)
    return b^


def pictures_of(dir: String) raises -> List[Picture]:
    var grid = load_npy_f64(dir + "/grid_thw.npy")
    var out = List[Picture]()
    for k in range(len(grid) // 3):
        var h = Int(grid[k * 3 + 1]) * 16
        var w = Int(grid[k * 3 + 2]) * 16
        var px = load_npy_f64(dir + "/image" + String(k) + ".npy")
        var rgb = List[UInt8](capacity=len(px))
        for v in px:
            rgb.append(UInt8(Int(v)))
        out.append(Picture(rgb^, h, w))
    return out^


def main() raises:
    var home = getenv("HOME")
    var args = argv()
    var model_dir = String(args[1]) if len(args) > 1 else home + "/.cache/noeira/local-ai/d1/d1-2B"
    var ref_dir = String(args[2]) if len(args) > 2 else home + "/.cache/noeira/d1/ref_vl"
    var fails = 0
    print("LFM3-VL vision tower + d1 with pictures vs d1's own code",
          "(fp32)" if not LF_BF16 else ("(LFM3_BF16, TF32 tower)" if LF_VIS_TF32 else "(LFM3_BF16)"))
    var t0 = perf_counter_ns()
    var d1 = D1.load(DeviceContext(), model_dir, vision=True)
    print("  d1 + vision loaded in", (perf_counter_ns() - t0) // 1_000_000, "ms")
    var dm = d1.vision.value().cfg.hidden
    var th = d1.vision.value().cfg.text_hidden
    var image_token = d1.vision.value().cfg.image_token
    var names: List[String] = ["vl_one", "vl_two"]
    var states: List[String] = [VL_ONE_STATE, VL_TWO_STATE]
    var is_json: List[Bool] = [False, True]
    var questions: List[String] = [VL_ONE_Q, VL_TWO_Q]

    for c in range(2):
        var dir = ref_dir + "/" + names[c]
        print(" ", names[c])
        var pics = pictures_of(dir)
        var pv = load_npy_f64(dir + "/pixel_values.npy")
        var per = 3 * 2 * 16 * 16
        # 1. patches and 2. the tower, picture by picture
        var poff = 0
        var toff = 0
        var worst_px = 0.0
        var worst_vis = 0.0
        var worst_at = String("")
        for k in range(len(pics)):
            var n = (pics[k].height // 16) * (pics[k].width // 16)
            var px = d1.vision.value().patchify(pics[k])
            worst_px = max(worst_px, max_abs(d1.lm.download(px, n * per), pv, poff * per, n * per))
            var taps = List[DeviceBuffer[DT]]()
            var feats = d1.vision.value().encode(pics[k], taps, True)
            _ = feats^
            var stages = List[String]()
            stages.append("vis_embed")
            for l in range(14):
                stages.append("vis_layer" + String(l))
            for s in range(len(stages)):
                var want = load_npy_f64(dir + "/" + stages[s] + ".npy")
                var e = rel_err(d1.lm.download(taps[s], n * dm), want, poff * dm, n * dm)
                if getenv("VL_DIAG") != "":
                    print("     diag", stages[s], "picture", k, " rel err", e)
                if e > worst_vis:
                    worst_vis = e
                    worst_at = stages[s] + " (picture " + String(k) + ")"
            var wp = load_npy_f64(dir + "/vis_proj.npy")
            var ep = rel_err(d1.lm.download(taps[len(taps) - 1], n // 4 * th), wp, toff * th, n // 4 * th)
            if ep > worst_vis:
                worst_vis = ep
                worst_at = "vis_proj (picture " + String(k) + ")"
            poff += n
            toff += n // 4
        check(fails, worst_px <= PIXEL_BOUND, String(len(pics)) + " picture(s): pixel values max |d| = " + String(worst_px))
        check(fails, worst_vis <= VIS_BOUND, "tower + projector, worst " + worst_at + "  rel err = " + String(worst_vis))
        # 3. the prompt and 5. the answers
        var got_s = d1.system_one(states[c], is_json[c], questions[c], pics)
        var meta = load_json(dir + "/row0_meta.json")
        var want_ids = load_npy_f64(dir + "/row0_ids.npy")
        var want_g = load_npy_f64(dir + "/row0_question.npy")
        var r0 = d1.last_rows[0].copy()
        var same = len(d1.last_rows) == 1 and len(r0.ids) == len(want_ids)
        if same:
            for i in range(len(want_ids)):
                if r0.ids[i] != Int(want_ids[i]) or r0.groups[i] != Int(want_g[i]):
                    same = False
        same = same and r0.state_len == meta.integer(meta.field(meta.root(), "state_len"))
        var ws = meta.field(meta.root(), "slots")
        var k = 0
        for q in range(meta.size(ws)):
            for j in range(meta.size(meta.at(ws, q))):
                same = same and k < len(r0.slots) and r0.slots[k] == meta.integer(meta.at(meta.at(ws, q), j))
                k += 1
        same = same and k == len(r0.slots)
        check(fails, same, "the packed row with the pictures' markup: " + String(len(r0.ids)) + " ids, state "
              + String(r0.state_len) + ", groups and slots exact")
        var got = parse_json(_bytes(got_s))
        var want = load_json(dir + "/answers.json")
        var ga = got.field(got.root(), "answers")
        var wa = want.field(want.root(), "answers")
        var worst_p = 0.0
        var picks = got.size(ga) == want.size(wa)
        for i in range(want.size(wa)):
            var nm = want.key_at(wa, i)
            var w = want.at(wa, i)
            var g = got.field(ga, nm)
            var kind = want.string(want.field(w, "type"))
            if kind == "noul":
                worst_p = max(worst_p, abs(got.number(got.field(g, "noul")) - want.number(want.field(w, "noul"))))
                continue
            if kind == "choice" and got.string(got.field(g, "choice")) != want.string(want.field(w, "choice")):
                picks = False
            if kind == "score" and got.number(got.field(g, "score")) != want.number(want.field(w, "score")):
                picks = False
            var wp = want.field(w, "probabilities")
            var gp = got.field(g, "probabilities")
            for j in range(want.size(wp)):
                worst_p = max(worst_p, abs(got.number(got.field(gp, want.key_at(wp, j))) - want.number(want.at(wp, j))))
        check(fails, picks and worst_p <= PROB_BOUND, String(want.size(wa)) + " answers, same picks, worst probability |d| = "
              + String(worst_p))
        # 4. the backbone with the features in place
        var feats = d1.vision.value().encode_all(pics)
        var taps = List[DeviceBuffer[DT]]()
        var t = len(r0.ids)
        var h = d1.lm.forward(r0.ids, r0.groups, r0.state_len, taps, True, image_feats=Optional(feats), image_token=image_token)
        # text rows (what the questions read) to the text gate's bound; image rows to IMAGE_ROW_BOUND
        var e0 = split_err(d1.lm.download(taps[0], t * th), load_npy_f64(dir + "/row0_embed.npy"), r0.ids, image_token, th)
        var wi = e0[0]
        var wt = e0[1]
        var wi_at = String("the embedding")
        var wt_at = String("the embedding")
        for l in range(25):
            var got_l = d1.lm.download(taps[l + 1], t * th) if l < 24 else d1.lm.download(h, t * th)
            var path = dir + ("/row0_layer" + String(l) + ".npy" if l < 24 else "/row0_final.npy")
            var se = split_err(got_l, load_npy_f64(path), r0.ids, image_token, th)
            var at = "layer " + String(l) if l < 24 else String("the final norm")
            if getenv("VL_DIAG") != "":
                print("     diag", at, " image rows", se[0], " text rows", se[1], " image row-relative", se[2])
            if se[0] > wi:
                wi = se[0]
                wi_at = at
            if se[1] > wt:
                wt = se[1]
                wt_at = at
        check(fails, wt <= STAGE_BOUND, "backbone, TEXT rows: embedding, 24 layers, final norm; worst " + wt_at
              + "  rel err = " + String(wt))
        comptime if not LF_BF16:
            check(fails, wi <= IMAGE_ROW_BOUND, "backbone, IMAGE rows (features in place); worst " + wi_at
                  + "  rel err = " + String(wi))
        else:
            print("  (not gated) backbone, IMAGE rows: worst", wi_at, " rel err =", wi, "- the tower's drift, carried")
        var want_logz = load_npy_f64(dir + "/row0_slot_logz.npy")
        var es = rel_err(d1.lm.slot_logz(h, t, r0.slots), want_logz)
        check(fails, es <= SLOT_BOUND, "slots' log-softmax over the vocabulary  rel err = " + String(es))

        if c == 0:
            var tu = d1.vision.value().unused()
            check(fails, len(tu) == 0, "every vision tensor used (" + String(len(tu)) + " unused)")
            # falsification
            var wp = load_npy_f64(dir + "/vis_proj.npy")
            var n0 = (pics[0].height // 16) * (pics[0].width // 16)
            var bits: List[Int] = [1, 2, 4]
            var what: List[String] = ["patches in raster order", "conv neighbours in token order", "RoPE axes swapped"]
            for b in range(3):
                comptime if LF_BF16:
                    print("  (skip) defect", what[b], "- the fp32 build gates the tower's defects (the bf16 drift is 0.1)")
                    continue
                d1.vision.value().debug = bits[b]
                var none = List[DeviceBuffer[DT]]()
                var f = d1.vision.value().encode(pics[0], none)
                d1.vision.value().debug = 0
                var e = rel_err(d1.lm.download(f, n0 // 4 * th), wp, 0, n0 // 4 * th)
                check(fails, e >= FALSIFY_FACTOR * VIS_BOUND, "defect caught: " + what[b] + "  projector rel err = " + String(e))
            var none2 = List[DeviceBuffer[DT]]()
            var h0 = d1.lm.forward(r0.ids, r0.groups, r0.state_len, none2)
            var e0 = rel_err(d1.lm.slot_logz(h0, t, r0.slots), want_logz)
            check(fails, e0 >= FALSIFY_FACTOR * SLOT_BOUND, "defect caught: the features left out  slots rel err = " + String(e0))
        # timing (warm)
        var t1 = perf_counter_ns()
        _ = d1.system_one(states[c], is_json[c], questions[c], pics)
        print("     system_one with", len(pics), "picture(s),", t, "tokens, warm:", Float64(perf_counter_ns() - t1) / 1e6, "ms")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
