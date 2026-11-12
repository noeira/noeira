"""LFM3-Omni's audio input (`noeira.audio.lfm3_mel`, `noeira.llm.lfm3.Lfm3Audio`)
and d1 with heard clips, against d1-Omni-2B's own code (docs/NN_MODEL_PORTS_PLAN.md
§12.6, step 2).

    build on the Orin: jetson_build.sh <name> tests/llm/test_lfm3_audio_reference_gpu.mojo [-D LFM3_BF16=1]
    ./<name> [d1-Omni dir] [ref dir]
      defaults: ~/.cache/noeira/local-ai/d1/d1-Omni-2B and ~/.cache/noeira/d1/omni_ref

The reference is `tools/d1/dump_d1_omni_reference.py` (d1-Omni's System One in
fp32, the clip in `media`). For each case:

1. MEL: the front end here on the reference's waveform against the same
   features in FLOAT64 (`mel_f64.npy`, numpy's rfft) and against the
   reference's `input_features`, each with a SPLIT bound: tight where the
   features are above -1 (mel power > 1e-8), loose in the near-silent bins. A
   bin with ~1e-12 of power is a cancellation, sensitive to everything: torch's
   float32 STFT is 7.8e-4 off float64 there (3e-5 above -1), and this window
   (`hann_window`, torch's float32 values) differs from torch's in 11 of 400
   values by one ulp (its vectorised cosine), which moves those bins by 8e-5
   and the others by 5e-6 (measured on the Orin with numpy, both windows in
   float64). Those bins are silence; the encoder end to end shows it (stage 2).
   The features live in [-2, ~1.5].
2. ENCODER, from the REFERENCE's mel (the stage isolated): the downsampler, each
   of the 17 encoder layers, the final norm and the projector — max |d| / max
   |reference|; then the projector from THIS front end's mel (end to end).
3. PROMPT: the row d1 packs with the clip's markup — ids, groups, state length,
   slots — exactly.
4. BACKBONE: the embedding with the features in place, each of the 24 layers,
   the final norm (audio rows and text rows apart) and the slots' log-softmax.
5. ANSWERS: `system_one` with the clip: the same picks, every probability
   within the bound.

Every audio tensor is used. Falsification on `heard_long` (~60 tokens, past the
32-token window): the window dropped, the attention bidirectional, the flatten
mel-major, the RoPE off — each must move the projector's output ten times past
the bound; a non-periodic window must move the mel ten times past its bound;
and the features left out must move the slots ten times past theirs.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.audio.lfm3_mel import Lfm3LogMel
from noeira.audio.mel import hann_window
from noeira.io.json import load_json, parse_json
from noeira.io.npy import load_npy_f64
from noeira.llm.d1 import D1
from noeira.llm.lfm3.model import LF_BF16
from noeira.nn.constants import DT


comptime MEL_BOUND: Float64 = 3e-5
"""Against float64 (`mel_f64.npy`) where the features are above -1: measured 1.14e-5 (heard_fr)."""
comptime MEL_LOW_BOUND: Float64 = 2e-4
"""Against float64 everywhere: measured 8.0e-5 (the window's ulps, module note)."""
comptime MEL_REF_BOUND: Float64 = 2e-4
"""Against the reference where its features are above -1: measured 1.05e-4."""
comptime MEL_REF_LOW_BOUND: Float64 = 2e-3
"""Against the reference everywhere (its float32 STFT): measured 7.8e-4."""
comptime ENC_BOUND: Float64 = 2e-2 if LF_BF16 else 1e-4
"""The encoder's stages and the projector. fp32: measured <= 5.4e-6 (from the
reference's mel), 2.4e-5 end to end. bf16 (tensor-core GEMMs on bf16 operands):
measured <= 5.4e-3 / 5.5e-3."""
comptime STAGE_BOUND: Float64 = 3e-2 if LF_BF16 else 1e-4
comptime AUDIO_ROW_BOUND: Float64 = 2e-3
comptime SLOT_BOUND: Float64 = 5e-3 if LF_BF16 else 1e-4
comptime PROB_BOUND: Float64 = 5e-3 if LF_BF16 else 1e-4
comptime FALSIFY_FACTOR: Float64 = 10.0
comptime ROOM_STATE = """{"robot": {"at": "door", "facing": "kitchen", "doing": "standing"}, "places": {"fridge": "3.1 m ahead, slightly left", "door": "here", "table": "2.0 m to the right", "sofa": "4.5 m behind"}, "heard": "the attached recording", "recently": ["walked to the door"]}"""
comptime QUESTIONS = """{"addressed": {"type": "noul", "instructions": "Is the speaker asking the robot to do something?"}, "destination": {"type": "choice", "instructions": "Where should the robot walk?", "criteria": {"fridge": "walk to the fridge", "door": "walk to the door", "table": "walk to the table", "sofa": "walk to the sofa", "nowhere": "do not walk anywhere"}}, "urgency": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["not urgent", "somewhat urgent", "very urgent"]}}"""


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


def max_abs(got: List[Float32], want: List[Float64]) raises -> Float64:
    if len(got) != len(want):
        raise Error("size " + String(len(got)) + " against " + String(len(want)))
    var d = 0.0
    for i in range(len(want)):
        d = max(d, abs(Float64(got[i]) - want[i]))
    return d


def split_err(got: List[Float32], want: List[Float64], ids: List[Int], token: Int, dim: Int) raises -> Tuple[Float64, Float64]:
    """The max |d| / max |reference| over the `token` rows and over the others."""
    var da = 0.0
    var dt = 0.0
    var m = 0.0
    for r in range(len(ids)):
        var rd = 0.0
        for j in range(dim):
            rd = max(rd, abs(Float64(got[r * dim + j]) - want[r * dim + j]))
            m = max(m, abs(want[r * dim + j]))
        if ids[r] == token:
            da = max(da, rd)
        else:
            dt = max(dt, rd)
    return (da / m, dt / m)


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


def f32(v: List[Float64]) -> List[Float32]:
    var out = List[Float32](capacity=len(v))
    for x in v:
        out.append(Float32(x))
    return out^


def main() raises:
    var home = getenv("HOME")
    var args = argv()
    var model_dir = String(args[1]) if len(args) > 1 else home + "/.cache/noeira/local-ai/d1/d1-Omni-2B"
    var ref_dir = String(args[2]) if len(args) > 2 else home + "/.cache/noeira/d1/omni_ref"
    var fails = 0
    print("LFM3-Omni audio input + d1 with clips vs d1-Omni's own code", "(LFM3_BF16)" if LF_BF16 else "(fp32)")
    var t0 = perf_counter_ns()
    var d1 = D1.load(DeviceContext(), model_dir, audio=True)
    print("  d1-Omni + audio loaded in", (perf_counter_ns() - t0) // 1_000_000, "ms")
    var dm = d1.audio.value().cfg.hidden
    var th = d1.audio.value().cfg.text_hidden
    var audio_token = d1.audio.value().cfg.audio_token
    var nl = len(d1.audio.value().cfg.layer_types)
    var names: List[String] = ["heard_en", "heard_fr", "heard_neg", "heard_long"]
    var fe = Lfm3LogMel()

    for c in range(len(names)):
        var dir = ref_dir + "/" + names[c]
        print(" ", names[c])
        var wave = f32(load_npy_f64(dir + "/wave.npy"))
        var want_mel = load_npy_f64(dir + "/mel.npy")
        var mel64 = load_npy_f64(dir + "/mel_f64.npy")
        var frames = fe.frames(len(wave))
        # 1. the front end
        var t1 = perf_counter_ns()
        var mel = fe.extract(wave)
        var mel_ms = Float64(perf_counter_ns() - t1) / 1e6
        var hi64 = 0.0
        var lo64 = 0.0
        var hi = 0.0
        var lo = 0.0
        for i in range(len(want_mel)):
            var d64 = abs(Float64(mel[i]) - mel64[i])
            var d = abs(Float64(mel[i]) - want_mel[i])
            lo64 = max(lo64, d64)
            lo = max(lo, d)
            if mel64[i] > -1.0:
                hi64 = max(hi64, d64)
            if want_mel[i] > -1.0:
                hi = max(hi, d)
        check(fails, hi64 <= MEL_BOUND and lo64 <= MEL_LOW_BOUND, String(len(wave)) + " samples -> " + String(frames)
              + " frames: mel vs float64 max |d| " + String(hi64) + " where > -1, " + String(lo64) + " everywhere ("
              + String(mel_ms) + " ms on the CPU)")
        check(fails, hi <= MEL_REF_BOUND and lo <= MEL_REF_LOW_BOUND, "mel vs the reference (float32 STFT): max |d| "
              + String(hi) + " where its features > -1, " + String(lo) + " everywhere")
        # 2. the encoder, from the reference's mel, stage by stage
        var t = d1.audio.value().tokens(len(wave))
        var taps = List[DeviceBuffer[DT]]()
        var feats = d1.audio.value().encode(f32(want_mel), frames, taps, True)
        _ = feats^
        var worst = 0.0
        var worst_at = String("")
        for s in range(len(taps)):
            var stage: String
            var width = dm
            if s == 0:
                stage = "down"
            elif s <= nl:
                stage = "enc_layer" + String(s - 1)
            elif s == nl + 1:
                stage = "enc_final"
            else:
                stage = "proj"
                width = th
            var e = rel_err(d1.lm.download(taps[s], t * width), load_npy_f64(dir + "/" + stage + ".npy"))
            if getenv("AUD_DIAG") != "":
                print("     diag", stage, " rel err", e)
            if e > worst:
                worst = e
                worst_at = stage
        check(fails, worst <= ENC_BOUND, String(t) + " tokens: downsampler, " + String(nl)
              + " layers, final norm, projector; worst " + worst_at + "  rel err = " + String(worst))
        var none = List[DeviceBuffer[DT]]()
        var want_proj = load_npy_f64(dir + "/proj.npy")
        var t2 = perf_counter_ns()
        var fe2 = d1.audio.value().encode_wave(wave, none)
        var e2e = rel_err(d1.lm.download(fe2, t * th), want_proj)
        var enc_ms = Float64(perf_counter_ns() - t2) / 1e6
        check(fails, e2e <= ENC_BOUND, "end to end (this front end's mel): projector rel err = " + String(e2e)
              + " (" + String(enc_ms) + " ms, mel included)")
        # 3. the prompt and 5. the answers
        var clips = List[List[Float32]]()
        clips.append(wave.copy())
        var got_s = d1.system_one(ROOM_STATE, True, QUESTIONS, clips=clips)
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
        check(fails, same, "the packed row with the clip's markup: " + String(len(r0.ids)) + " ids (" + String(t)
              + " audio), state " + String(r0.state_len) + ", groups and slots exact")
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
        var af = d1.audio.value().encode_all(clips)
        var btaps = List[DeviceBuffer[DT]]()
        var n = len(r0.ids)
        var h = d1.lm.forward(r0.ids, r0.groups, r0.state_len, btaps, True, audio_feats=Optional(af), audio_token=audio_token)
        var e0 = split_err(d1.lm.download(btaps[0], n * th), load_npy_f64(dir + "/row0_embed.npy"), r0.ids, audio_token, th)
        var wa_ = e0[0]
        var wt = e0[1]
        var wa_at = String("the embedding")
        var wt_at = String("the embedding")
        for l in range(25):
            var got_l = d1.lm.download(btaps[l + 1], n * th) if l < 24 else d1.lm.download(h, n * th)
            var path = dir + ("/row0_layer" + String(l) + ".npy" if l < 24 else "/row0_final.npy")
            var se = split_err(got_l, load_npy_f64(path), r0.ids, audio_token, th)
            var at = "layer " + String(l) if l < 24 else String("the final norm")
            if getenv("AUD_DIAG") != "":
                print("     diag", at, " audio rows", se[0], " text rows", se[1])
            if se[0] > wa_:
                wa_ = se[0]
                wa_at = at
            if se[1] > wt:
                wt = se[1]
                wt_at = at
        check(fails, wt <= STAGE_BOUND, "backbone, TEXT rows: embedding, 24 layers, final norm; worst " + wt_at
              + "  rel err = " + String(wt))
        comptime if not LF_BF16:
            check(fails, wa_ <= AUDIO_ROW_BOUND, "backbone, AUDIO rows (features in place); worst " + wa_at
                  + "  rel err = " + String(wa_))
        else:
            print("  (not gated) backbone, AUDIO rows: worst", wa_at, " rel err =", wa_)
        var want_logz = load_npy_f64(dir + "/row0_slot_logz.npy")
        var es = rel_err(d1.lm.slot_logz(h, n, r0.slots), want_logz)
        check(fails, es <= SLOT_BOUND, "slots' log-softmax over the vocabulary  rel err = " + String(es))

        if names[c] == "heard_long":
            var tu = d1.audio.value().unused()
            check(fails, len(tu) == 0, "every audio tensor used (" + String(len(tu)) + " unused)")
            comptime if LF_BF16:
                print("  (skip) the encoder's defects - the fp32 build gates them")
            else:
                if t <= d1.audio.value().cfg.window:
                    raise Error("heard_long has " + String(t) + " tokens: not past the window, the window defect would be vacuous")
                var bits: List[Int] = [1, 2, 4, 8]
                var what: List[String] = ["the sliding window dropped", "the attention bidirectional",
                                          "the flatten mel-major", "no RoPE"]
                for b in range(len(bits)):
                    d1.audio.value().debug = bits[b]
                    var nn = List[DeviceBuffer[DT]]()
                    var f = d1.audio.value().encode(f32(want_mel), frames, nn)
                    d1.audio.value().debug = 0
                    var e = rel_err(d1.lm.download(f, t * th), want_proj)
                    check(fails, e >= FALSIFY_FACTOR * ENC_BOUND, "defect caught: " + what[b] + "  projector rel err = " + String(e))
            var bad = Lfm3LogMel()
            bad.window = hann_window(bad.n_fft, False)
            var eb = max_abs(bad.extract(wave), mel64)
            check(fails, eb >= FALSIFY_FACTOR * MEL_LOW_BOUND, "defect caught: a symmetric (non-periodic) window  mel max |d| = " + String(eb))
            var none2 = List[DeviceBuffer[DT]]()
            var h0 = d1.lm.forward(r0.ids, r0.groups, r0.state_len, none2)
            var e0b = rel_err(d1.lm.slot_logz(h0, n, r0.slots), want_logz)
            check(fails, e0b >= FALSIFY_FACTOR * SLOT_BOUND, "defect caught: the features left out  slots rel err = " + String(e0b))
        # timing (warm)
        var t3 = perf_counter_ns()
        _ = d1.system_one(ROOM_STATE, True, QUESTIONS, clips=clips)
        print("     system_one with a", String(Float64(len(wave)) / 16000.0), "s clip,", n, "tokens, warm:",
              Float64(perf_counter_ns() - t3) / 1e6, "ms")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
