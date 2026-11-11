"""SAM 3's text side (`noeira.text.clip_bpe` + `noeira.perception.sam3.Sam3Text`)
against transformers' `Sam3Model.get_text_features`.

    pixi run -e detect python tools/perception/dump_sam3_reference.py --image x --text-only --words "door,table,sofa,bed,a red chair"
    pixi run -e apple mojo run -I . tests/perception/test_sam3_text_reference_gpu.mojo [ref dir]

Per word: the 32 ids and the mask exact (our tokenizer), CLIP's layers 0, 1,
11, 23 and the final norm, and SAM 3's 256-wide features — max |d| / max
|reference|. Every text tensor used but CLIP's own `text_projection` (SAM 3
does not apply it). Falsification: quick-GELU (CLIP's default) instead of
GELU; the padding keys visible (plain causal: only the padded rows move).
"""

from std.os import getenv, listdir
from std.sys import argv
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.npy import load_npy_f64
from noeira.nn.constants import DT
from noeira.perception.sam3 import Sam3Text, sam3_checkpoint
from noeira.perception.sam3.ops import SAM3_TF32
from noeira.text.clip_bpe import ClipBPE


comptime BOUND: Float64 = 1e-2 if SAM3_TF32 else 1e-4


def rel_err(got: List[Float32], want: List[Float64]) raises -> Float64:
    if len(got) != len(want):
        raise Error("size " + String(len(got)) + " against " + String(len(want)))
    var d = 0.0
    var m = 0.0
    for i in range(len(want)):
        d = max(d, abs(Float64(got[i]) - want[i]))
        m = max(m, abs(want[i]))
    return d / m if m > 0.0 else d


def download(ctx: DeviceContext, x: DeviceBuffer[DT], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.enqueue_copy(h, x.create_sub_buffer[DT](0, n))
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(h[i])
    return out^


def check(mut fails: Int, ok: Bool, what: String):
    print("  PASS " if ok else "  FAIL ", what)
    if not ok:
        fails += 1


def main() raises:
    var args = argv()
    var ref_dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/sam3/ref"
    var ckpt = sam3_checkpoint()
    var tok = ClipBPE.load(String(ckpt[byte=0:ckpt.byte_length() - 17]) + "tokenizer.json")
    var text = Sam3Text(DeviceContext(), ckpt)
    var fails = 0
    print("SAM 3 text encoder vs transformers", "(TF32)" if SAM3_TF32 else "(fp32)")
    var words: List[String] = ["door", "table", "sofa", "bed", "a red chair"]
    var layers: List[Int] = [0, 1, 11, 23]
    for w in words:
        var dir = ref_dir + "/words/" + w.replace(" ", "_")
        var toks = tok.encode(w, 32)
        var wi = load_npy_f64(dir + "/ids.npy")
        var wm = load_npy_f64(dir + "/mask.npy")
        var same = True
        for k in range(32):
            same = same and toks.ids[k] == Int(wi[k]) and toks.mask[k] == Int(wm[k])
        var taps = List[DeviceBuffer[DT]]()
        var f = text.forward(toks, taps, True)
        var worst = 0.0
        for l in layers:
            worst = max(worst, rel_err(download(text.wts.ctx, taps[l], 32 * 1024), load_npy_f64(dir + "/text_layer" + String(l) + ".npy")))
        var ef = rel_err(download(text.wts.ctx, taps[24], 32 * 1024), load_npy_f64(dir + "/text_final.npy"))
        var ep = rel_err(download(text.wts.ctx, f, 32 * 256), load_npy_f64(dir + "/text_features.npy"))
        check(fails, same and worst <= BOUND and ef <= BOUND and ep <= BOUND,
              repr(w) + ": ids exact " + String(same) + ", layers " + String(worst) + ", final " + String(ef) + ", features " + String(ep))
    var unused = text.wts.unused()
    check(fails, len(unused) == 1 and unused[0] == "encoder.text_projection.weight",
          "every text tensor used but CLIP's own projection (" + String(len(unused)) + " unused)")
    var dir0 = ref_dir + "/words/a_red_chair"
    var want = load_npy_f64(dir0 + "/text_features.npy")
    var bits: List[Int] = [2, 1]
    var what: List[String] = ["quick-GELU instead of GELU", "padding keys visible (padded rows only)"]
    for b in range(2):
        text.debug = bits[b]
        var none = List[DeviceBuffer[DT]]()
        var f = text.forward(tok.encode("a red chair", 32), none)
        text.debug = 0
        var e = rel_err(download(text.wts.ctx, f, 32 * 256), want)
        check(fails, e >= 10.0 * BOUND, "defect caught: " + what[b] + "  features rel err = " + String(e))
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
