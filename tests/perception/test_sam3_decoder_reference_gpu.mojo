"""SAM 3's DETR decoder and scoring (`noeira.perception.sam3.Sam3Decoder`)
against transformers', from the reference's own encoder output and text.

    pixi run -e apple mojo run -I . tests/perception/test_sam3_decoder_reference_gpu.mojo [ref dir]

Per word: each layer's 200 queries (after `output_layer_norm`), the boxes each
layer read, the presence logits, the final class logits and xyxy boxes — max
|d| / max |reference| (boxes and presence: max |d|). Every decoder and scoring
tensor used. Falsification: no box RPB; the sine box encoding in x, y order.
"""

from std.os import getenv
from std.sys import argv
from std.math import exp
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.npy import load_npy_f64
from noeira.nn.constants import DT
from noeira.perception.sam3 import Sam3Decoder, position_encoding, sam3_checkpoint
from noeira.perception.sam3.ops import SAM3_TF32


comptime BOUND: Float64 = 1e-2 if SAM3_TF32 else 2e-4
"""Queries and logits (relative), boxes and presence (absolute: boxes are in
[0, 1]). fp32, Metal, measured (5 words): queries <= 6.2e-5, logits <= 1.4e-4
absolute on values up to ~10 (relative ~1.4e-5), boxes <= 7.2e-5, presence
<= 3.3e-6 — larger than the encoders' because six refinements pass every box
through sigmoid / inverse sigmoid and the box arithmetic runs on the host in
fp32 (its sin / cos through float64). Twice the worst."""
comptime ABS_BOUND: Float64 = BOUND


def rel_err(got: List[Float32], want: List[Float64], off: Int = 0) raises -> Float64:
    var d = 0.0
    var m = 0.0
    for i in range(len(got)):
        d = max(d, abs(Float64(got[i]) - want[off + i]))
        m = max(m, abs(want[off + i]))
    return d / m if m > 0.0 else d


def max_abs(got: List[Float32], want: List[Float64], off: Int = 0) -> Float64:
    var d = 0.0
    for i in range(len(got)):
        d = max(d, abs(Float64(got[i]) - want[off + i]))
    return d


def download(ctx: DeviceContext, x: DeviceBuffer[DT], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.enqueue_copy(h, x.create_sub_buffer[DT](0, n))
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(h[i])
    return out^


def upload(ctx: DeviceContext, v: List[Float32]) raises -> DeviceBuffer[DT]:
    var h = ctx.enqueue_create_host_buffer[DT](len(v))
    for i in range(len(v)):
        h[i] = v[i]
    var d = ctx.enqueue_create_buffer[DT](len(v))
    ctx.enqueue_copy(d, h)
    return d^


def f32(v: List[Float64]) -> List[Float32]:
    var o = List[Float32](capacity=len(v))
    for x in v:
        o.append(Float32(x))
    return o^


def check(mut fails: Int, ok: Bool, what: String):
    print("  PASS " if ok else "  FAIL ", what)
    if not ok:
        fails += 1


def main() raises:
    var args = argv()
    var ref_dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/sam3/ref"
    var dec = Sam3Decoder(DeviceContext(), sam3_checkpoint())
    ref ctx = dec.wts.ctx
    var fails = 0
    print("SAM 3 DETR decoder + scoring vs transformers", "(TF32)" if SAM3_TF32 else "(fp32)")
    var pos = upload(ctx, position_encoding(72, 72))
    var words: List[String] = ["door", "table", "sofa", "bed", "a red chair"]
    for w in words:
        var dir = ref_dir + "/words/" + w.replace(" ", "_")
        var fused = upload(ctx, f32(load_npy_f64(dir + "/detr_enc.npy")))
        var text = upload(ctx, f32(load_npy_f64(dir + "/text_features.npy")))
        var mk = load_npy_f64(dir + "/mask.npy")
        var length = 0
        for x in mk:
            length += Int(x)
        var t0 = perf_counter_ns()
        var out = dec.forward(fused, pos, text, length, True)
        var ms = (perf_counter_ns() - t0) // 1_000_000
        var wq = load_npy_f64(dir + "/dec_hidden.npy")
        var wb = load_npy_f64(dir + "/dec_boxes.npy")
        var wp = load_npy_f64(dir + "/dec_presence.npy")
        var eq = 0.0
        var eb = 0.0
        var ep = 0.0
        for l in range(6):
            eq = max(eq, rel_err(download(ctx, out.layer_queries[l], 200 * 256), wq, l * 200 * 256))
            eb = max(eb, max_abs(out.layer_boxes[l], wb, l * 200 * 4))
            ep = max(ep, abs(Float64(out.layer_presence[l]) - wp[l]))
        var el = rel_err(out.logits, load_npy_f64(dir + "/pred_logits.npy"))
        var ebx = max_abs(out.boxes, load_npy_f64(dir + "/pred_boxes.npy"))
        var best = 0.0
        for q in range(200):
            best = max(best, 1.0 / (1.0 + exp(-Float64(out.logits[q]))))
        best *= 1.0 / (1.0 + exp(-Float64(out.presence)))
        check(fails, eq <= BOUND and eb <= ABS_BOUND and ep <= ABS_BOUND and el <= ABS_BOUND and ebx <= ABS_BOUND,
              repr(w) + ": queries " + String(eq) + ", boxes read " + String(eb) + ", presence " + String(ep)
              + ", logits " + String(el) + ", final boxes " + String(ebx) + "  (best score " + String(best) + ", " + String(ms) + " ms)")
    var ud = dec.wts.unused()
    var us = dec.sco.unused()
    check(fails, len(ud) == 0 and len(us) == 0, "every decoder and scoring tensor used (" + String(len(ud) + len(us)) + " unused)")
    var dir = ref_dir + "/words/door"
    var fused = upload(ctx, f32(load_npy_f64(dir + "/detr_enc.npy")))
    var text = upload(ctx, f32(load_npy_f64(dir + "/text_features.npy")))
    var want = load_npy_f64(dir + "/pred_boxes.npy")
    var bits: List[Int] = [1, 2]
    var what: List[String] = ["no box RPB", "sine box encoding in x, y order"]
    for b in range(2):
        dec.debug = bits[b]
        var out = dec.forward(fused, pos, text, 3)
        dec.debug = 0
        var e = max_abs(out.boxes, want)
        check(fails, e >= 10.0 * ABS_BOUND, "defect caught: " + what[b] + "  final boxes max |d| = " + String(e))
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
