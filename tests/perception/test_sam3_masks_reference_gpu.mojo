"""SAM 3's mask decoder (`noeira.perception.sam3.Sam3MaskDecoder`) against
transformers', from the reference's own queries, FPN levels, encoder output
and text.

    pixi run -e apple mojo run -I . tests/perception/test_sam3_masks_reference_gpu.mojo [ref dir]

For door and sofa: the 200 mask logit maps at 288 x 288 and the semantic map,
max |d| / max |reference|. Every tensor used but the pixel decoder's third
stage (in the checkpoint, never in the forward). Falsification: no prompt
cross-attention; the GroupNorm replaced by a per-pixel norm.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.npy import load_npy_f64
from noeira.nn.constants import DT
from noeira.perception.sam3 import Sam3MaskDecoder, sam3_checkpoint
from noeira.perception.sam3.ops import SAM3_TF32


comptime BOUND: Float64 = 1e-2 if SAM3_TF32 else 1e-4


def rel_err(got: List[Float32], want: List[Float64], off: Int = 0) raises -> Float64:
    var d = 0.0
    var m = 0.0
    for i in range(len(got)):
        d = max(d, abs(Float64(got[i]) - want[off + i]))
        m = max(m, abs(want[off + i]))
    return d / m if m > 0.0 else d


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


def f32(v: List[Float64], off: Int = 0, n: Int = -1) -> List[Float32]:
    var count = len(v) - off if n < 0 else n
    var o = List[Float32](capacity=count)
    for i in range(count):
        o.append(Float32(v[off + i]))
    return o^


def hwc(chw: List[Float64], c: Int, n: Int) -> List[Float32]:
    var o = List[Float32](length=c * n, fill=0.0)
    for ch in range(c):
        for p in range(n):
            o[p * c + ch] = Float32(chw[ch * n + p])
    return o^


def check(mut fails: Int, ok: Bool, what: String):
    print("  PASS " if ok else "  FAIL ", what)
    if not ok:
        fails += 1


def main() raises:
    var args = argv()
    var ref_dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/sam3/ref"
    var md = Sam3MaskDecoder(DeviceContext(), sam3_checkpoint())
    ref ctx = md.wts.ctx
    var fails = 0
    print("SAM 3 mask decoder vs transformers", "(TF32)" if SAM3_TF32 else "(fp32)")
    var f288 = upload(ctx, hwc(load_npy_f64(ref_dir + "/fpn0.npy"), 256, 288 * 288))
    var f144 = upload(ctx, hwc(load_npy_f64(ref_dir + "/fpn1.npy"), 256, 144 * 144))
    var words: List[String] = ["door", "sofa"]
    var keep_q = ctx.enqueue_create_buffer[DT](1)
    var keep_f = ctx.enqueue_create_buffer[DT](1)
    var keep_t = ctx.enqueue_create_buffer[DT](1)
    for w in words:
        var dir = ref_dir + "/words/" + w
        var q = upload(ctx, f32(load_npy_f64(dir + "/dec_hidden.npy"), 5 * 200 * 256, 200 * 256))
        var fused = upload(ctx, f32(load_npy_f64(dir + "/detr_enc.npy")))
        var text = upload(ctx, f32(load_npy_f64(dir + "/text_features.npy")))
        var t0 = perf_counter_ns()
        var m = md.forward(q, f288, f144, fused, text, 3)
        ctx.synchronize()
        var ms = (perf_counter_ns() - t0) // 1_000_000
        var em = rel_err(download(ctx, m.masks, 200 * 288 * 288), load_npy_f64(dir + "/pred_masks.npy"))
        var es = rel_err(download(ctx, m.semantic, 288 * 288), load_npy_f64(dir + "/semantic_seg.npy"))
        check(fails, em <= BOUND and es <= BOUND, repr(w) + ": 200 masks rel err " + String(em) + ", semantic " + String(es) + "  (" + String(ms) + " ms)")
        keep_q = q
        keep_f = fused
        keep_t = text
    var unused = md.wts.unused()
    var only_third = len(unused) == 4
    for u in unused:
        only_third = only_third and (u.startswith("pixel_decoder.conv_layers.2") or u.startswith("pixel_decoder.norms.2"))
    check(fails, only_third, "every tensor used but the pixel decoder's unused third stage (" + String(len(unused)) + " unused)")
    var want = load_npy_f64(ref_dir + "/words/sofa/pred_masks.npy")
    var bits: List[Int] = [1, 2]
    var what: List[String] = ["no prompt cross-attention", "GroupNorm as a per-pixel norm"]
    for b in range(2):
        md.debug = bits[b]
        var m = md.forward(keep_q, f288, f144, keep_f, keep_t, 3)
        md.debug = 0
        var e = rel_err(download(ctx, m.masks, 200 * 288 * 288), want)
        check(fails, e >= 10.0 * BOUND, "defect caught: " + what[b] + "  masks rel err = " + String(e))
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
