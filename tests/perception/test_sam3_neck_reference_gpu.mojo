"""SAM 3's vision neck (`noeira.perception.sam3.Sam3Neck`) against
transformers' `Sam3VisionNeck`, from the reference's own backbone output.

    pixi run -e apple mojo run -I . tests/perception/test_sam3_neck_reference_gpu.mojo [ref dir]

Each of the four FPN levels (288, 144, 72, 36) and its sine position
encoding, max |d| / max |reference|. Falsification: the 3x3 taps flipped, the
transposed conv's dy / dx swapped — each must move level 0 ten times past the
bound.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.npy import load_npy_f64
from noeira.nn.constants import DT
from noeira.perception.sam3 import Sam3Neck, FpnLevel, position_encoding, sam3_checkpoint
from noeira.perception.sam3.ops import SAM3_TF32


comptime BOUND: Float64 = 1e-2 if SAM3_TF32 else 1e-4
comptime POS_BOUND: Float64 = 1e-5


def rel_err_chw(got_hwc: List[Float32], want_chw: List[Float64], h: Int, w: Int, c: Int) raises -> Float64:
    """`got` token-major [h w, c] against `want` [c, h, w]."""
    if len(got_hwc) != len(want_chw) or len(want_chw) != h * w * c:
        raise Error("size " + String(len(got_hwc)) + " against " + String(len(want_chw)))
    var d = 0.0
    var m = 0.0
    for ch in range(c):
        for n in range(h * w):
            var wv = want_chw[ch * h * w + n]
            d = max(d, abs(Float64(got_hwc[n * c + ch]) - wv))
            m = max(m, abs(wv))
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
    var fails = 0
    print("SAM 3 vision neck vs transformers", "(TF32)" if SAM3_TF32 else "(fp32)")
    var neck = Sam3Neck(DeviceContext(), sam3_checkpoint())
    var x = load_npy_f64(ref_dir + "/vit_layer31.npy")
    var hx = neck.wts.ctx.enqueue_create_host_buffer[DT](len(x))
    for i in range(len(x)):
        hx[i] = Float32(x[i])
    var dx = neck.wts.ctx.enqueue_create_buffer[DT](len(x))
    neck.wts.ctx.enqueue_copy(dx, hx)
    var t0 = perf_counter_ns()
    var levels = neck.forward(dx)
    neck.wts.ctx.synchronize()
    print("  forward:", (perf_counter_ns() - t0) // 1_000_000, "ms")
    var worst = 0.0
    var worst_p = 0.0
    for i in range(4):
        ref L = levels[i]
        var e = rel_err_chw(download(neck.wts.ctx, L.x, L.h * L.w * 256), load_npy_f64(ref_dir + "/fpn" + String(i) + ".npy"), L.h, L.w, 256)
        var ep = rel_err_chw(position_encoding(L.h, L.w), load_npy_f64(ref_dir + "/fpn_pos" + String(i) + ".npy"), L.h, L.w, 256)
        print("     level", i, L.h, "x", L.w, " rel err", e, " position encoding", ep)
        worst = max(worst, e)
        worst_p = max(worst_p, ep)
    check(fails, worst <= BOUND, "4 FPN levels, worst rel err = " + String(worst))
    check(fails, worst_p <= POS_BOUND, "their sine position encodings, worst rel err = " + String(worst_p))
    var unused = neck.wts.unused()
    check(fails, len(unused) == 0, "every neck tensor used (" + String(len(unused)) + " unused)")
    var want0 = load_npy_f64(ref_dir + "/fpn0.npy")
    var bits: List[Int] = [1, 2]
    var what: List[String] = ["3x3 taps flipped", "transposed conv dy/dx swapped"]
    for b in range(2):
        neck.debug = bits[b]
        var lv = neck.forward(dx)
        neck.debug = 0
        var e = rel_err_chw(download(neck.wts.ctx, lv[0].x, 288 * 288 * 256), want0, 288, 288, 256)
        check(fails, e >= 10.0 * BOUND, "defect caught: " + what[b] + "  level 0 rel err = " + String(e))
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
