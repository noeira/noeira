"""SAM 3's vision backbone (`noeira.perception.sam3.Sam3ViT`) against
transformers' `Sam3ViTModel`, stage by stage.

    reference: pixi run -e detect python tools/perception/dump_sam3_reference.py --image <png>
    pixi run -e apple mojo run -I . tests/perception/test_sam3_vit_reference_gpu.mojo [ref dir]
      default ref dir ~/.cache/noeira/sam3/ref; the checkpoint: `sam3_checkpoint()`

The reference's own `pixel_values` in: the patch + tiled position embedding,
the pre-norm, and the dumped layers (windowed 0, 1, 6, 8, 30; global 7, 15,
23, 31 — 31 is the backbone's output), max |d| / max |reference| each.
Falsification: the RoPE's axes swapped, no global attention, the position
embedding stretched instead of tiled — each must move the output ten times
past the bound.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.json import load_json
from noeira.io.npy import load_npy_f64
from noeira.nn.constants import DT
from noeira.perception.sam3 import Sam3ViT, sam3_checkpoint
from noeira.perception.sam3.ops import SAM3_TF32


comptime BOUND: Float64 = 1e-2 if SAM3_TF32 else 1e-4
comptime FALSIFY: Float64 = 10.0 * BOUND


def rel_err(got: List[Float32], want: List[Float64]) raises -> Float64:
    if len(got) != len(want):
        raise Error("size " + String(len(got)) + " against the reference's " + String(len(want)))
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
    var fails = 0
    print("SAM 3 vision backbone vs transformers", "(TF32)" if SAM3_TF32 else "(fp32)")
    var t0 = perf_counter_ns()
    var vit = Sam3ViT(DeviceContext(), sam3_checkpoint())
    print("  weights loaded in", (perf_counter_ns() - t0) // 1_000_000, "ms")
    var pv = load_npy_f64(ref_dir + "/pixel_values.npy")
    var hpv = vit.wts.ctx.enqueue_create_host_buffer[DT](len(pv))
    for i in range(len(pv)):
        hpv[i] = Float32(pv[i])
    var dpv = vit.wts.ctx.enqueue_create_buffer[DT](len(pv))
    vit.wts.ctx.enqueue_copy(dpv, hpv)
    var taps = List[DeviceBuffer[DT]]()
    var t1 = perf_counter_ns()
    var out = vit.forward(dpv, taps, True)
    vit.wts.ctx.synchronize()
    print("  forward (with taps):", (perf_counter_ns() - t1) // 1_000_000, "ms")
    var n = 5184 * 1024
    var e = rel_err(download(vit.wts.ctx, taps[0], n), load_npy_f64(ref_dir + "/vit_embed.npy"))
    check(fails, e <= BOUND, "patch + tiled position embedding  rel err = " + String(e))
    e = rel_err(download(vit.wts.ctx, taps[1], n), load_npy_f64(ref_dir + "/vit_ln_pre.npy"))
    check(fails, e <= BOUND, "pre-norm  rel err = " + String(e))
    var meta = load_json(ref_dir + "/meta.json")
    var ls = meta.field(meta.root(), "layers")
    var worst = 0.0
    var worst_l = -1
    for i in range(meta.size(ls)):
        var l = meta.integer(meta.at(ls, i))
        var el = rel_err(download(vit.wts.ctx, taps[2 + l], n), load_npy_f64(ref_dir + "/vit_layer" + String(l) + ".npy"))
        print("     layer", l, "global" if (l == 7 or l == 15 or l == 23 or l == 31) else "windowed", " rel err", el)
        if el > worst:
            worst = el
            worst_l = l
    check(fails, worst <= BOUND, "layers, worst " + String(worst_l) + "  rel err = " + String(worst))
    var unused = vit.wts.unused()
    check(fails, len(unused) == 0, "every backbone tensor used (" + String(len(unused)) + " unused)")
    # falsification
    var want31 = load_npy_f64(ref_dir + "/vit_layer31.npy")
    var bits: List[Int] = [1, 2, 4]
    var what: List[String] = ["RoPE axes swapped", "no global attention", "position embedding stretched, not tiled"]
    for b in range(3):
        vit.set_debug(bits[b])
        var none = List[DeviceBuffer[DT]]()
        var o = vit.forward(dpv, none)
        var ef = rel_err(download(vit.wts.ctx, o, n), want31)
        check(fails, ef >= FALSIFY, "defect caught: " + what[b] + "  output rel err = " + String(ef))
    vit.set_debug(0)
    var t2 = perf_counter_ns()
    var none2 = List[DeviceBuffer[DT]]()
    _ = vit.forward(dpv, none2)
    vit.wts.ctx.synchronize()
    print("  forward, warm:", (perf_counter_ns() - t2) // 1_000_000, "ms")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
