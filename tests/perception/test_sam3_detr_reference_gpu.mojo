"""SAM 3's DETR encoder (`noeira.perception.sam3.Sam3DetrEncoder`) against
transformers', from the reference's own 72x72 level and text features.

    pixi run -e apple mojo run -I . tests/perception/test_sam3_detr_reference_gpu.mojo [ref dir]

Per word: the fused image tokens [5184, 256], max |d| / max |reference|.
Every tensor used. Falsification: the text's padding keys visible; the
position added to the self-attention's values too.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.npy import load_npy_f64
from noeira.nn.constants import DT
from noeira.perception.sam3 import Sam3DetrEncoder, position_encoding, sam3_checkpoint
from noeira.perception.sam3.ops import SAM3_TF32


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


def upload(ctx: DeviceContext, v: List[Float32]) raises -> DeviceBuffer[DT]:
    var h = ctx.enqueue_create_host_buffer[DT](len(v))
    for i in range(len(v)):
        h[i] = v[i]
    var d = ctx.enqueue_create_buffer[DT](len(v))
    ctx.enqueue_copy(d, h)
    return d^


def check(mut fails: Int, ok: Bool, what: String):
    print("  PASS " if ok else "  FAIL ", what)
    if not ok:
        fails += 1


def main() raises:
    var args = argv()
    var ref_dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/sam3/ref"
    var enc = Sam3DetrEncoder(DeviceContext(), sam3_checkpoint())
    ref ctx = enc.wts.ctx
    var fails = 0
    print("SAM 3 DETR encoder vs transformers", "(TF32)" if SAM3_TF32 else "(fp32)")
    var chw = load_npy_f64(ref_dir + "/fpn2.npy")
    var hwc = List[Float32](length=5184 * 256, fill=0.0)
    for c in range(256):
        for n in range(5184):
            hwc[n * 256 + c] = Float32(chw[c * 5184 + n])
    var vis = upload(ctx, hwc)
    var pos = upload(ctx, position_encoding(72, 72))
    var words: List[String] = ["door", "table", "sofa", "bed", "a red chair"]
    var t_door = ctx.enqueue_create_buffer[DT](1)
    var len_door = 0
    for w in words:
        var dir = ref_dir + "/words/" + w.replace(" ", "_")
        var tf = load_npy_f64(dir + "/text_features.npy")
        var tl = List[Float32](capacity=len(tf))
        for x in tf:
            tl.append(Float32(x))
        var text = upload(ctx, tl)
        var mk = load_npy_f64(dir + "/mask.npy")
        var length = 0
        for x in mk:
            length += Int(x)
        var none = List[DeviceBuffer[DT]]()
        var t0 = perf_counter_ns()
        var out = enc.forward(vis, pos, 5184, text, length, none)
        ctx.synchronize()
        var ms = (perf_counter_ns() - t0) // 1_000_000
        var e = rel_err(download(ctx, out, 5184 * 256), load_npy_f64(dir + "/detr_enc.npy"))
        check(fails, e <= BOUND, repr(w) + " (" + String(length) + " tokens): fused image tokens rel err = " + String(e) + "  (" + String(ms) + " ms)")
        if w == "a red chair":
            t_door = text
            len_door = length
    var unused = enc.wts.unused()
    check(fails, len(unused) == 0, "every encoder tensor used (" + String(len(unused)) + " unused)")
    var want = load_npy_f64(ref_dir + "/words/a_red_chair/detr_enc.npy")
    var bits: List[Int] = [1, 2]
    var what: List[String] = ["text padding keys visible", "position on the values too"]
    for b in range(2):
        enc.debug = bits[b]
        var none = List[DeviceBuffer[DT]]()
        var out = enc.forward(vis, pos, 5184, t_door, len_door, none)
        enc.debug = 0
        var e = rel_err(download(ctx, out, 5184 * 256), want)
        check(fails, e >= 10.0 * BOUND, "defect caught: " + what[b] + "  rel err = " + String(e))
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
