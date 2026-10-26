# +--------------------------------------------------------------------------+ #
# | nn <-> safetensors, by walked name: does a model survive the trip?
# +--------------------------------------------------------------------------+ #
"""Gates `noeira/nn/core/safetensors_io.mojo` on CPU and GPU.

    pixi run mojo run -I . tests/nn/test_safetensors_roundtrip.mojo

`tests/io/test_safetensors.mojo` gates the FORMAT. This file gates the WALK:
that the names a model writes are the names it reads, that states travel with
parameters, and that a device model's weights actually reach the file.

## The three ways a loader passes without working

* **It loads nothing.** Every check below is preceded by proving the
  destination model DIFFERS from the source first — a load that matched no
  names would otherwise satisfy "the values agree" only when both models were
  identically initialised, which is exactly how these gates are usually built.
* **It loads parameters and skips state.** BatchNorm running statistics are a
  separate walk. A model whose convolutions are restored and whose statistics
  are not is a different function, and nothing about the parameter count says
  so. The BatchNorm leg perturbs the statistics away from their init (mean 0,
  var 1) first, because at init they are exactly the values a dropped state
  walk would leave behind.
* **It saves the host mirror of a device model.** On GPU the weights live on
  the device and `param.data` holds whatever was last downloaded — for a model
  built with `make["gpu"]` and trained, the initialisation. A file full of
  random weights saves without error.
"""

from std.memory import bitcast
from std.testing import assert_true
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from noeira.io.safetensors import SafeTensors, SafeTensorsWriter
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Deterministic, Kaiming
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.safetensors_io import load_safetensors, save_safetensors
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.torch_names import (
    LoadTorchNamed,
    SaveTorchNamed,
    TorchNameMap,
)
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.models.conv import Conv2DBatchNormReLU
from noeira.nn.primitives.linear import Linear


comptime D = 4
comptime H = 6
comptime O = 3
comptime NET = Sequential[Linear[D, H], Linear[H, O]]
comptime BAD = Sequential[Linear[D, 8], Linear[8, O]]
comptime BN = Sequential[
    Conv2DBatchNormReLU[3, 4, 3, 1, 1, 8, 8],
    Conv2DBatchNormReLU[4, 4, 3, 1, 1, 8, 8],
]
comptime NET_VALS = (D * H + H) + (H * O + O)
# A BARE Linear, so the walked names are "weight" / "bias" and a two-entry
# map covers it. `Sequential` would prefix them with a child index and the
# map would be about the combinator rather than the layout conversion.
comptime LIN = Linear[D, H]
comptime LIN_VALS = D * H + H


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


struct Perturb(ParamVisitor):
    """Writes a distinct, non-init value into every tensor it visits.

    Used on BOTH walks. Running statistics start at mean 0 / var 1, which is
    also what a load that skipped the state walk would leave behind — so
    without this the BatchNorm leg would pass on a loader that never read a
    statistic."""

    var k: Int

    def __init__(out self):
        self.k = 0

    def __init__(out self, *, deinit move: Self):
        self.k = move.k

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        param.ensure(N)
        for i in range(N):
            self.k += 1
            param.data[i] = Scalar[DT](Float64(self.k % 97) * 0.03125 - 1.5)
        if ctx:
            param.upload_resident(ctx.value())


struct Capture(ParamVisitor):
    """Every visited value, in walk order, with its name."""

    var vals: List[Scalar[DT]]
    var names: List[String]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()
        self.names = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^
        self.names = move.names^

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        self.names.append(String(name))
        for i in range(N):
            self.vals.append(param.data[i])


struct ScribbleHost(ParamVisitor):
    """Overwrites the HOST mirror and does NOT upload, so host and device
    disagree.

    ⚠ This is what makes the export leg below non-vacuous. `Perturb`
    writes the host tensor AND uploads it, leaving the two in agreement --
    against which a `SaveTorchNamed` that never downloaded would still write
    exactly the right numbers and pass. After this the device holds the
    weights and the host holds -99, so only a save that downloads can
    produce a file that round-trips.

    Counts what it scribbled: a visitor that matched nothing would restore
    the vacuity it exists to remove, silently.
    """

    var n: Int

    def __init__(out self):
        self.n = 0

    def __init__(out self, *, deinit move: Self):
        self.n = move.n

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        param.ensure(N)
        for i in range(N):
            param.data[i] = Scalar[DT](-99.0)
        self.n += N


def lin_torch_map() raises -> TorchNameMap:
    """`Linear[D, H]` under PyTorch's names, shapes and layout.

    Ours is `[in, out]`; theirs is `[out, in]`. D != H on purpose, so a
    transpose that only reshaped would be caught by the shape check."""
    var m = TorchNameMap()
    m.add_linear(String("weight"), String("fc.weight"), H, D)
    var bs: List[Int] = [H]
    m.add(String("bias"), String("fc.bias"), bs)
    return m^


def differ(ref a: Capture, ref b: Capture) -> Int:
    """How many values differ, bitwise. 0 means identical."""
    if len(a.vals) != len(b.vals):
        return -1
    var n = 0
    for i in range(len(a.vals)):
        if bitcast[DType.uint32](Float32(a.vals[i])) != bitcast[
            DType.uint32
        ](Float32(b.vals[i])):
            n += 1
    return n


def _gpu_legs(mut fails: Int) raises:
    # ══════════════════════════════════════════════════════════════════════
    print("")
    print("GPU — the weights that reach the file are the DEVICE's")
    var ctx = DeviceContext()
    var g = NET.make["gpu", Deterministic](Optional(ctx))
    var pg = Perturb()
    g.for_each_param["gpu", Perturb](pg, Optional(ctx))
    _ = save_safetensors["gpu"](g, String("/tmp/st_gpu.safetensors"), Optional(ctx))
    var h = NET.make["gpu", Kaiming](Optional(ctx))
    var cg = Capture()
    g.for_each_param["gpu", Capture](cg, Optional(ctx))
    var ch0 = Capture()
    h.for_each_param["gpu", Capture](ch0, Optional(ctx))
    check(fails, "the two device models DIFFER before the load",
          differ(cg, ch0) > NET_VALS // 2,
          String(differ(cg, ch0)) + " differ")
    _ = load_safetensors["gpu"](h, String("/tmp/st_gpu.safetensors"), Optional(ctx))
    var ch1 = Capture()
    h.for_each_param["gpu", Capture](ch1, Optional(ctx))
    # ⚠ `Capture` DOWNLOADS. So this compares what the device holds, not the
    # host mirror `fill_param` just wrote — which is the only way to see a
    # missing upload.
    check(fails, "the DEVICE's weights are bit-identical after the load",
          differ(cg, ch1) == 0,
          String(len(cg.vals)) + " compared, " + String(differ(cg, ch1))
          + " differ")

    # ══════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════
    # `SaveTorchNamed` / `LoadTorchNamed` on a DEVICE model. The ACT backbone
    # path (`load_backbone_auto` -> `load_backbone_safetensors`) is the only
    # caller, and it instantiates the LOADER for "gpu" but never the SAVER, so
    # without this leg `SaveTorchNamed`'s `target == "gpu"` branch is compiled
    # nowhere and a defect in it would keep until someone exported from a GPU
    # run.
    print("")
    print("GPU, PyTorch names — the export DOWNLOADS and the import UPLOADS")
    var lg = LIN.make["gpu", Deterministic](Optional(ctx))
    var plg = Perturb()
    lg.for_each_param["gpu", Perturb](plg, Optional(ctx))
    var clg = Capture()
    lg.for_each_param["gpu", Capture](clg, Optional(ctx))

    var sh = ScribbleHost()
    lg.for_each_param["gpu", ScribbleHost](sh, Optional(ctx))
    check(fails, "the host mirror was poisoned before the export",
          sh.n == LIN_VALS, String(sh.n) + " of " + String(LIN_VALS))

    var sv = SaveTorchNamed[""](lin_torch_map())
    lg.for_each_param["gpu", SaveTorchNamed[""]](sv, Optional(ctx))
    sv.report(String("gpu linear export"))
    sv.writer.save(String("/tmp/st_gpu_torch.safetensors"))

    var lf = SafeTensors(String("/tmp/st_gpu_torch.safetensors"))
    check(fails, "the exported weight carries torch's shape [out, in]",
          lf.shape_str(String("fc.weight"))
          == "[" + String(H) + ", " + String(D) + "]",
          lf.shape_str(String("fc.weight")))

    var lh = LIN.make["gpu", Kaiming](Optional(ctx))
    var clh0 = Capture()
    lh.for_each_param["gpu", Capture](clh0, Optional(ctx))
    check(fails, "the two device Linears DIFFER before the load",
          differ(clg, clh0) > LIN_VALS // 2,
          String(differ(clg, clh0)) + " differ")

    var lv = LoadTorchNamed[""](
        SafeTensors(String("/tmp/st_gpu_torch.safetensors")), lin_torch_map()
    )
    lh.for_each_param["gpu", LoadTorchNamed[""]](lv, Optional(ctx))
    lv.report(String("gpu linear import"))
    var clh1 = Capture()
    lh.for_each_param["gpu", Capture](clh1, Optional(ctx))
    # ⚠ `clg` was captured BEFORE the host mirror was poisoned, and `Capture`
    # downloads — so this says the file holds what the DEVICE held, survived
    # the transpose out and back, and landed on the DEVICE again.
    check(fails, "the DEVICE round-trips through torch names and layout",
          differ(clg, clh1) == 0,
          String(len(clg.vals)) + " compared, " + String(differ(clg, clh1))
          + " differ")



def main() raises:
    var fails = 0
    print("nn <-> safetensors round trip")
    print("")

    # ══════════════════════════════════════════════════════════════════════
    print("CPU, parameters only")
    var a = NET.make["cpu", Deterministic](None)
    var p = Perturb()
    a.for_each_param["cpu", Perturb](p, None)
    var n_saved = save_safetensors["cpu"](a, String("/tmp/st_net.safetensors"))
    check(fails, "4 tensors written", n_saved == 4, String(n_saved))

    var f = SafeTensors(String("/tmp/st_net.safetensors"))
    check(fails, "names are the walked names",
          f.has(String("0.weight")) and f.has(String("1.bias")),
          f.names[0] + " ... " + f.names[len(f.names) - 1])
    check(fails, "shape is rank-1 [N]",
          f.shape_str(String("0.weight")) == "[" + String(D * H) + "]",
          f.shape_str(String("0.weight")))

    var b = NET.make["cpu", Kaiming](None)
    var ca = Capture()
    a.for_each_param["cpu", Capture](ca, None)
    var cb0 = Capture()
    b.for_each_param["cpu", Capture](cb0, None)
    # ⚠ The anti-vacuity check. Without it, "the values agree after loading"
    # is also true of a loader that loaded nothing into an identical model.
    check(fails, "the two models DIFFER before the load",
          differ(ca, cb0) > NET_VALS // 2,
          String(differ(ca, cb0)) + " of " + String(len(ca.vals)) + " differ")

    var n_loaded = load_safetensors["cpu"](b, String("/tmp/st_net.safetensors"))
    var cb1 = Capture()
    b.for_each_param["cpu", Capture](cb1, None)
    check(fails, "every value is bit-identical after the load",
          differ(ca, cb1) == 0 and len(ca.vals) == NET_VALS,
          String(len(ca.vals)) + " compared, " + String(differ(ca, cb1))
          + " differ")
    check(fails, "load count matches", n_loaded == 4, String(n_loaded))

    # ══════════════════════════════════════════════════════════════════════
    print("")
    print("CPU, BatchNorm — parameters AND running statistics")
    var c = BN.make["cpu", Deterministic](None)
    var pc = Perturb()
    c.for_each_param["cpu", Perturb](pc, None)
    c.for_each_state["cpu", Perturb](pc, None)
    var n_bn = save_safetensors["cpu"](c, String("/tmp/st_bn.safetensors"))

    var fbn = SafeTensors(String("/tmp/st_bn.safetensors"))
    var n_stats = 0
    for i in range(fbn.size()):
        if fbn.names[i].endswith("running_mean") or fbn.names[i].endswith(
            "running_var"
        ):
            n_stats += 1
    check(fails, "running statistics are in the file", n_stats == 4,
          String(n_stats) + " of " + String(n_bn) + " tensors")

    var d = BN.make["cpu", Kaiming](None)
    var sa = Capture()
    c.for_each_state["cpu", Capture](sa, None)
    var sd0 = Capture()
    d.for_each_state["cpu", Capture](sd0, None)
    check(fails, "the statistics DIFFER before the load",
          differ(sa, sd0) > 0,
          String(differ(sa, sd0)) + " of " + String(len(sa.vals)) + " differ")

    _ = load_safetensors["cpu"](d, String("/tmp/st_bn.safetensors"))
    var sd1 = Capture()
    d.for_each_state["cpu", Capture](sd1, None)
    check(fails, "statistics are bit-identical after the load",
          differ(sa, sd1) == 0 and len(sa.vals) > 0,
          String(len(sa.vals)) + " compared, " + String(differ(sa, sd1))
          + " differ")

    comptime if has_accelerator():
        _gpu_legs(fails)
    else:
        print("")
        print(
            "GPU legs: SKIPPED (no accelerator;"
            " run with -e apple / -e nvidia)"
        )

    print("")
    print("refusals")
    var drift = False
    try:
        var bad = BAD.make["cpu", Deterministic](None)
        _ = load_safetensors["cpu"](bad, String("/tmp/st_net.safetensors"))
    except:
        drift = True
    check(fails, "a differently-sized model raises", drift)

    var alien = False
    try:
        var e2 = BN.make["cpu", Deterministic](None)
        _ = load_safetensors["cpu"](e2, String("/tmp/st_net.safetensors"))
    except:
        alien = True
    check(fails, "a file with no names in common raises", alien)

    # A file that is one tensor short. Strict must refuse it; non-strict must
    # load the rest AND say how many.
    var w2 = SafeTensorsWriter()
    for i in range(f.size() - 1):
        var nm = String(f.names[i])
        var vals = f.read_f32(nm)
        var sh: List[Int] = [len(vals)]
        w2.add_f32(nm^, sh, vals, len(vals))
    w2.save(String("/tmp/st_partial.safetensors"))

    var partial_refused = False
    try:
        var b3 = NET.make["cpu", Kaiming](None)
        _ = load_safetensors["cpu"](b3, String("/tmp/st_partial.safetensors"))
    except:
        partial_refused = True
    check(fails, "a file missing one tensor raises under strict",
          partial_refused)

    var b4 = NET.make["cpu", Kaiming](None)
    var n_partial = load_safetensors["cpu"](
        b4, String("/tmp/st_partial.safetensors"), strict=False
    )
    check(fails, "strict=False loads the rest and counts them",
          n_partial == 3, String(n_partial) + " of 4")

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("safetensors round-trip gate failed")
