"""Load the published LeWM checkpoint, and compare against its torch run.

The dump is `tools/lewm/dump_lewm_reference.py` + `convert_ref_to_ours.py`:
every tensor already in THIS framework's names (`ours.<walk name>`) and
layouts, so loading is by name with no layout logic here (that lives in ONE
place, the converter's `MAP`). `noeira.deep_agents.act.refload.RefDump` reads
the format.

- `load_ref[target](module, dump_dir, prefix, ctx)`: every Param AND State of
  `module` from `ours.<prefix><name>`. A sub-module of the loss graph loads
  with the graph prefix of its node (e.g. one ViT block: `emb.0.3.<i>.`). Any
  name the dump lacks raises — a weight left at its init reads as a small
  disagreement, not as the missing weight it is.
- `ref_input[target](dump_dir, name, ctx)`: a dumped activation as a Tensor.
- `std_err(dump_dir, name, got, ctx)`: max |got - ref| / std(ref), the unit
  every LeWM gate is held in (torch's own float32 error is 1e-6..1e-5 of it).
"""

from std.math import abs, sqrt
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.module import Module
from noeira.deep_agents.act.refload import RefDump, _fill


struct LoadRef(ParamVisitor):
    """The visitor `load_ref` runs; public for walkables that are not a
    `Module` (a `ComputeGraph`: walk params AND state, then `check()`)."""

    var dump: RefDump
    var prefix: String
    var loaded: Int
    var missing: List[String]

    def __init__(out self, var dump: RefDump, prefix: String):
        self.dump = dump^
        self.prefix = prefix
        self.loaded = 0
        self.missing = List[String]()

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var key = String("ours.") + self.prefix + name
        if not self.dump.has(key):
            self.missing.append(key^)
            return
        var vals = self.dump.get(key)
        if len(vals) != N:
            raise Error(
                "load_ref: '" + key + "' has " + String(len(vals))
                + " values, the param holds " + String(N)
            )
        _fill(param, vals, ctx)
        self.loaded += 1

    def check(self) raises:
        if len(self.missing) > 0:
            var msg = String("load_ref: ") + String(len(self.missing)) + " name(s) not in the dump, e.g. "
            for i in range(min(3, len(self.missing))):
                msg += self.missing[i] + " "
            raise Error(msg)


def load_ref[
    target: StaticString, M: Module
](
    mut module: M, dump_dir: String, prefix: String, ctx: Optional[DeviceContext]
) raises -> Int:
    """Fill every Param and State of `module`; returns how many."""
    var v = LoadRef(RefDump(dump_dir), prefix)
    module.for_each_param[target](v, ctx)
    module.for_each_state[target](v, ctx)
    v.check()
    return v.loaded


def tf32_gemm[target: StaticString]() -> Bool:
    """True when this target's float32 GEMMs run TF32: on NVIDIA, MAX's
    multistage matmul cannot disable TF32 for float32 outside SM100 (a 5090
    is SM120). Metal has no TF32; the CPU none either. A reference gate then
    holds the GPU leg to a TF32 band (10-bit mantissa) — and checks gradients
    by DIRECTION, since TF32 rounding inside cancelling sums puts single
    gradients ~0.3 std off while a wrong VJP flips or rotates them. The CPU
    leg on the same machine keeps the exact float32 check."""
    comptime if target == "gpu":
        return has_nvidia_gpu_accelerator()
    return False


def ref_input[target: StaticString](
    dump_dir: String, name: String, ctx: Optional[DeviceContext]
) raises -> Tensor:
    var vals = RefDump(dump_dir).get(name)
    var t = Tensor.alloc(len(vals))
    for i in range(len(vals)):
        t.data[i] = vals[i]
    comptime if target == "gpu":
        t.upload(ctx.value())
    return t^


def std_err[target: StaticString](
    dump_dir: String, name: String, mut got: Tensor, ctx: Optional[DeviceContext]
) raises -> Float64:
    """max |got - ref| / std(ref) over the reference's elements."""
    comptime if target == "gpu":
        ctx.value().synchronize()
        got.download(ctx.value())
    var want = RefDump(dump_dir).get(name)
    var mean = 0.0
    for i in range(len(want)):
        mean += Float64(want[i])
    mean /= Float64(len(want))
    var var_ = 0.0
    var worst = 0.0
    for i in range(len(want)):
        var d = Float64(want[i]) - mean
        var_ += d * d
        worst = max(worst, abs(Float64(got.data[i]) - Float64(want[i])))
    var sd = sqrt(var_ / Float64(len(want)))
    return worst / (sd if sd > 0.0 else 1.0)
