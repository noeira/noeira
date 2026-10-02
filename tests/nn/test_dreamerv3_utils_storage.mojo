"""Gate for the storage migration of the four DreamerV3 utility files:
`normalize.mojo`, `polyak.mojo`, `zero_init.mojo`, `param_sync.mojo`.

- polyak: two `Sequential[Linear,Linear]`; src Deterministic, dst perturbed.
    tau=1.0 → dst forward == src forward (full copy);
    tau=0.5 → dst moved halfway (output between perturbed and src).
- zero_init: a reward-head-shaped Sequential; scale `3.weight`/`3.bias` to 0
    → forward output is all-zero (output Linear zeroed).
- param_sync: two graphs of the SAME small type, different init; collect from
    src, apply to dst → dst forward == src forward (name-match copy).
- normalize: PercentileNormalize.make("perc", …); feed a known sample, check
    stats() offset/scale are finite + sane.

Run: rm -f noeira.mojoc && \
  pixi run -e apple mojo run -I . tests/nn/test_dreamerv3_utils_storage.mojo
"""

from std.testing import assert_true
from std.math import isfinite
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Deterministic, Kaiming
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.rms_norm import RMSNorm
from noeira.nn.primitives.elementwise import Elementwise
from noeira.nn.primitives.ops.gelu_tanh_op import GELUTanhOp
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import InputSlot, Node

from noeira.deep_agents.dreamerv3.normalize import PercentileNormalize
from noeira.deep_agents.dreamerv3.polyak import polyak_module
from noeira.deep_agents.dreamerv3.zero_init import (
    scale_output_module, scale_output_graph,
)
from noeira.deep_agents.dreamerv3.param_sync import (
    collect_graph_params, apply_graph_params,
)


comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5

comptime VAL = Sequential[Linear[D, H], Linear[H, O]]
# Reward-head shape: out Linear is child index 3 → "3.weight" / "3.bias".
comptime HEAD = Sequential[
    Linear[D, H], RMSNorm[H], Elementwise[H, GELUTanhOp], Linear[H, O]
]
comptime PG = ComputeGraph[InputSlot["x", D], Node["lin", Linear[D, O], "x"]]


# Collects every param's values into a flat List (downloads on GPU) — lets the
# polyak gate verify the tau-mix at the PARAMETER level (output-equality only
# holds for tau=1.0 since a 2-layer net is nonlinear in its weights).
struct _ParamCollect(Movable, ParamVisitor):
    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

    def take(deinit self) -> List[Scalar[DT]]:
        return self.vals^

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        for i in range(N):
            self.vals.append(param.data[i])


def _collect[
    target: StaticString
](mut net: VAL, ctx: Optional[DeviceContext]) raises -> List[Scalar[DT]]:
    var c = _ParamCollect()
    net.for_each_param[target](c, ctx)
    return c^.take()


def _make_x[target: StaticString](ctx: Optional[DeviceContext]) raises -> Tensor:
    var x = Tensor.alloc(B * D)
    for i in range(B * D):
        x.data[i] = Scalar[DT]((i % 7) - 3) * 0.25
    comptime if target == "gpu":
        x.upload(ctx.value())
    return x^


# ── polyak ────────────────────────────────────────────────────────────
def _polyak_ok[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var x = _make_x[target](ctx)
    var so = Tensor.alloc(B * O)
    var do1 = Tensor.alloc(B * O)

    # src = Deterministic, dst = Kaiming (perturbed → differs from src).
    var src = VAL.make[target, Deterministic](ctx)
    var dst = VAL.make[target, Kaiming](ctx)

    # Param-level snapshots (a 2-layer net is nonlinear in its weights, so
    # output-equality only holds at tau=1.0 — verify the mix on the params).
    var sp = _collect[target](src, ctx)        # src params (fixed)
    var dp0 = _collect[target](dst, ctx)        # dst params before any mix

    var differs = False
    for i in range(len(sp)):
        if abs(dp0[i] - sp[i]) > Scalar[DT](1e-5):
            differs = True
    if not differs:
        return False

    # tau = 0.5 → each dst param: 0.5·src + 0.5·dst_old.
    polyak_module[target, VAL](src, dst, Scalar[DT](0.5), ctx)
    var dp_half = _collect[target](dst, ctx)
    for i in range(len(sp)):
        var want = Scalar[DT](0.5) * sp[i] + Scalar[DT](0.5) * dp0[i]
        if abs(dp_half[i] - want) > Scalar[DT](1e-5):
            return False

    # tau = 1.0 → full copy: dst params == src params AND output == src output.
    polyak_module[target, VAL](src, dst, Scalar[DT](1.0), ctx)
    var dp1 = _collect[target](dst, ctx)
    for i in range(len(sp)):
        if abs(dp1[i] - sp[i]) > Scalar[DT](1e-5):
            return False

    src.forward[target, B](TensorRefs[1](x), so, ctx)
    dst.forward[target, B](TensorRefs[1](x), do1, ctx)
    comptime if target == "gpu":
        so.download(ctx.value()); do1.download(ctx.value())
    for i in range(B * O):
        if abs(do1.data[i] - so.data[i]) > Scalar[DT](1e-4):
            return False
    return True


# ── zero_init ─────────────────────────────────────────────────────────
def _zero_init_ok[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var x = _make_x[target](ctx)
    var head = HEAD.make[target, Deterministic](ctx)
    var out = Tensor.alloc(B * O)

    # Before: output should be (generally) nonzero.
    head.forward[target, B](TensorRefs[1](x), out, ctx)
    comptime if target == "gpu":
        out.download(ctx.value())
    var any_nonzero = False
    for i in range(B * O):
        if abs(out.data[i]) > Scalar[DT](1e-6):
            any_nonzero = True
    if not any_nonzero:
        return False

    scale_output_module[target, HEAD](
        head, String("3.weight"), String("3.bias"), Scalar[DT](0.0), ctx
    )
    head.forward[target, B](TensorRefs[1](x), out, ctx)
    comptime if target == "gpu":
        out.download(ctx.value())
    for i in range(B * O):
        if abs(out.data[i]) > Scalar[DT](1e-6):
            return False
    return True


# ── param_sync ────────────────────────────────────────────────────────
def _param_sync_ok[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var x = _make_x[target](ctx)
    var src = PG.make[target, Deterministic](ctx)
    var dst = PG.make[target, Kaiming](ctx)

    var xin: Tensor
    comptime if target == "gpu":
        xin = Tensor.alloc_gpu(ctx.value(), B * D)
        ctx.value().enqueue_copy(xin.dev.value(), x.dev.value())
        xin.n = B * D
    else:
        xin = Tensor.alloc(B * D)
        for i in range(B * D):
            xin.data[i] = x.data[i]

    var so = Tensor.alloc(B * O)
    var do0 = Tensor.alloc(B * O)
    var do1 = Tensor.alloc(B * O)

    src.set_input["x", B](xin, ctx)
    src.forward[B, target](so, ctx)
    dst.set_input["x", B](xin, ctx)
    dst.forward[B, target](do0, ctx)
    comptime if target == "gpu":
        so.download(ctx.value())
        do0.download(ctx.value())

    var differs = False
    for i in range(B * O):
        if abs(do0.data[i] - so.data[i]) > Scalar[DT](1e-5):
            differs = True
    if not differs:
        return False

    var snap = collect_graph_params[target](src, ctx)
    apply_graph_params[target](dst, snap, ctx)

    dst.set_input["x", B](xin, ctx)
    dst.forward[B, target](do1, ctx)
    comptime if target == "gpu":
        do1.download(ctx.value())
    for i in range(B * O):
        if abs(do1.data[i] - so.data[i]) > Scalar[DT](1e-4):
            return False
    return True


# ── normalize ─────────────────────────────────────────────────────────
def _normalize_ok() raises -> Bool:
    var nrm = PercentileNormalize.make(
        String("perc"),
        rate=Scalar[DT](0.5),
        perclo=Scalar[DT](5.0),
        perchi=Scalar[DT](95.0),
        limit=Scalar[DT](1.0),
        debias=False,
    )
    comptime N = 11
    var sample = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    for i in range(N):
        sample[i] = Scalar[DT](i)  # 0..10
    nrm.update(sample, N)
    _ = sample^

    var st = nrm.stats()
    var offset = st[0]
    var scale = st[1]
    if not (isfinite(offset) and isfinite(scale)):
        return False
    # rate=0.5, debias off → lo = 0.5·plo, hi = 0.5·phi over 0..10.
    # plo = percentile(5) = 0.5, phi = percentile(95) = 9.5.
    # offset = 0.5·0.5 = 0.25 ; span = 0.5·(9.5-0.5) = 4.5 ; scale = max(1, 4.5).
    if abs(offset - Scalar[DT](0.25)) > Scalar[DT](1e-4):
        return False
    if abs(scale - Scalar[DT](4.5)) > Scalar[DT](1e-4):
        return False
    return True


def main() raises:
    print("DreamerV3 utils storage gate (normalize/polyak/zero_init/param_sync)")
    var c = DeviceContext()

    var pc = _polyak_ok["cpu"](None)
    print("  polyak    CPU:", "OK" if pc else "FAIL")
    var pg = _polyak_ok["gpu"](Optional(c))
    print("  polyak    GPU:", "OK" if pg else "FAIL")

    var zc = _zero_init_ok["cpu"](None)
    print("  zero_init CPU:", "OK" if zc else "FAIL")
    var zg = _zero_init_ok["gpu"](Optional(c))
    print("  zero_init GPU:", "OK" if zg else "FAIL")

    var sc = _param_sync_ok["cpu"](None)
    print("  param_sync CPU:", "OK" if sc else "FAIL")
    var sg = _param_sync_ok["gpu"](Optional(c))
    print("  param_sync GPU:", "OK" if sg else "FAIL")

    var nc = _normalize_ok()
    print("  normalize CPU:", "OK" if nc else "FAIL")

    assert_true(
        pc and pg and zc and zg and sc and sg and nc,
        "DreamerV3 utils storage gate",
    )
    print("DREAMERV3 UTILS STORAGE OK")
