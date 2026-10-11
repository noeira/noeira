# +--------------------------------------------------------------------------+ #
# | SmolVLA's prefill backward with the VLM's linears frozen
# +--------------------------------------------------------------------------+ #
"""What freezing the 112 VLM linears buys `train_state_proj = True`.

    pixi run -e apple  mojo run -I . benchmarks/peft/bench_prefill_no_dw.mojo
    pixi run -e nvidia mojo run -I . benchmarks/peft/bench_prefill_no_dw.mojo
    mojo build -I . -D BENCH_CPU=1 benchmarks/peft/bench_prefill_no_dw.mojo -o b && ./b

`state_proj` sits upstream of the frozen VLM, so its gradient needs
`SmolVLAPrefill.backward`: dL/d(prefix embeddings) through all 16 text
layers, 7 `Linear.vjp`s each. Only grad_input is wanted from those 112
calls, and until `Linear.freeze` every one of them also formed dW (one more
GEMM of the same size as dX, plus a bias reduction) and accumulated it into
a gradient buffer nothing reads.

The arms run on the SAME tower, alternated (trainable, frozen, trainable,
frozen) so a thermal or clock drift shows up as disagreement between the two
rounds of one arm instead of as a speed-up:

  * time      median of REPS backward calls per arm, after one warm-up
  * exactness grad_x of the frozen arm vs the trainable arm, every element,
              and `state_proj`'s weight gradient formed from each: both must
              be BIT-identical, because dX never reads dW
  * memory    gradient bytes the tower holds, host and device, before and
              after the freeze (`Param.make` allocates both on a GPU target)

⚠ The weights are `Deterministic`, not the checkpoint: the time does not
depend on the values, and the exactness property holds for any weights.
P = 2 cameras x 64 + SMOLVLA_N_LANG (11) + 1 = 140 tokens, B = 1: the
fine-tune's shape.
"""

from std.math import abs
from std.sys import get_defined_int
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor, ParamVisitorRT
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.deep_agents.smolvla.text import (
    SmolVLMTextLayers, SMOLLM_LAYERS, SMOLLM_DIM, SMOLLM_FF, SMOLLM_HEADS,
    SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, SMOLLM_KV_W, SMOLLM_THETA,
)
from noeira.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from noeira.deep_agents.smolvla.fused import SmolVLAPrefill
from noeira.deep_agents.smolvla.attn_mask import att_2d_mask_square, smolvla_ar
from noeira.deep_agents.smolvla.recording import SO101_N_LANG
from noeira.deep_agents.smolvla.finetune import state_proj_backward

comptime BENCH_CPU = get_defined_int["BENCH_CPU", 0]() != 0
comptime REPS = get_defined_int["REPS", 10]()

comptime L = SMOLLM_LAYERS
comptime W = SMOLLM_DIM
comptime N_CAM = 2
comptime IMG_TOK = 64
comptime P = N_CAM * IMG_TOK + SO101_N_LANG + 1
comptime S = 50
comptime B = 1
comptime SDIM = 32
comptime XN = B * P * W
comptime CACHE_N = L * B * P * SMOLLM_KV_W

comptime Tower = SmolVLMTextLayers[L, W, SMOLLM_FF, SMOLLM_KV_W]
comptime Cache = SmolVLAKVCache[L, P, S, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, B]
comptime Pre = SmolVLAPrefill[
    P, S, B, L, W, SMOLLM_FF, SMOLLM_HEADS, SMOLLM_KV_HEADS,
    SMOLLM_HEAD_DIM, SMOLLM_THETA, True,
]


struct GradBytes(Movable, ParamVisitor, ParamVisitorRT):
    """Bytes held by the gradient buffers of a walk, host and device."""

    var values: Int
    var frozen_values: Int
    var host: Int
    var dev: Int

    def __init__(out self):
        self.values = 0
        self.frozen_values = 0
        self.host = 0
        self.dev = 0

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.values += n
        if grad.n == 0:
            self.frozen_values += n
        self.host += len(grad.data) * 4
        if grad.dev:
            self.dev += grad.n * 4

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


def _mb(b: Int) -> Float64:
    return Float64(b) / (1024.0 * 1024.0)


def grad_bytes[target: StaticString](
    mut t: Tower, ctx: Optional[DeviceContext]
) raises -> GradBytes:
    var g = GradBytes()
    t.for_each_param[target](g, ctx)
    return g^


def freeze_linears(mut t: Tower):
    for i in range(L):
        t.layers[i].q.freeze()
        t.layers[i].k.freeze()
        t.layers[i].v.freeze()
        t.layers[i].o.freeze()
        t.layers[i].mlp.gate.freeze()
        t.layers[i].mlp.up.freeze()
        t.layers[i].mlp.down.freeze()


def unfreeze_linears[target: StaticString](
    mut t: Tower, ctx: Optional[DeviceContext]
) raises:
    for i in range(L):
        t.layers[i].q.unfreeze[target](ctx)
        t.layers[i].k.unfreeze[target](ctx)
        t.layers[i].v.unfreeze[target](ctx)
        t.layers[i].o.unfreeze[target](ctx)
        t.layers[i].mlp.gate.unfreeze[target](ctx)
        t.layers[i].mlp.up.unfreeze[target](ctx)
        t.layers[i].mlp.down.unfreeze[target](ctx)


def _sync[target: StaticString](ctx: Optional[DeviceContext]) raises:
    comptime if target == "gpu":
        ctx.value().synchronize()


def time_backward[target: StaticString](
    mut pre: Pre, mut tower: Tower, mut gck: Tensor, mut gcv: Tensor,
    mut grad_x: Tensor, ctx: Optional[DeviceContext],
) raises -> Float64:
    """Median ms of REPS backward calls, after one warm-up call."""
    pre.backward[target](tower, gck, gcv, grad_x, ctx)
    _sync[target](ctx)
    var ts = List[Float64]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        pre.backward[target](tower, gck, gcv, grad_x, ctx)
        _sync[target](ctx)
        ts.append(Float64(perf_counter_ns() - t0) / 1e6)
    sort(ts)
    return ts[len(ts) // 2]


def snapshot[target: StaticString](
    mut t: Tensor, n: Int, ctx: Optional[DeviceContext]
) raises -> List[Scalar[DT]]:
    comptime if target == "gpu":
        t.download(ctx.value())
    var out = List[Scalar[DT]](capacity=n)
    for i in range(n):
        out.append(t.data[i])
    return out^


def state_proj_grad[target: StaticString](
    mut grad_x: Tensor, mut state: Tensor, ctx: Optional[DeviceContext]
) raises -> List[Scalar[DT]]:
    """`state_proj`'s weight gradient from one grad_x, on a fresh layer."""
    var sp = Linear[SDIM, W].make[target, Deterministic](ctx)
    var g_tok = Tensor()
    var g_state = Tensor()
    state_proj_backward[target, B, P, W, SDIM](
        sp, state, grad_x, g_tok, g_state, ctx
    )
    return snapshot[target](sp.weight.grd, SDIM * W, ctx)


def _mismatches(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Int:
    var bad = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            bad += 1
    return bad


def run[target: StaticString](ctx: Optional[DeviceContext]) raises:
    print("target", target, " P", P, " layers", L, " W", W, " reps", REPS)
    var ar = smolvla_ar(N_CAM * IMG_TOK, SO101_N_LANG, 1, 0)
    var mask = att_2d_mask_square(ar)
    var tower = Tower.make[target, Deterministic](ctx)
    var cache = Cache.make[target](ctx)
    var pre = Pre.make[target](mask, ctx)

    var x = Tensor.alloc(XN)
    for i in range(XN):
        x.data[i] = Scalar[DT](0.02) * Scalar[DT]((i * 29 % 17) - 8)
    var gck = Tensor.alloc(CACHE_N)
    var gcv = Tensor.alloc(CACHE_N)
    for i in range(CACHE_N):
        gck.data[i] = Scalar[DT](0.001) * Scalar[DT]((i * 13 % 11) - 5)
        gcv.data[i] = Scalar[DT](0.001) * Scalar[DT]((i * 17 % 13) - 6)
    var state = Tensor.alloc(B * SDIM)
    for i in range(B * SDIM):
        state.data[i] = Scalar[DT](0.1) * Scalar[DT]((i % 9) - 4)
    var po = Tensor.alloc(XN)
    var grad_x = Tensor.alloc(XN)
    comptime if target == "gpu":
        var c = ctx.value()
        x.upload(c)
        gck.upload(c)
        gcv.upload(c)
        state.upload(c)
        po = Tensor.alloc_gpu(c, XN)
        grad_x = Tensor.alloc_gpu(c, XN)

    pre.run[target](tower, cache, x, po, ctx)
    _sync[target](ctx)

    var g0 = grad_bytes[target](tower, ctx)
    print(
        "  tower params ", g0.values, "   grad bytes host ", _mb(g0.host),
        " MB, device ", _mb(g0.dev), " MB", sep="",
    )

    var t_train = List[Float64]()
    var t_frozen = List[Float64]()
    var gx_train = List[Scalar[DT]]()
    var gx_frozen = List[Scalar[DT]]()
    for rnd in range(2):
        t_train.append(
            time_backward[target](pre, tower, gck, gcv, grad_x, ctx)
        )
        if rnd == 0:
            gx_train = snapshot[target](grad_x, XN, ctx)
        freeze_linears(tower)
        if rnd == 0:
            var g1 = grad_bytes[target](tower, ctx)
            print(
                "  frozen ", g1.frozen_values, " of ", g1.values,
                " values   grad bytes host ", _mb(g1.host), " MB, device ",
                _mb(g1.dev), " MB   released ",
                _mb(g0.host + g0.dev - g1.host - g1.dev), " MB", sep="",
            )
        t_frozen.append(
            time_backward[target](pre, tower, gck, gcv, grad_x, ctx)
        )
        if rnd == 0:
            gx_frozen = snapshot[target](grad_x, XN, ctx)
        unfreeze_linears[target](tower, ctx)
        print(
            "  round ", rnd, "   trainable ", t_train[rnd], " ms   frozen ",
            t_frozen[rnd], " ms   ratio ", t_frozen[rnd] / t_train[rnd],
            sep="",
        )

    var bad_x = _mismatches(gx_train, gx_frozen)
    var gx_mag = Float64(0)
    for i in range(XN):
        gx_mag = max(gx_mag, abs(Float64(gx_train[i])))

    # state_proj's gradient from each arm's grad_x
    var gxt = Tensor.alloc(XN)
    var gxf = Tensor.alloc(XN)
    for i in range(XN):
        gxt.data[i] = gx_train[i]
        gxf.data[i] = gx_frozen[i]
    comptime if target == "gpu":
        gxt.upload(ctx.value())
        gxf.upload(ctx.value())
    var sp_t = state_proj_grad[target](gxt, state, ctx)
    var sp_f = state_proj_grad[target](gxf, state, ctx)
    var bad_sp = _mismatches(sp_t, sp_f)
    var sp_mag = Float64(0)
    for i in range(len(sp_t)):
        sp_mag = max(sp_mag, abs(Float64(sp_t[i])))

    print(
        "  grad_x mismatches ", bad_x, " of ", XN, " (max|grad_x| ", gx_mag,
        ")   state_proj dW mismatches ", bad_sp, " of ", len(sp_t),
        " (max ", sp_mag, ")", sep="",
    )
    var mt = (t_train[0] + t_train[1]) / 2.0
    var mf = (t_frozen[0] + t_frozen[1]) / 2.0
    print(
        "  RESULT backward trainable ", mt, " ms   frozen ", mf,
        " ms   speed-up ", mt / mf, "x", sep="",
    )
    if bad_x != 0 or bad_sp != 0:
        raise Error("freezing the linears changed grad_x: dX read dW somewhere")
    if gx_mag == 0.0 or sp_mag == 0.0:
        raise Error("grad_x is zero: the exactness check compared nothing")


def main() raises:
    comptime if BENCH_CPU:
        run["cpu"](None)
    else:
        var ctx = DeviceContext()
        run["gpu"](Optional(ctx))
