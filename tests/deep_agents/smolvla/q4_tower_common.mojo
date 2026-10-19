"""The SmolVLA int4 text-tower gate, for `test_q4_tower[_gpu].mojo`.

Two text towers through the prefill, forward and backward:

    Q  `SmolVLMTextLayers[..., Q4=True]`: every linear a `QLinear`, its
       weight the RTN quantization of F's initial weight
    F  the fp32 tower, every linear holding deq(Q's weight) and a zero bias
       (a `QLinear` has none), frozen like Q

Both compute the same function, so this gates the 14 `proj_fwd` / `proj_vjp`
dispatches of the prefill at `Q4` and the int4 layer's weights: a projection
dispatched to the wrong `QLinear`, a transposed weight or a missing backward
shows as an O(1) difference in the cache's K/V, the final norm or the prefix
gradient. Compared: the cache after `run` (every layer's post-RoPE K and its
V), `run`'s output, and `backward`'s prefix gradient from a seeded cache
gradient.

Bands: on the CPU and Metal both towers run fp32 GEMMs over the same values
(1e-5). On NVIDIA Q's forward is MAX's int4 GEMM with bf16 activations, about
4e-3 per linear (`tests/nn/quant/test_qlinear_gpu.mojo`), compounding over
two layers of attention and norms: 3e-2.
"""

from std.math import abs
from std.random import random_float64, seed
from std.sys import default_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.quant.qlinear import QLinear, Q4_INT4_GEMM
from noeira.deep_agents.smolvla.text import SmolVLMTextLayers, SMOLLM_THETA
from noeira.deep_agents.smolvla.layer_weights import DecoderLayerWeights
from noeira.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from noeira.deep_agents.smolvla.fused import SmolVLAPrefill
from noeira.deep_agents.smolvla.attn_mask import att_2d_mask_square, smolvla_ar

comptime P = 6
comptime S = 3
comptime B = 1
comptime L = 2
comptime W = 64
comptime FF = 96
comptime HEADS = 2
comptime NKV = 1
comptime HD = 32
comptime KVW = NKV * HD
comptime XN_P = B * P * W
comptime CACHE_N = L * B * P * KVW

comptime TowerQ = SmolVLMTextLayers[L, W, FF, KVW, 0, True]
comptime TowerF = SmolVLMTextLayers[L, W, FF, KVW]
comptime Cache = SmolVLAKVCache[L, P, S, NKV, HD, B]
comptime PreQ = SmolVLAPrefill[
    P, S, B, L, W, FF, HEADS, NKV, HD, SMOLLM_THETA, True, 0, True
]
comptime PreF = SmolVLAPrefill[
    P, S, B, L, W, FF, HEADS, NKV, HD, SMOLLM_THETA, True
]


def _host[target: StaticString](mut t: Tensor, n: Int, ctx: Optional[DeviceContext]) raises -> List[Float64]:
    comptime if target == "gpu":
        t.download(ctx.value())
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(Float64(t.data[i]))
    return out^


def _err(ref a: List[Float64], ref b: List[Float64]) -> Float64:
    var d = Float64(0)
    var m = Float64(0)
    for i in range(len(b)):
        d = max(d, abs(a[i] - b[i]))
        m = max(m, abs(b[i]))
    return d / max(m, 1e-30)


def _pair[
    IN: Int, OUT: Int, //, target: StaticString
](
    mut lin: Linear[IN, OUT],
    mut ql: QLinear[IN, OUT],
    salt: Int,
    ctx: Optional[DeviceContext],
) raises -> Int:
    """Quantize a seeded weight into `ql`, then put deq(W) and a zero bias
    in `lin` and freeze it. Returns 1. ⚠ `salt` gives every projection its
    own weight: q / o, k / v and gate / up share shapes, and a dispatch that
    swapped two of them would be invisible between identical weights."""
    seed(salt)
    # A seeded weight rather than the initializer's: Q4_0 needs magnitudes
    # that vary within a group to have something to round.
    var w = List[Float32](capacity=OUT * IN)
    for _ in range(OUT * IN):
        w.append(Float32((random_float64() - 0.5) * 0.2))
    ql.set_weight[target](w, ctx)
    var deq = ql.dequantized()
    for o in range(OUT):
        for i in range(IN):
            lin.weight.val.data[i * OUT + o] = Scalar[DT](deq[o * IN + i])
    for o in range(OUT):
        lin.bias.val.data[o] = Scalar[DT](0)
    comptime if target == "gpu":
        lin.weight.val.upload_resident(ctx.value())
        lin.bias.val.upload_resident(ctx.value())
    lin.weight.val.version += 1
    lin.freeze()
    return 1


def _pair_layer[
    target: StaticString
](
    mut f: DecoderLayerWeights[W, FF, W, KVW, W],
    mut q: DecoderLayerWeights[W, FF, W, KVW, W, 0, True],
    layer: Int,
    ctx: Optional[DeviceContext],
) raises -> Int:
    var n = 0
    var s = 1000 * (layer + 1)
    n += _pair[target](f.q, q.q_q4, s + 1, ctx)
    n += _pair[target](f.k, q.k_q4, s + 2, ctx)
    n += _pair[target](f.v, q.v_q4, s + 3, ctx)
    n += _pair[target](f.o, q.o_q4, s + 4, ctx)
    n += _pair[target](f.mlp.gate, q.mlp.gate_q4, s + 5, ctx)
    n += _pair[target](f.mlp.up, q.mlp.up_q4, s + 6, ctx)
    n += _pair[target](f.mlp.down, q.mlp.down_q4, s + 7, ctx)
    return n


def run_q4_tower[target: StaticString](ctx: Optional[DeviceContext]) raises:
    comptime INT4 = target == "gpu" and Q4_INT4_GEMM
    comptime TOL = 3e-2 if INT4 else 1e-5
    var mask_pre = att_2d_mask_square(smolvla_ar(3, 2, 1, 0))

    var tf = TowerF.make[target, Deterministic](ctx)
    var tq = TowerQ.make[target, Deterministic](ctx)
    var paired = 0
    for i in range(L):
        paired += _pair_layer[target](tf.layers[i], tq.layers[i], i, ctx)
    # Off NVIDIA the int4 forward dequantizes: one shared scratch, as `load_q4`
    # sets it up.
    comptime if target == "gpu":
        tq.ensure_q4_scratch(ctx.value())
    # The int4 tower walks its norms only: 2 per layer + the final one, each
    # with one gamma.
    var nf = _CountParams()
    tq.for_each_param[target](nf, ctx)

    var cq = Cache.make[target](ctx)
    var cf = Cache.make[target](ctx)
    var pq = PreQ.make[target](mask_pre, ctx)
    var pf = PreF.make[target](mask_pre, ctx)

    seed(11)
    var x = Tensor.alloc(XN_P)
    for i in range(XN_P):
        x.data[i] = Scalar[DT](random_float64() - 0.5)
    var gk = Tensor.alloc(CACHE_N)
    var gv = Tensor.alloc(CACHE_N)
    for i in range(CACHE_N):
        gk.data[i] = Scalar[DT](random_float64() - 0.5)
        gv.data[i] = Scalar[DT](random_float64() - 0.5)
    var oq = Tensor.alloc(XN_P)
    var of_ = Tensor.alloc(XN_P)
    var gxq = Tensor.alloc(XN_P)
    var gxf = Tensor.alloc(XN_P)
    comptime if target == "gpu":
        var c = ctx.value()
        x.upload(c)
        gk.upload(c)
        gv.upload(c)
        oq = Tensor.alloc_gpu(c, XN_P)
        of_ = Tensor.alloc_gpu(c, XN_P)
        gxq = Tensor.alloc_gpu(c, XN_P)
        gxf = Tensor.alloc_gpu(c, XN_P)

    cq.reset()
    cf.reset()
    pq.run[target](tq, cq, x, oq, ctx)
    pf.run[target](tf, cf, x, of_, ctx)
    pq.backward[target](tq, gk, gv, gxq, ctx)
    pf.backward[target](tf, gk, gv, gxf, ctx)

    var e_k = _err(_host[target](cq.k, CACHE_N, ctx), _host[target](cf.k, CACHE_N, ctx))
    var e_v = _err(_host[target](cq.v, CACHE_N, ctx), _host[target](cf.v, CACHE_N, ctx))
    var e_o = _err(_host[target](oq, XN_P, ctx), _host[target](of_, XN_P, ctx))
    var gxf_h = _host[target](gxf, XN_P, ctx)
    var e_g = _err(_host[target](gxq, XN_P, ctx), gxf_h)
    var g_mag = Float64(0)
    for i in range(XN_P):
        g_mag = max(g_mag, abs(gxf_h[i]))
    print("  [", target, ", int4 GEMM: ", INT4, "] ", paired, " linears; int4 vs fp32 over deq(W),"
          " max |diff| / max |fp32|:  cache K ", e_k, "  cache V ", e_v,
          "  final norm ", e_o, "  prefix grad ", e_g, " (band ", TOL,
          "); the int4 tower walks ", nf.n, " params", sep="")
    assert_true(paired == 7 * L, "not every linear was paired")
    assert_true(nf.n == 2 * L + 1, "the int4 tower walks more than its norms")
    assert_true(g_mag > 1e-3, "the prefix gradient is ~0: the backward half is vacuous")
    assert_true(e_k < TOL and e_v < TOL and e_o < TOL and e_g < TOL,
                "the int4 tower differs from the fp32 tower over deq(W)")


from noeira.nn.core.param import ParamVisitor


struct _CountParams(ParamVisitor):
    var n: Int

    def __init__(out self):
        self.n = 0

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
        self.n += 1
