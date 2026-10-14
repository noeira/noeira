"""The SmolVLA QLoRA text-tower gate, for `test_qlora_tower[_gpu].mojo`.

Two text towers with a rank-R adapter on all seven linears of each layer,
through the prefill, forward and backward:

    Q  `SmolVLMTextLayers[..., R, Q4=True]`: int4 `QLinear`s + adapters
    F  the fp32 tower with deq(Q's weights) and zero biases, frozen, and the
       SAME adapters (A, B and α copied)

Compared: the cache's K / V and the final norm after `run`; after `backward`
from a seeded cache gradient, the prefix gradient and every adapter's dA and
dB (walked by name, in walk order, on both towers). B is seeded nonzero, so
dA is not vacuously zero. This is what the QLoRA arm of the fine-tune
trains: a missing or misrouted adapter hook in `proj_fwd` / `proj_vjp` at
`Q4` shows as an O(1) difference.

Bands: the CPU and Metal run fp32 GEMMs on both sides (1e-5). On NVIDIA Q's
forward is MAX's int4 GEMM with bf16 activations (`q4_tower_common.mojo`:
3e-2 for two layers).
"""

from std.math import abs
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.param import ParamVisitor
from noeira.nn.peft.lora import LoRAAdapter
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
comptime R = 4
comptime XN_P = B * P * W
comptime CACHE_N = L * B * P * KVW

comptime TowerQ = SmolVLMTextLayers[L, W, FF, KVW, R, True]
comptime TowerF = SmolVLMTextLayers[L, W, FF, KVW, R]
comptime Cache = SmolVLAKVCache[L, P, S, NKV, HD, B]
comptime PreQ = SmolVLAPrefill[
    P, S, B, L, W, FF, HEADS, NKV, HD, SMOLLM_THETA, True, R, True
]
comptime PreF = SmolVLAPrefill[
    P, S, B, L, W, FF, HEADS, NKV, HD, SMOLLM_THETA, True, R
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
    """A seeded weight RTN-quantized into `ql`; deq(W) and a zero bias into
    `lin`, frozen. Each projection its own salt (`q4_tower_common.mojo`)."""
    seed(salt)
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


def _adapt[
    IN: Int, OUT: Int, RR: Int, //, target: StaticString
](
    mut f: LoRAAdapter[IN, OUT, RR],
    mut q: LoRAAdapter[IN, OUT, RR],
    salt: Int,
    ctx: Optional[DeviceContext],
) raises -> Int:
    """A seeded nonzero B on F's adapter, then F's A and B into Q's."""
    seed(salt)
    comptime NB = LoRAAdapter[IN, OUT, RR].B_SIZE
    comptime NA = LoRAAdapter[IN, OUT, RR].A_SIZE
    for j in range(R):
        for o in range(OUT):
            f.lora_B.val.data[j * OUT + o] = Scalar[DT]((random_float64() - 0.5) * 0.2)
    for i in range(NA):
        q.lora_A.val.data[i] = f.lora_A.val.data[i]
    for i in range(NB):
        q.lora_B.val.data[i] = f.lora_B.val.data[i]
    comptime if target == "gpu":
        f.lora_A.val.upload_resident(ctx.value())
        f.lora_B.val.upload_resident(ctx.value())
        q.lora_A.val.upload_resident(ctx.value())
        q.lora_B.val.upload_resident(ctx.value())
    return 1


def _pair_layer[
    target: StaticString
](
    mut f: DecoderLayerWeights[W, FF, W, KVW, W, R],
    mut q: DecoderLayerWeights[W, FF, W, KVW, W, R, True],
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
    n += _adapt[target](f.q_lora, q.q_lora, s + 11, ctx)
    n += _adapt[target](f.k_lora, q.k_lora, s + 12, ctx)
    n += _adapt[target](f.v_lora, q.v_lora, s + 13, ctx)
    n += _adapt[target](f.o_lora, q.o_lora, s + 14, ctx)
    n += _adapt[target](f.mlp.gate_lora, q.mlp.gate_lora, s + 15, ctx)
    n += _adapt[target](f.mlp.up_lora, q.mlp.up_lora, s + 16, ctx)
    n += _adapt[target](f.mlp.down_lora, q.mlp.down_lora, s + 17, ctx)
    return n


struct _LoraGrads(ParamVisitor):
    """Every `lora_*` param's gradient, downloaded, with its name."""

    var names: List[String]
    var vals: List[Float64]

    def __init__(out self):
        self.names = List[String]()
        self.vals = List[Float64]()

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
        if "lora_" not in name:
            return
        self.names.append(name)
        comptime if target == "gpu":
            grad.download(ctx.value())
        for i in range(N):
            self.vals.append(Float64(grad.data[i]))


def run_qlora_tower[target: StaticString](ctx: Optional[DeviceContext]) raises:
    comptime INT4 = target == "gpu" and Q4_INT4_GEMM
    comptime TOL = 3e-2 if INT4 else 1e-5
    var mask_pre = att_2d_mask_square(smolvla_ar(3, 2, 1, 0))

    seed(3)
    var tf = TowerF.make[target, Deterministic](ctx)
    seed(4)
    var tq = TowerQ.make[target, Deterministic](ctx)
    tf.set_lora_alpha(8.0)
    tq.set_lora_alpha(8.0)
    var paired = 0
    for i in range(L):
        paired += _pair_layer[target](tf.layers[i], tq.layers[i], i, ctx)
    comptime if target == "gpu":
        tq.ensure_q4_scratch(ctx.value())

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
    var e_g = _err(_host[target](gxq, XN_P, ctx), _host[target](gxf, XN_P, ctx))
    var lq = _LoraGrads()
    var lf = _LoraGrads()
    tq.for_each_param[target](lq, ctx)
    tf.for_each_param[target](lf, ctx)
    var same_names = len(lq.names) == len(lf.names)
    if same_names:
        for i in range(len(lq.names)):
            same_names = same_names and lq.names[i] == lf.names[i]
    var e_ad = _err(lq.vals, lf.vals) if len(lq.vals) == len(lf.vals) else Float64(1e9)
    var mag = Float64(0)
    for e in lf.vals:
        mag = max(mag, abs(e))
    print("  [", target, ", int4 GEMM: ", INT4, "] ", paired, " linears and adapters;"
          " QLoRA vs LoRA over deq(W), max |diff| / max |ref|:  cache K ", e_k,
          "  cache V ", e_v, "  final norm ", e_o, "  prefix grad ", e_g,
          "  adapter grads ", e_ad, " (", len(lq.names), " params, max |grad| ", mag,
          ", band ", TOL, ")", sep="")
    assert_true(paired == 14 * L, "not every linear and adapter was paired")
    assert_true(same_names and len(lq.names) == 2 * 7 * L,
                "the two towers do not walk the same adapters")
    assert_true(mag > 1e-4, "the adapter gradients are ~0: the backward half is vacuous")
    assert_true(e_k < TOL and e_v < TOL and e_o < TOL and e_g < TOL and e_ad < TOL,
                "the QLoRA tower differs from LoRA over deq(W)")
