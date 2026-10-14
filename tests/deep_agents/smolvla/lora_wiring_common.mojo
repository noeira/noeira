"""The SmolVLA LoRA wiring gate, for `test_lora_wiring[_gpu].mojo`.

Two models through the whole training chain, prefix embeddings -> prefill ->
cache -> one denoising step -> loss -> both backwards:

    L  both towers with a rank-3 adapter beside every one of their linears
       (seven per layer, both expert layer kinds), B nonzero, scale 2
    M  the same towers WITHOUT adapters, each linear holding W + s · A B,
       the merged weight (`LoRAAdapter.add_into`)

Every adapter is in the computation exactly when L equals M, so this gates the
42 `lora_fwd` / `lora_vjp` hooks in `fused.mojo`: an adapter missing from a
forward, fed the wrong input, or adding into the wrong slot shows as a
difference in the output, the suffix gradient or the prefix gradient. The
adapters' own gradients are checked by the chain rule through the merged
weight: with W' = W + s · A B,

    dL/dA = s · dW' Bᵀ        dL/dB = s · Aᵀ dW'

where dW' is the merged linear's weight gradient, formed by `Linear.vjp` and
not by the adapter. The arithmetic of one adapter is pinned to PEFT's by
`tests/nn/peft/test_peft_io.mojo`; this is the wiring.

⚠ FIVE ADAPTERS HAVE A TRUE ZERO GRADIENT: the last text layer's q, o, gate,
up and down. The loss reaches the prefix only through the cache, which holds
each layer's K/V; the last layer's queries feed nothing but its own attention
output, which, with its MLP, feeds only the prefix's final hidden state, and
that is discarded. Their gradients must be exactly zero in both models, and
the gate counts them: a missing backward hook elsewhere makes a sixth. (A
first version of this gate expected four, forgetting q; the count caught it.)

⚠ The adapter gradients are compared at 1e-4 of their largest value, not at
the 1e-6 of the activations: each side is a separate fp32 GEMM over
activations that already differ by ~1e-6 between the two models, and an
adapter's gradient is a small difference of large terms. A wiring fault is
O(1) (checked by removing a hook), four orders above the band.
"""

from std.math import abs
from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.peft.lora import LoRAAdapter
from noeira.deep_agents.smolvla.text import SmolVLMTextLayers, SMOLLM_THETA
from noeira.deep_agents.smolvla.expert import SmolVLAExpert
from noeira.deep_agents.smolvla.layer_weights import DecoderLayerWeights
from noeira.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from noeira.deep_agents.smolvla.fused import SmolVLAPrefill, SmolVLADenoise
from noeira.deep_agents.smolvla.attn_mask import (
    att_2d_mask, att_2d_mask_square, smolvla_ar,
)

comptime P = 6
comptime S = 3
comptime B = 1
comptime L = 2
comptime W = 8
comptime FF_T = 12
comptime EW = 8
comptime EFF = 12
comptime HEADS = 2
comptime NKV = 1
comptime HD = 4
comptime KVW = NKV * HD
comptime XN_P = B * P * W
comptime XN_S = B * S * EW
comptime CACHE_N = L * B * P * KVW
comptime R = 3
comptime SCALE = 2.0

comptime TowerL = SmolVLMTextLayers[L, W, FF_T, KVW, R]
comptime TowerM = SmolVLMTextLayers[L, W, FF_T, KVW]
comptime ExpertL = SmolVLAExpert[L, EW, EFF, W, KVW, 2, R]
comptime ExpertM = SmolVLAExpert[L, EW, EFF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, S, NKV, HD, B]
comptime PreL = SmolVLAPrefill[
    P, S, B, L, W, FF_T, HEADS, NKV, HD, SMOLLM_THETA, True, R
]
comptime PreM = SmolVLAPrefill[
    P, S, B, L, W, FF_T, HEADS, NKV, HD, SMOLLM_THETA, True
]
comptime DenL = SmolVLADenoise[
    P, S, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True, R
]
comptime DenM = SmolVLADenoise[
    P, S, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True
]


struct Stats(Movable):
    var worst_a: Float64
    var worst_b: Float64
    var live: Int
    var dead: Int
    var dead_nonzero: Int
    var seq: Int
    var seq1: Int
    var at_a: Int
    var at_b: Int

    def __init__(out self):
        self.worst_a = 0
        self.worst_b = 0
        self.live = 0
        self.dead = 0
        self.dead_nonzero = 0
        self.seq = 0
        self.seq1 = 0
        self.at_a = -1
        self.at_b = -1


def _host[target: StaticString](mut t: Tensor, n: Int, ctx: Optional[DeviceContext]) raises -> List[Float64]:
    comptime if target == "gpu":
        t.download(ctx.value())
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(Float64(t.data[i]))
    return out^


def _pair[
    IN: Int, OUT: Int, //, target: StaticString, MODE: Int
](
    mut a: LoRAAdapter[IN, OUT, R],
    mut lin: Linear[IN, OUT],
    mut st: Stats,
    ctx: Optional[DeviceContext],
) raises:
    """MODE 0: give `a` a nonzero B and scale 2, and merge it into `lin`.
    MODE 1: `a`'s gradients against the chain rule through `lin`'s."""
    comptime RP = LoRAAdapter[IN, OUT, R].R_PAD
    comptime if MODE == 0:
        st.seq += 1
        for j in range(R):
            for o in range(OUT):
                a.lora_B.val.data[j * OUT + o] = Scalar[DT](
                    Float64(((j * 5 + o * 3 + st.seq) % 7) - 3) * 0.2
                )
        comptime if target == "gpu":
            a.lora_B.val.upload_resident(ctx.value())
        a.alpha = SCALE * Float64(R)
        a.add_into[target](lin.weight.val, ctx)
    else:
        st.seq1 += 1
        var av = _host[target](a.lora_A.val, IN * RP, ctx)
        var bv = _host[target](a.lora_B.val, RP * OUT, ctx)
        var da = _host[target](a.lora_A.grd, IN * RP, ctx)
        var db = _host[target](a.lora_B.grd, RP * OUT, ctx)
        var dw = _host[target](lin.weight.grd, IN * OUT, ctx)
        var s = a.scale()
        var ea = List[Float64](length=IN * R, fill=0.0)
        var eb = List[Float64](length=R * OUT, fill=0.0)
        var mag = Float64(0)
        for i in range(IN):
            for j in range(R):
                var acc = Float64(0)
                for o in range(OUT):
                    acc += dw[i * OUT + o] * bv[j * OUT + o]
                ea[i * R + j] = s * acc
                mag = max(mag, abs(s * acc))
        for j in range(R):
            for o in range(OUT):
                var acc = Float64(0)
                for i in range(IN):
                    acc += av[i * RP + j] * dw[i * OUT + o]
                eb[j * OUT + o] = s * acc
                mag = max(mag, abs(s * acc))
        if mag == 0.0:
            st.dead += 1
            for i in range(IN):
                for j in range(R):
                    if da[i * RP + j] != 0.0:
                        st.dead_nonzero += 1
            for j in range(R):
                for o in range(OUT):
                    if db[j * OUT + o] != 0.0:
                        st.dead_nonzero += 1
            return
        st.live += 1
        var ma = Float64(0)
        var mb = Float64(0)
        var sa = Float64(0)
        var sb = Float64(0)
        for i in range(IN):
            for j in range(R):
                ma = max(ma, abs(da[i * RP + j] - ea[i * R + j]))
                sa = max(sa, abs(ea[i * R + j]))
        for j in range(R):
            for o in range(OUT):
                mb = max(mb, abs(db[j * OUT + o] - eb[j * OUT + o]))
                sb = max(sb, abs(eb[j * OUT + o]))
        if ma / max(sa, 1e-30) > st.worst_a:
            st.worst_a = ma / max(sa, 1e-30)
            st.at_a = st.seq1
        if mb / max(sb, 1e-30) > st.worst_b:
            st.worst_b = mb / max(sb, 1e-30)
            st.at_b = st.seq1


def _pair_layer[
    QW: Int, KV_IN: Int, LW: Int, LFF: Int, //, target: StaticString, MODE: Int
](
    mut l: DecoderLayerWeights[LW, LFF, QW, KVW, KV_IN, R],
    mut m: DecoderLayerWeights[LW, LFF, QW, KVW, KV_IN],
    mut st: Stats,
    ctx: Optional[DeviceContext],
) raises:
    _pair[target, MODE](l.q_lora, m.q, st, ctx)
    _pair[target, MODE](l.k_lora, m.k, st, ctx)
    _pair[target, MODE](l.v_lora, m.v, st, ctx)
    _pair[target, MODE](l.o_lora, m.o, st, ctx)
    _pair[target, MODE](l.mlp.gate_lora, m.mlp.gate, st, ctx)
    _pair[target, MODE](l.mlp.up_lora, m.mlp.up, st, ctx)
    _pair[target, MODE](l.mlp.down_lora, m.mlp.down, st, ctx)


def _pair_all[
    target: StaticString, MODE: Int
](
    mut tl: TowerL, mut tm: TowerM, mut el: ExpertL, mut em: ExpertM,
    mut st: Stats, ctx: Optional[DeviceContext],
) raises:
    for i in range(L):
        _pair_layer[target, MODE](tl.layers[i], tm.layers[i], st, ctx)
    for i in range(ExpertL.N_SELF):
        _pair_layer[target, MODE](el.self_layers[i], em.self_layers[i], st, ctx)
    for i in range(ExpertL.N_CROSS):
        _pair_layer[target, MODE](el.cross_layers[i], em.cross_layers[i], st, ctx)


def _err(ref a: List[Float64], ref b: List[Float64]) -> Float64:
    var d = Float64(0)
    var m = Float64(0)
    for i in range(len(b)):
        d = max(d, abs(a[i] - b[i]))
        m = max(m, abs(b[i]))
    return d / max(m, 1e-30)


def run_wiring[target: StaticString](ctx: Optional[DeviceContext]) raises:
    var ar_pre = smolvla_ar(3, 2, 1, 0)
    var ar_full = smolvla_ar(3, 2, 1, S)
    var mask_pre = att_2d_mask_square(ar_pre)
    var mask_self = att_2d_mask(ar_full, P, P + S, 0, P + S)
    var mask_cross = att_2d_mask(ar_full, P, P + S, 0, P)

    seed(5)
    var tl = TowerL.make[target, Deterministic](ctx)
    var el = ExpertL.make[target, Deterministic](ctx)
    var tm = TowerM.make[target, Deterministic](ctx)
    var em = ExpertM.make[target, Deterministic](ctx)
    var st = Stats()
    _pair_all[target, 0](tl, tm, el, em, st, ctx)
    var n_adapters = st.seq

    var cl = Cache.make[target](ctx)
    var cm = Cache.make[target](ctx)
    var pl = PreL.make[target](mask_pre, ctx)
    var pm = PreM.make[target](mask_pre, ctx)
    var dl = DenL.make[target](mask_self, mask_cross, ctx)
    var dm = DenM.make[target](mask_self, mask_cross, ctx)

    var x = Tensor.alloc(XN_P)
    for i in range(XN_P):
        x.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.06
    var xs = Tensor.alloc(XN_S)
    for i in range(XN_S):
        xs.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.05
    var go = Tensor.alloc(XN_S)
    for i in range(XN_S):
        go.data[i] = Scalar[DT](Float32(((i * 23) % 7) - 3) * 0.3)
    comptime if target == "gpu":
        x.upload(ctx.value())
        xs.upload(ctx.value())
        go.upload(ctx.value())

    # The two models, through the same chain.
    var out_l = Tensor.alloc(XN_S)
    var out_m = Tensor.alloc(XN_S)
    var po = Tensor.alloc(XN_P)
    var gxs_l = Tensor.alloc(XN_S)
    var gxs_m = Tensor.alloc(XN_S)
    var gck_l = Tensor.alloc(CACHE_N)
    var gcv_l = Tensor.alloc(CACHE_N)
    var gck_m = Tensor.alloc(CACHE_N)
    var gcv_m = Tensor.alloc(CACHE_N)
    var gx_l = Tensor.alloc(XN_P)
    var gx_m = Tensor.alloc(XN_P)
    comptime if target == "gpu":
        var c = ctx.value()
        out_l = Tensor.alloc_gpu(c, XN_S)
        out_m = Tensor.alloc_gpu(c, XN_S)
        po = Tensor.alloc_gpu(c, XN_P)
        gxs_l = Tensor.alloc_gpu(c, XN_S)
        gxs_m = Tensor.alloc_gpu(c, XN_S)
        gck_l = Tensor.alloc_gpu(c, CACHE_N)
        gcv_l = Tensor.alloc_gpu(c, CACHE_N)
        gck_m = Tensor.alloc_gpu(c, CACHE_N)
        gcv_m = Tensor.alloc_gpu(c, CACHE_N)
        gx_l = Tensor.alloc_gpu(c, XN_P)
        gx_m = Tensor.alloc_gpu(c, XN_P)

    # Fresh models: every gradient starts at zero (`Param.make`).
    cl.reset()
    cm.reset()
    pl.run[target](tl, cl, x, po, ctx)
    pm.run[target](tm, cm, x, po, ctx)
    dl.step[target](el, cl, xs, out_l, ctx)
    dm.step[target](em, cm, xs, out_m, ctx)
    dl.backward[target](el, cl, go, gxs_l, gck_l, gcv_l, ctx)
    dm.backward[target](em, cm, go, gxs_m, gck_m, gcv_m, ctx)
    pl.backward[target](tl, gck_l, gcv_l, gx_l, ctx)
    pm.backward[target](tm, gck_m, gcv_m, gx_m, ctx)

    var e_out = _err(_host[target](out_l, XN_S, ctx), _host[target](out_m, XN_S, ctx))
    var e_gxs = _err(_host[target](gxs_l, XN_S, ctx), _host[target](gxs_m, XN_S, ctx))
    var e_gx = _err(_host[target](gx_l, XN_P, ctx), _host[target](gx_m, XN_P, ctx))
    var e_gck = _err(_host[target](gck_l, CACHE_N, ctx), _host[target](gck_m, CACHE_N, ctx))
    var gx_mag = Float64(0)
    var gxm = _host[target](gx_m, XN_P, ctx)
    for i in range(XN_P):
        gx_mag = max(gx_mag, abs(gxm[i]))
    _pair_all[target, 1](tl, tm, el, em, st, ctx)
    print("  [", target, "] adapted vs merged, max |diff| / max |merged|:  output ", e_out,
          "  suffix grad ", e_gxs, "  cache K grad ", e_gck, "  prefix grad ", e_gx,
          " (max |prefix grad| ", gx_mag, ")", sep="")
    print("  [", target, "] ", n_adapters, " adapters: ", st.live, " live, dA vs s·dW'Bᵀ ",
          st.worst_a, " (adapter #", st.at_a, "), dB vs s·Aᵀ dW' ", st.worst_b,
          " (#", st.at_b, "); ", st.dead, " with a true zero gradient (",
          st.dead_nonzero, " nonzero values among them)", sep="")
    print("        (adapters numbered 1.. in walk order: text layers 0..", L - 1,
          ", then expert self, then cross; q k v o gate up down each)", sep="")
    assert_true(n_adapters == 4 * 7, "not every linear of both towers got an adapter")
    assert_true(e_out < 1e-5 and e_gxs < 1e-5 and e_gx < 1e-5 and e_gck < 1e-5,
                "an adapter is missing from, or misplaced in, the fused loop")
    assert_true(gx_mag > 1e-3, "the prefix gradient is ~0: the tower half is vacuous")
    assert_true(st.worst_a < 1e-4 and st.worst_b < 1e-4,
                "an adapter's gradient disagrees with the chain rule through its merged twin")
    assert_true(st.dead == 5 and st.dead_nonzero == 0,
                "expected exactly the last text layer's q, o, gate, up, down to have no gradient")
