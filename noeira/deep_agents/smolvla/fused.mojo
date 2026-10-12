# +--------------------------------------------------------------------------+ #
# | SmolVLA — the fused loop, prefill pass
# +--------------------------------------------------------------------------+ #
"""Run the image+language+state prefix through the 16 VLM layers ONCE, filling
the per-layer KV cache the ten denoising steps then read.

This is mode 2 of the three in `smolvlm_with_expert.py` (`[prefix, None]`,
`fill_kv_cache=True`). Every layer runs ordinary self-attention over the prefix
and stores its **post-RoPE** K/V.

    for i in range(16):
        h  = layers[i].input_layernorm(x)
        q  = q(h);  k = k(h);  v = v(h)
        q  = RoPE(q);  k = RoPE(k)            <- rotate BEFORE caching
        cache.write_prefix(i, k, v)
        a  = attention(q, repeat_kv(k), repeat_kv(v), block_mask)
        x  = x + o(a)
        x  = x + mlp(post_attention_layernorm(x))
    x = norm(x)

⚠ **The K/V written to the cache are the post-RoPE ones**, and they are written
*before* the head broadcast — the cache holds 5 KV heads, not 15. Caching the
broadcast copies would be 3x the memory for the same information, and caching
pre-RoPE keys would be shape-identical and wrong (see `kv_cache.mojo`).

⚠ **The mask is not causal.** It is the prefix-LM block mask from
`attn_mask.mojo`: image and language form one bidirectional block, state its own.
Passing a causal mask here would run, and would change what every prefix token
sees.

Forward-only: this is V1's inference path, and none of the leaves are asked for
a vjp. Training (mode 1, both streams concatenated) reuses these weights through
a different driver.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer, Deterministic
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.primitives.rope import RoPE
from noeira.nn.primitives.repeat_kv_heads import RepeatKVHeads
from noeira.nn.primitives.masked_attention import MaskedAttention
from noeira.nn.primitives.concat import Concat
from noeira.nn.primitives.swiglu import SwiGLU
from noeira.nn.primitives.add import Add

from .text import (
    SmolVLMTextLayers, SMOLLM_DIM, SMOLLM_FF, SMOLLM_HEADS, SMOLLM_KV_HEADS,
    SMOLLM_HEAD_DIM, SMOLLM_REP, SMOLLM_LAYERS, SMOLLM_THETA, SMOLLM_KV_W,
)
from .kv_cache import SmolVLAKVCache
from .layer_weights import (
    LORA_Q, LORA_K, LORA_V, LORA_O, LORA_GATE, LORA_UP, LORA_DOWN,
)
from .expert import SmolVLAExpert
from .block_attention import BlockCrossAttention
from .grad_ops import (
    accum_into, copy_into, zero_into, suffix_tail, prefix_head,
)



def _load_cache_grad[
    target: StaticString, LAYER_N: Int
](
    layer: Int, mut src: Tensor, mut dst: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Read one layer's `[B*P*KVW]` slice out of a cache-shaped slab.

    The mirror of `_store_cache_grad`, and the two must index identically —
    `layer * LAYER_N` in both, or the denoise writes layer i's gradient where
    the prefill reads layer j's, which is finite, plausible and wrong.
    """
    var off = layer * LAYER_N
    comptime if target == "cpu":
        dst.ensure(LAYER_N)
        for i in range(LAYER_N):
            dst.data[i] = src.data[off + i]
    else:
        var c = ctx.value()
        dst.ensure_gpu(c, LAYER_N)
        c.enqueue_copy(
            dst.dev.value().create_sub_buffer[DT](0, LAYER_N),
            src.dev.value().create_sub_buffer[DT](off, LAYER_N),
        )


struct SmolVLAPrefill[
    P: Int,
    SUFFIX: Int,
    B: Int = 1,
    LAYERS: Int = SMOLLM_LAYERS,
    W: Int = SMOLLM_DIM,
    FF: Int = SMOLLM_FF,
    HEADS: Int = SMOLLM_HEADS,
    N_KV: Int = SMOLLM_KV_HEADS,
    HD: Int = SMOLLM_HEAD_DIM,
    THETA: Float64 = SMOLLM_THETA,
    # V2 stage 7. See `SmolVLADenoise`'s RECORD — same parameter, same reason.
    RECORD: Bool = False,
    # The tower's LoRA rank (`layer_weights.mojo`); 0 = no adapters, and every
    # adapter hook inside `proj_fwd` / `proj_vjp` compiles to nothing.
    LORA_R: Int = 0,
    # The tower's linears are int4 `QLinear`s (`layer_weights.mojo`, Q4).
    Q4: Bool = False,
](Movable):
    """The prefill driver. Owns the stateless leaves and every scratch slab, so
    one instance runs all 16 layers without allocating in the loop."""

    comptime REP: Int = Self.HEADS // Self.N_KV
    comptime KVW: Int = Self.N_KV * Self.HD
    comptime XN: Int = Self.P * Self.W
    comptime KVN: Int = Self.P * Self.KVW
    comptime FFN: Int = Self.P * Self.FF

    comptime RoPEQ = RoPE[Self.P, Self.HEADS, Self.HD, Self.THETA]
    comptime RoPEK = RoPE[Self.P, Self.N_KV, Self.HD, Self.THETA]
    comptime RepKV = RepeatKVHeads[Self.P, Self.N_KV, Self.REP, Self.HD]
    comptime Attn = MaskedAttention[Self.W, Self.HEADS, Self.P]
    comptime Pack = Concat[Self.XN, Self.XN, Self.XN]
    comptime Glu = SwiGLU[Self.FFN]
    comptime GluCat = Concat[Self.FFN, Self.FFN]
    comptime Res = Add[Self.XN]

    var rope_q: Self.RoPEQ
    var rope_k: Self.RoPEK
    var rep_k: Self.RepKV
    var rep_v: Self.RepKV
    var attn: Self.Attn
    var pack: Self.Pack
    var glu: Self.Glu
    var glu_cat: Self.GluCat
    var res: Self.Res

    # ── scratch ──────────────────────────────────────────────────────────
    # ONE pool, not eighteen fields. `TensorRefs[2]`/`[3]` require every operand
    # to share a single origin, and two `self.` fields do not — `TensorPack`'s
    # subscript returns a `MutAnyOrigin` ref precisely so adjacent slabs can be
    # passed together. Same reason `ComputeGraph` copies its inputs into an
    # owned pool rather than referencing the caller's tensors.
    comptime X = 0        # the running activation
    comptime H = 1
    comptime Q = 2
    comptime K = 3
    comptime V = 4
    comptime QR = 5
    comptime KR = 6
    comptime KX = 7
    comptime VX = 8
    comptime PACKED = 9
    comptime ATT = 10
    comptime AO = 11
    comptime X2 = 12
    comptime GATE = 13
    comptime UP = 14
    comptime CAT = 15
    comptime GLU = 16
    comptime DOWN = 17
    comptime N_BASE = 18
    comptime XO: Int = Self.N_BASE if Self.RECORD else Self.X
    comptime H2: Int = Self.N_BASE + 1 if Self.RECORD else Self.H
    comptime N_SLOTS: Int = (
        Self.N_BASE + 2 if Self.RECORD else Self.N_BASE
    )
    comptime N_POOLS: Int = Self.LAYERS + 1 if Self.RECORD else 1
    comptime LAST: Int = Self.LAYERS if Self.RECORD else 0

    var pools: List[TensorPack[Self.N_SLOTS]]

    # ── the backward's scratch ───────────────────────────────────────────
    comptime GXO = 0
    comptime GX2 = 1
    comptime GDOWN = 2
    comptime GGLU = 3
    comptime GCAT = 4
    comptime GUP = 5
    comptime GGATE = 6
    comptime GHA = 7
    comptime GHB = 8
    comptime GHC = 9
    comptime GAO = 10
    comptime GATT = 11
    comptime GPACKED = 12
    comptime GQR = 13
    comptime GKX = 14
    comptime GVX = 15
    comptime GKR = 16
    comptime GV = 17
    comptime GQ = 18
    comptime GK = 19
    comptime GRC = 20     # SwiGLU cache-refill scratch
    comptime GRC2 = 21    # MaskedAttention cache-refill scratch
    comptime GCK = 22     # this layer's slice of dL/d(cache K)
    comptime GCV = 23
    comptime G_SLOTS = 24

    var g: TensorPack[Self.G_SLOTS]

    def __init__(out self):
        comptime assert Self.W == Self.HEADS * Self.HD, (
            "SmolVLAPrefill: W must equal HEADS * HD"
        )
        self.rope_q = Self.RoPEQ()
        self.rope_k = Self.RoPEK()
        self.rep_k = Self.RepKV()
        self.rep_v = Self.RepKV()
        self.attn = Self.Attn()
        self.pack = Self.Pack()
        self.glu = Self.Glu()
        self.glu_cat = Self.GluCat()
        self.res = Self.Res()
        self.pools = List[TensorPack[Self.N_SLOTS]]()
        for _ in range(Self.N_POOLS):
            self.pools.append(TensorPack[Self.N_SLOTS]())
        self.g = TensorPack[Self.G_SLOTS]()

    def __init__(out self, *, deinit move: Self):
        self.rope_q = move.rope_q^
        self.rope_k = move.rope_k^
        self.rep_k = move.rep_k^
        self.rep_v = move.rep_v^
        self.attn = move.attn^
        self.pack = move.pack^
        self.glu = move.glu^
        self.glu_cat = move.glu_cat^
        self.res = move.res^
        self.pools = move.pools^
        self.g = move.g^

    @staticmethod
    def make[
        target: StaticString
    ](
        ref mask: List[Scalar[DT]], ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        """`mask` is the `[P, P]` additive block mask from `attn_mask.mojo`."""
        var s = Self()
        s.rope_q = Self.RoPEQ.make[target, Deterministic](ctx)
        s.rope_k = Self.RoPEK.make[target, Deterministic](ctx)
        s.rep_k = Self.RepKV.make[target, Deterministic](ctx)
        s.rep_v = Self.RepKV.make[target, Deterministic](ctx)
        s.attn = Self.Attn.make[target, Deterministic](ctx)
        s.attn.set_mask(mask.copy(), ctx)
        s.pack = Self.Pack.make[target, Deterministic](ctx)
        s.glu = Self.Glu.make[target, Deterministic](ctx)
        s.glu_cat = Self.GluCat.make[target, Deterministic](ctx)
        s.res = Self.Res.make[target, Deterministic](ctx)
        return s^

    def run[
        target: StaticString
    ](
        mut self,
        mut tower: SmolVLMTextLayers[
            Self.LAYERS, Self.W, Self.FF, Self.KVW, Self.LORA_R, Self.Q4
        ],
        mut cache: SmolVLAKVCache[
            Self.LAYERS, Self.P, Self.SUFFIX, Self.N_KV, Self.HD, Self.B
        ],
        mut x: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`x` is `[B, P*W]` prefix embeddings; `out` receives the final norm."""
        comptime TOK = Self.B * Self.P   # Linear/RMSNorm run per token
        comptime XN = Self.B * Self.XN

        # Seed the pool. The running activation lives in a slot from here on, so
        # every residual pairs two slabs of one origin.
        comptime if target == "cpu":
            self.pools[0][Self.X].ensure(XN)
            for i in range(XN):
                self.pools[0][Self.X].data[i] = x.data[i]
        else:
            var c = ctx.value()
            self.pools[0][Self.X].ensure_gpu(c, XN)
            c.enqueue_copy(
                self.pools[0][Self.X].dev.value().create_sub_buffer[DT](0, XN),
                x.dev.value().create_sub_buffer[DT](0, XN),
            )

        for i in range(Self.LAYERS):
            var pi = i if Self.RECORD else 0
            ref PK = self.pools[pi]
            # ── attention branch ─────────────────────────────────────────
            tower.layers[i].input_layernorm.forward[target, TOK](
                TensorRefs[1](PK[Self.X]), PK[Self.H], ctx
            )
            tower.layers[i].proj_fwd[LORA_Q, target, TOK](
                TensorRefs[1](PK[Self.H]), PK[Self.Q], ctx
            )
            tower.layers[i].proj_fwd[LORA_K, target, TOK](
                TensorRefs[1](PK[Self.H]), PK[Self.K], ctx
            )
            tower.layers[i].proj_fwd[LORA_V, target, TOK](
                TensorRefs[1](PK[Self.H]), PK[Self.V], ctx
            )
            # RoPE on q and k, never on v.
            self.rope_q.forward[target, Self.B](
                TensorRefs[1](PK[Self.Q]), PK[Self.QR], ctx
            )
            self.rope_k.forward[target, Self.B](
                TensorRefs[1](PK[Self.K]), PK[Self.KR], ctx
            )
            # Cache the POST-RoPE, PRE-broadcast K/V: 5 heads, not 15.
            cache.write_prefix[target](
                i, PK[Self.KR], PK[Self.V], ctx
            )
            self.rep_k.forward[target, Self.B](
                TensorRefs[1](PK[Self.KR]), PK[Self.KX], ctx
            )
            self.rep_v.forward[target, Self.B](
                TensorRefs[1](PK[Self.V]), PK[Self.VX], ctx
            )
            self.pack.forward[target, Self.B](
                TensorRefs[3](
                    PK[Self.QR], PK[Self.KX], PK[Self.VX]
                ),
                PK[Self.PACKED], ctx,
            )
            self.attn.forward[target, Self.B](
                TensorRefs[1](PK[Self.PACKED]), PK[Self.ATT], ctx
            )
            tower.layers[i].proj_fwd[LORA_O, target, TOK](
                TensorRefs[1](PK[Self.ATT]), PK[Self.AO], ctx
            )
            self.res.forward[target, Self.B](
                TensorRefs[2](PK[Self.X], PK[Self.AO]),
                PK[Self.X2], ctx,
            )

            # ── MLP branch ───────────────────────────────────────────────
            tower.layers[i].post_attention_layernorm.forward[target, TOK](
                TensorRefs[1](PK[Self.X2]), PK[Self.H2], ctx
            )
            tower.layers[i].proj_fwd[LORA_GATE, target, TOK](
                TensorRefs[1](PK[Self.H2]), PK[Self.GATE], ctx
            )
            tower.layers[i].proj_fwd[LORA_UP, target, TOK](
                TensorRefs[1](PK[Self.H2]), PK[Self.UP], ctx
            )
            # ⚠ (up, gate): SwiGLU reads [u ‖ v] -> u * silu(v), and the
            # reference computes down(silu(gate) * up). Reversed is the same
            # shape and a different function.
            self.glu_cat.forward[target, Self.B](
                TensorRefs[2](PK[Self.UP], PK[Self.GATE]),
                PK[Self.CAT], ctx,
            )
            self.glu.forward[target, Self.B](
                TensorRefs[1](PK[Self.CAT]), PK[Self.GLU], ctx
            )
            tower.layers[i].proj_fwd[LORA_DOWN, target, TOK](
                TensorRefs[1](PK[Self.GLU]), PK[Self.DOWN], ctx
            )
            self.res.forward[target, Self.B](
                TensorRefs[2](PK[Self.X2], PK[Self.DOWN]),
                PK[Self.XO], ctx,
            )

            comptime if Self.RECORD:
                var pn = i + 1
                comptime if target == "cpu":
                    self.pools[pn][Self.X].ensure(XN)
                    var dst = self.pools[pn][Self.X].data.unsafe_ptr()
                    var src = self.pools[pi][Self.XO].data.unsafe_ptr()
                    for t in range(XN):
                        dst[unsafe_offset=t] = src[unsafe_offset=t]
                else:
                    var c2 = ctx.value()
                    self.pools[pn][Self.X].ensure_gpu(c2, XN)
                    var sb = self.pools[pi][Self.XO].dev.value(
                    ).create_sub_buffer[DT](0, XN)
                    var db = self.pools[pn][Self.X].dev.value(
                    ).create_sub_buffer[DT](0, XN)
                    c2.enqueue_copy(db, sb)

        tower.norm.forward[target, TOK](
            TensorRefs[1](self.pools[Self.LAST][Self.X]), out, ctx
        )



    def backward[
        target: StaticString
    ](
        mut self,
        mut tower: SmolVLMTextLayers[
            Self.LAYERS, Self.W, Self.FF, Self.KVW, Self.LORA_R, Self.Q4
        ],
        mut grad_cache_k: Tensor,
        mut grad_cache_v: Tensor,
        mut grad_x: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Reverse of `run`, from the cache gradient down to the embeddings.

        `grad_cache_k` / `grad_cache_v` are what `SmolVLADenoise.backward`
        produced. `grad_x` receives dL/d(the prefix embeddings), whose STATE
        token slice is `state_proj`'s gradient.

        ⚠ **It starts from ZERO, not from a `grad_out`.** `run`'s `out` — the
        final norm of the prefix — is discarded by the training forward
        (`(_, suffix_out), _ = vlm_with_expert.forward(...)`), so the loss does
        not read it and its gradient is genuinely zero. EVERY bit of gradient
        in this pass enters sideways, through the per-layer cached K/V. A
        `grad_out` parameter here would be an invitation to pass something
        nonzero and train on a term the reference does not have.

        ⚠ **Two sources meet at KR and V.** Layer i's post-RoPE K and its V
        each feed TWO consumers: the cache (read by every denoising step) and
        this layer's own attention. Their gradients must be SUMMED. Taking
        only the attention side leaves `state_proj` with a gradient that looks
        reasonable and is missing most of itself; taking only the cache side
        silently deletes the prefix's internal structure.

        ⚠ **No VLM weight gradient is wanted** — the tower is frozen. Freeze
        it (`SmolVLAPolicy.freeze_backbone`, or `FreezeParams` over the tower)
        and `Linear.vjp` / `RMSNorm.vjp` form grad_input only: no dW GEMM, no
        bias or gamma sums, no gradient buffers. On an unfrozen tower they
        still form and accumulate every weight gradient, which nothing reads
        or zeroes (1.30x on a 5090 at one observation, 4.2x on Metal).
        """
        comptime assert Self.RECORD, (
            "SmolVLAPrefill.backward needs RECORD=True — the non-recording"
            " driver keeps one pool and its tape is the last layer's"
            " activations sixteen times over"
        )
        comptime TOK = Self.B * Self.P
        comptime XN = Self.B * Self.XN
        comptime KVN = Self.B * Self.KVN
        comptime FFN = Self.B * Self.FFN
        comptime CACHE_LAYER = Self.B * Self.P * Self.KVW

        zero_into[target, XN](self.g[Self.GXO], ctx)
        # An int4 tower's grad_input dequantizes: one shared scratch.
        comptime if Self.Q4 and target == "gpu":
            tower.ensure_q4_scratch(ctx.value())

        for ridx in range(Self.LAYERS):
            var i = Self.LAYERS - 1 - ridx
            ref PK = self.pools[i]

            # ── MLP branch, reversed ─────────────────────────────────────
            self.res.vjp[target, Self.B](
                TensorRefs[2](PK[Self.X2], PK[Self.DOWN]),
                self.g[Self.GXO],
                TensorRefs[2](self.g[Self.GX2], self.g[Self.GDOWN]),
                ctx,
            )
            tower.layers[i].proj_vjp[LORA_DOWN, target, TOK](
                TensorRefs[1](PK[Self.GLU]), self.g[Self.GDOWN],
                TensorRefs[1](self.g[Self.GGLU]), ctx,
            )
            # ⚠ SwiGLU is output-caching with ONE instance for all sixteen
            # layers — see `SmolVLADenoise.backward`. Refill with this layer's
            # own CAT before differentiating.
            self.glu.forward[target, Self.B](
                TensorRefs[1](PK[Self.CAT]), self.g[Self.GRC], ctx
            )
            self.glu.vjp[target, Self.B](
                TensorRefs[1](PK[Self.CAT]), self.g[Self.GGLU],
                TensorRefs[1](self.g[Self.GCAT]), ctx,
            )
            self.glu_cat.vjp[target, Self.B](
                TensorRefs[2](PK[Self.UP], PK[Self.GATE]),
                self.g[Self.GCAT],
                TensorRefs[2](self.g[Self.GUP], self.g[Self.GGATE]),
                ctx,
            )
            tower.layers[i].proj_vjp[LORA_UP, target, TOK](
                TensorRefs[1](PK[Self.H2]), self.g[Self.GUP],
                TensorRefs[1](self.g[Self.GHA]), ctx,
            )
            tower.layers[i].proj_vjp[LORA_GATE, target, TOK](
                TensorRefs[1](PK[Self.H2]), self.g[Self.GGATE],
                TensorRefs[1](self.g[Self.GHB]), ctx,
            )
            accum_into[target, XN](self.g[Self.GHA], self.g[Self.GHB], ctx)
            tower.layers[i].post_attention_layernorm.vjp[target, TOK](
                TensorRefs[1](PK[Self.X2]), self.g[Self.GHA],
                TensorRefs[1](self.g[Self.GHC]), ctx,
            )
            accum_into[target, XN](self.g[Self.GX2], self.g[Self.GHC], ctx)

            # ── attention branch, reversed ───────────────────────────────
            self.res.vjp[target, Self.B](
                TensorRefs[2](PK[Self.X], PK[Self.AO]),
                self.g[Self.GX2],
                TensorRefs[2](self.g[Self.GHC], self.g[Self.GAO]),
                ctx,
            )
            tower.layers[i].proj_vjp[LORA_O, target, TOK](
                TensorRefs[1](PK[Self.ATT]), self.g[Self.GAO],
                TensorRefs[1](self.g[Self.GATT]), ctx,
            )
            # ⚠ MaskedAttention is ALSO output-caching, ALSO one instance for
            # sixteen layers. Same refill, same reason.
            self.attn.forward[target, Self.B](
                TensorRefs[1](PK[Self.PACKED]), self.g[Self.GRC2], ctx
            )
            self.attn.vjp[target, Self.B](
                TensorRefs[1](PK[Self.PACKED]), self.g[Self.GATT],
                TensorRefs[1](self.g[Self.GPACKED]), ctx,
            )
            self.pack.vjp[target, Self.B](
                TensorRefs[3](PK[Self.QR], PK[Self.KX], PK[Self.VX]),
                self.g[Self.GPACKED],
                TensorRefs[3](
                    self.g[Self.GQR], self.g[Self.GKX], self.g[Self.GVX]
                ),
                ctx,
            )
            self.rep_k.vjp[target, Self.B](
                TensorRefs[1](PK[Self.KR]), self.g[Self.GKX],
                TensorRefs[1](self.g[Self.GKR]), ctx,
            )
            self.rep_v.vjp[target, Self.B](
                TensorRefs[1](PK[Self.V]), self.g[Self.GVX],
                TensorRefs[1](self.g[Self.GV]), ctx,
            )

            # ⚠ THE SECOND SOURCE. KR and V went to the cache as well as to
            # the attention, so the denoise's gradient is added here.
            _load_cache_grad[target, CACHE_LAYER](
                i, grad_cache_k, self.g[Self.GCK], ctx
            )
            _load_cache_grad[target, CACHE_LAYER](
                i, grad_cache_v, self.g[Self.GCV], ctx
            )
            accum_into[target, KVN](self.g[Self.GKR], self.g[Self.GCK], ctx)
            accum_into[target, KVN](self.g[Self.GV], self.g[Self.GCV], ctx)

            self.rope_k.vjp[target, Self.B](
                TensorRefs[1](PK[Self.K]), self.g[Self.GKR],
                TensorRefs[1](self.g[Self.GK]), ctx,
            )
            self.rope_q.vjp[target, Self.B](
                TensorRefs[1](PK[Self.Q]), self.g[Self.GQR],
                TensorRefs[1](self.g[Self.GQ]), ctx,
            )
            tower.layers[i].proj_vjp[LORA_Q, target, TOK](
                TensorRefs[1](PK[Self.H]), self.g[Self.GQ],
                TensorRefs[1](self.g[Self.GHA]), ctx,
            )
            tower.layers[i].proj_vjp[LORA_K, target, TOK](
                TensorRefs[1](PK[Self.H]), self.g[Self.GK],
                TensorRefs[1](self.g[Self.GHB]), ctx,
            )
            tower.layers[i].proj_vjp[LORA_V, target, TOK](
                TensorRefs[1](PK[Self.H]), self.g[Self.GV],
                TensorRefs[1](self.g[Self.GXO]), ctx,
            )
            accum_into[target, XN](self.g[Self.GHA], self.g[Self.GHB], ctx)
            accum_into[target, XN](self.g[Self.GHA], self.g[Self.GXO], ctx)
            tower.layers[i].input_layernorm.vjp[target, TOK](
                TensorRefs[1](PK[Self.X]), self.g[Self.GHA],
                TensorRefs[1](self.g[Self.GHB]), ctx,
            )
            accum_into[target, XN](self.g[Self.GHC], self.g[Self.GHB], ctx)
            copy_into[target, XN](self.g[Self.GXO], self.g[Self.GHC], ctx)

        copy_into[target, XN](grad_x, self.g[Self.GXO], ctx)
        _ = FFN


def _store_cache_grad[
    target: StaticString, LAYER_N: Int
](
    layer: Int, mut src: Tensor, mut dst: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Write one layer's `[B*P*KVW]` cache gradient into its slot.

    ⚠ WRITE, not accumulate. Each layer contributes to its own slot only, and
    an accumulate here would silently sum across the ten denoising steps of an
    inference-shaped call.
    """
    var off = layer * LAYER_N
    comptime if target == "cpu":
        for i in range(LAYER_N):
            dst.data[off + i] = src.data[i]
    else:
        var c = ctx.value()
        c.enqueue_copy(
            dst.dev.value().create_sub_buffer[DT](off, LAYER_N),
            src.dev.value().create_sub_buffer[DT](0, LAYER_N),
        )


@always_inline
def _prof_tick[
    target: StaticString
](
    profile: Bool, mut prof: List[Int], slot: Int, mut t: Int,
    ctx: Optional[DeviceContext],
) raises:
    """Charge the time since `t` to `prof[slot]`, after draining the device,
    and restart `t`. A no-op unless `profile` is on.

    A free function, not a method: `backward` holds `ref PK = self.pools[i]`
    across a whole layer, and a `mut self` call there invalidates that
    borrow. Borrowing `self.prof` alone leaves `pools` untouched."""
    if not profile:
        return
    comptime if target != "cpu":
        ctx.value().synchronize()
    var now = Int(perf_counter_ns())
    prof[slot] += now - t
    t = now


struct SmolVLADenoise[
    P: Int,
    S: Int,
    B: Int = 1,
    LAYERS: Int = SMOLLM_LAYERS,
    EW: Int = 720,
    EFF: Int = 2048,
    W: Int = SMOLLM_DIM,
    HEADS: Int = SMOLLM_HEADS,
    N_KV: Int = SMOLLM_KV_HEADS,
    HD: Int = SMOLLM_HEAD_DIM,
    THETA: Float64 = SMOLLM_THETA,
    SELF_EVERY: Int = 2,
    # A parameter, not `N_KV * HD`: Mojo does not unify `5 * (960 // 15)` with
    # the same value written another way, so every container in the loop must
    # name the SAME comptime expression. `SMOLLM_KV_W` is that one definition.
    KVW: Int = SMOLLM_KV_W,
    # V2. False is V1's driver, unchanged: one scratch pool, reused by all
    # sixteen layers, nothing kept. True keeps a pool PER LAYER, so the
    # activations a backward pass needs survive the forward that made them.
    RECORD: Bool = False,
    # The expert's LoRA rank (`layer_weights.mojo`); 0 = no adapters, and
    # every `lora_fwd` / `lora_vjp` below compiles to nothing.
    LORA_R: Int = 0,
](Movable):
    """One denoising step: the 50-token action suffix through the 16 expert
    layers, reading the prefix K/V the prefill left behind.

    Mode 3 of the three (`[None, suffix]`, cache present). Per layer:

        even i  SELF : k,v from the suffix's own stream; attend over
                       [prefix; suffix] with the block mask
        odd  i  CROSS: k,v are the VLM's CACHED prefix K/V pushed through the
                       expert's `[320, 320]` projections; attend over the prefix,
                       unmasked

    ⚠ **The two kinds rotate q from different origins.** A self layer uses
    absolute positions (the suffix sits after the prefix, so `POS_OFFSET = P`);
    a cross layer re-bases to zero — the reference does
    `expert_position_id -= min(expert_position_id)` before `apply_rope`, while
    the keys it attends to were rotated with their absolute prefix positions.
    Asymmetric, easy to miss, and shape-invisible if you get it wrong.

    ⚠ **A cross layer does NOT touch the cache.** Only the self layers extend
    it, and here they extend a *scratch copy* rather than the cache itself, so
    the reference's `crop(prefix_len)` has no counterpart and ten steps in a row
    see an identical prefix (`kv_cache.mojo`).

    Forward only, like `SmolVLAPrefill`.
    """

    comptime REP: Int = Self.HEADS // Self.N_KV
    comptime FULL: Int = Self.P + Self.S
    comptime XN: Int = Self.S * Self.EW      # the suffix stream, 720 wide
    comptime QN: Int = Self.S * Self.W       # queries, in the VLM's 960 geometry
    comptime KVN_S: Int = Self.S * Self.KVW
    comptime KVN_P: Int = Self.P * Self.KVW
    comptime KVN_F: Int = Self.FULL * Self.KVW
    comptime KX_F: Int = Self.FULL * Self.W
    comptime KX_P: Int = Self.P * Self.W
    comptime FFN: Int = Self.S * Self.EFF

    # q rotated from P for a self layer, from 0 for a cross layer.
    comptime RoPEQSelf = RoPE[Self.S, Self.HEADS, Self.HD, Self.THETA, Self.P]
    comptime RoPEQCross = RoPE[Self.S, Self.HEADS, Self.HD, Self.THETA, 0]
    comptime RoPEKSelf = RoPE[Self.S, Self.N_KV, Self.HD, Self.THETA, Self.P]
    comptime RepKVFull = RepeatKVHeads[
        Self.FULL, Self.N_KV, Self.REP, Self.HD
    ]
    comptime RepKVPre = RepeatKVHeads[Self.P, Self.N_KV, Self.REP, Self.HD]
    comptime AttnSelf = BlockCrossAttention[
        Self.W, Self.HEADS, Self.S, Self.FULL
    ]
    comptime AttnCross = BlockCrossAttention[
        Self.W, Self.HEADS, Self.S, Self.P
    ]
    comptime Glu = SwiGLU[Self.FFN]
    comptime GluCat = Concat[Self.FFN, Self.FFN]
    comptime Res = Add[Self.XN]

    var rope_q_self: Self.RoPEQSelf
    var rope_q_cross: Self.RoPEQCross
    var rope_k_self: Self.RoPEKSelf
    var rep_full_k: Self.RepKVFull
    var rep_full_v: Self.RepKVFull
    var rep_pre_k: Self.RepKVPre
    var rep_pre_v: Self.RepKVPre
    var attn_self: Self.AttnSelf
    var attn_cross: Self.AttnCross
    var glu: Self.Glu
    var glu_cat: Self.GluCat
    var res: Self.Res

    comptime X = 0
    comptime H = 1
    comptime Q = 2
    comptime QR = 3
    comptime KS = 4
    comptime VS = 5
    comptime KRS = 6
    comptime KXF = 7
    comptime VXF = 8
    comptime KP = 9
    comptime VP = 10
    comptime KXP = 11
    comptime VXP = 12
    comptime ATT = 13
    comptime AO = 14
    comptime X2 = 15
    comptime GATE = 16
    comptime UP = 17
    comptime CAT = 18
    comptime GLU = 19
    comptime DOWN = 20
    comptime N_BASE = 21
    # ⚠ Not a slot of its own when RECORD is off: the layer's output slot IS
    # its input slot, which is what V1 does today (the second residual writes
    # over X). Aliasing them costs the recording nothing and keeps inference
    # free of the extra slab AND of the copy below — the residual keeps
    # writing straight into X. One call site, one destination expression.
    comptime XO: Int = Self.N_BASE if Self.RECORD else Self.X
    # ⚠ Same reason as XO, for the SECOND layernorm's output. The forward
    # writes both norms into `H`, so without this split the tape would hold
    # the MLP norm's output where the attention norm's belonged, and every
    # q/k/v weight gradient in all sixteen layers would be formed against the
    # wrong input — finite, plausible, wrong.
    comptime H2: Int = Self.N_BASE + 1 if Self.RECORD else Self.H
    comptime N_SLOTS: Int = (
        Self.N_BASE + 2 if Self.RECORD else Self.N_BASE
    )
    # Pack `l` holds layer `l`'s activations, and pack LAYERS the value the
    # final norm consumes. Off, the list is one pack and every layer reuses
    # it — byte-for-byte V1.
    comptime N_POOLS: Int = Self.LAYERS + 1 if Self.RECORD else 1
    comptime LAST: Int = Self.LAYERS if Self.RECORD else 0

    var pools: List[TensorPack[Self.N_SLOTS]]

    # ── the backward's own scratch ───────────────────────────────────────
    # ONE pack, reused by every layer — unlike `pools`, a backward consumes
    # each layer's gradients before reaching the next, so nothing here has to
    # survive an iteration. The slot count is large because `Module.vjp`
    # ASSIGNS its grad_inputs: an activation with three consumers needs three
    # destinations and an explicit sum, not one shared slab that would
    # silently keep only the last writer.
    comptime GXO = 0        # running dL/d(layer output)   [B, S*EW]
    comptime GX2 = 1
    comptime GDOWN = 2
    comptime GGLU = 3       # [B, S*EFF]
    comptime GCAT = 4       # [B, 2*S*EFF]
    comptime GUP = 5
    comptime GGATE = 6
    comptime GHA = 7        # the contributions to dH, summed by hand
    comptime GHB = 8
    comptime GHC = 9
    comptime GAO = 10
    comptime GATT = 11      # [B, S*W]
    comptime GQR = 12
    comptime GQ = 13
    comptime GKXF = 14      # [B, FULL*W]    self
    comptime GVXF = 15
    comptime GKXP = 16      # [B, P*W]       cross
    comptime GVXP = 17
    comptime GSK = 18       # [B, FULL*KVW]
    comptime GSV = 19
    comptime GKRS = 20      # [B, S*KVW]
    comptime GKS = 21
    comptime GVS = 22
    comptime GKSP = 23      # [B, P*KVW]     cross
    comptime GVSP = 24
    comptime GKP = 25       # dL/d(cached prefix K) — see `backward`
    comptime GVP = 26
    comptime GRC = 27       # SwiGLU cache-refill scratch, see `backward`
    comptime G_SLOTS = 28

    var g: TensorPack[Self.G_SLOTS]

    # ── the backward's stage profile ─────────────────────────────────────
    # ⚠ OFF by default. On, `backward` DRAINS the device at every stage
    # boundary (about ten per layer) so each slot is the wall time of that
    # stage and not of its enqueues. The drains themselves cost ~1-2 ms per
    # backward, which is why `profile` is a flag and the numbers are read as
    # relative shares, not as the un-profiled step's absolute cost.
    comptime PR_GLUE = 0      # residual vjps, hand sums, copies, tails, stores
    comptime PR_MLP_DOWN = 1  # mlp.down.vjp        [S x EFF] -> [S x EW]
    comptime PR_GLU = 2       # SwiGLU refill + vjp, concat vjp
    comptime PR_MLP_UPGATE = 3  # mlp.up.vjp + mlp.gate.vjp
    comptime PR_NORM = 4      # both RMSNorm vjps per layer, plus the final norm's
    comptime PR_O = 5         # o.vjp               [S x W] -> [S x EW]
    comptime PR_ATTN = 6      # BlockCrossAttention.vjp (self and cross)
    comptime PR_REP = 7       # RepeatKVHeads.vjp, scratch rebuild
    comptime PR_ROPE = 8      # RoPE vjps
    comptime PR_QKV = 9       # q/k/v.vjp (self: 3 at S; cross: q at S, k/v at P)
    comptime PR_N = 10

    var profile: Bool
    var prof: List[Int]
    """Nanoseconds per stage, summed over every `backward` since the last
    `reset_profile`. Read with the caller's own observation count."""

    # ── the forward's stage profile, same slots ──────────────────────────
    # A SEPARATE flag and list, not `profile`/`prof`, on purpose: the
    # fine-tune turns `profile` on to split its backward by op, and reads its
    # "suffix fwd" phase from timers around the whole step. Draining inside
    # the step would inflate that phase and leak the forward's ticks into the
    # backward's table. `profile_step` is for the deploy-side question — WHY
    # is one denoising step 28 ms on the Orin — and is read by
    # `benchmarks/smolvla_denoise_stages_bench.mojo`.
    var profile_step: Bool
    var prof_step: List[Int]
    """Nanoseconds per stage, summed over every `step` since the last
    `reset_profile`. The scratch build / cache read are charged to PR_REP,
    the residual adds and the input copy to PR_GLUE."""

    def __init__(out self):
        self.rope_q_self = Self.RoPEQSelf()
        self.rope_q_cross = Self.RoPEQCross()
        self.rope_k_self = Self.RoPEKSelf()
        self.rep_full_k = Self.RepKVFull()
        self.rep_full_v = Self.RepKVFull()
        self.rep_pre_k = Self.RepKVPre()
        self.rep_pre_v = Self.RepKVPre()
        self.attn_self = Self.AttnSelf()
        self.attn_cross = Self.AttnCross()
        self.glu = Self.Glu()
        self.glu_cat = Self.GluCat()
        self.res = Self.Res()
        self.pools = List[TensorPack[Self.N_SLOTS]]()
        for _ in range(Self.N_POOLS):
            self.pools.append(TensorPack[Self.N_SLOTS]())
        self.g = TensorPack[Self.G_SLOTS]()
        self.profile = False
        self.prof = List[Int](length=Self.PR_N, fill=0)
        self.profile_step = False
        self.prof_step = List[Int](length=Self.PR_N, fill=0)

    def __init__(out self, *, deinit move: Self):
        self.rope_q_self = move.rope_q_self^
        self.rope_q_cross = move.rope_q_cross^
        self.rope_k_self = move.rope_k_self^
        self.rep_full_k = move.rep_full_k^
        self.rep_full_v = move.rep_full_v^
        self.rep_pre_k = move.rep_pre_k^
        self.rep_pre_v = move.rep_pre_v^
        self.attn_self = move.attn_self^
        self.attn_cross = move.attn_cross^
        self.glu = move.glu^
        self.glu_cat = move.glu_cat^
        self.res = move.res^
        self.pools = move.pools^
        self.g = move.g^
        self.profile = move.profile
        self.prof = move.prof^
        self.profile_step = move.profile_step
        self.prof_step = move.prof_step^

    def reset_profile(mut self):
        for i in range(Self.PR_N):
            self.prof[i] = 0
            self.prof_step[i] = 0

    @staticmethod
    def make[
        target: StaticString
    ](
        ref self_mask: List[Scalar[DT]], ref cross_mask: List[Scalar[DT]],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """`self_mask` is `[S, P+S]`, `cross_mask` is `[S, P]` — both from
        `attn_mask.att_2d_mask` over the same `ar`, so they cannot disagree."""
        var d = Self()
        d.rope_q_self = Self.RoPEQSelf.make[target, Deterministic](ctx)
        d.rope_q_cross = Self.RoPEQCross.make[target, Deterministic](ctx)
        d.rope_k_self = Self.RoPEKSelf.make[target, Deterministic](ctx)
        d.rep_full_k = Self.RepKVFull.make[target, Deterministic](ctx)
        d.rep_full_v = Self.RepKVFull.make[target, Deterministic](ctx)
        d.rep_pre_k = Self.RepKVPre.make[target, Deterministic](ctx)
        d.rep_pre_v = Self.RepKVPre.make[target, Deterministic](ctx)
        d.attn_self = Self.AttnSelf.make[target](self_mask, ctx)
        d.attn_cross = Self.AttnCross.make[target](cross_mask, ctx)
        d.glu = Self.Glu.make[target, Deterministic](ctx)
        d.glu_cat = Self.GluCat.make[target, Deterministic](ctx)
        d.res = Self.Res.make[target, Deterministic](ctx)
        return d^

    def step[
        target: StaticString
    ](
        mut self,
        mut expert: SmolVLAExpert[
            Self.LAYERS, Self.EW, Self.EFF, Self.W, Self.KVW, Self.SELF_EVERY,
            Self.LORA_R,
        ],
        mut cache: SmolVLAKVCache[
            Self.LAYERS, Self.P, Self.S, Self.N_KV, Self.HD, Self.B
        ],
        mut x: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`x` is `[B, S*EW]` suffix embeddings; `out` receives the final norm."""
        comptime TOK_S = Self.B * Self.S
        comptime TOK_P = Self.B * Self.P
        comptime TOK_F = Self.B * Self.FULL
        comptime XN = Self.B * Self.XN
        var t = Int(perf_counter_ns())

        comptime if target == "cpu":
            self.pools[0][Self.X].ensure(XN)
            for i in range(XN):
                self.pools[0][Self.X].data[i] = x.data[i]
        else:
            var c = ctx.value()
            self.pools[0][Self.X].ensure_gpu(c, XN)
            c.enqueue_copy(
                self.pools[0][Self.X].dev.value().create_sub_buffer[DT](0, XN),
                x.dev.value().create_sub_buffer[DT](0, XN),
            )
        _prof_tick[target](
            self.profile_step, self.prof_step, Self.PR_GLUE, t, ctx
        )

        for i in range(Self.LAYERS):
            var is_self = (i % Self.SELF_EVERY) == 0
            var li = i // Self.SELF_EVERY
            # ⚠ ONE borrow of `pools`, held for the whole layer body. Taking a
            # second (`self.pools[pi + 1]`) alongside it is two mutable borrows
            # of the same list and Mojo rejects it — which is why the hand-off
            # below goes through raw pointers / sub-buffers instead.
            var pi = i if Self.RECORD else 0
            ref PK = self.pools[pi]

            # ── attention branch ─────────────────────────────────────────
            if is_self:
                ref L = expert.self_layers[li]
                L.input_layernorm.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.X]), PK[Self.H], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_NORM, t, ctx
                )
                L.q.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.H]), PK[Self.Q], ctx
                )
                L.lora_fwd[LORA_Q, target, TOK_S](
                    TensorRefs[1](PK[Self.H]), PK[Self.Q], ctx
                )
                L.k.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.H]), PK[Self.KS], ctx
                )
                L.lora_fwd[LORA_K, target, TOK_S](
                    TensorRefs[1](PK[Self.H]), PK[Self.KS], ctx
                )
                L.v.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.H]), PK[Self.VS], ctx
                )
                L.lora_fwd[LORA_V, target, TOK_S](
                    TensorRefs[1](PK[Self.H]), PK[Self.VS], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_QKV, t, ctx
                )
                # absolute positions: the suffix sits after the prefix
                self.rope_q_self.forward[target, Self.B](
                    TensorRefs[1](PK[Self.Q]), PK[Self.QR], ctx
                )
                self.rope_k_self.forward[target, Self.B](
                    TensorRefs[1](PK[Self.KS]), PK[Self.KRS], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_ROPE, t, ctx
                )
                # [prefix; suffix] into SCRATCH — the cache is not touched.
                cache.build_scratch[target](
                    i, PK[Self.KRS], PK[Self.VS], ctx
                )
                self.rep_full_k.forward[target, Self.B](
                    TensorRefs[1](cache.sk), PK[Self.KXF], ctx
                )
                self.rep_full_v.forward[target, Self.B](
                    TensorRefs[1](cache.sv), PK[Self.VXF], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_REP, t, ctx
                )
                self.attn_self.forward[target, Self.B](
                    PK[Self.QR], PK[Self.KXF],
                    PK[Self.VXF], PK[Self.ATT], ctx,
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_ATTN, t, ctx
                )
                L.o.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.ATT]), PK[Self.AO], ctx
                )
                L.lora_fwd[LORA_O, target, TOK_S](
                    TensorRefs[1](PK[Self.ATT]), PK[Self.AO], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_O, t, ctx
                )
            else:
                ref L = expert.cross_layers[li]
                L.input_layernorm.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.X]), PK[Self.H], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_NORM, t, ctx
                )
                L.q.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.H]), PK[Self.Q], ctx
                )
                L.lora_fwd[LORA_Q, target, TOK_S](
                    TensorRefs[1](PK[Self.H]), PK[Self.Q], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_QKV, t, ctx
                )
                # ⚠ re-based to 0, unlike the self layers
                self.rope_q_cross.forward[target, Self.B](
                    TensorRefs[1](PK[Self.Q]), PK[Self.QR], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_ROPE, t, ctx
                )
                # k/v are the VLM's CACHED prefix K/V through [320,320] projs
                cache.read_layer_into[target](
                    i, PK[Self.KP], PK[Self.VP], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_REP, t, ctx
                )
                L.k.forward[target, TOK_P](
                    TensorRefs[1](PK[Self.KP]), PK[Self.KS], ctx
                )
                L.lora_fwd[LORA_K, target, TOK_P](
                    TensorRefs[1](PK[Self.KP]), PK[Self.KS], ctx
                )
                L.v.forward[target, TOK_P](
                    TensorRefs[1](PK[Self.VP]), PK[Self.VS], ctx
                )
                L.lora_fwd[LORA_V, target, TOK_P](
                    TensorRefs[1](PK[Self.VP]), PK[Self.VS], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_QKV, t, ctx
                )
                self.rep_pre_k.forward[target, Self.B](
                    TensorRefs[1](PK[Self.KS]), PK[Self.KXP], ctx
                )
                self.rep_pre_v.forward[target, Self.B](
                    TensorRefs[1](PK[Self.VS]), PK[Self.VXP], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_REP, t, ctx
                )
                self.attn_cross.forward[target, Self.B](
                    PK[Self.QR], PK[Self.KXP],
                    PK[Self.VXP], PK[Self.ATT], ctx,
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_ATTN, t, ctx
                )
                L.o.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.ATT]), PK[Self.AO], ctx
                )
                L.lora_fwd[LORA_O, target, TOK_S](
                    TensorRefs[1](PK[Self.ATT]), PK[Self.AO], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_O, t, ctx
                )

            self.res.forward[target, Self.B](
                TensorRefs[2](PK[Self.X], PK[Self.AO]),
                PK[Self.X2], ctx,
            )
            _prof_tick[target](
                self.profile_step, self.prof_step, Self.PR_GLUE, t, ctx
            )

            # ── MLP branch (identical for both kinds) ────────────────────
            if is_self:
                ref L = expert.self_layers[li]
                L.post_attention_layernorm.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.X2]), PK[Self.H2], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_NORM, t, ctx
                )
                L.mlp.gate.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), PK[Self.GATE], ctx
                )
                L.lora_fwd[LORA_GATE, target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), PK[Self.GATE], ctx
                )
                L.mlp.up.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), PK[Self.UP], ctx
                )
                L.lora_fwd[LORA_UP, target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), PK[Self.UP], ctx
                )
            else:
                ref L = expert.cross_layers[li]
                L.post_attention_layernorm.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.X2]), PK[Self.H2], ctx
                )
                _prof_tick[target](
                    self.profile_step, self.prof_step, Self.PR_NORM, t, ctx
                )
                L.mlp.gate.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), PK[Self.GATE], ctx
                )
                L.lora_fwd[LORA_GATE, target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), PK[Self.GATE], ctx
                )
                L.mlp.up.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), PK[Self.UP], ctx
                )
                L.lora_fwd[LORA_UP, target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), PK[Self.UP], ctx
                )
            _prof_tick[target](
                self.profile_step, self.prof_step, Self.PR_MLP_UPGATE, t, ctx
            )
            self.glu_cat.forward[target, Self.B](
                TensorRefs[2](PK[Self.UP], PK[Self.GATE]),
                PK[Self.CAT], ctx,
            )
            self.glu.forward[target, Self.B](
                TensorRefs[1](PK[Self.CAT]), PK[Self.GLU], ctx
            )
            _prof_tick[target](
                self.profile_step, self.prof_step, Self.PR_GLU, t, ctx
            )
            if is_self:
                expert.self_layers[li].mlp.down.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.GLU]), PK[Self.DOWN],
                    ctx,
                )
                expert.self_layers[li].lora_fwd[LORA_DOWN, target, TOK_S](
                    TensorRefs[1](PK[Self.GLU]), PK[Self.DOWN],
                    ctx,
                )
            else:
                expert.cross_layers[li].mlp.down.forward[target, TOK_S](
                    TensorRefs[1](PK[Self.GLU]), PK[Self.DOWN],
                    ctx,
                )
                expert.cross_layers[li].lora_fwd[LORA_DOWN, target, TOK_S](
                    TensorRefs[1](PK[Self.GLU]), PK[Self.DOWN],
                    ctx,
                )
            _prof_tick[target](
                self.profile_step, self.prof_step, Self.PR_MLP_DOWN, t, ctx
            )
            self.res.forward[target, Self.B](
                TensorRefs[2](PK[Self.X2], PK[Self.DOWN]),
                PK[Self.XO], ctx,
            )
            _prof_tick[target](
                self.profile_step, self.prof_step, Self.PR_GLUE, t, ctx
            )

            # Hand the running activation to the next layer's pack. Comptime-
            # dead when RECORD is off, where XO IS X and there is nowhere to
            # hand it to — V1 pays neither the slab nor the copy.
            comptime if Self.RECORD:
                var pn = i + 1
                comptime if target == "cpu":
                    self.pools[pn][Self.X].ensure(XN)
                    var dst = self.pools[pn][Self.X].data.unsafe_ptr()
                    var src = self.pools[pi][Self.XO].data.unsafe_ptr()
                    for t in range(XN):
                        dst[unsafe_offset=t] = src[unsafe_offset=t]
                else:
                    var c2 = ctx.value()
                    self.pools[pn][Self.X].ensure_gpu(c2, XN)
                    var sb = self.pools[pi][Self.XO].dev.value(
                    ).create_sub_buffer[DT](0, XN)
                    var db = self.pools[pn][Self.X].dev.value(
                    ).create_sub_buffer[DT](0, XN)
                    c2.enqueue_copy(db, sb)

        # The RECORD hand-off copy above lands in whichever slot follows it;
        # it is comptime-dead on the deploy path this profile is for.
        expert.norm.forward[target, TOK_S](
            TensorRefs[1](self.pools[Self.LAST][Self.X]), out, ctx
        )
        _prof_tick[target](
            self.profile_step, self.prof_step, Self.PR_NORM, t, ctx
        )

    def backward[
        target: StaticString
    ](
        mut self,
        mut expert: SmolVLAExpert[
            Self.LAYERS, Self.EW, Self.EFF, Self.W, Self.KVW, Self.SELF_EVERY,
            Self.LORA_R,
        ],
        mut cache: SmolVLAKVCache[
            Self.LAYERS, Self.P, Self.S, Self.N_KV, Self.HD, Self.B
        ],
        mut grad_out: Tensor,
        mut grad_x: Tensor,
        mut grad_cache_k: Tensor,
        mut grad_cache_v: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Reverse of `step`, over the tape `step[RECORD=True]` left behind.

        Accumulates into every expert `Param.grd` (the `nn` convention — the
        caller zeroes) and writes `grad_x`, dL/d(suffix embeddings), which the
        action projections need.

        ⚠ **Call `step` with the SAME instance first.** The tape is this
        object's `pools`. A `backward` on a stale tape reads a previous
        observation's activations and returns a finite, plausible, wrong
        gradient.

        `grad_cache_k` / `grad_cache_v` receive dL/d(the cached prefix K/V),
        laid out exactly like the cache itself — `[LAYERS * B * P * KVW]`.
        That is the whole gradient path into the VLM and therefore into
        `state_proj`. A caller running `train_state_proj = False` passes a
        scratch tensor and ignores it.

        ⚠ **Each layer contributes to its OWN cache slot and no other**, so
        these are written, not accumulated. A SELF layer's contribution is the
        PREFIX ROWS of the scratch gradient (`prefix_head`, the complement of
        the `suffix_tail` that carries the expert's own K/V); a CROSS layer's
        is what its `[320, 320]` projections push back. The two halves of a
        self layer's scratch must go to exactly one destination each — a
        gradient counted in both is doubled, one counted in neither is
        silently dropped.
        """
        comptime assert Self.RECORD, (
            "SmolVLADenoise.backward needs RECORD=True — the non-recording"
            " driver keeps one pool and its tape is the last layer's"
            " activations sixteen times over"
        )
        comptime TOK_S = Self.B * Self.S
        comptime TOK_P = Self.B * Self.P
        comptime XN = Self.B * Self.XN
        comptime QN = Self.B * Self.QN
        comptime CN = Self.B * Self.FFN
        comptime KVN_S = Self.B * Self.KVN_S
        comptime KVN_P = Self.B * Self.KVN_P
        comptime CACHE_LAYER = Self.B * Self.P * Self.KVW

        comptime if target == "cpu":
            grad_cache_k.ensure(Self.LAYERS * CACHE_LAYER)
            grad_cache_v.ensure(Self.LAYERS * CACHE_LAYER)
        else:
            grad_cache_k.ensure_gpu(
                ctx.value(), Self.LAYERS * CACHE_LAYER
            )
            grad_cache_v.ensure_gpu(
                ctx.value(), Self.LAYERS * CACHE_LAYER
            )

        var t = 0
        if self.profile:
            comptime if target != "cpu":
                ctx.value().synchronize()
            t = Int(perf_counter_ns())

        # out = norm(pools[LAST][X])
        expert.norm.vjp[target, TOK_S](
            TensorRefs[1](self.pools[Self.LAST][Self.X]),
            grad_out,
            TensorRefs[1](self.g[Self.GXO]),
            ctx,
        )
        _prof_tick[target](self.profile, self.prof, Self.PR_NORM, t, ctx)

        for ridx in range(Self.LAYERS):
            var i = Self.LAYERS - 1 - ridx
            var is_self = (i % Self.SELF_EVERY) == 0
            var li = i // Self.SELF_EVERY
            ref PK = self.pools[i]

            # ── MLP branch, reversed ─────────────────────────────────────
            # XO = X2 + DOWN
            self.res.vjp[target, Self.B](
                TensorRefs[2](PK[Self.X2], PK[Self.DOWN]),
                self.g[Self.GXO],
                TensorRefs[2](self.g[Self.GX2], self.g[Self.GDOWN]),
                ctx,
            )
            _prof_tick[target](self.profile, self.prof, Self.PR_GLUE, t, ctx)
            if is_self:
                expert.self_layers[li].mlp.down.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.GLU]), self.g[Self.GDOWN],
                    TensorRefs[1](self.g[Self.GGLU]), ctx,
                )
                expert.self_layers[li].lora_vjp[LORA_DOWN, target, TOK_S](
                    TensorRefs[1](PK[Self.GLU]), self.g[Self.GDOWN],
                    TensorRefs[1](self.g[Self.GGLU]), ctx,
                )
            else:
                expert.cross_layers[li].mlp.down.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.GLU]), self.g[Self.GDOWN],
                    TensorRefs[1](self.g[Self.GGLU]), ctx,
                )
                expert.cross_layers[li].lora_vjp[LORA_DOWN, target, TOK_S](
                    TensorRefs[1](PK[Self.GLU]), self.g[Self.GDOWN],
                    TensorRefs[1](self.g[Self.GGLU]), ctx,
                )
            _prof_tick[target](self.profile, self.prof, Self.PR_MLP_DOWN, t, ctx)

            # ⚠ SwiGLU is OUTPUT-CACHING and there is ONE instance for all
            # sixteen layers, so by now `self.glu`'s cache holds layer 15's
            # values and its `vjp` — which ignores `forward_input` entirely —
            # would differentiate every layer at layer 15's point. Re-running
            # the forward on this layer's CAT refills the cache with this
            # layer's values. The output goes to scratch, NOT back over the
            # tape, so the tape stays the forward's own record.
            self.glu.forward[target, Self.B](
                TensorRefs[1](PK[Self.CAT]), self.g[Self.GRC], ctx
            )
            self.glu.vjp[target, Self.B](
                TensorRefs[1](PK[Self.CAT]), self.g[Self.GGLU],
                TensorRefs[1](self.g[Self.GCAT]), ctx,
            )
            self.glu_cat.vjp[target, Self.B](
                TensorRefs[2](PK[Self.UP], PK[Self.GATE]),
                self.g[Self.GCAT],
                TensorRefs[2](self.g[Self.GUP], self.g[Self.GGATE]),
                ctx,
            )
            _prof_tick[target](self.profile, self.prof, Self.PR_GLU, t, ctx)
            if is_self:
                ref L = expert.self_layers[li]
                L.mlp.up.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), self.g[Self.GUP],
                    TensorRefs[1](self.g[Self.GHA]), ctx,
                )
                L.lora_vjp[LORA_UP, target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), self.g[Self.GUP],
                    TensorRefs[1](self.g[Self.GHA]), ctx,
                )
                L.mlp.gate.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), self.g[Self.GGATE],
                    TensorRefs[1](self.g[Self.GHB]), ctx,
                )
                L.lora_vjp[LORA_GATE, target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), self.g[Self.GGATE],
                    TensorRefs[1](self.g[Self.GHB]), ctx,
                )
            else:
                ref L = expert.cross_layers[li]
                L.mlp.up.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), self.g[Self.GUP],
                    TensorRefs[1](self.g[Self.GHA]), ctx,
                )
                L.lora_vjp[LORA_UP, target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), self.g[Self.GUP],
                    TensorRefs[1](self.g[Self.GHA]), ctx,
                )
                L.mlp.gate.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), self.g[Self.GGATE],
                    TensorRefs[1](self.g[Self.GHB]), ctx,
                )
                L.lora_vjp[LORA_GATE, target, TOK_S](
                    TensorRefs[1](PK[Self.H2]), self.g[Self.GGATE],
                    TensorRefs[1](self.g[Self.GHB]), ctx,
                )
            _prof_tick[target](self.profile, self.prof, Self.PR_MLP_UPGATE, t, ctx)
            # dH2 = up's + gate's — both read H2, so both contribute.
            accum_into[target, XN](self.g[Self.GHA], self.g[Self.GHB], ctx)
            _prof_tick[target](self.profile, self.prof, Self.PR_GLUE, t, ctx)
            if is_self:
                expert.self_layers[li].post_attention_layernorm.vjp[
                    target, TOK_S
                ](
                    TensorRefs[1](PK[Self.X2]), self.g[Self.GHA],
                    TensorRefs[1](self.g[Self.GHC]), ctx,
                )
            else:
                expert.cross_layers[li].post_attention_layernorm.vjp[
                    target, TOK_S
                ](
                    TensorRefs[1](PK[Self.X2]), self.g[Self.GHA],
                    TensorRefs[1](self.g[Self.GHC]), ctx,
                )
            _prof_tick[target](self.profile, self.prof, Self.PR_NORM, t, ctx)
            # X2 feeds the residual AND the norm.
            accum_into[target, XN](self.g[Self.GX2], self.g[Self.GHC], ctx)

            # ── attention branch, reversed ───────────────────────────────
            # X2 = X + AO
            self.res.vjp[target, Self.B](
                TensorRefs[2](PK[Self.X], PK[Self.AO]),
                self.g[Self.GX2],
                TensorRefs[2](self.g[Self.GHC], self.g[Self.GAO]),
                ctx,
            )
            _prof_tick[target](self.profile, self.prof, Self.PR_GLUE, t, ctx)
            if is_self:
                ref L = expert.self_layers[li]
                L.o.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.ATT]), self.g[Self.GAO],
                    TensorRefs[1](self.g[Self.GATT]), ctx,
                )
                L.lora_vjp[LORA_O, target, TOK_S](
                    TensorRefs[1](PK[Self.ATT]), self.g[Self.GAO],
                    TensorRefs[1](self.g[Self.GATT]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_O, t, ctx)
                self.attn_self.vjp[target, Self.B](
                    PK[Self.QR], PK[Self.KXF], PK[Self.VXF],
                    self.g[Self.GATT],
                    self.g[Self.GQR], self.g[Self.GKXF], self.g[Self.GVXF],
                    ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_ATTN, t, ctx)
                # ⚠ Rebuild this layer's [prefix; suffix] from the tape. The
                # cache's own scratch holds the LAST self layer's, and the
                # repeat's `vjp` takes a forward input.
                cache.build_scratch[target](
                    i, PK[Self.KRS], PK[Self.VS], ctx
                )
                self.rep_full_k.vjp[target, Self.B](
                    TensorRefs[1](cache.sk), self.g[Self.GKXF],
                    TensorRefs[1](self.g[Self.GSK]), ctx,
                )
                self.rep_full_v.vjp[target, Self.B](
                    TensorRefs[1](cache.sv), self.g[Self.GVXF],
                    TensorRefs[1](self.g[Self.GSV]), ctx,
                )
                # The prefix rows of GSK/GSV are dL/d(cached K/V) — the VLM's
                # gradient, discarded in this regime. Take the suffix rows.
                suffix_tail[
                    target, Self.B, Self.FULL * Self.KVW,
                    Self.P * Self.KVW, Self.S * Self.KVW,
                ](self.g[Self.GSK], self.g[Self.GKRS], ctx)
                suffix_tail[
                    target, Self.B, Self.FULL * Self.KVW,
                    Self.P * Self.KVW, Self.S * Self.KVW,
                ](self.g[Self.GSV], self.g[Self.GVS], ctx)
                # ⚠ The OTHER half of the same slab: the prefix rows are the
                # VLM's gradient. Extracted here so the two halves of GSK/GSV
                # are each consumed exactly once.
                prefix_head[
                    target, Self.B, Self.FULL * Self.KVW, Self.P * Self.KVW
                ](self.g[Self.GSK], self.g[Self.GKP], ctx)
                prefix_head[
                    target, Self.B, Self.FULL * Self.KVW, Self.P * Self.KVW
                ](self.g[Self.GSV], self.g[Self.GVP], ctx)
                _store_cache_grad[target, CACHE_LAYER](
                    i, self.g[Self.GKP], grad_cache_k, ctx
                )
                _store_cache_grad[target, CACHE_LAYER](
                    i, self.g[Self.GVP], grad_cache_v, ctx
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_REP, t, ctx)
                self.rope_k_self.vjp[target, Self.B](
                    TensorRefs[1](PK[Self.KS]), self.g[Self.GKRS],
                    TensorRefs[1](self.g[Self.GKS]), ctx,
                )
                self.rope_q_self.vjp[target, Self.B](
                    TensorRefs[1](PK[Self.Q]), self.g[Self.GQR],
                    TensorRefs[1](self.g[Self.GQ]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_ROPE, t, ctx)
                L.q.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.H]), self.g[Self.GQ],
                    TensorRefs[1](self.g[Self.GHA]), ctx,
                )
                L.lora_vjp[LORA_Q, target, TOK_S](
                    TensorRefs[1](PK[Self.H]), self.g[Self.GQ],
                    TensorRefs[1](self.g[Self.GHA]), ctx,
                )
                L.k.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.H]), self.g[Self.GKS],
                    TensorRefs[1](self.g[Self.GHB]), ctx,
                )
                L.lora_vjp[LORA_K, target, TOK_S](
                    TensorRefs[1](PK[Self.H]), self.g[Self.GKS],
                    TensorRefs[1](self.g[Self.GHB]), ctx,
                )
                L.v.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.H]), self.g[Self.GVS],
                    TensorRefs[1](self.g[Self.GXO]), ctx,
                )
                L.lora_vjp[LORA_V, target, TOK_S](
                    TensorRefs[1](PK[Self.H]), self.g[Self.GVS],
                    TensorRefs[1](self.g[Self.GXO]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_QKV, t, ctx)
                # dH = q's + k's + v's — H feeds all three.
                accum_into[target, XN](self.g[Self.GHA], self.g[Self.GHB], ctx)
                accum_into[target, XN](self.g[Self.GHA], self.g[Self.GXO], ctx)
                _prof_tick[target](self.profile, self.prof, Self.PR_GLUE, t, ctx)
                L.input_layernorm.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.X]), self.g[Self.GHA],
                    TensorRefs[1](self.g[Self.GHB]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_NORM, t, ctx)
            else:
                ref L = expert.cross_layers[li]
                L.o.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.ATT]), self.g[Self.GAO],
                    TensorRefs[1](self.g[Self.GATT]), ctx,
                )
                L.lora_vjp[LORA_O, target, TOK_S](
                    TensorRefs[1](PK[Self.ATT]), self.g[Self.GAO],
                    TensorRefs[1](self.g[Self.GATT]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_O, t, ctx)
                self.attn_cross.vjp[target, Self.B](
                    PK[Self.QR], PK[Self.KXP], PK[Self.VXP],
                    self.g[Self.GATT],
                    self.g[Self.GQR], self.g[Self.GKXP], self.g[Self.GVXP],
                    ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_ATTN, t, ctx)
                self.rep_pre_k.vjp[target, Self.B](
                    TensorRefs[1](PK[Self.KS]), self.g[Self.GKXP],
                    TensorRefs[1](self.g[Self.GKSP]), ctx,
                )
                self.rep_pre_v.vjp[target, Self.B](
                    TensorRefs[1](PK[Self.VS]), self.g[Self.GVXP],
                    TensorRefs[1](self.g[Self.GVSP]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_REP, t, ctx)
                # dL/d(cached prefix K/V). Formed because `Linear.vjp` needs a
                # destination, then dropped — see this method's docstring.
                L.k.vjp[target, TOK_P](
                    TensorRefs[1](PK[Self.KP]), self.g[Self.GKSP],
                    TensorRefs[1](self.g[Self.GKP]), ctx,
                )
                L.lora_vjp[LORA_K, target, TOK_P](
                    TensorRefs[1](PK[Self.KP]), self.g[Self.GKSP],
                    TensorRefs[1](self.g[Self.GKP]), ctx,
                )
                L.v.vjp[target, TOK_P](
                    TensorRefs[1](PK[Self.VP]), self.g[Self.GVSP],
                    TensorRefs[1](self.g[Self.GVP]), ctx,
                )
                L.lora_vjp[LORA_V, target, TOK_P](
                    TensorRefs[1](PK[Self.VP]), self.g[Self.GVSP],
                    TensorRefs[1](self.g[Self.GVP]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_QKV, t, ctx)
                _store_cache_grad[target, CACHE_LAYER](
                    i, self.g[Self.GKP], grad_cache_k, ctx
                )
                _store_cache_grad[target, CACHE_LAYER](
                    i, self.g[Self.GVP], grad_cache_v, ctx
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_GLUE, t, ctx)
                self.rope_q_cross.vjp[target, Self.B](
                    TensorRefs[1](PK[Self.Q]), self.g[Self.GQR],
                    TensorRefs[1](self.g[Self.GQ]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_ROPE, t, ctx)
                # ⚠ A cross layer's H feeds q ONLY: k and v come from the
                # cache, not from this stream. No sum here, and adding one
                # would double-count nothing — it would add a stale slab.
                L.q.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.H]), self.g[Self.GQ],
                    TensorRefs[1](self.g[Self.GHA]), ctx,
                )
                L.lora_vjp[LORA_Q, target, TOK_S](
                    TensorRefs[1](PK[Self.H]), self.g[Self.GQ],
                    TensorRefs[1](self.g[Self.GHA]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_QKV, t, ctx)
                L.input_layernorm.vjp[target, TOK_S](
                    TensorRefs[1](PK[Self.X]), self.g[Self.GHA],
                    TensorRefs[1](self.g[Self.GHB]), ctx,
                )
                _prof_tick[target](self.profile, self.prof, Self.PR_NORM, t, ctx)

            # X feeds the residual AND the first norm. This is the next
            # (earlier) layer's dL/d(output).
            accum_into[target, XN](self.g[Self.GHC], self.g[Self.GHB], ctx)
            copy_into[target, XN](self.g[Self.GXO], self.g[Self.GHC], ctx)
            _prof_tick[target](self.profile, self.prof, Self.PR_GLUE, t, ctx)

        copy_into[target, XN](grad_x, self.g[Self.GXO], ctx)
        _prof_tick[target](self.profile, self.prof, Self.PR_GLUE, t, ctx)
        _ = CN
        _ = QN
        _ = KVN_S
        _ = KVN_P
