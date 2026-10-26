"""MaskedAttention[DIM, N_HEADS, SEQ_LEN, USE_MAX_KERNELS] — attention with a
static additive mask (Dreamer 4 modality-gated / block-causal attention), on the
STORAGE surface. Transformed from legacy `nn.primitives.masked_attention`
(surface-only change; all kernels + host mask builders carried VERBATIM).

This is base scaled-dot-product attention + a mask term. The ONLY behavioural
difference vs plain attention:
  1. A per-(i,j) additive bias `mask[i*SEQ+j] ∈ {0.0, NEG}` is added to the score
     before the softmax max-shift. A masked entry gets score ≈ NEG → weight 0.
  2. Loops run the full [0, SEQ) range (no causal loop-bound) — the mask alone
     gates visibility. The backward is byte-for-byte the CAUSAL=False attention
     backward (masked weights are 0 in cache → contribute 0 to every gradient).

The default mask built by `make` is all-zero (all-allow) → bit-identical to
`ScaledDotProductAttention[..., CAUSAL=False]`. Install a real mask with
`set_mask`; `causal_mask`/`build_modality_mask` are host builders.

Output-caching (backward reads only the cache + grad_output). No Params. The
mask is an owned `Tensor` field (CPU `data` + optional GPU buffer); BMM scratch
slabs are separate owned `Tensor`s (one buffer per slab), no `mptr`.

The BMM fast path REUSES the base storage attention's mask-agnostic
pack/unpack/transpose/jvp kernels verbatim; only the softmax adds the mask.
"""

from noeira.nn.core.mm import mm, bmm
from std.math import exp, sqrt
from max.gpu import thread_idx, block_idx, block_dim, global_idx
from max.gpu.host import DeviceContext
from layout import row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP

# The BMM fast path reuses the storage attention's pack/unpack/transpose/jvp
# kernels verbatim (mask-agnostic); only the softmax differs (adds the mask),
# so MaskedAttention defines its own `_masked_softmax_kernel` below.
from .attention import (
    _attn_pack_qkv_fwd_kernel,
    _attn_unpack_out_kernel,
    _attn_pack_in_bwd_kernel,
    _attn_softmax_jvp_kernel,
    _attn_transpose_from_cache_kernel,
    _attn_transpose_scores_kernel,
    _attn_unpack_grad_kernel,
)


comptime MASK_NEG: Float64 = -1.0e30


# ──────────────────────────────────────────────────────────────────────
# Host mask builders (data-independent; depend only on token layout). Carried
# VERBATIM from the legacy leaf.
# ──────────────────────────────────────────────────────────────────────


def causal_mask(seq_len: Int) -> List[Scalar[DT]]:
    """Additive lower-triangular mask: position i may attend to j ≤ i."""
    var m = List[Scalar[DT]]()
    for i in range(seq_len):
        for j in range(seq_len):
            m.append(Scalar[DT](0.0) if j <= i else Scalar[DT](MASK_NEG))
    return m^


def all_allow_mask(seq_len: Int) -> List[Scalar[DT]]:
    """Additive all-zero mask → unrestricted (bidirectional) attention."""
    var m = List[Scalar[DT]]()
    for _ in range(seq_len * seq_len):
        m.append(Scalar[DT](0.0))
    return m^


def build_modality_mask[
    mode: StaticString
](
    modality_ids: List[Int], n_latents: Int, agent_mod_in: Int = -1
) -> List[Scalar[DT]]:
    """Port of `model.py:SpaceSelfAttentionModality._build_allow` (Dreamer 4).

    `modality_ids[k]` is the modality of token k; the first `n_latents` tokens
    are latent register tokens. Returns the additive `SEQ*SEQ` mask:
    allowed→0.0, disallowed→NEG.

    Modes (the `wm_*` modes treat the highest modality id as the AGENT modality
    and are only meaningful when agent tokens are present; they ignore
    `n_latents`):
      - "encoder": latents attend to all; non-latents only within own modality.
      - "decoder": latents attend only to latents; non-latents attend to
        latents + own modality.
      - "wm_agent": full mixing — every token attends to every token.
      - "wm_agent_isolated": non-agent queries attend to all NON-agent keys;
        agent queries attend ONLY to agent keys.
      - "wm_agent_bc": non-agent queries attend to all NON-agent keys; agent
        queries attend to ALL keys (paper §3.3 imagination/BC mask).
    """
    comptime assert (
        mode == "encoder"
        or mode == "decoder"
        or mode == "wm_agent"
        or mode == "wm_agent_isolated"
        or mode == "wm_agent_bc"
    ), "build_modality_mask: unknown mode"

    var S = len(modality_ids)
    var agent_mod = agent_mod_in
    if agent_mod < 0:
        agent_mod = 0
        for k in range(S):
            if modality_ids[k] > agent_mod:
                agent_mod = modality_ids[k]

    var m = List[Scalar[DT]]()
    for i in range(S):
        var is_q_lat = i < n_latents
        var q_is_agent = modality_ids[i] == agent_mod
        for j in range(S):
            var is_k_lat = j < n_latents
            var k_is_agent = modality_ids[j] == agent_mod
            var same_mod = modality_ids[i] == modality_ids[j]

            var allow: Bool
            comptime if mode == "encoder":
                allow = True if is_q_lat else same_mod
            elif mode == "decoder":
                if is_q_lat:
                    allow = is_k_lat
                else:
                    allow = is_k_lat or same_mod
            elif mode == "wm_agent":
                allow = True
            elif mode == "wm_agent_isolated":
                if q_is_agent:
                    allow = k_is_agent
                else:
                    allow = not k_is_agent
            else:
                if q_is_agent:
                    allow = True
                else:
                    allow = not k_is_agent

            m.append(Scalar[DT](0.0) if allow else Scalar[DT](MASK_NEG))
    return m^


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — custom per-(b,h) path, mirroring the base attention custom
# kernels with two differences: (1) the forward adds the additive mask to each
# score; (2) all j/i loops run the full [0, SEQ) range (no causal loop-bound).
# The backward kernels are otherwise the CAUSAL=False attention kernels (masked
# weights are 0 in cache). Carried VERBATIM. The shared mask buffer is [SEQ*SEQ]
# (batch- and head-independent).
# ──────────────────────────────────────────────────────────────────────


def _masked_fwd_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    K_OFF: Int, V_OFF: Int, ATTN_OFF: Int, ADT: DType = DT,
](
    output: Tile[ADT, BATCH, OUT_DIM],
    input: Tile[ADT, BATCH, IN_DIM],
    cache: Tile[DT, BATCH, CACHE_SIZE],
    mask: Tile[DT, SEQ * SEQ],
):
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))

    # Step 1: cache this head's Q/K/V slice (for backward).
    var n_qkv = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_qkv:
        var i = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        cache.ptr[unsafe_offset=b * CACHE_SIZE + i * DIM + h_off + d] = rebind[Scalar[ADT]](
            input.ptr[unsafe_offset=b * IN_DIM + i * DIM + h_off + d]
        ).cast[DT]()
        cache.ptr[unsafe_offset=b * CACHE_SIZE + K_OFF + i * DIM + h_off + d] = rebind[
            Scalar[ADT]
        ](input.ptr[unsafe_offset=b * IN_DIM + K_OFF + i * DIM + h_off + d]).cast[DT]()
        cache.ptr[unsafe_offset=b * CACHE_SIZE + V_OFF + i * DIM + h_off + d] = rebind[
            Scalar[ADT]
        ](input.ptr[unsafe_offset=b * IN_DIM + V_OFF + i * DIM + h_off + d]).cast[DT]()
        idx0 += bs

    # Step 2: per-row masked attention; threads stride over query rows i.
    var i = tid
    while i < SEQ:
        var max_score = Scalar[DT](-1e30)
        for j in range(SEQ):
            var s = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var q = rebind[Scalar[ADT]](
                    input.ptr[unsafe_offset=b * IN_DIM + i * DIM + h_off + d]
                ).cast[DT]()
                var k = rebind[Scalar[ADT]](
                    input.ptr[unsafe_offset=b * IN_DIM + K_OFF + j * DIM + h_off + d]
                ).cast[DT]()
                s += q * k
            s = s * scale + rebind[Scalar[DT]](mask.ptr[unsafe_offset=i * SEQ + j])
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            cache.ptr[unsafe_offset=aidx] = s
            if s > max_score:
                max_score = s

        var sum_exp = Scalar[DT](0)
        for j in range(SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var e = exp(rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) - max_score)
            cache.ptr[unsafe_offset=aidx] = e
            sum_exp += e

        var inv_sum = Scalar[DT](1) / sum_exp
        for j in range(SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            cache.ptr[unsafe_offset=aidx] = rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) * inv_sum

        for d in range(HEAD_DIM):
            var acc = Scalar[DT](0)
            for j in range(SEQ):
                var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
                var v = rebind[Scalar[ADT]](
                    input.ptr[unsafe_offset=b * IN_DIM + V_OFF + j * DIM + h_off + d]
                ).cast[DT]()
                acc += rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) * v
            output.ptr[unsafe_offset=b * OUT_DIM + i * DIM + h_off + d] = acc.cast[ADT]()
        i += bs


def _masked_zero_grad_kernel[
    BATCH: Int, IN_DIM: Int, ADT: DType = DT
](
    grad_input: Tile[ADT, BATCH, IN_DIM],
):
    var idx = Int(global_idx.x)
    if idx < BATCH * IN_DIM:
        grad_input.ptr[unsafe_offset=idx] = Scalar[ADT](0)


def _masked_dV_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int, V_OFF: Int, ATTN_OFF: Int,
    ADT: DType = DT,
](
    grad_input: Tile[ADT, BATCH, IN_DIM],
    grad_output: Tile[ADT, BATCH, OUT_DIM],
    cache: Tile[DT, BATCH, CACHE_SIZE],
):
    # dV[j] = Σ_i attn[i,j] * grad_out[i]. Full i-range (mask gates via attn=0).
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var n_jd = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_jd:
        var j = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        var acc = Scalar[DT](0)
        for i in range(SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var go = rebind[Scalar[ADT]](
                grad_output.ptr[unsafe_offset=b * OUT_DIM + i * DIM + h_off + d]
            ).cast[DT]()
            acc += rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) * go
        var dv_idx = b * IN_DIM + V_OFF + j * DIM + h_off + d
        grad_input.ptr[unsafe_offset=dv_idx] = (
            rebind[Scalar[ADT]](grad_input.ptr[unsafe_offset=dv_idx]).cast[DT]() + acc
        ).cast[ADT]()
        idx0 += bs


def _masked_dscore_dQ_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    K_OFF: Int, V_OFF: Int, ATTN_OFF: Int, ADT: DType = DT,
](
    grad_input: Tile[ADT, BATCH, IN_DIM],
    grad_output: Tile[ADT, BATCH, OUT_DIM],
    cache: Tile[DT, BATCH, CACHE_SIZE],
):
    # Per row i: dot_sum, then d_score (overwrites cache.attn), then dQ.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))

    var i = tid
    while i < SEQ:
        var dot_sum = Scalar[DT](0)
        for j in range(SEQ):
            var d_attn = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var go = rebind[Scalar[ADT]](
                    grad_output.ptr[unsafe_offset=b * OUT_DIM + i * DIM + h_off + d]
                ).cast[DT]()
                var v = rebind[Scalar[DT]](
                    cache.ptr[unsafe_offset=b * CACHE_SIZE + V_OFF + j * DIM + h_off + d]
                )
                d_attn += go * v
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            dot_sum += rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx]) * d_attn

        for j in range(SEQ):
            var d_attn = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var go = rebind[Scalar[ADT]](
                    grad_output.ptr[unsafe_offset=b * OUT_DIM + i * DIM + h_off + d]
                ).cast[DT]()
                var v = rebind[Scalar[DT]](
                    cache.ptr[unsafe_offset=b * CACHE_SIZE + V_OFF + j * DIM + h_off + d]
                )
                d_attn += go * v
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var attn_w = rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx])
            cache.ptr[unsafe_offset=aidx] = attn_w * (d_attn - dot_sum) * scale

        for d in range(HEAD_DIM):
            var acc = Scalar[DT](0)
            for j in range(SEQ):
                var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
                var d_score = rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx])
                var k = rebind[Scalar[DT]](
                    cache.ptr[unsafe_offset=b * CACHE_SIZE + K_OFF + j * DIM + h_off + d]
                )
                acc += d_score * k
            var dq_idx = b * IN_DIM + i * DIM + h_off + d
            grad_input.ptr[unsafe_offset=dq_idx] = (
                rebind[Scalar[ADT]](grad_input.ptr[unsafe_offset=dq_idx]).cast[DT]() + acc
            ).cast[ADT]()
        i += bs


def _masked_dK_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, CACHE_SIZE: Int, K_OFF: Int, ATTN_OFF: Int, ADT: DType = DT,
](
    grad_input: Tile[ADT, BATCH, IN_DIM],
    cache: Tile[DT, BATCH, CACHE_SIZE],
):
    # dK[j] = Σ_i d_score[i,j] * Q[i]. Reads d_score from cache.attn. Full range.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var n_jd = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_jd:
        var j = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        var acc = Scalar[DT](0)
        for i in range(SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var d_score = rebind[Scalar[DT]](cache.ptr[unsafe_offset=aidx])
            var q = rebind[Scalar[DT]](cache.ptr[unsafe_offset=b * CACHE_SIZE + i * DIM + h_off + d])
            acc += d_score * q
        var dk_idx = b * IN_DIM + K_OFF + j * DIM + h_off + d
        grad_input.ptr[unsafe_offset=dk_idx] = (
            rebind[Scalar[ADT]](grad_input.ptr[unsafe_offset=dk_idx]).cast[DT]() + acc
        ).cast[ADT]()
        idx0 += bs


# BMM fast path softmax — base attention's `_attn_softmax_kernel` with the causal
# zero-fill replaced by an additive mask before the max-shift. Full range; masked
# entries get score ≈ NEG → softmax weight 0. 1 block per (b,h). Carried VERBATIM.
def _masked_softmax_kernel[
    BATCH: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CACHE_SIZE: Int, SCORES: Int, BH: Int,
](
    scores: Tile[DT, SCORES],
    cache: Tile[DT, BATCH, CACHE_SIZE],
    mask: Tile[DT, SEQ * SEQ],
):
    var blk = Int(block_idx.x)
    if blk >= BH:
        return
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    comptime ATTN_OFF = 3 * SEQ * (N_HEADS * HEAD_DIM)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))
    var bh_off = blk * SEQ * SEQ
    var cache_attn_base = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ
    var i = tid
    while i < SEQ:
        var row_off = bh_off + i * SEQ
        var cache_row = cache_attn_base + i * SEQ
        var mx = Scalar[DT](-1e30)
        for j in range(SEQ):
            var s = rebind[Scalar[DT]](scores.ptr[unsafe_offset=row_off + j]) * scale + rebind[
                Scalar[DT]
            ](mask.ptr[unsafe_offset=i * SEQ + j])
            scores.ptr[unsafe_offset=row_off + j] = s
            if s > mx:
                mx = s
        var se = Scalar[DT](0)
        for j in range(SEQ):
            var e = exp(rebind[Scalar[DT]](scores.ptr[unsafe_offset=row_off + j]) - mx)
            scores.ptr[unsafe_offset=row_off + j] = e
            se += e
        var inv = Scalar[DT](1) / se
        for j in range(SEQ):
            var w = rebind[Scalar[DT]](scores.ptr[unsafe_offset=row_off + j]) * inv
            scores.ptr[unsafe_offset=row_off + j] = w
            cache.ptr[unsafe_offset=cache_row + j] = w
        i += bs


# ──────────────────────────────────────────────────────────────────────
struct MaskedAttention[
    DIM: Int,
    N_HEADS: Int,
    SEQ_LEN: Int,
    USE_MAX_KERNELS: Bool = True,
    ADT: DType = DT,
](Module):
    comptime ARITY: Int = 1
    # Activation-flow dtype. `MaskedAttention[D, H, S]` = fp32 (ACT_DT == DT, the
    # legacy path, byte-identical); `…[D, H, S, …, bfloat16]` flows its I/O
    # activations at bf16 while computing fp32 INTERNALLY (cache + QKᵀ/softmax/
    # attn·V + the additive mask all stay fp32; only the I/O-activation kernel
    # operands cast at the bf16 boundary). bf16-flow is GPU-only.
    comptime ACT_DT = Self.ADT
    comptime HEAD_DIM: Int = Self.DIM // Self.N_HEADS
    comptime IN_DIMS = Array[Int, 1](fill=Self.SEQ_LEN * Self.DIM * 3)
    # `Array` is not `ImplicitlyCopyable` (Mojo 1.0): indexing the comptime
    # `IN_DIMS` from a runtime context would materialize the whole array.
    comptime IN_DIM0 = Self.SEQ_LEN * Self.DIM * 3
    comptime OUT_DIM = Self.SEQ_LEN * Self.DIM
    comptime K_OFF: Int = Self.SEQ_LEN * Self.DIM
    comptime V_OFF: Int = 2 * Self.SEQ_LEN * Self.DIM
    comptime ATTN_OFF: Int = 3 * Self.SEQ_LEN * Self.DIM
    comptime CACHE_SIZE: Int = (
        3 * Self.SEQ_LEN * Self.DIM
        + Self.N_HEADS * Self.SEQ_LEN * Self.SEQ_LEN
    )
    comptime MASK_SIZE: Int = Self.SEQ_LEN * Self.SEQ_LEN

    # Cache (leaf-owned, output-caching) — [BATCH, CACHE_SIZE], lazy.
    var cache: Tensor
    # Mask [SEQ*SEQ] additive bias — owned Tensor (CPU data + optional GPU buf).
    var mask: Tensor
    # BMM scratch slabs (separate owned Tensors). Lazily sized.
    var sp0: Tensor
    var sp1: Tensor
    var sp2: Tensor
    var sp3: Tensor
    var ss0: Tensor
    var ss1: Tensor
    var is_gpu: Bool

    def __init__(out self):
        comptime assert (
            Self.DIM % Self.N_HEADS == 0
        ), "MaskedAttention: DIM must be divisible by N_HEADS"
        self.cache = Tensor()
        self.mask = Tensor()
        self.sp0 = Tensor()
        self.sp1 = Tensor()
        self.sp2 = Tensor()
        self.sp3 = Tensor()
        self.ss0 = Tensor()
        self.ss1 = Tensor()
        self.is_gpu = False

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "MaskedAttention: target must be 'cpu' or 'gpu'"
        )
        var a = Self()
        # Default = all-allow (bidirectional), equals non-causal SDPA.
        a.mask = Tensor.alloc(Self.MASK_SIZE)  # zero-filled = all-allow
        comptime if target != "cpu":
            if not ctx:
                raise Error(
                    "MaskedAttention.make[target='gpu']: ctx required"
                )
            a.is_gpu = True
            a.mask.upload(ctx.value())
        return a^

    def set_mask(mut self, var mask: List[Scalar[DT]], ctx: Optional[DeviceContext] = None) raises:
        if len(mask) != Self.MASK_SIZE:
            raise Error("MaskedAttention.set_mask: expected SEQ_LEN*SEQ_LEN")
        self.mask.ensure(Self.MASK_SIZE)
        for i in range(Self.MASK_SIZE):
            self.mask.data[i] = mask[i]
        self.mask.n = Self.MASK_SIZE
        if self.is_gpu:
            self.mask.upload(ctx.value())
        _ = mask^

    def _ensure_scratch_gpu[BATCH: Int](mut self, c: DeviceContext) raises:
        comptime PACKED = BATCH * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BATCH * Self.N_HEADS * Self.SEQ_LEN * Self.SEQ_LEN
        self.sp0.ensure_gpu(c, PACKED)
        self.sp1.ensure_gpu(c, PACKED)
        self.sp2.ensure_gpu(c, PACKED)
        self.sp3.ensure_gpu(c, PACKED)
        self.ss0.ensure_gpu(c, SCORES)
        self.ss1.ensure_gpu(c, SCORES)

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # CPU helper is fp32-only (`Tensor`) → rebind (sound, ACT_DT IS DT);
            # GPU helpers are ACT_DT-generic → pass activations directly.
            comptime if target == "cpu":
                ref in0d = rebind[Tensor](in0)
                ref outd = rebind[Tensor](out)
                self._forward_cpu[B](in0d, outd)
            else:
                var c = ctx.value()
                out.ensure_gpu(c, B * Self.OUT_DIM)
                self.cache.ensure_gpu(c, B * Self.CACHE_SIZE)
                comptime if Self.USE_MAX_KERNELS:
                    self._forward_gpu_bmm[B](in0, out, c)
                else:
                    self._forward_gpu_custom[B](in0, out, c)
        else:
            # ── bf16-flow path (GPU-only). I/O activations cast at the boundary;
            #    cache + the masked softmax/GEMMs stay fp32 (fp32-internal). ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow MaskedAttention is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            self.cache.ensure_gpu(c, B * Self.CACHE_SIZE)
            comptime if Self.USE_MAX_KERNELS:
                self._forward_gpu_bmm[B](in0, out, c)
            else:
                self._forward_gpu_custom[B](in0, out, c)

    def _forward_gpu_custom[
        B: Int
    ](
        mut self,
        mut in0: TensorImpl[Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        c: DeviceContext,
    ) raises:
        comptime lay_in = RowMajor[B, Self.IN_DIM0]
        comptime lay_out = RowMajor[B, Self.OUT_DIM]
        comptime lay_c = RowMajor[B, Self.CACHE_SIZE]
        comptime lay_m = RowMajor[Self.MASK_SIZE]
        comptime kernel = _masked_fwd_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIM0, Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF, Self.ADT,
        ]
        c.enqueue_function[kernel](
            out.tt["gpu", B, Self.OUT_DIM](),
            in0.tt["gpu", B, Self.IN_DIM0](),
            self.cache.tt["gpu", B, Self.CACHE_SIZE](),
            self.mask.tt["gpu", Self.MASK_SIZE](),
            grid_dim=B * Self.N_HEADS, block_dim=TPB,
        )

    def _forward_gpu_bmm[
        B: Int
    ](
        mut self,
        mut in0: TensorImpl[Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        c: DeviceContext,
    ) raises:
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        self._ensure_scratch_gpu[B](c)

        comptime lay_in = RowMajor[B, Self.IN_DIM0]
        comptime lay_out = RowMajor[B, Self.OUT_DIM]
        comptime lay_c = RowMajor[B, Self.CACHE_SIZE]
        comptime lay_m = RowMajor[Self.MASK_SIZE]
        comptime lay_p = RowMajor[PACKED]
        comptime lay_s = RowMajor[SCORES]

        # 1. pack QKV → (BH, SEQ, HEAD_DIM) + write cache.
        comptime pelems = B * Self.SEQ_LEN * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        comptime pack_k = _attn_pack_qkv_fwd_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIM0, Self.CACHE_SIZE, PACKED, Self.ADT,
        ]
        c.enqueue_function[pack_k](
            self.sp0.tt["gpu", PACKED](),
            self.sp1.tt["gpu", PACKED](),
            self.sp2.tt["gpu", PACKED](),
            self.cache.tt["gpu", B, Self.CACHE_SIZE](),
            in0.tt["gpu", B, Self.IN_DIM0](),
            grid_dim=pblocks, block_dim=TPB,
        )

        # 2. scores = Q @ Kᵀ.
        bmm[transpose_b=True, A0=BH, A1=Self.SEQ_LEN, A2=Self.HEAD_DIM, B0=BH, B1=Self.SEQ_LEN, B2=Self.HEAD_DIM, O0=BH, O1=Self.SEQ_LEN, O2=Self.SEQ_LEN](
            self.ss0.dev.value(), self.sp0.dev.value(), self.sp1.dev.value(), c
        )

        # 3. masked softmax in-place; mirror weights into cache.attn.
        comptime sm_k = _masked_softmax_kernel[
            B, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CACHE_SIZE, SCORES, BH,
        ]
        c.enqueue_function[sm_k](
            self.ss0.tt["gpu", SCORES](),
            self.cache.tt["gpu", B, Self.CACHE_SIZE](),
            self.mask.tt["gpu", Self.MASK_SIZE](),
            grid_dim=BH, block_dim=TPB,
        )

        # 4. packed_out = attn @ V.
        bmm[A0=BH, A1=Self.SEQ_LEN, A2=Self.SEQ_LEN, B0=BH, B1=Self.SEQ_LEN, B2=Self.HEAD_DIM, O0=BH, O1=Self.SEQ_LEN, O2=Self.HEAD_DIM](
            self.sp3.dev.value(), self.ss0.dev.value(), self.sp2.dev.value(), c
        )

        # 5. unpack → output.
        comptime up_k = _attn_unpack_out_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.OUT_DIM, PACKED, Self.ADT,
        ]
        c.enqueue_function[up_k](
            out.tt["gpu", B, Self.OUT_DIM](),
            self.sp3.tt["gpu", PACKED](),
            grid_dim=pblocks, block_dim=TPB,
        )

    def _forward_cpu[B: Int](mut self, mut in0: Tensor, mut out: Tensor) raises:
        # Scalar Float64 per-(b,h) path (mirrors legacy MaskedAttention CPU).
        out.ensure(B * Self.OUT_DIM)
        self.cache.ensure(B * Self.CACHE_SIZE)
        ref ip = in0.data
        ref op = out.data
        ref cp = self.cache.data
        ref mp = self.mask.data
        comptime IN = Self.IN_DIM0
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        comptime SD = Self.SEQ_LEN * Self.DIM
        var scale = 1.0 / sqrt(Float64(Self.HEAD_DIM))

        for b in range(B):
            for i in range(SD):
                cp[b * C + i] = ip[b * IN + i]
                cp[b * C + Self.K_OFF + i] = ip[b * IN + Self.K_OFF + i]
                cp[b * C + Self.V_OFF + i] = ip[b * IN + Self.V_OFF + i]

            for h in range(Self.N_HEADS):
                var h_off = h * Self.HEAD_DIM
                for i in range(Self.SEQ_LEN):
                    var max_score: Float64 = -1e30
                    for j in range(Self.SEQ_LEN):
                        var score: Float64 = 0.0
                        for d in range(Self.HEAD_DIM):
                            var q = Float64(ip[b * IN + i * Self.DIM + h_off + d])
                            var k = Float64(
                                ip[b * IN + Self.K_OFF + j * Self.DIM + h_off + d]
                            )
                            score += q * k
                        score = score * scale + Float64(mp[i * Self.SEQ_LEN + j])
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        cp[ai] = Scalar[DT](score)
                        if score > max_score:
                            max_score = score

                    var sum_exp: Float64 = 0.0
                    for j in range(Self.SEQ_LEN):
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        var e = exp(Float64(cp[ai]) - max_score)
                        cp[ai] = Scalar[DT](e)
                        sum_exp += e

                    var inv = 1.0 / sum_exp
                    for j in range(Self.SEQ_LEN):
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        cp[ai] = Scalar[DT](Float64(cp[ai]) * inv)

                    for d in range(Self.HEAD_DIM):
                        var acc: Float64 = 0.0
                        for j in range(Self.SEQ_LEN):
                            var ai = (
                                b * C + Self.ATTN_OFF
                                + h * Self.SEQ_LEN * Self.SEQ_LEN
                                + i * Self.SEQ_LEN + j
                            )
                            var v = Float64(
                                ip[b * IN + Self.V_OFF + j * Self.DIM + h_off + d]
                            )
                            acc += Float64(cp[ai]) * v
                        op[b * OUT + i * Self.DIM + h_off + d] = Scalar[DT](acc)

    # ----- Backward --------------------------------------------------------

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # forward_input unused — output-caching (reads only cache + grad_output).
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # CPU helper is fp32-only (`Tensor`) → rebind (sound, ACT_DT IS DT);
            # GPU helpers are ACT_DT-generic → pass activations directly.
            comptime if target == "cpu":
                ref god = rebind[Tensor](grad_output)
                ref gind = rebind[Tensor](gin)
                self._vjp_cpu[B](god, gind)
            else:
                var c = ctx.value()
                gin.ensure_gpu(c, B * Self.IN_DIM0)
                comptime if Self.USE_MAX_KERNELS:
                    self._vjp_gpu_bmm[B](grad_output, gin, c)
                else:
                    self._vjp_gpu_custom[B](grad_output, gin, c)
        else:
            # ── bf16-flow path (GPU-only). I/O activations cast at the boundary;
            #    cache + grad math stay fp32 (fp32-internal). ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow MaskedAttention is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_DIM0)
            comptime if Self.USE_MAX_KERNELS:
                self._vjp_gpu_bmm[B](grad_output, gin, c)
            else:
                self._vjp_gpu_custom[B](grad_output, gin, c)

    def _vjp_gpu_custom[
        B: Int
    ](
        mut self,
        mut grad_output: TensorImpl[Self.ACT_DT],
        mut gin: TensorImpl[Self.ACT_DT],
        c: DeviceContext,
    ) raises:
        comptime lay_in = RowMajor[B, Self.IN_DIM0]
        comptime lay_out = RowMajor[B, Self.OUT_DIM]
        comptime lay_c = RowMajor[B, Self.CACHE_SIZE]
        comptime grid_bh = B * Self.N_HEADS
        # 1) zero grad_input.
        comptime zk = _masked_zero_grad_kernel[B, Self.IN_DIM0, Self.ADT]
        comptime zn = (B * Self.IN_DIM0 + TPB - 1) // TPB
        c.enqueue_function[zk](
            gin.tt["gpu", B, Self.IN_DIM0](), grid_dim=zn, block_dim=TPB
        )
        # 2) dV.
        comptime dvk = _masked_dV_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIM0, Self.OUT_DIM, Self.CACHE_SIZE,
            Self.V_OFF, Self.ATTN_OFF, Self.ADT,
        ]
        c.enqueue_function[dvk](
            gin.tt["gpu", B, Self.IN_DIM0](),
            grad_output.tt["gpu", B, Self.OUT_DIM](),
            self.cache.tt["gpu", B, Self.CACHE_SIZE](),
            grid_dim=grid_bh, block_dim=TPB,
        )
        # 3) dscore + dQ.
        comptime dqk = _masked_dscore_dQ_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIM0, Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF, Self.ADT,
        ]
        c.enqueue_function[dqk](
            gin.tt["gpu", B, Self.IN_DIM0](),
            grad_output.tt["gpu", B, Self.OUT_DIM](),
            self.cache.tt["gpu", B, Self.CACHE_SIZE](),
            grid_dim=grid_bh, block_dim=TPB,
        )
        # 4) dK.
        comptime dkk = _masked_dK_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIM0, Self.CACHE_SIZE, Self.K_OFF, Self.ATTN_OFF, Self.ADT,
        ]
        c.enqueue_function[dkk](
            gin.tt["gpu", B, Self.IN_DIM0](),
            self.cache.tt["gpu", B, Self.CACHE_SIZE](),
            grid_dim=grid_bh, block_dim=TPB,
        )

    def _vjp_gpu_bmm[
        B: Int
    ](
        mut self,
        mut grad_output: TensorImpl[Self.ACT_DT],
        mut gin: TensorImpl[Self.ACT_DT],
        c: DeviceContext,
    ) raises:
        # Byte-for-byte the storage attention bmm backward (mask already baked
        # into cache.attn weights → masked entries are 0).
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        comptime SL = Self.SEQ_LEN
        comptime HD = Self.HEAD_DIM
        self._ensure_scratch_gpu[B](c)

        comptime lay_in = RowMajor[B, Self.IN_DIM0]
        comptime lay_out = RowMajor[B, Self.OUT_DIM]
        comptime lay_c = RowMajor[B, Self.CACHE_SIZE]
        comptime lay_p = RowMajor[PACKED]
        comptime lay_s = RowMajor[SCORES]

        comptime pelems = B * SL * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        comptime sblocks = (SCORES + TPB - 1) // TPB

        # 1. pack dout + cache Q/K/V.
        comptime pin_k = _attn_pack_in_bwd_kernel[
            B, Self.DIM, Self.N_HEADS, SL, HD,
            Self.IN_DIM0, Self.OUT_DIM, Self.CACHE_SIZE, PACKED, Self.ADT,
        ]
        c.enqueue_function[pin_k](
            self.sp0.tt["gpu", PACKED](),
            self.sp1.tt["gpu", PACKED](),
            self.sp2.tt["gpu", PACKED](),
            self.sp3.tt["gpu", PACKED](),
            grad_output.tt["gpu", B, Self.OUT_DIM](),
            self.cache.tt["gpu", B, Self.CACHE_SIZE](),
            grid_dim=pblocks, block_dim=TPB,
        )

        # 2. dattn(ss0) = dout @ Vᵀ.
        bmm[transpose_b=True, A0=BH, A1=SL, A2=HD, B0=BH, B1=SL, B2=HD, O0=BH, O1=SL, O2=SL](
            self.ss0.dev.value(), self.sp0.dev.value(), self.sp3.dev.value(), c
        )

        # 3. softmax jvp → dscore(ss1).
        comptime jvp_k = _attn_softmax_jvp_kernel[
            B, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        c.enqueue_function[jvp_k](
            self.ss1.tt["gpu", SCORES](),
            self.ss0.tt["gpu", SCORES](),
            self.cache.tt["gpu", B, Self.CACHE_SIZE](),
            grid_dim=BH, block_dim=TPB,
        )

        # 4. attn_T(ss0) = transpose(cache.attn).
        comptime tac_k = _attn_transpose_from_cache_kernel[
            B, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        c.enqueue_function[tac_k](
            self.ss0.tt["gpu", SCORES](),
            self.cache.tt["gpu", B, Self.CACHE_SIZE](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 5. dV(sp3) = attn_T(ss0) @ dout(sp0).
        bmm[A0=BH, A1=SL, A2=SL, B0=BH, B1=SL, B2=HD, O0=BH, O1=SL, O2=HD](
            self.sp3.dev.value(), self.ss0.dev.value(), self.sp0.dev.value(), c
        )

        # 6. dscore_T(ss0) = transpose(dscore(ss1)).
        comptime ts_k = _attn_transpose_scores_kernel[SL, SCORES, BH]
        c.enqueue_function[ts_k](
            self.ss0.tt["gpu", SCORES](),
            self.ss1.tt["gpu", SCORES](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 7. dK(sp0) = dscore_T(ss0) @ Q(sp1).
        bmm[A0=BH, A1=SL, A2=SL, B0=BH, B1=SL, B2=HD, O0=BH, O1=SL, O2=HD](
            self.sp0.dev.value(), self.ss0.dev.value(), self.sp1.dev.value(), c
        )

        # 8. dQ(sp1) = dscore(ss1) @ K(sp2).
        bmm[A0=BH, A1=SL, A2=SL, B0=BH, B1=SL, B2=HD, O0=BH, O1=SL, O2=HD](
            self.sp1.dev.value(), self.ss1.dev.value(), self.sp2.dev.value(), c
        )

        # 9. unpack dQ(sp1)/dK(sp0)/dV(sp3) → grad_input.
        comptime ug_k = _attn_unpack_grad_kernel[
            B, Self.DIM, Self.N_HEADS, SL, HD, Self.IN_DIM0, PACKED, Self.ADT,
        ]
        c.enqueue_function[ug_k](
            gin.tt["gpu", B, Self.IN_DIM0](),
            self.sp1.tt["gpu", PACKED](),
            self.sp0.tt["gpu", PACKED](),
            self.sp3.tt["gpu", PACKED](),
            grid_dim=pblocks, block_dim=TPB,
        )

    def _vjp_cpu[
        B: Int
    ](mut self, mut grad_output: Tensor, mut gin: Tensor) raises:
        # Scalar Float64 per-(b,h) path (mirrors legacy MaskedAttention CPU vjp).
        gin.ensure(B * Self.IN_DIM0)
        ref gop = grad_output.data
        ref gip = gin.data
        ref cp = self.cache.data
        comptime IN = Self.IN_DIM0
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        var scale = 1.0 / sqrt(Float64(Self.HEAD_DIM))

        for i in range(B * IN):
            gip[i] = 0.0

        for b in range(B):
            for h in range(Self.N_HEADS):
                var h_off = h * Self.HEAD_DIM

                # Step 1: dV[j] += attn[i,j] * grad_out[i].
                for i in range(Self.SEQ_LEN):
                    for j in range(Self.SEQ_LEN):
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        var attn_w = Float64(cp[ai])
                        for d in range(Self.HEAD_DIM):
                            var go = Float64(
                                gop[b * OUT + i * Self.DIM + h_off + d]
                            )
                            var dv_idx = (
                                b * IN + Self.V_OFF + j * Self.DIM + h_off + d
                            )
                            gip[dv_idx] = gip[dv_idx] + Scalar[DT](attn_w * go)

                # Step 2: softmax backward → dQ, dK.
                for i in range(Self.SEQ_LEN):
                    var dot_sum: Float64 = 0.0
                    for j in range(Self.SEQ_LEN):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.HEAD_DIM):
                            var go = Float64(
                                gop[b * OUT + i * Self.DIM + h_off + d]
                            )
                            var v = Float64(
                                cp[b * C + Self.V_OFF + j * Self.DIM + h_off + d]
                            )
                            d_attn += go * v
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        dot_sum += d_attn * Float64(cp[ai])

                    for j in range(Self.SEQ_LEN):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.HEAD_DIM):
                            var go = Float64(
                                gop[b * OUT + i * Self.DIM + h_off + d]
                            )
                            var v = Float64(
                                cp[b * C + Self.V_OFF + j * Self.DIM + h_off + d]
                            )
                            d_attn += go * v
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        var attn_w = Float64(cp[ai])
                        var d_score = attn_w * (d_attn - dot_sum) * scale
                        for d in range(Self.HEAD_DIM):
                            var q = Float64(cp[b * C + i * Self.DIM + h_off + d])
                            var k = Float64(
                                cp[b * C + Self.K_OFF + j * Self.DIM + h_off + d]
                            )
                            var dq_idx = b * IN + i * Self.DIM + h_off + d
                            gip[dq_idx] = gip[dq_idx] + Scalar[DT](d_score * k)
                            var dk_idx = (
                                b * IN + Self.K_OFF + j * Self.DIM + h_off + d
                            )
                            gip[dk_idx] = gip[dk_idx] + Scalar[DT](d_score * q)

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (no Param fields → no-op).
