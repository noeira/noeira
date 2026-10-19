"""MAEReplacer[NP, D, P_MIN, P_MAX, SEED] — masked-autoencoding patch dropout.

Transformed from legacy `nn.primitives.MAEReplacer` (surface-only change; the
keep-decision RNG, the CPU loops, and the 3 GPU kernels are carried over
verbatim).

Dreamer 4 trains the tokenizer with masked autoencoding: a random fraction of
the projected patch tokens is replaced by a learned `mask_token`, and the
reconstruction loss is applied only on the replaced patches
(`model.py:MAEReplacer` + `recon_loss_from_mae`). This leaf does the
replacement and remembers which patches were dropped (for the loss + backward).

Operates per frame at nn-BATCH = B·T: IN_DIM == OUT_DIM == NP·D.

    p_bt   ~ U(p_min, p_max)                 (one drop-rate per frame)
    keep   = U(0,1) < (1 - p_bt)             (per patch)
    out    = where(keep, input, mask_token)

`mask_token` (D) is the only parameter; its grad accumulates the grad-output
of every dropped patch (batch-reduced). Kept patches pass the gradient
straight through; dropped patches get grad_input 0.

RNG: PhiloxRandom seeded by comptime SEED at offset `base + idx`, base =
`rng_step * STRIDE`. The step is bumped by `advance_rng()` (once per training
iter), NOT per forward — so a gradcheck that never advances sees a frozen mask.
The keep decision is computed in Float32 on BOTH CPU and GPU (Metal has no
Float64), so the masks are bit-identical across targets. The backward
RECOMPUTES the keep decision (no need to read the stored mask).

`mae_keep()` returns the per-patch `keep` Tensor (1.0 kept / 0.0 dropped).
CPU + GPU.
"""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.random.philox import Random as PhiloxRandom
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor


comptime MAE_RTPB = 64


@always_inline
def _kept(
    seed: UInt64, base: UInt64, batch: Int, np: Int, bt: Int, i: Int,
    pmin: Scalar[DT], span: Scalar[DT],
) -> Bool:
    """Per-patch keep decision (Float32, identical on CPU & GPU)."""
    var rp = PhiloxRandom(seed=seed, offset=base + UInt64(bt))
    var p_bt = pmin + span * Scalar[DT](rp.step_uniform()[0])
    var keep_prob = Scalar[DT](1.0) - p_bt
    var ri = PhiloxRandom(
        seed=seed, offset=base + UInt64(batch) + UInt64(bt * np + i)
    )
    var u = Scalar[DT](ri.step_uniform()[0])
    return u < keep_prob


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _mae_fwd_kernel[
    BATCH: Int, NP: Int, D: Int, SEED: UInt64
](
    input: Tile[DT, BATCH, NP * D],
    mask_token: Tile[DT, D],
    output: Tile[DT, BATCH, NP * D],
    keep: Tile[DT, BATCH * NP],
    base: UInt64, pmin: Scalar[DT], span: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * NP * D:
        return
    var d = idx % D
    var rem = idx // D
    var i = rem % NP
    var bt = rem // NP
    var kept = _kept(SEED, base, BATCH, NP, bt, i, pmin, span)
    if kept:
        output.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](input.ptr[unsafe_offset=idx])
    else:
        output.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](mask_token.ptr[unsafe_offset=d])
    if d == 0:
        keep.ptr[unsafe_offset=bt * NP + i] = Scalar[DT](1.0) if kept else Scalar[DT](0.0)


def _mae_grad_input_kernel[
    BATCH: Int, NP: Int, D: Int, SEED: UInt64
](
    grad_output: Tile[DT, BATCH, NP * D],
    grad_input: Tile[DT, BATCH, NP * D],
    base: UInt64, pmin: Scalar[DT], span: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * NP * D:
        return
    var rem = idx // D
    var i = rem % NP
    var bt = rem // NP
    if _kept(SEED, base, BATCH, NP, bt, i, pmin, span):
        grad_input.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](grad_output.ptr[unsafe_offset=idx])
    else:
        grad_input.ptr[unsafe_offset=idx] = Scalar[DT](0.0)


def _mae_grad_token_kernel[
    BATCH: Int, NP: Int, D: Int, SEED: UInt64
](
    grad_output: Tile[DT, BATCH, NP * D],
    grad_token: Tile[DT, D],
    base: UInt64, pmin: Scalar[DT], span: Scalar[DT],
):
    # One block per channel d; threads reduce over dropped (bt, i) patches.
    var d = Int(block_idx.x)
    if d >= D:
        return
    var t = Int(thread_idx.x)
    var acc: Scalar[DT] = 0.0
    var p = t
    while p < BATCH * NP:
        var bt = p // NP
        var i = p % NP
        if not _kept(SEED, base, BATCH, NP, bt, i, pmin, span):
            acc += rebind[Scalar[DT]](grad_output.ptr[unsafe_offset=(bt * NP + i) * D + d])
        p += MAE_RTPB
    var total = block.sum[block_size=MAE_RTPB, broadcast=False](val=acc)
    if t == 0:
        grad_token.ptr[unsafe_offset=d] = rebind[Scalar[DT]](grad_token.ptr[unsafe_offset=d]) + total[0]


struct MAEReplacer[
    NP: Int, D: Int, P_MIN: Float64, P_MAX: Float64, SEED: UInt64
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.NP * Self.D)
    comptime OUT_DIM = Self.NP * Self.D

    var mask_token: Param["mask_token", False, Self.D]
    var keep: Tensor  # [BATCH * NP] per-patch keep flags (1.0 kept / 0.0 drop)
    var rng_step: UInt64
    var p_min_rt: Float64
    var p_max_rt: Float64

    def __init__(out self):
        self.mask_token = Param["mask_token", False, Self.D]()
        self.keep = Tensor()
        self.rng_step = 0
        self.p_min_rt = Self.P_MIN
        self.p_max_rt = Self.P_MAX

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "MAEReplacer: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.mask_token = Param["mask_token", False, Self.D].make[target](ctx)
        INIT.init_bias[target](m.mask_token.val, Self.D, ctx)
        return m^

    def set_p(mut self, p_min: Float64, p_max: Float64):
        self.p_min_rt = p_min
        self.p_max_rt = p_max

    def advance_rng(mut self):
        self.rng_step += 1

    def mae_keep(ref self) -> ref [self.keep] Tensor:
        return self.keep

    def mae_mask_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Raw per-patch `keep` pointer for the masked-recon loss ABI — the
        device ptr when the keep Tensor lives on GPU, else the host ptr.

        Restores the legacy accessor that `Dreamer4Encoder` / `Dreamer4Tokenizer`
        forward to: the storage rename of this method to `mae_keep` (returning
        the `Tensor`) left that forwarding path uncompiled, since the recon-loss
        kernels (`masked_recon_loss` / `masked_recon_grad_gpu`) consume a raw
        target-resident pointer, not a `Tensor`."""
        if self.keep.dev:
            return rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                self.keep.dev.value().unsafe_ptr()
            )
        return rebind[Pointer[Scalar[DT], MutAnyOrigin]](
            self.keep.data.unsafe_ptr()
        )

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime STRIDE = UInt64(B * (1 + Self.NP))
        var base = self.rng_step * STRIDE
        var pmin = Scalar[DT](self.p_min_rt)
        var span = Scalar[DT](self.p_max_rt - self.p_min_rt)
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            self.keep.ensure(B * Self.NP)
            var inp = TileTensor(in0.data, row_major[B, Self.OUT_DIM]())
            var o_v = TileTensor(out.data, row_major[B, Self.OUT_DIM]())
            var mt = TileTensor(self.mask_token.val.data, row_major[Self.D]())
            for bt in range(B):
                for i in range(Self.NP):
                    var kept = _kept(
                        Self.SEED, base, B, Self.NP, bt, i, pmin, span
                    )
                    self.keep.data[bt * Self.NP + i] = (
                        Scalar[DT](1.0) if kept else Scalar[DT](0.0)
                    )
                    for d in range(Self.D):
                        if kept:
                            o_v[bt, i * Self.D + d] = inp[bt, i * Self.D + d]
                        else:
                            o_v[bt, i * Self.D + d] = mt[d]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            self.keep.ensure_gpu(c, B * Self.NP)
            comptime lbo = RowMajor[B, Self.NP * Self.D]
            comptime ld = RowMajor[Self.D]
            comptime lk = RowMajor[B * Self.NP]
            comptime nb = (B * Self.NP * Self.D + 127) // 128
            comptime kern = _mae_fwd_kernel[B, Self.NP, Self.D, Self.SEED]
            c.enqueue_function[kern](
                in0.tt["gpu", B, Self.NP * Self.D](),
                self.mask_token.val.tt["gpu", Self.D](),
                out.tt["gpu", B, Self.NP * Self.D](),
                self.keep.tt["gpu", B * Self.NP](),
                base, pmin, span,
                grid_dim=nb, block_dim=128,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime STRIDE = UInt64(B * (1 + Self.NP))
        var base = self.rng_step * STRIDE
        var pmin = Scalar[DT](self.p_min_rt)
        var span = Scalar[DT](self.p_max_rt - self.p_min_rt)
        comptime if target == "cpu":
            gin.ensure(B * Self.OUT_DIM)
            var go = TileTensor(grad_output.data, row_major[B, Self.OUT_DIM]())
            var gi = TileTensor(gin.data, row_major[B, Self.OUT_DIM]())
            var gmt = TileTensor(self.mask_token.grd.data, row_major[Self.D]())
            for bt in range(B):
                for i in range(Self.NP):
                    var kept = _kept(
                        Self.SEED, base, B, Self.NP, bt, i, pmin, span
                    )
                    for d in range(Self.D):
                        var g = go[bt, i * Self.D + d]
                        if kept:
                            gi[bt, i * Self.D + d] = g
                        else:
                            gi[bt, i * Self.D + d] = Scalar[DT](0.0)
                            gmt[d] += g
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.OUT_DIM)
            comptime lbo = RowMajor[B, Self.NP * Self.D]
            comptime ld = RowMajor[Self.D]
            comptime nb = (B * Self.NP * Self.D + 127) // 128
            comptime gik = _mae_grad_input_kernel[
                B, Self.NP, Self.D, Self.SEED
            ]
            c.enqueue_function[gik](
                grad_output.tt["gpu", B, Self.NP * Self.D](),
                gin.tt["gpu", B, Self.NP * Self.D](),
                base, pmin, span, grid_dim=nb, block_dim=128,
            )
            comptime gtk = _mae_grad_token_kernel[
                B, Self.NP, Self.D, Self.SEED
            ]
            c.enqueue_function[gtk](
                grad_output.tt["gpu", B, Self.NP * Self.D](),
                self.mask_token.grd.tt["gpu", Self.D](),
                base, pmin, span,
                grid_dim=Self.D, block_dim=MAE_RTPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `mask_token` Param field).

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.D](
            self.mask_token.val, src.mask_token.val, tau, ctx
        )
