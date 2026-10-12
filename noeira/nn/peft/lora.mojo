"""LoRA: a low-rank adapter beside a frozen linear.

    y = x W + b + s · (x A) B        A [IN, R]   B [R, OUT]   s = α / R
                                    (rsLoRA: s = α / √R)

LoRA is additive, so the adapter does not own the layer it adapts.
`LoRAAdapter` is the side module: `forward_add` adds `s · (x A) B` into the
base layer's output, and `vjp_add` forms dA and dB and adds the adapter's
share of the input gradient onto the base layer's. SmolVLA's hand-written
layer loops call it after each base `forward` / `vjp`. `LoRALinear` packages
the adapter with a frozen `Linear` as a `Module`, for models composed from
modules.

## Orientation

noeira stores a linear's weight as [IN, OUT] (`y = x W`), so A is [IN, R]
and B is [R, OUT]. PEFT stores `lora_A.weight` as [R, IN] and
`lora_B.weight` as [OUT, R], torch's [out, in]: the PEFT reader and writer
transpose, as the SmolVLA loader does for `Linear`.

## Initialisation

PEFT's default: kaiming_uniform(a = √5) on the [R, IN] weight of A, which is
U(-1/√IN, 1/√IN), and B = 0, so the adapter starts as an exact no-op.

## Rank padding

The params hold `R_PAD = round_up(R, PAD_R)` columns of A and rows of B: the
logical R first, zeros after. MAX's GEMM on Metal is ~4x slower when a
contraction dim is not a multiple of 16 (the Q0 measurement), R is the
contraction of `h B` and of `gh Aᵀ`, and 8 is a common rank. The pad is exact
and STAYS zero under training: a pad column of A meets only pad rows of B, so
`h_pad = x A_pad = 0`, `dB_pad = s hᵀ_pad g = 0` and `dA_pad = xᵀ (s g
B_padᵀ) = 0`, and Adam, SGD and decoupled weight decay leave a zero parameter
with a zero gradient at zero. The scale uses the logical R.

`PAD_R` is a type parameter, not a per-backend constant, because the padded
sizes are what a checkpoint stores: a checkpoint written on one backend must
load on another. The PEFT files carry the logical R.

## The backward recomputes h = x A

instead of keeping it from the forward. That costs one GEMM R / OUT the size
of the base layer's, and it frees the adapter from pairing each vjp with the
forward before it (activation checkpointing replays forwards, and an expert
can run more than once per step).

## Merging

`merge_into(W)` adds `s · A B` to the base weight in fp32 and bumps the
weight's version, which `Linear`'s padded and bf16 weight copies are gated on;
`unmerge_from(W)` subtracts it. A merged adapter lives inside W, so
`forward_add` and `vjp_add` skip it, and `vjp_add` refuses a merged adapter
that still has gradients: it would train nothing.

## Paths

CPU: one `cblas_sgemm` per product on macOS, whose alpha and beta carry the
scale and the accumulation; plain loops elsewhere. GPU: MAX's matmul for the
products, small kernels for the scale, the accumulations and the transposes
of the two [T, R_PAD] matrices. xᵀ is never formed: dA is formed as
(ghᵀ x)ᵀ, which is R_PAD × IN.

⚠ On NVIDIA this generic path runs MAX's matmul with N = R_PAD, which MAX
sends to its allocating vendor fallback (`Linear`'s N-padding notes). A
cuBLAS path, whose alpha and beta also remove every elementwise kernel, is
the NVIDIA follow-up. fp32 only: the bf16-flow adapter comes with QLoRA.
"""

from std.math import sqrt
from std.random import random_float64
from std.sys import CompilationTarget
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from noeira.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.walkers import join_name
from ..core.mm import mm
from ..core.polyak import polyak_tensor
from ..primitives.linear import Linear


# ── kernels ────────────────────────────────────────────────────────────


def _axpy_kernel(
    y: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    s: Scalar[DT],
    n_arg: Int64,
):
    """`y[i] += s · x[i]`."""
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        y[unsafe_offset=i] += s * x[unsafe_offset=i]


def _scale_kernel(
    x: Pointer[Scalar[DT], MutAnyOrigin], s: Scalar[DT], n_arg: Int64
):
    """`x[i] = s · x[i]`."""
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        x[unsafe_offset=i] = s * x[unsafe_offset=i]


def _transpose_kernel(
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
):
    """`dst[c, r] = src[r, c]` over a row-major [rows, cols] `src`, one
    thread per element. Only ever given [T, R_PAD] matrices."""
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var i = Int(global_idx.x)
    if i < rows * cols:
        var r = i // cols
        var c = i % cols
        dst[unsafe_offset=c * rows + r] = src[unsafe_offset=i]


def _accum_transposed_kernel(
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    src: Pointer[Scalar[DT], MutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
):
    """`dst[r, c] += src[c, r]` over a row-major [rows, cols] `dst`."""
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var i = Int(global_idx.x)
    if i < rows * cols:
        var r = i // cols
        var c = i % cols
        dst[unsafe_offset=i] += src[unsafe_offset=c * rows + r]


@always_inline
def _blocks(n: Int) -> Int:
    return (n + 255) // 256


# ── CPU GEMM ───────────────────────────────────────────────────────────


def _gemm_cpu[
    TA: Bool, TB: Bool
](
    M: Int,
    N: Int,
    K: Int,
    alpha: Float32,
    a: Tensor,
    lda: Int,
    b: Tensor,
    ldb: Int,
    beta: Float32,
    mut c: Tensor,
    ldc: Int,
) raises:
    """`c[M, N] = alpha · op(a) @ op(b) + beta · c`, row-major fp32: op(a) is
    a[M, K] (row stride `lda`), or a[K, M]ᵀ when `TA`; op(b) is b[K, N], or
    b[N, K]ᵀ when `TB`. As in BLAS, `c` is not read when beta is 0."""
    comptime assert DT == DType.float32, "LoRA's CPU GEMM is fp32"
    comptime if CompilationTarget.is_macos():
        var cblas = get_cblas_f32_function()
        cblas(
            _CBLASOrder.ROW_MAJOR,
            _CBLASTranspose.TRANSPOSE if TA else _CBLASTranspose.NO_TRANSPOSE,
            _CBLASTranspose.TRANSPOSE if TB else _CBLASTranspose.NO_TRANSPOSE,
            Int32(M),
            Int32(N),
            Int32(K),
            alpha,
            rebind[Pointer[Float32, ImmutAnyOrigin]](a.data.unsafe_ptr()),
            Int32(lda),
            rebind[Pointer[Float32, ImmutAnyOrigin]](b.data.unsafe_ptr()),
            Int32(ldb),
            beta,
            rebind[Pointer[Float32, MutAnyOrigin]](c.data.unsafe_ptr()),
            Int32(ldc),
        )
    else:
        for i in range(M):
            for j in range(N):
                var acc = Float32(0)
                for k in range(K):
                    var av = a.data[k * lda + i] if TA else a.data[i * lda + k]
                    var bv = b.data[j * ldb + k] if TB else b.data[k * ldb + j]
                    acc += av * bv
                var prev = Float32(0)
                if beta != 0:
                    prev = beta * c.data[i * ldc + j]
                c.data[i * ldc + j] = alpha * acc + prev


# ── the adapter ────────────────────────────────────────────────────────


struct LoRAAdapter[
    IN: Int, OUT: Int, R: Int, PAD_R: Int = 16, RSLORA: Bool = False
](Movable):
    """`s · (x A) B` beside a linear [IN -> OUT] (module docstring)."""

    comptime R_PAD = ((Self.R + Self.PAD_R - 1) // Self.PAD_R) * Self.PAD_R
    comptime A_SIZE = Self.IN * Self.R_PAD
    comptime B_SIZE = Self.R_PAD * Self.OUT

    var lora_A: Param["lora_A", True, Self.A_SIZE]
    """[IN, R_PAD]: the logical [IN, R] block, then zero columns."""
    var lora_B: Param["lora_B", True, Self.B_SIZE]
    """[R_PAD, OUT]: the logical [R, OUT] block, then zero rows."""
    var alpha: Float64
    """PEFT's `lora_alpha`. The scale is α / R, or α / √R with `RSLORA`."""
    var merged: Bool
    """True while `s · A B` is inside the base weight (`merge_into`)."""
    # Scratch, lazily sized; `release_buffers` drops it.
    var h: Tensor
    """[T, R_PAD]: x A."""
    var gh: Tensor
    """[T, R_PAD]: s · g Bᵀ."""
    var hT: Tensor
    """[R_PAD, T] (GPU)."""
    var ghT: Tensor
    """[R_PAD, T] (GPU)."""
    var dAT: Tensor
    """[R_PAD, IN] (GPU): ghᵀ x, added transposed into dA."""
    var dB_tmp: Tensor
    """[R_PAD, OUT] (GPU): hᵀ g."""
    var prod: Tensor
    """GPU product before it is added in: [T, OUT] (forward), [T, IN] (vjp),
    [IN, OUT] (merge). MAX's matmul has no beta."""

    def __init__(out self):
        comptime assert Self.R >= 1, "LoRA rank must be at least 1"
        comptime assert Self.PAD_R >= 1, "PAD_R must be at least 1"
        self.lora_A = Param["lora_A", True, Self.A_SIZE]()
        self.lora_B = Param["lora_B", True, Self.B_SIZE]()
        self.alpha = Float64(Self.R)
        self.merged = False
        self.h = Tensor()
        self.gh = Tensor()
        self.hT = Tensor()
        self.ghT = Tensor()
        self.dAT = Tensor()
        self.dB_tmp = Tensor()
        self.prod = Tensor()

    @staticmethod
    def make[
        target: StaticString
    ](
        ctx: Optional[DeviceContext] = None,
        alpha: Float64 = Float64(Self.R),
    ) raises -> Self:
        """PEFT's initialisation: A ~ U(-1/√IN, 1/√IN) on the logical block
        (host RNG), its pad columns and all of B zero. α defaults to R, a
        scale of 1."""
        var a = Self()
        a.alpha = alpha
        a.lora_A = Param["lora_A", True, Self.A_SIZE].make[target](ctx)
        a.lora_B = Param["lora_B", True, Self.B_SIZE].make[target](ctx)
        var bound = 1.0 / sqrt(Float64(Self.IN))
        for i in range(Self.IN):
            for j in range(Self.R):
                a.lora_A.val.data[i * Self.R_PAD + j] = Scalar[DT](
                    (random_float64() * 2.0 - 1.0) * bound
                )
        comptime if target == "gpu":
            a.lora_A.val.upload(ctx.value())
            a.lora_B.val.upload(ctx.value())
        return a^

    def scale(self) -> Float64:
        comptime if Self.RSLORA:
            return self.alpha / sqrt(Float64(Self.R))
        else:
            return self.alpha / Float64(Self.R)

    # ── walks ──────────────────────────────────────────────────────────
    def walk[
        target: StaticString, V: ParamVisitor
    ](mut self, mut v: V, ctx: Optional[DeviceContext], prefix: String) raises:
        self.lora_A.visit_with[target](v, ctx, join_name(prefix, String("lora_A")))
        self.lora_B.visit_with[target](v, ctx, join_name(prefix, String("lora_B")))

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """⚠ The SAME two params as `walk`."""
        self.lora_A.zero_grad[target](ctx)
        self.lora_B.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        polyak_tensor[target, Self.A_SIZE](
            self.lora_A.val, src.lora_A.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](
            self.lora_B.val, src.lora_B.val, tau, ctx
        )

    def release_buffers(mut self):
        self.h.release()
        self.gh.release()
        self.hT.release()
        self.ghT.release()
        self.dAT.release()
        self.dB_tmp.release()
        self.prod.release()

    # ── frozen adapters (inference) ────────────────────────────────────
    def freeze(mut self):
        """Release both gradients and moments: `vjp_add` then forms the
        input gradient only."""
        self.lora_A.freeze()
        self.lora_B.freeze()
        self.dAT.release()
        self.dB_tmp.release()
        self.hT.release()
        self.ghT.release()

    def unfreeze[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext] = None) raises:
        self.lora_A.unfreeze[target](ctx)
        self.lora_B.unfreeze[target](ctx)

    def is_frozen(self) -> Bool:
        return self.lora_A.is_frozen() and self.lora_B.is_frozen()

    # ── forward / vjp ──────────────────────────────────────────────────
    def forward_add[
        target: StaticString, T: Int
    ](
        mut self,
        x: Tensor,
        mut y: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`y[T, OUT] += s · (x[T, IN] A) B`. Nothing once merged."""
        if self.merged:
            return
        var s = Scalar[DT](self.scale())
        comptime if target == "cpu":
            self.h.ensure(T * Self.R_PAD)
            _gemm_cpu[False, False](
                T, Self.R_PAD, Self.IN, 1.0,
                x, Self.IN, self.lora_A.val, Self.R_PAD,
                0.0, self.h, Self.R_PAD,
            )
            _gemm_cpu[False, False](
                T, Self.OUT, Self.R_PAD, s,
                self.h, Self.R_PAD, self.lora_B.val, Self.OUT,
                1.0, y, Self.OUT,
            )
        else:
            var c = ctx.value()
            self.h.ensure_gpu(c, T * Self.R_PAD)
            self.prod.ensure_gpu(c, T * Self.OUT)
            mm[A0=T, A1=Self.IN, B0=Self.IN, B1=Self.R_PAD, O0=T, O1=Self.R_PAD](
                self.h.dev.value(), x.dev.value(), self.lora_A.val.dev.value(), c
            )
            mm[A0=T, A1=Self.R_PAD, B0=Self.R_PAD, B1=Self.OUT, O0=T, O1=Self.OUT](
                self.prod.dev.value(), self.h.dev.value(),
                self.lora_B.val.dev.value(), c,
            )
            c.enqueue_function[_axpy_kernel](
                y.dev.value(),
                self.prod.dev.value(),
                s,
                Int64(T * Self.OUT),
                grid_dim=_blocks(T * Self.OUT),
                block_dim=256,
            )

    def vjp_add[
        target: StaticString, T: Int
    ](
        mut self,
        x: Tensor,
        g: Tensor,
        mut gx: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """With gh = s · g Bᵀ: dB += s · hᵀ g, dA += xᵀ gh, and
        `gx[T, IN] += gh Aᵀ`. `g` is the base layer's grad_output and `gx`
        already holds its grad_input (`Linear.vjp` overwrites it), which the
        adapter's share is added onto. A frozen adapter forms gx only."""
        var want_dA = self.lora_A.grd.n > 0
        var want_dB = self.lora_B.grd.n > 0
        if self.merged:
            if want_dA or want_dB:
                raise Error(
                    "LoRAAdapter.vjp_add: the adapter is merged into its base"
                    " weight and would train nothing; unmerge it, or freeze it"
                    " for inference"
                )
            return
        var s = Scalar[DT](self.scale())
        comptime if target == "cpu":
            self.gh.ensure(T * Self.R_PAD)
            if want_dB:
                self.h.ensure(T * Self.R_PAD)
                _gemm_cpu[False, False](
                    T, Self.R_PAD, Self.IN, 1.0,
                    x, Self.IN, self.lora_A.val, Self.R_PAD,
                    0.0, self.h, Self.R_PAD,
                )
                _gemm_cpu[True, False](
                    Self.R_PAD, Self.OUT, T, s,
                    self.h, Self.R_PAD, g, Self.OUT,
                    1.0, self.lora_B.grd, Self.OUT,
                )
            _gemm_cpu[False, True](
                T, Self.R_PAD, Self.OUT, s,
                g, Self.OUT, self.lora_B.val, Self.OUT,
                0.0, self.gh, Self.R_PAD,
            )
            if want_dA:
                _gemm_cpu[True, False](
                    Self.IN, Self.R_PAD, T, 1.0,
                    x, Self.IN, self.gh, Self.R_PAD,
                    1.0, self.lora_A.grd, Self.R_PAD,
                )
            _gemm_cpu[False, True](
                T, Self.IN, Self.R_PAD, 1.0,
                self.gh, Self.R_PAD, self.lora_A.val, Self.R_PAD,
                1.0, gx, Self.IN,
            )
        else:
            var c = ctx.value()
            if want_dB:
                self.h.ensure_gpu(c, T * Self.R_PAD)
                self.hT.ensure_gpu(c, Self.R_PAD * T)
                self.dB_tmp.ensure_gpu(c, Self.B_SIZE)
                mm[A0=T, A1=Self.IN, B0=Self.IN, B1=Self.R_PAD, O0=T, O1=Self.R_PAD](
                    self.h.dev.value(), x.dev.value(),
                    self.lora_A.val.dev.value(), c,
                )
                c.enqueue_function[_transpose_kernel](
                    self.h.dev.value(),
                    self.hT.dev.value(),
                    Int64(T),
                    Int64(Self.R_PAD),
                    grid_dim=_blocks(T * Self.R_PAD),
                    block_dim=256,
                )
                mm[A0=Self.R_PAD, A1=T, B0=T, B1=Self.OUT, O0=Self.R_PAD, O1=Self.OUT](
                    self.dB_tmp.dev.value(), self.hT.dev.value(), g.dev.value(), c
                )
                c.enqueue_function[_axpy_kernel](
                    self.lora_B.grd.dev.value(),
                    self.dB_tmp.dev.value(),
                    s,
                    Int64(Self.B_SIZE),
                    grid_dim=_blocks(Self.B_SIZE),
                    block_dim=256,
                )
            self.gh.ensure_gpu(c, T * Self.R_PAD)
            mm[
                transpose_b=True,
                A0=T, A1=Self.OUT, B0=Self.R_PAD, B1=Self.OUT, O0=T, O1=Self.R_PAD,
            ](self.gh.dev.value(), g.dev.value(), self.lora_B.val.dev.value(), c)
            c.enqueue_function[_scale_kernel](
                self.gh.dev.value(),
                s,
                Int64(T * Self.R_PAD),
                grid_dim=_blocks(T * Self.R_PAD),
                block_dim=256,
            )
            if want_dA:
                self.ghT.ensure_gpu(c, Self.R_PAD * T)
                self.dAT.ensure_gpu(c, Self.A_SIZE)
                c.enqueue_function[_transpose_kernel](
                    self.gh.dev.value(),
                    self.ghT.dev.value(),
                    Int64(T),
                    Int64(Self.R_PAD),
                    grid_dim=_blocks(T * Self.R_PAD),
                    block_dim=256,
                )
                mm[A0=Self.R_PAD, A1=T, B0=T, B1=Self.IN, O0=Self.R_PAD, O1=Self.IN](
                    self.dAT.dev.value(), self.ghT.dev.value(), x.dev.value(), c
                )
                c.enqueue_function[_accum_transposed_kernel](
                    self.lora_A.grd.dev.value(),
                    self.dAT.dev.value(),
                    Int64(Self.IN),
                    Int64(Self.R_PAD),
                    grid_dim=_blocks(Self.A_SIZE),
                    block_dim=256,
                )
            self.prod.ensure_gpu(c, T * Self.IN)
            mm[
                transpose_b=True,
                A0=T, A1=Self.R_PAD, B0=Self.IN, B1=Self.R_PAD, O0=T, O1=Self.IN,
            ](self.prod.dev.value(), self.gh.dev.value(), self.lora_A.val.dev.value(), c)
            c.enqueue_function[_axpy_kernel](
                gx.dev.value(),
                self.prod.dev.value(),
                Scalar[DT](1.0),
                Int64(T * Self.IN),
                grid_dim=_blocks(T * Self.IN),
                block_dim=256,
            )

    # ── merge ──────────────────────────────────────────────────────────
    def merge_into[
        target: StaticString
    ](mut self, mut w: Tensor, ctx: Optional[DeviceContext] = None) raises:
        """`w[IN, OUT] += s · A B` (the base `Linear`'s `weight.val`)."""
        if self.merged:
            raise Error("LoRAAdapter.merge_into: already merged")
        self._add_delta[target](w, Scalar[DT](self.scale()), ctx)
        self.merged = True

    def unmerge_from[
        target: StaticString
    ](mut self, mut w: Tensor, ctx: Optional[DeviceContext] = None) raises:
        """`w[IN, OUT] -= s · A B`: `merge_into`'s inverse, up to fp32
        rounding."""
        if not self.merged:
            raise Error("LoRAAdapter.unmerge_from: not merged")
        self._add_delta[target](w, -Scalar[DT](self.scale()), ctx)
        self.merged = False

    def _add_delta[
        target: StaticString
    ](
        mut self, mut w: Tensor, s: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        """`w += s · A B`, then a version bump: `Linear`'s padded and bf16
        weight copies are rebuilt only when `weight.val.version` moves."""
        comptime if target == "cpu":
            _gemm_cpu[False, False](
                Self.IN, Self.OUT, Self.R_PAD, s,
                self.lora_A.val, Self.R_PAD, self.lora_B.val, Self.OUT,
                1.0, w, Self.OUT,
            )
        else:
            var c = ctx.value()
            self.prod.ensure_gpu(c, Self.IN * Self.OUT)
            mm[A0=Self.IN, A1=Self.R_PAD, B0=Self.R_PAD, B1=Self.OUT, O0=Self.IN, O1=Self.OUT](
                self.prod.dev.value(), self.lora_A.val.dev.value(),
                self.lora_B.val.dev.value(), c,
            )
            c.enqueue_function[_axpy_kernel](
                w.dev.value(),
                self.prod.dev.value(),
                s,
                Int64(Self.IN * Self.OUT),
                grid_dim=_blocks(Self.IN * Self.OUT),
                block_dim=256,
            )
        w.version += 1


# ── LoRALinear: a frozen Linear and its adapter, as a Module ────────────


struct LoRALinear[
    IN_: Int, OUT_: Int, R: Int, PAD_R: Int = 16, RSLORA: Bool = False
](Module):
    """`Linear[IN_, OUT_]` (frozen by `make`) + `LoRAAdapter`, for models
    composed from modules. The params walk as the `Linear`'s (`weight`,
    `bias`, same names, same prefix) followed by `lora_A` and `lora_B`, so
    the frozen base keeps the names a checkpoint or a name map gave it."""

    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime Base = Linear[Self.IN_, Self.OUT_]
    comptime Adapter = LoRAAdapter[
        Self.IN_, Self.OUT_, Self.R, Self.PAD_R, Self.RSLORA
    ]

    var base: Self.Base
    var adapter: Self.Adapter

    def __init__(out self):
        self.base = Self.Base()
        self.adapter = Self.Adapter()

    @staticmethod
    def display_label() -> String:
        return String("LoRALinear")

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """The base initialised with `INIT` and then FROZEN, as PEFT freezes
        the layers it adapts; the adapter at PEFT's initialisation with
        α = R (`set_attr["lora_alpha"]` changes it). Load base weights into
        `base` as into any `Linear`: a frozen param keeps its value."""
        var l = Self()
        l.base = Self.Base.make[target, INIT](ctx)
        l.base.freeze()
        l.adapter = Self.Adapter.make[target](ctx)
        return l^

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "lora_alpha":
            self.adapter.alpha = Float64(value)
        else:
            self.base.set_attr[ATTR](value)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.base.forward[target, B, o, POLICY](inputs, out, ctx)
        ref x = inputs[0]
        self.adapter.forward_add[target, B](x, out, ctx)

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
        self.base.vjp[target, B, ofi, ogi, POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )
        ref x = forward_input[0]
        ref gx = grad_inputs[0]
        self.adapter.vjp_add[target, B](x, grad_output, gx, ctx)

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.base.for_each_param[target](visitor, ctx, prefix)
        self.adapter.walk[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """⚠ The SAME params as `for_each_param`. A frozen base has nothing
        to zero."""
        self.base.zero_grad[target](ctx)
        self.adapter.zero_grad[target](ctx)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.base.for_each_state[target](visitor, ctx, prefix)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.base.polyak_from[target](src.base, tau, ctx)
        self.adapter.polyak_from[target](src.adapter, tau, ctx)

    def release_buffers(mut self):
        self.base.release_buffers()
        self.adapter.release_buffers()

    def merge[target: StaticString](mut self, ctx: Optional[DeviceContext] = None) raises:
        """Fold the adapter into the base weight: the forward then runs the
        base GEMM alone."""
        self.adapter.merge_into[target](self.base.weight.val, ctx)

    def unmerge[target: StaticString](mut self, ctx: Optional[DeviceContext] = None) raises:
        self.adapter.unmerge_from[target](self.base.weight.val, ctx)
