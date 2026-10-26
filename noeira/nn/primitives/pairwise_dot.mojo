"""`PairwiseDot[D, BATCH]` and `RowDot[D]` — products of two ACTIVATIONS.

Every matmul elsewhere in `nn` is weight x activation. These two are not: both
inputs are activations, and both receive gradients. Forward-Backward
representations need exactly that, twice over —

    M = F(s,a,z) . B(s+)^T      the successor-measure matrix, [BATCH, BATCH]
    O = B(s)     . B(s+)^T      the orthonormality regulariser, same shape

so one primitive serves both loss terms of `docs/BFM_ZERO_SHOT_RL.md` §6
component 2. `RowDot` is its diagonal-only sibling, for the terms that pair
row i with row i and nothing else:

    a[i] = F(s_i,a_i,z_i) . B(s'_i)     the anchor term, [BATCH, 1]

⚠ **The anchor term is not a regulariser.** `-2·E[F(s,a,z)^T B(s')]` falls out
of expanding the square of the successor measure. Dropping it leaves a model
whose loss still descends and whose policy still moves, encoding nothing — see
§11, where it is ranked among the silent failures. `RowDot` exists so that term
is as cheap as it is meant to be: taking the diagonal of a `PairwiseDot` would
compute BATCH² dot products to keep BATCH of them.

Shapes and gradients:

    PairwiseDot   in ([B, D], [B, D]) -> out [B, B]
        M[i,j] = sum_k A[i,k]·C[j,k]
        dA     = G·C          i.e. dA[i,k] = sum_j G[i,j]·C[j,k]
        dC     = G^T·A        i.e. dC[j,k] = sum_i G[i,j]·A[i,k]

    RowDot        in ([B, D], [B, D]) -> out [B, 1]
        r[i]   = sum_k A[i,k]·C[i,k]
        dA[i,k] = G[i,0]·C[i,k];   dC[i,k] = G[i,0]·A[i,k]

Neither caches anything: the vjp reads `forward_input`, which the graph already
holds. That is why `MSEPerSample` — the closest existing 2-input leaf — carries
a `cache_diff` and these do not.

**`BATCH` is a compile-time parameter of `PairwiseDot`, not just of `forward`.**
The output row width IS the batch size, and `Module.OUT_DIM` is a struct-level
comptime Int, so it cannot be otherwise. `forward` asserts `B == BATCH` rather
than letting a mismatched combinator silently size a buffer to the wrong width.

⚠ Kernels are deliberately NAIVE — one thread per output element, an inner loop
over the contracted axis, no shared-memory tiling. This is not an oversight to
optimise away later: heavy blocked kernels hard-crash the Metal compiler on
Apple (see the project's GPU notes), and at the sizes involved the arithmetic
is not the cost. [1024, 1024] fp32 is 4 MB and roughly three are live at once.
Revisit only with a measurement, and only on NVIDIA.

⚠ `max_matmul` with `transpose_b` is NOT used here, and could not be: it
silently miscomputes at N=1. `PairwiseDot`'s N is BATCH, so a single-row
inference path reusing this primitive would land exactly on that bug.
"""

from noeira.nn.core.mm import mm, bmm
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile
from linalg.bmm import batched_matmul
from linalg.matmul import matmul as max_matmul

from noeira.nn.constants import DT, TPB


# ⚠⚠ The FORWARD dispatches to a GEMM. `M = A @ C^T` is the same operation
# `attention.mojo` computes for QK^T, so the library already had it — but WHICH
# entry point matters enormously, and only an on-target profile can tell you.
#
# MEASURED on NVIDIA, 3000 FB steps each, forward only, per step:
#
#     naive kernel     5 x 279.70 us  = 1.399 ms
#     batched_matmul  ~27 `matmul_kernel_naive`  = 2.46 ms   <- 1.75x WORSE
#     max_matmul       5 x   8.07 us  = 0.040 ms   <- 35x, `multistage_gemm`
#
# 35x, not the ~6x predicted from the nn Linears' 46 us `multistage_gemm` —
# that is a different (larger) shape. Whole-run GPU kernel totals: 14.38 s
# naive / 12.50 s batched / 10.25 s max_matmul.
#
# `batched_matmul` with a batch of ONE does not reach the tiled/tensor-core
# path: it fans out into ~5.4 `matmul_kernel_naive` launches per call and ends
# up 1.75x SLOWER than the hand kernel it replaced. `max_matmul` — the 2D entry
# point `linear.mojo` uses, which is where `multistage_gemm_kernel` comes from —
# is the right target when there is no batch axis.
#
# ⚠ On Apple the SAME `batched_matmul` swap measured 5.1x FASTER (2.39 ms ->
# 0.468 ms, 112 -> 573 GFLOP/s). The two platforms disagreed in DIRECTION, not
# just magnitude. An Apple benchmark cannot choose between these; profile on the
# box that runs the training.
#
# ⚠ This is NOT the `max_matmul` N=1 hazard the header warns about. That is
# about a silent miscompute when the output's N is 1; PairwiseDot's output is
# [BATCH, BATCH], so M and N are both BATCH and never 1.
#
# `PD_BATCH_DIM` is the rule rather than a hardcoded choice: a genuine batch
# axis (> 1) belongs in `batched_matmul`, which is what it is for; a batch of 1
# is a plain 2D GEMM and belongs in `max_matmul`. PairwiseDot contracts ONE
# [B, D] pair, so it is always 1 — but the dispatch states the reason instead of
# leaving the next reader to rediscover it.
#
# Override `PD_FORWARD` to profile a specific path; "naive" restores the hand
# kernel, kept because heavy blocked kernels have hard-crashed the Metal
# compiler on Apple before and a library GEMM is not automatically exempt.
comptime PD_BATCH_DIM: Int = 1
comptime PD_FORWARD: StaticString = "auto"
"""`"auto"` | `"max_matmul"` | `"batched_matmul"` | `"naive"`."""

comptime PD_FORWARD_RESOLVED: StaticString = PD_FORWARD if PD_FORWARD
    != "auto" else ("max_matmul" if PD_BATCH_DIM == 1 else "batched_matmul")
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# PairwiseDot kernels
# ──────────────────────────────────────────────────────────────────────


def _pd_forward_kernel[BATCH: Int, D: Int](
    a: Tile[DT, BATCH, D],
    c: Tile[DT, BATCH, D],
    m: Tile[DT, BATCH, BATCH],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * BATCH:
        return
    var i = idx // BATCH
    var j = idx % BATCH
    var s = Scalar[DT](0)
    for k in range(D):
        s += rebind[Scalar[DT]](a[i, k]) * rebind[Scalar[DT]](c[j, k])
    m[i, j] = s


def _pd_da_kernel[BATCH: Int, D: Int](
    g: Tile[DT, BATCH, BATCH],
    c: Tile[DT, BATCH, D],
    ga: Tile[DT, BATCH, D],
):
    # dA[i,k] = sum_j G[i,j]·C[j,k]
    var idx = Int(global_idx.x)
    if idx >= BATCH * D:
        return
    var i = idx // D
    var k = idx % D
    var s = Scalar[DT](0)
    for j in range(BATCH):
        s += rebind[Scalar[DT]](g[i, j]) * rebind[Scalar[DT]](c[j, k])
    ga[i, k] = s


def _pd_dc_kernel[BATCH: Int, D: Int](
    g: Tile[DT, BATCH, BATCH],
    a: Tile[DT, BATCH, D],
    gc: Tile[DT, BATCH, D],
):
    # dC[j,k] = sum_i G[i,j]·A[i,k]  — note G is read down its COLUMN j.
    var idx = Int(global_idx.x)
    if idx >= BATCH * D:
        return
    var j = idx // D
    var k = idx % D
    var s = Scalar[DT](0)
    for i in range(BATCH):
        s += rebind[Scalar[DT]](g[i, j]) * rebind[Scalar[DT]](a[i, k])
    gc[j, k] = s


struct PairwiseDot[D_: Int, BATCH: Int](Module):
    """`out[i, j] = sum_k A[i, k] · C[j, k]`, both inputs differentiable."""

    comptime ARITY = 2
    comptime IN_DIMS = Array[Int, 2](fill=Self.D_)
    comptime OUT_DIM = Self.BATCH

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime assert B == Self.BATCH, (
            "PairwiseDot: forward batch must equal the BATCH parameter — the"
            " output row width IS the batch size, so a mismatch would size"
            " every downstream buffer wrongly rather than merely process fewer"
            " rows"
        )
        ref a = inputs[0]
        ref c = inputs[1]
        comptime if target == "cpu":
            out.ensure(B * B)
            var a_t = TileTensor(a.data, row_major[B, Self.D_]())
            var c_t = TileTensor(c.data, row_major[B, Self.D_]())
            var m_t = TileTensor(out.data, row_major[B, B]())
            for i in range(B):
                for j in range(B):
                    var s = Scalar[DT](0)
                    for k in range(Self.D_):
                        s += a_t[i, k] * c_t[j, k]
                    m_t[i, j] = s
        else:
            var dc = ctx.value()
            out.ensure_gpu(dc, B * B)
            comptime if PD_FORWARD_RESOLVED == "max_matmul":
                # 2D `M = A @ C^T` — the entry point `linear.mojo` uses, which
                # is what dispatches to `multistage_gemm_kernel`.
                # ⚠ Bound to separate locals FIRST. `a` and `c` arrive from
                # one `TensorRefs[2, o]`, so they share an origin, and
                # `max_matmul` takes both mutably — passing
                # `a.dev.value()` / `c.dev.value()` inline trips the
                # exclusivity checker ("aliasing values passed mutably to 'a'
                # and 'b'"). `batched_matmul` accepted the inline form, which is
                # why this only appeared on the switch.
                var a_buf = a.dev.value()
                var c_buf = c.dev.value()
                var m_buf = out.dev.value()
                var m_v = TileTensor(m_buf, row_major[B, B]())
                var a_v = TileTensor(a_buf, row_major[B, Self.D_]())
                var c_v = TileTensor(c_buf, row_major[B, Self.D_]())
                max_matmul[transpose_b=True, target="gpu"](m_v, a_v, c_v, dc)
            elif PD_FORWARD_RESOLVED == "batched_matmul":
                # Kept for profiling and for the day this primitive grows a real
                # batch axis. At PD_BATCH_DIM == 1 it is measurably WORSE on
                # NVIDIA — see the note above.
                bmm[transpose_b=True, A0=PD_BATCH_DIM, A1=B, A2=Self.D_, B0=PD_BATCH_DIM, B1=B, B2=Self.D_, O0=PD_BATCH_DIM, O1=B, O2=B](
                    out.dev.value(), a.dev.value(), c.dev.value(), dc
                )
            else:
                comptime lay_in = RowMajor[B, Self.D_]
                comptime lay_m = RowMajor[B, B]
                comptime n_blocks = (B * B + TPB - 1) // TPB
                dc.enqueue_function[_pd_forward_kernel[B, Self.D_]](
                    a.tt["gpu", B, Self.D_](),
                    c.tt["gpu", B, Self.D_](),
                    out.tt["gpu", B, B](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime assert B == Self.BATCH, "PairwiseDot: vjp batch must equal BATCH"
        ref a = forward_input[0]
        ref c = forward_input[1]
        ref ga = grad_inputs[0]
        ref gc = grad_inputs[1]
        comptime if target == "cpu":
            ga.ensure(B * Self.D_)
            gc.ensure(B * Self.D_)
            var a_t = TileTensor(a.data, row_major[B, Self.D_]())
            var c_t = TileTensor(c.data, row_major[B, Self.D_]())
            var g_t = TileTensor(grad_output.data, row_major[B, B]())
            var ga_t = TileTensor(ga.data, row_major[B, Self.D_]())
            var gc_t = TileTensor(gc.data, row_major[B, Self.D_]())
            for i in range(B):
                for k in range(Self.D_):
                    var s = Scalar[DT](0)
                    for j in range(B):
                        s += g_t[i, j] * c_t[j, k]
                    ga_t[i, k] = s
            for j in range(B):
                for k in range(Self.D_):
                    var s = Scalar[DT](0)
                    for i in range(B):
                        s += g_t[i, j] * a_t[i, k]
                    gc_t[j, k] = s
        else:
            var dc = ctx.value()
            ga.ensure_gpu(dc, B * Self.D_)
            gc.ensure_gpu(dc, B * Self.D_)
            comptime lay_in = RowMajor[B, Self.D_]
            comptime lay_m = RowMajor[B, B]
            comptime n_blocks = (B * Self.D_ + TPB - 1) // TPB
            dc.enqueue_function[_pd_da_kernel[B, Self.D_]](
                grad_output.tt["gpu", B, B](),
                c.tt["gpu", B, Self.D_](),
                ga.tt["gpu", B, Self.D_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )
            dc.enqueue_function[_pd_dc_kernel[B, Self.D_]](
                grad_output.tt["gpu", B, B](),
                a.tt["gpu", B, Self.D_](),
                gc.tt["gpu", B, Self.D_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (param-less leaf -> no-op).


# ──────────────────────────────────────────────────────────────────────
# RowDot kernels
# ──────────────────────────────────────────────────────────────────────


def _rd_forward_kernel[BATCH: Int, D: Int](
    a: Tile[DT, BATCH, D],
    c: Tile[DT, BATCH, D],
    o: Tile[DT, BATCH, 1],
):
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var s = Scalar[DT](0)
    for k in range(D):
        s += rebind[Scalar[DT]](a[i, k]) * rebind[Scalar[DT]](c[i, k])
    o[i, 0] = s


def _rd_backward_kernel[BATCH: Int, D: Int](
    g: Tile[DT, BATCH, 1],
    a: Tile[DT, BATCH, D],
    c: Tile[DT, BATCH, D],
    ga: Tile[DT, BATCH, D],
    gc: Tile[DT, BATCH, D],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * D:
        return
    var i = idx // D
    var k = idx % D
    var go = rebind[Scalar[DT]](g[i, 0])
    ga[i, k] = go * rebind[Scalar[DT]](c[i, k])
    gc[i, k] = go * rebind[Scalar[DT]](a[i, k])


struct RowDot[D_: Int](Module):
    """`out[i, 0] = sum_k A[i, k] · C[i, k]` — the diagonal of `PairwiseDot`,
    computed without the other BATCH²-BATCH entries."""

    comptime ARITY = 2
    comptime IN_DIMS = Array[Int, 2](fill=Self.D_)
    comptime OUT_DIM = 1

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref a = inputs[0]
        ref c = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            var a_t = TileTensor(a.data, row_major[B, Self.D_]())
            var c_t = TileTensor(c.data, row_major[B, Self.D_]())
            var o_t = TileTensor(out.data, row_major[B, 1]())
            for i in range(B):
                var s = Scalar[DT](0)
                for k in range(Self.D_):
                    s += a_t[i, k] * c_t[i, k]
                o_t[i, 0] = s
        else:
            var dc = ctx.value()
            out.ensure_gpu(dc, B)
            comptime lay_in = RowMajor[B, Self.D_]
            comptime lay_o = RowMajor[B, 1]
            comptime n_blocks = (B + TPB - 1) // TPB
            dc.enqueue_function[_rd_forward_kernel[B, Self.D_]](
                a.tt["gpu", B, Self.D_](),
                c.tt["gpu", B, Self.D_](),
                out.tt["gpu", B, 1](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref a = forward_input[0]
        ref c = forward_input[1]
        ref ga = grad_inputs[0]
        ref gc = grad_inputs[1]
        comptime if target == "cpu":
            ga.ensure(B * Self.D_)
            gc.ensure(B * Self.D_)
            var a_t = TileTensor(a.data, row_major[B, Self.D_]())
            var c_t = TileTensor(c.data, row_major[B, Self.D_]())
            var g_t = TileTensor(grad_output.data, row_major[B, 1]())
            var ga_t = TileTensor(ga.data, row_major[B, Self.D_]())
            var gc_t = TileTensor(gc.data, row_major[B, Self.D_]())
            for i in range(B):
                var go = g_t[i, 0]
                for k in range(Self.D_):
                    ga_t[i, k] = go * c_t[i, k]
                    gc_t[i, k] = go * a_t[i, k]
        else:
            var dc = ctx.value()
            ga.ensure_gpu(dc, B * Self.D_)
            gc.ensure_gpu(dc, B * Self.D_)
            comptime lay_in = RowMajor[B, Self.D_]
            comptime lay_o = RowMajor[B, 1]
            comptime n_blocks = (B * Self.D_ + TPB - 1) // TPB
            dc.enqueue_function[_rd_backward_kernel[B, Self.D_]](
                grad_output.tt["gpu", B, 1](),
                a.tt["gpu", B, Self.D_](),
                c.tt["gpu", B, Self.D_](),
                ga.tt["gpu", B, Self.D_](),
                gc.tt["gpu", B, Self.D_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )
