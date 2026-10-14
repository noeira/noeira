"""Global grad-norm clipping over a Module (storage surface, CPU + GPU).

Two-pass walk via `model.for_each_param` (the storage ParamVisitor):
  1. accumulate ‖grad‖² across every Param,
  2. if ‖grad‖ > max_norm, scale every grad in place by max_norm / ‖grad‖.
Returns the pre-clip norm (diagnostics). `max_norm <= 0` is a no-op.

Non-finite norm (any NaN/±inf grad) → scale 0 → every grad is hard-zeroed
(`scale == 0` sentinel; a multiply would leave NaN·0 = NaN), so the optimizer
step becomes a no-op rather than poisoning the moments.

CPU: pure host loops over `grad.data`. GPU: per-Param block-reduction kernel
(comptime `N` layout) writes the param's Σg² into a reusable [1] device scalar,
D2H-accumulated on the host; then a per-Param scale kernel applies the host
scalar. This is storage-clean (no Pointer arrays, no runtime layouts) but
NOT CUDA-graph-capturable (the per-param D2H + host branch); the D2H-free
grouped version lands with the contiguous-arena optimizer (Phase D).

Caller convention: invoke AFTER all backward passes wrote into the params' grads
and BEFORE the optimizer update. Per-optimizer (per-model) clipping — no
cross-model global norm (matches the deep_agents convention).
"""

from std.math import sqrt
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor, ParamVisitorRT, walk_params
from ..core.param import ParamWalkable
from .param_arena import ParamArena


comptime GC_TPB: Int = 128  # single-block reduction width


def _sum_sq_kernel_rt(
    grad: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
    out_sum: Pointer[Scalar[DT], MutAnyOrigin],
):
    """Single-block sum of squares over `[0, n)`, runtime length: one
    kernel for every Param instead of one instantiation per size."""
    var n = Int(n_arg)
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < n:
        var g = grad[unsafe_offset=k]
        my_sum += g * g
        k += GC_TPB
    var total = block.sum[block_size=GC_TPB, broadcast=False](val=my_sum)
    if t == 0:
        out_sum[unsafe_offset=0] = total[0]


def _scale_kernel_rt(
    grad: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
    scale: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        var s = scale
        grad[unsafe_offset=i] = grad[unsafe_offset=i] * s if s != Scalar[DT](0.0) else Scalar[DT](0.0)


struct _SumSqCPU(ParamVisitor, ParamVisitorRT):
    var sum_sq: Scalar[DT]

    def __init__(out self):
        self.sum_sq = Scalar[DT](0.0)

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if grad.n == 0:  # frozen param: no gradient, nothing to do
            return
        for i in range(n):
            var g = grad.data[i]
            self.sum_sq += g * g

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
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
struct _ScaleCPU(ParamVisitor, ParamVisitorRT):
    var scale: Scalar[DT]

    def __init__(out self, scale: Scalar[DT]):
        self.scale = scale

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if grad.n == 0:  # frozen param: no gradient, nothing to do
            return
        if self.scale == Scalar[DT](0.0):
            for i in range(n):
                grad.data[i] = Scalar[DT](0.0)
        else:
            for i in range(n):
                grad.data[i] = grad.data[i] * self.scale

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
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
struct _SumSqGPU(ParamVisitor, ParamVisitorRT):
    var sum_sq: Scalar[DT]  # host accumulator across params
    var psum: Tensor  # reusable [1] device scalar

    def __init__(out self):
        self.sum_sq = Scalar[DT](0.0)
        self.psum = Tensor()

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if grad.n == 0:  # frozen param: no gradient, nothing to do
            return
        var c = ctx.value()
        self.psum.ensure_gpu(c, 1)
        c.enqueue_function[_sum_sq_kernel_rt](
            grad.dev.value(),
            Int64(n),
            self.psum.dev.value(),
            grid_dim=1,
            block_dim=GC_TPB,
        )
        self.psum.download(c)
        self.sum_sq += self.psum.data[0]

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
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


struct _ScaleGPU(ParamVisitor, ParamVisitorRT):
    var scale: Scalar[DT]

    def __init__(out self, scale: Scalar[DT]):
        self.scale = scale

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if grad.n == 0:  # frozen param: no gradient, nothing to do
            return
        var c = ctx.value()
        var nblk = (n + TPB - 1) // TPB
        c.enqueue_function[_scale_kernel_rt](
            grad.dev.value(),
            Int64(n),
            self.scale,
            grid_dim=nblk,
            block_dim=TPB,
        )

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
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


def _scale_from_norm(
    norm: Scalar[DT], max_norm: Scalar[DT], eps: Scalar[DT]
) -> Scalar[DT]:
    """`min(1, max_norm / max(norm, eps))`; non-finite norm → 0 (NaN guard)."""
    if norm - norm != Scalar[DT](0.0):  # True iff norm is non-finite
        return Scalar[DT](0.0)
    var denom = norm if norm > eps else eps
    var ratio = max_norm / denom
    return ratio if ratio < Scalar[DT](1.0) else Scalar[DT](1.0)


def clip_grad_norm[
    target: StaticString, M: ParamWalkable
](
    mut model: M,
    max_norm: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
    eps: Scalar[DT] = 1e-6,
) raises -> Scalar[DT]:
    """Clip the global L2 norm of `model`'s grads to `max_norm` in place.
    Returns the pre-clip norm. `max_norm <= 0` → no clip (norm still returned)."""
    var norm: Scalar[DT]
    comptime if target == "cpu":
        var ss = _SumSqCPU()
        walk_params[target](model, ss, ctx)
        norm = sqrt(ss.sum_sq)
        if max_norm > Scalar[DT](0.0):
            var sc = _ScaleCPU(_scale_from_norm(norm, max_norm, eps))
            if sc.scale < Scalar[DT](1.0):  # scale==1 → no-op skip
                walk_params[target](model, sc, ctx)
    else:
        var ss = _SumSqGPU()
        walk_params[target](model, ss, ctx)
        norm = sqrt(ss.sum_sq)
        if max_norm > Scalar[DT](0.0):
            var scale = _scale_from_norm(norm, max_norm, eps)
            if scale < Scalar[DT](1.0):
                var sc = _ScaleGPU(scale)
                walk_params[target](model, sc, ctx)
    return norm


# ──────────────────────────────────────────────────────────────────────
# Arena grad-clip — the capture-safe path over a ParamArena's contiguous
# grad buffer. THREE kernels, ZERO D2H during the clip (the per-param path
# above does a D2H per Param, so it can't be CUDA-graph-captured). Used by
# `Adam.clip_grads` / `SGD.clip_grads` when the optimizer is adopted.
# ──────────────────────────────────────────────────────────────────────


def _arena_sumsq_kernel(
    grd: Pointer[Scalar[DT], MutAnyOrigin],
    total_arg: Int64,
    partials: Pointer[Scalar[DT], MutAnyOrigin],
):
    """Flat grid over the grad arena: each block reduces its chunk of `g²` via
    block.sum; thread 0 writes the block total to `partials[block_idx]`."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var total = Int(total_arg)
    var flat = Int(global_idx.x)
    var my_sum: Scalar[DT] = 0.0
    if flat < total:
        var g = grd[unsafe_offset=flat]
        my_sum = g * g
    var tot = block.sum[block_size=TPB, broadcast=False](val=my_sum)
    if Int(thread_idx.x) == 0:
        partials[unsafe_offset=Int(block_idx.x)] = tot[0]


def _arena_finalize_kernel(
    partials: Pointer[Scalar[DT], MutAnyOrigin],
    n_blocks_arg: Int64,
    scale_buf: Pointer[Scalar[DT], MutAnyOrigin],
    norm_buf: Pointer[Scalar[DT], MutAnyOrigin],
    max_norm: Scalar[DT],
    eps: Scalar[DT],
):
    """Single-block reduction of the per-block partials → ‖grad‖, then
    `scale = min(1, max_norm/max(norm,eps))` (non-finite → 0). Writes both."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var n_blocks = Int(n_blocks_arg)
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < n_blocks:
        my_sum += partials[unsafe_offset=k]
        k += GC_TPB
    var s = block.sum[block_size=GC_TPB, broadcast=False](val=my_sum)
    if t == 0:
        var norm = sqrt(s[0])
        norm_buf[unsafe_offset=0] = norm
        if norm - norm != Scalar[DT](0.0):  # non-finite guard
            scale_buf[unsafe_offset=0] = Scalar[DT](0.0)
        elif max_norm <= Scalar[DT](0.0):
            scale_buf[unsafe_offset=0] = Scalar[DT](1.0)  # no clip
        else:
            var denom = norm if norm > eps else eps
            var ratio = max_norm / denom
            scale_buf[unsafe_offset=0] = ratio if ratio < Scalar[DT](1.0) else Scalar[DT](1.0)


def _arena_scale_kernel(
    grd: Pointer[Scalar[DT], MutAnyOrigin],
    total_arg: Int64,
    scale_buf: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`grd[i] *= scale_buf[0]` (scale 0 hard-zeroes — non-finite sentinel)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var total = Int(total_arg)
    var i = Int(global_idx.x)
    if i < total:
        var s = scale_buf[unsafe_offset=0]
        grd[unsafe_offset=i] = grd[unsafe_offset=i] * s if s != Scalar[DT](0.0) else Scalar[DT](0.0)


def _clip_arena_grads_kernels(
    mut arena: ParamArena,
    mut partials: Tensor,
    mut scale_buf: Tensor,
    mut norm_buf: Tensor,
    max_norm: Scalar[DT],
    ctx: DeviceContext,
    eps: Scalar[DT],
) raises:
    """The three on-device clip kernels (sumsq → finalize → scale), writing the
    pre-clip norm into `norm_buf` and the scale into `scale_buf`. NO allocation,
    NO D2H — every buffer is caller-owned and reused, so this whole sequence is
    CUDA-graph-capturable. `max_norm <= 0` → scale 1 (no clip)."""
    var total = arena.total
    if total == 0:
        return
    var nblk = (total + TPB - 1) // TPB
    ctx.enqueue_function[_arena_sumsq_kernel](
        arena.grd.dev.value(), Int64(total), partials.dev.value(),
        grid_dim=nblk, block_dim=TPB,
    )
    ctx.enqueue_function[_arena_finalize_kernel](
        partials.dev.value(), Int64(nblk), scale_buf.dev.value(),
        norm_buf.dev.value(), max_norm, eps,
        grid_dim=1, block_dim=GC_TPB,
    )
    ctx.enqueue_function[_arena_scale_kernel](
        arena.grd.dev.value(), Int64(total), scale_buf.dev.value(),
        grid_dim=nblk, block_dim=TPB,
    )


def clip_arena_grads_captured(
    mut arena: ParamArena,
    mut partials: Tensor,
    mut scale_buf: Tensor,
    mut norm_buf: Tensor,
    max_norm: Scalar[DT],
    ctx: DeviceContext,
    eps: Scalar[DT] = 1e-6,
) raises:
    """CUDA-graph-safe clip: caller-owned scratch (`partials` sized to the
    arena's block count, `scale_buf`/`norm_buf` size 1), NO allocation and NO
    D2H. The pre-clip norm is left in `norm_buf` on-device — read it at flush
    cadence (never per step) if you need the value. Use under capture instead of
    `clip_arena_grads` (which allocates + D2Hs each call → aborts in a capture
    region)."""
    _clip_arena_grads_kernels(
        arena, partials, scale_buf, norm_buf, max_norm, ctx, eps
    )


def clip_arena_grads(
    mut arena: ParamArena,
    max_norm: Scalar[DT],
    ctx: DeviceContext,
    eps: Scalar[DT] = 1e-6,
) raises -> Scalar[DT]:
    """Clip the global L2 norm of an adopted optimizer's contiguous grad arena in
    place; returns the pre-clip norm. Allocates scratch + one norm D2H per call —
    the convenience (NON-captured) path. Under CUDA-graph capture use
    `clip_arena_grads_captured` with persistent scratch instead. `max_norm <= 0`
    → no clip."""
    var total = arena.total
    if total == 0:
        return Scalar[DT](0.0)
    var nblk = (total + TPB - 1) // TPB
    var partials = Tensor.alloc_gpu(ctx, nblk)
    var scale_buf = Tensor.alloc_gpu(ctx, 1)
    var norm_buf = Tensor.alloc_gpu(ctx, 1)
    _clip_arena_grads_kernels(
        arena, partials, scale_buf, norm_buf, max_norm, ctx, eps
    )
    norm_buf.download(ctx)
    return norm_buf.data[0]
