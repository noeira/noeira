"""Adam — storage-native Adam / AdamW optimizer (CPU + GPU).

A `ParamVisitor`, but stateful: per-param 1st/2nd moments. Two GPU modes, ONE
class, IDENTICAL math (bit-parity, gated):

  - per-param (default, CPU + GPU): `step` walks `for_each_param`, one kernel /
    CPU loop per Param. The universal correctness path + CPU↔GPU parity check.
    Moments live on the `Param` (`m`/`v` Tensors), lazily zero-allocated.
  - arena (GPU, opt-in via `adopt`): a shared `ParamArena` packs params into
    contiguous val/grd buffers (+ this optimizer's own contiguous m/v arenas);
    `step` is ONE flat kernel over `[0,total)`, collapsing N launches → 1. Runs on
    Apple AND NVIDIA. `adopt` is a NO-OP on CPU → agent code is target-agnostic:

        opt.adopt[target](model, ctx); opt.step[target](model, ctx)

Decoupled weight decay (`wd > 0`, gated by `APPLY_DECAY` / the arena `decay_mask`)
makes this AdamW. Lifetime: the arena/moments are optimizer-owned; param slices
reference them (DeviceBuffer is refcounted → destruction order is safe).
"""

from std.sys import simd_width_of
from std.math import sqrt
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor, ParamVersionBump, ParamVisitorRT, ParamVisitorRef, walk_params
from ..core.param import ParamWalkable
from ..core.checkpoint import CheckpointScalars
from .param_arena import ParamArena, align_param_off
from .grad_clip import (
    clip_grad_norm, clip_arena_grads, clip_arena_grads_captured,
)
from .optimizer import Optimizer


def _adam_update_kernel_rt(
    param: Pointer[Scalar[DT], MutAnyOrigin],
    grad: Pointer[Scalar[DT], MutAnyOrigin],
    m: Pointer[Scalar[DT], MutAnyOrigin],
    v: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
    wd: Scalar[DT],
    apply_decay_arg: Int64,
):
    """`_adam_update_kernel` with the length at RUNTIME: one kernel for every
    Param instead of one instantiation per size. Reached through
    `ParamVisitorRef` (`visit_rt`)."""
    var apply_decay = Int(apply_decay_arg)
    var i = Int(global_idx.x)
    if i >= Int(n_arg):
        return
    var one = Scalar[DT](1.0)
    var p = param[unsafe_offset=i]
    if apply_decay != 0:
        p -= lr * wd * p
    var g = grad[unsafe_offset=i]
    var m_new = beta1 * m[unsafe_offset=i] + (one - beta1) * g
    var v_new = beta2 * v[unsafe_offset=i] + (one - beta2) * g * g
    m[unsafe_offset=i] = m_new
    v[unsafe_offset=i] = v_new
    var m_hat = m_new / bc1
    var v_hat = v_new / bc2
    param[unsafe_offset=i] = p - lr * m_hat / (sqrt(v_hat) + eps)


def _adam_advance_pow_kernel(
    powbuf: Pointer[Scalar[DT], MutAnyOrigin],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
):
    """Advance the device-resident bias-correction powers `[β₁ᵗ, β₂ᵗ]` (1
    thread). β^t lives on-device so it advances on every CUDA-graph REPLAY —
    a host-baked `bc` scalar would freeze at the capture-time step, scaling
    every replayed update by the early (t≈1) correction. Mirrors ScalarAdam's
    on-device `β^t` state."""
    if Int(global_idx.x) != 0:
        return
    powbuf[unsafe_offset=0] = powbuf[unsafe_offset=0] * beta1
    powbuf[unsafe_offset=1] = powbuf[unsafe_offset=1] * beta2


def _grouped_adam_kernel(
    val: Pointer[Scalar[DT], MutAnyOrigin],
    grd: Pointer[Scalar[DT], MutAnyOrigin],
    m: Pointer[Scalar[DT], MutAnyOrigin],
    v: Pointer[Scalar[DT], MutAnyOrigin],
    decay: Pointer[Scalar[DT], MutAnyOrigin],
    total_arg: Int64,
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    powbuf: Pointer[Scalar[DT], MutAnyOrigin],
    wd: Scalar[DT],
):
    """Arena update (all params at once over runtime-length flat buffers).
    `bc1/bc2` are read from the device `powbuf` (`[β₁ᵗ, β₂ᵗ]`, advanced by
    `_adam_advance_pow_kernel` just before) so they advance under graph replay.
    """
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var total = Int(total_arg)
    var i = Int(global_idx.x)
    if i >= total:
        return
    var one = Scalar[DT](1.0)
    var bc1 = one - powbuf[unsafe_offset=0]
    var bc2 = one - powbuf[unsafe_offset=1]
    var p = val[unsafe_offset=i]
    if decay[unsafe_offset=i] != Scalar[DT](0.0):
        p -= lr * wd * p
    var g = grd[unsafe_offset=i]
    var m_new = beta1 * m[unsafe_offset=i] + (one - beta1) * g
    var v_new = beta2 * v[unsafe_offset=i] + (one - beta2) * g * g
    m[unsafe_offset=i] = m_new
    v[unsafe_offset=i] = v_new
    var m_hat = m_new / bc1
    var v_hat = v_new / bc2
    val[unsafe_offset=i] = p - lr * m_hat / (sqrt(v_hat) + eps)


def _adam_warmup_lr_kernel(
    lr_buf: Pointer[Scalar[DT], MutAnyOrigin],
    step_buf: Pointer[Scalar[DT], MutAnyOrigin],
    target_lr: Scalar[DT],
    warmup: Scalar[DT],
):
    """On-device LinearWarmup: `lr = target·min(step/warmup, 1)`, then `step+=1`
    (1 thread). Both `lr` and `step` live on-device, so the schedule advances on
    every CUDA-graph REPLAY — a host `opt.lr = sched.lr_at(t)` write would freeze
    at the capture-time (near-0 ramp) value. `warmup <= 0` → constant target."""
    if Int(global_idx.x) != 0:
        return
    var t = step_buf[unsafe_offset=0]
    var lr = target_lr
    if warmup > Scalar[DT](0.0) and t < warmup:
        lr = target_lr * t / warmup
    lr_buf[unsafe_offset=0] = lr
    step_buf[unsafe_offset=0] = t + Scalar[DT](1.0)


def _grouped_adam_kernel_devlr(
    val: Pointer[Scalar[DT], MutAnyOrigin],
    grd: Pointer[Scalar[DT], MutAnyOrigin],
    m: Pointer[Scalar[DT], MutAnyOrigin],
    v: Pointer[Scalar[DT], MutAnyOrigin],
    decay: Pointer[Scalar[DT], MutAnyOrigin],
    total_arg: Int64,
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    powbuf: Pointer[Scalar[DT], MutAnyOrigin],
    lr_buf: Pointer[Scalar[DT], MutAnyOrigin],
    wd: Scalar[DT],
):
    """`_grouped_adam_kernel` but `lr` is read from the device `lr_buf` (written
    by `_adam_warmup_lr_kernel`) — the capture-safe scheduled-LR path. Identical
    math; only the LR source differs."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var total = Int(total_arg)
    var i = Int(global_idx.x)
    if i >= total:
        return
    var one = Scalar[DT](1.0)
    var lr = lr_buf[unsafe_offset=0]
    var bc1 = one - powbuf[unsafe_offset=0]
    var bc2 = one - powbuf[unsafe_offset=1]
    var p = val[unsafe_offset=i]
    if decay[unsafe_offset=i] != Scalar[DT](0.0):
        p -= lr * wd * p
    var g = grd[unsafe_offset=i]
    var m_new = beta1 * m[unsafe_offset=i] + (one - beta1) * g
    var v_new = beta2 * v[unsafe_offset=i] + (one - beta2) * g * g
    m[unsafe_offset=i] = m_new
    v[unsafe_offset=i] = v_new
    var m_hat = m_new / bc1
    var v_hat = v_new / bc2
    val[unsafe_offset=i] = p - lr * m_hat / (sqrt(v_hat) + eps)


# ── arena moment placement ───────────────────────────────────────────────


struct _MomentPlacer(ParamVisitor, ParamVisitorRT):
    """Rebind every Param's `m`/`v` Tensors to slices of the arena moments.

    ⚠ Without this, arena mode SILENTLY DROPS the optimizer moments from every
    checkpoint. `BinaryCheckpointWriter` gates on `m.n >= N and v.n >= N`, and
    in arena mode the per-param `m`/`v` are never touched (the grouped kernel
    reads `m_arena`/`v_arena` directly) — so they stay EMPTY, `has_m` is
    written as "0", and a resumed run restarts Adam from zero moments while
    reporting a successful load. The values are on the device the whole time;
    only the handle the checkpoint walk reads was missing.

    The slices alias the same offsets `ParamArena` used for val/grd, so the
    walk order is the one `adopt` just used. A param with no moments yet gets
    the freshly zeroed slice — Adam's initial state; one that already holds
    moments (a checkpoint loaded before `adopt`) has them copied in.
    """

    var m_arena: DeviceBuffer[DT]
    var v_arena: DeviceBuffer[DT]
    var off: Int

    def __init__(out self, m_arena: DeviceBuffer[DT], v_arena: DeviceBuffer[DT]):
        self.m_arena = m_arena
        self.v_arena = v_arena
        self.off = 0

    def __init__(out self, *, deinit move: Self):
        self.m_arena = move.m_arena
        self.v_arena = move.v_arena
        self.off = move.off

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            # Same rounding as `ParamArena` — the moments alias val/grd BY
            # OFFSET, so the two walks must land on identical boundaries.
            self.off = align_param_off(self.off)
            var m_sub = self.m_arena.create_sub_buffer[DT](self.off, n)
            var v_sub = self.v_arena.create_sub_buffer[DT](self.off, n)
            # ⚠ Moments that already exist — restored from a checkpoint
            # before `adopt`, or accumulated by per-param steps — are COPIED
            # into the arena, not dropped. Re-pointing alone silently discarded
            # a resume's moments whenever the load ran before the adopt, and
            # the resumed run was worse than a cold optimizer.
            if m.dev and v.dev and m.n >= n and v.n >= n:
                var c = ctx.value()
                c.enqueue_copy(m_sub, m.dev.value().create_sub_buffer[DT](0, n))
                c.enqueue_copy(v_sub, v.dev.value().create_sub_buffer[DT](0, n))
            m.dev = Optional(m_sub)
            m.n = n
            v.dev = Optional(v_sub)
            v.n = n
            self.off += n

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
struct Adam(Movable, ParamVisitor, ParamVisitorRT, Optimizer):
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var wd: Scalar[DT]
    var t: Int
    var _b1_pow: Scalar[DT]
    var _b2_pow: Scalar[DT]
    var bc1: Scalar[DT]
    var bc2: Scalar[DT]
    # Arena mode (GPU, set by `adopt`): shared val/grd packing + own m/v arenas.
    var arena: ParamArena
    var m_arena: Tensor
    var v_arena: Tensor
    # Device-resident bias-correction powers `[β₁ᵗ, β₂ᵗ]` (GPU arena path only,
    # allocated by `adopt`). Advanced on-device each step so the correction
    # advances under CUDA-graph replay instead of freezing at the host-baked
    # capture-time value. CPU + per-param paths use the host `bc1/bc2`.
    var _pow_dev: Tensor
    # Persistent grad-clip scratch (GPU arena, allocated by `adopt`): block
    # partials + scale + pre-clip norm. Owned here so `clip_grads_device` does
    # NO per-call allocation (capture-safe). Empty until `adopt` on GPU.
    var _clip_partials: Tensor
    var _clip_scale: Tensor
    var _clip_norm: Tensor
    # Opt-in on-device LR schedule (GPU arena). When `_sched_active`, the grouped
    # step runs an on-device LinearWarmup into `_lr_dev` (step in `_step_dev`)
    # and the update reads `_lr_dev` — capture-safe. OFF by default → the update
    # uses the host `lr` (unchanged, zero overhead). Drive via
    # `attach_warmup_schedule`.
    var _lr_dev: Tensor
    var _step_dev: Tensor
    var _sched_active: Bool
    var _sched_target_lr: Scalar[DT]
    var _sched_warmup: Int
    # Host-driven device LR (GPU arena). When set, the grouped step reads `lr`
    # from `_lr_dev` (the device-LR kernel) but does NOT run the on-device
    # warmup ramp — the HOST writes the next LR into `_lr_dev` each step via
    # `push_lr_device` (a 1-elem `enqueue_fill`, NO realloc → the buffer pointer
    # is stable so a captured graph that baked it stays valid). This is the
    # capture-safe path for an ARBITRARY host-computed schedule (e.g. cosine),
    # vs `attach_warmup_schedule`'s on-device LinearWarmup. Mutually exclusive
    # with `_sched_active`.
    var _lr_host_driven: Bool

    def __init__(out self):
        """No-arg default (satisfies Defaultable for the generic Trainer)."""
        self = Self(lr=Scalar[DT](1e-3))

    def __init__(
        out self,
        lr: Scalar[DT],  # required → disambiguates from the no-arg ctor
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
        wd: Scalar[DT] = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.t = 0
        self._b1_pow = Scalar[DT](1.0)
        self._b2_pow = Scalar[DT](1.0)
        self.bc1 = Scalar[DT](1.0)
        self.bc2 = Scalar[DT](1.0)
        self._pow_dev = Tensor()
        self._clip_partials = Tensor()
        self._clip_scale = Tensor()
        self._clip_norm = Tensor()
        self._lr_dev = Tensor()
        self._step_dev = Tensor()
        self._sched_active = False
        self._sched_target_lr = Scalar[DT](0.0)
        self._sched_warmup = 0
        self._lr_host_driven = False
        self.arena = ParamArena()
        self.m_arena = Tensor()
        self.v_arena = Tensor()

    def put_step_state(self, mut sc: CheckpointScalars, prefix: String) raises:
        """Record the step state a resume needs: `t` and the bias-correction
        powers `β₁ᵗ`, `β₂ᵗ`, as `<prefix>.t` / `.b1_pow` / `.b2_pow`.

        ⚠⚠ THE MOMENTS ALONE DO NOT RESUME ADAM. `m` and `v` ride the param
        sections, but a resume that restores them with `t = 0` divides settled
        moments by the step-1 corrections (1 − β₁ = 0.1, 1 − β₂ = 0.001), and
        the first few hundred updates come out ~0.3× the size they should be.
        The powers are stored rather than recomputed from `t` because the
        running product is what `begin_step` keeps, and `pow(β, t)` differs
        from it in the last bits."""
        sc.set_int(prefix + ".t", self.t)
        sc.set(prefix + ".b1_pow", Float64(self._b1_pow))
        sc.set(prefix + ".b2_pow", Float64(self._b2_pow))

    def take_step_state(mut self, sc: CheckpointScalars, prefix: String) raises:
        """Inverse of `put_step_state`. A checkpoint written without it (every
        file before `K` sections existed) leaves this optimizer untouched.
        Safe before or after `adopt`: a device mirror that already exists is
        rewritten, and one allocated later is seeded from these values."""
        if not sc.has(prefix + ".t"):
            return
        self.t = sc.get_int(prefix + ".t", 0)
        self._b1_pow = Scalar[DT](sc.get(prefix + ".b1_pow", 1.0))
        self._b2_pow = Scalar[DT](sc.get(prefix + ".b2_pow", 1.0))
        self.bc1 = Scalar[DT](1.0) - self._b1_pow
        self.bc2 = Scalar[DT](1.0) - self._b2_pow
        self._sync_step_state_dev()

    def _sync_step_state_dev(mut self) raises:
        """Host step state → the device mirrors, when they exist (GPU arena)."""
        if self._pow_dev.dev:
            var pd = self._pow_dev.dev.value()
            pd.create_sub_buffer[DT](0, 1).enqueue_fill(self._b1_pow)
            pd.create_sub_buffer[DT](1, 1).enqueue_fill(self._b2_pow)
        if self._step_dev.dev:
            self._step_dev.dev.value().enqueue_fill(Scalar[DT](self.t))

    def begin_step(mut self):
        """Bump the step counter + refresh bias corrections. Once per step."""
        self.t += 1
        self._b1_pow = self._b1_pow * self.beta1
        self._b2_pow = self._b2_pow * self.beta2
        self.bc1 = Scalar[DT](1.0) - self._b1_pow
        self.bc2 = Scalar[DT](1.0) - self._b2_pow

    def adopt[
        target: StaticString, M: ParamWalkable
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Engage arena mode (GPU); NO-OP on CPU. Packs val/grd via the shared
        ParamArena and allocates this optimizer's m/v arenas to match."""
        self.arena.adopt[target](model, ctx)
        comptime if target == "gpu":
            var c = ctx.value()
            self.m_arena = Tensor.alloc_gpu(c, self.arena.total)
            self.v_arena = Tensor.alloc_gpu(c, self.arena.total)
            # `[β₁ᵗ, β₂ᵗ]` seeded to β^0 = 1; advanced on-device each step.
            self._pow_dev = Tensor.alloc_gpu(c, 2)
            # Persistent grad-clip scratch (one block-partials slot per TPB chunk
            # of the arena, + scale + norm) so `clip_grads_device` never allocs.
            var nblk = (self.arena.total + TPB - 1) // TPB
            self._clip_partials = Tensor.alloc_gpu(c, nblk if nblk > 0 else 1)
            self._clip_scale = Tensor.alloc_gpu(c, 1)
            self._clip_norm = Tensor.alloc_gpu(c, 1)
            # Device LR + step for the opt-in on-device schedule (seeded to the
            # current host lr / step 0; only used when `attach_warmup_schedule`).
            self._lr_dev = Tensor.alloc_gpu(c, 1)
            self._lr_dev.dev.value().enqueue_fill(self.lr)
            self._step_dev = Tensor.alloc_gpu(c, 1)
            # `[β₁ᵗ, β₂ᵗ]` and the schedule step, seeded from the HOST step
            # state: β^0 = 1 and step 0 on a fresh optimizer, the restored
            # values when `take_step_state` ran before this adopt.
            self._sync_step_state_dev()
            # Give every Param a HANDLE on its slice of the moment arenas, so
            # the checkpoint walk still finds moments to write. See
            # `_MomentPlacer` — without it, adopting silently turns
            # `save_moments=True` into a no-op.
            var mp = _MomentPlacer(
                self.m_arena.dev.value(), self.v_arena.dev.value()
            )
            walk_params["gpu"](model, mp, ctx)

    def adopt_multi[
        target: StaticString, *Ms: ParamWalkable
    ](mut self, ctx: Optional[DeviceContext], mut *models: *Ms) raises:
        """`adopt` for a trainable set spread over several objects.

        ⚠ Everything `adopt` allocates is sized from `arena.total`, so this
        must build the arena across ALL the models before allocating — which
        is why it cannot be N calls to `adopt`. See `ParamArena.adopt_multi`.
        """
        self.arena.adopt_multi[target](ctx, *models)
        comptime if target == "gpu":
            var c = ctx.value()
            self.m_arena = Tensor.alloc_gpu(c, self.arena.total)
            self.v_arena = Tensor.alloc_gpu(c, self.arena.total)
            self._pow_dev = Tensor.alloc_gpu(c, 2)
            var nblk = (self.arena.total + TPB - 1) // TPB
            self._clip_partials = Tensor.alloc_gpu(c, nblk if nblk > 0 else 1)
            self._clip_scale = Tensor.alloc_gpu(c, 1)
            self._clip_norm = Tensor.alloc_gpu(c, 1)
            self._lr_dev = Tensor.alloc_gpu(c, 1)
            self._lr_dev.dev.value().enqueue_fill(self.lr)
            self._step_dev = Tensor.alloc_gpu(c, 1)
            # `[β₁ᵗ, β₂ᵗ]` and the schedule step, seeded from the HOST step
            # state: β^0 = 1 and step 0 on a fresh optimizer, the restored
            # values when `take_step_state` ran before this adopt.
            self._sync_step_state_dev()
            # ⚠ Every model, or the checkpoint silently loses the moments of
            # the ones that were skipped — `save_moments=True` becomes a
            # partial no-op and a resume comes back with a half-cold
            # optimizer.
            var mp = _MomentPlacer(
                self.m_arena.dev.value(), self.v_arena.dev.value()
            )
            comptime for i in range(models.__len__()):
                walk_params["gpu"](models[i], mp, ctx)

    def arena_step(mut self, c: DeviceContext) raises:
        """One grouped update over an arena adopted with `adopt_multi`.

        ⚠ The caller runs the `ParamVersionBump` walk itself, over the same
        models — `step` bundles it for a single model and there is no single
        model here.
        """
        if not self.arena.adopted:
            raise Error(
                "Adam.arena_step: no arena — call adopt_multi first, or use"
                " the per-parameter walk"
            )
        self.begin_step()
        self._grouped_step(c)

    def arena_clip(
        mut self, max_norm: Scalar[DT], c: DeviceContext
    ) raises -> Scalar[DT]:
        """GLOBAL grad-norm clip over the whole arena; returns the pre-clip
        norm.

        ⚠ Global is the point. Clipping each component of a trainable set to
        `max_norm` independently is a DIFFERENT algorithm from clipping their
        joint norm — with five components it can pass through a total norm of
        5x the limit — and it is what `clip_grads` would do if handed them one
        at a time.
        """
        if not self.arena.adopted:
            raise Error("Adam.arena_clip: no arena — call adopt_multi first")
        return clip_arena_grads(self.arena, max_norm, c)

    def step[
        target: StaticString, M: ParamWalkable
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Bump the step then update every Param. GPU+adopted → one arena kernel;
        CPU or un-adopted GPU → per-param walk."""
        self.begin_step()
        # Both walks go through ONE erased visitor type, so the model's walk
        # is instantiated once per target instead of once per visitor
        # (`ParamVisitorRef`, docs/COMPILE_TIME_PROFILING.md §4).
        comptime if target == "cpu":
            var me = ParamVisitorRef.of[Adam, "cpu"](self)
            model.for_each_param["cpu"](me, ctx)
        else:
            if self.arena.adopted:
                self._grouped_step(ctx.value())
            else:
                var me = ParamVisitorRef.of[Adam, "gpu"](self)
                model.for_each_param["gpu"](me, ctx)
        # AMP: invalidate cached bf16 weights — bump every param-value version so
        # leaves whose cached cast predates this step recast on next forward.
        # Host-only walk (no kernels); covers per-param AND arena paths. Not
        # CUDA-graph-capturable (host-side) — captured AMP is a Phase-5 concern.
        var _bump = ParamVersionBump()
        var bref = ParamVisitorRef.of[ParamVersionBump, target](_bump)
        model.for_each_param[target](bref, ctx)

    def _grouped_step(mut self, c: DeviceContext) raises:
        if self.arena.total == 0:
            return
        # Advance β^t on-device BEFORE the update reads it — captured into the
        # graph so it advances per replay (host `bc1/bc2` would freeze).
        c.enqueue_function[_adam_advance_pow_kernel](
            self._pow_dev.dev.value(),
            self.beta1,
            self.beta2,
            grid_dim=1,
            block_dim=1,
        )
        var nblk = (self.arena.total + TPB - 1) // TPB
        if self._sched_active:
            # On-device LinearWarmup → `_lr_dev` (advances `_step_dev`), then the
            # device-LR update kernel. Both captured → schedule advances per
            # replay. Default (no schedule) keeps the host-`lr` kernel below.
            c.enqueue_function[_adam_warmup_lr_kernel](
                self._lr_dev.dev.value(),
                self._step_dev.dev.value(),
                self._sched_target_lr,
                Scalar[DT](self._sched_warmup),
                grid_dim=1,
                block_dim=1,
            )
            c.enqueue_function[_grouped_adam_kernel_devlr](
                self.arena.val.dev.value(),
                self.arena.grd.dev.value(),
                self.m_arena.dev.value(),
                self.v_arena.dev.value(),
                self.arena.decay_mask.dev.value(),
                Int64(self.arena.total),
                self.beta1,
                self.beta2,
                self.eps,
                self._pow_dev.dev.value(),
                self._lr_dev.dev.value(),
                self.wd,
                grid_dim=nblk,
                block_dim=TPB,
            )
            return
        if self._lr_host_driven:
            # Host-driven device LR: read `lr` from `_lr_dev` (the host wrote it
            # via `push_lr_device` BEFORE this step / before the captured replay),
            # no on-device warmup ramp. Capture-safe — the kernel sequence is
            # fixed and `_lr_dev`'s pointer is stable.
            c.enqueue_function[_grouped_adam_kernel_devlr](
                self.arena.val.dev.value(),
                self.arena.grd.dev.value(),
                self.m_arena.dev.value(),
                self.v_arena.dev.value(),
                self.arena.decay_mask.dev.value(),
                Int64(self.arena.total),
                self.beta1,
                self.beta2,
                self.eps,
                self._pow_dev.dev.value(),
                self._lr_dev.dev.value(),
                self.wd,
                grid_dim=nblk,
                block_dim=TPB,
            )
            return
        c.enqueue_function[_grouped_adam_kernel](
            self.arena.val.dev.value(),
            self.arena.grd.dev.value(),
            self.m_arena.dev.value(),
            self.v_arena.dev.value(),
            self.arena.decay_mask.dev.value(),
            Int64(self.arena.total),
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self._pow_dev.dev.value(),
            self.wd,
            grid_dim=nblk,
            block_dim=TPB,
        )

    def step_captured(mut self, c: DeviceContext) raises:
        """Capture-safe arena step for a CUDA-graph-captured train loop. Runs ONLY
        the grouped device step (`_grouped_step`): it advances `β^t` on the device
        `_pow_dev` and reads the device LR (`_lr_dev`, written eagerly by
        `push_lr_device`), so the correction + schedule advance on every replay.
        Unlike `step`, it does NO host work — no host `begin_step` counter bump and
        no AMP `ParamVersionBump` walk (both host-side ⇒ not capturable). Requires
        `adopt` (arena mode); a no-op if the arena is empty."""
        self._grouped_step(c)

    def zero_grad[
        target: StaticString, M: ParamWalkable
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """GPU+adopted → zero the grad arena in ONE fill; else per-param via the
        model."""
        comptime if target == "gpu":
            if self.arena.adopted:
                self.arena.zero_grad(ctx.value())
                return
        model.zero_grad[target](ctx)

    def attach_warmup_schedule(mut self, target_lr: Scalar[DT], warmup_steps: Int):
        """Drive the LR with an ON-DEVICE LinearWarmup ramp (0 → target over
        `warmup_steps` update steps, then constant) so it's CUDA-graph-safe — the
        ramp advances on every replay. GPU-arena path only; a no-op stub for
        CPU/non-adopted (use host `set_lr` + a host schedule there). Replaces the
        host `opt.lr = sched.lr_at(t)` pattern, which freezes under capture."""
        self._sched_active = True
        self._sched_target_lr = target_lr
        self._sched_warmup = warmup_steps

    def set_lr(mut self, lr: Scalar[DT]):
        self.lr = lr

    def get_lr(self) -> Scalar[DT]:
        return self.lr

    def push_lr_device[
        target: StaticString
    ](mut self, lr: Scalar[DT], ctx: Optional[DeviceContext] = None) raises:
        """Capture-safe LR update for an arbitrary host-computed schedule. On the
        GPU arena path, writes `lr` into the device `_lr_dev` buffer (1-elem
        `enqueue_fill`, NO realloc → stable pointer) and engages the device-LR
        kernel, so a CUDA-graph-captured `step` reads the FRESH LR on every
        replay instead of the host-baked capture-time value. Call this EAGERLY
        each step before the captured replay. Off the GPU-arena path it just
        sets the host `lr` (the un-captured kernel reads it directly). Keeps the
        host `lr` in sync either way (so `get_lr` stays meaningful)."""
        self.lr = lr
        comptime if target == "gpu":
            if self.arena.adopted:
                self._lr_host_driven = True
                self._lr_dev.dev.value().enqueue_fill(lr)

    def clip_grads[
        target: StaticString, M: ParamWalkable
    ](
        mut self, mut model: M, max_norm: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        """Global grad-norm clip, returns the pre-clip norm. GPU+adopted → arena
        reduction+scale (capture-safe, no per-param D2H); else → per-param
        `clip_grad_norm`. Symmetric across targets."""
        comptime if target == "gpu":
            if self.arena.adopted:
                return clip_arena_grads(self.arena, max_norm, ctx.value())
        return clip_grad_norm[target](model, max_norm, ctx)

    def clip_grads_device[
        target: StaticString, M: ParamWalkable
    ](
        mut self, mut model: M, max_norm: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """CUDA-graph-safe grad-norm clip (GPU + adopted): on-device kernels over
        persistent scratch, NO allocation and NO D2H — drop this into the captured
        train-step sequence. The pre-clip norm lands in the device `_clip_norm`
        buffer; read it via `read_clip_norm` at flush cadence if you log it. Off
        the GPU-arena path it falls back to `clip_grads` (which D2Hs — NOT
        capture-safe; only correct when not capturing)."""
        comptime if target == "gpu":
            if self.arena.adopted:
                clip_arena_grads_captured(
                    self.arena, self._clip_partials, self._clip_scale,
                    self._clip_norm, max_norm, ctx.value(),
                )
                return
        _ = self.clip_grads[target, M](model, max_norm, ctx)

    def read_clip_norm(mut self, ctx: DeviceContext) raises -> Scalar[DT]:
        """D2H the last pre-clip grad norm from `clip_grads_device` (flush
        cadence only — NOT per step). 0 if never clipped on device."""
        if not self._clip_norm.dev:
            return Scalar[DT](0.0)
        self._clip_norm.download(ctx)
        return self._clip_norm.data[0]

    def has_clip_norm_dev(self) -> Bool:
        """Is there a device-resident pre-clip norm (GPU + `adopt` only)?

        The guard for `clip_norm_dev`: off the grouped-arena path
        `clip_grads_device` falls back to `clip_grads`, which computes the norm
        on the host and never writes the device buffer."""
        if self._clip_norm.dev:
            return True
        return False

    def clip_norm_dev(mut self) raises -> LayoutTensor[
        DT, Layout.row_major(1), MutAnyOrigin
    ]:
        """Device view of the last pre-clip grad norm — for a caller that logs
        the norm EVERY step and therefore must not download it.

        `read_clip_norm` is the flush-cadence host read; this is its
        capture-safe counterpart. A per-step `read_clip_norm` is a full device
        synchronization plus a D2H for one float that the logger averages over
        a window anyway — fold this into a device accumulator instead and drain
        the accumulator at flush.

        ⚠ Guard with `has_clip_norm_dev`. Raising rather than returning an
        empty view is deliberate: a silently-zero grad norm reads as "clipping
        never fired", which is a plausible-looking number, not an obvious
        failure."""
        if not self._clip_norm.dev:
            raise Error(
                "Adam.clip_norm_dev: no device clip-norm buffer. It is"
                " allocated by `adopt` on the GPU grouped-arena path; off that"
                " path `clip_grads_device` computes the norm on the host."
            )
        return self._clip_norm.lt["gpu", Layout.row_major(1)]()

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        """`visit` with the size at runtime (see `ParamVisitorRef`)."""
        comptime if target == "cpu":
            m.ensure(n)
            v.ensure(n)
            # With a comptime N the plain loop vectorised on its own; with a
            # runtime n it did not (measured: 2.75 -> 4.18 ms/step on a 1.2M-
            # param MLP), so the SIMD loop is spelled out.
            comptime W = simd_width_of[DT]()
            var pp = param.data.unsafe_ptr()
            var gp = grad.data.unsafe_ptr()
            var mp = m.data.unsafe_ptr()
            var vp = v.data.unsafe_ptr()
            var one = SIMD[DT, W](1.0)
            var lr = SIMD[DT, W](self.lr)
            var b1 = SIMD[DT, W](self.beta1)
            var b2 = SIMD[DT, W](self.beta2)
            var eps = SIMD[DT, W](self.eps)
            var bc1 = SIMD[DT, W](self.bc1)
            var bc2 = SIMD[DT, W](self.bc2)
            var wd = SIMD[DT, W](self.wd)
            var i = 0
            while i + W <= n:
                var p = pp.unsafe_load[width=W](i)
                if apply_decay:
                    p -= lr * wd * p
                var g = gp.unsafe_load[width=W](i)
                var m_new = b1 * mp.unsafe_load[width=W](i) + (one - b1) * g
                var v_new = b2 * vp.unsafe_load[width=W](i) + (one - b2) * g * g
                mp.unsafe_store(i, m_new)
                vp.unsafe_store(i, v_new)
                pp.unsafe_store(i, p - lr * (m_new / bc1) / (sqrt(v_new / bc2) + eps))
                i += W
            var one1 = Scalar[DT](1.0)
            while i < n:
                var p = pp[unsafe_offset=i]
                if apply_decay:
                    p -= self.lr * self.wd * p
                var g = gp[unsafe_offset=i]
                var m_new = self.beta1 * mp[unsafe_offset=i] + (one1 - self.beta1) * g
                var v_new = self.beta2 * vp[unsafe_offset=i] + (one1 - self.beta2) * g * g
                mp[unsafe_offset=i] = m_new
                vp[unsafe_offset=i] = v_new
                pp[unsafe_offset=i] = p - self.lr * (m_new / self.bc1) / (sqrt(v_new / self.bc2) + self.eps)
                i += 1
        else:
            var c = ctx.value()
            if not m.dev:
                m.ensure_gpu(c, n)
                m.dev.value().enqueue_fill(Scalar[DT](0))
                v.ensure_gpu(c, n)
                v.dev.value().enqueue_fill(Scalar[DT](0))
            var nblk = (n + TPB - 1) // TPB
            c.enqueue_function[_adam_update_kernel_rt](
                param.dev.value(),
                grad.dev.value(),
                m.dev.value(),
                v.dev.value(),
                Int64(n),
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.bc1,
                self.bc2,
                self.wd,
                Int64(apply_decay),
                grid_dim=nblk,
                block_dim=TPB,
            )

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        """The comptime-N visit forwards to `visit_rt`: ONE copy of the
        update, and the SIMD loop serves both paths (it is 5x the old scalar
        loop on a 1.2M-param MLP, 2.75 -> 0.55 ms/step)."""
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)

comptime AdamW = Adam
