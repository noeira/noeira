"""Trainer — minimal supervised classification trainer (storage surface).

Holds a `MODEL` (Module producing logits [BATCH, NC]), an `Adam`, and a
`CrossEntropyLoss[NC]`. `train_epoch` runs the batch loop (forward → CE loss
accumulate → CE vjp → model.vjp → opt.step) over a flat dataset; `eval_top1`
runs forward + argmax accuracy.

Data path (matches legacy "resident dataset + slice", no per-batch transfer):
- GPU: the whole dataset is uploaded to device ONCE (lazily, on first use), then
  each batch input is a zero-copy `create_sub_buffer` VIEW into that resident
  buffer (no per-batch H2D). The owned `batch_x/batch_y` just carry the view.
- CPU: the host `List` is already resident; each batch is a cheap `unsafe_memcpy` slice
  (a zero-copy view would need an offset threaded through `forward`, not worth it
  — the copy is a few % of CPU wall time).

Flat-dataset convention (the example builds these): `train_x` = [N·IN] row-major
images, `train_y` = [N·NC] one-hot labels; `test_x` = [N·IN], `test_labels` =
[N] integer class ids. Topology must match what was trained.
"""

from max.gpu.host import DeviceContext
from max.gpu import global_idx
from std.memory import unsafe_memcpy
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs, child_refs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..optimizer.adam import Adam
from ..optimizer.optimizer import Optimizer
from ..loss.cross_entropy import CrossEntropyLoss
from .shuffle_kernels import (
    init_identity_indices_kernel,
    fisher_yates_shuffle_kernel,
    increment_seed_kernel,
    gather_rows_kernel,
    gather_rows_at_counter_kernel,
    advance_counter_kernel,
    reset_counter_kernel,
)
from .augmenter import Augmenter, IdentityAugmenter
from ..optimizer.lr_scheduler import Scheduler, ConstantSchedule
from noeira.cuda import CUDAGraph, maybe_capture_replay


# ── AMP boundary cast kernels (fp32 ↔ MADT) ─────────────────────────────
# Used ONLY on the bf16-flow path (MADT != DT). The fp32 path never casts.
# Parametrized by the source/destination dtypes + length so one pair handles
# fp32→MADT (input/grad downcast) and MADT→fp32 (logits upcast for the loss).
def _cast_kernel[
    SRC: DType, DST: DType, N: Int
](
    src: LayoutTensor[SRC, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DST, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[SRC]](src[i]).cast[DST]()


@fieldwise_init
struct TrainResult(Movable & Deinitable):
    """Per-epoch training history returned by `Trainer.train_gpu`."""

    var epoch_train_loss: List[Float64]
    var epoch_test_top1: List[Float64]
    var epoch_train_s: List[Float64]
    var epoch_eval_s: List[Float64]

    @staticmethod
    def empty() -> Self:
        return Self(
            epoch_train_loss=List[Float64](),
            epoch_test_top1=List[Float64](),
            epoch_train_s=List[Float64](),
            epoch_eval_s=List[Float64](),
        )


struct Trainer[
    MODEL: Module, NC: Int, IN: Int, BATCH: Int, target: StaticString,
    POLICY: AMPPolicy = NoAMP,
    OPT: Optimizer = Adam,
    # Opt-in CUDA-graph capture of the per-batch step: the batch gather into
    # the FIXED owned `batch_x`/`batch_y` (offset from a device counter, so
    # one capture serves every batch), zero_grad → forward → CE accumulate →
    # vjp → opt.step. The per-epoch augmentation and shuffle run eagerly in
    # place before the epoch's replays. fp32-only.
    # Default OFF — anything that ALLOCATES inside the captured region aborts
    # stream capture on NVIDIA. The dW GEMMs of Linear/Conv2D own their split-K
    # workspace now, but only where MAX's split-K dispatch applies (sm_80 /
    # sm_89 / sm_120; H100 and B200 still allocate per call), and MAX
    # memory-manager allocations elsewhere in the step abort capture too
    # (`NOEIRA_ALLOC_TRACE=1` names the first blocker). Enable only on
    # a model whose step captures clean. No-op on non-NVIDIA (runs eagerly,
    # bit-identical).
    USE_TRAIN_CUDA_GRAPH: Bool = False,
](Movable & Deinitable):
    # Model activation-flow dtype: `DT` for an fp32 model, `bfloat16` for a
    # bf16-flow model (`Sequential[Linear[..,bf16], …]`). The CASTS only happen
    # at boundaries; the fp32 case (MADT == DT) is the EXISTING code path with
    # no casts and no extra buffers (bit-identical).
    comptime MADT = Self.MODEL.ACT_DT
    # Capture is GPU + fp32 only (the AMP step's cached-bf16 version bump is
    # host-side, so it can't ride a captured graph).
    comptime CAPTURE = (
        Self.USE_TRAIN_CUDA_GRAPH
        and (Self.target == "gpu")
        and (Self.MADT == DT)
    )

    var model: Self.MODEL
    var opt: Self.OPT
    var loss: CrossEntropyLoss[Self.NC]
    # Activation buffers that touch the MODEL flow at MADT. For an fp32 model
    # `TensorImpl[MADT]` IS `Tensor`, so these are unchanged.
    var batch_x: TensorImpl[Self.MADT]
    var batch_y: Tensor  # fp32 labels (consumed by the fp32 loss)
    var logits: TensorImpl[Self.MADT]  # model output (MADT)
    var grad: TensorImpl[Self.MADT]  # grad_output fed to model.vjp (MADT)
    var gi: TensorImpl[Self.MADT]  # model grad_input (MADT)
    # fp32 loss-boundary scratch (used ONLY when MADT != DT). The loss runs in
    # fp32: `logits` (MADT) is cast → `logits_f32`, and `grad_f32` (fp32, from
    # the loss vjp) is cast → `grad` (MADT). For MADT == DT these stay empty and
    # are never referenced (the fp32 path uses `logits`/`grad` directly).
    var logits_f32: Tensor
    var grad_f32: Tensor
    # fp32 input staging (used ONLY when MADT != DT). The resident dataset is
    # fp32, so a bf16 batch can't be a zero-copy fp32 sub-buffer view — the fp32
    # batch slice lands here (sub-buffer view / unsafe_memcpy / gather) and is then cast
    # into the owned bf16 `batch_x`. For MADT == DT this stays empty and unused:
    # the fp32 path stages directly into `batch_x` (zero-copy, unchanged).
    var batch_xf: Tensor
    # GPU-resident dataset (uploaded once; batches are sub-buffer views). fp32.
    var ds_x: Tensor
    var ds_y: Tensor
    var ds_tx: Tensor
    var _up_train: Bool
    var _up_test: Bool
    # Lazily-captured per-batch compute graph (None until first capture). Only
    # touched on the `CAPTURE` path; a no-op slot otherwise.
    var _train_graph: Optional[CUDAGraph]

    def __init__(out self):
        self.model = Self.MODEL()
        self.opt = Self.OPT()
        self.loss = CrossEntropyLoss[Self.NC]()
        self.batch_x = TensorImpl[Self.MADT]()
        self.batch_y = Tensor()
        self.logits = TensorImpl[Self.MADT]()
        self.grad = TensorImpl[Self.MADT]()
        self.gi = TensorImpl[Self.MADT]()
        self.logits_f32 = Tensor()
        self.grad_f32 = Tensor()
        self.batch_xf = Tensor()
        self.ds_x = Tensor()
        self.ds_y = Tensor()
        self.ds_tx = Tensor()
        self._up_train = False
        self._up_test = False
        self._train_graph = None

    @staticmethod
    def make[
        INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None, lr: Scalar[DT] = 1e-3
    ) raises -> Self:
        """`ctx` required on GPU (the caller owns it and reuses it for
        train/eval so all device buffers share one context)."""
        var t = Self()
        # One factory: allocates + initializes the model with INIT at construction.
        t.model = Self.MODEL.make[Self.target, INIT](ctx)
        comptime if Self.target == "cpu":
            t.loss = CrossEntropyLoss[Self.NC].make_cpu()
            # CPU is fp32-only (bf16-flow leaves are GPU-only) → MADT == DT here,
            # so `TensorImpl[MADT]` IS `Tensor`; allocate the MADT slab directly.
            t.batch_x = TensorImpl[Self.MADT].alloc(Self.BATCH * Self.IN)
            t.batch_y = Tensor.alloc(Self.BATCH * Self.NC)
        else:
            t.loss = CrossEntropyLoss[Self.NC].make_gpu(ctx.value())
            # batch_x/batch_y carry sub-buffer views on GPU — no owned slab.
            # EXCEPT under capture: the graph must read STABLE buffers, so
            # allocate fixed owned slabs and D2D-copy each batch's slice into
            # them (a sub-buffer view repoints the pointer per batch → would
            # invalidate the captured graph).
            comptime if Self.CAPTURE:
                t.batch_x = TensorImpl[Self.MADT].alloc_gpu(
                    ctx.value(), Self.BATCH * Self.IN
                )
                t.batch_y = Tensor.alloc_gpu(ctx.value(), Self.BATCH * Self.NC)
        t.opt = Self.OPT()
        t.opt.set_lr(lr)
        # Engage the optimizer's contiguous-arena mode (GPU single-kernel step);
        # a NO-OP on CPU. Must come AFTER the model is made + initialized — it
        # rebinds the model's param buffers to arena slices.
        t.opt.adopt[Self.target](t.model, ctx)
        return t^

    @staticmethod
    def _upload_dataset(
        mut dst: Tensor, ref src: List[Scalar[DT]], n: Int, ctx: DeviceContext
    ) raises:
        """Host List → resident device buffer (one-time, no per-batch H2D)."""
        dst.dev = ctx.enqueue_create_buffer[DT](n)
        var hb = ctx.enqueue_create_host_buffer[DT](n)
        ctx.synchronize()
        unsafe_memcpy(dest=hb.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
        ctx.enqueue_copy(dst.dev.value(), hb)
        ctx.synchronize()
        dst.n = n

    def _slice_train(mut self, x0: Int, y0: Int) raises:
        """Point the batch input/labels at GPU sub-buffer views of the resident
        set. fp32: `batch_x` IS the fp32 view (zero-copy, unchanged). bf16: the
        fp32 view lands in `batch_xf` (cast → `batch_x` happens in the step)."""
        comptime if Self.MADT == DT:
            ref bx = rebind[Tensor](self.batch_x)
            bx.dev = Optional(
                self.ds_x.dev.value().create_sub_buffer[DT](
                    x0, Self.BATCH * Self.IN
                )
            )
            bx.n = Self.BATCH * Self.IN
        else:
            self.batch_xf.dev = Optional(
                self.ds_x.dev.value().create_sub_buffer[DT](
                    x0, Self.BATCH * Self.IN
                )
            )
            self.batch_xf.n = Self.BATCH * Self.IN
        self.batch_y.dev = Optional(
            self.ds_y.dev.value().create_sub_buffer[DT](
                y0, Self.BATCH * Self.NC
            )
        )
        self.batch_y.n = Self.BATCH * Self.NC

    # ── AMP boundary-cast train step (factored across the 3 train-step sites) ──
    @staticmethod
    def _amp_train_step[
        B: Int
    ](
        mut model: Self.MODEL,
        mut opt: Self.OPT,
        mut loss: CrossEntropyLoss[Self.NC],
        mut model_in: TensorImpl[Self.MADT],
        mut dy: Tensor,
        mut logits: TensorImpl[Self.MADT],
        mut grad: TensorImpl[Self.MADT],
        mut gi: TensorImpl[Self.MADT],
        mut logits_f32: Tensor,
        mut grad_f32: Tensor,
        ctx: Optional[DeviceContext],
    ) raises:
        """forward → CE loss (fp32 boundary) → model.vjp → opt.step. `model_in`
        is the model input ALREADY at MADT (the caller casts/views it). zero_grad
        is the caller's responsibility (kept at the sites, unchanged).

        fp32 (MADT == DT): the EXISTING path verbatim — loss runs directly on
        `logits`/`grad`, no casts, `logits_f32`/`grad_f32` untouched. bf16: cast
        `logits`→`logits_f32` for the (fp32) loss, then `grad_f32`→`grad`."""
        comptime if Self.MADT == DT:
            # ── fp32 path (legacy, byte-identical) ──
            model.forward[Self.target, B, POLICY = Self.POLICY](
                child_refs[Self.MODEL.ARITY, Self.MADT](model_in), logits, ctx
            )
            # MADT IS DT here, but the opaque param doesn't collapse → rebind the
            # MADT activation buffers to `Tensor` for the fp32 loss (a no-op cast).
            ref logd = rebind[Tensor](logits)
            ref grdd = rebind[Tensor](grad)
            loss.forward_accumulate[Self.target, B](logd, dy, ctx)
            loss.vjp[Self.target, B](logd, dy, grdd, ctx)
            model.vjp[Self.target, B, POLICY = Self.POLICY](
                child_refs[Self.MODEL.ARITY, Self.MADT](model_in),
                grad,
                child_refs[Self.MODEL.ARITY, Self.MADT](gi),
                ctx,
            )
            opt.step[Self.target](model, ctx)
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                Self.target == "gpu"
            ), "AMP bf16-flow Trainer is GPU-only"
            comptime LOGN = B * Self.NC
            var c = ctx.value()
            model.forward["gpu", B, POLICY = Self.POLICY](
                child_refs[Self.MODEL.ARITY, Self.MADT](model_in), logits, ctx
            )
            # LOSS (fp32): cast logits (MADT) → logits_f32.
            logits_f32.ensure_gpu(c, LOGN)
            c.enqueue_function[
                _cast_kernel[Self.MADT, DT, LOGN]
            ](
                logits.lt["gpu", Layout.row_major(LOGN)](),
                logits_f32.lt["gpu", Layout.row_major(LOGN)](),
                grid_dim=(LOGN + TPB - 1) // TPB,
                block_dim=TPB,
            )
            loss.forward_accumulate["gpu", B](logits_f32, dy, ctx)
            loss.vjp["gpu", B](logits_f32, dy, grad_f32, ctx)
            # BACKWARD: cast grad_f32 (fp32) → grad (MADT).
            grad.ensure_gpu(c, LOGN)
            c.enqueue_function[
                _cast_kernel[DT, Self.MADT, LOGN]
            ](
                grad_f32.lt["gpu", Layout.row_major(LOGN)](),
                grad.lt["gpu", Layout.row_major(LOGN)](),
                grid_dim=(LOGN + TPB - 1) // TPB,
                block_dim=TPB,
            )
            model.vjp["gpu", B, POLICY = Self.POLICY](
                child_refs[Self.MODEL.ARITY, Self.MADT](model_in),
                grad,
                child_refs[Self.MODEL.ARITY, Self.MADT](gi),
                ctx,
            )
            opt.step["gpu"](model, ctx)

    @staticmethod
    def _cast_input_to_madt(
        mut batch_xf: Tensor,
        mut model_in: TensorImpl[Self.MADT],
        ctx: DeviceContext,
    ) raises:
        """bf16-flow only: cast the fp32 staging `batch_xf` → `model_in` (MADT).
        @staticmethod (explicit refs) so it doesn't alias `self` at the call
        sites (where both `batch_xf` and `batch_x` are `self` fields)."""
        comptime NX = Self.BATCH * Self.IN
        model_in.ensure_gpu(ctx, NX)
        ctx.enqueue_function[_cast_kernel[DT, Self.MADT, NX]](
            batch_xf.lt["gpu", Layout.row_major(NX)](),
            model_in.lt["gpu", Layout.row_major(NX)](),
            grid_dim=(NX + TPB - 1) // TPB,
            block_dim=TPB,
        )

    # ── CUDA-graph capture path (fp32): gather + compute, one graph ─────────

    def _compute_step_device(
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        """The device compute of one batch: zero_grad → forward → CE accumulate
        → CE vjp → model.vjp → opt.step, on the FIXED owned `batch_x`/`batch_y`.
        fp32-only (reaching here ⇒ `CAPTURE` ⇒ `MADT == DT`); the CE loss folds
        into the device accumulator, drained at epoch end via `read_accum`."""
        self.opt.zero_grad["gpu"](self.model, ctx)
        Self._amp_train_step[Self.BATCH](
            self.model,
            self.opt,
            self.loss,
            self.batch_x,
            self.batch_y,
            self.logits,
            self.grad,
            self.gi,
            self.logits_f32,
            self.grad_f32,
            ctx,
        )

    def _epoch_captured[
        N_TRAIN: Int, USE_AUG: Bool
    ](
        mut self,
        c: DeviceContext,
        mut aug: Tensor,
        mut indices: TensorImpl[DType.int32],
        mut bctr: TensorImpl[DType.int32],
    ) raises:
        """One epoch on the capture path. The step — gather batch `bctr` of the
        permutation `indices` from `aug` (USE_AUG) or the resident `ds_x` into
        `batch_x`/`batch_y`, advance `bctr`, then the device compute — is
        captured on the first batch of the run and replayed for every other.
        The caller rewrote `aug` and `indices` in place for this epoch."""
        comptime n_batches = N_TRAIN // Self.BATCH
        comptime blocks_gx = (Self.BATCH * Self.IN + TPB - 1) // TPB
        comptime blocks_gy = (Self.BATCH * Self.NC + TPB - 1) // TPB
        comptime LX = Layout.row_major(N_TRAIN, Self.IN)
        c.enqueue_function[reset_counter_kernel](
            bctr.lt["gpu", Layout.row_major(1)](), grid_dim=1, block_dim=1
        )
        for _ in range(n_batches):
            # Move the slot into a disjoint local (`take` leaves it None) so the
            # closure can borrow `self` without overlapping the slot's borrow.
            var g = Optional[CUDAGraph](None)
            if self._train_graph:
                g = Optional[CUDAGraph](self._train_graph.take())

            def _cap() capturing raises -> None:
                var src = self.ds_x.lt["gpu", LX]()
                comptime if USE_AUG:
                    src = aug.lt["gpu", LX]()
                c.enqueue_function[
                    gather_rows_at_counter_kernel[
                        N_TRAIN, Self.BATCH, Self.IN, DT
                    ]
                ](
                    rebind[Tensor](self.batch_x).lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.IN)
                    ](),
                    src,
                    indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                    bctr.lt["gpu", Layout.row_major(1)](),
                    grid_dim=blocks_gx,
                    block_dim=TPB,
                )
                c.enqueue_function[
                    gather_rows_at_counter_kernel[
                        N_TRAIN, Self.BATCH, Self.NC, DT
                    ]
                ](
                    self.batch_y.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.NC)
                    ](),
                    self.ds_y.lt["gpu", Layout.row_major(N_TRAIN, Self.NC)](),
                    indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                    bctr.lt["gpu", Layout.row_major(1)](),
                    grid_dim=blocks_gy,
                    block_dim=TPB,
                )
                c.enqueue_function[advance_counter_kernel](
                    bctr.lt["gpu", Layout.row_major(1)](),
                    grid_dim=1,
                    block_dim=1,
                )
                self._compute_step_device(Optional(c))

            maybe_capture_replay[_cap](g, c)
            self._train_graph = g^

    def train_epoch[
        N_TRAIN: Int
    ](
        mut self,
        ref train_x: List[Scalar[DT]],
        ref train_y: List[Scalar[DT]],
        ctx: Optional[DeviceContext],
    ) raises -> Scalar[DT]:
        comptime n_batches = N_TRAIN // Self.BATCH
        comptime if Self.target == "gpu":
            if not self._up_train:
                Self._upload_dataset(
                    self.ds_x, train_x, N_TRAIN * Self.IN, ctx.value()
                )
                Self._upload_dataset(
                    self.ds_y, train_y, N_TRAIN * Self.NC, ctx.value()
                )
                self._up_train = True
        self.loss.reset_accum[Self.target]()
        for nb in range(n_batches):
            var x0 = nb * Self.BATCH * Self.IN
            var y0 = nb * Self.BATCH * Self.NC
            comptime if Self.target == "cpu":
                # CPU is fp32-only → MADT == DT; rebind batch_x to the fp32 slab.
                ref bx = rebind[Tensor](self.batch_x)
                unsafe_memcpy(
                    dest=bx.data.unsafe_ptr(),
                    src=train_x.unsafe_ptr().unsafe_offset(x0),
                    count=Self.BATCH * Self.IN,
                )
                unsafe_memcpy(
                    dest=self.batch_y.data.unsafe_ptr(),
                    src=train_y.unsafe_ptr().unsafe_offset(y0),
                    count=Self.BATCH * Self.NC,
                )
            else:
                self._slice_train(x0, y0)
                # bf16: cast the fp32 staging (batch_xf) → batch_x (MADT).
                comptime if Self.MADT != DT:
                    Self._cast_input_to_madt(
                        self.batch_xf, self.batch_x, ctx.value()
                    )
            self.opt.zero_grad[Self.target](self.model, ctx)
            Self._amp_train_step[Self.BATCH](
                self.model,
                self.opt,
                self.loss,
                self.batch_x,
                self.batch_y,
                self.logits,
                self.grad,
                self.gi,
                self.logits_f32,
                self.grad_f32,
                ctx,
            )
        return self.loss.read_accum[Self.target](ctx)

    def eval_top1[
        N_TEST: Int
    ](
        mut self,
        ref test_x: List[Scalar[DT]],
        ref test_labels: List[Int32],
        ctx: Optional[DeviceContext],
    ) raises -> Float64:
        comptime n_batches = N_TEST // Self.BATCH
        comptime if Self.target == "gpu":
            if not self._up_test:
                Self._upload_dataset(
                    self.ds_tx, test_x, N_TEST * Self.IN, ctx.value()
                )
                self._up_test = True
        var correct = 0
        comptime LOGN = Self.BATCH * Self.NC
        for nb in range(n_batches):
            var x0 = nb * Self.BATCH * Self.IN
            comptime if Self.target == "cpu":
                # CPU is fp32-only → MADT == DT; rebind batch_x to the fp32 slab.
                ref bx = rebind[Tensor](self.batch_x)
                unsafe_memcpy(
                    dest=bx.data.unsafe_ptr(),
                    src=test_x.unsafe_ptr().unsafe_offset(x0),
                    count=Self.BATCH * Self.IN,
                )
            else:
                comptime if Self.MADT == DT:
                    ref bx = rebind[Tensor](self.batch_x)
                    comptime if Self.CAPTURE:
                        # `batch_x` is a FIXED owned buffer on the capture path —
                        # D2D-copy the test slice INTO it (repointing would
                        # replace the owned buffer with a view, corrupting both
                        # the captured-train invariant AND `ds_tx` on the next
                        # epoch's train copy into `batch_x`).
                        var sx = self.ds_tx.dev.value().create_sub_buffer[DT](
                            x0, Self.BATCH * Self.IN
                        )
                        ctx.value().enqueue_copy(bx.dev.value(), sx)
                    else:
                        bx.dev = Optional(
                            self.ds_tx.dev.value().create_sub_buffer[DT](
                                x0, Self.BATCH * Self.IN
                            )
                        )
                        bx.n = Self.BATCH * Self.IN
                else:
                    # bf16: fp32 view → batch_xf, then cast → batch_x (MADT).
                    self.batch_xf.dev = Optional(
                        self.ds_tx.dev.value().create_sub_buffer[DT](
                            x0, Self.BATCH * Self.IN
                        )
                    )
                    self.batch_xf.n = Self.BATCH * Self.IN
                    Self._cast_input_to_madt(
                        self.batch_xf, self.batch_x, ctx.value()
                    )
            self.model.forward[Self.target, Self.BATCH, POLICY=Self.POLICY](
                child_refs[Self.MODEL.ARITY, Self.MADT](self.batch_x),
                self.logits,
                ctx,
            )
            comptime if Self.MADT == DT:
                # fp32 path (unchanged): download logits, argmax on fp32 `data`.
                ref logd = rebind[Tensor](self.logits)
                comptime if Self.target == "gpu":
                    logd.download(ctx.value())
                for b in range(Self.BATCH):
                    var best = 0
                    var bestv = logd.data[b * Self.NC]
                    for c in range(1, Self.NC):
                        var v = logd.data[b * Self.NC + c]
                        if v > bestv:
                            bestv = v
                            best = c
                    if best == Int(test_labels[nb * Self.BATCH + b]):
                        correct += 1
            else:
                # bf16: cast MADT logits → fp32 logits_f32, download, argmax.
                var c = ctx.value()
                self.logits_f32.ensure_gpu(c, LOGN)
                c.enqueue_function[_cast_kernel[Self.MADT, DT, LOGN]](
                    self.logits.lt["gpu", Layout.row_major(LOGN)](),
                    self.logits_f32.lt["gpu", Layout.row_major(LOGN)](),
                    grid_dim=(LOGN + TPB - 1) // TPB,
                    block_dim=TPB,
                )
                self.logits_f32.download(c)
                for b in range(Self.BATCH):
                    var best = 0
                    var bestv = self.logits_f32.data[b * Self.NC]
                    for cc in range(1, Self.NC):
                        var v = self.logits_f32.data[b * Self.NC + cc]
                        if v > bestv:
                            bestv = v
                            best = cc
                    if best == Int(test_labels[nb * Self.BATCH + b]):
                        correct += 1
        return Float64(correct) / Float64(n_batches * Self.BATCH)

    def train_gpu[
        N_TRAIN: Int,
        N_TEST: Int,
        AUGMENTER: Augmenter = IdentityAugmenter,
        SCHEDULER: Scheduler = ConstantSchedule,
    ](
        mut self,
        ref train_x: List[Scalar[DT]],
        ref train_y: List[Scalar[DT]],
        ref test_x: List[Scalar[DT]],
        ref test_labels: List[Int32],
        ctx: Optional[DeviceContext],
        epochs: Int = 1,
        print_progress: Bool = True,
        shuffle: Bool = False,
        rng_seed: UInt64 = 42,
        aug_seed: UInt64 = 1000,
    ) raises -> TrainResult:
        """Whole-dataset GPU training run with per-epoch top-1 eval — the
        one-call convenience over `train_epoch`/`eval_top1`. Uploads train+test
        ONCE (resident), then each epoch trains over all batches and evaluates.

        `shuffle=True` runs an on-device Fisher-Yates permutation of the train
        set each epoch (device-resident indices/seed, gather into owned batch
        buffers — capture-friendly, no host involvement); `shuffle=False` slices
        contiguous sub-buffer views (zero-copy).

        `AUGMENTER` (default identity): when not a no-op, an augmentation buffer
        is allocated once and `AUGMENTER.augment` FULLY rewrites it from the raw
        resident train set each epoch; training then reads the augmented buffer
        (the labels are never augmented). `SCHEDULER` (default constant): when not
        constant, the optimizer LR is set to `base_lr * lr_scale_at(epoch,
        epochs)` each epoch (base_lr = the LR on the optimizer before the call).

        BatchNorm is toggled to train mode before each epoch and eval mode before
        the test pass via `set_attr["training"]` (no-op for nets without BN)."""
        comptime assert (
            Self.target == "gpu"
        ), "Trainer.train_gpu requires target='gpu'"
        comptime assert (
            N_TRAIN % Self.BATCH == 0
        ), "train_gpu: N_TRAIN must be divisible by BATCH"
        comptime assert (
            N_TEST % Self.BATCH == 0
        ), "train_gpu: N_TEST must be divisible by BATCH"
        comptime n_batches = N_TRAIN // Self.BATCH
        comptime blocks_init = (N_TRAIN + TPB - 1) // TPB
        comptime blocks_gx = (Self.BATCH * Self.IN + TPB - 1) // TPB
        comptime blocks_gy = (Self.BATCH * Self.NC + TPB - 1) // TPB
        comptime USE_AUG = not AUGMENTER.IS_NOOP
        comptime USE_SCHED = not SCHEDULER.IS_CONSTANT
        var c = ctx.value()
        var result = TrainResult.empty()

        # Resident train set (once). Test is uploaded lazily by eval_top1.
        if not self._up_train:
            Self._upload_dataset(self.ds_x, train_x, N_TRAIN * Self.IN, c)
            Self._upload_dataset(self.ds_y, train_y, N_TRAIN * Self.NC, c)
            self._up_train = True

        # Augmentation working buffer (X only; rewritten from raw ds_x each
        # epoch). Labels (ds_y) are never augmented.
        var aug = Tensor()
        comptime if USE_AUG:
            aug = Tensor.alloc_gpu(c, N_TRAIN * Self.IN)

        # LR schedule: capture the caller-set base LR; per-epoch LR is set to
        # base_lr * scale. Constant default leaves the LR exactly as set.
        var base_lr = self.opt.get_lr()

        # Shuffle scratch (device-resident; allocated once, reused per epoch).
        # Owned gather targets `gx`/`gy` — NOT self.batch_x (which eval_top1
        # repoints at a test sub-buffer, so reusing it as a gather target would
        # corrupt the test set).
        var indices = TensorImpl[DType.int32]()
        var seed = TensorImpl[DType.uint64]()
        var gx = Tensor()  # fp32 gather target (X)
        var gy = Tensor()  # fp32 gather target (labels — always fp32 for loss)
        # bf16: owned MADT model-input (gx is cast into it). Unused for fp32.
        var gxm = TensorImpl[Self.MADT]()
        # Capture: the device batch counter, and the permutation even without
        # shuffle (identity = the contiguous sweep).
        var bctr = TensorImpl[DType.int32]()
        comptime if Self.CAPTURE:
            bctr = TensorImpl[DType.int32].alloc_gpu(c, 1)
            if not shuffle:
                indices = TensorImpl[DType.int32].alloc_gpu(c, N_TRAIN)
                c.enqueue_function[init_identity_indices_kernel[N_TRAIN]](
                    indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                    grid_dim=blocks_init,
                    block_dim=TPB,
                )
        if shuffle:
            indices = TensorImpl[DType.int32].alloc_gpu(c, N_TRAIN)
            c.enqueue_function[init_identity_indices_kernel[N_TRAIN]](
                indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                grid_dim=blocks_init,
                block_dim=TPB,
            )
            seed.ensure(1)
            seed.data[0] = rng_seed
            seed.n = 1
            seed.upload(c)
            gx = Tensor.alloc_gpu(c, Self.BATCH * Self.IN)
            gy = Tensor.alloc_gpu(c, Self.BATCH * Self.NC)
            comptime if Self.MADT != DT:
                gxm = TensorImpl[Self.MADT].alloc_gpu(c, Self.BATCH * Self.IN)

        comptime if Self.CAPTURE:
            # Every version-gated weight cache (Linear's padded `w_pad`, the
            # bf16 casts) refreshes on EVERY forward: a replay never bumps the
            # host `version`, so the gate would let the eager eval after an
            # epoch of replays read a cache one optimizer step stale.
            self.model.set_attr["capture_recast"](Scalar[DT](1.0))
        for epoch in range(epochs):
            var t0 = perf_counter_ns()
            comptime if Self.CAPTURE:
                # LR onto the device so the captured `opt.step` reads the FRESH
                # per-epoch LR on each replay (a host-baked `lr` would freeze at
                # the capture-time value).
                var eplr = base_lr
                comptime if USE_SCHED:
                    eplr = base_lr * Scalar[DT](
                        SCHEDULER.lr_scale_at(epoch, epochs)
                    )
                self.opt.push_lr_device["gpu"](eplr, Optional(c))
            else:
                comptime if USE_SCHED:
                    self.opt.set_lr(
                        base_lr
                        * Scalar[DT](SCHEDULER.lr_scale_at(epoch, epochs))
                    )
            self.loss.reset_accum["gpu"]()
            self.model.set_attr["training"](Scalar[DT](1.0))
            # (Re)build the augmented training set from the raw resident set,
            # then this epoch's permutation (both in place, before any batch).
            comptime if USE_AUG:
                AUGMENTER.augment[N_TRAIN, Self.IN, DT](
                    c,
                    aug.lt["gpu", Layout.row_major(N_TRAIN, Self.IN)](),
                    self.ds_x.lt["gpu", Layout.row_major(N_TRAIN, Self.IN)](),
                    epoch,
                    aug_seed,
                )
            if shuffle:
                c.enqueue_function[fisher_yates_shuffle_kernel[N_TRAIN]](
                    indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                    seed.lt["gpu", Layout.row_major(1)](),
                    grid_dim=1,
                    block_dim=1,
                )
                c.enqueue_function[increment_seed_kernel](
                    seed.lt["gpu", Layout.row_major(1)](),
                    grid_dim=1,
                    block_dim=1,
                )
            comptime if Self.CAPTURE:
                self._epoch_captured[N_TRAIN, USE_AUG](c, aug, indices, bctr)
            else:
                for nb in range(n_batches):
                    if shuffle:
                        var off = nb * Self.BATCH
                        # Gather X from the augmented set (if any) else raw ds_x.
                        comptime if USE_AUG:
                            c.enqueue_function[
                                gather_rows_kernel[
                                    N_TRAIN, Self.BATCH, Self.IN, DT
                                ]
                            ](
                                gx.lt[
                                    "gpu", Layout.row_major(Self.BATCH, Self.IN)
                                ](),
                                aug.lt[
                                    "gpu", Layout.row_major(N_TRAIN, Self.IN)
                                ](),
                                indices.lt[
                                    "gpu", Layout.row_major(N_TRAIN)
                                ](),
                                Int32(off),
                                grid_dim=blocks_gx,
                                block_dim=TPB,
                            )
                        else:
                            c.enqueue_function[
                                gather_rows_kernel[
                                    N_TRAIN, Self.BATCH, Self.IN, DT
                                ]
                            ](
                                gx.lt[
                                    "gpu", Layout.row_major(Self.BATCH, Self.IN)
                                ](),
                                self.ds_x.lt[
                                    "gpu", Layout.row_major(N_TRAIN, Self.IN)
                                ](),
                                indices.lt[
                                    "gpu", Layout.row_major(N_TRAIN)
                                ](),
                                Int32(off),
                                grid_dim=blocks_gx,
                                block_dim=TPB,
                            )
                        c.enqueue_function[
                            gather_rows_kernel[N_TRAIN, Self.BATCH, Self.NC, DT]
                        ](
                            gy.lt[
                                "gpu", Layout.row_major(Self.BATCH, Self.NC)
                            ](),
                            self.ds_y.lt[
                                "gpu", Layout.row_major(N_TRAIN, Self.NC)
                            ](),
                            indices.lt["gpu", Layout.row_major(N_TRAIN)](),
                            Int32(off),
                            grid_dim=blocks_gy,
                            block_dim=TPB,
                        )
                        self.opt.zero_grad["gpu"](self.model, ctx)
                        # fp32: gx IS the model input. bf16: cast gx → gxm (MADT).
                        comptime if Self.MADT == DT:
                            ref mi = rebind[TensorImpl[Self.MADT]](gx)
                            Self._amp_train_step[Self.BATCH](
                                self.model, self.opt, self.loss,
                                mi, gy, self.logits, self.grad, self.gi,
                                self.logits_f32, self.grad_f32, ctx,
                            )
                        else:
                            gxm.ensure_gpu(c, Self.BATCH * Self.IN)
                            c.enqueue_function[
                                _cast_kernel[
                                    DT, Self.MADT, Self.BATCH * Self.IN
                                ]
                            ](
                                gx.lt[
                                    "gpu",
                                    Layout.row_major(Self.BATCH * Self.IN),
                                ](),
                                gxm.lt[
                                    "gpu",
                                    Layout.row_major(Self.BATCH * Self.IN),
                                ](),
                                grid_dim=blocks_gx,
                                block_dim=TPB,
                            )
                            Self._amp_train_step[Self.BATCH](
                                self.model, self.opt, self.loss,
                                gxm, gy, self.logits, self.grad, self.gi,
                                self.logits_f32, self.grad_f32, ctx,
                            )
                    else:
                        var x0 = nb * Self.BATCH * Self.IN
                        var y0 = nb * Self.BATCH * Self.NC
                        # Sub-buffer view of X from the augmented set else raw
                        # ds_x. fp32: the view lands in batch_x (zero-copy).
                        # bf16: it lands in the fp32 staging batch_xf (cast →
                        # batch_x below).
                        comptime if Self.MADT == DT:
                            ref bx = rebind[Tensor](self.batch_x)
                            comptime if USE_AUG:
                                bx.dev = Optional(
                                    aug.dev.value().create_sub_buffer[DT](
                                        x0, Self.BATCH * Self.IN
                                    )
                                )
                            else:
                                bx.dev = Optional(
                                    self.ds_x.dev.value().create_sub_buffer[DT](
                                        x0, Self.BATCH * Self.IN
                                    )
                                )
                            bx.n = Self.BATCH * Self.IN
                        else:
                            comptime if USE_AUG:
                                self.batch_xf.dev = Optional(
                                    aug.dev.value().create_sub_buffer[DT](
                                        x0, Self.BATCH * Self.IN
                                    )
                                )
                            else:
                                self.batch_xf.dev = Optional(
                                    self.ds_x.dev.value().create_sub_buffer[DT](
                                        x0, Self.BATCH * Self.IN
                                    )
                                )
                            self.batch_xf.n = Self.BATCH * Self.IN
                            Self._cast_input_to_madt(
                                self.batch_xf, self.batch_x, c
                            )
                        self.batch_y.dev = Optional(
                            self.ds_y.dev.value().create_sub_buffer[DT](
                                y0, Self.BATCH * Self.NC
                            )
                        )
                        self.batch_y.n = Self.BATCH * Self.NC
                        self.opt.zero_grad["gpu"](self.model, ctx)
                        Self._amp_train_step[Self.BATCH](
                            self.model, self.opt, self.loss,
                            self.batch_x, self.batch_y, self.logits, self.grad,
                            self.gi, self.logits_f32, self.grad_f32, ctx,
                        )
            var avg = Float64(self.loss.read_accum["gpu"](ctx))
            c.synchronize()
            var t1 = perf_counter_ns()
            self.model.set_attr["training"](Scalar[DT](0.0))
            var top1 = self.eval_top1[N_TEST](test_x, test_labels, ctx)
            var t2 = perf_counter_ns()
            result.epoch_train_loss.append(avg)
            result.epoch_test_top1.append(top1)
            result.epoch_train_s.append(Float64(t1 - t0) / 1e9)
            result.epoch_eval_s.append(Float64(t2 - t1) / 1e9)
            if print_progress:
                print(
                    "epoch " + String(epoch)
                    + " | train_loss=" + String(avg)
                    + " | test_top1=" + String(top1 * 100.0) + "%"
                    + " | train=" + String(Float64(t1 - t0) / 1e9) + "s"
                    + " | eval=" + String(Float64(t2 - t1) / 1e9) + "s"
                )
        return result^
