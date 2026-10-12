"""AutoregressiveTrainer — a task-specialized driver for next-token LM
training on the **storage** surface (CPU `List` / GPU `DeviceBuffer` `Tensor`s,
`TensorRefs` input packs).

The storage `Trainer` (`trainer.mojo`) is classification-specialized
(`CrossEntropyLoss[NC]` + `train_epoch`/`eval_top1` over a flat resident
dataset), so — unlike the legacy `AutoregressiveTrainer`, which *wrapped* a
generic per-step `Trainer` — this driver is **self-contained**: it owns the
`net` / `opt` / `SequenceCrossEntropyLoss` / scratch `Tensor`s and runs its own
per-step loop (forward → SeqCE loss → vjp → grad-clip → optimizer step) over
streaming random windows, with a cosine-with-warmup LR schedule, per-token CE
eval, and text generation. The example collapses to

    var artr = AutoregressiveTrainer[NET, OPT, VOCAB, SEQ, BATCH].make_from(
        net^, opt^, tok^, split^, ctx, base_lr, warmup_iters, total_iters,
        grad_clip=1.0,
    )
    # model-construction surgery on artr.net here (scaled-init / tie wiring)
    artr.fit(eval_every=250, n_val_windows=256)
    print(artr.generate("ROMEO:", 200, temperature=0.8))

It wraps the *regime*, not the model: the model / optimizer / loss live as
fields, and `make_from` engages the optimizer's contiguous arena (`opt.adopt`)
once `net` is in its final home. Any causal sequence model using the one-hot
`[SEQ·VOCAB]` in / logits `[SEQ·VOCAB]` out convention works (char- or BPE-
token, weight-tied or not). Model-specific init surgery (e.g. the GPT's c_proj
scaling / weight-tie wiring) stays at the call site — this driver doesn't know
about it.

GPU-only for now (eval + generation use device buffers). A recurrent model
(LSTM) can't use it until its recurrence is wrapped behind `Module.forward`/
`vjp` — the per-step loop drives those.
"""

from std.math import exp, cos
from std.random import random_float64, random_ui64
from max.gpu.host import DeviceContext
from layout import Layout

from noeira.nn.constants import DT, TPB
from ..core.module import Module
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import child_refs
from ..optimizer.optimizer import Optimizer
from ..optimizer.adam import Adam
from ..loss.sequence_cross_entropy import SequenceCrossEntropyLoss
from .trainer import _cast_kernel  # AMP boundary cast (fp32 ↔ MADT)
from noeira.nn.datasets import CharTokenizer, DatasetSplit, make_batch
from noeira.cuda import CUDAGraph, maybe_capture_replay
from noeira.nn.core.ptr import mptr
from .window_batch_kernels import window_onehot_kernel, advance_step_kernel


struct AutoregressiveTrainer[
    NET: Module,
    OPT: Optimizer,
    VOCAB: Int,
    SEQ: Int,
    BATCH: Int,
    target: StaticString = "gpu",
    # Opt-in CUDA-graph capture of the per-step DEVICE compute (forward → SeqCE
    # accumulate → vjp → grad-clip → opt.step). With DEVICE_BATCH the batch
    # build is captured too; without it the host build (sampling + one-hot +
    # upload into the PERSISTENT `in_t`/`tgt_t` buffers) stays eager and the
    # captured graph reads those fixed buffers. fp32-only (the
    # AMP step has a host-side version bump). Default OFF — on NVIDIA the nn
    # GEMM (`linalg.matmul`) allocates a split-K workspace per call, illegal
    # under stream capture; enable only once that's resolved. No-op on
    # non-NVIDIA (runs eagerly, bit-identical).
    USE_TRAIN_CUDA_GRAPH: Bool = False,
    # Training batches built ON THE DEVICE (`window_batch_kernels`): the train
    # corpus resident as int32, window starts drawn from a counter hash, the
    # one-hots written by a kernel — no host loop, no H2D, no wait on the
    # previous step, and the build rides inside the captured graph. False =
    # the host `make_batch` + one-hot + upload (the before/after baseline).
    # Eval and generation build on the host either way.
    DEVICE_BATCH: Bool = True,
](Movable):
    comptime IN_DIM = Self.SEQ * Self.VOCAB
    comptime OUT_DIM = Self.SEQ * Self.VOCAB
    comptime LOSS = SequenceCrossEntropyLoss[Self.SEQ, Self.VOCAB]
    # AMP: the net's activation dtype. `DT` for an fp32 net, `bfloat16` for a
    # bf16-flow net (e.g. GPT with all-bf16 leaves). Activations (input one-hot,
    # logits, grad_output, grad_input) flow at MADT; the loss + master weights
    # stay fp32 (the SeqCE softmax needs fp32). `MADT == DT` ⇒ zero casts.
    comptime MADT = Self.NET.ACT_DT
    # Capture covers fp32 AND bf16-flow. For bf16 the cached-bf16 weight recast
    # is a DEVICE kernel that must ride the captured graph and read the live fp32
    # master on every replay; `fit` sets `set_attr["capture_recast"]` so every
    # bf16 leaf recasts UNCONDITIONALLY (not version-gated) → the cast is always
    # recorded and never silently stale. The host-side `ParamVersionBump` not
    # running on replay is harmless (the unconditional recast supersedes it).
    comptime CAPTURE = Self.USE_TRAIN_CUDA_GRAPH

    var net: Self.NET
    var opt: Self.OPT
    var loss: Self.LOSS
    var ctx: DeviceContext
    # Reused per-step staging. Net-facing activations flow at MADT; `tgt_t` is
    # the loss target (fp32). `logits_f32`/`grad_f32` are the fp32 loss-boundary
    # scratch used ONLY on the bf16 path (untouched when MADT == DT).
    var in_t: TensorImpl[Self.MADT]
    var tgt_t: Tensor
    var logits: TensorImpl[Self.MADT]
    var grad: TensorImpl[Self.MADT]
    var gi: TensorImpl[Self.MADT]
    var logits_f32: Tensor
    var grad_f32: Tensor
    var tok: CharTokenizer
    var train_ids: List[Int]
    var val_ids: List[Int]
    # Cosine-with-warmup LR schedule (per-iter scale on base_lr).
    var base_lr: Scalar[DT]
    var warmup_iters: Int
    var total_iters: Int
    var min_lr_scale: Float64
    var grad_clip: Scalar[DT]
    # Lazily-captured per-step compute graph (None until the first capture). Only
    # touched on the `CAPTURE` path; a no-op slot otherwise.
    var _train_graph: Optional[CUDAGraph]
    # DEVICE_BATCH: the resident train corpus (int32), [seed, step], and the
    # last batch's window starts (read by the gate only).
    var corpus_dev: TensorImpl[DType.int32]
    var rng_dev: TensorImpl[DType.uint64]
    var starts_dev: TensorImpl[DType.int32]

    def __init__(
        out self,
        var net: Self.NET,
        var opt: Self.OPT,
        var loss: Self.LOSS,
        ctx: DeviceContext,
        var tok: CharTokenizer,
        var train_ids: List[Int],
        var val_ids: List[Int],
        base_lr: Scalar[DT],
        warmup_iters: Int,
        total_iters: Int,
        min_lr_scale: Float64,
        grad_clip: Scalar[DT],
    ):
        self.net = net^
        self.opt = opt^
        self.loss = loss^
        self.ctx = ctx
        self.in_t = TensorImpl[Self.MADT]()
        self.tgt_t = Tensor()
        self.logits = TensorImpl[Self.MADT]()
        self.grad = TensorImpl[Self.MADT]()
        self.gi = TensorImpl[Self.MADT]()
        self.logits_f32 = Tensor()
        self.grad_f32 = Tensor()
        self.tok = tok^
        self.train_ids = train_ids^
        self.val_ids = val_ids^
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr_scale = min_lr_scale
        self.grad_clip = grad_clip
        self._train_graph = None
        self.corpus_dev = TensorImpl[DType.int32]()
        self.rng_dev = TensorImpl[DType.uint64]()
        self.starts_dev = TensorImpl[DType.int32]()

    # ----- Factory --------------------------------------------------------

    @staticmethod
    def make_from(
        var net: Self.NET,
        var opt: Self.OPT,
        var tok: CharTokenizer,
        var split: DatasetSplit,
        ctx: DeviceContext,
        base_lr: Scalar[DT],
        warmup_iters: Int,
        total_iters: Int,
        min_lr_scale: Float64 = 0.1,
        grad_clip: Scalar[DT] = 0.0,
    ) raises -> Self:
        """Wrap a built `net` + `opt` + corpus + LR schedule. The model is
        expected to already be in its final home; do any model-construction
        surgery (tie wiring, scaled init) on `result.net` AFTER this call (the
        net won't move again). Engages the optimizer's contiguous arena
        (`opt.adopt`) here so the GPU step is a single grouped kernel and
        `clip_grads` runs on the arena (capture-safe, no per-param D2H)."""
        comptime assert (
            Self.target == "gpu"
        ), "AutoregressiveTrainer is GPU-only for now (eval/gen use device)"
        var loss = Self.LOSS.make_gpu(ctx)
        var s = Self(
            net^, opt^, loss^, ctx, tok^,
            split.train.copy(), split.val.copy(),
            base_lr, warmup_iters, total_iters, min_lr_scale, grad_clip,
        )
        # Arena adopt AFTER net is in its final home (self.net): rebinds the
        # model's param buffers to arena slices in place, so any subsequent
        # tie-wiring (which points the head at the embedding `Tensor` cells)
        # and scaled-init surgery operate on the arena-backed buffers.
        s.opt.adopt[Self.target](s.net, Optional(s.ctx))
        comptime if Self.DEVICE_BATCH:
            s._init_device_batch()
        return s^

    def _init_device_batch(mut self) raises:
        """Upload the train corpus as int32 and seed the window sampler from
        the process RNG (reproducible under `std.random.seed`). Allocates the
        one-hot buffers at full batch size so the graph's pointers never move."""
        var n = len(self.train_ids)
        if n < Self.SEQ + 1:
            raise Error("DEVICE_BATCH: train corpus shorter than SEQ + 1")
        self.corpus_dev.ensure(n)
        for i in range(n):
            self.corpus_dev.data[i] = Int32(self.train_ids[i])
        self.corpus_dev.n = n
        self.corpus_dev.upload(self.ctx)
        self.rng_dev.ensure(2)
        self.rng_dev.data[0] = random_ui64(0, UInt64.MAX)
        self.rng_dev.data[1] = 0
        self.rng_dev.n = 2
        self.rng_dev.upload(self.ctx)
        self.starts_dev = TensorImpl[DType.int32].alloc_gpu(self.ctx, Self.BATCH)
        comptime TOTAL = Self.BATCH * Self.IN_DIM
        self.in_t.n = TOTAL
        self.in_t.ensure_gpu(self.ctx, TOTAL)
        self.tgt_t.n = TOTAL
        self.tgt_t.ensure_gpu(self.ctx, TOTAL)

    def _device_batch(mut self) raises:
        """Enqueue one training batch's build into `in_t` / `tgt_t` (device
        only: capture-safe). """
        comptime TOTAL = Self.BATCH * Self.IN_DIM
        # `in_t` / `tgt_t` keep their full-batch allocation: eval refills them
        # at the same size, generation uses its own scratch.
        self.in_t.n = TOTAL
        self.tgt_t.n = TOTAL
        self.ctx.enqueue_function[
            window_onehot_kernel[Self.MADT, Self.BATCH, Self.SEQ, Self.VOCAB]
        ](
            mptr(self.corpus_dev.dev.value().unsafe_ptr()),
            Int64(self.corpus_dev.n - Self.SEQ),
            self.rng_dev.lt["gpu", Layout.row_major(2)](),
            self.in_t.lt["gpu", Layout.row_major(TOTAL)](),
            self.tgt_t.lt["gpu", Layout.row_major(TOTAL)](),
            self.starts_dev.lt["gpu", Layout.row_major(Self.BATCH)](),
            grid_dim=(TOTAL + TPB - 1) // TPB,
            block_dim=TPB,
        )
        self.ctx.enqueue_function[advance_step_kernel](
            self.rng_dev.lt["gpu", Layout.row_major(2)](),
            grid_dim=1,
            block_dim=1,
        )

    def _host_batch(mut self) raises:
        """The host build: sample windows, one-hot, upload (waits for the
        previous upload's copy to drain)."""
        var mb = make_batch(self.train_ids, Self.BATCH, Self.SEQ)
        Self._upload_onehot[Self.MADT](
            self.ctx, self.in_t, mb.inputs, Self.BATCH
        )
        Self._upload_onehot[DT](self.ctx, self.tgt_t, mb.targets, Self.BATCH)

    # ----- Schedule + one-hot helpers ------------------------------------

    def _lr_scale(self, it: Int) -> Scalar[DT]:
        """Linear warmup then cosine decay to `min_lr_scale`."""
        if it < self.warmup_iters:
            return Scalar[DT](Float64(it + 1) / Float64(self.warmup_iters))
        var denom = self.total_iters - self.warmup_iters
        if denom < 1:
            denom = 1
        var prog = Float64(it - self.warmup_iters) / Float64(denom)
        if prog > 1.0:
            prog = 1.0
        var c = 0.5 * (1.0 + cos(3.14159265358979 * prog))
        return Scalar[DT](
            self.min_lr_scale + (1.0 - self.min_lr_scale) * c
        )

    @staticmethod
    def _upload_onehot[
        ADT: DType
    ](
        ctx: DeviceContext, mut t: TensorImpl[ADT], ids: List[Int], n_rows: Int
    ) raises:
        """Host one-hot `[n_rows, SEQ·VOCAB]` from token ids `[n_rows·SEQ]`,
        then H2D upload into `t` (device buffer reallocated to fit). Generic over
        the dtype `ADT`: the net input flows at MADT (built directly at bf16 —
        0/1 are exact), the loss target at fp32. Static (takes `ctx` by value) so
        call sites can pass `self.in_t`/`self.tgt_t` as `mut t` without aliasing
        the whole `self`."""
        var total = n_rows * Self.IN_DIM
        t.ensure(total)
        for i in range(total):
            t.data[i] = Scalar[ADT](0)
        for r in range(n_rows):
            for tt in range(Self.SEQ):
                var tid = ids[r * Self.SEQ + tt]
                if tid >= 0 and tid < Self.VOCAB:
                    t.data[r * Self.IN_DIM + tt * Self.VOCAB + tid] = (
                        Scalar[ADT](1)
                    )
        t.n = total
        # Resident refresh (reuse the device buffer; no realloc) — bit-identical
        # to `upload` but keeps the buffer pointer stable so the captured compute
        # graph reads the SAME `in_t`/`tgt_t` each replay (the host rebuilds the
        # one-hot in place every step before the replay).
        t.upload_resident(ctx)

    # ----- Eval (per-token CE / top-1 over freshly sampled val windows) ---

    def _eval_loss_ids(
        mut self, in_ids: List[Int], tgt_ids: List[Int], n_batches: Int
    ) raises -> Float64:
        """Mean per-token CE (nats) over `n_batches` batches of pre-sampled
        window ids (`[n_batches·BATCH·SEQ]` flat)."""
        self.net.set_attr["training"](Scalar[DT](0.0))  # dropout off
        var total: Float64 = 0.0
        var stride = Self.BATCH * Self.SEQ
        for vb in range(n_batches):
            var bi = List[Int](capacity=stride)
            var bt = List[Int](capacity=stride)
            for k in range(stride):
                bi.append(in_ids[vb * stride + k])
                bt.append(tgt_ids[vb * stride + k])
            Self._upload_onehot[Self.MADT](self.ctx, self.in_t, bi, Self.BATCH)
            Self._upload_onehot[DT](self.ctx, self.tgt_t, bt, Self.BATCH)
            self.net.forward[Self.target, Self.BATCH](
                child_refs[Self.NET.ARITY, Self.MADT](self.in_t),
                self.logits,
                Optional(self.ctx),
            )
            total += self._loss_fwd[Self.BATCH]()
        self.net.set_attr["training"](Scalar[DT](1.0))
        return total / Float64(n_batches)

    def val_loss(mut self, n_windows: Int) raises -> Float64:
        """Mean per-token val CE (nats) over `n_windows` freshly sampled
        windows. `n_windows` must be a multiple of BATCH."""
        if n_windows % Self.BATCH != 0:
            raise Error("val_loss: n_windows must be a multiple of BATCH")
        var mb = make_batch(self.val_ids, n_windows, Self.SEQ)
        return self._eval_loss_ids(
            mb.inputs, mb.targets, n_windows // Self.BATCH
        )

    def val_top1(mut self, n_windows: Int) raises -> Float64:
        """Per-token top-1 argmax accuracy over `n_windows` val windows — a
        diagnostic that the loss is consistent with good next-token prediction
        (not artifactually low)."""
        if n_windows % Self.BATCH != 0:
            raise Error("val_top1: n_windows must be a multiple of BATCH")
        var mb = make_batch(self.val_ids, n_windows, Self.SEQ)
        self.net.set_attr["training"](Scalar[DT](0.0))
        var correct: Int = 0
        var count: Int = 0
        var stride = Self.BATCH * Self.SEQ
        var n_batches = n_windows // Self.BATCH
        for vb in range(n_batches):
            var bi = List[Int](capacity=stride)
            for k in range(stride):
                bi.append(mb.inputs[vb * stride + k])
            Self._upload_onehot[Self.MADT](self.ctx, self.in_t, bi, Self.BATCH)
            self.net.forward[Self.target, Self.BATCH](
                child_refs[Self.NET.ARITY, Self.MADT](self.in_t),
                self.logits,
                Optional(self.ctx),
            )
            self.logits.download(self.ctx)
            for b in range(Self.BATCH):
                for t in range(Self.SEQ):
                    var row = b * Self.OUT_DIM + t * Self.VOCAB
                    var best_v = Float64(self.logits.data[row])
                    var best_i = 0
                    for v in range(1, Self.VOCAB):
                        var x = Float64(self.logits.data[row + v])
                        if x > best_v:
                            best_v = x
                            best_i = v
                    if best_i == mb.targets[vb * stride + b * Self.SEQ + t]:
                        correct += 1
                    count += 1
        self.net.set_attr["training"](Scalar[DT](1.0))
        return Float64(correct) / Float64(count)

    # ----- AMP loss boundary (fp32 SeqCE; casts ONLY on the bf16 path) ----

    @staticmethod
    def _cast_logits_to_f32[
        B: Int
    ](
        mut logits: TensorImpl[Self.MADT],
        mut logits_f32: Tensor,
        ctx: DeviceContext,
    ) raises:
        """bf16-flow only: cast the MADT logits → fp32 `logits_f32` for SeqCE.
        @staticmethod (explicit refs) to avoid aliasing the whole `self`."""
        comptime LOGN = B * Self.OUT_DIM
        logits_f32.ensure_gpu(ctx, LOGN)
        ctx.enqueue_function[_cast_kernel[Self.MADT, DT, LOGN]](
            logits.lt["gpu", Layout.row_major(LOGN)](),
            logits_f32.lt["gpu", Layout.row_major(LOGN)](),
            grid_dim=(LOGN + TPB - 1) // TPB,
            block_dim=TPB,
        )

    @staticmethod
    def _cast_grad_to_madt[
        B: Int
    ](
        mut grad: TensorImpl[Self.MADT],
        mut grad_f32: Tensor,
        ctx: DeviceContext,
    ) raises:
        """bf16-flow only: cast the fp32 loss grad `grad_f32` → MADT `grad`."""
        comptime LOGN = B * Self.OUT_DIM
        grad.ensure_gpu(ctx, LOGN)
        ctx.enqueue_function[_cast_kernel[DT, Self.MADT, LOGN]](
            grad_f32.lt["gpu", Layout.row_major(LOGN)](),
            grad.lt["gpu", Layout.row_major(LOGN)](),
            grid_dim=(LOGN + TPB - 1) // TPB,
            block_dim=TPB,
        )

    def _loss_fwd[B: Int](mut self) raises -> Float64:
        """SeqCE forward through the fp32 boundary → mean batch loss. fp32 (MADT
        == DT): directly on `logits` (rebind, no-op). bf16: cast `logits` →
        `logits_f32` first (the loss/softmax stay fp32)."""
        comptime if Self.MADT == DT:
            ref logd = rebind[Tensor](self.logits)
            return Float64(
                self.loss.forward[Self.target, B](
                    logd, self.tgt_t, Optional(self.ctx)
                )
            )
        else:
            Self._cast_logits_to_f32[B](
                self.logits, self.logits_f32, self.ctx
            )
            return Float64(
                self.loss.forward[Self.target, B](
                    self.logits_f32, self.tgt_t, Optional(self.ctx)
                )
            )

    def _loss_bwd[B: Int](mut self) raises:
        """SeqCE vjp → `grad` (MADT). fp32: directly. bf16: vjp into `grad_f32`
        then cast → `grad`. Requires `_loss_fwd[B]` ran first (logits_f32 set)."""
        comptime if Self.MADT == DT:
            ref logd = rebind[Tensor](self.logits)
            ref grdd = rebind[Tensor](self.grad)
            self.loss.vjp[Self.target, B](
                logd, self.tgt_t, grdd, Optional(self.ctx)
            )
        else:
            self.loss.vjp[Self.target, B](
                self.logits_f32, self.tgt_t, self.grad_f32, Optional(self.ctx)
            )
            Self._cast_grad_to_madt[B](
                self.grad, self.grad_f32, self.ctx
            )

    # ----- One packaged training step ------------------------------------

    def _train_step(mut self, it: Int) raises -> Float64:
        """Sample a fresh random window batch, apply the cosine-warmup LR, and
        run one forward → SeqCE → vjp → grad-clip → optimizer step. Returns the
        (pre-step) train CE for this batch."""
        self.opt.set_lr(self.base_lr * self._lr_scale(it))
        comptime if Self.DEVICE_BATCH:
            self._device_batch()
        else:
            self._host_batch()
        self.opt.zero_grad[Self.target](self.net, Optional(self.ctx))
        self.net.forward[Self.target, Self.BATCH](
            child_refs[Self.NET.ARITY, Self.MADT](self.in_t),
            self.logits,
            Optional(self.ctx),
        )
        # SeqCE runs at fp32 (cast only on the bf16 path), grad cast back to MADT.
        var tl = self._loss_fwd[Self.BATCH]()
        self._loss_bwd[Self.BATCH]()
        self.net.vjp[Self.target, Self.BATCH](
            child_refs[Self.NET.ARITY, Self.MADT](self.in_t),
            self.grad,
            child_refs[Self.NET.ARITY, Self.MADT](self.gi),
            Optional(self.ctx),
        )
        if self.grad_clip > Scalar[DT](0.0):
            _ = self.opt.clip_grads[Self.target](
                self.net, self.grad_clip, Optional(self.ctx)
            )
        self.opt.step[Self.target](self.net, Optional(self.ctx))
        return tl

    # ----- CUDA-graph capture path (compute-only) ------------------------

    def _compute_step_device(mut self) raises:
        """The PURE-DEVICE compute captured into the graph: zero_grad → forward →
        SeqCE accumulate → SeqCE vjp → net.vjp → (device grad-clip) → opt.step.

        DEVICE_BATCH: first builds the batch into `in_t`/`tgt_t` (device
        kernels, captured). Otherwise reads them as the host refreshed them
        eagerly before the replay. The per-step train CE is folded into the loss's DEVICE
        accumulator (drained at flush) — never read here, since a D2H would break
        capture. Must enqueue the SAME kernel sequence every call.

        fp32 (MADT == DT): the loss runs directly on the (rebound) MADT buffers.
        bf16: the SeqCE boundary stays fp32 — cast logits→`logits_f32`,
        accumulate + vjp there, cast grad_f32→`grad` (all DEVICE kernels, so they
        ride the graph). The cached-bf16 weight recast inside `net.forward` is
        also a captured device kernel (see the `CAPTURE` note)."""
        var ctxo = Optional(self.ctx)
        comptime if Self.DEVICE_BATCH:
            self._device_batch()
        self.opt.zero_grad["gpu"](self.net, ctxo)
        self.net.forward["gpu", Self.BATCH](
            child_refs[Self.NET.ARITY, Self.MADT](self.in_t),
            self.logits,
            ctxo,
        )
        comptime if Self.MADT == DT:
            ref logd = rebind[Tensor](self.logits)
            ref grdd = rebind[Tensor](self.grad)
            self.loss.forward_accumulate["gpu", Self.BATCH](
                logd, self.tgt_t, ctxo
            )
            self.loss.vjp["gpu", Self.BATCH](logd, self.tgt_t, grdd, ctxo)
        else:
            # bf16: fp32 loss boundary (device casts ride the graph).
            Self._cast_logits_to_f32[Self.BATCH](
                self.logits, self.logits_f32, self.ctx
            )
            self.loss.forward_accumulate["gpu", Self.BATCH](
                self.logits_f32, self.tgt_t, ctxo
            )
            self.loss.vjp["gpu", Self.BATCH](
                self.logits_f32, self.tgt_t, self.grad_f32, ctxo
            )
            Self._cast_grad_to_madt[Self.BATCH](
                self.grad, self.grad_f32, self.ctx
            )
        self.net.vjp["gpu", Self.BATCH](
            child_refs[Self.NET.ARITY, Self.MADT](self.in_t),
            self.grad,
            child_refs[Self.NET.ARITY, Self.MADT](self.gi),
            ctxo,
        )
        if self.grad_clip > Scalar[DT](0.0):
            self.opt.clip_grads_device["gpu"](
                self.net, self.grad_clip, ctxo
            )
        self.opt.step["gpu"](self.net, ctxo)

    def _train_step_captured(mut self, it: Int) raises:
        """One captured/replayed train step. The LR push (and, without
        DEVICE_BATCH, the host batch build into the persistent `in_t`/`tgt_t`)
        run EAGERLY; the device compute is captured on the first call and
        replayed thereafter. The LR is pushed onto the device (`push_lr_device`)
        so the captured `opt.step` reads the FRESH cosine LR each replay instead
        of a host-baked capture-time value."""
        self.opt.push_lr_device["gpu"](
            self.base_lr * self._lr_scale(it), Optional(self.ctx)
        )
        comptime if not Self.DEVICE_BATCH:
            self._host_batch()
        # Move the slot into a disjoint local for the capture call (`take` leaves
        # it None) so the closure can borrow `self` without overlapping the
        # slot's mut borrow (mirrors the MBPO dynamics-graph pattern).
        var g = Optional[CUDAGraph](None)
        if self._train_graph:
            g = Optional[CUDAGraph](self._train_graph.take())

        def _captured() capturing raises -> None:
            self._compute_step_device()

        maybe_capture_replay[_captured](g, self.ctx)
        self._train_graph = g^

    # ----- The packaged training run -------------------------------------

    def fit(
        mut self,
        eval_every: Int = 0,
        n_val_windows: Int = 0,
        print_progress: Bool = True,
    ) raises -> Float64:
        """Run `total_iters` of next-token training. When `eval_every > 0` and
        `n_val_windows > 0`, reports per-token val CE on a fixed pre-sampled
        window set every `eval_every` iters. Returns the final val loss (or the
        last train loss if eval is off)."""
        var do_eval = eval_every > 0 and n_val_windows > 0
        if do_eval and n_val_windows % Self.BATCH != 0:
            raise Error("fit: n_val_windows must be a multiple of BATCH")
        # Pre-sample the val windows ONCE so the val curve is comparable across
        # iters.
        var val_mb = make_batch(
            self.val_ids, n_val_windows if do_eval else Self.BATCH, Self.SEQ
        )
        var n_val_batches = n_val_windows // Self.BATCH

        var last: Float64 = 0.0
        # CAPTURE: the per-step train CE is accumulated on-device (no per-step
        # D2H) and drained as a window-mean at each eval boundary. Reset the
        # accumulator before the run; the eager path keeps reading per-step CE.
        comptime if Self.CAPTURE:
            # Force every bf16 leaf to recast its bf16 weight UNCONDITIONALLY so
            # the cast kernel is recorded into the graph and reads the live fp32
            # master on every replay (no silent stale-weight risk). No-op for
            # fp32 leaves. Set once here, after any model surgery (tie/scale).
            self.net.set_attr["capture_recast"](Scalar[DT](1.0))
            self.loss.reset_accum["gpu"]()
        for it in range(self.total_iters):
            var tl: Float64 = 0.0
            comptime if Self.CAPTURE:
                self._train_step_captured(it)
            else:
                tl = self._train_step(it)
                last = tl
            if do_eval and (
                (it + 1) % eval_every == 0 or (it + 1) == self.total_iters
            ):
                comptime if Self.CAPTURE:
                    # Window-mean train CE from the device accumulator, then
                    # reset for the next window.
                    tl = Float64(
                        self.loss.read_accum["gpu"](Optional(self.ctx))
                    )
                    self.loss.reset_accum["gpu"]()
                var v = self._eval_loss_ids(
                    val_mb.inputs, val_mb.targets, n_val_batches
                )
                last = v
                if print_progress:
                    print(
                        "  iter " + String(it + 1) + "/"
                        + String(self.total_iters)
                        + "  train=" + String(Float32(tl))
                        + "  val=" + String(Float32(v))
                        + "  lr_scale=" + String(self._lr_scale(it))
                    )
        return last

    # ----- Sampling / generation -----------------------------------------

    @staticmethod
    def _sample_token[
        ADT: DType
    ](
        ref logits: TensorImpl[ADT], base: Int, temperature: Float64, top_k: Int
    ) -> Int:
        """nanoGPT-style: greedy if `temperature <= 0`, else top-k softmax over
        `logits.data[base : base+VOCAB]`. Generic over `ADT` — all reads go
        through `Float64(...)`, so a bf16-flow logits buffer works directly."""
        if temperature <= 0.0:
            var bv = Float64(logits.data[base])
            var bi = 0
            for v in range(1, Self.VOCAB):
                if Float64(logits.data[base + v]) > bv:
                    bv = Float64(logits.data[base + v])
                    bi = v
            return bi
        var inv_t = 1.0 / temperature
        var scaled = List[Float64](capacity=Self.VOCAB)
        for v in range(Self.VOCAB):
            scaled.append(Float64(logits.data[base + v]) * inv_t)
        var keep = List[Bool](capacity=Self.VOCAB)
        if top_k > 0 and top_k < Self.VOCAB:
            for _ in range(Self.VOCAB):
                keep.append(False)
            var work = List[Float64](capacity=Self.VOCAB)
            for v in range(Self.VOCAB):
                work.append(scaled[v])
            for _ in range(top_k):
                var bv: Float64 = -1e30
                var bi = 0
                for v in range(Self.VOCAB):
                    if work[v] > bv:
                        bv = work[v]
                        bi = v
                keep[bi] = True
                work[bi] = -1e30
        else:
            for _ in range(Self.VOCAB):
                keep.append(True)
        var m: Float64 = -1e30
        for v in range(Self.VOCAB):
            if keep[v] and scaled[v] > m:
                m = scaled[v]
        var se: Float64 = 0.0
        var exps = List[Float64](capacity=Self.VOCAB)
        for v in range(Self.VOCAB):
            if keep[v]:
                var e = exp(scaled[v] - m)
                exps.append(e)
                se += e
            else:
                exps.append(0.0)
        var u = random_float64(0.0, 1.0) * se
        var acc = 0.0
        for v in range(Self.VOCAB):
            acc += exps[v]
            if u < acc:
                return v
        return Self.VOCAB - 1

    def generate(
        mut self,
        prompt: String,
        n_tokens: Int,
        temperature: Float64 = 0.0,
        top_k: Int = 0,
        pad_id: Int = 0,
    ) raises -> String:
        """Autoregressively continue `prompt` for `n_tokens`. Front-anchored
        window: the last min(n_have, SEQ) ids sit at positions 0.., logits read
        at `n_eff - 1` (causal → the tail pad is invisible). Greedy when
        `temperature <= 0`, else top-k softmax sampling."""
        self.net.set_attr["training"](Scalar[DT](0.0))  # eval mode
        var all_ids = self.tok.encode(prompt)
        var prompt_len = len(all_ids)
        if prompt_len == 0:
            raise Error("generate: empty prompt")

        # Dedicated B=1 scratch — the reused `self.in_t`/`self.logits` carry
        # BATCH-sized device buffers from training; a B=1 forward would leave a
        # stale length vs the reallocated buffer (Metal copy overflow).
        var gen_in = TensorImpl[Self.MADT]()
        var gen_out = TensorImpl[Self.MADT]()
        for _gen in range(n_tokens):
            var n_have = len(all_ids)
            var n_eff = n_have if n_have <= Self.SEQ else Self.SEQ
            var first = 0 if n_have <= Self.SEQ else n_have - Self.SEQ
            var win = List[Int](capacity=Self.SEQ)
            for t in range(Self.SEQ):
                win.append(all_ids[first + t] if t < n_eff else pad_id)
            Self._upload_onehot[Self.MADT](self.ctx, gen_in, win, 1)
            self.net.forward[Self.target, 1](
                child_refs[Self.NET.ARITY, Self.MADT](gen_in),
                gen_out,
                Optional(self.ctx),
            )
            gen_out.download(self.ctx)
            var read_pos = n_eff - 1
            all_ids.append(
                Self._sample_token[Self.MADT](
                    gen_out, read_pos * Self.VOCAB, temperature, top_k
                )
            )

        var gen = List[Int](capacity=n_tokens)
        for i in range(prompt_len, len(all_ids)):
            gen.append(all_ids[i])
        self.net.set_attr["training"](Scalar[DT](1.0))
        return self.tok.decode(gen)
