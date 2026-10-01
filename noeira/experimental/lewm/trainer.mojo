"""LeWMTrainer — offline JEPA trainer over LeWMLossGraph (storage surface).

Owns the loss graph + one Adam + owned graph-IO scratch `Tensor`s. `train_step`
runs zero_grad → set_input → forward → mean-reduce → seed 1/B → vjp → (clip) →
Adam.step, returning the batch-mean loss. `collapse_probes` computes the
representation-collapse diagnostics off the `emb` node (var_min over latent
dims, mean |off-diagonal correlation|). `save_params` / `load_params` persist
the graph's parameters in a flat text format (Adam state not persisted —
eval/MPC only needs weights).

Storage notes vs the legacy nn version:
  - `Scratch`/`TargetStorage`/`assert_tag_for` are gone — the loss output and the
    1/B grad seed are owned `Tensor` fields (device buffer on GPU, host List on
    CPU); the device context is a stored `Optional[DeviceContext]`.
  - The Adam graph overloads (`make_graph`/`step_graph`/`zero_grad_graph`) are
    replaced by the uniform storage Adam: `opt.adopt`/`opt.zero_grad`/
    `opt.clip_grads`/`opt.step` over the graph (a `Module`). `weight_decay` is the
    AdamW `wd`; `max_grad_norm` is applied via `clip_grads` before the step.
  - Graph IO is storage `Tensor`s and `node_output(name)` (not raw pointers); the
    public facade keeps raw `TileTensor`/host-pointer args (the data/MPC stack
    feeds them), bridged into the graph internally.

The backward seed is the constant `1/BATCH`, written once at `make`. On GPU
`train_step` device-reduces the per-sample loss into a `(Σmean, count)`
accumulator and returns a `0` sentinel; the driver drains it at flush cadence
via `read_loss_accum` / `reset_loss_accum`.
"""

from std.collections import Dict
from max.gpu import thread_idx, global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.math import sqrt

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from noeira.nn import Tensor, ParamVisitor, Kaiming, Adam, Module
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.checkpoint import (
    BinaryCheckpointWriter,
    BinaryCheckpointReader,
    CheckpointReader,
    _write_file_bytes,
    _read_file_bytes,
    _is_v3_header,
    _split_lines,
)
from .encoder import LeWMEncoder
from .loss_graph import LeWMLossGraph

from noeira.deep_agents.loss.seed_grad_inv_batch import seed_grad_inv_batch


# ── grad-norm clip over the graph's params (CPU loops / GPU reduce+scale) ──
# The graph owns all params but is not a `Module`, so the optimizer's
# Module-constrained `clip_grads` can't take it; this is a two-pass clip via
# `graph.for_each_param` (the storage ParamVisitor), mirroring `grad_clip`.
comptime _GC_TPB: Int = 128


def _clip_sumsq_kernel[
    N: Int
](
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    out_sum: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var g = rebind[Scalar[DT]](grad[k])
        my_sum += g * g
        k += _GC_TPB
    var total = block.sum[block_size=_GC_TPB, broadcast=False](val=my_sum)
    if t == 0:
        out_sum[0] = total[0]


def _clip_scale_kernel[
    N: Int
](
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    scale: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < N:
        grad[i] = rebind[Scalar[DT]](grad[i]) * scale if scale != Scalar[DT](
            0.0
        ) else Scalar[DT](0.0)


def _scale_from_norm(
    norm: Scalar[DT], max_norm: Scalar[DT], eps: Scalar[DT]
) -> Scalar[DT]:
    """`min(1, max_norm / max(norm, eps))`; non-finite norm → 0 (NaN guard)."""
    if norm - norm != Scalar[DT](0.0):
        return Scalar[DT](0.0)
    var denom = norm if norm > eps else eps
    var ratio = max_norm / denom
    return ratio if ratio < Scalar[DT](1.0) else Scalar[DT](1.0)


struct _SumSqV(ParamVisitor):
    var sum_sq: Scalar[DT]
    var psum: Tensor  # reusable [1] device scalar (GPU)

    def __init__(out self):
        self.sum_sq = Scalar[DT](0.0)
        self.psum = Tensor()

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
        comptime if target == "cpu":
            for i in range(N):
                var g = grad.data[i]
                self.sum_sq += g * g
        else:
            var c = ctx.value()
            self.psum.ensure_gpu(c, 1)
            c.enqueue_function[_clip_sumsq_kernel[N]](
                grad.lt["gpu", Layout.row_major(N)](),
                self.psum.lt["gpu", Layout.row_major(1)](),
                grid_dim=1,
                block_dim=_GC_TPB,
            )
            self.psum.download(c)
            self.sum_sq += self.psum.data[0]


struct _ScaleV(ParamVisitor):
    var scale: Scalar[DT]

    def __init__(out self, scale: Scalar[DT]):
        self.scale = scale

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
        comptime if target == "cpu":
            if self.scale == Scalar[DT](0.0):
                for i in range(N):
                    grad.data[i] = Scalar[DT](0.0)
            else:
                for i in range(N):
                    grad.data[i] = grad.data[i] * self.scale
        else:
            var c = ctx.value()
            c.enqueue_function[_clip_scale_kernel[N]](
                grad.lt["gpu", Layout.row_major(N)](),
                self.scale,
                grid_dim=(N + TPB - 1) // TPB,
                block_dim=TPB,
            )


struct _MaskGradV(ParamVisitor):
    """Zero the grad of every param whose dotted name does NOT start with one
    of the kept prefixes — the AdaJEPA test-time-adaptation subset mask
    (docs/ADAJEPA_LEWM_TTA_PLAN.md §4). Runs between vjp and the optimizer
    step. Masked params stay bit-identical only with a FRESH optimizer (zero
    moments) and wd=0 — decoupled weight decay moves zero-grad params."""

    var keep: List[String]

    def __init__(out self, var keep: List[String]):
        self.keep = keep^

    def _kept(self, name: String) -> Bool:
        for p in self.keep:
            if name.startswith(p):
                return True
        return False

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
        if self._kept(name):
            return
        comptime if target == "cpu":
            for i in range(N):
                grad.data[i] = Scalar[DT](0.0)
        else:
            var c = ctx.value()
            c.enqueue_function[_clip_scale_kernel[N]](
                grad.lt["gpu", Layout.row_major(N)](),
                Scalar[DT](0.0),
                grid_dim=(N + TPB - 1) // TPB,
                block_dim=TPB,
            )


struct _ZeroMomentsV(ParamVisitor):
    """Zero every param's Adam m/v (host + device re-upload) — backs
    `reset_opt_moments`. Unallocated moments (never stepped) are left
    untouched: they are already fresh."""

    def __init__(out self):
        pass

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
        if m.n >= N:
            for i in range(N):
                m.data[i] = Scalar[DT](0.0)
            comptime if target == "gpu":
                m.upload(ctx.value())
        if v.n >= N:
            for i in range(N):
                v.data[i] = Scalar[DT](0.0)
            comptime if target == "gpu":
                v.upload(ctx.value())


# ── GPU loss reduction: acc[0] += mean(src); acc[1] += 1 ───────────────
# Single-block `block.sum` over the [BATCH] per-sample loss. No per-step
# D2H — the driver drains the accumulator at flush cadence.
def _reduce_mean_acc_kernel[
    BATCH: Int
](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        my_sum += src[unsafe_offset=k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = acc[unsafe_offset=0] + total[0] / Scalar[DT](BATCH)
        acc[unsafe_offset=1] = acc[unsafe_offset=1] + Scalar[DT](1.0)


# ── checkpoint / export visitors (storage ParamVisitor signature) ──────
struct _SaveVisitor(ParamVisitor):
    """Collect each param's values in for_each_param walk order (GPU: D2H)."""

    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

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
        comptime if target == "gpu":
            param.download(ctx.value())
        for i in range(N):
            self.vals.append(param.data[i])


struct _LoadVisitor(ParamVisitor):
    """Restore each param's values in for_each_param walk order (GPU: H2D)."""

    var vals: List[Scalar[DT]]
    var idx: Int

    def __init__(out self, var vals: List[Scalar[DT]]):
        self.vals = vals^
        self.idx = 0

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
        param.ensure(N)
        for i in range(N):
            param.data[i] = self.vals[self.idx]
            self.idx += 1
        param.n = N
        # A restored weight must advance `version`, or `Linear`'s version-gated
        # `w_pad` / `w_bf` keep serving the pre-restore weight (same rule as
        # `BinaryCheckpointReader`). `upload_resident`, not `upload`: `upload`
        # recreates the buffer and would detach a param from an adopted arena.
        param.version += 1
        comptime if target == "gpu":
            param.upload_resident(ctx.value())


struct _NamedExportVisitor(ParamVisitor):
    """Record each param/state's name → values (GPU: D2H). Owns the dict; feeds
    LeWMPredictor.sync_from_named for the MPC predictor-from-latents path."""

    var d: Dict[String, List[Scalar[DT]]]

    def __init__(out self):
        self.d = Dict[String, List[Scalar[DT]]]()

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
        comptime if target == "gpu":
            param.download(ctx.value())
        var vals = List[Scalar[DT]](length=N, fill=Scalar[DT](0.0))
        for i in range(N):
            vals[i] = param.data[i]
        self.d[name] = vals^

    def take(deinit self) -> Dict[String, List[Scalar[DT]]]:
        """Consume the visitor, yielding its accumulated dict (avoids a
        partial field-move-out of a still-live value)."""
        return self.d^


struct LeWMTrainer[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    ENC_PROJ_H: Int,
    ENC_FF_MULT: Int,
    T: Int,
    ACT: Int,
    SMOOTHED: Int,
    AE_MLP: Int,
    H: Int,
    N_PREDS: Int,
    PRED_HEADS: Int,
    PRED_FF: Int,
    DEPTH: Int,
    PRED_PROJ_H: Int,
    SIG_PROJ: Int,
    SIG_KNOTS: Int,
    BATCH: Int,
    train_target: StaticString = "cpu",
    PRED_DIM_HEAD: Int = 0,
    ENC: Module = LeWMEncoder[
        IN_CH,
        IMG,
        PATCH,
        (IMG // PATCH) * (IMG // PATCH),
        HIDDEN,
        ENC_HEADS,
        ENC_LAYERS,
        EMB,
        ENC_PROJ_H,
        ENC_FF_MULT,
    ],
](Movable & Deinitable):
    comptime LG = LeWMLossGraph[
        Self.IN_CH,
        Self.IMG,
        Self.PATCH,
        Self.HIDDEN,
        Self.ENC_HEADS,
        Self.ENC_LAYERS,
        Self.EMB,
        Self.ENC_PROJ_H,
        Self.ENC_FF_MULT,
        Self.T,
        Self.ACT,
        Self.SMOOTHED,
        Self.AE_MLP,
        Self.H,
        Self.N_PREDS,
        Self.PRED_HEADS,
        Self.PRED_FF,
        Self.DEPTH,
        Self.PRED_PROJ_H,
        Self.SIG_PROJ,
        Self.SIG_KNOTS,
        Self.PRED_DIM_HEAD,
        Self.ENC,
    ]
    comptime PIX = Self.T * Self.IN_CH * Self.IMG * Self.IMG
    comptime ACTIN = Self.T * Self.ACT
    comptime TE = Self.T * Self.EMB
    comptime HE = Self.H * Self.EMB

    var graph: Self.LG
    var opt: Adam
    var max_grad_norm: Scalar[DT]
    var ctx: Optional[DeviceContext]
    # Graph IO scratch: device buffer on GPU, host list on CPU.
    var loss_out: Tensor  # per-sample loss [BATCH]
    var grad_seed: Tensor  # constant 1/BATCH backward seed [BATCH]
    # GPU-only `(Σmean, count)` loss accumulator (drained at flush).
    var _loss_acc_dev: Optional[DeviceBuffer[DT]]
    # Host staging (RAII Lists): probe emb (BATCH·TE) + eval-loss D2H (BATCH).
    var emb_buf: List[Scalar[DT]]
    var loss_host: List[Scalar[DT]]
    var last_load_had_state: Bool
    """True if the last `load_params` restored State sections (BN running
    stats) — v3/v2 named checkpoints; False for legacy flat ckpts, which
    need a BN warmup before planning."""

    def __init__(out self):
        self.graph = Self.LG()
        self.opt = Adam()
        self.max_grad_norm = Scalar[DT](0.0)
        self.ctx = None
        self.loss_out = Tensor()
        self.grad_seed = Tensor()
        self._loss_acc_dev = None
        self.emb_buf = List[Scalar[DT]]()
        self.loss_host = List[Scalar[DT]]()
        self.last_load_had_state = False

    @staticmethod
    def make(
        lam: Scalar[DT] = 0.09,
        lr: Scalar[DT] = 1e-3,
        max_grad_norm: Scalar[DT] = 0.0,
        weight_decay: Scalar[DT] = 0.0,
        sigreg_resample: Bool = False,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var t = Self()
        t.graph = Self.LG.make[Self.train_target, Kaiming](ctx=ctx)
        t.graph.set_node_attr["sig_s", "multiplier"](lam)
        # Per-step SIGReg projection resampling (reference draws fresh
        # projections EVERY forward). Default off = bit-identical to before.
        if sigreg_resample:
            t.graph.set_node_attr["sig", "resample"](Scalar[DT](1.0))
        # Decoupled (AdamW-style) weight decay — wd>0 makes Adam → AdamW;
        # decay-exempt params (BatchNorm γ/β etc.) are skipped via apply_decay.
        t.opt = Adam(lr=lr, wd=weight_decay)
        t.max_grad_norm = max_grad_norm
        t.ctx = ctx
        comptime if Self.train_target == "gpu":
            var c = ctx.value()
            # NOTE: the graph owns all params but is not a `Module`, so Adam's
            # arena (`adopt`) — which requires a Module — can't be engaged here;
            # the step walks `graph.for_each_param` (per-param kernels). A
            # graph-aware grouped arena is a GPU-perf follow-up.
            t.loss_out = Tensor.alloc_gpu(c, Self.BATCH)
            t.grad_seed = Tensor.alloc_gpu(c, Self.BATCH)
            var acc = c.enqueue_create_buffer[DT](2)
            acc.enqueue_fill(0.0)
            t._loss_acc_dev = acc^
        else:
            t.loss_out = Tensor.alloc(Self.BATCH)
            t.grad_seed = Tensor.alloc(Self.BATCH)
        # The backward seed for a mean-over-batch loss is the constant 1/BATCH
        # in every slot — write it once (nothing in the step mutates it).
        seed_grad_inv_batch[Self.train_target, Self.BATCH](
            t.grad_seed.lt[
                Self.train_target, Layout.row_major(Self.BATCH, 1)
            ](),
            ctx=ctx,
        )
        t.emb_buf = List[Scalar[DT]](
            length=Self.BATCH * Self.TE, fill=Scalar[DT](0.0)
        )
        t.loss_host = List[Scalar[DT]](length=Self.BATCH, fill=Scalar[DT](0.0))
        return t^

    def _seed_input[
        slot_name: StaticString, N: Int
    ](
        mut self,
        src: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        """Bridge a raw input tile into the named graph input slot (storage
        `set_input` copies it into the pool)."""
        var tt = Tensor()
        comptime if Self.train_target == "cpu":
            tt.data = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
            for i in range(N):
                tt.data[i] = rebind[Scalar[DT]](src.ptr[unsafe_offset=i])
            tt.n = N
        else:
            var c = self.ctx.value()
            var sp = rebind[Pointer[Scalar[DT], MutAnyOrigin]](src.ptr)
            tt.dev = DeviceBuffer[DT](c, sp, N, owning=False)
            tt.n = N
        self.graph.set_input[slot_name, Self.BATCH](tt, self.ctx)

    def train_step[
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        pix: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
        act: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises -> Scalar[DT]:
        self.graph.zero_grad[Self.train_target](self.ctx)
        self._seed_input["pixels", Self.BATCH * Self.PIX](pix)
        self._seed_input["actions", Self.BATCH * Self.ACTIN](act)
        self.graph.forward[Self.BATCH, Self.train_target, POLICY](
            self.loss_out, self.ctx
        )

        var m: Scalar[DT] = 0.0
        comptime if Self.train_target == "cpu":
            for b in range(Self.BATCH):
                m += self.loss_out.data[b]
            m /= Scalar[DT](Self.BATCH)
        else:
            # Device reduce-accumulate; the driver drains `_loss_acc_dev` at
            # flush cadence. `m` stays a 0 sentinel (no per-step D2H).
            var c = self.ctx.value()
            comptime red = _reduce_mean_acc_kernel[Self.BATCH]
            c.enqueue_function[red](
                self.loss_out.dev.value().unsafe_ptr(),
                self._loss_acc_dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )

        self.graph.vjp[Self.BATCH, Self.train_target, POLICY](
            self.grad_seed, self.ctx
        )
        # Global grad-norm clip (0.0 = disabled) — two-pass over graph params.
        if self.max_grad_norm > Scalar[DT](0.0):
            var ss = _SumSqV()
            self.graph.for_each_param[Self.train_target, _SumSqV](ss, self.ctx)
            var scale = _scale_from_norm(
                sqrt(ss.sum_sq), self.max_grad_norm, Scalar[DT](1e-6)
            )
            if scale < Scalar[DT](1.0):
                var sc = _ScaleV(scale)
                self.graph.for_each_param[Self.train_target, _ScaleV](
                    sc, self.ctx
                )
        self.opt.begin_step()
        self.graph.for_each_param[Self.train_target](self.opt, self.ctx)
        return m

    def train_step_masked[
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        pix: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,

            origin=MutAnyOrigin,
            ...,
        ],
        act: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,

            origin=MutAnyOrigin,
            ...,
        ],
        keep_prefixes: List[String],
    ) raises -> Scalar[DT]:
        """`train_step` with a subset-parameter mask: after the backward pass,
        zero the grad of every param whose dotted name does not start with one
        of `keep_prefixes`, so only the kept subset is updated (AdaJEPA
        test-time adaptation, docs/ADAJEPA_LEWM_TTA_PLAN.md). The mask runs
        BEFORE the grad-norm clip so the norm covers only the applied update.
        Masked params are provably frozen only with a fresh Adam and wd=0."""
        self.graph.zero_grad[Self.train_target](self.ctx)
        self._seed_input["pixels", Self.BATCH * Self.PIX](pix)
        self._seed_input["actions", Self.BATCH * Self.ACTIN](act)
        self.graph.forward[Self.BATCH, Self.train_target, POLICY](
            self.loss_out, self.ctx
        )

        var m: Scalar[DT] = 0.0
        comptime if Self.train_target == "cpu":
            for b in range(Self.BATCH):
                m += self.loss_out.data[b]
            m /= Scalar[DT](Self.BATCH)
        else:
            var c = self.ctx.value()
            comptime red = _reduce_mean_acc_kernel[Self.BATCH]
            c.enqueue_function[red](
                self.loss_out.dev.value().unsafe_ptr(),
                self._loss_acc_dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )

        self.graph.vjp[Self.BATCH, Self.train_target, POLICY](
            self.grad_seed, self.ctx
        )
        var mask = _MaskGradV(keep_prefixes.copy())
        self.graph.for_each_param[Self.train_target, _MaskGradV](
            mask, self.ctx
        )
        if self.max_grad_norm > Scalar[DT](0.0):
            var ss = _SumSqV()
            self.graph.for_each_param[Self.train_target, _SumSqV](ss, self.ctx)
            var scale = _scale_from_norm(
                sqrt(ss.sum_sq), self.max_grad_norm, Scalar[DT](1e-6)
            )
            if scale < Scalar[DT](1.0):
                var sc = _ScaleV(scale)
                self.graph.for_each_param[Self.train_target, _ScaleV](
                    sc, self.ctx
                )
        self.opt.begin_step()
        self.graph.for_each_param[Self.train_target](self.opt, self.ctx)
        return m

    def eval_loss[
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        pix: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
        act: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises -> Scalar[DT]:
        """Forward-only batch-mean loss (no grad / no optimizer step)."""
        self._seed_input["pixels", Self.BATCH * Self.PIX](pix)
        self._seed_input["actions", Self.BATCH * Self.ACTIN](act)
        self.graph.forward[Self.BATCH, Self.train_target, POLICY](
            self.loss_out, self.ctx
        )

        var m: Scalar[DT] = 0.0
        comptime if Self.train_target == "cpu":
            for b in range(Self.BATCH):
                m += self.loss_out.data[b]
        else:
            # eval is off the capture hot loop — D2H the [BATCH] vector once.
            self.loss_out.download(self.ctx.value())
            for b in range(Self.BATCH):
                m += self.loss_out.data[b]
        return m / Scalar[DT](Self.BATCH)

    def forward_into[
        POLICY: AMPPolicy = NoAMP,
        ph_o: MutOrigin = MutAnyOrigin,
        th_o: MutOrigin = MutAnyOrigin,
    ](
        mut self,
        pix: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
        act: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
        pred_host: Pointer[Scalar[DT], ph_o],
        tgt_host: Pointer[Scalar[DT], th_o],
    ) raises:
        """Forward-only readout for eval/planning: run the graph over
        (pix, act) and copy the predicted latents (`pred` node) and the
        encoded target latents (`tgt` node) — both (BATCH, H·EMB) — to the
        caller's host buffers. No grad / no step."""
        self._seed_input["pixels", Self.BATCH * Self.PIX](pix)
        self._seed_input["actions", Self.BATCH * Self.ACTIN](act)
        self.graph.forward[Self.BATCH, Self.train_target, POLICY](
            self.loss_out, self.ctx
        )
        comptime N = Self.BATCH * Self.HE
        ref pred_src = self.graph.node_output["pred"]()
        ref tgt_src = self.graph.node_output["tgt"]()
        comptime if Self.train_target == "cpu":
            for i in range(N):
                pred_host[unsafe_offset=i] = pred_src.data[i]
                tgt_host[unsafe_offset=i] = tgt_src.data[i]
        else:
            var c = self.ctx.value()
            pred_src.download(c)
            tgt_src.download(c)
            for i in range(N):
                pred_host[unsafe_offset=i] = pred_src.data[i]
                tgt_host[unsafe_offset=i] = tgt_src.data[i]

    def read_node_into[
        name: StaticString,
        h_o: MutOrigin = MutAnyOrigin,
    ](mut self, host: Pointer[Scalar[DT], h_o], n: Int,) raises:
        """Copy the named graph node's output (n elements) to a host buffer.
        Valid after a forward has populated the node buffers."""
        ref src = self.graph.node_output[name]()
        comptime if Self.train_target == "cpu":
            for i in range(n):
                host[unsafe_offset=i] = src.data[i]
        else:
            src.download(self.ctx.value())
            for i in range(n):
                host[unsafe_offset=i] = src.data[i]

    def export_named_params(mut self) raises -> Dict[String, List[Scalar[DT]]]:
        """Snapshot all graph params AND state (BatchNorm running stats) as
        a name→values dict (CPU/GPU). Feeds `LeWMPredictor.sync_from_named`."""
        var v = _NamedExportVisitor()
        self.graph.for_each_param[Self.train_target, _NamedExportVisitor](
            v, self.ctx
        )
        self.graph.for_each_state[Self.train_target, _NamedExportVisitor](
            v, self.ctx
        )
        return v^.take()

    def set_bn_training(mut self, training: Bool) raises:
        """Flip the graph's BatchNorm layers between training (batch stats +
        EMA update) and eval (running stats) mode. BN lives in the encoder
        projector (node "emb") and PredProj (node "pred")."""
        var v = Scalar[DT](1.0) if training else Scalar[DT](0.0)
        self.graph.set_node_attr["emb", "training"](v)
        self.graph.set_node_attr["pred", "training"](v)

    def reset_loss_accum(mut self) raises:
        """Zero the device `(Σmean, count)` loss accumulator (GPU, flush)."""
        comptime if Self.train_target == "gpu":
            self._loss_acc_dev.value().enqueue_fill(0.0)

    def read_loss_accum(mut self) raises -> Scalar[DT]:
        """D2H the device loss accumulator once and return the window mean
        (Σmean / count). 0 if no steps. GPU only."""
        comptime if Self.train_target == "gpu":
            var c = self.ctx.value()
            var h = c.enqueue_create_host_buffer[DT](2)
            c.enqueue_copy(h, self._loss_acc_dev.value())
            c.synchronize()
            var s = h.unsafe_ptr()[unsafe_offset=0]
            var n = h.unsafe_ptr()[unsafe_offset=1]
            if n == Scalar[DT](0.0):
                return Scalar[DT](0.0)
            return s / n
        else:
            return Scalar[DT](0.0)

    def collapse_probes(mut self) raises -> Tuple[Scalar[DT], Scalar[DT]]:
        """(var_min, gram_off) over the last forward's `emb`, viewed as
        BATCH·T samples of EMB latent dims. Healthy: var_min > 0.1,
        gram_off < 0.5 (legacy thresholds)."""
        comptime ns = Self.BATCH * Self.T
        comptime D = Self.EMB
        ref emb_src = self.graph.node_output["emb"]()
        comptime if Self.train_target == "cpu":
            for i in range(ns * D):
                self.emb_buf[i] = emb_src.data[i]
        else:
            emb_src.download(self.ctx.value())
            for i in range(ns * D):
                self.emb_buf[i] = emb_src.data[i]

        # per-dim mean + variance
        var mean = List[Scalar[DT]](length=D, fill=Scalar[DT](0.0))
        var std = List[Scalar[DT]](length=D, fill=Scalar[DT](0.0))
        var var_min = Scalar[DT](1e30)
        for d in range(D):
            var s: Scalar[DT] = 0.0
            for r in range(ns):
                s += self.emb_buf[r * D + d]
            var mu = s / Scalar[DT](ns)
            mean[d] = mu
            var vv: Scalar[DT] = 0.0
            for r in range(ns):
                var df = self.emb_buf[r * D + d] - mu
                vv += df * df
            vv /= Scalar[DT](ns)
            std[d] = sqrt(vv + Scalar[DT](1e-8))
            if vv < var_min:
                var_min = vv

        # mean |off-diagonal correlation|
        var acc: Scalar[DT] = 0.0
        var cnt: Int = 0
        for i in range(D):
            for j in range(D):
                if i == j:
                    continue
                var cc: Scalar[DT] = 0.0
                for r in range(ns):
                    cc += (self.emb_buf[r * D + i] - mean[i]) * (
                        self.emb_buf[r * D + j] - mean[j]
                    )
                cc /= Scalar[DT](ns)
                acc += (cc / (std[i] * std[j])).__abs__()
                cnt += 1
        var gram_off = acc / Scalar[DT](cnt)
        return (var_min, gram_off)

    def snapshot_all(mut self) raises -> List[Scalar[DT]]:
        """Params + state (BN running stats) in walk order — in-memory
        snapshot for the AdaJEPA TTA per-episode model reset (restore with
        `restore_all`; docs/ADAJEPA_LEWM_TTA_PLAN.md §5). Adam moments are
        NOT captured: kept-subset moments survive the restore, which is why
        a TTA episode must use one constant mask set (plan §4)."""
        var v = _SaveVisitor()
        self.graph.for_each_param[Self.train_target, _SaveVisitor](v, self.ctx)
        self.graph.for_each_state[Self.train_target, _SaveVisitor](v, self.ctx)
        return v.vals.copy()

    def restore_all(mut self, vals: List[Scalar[DT]]) raises:
        """Restore a `snapshot_all` capture (params + state, same walk
        order; GPU: re-uploads)."""
        var v = _LoadVisitor(vals.copy())
        self.graph.for_each_param[Self.train_target, _LoadVisitor](v, self.ctx)
        self.graph.for_each_state[Self.train_target, _LoadVisitor](v, self.ctx)

    def save_params(mut self, path: String, save_moments: Bool = True) raises:
        """Write a v3 binary named checkpoint (nn.core.checkpoint): Param
        sections (+ Adam moments when populated — exact training resume)
        then State sections (BN running stats — a v3 load restores them, so
        planning needs NO BN warmup). Replaces the legacy positional flat
        text this trainer used to write; `load_params` still reads it."""
        var w = BinaryCheckpointWriter(save_moments)
        w.mode = 0
        self.graph.for_each_param[Self.train_target, BinaryCheckpointWriter](
            w, self.ctx
        )
        w.mode = 1
        self.graph.for_each_state[Self.train_target, BinaryCheckpointWriter](
            w, self.ctx
        )
        _write_file_bytes(path, w.content)

    def load_params(mut self, path: String) raises:
        """Load a checkpoint, dispatching on the header: v3 binary / v2
        named text (name+size validated against the graph walk; BN state
        restored; sets `last_load_had_state=True`) or the legacy positional
        flat text (params only; `last_load_had_state=False` → warm BN
        running stats before planning). ⚠ v3 checkpoints may carry Adam
        moments — call `reset_opt_moments()` after loading if you need the
        fresh-optimizer invariant (test-time adaptation,
        docs/ADAJEPA_LEWM_TTA_PLAN.md §4)."""
        self.last_load_had_state = False
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var r = BinaryCheckpointReader(bytes^)
            r.mode = 0
            self.graph.for_each_param[
                Self.train_target, BinaryCheckpointReader
            ](r, self.ctx)
            r.mode = 1
            self.graph.for_each_state[
                Self.train_target, BinaryCheckpointReader
            ](r, self.ctx)
            self.last_load_had_state = True
            return
        var content: String
        with open(path, "r") as f:
            content = String(f.read())
        var lines = _split_lines(content)
        if len(lines) > 0 and lines[0].startswith("storage-ckpt"):
            var body = List[String]()
            for li in range(1, len(lines)):
                body.append(lines[li])
            var r2 = CheckpointReader(body^)
            r2.mode = 0
            self.graph.for_each_param[Self.train_target, CheckpointReader](
                r2, self.ctx
            )
            r2.mode = 1
            self.graph.for_each_state[Self.train_target, CheckpointReader](
                r2, self.ctx
            )
            self.last_load_had_state = True
            return
        # Legacy positional flat text: "count\nval\n..." (params only).
        var n = Int(lines[0])
        var vals = List[Scalar[DT]]()
        for i in range(n):
            vals.append(Scalar[DT](Float64(lines[i + 1])))
        var v = _LoadVisitor(vals^)
        self.graph.for_each_param[Self.train_target, _LoadVisitor](v, self.ctx)

    def reset_opt_moments(mut self) raises:
        """Zero Adam m/v for every param — restores the fresh-optimizer
        precondition after `load_params` on a moments-carrying v3 checkpoint
        (the TTA mask invariant needs zero moments; the trainer's host step
        counter is already fresh)."""
        var z = _ZeroMomentsV()
        self.graph.for_each_param[Self.train_target, _ZeroMomentsV](
            z, self.ctx
        )
