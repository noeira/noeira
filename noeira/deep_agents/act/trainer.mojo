# +--------------------------------------------------------------------------+ #
# | ACTTrainer — offline behaviour cloning over ACTLossGraph
# +--------------------------------------------------------------------------+ #
"""Owns the loss graph + one AdamW + the scratch a step needs.

Transposed from `experimental/lewm/trainer.mojo`, which is the established
shape for "one ComputeGraph, one optimizer": a step is

    zero_grad -> set_input x4 -> forward -> seed grad 1/B -> vjp -> clip -> step

On GPU the optimizer runs in GROUPED ARENA mode: `Adam.adopt` packs every
parameter into contiguous val/grd/m/v device buffers, so `zero_grad`, the
grad-norm clip and the update are a handful of kernels over the whole model
instead of one kernel per parameter. On CPU `adopt` is a no-op and `Adam` is
used as a `ParamVisitor` over `graph.for_each_param`, as before.

CPU and GPU: `target` is a comptime parameter. On GPU the sampler still produces
host `List`s, so a step is host-fill -> upload -> graph -> download the three
per-sample loss vectors. Weights, activations and gradients stay on device.

## The arena, and when it is engaged

`Adam.adopt` used to require a `Module`, which a `ComputeGraph` is not, so this
trainer walked `for_each_param` launching one kernel per parameter. The bound is
now `ParamWalkable` (`nn/core/param.mojo`) — the param walk and nothing else —
which a graph satisfies. Measured on `act_so101_profile_gpu.mojo`, the per-param
path was 10.5% of every kernel launch in the run for 1.0% of the kernel time,
and the unconditional host-side grad-norm walk below it was a device
synchronization PER PARAMETER (see `_SumSq`).

⚠ Adoption is LAZY — first `train_step`, not `make`. `ParamArena.adopt` rebinds
each Param's device buffer to a slice of the arena, and `Tensor.upload`
RECREATES a device buffer: anything that uploads into a parameter after
adoption (a checkpoint load, a test injecting reference weights) would silently
detach it, after which the grouped step updates arena memory the model no
longer reads. Loading before the first step is therefore always safe; the
checkpoint and refload paths additionally use `upload_resident`, which reuses
the existing buffer, so a load AFTER adoption is safe too.

## Logging: per-step readback vs device accumulators

There are two families of step method and they must not be mixed in one loop:

  * `train_step` / `train_step_device` / `eval_step*` return an
    `ACTStepResult`, which costs four D2Hs behind two device synchronizations
    every step;
  * `train_step_device_accum` / `eval_step_device_accum` /
    `eval_step_resident_accum` return NOTHING and fold the same four scalars
    into device accumulators, drained by `train_metrics` / `val_metrics` at
    logging cadence.

The second family exists because those four downloads were the last host
traffic in an ACT step — see `docs/ACT_GPU_DATA_PATH.md`. Removing them makes
the per-step sequence pure kernels, which is what a CUDA graph can capture.

⚠ A loop that called both families and then read `train_metrics` would report a
window covering only half its steps, and the number would look plausible.

## Train vs eval

`train_mode(True/False)` sets three things together, because getting one of them
wrong is silent:

  * `set_attr["training"]` — dropout, and BatchNorm batch-vs-running statistics;
  * `zs.multiplier` — 1.0 samples the CVAE latent, 0.0 makes the latent token
    `latent_out_proj(0)`, which is exactly the reference's test-time `z = 0`
    (`detr_vae.py:110`);
  * `z.deterministic` — off in training (the reparameterization is the point),
    on in eval so a validation number is reproducible rather than a fresh draw.
    `set_deterministic_latent(True)` pins it in training too, and STICKS —
    writing the node attribute directly does not, because every mode switch
    rewrites it.

⚠ The reference's model selection is **validation L1**, not the total loss
(`imitate_episodes.py` tracks `min_val_loss` over the summed dict but the L1 term
is what the policy is judged on). `eval_step` returns both.

## Deviations, all deliberate — see `config.mojo` for the full list

* ONE learning rate. The reference gives backbone params `lr_backbone = 1e-5` in
  a second AdamW group; `Adam` here has no name filter, and a 10x-lower rate on
  a FROM-SCRATCH backbone would freeze the vision tower rather than gently
  fine-tune it.
* Gradient clipping IS applied. The reference parses `--clip_max_norm 0.1` and
  never uses it; an unclipped from-scratch ResNet on four episodes is a
  divergence waiting to happen. Set `max_grad_norm = 0.0` to match the
  reference exactly.
"""

from std.math import sqrt

from max.gpu.host import DeviceContext
from layout import Layout

from noeira.nn.constants import DT
from noeira.nn import Adam, Kaiming, Tensor
from noeira.nn.core.module import Module
from noeira.nn.core.param import ParamVisitor, ParamVisitorRT, walk_params, ParamVisitorRef
from noeira.nn.models.resnet18 import (
    RESNET18_OUT_CH,
    ResNet18Backbone,
    ResNet18OutH,
    ResNet18OutW,
)
from noeira.nn.core.checkpoint import (
    BinaryCheckpointReader,
    BinaryCheckpointWriter,
    CheckpointScalars,
    _is_v3_header,
    _read_file_bytes,
    _write_file_bytes,
)
from noeira.deep_agents.loss.seed_grad_inv_batch import seed_grad_inv_batch
from ..training.device_mean_accum import DeviceMeanAccum
from noeira.cuda import CUDAGraph, maybe_capture_replay

from .config import (
    ACT_CLIP_MAX_NORM,
    ACT_DROPOUT,
    ACT_KL_WEIGHT,
    ACT_SHAPE_WEIGHT,
    ACT_LR,
    ACT_WEIGHT_DECAY,
)
from .data_gpu import ACTDeviceDataset
from .loss_graph import ACTLossGraph
from noeira.io.hf import HF_MODEL, hf_download_file
from noeira.io.safetensors import SafeTensors
from noeira.nn.core.torch_names import LoadTorchNamed, SaveTorchNamed
from noeira.nn.models.resnet18_torch import (
    RESNET18_TV_FILE,
    RESNET18_TV_REPO,
    resnet18_torch_map,
)

from .refload import LoadPrefixedParams, RefDump


# ── grad-norm clip over a graph ──────────────────────────────────────────
# The graph owns every parameter but is not a `Module`, so `Adam.clip_grads`
# (Module-constrained) cannot take it. Two visitor passes: sum of squares, then
# scale. Mirrors `lewm/trainer.mojo`'s `_SumSqV`, CPU-only.


struct _SumSq(ParamVisitor, ParamVisitorRT):
    """Sum of squared gradients over every parameter.

    ⚠ CPU ONLY now. On GPU this DOWNLOADS each gradient slab to sum it on the
    host — a device synchronization PER PARAMETER, and (contrary to what this
    note used to claim) it was NOT gated on `max_grad_norm`: only the SCALING
    was conditional, the sum ran on every step because the norm is reported in
    `ACTStepResult`. For an ACT model that is ~150 syncs per step, which is
    most of the 148 synchronizations per pass in `docs/GPU_STEP_PERF.md`. The
    GPU path now uses the optimizer's arena reduction (`clip_grads_device`):
    three kernels over the contiguous grad arena and one D2H for the reported
    norm.
    """

    var sum_sq: Float64

    def __init__(out self):
        self.sum_sq = 0.0

    def __init__(out self, *, deinit move: Self):
        self.sum_sq = move.sum_sq

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
        comptime if target != "cpu":
            grad.download(ctx.value())
        for i in range(n):
            var g = Float64(grad.data[i])
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
struct _ScaleGrads(ParamVisitor, ParamVisitorRT):
    var scale: Scalar[DT]

    def __init__(out self):
        self.scale = Scalar[DT](1.0)

    def __init__(out self, scale: Scalar[DT]):
        self.scale = scale

    def __init__(out self, *, deinit move: Self):
        self.scale = move.scale

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
        comptime if target != "cpu":
            # ⚠ Unreachable: the GPU clip is the optimizer's arena path. Kept
            # only so the visitor stays target-generic — and `upload` here
            # would DETACH the param from the arena (see the header).
            for i in range(n):
                grad.data[i] = grad.data[i] * self.scale
            grad.upload_resident(ctx.value())
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
@fieldwise_init
struct ACTStepResult(ImplicitlyCopyable):
    """What one step produced. `l1` is the reference's model-selection metric;
    `loss` is what the optimizer actually descended."""

    var loss: Float64
    var l1: Float64
    var kl: Float64
    var grad_norm: Float64


@fieldwise_init
struct ACTWindowMetrics(ImplicitlyCopyable):
    """The same four scalars, MEANED OVER A WINDOW of steps.

    Deliberately not an `ACTStepResult`. The two carry the same field names and
    mean different things, and the way that goes wrong is silent: a window mean
    logged as a step value simply looks like a smoother curve. `n` is how many
    steps the window held — 0 means nothing was accumulated and the means are
    all 0, which is not the same as a loss of 0."""

    var loss: Float64
    var l1: Float64
    var kl: Float64
    var grad_norm: Float64
    var n: Int


struct ACTMetricAccum(Movable & Deinitable):
    """(loss, l1, kl, grad_norm) over a logging window.

    GPU: four `DeviceMeanAccum`s, each a `[2]` device buffer holding
    `[sum_of_batch_means, count]`, folded by a one-block reduction kernel with
    NO D2H — so a step that accumulates into these is capture-safe and the host
    never waits on it. CPU: plain host scalars, because the numbers are already
    on the host there and a device accumulator would buy nothing.

    Drained by `ACTTrainer.train_metrics` / `val_metrics` at logging cadence:
    ONE D2H per accumulator per flush instead of four per step.
    """

    var loss: DeviceMeanAccum
    var l1: DeviceMeanAccum
    var kl: DeviceMeanAccum
    var gn: DeviceMeanAccum
    # CPU mirror. Sums of per-batch means; divided by `n` on read.
    var _h_loss: Float64
    var _h_l1: Float64
    var _h_kl: Float64
    var _h_gn: Float64
    var n: Int
    """Steps folded into this window, counted on the HOST.

    ⚠ The CPU path's only count. On GPU it is a fallback that `read` does NOT
    use: under CUDA-graph replay the fold kernel runs inside the captured
    region and this line does not, so the host counter freezes while the
    device one keeps advancing. `read` takes both mean and count off the
    device, which is right whether the step ran directly or was replayed."""

    def __init__(out self):
        self.loss = DeviceMeanAccum()
        self.l1 = DeviceMeanAccum()
        self.kl = DeviceMeanAccum()
        self.gn = DeviceMeanAccum()
        self._h_loss = 0.0
        self._h_l1 = 0.0
        self._h_kl = 0.0
        self._h_gn = 0.0
        self.n = 0

    def __init__(out self, *, deinit move: Self):
        self.loss = move.loss^
        self.l1 = move.l1^
        self.kl = move.kl^
        self.gn = move.gn^
        self._h_loss = move._h_loss
        self._h_l1 = move._h_l1
        self._h_kl = move._h_kl
        self._h_gn = move._h_gn
        self.n = move.n

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var a = Self()
        comptime if target != "cpu":
            a.loss = DeviceMeanAccum.make["gpu"](ctx=ctx)
            a.l1 = DeviceMeanAccum.make["gpu"](ctx=ctx)
            a.kl = DeviceMeanAccum.make["gpu"](ctx=ctx)
            a.gn = DeviceMeanAccum.make["gpu"](ctx=ctx)
        return a^

    def read[target: StaticString](mut self) raises -> ACTWindowMetrics:
        """Window means. GPU: four `[2]`-buffer D2Hs behind their own syncs —
        flush cadence only.

        The step COUNT comes off the device with the loss mean, not from the
        host `n`: see the field's note. The `gn` accumulator can legitimately
        hold fewer folds than the others (a forward-only window never touches
        it), so the loss window is what defines `n`."""
        comptime if target == "cpu":
            if self.n == 0:
                return ACTWindowMetrics(0.0, 0.0, 0.0, 0.0, 0)
            var d = Float64(self.n)
            return ACTWindowMetrics(
                self._h_loss / d, self._h_l1 / d, self._h_kl / d,
                self._h_gn / d, self.n,
            )
        else:
            # Each mean stands on its own (`DeviceMeanAccum.read` already
            # returns 0 for an empty window); only `n` is taken from the loss
            # accumulator. Deliberately NOT an early return on `n == 0` — that
            # would make the grad-norm mean depend on whether the loss window
            # happened to be fed, which is a coupling nothing else implies.
            var m_loss = Scalar[DT](0.0)
            var n_dev = 0
            self.loss.read_into["gpu"](m_loss, n_dev)
            return ACTWindowMetrics(
                Float64(m_loss),
                Float64(self.l1.read["gpu"]()),
                Float64(self.kl.read["gpu"]()),
                Float64(self.gn.read["gpu"]()),
                n_dev,
            )

    def reset[target: StaticString](mut self) raises:
        comptime if target == "cpu":
            self._h_loss = 0.0
            self._h_l1 = 0.0
            self._h_kl = 0.0
            self._h_gn = 0.0
        else:
            self.loss.reset["gpu"]()
            self.l1.reset["gpu"]()
            self.kl.reset["gpu"]()
            self.gn.reset["gpu"]()
        self.n = 0


struct ACTTrainer[
    QPOS: Int,
    ADIM: Int,
    N_CAM: Int,
    IMG_H: Int,
    IMG_W: Int,
    K: Int,
    DIM: Int,
    HEADS: Int,
    FF: Int,
    LATENT: Int,
    N_ENC: Int,
    N_DEC: Int,
    BATCH: Int,
    P: Float64 = ACT_DROPOUT,
    target: StaticString = "cpu",
    # See `loss_graph.mojo` — swappable so a GPU-vs-CPU gate need not
    # instantiate ResNet18's 40 layers twice. Default unchanged.
    FEAT_CH: Int = RESNET18_OUT_CH,
    OH: Int = ResNet18OutH[IMG_H],
    OW: Int = ResNet18OutW[IMG_W],
    BACKBONE: Module = ResNet18Backbone[3, IMG_H, IMG_W],
](Movable & Deinitable):
    comptime LG = ACTLossGraph[
        Self.QPOS,
        Self.ADIM,
        Self.N_CAM,
        Self.IMG_H,
        Self.IMG_W,
        Self.K,
        Self.DIM,
        Self.HEADS,
        Self.FF,
        Self.LATENT,
        Self.N_ENC,
        Self.N_DEC,
        Self.P,
        Self.FEAT_CH,
        Self.OH,
        Self.OW,
        Self.N_CAM * Self.OH * Self.OW,
        2 + Self.N_CAM * Self.OH * Self.OW,
        Self.K + 2,
        Self.BACKBONE,
    ]
    comptime ENC_SEQ: Int = Self.K + 2
    comptime IMG_ELEMS: Int = Self.N_CAM * 3 * Self.IMG_H * Self.IMG_W

    var graph: Self.LG
    var opt: Adam
    var max_grad_norm: Scalar[DT]
    var loss_out: Tensor
    var grad_seed: Tensor
    # Reusable input staging — `set_input` copies into the graph's own pool, so
    # these exist only to bridge the sampler's `List` buffers into a `Tensor`.
    var t_qpos: Tensor
    var t_images: Tensor
    var t_actions: Tensor
    var t_valid: Tensor
    var ctx: Optional[DeviceContext]
    var _adopted: Bool
    """Has `opt.adopt` packed the graph into the grouped arena yet? (GPU only.)

    Lazy on purpose — see "The arena, and when it is engaged" in the header:
    adoption rebinds every parameter's device buffer, so it must happen after
    anything that loads or injects weights by recreating one."""
    var _train_acc: ACTMetricAccum
    """Device-resident logging window for the training steps.

    Fed by `train_step_device_accum`, drained by `train_metrics`. Exists so the
    per-step path contains no D2H: `_read_terms` downloads three `[BATCH]`
    vectors behind a synchronization and `read_clip_norm` a fourth scalar
    behind another, EVERY step, for four numbers the logger averages over
    `LOG_EVERY` of them."""
    var _val_acc: ACTMetricAccum
    """The same, for validation passes — a separate window because folding
    validation batches into the training means would quietly bias the training
    curve toward the eval-mode model."""
    var _train_graph: Optional[CUDAGraph]
    """Lazily-captured CUDA graph for the accumulating train step (GPU only).

    None until `train_step_device_captured`'s first call, and a no-op shell on
    non-NVIDIA where `CUDAGraph` compiles to nothing. See that method."""
    var deterministic_latent: Bool
    """Pin the CVAE draw to its mean in TRAINING mode too.

    ⚠ Exists because `train_mode` sets three coupled flags at once, one of
    which is `z.deterministic`. Without a sticky preference, an explicit
    `set_node_attr["z", "deterministic"](1.0)` is silently undone by the next
    `eval_step`/`predict` (both restore training mode on exit) — the caller's
    override looks applied and is gone one call later. Off by default: sampling
    the latent IS the CVAE."""

    def __init__(out self):
        self.graph = Self.LG()
        self.opt = Adam()
        self.max_grad_norm = Scalar[DT](0.0)
        self.loss_out = Tensor()
        self.grad_seed = Tensor()
        self.t_qpos = Tensor()
        self.t_images = Tensor()
        self.t_actions = Tensor()
        self.t_valid = Tensor()
        self.ctx = None
        self._adopted = False
        self._train_acc = ACTMetricAccum()
        self._val_acc = ACTMetricAccum()
        self._train_graph = None
        self.deterministic_latent = False

    def __init__(out self, *, deinit move: Self):
        self.graph = move.graph^
        self.opt = move.opt^
        self.max_grad_norm = move.max_grad_norm
        self.loss_out = move.loss_out^
        self.grad_seed = move.grad_seed^
        self.t_qpos = move.t_qpos^
        self.t_images = move.t_images^
        self.t_actions = move.t_actions^
        self.t_valid = move.t_valid^
        self.ctx = move.ctx^
        self._adopted = move._adopted
        self._train_acc = move._train_acc^
        self._val_acc = move._val_acc^
        self._train_graph = move._train_graph^
        self.deterministic_latent = move.deterministic_latent

    @staticmethod
    def make(
        lr: Scalar[DT] = Scalar[DT](ACT_LR),
        kl_weight: Scalar[DT] = Scalar[DT](ACT_KL_WEIGHT),
        weight_decay: Scalar[DT] = Scalar[DT](ACT_WEIGHT_DECAY),
        max_grad_norm: Scalar[DT] = Scalar[DT](ACT_CLIP_MAX_NORM),
        ctx: Optional[DeviceContext] = None,
        shape_weight: Scalar[DT] = Scalar[DT](ACT_SHAPE_WEIGHT),
    ) raises -> Self:
        comptime if Self.target != "cpu":
            if not ctx:
                raise Error(
                    "ACTTrainer.make[target='gpu']: a DeviceContext is required"
                )
        var t = Self()
        t.ctx = ctx
        t.graph = Self.LG.make[Self.target, Kaiming](ctx)
        t.graph.set_node_attr["kls", "multiplier"](kl_weight)
        # the chunk-shape term: the two negations are fixed, the weight is the knob
        t.graph.set_node_attr["ahat_lo_neg", "multiplier"](Scalar[DT](-1.0))
        t.graph.set_node_attr["act_lo_neg", "multiplier"](Scalar[DT](-1.0))
        t.graph.set_node_attr["l1ds", "multiplier"](shape_weight)
        t.opt = Adam(lr=lr, wd=weight_decay)
        t.max_grad_norm = max_grad_norm
        # ⚠ The graph-IO staging tensors keep a HOST copy on both targets: the
        # sampler produces host `List`s, so a GPU step is host-fill then upload.
        # `loss_out` needs both — the graph writes it on device and the step
        # reads batch means from the host side.
        comptime if Self.target == "cpu":
            t.loss_out = Tensor.alloc(Self.BATCH)
            t.grad_seed = Tensor.alloc(Self.BATCH)
        else:
            var c = ctx.value()
            t.loss_out = Tensor.alloc(Self.BATCH)
            t.loss_out.ensure_gpu(c, Self.BATCH)
            t.grad_seed = Tensor.alloc(Self.BATCH)
            t.grad_seed.ensure_gpu(c, Self.BATCH)
        # The backward seed for a mean-over-batch loss is the constant 1/BATCH;
        # nothing in a step mutates it, so it is written once.
        seed_grad_inv_batch[Self.target, Self.BATCH](
            t.grad_seed.lt[Self.target, Layout.row_major(Self.BATCH, 1)](),
            ctx=ctx,
        )
        t.t_qpos = Tensor.alloc(Self.BATCH * Self.QPOS)
        t.t_images = Tensor.alloc(Self.BATCH * Self.IMG_ELEMS)
        t.t_actions = Tensor.alloc(Self.BATCH * Self.K * Self.ADIM)
        t.t_valid = Tensor.alloc(Self.BATCH * Self.ENC_SEQ)
        comptime if Self.target != "cpu":
            var c2 = ctx.value()
            t.t_qpos.ensure_gpu(c2, Self.BATCH * Self.QPOS)
            t.t_images.ensure_gpu(c2, Self.BATCH * Self.IMG_ELEMS)
            t.t_actions.ensure_gpu(c2, Self.BATCH * Self.K * Self.ADIM)
            t.t_valid.ensure_gpu(c2, Self.BATCH * Self.ENC_SEQ)
        t._train_acc = ACTMetricAccum.make[Self.target](ctx)
        t._val_acc = ACTMetricAccum.make[Self.target](ctx)
        t.train_mode(True)
        return t^

    def train_mode(mut self, training: Bool):
        """Flip dropout/BN, latent sampling and latent scaling TOGETHER.

        Splitting these across call sites is how a validation number ends up
        measured on a model that is still sampling its latent and still
        dropping units — a number that is wrong in a direction that looks like
        underfitting.
        """
        self.graph.set_attr["training"](
            Scalar[DT](1.0) if training else Scalar[DT](0.0)
        )
        self.graph.set_node_attr["zs", "multiplier"](
            Scalar[DT](1.0) if training else Scalar[DT](0.0)
        )
        # Eval always pins the draw; training pins it only if the caller asked.
        self.graph.set_node_attr["z", "deterministic"](
            Scalar[DT](0.0) if (
                training and not self.deterministic_latent
            ) else Scalar[DT](1.0)
        )

    def freeze_backbone_norm(mut self, v: Bool = True):
        """`FrozenBatchNorm2d` for the vision backbone — what BOTH ACT
        implementations build theirs with.

        Statistics AND affine become constants, so the ImageNet values survive
        training instead of being EMA'd away in the first few hundred steps
        (momentum 0.1 leaves 2.7e-05 of the original after 100). See
        `BatchNorm2D.frozen`.

        ⚠ ONLY MEANINGFUL WITH PRETRAINED STATISTICS. Frozen at the init values
        — mean 0, var 1, gamma 1, beta 0 — BatchNorm is the identity, so
        freezing a RANDOM backbone does not "hold" a normalization, it deletes
        one. `load_backbone` therefore turns this on itself, and this method is
        for the ablation.

        Broadcast to the whole graph, which is safe because the backbone is the
        only thing in it carrying BatchNorm: the transformer stacks use
        LayerNorm and every other module's `set_attr` ignores an unknown name.
        `frozen` overrides `training`, so a later `train_mode(True)` cannot
        undo this.
        """
        self.graph.set_attr["frozen"](
            Scalar[DT](1.0) if v else Scalar[DT](0.0)
        )

    def set_deterministic_latent(mut self, v: Bool):
        """Pin (or unpin) the CVAE draw in training mode, stickily.

        Use this rather than reaching for `set_node_attr["z", "deterministic"]`
        directly: the node attribute is rewritten on every mode switch, so a
        direct write survives only until the next `eval_step` or `predict`.
        """
        self.deterministic_latent = v
        self.graph.set_node_attr["z", "deterministic"](
            Scalar[DT](1.0) if v else Scalar[DT](0.0)
        )

    def _seed_inputs(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
    ) raises:
        """Copy one batch in. `valid` is the sampler's width-K mask; the graph
        wants width K+2, whose two leading entries ([CLS] and qpos) are never
        padding (`detr_vae.py:96`)."""
        for i in range(Self.BATCH * Self.QPOS):
            self.t_qpos.data[i] = qpos[i]
        for i in range(Self.BATCH * Self.IMG_ELEMS):
            self.t_images.data[i] = images[i]
        for i in range(Self.BATCH * Self.K * Self.ADIM):
            self.t_actions.data[i] = actions[i]
        for b in range(Self.BATCH):
            var base = b * Self.ENC_SEQ
            self.t_valid.data[base] = Scalar[DT](1.0)
            self.t_valid.data[base + 1] = Scalar[DT](1.0)
            for t in range(Self.K):
                self.t_valid.data[base + 2 + t] = valid[b * Self.K + t]

        comptime if Self.target != "cpu":
            # ⚠ `upload_resident`, NOT `upload`. `upload` recreates the device
            # buffer on EVERY call and synchronizes TWICE; these four tensors
            # are fixed-size (BATCH x the model dims) and allocated once at
            # `make`, so every one of those reallocations was pure churn.
            #
            # nsys on a 5090, 60 steps: 8,783 device alloc/free pairs, mean
            # `cuMemFree` 310 us (a plain free is 10-20 us — this one drains
            # outstanding work), totalling **3.45 s against 3.43 s of kernel
            # time**. The whole step spent as long managing memory as
            # computing. These four are 4 of the ~70 pairs per pass and 4 of
            # the ~148 synchronizations; the rest is below.
            var c = self.ctx.value()
            self.t_qpos.upload_resident(c)
            self.t_images.upload_resident(c)
            self.t_actions.upload_resident(c)
            self.t_valid.upload_resident(c)
        self.graph.set_input["qpos", Self.BATCH](self.t_qpos, self.ctx)
        self.graph.set_input["images", Self.BATCH](self.t_images, self.ctx)
        self.graph.set_input["actions", Self.BATCH](self.t_actions, self.ctx)
        self.graph.set_input["enc_valid", Self.BATCH](self.t_valid, self.ctx)

    def seed_inputs_device(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
        val: Bool = False,
    ) raises:
        """Draw + gather a batch entirely on the device, then seed the graph.

        The device replacement for `sample_batch` + `_seed_inputs`. Nothing
        crosses the bus: the sampler writes the four staging tensors' DEVICE
        buffers, and `ComputeGraph.set_input` is a device-to-device
        `enqueue_copy` on GPU, so the whole path is kernels.

        What it removes per call, measured on the host path it replaces:
        16.1 ms of `sample_batch` with the GPU idle, two 29.5 MB
        element-by-element fills into pinned memory, the 29.5 MB H2D, and the
        four `upload_resident` device synchronizations.

        ⚠ It does NOT reproduce `sample_batch`'s batches. The device draws with
        Philox and the host with a xorshift, so the two samplers walk different
        streams and no seed reconciles them; `tests/.../test_act_dataset_gpu`
        gates the part that must agree (the gather, given the same rows).
        Anything that needs a SPECIFIC batch — a reference comparison, a
        reproducible eval — must keep using the host path.
        """
        comptime assert Self.target != "cpu", (
            "seed_inputs_device is GPU-only"
        )
        var c = self.ctx.value()
        ds.sample[Self.BATCH, Self.K](
            val,
            self.t_qpos,
            self.t_images,
            self.t_actions,
            self.t_valid,
            c,
        )
        self.graph.set_input["qpos", Self.BATCH](self.t_qpos, self.ctx)
        self.graph.set_input["images", Self.BATCH](self.t_images, self.ctx)
        self.graph.set_input["actions", Self.BATCH](self.t_actions, self.ctx)
        self.graph.set_input["enc_valid", Self.BATCH](self.t_valid, self.ctx)

    def train_step_device(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
    ) raises -> ACTStepResult:
        """`train_step` with the data path on the device.

        Identical to `train_step` from the forward onward — same graph, same
        clip, same optimizer — so the only difference is where the batch came
        from."""
        self._ensure_adopted()
        self.opt.zero_grad[Self.target](self.graph, self.ctx)
        self.seed_inputs_device(ds, False)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.graph.vjp[Self.BATCH, Self.target](self.grad_seed, self.ctx)
        self.opt.clip_grads_device[Self.target](
            self.graph, self.max_grad_norm, self.ctx
        )
        var gn = Float64(self.opt.read_clip_norm(self.ctx.value()))
        self.opt.step[Self.target](self.graph, self.ctx)
        return ACTStepResult(terms.loss, terms.l1, terms.kl, gn)

    def eval_step_device(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
        val: Bool = True,
    ) raises -> ACTStepResult:
        """Forward-only on a FRESH batch drawn on the device.

        The validation counterpart of `train_step_device`. Distinct from
        `eval_step_resident`, which re-scores whatever is already in the input
        slots: a validation pass needs its own batches, so this seeds.

        ⚠ The caller must pin `ds`'s RNG offset around the pass
        (`set_offset`), exactly as the host path pins `ds.rng`. Validation that
        scores different batches every time makes `best_val` the minimum of a
        noisy estimate — it selects the luckiest draw, not the best model."""
        self.train_mode(False)
        self.seed_inputs_device(ds, val)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.train_mode(True)
        return terms

    def eval_step_resident(mut self) raises -> ACTStepResult:
        """Forward-only on the batch ALREADY in the graph's input slots.

        `eval_step` re-seeds from host lists, which under the device data path
        would reintroduce the four `upload_resident` uploads and their syncs
        just to measure a forward. Nothing between `train_step_device` and here
        touches the input slots — `forward` and `vjp` read them, the optimizer
        does not — so the batch is still there and re-seeding is pure cost.

        ⚠ Only valid straight after a `*_device` step. Called on its own it
        measures a forward over whatever happened to be in the slots, which is
        a stale batch, not an error you would notice."""
        self.train_mode(False)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.train_mode(True)
        return terms

    def _read_terms(mut self) raises -> ACTStepResult:
        """Batch means of (loss, l1, kl), read off the graph's own nodes."""
        var lo = Float64(0.0)
        var l1 = Float64(0.0)
        var kl = Float64(0.0)
        ref l1n = self.graph.node_output["l1"]()
        ref kln = self.graph.node_output["kl"]()
        comptime if Self.target != "cpu":
            # ⚠ ONE sync for three reads. `Tensor.download` synchronizes on
            # every call, so the obvious spelling (an explicit `synchronize`
            # then three `download`s) costs FOUR full device drains per call
            # and this runs twice per training iteration — 8 of the ~28
            # synchronizations a step was paying. `download_enqueue` /
            # `download_finalize` exist for exactly this; the enqueues are
            # ordered behind the forward on the same stream, so the leading
            # `synchronize()` was redundant too.
            var c = self.ctx.value()
            self.loss_out.download_enqueue(c)
            l1n.download_enqueue(c)
            kln.download_enqueue(c)
            c.synchronize()
            self.loss_out.download_finalize()
            l1n.download_finalize()
            kln.download_finalize()
        for b in range(Self.BATCH):
            lo += Float64(self.loss_out.data[b])
            l1 += Float64(l1n.data[b])
            kl += Float64(kln.data[b])
        var n = Float64(Self.BATCH)
        return ACTStepResult(lo / n, l1 / n, kl / n, 0.0)

    # ── device-resident logging (no per-step D2H) ────────────────────────
    #
    # `_read_terms` + `read_clip_norm` are what stand between an ACT step and
    # a CUDA-graph capture, and they are ours: four downloads behind two full
    # device synchronizations, every step, for four scalars the logger averages
    # over `LOG_EVERY` steps. The pattern below is SAC's
    # (`sac/trainer.mojo::train_device_kernels`): reduce each `[BATCH]` vector
    # on device into a `[2]` `[sum_of_means, count]` accumulator, and drain the
    # accumulators at flush cadence.
    #
    # ⚠ The two families do NOT mix. `train_step`/`train_step_device` return a
    # per-step `ACTStepResult` and touch no accumulator; `*_accum` folds and
    # returns nothing. A loop that called both and then read `train_metrics`
    # would report a window over only half its steps — so pick one per call
    # site, which is what the examples do under their `GPU_DATA` switch.

    def _accum_terms[VAL: Bool](mut self) raises:
        """Fold this forward's (loss, l1, kl) batch means into a window.

        GPU: three one-block reductions, no bus traffic, capture-safe. CPU: the
        same means computed on the host, where they already live."""
        comptime lb = Layout.row_major(Self.BATCH)
        ref l1n = self.graph.node_output["l1"]()
        ref kln = self.graph.node_output["kl"]()
        comptime if Self.target == "cpu":
            var lo = Float64(0.0)
            var a1 = Float64(0.0)
            var ak = Float64(0.0)
            for b in range(Self.BATCH):
                lo += Float64(self.loss_out.data[b])
                a1 += Float64(l1n.data[b])
                ak += Float64(kln.data[b])
            var d = Float64(Self.BATCH)
            comptime if VAL:
                self._val_acc._h_loss += lo / d
                self._val_acc._h_l1 += a1 / d
                self._val_acc._h_kl += ak / d
                self._val_acc.n += 1
            else:
                self._train_acc._h_loss += lo / d
                self._train_acc._h_l1 += a1 / d
                self._train_acc._h_kl += ak / d
                self._train_acc.n += 1
        else:
            var v1 = l1n.lt["gpu", lb]()
            var vk = kln.lt["gpu", lb]()
            var vl = self.loss_out.lt["gpu", lb]()
            comptime if VAL:
                self._val_acc.loss.accumulate_gpu_lt[Self.BATCH](vl)
                self._val_acc.l1.accumulate_gpu_lt[Self.BATCH](v1)
                self._val_acc.kl.accumulate_gpu_lt[Self.BATCH](vk)
                self._val_acc.n += 1
            else:
                self._train_acc.loss.accumulate_gpu_lt[Self.BATCH](vl)
                self._train_acc.l1.accumulate_gpu_lt[Self.BATCH](v1)
                self._train_acc.kl.accumulate_gpu_lt[Self.BATCH](vk)
                self._train_acc.n += 1

    def _accum_grad_norm(mut self) raises:
        """Fold the pre-clip grad norm `clip_grads_device` just wrote into the
        TRAINING window, off its device buffer.

        ⚠ Silently a no-op when the optimizer has no device norm to offer,
        which is exactly the case where `clip_grads_device` fell back to the
        host path — there the norm was computed on the host and this window is
        not the thing reporting it. `train_step_device_accum` requires the
        arena, so on its path the buffer is always there."""
        comptime if Self.target != "cpu":
            if self.opt.has_clip_norm_dev():
                var v = self.opt.clip_norm_dev()
                self._train_acc.gn.accumulate_gpu_lt[1](v)

    def train_step_device_accum(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
    ) raises:
        """`train_step_device` with NO host readback.

        Identical arithmetic — same draw, same graph, same clip, same optimizer
        — but the four logged scalars are reduced into device accumulators
        instead of downloaded, so the per-step sequence is pure kernels: no
        D2H, no synchronization, no allocation. That is the shape a CUDA graph
        can capture, and it is worth having even while capture is blocked
        (`docs/ACT_GPU_DATA_PATH.md`), because the host stops draining the
        pipeline twice per step.

        ⚠ Returns nothing, deliberately. A per-step value is not available
        without the synchronization this exists to remove, and returning a
        zero or a stale one would read as a training regression rather than as
        a missing number. Read the window with `train_metrics`."""
        self._ensure_adopted()
        self.opt.zero_grad[Self.target](self.graph, self.ctx)
        self.seed_inputs_device(ds, False)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        self._accum_terms[False]()
        self.graph.vjp[Self.BATCH, Self.target](self.grad_seed, self.ctx)
        self.opt.clip_grads_device[Self.target](
            self.graph, self.max_grad_norm, self.ctx
        )
        self._accum_grad_norm()
        self.opt.step[Self.target](self.graph, self.ctx)

    # ── CUDA-graph capture (optional; NVIDIA only) ───────────────────────
    #
    # ⚠ NOT USABLE YET, AND NOT BECAUSE OF THIS CODE. MAX's split-K allocates
    # its reduction workspace per call with the SYNCHRONOUS driver allocator,
    # which is illegal inside a capture region — measured, and it aborts the
    # capture (`bench_cuda_graph_splitk_capture.mojo`, and the table in
    # `docs/ACT_GPU_DATA_PATH.md`). Our OWN split-K uses persistent scratch and
    # is fine; what remains is the shapes MAX still routes itself. This path is
    # here so that when that lands the test is one comptime flag away.
    #
    # Everything else the capture needs is already true of
    # `train_step_device_accum`: the batch is drawn on device, the optimizer is
    # the grouped arena with an on-device β^t, the clip uses persistent scratch,
    # and the metrics are folded rather than downloaded. What is left is the
    # HOST work in that step, which does not run on a replay:
    #
    #   `_ensure_adopted`      -> `prepare_device_capture` does it eagerly
    #   `opt.begin_step`       -> re-issued below on replay (host `t`)
    #   `ParamVersionBump`     -> defeated by `capture_recast`, see below
    #   `ds.offset_host += 2B` -> `note_replayed_sample`, below
    #   `_train_acc.n += 1`    -> not needed; the window count is device-side

    def prepare_device_capture(mut self) raises:
        """Make the accumulating step safe to capture. Call ONCE, before the
        loop, after any weight loading.

        Two things, and the second is the one that bites:

        * adopt the optimizer arena eagerly, so the allocation `adopt` performs
          cannot land inside a capture region;
        * set `capture_recast`, which makes every `Linear`/`Conv2D` re-pad and
          re-cast its weight cache UNCONDITIONALLY instead of gating on
          `param.version`.

        ⚠ Without the second, ACT trains against FROZEN WEIGHTS under replay
        and nothing says so. The version bump lives in a host walk at the end
        of `Adam.step`; a replay never runs it, so the version never advances,
        so `_ensure_w_pad` decides its cached padded weight is still current —
        forever. The forward then reads the capture-time weights while the
        optimizer diligently updates memory nobody reads. The loss curve goes
        flat, which reads as a bad learning rate. This project has already been
        bitten by exactly this cache, in the non-captured case (`w_pad` gated
        on a version bumped in the wrong place).

        The cost is a re-pad every forward rather than one per optimizer step.
        That is the price of capture and it is why this is not the default."""
        comptime assert Self.target != "cpu", (
            "prepare_device_capture is GPU-only"
        )
        self._ensure_adopted()
        self.graph.set_attr["capture_recast"](Scalar[DT](1.0))

    def train_step_device_captured(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
    ) raises:
        """`train_step_device_accum`, captured on first call and replayed after.

        On non-NVIDIA `CUDAGraph` is a compile-time no-op and this runs the
        step directly every call — bit-identical to calling
        `train_step_device_accum`, which is what makes the path verifiable on
        Metal even though only CUDA can actually capture.

        ⚠ Call `prepare_device_capture` first. It is not done here because it
        must happen after checkpoint loading, and a method that silently did it
        on first use would be doing it at exactly the wrong moment for a
        caller who loads weights lazily.

        ⚠ THE FIRST CALL TAKES TWO OPTIMIZER STEPS. `maybe_capture_replay` runs
        the closure once to settle the stream and once more under capture; both
        are real steps that update weights and advance the samplers. Harmless
        over a training run, and worth knowing before comparing step counts.

        ⚠ Run a few eager `train_step_device_accum`s before the first captured
        call. One warmup inside the helper settles the stream, not necessarily
        every lazy device allocation the graph performs — anything still
        allocating on step 2 would allocate inside the capture region and abort
        it. `CAPTURE_WARMUP` in the examples is that.
        """
        comptime assert Self.target != "cpu", (
            "train_step_device_captured is GPU-only"
        )
        # Was there a live captured graph BEFORE this call? If so, this call is
        # a replay and none of the step's host work ran — see the block above.
        # A disabled graph does not count: `maybe_capture_replay` runs the
        # closure directly in that case, so the host work DID run.
        var replayed = False
        if self._train_graph:
            replayed = not self._train_graph.value().is_disabled()

        # Move the slot into a disjoint local so the closure can borrow `self`
        # without overlapping the slot's mutable borrow (the same dance
        # `nn/training/autoregressive_trainer.mojo` does).
        var g = Optional[CUDAGraph](None)
        if self._train_graph:
            g = Optional[CUDAGraph](self._train_graph.take())

        def _captured() capturing raises -> None:
            self.train_step_device_accum(ds)

        maybe_capture_replay[_captured](g, self.ctx.value())
        self._train_graph = g^

        if replayed:
            # Host bookkeeping the replayed kernels have their device-side
            # counterpart for, but which no host line advanced.
            #
            # `begin_step` refreshes the HOST `t` / bias corrections. The
            # grouped kernel reads `_pow_dev`, which the capture advances on
            # device, so training is already correct without this — what it
            # buys is a checkpoint whose step count and bias correction match
            # the weights it is saved beside.
            self.opt.begin_step()
            ds.note_replayed_sample[Self.BATCH]()

    def has_captured_graph(self) -> Bool:
        """Is a usable captured graph live? False on non-NVIDIA, and false when
        the driver offered no stream to capture (`CUDAGraph.is_disabled`), in
        which case the step is running directly and correctly."""
        if self._train_graph:
            return not self._train_graph.value().is_disabled()
        return False

    def captured_graph_nodes(self) -> Int:
        """Kernel nodes in the captured graph; 0 if nothing is captured.

        Worth printing once. A capture that silently recorded nothing — the
        closure enqueued on a stream other than the one being captured — looks
        exactly like a working one from the outside, and replaying an empty
        graph is a training loop that does nothing at full speed."""
        if self._train_graph:
            return self._train_graph.value().num_nodes()
        return 0

    def eval_step_device_accum(
        mut self,
        mut ds: ACTDeviceDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
        val: Bool = True,
    ) raises:
        """`eval_step_device` folding into the VALIDATION window instead of
        downloading. Read it with `val_metrics` after the pass.

        ⚠ Same pinning requirement as `eval_step_device`: pin `ds`'s RNG offset
        around the whole pass (`set_offset`), or every validation scores
        different batches and `best_val` selects the luckiest draw."""
        self.train_mode(False)
        self.seed_inputs_device(ds, val)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        self._accum_terms[True]()
        self.train_mode(True)

    def eval_step_resident_accum(mut self) raises:
        """`eval_step_resident` folding into the validation window.

        ⚠ Only valid straight after a `*_device*` step — see
        `eval_step_resident`. Re-scores the batch already in the input slots."""
        self.train_mode(False)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        self._accum_terms[True]()
        self.train_mode(True)

    def train_metrics(
        mut self, reset: Bool = True
    ) raises -> ACTWindowMetrics:
        """Drain the training window: means since the last reset, and the step
        count. FOUR D2Hs — call at logging cadence, never per step.

        `reset=False` peeks without clearing, for a progress line that must not
        disturb the logger's window."""
        var w = self._train_acc.read[Self.target]()
        if reset:
            self._train_acc.reset[Self.target]()
        return w

    def val_metrics(mut self, reset: Bool = True) raises -> ACTWindowMetrics:
        """Drain the validation window — the mean over the pass's batches.

        ⚠ `grad_norm` is always 0 here: a forward-only pass computes no
        gradients. It is reported rather than omitted so the two windows share
        a type."""
        var w = self._val_acc.read[Self.target]()
        if reset:
            self._val_acc.reset[Self.target]()
        return w

    def _ensure_adopted(mut self) raises:
        """Pack the graph into the optimizer's grouped arena, once, on GPU.

        Deliberately NOT done in `make`: see the header. Everything that seeds
        weights — `load`, `load_reference_params`, a test injecting a CPU
        model's parameters — runs between `make` and the first `train_step`,
        and adoption after that point is only safe because those paths upload
        into the RESIDENT buffer."""
        comptime if Self.target != "cpu":
            if not self._adopted:
                self.opt.adopt[Self.target](self.graph, self.ctx)
                self._adopted = True

    def train_step(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
    ) raises -> ACTStepResult:
        self._ensure_adopted()
        # GPU+adopted: ONE fill over the contiguous grad arena. CPU: the
        # per-param walk, unchanged.
        self.opt.zero_grad[Self.target](self.graph, self.ctx)
        self._seed_inputs(qpos, images, actions, valid)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.graph.vjp[Self.BATCH, Self.target](self.grad_seed, self.ctx)

        var gn: Float64
        comptime if Self.target == "cpu":
            var ss = _SumSq()
            walk_params[Self.target](self.graph, ss, self.ctx, String(""))
            gn = ss.sum_sq ** 0.5
            if self.max_grad_norm > Scalar[DT](0.0) and gn > Float64(
                self.max_grad_norm
            ):
                # A non-finite norm scales to zero rather than propagating NaN
                # into every weight — mirrors
                # `lewm/trainer.mojo::_scale_from_norm`.
                var sc = Scalar[DT](0.0)
                if gn == gn:
                    sc = self.max_grad_norm / Scalar[DT](gn)
                var scaler = _ScaleGrads(sc)
                walk_params[Self.target](self.graph, scaler, self.ctx, String("")
                )
        else:
            # Same clip, on device: sum-of-squares over the grad arena →
            # `scale = min(1, max_norm/‖g‖)` (non-finite → 0, `max_norm <= 0` →
            # 1) → one scaling pass. Persistent scratch, so no allocation. The
            # ONE D2H is the pre-clip norm, which `ACTStepResult` reports; the
            # host walk it replaces synchronized once per PARAMETER.
            self.opt.clip_grads_device[Self.target](
                self.graph, self.max_grad_norm, self.ctx
            )
            gn = Float64(self.opt.read_clip_norm(self.ctx.value()))

        # GPU+adopted: one grouped kernel over the whole arena. CPU: the
        # per-param `ParamVisitor` walk `begin_step` + `for_each_param` did.
        self.opt.step[Self.target](self.graph, self.ctx)
        return ACTStepResult(terms.loss, terms.l1, terms.kl, gn)

    def eval_step(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
    ) raises -> ACTStepResult:
        """Forward only, in eval mode (z = 0, no dropout, BN running stats).

        Restores training mode on the way out, so a validation call inside a
        training loop cannot silently leave the model in eval.
        """
        self.train_mode(False)
        self._seed_inputs(qpos, images, actions, valid)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        var terms = self._read_terms()
        self.train_mode(True)
        return terms

    def predict(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
        mut out_actions: List[Scalar[DT]],
    ) raises:
        """The inference path: run in eval mode and read `a_hat`.

        `actions` is still required — the CVAE encoder runs and its output is
        scaled to zero rather than the encoder being skipped (see
        `loss_graph.mojo`). Zeros are a fine argument; nothing downstream of the
        zeroed latent reads them.
        """
        self.train_mode(False)
        self._seed_inputs(qpos, images, actions, valid)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        comptime N = Self.BATCH * Self.K * Self.ADIM
        if len(out_actions) != N:
            out_actions = List[Scalar[DT]](unsafe_uninit_length=N)
        ref ahat = self.graph.node_output["ahat"]()
        comptime if Self.target != "cpu":
            var c = self.ctx.value()
            c.synchronize()
            ahat.download(c)
        for i in range(N):
            out_actions[i] = ahat.data[i]
        self.train_mode(True)



    def predict_prior_sample(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
        mut out_actions: List[Scalar[DT]],
    ) raises:
        """Inference with `z ~ N(0, I)` — a draw from the prior — where
        `predict` uses the prior MEAN (`z = 0`, the reference's rule).

        On demonstrations whose future timing varies across operators the
        prior mean is the flat conditional median (one action for all 40
        positions, 2026-09-20); a draw commits to one style per query, the way
        the nearest-neighbour control (0.72 on LIBERO's inits) commits to one
        demonstration. Meaningful only for a fit whose KL let the latent stay
        informative (`ACT_KL` below the paper's 10); on a collapsed latent the
        draw changes nothing. Eval mode otherwise (no dropout, BN running
        stats). `actions` is unread past the encoder; zeros are fine.
        """
        self.train_mode(False)
        self.graph.set_node_attr["zs", "multiplier"](Scalar[DT](1.0))
        self.graph.set_node_attr["z", "deterministic"](Scalar[DT](0.0))
        self.graph.set_node_attr["z", "prior"](Scalar[DT](1.0))
        self._seed_inputs(qpos, images, actions, valid)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        comptime N = Self.BATCH * Self.K * Self.ADIM
        if len(out_actions) != N:
            out_actions = List[Scalar[DT]](unsafe_uninit_length=N)
        ref ahat = self.graph.node_output["ahat"]()
        comptime if Self.target != "cpu":
            var c = self.ctx.value()
            c.synchronize()
            ahat.download(c)
        for i in range(N):
            out_actions[i] = ahat.data[i]
        self.graph.set_node_attr["z", "prior"](Scalar[DT](0.0))
        self.train_mode(True)
    def predict_with_posterior(
        mut self,
        ref qpos: List[Scalar[DT]],
        ref images: List[Scalar[DT]],
        ref actions: List[Scalar[DT]],
        ref valid: List[Scalar[DT]],
        mut out_actions: List[Scalar[DT]],
    ) raises:
        """A DIAGNOSTIC forward, not a policy: eval mode (no dropout, BN
        running stats, the latent draw pinned to its mean) but the latent
        token kept — `z = mu(qpos, actions)`, the CVAE posterior of the chunk
        handed in as `actions` — where `predict` scales it to zero, the
        reference's inference rule (`detr_vae.py:110`).

        It exists to split two readings of a flat inference chunk. The box
        fits of 2026-09-20 predicted ONE action for all 40 positions under
        `predict` while their TRAINING L1 (0.25-0.35) sat below any flat
        predictor's floor (~0.41): in training the decoder does shape its
        chunk, and the only input that differs between the two modes is the
        latent. If this forward's chunk varies and `predict`'s does not, the
        shape lives in `z` alone — the encoder is doing the decoder's job and
        the prior mean carries none of it. That is the CVAE failure the paper's
        KL weight is meant to prevent, and a shape term on the decoder's own
        output (`ACT_SHAPE`) is the lever that does not depend on `z`.
        """
        self.train_mode(False)
        self.graph.set_node_attr["zs", "multiplier"](Scalar[DT](1.0))
        self._seed_inputs(qpos, images, actions, valid)
        self.graph.forward[Self.BATCH, Self.target](self.loss_out, self.ctx)
        comptime N = Self.BATCH * Self.K * Self.ADIM
        if len(out_actions) != N:
            out_actions = List[Scalar[DT]](unsafe_uninit_length=N)
        ref ahat = self.graph.node_output["ahat"]()
        comptime if Self.target != "cpu":
            var c = self.ctx.value()
            c.synchronize()
            ahat.download(c)
        for i in range(N):
            out_actions[i] = ahat.data[i]
        self.train_mode(True)

    # ── checkpoints ──────────────────────────────────────────────────────

    def save(mut self, path: String, save_moments: Bool = True) raises:
        """A v3 binary named checkpoint: Param sections (+ Adam moments) then
        State sections (BatchNorm running statistics).

        ⚠ The State pass is not optional. Without the running statistics a
        reloaded model runs BatchNorm on whatever its init held (mean 0,
        var 1), and every prediction is wrong in a way that looks like a
        training failure rather than a load failure.

        ⚠ With the moments goes Adam's step state (`opt.t`, `β₁ᵗ`, `β₂ᵗ`, as
        `K` scalars): moments restored under `t = 0` are divided by the
        step-1 bias corrections, and a resumed run's first few hundred
        updates come out ~0.3× the size they should.
        """
        var w = BinaryCheckpointWriter(save_moments)
        w.mode = 0
        walk_params[Self.target](self.graph, w, self.ctx)
        w.mode = 1
        var _sref1 = ParamVisitorRef.of[type_of(w), Self.target](w)
        self.graph.for_each_state[Self.target](_sref1, self.ctx)
        if save_moments:
            var sc = CheckpointScalars()
            self.opt.put_step_state(sc, "opt")
            w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_backbone(
        mut self, dump_dir: String, freeze_norm: Bool = True
    ) raises -> Int:
        """Fill the vision backbone from a `dump_resnet18_imagenet.py` dump.

        Returns the number of tensors filled. Raises if the dump names a tensor
        the backbone does not have, or sizes it differently — a pretrained
        loader that silently leaves half the network random is worse than none,
        because the run then reports "pretraining did not help".

        ⚠ `feat.0.` is the backbone's path inside the ACT graph:
        `Tokenwise[N_CAM, BACKBONE]` contributes the `.0`. The dump names the
        same tensors `rn18in.*`, backbone-local, which is the mapping
        `dump_act_reference.py:emit_resnet18` writes for the standalone gate —
        so both use one mapping and the gate covers this path too.

        ⚠ TWO WALKS. Weights are parameters; BatchNorm running statistics are
        STATE. Pretrained convolutions carrying init statistics (mean 0, var 1)
        are not the pretrained network, so skipping the second walk would load
        45 MB of weights and still change the function.
        """
        comptime GP = "feat.0."
        comptime DP = "rn18in."
        var wl = LoadPrefixedParams[GP, DP](RefDump(String(dump_dir)))
        self.graph.for_each_param[Self.target, LoadPrefixedParams[GP, DP]](
            wl, self.ctx
        )
        if len(wl.missing) > 0:
            raise Error(
                "load_backbone: " + String(len(wl.missing))
                + " backbone weights absent from the dump, first '"
                + wl.missing[0] + "' — the dump was written for a different"
                " ResNet variant or a different resolution"
            )
        var sl = LoadPrefixedParams[GP, DP](RefDump(String(dump_dir)))
        self.graph.for_each_state[Self.target, LoadPrefixedParams[GP, DP]](
            sl, self.ctx
        )
        if len(sl.missing) > 0:
            raise Error(
                "load_backbone: " + String(len(sl.missing))
                + " BatchNorm running statistics absent, first '"
                + sl.missing[0] + "'"
            )
        if len(wl.loaded) == 0:
            raise Error(
                "load_backbone: matched NOTHING under '" + String(GP)
                + "' — the graph's backbone path changed and this loader did"
                " not, so the weights would have been silently discarded"
            )
        # ⚠ FREEZING IS PART OF LOADING, not a separate decision, because the
        # references never do one without the other: `norm_layer=
        # FrozenBatchNorm2d` is passed at the same `resnet18(pretrained=...)`
        # call. Loading ImageNet statistics and then letting training-mode
        # BatchNorm EMA them away is the worst of both — the cost of the load
        # with none of the benefit, and it reads as "pretraining did not help".
        # `freeze_norm=False` is for the ablation that measures exactly that.
        if freeze_norm:
            self.freeze_backbone_norm(True)
        return len(wl.loaded) + len(sl.loaded)

    def load_backbone_safetensors(
        mut self,
        var path: String = String(""),
        freeze_norm: Bool = True,
        verbose: Bool = True,
    ) raises -> Int:
        """Fill the vision backbone from an ImageNet `.safetensors`, no Python.

        The default `path` is empty, meaning "fetch `timm/resnet18.tv_in1k`
        from the Hub and cache it" — that repo is torchvision's `IMAGENET1K_V1`
        republished with a safetensors file, so this is the same 45 MB of
        weights `load_backbone` reads from a `dump_resnet18_imagenet.py` dump,
        without `torch`, `torchvision` or the `act-ref` environment.

        ⚠ "tv_in1k" is a claim in a repo name, not a proof. It is checked:
        `tests/nn/test_safetensors_resnet18_torch.mojo` compares every value of
        this file against that dump and requires BIT equality (11,190,912
        values). If timm ever re-uploads different weights under the same name,
        that gate goes red before a training run silently changes.

        Same contract as `load_backbone`: returns the number of tensors filled,
        raises rather than half-loading, and freezes the BatchNorms unless told
        not to.

        ⚠ TWO WALKS, one visitor. Weights are parameters; BatchNorm running
        statistics are STATE. Pretrained convolutions carrying this framework's
        init statistics (mean 0, var 1) are not the pretrained network, they
        are a different function that happens to share its weights. Reusing one
        visitor across both walks means the coverage check sees the whole
        backbone rather than half of it.
        """
        comptime GP = "feat.0."
        if path == "":
            path = hf_download_file(
                String(RESNET18_TV_REPO),
                String(RESNET18_TV_FILE),
                HF_MODEL,
                verbose=verbose,
            )
        var v = LoadTorchNamed[GP](
            SafeTensors(path^), resnet18_torch_map(3)
        )
        self.graph.for_each_param[Self.target, LoadTorchNamed[GP]](v, self.ctx)
        self.graph.for_each_state[Self.target, LoadTorchNamed[GP]](v, self.ctx)
        v.report(String("load_backbone_safetensors"))
        # ⚠ FREEZING IS PART OF LOADING — see `load_backbone` for why the two
        # are one decision and not two.
        if freeze_norm:
            self.freeze_backbone_norm(True)
        return len(v.loaded) + len(v.zeroed)

    def load_backbone_auto(
        mut self, spec: String, freeze_norm: Bool = True, verbose: Bool = True
    ) raises -> Int:
        """Dispatch `ACT_PRETRAINED` to the right backbone loader.

            "hub" | "1"          fetch `timm/resnet18.tv_in1k` and use it
            "*.safetensors"      that file, torchvision-named
            anything else        a `dump_resnet18_imagenet.py` DIRECTORY

        One rule, one place. Both `examples/so101/act_so101_train_cpu.mojo` and
        `..._gpu.mojo` read the same variable, and a rule spelled out at two
        call sites is this repo's most frequent defect shape.

        The dump path stays first-class rather than deprecated: it is the
        ORACLE the safetensors path is gated against
        (`tests/nn/test_safetensors_resnet18_torch.mojo`), so a run that wants
        to bypass the network — or to check the two agree — still can.
        """
        if spec == "hub" or spec == "1":
            if verbose:
                print(
                    "  backbone  ImageNet from the Hub ("
                    + String(RESNET18_TV_REPO) + ")"
                )
            return self.load_backbone_safetensors(
                String(""), freeze_norm, verbose
            )
        if spec.endswith(".safetensors"):
            return self.load_backbone_safetensors(
                String(spec), freeze_norm, verbose
            )
        return self.load_backbone(spec, freeze_norm)

    def save_backbone_safetensors(
        mut self, var path: String
    ) raises -> Int:
        """Export the vision backbone under TORCHVISION's names and layout.

        The file a `torchvision.models.resnet18()` can `load_state_dict` (with
        `strict=False`: it has no `fc`, and `num_batches_tracked` is bookkeeping
        we do not carry). The point is to get a fine-tuned backbone back out to
        the ecosystem, so the names, the shapes and the layout are theirs, not
        ours — that is what `torch_names.mojo` is for."""
        comptime GP = "feat.0."
        var v = SaveTorchNamed[GP](resnet18_torch_map(3))
        self.graph.for_each_param[Self.target, SaveTorchNamed[GP]](v, self.ctx)
        self.graph.for_each_state[Self.target, SaveTorchNamed[GP]](v, self.ctx)
        v.report(String("save_backbone_safetensors"))
        v.writer.add_metadata(String("producer"), String("noeira"))
        v.writer.add_metadata(String("format"), String("pt"))
        v.writer.save(path^)
        return len(v.written)

    def load(mut self, path: String) raises:
        var bytes = _read_file_bytes(path)
        if not _is_v3_header(bytes):
            raise Error(
                "ACTTrainer.load: '" + path + "' is not a v3 binary checkpoint"
            )
        var r = BinaryCheckpointReader(bytes^)
        r.mode = 0
        walk_params[Self.target](self.graph, r, self.ctx)
        r.mode = 1
        var _sref2 = ParamVisitorRef.of[type_of(r), Self.target](r)
        self.graph.for_each_state[Self.target](_sref2, self.ctx)
        # ⚠ NOT REDUNDANT: the call above hands the state pass a POINTER to
        # `r`, so without a later mention Mojo destroys `r` at that line and
        # the pass reads freed memory. See `BinaryCheckpointReader.finish`.
        var sc = r.read_scalars()
        r.finish()
        # A checkpoint written with moments carries the step state too; one
        # from before `K` sections existed leaves the optimizer at t = 0.
        self.opt.take_step_state(sc, "opt")
