"""LeWMRefTrainer — trains the reference-exact LeWM the way le-wm-main's
`train.py` does (docs/LEWM_REOPEN_PLAN.md P6).

One `train_step` = `lejepa_forward` + `loss.backward()` + Lightning's
`gradient_clip_val` + `AdamW.step()`, over `ref_model.LeWMLossGraphRef` at the
paper's dimensions:

  * the loss graph in TRAIN mode (BatchNorm on batch statistics, unbiased
    running-variance update), batch mean via the 1/B seed;
  * the predictor's attention `to_qkv` bias — ours has one, the reference
    does not (ref_model.mojo) — gets its gradient ZEROED before the clip: it
    takes no part in the norm, its moments stay 0, and decay of 0 is 0, so it
    stays at the 0 it was loaded with;
  * torch's `clip_grad_norm_`: total norm over every gradient, coefficient
    max_norm / (norm + 1e-6), applied only when below 1;
  * AdamW with decoupled decay on EVERY parameter — stable-pretraining's
    `exclude_bias_norm` defaults to False, so biases, LN / BN affines, the CLS
    token and the position embeddings decay too. `nn.Adam` skips params built
    with `apply_decay=False`; `_AdamWAll` forwards every one with decay on;
  * activation checkpointing of the 12 ViT blocks (`checkpoint`, on by
    default): the encoder's memory is its block inputs plus one block;
  * SIGReg: a fresh projection matrix per step (`resample`, the default, like
    the reference's per-forward `torch.randn`) or an injected one
    (`set_sigreg_a`, for the torch gates).

Dropout: the reference trains the predictor with dropout 0.1; this graph has
none. The parity gates run torch with dropout off (`dropout_off`).
"""

from std.math import sqrt
from std.time import perf_counter_ns
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Kaiming
from noeira.nn.optimizer.adam import Adam
from noeira.deep_agents.act.refload import RefDump
from .ref_model import LeWMLossGraphRef
from .ref_load import LoadRef
from noeira.nn.optimizer.grad_clip import _sum_sq_kernel_rt, GC_TPB
from .trainer import _clip_scale_kernel


comptime REF_T = 4
"""Frames per training window: history 3 + 1 prediction."""
comptime REF_ACT_IN = 10
"""Frameskip 5 x 2: the action embedder's input per frame."""
comptime REF_IMG = 224

comptime LeWMRefGraph = LeWMLossGraphRef[
    3, REF_IMG, 14, 192, 3, 12, 192, 2048,
    REF_T, REF_ACT_IN, 3, 1,
    16, 64, 2048, 6,
    1024, 17,
]
"""The published model's training graph: ViT-tiny/14 at 224, projector 2048,
predictor 6 x (16 heads x 64), FF 2048, SIGReg 1024 projections x 17 knots."""


comptime _HW = REF_IMG * REF_IMG


def _lewm_pixels_kernel(
    src: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    n_frames: Int64,
):
    """`dst[f, c, p] = (src[f, p, c] / 255 - mean[c]) / std[c]`: the dataset's
    uint8 HWC frames to the encoder's ImageNet-normalised CHW, in torch's
    float32 order (`ToImage`: divide, then normalise). One thread per OUTPUT
    element, so the writes coalesce."""
    var idx = Int(global_idx.x)
    if idx >= Int(n_frames) * 3 * _HW:
        return
    var f = idx // (3 * _HW)
    var rem = idx % (3 * _HW)
    var c = rem // _HW
    var p = rem % _HW
    var x = Scalar[DT](src[unsafe_offset=(f * _HW + p) * 3 + c]) / Scalar[DT](255.0)
    var m = Scalar[DT](0.485)
    var sd = Scalar[DT](0.229)
    if c == 1:
        m = Scalar[DT](0.456)
        sd = Scalar[DT](0.224)
    elif c == 2:
        m = Scalar[DT](0.406)
        sd = Scalar[DT](0.225)
    dst[unsafe_offset=idx] = (x - m) / sd


def _clip_coef_kernel(
    parts: Pointer[Scalar[DT], MutAnyOrigin],
    k: Int64,
    max_norm: Scalar[DT],
    res: Pointer[Scalar[DT], MutAnyOrigin],
):
    """One thread: norm = sqrt(Σ parts[0..k)) in order, then torch's
    `clip_grad_norm_` coefficient min(1, max_norm / (norm + 1e-6)) —
    res = [norm, coef]. On the device, so the step never waits on the host."""
    if Int(global_idx.x) != 0:
        return
    var s = Scalar[DT](0)
    for i in range(Int(k)):
        s += parts[unsafe_offset=i]
    var norm = sqrt(s)
    var coef = max_norm / (norm + Scalar[DT](1e-6))
    if coef > Scalar[DT](1):
        coef = Scalar[DT](1)
    res[unsafe_offset=0] = norm
    res[unsafe_offset=1] = coef


def _scale_by_dev_kernel(
    grad: Pointer[Scalar[DT], MutAnyOrigin],
    n: Int64,
    coef: Pointer[Scalar[DT], MutAnyOrigin],
):
    """grad *= coef[1] (torch multiplies by the clamped coefficient always;
    × 1.0 is exact)."""
    var i = Int(global_idx.x)
    if i < Int(n):
        grad[unsafe_offset=i] = grad[unsafe_offset=i] * coef[unsafe_offset=1]


def _is_pred_qkv_bias(name: String) -> Bool:
    """`pred_raw.<i>.attn.0.0.bias`: the predictor's qkv bias (see header)."""
    return name.startswith("pred_raw.") and name.endswith(".attn.0.0.bias")


struct _MaskPredQKVBias(ParamVisitor):
    var n_masked: Int

    def __init__(out self):
        self.n_masked = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if not _is_pred_qkv_bias(name):
            return
        self.n_masked += 1
        comptime if target == "cpu":
            for i in range(N):
                grad.data[i] = Scalar[DT](0.0)
        else:
            grad.dev.value().enqueue_fill(Scalar[DT](0.0))


struct _GradSumSq(ParamVisitor):
    """Σ g² over every gradient. CPU: Float64 on the host. GPU: one
    single-block reduction per parameter into `parts[k]` on the device (the
    caller downloads the few hundred partials once and adds them in Float64)
    — downloading every gradient (72 MB) cost 0.05 s a step."""

    var sumsq: Float64
    var k: Int
    var parts: Optional[DeviceBuffer[DT]]

    def __init__(out self, parts: Optional[DeviceBuffer[DT]] = None):
        self.sumsq = 0.0
        self.k = 0
        self.parts = parts

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            ctx.value().enqueue_function[_sum_sq_kernel_rt](
                grad.dev.value(),
                Int64(N),
                self.parts.value().create_sub_buffer[DT](self.k, 1),
                grid_dim=1,
                block_dim=GC_TPB,
            )
            self.k += 1
            return
        var s = 0.0
        for i in range(N):
            var g = Float64(grad.data[i])
            s += g * g
        self.sumsq += s


struct _GradScale(ParamVisitor):
    var scale: Scalar[DT]

    def __init__(out self, scale: Scalar[DT]):
        self.scale = scale

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "cpu":
            for i in range(N):
                grad.data[i] = grad.data[i] * self.scale
        else:
            ctx.value().enqueue_function[_clip_scale_kernel[N]](
                grad.lt["gpu", Layout.row_major(N)](),
                self.scale,
                grid_dim=(N + TPB - 1) // TPB,
                block_dim=TPB,
            )


struct _GradScaleDev(ParamVisitor):
    var coef: DeviceBuffer[DT]

    def __init__(out self, coef: DeviceBuffer[DT]):
        self.coef = coef

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        ctx.value().enqueue_function[_scale_by_dev_kernel](
            grad.dev.value(), Int64(N), self.coef,
            grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )


struct _AdamWAll(ParamVisitor):
    """`nn.Adam`'s update with decay on every parameter (torch AdamW's
    default param group). The math, the moments and the version bump are
    Adam's own: this only overrides `apply_decay`."""

    var adam: Adam
    var decay_all: Bool

    def __init__(out self, lr: Scalar[DT], wd: Scalar[DT], decay_all: Bool = True):
        self.adam = Adam(lr=lr, wd=wd)
        self.decay_all = decay_all

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.adam.visit_rt[target](
            name, param, grad, m, v, N, apply_decay or self.decay_all, ctx
        )


@fieldwise_init
struct RefStepStats(Copyable, Movable, Writable):
    var loss: Float64
    var pred_loss: Float64
    var sigreg_loss: Float64
    var grad_norm: Float64
    """The pre-clip total norm (torch `clip_grad_norm_`'s return)."""


struct LeWMRefTrainer[target: StaticString, B: Int](Movable):
    comptime PIX = REF_T * 3 * REF_IMG * REF_IMG
    comptime ACT = REF_T * REF_ACT_IN

    var graph: LeWMRefGraph
    var opt: _AdamWAll
    var max_norm: Float64
    var ctx: Optional[DeviceContext]
    var loss: Tensor
    var seed: Tensor
    var pix: Tensor
    var act: Tensor
    var a_buf: Optional[DeviceBuffer[DT]]
    var pix_host: List[HostBuffer[DType.uint8]]
    """Two pinned uint8 staging slots, (B, T, 224, 224, 3) each: the host
    fills one while the GPU trains on the other (`submit_staged`)."""
    var pix_u8: Optional[DeviceBuffer[DType.uint8]]
    var norm_parts: Tensor
    """Per-parameter Σ g² partials (GPU clip)."""
    var norm_dev: Tensor
    """[norm, coef] of the last step's clip, on the device."""
    var norm_host: Float64
    var profile: Bool
    """Synchronise between the stages of `train_step` and accumulate their
    wall time in `t_stage` (seconds): copy-in, forward, vjp, clip, AdamW."""
    var t_stage: List[Float64]

    def __init__(
        out self,
        ctx: Optional[DeviceContext],
        lr: Float64 = 5e-5,
        wd: Float64 = 1e-3,
        max_norm: Float64 = 1.0,
        sigreg_lambda: Float64 = 0.09,
        decay_all: Bool = True,
        checkpoint: Bool = True,
    ) raises:
        comptime assert Self.target == "cpu" or Self.target == "gpu"
        self.graph = LeWMRefGraph.make[Self.target, Kaiming](ctx)
        # the 12 ViT blocks recompute their forward in the vjp (ref_model)
        self.graph.set_node_attr["emb", "checkpoint"](Scalar[DT](1 if checkpoint else 0))
        self.graph.set_node_attr["sig_s", "multiplier"](Scalar[DT](sigreg_lambda))
        self.graph.set_node_attr["sig", "resample"](Scalar[DT](1))
        self.opt = _AdamWAll(Scalar[DT](lr), Scalar[DT](wd), decay_all)
        self.max_norm = max_norm
        self.ctx = ctx
        self.loss = Tensor.alloc(Self.B)
        self.seed = Tensor.alloc(Self.B)
        for i in range(Self.B):
            self.seed.data[i] = Scalar[DT](1.0 / Float64(Self.B))
        self.pix = Tensor.alloc(Self.B * Self.PIX)
        self.act = Tensor.alloc(Self.B * Self.ACT)
        self.a_buf = None
        self.pix_host = List[HostBuffer[DType.uint8]]()
        self.pix_u8 = None
        self.norm_parts = Tensor()
        self.norm_dev = Tensor()
        self.norm_host = 0.0
        self.profile = False
        self.t_stage = List[Float64](length=5, fill=0.0)
        comptime if Self.target == "gpu":
            self.seed.upload(ctx.value())

    def _lap(mut self, stage: Int, mut t: Int) raises:
        if not self.profile:
            return
        comptime if Self.target == "gpu":
            self.ctx.value().synchronize()
        var now = Int(perf_counter_ns())
        self.t_stage[stage] += Float64(now - t) / 1e9
        t = now

    def load(mut self, dump_dir: String) raises -> Int:
        """Every Param and BN State from a converted dump (`ours.<name>`)."""
        var lv = LoadRef(RefDump(dump_dir), String(""))
        self.graph.for_each_param[Self.target](lv, self.ctx)
        self.graph.for_each_state[Self.target](lv, self.ctx)
        lv.check()
        return lv.loaded

    def set_lr(mut self, lr: Float64):
        self.opt.adam.lr = Scalar[DT](lr)

    def set_sigreg_a(mut self, a: List[Scalar[DT]]) raises:
        """Inject the (D, P) column-normalised projection matrix for the next
        steps (turns `resample` off until `resample_sigreg`)."""
        var c = self.ctx.value() if self.ctx else DeviceContext()
        var buf = c.enqueue_create_buffer[DT](len(a))
        with buf.map_to_host() as h:
            for i in range(len(a)):
                h[i] = a[i]
        self.graph.set_node_attr_buf["sig", "fixed_a"](buf)
        self.a_buf = buf

    def train_step(
        mut self, pix: List[Scalar[DT]], act: List[Scalar[DT]]
    ) raises -> RefStepStats:
        """pix: (B, T, 3, 224, 224) ImageNet-normalised; act: (B, T, 10)
        z-scored, NaN already zeroed (train.py's `nan_to_num`)."""
        var t = Int(perf_counter_ns())
        for i in range(Self.B * Self.PIX):
            self.pix.data[i] = pix[i]
        comptime if Self.target == "gpu":
            self.pix.upload(self.ctx.value())
        self._set_actions(act)
        self._lap(0, t)
        self._submit(t)
        return self.finish()

    def staging(mut self, slot: Int = 0) raises -> Pointer[Scalar[DType.uint8], MutAnyOrigin]:
        """Pinned uint8 staging slot 0 or 1: (B, T, 224, 224, 3), the dataset's
        own layout — a loader writes each window straight into it
        (`LewmPushTExpert.sample_clip_pixels_uint8`). GPU.

        ⚠ Slot s is read by an ASYNC copy once submitted: refill it only
        after the `finish()` of the step that read it."""
        comptime assert Self.target == "gpu", "staging: GPU only"
        if len(self.pix_host) == 0:
            var c = self.ctx.value()
            for _ in range(2):
                self.pix_host.append(c.enqueue_create_host_buffer[DType.uint8](Self.B * Self.PIX))
            self.pix_u8 = c.enqueue_create_buffer[DType.uint8](Self.B * Self.PIX)
            c.synchronize()
        return rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](
            self.pix_host[slot].unsafe_ptr()
        )

    def train_step_staged(mut self, act: List[Scalar[DT]], slot: Int = 0) raises -> RefStepStats:
        """`submit_staged` then `finish`."""
        self.submit_staged(act, slot)
        return self.finish()

    def submit_staged(mut self, act: List[Scalar[DT]], slot: Int = 0) raises:
        """Enqueue a whole training step on the frames in `staging(slot)` and
        return WITHOUT waiting: 77 MB of uint8 go up (not 308 MB of float32),
        normalised on the device by `_lewm_pixels_kernel` (bit-equal to
        torch's float32 `ToImage`); the clip runs on the device. The host is
        free to fill the other slot; `finish()` waits and reads the step.
        (With `profile` on, every stage synchronises.)"""
        comptime assert Self.target == "gpu", "submit_staged: GPU only"
        var t = Int(perf_counter_ns())
        var c = self.ctx.value()
        if len(self.pix_host) == 0:
            raise Error("submit_staged: call staging() and fill it first")
        c.enqueue_copy(self.pix_u8.value(), self.pix_host[slot])
        self.pix.ensure_gpu(c, Self.B * Self.PIX)
        c.enqueue_function[_lewm_pixels_kernel](
            self.pix_u8.value(),
            self.pix.dev.value(),
            Int64(Self.B * REF_T),
            grid_dim=(Self.B * Self.PIX + TPB - 1) // TPB,
            block_dim=TPB,
        )
        self._set_actions(act)
        self._lap(0, t)
        self._submit(t)

    def _set_actions(mut self, act: List[Scalar[DT]]) raises:
        for i in range(Self.B * Self.ACT):
            self.act.data[i] = act[i]
        comptime if Self.target == "gpu":
            self.act.upload(self.ctx.value())

    def _submit(mut self, mut t: Int) raises:
        self.graph.zero_grad[Self.target](self.ctx)
        self.graph.set_input["pixels", Self.B](self.pix, self.ctx)
        self.graph.set_input["actions", Self.B](self.act, self.ctx)
        self.graph.forward[Self.B, Self.target](self.loss, self.ctx)
        self._lap(1, t)
        self.graph.vjp[Self.B, Self.target](self.seed, self.ctx)
        self._lap(2, t)

        var mask = _MaskPredQKVBias()
        self.graph.for_each_param[Self.target](mask, self.ctx)
        if mask.n_masked != 6:
            raise Error("LeWMRefTrainer: masked " + String(mask.n_masked) + " predictor qkv biases, expected 6")
        # torch.nn.utils.clip_grad_norm_: clip_coef = max_norm / (norm + 1e-6),
        # clamped to 1
        comptime if Self.target == "gpu":
            var c = self.ctx.value()
            self.norm_parts.ensure_gpu(c, 512)
            self.norm_dev.ensure_gpu(c, 2)
            var ss = _GradSumSq(self.norm_parts.dev.value())
            self.graph.for_each_param[Self.target](ss, self.ctx)
            if ss.k > 512:
                raise Error("LeWMRefTrainer: more than 512 parameter tensors")
            c.enqueue_function[_clip_coef_kernel](
                self.norm_parts.dev.value(), Int64(ss.k), Scalar[DT](self.max_norm),
                self.norm_dev.dev.value(), grid_dim=1, block_dim=1,
            )
            var sc = _GradScaleDev(self.norm_dev.dev.value())
            self.graph.for_each_param[Self.target](sc, self.ctx)
        else:
            var ss = _GradSumSq()
            self.graph.for_each_param[Self.target](ss, self.ctx)
            self.norm_host = sqrt(ss.sumsq)
            var coef = self.max_norm / (self.norm_host + 1e-6)
            if coef < 1.0:
                var sc = _GradScale(Scalar[DT](coef))
                self.graph.for_each_param[Self.target](sc, self.ctx)
        self._lap(3, t)
        self.opt.adam.begin_step()
        self.graph.for_each_param[Self.target](self.opt, self.ctx)
        self._lap(4, t)

    def finish(mut self) raises -> RefStepStats:
        """Wait for the submitted step; its loss, the two terms and the
        pre-clip gradient norm."""
        var stats = RefStepStats(0.0, 0.0, 0.0, self.norm_host)
        comptime if Self.target == "gpu":
            var c = self.ctx.value()
            c.synchronize()
            self.loss.download(c)
            self.graph.node_output["pl"]().download(c)
            self.graph.node_output["sig"]().download(c)
            self.norm_dev.download(c)
            stats.grad_norm = Float64(self.norm_dev.data[0])
        for b in range(Self.B):
            stats.loss += Float64(self.loss.data[b])
            stats.pred_loss += Float64(self.graph.node_output["pl"]().data[b])
            stats.sigreg_loss += Float64(self.graph.node_output["sig"]().data[b])
        stats.loss /= Float64(Self.B)
        stats.pred_loss /= Float64(Self.B)
        stats.sigreg_loss /= Float64(Self.B)
        return stats^
