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
  * SIGReg: a fresh projection matrix per step (`resample`, the default, like
    the reference's per-forward `torch.randn`) or an injected one
    (`set_sigreg_a`, for the torch gates).

Dropout: the reference trains the predictor with dropout 0.1; this graph has
none. The parity gates run torch with dropout off (`dropout_off`).
"""

from std.math import sqrt
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Kaiming
from noeira.nn.optimizer.adam import Adam
from noeira.deep_agents.act.refload import RefDump
from .ref_model import LeWMLossGraphRef
from .ref_load import LoadRef
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
    """Σ g² over every gradient, accumulated in Float64 on the host."""

    var sumsq: Float64

    def __init__(out self):
        self.sumsq = 0.0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            grad.download(ctx.value())
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

    def __init__(
        out self,
        ctx: Optional[DeviceContext],
        lr: Float64 = 5e-5,
        wd: Float64 = 1e-3,
        max_norm: Float64 = 1.0,
        sigreg_lambda: Float64 = 0.09,
        decay_all: Bool = True,
    ) raises:
        comptime assert Self.target == "cpu" or Self.target == "gpu"
        self.graph = LeWMRefGraph.make[Self.target, Kaiming](ctx)
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
        comptime if Self.target == "gpu":
            self.seed.upload(ctx.value())

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
        for i in range(Self.B * Self.PIX):
            self.pix.data[i] = pix[i]
        for i in range(Self.B * Self.ACT):
            self.act.data[i] = act[i]
        comptime if Self.target == "gpu":
            self.pix.upload(self.ctx.value())
            self.act.upload(self.ctx.value())
        self.graph.zero_grad[Self.target](self.ctx)
        self.graph.set_input["pixels", Self.B](self.pix, self.ctx)
        self.graph.set_input["actions", Self.B](self.act, self.ctx)
        self.graph.forward[Self.B, Self.target](self.loss, self.ctx)
        self.graph.vjp[Self.B, Self.target](self.seed, self.ctx)

        var mask = _MaskPredQKVBias()
        self.graph.for_each_param[Self.target](mask, self.ctx)
        if mask.n_masked != 6:
            raise Error("LeWMRefTrainer: masked " + String(mask.n_masked) + " predictor qkv biases, expected 6")
        var ss = _GradSumSq()
        self.graph.for_each_param[Self.target](ss, self.ctx)
        var norm = sqrt(ss.sumsq)
        # torch.nn.utils.clip_grad_norm_: clip_coef = max_norm / (norm + 1e-6),
        # clamped to 1 (a no-op multiply when not clipping)
        var coef = self.max_norm / (norm + 1e-6)
        if coef < 1.0:
            var sc = _GradScale(Scalar[DT](coef))
            self.graph.for_each_param[Self.target](sc, self.ctx)
        self.opt.adam.begin_step()
        self.graph.for_each_param[Self.target](self.opt, self.ctx)

        var stats = RefStepStats(0.0, 0.0, 0.0, norm)
        comptime if Self.target == "gpu":
            var c = self.ctx.value()
            c.synchronize()
            self.loss.download(c)
            self.graph.node_output["pl"]().download(c)
            self.graph.node_output["sig"]().download(c)
        for b in range(Self.B):
            stats.loss += Float64(self.loss.data[b])
            stats.pred_loss += Float64(self.graph.node_output["pl"]().data[b])
            stats.sigreg_loss += Float64(self.graph.node_output["sig"]().data[b])
        stats.loss /= Float64(Self.B)
        stats.pred_loss /= Float64(Self.B)
        stats.sigreg_loss /= Float64(Self.B)
        return stats^
