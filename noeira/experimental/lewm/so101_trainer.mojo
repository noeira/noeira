"""LeWMProprioTrainer — `LeWMRefTrainer`'s training step over
`ref_model.LeWMLossGraphProprio` (the SO-101 world model v2:
noeira-docs/SO101_LEWM_PLAN.md S1 v2).

Same recipe, same helpers (`ref_trainer`'s qkv-bias mask, torch-style clip,
AdamW on every parameter, `_DumpWriter`): only the graph differs, by a third
input — each frame's proprioception — and the two nodes that fold it into the
latent. A separate struct because the trainer is not generic over its graph
(the graph trait has no node attributes) and the PushT graph must keep its
parameter set for its parity gates. Floats in (`train_step` / `loss_of`); no
uint8 staging path.
"""

from std.math import sqrt
from std.os import makedirs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Kaiming
from .ref_model import LeWMLossGraphProprio
from .ref_load import LoadRef
from noeira.deep_agents.act.refload import RefDump
from .ref_trainer import (
    REF_T,
    RefStepStats,
    WMWindowLoss,
    copy_last_loss,
    _AdamWAll,
    _MaskPredQKVBias,
    _GradSumSq,
    _GradScale,
    _GradScaleDev,
    _clip_coef_kernel,
    _DumpWriter,
)


struct LeWMProprioTrainer[
    target: StaticString, B: Int, IN_CH: Int, IMG: Int, ACT_IN: Int, PROP: Int,
](Movable, WMWindowLoss):
    comptime PIX = REF_T * Self.IN_CH * Self.IMG * Self.IMG
    comptime ACT = REF_T * Self.ACT_IN
    comptime PRO = REF_T * Self.PROP
    comptime Graph = LeWMLossGraphProprio[
        Self.IN_CH, Self.IMG, 14, 192, 3, 12, 192, 2048,
        REF_T, Self.ACT_IN, Self.PROP, 3, 1,
        16, 64, 2048, 6,
        1024, 17,
    ]

    var graph: Self.Graph
    var opt: _AdamWAll
    var max_norm: Float64
    var ctx: Optional[DeviceContext]
    var loss: Tensor
    var seed: Tensor
    var pix: Tensor
    var act: Tensor
    var pro: Tensor
    var norm_parts: Tensor
    var norm_dev: Tensor
    var norm_host: Float64
    var dropout_on: Bool

    def __init__(
        out self,
        ctx: Optional[DeviceContext],
        lr: Float64 = 5e-5,
        wd: Float64 = 1e-3,
        max_norm: Float64 = 1.0,
        sigreg_lambda: Float64 = 0.09,
        checkpoint: Bool = True,
        dropout: Bool = False,
    ) raises:
        comptime assert Self.target == "cpu" or Self.target == "gpu"
        self.graph = Self.Graph.make[Self.target, Kaiming](ctx)
        self.graph.set_node_attr["emb", "checkpoint"](Scalar[DT](1 if checkpoint else 0))
        self.dropout_on = dropout
        self.graph.set_node_attr["pred_raw", "dropout"](Scalar[DT](1 if dropout else 0))
        self.graph.set_node_attr["sig_s", "multiplier"](Scalar[DT](sigreg_lambda))
        self.graph.set_node_attr["sig", "resample"](Scalar[DT](1))
        self.opt = _AdamWAll(Scalar[DT](lr), Scalar[DT](wd), True)
        self.max_norm = max_norm
        self.ctx = ctx
        self.loss = Tensor.alloc(Self.B)
        self.seed = Tensor.alloc(Self.B)
        for i in range(Self.B):
            self.seed.data[i] = Scalar[DT](1.0 / Float64(Self.B))
        self.pix = Tensor.alloc(Self.B * Self.PIX)
        self.act = Tensor.alloc(Self.B * Self.ACT)
        self.pro = Tensor.alloc(Self.B * Self.PRO)
        self.norm_parts = Tensor()
        self.norm_dev = Tensor()
        self.norm_host = 0.0
        comptime if Self.target == "gpu":
            self.seed.upload(ctx.value())

    def load(mut self, dump_dir: String) raises -> Int:
        """Every Param and BN State from a dump (`save_dump`)."""
        var lv = LoadRef(RefDump(dump_dir), String(""))
        self.graph.for_each_param[Self.target](lv, self.ctx)
        self.graph.for_each_state[Self.target](lv, self.ctx)
        lv.check()
        return lv.loaded

    def set_lr(mut self, lr: Float64):
        self.opt.adam.lr = Scalar[DT](lr)

    def set_eval(mut self):
        self.graph.set_attr["training"](Scalar[DT](0))
        self.graph.set_node_attr["pred_raw", "dropout"](Scalar[DT](0))

    def set_train(mut self):
        self.graph.set_attr["training"](Scalar[DT](1))
        self.graph.set_node_attr["pred_raw", "dropout"](Scalar[DT](1 if self.dropout_on else 0))

    def _inputs(
        mut self, pix: List[Scalar[DT]], act: List[Scalar[DT]], pro: List[Scalar[DT]]
    ) raises:
        for i in range(Self.B * Self.PIX):
            self.pix.data[i] = pix[i]
        for i in range(Self.B * Self.ACT):
            self.act.data[i] = act[i]
        for i in range(Self.B * Self.PRO):
            self.pro.data[i] = pro[i]
        comptime if Self.target == "gpu":
            self.pix.upload(self.ctx.value())
            self.act.upload(self.ctx.value())
            self.pro.upload(self.ctx.value())
        self.graph.set_input["pixels", Self.B](self.pix, self.ctx)
        self.graph.set_input["actions", Self.B](self.act, self.ctx)
        self.graph.set_input["proprio", Self.B](self.pro, self.ctx)

    def train_step(
        mut self, pix: List[Scalar[DT]], act: List[Scalar[DT]], pro: List[Scalar[DT]]
    ) raises -> RefStepStats:
        """pix (B, T, IN_CH, IMG, IMG) normalised; act (B, T, ACT_IN) and
        pro (B, T, PROP) z-scored."""
        self.graph.zero_grad[Self.target](self.ctx)
        self._inputs(pix, act, pro)
        self.graph.forward[Self.B, Self.target](self.loss, self.ctx)
        self.graph.vjp[Self.B, Self.target](self.seed, self.ctx)
        var mask = _MaskPredQKVBias()
        self.graph.for_each_param[Self.target](mask, self.ctx)
        if mask.n_masked != 6:
            raise Error("LeWMProprioTrainer: masked " + String(mask.n_masked) + " predictor qkv biases, expected 6")
        comptime if Self.target == "gpu":
            var c = self.ctx.value()
            self.norm_parts.ensure_gpu(c, 512)
            self.norm_dev.ensure_gpu(c, 2)
            var ss = _GradSumSq(self.norm_parts.dev.value())
            self.graph.for_each_param[Self.target](ss, self.ctx)
            if ss.k > 512:
                raise Error("LeWMProprioTrainer: more than 512 parameter tensors")
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
        self.opt.adam.begin_step()
        self.graph.for_each_param[Self.target](self.opt, self.ctx)
        var stats = self._stats()
        comptime if Self.target == "gpu":
            self.norm_dev.download(self.ctx.value())
            stats.grad_norm = Float64(self.norm_dev.data[0])
        else:
            stats.grad_norm = self.norm_host
        return stats^

    def loss_of(
        mut self, pix: List[Scalar[DT]], act: List[Scalar[DT]], pro: List[Scalar[DT]]
    ) raises -> RefStepStats:
        """Forward only, in the current mode (no update)."""
        self._inputs(pix, act, pro)
        self.graph.forward[Self.B, Self.target](self.loss, self.ctx)
        return self._stats()

    def window_losses(
        mut self, pix: List[Scalar[DT]], act: List[Scalar[DT]], pro: List[Scalar[DT]]
    ) raises -> Tuple[Float64, Float64]:
        """`WMWindowLoss`: (pred loss, copy-last loss)."""
        var st = self.loss_of(pix, act, pro)
        var c = copy_last_loss[Self.target, Self.B, 3 * 192](
            self.graph.node_output["ctx_x"](), self.graph.node_output["tgt"](), self.ctx
        )
        return (st.pred_loss, c)

    def _stats(mut self) raises -> RefStepStats:
        var stats = RefStepStats(0.0, 0.0, 0.0, 0.0)
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

    def save_dump(mut self, dir: String) raises -> Int:
        """Every Param and BN State to `dir` (`ref_trainer._DumpWriter`'s
        format, which `load_ref` / `load` read)."""
        makedirs(dir, exist_ok=True)
        var w = _DumpWriter(dir)
        self.graph.for_each_param[Self.target](w, self.ctx)
        self.graph.for_each_state[Self.target](w, self.ctx)
        with open(dir + "/manifest.txt", "w") as f:
            f.write(w.manifest)
        var n = 0
        for b in w.manifest.as_bytes():
            if b == UInt8(ord("\n")):
                n += 1
        return n
