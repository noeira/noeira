"""GaussianHead[IN, ACT] — CleanRL-style PPO actor head (storage design).

State-dependent mean (Linear) + a single learnable state-independent `log_std`
vector. Output is `[μ | log σ]` of width `2*ACT`; `log σ` is the clamped
broadcast of the learnable vector (state-independent, CleanRL convention).

STORAGE migration: the storage `Module` surface (`TensorRefs[1, o]` in, owned
`Tensor` out; `.lt[target, layout]()` device views; `Param` = val+grd Tensors).
`for_each_param` / `zero_grad` are the inherited reflection defaults (they
auto-discover `weight` / `bias` / `log_std`). No `_cached_input_ptr` — `vjp`
receives `forward_input` and reads the cache from it directly (like `Linear`).

The math + the five GPU kernels are unchanged from the legacy leaf; only the
storage scaffolding (views, Param access, no cached pointer) differs.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.param import Param, ParamVisitor
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] = 2.0


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one thread per output / param element (verbatim from the
# legacy leaf; they take `MutAnyOrigin` LayoutTensors == what `.lt` yields).
# ──────────────────────────────────────────────────────────────────────


def _gauss_head_forward_kernel[BATCH: Int, IN: Int, ACT: Int](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(IN, ACT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    log_std: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * 2 * ACT
    if idx < total:
        var b = idx // (2 * ACT)
        var j = idx % (2 * ACT)
        if j < ACT:
            var acc = rebind[Scalar[DT]](bias[j])
            for i in range(IN):
                acc += rebind[Scalar[DT]](input[b, i]) * rebind[Scalar[DT]](
                    weight[i, j]
                )
            output[b, j] = acc
        else:
            var k = j - ACT
            var v = rebind[Scalar[DT]](log_std[k])
            if v < LOG_STD_MIN:
                v = LOG_STD_MIN
            elif v > LOG_STD_MAX:
                v = LOG_STD_MAX
            output[b, j] = v


def _gauss_head_grad_input_kernel[BATCH: Int, IN: Int, ACT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(IN, ACT), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * IN
    if idx < total:
        var b = idx // IN
        var i = idx % IN
        var acc: Scalar[DT] = 0.0
        for j in range(ACT):
            acc += rebind[Scalar[DT]](grad_output[b, j]) * rebind[Scalar[DT]](
                weight[i, j]
            )
        grad_input[b, i] = acc


def _gauss_head_grad_w_kernel[BATCH: Int, IN: Int, ACT: Int](
    cache: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    grad_w: LayoutTensor[DT, Layout.row_major(IN, ACT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = IN * ACT
    if idx < total:
        var i = idx // ACT
        var j = idx % ACT
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](cache[b, i]) * rebind[Scalar[DT]](
                grad_output[b, j]
            )
        grad_w[i, j] = rebind[Scalar[DT]](grad_w[i, j]) + s


def _gauss_head_grad_b_kernel[BATCH: Int, ACT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    grad_b: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < ACT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](grad_output[b, j])
        grad_b[j] = rebind[Scalar[DT]](grad_b[j]) + s


def _gauss_head_grad_ls_kernel[BATCH: Int, ACT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    grad_ls: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < ACT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](grad_output[b, ACT + j])
        grad_ls[j] = rebind[Scalar[DT]](grad_ls[j]) + s


# ──────────────────────────────────────────────────────────────────────
# GaussianHead.
# ──────────────────────────────────────────────────────────────────────


struct GaussianHead[IN: Int, ACT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN)
    comptime OUT_DIM = 2 * Self.ACT
    comptime W_SIZE = Self.IN * Self.ACT
    comptime B_SIZE = Self.ACT
    comptime LS_SIZE = Self.ACT
    comptime DEFAULT_LOG_STD_INIT: Scalar[DT] = 0.0

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    var log_std: Param["log_std", False, Self.LS_SIZE]

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.log_std = Param["log_std", False, Self.LS_SIZE]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "GaussianHead: target must be 'cpu' or 'gpu'"
        )
        var h = Self()
        h.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        h.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        h.log_std = Param["log_std", False, Self.LS_SIZE].make[target](ctx)
        INIT.init_weight[target](
            h.weight.val, Self.W_SIZE, Self.IN, Self.ACT, ctx
        )
        INIT.init_bias[target](h.bias.val, Self.B_SIZE, ctx)
        # log_std seeded to a constant (host-fill → upload on GPU), mirroring
        # the legacy leaf's `enqueue_fill` default.
        for k in range(Self.LS_SIZE):
            h.log_std.val.data[k] = Self.DEFAULT_LOG_STD_INIT
        comptime if target == "gpu":
            h.log_std.val.upload(ctx.value())
        return h^

    def set_log_std_init[
        target: StaticString
    ](
        mut self, value: Scalar[DT], ctx: Optional[DeviceContext] = None
    ) raises:
        """Override the log_std initialization. Call after `make`. Fills the
        host `.data` and (on GPU) writes it into the EXISTING device buffer.

        ⚠⚠ `upload_resident`, NOT `upload`. `Tensor.upload` REALLOCATES the
        device buffer on every call, and by the time a caller can reach this
        (after `make`) the trainer's Adam has already ADOPTED the old buffer
        into its arena. With `upload` the forward read the new buffer — frozen
        at `value` — while Adam updated the old one, so the log-std never
        learned: the first GPU PPO run on the tower (2026-09-26) logged its
        entropy at exactly the init value, 2.5136 for 6 joints at -1, after 640
        updates. No GPU caller existed before that run."""
        for k in range(Self.LS_SIZE):
            self.log_std.val.data[k] = value
        comptime if target == "gpu":
            self.log_std.val.upload_resident(ctx.value())

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            var x = TileTensor(in0.data, row_major[B, Self.IN]())
            var w = TileTensor(self.weight.val.data, row_major[Self.IN, Self.ACT]())
            var bt = TileTensor(self.bias.val.data, row_major[Self.ACT]())
            var ls = TileTensor(self.log_std.val.data, row_major[Self.ACT]())
            var out_v = TileTensor(out.data, row_major[B, 2 * Self.ACT]())
            for bi in range(B):
                # mu = x @ W + b
                for j in range(Self.ACT):
                    var acc = bt[j]
                    for i in range(Self.IN):
                        acc += x[bi, i] * w[i, j]
                    out_v[bi, j] = acc
                # log_std broadcast + clamp
                for j in range(Self.ACT):
                    var v = ls[j]
                    if v < LOG_STD_MIN:
                        v = LOG_STD_MIN
                    elif v > LOG_STD_MAX:
                        v = LOG_STD_MAX
                    out_v[bi, Self.ACT + j] = v
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            comptime n_blocks = (B * 2 * Self.ACT + TPB - 1) // TPB
            comptime fwd_kernel = _gauss_head_forward_kernel[B, Self.IN, Self.ACT]
            c.enqueue_function[fwd_kernel](
                in0.lt["gpu", Layout.row_major(B, Self.IN)](),
                self.weight.val.lt["gpu", Layout.row_major(Self.IN, Self.ACT)](),
                self.bias.val.lt["gpu", Layout.row_major(Self.ACT)](),
                self.log_std.val.lt["gpu", Layout.row_major(Self.ACT)](),
                out.lt["gpu", Layout.row_major(B, 2 * Self.ACT)](),
                grid_dim=n_blocks, block_dim=TPB,
            )

    # ----- Backward --------------------------------------------------------

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.IN)
            var w = TileTensor(self.weight.val.data, row_major[Self.IN, Self.ACT]())
            var go = TileTensor(grad_output.data, row_major[B, 2 * Self.ACT]())
            var x = TileTensor(fin.data, row_major[B, Self.IN]())
            var gi = TileTensor(gin.data, row_major[B, Self.IN]())
            # (1) grad_b, grad_log_std
            var gb = TileTensor(self.bias.grd.data, row_major[Self.ACT]())
            var gls = TileTensor(self.log_std.grd.data, row_major[Self.ACT]())
            for j in range(Self.ACT):
                var acc_b: Scalar[DT] = 0.0
                var acc_l: Scalar[DT] = 0.0
                for bi in range(B):
                    acc_b += go[bi, j]
                    acc_l += go[bi, Self.ACT + j]
                gb[j] = gb[j] + acc_b
                gls[j] = gls[j] + acc_l
            # (2) grad_w += xᵀ @ grad_mu
            var gw = TileTensor(self.weight.grd.data, row_major[Self.IN, Self.ACT]())
            for i in range(Self.IN):
                for j in range(Self.ACT):
                    var acc: Scalar[DT] = 0.0
                    for bi in range(B):
                        acc += x[bi, i] * go[bi, j]
                    gw[i, j] = gw[i, j] + acc
            # (3) grad_input = grad_mu @ Wᵀ
            for bi in range(B):
                for i in range(Self.IN):
                    var acc: Scalar[DT] = 0.0
                    for j in range(Self.ACT):
                        acc += go[bi, j] * w[i, j]
                    gi[bi, i] = acc
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN)
            var go_lt = grad_output.lt["gpu", Layout.row_major(B, 2 * Self.ACT)]()
            # (1) grad_b, grad_log_std
            comptime n_blocks_gb = (Self.ACT + TPB - 1) // TPB
            c.enqueue_function[_gauss_head_grad_b_kernel[B, Self.ACT]](
                go_lt,
                self.bias.grd.lt["gpu", Layout.row_major(Self.ACT)](),
                grid_dim=n_blocks_gb, block_dim=TPB,
            )
            c.enqueue_function[_gauss_head_grad_ls_kernel[B, Self.ACT]](
                go_lt,
                self.log_std.grd.lt["gpu", Layout.row_major(Self.ACT)](),
                grid_dim=n_blocks_gb, block_dim=TPB,
            )
            # (2) grad_w (reads forward_input as the cache)
            comptime n_blocks_gw = (Self.W_SIZE + TPB - 1) // TPB
            c.enqueue_function[_gauss_head_grad_w_kernel[B, Self.IN, Self.ACT]](
                fin.lt["gpu", Layout.row_major(B, Self.IN)](),
                go_lt,
                self.weight.grd.lt["gpu", Layout.row_major(Self.IN, Self.ACT)](),
                grid_dim=n_blocks_gw, block_dim=TPB,
            )
            # (3) grad_input
            comptime n_blocks_gi = (B * Self.IN + TPB - 1) // TPB
            c.enqueue_function[_gauss_head_grad_input_kernel[B, Self.IN, Self.ACT]](
                go_lt,
                self.weight.val.lt["gpu", Layout.row_major(Self.IN, Self.ACT)](),
                gin.lt["gpu", Layout.row_major(B, Self.IN)](),
                grid_dim=n_blocks_gi, block_dim=TPB,
            )
