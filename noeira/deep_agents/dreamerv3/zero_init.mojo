"""Scale the reward-predictor and critic OUTPUT layers at init (Finding 4).

DreamerV3 paper (Robust predictions / Critic learning, p.6): *"randomly
initialized reward predictor and critic networks can result in large
predicted rewards [...] that can delay the onset of learning. We thus
initialize the output weight matrix of the reward predictor and critic to
zeros."* On a negative-reward task (e.g. Pendulum) a non-neutral head makes
the imagined λ-returns optimistically positive and the actor optimizes a
reward landscape that doesn't exist.

We generalize the paper's hard zero to a SCALE applied to the Kaiming-init
output layer (`scale=0` == the paper's zero-init):
  * `scale = 0.0`  → exact zero (best for negative-reward tasks; Pendulum).
  * `scale` small (e.g. 0.1) → near-neutral but keeps a little of the random
    Kaiming asymmetry. On POSITIVE-reward tasks (CartPole) the pre-zero-init
    optimism actually helped exploration / fast solve; a small scale restores
    some of it without the large-prediction blow-up of full Kaiming.
The output is `scale·Kaiming`, so predictions start ≈ `scale`× the Kaiming
magnitude over the symexp bins.

The reward / value / slow-value heads are 1-hidden-layer MLPs (`nets.mojo`
pins the head MLP depth to 1), so the output `Linear` is Sequential child
index 3 → params `3.weight` / `3.bias`. Inside the reward `ComputeGraph`
(`RewLossGraph`) the head is node `rew`, so the names are prefixed
`rew.3.weight` / `rew.3.bias`. We scale BOTH weight and bias.

Only the OUTPUT layer is scaled — scaling the whole head toward 0 would also
shrink the downstream weight and choke the hidden layers' gradient.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor, walk_params, ParamVisitorRT
from noeira.nn.core.module import Module
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import GraphDecl


def _scale_k_rt(dst: Pointer[Scalar[DT], MutAnyOrigin], n_arg: Int64, scale: Scalar[DT]):
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        dst[unsafe_offset=i] = scale * dst[unsafe_offset=i]


struct _ScaleOutVisitor(ParamVisitor, ParamVisitorRT):
    """Scales the two named output-layer params (weight + bias) in place;
    branches on `target` internally (host loop on CPU, scale kernel on GPU)."""

    var wname: String
    var bname: String
    var scale: Scalar[DT]

    def __init__(out self, wname: String, bname: String, scale: Scalar[DT]):
        self.wname = wname
        self.bname = bname
        self.scale = scale

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
        if name == self.wname or name == self.bname:
            # A written weight must advance `version`: `Linear`'s GPU forward
            # reads a version-gated padded copy (`w_pad`) and would keep the
            # old weight.
            param.version += 1
            comptime if target == "cpu":
                for k in range(n):
                    param.data[k] = self.scale * param.data[k]
            else:
                var c = ctx.value()
                var nblk = (n + TPB - 1) // TPB
                c.enqueue_function[_scale_k_rt](
                    param.dev.value(),
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


def scale_output_module[
    target: StaticString, M: Module
](
    mut m: M,
    wname: String,
    bname: String,
    scale: Scalar[DT],
    ctx: Optional[DeviceContext],
) raises:
    """Scale the output-layer params (`wname`/`bname`) of a Sequential head by
    `scale` (0.0 == exact zero-init)."""
    var v = _ScaleOutVisitor(wname, bname, scale)
    walk_params[target](m, v, ctx)


def scale_output_graph[
    target: StaticString, *DECLS: GraphDecl
](
    mut g: ComputeGraph[*DECLS],
    wname: String,
    bname: String,
    scale: Scalar[DT],
    ctx: Optional[DeviceContext],
) raises:
    """Scale the output-layer params of a head wrapped in a ComputeGraph by
    `scale` (0.0 == exact zero-init). Names are prefixed by the node name
    (e.g. `rew.3.weight` / `rew.3.bias`)."""
    var v = _ScaleOutVisitor(wname, bname, scale)
    walk_params[target](g, v, ctx)
