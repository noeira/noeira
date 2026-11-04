"""ActStep — the policy over the simulator lanes.

`a = decoder([tanh(encoder(goal ⊙ gscale)), hist ⊙ oscale])` for L lanes,
from each lane's 10-step observation history (oldest first, obs order) and
goal window. Inputs and output are `nn` tensors on `target` (device buffers
on "gpu"); the scale/flatten steps are `lanes.mojo` phases.
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.config import SMConfigT, SMPolicyOf
from noeira.deep_agents.supermimic.lanes import sm_dec_in, sm_enc_in, talloc, tp
from noeira.deep_agents.supermimic.robot import SMDims
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


struct ActStep[C: SMConfigT](Movable):
    def __init__(out self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    def step[target: StaticString, L: Int](
        mut self, mut pol: SMPolicyOf[Self.C], mut hist: TensorImpl[DT], mut goals: TensorImpl[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> TensorImpl[DT]:
        """`hist` [L · 10 · OBS], `goals` [L · GOAL] → actions [L · n]."""
        comptime R = Self.C.ROBOT
        comptime D = SMDims[R]
        var ein = talloc[target](L * D.GOAL, ctx)
        sm_enc_in[target, R](ctx, L, tp[target](goals), tp[target](ein))
        var z = talloc[target](L * D.LATENT, ctx)
        pol.enc.forward[target, L](TensorRefs[1, _, DT](ein), z, ctx)
        var din = talloc[target](L * D.DEC_IN, ctx)
        sm_dec_in[target, R](ctx, L, tp[target](z), tp[target](hist), D.OBS, D.HIST * D.OBS, tp[target](din))
        var a = talloc[target](L * R.NDOF, ctx)
        pol.dec.forward[target, L](TensorRefs[1, _, DT](din), a, ctx)
        return a^
