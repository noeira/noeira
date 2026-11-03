"""ActStep — the policy over the simulator lanes.

`a = decoder([tanh(encoder(goal ⊙ gscale)), hist ⊙ oscale])` for L lanes,
from each lane's 10-step observation history (oldest first, obs order) and
goal window. Inputs and output are `nn` tensors, so the GPU path is the same
two forwards on device buffers plus the two scale/flatten kernels.
"""

from noeira.deep_agents.supermimic.config import SMConfigT, SMPolicyOf
from noeira.deep_agents.supermimic.policy import sm_goal_scale, sm_hist_flat_index, sm_obs_scale
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
        mut self, mut pol: SMPolicyOf[Self.C], hist: TensorImpl[DT], goals: TensorImpl[DT]
    ) raises -> TensorImpl[DT]:
        """`hist` [L · 10 · OBS], `goals` [L · GOAL] → actions [L · n]."""
        comptime if target == "gpu":
            raise Error("SuperMimic ActStep: the GPU path is not built yet")
        comptime R = Self.C.ROBOT
        comptime D = SMDims[R]
        var ein = TensorImpl[DT].alloc(L * D.GOAL)
        for i in range(L):
            for k in range(D.GOAL):
                ein.data[i * D.GOAL + k] = goals.data[i * D.GOAL + k] * Scalar[DT](sm_goal_scale[R.NDOF](k))
        var z = TensorImpl[DT].alloc(L * D.LATENT)
        pol.enc.forward["cpu", L](TensorRefs[1, _, DT](ein), z, None)
        var din = TensorImpl[DT].alloc(L * D.DEC_IN)
        for i in range(L):
            for k in range(D.LATENT):
                din.data[i * D.DEC_IN + k] = z.data[i * D.LATENT + k]
            for hh in range(D.HIST):
                for k in range(D.OBS):
                    din.data[i * D.DEC_IN + D.LATENT + sm_hist_flat_index[R.NDOF](hh, k)] = (
                        hist.data[(i * D.HIST + hh) * D.OBS + k] * Scalar[DT](sm_obs_scale[R.NDOF](k))
                    )
        var a = TensorImpl[DT].alloc(L * R.NDOF)
        pol.dec.forward["cpu", L](TensorRefs[1, _, DT](din), a, None)
        return a^
