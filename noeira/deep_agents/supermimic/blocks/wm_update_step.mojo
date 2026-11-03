"""WorldModelUpdateStep — one Eq. 1 step on a batch of k-step windows.

`WorldModelTrainer.update`'s inner step: zero the grads, the k-step loss and
its reverse scan (`world_model.sm_wm_window_loss`), `clip_grad_norm(1.0)`
over dynamics + termination head together, Adam(lr).
"""

from noeira.deep_agents.supermimic.config import SMConfigT, SMWorldModelOf
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss, sm_wm_window_loss
from noeira.nn.constants import DT
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm


comptime SM_WM_LR: Float64 = 2e-4
comptime SM_WM_CLIP: Float64 = 1.0


struct WorldModelUpdateStep[C: SMConfigT](Movable):
    var opt: Adam

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](SM_WM_LR))

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^

    def step[target: StaticString, B: Int](
        mut self, mut wm: SMWorldModelOf[Self.C], mut w: SMWindowBatch[Self.C.ROBOT]
    ) raises -> Tuple[SMWMLoss, Float64]:
        """Returns the loss terms and the pre-clip gradient norm."""
        comptime if target == "gpu":
            raise Error("SuperMimic WorldModelUpdateStep: the GPU path is not built yet")
        wm.zero_grad["cpu"](None)
        var l = sm_wm_window_loss[Self.C.ROBOT, Self.C.WM_H, Self.C.TERM_H, B](wm, w, True)
        var gn = clip_grad_norm["cpu"](wm, Scalar[DT](SM_WM_CLIP), None)
        self.opt.step["cpu"](wm, None)
        return (l^, Float64(gn))
