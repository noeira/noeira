"""WorldModelUpdateStep — one Eq. 1 step on a batch of k-step windows.

`WorldModelTrainer.update`'s inner step: zero the grads, the k-step loss and
its reverse scan (`world_model.sm_wm_window_loss`), `clip_grad_norm(1.0)`
over dynamics + termination head together, Adam(lr). On "gpu" the windows
must already be on the device (`SMWindowBatch.to_device`).
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.config import SMConfigT, SMWorldModelOf
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss, sm_wm_window_loss
from noeira.nn.constants import DT
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm


comptime SM_WM_CLIP: Float64 = 1.0


struct WorldModelUpdateStep[C: SMConfigT](Movable):
    """On "gpu" the optimiser adopts the model into its arena at the first
    step: the global-norm clip is then one fused reduction + scale on the
    device (no per-param D2H, which the profile showed was ~7 % of GPU time
    and most of the host syncs). The norm stays on the device
    (`opt.read_clip_norm` reads it)."""
    var opt: Adam
    var adopted: Bool
    var w_term: Float64       # the termination BCE's weight (Eq. 1: 1)

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](Self.C.DEF_LR))
        self.adopted = False
        self.w_term = 1.0

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^
        self.adopted = move.adopted
        self.w_term = move.w_term

    def step[target: StaticString, B: Int](
        mut self, mut wm: SMWorldModelOf[Self.C], mut w: SMWindowBatch[Self.C.ROBOT], ctx: Optional[DeviceContext] = None
    ) raises -> Tuple[SMWMLoss, Float64]:
        """Returns the loss terms and the pre-clip gradient norm (on "gpu":
        −1, the norm stays on the device)."""
        comptime if target == "gpu":
            if not self.adopted:
                self.opt.adopt[target](wm, ctx)
                self.adopted = True
        wm.zero_grad[target](ctx)
        var l = sm_wm_window_loss[Self.C.ROBOT, Self.C.WM_H, Self.C.TERM_H, B, target](wm, w, True, self.w_term, ctx)
        var gn: Float64
        comptime if target == "gpu":
            self.opt.clip_grads_device[target](wm, Scalar[DT](SM_WM_CLIP), ctx)
            gn = -1.0
        else:
            gn = Float64(clip_grad_norm[target](wm, Scalar[DT](SM_WM_CLIP), ctx))
        self.opt.step[target](wm, ctx)
        return (l^, gn)
