"""ObjectModelUpdateStep — one step of the object network on a batch of k-step
windows, the tracker's world model FROZEN beside it.

`WorldModelUpdateStep`'s shape with the joint loss (`object_model.
sm_obj_window_loss`): zero both models' grads, the loss and its reverse scan,
`clip_grad_norm(1.0)` over the object network alone, Adam(lr) on it alone. The
world model's gradients are computed by the scan and dropped: the tracker's
checkpoint is what the policy was trained through, and it stays that.

On "gpu" both batches must already be on the device.
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.config import SMConfigT, SMWorldModelOf
from noeira.deep_agents.supermimic.object_model import SMObjLoss, SMObjWindowBatch, SMObjectModel, sm_obj_window_loss
from noeira.deep_agents.supermimic.world_model import SMWindowBatch
from noeira.nn.constants import DT
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm


comptime SM_OBJ_CLIP: Float64 = 1.0


struct ObjectModelUpdateStep[C: SMConfigT, HO: Int](Movable):
    """As `WorldModelUpdateStep`: on "gpu" the optimiser adopts the object
    network at the first step and the clip is one fused reduction there."""
    var opt: Adam
    var adopted: Bool
    var w_term: Float64       # the termination BCE's weight (it reaches the object network through the robot's correction)
    var w_obj: Float64        # the object error's weight beside the robot's

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](Self.C.DEF_LR))
        self.adopted = False
        self.w_term = 1.0
        self.w_obj = 1.0

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^
        self.adopted = move.adopted
        self.w_term = move.w_term
        self.w_obj = move.w_obj

    def step[target: StaticString, B: Int](
        mut self, mut wm: SMWorldModelOf[Self.C], mut om: SMObjectModel[Self.C.ROBOT, Self.HO],
        mut w: SMWindowBatch[Self.C.ROBOT], mut wo: SMObjWindowBatch, ctx: Optional[DeviceContext] = None,
    ) raises -> Tuple[SMObjLoss, Float64]:
        """Returns the loss terms and the object network's pre-clip gradient
        norm (on "gpu": −1, the norm stays on the device)."""
        comptime if target == "gpu":
            if not self.adopted:
                self.opt.adopt[target](om, ctx)
                self.adopted = True
        wm.zero_grad[target](ctx)
        om.zero_grad[target](ctx)
        var l = sm_obj_window_loss[Self.C.ROBOT, Self.C.WM_H, Self.C.TERM_H, Self.HO, B, target](
            wm, om, w, wo, True, self.w_term, self.w_obj, ctx
        )
        var gn: Float64
        comptime if target == "gpu":
            self.opt.clip_grads_device[target](om, Scalar[DT](SM_OBJ_CLIP), ctx)
            gn = -1.0
        else:
            gn = Float64(clip_grad_norm[target](om, Scalar[DT](SM_OBJ_CLIP), ctx))
        self.opt.step[target](om, ctx)
        return (l^, gn)
