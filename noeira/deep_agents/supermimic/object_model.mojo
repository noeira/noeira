# +--------------------------------------------------------------------------+ #
# | SuperMimic with an object: the object network and the joint k-step loss
# +--------------------------------------------------------------------------+ #
"""The world model of a robot AND one free object, as an addition to the
tracker's own (`world_model.mojo`), which is used unchanged.
`docs/OBJECT_INTERACTION_PLAN.md` §4.2, step O1.3.

    one step:  x  = [canonical(s), a, normalise(ξ)]                (the robot's input)
               xo = [x, feat(s, o), normalise(ctx)]                (OBJ_IN = WM_IN + 17)
               dl  = dynamics(x)                                   (the tracker's, DELTA)
               dlo = object(xo)        (MLP OBJ_IN → HO → HO → HO → DELTA + 12, SiLU,
                                        last layer zero-initialised)
               s' = integrate(s, dl + dlo[:DELTA])                 the robot, corrected
               o' = integrate_o(s, o, dlo[DELTA:])                 the object
    the head:  the tracker's termination head, on s'

    loss = (1/k) Σ_j mean_b [ Σ e_j² + w_obj Σ eo_j² ] + w_term (1/k) Σ_j mean_b BCE

open loop from the logged (s_0, o_0), e_j and eo_j the errors of the predicted
(s_{j+1}, o_{j+1}) against the logged ones: Eq. 1 with the object's terms.

WHY A SECOND NETWORK and not a wider one. The object network's last layer
starts at zero, so at the start the robot's prediction is the tracker's world
model's EXACTLY and the object moves at constant velocity (an object at rest
stays at rest): the plan's "zero-started" extension, without editing the
tracker's weights. The tracker's checkpoint loads as it is; its dynamics can
stay frozen or keep fitting, the caller's choice (this function adds both
models' gradients; which optimiser steps is decided outside). The first
DELTA outputs are how carrying a mass changes the ROBOT's dynamics.

The gradient flows through the whole chain: the object's pose reaches later
robot steps through the object network's input, and the robot's heading
reaches the object through its features and its integration.

Param names: `object.<i>.weight|bias`; weights `[IN, OUT]` row-major.
"""

from std.math import exp, log

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.lanes import (
    SM_TERM_IN, sm_canon_back, sm_wm_dyn_back, sm_wm_err_back, sm_wm_in, sm_wm_step, talloc, tp,
)
from noeira.deep_agents.supermimic.object_lanes import (
    SMObjDims, sm_obj_dyn_back, sm_obj_err_back, sm_obj_feat_back, sm_obj_in, sm_obj_step,
)
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX, SM_OBJ_DELTA
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWorldModel
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.param import ParamVisitor, ParamWalkable
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.walkers import join_name
from noeira.nn.primitives.linear_swish import LinearSwish
from noeira.nn.primitives.zero_linear import ZeroLinear


comptime SMObjDynamics[R: TrackingRobot, HO: Int] = Sequential[
    LinearSwish[SMObjDims[R].IN, HO], LinearSwish[HO, HO], LinearSwish[HO, HO], ZeroLinear[HO, SMObjDims[R].OUT],
]


struct SMObjectModel[R: TrackingRobot, HO: Int](ParamWalkable):
    """The object network, walked on its own (its optimiser and its clip are
    its own, so the tracker's world model can stay frozen beside it)."""
    var dyn: SMObjDynamics[Self.R, Self.HO]

    def __init__(out self, var dyn: SMObjDynamics[Self.R, Self.HO]):
        self.dyn = dyn^

    def __init__(out self, *, deinit move: Self):
        self.dyn = move.dyn^

    @staticmethod
    def make[INIT: Initializer, target: StaticString = "cpu"](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self(SMObjDynamics[Self.R, Self.HO].make[target, INIT](ctx))

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext], prefix: String = String("")
    ) raises:
        self.dyn.for_each_param[target](visitor, ctx, join_name(prefix, String("object")))

    def zero_grad[target: StaticString](mut self, ctx: Optional[DeviceContext]) raises:
        self.dyn.zero_grad[target](ctx)

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext], prefix: String = String("")
    ) raises:
        self.dyn.for_each_state[target](visitor, ctx, join_name(prefix, String("object")))


struct SMObjWindowBatch(Movable):
    """The object's side of `SMWindowBatch`, t-major, for the same k and B:
    states [(k+1)·B·13] logged object states, ctx [B·2] (mass, friction)."""
    var k: Int
    var b: Int
    var states: TensorImpl[DT]
    var ctx: TensorImpl[DT]

    def __init__(out self, k: Int, b: Int):
        self.k = k
        self.b = b
        self.states = TensorImpl[DT].alloc((k + 1) * b * SM_OBJ)
        self.ctx = TensorImpl[DT].alloc(b * SM_OBJ_CTX)

    def __init__(out self, *, deinit move: Self):
        self.k = move.k
        self.b = move.b
        self.states = move.states^
        self.ctx = move.ctx^

    def to_device(mut self, ctx: DeviceContext) raises:
        self.states.upload_resident(ctx)
        self.ctx.upload_resident(ctx)


@fieldwise_init
struct SMObjLoss(Copyable, Movable):
    var loss: Float64            # state_loss + w_obj · obj_loss + w_term · term_bce
    var state_loss: Float64      # the robot's, as `SMWMLoss.state_loss`
    var term_bce: Float64
    var obj_loss: Float64        # (1/k) Σ_j mean_b Σ eo²
    var nse_1step: Float64       # the robot's open-loop error, first and last step
    var nse_kstep: Float64
    var obj_nse_1step: Float64   # the object's: mean of eo² over its 12 entries
    var obj_nse_kstep: Float64

    def add_scaled(mut self, l: SMObjLoss, w: Float64):
        self.loss += w * l.loss
        self.state_loss += w * l.state_loss
        self.term_bce += w * l.term_bce
        self.obj_loss += w * l.obj_loss
        self.nse_1step += w * l.nse_1step
        self.nse_kstep += w * l.nse_kstep
        self.obj_nse_1step += w * l.obj_nse_1step
        self.obj_nse_kstep += w * l.obj_nse_kstep


def sm_obj_window_loss[R: TrackingRobot, H: Int, TH: Int, HO: Int, B: Int, target: StaticString = "cpu"](
    mut wm: SMWorldModel[R, H, TH], mut om: SMObjectModel[R, HO], mut w: SMWindowBatch[R], mut wo: SMObjWindowBatch,
    grad: Bool, w_term: Float64 = 1.0, w_obj: Float64 = 1.0, ctx: Optional[DeviceContext] = None,
) raises -> SMObjLoss:
    """The joint loss on one batch of windows. With `grad`, ADDS d loss / d
    params into BOTH models' grads (call both `zero_grad`s first). Targets as
    `sm_wm_window_loss`: on "gpu" both batches are on the device and both
    models made there."""
    comptime D = SMDims[R]
    comptime OD = SMObjDims[R]
    comptime S = D.S
    comptime n = R.NDOF
    if w.b != B or wo.b != B:
        raise Error("sm_obj_window_loss: batches of " + String(w.b) + " and " + String(wo.b) + " lanes, compiled for " + String(B))
    if w.k != wo.k:
        raise Error("sm_obj_window_loss: windows of " + String(w.k) + " and " + String(wo.k) + " steps")
    var K = w.k
    var sp = talloc[target]((K + 1) * B * S, ctx)                  # the predicted robot chain
    var op = talloc[target]((K + 1) * B * SM_OBJ, ctx)             # the predicted object chain
    var errs = talloc[target](K * B * D.DELTA, ctx)
    var errso = talloc[target](K * B * SM_OBJ_DELTA, ctx)
    var sq = talloc[target](K * B, ctx)
    var sqo = talloc[target](K * B, ctx)
    var lgall = talloc[target](K * B, ctx)
    comptime if target == "gpu":
        ctx.value().enqueue_copy(sp.dev.value().create_sub_buffer[DT](0, B * S), w.states.dev.value().create_sub_buffer[DT](0, B * S))
        ctx.value().enqueue_copy(op.dev.value().create_sub_buffer[DT](0, B * SM_OBJ), wo.states.dev.value().create_sub_buffer[DT](0, B * SM_OBJ))
    else:
        for i in range(B * S):
            sp.data[i] = w.states.data[i]
        for i in range(B * SM_OBJ):
            op.data[i] = wo.states.data[i]
    var xs = List[TensorImpl[DT]]()
    var xos = List[TensorImpl[DT]]()
    var ds = List[TensorImpl[DT]]()          # the robot's TOTAL delta of step j
    var dos = List[TensorImpl[DT]]()         # the object network's output
    var tins = List[TensorImpl[DT]]()
    for j in range(K):
        var sj = tp[target](sp).unsafe_offset(j * B * S)
        var oj = tp[target](op).unsafe_offset(j * B * SM_OBJ)
        var x = talloc[target](B * D.WM_IN, ctx)
        sm_wm_in[target, R](ctx, B, sj, tp[target](w.a_exec).unsafe_offset(j * B * n), tp[target](w.xi), tp[target](x))
        var dl = talloc[target](B * D.DELTA, ctx)
        wm.dyn.forward[target, B](TensorRefs[1, _, DT](x), dl, ctx)
        var xo = talloc[target](B * OD.IN, ctx)
        sm_obj_in[target, R](ctx, B, tp[target](x), sj, oj, tp[target](wo.ctx), tp[target](xo))
        var dlo = talloc[target](B * OD.OUT, ctx)
        om.dyn.forward[target, B](TensorRefs[1, _, DT](xo), dlo, ctx)
        sm_obj_step[target, R](
            ctx, B, sj, oj, tp[target](dlo), tp[target](dl), tp[target](op).unsafe_offset((j + 1) * B * SM_OBJ),
            tp[target](wo.states).unsafe_offset((j + 1) * B * SM_OBJ), tp[target](errso).unsafe_offset(j * B * SM_OBJ_DELTA),
            tp[target](sqo).unsafe_offset(j * B),
        )
        var tin = talloc[target](B * SM_TERM_IN, ctx)
        sm_wm_step[target, R](
            ctx, B, sj, tp[target](dl), tp[target](sp).unsafe_offset((j + 1) * B * S),
            tp[target](w.states).unsafe_offset((j + 1) * B * S), tp[target](w.ref_root).unsafe_offset(j * B * 7),
            tp[target](errs).unsafe_offset(j * B * D.DELTA), tp[target](tin), tp[target](sq).unsafe_offset(j * B),
        )
        var lg = talloc[target](B, ctx)
        wm.term.forward[target, B](TensorRefs[1, _, DT](tin), lg, ctx)
        comptime if target == "gpu":
            ctx.value().enqueue_copy(lgall.dev.value().create_sub_buffer[DT](j * B, B), lg.dev.value())
        else:
            for b in range(B):
                lgall.data[j * B + b] = lg.data[b]
        xs.append(x^)
        xos.append(xo^)
        ds.append(dl^)
        dos.append(dlo^)
        tins.append(tin^)
    # the scalars, on the host
    comptime if target == "gpu":
        sq.download_enqueue(ctx.value())
        sqo.download_enqueue(ctx.value())
        lgall.download_enqueue(ctx.value())
        w.done.download_enqueue(ctx.value())
        ctx.value().synchronize()
        sq.download_finalize()
        sqo.download_finalize()
        lgall.download_finalize()
        w.done.download_finalize()
    var state_tot = 0.0
    var obj_tot = 0.0
    var bce_tot = 0.0
    var nse1 = 0.0
    var nsek = 0.0
    var onse1 = 0.0
    var onsek = 0.0
    for j in range(K):
        var nse = 0.0
        var onse = 0.0
        for b in range(B):
            var v = Float64(sq.data[j * B + b])
            var vo = Float64(sqo.data[j * B + b])
            state_tot += v / Float64(B)
            obj_tot += vo / Float64(B)
            nse += v / Float64(D.DELTA * B)
            onse += vo / Float64(SM_OBJ_DELTA * B)
            var l = Float64(lgall.data[j * B + b])
            var y = Float64(w.done.data[j * B + b])
            bce_tot += (max(l, 0.0) - l * y + log(1.0 + exp(-abs(l)))) / Float64(B)
        if j == 0:
            nse1 = nse
            onse1 = onse
        nsek = nse
        onsek = onse
    var fk = Float64(K)
    var out = SMObjLoss(
        state_tot / fk + w_obj * obj_tot / fk + w_term * bce_tot / fk, state_tot / fk, bce_tot / fk, obj_tot / fk,
        nse1, nsek, onse1, onsek,
    )
    if not grad:
        return out^

    # ── the reverse scan ──
    var gs = talloc[target](B * S, ctx)                 # d loss / d s_{j+1}
    var go = talloc[target](B * SM_OBJ, ctx)            # d loss / d o_{j+1}
    var gprev = talloc[target](B * S, ctx)
    var goprev = talloc[target](B * SM_OBJ, ctx)
    var glg = talloc[target](B, ctx)
    var gtin = talloc[target](B * SM_TERM_IN, ctx)
    var gdl = talloc[target](B * D.DELTA, ctx)
    var gdlo = talloc[target](B * OD.OUT, ctx)
    var gx = talloc[target](B * D.WM_IN, ctx)
    var gxo = talloc[target](B * OD.IN, ctx)
    var relg = talloc[target](B, ctx)
    var redl = talloc[target](B * D.DELTA, ctx)
    var redlo = talloc[target](B * OD.OUT, ctx)
    var cs = Scalar[DT](2.0 / Float64(K * B))
    var co = Scalar[DT](2.0 * w_obj / Float64(K * B))
    var cb = Scalar[DT](w_term / Float64(K * B))
    var j = K - 1
    while j >= 0:
        var sj = tp[target](sp).unsafe_offset(j * B * S)
        var oj = tp[target](op).unsafe_offset(j * B * SM_OBJ)
        var s1 = tp[target](sp).unsafe_offset((j + 1) * B * S)
        sm_wm_err_back[target, R](
            ctx, B, s1, tp[target](w.states).unsafe_offset((j + 1) * B * S), tp[target](errs).unsafe_offset(j * B * D.DELTA),
            tp[target](lgall).unsafe_offset(j * B), tp[target](w.done).unsafe_offset(j * B), cs, cb, tp[target](gs), tp[target](glg),
        )
        sm_obj_err_back[target](
            ctx, B, tp[target](op).unsafe_offset((j + 1) * B * SM_OBJ), tp[target](wo.states).unsafe_offset((j + 1) * B * SM_OBJ),
            tp[target](errso).unsafe_offset(j * B * SM_OBJ_DELTA), co, tp[target](go),
        )
        # ⚠ every `vjp` below is preceded by a re-run of that step's forward:
        # the fused leaves keep the activations of their LAST forward call
        # (`sm_wm_window_loss`'s note).
        wm.term.forward[target, B](TensorRefs[1, _, DT](tins[j]), relg, ctx)
        wm.term.vjp[target, B](TensorRefs[1, _, DT](tins[j]), glg, TensorRefs[1, _, DT](gtin), ctx)
        # the robot: s_{j+1} = integrate(s_j, dl_total); gprev is WRITTEN here
        sm_wm_dyn_back[target, R](
            ctx, B, tp[target](tins[j]), tp[target](gtin), sj, s1, tp[target](ds[j]), tp[target](gs), tp[target](gprev), tp[target](gdl),
        )
        # the object: o_{j+1} = integrate_o(s_j, o_j, dlo[DELTA:]); gprev gets the heading's share
        sm_obj_dyn_back[target, R](
            ctx, B, sj, oj, tp[target](dos[j]), tp[target](go), tp[target](gprev), tp[target](goprev), tp[target](gdl), tp[target](gdlo),
        )
        om.dyn.forward[target, B](TensorRefs[1, _, DT](xos[j]), redlo, ctx)
        om.dyn.vjp[target, B](TensorRefs[1, _, DT](xos[j]), gdlo, TensorRefs[1, _, DT](gxo), ctx)
        wm.dyn.forward[target, B](TensorRefs[1, _, DT](xs[j]), redl, ctx)
        wm.dyn.vjp[target, B](TensorRefs[1, _, DT](xs[j]), gdl, TensorRefs[1, _, DT](gx), ctx)
        # (s_0, o_0) are data: no feature VJP at j = 0
        sm_obj_feat_back[target, R](
            ctx, B, sj, oj, tp[target](gxo), tp[target](gx), tp[target](gprev), tp[target](goprev), tp[target](go), j > 0,
        )
        sm_canon_back[target, R](ctx, B, D.WM_IN, sj, tp[target](gx), tp[target](gprev), tp[target](gs), j > 0)
        j -= 1
    return out^
