# +--------------------------------------------------------------------------+ #
# | SuperMimic's object interface: its H-step rollout and policy gradient
# +--------------------------------------------------------------------------+ #
"""The tracker's imagined rollout (`imagination.sm_imagine`) with one free
object beside the robot, and the gradient of `loss = −J / H` with respect to
the OBJECT ENCODER's parameters. `docs/OBJECT_INTERACTION_PLAN.md` §4, step
O1.4.

    for t in 0..H−1:
        z_t      = enc_motion(goal_t)                         frozen, no gradient needed
        e_t      = enc_obj([features(s_t, o_t, o_ref,t), hist_t])
        a_t      = dec([z_t + gain · e_t, hist_t])            frozen, carries the gradient
        dl       = wm.dyn([canonical(s_t), a_t, ξ])           the tracker's dynamics
        dlo      = om.dyn([that input, feat(s_t, o_t), ctx])  the object network (O1.3)
        s_{t+1}  = integrate(s_t, dl + dlo[:DELTA])
        o_{t+1}  = integrate_o(s_t, o_t, dlo[DELTA:])
        r_{t+1}  = the tracker's eight terms + the object's three (`object_task`)
        d̂_{t+1}  = σ(term_head(features))                    c_{t+1} = c_t (1 − d̂_{t+1})
    J = mean_n [ Σ_t γᵗ c_t r_{t+1} + γᴴ c_H V(s_H, o_H) ]

The reverse scan is `sm_imagine`'s with a second carried cotangent, ∂/∂o_{t+1},
and `object_model.sm_obj_window_loss`'s steps through the object network. Both
dynamics networks, the decoder, the termination head and the critic are
frozen HERE: their `vjp`s leave gradients in their own grads, which the caller
never applies. The motion encoder is not differentiated at all (its input is
data).

⚠ Each step's `forward` is re-run before its `vjp` (`world_model.mojo`'s note).
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.config import SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.imagination import SMImagBatch, SMImagOut
from noeira.deep_agents.supermimic.lanes import (
    SM_TERM_IN, SMImagScratch, sm_copy, sm_dec_back, sm_dec_in, sm_enc_in, sm_imag_chain, sm_imag_init, sm_imag_j,
    sm_imag_surv, sm_imag_wm_back, sm_wm_in, talloc, tp,
)
from noeira.deep_agents.supermimic.object_imag_lanes import (
    SMObjImagDims, sm_oc_in, sm_oc_v_back, sm_oe_back, sm_oe_in, sm_oe_mix, sm_oe_scale, sm_oi_heads_back, sm_oi_step,
)
from noeira.deep_agents.supermimic.object_lanes import SMObjDims, sm_obj_dyn_back, sm_obj_feat_back, sm_obj_in
from noeira.deep_agents.supermimic.object_policy import SMObjConfigT, SMObjCriticOf, SMObjEncOf, SMObjModelOf
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX
from noeira.deep_agents.supermimic.object_task import SM_OBJ_TERMS, SM_OE_SCRATCH, SM_OREF, SM_ORW
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


struct SMObjImagBatch[R: TrackingRobot](Movable):
    """The tracker's start batch, plus the object's: obj0 [n·13] its state at
    the start, octx [n·2] (mass, friction), orefs [(H+1)·n·19] its reference
    rows for the frames of `base.refs` (t-major), and rw [6] the reward's data
    (`object_task.sm_obj_reward_params`)."""
    var base: SMImagBatch[Self.R]
    var obj0: TensorImpl[DT]
    var octx: TensorImpl[DT]
    var orefs: TensorImpl[DT]
    var rw: TensorImpl[DT]

    def __init__(out self, n: Int, h: Int):
        self.base = SMImagBatch[Self.R](n, h)
        self.obj0 = TensorImpl[DT].alloc(n * SM_OBJ)
        self.octx = TensorImpl[DT].alloc(n * SM_OBJ_CTX)
        self.orefs = TensorImpl[DT].alloc((h + 1) * n * SM_OREF)
        self.rw = TensorImpl[DT].alloc(SM_ORW)

    def __init__(out self, *, deinit move: Self):
        self.base = move.base^
        self.obj0 = move.obj0^
        self.octx = move.octx^
        self.orefs = move.orefs^
        self.rw = move.rw^

    def to_device(mut self, ctx: DeviceContext) raises:
        """Copy the host fields into their device buffers (kept across calls)."""
        self.base.to_device(ctx)
        self.obj0.upload_resident(ctx)
        self.octx.upload_resident(ctx)
        self.orefs.upload_resident(ctx)
        self.rw.upload_resident(ctx)


struct SMObjImagOut[R: TrackingRobot](Movable):
    """The tracker's rollout output, plus objs [(H+1)·n·13] the object's
    imagined states and oterms [H·n·3] its reward terms (already inside
    `base.rewards`), t-major, on the rollout's target."""
    var base: SMImagOut[Self.R]
    var objs: TensorImpl[DT]
    var oterms: TensorImpl[DT]

    def __init__(out self, var base: SMImagOut[Self.R], var objs: TensorImpl[DT], var oterms: TensorImpl[DT]):
        self.base = base^
        self.objs = objs^
        self.oterms = oterms^

    def __init__(out self, *, deinit move: Self):
        self.base = move.base^
        self.objs = move.objs^
        self.oterms = move.oterms^

    def to_host(mut self, ctx: DeviceContext) raises:
        self.base.to_host(ctx)
        self.objs.download(ctx)
        self.oterms.download(ctx)


def sm_imagine_obj[OC: SMObjConfigT, N: Int, target: StaticString = "cpu"](
    mut wm: SMWorldModelOf[OC.BASE],
    mut pol: SMPolicyOf[OC.BASE],
    mut om: SMObjModelOf[OC],
    mut enc: SMObjEncOf[OC],
    mut critic: SMObjCriticOf[OC],
    mut tb: SMTables[DT],
    mut ib: SMObjImagBatch[OC.BASE.ROBOT],
    gamma: Float64,
    gain: Float64,
    grad: Bool,
    noise_seed: UInt32 = 0,
    ctx: Optional[DeviceContext] = None,
) raises -> SMObjImagOut[OC.BASE.ROBOT]:
    """The rollout; with `grad`, ADDS d(−J/H)/d params into the OBJECT
    ENCODER's grads (zero them first). Every other network's grads are
    polluted. `noise_seed` as `sm_imagine`'s."""
    comptime R = OC.BASE.ROBOT
    comptime D = SMDims[R]
    comptime OD = SMObjDims[R]
    comptime ID = SMObjImagDims[R]
    comptime n = R.NDOF
    comptime S = D.S
    comptime OBS = D.OBS
    comptime HIST = D.HIST
    comptime LAT = D.LATENT
    comptime SC = SMImagScratch[R]
    if ib.base.n != N:
        raise Error("sm_imagine_obj: " + String(ib.base.n) + " lanes, compiled for " + String(N))
    var H = ib.base.h
    var o: SMImagOut[R]
    comptime if target == "gpu":
        o = SMImagOut[R](N, H, ctx)
    else:
        o = SMImagOut[R](N, H)
    var objs = talloc[target]((H + 1) * N * SM_OBJ, ctx)
    var oterms = talloc[target](H * N * SM_OBJ_TERMS, ctx)
    sm_copy[target](ctx, objs, 0, ib.obj0, 0, N * SM_OBJ)
    var O = talloc[target]((HIST + H) * N * OBS, ctx)             # observation slots
    sm_copy[target](ctx, O, 0, ib.base.hist0, 0, HIST * N * OBS)
    var scratch = talloc[target](N * SC.SIZE, ctx)
    var tree = tb.tree_p[target]()
    var table = tb.head_p[target]()
    var g = Scalar[DT](gain)
    sm_imag_init[target, R](
        ctx, N, tp[target](ib.base.state), tp[target](o.states), tp[target](ib.base.refs), tp[target](o.e_root), tp[target](o.survival)
    )

    var eoins = List[TensorImpl[DT]]()
    var decs = List[TensorImpl[DT]]()
    var xs = List[TensorImpl[DT]]()
    var xos = List[TensorImpl[DT]]()
    var dls = List[TensorImpl[DT]]()          # the robot's TOTAL delta of step t
    var dlos = List[TensorImpl[DT]]()         # the object network's output
    var tins = List[TensorImpl[DT]]()
    var ein = talloc[target](N * D.GOAL, ctx)
    var z = talloc[target](N * LAT, ctx)
    var e = talloc[target](N * LAT, ctx)
    var act = talloc[target](N * n, ctx)
    var lg = talloc[target](N, ctx)
    for t in range(H):
        var st = tp[target](o.states).unsafe_offset(t * N * S)
        var ot = tp[target](objs).unsafe_offset(t * N * SM_OBJ)
        var oreft = tp[target](ib.orefs).unsafe_offset(t * N * SM_OREF)
        var Ot = tp[target](O).unsafe_offset(t * N * OBS)
        # the policy: the frozen motion encoder, the object encoder, the frozen decoder
        sm_enc_in[target, R](ctx, N, tp[target](ib.base.goals).unsafe_offset(t * N * D.GOAL), tp[target](ein))
        pol.enc.forward[target, N](TensorRefs[1, _, DT](ein), z, ctx)
        var eoin = talloc[target](N * ID.EO_IN, ctx)
        sm_oe_in[target, R](ctx, N, st, ot, oreft, Ot, N * OBS, OBS, tp[target](eoin))
        enc.forward[target, N](TensorRefs[1, _, DT](eoin), e, ctx)
        sm_oe_mix[target](ctx, N * LAT, g, tp[target](e), tp[target](z))
        var din = talloc[target](N * D.DEC_IN, ctx)
        sm_dec_in[target, R](ctx, N, tp[target](z), Ot, N * OBS, OBS, tp[target](din))
        pol.dec.forward[target, N](TensorRefs[1, _, DT](din), act, ctx)
        # the world model: the tracker's dynamics and the object network
        var x = talloc[target](N * D.WM_IN, ctx)
        sm_wm_in[target, R](ctx, N, st, tp[target](act), tp[target](ib.base.xi), tp[target](x))
        var dl = talloc[target](N * D.DELTA, ctx)
        wm.dyn.forward[target, N](TensorRefs[1, _, DT](x), dl, ctx)
        var xo = talloc[target](N * OD.IN, ctx)
        sm_obj_in[target, R](ctx, N, tp[target](x), st, ot, tp[target](ib.octx), tp[target](xo))
        var dlo = talloc[target](N * OD.OUT, ctx)
        om.dyn.forward[target, N](TensorRefs[1, _, DT](xo), dlo, ctx)
        var tin = talloc[target](N * SM_TERM_IN, ctx)
        sm_oi_step[target, R](
            ctx, t, N, noise_seed, tree, table, tp[target](o.states), tp[target](dl), tp[target](dlo), tp[target](act),
            tp[target](o.actions), tp[target](ib.base.prev_action), tp[target](ib.base.xi), tp[target](ib.base.refs), tp[target](O),
            tp[target](scratch), tp[target](o.rewards), tp[target](tin), tp[target](o.e_root),
            tp[target](objs), tp[target](ib.orefs), tp[target](ib.rw), tp[target](oterms),
        )
        wm.term.forward[target, N](TensorRefs[1, _, DT](tin), lg, ctx)
        sm_imag_surv[target](ctx, t, N, tp[target](lg), tp[target](o.dhat), tp[target](o.survival))
        eoins.append(eoin^)
        decs.append(din^)
        xs.append(x^)
        xos.append(xo^)
        dls.append(dl^)
        dlos.append(dlo^)
        tins.append(tin^)

    # the interface's critic at (s_H, o_H)
    var cin = talloc[target](N * ID.OC_IN, ctx)
    sm_oc_in[target, R](
        ctx, N, N, tp[target](o.states).unsafe_offset(H * N * S), tp[target](ib.base.goals).unsafe_offset(H * N * D.GOAL),
        tp[target](ib.base.xi), tp[target](o.e_root).unsafe_offset(H * N * D.EROOT), tp[target](objs).unsafe_offset(H * N * SM_OBJ),
        tp[target](ib.orefs).unsafe_offset(H * N * SM_OREF), tp[target](ib.octx), tp[target](cin),
    )
    critic.forward[target, N](TensorRefs[1, _, DT](cin), o.value_h, ctx)
    var jb = talloc[target](2 * N, ctx)
    sm_imag_j[target](ctx, N, H, Scalar[DT](gamma), tp[target](o.survival), tp[target](o.rewards), tp[target](o.value_h), tp[target](jb))
    comptime if target == "gpu":
        jb.download(ctx.value())
    var jsum = 0.0
    var usum = 0.0
    for b in range(N):
        jsum += Float64(jb.data[b])
        usum += Float64(jb.data[N + b])
    o.j = jsum / Float64(N)
    o.return_undisc = usum / Float64(N)
    if not grad:
        return SMObjImagOut[R](o^, objs^, oterms^)

    # ── the survival chain, closed form: g_r[t], g_logit[t], g_V ──
    var g_r = talloc[target](H * N, ctx)
    var g_lg = talloc[target](H * N, ctx)
    var gv = talloc[target](N, ctx)
    sm_imag_chain[target](
        ctx, N, H, Scalar[DT](gamma), Scalar[DT](-1.0 / (Float64(H) * Float64(N))), tp[target](o.value_h),
        tp[target](o.survival), tp[target](o.dhat), tp[target](o.rewards), tp[target](g_r), tp[target](g_lg), tp[target](gv),
    )

    # ── V(s_H, o_H) ──
    var gs = talloc[target](N * S, ctx)              # ∂/∂s_{t+1}
    var go = talloc[target](N * SM_OBJ, ctx)         # ∂/∂o_{t+1}
    var oscr = talloc[target](N * SM_OE_SCRATCH, ctx)
    var vre = talloc[target](N, ctx)
    critic.forward[target, N](TensorRefs[1, _, DT](cin), vre, ctx)
    var gcin = talloc[target](N * ID.OC_IN, ctx)
    critic.vjp[target, N](TensorRefs[1, _, DT](cin), gv, TensorRefs[1, _, DT](gcin), ctx)
    sm_oc_v_back[target, R](
        ctx, N, tp[target](o.states).unsafe_offset(H * N * S), tp[target](objs).unsafe_offset(H * N * SM_OBJ),
        tp[target](ib.base.refs).unsafe_offset(H * N * D.REF), tp[target](ib.orefs).unsafe_offset(H * N * SM_OREF),
        tp[target](gcin), tp[target](oscr), tp[target](gs), tp[target](go),
    )

    var gO = talloc[target]((HIST + H) * N * OBS, ctx)
    var gap = talloc[target](N * n, ctx)            # ∂/∂a_t from step t+1
    var ga = talloc[target](N * n, ctx)             # ∂/∂a_t, all paths
    var gprev = talloc[target](N * S, ctx)
    var goprev = talloc[target](N * SM_OBJ, ctx)
    var gdl = talloc[target](N * D.DELTA, ctx)
    var gdlo = talloc[target](N * OD.OUT, ctx)
    var gx = talloc[target](N * D.WM_IN, ctx)
    var gxo = talloc[target](N * OD.IN, ctx)
    var glg = talloc[target](N, ctx)
    var gtin = talloc[target](N * SM_TERM_IN, ctx)
    var gdin = talloc[target](N * D.DEC_IN, ctx)
    var gz = talloc[target](N * LAT, ctx)
    var ge = talloc[target](N * LAT, ctx)
    var geoin = talloc[target](N * ID.EO_IN, ctx)
    var relg = talloc[target](N, ctx)
    var redl = talloc[target](N * D.DELTA, ctx)
    var redlo = talloc[target](N * OD.OUT, ctx)
    var rea = talloc[target](N * n, ctx)
    var ree = talloc[target](N * LAT, ctx)
    var t = H - 1
    while t >= 0:
        var st = tp[target](o.states).unsafe_offset(t * N * S)
        var ot = tp[target](objs).unsafe_offset(t * N * SM_OBJ)
        var gOt = tp[target](gO).unsafe_offset(t * N * OBS)
        # termination head at step t (re-run its forward first)
        wm.term.forward[target, N](TensorRefs[1, _, DT](tins[t]), relg, ctx)
        sm_copy[target](ctx, glg, 0, g_lg, t * N, N)
        wm.term.vjp[target, N](TensorRefs[1, _, DT](tins[t]), glg, TensorRefs[1, _, DT](gtin), ctx)
        # the heads and the object's terms at (s_{t+1}, o_{t+1}), then the robot's integrator
        sm_oi_heads_back[target, R](
            ctx, t, N, tree, table, tp[target](o.states), tp[target](o.actions), tp[target](ib.base.prev_action),
            tp[target](ib.base.xi), tp[target](ib.base.refs), tp[target](tins[t]), tp[target](gtin), tp[target](g_r), tp[target](gO),
            tp[target](scratch), tp[target](gs), tp[target](ga), tp[target](gap), tp[target](dls[t]), tp[target](gprev), tp[target](gdl),
            tp[target](objs), tp[target](ib.orefs), tp[target](ib.rw), tp[target](go),
        )
        # the object: o_{t+1} = integrate_o(s_t, o_t, dlo[DELTA:]); gprev gets the heading's share
        sm_obj_dyn_back[target, R](
            ctx, N, st, ot, tp[target](dlos[t]), tp[target](go), tp[target](gprev), tp[target](goprev), tp[target](gdl), tp[target](gdlo),
        )
        om.dyn.forward[target, N](TensorRefs[1, _, DT](xos[t]), redlo, ctx)
        om.dyn.vjp[target, N](TensorRefs[1, _, DT](xos[t]), gdlo, TensorRefs[1, _, DT](gxo), ctx)
        wm.dyn.forward[target, N](TensorRefs[1, _, DT](xs[t]), redl, ctx)
        wm.dyn.vjp[target, N](TensorRefs[1, _, DT](xs[t]), gdl, TensorRefs[1, _, DT](gx), ctx)
        # the two networks' inputs: gx += gxo[:WM_IN]; go = ∂/∂o_t; ga += ∂/∂a_t; gs = ∂/∂s_t
        sm_obj_feat_back[target, R](ctx, N, st, ot, tp[target](gxo), tp[target](gx), tp[target](gprev), tp[target](goprev), tp[target](go), t > 0)
        sm_imag_wm_back[target, R](ctx, N, t > 0, st, tp[target](gx), tp[target](gprev), tp[target](gs), tp[target](ga))
        # the frozen decoder at step t, then the object encoder it reads through the mix
        pol.dec.forward[target, N](TensorRefs[1, _, DT](decs[t]), rea, ctx)
        pol.dec.vjp[target, N](TensorRefs[1, _, DT](decs[t]), ga, TensorRefs[1, _, DT](gdin), ctx)
        sm_dec_back[target, R](ctx, N, tp[target](gdin), tp[target](gz), gOt, N * OBS, OBS)
        sm_oe_scale[target](ctx, N * LAT, g, tp[target](gz), tp[target](ge))
        enc.forward[target, N](TensorRefs[1, _, DT](eoins[t]), ree, ctx)
        enc.vjp[target, N](TensorRefs[1, _, DT](eoins[t]), ge, TensorRefs[1, _, DT](geoin), ctx)
        sm_oe_back[target, R](
            ctx, N, t > 0, st, ot, tp[target](ib.orefs).unsafe_offset(t * N * SM_OREF), tp[target](geoin), tp[target](oscr),
            gOt, N * OBS, OBS, tp[target](gs), tp[target](go),
        )
        t -= 1
    return SMObjImagOut[R](o^, objs^, oterms^)
