# +--------------------------------------------------------------------------+ #
# | SuperMimic with an object: the per-lane phases of the interface's rollout
# +--------------------------------------------------------------------------+ #
"""The object interface's phases, in `lanes.mojo`'s form (one `_*_lane` body
per phase, a runner that loops on "cpu" and launches one kernel on "gpu").
They sit between the tracker's phases and the object block's
(`object_lanes.mojo`) in `object_imagination.sm_imagine_obj`.

    forward, step t   sm_oe_in     eoin = [features(s_t, o_t, o_ref,t), history ⊙ oscale]
                      sm_oe_mix    l = z + gain · e          (z: the motion latent)
                      sm_oi_step   dl += dlo[:DELTA]; o_{t+1} = integrate(s_t, o_t, dlo[DELTA:]);
                                   then the tracker's step (`lanes._imag_step_lane`) and the
                                   object's three reward terms at (s_{t+1}, o_{t+1})
    the critic        sm_oc_in     [the tracker's critic row, features, normalise(ctx)]
    reverse           sm_oc_v_back      V's input → ∂/∂s_H, ∂/∂o_H
                      sm_oi_heads_back  the heads' VJP at s_{t+1} with the object's terms on
                                        the same FK, then the robot's integrator
                      sm_oe_back        the object encoder's input → ∂/∂history, ∂/∂s_t, ∂/∂o_t

Same kernel-safety rules as `lanes.mojo`.
"""

from std.math import ceildiv

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.heads import sm_heads_lane, sm_heads_vjp_close_lane, sm_heads_vjp_open_lane
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import (
    SM_HEIGHT_THR, SM_TILT_THR, SM_TPB, SMImagScratch, _imag_step_lane, _root_q, _sign,
)
from noeira.deep_agents.supermimic.object_lanes import SMObjDims
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX, sm_obj_ctx_normalize, sm_obj_integrate_lane
from noeira.deep_agents.supermimic.object_task import (
    SM_OBJ_TERMS, SM_OE_FEAT, SM_OE_SCRATCH, SM_OREF, sm_obj_reward_lane, sm_obj_reward_vjp_lane, sm_oe_feat_lane,
    sm_oe_feat_vjp_lane,
)
from noeira.deep_agents.supermimic.policy import (
    sm_goal_scale, sm_hist_flat_index, sm_obs_scale, sm_root_error_vjp_lane,
)
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.state_spec import sm_canonical_lane, sm_canonical_vjp_lane, sm_integrate_vjp_lane, sm_xi_normalize
from noeira.nn.constants import DT


struct SMObjImagDims[R: TrackingRobot]:
    """The interface's widths on robot R."""
    comptime EO_IN = SM_OE_FEAT + SMDims[Self.R].HIST_FLAT                      # the object encoder's input
    comptime OC_IN = SMDims[Self.R].CRITIC_IN + SM_OE_FEAT + SM_OBJ_CTX         # the interface critic's input


@always_inline
def _grid(n: Int) -> Int:
    return ceildiv(n, SM_TPB)


# ── forward ─────────────────────────────────────────────────────────────────
@always_inline
def _oe_in_lane[R: TrackingRobot](b: Int, s: FP[DT], o: FP[DT], oref: FP[DT], O: FP[DT], hstride: Int, bstride: Int, eoin: FP[DT]):
    """eoin[b] = [features(s[b], o[b], oref[b]), the 10-slot history ⊙ oscale,
    term-major]; history slot hh of lane b is O[hh·hstride + b·bstride : +OBS]."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var er = eoin.unsafe_offset(b * SMObjImagDims[R].EO_IN)
    sm_oe_feat_lane[DT, R](s.unsafe_offset(b * D.S), o.unsafe_offset(b * SM_OBJ), oref.unsafe_offset(b * SM_OREF), er)
    for hh in range(D.HIST):
        var ob = hh * hstride + b * bstride
        for k in range(D.OBS):
            er[unsafe_offset=SM_OE_FEAT + sm_hist_flat_index[n](hh, k)] = O[unsafe_offset=ob + k] * sm_obs_scale[n, DT](k)


def _k_oe_in[R: TrackingRobot](nb: Int64, s: FP[DT], o: FP[DT], oref: FP[DT], O: FP[DT], hstride: Int64, bstride: Int64, eoin: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _oe_in_lane[R](b, s, o, oref, O, Int(hstride), Int(bstride), eoin)


def sm_oe_in[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, s: FP[DT], o: FP[DT], oref: FP[DT], O: FP[DT], hstride: Int, bstride: Int, eoin: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_oe_in[R]](
            Int64(nb), s, o, oref, O, Int64(hstride), Int64(bstride), eoin, grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _oe_in_lane[R](b, s, o, oref, O, hstride, bstride, eoin)


def _k_axpy(n: Int64, a: Scalar[DT], x: FP[DT], y: FP[DT]):
    var i = Int(global_idx.x)
    if i < Int(n):
        y[unsafe_offset=i] += a * x[unsafe_offset=i]


def sm_oe_mix[target: StaticString](ctx: Optional[DeviceContext], n: Int, gain: Scalar[DT], e: FP[DT], z: FP[DT]) raises:
    """z[0:n] += gain · e[0:n]: the latent the decoder reads."""
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_axpy](Int64(n), gain, e, z, grid_dim=_grid(n), block_dim=SM_TPB)
    else:
        for i in range(n):
            z[unsafe_offset=i] += gain * e[unsafe_offset=i]


def _k_scale(n: Int64, a: Scalar[DT], x: FP[DT], y: FP[DT]):
    var i = Int(global_idx.x)
    if i < Int(n):
        y[unsafe_offset=i] = a * x[unsafe_offset=i]


def sm_oe_scale[target: StaticString](ctx: Optional[DeviceContext], n: Int, gain: Scalar[DT], x: FP[DT], y: FP[DT]) raises:
    """y[0:n] = gain · x[0:n]: the mix's VJP into the object encoder's output."""
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_scale](Int64(n), gain, x, y, grid_dim=_grid(n), block_dim=SM_TPB)
    else:
        for i in range(n):
            y[unsafe_offset=i] = gain * x[unsafe_offset=i]


@always_inline
def _oi_step_lane[R: TrackingRobot](
    b: Int, t: Int, nb: Int, seed: UInt32,
    tree: FP[DT], table: FP[DT], states: FP[DT], dl: FP[DT], dlo: FP[DT], act: FP[DT], actions: FP[DT], prev: FP[DT],
    xi: FP[DT], refs: FP[DT], O: FP[DT], scratch: FP[DT], rewards: FP[DT], tin: FP[DT], eroot: FP[DT],
    objs: FP[DT], orefs: FP[DT], rw: FP[DT], oterms: FP[DT],
):
    """Step t of lane b with an object. `dl[b]` becomes the robot's TOTAL
    delta (the tracker's plus the object network's correction); the object is
    integrated from (s_t, o_t); then the tracker's step, and the object's three
    terms at (s_{t+1}, o_{t+1}) against orefs[t] join the reward. The hands
    are read from the FK the heads left in the scratch."""
    comptime D = SMDims[R]
    comptime OUT = SMObjDims[R].OUT
    var row = dlo.unsafe_offset(b * OUT)
    for d in range(D.DELTA):
        dl[unsafe_offset=b * D.DELTA + d] += row[unsafe_offset=d]
    var o1 = objs.unsafe_offset(((t + 1) * nb + b) * SM_OBJ)
    sm_obj_integrate_lane[DT, R](
        states.unsafe_offset((t * nb + b) * D.S), objs.unsafe_offset((t * nb + b) * SM_OBJ), row.unsafe_offset(D.DELTA), o1
    )
    _imag_step_lane[R](b, t, nb, seed, tree, table, states, dl, act, actions, prev, xi, refs, O, scratch, rewards, tin, eroot)
    var ro = sm_obj_reward_lane[DT, R](
        table, scratch.unsafe_offset(b * SMImagScratch[R].SIZE), o1, orefs.unsafe_offset((t * nb + b) * SM_OREF), rw,
        oterms.unsafe_offset((t * nb + b) * SM_OBJ_TERMS),
    )
    rewards[unsafe_offset=t * nb + b] += ro


def _k_oi_step[R: TrackingRobot](
    t: Int64, nb: Int64, seed: UInt32,
    tree: FP[DT], table: FP[DT], states: FP[DT], dl: FP[DT], dlo: FP[DT], act: FP[DT], actions: FP[DT], prev: FP[DT],
    xi: FP[DT], refs: FP[DT], O: FP[DT], scratch: FP[DT], rewards: FP[DT], tin: FP[DT], eroot: FP[DT],
    objs: FP[DT], orefs: FP[DT], rw: FP[DT], oterms: FP[DT],
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _oi_step_lane[R](
            b, Int(t), Int(nb), seed, tree, table, states, dl, dlo, act, actions, prev, xi, refs, O, scratch, rewards, tin, eroot,
            objs, orefs, rw, oterms,
        )


def sm_oi_step[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], t: Int, nb: Int, seed: UInt32,
    tree: FP[DT], table: FP[DT], states: FP[DT], dl: FP[DT], dlo: FP[DT], act: FP[DT], actions: FP[DT], prev: FP[DT],
    xi: FP[DT], refs: FP[DT], O: FP[DT], scratch: FP[DT], rewards: FP[DT], tin: FP[DT], eroot: FP[DT],
    objs: FP[DT], orefs: FP[DT], rw: FP[DT], oterms: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_oi_step[R]](
            Int64(t), Int64(nb), seed, tree, table, states, dl, dlo, act, actions, prev, xi, refs, O, scratch, rewards, tin, eroot,
            objs, orefs, rw, oterms, grid_dim=_grid(nb), block_dim=SM_TPB,
        )
    else:
        for b in range(nb):
            _oi_step_lane[R](
                b, t, nb, seed, tree, table, states, dl, dlo, act, actions, prev, xi, refs, O, scratch, rewards, tin, eroot,
                objs, orefs, rw, oterms,
            )


# ── the interface critic's input ────────────────────────────────────────────
@always_inline
def _oc_in_lane[R: TrackingRobot](
    r: Int, nb: Int, states: FP[DT], goals: FP[DT], xi: FP[DT], eroot: FP[DT], objs: FP[DT], orefs: FP[DT], octx: FP[DT], cin: FP[DT]
):
    """Row r = t·nb + b: [canonical(s), goal ⊙ gscale, normalise(ξ_b), e_root
    (the tracker's critic row), features(s, o, o_ref), normalise(ctx_b)]."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var b = r % nb
    var cr = cin.unsafe_offset(r * SMObjImagDims[R].OC_IN)
    sm_canonical_lane[DT, R](states.unsafe_offset(r * D.S), cr)
    for k in range(D.GOAL):
        cr[unsafe_offset=D.S + k] = goals[unsafe_offset=r * D.GOAL + k] * sm_goal_scale[n, DT](k)
    for k in range(D.X):
        cr[unsafe_offset=D.S + D.GOAL + k] = sm_xi_normalize[n, DT](k, xi[unsafe_offset=b * D.X + k])
    for k in range(D.EROOT):
        cr[unsafe_offset=D.S + D.GOAL + D.X + k] = eroot[unsafe_offset=r * D.EROOT + k]
    sm_oe_feat_lane[DT, R](
        states.unsafe_offset(r * D.S), objs.unsafe_offset(r * SM_OBJ), orefs.unsafe_offset(r * SM_OREF), cr.unsafe_offset(D.CRITIC_IN)
    )
    comptime for k in range(SM_OBJ_CTX):
        cr[unsafe_offset=D.CRITIC_IN + SM_OE_FEAT + k] = sm_obj_ctx_normalize[DT](k, octx[unsafe_offset=b * SM_OBJ_CTX + k])


def _k_oc_in[R: TrackingRobot](
    rows: Int64, nb: Int64, states: FP[DT], goals: FP[DT], xi: FP[DT], eroot: FP[DT], objs: FP[DT], orefs: FP[DT], octx: FP[DT], cin: FP[DT]
):
    var r = Int(global_idx.x)
    if r < Int(rows):
        _oc_in_lane[R](r, Int(nb), states, goals, xi, eroot, objs, orefs, octx, cin)


def sm_oc_in[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], rows: Int, nb: Int,
    states: FP[DT], goals: FP[DT], xi: FP[DT], eroot: FP[DT], objs: FP[DT], orefs: FP[DT], octx: FP[DT], cin: FP[DT],
) raises:
    """`rows` critic inputs from t-major rows of nb lanes (pointers at row 0)."""
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_oc_in[R]](
            Int64(rows), Int64(nb), states, goals, xi, eroot, objs, orefs, octx, cin, grid_dim=_grid(rows), block_dim=SM_TPB
        )
    else:
        for r in range(rows):
            _oc_in_lane[R](r, nb, states, goals, xi, eroot, objs, orefs, octx, cin)


# ── reverse ─────────────────────────────────────────────────────────────────
@always_inline
def _oc_v_back_lane[R: TrackingRobot](
    b: Int, sH: FP[DT], oH: FP[DT], refH: FP[DT], orefH: FP[DT], gcin: FP[DT], oscr: FP[DT], gs: FP[DT], go: FP[DT]
):
    """gs[b] = ∂V/∂s_H, go[b] = ∂V/∂o_H through the critic's input (both WRITTEN)."""
    comptime D = SMDims[R]
    var gsb = gs.unsafe_offset(b * D.S)
    for i in range(D.S):
        gsb[unsafe_offset=i] = 0
    var gob = go.unsafe_offset(b * SM_OBJ)
    comptime for i in range(SM_OBJ):
        gob[unsafe_offset=i] = 0
    var s = sH.unsafe_offset(b * D.S)
    var g = gcin.unsafe_offset(b * SMObjImagDims[R].OC_IN)
    sm_canonical_vjp_lane[DT, R](s, g, gsb)
    sm_root_error_vjp_lane[DT](s, _root_q[R](refH.unsafe_offset(b * D.REF)), g.unsafe_offset(D.S + D.GOAL + D.X), gsb)
    sm_oe_feat_vjp_lane[DT, R](
        s, oH.unsafe_offset(b * SM_OBJ), orefH.unsafe_offset(b * SM_OREF), g.unsafe_offset(D.CRITIC_IN),
        oscr.unsafe_offset(b * SM_OE_SCRATCH), gsb, gob,
    )


def _k_oc_v_back[R: TrackingRobot](
    nb: Int64, sH: FP[DT], oH: FP[DT], refH: FP[DT], orefH: FP[DT], gcin: FP[DT], oscr: FP[DT], gs: FP[DT], go: FP[DT]
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _oc_v_back_lane[R](b, sH, oH, refH, orefH, gcin, oscr, gs, go)


def sm_oc_v_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int,
    sH: FP[DT], oH: FP[DT], refH: FP[DT], orefH: FP[DT], gcin: FP[DT], oscr: FP[DT], gs: FP[DT], go: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_oc_v_back[R]](Int64(nb), sH, oH, refH, orefH, gcin, oscr, gs, go, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _oc_v_back_lane[R](b, sH, oH, refH, orefH, gcin, oscr, gs, go)


@always_inline
def _oi_heads_back_lane[R: TrackingRobot](
    b: Int, t: Int, nb: Int,
    tree: FP[DT], table: FP[DT], states: FP[DT], actions: FP[DT], prev: FP[DT], xi: FP[DT], refs: FP[DT],
    tin: FP[DT], gtin: FP[DT], g_r: FP[DT], gO: FP[DT], scratch: FP[DT],
    gs: FP[DT], ga: FP[DT], gap: FP[DT], dl: FP[DT], gprev: FP[DT], gdl: FP[DT],
    objs: FP[DT], orefs: FP[DT], rw: FP[DT], go: FP[DT],
):
    """`lanes._imag_heads_back_lane` with the object's terms: the heads' VJP
    at s_{t+1} is opened, the object's reward adds the hands' cotangents to
    the SAME FK block and its own into go[b] (∂/∂o_{t+1}, ADDS), then the FK
    is closed once; then the robot's integrator (`dl` is the TOTAL delta)."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime SC = SMImagScratch[R]
    var sc = scratch.unsafe_offset(b * SC.SIZE)
    var gsb = gs.unsafe_offset(b * D.S)
    var s1 = states.unsafe_offset(((t + 1) * nb + b) * D.S)
    var x0 = tin[unsafe_offset=b * 4 + 0]
    var x1 = tin[unsafe_offset=b * 4 + 1]
    sc[unsafe_offset=SC.GFEAT] = (gtin[unsafe_offset=b * 4 + 0] + _sign(x0) * gtin[unsafe_offset=b * 4 + 2]) / Scalar[DT](SM_HEIGHT_THR)
    sc[unsafe_offset=SC.GFEAT + 1] = (gtin[unsafe_offset=b * 4 + 1] + _sign(x1) * gtin[unsafe_offset=b * 4 + 3]) / Scalar[DT](SM_TILT_THR)
    var at = actions.unsafe_offset((t * nb + b) * n)
    var ap = prev.unsafe_offset(b * n) if t == 0 else actions.unsafe_offset(((t - 1) * nb + b) * n)
    var rf = refs.unsafe_offset((t * nb + b) * D.REF)
    var gr = g_r[unsafe_offset=t * nb + b]
    _ = sm_heads_lane[DT, R](
        tree, table, s1, at, ap, xi.unsafe_offset(b * D.X), rf,
        sc.unsafe_offset(SC.TERMS), sc.unsafe_offset(SC.FEAT), sc.unsafe_offset(SC.OBS), sc,
    )
    sm_heads_vjp_open_lane[DT, R](
        table, s1, at, ap, rf, gr, sc, sc.unsafe_offset(SC.GS1), sc.unsafe_offset(SC.GA1), sc.unsafe_offset(SC.GAP1),
    )
    sm_obj_reward_vjp_lane[DT, R](
        table, sc, objs.unsafe_offset(((t + 1) * nb + b) * SM_OBJ), orefs.unsafe_offset((t * nb + b) * SM_OREF), rw, gr,
        sc.unsafe_offset(D.FK_BLOCK), go.unsafe_offset(b * SM_OBJ),
    )
    sm_heads_vjp_close_lane[DT, R](
        tree, s1, sc.unsafe_offset(SC.GFEAT), gO.unsafe_offset(((D.HIST + t) * nb + b) * D.OBS), sc,
        sc.unsafe_offset(SC.GS1), sc.unsafe_offset(SC.GA1),
    )
    for k in range(D.S):
        gsb[unsafe_offset=k] += sc[unsafe_offset=SC.GS1 + k]
    for k in range(n):
        ga[unsafe_offset=b * n + k] = sc[unsafe_offset=SC.GA1 + k] + gap[unsafe_offset=b * n + k]
        gap[unsafe_offset=b * n + k] = sc[unsafe_offset=SC.GAP1 + k]
    var gp = gprev.unsafe_offset(b * D.S)
    for i in range(D.S):
        gp[unsafe_offset=i] = 0
    sm_integrate_vjp_lane[DT, R](
        states.unsafe_offset((t * nb + b) * D.S), dl.unsafe_offset(b * D.DELTA), gsb, gp, gdl.unsafe_offset(b * D.DELTA)
    )


def _k_oi_heads_back[R: TrackingRobot](
    t: Int64, nb: Int64,
    tree: FP[DT], table: FP[DT], states: FP[DT], actions: FP[DT], prev: FP[DT], xi: FP[DT], refs: FP[DT],
    tin: FP[DT], gtin: FP[DT], g_r: FP[DT], gO: FP[DT], scratch: FP[DT],
    gs: FP[DT], ga: FP[DT], gap: FP[DT], dl: FP[DT], gprev: FP[DT], gdl: FP[DT],
    objs: FP[DT], orefs: FP[DT], rw: FP[DT], go: FP[DT],
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _oi_heads_back_lane[R](
            b, Int(t), Int(nb), tree, table, states, actions, prev, xi, refs, tin, gtin, g_r, gO, scratch, gs, ga, gap, dl, gprev, gdl,
            objs, orefs, rw, go,
        )


def sm_oi_heads_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], t: Int, nb: Int,
    tree: FP[DT], table: FP[DT], states: FP[DT], actions: FP[DT], prev: FP[DT], xi: FP[DT], refs: FP[DT],
    tin: FP[DT], gtin: FP[DT], g_r: FP[DT], gO: FP[DT], scratch: FP[DT],
    gs: FP[DT], ga: FP[DT], gap: FP[DT], dl: FP[DT], gprev: FP[DT], gdl: FP[DT],
    objs: FP[DT], orefs: FP[DT], rw: FP[DT], go: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_oi_heads_back[R]](
            Int64(t), Int64(nb), tree, table, states, actions, prev, xi, refs, tin, gtin, g_r, gO, scratch, gs, ga, gap, dl, gprev, gdl,
            objs, orefs, rw, go, grid_dim=_grid(nb), block_dim=SM_TPB,
        )
    else:
        for b in range(nb):
            _oi_heads_back_lane[R](
                b, t, nb, tree, table, states, actions, prev, xi, refs, tin, gtin, g_r, gO, scratch, gs, ga, gap, dl, gprev, gdl,
                objs, orefs, rw, go,
            )


@always_inline
def _oe_back_lane[R: TrackingRobot](
    b: Int, use: Int, s: FP[DT], o: FP[DT], oref: FP[DT], geoin: FP[DT], oscr: FP[DT],
    gO: FP[DT], hstride: Int, bstride: Int, gs: FP[DT], go: FP[DT],
):
    """The object encoder input's VJP: gO += ∂/∂history; when `use` (t > 0:
    s_0 and o_0 are data), gs[b] += ∂/∂s_t and go[b] += ∂/∂o_t through the
    features."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var gr = geoin.unsafe_offset(b * SMObjImagDims[R].EO_IN)
    for hh in range(D.HIST):
        var ob = hh * hstride + b * bstride
        for k in range(D.OBS):
            gO[unsafe_offset=ob + k] += gr[unsafe_offset=SM_OE_FEAT + sm_hist_flat_index[n](hh, k)] * sm_obs_scale[n, DT](k)
    if use != 0:
        sm_oe_feat_vjp_lane[DT, R](
            s.unsafe_offset(b * D.S), o.unsafe_offset(b * SM_OBJ), oref.unsafe_offset(b * SM_OREF), gr,
            oscr.unsafe_offset(b * SM_OE_SCRATCH), gs.unsafe_offset(b * D.S), go.unsafe_offset(b * SM_OBJ),
        )


def _k_oe_back[R: TrackingRobot](
    nb: Int64, use: Int64, s: FP[DT], o: FP[DT], oref: FP[DT], geoin: FP[DT], oscr: FP[DT],
    gO: FP[DT], hstride: Int64, bstride: Int64, gs: FP[DT], go: FP[DT],
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _oe_back_lane[R](b, Int(use), s, o, oref, geoin, oscr, gO, Int(hstride), Int(bstride), gs, go)


def sm_oe_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, use: Bool, s: FP[DT], o: FP[DT], oref: FP[DT], geoin: FP[DT], oscr: FP[DT],
    gO: FP[DT], hstride: Int, bstride: Int, gs: FP[DT], go: FP[DT],
) raises:
    var u = 1 if use else 0
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_oe_back[R]](
            Int64(nb), Int64(u), s, o, oref, geoin, oscr, gO, Int64(hstride), Int64(bstride), gs, go,
            grid_dim=_grid(nb), block_dim=SM_TPB,
        )
    else:
        for b in range(nb):
            _oe_back_lane[R](b, u, s, o, oref, geoin, oscr, gO, hstride, bstride, gs, go)
