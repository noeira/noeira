# +--------------------------------------------------------------------------+ #
# | SuperMimic M5: the H-step imagined rollout and its policy gradient
# +--------------------------------------------------------------------------+ #
"""The reproduction's `imagine` (`imagination/rollout.py`) + `ImagStep`
(`compiled.py`) + `policy_objective` (`objective.py`) through the frozen world
model, and the gradient of `loss = −J / H` with respect to the policy's
parameters by a hand-written reverse scan.

    for t in 0..H−1:
        a_t      = policy(hist_t, goal_t)          hist_t = O[t : t + 10]
        s_{t+1}  = wm.step(s_t, a_t, ξ)
        r_{t+1}, features, o_t = heads(s_{t+1}, a_t, a_{t−1}, ξ, ref_t)
        d_{t+1}  = σ(term_head(features))          c_{t+1} = c_t (1 − d_{t+1})
        O[10 + t] = o_t                            (noise-free: obs_noise=False)
    J = mean_n [ Σ_t γᵗ c_t r_{t+1} + γᴴ c_H V(s_H) ]

O[0..9] is the start's logged history. The reverse scan carries, per lane,
∂/∂s_{t+1}, ∂/∂a_t from step t+1's action-rate term, and ∂/∂O for the
history slots still ahead; the survival chain's gradient (∂J/∂c_t,
∂J/∂d_{t+1}) is closed-form from the forward values. The world model and the
critic are FROZEN: their `vjp`s accumulate into their own grads, which the
caller must zero before training them.

⚠ Each step's `forward` is re-run before its `vjp` (the fused leaves keep only
the last forward's activations; see `world_model.mojo`).
"""

from std.math import exp

from noeira.deep_agents.supermimic.g1_fk import FP, G1FKTree, Q4
from noeira.deep_agents.supermimic.heads import (
    G1HeadsTable, SM_N_TERMS, SM_N_TRACKED, SM_REF_BODY, SM_REF_DIM, SM_SCRATCH, g1_heads_lane,
    g1_heads_vjp_lane,
)
from noeira.deep_agents.supermimic.policy import (
    SMCritic, SMPolicy, SM_ACT, SM_CRITIC_IN, SM_DEC_IN, SM_EROOT, SM_GOAL_DIM, SM_HIST, SM_LATENT,
    SM_OBS, sm_goal_scale, sm_hist_flat_index, sm_obs_scale, sm_root_error_lane, sm_root_error_vjp_lane,
)
from noeira.deep_agents.supermimic.state_spec import (
    SM_DELTA_DIM, SM_FEAT_DIM, sm_canonical_lane, sm_canonical_vjp_lane, sm_integrate_lane,
    sm_integrate_vjp_lane, sm_xi_normalize,
)
from noeira.deep_agents.supermimic.world_model import (
    SMWorldModel, SM_HEIGHT_THR, SM_TERM_IN, SM_TILT_THR, SM_WM_IN,
)
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


comptime _S: Int = 71
comptime _XI: Int = 33
comptime _RR: Int = SM_N_TRACKED * SM_REF_BODY      # the reference root's offset in a ref row


struct SMImagBatch(Movable):
    """Start states and the per-step data of an imagined rollout (t-major).
    hist0 [10·n·93] the start's observation history (oldest first);
    goals [(H+1)·n·680], refs [(H+1)·n·189] for frames min(frame + t, clip end)."""
    var n: Int
    var h: Int
    var state: List[Scalar[DT]]
    var prev_action: List[Scalar[DT]]
    var xi: List[Scalar[DT]]
    var hist0: List[Scalar[DT]]
    var goals: List[Scalar[DT]]
    var refs: List[Scalar[DT]]

    def __init__(out self, n: Int, h: Int):
        self.n = n
        self.h = h
        self.state = List[Scalar[DT]](length=n * _S, fill=0)
        self.prev_action = List[Scalar[DT]](length=n * SM_ACT, fill=0)
        self.xi = List[Scalar[DT]](length=n * _XI, fill=0)
        self.hist0 = List[Scalar[DT]](length=SM_HIST * n * SM_OBS, fill=0)
        self.goals = List[Scalar[DT]](length=(h + 1) * n * SM_GOAL_DIM, fill=0)
        self.refs = List[Scalar[DT]](length=(h + 1) * n * SM_REF_DIM, fill=0)

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.h = move.h
        self.state = move.state^
        self.prev_action = move.prev_action^
        self.xi = move.xi^
        self.hist0 = move.hist0^
        self.goals = move.goals^
        self.refs = move.refs^


struct SMImagOut(Movable):
    """Per-step results, t-major: states [(H+1)·n·71], actions [H·n·29],
    rewards [H·n] (r_{t+1}), dhat [(H+1)·n] (d_0 = 0), survival [(H+1)·n],
    e_root [(H+1)·n·4], value_h [n]; and the objective."""
    var j: Float64
    var return_undisc: Float64
    var states: List[Scalar[DT]]
    var actions: List[Scalar[DT]]
    var rewards: List[Scalar[DT]]
    var dhat: List[Scalar[DT]]
    var survival: List[Scalar[DT]]
    var e_root: List[Scalar[DT]]
    var value_h: List[Scalar[DT]]

    def __init__(out self, n: Int, h: Int):
        self.j = 0
        self.return_undisc = 0
        self.states = List[Scalar[DT]](length=(h + 1) * n * _S, fill=0)
        self.actions = List[Scalar[DT]](length=h * n * SM_ACT, fill=0)
        self.rewards = List[Scalar[DT]](length=h * n, fill=0)
        self.dhat = List[Scalar[DT]](length=(h + 1) * n, fill=0)
        self.survival = List[Scalar[DT]](length=(h + 1) * n, fill=0)
        self.e_root = List[Scalar[DT]](length=(h + 1) * n * SM_EROOT, fill=0)
        self.value_h = List[Scalar[DT]](length=n, fill=0)

    def __init__(out self, *, deinit move: Self):
        self.j = move.j
        self.return_undisc = move.return_undisc
        self.states = move.states^
        self.actions = move.actions^
        self.rewards = move.rewards^
        self.dhat = move.dhat^
        self.survival = move.survival^
        self.e_root = move.e_root^
        self.value_h = move.value_h^


@always_inline
def _r(mut l: List[Scalar[DT]], off: Int) -> FP[DT]:
    return l.unsafe_ptr().unsafe_offset(off).as_unsafe_any_origin()


@always_inline
def _t(mut t: TensorImpl[DT], off: Int) -> FP[DT]:
    return t.data.unsafe_ptr().unsafe_offset(off).as_unsafe_any_origin()


@always_inline
def _sigmoid(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x)) if x >= 0 else exp(x) / (1.0 + exp(x))


@always_inline
def _root_q(mut l: List[Scalar[DT]], off: Int) -> Q4[DT]:
    return Q4[DT](l[off + _RR + 3], l[off + _RR + 4], l[off + _RR + 5], l[off + _RR + 6])


@always_inline
def _root_p(mut l: List[Scalar[DT]], off: Int) -> Q4[DT]:
    return Q4[DT](l[off + _RR], l[off + _RR + 1], l[off + _RR + 2], 0)


def sm_imagine[
    HW: Int, TH: Int, E1: Int, E2: Int, E3: Int, E4: Int, D1: Int, D2: Int, D3: Int, D4: Int,
    C1: Int, C2: Int, N: Int,
](
    mut wm: SMWorldModel[HW, TH],
    mut pol: SMPolicy[E1, E2, E3, E4, D1, D2, D3, D4],
    mut critic: SMCritic[C1, C2],
    mut tree: G1FKTree[DT],
    mut table: G1HeadsTable[DT],
    mut ib: SMImagBatch,
    gamma: Float64,
    grad: Bool,
) raises -> SMImagOut:
    """The rollout; with `grad`, ADDS d(−J/H)/d params into the policy's grads
    (zero them first). The world model's and critic's grads are polluted."""
    if ib.n != N:
        raise Error("sm_imagine: " + String(ib.n) + " lanes, compiled for " + String(N))
    var H = ib.h
    var o = SMImagOut(N, H)
    var xin = List[Scalar[DT]](length=N * _XI, fill=0)
    for i in range(N * _XI):
        xin[i] = Scalar[DT](sm_xi_normalize(i % _XI, Float64(ib.xi[i])))
    var O = List[Scalar[DT]](length=(SM_HIST + H) * N * SM_OBS, fill=0)    # observation slots
    for i in range(SM_HIST * N * SM_OBS):
        O[i] = ib.hist0[i]
    for i in range(N * _S):
        o.states[i] = ib.state[i]
    var encs = List[TensorImpl[DT]]()
    var decs = List[TensorImpl[DT]]()
    var xs = List[TensorImpl[DT]]()
    var dls = List[TensorImpl[DT]]()
    var tins = List[TensorImpl[DT]]()
    var scratch = List[Scalar[DT]](length=SM_SCRATCH, fill=0)
    var terms = List[Scalar[DT]](length=SM_N_TERMS, fill=0)
    var feat = List[Scalar[DT]](length=2, fill=0)
    for b in range(N):
        o.survival[b] = 1
        sm_root_error_lane[DT](_r(o.states, b * _S), _root_p(ib.refs, b * SM_REF_DIM), _root_q(ib.refs, b * SM_REF_DIM), _r(o.e_root, b * SM_EROOT))

    for t in range(H):
        # the policy
        var ein = TensorImpl[DT].alloc(N * SM_GOAL_DIM)
        for b in range(N):
            for k in range(SM_GOAL_DIM):
                ein.data[b * SM_GOAL_DIM + k] = ib.goals[(t * N + b) * SM_GOAL_DIM + k] * Scalar[DT](sm_goal_scale(k))
        var z = TensorImpl[DT].alloc(N * SM_LATENT)
        pol.enc.forward["cpu", N](TensorRefs[1, _, DT](ein), z, None)
        var din = TensorImpl[DT].alloc(N * SM_DEC_IN)
        for b in range(N):
            for k in range(SM_LATENT):
                din.data[b * SM_DEC_IN + k] = z.data[b * SM_LATENT + k]
            for hh in range(SM_HIST):
                var ob = ((t + hh) * N + b) * SM_OBS
                for k in range(SM_OBS):
                    din.data[b * SM_DEC_IN + SM_LATENT + sm_hist_flat_index(hh, k)] = O[ob + k] * Scalar[DT](sm_obs_scale(k))
        var act = TensorImpl[DT].alloc(N * SM_ACT)
        pol.dec.forward["cpu", N](TensorRefs[1, _, DT](din), act, None)
        for i in range(N * SM_ACT):
            o.actions[t * N * SM_ACT + i] = act.data[i]
        # the world model
        var x = TensorImpl[DT].alloc(N * SM_WM_IN)
        for b in range(N):
            sm_canonical_lane[DT](_r(o.states, (t * N + b) * _S), _t(x, b * SM_WM_IN))
            for k in range(SM_ACT):
                x.data[b * SM_WM_IN + SM_FEAT_DIM + k] = act.data[b * SM_ACT + k]
            for k in range(_XI):
                x.data[b * SM_WM_IN + SM_FEAT_DIM + SM_ACT + k] = xin[b * _XI + k]
        var dl = TensorImpl[DT].alloc(N * SM_DELTA_DIM)
        wm.dyn.forward["cpu", N](TensorRefs[1, _, DT](x), dl, None)
        var tin = TensorImpl[DT].alloc(N * SM_TERM_IN)
        for b in range(N):
            var sn = ((t + 1) * N + b) * _S
            sm_integrate_lane[DT](_r(o.states, (t * N + b) * _S), _t(dl, b * SM_DELTA_DIM), _r(o.states, sn))
            var ap = _r(ib.prev_action, b * SM_ACT) if t == 0 else _r(o.actions, ((t - 1) * N + b) * SM_ACT)
            var rr = g1_heads_lane[DT](
                tree.ptr(), table.ptr(), _r(o.states, sn), _r(o.actions, (t * N + b) * SM_ACT), ap,
                _r(ib.xi, b * _XI), _r(ib.refs, (t * N + b) * SM_REF_DIM), _r(terms, 0), _r(feat, 0),
                _r(O, ((SM_HIST + t) * N + b) * SM_OBS), _r(scratch, 0),
            )
            o.rewards[t * N + b] = rr
            var x0 = feat[0] / Scalar[DT](SM_HEIGHT_THR)
            var x1 = feat[1] / Scalar[DT](SM_TILT_THR)
            tin.data[b * 4 + 0] = x0
            tin.data[b * 4 + 1] = x1
            tin.data[b * 4 + 2] = abs(x0)
            tin.data[b * 4 + 3] = abs(x1)
            var ro = ((t + 1) * N + b) * SM_REF_DIM
            sm_root_error_lane[DT](_r(o.states, sn), _root_p(ib.refs, ro), _root_q(ib.refs, ro), _r(o.e_root, ((t + 1) * N + b) * SM_EROOT))
        var lg = TensorImpl[DT].alloc(N)
        wm.term.forward["cpu", N](TensorRefs[1, _, DT](tin), lg, None)
        for b in range(N):
            var d = Scalar[DT](_sigmoid(Float64(lg.data[b])))
            o.dhat[(t + 1) * N + b] = d
            o.survival[(t + 1) * N + b] = o.survival[t * N + b] * (1 - d)
        encs.append(ein^)
        decs.append(din^)
        xs.append(x^)
        dls.append(dl^)
        tins.append(tin^)

    # the critic at s_H
    var cin = TensorImpl[DT].alloc(N * SM_CRITIC_IN)
    for b in range(N):
        sm_canonical_lane[DT](_r(o.states, (H * N + b) * _S), _t(cin, b * SM_CRITIC_IN))
        for k in range(SM_GOAL_DIM):
            cin.data[b * SM_CRITIC_IN + SM_FEAT_DIM + k] = ib.goals[(H * N + b) * SM_GOAL_DIM + k] * Scalar[DT](sm_goal_scale(k))
        for k in range(_XI):
            cin.data[b * SM_CRITIC_IN + SM_FEAT_DIM + SM_GOAL_DIM + k] = xin[b * _XI + k]
        for k in range(SM_EROOT):
            cin.data[b * SM_CRITIC_IN + SM_FEAT_DIM + SM_GOAL_DIM + _XI + k] = o.e_root[(H * N + b) * SM_EROOT + k]
    var vh = TensorImpl[DT].alloc(N)
    critic.forward["cpu", N](TensorRefs[1, _, DT](cin), vh, None)
    var jsum = 0.0
    var usum = 0.0
    for b in range(N):
        o.value_h[b] = vh.data[b]
        var jb = 0.0
        var gp = 1.0
        for t in range(H):
            jb += gp * Float64(o.survival[t * N + b]) * Float64(o.rewards[t * N + b])
            usum += Float64(o.survival[t * N + b]) * Float64(o.rewards[t * N + b])
            gp *= gamma
        jb += gp * Float64(o.survival[H * N + b]) * Float64(vh.data[b])
        jsum += jb
    o.j = jsum / Float64(N)
    o.return_undisc = usum / Float64(N)
    if not grad:
        return o^

    # ── the survival chain, closed form: g_r[t], g_logit[t], g_V ──
    var gJ = -1.0 / (Float64(H) * Float64(N))
    var g_r = List[Float64](length=H * N, fill=0.0)
    var g_lg = List[Float64](length=H * N, fill=0.0)
    var gv = TensorImpl[DT].alloc(N)
    var gH = 1.0
    for _ in range(H):
        gH *= gamma
    for b in range(N):
        var gc = gJ * gH * Float64(vh.data[b])          # ∂J/∂c_H
        gv.data[b] = Scalar[DT](gJ * gH * Float64(o.survival[H * N + b]))
        var gp = gH
        var t = H - 1
        while t >= 0:
            gp /= gamma
            var c_t = Float64(o.survival[t * N + b])
            var d1 = Float64(o.dhat[(t + 1) * N + b])
            g_r[t * N + b] = gJ * gp * c_t
            var g_d = -gc * c_t                         # c_{t+1} = c_t (1 − d_{t+1})
            g_lg[t * N + b] = g_d * d1 * (1.0 - d1)
            gc = gJ * gp * Float64(o.rewards[t * N + b]) + gc * (1.0 - d1)
            t -= 1

    # ── V(s_H) ──
    var gs = List[Scalar[DT]](length=N * _S, fill=0)            # ∂/∂s_{t+1}
    critic.forward["cpu", N](TensorRefs[1, _, DT](cin), vh, None)
    var gcin = TensorImpl[DT].alloc(N * SM_CRITIC_IN)
    critic.vjp["cpu", N](TensorRefs[1, _, DT](cin), gv, TensorRefs[1, _, DT](gcin), None)
    for b in range(N):
        sm_canonical_vjp_lane[DT](_r(o.states, (H * N + b) * _S), _t(gcin, b * SM_CRITIC_IN), _r(gs, b * _S))
        sm_root_error_vjp_lane[DT](
            _r(o.states, (H * N + b) * _S), _root_q(ib.refs, (H * N + b) * SM_REF_DIM),
            _t(gcin, b * SM_CRITIC_IN + SM_FEAT_DIM + SM_GOAL_DIM + _XI), _r(gs, b * _S),
        )

    var gO = List[Scalar[DT]](length=(SM_HIST + H) * N * SM_OBS, fill=0)
    var gap = List[Scalar[DT]](length=N * SM_ACT, fill=0)        # ∂/∂a_t from step t+1
    var gfeat = List[Scalar[DT]](length=2, fill=0)
    var t = H - 1
    while t >= 0:
        # termination head at step t (re-run its forward first)
        var relg = TensorImpl[DT].alloc(N)
        wm.term.forward["cpu", N](TensorRefs[1, _, DT](tins[t]), relg, None)
        var glg = TensorImpl[DT].alloc(N)
        for b in range(N):
            glg.data[b] = Scalar[DT](g_lg[t * N + b])
        var gtin = TensorImpl[DT].alloc(N * SM_TERM_IN)
        wm.term.vjp["cpu", N](TensorRefs[1, _, DT](tins[t]), glg, TensorRefs[1, _, DT](gtin), None)
        var ga = TensorImpl[DT].alloc(N * SM_ACT)                 # ∂/∂a_t, all paths
        var gap_new = List[Scalar[DT]](length=N * SM_ACT, fill=0)
        for b in range(N):
            var x0 = tins[t].data[b * 4 + 0]
            var x1 = tins[t].data[b * 4 + 1]
            var s0 = Scalar[DT](1) if x0 > 0 else (Scalar[DT](-1) if x0 < 0 else Scalar[DT](0))
            var s1 = Scalar[DT](1) if x1 > 0 else (Scalar[DT](-1) if x1 < 0 else Scalar[DT](0))
            gfeat[0] = (gtin.data[b * 4 + 0] + s0 * gtin.data[b * 4 + 2]) / Scalar[DT](SM_HEIGHT_THR)
            gfeat[1] = (gtin.data[b * 4 + 1] + s1 * gtin.data[b * 4 + 3]) / Scalar[DT](SM_TILT_THR)
            var sn = ((t + 1) * N + b) * _S
            var ap = _r(ib.prev_action, b * SM_ACT) if t == 0 else _r(o.actions, ((t - 1) * N + b) * SM_ACT)
            var obs_tmp = List[Scalar[DT]](length=SM_OBS, fill=0)
            _ = g1_heads_lane[DT](
                tree.ptr(), table.ptr(), _r(o.states, sn), _r(o.actions, (t * N + b) * SM_ACT), ap,
                _r(ib.xi, b * _XI), _r(ib.refs, (t * N + b) * SM_REF_DIM), _r(terms, 0), _r(feat, 0),
                _r(obs_tmp, 0), _r(scratch, 0),
            )
            var g_s1 = List[Scalar[DT]](length=_S, fill=0)
            var g_a1 = List[Scalar[DT]](length=SM_ACT, fill=0)
            var g_ap1 = List[Scalar[DT]](length=SM_ACT, fill=0)
            g1_heads_vjp_lane[DT](
                tree.ptr(), table.ptr(), _r(o.states, sn), _r(o.actions, (t * N + b) * SM_ACT), ap,
                _r(ib.refs, (t * N + b) * SM_REF_DIM), Scalar[DT](g_r[t * N + b]), _r(gfeat, 0),
                _r(gO, ((SM_HIST + t) * N + b) * SM_OBS), _r(scratch, 0),
                _r(g_s1, 0), _r(g_a1, 0), _r(g_ap1, 0),
            )
            for k in range(_S):
                gs[b * _S + k] += g_s1[k]
            for k in range(SM_ACT):
                ga.data[b * SM_ACT + k] = g_a1[k] + gap[b * SM_ACT + k]
                gap_new[b * SM_ACT + k] = g_ap1[k]
        # the world-model step: s_{t+1} = integrate(s_t, dyn([canonical(s_t), a_t, ξ]))
        var gprev = List[Scalar[DT]](length=N * _S, fill=0)
        var gdl = TensorImpl[DT].alloc(N * SM_DELTA_DIM)
        for b in range(N):
            sm_integrate_vjp_lane[DT](
                _r(o.states, (t * N + b) * _S), _t(dls[t], b * SM_DELTA_DIM), _r(gs, b * _S),
                _r(gprev, b * _S), _t(gdl, b * SM_DELTA_DIM),
            )
        var redl = TensorImpl[DT].alloc(N * SM_DELTA_DIM)
        wm.dyn.forward["cpu", N](TensorRefs[1, _, DT](xs[t]), redl, None)
        var gx = TensorImpl[DT].alloc(N * SM_WM_IN)
        wm.dyn.vjp["cpu", N](TensorRefs[1, _, DT](xs[t]), gdl, TensorRefs[1, _, DT](gx), None)
        for b in range(N):
            if t > 0:
                sm_canonical_vjp_lane[DT](_r(o.states, (t * N + b) * _S), _t(gx, b * SM_WM_IN), _r(gprev, b * _S))
            for k in range(SM_ACT):
                ga.data[b * SM_ACT + k] += gx.data[b * SM_WM_IN + SM_FEAT_DIM + k]
        # the policy at step t
        var rea = TensorImpl[DT].alloc(N * SM_ACT)
        pol.dec.forward["cpu", N](TensorRefs[1, _, DT](decs[t]), rea, None)
        var gdin = TensorImpl[DT].alloc(N * SM_DEC_IN)
        pol.dec.vjp["cpu", N](TensorRefs[1, _, DT](decs[t]), ga, TensorRefs[1, _, DT](gdin), None)
        var gz = TensorImpl[DT].alloc(N * SM_LATENT)
        for b in range(N):
            for k in range(SM_LATENT):
                gz.data[b * SM_LATENT + k] = gdin.data[b * SM_DEC_IN + k]
            for hh in range(SM_HIST):
                var ob = ((t + hh) * N + b) * SM_OBS
                for k in range(SM_OBS):
                    gO[ob + k] += gdin.data[b * SM_DEC_IN + SM_LATENT + sm_hist_flat_index(hh, k)] * Scalar[DT](sm_obs_scale(k))
        var rez = TensorImpl[DT].alloc(N * SM_LATENT)
        pol.enc.forward["cpu", N](TensorRefs[1, _, DT](encs[t]), rez, None)
        var gein = TensorImpl[DT].alloc(N * SM_GOAL_DIM)
        pol.enc.vjp["cpu", N](TensorRefs[1, _, DT](encs[t]), gz, TensorRefs[1, _, DT](gein), None)
        gs = gprev^
        gap = gap_new^
        t -= 1
    return o^
