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
        O[10 + t] = o_t (+ Table A.1's noise with a seed)
    J = mean_n [ Σ_t γᵗ c_t r_{t+1} + γᴴ c_H V(s_H) ]

O[0..9] is the start's logged history. The reverse scan carries, per lane,
∂/∂s_{t+1}, ∂/∂a_t from step t+1's action-rate term, and ∂/∂O for the
history slots still ahead; the survival chain's gradient (∂J/∂c_t,
∂J/∂d_{t+1}) is closed-form from the forward values. The world model and the
critic are FROZEN: their `vjp`s accumulate into their own grads, which the
caller must zero before training them.

One scan for both targets: every per-lane loop is a `lanes.mojo` phase, so
`target` only decides whether the lanes run as a CPU loop or as kernels. On
"gpu", the batch is on the device (`SMImagBatch.to_device`), the tables too
(`SMTables.to_device`), the networks were made there, and the output stays
there (only J's per-lane terms come back).

⚠ Each step's `forward` is re-run before its `vjp` (the fused leaves keep only
the last forward's activations; see `world_model.mojo`).
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.config import SMConfigT, SMCriticOf, SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.lanes import (
    SM_TERM_IN, SMImagScratch, sm_copy, sm_critic_in, sm_dec_back, sm_dec_in, sm_enc_in, sm_imag_chain,
    sm_imag_heads_back, sm_imag_init, sm_imag_j, sm_imag_step, sm_imag_surv, sm_imag_v_back, sm_imag_wm_back,
    sm_wm_in, talloc, tp,
)
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


struct SMImagBatch[R: TrackingRobot](Movable):
    """Start states and the per-step data of an imagined rollout (t-major), as
    `nn` tensors (host `.data`, device `.dev` after `to_device`).
    hist0 [10·n·OBS] the start's observation history (oldest first);
    goals [(H+1)·n·GOAL], refs [(H+1)·n·REF] for frames min(frame + t, clip end)."""
    var n: Int
    var h: Int
    var state: TensorImpl[DT]
    var prev_action: TensorImpl[DT]
    var xi: TensorImpl[DT]
    var hist0: TensorImpl[DT]
    var goals: TensorImpl[DT]
    var refs: TensorImpl[DT]

    def __init__(out self, n: Int, h: Int):
        self.n = n
        self.h = h
        self.state = TensorImpl[DT].alloc(n * SMDims[Self.R].S)
        self.prev_action = TensorImpl[DT].alloc(n * Self.R.NDOF)
        self.xi = TensorImpl[DT].alloc(n * SMDims[Self.R].X)
        self.hist0 = TensorImpl[DT].alloc(SMDims[Self.R].HIST * n * SMDims[Self.R].OBS)
        self.goals = TensorImpl[DT].alloc((h + 1) * n * SMDims[Self.R].GOAL)
        self.refs = TensorImpl[DT].alloc((h + 1) * n * SMDims[Self.R].REF)

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.h = move.h
        self.state = move.state^
        self.prev_action = move.prev_action^
        self.xi = move.xi^
        self.hist0 = move.hist0^
        self.goals = move.goals^
        self.refs = move.refs^

    def to_device(mut self, ctx: DeviceContext) raises:
        """Copy the host fields into their device buffers (kept across calls)."""
        self.state.upload_resident(ctx)
        self.prev_action.upload_resident(ctx)
        self.xi.upload_resident(ctx)
        self.hist0.upload_resident(ctx)
        self.goals.upload_resident(ctx)
        self.refs.upload_resident(ctx)


def _alloc(n: Int, ctx: Optional[DeviceContext]) raises -> TensorImpl[DT]:
    if ctx:
        return TensorImpl[DT].alloc_gpu(ctx.value(), n)
    return TensorImpl[DT].alloc(n)


struct SMImagOut[R: TrackingRobot](Movable):
    """Per-step results, t-major, on the rollout's target: states
    [(H+1)·n·S], actions [H·n·NDOF], rewards [H·n] (r_{t+1}), dhat [(H+1)·n]
    (d_0 = 0), survival [(H+1)·n], e_root [(H+1)·n·4], value_h [n]; and the
    objective (host scalars)."""
    var j: Float64
    var return_undisc: Float64
    var states: TensorImpl[DT]
    var actions: TensorImpl[DT]
    var rewards: TensorImpl[DT]
    var dhat: TensorImpl[DT]
    var survival: TensorImpl[DT]
    var e_root: TensorImpl[DT]
    var value_h: TensorImpl[DT]

    def __init__(out self, n: Int, h: Int):
        """Host tensors."""
        self.j = 0
        self.return_undisc = 0
        self.states = TensorImpl[DT].alloc((h + 1) * n * SMDims[Self.R].S)
        self.actions = TensorImpl[DT].alloc(h * n * Self.R.NDOF)
        self.rewards = TensorImpl[DT].alloc(h * n)
        self.dhat = TensorImpl[DT].alloc((h + 1) * n)
        self.survival = TensorImpl[DT].alloc((h + 1) * n)
        self.e_root = TensorImpl[DT].alloc((h + 1) * n * SMDims[Self.R].EROOT)
        self.value_h = TensorImpl[DT].alloc(n)

    def __init__(out self, n: Int, h: Int, ctx: Optional[DeviceContext]) raises:
        """Device tensors when `ctx` is given, host tensors otherwise."""
        self.j = 0
        self.return_undisc = 0
        self.states = _alloc((h + 1) * n * SMDims[Self.R].S, ctx)
        self.actions = _alloc(h * n * Self.R.NDOF, ctx)
        self.rewards = _alloc(h * n, ctx)
        self.dhat = _alloc((h + 1) * n, ctx)
        self.survival = _alloc((h + 1) * n, ctx)
        self.e_root = _alloc((h + 1) * n * SMDims[Self.R].EROOT, ctx)
        self.value_h = _alloc(n, ctx)

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

    def to_host(mut self, ctx: DeviceContext) raises:
        """Download every field (the GPU path's outputs stay on the device)."""
        self.states.download_enqueue(ctx)
        self.actions.download_enqueue(ctx)
        self.rewards.download_enqueue(ctx)
        self.dhat.download_enqueue(ctx)
        self.survival.download_enqueue(ctx)
        self.e_root.download_enqueue(ctx)
        self.value_h.download_enqueue(ctx)
        ctx.synchronize()
        self.states.download_finalize()
        self.actions.download_finalize()
        self.rewards.download_finalize()
        self.dhat.download_finalize()
        self.survival.download_finalize()
        self.e_root.download_finalize()
        self.value_h.download_finalize()


def sm_imagine[C: SMConfigT, N: Int, target: StaticString = "cpu"](
    mut wm: SMWorldModelOf[C],
    mut pol: SMPolicyOf[C],
    mut critic: SMCriticOf[C],
    mut tb: SMTables[DT],
    mut ib: SMImagBatch[C.ROBOT],
    gamma: Float64,
    grad: Bool,
    noise_seed: UInt32 = 0,
    ctx: Optional[DeviceContext] = None,
) raises -> SMImagOut[C.ROBOT]:
    """The rollout; with `grad`, ADDS d(−J/H)/d params into the policy's grads
    (zero them first). The world model's and critic's grads are polluted.
    `noise_seed` ≠ 0 adds Table A.1's uniform sensor noise to each imagined
    observation as it enters the history (`rollout.py`'s `add_noise`), drawn
    by a counter hash of (seed, t, lane, entry): the same on both targets. It
    is data, so the gradient path is unchanged."""
    comptime R = C.ROBOT
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime S = D.S
    comptime OBS = D.OBS
    comptime HIST = D.HIST
    comptime SC = SMImagScratch[R]
    if ib.n != N:
        raise Error("sm_imagine: " + String(ib.n) + " lanes, compiled for " + String(N))
    var H = ib.h
    var o: SMImagOut[R]
    comptime if target == "gpu":
        o = SMImagOut[R](N, H, ctx)
    else:
        o = SMImagOut[R](N, H)
    var O = talloc[target]((HIST + H) * N * OBS, ctx)             # observation slots
    sm_copy[target](ctx, O, 0, ib.hist0, 0, HIST * N * OBS)
    var scratch = talloc[target](N * SC.SIZE, ctx)
    var tree = tb.tree_p[target]()
    var table = tb.head_p[target]()
    sm_imag_init[target, R](ctx, N, tp[target](ib.state), tp[target](o.states), tp[target](ib.refs), tp[target](o.e_root), tp[target](o.survival))

    var encs = List[TensorImpl[DT]]()
    var decs = List[TensorImpl[DT]]()
    var xs = List[TensorImpl[DT]]()
    var dls = List[TensorImpl[DT]]()
    var tins = List[TensorImpl[DT]]()
    var z = talloc[target](N * D.LATENT, ctx)
    var act = talloc[target](N * n, ctx)
    var lg = talloc[target](N, ctx)
    for t in range(H):
        # the policy
        var ein = talloc[target](N * D.GOAL, ctx)
        sm_enc_in[target, R](ctx, N, tp[target](ib.goals).unsafe_offset(t * N * D.GOAL), tp[target](ein))
        pol.enc.forward[target, N](TensorRefs[1, _, DT](ein), z, ctx)
        var din = talloc[target](N * D.DEC_IN, ctx)
        sm_dec_in[target, R](ctx, N, tp[target](z), tp[target](O).unsafe_offset(t * N * OBS), N * OBS, OBS, tp[target](din))
        pol.dec.forward[target, N](TensorRefs[1, _, DT](din), act, ctx)
        # the world model
        var x = talloc[target](N * D.WM_IN, ctx)
        sm_wm_in[target, R](ctx, N, tp[target](o.states).unsafe_offset(t * N * S), tp[target](act), tp[target](ib.xi), tp[target](x))
        var dl = talloc[target](N * D.DELTA, ctx)
        wm.dyn.forward[target, N](TensorRefs[1, _, DT](x), dl, ctx)
        var tin = talloc[target](N * SM_TERM_IN, ctx)
        sm_imag_step[target, R](
            ctx, t, N, noise_seed, tree, table, tp[target](o.states), tp[target](dl), tp[target](act),
            tp[target](o.actions), tp[target](ib.prev_action), tp[target](ib.xi), tp[target](ib.refs), tp[target](O),
            tp[target](scratch), tp[target](o.rewards), tp[target](tin), tp[target](o.e_root),
        )
        wm.term.forward[target, N](TensorRefs[1, _, DT](tin), lg, ctx)
        sm_imag_surv[target](ctx, t, N, tp[target](lg), tp[target](o.dhat), tp[target](o.survival))
        encs.append(ein^)
        decs.append(din^)
        xs.append(x^)
        dls.append(dl^)
        tins.append(tin^)

    # the critic at s_H
    var cin = talloc[target](N * D.CRITIC_IN, ctx)
    sm_critic_in[target, R](
        ctx, N, N, tp[target](o.states).unsafe_offset(H * N * S), tp[target](ib.goals).unsafe_offset(H * N * D.GOAL),
        tp[target](ib.xi), tp[target](o.e_root).unsafe_offset(H * N * D.EROOT), tp[target](cin),
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
        return o^

    # ── the survival chain, closed form: g_r[t], g_logit[t], g_V ──
    var g_r = talloc[target](H * N, ctx)
    var g_lg = talloc[target](H * N, ctx)
    var gv = talloc[target](N, ctx)
    sm_imag_chain[target](
        ctx, N, H, Scalar[DT](gamma), Scalar[DT](-1.0 / (Float64(H) * Float64(N))), tp[target](o.value_h),
        tp[target](o.survival), tp[target](o.dhat), tp[target](o.rewards), tp[target](g_r), tp[target](g_lg), tp[target](gv),
    )

    # ── V(s_H) ──
    var gs = talloc[target](N * S, ctx)              # ∂/∂s_{t+1}
    var vre = talloc[target](N, ctx)
    critic.forward[target, N](TensorRefs[1, _, DT](cin), vre, ctx)
    var gcin = talloc[target](N * D.CRITIC_IN, ctx)
    critic.vjp[target, N](TensorRefs[1, _, DT](cin), gv, TensorRefs[1, _, DT](gcin), ctx)
    sm_imag_v_back[target, R](
        ctx, N, tp[target](o.states).unsafe_offset(H * N * S), tp[target](ib.refs).unsafe_offset(H * N * D.REF),
        tp[target](gcin), tp[target](gs),
    )

    var gO = talloc[target]((HIST + H) * N * OBS, ctx)
    var gap = talloc[target](N * n, ctx)            # ∂/∂a_t from step t+1
    var ga = talloc[target](N * n, ctx)             # ∂/∂a_t, all paths
    var gprev = talloc[target](N * S, ctx)
    var gdl = talloc[target](N * D.DELTA, ctx)
    var gx = talloc[target](N * D.WM_IN, ctx)
    var glg = talloc[target](N, ctx)
    var gtin = talloc[target](N * SM_TERM_IN, ctx)
    var gdin = talloc[target](N * D.DEC_IN, ctx)
    var gz = talloc[target](N * D.LATENT, ctx)
    var gein = talloc[target](N * D.GOAL, ctx)
    var relg = talloc[target](N, ctx)
    var redl = talloc[target](N * D.DELTA, ctx)
    var rea = talloc[target](N * n, ctx)
    var rez = talloc[target](N * D.LATENT, ctx)
    var t = H - 1
    while t >= 0:
        # termination head at step t (re-run its forward first)
        wm.term.forward[target, N](TensorRefs[1, _, DT](tins[t]), relg, ctx)
        sm_copy[target](ctx, glg, 0, g_lg, t * N, N)
        wm.term.vjp[target, N](TensorRefs[1, _, DT](tins[t]), glg, TensorRefs[1, _, DT](gtin), ctx)
        # the heads at s_{t+1}, then the integrator
        sm_imag_heads_back[target, R](
            ctx, t, N, tree, table, tp[target](o.states), tp[target](o.actions), tp[target](ib.prev_action),
            tp[target](ib.xi), tp[target](ib.refs), tp[target](tins[t]), tp[target](gtin), tp[target](g_r), tp[target](gO),
            tp[target](scratch), tp[target](gs), tp[target](ga), tp[target](gap), tp[target](dls[t]), tp[target](gprev), tp[target](gdl),
        )
        # the dynamics: s_{t+1} = integrate(s_t, dyn([canonical(s_t), a_t, ξ]))
        wm.dyn.forward[target, N](TensorRefs[1, _, DT](xs[t]), redl, ctx)
        wm.dyn.vjp[target, N](TensorRefs[1, _, DT](xs[t]), gdl, TensorRefs[1, _, DT](gx), ctx)
        sm_imag_wm_back[target, R](ctx, N, t > 0, tp[target](o.states).unsafe_offset(t * N * S), tp[target](gx), tp[target](gprev), tp[target](gs), tp[target](ga))
        # the policy at step t
        pol.dec.forward[target, N](TensorRefs[1, _, DT](decs[t]), rea, ctx)
        pol.dec.vjp[target, N](TensorRefs[1, _, DT](decs[t]), ga, TensorRefs[1, _, DT](gdin), ctx)
        sm_dec_back[target, R](ctx, N, tp[target](gdin), tp[target](gz), tp[target](gO).unsafe_offset(t * N * OBS), N * OBS, OBS)
        pol.enc.forward[target, N](TensorRefs[1, _, DT](encs[t]), rez, ctx)
        pol.enc.vjp[target, N](TensorRefs[1, _, DT](encs[t]), gz, TensorRefs[1, _, DT](gein), ctx)
        t -= 1
    return o^
