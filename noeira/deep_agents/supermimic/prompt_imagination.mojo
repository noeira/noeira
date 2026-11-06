# +--------------------------------------------------------------------------+ #
# | SuperMimic's prompt interface: its H-step rollout and policy gradient
# +--------------------------------------------------------------------------+ #
"""The downstream interface of the paper's Appendix B and G, through the frozen
world model and the frozen decoder:

    for t in 0..H−1:
        l_t      = enc_prompt([cmd_t, hist_t])      hist_t = O[t : t + 10]
        a_t      = dec([l_t, hist_t])               no gradient into its weights
        s_{t+1}  = wm.step(s_t, a_t, ξ)
        r_{t+1}  = the task reward at s_{t+1} (`prompt.mojo`)
        d̂_{t+1}  = σ(fall logit)                   c_{t+1} = c_t (1 − d̂_{t+1})
        O[10 + t] = observation (+ Table A.1's noise with a seed)
    J = mean_n [ Σ_t γᵗ c_t r_{t+1} + γᴴ c_H V(s_H, cmd_H, ξ) ]
    loss = −J / H + w_align · mean_{t, reference lanes} |l_t − l*_t|² / LATENT

`sm_imagine` is the reference for everything shared: the reverse scan is its
structure with the term head replaced by the logit's closed form, and only the
prompt encoder's parameters are meant to move. The decoder's and world model's
`vjp`s carry the gradient through them and leave it in their own grads, which
are never applied here (the caller's optimiser holds the encoder alone).

Lanes below `nref` are the reference rollouts: their commands are read from a
clip, and `lat` holds the motion encoder's latent at each step, frozen, for
the alignment (Appendix G: weight 0.1, "with a stop-gradient on the motion
latent"). The other lanes hold one command.
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.config import SMConfigT, SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.imagination import SMImagOut
from noeira.deep_agents.supermimic.lanes import (
    sm_copy, sm_dec_back, sm_dec_in, sm_imag_chain, sm_imag_j, sm_imag_surv, sm_imag_wm_back, sm_wm_in, talloc, tp,
)
from noeira.deep_agents.supermimic.prompt import SMSpec
from noeira.deep_agents.supermimic.prompt_lanes import (
    SMPromptImagScratch, sm_p_add, sm_p_align, sm_p_critic_in, sm_p_enc_back, sm_p_enc_in, sm_p_heads_back, sm_p_init,
    sm_p_step, sm_p_v_back,
)
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfigT, SMPromptCriticOf, SMPromptEncOf
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


struct SMPromptBatch[R: TrackingRobot, CMD: Int](Movable):
    """Start states and the per-step data of an imagined rollout (t-major):
    hist0 [10·n·OBS] the start's observation history (oldest first);
    cmds [(H+1)·n·CMD]; specs [H·n·SPEC]; lat [H·n·LATENT] the motion latents.
    Lanes below `nref` carry reference commands."""
    var n: Int
    var h: Int
    var nref: Int
    var state: TensorImpl[DT]
    var prev_action: TensorImpl[DT]
    var xi: TensorImpl[DT]
    var hist0: TensorImpl[DT]
    var cmds: TensorImpl[DT]
    var specs: TensorImpl[DT]
    var lat: TensorImpl[DT]

    def __init__(out self, n: Int, h: Int, nref: Int):
        comptime D = SMDims[Self.R]
        self.n = n
        self.h = h
        self.nref = nref
        self.state = TensorImpl[DT].alloc(n * D.S)
        self.prev_action = TensorImpl[DT].alloc(n * Self.R.NDOF)
        self.xi = TensorImpl[DT].alloc(n * D.X)
        self.hist0 = TensorImpl[DT].alloc(D.HIST * n * D.OBS)
        self.cmds = TensorImpl[DT].alloc((h + 1) * n * Self.CMD)
        self.specs = TensorImpl[DT].alloc(h * n * SMSpec.SIZE)
        self.lat = TensorImpl[DT].alloc(h * n * D.LATENT)

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.h = move.h
        self.nref = move.nref
        self.state = move.state^
        self.prev_action = move.prev_action^
        self.xi = move.xi^
        self.hist0 = move.hist0^
        self.cmds = move.cmds^
        self.specs = move.specs^
        self.lat = move.lat^

    def ensure_device(mut self, ctx: DeviceContext) raises:
        """Device buffers for a batch the device fills itself."""
        comptime D = SMDims[Self.R]
        self.state.ensure_gpu(ctx, self.n * D.S)
        self.prev_action.ensure_gpu(ctx, self.n * Self.R.NDOF)
        self.xi.ensure_gpu(ctx, self.n * D.X)
        self.hist0.ensure_gpu(ctx, D.HIST * self.n * D.OBS)
        self.cmds.ensure_gpu(ctx, (self.h + 1) * self.n * Self.CMD)
        self.specs.ensure_gpu(ctx, self.h * self.n * SMSpec.SIZE)
        self.lat.ensure_gpu(ctx, self.h * self.n * D.LATENT)

    def to_device(mut self, ctx: DeviceContext) raises:
        """Copy the host fields into their device buffers (kept across calls)."""
        self.state.upload_resident(ctx)
        self.prev_action.upload_resident(ctx)
        self.xi.upload_resident(ctx)
        self.hist0.upload_resident(ctx)
        self.cmds.upload_resident(ctx)
        self.specs.upload_resident(ctx)
        self.lat.upload_resident(ctx)


def sm_imagine_prompt[P: SMPromptConfigT, N: Int, target: StaticString = "cpu"](
    mut wm: SMWorldModelOf[P.BASE],
    mut pol: SMPolicyOf[P.BASE],
    mut enc: SMPromptEncOf[P],
    mut critic: SMPromptCriticOf[P],
    mut tb: SMTables[DT],
    mut ib: SMPromptBatch[P.BASE.ROBOT, P.CMD],
    gamma: Float64,
    w_align: Float64,
    grad: Bool,
    noise_seed: UInt32 = 0,
    ctx: Optional[DeviceContext] = None,
) raises -> SMImagOut[P.BASE.ROBOT]:
    """The rollout, with the alignment's mean squared error in `align_mse`;
    with `grad`, ADDS d(loss)/d(encoder params) into the encoder's grads (zero
    them first). The world model's, decoder's and critic's grads are polluted."""
    comptime R = P.BASE.ROBOT
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime S = D.S
    comptime OBS = D.OBS
    comptime HIST = D.HIST
    comptime CMD = P.CMD
    comptime LAT = D.LATENT
    comptime SPEC = SMSpec.SIZE
    comptime ENC_IN = CMD + D.HIST_FLAT
    comptime CRIT_IN = S + CMD + D.X
    comptime SC = SMPromptImagScratch[R]
    if ib.n != N:
        raise Error("sm_imagine_prompt: " + String(ib.n) + " lanes, compiled for " + String(N))
    var H = ib.h
    var NREF = ib.nref
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
    sm_p_init[target, R](ctx, N, tp[target](ib.state), tp[target](o.states), tp[target](o.survival))

    var c_align = Scalar[DT](2.0 * w_align / (Float64(H) * Float64(max(NREF, 1)) * Float64(LAT)))
    var encs = List[TensorImpl[DT]]()
    var decs = List[TensorImpl[DT]]()
    var xs = List[TensorImpl[DT]]()
    var dls = List[TensorImpl[DT]]()
    var gaz = talloc[target](H * N * LAT, ctx)                    # the alignment's ∂/∂l_t
    var asq = talloc[target](H * N, ctx)
    var z = talloc[target](N * LAT, ctx)
    var act = talloc[target](N * n, ctx)
    var lg = talloc[target](N, ctx)
    for t in range(H):
        # the prompt encoder, the frozen decoder
        var ein = talloc[target](N * ENC_IN, ctx)
        sm_p_enc_in[target, R, CMD](ctx, N, tp[target](ib.cmds).unsafe_offset(t * N * CMD), tp[target](O).unsafe_offset(t * N * OBS), N * OBS, OBS, tp[target](ein))
        enc.forward[target, N](TensorRefs[1, _, DT](ein), z, ctx)
        sm_p_align[target, LAT](
            ctx, N, NREF, c_align, tp[target](z), tp[target](ib.lat).unsafe_offset(t * N * LAT),
            tp[target](gaz).unsafe_offset(t * N * LAT), tp[target](asq).unsafe_offset(t * N),
        )
        var din = talloc[target](N * D.DEC_IN, ctx)
        sm_dec_in[target, R](ctx, N, tp[target](z), tp[target](O).unsafe_offset(t * N * OBS), N * OBS, OBS, tp[target](din))
        pol.dec.forward[target, N](TensorRefs[1, _, DT](din), act, ctx)
        # the world model
        var x = talloc[target](N * D.WM_IN, ctx)
        sm_wm_in[target, R](ctx, N, tp[target](o.states).unsafe_offset(t * N * S), tp[target](act), tp[target](ib.xi), tp[target](x))
        var dl = talloc[target](N * D.DELTA, ctx)
        wm.dyn.forward[target, N](TensorRefs[1, _, DT](x), dl, ctx)
        sm_p_step[target, R](
            ctx, t, N, noise_seed, tree, table, tp[target](o.states), tp[target](dl), tp[target](act),
            tp[target](o.actions), tp[target](ib.prev_action), tp[target](ib.xi), tp[target](ib.specs), tp[target](O),
            tp[target](scratch), tp[target](o.rewards), tp[target](lg),
        )
        sm_imag_surv[target](ctx, t, N, tp[target](lg), tp[target](o.dhat), tp[target](o.survival))
        encs.append(ein^)
        decs.append(din^)
        xs.append(x^)
        dls.append(dl^)

    # the critic at s_H
    var cin = talloc[target](N * CRIT_IN, ctx)
    sm_p_critic_in[target, R, CMD](
        ctx, N, N, tp[target](o.states).unsafe_offset(H * N * S), tp[target](ib.cmds).unsafe_offset(H * N * CMD),
        tp[target](ib.xi), tp[target](cin),
    )
    critic.forward[target, N](TensorRefs[1, _, DT](cin), o.value_h, ctx)
    var jb = talloc[target](2 * N, ctx)
    sm_imag_j[target](ctx, N, H, Scalar[DT](gamma), tp[target](o.survival), tp[target](o.rewards), tp[target](o.value_h), tp[target](jb))
    comptime if target == "gpu":
        jb.download(ctx.value())
        asq.download(ctx.value())
    var jsum = 0.0
    var usum = 0.0
    for b in range(N):
        jsum += Float64(jb.data[b])
        usum += Float64(jb.data[N + b])
    var aacc = 0.0
    for i in range(H * N):
        aacc += Float64(asq.data[i])
    o.j = jsum / Float64(N)
    o.return_undisc = usum / Float64(N)
    o.align_mse = aacc / (Float64(H) * Float64(max(NREF, 1)) * Float64(LAT))
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
    var gcin = talloc[target](N * CRIT_IN, ctx)
    critic.vjp[target, N](TensorRefs[1, _, DT](cin), gv, TensorRefs[1, _, DT](gcin), ctx)
    sm_p_v_back[target, R, CMD](ctx, N, tp[target](o.states).unsafe_offset(H * N * S), tp[target](gcin), tp[target](gs))

    var gO = talloc[target]((HIST + H) * N * OBS, ctx)
    var gap = talloc[target](N * n, ctx)            # ∂/∂a_t from step t+1
    var ga = talloc[target](N * n, ctx)             # ∂/∂a_t, all paths
    var gprev = talloc[target](N * S, ctx)
    var gdl = talloc[target](N * D.DELTA, ctx)
    var gx = talloc[target](N * D.WM_IN, ctx)
    var gdin = talloc[target](N * D.DEC_IN, ctx)
    var gz = talloc[target](N * LAT, ctx)
    var gein = talloc[target](N * ENC_IN, ctx)
    var redl = talloc[target](N * D.DELTA, ctx)
    var rea = talloc[target](N * n, ctx)
    var rez = talloc[target](N * LAT, ctx)
    var t = H - 1
    while t >= 0:
        # the reward and the logit at s_{t+1}, then the integrator
        sm_p_heads_back[target, R](
            ctx, t, N, tree, table, tp[target](o.states), tp[target](o.actions), tp[target](ib.prev_action),
            tp[target](ib.specs), tp[target](g_r), tp[target](g_lg), tp[target](gO), tp[target](scratch),
            tp[target](gs), tp[target](ga), tp[target](gap), tp[target](dls[t]), tp[target](gprev), tp[target](gdl),
        )
        # the dynamics: s_{t+1} = integrate(s_t, dyn([canonical(s_t), a_t, ξ]))
        wm.dyn.forward[target, N](TensorRefs[1, _, DT](xs[t]), redl, ctx)
        wm.dyn.vjp[target, N](TensorRefs[1, _, DT](xs[t]), gdl, TensorRefs[1, _, DT](gx), ctx)
        sm_imag_wm_back[target, R](ctx, N, t > 0, tp[target](o.states).unsafe_offset(t * N * S), tp[target](gx), tp[target](gprev), tp[target](gs), tp[target](ga))
        # the frozen decoder at step t, then the encoder it reads
        pol.dec.forward[target, N](TensorRefs[1, _, DT](decs[t]), rea, ctx)
        pol.dec.vjp[target, N](TensorRefs[1, _, DT](decs[t]), ga, TensorRefs[1, _, DT](gdin), ctx)
        sm_dec_back[target, R](ctx, N, tp[target](gdin), tp[target](gz), tp[target](gO).unsafe_offset(t * N * OBS), N * OBS, OBS)
        sm_p_add[target](ctx, N * LAT, tp[target](gaz).unsafe_offset(t * N * LAT), tp[target](gz))
        enc.forward[target, N](TensorRefs[1, _, DT](encs[t]), rez, ctx)
        enc.vjp[target, N](TensorRefs[1, _, DT](encs[t]), gz, TensorRefs[1, _, DT](gein), ctx)
        sm_p_enc_back[target, R, CMD](ctx, N, tp[target](gein), tp[target](gO).unsafe_offset(t * N * OBS), N * OBS, OBS)
        t -= 1
    return o^
