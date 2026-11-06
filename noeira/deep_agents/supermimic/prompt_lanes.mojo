# +--------------------------------------------------------------------------+ #
# | SuperMimic's prompt interface: the per-lane phases of its scan
# +--------------------------------------------------------------------------+ #
"""The phases `prompt_imagination.mojo` adds to `lanes.mojo`, under the same
rules: each is a `_*_lane` function with a CPU loop and a kernel over it, no
heap, no Float64 at runtime, scratch in per-lane slices. Everything that does
not depend on the task (the world model's input and back-pass, the survival
chain, J, the decoder's plumbing) is `lanes.mojo`'s own.
"""

from std.math import ceildiv

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.heads import sm_observation_lane
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import SM_TPB, sm_hash_u01
from noeira.deep_agents.supermimic.policy import sm_hist_flat_index, sm_obs_noise_half, sm_obs_scale
from noeira.deep_agents.supermimic.prompt import (
    SMPromptScratch, SMSpec, sm_prompt_fall_logit, sm_prompt_reward_lane, sm_prompt_reward_vjp_lane,
)
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.state_spec import (
    sm_canonical_lane, sm_canonical_vjp_lane, sm_integrate_lane, sm_integrate_vjp_lane, sm_xi_normalize,
)
from noeira.nn.constants import DT


@always_inline
def _grid(n: Int) -> Int:
    return ceildiv(n, SM_TPB)


struct SMPromptImagScratch[R: TrackingRobot]:
    """One lane's scratch in the scan: the reward's, then its cotangents."""
    comptime OBS = SMPromptScratch[Self.R].SIZE
    comptime GS1 = Self.OBS + SMDims[Self.R].OBS
    comptime GA1 = Self.GS1 + SMDims[Self.R].S
    comptime GAP1 = Self.GA1 + Self.R.NDOF
    comptime SIZE = Self.GAP1 + Self.R.NDOF


# ── the prompt encoder's input: [command, history ⊙ oscale] ─────────────────
@always_inline
def _p_enc_in_lane[R: TrackingRobot, CMD: Int](b: Int, cmd: FP[DT], O: FP[DT], hstride: Int, bstride: Int, ein: FP[DT]):
    """ein[b] = [cmd[b], the 10-slot history ⊙ oscale, term-major]; history
    slot hh of lane b is O[hh·hstride + b·bstride : +OBS]."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var er = ein.unsafe_offset(b * (CMD + D.HIST_FLAT))
    for k in range(CMD):
        er[unsafe_offset=k] = cmd[unsafe_offset=b * CMD + k]
    for hh in range(D.HIST):
        var ob = hh * hstride + b * bstride
        for k in range(D.OBS):
            er[unsafe_offset=CMD + sm_hist_flat_index[n](hh, k)] = O[unsafe_offset=ob + k] * sm_obs_scale[n, DT](k)


def _k_p_enc_in[R: TrackingRobot, CMD: Int](nb: Int64, cmd: FP[DT], O: FP[DT], hstride: Int64, bstride: Int64, ein: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _p_enc_in_lane[R, CMD](b, cmd, O, Int(hstride), Int(bstride), ein)


def sm_p_enc_in[target: StaticString, R: TrackingRobot, CMD: Int](
    ctx: Optional[DeviceContext], nb: Int, cmd: FP[DT], O: FP[DT], hstride: Int, bstride: Int, ein: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_enc_in[R, CMD]](
            Int64(nb), cmd, O, Int64(hstride), Int64(bstride), ein, grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _p_enc_in_lane[R, CMD](b, cmd, O, hstride, bstride, ein)


@always_inline
def _p_enc_back_lane[R: TrackingRobot, CMD: Int](b: Int, gein: FP[DT], gO: FP[DT], hstride: Int, bstride: Int):
    """gO += ∂/∂history of the encoder's input. The command is data."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var gr = gein.unsafe_offset(b * (CMD + D.HIST_FLAT))
    for hh in range(D.HIST):
        var ob = hh * hstride + b * bstride
        for k in range(D.OBS):
            gO[unsafe_offset=ob + k] += gr[unsafe_offset=CMD + sm_hist_flat_index[n](hh, k)] * sm_obs_scale[n, DT](k)


def _k_p_enc_back[R: TrackingRobot, CMD: Int](nb: Int64, gein: FP[DT], gO: FP[DT], hstride: Int64, bstride: Int64):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _p_enc_back_lane[R, CMD](b, gein, gO, Int(hstride), Int(bstride))


def sm_p_enc_back[target: StaticString, R: TrackingRobot, CMD: Int](
    ctx: Optional[DeviceContext], nb: Int, gein: FP[DT], gO: FP[DT], hstride: Int, bstride: Int
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_enc_back[R, CMD]](
            Int64(nb), gein, gO, Int64(hstride), Int64(bstride), grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _p_enc_back_lane[R, CMD](b, gein, gO, hstride, bstride)


# ── the rollout ─────────────────────────────────────────────────────────────
@always_inline
def _p_init_lane[R: TrackingRobot](b: Int, state0: FP[DT], states: FP[DT], surv: FP[DT]):
    comptime D = SMDims[R]
    for i in range(D.S):
        states[unsafe_offset=b * D.S + i] = state0[unsafe_offset=b * D.S + i]
    surv[unsafe_offset=b] = 1


def _k_p_init[R: TrackingRobot](nb: Int64, state0: FP[DT], states: FP[DT], surv: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _p_init_lane[R](b, state0, states, surv)


def sm_p_init[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, state0: FP[DT], states: FP[DT], surv: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_init[R]](Int64(nb), state0, states, surv, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _p_init_lane[R](b, state0, states, surv)


@always_inline
def _p_step_lane[R: TrackingRobot](
    b: Int, t: Int, nb: Int, seed: UInt32,
    tree: FP[DT], table: FP[DT], states: FP[DT], dl: FP[DT], act: FP[DT], actions: FP[DT], prev: FP[DT],
    xi: FP[DT], specs: FP[DT], O: FP[DT], scratch: FP[DT], rewards: FP[DT], lg: FP[DT],
):
    """Step t of lane b: s_{t+1} = integrate(s_t, dl), a_t stored, the task
    reward and the fall logit at s_{t+1}, the new observation slot (Table
    A.1's noise when `seed` ≠ 0)."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime SC = SMPromptImagScratch[R]
    var s0 = states.unsafe_offset((t * nb + b) * D.S)
    var s1 = states.unsafe_offset(((t + 1) * nb + b) * D.S)
    sm_integrate_lane[DT, R](s0, dl.unsafe_offset(b * D.DELTA), s1)
    var at = actions.unsafe_offset((t * nb + b) * n)
    for k in range(n):
        at[unsafe_offset=k] = act[unsafe_offset=b * n + k]
    var ap = prev.unsafe_offset(b * n) if t == 0 else actions.unsafe_offset(((t - 1) * nb + b) * n)
    var sc = scratch.unsafe_offset(b * SC.SIZE)
    rewards[unsafe_offset=t * nb + b] = sm_prompt_reward_lane[DT, R](
        tree, table, s1, at, ap, specs.unsafe_offset((t * nb + b) * SMSpec.SIZE), sc,
    )
    var obs = O.unsafe_offset(((D.HIST + t) * nb + b) * D.OBS)
    sm_observation_lane[DT, R](table, s1, at, xi.unsafe_offset(b * D.X), obs)
    if seed != 0:
        for k in range(D.OBS):
            var hw = sm_obs_noise_half[n, DT](k)
            if hw > 0:
                var u = sm_hash_u01(seed, UInt32(((t * nb + b) * D.OBS + k)))
                obs[unsafe_offset=k] += (Scalar[DT](2) * Scalar[DT](u) - 1) * hw
    lg[unsafe_offset=b] = sm_prompt_fall_logit[DT](s1)


def _k_p_step[R: TrackingRobot](
    t: Int64, nb: Int64, seed: UInt32,
    tree: FP[DT], table: FP[DT], states: FP[DT], dl: FP[DT], act: FP[DT], actions: FP[DT], prev: FP[DT],
    xi: FP[DT], specs: FP[DT], O: FP[DT], scratch: FP[DT], rewards: FP[DT], lg: FP[DT],
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _p_step_lane[R](b, Int(t), Int(nb), seed, tree, table, states, dl, act, actions, prev, xi, specs, O, scratch, rewards, lg)


def sm_p_step[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], t: Int, nb: Int, seed: UInt32,
    tree: FP[DT], table: FP[DT], states: FP[DT], dl: FP[DT], act: FP[DT], actions: FP[DT], prev: FP[DT],
    xi: FP[DT], specs: FP[DT], O: FP[DT], scratch: FP[DT], rewards: FP[DT], lg: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_step[R]](
            Int64(t), Int64(nb), seed, tree, table, states, dl, act, actions, prev, xi, specs, O, scratch, rewards, lg,
            grid_dim=_grid(nb), block_dim=SM_TPB,
        )
    else:
        for b in range(nb):
            _p_step_lane[R](b, t, nb, seed, tree, table, states, dl, act, actions, prev, xi, specs, O, scratch, rewards, lg)


@always_inline
def _p_heads_back_lane[R: TrackingRobot](
    b: Int, t: Int, nb: Int,
    tree: FP[DT], table: FP[DT], states: FP[DT], actions: FP[DT], prev: FP[DT], specs: FP[DT],
    g_r: FP[DT], g_lg: FP[DT], gO: FP[DT], scratch: FP[DT],
    gs: FP[DT], ga: FP[DT], gap: FP[DT], dl: FP[DT], gprev: FP[DT], gdl: FP[DT],
):
    """Reverse step t of lane b, as `_imag_heads_back_lane`: from ∂/∂s_{t+1}
    (gs) and ∂/∂a_t from step t+1's action rate (gap), the reward's VJP at
    s_{t+1} (forward re-run for its FK), then through the integrator. Leaves
    gs total, ga[b] = ∂/∂a_t so far, gap[b] = ∂/∂a_{t−1}, gprev = ∂/∂s_t
    through the integrator, gdl = ∂/∂dl."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime SC = SMPromptImagScratch[R]
    var sc = scratch.unsafe_offset(b * SC.SIZE)
    var gsb = gs.unsafe_offset(b * D.S)
    var s1 = states.unsafe_offset(((t + 1) * nb + b) * D.S)
    var at = actions.unsafe_offset((t * nb + b) * n)
    var ap = prev.unsafe_offset(b * n) if t == 0 else actions.unsafe_offset(((t - 1) * nb + b) * n)
    var spec = specs.unsafe_offset((t * nb + b) * SMSpec.SIZE)
    _ = sm_prompt_reward_lane[DT, R](tree, table, s1, at, ap, spec, sc)
    sm_prompt_reward_vjp_lane[DT, R](
        tree, table, s1, at, ap, spec, g_r[unsafe_offset=t * nb + b], g_lg[unsafe_offset=t * nb + b],
        gO.unsafe_offset(((D.HIST + t) * nb + b) * D.OBS), sc,
        sc.unsafe_offset(SC.GS1), sc.unsafe_offset(SC.GA1), sc.unsafe_offset(SC.GAP1),
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


def _k_p_heads_back[R: TrackingRobot](
    t: Int64, nb: Int64,
    tree: FP[DT], table: FP[DT], states: FP[DT], actions: FP[DT], prev: FP[DT], specs: FP[DT],
    g_r: FP[DT], g_lg: FP[DT], gO: FP[DT], scratch: FP[DT],
    gs: FP[DT], ga: FP[DT], gap: FP[DT], dl: FP[DT], gprev: FP[DT], gdl: FP[DT],
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _p_heads_back_lane[R](b, Int(t), Int(nb), tree, table, states, actions, prev, specs, g_r, g_lg, gO, scratch, gs, ga, gap, dl, gprev, gdl)


def sm_p_heads_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], t: Int, nb: Int,
    tree: FP[DT], table: FP[DT], states: FP[DT], actions: FP[DT], prev: FP[DT], specs: FP[DT],
    g_r: FP[DT], g_lg: FP[DT], gO: FP[DT], scratch: FP[DT],
    gs: FP[DT], ga: FP[DT], gap: FP[DT], dl: FP[DT], gprev: FP[DT], gdl: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_heads_back[R]](
            Int64(t), Int64(nb), tree, table, states, actions, prev, specs, g_r, g_lg, gO, scratch, gs, ga, gap, dl, gprev, gdl,
            grid_dim=_grid(nb), block_dim=SM_TPB,
        )
    else:
        for b in range(nb):
            _p_heads_back_lane[R](b, t, nb, tree, table, states, actions, prev, specs, g_r, g_lg, gO, scratch, gs, ga, gap, dl, gprev, gdl)


# ── the critic: V(canonical(s), command, normalise(ξ)) ──────────────────────
@always_inline
def _p_critic_in_lane[R: TrackingRobot, CMD: Int](r: Int, nb: Int, states: FP[DT], cmds: FP[DT], xi: FP[DT], cin: FP[DT]):
    """Row r = t·nb + b."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var b = r % nb
    var cr = cin.unsafe_offset(r * (D.S + CMD + D.X))
    sm_canonical_lane[DT, R](states.unsafe_offset(r * D.S), cr)
    for k in range(CMD):
        cr[unsafe_offset=D.S + k] = cmds[unsafe_offset=r * CMD + k]
    for k in range(D.X):
        cr[unsafe_offset=D.S + CMD + k] = sm_xi_normalize[n, DT](k, xi[unsafe_offset=b * D.X + k])


def _k_p_critic_in[R: TrackingRobot, CMD: Int](rows: Int64, nb: Int64, states: FP[DT], cmds: FP[DT], xi: FP[DT], cin: FP[DT]):
    var r = Int(global_idx.x)
    if r < Int(rows):
        _p_critic_in_lane[R, CMD](r, Int(nb), states, cmds, xi, cin)


def sm_p_critic_in[target: StaticString, R: TrackingRobot, CMD: Int](
    ctx: Optional[DeviceContext], rows: Int, nb: Int, states: FP[DT], cmds: FP[DT], xi: FP[DT], cin: FP[DT]
) raises:
    """`rows` critic inputs from t-major rows of nb lanes (pointers at row 0)."""
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_critic_in[R, CMD]](
            Int64(rows), Int64(nb), states, cmds, xi, cin, grid_dim=_grid(rows), block_dim=SM_TPB
        )
    else:
        for r in range(rows):
            _p_critic_in_lane[R, CMD](r, nb, states, cmds, xi, cin)


@always_inline
def _p_v_back_lane[R: TrackingRobot, CMD: Int](b: Int, sH: FP[DT], gcin: FP[DT], gs: FP[DT]):
    """gs[b] = ∂V(s_H)/∂s_H through the critic's canonical input."""
    comptime D = SMDims[R]
    var gsb = gs.unsafe_offset(b * D.S)
    for i in range(D.S):
        gsb[unsafe_offset=i] = 0
    sm_canonical_vjp_lane[DT, R](sH.unsafe_offset(b * D.S), gcin.unsafe_offset(b * (D.S + CMD + D.X)), gsb)


def _k_p_v_back[R: TrackingRobot, CMD: Int](nb: Int64, sH: FP[DT], gcin: FP[DT], gs: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _p_v_back_lane[R, CMD](b, sH, gcin, gs)


def sm_p_v_back[target: StaticString, R: TrackingRobot, CMD: Int](
    ctx: Optional[DeviceContext], nb: Int, sH: FP[DT], gcin: FP[DT], gs: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_v_back[R, CMD]](Int64(nb), sH, gcin, gs, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _p_v_back_lane[R, CMD](b, sH, gcin, gs)


# ── the alignment to the motion latent, on reference rollouts ───────────────
@always_inline
def _p_align_lane[L: Int](b: Int, nref: Int, c: Scalar[DT], z: FP[DT], lat: FP[DT], gaz: FP[DT], asq: FP[DT]):
    """Lanes below `nref` carry reference commands: gaz[b] = c·(z − lat), the
    loss's gradient once scaled; asq[b] = |z − lat|². Others: 0."""
    var acc = Scalar[DT](0)
    if b < nref:
        for k in range(L):
            var d = z[unsafe_offset=b * L + k] - lat[unsafe_offset=b * L + k]
            acc += d * d
            gaz[unsafe_offset=b * L + k] = c * d
    else:
        for k in range(L):
            gaz[unsafe_offset=b * L + k] = 0
    asq[unsafe_offset=b] = acc


def _k_p_align[L: Int](nb: Int64, nref: Int64, c: Scalar[DT], z: FP[DT], lat: FP[DT], gaz: FP[DT], asq: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _p_align_lane[L](b, Int(nref), c, z, lat, gaz, asq)


def sm_p_align[target: StaticString, L: Int](
    ctx: Optional[DeviceContext], nb: Int, nref: Int, c: Scalar[DT], z: FP[DT], lat: FP[DT], gaz: FP[DT], asq: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_align[L]](Int64(nb), Int64(nref), c, z, lat, gaz, asq, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _p_align_lane[L](b, nref, c, z, lat, gaz, asq)


def _k_p_add(n: Int64, src: FP[DT], dst: FP[DT]):
    var i = Int(global_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] += src[unsafe_offset=i]


def sm_p_add[target: StaticString](ctx: Optional[DeviceContext], n: Int, src: FP[DT], dst: FP[DT]) raises:
    """dst[0:n] += src[0:n]."""
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_add](Int64(n), src, dst, grid_dim=_grid(n), block_dim=SM_TPB)
    else:
        for i in range(n):
            dst[unsafe_offset=i] += src[unsafe_offset=i]


# ── a mixed batch of lanes: the motion latent below `split`, the prompt's above ──
def _k_p_merge[L: Int](nl: Int64, split: Int64, zp: FP[DT], z: FP[DT]):
    var i = Int(global_idx.x)
    if i >= Int(nl) * L:
        return
    if i // L >= Int(split):
        z[unsafe_offset=i] = zp[unsafe_offset=i]


def sm_p_merge[target: StaticString, L: Int](ctx: Optional[DeviceContext], nl: Int, split: Int, zp: FP[DT], z: FP[DT]) raises:
    """z[l] = zp[l] for the lanes from `split` on."""
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_p_merge[L]](Int64(nl), Int64(split), zp, z, grid_dim=_grid(nl * L), block_dim=SM_TPB)
    else:
        for i in range(split * L, nl * L):
            z[unsafe_offset=i] = zp[unsafe_offset=i]
