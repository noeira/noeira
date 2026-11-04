# +--------------------------------------------------------------------------+ #
# | SuperMimic: the per-lane phases of the scans, one body for both targets
# +--------------------------------------------------------------------------+ #
"""Every per-lane loop of the world-model loss (`world_model.mojo`), the
imagined rollout (`imagination.mojo`) and the critic's inputs is a `_*_lane`
function here, with two runners:

    phase[target](ctx, N, args...)
        "cpu" → for b in range(N): _phase_lane(b, args...)
        "gpu" → one kernel, one thread per lane, calling the same _phase_lane

so the scans are written once and `target` only picks where the lanes run.
Pointers are `tp[target](tensor)`: the host list on "cpu", the device buffer
on "gpu". Scalars cross into kernels as `Int64` / `Scalar[DT]` (an `Int` is
not device-passable).

Kernel-safety rules the lane math follows (Metal is the strictest target):
no heap allocation, no Float64 at runtime (Metal has no double: scale helpers
take the `dt` they return), no per-thread array indexed at runtime (scratch
lives in per-lane slices of device tensors).
"""

from std.math import ceildiv, exp
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.heads import (
    SM_N_TERMS, SMHeadLayout, sm_heads_lane, sm_heads_vjp_lane, sm_term_features, sm_term_features_vjp,
)
from noeira.deep_agents.supermimic.kinematics import FP, Q4
from noeira.deep_agents.supermimic.policy import (
    sm_goal_scale, sm_hist_flat_index, sm_obs_noise_half, sm_obs_scale, sm_root_error_lane, sm_root_error_vjp_lane,
)
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.state_spec import (
    sm_canonical_lane, sm_canonical_vjp_lane, sm_error_lane, sm_error_vjp_lane, sm_integrate_lane,
    sm_integrate_vjp_lane, sm_xi_normalize,
)
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl


comptime SM_TPB: Int = 64
comptime SM_TERM_IN: Int = 4
comptime SM_HEIGHT_THR: Float64 = 0.25
comptime SM_TILT_THR: Float64 = 0.8


# ── target plumbing ─────────────────────────────────────────────────────────
@always_inline
def tp[target: StaticString](mut t: TensorImpl[DT]) -> FP[DT]:
    """The tensor's storage on `target`: host list or device buffer."""
    comptime if target == "gpu":
        return mptr(t.dev.value().unsafe_ptr())
    else:
        return mptr(t.data)


def talloc[target: StaticString](n: Int, ctx: Optional[DeviceContext]) raises -> TensorImpl[DT]:
    """A zero-filled tensor of `n` on `target`."""
    return TensorImpl[DT].make[target](n, ctx)


@always_inline
def _grid(n: Int) -> Int:
    return ceildiv(n, SM_TPB)


@always_inline
def _sign[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return Scalar[dt](1) if x > 0 else (Scalar[dt](-1) if x < 0 else Scalar[dt](0))


@always_inline
def sm_hash_u01(seed: UInt32, i: UInt32) -> Float32:
    """A counter-based uniform in [0, 1): lowbias32 of (seed, i). The same
    value on every target, so a noisy rollout is reproducible across them."""
    var x = i * UInt32(0x9E3779B9) + seed * UInt32(0x85EBCA6B) + UInt32(0x632BE59B)
    x ^= x >> 16
    x *= UInt32(0x7FEB352D)
    x ^= x >> 15
    x *= UInt32(0x846CA68B)
    x ^= x >> 16
    return Float32(x >> 8) * Float32(1.0 / 16777216.0)


# ════════════════════════════════════════════════════════════════════════════
# The world model's k-step loss
# ════════════════════════════════════════════════════════════════════════════
@always_inline
def _wm_in_lane[R: TrackingRobot](b: Int, s: FP[DT], a: FP[DT], xi: FP[DT], x: FP[DT]):
    """x[b] = [canonical(s[b]), a[b], normalise(ξ[b])]."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var xr = x.unsafe_offset(b * D.WM_IN)
    sm_canonical_lane[DT, R](s.unsafe_offset(b * D.S), xr)
    for k in range(n):
        xr[unsafe_offset=D.S + k] = a[unsafe_offset=b * n + k]
    for k in range(D.X):
        xr[unsafe_offset=D.S + n + k] = sm_xi_normalize[n, DT](k, xi[unsafe_offset=b * D.X + k])


def _k_wm_in[R: TrackingRobot](nb: Int64, s: FP[DT], a: FP[DT], xi: FP[DT], x: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _wm_in_lane[R](b, s, a, xi, x)


def sm_wm_in[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, s: FP[DT], a: FP[DT], xi: FP[DT], x: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_wm_in[R]](Int64(nb), s, a, xi, x, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _wm_in_lane[R](b, s, a, xi, x)


@always_inline
def _wm_step_lane[R: TrackingRobot](
    b: Int, s: FP[DT], dl: FP[DT], s1: FP[DT], st1: FP[DT], rr: FP[DT], err: FP[DT], tin: FP[DT], sq: FP[DT]
):
    """s1[b] = integrate(s[b], dl[b]); err[b] = error(s1[b], st1[b]) and its
    square sum sq[b]; tin[b] = the termination head's input against the
    reference root rr[b] (pos 3, quat 4)."""
    comptime D = SMDims[R]
    var s1b = s1.unsafe_offset(b * D.S)
    sm_integrate_lane[DT, R](s.unsafe_offset(b * D.S), dl.unsafe_offset(b * D.DELTA), s1b)
    var eb = err.unsafe_offset(b * D.DELTA)
    sm_error_lane[DT, R](s1b, st1.unsafe_offset(b * D.S), eb)
    var acc = Scalar[DT](0)
    for d in range(D.DELTA):
        acc += eb[unsafe_offset=d] * eb[unsafe_offset=d]
    sq[unsafe_offset=b] = acc
    var r = rr.unsafe_offset(b * 7)
    var q = Q4[DT](r[unsafe_offset=3], r[unsafe_offset=4], r[unsafe_offset=5], r[unsafe_offset=6])
    var f = sm_term_features[DT](s1b, r[unsafe_offset=2], q)
    var x0 = f[0] / Scalar[DT](SM_HEIGHT_THR)
    var x1 = f[1] / Scalar[DT](SM_TILT_THR)
    tin[unsafe_offset=b * 4 + 0] = x0
    tin[unsafe_offset=b * 4 + 1] = x1
    tin[unsafe_offset=b * 4 + 2] = abs(x0)
    tin[unsafe_offset=b * 4 + 3] = abs(x1)


def _k_wm_step[R: TrackingRobot](
    nb: Int64, s: FP[DT], dl: FP[DT], s1: FP[DT], st1: FP[DT], rr: FP[DT], err: FP[DT], tin: FP[DT], sq: FP[DT]
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _wm_step_lane[R](b, s, dl, s1, st1, rr, err, tin, sq)


def sm_wm_step[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int,
    s: FP[DT], dl: FP[DT], s1: FP[DT], st1: FP[DT], rr: FP[DT], err: FP[DT], tin: FP[DT], sq: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_wm_step[R]](
            Int64(nb), s, dl, s1, st1, rr, err, tin, sq, grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _wm_step_lane[R](b, s, dl, s1, st1, rr, err, tin, sq)


@always_inline
def _wm_err_back_lane[R: TrackingRobot](
    b: Int, s1: FP[DT], st1: FP[DT], err: FP[DT], lg: FP[DT], done: FP[DT],
    cs: Scalar[DT], cb: Scalar[DT], gs: FP[DT], glg: FP[DT],
):
    """gs[b] += ∂(cs·|err|²/2·2)/∂s1 (cs = 2/(K·B) already holds the 2);
    glg[b] = (σ(lg) − y)·cb, the BCE's logit gradient."""
    comptime D = SMDims[R]
    var eb = err.unsafe_offset(b * D.DELTA)
    for d in range(D.DELTA):
        eb[unsafe_offset=d] = eb[unsafe_offset=d] * cs        # err becomes its own cotangent
    sm_error_vjp_lane[DT, R](s1.unsafe_offset(b * D.S), st1.unsafe_offset(b * D.S), eb, gs.unsafe_offset(b * D.S))
    var l = lg[unsafe_offset=b]
    var sg = Scalar[DT](1) / (Scalar[DT](1) + exp(-l)) if l >= 0 else exp(l) / (Scalar[DT](1) + exp(l))
    glg[unsafe_offset=b] = (sg - done[unsafe_offset=b]) * cb


def _k_wm_err_back[R: TrackingRobot](
    nb: Int64, s1: FP[DT], st1: FP[DT], err: FP[DT], lg: FP[DT], done: FP[DT],
    cs: Scalar[DT], cb: Scalar[DT], gs: FP[DT], glg: FP[DT],
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _wm_err_back_lane[R](b, s1, st1, err, lg, done, cs, cb, gs, glg)


def sm_wm_err_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, s1: FP[DT], st1: FP[DT], err: FP[DT], lg: FP[DT], done: FP[DT],
    cs: Scalar[DT], cb: Scalar[DT], gs: FP[DT], glg: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_wm_err_back[R]](
            Int64(nb), s1, st1, err, lg, done, cs, cb, gs, glg, grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _wm_err_back_lane[R](b, s1, st1, err, lg, done, cs, cb, gs, glg)


@always_inline
def _term_in_back[R: TrackingRobot](b: Int, tin: FP[DT], gtin: FP[DT], s1b: FP[DT], gsb: FP[DT]):
    """The termination input's VJP (x = f/thr, |x|) into the state's cotangent."""
    var x0 = tin[unsafe_offset=b * 4 + 0]
    var x1 = tin[unsafe_offset=b * 4 + 1]
    var g0 = (gtin[unsafe_offset=b * 4 + 0] + _sign(x0) * gtin[unsafe_offset=b * 4 + 2]) / Scalar[DT](SM_HEIGHT_THR)
    var g1 = (gtin[unsafe_offset=b * 4 + 1] + _sign(x1) * gtin[unsafe_offset=b * 4 + 3]) / Scalar[DT](SM_TILT_THR)
    sm_term_features_vjp[DT](s1b, g0, g1, gsb)


@always_inline
def _wm_dyn_back_lane[R: TrackingRobot](
    b: Int, tin: FP[DT], gtin: FP[DT], s: FP[DT], s1: FP[DT], dl: FP[DT], gs: FP[DT], gprev: FP[DT], gdl: FP[DT]
):
    """gs[b] += the termination input's VJP at s1; then through
    s1 = integrate(s, dl): gprev[b] = ∂/∂s (written), gdl[b] = ∂/∂dl."""
    comptime D = SMDims[R]
    var gsb = gs.unsafe_offset(b * D.S)
    _term_in_back[R](b, tin, gtin, s1.unsafe_offset(b * D.S), gsb)
    var gp = gprev.unsafe_offset(b * D.S)
    for i in range(D.S):
        gp[unsafe_offset=i] = 0
    sm_integrate_vjp_lane[DT, R](s.unsafe_offset(b * D.S), dl.unsafe_offset(b * D.DELTA), gsb, gp, gdl.unsafe_offset(b * D.DELTA))


def _k_wm_dyn_back[R: TrackingRobot](
    nb: Int64, tin: FP[DT], gtin: FP[DT], s: FP[DT], s1: FP[DT], dl: FP[DT], gs: FP[DT], gprev: FP[DT], gdl: FP[DT]
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _wm_dyn_back_lane[R](b, tin, gtin, s, s1, dl, gs, gprev, gdl)


def sm_wm_dyn_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int,
    tin: FP[DT], gtin: FP[DT], s: FP[DT], s1: FP[DT], dl: FP[DT], gs: FP[DT], gprev: FP[DT], gdl: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_wm_dyn_back[R]](
            Int64(nb), tin, gtin, s, s1, dl, gs, gprev, gdl, grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _wm_dyn_back_lane[R](b, tin, gtin, s, s1, dl, gs, gprev, gdl)


@always_inline
def _canon_back_lane[R: TrackingRobot](b: Int, row: Int, s: FP[DT], gx: FP[DT], gprev: FP[DT], gs: FP[DT], use: Int):
    """gs[b] = gprev[b] + (use ? ∂canonical(s[b])/∂s · gx[b, :S] : 0); `row`
    is gx's row width."""
    comptime D = SMDims[R]
    var gp = gprev.unsafe_offset(b * D.S)
    if use != 0:
        sm_canonical_vjp_lane[DT, R](s.unsafe_offset(b * D.S), gx.unsafe_offset(b * row), gp)
    var gsb = gs.unsafe_offset(b * D.S)
    for i in range(D.S):
        gsb[unsafe_offset=i] = gp[unsafe_offset=i]


def _k_canon_back[R: TrackingRobot](nb: Int64, row: Int64, s: FP[DT], gx: FP[DT], gprev: FP[DT], gs: FP[DT], use: Int64):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _canon_back_lane[R](b, Int(row), s, gx, gprev, gs, Int(use))


def sm_canon_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, row: Int, s: FP[DT], gx: FP[DT], gprev: FP[DT], gs: FP[DT], use: Bool
) raises:
    """gs = gprev (+ the canonical VJP of gx's first S columns when `use`)."""
    var u = 1 if use else 0
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_canon_back[R]](
            Int64(nb), Int64(row), s, gx, gprev, gs, Int64(u), grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _canon_back_lane[R](b, row, s, gx, gprev, gs, u)


# ════════════════════════════════════════════════════════════════════════════
# Shared plumbing: copies, clamps
# ════════════════════════════════════════════════════════════════════════════
def sm_copy[target: StaticString](
    ctx: Optional[DeviceContext], mut dst: TensorImpl[DT], doff: Int, mut src: TensorImpl[DT], soff: Int, count: Int
) raises:
    """dst[doff : doff + count] = src[soff : soff + count], on `target`."""
    comptime if target == "gpu":
        ctx.value().enqueue_copy(
            dst.dev.value().create_sub_buffer[DT](doff, count), src.dev.value().create_sub_buffer[DT](soff, count)
        )
    else:
        for i in range(count):
            dst.data[doff + i] = src.data[soff + i]


def _k_clamp(n: Int64, p: FP[DT], c: Scalar[DT]):
    var i = Int(global_idx.x)
    if i < Int(n):
        p[unsafe_offset=i] = min(max(p[unsafe_offset=i], -c), c)


def sm_clamp[target: StaticString](ctx: Optional[DeviceContext], n: Int, p: FP[DT], c: Scalar[DT]) raises:
    """Every element of p[0:n] into [−c, c] (torch's `clip_grad_value_`)."""
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_clamp](Int64(n), p, c, grid_dim=_grid(n), block_dim=SM_TPB)
    else:
        for i in range(n):
            p[unsafe_offset=i] = min(max(p[unsafe_offset=i], -c), c)


# ════════════════════════════════════════════════════════════════════════════
# The policy's inputs (act and imagination)
# ════════════════════════════════════════════════════════════════════════════
@always_inline
def _enc_in_lane[R: TrackingRobot](b: Int, goals: FP[DT], ein: FP[DT]):
    """ein[b] = goal[b] ⊙ gscale."""
    comptime G = SMDims[R].GOAL
    for k in range(G):
        ein[unsafe_offset=b * G + k] = goals[unsafe_offset=b * G + k] * sm_goal_scale[R.NDOF, DT](k)


def _k_enc_in[R: TrackingRobot](nb: Int64, goals: FP[DT], ein: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _enc_in_lane[R](b, goals, ein)


def sm_enc_in[target: StaticString, R: TrackingRobot](ctx: Optional[DeviceContext], nb: Int, goals: FP[DT], ein: FP[DT]) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_enc_in[R]](Int64(nb), goals, ein, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _enc_in_lane[R](b, goals, ein)


@always_inline
def _dec_in_lane[R: TrackingRobot](b: Int, z: FP[DT], O: FP[DT], hstride: Int, bstride: Int, din: FP[DT]):
    """din[b] = [z[b], the 10-slot history ⊙ oscale, term-major]; history slot
    hh of lane b is O[hh·hstride + b·bstride : +OBS]."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var dr = din.unsafe_offset(b * D.DEC_IN)
    for k in range(D.LATENT):
        dr[unsafe_offset=k] = z[unsafe_offset=b * D.LATENT + k]
    for hh in range(D.HIST):
        var ob = hh * hstride + b * bstride
        for k in range(D.OBS):
            dr[unsafe_offset=D.LATENT + sm_hist_flat_index[n](hh, k)] = O[unsafe_offset=ob + k] * sm_obs_scale[n, DT](k)


def _k_dec_in[R: TrackingRobot](nb: Int64, z: FP[DT], O: FP[DT], hstride: Int64, bstride: Int64, din: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _dec_in_lane[R](b, z, O, Int(hstride), Int(bstride), din)


def sm_dec_in[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, z: FP[DT], O: FP[DT], hstride: Int, bstride: Int, din: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_dec_in[R]](
            Int64(nb), z, O, Int64(hstride), Int64(bstride), din, grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _dec_in_lane[R](b, z, O, hstride, bstride, din)


@always_inline
def _dec_back_lane[R: TrackingRobot](b: Int, gdin: FP[DT], gz: FP[DT], gO: FP[DT], hstride: Int, bstride: Int):
    """The decoder input's VJP: gz[b] = ∂/∂z (written), gO += ∂/∂history."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var gr = gdin.unsafe_offset(b * D.DEC_IN)
    for k in range(D.LATENT):
        gz[unsafe_offset=b * D.LATENT + k] = gr[unsafe_offset=k]
    for hh in range(D.HIST):
        var ob = hh * hstride + b * bstride
        for k in range(D.OBS):
            gO[unsafe_offset=ob + k] += gr[unsafe_offset=D.LATENT + sm_hist_flat_index[n](hh, k)] * sm_obs_scale[n, DT](k)


def _k_dec_back[R: TrackingRobot](nb: Int64, gdin: FP[DT], gz: FP[DT], gO: FP[DT], hstride: Int64, bstride: Int64):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _dec_back_lane[R](b, gdin, gz, gO, Int(hstride), Int(bstride))


def sm_dec_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, gdin: FP[DT], gz: FP[DT], gO: FP[DT], hstride: Int, bstride: Int
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_dec_back[R]](
            Int64(nb), gdin, gz, gO, Int64(hstride), Int64(bstride), grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _dec_back_lane[R](b, gdin, gz, gO, hstride, bstride)


# ════════════════════════════════════════════════════════════════════════════
# The imagined rollout
# ════════════════════════════════════════════════════════════════════════════
struct SMImagScratch[R: TrackingRobot]:
    """One lane's scratch: the heads' FK block and cotangents, then the
    rollout's per-lane temporaries (never per-thread arrays: Metal)."""
    comptime TERMS = SMHeadLayout[Self.R].SCRATCH
    comptime FEAT = Self.TERMS + SM_N_TERMS
    comptime GFEAT = Self.FEAT + 2
    comptime OBS = Self.GFEAT + 2
    comptime GS1 = Self.OBS + SMDims[Self.R].OBS
    comptime GA1 = Self.GS1 + SMDims[Self.R].S
    comptime GAP1 = Self.GA1 + Self.R.NDOF
    comptime SIZE = Self.GAP1 + Self.R.NDOF


@always_inline
def _root_p[R: TrackingRobot](rf: FP[DT]) -> Q4[DT]:
    """The reference root position in a ref row (after the tracked bodies)."""
    comptime RR = 13 * R.N_TRACKED
    return Q4[DT](rf[unsafe_offset=RR], rf[unsafe_offset=RR + 1], rf[unsafe_offset=RR + 2], 0)


@always_inline
def _root_q[R: TrackingRobot](rf: FP[DT]) -> Q4[DT]:
    comptime RR = 13 * R.N_TRACKED
    return Q4[DT](rf[unsafe_offset=RR + 3], rf[unsafe_offset=RR + 4], rf[unsafe_offset=RR + 5], rf[unsafe_offset=RR + 6])


@always_inline
def _imag_init_lane[R: TrackingRobot](b: Int, state0: FP[DT], states: FP[DT], refs: FP[DT], eroot: FP[DT], surv: FP[DT]):
    """states[0, b] = the start; survival 1; the start's root error."""
    comptime D = SMDims[R]
    for i in range(D.S):
        states[unsafe_offset=b * D.S + i] = state0[unsafe_offset=b * D.S + i]
    surv[unsafe_offset=b] = 1
    var rf = refs.unsafe_offset(b * D.REF)
    sm_root_error_lane[DT](states.unsafe_offset(b * D.S), _root_p[R](rf), _root_q[R](rf), eroot.unsafe_offset(b * D.EROOT))


def _k_imag_init[R: TrackingRobot](nb: Int64, state0: FP[DT], states: FP[DT], refs: FP[DT], eroot: FP[DT], surv: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _imag_init_lane[R](b, state0, states, refs, eroot, surv)


def sm_imag_init[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, state0: FP[DT], states: FP[DT], refs: FP[DT], eroot: FP[DT], surv: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_imag_init[R]](Int64(nb), state0, states, refs, eroot, surv, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _imag_init_lane[R](b, state0, states, refs, eroot, surv)


@always_inline
def _imag_step_lane[R: TrackingRobot](
    b: Int, t: Int, nb: Int, seed: UInt32,
    tree: FP[DT], table: FP[DT], states: FP[DT], dl: FP[DT], act: FP[DT], actions: FP[DT], prev: FP[DT],
    xi: FP[DT], refs: FP[DT], O: FP[DT], scratch: FP[DT], rewards: FP[DT], tin: FP[DT], eroot: FP[DT],
):
    """Step t of lane b: s_{t+1} = integrate(s_t, dl), a_t stored, the heads
    at s_{t+1} (reward, termination input, the new observation slot, with
    Table A.1's noise when `seed` ≠ 0), the root error at t + 1."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime SC = SMImagScratch[R]
    var s0 = states.unsafe_offset((t * nb + b) * D.S)
    var s1 = states.unsafe_offset(((t + 1) * nb + b) * D.S)
    sm_integrate_lane[DT, R](s0, dl.unsafe_offset(b * D.DELTA), s1)
    var at = actions.unsafe_offset((t * nb + b) * n)
    for k in range(n):
        at[unsafe_offset=k] = act[unsafe_offset=b * n + k]
    var ap = prev.unsafe_offset(b * n) if t == 0 else actions.unsafe_offset(((t - 1) * nb + b) * n)
    var sc = scratch.unsafe_offset(b * SC.SIZE)
    var obs = O.unsafe_offset(((D.HIST + t) * nb + b) * D.OBS)
    var r = sm_heads_lane[DT, R](
        tree, table, s1, at, ap, xi.unsafe_offset(b * D.X), refs.unsafe_offset((t * nb + b) * D.REF),
        sc.unsafe_offset(SC.TERMS), sc.unsafe_offset(SC.FEAT), obs, sc,
    )
    rewards[unsafe_offset=t * nb + b] = r
    if seed != 0:
        for k in range(D.OBS):
            var hw = sm_obs_noise_half[n, DT](k)
            if hw > 0:
                var u = sm_hash_u01(seed, UInt32(((t * nb + b) * D.OBS + k)))
                obs[unsafe_offset=k] += (Scalar[DT](2) * Scalar[DT](u) - 1) * hw
    var x0 = sc[unsafe_offset=SC.FEAT] / Scalar[DT](SM_HEIGHT_THR)
    var x1 = sc[unsafe_offset=SC.FEAT + 1] / Scalar[DT](SM_TILT_THR)
    tin[unsafe_offset=b * 4 + 0] = x0
    tin[unsafe_offset=b * 4 + 1] = x1
    tin[unsafe_offset=b * 4 + 2] = abs(x0)
    tin[unsafe_offset=b * 4 + 3] = abs(x1)
    var rf1 = refs.unsafe_offset(((t + 1) * nb + b) * D.REF)
    sm_root_error_lane[DT](s1, _root_p[R](rf1), _root_q[R](rf1), eroot.unsafe_offset(((t + 1) * nb + b) * D.EROOT))


def _k_imag_step[R: TrackingRobot](
    t: Int64, nb: Int64, seed: UInt32,
    tree: FP[DT], table: FP[DT], states: FP[DT], dl: FP[DT], act: FP[DT], actions: FP[DT], prev: FP[DT],
    xi: FP[DT], refs: FP[DT], O: FP[DT], scratch: FP[DT], rewards: FP[DT], tin: FP[DT], eroot: FP[DT],
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _imag_step_lane[R](b, Int(t), Int(nb), seed, tree, table, states, dl, act, actions, prev, xi, refs, O, scratch, rewards, tin, eroot)


def sm_imag_step[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], t: Int, nb: Int, seed: UInt32,
    tree: FP[DT], table: FP[DT], states: FP[DT], dl: FP[DT], act: FP[DT], actions: FP[DT], prev: FP[DT],
    xi: FP[DT], refs: FP[DT], O: FP[DT], scratch: FP[DT], rewards: FP[DT], tin: FP[DT], eroot: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_imag_step[R]](
            Int64(t), Int64(nb), seed, tree, table, states, dl, act, actions, prev, xi, refs, O, scratch, rewards, tin, eroot,
            grid_dim=_grid(nb), block_dim=SM_TPB,
        )
    else:
        for b in range(nb):
            _imag_step_lane[R](b, t, nb, seed, tree, table, states, dl, act, actions, prev, xi, refs, O, scratch, rewards, tin, eroot)


@always_inline
def _imag_surv_lane(b: Int, t: Int, nb: Int, lg: FP[DT], dhat: FP[DT], surv: FP[DT]):
    """d̂_{t+1} = σ(logit), c_{t+1} = c_t (1 − d̂_{t+1})."""
    var l = lg[unsafe_offset=b]
    var d = Scalar[DT](1) / (Scalar[DT](1) + exp(-l)) if l >= 0 else exp(l) / (Scalar[DT](1) + exp(l))
    dhat[unsafe_offset=(t + 1) * nb + b] = d
    surv[unsafe_offset=(t + 1) * nb + b] = surv[unsafe_offset=t * nb + b] * (1 - d)


def _k_imag_surv(t: Int64, nb: Int64, lg: FP[DT], dhat: FP[DT], surv: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _imag_surv_lane(b, Int(t), Int(nb), lg, dhat, surv)


def sm_imag_surv[target: StaticString](ctx: Optional[DeviceContext], t: Int, nb: Int, lg: FP[DT], dhat: FP[DT], surv: FP[DT]) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_imag_surv](Int64(t), Int64(nb), lg, dhat, surv, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _imag_surv_lane(b, t, nb, lg, dhat, surv)


@always_inline
def _critic_in_lane[R: TrackingRobot](r: Int, nb: Int, states: FP[DT], goals: FP[DT], xi: FP[DT], eroot: FP[DT], cin: FP[DT]):
    """Row r = t·nb + b of the critic input: [canonical(s), goal ⊙ gscale,
    normalise(ξ_b), e_root]."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var b = r % nb
    var cr = cin.unsafe_offset(r * D.CRITIC_IN)
    sm_canonical_lane[DT, R](states.unsafe_offset(r * D.S), cr)
    for k in range(D.GOAL):
        cr[unsafe_offset=D.S + k] = goals[unsafe_offset=r * D.GOAL + k] * sm_goal_scale[n, DT](k)
    for k in range(D.X):
        cr[unsafe_offset=D.S + D.GOAL + k] = sm_xi_normalize[n, DT](k, xi[unsafe_offset=b * D.X + k])
    for k in range(D.EROOT):
        cr[unsafe_offset=D.S + D.GOAL + D.X + k] = eroot[unsafe_offset=r * D.EROOT + k]


def _k_critic_in[R: TrackingRobot](rows: Int64, nb: Int64, states: FP[DT], goals: FP[DT], xi: FP[DT], eroot: FP[DT], cin: FP[DT]):
    var r = Int(global_idx.x)
    if r < Int(rows):
        _critic_in_lane[R](r, Int(nb), states, goals, xi, eroot, cin)


def sm_critic_in[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], rows: Int, nb: Int, states: FP[DT], goals: FP[DT], xi: FP[DT], eroot: FP[DT], cin: FP[DT]
) raises:
    """`rows` critic inputs from t-major rows of nb lanes (pointers at row 0)."""
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_critic_in[R]](
            Int64(rows), Int64(nb), states, goals, xi, eroot, cin, grid_dim=_grid(rows), block_dim=SM_TPB
        )
    else:
        for r in range(rows):
            _critic_in_lane[R](r, nb, states, goals, xi, eroot, cin)


@always_inline
def _imag_j_lane(b: Int, nb: Int, h: Int, gamma: Scalar[DT], surv: FP[DT], rewards: FP[DT], vh: FP[DT], jb: FP[DT]):
    """jb[b] = Σ_t γᵗ c_t r_{t+1} + γᴴ c_H V(s_H); jb[nb + b] = Σ_t c_t r_{t+1}."""
    var j = Scalar[DT](0)
    var u = Scalar[DT](0)
    var gp = Scalar[DT](1)
    for t in range(h):
        var cr = surv[unsafe_offset=t * nb + b] * rewards[unsafe_offset=t * nb + b]
        j += gp * cr
        u += cr
        gp *= gamma
    j += gp * surv[unsafe_offset=h * nb + b] * vh[unsafe_offset=b]
    jb[unsafe_offset=b] = j
    jb[unsafe_offset=nb + b] = u


def _k_imag_j(nb: Int64, h: Int64, gamma: Scalar[DT], surv: FP[DT], rewards: FP[DT], vh: FP[DT], jb: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _imag_j_lane(b, Int(nb), Int(h), gamma, surv, rewards, vh, jb)


def sm_imag_j[target: StaticString](
    ctx: Optional[DeviceContext], nb: Int, h: Int, gamma: Scalar[DT], surv: FP[DT], rewards: FP[DT], vh: FP[DT], jb: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_imag_j](Int64(nb), Int64(h), gamma, surv, rewards, vh, jb, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _imag_j_lane(b, nb, h, gamma, surv, rewards, vh, jb)


@always_inline
def _imag_chain_lane(
    b: Int, nb: Int, h: Int, gamma: Scalar[DT], gJ: Scalar[DT],
    vh: FP[DT], surv: FP[DT], dhat: FP[DT], rewards: FP[DT], g_r: FP[DT], g_lg: FP[DT], gv: FP[DT],
):
    """The survival chain's gradient, closed form from the forward values:
    g_r[t] = ∂J/∂r_{t+1}, g_lg[t] = ∂J/∂logit_{t+1}, gv = ∂J/∂V(s_H)
    (J scaled by gJ = −1/(H·N))."""
    var gH = Scalar[DT](1)
    for _ in range(h):
        gH *= gamma
    var gc = gJ * gH * vh[unsafe_offset=b]                  # ∂J/∂c_H
    gv[unsafe_offset=b] = gJ * gH * surv[unsafe_offset=h * nb + b]
    var gp = gH
    var t = h - 1
    while t >= 0:
        gp /= gamma
        var c_t = surv[unsafe_offset=t * nb + b]
        var d1 = dhat[unsafe_offset=(t + 1) * nb + b]
        g_r[unsafe_offset=t * nb + b] = gJ * gp * c_t
        var g_d = -gc * c_t                                  # c_{t+1} = c_t (1 − d_{t+1})
        g_lg[unsafe_offset=t * nb + b] = g_d * d1 * (1 - d1)
        gc = gJ * gp * rewards[unsafe_offset=t * nb + b] + gc * (1 - d1)
        t -= 1


def _k_imag_chain(
    nb: Int64, h: Int64, gamma: Scalar[DT], gJ: Scalar[DT],
    vh: FP[DT], surv: FP[DT], dhat: FP[DT], rewards: FP[DT], g_r: FP[DT], g_lg: FP[DT], gv: FP[DT],
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _imag_chain_lane(b, Int(nb), Int(h), gamma, gJ, vh, surv, dhat, rewards, g_r, g_lg, gv)


def sm_imag_chain[target: StaticString](
    ctx: Optional[DeviceContext], nb: Int, h: Int, gamma: Scalar[DT], gJ: Scalar[DT],
    vh: FP[DT], surv: FP[DT], dhat: FP[DT], rewards: FP[DT], g_r: FP[DT], g_lg: FP[DT], gv: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_imag_chain](
            Int64(nb), Int64(h), gamma, gJ, vh, surv, dhat, rewards, g_r, g_lg, gv, grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _imag_chain_lane(b, nb, h, gamma, gJ, vh, surv, dhat, rewards, g_r, g_lg, gv)


@always_inline
def _imag_v_back_lane[R: TrackingRobot](b: Int, sH: FP[DT], refH: FP[DT], gcin: FP[DT], gs: FP[DT]):
    """gs[b] = ∂V(s_H)/∂s_H through the critic input (canonical + root error)."""
    comptime D = SMDims[R]
    var gsb = gs.unsafe_offset(b * D.S)
    for i in range(D.S):
        gsb[unsafe_offset=i] = 0
    var s = sH.unsafe_offset(b * D.S)
    var g = gcin.unsafe_offset(b * D.CRITIC_IN)
    sm_canonical_vjp_lane[DT, R](s, g, gsb)
    sm_root_error_vjp_lane[DT](s, _root_q[R](refH.unsafe_offset(b * D.REF)), g.unsafe_offset(D.S + D.GOAL + D.X), gsb)


def _k_imag_v_back[R: TrackingRobot](nb: Int64, sH: FP[DT], refH: FP[DT], gcin: FP[DT], gs: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _imag_v_back_lane[R](b, sH, refH, gcin, gs)


def sm_imag_v_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, sH: FP[DT], refH: FP[DT], gcin: FP[DT], gs: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_imag_v_back[R]](Int64(nb), sH, refH, gcin, gs, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _imag_v_back_lane[R](b, sH, refH, gcin, gs)


@always_inline
def _imag_heads_back_lane[R: TrackingRobot](
    b: Int, t: Int, nb: Int,
    tree: FP[DT], table: FP[DT], states: FP[DT], actions: FP[DT], prev: FP[DT], xi: FP[DT], refs: FP[DT],
    tin: FP[DT], gtin: FP[DT], g_r: FP[DT], gO: FP[DT], scratch: FP[DT],
    gs: FP[DT], ga: FP[DT], gap: FP[DT], dl: FP[DT], gprev: FP[DT], gdl: FP[DT],
):
    """Reverse step t of lane b, from ∂/∂s_{t+1} (gs) and ∂/∂a_t from step
    t+1's action rate (gap): the heads' VJP at s_{t+1} (forward re-run for
    its FK), then through s_{t+1} = integrate(s_t, dl). Leaves gs = ∂/∂s_{t+1}
    total, ga[b] = ∂/∂a_t so far, gap[b] = ∂/∂a_{t−1}, gprev = ∂/∂s_t through
    the integrator, gdl = ∂/∂dl."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime SC = SMImagScratch[R]
    var sc = scratch.unsafe_offset(b * SC.SIZE)
    var gsb = gs.unsafe_offset(b * D.S)
    var s1 = states.unsafe_offset(((t + 1) * nb + b) * D.S)
    # the termination input's VJP → g_feat (2), divided by the thresholds
    var x0 = tin[unsafe_offset=b * 4 + 0]
    var x1 = tin[unsafe_offset=b * 4 + 1]
    sc[unsafe_offset=SC.GFEAT] = (gtin[unsafe_offset=b * 4 + 0] + _sign(x0) * gtin[unsafe_offset=b * 4 + 2]) / Scalar[DT](SM_HEIGHT_THR)
    sc[unsafe_offset=SC.GFEAT + 1] = (gtin[unsafe_offset=b * 4 + 1] + _sign(x1) * gtin[unsafe_offset=b * 4 + 3]) / Scalar[DT](SM_TILT_THR)
    var at = actions.unsafe_offset((t * nb + b) * n)
    var ap = prev.unsafe_offset(b * n) if t == 0 else actions.unsafe_offset(((t - 1) * nb + b) * n)
    var rf = refs.unsafe_offset((t * nb + b) * D.REF)
    _ = sm_heads_lane[DT, R](
        tree, table, s1, at, ap, xi.unsafe_offset(b * D.X), rf,
        sc.unsafe_offset(SC.TERMS), sc.unsafe_offset(SC.FEAT), sc.unsafe_offset(SC.OBS), sc,
    )
    sm_heads_vjp_lane[DT, R](
        tree, table, s1, at, ap, rf, g_r[unsafe_offset=t * nb + b], sc.unsafe_offset(SC.GFEAT),
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


def _k_imag_heads_back[R: TrackingRobot](
    t: Int64, nb: Int64,
    tree: FP[DT], table: FP[DT], states: FP[DT], actions: FP[DT], prev: FP[DT], xi: FP[DT], refs: FP[DT],
    tin: FP[DT], gtin: FP[DT], g_r: FP[DT], gO: FP[DT], scratch: FP[DT],
    gs: FP[DT], ga: FP[DT], gap: FP[DT], dl: FP[DT], gprev: FP[DT], gdl: FP[DT],
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _imag_heads_back_lane[R](b, Int(t), Int(nb), tree, table, states, actions, prev, xi, refs, tin, gtin, g_r, gO, scratch, gs, ga, gap, dl, gprev, gdl)


def sm_imag_heads_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], t: Int, nb: Int,
    tree: FP[DT], table: FP[DT], states: FP[DT], actions: FP[DT], prev: FP[DT], xi: FP[DT], refs: FP[DT],
    tin: FP[DT], gtin: FP[DT], g_r: FP[DT], gO: FP[DT], scratch: FP[DT],
    gs: FP[DT], ga: FP[DT], gap: FP[DT], dl: FP[DT], gprev: FP[DT], gdl: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_imag_heads_back[R]](
            Int64(t), Int64(nb), tree, table, states, actions, prev, xi, refs, tin, gtin, g_r, gO, scratch, gs, ga, gap, dl, gprev, gdl,
            grid_dim=_grid(nb), block_dim=SM_TPB,
        )
    else:
        for b in range(nb):
            _imag_heads_back_lane[R](b, t, nb, tree, table, states, actions, prev, xi, refs, tin, gtin, g_r, gO, scratch, gs, ga, gap, dl, gprev, gdl)


@always_inline
def _imag_wm_back_lane[R: TrackingRobot](b: Int, use: Int, s: FP[DT], gx: FP[DT], gprev: FP[DT], gs: FP[DT], ga: FP[DT]):
    """The dynamics input's VJP: ga[b] += ∂/∂a_t; gs[b] = gprev[b] (+ the
    canonical VJP at s_t when `use`, i.e. t > 0: s_0 is data)."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    var gxr = gx.unsafe_offset(b * D.WM_IN)
    for k in range(n):
        ga[unsafe_offset=b * n + k] += gxr[unsafe_offset=D.S + k]
    _canon_back_lane[R](b, D.WM_IN, s, gx, gprev, gs, use)


def _k_imag_wm_back[R: TrackingRobot](nb: Int64, use: Int64, s: FP[DT], gx: FP[DT], gprev: FP[DT], gs: FP[DT], ga: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _imag_wm_back_lane[R](b, Int(use), s, gx, gprev, gs, ga)


def sm_imag_wm_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, use: Bool, s: FP[DT], gx: FP[DT], gprev: FP[DT], gs: FP[DT], ga: FP[DT]
) raises:
    var u = 1 if use else 0
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_imag_wm_back[R]](Int64(nb), Int64(u), s, gx, gprev, gs, ga, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _imag_wm_back_lane[R](b, u, s, gx, gprev, gs, ga)


# ════════════════════════════════════════════════════════════════════════════
# The critic's fit
# ════════════════════════════════════════════════════════════════════════════
@always_inline
def _gae_lane(b: Int, nb: Int, h: Int, gamma: Scalar[DT], lam: Scalar[DT], values: FP[DT], rewards: FP[DT], dhat: FP[DT], targets: FP[DT]):
    """R_t = r_{t+1} + γ (1 − d̂_{t+1}) [(1 − λ) V_{t+1} + λ R_{t+1}], R_H = V_H."""
    var r = values[unsafe_offset=h * nb + b]
    var t = h - 1
    while t >= 0:
        r = rewards[unsafe_offset=t * nb + b] + gamma * (1 - dhat[unsafe_offset=(t + 1) * nb + b]) * (
            (1 - lam) * values[unsafe_offset=(t + 1) * nb + b] + lam * r
        )
        targets[unsafe_offset=t * nb + b] = r
        t -= 1


def _k_gae(nb: Int64, h: Int64, gamma: Scalar[DT], lam: Scalar[DT], values: FP[DT], rewards: FP[DT], dhat: FP[DT], targets: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _gae_lane(b, Int(nb), Int(h), gamma, lam, values, rewards, dhat, targets)


def sm_gae[target: StaticString](
    ctx: Optional[DeviceContext], nb: Int, h: Int, gamma: Scalar[DT], lam: Scalar[DT],
    values: FP[DT], rewards: FP[DT], dhat: FP[DT], targets: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_gae](Int64(nb), Int64(h), gamma, lam, values, rewards, dhat, targets, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _gae_lane(b, nb, h, gamma, lam, values, rewards, dhat, targets)


@always_inline
def _mb_lane(i: Int, mb: Int, cin: Int, idx: FP[DT], X: FP[DT], targets: FP[DT], xin: FP[DT], y: FP[DT]):
    """Minibatch row i: gather X[idx[i]] and its target (idx as exact floats)."""
    var row = Int(idx[unsafe_offset=i])
    for k in range(cin):
        xin[unsafe_offset=i * cin + k] = X[unsafe_offset=row * cin + k]
    y[unsafe_offset=i] = targets[unsafe_offset=row]


def _k_mb(mb: Int64, cin: Int64, idx: FP[DT], X: FP[DT], targets: FP[DT], xin: FP[DT], y: FP[DT]):
    var i = Int(global_idx.x)
    if i < Int(mb):
        _mb_lane(i, Int(mb), Int(cin), idx, X, targets, xin, y)


def sm_mb_gather[target: StaticString](
    ctx: Optional[DeviceContext], mb: Int, cin: Int, idx: FP[DT], X: FP[DT], targets: FP[DT], xin: FP[DT], y: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_mb](Int64(mb), Int64(cin), idx, X, targets, xin, y, grid_dim=_grid(mb), block_dim=SM_TPB)
    else:
        for i in range(mb):
            _mb_lane(i, mb, cin, idx, X, targets, xin, y)


@always_inline
def _mse_lane(i: Int, mb: Int, pred: FP[DT], y: FP[DT], g: FP[DT], sq: FP[DT]):
    var d = pred[unsafe_offset=i] - y[unsafe_offset=i]
    sq[unsafe_offset=i] = d * d
    g[unsafe_offset=i] = Scalar[DT](2) * d / Scalar[DT](mb)


def _k_mse(mb: Int64, pred: FP[DT], y: FP[DT], g: FP[DT], sq: FP[DT]):
    var i = Int(global_idx.x)
    if i < Int(mb):
        _mse_lane(i, Int(mb), pred, y, g, sq)


def sm_mse[target: StaticString](ctx: Optional[DeviceContext], mb: Int, pred: FP[DT], y: FP[DT], g: FP[DT], sq: FP[DT]) raises:
    """The MSE's gradient (2 d / mb) and the squared errors."""
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_mse](Int64(mb), pred, y, g, sq, grid_dim=_grid(mb), block_dim=SM_TPB)
    else:
        for i in range(mb):
            _mse_lane(i, mb, pred, y, g, sq)
