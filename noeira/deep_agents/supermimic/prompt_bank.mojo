# +--------------------------------------------------------------------------+ #
# | SuperMimic's prompt interface: its data — reference commands, held commands
# +--------------------------------------------------------------------------+ #
"""What an interface's commands ARE. The code is the same for every interface;
this data is what differs (`noeira-docs/SUPERMIMIC_BFM_HYBRID_PLAN.md` §0):

                     reference command, per frame     held command, per bank row
    velocity (CMD 3) the centred 25-frame mean        uniform over the paper's ranges
                     velocity, normalised
    z (CMD 32)       the centred 25-frame mean        z of a reward from a generated
                     tracker latent, centred          family, with that reward's spec

Appendix G maps one for one: half the rollouts read a reference command (their
reward is the velocity kernel against it, and their latent is pulled towards
the motion encoder's at the same step), half hold a command.

Z. A reward is a function over frames; its z is the mean of the tracker's latent
weighted by that reward, centred on the pool's mean:

    z(r) = s · (Σ_f r(f) l(f) / Σ_f r(f) − mean_f l(f))

one pass over a fixed frame pool, for any reward; `s` puts z in units of the
windowed latent's spread. ⚠ z IS NOT PORTABLE ACROSS
POOLS: the same reward on other clips lands elsewhere (measured: the velocity
it encodes reads 0.6 m/s off, against 0.04 on the same pool), so the pool is
part of the checkpoint. r(f) MULTIPLIES a spec's kernels, on the frame's
WINDOWED velocity, where the training reward adds them on the instantaneous
state (`prompt.mojo`): an AND of sets picks the frames a compound means, and a
gait swings through any tight instantaneous kernel within a stride.

THE FAMILY. `sm_draw_spec`: velocity, posture, hands, and compounds of two.
Hand targets are taken from a frame, so they are reachable and have data. A
draw with too few matching frames (effective sample size below `SM_B_MIN_ESS`)
has no z worth the name and is dropped. `sm_spec_held_out` keeps a whole region
out of training for the zero-shot gate: walking with a hand placed.

⚠ SEED-derived when the library is: stays on private compute.
"""

from std.math import exp, sqrt
from std.random import random_float64, random_ui64

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.config import SMConfigT
from noeira.deep_agents.supermimic.kinematics import FP, sm_fk_lane
from noeira.deep_agents.supermimic.lanes import sm_enc_in, talloc, tp
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.prompt import SM_P_LEAN_DEFAULT, SMSpec, sm_prompt_atoms_lane, sm_spec_velocity
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


comptime SM_B_WINDOW_HALF: Int = 12          # the centred 25-frame mean
comptime SM_B_MIN_ESS: Float64 = 32.0
comptime SM_B_E_CUT: Float64 = 14.0
comptime SM_B_LOG_B_MAX: Float64 = 8.317766166719343   # ln 4096
comptime SM_B_SHARPEN_STEPS: Int = 9
comptime SM_B_LATENT_BATCH: Int = 1024
# Appendix G's uniform ranges, and the command's normalisation
comptime SM_B_VX_LO: Float64 = -1.5
comptime SM_B_VX_HI: Float64 = 2.0
comptime SM_B_VY: Float64 = 1.32
comptime SM_B_WZ: Float64 = 3.35
# the family's inverse variances beyond Table A.6's
comptime SM_B_IW_H: Float64 = 156.25         # σ 0.08 m
comptime SM_B_IW_TILT: Float64 = 44.44       # σ 0.15
comptime SM_B_IW_HAND: Float64 = 44.44       # σ 0.15 m
comptime SM_B_TASK_WEIGHT: Float64 = 4.0     # Σ w, Table A.6's


struct SMPromptData[CMD: Int](Movable):
    """cmd_ref [F·CMD] reference commands; vel_ref [F·3] the windowed (vx, vy,
    wz); lat [F·LATENT] the motion encoder's latent; bank [K·(CMD + SPEC)]. The
    pool (z only) keeps the interface open to rewards the bank never held: its
    frames' atoms [P·12], latents [P·CMD] and the mean latent."""
    var frames: Int
    var bank_n: Int
    var cmd_ref: TensorImpl[DT]
    var vel_ref: TensorImpl[DT]
    var lat: TensorImpl[DT]
    var bank: TensorImpl[DT]
    var pool_n: Int
    var pool_atoms: List[Scalar[DT]]
    var pool_lat: List[Scalar[DT]]
    var lbar: List[Scalar[DT]]
    var z_scale: Float64
    var z_beta: Float64            # z retrieves with r^beta: 1 is the reward itself
    var z_ess: Float64             # > 0: each reward sharpens to this effective sample size

    def __init__(out self, frames: Int, bank_n: Int, latent: Int):
        self.frames = frames
        self.bank_n = bank_n
        self.cmd_ref = TensorImpl[DT].alloc(frames * Self.CMD)
        self.vel_ref = TensorImpl[DT].alloc(frames * 3)
        self.lat = TensorImpl[DT].alloc(frames * latent)
        self.bank = TensorImpl[DT].alloc(bank_n * (Self.CMD + SMSpec.SIZE))
        self.pool_n = 0
        self.pool_atoms = List[Scalar[DT]]()
        self.pool_lat = List[Scalar[DT]]()
        self.lbar = List[Scalar[DT]]()
        self.z_scale = 1.0
        self.z_beta = 1.0
        self.z_ess = 0.0

    def __init__(out self, *, deinit move: Self):
        self.frames = move.frames
        self.bank_n = move.bank_n
        self.cmd_ref = move.cmd_ref^
        self.vel_ref = move.vel_ref^
        self.lat = move.lat^
        self.bank = move.bank^
        self.pool_n = move.pool_n
        self.pool_atoms = move.pool_atoms^
        self.pool_lat = move.pool_lat^
        self.lbar = move.lbar^
        self.z_scale = move.z_scale
        self.z_beta = move.z_beta
        self.z_ess = move.z_ess

    def z_of(mut self, spec: FP[DT], z: FP[DT]) -> Float64:
        """z of any reward over the pool, in command units; returns its
        effective sample size."""
        var ess = sm_z_of_spec(
            spec, self.pool_atoms.unsafe_ptr().as_unsafe_any_origin(), self.pool_lat.unsafe_ptr().as_unsafe_any_origin(),
            self.lbar.unsafe_ptr().as_unsafe_any_origin(), self.pool_n, Self.CMD, z, self.z_beta, self.z_ess,
        )
        if ess >= 1:
            for k in range(Self.CMD):
                z[unsafe_offset=k] *= Scalar[DT](self.z_scale)
        return ess

    def to_device(mut self, ctx: DeviceContext) raises:
        self.cmd_ref.upload_resident(ctx)
        self.vel_ref.upload_resident(ctx)
        self.lat.upload_resident(ctx)
        self.bank.upload_resident(ctx)


# ── per-frame quantities ────────────────────────────────────────────────────
def sm_frame_atoms[R: TrackingRobot](mut lib: SMMotionLibrary[R], mut tb: SMTables[DT]) -> List[Scalar[DT]]:
    """The 12 atoms of every library frame [F·12], instantaneous."""
    comptime D = SMDims[R]
    var nf = lib.num_frames
    var out = List[Scalar[DT]](length=nf * SMSpec.ATOMS, fill=0)
    var s = List[Scalar[DT]](length=D.S, fill=0)
    var fk = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var sp = s.unsafe_ptr().as_unsafe_any_origin()
    var pos = fk.unsafe_ptr().as_unsafe_any_origin()
    var quat = pos.unsafe_offset(D.NB * 3)
    var lin = pos.unsafe_offset(D.NB * 7)
    var ang = pos.unsafe_offset(D.NB * 10)
    var op = out.unsafe_ptr().as_unsafe_any_origin()
    for f in range(nf):
        lib.ref_qpos_qvel(f, sp, sp.unsafe_offset(D.NQ))
        sm_fk_lane[DT, D.NB](tb.tree_ptr(), sp, sp.unsafe_offset(D.NQ), pos, quat, lin, ang)
        sm_prompt_atoms_lane[DT, R](tb.head_ptr(), sp, pos, op.unsafe_offset(f * SMSpec.ATOMS))
    return out^


def sm_window_mean[R: TrackingRobot](
    lib: SMMotionLibrary[R], src: List[Scalar[DT]], row: Int, col: Int, dim: Int, half: Int = SM_B_WINDOW_HALF
) -> List[Scalar[DT]]:
    """The centred (2·half + 1)-frame mean of columns col .. col + dim of `src`
    (rows of width `row`), per clip: the window shrinks at a clip's ends and
    never crosses one. Accumulates in Float64: a float32 running sum drifts
    with the row count."""
    var nf = lib.num_frames
    var out = List[Scalar[DT]](length=nf * dim, fill=0)
    var acc = List[Float64](length=dim, fill=0.0)
    for c in range(lib.num_clips):
        var st = lib.clip_start[c]
        var ln = lib.clip_len[c]
        var cum = List[Float64](length=(ln + 1) * dim, fill=0.0)
        for i in range(ln):
            for k in range(dim):
                acc[k] = cum[i * dim + k] + Float64(src[(st + i) * row + col + k])
                cum[(i + 1) * dim + k] = acc[k]
        for i in range(ln):
            var lo = max(i - half, 0)
            var hi = min(i + half + 1, ln)
            for k in range(dim):
                out[(st + i) * dim + k] = Scalar[DT]((cum[hi * dim + k] - cum[lo * dim + k]) / Float64(hi - lo))
    return out^


def sm_frame_latents[C: SMConfigT](mut tr: SuperMimicTrainer[C], mut lib: SMMotionLibrary[C.ROBOT]) raises -> List[Scalar[DT]]:
    """The motion encoder's latent at every library frame [F·LATENT]."""
    comptime T = C.TARGET
    comptime R = C.ROBOT
    comptime D = SMDims[R]
    comptime N = SM_B_LATENT_BATCH
    var ctx = tr.ctx
    var nf = lib.num_frames
    var out = List[Scalar[DT]](length=nf * D.LATENT, fill=0)
    var goals = TensorImpl[DT].alloc(N * D.GOAL)
    comptime if T == "gpu":
        goals.ensure_gpu(ctx.value(), N * D.GOAL)
    var ein = talloc[T](N * D.GOAL, ctx)
    var z = talloc[T](N * D.LATENT, ctx)
    var f = 0
    while f < nf:
        var take = min(N, nf - f)
        for i in range(take):
            lib.goal_window(f + i, goals.data.unsafe_ptr().unsafe_offset(i * D.GOAL).as_unsafe_any_origin())
        comptime if T == "gpu":
            goals.upload_resident(ctx.value())
        sm_enc_in[T, R](ctx, N, tp[T](goals), tp[T](ein))
        tr.pol.enc.forward[T, N](TensorRefs[1, _, DT](ein), z, ctx)
        comptime if T == "gpu":
            z.download(ctx.value())
        for i in range(take * D.LATENT):
            out[f * D.LATENT + i] = z.data[i]
        f += take
    return out^


# ── the family of rewards ───────────────────────────────────────────────────
def _active(spec: FP[DT], g: Int) -> Bool:
    return spec[unsafe_offset=SMSpec.W + g] > 0


def sm_spec_held_out(spec: FP[DT]) -> Bool:
    """Walking with a hand placed: the region the zero-shot gate scores and
    training never sees."""
    var vx = spec[unsafe_offset=SMSpec.T + SMSpec.A_VX]
    var vy = spec[unsafe_offset=SMSpec.T + SMSpec.A_VY]
    var moving = _active(spec, SMSpec.G_VEL) and vx * vx + vy * vy > Scalar[DT](0.09)
    return moving and (_active(spec, SMSpec.G_LHAND) or _active(spec, SMSpec.G_RHAND))


def _spec_velocity_part(spec: FP[DT]):
    var v = random_float64(0.0, 1.0)
    var vx = 0.0 if v < 0.1 else random_float64(SM_B_VX_LO, SM_B_VX_HI)
    var vy = 0.0 if v < 0.1 else random_float64(-SM_B_VY, SM_B_VY)
    var wz = 0.0 if v < 0.1 else random_float64(-SM_B_WZ, SM_B_WZ)
    spec[unsafe_offset=SMSpec.T + SMSpec.A_VX] = Scalar[DT](vx)
    spec[unsafe_offset=SMSpec.T + SMSpec.A_VY] = Scalar[DT](vy)
    spec[unsafe_offset=SMSpec.T + SMSpec.A_WZ] = Scalar[DT](wz)
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_VX] = 4
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_VY] = 4
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_WZ] = 2
    spec[unsafe_offset=SMSpec.W + SMSpec.G_VEL] = 1
    spec[unsafe_offset=SMSpec.W + SMSpec.G_YAW] = 1


def _spec_height_part(spec: FP[DT]):
    spec[unsafe_offset=SMSpec.T + SMSpec.A_H] = Scalar[DT](random_float64(0.45, 0.82))
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_H] = Scalar[DT](SM_B_IW_H)
    spec[unsafe_offset=SMSpec.W + SMSpec.G_H] = 1


def _spec_tilt_part(spec: FP[DT]):
    spec[unsafe_offset=SMSpec.T + SMSpec.A_GX] = Scalar[DT](random_float64(-0.25, 0.25))
    spec[unsafe_offset=SMSpec.T + SMSpec.A_GY] = Scalar[DT](random_float64(-0.25, 0.25))
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_GX] = Scalar[DT](SM_B_IW_TILT)
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_GY] = Scalar[DT](SM_B_IW_TILT)
    spec[unsafe_offset=SMSpec.W + SMSpec.G_TILT] = 1
    spec[unsafe_offset=SMSpec.LEAN] = 0                       # it asks for what the lean price forbids


def _spec_hand_part(spec: FP[DT], pool: FP[DT], f: Int, left: Bool):
    var a0 = SMSpec.A_LX if left else SMSpec.A_RX
    var g = SMSpec.G_LHAND if left else SMSpec.G_RHAND
    for k in range(3):
        spec[unsafe_offset=SMSpec.T + a0 + k] = pool[unsafe_offset=f * SMSpec.ATOMS + a0 + k]
        spec[unsafe_offset=SMSpec.IW + a0 + k] = Scalar[DT](SM_B_IW_HAND)
    spec[unsafe_offset=SMSpec.W + g] = 1


def sm_draw_spec(spec: FP[DT], pool: FP[DT], n_pool: Int):
    """One reward from the family: velocity 35 %, posture 20 %, hands 20 %,
    compounds of two 25 %. Its group weights share `SM_B_TASK_WEIGHT`."""
    for k in range(SMSpec.SIZE):
        spec[unsafe_offset=k] = 0
    spec[unsafe_offset=SMSpec.LEAN] = Scalar[DT](SM_P_LEAN_DEFAULT)
    var u = random_float64(0.0, 1.0)
    var f = Int(random_ui64(0, UInt64(n_pool - 1)))
    if u < 0.35:
        _spec_velocity_part(spec)
    elif u < 0.55:
        var v = random_float64(0.0, 1.0)
        if v < 0.6:
            _spec_height_part(spec)
        if v >= 0.4:
            _spec_tilt_part(spec)
    elif u < 0.75:
        var v = random_float64(0.0, 1.0)
        if v < 0.65:
            _spec_hand_part(spec, pool, f, True)
        if v >= 0.35:
            _spec_hand_part(spec, pool, f, False)
    else:
        var v = random_float64(0.0, 1.0)
        if v < 0.4:
            _spec_velocity_part(spec)
            _spec_height_part(spec)
        elif v < 0.7:
            _spec_height_part(spec)
            _spec_hand_part(spec, pool, f, random_float64(0.0, 1.0) < 0.5)
        else:
            _spec_velocity_part(spec)
            _spec_hand_part(spec, pool, f, random_float64(0.0, 1.0) < 0.5)
    var ng = 0
    for g in range(SMSpec.GROUPS):
        if _active(spec, g):
            ng += 1
    var w = Scalar[DT](SM_B_TASK_WEIGHT / Float64(max(ng, 1)))
    for g in range(SMSpec.GROUPS):
        if _active(spec, g):
            spec[unsafe_offset=SMSpec.W + g] = w


def _ess_of(es: List[Float64], emin: Float64, b: Float64) -> Float64:
    """The effective sample size of `exp(-b (e - emin))`."""
    var s1 = 0.0
    var s2 = 0.0
    for i in range(len(es)):
        var x = b * (es[i] - emin)
        if x > SM_B_E_CUT:
            continue
        var r = exp(-x)
        s1 += r
        s2 += r * r
    return s1 * s1 / max(s2, 1e-300)


def _sharpest(es: List[Float64], emin: Float64, target: Float64) -> Float64:
    """The largest b in [1, e^SM_B_LOG_B_MAX] with ESS >= `target`; 1 if even
    b = 1 falls short, so a reward the pool barely holds is left as it is."""
    if _ess_of(es, emin, 1.0) <= target:
        return 1.0
    if _ess_of(es, emin, exp(SM_B_LOG_B_MAX)) >= target:
        return exp(SM_B_LOG_B_MAX)
    var lo = 0.0
    var hi = SM_B_LOG_B_MAX
    for _ in range(SM_B_SHARPEN_STEPS):
        var mid = 0.5 * (lo + hi)
        if _ess_of(es, emin, exp(mid)) >= target:
            lo = mid
        else:
            hi = mid
    return exp(lo)


def sm_z_of_spec(
    spec: FP[DT], pool_atoms: FP[DT], pool_lat: FP[DT], lbar: FP[DT], n_pool: Int, latent: Int, z: FP[DT],
    beta: Float64 = 1.0, ess_target: Float64 = 0.0,
) -> Float64:
    """z(r^beta) over the pool for the PRODUCT of the spec's kernels (module doc);
    returns the effective sample size (Σr)² / Σr². `z` is untouched below 1.

    Most of the motion stands still, so at beta = 1 a slow command retrieves
    mostly standing. `beta` > 1 sharpens the retrieval and leaves the reward alone,
    but one beta cannot suit every reward: 8 loses most large commands. With
    `ess_target` > 0 each reward sharpens as far as the pool lets it."""
    var es = List[Float64]()
    var fs = List[Int]()
    for f in range(n_pool):
        var e = 0.0
        for a in range(SMSpec.ATOMS):
            var iw = Float64(spec[unsafe_offset=SMSpec.IW + a])
            if iw > 0:
                var d = Float64(pool_atoms[unsafe_offset=f * SMSpec.ATOMS + a]) - Float64(spec[unsafe_offset=SMSpec.T + a])
                e += beta * iw * d * d
                if e > SM_B_E_CUT:      # e⁻¹⁴ ~ 1e-6: this frame says nothing
                    break
        if e > SM_B_E_CUT:
            continue
        es.append(e)
        fs.append(f)
    if len(es) == 0:
        return 0.0
    var emin = 0.0
    var b = 1.0
    if ess_target > 0:
        emin = es[0]
        for i in range(len(es)):
            emin = min(emin, es[i])
        b = _sharpest(es, emin, ess_target)
    var s1 = 0.0
    var s2 = 0.0
    var acc = List[Float64](length=latent, fill=0.0)
    for i in range(len(es)):
        var x = b * (es[i] - emin)
        if x > SM_B_E_CUT:
            continue
        var r = exp(-x)
        s1 += r
        s2 += r * r
        for k in range(latent):
            acc[k] += r * Float64(pool_lat[unsafe_offset=fs[i] * latent + k])
    if s1 <= 0:
        return 0.0
    var ess = s1 * s1 / s2
    if ess >= 1:
        for k in range(latent):
            z[unsafe_offset=k] = Scalar[DT](acc[k] / s1) - lbar[unsafe_offset=k]
    return ess


# ── the two interfaces' data ────────────────────────────────────────────────
def _common[C: SMConfigT, CMD: Int](
    mut tr: SuperMimicTrainer[C], mut lib: SMMotionLibrary[C.ROBOT], bank_n: Int, mut atoms: List[Scalar[DT]], mut lat: List[Scalar[DT]],
) raises -> SMPromptData[CMD]:
    """The tables both interfaces share: the windowed velocity and the latent."""
    comptime D = SMDims[C.ROBOT]
    var d = SMPromptData[CMD](lib.num_frames, bank_n, D.LATENT)
    atoms = sm_frame_atoms[C.ROBOT](lib, tr.tables)
    lat = sm_frame_latents[C](tr, lib)
    var vel = sm_window_mean[C.ROBOT](lib, atoms, SMSpec.ATOMS, SMSpec.A_VX, 3)
    # the windowed velocity replaces the instantaneous one in the frames' atoms
    for f in range(lib.num_frames):
        for k in range(3):
            d.vel_ref.data[f * 3 + k] = vel[f * 3 + k]
            atoms[f * SMSpec.ATOMS + SMSpec.A_VX + k] = vel[f * 3 + k]
    d.lat.data = lat.copy()
    return d^


def _fill_velocity[CMD: Int](mut d: SMPromptData[CMD], frames: Int, bank_n: Int):
    """The paper's velocity interface (the positive control)."""
    comptime RW = CMD + SMSpec.SIZE
    var sv = Scalar[DT](1.0 / SM_B_VX_HI)
    var sy = Scalar[DT](1.0 / SM_B_VY)
    var sw = Scalar[DT](1.0 / SM_B_WZ)
    for f in range(frames):
        d.cmd_ref.data[f * CMD] = d.vel_ref.data[f * 3] * sv
        d.cmd_ref.data[f * CMD + 1] = d.vel_ref.data[f * 3 + 1] * sy
        d.cmd_ref.data[f * CMD + 2] = d.vel_ref.data[f * 3 + 2] * sw
    var bp = d.bank.data.unsafe_ptr().as_unsafe_any_origin()
    for k in range(bank_n):
        var u = random_float64(0.0, 1.0)
        var vx = 0.0 if u < 0.1 else random_float64(SM_B_VX_LO, SM_B_VX_HI)
        var vy = 0.0 if u < 0.1 else random_float64(-SM_B_VY, SM_B_VY)
        var wz = 0.0 if u < 0.1 else random_float64(-SM_B_WZ, SM_B_WZ)
        d.bank.data[k * RW] = Scalar[DT](vx) * sv
        d.bank.data[k * RW + 1] = Scalar[DT](vy) * sy
        d.bank.data[k * RW + 2] = Scalar[DT](wz) * sw
        sm_spec_velocity[DT](bp.unsafe_offset(k * RW + CMD), Scalar[DT](vx), Scalar[DT](vy), Scalar[DT](wz))


def _fill_z[R: TrackingRobot, CMD: Int](
    mut d: SMPromptData[CMD], lib: SMMotionLibrary[R], bank_n: Int, pool_n: Int, atoms: List[Scalar[DT]], lat: List[Scalar[DT]], hold_out: Bool = True,
):
    """The reward-derived z: the module's first table."""
    comptime RW = CMD + SMSpec.SIZE
    var nf = lib.num_frames
    var mean = List[Float64](length=CMD, fill=0.0)
    for f in range(nf):
        for k in range(CMD):
            mean[k] += Float64(lat[f * CMD + k])
    var lbar = List[Scalar[DT]](length=CMD, fill=0)
    for k in range(CMD):
        lbar[k] = Scalar[DT](mean[k] / Float64(nf))
    # the reference command: the windowed latent, centred, in units of its own
    # spread (~0.2 raw, beside a history whose entries are ~1: unscaled, a fresh
    # encoder would barely read its command)
    var lw = sm_window_mean[R](lib, lat, CMD, 0, CMD)
    var ss = 0.0
    for i in range(nf):
        for k in range(CMD):
            var e = Float64(lw[i * CMD + k]) - Float64(lbar[k])
            ss += e * e
    d.z_scale = 1.0 / max(sqrt(ss / Float64(nf * CMD)), 1e-6)
    for i in range(nf):
        for k in range(CMD):
            d.cmd_ref.data[i * CMD + k] = (lw[i * CMD + k] - lbar[k]) * Scalar[DT](d.z_scale)
    # the pool: every `stride`-th frame
    var stride = max(nf // pool_n, 1)
    var np = nf // stride
    var pa = List[Scalar[DT]](length=np * SMSpec.ATOMS, fill=0)
    var pl = List[Scalar[DT]](length=np * CMD, fill=0)
    for p in range(np):
        for k in range(SMSpec.ATOMS):
            pa[p * SMSpec.ATOMS + k] = atoms[p * stride * SMSpec.ATOMS + k]
        for k in range(CMD):
            pl[p * CMD + k] = lat[p * stride * CMD + k]
    d.pool_n = np
    d.pool_atoms = pa^
    d.pool_lat = pl^
    d.lbar = lbar^
    var bp = d.bank.data.unsafe_ptr().as_unsafe_any_origin()
    var pap = d.pool_atoms.unsafe_ptr().as_unsafe_any_origin()
    var k = 0
    var dropped_ess = 0
    var dropped_held = 0
    while k < bank_n:
        var row = bp.unsafe_offset(k * RW)
        sm_draw_spec(row.unsafe_offset(CMD), pap, np)
        if hold_out and sm_spec_held_out(row.unsafe_offset(CMD)):
            dropped_held += 1
            continue
        var ess = d.z_of(row.unsafe_offset(CMD), row)
        if ess < SM_B_MIN_ESS:
            dropped_ess += 1
            continue
        k += 1
    print("prompt bank:", bank_n, "rewards over", np, "frames;", dropped_ess, "dropped for too few frames,", dropped_held, "held out; z ×", d.z_scale)


def sm_prompt_data[C: SMConfigT, CMD: Int](
    mut tr: SuperMimicTrainer[C], mut lib: SMMotionLibrary[C.ROBOT], bank_n: Int, pool_n: Int = 131072, z_beta: Float64 = 1.0, z_ess: Float64 = 0.0, hold_out: Bool = True
) raises -> SMPromptData[CMD]:
    """An interface's tables from a tracker and a library: the velocity
    command when CMD is 3, a reward-derived z when CMD is the latent's width."""
    var atoms = List[Scalar[DT]]()
    var lat = List[Scalar[DT]]()
    if z_ess > 0 and z_ess < SM_B_MIN_ESS:
        raise Error("sm_prompt_data: z_ess " + String(z_ess) + " sharpens rewards below the bank's floor of " + String(SM_B_MIN_ESS))
    var d = _common[C, CMD](tr, lib, bank_n, atoms, lat)
    d.z_beta = z_beta
    d.z_ess = z_ess
    comptime if CMD == 3:
        _fill_velocity[CMD](d, lib.num_frames, bank_n)
    else:
        comptime assert CMD == SMDims[C.ROBOT].LATENT, "CMD is 3 (velocity) or the latent's width (z)"
        _fill_z[C.ROBOT, CMD](d, lib, bank_n, pool_n, atoms, lat, hold_out)
    return d^
