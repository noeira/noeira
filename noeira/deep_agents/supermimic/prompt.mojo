# +--------------------------------------------------------------------------+ #
# | SuperMimic's prompt interface: the task reward and the fall logit
# +--------------------------------------------------------------------------+ #
"""What a prompt asks of the robot, as an exact differentiable function of the
imagined state, and its VJP. `noeira-docs/SUPERMIMIC_BFM_HYBRID_PLAN.md` §0.

A prompt interface is a new encoder over the tracker's frozen decoder, trained
through the world model on its own reward (the paper's Appendix B and G). The
paper's task is velocity tracking; this reward generalises Table A.6 from one
command to a SPEC per lane: kernels over a fixed set of ATOMS,

    atoms (12)  vx, vy     root velocity, heading frame
                wz         world yaw rate
                h          root height
                gx, gy     projected gravity, root frame
                lx ly lz   left hand relative to the root, heading frame
                rx ry rz   right hand, same

    r = dt · [ Σ_g w_g exp(−Σ_{a in g} iw_a (a − t_a)²)      the task
               − 0.3 |a_t − a_{t−1}|²                         action smoothness
               − 2 Σ_j (relu(lo − q) + relu(q − hi))          joint limits
               − w_lean sin²φ                                 sideways torso lean
               − 0.05 (ω_roll² + ω_pitch²) ]                  torso rates

    groups g    {vx vy} {wz} {h} {gx gy} {left hand} {right hand}

Table A.6 is the spec w = (2, 2, 0, 0, 0, 0), iw = (1/0.5², 1/0.5², 1/0.5, 0…),
w_lean = 4. A spec is [t (12) | iw (12) | w (6) | w_lean]. `w_lean` is per spec
because a tilt prompt asks for what that term forbids.

THE KERNELS ADD UP where a prompt's z multiplies them (`prompt_bank.mojo`): a
product has no gradient while any one kernel is far from its target, and a
sum's maximum is the product's whenever the targets can be met together.

THE FALL LOGIT replaces the tracker's learned termination head, whose inputs
are errors against a reference a held prompt does not have. Table A.6's rule
(h < 0.35 m or 1 + u_z > 0.7) as one logit, so the rollout's closed-form
survival chain applies unchanged:

    logit = max((0.35 − h) / 0.02, (1 + u_z − 0.7) / 0.05)

At the stand it is about −21 and the survival is flat; at 0.45 m about −5.

Pointer-in / pointer-out, no allocation, no Float64 at runtime: the bodies of
GPU kernels (`lanes.mojo`'s rules). Scratch sizes are `SMPromptScratch[R]`.
"""

from std.math import exp

from noeira.deep_agents.supermimic.heads import (
    SMHeadLayout, qapply_inv, qapply_inv_vjp_q, sm_observation_vjp_lane, yaw_quat, yaw_quat_vjp,
)
from noeira.deep_agents.supermimic.kinematics import FP, Q4, qapply, qapply_vjp, sm_fk_lane, sm_fk_vjp_lane
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.state_spec import qapply_inv_vjp


comptime SM_P_DT: Float64 = 0.02
comptime SM_P_W_ACTION_RATE: Float64 = -0.3
comptime SM_P_W_JOINT_LIMIT: Float64 = -2.0
comptime SM_P_W_TORSO_RATE: Float64 = -0.05
comptime SM_P_LEAN_DEFAULT: Float64 = 4.0
comptime SM_P_FALL_H: Float64 = 0.35
comptime SM_P_FALL_TILT: Float64 = 0.7
comptime SM_P_FALL_TAU_H: Float64 = 0.02
comptime SM_P_FALL_TAU_TILT: Float64 = 0.05


struct SMSpec:
    """A spec's layout, and the atoms' indices."""
    comptime ATOMS = 12
    comptime GROUPS = 6
    comptime T = 0
    comptime IW = Self.ATOMS
    comptime W = 2 * Self.ATOMS
    comptime LEAN = Self.W + Self.GROUPS
    comptime SIZE = Self.LEAN + 1
    comptime A_VX = 0
    comptime A_VY = 1
    comptime A_WZ = 2
    comptime A_H = 3
    comptime A_GX = 4
    comptime A_GY = 5
    comptime A_LX = 6
    comptime A_RX = 9
    comptime G_VEL = 0
    comptime G_YAW = 1
    comptime G_H = 2
    comptime G_TILT = 3
    comptime G_LHAND = 4
    comptime G_RHAND = 5


@always_inline
def sm_spec_group(a: Int) -> Int:
    """The kernel group of atom `a`."""
    if a < 2:
        return SMSpec.G_VEL
    if a < 4:
        return SMSpec.G_YAW if a == 2 else SMSpec.G_H
    if a < 6:
        return SMSpec.G_TILT
    return SMSpec.G_LHAND if a < 9 else SMSpec.G_RHAND


@always_inline
def sm_spec_velocity[dt: DType](spec: FP[dt], vx: Scalar[dt], vy: Scalar[dt], wz: Scalar[dt]):
    """Table A.6 for the command (vx, vy, wz): writes the whole spec."""
    for k in range(SMSpec.SIZE):
        spec[unsafe_offset=k] = 0
    spec[unsafe_offset=SMSpec.T + SMSpec.A_VX] = vx
    spec[unsafe_offset=SMSpec.T + SMSpec.A_VY] = vy
    spec[unsafe_offset=SMSpec.T + SMSpec.A_WZ] = wz
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_VX] = 4         # 1 / 0.5²
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_VY] = 4
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_WZ] = 2         # 1 / 0.5
    spec[unsafe_offset=SMSpec.W + SMSpec.G_VEL] = 2
    spec[unsafe_offset=SMSpec.W + SMSpec.G_YAW] = 2
    spec[unsafe_offset=SMSpec.LEAN] = Scalar[dt](SM_P_LEAN_DEFAULT)


struct SMPromptScratch[R: TrackingRobot]:
    """One lane's scratch for the reward: the heads' FK block and cotangents,
    then the atoms and their cotangents."""
    comptime ATOMS = SMHeadLayout[Self.R].SCRATCH
    comptime GATOMS = Self.ATOMS + SMSpec.ATOMS
    comptime SIZE = Self.GATOMS + SMSpec.ATOMS


@always_inline
def _at[dt: DType](p: FP[dt], i: Int) -> Scalar[dt]:
    return p[unsafe_offset=i]


@always_inline
def _v3[dt: DType](p: FP[dt], i: Int) -> Q4[dt]:
    return Q4[dt](p[unsafe_offset=i], p[unsafe_offset=i + 1], p[unsafe_offset=i + 2], 0)


@always_inline
def _q4[dt: DType](p: FP[dt], i: Int) -> Q4[dt]:
    return Q4[dt](p[unsafe_offset=i], p[unsafe_offset=i + 1], p[unsafe_offset=i + 2], p[unsafe_offset=i + 3])


@always_inline
def _add3[dt: DType](p: FP[dt], i: Int, v: Q4[dt]):
    p[unsafe_offset=i] += v[0]
    p[unsafe_offset=i + 1] += v[1]
    p[unsafe_offset=i + 2] += v[2]


@always_inline
def _add4[dt: DType](p: FP[dt], i: Int, v: Q4[dt]):
    p[unsafe_offset=i] += v[0]
    p[unsafe_offset=i + 1] += v[1]
    p[unsafe_offset=i + 2] += v[2]
    p[unsafe_offset=i + 3] += v[3]


@always_inline
def _grav[dt: DType]() -> Q4[dt]:
    return Q4[dt](0, 0, -1, 0)


@always_inline
def _ez[dt: DType]() -> Q4[dt]:
    return Q4[dt](0, 0, 1, 0)


# ── the atoms ───────────────────────────────────────────────────────────────
def sm_prompt_atoms_lane[dt: DType, R: TrackingRobot](table: FP[dt], s: FP[dt], pos: FP[dt], atoms: FP[dt]) where dt.is_floating_point():
    """The 12 atoms of state `s`, whose FK's body positions are `pos`."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    var q = _q4(s, 3)
    var hq = yaw_quat(q)
    var vh = qapply_inv(hq, _v3(s, D.NQ))
    var g = qapply_inv(q, _grav[dt]())
    var rp = _v3(s, 0)
    var pl = qapply_inv(hq, _v3(pos, Int(_at(table, L.HANDS)) * 3) - rp)
    var pr = qapply_inv(hq, _v3(pos, Int(_at(table, L.HANDS + 1)) * 3) - rp)
    atoms[unsafe_offset=SMSpec.A_VX] = vh[0]
    atoms[unsafe_offset=SMSpec.A_VY] = vh[1]
    atoms[unsafe_offset=SMSpec.A_WZ] = qapply(q, _v3(s, D.NQ + 3))[2]
    atoms[unsafe_offset=SMSpec.A_H] = _at(s, 2)
    atoms[unsafe_offset=SMSpec.A_GX] = g[0]
    atoms[unsafe_offset=SMSpec.A_GY] = g[1]
    comptime for c in range(3):
        atoms[unsafe_offset=SMSpec.A_LX + c] = pl[c]
        atoms[unsafe_offset=SMSpec.A_RX + c] = pr[c]


@always_inline
def _kernel[dt: DType](atoms: FP[dt], spec: FP[dt], a0: Int, n: Int) -> Scalar[dt] where dt.is_floating_point():
    """exp(−Σ iw (a − t)²) over atoms a0 .. a0 + n."""
    var e = Scalar[dt](0)
    for k in range(a0, a0 + n):
        var d = _at(atoms, k) - _at(spec, SMSpec.T + k)
        e += _at(spec, SMSpec.IW + k) * d * d
    return exp(-e)


@always_inline
def _task[dt: DType](atoms: FP[dt], spec: FP[dt]) -> Scalar[dt] where dt.is_floating_point():
    """Σ_g w_g k_g, the task's undiscounted per-second value."""
    var r = _at(spec, SMSpec.W + SMSpec.G_VEL) * _kernel(atoms, spec, SMSpec.A_VX, 2)
    r += _at(spec, SMSpec.W + SMSpec.G_YAW) * _kernel(atoms, spec, SMSpec.A_WZ, 1)
    r += _at(spec, SMSpec.W + SMSpec.G_H) * _kernel(atoms, spec, SMSpec.A_H, 1)
    r += _at(spec, SMSpec.W + SMSpec.G_TILT) * _kernel(atoms, spec, SMSpec.A_GX, 2)
    r += _at(spec, SMSpec.W + SMSpec.G_LHAND) * _kernel(atoms, spec, SMSpec.A_LX, 3)
    r += _at(spec, SMSpec.W + SMSpec.G_RHAND) * _kernel(atoms, spec, SMSpec.A_RX, 3)
    return r


@always_inline
def sm_prompt_fall_logit[dt: DType](s: FP[dt]) -> Scalar[dt] where dt.is_floating_point():
    var uz = qapply_inv(_q4(s, 3), _grav[dt]())[2]
    var x0 = (Scalar[dt](SM_P_FALL_H) - _at(s, 2)) / Scalar[dt](SM_P_FALL_TAU_H)
    var x1 = (Scalar[dt](1) + uz - Scalar[dt](SM_P_FALL_TILT)) / Scalar[dt](SM_P_FALL_TAU_TILT)
    return max(x0, x1)


@always_inline
def sm_prompt_fell[dt: DType](s: FP[dt]) -> Bool where dt.is_floating_point():
    """Table A.6's termination: the sign of the fall logit."""
    return sm_prompt_fall_logit[dt](s) > 0


# ── forward ─────────────────────────────────────────────────────────────────
def sm_prompt_reward_lane[dt: DType, R: TrackingRobot](
    tree: FP[dt], table: FP[dt], s: FP[dt], a: FP[dt], a_prev: FP[dt], spec: FP[dt], scratch: FP[dt],
) -> Scalar[dt] where dt.is_floating_point():
    """The reward of reaching `s` (= s_{t+1}) by `a` after `a_prev` under
    `spec`. `scratch` holds `SMPromptScratch[R].SIZE`; its FK block and atoms
    are what `sm_prompt_reward_vjp_lane` reads back."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    comptime SC = SMPromptScratch[R]
    var pos = scratch
    var quat = scratch.unsafe_offset(D.NB * 3)
    var lin = scratch.unsafe_offset(D.NB * 7)
    var ang = scratch.unsafe_offset(D.NB * 10)
    sm_fk_lane[dt, D.NB](tree, s, s.unsafe_offset(D.NQ), pos, quat, lin, ang)
    var atoms = scratch.unsafe_offset(SC.ATOMS)
    sm_prompt_atoms_lane[dt, R](table, s, pos, atoms)
    var r = _task(atoms, spec)
    var ar = Scalar[dt](0)
    var jl = Scalar[dt](0)
    for j in range(D.NDOF):
        var d = _at(a, j) - _at(a_prev, j)
        ar += d * d
        var q = _at(s, 7 + j)
        jl += max(_at(table, L.LO + j) - q, Scalar[dt](0)) + max(q - _at(table, L.HI + j), Scalar[dt](0))
    r += Scalar[dt](SM_P_W_ACTION_RATE) * ar + Scalar[dt](SM_P_W_JOINT_LIMIT) * jl
    # the anchor body is the torso on a humanoid
    var ti = Int(_at(table, L.TRACKED + Int(_at(table, L.ANCHOR))))
    var hq = yaw_quat(_q4(s, 3))
    var uh = qapply_inv(hq, qapply(_q4(quat, ti * 4), _ez[dt]()))
    var wh = qapply_inv(hq, _v3(ang, ti * 3))
    r -= _at(spec, SMSpec.LEAN) * uh[1] * uh[1]
    r += Scalar[dt](SM_P_W_TORSO_RATE) * (wh[0] * wh[0] + wh[1] * wh[1])
    return Scalar[dt](SM_P_DT) * r


# ── VJP ─────────────────────────────────────────────────────────────────────
@always_inline
def _kernel_vjp[dt: DType](atoms: FP[dt], spec: FP[dt], a0: Int, n: Int, g: Scalar[dt], gat: FP[dt]) where dt.is_floating_point():
    """gat[a] = ∂/∂a of g · exp(−Σ iw (a − t)²), the group's w inside g."""
    var k = _kernel(atoms, spec, a0, n)
    for i in range(a0, a0 + n):
        gat[unsafe_offset=i] = -g * k * 2 * _at(spec, SMSpec.IW + i) * (_at(atoms, i) - _at(spec, SMSpec.T + i))


def sm_prompt_reward_vjp_lane[dt: DType, R: TrackingRobot](
    tree: FP[dt], table: FP[dt], s: FP[dt], a: FP[dt], a_prev: FP[dt], spec: FP[dt],
    g_r: Scalar[dt], g_lg: Scalar[dt], g_obs: FP[dt], scratch: FP[dt],
    g_s: FP[dt], g_a: FP[dt], g_ap: FP[dt],
) where dt.is_floating_point():
    """`g_s` (S), `g_a` (n), `g_ap` (n) ← the VJP of (reward, fall logit,
    observation) for the cotangents `g_r`, `g_lg`, `g_obs`. `scratch` must hold
    `sm_prompt_reward_lane`'s FK and atoms; the rest is overwritten. Outputs
    are written, not added to."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    comptime SC = SMPromptScratch[R]
    var pos = scratch
    var quat = scratch.unsafe_offset(D.NB * 3)
    var ang = scratch.unsafe_offset(D.NB * 10)
    var gpos = scratch.unsafe_offset(D.FK_BLOCK)
    var gquat = gpos.unsafe_offset(D.NB * 3)
    var glin = gpos.unsafe_offset(D.NB * 7)
    var gang = gpos.unsafe_offset(D.NB * 10)
    var gqpos = gpos.unsafe_offset(D.FK_BLOCK)
    var gqvel = gqpos.unsafe_offset(D.NQ)
    var atoms = scratch.unsafe_offset(SC.ATOMS)
    var gat = scratch.unsafe_offset(SC.GATOMS)
    for i in range(D.FK_BLOCK):
        gpos[unsafe_offset=i] = 0
    for i in range(D.S):
        g_s[unsafe_offset=i] = 0

    # the task, onto the atoms
    var gdt = g_r * Scalar[dt](SM_P_DT)
    _kernel_vjp(atoms, spec, SMSpec.A_VX, 2, gdt * _at(spec, SMSpec.W + SMSpec.G_VEL), gat)
    _kernel_vjp(atoms, spec, SMSpec.A_WZ, 1, gdt * _at(spec, SMSpec.W + SMSpec.G_YAW), gat)
    _kernel_vjp(atoms, spec, SMSpec.A_H, 1, gdt * _at(spec, SMSpec.W + SMSpec.G_H), gat)
    _kernel_vjp(atoms, spec, SMSpec.A_GX, 2, gdt * _at(spec, SMSpec.W + SMSpec.G_TILT), gat)
    _kernel_vjp(atoms, spec, SMSpec.A_LX, 3, gdt * _at(spec, SMSpec.W + SMSpec.G_LHAND), gat)
    _kernel_vjp(atoms, spec, SMSpec.A_RX, 3, gdt * _at(spec, SMSpec.W + SMSpec.G_RHAND), gat)

    # action smoothness, joint limits
    for j in range(D.NDOF):
        var d = (_at(a, j) - _at(a_prev, j)) * Scalar[dt](2 * SM_P_W_ACTION_RATE) * gdt
        g_a[unsafe_offset=j] = d
        g_ap[unsafe_offset=j] = -d
        var q = _at(s, 7 + j)
        var gj = Scalar[dt](0)
        if _at(table, L.LO + j) - q > 0:
            gj -= 1
        if q - _at(table, L.HI + j) > 0:
            gj += 1
        g_s[unsafe_offset=7 + j] = gj * Scalar[dt](SM_P_W_JOINT_LIMIT) * gdt

    # the torso's lean and rates, onto the FK and the heading
    var q = _q4(s, 3)
    var hq = yaw_quat(q)
    var ti = Int(_at(table, L.TRACKED + Int(_at(table, L.ANCHOR))))
    var tq = _q4(quat, ti * 4)
    var uz = qapply(tq, _ez[dt]())
    var uh = qapply_inv(hq, uz)
    var angv = _v3(ang, ti * 3)
    var wh = qapply_inv(hq, angv)
    var gu = qapply_inv_vjp(hq, uz, Q4[dt](0, -gdt * _at(spec, SMSpec.LEAN) * 2 * uh[1], 0, 0))
    _add4(gquat, ti * 4, qapply_vjp(tq, _ez[dt](), gu[1])[0])
    var k_rate = gdt * Scalar[dt](2 * SM_P_W_TORSO_RATE)
    var gw = qapply_inv_vjp(hq, angv, Q4[dt](k_rate * wh[0], k_rate * wh[1], 0, 0))
    _add3(gang, ti * 3, gw[1])
    var g_hq = gu[0] + gw[0]

    # the fall logit, whichever branch `max` took
    var uzr = qapply_inv(q, _grav[dt]())[2]
    var x0 = (Scalar[dt](SM_P_FALL_H) - _at(s, 2)) / Scalar[dt](SM_P_FALL_TAU_H)
    var x1 = (Scalar[dt](1) + uzr - Scalar[dt](SM_P_FALL_TILT)) / Scalar[dt](SM_P_FALL_TAU_TILT)
    var gh = gat[unsafe_offset=SMSpec.A_H]
    var ggz = Scalar[dt](0)
    if x0 >= x1:
        gh -= g_lg / Scalar[dt](SM_P_FALL_TAU_H)
    else:
        ggz = g_lg / Scalar[dt](SM_P_FALL_TAU_TILT)

    # the atoms, onto the state
    var vh_g = qapply_inv_vjp(hq, _v3(s, D.NQ), Q4[dt](gat[unsafe_offset=SMSpec.A_VX], gat[unsafe_offset=SMSpec.A_VY], 0, 0))
    g_hq += vh_g[0]
    _add3(g_s, D.NQ, vh_g[1])
    var wz_g = qapply_vjp(q, _v3(s, D.NQ + 3), Q4[dt](0, 0, gat[unsafe_offset=SMSpec.A_WZ], 0))
    var g_q = wz_g[0]
    _add3(g_s, D.NQ + 3, wz_g[1])
    g_s[unsafe_offset=2] += gh
    g_q += qapply_inv_vjp_q(q, _grav[dt](), Q4[dt](gat[unsafe_offset=SMSpec.A_GX], gat[unsafe_offset=SMSpec.A_GY], ggz, 0))
    var rp = _v3(s, 0)
    var hl = Int(_at(table, L.HANDS))
    var hr = Int(_at(table, L.HANDS + 1))
    var pl_g = qapply_inv_vjp(
        hq, _v3(pos, hl * 3) - rp,
        Q4[dt](gat[unsafe_offset=SMSpec.A_LX], gat[unsafe_offset=SMSpec.A_LX + 1], gat[unsafe_offset=SMSpec.A_LX + 2], 0),
    )
    var pr_g = qapply_inv_vjp(
        hq, _v3(pos, hr * 3) - rp,
        Q4[dt](gat[unsafe_offset=SMSpec.A_RX], gat[unsafe_offset=SMSpec.A_RX + 1], gat[unsafe_offset=SMSpec.A_RX + 2], 0),
    )
    g_hq += pl_g[0] + pr_g[0]
    _add3(gpos, hl * 3, pl_g[1])
    _add3(gpos, hr * 3, pr_g[1])
    _add3(g_s, 0, -(pl_g[1] + pr_g[1]))
    g_q += yaw_quat_vjp(q, g_hq)
    _add4(g_s, 3, g_q)

    # through the FK
    sm_fk_vjp_lane[dt, D.NB, D.NQ, D.NV](tree, s, s.unsafe_offset(D.NQ), pos, quat, ang, gpos, gquat, glin, gang, gqpos, gqvel)
    for i in range(D.NQ):
        g_s[unsafe_offset=i] += gqpos[unsafe_offset=i]
    for i in range(D.NV):
        g_s[unsafe_offset=D.NQ + i] += gqvel[unsafe_offset=i]

    # the observation
    sm_observation_vjp_lane[dt, R](s, g_obs, g_s, g_a)
