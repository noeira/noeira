"""Subtree velocity sensors — MuJoCo `subtreelinvel`.

MuJoCo fills `data.subtree_linvel[body]` in `mj_subtreeVel` and exposes it via
a `<subtreelinvel body="..."/>` sensor. It is the linear velocity of the
subtree's centre of mass:

    subtree_linvel[b] = (sum_i m_i * v_i) / (sum_i m_i),  i in subtree(b)

where `v_i` is the world-frame linear velocity of body i's CoM. Our
`Data.xvel` is exactly that (`_vel_body` propagates
`v = v_parent + w_parent x (xipos_i - xipos_parent)`), so no extra kinematics
are needed — only the mass-weighted walk.

Used by the dm_control suite for the forward-speed term in cheetah, walker,
hopper and humanoid (`sensordata['torso_subtreelinvel']`).

TWO PRECONDITIONS, both easy to get wrong:

  * `xvel` must be current with the integrated `qvel`. The integrator writes
    it mid-step, so an env reading this from a reward hook must set
    `Phyics3dEnvConfig.SYNC_FK_AFTER_STEP` (which also runs `_fields_vel`).
  * body records are the packed `Model.bodies` host list, indexed with the
    `BODY_IDX_*` column constants.

⚠ COMPILE-TIME FOOTGUN — do NOT "simplify" `walk_to_root` back to a `while`.
The parent walk is a BOUNDED `for ... break` on purpose. Written as an
unbounded, data-dependent `while`, calling `subtree_linvel` from inside a loop
took the Mojo compiler from ~2 s to >150 s (never finished). The NESTING is
the trigger: the same `while` called once, not from a loop, compiles fine —
as does `continue`, multiple `List` params, genericity over `DType`, and an
inner `for`. Minimal reproducer, no physics3d involved:

    def g(xs: List[Float64], body: Int, root: Int) -> Bool:
        var b = body
        while b >= 0:                        # <- unbounded, data-dependent
            if b == root: return True
            b = Int(xs[b]) - 1
        return False

    def f(xs: List[Float64], n: Int, root: Int) -> Float64:
        var acc = Float64(0)
        for i in range(n):
            if not g(xs, i, root): continue
            acc += xs[i]
        return acc

    def main():
        var xs = List[Float64]()
        for i in range(21): xs.append(1.0)
        var a = Float64(0)
        for b in range(7): a += f(xs, 7, b)  # <- drop this loop => 2 s
        print(a)

Bounding is not a hack here: a parent chain cannot revisit a body, so `nbody`
is an exact bound.
"""

from layout import TileTensor, TensorLayout

from ..gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_PARENT,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
)
from ..fields import DimsLike
from ..kinematics.quat_math import gpu_quat_rotate


def walk_to_root[
    DTYPE: DType
](
    m_bodies: List[Scalar[DTYPE]], body: Int, root: Int, nbody: Int
) -> Bool:
    """True when `body` is `root` or a descendant of it.

    Bounded by `nbody`. Read the module docstring before making this a
    `while`.
    """
    var b = body
    for _ in range(nbody):
        if b < 0:
            break
        if b == root:
            return True
        b = Int(m_bodies[b * MODEL_BODY_SIZE + BODY_IDX_PARENT])
    return False


def subtree_linvel[
    DTYPE: DType
](
    xvel: List[Scalar[DTYPE]],
    m_bodies: List[Scalar[DTYPE]],
    nbody: Int,
    root: Int,
    mut vx: Float64,
    mut vy: Float64,
    mut vz: Float64,
):
    """`data.subtree_linvel[root]` — CoM velocity of the subtree at `root`.

    `xvel` is `Data.xvel.data` (NBODY*3, world-frame CoM velocity per body);
    `m_bodies` is `Model.bodies.data`. Writes (0,0,0) for a massless subtree,
    matching MuJoCo's guard.
    """
    var total_mass = Float64(0)
    var px = Float64(0)
    var py = Float64(0)
    var pz = Float64(0)

    for b in range(nbody):
        if not walk_to_root(m_bodies, b, root, nbody):
            continue
        var mass = Float64(m_bodies[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
        if mass == 0.0:
            continue
        total_mass += mass
        px += mass * Float64(xvel[b * 3 + 0])
        py += mass * Float64(xvel[b * 3 + 1])
        pz += mass * Float64(xvel[b * 3 + 2])

    if total_mass <= 0.0:
        vx = 0.0
        vy = 0.0
        vz = 0.0
        return
    vx = px / total_mass
    vy = py / total_mass
    vz = pz / total_mass


# =============================================================================
# GPU-batched counterparts
# =============================================================================
#
# The functions above take host `List`s and compute in Float64, so a kernel can
# call neither (Metal has no `double`). These are the same arithmetic over the
# batched field/model tensors, in `DTYPE`.
#
# ⚠ The `for ... break` bound in `walk_to_root_gpu` is load-bearing for the
# SAME reason as on the host — read the module docstring. On the GPU there is a
# second reason: an unbounded data-dependent `while` inside a kernel is a
# divergence hazard, and every lane walks a different chain length.


@always_inline
def walk_to_root_gpu[
    DTYPE: DType,
    D: DimsLike,
    L_BODIES: TensorLayout](
    dims: D,
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    body: Int,
    root: Int,
) -> Bool:
    """True when `body` is `root` or a descendant of it. Batched `walk_to_root`.

    `bodies` is the SHARED (unbatched) `Model.bodies` tensor — the kinematic
    tree is model state, identical across lanes.
    """
    var nbody = dims.get_nbody()
    var b = body
    for _ in range(nbody):
        if b < 0:
            break
        if b == root:
            return True
        b = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
    return False


@always_inline
def subtree_linvel_gpu[
    DTYPE: DType,
    D: DimsLike,
    L_XVEL: TensorLayout,
    L_BODIES: TensorLayout](
    dims: D,
    xvel: TileTensor[DTYPE, L_XVEL, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    env: Int,
    root: Int,
    mut vx: Scalar[DTYPE],
    mut vy: Scalar[DTYPE],
    mut vz: Scalar[DTYPE],
):
    """`data.subtree_linvel[root]` for one lane. Batched `subtree_linvel`.

    Writes (0,0,0) for a massless subtree, matching MuJoCo's guard and the host
    version. This is what feeds every `torso_subtreelinvel` reward in the
    suite's locomotion tasks (cheetah, walker, hopper, humanoid, humanoid_cmu).
    """
    var nbody = dims.get_nbody()
    comptime ZERO = Scalar[DTYPE](0)
    var total_mass = ZERO
    var px = ZERO
    var py = ZERO
    var pz = ZERO

    for b in range(nbody):
        if not walk_to_root_gpu[DTYPE](dims,bodies, b, root):
            continue
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        if mass == ZERO:
            continue
        total_mass += mass
        px += mass * rebind[Scalar[DTYPE]](xvel[env, b * 3 + 0])
        py += mass * rebind[Scalar[DTYPE]](xvel[env, b * 3 + 1])
        pz += mass * rebind[Scalar[DTYPE]](xvel[env, b * 3 + 2])

    if total_mass <= ZERO:
        vx = ZERO
        vy = ZERO
        vz = ZERO
        return
    vx = px / total_mass
    vy = py / total_mass
    vz = pz / total_mass


@always_inline
def subtree_angmom_gpu[
    DTYPE: DType,
    D: DimsLike,
    L_B3: TensorLayout,
    L_XQUAT: TensorLayout,
    L_BODIES: TensorLayout](
    dims: D,
    xipos: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    xvel: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    xangvel: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    env: Int,
    root: Int,
    mut lx: Scalar[DTYPE],
    mut ly: Scalar[DTYPE],
    mut lz: Scalar[DTYPE],
):
    """`data.subtree_angmom[root]` for one lane — MuJoCo's `mj_subtreeVel`.

    Angular momentum of the subtree at `root`, about that subtree's centre of
    mass, in the WORLD frame:

        L = sum_i [ R_i (I_i . (R_i^T w_i)) + m_i (xipos_i - com) x (v_i - vcom) ]

    over the bodies of the subtree, where `R_i` is the body's INERTIAL frame
    orientation and `I_i` its diagonal inertia in that frame.

    ⚠⚠ THIS IS THE DEFINITION, NOT MUJOCO'S RECURSION, AND THE SUBSTITUTION
    WAS MEASURED BEFORE IT WAS MADE. `mj_subtreeVel`
    (engine_core_smooth.c:2249) evaluates the same quantity with two REVERSE
    passes over the body list, carrying a `body_vel` scratch, a momentum
    accumulator it later divides by `body_subtreemass`, and a per-parent
    shift term `(com_i - com_p) x subtreemass_i (vlin_i - vlin_p)`. Porting
    that shape would have cost a `body_subtreemass` column in the body record
    (28 -> 29, every model's layout), a `Data.subtree_angmom` field, a new
    pass in the step with a new ordering constraint, and a 26th buffer in a
    sensor kernel that fails with NO DIAGNOSTIC at 29. The direct sum needs
    none of them: every operand is already bound by the sensor eval.

    The two agree to 2.2e-16 on a five-body fixture with a free joint, a ball
    joint and two hinges, at every one of the five roots — they are the same
    identity, evaluated in a different order. ⚠ A DIFFERENT ORDER, so this is
    NOT bit-identical to MuJoCo and its gate must not claim to be.

    ⚠ `R_i` IS COMPOSED, NOT READ. `Data` has no `ximat`; it has the body's
    world `xquat` and the body record's LOCAL inertial quaternion. Rotating a
    vector by `R_body . R_iquat` is two `gpu_quat_rotate` calls and by its
    transpose two more with the conjugates, which avoids materialising a
    matrix and matches how `pose_transmission` composes a site's frame.

    Massless bodies are kept in the sum rather than skipped: their inertia is
    zero too, so both terms vanish, and the branch would only differ for a
    body MuJoCo cannot build.
    """
    var nbody = dims.get_nbody()
    comptime ZERO = Scalar[DTYPE](0)

    # The subtree's own CoM velocity — the SAME walk `subtreelinvel` serves,
    # so the two sensors cannot drift apart.
    var vcx = ZERO
    var vcy = ZERO
    var vcz = ZERO
    subtree_linvel_gpu[DTYPE](dims, xvel, bodies, env, root, vcx, vcy, vcz)

    var cx = rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 0])
    var cy = rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 1])
    var cz = rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 2])

    lx = ZERO
    ly = ZERO
    lz = ZERO

    for b in range(nbody):
        if not walk_to_root_gpu[DTYPE](dims, bodies, b, root):
            continue

        # ── spin: R_i (I_i . (R_i^T w_i)) ────────────────────────────────
        var wx = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 0])
        var wy = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 1])
        var wz = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 2])
        var bqx = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_W])

        var wb = gpu_quat_rotate(-bqx, -bqy, -bqz, bqw, wx, wy, wz)
        var wi = gpu_quat_rotate(-iqx, -iqy, -iqz, iqw, wb[0], wb[1], wb[2])
        var hx = wi[0] * rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IXX])
        var hy = wi[1] * rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IYY])
        var hz = wi[2] * rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IZZ])
        var hb = gpu_quat_rotate(iqx, iqy, iqz, iqw, hx, hy, hz)
        var hw = gpu_quat_rotate(bqx, bqy, bqz, bqw, hb[0], hb[1], hb[2])
        lx += hw[0]
        ly += hw[1]
        lz += hw[2]

        # ── orbital: m_i (xipos_i - com) x (v_i - vcom) ───────────────────
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        var dx = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 0]) - cx
        var dy = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 1]) - cy
        var dz = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 2]) - cz
        var dvx = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 0]) - vcx
        var dvy = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 1]) - vcy
        var dvz = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 2]) - vcz
        lx += mass * (dy * dvz - dz * dvy)
        ly += mass * (dz * dvx - dx * dvz)
        lz += mass * (dx * dvy - dy * dvx)
