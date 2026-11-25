"""A per-lane MASS for a free body: the carried object of a batch.

A batched `Data` holds state per lane and ONE `Model` for every lane, so a
body's mass is the batch's. This is the side channel for the mass of an
object the lanes carry, push or throw: each lane gives it its own.

WHAT A LANE SETS. A SCALE `s > 0` on the mass of a body that is FREE-JOINTED,
has NO CHILD and sits in no equality: its own kinematic tree. Its inertia
scales with it (the same shape, another density). `set_lane_mass_scale`
refuses any other body: a link's mass enters its ancestors' rows of the mass
matrix, and nothing below holds for it.

WHY THAT BODY, AND WHY THIS IS EXACT. Every term of the dynamics that holds
such a body's mass is linear in it and confined to its six dofs:

    its 6 x 6 block of the mass matrix        (CRBA: m, I and the CoM's lever)
    its six bias forces                       (RNE: gravity, w x I w, the lever)

so the model's mass times `s` IS that block times `s` and those six entries
times `s`. The step applies the two scalings right after the stages that
produce them; nothing inside CRBA or RNE changes. With `s` a power of two the
result equals, bit for bit, a batch whose model holds the scaled mass.

THE SOLVER'S WEIGHT. MuJoCo also derives `body_invweight0` from the masses at
model time, and `mj_diagApprox` prices every contact row with it. For a body
alone in its tree it is its own inverse mass and inertia: over `s`. The
Newton reads it from the shared model, so under this flag it binds
`Data.lane_body_invweight0` instead, `[BATCH * NBODY, 2]`: the model's table
once per lane, the scaled bodies' rows divided by their scale
(`contact_solve._precompute_contact_normal[BW_LANES=True]` adds the lane's row
offset). Leaving it at the model's is not harmless: on the G1's crate at 3 kg
it moves single carries by centimetres.

WHAT STAYS THE MODEL'S.
  * `stat.meaninertia`, the scale of the Newton's and the noslip's convergence
    tests: one scalar of the whole model. Measured on the G1's crate at 2 and
    4 kg against the model built at that mass: 1e-17 at the first step, 1e-7
    over a clip, no outcome changed.
  * `dof_invweight0` and `dof_M0` of the body's dofs: a free joint with
    neither limit nor friction loss has no reader for them at run time.
  * `gravcomp`, the fluid forces and `compute_rne_post` (cacc, cfrc_int,
    cfrc_ext and their sensors) read the model's mass: the setter refuses a
    body with `gravcomp`, the Euler integrator refuses `RNE_POST` with this
    flag, and a model with a fluid is the caller's to keep away.

`Data.lane_mass`, `[BATCH, NBODY]`: the scale per lane and body; 0 = the
model's mass. Read only by an integrator built with `LANE_MASS=True` (Euler,
Newton solver), and only after `lane_mass_reset` has filled the weights.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..joint_types import JNT_FREE
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    Dims,
    DimsLike,
    DYN2,
    rl2,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    BODY_IDX_GRAVCOMP,
    BODY_IDX_PARENT,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)

comptime LANE_MASS_TPB: Int = 64


@always_inline
def _lane_scale[DTYPE: DType, L_LM: TensorLayout](
    env: Int, body: Int, lane_mass: TileTensor[DTYPE, L_LM, MutAnyOrigin]
) -> Scalar[DTYPE]:
    """The lane's scale for `body`: 1 where none is set."""
    var s = rebind[Scalar[DTYPE]](lane_mass[env, body])
    if s > 0:
        return s
    return Scalar[DTYPE](1)


def _lane_mass_M_env[
    DTYPE: DType,
    D: DimsLike,
    L_LM: TensorLayout,
    L_JOINTS: TensorLayout,
    L_M: TensorLayout,
](
    env: Int,
    dims: D,
    lane_mass: TileTensor[DTYPE, L_LM, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
):
    """Each scaled free body's 6 x 6 block of `M` (dense, row-major) times
    its scale. After CRBA, before the armature."""
    var nv = dims.get_nv()
    for j in range(dims.get_njoint()):
        if Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE])) != JNT_FREE:
            continue
        var body = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        var s = _lane_scale(env, body, lane_mass)
        if s == 1:
            continue
        var da = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        for r in range(6):
            for c in range(6):
                var o = (da + r) * nv + (da + c)
                M[env, o] = rebind[Scalar[DTYPE]](M[env, o]) * s


def _lane_mass_bias_env[
    DTYPE: DType,
    D: DimsLike,
    L_LM: TensorLayout,
    L_JOINTS: TensorLayout,
    L_NV: TensorLayout,
](
    env: Int,
    dims: D,
    lane_mass: TileTensor[DTYPE, L_LM, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bias: TileTensor[DTYPE, L_NV, MutAnyOrigin],
):
    """Each scaled free body's six bias forces times its scale. After RNE,
    before the smooth force is assembled."""
    for j in range(dims.get_njoint()):
        if Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE])) != JNT_FREE:
            continue
        var body = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        var s = _lane_scale(env, body, lane_mass)
        if s == 1:
            continue
        var da = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        for k in range(6):
            bias[env, da + k] = rebind[Scalar[DTYPE]](bias[env, da + k]) * s


def _lane_mass_M_kernel[DTYPE: DType, NV: Int, NBODY: Int, NJOINT: Int, BATCH: Int](
    lane_mass: Tile[DTYPE, BATCH, NBODY],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    M: Tile[DTYPE, BATCH, NV * NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _lane_mass_M_env[DTYPE](env, Dims[nv=NV, nbody=NBODY, njoint=NJOINT](), lane_mass, joints, M)


def _lane_mass_bias_kernel[DTYPE: DType, NV: Int, NBODY: Int, NJOINT: Int, BATCH: Int](
    lane_mass: Tile[DTYPE, BATCH, NBODY],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    bias: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _lane_mass_bias_env[DTYPE](env, Dims[nv=NV, nbody=NBODY, njoint=NJOINT](), lane_mass, joints, bias)


def apply_lane_mass_M[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`scratch.M`: the scaled free bodies' blocks times their lane's scale.
    Call after `compute_mass_matrix`, before the armature is added."""
    comptime if target == "cpu":
        var dm = d.dims
        var lm_v = d.lane_mass.tt_dyn["cpu", DYN2](rl2(BATCH, dm.get_nbody()))
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl2(dm.get_njoint(), MODEL_JOINT_SIZE))
        var M_v = scratch.M.tt_dyn["cpu", DYN2](rl2(BATCH, dm.get_nv() * dm.get_nv()))
        for e in range(BATCH):
            _lane_mass_M_env[DTYPE](e, dm, lm_v, joints_v, M_v)
    else:
        comptime BLOCKS = (BATCH + LANE_MASS_TPB - 1) // LANE_MASS_TPB
        ctx.value().enqueue_function[_lane_mass_M_kernel[DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH]](
            d.lane_mass.tt["gpu", BATCH, D.NBODY](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            scratch.M.tt["gpu", BATCH, D.NV * D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LANE_MASS_TPB,),
        )


def apply_lane_mass_bias[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`scratch.bias`: the scaled free bodies' six entries times their lane's
    scale. Call after `compute_bias_forces_rne`, before the smooth force."""
    comptime if target == "cpu":
        var dm = d.dims
        var lm_v = d.lane_mass.tt_dyn["cpu", DYN2](rl2(BATCH, dm.get_nbody()))
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl2(dm.get_njoint(), MODEL_JOINT_SIZE))
        var bias_v = scratch.bias.tt_dyn["cpu", DYN2](rl2(BATCH, dm.get_nv()))
        for e in range(BATCH):
            _lane_mass_bias_env[DTYPE](e, dm, lm_v, joints_v, bias_v)
    else:
        comptime BLOCKS = (BATCH + LANE_MASS_TPB - 1) // LANE_MASS_TPB
        ctx.value().enqueue_function[_lane_mass_bias_kernel[DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH]](
            d.lane_mass.tt["gpu", BATCH, D.NBODY](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            scratch.bias.tt["gpu", BATCH, D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LANE_MASS_TPB,),
        )


# ── the host's side ──────────────────────────────────────────────────────────

def lane_mass_reset[DTYPE: DType, D: DimsLike, BATCH: Int](
    mut d: Data[DTYPE, D, BATCH], m: Model[DTYPE, D]
):
    """Host: no lane scales anything, and every lane's solver weights are the
    model's. Call once the model is built, before the first step of an
    integrator built with `LANE_MASS=True` (a device run then uploads
    `d.lane_mass` and `d.lane_body_invweight0`)."""
    var nb = d.dims.get_nbody()
    for i in range(BATCH * nb):
        d.lane_mass.data[i] = Scalar[DTYPE](0)
    for l in range(BATCH):
        for i in range(nb * 2):
            d.lane_body_invweight0.data[l * nb * 2 + i] = m.body_invweight0.data[i]


def set_lane_mass_scale[DTYPE: DType, D: DimsLike, BATCH: Int](
    mut d: Data[DTYPE, D, BATCH], m: Model[DTYPE, D], env: Int, body: Int, s: Float64
) raises:
    """Host: lane `env` holds `body`'s mass times `s` (1 = the model's):
    the scale the dynamics reads and the body's solver weight. Refuses a body
    this module's argument does not hold for (see the header). A device run
    then uploads `d.lane_mass` and `d.lane_body_invweight0`."""
    var nb = d.dims.get_nbody()
    var nj = d.dims.get_njoint()
    if env < 0 or env >= BATCH:
        raise Error("set_lane_mass_scale: lane " + String(env) + " of " + String(BATCH))
    if body <= 0 or body >= nb:
        raise Error("set_lane_mass_scale: body " + String(body) + " is the world or outside the model's " + String(nb) + " bodies")
    if not s > 0.0:
        raise Error("set_lane_mass_scale: a scale above 0, got " + String(s))
    var n_joints = 0
    var free = False
    for j in range(nj):
        if Int(m.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_BODY_ID]) == body:
            n_joints += 1
            free = Int(m.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE]) == JNT_FREE
    if n_joints != 1 or not free:
        raise Error("set_lane_mass_scale: body " + String(body) + " is not a free-jointed body (its joints: " + String(n_joints) + ")")
    for b in range(1, nb):
        if Int(m.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_PARENT]) == body:
            raise Error("set_lane_mass_scale: body " + String(body) + " has a child (body " + String(b) + "): its mass is not its own block's alone")
    if d.dims.get_nequality() > 0:
        raise Error("set_lane_mass_scale: the model has equality constraints, whose rows read the shared solver weights")
    if Float64(m.bodies.data[body * MODEL_BODY_SIZE + BODY_IDX_GRAVCOMP]) != 0.0:
        raise Error("set_lane_mass_scale: body " + String(body) + " has gravcomp, which reads the model's mass")
    d.lane_mass.data[env * nb + body] = Scalar[DTYPE](s)
    for k in range(2):
        d.lane_body_invweight0.data[(env * nb + body) * 2 + k] = Scalar[DTYPE](Float64(m.body_invweight0.data[body * 2 + k]) / s)


def _lane_weights_kernel[DTYPE: DType, NBODY: Int, BATCH: Int](
    lane_mass: Tile[DTYPE, BATCH, NBODY],
    body_invweight0: Tile[DTYPE, NBODY, 2],
    lane_bw: Tile[DTYPE, BATCH * NBODY, 2],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    for b in range(NBODY):
        var s = rebind[Scalar[DTYPE]](lane_mass[env, b])
        if not s > 0:
            s = Scalar[DTYPE](1)
        lane_bw[env * NBODY + b, 0] = rebind[Scalar[DTYPE]](body_invweight0[b, 0]) / s
        lane_bw[env * NBODY + b, 1] = rebind[Scalar[DTYPE]](body_invweight0[b, 1]) / s


def sync_lane_mass_weights[DTYPE: DType, D: DimsLike, BATCH: Int](
    mut d: Data[DTYPE, D, BATCH], mut m: Model[DTYPE, D], ctx: DeviceContext
) raises:
    """Device: `d.lane_body_invweight0` from the device's `d.lane_mass` and the
    model's weights, for a caller that writes the scales on the device (a
    sampler at a lane's reset). The scales it finds must be ones
    `set_lane_mass_scale` would accept: nothing checks them here."""
    comptime BLOCKS = (BATCH + LANE_MASS_TPB - 1) // LANE_MASS_TPB
    ctx.enqueue_function[_lane_weights_kernel[DTYPE, D.NBODY, BATCH]](
        d.lane_mass.tt["gpu", BATCH, D.NBODY](),
        m.body_invweight0.tt["gpu", D.NBODY, 2](),
        d.lane_body_invweight0.tt["gpu", BATCH * D.NBODY, 2](),
        grid_dim=(BLOCKS,),
        block_dim=(LANE_MASS_TPB,),
    )
