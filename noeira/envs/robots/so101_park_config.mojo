"""Config for the P0 scene-budget probe — deliberately the emptiest one here.

⚠⚠ THIS IS NOT A TASK AND MUST NEVER BECOME ONE. Reward is a constant 0 and
nothing ever terminates. That is not laziness, it is the measurement: P0 asks
what a PARKED SLOT costs the physics step, so every quantity that is not the
physics step is held at zero. A reward hook that read `xpos` would put a
per-body loop into the very kernel whose per-body cost is the thing being
measured, and the curve would then include it.

For the same reason there is no obs shaping: `custom_extract_obs_gpu` is left
at the trait default, so the model's default observation is used unchanged
across every k.

## The three legs this serves

`docs/TASK_LAYER_IMPLEMENTATION.md` §1. Leg 1 sweeps the slot count with
`max_contacts` PINNED (see `so101_park_xml.PARK_MAX_CONTACTS`); leg 2 sweeps
`max_contacts` alone at k=0; leg 3 is `REPARK` below.

## ⚠ REPARK — leg 3, and what it can and cannot do

`docs/TASK_LAYER_IMPLEMENTATION.md` Gap D: a parked free body does not STAY
parked. Gravity is a `Model` field, shared across lanes, so a slot written to
its park pose at reset then FALLS for the whole episode — `qvel` grows, and its
broadphase AABB moves every step, so SAP re-sorts it every step. "Parked" as
`docs/TASK_LAYER_PLAN.md` §3.3 specifies it is an initial condition, not an
invariant. `REPARK=True` makes it an invariant by rewriting the pose every
step, which is what would ship.

⚠⚠ AND IT PINS THE POSE ONLY — IT CANNOT ZERO THE VELOCITY. `pre_step_gpu`'s
ABI is `(qpos, meta, env)`: there is no `qvel` operand
(`phyics3d_env_config.mojo:641`, called at `phyics3d_batched_env.mojo:1381`).
So a reparked slot holds its position while `qvel` keeps accumulating downward
— harmless over a 300-step horizon (~11.8 m/s, and the position is pinned
regardless), but NOT what should ship. **Adding `qvel` to that hook is a P3
plumbing item**, and it is exactly the sort of thing that is invisible until
someone tries to use the hook for this. Leg 3's number is therefore a LOWER
BOUND on the repark cost and an UPPER BOUND on the saving.

⚠ The park pose comes from `so101_park_pose.mojo`, which the scene generator
emits. Do not spell it here — see that file's header for why.
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    META_IDX_PREV_X,
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    rk4_extra_workspace_size,
)

from .so101_park_pose import PARK_X, PARK_Y, PARK_Z, PARK_SPACING
from .so101_park_xml import ARM_NQ, SLOT_NQ, PARK_NMESH_VERTS
from ..phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


struct So101ParkProbeConfig[
    NQ_MODEL: Int,
    NV_MODEL: Int,
    N_SLOTS: Int,
    REPARK: Bool = False,
](Phyics3dEnvConfig):
    comptime FRAME_SKIP: Int = 2
    comptime HAS_GPU_HOOKS: Bool = True
    # ⚠ EULER, AS MuJoCo RUNS THIS MODEL. Neither the Menagerie SO-101 nor the
    # generated scenes set `<option integrator>`, so MuJoCo steps them under
    # Euler; RK4 was this config's inherited default, not the model's. RK4
    # at frame skip 2 ran the whole pipeline 8 times per env step — 8
    # collision launches, 8 Newton solves — for 4x the cost of Euler's 2
    # (PERFORMANCE.md §13.38, the parked-slot probe). Fidelity: the studio
    # path under Euler agrees with MuJoCo to 4.2e-17 over 50 steps on the
    # k=0 and k=13 park scenes (2026-09-07). Position actuators at dt=0.002
    # are what MuJoCo's own default runs them with.
    comptime INTEGRATOR: StaticString = "euler"

    # ⚠ SO-ARM101 SHIPS A MOCAP BODY — `<body name="target" mocap="true">`,
    # body 8. `Phyics3dBatchedEnv.__init__` RAISES if a mocap-flagged body
    # exists while this is False, and that guard is right: for a TASK, a
    # target frozen at its XML pose is a silently easier task.
    #
    # Here it is frozen ON PURPOSE and that is not the same failure. There is
    # no task, no reward reads the target, and `init_qpos_gpu` deliberately
    # leaves `mocap_pos`/`mocap_quat` alone so every lane and every repeat sees
    # the identical scene. A target that moved per episode would make the
    # contact set vary run to run, which is the one thing a throughput sweep
    # cannot tolerate.
    #
    # ⚠ The target is `contype=0 conaffinity=0`, so it collides with nothing
    # and costs one body in FK — it does not perturb what P0 is measuring.
    comptime USES_MOCAP: Bool = True

    # ⚠ NOT the horizon. The probe never resets mid-run — it times a fixed
    # number of `step_batch` calls — so this only has to be larger than that
    # count for nothing to truncate. See the probe's own note on why a reset
    # in the middle of a timed region would be a measurement bug.
    comptime MAX_STEPS: Int = 1_000_000

    # ⚠⚠ NONZERO, AND THE WHOLE PROBE DEPENDS ON IT. Zero means MESH GEOMS DO
    # NOT COLLIDE — the arm's 30 collision meshes would silently stop being
    # colliders and P0 would measure a capsule-free arm with some boxes near
    # it. SO-ARM101 is the first batched model in this tree with collidable
    # meshes and that is precisely why it is the right model to price a
    # budget on.
    comptime NMESH_VERTS: Int = PARK_NMESH_VERTS

    comptime INTEGRATOR_WS_EXTRA: Int = 0  # Euler needs no extra workspace

    # === CPU hooks — present for the trait; the probe is GPU-only ===
    @staticmethod
    def pre_step_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut prev_x: Scalar[DTYPE],
    ):
        pass

    @staticmethod
    def compute_reward_and_done_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        return (Scalar[DTYPE](0.0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return 0.002

    @staticmethod
    def get_reset_noise() -> Float64:
        # ⚠ ZERO ON PURPOSE. Reset noise would give each lane a different arm
        # pose and therefore a different contact set, so the contact count
        # would vary lane to lane and run to run. The probe wants the SAME
        # physics in every lane so that the only thing moving across the sweep
        # is the slot count.
        return 0.0

    # === GPU: pre-step — leg 3's repark, or nothing ===
    @always_inline
    @staticmethod
    def pre_step_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
    ):
        comptime if Self.REPARK:
            # Pin every parked slot back onto its declared pose. 7 writes per
            # slot: 3 position + a unit quaternion.
            #
            # ⚠ THE QUATERNION IS NOT OPTIONAL. A free body's qpos is
            # (x, y, z, qw, qx, qy, qz); writing only the position leaves the
            # slot free to TUMBLE about a pinned point, which keeps its AABB
            # moving — the exact broadphase churn repark exists to remove. The
            # measurement would then show repark saving almost nothing and the
            # conclusion would be wrong.
            #
            # ⚠⚠ `comptime for`, NOT a runtime loop — AND THAT IS A METAL
            # HARD ERROR, NOT A STYLE POINT. The park pose is Float64, so a
            # runtime `s` makes `PARK_X + Float64(s) * PARK_SPACING` an f64
            # multiply-add INSIDE the kernel. Metal has no double:
            #
            #   Function 'air.convert.f.f32.f.f64' has Metal-unsupported
            #   instructions ... LLVM ERROR: Failed to verify LLVM IR for Metal
            #
            # With `s` comptime the whole expression folds to a constant and
            # only the `Scalar[DTYPE]` store survives. Measured 2026-09-02:
            # this exact change is the difference between the probe failing to
            # build on Apple and building clean.
            comptime for s in range(Self.N_SLOTS):
                comptime a = ARM_NQ + s * SLOT_NQ
                comptime px = PARK_X + Float64(s) * PARK_SPACING
                qpos[env, a + 0] = Scalar[DTYPE](px)
                qpos[env, a + 1] = Scalar[DTYPE](PARK_Y)
                qpos[env, a + 2] = Scalar[DTYPE](PARK_Z)
                qpos[env, a + 3] = Scalar[DTYPE](1.0)
                qpos[env, a + 4] = Scalar[DTYPE](0.0)
                qpos[env, a + 5] = Scalar[DTYPE](0.0)
                qpos[env, a + 6] = Scalar[DTYPE](0.0)

    # === GPU: reward + termination — constant zero, never done ===
    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        ACTION_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
        qvel: Tile[DTYPE, BATCH_SIZE, NV_F],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY_F * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        bodies: Tile[DTYPE, NBODY_F, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        cfrc_ext: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY_F * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        return (Scalar[DTYPE](0.0), False)

    # === GPU: qpos init — the scene's own pose is already right ===
    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NJOINT_F: Int,
        NV_F: Int,
        NBODY_M: Int,
        NGEOM_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
        qvel: Tile[DTYPE, BATCH_SIZE, NV_F],
        joints: Tile[DTYPE, NJOINT_F, MODEL_JOINT_SIZE],
        mocap_pos: Tile[DTYPE, BATCH_SIZE, NBODY_M * 3],
        mocap_quat: Tile[DTYPE, BATCH_SIZE, NBODY_M * 4],
        bodies: Tile[DTYPE, NBODY_M, MODEL_BODY_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
        seed: Int,
    ):
        # `reset` restores the model's reference pose, which for a parked slot
        # IS its park pose — it is where the MJCF declares the body. Nothing
        # to do.
        pass
