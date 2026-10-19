"""Unitree G1 — BFM-Zero's environment: torque-level PD, 64-D proprio state.

    obs     = [q - q_default (29), qdot (29), projected gravity (3),
               root angular velocity * 0.25 (3)]                        (64)
    action  = 29 PD targets in [-1, 1], through the reference's rescale:
                  a *= 5 ; clip +-5 ; target = a * 0.25 * effort/kp + q_default
    torque  = kp (target - q) - kd qdot, clipped to +-effort, EVERY substep
    control = 50 Hz = 4 substeps of 1/200 s (the reference's `sim.fps` and
              `control_decimation`)
    reward  = 0 — BFM-Zero's agent never reads the env reward; the only
              rewards in any of its losses are the discriminator's and the
              auxiliary penalties, both computed agent-side
    reset   = the reference's `init_state`: pelvis at (0, 0, 0.8), identity
              orientation, joints at their default angles, at rest — i.e. the
              XML's `stand` keyframe. Reference-state init from a motion and
              the 30 % lie-down start come with the motion loader (G1 rung).
    episode = 500 control steps, NO early termination of any kind
              (`humanoidverse_isaac.py:699-706` asserts every terminator off)

Source: `references/BFM-Zero-main/humanoidverse/envs/legged_base_task/
legged_robot_base.py` (`_compute_torques`, `_pre_physics_step`), the obs
composition in `agents/envs/humanoidverse_isaac.py:404-450`, and the
constants in `config/robot/g1/g1_29dof_hard_waist.yaml` via the generated
`unitree_g1_pd.mojo`.

⚠ THE PD RUNS AT EVERY SUBSTEP ON BOTH DEVICES. The GPU hook is called per
substep by construction; the CPU hook is not, so this config sets
`CUSTOM_ACTIONS_EVERY_SUBSTEP` and `Phyics3dEnv.step` re-invokes it. The
difference is not academic: `kp` is 99 on the legs and 300 on the waist,
and a target held for 4 substeps at 200 Hz instead of re-evaluated is a
different closed loop. `test_unitree_g1_vs_mujoco` drives MuJoCo with the
reference's own torque law in Python per substep and compares `qpos`.

⚠ ANGULAR VELOCITY IS IN THE PELVIS FRAME. MuJoCo's free-joint `qvel[3:6]`
is expressed in the child body's frame, which is what IsaacLab's
`root_ang_vel_b` is; no rotation is applied. Projected gravity is
`R_root^T (0, 0, -1)`, from the root quaternion in `qpos[3:7]` (w, x, y, z).

⚠ NO OBSERVATION NOISE, NO HISTORY, NO PRIVILEGED STATE, NO LAST ACTION HERE.
Those are the deployable recipe's (G4 rung) and each is a separate,
switchable ingredient; this is the privileged no-DR body the paper's own
`BFM-Zero-priv` row was trained on, minus the privileged features that the
FK-side observation adds at the G1 rung.

⚠ `RECORD_PREV_ACTION` CANNOT BE USED: it needs 2 x ACTION_DIM = 58 meta
slots and `METADATA_SIZE` is 27. The reference's `last_action` observation
gets its own storage when the history observation lands.
"""

from std.math import sqrt
from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    MODEL_JOINT_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
    METADATA_SIZE,
)
from ..phyics3d_env_config import Phyics3dEnvConfig
from .unitree_g1_xml import (
    UNITREE_G1_NMESH_VERTS,
    UNITREE_G1_OBS_DIM,
    UNITREE_G1_STATE_DIM,
    ROOT_QPOS_SIZE,
    ROOT_QVEL_SIZE,
    TORSO_BODY_IDX,
)
from .unitree_g1_priv_obs import (
    G1_PRIV_DIM,
    G1_N_SKELETON,
    G1_PRIV_OFF_HEIGHT,
    g1_skeleton_body,
    g1_heading_inv,
    g1_origin_velocity,
    g1_priv_body,
    g1_priv_scatter,
    g1_head_pose_vel,
)
from .unitree_g1_pd import (
    G1_N_DOF,
    G1_ACTION_SCALE,
    G1_ACTION_CLIP,
    G1_NORMALIZE_FROM,
    G1_NORMALIZE_TO,
    G1_INIT_ROOT_Z,
    G1_SIM_TIMESTEP,
    G1_CONTROL_DECIMATION,
    g1_kp,
    g1_kd,
    g1_effort,
    g1_default_pos,
)
from noeira.core.tiles import Tile


# `obs_scales.base_ang_vel` in `bfm_zero_obs.yaml` — the paper's omega/4.
comptime G1_ANG_VEL_SCALE: Float64 = 0.25


@always_inline
def g1_pd_target(i: Int, a: Float64) -> Float64:
    """Policy output -> joint target, `legged_robot_base.py:222-297`.

    `a * normalize_to/normalize_from`, clipped to `+-action_clip`, times
    `action_scale * effort / kp` (the `action_rescale` branch), plus the
    default angle. At the reference's values a full-scale action moves the
    target by `1.25 * effort / kp`: 1.75 rad on a hip, 0.37 rad on the waist.
    """
    var s = a * (G1_NORMALIZE_TO / G1_NORMALIZE_FROM)
    if s > G1_ACTION_CLIP:
        s = G1_ACTION_CLIP
    elif s < -G1_ACTION_CLIP:
        s = -G1_ACTION_CLIP
    return s * G1_ACTION_SCALE * g1_effort(i) / g1_kp(i) + g1_default_pos(i)


@always_inline
def g1_pd_torque(i: Int, target: Float64, q: Float64, qd: Float64) -> Float64:
    """`_compute_torques`, control_type P, `clip_torques` True."""
    var tau = g1_kp(i) * (target - q) - g1_kd(i) * qd
    var lim = g1_effort(i)
    if tau > lim:
        return lim
    if tau < -lim:
        return -lim
    return tau


@always_inline
def _clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    if x > hi:
        return hi
    if x < lo:
        return lo
    return x


@always_inline
def _projected_gravity(
    qw: Float64, qx: Float64, qy: Float64, qz: Float64
) -> Tuple[Float64, Float64, Float64]:
    """`R(q)^T (0, 0, -1)` — minus the third ROW of the rotation matrix.

    Identity quaternion gives (0, 0, -1); the reference's `projected_gravity`
    is `quat_rotate_inverse(root_quat, (0, 0, -1))`, the same quantity.
    """
    return (
        2.0 * (qw * qy - qx * qz),
        -2.0 * (qy * qz + qw * qx),
        2.0 * (qx * qx + qy * qy) - 1.0,
    )


struct UnitreeG1Config(Phyics3dEnvConfig):
    comptime FRAME_SKIP: Int = G1_CONTROL_DECIMATION
    comptime MAX_STEPS: Int = 500
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # The reference model carries no `<option integrator>` — MuJoCo's Euler.
    comptime INTEGRATOR: StaticString = "euler"
    # The 64-D obs reads qpos/qvel only; FK products are not consulted yet.
    # Flip to True when the privileged body-frame features arrive.
    # G3.0: the privileged observation reads xpos/xquat/xipos/xvel/xangvel
    # AFTER the control step, so the FK products and body velocities must
    # describe the integrated state, not the one before the last substep.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime HAS_GPU_HOOKS: Bool = True
    comptime HAS_CUSTOM_ACTUATION_GPU: Bool = True
    comptime CUSTOM_ACTIONS_EVERY_SUBSTEP: Bool = True
    comptime NORMALIZED_ACTIONS: Bool = False
    comptime NMESH_VERTS: Int = UNITREE_G1_NMESH_VERTS

    # ── shared arithmetic ─────────────────────────────────────────────────

    @always_inline
    @staticmethod
    def _write_pd_torques[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_actuators: List[Scalar[DTYPE]],
        actions: List[Float64],
    ):
        for i in range(D.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        for i in range(G1_N_DOF):
            var a = actions[i] if i < len(actions) else 0.0
            var q = Float64(d.qpos.data[ROOT_QPOS_SIZE + i])
            var qd = Float64(d.qvel.data[ROOT_QVEL_SIZE + i])
            var ao = i * MODEL_ACTUATOR_SIZE
            d.qfrc.data[ROOT_QVEL_SIZE + i] = Scalar[DTYPE](
                _clamp(
                    g1_pd_torque(i, g1_pd_target(i, a), q, qd),
                    Float64(m_actuators[ao + ACT_IDX_CTRL_MIN]),
                    Float64(m_actuators[ao + ACT_IDX_CTRL_MAX]),
                )
            )

    # ── CPU hooks ─────────────────────────────────────────────────────────

    @staticmethod
    def custom_apply_actions_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        m_tendons: List[Scalar[DTYPE]],
        m_actuators: List[Scalar[DTYPE]],
        m_act_tendons: List[Scalar[DTYPE]],
        actions: List[Float64],
    ) -> Bool:
        """The reference's PD torque on every hinge, written straight to
        `qfrc` — a `<motor>` with gear 1 is `qfrc[dof] = clamp(ctrl)`.

        ⚠⚠ TWO CLIPS, AND THEY ARE NOT THE SAME NUMBER. The reference clips
        the PD torque to the yaml's `dof_effort_limit` (139 N m on the hip
        pitch/roll) and then hands it to the simulator as `ctrl`, where the
        MuJoCo `<motor ctrlrange>` clamps it AGAIN — and the sim-to-sim XML
        says 88 N m on those four hips. `test_unitree_g1_vs_mujoco` found the
        first draft of this hook, which applied only the effort clip, 0.118
        rad off MuJoCo at step 72 of a driven rollout. Both clips are
        applied here in the reference's order: effort first (the table),
        then the actuator record's ctrlrange. Called every substep
        (`CUSTOM_ACTIONS_EVERY_SUBSTEP`).
        """
        Self._write_pd_torques(d, m_actuators, actions)
        return True

    @staticmethod
    def custom_extract_obs_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        for i in range(G1_N_DOF):
            obs.append(
                d.qpos.data[ROOT_QPOS_SIZE + i]
                - Scalar[DTYPE](g1_default_pos(i))
            )
        for i in range(G1_N_DOF):
            obs.append(d.qvel.data[ROOT_QVEL_SIZE + i])
        var g = _projected_gravity(
            Float64(d.qpos.data[3]),
            Float64(d.qpos.data[4]),
            Float64(d.qpos.data[5]),
            Float64(d.qpos.data[6]),
        )
        obs.append(Scalar[DTYPE](g[0]))
        obs.append(Scalar[DTYPE](g[1]))
        obs.append(Scalar[DTYPE](g[2]))
        for k in range(3):
            obs.append(d.qvel.data[3 + k] * Scalar[DTYPE](G1_ANG_VEL_SCALE))

        # ── privileged `max_local_self` (463): the simulator's 30 bodies
        # + the virtual head, in the heading frame (G3.0) ────────────────
        var priv = Array[Scalar[DTYPE], G1_PRIV_DIM](fill=Scalar[DTYPE](0))
        var rb = g1_skeleton_body(0)
        var rootx = d.xpos.data[rb * 3 + 0]
        var rooty = d.xpos.data[rb * 3 + 1]
        var rootz = d.xpos.data[rb * 3 + 2]
        var h = g1_heading_inv[DTYPE](
            d.xquat.data[rb * 4 + 0], d.xquat.data[rb * 4 + 1],
            d.xquat.data[rb * 4 + 2], d.xquat.data[rb * 4 + 3],
        )
        priv[G1_PRIV_OFF_HEIGHT] = rootz
        var tp = Array[Scalar[DTYPE], 13](fill=Scalar[DTYPE](0))  # torso origin pose + vel
        for s in range(G1_N_SKELETON):
            var b = g1_skeleton_body(s)
            var vo = g1_origin_velocity[DTYPE](
                d.xvel.data[b * 3 + 0], d.xvel.data[b * 3 + 1], d.xvel.data[b * 3 + 2],
                d.xangvel.data[b * 3 + 0], d.xangvel.data[b * 3 + 1], d.xangvel.data[b * 3 + 2],
                d.xpos.data[b * 3 + 0], d.xpos.data[b * 3 + 1], d.xpos.data[b * 3 + 2],
                d.xipos.data[b * 3 + 0], d.xipos.data[b * 3 + 1], d.xipos.data[b * 3 + 2],
            )
            var f = g1_priv_body[DTYPE](
                h[0], h[1], h[2], h[3], rootx, rooty, rootz,
                d.xpos.data[b * 3 + 0], d.xpos.data[b * 3 + 1], d.xpos.data[b * 3 + 2],
                d.xquat.data[b * 4 + 0], d.xquat.data[b * 4 + 1],
                d.xquat.data[b * 4 + 2], d.xquat.data[b * 4 + 3],
                vo[0], vo[1], vo[2],
                d.xangvel.data[b * 3 + 0], d.xangvel.data[b * 3 + 1], d.xangvel.data[b * 3 + 2],
            )
            g1_priv_scatter[DTYPE](s, f, priv)
            if b == TORSO_BODY_IDX:
                tp[0] = d.xpos.data[b * 3 + 0]
                tp[1] = d.xpos.data[b * 3 + 1]
                tp[2] = d.xpos.data[b * 3 + 2]
                tp[3] = d.xquat.data[b * 4 + 0]
                tp[4] = d.xquat.data[b * 4 + 1]
                tp[5] = d.xquat.data[b * 4 + 2]
                tp[6] = d.xquat.data[b * 4 + 3]
                tp[7] = vo[0]
                tp[8] = vo[1]
                tp[9] = vo[2]
                tp[10] = d.xangvel.data[b * 3 + 0]
                tp[11] = d.xangvel.data[b * 3 + 1]
                tp[12] = d.xangvel.data[b * 3 + 2]
        var hd = g1_head_pose_vel[DTYPE](
            tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], tp[6],
            tp[7], tp[8], tp[9], tp[10], tp[11], tp[12],
        )
        var fh = g1_priv_body[DTYPE](
            h[0], h[1], h[2], h[3], rootx, rooty, rootz,
            hd[0], hd[1], hd[2], tp[3], tp[4], tp[5], tp[6],
            hd[3], hd[4], hd[5], tp[10], tp[11], tp[12],
        )
        g1_priv_scatter[DTYPE](G1_N_SKELETON, fh, priv)
        for i in range(G1_PRIV_DIM):
            obs.append(priv[i])
        return True

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`init_state`: pelvis at (0, 0, 0.8), identity quaternion, default
        joint angles, at rest. Deterministic — the reference adds no reset
        noise either (`noise_to_initial_level: 0`)."""
        for i in range(D.NQ):
            d.qpos.data[i] = Scalar[DTYPE](0)
        for i in range(D.NV):
            d.qvel.data[i] = Scalar[DTYPE](0)
        d.qpos.data[2] = Scalar[DTYPE](G1_INIT_ROOT_Z)
        d.qpos.data[3] = Scalar[DTYPE](1)
        for i in range(G1_N_DOF):
            d.qpos.data[ROOT_QPOS_SIZE + i] = Scalar[DTYPE](g1_default_pos(i))

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
        return (Scalar[DTYPE](0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return G1_SIM_TIMESTEP

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.0

    # ── GPU hooks ─────────────────────────────────────────────────────────

    @always_inline
    @staticmethod
    def custom_apply_actions_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        NTENDON_F: Int,
        ACTION_DIM: Int,
        NA_F: Int,
        NACT_F: Int,
    ](
        qfrc: Tile[DTYPE, BATCH_SIZE, NV],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
        tendons: Tile[DTYPE, NTENDON_F, MODEL_TENDON_SIZE],
        acts: Tile[DTYPE, NACT_F * MODEL_ACTUATOR_SIZE],
        act_tendons: Tile[DTYPE, NTENDON_F * MODEL_ACT_TENDON_SIZE],
        env: Int,
    ):
        """The CPU hook's twin, one lane, called every substep by the batched
        env. Same arithmetic in Float64 on the host-dtype operands, so the
        two devices agree to the dtype's rounding and nothing else."""
        for i in range(NV):
            qfrc[env, i] = Scalar[DTYPE](0)
        for i in range(G1_N_DOF):
            var a = Float64(rebind[Scalar[DTYPE]](actions[env, i]))
            var q = Float64(rebind[Scalar[DTYPE]](qpos[env, ROOT_QPOS_SIZE + i]))
            var qd = Float64(rebind[Scalar[DTYPE]](qvel[env, ROOT_QVEL_SIZE + i]))
            var ao = i * MODEL_ACTUATOR_SIZE
            qfrc[env, ROOT_QVEL_SIZE + i] = Scalar[DTYPE](
                _clamp(
                    g1_pd_torque(i, g1_pd_target(i, a), q, qd),
                    Float64(rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MIN])),
                    Float64(rebind[Scalar[DTYPE]](acts[ao + ACT_IDX_CTRL_MAX])),
                )
            )

    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        OBS_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
    ) -> Bool:
        """Byte-for-byte the CPU observation's order — the batched trainer's
        checkpoint is what the single-env eval loads."""
        # ⚠ Unrolled at compile time so `g1_default_pos(i)` — a float64
        # constant table — folds into float32 immediates: a Metal kernel
        # cannot carry a double at all (`select double -0.2` failed IR
        # verification once this kernel grew past what LLVM unrolled alone).
        comptime for i in range(G1_N_DOF):
            obs[env, i] = qpos[env, ROOT_QPOS_SIZE + i] - Scalar[DTYPE](
                g1_default_pos(i)
            )
        for i in range(G1_N_DOF):
            obs[env, G1_N_DOF + i] = qvel[env, ROOT_QVEL_SIZE + i]
        var g = _projected_gravity(
            Float64(rebind[Scalar[DTYPE]](qpos[env, 3])),
            Float64(rebind[Scalar[DTYPE]](qpos[env, 4])),
            Float64(rebind[Scalar[DTYPE]](qpos[env, 5])),
            Float64(rebind[Scalar[DTYPE]](qpos[env, 6])),
        )
        var b = 2 * G1_N_DOF
        obs[env, b + 0] = Scalar[DTYPE](g[0])
        obs[env, b + 1] = Scalar[DTYPE](g[1])
        obs[env, b + 2] = Scalar[DTYPE](g[2])
        for k in range(3):
            obs[env, b + 3 + k] = qvel[env, 3 + k] * Scalar[DTYPE](
                G1_ANG_VEL_SCALE
            )

        # ── privileged `max_local_self` (463), the CPU hook's arithmetic on
        # the lane's field tensors (G3.0) ──────────────────────────────────
        var priv = Array[Scalar[DTYPE], G1_PRIV_DIM](fill=Scalar[DTYPE](0))
        var rb = g1_skeleton_body(0)
        var rootx = rebind[Scalar[DTYPE]](xpos[env, rb * 3 + 0])
        var rooty = rebind[Scalar[DTYPE]](xpos[env, rb * 3 + 1])
        var rootz = rebind[Scalar[DTYPE]](xpos[env, rb * 3 + 2])
        var h = g1_heading_inv[DTYPE](
            rebind[Scalar[DTYPE]](xquat[env, rb * 4 + 0]),
            rebind[Scalar[DTYPE]](xquat[env, rb * 4 + 1]),
            rebind[Scalar[DTYPE]](xquat[env, rb * 4 + 2]),
            rebind[Scalar[DTYPE]](xquat[env, rb * 4 + 3]),
        )
        priv[G1_PRIV_OFF_HEIGHT] = rootz
        var tp = Array[Scalar[DTYPE], 13](fill=Scalar[DTYPE](0))
        for s in range(G1_N_SKELETON):
            var bb = g1_skeleton_body(s)
            var px = rebind[Scalar[DTYPE]](xpos[env, bb * 3 + 0])
            var py = rebind[Scalar[DTYPE]](xpos[env, bb * 3 + 1])
            var pz = rebind[Scalar[DTYPE]](xpos[env, bb * 3 + 2])
            var qx = rebind[Scalar[DTYPE]](xquat[env, bb * 4 + 0])
            var qy = rebind[Scalar[DTYPE]](xquat[env, bb * 4 + 1])
            var qz = rebind[Scalar[DTYPE]](xquat[env, bb * 4 + 2])
            var qw = rebind[Scalar[DTYPE]](xquat[env, bb * 4 + 3])
            var wx = rebind[Scalar[DTYPE]](xangvel[env, bb * 3 + 0])
            var wy = rebind[Scalar[DTYPE]](xangvel[env, bb * 3 + 1])
            var wz = rebind[Scalar[DTYPE]](xangvel[env, bb * 3 + 2])
            var vo = g1_origin_velocity[DTYPE](
                rebind[Scalar[DTYPE]](xvel[env, bb * 3 + 0]),
                rebind[Scalar[DTYPE]](xvel[env, bb * 3 + 1]),
                rebind[Scalar[DTYPE]](xvel[env, bb * 3 + 2]),
                wx, wy, wz, px, py, pz,
                rebind[Scalar[DTYPE]](xipos[env, bb * 3 + 0]),
                rebind[Scalar[DTYPE]](xipos[env, bb * 3 + 1]),
                rebind[Scalar[DTYPE]](xipos[env, bb * 3 + 2]),
            )
            var f = g1_priv_body[DTYPE](
                h[0], h[1], h[2], h[3], rootx, rooty, rootz,
                px, py, pz, qx, qy, qz, qw, vo[0], vo[1], vo[2], wx, wy, wz,
            )
            g1_priv_scatter[DTYPE](s, f, priv)
            if bb == TORSO_BODY_IDX:
                tp[0] = px
                tp[1] = py
                tp[2] = pz
                tp[3] = qx
                tp[4] = qy
                tp[5] = qz
                tp[6] = qw
                tp[7] = vo[0]
                tp[8] = vo[1]
                tp[9] = vo[2]
                tp[10] = wx
                tp[11] = wy
                tp[12] = wz
        var hd = g1_head_pose_vel[DTYPE](
            tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], tp[6],
            tp[7], tp[8], tp[9], tp[10], tp[11], tp[12],
        )
        var fh = g1_priv_body[DTYPE](
            h[0], h[1], h[2], h[3], rootx, rooty, rootz,
            hd[0], hd[1], hd[2], tp[3], tp[4], tp[5], tp[6],
            hd[3], hd[4], hd[5], tp[10], tp[11], tp[12],
        )
        g1_priv_scatter[DTYPE](G1_N_SKELETON, fh, priv)
        for i in range(G1_PRIV_DIM):
            obs[env, UNITREE_G1_STATE_DIM + i] = priv[i]
        return True

    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        ACTION_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        cfrc_ext: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        return (Scalar[DTYPE](0), False)

    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NJOINT: Int,
        NV: Int,
        NBODY: Int,
        NGEOM_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
        mocap_pos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        mocap_quat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
        seed: Int,
    ):
        """`custom_reset_cpu`'s twin: the same deterministic stand pose on
        every lane."""
        for i in range(NQ):
            qpos[env, i] = Scalar[DTYPE](0)
        for i in range(NV):
            qvel[env, i] = Scalar[DTYPE](0)
        qpos[env, 2] = Scalar[DTYPE](G1_INIT_ROOT_Z)
        qpos[env, 3] = Scalar[DTYPE](1)
        for i in range(G1_N_DOF):
            qpos[env, ROOT_QPOS_SIZE + i] = Scalar[DTYPE](g1_default_pos(i))
