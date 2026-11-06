# +--------------------------------------------------------------------------+ #
# | SuperMimic: the robot interface — what the method needs from a body
# +--------------------------------------------------------------------------+ #
"""Everything SuperMimic knows about a robot goes through two traits, so a new
humanoid (RP1, ToddlerBot, …) is one conformer of each.

`TrackingRobot` (comptime, never instantiated): the dimensions, the BODIES BY
NAME (tracked set, anchor, root, feet, the body whose CoM is randomised), the
default pose and the soft-limit factor, and `SIM`, its simulator lane.
Requirements on the model: a single tree with a FREE root and HINGE joints
otherwise (MuJoCo order: qpos = root pos, root quat, hinges).

`TrackingSim`: one simulator lane. It owns the action semantics: a control
step applies `target = q0 + scale·a − bias` (mjlab's joint-position action
with the encoder bias), whatever the underlying env's own action convention.
It exposes the model's body / joint / geom records for the tables SuperMimic
builds once (FK tree, foot geoms, limits) and for per-episode randomisation.

`TrackingBatchedSim`: N lanes of the same robot on the device (the GPU env,
`env_gpu.mojo`), with the same action law. Its model records are SHARED by
the lanes (one model per batch); a sim with `LANE_DR` also takes per-lane
overrides of contact friction and CoM by body (`lane_dr_ptr`,
`physics3d/constraints/lane_dr.mojo`), which is how the env randomises them
per lane. Per-lane state (qpos, qvel) is device memory.

Every width the method uses follows from the robot (`SMDims[R]`):

    state     S  = nq + nv                      71 on the G1 (29 DoF)
    delta     D  = 3 + 3 + n + 3 + 3 + n        70
    obs       O  = 3n + 6                       93
    context   X  = 1 + n + 3 (friction, bias, CoM offset)   33
    goal / f  G  = 2n + 10                      68
    ref row   R  = 13·n_tracked + 7             189
"""

from std.math import ceildiv

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.core.cont_action import ContAction
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1 import UnitreeG1BatchedDR
from noeira.envs.robots.unitree_g1_mjlab import UnitreeG1Mjlab, UnitreeG1MjlabBatchedDR
from noeira.envs.robots.unitree_g1_mjlab_pd import G1M_N_DOF, g1m_action_scale, g1m_default_pos
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos, g1_effort, g1_kp
from noeira.physics3d.gpu.constants import MODEL_BODY_SIZE, MODEL_GEOM_SIZE, MODEL_JOINT_SIZE
from noeira.physics3d.parser.runtime_load import parse_model_runtime


trait TrackingSim(Movable & Deinitable):
    """One simulator lane."""

    @staticmethod
    def create() raises -> Self:
        ...

    def set_state(mut self, qpos: List[Float64], qvel: List[Float64]):
        """Write (qpos, qvel) and refresh kinematics."""
        ...

    def control_step(mut self, a: List[Float64], bias: List[Float64]):
        """One control step (all substeps) with the PD target
        `q0 + scale·a − bias` per joint."""
        ...

    def qpos(self, i: Int) -> Float64:
        ...

    def qvel(self, i: Int) -> Float64:
        ...

    def body_field(self, b: Int, k: Int) -> Float64:
        """Model body record `b`, field `k` (`BODY_IDX_*`)."""
        ...

    def set_body_field(mut self, b: Int, k: Int, v: Float64):
        ...

    def joint_field(self, j: Int, k: Int) -> Float64:
        ...

    def geom_field(self, g: Int, k: Int) -> Float64:
        ...

    def set_geom_field(mut self, g: Int, k: Int, v: Float64):
        ...

    def num_geoms(self) -> Int:
        ...

    # The window, for `supermimic_viewer` (forwards to the env's renderer).
    def init_renderer(mut self) raises -> Bool:
        ...

    def render_frame(mut self) raises:
        ...

    def renderer_delay(self, ms: Int):
        ...

    def check_renderer_quit(mut self) -> Bool:
        ...

    def renderer_paused(self) -> Bool:
        ...

    def start_recording(mut self, path: String, fps: Int) raises:
        ...

    def stop_recording(mut self) raises:
        ...

    def close_renderer(mut self) raises:
        ...


trait TrackingBatchedSim(Movable & Deinitable):
    """N simulator lanes on the device. Pointers are device pointers into
    float32 buffers; lane l's slice starts at l·width."""
    comptime N: Int
    comptime LANE_DR: Bool       # takes per-lane friction / CoM overrides

    @staticmethod
    def create(ctx: DeviceContext) raises -> Self:
        ...

    def lane_dr_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """[N · (NBODY + 1) · LANE_DR_STRIDE] on the device (world body
        included); read by the physics only when `LANE_DR`."""
        ...

    def qpos_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """[N · nq] (MuJoCo layout)."""
        ...

    def qvel_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """[N · nv]; the free joint's angular velocity in the body frame."""
        ...

    def control_step(
        mut self, ctx: DeviceContext, a: Pointer[Scalar[DT], MutAnyOrigin],
        bias: Pointer[Scalar[DT], MutAnyOrigin], bias_stride: Int, seed: UInt64,
    ) raises:
        """One control step for every lane, PD target `q0 + scale·a − bias`:
        a [N · n], lane l's bias at bias[l · bias_stride : + n]."""
        ...

    def reset_lanes(mut self, ctx: DeviceContext, mask: Pointer[Scalar[DT], MutAnyOrigin], seed: UInt64) raises:
        """Clear the lanes with mask[l] > 0.5 (solver carries, activations) to
        a fresh-episode state; the caller then writes their qpos / qvel and
        calls `refresh`."""
        ...

    def refresh(mut self, ctx: DeviceContext) raises:
        """Kinematics from the current qpos / qvel (after writing them)."""
        ...

    def body_field(self, b: Int, k: Int) -> Float64:
        """Model body record b, field k (the host copy of the shared model)."""
        ...

    def set_body_field_all(mut self, ctx: DeviceContext, b: Int, k: Int, v: Float64) raises:
        """Model body record b, field k, for every lane (shared model)."""
        ...

    def set_geom_field_all(mut self, ctx: DeviceContext, g: Int, k: Int, v: Float64) raises:
        ...


trait TrackingRobot:
    comptime NQ: Int
    comptime NV: Int
    comptime NDOF: Int
    comptime NBODY: Int          # bodies EXCLUDING the world
    comptime NJOINT: Int
    comptime N_TRACKED: Int
    comptime SIM: TrackingSim

    @staticmethod
    def mjcf_path() -> String:
        ...

    @staticmethod
    def tracked_bodies() -> List[String]:
        """The reward's tracked bodies, in the reference's order."""
        ...

    @staticmethod
    def anchor_body() -> String:
        """Re-anchoring frame of the reward (must be tracked)."""
        ...

    @staticmethod
    def root_body() -> String:
        ...

    @staticmethod
    def foot_bodies() -> List[String]:
        """Collision geoms on these bodies or their descendants are the feet."""
        ...

    @staticmethod
    def com_body() -> String:
        """The body whose CoM offset is randomised (ξ's last 3)."""
        ...

    @staticmethod
    def hand_bodies() -> List[String]:
        """Left, then right: the bodies the prompt reward places (`prompt.mojo`)."""
        ...

    @staticmethod
    def default_joint_pos(j: Int) -> Float64:
        ...

    @staticmethod
    def soft_limit_factor() -> Float64:
        ...


struct SMDims[R: TrackingRobot]:
    """The method's widths for robot R."""
    comptime NDOF = Self.R.NDOF
    comptime NQ = Self.R.NQ
    comptime NV = Self.R.NV
    comptime NB = Self.R.NBODY
    comptime S = Self.R.NQ + Self.R.NV
    comptime DELTA = 12 + 2 * Self.R.NDOF
    comptime OBS = 3 * Self.R.NDOF + 6
    comptime X = 1 + Self.R.NDOF + 3
    comptime GOAL_FRAME = 2 * Self.R.NDOF + 10
    comptime N_TRACKED = Self.R.N_TRACKED
    comptime REF_BODY = 13
    comptime REF = 13 * Self.R.N_TRACKED + 7
    comptime FK_BLOCK = 13 * Self.R.NBODY
    comptime GOAL_K = 10                                    # future frames in the goal
    comptime GOAL = Self.GOAL_K * (2 * Self.R.NDOF + 10)
    comptime HIST = 10                                      # observation history
    comptime HIST_FLAT = Self.HIST * (3 * Self.R.NDOF + 6)
    comptime LATENT = 32
    comptime DEC_IN = Self.LATENT + Self.HIST_FLAT
    comptime EROOT = 4
    comptime CRITIC_IN = Self.R.NQ + Self.R.NV + Self.GOAL + 1 + Self.R.NDOF + 3 + Self.EROOT
    comptime WM_IN = Self.R.NQ + Self.R.NV + Self.R.NDOF + 1 + Self.R.NDOF + 3   # [canonical S | a n | ξ X]


def sm_body_ids[R: TrackingRobot](names: List[String]) raises -> List[Int]:
    """Model body ids of `names` (world = 0), resolved from the robot's MJCF."""
    var fmd = parse_model_runtime(R.mjcf_path())
    var out = List[Int]()
    for k in range(len(names)):
        var hit = -1
        for b in range(len(fmd.body_names)):
            if fmd.body_names[b] == names[k]:
                hit = b
                break
        if hit < 0:
            raise Error("sm_body_ids: no body '" + names[k] + "' in " + R.mjcf_path())
        out.append(hit)
    if len(fmd.body_names) != R.NBODY + 1:
        raise Error(
            "sm_body_ids: the MJCF has " + String(len(fmd.body_names) - 1) + " bodies, the robot declares "
            + String(R.NBODY)
        )
    return out^


# ── the G1 (BFM-Zero's body, noeira's `UnitreeG1`) ──────────────────────────
struct G1Sim(TrackingSim):
    """`UnitreeG1` on the CPU. Its env action is BFM-Zero's
    (`clip(5a, ±5)·0.25·effort/kp + q0`), so the SuperMimic action a with
    bias b is handed over as `a/5 − b·kp/(1.25·effort)`: the target is then
    `q0 + 0.25·effort/kp·a − b`, mjlab's law (the clip binds at |a| > 5)."""
    var env: UnitreeG1[DType.float64]

    def __init__(out self, var env: UnitreeG1[DType.float64]):
        self.env = env^

    def __init__(out self, *, deinit move: Self):
        self.env = move.env^

    @staticmethod
    def create() raises -> Self:
        var e = UnitreeG1[DType.float64]()
        _ = e.reset()
        return Self(e^)

    def set_state(mut self, qpos: List[Float64], qvel: List[Float64]):
        self.env.set_state(qpos, qvel)

    def control_step(mut self, a: List[Float64], bias: List[Float64]):
        var act = ContAction[G1_N_DOF]()
        for j in range(G1_N_DOF):
            act.data[j] = a[j] / 5.0 - bias[j] * g1_kp(j) / (1.25 * g1_effort(j))
        _ = self.env.step(act)

    def qpos(self, i: Int) -> Float64:
        return Float64(self.env.d.qpos.data[i])

    def qvel(self, i: Int) -> Float64:
        return Float64(self.env.d.qvel.data[i])

    def body_field(self, b: Int, k: Int) -> Float64:
        return Float64(self.env.mf.bodies.data[b * MODEL_BODY_SIZE + k])

    def set_body_field(mut self, b: Int, k: Int, v: Float64):
        self.env.mf.bodies.data[b * MODEL_BODY_SIZE + k] = v

    def joint_field(self, j: Int, k: Int) -> Float64:
        return Float64(self.env.mf.joints.data[j * MODEL_JOINT_SIZE + k])

    def geom_field(self, g: Int, k: Int) -> Float64:
        return Float64(self.env.mf.geoms.data[g * MODEL_GEOM_SIZE + k])

    def set_geom_field(mut self, g: Int, k: Int, v: Float64):
        self.env.mf.geoms.data[g * MODEL_GEOM_SIZE + k] = v

    def num_geoms(self) -> Int:
        return UnitreeG1[DType.float64].NGEOM

    def init_renderer(mut self) raises -> Bool:
        return self.env.init_renderer()

    def render_frame(mut self) raises:
        self.env.render_frame()

    def renderer_delay(self, ms: Int):
        self.env.renderer_delay(ms)

    def check_renderer_quit(mut self) -> Bool:
        return self.env.check_renderer_quit()

    def renderer_paused(self) -> Bool:
        return self.env.renderer_paused()

    def start_recording(mut self, path: String, fps: Int) raises:
        self.env.start_recording(path, fps)

    def stop_recording(mut self) raises:
        self.env.stop_recording()

    def close_renderer(mut self) raises:
        self.env.close_renderer()


def _k_g1_action(
    n_lanes: Int64, a: Pointer[Scalar[DT], MutAnyOrigin], bias: Pointer[Scalar[DT], MutAnyOrigin], bias_stride: Int64,
    k: Pointer[Scalar[DT], MutAnyOrigin], dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """BFM-Zero's env action for the SuperMimic action a and bias b:
    `a/5 − b · kp/(1.25·effort)` (k holds kp/(1.25·effort) per joint)."""
    var i = Int(global_idx.x)
    if i >= Int(n_lanes) * G1_N_DOF:
        return
    var l = i // G1_N_DOF
    var j = i % G1_N_DOF
    dst[unsafe_offset=i] = a[unsafe_offset=i] / 5 - bias[unsafe_offset=l * Int(bias_stride) + j] * k[unsafe_offset=j]


def _k_copy_mask(n: Int64, src: Pointer[Scalar[DT], MutAnyOrigin], dst: Pointer[Scalar[DT], MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


struct G1BatchedSim[N_: Int](TrackingBatchedSim):
    """`UnitreeG1BatchedDR[N]` (float32, NVIDIA only: the G1's nv overflows
    Metal's per-thread stack), with per-lane friction / CoM overrides. The same action law as `G1Sim`, as a kernel
    writing the env's action buffer; resets go through the env's own
    `selective_reset_batch` (it clears what `mj_resetData` clears)."""
    comptime N = Self.N_
    comptime LANE_DR = True
    var env: UnitreeG1BatchedDR[Self.N_]
    var kgain: DeviceBuffer[DT]

    def __init__(out self, var env: UnitreeG1BatchedDR[Self.N_], var kgain: DeviceBuffer[DT]):
        self.env = env^
        self.kgain = kgain^

    def __init__(out self, *, deinit move: Self):
        self.env = move.env^
        self.kgain = move.kgain^

    @staticmethod
    def create(ctx: DeviceContext) raises -> Self:
        var env = UnitreeG1BatchedDR[Self.N_](ctx)
        env.reset_batch[Self.N_](Optional(ctx), 0)
        var k = ctx.enqueue_create_buffer[DT](G1_N_DOF)
        var hk = ctx.enqueue_create_host_buffer[DT](G1_N_DOF)
        ctx.synchronize()
        for j in range(G1_N_DOF):
            hk[j] = Scalar[DT](g1_kp(j) / (1.25 * g1_effort(j)))
        ctx.enqueue_copy(k, hk)
        ctx.synchronize()
        return Self(env^, k^)

    def lane_dr_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return self.env.lane_dr_ptr()

    def qpos_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self.env.d.qpos.dev.value().unsafe_ptr())

    def qvel_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self.env.d.qvel.dev.value().unsafe_ptr())

    def control_step(
        mut self, ctx: DeviceContext, a: Pointer[Scalar[DT], MutAnyOrigin],
        bias: Pointer[Scalar[DT], MutAnyOrigin], bias_stride: Int, seed: UInt64,
    ) raises:
        comptime NA = Self.N_ * G1_N_DOF
        ctx.enqueue_function[_k_g1_action](
            Int64(Self.N_), a, bias, Int64(bias_stride), mptr(self.kgain.unsafe_ptr()), self.env.action_ptr(),
            grid_dim=ceildiv(NA, 128), block_dim=128,
        )
        self.env.step_batch[Self.N_](Optional(ctx), seed)

    def reset_lanes(mut self, ctx: DeviceContext, mask: Pointer[Scalar[DT], MutAnyOrigin], seed: UInt64) raises:
        ctx.enqueue_function[_k_copy_mask](
            Int64(Self.N_), mask, mptr(self.env._done.unsafe_ptr()), grid_dim=ceildiv(Self.N_, 128), block_dim=128
        )
        self.env.selective_reset_batch[Self.N_](Optional(ctx), seed)

    def refresh(mut self, ctx: DeviceContext) raises:
        self.env._run_fields_fk(ctx)
        self.env._run_fields_vel(ctx)

    def body_field(self, b: Int, k: Int) -> Float64:
        return Float64(self.env.mf.bodies.data[b * MODEL_BODY_SIZE + k])

    def set_body_field_all(mut self, ctx: DeviceContext, b: Int, k: Int, v: Float64) raises:
        self.env.mf.bodies.data[b * MODEL_BODY_SIZE + k] = Scalar[DT](v)
        self.env.mf.bodies.upload_resident(ctx)

    def set_geom_field_all(mut self, ctx: DeviceContext, g: Int, k: Int, v: Float64) raises:
        self.env.mf.geoms.data[g * MODEL_GEOM_SIZE + k] = Scalar[DT](v)
        self.env.mf.geoms.upload_resident(ctx)


struct G1Robot(TrackingRobot):
    """noeira's BFM-Zero G1: 29 DoF, 41 bodies (8 massless foot frames and 2
    fixed rubber hands among them), the reproduction's 14 tracked bodies."""
    comptime NQ = 36
    comptime NV = 35
    comptime NDOF = G1_N_DOF
    comptime NBODY = UnitreeG1[DType.float64].NBODY - 1
    comptime NJOINT = UnitreeG1[DType.float64].NJOINT
    comptime N_TRACKED = 14
    comptime SIM = G1Sim

    @staticmethod
    def mjcf_path() -> String:
        return "noeira/envs/robots/assets/unitree_g1.xml"

    @staticmethod
    def tracked_bodies() -> List[String]:
        return [
            "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
            "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
            "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
        ]

    @staticmethod
    def anchor_body() -> String:
        return "torso_link"

    @staticmethod
    def root_body() -> String:
        return "pelvis"

    @staticmethod
    def foot_bodies() -> List[String]:
        return ["left_ankle_roll_link", "right_ankle_roll_link"]

    @staticmethod
    def com_body() -> String:
        return "torso_link"

    @staticmethod
    def hand_bodies() -> List[String]:
        return ["left_wrist_yaw_link", "right_wrist_yaw_link"]

    @staticmethod
    def default_joint_pos(j: Int) -> Float64:
        return g1_default_pos(j)

    @staticmethod
    def soft_limit_factor() -> Float64:
        return 0.9       # mjlab's G1 `soft_joint_pos_limit_factor`


# ── the G1 as mjlab models it (the reproduction's body, `UnitreeG1Mjlab`) ────
struct G1MjlabSim(TrackingSim):
    """`UnitreeG1Mjlab` on the CPU. Its env action is mjlab's own (`q0 +
    scale·a`, unclipped), so the SuperMimic action a with bias b is handed
    over as `a − b/scale`: the target is `q0 + scale·a − b`."""
    var env: UnitreeG1Mjlab[DType.float64]

    def __init__(out self, var env: UnitreeG1Mjlab[DType.float64]):
        self.env = env^

    def __init__(out self, *, deinit move: Self):
        self.env = move.env^

    @staticmethod
    def create() raises -> Self:
        var e = UnitreeG1Mjlab[DType.float64]()
        _ = e.reset()
        return Self(e^)

    def set_state(mut self, qpos: List[Float64], qvel: List[Float64]):
        self.env.set_state(qpos, qvel)

    def control_step(mut self, a: List[Float64], bias: List[Float64]):
        var act = ContAction[G1M_N_DOF]()
        for j in range(G1M_N_DOF):
            act.data[j] = a[j] - bias[j] / g1m_action_scale(j)
        _ = self.env.step(act)

    def qpos(self, i: Int) -> Float64:
        return Float64(self.env.d.qpos.data[i])

    def qvel(self, i: Int) -> Float64:
        return Float64(self.env.d.qvel.data[i])

    def body_field(self, b: Int, k: Int) -> Float64:
        return Float64(self.env.mf.bodies.data[b * MODEL_BODY_SIZE + k])

    def set_body_field(mut self, b: Int, k: Int, v: Float64):
        self.env.mf.bodies.data[b * MODEL_BODY_SIZE + k] = v

    def joint_field(self, j: Int, k: Int) -> Float64:
        return Float64(self.env.mf.joints.data[j * MODEL_JOINT_SIZE + k])

    def geom_field(self, g: Int, k: Int) -> Float64:
        return Float64(self.env.mf.geoms.data[g * MODEL_GEOM_SIZE + k])

    def set_geom_field(mut self, g: Int, k: Int, v: Float64):
        self.env.mf.geoms.data[g * MODEL_GEOM_SIZE + k] = v

    def num_geoms(self) -> Int:
        return UnitreeG1Mjlab[DType.float64].NGEOM

    def init_renderer(mut self) raises -> Bool:
        return self.env.init_renderer()

    def render_frame(mut self) raises:
        self.env.render_frame()

    def renderer_delay(self, ms: Int):
        self.env.renderer_delay(ms)

    def check_renderer_quit(mut self) -> Bool:
        return self.env.check_renderer_quit()

    def renderer_paused(self) -> Bool:
        return self.env.renderer_paused()

    def start_recording(mut self, path: String, fps: Int) raises:
        self.env.start_recording(path, fps)

    def stop_recording(mut self) raises:
        self.env.stop_recording()

    def close_renderer(mut self) raises:
        self.env.close_renderer()


def _k_g1m_action(
    n_lanes: Int64, a: Pointer[Scalar[DT], MutAnyOrigin], bias: Pointer[Scalar[DT], MutAnyOrigin], bias_stride: Int64,
    k: Pointer[Scalar[DT], MutAnyOrigin], dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """mjlab's env action for the SuperMimic action a and bias b: `a − b·k`,
    k = 1/scale per joint."""
    var i = Int(global_idx.x)
    if i >= Int(n_lanes) * G1M_N_DOF:
        return
    var l = i // G1M_N_DOF
    var j = i % G1M_N_DOF
    dst[unsafe_offset=i] = a[unsafe_offset=i] - bias[unsafe_offset=l * Int(bias_stride) + j] * k[unsafe_offset=j]


struct G1MjlabBatchedSim[N_: Int](TrackingBatchedSim):
    """`UnitreeG1MjlabBatchedDR[N]` (float32), per-lane friction / CoM
    overrides, `G1MjlabSim`'s action law as a kernel."""
    comptime N = Self.N_
    comptime LANE_DR = True
    var env: UnitreeG1MjlabBatchedDR[Self.N_]
    var kgain: DeviceBuffer[DT]

    def __init__(out self, var env: UnitreeG1MjlabBatchedDR[Self.N_], var kgain: DeviceBuffer[DT]):
        self.env = env^
        self.kgain = kgain^

    def __init__(out self, *, deinit move: Self):
        self.env = move.env^
        self.kgain = move.kgain^

    @staticmethod
    def create(ctx: DeviceContext) raises -> Self:
        var env = UnitreeG1MjlabBatchedDR[Self.N_](ctx)
        env.reset_batch[Self.N_](Optional(ctx), 0)
        var k = ctx.enqueue_create_buffer[DT](G1M_N_DOF)
        var hk = ctx.enqueue_create_host_buffer[DT](G1M_N_DOF)
        ctx.synchronize()
        for j in range(G1M_N_DOF):
            hk[j] = Scalar[DT](1.0 / g1m_action_scale(j))
        ctx.enqueue_copy(k, hk)
        ctx.synchronize()
        return Self(env^, k^)

    def lane_dr_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return self.env.lane_dr_ptr()

    def qpos_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self.env.d.qpos.dev.value().unsafe_ptr())

    def qvel_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self.env.d.qvel.dev.value().unsafe_ptr())

    def control_step(
        mut self, ctx: DeviceContext, a: Pointer[Scalar[DT], MutAnyOrigin],
        bias: Pointer[Scalar[DT], MutAnyOrigin], bias_stride: Int, seed: UInt64,
    ) raises:
        comptime NA = Self.N_ * G1M_N_DOF
        ctx.enqueue_function[_k_g1m_action](
            Int64(Self.N_), a, bias, Int64(bias_stride), mptr(self.kgain.unsafe_ptr()), self.env.action_ptr(),
            grid_dim=ceildiv(NA, 128), block_dim=128,
        )
        self.env.step_batch[Self.N_](Optional(ctx), seed)

    def reset_lanes(mut self, ctx: DeviceContext, mask: Pointer[Scalar[DT], MutAnyOrigin], seed: UInt64) raises:
        ctx.enqueue_function[_k_copy_mask](
            Int64(Self.N_), mask, mptr(self.env._done.unsafe_ptr()), grid_dim=ceildiv(Self.N_, 128), block_dim=128
        )
        self.env.selective_reset_batch[Self.N_](Optional(ctx), seed)

    def refresh(mut self, ctx: DeviceContext) raises:
        self.env._run_fields_fk(ctx)
        self.env._run_fields_vel(ctx)

    def body_field(self, b: Int, k: Int) -> Float64:
        return Float64(self.env.mf.bodies.data[b * MODEL_BODY_SIZE + k])

    def set_body_field_all(mut self, ctx: DeviceContext, b: Int, k: Int, v: Float64) raises:
        self.env.mf.bodies.data[b * MODEL_BODY_SIZE + k] = Scalar[DT](v)
        self.env.mf.bodies.upload_resident(ctx)

    def set_geom_field_all(mut self, ctx: DeviceContext, g: Int, k: Int, v: Float64) raises:
        self.env.mf.geoms.data[g * MODEL_GEOM_SIZE + k] = Scalar[DT](v)
        self.env.mf.geoms.upload_resident(ctx)


struct G1MjlabRobot(TrackingRobot):
    """The G1 as mjlab models it (`UnitreeG1Mjlab`): 29 DoF, 30 bodies, the
    reproduction's 14 tracked bodies, mjlab's knees-bent default pose. The
    joint order is `G1Robot`'s, so the same library rows drive it."""
    comptime NQ = 36
    comptime NV = 35
    comptime NDOF = G1M_N_DOF
    comptime NBODY = UnitreeG1Mjlab[DType.float64].NBODY - 1
    comptime NJOINT = UnitreeG1Mjlab[DType.float64].NJOINT
    comptime N_TRACKED = 14
    comptime SIM = G1MjlabSim

    @staticmethod
    def mjcf_path() -> String:
        return "noeira/envs/robots/assets/unitree_g1_mjlab.xml"

    @staticmethod
    def tracked_bodies() -> List[String]:
        return G1Robot.tracked_bodies()

    @staticmethod
    def anchor_body() -> String:
        return "torso_link"

    @staticmethod
    def root_body() -> String:
        return "pelvis"

    @staticmethod
    def foot_bodies() -> List[String]:
        return ["left_ankle_roll_link", "right_ankle_roll_link"]

    @staticmethod
    def com_body() -> String:
        return "torso_link"

    @staticmethod
    def hand_bodies() -> List[String]:
        return ["left_wrist_yaw_link", "right_wrist_yaw_link"]

    @staticmethod
    def default_joint_pos(j: Int) -> Float64:
        return g1m_default_pos(j)

    @staticmethod
    def soft_limit_factor() -> Float64:
        return 0.9
