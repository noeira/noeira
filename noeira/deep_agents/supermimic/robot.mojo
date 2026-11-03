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

Every width the method uses follows from the robot (`SMDims[R]`):

    state     S  = nq + nv                      71 on the G1 (29 DoF)
    delta     D  = 3 + 3 + n + 3 + 3 + n        70
    obs       O  = 3n + 6                       93
    context   X  = 1 + n + 3 (friction, bias, CoM offset)   33
    goal / f  G  = 2n + 10                      68
    ref row   R  = 13·n_tracked + 7             189
"""

from max.gpu.host import DeviceContext

from noeira.core.cont_action import ContAction
from noeira.envs.robots import UnitreeG1
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
    def default_joint_pos(j: Int) -> Float64:
        return g1_default_pos(j)

    @staticmethod
    def soft_limit_factor() -> Float64:
        return 0.9       # mjlab's G1 `soft_joint_pos_limit_factor`
