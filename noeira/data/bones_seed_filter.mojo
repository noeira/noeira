# +--------------------------------------------------------------------------+ #
# | BONES-SEED feasibility filter on OUR G1 (SuperMimic M1b)
# +--------------------------------------------------------------------------+ #
"""The paper's Appendix E feasibility filter, as the SuperMimic reproduction
writes it (`scripts/data/filter_bones_seed.py`, `feasibility_filter`), run on
noeira's BFM-Zero G1 instead of mjlab's capsule G1.

A clip at 50 Hz is dropped for the FIRST of:
  * `elevated_support`: the lowest of the 30 link origins is above 0.20 m
    for ≥ 0.6 s (the clip stands on something that is not the floor);
  * `ground_penetration`: that lowest origin goes below −0.12 m once, or
    below −0.08 m on more than 80 % of the frames;
  * `floating_feet`: for ≥ 0.5 s the robot is upright (projected gravity
    z < −0.9), standing (pelvis z in (0.60, 0.85)), still (pelvis speed
    < 0.1 m/s, joint speed RMS < 0.5 rad/s) AND the lowest point of the foot
    collision geoms is above 0.02 m (it stands on an invisible prop).
Otherwise it is kept (`""`).

WHAT IS OURS, NOT THE REPRODUCTION'S
====================================
* THE BODY. The 30 link origins are `g1_skeleton_body`'s: our 41 bodies minus
  the world, the 8 massless `dummy_*f_*` foot frames and the 2 fixed
  `*_rubber_hand` bodies, which leaves exactly mjlab's 30 names. The sole is
  our feet's collision geoms: 4 spheres of radius 0.005 m per foot, not
  mjlab's 7 capsules. The reproduction's lowest-point formula
  `centre z − half·|axis z| − radius` is `centre z − radius` for a sphere
  (half-length 0), so its code runs unchanged on our model; the oracle
  (`tools/supermimic/make_synthetic_seed.py`) does exactly that.
* FLOAT64 THROUGHOUT. The reproduction filters float32 copies (its npz
  stores float32); a clip within float32 rounding of a threshold
  (~1e-8 m) can come out the other way. The verdict counts are body-
  dependent long before that matters.
* THE PELVIS'S VELOCITY AND POSE are the 71-d row's root columns, which are
  the free joint's — the reproduction reads the same quantities back from
  MuJoCo's `cvel`/`xquat` (equal up to rounding).
"""

from std.math import sqrt

from noeira.data.bones_seed import MimicMotion, SEED_REF_DIM
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF
from noeira.envs.robots.unitree_g1_priv_obs import G1_N_SKELETON, g1_skeleton_body
from noeira.envs.robots.unitree_g1_xml import LEFT_ANKLE_ROLL_BODY_IDX, RIGHT_ANKLE_ROLL_BODY_IDX
from noeira.physics3d.constants import GEOM_SPHERE
from noeira.physics3d.gpu.constants import (
    BODY_IDX_PARENT, GEOM_IDX_BODY, GEOM_IDX_CONAFFINITY, GEOM_IDX_CONTYPE, GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y, GEOM_IDX_POS_Z, GEOM_IDX_RADIUS, GEOM_IDX_TYPE, MODEL_BODY_SIZE, MODEL_GEOM_SIZE,
)


# The reproduction's thresholds (filter_bones_seed.py, "paper Appendix E + plan D16").
comptime SEED_ELEVATED_Z: Float64 = 0.20
comptime SEED_ELEVATED_S: Float64 = 0.6
comptime SEED_PENETRATE_Z_HARD: Float64 = -0.12
comptime SEED_PENETRATE_Z_SOFT: Float64 = -0.08
comptime SEED_PENETRATE_FRAC: Float64 = 0.80
comptime SEED_FLOAT_Z: Float64 = 0.02
comptime SEED_FLOAT_S: Float64 = 0.5
comptime SEED_UPRIGHT_GZ: Float64 = 0.9
comptime SEED_STAND_Z_LO: Float64 = 0.60
comptime SEED_STAND_Z_HI: Float64 = 0.85
comptime SEED_STILL_V: Float64 = 0.1
comptime SEED_STILL_QD: Float64 = 0.5
comptime SEED_FILTER_FPS: Float64 = 50.0
comptime SEED_N_FOOT_SPHERES: Int = 8   # 4 per foot on our G1

# 71-d row columns (data/bones_seed.mojo's MimicMotion)
comptime _C_POS: Int = 0
comptime _C_QUAT: Int = 3
comptime _C_LINVEL: Int = 36
comptime _C_QD: Int = 42


struct SeedBodyHeights(Movable):
    """Per frame: `zmin`, the lowest of the 30 link origins' world z, and
    `foot_z`, the lowest point of the foot collision geoms."""
    var n: Int
    var zmin: List[Float64]
    var foot_z: List[Float64]

    def __init__(out self, n: Int):
        self.n = n
        self.zmin = List[Float64](length=n, fill=0.0)
        self.foot_z = List[Float64](length=n, fill=0.0)

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.zmin = move.zmin^
        self.foot_z = move.foot_z^


def longest_run(mask: List[Bool]) -> Int:
    """The longest run of consecutive `True` (`_longest_run`)."""
    var best = 0
    var cur = 0
    for i in range(len(mask)):
        cur = cur + 1 if mask[i] else 0
        best = max(best, cur)
    return best


struct _FootSpheres(Movable):
    var body: List[Int]
    var pos: List[Float64]      # k × 3, body frame
    var radius: List[Float64]

    def __init__(out self):
        self.body = List[Int]()
        self.pos = List[Float64]()
        self.radius = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.body = move.body^
        self.pos = move.pos^
        self.radius = move.radius^


def _foot_spheres(env: UnitreeG1[DType.float64]) raises -> _FootSpheres:
    """Our feet's collision geoms: every colliding sphere on an ankle-roll link
    or on a body hanging off one (the left foot's spheres sit in the massless
    `dummy_lf_*` frames, the right foot's directly on the link). Refuses any
    other count, so a changed asset cannot silently move the sole."""
    var out = _FootSpheres()
    var n_left = 0
    var n_right = 0
    for g in range(UnitreeG1[DType.float64].NGEOM):
        var o = g * MODEL_GEOM_SIZE
        if Int(env.mf.geoms.data[o + GEOM_IDX_TYPE]) != GEOM_SPHERE:
            continue
        if Int(env.mf.geoms.data[o + GEOM_IDX_CONTYPE]) == 0 and Int(env.mf.geoms.data[o + GEOM_IDX_CONAFFINITY]) == 0:
            continue
        var b = Int(env.mf.geoms.data[o + GEOM_IDX_BODY])
        var side = b
        if side != LEFT_ANKLE_ROLL_BODY_IDX and side != RIGHT_ANKLE_ROLL_BODY_IDX and b > 0:
            side = Int(env.mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_PARENT])
        if side == LEFT_ANKLE_ROLL_BODY_IDX:
            n_left += 1
        elif side == RIGHT_ANKLE_ROLL_BODY_IDX:
            n_right += 1
        else:
            raise Error("bones_seed_filter: colliding sphere geom " + String(g) + " is not on a foot (body " + String(b) + ")")
        out.body.append(b)
        out.pos.append(Float64(env.mf.geoms.data[o + GEOM_IDX_POS_X]))
        out.pos.append(Float64(env.mf.geoms.data[o + GEOM_IDX_POS_Y]))
        out.pos.append(Float64(env.mf.geoms.data[o + GEOM_IDX_POS_Z]))
        out.radius.append(Float64(env.mf.geoms.data[o + GEOM_IDX_RADIUS]))
    if n_left != SEED_N_FOOT_SPHERES // 2 or n_right != SEED_N_FOOT_SPHERES // 2:
        raise Error(
            "bones_seed_filter: expected 4 foot spheres per foot, found " + String(n_left) + " left, "
            + String(n_right) + " right"
        )
    return out^


def seed_body_heights(motion: MimicMotion, mut env: UnitreeG1[DType.float64]) raises -> SeedBodyHeights:
    """FK of every frame (pose only) on our G1: the 30 link origins' lowest z
    and the feet's lowest point. Leaves `env` in the clip's last pose."""
    var feet = _foot_spheres(env)
    var out = SeedBodyHeights(motion.n)
    var qp = List[Float64](length=UnitreeG1[DType.float64].NQ, fill=0.0)
    var qv = List[Float64](length=UnitreeG1[DType.float64].NV, fill=0.0)
    for f in range(motion.n):
        var r = f * SEED_REF_DIM
        for k in range(7 + G1_N_DOF):
            qp[k] = motion.frames[r + k]
        env.set_state(qp, qv)
        var zmin = Float64.MAX
        for s in range(G1_N_SKELETON):
            zmin = min(zmin, Float64(env.d.xpos.data[g1_skeleton_body(s) * 3 + 2]))
        out.zmin[f] = zmin
        var fz = Float64.MAX
        for k in range(len(feet.body)):
            var b = feet.body[k]
            # xquat is (x, y, z, w); world z of R·p
            var qx = Float64(env.d.xquat.data[b * 4 + 0])
            var qy = Float64(env.d.xquat.data[b * 4 + 1])
            var qz = Float64(env.d.xquat.data[b * 4 + 2])
            var qw = Float64(env.d.xquat.data[b * 4 + 3])
            var px = feet.pos[k * 3 + 0]
            var py = feet.pos[k * 3 + 1]
            var pz = feet.pos[k * 3 + 2]
            var rz = (2.0 * (qx * qz - qw * qy)) * px + (2.0 * (qy * qz + qw * qx)) * py + (1.0 - 2.0 * (qx * qx + qy * qy)) * pz
            fz = min(fz, Float64(env.d.xpos.data[b * 3 + 2]) + rz - feet.radius[k])
        out.foot_z[f] = fz
    return out^


def feasibility_reason(motion: MimicMotion, heights: SeedBodyHeights) raises -> String:
    """`feasibility_filter`'s verdict from the clip's rows and its body
    heights: `""` keeps, otherwise the reason."""
    if motion.fps != SEED_FILTER_FPS:
        raise Error("bones_seed_filter: the thresholds are in 50 Hz frames, got " + String(motion.fps) + " Hz")
    if heights.n != motion.n:
        raise Error("bones_seed_filter: " + String(heights.n) + " heights for " + String(motion.n) + " frames")
    var n = motion.n
    var elevated = List[Bool](length=n, fill=False)
    var any_hard = False
    var n_soft = 0
    for f in range(n):
        elevated[f] = heights.zmin[f] > SEED_ELEVATED_Z
        any_hard = any_hard or heights.zmin[f] < SEED_PENETRATE_Z_HARD
        if heights.zmin[f] < SEED_PENETRATE_Z_SOFT:
            n_soft += 1
    if Float64(longest_run(elevated)) >= SEED_ELEVATED_S * SEED_FILTER_FPS:
        return "elevated_support"
    if any_hard or Float64(n_soft) / Float64(n) > SEED_PENETRATE_FRAC:
        return "ground_penetration"
    var gate = List[Bool](length=n, fill=False)
    var floating = List[Bool](length=n, fill=False)
    for f in range(n):
        var r = f * SEED_REF_DIM
        var qx = motion.frames[r + _C_QUAT + 1]
        var qy = motion.frames[r + _C_QUAT + 2]
        var gz = -(1.0 - 2.0 * (qx * qx + qy * qy))
        var pz = motion.frames[r + _C_POS + 2]
        var vx = motion.frames[r + _C_LINVEL + 0]
        var vy = motion.frames[r + _C_LINVEL + 1]
        var vz = motion.frames[r + _C_LINVEL + 2]
        var ss = 0.0
        for j in range(G1_N_DOF):
            var w = motion.frames[r + _C_QD + j]
            ss += w * w
        var upright = gz < -SEED_UPRIGHT_GZ
        var standing = pz > SEED_STAND_Z_LO and pz < SEED_STAND_Z_HI
        var still = sqrt(vx * vx + vy * vy + vz * vz) < SEED_STILL_V and sqrt(ss / Float64(G1_N_DOF)) < SEED_STILL_QD
        gate[f] = upright and standing and still
        floating[f] = gate[f] and heights.foot_z[f] > SEED_FLOAT_Z
    if Float64(longest_run(gate)) >= SEED_FLOAT_S * SEED_FILTER_FPS:
        if Float64(longest_run(floating)) >= SEED_FLOAT_S * SEED_FILTER_FPS:
            return "floating_feet"
    return ""


def seed_feasibility(motion: MimicMotion, mut env: UnitreeG1[DType.float64]) raises -> String:
    """The verdict for one 50 Hz clip on our G1 (`""` keeps)."""
    var h = seed_body_heights(motion, env)
    return feasibility_reason(motion, h)
