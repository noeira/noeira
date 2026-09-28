# +--------------------------------------------------------------------------+ #
# | The so101_tower expert's PLAN — object poses in, joint targets out
# +--------------------------------------------------------------------------+ #
"""The scripted cube-in-bowl expert's planning half, as a library.

`examples/so101/tower_expert_record.mojo` is the SIM executor of these plans
(it steps the env, records demos, adds noise, closes the loop on the servo
sag); an executor on the REAL arm drives the same plans through the servos.
The split is PLAN (here: poses -> joint targets, pure kinematics plus a static
collision pass on a sim copy of the scene) vs EXECUTE (there: stepping,
triggers, recording).

    var planner = TowerGraspPlanner(env, body_names)     # the arm's FK/IK on `env`
    planner.posture.human = True                        # the operator's grasp, drawn
    planner.clear_plan = True                           # a grasp the arm can reach
    planner.set_support(body_names, ["bowl_bowl"])       # optional: pick OUT of the bowl
    var plan = planner.plan_grasp(env, brick_pos, brick_yaw, q_start, jaw_open)
    planner.plan_place(env, plan, bowl_pos)
    for leg in plan.legs(place=True, close_steps=1):     # the sequence the recorder runs
        ...

## Inputs, outputs, units

- `env`: a HOST `TowerExpertEnv` (the teleop config, float64, CPU). The planner
  uses its forward kinematics (IK) and, with `clear_plan`, its contact
  detection. For a real-arm plan, place the brick's and bowl's free joints at
  the ESTIMATED poses first: the collision pass reads them from the env.
- Poses in WORLD metres (the arm's base at the origin, +x forward, +y left);
  `brick_yaw` in rad about +z; joints in MODEL radians at the SIM's joint zero
  (the follower zero and LeRobot units are the executor's business —
  `tasks/so101_tower_rig.mojo`'s `So101TowerUnits`).
- `TowerGraspPlan`: the legs' joint targets (`q_pre`, `waypoints`, `q_grasp`,
  `q_lift`, `q_carry`, `q_place`), `tip_goal` (the fingertip point the close
  trigger measures against, with `fingertip_point`), `close_on_tip`, the drawn
  posture and the plan's diagnostics. `legs()` lists them as data.

## ⚠ The sim executor re-plans the PLACE from the live bowl

`plan_place` is a separate call: the recorder calls it after the lift with the
bowl where it IS (a grasp can nudge it), which keeps its demos byte-identical
to the pre-library expert; a real executor calls it once with the estimated
pose. Same function either way.

## ⚠ Random draws

`plan_grasp` draws the posture (and, with `clear_plan`, redraws it) from the
GLOBAL `random_float64` stream, in the order the pre-library expert did — the
recorder's noise draws interleave with these, and a reordering changes every
demo's bytes.
"""

from std.math import atan2, cos, floor, pi, sin, sqrt
from std.random import random_float64, seed as seed_rng

from noeira.core.cont_action import ContAction
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.math3d import Quat, Vec3
from noeira.physics3d.collision.broadphase_sap import detect_contacts_auto
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST, CONTACT_IDX_NZ, CONTACT_SIZE, META_IDX_NUM_CONTACTS,
)
from noeira.tasks.family_config import So101TowerTeleopConfig
from noeira.tasks.so101_tower_xml import So101TowerModel


comptime TowerExpertEnv = Phyics3dEnv[
    So101TowerModel, So101TowerTeleopConfig, DType.float64, False
]
"""The host env the planner (and the sim executor) runs on."""
comptime E = TowerExpertEnv
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime ACT = 6
comptime N_ARM = 5
comptime GS = So101TowerTeleopConfig.GRIPPER_SITE
comptime GRIPPER_BODY = So101TowerTeleopConfig.GRIPPER_BODY

comptime TIP_REACH: Float64 = 0.019
"""`grasp_center` (gripper z -0.068) to the fingertip boxes' middle (z
-0.087), along the finger. A vertical grasp moves that offset straight up and
`Z_GRASP` absorbs it; a grasp tilted by 35 deg moves the tips 11 mm sideways
off the brick, which then meets one jaw and tips over (2/20, wrist view of
the human-posture run, 2026-09-23). So a drawn tilt aims the tips."""
comptime IK_ROWS = 7
"""Position (3), the finger's horizontal components (2), the pinch axis's
yaw (sin, 1) and its direction (1 - cos, 1; weighted only in `--posture
human`, where the wrist roll's SIGN matters: the wrist camera turns with it,
and an unweighted fold left a quarter of the grasps at roll -79 against the
operator's +77)."""
comptime HUMAN_JAW_OPEN: Float64 = 0.6
comptime HUMAN_Z_GRASP: Float64 = 0.015
comptime PLAN_TRIES = 12
"""`--clear-plan`: postures drawn per episode before the least-colliding one
is taken anyway."""
comptime PLAN_PINCH_TRIES = 0
"""`--clear-plan`: redraws that change only the PINCH and keep the drawn tilt.
0: measured, it does not undo the low-tilt skew (successes' median tilt 23 ->
26 deg) and cost 73 -> 62/150 on the same seeds; the skew is compensated in
the tilt RANGE instead (see `CLEAR_PLAN_TILT`)."""
comptime CLEAR_PLAN_TILT = "20,65"
"""`--clear-plan`'s tilt DRAW (deg), unless `--tilt-range` is given: the
collision redraws reject steep tilts more often, so the drawn range sits above
the operator's for the KEPT episodes to match it. Measured (150 draws, seed
61000): draw 10..55 -> successes' tilt median 23 (73/150); 20..65 -> median 35,
p5..p95 22..58 (67/150; the operator: median 35, printed set 43); 25..70 -> 36
(60/150); 30..70 -> 40 (56/150)."""
comptime DESK_CLEAR_M: Float64 = -0.008
"""`--clear-plan`: where the fingers' lowest point is planned relative to the
desk at the grasp — NEGATIVE = pressed INTO it by that much (the finger boxes
reach 17 mm past the aimed tip point, so a fixed grasp height put them 9-29
mm into the desk and the close fired 20-40 mm off target). Measured on 150
draws (seed 61000, real-layout regions): hovering 2 mm ABOVE 31/150 (tips on
target, but the close shoves the hovering arm aside), pressed 3 mm 57, 8 mm
73, 12 mm 67, 16 mm 60, 22 mm 57; the fixed height (no flag) 55. The press
is what lets the moving jaw squeeze the brick against a fixed finger that
does not yield. `--desk-clear-mm` overrides it."""
comptime SUPPORT_UP_COS: Float64 = 0.8
"""`support_by_normal`: a contact with a SUPPORT body counts as the support
(the height loop) when its normal is within ~37 deg of vertical, and as a wall
(the veto) otherwise. Our contact rows carry bodies, not geoms, so the bowl's
floor and its walls are told apart by the normal."""
comptime PATH_LEGS = 5
"""The legs `path_report` samples: 0 pre (start -> pre-grasp, jaw start ->
open, vs obstacles + desk), 1 descent (the approach waypoints, jaw open, vs
obstacles incl. the brick, not the desk — the press is intended), 2 lift (jaw
closed, vs obstacles minus the brick), 3 carry and 4 place (the arm AND the
carried brick vs obstacles minus the brick, plus the desk)."""
comptime PATH_SAMPLES = 6
"""Points per leg (the leg's end included, its start not)."""
comptime VIA_DZ: Float64 = 0.10
"""`via_mode`: the via-point is the pre-grasp point this much higher (m), same
tilt and pinch — the arm unfolds UP and over, then comes down to the pre-grasp
pose, instead of sweeping low from the fold (the pre leg's joint-space line
crossed the bowl or the brick in 7/10 bowl and 7/14 pushed-brick failures)."""
comptime N_VIA = 30
"""The via leg's step budget (the pre leg keeps `N_PRE` after it)."""
comptime FALLBACK_TILT_DEG: Float64 = 15.0
comptime FALLBACK_STEPS = 3
"""`fallback`: when none of the `PLAN_TRIES` drawn postures is clear, the
near-vertical tilts 15, 10, 5 deg (`FALLBACK_TILT_DEG` down by 5 per step),
`FALLBACK_PINCHES` pinch draws each, before the
least-colliding one is taken. Not 0: a vertical grasp switches the
executor's close to the height trigger, a live brick read (sim only)."""
comptime FALLBACK_PINCHES = 3
comptime PLAN_PEN_OK_MM: Float64 = 1.0
"""`--clear-plan`: a plan whose poses penetrate an obstacle by more is
redrawn."""
comptime APPROACH_D: Float64 = 0.07
"""A tilted grasp's approach length (m): the pre-grasp pose puts the tips this
far back ALONG THE FINGER AXIS from their grasp point (the vertical expert's
`Z_PRE - Z_GRASP`), and the descent follows that line."""
comptime APPROACH_WAYPOINTS = 3
"""IK solutions along the approach line; the joints interpolate between them,
so the tips stay near the line instead of cutting a chord."""
comptime ROLL_SEED_HUMAN: Float64 = 1.34
"""wrist_roll's seed on the tangential pinch (rad, 77 deg): the operator's
median grasp roll through the follower zero. The pinch axis is folded mod
180 deg, so without it the IK picks either sign."""
comptime Z_PRE = 0.08
comptime Z_GRASP = 0.01
"""The gripper site's height above the brick CENTRE at the grasp. It was
0.03 — measured against the jaws' phantom hull, before the box jaws — and
the tips only brushed the brick's top edge, so the grasp worked by the slow
ramp sweeping the brick into the fixed jaw. Sweep on 20 placements, stiff
jaws (close steps 15 / 1): 0.03 16/0, 0.025 16/0, 0.02 16/0, 0.015 16/13,
0.01 16/16, 0.005 16/16. At 0.01 a one-step close grasps as well as the
ramp (30/40 on other seeds; 25/40 with flat noise 0.02)."""
comptime Z_LIFT = 0.15
comptime Z_CARRY = 0.16
comptime Z_PLACE = 0.10

# The legs' step budgets at 31.25 Hz (the sim executor's; a real executor may
# rescale them to its own rate).
comptime N_PRE = 40
comptime N_DESCEND = 40
comptime N_CLOSE = 1
"""The close is a STEP: the jaw command goes to closed at once. It was a
30-step ramp, then 15; a step labels the arrival state with the full-range
jump the fitted policy needs (a ramp's first row is 1/30 of the range below
open, next to the approach's "open" at the same state, and the policy's jaw
never moved). A step only works with the jaw DEEP enough around the brick —
see `Z_GRASP`: at the old 0.03 the tip brushed the brick's top edge (0.14 mm
of overlap, in MuJoCo too) and a fast close missed it every time (0/20)."""
comptime N_LIFT = 40
comptime N_CARRY = 50
comptime N_PLACE = 25
comptime N_OPEN = 20
comptime N_RETREAT = 20
comptime N_HOLD_MAX = 120
"""Steps to wait for the predicate to hold after the last leg."""


# ── kinematics on the env ────────────────────────────────────────────────


struct TowerArm(Movable):
    """The IK's view of the arm: FK through the env, joint limits, seeds."""

    var lo: List[Float64]
    var hi: List[Float64]
    var seeds: List[List[Float64]]

    def __init__(out self, mut env: E) raises:
        var sf = So101TowerModel.make_spec_fields[DType.float64]()
        var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, ACT)
        var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, ACT)
        self.lo = List[Float64]()
        self.hi = List[Float64]()
        for i in range(ACT):
            self.lo.append(Float64(lo_col[i]))
            self.hi.append(Float64(hi_col[i]))
        self.seeds = List[List[Float64]]()
        self.seeds.append([0.0, -0.5, 0.5, 0.5, 0.0])
        self.seeds.append([0.0, 0.3, 0.8, 0.4, 0.0])
        self.seeds.append([0.0, 0.8, 0.0, 0.0, 0.0])
        self.seeds.append([0.0, -1.0, 1.2, 1.0, 0.0])
        _ = env

    def fk(
        self, mut env: E, ref q: List[Float64],
        mut p: List[Float64], mut fz: List[Float64], mut gx: List[Float64],
    ):
        """Site position, the finger axis (gripper -z) and the pinch axis
        (gripper +x) in the world, at arm joints `q` — the env's FK."""
        for i in range(N_ARM):
            env.d.qpos.data[i] = q[i]
        env._fields_fk()
        for k in range(3):
            p[k] = Float64(env.d.site_xpos.data[GS * 3 + k])
        var o = GRIPPER_BODY * 4
        # xquat is stored (x, y, z, w); Quat is (w, x, y, z)
        var qw = Quat(
            Float64(env.d.xquat.data[o + 3]), Float64(env.d.xquat.data[o]),
            Float64(env.d.xquat.data[o + 1]), Float64(env.d.xquat.data[o + 2]),
        )
        var f = qw.rotate_vec(Vec3(0.0, 0.0, -1.0))
        var g = qw.rotate_vec(Vec3(1.0, 0.0, 0.0))
        fz[0] = f.x
        fz[1] = f.y
        fz[2] = f.z
        gx[0] = g.x
        gx[1] = g.y
        gx[2] = g.z

    def _feats(
        self, mut env: E, ref q: List[Float64], yaw: Float64,
        mut out: List[Float64], tip: Bool = False,
    ):
        """`tip`: the position feature is the FINGERTIP point, `TIP_REACH`
        further along the finger than `grasp_center` (see `TIP_REACH`)."""
        var p = List[Float64](length=3, fill=0.0)
        var fz = List[Float64](length=3, fill=0.0)
        var gx = List[Float64](length=3, fill=0.0)
        self.fk(env, q, p, fz, gx)
        var d = TIP_REACH if tip else 0.0
        out[0] = p[0] + d * fz[0]
        out[1] = p[1] + d * fz[1]
        out[2] = p[2] + d * fz[2]
        out[3] = fz[0]
        out[4] = fz[1]
        # sin of the yaw error between the pinch axis and the wanted axis
        out[5] = gx[0] * sin(yaw) - gx[1] * cos(yaw)
        # 1 - cos of it: zero only when the axis points the WANTED WAY, so
        # it picks the wrist roll's sign (weighted only when `directed`)
        out[6] = 1.0 - (gx[0] * cos(yaw) + gx[1] * sin(yaw))

    def _ik_once(
        self, mut env: E, ref target: List[Float64], ref q0: List[Float64],
        yaw: Float64, w_tilt: Float64, w_yaw: Float64, iters: Int,
        mut q_out: List[Float64], tilt: Float64 = 0.0, directed: Bool = False,
    ) -> Float64:
        """Damped least squares from `q0`; returns the position error.
        `tilt` (rad): the finger's wanted angle from straight down, leaning
        OUTWARD along the target's radial direction (0 = vertical)."""
        var rad = atan2(target[1], target[0])
        var fgoal_x = sin(tilt) * cos(rad)
        var fgoal_y = sin(tilt) * sin(rad)
        # a drawn tilt aims the FINGERTIPS, not `grasp_center`; the target
        # moves down by the same `TIP_REACH`, so at tilt 0 it is the old one
        var tip = tilt > 0.0
        var tgt = List[Float64]()
        tgt.append(target[0])
        tgt.append(target[1])
        tgt.append(target[2] - (TIP_REACH if tip else 0.0))
        var q = List[Float64]()
        for i in range(N_ARM):
            q.append(q0[i])
        var W = List[Float64]()
        W.append(1.0)
        W.append(1.0)
        W.append(1.0)
        W.append(w_tilt)
        W.append(w_tilt)
        W.append(w_yaw)
        # the directed-pinch row: weight 0 unless `directed`, so the default
        # expert's normal system is unchanged, bit for bit
        W.append(w_yaw if directed else 0.0)
        var f = List[Float64](length=IK_ROWS, fill=0.0)
        var f2 = List[Float64](length=IK_ROWS, fill=0.0)
        var e = List[Float64](length=IK_ROWS, fill=0.0)
        var J = List[Float64](length=IK_ROWS * N_ARM, fill=0.0)
        var H = List[Float64](length=N_ARM * N_ARM, fill=0.0)
        var g = List[Float64](length=N_ARM, fill=0.0)
        var perr = 1.0
        for _ in range(iters):
            self._feats(env, q, yaw, f, tip)
            for r in range(IK_ROWS):
                var goal = tgt[r] if r < 3 else (
                    fgoal_x if r == 3 else (fgoal_y if r == 4 else 0.0)
                )
                e[r] = (goal - f[r]) * W[r]
            perr = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
            var dfx = f[3] - fgoal_x
            var dfy = f[4] - fgoal_y
            if perr < 5e-4 and sqrt(dfx * dfx + dfy * dfy) < 0.05:
                break
            var eps = 1e-6
            for j in range(N_ARM):
                var dq = List[Float64]()
                for i in range(N_ARM):
                    dq.append(q[i])
                dq[j] += eps
                self._feats(env, dq, yaw, f2, tip)
                for r in range(IK_ROWS):
                    J[r * N_ARM + j] = (f2[r] - f[r]) * W[r] / eps
            # H = J^T J + lambda I,  g = J^T e
            for a in range(N_ARM):
                g[a] = 0.0
                for b in range(N_ARM):
                    var s = 0.0
                    for r in range(IK_ROWS):
                        s += J[r * N_ARM + a] * J[r * N_ARM + b]
                    H[a * N_ARM + b] = s + (1e-5 if a == b else 0.0)
                for r in range(IK_ROWS):
                    g[a] += J[r * N_ARM + a] * e[r]
            var step = _solve5(H, g)
            for i in range(N_ARM):
                var s = step[i]
                if s > 0.25:
                    s = 0.25
                if s < -0.25:
                    s = -0.25
                q[i] += s
                if q[i] < self.lo[i]:
                    q[i] = self.lo[i]
                if q[i] > self.hi[i]:
                    q[i] = self.hi[i]
        for i in range(N_ARM):
            q_out[i] = q[i]
        return perr

    def ik(
        self, mut env: E, ref target: List[Float64], ref q0: List[Float64],
        yaw: Float64, mut q_out: List[Float64], tilt: Float64 = 0.0,
        roll_seed: Float64 = 0.0, use_roll_seed: Bool = False,
    ) -> Float64:
        """Restarts x a relaxing tilt weight; keeps the best position error.
        Restores the env's arm qpos to what it was.

        `tilt` > 0 targets a finger leaning outward by that angle (the human
        posture, `--posture human`); its weights start 6x stronger, since a
        drawn tilt is a TARGET and the vertical one was only a preference.
        `use_roll_seed` starts every restart's wrist_roll at `roll_seed`:
        the pinch axis is folded mod 180 deg, so the roll's SIGN is chosen by
        the seed (the human's is always positive)."""
        var saved = List[Float64]()
        for i in range(NQ):
            saved.append(Float64(env.d.qpos.data[i]))
        var best = 1e9
        var q_try = List[Float64](length=N_ARM, fill=0.0)
        var weights: List[Float64] = [0.05, 0.02, 0.008, 0.003]
        if tilt > 0.0:
            weights = [0.3, 0.12, 0.05, 0.02]
        var pan = atan2(target[1], target[0])
        for wi in range(len(weights)):
            for s in range(len(self.seeds) + 1):
                var s0 = List[Float64]()
                for i in range(N_ARM):
                    s0.append(q0[i] if s == 0 else self.seeds[s - 1][i])
                s0[0] = pan
                if use_roll_seed and s > 0:
                    s0[4] = roll_seed
                var err = self._ik_once(
                    env, target, s0, yaw, weights[wi], 0.02, 100, q_try, tilt,
                    use_roll_seed,
                )
                if err < best - 1e-4:
                    best = err
                    for i in range(N_ARM):
                        q_out[i] = q_try[i]
            if best < 1e-3:
                break
        for i in range(NQ):
            env.d.qpos.data[i] = saved[i]
        env._fields_fk()
        return best


def _solve5(ref H: List[Float64], ref g: List[Float64]) -> List[Float64]:
    """Gaussian elimination with partial pivoting on the 5x5 normal system."""
    var n = N_ARM
    var A = List[Float64]()
    for i in range(n * n):
        A.append(H[i])
    var b = List[Float64]()
    for i in range(n):
        b.append(g[i])
    for c in range(n):
        var piv = c
        for r in range(c + 1, n):
            if abs(A[r * n + c]) > abs(A[piv * n + c]):
                piv = r
        if piv != c:
            for k in range(n):
                var t = A[c * n + k]
                A[c * n + k] = A[piv * n + k]
                A[piv * n + k] = t
            var tb = b[c]
            b[c] = b[piv]
            b[piv] = tb
        var d = A[c * n + c]
        if abs(d) < 1e-14:
            continue
        for r in range(c + 1, n):
            var f = A[r * n + c] / d
            for k in range(c, n):
                A[r * n + k] -= f * A[c * n + k]
            b[r] -= f * b[c]
    var x = List[Float64](length=n, fill=0.0)
    for c in range(n - 1, -1, -1):
        var s = b[c]
        for k in range(c + 1, n):
            s -= A[c * n + k] * x[k]
        var d = A[c * n + c]
        x[c] = s / d if abs(d) > 1e-14 else 0.0
    return x^


# ── the posture ──────────────────────────────────────────────────────────


struct GraspPosture(Copyable, Movable):
    """The grasp's drawn posture: vertical (the default expert) or the
    operator's tilt and pinch (`human`)."""

    var human: Bool
    """`--posture human`: per episode, a drawn tilt and a face-snapped pinch
    that cover the operator's measured grasps (see `draw`)."""
    var tilt_lo: Float64
    var tilt_hi: Float64
    var tilt: Float64       # this episode's drawn tilt (rad), 0 = vertical
    var tangential: Bool    # kept for the log: the snapped face is closer to tangential than radial
    var pinch_lo: Float64
    var pinch_hi: Float64
    var pinch_target: Float64  # this episode's drawn pinch from radial (rad), before the snap

    def __init__(out self):
        self.human = False
        self.tilt_lo = 10.0 * pi / 180.0
        self.tilt_hi = 55.0 * pi / 180.0
        self.tilt = 0.0
        self.tangential = False
        self.pinch_lo = 35.0 * pi / 180.0
        self.pinch_hi = 85.0 * pi / 180.0
        self.pinch_target = 0.0

    def draw(mut self):
        """The operator's grasp, drawn — `--posture human`.

        Measured on the 50 kept real cube-in-bowl episodes at the grasp frame
        (`tools/soarm/grasp_posture.py`, 2026-09-23), in the expert's own IK
        features: TILT, the finger's angle from straight down leaning
        outward, p5..p95 = 9..60 deg, median 35; PINCH, the pinch axis from
        radial folded to (-90, 90], 34..86, median 72 — always on the
        positive side (wrist_roll 35..103, median 77). The vertical expert
        was tilt 0.2 / pinch 0, and students trained on it ignored real
        frames (40-47 deg mean action error on the real import).

        Tilt ~ U(`tilt_lo`, `tilt_hi`); pinch target ~ U(`pinch_lo`,
        `pinch_hi`), then SNAPPED to the brick's nearest face normal
        (`pinch_yaw`): a pinch between two face normals closes on corners.
        With the cube axis-aligned (today's tasks) the snap moves a 45-deg
        world line to 0 or 90, so the realised pinch is mirrored (-65 median,
        measured) — the operator's real Duplo lies at arbitrary yaw; the
        task's opt-in brick yaw draw makes the snap reproduce it."""
        self.tilt = random_float64(self.tilt_lo, self.tilt_hi)
        self.pinch_target = random_float64(self.pinch_lo, self.pinch_hi)

    def draw_pinch(mut self):
        """Redraw the pinch only, keeping the drawn tilt (`--clear-plan`)."""
        self.pinch_target = random_float64(self.pinch_lo, self.pinch_hi)

    def pinch_yaw(mut self, bearing: Float64, brick_yaw: Float64) -> Float64:
        """The pinch axis's world yaw for a brick at `bearing` whose own yaw
        is `brick_yaw` (rad): the radial direction, or — in `--posture
        human` — the brick face normal nearest `bearing + pinch_target`."""
        if not self.human:
            return bearing
        var want = bearing + self.pinch_target
        var k = Float64(Int(floor((want - brick_yaw) / (pi / 2.0) + 0.5)))
        var y = brick_yaw + k * (pi / 2.0)
        var rel = y - bearing
        # the log's "tangential": the snapped line is closer to 90 than to 0
        var c = cos(rel)
        self.tangential = c * c < 0.5
        return y


# ── the plan ─────────────────────────────────────────────────────────────


@fieldwise_init
struct PlanLeg(Copyable, Movable):
    """One leg of the sequence, as data. `q` is empty for a HOLD (keep the
    arm where it is commanded)."""

    var name: String
    var q: List[Float64]
    """The arm's joint target, MODEL radians (empty: hold)."""
    var grip_open: Bool
    """The jaw's target: open (`jaw_open`) or closed (its lower limit)."""
    var steps: Int
    """The step budget at 31.25 Hz."""
    var close_on_tip: Bool
    """The close fires DURING this leg once the fingertip point is within the
    executor's tip distance of `tip_goal` and the arm has settled; the leg
    ends there."""
    var until_held: Bool
    """Hold until the task's goal holds (the sim's success test; a real
    executor substitutes its own)."""


struct TowerGraspPlan(Copyable, Movable):
    var q_start: List[Float64]
    var q_pre: List[Float64]
    var q_via: List[Float64]
    """The via-point before the pre-grasp (`via_mode`); empty: none."""
    var waypoints: List[List[Float64]]
    """The tilted approach ALONG THE FINGER: `waypoints[0]` is `q_pre`, the
    last is `q_grasp`; empty for a vertical grasp."""
    var q_grasp: List[Float64]
    var q_lift: List[Float64]
    var q_carry: List[Float64]
    var q_place: List[Float64]
    var tip_goal: List[Float64]
    """The fingertip point's goal at the grasp (world m): the close trigger's
    reference (`fingertip_point`)."""
    var close_on_tip: Bool
    """The close fires on the fingertip distance (a tilted grasp). False: the
    vertical expert's height trigger (a live brick read, sim only)."""
    var bearing: Float64
    var yaw: Float64
    """The pinch axis's world yaw (rad)."""
    var tilt: Float64
    var tangential: Bool
    var e_pre: Float64
    var e_grasp: Float64
    var e_lift: Float64
    var e_carry: Float64
    var e_place: Float64
    """IK position errors (m)."""
    var raise_mm: Float64
    var pen_mm: Float64
    var tries: Int
    """`clear_plan` only: the grasp height's raise, the worst obstacle
    penetration left, the postures drawn."""
    var placed: Bool
    var path_pen_mm: List[Float64]
    """`path_report`: the worst penetration along each leg's JOINT-SPACE path
    (the executor ramps linearly), in `PATH_LEGS` order; -1 = not checked."""

    def __init__(out self):
        self.q_start = List[Float64](length=N_ARM, fill=0.0)
        self.q_pre = List[Float64](length=N_ARM, fill=0.0)
        self.q_via = List[Float64]()
        self.waypoints = List[List[Float64]]()
        self.q_grasp = List[Float64](length=N_ARM, fill=0.0)
        self.q_lift = List[Float64](length=N_ARM, fill=0.0)
        self.q_carry = List[Float64](length=N_ARM, fill=0.0)
        self.q_place = List[Float64](length=N_ARM, fill=0.0)
        self.tip_goal = List[Float64](length=3, fill=0.0)
        self.close_on_tip = False
        self.bearing = 0.0
        self.yaw = 0.0
        self.tilt = 0.0
        self.tangential = False
        self.e_pre = 0.0
        self.e_grasp = 0.0
        self.e_lift = 0.0
        self.e_carry = 0.0
        self.e_place = 0.0
        self.raise_mm = 0.0
        self.pen_mm = 0.0
        self.tries = 0
        self.placed = False
        self.path_pen_mm = List[Float64](length=PATH_LEGS, fill=-1.0)

    def legs(self, place: Bool, close_steps: Int = N_CLOSE) -> List[PlanLeg]:
        """The sequence `tower_expert_record.mojo` runs from a reset, as
        data: pre, the descent (one leg per approach waypoint, the close armed
        on the last), the close, the lift, then — `place` — carry, place, open,
        retreat, and a final hold until the goal holds. The DAgger handover
        and `--return-rest` are the recorder's own."""
        var out = List[PlanLeg]()
        var none = List[Float64]()
        if len(self.q_via) > 0:
            out.append(PlanLeg("via", self.q_via.copy(), True, N_VIA, False, False))
        out.append(PlanLeg("pre", self.q_pre.copy(), True, N_PRE, False, False))
        if len(self.waypoints) > 0:
            var n_wp = len(self.waypoints) - 1
            for j in range(1, n_wp + 1):
                out.append(PlanLeg(
                    "descend", self.waypoints[j].copy(), True, N_DESCEND // n_wp,
                    j == n_wp and self.close_on_tip, False,
                ))
        else:
            out.append(PlanLeg("descend", self.q_grasp.copy(), True, N_DESCEND, False, False))
        out.append(PlanLeg("close", none.copy(), False, close_steps, False, False))
        out.append(PlanLeg("lift", self.q_lift.copy(), False, N_LIFT, False, False))
        if place:
            out.append(PlanLeg("carry", self.q_carry.copy(), False, N_CARRY, False, False))
            out.append(PlanLeg("place", self.q_place.copy(), False, N_PLACE, False, False))
            out.append(PlanLeg("open", none.copy(), True, N_OPEN, False, False))
            out.append(PlanLeg("retreat", self.q_carry.copy(), True, N_RETREAT, False, False))
        out.append(PlanLeg("hold", none.copy(), not place, N_HOLD_MAX, False, True))
        return out^


def _above(ref p: List[Float64], dz: Float64) -> List[Float64]:
    var t = List[Float64]()
    t.append(p[0])
    t.append(p[1])
    t.append(p[2] + dz)
    return t^


def fingertip_point(mut env: E) -> List[Float64]:
    """The fingertip point (`grasp_center` + `TIP_REACH` along the finger) in
    the world, from the env's CURRENT FK — the close trigger measures it
    against `TowerGraspPlan.tip_goal`."""
    var o = GRIPPER_BODY * 4
    var qw = Quat(
        Float64(env.d.xquat.data[o + 3]), Float64(env.d.xquat.data[o]),
        Float64(env.d.xquat.data[o + 1]), Float64(env.d.xquat.data[o + 2]),
    )
    var f = qw.rotate_vec(Vec3(0.0, 0.0, -1.0))
    var p = List[Float64]()
    p.append(Float64(env.d.site_xpos.data[GS * 3]) + TIP_REACH * f.x)
    p.append(Float64(env.d.site_xpos.data[GS * 3 + 1]) + TIP_REACH * f.y)
    p.append(Float64(env.d.site_xpos.data[GS * 3 + 2]) + TIP_REACH * f.z)
    return p^


# ── the planner ──────────────────────────────────────────────────────────


struct TowerGraspPlanner(Movable):
    var arm: TowerArm
    var posture: GraspPosture
    var z_grasp: Float64
    """The grasp point's height above the brick centre (m): the vertical
    expert's `grasp_center`, the human posture's fingertip point."""
    var clear_plan: Bool
    """Plan a grasp the arm can physically reach (see `plan_grasp`)."""
    var desk_clear_m: Float64
    var desk_jaw: Float64
    """The jaw angle (rad) the desk check poses the gripper at; < -1 (the
    default): the approach's `jaw_open`. The open jaw's tip is the gripper's
    LOWEST point, and it closes UPWARD along its arc — see `plan_grasp`.
    Measured at 0.13 and -0.17: 68 and 69 vs 63/150, within noise; kept as
    an option, not a default."""
    var pinch_offset_m: Float64
    """The tilted grasp's aim along the pinch axis (m, gripper +x): the brick
    ends this far on the FIXED finger's side of `grasp_center`. 0 (the
    default): the brick at `grasp_center`, which is 31 mm from the fixed
    finger's inner face — a 25 mm brick gripped against it has its centre
    18.6 mm from there, so the moving jaw first SWEEPS the brick that far
    across the desk (measured, 25 Sep, MuJoCo and ours alike). The recorder's
    `--clear-plan` uses 12 mm (`CLEAR_PLAN_PINCH_OFFSET_MM` there: 47 -> 60 %
    on 300 draws)."""
    var arm_bodies: List[Int]
    """The moving arm's bodies (upper arm to jaw) — the collision pass's one
    side."""
    var obstacles: List[Int]
    """What they must not pass through at all: base, shoulder, the tower
    stand, the bowl, and the brick while the jaw is open."""
    var desk: List[Int]
    """The desk (its body)."""
    var support: List[Int]
    """What the grasp height is set against (`desk_clear_m` into it): the
    desk by default; `set_support` names another (the bowl, to pick the brick
    OUT of it)."""
    var support_by_normal: Bool
    var fallback: Bool
    """`clear_plan`: try the near-vertical tilts (`FALLBACK_TILT_DEG`) when no
    drawn posture is clear. ⚠ MEASURED, NOT A WIN (300 draws on b8f668315):
    217 vs 225/300 without. It reaches 45 episodes and finds a clear plan in 35,
    but those fail in the EXECUTION (16/45 succeed vs 19/49 on the least-
    colliding drawn posture): near the base a steep grasp needs the elbow and
    wrist far out (they lag 20-36 deg at the close) and the fingers push the
    brick (11) or pinch nothing (6). Off by default."""
    var via_mode: Int
    """0 none; 1 a via-point (`VIA_DZ` above the pre-grasp) when the direct
    pre leg's joint-space path collides; 2 always."""
    var path_check: Bool
    """`clear_plan` + this: a posture is also redrawn when a leg's JOINT-SPACE
    path (pre, descent, lift, and — given the bowl — carry and place) collides,
    not only its waypoints (`PATH_LEGS`). Measured: successes' paths almost
    never collide; pushed bricks, bowl hits and shoulder/stand hits mostly do."""
    var path_report: Bool
    """Measure each leg's joint-space path against the obstacles into
    `TowerGraspPlan.path_pen_mm` (report only: nothing is redrawn)."""
    var brick: Int
    """Count only the support's near-vertical contacts as the support, and
    keep its other contacts (walls) in the veto — see `SUPPORT_UP_COS`."""

    def __init__(out self, mut env: E, ref body_names: List[String]) raises:
        """`body_names`: the composed scene's bodies in index order (the
        parser's `FlatModelDef.body_names`)."""
        self.arm = TowerArm(env)
        self.posture = GraspPosture()
        self.z_grasp = Z_GRASP
        self.clear_plan = False
        self.desk_clear_m = DESK_CLEAR_M
        self.desk_jaw = -10.0
        self.pinch_offset_m = 0.0
        self.arm_bodies = List[Int]()
        self.obstacles = List[Int]()
        self.desk = List[Int]()
        for b in range(len(body_names)):
            var bn = body_names[b]
            if bn in ["robot_upper_arm", "robot_lower_arm", "robot_wrist", "robot_gripper"] or bn.startswith("robot_moving_jaw"):
                self.arm_bodies.append(b)
            elif bn in ["robot_base", "robot_shoulder", "tower_stand", "bowl_bowl", "brick_brick"]:
                self.obstacles.append(b)
            elif bn == "desk_mat":
                self.desk.append(b)
        self.support = self.desk.copy()
        self.support_by_normal = False
        self.path_check = False
        self.via_mode = 0
        self.fallback = False
        self.path_report = False
        self.brick = -1
        for b in range(len(body_names)):
            if body_names[b] == "brick_brick":
                self.brick = b
        if len(self.arm_bodies) != 5 or len(self.obstacles) != 5 or len(self.desk) != 1:
            raise Error("the planner did not find its 5 arm bodies, 5 obstacles and the desk")

    def set_support(
        mut self, ref body_names: List[String], ref names: List[String],
        by_normal: Bool = True,
    ) raises:
        """Set the grasp height against these bodies (by name) instead of the
        desk — e.g. `["bowl_bowl"]` to pick the brick out of the bowl: its
        FLOOR sets the height (`by_normal`), its WALLS stay in the veto (a
        support body may also be an obstacle)."""
        self.support = List[Int]()
        for n in names:
            var found = -1
            for b in range(len(body_names)):
                if body_names[b] == n:
                    found = b
            if found < 0:
                raise Error("set_support: no body named " + n)
            self.support.append(found)
        self.support_by_normal = by_normal

    def _obst_without_brick(self) -> List[Int]:
        var o = List[Int]()
        for b in self.obstacles:
            if b != self.brick:
                o.append(b)
        return o^

    def plan_grasp(
        mut self, mut env: E, ref pb: List[Float64], brick_yaw: Float64,
        ref q: List[Float64], jaw_open: Float64,
        pw: List[Float64] = List[Float64](),
    ) raises -> TowerGraspPlan:
        """The pick: pre-grasp, the approach, the grasp and the lift, for a
        brick at `pb` (world m) with yaw `brick_yaw`, from the arm at `q`.
        Draws this episode's posture when `posture.human`.

        ⚠ `clear_plan`: a plan the arm can physically execute. Per drawn
        posture, the grasp height is set so the fingers' lowest point sits
        `desk_clear_m` against the desk (a fixed height put the finger boxes
        9-29 mm INTO it: the descent stopped short and the close fired 20-40
        mm off target), then every waypoint is checked against the base,
        shoulder, stand, bowl and brick with the jaw open; a colliding posture
        is redrawn, up to PLAN_TRIES, and the least-colliding one is kept if
        none is clear.

        `path_check` (with `clear_plan`) also rejects a posture whose legs'
        joint-space PATHS collide; `pw` (the bowl, world m) adds the carry and
        place legs to that check (their plan is provisional: the sim executor
        re-plans the place after the lift, `plan_place`)."""
        var plan = TowerGraspPlan()
        for i in range(N_ARM):
            plan.q_start[i] = q[i]
        var bearing = atan2(pb[1], pb[0])
        plan.bearing = bearing
        if self.posture.human:
            self.posture.draw()
        var yaw = self.posture.pinch_yaw(bearing, brick_yaw)
        var tl = self.posture.tilt if self.posture.human else 0.0
        var rs = ROLL_SEED_HUMAN if self.posture.human else 0.0
        var use_rs = self.posture.human
        var e1 = 0.0
        var e2 = 0.0
        plan.close_on_tip = tl > 0.0
        if tl > 0.0:
            var n_wp = APPROACH_WAYPOINTS
            if not self.clear_plan:
                e2 = _plan_tilted(self, env, pb, bearing, yaw, tl, rs, q, 0.0, plan.waypoints, plan.tip_goal)
            else:
                var best_pen = 1.0e9
                var best_tilt = self.posture.tilt
                var best_pinch = self.posture.pinch_target
                comptime N_FB = FALLBACK_STEPS * FALLBACK_PINCHES
                var n_fb = N_FB if self.fallback else 0
                var last = PLAN_TRIES + n_fb
                for attempt in range(last + 1):
                    if attempt == last:
                        # none was clear: re-plan the least-colliding posture
                        self.posture.tilt = best_tilt
                        self.posture.pinch_target = best_pinch
                    elif attempt >= PLAN_TRIES:
                        # the near-vertical fallback (`fallback`)
                        var k = attempt - PLAN_TRIES
                        self.posture.tilt = (
                            FALLBACK_TILT_DEG - 5.0 * Float64(k // FALLBACK_PINCHES)
                        ) * pi / 180.0
                        self.posture.draw_pinch()
                    elif attempt > 0 and attempt <= PLAN_PINCH_TRIES:
                        self.posture.draw_pinch()
                    elif attempt > 0:
                        self.posture.draw()
                    yaw = self.posture.pinch_yaw(bearing, brick_yaw)
                    tl = self.posture.tilt
                    var raise_m = 0.0
                    e2 = _plan_tilted(self, env, pb, bearing, yaw, tl, rs, q, raise_m, plan.waypoints, plan.tip_goal)
                    var dj = jaw_open if self.desk_jaw < -1.0 else self.desk_jaw
                    for _ in range(3):
                        var dp = _pose_penetration_mm(
                            env, plan.waypoints[n_wp], dj, self.arm_bodies,
                            self.support, 1 if self.support_by_normal else 0,
                            self.support.copy(),
                        )
                        if dp <= 0.0:
                            break
                        raise_m += dp / 1000.0 + self.desk_clear_m
                        e2 = _plan_tilted(self, env, pb, bearing, yaw, tl, rs, q, raise_m, plan.waypoints, plan.tip_goal)
                    var pen = 0.0
                    for j in range(n_wp + 1):
                        pen = max(pen, _pose_penetration_mm(
                            env, plan.waypoints[j], jaw_open, self.arm_bodies,
                            self.obstacles, 2 if self.support_by_normal else 0,
                            self.support.copy(),
                        ))
                    if self.path_check:
                        var j0 = Float64(env.d.qpos.data[N_ARM])
                        var with_desk = self.obstacles.copy()
                        for b in self.desk:
                            with_desk.append(b)
                        pen = max(pen, _path_pen(
                            env, q.copy(), plan.waypoints[0].copy(), j0, jaw_open,
                            self.arm_bodies.copy(), with_desk.copy(),
                        ))
                        for j in range(1, n_wp + 1):
                            pen = max(pen, _path_pen(
                                env, plan.waypoints[j - 1].copy(),
                                plan.waypoints[j].copy(), jaw_open, jaw_open,
                                self.arm_bodies.copy(), self.obstacles.copy(),
                            ))
                        var nb = List[Int]()
                        for b in self.obstacles:
                            if b != self.brick:
                                nb.append(b)
                        var ql = List[Float64](length=N_ARM, fill=0.0)
                        _ = self.arm.ik(env, _above(pb, Z_LIFT), plan.waypoints[n_wp], yaw, ql, tl, rs, use_rs)
                        pen = max(pen, _path_pen(
                            env, plan.waypoints[n_wp].copy(), ql.copy(),
                            self.arm.lo[5], self.arm.lo[5], self.arm_bodies.copy(),
                            nb.copy(),
                        ))
                        if len(pw) == 3:
                            var nd = nb.copy()
                            for b in self.desk:
                                nd.append(b)
                            var qc = List[Float64](length=N_ARM, fill=0.0)
                            _ = self.arm.ik(env, _above(pw, Z_CARRY), ql, yaw, qc, tl, rs, use_rs)
                            var qp = List[Float64](length=N_ARM, fill=0.0)
                            _ = self.arm.ik(env, _above(pw, Z_PLACE), qc, yaw, qp, tl, rs, use_rs)
                            pen = max(pen, _path_pen(
                                env, ql.copy(), qc.copy(), self.arm.lo[5],
                                self.arm.lo[5], self.arm_bodies.copy(), nd.copy(),
                            ))
                            pen = max(pen, _path_pen(
                                env, qc.copy(), qp.copy(), self.arm.lo[5],
                                self.arm.lo[5], self.arm_bodies.copy(), nd.copy(),
                            ))
                    plan.raise_mm = raise_m * 1000.0
                    plan.pen_mm = pen
                    plan.tries = attempt + 1
                    if pen <= PLAN_PEN_OK_MM or attempt == last:
                        break
                    if pen < best_pen:
                        best_pen = pen
                        best_tilt = self.posture.tilt
                        best_pinch = self.posture.pinch_target
            for i in range(N_ARM):
                plan.q_pre[i] = plan.waypoints[0][i]
                plan.q_grasp[i] = plan.waypoints[n_wp][i]
        else:
            e1 = self.arm.ik(env, _above(pb, Z_PRE), q, yaw, plan.q_pre, tl, rs, use_rs)
            e2 = self.arm.ik(env, _above(pb, self.z_grasp), plan.q_pre, yaw, plan.q_grasp, tl, rs, use_rs)
        plan.e_lift = self.arm.ik(env, _above(pb, Z_LIFT), plan.q_grasp, yaw, plan.q_lift, tl, rs, use_rs)
        if self.via_mode > 0:
            var use_via = self.via_mode == 2
            if not use_via:
                var with_desk = self.obstacles.copy()
                for b in self.desk:
                    with_desk.append(b)
                use_via = _path_pen(
                    env, q.copy(), plan.q_pre.copy(), Float64(env.d.qpos.data[N_ARM]),
                    jaw_open, self.arm_bodies.copy(), with_desk.copy(),
                ) > PLAN_PEN_OK_MM
            if use_via:
                # the pre-grasp's own IK target, raised: for a tilted grasp
                # the tip point APPROACH_D back along the finger (+TIP_REACH,
                # the IK's tip mode), for a vertical one Z_PRE above the brick
                var t = List[Float64]()
                if tl > 0.0:
                    t.append(plan.tip_goal[0] - APPROACH_D * sin(tl) * cos(plan.bearing))
                    t.append(plan.tip_goal[1] - APPROACH_D * sin(tl) * sin(plan.bearing))
                    t.append(plan.tip_goal[2] + APPROACH_D * cos(tl) + TIP_REACH + VIA_DZ)
                else:
                    t = _above(pb, Z_PRE + VIA_DZ)
                plan.q_via = List[Float64](length=N_ARM, fill=0.0)
                _ = self.arm.ik(env, t, plan.q_pre, yaw, plan.q_via, tl, rs, use_rs)
        plan.e_pre = e1
        plan.e_grasp = e2
        plan.yaw = yaw
        plan.tilt = tl
        plan.tangential = self.posture.tangential
        if self.path_report:
            var j0 = Float64(env.d.qpos.data[N_ARM])
            var with_desk = self.obstacles.copy()
            for b in self.desk:
                with_desk.append(b)
            plan.path_pen_mm[0] = _path_pen(env, plan.q_start.copy(), plan.q_pre.copy(), j0, jaw_open, self.arm_bodies.copy(), with_desk.copy())
            var dmax = 0.0
            if len(plan.waypoints) > 0:
                for j in range(1, len(plan.waypoints)):
                    dmax = max(dmax, _path_pen(env, plan.waypoints[j - 1].copy(), plan.waypoints[j].copy(), jaw_open, jaw_open, self.arm_bodies.copy(), self.obstacles.copy()))
            else:
                dmax = _path_pen(env, plan.q_pre.copy(), plan.q_grasp.copy(), jaw_open, jaw_open, self.arm_bodies.copy(), self.obstacles.copy())
            plan.path_pen_mm[1] = dmax
            var nb = self._obst_without_brick()
            plan.path_pen_mm[2] = _path_pen(env, plan.q_grasp.copy(), plan.q_lift.copy(), self.arm.lo[5], self.arm.lo[5], self.arm_bodies.copy(), nb.copy())
        return plan^

    def plan_place(
        mut self, mut env: E, mut plan: TowerGraspPlan, ref pw: List[Float64],
    ) raises:
        """The place, for a bowl at `pw` (world m): carry above it, lower
        into it; retreat reuses the carry pose. Seeded from the plan's lift."""
        var tl = plan.tilt
        var rs = ROLL_SEED_HUMAN if self.posture.human else 0.0
        var use_rs = self.posture.human
        plan.e_carry = self.arm.ik(env, _above(pw, Z_CARRY), plan.q_lift, plan.yaw, plan.q_carry, tl, rs, use_rs)
        plan.e_place = self.arm.ik(env, _above(pw, Z_PLACE), plan.q_carry, plan.yaw, plan.q_place, tl, rs, use_rs)
        plan.placed = True
        if self.path_report:
            # the carried brick moves with the gripper: check it too. Its
            # pose in the sim env is where it lies NOW, so this checks the
            # ARM against the scene and the brick only as a static obstacle
            # removed from the set — the brick-in-hand vs the bowl rim is
            # the executor's to see (a static pass cannot move a free body)
            var nb = self._obst_without_brick()
            for b in self.desk:
                nb.append(b)
            plan.path_pen_mm[3] = _path_pen(env, plan.q_lift.copy(), plan.q_carry.copy(), self.arm.lo[5], self.arm.lo[5], self.arm_bodies.copy(), nb.copy())
            plan.path_pen_mm[4] = _path_pen(env, plan.q_carry.copy(), plan.q_place.copy(), self.arm.lo[5], self.arm.lo[5], self.arm_bodies.copy(), nb.copy())


def _plan_tilted(
    mut pl: TowerGraspPlanner, mut env: E, ref pb: List[Float64], bearing: Float64,
    yaw: Float64, tl: Float64, rs: Float64, ref q: List[Float64],
    raise_m: Float64, mut waypoints: List[List[Float64]],
    mut tip_goal: List[Float64],
) -> Float64:
    """The tilted grasp's approach ALONG THE FINGER: IK targets on the line
    through the tips' grasp point (raised by `raise_m`), `APPROACH_D` back
    along the finger axis, `APPROACH_WAYPOINTS` legs. Fills `waypoints`
    (the first is the pre-grasp, the last the grasp) and sets `tip_goal`;
    returns the grasp pose's IK error."""
    var fx = sin(tl) * cos(bearing)
    var fy = sin(tl) * sin(bearing)
    var fzv = -cos(tl)
    # `pinch_offset_m`: the aim moves along the pinch axis (gripper +x, which
    # the directed IK row points along `yaw`) so the brick lands BETWEEN the
    # jaws rather than at `grasp_center` — see `TowerGraspPlanner`
    tip_goal[0] = pb[0] + pl.pinch_offset_m * cos(yaw)
    tip_goal[1] = pb[1] + pl.pinch_offset_m * sin(yaw)
    tip_goal[2] = pb[2] + pl.z_grasp - TIP_REACH + raise_m
    # the IK's tip mode aims `target - (0, 0, TIP_REACH)`: hand it the
    # tip point lifted by TIP_REACH
    var n_wp = APPROACH_WAYPOINTS
    var prev = q.copy()
    var e = 0.0
    waypoints.clear()
    for j in range(n_wp + 1):
        var sback = APPROACH_D * Float64(n_wp - j) / Float64(n_wp)
        var t = List[Float64]()
        t.append(tip_goal[0] - sback * fx)
        t.append(tip_goal[1] - sback * fy)
        t.append(tip_goal[2] - sback * fzv + TIP_REACH)
        var qj = List[Float64](length=N_ARM, fill=0.0)
        e = pl.arm.ik(env, t, prev, yaw, qj, tl, rs, True)
        prev = qj.copy()
        waypoints.append(qj^)
    return e


def _path_pen(
    mut env: E, qa: List[Float64], qb: List[Float64], ja: Float64,
    jb: Float64, arm: List[Int], obst: List[Int],
) raises -> Float64:
    """The worst penetration (mm) along the straight joint-space line from
    (qa, ja) to (qb, jb), at `PATH_SAMPLES` points — the executor's ramp."""
    var worst = 0.0
    for k in range(1, PATH_SAMPLES + 1):
        var t = Float64(k) / Float64(PATH_SAMPLES)
        var q = List[Float64]()
        for i in range(N_ARM):
            q.append(qa[i] + (qb[i] - qa[i]) * t)
        worst = max(worst, _pose_penetration_mm(
            env, q, ja + (jb - ja) * t, arm, obst
        ))
    return worst


def _pose_penetration_mm(
    mut env: E, ref q: List[Float64], jaw: Float64,
    ref arm_bodies: List[Int], ref obstacles: List[Int],
    normal_mode: Int = 0, normal_bodies: List[Int] = List[Int](),
) raises -> Float64:
    """The deepest penetration (mm) between an ARM body and an OBSTACLE with
    the arm at joints `q` and the jaw at `jaw` — a static collision check of
    a planned pose. The env's qpos, FK and contact set are restored.

    `normal_mode` (contacts whose other body is in `normal_bodies`): 0 all
    count; 1 only the near-VERTICAL ones (a floor, `SUPPORT_UP_COS`); 2 all
    but those (the walls)."""
    var saved = List[Float64]()
    for i in range(NQ):
        saved.append(Float64(env.d.qpos.data[i]))
    for i in range(N_ARM):
        env.d.qpos.data[i] = q[i]
    env.d.qpos.data[N_ARM] = jaw
    env._fields_fk()
    detect_contacts_auto["cpu", DType.float64, BATCH=1](env.d, env.mf, None)
    var worst = 0.0
    var nc = Int(env.d.meta.data[META_IDX_NUM_CONTACTS])
    for c in range(nc):
        var o = c * CONTACT_SIZE
        var ba = Int(env.d.contacts.data[o + CONTACT_IDX_BODY_A])
        var bb = Int(env.d.contacts.data[o + CONTACT_IDX_BODY_B])
        var hit = (ba in arm_bodies and bb in obstacles) or (
            bb in arm_bodies and ba in obstacles
        )
        if hit and normal_mode != 0:
            var other = bb if ba in arm_bodies else ba
            if other in normal_bodies:
                var up = abs(Float64(env.d.contacts.data[o + CONTACT_IDX_NZ])) >= SUPPORT_UP_COS
                if (normal_mode == 1 and not up) or (normal_mode == 2 and up):
                    hit = False
        var d = Float64(env.d.contacts.data[o + CONTACT_IDX_DIST])
        if hit and -d > worst:
            worst = -d
    for i in range(NQ):
        env.d.qpos.data[i] = saved[i]
    env._fields_fk()
    detect_contacts_auto["cpu", DType.float64, BATCH=1](env.d, env.mf, None)
    return worst * 1000.0


# ── clean plans, and a place as a reversed grasp (from the rig executor) ──
#
# Ported from `examples/so101/tower_expert_real.mojo` (noeira-72, 2133be64e):
# its reference numbers are the check — `--plan-cycle` 64/68 both tasks,
# `--plan-bowl-pick` centre 66/68 (jaw 0.35, tilt 5..35, offset 12),
# `--plan-only` 64/68 on the printed set's layouts.


comptime IK_OK_MM: Float64 = 10.0
"""A plan whose IK misses a leg's target by more needs `force` to run."""
comptime ABOVE_M: Float64 = 0.06
"""A place's way in and out: this far above its pre-grasp tip point, so the
held brick clears the bowl's 45 mm rim."""


@fieldwise_init
struct PlanCfg(Copyable, Movable):
    """One grasp plan's settings: the tilt draw, the jaw the collision pass
    opens, the support surface and the fingers' height against it."""

    var tilt_lo_deg: Float64
    var tilt_hi_deg: Float64
    var jaw: Float64
    var support: String
    """`desk_mat` (the desk, every contact) or `bowl_bowl` (its FLOOR, by the
    contact normal: the walls stay in the veto — `set_support`)."""
    var clear_m: Float64
    """`desk_clear_m`: the fingers' lowest point against the support,
    negative = pressed into it."""


struct Planned(Movable):
    var plan: TowerGraspPlan
    var ok: Bool
    var draws: Int
    var seed: Int

    def __init__(out self, var plan: TowerGraspPlan, ok: Bool, draws: Int, seed: Int):
        self.plan = plan^
        self.ok = ok
        self.draws = draws
        self.seed = seed


def plan_is_clean(plan: TowerGraspPlan) -> Bool:
    return (
        plan.pen_mm <= PLAN_PEN_OK_MM and max(plan.e_grasp, plan.e_lift) * 1000.0 <= IK_OK_MM
        and plan.close_on_tip
    )


def plan_clean(
    mut env: E, mut planner: TowerGraspPlanner, ref body_names: List[String],
    ref qs: List[Float64], ref pb: List[Float64], yaw: Float64,
    ref q_seed: List[Float64], cfg: PlanCfg, seed: Int, draws: Int,
) raises -> Planned:
    """A grasp plan for a brick at `pb` in the scene `qs`, redrawn (up to
    `draws` postures, seeds `seed`, `seed + 1`, ...) until clean: the planner
    redraws a COLLIDING posture itself, not one the IK cannot reach. The
    planner's settings are restored."""
    var v0 = List[Float64](length=NV, fill=0.0)
    var saved_clear = planner.desk_clear_m
    var saved_lo = planner.posture.tilt_lo
    var saved_hi = planner.posture.tilt_hi
    planner.desk_clear_m = cfg.clear_m
    planner.posture.tilt_lo = cfg.tilt_lo_deg * pi / 180.0
    planner.posture.tilt_hi = cfg.tilt_hi_deg * pi / 180.0
    var support: List[String] = [cfg.support]
    planner.set_support(body_names, support, cfg.support != "desk_mat")
    var plan = TowerGraspPlan()
    var ok = False
    var n = 0
    for d in range(draws):
        env.set_state(qs, v0)
        seed_rng(seed + d)
        plan = planner.plan_grasp(env, pb, yaw, q_seed, cfg.jaw)
        n = d + 1
        ok = plan_is_clean(plan)
        if ok:
            break
    var desk: List[String] = ["desk_mat"]
    planner.set_support(body_names, desk, False)
    planner.desk_clear_m = saved_clear
    planner.posture.tilt_lo = saved_lo
    planner.posture.tilt_hi = saved_hi
    env.set_state(qs, v0)
    return Planned(plan^, ok, n, seed + n - 1)


def with_brick(
    ref qs: List[Float64], brick_adr: Int, x: Float64, y: Float64, z: Float64, yaw: Float64,
) -> List[Float64]:
    var q = qs.copy()
    q[brick_adr] = x
    q[brick_adr + 1] = y
    q[brick_adr + 2] = z
    q[brick_adr + 3] = cos(yaw / 2.0)
    q[brick_adr + 4] = 0.0
    q[brick_adr + 5] = 0.0
    q[brick_adr + 6] = sin(yaw / 2.0)
    return q^


def above_pose(
    mut env: E, mut planner: TowerGraspPlanner, plan: TowerGraspPlan, dz: Float64,
) -> Tuple[List[Float64], Float64]:
    """The joints that put the plan's pre-grasp tip point `dz` HIGHER, with
    the plan's tilt and pinch (IK seeded from its pre-grasp): the way in and
    out of the bowl passes over its rim, not through it. Returns (joints, IK
    error in m)."""
    var fx = sin(plan.tilt) * cos(plan.bearing)
    var fy = sin(plan.tilt) * sin(plan.bearing)
    var fz = -cos(plan.tilt)
    var t = List[Float64]()
    t.append(plan.tip_goal[0] - APPROACH_D * fx)
    t.append(plan.tip_goal[1] - APPROACH_D * fy)
    # the IK's tip mode aims `target - (0, 0, TIP_REACH)` (`_plan_tilted`)
    t.append(plan.tip_goal[2] - APPROACH_D * fz + dz + TIP_REACH)
    var q = List[Float64](length=N_ARM, fill=0.0)
    var e = planner.arm.ik(env, t, plan.q_pre, plan.yaw, q, plan.tilt, ROLL_SEED_HUMAN, True)
    return (q^, e)


def pick_place_legs(
    mut env: E, mut planner: TowerGraspPlanner,
    pick: TowerGraspPlan, place: TowerGraspPlan, jaw_pick: Float64, jaw_place: Float64,
    pick_in_bowl: Bool, mut jaws: List[Float64],
) -> List[PlanLeg]:
    """A pick and a PLACE made of a second grasp plan run backwards: the
    pick's pre, descent and close; out of a bowl, back UP the approach
    before the lift (a joint-space lift from its floor can sweep into the
    wall); the lift; carried to `ABOVE_M` over the place plan's pre-grasp,
    down to it, LOWERED along its approach (jaw closed), opened, retreated
    back up the approach and to the pose above. `jaws`: each leg's open-jaw
    target."""
    var out = List[PlanLeg]()
    jaws.clear()
    var none = List[Float64]()
    for leg in pick.legs(place=False, close_steps=1):
        if leg.name == "hold":
            continue
        if leg.name == "lift" and pick_in_bowl:
            var n_pw = len(pick.waypoints) - 1
            for j in range(n_pw - 1, -1, -1):
                out.append(PlanLeg("out", pick.waypoints[j].copy(), False, N_DESCEND // n_pw, False, False))
                jaws.append(jaw_pick)
        out.append(leg.copy())
        jaws.append(jaw_pick)
    var ab = above_pose(env, planner, place, ABOVE_M)
    var above_ok = ab[1] * 1000.0 <= IK_OK_MM
    if above_ok:
        out.append(PlanLeg("carry", ab[0].copy(), False, N_CARRY, False, False))
        jaws.append(jaw_place)
        out.append(PlanLeg("down", place.q_pre.copy(), False, N_DESCEND // 2, False, False))
        jaws.append(jaw_place)
    else:
        out.append(PlanLeg("carry", place.q_pre.copy(), False, N_CARRY, False, False))
        jaws.append(jaw_place)
    var n_wp = len(place.waypoints) - 1
    for j in range(1, n_wp + 1):
        out.append(PlanLeg("lower", place.waypoints[j].copy(), False, N_DESCEND // n_wp, False, False))
        jaws.append(jaw_place)
    out.append(PlanLeg("open", none.copy(), True, N_OPEN, False, False))
    jaws.append(jaw_place)
    for j in range(n_wp - 1, -1, -1):
        out.append(PlanLeg("retreat", place.waypoints[j].copy(), True, N_DESCEND // n_wp, False, False))
        jaws.append(jaw_place)
    if above_ok:
        out.append(PlanLeg("up", ab[0].copy(), True, N_DESCEND // 2, False, False))
        jaws.append(jaw_place)
    return out^


# ── the LOW release in the bowl (the rig's task 1, shared with the sim) ───


comptime RELEASE_GAP_M: Float64 = 0.005
"""A place plan's fingers this far ABOVE the support at the release (the
pick pressed them into it, so the brick's bottom is ~8 mm up)."""
comptime BOWL_RELEASE_JAW: Float64 = 0.25
"""The release INSIDE the bowl opens only this far: at 0.35 the rig's bowl
was pushed 9-14 mm in every task 1 of the first cycle run (the moving jaw
swinging out toward the wall as it opens); the 25 mm brick is free from
~0.14 (noeira-72, 50023f549)."""
comptime PLACE_DRAWS = 8
"""Postures drawn for a clean place plan (`plan_clean`)."""
comptime BRICK_QADR = 13
comptime BOWL_QADR = 6
"""The free joints' qpos addresses in the tower scene (`placement/so101_tower`
`free_qadr`: bowl, then brick)."""


def bowl_release_cfg() -> PlanCfg:
    """The low release in the bowl: tilt 5..35, the jaw opened to
    `BOWL_RELEASE_JAW`, the bowl's FLOOR as the support (by the contact normal;
    the walls stay in the veto), the fingers `RELEASE_GAP_M` above it — the
    rig executor's `cfg_bowl_release`."""
    return PlanCfg(5.0, 35.0, BOWL_RELEASE_JAW, String("bowl_bowl"), RELEASE_GAP_M)


def brick_rest_in_bowl_z(
    mut env: E, ref q_scene: List[Float64], ref lo: List[Float64],
    ref hi: List[Float64],
) raises -> Float64:
    """The brick's resting height IN the bowl (world m): dropped 2 cm above
    the floor at the bowl's centre in the sim scene and settled, the arm held
    where `q_scene` has it. ⚠ Steps `env`: the caller restores its state."""
    var qs = q_scene.copy()
    qs[BRICK_QADR] = qs[BOWL_QADR]
    qs[BRICK_QADR + 1] = qs[BOWL_QADR + 1]
    qs[BRICK_QADR + 2] = qs[BOWL_QADR + 2] + 0.045
    var v0 = List[Float64](length=NV, fill=0.0)
    env.set_state(qs, v0)
    var hold = ContAction[ACT]()
    for k in range(ACT):
        hold.data[k] = 2.0 * (qs[k] - lo[k]) / (hi[k] - lo[k]) - 1.0
    for _ in range(60):
        _ = env.step(hold)
    return Float64(env.d.qpos.data[BRICK_QADR + 2])

