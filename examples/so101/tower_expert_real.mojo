"""The scripted cube-in-bowl expert on the REAL arm: the overhead camera reads
the brick and the bowl, the sim planner plans the grasp, the servos run it.

    pixi run -e jetson mojo run -I . examples/so101/tower_expert_real.mojo \\
        --camera /dev/soarm_cam_overhead                # DRY RUN: reads, plans, prints
    pixi run -e jetson mojo run -I . examples/so101/tower_expert_real.mojo \\
        --camera /dev/soarm_cam_overhead --arm --episodes 10
    pixi run mojo run -I . examples/so101/tower_expert_real.mojo \\
        --plan-only projects/so101-tower/poses/cube-in-bowl-printed_start_poses.csv

## WHAT IT MEASURES

The expert reads nothing but the arm's joints and the two object poses. If it
cannot put the cube in the bowl on the rig with good poses, the gap is the
physics and the servos, and no perception or learning fixes it; if it can, a
sim-derived controller works on the real arm. Vision is out of the loop
except for the two poses read before the episode and the outcome read after.

## ⚠⚠ THIS MOVES A REAL ARM (with `--arm`)

Without `--arm` nothing is energised and no goal is written: the camera, the
estimates, the joint map, the planner and the plan all run, and the plan is
printed. With `--arm`:
- the goal is parked on the present pose before torque goes on;
- every goal is clamped to the calibrated range and to present ±80 ticks
  until the arm has caught up, ±512 after (`SO101Arm.write_goals`, the
  recorders' two-phase clamp);
- Enter during an episode ABORTS it: the arm ramps back to where the episode
  started;
- the run ends through `deploy_shutdown.return_and_release` from a `finally`
  (back to the pose the run started from, release on the operator's word).
  ⚠ A `finally` does not run on a signal: after Ctrl-C the follower holds its
  pose — `pixi run soarm-torque-off`.

## THE PLAN IS THE SIM'S (`tasks/so101_tower_expert_plan.mojo`)

`TowerGraspPlanner` in the human posture with `clear_plan`, as
`tower_expert_record.mojo --posture human --clear-plan` runs it: the tilt
drawn from 20..65 deg, the pinch snapped to the brick's faces, the grasp height
set against the desk and every pose checked against the base, the stand, the
bowl and the brick. The sim env it plans on is put in the RIG's state first:
the arm's six joints as measured (through the follower zero,
`SimJointMap.tower_follower`) and the brick and bowl at the estimates (their
heights are the sim's own resting heights). The place is planned once, from
the start estimate of the bowl.

⚠ `--desk-clear-mm` defaults to -3 (the fingers planned 3 mm INTO the desk),
not the sim's -8: in sim the press is what lets the moving jaw squeeze the
brick against the fixed finger; on the rig the servo's force limit decides
how hard it presses. Start here and measure.

## THE EXECUTION: the sim executor's ramps, with the servos' own waits

Per leg the joint command ramps linearly from the last command to the target
over the leg's step budget at 30 Hz (the recordings' rate), as the sim
executor's default demonstrator does. The real servos lag their goals where
the sim's stiff actuators did not, so three things differ, all printed:
1. after each moving leg the arm is given up to `--leg-settle` steps to settle
   (every joint slower than `--settled-vel`, the servo's Present_Velocity);
2. the close is armed on the last descent leg: it fires once the FK
   fingertip is within `--tip-close-mm` of the plan's `tip_goal` and the arm
   has settled, and past the ramp the arm is given `--settle-steps` more steps
   to get there — then it closes anyway, and the report says which;
3. the close holds the arm and waits for the jaw to stop (it stalls on the
   brick), at least `CLOSE_MIN` steps: the sim's close is one step because
   its jaw is stiff.
The sim executor's sag integral is OFF unless `--sag-ki` is given (measure
the servo's own sag first, from the trace). No contact is read anywhere.

After the retreat the arm ramps back to where the episode started (the
recorder's `--return-rest`), out of the camera's way, and the outcome is read.

## THE OUTCOME: the camera's brick-in-bowl

Over ~1 s of frames with the arm home: the brick found (any coverage — the
bowl's wall hides part of it) within `--in-bowl-mm` of the bowl on at least
half the frames (60 mm horizontal, not the task's 45: see `IN_BOWL_MM`). Otherwise the bucket: MISSED (the
brick within 15 mm of where it started), MOVED (elsewhere on the desk), or
LOST (not seen).

## `--dataset NAME`: the episodes as a LeRobot dataset (with `--arm`)

The expert as a DATA ENGINE: every episode is recorded as `record_ui.mojo`
records a teleoperated one — `projects/<--project, so101-tower>/datasets/NAME`,
checkpointed after each episode (`--resume` continues one), both cameras
(`--camera` -> `observation.images.overhead`, `--wrist-camera` ->
`observation.images.wrist`, 640x480 at 30 Hz, the overhead camera pacing the
tick), `observation.state` the follower's measured joints and `action` the
commanded ones, in LeRobot units (degrees, the gripper 0..100, through the
calibration — no follower zero), and `cube-in-bowl-printed`'s task string
(`TASK_LANGUAGE`, `--task`). An episode runs from the first leg to the arm folded
and still at home (`N_REST_HOLD`). The camera's verdict keeps a SUCCESS and
rejects anything else (`meta/rejected_episodes.json`, skipped on import);
the operator can flip it at the prompt. ⚠ The gripper's action when closed
is 0 (the expert commands fully closed; an operator's leader rarely is).

    pixi run -e jetson mojo run -I . examples/so101/tower_expert_real.mojo --arm \\
        --camera /dev/soarm_cam_overhead --wrist-camera /dev/soarm_cam_wrist \\
        --dataset cube-in-bowl-expert --episodes 50

## `--cycle [--auto]`: task 1 and task 2 in a loop, the scene resetting itself

TASK 1 puts the brick in the bowl; TASK 2 takes it out and puts it at a
drawn spot on the desk — which is task 1's next layout. The camera decides
which task a scene calls for (the brick found in the bowl: task 2).
- Task 1 RELEASES LOW at the bowl's centre instead of dropping from 10 cm:
  the place is a second grasp plan (a virtual brick there, the fingers 5 mm
  above the floor) run backwards — carried to 6 cm over its approach, down
  it, opened, back up (the jaw opening only to 0.25 inside the bowl: at
  0.35 the bowl was pushed 9-14 mm per release).
- Task 2 takes the brick at the camera's in-bowl read minus its measured
  bias (IN_BOWL_CAM_BIAS_*: the wall hides the brick's lower part), kept
  within 30 mm of the centre; the release point cross-checks it and gives
  the yaw when they agree. A close that ends below 0 rad (empty, or on the
  bowl's wall) opens, backs out along the approach, and goes home.
- Just before the arm moves the scene is re-checked: a bowl or brick moved
  more than 15 mm since the plan means re-read and re-plan.
- Task 2 picks with the bowl's FLOOR as the support (`set_support`, the walls
  stay in the collision veto), the jaw at `BOWL_JAW` (0.35 rad: 0.6 swings
  the moving jaw into the floor and walls), the tilt 5..35 deg, the aim 12
  mm along the pinch axis (`--pinch-offset-mm`, the brick BETWEEN the jaws),
  and leaves the bowl back up its approach before the lift.
- Planning, `--plan-cycle` on the printed set's 70 layouts (the sim's rest
  pose): both tasks clean on 64 of 68. The bowl pick alone
  (`--plan-bowl-pick`): the brick at the centre 66/68, 15 mm off it 44 %.
- `--auto`: no prompts — a clean plan runs, the camera's verdict keeps or
  rejects a recorded episode; Enter during an episode aborts it, `q` + Enter
  between episodes stops. A scene with no clean plan waits for the operator.
- Recorded, task 2's instruction is `--task2` ("Take the blue cube out of
  the yellow octogonal bowl and put it on the desk").
⚠ Not yet run on the arm.

## `--plan-only CSV`: the planning half alone

No camera and no arm: every layout of a start-poses file
(`tower_pose_real_check.mojo` writes one per recorded dataset) is planned from
the sim's rest pose, and the plans that would need `force` are counted.

## Output (`--out`, default `projects/so101-tower/rig_runs/<stamp>/`)

`episodes.tsv` (one row per episode: estimates, plan, close, outcome),
`epN.tsv` (every control tick: leg, commanded and measured joints in model
rad, joint velocities, Present_Load in % of max torque, the FK fingertip and
its distance to `tip_goal`), and
`epN_start.png` / `epN_end.png` (the overhead frames the poses and the
outcome were read from).
"""

from std.builtin.sort import sort
from std.math import atan2, cos, floor, sin, sqrt, pi
from std.os import makedirs
from std.random import seed as seed_rng, random_float64
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.core.concurrent.thread import sleep_us
from noeira.core.cont_action import ContAction
from noeira.core.project import project_dataset_dir
from noeira.core.run import epoch_seconds, iso8601_utc
from noeira.data.lerobot_rejected import reject_episode
from noeira.data.lerobot_write import LeRobotWriter, open_recording
from noeira.io.fileio import StdinReader, stdin_is_tty
from noeira.io.png import save_png
from noeira.physics3d.collision.broadphase_sap import detect_contacts_auto
from noeira.physics3d.gpu.constants import (
    MODEL_CURRICULUM_SIZE, CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST, CONTACT_IDX_POS_X, CONTACT_IDX_POS_Y, CONTACT_IDX_POS_Z,
    META_IDX_NUM_CONTACTS,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.robot.so101 import SO101Arm, SO101_N, joint_name
from noeira.robot.so101.deploy_shutdown import return_and_release
from noeira.robot.so101.ports import follower_port, port_refusal
from noeira.robot.so101.sim_map import SimJointMap
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig, So101TowerTeleopConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.so101_tower_expert_plan import (
    TowerExpertEnv, TowerGraspPlanner, TowerGraspPlan, PlanLeg, fingertip_point,
    ACT, N_ARM, NQ, HUMAN_JAW_OPEN, HUMAN_Z_GRASP, CLEAR_PLAN_TILT,
    PLAN_PEN_OK_MM, IK_OK_MM, PlanCfg, Planned, plan_clean, with_brick, pick_place_legs,
    N_OPEN, N_DESCEND, APPROACH_D, TIP_REACH, Z_LIFT, _pose_penetration_mm,
)
from noeira.tasks.so101_tower_overhead import (
    OVERHEAD_CALIB, tower_overhead_pose, tower_desk_roi, printed_brick_hsv,
    printed_bowl_hsv, pose_confident, TowerArmFK,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.utils.fmt import fixed
from noeira.vision.calib_file import read_calib
from noeira.vision.camera_thread import CameraReader
from noeira.vision.fisheye import FisheyeLens
from noeira.vision.opencv import opencv_shim_available
from noeira.vision.tabletop_pose import (
    RigCamera, PrismModel, PoseEstimate, DeskROI, estimate_prism_pose, model_silhouette,
)
from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic

comptime E = TowerExpertEnv
comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]
comptime CFG = So101TowerTeleopConfig
comptime NV = So101TowerModel.NV
comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime TASK = "so101_tower_cube_in_bowl"
comptime OUT_ROOT = "projects/so101-tower/rig_runs"

comptime HZ = 30
comptime PERIOD_NS = 33_333_333
"""30 Hz, the recorded datasets' rate (`record_ui.mojo`); recording, the
overhead camera IS the clock. The plans' step budgets are at the sim's 31.25
Hz, so each leg runs 4 % slower than in sim (the first rig run, 10/10, was
at 31.25 before recording existed)."""
comptime N_REST_HOLD = 15
"""Recording: frames held at rest after the ramp home (the sim recorder's
`--return-rest` hold) — the episode ends folded and still."""
comptime TASK_LANGUAGE = "Grab the blue cube and put it in the yellow octogonal bowl"
"""`cube-in-bowl-printed`'s task string, byte for byte, misspelling included
(its `meta/tasks.parquet`; SmolVLA's token table for it, f61c6e2b1, is 15
tokens): episodes merged with that set, or fine-tuned next to it, must carry
the same instruction. NOT the sim task's `language=` ("Grab the cube and put
it in the bowl"), which is the older `cube-in-bowl` set's. `--task` overrides."""
comptime DEFAULT_PROJECT = "so101-tower"
comptime MAX_STEP_TICKS = 80
comptime TRACK_STEP_TICKS = 512
"""The two-phase clamp on the goal's lead over the present position
(`SO101Arm.track_step_ticks`): 80 ticks while the follower catches up after
torque on (`--step`), 512 once it tracks — the recorders' and the SmolVLA
deployment's, so the arm moves like the arm that recorded the demos. ⚠ The
clamp also bounds the GRIPPER's squeeze: stalled on the brick, its goal sits
at most this far past its position, and a single small clamp (60 ticks, ~5
deg) would have squeezed with a twentieth of the teleop's force."""
comptime TIP_CLOSE_MM: Float64 = 8.0
"""The sim executor's `TIP_CLOSE_MM_DEFAULT` (`--tip-close-mm`)."""
comptime SETTLED_VEL: Float64 = 0.15
"""Rad/s, every joint incl. the jaw (the sim executor's; `--settled-vel`).
The servo's Present_Velocity is coarse: retune from the trace."""
comptime SETTLE_STEPS = 45
"""Past the last descent ramp, steps to wait for the close trigger before
closing anyway (`--settle-steps`; 1.4 s)."""
comptime LEG_SETTLE = 20
"""Past every other moving leg's ramp, steps to wait for the arm to settle
(`--leg-settle`; 0.64 s)."""
comptime CLOSE_MIN = 10
comptime CLOSE_MAX = 40
"""The close: at least / at most this many steps, done once the jaw has been
slower than `settled_vel` for 3 steps."""
comptime N_HOME = 60
"""The ramp back to the episode's start pose (the recorder's
`N_RETURN_REST`)."""
comptime SAG_MAX: Float64 = 0.35
"""The sag integral's clamp (rad), the sim executor's."""
comptime DESK_CLEAR_MM: Float64 = -3.0
comptime PLACE_OK_MM: Float64 = 30.0
"""Task 2: the brick within this of the target spot (the camera reads a
brick on the desk to ~5 mm; the release drops it ~8 mm)."""
comptime RELEASE_GAP_M: Float64 = 0.005
"""A place plan's fingers this far ABOVE the support at the release (the
pick pressed them into it, so the brick's bottom is ~8 mm up)."""
comptime SPOT_MIN_BOWL_M: Float64 = 0.12
"""Task 2's target spot at least this far from the bowl's centre."""
comptime TASK2_LANGUAGE = "Take the blue cube out of the yellow octogonal bowl and put it on the desk"
"""Task 2's instruction (`--task2`), worded like the printed set's."""
comptime BOWL_JAW: Float64 = 0.30
"""The jaw (model rad, the gripper's measured line since dbd873e15) for
the task-2 pick INSIDE the bowl: 0.6 swings the moving jaw into the floor
and the walls (planning, 340 picks: 27 clean at 0.6 against 110 at 0.35
with the 12 mm aim offset); 0.30 with the bowl's lift tolerance
(`plan_bowl_pick`): 318/340 (off the centre 92 %, against 55 % at 0.35) —
~39 mm between the pads around the 25 mm brick, enough now that the wrist
camera corrects the aim."""
comptime BOWL_RELEASE_JAW: Float64 = 0.25
"""Task 1's release INSIDE the bowl opens only this far (`--bowl-release-jaw`):
at 0.35 the bowl was pushed 9-14 mm in every task 1 of the first cycle run
(the moving jaw swinging out toward the wall as it opens); the 25 mm brick
is free from ~0.14."""
comptime JAW_EMPTY_RAD: Float64 = 0.0
"""After the close, a jaw below this closed on nothing (or on a corner): the
pick is abandoned. Measured (the gripper's measured line): holding the brick
0.116-0.121 rad, empty -0.149, a corner grip -0.095 (dropped on the carry)."""
comptime LOOK_TICKS = 6
"""The wrist look's hold at the pre-grasp (0.2 s at 30 Hz)."""
comptime REAIM_TEST_MM: Float64 = 10.0
"""`--plan-cycle`'s offline check of the fast re-aim: the brick moved this
far (a direction drawn per layout) and turned 15 deg."""
comptime REAIM_IK_ITERS = 100
comptime REAIM_IK_OK_MM: Float64 = 3.0
"""The fast re-aim's IK must reach every approach waypoint to this."""
comptime REAIM_MIN_STEPS = 6
comptime REAIM_DEG_PER_STEP: Float64 = 1.0
"""The move to the re-aimed pre-grasp: this many degrees per step at most
(the descent runs ~0.8), at least REAIM_MIN_STEPS."""
comptime N_PRE_FULL = 40
comptime LOOK_COV_MIN: Float64 = 0.7
comptime LOOK_COV_MAX: Float64 = 1.7
comptime LOOK_RES_MAX: Float64 = 0.5
"""`look_confident`'s bounds (the overhead's: coverage 0.75..1.3, residual
< 0.35)."""
comptime LOOK_MIN_CONF = 3
comptime LOOK_MAX_CORR_MM: Float64 = 25.0
"""The wrist correction acts (both tasks) when at least LOOK_MIN_CONF of the
look's frames read the brick confidently and it moves the aim by at most
this. Cycle run 3: the two in-bowl misses were the wrist's two largest
offsets from the aim (12.7, 13.6 mm, the yaw 20-44 deg off), the two
successes 7.7 and 3.7; with the correction (run 4) task 2 went 4/4. Run 4's
two task-1 failures (flung, empty) had the wrist 13 mm off the aim: every
desk pick read +8..+20 mm in x (the arm's sag, which the overhead-planned
aim carries and the wrist read relative to the gripper does not)."""
comptime LOOK_ROI_M: Float64 = 0.08
"""The wrist look searches within this of the planned brick: the camera
also sees the blue tower stand."""
comptime WRIST_CALIB = "projects/so101-tower/cameras/camera_wrist.txt"
comptime SCENE_MOVED_MM: Float64 = 15.0
"""Before the arm moves, a bowl or a brick this far from where the plan was
made means the scene changed: re-read and re-plan. Above the in-bowl read's
jitter (~10 mm between reads of a still brick)."""
comptime IN_BOWL_CAM_BIAS_X: Float64 = -0.0059
comptime IN_BOWL_CAM_BIAS_Y: Float64 = -0.0028
"""The camera's read of a brick IN the bowl minus where it really is (m):
the mean of (read - release point) over six undisturbed releases of the
first two cycle runs, spread ~3 mm (the wall hides the brick's lower part,
the fit shifts)."""
comptime DR_YAW_AGREE_M: Float64 = 0.015
"""Task 2 takes the last release's yaw when the corrected camera read is
within this of it (the brick undisturbed); else the camera's."""
comptime BOWL_PICK_MAX_OFF_M: Float64 = 0.03
"""Task 2's pick point is kept within this of the bowl's centre (the inner
floor leaves the brick's centre ~37 mm at most; nearer the wall the pick
plans clean less often)."""
comptime SPOT_TRIES = 12
comptime BOWL_LIFT_OK_MM: Float64 = 60.0
"""A bowl pick's lift pose may miss by this (see `plan_bowl_pick`)."""
comptime PICK_DRAWS = 8
"""Postures drawn per plan (each with the planner's own collision redraws)
before giving up: the planner does not redraw a posture the IK misses."""
comptime IN_BOWL_MM: Float64 = 60.0
"""Brick centre to bowl centre, horizontal, for SUCCESS. Not the task's 45 mm
(3D, body origins): in the bowl the camera sees only the brick's upper half
over the wall, and the fit reads it 29-48 mm from the bowl's centre (first rig
run, 2026-09-25, 10 bricks in the bowl, single frames) — at 45 two of ten
real successes were scored MOVED. A brick BESIDE the bowl is at least 68 mm
away (the octagon's apothem 56 + half the brick); 60 is between."""
comptime MISSED_MM: Float64 = 15.0
comptime STILL_MM: Float64 = 1.5
comptime SCENE_TIMEOUT_S: Float64 = 120.0
comptime MIN_SEP_M: Float64 = 0.088
"""The brick's centre at least this far from the bowl's (the sim check's
redraw rule): nearer, it is in or against the bowl."""
comptime DROP_ABORT = 8
"""Consecutive ticks with a partial bus read before the episode aborts."""


def _draw_outline(
    mut img: List[UInt8], cam: RigCamera, model: PrismModel, plane_z: Float64,
    x: Float64, y: Float64, yaw: Float64, r: UInt8, g: UInt8, b: UInt8,
):
    """The prism's silhouette, drawn into an RGB frame."""
    var poly = model_silhouette(cam, model, plane_z, x, y, yaw)
    for i in range(len(poly)):
        var a = poly[i]
        var c = poly[(i + 1) % len(poly)]
        var n = Int(max(abs(c[0] - a[0]), abs(c[1] - a[1]))) + 1
        for k in range(n + 1):
            var t = Float64(k) / Float64(n)
            var u = Int(a[0] + t * (c[0] - a[0]) + 0.5)
            var v = Int(a[1] + t * (c[1] - a[1]) + 0.5)
            if u >= 0 and v >= 0 and u < cam.width and v < cam.height:
                var o = (v * cam.width + u) * 3
                img[o] = r
                img[o + 1] = g
                img[o + 2] = b


def _yaw90_deg(a: Float64, b: Float64) -> Float64:
    """|a - b| on the cube's 90-degree circle, degrees."""
    var d = (a - b) * 180.0 / pi
    while d > 45.0:
        d -= 90.0
    while d < -45.0:
        d += 90.0
    return abs(d)


def _deg(r: Float64) -> String:
    return fixed(r * 180.0 / pi, 1)


def _circ_mean(ys: List[Float64], period: Float64) -> Float64:
    """The mean of angles defined modulo `period`, in [0, period)."""
    var sc = 0.0
    var ss = 0.0
    var w = 2.0 * pi / period
    for y in ys:
        sc += cos(w * y)
        ss += sin(w * y)
    var m = atan2(ss, sc) / w
    if m < 0.0:
        m += period
    return m


@fieldwise_init
struct Scene(Copyable, Movable):
    """The brick and the bowl, still, from the overhead camera (world m)."""

    var brick_x: Float64
    var brick_y: Float64
    var brick_yaw: Float64
    var bowl_x: Float64
    var bowl_y: Float64
    var bowl_yaw: Float64
    var in_bowl: Bool
    """The brick is IN the bowl: found (any coverage — the wall hides part of
    it) within `IN_BOWL_MM` of the bowl; its pose is then the camera's rough
    read, not a confident one."""


struct Rig(Movable):
    """The follower, its map to the model's joints, and the control tick."""

    var arm: SO101Arm
    var jmap: SimJointMap
    var armed: Bool
    var raw: List[Int32]
    """The last full position read, ticks."""
    var q_cmd: List[Float64]
    """The command, model rad, 6 (the jaw last)."""
    var q_ref: List[Float64]
    var sag_bias: List[Float64]
    var sag_ki: Float64
    var q: List[Float64]
    """Measured, model rad (unclamped), 6."""
    var v: List[Float64]
    """Measured, rad/s, 6."""
    var load: List[Float64]
    """Present_Load, % of the servo's maximum torque, signed, 6."""
    var load_peak: List[Float64]
    """This leg's largest |load| per joint."""
    var tip: List[Float64]
    var tip_goal: List[Float64]
    var jaw_open: Float64
    var jaw_lo: Float64
    var settled_vel: Float64
    var tip_close_mm: Float64
    var t_next: Int
    var tick_n: Int
    var drops: Int
    var drops_run: Int
    var late: Int
    var leg: String
    var trace: String
    var cams: List[CameraReader]
    """The overhead camera, then (recording) the wrist camera."""
    var writers: List[LeRobotWriter]
    """Recording: the dataset writer (one), else empty."""
    var frames: List[List[UInt8]]
    var recording_now: Bool
    """Inside a recorded episode: the cameras pace the tick and every tick is
    a dataset row."""
    var rec_rows: Int

    def __init__(out self, var arm: SO101Arm, var jmap: SimJointMap):
        self.arm = arm^
        self.jmap = jmap^
        self.armed = False
        self.raw = List[Int32](length=SO101_N, fill=Int32(0))
        self.q_cmd = List[Float64](length=ACT, fill=0.0)
        self.q_ref = List[Float64](length=N_ARM, fill=0.0)
        self.sag_bias = List[Float64](length=N_ARM, fill=0.0)
        self.sag_ki = 0.0
        self.q = List[Float64](length=ACT, fill=0.0)
        self.v = List[Float64](length=ACT, fill=0.0)
        self.load = List[Float64](length=ACT, fill=0.0)
        self.load_peak = List[Float64](length=ACT, fill=0.0)
        self.tip = List[Float64](length=3, fill=0.0)
        self.tip_goal = List[Float64](length=3, fill=0.0)
        self.jaw_open = HUMAN_JAW_OPEN
        self.jaw_lo = 0.0
        self.settled_vel = SETTLED_VEL
        self.tip_close_mm = TIP_CLOSE_MM
        self.t_next = 0
        self.tick_n = 0
        self.drops = 0
        self.drops_run = 0
        self.late = 0
        self.leg = String("")
        self.trace = String("")
        self.cams = List[CameraReader]()
        self.writers = List[LeRobotWriter]()
        self.frames = List[List[UInt8]]()
        self.recording_now = False
        self.rec_rows = 0

    def read_joints(mut self) raises -> Bool:
        """Positions only (torque off or on); False on a partial read."""
        var raw = List[Int32](length=SO101_N, fill=Int32(0))
        if self.arm.read_positions(Span(raw)) != SO101_N:
            return False
        for i in range(SO101_N):
            self.raw[i] = raw[i]
            self.q[i] = self.jmap.to_sim_unclamped(self.arm.cal, i, raw[i])
        return True

    def arm_torque(mut self, step_ticks: Int) raises:
        """Position mode, the goal parked on the present pose, torque on."""
        if not self.read_joints():
            raise Error("the follower did not report 6 positions — not arming")
        self.arm.set_position_mode()
        var park = self.raw.copy()
        self.arm.max_step_ticks = 0
        self.arm.write_goals(Span(park))
        self.arm.max_step_ticks = step_ticks
        self.arm.set_torque(True)
        self.armed = True

    def begin(mut self, mut env: E) raises:
        """Start a command stream from the measured pose."""
        var ok = False
        for _ in range(5):
            if self.sense(env):
                ok = True
                break
        if not ok:
            raise Error("the follower did not report its pose")
        for i in range(ACT):
            self.q_cmd[i] = self.q[i]
        for i in range(N_ARM):
            self.q_ref[i] = self.q[i]
            self.sag_bias[i] = 0.0
        self.t_next = Int(perf_counter_ns()) + PERIOD_NS
        self.drops_run = 0

    def sense(mut self, mut env: E) raises -> Bool:
        """Positions, velocities, and the FK fingertip (on the planner's env)."""
        var raw = List[Int32](length=SO101_N, fill=Int32(0))
        var vraw = List[Int32](length=SO101_N, fill=Int32(0))
        var lraw = List[Int32](length=SO101_N, fill=Int32(0))
        if self.arm.read_positions(Span(raw)) != SO101_N:
            return False
        if self.arm.read_velocities(Span(vraw)) != SO101_N:
            return False
        if self.arm.read_loads(Span(lraw)) != SO101_N:
            return False
        for i in range(SO101_N):
            self.raw[i] = raw[i]
            self.q[i] = self.jmap.to_sim_unclamped(self.arm.cal, i, raw[i])
            self.v[i] = Float64(vraw[i]) * 2.0 * pi / 4096.0
            self.load[i] = Float64(lraw[i]) / 10.0
            self.load_peak[i] = max(self.load_peak[i], abs(self.load[i]))
        for i in range(ACT):
            env.d.qpos.data[i] = self.q[i]
        env._fields_fk()
        var t = fingertip_point(env)
        for k in range(3):
            self.tip[k] = t[k]
        return True

    def tick(mut self, mut env: E, mut stdin: StdinReader) raises:
        """Wait for the period (recording: for the cameras' next frames),
        measure, write the command; one trace row and, recording, one dataset
        row (the frames, the state just measured, the command just written —
        `record.mojo`'s order)."""
        if stdin.has_input():
            raise Error("ABORT: stopped by the operator")
        if self.recording_now:
            for c in range(len(self.cams)):
                var got = self.cams[c].take_blocking(self.frames[c])
                if not got:
                    raise Error("a camera stopped delivering frames")
        else:
            var now = Int(perf_counter_ns())
            if now > self.t_next:
                self.late += 1
                self.t_next = now
            while Int(perf_counter_ns()) < self.t_next:
                pass
            self.t_next += PERIOD_NS
        self.tick_n += 1
        if self.sense(env):
            self.drops_run = 0
        else:
            self.drops += 1
            self.drops_run += 1
            if self.drops_run >= DROP_ABORT:
                raise Error("ABORT: " + String(DROP_ABORT) + " partial bus reads in a row")
        var goals = List[Int32](length=SO101_N, fill=Int32(0))
        for i in range(SO101_N):
            goals[i] = self.jmap.from_sim(self.arm.cal, i, self.q_cmd[i])
        self.arm.write_goals(Span(goals))
        if self.recording_now:
            # LeRobot units, as `record.mojo` writes them: the follower's
            # measured ticks and the commanded ticks, through the calibration
            # (degrees; the gripper 0..100) — no follower zero
            var state = List[Float64]()
            var action = List[Float64]()
            for i in range(SO101_N):
                state.append(self.arm.cal.degrees(i, self.raw[i]))
                action.append(self.arm.cal.degrees(i, goals[i]))
            self.writers[0].add_frame(state, action, self.frames)
            self.rec_rows += 1
        var row = String(self.tick_n) + "\t" + self.leg
        for i in range(ACT):
            row += "\t" + fixed(self.q_cmd[i], 4)
        for i in range(ACT):
            row += "\t" + fixed(self.q[i], 4)
        for i in range(ACT):
            row += "\t" + fixed(self.v[i], 3)
        for i in range(ACT):
            row += "\t" + fixed(self.load[i], 1)
        for k in range(3):
            row += "\t" + fixed(self.tip[k], 4)
        self.trace += row + "\t" + fixed(self.tip_dist_mm(), 1) + "\n"

    def sag_update(mut self):
        if self.sag_ki <= 0.0:
            return
        for i in range(N_ARM):
            var b = self.sag_bias[i] + self.sag_ki * (self.q_ref[i] - self.q[i])
            self.sag_bias[i] = max(-SAG_MAX, min(SAG_MAX, b))

    def set_arm(mut self, ref target: List[Float64]):
        for i in range(N_ARM):
            self.q_ref[i] = target[i]
            self.q_cmd[i] = target[i] + self.sag_bias[i]

    def settled(self) -> Bool:
        for i in range(ACT):
            if abs(self.v[i]) > self.settled_vel:
                return False
        return True

    def tip_dist_mm(self) -> Float64:
        var d = 0.0
        for k in range(3):
            d += (self.tip[k] - self.tip_goal[k]) ** 2
        return sqrt(d) * 1000.0

    def arm_err_deg(self, ref target: List[Float64]) -> Float64:
        var w = 0.0
        for i in range(N_ARM):
            w = max(w, abs(self.q[i] - target[i]))
        return w * 180.0 / pi

    def run_leg(
        mut self, mut env: E, leg: PlanLeg, settle_steps: Int, leg_settle: Int,
        mut stdin: StdinReader,
    ) raises -> String:
        """One plan leg; returns its report line."""
        self.leg = leg.name
        for i in range(ACT):
            self.load_peak[i] = 0.0
        var g_target = self.jaw_open if leg.grip_open else self.jaw_lo
        var target: List[Float64]
        if len(leg.q) > 0:
            target = leg.q.copy()
        else:
            # a HOLD keeps the arm's reference (the sim executor's `hold`)
            target = self.q_ref.copy()
        var q0 = self.q_cmd.copy()
        var r0 = self.q_ref.copy()
        if leg.name == "close":
            self.q_cmd[5] = g_target
            var slow = 0
            var n = 0
            for k in range(CLOSE_MAX):
                self.tick(env, stdin)
                self.sag_update()
                self.set_arm(target)
                n = k + 1
                slow = slow + 1 if abs(self.v[5]) < self.settled_vel else 0
                if n >= CLOSE_MIN and slow >= 3:
                    break
            return (
                "close   " + String(n) + " steps | jaw " + fixed(self.q[5], 3)
                + " rad (commanded " + fixed(g_target, 3) + ") | tip "
                + fixed(self.tip_dist_mm(), 1) + " mm from tip_goal" + self.load_report()
            )
        for k in range(leg.steps):
            var a = Float64(k + 1) / Float64(leg.steps)
            for i in range(N_ARM):
                self.q_ref[i] = r0[i] + (target[i] - r0[i]) * a
                self.q_cmd[i] = self.q_ref[i] + self.sag_bias[i]
            if leg.close_on_tip:
                self.q_cmd[5] = g_target
            else:
                self.q_cmd[5] = q0[5] + (g_target - q0[5]) * a
            self.tick(env, stdin)
            self.sag_update()
            if (
                leg.close_on_tip and k + 1 >= 3
                and self.tip_dist_mm() < self.tip_close_mm and self.settled()
            ):
                return (
                    pad8(leg.name) + String(k + 1) + " steps | CLOSE TRIGGERED on the ramp: tip "
                    + fixed(self.tip_dist_mm(), 1) + " mm | arm err " + fixed(self.arm_err_deg(target), 1) + " deg"
                    + self.load_report()
                )
        var extra = settle_steps if leg.close_on_tip else leg_settle
        var waited = 0
        var ok = False
        for _ in range(extra):
            if leg.close_on_tip:
                if self.tip_dist_mm() < self.tip_close_mm and self.settled():
                    ok = True
                    break
            elif self.settled():
                ok = True
                break
            self.set_arm(target)
            self.tick(env, stdin)
            self.sag_update()
            waited += 1
        var s = pad8(leg.name) + String(leg.steps) + "+" + String(waited) + " steps | "
        if leg.close_on_tip:
            s += (
                "CLOSE TRIGGERED after the ramp" if ok else "⚠ NO TRIGGER, closing anyway"
            ) + ": tip " + fixed(self.tip_dist_mm(), 1) + " mm"
        else:
            s += ("settled" if ok else "⚠ not settled")
        s += " | arm err " + fixed(self.arm_err_deg(target), 1) + " deg | jaw " + fixed(self.q[5], 3)
        return s + self.load_report()

    def load_report(self) -> String:
        """This leg's peak |load| per joint and the load now, % of max torque."""
        var s = String(" | load % peak/now")
        for i in range(ACT):
            s += " " + fixed(self.load_peak[i], 0) + "/" + fixed(self.load[i], 0)
        return s

    def ramp_to(
        mut self, mut env: E, ref target6: List[Float64], steps: Int, var name: String,
        mut stdin: StdinReader,
    ) raises:
        """All six joints to `target6` (the jaw included), sag bias dropped."""
        self.leg = name^
        var q0 = self.q_cmd.copy()
        for i in range(N_ARM):
            self.sag_bias[i] = 0.0
        for k in range(steps):
            var a = Float64(k + 1) / Float64(steps)
            for i in range(ACT):
                self.q_cmd[i] = q0[i] + (target6[i] - q0[i]) * a
            for i in range(N_ARM):
                self.q_ref[i] = self.q_cmd[i]
            self.tick(env, stdin)
        for _ in range(LEG_SETTLE):
            if self.settled():
                break
            self.tick(env, stdin)


def pad8(s: String) -> String:
    var out = s
    while out.byte_length() < 8:
        out += " "
    return out


def read_scene(
    mut reader: CameraReader, cam: RigCamera, mut frame: List[UInt8],
    mut stdin: StdinReader,
) raises -> Optional[Scene]:
    """The bowl confident and still for a second, and the brick either
    confident on the desk (still too) or IN the bowl (found near it, every
    frame of the second). None on 'q' or the timeout."""
    var roi = tower_desk_roi()
    var cb = printed_brick_hsv()
    var co = printed_bowl_hsv()
    var brick = PrismModel.tower_brick()
    var bowl = PrismModel.tower_bowl()
    var wt = List[Float64]()
    var bx = List[Float64]()
    var by = List[Float64]()
    var byaw = List[Float64]()
    var ox = List[Float64]()
    var oy = List[Float64]()
    var oyaw = List[Float64]()
    var t0 = perf_counter_ns()
    var last_print = -10.0
    var in_bowl_w = False
    while True:
        var now = Float64(perf_counter_ns() - t0) * 1e-9
        if now > SCENE_TIMEOUT_S:
            print("  no still, confident brick + bowl in", Int(SCENE_TIMEOUT_S), "s")
            return None
        if stdin.has_input():
            var l = stdin.line()
            if l == "q":
                return None
        if reader.take_latest(frame) == 0:
            _ = sleep_us(2000)
            continue
        var eb = estimate_prism_pose(frame, cam, cb, brick, roi)
        var eo = estimate_prism_pose(frame, cam, co, bowl, roi)
        var near = eb.found and pose_confident(eo) and sqrt(
            (eb.x - eo.x) ** 2 + (eb.y - eo.y) ** 2
        ) * 1000.0 < IN_BOWL_MM
        if len(wt) > 0 and near != in_bowl_w:
            wt.clear()
            bx.clear()
            by.clear()
            byaw.clear()
            ox.clear()
            oy.clear()
            oyaw.clear()
        in_bowl_w = near
        if not ((pose_confident(eb) or near) and pose_confident(eo)):
            if now - last_print > 3.0:
                last_print = now
                print(
                    "  waiting: brick", "ok" if pose_confident(eb) else (
                        "seen cov " + fixed(eb.coverage, 2) + " res " + fixed(eb.residual, 2)
                        if eb.found else "not seen"
                    ), "| bowl", "ok" if pose_confident(eo) else (
                        "seen cov " + fixed(eo.coverage, 2) + " res " + fixed(eo.residual, 2)
                        if eo.found else "not seen"
                    ),
                )
            wt.clear()
            bx.clear()
            by.clear()
            byaw.clear()
            ox.clear()
            oy.clear()
            oyaw.clear()
            continue
        wt.append(now)
        bx.append(eb.x)
        by.append(eb.y)
        byaw.append(eb.yaw)
        ox.append(eo.x)
        oy.append(eo.y)
        oyaw.append(eo.yaw)
        while len(wt) > 0 and wt[0] < now - 1.0:
            _ = wt.pop(0)
            _ = bx.pop(0)
            _ = by.pop(0)
            _ = byaw.pop(0)
            _ = ox.pop(0)
            _ = oy.pop(0)
            _ = oyaw.pop(0)
        var n = len(wt)
        if n < 10 or now - wt[0] < 0.8:
            continue
        var mbx = 0.0
        var mby = 0.0
        var mox = 0.0
        var moy = 0.0
        for k in range(n):
            mbx += bx[k] / Float64(n)
            mby += by[k] / Float64(n)
            mox += ox[k] / Float64(n)
            moy += oy[k] / Float64(n)
        var spread = 0.0
        for k in range(n):
            if not in_bowl_w:
                # in the bowl the half-hidden brick's fit jitters; only the
                # bowl has to be still
                spread = max(spread, sqrt((bx[k] - mbx) ** 2 + (by[k] - mby) ** 2))
            spread = max(spread, sqrt((ox[k] - mox) ** 2 + (oy[k] - moy) ** 2))
        if spread * 1000.0 > STILL_MM:
            continue
        return Scene(
            mbx, mby, _circ_mean(byaw, brick.period), mox, moy,
            _circ_mean(oyaw, bowl.period), in_bowl_w,
        )


struct WristLook(Movable):
    """The wrist camera's read of the brick at the pre-grasp pose."""

    var n_frames: Int
    var n_conf: Int
    var x: Float64
    var y: Float64
    var yaw: Float64
    var frame: List[UInt8]
    var cam_pos: Vec3d
    var cam_rot: Mat3d

    def __init__(out self):
        self.n_frames = 0
        self.n_conf = 0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.frame = List[UInt8]()
        self.cam_pos = Vec3d(0.0, 0.0, 1.0)
        self.cam_rot = Mat3d.identity()


def look_confident(e: PoseEstimate) -> Bool:
    """The wrist look's own confidence, looser than `pose_confident` (tuned
    on the overhead camera 45 cm away): 10-15 cm away the arm's sag puts the
    real camera LOWER than its FK, so the brick looks bigger than the model
    predicts — cycle run 6, ep 1: the brick plainly in view, found 3.6 mm
    from the plan's, but coverage 1.40 and residual 0.40 rejected every
    frame, and the uncorrected pick (at 38 cm reach) closed empty. The
    correction stays guarded by the median over the frames and its 25 mm
    cap."""
    return (
        e.found and e.coverage >= LOOK_COV_MIN and e.coverage <= LOOK_COV_MAX
        and e.residual < LOOK_RES_MAX
    )


def wrist_look(
    mut rig: Rig, mut env: E, mut afk: TowerArmFK, wci: Int, wlens: FisheyeLens,
    pick_x: Float64, pick_y: Float64, pick_z: Float64, mut stdin: StdinReader,
) raises -> WristLook:
    """LOG ONLY: hold the arm still `LOOK_TICKS` ticks at the pre-grasp and
    read the brick through the wrist camera — its pose from the arm's FK at
    the measured joints (`TowerArmFK.camera_pose`), the estimator searching
    within `LOOK_ROI_M` of where the plan puts the brick, on the plane the
    brick rests on (the desk, or the bowl's floor). The plan is not changed:
    this collects "arm still, ~10 cm away" data to decide whether a wrist
    correction would help (offline, on teleop data, it did not beat the
    overhead camera — but the arm never stops there)."""
    var out = WristLook()
    var plane = pick_z - 0.0125
    var roi = DeskROI(plane, pick_x - LOOK_ROI_M, pick_x + LOOK_ROI_M, pick_y - LOOK_ROI_M, pick_y + LOOK_ROI_M)
    var brick = PrismModel.tower_brick()
    var xs = List[Float64]()
    var ys = List[Float64]()
    var yaws = List[Float64]()
    var buf = List[UInt8](length=640 * 480 * 3, fill=UInt8(0))
    rig.leg = String("look")
    for _ in range(LOOK_TICKS):
        rig.tick(env, stdin)
        var got: Bool
        if rig.recording_now:
            buf = rig.frames[1].copy()
            got = True
        else:
            got = rig.cams[1].take_latest(buf) > 0
        if not got:
            continue
        afk.set_qpos(rig.q)
        var cp = afk.camera_pose(wci)
        var wcam = RigCamera(wlens, cp[0], cp[1])
        out.n_frames += 1
        var ew = estimate_prism_pose(buf, wcam, printed_brick_hsv(), brick, roi, max_coverage=3.0)
        out.frame = buf.copy()
        out.cam_pos = cp[0]
        out.cam_rot = cp[1]
        if look_confident(ew):
            xs.append(ew.x)
            ys.append(ew.y)
            yaws.append(ew.yaw)
    out.n_conf = len(xs)
    if out.n_conf > 0:
        sort(xs)
        sort(ys)
        out.x = xs[len(xs) // 2]
        out.y = ys[len(ys) // 2]
        out.yaw = _circ_mean(yaws, brick.period)
    return out^


def _ik_from(
    mut env: E, mut planner: TowerGraspPlanner, ref t: List[Float64], ref q0: List[Float64],
    yaw: Float64, tilt: Float64, mut q_out: List[Float64],
) -> Float64:
    """Damped least squares from `q0` over the planner's own relaxing tilt
    weights (`TowerArm.ik`'s), no restarts: the best position error."""
    var weights: List[Float64] = [0.3, 0.12, 0.05, 0.02]
    var best = 1e9
    var qt = List[Float64](length=N_ARM, fill=0.0)
    for w in weights:
        var e = planner.arm._ik_once(env, t, q0, yaw, w, 0.02, REAIM_IK_ITERS, qt, tilt, True)
        if e < best - 1e-5:
            best = e
            for i in range(N_ARM):
                q_out[i] = qt[i]
        if best < 1e-3:
            break
    return best


def _reaim_face(
    mut env: E, mut planner: TowerGraspPlanner, plan: TowerGraspPlan,
    ref qs_new: List[Float64], new_x: Float64, new_y: Float64, pick_z: Float64,
    yaw_n: Float64, jaw: Float64,
) raises -> Tuple[TowerGraspPlan, Bool, String, Float64]:
    """The re-aim for ONE pinch face (`yaw_n`): (plan, ok, why, penetration)."""
    var p = plan.copy()
    var tip = List[Float64]()
    tip.append(new_x + planner.pinch_offset_m * cos(yaw_n))
    tip.append(new_y + planner.pinch_offset_m * sin(yaw_n))
    tip.append(plan.tip_goal[2])
    var bearing = atan2(new_y, new_x)
    var tl = plan.tilt
    var fx = sin(tl) * cos(bearing)
    var fy = sin(tl) * sin(bearing)
    var fzv = -cos(tl)
    var n_wp = len(plan.waypoints) - 1
    var worst = 0.0
    for j in range(n_wp + 1):
        var sback = APPROACH_D * Float64(n_wp - j) / Float64(n_wp)
        var t = List[Float64]()
        t.append(tip[0] - sback * fx)
        t.append(tip[1] - sback * fy)
        t.append(tip[2] - sback * fzv + TIP_REACH)
        var qj = List[Float64](length=N_ARM, fill=0.0)
        var e = _ik_from(env, planner, t, plan.waypoints[j], yaw_n, tl, qj)
        worst = max(worst, e)
        p.waypoints[j] = qj^
    var tlift: List[Float64] = [new_x, new_y, pick_z + Z_LIFT]
    var ql = List[Float64](length=N_ARM, fill=0.0)
    var el = _ik_from(env, planner, tlift, plan.q_lift, yaw_n, tl, ql)
    for i in range(N_ARM):
        p.q_pre[i] = p.waypoints[0][i]
        p.q_grasp[i] = p.waypoints[n_wp][i]
        p.q_lift[i] = ql[i]
    for k in range(3):
        p.tip_goal[k] = tip[k]
    p.yaw = yaw_n
    p.bearing = bearing
    # the obstacles, with the brick where the wrist saw it, the jaw open
    var v0 = List[Float64](length=NV, fill=0.0)
    env.set_state(qs_new, v0)
    var pen = 0.0
    for j in range(n_wp + 1):
        pen = max(pen, _pose_penetration_mm(env, p.waypoints[j], jaw, planner.arm_bodies, planner.obstacles))
    env.set_state(qs_new, v0)
    var ok = worst * 1000.0 <= REAIM_IK_OK_MM and el * 1000.0 <= BOWL_LIFT_OK_MM and pen <= PLAN_PEN_OK_MM
    var why = (
        "IK " + fixed(worst * 1000.0, 1) + " mm (lift " + fixed(el * 1000.0, 1) + ") | penetration "
        + fixed(pen, 1) + " mm"
    )
    return (p^, ok, why, pen)


def reaim_pick(
    mut env: E, mut planner: TowerGraspPlanner, plan: TowerGraspPlan,
    ref qs_new: List[Float64], new_x: Float64, new_y: Float64, pick_z: Float64,
    brick_yaw: Float64, jaw: Float64,
) raises -> Tuple[TowerGraspPlan, Bool, String]:
    """THE FAST WRIST CORRECTION: the pick KEEPS its posture (the tilt, the
    grasp height) and the approach line moves to the brick the wrist camera
    saw; each waypoint is re-solved from the old one over the planner's tilt
    weights (ms, where a re-plan draws postures for seconds) and checked
    against the obstacles with the jaw open. The pinch goes on the brick
    face nearest the old pinch, or the next one if that one collides (a
    wall, in the bowl). Returns (plan, ok, why)."""
    var k = Float64(Int(floor((plan.yaw - brick_yaw) / (pi / 2.0) + 0.5)))
    var y1 = brick_yaw + k * (pi / 2.0)
    # the second-nearest face: the other side of the old pinch
    var y2 = y1 + (pi / 2.0 if plan.yaw > y1 else -pi / 2.0)
    var r1 = _reaim_face(env, planner, plan, qs_new, new_x, new_y, pick_z, y1, jaw)
    if r1[1]:
        return (r1[0].copy(), True, r1[2])
    var r2 = _reaim_face(env, planner, plan, qs_new, new_x, new_y, pick_z, y2, jaw)
    if r2[1]:
        return (r2[0].copy(), True, r2[2] + " (the next face)")
    return (r1[0].copy(), False, r1[2] + " / next face: " + r2[2])


def splice_pick(
    pick: TowerGraspPlan, ref old_legs: List[PlanLeg], ref old_jaws: List[Float64],
    jaw_pick: Float64, pick_in_bowl: Bool, mut jaws: List[Float64],
) -> List[PlanLeg]:
    """The re-aimed pick's legs, then the OLD legs from the carry on (the
    place is unchanged — its legs need no new IK)."""
    var out = List[PlanLeg]()
    jaws.clear()
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
    var start = -1
    for i in range(len(old_legs)):
        if old_legs[i].name == "carry":
            start = i
            break
    if start >= 0:
        for i in range(start, len(old_legs)):
            out.append(old_legs[i].copy())
            jaws.append(old_jaws[i])
    return out^


def scene_unchanged(
    mut reader: CameraReader, cam: RigCamera, mut frame: List[UInt8], sc: Scene,
) raises -> String:
    """Just before the arm moves: are the bowl and the brick still where the
    plan was made? Empty if so, else why not. A plan made on a scene the
    operator has since changed drags the bowl (cycle run 2, ep 10: the bowl
    moved 159 mm)."""
    var roi = tower_desk_roi()
    var brick = PrismModel.tower_brick()
    var bowl = PrismModel.tower_bowl()
    var ox = List[Float64]()
    var oy = List[Float64]()
    var bx = List[Float64]()
    var by = List[Float64]()
    var n_frames = 0
    var t0 = perf_counter_ns()
    while n_frames < 8 and Float64(perf_counter_ns() - t0) * 1e-9 < 2.0:
        if reader.take_latest(frame) == 0:
            _ = sleep_us(2000)
            continue
        n_frames += 1
        var eo = estimate_prism_pose(frame, cam, printed_bowl_hsv(), bowl, roi)
        var eb = estimate_prism_pose(frame, cam, printed_brick_hsv(), brick, roi)
        if pose_confident(eo):
            ox.append(eo.x)
            oy.append(eo.y)
        if (sc.in_bowl and eb.found) or (not sc.in_bowl and pose_confident(eb)):
            bx.append(eb.x)
            by.append(eb.y)
    if len(ox) * 2 < max(n_frames, 1):
        return String("the bowl is not seen confidently")
    sort(ox)
    sort(oy)
    var dbowl = sqrt((ox[len(ox) // 2] - sc.bowl_x) ** 2 + (oy[len(oy) // 2] - sc.bowl_y) ** 2) * 1000.0
    if dbowl > SCENE_MOVED_MM:
        return "the bowl moved " + fixed(dbowl, 1) + " mm"
    if len(bx) * 2 < max(n_frames, 1):
        return String("the brick is not seen")
    sort(bx)
    sort(by)
    var dbrick = sqrt((bx[len(bx) // 2] - sc.brick_x) ** 2 + (by[len(by) // 2] - sc.brick_y) ** 2) * 1000.0
    if dbrick > SCENE_MOVED_MM:
        return "the brick moved " + fixed(dbrick, 1) + " mm"
    return String("")


def read_outcome(
    mut reader: CameraReader, cam: RigCamera, mut frame: List[UInt8],
    sc: Scene, in_bowl_mm: Float64,
) raises -> Tuple[String, Float64, Float64, Float64, Float64]:
    """(bucket, brick x, brick y, median brick-to-bowl mm, bowl moved mm)
    over ~1 s of frames; SUCCESS when at least half the frames see the brick
    within `in_bowl_mm`. The brick is taken at ANY coverage: in the bowl its wall hides
    part of it."""
    var roi = tower_desk_roi()
    var brick = PrismModel.tower_brick()
    var bowl = PrismModel.tower_bowl()
    var n_seen = 0
    var n_in = 0
    var sx = 0.0
    var sy = 0.0
    var ds = List[Float64]()
    var sbm = 0.0
    var n_frames = 0
    var t0 = perf_counter_ns()
    while n_frames < 25 and Float64(perf_counter_ns() - t0) * 1e-9 < 3.0:
        if reader.take_latest(frame) == 0:
            _ = sleep_us(2000)
            continue
        n_frames += 1
        var eb = estimate_prism_pose(frame, cam, printed_brick_hsv(), brick, roi)
        var eo = estimate_prism_pose(frame, cam, printed_bowl_hsv(), bowl, roi)
        var wx = sc.bowl_x
        var wy = sc.bowl_y
        if pose_confident(eo):
            wx = eo.x
            wy = eo.y
            sbm += sqrt((eo.x - sc.bowl_x) ** 2 + (eo.y - sc.bowl_y) ** 2) * 1000.0
        if not eb.found:
            continue
        n_seen += 1
        var d = sqrt((eb.x - wx) ** 2 + (eb.y - wy) ** 2) * 1000.0
        sx += eb.x
        sy += eb.y
        ds.append(d)
        if d < in_bowl_mm:
            n_in += 1
    if n_seen * 2 < max(n_frames, 1):
        return (String("LOST"), 0.0, 0.0, -1.0, sbm / Float64(max(n_frames, 1)))
    var ex = sx / Float64(n_seen)
    var ey = sy / Float64(n_seen)
    # the MEDIAN distance: a half-hidden brick's fit jumps on some frames
    # (the first run's means were 12 mm above its last frames)
    sort(ds)
    var d = ds[len(ds) // 2]
    var bm = sbm / Float64(max(n_frames, 1))
    if n_in * 2 >= n_seen:
        return (String("SUCCESS"), ex, ey, d, bm)
    var moved = sqrt((ex - sc.brick_x) ** 2 + (ey - sc.brick_y) ** 2) * 1000.0
    if moved < MISSED_MM:
        return (String("MISSED"), ex, ey, d, bm)
    return (String("MOVED"), ex, ey, d, bm)


def scene_qpos(
    ref q_scene: List[Float64], ref arm_q: List[Float64], sc: Scene,
    brick_adr: Int, bowl_adr: Int, ref lo: List[Float64], ref hi: List[Float64],
) -> List[Float64]:
    """The sim scene in the rig's state: the arm's joints (clamped to the
    model's limits) and the props at the estimates, at their resting heights."""
    var qs = q_scene.copy()
    for k in range(ACT):
        qs[k] = max(lo[k], min(hi[k], arm_q[k]))
    qs[brick_adr] = sc.brick_x
    qs[brick_adr + 1] = sc.brick_y
    qs[brick_adr + 3] = cos(sc.brick_yaw / 2.0)
    qs[brick_adr + 4] = 0.0
    qs[brick_adr + 5] = 0.0
    qs[brick_adr + 6] = sin(sc.brick_yaw / 2.0)
    qs[bowl_adr] = sc.bowl_x
    qs[bowl_adr + 1] = sc.bowl_y
    qs[bowl_adr + 3] = cos(sc.bowl_yaw / 2.0)
    qs[bowl_adr + 4] = 0.0
    qs[bowl_adr + 5] = 0.0
    qs[bowl_adr + 6] = sin(sc.bowl_yaw / 2.0)
    return qs^


def plan_warnings(plan: TowerGraspPlan) -> String:
    """Why a plan should not run without `force` (empty: none)."""
    var w = String("")
    var worst_ik = max(max(plan.e_grasp, plan.e_lift), max(plan.e_carry, plan.e_place)) * 1000.0
    if worst_ik > IK_OK_MM:
        w += "  ⚠ the IK misses a leg by " + fixed(worst_ik, 1) + " mm\n"
    if plan.pen_mm > PLAN_PEN_OK_MM:
        w += "  ⚠ the plan passes " + fixed(plan.pen_mm, 1) + " mm into an obstacle\n"
    if not plan.close_on_tip:
        w += "  ⚠ not a fingertip-triggered plan\n"
    return w^


def print_legs(legs: List[PlanLeg], jaws: List[Float64]):
    for li in range(len(legs)):
        var leg = legs[li].copy()
        if leg.name == "hold":
            continue
        var qs = String("")
        for i in range(len(leg.q)):
            qs += " " + _deg(leg.q[i])
        print(
            "    " + pad8(leg.name), leg.steps, ("open " + fixed(jaws[li], 2)) if leg.grip_open else "closed   ",
            "close_on_tip" if leg.close_on_tip else "", "q deg" + (qs if len(leg.q) > 0 else " (hold)"),
        )


def print_plan(plan: TowerGraspPlan, legs: List[PlanLeg]):
    print(
        "  PLAN tilt", _deg(plan.tilt), "deg | pinch", "tangential" if plan.tangential else "radial",
        fixed((plan.yaw - plan.bearing) * 180.0 / pi, 1), "deg from radial | tip_goal",
        fixed(plan.tip_goal[0], 3), fixed(plan.tip_goal[1], 3), fixed(plan.tip_goal[2], 3),
        "| raised", fixed(plan.raise_mm, 1), "mm | penetration", fixed(plan.pen_mm, 1),
        "mm |", plan.tries, "tries",
    )
    print(
        "       IK err mm: grasp", fixed(plan.e_grasp * 1000.0, 1), "lift",
        fixed(plan.e_lift * 1000.0, 1), "carry", fixed(plan.e_carry * 1000.0, 1),
        "place", fixed(plan.e_place * 1000.0, 1),
    )
    for leg in legs:
        var qs = String("")
        for i in range(len(leg.q)):
            qs += " " + _deg(leg.q[i])
        print(
            "    " + pad8(leg.name), leg.steps, "open  " if leg.grip_open else "closed",
            "close_on_tip" if leg.close_on_tip else "", "q deg" + (qs if len(leg.q) > 0 else " (hold)"),
        )


struct TaskPlan(Movable):
    """One cycle task's plan: its legs (with each leg's open-jaw target), the
    close trigger's goal, the release spot and the diagnostics."""

    var legs: List[PlanLeg]
    var jaws: List[Float64]
    var tip_goal: List[Float64]
    var tx: Float64
    var ty: Float64
    """Where the brick is released (task 1: the bowl's centre; task 2: the
    drawn desk spot)."""
    var place_yaw: Float64
    var pick_x: Float64
    var pick_y: Float64
    var pick_z: Float64
    """Where the pick plan assumes the brick's centre (world m)."""
    var pick_plan: TowerGraspPlan
    var place_plan: TowerGraspPlan
    var pick_seed0: Int
    """The first seed the pick was planned from: a re-plan from the wrist
    look starts there too, so it draws the same posture."""
    var tilt_pick: Float64
    var tilt_place: Float64
    var seed: Int
    var draws: Int
    var ok: Bool

    def __init__(out self):
        self.legs = List[PlanLeg]()
        self.jaws = List[Float64]()
        self.tip_goal = List[Float64](length=3, fill=0.0)
        self.tx = 0.0
        self.ty = 0.0
        self.place_yaw = 0.0
        self.pick_x = 0.0
        self.pick_y = 0.0
        self.pick_z = 0.0
        self.pick_plan = TowerGraspPlan()
        self.place_plan = TowerGraspPlan()
        self.pick_seed0 = 0
        self.tilt_pick = 0.0
        self.tilt_place = 0.0
        self.seed = 0
        self.draws = 0
        self.ok = False


def plan_task(
    task_n: Int, mut env: E, mut planner: TowerGraspPlanner, ref body_names: List[String],
    ref qs: List[Float64], ref q5: List[Float64], sc: Scene, ref q_scene: List[Float64],
    brick_adr: Int, bz: Float64, cfg_desk_pick: PlanCfg, cfg_desk_release: PlanCfg,
    cfg_bowl_pick: PlanCfg, cfg_bowl_release: PlanCfg, jaw_open: Float64, bowl_jaw: Float64,
    dr_valid: Bool, dr_x: Float64, dr_y: Float64, dr_yaw: Float64, seed: Int, verbose: Bool,
) raises -> TaskPlan:
    """TASK 1: pick the brick from the desk and RELEASE IT LOW at the bowl's
    centre (a place plan for a virtual brick there, the fingers 5 mm above
    the floor). TASK 2: pick it in the bowl at the camera's bias-corrected
    read (the last release `dr_*`, in the world, cross-checks it and gives
    the yaw), kept within `BOWL_PICK_MAX_OFF_M` of the centre — and
    release it at a drawn desk spot clear of the bowl."""
    var tp = TaskPlan()
    var draw = 0
    var rest_z = q_scene[brick_adr + 2]
    if task_n == 1:
        var pb: List[Float64] = [sc.brick_x, sc.brick_y, rest_z]
        tp.pick_seed0 = seed + draw
        var pk = plan_clean(env, planner, body_names, qs, pb, sc.brick_yaw, q5, cfg_desk_pick, seed + draw, PICK_DRAWS)
        draw += pk.draws
        var pc: List[Float64] = [sc.bowl_x, sc.bowl_y, bz]
        var qv = with_brick(qs, brick_adr, sc.bowl_x, sc.bowl_y, bz, pk.plan.yaw)
        var pl = plan_clean(env, planner, body_names, qv, pc, pk.plan.yaw, pk.plan.q_lift, cfg_bowl_release, seed + draw, PICK_DRAWS)
        draw += pl.draws
        tp.legs = pick_place_legs(env, planner, pk.plan, pl.plan, jaw_open, cfg_bowl_release.jaw, False, tp.jaws)
        tp.pick_plan = pk.plan.copy()
        tp.place_plan = pl.plan.copy()
        for k in range(3):
            tp.tip_goal[k] = pk.plan.tip_goal[k]
        tp.tx = sc.bowl_x
        tp.ty = sc.bowl_y
        tp.pick_x = sc.brick_x
        tp.pick_y = sc.brick_y
        tp.pick_z = rest_z
        tp.place_yaw = pl.plan.yaw
        tp.tilt_pick = pk.plan.tilt
        tp.tilt_place = pl.plan.tilt
        tp.seed = pk.seed
        tp.ok = pk.ok and pl.ok
        if verbose:
            print("  TASK 1: brick -> the bowl's centre, released low | seed", tp.seed)
            print("  pick:")
            print_plan(pk.plan, List[PlanLeg]())
            print("  release in the bowl:")
            print_plan(pl.plan, List[PlanLeg]())
            print_legs(tp.legs, tp.jaws)
            if not pk.ok:
                print("  ⚠ no clean pick in", pk.draws, "draws")
            if not pl.ok:
                print("  ⚠ no clean release in", pl.draws, "draws")
        tp.draws = draw
        return tp^
    var px: Float64
    var py: Float64
    var pyaw: Float64
    var src: String
    # THE CAMERA'S IN-BOWL READ, BIAS-CORRECTED, is the pick point: minus
    # `IN_BOWL_CAM_BIAS_*` it matched the release point to ~5 mm on six
    # undisturbed bricks, and it caught the one the release point did not
    # (cycle run 2, ep 4: the brick 13 mm off the release along the pinch
    # axis — it sat differently in the gripper — and three picks at the
    # release point closed empty, the failed ones nudging it further). The
    # release point is the cross-check, and gives the yaw when they agree.
    var cx = sc.brick_x - IN_BOWL_CAM_BIAS_X
    var cy = sc.brick_y - IN_BOWL_CAM_BIAS_Y
    var dx = cx - sc.bowl_x
    var dy = cy - sc.bowl_y
    var dd = sqrt(dx * dx + dy * dy)
    var k = min(1.0, BOWL_PICK_MAX_OFF_M / dd) if dd > 0.0 else 0.0
    px = sc.bowl_x + dx * k
    py = sc.bowl_y + dy * k
    var dr_cam = sqrt((cx - dr_x) ** 2 + (cy - dr_y) ** 2)
    if dr_valid and dr_cam <= DR_YAW_AGREE_M:
        pyaw = dr_yaw
        src = (
            "the camera, bias-corrected (" + fixed(dd * 1000.0, 1) + " mm off the bowl's centre;"
            + " the last release " + fixed(dr_cam * 1000.0, 1) + " mm away, its yaw)"
        )
    else:
        pyaw = sc.brick_yaw
        src = "the camera, bias-corrected (" + fixed(dd * 1000.0, 1) + " mm off the bowl's centre"
        if dr_valid:
            src += "; ⚠ " + fixed(dr_cam * 1000.0, 1) + " mm from the last release — the brick moved, the camera's yaw"
        src += ")"
        if k < 1.0:
            src += " pulled to " + fixed(BOWL_PICK_MAX_OFF_M * 1000.0, 0) + " mm"
    var pb: List[Float64] = [px, py, bz]
    var qb = with_brick(qs, brick_adr, px, py, bz, pyaw)
    tp.pick_seed0 = seed + draw
    var pk = plan_bowl_pick(env, planner, body_names, qb, pb, pyaw, q5, cfg_bowl_pick, seed + draw)
    draw += pk.draws
    # the spot: drawn in the brick's region, clear of the bowl, with a clean
    # release plan
    var pl = Planned(TowerGraspPlan(), False, 0, 0)
    seed_rng(seed + draw)
    draw += 1
    var spots = List[Float64]()
    for _ in range(SPOT_TRIES):
        var sx = random_float64(0.12, 0.35)
        var sy = random_float64(-0.22, 0.19)
        spots.append(sx)
        spots.append(sy)
    for t in range(SPOT_TRIES):
        var sx = spots[2 * t]
        var sy = spots[2 * t + 1]
        if sqrt((sx - sc.bowl_x) ** 2 + (sy - sc.bowl_y) ** 2) < SPOT_MIN_BOWL_M:
            continue
        var ps: List[Float64] = [sx, sy, rest_z]
        var qv = with_brick(qb, brick_adr, sx, sy, rest_z, pk.plan.yaw)
        pl = plan_clean(env, planner, body_names, qv, ps, pk.plan.yaw, pk.plan.q_lift, cfg_desk_release, seed + draw, 3)
        draw += pl.draws
        tp.tx = sx
        tp.ty = sy
        if pl.ok:
            break
    tp.legs = pick_place_legs(env, planner, pk.plan, pl.plan, bowl_jaw, jaw_open, True, tp.jaws)
    tp.pick_plan = pk.plan.copy()
    tp.place_plan = pl.plan.copy()
    for k in range(3):
        tp.tip_goal[k] = pk.plan.tip_goal[k]
    tp.place_yaw = pl.plan.yaw
    tp.pick_x = px
    tp.pick_y = py
    tp.pick_z = bz
    tp.tilt_pick = pk.plan.tilt
    tp.tilt_place = pl.plan.tilt
    tp.seed = pk.seed
    tp.ok = pk.ok and pl.ok
    tp.draws = draw
    if verbose:
        print(
            "  TASK 2: brick out of the bowl -> the desk at (", fixed(tp.tx * 1000.0, 1), ",",
            fixed(tp.ty * 1000.0, 1), ") mm | the brick taken at (", fixed(px * 1000.0, 1), ",",
            fixed(py * 1000.0, 1), ") from", src, "| seed", tp.seed,
        )
        print("  pick in the bowl:")
        print_plan(pk.plan, List[PlanLeg]())
        print("  release on the desk:")
        print_plan(pl.plan, List[PlanLeg]())
        print_legs(tp.legs, tp.jaws)
        if not pk.ok:
            print("  ⚠ no clean pick in", pk.draws, "draws")
        if not pl.ok:
            print("  ⚠ no clean release spot in", SPOT_TRIES, "tries")
    return tp^


def plan_cycles(
    path: String, mut env: E, mut planner: TowerGraspPlanner, ref body_names: List[String],
    ref q_scene: List[Float64], brick_adr: Int, bowl_adr: Int, seed0: Int,
    cfg_desk_pick: PlanCfg, cfg_desk_release: PlanCfg, cfg_bowl_pick: PlanCfg,
    cfg_bowl_release: PlanCfg, jaw_open: Float64, bowl_jaw: Float64,
) raises:
    """`--plan-cycle CSV`: the cycle's two plans on every layout of a
    start-poses file, from the sim's rest pose — task 1 from the recorded
    brick, then task 2 from the brick released at that bowl's centre. No
    camera, no arm."""
    var bz = brick_in_bowl_z(env, q_scene, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
    var lines = open(path, "r").read().split("\n")
    var arm_q = List[Float64]()
    for k in range(ACT):
        arm_q.append(q_scene[k])
    var q5 = List[Float64]()
    for k in range(N_ARM):
        q5.append(q_scene[k])
    var n = 0
    var ok1 = 0
    var ok2 = 0
    var both = 0
    var draws = 0
    var reaim1 = 0
    var reaim2 = 0
    var reaim_ms = 0.0
    print("  ep  brick mm      bowl mm       task1 tilts   task2 tilts   spot mm       ok")
    for li in range(1, len(lines)):
        var cols = lines[li].split(",")
        if len(cols) < 5 or String(cols[1]) == "" or String(cols[3]) == "":
            continue
        var sc = Scene(
            Float64(String(cols[1])) / 1000.0, Float64(String(cols[2])) / 1000.0, 0.0,
            Float64(String(cols[3])) / 1000.0, Float64(String(cols[4])) / 1000.0, 0.0, False,
        )
        if sqrt((sc.brick_x - sc.bowl_x) ** 2 + (sc.brick_y - sc.bowl_y) ** 2) < MIN_SEP_M:
            continue
        var qs = scene_qpos(q_scene, arm_q, sc, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
        var t1 = plan_task(
            1, env, planner, body_names, qs, q5, sc, q_scene, brick_adr, bz,
            cfg_desk_pick, cfg_desk_release, cfg_bowl_pick, cfg_bowl_release,
            jaw_open, bowl_jaw, False, 0.0, 0.0, 0.0, seed0 + 100 * n, n == 0,
        )
        # task 2 from the brick where task 1 released it
        var sc2 = Scene(sc.bowl_x, sc.bowl_y, t1.place_yaw, sc.bowl_x, sc.bowl_y, 0.0, True)
        var qs2 = scene_qpos(q_scene, arm_q, sc2, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
        var t2 = plan_task(
            2, env, planner, body_names, qs2, q5, sc2, q_scene, brick_adr, bz,
            cfg_desk_pick, cfg_desk_release, cfg_bowl_pick, cfg_bowl_release,
            jaw_open, bowl_jaw, True, sc.bowl_x, sc.bowl_y, t1.place_yaw, seed0 + 100 * n + 50, n == 0,
        )
        # the fast re-aim, as if the wrist saw each brick REAIM_TEST_MM off
        # (a direction drawn per layout) and turned 15 deg
        seed_rng(seed0 + 7000 + n)
        var ang = random_float64(0.0, 2.0 * pi)
        var ox = REAIM_TEST_MM / 1000.0 * cos(ang)
        var oy = REAIM_TEST_MM / 1000.0 * sin(ang)
        var t_a = perf_counter_ns()
        var qa = with_brick(qs, brick_adr, t1.pick_x + ox, t1.pick_y + oy, t1.pick_z, sc.brick_yaw + 0.26)
        var ra1 = reaim_pick(env, planner, t1.pick_plan, qa, t1.pick_x + ox, t1.pick_y + oy, t1.pick_z, sc.brick_yaw + 0.26, jaw_open)
        var qb = with_brick(qs2, brick_adr, t2.pick_x + ox, t2.pick_y + oy, t2.pick_z, t1.place_yaw + 0.26)
        var ra2 = reaim_pick(env, planner, t2.pick_plan, qb, t2.pick_x + ox, t2.pick_y + oy, t2.pick_z, t1.place_yaw + 0.26, bowl_jaw)
        reaim_ms += Float64(perf_counter_ns() - t_a) * 1e-6 / 2.0
        if t1.ok and ra1[1]:
            reaim1 += 1
        if t2.ok and ra2[1]:
            reaim2 += 1
        if n < 3:
            print("    re-aim task 1:", ra1[2], "| task 2:", ra2[2])
        n += 1
        draws += t1.draws + t2.draws
        if t1.ok:
            ok1 += 1
        if t2.ok:
            ok2 += 1
        if t1.ok and t2.ok:
            both += 1
        print(
            "  " + pad8(String(cols[0])), fixed(sc.brick_x * 1000.0, 0), fixed(sc.brick_y * 1000.0, 0),
            "  ", fixed(sc.bowl_x * 1000.0, 0), fixed(sc.bowl_y * 1000.0, 0), "  ",
            _deg(t1.tilt_pick), _deg(t1.tilt_place), "  ", _deg(t2.tilt_pick), _deg(t2.tilt_place),
            "  ", fixed(t2.tx * 1000.0, 0), fixed(t2.ty * 1000.0, 0), "  ",
            ("ok" if t1.ok else "T1-FAIL") + " " + ("ok" if t2.ok else "T2-FAIL"),
        )
    print(
        "  layouts", n, "| task 1 clean", ok1, "| task 2 clean", ok2, "| both", both,
        "| draws per layout", fixed(Float64(draws) / Float64(max(n, 1)), 1),
    )
    print(
        "  fast re-aim (", fixed(REAIM_TEST_MM, 0), "mm off, +15 deg): clean for task 1", reaim1, "/", ok1,
        "| task 2", reaim2, "/", ok2, "| mean", fixed(reaim_ms / Float64(max(n, 1)), 1), "ms per pick",
    )


def plan_layouts(
    path: String, mut env: E, mut planner: TowerGraspPlanner,
    ref q_scene: List[Float64], brick_adr: Int, bowl_adr: Int, seed0: Int,
    jaw_open: Float64,
) raises:
    """`--plan-only CSV`: plan every layout of a start-poses file
    (`tower_pose_real_check.mojo`'s: episode, brick x/y, bowl x/y in world mm)
    from the sim's rest pose, brick yaw 0 (the file has none). No camera, no
    arm: the planning half on the real layouts, before a desk session."""
    var lines = open(path, "r").read().split("\n")
    var v0 = List[Float64](length=NV, fill=0.0)
    var arm_q = List[Float64]()
    for k in range(ACT):
        arm_q.append(q_scene[k])
    var n = 0
    var n_clean = 0
    var n_near = 0
    var n_near_clean = 0
    var n_skip = 0
    print("  ep  brick mm        bowl mm         r_mm  tilt tries pen_mm  ik_mm  ok")
    for li in range(1, len(lines)):
        var cols = lines[li].split(",")
        if len(cols) < 5:
            continue
        if String(cols[1]) == "" or String(cols[3]) == "":
            n_skip += 1
            continue
        var sc = Scene(
            Float64(String(cols[1])) / 1000.0, Float64(String(cols[2])) / 1000.0, 0.0,
            Float64(String(cols[3])) / 1000.0, Float64(String(cols[4])) / 1000.0, 0.0,
            False,
        )
        if sqrt((sc.brick_x - sc.bowl_x) ** 2 + (sc.brick_y - sc.bowl_y) ** 2) < MIN_SEP_M:
            n_skip += 1
            continue
        var qs = scene_qpos(q_scene, arm_q, sc, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
        var pb: List[Float64] = [sc.brick_x, sc.brick_y, q_scene[brick_adr + 2]]
        var pw: List[Float64] = [sc.bowl_x, sc.bowl_y, q_scene[bowl_adr + 2]]
        var q5 = List[Float64]()
        for k in range(N_ARM):
            q5.append(qs[k])
        env.set_state(qs, v0)
        seed_rng(seed0 + n)
        var plan = planner.plan_grasp(env, pb, sc.brick_yaw, q5, jaw_open)
        planner.plan_place(env, plan, pw)
        var ok = plan_warnings(plan).byte_length() == 0
        var r = sqrt(sc.brick_x ** 2 + sc.brick_y ** 2)
        n += 1
        if ok:
            n_clean += 1
        if r < 0.25:
            n_near += 1
            if ok:
                n_near_clean += 1
        var worst_ik = max(max(plan.e_grasp, plan.e_lift), max(plan.e_carry, plan.e_place)) * 1000.0
        print(
            "  " + pad8(String(cols[0])), fixed(sc.brick_x * 1000.0, 0), fixed(sc.brick_y * 1000.0, 0),
            "  ", fixed(sc.bowl_x * 1000.0, 0), fixed(sc.bowl_y * 1000.0, 0), "  ",
            fixed(r * 1000.0, 0), " ", _deg(plan.tilt), " ", plan.tries, " ",
            fixed(plan.pen_mm, 1), " ", fixed(worst_ik, 1), " ", "ok" if ok else "FORCE",
        )
    print(
        "  layouts planned", n, "| clean (no force needed)", n_clean, "| bricks nearer than 0.25 m:",
        n_near, "(clean", String(n_near_clean) + ")", "| skipped", n_skip,
    )


def brick_in_bowl_z(
    mut env: E, ref q_scene: List[Float64], brick_adr: Int, bowl_adr: Int,
    ref lo: List[Float64], ref hi: List[Float64],
) raises -> Float64:
    """The brick's resting height IN the bowl (world m): dropped 2 cm above
    the floor at the bowl's centre and settled — with the bowl upright at a
    fixed CLEAR spot (0.30, 0.10), not the scene's own pose: a bowl near the
    base touches the held arm, is lifted during the settle, and the brick
    then "rests" 10.8 mm below the bowl's origin instead of 16.5 (noeira-54,
    MuJoCo agrees). The height does not depend on where the bowl is."""
    var qs = q_scene.copy()
    qs[bowl_adr] = 0.30
    qs[bowl_adr + 1] = 0.10
    qs[bowl_adr + 3] = 1.0
    qs[bowl_adr + 4] = 0.0
    qs[bowl_adr + 5] = 0.0
    qs[bowl_adr + 6] = 0.0
    qs[brick_adr] = qs[bowl_adr]
    qs[brick_adr + 1] = qs[bowl_adr + 1]
    qs[brick_adr + 2] = qs[bowl_adr + 2] + 0.045
    qs[brick_adr + 3] = 1.0
    qs[brick_adr + 4] = 0.0
    qs[brick_adr + 5] = 0.0
    qs[brick_adr + 6] = 0.0
    var v0 = List[Float64](length=NV, fill=0.0)
    env.set_state(qs, v0)
    var hold = ContAction[ACT]()
    for k in range(ACT):
        hold.data[k] = 2.0 * (qs[k] - lo[k]) / (hi[k] - lo[k]) - 1.0
    for _ in range(60):
        _ = env.step(hold)
    var z = Float64(env.d.qpos.data[brick_adr + 2])
    var dxy = sqrt(
        (Float64(env.d.qpos.data[brick_adr]) - qs[bowl_adr]) ** 2
        + (Float64(env.d.qpos.data[brick_adr + 1]) - qs[bowl_adr + 1]) ** 2
    )
    print(
        "  brick in the bowl (sim, settled): z", fixed(z, 4), "| bowl z",
        fixed(qs[bowl_adr + 2], 4), "| drifted", fixed(dxy * 1000.0, 1), "mm from the centre",
    )
    return z


def plan_bowl_pick(
    mut env: E, mut planner: TowerGraspPlanner, ref body_names: List[String],
    ref qs: List[Float64], ref pb: List[Float64], yaw: Float64, ref q5: List[Float64],
    cfg: PlanCfg, seed: Int,
) raises -> Planned:
    """`plan_clean` for a pick IN the bowl, with the LIFT held to
    `BOWL_LIFT_OK_MM` instead of 10: out of the bowl the arm first backs up
    its approach, and the lift is then a pose 15 cm up, in the air — near the
    wall the steep bowl postures miss it by 15-60 mm, and cycle run 5 found no
    pick at all for bricks 15-20 mm off the centre (the grasp itself clean to
    < 1 mm). The collision veto and the grasp's IK are unchanged."""
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
    for d in range(PICK_DRAWS):
        env.set_state(qs, v0)
        seed_rng(seed + d)
        plan = planner.plan_grasp(env, pb, yaw, q5, cfg.jaw)
        n = d + 1
        ok = (
            plan.pen_mm <= PLAN_PEN_OK_MM and plan.e_grasp * 1000.0 <= IK_OK_MM
            and plan.e_lift * 1000.0 <= BOWL_LIFT_OK_MM and plan.close_on_tip
        )
        if ok:
            break
    var desk: List[String] = ["desk_mat"]
    planner.set_support(body_names, desk, False)
    planner.desk_clear_m = saved_clear
    planner.posture.tilt_lo = saved_lo
    planner.posture.tilt_hi = saved_hi
    env.set_state(qs, v0)
    return Planned(plan^, ok, n, seed + n - 1)


def plan_bowl_picks(
    path: String, mut env: E, mut planner: TowerGraspPlanner,
    ref q_scene: List[Float64], brick_adr: Int, bowl_adr: Int, seed0: Int,
    jaw_open: Float64, ref names: List[String],
) raises:
    """`--plan-bowl-pick CSV`: task 2's pick — the brick IN the bowl at each
    layout's bowl pose, at the centre and 15 mm off it — planned from the sim's
    rest pose. The feasibility question: does the planner find a grasp that
    clears the bowl's 45 mm wall?"""
    var bz = brick_in_bowl_z(env, q_scene, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
    # the grasp height against the bowl's FLOOR (`set_support`, by the
    # contact normal; the walls stay in the veto), the fingers touching it
    var cfg = PlanCfg(
        planner.posture.tilt_lo * 180.0 / pi, planner.posture.tilt_hi * 180.0 / pi,
        jaw_open, String("bowl_bowl"), 0.0,
    )
    var lines = open(path, "r").read().split("\n")
    var v0 = List[Float64](length=NV, fill=0.0)
    var arm_q = List[Float64]()
    for k in range(ACT):
        arm_q.append(q_scene[k])
    var offs: List[Float64] = [0.0, 0.0, 0.015, 0.0, 0.0, 0.015, -0.015, 0.0, 0.0, -0.015]
    var n = 0
    var n_clean = 0
    var n_draws = 0
    var pen_sum = 0.0
    var tilt_ok = List[Float64]()
    for li in range(1, len(lines)):
        var cols = lines[li].split(",")
        if len(cols) < 5 or String(cols[3]) == "":
            continue
        var wx = Float64(String(cols[3])) / 1000.0
        var wy = Float64(String(cols[4])) / 1000.0
        for o in range(len(offs) // 2):
            var sc = Scene(wx + offs[2 * o], wy + offs[2 * o + 1], 0.0, wx, wy, 0.0, True)
            var qs = scene_qpos(q_scene, arm_q, sc, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
            qs[brick_adr + 2] = bz
            var pb: List[Float64] = [sc.brick_x, sc.brick_y, bz]
            var q5 = List[Float64]()
            for k in range(N_ARM):
                q5.append(qs[k])
            var pl = plan_bowl_pick(env, planner, names, qs, pb, 0.0, q5, cfg, seed0 + 1000 * n)
            var plan = pl.plan.copy()
            var ok = pl.ok
            var draws = pl.draws
            var worst_ik = plan.e_grasp * 1000.0
            env.set_state(qs, v0)
            n_draws += draws
            n += 1
            pen_sum += plan.pen_mm
            if ok:
                n_clean += 1
                tilt_ok.append(plan.tilt * 180.0 / pi)
            if o > 0 and not ok:
                print(
                    "    off-centre FAIL: pen", fixed(plan.pen_mm, 1), "mm | IK grasp",
                    fixed(plan.e_grasp * 1000.0, 1), "lift", fixed(plan.e_lift * 1000.0, 1), "mm | tilt", _deg(plan.tilt),
                )
            if o == 0 and not ok and n < 0:
                # what the grasp pose (jaw open) penetrates: body pair, depth,
                # contact height and distance from the bowl's axis
                var qg = plan.waypoints[len(plan.waypoints) - 1].copy() if len(plan.waypoints) > 0 else plan.q_grasp.copy()
                var saved = List[Float64]()
                for k in range(NQ):
                    saved.append(Float64(env.d.qpos.data[k]))
                for k in range(N_ARM):
                    env.d.qpos.data[k] = qg[k]
                env.d.qpos.data[N_ARM] = jaw_open
                env._fields_fk()
                detect_contacts_auto["cpu", DType.float64, BATCH=1](env.d, env.mf, None)
                var nc = Int(env.d.meta.data[META_IDX_NUM_CONTACTS])
                var hits = String("")
                for c in range(nc):
                    var co = c * CONTACT_SIZE
                    var dd = Float64(env.d.contacts.data[co + CONTACT_IDX_DIST])
                    if dd > -0.001:
                        continue
                    var ba = Int(env.d.contacts.data[co + CONTACT_IDX_BODY_A])
                    var bb = Int(env.d.contacts.data[co + CONTACT_IDX_BODY_B])
                    var cx = Float64(env.d.contacts.data[co + CONTACT_IDX_POS_X])
                    var cy = Float64(env.d.contacts.data[co + CONTACT_IDX_POS_Y])
                    var cz = Float64(env.d.contacts.data[co + CONTACT_IDX_POS_Z])
                    hits += (
                        "\n      " + names[ba] + "-" + names[bb] + " " + fixed(-dd * 1000.0, 1)
                        + " mm at z " + fixed(cz * 1000.0, 1) + " r " + fixed(sqrt((cx - wx) ** 2 + (cy - wy) ** 2) * 1000.0, 1)
                    )
                for k in range(NQ):
                    env.d.qpos.data[k] = saved[k]
                env._fields_fk()
                print("    contacts at the grasp:" + hits)
            if o == 0:
                print(
                    "  bowl", fixed(wx * 1000.0, 0), fixed(wy * 1000.0, 0), "| centre: tilt",
                    _deg(plan.tilt), "draws", draws, "tries", plan.tries, "pen", fixed(plan.pen_mm, 1),
                    "mm ik", fixed(worst_ik, 1), "mm raised", fixed(plan.raise_mm, 1),
                    "tip z", fixed(plan.tip_goal[2], 3), "ok" if ok else "FAIL",
                )
    var tmin = 1e9
    var tmax = -1e9
    for t in tilt_ok:
        tmin = min(tmin, t)
        tmax = max(tmax, t)
    print(
        "  bowl picks planned", n, "| clean within", PICK_DRAWS, "draws:", n_clean,
        "| draws per pick", fixed(Float64(n_draws) / Float64(max(n, 1)), 1), "| mean penetration (last draw)",
        fixed(pen_sum / Float64(max(n, 1)), 1), "mm | clean tilts", fixed(tmin, 0), "..", fixed(tmax, 0), "deg",
    )


def read_place_outcome(
    mut reader: CameraReader, cam: RigCamera, mut frame: List[UInt8],
    sc: Scene, tx: Float64, ty: Float64,
) raises -> Tuple[String, Float64, Float64, Float64]:
    """Task 2's outcome over ~1 s of frames: (bucket, brick x, y, mm from the
    target spot). SUCCESS: the brick confident on the desk within
    `PLACE_OK_MM` of the spot; MISSED: still in the bowl; MOVED: elsewhere;
    LOST: not seen."""
    var roi = tower_desk_roi()
    var brick = PrismModel.tower_brick()
    var n_frames = 0
    var n_conf = 0
    var n_bowl = 0
    var xs = List[Float64]()
    var ys = List[Float64]()
    var t0 = perf_counter_ns()
    while n_frames < 25 and Float64(perf_counter_ns() - t0) * 1e-9 < 3.0:
        if reader.take_latest(frame) == 0:
            _ = sleep_us(2000)
            continue
        n_frames += 1
        var eb = estimate_prism_pose(frame, cam, printed_brick_hsv(), brick, roi)
        if not eb.found:
            continue
        if sqrt((eb.x - sc.bowl_x) ** 2 + (eb.y - sc.bowl_y) ** 2) * 1000.0 < IN_BOWL_MM:
            n_bowl += 1
            continue
        if pose_confident(eb):
            n_conf += 1
            xs.append(eb.x)
            ys.append(eb.y)
    var half = max(n_frames, 1) // 2
    if n_conf > half:
        sort(xs)
        sort(ys)
        var ex = xs[len(xs) // 2]
        var ey = ys[len(ys) // 2]
        var d = sqrt((ex - tx) ** 2 + (ey - ty) ** 2) * 1000.0
        return (String("SUCCESS") if d < PLACE_OK_MM else String("MOVED"), ex, ey, d)
    if n_bowl > half:
        return (String("MISSED"), sc.bowl_x, sc.bowl_y, -1.0)
    return (String("LOST"), 0.0, 0.0, -1.0)


def _usage():
    print(
        "usage: tower_expert_real.mojo --camera DEV [--arm] [--episodes N] [--seed S]\n"
        "       [--desk-clear-mm MM] [--tilt-range LO,HI] [--jaw-open RAD] [--tip-close-mm MM]\n"
        "       [--settled-vel RAD_S] [--settle-steps N] [--leg-settle N] [--sag-ki K] [--pinch-offset-mm MM]\n"
        "       [--step TICKS] [--in-bowl-mm MM] [--calib FILE] [--port DEV] [--out DIR]\n"
        "       [--dataset NAME --wrist-camera DEV [--project P] [--task STR] [--resume]]\n"
        "       [--cycle [--auto] [--task2 STR] [--bowl-jaw RAD] [--bowl-release-jaw RAD] [--no-wrist-correct]]\n"
        "       tower_expert_real.mojo --plan-only START_POSES.csv [--seed S] [--tilt-range ..]"
    )


def main() raises:
    var args = argv()
    var camera = String("")
    var calib_path = String(OVERHEAD_CALIB)
    var port = String("")
    var live = False
    var n_episodes = 10
    var seed0 = 0
    var desk_clear_mm = DESK_CLEAR_MM
    var tilt_range = String(CLEAR_PLAN_TILT)
    var jaw_open = HUMAN_JAW_OPEN
    var tip_close_mm = TIP_CLOSE_MM
    var settled_vel = SETTLED_VEL
    var settle_steps = SETTLE_STEPS
    var leg_settle = LEG_SETTLE
    var sag_ki = 0.0
    var step_ticks = MAX_STEP_TICKS
    var in_bowl_mm = IN_BOWL_MM
    var out_dir = String("")
    var plan_only = String("")
    var plan_bowl = String("")
    var plan_cycle = String("")
    var pinch_offset_mm = 0.0
    var pinch_offset_set = False
    var cycle = False
    var auto = False
    var task2 = String(TASK2_LANGUAGE)
    var bowl_jaw = BOWL_JAW
    var bowl_release_jaw = BOWL_RELEASE_JAW
    var wrist_correct = True
    var project = String(DEFAULT_PROJECT)
    var dataset = String("")
    var wrist = String("")
    var task = String(TASK_LANGUAGE)
    var resume = False
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--arm":
            live = True
            i += 1
            continue
        if a == "--resume":
            resume = True
            i += 1
            continue
        if a == "--cycle":
            cycle = True
            i += 1
            continue
        if a == "--auto":
            auto = True
            i += 1
            continue
        if a == "--no-wrist-correct":
            wrist_correct = False
            i += 1
            continue
        if a == "--help" or a == "-h":
            _usage()
            return
        if i + 1 >= len(args):
            _usage()
            raise Error("flag " + a + " needs a value")
        var v = String(args[i + 1])
        if a == "--camera":
            camera = v
        elif a == "--calib":
            calib_path = v
        elif a == "--port":
            port = v
        elif a == "--episodes":
            n_episodes = Int(v)
        elif a == "--seed":
            seed0 = Int(v)
        elif a == "--desk-clear-mm":
            desk_clear_mm = Float64(v)
        elif a == "--tilt-range":
            tilt_range = v
        elif a == "--jaw-open":
            jaw_open = Float64(v)
        elif a == "--tip-close-mm":
            tip_close_mm = Float64(v)
        elif a == "--settled-vel":
            settled_vel = Float64(v)
        elif a == "--settle-steps":
            settle_steps = Int(v)
        elif a == "--leg-settle":
            leg_settle = Int(v)
        elif a == "--sag-ki":
            sag_ki = Float64(v)
        elif a == "--step":
            step_ticks = Int(v)
        elif a == "--in-bowl-mm":
            in_bowl_mm = Float64(v)
        elif a == "--out":
            out_dir = v
        elif a == "--plan-only":
            plan_only = v
        elif a == "--pinch-offset-mm":
            pinch_offset_mm = Float64(v)
            pinch_offset_set = True
        elif a == "--task2":
            task2 = v
        elif a == "--bowl-jaw":
            bowl_jaw = Float64(v)
        elif a == "--bowl-release-jaw":
            bowl_release_jaw = Float64(v)
        elif a == "--plan-cycle":
            plan_cycle = v
        elif a == "--plan-bowl-pick":
            plan_bowl = v
            plan_only = String("")
        elif a == "--project":
            project = v
        elif a == "--dataset":
            dataset = v
        elif a == "--wrist-camera":
            wrist = v
        elif a == "--task":
            task = v
        else:
            _usage()
            raise Error("unknown flag " + a)
        i += 2
    if plan_cycle != "" and not pinch_offset_set:
        pinch_offset_mm = 12.0
    if plan_bowl != "" or plan_cycle != "":
        plan_only = String("-")
    if camera == "" and plan_only == "":
        _usage()
        raise Error("--camera <index | /dev/... path> is required")
    if cycle and not pinch_offset_set:
        # the bowl's picks and releases need the brick BETWEEN the jaws: at
        # the narrow BOWL_JAW the moving jaw otherwise lands in it
        pinch_offset_mm = 12.0
    if auto and not cycle:
        raise Error("--auto runs the self-resetting cycle: it needs --cycle")
    var recording = dataset != "" and plan_only == ""
    var dataset_dir = String("")
    if recording:
        if not live:
            raise Error("--dataset records the episodes the ARM runs: it needs --arm")
        if wrist == "":
            raise Error(
                "--dataset needs --wrist-camera (the recordings have both views:"
                " observation.images.overhead, observation.images.wrist)"
            )
        dataset_dir = project_dataset_dir(project, dataset)
    if plan_only == "" and not opencv_shim_available():
        raise Error("the OpenCV shim is not built: pixi run build-opencv")
    var tr = tilt_range.split(",")
    if len(tr) != 2:
        raise Error("--tilt-range needs lo,hi in degrees, got " + tilt_range)
    if out_dir == "":
        out_dir = String(OUT_ROOT) + "/" + iso8601_utc(epoch_seconds()).replace(":", "-")
    if plan_only == "":
        makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    if live:
        print("so101_tower cube-in-bowl — the SIM EXPERT ON THE REAL ARM  [LIVE]")
    else:
        print("so101_tower cube-in-bowl — DRY RUN (reads, plans, prints; nothing moves)")
        print("  pass --arm to run the plans on the arm")
    print("=" * 70)

    # ── the planner's env, in the sim's own resting scene ────────────────
    var ctx = DeviceContext()
    var env = E(ctx)
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3], rheights[0],
    )
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DType.float64](cw[k])
    _ = env.reset()
    var mw = task_meta_words(
        String(TASK), String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    for k in range(len(mw[0])):
        env.d.meta.data[mw[0][k]] = Scalar[DType.float64](mw[1][k])
    var body_names = List[String]()
    var brick_body = -1
    var bowl_body = -1
    for b in range(len(fmd.body_names)):
        body_names.append(String(fmd.body_names[b]))
        if body_names[b] == "brick_brick":
            brick_body = b
        if body_names[b] == "bowl_bowl":
            bowl_body = b
    var brick_adr = -1
    var bowl_adr = -1
    var adr = 0
    for j in range(len(fmd.joints)):
        if fmd.joints[j].nq == 7 and fmd.joints[j].body_id == brick_body:
            brick_adr = adr
        if fmd.joints[j].nq == 7 and fmd.joints[j].body_id == bowl_body:
            bowl_adr = adr
        adr += fmd.joints[j].nq
    if brick_adr < 0 or bowl_adr < 0:
        raise Error("the brick's and the bowl's free joints were not found")
    var planner = TowerGraspPlanner(env, body_names)
    planner.posture.human = True
    planner.clear_plan = True
    planner.z_grasp = HUMAN_Z_GRASP
    planner.desk_clear_m = desk_clear_mm / 1000.0
    # the aim moved along the pinch axis so the brick sits BETWEEN the jaws
    # (0: the moving jaw sweeps it ~18 mm into the fixed finger — the first
    # rig run's 10/10; the sim recorder uses 12)
    planner.pinch_offset_m = pinch_offset_mm / 1000.0
    planner.posture.tilt_lo = Float64(String(tr[0])) * pi / 180.0
    planner.posture.tilt_hi = Float64(String(tr[1])) * pi / 180.0
    # the props' resting heights: the task's reset, settled as the recorder
    # settles it (5 steps, the arm holding)
    var q_scene = posed_qpos[So101TowerPlacement](
        String(TASK), String(FAMILY), So101TowerConfig.SLOT_RADIUS, UInt64(0),
    )
    var v0 = List[Float64](length=NV, fill=0.0)
    env.set_state(q_scene, v0)
    var hold = ContAction[ACT]()
    for k in range(ACT):
        hold.data[k] = 2.0 * (q_scene[k] - planner.arm.lo[k]) / (planner.arm.hi[k] - planner.arm.lo[k]) - 1.0
    for _ in range(5):
        _ = env.step(hold)
    for k in range(NQ):
        q_scene[k] = Float64(env.d.qpos.data[k])
    print(
        "  sim resting heights: brick z", fixed(q_scene[brick_adr + 2], 4), "| bowl z",
        fixed(q_scene[bowl_adr + 2], 4), "(world m)",
    )

    var tlo = Float64(String(tr[0]))
    var thi = Float64(String(tr[1]))
    var cfg_desk_pick = PlanCfg(tlo, thi, jaw_open, String("desk_mat"), desk_clear_mm / 1000.0)
    var cfg_desk_release = PlanCfg(tlo, thi, jaw_open, String("desk_mat"), RELEASE_GAP_M)
    var cfg_bowl_pick = PlanCfg(5.0, 35.0, bowl_jaw, String("bowl_bowl"), 0.0)
    var cfg_bowl_release = PlanCfg(5.0, 35.0, bowl_release_jaw, String("bowl_bowl"), RELEASE_GAP_M)
    if plan_cycle != "":
        plan_cycles(
            plan_cycle, env, planner, body_names, q_scene, brick_adr, bowl_adr, seed0,
            cfg_desk_pick, cfg_desk_release, cfg_bowl_pick, cfg_bowl_release, jaw_open, bowl_jaw,
        )
        return
    if plan_bowl != "":
        plan_bowl_picks(
            plan_bowl, env, planner, q_scene, brick_adr, bowl_adr, seed0, jaw_open,
            body_names,
        )
        return
    if plan_only != "":
        plan_layouts(
            plan_only, env, planner, q_scene, brick_adr, bowl_adr, seed0, jaw_open,
        )
        return

    # ── the camera ───────────────────────────────────────────────────────
    var lens_cal = read_calib(calib_path)
    lens_cal.require_size(640, 480)
    var lens = FisheyeLens.from_calib(lens_cal)
    var pose = tower_overhead_pose()
    var cam = RigCamera(lens, pose.pos, pose.rot_mj)
    print("  camera pose:", pose.source)
    var cams = List[CameraReader]()
    var specs: List[String] = [camera]
    if wrist != "":
        # recorded (the dataset's second view) and/or the wrist look's source
        specs.append(wrist)
    for c in range(len(specs)):
        var reader = CameraReader.from_spec(specs[c], 640, 480, Float64(HZ), rgb=True)
        reader.start()
        if reader.frame_bytes() != 640 * 480 * 3:
            raise Error("camera " + specs[c] + " delivers " + String(reader.frame_bytes()) + " bytes, not 640x480x3")
        var fps = reader.negotiated_fps()
        print("  camera", specs[c], "->", reader.resolved_node(), fixed(fps, 1), "fps")
        if recording and fps > 0.0 and fps < Float64(HZ) - 1.0:
            raise Error("camera " + specs[c] + " negotiated " + fixed(fps, 1) + " fps, below the " + String(HZ) + " the dataset claims")
        cams.append(reader^)
    var frame = List[UInt8](length=640 * 480 * 3, fill=UInt8(0))
    # the wrist look (LOG ONLY): the wrist camera's lens, and its pose from
    # the arm's FK
    var has_wrist = wrist != ""
    var wcal = read_calib(String(WRIST_CALIB))
    wcal.require_size(640, 480)
    var wlens = FisheyeLens.from_calib(wcal)
    var afk = TowerArmFK()
    var wci = afk.camera_index("wrist_cam")

    # ── the arm ──────────────────────────────────────────────────────────
    var the_port = follower_port(port)
    var why = port_refusal(the_port, String("follower"))
    if why.byte_length() > 0:
        raise Error("tower_expert_real: " + why)
    print("  opening", the_port)
    var arm = SO101Arm(the_port, max_step_ticks=step_ticks, track_step_ticks=TRACK_STEP_TICKS)
    arm.bus.timeout_ms = 20
    var lo = Array[Float64, SO101_N](fill=0.0)
    var hi = Array[Float64, SO101_N](fill=0.0)
    for k in range(SO101_N):
        lo[k] = planner.arm.lo[k]
        hi[k] = planner.arm.hi[k]
    var jmap = SimJointMap.tower_follower(arm.cal, lo^, hi^)
    print("  " + jmap.describe())
    # the map's inverse, at interior points (`deploy_reach_real.mojo`: a sign
    # or offset error in `from_sim` is a mirrored pose at full slew)
    var worst = 0.0
    for k in range(SO101_N):
        for s in range(3):
            var val = jmap.sim_lo[k] + (0.25 + 0.25 * Float64(s)) * (jmap.sim_hi[k] - jmap.sim_lo[k])
            worst = max(worst, abs(jmap.to_sim(arm.cal, k, jmap.from_sim(arm.cal, k, val)) - val))
    if worst > 0.02:
        raise Error("to_sim/from_sim do not round-trip (worst " + fixed(worst, 4) + " rad) — not arming")
    var rig = Rig(arm^, jmap^)
    for _ in range(len(cams)):
        rig.frames.append(List[UInt8](length=640 * 480 * 3, fill=UInt8(0)))
    rig.cams = cams^
    if recording:
        var names = List[String]()
        for k in range(SO101_N):
            names.append(joint_name(k) + ".pos")
        var cam_names: List[String] = ["observation.images.overhead", "observation.images.wrist"]
        rig.writers.append(open_recording(
            dataset_dir.copy(), HZ, names.copy(), names.copy(), cam_names^, 480, 640, resume,
        ))
        print(
            "  RECORDING ->", dataset_dir, "(" + String(rig.writers[0].n_episodes()),
            "episodes already there) | task:", task,
        )
    rig.jaw_open = jaw_open
    rig.jaw_lo = planner.arm.lo[5]
    rig.settled_vel = settled_vel
    rig.tip_close_mm = tip_close_mm
    rig.sag_ki = sag_ki
    if not rig.read_joints():
        raise Error("the follower did not report 6 positions")
    var start_pose = List[Int32](length=SO101_N, fill=Int32(0))
    for k in range(SO101_N):
        start_pose[k] = rig.raw[k]
    var stdin = StdinReader()
    var interactive = stdin_is_tty()
    print(
        "  plan: human posture, clear_plan, tilt", tilt_range, "deg, pinch offset", fixed(pinch_offset_mm, 1), "mm, desk clear",
        fixed(desk_clear_mm, 1), "mm, jaw open", fixed(jaw_open, 2), "rad | exec: tip close",
        fixed(tip_close_mm, 1), "mm, settled <", fixed(settled_vel, 2), "rad/s, sag ki",
        fixed(sag_ki, 3), "| out", out_dir,
    )

    var summary = String(
        "ep\ttask\tseed\tbrick_x\tbrick_y\tbrick_yaw_deg\tbowl_x\tbowl_y\tin_bowl\ttarget_x\ttarget_y"
        "\ttilt_pick_deg\ttilt_place_deg\tdraws\tclose\ttip_at_close_mm\tjaw_after_close\toutcome"
        "\tend_x\tend_y\tend_mm\tlate_ticks\tdropped\tdataset"
        "\tpick_x\tpick_y\tlook_conf\tlook_x\tlook_y\tlook_yaw_deg\tlook_used\n"
    )
    var n_run = 0
    var n_ok = 0
    var n_kept = 0
    var n_rejected = 0
    var buckets = String("")
    var draw = 0
    var bz = 0.0
    if cycle:
        bz = brick_in_bowl_z(env, q_scene, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
    # dead reckoning: where the last task-1 release put the brick, in the
    # WORLD (the camera reads a brick in the bowl 29-48 mm off; the bowl is
    # pushed during task 1, the brick is not)
    var dr_valid = False
    var dr_x = 0.0
    var dr_y = 0.0
    var dr_yaw = 0.0
    try:
        var ep = 0
        while ep < n_episodes:
            print("\n[episode", ep + 1, "/", n_episodes, "] the arm at rest, hands off")
            if auto and stdin.has_input():
                if stdin.line() == "q":
                    break
            var sc_opt = read_scene(rig.cams[0], cam, frame, stdin)
            if not sc_opt:
                print("  quit")
                break
            var sc = sc_opt.value().copy()
            save_png(out_dir + "/ep" + String(ep + 1) + "_start.png", frame, 640, 480, 3)
            var sep = sqrt((sc.brick_x - sc.bowl_x) ** 2 + (sc.brick_y - sc.bowl_y) ** 2)
            print(
                "  brick (", fixed(sc.brick_x * 1000.0, 1), ",", fixed(sc.brick_y * 1000.0, 1),
                ") mm yaw", _deg(sc.brick_yaw), "IN THE BOWL" if sc.in_bowl else "on the desk",
                "| bowl (", fixed(sc.bowl_x * 1000.0, 1), ",", fixed(sc.bowl_y * 1000.0, 1),
                ") mm | apart", fixed(sep * 1000.0, 1), "mm",
            )
            var task_n = 1
            if sc.in_bowl:
                if not cycle:
                    print("  ⚠ the brick is in the bowl — take it out (or run --cycle) and press Enter")
                    _ = stdin.line()
                    continue
                task_n = 2
            elif sep < MIN_SEP_M:
                print("  ⚠ the brick is against the bowl — move it and press Enter")
                _ = stdin.line()
                continue
            else:
                if not (sc.brick_x >= 0.10 and sc.brick_x <= 0.37 and sc.brick_y >= -0.24 and sc.brick_y <= 0.21):
                    print("  ⚠ the brick is outside the sim's desk_brick region (x 0.10..0.37, y -0.24..0.21)")
            if not (sc.bowl_x >= 0.13 and sc.bowl_x <= 0.38 and sc.bowl_y >= -0.23 and sc.bowl_y <= 0.19):
                print("  ⚠ the bowl is outside the sim's desk_bowl region (x 0.13..0.38, y -0.23..0.19)")
            if not rig.read_joints():
                print("  the follower did not report its pose; retrying")
                continue
            var q_home = rig.q.copy()
            # the planner's env in the rig's state
            var qs = scene_qpos(q_scene, rig.q, sc, brick_adr, bowl_adr, planner.arm.lo, planner.arm.hi)
            var q5 = List[Float64]()
            for k in range(N_ARM):
                q5.append(qs[k])
            # plan, and let the operator look at it (--auto: run a clean plan)
            var choice = String("r")
            var legs = List[PlanLeg]()
            var jaws = List[Float64]()
            var tip_goal = List[Float64](length=3, fill=0.0)
            var pick_x = sc.brick_x
            var pick_y = sc.brick_y
            var pick_z = q_scene[brick_adr + 2]
            var pick_yaw = sc.brick_yaw
            var pick_plan = TowerGraspPlan()
            var pick_seed0 = 0
            var tx = sc.bowl_x
            var ty = sc.bowl_y
            var tilt_pick = 0.0
            var tilt_place = 0.0
            var seed_used = 0
            var n_draws = 0
            var place_yaw = 0.0
            var redraws = 0
            while choice == "r":
                var ok_all: Bool
                if not cycle:
                    # the validated task 1 (10/10): the library's legs,
                    # the place a drop from Z_PLACE
                    var pb: List[Float64] = [sc.brick_x, sc.brick_y, q_scene[brick_adr + 2]]
                    var pw: List[Float64] = [sc.bowl_x, sc.bowl_y, q_scene[bowl_adr + 2]]
                    env.set_state(qs, v0)
                    seed_used = seed0 + draw
                    draw += 1
                    seed_rng(seed_used)
                    var plan = planner.plan_grasp(env, pb, sc.brick_yaw, q5, jaw_open)
                    planner.plan_place(env, plan, pw)
                    env.set_state(qs, v0)
                    legs = plan.legs(place=True, close_steps=1)
                    jaws.clear()
                    for _ in range(len(legs)):
                        jaws.append(jaw_open)
                    for k in range(3):
                        tip_goal[k] = plan.tip_goal[k]
                    tilt_pick = plan.tilt
                    tilt_place = plan.tilt
                    n_draws = 1
                    print("  seed", seed_used)
                    print_plan(plan, legs)
                    var warn = plan_warnings(plan)
                    if warn.byte_length() > 0:
                        print(warn, end="")
                    ok_all = warn.byte_length() == 0
                else:
                    var tp = plan_task(
                        task_n, env, planner, body_names, qs, q5, sc, q_scene, brick_adr, bz,
                        cfg_desk_pick, cfg_desk_release, cfg_bowl_pick, cfg_bowl_release,
                        jaw_open, bowl_jaw, dr_valid, dr_x, dr_y, dr_yaw, seed0 + draw, True,
                    )
                    draw += tp.draws
                    legs = tp.legs.copy()
                    jaws = tp.jaws.copy()
                    for k in range(3):
                        tip_goal[k] = tp.tip_goal[k]
                    tx = tp.tx
                    pick_plan = tp.pick_plan.copy()
                    pick_seed0 = tp.pick_seed0
                    pick_x = tp.pick_x
                    pick_y = tp.pick_y
                    pick_z = tp.pick_z
                    ty = tp.ty
                    place_yaw = tp.place_yaw
                    tilt_pick = tp.tilt_pick
                    tilt_place = tp.tilt_place
                    seed_used = tp.seed
                    n_draws = tp.draws
                    ok_all = tp.ok
                if auto:
                    if ok_all:
                        choice = String("")
                    else:
                        redraws += 1
                        if redraws < 3:
                            choice = String("r")
                        else:
                            print("  no clean plan — fix the scene by hand and press Enter (q = quit)")
                            stdin.discard_pending()
                            choice = String("s") if stdin.line() != "q" else String("q")
                    continue
                if not live:
                    print("  (dry run) Enter = next episode | r = redraw | q = quit")
                else:
                    print(
                        "  Enter = RUN IT" + ("" if ok_all else " (type 'force': see ⚠ above)")
                        + " | r = redraw | s = re-read the scene | q = quit"
                    )
                stdin.discard_pending()
                choice = stdin.line()
                if live and not ok_all and choice == "":
                    print("  refused: type 'force' to run this plan")
                    choice = String("r")
            if choice == "q":
                break
            if choice == "s":
                continue
            if not live:
                ep += 1
                continue

            # ── the episode, on the arm ──────────────────────────────────
            var changed = scene_unchanged(rig.cams[0], cam, frame, sc)
            if changed != "":
                print("  ⚠ the scene changed since it was read (" + changed + ") — re-reading, re-planning")
                continue
            if not rig.armed:
                rig.arm_torque(step_ticks)
                print("  follower torque ON")
            n_run += 1
            rig.trace = String(
                "tick\tleg\tcmd0\tcmd1\tcmd2\tcmd3\tcmd4\tcmd5\tq0\tq1\tq2\tq3\tq4\tq5"
                "\tv0\tv1\tv2\tv3\tv4\tv5\tload0\tload1\tload2\tload3\tload4\tload5\ttip_x\ttip_y\ttip_z\ttip_dist_mm\n"
            )
            for k in range(3):
                rig.tip_goal[k] = tip_goal[k]
            rig.tick_n = 0
            rig.late = 0
            rig.drops = 0
            var close_how = String("-")
            var tip_close = -1.0
            var jaw_after = -1.0
            var aborted = String("")
            var empty_close = False
            var look_n = -1
            var look_x = 0.0
            var look_y = 0.0
            var look_yaw = 0.0
            var look_img = List[UInt8]()
            var look_used = False
            print("  running — press Enter to ABORT")
            stdin.discard_pending()
            var rec_index = -1
            if recording:
                for c in range(len(rig.cams)):
                    _ = rig.cams[c].drain()
                rec_index = rig.writers[0].n_episodes()
                rig.writers[0].begin_episode(task.copy() if task_n == 1 else task2.copy())
                rig.rec_rows = 0
                rig.recording_now = True
            try:
                rig.begin(env)
                var li = -1
                var looked = False
                while li + 1 < len(legs):
                    li += 1
                    if legs[li].name == "hold":
                        continue
                    rig.jaw_open = jaws[li]
                    var rep = rig.run_leg(env, legs[li], settle_steps, leg_settle, stdin)
                    print("   ", rep)
                    if legs[li].name == "pre" and has_wrist and not looked:
                        looked = True
                        var lk = wrist_look(rig, env, afk, wci, wlens, pick_x, pick_y, pick_z, stdin)
                        look_n = lk.n_conf
                        if lk.n_conf > 0:
                            look_x = lk.x
                            look_y = lk.y
                            look_yaw = lk.yaw
                            var corr = sqrt((lk.x - pick_x) ** 2 + (lk.y - pick_y) ** 2) * 1000.0
                            var use = (
                                wrist_correct and cycle and lk.n_conf >= LOOK_MIN_CONF
                                and corr <= LOOK_MAX_CORR_MM
                            )
                            print(
                                "    look    wrist: brick (", fixed(lk.x * 1000.0, 1), ",", fixed(lk.y * 1000.0, 1),
                                ") mm yaw", _deg(lk.yaw), "|", lk.n_conf, "/", lk.n_frames,
                                "confident | vs the plan's brick dx", fixed((lk.x - pick_x) * 1000.0, 1),
                                "dy", fixed((lk.y - pick_y) * 1000.0, 1), "mm, yaw",
                                fixed(_yaw90_deg(lk.yaw, pick_yaw), 1), "deg",
                                "-> CORRECTING" if use else "(LOG ONLY)",
                            )
                            if use:
                                # THE WRIST CORRECTION: the brick where the
                                # wrist camera sees it. The read and the
                                # plan's brick go through the SAME FK camera
                                # pose, so the arm's model error (the sag)
                                # mostly cancels over the 7 cm left. First
                                # the fast re-aim (same posture, ms); if not
                                # clean, a full re-plan from the same seed;
                                # else the plan as it was.
                                var t_re = perf_counter_ns()
                                var in_bowl_pick = task_n == 2
                                var jaw_pk = bowl_jaw if in_bowl_pick else jaw_open
                                var cfg_pk = cfg_bowl_pick.copy() if in_bowl_pick else cfg_desk_pick.copy()
                                var pb2: List[Float64] = [lk.x, lk.y, pick_z]
                                var qb2 = with_brick(qs, brick_adr, lk.x, lk.y, pick_z, lk.yaw)
                                var ra = reaim_pick(env, planner, pick_plan, qb2, lk.x, lk.y, pick_z, lk.yaw, jaw_pk)
                                var new_legs = List[PlanLeg]()
                                var new_jaws = List[Float64]()
                                var new_tip = List[Float64]()
                                var how = String("")
                                if ra[1]:
                                    new_legs = splice_pick(ra[0], legs, jaws, jaw_pk, in_bowl_pick, new_jaws)
                                    new_tip = ra[0].tip_goal.copy()
                                    how = "re-aimed (same posture; " + ra[2] + ")"
                                else:
                                    print("    look    the re-aim is not clean (" + ra[2] + ") — re-planning")
                                    var pk2 = (
                                        plan_bowl_pick(env, planner, body_names, qb2, pb2, lk.yaw, q5, cfg_pk, pick_seed0)
                                        if in_bowl_pick
                                        else plan_clean(env, planner, body_names, qb2, pb2, lk.yaw, q5, cfg_pk, pick_seed0, PICK_DRAWS)
                                    )
                                    if pk2.ok:
                                        new_legs = splice_pick(pk2.plan, legs, jaws, jaw_pk, in_bowl_pick, new_jaws)
                                        new_tip = pk2.plan.tip_goal.copy()
                                        how = "re-planned (tilt " + _deg(pk2.plan.tilt) + ")"
                                if len(new_legs) > 0:
                                    # the move to the new pre-grasp is short:
                                    # its ramp scaled to it, not 40 steps
                                    var dq = 0.0
                                    for k in range(N_ARM):
                                        dq = max(dq, abs(new_legs[0].q[k] - rig.q_ref[k]))
                                    var n_pre = max(REAIM_MIN_STEPS, min(N_PRE_FULL, Int(dq * 180.0 / pi / REAIM_DEG_PER_STEP) + 1))
                                    new_legs[0] = PlanLeg("pre", new_legs[0].q.copy(), new_legs[0].grip_open, n_pre, False, False)
                                    legs = new_legs^
                                    jaws = new_jaws^
                                    for k in range(3):
                                        rig.tip_goal[k] = new_tip[k]
                                    li = -1
                                    look_used = True
                                    print(
                                        "    look    " + how + " | tip_goal moved",
                                        fixed(sqrt((new_tip[0] - tip_goal[0]) ** 2 + (new_tip[1] - tip_goal[1]) ** 2) * 1000.0, 1),
                                        "mm | computed in", fixed(Float64(perf_counter_ns() - t_re) * 1e-6, 0),
                                        "ms | to the new pre-grasp in", n_pre, "steps",
                                    )
                                else:
                                    print("    look    no clean re-aim or re-plan at the wrist's brick — keeping the plan")
                        else:
                            print("    look    wrist: no confident brick in", lk.n_frames, "frames (LOG ONLY)")
                        if len(lk.frame) > 0:
                            var img = lk.frame.copy()
                            var lcam = RigCamera(wlens, lk.cam_pos, lk.cam_rot)
                            _draw_outline(img, lcam, PrismModel.tower_brick(), pick_z - 0.0125, pick_x, pick_y, pick_yaw, 0, 255, 0)
                            if lk.n_conf > 0:
                                _draw_outline(img, lcam, PrismModel.tower_brick(), pick_z - 0.0125, lk.x, lk.y, lk.yaw, 255, 0, 0)
                            look_img = img^
                        if li < 0:
                            # the legs were replaced by the wrist correction:
                            # nothing below may read the old index (cycle run
                            # 4 crashed on legs[-1] here — a hard abort, no
                            # shutdown, the arm left energised)
                            continue
                    if legs[li].close_on_tip:
                        tip_close = rig.tip_dist_mm()
                        close_how = String("trigger") if rep.find("TRIGGERED") >= 0 else String("timeout")
                    if legs[li].name == "close":
                        jaw_after = rig.q[5]
                        if jaw_after < JAW_EMPTY_RAD:
                            # holding the brick the jaw stalls at 0.116-0.121;
                            # empty it closes to -0.149 (first cycle run)
                            print(
                                "  ⚠ EMPTY CLOSE (jaw", fixed(jaw_after, 3),
                                "rad < " + fixed(JAW_EMPTY_RAD, 2) + "): the pick missed —"
                                " opening, backing out along the approach, going home",
                            )
                            empty_close = True
                            # OPEN FIRST, then back out with the jaw open: a
                            # half-closed jaw may hold the bowl's wall, and a
                            # ramp home with it closed dragged the bowl to
                            # the arm's base (first cycle run, ep 5)
                            var none = List[Float64]()
                            print("   ", rig.run_leg(env, PlanLeg("open", none.copy(), True, N_OPEN, False, False), settle_steps, leg_settle, stdin))
                            for bj in range(li - 1, -1, -1):
                                if legs[bj].name == "descend" or legs[bj].name == "pre":
                                    rig.jaw_open = jaws[bj]
                                    var back = PlanLeg("back", legs[bj].q.copy(), True, legs[bj].steps, False, False)
                                    print("   ", rig.run_leg(env, back, settle_steps, leg_settle, stdin))
                            break
            except e:
                aborted = String(e)
                print("  ⚠", aborted)
            # home, out of the camera's way (also after an abort)
            try:
                if stdin.has_input():
                    _ = stdin.line()
                rig.ramp_to(env, q_home, N_HOME, String("home"), stdin)
                if recording:
                    rig.leg = String("rest")
                    for _ in range(N_REST_HOLD):
                        rig.tick(env, stdin)
            except e:
                print("  ⚠ the ramp home failed:", e, "— ending the run")
                raise Error(String(e))
            rig.recording_now = False
            var outcome: String
            var ex: Float64
            var ey: Float64
            var emm: Float64
            if task_n == 1:
                var oc = read_outcome(rig.cams[0], cam, frame, sc, in_bowl_mm)
                outcome = oc[0]
                ex = oc[1]
                ey = oc[2]
                emm = oc[3]
                print(
                    "  -> brick", fixed(ex * 1000.0, 1), fixed(ey * 1000.0, 1), "mm, from the bowl",
                    fixed(emm, 1), "mm | bowl moved", fixed(oc[4], 1), "mm",
                )
            else:
                var op = read_place_outcome(rig.cams[0], cam, frame, sc, tx, ty)
                outcome = op[0]
                ex = op[1]
                ey = op[2]
                emm = op[3]
                print(
                    "  -> brick", fixed(ex * 1000.0, 1), fixed(ey * 1000.0, 1), "mm,",
                    fixed(emm, 1), "mm from the spot",
                )
            if aborted != "":
                outcome = String("ABORTED")
            save_png(out_dir + "/ep" + String(ep + 1) + "_end.png", frame, 640, 480, 3)
            if len(look_img) > 0:
                save_png(out_dir + "/ep" + String(ep + 1) + "_wrist_look.png", look_img, 640, 480, 3)
            if outcome == "SUCCESS":
                n_ok += 1
            buckets += " " + String(task_n) + ":" + outcome
            print(
                "  -> TASK", task_n, outcome, "(empty close)" if empty_close else "",
                "| late ticks", rig.late, "| dropped", rig.drops,
            )
            # the dead reckoning for the next task 2
            if cycle and task_n == 1:
                dr_valid = outcome == "SUCCESS"
                if dr_valid:
                    dr_x = tx
                    dr_y = ty
                    dr_yaw = place_yaw
            elif cycle and outcome != "MISSED":
                # a MISSED pick left the brick in the bowl, most likely where
                # it was (the next read checks the camera agrees)
                dr_valid = False
            with open(out_dir + "/ep" + String(ep + 1) + ".tsv", "w") as fh:
                fh.write(rig.trace)
            var kept = String("-")
            if recording:
                # the camera's verdict decides, the operator may flip it
                # (not in --auto): a rejected episode stays in the files and
                # the importer skips it (`record.mojo`'s discard)
                var keep = outcome == "SUCCESS"
                if not auto:
                    stdin.discard_pending()
                    print(
                        "  dataset episode", rec_index, "(" + String(rig.rec_rows), "frames):",
                        "KEEP" if keep else "REJECT", "— Enter = agree | k = keep | r = reject",
                    )
                    var v = stdin.line()
                    if v == "k":
                        keep = True
                    elif v == "r":
                        keep = False
                rig.writers[0].end_episode()
                if not keep:
                    _ = reject_episode(dataset_dir, rec_index)
                    n_rejected += 1
                else:
                    n_kept += 1
                kept = String("kept") if keep else String("rejected")
                print("  ", kept, "->", dataset_dir, "(episode", String(rec_index) + ")")
            summary += (
                String(ep + 1) + "\t" + String(task_n) + "\t" + String(seed_used) + "\t"
                + fixed(sc.brick_x, 4) + "\t" + fixed(sc.brick_y, 4) + "\t" + _deg(sc.brick_yaw) + "\t"
                + fixed(sc.bowl_x, 4) + "\t" + fixed(sc.bowl_y, 4) + "\t" + String(Int(sc.in_bowl)) + "\t"
                + fixed(tx, 4) + "\t" + fixed(ty, 4) + "\t" + _deg(tilt_pick) + "\t" + _deg(tilt_place) + "\t"
                + String(n_draws) + "\t" + close_how + "\t" + fixed(tip_close, 1) + "\t" + fixed(jaw_after, 3)
                + "\t" + outcome + "\t" + fixed(ex, 4) + "\t" + fixed(ey, 4) + "\t" + fixed(emm, 1) + "\t"
                + String(rig.late) + "\t" + String(rig.drops) + "\t" + kept
                + "\t" + fixed(pick_x, 4) + "\t" + fixed(pick_y, 4) + "\t" + String(look_n)
                + "\t" + fixed(look_x, 4) + "\t" + fixed(look_y, 4) + "\t" + _deg(look_yaw)
                + "\t" + String(Int(look_used)) + "\n"
            )
            with open(out_dir + "/episodes.tsv", "w") as fh:
                fh.write(summary)
            ep += 1
    finally:
        var released = return_and_release(rig.arm, start_pose, rig.armed, True, stdin, interactive)
        if not released:
            print("⚠ the follower is STILL ENERGISED — run `pixi run soarm-torque-off` once it is safe")
        if recording:
            try:
                if rig.recording_now:
                    # the run died inside an episode: keep the files whole,
                    # and the episode out of the dataset
                    var idx = rig.writers[0].n_episodes()
                    rig.recording_now = False
                    if rig.rec_rows > 0:
                        rig.writers[0].end_episode()
                        _ = reject_episode(dataset_dir, idx)
                        n_rejected += 1
                if n_kept + n_rejected > 0:
                    print("\nwriting the dataset ...")
                    rig.writers[0].close()
            except e:
                print("⚠ closing the dataset failed:", e)
        for c in range(len(rig.cams)):
            try:
                rig.cams[c].stop()
            except:
                pass

    print("=" * 70)
    if live:
        print("  episodes run", n_run, "| SUCCESS", n_ok, "| outcomes:" + buckets)
    if recording:
        print("  dataset", dataset_dir, ": kept", n_kept, "| rejected", n_rejected)
    print("  out:", out_dir)
    print("=" * 70)
