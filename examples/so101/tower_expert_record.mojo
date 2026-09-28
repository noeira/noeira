"""A scripted EXPERT for the tower tasks — waypoints + inverse kinematics on the
CPU env, headless, writing `.demo` files the SAC driver loads with `--demos`.

    pixi run mojo run -I . examples/so101/tower_expert_record.mojo                       # lift_brick, 20 episodes
    pixi run mojo run -I . examples/so101/tower_expert_record.mojo so101_tower_cube_in_bowl --episodes 50
    pixi run mojo run -I . examples/so101/tower_expert_record.mojo --episodes 30 --noise 0.05 --seed 7
    pixi run python tools/demo/demo_stats.py projects/so101-tower/demos/*.demo --episodes

## ⚠ THE PLAN LIVES IN THE LIBRARY: `tasks/so101_tower_expert_plan.mojo`

The posture draw, the IK and `--clear-plan`'s collision pass are
`TowerGraspPlanner` there (a real-arm executor plans with the same code); this
file is the SIM EXECUTOR — stepping, triggers, noise, the sag integral,
recording, DAgger. `--print-plan` prints each episode's plan as the library's
legs. Moved on 25 Sep with the same bytes for the default, the human posture
and lift_brick (checked); `--clear-plan` now reads the brick's yaw once rather
than after each redraw (the env's poses lag a substep and the collision pass
refreshed them — 0.3 deg, 64 -> 63/150).

## WHY AN EXPERT

Teleoperating the sim from a screen is hard for a reason no fix removes: a
human reads depth from parallax and touch, and gets neither. After the jaw
and contact fixes (see the bake and `gen_tower_props.py`) Denis could lift the
cube "a few times" and it still "did not feel 100% reality". HIL-SERL needs
10-20 clean successes to start from; a script that reaches, pinches, lifts
and places gives hundreds at physics speed, on randomised placements, in the
trainer's own observation and reward. Every success in the file is a real
pinch under the family's contact model — the rung (reward > 1.5) proves it.

## THE PLAN, PER EPISODE

Placements are the task's own (`posed_reset.posed_qpos` at `--seed + k`, the
host sampler over the family's regions). Site targets for `grasp_center`,
the fixed finger pointing DOWN and the pinch axis aligned RADIALLY (the cube
is axis-aligned at reset and the jaw closes across it either way):

    pre-grasp   brick + 8 cm            open
    grasp       brick + 3 cm            open, then CLOSE over 1 s
    lift        brick + 15 cm           closed      -> lift_brick holds here
    carry       bowl  + 16 cm           closed      (cube_in_bowl only)
    place       bowl  + 10 cm           closed, then OPEN
    retreat     bowl  + 16 cm           open        -> Near holds

Each leg interpolates joint targets linearly over its step budget, so the
actions are the normalised joint targets the batched env takes. `--noise`
adds per-step Gaussian noise to those targets WHILE THE JAW IS OPEN — flat
on the approach, tapered to zero along the descent (the last noisy target is
where the jaw closes: 11 mm off at 0.02 flat, 10/20); the lift and the hold
are noise-free, because noise on the lift jiggles the pinch.
Leave the noise ON for a training file: a policy fitted to noiseless ramps
reads its phase off the velocity words, and a 0.004 action error puts the
stiff servo 0.6 rad/s off the ramp, where the fit is unconstrained — the
closed loop then diverges at step 2 (`tower_policy_probe.mojo`, 2026-09-20).
The recorded action is the EXECUTED (noisy) one; its conditional mean is the
clean ramp, which is what the L1 fit converges to.

Success is the family's own predicate held `HOLD_STEPS` consecutive steps
(the recorder's rule); the episode ends there. A failed episode (the script
ran out) is dropped unless `--keep-failures`.

## THE IK

Damped least squares on the env's OWN forward kinematics (`set_state` +
`site_xpos` / `xquat`), finite-difference Jacobian over the five arm joints,
rows = site position (3) + the finger direction's horizontal components (2,
weighted) + the pinch-axis yaw error (1, weighted). Restarts from five seed
poses with the pan pre-aimed, over a relaxing tilt weight, keeping the best
position error. Measured in MuJoCo on the same scene before the port: a
strict vertical finger is unreachable beyond ~33 cm radius and the pinch
works tilted up to ~20 deg, so relaxing the weight is what took the
prototype from 3 of 10 to 10 of 10 placements in the bowl.

DAGGER (`--policy CKPT`): the checkpoint drives, greedy, until the jaw is
within `--handover-mm` (36, `CLOSE_REACH_MM`) of the brick with the arm settled, or for
`--policy-steps` (140); the expert then closes and lifts FROM THE POLICY'S
OWN STATE — a short descent to the grasp pose from wherever the jaw is, the
close, the lift, the hold — and those rows carry the INTERVENED flag. Five
BC runs and a pure-BC run parked at the arrival because no recorded row
says "close from here" for the states the policy reaches on its own; these
are those rows, HIL-SERL's human intervention with the expert as the human.
First smoke (checkpoint 69b67456, stiff jaws): handover in 7 of 12, lifts
in 11 of 12, 1113 of 2113 rows intervened.

DAGGER FROM A RECORDED STUDENT (`--handover-from FILE.demo`): the vision
student cannot run in this CPU recorder (it needs the device tracer), so its
episodes are recorded by `tower_act_eval.mojo --record-demo` and handed over
HERE, offline. Per recorded episode the handover row is the first row at or
after `HANDOVER_MIN_STEPS` with the jaw within `--handover-mm` of the brick,
MOVING OR NOT, else the student's closest row (`_handover_row` says why this
is not the `--policy` rule), the sim is put in that row's full state
(`obs_at` with its qpos + qvel words — brick and bowl included, so the
placement comes with it), and the expert runs from there exactly as after a
live handover. Only the expert's rows are written, all INTERVENED: the
student's own actions are not labels. Each kept episode starts at the
student's arrival — the "close from HERE" rows the student lacks.

HUMAN POSTURE (`--posture human`): per episode the IK targets a drawn TILT
(the finger leaning outward, `--tilt-range`, default 10..55 deg) and a drawn
PINCH from radial (`--pinch-range`, 35..85), snapped to the brick's nearest
face normal; wrist_roll is seeded positive — the operator's measured grasp
(`tools/soarm/grasp_posture.py`: tilt median 35, pinch 72, roll 77), where
the default expert grasps vertical and radial. A tilted grasp needs its own
mechanics, all active only in this mode:
  - the IK aims the FINGERTIPS (`TIP_REACH` past `grasp_center`);
  - the approach runs ALONG THE FINGER AXIS (`APPROACH_D`, through
    `APPROACH_WAYPOINTS` IK solutions), not vertically;
  - the close fires on the tips' distance to their grasp point
    (`--tip-close-mm`), not on `grasp_center`'s height;
  - integral action on the arm's tracking error (`SAG_KI`) takes out the
    shoulder's sag in the loaded poses;
  - the jaw opens to `HUMAN_JAW_OPEN` (0.6 rad) and the grasp is
    `HUMAN_Z_GRASP` (15 mm) — both overridable.
Measured, cube_in_bowl, folded start, 60 episodes at seed 21000: 30/60
(tilt < 20 deg 9/11, 20-30 11/13, 30-40 4/10, > 40 6/26); the realised tilt
matches the operator's (p5 / median / p95 13 / 34 / 59 against 9 / 35 / 60).
The history, 20 episodes each: vertical approach 2/20, tips aimed 4/20,
along the finger 14/20 at jaw 0.6 (5/20 at 0.9). ⚠ The pinch is MIRRORED
(-65 median against +72) while the cube is axis-aligned: the operator's
Duplo lies at arbitrary yaw; the task's opt-in brick yaw draw (DR session)
is what lets the snap reproduce it. The default (`expert`) draws nothing
and is byte-identical to before (20/20 at seed 11000, same bytes).

⚠ THE IK SETS THE ARM'S qpos TO EVALUATE FK AND RESTORES THE STATE AFTER.
It never steps physics. The props' qpos are untouched.
"""

from std.math import sqrt, atan2, cos, sin, log, pi, floor
from std.random import seed as seed_rng, random_float64
from std.sys import argv
from std.pathlib import Path

from noeira.nn.constants import DT
from noeira.core.cont_action import ContAction
from noeira.core.run import epoch_seconds, iso8601_utc
from noeira.io.proc import quote_arg, run_capture
from noeira.deep_agents.demos.ctrl_range import CtrlRange
from noeira.deep_agents.demos.file import DemoSet, read_demo_file
from noeira.deep_agents.demos.recorder import EpisodeRecorder, Handover
from noeira.deep_agents.data.any_replay import AnyReplay
from noeira.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from noeira.tasks.sac_family_policy import (
    SacFamilyPolicy, HIDDEN as FAMILY_HIDDEN, POLICY_BATCH, POLICY_CAP,
)
from noeira.deep_agents.training.blocks import ReplaySampleStep
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.math3d import Quat, Vec3
from max.gpu.host import DeviceContext
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig, So101TowerTeleopConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.host_reward import family_reward_host
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST,
)
from noeira.physics3d.collision.broadphase_sap import detect_contacts_auto
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.so101_tower_expert_plan import (
    TowerExpertEnv, TowerGraspPlanner, TowerGraspPlan, ACT, N_ARM, NQ, GS, N_VIA,
    GRIPPER_BODY, TIP_REACH, HUMAN_JAW_OPEN, HUMAN_Z_GRASP, CLEAR_PLAN_TILT,
    Z_GRASP, N_PRE, N_DESCEND, N_CLOSE, N_LIFT, N_CARRY, N_PLACE, N_OPEN,
    N_RETREAT, N_HOLD_MAX, PLACE_DRAWS, BOWL_RELEASE_JAW, BRICK_QADR, BOWL_QADR,
    plan_clean, pick_place_legs, with_brick, bowl_release_cfg,
    brick_rest_in_bowl_z, PlanLeg,
)
from noeira.tasks.spec import load_family
from noeira.utils.fmt import fixed

comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime DEFAULT_TASK = "so101_tower_lift_brick"
comptime DEMO_DIR = "projects/so101-tower/demos"
comptime CFG = So101TowerTeleopConfig
comptime E = TowerExpertEnv
"""The planner's env (`tasks/so101_tower_expert_plan.mojo`): the teleop config,
float64, host."""
comptime CLEAR_PLAN_PINCH_OFFSET_MM: Float64 = 12.0
"""`--clear-plan`'s aim along the pinch axis (`TowerGraspPlanner.pinch_offset_m`)
unless `--pinch-offset-mm` is given. Measured, 300 draws (seeds 61000 + 62000,
real-layout regions): 0 mm 140, 6 mm 177, 9 mm 171, 12 mm 179, 16 mm 164,
19 mm 47/150 (the fixed finger lands on the brick). At 0 every grasp is the
moving jaw sweeping the brick 18 mm across the desk into the fixed finger; the
sim's brick-desk friction (1.5 — MuJoCo takes the larger of the two geoms')
tips or jams it (slips 36 -> 18 of 149 at 12 mm).

⚠ AFTER dbd873e15 (the desk's own friction 0.4, the gripper's measured map)
the sweep works, as it does on the rig: 0 mm 200, 6 mm 197, 12 mm 209 / 300
— within noise. 12 stays (the best measured); the offset is no longer what
carries the rate."""
comptime NV = So101TowerModel.NV
comptime NB = So101TowerModel.NBODY
comptime TIP_CLOSE_MM_DEFAULT: Float64 = 8.0
"""A tilted grasp closes when the tips are this close to their grasp point and
the arm has settled (`--tip-close-mm`)."""
comptime SAG_KI: Float64 = 0.15
"""Integral gain on the arm's tracking error, per step (`Expert.integral`)."""
comptime SAG_MAX: Float64 = 0.35
"""The bias's clamp (rad, 20 deg) — above the worst sag measured (16 deg)."""
comptime GB = So101TowerConfig.OBS_GOAL_BASE
"""The goal words: obs[GB+3..GB+5] is the jaw-to-brick vector (the reach)."""

# ── DAgger: the policy drives to its own arrival, the expert takes over ──
comptime HIDDEN = FAMILY_HIDDEN
"""The family SAC widths, from `noeira/tasks/sac_family_policy.mojo` — the
driver trains with the same constants, so a `--policy` checkpoint loads."""
comptime BATCH = POLICY_BATCH
comptime CAP = POLICY_CAP
comptime Agent = SacFamilyPolicy[E.OBS_DIM, ACT]
comptime N_DESCEND_HANDOVER = 20
"""The expert's descent after a handover: the policy already brought the
jaw near the brick, so a short ramp to the grasp pose from wherever it is."""
comptime JAW_OPEN: Float64 = 0.9
"""The jaw's OPEN target (rad) on the approach — `--jaw-open`; the range is
-0.17..1.75. The moving tip swings DOWN in an arc as the jaw closes (its
lowest corner 80 mm above the desk at 1.75, ~40 at 0.9, 20 at 0.6, the
brick's top at 27): from wide open the close is a long swing whose tip
lands on the brick's top edge and whose success depends on the exact
arrival (a close at the first settled row: 7/40; at the ramp's end 30/40).
Half open, the close is a short swing and grasps 30/40 from the first
settled row — so the close can be state-triggered (`Z_CLOSE_ABOVE_MM`)."""
comptime HANDOVER_MIN_STEPS = 20
comptime CLOSE_REACH_MM: Float64 = 36.0
"""The DAgger handover's reach (obs words GB+3..5): measured on 300 z-0.01
demos, the reach at the grasp pose was 24.7-35.5 mm (median 24.9; the
spread is the lateral error over placements)."""
comptime Z_CLOSE_ABOVE_MM: Float64 = 27.0
"""The close is STATE-TRIGGERED: the descent ends and the jaw closes at the
first row where the gripper site is within this HEIGHT above the brick
centre (the reach word GB+5, which is brick − site, so it reads −25 at the
grasp) and the arm has settled (`SETTLED_VEL`). Measured on 300 z-0.01
demos: the ramp descent overshoots to −23.1 mm and settles at −23.6..−24.8
over its last 7 rows (p10 of placements: −31.7, the far ones the IK cannot
reach lower — those fall back to the ramp's end). Those 7 rows were the
defect: "settled at the grasp pose, jaw open, label OPEN" next to ONE row
of the same state labelled CLOSED, and the fitted policy's jaw command at
that state was open (the probe's teacher-forced error at the close row:
0.34, the full jump). ⚠ NOT THE COMMANDED HEIGHT: a test against the IK
target within 4 mm never fired (IK error + gravity sag), and a reach-norm
test (< 36 mm) fired 10 mm too high (1/40 clean, 0/12 DAgger)."""
comptime SETTLED_VEL: Float64 = 0.15
"""rad/s, every arm joint: settled (the median over the 7 rows before the
recorded close was 0.04-0.10; the descent runs at 0.39)."""
comptime HOLD_STEPS: Int = 31
"""The goal held this many consecutive steps = success (the recorder's rule)."""
comptime FB_MIN_STEP: Float64 = 0.02
"""Feedback demonstrator: the slowest a joint is driven, rad per control step
(0.6 rad/s) — the pull-back speed of a joint that noise displaced."""
comptime FB_VEL: Float64 = 0.05
"""Feedback demonstrator: the arm has settled when every joint is slower than
this, rad/s."""

# The legs' step budgets at 31.25 Hz.
comptime N_RETURN_REST = 60
"""`--return-rest`: steps to fold from the retreat pose back to the start
(~1.9 s; the real fold takes 1-3 s)."""
comptime N_REST_HOLD = 15
"""Steps held at rest after the fold (0.5 s of the real ~3 s): enough to
show "stay folded", short of teaching the idle."""



# ── the episode ──────────────────────────────────────────────────────────


struct Expert(Movable):
    var planner: TowerGraspPlanner
    """The plan half (`tasks/so101_tower_expert_plan.mojo`): FK/IK, the drawn
    posture, the collision pass. This struct is the EXECUTOR."""
    var ctrl: CtrlRange
    var rec: EpisodeRecorder
    """The episode bookkeeping (`deep_agents/demos`): rows, the HOLD_STEPS
    success rule, keep / drop, the file rewritten per kept episode."""
    var noise: Float64
    var feedback: Bool
    var flat_noise: Bool
    var intervening: Bool
    """Rows recorded while True carry the INTERVENED flag (the expert
    driving after a `--policy` handover — HIL-SERL's human, scripted)."""
    var human_posture: Bool
    """`--posture human` (mirrors `planner.posture.human`): the executor's
    own switches — the integral action, the tip trigger's defaults."""
    var integral: Bool
    """Integral action on the arm joints' tracking error (on with `--posture
    human`): the tilted poses load the shoulder, and the position servo rests
    8-16 deg of lift short of its target, the gripper 10-40 mm above the
    grasp height, so the close fired at the ramp's end over the brick (2/20).
    Each step `sag_bias += SAG_KI * (reference - q)`, and the command is the
    ramp's reference plus the bias — what a human on the leader arm does by
    eye."""
    var sag_bias: List[Float64]
    var return_rest: Bool
    """`--return-rest`: after a success, fold the arm back to the episode's
    own start pose and hold it — every real episode ends that way (50/50,
    then ~3 s at rest), and without it a student meets the fold back and the
    rest on real frames with no row that shows them (the DR session's phase
    split of H seed 1's real error: rest + return = 54 % of the rows, roll
    error 46-48 deg there). No idle START is recorded: it could freeze a
    policy at rest on the arm."""
    var q_home: List[Float64]
    var tip_trigger: Bool
    """This episode's close fires on the TIP distance (a tilted grasp), not on
    `grasp_center`'s height, which assumes a vertical finger."""
    var tip_goal: List[Float64]
    var tip_close_mm: Float64
    var q_ref: List[Float64]
    """The ramp's reference WITHOUT the bias — what `hold` holds, so the bias
    is not counted twice."""
    var close_steps: Int
    var close_above_mm: Float64  # `--close-above-mm`: the close's height trigger; 0 = off (close at the ramp's end)
    var jaw_open: Float64     # `--jaw-open`: the jaw's OPEN target (rad) on the approach; the moving tip
                              # swings down in an arc as it closes (80 mm up at 1.75, ~40 at 0.9, 20 at 0.6),
                              # so a half-open approach makes the close a short swing
    var frame_skip: Int
    var timestep: Float64
    var low_place: Bool
    """`--place low`: the place is the rig executor's — a second grasp plan
    for a virtual brick at the bowl's centre, run backwards, released LOW
    (the fingers `RELEASE_GAP_M` above the floor, the jaw opened only to
    `BOWL_RELEASE_JAW`), entering and leaving `ABOVE_M` over the approach
    (`pick_place_legs`). `drop` (the old place): carried 16 cm above the bowl,
    lowered to 10 cm, opened — the brick falls and bounces off the wall."""
    var bowl_dz: Float64
    """The brick's resting height in the bowl above the bowl body's origin
    (m), measured once in the sim (`brick_rest_in_bowl_z`)."""
    var body_names: List[String]
    var place_seed: Int
    var dump_dir: String
    """`--dump-close DIR`: per episode, the state at the close and every
    control step of the close and the lift — `DIR/ep_<k>.txt`, the input of
    `tools/soarm/replay_close_mujoco.py` (the same steps in MuJoCo 3.12)."""
    var dump_rows: List[String]
    var dumping: Bool
    var print_plan: Bool
    """`--print-plan`: print each episode's plan as the library's legs, in
    degrees — what a real-arm executor receives. ⚠ Its place legs are planned
    from the bowl's START pose, and that extra IK refreshes the env's FK (its
    poses lag a substep), so demo bytes are NOT those of a run without it."""
    var q_cmd: List[Float64]
    var obs: List[Scalar[DT]]
    var prev_obs: List[Scalar[DT]]
    var act_l: List[Float64]
    var steps: Int
    var rung_rows: Int

    def __init__(
        out self, mut env: E, ref body_names: List[String], noise: Float64,
        feedback: Bool = False, out_path: String = "",
        keep_failures: Bool = False,
    ) raises:
        self.planner = TowerGraspPlanner(env, body_names)
        self.ctrl = CtrlRange(
            self.planner.arm.lo.copy(), self.planner.arm.hi.copy()
        )
        self.rec = EpisodeRecorder(
            E.OBS_DIM, ACT, out_path, keep_failures, HOLD_STEPS
        )
        self.noise = noise
        self.feedback = feedback
        self.flat_noise = False
        self.intervening = False
        self.human_posture = False
        self.integral = False
        self.sag_bias = List[Float64](length=N_ARM, fill=0.0)
        self.return_rest = False
        self.q_home = List[Float64](length=N_ARM, fill=0.0)
        self.tip_trigger = False
        self.tip_goal = List[Float64](length=3, fill=0.0)
        self.tip_close_mm = TIP_CLOSE_MM_DEFAULT
        self.q_ref = List[Float64](length=N_ARM, fill=0.0)
        self.close_steps = N_CLOSE
        self.close_above_mm = Z_CLOSE_ABOVE_MM
        self.jaw_open = JAW_OPEN
        self.frame_skip = CFG.FRAME_SKIP
        self.timestep = So101TowerModel.TIMESTEP
        self.print_plan = False
        self.low_place = False
        self.bowl_dz = 0.0
        self.body_names = body_names.copy()
        self.place_seed = 0
        self.dump_dir = String("")
        self.dump_rows = List[String]()
        self.dumping = False
        self.q_cmd = List[Float64](length=ACT, fill=0.0)
        self.obs = List[Scalar[DT]](length=E.OBS_DIM, fill=Scalar[DT](0))
        self.prev_obs = List[Scalar[DT]](length=E.OBS_DIM, fill=Scalar[DT](0))
        self.act_l = List[Float64](length=ACT, fill=0.0)
        self.steps = 0
        self.rung_rows = 0

    def _normalized(self, i: Int, q: Float64) -> Float64:
        return self.ctrl.normalize(i, q)

    def _apply(mut self, mut env: E) raises -> Bool:
        """Step the env with `self.act_l`, pay the family's reward, record the
        transition (flagged INTERVENED while `self.intervening`). Returns
        True when the goal has held `HOLD_STEPS` steps."""
        var action = ContAction[ACT]()
        for i in range(ACT):
            action.data[i] = self.act_l[i]
        for i in range(E.OBS_DIM):
            self.prev_obs[i] = self.obs[i]
        var out = env.step(action)
        for i in range(E.OBS_DIM):
            self.obs[i] = Scalar[DT](out[0].data[i])
        if self.dumping:
            # the ctrl the env applied (`actuation.mojo`: affine onto the
            # ctrlrange, clamped), then the qpos it reached
            var row = String("step")
            for i in range(ACT):
                var lo = self.planner.arm.lo[i]
                var hi = self.planner.arm.hi[i]
                var c = lo + (self.act_l[i] + 1.0) * 0.5 * (hi - lo)
                c = min(max(c, lo), hi)
                row += " " + String(c)
            row += " |"
            for i in range(NQ):
                row += " " + String(Float64(env.d.qpos.data[i]))
            self.dump_rows.append(row)
        self.steps += 1
        var rd = family_reward_host[CFG, DType.float64, E.MD, ACT](
            env.d, env.mf, self.act_l, self.steps, self.frame_skip,
            self.timestep,
        )
        var r = Float64(rd[0])
        if r > 1.5:
            self.rung_rows += 1
        return self.rec.record(
            self.prev_obs, self.act_l, r, self.obs, rd[1], self.intervening
        )

    def tip_dist_mm(self, mut env: E) -> Float64:
        """The fingertip point (`grasp_center` + `TIP_REACH` along the finger)
        to `tip_goal`, from the env's current FK."""
        var o = GRIPPER_BODY * 4
        var qw = Quat(
            Float64(env.d.xquat.data[o + 3]), Float64(env.d.xquat.data[o]),
            Float64(env.d.xquat.data[o + 1]), Float64(env.d.xquat.data[o + 2]),
        )
        var f = qw.rotate_vec(Vec3(0.0, 0.0, -1.0))
        var dx = Float64(env.d.site_xpos.data[GS * 3]) + TIP_REACH * f.x - self.tip_goal[0]
        var dy = Float64(env.d.site_xpos.data[GS * 3 + 1]) + TIP_REACH * f.y - self.tip_goal[1]
        var dz = Float64(env.d.site_xpos.data[GS * 3 + 2]) + TIP_REACH * f.z - self.tip_goal[2]
        return sqrt(dx * dx + dy * dy + dz * dz) * 1000.0

    def reach_mm(self) -> Float64:
        var x = Float64(self.obs[GB + 3])
        var y = Float64(self.obs[GB + 4])
        var z = Float64(self.obs[GB + 5])
        return sqrt(x * x + y * y + z * z) * 1000.0

    def policy_approach(
        mut self, mut env: E, mut agent: Agent, handover_mm: Float64,
        max_steps: Int,
    ) raises -> Tuple[Bool, Bool]:
        """DAgger's first half: the CHECKPOINT drives (greedy) until the jaw
        is within `handover_mm` of the brick with the arm settled, or
        `max_steps` have passed. Rows are recorded unflagged. Returns
        (handed over at the arrival, episode already done)."""
        var a32 = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
        var h = Handover(HANDOVER_MIN_STEPS, max_steps)
        for k in range(h.max_steps):
            agent.select_greedy_action(self.obs, a32)
            for i in range(ACT):
                self.act_l[i] = Float64(a32[i])
            if self._apply(env):
                return (False, True)
            if h.arrived_at(k, self.reach_mm() < handover_mm and self.settled()):
                return (True, False)
        return (False, False)

    def settled(self) -> Bool:
        """Every joint — the jaw included — slower than `SETTLED_VEL`."""
        for i in range(ACT):
            if abs(Float64(self.obs[NQ + i])) > SETTLED_VEL:
                return False
        return True

    def step_to(
        mut self, mut env: E, ref q_target: List[Float64], grip_open: Bool,
        n_steps: Int, taper_noise: Bool = False, until_held: Bool = False,
        close_on_height: Bool = False, past_success: Bool = False,
    ) raises -> Bool:
        """Drive the joints to `q_target` and the gripper open/closed;
        record each transition. Returns True when the goal held
        `HOLD_STEPS` steps (the episode is over).

        ⚠ TWO DEMONSTRATORS. The default interpolates the COMMAND from where
        it was to the target over exactly `n_steps` (ramps) — a function of
        time, not of the state: a policy fitted to it to L1 0.004 still
        parked, because its own rollout arrives at the grasp pose a few mm
        off the recorded arrival and no recorded state says "close from
        here" (`tower_policy_probe.mojo`, run dd8d4a64: the gripper word sat
        1.8 rad from the recording from step 90 on, reward 1.233 for ever).
        `--feedback` commands `q_now + clamp(q_target - q_now, ±d)` every
        step — a STATE-FEEDBACK controller whose action is a function of the
        observation and that moves on when the command has reached the
        target and the arm has settled. ⚠ NOT YET THE BETTER DEMONSTRATOR:
        17/20 clean against the ramps' 18/20 on the same seeds, and 9/20 at
        noise 0.02 against 16/20 — the arm jitters through the arrival test
        and the phases run to their cap. Opt-in until that is understood.
        `--flat-noise` keeps the descent's noise flat instead of tapered:
        the arrivals it records are PERTURBED, and the successes among them
        are the "close from here" states the ramps never wrote.
        `d` is the ramp's own speed (the phase's distance over `n_steps`),
        floored at `FB_MIN_STEP` so a joint that should not move is still
        pulled back when noise moves it."""
        var q_start = List[Float64]()
        for i in range(ACT):
            q_start.append(self.q_cmd[i])
        var g_target = self.jaw_open if grip_open else self.planner.arm.lo[5]
        var dmax = List[Float64]()
        for i in range(N_ARM):
            var qi = Float64(env.d.qpos.data[i])
            var d = abs(q_target[i] - qi) / Float64(n_steps)
            dmax.append(d if d > FB_MIN_STEP else FB_MIN_STEP)
        var g0 = Float64(env.d.qpos.data[5])
        var dg = abs(g_target - g0) / Float64(n_steps)
        if dg < FB_MIN_STEP:
            dg = FB_MIN_STEP
        var max_steps = 2 * n_steps if self.feedback and not until_held else n_steps
        var settled = 0
        for k in range(max_steps):
            var a = Float64(k + 1) / Float64(n_steps)
            if a > 1.0:
                a = 1.0
            if self.feedback:
                for i in range(N_ARM):
                    var qi = Float64(env.d.qpos.data[i])
                    var e = q_target[i] - qi
                    if e > dmax[i]:
                        e = dmax[i]
                    if e < -dmax[i]:
                        e = -dmax[i]
                    self.q_cmd[i] = qi + e
                # ⚠ THE GRIPPER IS RATE-LIMITED ON ITS COMMAND, NOT ON ITS
                # POSITION: closed on the cube it stalls above `lo`, and a
                # command of `g_now - d` squeezes with kp·d only — the brick
                # slipped out on the lift (6/20). The command walks to `lo`
                # and stays there, as the ramps did; the squeeze is kp·(g - lo).
                var eg = g_target - self.q_cmd[5]
                if eg > dg:
                    eg = dg
                if eg < -dg:
                    eg = -dg
                self.q_cmd[5] = self.q_cmd[5] + eg
            else:
                for i in range(N_ARM):
                    var ref_i = q_start[i] + (q_target[i] - q_start[i]) * a
                    self.q_ref[i] = ref_i
                    self.q_cmd[i] = ref_i + self.sag_bias[i] if self.integral else ref_i
                self.q_cmd[5] = q_start[5] + (g_target - q_start[5]) * a
            if close_on_height:
                # the descent's jaw target is a STEP to `jaw_open`: after a
                # DAgger handover the policy's jaw is wide open, and a ramp
                # from there is 20 rows of a time-indexed label at a settled
                # arm — the defect this trigger removes. A step makes the
                # half-close one state-indexed event (jaw at `jaw_open`,
                # stopped: `settled` includes the jaw) and the close the next.
                self.q_cmd[5] = g_target
            for i in range(ACT):
                var v = self._normalized(i, self.q_cmd[i])
                # ⚠ WHILE THE JAW IS OPEN ONLY — the approach and the descent.
                # Noise on the lift jiggles the pinch (10/20 at 0.01); noise on
                # the approach is the coverage the closed loop NEEDS: a policy
                # fitted to noiseless ramps reads its phase off the velocity
                # words, and one 0.004 action error puts the stiff servo 0.6
                # rad/s off the ramp, where the fit says nothing (probe, 20 Sep).
                # ⚠ AND TAPERED TO ZERO ALONG THE DESCENT: the last noisy
                # target is where the jaw sits when it closes, and 0.02 of the
                # range is 11 mm at the cube — 10/20 with the noise flat.
                if self.noise > 0.0 and grip_open:
                    var sigma = self.noise * (1.0 - a) if (taper_noise and not self.flat_noise) else self.noise
                    var u1 = random_float64(1e-12, 1.0)
                    var u2 = random_float64(0.0, 1.0)
                    v += sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
                    if v > 1.0:
                        v = 1.0
                    if v < -1.0:
                        v = -1.0
                self.act_l[i] = v
            if self._apply(env) and not past_success:
                return True
            if self.integral and not self.feedback:
                for i in range(N_ARM):
                    var b = self.sag_bias[i] + SAG_KI * (
                        self.q_ref[i] - Float64(env.d.qpos.data[i])
                    )
                    if b > SAG_MAX:
                        b = SAG_MAX
                    if b < -SAG_MAX:
                        b = -SAG_MAX
                    self.sag_bias[i] = b
            # the descent: hand over to the close at the first settled row
            # at the grasp height — see `Z_CLOSE_ABOVE_MM`
            if (
                close_on_height and self.tip_trigger and k + 1 >= 3
                and self.tip_dist_mm(env) < self.tip_close_mm
                and self.settled()
                and abs(self.q_cmd[5] - g_target) < 1e-9
            ):
                return False
            if (
                close_on_height and not self.tip_trigger
                and self.close_above_mm > 0.0 and k + 1 >= 3
                and Float64(self.obs[GB + 5]) * 1000.0 > -self.close_above_mm
                and self.settled()
                # and the jaw's command has reached its open target: after a
                # DAgger handover the policy's jaw is FULLY open and the
                # descent ramps it to `jaw_open`; a close from wide open is
                # the long swing that misses (2/12 without this)
                and abs(self.q_cmd[5] - g_target) < 1e-9
            ):
                return False
            if self.feedback and not until_held:
                # arrived: the COMMAND of every arm joint is at its target
                # (the clamp is inactive — under gravity the joint itself
                # rests a sag below, and a position test never passes), the
                # arm has stopped moving, and the gripper's command is at its
                # target.
                var arrived = True
                for i in range(N_ARM):
                    if abs(self.q_cmd[i] - q_target[i]) > 1e-9:
                        arrived = False
                    if abs(Float64(env.d.qvel.data[i])) > FB_VEL:
                        arrived = False
                if abs(self.q_cmd[5] - g_target) > 1e-9:
                    arrived = False
                if arrived:
                    settled += 1
                else:
                    settled = 0
                if settled >= 2 and k + 1 >= n_steps // 2:
                    break
        return False

    def hold(
        mut self, mut env: E, grip_open: Bool, n_steps: Int,
        until_held: Bool = False, past_success: Bool = False,
    ) raises -> Bool:
        var q = List[Float64]()
        for i in range(N_ARM):
            q.append(self.q_ref[i] if self.integral else self.q_cmd[i])
        return self.step_to(
            env, q, grip_open, n_steps, until_held=until_held,
            past_success=past_success,
        )


def _body_pos(mut env: E, b: Int) -> List[Float64]:
    var p = List[Float64]()
    for k in range(3):
        p.append(Float64(env.d.xpos.data[b * 3 + k]))
    return p^


def _body_yaw(mut env: E, b: Int) -> Float64:
    """Body `b`'s yaw about world z (rad), from its world quaternion."""
    var o = b * 4
    # xquat is stored (x, y, z, w)
    var x = Float64(env.d.xquat.data[o])
    var y = Float64(env.d.xquat.data[o + 1])
    var z = Float64(env.d.xquat.data[o + 2])
    var w = Float64(env.d.xquat.data[o + 3])
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _row_reach_mm(ref d: DemoSet, r: Int) -> Float64:
    """`Expert.reach_mm` read off a recorded row's observation."""
    var b = r * d.obs_dim + GB
    var x = Float64(d.obs[b + 3])
    var y = Float64(d.obs[b + 4])
    var z = Float64(d.obs[b + 5])
    return sqrt(x * x + y * y + z * z) * 1000.0


def _row_settled(ref d: DemoSet, r: Int) -> Bool:
    """`Expert.settled` read off a recorded row's observation."""
    for i in range(ACT):
        if abs(Float64(d.obs[r * d.obs_dim + NQ + i])) > SETTLED_VEL:
            return False
    return True


def _handover_row(
    ref d: DemoSet, e: Int, handover_mm: Float64, settled: Bool = False
) -> Tuple[Int, Bool]:
    """The row of episode `e` the expert takes over at: the FIRST row at or
    after `HANDOVER_MIN_STEPS` with the jaw within `handover_mm` of the brick
    (an arrival), else the row where the student came CLOSEST.

    ⚠ NOT THE `--policy` RULE, AND ON PURPOSE. That rule also waits for the
    arm to settle and falls back to a step cap. The first vision student
    (ACT f4ef105f, 128 recorded episodes) arrives MOVING and closes at once:
    its failed episodes reach a median 32 mm at step ~62 with the jaw still
    open (0.91 rad), close, miss, lift away and hover 130-170 mm off — only
    29 of 92 ever had a settled row under 36 mm, so the settled rule fell
    back to the cap at row 140, AFTER the miss, and the expert labelled a
    fresh approach from far away instead of the grasp the student gets
    wrong. The first row under the radius is before the student's close.

    `settled` (`--handover-settled`): the first row under the radius WITH
    every joint slower than `SETTLED_VEL`, and NO fallback — an episode with
    no such row returns row -1 and is skipped. The arm, not the rule, is the
    experiment here: the moving-arrival rows are the leading suspect for the
    first DAgger round's result (ACT sees joint positions and images, no
    velocities, so a moving and a settled arrival can look alike while their
    labels differ), and these rows are the unambiguous half."""
    var start = d.ep_start[e]
    var n = d.ep_len[e]
    if settled:
        for k in range(HANDOVER_MIN_STEPS, n):
            if _row_reach_mm(d, start + k) < handover_mm and _row_settled(d, start + k):
                return (start + k, True)
        return (-1, False)
    for k in range(HANDOVER_MIN_STEPS, n):
        if _row_reach_mm(d, start + k) < handover_mm:
            return (start + k, True)
    var best = start
    var best_r = _row_reach_mm(d, start)
    for k in range(1, n):
        var rk = _row_reach_mm(d, start + k)
        if rk < best_r:
            best_r = rk
            best = start + k
    return (best, False)


def run_episode(
    mut env: E, mut ex: Expert, brick: Int, bowl: Int, place: Bool,
    ep: Int, verbose: Bool,
    agent: Optional[Pointer[Agent, MutAnyOrigin]] = None,
    handover_mm: Float64 = CLOSE_REACH_MM, policy_steps: Int = 140,
    handed_in: Int = -1,
) raises -> Bool:
    """One scripted pick (and place). Returns success.

    With `agent` (DAgger, `--policy`): the checkpoint drives first —
    `policy_approach` — and the expert takes over FROM THE POLICY'S OWN
    STATE, its rows flagged INTERVENED: the descent from wherever the jaw
    is (a short ramp if the policy arrived, the full pre + descent if it
    did not), then the close, the lift and the hold. Those rows are the
    ones no expert-only file holds — "close from HERE", where here is a
    state the policy reaches on its own (five runs parked on exactly that)."""
    ex.rec.begin()
    ex.steps = 0
    ex.rung_rows = 0
    ex.intervening = False
    var handed = False
    var done = False
    var policy_steps_used = 0
    if agent:
        var res = ex.policy_approach(env, agent.value()[], handover_mm, policy_steps)
        handed = res[0]
        done = res[1]
        policy_steps_used = ex.steps
        ex.intervening = True
    elif handed_in >= 0:
        # `--handover-from`: the env is already in the student's state
        handed = handed_in == 1
        ex.intervening = True
    for i in range(ACT):
        ex.q_cmd[i] = Float64(env.d.qpos.data[i])
    for i in range(N_ARM):
        ex.q_ref[i] = Float64(env.d.qpos.data[i])
        ex.sag_bias[i] = 0.0
    var from_reset = not agent and handed_in < 0
    if from_reset:
        for i in range(N_ARM):
            ex.q_home[i] = Float64(env.d.qpos.data[i])
    var pb = _body_pos(env, brick)
    var q = List[Float64]()
    for i in range(N_ARM):
        q.append(Float64(env.d.qpos.data[i]))
    # THE PLAN (`tasks/so101_tower_expert_plan.mojo`): posture draw, IK,
    # and with --clear-plan the collision pass; this function executes it
    var pw0 = _body_pos(env, bowl) if place and ex.planner.path_check else List[Float64]()
    var plan = ex.planner.plan_grasp(
        env, pb, _body_yaw(env, brick), q, ex.jaw_open, pw0
    )
    for k in range(3):
        ex.tip_goal[k] = plan.tip_goal[k]
    ex.tip_trigger = plan.close_on_tip
    var q1 = plan.q_pre.copy()
    var q2 = plan.q_grasp.copy()
    var q3 = plan.q_lift.copy()
    var waypoints = plan.waypoints.copy()
    if verbose and ex.human_posture:
        print("  ep", ep, "posture: tilt", fixed(plan.tilt * 180.0 / pi, 1),
              "deg | pinch pair", "tangential" if plan.tangential else "radial",
              "| pinch from radial", fixed((plan.yaw - plan.bearing) * 180.0 / pi, 1), "deg")
    if verbose and ex.planner.clear_plan and len(waypoints) > 0:
        print("  ep", ep, "clear plan: tries", plan.tries, "| raised",
              fixed(plan.raise_mm, 1), "mm | obstacle penetration",
              fixed(plan.pen_mm, 1), "mm")
    if verbose:
        print("  ep", ep, "brick", fixed(pb[0], 3), fixed(pb[1], 3),
              " ik err mm: pre", fixed(plan.e_pre * 1000.0, 1), "grasp",
              fixed(plan.e_grasp * 1000.0, 1), "lift", fixed(plan.e_lift * 1000.0, 1))
    if ex.print_plan:
        var shown = plan.copy()
        ex.planner.plan_place(env, shown, _body_pos(env, bowl))
        print("  ep", ep, "PLAN brick", fixed(pb[0], 3), fixed(pb[1], 3), "yaw",
              fixed(_body_yaw(env, brick) * 180.0 / pi, 1), "| tip_goal",
              fixed(plan.tip_goal[0], 3), fixed(plan.tip_goal[1], 3),
              fixed(plan.tip_goal[2], 3), "| close_on_tip", plan.close_on_tip)
        for leg in shown.legs(place, ex.close_steps):
            var qs = String("")
            for i in range(len(leg.q)):
                qs += " " + fixed(leg.q[i] * 180.0 / pi, 1)
            print("    leg", leg.name, "| steps", leg.steps, "| grip",
                  "open" if leg.grip_open else "closed", "| close_on_tip",
                  leg.close_on_tip, "| until_held", leg.until_held,
                  "| q deg" + (qs if len(leg.q) > 0 else " (hold)"))
    if verbose and agent:
        print("  ep", ep, " policy drove", policy_steps_used, "steps ->",
              "handover at reach " + fixed(ex.reach_mm(), 1) + " mm" if handed
              else "no arrival (cap), the expert does the full approach")
    if not done and not handed and len(plan.q_via) > 0:
        # up and over first (`--via`), then down to the pre-grasp
        done = ex.step_to(env, plan.q_via, True, N_VIA)
    if not done and not handed:
        done = ex.step_to(env, q1, True, N_PRE)
    if not done and len(waypoints) > 0 and not handed:
        # the tilted descent: along the finger, one leg per waypoint, the
        # close armed on the last
        var n_wp = len(waypoints) - 1
        var leg = N_DESCEND // n_wp
        for j in range(1, n_wp + 1):
            if done:
                break
            done = ex.step_to(
                env, waypoints[j], True, leg, taper_noise=True,
                close_on_height=j == n_wp,
            )
    elif not done:
        done = ex.step_to(
            env, q2, True, N_DESCEND_HANDOVER if handed else N_DESCEND,
            taper_noise=True, close_on_height=True,
        )
    if not done:
        # ⚠ THE CLOSE IS A STEP (`--close-steps`, default 1; it was 30,
        # then 15) — see `N_CLOSE` and `Z_GRASP`. A 30-step ramp labels the arrival state — arm settled at the
        # grasp pose, jaw open — with a command 1/30 of the range below
        # open, next to the approach's "open" at the same state: the fitted
        # policy's jaw command there is a hair below open, the jaw does not
        # move, the state does not change, and it waits for ever (probe,
        # runs dd8d4a64 and c40b4a8b: gripper word 1.8 rad off the
        # recording from step 90 on). Measured on 20 placements: 30 steps
        # 18/20, 15 steps 18/20, 8 steps 14/20, 4 steps 0/20, 1 step 13/20 —
        # the fast closes knock the cube. Fifteen doubles the label.
        if verbose and not handed:
            # the close's starting point: how far the tips are from their
            # goal, and whether the descent already pushed the brick
            var pc = _body_pos(env, brick)
            print("  ep", ep, "at close: tip", fixed(ex.tip_dist_mm(env), 1),
                  "mm | brick moved", fixed(sqrt((pc[0] - pb[0]) ** 2
                  + (pc[1] - pb[1]) ** 2) * 1000.0, 1), "mm xy",
                  fixed((pc[2] - pb[2]) * 1000.0, 1), "mm z")
            var qerr = String("")
            for i in range(N_ARM):
                qerr += " " + fixed(
                    (Float64(env.d.qpos.data[i]) - waypoints[len(waypoints) - 1][i]
                     if len(waypoints) > 0 else Float64(env.d.qpos.data[i]) - q2[i])
                    * 180.0 / pi, 1)
            var o = GRIPPER_BODY * 4
            var qw = Quat(
                Float64(env.d.xquat.data[o + 3]), Float64(env.d.xquat.data[o]),
                Float64(env.d.xquat.data[o + 1]), Float64(env.d.xquat.data[o + 2]),
            )
            var f = qw.rotate_vec(Vec3(0.0, 0.0, -1.0))
            var tz = Float64(env.d.site_xpos.data[GS * 3 + 2]) + TIP_REACH * f.z - ex.tip_goal[2]
            var nc = Int(env.d.meta.data[META_IDX_NUM_CONTACTS])
            var pairs = String("")
            for c in range(nc):
                pairs += " " + String(Int(env.d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_A])) + "-" + String(Int(env.d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_B]))
            # the brick in the GRIPPER's frame, from `grasp_center`: x is the
            # pinch axis (fixed finger -> moving jaw), z along the finger
            var gc = List[Float64]()
            for k in range(3):
                gc.append(Float64(env.d.site_xpos.data[GS * 3 + k]))
            var inv = qw.conjugate()
            var rel = inv.rotate_vec(Vec3(pc[0] - gc[0], pc[1] - gc[1], pc[2] - gc[2]))
            print("  ep", ep, "at close: brick in gripper mm x", fixed(rel.x * 1000.0, 1),
                  "y", fixed(rel.y * 1000.0, 1), "z", fixed(rel.z * 1000.0, 1))
            print("  ep", ep, "at close: q - target deg", qerr, "| tip dz",
                  fixed(tz * 1000.0, 1), "mm | ncon", nc, "| bodies", pairs)
        var pbc = _body_pos(env, brick)
        if ex.dump_dir.byte_length() > 0 and not handed:
            ex.dump_rows.clear()
            var r0 = String("qpos")
            for i in range(NQ):
                r0 += " " + String(Float64(env.d.qpos.data[i]))
            var r1 = String("qvel")
            for i in range(NV):
                r1 += " " + String(Float64(env.d.qvel.data[i]))
            ex.dump_rows.append(r0)
            ex.dump_rows.append(r1)
            ex.dumping = True
        done = ex.hold(env, False, ex.close_steps)
        if verbose and not handed:
            var pac = _body_pos(env, brick)
            print("  ep", ep, "during close: brick moved", fixed(sqrt(
                (pac[0] - pbc[0]) ** 2 + (pac[1] - pbc[1]) ** 2) * 1000.0, 1),
                "mm xy", fixed((pac[2] - pbc[2]) * 1000.0, 1), "mm z | jaw",
                fixed(Float64(env.d.qpos.data[5]), 3))
    if not done:
        done = ex.step_to(env, q3, False, N_LIFT)
        if ex.dumping:
            ex.dumping = False
            var txt = String("")
            for r in ex.dump_rows:
                txt += r + "\n"
            with open(ex.dump_dir + "/ep_" + String(ep) + ".txt", "w") as fh:
                fh.write(txt)
        if verbose:
            var pl = _body_pos(env, brick)
            print("  ep", ep, "after lift: brick dz", fixed((pl[2] - pb[2]) * 1000.0, 1),
                  "mm | jaw", fixed(Float64(env.d.qpos.data[5]), 3), "rad")
    if place and not done and ex.low_place:
        # THE RIG'S PLACE (`pick_place_legs`), planned HERE from the bowl
        # where it is after the lift. ⚠ The planner's collision pass works on
        # the env: the live state is saved and restored around it.
        var pw = _body_pos(env, bowl)
        var qsave = List[Float64]()
        for i in range(NQ):
            qsave.append(Float64(env.d.qpos.data[i]))
        var vsave = List[Float64]()
        for i in range(NV):
            vsave.append(Float64(env.d.qvel.data[i]))
        var pc: List[Float64] = [pw[0], pw[1], pw[2] + ex.bowl_dz]
        var qv = with_brick(qsave, BRICK_QADR, pc[0], pc[1], pc[2], plan.yaw)
        var pl = plan_clean(
            env, ex.planner, ex.body_names, qv, pc, plan.yaw, plan.q_lift,
            bowl_release_cfg(), ex.place_seed + ep * 101, PLACE_DRAWS,
        )
        var jaws = List[Float64]()
        var legs = pick_place_legs(
            env, ex.planner, plan, pl.plan, ex.jaw_open, BOWL_RELEASE_JAW,
            False, jaws,
        )
        env.set_state(qsave, vsave)
        if verbose:
            print("  ep", ep, "low place: clean", pl.ok, "| draws", pl.draws,
                  "| tilt", fixed(pl.plan.tilt * 180.0 / pi, 1), "| pen",
                  fixed(pl.plan.pen_mm, 1), "mm")
        # the legs after the pick's lift (the recorder ran those already)
        var after_lift = False
        var jaw_saved = ex.jaw_open
        for k in range(len(legs)):
            if not after_lift:
                if legs[k].name == "lift":
                    after_lift = True
                continue
            if done:
                break
            ex.jaw_open = jaws[k]
            if len(legs[k].q) == 0:
                done = ex.hold(env, legs[k].grip_open, legs[k].steps)
            else:
                done = ex.step_to(env, legs[k].q, legs[k].grip_open, legs[k].steps)
        ex.jaw_open = jaw_saved
    elif place and not done:
        # the place is planned HERE, from the bowl where it is after the
        # lift (a grasp can nudge it) — see the library's header
        var pw = _body_pos(env, bowl)
        ex.planner.plan_place(env, plan, pw)
        var q4 = plan.q_carry.copy()
        var q5 = plan.q_place.copy()
        if verbose and ex.planner.path_report:
            var ps = String("")
            for k in range(len(plan.path_pen_mm)):
                ps += " " + fixed(plan.path_pen_mm[k], 1)
            print("  ep", ep, "path pen mm (pre descent lift carry place):" + ps)
        if verbose:
            print("  ep", ep, "bowl", fixed(pw[0], 3), fixed(pw[1], 3),
                  " ik err mm: carry", fixed(plan.e_carry * 1000.0, 1), "place",
                  fixed(plan.e_place * 1000.0, 1))
        done = ex.step_to(env, q4, False, N_CARRY)
        if not done:
            done = ex.step_to(env, q5, False, N_PLACE)
        if not done:
            done = ex.hold(env, True, N_OPEN)
        if not done:
            done = ex.step_to(env, q4, True, N_RETREAT)
    if not done:
        done = ex.hold(env, not place, N_HOLD_MAX, until_held=True)
    if done and ex.return_rest and from_reset:
        # the fold back, then a short rest: recorded PAST the success test,
        # which already holds here (the brick is in the bowl)
        var home = ex.q_home.copy()
        _ = ex.step_to(env, home, False, N_RETURN_REST, past_success=True)
        _ = ex.hold(env, False, N_REST_HOLD, past_success=True)
    ex.intervening = False
    var brick_z = Float64(env.d.xpos.data[brick * 3 + 2])
    print(
        "  ep", ep, "->", "SUCCESS" if done else "failed", " steps", ex.steps,
        " return", fixed(ex.rec.ret, 1), " rows>1.5 (rung)", ex.rung_rows,
        " brick z", fixed(brick_z, 3),
    )
    return done


def _usage():
    print("usage: tower_expert_record.mojo [task] [--episodes N] [--seed S]"
          " [--noise SIGMA] [--flat-noise] [--close-steps N] [--z-grasp M] [--jaw-open RAD]\n"
          "       [--close-above-mm MM] [--feedback] [--out FILE]\n"
          "       [--policy CKPT [--handover-mm MM] [--policy-steps N]]   # DAgger\n"
          "       [--handover-from STUDENT.demo [--handover-mm MM] [--handover-settled]]"
          "   # DAgger from a recorded (vision) student\n"
          "       [--posture expert|human [--tilt-range LO,HI] [--pinch-range LO,HI]"
          " [--tip-close-mm MM] [--clear-plan [--desk-clear-mm MM] [--pinch-offset-mm MM]"
          " [--desk-jaw RAD] [--via none|auto|always] [--path-check] [--path-report]"
          " [--fallback on|off] [--place low|drop]]]"
          " [--return-rest]\n"
          "       [--print-plan] [--dump-close DIR] [--keep-failures] [--quiet]")


def main() raises:
    var args = argv()
    var task = String(DEFAULT_TASK)
    var n_episodes = 20
    var seed0 = 0
    var noise = 0.0
    var feedback = False
    var flat_noise = False
    var close_steps = N_CLOSE
    var z_grasp = Z_GRASP
    var z_grasp_set = False
    var clear_plan = False
    var print_plan = False
    var dump_dir = String("")
    var desk_jaw = -10.0
    var path_report = False
    var path_check = False
    var via_mode = -1
    var fallback = False
    var place_mode = String("")
    var fallback_set = False
    var pinch_offset_mm = 0.0
    var pinch_offset_set = False
    var desk_clear_mm = 0.0
    var desk_clear_set = False
    var close_above_mm = Z_CLOSE_ABOVE_MM
    var jaw_open = -1.0
    var policy_ckpt = String("")
    var handover_mm = CLOSE_REACH_MM
    var policy_steps = 140
    var handover_from = String("")
    var posture = String("expert")
    var tilt_range = String("10,55")
    var tilt_range_set = False
    var pinch_range = String("35,85")
    var tip_close_mm = TIP_CLOSE_MM_DEFAULT
    var return_rest = False
    var handover_settled = False
    var out_path = String("")
    var keep_failures = False
    var verbose = True
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--episodes" and i + 1 < len(args):
            n_episodes = Int(String(args[i + 1]))
            i += 2
        elif a == "--seed" and i + 1 < len(args):
            seed0 = Int(String(args[i + 1]))
            i += 2
        elif a == "--noise" and i + 1 < len(args):
            noise = Float64(String(args[i + 1]))
            i += 2
        elif a == "--out" and i + 1 < len(args):
            out_path = String(args[i + 1])
            i += 2
        elif a == "--keep-failures":
            keep_failures = True
            i += 1
        elif a == "--quiet":
            verbose = False
            i += 1
        elif a == "--feedback":
            feedback = True
            i += 1
        elif a == "--flat-noise":
            flat_noise = True
            i += 1
        elif a == "--close-steps" and i + 1 < len(args):
            close_steps = Int(String(args[i + 1]))
            i += 2
        elif a == "--pinch-offset-mm" and i + 1 < len(args):
            pinch_offset_mm = Float64(String(args[i + 1]))
            pinch_offset_set = True
            i += 2
        elif a == "--via" and i + 1 < len(args):
            var vm = String(args[i + 1])
            if vm == "none":
                via_mode = 0
            elif vm == "auto":
                via_mode = 1
            elif vm == "always":
                via_mode = 2
            else:
                raise Error("--via is none, auto or always, not " + vm)
            i += 2
        elif a == "--place" and i + 1 < len(args):
            var pm = String(args[i + 1])
            if pm != "low" and pm != "drop":
                raise Error("--place is low or drop, not " + pm)
            place_mode = pm
            i += 2
        elif a == "--fallback" and i + 1 < len(args):
            fallback = String(args[i + 1]) == "on"
            fallback_set = True
            i += 2
        elif a == "--path-check":
            path_check = True
            i += 1
        elif a == "--path-report":
            path_report = True
            i += 1
        elif a == "--desk-jaw" and i + 1 < len(args):
            desk_jaw = Float64(String(args[i + 1]))
            i += 2
        elif a == "--dump-close" and i + 1 < len(args):
            dump_dir = String(args[i + 1])
            i += 2
        elif a == "--print-plan":
            print_plan = True
            i += 1
        elif a == "--clear-plan":
            clear_plan = True
            i += 1
        elif a == "--desk-clear-mm" and i + 1 < len(args):
            desk_clear_mm = Float64(String(args[i + 1]))
            desk_clear_set = True
            i += 2
        elif a == "--z-grasp" and i + 1 < len(args):
            z_grasp = Float64(String(args[i + 1]))
            z_grasp_set = True
            i += 2
        elif a == "--close-above-mm" and i + 1 < len(args):
            close_above_mm = Float64(String(args[i + 1]))
            i += 2
        elif a == "--jaw-open" and i + 1 < len(args):
            jaw_open = Float64(String(args[i + 1]))
            i += 2
        elif a == "--policy" and i + 1 < len(args):
            policy_ckpt = String(args[i + 1])
            i += 2
        elif a == "--handover-mm" and i + 1 < len(args):
            handover_mm = Float64(String(args[i + 1]))
            i += 2
        elif a == "--posture" and i + 1 < len(args):
            posture = String(args[i + 1])
            i += 2
        elif a == "--tilt-range" and i + 1 < len(args):
            tilt_range = String(args[i + 1])
            tilt_range_set = True
            i += 2
        elif a == "--return-rest":
            return_rest = True
            i += 1
        elif a == "--tip-close-mm" and i + 1 < len(args):
            tip_close_mm = Float64(String(args[i + 1]))
            i += 2
        elif a == "--pinch-range" and i + 1 < len(args):
            pinch_range = String(args[i + 1])
            i += 2
        elif a == "--handover-settled":
            handover_settled = True
            i += 1
        elif a == "--handover-from" and i + 1 < len(args):
            handover_from = String(args[i + 1])
            i += 2
        elif a == "--policy-steps" and i + 1 < len(args):
            policy_steps = Int(String(args[i + 1]))
            i += 2
        elif a == "--help" or a == "-h":
            _usage()
            return
        elif a.startswith("--"):
            _usage()
            raise Error("unrecognised argument: " + a)
        else:
            task = a
            i += 1
    # cube_in_bowl and its variants (`_wide`: the real layouts' regions)
    var place = task.startswith("so101_tower_cube_in_bowl")
    if task != "so101_tower_lift_brick" and not place:
        raise Error("the expert knows so101_tower_lift_brick and so101_tower_cube_in_bowl*, not " + task)
    seed_rng(seed0)
    if out_path.byte_length() == 0:
        var stamp = iso8601_utc(epoch_seconds()).replace(":", "-")
        out_path = String(DEMO_DIR) + "/" + stamp + "_" + task + (
            "_dagger.demo" if policy_ckpt.byte_length() > 0
            or handover_from.byte_length() > 0 else "_expert.demo"
        )
    if not Path(DEMO_DIR).exists():
        _ = run_capture(String("mkdir -p ") + quote_arg(String(DEMO_DIR)), 4096)

    print("=" * 66)
    print("so101_tower —", task, "— SCRIPTED EXPERT (waypoints + IK)")
    print("=" * 66)
    print("  episodes", n_episodes, " seed", seed0, " noise", noise,
          " demonstrator", "feedback" if feedback else "ramps", " out", out_path)

    var ctx = DeviceContext()
    var env = E(ctx)
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DType.float64](cw[k])
    var mw = task_meta_words(
        task, String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    var brick = -1
    var bowl = -1
    for b in range(len(fmd.body_names)):
        if String(fmd.body_names[b]) == "brick_brick":
            brick = b
        if String(fmd.body_names[b]) == "bowl_bowl":
            bowl = b
    if brick < 0 or bowl < 0:
        raise Error("brick_brick / bowl_bowl not found in the composed scene")
    if verbose:
        var names = String("  bodies:")
        for b in range(len(fmd.body_names)):
            names += " " + String(b) + "=" + String(fmd.body_names[b])
        print(names)

    var body_names = List[String]()
    for b in range(len(fmd.body_names)):
        body_names.append(String(fmd.body_names[b]))
    var ex = Expert(env, body_names, noise, feedback, out_path, keep_failures)
    ex.flat_noise = flat_noise
    ex.close_steps = close_steps
    ex.planner.z_grasp = z_grasp
    ex.close_above_mm = close_above_mm
    if jaw_open > 0.0:
        ex.jaw_open = jaw_open
    if posture != "expert" and posture != "human":
        raise Error("--posture is expert or human, not " + posture)
    ex.human_posture = posture == "human"
    ex.planner.posture.human = ex.human_posture
    ex.planner.clear_plan = clear_plan
    ex.print_plan = print_plan
    ex.dump_dir = dump_dir
    ex.planner.desk_jaw = desk_jaw
    ex.planner.path_report = path_report
    ex.planner.path_check = path_check
    # `--clear-plan` defaults to `--via auto`: 300 draws on dbd873e15 (desk
    # friction 0.4), none 209, auto 225, always 201 — pushed bricks 26 -> 12
    if via_mode < 0:
        via_mode = 1 if clear_plan else 0
    ex.planner.via_mode = via_mode
    ex.planner.fallback = fallback
    # `--clear-plan` defaults to the rig's LOW place (synced with
    # tower_expert_real's task 1)
    if place_mode == "":
        place_mode = String("low") if clear_plan else String("drop")
    ex.low_place = place_mode == "low" and task.startswith("so101_tower_cube_in_bowl")
    if ex.low_place and not ex.human_posture:
        raise Error("--place low plans a tilted grasp backwards: it needs --posture human")
    ex.place_seed = seed0 * 7 + 1
    if path_check and not clear_plan:
        raise Error("--path-check extends --clear-plan's collision pass: add --clear-plan")
    if clear_plan and not pinch_offset_set:
        pinch_offset_mm = CLEAR_PLAN_PINCH_OFFSET_MM
    ex.planner.pinch_offset_m = pinch_offset_mm / 1000.0
    if desk_clear_set:
        ex.planner.desk_clear_m = desk_clear_mm / 1000.0
    if clear_plan and not ex.human_posture:
        raise Error("--clear-plan plans the TILTED grasp: it needs --posture human")
    ex.integral = ex.human_posture
    if ex.human_posture:
        # the tilted grasp's own defaults (60-episode runs, seed 21000):
        # the jaw half as open as the vertical expert's (the operator's
        # opens 25-41 %; 0.9 rad hits the desk tilted, 5/20 against 14/20)
        # and the tips 5 mm higher (the operator grasps ~1 cm higher)
        if jaw_open <= 0.0:
            ex.jaw_open = HUMAN_JAW_OPEN
        if not z_grasp_set:
            ex.planner.z_grasp = HUMAN_Z_GRASP
    if clear_plan and not tilt_range_set:
        tilt_range = String(CLEAR_PLAN_TILT)
    var tr = tilt_range.split(",")
    if len(tr) != 2:
        raise Error("--tilt-range needs lo,hi in degrees, got " + tilt_range)
    ex.planner.posture.tilt_lo = Float64(String(tr[0])) * pi / 180.0
    ex.planner.posture.tilt_hi = Float64(String(tr[1])) * pi / 180.0
    var prr = pinch_range.split(",")
    if len(prr) != 2:
        raise Error("--pinch-range needs lo,hi in degrees, got " + pinch_range)
    ex.planner.posture.pinch_lo = Float64(String(prr[0])) * pi / 180.0
    ex.planner.posture.pinch_hi = Float64(String(prr[1])) * pi / 180.0
    ex.tip_close_mm = tip_close_mm
    ex.return_rest = return_rest
    if ex.human_posture:
        print("  posture  : HUMAN — tilt ~ U(", tilt_range, ") deg outward, pinch",
              "~ U(", pinch_range, ") deg from radial, snapped to the brick's faces")
    var agent: Agent = SAC["cpu", E.OBS_DIM, ACT, BATCH, CAP, HIDDEN](
        action_scale=1.0, learning_starts=0,
    )
    var agent_ptr = Optional[Pointer[Agent, MutAnyOrigin]](None)
    if policy_ckpt.byte_length() > 0:
        if not Path(policy_ckpt).exists():
            raise Error("--policy: no such checkpoint: " + policy_ckpt)
        agent.load(policy_ckpt)
        agent_ptr = Pointer(to=agent).as_unsafe_any_origin()
        print("  DAgger   : the checkpoint drives to its own arrival"
              " (handover under", handover_mm, "mm settled, cap",
              policy_steps, "steps); the expert closes and lifts from there,"
              " those rows flagged INTERVENED")
    if handover_from.byte_length() > 0:
        if policy_ckpt.byte_length() > 0:
            raise Error("--handover-from and --policy are two DAgger sources; pick one")
        if not Path(handover_from).exists():
            raise Error("--handover-from: no such file " + handover_from)
        var sd = read_demo_file(handover_from)
        if sd.obs_dim != E.OBS_DIM or sd.act_dim != ACT:
            raise Error(handover_from + ": obs " + String(sd.obs_dim) + " act "
                        + String(sd.act_dim) + ", this env is obs "
                        + String(E.OBS_DIM) + " act " + String(ACT))
        var n_src = len(sd.ep_len)
        if handover_settled:
            print("  DAgger   : handing over from", n_src, "recorded student episodes (",
                  handover_from, ") — the first SETTLED row under", handover_mm,
                  "mm, episodes without one SKIPPED; only the expert's rows are"
                  " written, INTERVENED")
        else:
            print("  DAgger   : handing over from", n_src, "recorded student episodes (",
                  handover_from, ") — the first row under", handover_mm, "mm (moving"
                  " or not), else the closest row; only the expert's rows are"
                  " written, INTERVENED")
        var n_skipped = 0
        var n_ok_h = 0
        var n_arrived = 0
        var n_src_ok = 0
        for ep in range(n_src):
            if n_episodes > 0 and ep >= n_episodes:
                break
            if sd.ep_success[ep]:
                n_src_ok += 1
            var hr = _handover_row(sd, ep, handover_mm, handover_settled)
            if hr[0] < 0:
                n_skipped += 1
                continue
            if hr[1]:
                n_arrived += 1
            _ = env.reset()
            for k in range(len(mw[0])):
                env.d.meta.data[mw[0][k]] = Scalar[DType.float64](mw[1][k])
            var qh = List[Float64](length=NQ, fill=0.0)
            var vh = List[Float64](length=NV, fill=0.0)
            for k in range(NQ):
                qh[k] = Float64(sd.obs[hr[0] * sd.obs_dim + k])
            for k in range(NV):
                vh[k] = Float64(sd.obs[hr[0] * sd.obs_dim + NQ + k])
            var sh = env.obs_at(qh, vh)
            for k in range(E.OBS_DIM):
                ex.obs[k] = Scalar[DT](sh.data[k])
            if verbose:
                var src_ok = sd.ep_success[ep]
                var h_row = hr[0] - sd.ep_start[ep]
                var src_len = sd.ep_len[ep]
                var h_reach = _row_reach_mm(sd, hr[0])
                var how = String("(arrival)") if hr[1] else String("(closest)")
                var st = String("succeeded") if src_ok else String("failed")
                print("  ep", ep, " student", st, "| handover at row", h_row,
                      "of", src_len, how, "reach", fixed(h_reach, 1), "mm")
            var ok_h = run_episode(
                env, ex, brick, bowl, place, ep, verbose, None, handover_mm,
                policy_steps, handed_in=1 if hr[1] else 0,
            )
            _ = ex.rec.end(success=ok_h)
            if ok_h:
                n_ok_h += 1
        print("-" * 66)
        print("  student episodes", n_src, "| the student succeeded in", n_src_ok,
              "| arrivals", n_arrived, "| skipped (no settled row)", n_skipped,
              "| the expert completed", n_ok_h, "->", out_path)
        print("  ", ex.rec.demos.summary())
        return

    if ex.low_place:
        # the brick's resting height in the bowl, once (the bowl sits on the
        # same desk in every layout); every episode resets the env after it
        _ = env.reset()
        for k in range(len(mw[0])):
            env.d.meta.data[mw[0][k]] = Scalar[DType.float64](mw[1][k])
        var qz = posed_qpos[So101TowerPlacement](
            task, String(FAMILY), So101TowerConfig.SLOT_RADIUS, UInt64(seed0),
        )
        # ⚠ AT A CLEAR SPOT, NOT THE LAYOUT'S: a bowl drawn near the base
        # (x 0.14 at seed 61000) meets the FOLDED arm, which lifts it during
        # the settle (-10.8 mm measured); the height does not depend on where
        # the bowl stands, only that nothing else touches it
        qz[BOWL_QADR] = 0.30
        qz[BOWL_QADR + 1] = 0.10
        qz[BOWL_QADR + 3] = 1.0
        qz[BOWL_QADR + 4] = 0.0
        qz[BOWL_QADR + 5] = 0.0
        qz[BOWL_QADR + 6] = 0.0
        var zr = brick_rest_in_bowl_z(env, qz, ex.planner.arm.lo, ex.planner.arm.hi)
        # against the bowl's SETTLED height: the placement starts props a
        # little above the desk, and they fall during the settle
        var zb = Float64(env.d.qpos.data[BOWL_QADR + 2])
        ex.bowl_dz = zr - zb
        print("  place    : LOW in the bowl (the rig's) — brick rests",
              fixed(ex.bowl_dz * 1000.0, 1), "mm above the bowl's origin")
    var n_ok = 0
    for ep in range(n_episodes):
        _ = env.reset()
        for k in range(len(mw[0])):
            env.d.meta.data[mw[0][k]] = Scalar[DType.float64](mw[1][k])
        var q0 = posed_qpos[So101TowerPlacement](
            task, String(FAMILY), So101TowerConfig.SLOT_RADIUS,
            UInt64(seed0 + ep),
        )
        var v0 = List[Float64](length=NV, fill=0.0)
        var s0 = env.obs_at(q0, v0)
        for k in range(E.OBS_DIM):
            ex.obs[k] = Scalar[DT](s0.data[k])
        # let the props settle on the desk before reading their poses
        var zero = ContAction[ACT]()
        for k in range(ACT):
            zero.data[k] = ex._normalized(k, Float64(env.d.qpos.data[k]))
        for _ in range(5):
            var o = env.step(zero)
            for k in range(E.OBS_DIM):
                ex.obs[k] = Scalar[DT](o[0].data[k])
        var ok = run_episode(
            env, ex, brick, bowl, place, ep, verbose, agent_ptr, handover_mm,
            policy_steps,
        )
        _ = ex.rec.end(success=ok)
        if ok:
            n_ok += 1
    print("-" * 66)
    print("  ", n_ok, "of", n_episodes, "episodes succeeded ->", out_path)
    print("  ", ex.rec.demos.summary())
