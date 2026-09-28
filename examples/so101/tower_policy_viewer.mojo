"""Watch a trained `so101_tower` SAC policy — ImGui sidebar, props placed, free camera.

    pixi run build-imgui                                                     # ONCE
    pixi run mojo run -I . examples/so101/tower_policy_viewer.mojo
    pixi run mojo run -I . examples/so101/tower_policy_viewer.mojo so101_tower_lift_brick
    pixi run mojo run -I . examples/so101/tower_policy_viewer.mojo so101_tower_lift_brick \\
        --ckpt <run_id>

The interactive counterpart of `examples/tasks/sac_tower_gpu.mojo`'s final
eval: the same checkpoint, the same family env on the CPU, the same greedy
action, with the viewer's sidebar around it — pause / single-step, screenshot,
recording, the observation plot, and a camera the mouse owns. It is for
answering "WHERE does the policy get stuck": the status line prints the two
distances the reward is built from, in millimetres, every step.

## WHAT IT SHOWS

The first argument is the TASK (default `so101_tower_lift_brick`), the second
(or `--ckpt`) an explicit checkpoint: a RUN ID (its `checkpoints/last.ckpt`) or
a .ckpt path. With no checkpoint named, every
`projects/so101-tower/runs/*<task>*/checkpoints/last.ckpt` is listed, newest
first, in the sidebar's variant combo — so runs can be compared live.

The status line, from the observation's own goal words
(`So101TowerConfig.OBS_GOAL_BASE`: gripper, subject − gripper, target −
subject): `reach` is the distance from the PINCH CENTRE (`grasp_center`, the
family's gripper site since 2026-09-19) to the goal's subject, against the
2 cm `REACH_RADIUS`; `goal` is the target's offset from the subject — for
`Above(brick, desk, m)` the vertical shortfall to the required height, for
`Near(brick, bowl, r)` the distance to the bowl. A policy parked ON the brick
reads `reach` at 20-30 mm and `goal` unchanging; one that grasps reads `reach`
under 10 mm and `goal` moving.

## ⚠⚠ WHAT MAKES A FAMILY DIFFERENT FROM THE REACH VIEWER

1. **The props are placed after every reset** through `ViewerState.reset_qpos`
   (`tasks/posed_reset.posed_qpos`, the host sampler at seed 0): without it
   the free slots sit 50 m up in `qpos0` and the policy sees a scene it never
   trained on.
2. **The task's tape, mask, init and shaping words go into `meta`** through
   `ViewerState.reset_meta_*` (`tasks/posed_reset.task_meta_words`): the
   observation hook computes the goal words FROM the tape and zeroes inactive
   slots BY the mask, and `Phyics3dEnv.reset` writes neither. A checkpoint
   viewed without them gets zeros where it expects its goal — it looks broken
   while being fine.
3. **The reward on the CPU env is zero by design** (`So101FamilyConfig.
   compute_reward_and_done_cpu`); the family's reward is a GPU kernel. So the
   sparkline is flat here, and the status line's distances are the signal.
4. **No smoothing variants.** The reach viewer's combo walks an EMA the real
   deploy applies; nothing deploys this policy yet, so the combo lists
   checkpoints instead. The greedy checkbox is `tanh(mu)`, as the eval.

⚠ `pixi run build-imgui` first; RUN FROM THE REPO ROOT; ON THE LAPTOP. CPU
physics + CPU policy (a 256x256 MLP beside one arm at 31 Hz). Cameras: `1`
wrist, `2` overhead, free by default.
"""

from std.os import listdir
from std.pathlib import Path
from std.random import seed
from std.sys import argv

from noeira.nn.constants import DT
from noeira.deep_agents.data.any_replay import AnyReplay
from noeira.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from noeira.tasks.sac_family_policy import (
    SacFamilyPolicy, HIDDEN as FAMILY_HIDDEN, POLICY_BATCH, POLICY_CAP,
)
from noeira.deep_agents.training.blocks import ReplaySampleStep
from noeira.envs.dm_control.viewer_core import (
    ActionSource, DRIVE_POLICY, ViewerState, run_view, task_index,
)
from noeira.render.imgui import imgui_shim_available
from noeira.render.renderer3d import Renderer3D
from noeira.utils.fmt import fixed

from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.core.run import resolve_checkpoint

comptime SEED: Int = 0
comptime FAMILY = "so101_tower"
comptime RUNS_DIR = "projects/so101-tower/runs"
comptime DEFAULT_TASK = "so101_tower_lift_brick"

comptime OBS_DIM = So101TowerModel.OBS_DIM
"""49 = qpos(20) + qvel(18) + active(2) + goal words(9). ⚠ FROM THE MODEL DEF,
the number the trainer's env allocated."""
comptime ACT_DIM = 6
comptime HIDDEN = FAMILY_HIDDEN
"""The family SAC widths, from `noeira/tasks/sac_family_policy.mojo` — the
driver trains with the same constants, so a `--policy` checkpoint loads."""
comptime BATCH = POLICY_BATCH
comptime CAP = POLICY_CAP
comptime ACTION_SCALE = 1.0
"""⚠ MUST MATCH the driver's `ACTION_SCALE` (1.0): the family's action space
is NORMALIZED [-1, 1] per joint."""
comptime GOAL_BASE = So101TowerConfig.OBS_GOAL_BASE
comptime REACH_RADIUS_MM = So101TowerConfig.REACH_RADIUS * 1000.0


def task_names() -> List[String]:
    var t = List[String]()
    t.append(String("so101_tower_lift_brick"))
    t.append(String("so101_tower_cube_in_bowl"))
    t.append(String("so101_tower_reach_clear"))
    return t^


def find_checkpoints(task: String) raises -> List[String]:
    """Every `RUNS_DIR/*<task>*/checkpoints/last.ckpt`, NEWEST FIRST (run
    directories start with the date, so reverse lexical order is newest
    first)."""
    var out = List[String]()
    if not Path(RUNS_DIR).exists():
        return out^
    var names = List[String]()
    for e in listdir(RUNS_DIR):
        var n = String(e)
        if task in n:
            var p = String(RUNS_DIR) + "/" + n + "/checkpoints/last.ckpt"
            if Path(p).exists():
                names.append(p)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[j] > names[i]:
                names[i], names[j] = names[j], names[i]
    return names^


struct TowerPolicy(ActionSource, Movable):
    """One or more SAC checkpoints of the tower family, driving the arm.

    Built through the `SAC[...]` preset so the parameter layout is the
    driver's by construction (`StochasticActor` + `Sequential` are what
    `SACActorNet` / `SACCriticNet` expand to); a checkpoint is the gate.

    ⚠ `learning_starts=0` IS LOAD-BEARING for the non-greedy path: below it
    `select_action` is the uniform-random warmup branch.
    """

    var agent: SacFamilyPolicy[OBS_DIM, ACT_DIM]
    var paths: List[String]
    var labels: List[String]
    var current: Int
    var step_idx: Int
    var reach_mm: Float64
    var goal_mm: Float64
    var goal_dz_mm: Float64

    def __init__(out self, task: String, explicit: String) raises:
        self.agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, CAP, HIDDEN](
            action_scale=ACTION_SCALE, learning_starts=0,
        )
        self.paths = List[String]()
        self.labels = List[String]()
        self.current = -1
        self.step_idx = 1
        self.reach_mm = -1.0
        self.goal_mm = -1.0
        self.goal_dz_mm = 0.0
        if explicit:
            # taken AS GIVEN, not probed
            self.paths.append(explicit.copy())
            self.labels.append(
                explicit.copy() if Path(explicit).exists()
                else explicit + " (missing)"
            )
        else:
            var found = find_checkpoints(task)
            for i in range(len(found)):
                self.paths.append(found[i].copy())
                # the run id is the directory name: RUNS_DIR/<run>/checkpoints/last.ckpt
                var rel = String(found[i][byte = String(RUNS_DIR).byte_length() + 1 :])
                var cut = rel.find("/")
                self.labels.append(
                    String(rel[byte = 0 : cut]) if cut > 0 else rel
                )
        if len(self.paths) == 0:
            print("  ⚠ no checkpoint for", task, "under", RUNS_DIR)
            print("    Train one:  pixi run -e nvidia mojo run -I ."
                  " examples/tasks/sac_tower_gpu.mojo", task)
        else:
            print("  checkpoints (newest first):")
            for i in range(len(self.labels)):
                print("    ", i, self.labels[i])

    def obs_dim(self) -> Int:
        return OBS_DIM

    def act_dim(self) -> Int:
        return ACT_DIM

    def variant_labels(self) -> List[String]:
        return self.labels.copy()

    def choose(mut self, i: Int) raises:
        """Load checkpoint `i`, or raise WITHOUT SIDE EFFECTS."""
        if i < 0 or i >= len(self.paths):
            raise Error("variant out of range: " + String(i))
        self.agent.load(self.paths[i])
        self.current = i
        print("  loaded", self.paths[i])

    def status(self) -> String:
        if self.current < 0:
            return String("no checkpoint loaded — drive modes still work")
        if self.reach_mm < 0.0:
            return self.labels[self.current] + String(" loaded")
        var mark = (
            String(" ✓") if self.reach_mm <= REACH_RADIUS_MM else String("")
        )
        return (
            String("reach ") + fixed(self.reach_mm, 1) + " / "
            + fixed(REACH_RADIUS_MM, 0) + " mm" + mark
            + "   goal " + fixed(self.goal_mm, 1) + " mm (dz "
            + fixed(self.goal_dz_mm, 1) + ")   " + self.labels[self.current]
        )

    def act(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        greedy: Bool,
    ) raises:
        if self.current < 0:
            for j in range(ACT_DIM):
                action_out[j] = Scalar[DT](0)
            return
        # the goal words: [GOAL_BASE..+3) gripper, [+3..+6) subject - gripper,
        # [+6..+9) target - subject — `task_hooks.write_task_obs`'s layout
        var rx = Float64(obs[GOAL_BASE + 3])
        var ry = Float64(obs[GOAL_BASE + 4])
        var rz = Float64(obs[GOAL_BASE + 5])
        self.reach_mm = ((rx * rx + ry * ry + rz * rz) ** 0.5) * 1000.0
        var gx = Float64(obs[GOAL_BASE + 6])
        var gy = Float64(obs[GOAL_BASE + 7])
        var gz = Float64(obs[GOAL_BASE + 8])
        self.goal_mm = ((gx * gx + gy * gy + gz * gz) ** 0.5) * 1000.0
        self.goal_dz_mm = gz * 1000.0
        if greedy:
            self.agent.select_greedy_action(obs, action_out)
        else:
            self.agent.select_action(obs, action_out, self.step_idx)
            self.step_idx += 1


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return
    var args = argv()
    var positional = List[String]()
    var explicit = String("")
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--ckpt" and ai + 1 < len(args):
            explicit = String(args[ai + 1])
            ai += 2
            continue
        positional.append(a)
        ai += 1
    var task = positional[0].copy() if len(positional) > 0 else String(DEFAULT_TASK)
    if not explicit and len(positional) > 1:
        explicit = positional[1].copy()
    if explicit:
        # a RUN ID -> its `checkpoints/last.ckpt` (what the task layer's SAC
        # driver writes); a file is taken as is; anything else raises here.
        explicit = resolve_checkpoint(explicit, String("last"))
    var ti = task_index(task, task_names())
    if ti < 0:
        print("unknown task:", task, "— this viewer registers:")
        for n in task_names():
            print("   ", n)
        return

    print("=" * 66)
    print("so101_tower —", task, "— trained SAC policy, ImGui viewer")
    print("=" * 66)
    var pol_src = TowerPolicy(task, explicit^)
    print(
        "  status line: reach = pinch centre -> subject (mm, vs the 2 cm"
        "\n  radius); goal = target - subject (mm, dz = vertical part)."
        "\n  the CPU env pays NO reward (the family's reward is a GPU kernel),"
        "\n  so the sparkline is flat: read the distances."
        "\n  camera: free (mouse). `1` wrist_cam, `2` overhead_cam."
    )
    print("=" * 66)

    var domains = List[String]()
    domains.append(String(FAMILY))
    var td = List[Int]()
    for _ in range(len(task_names())):
        td.append(0)
    var st = ViewerState(
        ti, DRIVE_POLICY, 1.0, task_names(), domains^, td^
    )
    st.policy_variant = 0
    st.free_camera = True
    var pol = Pointer(to=pol_src).as_unsafe_any_origin()
    while not st.quit:
        var name = task_names()[st.task]
        st.reset_qpos = posed_qpos[So101TowerPlacement](
            name, String(FAMILY), So101TowerConfig.SLOT_RADIUS
        )
        var mw = task_meta_words(
            name, String(FAMILY), So101TowerConfig.SHAPE_W_GOAL,
            So101TowerConfig.SHAPE_W_REACH, So101TowerConfig.GOAL_MARGIN,
            So101TowerConfig.REACH_MARGIN,
        )
        st.reset_meta_idx = mw[0].copy()
        st.reset_meta_val = mw[1].copy()
        run_view[So101TowerModel, So101TowerConfig, TowerPolicy](name, st, pol)
    _ = pol_src

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
