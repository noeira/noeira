"""dm_control `walker` x 3 tasks with the SAC LADDER as a fourth drive mode.

    pixi run build-imgui                                        # ONCE
    pixi run mojo run -I . examples/dm_control/dm_walker_policy_viewer.mojo
    pixi run mojo run -I . examples/dm_control/dm_walker_policy_viewer.mojo walker_run
    pixi run mojo run -I . examples/dm_control/dm_walker_policy_viewer.mojo walker_stand policy
    pixi run mojo run -I . examples/dm_control/dm_walker_policy_viewer.mojo --run <run_id>

argv picks the task that opens first, then the drive mode (zero | random |
sweep | policy) and an action scale; every task and mode stays selectable in the
window. Defaults: `walker_walk`, `policy`. `--run` names the training run whose
`checkpoints/` ladder is probed first.

WHAT IT ADDS OVER `dm_viewer_imgui.mojo`. That viewer answers "is the MODEL
built the way I think" and can only shake the joints. This one adds the fourth
mode the other three cannot: an actual trained policy, picked live from the
ladder `sac_dm_walker_training_gpu.mojo` writes. So the picture goes from "a
walker tumbling under random torque" to "the gait this checkpoint learned", and
the sidebar's +/- buttons walk the ladder rung by rung — which is how the
emergence of that gait is read, and how a ladder that PLATEAUED (the M2 failure
`docs/BFM_ZERO_SHOT_RL.md` §13 measured) becomes visible in seconds rather than
inferred from a return curve.

    stand  MOVE_SPEED 0.0     walk  MOVE_SPEED 1.0     run  MOVE_SPEED 8.0

⚠ THE THREE TASKS SHARE ONE MODEL AND ONE OBSERVATION. Only the REWARD differs,
so any checkpoint drives any of the three envs — the sidebar deliberately lets
the ckpt task and the env task disagree, because "run the walk policy, read the
stand reward" is a real question and the reward sparkline answers it. What you
CANNOT read off this viewer is the training return: the env here is the CPU
float32 physics path, the ladder was trained on the GPU batched path, so treat
small differences as path noise and only large ones as a finding.

⚠ THE CHECKPOINTS ARE NOT IN THE REPO. `CKPT_DIRS` probes `checkpoints/` then
the project root, and a rung that is missing is reported as such in the sidebar
and left un-driven rather than silently driving an uninitialised net. If EVERY
rung is missing, train a ladder first (see the training script) — or check that
`SEGMENT_STEPS` / `N_SEGMENTS` below still match the ones it used.

⚠ CPU PHYSICS + CPU POLICY, on purpose: one env at 60 Hz needs no GPU, and a
256x256 MLP forward pass per frame is free next to the physics step. This must
run on the LAPTOP — it opens an SDL3 window and blocks on it.
"""

from std.pathlib import Path
from std.random import seed
from std.sys import argv
from std.os.path import exists
from noeira.core.project import projects_root
from noeira.io.proc import quote_arg, run_capture

from noeira.nn.constants import DT
from noeira.deep_agents.data.any_replay import AnyReplay
from noeira.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from noeira.deep_agents.training.blocks import ReplaySampleStep
from noeira.envs.dm_control.viewer_core import (
    ActionSource, ViewerState, run_view, task_index, parse_drive, DRIVE_POLICY,
)
from noeira.render.imgui import imgui_shim_available
from noeira.render.renderer3d import Renderer3D

from noeira.envs.dm_control.walker.walker_xml import DMWalkerModel
from noeira.envs.dm_control.walker.walker_config import DMWalkerConfig

comptime SEED: Int = 0

comptime OBS_DIM = DMWalkerModel.OBS_DIM
comptime ACT_DIM = DMWalkerModel.ACTION_DIM
comptime HIDDEN = 256
"""Architecture ONLY — it must match the ladder's. `BATCH`/`CAP` size the replay
buffer this agent never uses, so they stay small; the checkpoint holds no
replay."""
comptime BATCH = 256
comptime CAP = 1000

# ══ LADDER GEOMETRY — MUST MATCH `sac_dm_walker_training_gpu.mojo` ═══════
# Rung filenames are reconstructed as `(k+1) * SEGMENT_STEPS`, exactly as
# `examples/fb/collect_walker_sac.mojo` does it. A mismatch here makes every
# rung report MISSING with the path it tried, so compare that against
# `ls checkpoints/sac_dm_walker_*.ckpt.*` before touching anything else.
comptime SEGMENT_STEPS = 32_000
comptime N_SEGMENTS = 20
comptime CKPT_PREFIX = "sac_dm_walker_"


def ckpt_dirs(run_dir: String) -> List[String]:
    """Where a rung might live, in probe order.

    The training script now writes into its run directory
    (`<run_dir>/checkpoints/`, probed first when `--run` is given); older
    ladders sit in the CWD or were moved into `checkpoints/`. Probing all of
    them means no layout has to be special-cased, and `status()` reports which
    file actually loaded.
    """
    var d = List[String]()
    if run_dir.byte_length() > 0:
        d.append(run_dir + "/checkpoints/")
    d.append(String("checkpoints/"))
    d.append(String(""))
    return d^


def ckpt_tasks() -> List[String]:
    var t = List[String]()
    t.append(String("stand"))
    t.append(String("walk"))
    t.append(String("run"))
    return t^


def _stamped(prefix: String, step: Int) raises -> String:
    """Mirror of the training script's `_stamped` — the two must agree."""
    var s = String(step)
    var pad = String("")
    for _ in range(8 - s.byte_length()):
        pad += "0"
    return prefix + ".ckpt." + pad + s


def _kilo(step: Int) -> String:
    """`640000` -> `640k`. The rung labels have ~90 px of combo to live in."""
    return String(step // 1000) + "k"


# ═══════════════════════════════════════════════════════════════════════════
# the ActionSource — a SAC ladder over the three walker tasks
# ═══════════════════════════════════════════════════════════════════════════


struct WalkerLadder(ActionSource, Movable, Deinitable):
    """The 3 x N_SEGMENTS SAC ladder as one selectable policy.

    ONE agent, reloaded in place. `load` overwrites the actor + twin critics
    from an 872 kB file, so switching rung is a few milliseconds and the net
    architecture is constructed exactly once — rebuilding the agent per rung
    would re-allocate the replay buffer 60 times over a session.

    VARIANTS ARE FLAT AND TASK-MAJOR (stand r1..rN, walk r1..rN, run r1..rN),
    which is what makes the sidebar's +/- buttons step the RUNG within a task —
    the axis worth sweeping — while the combo jumps anywhere.

    ⚠ `learning_starts=0` IS LOAD-BEARING for the non-greedy path.
    `select_action` takes the uniform-random WARMUP branch below that threshold,
    so with the default 1000 the "stochastic" toggle would silently be a
    random-action toggle — the same trap `collect_walker_sac.mojo` documents.
    """

    var agent: SACAgent[
        "cpu",
        ReplaySampleStep[AnyReplay["cpu", OBS_DIM, ACT_DIM, CAP], BATCH],
        SACActorNet[OBS_DIM, ACT_DIM, HIDDEN],
        SACCriticNet[OBS_DIM, ACT_DIM, HIDDEN],
    ]
    var paths: List[String]
    """Resolved path per variant, or "" for a rung no directory holds."""
    var labels: List[String]
    var current: Int
    var loaded: Bool
    var step_idx: Int
    """Monotonic counter for the stochastic path only — `select_action` reads it
    to decide warmup vs policy, and with `learning_starts=0` any positive value
    is the policy."""

    def __init__(out self, run_dir: String) raises:
        self.agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, CAP, HIDDEN](
            action_scale=1.0,
            learning_starts=0,
        )
        self.paths = List[String]()
        self.labels = List[String]()
        self.current = -1
        self.loaded = False
        self.step_idx = 1

        var tasks = ckpt_tasks()
        var dirs = ckpt_dirs(run_dir)
        var n_found = 0
        for t in range(len(tasks)):
            var prefix = String(CKPT_PREFIX) + tasks[t]
            for k in range(N_SEGMENTS):
                var at = (k + 1) * SEGMENT_STEPS
                var name = _stamped(prefix, at)
                var found = String("")
                for d in range(len(dirs)):
                    var cand = dirs[d] + name
                    if Path(cand).exists():
                        found = cand
                        break
                self.paths.append(found)
                # The label carries the MISSING mark, so a hole in the ladder is
                # visible in the combo itself and not only after clicking it.
                var mark = String("") if found else String(" (missing)")
                self.labels.append(
                    tasks[t] + " " + _kilo(at) + mark
                )
                if found:
                    n_found += 1
        print(
            "  ladder:", n_found, "of", len(self.paths),
            "rungs found (", len(tasks), "tasks x", N_SEGMENTS, ")",
        )
        if n_found == 0:
            print("  ⚠ NO rung found. Looked for e.g.")
            print("     checkpoints/" + _stamped(
                String(CKPT_PREFIX) + String("walk"), SEGMENT_STEPS
            ))
            print("    Train a ladder first:")
            print(
                "     pixi run -e nvidia mojo run -I ."
                " examples/dm_control/sac_dm_walker_training_gpu.mojo"
            )

    # ─── ActionSource ───────────────────────────────────────────────────

    def obs_dim(self) -> Int:
        return OBS_DIM

    def act_dim(self) -> Int:
        return ACT_DIM

    def variant_labels(self) -> List[String]:
        return self.labels.copy()

    def choose(mut self, i: Int) raises:
        """Load variant `i`, or raise WITHOUT SIDE EFFECTS.

        ⚠ ALL-OR-NOTHING ON PURPOSE. Marking the new rung selected and then
        failing to load it would leave the sidebar naming one rung, the status
        line naming another and zero torque driving the walker — three
        statements about the same thing, none of them the whole truth. Failing
        clean means the previously loaded rung keeps driving and the only new
        information is the printed reason; the combo already labels a missing
        rung "(missing)" before it is ever clicked.
        """
        if i < 0 or i >= len(self.paths):
            raise Error("variant out of range: " + String(i))
        if not self.paths[i]:
            raise Error("rung not on disk: " + self.labels[i])
        self.agent.load(self.paths[i])
        self.current = i
        self.loaded = True

    def status(self) -> String:
        if self.current < 0:
            return String("no ckpt selected")
        if not self.loaded:
            return String("MISSING ") + self.labels[self.current]
        return self.labels[self.current] + String(" loaded")

    def act(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        greedy: Bool,
    ) raises:
        if not self.loaded:
            # Leave the action at zero rather than running an untrained net: a
            # jittering walker would read as "this rung is bad", not "this rung
            # is absent".
            for j in range(ACT_DIM):
                action_out[j] = Scalar[DT](0)
            return
        if greedy:
            self.agent.select_greedy_action(obs, action_out)
        else:
            self.agent.select_action(obs, action_out, self.step_idx)
            self.step_idx += 1


# ═══════════════════════════════════════════════════════════════════════════
# the three-task front end
# ═══════════════════════════════════════════════════════════════════════════


def task_names() -> List[String]:
    """⚠ POSITIONALLY COUPLED TO `dispatch` — index i here is the arm
    `st.task == i` there."""
    var t = List[String]()
    t.append(String("walker_stand"))
    t.append(String("walker_walk"))
    t.append(String("walker_run"))
    return t^


def domain_names() -> List[String]:
    var d = List[String]()
    d.append(String("walker"))
    return d^


def task_domain() -> List[Int]:
    var t = List[Int]()
    for _ in range(3):
        t.append(0)
    return t^


def dispatch(
    mut st: ViewerState, policy: Pointer[WalkerLadder, MutAnyOrigin]
) raises:
    """Run whichever task `st.task` names, and return when it wants another.

    The MOVE_SPEED values are the dm_control ones and must match the training
    script's, or the reward sparkline reads a different task than the checkpoint
    was trained for while both still look plausible.
    """
    var name = task_names()[st.task]
    if st.task == 0:
        run_view[DMWalkerModel, DMWalkerConfig[0.0], WalkerLadder](
            name, st, policy
        )
    elif st.task == 1:
        run_view[DMWalkerModel, DMWalkerConfig[1.0], WalkerLadder](
            name, st, policy
        )
    elif st.task == 2:
        run_view[DMWalkerModel, DMWalkerConfig[8.0], WalkerLadder](
            name, st, policy
        )
    else:
        print("unknown task index:", st.task)
        st.quit = True


def _run_dir(handle: String) raises -> String:
    """`handle` as a run directory: a path to one, or a run id under `runs/` or
    any `projects/*/runs/`. Raises when none of them holds a `checkpoints/`."""
    if exists(handle + "/checkpoints"):
        return handle
    if exists(String("runs/") + handle + "/checkpoints"):
        return String("runs/") + handle
    var root = projects_root()
    var listing = run_capture(
        String("ls -d ") + quote_arg(root) + "/*/runs/" + quote_arg(handle)
        + " 2>/dev/null; true",
        1 << 16,
    )
    for line in listing.split("\n"):
        var d = String(String(line).strip())
        if d.byte_length() > 0 and exists(d + "/checkpoints"):
            return d
    raise Error(
        "--run '" + handle + "' is neither a run directory nor a run id under"
        " runs/ or " + root + "/*/runs/"
    )


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    # `--run <run_id|run_dir>` is pulled out first; the rest stay positional.
    var av = argv()
    var run_arg = _flag(String("--run"), String(""))
    var run_dir = _run_dir(run_arg) if run_arg.byte_length() > 0 else String("")
    var args = List[String]()
    var ai = 0
    while ai < len(av):
        if ai > 0 and String(av[ai]) == "--run":
            ai += 2
            continue
        args.append(String(av[ai]))
        ai += 1
    var start = String(args[1]) if len(args) > 1 else String("walker_walk")
    var task = task_index(start, task_names())
    if task < 0:
        print("unknown task:", start, "— this front end registers:")
        var names = task_names()
        for i in range(len(names)):
            print("   ", names[i])
        return

    var drive = parse_drive(String(args[2])) if len(args) > 2 else DRIVE_POLICY
    var scale = Float64(1.0)
    if len(args) > 3:
        try:
            scale = Float64(String(args[3]))
        except:
            print("bad scale, using 1.0")

    print("=" * 66)
    print("dm_control walker x 3 tasks + SAC ladder")
    print("=" * 66)
    var ladder = WalkerLadder(run_dir)

    var st = ViewerState(
        task, drive, scale, task_names(), domain_names(), task_domain()
    )
    # Open on the EXPERT rung of the task that opens first — the interesting
    # one; rung 1 is near-random by construction.
    #
    # ⚠ THIS ARITHMETIC COUPLES TWO TABLES: it only lands on the right block
    # because `ckpt_tasks()` lists the ladder's tasks in the SAME order as
    # `task_names()` lists the envs. Reorder one and the viewer opens the walk
    # policy in the stand env — which is a legal thing to ask for here, and
    # therefore looks like a deliberate choice rather than a bug.
    st.policy_variant = Int32((task + 1) * N_SEGMENTS - 1)

    # ⚠ THE LADDER OUTLIVES EVERY ENV, and must: it holds the loaded weights,
    # so a task switch reuses them instead of re-reading the checkpoint.
    var pol = Pointer(to=ladder).as_unsafe_any_origin()
    while not st.quit:
        dispatch(st, pol)
    _ = ladder  # lifetime extender for `pol`

    # The last task's window is handed OUT, not closed, whenever the loop ends
    # on a switch — and `dispatch`'s unknown-index arm ends it without one.
    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
