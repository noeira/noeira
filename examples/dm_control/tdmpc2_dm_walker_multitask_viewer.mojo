"""MULTI-TASK TD-MPC2 checkpoint driving dm_control `walker` — watch all three.

    pixi run build-imgui                                        # ONCE
    pixi run -e apple mojo run -I . examples/dm_control/tdmpc2_dm_walker_multitask_viewer.mojo
    pixi run -e apple mojo run -I . examples/dm_control/tdmpc2_dm_walker_multitask_viewer.mojo walker_run
    pixi run -e apple mojo run -I . examples/dm_control/tdmpc2_dm_walker_multitask_viewer.mojo --ckpt <run_id>

The multi-task counterpart of `tdmpc2_dm_walker_policy_viewer.mojo`. One
checkpoint holds ONE task-conditioned world model for stand + walk + run, so
the axis that replaces "which checkpoint" is **which task id the agent is
conditioned on** — and that id is chosen INDEPENDENTLY of which env is running.

## The cross-task matrix, live

`run_view` picks the ENV (its reward, via `MOVE_SPEED`); the variant picks the
TASK ID fed to the encoder, the dynamics, the reward/Q heads and the planner.
Crossing them is the whole point:

  * id == env  — the diagonal. What the training eval measured.
  * id != env  — drive the walk env while telling the agent it is standing.

The offline probe (`tdmpc2_dm_walker_multitask_probe.mojo`) reports that
crossing as a 3x3 table of returns. Here you WATCH it, which shows something a
number cannot: whether a wrong id produces a different GAIT or merely a worse
score. A model whose conditioning works should visibly stop trying to move when
told it is standing, even in the run env.

⚠ The reward sparkline always reflects the ENV's reward, never the id's. That
is what makes the off-diagonal readable: same reward function, different
conditioning.

## ⚠ MPC vs the policy prior

Unchanged from the single-task viewer: TD-MPC2 acts by PLANNING (MPPI through
the learned world model), and `π` is a prior that seeds the search, not the
deployed controller. The eval returns quoted from a training run with
`USE_MPC=True` are the PLANNER's scores. The `prior` rows are real-time and
typically weaker; a large gap means the world model + value function carry the
performance, a small one means π absorbed the plan.

## The iteration ladder

MPPI cost is near-linear in `MPC_ITERS`, the only budget knob that can change
at runtime (samples and pi-trajs size device buffers at construction). Each
task id contributes one MPC row per rung plus a `prior` row; the sidebar's +/-
buttons walk it. Only the rung equal to `MPC_ITERS` matches training, so the
viewer opens there — the cheaper rungs plan less and act worse BY DESIGN.

## ⚠ DIMS MUST MATCH THE TRAINING SCRIPT

`load_state` restores slabs BY NAME, so ENC / LATENT / MLP / BINS / SN /
TASK_EMB / NUM_TASKS mismatches fail loudly — but `VMIN`/`VMAX`/`H` and the
MPPI budget are NOT stored and a mismatch there loads clean and plans
differently. Keep them in step with `tdmpc2_dm_walker_multitask_gpu.mojo`.

⚠ TASK IDS ARE POSITIONAL. Row 1 of the embedding table means "walk" only
because the training script's table said so. `TASK_LABELS` below MUST match
`T_STAND / T_WALK / T_RUN` there.

`B` and `CAP` are deliberately tiny: they size training scratch and the replay
ring, neither of which a viewer touches, and neither is in the checkpoint.

## ⚠ CHECKPOINTS ARE NOT IN THE REPO

Probes `checkpoints/` then the project root; `--ckpt <run_id|path>` adds a
run's `checkpoints/last.ckpt` (or a file) as the first checkpoint. ⚠ Checkpoints written before
`8d7f07d8` are INCOMPATIBLE — the MT dynamics' final layer changed shape when
its output width was fixed from LATENT+TASK_EMB back to LATENT — and will fail
to load rather than load wrong. That is the intended behaviour; those runs
trained a world model that never converged.

⚠ CPU PHYSICS + GPU AGENT, as in the single-task viewer: the env is the CPU
float32 path while the checkpoint trained on the GPU batched path. Treat small
differences as path noise. The agent keeps its own DeviceContext.

⚠ MPPI WARM-START SURVIVES AN ENV RESET (no reset hook on `ActionSource`), and
`choose` also re-seeds it on a task-id switch — do not read the first frame
after either as policy behaviour.
"""

from std.pathlib import Path
from std.random import seed
from std.sys import argv
from noeira.core.run import resolve_checkpoint, run_id_of_checkpoint
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.tdmpc2.config_mt import TDMPC2MultiTask
from noeira.deep_agents.tdmpc2.agent_mt import TDMPC2MultiTaskAgent
from noeira.envs.dm_control.viewer_core import (
    ActionSource, ViewerState, run_view, task_index, parse_drive, DRIVE_POLICY,
)
from noeira.render.imgui import imgui_shim_available
from noeira.render.renderer3d import Renderer3D

from noeira.envs.dm_control.walker.walker_xml import DMWalkerModel
from noeira.envs.dm_control.walker.walker_config import DMWalkerConfig

comptime SEED: Int = 0

comptime MAX_OBS = DMWalkerModel.OBS_DIM      # 24
comptime MAX_ACT = DMWalkerModel.ACTION_DIM   #  6

# ══ ARCHITECTURE — MUST MATCH tdmpc2_dm_walker_multitask_gpu.mojo ═══════
comptime NUM_TASKS = 3
comptime TASK_EMB = 32
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime H = 3
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

# Viewer-only sizes (never trains, never fills a replay).
comptime B = 8
comptime CAP = 1024

comptime AgentT = TDMPC2MultiTaskAgent[
    "gpu", MAX_OBS, ENC, MAX_ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H,
    CAP, NUM_TASKS, TASK_EMB, 0.0,
    MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
]


def mpc_iter_ladder() -> List[Int]:
    """MPPI iteration budgets offered as variants, cheapest first."""
    var v = List[Int]()
    v.append(1)
    v.append(2)
    v.append(4)
    v.append(6)
    return v^


def task_labels() -> List[String]:
    """⚠ POSITIONALLY COUPLED to the training script's T_STAND / T_WALK /
    T_RUN. Index i here IS the embedding row fed to the agent."""
    var t = List[String]()
    t.append(String("stand"))
    t.append(String("walk"))
    t.append(String("run"))
    return t^


def ckpt_dirs() -> List[String]:
    var d = List[String]()
    d.append(String("checkpoints/"))
    d.append(String(""))
    return d^


def ckpt_names() -> List[String]:
    """Multi-task checkpoint filenames, newest naming first.

    ⚠ `tdmpc2_dm_walker_multitask.ckpt` (untagged) is the PRE-FIX run — the
    130k MPC-off / UTD=0.125 one whose world model never converged. It is
    listed last and will FAIL to load against the current nets, which is
    correct: its dynamics layer has the old LATENT+TASK_EMB output width.
    """
    var n = List[String]()
    n.append(String("tdmpc2_dm_walker_multitask_mpc_utd8.ckpt"))
    n.append(String("tdmpc2_dm_walker_multitask_mpcoff_utd8.ckpt"))
    n.append(String("tdmpc2_dm_walker_multitask_mpc_utd1.ckpt"))
    n.append(String("tdmpc2_dm_walker_multitask.ckpt"))
    return n^


# ═══════════════════════════════════════════════════════════════════════════
# the ActionSource — one checkpoint, three task ids, two acting modes
# ═══════════════════════════════════════════════════════════════════════════


struct TDMPC2WalkerMT(ActionSource, Movable, Deinitable):
    """A multi-task checkpoint as a selectable policy, across task ids.

    ONE agent, reloaded in place. Variants are CHECKPOINT-major then TASK-ID
    major then mode, so the sidebar's +/- buttons step the task id on a fixed
    checkpoint — which is the comparison this viewer exists for.
    """

    var agent: AgentT
    var paths: List[String]
    var labels: List[String]
    var task_ids: List[Int]
    """Embedding row per variant — INDEPENDENT of the env `run_view` runs."""
    var use_mpc: List[Bool]
    var iters: List[Int]
    """MPPI iteration budget per variant; 0 on `prior` rows, which never plan."""
    var current: Int
    var loaded: Bool
    var loaded_path: String
    """Which file is currently in the nets — lets a task-id switch skip the
    re-read, which is the common case when stepping +/-."""

    def __init__(out self, ckpt: String) raises:
        var ctx = DeviceContext()
        self.agent = TDMPC2MultiTask[
            "gpu", MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
            ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
            NUM_SAMPLES=MPC_SAMPLES, NUM_PI_TRAJS=MPC_PI_TRAJS,
            NUM_ELITES=MPC_ELITES, NUM_ITERS=MPC_ITERS,
        ](ctx=ctx, action_scale=Scalar[DT](1.0), learning_starts=0)
        self.paths = List[String]()
        self.labels = List[String]()
        self.task_ids = List[Int]()
        self.use_mpc = List[Bool]()
        self.iters = List[Int]()
        self.current = -1
        self.loaded = False
        self.loaded_path = String("")

        var names = ckpt_names()
        var dirs = ckpt_dirs()
        var tl = task_labels()
        var ladder = mpc_iter_ladder()
        var n_found = 0
        # `--ckpt` (already resolved to a file) goes first, then the probed names.
        var founds = List[String]()
        var shorts = List[String]()
        if ckpt.byte_length() > 0:
            var rid = run_id_of_checkpoint(ckpt)
            founds.append(ckpt)
            shorts.append(rid if rid.byte_length() > 0 else ckpt)
        for i in range(len(names)):
            for d in range(len(dirs)):
                var cand = dirs[d] + names[i]
                if Path(cand).exists():
                    founds.append(cand)
                    shorts.append(
                        names[i].replace("tdmpc2_dm_walker_", "").replace(
                            ".ckpt", ""
                        )
                    )
                    break
        for i in range(len(founds)):
            var found = founds[i]
            n_found += 1
            var short = shorts[i]
            for t in range(NUM_TASKS):
                for k in range(len(ladder)):
                    self.paths.append(found)
                    self.labels.append(
                        short + " id=" + tl[t] + " MPC i" + String(ladder[k])
                    )
                    self.task_ids.append(t)
                    self.use_mpc.append(True)
                    self.iters.append(ladder[k])
                self.paths.append(found)
                self.labels.append(short + " id=" + tl[t] + " prior")
                self.task_ids.append(t)
                self.use_mpc.append(False)
                self.iters.append(0)

        print("  checkpoints found:", n_found, "→", len(self.paths), "variants")
        if n_found == 0:
            print("  ⚠ NO multi-task checkpoint found. Looked in checkpoints/")
            print("      and ./ for:")
            for i in range(len(names)):
                print("       ", names[i])
            print("    Train one first:")
            print(
                "      pixi run -e nvidia mojo run -I ."
                " examples/dm_control/tdmpc2_dm_walker_multitask_gpu.mojo"
            )
            print("    The walker will stand inert until then.")

    # ─── ActionSource ───────────────────────────────────────────────────

    def obs_dim(self) -> Int:
        return MAX_OBS

    def act_dim(self) -> Int:
        return MAX_ACT

    def variant_labels(self) -> List[String]:
        return self.labels.copy()

    def choose(mut self, i: Int) raises:
        """Load variant `i`, or raise WITHOUT SIDE EFFECTS.

        The file is re-read only when it actually changes — stepping the task
        id or the iteration rung on one checkpoint is then free, which is the
        common case here (12 variants per checkpoint).
        """
        if i < 0 or i >= len(self.paths):
            raise Error("variant out of range: " + String(i))
        if not self.paths[i]:
            raise Error("checkpoint not on disk: " + self.labels[i])
        if self.paths[i] != self.loaded_path:
            self.agent.load_state(self.paths[i])
            self.loaded_path = self.paths[i]
        self.agent.set_task(self.task_ids[i])
        # ⚠ A plan carried over from the previous task id is meaningless under
        # the new conditioning — drop the warm-start on every switch.
        self.agent.mpc_start_episode()
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
            for j in range(MAX_ACT):
                action_out[j] = Scalar[DT](0)
            return
        if self.use_mpc[self.current]:
            self.agent.select_action_mpc(
                obs, action_out, explore=not greedy,
                num_iters=self.iters[self.current],
            )
        else:
            self.agent.select_action(obs, action_out, explore=not greedy)


# ═══════════════════════════════════════════════════════════════════════════
# the three-task front end
# ═══════════════════════════════════════════════════════════════════════════


def task_names() -> List[String]:
    """⚠ POSITIONALLY COUPLED TO `dispatch`. These name the ENV (the reward),
    NOT the agent's conditioning — that is the variant axis."""
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
    mut st: ViewerState, policy: Pointer[TDMPC2WalkerMT, MutAnyOrigin]
) raises:
    """Run whichever ENV `st.task` names. The agent's task id is independent —
    see the header: crossing the two is what this viewer is for."""
    var name = task_names()[st.task]
    if st.task == 0:
        run_view[DMWalkerModel, DMWalkerConfig[0.0], TDMPC2WalkerMT](
            name, st, policy
        )
    elif st.task == 1:
        run_view[DMWalkerModel, DMWalkerConfig[1.0], TDMPC2WalkerMT](
            name, st, policy
        )
    elif st.task == 2:
        run_view[DMWalkerModel, DMWalkerConfig[8.0], TDMPC2WalkerMT](
            name, st, policy
        )
    else:
        print("unknown task index:", st.task)
        st.quit = True


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

    # `--ckpt <run_id|path>` is pulled out first; the rest stay positional.
    var av = argv()
    var ckpt_arg = _flag(String("--ckpt"), String(""))
    var ckpt = String("")
    if ckpt_arg.byte_length() > 0:
        ckpt = resolve_checkpoint(ckpt_arg, String("last"))
    var args = List[String]()
    var ai = 0
    while ai < len(av):
        if ai > 0 and String(av[ai]) == "--ckpt":
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

    print("=" * 70)
    print("dm_control walker x 3 envs + MULTI-TASK TD-MPC2 checkpoint")
    print("=" * 70)
    print("  MPPI:", MPC_SAMPLES, "+", MPC_PI_TRAJS, "trajs,", MPC_ITERS,
          "iters, horizon", H, " task_emb", TASK_EMB)
    print("  ⚠ the MPC variants plan EVERY frame — expect a few Hz, not 60")
    print("  ⚠ the variant picks the agent's TASK ID; the sidebar's task combo")
    print("    picks the ENV. Crossing them is the point — the reward")
    print("    sparkline always reads the ENV's reward.")
    var walker = TDMPC2WalkerMT(ckpt)

    var st = ViewerState(
        task, drive, scale, task_names(), domain_names(), task_domain()
    )
    # Open on the id matching the starting env, at the TRAINING iteration rung
    # — the diagonal cell, i.e. what the eval return measured. Every other
    # variant is a probe off that.
    var ladder = mpc_iter_ladder()
    var rung = 0
    for k in range(len(ladder)):
        if ladder[k] == MPC_ITERS:
            rung = k
    var per_task = len(ladder) + 1
    st.policy_variant = Int32(task * per_task + rung)

    var pol = Pointer(to=walker).as_unsafe_any_origin()
    while not st.quit:
        dispatch(st, pol)
    _ = walker  # lifetime extender for `pol`

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
