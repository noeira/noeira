"""TD-MPC2 checkpoint driving dm_control `walker` — watch the gait.

    pixi run build-imgui                                        # ONCE
    pixi run -e apple mojo run -I . examples/dm_control/tdmpc2_dm_walker_policy_viewer.mojo
    pixi run -e apple mojo run -I . examples/dm_control/tdmpc2_dm_walker_policy_viewer.mojo walker_run
    pixi run -e apple mojo run -I . examples/dm_control/tdmpc2_dm_walker_policy_viewer.mojo --ckpt <run_id>

argv picks the task that opens first, then the drive mode (zero | random |
sweep | policy) and an action scale; every task and mode stays selectable in
the window. Defaults: `walker_walk`, `policy`.

The TD-MPC2 counterpart of `dm_walker_policy_viewer.mojo` (the SAC ladder one),
simplified: TD-MPC2 writes ONE checkpoint per run, not a 20-rung ladder — the
driver overwrites `checkpoint_path` on every save — so there is no rung axis to
sweep. The axis that replaces it is the ACTING MODE, and that one matters more.

## ⚠ MPC vs the policy prior — the two are NOT the same agent

TD-MPC2 acts by PLANNING: MPPI rolls candidate action sequences through the
learned world model and ranks them by predicted reward + terminal Q. The policy
network `π` is a prior that seeds and regularizes that search; it is not the
deployed controller. `train_batched`'s eval inherits the training `USE_MPC`, so
a reported `eval/mean_return` from an MPC run is the PLANNER's score.

## The iteration ladder — the frame-rate knob

MPPI cost is near-linear in `MPC_ITERS`, and iterations are the ONLY budget
parameter that can change at runtime (samples and pi-trajs size the planner's
device buffers at construction). So each checkpoint contributes one MPC row per
rung of `MPC_ITER_LADDER` plus a `prior` row, and the sidebar's +/- buttons
walk that ladder live.

⚠ THE LADDER IS A SPEED/QUALITY TRADE, NOT A FREE SPEEDUP. Only the rung equal
to `MPC_ITERS` matches what training used and what the reported eval return
measured; the cheaper rungs plan less and will act worse. The viewer opens on
the training rung for that reason. Watching WHERE the gait breaks as you step
down is the useful part — a policy that still walks at i1 has absorbed the plan
into the prior, one that collapses at i3 is leaning on the planner.

Both acting modes are offered as variants, and they will not look alike:

  * `MPC`   — `select_action_mpc`, the thing the eval return measured. GPU
    only, and it runs the full MPPI budget PER FRAME, so expect single-digit
    to low-tens Hz on Apple/Metal, not 60. Use pause/step for a close look.
  * `prior` — `a = π(encode(obs))`, real-time, and typically WEAKER. If the
    prior looks much worse than the eval number led you to expect, that is the
    expected MPC-vs-prior gap, not a broken checkpoint.

Reading the two against each other is the point: a large gap means the world
model + value function are carrying the performance; a small one means π has
absorbed the plan and could be deployed without a planner.

## ⚠ THE DIMS BELOW MUST MATCH THE TRAINING SCRIPT

`load_state` restores parameter slabs BY NAME. Architecture dims (ENC / LATENT
/ MLP / BINS / SN) are part of those shapes, so a mismatch fails loudly — but
`VMIN`/`VMAX`/`H` and the MPPI budget are NOT stored, and a mismatch there
loads clean and plans differently. Keep them in step with
`tdmpc2_dm_walker_batched_gpu.mojo` (or `tdmpc2_dm_walker_gpu.mojo`).

`B` and `CAP` are deliberately TINY here: they size the training batch scratch
and the replay ring, neither of which a viewer uses, and neither is in the
checkpoint. The 1 M-slot ring the training script asks for would cost ~136 MB
of host memory to open a window with.

## ⚠ THE CHECKPOINT IS NOT IN THE REPO

`CKPT_DIRS` probes `checkpoints/` then the project root, for both the batched
and single-env filenames of all three tasks; `--ckpt <run_id|path>` adds a
run's `checkpoints/last.ckpt` (or a file) as the first checkpoint. A variant
whose file is absent is labelled `(missing)` in the combo and left un-driven
rather than silently driving an uninitialised net.

⚠ CPU PHYSICS + GPU AGENT. The viewer's env is the CPU float32 path while the
checkpoint trained on the GPU batched path, so treat small differences as path
noise and only large ones as a finding. The agent keeps its OWN DeviceContext
(the viewer builds one for rendering); nothing crosses between them — the
`ActionSource` interface is host `List`s.

⚠ MPPI WARM-START SURVIVES AN ENV RESET. `ActionSource` has no reset hook, so
the planner's shifted mean carries one step across a reset before the first
refit corrects it. Harmless for watching; do not read the first frame of an
episode as policy behaviour.
"""

from std.pathlib import Path
from std.random import seed
from std.sys import argv
from noeira.core.run import resolve_checkpoint, run_id_of_checkpoint
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.tdmpc2.config import TDMPC2
from noeira.deep_agents.tdmpc2.agent import TDMPC2Agent
from noeira.envs.dm_control.viewer_core import (
    ActionSource, ViewerState, run_view, task_index, parse_drive, DRIVE_POLICY,
)
from noeira.render.imgui import imgui_shim_available
from noeira.render.renderer3d import Renderer3D

from noeira.envs.dm_control.walker.walker_xml import DMWalkerModel
from noeira.envs.dm_control.walker.walker_config import DMWalkerConfig

comptime SEED: Int = 0

comptime OBS_DIM = DMWalkerModel.OBS_DIM      # 24
comptime ACT_DIM = DMWalkerModel.ACTION_DIM   #  6

# ══ ARCHITECTURE — MUST MATCH THE TRAINING SCRIPT ═══════════════════════
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime H = 3
# MPPI budget — also the training script's. Lower these if the MPC variant is
# too slow to watch; the gait degrades gracefully as the budget shrinks, which
# is itself worth seeing.
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4
# ── the live frame-rate knob ────────────────────────────────────────────
# MPPI cost is near-linear in the iteration count, and it is the ONLY budget
# parameter that can vary at runtime — NUM_SAMPLES / NUM_PI_TRAJS size the
# planner's device buffers at construction. So the variant list crosses
# checkpoint x mode x THIS, and the sidebar's +/- buttons walk it.
#
# Measured on an M1 Pro at the walker's dims (S=256, H=3): iters=4 is ~53 ms
# (~19 Hz) and the cost scales down close to linearly, so the low rungs are
# what make an interactive frame rate reachable. The gait degrades gracefully
# as the budget shrinks — watching WHERE it breaks is the point of having the
# knob, not just the Hz.
# ⚠ A list-returning `def`, not a comptime array: a comptime `Array` cannot be
# materialized to runtime (it is not ImplicitlyCopyable). Same pattern as
# `task_names()` / `ckpt_tasks()` below.

# Viewer-only sizes (see the header): never trains, never fills a replay.
comptime B = 8
comptime CAP = 1024

comptime AgentT = TDMPC2Agent[
    "gpu", OBS_DIM, ENC, ACT_DIM, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H,
    CAP, MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
]


def mpc_iter_ladder() -> List[Int]:
    """MPPI iteration budgets offered as variants, cheapest first."""
    var v = List[Int]()
    v.append(1)
    v.append(2)
    v.append(3)
    v.append(4)
    v.append(6)
    return v^


def ckpt_dirs() -> List[String]:
    """Where a checkpoint might live, in probe order (training writes to CWD;
    the ladders on this machine were moved into `checkpoints/`)."""
    var d = List[String]()
    d.append(String("checkpoints/"))
    d.append(String(""))
    return d^


def ckpt_names() -> List[String]:
    """Every filename the two TD-MPC2 walker training scripts can write.

    ⚠ POSITIONALLY COUPLED to the `_mpc` / `_mpcoff` suffix those scripts pick
    from their own `USE_MPC`. A run trained MPC-off still loads and still
    plans here — the checkpoint holds no acting mode — but its world model was
    never shaped by planning data, so expect the MPC variant to disappoint.
    """
    var tasks = List[String]()
    tasks.append(String("stand"))
    tasks.append(String("walk"))
    tasks.append(String("run"))

    var n = List[String]()
    for t in range(len(tasks)):
        n.append("tdmpc2_dm_walker_batched_" + tasks[t] + "_mpc.ckpt")
        n.append("tdmpc2_dm_walker_batched_" + tasks[t] + "_mpcoff.ckpt")
        n.append("tdmpc2_dm_walker_" + tasks[t] + "_mpc.ckpt")
        n.append("tdmpc2_dm_walker_" + tasks[t] + "_mpcoff.ckpt")
    return n^


# ═══════════════════════════════════════════════════════════════════════════
# the ActionSource — one checkpoint, two acting modes
# ═══════════════════════════════════════════════════════════════════════════


struct TDMPC2Walker(ActionSource, Movable, Deinitable):
    """A TD-MPC2 checkpoint as a selectable policy, in both acting modes.

    ONE agent, reloaded in place: `load_state` overwrites the world model +
    Q ensemble + policy, so switching checkpoint is a few milliseconds and the
    nets (and the MPPI planner's device scratch) are built exactly once.

    VARIANTS ARE FLAT AND CHECKPOINT-MAJOR — `<ckpt> MPC`, `<ckpt> prior`,
    next checkpoint, … — so the sidebar's +/- buttons toggle the ACTING MODE
    on one checkpoint, which is the comparison worth stepping through.
    """

    var agent: AgentT
    var paths: List[String]
    """Resolved path per variant, or "" for a checkpoint no directory holds."""
    var labels: List[String]
    var use_mpc: List[Bool]
    """Acting mode per variant, parallel to `paths`."""
    var iters: List[Int]
    """MPPI iteration budget per variant, parallel to `paths`. 0 on the
    `prior` rows, which never plan."""
    var current: Int
    var loaded: Bool

    def __init__(out self, ckpt: String) raises:
        # The agent owns its own context — `run_view` builds a separate one for
        # the renderer. Safe because the two never share a buffer: this trait
        # speaks host Lists.
        var ctx = DeviceContext()
        self.agent = TDMPC2[
            "gpu", OBS_DIM, ACT_DIM, B, CAP, ENC, LATENT, MLP, BINS, SN,
            VMIN, VMAX, H, MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
        ](ctx=ctx, action_scale=Scalar[DT](1.0), learning_starts=0)
        self.paths = List[String]()
        self.labels = List[String]()
        self.use_mpc = List[Bool]()
        self.iters = List[Int]()
        self.current = -1
        self.loaded = False

        var names = ckpt_names()
        var dirs = ckpt_dirs()
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
        # A missing checkpoint contributes NO variants at all — with 12
        # candidate filenames x 2 modes, listing the absent ones would bury
        # the one that exists in 22 lines of "(missing)".
        for i in range(len(founds)):
            var found = founds[i]
            n_found += 1
            var short = shorts[i]
            # One MPC row per iteration budget, cheapest first, then the
            # non-planning prior. +/- therefore steps the frame-rate knob.
            var ladder = mpc_iter_ladder()
            for k in range(len(ladder)):
                var it = ladder[k]
                self.paths.append(found)
                self.labels.append(short + " MPC i" + String(it))
                self.use_mpc.append(True)
                self.iters.append(it)
            self.paths.append(found)
            self.labels.append(short + " prior")
            self.use_mpc.append(False)
            self.iters.append(0)

        print("  checkpoints found:", n_found, "→", len(self.paths), "variants")
        for i in range(len(self.labels)):
            print("    ", self.labels[i], " ←", self.paths[i])
        if n_found == 0:
            print("  ⚠ NO checkpoint found. Looked in checkpoints/ and ./ for")
            print("      tdmpc2_dm_walker_batched_<task>_{mpc,mpcoff}.ckpt")
            print("      tdmpc2_dm_walker_<task>_{mpc,mpcoff}.ckpt")
            print("    Train one first:")
            print(
                "      pixi run -e nvidia mojo run -I ."
                " examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo"
            )
            print("    The walker will stand inert until then.")

    # ─── ActionSource ───────────────────────────────────────────────────

    def obs_dim(self) -> Int:
        return OBS_DIM

    def act_dim(self) -> Int:
        return ACT_DIM

    def variant_labels(self) -> List[String]:
        return self.labels.copy()

    def choose(mut self, i: Int) raises:
        """Load variant `i`, or raise WITHOUT SIDE EFFECTS — a failed switch
        leaves the previously loaded checkpoint driving, so the sidebar, the
        status line and the torque never disagree about which policy is live.
        """
        if i < 0 or i >= len(self.paths):
            raise Error("variant out of range: " + String(i))
        if not self.paths[i]:
            raise Error("checkpoint not on disk: " + self.labels[i])
        # Re-reading the same file when only the MODE changed is a few ms and
        # keeps this branch-free; the planner's warm-start is reset either way.
        self.agent.load_state(self.paths[i])
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
            # Zero torque rather than an untrained net: a jittering walker
            # reads as "this checkpoint is bad", not "there is no checkpoint".
            for j in range(ACT_DIM):
                action_out[j] = Scalar[DT](0)
            return
        if self.use_mpc[self.current]:
            # `greedy` → the eval-time planner (no exploration noise on the
            # selected action), which is what produced the eval return.
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
    mut st: ViewerState, policy: Pointer[TDMPC2Walker, MutAnyOrigin]
) raises:
    """Run whichever task `st.task` names, and return when it wants another.

    ⚠ All three tasks share ONE model and ONE observation — only the REWARD
    differs — so any checkpoint drives any of the three envs. The sidebar
    deliberately lets the checkpoint's task and the env's task disagree:
    "drive the walk policy, read the stand reward" is a real question and the
    reward sparkline answers it. The MOVE_SPEED values are dm_control's and
    must match the training script's.
    """
    var name = task_names()[st.task]
    if st.task == 0:
        run_view[DMWalkerModel, DMWalkerConfig[0.0], TDMPC2Walker](
            name, st, policy
        )
    elif st.task == 1:
        run_view[DMWalkerModel, DMWalkerConfig[1.0], TDMPC2Walker](
            name, st, policy
        )
    elif st.task == 2:
        run_view[DMWalkerModel, DMWalkerConfig[8.0], TDMPC2Walker](
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

    print("=" * 66)
    print("dm_control walker x 3 tasks + TD-MPC2 checkpoint")
    print("=" * 66)
    print("  MPPI budget:", MPC_SAMPLES, "+", MPC_PI_TRAJS, "trajs,",
          MPC_ITERS, "iters, horizon", H)
    print("  ⚠ the MPC variant plans EVERY frame — expect a few Hz, not 60")
    var walker = TDMPC2Walker(ckpt)

    var st = ViewerState(
        task, drive, scale, task_names(), domain_names(), task_domain()
    )
    # Open on the rung whose iteration budget MATCHES TRAINING (`MPC_ITERS`),
    # not variant 0 — variant 0 is now the cheapest rung (i1), and opening
    # there would show a deliberately under-budgeted planner while claiming to
    # show "the checkpoint". The eval return the training run reported is the
    # MPC_ITERS one; every other rung is a speed/quality probe off it.
    var open_at = 0
    var ladder = mpc_iter_ladder()
    for k in range(len(ladder)):
        if ladder[k] == MPC_ITERS:
            open_at = k
    st.policy_variant = Int32(open_at)

    # ⚠ THE POLICY OUTLIVES EVERY ENV, and must: it holds the loaded weights
    # and the planner's device scratch, so a task switch reuses them instead of
    # re-reading the checkpoint and re-allocating the MPPI buffers.
    var pol = Pointer(to=walker).as_unsafe_any_origin()
    while not st.quit:
        dispatch(st, pol)
    _ = walker  # lifetime extender for `pol`

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
