"""Task-agnostic half of the dm_control viewer — state, sidebar, run loop.

`viewer.mojo` is the other half: the 48-arm `dispatch` that names every task's
compile-time `Phyics3dEnv[MODEL, CONFIG]`, plus `run_viewer` around it.

⚠ THE SPLIT IS A BUILD-TIME LEVER, not tidiness. `dispatch` instantiates
`run_view` 48 times, and importing the module that holds it pays for all 48
even if the importer only wants two. Keeping everything task-agnostic HERE lets
a front end with its own short task list — `examples/dm_control/
dm_viewer_imgui_two.mojo`, which exists so this code can be exercised in
seconds rather than minutes — import `run_view` without dragging the full
dispatch in behind it.

⚠ SO NOTHING IN THIS FILE MAY NAME A CONCRETE TASK. The task table lives on
`ViewerState`, supplied by the caller, precisely so that stays true.

THE SAME RULE COVERS POLICIES. `DRIVE_POLICY` drives the actuators from an
`ActionSource` — a four-method trait over HOST LISTS — and the conformer, its
weights and its checkpoint paths all live in the front end
(`examples/dm_control/dm_walker_policy_viewer.mojo` is the first one). So this
module still depends on nothing from `nn` / `deep_agents`, and the 48-task
viewer compiles no policy code at all: `POLICY` defaults to `NoPolicy`.

Read `viewer.mojo`'s header for the controls, the drive modes, and what the
tool can and cannot tell you.
"""

from std.random import random_float64
from std.math import sin, pi, min, max
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.envs.phyics3d_env import Phyics3dEnv, Phyics3dEnvConfig
from noeira.physics3d.fields import Data, Model, DimsLike
from noeira.physics3d.model import ModelDefLike
from noeira.render.renderer3d import Renderer3D, RendererHandoff
from noeira.math3d import Vec3 as Vec3Generic
from noeira.render.imgui import (
    imgui_shim_available,
    ig_begin_panel, ig_end, ig_begin_child, ig_end_child,
    ig_text, ig_text_colored, ig_text_disabled,
    ig_separator_text, ig_same_line, ig_spacing,
    ig_button, ig_toggle_button, ig_selectable, ig_checkbox,
    ig_slider_float, ig_combo, ig_input_text, ig_tree_node, ig_tree_pop,
    ig_set_next_item_width, ig_plot_lines,
    ig_push_id_int, ig_pop_id, ig_style_dark, ig_content_width,
    TextBuffer,
)


# ── drive modes ─────────────────────────────────────────────────────────────
#
#   zero    no torque. The honest reset pose plus gravity: where the model
#           actually settles, and whether it spawns intersecting the floor.
#   random  uniform in [-1, 1] per actuator, resampled every HOLD_STEPS.
#           Shakes every joint; spots one that cannot move or has no limit.
#   sweep   a slow out-of-phase sine per actuator (DEFAULT). The clearest way
#           to read joint AXES and RANGES — each joint traces its full arc
#           smoothly instead of jittering.
#   policy  a TRAINED policy drives the actuators — available only when the
#           front end passed an `ActionSource` to `run_view` (see the trait).
#           This is the one mode whose picture is supposed to look like the
#           task; the other three exist to inspect the MODEL, not the agent.
comptime DRIVE_ZERO: Int = 0
comptime DRIVE_RANDOM: Int = 1
comptime DRIVE_SWEEP: Int = 2
comptime DRIVE_POLICY: Int = 3

comptime HOLD_STEPS: Int = 25
comptime SWEEP_PERIOD: Float64 = 120.0
comptime EPISODE_STEPS: Int = 1000
"""Default viewer-side episode length. Override per front end with
`ViewerState.episode_steps`; 0 disables it."""

comptime KEY_N: Int = 110
"""`n` — force a new episode. Free because the renderer's own bindings claim
ESC, SPACE, R, S, V and 1-9 and forward everything else (`take_key`)."""
comptime FRAME_TARGET_MS: Int = 16
"""Target frame PERIOD, honoured as a LIMITER — not a flat sleep.

⚠ THIS WAS `renderer_delay(16)`, i.e. an unconditional `SDL_Delay`, and on a
model whose step is expensive that is pure waste: `reassemble_5`'s frame spent
**39.6% of its wall clock** in `SDL_SYS_DelayNS` (measured 2026-08-18, `sample`
over 25 s, 8109/20494 samples) while the physics it was pacing already took
~30 ms. Sixteen milliseconds added to a 30 ms frame is not a 60 Hz cap, it is a
21 Hz one. The loop now sleeps only the REMAINDER of the period, so a cheap
task (`reach_site`, 0.66 ms of physics) is capped at 60 Hz exactly as before
and an expensive one runs as fast as it can."""
comptime SIDEBAR_W: Float32 = 320.0
comptime PLOT_N: Int = 120
"""Samples in the reward sparkline. A ring buffer, so `ig_plot_lines` gets the
write cursor as its offset and plots oldest-first without a rotation."""


def drive_names(with_policy: Bool = False) -> List[String]:
    """Combo entries for the drive modes, in DRIVE_* index order.

    `with_policy` appends "policy" — omitted when the front end passed no
    `ActionSource`, so a viewer without one cannot offer a mode that would do
    nothing. The flag rather than a constant list because the SAME sidebar code
    serves both front ends.
    """
    var d = List[String]()
    d.append(String("zero"))
    d.append(String("random"))
    d.append(String("sweep"))
    if with_policy:
        d.append(String("policy"))
    return d^


def parse_drive(name: String) -> Int:
    if name == "zero":
        return DRIVE_ZERO
    if name == "random":
        return DRIVE_RANDOM
    if name == "policy":
        return DRIVE_POLICY
    return DRIVE_SWEEP


def task_index(name: String, names: List[String]) -> Int:
    """Position of `name` in a task table, or -1.

    Takes the table rather than reaching for a global one: this module must not
    know which tasks exist (see the header).
    """
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


def _contains(haystack: String, needle: String) -> Bool:
    """Substring test — `find` returning -1 is the whole implementation."""
    if needle.byte_length() == 0:
        return True
    return haystack.find(needle) != -1


# ═══════════════════════════════════════════════════════════════════════════
# policy plug — what `DRIVE_POLICY` calls into
# ═══════════════════════════════════════════════════════════════════════════


trait ActionSource:
    """A trained policy the viewer can drive the actuators with.

    ⚠ HOST LISTS, NOT TENSORS, AND DELIBERATELY SO. One env at 60 Hz makes the
    per-step copy free, and the list interface is what keeps this trait — and
    therefore `viewer_core` — free of any dependency on `nn` / `deep_agents`.
    The 48-task viewer must stay compilable without them.

    ⚠ THE DIMS ARE RUNTIME, so a mismatch cannot be a compile error. Report the
    policy's own dims here and `run_view` refuses to drive a model they do not
    fit, rather than reading past the end of an observation — the failure mode
    when a walker checkpoint meets a cheetah.

    VARIANTS ARE A FLAT LIST because the interesting selector — which rung of a
    SAC training ladder — is one axis, and a flat list keeps the sidebar's combo
    non-generic. A source with two axes (task x rung) flattens them in
    task-major order, so the sidebar's +/- buttons still step the inner axis.
    """

    def obs_dim(self) -> Int:
        ...

    def act_dim(self) -> Int:
        ...

    def variant_labels(self) -> List[String]:
        """Selectable variants, e.g. one per ladder rung. May be empty."""
        ...

    def choose(mut self, i: Int) raises:
        """Make variant `i` the live policy (loads its weights)."""
        ...

    def status(self) -> String:
        """One short line for the sidebar: which variant, and whether it
        loaded. Called every frame, so keep it cheap."""
        ...

    def act(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        greedy: Bool,
    ) raises:
        """Write `act_dim()` actions for `obs`. `greedy` asks for the
        deterministic action (the policy's mean) rather than a sample."""
        ...


@fieldwise_init
struct NoPolicy(ActionSource, Copyable, Movable):
    """The default `ActionSource`: there isn't one.

    Exists so `run_view`'s policy parameter can have a default and the 48
    existing call sites stay untouched. Every method is trivial, so the arm is
    dead code the moment `policy=None`.
    """

    def obs_dim(self) -> Int:
        return 0

    def act_dim(self) -> Int:
        return 0

    def variant_labels(self) -> List[String]:
        return List[String]()

    def choose(mut self, i: Int) raises:
        pass

    def status(self) -> String:
        return String("")

    def act(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        greedy: Bool,
    ) raises:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# step observer — what a RECORDER hangs off the loop
# ═══════════════════════════════════════════════════════════════════════════


trait StepObserver:
    """Sees every env step the viewer takes, with the PHYSICS STATE behind it.

    `ActionSource` is the viewer's input; this is its output. A recorder
    (`examples/so101/tower_teleop_record.mojo`) needs three things the
    action hook cannot give it: the observation BEFORE the step beside the
    one after, the env's `Data` (contacts, site poses — what a family reward
    is computed from; the CPU family config pays no reward of its own), and
    the episode boundaries. This trait carries exactly those.

    ⚠ GENERIC OVER THE DIMS PROVIDER, like every config hook, so this module
    still names no concrete model. `d` is `Phyics3dEnv.d` (BATCH 1) and `mf`
    its `Model` — the region table (`curriculum`), the site and body tables.

    `on_step` RETURNS whether the episode should END NOW (True = the loop
    resets as if `n` were pressed): a recorder that terminates on success
    asks for it here rather than reaching into the env. `key` is the keycode
    the renderer handed the application this frame (0 = none), AFTER the
    loop read it for its own `n`.
    """

    def on_step[DTYPE: DType, D: DimsLike](
        mut self,
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
        ref prev_obs: List[Scalar[DT]],
        ref action: List[Float64],
        ref obs: List[Scalar[DT]],
        done: Bool,
        step_i: Int,
        key: Int,
    ) raises -> Bool:
        ...

    def on_episode_end(mut self, step_i: Int, manual: Bool) raises:
        """The episode just ended (task `done`, the viewer's timeout, `n`,
        the sidebar button, or `on_step` asking): `step_i` steps were taken.
        Called BEFORE the reset, so the state is still the final one."""
        ...


@fieldwise_init
struct NoObserver(StepObserver, Copyable, Movable):
    """The default `StepObserver`: nobody is watching."""

    def on_step[DTYPE: DType, D: DimsLike](
        mut self,
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
        ref prev_obs: List[Scalar[DT]],
        ref action: List[Float64],
        ref obs: List[Scalar[DT]],
        done: Bool,
        step_i: Int,
        key: Int,
    ) raises -> Bool:
        return False

    def on_episode_end(mut self, step_i: Int, manual: Bool) raises:
        pass


def _fmt2(v: Float64) -> String:
    """Two decimals without a formatting library."""
    var scaled = Int(v * 100.0 + (0.5 if v >= 0 else -0.5))
    var whole = scaled // 100
    var frac = scaled % 100
    if frac < 0:
        frac = -frac
    var f = String(frac) if frac >= 10 else "0" + String(frac)
    return String(whole) + "." + f


# ═══════════════════════════════════════════════════════════════════════════
# state
# ═══════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct ViewerState(Copyable, Movable):
    """Everything that must SURVIVE a task switch.

    A switch tears the env down, so anything held in `run_view`'s locals dies
    with it. Drive mode, scale and the filter live here so picking a new robot
    does not silently reset how it is being driven or re-show the list the user
    just narrowed — the two most annoying ways a picker can behave.

    THE WINDOW CROSSES THE GAP TOO, via `handoff`. That is the difference
    between "the viewer switched model" and "the viewer closed and a new one
    opened somewhere": see the field's own note.
    """

    var task: Int
    var drive: Int32
    """Int32, not Int: `ig_combo` writes it through a C `int*`."""
    var scale: Float32
    """Float32 for the same reason — `ig_slider_float` owns a `float*`."""
    var quit: Bool
    """Set when the window closed for real, as opposed to for a switch."""
    var filter: TextBuffer
    var show_plot: Bool

    var policy_variant: Int32
    """Index into the `ActionSource`'s variant list — Int32 for `ig_combo`.

    HERE AND NOT IN `run_view` for the same reason as `drive`: switching task
    must not silently reset which policy is driving. The source itself lives in
    the FRONT END (it holds weights and outlives every env), so this is only the
    selection, and a source with fewer variants than this clamps on entry."""
    var policy_greedy: Bool
    """True = the deterministic action (the actor's mean). The default, because
    a viewer is for reading the LEARNED GAIT; SAC's stochastic action adds the
    entropy-calibrated jitter that is dataset coverage, not behaviour."""
    var free_camera: Bool
    """Open on the FREE camera (dm_control's -1) instead of model camera 0.

    ⚠ DEFAULTS TO FALSE so every existing front end opens exactly as before.
    Set it when the model's camera 0 is an ONBOARD one: `so_arm101.xml` ships a
    single `<camera name="wrist_cam">` bolted to the wrist, and `active_camera`
    starts at 0 — so the viewer opens looking down the gripper, whipping around
    with every joint, and the mouse cannot fight it back (a body-attached
    camera is re-aimed every frame; see `model_renderer.render`). The free
    camera is the only one the mouse fully owns.

    Kept HERE and not in `run_view` for the same reason as `drive`: it is a
    user choice, and clicking a numbered camera then switching task must not
    silently snap back to free. The loop writes it on every camera button."""
    var episode_steps: Int
    """Steps before the viewer forces a new episode. **0 means never**, which
    is what a teleoperated front end wants — there is no task to time out, and
    a reset mid-motion is a jolt with no meaning behind it. Defaults to
    `EPISODE_STEPS`, so every existing front end is unchanged.

    ⚠ This is only the VIEWER's limit. A task that returns `done` still ends
    its episode; see `CONFIG.MAX_STEPS`."""

    var tasks: List[String]
    var domains: List[String]
    var domain_of: List[Int]
    """The task table the sidebar lists and `st.task` indexes — SUPPLIED BY THE
    CALLER, never built here.

    ⚠ THIS IS WHAT KEEPS THIS MODULE TASK-AGNOSTIC, and the agnosticism is a
    build-time lever (see the header). It also means the table and the caller's
    dispatch are positionally coupled: index i in `tasks` must be the arm the
    caller runs for `st.task == i`, or the user clicks one robot and gets
    another — confusing precisely because everything still works.

    `domain_of[i]` is the index into `domains` of task i's group; the tree
    shows a task under that heading and nowhere else."""

    var handoff: Optional[RendererHandoff]
    """The live SDL window + GPU device BETWEEN two tasks — set only while no
    env owns them.

    ⚠ WITHOUT THIS THE WINDOW MOVES ITSELF. Destroying and re-creating it per
    switch means the OS re-places the replacement, which on a multi-monitor
    desktop puts it back on the PRIMARY display: drag the viewer to an external
    screen, pick another robot, and it jumps home. Carrying the window also
    keeps its size, its ImGui state (the sidebar's scroll position and expanded
    domains) and skips a re-blink plus shader/font/mesh rebuild.

    ⚠ SET MEANS UNOWNED, AND UNOWNED MEANS LEAKABLE. Whoever holds this must
    hand it to the next `init_renderer` or end it with
    `Renderer3D.close_handoff`; `run_viewer` is the one place that guarantees
    both."""
    var reset_meta_idx: List[Int]
    var reset_meta_val: List[Float64]
    """`meta` words written after EVERY reset, before `reset_qpos` is applied
    — empty for none. A task family's observation hook reads the task's TAPE
    (`META_IDX_TASK_PARAM_*`), its active MASK (`META_IDX_TASK_ACTIVE`) and
    its init words out of `d.meta`, and `Phyics3dEnv.reset` writes none of
    them: a family policy viewed through `run_view` alone gets goal words
    computed from an empty tape and slot words zeroed by a zero mask — an
    observation it never trained on, so it looks broken while the checkpoint
    is fine. `tasks/posed_reset.task_meta_words` fills these for a task."""
    var reset_qpos: List[Float64]
    """A full `qpos` the env is put into after EVERY reset — empty for none.

    ⚠ DEFAULTS TO EMPTY so every existing front end resets exactly as before.
    A task FAMILY parks its free slots 50 m up in `qpos0` and places them at
    reset on the HOST (`tasks/sampler` + `tasks/reset.reset_slots`), which
    `Phyics3dEnv.reset` knows nothing about: a family viewed through
    `run_view` alone shows props falling from the sky. A front end that
    knows the family computes the posed `qpos` once and puts it here; the
    viewer writes it through `obs_at` (state + observation, so a policy
    reads the posed state too) at the initial reset, the button, and every
    episode end. Its length must be the model's `nq`, or it is ignored with
    a printed line rather than a silently truncated state."""
    var pip_cameras: List[Int]
    """Model camera indices drawn as picture-in-picture insets at the right
    edge of the 3D viewport, stacked from the top — empty for none. A
    teleoperator on the free camera keeps the overhead and wrist views in
    sight without switching; applied after the renderer is built, like
    `free_camera`."""
    var frame_ms: Int
    """The frame period the loop paces to, ms. Defaults to `FRAME_TARGET_MS`
    (~60 fps). A teleoperation front end sets it to the env's CONTROL period
    so the sim runs at real time under the human — at 60 fps a 32 ms control
    step runs the world twice as fast as the hand driving it."""
    var reset_curriculum: List[Float64]
    """`Model.curriculum` words written ONCE, after the env is built — empty
    for none. A task family's region table lives there
    (`gpu_eval.region_table_words`), and the SAC driver uploads it beside the
    per-lane `meta`; a `StepObserver` that evaluates the family's reward on
    the host reads it from `mf`, so a front end that records must fill it.
    Its length must be `MODEL_CURRICULUM_SIZE` or it is ignored with a printed
    line."""
    var record_path: String
    """Record the 3D view (insets included, sidebar excluded) to this file
    from the first step, then close the viewer after `record_episodes`
    episodes — an unattended clip. Empty (the default) = the record button
    only, so every existing front end is unchanged. The frame rate is the
    loop's own, `1000 / frame_ms`, so a front end pacing to the control period
    records in real time; `ffmpeg` picks the codec from the extension."""
    var record_episodes: Int
    var free_camera_eye: List[Float64]
    var free_camera_target: List[Float64]
    """With `free_camera`, an exact free-camera pose (x, y, z each) instead of
    the automatic 3/4-view reframe — a framed shot for a recorded clip. Empty
    (the default) = the reframe, as before."""

    def __init__(
        out self,
        task: Int,
        drive: Int,
        scale: Float64,
        var tasks: List[String],
        var domains: List[String],
        var domain_of: List[Int],
    ):
        self.task = task
        self.drive = Int32(drive)
        self.scale = Float32(scale)
        self.quit = False
        self.filter = TextBuffer()
        self.show_plot = True
        self.policy_variant = 0
        self.policy_greedy = True
        self.free_camera = False
        self.episode_steps = EPISODE_STEPS
        self.tasks = tasks^
        self.domains = domains^
        self.domain_of = domain_of^
        self.handoff = None
        self.reset_meta_idx = List[Int]()
        self.reset_meta_val = List[Float64]()
        self.reset_qpos = List[Float64]()
        self.frame_ms = FRAME_TARGET_MS
        self.pip_cameras = List[Int]()
        self.reset_curriculum = List[Float64]()
        self.record_path = String("")
        self.record_episodes = 1
        self.free_camera_eye = List[Float64]()
        self.free_camera_target = List[Float64]()


@fieldwise_init
struct ViewControls(Copyable, Movable):
    """Renderer state IN, requests OUT — the panel never touches an env.

    Keeping it plain data is what lets the sidebar be ordinary non-generic
    code instead of something parameterised on the env's compile-time type.
    That is a compile-time lever, not only a tidiness one: see this module's
    header.
    """

    var n_cameras: Int
    var current_camera: Int
    var paused: Bool
    var recording: Bool
    var rec_frames: Int

    var want_camera: Int
    """-1 when no numbered camera button was pressed this frame."""
    var want_free_camera: Bool
    var want_screenshot: Bool
    var want_toggle_pause: Bool
    var want_toggle_record: Bool

    def __init__(out self, n_cameras: Int, current_camera: Int, paused: Bool,
                 recording: Bool, rec_frames: Int):
        self.n_cameras = n_cameras
        self.current_camera = current_camera
        self.paused = paused
        self.recording = recording
        self.rec_frames = rec_frames
        self.want_camera = -1
        self.want_free_camera = False
        self.want_screenshot = False
        self.want_toggle_pause = False
        self.want_toggle_record = False


@fieldwise_init
struct SidebarOut(Copyable, Movable):
    """What the user asked for this frame."""

    var picked: Int
    """Task id clicked in the tree, or -1."""
    var reset_episode: Bool
    var zero_now: Bool
    var pick_variant: Int
    """Policy variant chosen this frame, or -1 for "unchanged".

    A REQUEST, not the selection: loading weights is the front end's job, so the
    sidebar reports the click and `run_view` calls `choose`. `st.policy_variant`
    is only updated once that succeeded."""


# ═══════════════════════════════════════════════════════════════════════════
# UI components — plain functions over plain data, composed by `build_sidebar`
# ═══════════════════════════════════════════════════════════════════════════


def ui_status(name: String, episode: Int, step_i: Int, ep_return: Float64,
              obs_dim: Int, act_dim: Int) raises:
    """Header: which task, and how its episode is going."""
    ig_text_colored(name, 0.47, 0.90, 1.0)
    ig_text_disabled(
        String("obs ") + String(obs_dim) + String("   act ") + String(act_dim)
    )
    ig_text(
        String("ep ") + String(episode) + String("   step ") + String(step_i)
    )
    ig_text(String("return ") + _fmt2(ep_return))


def ui_reward_plot(history: List[Float32], cursor: Int) raises:
    """A sparkline of recent per-step reward.

    ⚠ THE CURSOR IS PASSED, NOT ROTATED AWAY. `ig_plot_lines` takes a ring
    buffer's write index as `offset` and reads from there, so the oldest sample
    is drawn leftmost with no copying.
    """
    ig_text_disabled(String("reward"))
    # ⚠ LABEL HIDDEN ("##" PREFIX), CAPTION DRAWN ABOVE. ImGui puts a plot's
    # label to its RIGHT, inside the same item — so a full-width plot pushes
    # the label off the panel and leaves a clipped letter at the edge.
    ig_plot_lines(
        String("##reward"), history, offset=cursor, lo=0.0, hi=1.0,
        w=-1.0, h=44.0,
    )


def ui_task_tree(
    mut st: ViewerState,
    tasks: List[String],
    domains: List[String],
    domain_of: List[Int],
    height: Float32,
) raises -> Int:
    """Filter box + domain tree. Returns the picked task id, or -1.

    A non-empty filter FLATTENS the tree: matching tasks are listed directly,
    because making the user expand a domain to reach a result they just named
    would defeat the search.
    """
    var picked = -1

    ig_set_next_item_width(-1.0)
    _ = ig_input_text(String("##filter"), st.filter, String("filter tasks..."))
    var query = st.filter.value()

    # The child is what makes 48 tasks a non-problem: it SCROLLS. The
    # hand-rolled sidebar had to cap the row count and print
    # "+N more — narrow the filter" instead.
    if ig_begin_child(String("tasks"), 0.0, height, True):
        if query.byte_length() > 0:
            var n_shown = 0
            for i in range(len(tasks)):
                if not _contains(tasks[i], query):
                    continue
                n_shown += 1
                if ig_selectable(tasks[i], i == st.task):
                    picked = i
            if n_shown == 0:
                ig_text_disabled(String("no match"))
        else:
            for d in range(len(domains)):
                # ⚠ ID PUSHED PER DOMAIN. ImGui identifies widgets by label; a
                # task whose name matched its domain's would otherwise collide
                # with the header and one of the two would stop responding.
                ig_push_id_int(d)
                if ig_tree_node(domains[d]):
                    for i in range(len(tasks)):
                        if domain_of[i] != d:
                            continue
                        if ig_selectable(tasks[i], i == st.task):
                            picked = i
                    ig_tree_pop()
                ig_pop_id()
    ig_end_child()
    return picked


def ui_drive_controls(
    mut st: ViewerState, has_policy: Bool = False
) raises -> SidebarOut:
    """Drive mode, action scale, and the two episode buttons."""
    ig_set_next_item_width(-60.0)
    _ = ig_combo(String("mode"), st.drive, drive_names(has_policy))

    ig_set_next_item_width(-60.0)
    # A real DRAG: the value streams for the whole gesture. This is the
    # capability the hand-rolled widget layer could not provide at all, which
    # is why scale used to be two nudge buttons.
    _ = ig_slider_float(String("scale"), st.scale, 0.0, 8.0, String("%.2f"))
    # ⚠ SAID OUT LOUD BECAUSE POLICY MODE IGNORES IT. Scaling a trained
    # policy's output is a torque-limit experiment, not a viewing control, and
    # inheriting a `scale 0.4` from argv would make a competent checkpoint look
    # broken with no visible cause.
    if Int(st.drive) == DRIVE_POLICY:
        ig_text_disabled(String("(scale drives random/sweep only)"))

    var half = (ig_content_width() - 8.0) * 0.5
    var out = SidebarOut(-1, False, False, -1)
    if ig_button(String("reset episode"), half, 0.0):
        out.reset_episode = True
    ig_same_line()
    if ig_button(String("zero torque"), half, 0.0):
        out.zero_now = True
    return out^


def ui_policy_controls(
    mut st: ViewerState, labels: List[String], status: String
) raises -> Int:
    """Variant picker + greedy toggle for the live `ActionSource`.

    Returns the variant to load, or -1. NON-GENERIC like the rest of the
    sidebar: it sees a label list and a status string, never the policy.
    """
    ig_separator_text(String("policy"))
    ig_text_disabled(status)

    var picked = -1
    if len(labels) > 1:
        # The combo is the direct jump; the two buttons are the sweep. Stepping
        # matters more than jumping here — walking the ladder rung by rung is
        # how the gait's emergence is actually read.
        ig_set_next_item_width(-60.0)
        var before = st.policy_variant
        _ = ig_combo(String("ckpt"), st.policy_variant, labels)
        if st.policy_variant != before:
            picked = Int(st.policy_variant)
            # ROLLED BACK HERE, RE-APPLIED BY `run_view` IF THE LOAD WORKED.
            # `ig_combo` writes the field directly, so a rung that fails to load
            # would otherwise leave the combo naming a policy that is not
            # driving — the one lie a status line cannot correct.
            st.policy_variant = before

        var half = (ig_content_width() - 8.0) * 0.5
        var cur = Int(st.policy_variant)
        if ig_button(String("- rung"), half, 0.0) and cur > 0:
            picked = cur - 1
        ig_same_line()
        if ig_button(String("+ rung"), half, 0.0) and cur + 1 < len(labels):
            picked = cur + 1

    _ = ig_checkbox(String("greedy (policy mean)"), st.policy_greedy)
    return picked


def ui_view_controls(mut vc: ViewControls) raises:
    """Camera selection and capture. Writes its answers back into `vc`."""
    vc.want_camera = -1
    vc.want_free_camera = False
    vc.want_screenshot = False
    vc.want_toggle_pause = False
    vc.want_toggle_record = False

    ig_text_disabled(String("camera"))
    # "free" is NOT one of the model's cameras — it is dm_control's -1, the
    # absence of one, and the only camera the mouse fully controls. It leads
    # the row because it is where dm_control's own viewer starts.
    #
    # Sized from the model, not fixed: dm_control models carry between one and
    # four cameras, plus this button.
    var n_slots = vc.n_cameras + 1
    var w = (ig_content_width() - Float32(n_slots - 1) * 8.0) / Float32(n_slots)
    if ig_toggle_button(String("free"), vc.current_camera < 0, w, 0.0):
        vc.want_free_camera = True
    for c in range(vc.n_cameras):
        ig_same_line()
        if ig_toggle_button(String(c + 1), c == vc.current_camera, w, 0.0):
            vc.want_camera = c

    # ⚠ "###" PINS THE ID WHILE THE TEXT VARIES. ImGui derives a widget's
    # identity from its LABEL, so a button whose caption changes is a DIFFERENT
    # widget each frame — the press and the release land on two different ids
    # and the click never completes.
    #
    # That is not theoretical: the record button showed a live frame count
    # ("rec 1", "rec 2", ...), so once recording started its id changed every
    # single frame and it could no longer be clicked to STOP. It would start a
    # recording and then refuse to end it; the only way out was switching task.
    #
    # Everything after "###" is the id and is not drawn; everything before is
    # drawn and does NOT contribute to the id (unlike "##", where the visible
    # part still does — which would not have helped here).
    var third = (ig_content_width() - 16.0) / 3.0
    if ig_toggle_button(
        String("resume###pause") if vc.paused else String("pause###pause"),
        vc.paused, third, 0.0,
    ):
        vc.want_toggle_pause = True
    ig_same_line()
    if ig_button(String("shot"), third, 0.0):
        vc.want_screenshot = True
    ig_same_line()
    if ig_toggle_button(
        (String("rec ") + String(vc.rec_frames) + String("###rec"))
        if vc.recording else String("rec###rec"),
        vc.recording, third, 0.0,
    ):
        vc.want_toggle_record = True


def build_sidebar(
    name: String,
    episode: Int,
    step_i: Int,
    ep_return: Float64,
    obs_dim: Int,
    act_dim: Int,
    history: List[Float32],
    cursor: Int,
    panel_h: Float32,
    mut st: ViewerState,
    tasks: List[String],
    domains: List[String],
    domain_of: List[Int],
    mut vc: ViewControls,
    has_policy: Bool = False,
    variant_labels: List[String] = List[String](),
    policy_status: String = String(""),
) raises -> SidebarOut:
    """The whole panel, in one NON-GENERIC function.

    ⚠ KEEP IT NON-GENERIC. `run_view` is instantiated once per task, so every
    line inside it is compiled 48 times; every line here is compiled once.
    Taking the env's state as plain arguments and returning requests is what
    buys that, and it is the difference between the sidebar being free and it
    dominating the build.
    """
    ig_begin_panel(String("dm_control"), 0.0, 0.0, SIDEBAR_W, panel_h)

    ui_status(name, episode, step_i, ep_return, obs_dim, act_dim)
    if st.show_plot:
        ui_reward_plot(history, cursor)

    ig_separator_text(String("task"))
    # The tree takes what is left after the fixed-height sections below it, so
    # the list grows with the window instead of being clipped by a hardcoded
    # row budget. The policy block adds three rows, so it takes its own bite.
    var reserved = Float32(500.0) if has_policy else Float32(400.0)
    var picked = ui_task_tree(st, tasks, domains, domain_of, panel_h - reserved)

    ig_separator_text(String("drive"))
    var out = ui_drive_controls(st, has_policy)
    out.picked = picked

    if has_policy:
        out.pick_variant = ui_policy_controls(st, variant_labels, policy_status)

    ig_separator_text(String("view"))
    ui_view_controls(vc)
    ig_spacing()
    ig_text_disabled(String("drag: orbit   shift+drag: pan"))

    ig_end()
    return out^


# ═══════════════════════════════════════════════════════════════════════════
# the viewer loop
# ═══════════════════════════════════════════════════════════════════════════


def run_view[
    MODEL: ModelDefLike,
    CONFIG: Phyics3dEnvConfig,
    POLICY: ActionSource = NoPolicy,
    OBSERVER: StepObserver = NoObserver,
](
    name: String,
    mut st: ViewerState,
    policy: Optional[Pointer[POLICY, MutAnyOrigin]] = None,
    observer: Optional[Pointer[OBSERVER, MutAnyOrigin]] = None,
) raises:
    """The viewer loop for one task, running until it quits or switches.

    ⚠ `policy` IS OWNED BY THE FRONT END, and must be, for two reasons: it holds
    network weights that would be reloaded on every task switch if it died with
    the env, and it is the only part of the viewer that depends on
    `deep_agents` — keeping it outside means the 48-task `dispatch` never
    compiles a policy at all (`POLICY` defaults to the no-op `NoPolicy`).

    Returns through `st`: either `st.quit`, or `st.task` now naming a DIFFERENT
    task, which `run_viewer` then launches. It cannot call itself with the new
    task — each task is a separate compile-time instantiation, so recursion
    would try to instantiate all 48 inside all 48.

    ⚠ PARAMETERISED ON MODEL+CONFIG, NOT ON THE ENV TYPE. The obvious
    factoring — `def run_view[E: BoxContinuousActionEnv & RenderableEnv](mut
    env: E)` — does not compile: `E.ActionType()` cannot be constructed through
    the trait bound ("no matching function in initialization"). Building the
    concrete `Phyics3dEnv[...]` inside makes `E.ActionType` a real type again.

    ⚠ SWITCHING TASKS REBUILDS THE ENV, NOT THE WINDOW. `Phyics3dEnv[MODEL,
    CONFIG]` is a distinct type per task and its renderer dies with it, but the
    SDL window and GPU device are handed across the gap in `st.handoff` and
    ADOPTED here — so a switch keeps the window's monitor, position, size and
    ImGui state, and re-does only what is model-specific (cameras, lights,
    skybox, the geom mesh caches). Expect the CAMERA to reframe, since it is
    the model's; expect nothing else to move.

    ⚠ ADOPTING TAKES OWNERSHIP. `st.handoff` is cleared the moment the env has
    it, and re-filled only by `detach_renderer` on the way out — so exactly one
    party can free the window at any time.

    `observer` (a `StepObserver`, default none) sees every step with the env's
    `Data` and every episode end — the recorder's seam. Like `policy` it is
    owned by the front end and only lent here.
    """
    comptime E = Phyics3dEnv[MODEL, CONFIG, DT, False]
    comptime ACT_DIM = E.ACTION_DIM

    print("=" * 66)
    print("dm_control viewer (ImGui) —", name)
    print("=" * 66)
    print("  obs dim    =", E.OBS_DIM)
    print("  action dim =", ACT_DIM)
    print("  pick another task in the window, or close it to quit")

    var ctx = DeviceContext()
    var env = E(ctx)
    if len(st.reset_curriculum) > 0:
        # `ViewerState.reset_curriculum`: the family's region table, once.
        if len(st.reset_curriculum) != len(env.mf.curriculum.data):
            print("  reset_curriculum has", len(st.reset_curriculum),
                  "words but the model's curriculum has",
                  len(env.mf.curriculum.data), "— ignored")
        else:
            for i in range(len(st.reset_curriculum)):
                env.mf.curriculum.data[i] = Scalar[DT](st.reset_curriculum[i])

    # The live observation, kept in step with the env so the policy always reads
    # the state it is about to act on.
    var obs_l = List[Scalar[DT]](length=E.OBS_DIM, fill=Scalar[DT](0))
    var act_l = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0))
    var prev_obs_l = List[Scalar[DT]](length=E.OBS_DIM, fill=Scalar[DT](0))
    var act_f = List[Float64](length=ACT_DIM, fill=0.0)
    var have_obs = Bool(observer)
    var s0 = env.reset()
    for i in range(E.OBS_DIM):
        obs_l[i] = Scalar[DT](s0.data[i])
    for mk in range(len(st.reset_meta_idx)):
        env.d.meta.data[st.reset_meta_idx[mk]] = Scalar[DT](st.reset_meta_val[mk])
    if len(st.reset_qpos) > 0:
        # `ViewerState.reset_qpos`: the posed state AND its observation.
        if len(st.reset_qpos) != MODEL.NQ:
            print("  reset_qpos has", len(st.reset_qpos),
                  "words but the model's nq is", MODEL.NQ, "— ignored")
        else:
            var v0 = List[Float64](length=MODEL.NV, fill=0.0)
            var sp = env.obs_at(st.reset_qpos, v0)
            for i in range(E.OBS_DIM):
                obs_l[i] = Scalar[DT](sp.data[i])

    # ── policy plug ──────────────────────────────────────────────────────
    var have_pol = Bool(policy)
    var pol_labels = List[String]()
    if have_pol:
        var p = policy.value()
        # ⚠ RUNTIME DIM CHECK, because the trait deals in host LISTS and cannot
        # express the constraint at compile time. Without it a walker policy
        # driving a cheetah reads 6 observations past the end and writes an
        # action list that is the wrong length — a segfault at best, plausible
        # nonsense at worst.
        if p[].obs_dim() != E.OBS_DIM or p[].act_dim() != ACT_DIM:
            print(
                "  ⚠ policy is", p[].obs_dim(), "obs /", p[].act_dim(),
                "act but this task is", E.OBS_DIM, "/", ACT_DIM,
                "— POLICY MODE DISABLED here",
            )
            have_pol = False
        else:
            pol_labels = p[].variant_labels()
            if Int(st.policy_variant) >= len(pol_labels):
                st.policy_variant = Int32(
                    len(pol_labels) - 1 if len(pol_labels) > 0 else 0
                )
            try:
                p[].choose(Int(st.policy_variant))
            except e:
                print("  policy variant failed to load:", e)
            print("  policy:", p[].status())
    if not have_pol and Int(st.drive) == DRIVE_POLICY:
        # The combo only lists the modes that exist, so an out-of-range index
        # would leave it displaying a blank entry no click could correct.
        st.drive = Int32(DRIVE_SWEEP)

    var adopt = st.handoff.copy()
    # Cleared BEFORE the call, not after: if `init_renderer` raised with `st`
    # still naming the window the env now owns, the error path below would
    # close it a second time.
    st.handoff = None
    if not env.init_renderer(True, adopt):
        print("No renderer available — is SDL3 present?")
        if adopt:
            Renderer3D.close_handoff(adopt.value().copy())
        st.quit = True
        return

    # ⚠ AFTER `init_renderer`, and per task rather than once: the env owns the
    # model renderer, so a task switch rebuilds it with `active_camera = 0` and
    # the choice has to be re-applied. The reframe itself is deferred to the
    # first `render`, which is where the camera's current distance — the only
    # model-scale information available — is still on hand.
    if st.free_camera:
        if len(st.free_camera_eye) == 3 and len(st.free_camera_target) == 3:
            env.renderer_set_free_camera(
                Vec3Generic[DType.float64](
                    st.free_camera_eye[0], st.free_camera_eye[1],
                    st.free_camera_eye[2],
                ),
                Vec3Generic[DType.float64](
                    st.free_camera_target[0], st.free_camera_target[1],
                    st.free_camera_target[2],
                ),
            )
        else:
            env.renderer_request_free_camera()
    if len(st.pip_cameras) > 0:
        env.renderer_set_pip_cameras(st.pip_cameras.copy())

    # Idempotent, and on the adopt path it is a no-op that returns True: the
    # ImGui context is attached to the window and device, both of which just
    # came across intact.
    var have_ui = env.imgui_init()
    if not have_ui:
        # The scene is perfectly usable without a sidebar, so this degrades
        # rather than quitting. Note the strip is NOT reserved and the HUD is
        # NOT hidden on this path: reserving space for a panel that will never
        # be drawn would waste a fifth of the window, and the HUD is then the
        # only remaining UI.
        print("  (continuing without the ImGui sidebar)")
    else:
        ig_style_dark()
        # The built-in HUD lists the same keybinds and reports the same camera,
        # step and pause state the sidebar shows — and it paints them OVER the
        # robot. With a panel up it is pure occlusion.
        env.renderer_set_show_hud(False)
        # Reserve the strip. This INSETS the 3D viewport and corrects the
        # camera aspect, so the scene sits beside the panel rather than behind
        # it — a panel over an uninset scene would hide a fifth of the robot.
        env.set_ui_sidebar_width(Int(SIDEBAR_W))

    # Copied out ONCE, not read from `st` per frame: `build_sidebar` takes `st`
    # mutably, so handing it `st.tasks` in the same call would be a borrow of a
    # field of the thing being mutated.
    var tasks = st.tasks.copy()
    var domains = st.domains.copy()
    var domain_of = st.domain_of.copy()
    var held = E.ActionType()
    var step_i = 0
    var episode = 0
    var ep_return = Float64(0)
    var switching = False

    var history = List[Float32]()
    for _ in range(PLOT_N):
        history.append(0.0)
    var cursor = 0

    var frame_t0 = perf_counter_ns()

    # `ViewerState.record_path`: an unattended clip. Started after the camera
    # and inset requests so the first frame is already the requested view.
    var auto_record = st.record_path.byte_length() > 0
    var record_done = False
    if auto_record:
        var fps = max(1, (1000 + st.frame_ms // 2) // max(1, st.frame_ms))
        env.start_recording(st.record_path, fps, 1)

    while env.is_renderer_open():
        # Pump events FIRST: ImGui drains the queue in its NewFrame, so events
        # polled here are the ones this frame's widgets react to.
        if env.check_renderer_quit():
            break

        env.imgui_new_frame()

        var vc = ViewControls(
            env.renderer_n_cameras(), env.renderer_current_camera(),
            env.renderer_paused(), env.renderer_is_recording(),
            env.renderer_recording_frames(),
        )
        var ui = SidebarOut(-1, False, False, -1)
        if have_ui:
            var pol_status = String("")
            if have_pol:
                pol_status = policy.value()[].status()
            ui = build_sidebar(
                name, episode, step_i, ep_return, E.OBS_DIM, ACT_DIM,
                history, cursor, Float32(E.RENDER_HEIGHT), st,
                tasks, domains, domain_of, vc,
                have_pol, pol_labels, pol_status,
            )

        # ── apply what the UI asked for ──────────────────────────────────
        if have_pol and ui.pick_variant >= 0:
            # `st.policy_variant` advances ONLY on a load that worked, so a
            # missing rung leaves the previous policy driving instead of
            # silently switching to an uninitialised net.
            try:
                policy.value()[].choose(ui.pick_variant)
                st.policy_variant = Int32(ui.pick_variant)
            except e:
                print("  policy variant failed to load:", e)
        if ui.reset_episode:
            if have_obs:
                observer.value()[].on_episode_end(step_i, True)
            var sr = env.reset()
            for i in range(E.OBS_DIM):
                obs_l[i] = Scalar[DT](sr.data[i])
            for mk in range(len(st.reset_meta_idx)):
                env.d.meta.data[st.reset_meta_idx[mk]] = Scalar[DT](st.reset_meta_val[mk])
            if len(st.reset_qpos) > 0:
                # `ViewerState.reset_qpos`: the posed state AND its observation.
                if len(st.reset_qpos) != MODEL.NQ:
                    print("  reset_qpos has", len(st.reset_qpos),
                          "words but the model's nq is", MODEL.NQ, "— ignored")
                else:
                    var v0 = List[Float64](length=MODEL.NV, fill=0.0)
                    var sp = env.obs_at(st.reset_qpos, v0)
                    for i in range(E.OBS_DIM):
                        obs_l[i] = Scalar[DT](sp.data[i])
            step_i = 0
            ep_return = 0.0
        elif ui.zero_now:
            st.drive = Int32(DRIVE_ZERO)
            st.scale = 1.0
        if vc.want_free_camera:
            env.renderer_request_free_camera()
            st.free_camera = True
        if vc.want_camera >= 0:
            env.renderer_request_camera(vc.want_camera)
            st.free_camera = False
        if vc.want_screenshot:
            env.renderer_request_screenshot()
        if vc.want_toggle_pause:
            env.renderer_toggle_pause()
        if vc.want_toggle_record:
            # ⚠ CATCH: video encoding spawns `ffmpeg`, and a failure to do so
            # raised straight out of the frame loop and killed the viewer.
            # Screenshots already degrade to a printed message; recording must
            # not be the one button that can end the session.
            try:
                env.renderer_toggle_recording()
            except e:
                print("recording unavailable:", e)
                print(
                    "  `ffmpeg` lives in the pixi env — launch through"
                    " `pixi run`, not the bare binary"
                )

        # Leave BEFORE stepping: the new task's env is built by `run_viewer`,
        # and there is nothing to gain from one more frame of the old one.
        #
        # ⚠ `render_frame` STILL RUNS below on the switching frame, because
        # `imgui_new_frame` already opened an ImGui frame and only `end_frame`
        # closes it. Breaking here instead would leave that frame open and the
        # NEXT task's first `NewFrame` would assert.
        if ui.picked >= 0 and ui.picked != st.task:
            st.task = ui.picked
            switching = True

        # ⚠ PAUSE MUST GATE THE STEP, not merely label the button. Both this
        # viewer and the ui.mojo one read `renderer_paused()` only to choose
        # between "pause" and "resume" and then stepped anyway, so the button
        # toggled a flag that nothing downstream honoured — the renderer's own
        # `is_paused` only ever froze its HUD step counter.
        #
        # `renderer_step_once()` is the RIGHT-arrow single-step: true for
        # exactly the one frame after the key, so a paused sim advances one
        # step and stops. It is reset at the top of every `check_quit`, which
        # this loop calls once per iteration, so reading it here is correct.
        var stepping = not switching
        if stepping and env.renderer_paused() and not env.renderer_step_once():
            stepping = False

        var action = E.ActionType()
        if stepping:
            if Int(st.drive) == DRIVE_RANDOM:
                if step_i % HOLD_STEPS == 0:
                    for a in range(ACT_DIM):
                        held.data[a] = (
                            random_float64(-1.0, 1.0) * Float64(st.scale)
                        )
                for a in range(ACT_DIM):
                    action.data[a] = held.data[a]
            elif Int(st.drive) == DRIVE_SWEEP:
                var t = Float64(step_i) / SWEEP_PERIOD
                for a in range(ACT_DIM):
                    var phase = Float64(a) / Float64(
                        ACT_DIM if ACT_DIM > 0 else 1
                    )
                    action.data[a] = (
                        sin(2.0 * pi * (t + phase)) * Float64(st.scale)
                    )
            elif Int(st.drive) == DRIVE_POLICY and have_pol:
                # NOT scaled by `st.scale` — see `ui_drive_controls`.
                policy.value()[].act(obs_l, act_l, st.policy_greedy)
                for a in range(ACT_DIM):
                    action.data[a] = Float64(act_l[a])

            if have_obs:
                for i in range(E.OBS_DIM):
                    prev_obs_l[i] = obs_l[i]
                for a in range(ACT_DIM):
                    act_f[a] = action.data[a]
            var out = env.step(action)
            for i in range(E.OBS_DIM):
                obs_l[i] = Scalar[DT](out[0].data[i])
            ep_return += Float64(out[1])
            history[cursor] = Float32(out[1])
            cursor = (cursor + 1) % PLOT_N
            step_i += 1

            # `N` for a new episode. The renderer claims ESC/SPACE/R/S/V/1-9
            # for its own bindings and forwards the rest, so this is a free
            # key — and a manual reset is the only kind that makes sense when
            # a human, not a policy, is driving.
            var key = env.renderer_take_key()
            var manual_reset = key == KEY_N
            # The observer sees the step with the env's state still the
            # post-step one, and may ask for the episode to end here.
            if have_obs:
                if observer.value()[].on_step(
                    env.d, env.mf, prev_obs_l, act_f, obs_l, out[2],
                    step_i, key,
                ):
                    manual_reset = True

            var timed_out = st.episode_steps > 0 and step_i >= st.episode_steps
            if out[2] or timed_out or manual_reset:
                episode += 1
                print("  episode", episode, "ended after", step_i,
                      "steps, return =", ep_return,
                      "(manual)" if manual_reset else "")
                if have_obs:
                    observer.value()[].on_episode_end(step_i, manual_reset)
                if auto_record and episode >= st.record_episodes:
                    # Stopped BEFORE this frame renders the reset pose, so the
                    # clip ends on the episode's last state; the loop leaves
                    # after `render_frame`, which closes the open ImGui frame.
                    env.stop_recording()
                    record_done = True
                var sr = env.reset()
                for i in range(E.OBS_DIM):
                    obs_l[i] = Scalar[DT](sr.data[i])
                for mk in range(len(st.reset_meta_idx)):
                    env.d.meta.data[st.reset_meta_idx[mk]] = Scalar[DT](st.reset_meta_val[mk])
                if len(st.reset_qpos) > 0:
                    # `ViewerState.reset_qpos`: the posed state AND its observation.
                    if len(st.reset_qpos) != MODEL.NQ:
                        print("  reset_qpos has", len(st.reset_qpos),
                              "words but the model's nq is", MODEL.NQ, "— ignored")
                    else:
                        var v0 = List[Float64](length=MODEL.NV, fill=0.0)
                        var sp = env.obs_at(st.reset_qpos, v0)
                        for i in range(E.OBS_DIM):
                            obs_l[i] = Scalar[DT](sp.data[i])
                step_i = 0
                ep_return = 0.0

        env.render_frame()
        if switching or record_done:
            break

        # Sleep only what is LEFT of the target period. `frame_t0` is taken
        # after the sleep, so the next frame's budget starts when this one's
        # work does — measuring from before the sleep would charge each frame
        # for the previous frame's idle.
        var spent_ms = Int((perf_counter_ns() - frame_t0) // 1_000_000)
        if spent_ms < st.frame_ms:
            env.renderer_delay(st.frame_ms - spent_ms)
        frame_t0 = perf_counter_ns()

    st.quit = not switching
    if switching:
        # KEEP the window: only this env's model-specific GPU state goes.
        st.handoff = env.detach_renderer()
        print("  switching to", tasks[st.task])
    else:
        # Real close — the renderer still owns the window, so the ordinary
        # teardown is the correct one and `st.handoff` stays empty.
        env.close_renderer()
        print("viewer closed")
