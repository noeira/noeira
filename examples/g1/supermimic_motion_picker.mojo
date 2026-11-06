# +--------------------------------------------------------------------------+ #
# | SuperMimic on mjlab's G1 — pick a motion from a list, watch it tracked
# +--------------------------------------------------------------------------+ #
"""An ImGui front end for a SuperMimic checkpoint: a filterable, per-category
list of motions; click one and the robot tracks it from its first frame,
the benchmark's protocol (Appendix F: no noise, no domain randomisation, no
episode cap), live on the CPU env.

    pixi run build-imgui                                   # ONCE
    pixi run mojo build -I . -Xlinker -ld_classic examples/g1/supermimic_motion_picker.mojo -o /tmp/smpick
    pixi run /tmp/smpick --ckpt <last.ckpt> --store seed_demo.h5 --catalog demo_catalog.tsv [--start NAME] [--keep-going]

`--store` is ONE store whose episode i is the catalogue's row i: build both
with `tools/supermimic/demo_catalog.py` (one take per SEED motion, joined
with SEED's metadata and the GPU benchmark's per-clip scores) and
`supermimic_demo_store.mojo` (copies those takes out of the 260 SEED
stores, ~0.44 GB). The two are checked against each other at start: same
clip count, same names, same frame counts.

THE SIDEBAR: the selected motion (name, category, length, SEED's
description, and its score in the GPU benchmark of the full set), the live
run (progress, normalised return so far, per-step reward), playback
(restart, previous / next, loop, auto-next, continue after failure,
speed), the motion list (filter
box; a non-empty filter flattens the category tree), and the view (camera,
pause, screenshot, recording).

THE NUMBERS: the live normalised return is Σ r / (0.1 · steps), 1 =
perfect, as the benchmark scores a clip; "FAILED" is the benchmark's
termination. The catalogue's "eval" column is the GPU benchmark's number
for the same take. "continue after failure" turns the termination off
(`SMTrackingEnv.terminations`): the run goes on to the clip's end, the
step the benchmark would have stopped at is shown with its score, and the
return to the end is not the benchmark's (no training episode went past a
failure, so expect the robot on the floor).
⚠ CPU PHYSICS. The env is the float64 single-lane path; the checkpoint
trained against the float32 batched one. The two agree to 1e-7 over a step
(`test_supermimic_env_gpu` [2]) but long clips diverge, so the live number
can differ from the eval column: judge the motion, take the benchmark's
numbers from `supermimic_eval.mojo`.
⚠ Paper widths only (the 320k checkpoint's); `-D SM_PAPER=0` for a
smoke-width one. mjlab's G1 only (`supermimic_viewer.mojo -D SM_BFM_G1=1`
plays BFM-Zero G1 checkpoints).
⚠ The robot is drawn; the reference is not (no ghost model yet).
⚠ SEED is confidential: run on a private machine, never publish the frames.
"""

from std.sys import argv, get_defined_int
from std.time import perf_counter_ns

from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, SMDims
from noeira.envs.robots import UnitreeG1Mjlab
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.render.imgui import (
    TextBuffer, ig_begin_child, ig_begin_panel, ig_button, ig_checkbox, ig_content_width, ig_end,
    ig_end_child, ig_input_text, ig_is_item_hovered, ig_plot_lines, ig_pop_id, ig_progress_bar, ig_push_id_int, ig_same_line,
    ig_selectable, ig_separator_text, ig_set_next_item_open, ig_set_next_item_width, ig_set_tooltip, ig_slider_float,
    ig_spacing, ig_style_dark, ig_text, ig_text_colored, ig_text_disabled, ig_text_wrapped,
    ig_toggle_button, ig_tree_node, ig_tree_pop, imgui_shim_available,
)


comptime P = get_defined_int["SM_PAPER", 1]() != 0
comptime Robot = G1MjlabRobot
# the training example's widths (a checkpoint loads only into its own), one lane
comptime C = SMConfig[
    "cpu", Robot,
    2048 if P else 128, 64 if P else 64, 2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32,
    2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32, 256 if P else 64, 128 if P else 32,
    1, 64, 16, 32,
]
comptime D = SMDims[Robot]

comptime SIDEBAR_W: Float32 = 380.0
comptime PLOT_N = 150          # 3 s of per-step reward at 50 Hz
comptime CONTROL_MS = 20.0     # one control step = 20 ms of motion
comptime MAX_FLAT = 400        # filtered rows drawn at most


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _fmt(v: Float64, digits: Int = 2) -> String:
    """`v` with `digits` decimals (the stdlib prints 17 significant digits)."""
    if v != v:
        return String("-")
    var scale = 1
    for _ in range(digits):
        scale *= 10
    var neg = v < 0
    var a = -v if neg else v
    var r = Int(a * Float64(scale) + 0.5)
    var frac = String(r % scale)
    while frac.byte_length() < digits:
        frac = "0" + frac
    return (String("-") if neg else String("")) + String(r // scale) + (String(".") + frac if digits > 0 else String(""))


def _lower(s: String) -> String:
    return s.lower()


struct Catalog(Movable, Sized):
    """`demo_catalog.tsv`, one row per clip of the demo store, in its order."""

    var name: List[String]
    var label: List[String]      # the name without its actor suffix
    var category: List[String]
    var frames: List[Int]
    var eval_ret: List[Float64]
    var eval_failed: List[Bool]
    var desc: List[String]
    var search: List[String]     # lower-case name + category + description
    var cats: List[String]       # categories, in catalogue order
    var cat_rows: List[List[Int]]

    def __init__(out self, path: String) raises:
        self.name = List[String]()
        self.label = List[String]()
        self.category = List[String]()
        self.frames = List[Int]()
        self.eval_ret = List[Float64]()
        self.eval_failed = List[Bool]()
        self.desc = List[String]()
        self.search = List[String]()
        self.cats = List[String]()
        self.cat_rows = List[List[Int]]()
        var text: String
        with open(path, "r") as f:
            text = f.read()
        var lines = text.split("\n")
        for li in range(1, len(lines)):
            var ln = String(lines[li])
            if ln.byte_length() == 0:
                continue
            var p = ln.split("\t")
            if len(p) != 8:
                raise Error("catalogue line " + String(li + 1) + ": " + String(len(p)) + " fields, want 8")
            var nm = String(p[1])
            var cut = nm.find("__")
            self.name.append(nm.copy())
            self.label.append(String(nm[byte = 0 : cut]) if cut > 0 else nm.copy())
            self.category.append(String(p[2]))
            self.frames.append(Int(String(p[4])))
            var er = String(p[5])
            self.eval_ret.append(Float64(er) if er != "nan" else Float64(0) / Float64(0))
            self.eval_failed.append(String(p[6]) == "1")
            self.desc.append(String(p[7]))
            self.search.append(_lower(nm + " " + String(p[2]) + " " + String(p[7])))
            var c = String(p[2])
            var k = -1
            for j in range(len(self.cats)):
                if self.cats[j] == c:
                    k = j
            if k < 0:
                self.cats.append(c.copy())
                self.cat_rows.append(List[Int]())
                k = len(self.cats) - 1
            self.cat_rows[k].append(len(self.name) - 1)

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.label = move.label^
        self.category = move.category^
        self.frames = move.frames^
        self.eval_ret = move.eval_ret^
        self.eval_failed = move.eval_failed^
        self.desc = move.desc^
        self.search = move.search^
        self.cats = move.cats^
        self.cat_rows = move.cat_rows^

    def __len__(self) -> Int:
        return len(self.name)

    def row_label(self, i: Int) -> String:
        var s = self.label[i] + "  " + _fmt(self.eval_ret[i])
        if self.eval_failed[i]:
            s += " FAILED"
        return s + "##" + String(i)


def _library(path: String, cat: Catalog) raises -> SMMotionLibrary[Robot]:
    """Every clip of the store, checked against the catalogue row by row."""
    var st = TrajectoryStore(path.copy())
    if st.n_episodes() != len(cat):
        raise Error(path + " has " + String(st.n_episodes()) + " clips, the catalogue " + String(len(cat)))
    var lens = List[Int]()
    for e in range(st.n_episodes()):
        var l = Int(st.episodes.ep_len[e])
        if l != cat.frames[e]:
            raise Error("clip " + String(e) + ": " + String(l) + " frames in the store, " + String(cat.frames[e]) + " in the catalogue")
        var nm = st.manifest.task_text(e)
        if nm != cat.name[e]:
            raise Error("clip " + String(e) + " is " + nm + " in the store, " + cat.name[e] + " in the catalogue")
        lens.append(l)
    var refc = st.load_column[DType.float32](String("ref"))
    var rows = List[Scalar[DT]](length=len(refc), fill=0)
    for i in range(len(refc)):
        rows[i] = refc[i]
    var q0 = List[Float64](length=Robot.NDOF, fill=0.0)
    for j in range(Robot.NDOF):
        q0[j] = Robot.default_joint_pos(j)
    return SMMotionLibrary[Robot](rows^, lens, q0)


@fieldwise_init
struct UiOut(Copyable, Movable):
    var pick: Int             # clip clicked, or -1
    var restart: Bool
    var camera: Int           # -1 none, -2 free, else the camera index
    var pause: Bool
    var shot: Bool
    var record: Bool


struct Ui(Movable):
    """The sidebar's state between frames."""

    var filter: TextBuffer
    var loop: Bool
    var auto_next: Bool
    var keep_going: Bool         # continue after failure: no termination
    var speed: Float32
    var history: List[Float32]
    var cursor: Int

    def __init__(out self):
        self.filter = TextBuffer()
        self.loop = True
        self.auto_next = False
        self.keep_going = False
        self.speed = 1.0
        self.history = List[Float32](length=PLOT_N, fill=0.0)
        self.cursor = 0

    def __init__(out self, *, deinit move: Self):
        self.filter = move.filter^
        self.loop = move.loop
        self.auto_next = move.auto_next
        self.keep_going = move.keep_going
        self.speed = move.speed
        self.history = move.history^
        self.cursor = move.cursor


def _sidebar(
    mut ui: Ui, cat: Catalog, sel: Int, step: Int, ret: Float64, fail_step: Int, fail_ret: Float64,
    last: String, panel_h: Float32,
    n_cameras: Int, current_camera: Int, paused: Bool, recording: Bool, rec_frames: Int, sim_hz: Float64,
) raises -> UiOut:
    """The whole panel; returns this frame's requests (NON-GENERIC, built once)."""
    var out = UiOut(-1, False, -1, False, False, False)
    ig_begin_panel(String("SuperMimic"), 0.0, 0.0, SIDEBAR_W, panel_h)

    # ── the selected motion ────────────────────────────────────────────
    ig_text_colored(cat.label[sel], 0.47, 0.90, 1.0)
    ig_text_disabled(cat.category[sel] + "   " + String(cat.frames[sel]) + " frames ("
                     + _fmt(Float64(cat.frames[sel]) / 50.0, 1) + " s)")
    ig_text_wrapped(cat.desc[sel])
    ig_text(String("eval (GPU, full set): ") + _fmt(cat.eval_ret[sel]) + (String(" FAILED") if cat.eval_failed[sel] else String(" completed")))

    # ── the live run ──────────────────────────────────────────────────
    ig_separator_text(String("live"))
    var n = cat.frames[sel]
    ig_progress_bar(Float32(min(step, n)) / Float32(max(n, 1)), -1.0, 0.0,
                    String(step) + " / " + String(n))
    ig_text(String("normalised return ") + _fmt(ret / (0.1 * Float64(max(step, 1)))))
    if fail_step > 0:
        ig_text_colored(String("benchmark: FAILED at step ") + String(fail_step) + " ("
                        + _fmt(fail_ret / (0.1 * Float64(fail_step))) + "), continuing", 1.0, 0.45, 0.35)
    ig_text_disabled(String("per-step reward (max 0.1)"))
    ig_plot_lines(String("##r"), ui.history, offset=ui.cursor, lo=0.0, hi=0.1, w=-1.0, h=40.0)
    if last.byte_length() > 0:
        ig_text_disabled(String("last run: ") + last)
    ig_text_disabled(String("sim ") + _fmt(sim_hz, 0) + " steps/s (real time = 50)")

    # ── playback ──────────────────────────────────────────────────────
    ig_separator_text(String("playback"))
    var third = (ig_content_width() - 16.0) / 3.0
    if ig_button(String("< prev"), third, 0.0):
        out.pick = (sel - 1 + len(cat)) % len(cat)
    ig_same_line()
    if ig_button(String("restart"), third, 0.0):
        out.restart = True
    ig_same_line()
    if ig_button(String("next >"), third, 0.0):
        out.pick = (sel + 1) % len(cat)
    _ = ig_checkbox(String("loop"), ui.loop)
    ig_same_line()
    _ = ig_checkbox(String("auto-next"), ui.auto_next)
    _ = ig_checkbox(String("continue after failure"), ui.keep_going)
    if ig_is_item_hovered():
        ig_set_tooltip(String("Keep tracking past the benchmark's failure (height off by > 25 cm or tilt > 0.8)\n"
                              "to the clip's end. Training never saw those states: expect a fall."))
    ig_set_next_item_width(-60.0)
    _ = ig_slider_float(String("speed"), ui.speed, 0.1, 2.0, String("%.2fx"))

    # ── the motion list ───────────────────────────────────────────────
    ig_separator_text(String("motions (") + String(len(cat)) + ")")
    ig_set_next_item_width(-1.0)
    _ = ig_input_text(String("##filter"), ui.filter, String("filter: name, category, description..."))
    var query = _lower(ui.filter.value())
    if ig_begin_child(String("list"), 0.0, panel_h - 640.0, True):
        if query.byte_length() > 0:
            var shown = 0
            var hits = 0
            for i in range(len(cat)):
                if cat.search[i].find(query) < 0:
                    continue
                hits += 1
                if shown < MAX_FLAT:
                    shown += 1
                    if ig_selectable(cat.row_label(i), i == sel):
                        out.pick = i
            if hits == 0:
                ig_text_disabled(String("no match"))
            elif hits > shown:
                ig_text_disabled(String("+") + String(hits - shown) + " more: narrow the filter")
        else:
            for k in range(len(cat.cats)):
                ig_push_id_int(k)
                var has_sel = False
                for i in cat.cat_rows[k]:
                    if i == sel:
                        has_sel = True
                if has_sel:
                    ig_set_next_item_open(True)
                if ig_tree_node(cat.cats[k] + " (" + String(len(cat.cat_rows[k])) + ")"):
                    for i in cat.cat_rows[k]:
                        if ig_selectable(cat.row_label(i), i == sel):
                            out.pick = i
                    ig_tree_pop()
                ig_pop_id()
    ig_end_child()

    # ── the view ──────────────────────────────────────────────────────
    ig_separator_text(String("view"))
    var slots = n_cameras + 1
    var w = (ig_content_width() - Float32(slots - 1) * 8.0) / Float32(slots)
    if ig_toggle_button(String("free"), current_camera < 0, w, 0.0):
        out.camera = -2
    for c in range(n_cameras):
        ig_same_line()
        if ig_toggle_button(String(c + 1), c == current_camera, w, 0.0):
            out.camera = c
    if ig_toggle_button(String("resume###pause") if paused else String("pause###pause"), paused, third, 0.0):
        out.pause = True
    ig_same_line()
    if ig_button(String("shot"), third, 0.0):
        out.shot = True
    ig_same_line()
    if ig_toggle_button(
        (String("rec ") + String(rec_frames) + "###rec") if recording else String("rec###rec"),
        recording, third, 0.0,
    ):
        out.record = True
    ig_spacing()
    ig_text_disabled(String("drag: orbit   shift+drag: pan"))
    ig_end()
    return out^


def main() raises:
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return
    var raw = argv()
    var args = List[String]()
    for i in range(len(raw)):
        args.append(String(raw[i]))
    var ckpt = _opt(args, String("--ckpt"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var catalog = _opt(args, String("--catalog"), String(""))
    var start = _opt(args, String("--start"), String(""))
    if ckpt == "" or store == "" or catalog == "":
        raise Error("give --ckpt PATH --store DEMO_STORE.h5 --catalog DEMO_CATALOG.tsv [--start NAME] [--keep-going]")

    var cat = Catalog(catalog)
    print("catalogue:", len(cat), "motions in", len(cat.cats), "categories")
    var trainer = SuperMimicTrainer[C].for_robot(Robot.SIM.create())
    var ctr = trainer.load_state(ckpt)
    print("checkpoint at iteration", ctr[0], "(", ctr[1], "updates )")
    var env = SMTrackingEnv[Robot](
        1, _library(store, cat), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, max_episode_steps=0,
    )
    print("library:", env.lib.num_clips, "clips,", env.lib.num_frames, "frames")
    var rb = SMReplayOf[Robot](1, 64)

    var sel = 0
    if start.byte_length() > 0:
        for i in range(len(cat)):
            if cat.name[i].find(start) >= 0:
                sel = i
                break
    var first: List[Int] = [env.lib.clip_start[sel]]
    env.restart(first)

    if not env.lanes[0].sim.env.init_renderer(True):
        raise Error("no window (the renderer did not open)")
    var have_ui = env.lanes[0].sim.env.imgui_init()
    if have_ui:
        ig_style_dark()
        env.lanes[0].sim.env.renderer_set_show_hud(False)
        env.lanes[0].sim.env.set_ui_sidebar_width(Int(SIDEBAR_W))
    else:
        print("  (no ImGui sidebar: `pixi run build-imgui`)")

    var ui = Ui()
    for i in range(len(args)):
        if args[i] == "--keep-going":
            ui.keep_going = True
    var hist = TensorImpl[DT].alloc(D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(D.GOAL)
    var step = 0
    var ret = 0.0
    var fail_step = -1          # the step the benchmark would have stopped at, this run
    var fail_ret = 0.0
    var ended = False
    var last = String("")
    var sim_hz = 0.0
    var frame_t0 = perf_counter_ns()
    var budget = 0.0
    while env.lanes[0].sim.env.is_renderer_open():
        if env.lanes[0].sim.env.check_renderer_quit():
            break
        env.lanes[0].sim.env.imgui_new_frame()
        var req = UiOut(-1, False, -1, False, False, False)
        if have_ui:
            req = _sidebar(
                ui, cat, sel, step, ret, fail_step, fail_ret, last, Float32(UnitreeG1Mjlab[DType.float64].RENDER_HEIGHT),
                env.lanes[0].sim.env.renderer_n_cameras(), env.lanes[0].sim.env.renderer_current_camera(), env.lanes[0].sim.env.renderer_paused(),
                env.lanes[0].sim.env.renderer_is_recording(), env.lanes[0].sim.env.renderer_recording_frames(), sim_hz,
            )
        if req.camera == -2:
            env.lanes[0].sim.env.renderer_request_free_camera()
        elif req.camera >= 0:
            env.lanes[0].sim.env.renderer_request_camera(req.camera)
        if req.pause:
            env.lanes[0].sim.env.renderer_toggle_pause()
        if req.shot:
            env.lanes[0].sim.env.renderer_request_screenshot()
        if req.record:
            try:
                env.lanes[0].sim.env.renderer_toggle_recording()
            except e:
                print("recording unavailable (launch through `pixi run` for ffmpeg):", e)
        if req.pick >= 0 or req.restart:
            if req.pick >= 0:
                sel = req.pick
                print("motion", sel, ":", cat.name[sel], "|", cat.frames[sel], "frames | eval", _fmt(cat.eval_ret[sel]))
            first[0] = env.lib.clip_start[sel]
            env.restart(first)
            step = 0
            ret = 0.0
            fail_step = -1
            ended = False
            for k in range(PLOT_N):
                ui.history[k] = 0.0

        # Control steps owed this frame at the chosen speed (at most 4, so a
        # slow machine plays slower rather than freezing the window).
        var now = perf_counter_ns()
        var dt_ms = Float64(now - frame_t0) / 1.0e6
        frame_t0 = now
        var n_steps = 0
        if not ended:
            if env.lanes[0].sim.env.renderer_paused():
                n_steps = 1 if env.lanes[0].sim.env.renderer_step_once() else 0
            else:
                budget = min(budget + dt_ms * Float64(ui.speed) / CONTROL_MS, 4.0)
                n_steps = Int(budget)
                budget -= Float64(n_steps)
        var t_sim = perf_counter_ns()
        for _ in range(n_steps):
            env.next_frame[0] = env.lib.clip_start[sel]
            env.terminations = not ui.keep_going
            env.policy_inputs(hist, goals)
            var a = trainer.act[1](hist, goals)
            _ = env.step(a.data, rb)
            var r = env.last_reward[0]
            ret += r
            step += 1
            ui.history[ui.cursor] = Float32(r)
            ui.cursor = (ui.cursor + 1) % PLOT_N
            if env.last_violated[0] and fail_step < 0:
                fail_step = step
                fail_ret = ret
            if env.lanes[0].ep_len == 0:        # the clip ended; the lane is back at its first frame
                var failed = env.last_terminated[0]
                last = _fmt(ret / (0.1 * Float64(max(step, 1)))) + (String(" FAILED at step ") + String(step) if failed else String(" completed"))
                if not failed and fail_step > 0:
                    last = (_fmt(ret / (0.1 * Float64(max(step, 1)))) + " to the end; benchmark FAILED at step "
                            + String(fail_step) + " (" + _fmt(fail_ret / (0.1 * Float64(fail_step))) + ")")
                print("  ", cat.label[sel], ":", last)
                if ui.auto_next:
                    sel = (sel + 1) % len(cat)
                    first[0] = env.lib.clip_start[sel]
                    env.restart(first)
                elif not ui.loop:
                    ended = True
                step = 0
                ret = 0.0
                fail_step = -1
                break
        if n_steps > 0:
            var spent = Float64(perf_counter_ns() - t_sim) / 1.0e9
            if spent > 0:
                var hz = Float64(n_steps) / spent
                sim_hz = hz if sim_hz == 0 else 0.9 * sim_hz + 0.1 * hz
        env.lanes[0].sim.env.render_frame()
        env.lanes[0].sim.env.renderer_delay(1)
    env.lanes[0].sim.env.close_renderer()
