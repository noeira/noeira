"""Camera studio — a live camera in a window, with ArUco detection on top.

    pixi run build-imgui                  # ONCE, the UI
    pixi run build-opencv                 # ONCE, the camera + detector
    pixi run mojo run -I . examples/vision/camera_studio.mojo
    pixi run mojo run -I . examples/vision/camera_studio.mojo --device 1
    pixi run mojo run -I . examples/vision/camera_studio.mojo --file tests/fixtures/vision/capture_12f.mp4

WHAT THIS IS
============
A playground, and deliberately much less than `examples/physics3d/physics_studio.mojo`:
one source, one view, no scene, no editing, no persistence. It exists so the
vision stack can be LOOKED AT before anything depends on it — a camera that
opens, frames that arrive at a measurable rate, and markers that light up when
you hold one in front of the lens.

⚠⚠ **IT IS ALSO THE FIRST THING THAT RUNS THE VISION PATH LIVE.** Every gate
in `tests/vision/` compares against a committed file, which is the right way to
gate marshalling and says NOTHING about whether a real camera opens, negotiates
a resolution, or keeps up. Those are exactly the questions a fixture cannot
ask, and they are why this is worth building before `deploy_reach_real` learns
to read a marker.

⚠ NO CAMERA NEEDED. `--file` plays any video (it loops), so the whole UI can be
exercised on the committed 3 KB fixture. `--image` shows a still.

CALIBRATING A CAMERA WITH IT
============================
1. `pixi run python tools/vision/make_printable_marker.py --charuco`
2. Print at 100%, measure the ruler, mount it FLAT (a clipboard is fine).
3. Run the studio, tick **detect board**. It should report 24 / 24 corners.
4. Click **capture view** at six or more DIFFERENT poses — near, far, tilted
   left/right/up/down, and with the board out toward the frame CORNERS.
5. Click **calibrate**. fx/fy/cx/cy/rms appear and `fx` is applied.

⚠⚠ **THE VIEWS MUST DIFFER, AND `rms` WILL NOT TELL YOU IF THEY DO NOT.** Six
captures of one pose is one view with six times the confidence in a badly
determined focal length, and it fits *beautifully* — rms is a residual of the
model against the corners it was given, never a statement about accuracy. Tilt
between every capture; put the board where distortion actually lives, at the
edges.

⚠ The board's five numbers are a CONTRACT with the printed sheet, not settings:
`--board 5x7 --square-mm 30 --marker-mm 22` must match what was printed. A
mismatched board does not fail — it finds different corners at different board
coordinates and calibrates a confidently wrong camera.

TESTING IT WITH A REAL PRINTED MARKER
=====================================
1. `pixi run python tools/vision/make_printable_marker.py --size-mm 60`
2. Print the **PDF** at **100%**, never "fit to page".
3. Measure the ruler on the sheet. If it is not 100 mm, measure the marker's
   BLACK SQUARE and use that number instead of the nominal one.
4. Tape it to something FLAT and run the studio.
5. Set `marker mm` to the measured side. Leave `dict` at 4x4_50.

⚠⚠ **`marker mm` IS THE BLACK SQUARE, NOT THE WHITE BORDER.** Every distance
below scales linearly with it, and the pose will look perfectly reasonable
either way — a wrong marker size does not produce a wrong-looking answer, it
produces a wrong-scaled one.

⚠ THE `fx` SLIDER IS A ONE-PARAMETER CALIBRATION, AND A CRUDE ONE. Hold the
marker at a distance you have actually measured, then move `fx` until `z`
agrees. That gets ranging roughly right and fixes NOTHING else: the principal
point stays assumed at the image centre and lens distortion stays ignored, so
accuracy degrades toward the edges of the frame. It is a sanity check, not a
substitute for the ChArUco calibration in `tests/vision/` (group E).

WHAT TO LOOK FOR
================
  * **capture ms** — the time `cap.read()` blocks. A camera that quietly
    negotiated 1920x1080 instead of the requested size shows up here as tens of
    milliseconds, long before it shows up as a control loop that misses its
    deadline.
  * **the resolution line** — what the device ACTUALLY gave, never what was
    asked for. OpenCV reports no error when it substitutes.
  * **markers** — ids and corner overlay. The pose readout needs a focal
    length, and the slider's default is a GUESS (see `FX_GUESS`), so treat the
    distance as an order of magnitude until the camera is calibrated.
"""

from std.sys import argv
from std.math import atan
from std.time import perf_counter_ns

from noeira.render.imgui import (
    IgTexture,
    ig_begin_panel,
    ig_begin_window,
    ig_button,
    ig_checkbox,
    ig_combo,
    ig_end,
    ig_framerate,
    ig_last_item_rect,
    ig_overlay_line,
    ig_plot_lines,
    ig_same_line,
    ig_separator,
    ig_separator_text,
    ig_slider_float,
    ig_text,
    ig_text_wrapped,
    ig_text_colored,
    ig_text_disabled,
    imgui_shim_available,
)
from noeira.render.renderer3d import Renderer3D
from noeira.utils.fmt import fixed
from noeira.vision.calib_file import CameraCalib, read_calib, write_calib
from noeira.vision.camera_thread import (
    camera_spec_index, open_camera_spec,
)
from noeira.vision.opencv import (
    ArucoDetector,
    CALIB_FIX_K3,
    CALIB_ZERO_TANGENT_DIST,
    CharucoBoard,
    DICT_4X4_50,
    SOLVEPNP_IPPE_SQUARE,
    VideoCapture,
    calibrate_camera,
    imread,
    opencv_shim_available,
    solve_pnp,
)

comptime WIN_W = 1180
comptime WIN_H = 760

comptime MAX_W = 1920
comptime MAX_H = 1080

comptime FX_GUESS: Float32 = 600.0
"""⚠ A GUESS, AND THE UI SAYS SO. A pose from an uncalibrated camera is a
distance with an unknown scale factor on it: everything is proportional to the
focal length, so a 20% error in `fx` is a 20% error in Z. Run the ChArUco
calibration (`tests/vision/`, group E) before any of this drives an arm."""

comptime MARKER_MM: Float32 = 40.0

# ⚠⚠ THESE FIVE NUMBERS ARE A CONTRACT WITH THE PRINTED SHEET, not settings.
# They must match `tools/vision/make_printable_marker.py --charuco` exactly —
# a board built from different numbers does not fail, it finds a different set
# of corners at different board coordinates and calibrates a confidently wrong
# camera. The generator prints them on its sheet for exactly this reason.
#
# ⚠ RUNTIME, NOT COMPTIME (`--board`, `--square-mm`, `--marker-mm`). These are
# defaults, and they match the generator's. Printing a different board must not
# cost a rebuild — the same reasoning `deploy_reach_real.mojo` records for its
# own bring-up knobs, and the same trap: five numbers now live in the
# generator, this file and the calibration fixture, so they can drift.
comptime BOARD_SX = 5
comptime BOARD_SY = 7
comptime BOARD_SQUARE_MM: Float32 = 30.0
comptime BOARD_MARKER_MM: Float32 = 22.0

comptime MIN_CORNERS = 8
"""Below this a view constrains almost nothing and mostly adds noise."""
comptime MIN_VIEWS = 6
"""⚠ AND THEY MUST DIFFER. Six views of the same pose is one view with six
times the confidence in the wrong answer — tilt the board between captures."""


def _fov_deg(pixels: Float64, f: Float64) -> Float64:
    """Full field of view in degrees for a sensor `pixels` wide and focal `f`.

    ⚠ THE CHEAPEST REALITY CHECK THERE IS. A calibration can be internally
    consistent and still describe a lens you do not own; the field of view is
    the one output you can verify by looking at the picture.
    """
    return 2.0 * atan(pixels / 2.0 / f) * 180.0 / 3.14159265358979323846


def _plot_push(mut buf: List[Float32], v: Float32, cap: Int = 180):
    """A scrolling history for `ig_plot_lines`. Oldest first."""
    buf.append(v)
    if len(buf) > cap:
        var trimmed = List[Float32](capacity=cap)
        for i in range(len(buf) - cap, len(buf)):
            trimmed.append(buf[i])
        buf = trimmed^


def main() raises:
    # ── arguments ───────────────────────────────────────────────────────────
    var device_spec = String("0")
    var path = String("")
    var is_image = False
    var board_sx = BOARD_SX
    var board_sy = BOARD_SY
    var square_mm = BOARD_SQUARE_MM
    var board_marker_mm = BOARD_MARKER_MM
    var calib_path = String("")
    var cam_name = String("")
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--device" and i + 1 < len(args):
            device_spec = String(args[i + 1])
        elif a == "--file" and i + 1 < len(args):
            path = String(args[i + 1])
        elif a == "--image" and i + 1 < len(args):
            path = String(args[i + 1])
            is_image = True
        elif a == "--board" and i + 1 < len(args):
            # "5x7"
            var spec = String(args[i + 1])
            var xi = spec.find("x")
            if xi > 0:
                board_sx = Int(spec[byte=0:xi])
                board_sy = Int(spec[byte = xi + 1 :])
        elif a == "--square-mm" and i + 1 < len(args):
            square_mm = Float32(Float64(String(args[i + 1])))
        elif a == "--marker-mm" and i + 1 < len(args):
            board_marker_mm = Float32(Float64(String(args[i + 1])))
        elif a == "--calib" and i + 1 < len(args):
            calib_path = String(args[i + 1])
        elif a == "--camera" and i + 1 < len(args):
            cam_name = String(args[i + 1])

    # ⚠ ONE FILE PER CAMERA, NAMED. A single `camera_calib.txt` is how a
    # two-camera rig ends up with the front camera's intrinsics applied to the
    # side one — which fails no check and simply mis-ranges everything.
    if cam_name == "":
        # ⚠ THE LAST PATH SEGMENT, NOT THE WHOLE SPEC. `cam_name` becomes a
        # FILE NAME two lines down, so "cam/dev/soarm_cam_wrist" would ask for
        # a directory that does not exist — a confusing failure for something
        # that is only a default.
        var parts = device_spec.split("/")
        cam_name = String("cam") + String(parts[len(parts) - 1])
    if calib_path == "":
        calib_path = String("scratch/camera_") + cam_name + ".txt"

    if not opencv_shim_available():
        print("OpenCV shim not built.  Run:  pixi run build-opencv")
        return
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    # ── the source ──────────────────────────────────────────────────────────
    # ⚠ OPEN IT BEFORE THE WINDOW. A camera that refuses to open should print
    # one line and exit, not flash a window first — and on macOS the FIRST
    # `cap_open` is what triggers the OS camera-permission prompt, which is far
    # less confusing when nothing else is on screen yet.
    var bgr = List[UInt8]()
    var frame_w: Int
    var frame_h: Int
    var source_label: String
    # ⚠ NOT `from_file("")` AS A PLACEHOLDER — that raises. See
    # `VideoCapture.closed`.
    var cap = VideoCapture.closed()

    if is_image:
        var g = imread(path, bgr)
        frame_w = g[0]
        frame_h = g[1]
        source_label = String("image: ") + path
    else:
        # ⚠ A PLAYGROUND MUST NOT GREET YOU WITH A STACK TRACE. "device 99
        # does not exist" and "another app holds the camera" are ordinary
        # outcomes here, not defects, so they print one line and exit.
        try:
            if path.byte_length() > 0:
                cap = VideoCapture.from_file(path)
                source_label = String("file: ") + path
            else:
                cap = open_camera_spec(device_spec, 1280, 720, 30.0)
                source_label = String("camera ") + String(device_spec)
        except e:
            print("could not open the source:", e)
            if path.byte_length() == 0:
                print(
                    "  Try another --device N (OpenCV prints the valid range"
                    " above), or run without a camera:"
                )
                print(
                    "    mojo run -I . examples/vision/camera_studio.mojo"
                    " --file tests/fixtures/vision/capture_12f.mp4"
                )
                print(
                    "  On macOS the first run asks for camera permission;"
                    " if you dismissed it, re-enable it in"
                )
                print("  System Settings > Privacy & Security > Camera.")
            return
        # ⚠ A FILE REPORTS ITS GEOMETRY UP FRONT; A CAMERA OFTEN DOES NOT until
        # the first frame arrives. Read one now so the texture is sized from
        # what the device actually produced rather than from what it claimed.
        if not cap.read(bgr):
            print("source opened but produced no frame:", source_label)
            cap.close()
            return
        frame_w = cap.width
        frame_h = cap.height

    if frame_w <= 0:
        print("no usable source. Tried:", source_label)
        return
    if frame_w > MAX_W or frame_h > MAX_H:
        print(
            "frame is",
            frame_w,
            "x",
            frame_h,
            "— larger than this demo's",
            MAX_W,
            "x",
            MAX_H,
            "budget",
        )
        return

    print("source:", source_label, "->", frame_w, "x", frame_h)

    # ── the window ──────────────────────────────────────────────────────────
    var r = Renderer3D(WIN_W, WIN_H)
    var title = String("camera studio — ") + source_label
    r.init(title)
    if not r.imgui_init():
        print("ImGui declined this device")
        return

    var tex = IgTexture(r.device.value(), frame_w, frame_h)
    var rgba = List[UInt8](unsafe_uninit_length=frame_w * frame_h * 4)

    var det = ArucoDetector(DICT_4X4_50)
    var ids = List[Int32]()
    var corners = List[Float32]()

    var detect_on = True
    var scale = Float32(1.0)
    var fx = FX_GUESS
    var marker_mm = MARKER_MM
    var known_mm = Float32(300.0)
    var calib_on = False
    if board_marker_mm >= square_mm:
        print("--marker-mm must be smaller than --square-mm")
        return
    var board = CharucoBoard(
        board_sx,
        board_sy,
        square_mm / 1000.0,
        board_marker_mm / 1000.0,
        DICT_4X4_50,
    )
    var board_corner_total = (board_sx - 1) * (board_sy - 1)
    var board_xyz = List[Float32]()
    _ = board.board_corners(board_xyz)
    var b_corners = List[Float32]()
    var b_ids = List[Int32]()
    var n_board_seen: Int
    var cal_obj = List[Float64]()
    var cal_img = List[Float64]()
    var cal_counts = List[Int32]()
    var cal_k = List[Float64]()
    var cal_dist = List[Float64]()
    var cal_rms = Float64(0.0)
    var cal_done = False
    # ⚠ WHERE THE CAPTURED CORNERS LANDED, which is the diagnostic a low `rms`
    # cannot give you. `calibrateCamera` puts the principal point where the
    # DATA is: views that all sit in the middle-bottom of the frame produce a
    # confident, low-residual fit with `cy` dragged toward them. Measured on a
    # real run: cx was 0.7% off centre and cy 6.1%, from six views that never
    # reached the top of the image.
    # ⚠ A CALIBRATION THAT LIVES ONLY IN A WINDOW IS A CALIBRATION YOU WILL DO
    # AGAIN. Loading is best-effort and silent on absence: a first run has no
    # file, and that is not an error.
    var cal_loaded = False
    # ⚠ ABSENCE IS NORMAL, A BAD FILE IS NOT. A first run has no calibration
    # and that is not an error; a file that exists and does not parse is worth
    # a line, and the pre-2026-09 six-float format lands exactly there.
    var have_file: Bool
    try:
        with open(calib_path, "r") as f:
            _ = f.read()
        have_file = True
    except:
        have_file = False
    if have_file:
        try:
            var prev = read_calib(calib_path)
            if prev.model != "pinhole":
                # the studio's pose readout is OpenCV's pinhole solve_pnp
                raise Error("a " + prev.model + " calibration, not loaded")
            cal_k = prev.k_matrix()
            cal_dist = prev.dist.copy()
            fx = Float32(prev.fx)
            cal_done = True
            cal_loaded = True
            print("loaded calibration from", calib_path, "fx", prev.fx)
            if prev.width != frame_w or prev.height != frame_h:
                print(
                    "  ⚠⚠ MEASURED AT",
                    prev.width,
                    "x",
                    prev.height,
                    "and this source gives",
                    frame_w,
                    "x",
                    frame_h,
                )
                print("     fx AND cx are both wrong by the ratio —")
                print("     recalibrate at this size rather than scaling.")
        except e:
            print("⚠ could not read", calib_path, "-", e)
            print("  A pre-2026-09 six-float calibration lands here.")
            print("  Recapture — the new format carries the camera's name")
            print("  and the size it was measured at, so it cannot be")
            print("  applied to the wrong picture.")

    var cov_x0 = Float32(1.0e9)
    var cov_x1 = Float32(-1.0e9)
    var cov_y0 = Float32(1.0e9)
    var cov_y1 = Float32(-1.0e9)
    var last_z_mm = Float64(0.0)
    var have_z: Bool
    var paused = False
    var n_markers: Int
    var frames = 0
    var cap_ms_hist = List[Float32]()
    var last_cap_ms = Float32(0.0)

    var obj: List[Float64]
    var img_xy: List[Float64]
    var k: List[Float64]
    var rvec = List[Float64]()
    var tvec = List[Float64]()

    var src_names = List[String]()
    src_names.append(String("4x4_50"))
    src_names.append(String("4x4_100"))
    src_names.append(String("5x5_50"))
    src_names.append(String("6x6_250"))
    var dict_vals = List[Int]()
    dict_vals.append(0)
    dict_vals.append(1)
    dict_vals.append(4)
    dict_vals.append(10)
    var dict_sel = Int32(0)

    while not r.check_quit():
        # ── grab ────────────────────────────────────────────────────────────
        if not is_image and not paused:
            var t0 = perf_counter_ns()
            var ok = cap.read(bgr)
            if not ok:
                # ⚠ A FILE ENDS; A CAMERA FAILING LOOKS THE SAME FROM HERE.
                # Reopening is right for a file (it loops) and is also the only
                # useful response to a camera that dropped off the bus.
                cap.close()
                if path.byte_length() > 0:
                    cap = VideoCapture.from_file(path)
                else:
                    cap = open_camera_spec(device_spec, 1280, 720, 30.0)
                _ = cap.read(bgr)
            last_cap_ms = Float32(Float64(perf_counter_ns() - t0) / 1_000_000.0)
            _plot_push(cap_ms_hist, last_cap_ms)
            frames += 1

        # ── BGR HWC -> RGBA, which is what the GPU texture wants ────────────
        # ⚠ NOT `bgr_hwc_to_rgb_chw`. That one is for the ACT store's layout;
        # this is interleaved RGBA for a sampler. Two different destinations,
        # and using either in the other's place produces an image that still
        # looks like an image.
        var n = frame_w * frame_h
        for i in range(n):
            rgba[i * 4 + 0] = bgr[i * 3 + 2]
            rgba[i * 4 + 1] = bgr[i * 3 + 1]
            rgba[i * 4 + 2] = bgr[i * 3 + 0]
            rgba[i * 4 + 3] = 255
        _ = tex.upload(rgba)

        if detect_on:
            n_markers = det.detect(bgr, frame_w, frame_h, 3, ids, corners)
        else:
            n_markers = 0
        # ⚠ ONLY WHEN ASKED. ChArUco detection runs its own marker pass, so
        # leaving it on doubles the per-frame cost for a panel nobody is
        # looking at — and this window's whole job is to report honest timings.
        if calib_on:
            n_board_seen = board.detect(
                bgr, frame_w, frame_h, 3, b_corners, b_ids
            )
        else:
            n_board_seen = 0

        # ── UI ──────────────────────────────────────────────────────────────
        r.imgui_new_frame()

        _ = ig_begin_panel(String("controls"), 0.0, 0.0, 300.0, Float32(WIN_H))
        ig_separator_text(String("source"))
        ig_text(source_label)
        ig_text(String("actual  ") + String(frame_w) + " x " + String(frame_h))
        if not is_image:
            ig_text_disabled(
                String("requested 1280 x 720 — a camera may ignore it")
            )
        ig_text(String("frames  ") + String(frames))
        ig_text(String("capture ") + fixed(Float64(last_cap_ms), 2) + " ms")
        ig_text(String("ui      ") + fixed(Float64(ig_framerate()), 1) + " fps")
        if len(cap_ms_hist) > 2:
            ig_plot_lines(String("ms"), cap_ms_hist, 0, 0.0, 60.0, 60.0)
        if not is_image:
            if ig_button(String("pause") if not paused else String("resume")):
                paused = not paused

        # ⚠⚠ EVERY CLICKABLE CONTROL BELOW SITS AT A FIXED POSITION, AND THAT
        # IS THE WHOLE LAYOUT RULE HERE. The first version printed one line per
        # detected marker id and hid whole sections behind `if`, so the panel's
        # height changed with what the camera happened to see. A ChArUco board
        # oscillating between 15 and 17 corners then moved every checkbox and
        # button below it, several times a second — a UI you have to CHASE.
        # Volatile text is therefore fixed-line-count, and a control that
        # cannot act is still DRAWN, just greyed and inert.
        ig_separator_text(String("markers"))
        _ = ig_checkbox(String("detect"), detect_on)
        if ig_combo(String("dict"), dict_sel, src_names):
            det.close()
            det = ArucoDetector(dict_vals[Int(dict_sel)])
        # One line, always: the ids joined rather than listed.
        if detect_on:
            var id_line = String("found ") + String(n_markers)
            if n_markers > 0:
                id_line += ":"
                for m in range(n_markers):
                    if m > 0:
                        id_line += ","
                    id_line += " " + String(ids[m])
            ig_text_wrapped(id_line)
        else:
            ig_text_disabled(String("detection off"))

        # ⚠ THE HEADING IS A CLAIM ABOUT THE NUMBERS BELOW IT, so it changes
        # when the numbers do. It read "uncalibrated" permanently, including
        # after a twelve-pose calibration had been loaded from disk.
        if cal_done:
            ig_separator_text(String("pose (calibrated)"))
            ig_text_disabled(String("full K and dist — the slider is inert"))
        else:
            ig_separator_text(String("pose (uncalibrated)"))
            ig_text_colored(
                String("fx is a GUESS until you calibrate"),
                1.0, 0.75, 0.2, 1.0,
            )
        _ = ig_slider_float(String("fx px"), fx, 200.0, 2000.0)
        _ = ig_slider_float(String("marker mm"), marker_mm, 5.0, 200.0)
        have_z = False
        if detect_on and n_markers > 0:
            var half = Float64(marker_mm) / 2000.0
            obj = List[Float64]()
            obj.append(-half)
            obj.append(half)
            obj.append(0.0)
            obj.append(half)
            obj.append(half)
            obj.append(0.0)
            obj.append(half)
            obj.append(-half)
            obj.append(0.0)
            obj.append(-half)
            obj.append(-half)
            obj.append(0.0)
            img_xy = List[Float64]()
            for i in range(8):
                img_xy.append(Float64(corners[i]))
            # ⚠⚠ ONCE A CALIBRATION EXISTS, USE ALL OF IT — NOT JUST `fx`.
            # This used to build `k` from the slider with the principal point
            # ASSUMED at the image centre and an EMPTY `dist`, and it kept
            # doing that after a calibration had measured both. On the lens
            # this was written for that is not a rounding difference: `cy`
            # came out 40 px (5.6%) off centre and `k1` = -0.25, so the pose
            # was being computed from a camera model the program itself had
            # already disproved — and the readout looked perfectly reasonable.
            #
            # ⚠ `dist` GOES TO `solve_pnp` DIRECTLY, which is why this tree
            # needs no image-space undistortion: OpenCV undistorts the four
            # corner points inside the solver.
            k = List[Float64]()
            var pnp_dist = List[Float64]()
            if cal_done:
                for i in range(9):
                    k.append(cal_k[i])
                for i in range(len(cal_dist)):
                    pnp_dist.append(cal_dist[i])
            else:
                k.append(Float64(fx))
                k.append(0.0)
                k.append(Float64(frame_w) / 2.0)
                k.append(0.0)
                k.append(Float64(fx))
                k.append(Float64(frame_h) / 2.0)
                k.append(0.0)
                k.append(0.0)
                k.append(1.0)
            try:
                solve_pnp(
                    obj, img_xy, k, pnp_dist, rvec, tvec, SOLVEPNP_IPPE_SQUARE
                )
                last_z_mm = tvec[2] * 1000.0
                have_z = True
            except:
                have_z = False
        # ⚠ CAMERA CONVENTION: x right, y DOWN, z along the optical axis. With
        # the marker centred in the frame x and y read near zero — a free check
        # on the assumed principal point that needs no tape measure.
        # Three lines whether or not there is a pose.
        if have_z:
            ig_text(String("x ") + fixed(tvec[0] * 1000.0, 1) + " mm")
            ig_text(String("y ") + fixed(tvec[1] * 1000.0, 1) + " mm")
            ig_text(String("z ") + fixed(tvec[2] * 1000.0, 1) + " mm")
        else:
            ig_text_disabled(String("x -"))
            ig_text_disabled(String("y -"))
            ig_text_disabled(String("z -"))

        # ── the range check, done FOR you ───────────────────────────────────
        #
        # ⚠ WHY THIS IS ONE CLICK AND NOT A PROCEDURE. `z` scales LINEARLY with
        # `fx`: solvePnP sees a marker `fx * S / Z` pixels wide, so assuming
        # the wrong focal length returns `Z * fx_assumed / fx_true`. That
        # inverts exactly — `fx_true = fx_assumed * z_known / z_shown` — so ONE
        # measured distance fixes ranging, and there is no reason to make
        # anyone do that arithmetic by hand.
        ig_separator_text(String("range check"))
        _ = ig_slider_float(String("known z mm"), known_mm, 50.0, 1500.0)
        if have_z:
            var suggested = Float64(fx) * Float64(known_mm) / last_z_mm
            var err_pct = (
                (last_z_mm - Float64(known_mm)) / Float64(known_mm) * 100.0
            )
            ig_text(String("implied fx ") + fixed(suggested, 1) + " px")
            ig_text(String("z is off by ") + fixed(err_pct, 1) + " %")
            # ⚠⚠ ONCE CALIBRATED THIS SECTION CHANGES JOB: from a crude
            # one-parameter FIT to an INDEPENDENT CHECK, and the button has to
            # go with it. The pose no longer reads the slider, so snapping it
            # would move a number that changes nothing on screen — a control
            # that silently does nothing, which is worse than one that is
            # visibly greyed.
            #
            # ⚠ AND THE CHECK IS THE BETTER HALF. `rms` is a residual against
            # the corners the fit was given; a tape measure is evidence from
            # outside the fit entirely. Agreement within ~1-2% at TWO
            # different distances is the cheapest confirmation available.
            if cal_done:
                _ = ig_button(String("snap fx (calibrated — inert)"))
            elif ig_button(String("snap fx to this distance")):
                fx = Float32(suggested)
        else:
            ig_text_disabled(String("implied fx -"))
            ig_text_disabled(String("z is off by -"))
            # ⚠ DRAWN BUT INERT, not hidden: a button that vanishes takes
            # everything below it with it.
            _ = ig_button(String("snap fx (needs a marker)"))
        if cal_done:
            ig_text_disabled(String("Measure a real distance, set it above,"))
            ig_text_disabled(String("and read the error at TWO distances."))
        else:
            ig_text_disabled(String("Measure a real distance, set it above,"))
            ig_text_disabled(String("click, then verify at a SECOND distance."))

        # ── calibration ─────────────────────────────────────────────────────
        ig_separator_text(String("calibration (ChArUco)"))
        _ = ig_checkbox(String("detect board"), calib_on)
        ig_text_disabled(
            String("board ")
            + String(board_sx)
            + "x"
            + String(board_sy)
            + "  "
            + fixed(Float64(square_mm), 0)
            + "/"
            + fixed(Float64(board_marker_mm), 0)
            + " mm"
        )
        if calib_on:
            ig_text(
                String("corners ")
                + String(n_board_seen)
                + " / "
                + String(board_corner_total)
            )
        else:
            ig_text_disabled(String("corners -"))
        ig_text(String("views   ") + String(len(cal_counts)))

        # ⚠ A PARTIAL BOARD IS NOT A PROBLEM, IT IS THE POINT. `detectBoard`
        # returns only VISIBLE corners with their ids, so a count that wanders
        # between 15 and 17 still contributes every corner it found — that is
        # exactly why a ChArUco board beats a plain chessboard, which must be
        # wholly visible or is discarded. The count moving is not a reason to
        # wait; a count below `MIN_CORNERS` is.
        var can_capture = calib_on and n_board_seen >= MIN_CORNERS
        if ig_button(String("capture view")) and can_capture:
            # ⚠ BY ID, NEVER POSITIONALLY. Only visible corners come back, so
            # the n-th detection is not the n-th board corner — pairing them by
            # position calibrates to nonsense while looking entirely healthy.
            for i in range(n_board_seen):
                var bid = Int(b_ids[i])
                cal_obj.append(Float64(board_xyz[bid * 3]))
                cal_obj.append(Float64(board_xyz[bid * 3 + 1]))
                cal_obj.append(Float64(board_xyz[bid * 3 + 2]))
                cal_img.append(Float64(b_corners[i * 2]))
                cal_img.append(Float64(b_corners[i * 2 + 1]))
            cal_counts.append(Int32(n_board_seen))
            for i in range(n_board_seen):
                var px = b_corners[i * 2]
                var py = b_corners[i * 2 + 1]
                if px < cov_x0:
                    cov_x0 = px
                if px > cov_x1:
                    cov_x1 = px
                if py < cov_y0:
                    cov_y0 = py
                if py > cov_y1:
                    cov_y1 = py
        ig_same_line()
        if ig_button(String("clear")):
            cal_obj = List[Float64]()
            cal_img = List[Float64]()
            cal_counts = List[Int32]()
            cal_done = False
            cov_x0 = 1.0e9
            cov_x1 = -1.0e9
            cov_y0 = 1.0e9
            cov_y1 = -1.0e9

        var can_calibrate = len(cal_counts) >= MIN_VIEWS
        if ig_button(String("calibrate")) and can_calibrate:
            var out = calibrate_camera(
                cal_obj,
                cal_img,
                cal_counts,
                frame_w,
                frame_h,
                cal_k,
                cal_dist,
                CALIB_ZERO_TANGENT_DIST | CALIB_FIX_K3,
            )
            cal_rms = out[1]
            cal_done = True
            fx = Float32(cal_k[0])
            print("calibrated from", len(cal_counts), "views:")
            print("  fx", cal_k[0], " fy", cal_k[4])
            print("  cx", cal_k[2], " cy", cal_k[5])
            print("  k1", cal_dist[0], " k2", cal_dist[1])
            print("  rms", cal_rms, "px")
            cal_loaded = False
            # ⚠ WRITTEN IMMEDIATELY, not behind a button. The expensive part of
            # a calibration is the twelve poses you held a board through; the
            # cheap part is a file. Making the save a separate click is how the
            # expensive part gets repeated.
            try:
                var out_cal = CameraCalib(
                    cam_name.copy(),
                    camera_spec_index(device_spec),
                    frame_w,
                    frame_h,
                    cal_k[0],
                    cal_k[4],
                    cal_k[2],
                    cal_k[5],
                )
                for di in range(len(cal_dist)):
                    out_cal.dist.append(cal_dist[di])
                out_cal.rms_px = cal_rms
                # ⚠ INTRINSICS ONLY. Where the camera IS belongs to
                # `examples/so101/calibrate_camera_extrinsics.mojo`, which
                # reads this file, adds the extrinsics and writes it back.
                write_calib(calib_path, out_cal)
                print("  saved to", calib_path)
            except e:
                print("  COULD NOT SAVE to", calib_path, "-", e)
        if not can_capture:
            ig_text_disabled(
                String("capture needs ") + String(MIN_CORNERS) + "+ corners"
            )
        elif not can_calibrate:
            ig_text_disabled(
                String("calibrate needs ") + String(MIN_VIEWS) + ", TILTED"
            )
        else:
            ig_text(String("ready — tilt between captures"))

        # ── coverage: the number `rms` cannot give you ──────────────────────
        #
        # ⚠⚠ A LOW `rms` WITH POOR COVERAGE IS THE DANGEROUS COMBINATION, not
        # a reassuring one. The residual only says the model explains the
        # corners it was given; if those corners all sit in one part of the
        # frame it explains them beautifully AND puts the principal point in
        # the middle of them. 70% is a floor, not a target — lens distortion
        # lives at the EDGES, so a calibration whose corners never went there
        # has no evidence about the part of the image it will be used on.
        if len(cal_counts) > 0:
            var cw = (cov_x1 - cov_x0) / Float32(frame_w) * 100.0
            var ch = (cov_y1 - cov_y0) / Float32(frame_h) * 100.0
            var cov_line = (
                String("coverage ")
                + fixed(Float64(cw), 0)
                + "% x "
                + fixed(Float64(ch), 0)
                + "%"
            )
            if cw < 70.0 or ch < 70.0:
                ig_text_colored(cov_line + " LOW", 1.0, 0.5, 0.3, 1.0)
                ig_text_disabled(String("push the board into the frame"))
                ig_text_disabled(String("corners, especially top and bottom"))
            else:
                ig_text(cov_line)
                ig_text_disabled(String("(70% is a floor, not a target)"))
        else:
            ig_text_disabled(String("coverage -"))
            ig_text_disabled(String(" "))
            ig_text_disabled(String(" "))

        # ⚠ RMS IS A FIT RESIDUAL, NOT AN ACCURACY. It says the model explains
        # the corners it was given; views that all face the board head-on fit
        # beautifully and still leave the focal length badly determined.
        if cal_done:
            ig_text(
                String("fx ")
                + fixed(cal_k[0], 1)
                + "  fy "
                + fixed(cal_k[4], 1)
            )
            ig_text(
                String("cx ")
                + fixed(cal_k[2], 1)
                + "  cy "
                + fixed(cal_k[5], 1)
            )
            if cal_rms > 1.0:
                ig_text_colored(
                    String("rms ") + fixed(cal_rms, 3) + " px — recapture",
                    1.0,
                    0.5,
                    0.3,
                    1.0,
                )
            else:
                ig_text(String("rms ") + fixed(cal_rms, 3) + " px")

            # ⚠ THE THREE CHECKS THAT COST NOTHING AND ARE WORTH MORE THAN THE
            # RESIDUAL. Square pixels means fx/fy ~ 1; a webcam's principal
            # point lands within ~1-2% of centre; and the field of view has to
            # match the picture you can see with your own eyes.
            var ar = cal_k[0] / cal_k[4]
            var ppx = (cal_k[2] - Float64(frame_w) / 2.0) / Float64(frame_w)
            var ppy = (cal_k[5] - Float64(frame_h) / 2.0) / Float64(frame_h)
            ig_text(String("fx/fy ") + fixed(ar, 4))
            var off_line = (
                String("centre off ")
                + fixed(ppx * 100.0, 1)
                + "% , "
                + fixed(ppy * 100.0, 1)
                + "%"
            )
            if abs(ppx) > 0.02 or abs(ppy) > 0.02:
                ig_text_colored(off_line + " HIGH", 1.0, 0.5, 0.3, 1.0)
            else:
                ig_text(off_line)
            if cal_loaded:
                ig_text_disabled(String("(loaded from file, not measured"))
                ig_text_disabled(String(" this session)"))
            else:
                ig_text_disabled(String("views ") + String(len(cal_counts)))
                ig_text_disabled(String(" "))
            ig_text(
                String("H fov ")
                + fixed(_fov_deg(Float64(frame_w), cal_k[0]), 1)
                + "  V "
                + fixed(_fov_deg(Float64(frame_h), cal_k[4]), 1)
            )
        else:
            ig_text_disabled(String("fx -  fy -"))
            ig_text_disabled(String("cx -  cy -"))
            ig_text_disabled(String("rms -"))

        ig_separator()
        _ = ig_slider_float(String("zoom"), scale, 0.25, 2.0)
        ig_end()

        # ── the image, with the detections drawn over it ────────────────────
        _ = ig_begin_window(
            String("view"),
            310.0,
            10.0,
            Float32(frame_w) * scale + 20.0,
            Float32(frame_h) * scale + 60.0,
        )
        var dw = Float32(frame_w) * scale
        var dh = Float32(frame_h) * scale
        tex.image(dw, dh)
        # ⚠ ASK IMGUI WHERE THE IMAGE WENT rather than recomputing it: the
        # window has padding, a title bar and possibly a scrollbar, and an
        # overlay drawn at a guessed origin is subtly and permanently offset.
        var rect = ig_last_item_rect()
        # ⚠ THE COVERAGE BOX, DRAWN. A percentage tells you the fit is
        # under-constrained; seeing the rectangle tells you WHERE to move the
        # board next, which is the only actionable form of that information.
        if calib_on and len(cal_counts) > 0:
            var bx0 = rect[0] + cov_x0 * scale
            var by0 = rect[1] + cov_y0 * scale
            var bx1 = rect[0] + cov_x1 * scale
            var by1 = rect[1] + cov_y1 * scale
            var amber = UInt32(0xFF20A0FF)
            ig_overlay_line(bx0, by0, bx1, by0, amber, 2.0)
            ig_overlay_line(bx1, by0, bx1, by1, amber, 2.0)
            ig_overlay_line(bx1, by1, bx0, by1, amber, 2.0)
            ig_overlay_line(bx0, by1, bx0, by0, amber, 2.0)

        if detect_on and n_markers > 0:
            for m in range(n_markers):
                for c in range(4):
                    var c2 = (c + 1) % 4
                    ig_overlay_line(
                        rect[0] + corners[m * 8 + c * 2] * scale,
                        rect[1] + corners[m * 8 + c * 2 + 1] * scale,
                        rect[0] + corners[m * 8 + c2 * 2] * scale,
                        rect[1] + corners[m * 8 + c2 * 2 + 1] * scale,
                        0xFF00FF00,
                        2.0,
                    )
        ig_end()

        r.begin_frame()
        r.end_frame()

    board.close()
    det.close()
    tex.close()
    if not is_image:
        cap.close()
