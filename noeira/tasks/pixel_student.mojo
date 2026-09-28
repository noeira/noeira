"""The pixel student's contract — network, observation and manifest, written
once for the trainer that makes it and the real-arm deploy that runs it.

    from noeira.tasks.pixel_student import (
        StudentNet, N_CAMS, OBS_PX, IN_DIM, frame_to_planes, joints_to_planes,
        student_act, write_pixel_manifest, check_pixel_manifest,
    )

The trainer is `noeira/tasks/pixel_dagger_tower.mojo` (DAgger from a state
PPO teacher); the deploy is `examples/so101/pixel_student_deploy_real.mojo`.

## The observation, `C_IN` planes of `OBS_PX` x `OBS_PX`, NCHW

- camera `k` (slot 0 overhead, slot 1 wrist — the rig's order) in planes
  3k..3k+2: RGB / 255 - 0.5, row 0 at the TOP;
- then the six actuated joints, model radians x `JOINT_SCALE`, each
  broadcast over a whole plane.

⚠⚠ THE SIM PICTURE IS A SQUARE. The trainer traces each camera at `RENDER` x
`RENDER` with the model's `fovy` (73.7398 deg) in BOTH directions
(`raytrace/camera.mojo`: `half_w = half_h * W / H`), then averages
`RENDER / OBS_PX` blocks. A real camera, undistorted to the sim pinhole at
640 x 480 (`CameraReader.set_undistort`), sees 73.74 deg vertically and 90
horizontally. So the real frame is CROPPED to its centred H x H square —
the same angular window — before the same block average
(`frame_to_planes`). Resizing the whole 4:3 frame to a square instead would
squeeze every object horizontally by 3/4 and move everything off-centre
toward the middle: a policy trained on the square would reach for bricks
that are not where it sees them. Gated against the renderer by
`tests/tasks/test_pixel_student_crop.mojo`.

## The action

Six words in [-1, 1] through `delta_action.delta_target` (the measured joint
+ a * 0.05 rad, gripper 0.2) — exactly the PPO teacher's action space.
`student_act` clamps, and with `gripper_sign` snaps the gripper word to +-1.
"""

from std.sys import is_defined

from noeira.io.json import JsonDoc, load_json
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.conv2d import Conv2D
from noeira.nn.primitives.flatten import Flatten
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.tasks.delta_action import DELTA_ACT, DELTA_ARM, DELTA_GRIPPER, ACT_HIST

comptime ACT = DELTA_ACT
comptime N_CAMS = 1 if is_defined["DAGGER_WRIST_ONLY"]() else 2
comptime RENDER = 64
"""The traced resolution in the trainer (1 sample)."""
comptime OBS_PX = 32 if is_defined["DAGGER_PX_32"]() else 16
"""16 (Squint's) by default, `-D DAGGER_PX_32` for 32. ⚠ A BUILD CHOICE the
checkpoint depends on: the manifest records it and the deploy refuses a
mismatch."""
comptime PLANE = OBS_PX * OBS_PX
comptime IMG = 3 * N_CAMS * PLANE
comptime JOINT_VEL = is_defined["DAGGER_JOINT_VEL"]()
"""`-D DAGGER_JOINT_VEL`: the six joint VELOCITIES as six more planes. The
cube-in-bowl teacher's quick corrections depend on velocity it reads from its
state (its `qvel` words) and a single frame cannot show; the real servos
report velocity (`SO101Arm.read_velocities`), so the plane is deployable."""
comptime PROPRIO_STATE = 2 * ACT if JOINT_VEL else ACT
"""The state's joint planes: the angles, then (with JOINT_VEL) the velocities."""
comptime HIST_WORDS = ACT_HIST * ACT
comptime PROPRIO = PROPRIO_STATE + HIST_WORDS
"""All the joint planes: the state's, then (`-D TASK_PPO_ACT_HIST=K`) the
last K executed actions, most recent first, unscaled ([-1, 1])."""
comptime C_IN = 3 * N_CAMS + PROPRIO
comptime IN_DIM = C_IN * PLANE
comptime HID = 256
comptime JOINT_SCALE: Float64 = 0.5
comptime JOINT_VEL_SCALE: Float64 = 0.2
"""rad/s -> plane value: the arm's joint speeds reach ~2-3 rad/s."""
comptime IMAGE_OFFSET: Float64 = 0.5
comptime GRIPPER_ACT = ACT - 1
comptime CAM_FOVY_DEG: Float64 = 73.7398
"""Both rig cameras' `fovy` (the stand's and the wrist mount's MJCF); the
real frames are undistorted to a pinhole at this fovy."""
comptime MANIFEST_KIND = "noeira.pixel_student.v1"

# ── the camera WINDOWS (`-D DAGGER_WINDOW`) ─────────────────────────────────
#
# ⚠⚠ WITHOUT THE DEFINE, EVERY CAMERA IS THE CENTRED SQUARE of its 4:3 frame
# (the first students, `pixel_lift` / `pixel_bowl`). WITH IT, each camera's
# observation is a fixed WINDOW of its REAL 4:3 frame (undistorted to the sim
# pinhole: fovy 73.74 deg vertically, 90 horizontally), in normalised frame
# coordinates (u right, v down, [0, 1]), area-averaged to OBS_PX x OBS_PX —
# the same function on the sim render and on the real frame, so the two
# pictures are the same window of the same camera.
#
# The OVERHEAD window is MEASURED (`tools/tasks/tower_overhead_roi.mojo`, 300
# cube-in-bowl placements, 27 Sep): the props' footprint spans u 0.10-0.67,
# v 0.33-1.0 of the 4:3 frame — it reaches past the centred square's left
# edge (u 0.125), so the square CUT some placements, and everything above
# v 0.33 (the wall, the door, the speaker on the real rig) is never a prop.
# The window below is that footprint with ~3 % margin. The WRIST window stays
# the centred square: the wrist sees what the gripper approaches.
comptime WINDOWED = is_defined["DAGGER_WINDOW"]()
comptime OVERHEAD_RENDER_W = 128 if WINDOWED else RENDER
comptime OVERHEAD_RENDER_H = 96 if WINDOWED else RENDER
"""The trainer's overhead trace: the FULL 4:3 frame when windowed (it then
covers the real frame's whole field), the centred square otherwise."""


def camera_window(name: String) -> Tuple[Float64, Float64, Float64, Float64]:
    """(u0, v0, u1, v1) of camera `name`'s observation in its real 4:3 frame."""
    comptime if WINDOWED:
        if name == "overhead":
            return (0.0703125, 0.2916667, 0.703125, 1.0)
    return (0.125, 0.0, 0.875, 1.0)


def render_cover(name: String) -> Tuple[Float64, Float64, Float64, Float64]:
    """(u0, v0, u1, v1) of the real frame the trainer's render of `name` covers:
    the whole 4:3 frame for the windowed overhead, the centred square
    otherwise (a square render at the camera's fovy)."""
    comptime if WINDOWED:
        if name == "overhead":
            return (0.0, 0.0, 1.0, 1.0)
    return (0.125, 0.0, 0.875, 1.0)


def window_in_render(
    name: String, rw: Int, rh: Int
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Camera `name`'s window in PIXELS of a `rw` x `rh` render covering
    `render_cover(name)` — the rectangle the trainer's kernel averages."""
    var w = camera_window(name)
    var c = render_cover(name)
    var sx = Float64(rw) / (c[2] - c[0])
    var sy = Float64(rh) / (c[3] - c[1])
    return ((w[0] - c[0]) * sx, (w[1] - c[1]) * sy,
            (w[2] - c[0]) * sx, (w[3] - c[1]) * sy)


@always_inline
def area_cell_bounds(
    x0: Float64, y0: Float64, x1: Float64, y1: Float64, ox: Int, oy: Int,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Output cell (ox, oy)'s rectangle in source pixels, of the window
    (x0, y0)-(x1, y1) split into OBS_PX x OBS_PX."""
    var cw = (x1 - x0) / Float64(OBS_PX)
    var ch = (y1 - y0) / Float64(OBS_PX)
    return (x0 + Float64(ox) * cw, y0 + Float64(oy) * ch,
            x0 + Float64(ox + 1) * cw, y0 + Float64(oy + 1) * ch)

comptime StudentNet = Sequential[
    Conv2D[C_IN, 32, 3, 1, 1, OBS_PX, OBS_PX], ReLU[32 * PLANE],
    Conv2D[32, 64, 3, 2, 1, OBS_PX, OBS_PX], ReLU[64 * (PLANE // 4)],
    Conv2D[64, 64, 3, 2, 1, OBS_PX // 2, OBS_PX // 2], ReLU[64 * (PLANE // 16)],
    Flatten[64 * (PLANE // 16)],
    LinearReLU[64 * (PLANE // 16), HID],
    LinearReLU[HID, HID],
    Linear[HID, ACT],
]


def camera_names() -> List[String]:
    """The cameras in slot order, as the manifest names them."""
    var n = List[String]()
    comptime if N_CAMS == 2:
        n.append(String("overhead"))
    n.append(String("wrist"))
    return n^


@always_inline
def student_act(v: Scalar[DT], j: Int, grip_sign: Bool) -> Scalar[DT]:
    """The student's output as executed: clamped to [-1, 1]; with
    `grip_sign` the GRIPPER word snapped to +-1."""
    if grip_sign and j == GRIPPER_ACT:
        return Scalar[DT](1) if v > Scalar[DT](0) else Scalar[DT](-1)
    if v > Scalar[DT](1):
        return Scalar[DT](1)
    if v < Scalar[DT](-1):
        return Scalar[DT](-1)
    return v


@always_inline
def _cover(a: Float64, b: Float64, p: Int) -> Float64:
    """Length of [a, b) inside source pixel [p, p+1)."""
    var lo = a if a > Float64(p) else Float64(p)
    var hi = b if b < Float64(p + 1) else Float64(p + 1)
    return hi - lo if hi > lo else 0.0


def frame_to_planes(
    ref frame: List[UInt8], w: Int, h: Int, cam: Int,
    mut x: List[Scalar[DT]],
) raises:
    """One real camera frame — CHW RGB uint8, `w` x `h`, row 0 at the top,
    undistorted to the sim pinhole — into camera `cam`'s three planes of
    `x`: its window (`camera_window`; the centred square by default) of the
    4:3 frame, AREA-averaged to OBS_PX x OBS_PX, / 255 - 0.5. With the
    centred square of a 640 x 480 frame the cells fall on whole pixels and
    this is the plain block mean. See the module docstring for why a crop."""
    if len(frame) != 3 * w * h:
        raise Error("pixel student: frame holds " + String(len(frame))
                    + " bytes, expected 3 x " + String(w) + " x " + String(h))
    var win = camera_window(camera_names()[cam])
    var x0 = win[0] * Float64(w)
    var y0 = win[1] * Float64(h)
    var x1 = win[2] * Float64(w)
    var y1 = win[3] * Float64(h)
    for c in range(3):
        for oy in range(OBS_PX):
            for ox in range(OBS_PX):
                var cb = area_cell_bounds(x0, y0, x1, y1, ox, oy)
                var acc = 0.0
                var wsum = 0.0
                for py in range(Int(cb[1]), min(Int(cb[3]) + 1, h)):
                    var wy = _cover(cb[1], cb[3], py)
                    if wy <= 0.0:
                        continue
                    for px in range(Int(cb[0]), min(Int(cb[2]) + 1, w)):
                        var wx = _cover(cb[0], cb[2], px)
                        if wx <= 0.0:
                            continue
                        acc += wx * wy * Float64(frame[c * w * h + py * w + px])
                        wsum += wx * wy
                x[(3 * cam + c) * PLANE + oy * OBS_PX + ox] = Scalar[DT](
                    acc / (255.0 * wsum) - IMAGE_OFFSET
                )


def render_to_planes(
    ref rgb: List[Scalar[DT]], rw: Int, rh: Int, cam: Int,
    mut x: List[Scalar[DT]],
) raises:
    """A SIM render of camera `cam` (the tracer's `rgb`: `rw` x `rh`, HWC
    floats in [0, 1], top row first, covering `render_cover`) into its
    planes: its window, area-averaged — the host twin of the trainer's
    `_pack_camera_kernel`, for tools that drive the student one env at a
    time on the CPU (the viewer) and for the gates."""
    if len(rgb) != rw * rh * 3:
        raise Error("pixel student: render of " + String(len(rgb))
                    + " floats is not " + String(rw) + "x" + String(rh) + "x3")
    var r = window_in_render(camera_names()[cam], rw, rh)
    for c in range(3):
        for oy in range(OBS_PX):
            for ox in range(OBS_PX):
                var cb = area_cell_bounds(r[0], r[1], r[2], r[3], ox, oy)
                var acc = 0.0
                var wsum = 0.0
                for py in range(Int(cb[1]), min(Int(cb[3]) + 1, rh)):
                    var wy = _cover(cb[1], cb[3], py)
                    if wy <= 0.0:
                        continue
                    for px in range(Int(cb[0]), min(Int(cb[2]) + 1, rw)):
                        var wx = _cover(cb[0], cb[2], px)
                        if wx <= 0.0:
                            continue
                        acc += wx * wy * Float64(rgb[(py * rw + px) * 3 + c])
                        wsum += wx * wy
                x[(3 * cam + c) * PLANE + oy * OBS_PX + ox] = Scalar[DT](
                    acc / wsum - IMAGE_OFFSET
                )


def joint_vels_to_planes(ref qd: List[Float64], mut x: List[Scalar[DT]]):
    """The six joint velocities (model rad/s) into their planes — a no-op in
    a build without `DAGGER_JOINT_VEL`."""
    comptime if JOINT_VEL:
        for j in range(ACT):
            var v = Scalar[DT](qd[j] * JOINT_VEL_SCALE)
            var base = IMG + (ACT + j) * PLANE
            for p in range(PLANE):
                x[base + p] = v


def act_hist_to_planes(ref hist: List[Float64], mut x: List[Scalar[DT]]):
    """The last ACT_HIST executed actions (`act_hist_push`'s layout) into
    their planes — a no-op in a build without `TASK_PPO_ACT_HIST`."""
    for k in range(HIST_WORDS):
        var v = Scalar[DT](hist[k])
        var base = IMG + (PROPRIO_STATE + k) * PLANE
        for p in range(PLANE):
            x[base + p] = v


def act_hist_push(mut hist: List[Float64], ref a: List[Float64]):
    """Shift the history by one action and put `a` (the six words AS
    EXECUTED, clipped to [-1, 1]) in slot 0. `hist` holds HIST_WORDS words,
    zero at an episode's start."""
    for k in range(HIST_WORDS - 1, ACT - 1, -1):
        hist[k] = hist[k - ACT]
    comptime if ACT_HIST > 0:
        for j in range(ACT):
            var v = a[j]
            hist[j] = 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)


def joints_to_planes(ref q: List[Float64], mut x: List[Scalar[DT]]):
    """The six joints (model radians) broadcast into their planes."""
    for j in range(ACT):
        var v = Scalar[DT](q[j] * JOINT_SCALE)
        var base = IMG + j * PLANE
        for p in range(PLANE):
            x[base + p] = v


def write_pixel_manifest(
    path: String, task: String, teacher: String, gripper_sign: Bool,
    control_period_s: Float64, delta_arm: Float64 = DELTA_ARM,
    delta_gripper: Float64 = DELTA_GRIPPER, lag_tau: String = "",
    lag_delay: String = "",
) raises:
    """The student's contract as JSON — at `checkpoints/norm.json`, the file
    `project-promote` copies beside the weights, so a promoted pixel policy
    carries it to the deploy."""
    var cams = camera_names()
    var cl = String("")
    for i in range(len(cams)):
        cl += ("" if i == 0 else ", ") + '"' + cams[i] + '"'
    var s = String("{\n")
    s += '  "kind": "' + String(MANIFEST_KIND) + '",\n'
    s += '  "task": "' + task + '",\n'
    s += '  "teacher": "' + teacher + '",\n'
    s += '  "cameras": [' + cl + '],\n'
    s += '  "obs_px": ' + String(OBS_PX) + ',\n'
    s += '  "render": ' + String(RENDER) + ',\n'
    s += '  "fovy_deg": ' + String(CAM_FOVY_DEG) + ',\n'
    s += '  "frame": "' + ("workspace window (camera_window), area mean, /255 - 0.5" if WINDOWED else "centre square crop, block mean, /255 - 0.5") + '",\n'
    s += '  "window": "' + ("workspace" if WINDOWED else "centre-square") + '",\n'
    s += '  "joint_scale": ' + String(JOINT_SCALE) + ',\n'
    s += '  "proprio": "' + ("q+qd" if JOINT_VEL else "q") + '",\n'
    s += '  "joint_vel_scale": ' + String(JOINT_VEL_SCALE) + ',\n'
    s += '  "act_hist": ' + String(ACT_HIST) + ',\n'
    s += '  "joint_units": "model radians (tower_follower joint zero)",\n'
    s += '  "delta_arm": ' + String(delta_arm) + ',\n'
    s += '  "delta_gripper": ' + String(delta_gripper) + ',\n'
    s += '  "servo_lag": "tau ' + lag_tau + ' ms, delay ' + lag_delay + ' ticks",\n'
    s += '  "gripper_sign": ' + ("true" if gripper_sign else "false") + ',\n'
    s += '  "control_period_s": ' + String(control_period_s) + '\n'
    s += "}\n"
    with open(path, "w") as f:
        f.write(s)


struct PixelManifest(Copyable, Movable):
    var task: String
    var teacher: String
    var gripper_sign: Bool
    var control_period_s: Float64
    var delta_arm: Float64
    var delta_gripper: Float64
    """The per-step scales the policy was TRAINED with — every executor of
    it must use them (a student acts in its teacher's units)."""

    def __init__(out self):
        self.task = String("")
        self.teacher = String("")
        self.gripper_sign = False
        self.control_period_s = 0.0
        self.delta_arm = DELTA_ARM
        self.delta_gripper = DELTA_GRIPPER


def _num(ref doc: JsonDoc, r: Int, k: String, path: String) raises -> Float64:
    var n = doc.field(r, k)
    if n < 0:
        raise Error("pixel student: " + path + " has no '" + k + "'")
    return doc.number(n)


def check_pixel_manifest(path: String) raises -> PixelManifest:
    """Read a manifest and REFUSE one this build cannot run: another kind,
    another resolution, camera set, joint scale or action scale. Each would
    feed the network an observation it was not trained on, or command the
    arm at another step size — and none of them would raise anywhere else."""
    var doc = load_json(path)
    var r = doc.root()
    var kind = doc.field(r, "kind")
    if kind < 0 or doc.string(kind) != String(MANIFEST_KIND):
        raise Error("pixel student: " + path + " is not a "
                    + String(MANIFEST_KIND) + " manifest")

    var px = Int(_num(doc, r, "obs_px", path))
    if px != OBS_PX:
        raise Error("pixel student: the policy was trained at " + String(px)
                    + "x" + String(px) + ", this build is " + String(OBS_PX)
                    + (" (build with -D DAGGER_PX_32)" if px == 32 else ""))
    var cams = doc.field(r, "cameras")
    var want = camera_names()
    if cams < 0 or doc.size(cams) != len(want):
        raise Error("pixel student: camera set differs from this build's ("
                    + String(len(want)) + " cameras)")
    for i in range(len(want)):
        if doc.string(doc.at(cams, i)) != want[i]:
            raise Error("pixel student: camera slot " + String(i) + " is '"
                        + doc.string(doc.at(cams, i)) + "', this build's is '"
                        + want[i] + "'")
    var wf = doc.field(r, "window")
    var window = doc.string(wf) if wf >= 0 else String("centre-square")
    if window != ("workspace" if WINDOWED else "centre-square"):
        raise Error("pixel student: the policy's camera window is '" + window
                    + "', this build's is '" + ("workspace" if WINDOWED else "centre-square")
                    + "'" + (" (build with -D DAGGER_WINDOW)" if window == "workspace" else ""))
    var pr = doc.field(r, "proprio")
    var proprio = doc.string(pr) if pr >= 0 else String("q")
    if proprio != ("q+qd" if JOINT_VEL else "q"):
        raise Error("pixel student: the policy's joint input is '" + proprio
                    + "', this build's is '" + ("q+qd" if JOINT_VEL else "q")
                    + "'" + (" (build with -D DAGGER_JOINT_VEL)"
                             if proprio == "q+qd" else ""))
    var ah = doc.field(r, "act_hist")
    var act_hist = Int(doc.number(ah)) if ah >= 0 else 0
    if act_hist != ACT_HIST:
        raise Error("pixel student: the policy sees its last " + String(act_hist)
                    + " actions, this build " + String(ACT_HIST)
                    + " (build with -D TASK_PPO_ACT_HIST=" + String(act_hist) + ")")
    if abs(_num(doc, r, "joint_scale", path) - JOINT_SCALE) > 1e-9:
        raise Error("pixel student: joint_scale differs from this build's")
    var da = _num(doc, r, "delta_arm", path)
    var dg = _num(doc, r, "delta_gripper", path)
    if not (da > 0.0 and da < 1.0 and dg > 0.0 and dg < 2.0):
        raise Error("pixel student: implausible action scales " + String(da)
                    + " / " + String(dg))
    if abs(_num(doc, r, "fovy_deg", path) - CAM_FOVY_DEG) > 1e-3:
        raise Error("pixel student: the camera fovy differs from this build's")
    var m = PixelManifest()
    var t = doc.field(r, "task")
    if t >= 0:
        m.task = doc.string(t)
    var te = doc.field(r, "teacher")
    if te >= 0:
        m.teacher = doc.string(te)
    var g = doc.field(r, "gripper_sign")
    if g >= 0:
        m.gripper_sign = doc.boolean(g)
    m.control_period_s = _num(doc, r, "control_period_s", path)
    m.delta_arm = da
    m.delta_gripper = dg
    return m^
