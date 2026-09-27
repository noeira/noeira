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
from noeira.tasks.delta_action import DELTA_ACT, DELTA_ARM, DELTA_GRIPPER

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
comptime C_IN = 3 * N_CAMS + ACT
comptime IN_DIM = C_IN * PLANE
comptime HID = 256
comptime JOINT_SCALE: Float64 = 0.5
comptime IMAGE_OFFSET: Float64 = 0.5
comptime GRIPPER_ACT = ACT - 1
comptime CAM_FOVY_DEG: Float64 = 73.7398
"""Both rig cameras' `fovy` (the stand's and the wrist mount's MJCF); the
real frames are undistorted to a pinhole at this fovy."""
comptime MANIFEST_KIND = "noeira.pixel_student.v1"

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


def frame_to_planes(
    ref frame: List[UInt8], w: Int, h: Int, cam: Int,
    mut x: List[Scalar[DT]],
) raises:
    """One real camera frame — CHW RGB uint8, `w` x `h`, row 0 at the top,
    undistorted to the sim pinhole — into camera `cam`'s three planes of
    `x`: the centred `h` x `h` square, averaged over (h / OBS_PX)^2 blocks,
    / 255 - 0.5. See the module docstring for why a CROP."""
    if w < h:
        raise Error("pixel student: a " + String(w) + "x" + String(h)
                    + " frame is taller than wide")
    if h % OBS_PX != 0:
        raise Error("pixel student: frame height " + String(h)
                    + " is not a multiple of " + String(OBS_PX))
    if len(frame) != 3 * w * h:
        raise Error("pixel student: frame holds " + String(len(frame))
                    + " bytes, expected 3 x " + String(w) + " x " + String(h))
    var b = h // OBS_PX
    var x0 = (w - h) // 2
    var inv = 1.0 / (255.0 * Float64(b * b))
    for c in range(3):
        for oy in range(OBS_PX):
            for ox in range(OBS_PX):
                var acc = 0
                for dy in range(b):
                    var row = c * w * h + (oy * b + dy) * w + x0 + ox * b
                    for dx in range(b):
                        acc += Int(frame[row + dx])
                x[(3 * cam + c) * PLANE + oy * OBS_PX + ox] = Scalar[DT](
                    Float64(acc) * inv - IMAGE_OFFSET
                )


def render_to_planes(
    ref rgb: List[Scalar[DT]], r: Int, cam: Int, mut x: List[Scalar[DT]]
) raises:
    """A SIM render (the tracer's `rgb`: `r` x `r`, HWC floats in [0, 1], top
    row first) into camera `cam`'s planes — the host twin of the trainer's
    `_pack_camera_kernel` (block mean over (r / OBS_PX)^2, minus 0.5), for
    tools that drive the student one env at a time on the CPU (the viewer)."""
    if r % OBS_PX != 0 or len(rgb) != r * r * 3:
        raise Error("pixel student: a " + String(r) + "x" + String(r)
                    + " render does not block-average to " + String(OBS_PX))
    var f = r // OBS_PX
    var inv = 1.0 / Float64(f * f)
    for c in range(3):
        for oy in range(OBS_PX):
            for ox in range(OBS_PX):
                var acc = 0.0
                for dy in range(f):
                    for dx in range(f):
                        acc += Float64(rgb[((oy * f + dy) * r + ox * f + dx) * 3 + c])
                x[(3 * cam + c) * PLANE + oy * OBS_PX + ox] = Scalar[DT](
                    acc * inv - IMAGE_OFFSET
                )


def joints_to_planes(ref q: List[Float64], mut x: List[Scalar[DT]]):
    """The six joints (model radians) broadcast into their planes."""
    for j in range(ACT):
        var v = Scalar[DT](q[j] * JOINT_SCALE)
        var base = IMG + j * PLANE
        for p in range(PLANE):
            x[base + p] = v


def write_pixel_manifest(
    path: String, task: String, teacher: String, gripper_sign: Bool,
    control_period_s: Float64,
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
    s += '  "frame": "centre square crop, block mean, /255 - 0.5",\n'
    s += '  "joint_scale": ' + String(JOINT_SCALE) + ',\n'
    s += '  "joint_units": "model radians (tower_follower joint zero)",\n'
    s += '  "delta_arm": ' + String(DELTA_ARM) + ',\n'
    s += '  "delta_gripper": ' + String(DELTA_GRIPPER) + ',\n'
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

    def __init__(out self):
        self.task = String("")
        self.teacher = String("")
        self.gripper_sign = False
        self.control_period_s = 0.0


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
    if abs(_num(doc, r, "joint_scale", path) - JOINT_SCALE) > 1e-9:
        raise Error("pixel student: joint_scale differs from this build's")
    if abs(_num(doc, r, "delta_arm", path) - DELTA_ARM) > 1e-9 or abs(
        _num(doc, r, "delta_gripper", path) - DELTA_GRIPPER
    ) > 1e-9:
        raise Error("pixel student: the action scale differs from this build's")
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
    return m^
