# +--------------------------------------------------------------------------+ #
# | A camera calibration that says WHICH camera and at WHAT size
# +--------------------------------------------------------------------------+ #
"""Read and write one camera's intrinsics and extrinsics, as keyed text.

`docs/VISION_ASSESSMENT_2026_09_09.md` §2 item 2.

## ⚠⚠ WHY THIS REPLACES SIX BARE FLOATS

`camera_studio.mojo` wrote `fx fy cx cy k1 k2` and nothing else, and for a
while that file had exactly one reader — the studio that wrote it. The moment a
SECOND program reads a calibration, the six floats become dangerous, because
every one of them is silently wrong under a change the file cannot describe:

| what changed | what the six floats do |
|---|---|
| a different camera on the rig | apply confidently; ranging off by the focal ratio |
| the same camera at 1280x720 instead of 640x480 | **`fx` and `cx` are both off by exactly 2x** |
| the camera was moved | intrinsics still fine, extrinsics silently void |

So the file carries its **identity** (`name`, `device`) and its **image size**,
and `require_size` refuses a mismatch rather than scaling something it was
never told about. A calibration you cannot misapply is worth more than one with
another decimal place.

⚠ **THE SIZE CHECK IS A REFUSAL, NOT A RESCALE.** `fx` and `cx` do scale
linearly with a pure resize, so rescaling would be *arithmetically* defensible
and is still wrong here: a camera reporting a different resolution has usually
changed its CROP or its binning, not just its sampling, and those move the
field of view without moving the pixel count in any recoverable way. Recalibrate.

## The format

Keyed lines, `#` comments, order irrelevant, unknown keys ignored so a newer
writer does not break an older reader:

```
mojo-rl-camera-calibration 1
name front
device 0
size 640 480
intrinsics <fx> <fy> <cx> <cy>
model pinhole | fisheye
dist <k1> <k2> [p1 p2 k3 ...]     (pinhole: OpenCV's radial-tangential vector)
dist <k1> <k2> <k3> <k4>          (fisheye: Kannala-Brandt, `vision/fisheye.mojo`)
rms_px <the intrinsics fit residual>
extrinsics_rot <r00..r22, row-major, camera -> base>
extrinsics_trans <tx ty tz, metres>
rms_mm <the extrinsics fit residual, IN THE ROBOT FRAME>
poses <how many correspondences the extrinsics used>
```

⚠ **`extrinsics_*` IS OPTIONAL AND ITS ABSENCE IS NOT AN ERROR** — intrinsics
are measured first and are useful alone (they are what `solve_pnp` needs).

⚠ **`model` IS ABSENT FROM EVERY FILE WRITTEN BEFORE THE FISHEYE MODEL** and
reads as `pinhole`, which is what those files are. A fisheye file's four
`dist` terms are NOT the first four radial-tangential terms: `FisheyeLens`
refuses a `pinhole` file, and a pinhole consumer (`solve_pnp`) must check
`model` before passing `dist`.
`has_extrinsics` says which kind of file you have; a consumer that needs a
robot-frame answer must check it rather than reading zeros as a pose at the
base origin.
"""

from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]

comptime MAGIC = "mojo-rl-camera-calibration"
comptime VERSION = 1


struct CameraCalib(Copyable, Movable, Writable):
    """One camera: what it sees like, and optionally where it is."""

    var name: String
    """The rig's name for it — `front`, `side`, `wrist`. ⚠ THE RIG'S, NOT
    OpenCV's: device indices are assigned by enumeration order and move when a
    camera is unplugged, which is exactly why `act_so101_deploy_real.mojo`
    ships a `--snap` mode to check that device i is really slot i."""

    var device: Int
    """The index it was MEASURED on, kept as a hint and never as identity."""

    var width: Int
    var height: Int

    var fx: Float64
    var fy: Float64
    var cx: Float64
    var cy: Float64

    var model: String
    """`pinhole` (OpenCV's radial-tangential `dist`) or `fisheye`
    (Kannala-Brandt, 4 terms — `vision/fisheye.mojo`)."""

    var dist: List[Float64]
    """OpenCV's distortion vector, however many terms were fitted. Pass it
    straight to `solve_pnp` — which takes a `dist` argument, so a marker pose
    is undistorted for free and needs no image-space undistortion."""

    var rms_px: Float64
    """The intrinsics fit residual. ⚠ A RESIDUAL, NOT AN ACCURACY, and a low
    one with poor coverage is the dangerous combination — see the studio."""

    var has_extrinsics: Bool
    var rot: Mat3d
    """Camera -> base rotation, if `has_extrinsics`."""
    var trans: Vec3d
    """Camera origin in the base frame, metres, if `has_extrinsics`."""
    var rms_mm: Float64
    """The extrinsics fit residual, in millimetres in the ROBOT frame."""
    var poses: Int
    """Correspondences behind the extrinsics. ⚠ THREE FITS EXACTLY AND PROVES
    NOTHING (`extrinsics.MIN_POINTS`), so this number qualifies `rms_mm`."""

    def __init__(
        out self,
        var name: String,
        device: Int,
        width: Int,
        height: Int,
        fx: Float64,
        fy: Float64,
        cx: Float64,
        cy: Float64,
    ):
        self.name = name^
        self.device = device
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.model = String("pinhole")
        self.dist = List[Float64]()
        self.rms_px = 0.0
        self.has_extrinsics = False
        self.rot = Mat3d.identity()
        self.trans = Vec3d.zero()
        self.rms_mm = 0.0
        self.poses = 0

    def k_matrix(self) -> List[Float64]:
        """The 3x3 camera matrix, row-major — `solve_pnp`'s `k` argument."""
        var k = List[Float64]()
        k.append(self.fx)
        k.append(0.0)
        k.append(self.cx)
        k.append(0.0)
        k.append(self.fy)
        k.append(self.cy)
        k.append(0.0)
        k.append(0.0)
        k.append(1.0)
        return k^

    def require_size(self, w: Int, h: Int) raises:
        """Refuse to be used at a size this calibration was not measured at.

        ⚠ CALL THIS ONCE, RIGHT AFTER THE CAMERA REPORTS WHAT IT ACTUALLY
        NEGOTIATED — never after what was REQUESTED. OpenCV substitutes a
        resolution without reporting an error, which is the whole reason
        `cap_props` exists.
        """
        if w != self.width or h != self.height:
            raise (
                String("calibration '")
                + self.name
                + "' was measured at "
                + String(self.width)
                + "x"
                + String(self.height)
                + " but the frame is "
                + String(w)
                + "x"
                + String(h)
                + ". fx and cx are both wrong by the ratio — recalibrate at"
                " this size rather than scaling them."
            )

    def base_from_camera(self, p_cam: Vec3d) raises -> Vec3d:
        """A point seen by the camera, in the robot base frame. Metres."""
        if not self.has_extrinsics:
            raise (
                String("calibration '")
                + self.name
                + "' has no extrinsics — it cannot place anything in the"
                " robot frame. Run the extrinsics capture first."
            )
        return self.rot * p_cam + self.trans

    def write_to(self, mut writer: Some[Writer]):
        writer.write(MAGIC, " ", VERSION, "\n")
        writer.write("name ", self.name, "\n")
        writer.write("device ", self.device, "\n")
        writer.write("size ", self.width, " ", self.height, "\n")
        writer.write(
            "intrinsics ", self.fx, " ", self.fy, " ", self.cx, " ",
            self.cy, "\n",
        )
        writer.write("model ", self.model, "\n")
        if len(self.dist) > 0:
            writer.write("dist")
            for i in range(len(self.dist)):
                writer.write(" ", self.dist[i])
            writer.write("\n")
        writer.write("rms_px ", self.rms_px, "\n")
        if self.has_extrinsics:
            writer.write(
                "extrinsics_rot ",
                self.rot.m00, " ", self.rot.m01, " ", self.rot.m02, " ",
                self.rot.m10, " ", self.rot.m11, " ", self.rot.m12, " ",
                self.rot.m20, " ", self.rot.m21, " ", self.rot.m22, "\n",
            )
            writer.write(
                "extrinsics_trans ",
                self.trans.x, " ", self.trans.y, " ", self.trans.z, "\n",
            )
            writer.write("rms_mm ", self.rms_mm, "\n")
            writer.write("poses ", self.poses, "\n")


def write_calib(path: String, calib: CameraCalib) raises:
    """⚠ WRITTEN IMMEDIATELY, NOT BEHIND A CONFIRMATION. The expensive part of
    a calibration is the poses somebody held an arm through; the cheap part is
    a file."""
    with open(path, "w") as f:
        f.write(String(calib))


def read_calib(path: String) raises -> CameraCalib:
    """Parse one. Unknown keys are IGNORED so a newer writer stays readable.

    Raises:
        If the magic line, the version, `size` or `intrinsics` is missing or
        malformed. ⚠ THOSE FOUR ARE REQUIRED because every one of them is a
        way to apply a calibration to the wrong picture.
    """
    var text: String
    with open(path, "r") as f:
        text = f.read()

    var name = String("")
    var device = -1
    var w = -1
    var h = -1
    var fx = 0.0
    var fy = 0.0
    var cx = 0.0
    var cy = 0.0
    var dist = List[Float64]()
    var model = String("pinhole")
    var rms_px = 0.0
    var rot = Mat3d.identity()
    var trans = Vec3d.zero()
    var rms_mm = 0.0
    var poses = 0
    var seen_magic = False
    var seen_size = False
    var seen_intr = False
    var seen_rot = False
    var seen_trans = False

    var lines = text.split("\n")
    for li in range(len(lines)):
        var parts = lines[li].split()
        if len(parts) == 0:
            continue
        var key = String(parts[0])
        if key.startswith("#"):
            continue
        if key == MAGIC:
            if len(parts) < 2 or Int(String(parts[1])) != VERSION:
                raise (
                    String(path)
                    + ": this reader speaks version "
                    + String(VERSION)
                    + " and the file does not say so"
                )
            seen_magic = True
        elif key == "name" and len(parts) >= 2:
            name = String(parts[1])
        elif key == "device" and len(parts) >= 2:
            device = Int(String(parts[1]))
        elif key == "size" and len(parts) >= 3:
            w = Int(String(parts[1]))
            h = Int(String(parts[2]))
            seen_size = True
        elif key == "intrinsics" and len(parts) >= 5:
            fx = Float64(String(parts[1]))
            fy = Float64(String(parts[2]))
            cx = Float64(String(parts[3]))
            cy = Float64(String(parts[4]))
            seen_intr = True
        elif key == "model" and len(parts) >= 2:
            model = String(parts[1])
            if model != "pinhole" and model != "fisheye":
                raise String(path) + ": unknown camera model '" + model + "'"
        elif key == "dist":
            for i in range(1, len(parts)):
                dist.append(Float64(String(parts[i])))
        elif key == "rms_px" and len(parts) >= 2:
            rms_px = Float64(String(parts[1]))
        elif key == "extrinsics_rot" and len(parts) >= 10:
            var r = List[Float64]()
            for i in range(1, 10):
                r.append(Float64(String(parts[i])))
            rot = Mat3d(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
            seen_rot = True
        elif key == "extrinsics_trans" and len(parts) >= 4:
            trans = Vec3d(
                Float64(String(parts[1])),
                Float64(String(parts[2])),
                Float64(String(parts[3])),
            )
            seen_trans = True
        elif key == "rms_mm" and len(parts) >= 2:
            rms_mm = Float64(String(parts[1]))
        elif key == "poses" and len(parts) >= 2:
            poses = Int(String(parts[1]))

    if not seen_magic:
        raise (
            String(path)
            + ": not a camera calibration (the first line must be '"
            + MAGIC
            + " "
            + String(VERSION)
            + "')"
        )
    if not seen_size:
        raise (
            String(path)
            + ": no 'size' line. A calibration without the image size it was"
            " measured at cannot be checked against a frame, and fx/cx are"
            " meaningless without it."
        )
    if not seen_intr:
        raise String(path) + ": no 'intrinsics' line"

    var out = CameraCalib(name^, device, w, h, fx, fy, cx, cy)
    out.dist = dist^
    out.model = model^
    out.rms_px = rms_px
    # ⚠ BOTH HALVES OR NEITHER. A rotation without a translation is not a
    # partial pose, it is a pose with the origin silently at the base — which
    # reads as a plausible number everywhere it is used.
    if seen_rot != seen_trans:
        raise (
            String(path)
            + ": has one half of the extrinsics. A rotation without a"
            " translation places the camera at the robot's origin, which"
            " looks like an answer."
        )
    if seen_rot:
        out.has_extrinsics = True
        out.rot = rot
        out.trans = trans
        out.rms_mm = rms_mm
        out.poses = poses
    return out^
