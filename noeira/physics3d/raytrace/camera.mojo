"""`mj_camlight`'s camera half, as tensor reads — the batched twin.

`kinematics/camera_frame.mojo` composes a camera's world pose from `Vec3`/
`Quat` VALUES, at `float64`, on the host. It says in its own docstring that the
call is worth revisiting "when batched camera observations land — a ray tracer
over `[N_ENVS, ...]` wants the pose per env on the device". This is that: the
same five modes, read out of `Model.cameras` and `Data.xpos`/`xquat`/
`subtree_com` with an `env` index, so ONE implementation serves the host and a
Metal/CUDA kernel. It is the same shape `ray/model.mojo` settled on and for the
same reason — a thread owns one PIXEL, not a scene, so nothing may be hoisted
into a per-thread array (`87960e10`, the fourth Metal miscompute of that class
in this engine).

⚠⚠ A FRAME, NOT A QUATERNION, AND THAT IS NOT A STYLE CHOICE.
`mjCAMLIGHT_TARGETBODY` builds `cam_xmat` from three cross products
(`engine_core_smooth.c:407-424`) and never forms a quaternion. Returning the
three axes keeps that branch exact instead of round-tripping it through a
matrix-to-quaternion conversion whose sign convention is one more thing to get
wrong.

⚠ THE CAMERA LOOKS DOWN ITS OWN **-Z** (`mjCCamera`), so `zaxis` below points
BACKWARDS out of the lens. A `<rangefinder>` fires along its site's **+Z**. The
two conventions sit three hundred lines apart in the same reference file and
are opposite; `sensors/rangefinder.mojo` carries the same warning from the
other side.
"""

from std.math import sqrt
from layout import TileTensor, TensorLayout

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..gpu.constants import (
    MODEL_CAM_SIZE,
    CAM_IDX_BODY,
    CAM_IDX_POS_X,
    CAM_IDX_POS_Y,
    CAM_IDX_POS_Z,
    CAM_IDX_QUAT_X,
    CAM_IDX_QUAT_Y,
    CAM_IDX_QUAT_Z,
    CAM_IDX_QUAT_W,
    CAM_IDX_FOVY,
    CAM_IDX_TAN_HALF_FOVY,
    CAM_MODE_FIXED,
    CAM_MODE_TRACK,
    CAM_MODE_TRACKCOM,
    CAM_MODE_TARGETBODY,
    CAM_MODE_TARGETBODYCOM,
    CAM_IDX_MODE,
    CAM_IDX_TARGET_BODY,
    CAM_IDX_POS0_X,
    CAM_IDX_POS0_Y,
    CAM_IDX_POS0_Z,
    CAM_IDX_POSCOM0_X,
    CAM_IDX_POSCOM0_Y,
    CAM_IDX_POSCOM0_Z,
    CAM_IDX_QUAT0_X,
    CAM_IDX_QUAT0_Y,
    CAM_IDX_QUAT0_Z,
    CAM_IDX_QUAT0_W,
)

# ⚠ RE-EXPORTED, NOT REDEFINED. The values live in `gpu/constants.mojo`
# alongside the record they are written into, and `fields_build` carries a
# comptime assert that they equal the parser's `CAM_MODE_*`. The `RT_` prefix
# is only so a file importing both spellings can tell them apart.
comptime RT_CAM_MODE_FIXED: Int = CAM_MODE_FIXED
comptime RT_CAM_MODE_TRACK: Int = CAM_MODE_TRACK
comptime RT_CAM_MODE_TRACKCOM: Int = CAM_MODE_TRACKCOM
comptime RT_CAM_MODE_TARGETBODY: Int = CAM_MODE_TARGETBODY
comptime RT_CAM_MODE_TARGETBODYCOM: Int = CAM_MODE_TARGETBODYCOM


@fieldwise_init
struct CameraFrame[DTYPE: DType](Copyable, Movable):
    """`cam_xpos` and the three COLUMNS of `cam_xmat`, plus the frustum.

    `xaxis`/`yaxis`/`zaxis` are the camera's own +X/+Y/+Z expressed in world
    coordinates — MuJoCo stores them as the columns of `cam_xmat`, which is
    what `cam_xmat @ v_local` multiplies. The optical axis is **-zaxis**.
    """

    var pos: Vec3Generic[Self.DTYPE]
    var xaxis: Vec3Generic[Self.DTYPE]
    var yaxis: Vec3Generic[Self.DTYPE]
    var zaxis: Vec3Generic[Self.DTYPE]
    var fovy: Scalar[Self.DTYPE]
    """VERTICAL field of view in DEGREES, straight off `mjModel.cam_fovy`.

    ⚠ CARRIED FOR INSPECTION, NOT USED BY `camera_pixel_ray`. The frustum is
    built from `tan_half_fovy` below, because `std.math.tan` is CPU-only."""

    var tan_half_fovy: Scalar[Self.DTYPE]
    """`tan(fovy/2)` in radians, baked at model build — `CAM_IDX_TAN_HALF_FOVY`.
    """


@always_inline
def _axes_of[
    DTYPE: DType
](q: QuatGeneric[DTYPE]) -> Tuple[
    Vec3Generic[DTYPE], Vec3Generic[DTYPE], Vec3Generic[DTYPE]
] where DTYPE.is_floating_point():
    """The three columns of the rotation matrix `q` denotes."""
    return (
        q.rotate_vec(Vec3Generic[DTYPE](1, 0, 0)),
        q.rotate_vec(Vec3Generic[DTYPE](0, 1, 0)),
        q.rotate_vec(Vec3Generic[DTYPE](0, 0, 1)),
    )


def camera_world_frame[
    DTYPE: DType,
    L_CAM: TensorLayout,
    L_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_COM: TensorLayout,
](
    cameras: TileTensor[DTYPE, L_CAM, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_COM, MutAnyOrigin],
    env: Int,
    cam: Int,
) -> CameraFrame[DTYPE] where DTYPE.is_floating_point():
    """`d->cam_xpos[cam]` / `d->cam_xmat[cam]` for one lane.

    ⚠ `xpos`/`xquat`/`subtree_com` ARE `[BATCH, NBODY*k]`, indexed
    `[env, body*k + c]` — the layout every other batched consumer in this
    engine reads, including `ray_model`. Passing a single-env slab means
    passing `env = 0` with a `[1, NBODY*k]` layout, not a 1-D tensor.

    ⚠ TRACK and TRACKCOM READ THE REFERENCE POSE and return the wrong picture
    (camera at the body origin, looking along world -Z) if
    `init_camera_reference` never ran. That is checked on the HOST, at the
    renderer's entry point, because a kernel cannot raise — see
    `CAM_IDX_REF_SET`.
    """
    var cb = cam * MODEL_CAM_SIZE
    var body = Int(rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_BODY]))
    var mode = Int(rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_MODE]))
    var fovy = rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_FOVY])
    var thf = rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_TAN_HALF_FOVY])

    var lp = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_POS_X]),
        rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_POS_Y]),
        rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_POS_Z]),
    )
    # ⚠ THE RECORD IS (x, y, z, w) AND THE CONSTRUCTOR IS (w, x, y, z).
    var lq = QuatGeneric[DTYPE](
        rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_QUAT_W]),
        rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_QUAT_X]),
        rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_QUAT_Y]),
        rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_QUAT_Z]),
    )

    var bp = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](xpos[env, body * 3 + 0]),
        rebind[Scalar[DTYPE]](xpos[env, body * 3 + 1]),
        rebind[Scalar[DTYPE]](xpos[env, body * 3 + 2]),
    )
    # ⚠ `Data.xquat` IS PACKED (x, y, z, w) AND `Quat` TAKES (w, x, y, z) —
    # the same swap `ray/model.mojo` spells three lines from here. Reading it
    # in order gives a rotation that is wrong by a quarter turn about a
    # plausible-looking axis, which renders a picture rather than an error.
    var bq = QuatGeneric[DTYPE](
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3]),
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0]),
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1]),
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2]),
    )

    # ── `mj_local2Global`, which EVERY mode runs first ────────────────────
    var pos = bp + bq.rotate_vec(lp)
    var wq = bq * lq
    var ax = _axes_of[DTYPE](wq)
    var xa = ax[0]
    var ya = ax[1]
    var za = ax[2]

    if mode == RT_CAM_MODE_TRACK or mode == RT_CAM_MODE_TRACKCOM:
        # Fixed GLOBAL orientation — the reference frame, not the body's.
        var q0 = QuatGeneric[DTYPE](
            rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_QUAT0_W]),
            rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_QUAT0_X]),
            rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_QUAT0_Y]),
            rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_QUAT0_Z]),
        )
        var a0 = _axes_of[DTYPE](q0)
        xa = a0[0]
        ya = a0[1]
        za = a0[2]
        if mode == RT_CAM_MODE_TRACK:
            pos = bp + Vec3Generic[DTYPE](
                rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_POS0_X]),
                rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_POS0_Y]),
                rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_POS0_Z]),
            )
        else:
            var com = Vec3Generic[DTYPE](
                rebind[Scalar[DTYPE]](subtree_com[env, body * 3 + 0]),
                rebind[Scalar[DTYPE]](subtree_com[env, body * 3 + 1]),
                rebind[Scalar[DTYPE]](subtree_com[env, body * 3 + 2]),
            )
            pos = com + Vec3Generic[DTYPE](
                rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_POSCOM0_X]),
                rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_POSCOM0_Y]),
                rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_POSCOM0_Z]),
            )
    elif (
        mode == RT_CAM_MODE_TARGETBODY or mode == RT_CAM_MODE_TARGETBODYCOM
    ):
        var tb = Int(
            rebind[Scalar[DTYPE]](cameras[cb + CAM_IDX_TARGET_BODY])
        )
        # ⚠ `if (id1 >= 0)` — a camera declaring `mode="targetbody"` with no
        # resolvable `target=` keeps the FIXED frame. Reproduced, not
        # improved: the reference leaves `cam_xmat` as `mj_local2Global` set
        # it, and an unresolved name is already announced at parse time.
        if tb >= 0:
            var tgt: Vec3Generic[DTYPE]
            if mode == RT_CAM_MODE_TARGETBODY:
                tgt = Vec3Generic[DTYPE](
                    rebind[Scalar[DTYPE]](xpos[env, tb * 3 + 0]),
                    rebind[Scalar[DTYPE]](xpos[env, tb * 3 + 1]),
                    rebind[Scalar[DTYPE]](xpos[env, tb * 3 + 2]),
                )
            else:
                tgt = Vec3Generic[DTYPE](
                    rebind[Scalar[DTYPE]](subtree_com[env, tb * 3 + 0]),
                    rebind[Scalar[DTYPE]](subtree_com[env, tb * 3 + 1]),
                    rebind[Scalar[DTYPE]](subtree_com[env, tb * 3 + 2]),
                )
            # `matT[6..8] = normalize(cam_xpos - pos)` — the BACK axis, since
            # the lens looks down -Z.
            var zz = pos - tgt
            var zl = zz.length()
            if zl > Scalar[DTYPE](1e-12):
                za = zz / zl
                # `mji_cross(matT, matT+3, matT+6)` with `matT+3 = (0,0,1)`.
                var world_up = Vec3Generic[DTYPE](0, 0, 1)
                var xx = world_up.cross(za)
                var xl = xx.length()
                # ⚠ A DIVERGENCE, AND A DELIBERATE ONE. MuJoCo normalizes
                # unconditionally, so a camera looking straight down gets a
                # zero-length x-axis and `mju_normalize3` leaves it at zero —
                # a SINGULAR frame that renders a degenerate image. A
                # top-down `targetbody` camera is a thing people write, so
                # pick a well-defined roll instead of a broken one.
                if xl > Scalar[DTYPE](1e-9):
                    xa = xx / xl
                else:
                    xa = Vec3Generic[DTYPE](1, 0, 0)
                # `mji_cross(matT+3, matT+6, matT)`.
                ya = za.cross(xa)
                var yl = ya.length()
                if yl > Scalar[DTYPE](1e-12):
                    ya = ya / yl

    return CameraFrame[DTYPE](pos, xa, ya, za, fovy, thf)


@always_inline
def camera_sample_ray[
    DTYPE: DType
](
    frame: CameraFrame[DTYPE],
    width: Int,
    height: Int,
    px: Int,
    py: Int,
    ox: Scalar[DTYPE],
    oy: Scalar[DTYPE],
) -> Vec3Generic[DTYPE] where DTYPE.is_floating_point():
    """The world-space direction through pixel `(px, py)` offset by
    `(ox, oy)` pixels from its centre (`+oy` is DOWN the image), NORMALISED.
    `camera_pixel_ray` is this at `(0, 0)`; the multisample positions use it
    (`render.render_pixel`'s `SAMPLES`).

    `render_util.compute_ray`'s `fovy` branch. The intrinsics branch (a
    `<camera sensorsize=... focal=...>`) is not carried yet: nothing in this
    tree parses those attributes, so implementing the branch would be
    unreachable code with no gate.

    ⚠ `py` COUNTS DOWN FROM THE TOP. `v = (py + 0.5)/height` and `y` runs from
    `+half_h` to `-half_h`, so row 0 is the TOP of the image — the raster
    convention every consumer of a `[H, W]` buffer expects, and the one the
    reference uses. Flipping it renders the world upside down, which is a
    picture that looks plausible in a unit test and wrong in a video.

    ⚠ `znear` IS ABSENT ON PURPOSE. It scales all three components of the
    local direction equally and the result is normalised, so it cannot change
    the direction — and the planar depth `render_pixel` reports is taken from
    the NORMALISED direction's z, so it cannot change that either. The
    reference threads `rc.znear` through this call; here it would be a
    parameter no output depends on.
    """
    # ⚠ `tan` IS NOT CALLED HERE — it is CPU-only and this body runs in a
    # kernel. `tan(fovy/2)` is baked into the record at model build; see
    # `CAM_IDX_TAN_HALF_FOVY`.
    var half_h = frame.tan_half_fovy
    var half_w = half_h * (Scalar[DTYPE](width) / Scalar[DTYPE](height))

    var u = (Scalar[DTYPE](px) + Scalar[DTYPE](0.5) + ox) / Scalar[DTYPE](width)
    var v = (Scalar[DTYPE](py) + Scalar[DTYPE](0.5) + oy) / Scalar[DTYPE](height)
    var lx = -half_w + Scalar[DTYPE](2) * half_w * u
    var ly = half_h - Scalar[DTYPE](2) * half_h * v

    # `cam_xmat @ (lx, ly, -1)`, i.e. down the optical axis (-Z) with the
    # frustum offsets on the other two.
    var d = frame.xaxis * lx + frame.yaxis * ly - frame.zaxis
    var n = d.length()
    if n > Scalar[DTYPE](0):
        return d / n
    return -frame.zaxis


@always_inline
def camera_pixel_ray[
    DTYPE: DType
](
    frame: CameraFrame[DTYPE],
    width: Int,
    height: Int,
    px: Int,
    py: Int,
) -> Vec3Generic[DTYPE] where DTYPE.is_floating_point():
    """The world-space direction of pixel `(px, py)`'s CENTRE, NORMALISED —
    `camera_sample_ray` at offset `(0, 0)`. See it for the conventions."""
    return camera_sample_ray[DTYPE](
        frame, width, height, px, py, Scalar[DTYPE](0), Scalar[DTYPE](0)
    )