"""OpenCV bindings — camera capture today, detection and calibration next.

    pixi run build-opencv       # ONCE, builds the C++ shim this binds to
    from noeira.vision.opencv import VideoCapture, opencv_shim_available

Every function here is a thin call into `opencv_shim.cpp`, a flat C API over
OpenCV 5. `docs/OPENCV_SHIM_SCOPE.md` carries the full surface and the reasons;
what matters at the call site:

⚠ REQUIRES A BUILT SHIM. `pixi run build-opencv` produces `libmojo_cv.dylib`
beside this file. It is NOT tracked in git, so a fresh clone has to build it,
and the failure mode is a dlopen ABORT at the first call rather than a compile
error. `opencv_shim_available()` answers the question without touching FFI.

⚠⚠ FRAMES COME BACK BGR AND HWC. The rest of this project — the ACT store
above all — holds images CHW and RGB. A silent channel swap raises no error
anywhere and simply produces wrong answers, which is the exact class of defect
this tree keeps recording. `bgr_hwc_to_rgb_chw` is the ONE place that
conversion is allowed to happen, so that it has a fixed point to be gated at.

⚠ NOTHING IS VENDORED. Unlike Dear ImGui, OpenCV is already in every pixi env
as a full C++ build, so the shim compiles against the env and nothing is
cloned. That also means the dylib carries an `-rpath` into the env: it is a
development artifact, not a redistributable.
"""

from std.os import abort, getenv
from std.sys import CompilationTarget
from std.pathlib import Path
from std.ffi import _Global, OwnedDLHandle, _get_dylib_function, c_char

comptime Ptr = Pointer


def untracked[
    T: AnyType, o: Origin
](p: Pointer[T, o]) -> Pointer[T, MutUntrackedOrigin]:
    """Re-key a pointer's origin for an FFI call. Same helper as
    `render/sdl`, duplicated rather than imported so `vision` does not depend
    on `render` for three lines."""
    return rebind[Pointer[T, MutUntrackedOrigin]](p)


# ═══════════════════════════════════════════════════════════════════════════
# status codes — mirror `opencv_shim.cpp`
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠ NEGATIVE IS AN ERROR, ZERO IS OK, and positive is never a status. Anything
# a caller wants beyond success comes out through a pointer, so a status can
# never be confused with a value.

comptime CV_OK: Int32 = 0
comptime CV_ERR_CV: Int32 = -1
comptime CV_ERR_STD: Int32 = -2
comptime CV_ERR_UNKNOWN: Int32 = -3
comptime CV_ERR_ARG: Int32 = -4
comptime CV_ERR_CAPACITY: Int32 = -5
comptime CV_ERR_NO_FRAME: Int32 = -6

# ── OpenCV enum values, READ OFF THIS BUILD, not remembered ─────────────────
#
# ⚠⚠ `SOLVEPNP_IPPE_SQUARE` IS 5 HERE AND WAS 7 IN OpenCV 4.x. The enum was
# renumbered. A stale 7 does not fail — it selects a DIFFERENT ESTIMATOR and
# returns a pose, which is the worst possible failure mode. Every constant
# below was printed from the installed cv2, and any addition must be too.
comptime SOLVEPNP_ITERATIVE: Int = 0
comptime SOLVEPNP_IPPE_SQUARE: Int = 5
"""The four-coplanar-corner estimator a square fiducial actually is.

⚠ NOT INTERCHANGEABLE WITH `SOLVEPNP_ITERATIVE`, which on four coplanar points
is a different and worse estimator that still returns an answer."""
comptime DICT_4X4_50: Int = 0

comptime CALIB_ZERO_TANGENT_DIST: Int = 8
comptime CALIB_FIX_K3: Int = 128
"""⚠ THE CALIBRATION FLAGS ARE PART OF THE ANSWER, NOT A TUNING KNOB.

They decide how long the distortion vector is and which coefficients are fitted
at all, so two runs with different flags are not two estimates of one quantity
— they are answers to different questions. Whatever a gate pins, the caller
must pass.

⚠⚠ `CALIB_FIX_K3` IS 128, AND THIS FILE FIRST SAID 64. That is the second time
a remembered OpenCV constant was wrong here (see `SOLVEPNP_IPPE_SQUARE`), and
it is the same failure mode: the wrong flag does not raise, it fits a DIFFERENT
MODEL and returns a plausible calibration — fx 602.311 against cv2's 602.379,
a disagreement in the fourth digit that no sanity check on "is fx near 600"
would ever catch. The bit-equality gate caught it on the first run. PRINT EVERY
CONSTANT FROM THE INSTALLED cv2."""
comptime FISHEYE_CALIB_USE_INTRINSIC_GUESS: Int = 1
comptime FISHEYE_CALIB_FIX_K2: Int = 0x40
comptime FISHEYE_CALIB_FIX_K3: Int = 0x80
comptime FISHEYE_CALIB_FIX_K4: Int = 0x800
comptime FISHEYE_CALIB_RECOMPUTE_EXTRINSIC: Int = 1 << 23
comptime FISHEYE_CALIB_CHECK_COND: Int = 1 << 24
comptime FISHEYE_CALIB_FIX_SKEW: Int = 1 << 25
"""`cv::CALIB_*` for `cv::fisheye::calibrate`, read off OpenCV 5.0's
`calib.hpp` (they are shared enum values there, "for fisheye model only")."""


# ═══════════════════════════════════════════════════════════════════════════
# dylib loading — the `render/imgui` and `io/serial` pattern
# ═══════════════════════════════════════════════════════════════════════════


def _lib_name() -> String:
    comptime if CompilationTarget.is_macos():
        return String("libmojo_cv.dylib")
    elif CompilationTarget.is_linux():
        return String("libmojo_cv.so")
    else:
        comptime assert False, "OS is not supported"


def _candidates() -> List[String]:
    """Where to look, most explicit first.

    Kept separate from `_init_handle` so `opencv_shim_available()` can check
    the SAME list without dlopening anything — a probe consulting a different
    list would answer a different question than the loader asks.
    """
    var name = _lib_name()
    var out = List[String]()
    var override = getenv("NOEIRA_OPENCV_LIB")
    if override.byte_length() > 0:
        out.append(override)
    var root = getenv("PIXI_PROJECT_ROOT")
    if root.byte_length() > 0:
        out.append(root + "/noeira/vision/opencv/" + name)
    out.append("noeira/vision/opencv/" + name)
    out.append(name)
    return out^


def opencv_shim_available() -> Bool:
    """True when the shim can be found WITHOUT dlopening it.

    `_Global` aborts the process on a missing library, which is right for a
    hard dependency and the wrong first impression for an optional one.
    """
    var c = _candidates()
    for i in range(len(c)):
        if Path(c[i]).exists():
            return True
    return False


def _init_handle() -> OwnedDLHandle:
    """Non-raising, as `_Global` demands; aborts with the paths it tried
    rather than returning an uninitialised handle, which would turn a missing
    library into a segfault at the first unrelated call."""
    var c = _candidates()
    for i in range(len(c)):
        try:
            return OwnedDLHandle(c[i])
        except:
            pass
    var tried = String("")
    for i in range(len(c)):
        tried += "\n  - " + c[i]
    abort(
        "OpenCV shim not found. Tried:"
        + tried
        + "\nBuild it with `pixi run build-opencv`, or set"
        + " NOEIRA_OPENCV_LIB=/path/to/"
        + _lib_name()
    )


comptime lib = _Global["NOEIRA_OPENCV", _init_handle]()


# ═══════════════════════════════════════════════════════════════════════════
# A — lifecycle
# ═══════════════════════════════════════════════════════════════════════════


def _c_string(ref buf: List[UInt8]) -> String:
    """A NUL-terminated C buffer as a String, stopping at the NUL."""
    var out = String("")
    for i in range(len(buf)):
        if buf[i] == 0:
            break
        out += chr(Int(buf[i]))
    return out^


def cv_last_error() raises -> String:
    """The last message the shim recorded, or "" if it has recorded none.

    ⚠ THREAD-LOCAL ON THE C SIDE, so this reports what THIS thread's last call
    failed with. A shared buffer would be a data race whose symptom is a wrong
    message rather than a crash.
    """
    var p = _get_dylib_function[
        lib, "nra_cv_last_error", def() thin -> Ptr[c_char, MutUntrackedOrigin]
    ]()()
    # ⚠ NEVER NULL by construction — the shim returns "" when it has nothing
    # to say, so a caller can print this unconditionally.
    return String(unsafe_from_utf8_ptr=p)


def _check(status: Int32, what: String) raises:
    """Turn a status code into an exception carrying the C-side message."""
    if status == CV_OK:
        return
    var msg = String("opencv: ") + what + " failed (" + String(status) + ")"
    var detail = cv_last_error()
    if detail.byte_length() > 0:
        msg += ": " + detail
    raise msg


def cv_version() raises -> Tuple[Int, Int]:
    """`(major, minor)` of the OpenCV the shim was LINKED against.

    ⚠ A REAL COMPATIBILITY QUESTION, not a banner. OpenCV 5 moved `solvePnP`
    and `calibrateCamera` between headers and DELETED
    `estimatePoseSingleMarkers`, `calibrateCameraCharuco` and
    `calibrateHandEye`; a 4.x here would compile and then behave differently.
    Doubles as the cheapest possible "did the dylib actually load" probe.
    """
    var major = Int32(0)
    var minor = Int32(0)
    var st = _get_dylib_function[
        lib,
        "nra_cv_version",
        def(
            Ptr[Int32, MutUntrackedOrigin], Ptr[Int32, MutUntrackedOrigin]
        ) thin -> Int32,
    ]()(untracked(Ptr(to=major)), untracked(Ptr(to=minor)))
    _check(st, "version")
    return (Int(major), Int(minor))


def cv_set_num_threads(n: Int) raises:
    """⚠ NOT A PERFORMANCE KNOB — A GATE REQUIREMENT.

    `calibrateCamera` is an iterative LM fit and OpenCV's `parallel_for_` can
    change reduction order with the thread count. Bit-equality against Python
    `cv2` is a claim about identical inputs AND identical scheduling, so both
    sides of a parity gate call this with 1.
    """
    _check(
        _get_dylib_function[
            lib, "nra_cv_set_num_threads", def(Int32) thin -> Int32
        ]()(Int32(n)),
        "set_num_threads",
    )


# ═══════════════════════════════════════════════════════════════════════════
# B — capture
# ═══════════════════════════════════════════════════════════════════════════


struct VideoCapture(Movable):
    """A camera or a video file, yielding BGR HWC frames.

    ⚠ CLOSE IT EXPLICITLY. There is no destructor here: a capture left open
    holds a camera device against the whole machine, and `close()` is
    idempotent so a `try/finally` around a control loop is the right shape.
    (`examples/so101/deploy_reach_real.mojo` records what happens when a
    `finally` does NOT run on an abort — the same hazard, one device over.)
    """

    var _h: Int
    """The C `void*`, as an address.

    ⚠ AN `Int`, NOT A `Pointer`, AND THAT IS DELIBERATE. Mojo's `Pointer` is a
    SAFE type: it carries an origin and promises a dereferenceable pointee. We
    never dereference this — it is an opaque token minted and freed by
    `opencv_shim.cpp` — so modelling it as a `Pointer` would claim something
    false, and Mojo 1.0 gives it neither a null value nor a raw-address
    constructor anyway. `0` means closed. A `void*` and an `Int` are both one
    64-bit register in the C ABI, so the FFI signatures below are exact."""
    var width: Int
    var height: Int
    var channels: Int
    var fps: Float64
    var frame_count: Int
    """0 for a live device; only meaningful for a file."""
    var node: String
    """For a path-opened camera, the `/dev/videoN` the name resolved to —
    empty otherwise. ⚠ WORTH PRINTING: a udev name is stable, the node behind
    it is not, and which physical camera answered is the one thing a swapped
    cable changes silently."""

    def __init__(out self, _h: Int) raises:
        """Private — use `from_file`, `device` or `closed`."""
        self._h = _h
        self.width = 0
        self.height = 0
        # ⚠ ASSUMED, THEN CONFIRMED. The channel count is not a capture
        # property in OpenCV; it is a property of the frame that comes back, so
        # `read` overwrites this with what actually arrived.
        self.channels = 3
        self.fps = 0.0
        self.frame_count = 0
        self.node = String("")
        # ⚠ A CLOSED CAPTURE HAS NO PROPERTIES TO READ. Asking the shim about
        # handle 0 is an error, not a zero, so the closed state must skip it.
        if _h != 0:
            self._refresh_props()

    @staticmethod
    def closed() raises -> Self:
        """A capture that owns nothing, for a variable declared before its
        source is known.

        ⚠ THIS EXISTS BECAUSE THE OBVIOUS PLACEHOLDER IS A LANDMINE:
        `VideoCapture.from_file("")` RAISES — an empty path is not a file —
        so using it to declare a variable kills the program at startup rather
        than at first use. `close()` already produces this state; this makes it
        constructible.
        """
        return Self(0)

    def is_open(self) -> Bool:
        return self._h != 0

    @staticmethod
    def from_file(path: String) raises -> Self:
        """Open a video file.

        ⚠ THIS IS WHAT THE GATE USES. A capture path tested only against a live
        camera has no gate at all — the frames are never the same twice, so
        there is nothing to compare against. Decoding a committed file through
        the same dylib on both sides IS comparable, bit for bit.
        """
        var p = path
        var h = _get_dylib_function[
            lib,
            "nra_cv_cap_open_file",
            def(Ptr[c_char, MutUntrackedOrigin]) thin -> Int,
        ]()(untracked(p.as_c_string_span().ptr()))
        if h == 0:
            raise String("opencv: cannot open ") + path + ": " + cv_last_error()
        return Self(h)

    @staticmethod
    def device(
        index: Int, width: Int = 0, height: Int = 0, fps: Float64 = 0.0
    ) raises -> Self:
        """Open a live camera.

        ⚠ THE SIZE AND RATE ARE REQUESTS, NOT SETTINGS. A camera is free to
        ignore them, and OpenCV reports no error when it does — which is why
        `width`/`height`/`fps` on the returned struct are read back from the
        device rather than echoed from these arguments. A buffer sized from
        what you ASKED for, filled by a device that gave something else, is a
        defect that surfaces much later as a wrong answer.
        """
        var h = _get_dylib_function[
            lib,
            "nra_cv_cap_open",
            def(Int32, Int32, Int32, Float64) thin -> Int,
        ]()(Int32(index), Int32(width), Int32(height), fps)
        if h == 0:
            raise String("opencv: cannot open device ") + String(
                index
            ) + ": " + cv_last_error()
        return Self(h)

    @staticmethod
    def device_path(
        path: String,
        width: Int = 0,
        height: Int = 0,
        fps: Float64 = 0.0,
        fourcc: String = String(""),
    ) raises -> Self:
        """Open a live camera by DEVICE PATH — `/dev/soarm_cam_wrist`.

        ⚠ WHY A PATH AT ALL. A camera index is what moves between boots: the
        two SO-101 cameras are the same model with the same burned-in serial,
        so only the USB topology tells them apart, and `video0`/`video2` are
        assigned in enumeration order. `/etc/udev/rules.d/99-soarm.rules` gives
        each a stable name; opening by index throws that away
        (`docs/JETSON_DEPLOYMENT.md` §3). Feeding the policy the WRONG CAMERA
        is not a crash — it is a well-formed observation of the wrong thing.

        ⚠⚠ ON LINUX THE PATH IS RESOLVED TO AN INDEX FIRST, and that is not a
        shortcut — OpenCV's V4L2 backend CANNOT capture by name. Measured on
        the Orin (OpenCV 5.0, JetPack 6.2): opening `/dev/soarm_cam_overhead` by
        name prints "backend is generally available but can't be used to
        capture by name" and fails. The udev symlink is still what makes the
        name stable; it is dereferenced here to whatever `videoN` it points at
        today, which is all a symlink was ever for.

        ⚠ NOT `from_file`. That opens with CAP_ANY, which on a `/dev/video*`
        path can be won by FFMPEG's v4l2 demuxer instead of the V4L2 backend —
        a different capture path with its own format negotiation. This names
        the backend; see `opencv_shim.cpp`.

        `fourcc` is a four-character pixel format (`"MJPG"`, `"YUYV"`), empty
        to leave the device's default alone. Like the size it is a REQUEST:
        read back what was negotiated with `fourcc()`.
        """
        var p = path
        var f = fourcc
        # The node the name resolved to, filled by the shim so a caller can
        # SAY which device a stable name landed on this boot.
        var node = List[UInt8](length=256, fill=0)
        var h = _get_dylib_function[
            lib,
            "nra_cv_cap_open_path",
            def(
                Ptr[c_char, MutUntrackedOrigin],
                Int32,
                Int32,
                Float64,
                Ptr[c_char, MutUntrackedOrigin],
                Ptr[UInt8, MutUntrackedOrigin],
                Int32,
            ) thin -> Int,
        ]()
        # ⚠ An empty fourcc is passed as an empty STRING, not NULL: the shim
        # checks the length, so "" leaves the device's format alone exactly as
        # NULL would, and nothing here has to build a null pointer.
        var handle = h(
            untracked(p.as_c_string_span().ptr()),
            Int32(width),
            Int32(height),
            fps,
            untracked(f.as_c_string_span().ptr()),
            untracked(Ptr(to=node[0])),
            Int32(256),
        )
        var where = _c_string(node)
        if handle == 0:
            raise (
                String("opencv: cannot open device ") + path
                + (" (-> " + where + ")" if where.byte_length() > 0
                   and where != path else String(""))
                + ": " + cv_last_error()
            )
        var out = Self(handle)
        out.node = where^
        return out^

    def fourcc(self) raises -> String:
        """The pixel format the device actually negotiated, or "" if unknown.

        ⚠ WORTH PRINTING BEFORE A POLICY RUNS. Recording in one format and
        deploying in another shows the policy different pixels than it trained
        on, and nothing else in the stack reports it.
        """
        if self._h == 0:
            return String("")
        var buf = List[UInt8](length=5, fill=0)
        var st = _get_dylib_function[
            lib,
            "nra_cv_cap_fourcc",
            def(Int, Ptr[UInt8, MutUntrackedOrigin], Int32) thin -> Int32,
        ]()(self._h, untracked(Ptr(to=buf[0])), Int32(5))
        _check(st, "cap_fourcc")
        var out = String("")
        for i in range(4):
            if buf[i] == 0:
                break
            out += chr(Int(buf[i]))
        return out^

    def _refresh_props(mut self) raises:
        var w = Int32(0)
        var h = Int32(0)
        var f = Float64(0.0)
        var n = Int32(0)
        var st = _get_dylib_function[
            lib,
            "nra_cv_cap_props",
            def(
                Int,
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            self._h,
            untracked(Ptr(to=w)),
            untracked(Ptr(to=h)),
            untracked(Ptr(to=f)),
            untracked(Ptr(to=n)),
        )
        _check(st, "cap_props")
        self.width = Int(w)
        self.height = Int(h)
        self.fps = f
        self.frame_count = Int(n)

    def frame_bytes(self) -> Int:
        """Bytes one frame needs, from the CURRENT reported geometry."""
        return self.width * self.height * self.channels

    def read(mut self, mut out: List[UInt8]) raises -> Bool:
        """Read one frame into `out`, resizing it to fit. BGR, HWC.

        Returns False at end of stream — which is an outcome, not an error, and
        the only non-exceptional way this returns False. Anything else raises.
        """
        var need = self.frame_bytes()
        if need <= 0:
            # A device that reported no geometry: ask for a generous buffer and
            # let the first frame tell us the truth.
            need = 1920 * 1080 * 3
        if len(out) < need:
            out.resize(need, 0)

        var w = Int32(0)
        var h = Int32(0)
        var c = Int32(0)
        var st = _get_dylib_function[
            lib,
            "nra_cv_cap_read",
            def(
                Int,
                Ptr[UInt8, MutUntrackedOrigin],
                Int32,
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            self._h,
            untracked(Ptr(to=out[0])),
            Int32(len(out)),
            untracked(Ptr(to=w)),
            untracked(Ptr(to=h)),
            untracked(Ptr(to=c)),
        )
        if st == CV_ERR_NO_FRAME:
            return False
        # ⚠ THE SHIM REPORTS THE GEOMETRY BEFORE IT CHECKS CAPACITY, so a
        # too-small buffer is recoverable: adopt what it told us and retry once.
        # This is the live-camera case where the negotiated size differed from
        # the requested one.
        if st == CV_ERR_CAPACITY:
            self.width = Int(w)
            self.height = Int(h)
            self.channels = Int(c)
            out.resize(self.frame_bytes(), 0)
            return self.read(out)
        _check(st, "cap_read")
        self.width = Int(w)
        self.height = Int(h)
        self.channels = Int(c)
        return True

    def close(mut self):
        """Release the device. Idempotent — the shim ignores a stale handle."""
        if self._h == 0:
            return
        try:
            _get_dylib_function[
                lib,
                "nra_cv_cap_close",
                def(Int) thin -> None,
            ]()(self._h)
        except:
            # Nothing to report to and nothing to retry; dropping the device
            # would be worse than dropping the message.
            pass
        self._h = Int()


# ═══════════════════════════════════════════════════════════════════════════
# the ONE conversion
# ═══════════════════════════════════════════════════════════════════════════


def bgr_hwc_to_rgb_chw(
    src: List[UInt8], mut dst: List[UInt8], width: Int, height: Int
) raises:
    """OpenCV's frame layout to this project's. BGR HWC -> RGB CHW.

    ⚠⚠ ONE FUNCTION, ON PURPOSE. Both halves of this — the channel ORDER and
    the memory LAYOUT — are silent when wrong: no size changes, no error is
    raised, and the image still looks like an image. Inlining it at each call
    site is how two call sites end up disagreeing, so every consumer of a
    captured frame goes through here and this is what the gate pins.
    """
    var n = width * height
    if len(src) < n * 3:
        raise String("bgr_hwc_to_rgb_chw: source holds ") + String(
            len(src)
        ) + " bytes, need " + String(n * 3)
    if len(dst) < n * 3:
        dst.resize(n * 3, 0)
    for i in range(n):
        # src is B,G,R interleaved; dst is R plane, G plane, B plane.
        dst[i] = src[i * 3 + 2]
        dst[n + i] = src[i * 3 + 1]
        dst[2 * n + i] = src[i * 3]


# ═══════════════════════════════════════════════════════════════════════════
# C — marker detection
# ═══════════════════════════════════════════════════════════════════════════


def imread(path: String, mut out: List[UInt8]) raises -> Tuple[Int, Int, Int]:
    """Read an image file as BGR HWC, returning `(width, height, channels)`.

    The detection gate's input, and the only way to exercise detection without
    a camera. Same output contract as `VideoCapture.read`: caller-sized buffer,
    resized here if short.
    """
    var p = path
    var need = 1920 * 1080 * 3
    if len(out) < need:
        out.resize(need, 0)
    var w = Int32(0)
    var h = Int32(0)
    var c = Int32(0)
    var st = _get_dylib_function[
        lib,
        "nra_cv_imread",
        def(
            Ptr[c_char, MutUntrackedOrigin],
            Ptr[UInt8, MutUntrackedOrigin],
            Int32,
            Ptr[Int32, MutUntrackedOrigin],
            Ptr[Int32, MutUntrackedOrigin],
            Ptr[Int32, MutUntrackedOrigin],
        ) thin -> Int32,
    ]()(
        untracked(p.as_c_string_span().ptr()),
        untracked(Ptr(to=out[0])),
        Int32(len(out)),
        untracked(Ptr(to=w)),
        untracked(Ptr(to=h)),
        untracked(Ptr(to=c)),
    )
    _check(st, String("imread ") + path)
    return (Int(w), Int(h), Int(c))


struct ArucoDetector(Movable):
    """A marker detector over one predefined dictionary.

    ⚠ CLOSE IT EXPLICITLY, like `VideoCapture` — see that struct for why there
    is no destructor.
    """

    var _h: Int

    def __init__(out self, dict_id: Int = DICT_4X4_50) raises:
        self._h = _get_dylib_function[
            lib, "nra_cv_aruco_create", def(Int32) thin -> Int
        ]()(Int32(dict_id))
        if self._h == 0:
            raise String("opencv: cannot create detector: ") + cv_last_error()

    def detect(
        self,
        img: List[UInt8],
        width: Int,
        height: Int,
        channels: Int,
        mut ids: List[Int32],
        mut corners: List[Float32],
        max_markers: Int = 32,
    ) raises -> Int:
        """Detect markers. Returns how many were found.

        `corners` receives 8 floats per marker — four corners, xy, in OpenCV's
        CLOCKWISE order starting top-left.

        ⚠⚠ THAT ORDER IS THE CONTRACT, NOT A DETAIL. `solve_pnp` pairs image
        points with object points POSITIONALLY, so reordering one without the
        other yields a plausible pose that is silently rotated.

        ⚠ FINDING NOTHING IS NOT AN ERROR. An empty view returns 0, because
        "no marker" is the normal state of a camera and an exception would make
        every caller's loop the wrong shape.
        """
        if len(ids) < max_markers:
            ids.resize(max_markers, 0)
        if len(corners) < max_markers * 8:
            corners.resize(max_markers * 8, 0.0)
        var n = Int32(0)
        var st = _get_dylib_function[
            lib,
            "nra_cv_aruco_detect",
            def(
                Int,
                Ptr[UInt8, MutUntrackedOrigin],
                Int32,
                Int32,
                Int32,
                Int32,
                Int32,
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Float32, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            self._h,
            untracked(Ptr(to=img[0])),
            Int32(width),
            Int32(height),
            Int32(channels),
            Int32(0),  # stride 0 = tightly packed
            Int32(max_markers),
            untracked(Ptr(to=ids[0])),
            untracked(Ptr(to=corners[0])),
            untracked(Ptr(to=n)),
        )
        _check(st, "aruco_detect")
        return Int(n)

    def close(mut self):
        """Idempotent, like `VideoCapture.close`."""
        if self._h == 0:
            return
        try:
            _get_dylib_function[
                lib, "nra_cv_aruco_destroy", def(Int) thin -> None
            ]()(self._h)
        except:
            pass
        self._h = 0


# ═══════════════════════════════════════════════════════════════════════════
# D — pose
# ═══════════════════════════════════════════════════════════════════════════


def solve_pnp(
    obj_xyz: List[Float64],
    img_xy: List[Float64],
    k: List[Float64],
    dist: List[Float64],
    mut rvec: List[Float64],
    mut tvec: List[Float64],
    flags: Int = SOLVEPNP_IPPE_SQUARE,
) raises:
    """Object pose from 3D-2D correspondences. `k` is a row-major 3x3.

    ⚠ A SQUARE MARKER HAS A GENUINE TWO-FOLD POSE AMBIGUITY NEAR HEAD-ON.
    Position stays solid; ORIENTATION can flip between two solutions frame to
    frame. That is the geometry of four coplanar points, not a defect here —
    design against `tvec`, and treat `rvec` near head-on with suspicion.
    """
    var n = len(obj_xyz) // 3
    if n < 4:
        raise String("solve_pnp: need >= 4 points, got ") + String(n)
    if len(img_xy) < n * 2:
        raise String("solve_pnp: ") + String(
            n
        ) + " object points but " + String(len(img_xy) // 2) + " image points"
    if len(k) != 9:
        raise String("solve_pnp: K must be 9 values, got ") + String(len(k))
    if len(rvec) < 3:
        rvec.resize(3, 0.0)
    if len(tvec) < 3:
        tvec.resize(3, 0.0)

    # An empty `dist` means "already undistorted"; the shim substitutes zeros.
    var n_dist = len(dist)
    var dist_ptr = untracked(Ptr(to=k[0]))  # never read when n_dist == 0
    if n_dist > 0:
        dist_ptr = untracked(Ptr(to=dist[0]))

    var st = _get_dylib_function[
        lib,
        "nra_cv_solve_pnp",
        def(
            Ptr[Float64, MutUntrackedOrigin],
            Ptr[Float64, MutUntrackedOrigin],
            Int32,
            Ptr[Float64, MutUntrackedOrigin],
            Ptr[Float64, MutUntrackedOrigin],
            Int32,
            Int32,
            Ptr[Float64, MutUntrackedOrigin],
            Ptr[Float64, MutUntrackedOrigin],
        ) thin -> Int32,
    ]()(
        untracked(Ptr(to=obj_xyz[0])),
        untracked(Ptr(to=img_xy[0])),
        Int32(n),
        untracked(Ptr(to=k[0])),
        dist_ptr,
        Int32(n_dist),
        Int32(flags),
        untracked(Ptr(to=rvec[0])),
        untracked(Ptr(to=tvec[0])),
    )
    _check(st, "solve_pnp")


def rodrigues(rvec: List[Float64], mut r9: List[Float64]) raises:
    """Axis-angle to a row-major 3x3 rotation matrix.

    A pose is only useful once it composes with the robot's frames, and `rvec`
    is not a matrix.
    """
    if len(rvec) < 3:
        raise String("rodrigues: rvec must be 3 values")
    if len(r9) < 9:
        r9.resize(9, 0.0)
    _check(
        _get_dylib_function[
            lib,
            "nra_cv_rodrigues",
            def(
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(untracked(Ptr(to=rvec[0])), untracked(Ptr(to=r9[0]))),
        "rodrigues",
    )


# ═══════════════════════════════════════════════════════════════════════════
# E — calibration
# ═══════════════════════════════════════════════════════════════════════════


struct CharucoBoard(Movable):
    """A ChArUco board and its detector, as one object.

    ⚠ ONE HANDLE ON PURPOSE. `CharucoDetector` is built FROM a board and the
    caller must keep that board alive; two handles would let one be freed while
    the other reads it, with no diagnostic. Close it explicitly, like the
    others.
    """

    var _h: Int
    var squares_x: Int
    var squares_y: Int

    def __init__(
        out self,
        squares_x: Int,
        squares_y: Int,
        square_m: Float32,
        marker_m: Float32,
        dict_id: Int = DICT_4X4_50,
    ) raises:
        self._h = _get_dylib_function[
            lib,
            "nra_cv_charuco_create",
            def(Int32, Int32, Float32, Float32, Int32) thin -> Int,
        ]()(
            Int32(squares_x),
            Int32(squares_y),
            square_m,
            marker_m,
            Int32(dict_id),
        )
        if self._h == 0:
            raise String("opencv: cannot create board: ") + cv_last_error()
        self.squares_x = squares_x
        self.squares_y = squares_y

    def board_corners(self, mut xyz: List[Float32]) raises -> Int:
        """The board's chessboard corners in BOARD metres, indexed by id.

        ⚠⚠ ASK THE BOARD, DO NOT DERIVE IT. The ChArUco corner layout CHANGED
        in OpenCV 4.6 for even row counts (`setLegacyPattern` exists to restore
        the old one). Computing these from `squares_x`/`squares_y` in Mojo
        would be a second implementation of a convention that has already moved
        once, and it would be silently wrong on exactly one board shape.
        """
        var cap = (self.squares_x - 1) * (self.squares_y - 1)
        if len(xyz) < cap * 3:
            xyz.resize(cap * 3, 0.0)
        var n = Int32(0)
        _check(
            _get_dylib_function[
                lib,
                "nra_cv_charuco_board_corners",
                def(
                    Int,
                    Ptr[Float32, MutUntrackedOrigin],
                    Int32,
                    Ptr[Int32, MutUntrackedOrigin],
                ) thin -> Int32,
            ]()(
                self._h,
                untracked(Ptr(to=xyz[0])),
                Int32(len(xyz) // 3),
                untracked(Ptr(to=n)),
            ),
            "charuco_board_corners",
        )
        return Int(n)

    def detect(
        self,
        img: List[UInt8],
        width: Int,
        height: Int,
        channels: Int,
        mut corners: List[Float32],
        mut ids: List[Int32],
    ) raises -> Int:
        """Detect the board's corners. Returns how many were VISIBLE.

        ⚠ ONLY VISIBLE CORNERS COME BACK, and `ids` says which. Pairing them
        with `board_corners` POSITIONALLY instead of BY ID is a silent
        mismatch that calibrates to nonsense — the count is right, the
        correspondence is not, and nothing raises.
        """
        var cap = (self.squares_x - 1) * (self.squares_y - 1)
        if len(ids) < cap:
            ids.resize(cap, 0)
        if len(corners) < cap * 2:
            corners.resize(cap * 2, 0.0)
        var n = Int32(0)
        _check(
            _get_dylib_function[
                lib,
                "nra_cv_charuco_detect",
                def(
                    Int,
                    Ptr[UInt8, MutUntrackedOrigin],
                    Int32,
                    Int32,
                    Int32,
                    Int32,
                    Int32,
                    Ptr[Float32, MutUntrackedOrigin],
                    Ptr[Int32, MutUntrackedOrigin],
                    Ptr[Int32, MutUntrackedOrigin],
                ) thin -> Int32,
            ]()(
                self._h,
                untracked(Ptr(to=img[0])),
                Int32(width),
                Int32(height),
                Int32(channels),
                Int32(0),
                Int32(len(ids)),
                untracked(Ptr(to=corners[0])),
                untracked(Ptr(to=ids[0])),
                untracked(Ptr(to=n)),
            ),
            "charuco_detect",
        )
        return Int(n)

    def close(mut self):
        if self._h == 0:
            return
        try:
            _get_dylib_function[
                lib, "nra_cv_charuco_destroy", def(Int) thin -> None
            ]()(self._h)
        except:
            pass
        self._h = 0


def calibrate_camera(
    obj_xyz: List[Float64],
    img_xy: List[Float64],
    counts: List[Int32],
    img_w: Int,
    img_h: Int,
    mut k: List[Float64],
    mut dist: List[Float64],
    flags: Int = 0,
) raises -> Tuple[Int, Float64]:
    """Intrinsics from several views. Returns `(n_dist, rms)`.

    `obj_xyz` and `img_xy` are the views' correspondences CONCATENATED;
    `counts[v]` says how many belong to view v.

    ⚠⚠ `n_dist` IS RETURNED, AND ASSUMING 5 IS A SILENT TRUNCATION. The
    distortion vector is 4, 5, 8, 12 or 14 long depending on `flags`; a caller
    that copies a fixed 5 out of a 14-long answer keeps a lens model that is
    wrong in a way nothing reports. `dist` is sized to 14 here for that reason.

    ⚠ ITERATIVE (Levenberg-Marquardt). Comparing this against Python cv2 bit
    for bit also requires the same thread count — see `cv_set_num_threads`.
    """
    if len(k) < 9:
        k.resize(9, 0.0)
    if len(dist) < 14:
        dist.resize(14, 0.0)
    var n_dist = Int32(0)
    var rms = Float64(0.0)
    _check(
        _get_dylib_function[
            lib,
            "nra_cv_calibrate_camera",
            def(
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
                Int32,
                Int32,
                Int32,
                Int32,
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            untracked(Ptr(to=obj_xyz[0])),
            untracked(Ptr(to=img_xy[0])),
            untracked(Ptr(to=counts[0])),
            Int32(len(counts)),
            Int32(img_w),
            Int32(img_h),
            Int32(flags),
            untracked(Ptr(to=k[0])),
            untracked(Ptr(to=dist[0])),
            untracked(Ptr(to=n_dist)),
            untracked(Ptr(to=rms)),
        ),
        "calibrate_camera",
    )
    return (Int(n_dist), rms)


def fisheye_calibrate(
    obj_xyz: List[Float64],
    img_xy: List[Float64],
    counts: List[Int32],
    img_w: Int,
    img_h: Int,
    mut k: List[Float64],
    mut d: List[Float64],
    mut rvecs: List[Float64],
    mut tvecs: List[Float64],
    flags: Int = FISHEYE_CALIB_RECOMPUTE_EXTRINSIC | FISHEYE_CALIB_FIX_SKEW,
) raises -> Float64:
    """`cv::fisheye::calibrate` — the Kannala-Brandt model, 4 terms. Returns
    the rms reprojection error in pixels; `k` gets 9 (row-major), `d` 4,
    `rvecs` / `tvecs` 3 per view. Same concatenated layout as
    `calibrate_camera`.

    ⚠ THE RUNTIME MODEL IS `vision/fisheye.mojo`, NOT THIS. This is the fit;
    what the importer and the deploy path evaluate is the Mojo port, gated
    against `fisheye_project` / `fisheye_undistort_map` below."""
    var nv = len(counts)
    # ⚠ `k` / `d` are IN-OUT: under `FISHEYE_CALIB_USE_INTRINSIC_GUESS` their
    # contents are the starting point, so they are resized, never cleared.
    k.resize(9, 0.0)
    d.resize(4, 0.0)
    rvecs.resize(nv * 3, 0.0)
    tvecs.resize(nv * 3, 0.0)
    var rms = Float64(0.0)
    _check(
        _get_dylib_function[
            lib,
            "nra_cv_fisheye_calibrate",
            def(
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
                Int32,
                Int32,
                Int32,
                Int32,
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            untracked(Ptr(to=obj_xyz[0])),
            untracked(Ptr(to=img_xy[0])),
            untracked(Ptr(to=counts[0])),
            Int32(nv),
            Int32(img_w),
            Int32(img_h),
            Int32(flags),
            untracked(Ptr(to=k[0])),
            untracked(Ptr(to=d[0])),
            untracked(Ptr(to=rms)),
            untracked(Ptr(to=rvecs[0])),
            untracked(Ptr(to=tvecs[0])),
        ),
        "fisheye_calibrate",
    )
    return rms


def fisheye_project(
    obj_xyz: List[Float64],
    rvec: List[Float64],
    tvec: List[Float64],
    k: List[Float64],
    d: List[Float64],
    mut img_xy: List[Float64],
) raises:
    """`cv::fisheye::projectPoints` — the ORACLE the Mojo model is gated on."""
    var n = len(obj_xyz) // 3
    img_xy.resize(n * 2, 0.0)
    _check(
        _get_dylib_function[
            lib,
            "nra_cv_fisheye_project",
            def(
                Ptr[Float64, MutUntrackedOrigin],
                Int32,
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            untracked(Ptr(to=obj_xyz[0])),
            Int32(n),
            untracked(Ptr(to=rvec[0])),
            untracked(Ptr(to=tvec[0])),
            untracked(Ptr(to=k[0])),
            untracked(Ptr(to=d[0])),
            untracked(Ptr(to=img_xy[0])),
        ),
        "fisheye_project",
    )


def fisheye_undistort_map(
    k: List[Float64],
    d: List[Float64],
    k_new: List[Float64],
    out_w: Int,
    out_h: Int,
    mut map_x: List[Float32],
    mut map_y: List[Float32],
) raises:
    """`cv::fisheye::initUndistortRectifyMap` (R = I, P = `k_new`), float
    maps `[out_h * out_w]` — the ORACLE for `fisheye.UndistortMap`."""
    map_x.resize(out_w * out_h, 0.0)
    map_y.resize(out_w * out_h, 0.0)
    _check(
        _get_dylib_function[
            lib,
            "nra_cv_fisheye_undistort_map",
            def(
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Int32,
                Int32,
                Ptr[Float32, MutUntrackedOrigin],
                Ptr[Float32, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            untracked(Ptr(to=k[0])),
            untracked(Ptr(to=d[0])),
            untracked(Ptr(to=k_new[0])),
            Int32(out_w),
            Int32(out_h),
            untracked(Ptr(to=map_x[0])),
            untracked(Ptr(to=map_y[0])),
        ),
        "fisheye_undistort_map",
    )


# ═══════════════════════════════════════════════════════════════════════════
# F — the one piece of linear algebra we would otherwise have to write
# ═══════════════════════════════════════════════════════════════════════════


def svd_3x3(
    a9: List[Float64],
    mut u9: List[Float64],
    mut s3: List[Float64],
    mut vt9: List[Float64],
) raises:
    """3x3 SVD, all row-major. `vt` is ALREADY TRANSPOSED, as OpenCV names it.

    Exists so the camera->base extrinsics fit does not need a Jacobi
    implementation: `math3d` has no SVD or eigensolver, and `cv::SVD` is
    already linked.
    """
    if len(a9) != 9:
        raise String("svd_3x3: input must be 9 values, got ") + String(len(a9))
    if len(u9) < 9:
        u9.resize(9, 0.0)
    if len(s3) < 3:
        s3.resize(3, 0.0)
    if len(vt9) < 9:
        vt9.resize(9, 0.0)
    _check(
        _get_dylib_function[
            lib,
            "nra_cv_svd_3x3",
            def(
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            untracked(Ptr(to=a9[0])),
            untracked(Ptr(to=u9[0])),
            untracked(Ptr(to=s3[0])),
            untracked(Ptr(to=vt9[0])),
        ),
        "svd_3x3",
    )
