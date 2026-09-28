# +--------------------------------------------------------------------------+ #
# | A camera on its own thread, because a blocking read is mostly WAITING
# +--------------------------------------------------------------------------+ #
"""One OS thread per camera, frames handed over through an SPSC byte ring.

    var cam = CameraReader(device=0, width=640, height=480, fps=30)
    cam.start()
    ...
    var frame = List[UInt8](unsafe_uninit_length = cam.frame_bytes())
    if cam.take(frame):        # one frame, BGR24 (or RGB24 with rgb=True)
        ...
    cam.stop()

## Why a thread, measured rather than assumed

`tools/soarm/record_budget.mojo` against the real cameras, 30 Hz, 640x480:

    one blocking read       mean 17.11 ms   worst 137.71 ms
    bgr->rgb + encode       mean  1.99 ms
    leader sync_read        mean  1.25 ms
    tick, 2 cameras SERIAL  mean 34.66 ms   worst 99.63 ms   -> budget is 33.3

⚠ **THE 17 ms IS WAIT, NOT WORK.** At 30 fps a frame arrives every 33.3 ms and
a blocking read sleeps until the next one; two cameras read one after the
other pay two INDEPENDENT waits, because nothing synchronises their shutters.
So the cost of a second camera is not "a bit more CPU", it is another whole
frame interval — the mean alone blows the budget, before any jitter.

That is also why the fix is threads and not optimisation: there is nothing to
speed up. Reading both cameras at once overlaps the two waits, and the loop
then pays roughly one wait instead of the sum. The encoder and the servo bus
are ~2 ms and ~1 ms and were never the problem.

## What crosses the thread boundary

Bytes only, one frame per slot — the rule `core/concurrent` states outright.
No Mojo value is shared: the `VideoCapture` is opened in `on_start` and lives
entirely on the worker thread, the way `io/http_sink.mojo` builds its libcurl
handle there.

⚠ **THE POLICY IS DROP-AND-COUNT, AND THE COUNT IS PART OF THE RECORDING.**
If the consumer falls behind, `try_push` refuses the new frame rather than
stalling the camera. A recording that silently lost frames looks exactly like
one that did not, so `dropped()` and `starved()` are printed by the recorder
the way `teleop.mojo` prints its dropped ticks.

⚠ **A RING OF FRAMES IS FIFO, NOT "LATEST WINS".** That is deliberate: a
recorder wants every frame at 30 fps, not the freshest one. A viewer would
want the opposite, and should drain to the last claim rather than change this.

⚠ `SharedRing` / `SharedBlock`, never the bare `SpscRing` / `ControlBlock` —
the bare owners are freed at their last mention, which is the `view()` that
built the worker, and the thread then reads freed memory
(`_taking_a_view_is_the_owners_last_use`).
"""

from std.memory import Pointer, unsafe_memcpy
from std.os.path import exists
from std.sys import CompilationTarget

from ..core.concurrent.thread import sleep_us
from ..core.concurrent.block import SharedBlock
from ..core.concurrent.ring import SharedRing
from ..core.concurrent.worker import (
    POLL_DID_WORK, POLL_IDLE, BackgroundThread, BackgroundWorker, WorkerCtl,
)
from .opencv import VideoCapture, opencv_shim_available
from .preprocess import camera_frame_to_chw_rgb
from .calib_file import read_calib
from .fisheye import FisheyeLens, Pinhole, UndistortMap
from .resize_pad import camera_frame_to_siglip


@always_inline
def _erase(mut lst: List[UInt8]) -> Pointer[UInt8, MutUntrackedOrigin]:
    """A List's base pointer with its origin erased to the ring's.

    ⚠ `MutUntrackedOrigin` and a `List`'s tracked origin are SIBLINGS, not
    convertible — `origin_cast` does not exist on `Pointer` and
    `as_unsafe_any_origin()` produces a third, equally incompatible one. The
    bridge is `rebind`, exactly as `io/parquet/thrift.mojo:byte_ptr` does it.
    """
    return rebind[Pointer[UInt8, MutUntrackedOrigin]](
        lst.unsafe_ptr().as_unsafe_any_origin()
    )


@always_inline
def _erase_f32(
    mut lst: List[Float32],
) -> Pointer[UInt8, MutUntrackedOrigin]:
    """`_erase` for the SigLIP float block: its BYTES, origin erased, so one
    memcpy carries the floats into a ring slot bit-for-bit."""
    return rebind[Pointer[UInt8, MutUntrackedOrigin]](
        lst.unsafe_ptr().unsafe_bitcast[UInt8]().as_unsafe_any_origin()
    )


def _pack_fourcc(code: String) -> Int64:
    """Four characters into an Int64, little-endian; 0 when absent."""
    if code.byte_length() != 4:
        return Int64(0)
    var b = code.as_bytes()
    var v = Int64(0)
    for i in range(4):
        v |= Int64(Int(b[i])) << Int64(8 * i)
    return v


def _unpack_fourcc(v: Int64) -> String:
    if v == 0:
        return String("")
    var out = String("")
    for i in range(4):
        var c = Int((v >> Int64(8 * i)) & Int64(0xFF))
        if c < 32 or c > 126:
            return String("")
        out += chr(c)
    return out^


def _open_hint(path: String) -> String:
    """Why a named camera might not have opened — the udev case, said once."""
    if path.byte_length() == 0:
        return String("")
    if exists(path):
        return String(
            " (the path exists, so it is in use by another process, or it is a"
            " metadata node rather than a capture one — only the node with"
            " ID_V4L_CAPABILITIES=:capture: streams)"
        )
    return String(
        " — " + path + " does not exist. On the board these are udev symlinks:"
        " check `ls -l /dev/soarm_cam_*` and"
        " /etc/udev/rules.d/99-soarm.rules (docs/JETSON_DEPLOYMENT.md §3)."
    )


def default_fourcc() -> String:
    """The pixel format a path-opened camera is asked for when none is given.

    ⚠⚠ MJPEG ON LINUX, AND IT IS A BANDWIDTH DECISION, NOT A QUALITY ONE.
    YUYV 640x480 at 30 fps is 147 Mbit/s PER CAMERA — two of them is 295 of a
    USB 2.0 bus's ~320 practical Mbit/s — and camera traffic is ISOCHRONOUS,
    which RESERVES bus time, while the Feetech bus is bulk and takes what is
    left. On this board the wrist camera shares `tegra-xusb` with both servo
    links (`docs/JETSON_DEPLOYMENT.md` §3.1), so a YUYV stream is a servo
    jitter source, and on a 30 Hz control loop jitter costs more than latency.
    MJPEG is ~15x less. §3.2 had already decided this; nothing was asking for
    it, so every camera came up YUYV.

    ⚠ Pass `--fourcc none` to leave the device's own default alone, or any
    four characters to ask for something else. What was NEGOTIATED is printed
    either way — a request is not a setting.

    macOS opens by index through AVFoundation, where this does not apply.
    """
    comptime if CompilationTarget.is_macos():
        return String("")
    return String("MJPG")


def _node_index(node: String) -> Int:
    """The N in `/dev/videoN`, or -1 for anything else."""
    comptime PREFIX = "/dev/video"
    if not node.startswith(PREFIX):
        return -1
    var digits = String(node[byte=PREFIX.byte_length() : node.byte_length()])
    if digits.byte_length() == 0:
        return -1
    var n = 0
    var b = digits.as_bytes()
    for i in range(digits.byte_length()):
        var c = Int(b[i])
        if c < 48 or c > 57:
            return -1
        n = n * 10 + (c - 48)
    return n


def camera_spec_is_path(spec: String) -> Bool:
    """Whether `spec` names a device PATH rather than an index.

    ⚠ THE TEST IS "NOT ALL DIGITS", not "starts with /". A relative path is
    still a path, and an index is never anything but digits — so this errs
    toward treating an odd argument as a path, where the failure names the
    string the operator typed instead of silently opening camera 0.
    """
    if spec.byte_length() == 0:
        return False
    var b = spec.as_bytes()
    for i in range(spec.byte_length()):
        var c = Int(b[i])
        if c < 48 or c > 57:
            return True
    return False


def camera_spec_index(spec: String) raises -> Int:
    """The index in a `--devices` entry, or -1 when it names a PATH.

    ⚠ -1 IS "THERE IS NO INDEX", not a default or an error. `CameraCalib.device`
    records the index a calibration was measured on as a HINT (its own field
    note says never as identity), and a camera opened as
    `/dev/soarm_cam_wrist` genuinely has no stable one — writing 0 there would
    be inventing a fact about the rig.
    """
    if camera_spec_is_path(spec):
        return -1
    return Int(spec)


def open_camera_spec(
    spec: String,
    width: Int = 0,
    height: Int = 0,
    fps: Float64 = 0.0,
    fourcc: String = String(""),
) raises -> VideoCapture:
    """Open one `--devices` entry — index or path — WITHOUT a reader thread.

    ⚠ THE SAME RESOLUTION AS `CameraReader.from_spec`, written once. The tools
    that drive a `VideoCapture` directly (the tick-budget bench, the extrinsics
    fit, the camera studio) must agree with the recorders about what `0` and
    what `/dev/soarm_cam_wrist` mean, or the rig is calibrated through one
    camera and recorded through another.
    """
    if camera_spec_is_path(spec):
        return VideoCapture.device_path(spec, width, height, fps, fourcc)
    return VideoCapture.device(Int(spec), width, height, fps)


def parse_camera_specs(csv: String) raises -> List[String]:
    """Split `--devices` into per-slot specs: indices, paths, or a mix.

    ⚠ COMMAS, NOT SPACES. A space-separated list is consumed by the shell as
    separate arguments and the recorder then sees one camera; the error for
    that used to say only "count mismatch".
    """
    var out = List[String]()
    var parts = csv.split(",")
    for i in range(len(parts)):
        var s = String(String(parts[i]).strip())
        if s.byte_length() > 0:
            out.append(s^)
    if len(out) == 0:
        raise Error(
            "camera_thread: --devices '" + csv + "' names no camera. Give"
            " COMMA-separated indices or paths, e.g. `0,1` or"
            " `/dev/soarm_cam_overhead,/dev/soarm_cam_wrist`."
        )
    return out^


comptime CELL_STATE = 0
"""0 = starting, 1 = open and reading, -1 = the camera never opened."""
comptime CELL_READ_FAIL = 8
comptime CELL_GEOMETRY = 16
"""`height * 100000 + width`, published once the device reports it."""
comptime CELL_FOURCC = 24
"""The negotiated pixel format's four characters, packed little-endian into an
Int64, or 0 when the device did not report one.

⚠ PUBLISHED THROUGH A CELL BECAUSE THE WORKER CANNOT RAISE OR PRINT. It is the
owner that reports it, and it is worth reporting: recording in MJPEG and
deploying in YUYV shows a policy different pixels than it trained on
(`docs/JETSON_DEPLOYMENT.md` §3.2)."""
comptime CELL_NODE = 32
"""The `/dev/videoN` index a path-opened camera resolved to, plus one (so 0
still means "nothing published"). ⚠ AN INDEX, NOT A NAME: the resolution
happens on the camera thread and no Mojo-owned String may cross that
boundary. A udev name is stable and the node behind it is NOT, so which one
answered is worth saying out loud."""
comptime CELL_FPS = 40
"""The frame rate the device NEGOTIATED, x1000, published once at open.

⚠ A REQUEST IS NOT A SETTING, and this is the one property that was never read
back. The size is checked against the ring and the format is printed, but the
rate was asked for and forgotten — so a camera quietly running at 15 fps looked
exactly like one running at 30."""
comptime CELL_FRAMES = 48
"""Frames the camera thread has actually delivered.

⚠ ITS OWN CACHE LINE, like every other cell here (0, 8, 16, 24, 32, 40, 48 —
`CELLS_PER_LINE` apart, `block.mojo` states the rule). This one earns it: it
is `fetch_add`-ed by the camera thread on EVERY frame while the owner reads
CELL_STATE and CELL_GEOMETRY every tick, so packing them together would
invalidate the owner's line 30+ times a second per camera. It was written at
25/26/27 first — four cells in one line, the exact false sharing the padding
convention exists to prevent. ⚠ THE NEGOTIATED RATE IS
WHAT THE DRIVER CLAIMS; this is what arrived. They disagree when the bus is
saturated or the exposure is long, and only the second one is the truth."""
comptime N_CELLS = 56

comptime DEFAULT_SLOTS = 8
"""Frames of slack. At 30 fps that is 0.27 s — long enough to ride out a slow
tick, short enough that a consumer which has genuinely stopped starts dropping
rather than growing a queue of stale frames it will never catch up on."""


struct _CamWorker(BackgroundWorker):
    """Runs on the camera thread. Owns the `VideoCapture`."""

    var ring: SharedRing
    var block: SharedBlock
    var device: Int
    var path: String
    """Non-empty means OPEN BY PATH; `device` is then unused."""
    var fourcc: String
    var width: Int
    var height: Int
    var fps: Float64
    var cap: VideoCapture
    var buf: List[UInt8]
    var out: List[UInt8]
    """The resized CHW frame, when `out_w > 0`. Owned by this thread."""
    var out_w: Int
    var out_h: Int
    var siglip: Int
    """When > 0, the SmolVLA preprocess runs HERE: each frame becomes the
    `3*siglip*siglip` float block in `[-1, 1]` (`camera_frame_to_siglip`,
    resize_with_pad + BGR->RGB), and the ring carries those floats' bytes."""
    var f32: List[Float32]
    var opened: Bool
    var rgb: Bool
    var und_map: List[UndistortMap]
    """Empty, or the ONE fisheye -> sim-pinhole map applied to each native
    frame before the ACT resize (`CameraReader.set_undistort`)."""
    var und: List[UInt8]

    def __init__(
        out self,
        var ring: SharedRing,
        var block: SharedBlock,
        device: Int,
        var path: String,
        var fourcc: String,
        width: Int,
        height: Int,
        fps: Float64,
        rgb: Bool,
        out_w: Int,
        out_h: Int,
        siglip: Int,
        var und_map: List[UndistortMap],
    ) raises:
        self.ring = ring^
        self.block = block^
        self.device = device
        self.path = path^
        self.fourcc = fourcc^
        self.width = width
        self.height = height
        self.fps = fps
        # ⚠ A CLOSED PLACEHOLDER, not an Optional. `on_start` cannot raise, so
        # the field must already hold something valid; `VideoCapture.closed()`
        # exists for exactly this.
        self.cap = VideoCapture.closed()
        self.buf = List[UInt8]()
        self.out = List[UInt8]()
        self.out_w = out_w
        self.out_h = out_h
        self.siglip = siglip
        self.f32 = List[Float32]()
        self.opened = False
        self.rgb = rgb
        self.und_map = und_map^
        self.und = List[UInt8]()

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring^
        self.block = move.block^
        self.device = move.device
        self.path = move.path^
        self.fourcc = move.fourcc^
        self.width = move.width
        self.height = move.height
        self.fps = move.fps
        self.cap = move.cap^
        self.buf = move.buf^
        self.out = move.out^
        self.out_w = move.out_w
        self.out_h = move.out_h
        self.siglip = move.siglip
        self.f32 = move.f32^
        self.opened = move.opened
        self.rgb = move.rgb
        self.und_map = move.und_map^
        self.und = move.und^

    def on_start(mut self, ctl: WorkerCtl):
        # ⚠ THE DEVICE IS OPENED HERE, ON THE THREAD THAT WILL READ IT — the
        # same rule `io/http_sink.mojo` follows for its libcurl handle.
        try:
            if self.path.byte_length() > 0:
                self.cap = VideoCapture.device_path(
                    self.path, self.width, self.height, self.fps, self.fourcc
                )
            else:
                self.cap = VideoCapture.device(
                    self.device, self.width, self.height, self.fps
                )
            self.block.release_store(
                CELL_FOURCC, _pack_fourcc(self.cap.fourcc())
            )
            self.block.release_store(
                CELL_NODE, Int64(_node_index(self.cap.node) + 1)
            )
            self.block.release_store(
                CELL_FPS, Int64(self.cap.fps * 1000.0)
            )
            self.width = self.cap.width
            self.height = self.cap.height
            self.buf = List[UInt8](
                unsafe_uninit_length = self.cap.frame_bytes()
            )
            if self.out_w > 0:
                self.out = List[UInt8](
                    unsafe_uninit_length = self.out_w * self.out_h * 3
                )
            if self.siglip > 0:
                self.f32 = List[Float32](
                    length = 3 * self.siglip * self.siglip, fill=Float32(0)
                )
            self.opened = True
            self.block.release_store(
                CELL_GEOMETRY, Int64(self.height * 100000 + self.width)
            )
            self.block.release_store(CELL_STATE, Int64(1))
        except:
            # Nothing can be raised out of a worker thread, so the failure is
            # published as a cell and the owner turns it back into an error.
            self.block.release_store(CELL_STATE, Int64(-1))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        if not self.opened:
            return POLL_IDLE
        # ⚠ A PRODUCER MUST GO IDLE ON STOP, OR `stop()` NEVER RETURNS. The
        # drive loop keeps polling after `RUNNING` clears and exits at the
        # FIRST `POLL_IDLE` — which a live camera never produces, because
        # there is always another frame. Without this the join blocks
        # forever; measured as a hang, not a slow build (the whole binary
        # compiles in 2.3 s).
        #
        # There is nothing to drain on this side: frames already in the ring
        # belong to the consumer, and the camera has no backlog of its own.
        if ctl.should_stop():
            return POLL_IDLE
        var got: Bool
        try:
            got = self.cap.read(self.buf)
        except:
            _ = self.block.fetch_add(CELL_READ_FAIL, Int64(1))
            return POLL_IDLE
        if not got:
            # End of stream: a live camera does not do this, but `--file`
            # playback does, and an idle poll lets `stop()` finish promptly.
            return POLL_IDLE
        var n = self.cap.frame_bytes()
        # ⚠ THE CHANNEL SWAP BELONGS ON THIS THREAD. OpenCV gives BGR and the
        # encoder wants RGB24; doing it in the record loop cost a measured
        # **9.8 ms worst** of a 33.3 ms tick, purely shuffling bytes. This
        # thread spends most of its life blocked in `read()`, so the same work
        # is free here.
        if self.rgb:
            for p in range(0, n, 3):
                var t = self.buf[p]
                self.buf[p] = self.buf[p + 2]
                self.buf[p + 2] = t
        # ⚠⚠ AND SO DOES THE RESIZE, FOR THE SAME REASON ONE LEVEL UP. The ACT
        # control loop paid a measured 9.0 ms per query turning two 640x480
        # BGR frames into 320x240 CHW RGB, against a 33.3 ms frame period it
        # was missing by 1.4 ms — so the whole loop ran at 20 Hz instead of 30.
        # Here the work is free: this thread is blocked in `read()` for most of
        # a frame period whatever it does.
        #
        # ⚠ THE PIXELS DO NOT CHANGE. Same function, same inputs, a different
        # thread — `camera_frame_to_chw_rgb` is pure, and the bit-exactness
        # gate against PIL (`tests/vision/test_resize_deploy_vs_import.mojo`)
        # covers it wherever it runs.
        if self.out_w > 0:
            try:
                if len(self.und_map) > 0:
                    # ⚠ THE SAME `UndistortMap` THE IMPORTER'S `--undistort`
                    # APPLIED, before the same resize — so the deployed frame
                    # and the stored one are in one camera model. Channel
                    # order is irrelevant to it (per channel, like the resize).
                    ref um = self.und_map[0]
                    if um.src_w != self.width or um.src_h != self.height:
                        raise Error("undistort: calibration size != frame size")
                    if len(self.und) != len(self.buf):
                        self.und = List[UInt8](length=len(self.buf), fill=0)
                    um.apply_hwc(self.buf, 0, 3, self.und, 0)
                    camera_frame_to_chw_rgb(
                        self.und, self.width, self.height,
                        self.out, self.out_w, self.out_h,
                    )
                else:
                    camera_frame_to_chw_rgb(
                        self.buf, self.width, self.height,
                        self.out, self.out_w, self.out_h,
                    )
            except:
                _ = self.block.fetch_add(CELL_READ_FAIL, Int64(1))
                return POLL_IDLE
            n = self.out_w * self.out_h * 3
        # ⚠⚠ AND THE SMOLVLA PREPROCESS, FOR THE SAME REASON AGAIN — with a
        # larger prize. Under `--sync` the SmolVLA deploy takes its
        # observation with nothing in flight, so the arm HOLDS for the whole
        # build: measured 38-67 ms on the Orin for two 640x480 frames through
        # `resize_with_pad` to 512x512, on top of the query. Here the same
        # function runs on the frame the moment it lands, and the control
        # loop's observation build becomes one memcpy of the newest block.
        #
        # ⚠ THE PIXELS DO NOT CHANGE: `camera_frame_to_siglip` is the function
        # `fill_camera_images` called on the control thread, with the same
        # arguments (`swap_rb` = the frame is still BGR, i.e. `not rgb`).
        # `tests/deep_agents/smolvla/test_store_prefix.mojo` is its gate.
        if self.siglip > 0:
            try:
                camera_frame_to_siglip(
                    self.buf, self.width, self.height, not self.rgb,
                    self.f32, 0, self.siglip,
                )
            except:
                _ = self.block.fetch_add(CELL_READ_FAIL, Int64(1))
                return POLL_IDLE
            n = 3 * self.siglip * self.siglip * 4
        # Zero-copy claim, then one memcpy into the slot — the same shape
        # `io/http_sink.mojo:frame_into` uses, and it sidesteps handing a
        # `List`-derived pointer to a `MutUntrackedOrigin` parameter.
        _ = self.block.fetch_add(CELL_FRAMES, Int64(1))
        var v = self.ring.view()
        var slot = v.begin_push()
        if not slot.ok():
            # Drop-and-count: the consumer is behind. Stalling the camera
            # instead would push the backlog onto the device's own buffer,
            # where it is invisible and unbounded.
            v.drop_full()
            return POLL_DID_WORK
        if self.siglip > 0:
            unsafe_memcpy(dest=slot.data(), src=_erase_f32(self.f32), count=n)
        else:
            unsafe_memcpy(
                dest=slot.data(),
                src=_erase(self.out) if self.out_w > 0 else _erase(self.buf),
                count=n,
            )
        v.end_push(n)
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        if self.opened:
            self.cap.close()
            self.opened = False


struct CameraReader(Movable):
    """A camera, its thread, and the ring between them."""

    var ring: SharedRing
    var block: SharedBlock
    var device: Int
    var path: String
    """Non-empty when the camera was named by PATH; `device` is then unused."""
    var fourcc: String
    """The pixel format REQUESTED for a path-opened camera ("" = the device's
    own default). What was negotiated is `negotiated_fourcc()`, after `start`."""
    var width: Int
    var height: Int
    """The camera's NATIVE size. ⚠ Not the size of a frame `take` returns when
    `out_w > 0` — use `frame_bytes()`, which is what the ring actually holds."""
    var out_w: Int
    var out_h: Int
    """Resize done ON THE CAMERA THREAD, 0 for none. When set, frames come back
    as CHW RGB uint8 at this size instead of native BGR HWC."""
    var siglip: Int
    """> 0: the thread delivers SmolVLA's `3*siglip*siglip` float block (its
    bytes) instead of pixels — see `_CamWorker.siglip`. Exclusive with
    `out_w`, and `frame_bytes()` reports the block."""
    var fps: Float64
    var _thread: Optional[BackgroundThread[_CamWorker]]
    var _starved: Int
    var running: Bool
    var rgb: Bool
    """True when frames are delivered RGB24 instead of OpenCV's BGR."""
    var undistort_calib: String
    """A fisheye calibration applied before the ACT resize, or "" — see
    `set_undistort`."""
    var undistort_fovy: Float64

    def __init__(
        out self,
        device: Int,
        width: Int = 640,
        height: Int = 480,
        fps: Float64 = 30.0,
        slots: Int = DEFAULT_SLOTS,
        rgb: Bool = False,
        out_w: Int = 0,
        out_h: Int = 0,
        siglip: Int = 0,
    ) raises:
        self = Self(
            String(""), device, width, height, fps, slots, rgb, String(""),
            out_w, out_h, siglip,
        )

    @staticmethod
    def at_path(
        path: String,
        width: Int = 640,
        height: Int = 480,
        fps: Float64 = 30.0,
        slots: Int = DEFAULT_SLOTS,
        rgb: Bool = False,
        fourcc: String = String(""),
        out_w: Int = 0,
        out_h: Int = 0,
        siglip: Int = 0,
    ) raises -> Self:
        """A camera named by device path — `/dev/soarm_cam_overhead`.

        ⚠ A PATH IS THE ONLY STABLE NAME ON THE BOARD. Both SO-101 cameras
        report the same burned-in serial, so only the USB topology separates
        them and `video0`/`video2` are assigned in enumeration order; the udev
        rules in `docs/JETSON_DEPLOYMENT.md` §3 turn that into a fixed name.
        Opening by index throws the distinction away, and the failure is not a
        crash — it is the policy confidently acting on the wrong camera.

        `out_w`/`out_h` move the ACT preprocess ONTO THE CAMERA THREAD: frames
        then arrive as CHW RGB uint8 at that size instead of native BGR HWC.
        ⚠ It is a latency decision, not a convenience — see the note in
        `_CamWorker.poll`. Leave both 0 for a recorder, which needs the full
        frame for its encoder. `siglip` is the SmolVLA counterpart: the
        thread delivers the 512x512 `[-1, 1]` float block instead.
        """
        return Self(path, 0, width, height, fps, slots, rgb, fourcc,
                    out_w, out_h, siglip)

    @staticmethod
    def from_spec(
        spec: String,
        width: Int = 640,
        height: Int = 480,
        fps: Float64 = 30.0,
        slots: Int = DEFAULT_SLOTS,
        rgb: Bool = False,
        fourcc: String = String(""),
        out_w: Int = 0,
        out_h: Int = 0,
        siglip: Int = 0,
    ) raises -> Self:
        """An index or a path, as `--devices` gives it."""
        if camera_spec_is_path(spec):
            return Self.at_path(
                spec, width, height, fps, slots, rgb, fourcc, out_w, out_h,
                siglip,
            )
        return Self(
            Int(spec), width, height, fps, slots, rgb, out_w, out_h, siglip
        )

    def __init__(
        out self,
        var path: String,
        device: Int,
        width: Int,
        height: Int,
        fps: Float64,
        slots: Int,
        rgb: Bool,
        var fourcc: String,
        out_w: Int,
        out_h: Int,
        siglip: Int,
    ) raises:
        if siglip > 0 and out_w > 0:
            raise Error(
                "camera_thread: `out_w` (ACT's CHW resize) and `siglip`"
                " (SmolVLA's float block) are two deliveries; ask for one"
            )
        if not opencv_shim_available():
            raise Error(
                "camera_thread: the OpenCV shim is not built — `pixi run"
                " build-opencv`"
            )
        if width <= 0 or height <= 0:
            raise Error("camera_thread: a camera needs a positive size")
        # ⚠ "" MEANS "NO PREFERENCE", WHICH IS NOT THE SAME AS "NO REQUEST":
        # a path-opened camera gets `default_fourcc()`, and only the explicit
        # word `none` leaves the device to its own default. Without this every
        # V4L2 camera came up YUYV — see `default_fourcc` for why that matters
        # on a shared USB bus.
        if fourcc == "none":
            fourcc = String("")
        elif fourcc.byte_length() == 0 and path.byte_length() > 0:
            fourcc = default_fourcc()
        if fourcc.byte_length() != 0 and fourcc.byte_length() != 4:
            raise Error(
                "camera_thread: a fourcc is four characters, or `none` (got '"
                + fourcc + "')"
            )
        # ⚠ THE RING IS SIZED FOR WHAT IT WILL CARRY, which is the RESIZED
        # frame when the worker resizes. At 320x240 CHW that is 230 KB against
        # 921 KB native — the eight slots go from 7.4 MB to 1.8 MB per camera.
        var slot_bytes = width * height * 3
        if siglip > 0:
            slot_bytes = 3 * siglip * siglip * 4
        elif out_w > 0:
            slot_bytes = out_w * out_h * 3
        self.ring = SharedRing(slots, slot_bytes)
        self.block = SharedBlock(N_CELLS)
        self.device = device
        self.path = path^
        self.fourcc = fourcc^
        self.width = width
        self.height = height
        self.out_w = out_w
        self.out_h = out_h
        self.siglip = siglip
        self.fps = fps
        self._thread = None
        self._starved = 0
        self.running = False
        self.rgb = rgb
        self.undistort_calib = String("")
        self.undistort_fovy = 0.0

    def __init__(out self, *, deinit move: Self):
        self.ring = move.ring^
        self.block = move.block^
        self.device = move.device
        self.path = move.path^
        self.fourcc = move.fourcc^
        self.width = move.width
        self.height = move.height
        self.fps = move.fps
        self.out_w = move.out_w
        self.out_h = move.out_h
        self.siglip = move.siglip
        self._thread = move._thread^
        self._starved = move._starved
        self.running = move.running
        self.rgb = move.rgb
        self.undistort_calib = move.undistort_calib^
        self.undistort_fovy = move.undistort_fovy

    def label(self) -> String:
        """How this camera is named in a message: the path, or `device N`."""
        if self.path.byte_length() > 0:
            return self.path
        return String("device ") + String(self.device)

    def resolved_node(self) -> String:
        """The `/dev/videoN` a path-opened camera landed on, or "" — valid
        after `start`."""
        var v = Int(self.block.acquire_load(CELL_NODE))
        if v <= 0:
            return String("")
        return String("/dev/video") + String(v - 1)

    def negotiated_fps(self) -> Float64:
        """The rate the device reported, or 0.0 — valid after `start`."""
        return Float64(self.block.acquire_load(CELL_FPS)) / 1000.0

    def frames_delivered(self) -> Int:
        """How many frames the camera thread has produced since `start`.

        ⚠ THE MEASUREMENT THAT SETTLES A STARVED CONSUMER. A control loop that
        finds an empty ring is either faster than the camera (fine — its
        frames are fresh) or being starved by a camera that is not keeping its
        claimed rate (not fine). Dividing this by the elapsed time says which.
        """
        return Int(self.block.acquire_load(CELL_FRAMES))

    def negotiated_fourcc(self) -> String:
        """The pixel format the device settled on, or "" — valid after `start`.
        """
        return _unpack_fourcc(self.block.acquire_load(CELL_FOURCC))

    def set_undistort(mut self, calib_path: String, fovy: Float64) raises:
        """Bring each native frame to the SIM's pinhole (`vision/fisheye.mojo`,
        `fovy` degrees at the native size) before the ACT resize — what a
        store imported with `--undistort` holds. Call before `start`.

        ⚠ ACT DELIVERY ONLY (`out_w > 0`). A recorder's raw frames must stay
        raw: they are the dataset, and the dataset is undistorted at IMPORT,
        where the calibration can be redone without re-recording."""
        if self.running:
            raise Error("camera_thread: set_undistort after start")
        if self.out_w <= 0:
            raise Error(
                "camera_thread: undistortion applies to the ACT delivery"
                " (out_w > 0); a raw recorder frame is undistorted at import"
            )
        self.undistort_calib = calib_path
        self.undistort_fovy = fovy

    def frame_bytes(self) -> Int:
        """Bytes one frame occupies AS DELIVERED — resized when the worker
        resizes, native otherwise. Every `take` sizes its buffer from this."""
        if self.siglip > 0:
            return 3 * self.siglip * self.siglip * 4
        if self.out_w > 0:
            return self.out_w * self.out_h * 3
        return self.width * self.height * 3

    def _undistort_map(self) raises -> List[UndistortMap]:
        var out = List[UndistortMap]()
        if self.undistort_calib.byte_length() == 0:
            return out^
        var cal = read_calib(self.undistort_calib)
        cal.require_size(self.width, self.height)
        var um = UndistortMap(
            FisheyeLens.from_calib(cal),
            Pinhole.sim(self.undistort_fovy, self.width, self.height),
        )
        if um.n_outside > 0:
            raise Error(
                "camera_thread: " + self.undistort_calib + " leaves "
                + String(um.n_outside) + " pixels of the pinhole outside the lens"
            )
        out.append(um^)
        return out^

    def start(mut self, wait_ms: Int = 4000) raises:
        """Spawn the thread and WAIT for the device to actually open.

        ⚠ WAITING IS THE POINT. `on_start` runs on the other thread, so
        without this a caller's first `take()` returns False for reasons it
        cannot distinguish: a camera still warming up, and a camera that does
        not exist. Blocking here turns the second one into an error naming the
        device.
        """
        if self.running:
            raise Error("camera_thread: already started")
        self._thread = BackgroundThread(
            _CamWorker(
                self.ring, self.block, self.device, self.path, self.fourcc,
                self.width, self.height, self.fps, self.rgb,
                self.out_w, self.out_h, self.siglip, self._undistort_map(),
            )
        )
        self.running = True

        var waited = 0
        while waited < wait_ms:
            var st = self.block.acquire_load(CELL_STATE)
            if st == 1:
                var g = Int(self.block.acquire_load(CELL_GEOMETRY))
                var h = g // 100000
                var w = g % 100000
                # ⚠ ADOPT WHAT THE DEVICE REPORTED. `VideoCapture.device`'s
                # header is explicit that the size is a REQUEST; a ring slot
                # sized from what we asked for and frames of another size is a
                # silent corruption, so refuse instead.
                if w != self.width or h != self.height:
                    self.stop()
                    raise Error(
                        "camera_thread: " + self.label()
                        + " negotiated " + String(w) + "x" + String(h)
                        + ", not the " + String(self.width) + "x"
                        + String(self.height) + " that sized the ring."
                        " Construct the reader with the size the camera"
                        " actually supports."
                    )
                return
            if st == -1:
                self.stop()
                raise Error(
                    "camera_thread: " + self.label() + " did not open"
                    + _open_hint(self.path)
                )
            _sleep_ms(10)
            waited += 10
        self.stop()
        raise Error(
            "camera_thread: " + self.label() + " did not report ready within "
            + String(wait_ms) + " ms"
        )

    def take(mut self, mut out: List[UInt8]) raises -> Bool:
        """Copy the oldest queued frame into `out`. False when none is ready.

        BGR24, row-major — the format `VideoCapture` produces. The recorder
        swaps to RGB24 on its way into the encoder.
        """
        if len(out) < self.frame_bytes():
            raise Error(
                "camera_thread: a " + String(len(out)) + "-byte buffer for a "
                + String(self.frame_bytes()) + "-byte frame"
            )
        var v = self.ring.view()
        var c = v.begin_pop()
        if not c.ok():
            self._starved += 1
            return False
        # `rebind` to the ring's origin — the same bridge `thrift.byte_ptr`
        # uses. `MutUntrackedOrigin` and a List's origin are siblings, not
        # convertible (`_declare_the_origin_the_producer_makes`).
        unsafe_memcpy(
            dest=_erase(out),
            src=c.data(),
            count=c.len,
        )
        v.end_pop()
        return True

    def take_latest(mut self, mut out: List[UInt8]) raises -> Int:
        """Copy the NEWEST queued frame into `out`, discarding older ones.

        Returns how many frames were consumed; 0 means nothing was ready and
        `out` is untouched.

        ⚠⚠ **A CONTROLLER WANTS THE NEWEST FRAME; A RECORDER WANTS THE OLDEST.**
        `take` and `take_blocking` hand back the oldest queued frame, which is
        exactly right for `record.mojo` — every frame is data and dropping one
        puts a hole in the video. A policy is the opposite case: a frame that
        has been sitting in the ring is a stale observation, and acting on it
        adds its age to the control loop's latency for no benefit. Anything a
        controller skips here is a frame it was never going to be able to act
        on.

        This matters more the slower inference is. At ~95 ms per ACT forward
        against a 30 fps camera, roughly three frames queue during every
        query; taking the oldest would mean acting on a 100 ms old view of the
        world, on top of the 95 ms the forward itself costs.
        """
        if len(out) < self.frame_bytes():
            raise Error(
                "camera_thread: a " + String(len(out)) + "-byte buffer for a "
                + String(self.frame_bytes()) + "-byte frame"
            )
        var v = self.ring.view()
        var n = 0
        while True:
            var c = v.begin_pop()
            if not c.ok():
                break
            # ⚠ COPY EVERY ONE, rather than peeking ahead to find the last.
            # `begin_pop` is the only way to know whether another frame
            # follows, and the copy it commits to cannot be taken back — so
            # the loop overwrites `out` each time and the final iteration is
            # the one that survives. At 921 KB and a queue depth of three or
            # four this is well under a millisecond, against a 95 ms forward.
            unsafe_memcpy(dest=_erase(out), src=c.data(), count=c.len)
            v.end_pop()
            n += 1
        if n == 0:
            self._starved += 1
        return n

    def take_blocking(
        mut self, mut out: List[UInt8], timeout_ms: Int = 2000
    ) raises -> Bool:
        """`take`, but WAIT for the next frame. False only on timeout.

        ⚠ THIS IS WHAT PACES A RECORDER, and the measurement is the reason.
        With both sides threaded, a loop clocked to 30 Hz by a spin still lost
        22 frames over 8 s: the camera free-runs at its own rate and any
        surplus fills the ring. Letting the CAMERA be the clock removes the
        mismatch entirely — one tick per frame, by construction.

        ⚠ A TIMEOUT, NOT a forever-wait. `pop_blocking`'s own warning: a dead
        producer never delivers. False here means the camera stopped, which a
        recorder must treat as the end of the run rather than retry.
        """
        if len(out) < self.frame_bytes():
            raise Error(
                "camera_thread: a " + String(len(out)) + "-byte buffer for a "
                + String(self.frame_bytes()) + "-byte frame"
            )
        var v = self.ring.view()
        var c = v.pop_blocking(timeout_ms * 1000)
        if not c.ok():
            self._starved += 1
            return False
        unsafe_memcpy(dest=_erase(out), src=c.data(), count=c.len)
        v.end_pop()
        return True

    def drain(mut self) -> Int:
        """Discard every queued frame. Returns how many.

        Called just before an episode starts: frames captured while the
        operator was reading a prompt are not part of the episode, and leaving
        them in the ring would prepend them to it.
        """
        var v = self.ring.view()
        var n = 0
        while True:
            var c = v.begin_pop()
            if not c.ok():
                return n
            v.end_pop()
            n += 1

    def frames(self) -> Int:
        return self.ring.view().pushed()

    def dropped(self) -> Int:
        """Frames the CAMERA produced and the ring refused — the consumer was
        behind. Every one is a frame missing from the recording."""
        return self.ring.view().dropped()

    def starved(self) -> Int:
        """Ticks where the consumer asked and nothing was queued."""
        return self._starved

    def depth(self) -> Int:
        return self.ring.view().depth()

    def read_failures(self) -> Int:
        return Int(self.block.acquire_load(CELL_READ_FAIL))

    def stop(mut self, drain_ms: Int = 500) raises:
        if not self.running:
            return
        if self._thread:
            self._thread.value().stop(drain_ms)
        self._thread = None
        self.running = False


def _sleep_ms(ms: Int):
    _ = sleep_us(ms * 1000)
