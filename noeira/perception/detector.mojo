# +--------------------------------------------------------------------------+ #
# | The open-vocabulary detector as a service: words -> masks -> 3D places
# +--------------------------------------------------------------------------+ #
"""SPATIAL_PERCEPTION_PLAN P3 step 2: the oracle labels of step 1 replaced by
SAM 3 (`noeira.perception.sam3`) on the head camera's RGB, the depth lifting
its masks into 3D — into `places.PlaceMap` like the oracle's labels were.

    var det = Sam3Service.from_spec("sam3")              # loads in the background (~10 s on the Orin)
    det.start(picture, words)                            # a Picture (RGB24) and ["fridge", "door"]
    while not det.poll(): ...                            # once a frame; ~0.7 s + 0.17 s a word (Orin, bf16)
    var found = det.result()                             # List[Detection]: word, score, box, mask
    var p = mask_place(spec, depth, found[0].mask)       # its 3D place (sensor frame), [] if too few points
    detections_endpoints_world(origin, q, spec, depth, found, label_of_word, 4, ends, hits, labels)
    place_map.add_scan(ends, hits, labels)               # as the oracle's labelled returns

## The service

In this process, on its own thread (`noeira.ai.local_service`, as `D1Service`):
`start` never blocks, `poll` once a frame, `result` raises the model's error.
One request = one picture and its words: the picture is encoded ONCE, then
each word segments it (`Sam3.encode_image` / `Sam3.segment`). SAM 3 is slow
(0.7 s an image on the Orin): a LOW-RATE place-building service — the map does
not move; the robot asks about what it is looking at, not every frame.

Request: `[u32 words][u32 len][utf-8]... [picture]` (`Picture.append_to`).
Reply: `[u32 count]` then per detection `[u32 word][f32 score][f32 x0 y0 x1 y1]
[u32 h][u32 w][mask bits, row-major, LSB first, ceil(h w / 8) bytes]`.

Spec: `sam3` (`SAM3_CHECKPOINT`, else the Hub cache's `facebook/sam3`) or
`sam3:<model.safetensors>` (the bf16 detector-only extract fits the Orin's
disk). Build with `-D SAM3_BF16=1` on NVIDIA (or `-D SAM3_TF32=1`). Licence:
Meta's SAM licence (gated on the Hub).

## Lifting (any detector's masks)

The depth image must be REGISTERED to the RGB (the same pixel grid: NYU's
labelled frames are; a D435 does it with `rs2::align`), `spec` its intrinsics
(`depth.DepthCameraSpec`: OpenCV `fx fy cx cy`, planar depth, 0 = none). Points
are in the SENSOR frame (REP-103: x forward, y left, z up), `depth_to_points`'s
convention: `p = z (1, -(u - cx) / fx, -(v - cy) / fy)`.

`mask_place`: the per-axis MEDIAN of the mask's depth points (empty under
`min_points`) — `tools/perception/detector_eval.py`'s rule: a mask edge
bleeding onto the wall behind drags a mean by metres.

`detections_endpoints_world`: every `stride`-th pixel with depth becomes a
world endpoint and a hit (the depth camera's points, as `occupancy.
depth_endpoints_world` makes them), labelled with the word of the highest-
scoring detection whose mask holds it (`label_of_word[word]`), -1 otherwise.
⚠ A MASK IS NOT AN OBJECT: two instances of one word share its label, and a
false detection labels whatever it covered — the place map's two-hits rule
and re-perception (P3 step 1's rules) are what absorb it.
"""

from std.builtin.sort import sort
from std.memory import bitcast
from max.gpu.host import DeviceContext

from noeira.ai.picture import Picture, picture_from
from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from noeira.perception.sam3 import Sam3
from noeira.perception.sam3.ops import sam3_checkpoint
from .depth import DepthCameraSpec


comptime _F64 = DType.float64
comptime DET_REQ_SLOT: Int = 8 << 20
"""A 1280 x 720 frame (2.8 MB) and its words, with room."""
comptime DET_REP_SLOT: Int = 16 << 20
"""~400 detections' packed masks at 640 x 480 (38 KB each)."""


struct Detection(Copyable, Movable):
    var word: Int
    """The word's index in the request."""
    var score: Float32
    var box: List[Float32]
    """x0, y0, x1, y1 in the picture's pixels."""
    var mask: List[UInt8]
    """[height x width] 0 / 1, the picture's size."""
    var height: Int
    var width: Int

    def __init__(
        out self, word: Int, score: Float32, var box: List[Float32], var mask: List[UInt8], height: Int, width: Int,
    ):
        self.word = word
        self.score = score
        self.box = box^
        self.mask = mask^
        self.height = height
        self.width = width


# ── the model, on the worker thread ───────────────────────────────────────

struct Sam3Model(LocalModel):
    var path: String
    var threshold: Float32
    var sam: Optional[Sam3]

    def __init__(out self, var path: String, threshold: Float32):
        self.path = path^
        self.threshold = threshold
        self.sam = None

    def load(mut self) raises:
        self.sam = Sam3(DeviceContext(), self.path)

    def warm(mut self) raises:
        """A grey 640 x 480 frame and one word: the kernels, cuBLAS / cuDNN plans."""
        var grey = List[UInt8](length=480 * 640 * 3, fill=128)
        var img = self.sam.value().encode_image(Picture(grey^, 480, 640))
        _ = self.sam.value().segment(img, String("object"), self.threshold)

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        var at = 0
        var nw = _get_u32(request, at)
        at += 4
        var words = List[String]()
        for _ in range(nw):
            var n = _get_u32(request, at)
            at += 4
            if at + n > len(request):
                raise Error("sam3 service: a truncated word")
            var t = List[UInt8](capacity=n)
            for i in range(n):
                t.append(request[at + i])
            at += n
            words.append(String(unsafe_from_utf8=Span(t)))
        var pic = picture_from(request, at)
        if at != len(request):
            raise Error("sam3 service: " + String(len(request) - at) + " bytes after the picture")
        var img = self.sam.value().encode_image(pic)
        var out = List[UInt8]()
        _put_u32(out, 0)
        var count = 0
        for wi in range(len(words)):
            var found = self.sam.value().segment(img, words[wi], self.threshold)
            for ref inst in found:
                _put_u32(out, wi)
                _put_f32(out, inst.score)
                for k in range(4):
                    _put_f32(out, inst.box[k])
                _put_u32(out, pic.height)
                _put_u32(out, pic.width)
                _pack_bits(out, inst.mask)
                count += 1
        for k in range(4):
            out[k] = UInt8((count >> (8 * k)) & 0xFF)
        return out^


# ── the caller's end ──────────────────────────────────────────────────────

struct Sam3Service(Movable):
    var call: LocalCall
    var bg: BackgroundThread[LocalWorker[Sam3Model]]

    def __init__(out self, var path: String = String(""), threshold: Float32 = 0.3) raises:
        """`path`: a SAM 3 `model.safetensors` (empty: `sam3_checkpoint()`, read
        on the caller's thread so a missing checkpoint fails HERE). `threshold`:
        the score a detection must pass (the processor's 0.3)."""
        if path.byte_length() == 0:
            path = sam3_checkpoint()
        var call = LocalCall(DET_REQ_SLOT, DET_REP_SLOT)
        var bg = BackgroundThread(LocalWorker(Sam3Model(path^, threshold), call))
        self.call = call^
        self.bg = bg^

    @staticmethod
    def from_spec(spec: String) raises -> Sam3Service:
        """`sam3` or `sam3:<model.safetensors>`."""
        if spec == "sam3":
            return Sam3Service()
        if spec.startswith("sam3:"):
            return Sam3Service(String(spec[byte=5:]))
        raise Error("detector: Sam3Service takes 'sam3' or 'sam3:<model.safetensors>', not '" + spec + "'")

    def state(self) -> Int:
        """`local_service.SVC_LOADING` / `SVC_READY` / `SVC_FAILED`."""
        return self.call.state()

    def load_ms(self) -> Int:
        return self.call.load_ms()

    def start(mut self, ref picture: Picture, ref words: List[String]) raises:
        """Queue one picture and its words. Never blocks."""
        if len(words) == 0:
            raise Error("detector: no word to look for")
        var b = List[UInt8](capacity=16 + picture.framed_bytes() + 64 * len(words))
        _put_u32(b, len(words))
        for w in words:
            _put_u32(b, w.byte_length())
            for x in w.as_bytes():
                b.append(x)
        picture.append_to(b)
        self.call.begin(b, String("detector (sam3)"))

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        if self.call.poll():
            return True
        var waited = 0
        while waited < timeout_ms * 1000:
            _ = sleep_us(500)
            waited += 500
            if self.call.poll():
                return True
        return False

    def result(mut self) raises -> List[Detection]:
        var b = self.call.take(String("detector (sam3)"))
        var at = 0
        var count = _get_u32(b, at)
        at += 4
        var out = List[Detection]()
        for _ in range(count):
            var word = _get_u32(b, at)
            var score = _get_f32(b, at + 4)
            var box = List[Float32](capacity=4)
            for k in range(4):
                box.append(_get_f32(b, at + 8 + 4 * k))
            var h = _get_u32(b, at + 24)
            var w = _get_u32(b, at + 28)
            at += 32
            var mask = _unpack_bits(b, at, h * w)
            at += (h * w + 7) // 8
            out.append(Detection(word, score, box^, mask^, h, w))
        if at != len(b):
            raise Error("detector (sam3): " + String(len(b) - at) + " bytes after the detections")
        return out^

    def latency_ms(self) -> Float64:
        """The model's own time for the last call."""
        return self.call.latency_ms()

    def cancel(mut self) raises:
        self.call.cancel()


# ── lifting masks into 3D ─────────────────────────────────────────────────

def _check_size(spec: DepthCameraSpec, ref depth: List[Float64], ref mask: List[UInt8]) raises:
    if len(depth) != spec.npix() or len(mask) != spec.npix():
        raise Error("detector: depth (" + String(len(depth)) + ") and mask (" + String(len(mask))
                    + ") must both be the camera's " + String(spec.width) + "x" + String(spec.height)
                    + " pixels (a depth image registered to the RGB)")


def mask_points(spec: DepthCameraSpec, ref depth: List[Float64], ref mask: List[UInt8]) raises -> List[Float64]:
    """The mask's pixels with depth, unprojected (sensor frame, 3 floats each)."""
    _check_size(spec, depth, mask)
    var out = List[Float64]()
    for v in range(spec.height):
        var yo = (Float64(v) - spec.cy) / spec.fy
        for u in range(spec.width):
            var p = v * spec.width + u
            var z = depth[p]
            if mask[p] == 0 or z <= 0.0:
                continue
            out.append(z)
            out.append(-z * (Float64(u) - spec.cx) / spec.fx)
            out.append(-z * yo)
    return out^


def _median(var xs: List[Float64]) -> Float64:
    """numpy's: the mean of the two middle values for an even count."""
    sort(xs)
    var n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mask_place(
    spec: DepthCameraSpec, ref depth: List[Float64], ref mask: List[UInt8], min_points: Int = 20,
) raises -> List[Float64]:
    """The mask's 3D place, per-axis median of its depth points (sensor frame);
    empty when fewer than `min_points` of its pixels have depth."""
    var pts = mask_points(spec, depth, mask)
    var n = len(pts) // 3
    if n < min_points:
        return List[Float64]()
    var out = List[Float64](capacity=3)
    for a in range(3):
        var c = List[Float64](capacity=n)
        for i in range(n):
            c.append(pts[3 * i + a])
        out.append(_median(c^))
    return out^


def detections_endpoints_world(
    origin: Vec3Generic[_F64],
    q: QuatGeneric[_F64],
    spec: DepthCameraSpec,
    ref depth: List[Float64],
    ref detections: List[Detection],
    ref label_of_word: List[Int],
    stride: Int,
    mut endpoints: List[Float64],
    mut hits: List[Bool],
    mut labels: List[Int],
) raises:
    """A registered depth image and its detections to world endpoints, hits and
    labels for `PlaceMap.add_scan` (`origin` / `q`: the camera's world pose,
    sensor frame REP-103). Every `stride`-th pixel with depth; its label is
    `label_of_word[word]` of the best-scoring detection covering it, else -1."""
    if len(depth) != spec.npix():
        raise Error("detector: a depth image of " + String(len(depth)) + " pixels for a "
                    + String(spec.width) + "x" + String(spec.height) + " camera")
    for ref d in detections:
        if d.height != spec.height or d.width != spec.width:
            raise Error("detector: a " + String(d.width) + "x" + String(d.height) + " mask for a "
                        + String(spec.width) + "x" + String(spec.height) + " depth image (register it to the RGB)")
        if d.word < 0 or d.word >= len(label_of_word):
            raise Error("detector: no label for word " + String(d.word))
    endpoints = List[Float64]()
    hits = List[Bool]()
    labels = List[Int]()
    var s = max(1, stride)
    for p in range(0, spec.npix(), s):
        var z = depth[p]
        if z <= 0.0:
            continue
        var v = p // spec.width
        var u = p - v * spec.width
        var local = Vec3Generic[_F64](z, -z * (Float64(u) - spec.cx) / spec.fx, -z * (Float64(v) - spec.cy) / spec.fy)
        var w = q.rotate_vec(local)
        endpoints.append(origin.x + w.x)
        endpoints.append(origin.y + w.y)
        endpoints.append(origin.z + w.z)
        hits.append(True)
        var label = -1
        var best = Float32(-1.0)
        for ref d in detections:
            if d.mask[p] != 0 and d.score > best:
                best = d.score
                label = label_of_word[d.word]
        labels.append(label)


# ── framing ───────────────────────────────────────────────────────────────

def _put_u32(mut out: List[UInt8], v: Int):
    for k in range(4):
        out.append(UInt8((v >> (8 * k)) & 0xFF))


def _get_u32(ref b: List[UInt8], at: Int) raises -> Int:
    if at + 4 > len(b):
        raise Error("detector: a truncated message")
    var v = 0
    for k in range(4):
        v |= Int(b[at + k]) << (8 * k)
    return v


def _put_f32(mut out: List[UInt8], x: Float32):
    _put_u32(out, Int(bitcast[DType.uint32](x)))


def _get_f32(ref b: List[UInt8], at: Int) raises -> Float32:
    return bitcast[DType.float32](UInt32(_get_u32(b, at)))


def _pack_bits(mut out: List[UInt8], ref mask: List[UInt8]):
    var n = len(mask)
    var at = len(out)
    out.resize(at + (n + 7) // 8, 0)
    for i in range(n):
        if mask[i] != 0:
            out[at + i // 8] |= UInt8(1) << UInt8(i % 8)


def _unpack_bits(ref b: List[UInt8], at: Int, n: Int) raises -> List[UInt8]:
    if at + (n + 7) // 8 > len(b):
        raise Error("detector: a truncated mask")
    var out = List[UInt8](capacity=n)
    for i in range(n):
        out.append((b[at + i // 8] >> UInt8(i % 8)) & 1)
    return out^

