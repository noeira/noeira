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
not move; the robot asks about what it is looking at, not every frame. The
real-time alternative on COCO's classes is `rfdetr.service.RfDetrService`
(same `Detection`s, same lifting: `noeira.perception.detections`, which also
holds the request / reply framing).

Spec: `sam3` (`SAM3_CHECKPOINT`, else the Hub cache's `facebook/sam3`) or
`sam3:<model.safetensors>` (the bf16 detector-only extract fits the Orin's
disk). Build with `-D SAM3_BF16=1` on NVIDIA (or `-D SAM3_TF32=1`). Licence:
Meta's SAM licence (gated on the Hub).
"""

from max.gpu.host import DeviceContext

from noeira.ai.picture import Picture
from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from noeira.perception.sam3 import Sam3
from noeira.perception.sam3.ops import sam3_checkpoint
from .detections import (
    Detection, DET_REQ_SLOT, DET_REP_SLOT, encode_request, decode_request, ReplyWriter, decode_reply,
    mask_points, mask_place, detections_endpoints_world,
)


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
        var req = decode_request(request)
        var img = self.sam.value().encode_image(req.picture)
        var reply = ReplyWriter()
        for wi in range(len(req.words)):
            var found = self.sam.value().segment(img, req.words[wi], self.threshold)
            for ref inst in found:
                reply.add(wi, inst.score, inst.box, inst.mask, req.picture.height, req.picture.width)
        return reply.done()


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
        var b = encode_request(picture, words)
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
        return decode_reply(b, String("detector (sam3)"))

    def latency_ms(self) -> Float64:
        """The model's own time for the last call."""
        return self.call.latency_ms()

    def cancel(mut self) raises:
        self.call.cancel()
