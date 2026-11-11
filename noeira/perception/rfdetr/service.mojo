# +--------------------------------------------------------------------------+ #
# | RF-DETR as a detector service: COCO words -> masks -> 3D places
# +--------------------------------------------------------------------------+ #
"""`RfDetrService`: `Sam3Service`'s interface (`noeira.perception.detector`) over
`RfDetr` — the real-time, CLOSED-vocabulary detector (COCO's 80 classes,
Apache-2.0). Same `Detection`s, same lifting (`noeira.perception.detections`).

    var det = RfDetrService.from_spec("rfdetr")          # loads in the background
    det.start(picture, words)                            # ["fridge", "sofa", "table"]: each must name a COCO class
    while not det.poll(): ...                            # ONE forward whatever the words
    var found = det.result()                             # List[Detection] (word = index in the request)

A word must name a COCO class (`coco_id`): its COCO name ("refrigerator",
"couch", "dining table", "tv", ...) or an alias ("fridge", "sofa", "table",
"television"). Anything else — "door", which COCO does not have — FAILS the
call with the reason, at `start` (no model needed to know). The words beyond
COCO come by fine-tuning (SPATIAL_PERCEPTION_PLAN P3).

Unlike SAM 3 the cost does not grow with the words: one forward, then the
words select among its 100 x 91 (query, class) scores (`RfDetr.post_process`'s
`keep`: rfdetr's top-100 over all classes, then the words' classes).

Spec: `rfdetr` (`RFDETR_CHECKPOINT`, else `~/.cache/noeira/rfdetr/seg-small/model.safetensors`)
or `rfdetr:<model.safetensors>`.
"""

from max.gpu.host import DeviceContext

from noeira.ai.picture import Picture
from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from noeira.perception.detections import (
    Detection, DET_REQ_SLOT, DET_REP_SLOT, encode_request, decode_request, ReplyWriter, decode_reply,
)
from .rfdetr import RfDetr, rfdetr_checkpoint


def coco_names() -> List[String]:
    """COCO category id -> name (the model's label index IS the id; "" = unused)."""
    return [
        "", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "", "backpack", "umbrella", "", "", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "", "dining table", "", "", "toilet", "", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]


def coco_id(word: String) raises -> Int:
    """The COCO category id a word names (its COCO name or an alias)."""
    var w = word.lower()
    if w == "fridge":
        w = String("refrigerator")
    elif w == "sofa":
        w = String("couch")
    elif w == "table":
        w = String("dining table")
    elif w == "television" or w == "tv set":
        w = String("tv")
    elif w == "plant":
        w = String("potted plant")
    elif w == "phone":
        w = String("cell phone")
    var names = coco_names()
    for i in range(len(names)):
        if names[i].byte_length() > 0 and names[i] == w:
            return i
    raise Error("detector (rfdetr): '" + word + "' is not one of COCO's 80 classes (RF-DETR's vocabulary is"
                + " closed: a word beyond COCO needs fine-tuning, or the open-vocabulary Sam3Service)")


# ── the model, on the worker thread ───────────────────────────────────────

struct RfDetrModel(LocalModel):
    var path: String
    var threshold: Float32
    var det: Optional[RfDetr]

    def __init__(out self, var path: String, threshold: Float32):
        self.path = path^
        self.threshold = threshold
        self.det = None

    def load(mut self) raises:
        self.det = RfDetr(DeviceContext(), self.path)

    def warm(mut self) raises:
        var grey = List[UInt8](length=480 * 640 * 3, fill=128)
        _ = self.det.value().detect(Picture(grey^, 480, 640), self.threshold)

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        var req = decode_request(request)
        var ids = List[Int]()
        for ref w in req.words:
            ids.append(coco_id(w))
        var found = self.det.value().detect(req.picture, self.threshold, ids)
        var reply = ReplyWriter()
        for ref inst in found:
            for wi in range(len(ids)):
                if ids[wi] == inst.label:  # (two words naming one class each get it)
                    reply.add(wi, inst.score, inst.box, inst.mask, req.picture.height, req.picture.width)
        return reply.done()


# ── the caller's end ──────────────────────────────────────────────────────

struct RfDetrService(Movable):
    var call: LocalCall
    var bg: BackgroundThread[LocalWorker[RfDetrModel]]

    def __init__(out self, var path: String = String(""), threshold: Float32 = 0.3) raises:
        """`path`: an RF-DETR `model.safetensors` (empty: `rfdetr_checkpoint()`).
        `threshold`: the score a detection must pass (0.3, as `Sam3Service`)."""
        if path.byte_length() == 0:
            path = rfdetr_checkpoint()
        var call = LocalCall(DET_REQ_SLOT, DET_REP_SLOT)
        var bg = BackgroundThread(LocalWorker(RfDetrModel(path^, threshold), call))
        self.call = call^
        self.bg = bg^

    @staticmethod
    def from_spec(spec: String) raises -> RfDetrService:
        """`rfdetr` or `rfdetr:<model.safetensors>`."""
        if spec == "rfdetr":
            return RfDetrService()
        if spec.startswith("rfdetr:"):
            return RfDetrService(String(spec[byte=7:]))
        raise Error("detector: RfDetrService takes 'rfdetr' or 'rfdetr:<model.safetensors>', not '" + spec + "'")

    def state(self) -> Int:
        return self.call.state()

    def load_ms(self) -> Int:
        return self.call.load_ms()

    def start(mut self, ref picture: Picture, ref words: List[String]) raises:
        """Queue one picture and its words. Never blocks; a word COCO does not
        name raises HERE."""
        for ref w in words:
            _ = coco_id(w)
        var b = encode_request(picture, words)
        self.call.begin(b, String("detector (rfdetr)"))

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
        var b = self.call.take(String("detector (rfdetr)"))
        return decode_reply(b, String("detector (rfdetr)"))

    def latency_ms(self) -> Float64:
        return self.call.latency_ms()

    def cancel(mut self) raises:
        self.call.cancel()
