# +--------------------------------------------------------------------------+ #
# | The speaker model on its own thread: an utterance in, 256 numbers out
# +--------------------------------------------------------------------------+ #
"""`SpeakerService`: the voice embedding of an utterance, computed off the
render loop (`noeira.ai.local_service`), polled like the other in-process
models.

    var spk = SpeakerService.from_spec("native")       # loads in the background
    spk.start(samples)                                  # List[Int16], 16 kHz mono; never blocks
    if spk.poll():                                      # once a frame
        var emb = spk.result()                          # List[Float32], 256 values; raises on a failure

Spec: `native` = `~/.cache/noeira/local-ai/speaker/resnet34.safetensors`
(`tools/speaker/extract_wespeaker.py`); `native:<file>` another file.

ON THE CPU, one thread: about 1.3 s of computing a second of speech on the
Orin (`tests/audio/test_speaker_reference.mojo`), so an utterance's voice is
known a few seconds after its words. The people store is built for that
(`noeira.brain.people`: the words first, the voice when it comes). One
utterance at a time: `start` while one is in flight raises, `busy` says so.

An utterance longer than `SPK_MAX_S` is cut to its first seconds (the error
rate falls little past 10 s); one shorter than 8 frames (0.1 s) fails.
"""

from std.os import getenv

from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from .fbank import kaldi_fbank, subtract_mean
from .model import SpeakerNet, SPK_EMB


comptime SPK_MAX_S: Int = 10
comptime SPK_REQ_SLOT: Int = SPK_MAX_S * 16000 * 2 + 64
comptime SPK_REP_SLOT: Int = SPK_EMB * 4 + 64


struct SpeakerModel(LocalModel):
    var path: String
    var net: Optional[SpeakerNet]

    def __init__(out self, var path: String):
        self.path = path^
        self.net = None

    def load(mut self) raises:
        self.net = Optional(SpeakerNet.load(self.path))

    def warm(mut self) raises:
        pass

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        """Little-endian int16 samples -> little-endian float32 embedding."""
        var n = len(request) // 2
        var x = List[Float32](capacity=n)
        for i in range(n):
            var v = Int(request[2 * i]) | (Int(request[2 * i + 1]) << 8)
            if v >= 32768:
                v -= 65536
            x.append(Float32(v))
        var feats = kaldi_fbank(x)
        subtract_mean(feats)
        var emb = self.net.value().embed(feats)
        var out = List[UInt8](capacity=len(emb) * 4)
        var bp = emb.unsafe_ptr().unsafe_bitcast[UInt8]()
        for k in range(len(emb) * 4):
            out.append(bp[unsafe_offset=k])
        return out^


struct SpeakerService(Movable):
    var call: LocalCall
    var bg: BackgroundThread[LocalWorker[SpeakerModel]]
    var started: Bool

    def __init__(out self, var path: String = String("")) raises:
        if path.byte_length() == 0:
            path = getenv("HOME") + "/.cache/noeira/local-ai/speaker/resnet34.safetensors"
        var call = LocalCall(SPK_REQ_SLOT, SPK_REP_SLOT)
        var bg = BackgroundThread(LocalWorker(SpeakerModel(path^), call))
        self.call = call^
        self.bg = bg^
        self.started = False

    @staticmethod
    def from_spec(spec: String) raises -> SpeakerService:
        if spec == "native":
            return SpeakerService()
        if spec.startswith("native:"):
            return SpeakerService(String(spec[byte=7:]))
        raise Error("speaker: SpeakerService takes 'native' or 'native:<file>', not '" + spec + "'")

    def state(self) -> Int:
        return self.call.state()

    def busy(self) -> Bool:
        """An utterance is in flight (its result not taken yet)."""
        return self.started

    def start(mut self, ref samples: List[Int16]) raises:
        """Queue an utterance (16 kHz mono); its first `SPK_MAX_S` seconds are used."""
        if self.started:
            raise Error("speaker: an utterance is already in flight")
        var n = min(len(samples), SPK_MAX_S * 16000)
        var b = List[UInt8](capacity=n * 2)
        for i in range(n):
            var v = Int(samples[i]) & 0xFFFF
            b.append(UInt8(v & 0xFF))
            b.append(UInt8(v >> 8))
        self.call.begin(b, String("speaker"))
        self.started = True

    def poll(mut self) raises -> Bool:
        """True once the utterance in flight is done (either way)."""
        return self.started and self.call.poll()

    def result(mut self) raises -> List[Float32]:
        """The embedding; raises when the model failed on the utterance."""
        while not self.call.poll():
            _ = sleep_us(500)
        self.started = False
        var b = self.call.take(String("speaker"))
        if len(b) != SPK_EMB * 4:
            raise Error("speaker: a reply of " + String(len(b)) + " bytes")
        var out = List[Float32](capacity=SPK_EMB)
        var fp = b.unsafe_ptr().unsafe_bitcast[Float32]()
        for k in range(SPK_EMB):
            out.append(fp[unsafe_offset=k])
        return out^

    def latency_ms(self) -> Float64:
        return self.call.latency_ms()
