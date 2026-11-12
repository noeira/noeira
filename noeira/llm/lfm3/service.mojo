# +--------------------------------------------------------------------------+ #
# | LFM3 chat as a ChatService: spoken replies in this process, on their own thread
# +--------------------------------------------------------------------------+ #
"""`Lfm3ChatService` is what `ChatClient` is to the voice loop, without the
server: `start(messages, system)` / `poll` once a frame / `done` / `result() ->
ChatResponse` / `cancel`, with LFM3-1.7B (`Lfm3Chat`, the LFM3 backbone on the
device) on a worker thread (`noeira.ai.local_service`).

    var llm = Lfm3ChatService.from_spec("native")   # loads in the background (~3 s, int8)
    llm.set_reply_limits(120)
    llm.start(msgs, system)                          # List[ChatMessage]: user / assistant turns
    while not llm.done(): _ = llm.poll()
    var r = llm.result()                             # r.text, r.output_tokens, r.first_token_ms

    comptime Loop = G1VoiceLoopOf[STT, TTS, DEC, Lfm3ChatService]

Spec: `native` = `~/.cache/noeira/local-ai/lfm3/LFM3-1.7B` with int8 weights
(1.7 GB, ~41 tokens/s on the Orin; docs/NN_MODEL_PORTS_PLAN.md §12.7);
`native-bf16` the bf16 build of the same (3.5 GB, ~24 tokens/s, the reference-exact
one); `native:<dir>` / `native-bf16:<dir>` another LFM3 text checkpoint. Build
with `-D LFM3_BF16=1` on NVIDIA.

Each request is a conversation of its own (`messages`, as the HTTP client
takes them), rendered with Liquid's template, thinking off, and answered with
Liquid's sampler (temperature 0.1, top_k 50, repetition penalty 1.1; a fixed
seed, so the same request gives the same reply). The system prompt is computed
once and reused while it does not change (`Lfm3Chat.keep_system`): a reply
costs its own messages, not the robot's persona.

Not served: tools, pictures, streaming (`poll` returns "" and the text comes
with `result`; a short spoken reply is ~0.5 s).

The request, one message on the ring: `[u32 max_tokens] system \\0` then per
message `role (u|a) text \\0`. The reply: `[u32 input tokens][u32 output
tokens][u8 ended][u32 first token us] text`.
"""

from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.ai.chat import ChatService, ChatMessage, ChatResponse, ToolSpec
from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from .chat import Lfm3Chat, ChatSettings


comptime CS_REQ_SLOT: Int = 256 * 1024
comptime CS_REP_SLOT: Int = 64 * 1024
comptime CS_CAP: Int = 4096
"""The longest conversation one request may carry, in tokens."""


def _put_u32(mut b: List[UInt8], v: Int):
    for k in range(4):
        b.append(UInt8((v >> (8 * k)) & 0xFF))


def _get_u32(ref b: List[UInt8], at: Int) -> Int:
    var v = 0
    for k in range(4):
        v |= Int(b[at + k]) << (8 * k)
    return v


def _text(ref b: List[UInt8], a: Int, e: Int) -> String:
    var t = List[UInt8](capacity=max(e - a, 0))
    for i in range(a, e):
        t.append(b[i])
    return String(unsafe_from_utf8=Span(t))


def _find0(ref b: List[UInt8], start: Int) -> Int:
    for i in range(start, len(b)):
        if b[i] == 0:
            return i
    return -1


struct Lfm3ChatModel(LocalModel):
    var dir: String
    var int8: Bool
    var chat: Optional[Lfm3Chat]

    def __init__(out self, var dir: String, int8: Bool):
        self.dir = dir^
        self.int8 = int8
        self.chat = None

    def load(mut self) raises:
        var c = Lfm3Chat.load(DeviceContext(), self.dir, CS_CAP, self.int8)
        c.keep_system = True
        self.chat = Optional(c^)

    def warm(mut self) raises:
        ref c = self.chat.value()
        c.reset(String("You are a robot."))
        c.user(String("Hello!"))
        var s = ChatSettings()
        s.max_tokens = 8  # the chunk path (prefill) and the one-token path, both
        c.start(s)
        while c.next():
            pass

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        """The request (module note) -> the reply's frame."""
        var n = len(request)
        if n < 5:
            raise Error("lfm3 chat: a request is [u32 max_tokens] system \\0 (role text \\0)*")
        var max_tokens = _get_u32(request, 0)
        var cut = _find0(request, 4)
        if cut < 0:
            raise Error("lfm3 chat: no end to the system prompt")
        ref c = self.chat.value()
        c.reset(_text(request, 4, cut))
        var at = cut + 1
        var last = String("")
        while at < n:
            var e = _find0(request, at + 1)
            if e < 0:
                raise Error("lfm3 chat: a message without its end")
            var role = request[at]
            var text = _text(request, at + 1, e)
            if role == 117:  # u
                c.user(text)
                last = "user"
            elif role == 97:  # a
                c.assistant(text)
                last = "assistant"
            else:
                raise Error("lfm3 chat: only user and assistant messages are served")
            at = e + 1
        if last != "user":
            raise Error("lfm3 chat: the conversation must end with a user message")
        var s = ChatSettings()
        s.max_tokens = max(1, max_tokens)
        var input_tokens = c.cache.pos + len(c.pending)
        var t0 = perf_counter_ns()
        c.start(s)
        var first_us = Int(perf_counter_ns() - t0) // 1000
        while c.next():
            pass
        var ended = len(c.reply) > 0 and c.reply[len(c.reply) - 1] == c.eos
        var text = c.reply_text()
        var b = List[UInt8](capacity=13 + text.byte_length())
        _put_u32(b, input_tokens)
        _put_u32(b, len(c.reply))
        b.append(UInt8(1) if ended else UInt8(0))
        _put_u32(b, first_us)
        for x in text.as_bytes():
            b.append(x)
        return b^


struct Lfm3ChatService(Movable, ChatService):
    var call: LocalCall
    var bg: BackgroundThread[LocalWorker[Lfm3ChatModel]]
    var max_tokens: Int
    var started: Bool
    var over: Bool
    """The call in flight has finished (`poll` saw it)."""

    def __init__(out self, var dir: String = String(""), int8: Bool = True) raises:
        if dir.byte_length() == 0:
            dir = getenv("HOME") + "/.cache/noeira/local-ai/lfm3/LFM3-1.7B"
        var call = LocalCall(CS_REQ_SLOT, CS_REP_SLOT)
        var bg = BackgroundThread(LocalWorker(Lfm3ChatModel(dir^, int8), call))
        self.call = call^
        self.bg = bg^
        self.max_tokens = 256
        self.started = False
        self.over = False

    @staticmethod
    def from_spec(spec: String) raises -> Lfm3ChatService:
        """`native` / `native:<dir>` (int8 weights), `native-bf16` / `native-bf16:<dir>`."""
        if spec == "native":
            return Lfm3ChatService()
        if spec.startswith("native:"):
            return Lfm3ChatService(String(spec[byte=7:]))
        if spec == "native-bf16":
            return Lfm3ChatService(String(""), False)
        if spec.startswith("native-bf16:"):
            return Lfm3ChatService(String(spec[byte=12:]), False)
        raise Error("chat: Lfm3ChatService takes 'native[-bf16]' or 'native[-bf16]:<dir>', not '" + spec + "'")

    def state(self) -> Int:
        return self.call.state()

    def load_ms(self) -> Int:
        return self.call.load_ms()

    def set_reply_limits(mut self, max_tokens: Int) raises:
        """At most `max_tokens` tokens; thinking is always off here."""
        self.max_tokens = max_tokens

    def warm_up(mut self):
        pass  # the worker loads and warms itself up on its own thread

    def start(
        mut self,
        ref messages: List[ChatMessage],
        system: String = String(""),
        tools: List[ToolSpec] = List[ToolSpec](),
        stream: Bool = True,
    ) raises:
        """Queue the reply (never blocks). `stream` is accepted and ignored: the
        text comes with `result`."""
        if len(tools) > 0:
            raise Error("chat (LFM3): tools are not served")
        var b = List[UInt8](capacity=256)
        _put_u32(b, self.max_tokens)
        for x in system.as_bytes():
            if x == 0:
                raise Error("chat (LFM3): a NUL byte in the system prompt")
            b.append(x)
        b.append(0)
        for ref m in messages:
            if len(m.images_b64) > 0:
                raise Error("chat (LFM3): pictures are not served")
            if m.role == "user":
                b.append(117)
            elif m.role == "assistant":
                b.append(97)
            else:
                raise Error("chat (LFM3): a '" + m.role + "' message (user and assistant only)")
            for x in m.text.as_bytes():
                if x == 0:
                    raise Error("chat (LFM3): a NUL byte in a message")
                b.append(x)
            b.append(0)
        self.call.begin(b, String("chat (LFM3)"))
        self.started = True
        self.over = False

    def poll(mut self, timeout_ms: Int = 0) raises -> String:
        if self.call.poll():
            self.over = True
            return String("")
        var waited = 0
        while waited < timeout_ms * 1000:
            _ = sleep_us(500)
            waited += 500
            if self.call.poll():
                self.over = True
                break
        return String("")

    def done(self) -> Bool:
        """True once the call is over (either way), or none was started.
        Reads the state `poll` advanced: call `poll` once a frame."""
        return not self.started or self.over

    def result(mut self) raises -> ChatResponse:
        while not self.call.poll():
            _ = sleep_us(500)
        self.started = False
        var b = self.call.take(String("chat (LFM3)"))
        if len(b) < 13:
            raise Error("chat (LFM3): a short reply")
        var r = ChatResponse()
        r.input_tokens = _get_u32(b, 0)
        r.output_tokens = _get_u32(b, 4)
        r.stop_reason = String("stop") if b[8] == 1 else String("length")
        r.first_token_ms = Float64(_get_u32(b, 9)) / 1000.0
        r.text = _text(b, 13, len(b))
        r.model = String("LFM3-1.7B (in process)")
        r.latency_ms = self.call.latency_ms()
        r.message = ChatMessage.assistant(r.text.copy())
        return r^

    def cancel(mut self) raises:
        self.call.cancel()
        self.started = False
