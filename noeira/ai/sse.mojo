# +--------------------------------------------------------------------------+ #
# | Server-sent events — the framing every streaming model API uses
# +--------------------------------------------------------------------------+ #
"""An incremental `text/event-stream` parser.

    var p = SseParser()
    for chunk in network_chunks:
        var events = p.feed(chunk)
        for e in events:
            handle(e.event, e.data)

The WHATWG framing, reduced to what model APIs send: `event:` and `data:`
fields, events separated by a blank line, `:` comment lines (keep-alives)
ignored, several `data:` lines joined with `\\n`. `id:` / `retry:` are
parsed away and dropped — a model stream is never resumed by id.

⚠ A CHUNK BOUNDARY CAN FALL ANYWHERE — mid-line, mid-UTF-8-character,
between the `\\r` and `\\n` of a CRLF. The parser holds the unterminated tail
in bytes and only decodes COMPLETE lines, so no boundary can split a
character or a field.
"""

from noeira.core.bytes import string_from_byte_span


@fieldwise_init
struct SseEvent(Copyable, Movable):
    var event: String
    """The `event:` field; "" when absent (OpenAI sends none)."""
    var data: String


struct SseParser(Movable):
    var _tail: List[UInt8]
    var _event: String
    var _data: String
    var _has_data: Bool

    def __init__(out self):
        self._tail = List[UInt8]()
        self._event = String("")
        self._data = String("")
        self._has_data = False

    def __init__(out self, *, deinit move: Self):
        self._tail = move._tail^
        self._event = move._event^
        self._data = move._data^
        self._has_data = move._has_data

    def _line(mut self, ref buf: List[UInt8], start: Int, end: Int, mut out: List[SseEvent]):
        var e = end
        if e > start and buf[e - 1] == 0x0D:  # CR of a CRLF
            e -= 1
        if e == start:  # blank line: dispatch
            if self._has_data:
                out.append(SseEvent(self._event, self._data))
            self._event = String("")
            self._data = String("")
            self._has_data = False
            return
        if buf[start] == 0x3A:  # ":" comment / keep-alive
            return
        var colon = -1
        for i in range(start, e):
            if buf[i] == 0x3A:
                colon = i
                break
        var name: String
        var value: String
        if colon < 0:
            name = string_from_byte_span(buf, start, e)
            value = String("")
        else:
            name = string_from_byte_span(buf, start, colon)
            var v0 = colon + 1
            if v0 < e and buf[v0] == 0x20:  # one optional space
                v0 += 1
            value = string_from_byte_span(buf, v0, e)
        if name == "data":
            if self._has_data:
                self._data += "\n"
            self._data += value
            self._has_data = True
        elif name == "event":
            self._event = value^

    def feed(mut self, ref chunk: List[UInt8]) -> List[SseEvent]:
        """Consume a chunk; return every event it completed."""
        var out = List[SseEvent]()
        var buf = self._tail.copy()
        for i in range(len(chunk)):
            buf.append(chunk[i])
        var start = 0
        for i in range(len(buf)):
            if buf[i] == 0x0A:
                self._line(buf, start, i, out)
                start = i + 1
        var rest = List[UInt8](capacity=len(buf) - start)
        for i in range(start, len(buf)):
            rest.append(buf[i])
        self._tail = rest^
        return out^

    def finish(mut self) -> List[SseEvent]:
        """Flush at end of stream: a last event not followed by a blank line
        still counts (some servers close right after the final `data:`)."""
        var out = List[SseEvent]()
        if len(self._tail) > 0:
            var t = self._tail.copy()
            self._line(t, 0, len(t), out)
            self._tail = List[UInt8]()
        if self._has_data:
            out.append(SseEvent(self._event, self._data))
            self._event = String("")
            self._data = String("")
            self._has_data = False
        return out^
