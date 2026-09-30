# +--------------------------------------------------------------------------+ #
# | Transport — one POST with retries, shared by every AI client
# +--------------------------------------------------------------------------+ #
"""The part every provider has in common: POST, retry what is retryable,
surface the server's own error body, time the call.

Built on `noeira/io/http.mojo` (the libcurl shim — `pixi run build-http`
once). A client keeps ONE `HttpClient`, so a demo that calls a model in a
loop pays for the TLS handshake once.

## What is retried

Transport failures (DNS, reset, timeout) and the statuses a provider uses
for "not now": 408, 409, 429, 500, 502, 503, 504 and 529 (Anthropic and
TypeSafe's "overloaded"). Backoff doubles from `base_delay_s`. Everything
else — 400, 401, 403, 404, 422 — is a mistake in the request, and a retry
would only repeat it; it raises at once with the body, because the body is
where the provider says WHICH field was wrong.
"""

from std.time import perf_counter_ns, sleep

from noeira.io.http import HttpClient, HttpResponse
from noeira.io.json import JsonDoc, parse_json


def is_retryable(status: Int) -> Bool:
    return (
        status == 408 or status == 409 or status == 429 or status == 500
        or status == 502 or status == 503 or status == 504 or status == 529
    )


def _bytes_of(s: String) -> List[UInt8]:
    var b = List[UInt8](capacity=s.byte_length())
    var src = s.as_bytes()
    for i in range(s.byte_length()):
        b.append(src[i])
    return b^


struct ApiReply(Movable):
    """A 2xx response and how long it took, retries included."""

    var body: List[UInt8]
    var latency_ms: Float64
    var attempts: Int

    def __init__(out self, var body: List[UInt8], latency_ms: Float64, attempts: Int):
        self.body = body^
        self.latency_ms = latency_ms
        self.attempts = attempts

    def json(self) raises -> JsonDoc:
        return parse_json(self.body.copy())

    def take_body(deinit self) -> List[UInt8]:
        return self.body^


def post_api(
    mut http: HttpClient,
    var url: String,
    ref body: List[UInt8],
    var content_type: String,
    var what: String,
    retries: Int = 2,
    base_delay_s: Float64 = 0.5,
) raises -> ApiReply:
    """POST `body`; return the 2xx body or raise with status + server body.

    `what` names the call in the error ("anthropic chat", "jev decide"),
    since a bare URL does not say which demo step failed.
    """
    var t0 = perf_counter_ns()
    var delay = base_delay_s
    var last = String("")
    for attempt in range(retries + 1):
        if attempt > 0:
            sleep(delay)
            delay *= 2.0
        var resp: HttpResponse
        try:
            resp = http.request(String("POST"), url, body, content_type)
        except e:
            last = String(e)
            continue
        if resp.ok():
            var ms = Float64(perf_counter_ns() - t0) / 1e6
            return ApiReply(resp^.take_body(), ms, attempt + 1)
        last = "HTTP " + String(resp.status) + ": " + resp.text()
        if not is_retryable(resp.status):
            break
    raise Error(what + " failed (" + url + "): " + last)


def post_json_api(
    mut http: HttpClient,
    var url: String,
    var json_body: String,
    var what: String,
    retries: Int = 2,
) raises -> ApiReply:
    var b = _bytes_of(json_body)
    return post_api(http, url^, b, String("application/json"), what^, retries)


# ═══════════════════════════════════════════════════════════════════════════
# multipart/form-data — the upload shape of every speech-to-text endpoint
# ═══════════════════════════════════════════════════════════════════════════


struct MultipartForm(Movable):
    """RFC 7578 body builder.

        var f = MultipartForm()
        f.field("model", "whisper-1")
        f.file("file", "audio.wav", "audio/wav", wav_bytes)
        var body = f.finish()
        http.request("POST", url, body, f.content_type())

    ⚠ THE BOUNDARY IS FIXED, not random. It only has to not occur inside a
    part; a 38-character marker in a WAV's PCM bytes is not a real risk, and
    a fixed one keeps request bodies reproducible for the mock-server gate.
    """

    var _body: List[UInt8]
    var boundary: String

    def __init__(out self):
        self._body = List[UInt8]()
        self.boundary = String("----noeira-form-7f3c1a9e5b2d4086")

    def __init__(out self, *, deinit move: Self):
        self._body = move._body^
        self.boundary = move.boundary^

    def _put(mut self, s: String):
        var b = s.as_bytes()
        for i in range(s.byte_length()):
            self._body.append(b[i])

    def field(mut self, name: String, value: String):
        self._put("--" + self.boundary + "\r\n")
        self._put('Content-Disposition: form-data; name="' + name + '"\r\n\r\n')
        self._put(value)
        self._put("\r\n")

    def file(
        mut self,
        name: String,
        filename: String,
        mime: String,
        ref data: List[UInt8],
    ):
        self._put("--" + self.boundary + "\r\n")
        self._put(
            'Content-Disposition: form-data; name="' + name + '"; filename="'
            + filename + '"\r\n'
        )
        self._put("Content-Type: " + mime + "\r\n\r\n")
        for i in range(len(data)):
            self._body.append(data[i])
        self._put("\r\n")

    def content_type(self) -> String:
        return "multipart/form-data; boundary=" + self.boundary

    def finish(mut self) -> List[UInt8]:
        self._put("--" + self.boundary + "--\r\n")
        return self._body.copy()
