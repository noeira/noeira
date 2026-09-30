# +--------------------------------------------------------------------------+ #
# | Transport — one POST with retries, shared by every AI client
# +--------------------------------------------------------------------------+ #
"""The part every provider has in common: POST, retry what is retryable,
surface the server's own error body, time the call — blocking (`post_api`)
or driven by the caller's loop (`ApiCall`), with ONE retry policy for both,
since `post_api` is an `ApiCall` polled to the end.

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

from std.math import min
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


comptime CALL_IDLE = 0
comptime CALL_RUNNING = 1
comptime CALL_BACKOFF = 2
"""Waiting out a retry delay — without blocking: `poll` relaunches once the
delay has passed."""
comptime CALL_DONE = 3
comptime CALL_FAILED = 4


struct ApiCall(Movable):
    """One POST driven by the caller's loop: the non-blocking twin of
    `post_api`, with the same retry policy.

        var call = ApiCall()
        call.begin(http, url, body, "application/json", "jev decide")
        while not call.poll(http):          # 0 ms: returns at once
            var chunk = call.read_stream(http)
            draw_frame()
        var reply = call.reply(http)        # raises if the call failed

    ⚠ A RETRY IS ONLY SAFE BEFORE THE FIRST STREAMED BYTE. Once `read_stream`
    has handed body bytes to the caller, restarting would replay them; a
    failure after that point is final.
    """

    var url: String
    var body: List[UInt8]
    var content_type: String
    var what: String
    var state: Int
    var retries_left: Int
    var delay_s: Float64
    var attempts: Int
    var delivered: Bool
    var error: String
    var _t0_ns: Int
    var _resume_ns: Int
    var _reply_body: List[UInt8]
    var latency_ms: Float64

    def __init__(out self):
        self.url = String("")
        self.body = List[UInt8]()
        self.content_type = String("")
        self.what = String("")
        self.state = CALL_IDLE
        self.retries_left = 0
        self.delay_s = 0.5
        self.attempts = 0
        self.delivered = False
        self.error = String("")
        self._t0_ns = 0
        self._resume_ns = 0
        self._reply_body = List[UInt8]()
        self.latency_ms = 0.0

    def __init__(out self, *, deinit move: Self):
        self.url = move.url^
        self.body = move.body^
        self.content_type = move.content_type^
        self.what = move.what^
        self.state = move.state
        self.retries_left = move.retries_left
        self.delay_s = move.delay_s
        self.attempts = move.attempts
        self.delivered = move.delivered
        self.error = move.error^
        self._t0_ns = move._t0_ns
        self._resume_ns = move._resume_ns
        self._reply_body = move._reply_body^
        self.latency_ms = move.latency_ms

    def elapsed_ms(self) -> Float64:
        """Since `begin`, retries included."""
        return Float64(perf_counter_ns() - self._t0_ns) / 1e6

    def active(self) -> Bool:
        return self.state == CALL_RUNNING or self.state == CALL_BACKOFF

    def finished(self) -> Bool:
        return self.state == CALL_DONE or self.state == CALL_FAILED

    def begin(
        mut self,
        mut http: HttpClient,
        var url: String,
        var body: List[UInt8],
        var content_type: String,
        var what: String,
        retries: Int = 2,
        base_delay_s: Float64 = 0.5,
    ) raises:
        if self.active() or http.running():
            raise Error(what + ": a call is already in flight on this client — poll or cancel it")
        self.url = url^
        self.body = body^
        self.content_type = content_type^
        self.what = what^
        self.retries_left = retries
        self.delay_s = base_delay_s
        self.attempts = 0
        self.delivered = False
        self.error = String("")
        self._reply_body = List[UInt8]()
        self.latency_ms = 0.0
        self._t0_ns = perf_counter_ns()
        self._launch(http)

    def _launch(mut self, mut http: HttpClient) raises:
        self.attempts += 1
        self.state = CALL_RUNNING
        try:
            http.start(String("POST"), self.url, self.body, self.content_type)
        except e:
            self._retry_or_fail(String(e))

    def _retry_or_fail(mut self, var why: String):
        if self.retries_left > 0 and not self.delivered:
            self.retries_left -= 1
            self.state = CALL_BACKOFF
            self._resume_ns = perf_counter_ns() + Int(self.delay_s * 1e9)
            self.delay_s *= 2.0
            self.error = why^
        else:
            self.state = CALL_FAILED
            self.error = self.what + " failed (" + self.url + "): " + why

    def poll(mut self, mut http: HttpClient, timeout_ms: Int = 0) raises -> Bool:
        """Advance the call; True once it is DONE or FAILED. Waits up to
        `timeout_ms` for network activity (0 = not at all)."""
        if self.state == CALL_BACKOFF:
            var now = perf_counter_ns()
            if now < self._resume_ns:
                if timeout_ms > 0:
                    var wait_s = Float64(self._resume_ns - now) / 1e9
                    sleep(min(wait_s, Float64(timeout_ms) / 1e3))
                return False
            self._launch(http)
        if self.state != CALL_RUNNING:
            return self.finished()
        if not http.poll(timeout_ms):
            return False
        var resp: HttpResponse
        try:
            resp = http.finish()
        except e:
            self._retry_or_fail(String(e))
            return self.finished()
        if resp.ok():
            self.state = CALL_DONE
            self.latency_ms = Float64(perf_counter_ns() - self._t0_ns) / 1e6
            self._reply_body = resp^.take_body()
            return True
        var why = "HTTP " + String(resp.status) + ": " + resp.text()
        if is_retryable(resp.status):
            self._retry_or_fail(why^)
        else:
            self.state = CALL_FAILED
            self.error = self.what + " failed (" + self.url + "): " + why
        return self.finished()

    def read_stream(mut self, mut http: HttpClient) raises -> List[UInt8]:
        """Body bytes of a SUCCESSFUL response that arrived since the last
        call. Empty while the status is unknown or not 2xx — an error body is
        never mistaken for stream data; it lands in the raised error."""
        if self.state != CALL_RUNNING and self.state != CALL_DONE:
            return List[UInt8]()
        var st = http.status()
        if st < 200 or st >= 300:
            return List[UInt8]()
        var b = http.read_new()
        if len(b) > 0:
            self.delivered = True
        return b^

    def wait(mut self, mut http: HttpClient) raises -> ApiReply:
        """Block until finished — the synchronous path."""
        while not self.poll(http, 100):
            pass
        return self.reply()

    def reply(mut self) raises -> ApiReply:
        if self.state == CALL_FAILED:
            raise Error(self.error)
        if self.state != CALL_DONE:
            raise Error(self.what + ": reply() before the call finished")
        return ApiReply(self._reply_body.copy(), self.latency_ms, self.attempts)

    def cancel(mut self, mut http: HttpClient) raises:
        if self.state == CALL_RUNNING:
            http.cancel()
        if self.active():
            self.state = CALL_FAILED
            self.error = self.what + ": cancelled"


def warm_up(mut http: HttpClient, var url: String, timeout_ms: Int = 10000):
    """Open (and keep) the connection to `url`'s host before a real-time loop.

    ⚠ THE FIRST CALL ON A CLIENT PAYS THE TLS HANDSHAKE INSIDE ONE POLL —
    certificate-chain verification is CPU work, measured at 8-42 ms in a
    single `poll(0)` (the HF router and TypeSafe, from a Mac), i.e. one or
    two dropped frames. Later polls on the kept-alive connection cost
    < 0.5 ms. A `HEAD` here moves that cost before the loop.

    ⚠ IT MUST GO THROUGH THE ASYNC PATH. A blocking `request()` runs on
    libcurl's easy interface, whose connection cache is NOT the one the
    handle's multi uses — the first version warmed that cache, and the
    first real poll still paid 5-7 ms. `start` + `poll` puts the connection
    where `ApiCall` will look for it.

    The status is ignored (a 404 or 405 still leaves the connection open),
    and so is failure: warming is an optimisation, the real call will report
    a real problem.
    """
    try:
        http.start(String("HEAD"), url^)
        var waited = 0
        while not http.poll(100):
            waited += 100
            if waited >= timeout_ms:
                http.cancel()
                return
        _ = http.finish()
    except:
        try:
            http.cancel()
        except:
            pass


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
    since a bare URL does not say which demo step failed. Blocking: it is an
    `ApiCall` polled to the end, so the two paths share one retry policy.
    """
    var call = ApiCall()
    call.begin(http, url^, body.copy(), content_type^, what^, retries, base_delay_s)
    return call.wait(http)


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
