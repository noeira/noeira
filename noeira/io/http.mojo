# +--------------------------------------------------------------------------+ #
# | HTTP over libcurl — the last of Python out of the data path
# +--------------------------------------------------------------------------+ #
"""An HTTP client, without Python and without a subprocess.

    pixi run build-http          # ONCE, builds the C shim this binds to

    var c = HttpClient()
    c.bearer(api_key)
    var r = c.post_json(url, body)
    if not r.ok():
        raise Error(r.text())

    _ = http_download(url, "cache/x.h5", expect_size=n, label="dataset")

    c.start("POST", url, body, "application/json")   # async: returns at once
    while not c.poll(0):                              # advance; never blocks
        consume(c.read_new())                         # bytes as they arrive
    var r = c.finish()

Every net call in `noeira` used to go through Python `urllib` or a `curl`
subprocess. `urllib` is the more expensive of the two, and not because of
throughput: `RemoteLogger.flush` was the reason a *training binary* had to
find a CPython at all (see `pixi.toml`'s note on pinning
`libpython3.13.{so,dylib}` for exactly that call). This module removes the
last reason.

## Why libcurl rather than a socket stack

TLS, certificate verification, redirect chains, `Range` resume, chunked
transfer, `Expect: 100-continue`, proxies — every one is a place to be subtly
wrong, and none of them is the project's subject. `libcurl` is already in the
environment (`pixi.toml` declares `curl`; the package ships `libcurl.4`), so
this adds ZERO dependencies. It also brings HTTP/2 for free, since the env's
build links `nghttp2`.

Flare (`references/flare-main/`) was evaluated as the alternative and is a
good library, but `from flare.http import ...` pulls **187 modules / 78k
lines** — HTTP/2, HTTP/3, QUIC, a kqueue/epoll/io_uring reactor — into the
compile for what this tree actually asks for: four call sites, all clients.

## Why a C shim rather than direct FFI

`curl_easy_setopt` is C-variadic and Mojo's `external_call` emits a fixed
prototype — the `ioctl` trap `noeira/io/serial/native.mojo` documents at
length, whose symptom is a silent runtime failure rather than a compile error.
`native/nra_http.c` presents a fixed-arity API instead, and (having crossed
the boundary once) also owns the write callback and the progress line.

⚠ REQUIRES A BUILT SHIM. `pixi run build-http` produces `libnra_http.dylib`
beside this file; it is not tracked in git, and a missing one is a dlopen
ABORT at the first call rather than a compile error. `http_shim_available()`
answers the question without touching FFI.

⚠ AN HTTP STATUS IS NOT AN ERROR HERE. A clean 404 is a successful transfer:
`request()` returns it and the caller decides. That is deliberate — `hf.mojo`
treats a 404 as fatal, `fetch_to_cache` treats a 416 as "already complete",
and `remote.mojo` needs the SERVER'S BODY to tell three different auth
failures apart. Only `expect=` opts into raising.
"""

from std.ffi import OwnedDLHandle, _Global, _get_dylib_function, c_char
from std.os import abort, getenv
from std.pathlib import Path
from std.sys import CompilationTarget

comptime Ptr = Pointer


def untracked[
    T: AnyType, o: Origin
](p: Pointer[T, o]) -> Pointer[T, MutUntrackedOrigin]:
    """Re-key a pointer's origin for an FFI call — the `vision/opencv` helper,
    duplicated for the same reason it was: three lines are cheaper than a
    dependency between packages."""
    return rebind[Pointer[T, MutUntrackedOrigin]](p)


# ═══════════════════════════════════════════════════════════════════════════
# dylib loading — the `io/serial`, `render/imgui`, `vision/opencv` pattern
# ═══════════════════════════════════════════════════════════════════════════


def _lib_name() -> String:
    comptime if CompilationTarget.is_macos():
        return String("libnra_http.dylib")
    elif CompilationTarget.is_linux():
        return String("libnra_http.so")
    else:
        comptime assert False, "OS is not supported"


def _candidates() -> List[String]:
    """Where to look, most explicit first. Shared with the availability probe
    so the two cannot answer different questions."""
    var name = _lib_name()
    var out = List[String]()
    var override = getenv("NOEIRA_HTTP_LIB")
    if override.byte_length() > 0:
        out.append(override)
    var root = getenv("PIXI_PROJECT_ROOT")
    if root.byte_length() > 0:
        out.append(root + "/noeira/io/" + name)
    out.append("noeira/io/" + name)
    out.append(name)
    return out^


def http_shim_available() -> Bool:
    """True when the shim can be found WITHOUT dlopening it.

    `_Global` aborts the process on a missing library, which is right for a
    hard dependency and the wrong first impression here — a training run whose
    only net use is an optional dashboard should say what to build, not die in
    the loader.
    """
    var c = _candidates()
    for i in range(len(c)):
        if Path(c[i]).exists():
            return True
    return False


def _init_handle() -> OwnedDLHandle:
    # ⚠ A FILE THAT EXISTS CAN STILL FAIL TO LOAD — wrong architecture (a
    # `.so` copied from an x86 box), or a `libcurl.so.4` its rpath no longer
    # reaches. Keep dlopen's reason per candidate; "not found" alone sent a
    # Jetson user looking for a file that was there.
    var c = _candidates()
    var tried = String("")
    for i in range(len(c)):
        try:
            return OwnedDLHandle(c[i])
        except e:
            tried += "\n  - " + c[i]
            if Path(c[i]).exists():
                tried += "\n      exists but failed to load: " + String(e)
    abort(
        "http shim not found or not loadable. Tried:"
        + tried
        + "\nBuild it with `pixi run build-http`, or set"
        + " NOEIRA_HTTP_LIB=/path/to/"
        + _lib_name()
    )


comptime lib = _Global["NOEIRA_HTTP", _init_handle]()


def curl_version() raises -> String:
    """The libcurl the shim was LINKED against, TLS backend included.

    Worth printing in a bug report: which TLS library and which HTTP versions
    are compiled in is the first thing a connection failure depends on.
    """
    var p = _get_dylib_function[
        lib, "nra_http_curl_version", def () thin -> Ptr[c_char, MutUntrackedOrigin]
    ]()()
    return String(unsafe_from_utf8_ptr=p)


# ═══════════════════════════════════════════════════════════════════════════
# Response
# ═══════════════════════════════════════════════════════════════════════════


struct HttpResponse(Movable):
    """A finished HTTP response. `body` is the authoritative payload."""

    var status: Int
    var body: List[UInt8]

    def __init__(out self, status: Int, var body: List[UInt8]):
        self.status = status
        self.body = body^

    def __init__(out self, *, deinit move: Self):
        self.status = move.status
        self.body = move.body^

    def ok(self) -> Bool:
        """2xx. A redirect never reaches here — libcurl follows them."""
        return self.status >= 200 and self.status < 300

    def take_body(deinit self) -> List[UInt8]:
        """Consume the response and hand back its bytes without a copy.

        ⚠ `r.body^` at a call site does NOT compile: moving one field out of a
        live struct is "destroyed out of the middle of a value". A consuming
        method is the sanctioned way to get the bytes for free.
        """
        return self.body^

    def text(self) raises -> String:
        """The body as text.

        ⚠ STOPS AT AN EMBEDDED NUL. JSON and error pages do not carry one;
        anything binary must read `body` instead.
        """
        if len(self.body) == 0:
            return String("")
        var b = self.body.copy()
        b.append(0)
        return String(unsafe_from_utf8_ptr=b.unsafe_ptr())


# ═══════════════════════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════════════════════


struct HttpClient(Movable & Deinitable):
    """One libcurl easy handle, reused across requests.

    ⚠ REUSE IS THE POINT. The handle carries the live connection, the TLS
    session cache and the DNS cache, so a `RemoteLogger` that keeps one client
    pays for the TLS handshake once per run instead of once per flush. A
    client per request is correct but wasteful; a client per *thread* is the
    rule — a libcurl easy handle must not be shared across threads.
    """

    var _h: Int
    """The `nra_http*` as an address. 0 once closed."""
    var _hdr_names: List[String]
    var _hdr_values: List[String]
    """Persistent headers, re-applied per request because the C side clears
    its `curl_slist` on every reset."""
    var user_agent: String

    def __init__(
        out self,
        timeout_ms: Int = 30000,
        connect_timeout_ms: Int = 10000,
        user_agent: String = String("noeira/1.0"),
    ) raises:
        self._h = _get_dylib_function[lib, "nra_http_new", def () thin -> Int]()()
        if self._h == 0:
            raise Error("http: curl_easy_init failed")
        self._hdr_names = List[String]()
        self._hdr_values = List[String]()
        self.user_agent = user_agent
        _get_dylib_function[
            lib, "nra_http_set_timeout_ms", def (Int, Int, Int) thin -> None
        ]()(self._h, timeout_ms, connect_timeout_ms)
        # The env's CA bundle, when there is one. libcurl's compiled-in
        # default already points here, but a binary run OUTSIDE pixi keeps
        # working only if we say so explicitly.
        var prefix = getenv("CONDA_PREFIX")
        if prefix.byte_length() > 0:
            var ca = prefix + "/ssl/cacert.pem"
            if Path(ca).exists():
                # `as_c_string_slice` mutates (it appends the NUL), so every
                # string handed to FFI in this file must be a local `var`.
                _ = _get_dylib_function[
                    lib,
                    "nra_http_set_cainfo",
                    def (Int, Ptr[c_char, MutUntrackedOrigin]) thin -> Int32,
                ]()(self._h, untracked(ca.as_c_string_span().ptr()))

    def __init__(out self, *, deinit move: Self):
        self._h = move._h
        self._hdr_names = move._hdr_names^
        self._hdr_values = move._hdr_values^
        self.user_agent = move.user_agent^

    def __deinit__(deinit self):
        if self._h == 0:
            return
        try:
            _get_dylib_function[lib, "nra_http_free", def (Int) thin -> None]()(
                self._h
            )
        except:
            pass  # unreportable here; the handle leaks rather than the process dying

    # ── configuration ─────────────────────────────────────────────────

    def header(mut self, name: String, value: String) raises:
        """Set a header sent with every subsequent request. Replaces a
        previous value for the same name."""
        for i in range(len(self._hdr_names)):
            if self._hdr_names[i] == name:
                self._hdr_values[i] = value
                return
        self._hdr_names.append(name)
        self._hdr_values.append(value)

    def bearer(mut self, api_key: String) raises:
        """`Authorization: Bearer <key>`.

        ⚠ An EMPTY key produces a bare `Bearer `, which a server reports the
        same way as a wrong key. Callers that read a key from the environment
        should check it before it gets here — `RemoteCatalog.from_env` does.
        """
        self.header(String("Authorization"), "Bearer " + api_key)

    def timeout_ms(mut self, total: Int, connect: Int = 10000) raises:
        """Total and connect timeouts. `total <= 0` disables the total one —
        which is what a multi-GB download wants, since a total timeout large
        enough for the slowest acceptable link still kills a healthy transfer
        of a bigger file. Use `stall_guard` there instead."""
        _get_dylib_function[
            lib, "nra_http_set_timeout_ms", def (Int, Int, Int) thin -> None
        ]()(self._h, total, connect)

    def stall_guard(mut self, bytes_per_s: Int = 1024, seconds: Int = 60) raises:
        """Abort when throughput stays under `bytes_per_s` for `seconds`."""
        _get_dylib_function[
            lib, "nra_http_set_low_speed", def (Int, Int, Int) thin -> None
        ]()(self._h, bytes_per_s, seconds)

    def max_body(mut self, bytes: Int) raises:
        """Cap on an in-memory response. Default 64 MiB; a bigger answer fails
        rather than growing the heap without bound."""
        _get_dylib_function[
            lib, "nra_http_set_max_body", def (Int, Int64) thin -> None
        ]()(self._h, Int64(bytes))

    # ── the one call everything else goes through ─────────────────────

    def _reset(mut self) raises:
        if self.running():
            raise Error(
                "http: a request is already in flight on this client — poll it"
                " to the end or cancel() it first"
            )
        _get_dylib_function[lib, "nra_http_reset", def (Int) thin -> None]()(
            self._h
        )
        var ua = "User-Agent: " + self.user_agent
        _ = self._add_header(ua)
        for i in range(len(self._hdr_names)):
            _ = self._add_header(
                self._hdr_names[i] + ": " + self._hdr_values[i]
            )

    def _add_header(mut self, var line: String) raises -> Int32:
        """⚠ EVERY STRING THAT REACHES FFI IN THIS FILE IS `var`-OWNED.
        A borrowed parameter bound to a caller's TEMPORARY —
        `gunzip_file(String(DIR) + "/x.gz", out)` — can be destroyed before
        the callee runs (Mojo destroys at last use), and the C side then reads
        freed memory. It fails as a bare `fopen` returning NULL, from a path
        the error message prints CORRECTLY, and it reproduces only in the
        caller's context: the same call in isolation reads the freed bytes
        and works. Taking ownership transfers the temporary into the call and
        keeps it alive."""
        var st = _get_dylib_function[
            lib,
            "nra_http_add_header",
            def (Int, Ptr[c_char, MutUntrackedOrigin]) thin -> Int32,
        ]()(self._h, untracked(line.as_c_string_span().ptr()))
        if st != 0:
            raise Error("http: cannot add header: " + line)
        return st

    def _error(self) raises -> String:
        var p = _get_dylib_function[
            lib, "nra_http_error", def (Int) thin -> Ptr[c_char, MutUntrackedOrigin]
        ]()(self._h)
        return String(unsafe_from_utf8_ptr=p)

    def _perform(mut self, var method: String, var url: String) raises -> Int:
        """Run the configured request. Returns the HTTP status.

        Raises only on a TRANSPORT failure — DNS, TLS, a dead connection, a
        write that could not land. The status comes back as a value.
        """
        var rc = _get_dylib_function[
            lib,
            "nra_http_perform",
            def (
                Int,
                Ptr[c_char, MutUntrackedOrigin],
                Ptr[c_char, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            self._h,
            untracked(method.as_c_string_span().ptr()),
            untracked(url.as_c_string_span().ptr()),
        )
        if rc != 0:
            raise Error(
                "http: " + method + " " + url + " failed: " + self._error()
                + " (curl " + String(rc) + ")"
            )
        return _get_dylib_function[lib, "nra_http_status", def (Int) thin -> Int]()(
            self._h
        )

    def _take_body(mut self) raises -> List[UInt8]:
        var n = _get_dylib_function[
            lib, "nra_http_body_len", def (Int) thin -> Int
        ]()(self._h)
        var out = List[UInt8]()
        if n <= 0:
            return out^
        out.resize(n, 0)
        var got = _get_dylib_function[
            lib,
            "nra_http_body_copy",
            def (Int, Ptr[UInt8, MutUntrackedOrigin], Int) thin -> Int,
        ]()(self._h, untracked(Ptr(to=out[0])), n)
        if got != n:
            raise Error(
                "http: body copy returned " + String(got) + " of " + String(n)
            )
        return out^

    def request(
        mut self,
        var method: String,
        var url: String,
        body: List[UInt8] = List[UInt8](),
        var content_type: String = String(""),
        expect: Int = -1,
    ) raises -> HttpResponse:
        """One request, with the response read into memory.

        `expect` raises when the status differs, and the message carries the
        SERVER'S BODY — the monitor answers `{"error":"Unauthorized"}`,
        `{"error":"Missing API key"}` and `{"error":"Invalid API key"}` to
        three different faults, so dropping the body would drop the diagnosis.
        """
        self._reset()
        if content_type.byte_length() > 0:
            _ = self._add_header("Content-Type: " + content_type)
        if len(body) > 0:
            var st = _get_dylib_function[
                lib,
                "nra_http_set_body",
                def (Int, Ptr[UInt8, MutUntrackedOrigin], Int) thin -> Int32,
            ]()(self._h, untracked(Ptr(to=body[0])), len(body))
            if st != 0:
                raise Error("http: cannot buffer a " + String(len(body)) + " byte body")
        var status = self._perform(method, url)
        var resp = HttpResponse(status, self._take_body())
        if expect >= 0 and status != expect:
            raise Error(
                method + " " + url + " -> " + String(status) + " (expected "
                + String(expect) + "): " + resp.text()
            )
        return resp^

    # ── async: streaming and background requests ──────────────────────
    #
    #     c.start("POST", url, body, "application/json")
    #     while not c.poll(0):          # never blocks; call once per frame
    #         var chunk = c.read_new()  # body bytes that arrived since last call
    #         draw()
    #     var r = c.finish()            # the whole response, as request() gives
    #
    # ⚠ NOTHING HAPPENS BETWEEN POLLS. There is no thread: the transfer only
    # advances inside `poll`, so a loop that stops polling stalls the request
    # (and, past the server's patience, times it out). `poll(timeout_ms)` with
    # a positive timeout waits for socket activity, which is how a blocking
    # caller spins without burning a core.

    def start(
        mut self,
        var method: String,
        var url: String,
        body: List[UInt8] = List[UInt8](),
        var content_type: String = String(""),
    ) raises:
        """Launch a request and return at once. Same arguments as `request`."""
        self._reset()
        if content_type.byte_length() > 0:
            _ = self._add_header("Content-Type: " + content_type)
        if len(body) > 0:
            var st = _get_dylib_function[
                lib,
                "nra_http_set_body",
                def (Int, Ptr[UInt8, MutUntrackedOrigin], Int) thin -> Int32,
            ]()(self._h, untracked(Ptr(to=body[0])), len(body))
            if st != 0:
                raise Error("http: cannot buffer a " + String(len(body)) + " byte body")
        var rc = _get_dylib_function[
            lib,
            "nra_http_start",
            def (
                Int,
                Ptr[c_char, MutUntrackedOrigin],
                Ptr[c_char, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            self._h,
            untracked(method.as_c_string_span().ptr()),
            untracked(url.as_c_string_span().ptr()),
        )
        if rc != 0:
            raise Error("http: cannot start " + method + " " + url + " (" + String(rc) + ")")

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        """Advance the in-flight request. True once it has finished (or when
        nothing is in flight); then call `finish`."""
        var r = _get_dylib_function[lib, "nra_http_poll", def (Int, Int32) thin -> Int32]()(
            self._h, Int32(timeout_ms)
        )
        if r < 0:
            raise Error("http: poll failed")
        return r == 1

    def running(self) -> Bool:
        try:
            return (
                _get_dylib_function[lib, "nra_http_running", def (Int) thin -> Int32]()(
                    self._h
                )
                != 0
            )
        except:
            return False

    def status(self) raises -> Int:
        """The response status so far — 0 until the headers arrive."""
        return _get_dylib_function[lib, "nra_http_status", def (Int) thin -> Int]()(
            self._h
        )

    def read_new(mut self) raises -> List[UInt8]:
        """Body bytes that arrived since the previous call (possibly none).
        Works during the transfer and after it, until the next request."""
        var n = _get_dylib_function[lib, "nra_http_unread_len", def (Int) thin -> Int]()(
            self._h
        )
        var out = List[UInt8]()
        if n <= 0:
            return out^
        out.resize(n, 0)
        var got = _get_dylib_function[
            lib,
            "nra_http_read_new",
            def (Int, Ptr[UInt8, MutUntrackedOrigin], Int) thin -> Int,
        ]()(self._h, untracked(Ptr(to=out[0])), n)
        out.resize(got, 0)
        return out^

    def finish(mut self) raises -> HttpResponse:
        """The finished response. Raises on a TRANSPORT failure (like
        `request`), or when the request has not finished yet."""
        if self.running():
            raise Error("http: finish() while the request is still running")
        var rc = _get_dylib_function[
            lib, "nra_http_async_result", def (Int) thin -> Int32
        ]()(self._h)
        if rc != 0:
            raise Error("http: async request failed: " + self._error() + " (curl " + String(rc) + ")")
        return HttpResponse(self.status(), self._take_body())

    def cancel(mut self) raises:
        """Abandon the in-flight request (no-op when none). The connection is
        dropped; the next request opens a new one."""
        _get_dylib_function[lib, "nra_http_cancel", def (Int) thin -> None]()(self._h)

    def get(mut self, var url: String, expect: Int = -1) raises -> HttpResponse:
        return self.request(String("GET"), url, expect=expect)

    def post_json(
        mut self, var url: String, var json_body: String, expect: Int = -1
    ) raises -> HttpResponse:
        var b = List[UInt8]()
        for i in range(json_body.byte_length()):
            b.append(json_body.as_bytes()[i])
        return self.request(
            String("POST"),
            url,
            b^,
            String("application/json"),
            expect,
        )

    # ── transfers ─────────────────────────────────────────────────────

    def download(
        mut self,
        var url: String,
        var dest: String,
        resume_from: Int = 0,
        var label: String = String(""),
    ) raises -> HttpResponse:
        """GET `url` into `dest`, appending when `resume_from > 0`.

        The file is opened ONLY for a successful response, so a 404's body
        lands in the returned `HttpResponse` instead of on top of a partial
        download. A server that ignores `Range` and answers 200 raises rather
        than appending a whole file onto a prefix — see `fetch_to_cache`,
        which catches that and restarts from zero.

        ⚠ Content encoding is forced OFF for a file transfer. With it on,
        `Content-Length` describes the compressed stream while the file holds
        the decompressed one, and every caller here counts bytes.
        """
        self._reset()
        var st = _get_dylib_function[
            lib,
            "nra_http_set_out_file",
            def (Int, Ptr[c_char, MutUntrackedOrigin]) thin -> Int32,
        ]()(self._h, untracked(dest.as_c_string_span().ptr()))
        if st != 0:
            raise Error("http: cannot set the output file " + dest)
        if resume_from > 0:
            _get_dylib_function[
                lib, "nra_http_set_resume_from", def (Int, Int64) thin -> None
            ]()(self._h, Int64(resume_from))
        if label.byte_length() > 0:
            _get_dylib_function[
                lib,
                "nra_http_set_progress",
                def (Int, Int32, Ptr[c_char, MutUntrackedOrigin]) thin -> None,
            ]()(
                self._h,
                Int32(1),
                untracked(label.as_c_string_span().ptr()),
            )
        var status = self._perform(String("GET"), url)
        return HttpResponse(status, self._take_body())

    def upload(
        mut self,
        var url: String,
        var path: String,
        var method: String = String("PUT"),
        var label: String = String(""),
    ) raises -> HttpResponse:
        """Stream a local file as the request body."""
        self._reset()
        var st = _get_dylib_function[
            lib,
            "nra_http_set_upload_file",
            def (Int, Ptr[c_char, MutUntrackedOrigin]) thin -> Int32,
        ]()(self._h, untracked(path.as_c_string_span().ptr()))
        if st != 0:
            raise Error("http: cannot open " + path + " for upload")
        if label.byte_length() > 0:
            _get_dylib_function[
                lib,
                "nra_http_set_progress",
                def (Int, Int32, Ptr[c_char, MutUntrackedOrigin]) thin -> None,
            ]()(
                self._h,
                Int32(1),
                untracked(label.as_c_string_span().ptr()),
            )
        var status = self._perform(method, url)
        return HttpResponse(status, self._take_body())

    def zstd_to_file(mut self, on: Bool = True) raises:
        """Decode the body through zstd on its way to the output file.

        ⚠ THE DECOMPRESSOR SURVIVES A RETRY, and that is the point. A 13 GB
        `.zst` streamed into a 47 GB `.h5` must never land the compressed file
        on disk, so a dropped connection has to resume the DOWNLOAD at a
        compressed offset while the DECOMPRESSOR continues from where it
        stopped. `zstd_read()` is the offset to pass as `resume_from` — bytes
        FULLY fed to the decompressor, so it can never point into a
        half-consumed frame.

        ⚠ The output file must be OPENED IN APPEND MODE on that retry, which
        `download(resume_from > 0)` already does. Turning this off frees the
        stream; the next `zstd_to_file(True)` starts a fresh one.
        """
        var st = _get_dylib_function[
            lib, "nra_http_zstd_enable", def (Int, Int32) thin -> Int32
        ]()(self._h, Int32(1) if on else Int32(0))
        if st != 0:
            raise Error("http: cannot start a zstd stream")

    def zstd_read(self) raises -> Int:
        """Compressed bytes fully consumed — the offset a retry resumes at."""
        return Int(
            _get_dylib_function[
                lib, "nra_http_zstd_read", def (Int) thin -> Int64
            ]()(self._h)
        )

    def range_ignored(self) raises -> Bool:
        """True when the last transfer asked for a `Range` and the server
        answered 200. The bytes were NOT written — restart from zero."""
        return (
            _get_dylib_function[
                lib, "nra_http_range_ignored", def (Int) thin -> Int32
            ]()(self._h)
            != 0
        )

    def content_length(self) raises -> Int:
        """`Content-Length` of the last response, or -1 when it had none
        (a chunked reply). For a resumed transfer this is the REMAINDER."""
        return Int(
            _get_dylib_function[
                lib, "nra_http_content_length", def (Int) thin -> Int64
            ]()(self._h)
        )


# ═══════════════════════════════════════════════════════════════════════════
# gzip
# ═══════════════════════════════════════════════════════════════════════════


def gunzip(ref data: List[UInt8]) raises -> List[UInt8]:
    """Decompress a gzip (or zlib) stream.

    ⚠ HERE BECAUSE IT IS A CONTENT ENCODING, not because it is I/O in general.
    libcurl decodes `Content-Encoding: gzip` on its own; what it will not do
    is decode a body whose compression is announced by a `.gz` in the URL —
    which is how MNIST ships. zlib is already linked into the shim, so this is
    two C entry points rather than a second dependency.

    ⚠ SIZED FROM THE GZIP TRAILER, whose `ISIZE` is the uncompressed length
    MOD 2^32. Exact below 4 GiB and wrong above it, so anything larger has to
    stream — this raises rather than returning a truncated buffer.
    """
    if len(data) == 0:
        return List[UInt8]()
    var want = _get_dylib_function[
        lib, "nra_gzip_isize", def (Ptr[UInt8, MutUntrackedOrigin], Int) thin -> Int
    ]()(untracked(Ptr(to=data[0])), len(data))
    if want < 0:
        raise Error("gunzip: not a gzip stream (no 0x1f 0x8b header)")
    var out = List[UInt8]()
    out.resize(want if want > 0 else 1, 0)
    var got = _get_dylib_function[
        lib,
        "nra_gzip_inflate",
        def (
            Ptr[UInt8, MutUntrackedOrigin],
            Int,
            Ptr[UInt8, MutUntrackedOrigin],
            Int,
        ) thin -> Int,
    ]()(untracked(Ptr(to=data[0])), len(data), untracked(Ptr(to=out[0])), len(out))
    if got < 0:
        raise Error(
            "gunzip: inflate failed (" + String(got) + ") — the stream is"
            " truncated, or larger than the 4 GiB the trailer can describe"
        )
    out.resize(got, 0)
    return out^


def inflate_into(ref data: List[UInt8], expected: Int) raises -> List[UInt8]:
    """Inflate a zlib OR gzip stream whose decompressed size is already known.

    ⚠ `gunzip` CANNOT DO THIS. It sizes the output from the gzip trailer's
    `ISIZE`, and a raw zlib stream — which is what a PNG's `IDAT` is — has no
    trailer to read. Here the caller knows the exact size from the image
    header, so one allocation and one call suffice.
    """
    if len(data) == 0 or expected <= 0:
        return List[UInt8]()
    var out = List[UInt8]()
    out.resize(expected, 0)
    var got = _get_dylib_function[
        lib,
        "nra_gzip_inflate",
        def (
            Ptr[UInt8, MutUntrackedOrigin],
            Int,
            Ptr[UInt8, MutUntrackedOrigin],
            Int,
        ) thin -> Int,
    ]()(
        untracked(Ptr(to=data[0])), len(data), untracked(Ptr(to=out[0])), expected
    )
    if got < 0:
        raise Error(
            "inflate_into: the stream is corrupt, or decompresses to more than"
            " the " + String(expected) + " bytes the caller expected"
        )
    out.resize(got, 0)
    return out^


def deflate(ref data: List[UInt8], level: Int = 6) raises -> List[UInt8]:
    """A zlib stream of `data` — the other direction of `inflate_into`.

    Used by `io/png.mojo`'s encoder. `IDAT` is a zlib stream, not a raw
    deflate one, which is what `compress2` produces.
    """
    if len(data) == 0:
        return List[UInt8]()
    var cap = _get_dylib_function[
        lib, "nra_zlib_compress_bound", def (Int) thin -> Int
    ]()(len(data))
    var out = List[UInt8]()
    out.resize(cap, 0)
    var n = _get_dylib_function[
        lib,
        "nra_zlib_compress",
        def (
            Ptr[UInt8, MutUntrackedOrigin],
            Int,
            Ptr[UInt8, MutUntrackedOrigin],
            Int,
            Int32,
        ) thin -> Int,
    ]()(
        untracked(Ptr(to=data[0])), len(data),
        untracked(Ptr(to=out[0])), cap, Int32(level),
    )
    if n < 0:
        raise Error("deflate: zlib refused a " + String(len(data)) + "-byte input")
    out.resize(n, 0)
    return out^


def crc32(ref data: List[UInt8], seed: Int = 0) raises -> Int:
    """CRC-32 as PNG defines it (zlib's, the same polynomial and reflection)."""
    if len(data) == 0:
        return seed
    return Int(
        _get_dylib_function[
            lib,
            "nra_crc32",
            def (Int, Ptr[UInt8, MutUntrackedOrigin], Int) thin -> Int,
        ]()(seed, untracked(Ptr(to=data[0])), len(data))
    )


def gunzip_file(var src: String, var dst: String) raises -> Int:
    """Decompress `src` to `dst`, streaming. Returns the byte count written.

    ⚠ USE THIS, NOT `gunzip`, FOR AN ARCHIVE. The in-memory form needs the
    compressed input and the decompressed output resident at once — 350 MB for
    CIFAR-10's 162 MB tarball, and unbounded in general. This holds two 1 MB
    buffers whatever the file size.
    """
    var n = _get_dylib_function[
        lib,
        "nra_gzip_inflate_file",
        def (
            Ptr[c_char, MutUntrackedOrigin], Ptr[c_char, MutUntrackedOrigin]
        ) thin -> Int64,
    ]()(
        untracked(src.as_c_string_span().ptr()),
        untracked(dst.as_c_string_span().ptr()),
    )
    if n < 0:
        var why = String("unknown")
        if n == -1: why = String("cannot open " + src)
        elif n == -2: why = String("cannot open " + dst + " for writing")
        elif n == -3: why = String("zlib refused to start")
        elif n == -4: why = String("the stream is corrupt or truncated")
        elif n == -5: why = String("a write to " + dst + " failed")
        raise Error("gunzip_file: " + why)
    return Int(n)


# ═══════════════════════════════════════════════════════════════════════════
# one-shot helpers
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠ ONE CLIENT PER CALL, so each pays a fresh TLS handshake. Right for a
# one-off; wrong in a loop, where an `HttpClient` should be kept.


def http_get_bytes(url: String, expect: Int = 200) raises -> List[UInt8]:
    var c = HttpClient()
    var r = c.get(url, expect)
    return r^.take_body()


def http_get_text(url: String, expect: Int = 200) raises -> String:
    var c = HttpClient()
    var r = c.get(url, expect)
    return r.text()
