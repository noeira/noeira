# +--------------------------------------------------------------------------+ #
# | A model in this process, on a thread of its own, polled like an API call
# +--------------------------------------------------------------------------+ #
"""What lets a model written in Mojo stand where an HTTP client stood: the same
`start` / `poll` / `result` / `cancel`, with the model on a worker thread
instead of behind a socket.

    struct MyModel(LocalModel):            # built empty on the caller's thread
        def load(mut self) raises: ...     # on the WORKER: weights, device, ...
        def warm(mut self) raises: ...     # one untimed call
        def serve(mut self, ref req: List[UInt8]) raises -> List[UInt8]: ...

    var call = LocalCall(req_slot=4096, rep_slot=4 << 20)
    var bg = BackgroundThread(LocalWorker(MyModel(...), call))
    call.begin(request_bytes)              # never blocks
    while not call.poll(): ...             # once a frame
    var reply = call.take(String("my model"))   # raises the model's error

## Why a thread and not a call

A real-time caller (`G1VoiceLoopOf`) polls once a frame and must never block:
Kokoro takes 125-260 ms per reply on the Orin and Parakeet 60-90 ms per
command — several frames each. So the model runs where a server would have
run, but in-process: no HTTP, no second program, no Python.

## The contract

* Only BYTES cross (`SharedRing`, as everywhere in `noeira.core.concurrent`).
  A request and a reply are framed: `[id i64][status i64][latency us i64]`
  then the payload. The model sees and returns only the payload.
* `load` runs on the worker thread, so the model's `DeviceContext`, its
  buffers and anything thread-bound belong to that thread. Its outcome is a
  cell: `SVC_LOADING`, `SVC_READY`, `SVC_FAILED` (`LocalCall.state`). A failed
  load answers every request with the load's error, so a caller's `result`
  says WHY rather than timing out.
* One call in flight. `cancel` forgets the call's id; the model cannot be
  interrupted mid-forward, so its reply still arrives and is dropped as stale.
  A `begin` after a `cancel` therefore waits behind the abandoned call.

⚠ CRASH ISOLATION IS GONE. A fault inside the model (a GPU ILLEGAL_ADDRESS, an
abort) takes the whole process down, where a server would only have dropped a
connection. Do not put a model here in the process that runs a robot's
balance controller.
"""

from std.time import perf_counter_ns

from noeira.core.concurrent.block import SharedBlock
from noeira.core.concurrent.ring import SharedRing
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import (
    BackgroundWorker, WorkerCtl, POLL_DID_WORK, POLL_IDLE,
)


comptime SVC_LOADING: Int = 0
comptime SVC_READY: Int = 1
comptime SVC_FAILED: Int = 2

comptime _CELL_STATE: Int = 0
comptime _CELL_LOAD_MS: Int = 1
comptime _CELLS: Int = 8

comptime _HEADER: Int = 24
comptime _OK: Int64 = 0
comptime _ERR: Int64 = 1


trait LocalModel(Movable & Deinitable):
    """A model served by a `LocalWorker`. Built EMPTY (configuration only) on
    the caller's thread; everything heavy happens in `load`, on the worker's."""

    def load(mut self) raises:
        ...

    def warm(mut self) raises:
        """One untimed call after `load`, so the first real one is not the
        slow one (first launches, allocator growth, algorithm choice)."""
        ...

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        ...


# ── framing ───────────────────────────────────────────────────────────────

def _put_i64(d: Pointer[UInt8, MutUntrackedOrigin], at: Int, v: Int64):
    for b in range(8):
        d[unsafe_offset=at + b] = UInt8((v >> Int64(8 * b)) & 0xFF)


def _get_i64(d: Pointer[UInt8, MutUntrackedOrigin], at: Int) -> Int64:
    var v: Int64 = 0
    for b in range(8):
        v |= Int64(d[unsafe_offset=at + b]) << Int64(8 * b)
    return v


@fieldwise_init
struct SvcMsg(Movable):
    var id: Int64
    var status: Int64
    var latency_us: Int64
    var payload: List[UInt8]


def svc_push(
    ring: SharedRing, id: Int64, status: Int64, latency_us: Int64, ref payload: List[UInt8],
) -> Bool:
    """Frame and push; False when the ring is full or the message too big."""
    var n = _HEADER + len(payload)
    if n > ring.slot_bytes():
        return False
    var c = ring.begin_push()
    if not c.ok():
        return False
    var d = c.data()
    _put_i64(d, 0, id)
    _put_i64(d, 8, status)
    _put_i64(d, 16, latency_us)
    for i in range(len(payload)):
        d[unsafe_offset=_HEADER + i] = payload[i]
    ring.end_push(n)
    return True


def svc_pop(ring: SharedRing) -> Optional[SvcMsg]:
    var c = ring.begin_pop()
    if not c.ok():
        return None
    var d = c.data()
    var n = c.len - _HEADER
    var payload = List[UInt8](capacity=max(n, 0))
    for i in range(n):
        payload.append(d[unsafe_offset=_HEADER + i])
    var m = SvcMsg(_get_i64(d, 0), _get_i64(d, 8), _get_i64(d, 16), payload^)
    ring.end_pop()
    return m^


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8](capacity=s.byte_length())
    for x in s.as_bytes():
        b.append(x)
    return b^


def _text(ref b: List[UInt8]) -> String:
    return String(unsafe_from_utf8=Span(b))


# ── the worker side ───────────────────────────────────────────────────────

struct LocalWorker[M: LocalModel](BackgroundWorker):
    """Loads `M` on its thread, then answers requests one by one."""

    var model: Self.M
    var req: SharedRing
    var rep: SharedRing
    var cells: SharedBlock
    var error: String

    def __init__(out self, var model: Self.M, ref call: LocalCall):
        self.model = model^
        self.req = call.req
        self.rep = call.rep
        self.cells = call.cells
        self.error = String("")

    def __init__(out self, *, deinit move: Self):
        self.model = move.model^
        self.req = move.req
        self.rep = move.rep
        self.cells = move.cells
        self.error = move.error^

    def on_start(mut self, ctl: WorkerCtl):
        var t0 = perf_counter_ns()
        try:
            self.model.load()
            self.model.warm()
            self.cells.release_store(_CELL_LOAD_MS, Int64((perf_counter_ns() - t0) // 1_000_000))
            self.cells.release_store(_CELL_STATE, Int64(SVC_READY))
        except e:
            self.error = String(e)
            self.cells.release_store(_CELL_STATE, Int64(SVC_FAILED))

    def poll(mut self, ctl: WorkerCtl) -> Int:
        var m = svc_pop(self.req)
        if not m:
            return POLL_IDLE
        var id = m.value().id
        var t0 = perf_counter_ns()
        var status = _OK
        var out: List[UInt8]
        if self.error.byte_length() > 0:
            status = _ERR
            out = _bytes(String("the model failed to load: ") + self.error)
        else:
            try:
                out = self.model.serve(m.value().payload)
            except e:
                status = _ERR
                out = _bytes(String(e))
        var us = Int64((perf_counter_ns() - t0) // 1000)
        if len(out) + _HEADER > self.rep.slot_bytes():
            status = _ERR
            out = _bytes(String("a reply of ") + String(len(out)) + " bytes does not fit the "
                         + String(self.rep.slot_bytes()) + "-byte reply slot")
        # The ring holds 4: full only when the caller stopped polling. Wait up
        # to a second for it, then drop the reply (the caller has moved on).
        for _ in range(1000):
            if svc_push(self.rep, id, status, us, out) or ctl.should_stop():
                break
            _ = sleep_us(1000)
        return POLL_DID_WORK

    def on_stop(mut self, ctl: WorkerCtl):
        pass


# ── the caller side ───────────────────────────────────────────────────────

struct LocalCall(Movable):
    """The caller's end: one request ring, one reply ring, the state cells."""

    var req: SharedRing
    var rep: SharedRing
    var cells: SharedBlock
    var _next: Int64
    var _cur: Int64
    """The call in flight, -1 for none."""
    var _done: Bool
    var _status: Int64
    var _latency_us: Int64
    var _payload: List[UInt8]

    def __init__(out self, req_slot: Int, rep_slot: Int) raises:
        self.req = SharedRing(2, req_slot)
        self.rep = SharedRing(4, rep_slot)
        self.cells = SharedBlock(_CELLS)
        self._next = 0
        self._cur = -1
        self._done = False
        self._status = _OK
        self._latency_us = 0
        self._payload = List[UInt8]()

    def state(self) -> Int:
        return Int(self.cells.acquire_load(_CELL_STATE))

    def load_ms(self) -> Int:
        """Load + warm-up time, once `state()` is `SVC_READY`."""
        return Int(self.cells.acquire_load(_CELL_LOAD_MS))

    def begin(mut self, ref payload: List[UInt8], what: String) raises:
        """Queue a request. Never blocks: raises if it cannot be queued."""
        self._next += 1
        if not svc_push(self.req, self._next, _OK, 0, payload):
            raise Error(what + ": could not queue the request (" + String(len(payload))
                        + " bytes; the queue holds 2 of at most " + String(self.req.slot_bytes()) + ")")
        self._cur = self._next
        self._done = False
        self._payload = List[UInt8]()

    def poll(mut self) -> Bool:
        """True once the call is over — succeeded OR failed; `take` says which.
        Also True when there is no call, so a caller's loop cannot hang."""
        if self._cur < 0 or self._done:
            return True
        while True:
            var m = svc_pop(self.rep)
            if not m:
                return False
            if m.value().id != self._cur:
                continue  # a cancelled call's reply
            self._status = m.value().status
            self._latency_us = m.value().latency_us
            self._payload = m.value().payload.copy()
            self._done = True
            return True

    def take(mut self, what: String) raises -> List[UInt8]:
        """The reply's payload, or the model's error raised."""
        if self._cur < 0:
            raise Error(what + ": no call in flight")
        if not self._done:
            raise Error(what + ": the call has not finished (poll first)")
        self._cur = -1
        self._done = False
        if self._status != _OK:
            raise Error(what + ": " + _text(self._payload))
        var p = self._payload^
        self._payload = List[UInt8]()
        return p^

    def latency_ms(self) -> Float64:
        """The model's own time for the last call (not the queueing)."""
        return Float64(self._latency_us) / 1000.0

    def cancel(mut self):
        self._cur = -1
        self._done = False
