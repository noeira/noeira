"""`.demo` — a flat file of recorded transitions, for a replay's demo prefix.

    from noeira.deep_agents.demos.file import DemoSet, write_demo_file, read_demo_file

    var demos = DemoSet(obs_dim, act_dim)
    demos.begin_episode()
    demos.add(obs, act, reward, next_obs, done, intervened=False)
    demos.end_episode(success=True)
    write_demo_file(path, demos)
    ...
    var d = read_demo_file(path)        # what `--demos` loads

## WHAT IT IS FOR

HIL-SERL (`references/hil-serl-main`, `examples/train_rlpd.py`) keeps TWO
replay buffers — the online one and a DEMO one seeded from
`record_demos.py`'s pickles and fed every human intervention — and samples
half of every batch from each (RLPD's symmetric sampling). This file is the
pickle's role here: what `examples/so101/tower_teleop_record.mojo` writes and
what `sac_family_driver`'s `--demos` reads into the replay's pinned prefix
(`StoreReplayGpu.pin_demo_prefix`).

## THE FORMAT (little-endian, version 1)

    magic       8 bytes   "MRLDEMO1"
    version     u32       1
    obs_dim     u32
    act_dim     u32
    n           u64       transitions
    n_episodes  u32
    rows        n x [obs f32*obs_dim | act f32*act_dim | rew f32 |
                     next_obs f32*obs_dim | done f32 | flags u32]
    episodes    n_episodes x [start u64 | length u64 | success u32]

`flags` bit 0 = the transition came from a human INTERVENTION while a policy
was driving (HIL-SERL's `intervene_action`); bit 1 = the episode it belongs
to succeeded. Both are per-row so a loader can filter without the table.

⚠ FLOAT32 ON DISK WHATEVER `DT` IS. The trainer's replay stores `DT`
(float32); the CPU env that records runs float64. Storing the narrower one is
what the replay would keep anyway, and it makes the file independent of the
build's `DT`.

⚠ `done` IS THE REPLAY'S `done`, NOT "the episode ended". The family driver
runs `TERMINATE_ON_UNHEALTHY=False`, so an online transition's `done` is 0
throughout (truncation is not a terminal, see `driver_offpolicy`). A recorder
writes the same convention — 0 unless the env itself terminated — so a demo
row bootstraps exactly as its online twin would. The episode boundary lives
in the episodes table, where the success flag is.
"""

from noeira.io.fileio import write_file_atomic
from std.memory import bitcast
from std.os.path import exists


comptime DEMO_MAGIC: StaticString = "MRLDEMO1"
comptime DEMO_VERSION: UInt32 = 1
comptime DEMO_FLAG_INTERVENED: UInt32 = 1
comptime DEMO_FLAG_SUCCESS: UInt32 = 2


struct DemoSet(Movable):
    """Transitions plus an episode table, all host-side."""

    var obs_dim: Int
    var act_dim: Int
    var obs: List[Float32]
    var act: List[Float32]
    var rew: List[Float32]
    var nobs: List[Float32]
    var done: List[Float32]
    var flags: List[UInt32]
    var ep_start: List[Int]
    var ep_len: List[Int]
    var ep_success: List[Bool]
    var _open_start: Int
    """Row index where the episode being recorded began, -1 when none."""

    def __init__(out self, obs_dim: Int, act_dim: Int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs = List[Float32]()
        self.act = List[Float32]()
        self.rew = List[Float32]()
        self.nobs = List[Float32]()
        self.done = List[Float32]()
        self.flags = List[UInt32]()
        self.ep_start = List[Int]()
        self.ep_len = List[Int]()
        self.ep_success = List[Bool]()
        self._open_start = -1

    def count(self) -> Int:
        return len(self.rew)

    def n_episodes(self) -> Int:
        return len(self.ep_start)

    def n_successes(self) -> Int:
        var n = 0
        for i in range(len(self.ep_success)):
            if self.ep_success[i]:
                n += 1
        return n

    def n_intervened(self) -> Int:
        var n = 0
        for i in range(len(self.flags)):
            if (self.flags[i] & DEMO_FLAG_INTERVENED) != 0:
                n += 1
        return n

    # ── recording ────────────────────────────────────────────────────────

    def begin_episode(mut self):
        """Start an episode at the current row. An episode already open is
        DISCARDED (its rows dropped), so a recorder that resets mid-episode
        loses only that attempt."""
        if self._open_start >= 0:
            self.discard_episode()
        self._open_start = self.count()

    def open_rows(self) -> Int:
        """Rows recorded in the episode currently open (0 when none)."""
        if self._open_start < 0:
            return 0
        return self.count() - self._open_start

    def add[T: DType, TA: DType](
        mut self,
        ref obs: List[Scalar[T]],
        ref act: List[Scalar[TA]],
        reward: Float64,
        ref next_obs: List[Scalar[T]],
        done: Float64,
        intervened: Bool = False,
    ) raises:
        if len(obs) != self.obs_dim or len(next_obs) != self.obs_dim:
            raise Error(
                "DemoSet.add: obs has " + String(len(obs)) + " / "
                + String(len(next_obs)) + " words, the set is "
                + String(self.obs_dim) + " wide"
            )
        if len(act) != self.act_dim:
            raise Error(
                "DemoSet.add: action has " + String(len(act))
                + " words, the set is " + String(self.act_dim) + " wide"
            )
        if self._open_start < 0:
            self._open_start = self.count()
        for i in range(self.obs_dim):
            self.obs.append(Float32(obs[i]))
        for j in range(self.act_dim):
            self.act.append(Float32(act[j]))
        self.rew.append(Float32(reward))
        for i in range(self.obs_dim):
            self.nobs.append(Float32(next_obs[i]))
        self.done.append(Float32(done))
        self.flags.append(DEMO_FLAG_INTERVENED if intervened else UInt32(0))

    def end_episode(mut self, success: Bool):
        """Close the open episode and keep it, stamping every row's success
        bit. A closed set with no open episode is a no-op."""
        if self._open_start < 0:
            return
        var n = self.count() - self._open_start
        if n == 0:
            self._open_start = -1
            return
        if success:
            for r in range(self._open_start, self.count()):
                self.flags[r] = self.flags[r] | DEMO_FLAG_SUCCESS
        self.ep_start.append(self._open_start)
        self.ep_len.append(n)
        self.ep_success.append(success)
        self._open_start = -1

    def discard_episode(mut self):
        """Drop the open episode's rows."""
        if self._open_start < 0:
            return
        var s = self._open_start
        self.obs.resize(s * self.obs_dim, Float32(0))
        self.act.resize(s * self.act_dim, Float32(0))
        self.rew.resize(s, Float32(0))
        self.nobs.resize(s * self.obs_dim, Float32(0))
        self.done.resize(s, Float32(0))
        self.flags.resize(s, UInt32(0))
        self._open_start = -1

    # ── reading one row out, in the replay's dtype ────────────────────────

    def row_obs[T: DType](self, r: Int, mut out: List[Scalar[T]]):
        for i in range(self.obs_dim):
            out[i] = Scalar[T](self.obs[r * self.obs_dim + i])

    def row_next_obs[T: DType](self, r: Int, mut out: List[Scalar[T]]):
        for i in range(self.obs_dim):
            out[i] = Scalar[T](self.nobs[r * self.obs_dim + i])

    def row_act[T: DType](self, r: Int, mut out: List[Scalar[T]]):
        for j in range(self.act_dim):
            out[j] = Scalar[T](self.act[r * self.act_dim + j])

    def row_intervened(self, r: Int) -> Bool:
        return (self.flags[r] & DEMO_FLAG_INTERVENED) != 0

    def row_success(self, r: Int) -> Bool:
        return (self.flags[r] & DEMO_FLAG_SUCCESS) != 0

    def summary(self) -> String:
        return (
            String(self.count()) + " transitions in "
            + String(self.n_episodes()) + " episodes ("
            + String(self.n_successes()) + " successful, "
            + String(self.n_intervened()) + " intervened rows), obs "
            + String(self.obs_dim) + " act " + String(self.act_dim)
        )


# ── bytes ─────────────────────────────────────────────────────────────────


def _put_u32(mut b: List[UInt8], v: UInt32):
    for k in range(4):
        b.append(UInt8((v >> UInt32(8 * k)) & UInt32(0xFF)))


def _put_u64(mut b: List[UInt8], v: UInt64):
    for k in range(8):
        b.append(UInt8((v >> UInt64(8 * k)) & UInt64(0xFF)))


def _put_f32(mut b: List[UInt8], v: Float32):
    _put_u32(b, bitcast[DType.uint32](v))


def _get_u32(b: List[UInt8], off: Int) -> UInt32:
    var v = UInt32(0)
    for k in range(4):
        v |= UInt32(b[off + k]) << UInt32(8 * k)
    return v


def _get_u64(b: List[UInt8], off: Int) -> UInt64:
    var v = UInt64(0)
    for k in range(8):
        v |= UInt64(b[off + k]) << UInt64(8 * k)
    return v


def _get_f32(b: List[UInt8], off: Int) -> Float32:
    return bitcast[DType.float32](_get_u32(b, off))


comptime _HEADER_BYTES: Int = 8 + 4 + 4 + 4 + 8 + 4


def _row_bytes(obs_dim: Int, act_dim: Int) -> Int:
    return 4 * (obs_dim + act_dim + 1 + obs_dim + 1) + 4


def write_demo_file(path: String, ref d: DemoSet) raises:
    """Write `d` (closed episodes only — an open one is not written)."""
    var n = 0
    for e in range(d.n_episodes()):
        n += d.ep_len[e]
    var b = List[UInt8](
        capacity=_HEADER_BYTES + n * _row_bytes(d.obs_dim, d.act_dim)
        + 20 * d.n_episodes()
    )
    for i in range(8):
        b.append(UInt8(ord(DEMO_MAGIC[byte=i])))
    _put_u32(b, DEMO_VERSION)
    _put_u32(b, UInt32(d.obs_dim))
    _put_u32(b, UInt32(d.act_dim))
    _put_u64(b, UInt64(n))
    _put_u32(b, UInt32(d.n_episodes()))
    # Rows are written EPISODE BY EPISODE so an open episode's rows (which
    # sit past the last closed one) never leak into the file.
    var written = 0
    for e in range(d.n_episodes()):
        for r in range(d.ep_start[e], d.ep_start[e] + d.ep_len[e]):
            for i in range(d.obs_dim):
                _put_f32(b, d.obs[r * d.obs_dim + i])
            for j in range(d.act_dim):
                _put_f32(b, d.act[r * d.act_dim + j])
            _put_f32(b, d.rew[r])
            for i in range(d.obs_dim):
                _put_f32(b, d.nobs[r * d.obs_dim + i])
            _put_f32(b, d.done[r])
            _put_u32(b, d.flags[r])
            written += 1
    # The table, re-based on the written order (episodes are contiguous and
    # in order in `d`, so starts are cumulative lengths).
    var start = 0
    for e in range(d.n_episodes()):
        _put_u64(b, UInt64(start))
        _put_u64(b, UInt64(d.ep_len[e]))
        _put_u32(b, UInt32(1) if d.ep_success[e] else UInt32(0))
        start += d.ep_len[e]
    if written != n:
        raise Error("write_demo_file: wrote " + String(written) + " rows of " + String(n))
    # Atomic and chunked: a crash mid-write must not leave a truncated
    # `.demo` where the previous good one was.
    write_file_atomic(path, b)


def read_demo_file(path: String) raises -> DemoSet:
    if not exists(path):
        raise Error("read_demo_file: no such file: " + path)
    var b: List[UInt8]
    with open(path, "r") as f:
        b = f.read_bytes()
    if len(b) < _HEADER_BYTES:
        raise Error("read_demo_file: " + path + " is too short to be a .demo")
    for i in range(8):
        if b[i] != UInt8(ord(DEMO_MAGIC[byte=i])):
            raise Error("read_demo_file: " + path + " has no MRLDEMO1 magic")
    var version = _get_u32(b, 8)
    if version != DEMO_VERSION:
        raise Error(
            "read_demo_file: version " + String(version) + " (this reader is "
            + String(DEMO_VERSION) + ")"
        )
    var obs_dim = Int(_get_u32(b, 12))
    var act_dim = Int(_get_u32(b, 16))
    var n = Int(_get_u64(b, 20))
    var n_ep = Int(_get_u32(b, 28))
    var rb = _row_bytes(obs_dim, act_dim)
    var need = _HEADER_BYTES + n * rb + 20 * n_ep
    if len(b) != need:
        raise Error(
            "read_demo_file: " + path + " is " + String(len(b))
            + " bytes; the header says " + String(need)
        )
    var d = DemoSet(obs_dim, act_dim)
    var off = _HEADER_BYTES
    for _ in range(n):
        for _ in range(obs_dim):
            d.obs.append(_get_f32(b, off))
            off += 4
        for _ in range(act_dim):
            d.act.append(_get_f32(b, off))
            off += 4
        d.rew.append(_get_f32(b, off))
        off += 4
        for _ in range(obs_dim):
            d.nobs.append(_get_f32(b, off))
            off += 4
        d.done.append(_get_f32(b, off))
        off += 4
        d.flags.append(_get_u32(b, off))
        off += 4
    for _ in range(n_ep):
        var s = Int(_get_u64(b, off))
        var l = Int(_get_u64(b, off + 8))
        var ok = _get_u32(b, off + 16) != 0
        off += 20
        if s < 0 or l < 0 or s + l > n:
            raise Error("read_demo_file: episode table out of range in " + path)
        d.ep_start.append(s)
        d.ep_len.append(l)
        d.ep_success.append(ok)
    return d^
