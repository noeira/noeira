"""EpisodicReplayGpu[S, A, X] — the device sibling of `EpisodicReplay`.

Same storage (`[env, time]` rings, row (e, t) at `e·T + t`, one write pointer
and episode counter per env), same row semantics and the same window / start
rules (`episodic_replay_buffer.mojo`), with every row on the device so a
batched simulator writes it and the learner samples it without a host round
trip.

Writing is one call per control step for all envs (`add_step`): every env
writes its transition row, and an env whose episode ends also writes the
TERMINAL row of its final state and closes the episode — the pattern of
`EpisodicReplay.add` + `add(terminal)` + `end_episode` in a model-based
recorder, done in one kernel.

Sampling is exact-uniform with replacement over the eligible rows, as the
CPU ring's `sample_indices`: a kernel per env tests its rows and compacts the
eligible ones (in flat order), one prefix sum over the env counts gives every
eligible row a global rank, and each draw maps a Philox uniform to a rank and
the rank to its row. `windows_at` / `starts_at` gather t-major batches for
given flat rows, so the CPU ring's gathers are the reference.

Integer columns (frame, episode id, flags) are int32; flags: bit 0 valid,
bit 1 terminal, bit 2 episode start.
"""

from std.math import ceildiv
from std.random.philox import Random as PhiloxRandom

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr


comptime I32 = DType.int32
comptime FPf = Pointer[Scalar[DT], MutAnyOrigin]
comptime FPi = Pointer[Int32, MutAnyOrigin]
comptime _TPB: Int = 128

comptime FLAG_VALID: Int32 = 1
comptime FLAG_TERMINAL: Int32 = 2
comptime FLAG_EP_START: Int32 = 4

# add_step's per-env int32 metadata, 4 per env
comptime META_FRAME0: Int = 0     # the transition row's frame
comptime META_FRAME1: Int = 1     # the terminal row's frame (if ending)
comptime META_EP_START: Int = 2   # the transition row starts its episode
comptime META_ENDING: Int = 3     # write the terminal row and close the episode


@always_inline
def _grid(n: Int) -> Int:
    return ceildiv(n, _TPB)


# ── kernels ─────────────────────────────────────────────────────────────────
def _k_fill_i32(n: Int64, p: FPi, v: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        p[unsafe_offset=i] = v


@always_inline
def _write_row[S: Int, A: Int, X: Int](
    i: Int, s: FPf, a: FPf, ae: FPf, xi: FPf, r: FPf, done: FPf, frame: FPi, epid: FPi, flags: FPi,
    src_s: FPf, src_a: FPf, src_ae: FPf, src_x: FPf, rv: Scalar[DT], dv: Scalar[DT], fv: Int32, ev: Int32, flv: Int32,
    zero_actions: Bool,
):
    for k in range(S):
        s[unsafe_offset=i * S + k] = src_s[unsafe_offset=k]
    for k in range(A):
        a[unsafe_offset=i * A + k] = 0 if zero_actions else src_a[unsafe_offset=k]
        ae[unsafe_offset=i * A + k] = 0 if zero_actions else src_ae[unsafe_offset=k]
    for k in range(X):
        xi[unsafe_offset=i * X + k] = src_x[unsafe_offset=k]
    r[unsafe_offset=i] = rv
    done[unsafe_offset=i] = dv
    frame[unsafe_offset=i] = fv
    epid[unsafe_offset=i] = ev
    flags[unsafe_offset=i] = flv


def _k_add_step[S: Int, A: Int, X: Int](
    E: Int64, T: Int64, ptr: FPi, epc: FPi,
    s: FPf, a: FPf, ae: FPf, xi: FPf, r: FPf, done: FPf, frame: FPi, epid: FPi, flags: FPi,
    in_s0: FPf, in_a0: FPf, in_ae0: FPf, in_x0: FPf, in_rd: FPf, in_s1: FPf, meta: FPi,
):
    var e = Int(global_idx.x)
    if e >= Int(E):
        return
    var t_ = Int(T)
    var p = Int(ptr[unsafe_offset=e])
    var ep = epc[unsafe_offset=e]
    var m = meta.unsafe_offset(e * 4)
    var lbl = in_rd[unsafe_offset=e * 2 + 1]
    var fl0 = FLAG_VALID | (FLAG_EP_START if m[unsafe_offset=META_EP_START] != 0 else Int32(0))
    _write_row[S, A, X](
        e * t_ + p, s, a, ae, xi, r, done, frame, epid, flags,
        in_s0.unsafe_offset(e * S), in_a0.unsafe_offset(e * A), in_ae0.unsafe_offset(e * A), in_x0.unsafe_offset(e * X),
        in_rd[unsafe_offset=e * 2], lbl, m[unsafe_offset=META_FRAME0], ep, fl0, False,
    )
    p = (p + 1) % t_
    if m[unsafe_offset=META_ENDING] != 0:
        _write_row[S, A, X](
            e * t_ + p, s, a, ae, xi, r, done, frame, epid, flags,
            in_s1.unsafe_offset(e * S), in_a0, in_ae0, in_x0.unsafe_offset(e * X),
            0, lbl, m[unsafe_offset=META_FRAME1], ep, FLAG_VALID | FLAG_TERMINAL, True,
        )
        p = (p + 1) % t_
        epc[unsafe_offset=e] = ep + 1
    ptr[unsafe_offset=e] = Int32(p)


@always_inline
def _dist(ptr_e: Int, t: Int, T: Int) -> Int:
    """Rows written after row t in env e's ring: (ptr − t − 1) mod T."""
    var d = (ptr_e - t - 1) % T
    return d + T if d < 0 else d


@always_inline
def _window_ok(e: Int, t: Int, k: Int, T: Int, ptr_e: Int, epid: FPi, flags: FPi) -> Bool:
    var i = e * T + t
    var fi = flags[unsafe_offset=i]
    var ok = (fi & FLAG_VALID) != 0 and (fi & FLAG_TERMINAL) == 0
    var j = 1
    while ok and j <= k:
        var ij = e * T + (t + j) % T
        var fj = flags[unsafe_offset=ij]
        ok = (fj & FLAG_VALID) != 0 and epid[unsafe_offset=ij] == epid[unsafe_offset=i]
        if ok and j < k:
            ok = (fj & FLAG_TERMINAL) == 0
        j += 1
    return ok and _dist(ptr_e, t, T) >= k


@always_inline
def _start_ok(e: Int, t: Int, hist: Int, T: Int, ptr_e: Int, flags: FPi) -> Bool:
    var fi = flags[unsafe_offset=e * T + t]
    return (fi & FLAG_VALID) != 0 and (fi & FLAG_TERMINAL) == 0 and _dist(ptr_e, t, T) < T - hist - 1


def _k_compact(E: Int64, T: Int64, k: Int64, starts: Int64, ptr: FPi, epid: FPi, flags: FPi, elig: FPi, count: FPi):
    """Env e's eligible rows, in order, into elig[e·T :]; their number into count[e]."""
    var e = Int(global_idx.x)
    if e >= Int(E):
        return
    var t_ = Int(T)
    var pe = Int(ptr[unsafe_offset=e])
    var c = 0
    for t in range(t_):
        var ok = _start_ok(e, t, Int(k), t_, pe, flags) if starts != 0 else _window_ok(e, t, Int(k), t_, pe, epid, flags)
        if ok:
            elig[unsafe_offset=e * t_ + c] = Int32(e * t_ + t)
            c += 1
    count[unsafe_offset=e] = Int32(c)


def _k_prefix(E: Int64, count: FPi, prefix: FPi):
    """Inclusive prefix sum over the env counts (one thread: E is small)."""
    if Int(global_idx.x) != 0:
        return
    var acc = Int32(0)
    for e in range(Int(E)):
        acc += count[unsafe_offset=e]
        prefix[unsafe_offset=e] = acc


def _k_draw(n: Int64, E: Int64, T: Int64, seed: UInt64, prefix: FPi, elig: FPi, idx: FPi):
    """Draw i: a uniform rank among the eligible rows, then its row."""
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var total = Int(prefix[unsafe_offset=Int(E) - 1])
    if total == 0:
        idx[unsafe_offset=i] = -1
        return
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=0)
    var u = Float32(philox.step_uniform()[0])
    var g = Int(u * Float32(total))
    if g >= total:
        g = total - 1
    # the first env whose inclusive prefix exceeds g
    var lo = 0
    var hi = Int(E) - 1
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int(prefix[unsafe_offset=mid]) > g:
            hi = mid
        else:
            lo = mid + 1
    var before = 0 if lo == 0 else Int(prefix[unsafe_offset=lo - 1])
    idx[unsafe_offset=i] = elig[unsafe_offset=lo * Int(T) + (g - before)]


def _k_windows[S: Int, A: Int, X: Int](
    n: Int64, k: Int64, T: Int64, idx: FPi,
    s: FPf, a: FPf, ae: FPf, xi: FPf, r: FPf, done: FPf, frame: FPi,
    o_s: FPf, o_ae: FPf, o_a: FPf, o_x: FPf, o_r: FPf, o_d: FPf, o_f: FPi,
):
    """`EpisodicReplay.windows_at`, one thread per window."""
    var c = Int(global_idx.x)
    var nn = Int(n)
    if c >= nn:
        return
    var t_ = Int(T)
    var row0 = Int(idx[unsafe_offset=c])
    var e = row0 // t_
    var t = row0 % t_
    for q in range(X):
        o_x[unsafe_offset=c * X + q] = xi[unsafe_offset=row0 * X + q]
    for j in range(Int(k) + 1):
        var i = e * t_ + (t + j) % t_
        for q in range(S):
            o_s[unsafe_offset=(j * nn + c) * S + q] = s[unsafe_offset=i * S + q]
        if j < Int(k):
            for q in range(A):
                o_ae[unsafe_offset=(j * nn + c) * A + q] = ae[unsafe_offset=i * A + q]
                o_a[unsafe_offset=(j * nn + c) * A + q] = a[unsafe_offset=i * A + q]
            o_r[unsafe_offset=j * nn + c] = r[unsafe_offset=i]
            o_f[unsafe_offset=j * nn + c] = frame[unsafe_offset=i]
        if j >= 1:
            o_d[unsafe_offset=(j - 1) * nn + c] = done[unsafe_offset=i]


def _k_starts[S: Int, A: Int, X: Int](
    n: Int64, hist: Int64, T: Int64, idx: FPi,
    s: FPf, a: FPf, xi: FPf, frame: FPi, epid: FPi, flags: FPi,
    o_s: FPf, o_x: FPf, o_f: FPi, o_prev: FPf, o_hs: FPf, o_ha: FPf,
):
    """`EpisodicReplay.starts_at`, one thread per start: rows t−h+1 .. t,
    rows outside the episode backfilled with its first row in the window;
    the clean action before each history row (0 at an episode start)."""
    var c = Int(global_idx.x)
    var nn = Int(n)
    if c >= nn:
        return
    var t_ = Int(T)
    var hh_ = Int(hist)
    var row0 = Int(idx[unsafe_offset=c])
    var e = row0 // t_
    var t = row0 % t_
    var ep0 = epid[unsafe_offset=row0]
    for q in range(S):
        o_s[unsafe_offset=c * S + q] = s[unsafe_offset=row0 * S + q]
    for q in range(X):
        o_x[unsafe_offset=c * X + q] = xi[unsafe_offset=row0 * X + q]
    o_f[unsafe_offset=c] = frame[unsafe_offset=row0]
    # the first history slot inside the episode (no per-thread arrays: Metal)
    var first_ok = hh_ - 1
    for h in range(hh_):
        var ih = e * t_ + (t + h - (hh_ - 1) + t_) % t_
        if epid[unsafe_offset=ih] == ep0 and (flags[unsafe_offset=ih] & FLAG_VALID) != 0:
            first_ok = h
            break
    var i_first = e * t_ + (t + first_ok - (hh_ - 1) + t_) % t_
    for h in range(hh_):
        var ih = e * t_ + (t + h - (hh_ - 1) + t_) % t_
        if not (epid[unsafe_offset=ih] == ep0 and (flags[unsafe_offset=ih] & FLAG_VALID) != 0):
            ih = i_first
        for q in range(S):
            o_hs[unsafe_offset=(h * nn + c) * S + q] = s[unsafe_offset=ih * S + q]
        var tp = e * t_ + ((ih - e * t_) - 1 + t_) % t_
        var prev_ok = epid[unsafe_offset=tp] == ep0 and (flags[unsafe_offset=tp] & FLAG_VALID) != 0 and (flags[unsafe_offset=ih] & FLAG_EP_START) == 0
        for q in range(A):
            o_ha[unsafe_offset=(h * nn + c) * A + q] = a[unsafe_offset=tp * A + q] if prev_ok else Scalar[DT](0)
    for q in range(A):
        o_prev[unsafe_offset=c * A + q] = o_ha[unsafe_offset=((hh_ - 1) * nn + c) * A + q]


# ── the ring ────────────────────────────────────────────────────────────────
struct EpisodicReplayGpu[S_: Int, A_: Int, X_: Int](Movable):
    comptime S = Self.S_
    comptime A = Self.A_
    comptime X = Self.X_

    var E: Int
    var T: Int
    var s: DeviceBuffer[DT]
    var a: DeviceBuffer[DT]
    var a_exec: DeviceBuffer[DT]
    var xi: DeviceBuffer[DT]
    var r: DeviceBuffer[DT]
    var done: DeviceBuffer[DT]
    var frame: DeviceBuffer[I32]
    var ep_id: DeviceBuffer[I32]
    var flags: DeviceBuffer[I32]
    var ptr: DeviceBuffer[I32]
    var ep_counter: DeviceBuffer[I32]
    var elig: DeviceBuffer[I32]
    var count: DeviceBuffer[I32]
    var prefix: DeviceBuffer[I32]

    def __init__(out self, ctx: DeviceContext, num_envs: Int, capacity: Int) raises:
        self.E = num_envs
        self.T = capacity
        var n = num_envs * capacity
        self.s = ctx.enqueue_create_buffer[DT](n * Self.S)
        self.a = ctx.enqueue_create_buffer[DT](n * Self.A)
        self.a_exec = ctx.enqueue_create_buffer[DT](n * Self.A)
        self.xi = ctx.enqueue_create_buffer[DT](n * Self.X)
        self.r = ctx.enqueue_create_buffer[DT](n)
        self.done = ctx.enqueue_create_buffer[DT](n)
        self.frame = ctx.enqueue_create_buffer[I32](n)
        self.ep_id = ctx.enqueue_create_buffer[I32](n)
        self.flags = ctx.enqueue_create_buffer[I32](n)
        self.ptr = ctx.enqueue_create_buffer[I32](num_envs)
        self.ep_counter = ctx.enqueue_create_buffer[I32](num_envs)
        self.elig = ctx.enqueue_create_buffer[I32](n)
        self.count = ctx.enqueue_create_buffer[I32](num_envs)
        self.prefix = ctx.enqueue_create_buffer[I32](num_envs)
        # only the bookkeeping must start defined: rows are read only once valid
        self._fill(ctx, self.ep_id, n, -1)
        self._fill(ctx, self.flags, n, 0)
        self._fill(ctx, self.ptr, num_envs, 0)
        self._fill(ctx, self.ep_counter, num_envs, 0)
        ctx.synchronize()

    def __init__(out self, *, deinit move: Self):
        self.E = move.E
        self.T = move.T
        self.s = move.s^
        self.a = move.a^
        self.a_exec = move.a_exec^
        self.xi = move.xi^
        self.r = move.r^
        self.done = move.done^
        self.frame = move.frame^
        self.ep_id = move.ep_id^
        self.flags = move.flags^
        self.ptr = move.ptr^
        self.ep_counter = move.ep_counter^
        self.elig = move.elig^
        self.count = move.count^
        self.prefix = move.prefix^

    @staticmethod
    def _fill(ctx: DeviceContext, mut b: DeviceBuffer[I32], n: Int, v: Int32) raises:
        ctx.enqueue_function[_k_fill_i32](Int64(n), mptr(b.unsafe_ptr()), v, grid_dim=_grid(n), block_dim=_TPB)

    def num_envs(self) -> Int:
        return self.E

    def capacity(self) -> Int:
        return self.T

    def add_step(
        mut self, ctx: DeviceContext, s0: FPf, a0: FPf, ae0: FPf, x0: FPf, rd: FPf, s1: FPf, meta: FPi
    ) raises:
        """One control step for every env, device inputs: s0 [E·S], a0 / ae0
        [E·A] (clean / executed), x0 [E·X], rd [E·2] (reward, termination
        label), s1 [E·S] (the final state, read only for ending envs), meta
        [E·4] int32 (`META_*`). An ending env's terminal row carries s1, the
        label, frame1, zero actions and reward, then its episode closes."""
        ctx.enqueue_function[_k_add_step[Self.S, Self.A, Self.X]](
            Int64(self.E), Int64(self.T), mptr(self.ptr.unsafe_ptr()), mptr(self.ep_counter.unsafe_ptr()),
            mptr(self.s.unsafe_ptr()), mptr(self.a.unsafe_ptr()), mptr(self.a_exec.unsafe_ptr()), mptr(self.xi.unsafe_ptr()),
            mptr(self.r.unsafe_ptr()), mptr(self.done.unsafe_ptr()), mptr(self.frame.unsafe_ptr()),
            mptr(self.ep_id.unsafe_ptr()), mptr(self.flags.unsafe_ptr()),
            s0, a0, ae0, x0, rd, s1, meta,
            grid_dim=_grid(self.E), block_dim=_TPB,
        )

    def _eligible(mut self, ctx: DeviceContext, k: Int, starts: Bool) raises:
        ctx.enqueue_function[_k_compact](
            Int64(self.E), Int64(self.T), Int64(k), Int64(1 if starts else 0), mptr(self.ptr.unsafe_ptr()),
            mptr(self.ep_id.unsafe_ptr()), mptr(self.flags.unsafe_ptr()), mptr(self.elig.unsafe_ptr()),
            mptr(self.count.unsafe_ptr()), grid_dim=_grid(self.E), block_dim=_TPB,
        )
        ctx.enqueue_function[_k_prefix](Int64(self.E), mptr(self.count.unsafe_ptr()), mptr(self.prefix.unsafe_ptr()), grid_dim=1, block_dim=1)

    def sample_windows(mut self, ctx: DeviceContext, k: Int, n: Int, seed: UInt64, idx: FPi) raises:
        """n flat rows where a length-k window may start, uniform with
        replacement, into idx (−1 everywhere if none is eligible yet)."""
        self._eligible(ctx, k, False)
        self._draw(ctx, n, seed, idx)

    def sample_starts(mut self, ctx: DeviceContext, hist: Int, n: Int, seed: UInt64, idx: FPi) raises:
        """n flat rows where a start with an h-row history may be drawn."""
        self._eligible(ctx, hist, True)
        self._draw(ctx, n, seed, idx)

    def _draw(mut self, ctx: DeviceContext, n: Int, seed: UInt64, idx: FPi) raises:
        ctx.enqueue_function[_k_draw](
            Int64(n), Int64(self.E), Int64(self.T), seed, mptr(self.prefix.unsafe_ptr()), mptr(self.elig.unsafe_ptr()), idx,
            grid_dim=_grid(n), block_dim=_TPB,
        )

    def eligible_total(mut self, ctx: DeviceContext) raises -> Int:
        """The last `sample_*` call's eligible-row count (synchronises)."""
        var h = ctx.enqueue_create_host_buffer[I32](self.E)
        ctx.enqueue_copy(h, self.prefix)
        ctx.synchronize()
        return Int(h[self.E - 1])

    def windows_at(
        mut self, ctx: DeviceContext, idx: FPi, n: Int, k: Int,
        o_states: FPf, o_a_exec: FPf, o_a_clean: FPf, o_xi: FPf, o_reward: FPf, o_done: FPf, o_frame: FPi,
    ) raises:
        """`EpisodicReplay.windows_at` into device outputs of the same shapes."""
        ctx.enqueue_function[_k_windows[Self.S, Self.A, Self.X]](
            Int64(n), Int64(k), Int64(self.T), idx,
            mptr(self.s.unsafe_ptr()), mptr(self.a.unsafe_ptr()), mptr(self.a_exec.unsafe_ptr()), mptr(self.xi.unsafe_ptr()),
            mptr(self.r.unsafe_ptr()), mptr(self.done.unsafe_ptr()), mptr(self.frame.unsafe_ptr()),
            o_states, o_a_exec, o_a_clean, o_xi, o_reward, o_done, o_frame,
            grid_dim=_grid(n), block_dim=_TPB,
        )

    def starts_at(
        mut self, ctx: DeviceContext, idx: FPi, n: Int, hist: Int,
        o_state: FPf, o_xi: FPf, o_frame: FPi, o_prev_action: FPf, o_hist_states: FPf, o_hist_actions: FPf,
    ) raises:
        """`EpisodicReplay.starts_at` into device outputs of the same shapes."""
        ctx.enqueue_function[_k_starts[Self.S, Self.A, Self.X]](
            Int64(n), Int64(hist), Int64(self.T), idx,
            mptr(self.s.unsafe_ptr()), mptr(self.a.unsafe_ptr()), mptr(self.xi.unsafe_ptr()), mptr(self.frame.unsafe_ptr()),
            mptr(self.ep_id.unsafe_ptr()), mptr(self.flags.unsafe_ptr()),
            o_state, o_xi, o_frame, o_prev_action, o_hist_states, o_hist_actions,
            grid_dim=_grid(n), block_dim=_TPB,
        )
