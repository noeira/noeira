"""EpisodicReplay[S, A, X] — the CPU per-env episodic window replay.

Conforms to `EpisodicReplayBuffer` (see there for the row and window
semantics). Storage is `[env, time]`: row (e, t) at flat index `e·T + t`, a
write pointer and an episode counter per env. Sampling is uniform with
replacement over the eligible rows, as `torch.multinomial` of the mask;
`windows_at` / `starts_at` take the flat indices directly so another run's
draws can be replayed exactly.

Ported from the SuperMimic reproduction's `data/replay_buffer.py` and gated
against it (`tests/deep_agents/data/test_episodic_replay.mojo`). The context
channel is named `xi` after its first user (the sim context ξ).
"""

from std.random import random_ui64

from noeira.nn.constants import DT
from .episodic_replay_buffer import EpisodicReplayBuffer


struct EpisodicReplay[S_: Int, A_: Int, X_: Int](EpisodicReplayBuffer):
    comptime S = Self.S_
    comptime A = Self.A_
    comptime X = Self.X_

    var E: Int
    var T: Int
    var s: List[Scalar[DT]]
    var a: List[Scalar[DT]]
    var a_exec: List[Scalar[DT]]
    var xi: List[Scalar[DT]]
    var r: List[Scalar[DT]]
    var done: List[Scalar[DT]]
    var frame: List[Int]
    var ep_id: List[Int]
    var terminal: List[Bool]
    var ep_start: List[Bool]
    var valid: List[Bool]
    var ptr: List[Int]
    var ep_counter: List[Int]

    def __init__(out self, num_envs: Int, capacity: Int):
        self.E = num_envs
        self.T = capacity
        var n = num_envs * capacity
        self.s = List[Scalar[DT]](length=n * Self.S, fill=0)
        self.a = List[Scalar[DT]](length=n * Self.A, fill=0)
        self.a_exec = List[Scalar[DT]](length=n * Self.A, fill=0)
        self.xi = List[Scalar[DT]](length=n * Self.X, fill=0)
        self.r = List[Scalar[DT]](length=n, fill=0)
        self.done = List[Scalar[DT]](length=n, fill=0)
        self.frame = List[Int](length=n, fill=0)
        self.ep_id = List[Int](length=n, fill=-1)
        self.terminal = List[Bool](length=n, fill=False)
        self.ep_start = List[Bool](length=n, fill=False)
        self.valid = List[Bool](length=n, fill=False)
        self.ptr = List[Int](length=num_envs, fill=0)
        self.ep_counter = List[Int](length=num_envs, fill=0)

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
        self.terminal = move.terminal^
        self.ep_start = move.ep_start^
        self.valid = move.valid^
        self.ptr = move.ptr^
        self.ep_counter = move.ep_counter^

    def add(
        mut self, e: Int, s: List[Scalar[DT]], a: List[Scalar[DT]], a_exec: List[Scalar[DT]],
        x: List[Scalar[DT]], r: Float64, done: Float64, frame: Int, terminal: Bool, ep_start: Bool,
    ):
        var p = self.ptr[e]
        var i = e * self.T + p
        for k in range(Self.S):
            self.s[i * Self.S + k] = s[k]
        for k in range(Self.A):
            self.a[i * Self.A + k] = a[k]
            self.a_exec[i * Self.A + k] = a_exec[k]
        for k in range(Self.X):
            self.xi[i * Self.X + k] = x[k]
        self.r[i] = Scalar[DT](r)
        self.done[i] = Scalar[DT](done)
        self.frame[i] = frame
        self.ep_id[i] = self.ep_counter[e]
        self.terminal[i] = terminal
        self.ep_start[i] = ep_start
        self.valid[i] = True
        self.ptr[e] = (p + 1) % self.T

    def end_episode(mut self, e: Int):
        self.ep_counter[e] += 1

    def num_envs(self) -> Int:
        return self.E

    def capacity(self) -> Int:
        return self.T

    @always_inline
    def _at(self, e: Int, t: Int) -> Int:
        return e * self.T + (t % self.T)

    def window_mask(self, k: Int) -> List[Bool]:
        var m = List[Bool](length=self.E * self.T, fill=False)
        for e in range(self.E):
            for t in range(self.T):
                var i = e * self.T + t
                var ok = self.valid[i] and not self.terminal[i]
                var j = 1
                while ok and j <= k:
                    var ij = self._at(e, t + j)
                    ok = self.valid[ij] and self.ep_id[ij] == self.ep_id[i]
                    if ok and j < k:
                        ok = not self.terminal[ij]
                    j += 1
                if ok:
                    var dist = (self.ptr[e] - t - 1) % self.T
                    if dist < 0:
                        dist += self.T
                    ok = dist >= k
                m[i] = ok
        return m^

    def start_mask(self, hist: Int) -> List[Bool]:
        var m = List[Bool](length=self.E * self.T, fill=False)
        for e in range(self.E):
            for t in range(self.T):
                var i = e * self.T + t
                var dist = (self.ptr[e] - t - 1) % self.T
                if dist < 0:
                    dist += self.T
                m[i] = self.valid[i] and not self.terminal[i] and dist < self.T - hist - 1
        return m^

    @staticmethod
    def sample_indices(mask: List[Bool], n: Int) raises -> List[Int]:
        """Uniform with replacement over the True entries (flat indices)."""
        var elig = List[Int]()
        for i in range(len(mask)):
            if mask[i]:
                elig.append(i)
        if len(elig) == 0:
            raise Error("EpisodicReplay: no sampleable rows yet")
        var out = List[Int](length=n, fill=0)
        for j in range(n):
            out[j] = elig[Int(random_ui64(0, UInt64(len(elig) - 1)))]
        return out^

    def windows_at(self, idx: List[Int], k: Int) -> EpisodicWindows[Self.S, Self.A, Self.X]:
        """The windows starting at flat rows `idx` (t-major outputs)."""
        var n = len(idx)
        var w = EpisodicWindows[Self.S, Self.A, Self.X](k, n)
        for c in range(n):
            var e = idx[c] // self.T
            var t = idx[c] % self.T
            var i0 = e * self.T + t
            for k2 in range(Self.X):
                w.xi[c * Self.X + k2] = self.xi[i0 * Self.X + k2]
            for j in range(k + 1):
                var i = self._at(e, t + j)
                for q in range(Self.S):
                    w.states[(j * n + c) * Self.S + q] = self.s[i * Self.S + q]
                if j < k:
                    for q in range(Self.A):
                        w.a_exec[(j * n + c) * Self.A + q] = self.a_exec[i * Self.A + q]
                        w.a_clean[(j * n + c) * Self.A + q] = self.a[i * Self.A + q]
                    w.reward[j * n + c] = self.r[i]
                    w.frame[j * n + c] = self.frame[i]
                if j >= 1:
                    w.done[(j - 1) * n + c] = self.done[i]
        return w^

    def starts_at(self, idx: List[Int], hist: Int) -> EpisodicStarts[Self.S, Self.A, Self.X]:
        var n = len(idx)
        var st = EpisodicStarts[Self.S, Self.A, Self.X](hist, n)
        for c in range(n):
            var e = idx[c] // self.T
            var t = idx[c] % self.T
            var i0 = e * self.T + t
            for q in range(Self.S):
                st.state[c * Self.S + q] = self.s[i0 * Self.S + q]
            for q in range(Self.X):
                st.xi[c * Self.X + q] = self.xi[i0 * Self.X + q]
            st.frame[c] = self.frame[i0]
            # rows t−hist+1 .. t; outside the episode → its first row in the window
            var tt = List[Int](length=hist, fill=0)
            var same = List[Bool](length=hist, fill=False)
            var first_ok = hist - 1
            for h in range(hist):
                var ih = self._at(e, t + h - (hist - 1) + self.T)
                tt[h] = ih
                same[h] = self.ep_id[ih] == self.ep_id[i0] and self.valid[ih]
            for h in range(hist):
                if same[h]:
                    first_ok = h
                    break
            for h in range(hist):
                var ih = tt[h] if same[h] else tt[first_ok]
                for q in range(Self.S):
                    st.hist_states[(h * n + c) * Self.S + q] = self.s[ih * Self.S + q]
                var tp = e * self.T + ((ih - e * self.T) - 1 + self.T) % self.T
                var prev_ok = self.ep_id[tp] == self.ep_id[i0] and self.valid[tp] and not self.ep_start[ih]
                for q in range(Self.A):
                    st.hist_actions[(h * n + c) * Self.A + q] = self.a[tp * Self.A + q] if prev_ok else Scalar[DT](0)
            for q in range(Self.A):
                st.prev_action[c * Self.A + q] = st.hist_actions[((hist - 1) * n + c) * Self.A + q]
        return st^


struct EpisodicWindows[S: Int, A: Int, X: Int](Movable):
    """k-step windows, t-major: states [(k+1)·n·71], a_exec / a_clean
    [k·n·29], xi [n·33] (at the window start), reward / done / frame [k·n]
    (done = the label of state j+1)."""
    var k: Int
    var n: Int
    var states: List[Scalar[DT]]
    var a_exec: List[Scalar[DT]]
    var a_clean: List[Scalar[DT]]
    var xi: List[Scalar[DT]]
    var reward: List[Scalar[DT]]
    var done: List[Scalar[DT]]
    var frame: List[Int]

    def __init__(out self, k: Int, n: Int):
        self.k = k
        self.n = n
        self.states = List[Scalar[DT]](length=(k + 1) * n * Self.S, fill=0)
        self.a_exec = List[Scalar[DT]](length=k * n * Self.A, fill=0)
        self.a_clean = List[Scalar[DT]](length=k * n * Self.A, fill=0)
        self.xi = List[Scalar[DT]](length=n * Self.X, fill=0)
        self.reward = List[Scalar[DT]](length=k * n, fill=0)
        self.done = List[Scalar[DT]](length=k * n, fill=0)
        self.frame = List[Int](length=k * n, fill=0)

    def __init__(out self, *, deinit move: Self):
        self.k = move.k
        self.n = move.n
        self.states = move.states^
        self.a_exec = move.a_exec^
        self.a_clean = move.a_clean^
        self.xi = move.xi^
        self.reward = move.reward^
        self.done = move.done^
        self.frame = move.frame^


struct EpisodicStarts[S: Int, A: Int, X: Int](Movable):
    """Start states with their history: state [n·71], xi [n·33], frame [n],
    prev_action [n·29], hist_states [h·n·71] (oldest first), hist_actions
    [h·n·29] (the clean action before each history state)."""
    var hist: Int
    var n: Int
    var state: List[Scalar[DT]]
    var xi: List[Scalar[DT]]
    var frame: List[Int]
    var prev_action: List[Scalar[DT]]
    var hist_states: List[Scalar[DT]]
    var hist_actions: List[Scalar[DT]]

    def __init__(out self, hist: Int, n: Int):
        self.hist = hist
        self.n = n
        self.state = List[Scalar[DT]](length=n * Self.S, fill=0)
        self.xi = List[Scalar[DT]](length=n * Self.X, fill=0)
        self.frame = List[Int](length=n, fill=0)
        self.prev_action = List[Scalar[DT]](length=n * Self.A, fill=0)
        self.hist_states = List[Scalar[DT]](length=hist * n * Self.S, fill=0)
        self.hist_actions = List[Scalar[DT]](length=hist * n * Self.A, fill=0)

    def __init__(out self, *, deinit move: Self):
        self.hist = move.hist
        self.n = move.n
        self.state = move.state^
        self.xi = move.xi^
        self.frame = move.frame^
        self.prev_action = move.prev_action^
        self.hist_states = move.hist_states^
        self.hist_actions = move.hist_actions^
