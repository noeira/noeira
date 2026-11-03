# +--------------------------------------------------------------------------+ #
# | SuperMimic M2d: the replay ring — k-step windows and history starts
# +--------------------------------------------------------------------------+ #
"""The reproduction's `ReplayBuffer` (`data/replay_buffer.py`), on the host.

A ring of `capacity` rows per env. A row holds the state s_t at the start of a
control step and the transition that followed: clean action a_t, executed
action, ξ, reward r_{t+1}, the termination label `done` of the row's state,
the target frame, and bookkeeping (episode id, `terminal`, `ep_start`). When
an episode ends a TERMINAL row holds s_{T+1} (no transition out of it).

    window (k): rows t..t+k valid, one episode, t..t+k−1 non-terminal, and
                the window not reaching the write pointer
    start (h):  a non-terminal row t with the h rows before it (backfilled
                with the episode's first row before its start; the action
                before each history state is 0 at the episode's start)

Sampling is uniform over the eligible rows (`torch.multinomial` of the mask);
`*_at` take the flat indices `e·T + t` directly, to replay another run's draws.
"""

from std.random import random_ui64

from noeira.nn.constants import DT


comptime RB_S: Int = 71
comptime RB_A: Int = 29
comptime RB_XI: Int = 33


struct SMReplayBuffer(Movable):
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
        self.s = List[Scalar[DT]](length=n * RB_S, fill=0)
        self.a = List[Scalar[DT]](length=n * RB_A, fill=0)
        self.a_exec = List[Scalar[DT]](length=n * RB_A, fill=0)
        self.xi = List[Scalar[DT]](length=n * RB_XI, fill=0)
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
        xi: List[Scalar[DT]], r: Float64, done: Float64, frame: Int, terminal: Bool, ep_start: Bool,
    ):
        var p = self.ptr[e]
        var i = e * self.T + p
        for k in range(RB_S):
            self.s[i * RB_S + k] = s[k]
        for k in range(RB_A):
            self.a[i * RB_A + k] = a[k]
            self.a_exec[i * RB_A + k] = a_exec[k]
        for k in range(RB_XI):
            self.xi[i * RB_XI + k] = xi[k]
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
            raise Error("SMReplayBuffer: no sampleable rows yet")
        var out = List[Int](length=n, fill=0)
        for j in range(n):
            out[j] = elig[Int(random_ui64(0, UInt64(len(elig) - 1)))]
        return out^

    def windows_at(self, idx: List[Int], k: Int) -> SMWindows:
        """The windows starting at flat rows `idx` (t-major outputs)."""
        var n = len(idx)
        var w = SMWindows(k, n)
        for c in range(n):
            var e = idx[c] // self.T
            var t = idx[c] % self.T
            var i0 = e * self.T + t
            for k2 in range(RB_XI):
                w.xi[c * RB_XI + k2] = self.xi[i0 * RB_XI + k2]
            for j in range(k + 1):
                var i = self._at(e, t + j)
                for q in range(RB_S):
                    w.states[(j * n + c) * RB_S + q] = self.s[i * RB_S + q]
                if j < k:
                    for q in range(RB_A):
                        w.a_exec[(j * n + c) * RB_A + q] = self.a_exec[i * RB_A + q]
                        w.a_clean[(j * n + c) * RB_A + q] = self.a[i * RB_A + q]
                    w.reward[j * n + c] = self.r[i]
                    w.frame[j * n + c] = self.frame[i]
                if j >= 1:
                    w.done[(j - 1) * n + c] = self.done[i]
        return w^

    def starts_at(self, idx: List[Int], hist: Int) -> SMStarts:
        var n = len(idx)
        var st = SMStarts(hist, n)
        for c in range(n):
            var e = idx[c] // self.T
            var t = idx[c] % self.T
            var i0 = e * self.T + t
            for q in range(RB_S):
                st.state[c * RB_S + q] = self.s[i0 * RB_S + q]
            for q in range(RB_XI):
                st.xi[c * RB_XI + q] = self.xi[i0 * RB_XI + q]
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
                for q in range(RB_S):
                    st.hist_states[(h * n + c) * RB_S + q] = self.s[ih * RB_S + q]
                var tp = e * self.T + ((ih - e * self.T) - 1 + self.T) % self.T
                var prev_ok = self.ep_id[tp] == self.ep_id[i0] and self.valid[tp] and not self.ep_start[ih]
                for q in range(RB_A):
                    st.hist_actions[(h * n + c) * RB_A + q] = self.a[tp * RB_A + q] if prev_ok else Scalar[DT](0)
            for q in range(RB_A):
                st.prev_action[c * RB_A + q] = st.hist_actions[((hist - 1) * n + c) * RB_A + q]
        return st^


struct SMWindows(Movable):
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
        self.states = List[Scalar[DT]](length=(k + 1) * n * RB_S, fill=0)
        self.a_exec = List[Scalar[DT]](length=k * n * RB_A, fill=0)
        self.a_clean = List[Scalar[DT]](length=k * n * RB_A, fill=0)
        self.xi = List[Scalar[DT]](length=n * RB_XI, fill=0)
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


struct SMStarts(Movable):
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
        self.state = List[Scalar[DT]](length=n * RB_S, fill=0)
        self.xi = List[Scalar[DT]](length=n * RB_XI, fill=0)
        self.frame = List[Int](length=n, fill=0)
        self.prev_action = List[Scalar[DT]](length=n * RB_A, fill=0)
        self.hist_states = List[Scalar[DT]](length=hist * n * RB_S, fill=0)
        self.hist_actions = List[Scalar[DT]](length=hist * n * RB_A, fill=0)

    def __init__(out self, *, deinit move: Self):
        self.hist = move.hist
        self.n = move.n
        self.state = move.state^
        self.xi = move.xi^
        self.frame = move.frame^
        self.prev_action = move.prev_action^
        self.hist_states = move.hist_states^
        self.hist_actions = move.hist_actions^
