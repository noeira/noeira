"""Asynchronous execution of action chunks under an inference delay.

    from noeira.deep_agents.bc.async_chunk import (
        AsyncChunkExec, add_knot_residual,
    )

A chunked policy (ACT, a VLA, a chunk BC net) answers a query `d` control
steps after the observation it was computed from. `AsyncChunkExec` is the
bookkeeping of the pipelined loop around it, with no policy and no
environment inside, so the delay can be emulated around a policy that is
itself instantaneous and gated without either:

    each control step t:
        if ex.needs_query():      # no query in flight, <= d actions left
            chunk = policy(obs_t)             # the observation of step t
            ex.submit(chunk)                  # usable from step t + d
        if ex.lands_now():
            ex.land()
        if ex.has_action(): a = ex.peek(j) for each j
        else:               a = hold          # only before the first chunk
        env.step(a)
        ex.advance()

## The index rule

A chunk's action `i` is meant for step `t_query + i`. A chunk that lands at
`t_query + d` is therefore executed FROM ITS INDEX `d`, for `n_exec` steps.
The query starts when exactly `d` actions of the executing chunk are left, so
it lands on the step the executing chunk runs out: no hold, no overlap.

⚠ THE FIRST CHUNK IS EXECUTED FROM INDEX 0. Nothing was executing while it
was computed: the robot held its pose for `d` steps, so the state the plan
starts from is still the state it was computed for.

With `d = 0` this is the synchronous loop: query at a chunk boundary, the
answer used at once from index 0.

## The committed actions

`committed` returns the actions that will still execute while a query
started now is in flight: the `d` left in the executing chunk (ARLI's first
state addition, arXiv 2608.23831).
"""


struct AsyncChunkExec(Movable):
    var h: Int
    """Actions in a chunk."""
    var act: Int
    """Words per action."""
    var n_exec: Int
    """Actions executed from each chunk."""
    var d: Int
    """Control steps between a query's observation and its chunk."""

    var t: Int
    """Control steps advanced since `reset`."""
    var cur: List[Float64]
    var have_cur: Bool
    var cur_start: Int
    var cur_idx: Int
    """Index in `cur` of the next action to execute."""
    var cur_end: Int
    var cur_query_t: Int
    var pend: List[Float64]
    var have_pend: Bool
    var pend_land: Int
    var pend_start: Int
    var pend_query_t: Int
    var n_queries: Int
    var n_holds: Int
    """Steps advanced with no action to give (the wait for the first chunk)."""

    def __init__(out self, h: Int, act: Int, n_exec: Int, d: Int) raises:
        if h < 1 or act < 1 or n_exec < 1 or d < 0:
            raise Error(
                "AsyncChunkExec: h " + String(h) + ", act " + String(act)
                + ", n_exec " + String(n_exec) + ", d " + String(d)
            )
        if d + n_exec > h:
            raise Error(
                "AsyncChunkExec: a chunk of " + String(h) + " cannot give "
                + String(n_exec) + " actions from its index " + String(d)
                + " — the delay plus the executed part must fit the chunk"
            )
        self.h = h
        self.act = act
        self.n_exec = n_exec
        self.d = d
        self.t = 0
        self.cur = List[Float64](length=h * act, fill=0.0)
        self.have_cur = False
        self.cur_start = 0
        self.cur_idx = 0
        self.cur_end = 0
        self.cur_query_t = 0
        self.pend = List[Float64](length=h * act, fill=0.0)
        self.have_pend = False
        self.pend_land = 0
        self.pend_start = 0
        self.pend_query_t = 0
        self.n_queries = 0
        self.n_holds = 0

    def reset(mut self):
        self.t = 0
        self.have_cur = False
        self.cur_start = 0
        self.cur_idx = 0
        self.cur_end = 0
        self.cur_query_t = 0
        self.have_pend = False
        self.pend_land = 0
        self.pend_start = 0
        self.pend_query_t = 0
        self.n_queries = 0
        self.n_holds = 0

    def needs_query(self) -> Bool:
        """No query in flight, and the executing chunk has at most `d`
        actions left (or there is none yet)."""
        if self.have_pend:
            return False
        if not self.have_cur:
            return True
        return self.cur_end - self.cur_idx <= self.d

    def next_start(self) -> Int:
        """The index a chunk submitted NOW will be executed from."""
        if not self.have_cur:
            return 0
        return self.d

    def submit(mut self, ref chunk: List[Float64]) raises:
        """The chunk computed from this step's observation; it lands `d`
        steps from now."""
        if not self.needs_query():
            raise Error("AsyncChunkExec.submit: no query is due at this step")
        if len(chunk) != self.h * self.act:
            raise Error(
                "AsyncChunkExec.submit: a chunk of " + String(len(chunk))
                + " words, expected " + String(self.h * self.act)
            )
        self.pend_start = self.next_start()
        for k in range(self.h * self.act):
            self.pend[k] = chunk[k]
        self.have_pend = True
        self.pend_land = self.t + self.d
        self.pend_query_t = self.t
        self.n_queries += 1

    def lands_now(self) -> Bool:
        """The pending chunk is ready and the executing one has run out."""
        if not self.have_pend or self.t < self.pend_land:
            return False
        return (not self.have_cur) or self.cur_idx >= self.cur_end

    def land(mut self) raises:
        if not self.lands_now():
            raise Error("AsyncChunkExec.land: nothing lands at this step")
        for k in range(self.h * self.act):
            self.cur[k] = self.pend[k]
        self.cur_start = self.pend_start
        self.cur_idx = self.pend_start
        self.cur_end = self.pend_start + self.n_exec
        self.cur_query_t = self.pend_query_t
        self.have_cur = True
        self.have_pend = False

    def has_action(self) -> Bool:
        return self.have_cur and self.cur_idx < self.cur_end

    def peek(self, j: Int) -> Float64:
        """Word `j` of the action this step executes. Call only when
        `has_action()`."""
        return self.cur[self.cur_idx * self.act + j]

    def exec_fraction(self) -> Float64:
        """How far through the executed part of the chunk this step is, in
        [0, 1)."""
        return Float64(self.cur_idx - self.cur_start) / Float64(self.n_exec)

    def advance(mut self):
        """One control step has been commanded."""
        if self.has_action():
            self.cur_idx += 1
        else:
            self.n_holds += 1
        self.t += 1

    def committed(self, mut out: List[Float64], dmax: Int) raises:
        """The actions still to execute from the current chunk, oldest first,
        zero-padded to `dmax` actions. At a query's start these are the
        actions that run while it is in flight."""
        if len(out) != dmax * self.act:
            raise Error(
                "AsyncChunkExec.committed: out holds " + String(len(out))
                + " words, expected " + String(dmax * self.act)
            )
        for k in range(dmax * self.act):
            out[k] = 0.0
        if not self.have_cur:
            return
        var left = self.cur_end - self.cur_idx
        if left > dmax:
            raise Error(
                "AsyncChunkExec.committed: " + String(left)
                + " actions are left and dmax is " + String(dmax)
            )
        for i in range(left):
            for j in range(self.act):
                out[i * self.act + j] = self.cur[
                    (self.cur_idx + i) * self.act + j
                ]


def add_knot_residual(
    mut chunk: List[Float64], start: Int, n_exec: Int, act: Int,
    ref knots: List[Float64], n_knots: Int, cap: Float64,
) raises:
    """Add a bounded residual to actions `[start, start + n_exec)` of a chunk
    and clamp the result to [-1, 1].

    The residual is piecewise linear in time between `n_knots` knots of
    `act` words each (`knots` in [-1, 1], scaled by `cap`): knot 0 at the
    first executed action, the last knot at the last one. One knot is a
    constant offset."""
    if n_knots < 1 or len(knots) != n_knots * act:
        raise Error(
            "add_knot_residual: " + String(len(knots)) + " words for "
            + String(n_knots) + " knots of " + String(act)
        )
    if start < 0 or (start + n_exec) * act > len(chunk):
        raise Error(
            "add_knot_residual: the window [" + String(start) + ", "
            + String(start + n_exec) + ") leaves the chunk"
        )
    for i in range(n_exec):
        var k0 = 0
        var f = 0.0
        if n_knots > 1 and n_exec > 1:
            var u = Float64(i) * Float64(n_knots - 1) / Float64(n_exec - 1)
            k0 = Int(u)
            if k0 > n_knots - 2:
                k0 = n_knots - 2
            f = u - Float64(k0)
        for j in range(act):
            var r = knots[k0 * act + j]
            if n_knots > 1:
                r = (1.0 - f) * r + f * knots[(k0 + 1) * act + j]
            var a = chunk[(start + i) * act + j] + cap * r
            if a > 1.0:
                a = 1.0
            if a < -1.0:
                a = -1.0
            chunk[(start + i) * act + j] = a
