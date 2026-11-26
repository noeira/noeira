"""`AsyncChunkExec` — the pipelined chunk loop under a delay, with no policy.

    pixi run mojo run -I . tests/deep_agents/test_async_chunk.mojo

A chunk here is filled with markers: action `i` of the chunk queried at step
`q` is `q + i / 1000` in every word, so an executed action names both the
query it came from and its index. What this pins:

  1. THE INDEX RULE. Under a delay, every chunk after the first is executed
     from its index `d`: the action run at step `t` is index `t - q` of the
     chunk queried at `q`. A loop that executed from index 0 would run
     actions `d` steps late and fail this.
  2. NO HOLD AFTER THE FIRST CHUNK. The only steps without an action are the
     `d` steps before the first chunk lands; it is executed from index 0.
  3. `d = 0` IS THE SYNCHRONOUS LOOP: a query every `n_exec` steps, index 0.
  4. THE COMMITTED ACTIONS at a query's start are the `d` actions that then
     run while it is in flight, in order.
  5. A delay that does not fit the chunk, and a query nobody asked for, are
     refused.
  6. The knot residual: the ends carry the knots, the window's outside is
     untouched, the result is clamped.
"""

from std.testing import assert_equal, assert_true, assert_almost_equal, TestSuite

from noeira.deep_agents.bc.async_chunk import AsyncChunkExec, add_knot_residual


comptime H: Int = 16
comptime ACT: Int = 2


def _marker_chunk(q: Int) -> List[Float64]:
    var c = List[Float64](length=H * ACT, fill=0.0)
    for i in range(H):
        for j in range(ACT):
            c[i * ACT + j] = Float64(q) + Float64(i) / 1000.0
    return c^


def _run(
    d: Int, n_exec: Int, steps: Int, mut query_t: List[Int],
    mut exec_q: List[Int], mut exec_i: List[Int],
) raises -> Int:
    """Drive the loop; per step the query and index of the executed action
    (-1 on a hold). Returns the number of holds."""
    var ex = AsyncChunkExec(H, ACT, n_exec, d)
    for t in range(steps):
        if ex.needs_query():
            var c = _marker_chunk(t)
            ex.submit(c)
            query_t.append(t)
        if ex.lands_now():
            ex.land()
        if ex.has_action():
            var a = ex.peek(0)
            var q = Int(a + 1.0e-9)
            var i = Int((a - Float64(q)) * 1000.0 + 0.5)
            assert_almost_equal(ex.peek(1), a)
            exec_q.append(q)
            exec_i.append(i)
        else:
            exec_q.append(-1)
            exec_i.append(-1)
        ex.advance()
    return ex.n_holds


def test_index_rule_under_delay() raises:
    var query_t = List[Int]()
    var exec_q = List[Int]()
    var exec_i = List[Int]()
    var d = 5
    var n_exec = 8
    var holds = _run(d, n_exec, 80, query_t, exec_q, exec_i)
    # the wait for the first chunk, and nothing after it
    assert_equal(holds, d)
    for t in range(d):
        assert_equal(exec_q[t], -1)
    # the first chunk: queried at 0, executed from index 0
    for k in range(n_exec):
        assert_equal(exec_q[d + k], 0)
        assert_equal(exec_i[d + k], k)
    # every later action is index (t - q) of the chunk queried at q
    var later = 0
    for t in range(d + n_exec, 80):
        assert_true(exec_q[t] > 0)
        assert_equal(exec_i[t], t - exec_q[t])
        assert_true(exec_i[t] >= d)
        assert_true(exec_i[t] < d + n_exec)
        later += 1
    assert_true(later > 60)
    # queries are n_exec apart once the loop is running, each started d
    # steps before the executing chunk runs out
    assert_equal(query_t[0], 0)
    assert_equal(query_t[1], d + n_exec - d)
    for k in range(2, len(query_t)):
        assert_equal(query_t[k] - query_t[k - 1], n_exec)


def test_zero_delay_is_synchronous() raises:
    var query_t = List[Int]()
    var exec_q = List[Int]()
    var exec_i = List[Int]()
    var n_exec = 4
    var holds = _run(0, n_exec, 40, query_t, exec_q, exec_i)
    assert_equal(holds, 0)
    for t in range(40):
        assert_equal(exec_q[t], (t // n_exec) * n_exec)
        assert_equal(exec_i[t], t % n_exec)
    assert_equal(len(query_t), 10)


def test_committed_are_the_actions_that_then_run() raises:
    var d = 3
    var n_exec = 6
    var dmax = 5
    var ex = AsyncChunkExec(H, ACT, n_exec, d)
    var com = List[Float64](length=dmax * ACT, fill=-7.0)
    var saved = List[Float64]()
    var saved_at = -1
    var checked = 0
    for t in range(60):
        if ex.needs_query():
            ex.committed(com, dmax)
            if t == 0:
                # nothing is executing before the first chunk
                for k in range(dmax * ACT):
                    assert_equal(com[k], 0.0)
            else:
                saved = com.copy()
                saved_at = t
                # the padding beyond d actions is zero
                for k in range(d * ACT, dmax * ACT):
                    assert_equal(com[k], 0.0)
            var c = _marker_chunk(t)
            ex.submit(c)
        if ex.lands_now():
            ex.land()
        if ex.has_action() and saved_at >= 0 and t - saved_at < d:
            assert_almost_equal(ex.peek(0), saved[(t - saved_at) * ACT])
            assert_almost_equal(ex.peek(1), saved[(t - saved_at) * ACT + 1])
            checked += 1
        ex.advance()
    assert_true(checked > 15)


def test_refusals() raises:
    var refused = False
    try:
        var bad = AsyncChunkExec(H, ACT, 10, 7)   # 7 + 10 > 16
        _ = bad.h
    except e:
        refused = True
    assert_true(refused)

    var ex = AsyncChunkExec(H, ACT, 8, 4)
    var c = _marker_chunk(0)
    ex.submit(c)
    var twice = False
    try:
        ex.submit(c)          # one query is already in flight
    except e:
        twice = True
    assert_true(twice)


def test_knot_residual() raises:
    var chunk = List[Float64](length=H * ACT, fill=0.25)
    var knots: List[Float64] = [1.0, -1.0, -1.0, 0.0]   # 2 knots of 2 words
    add_knot_residual(chunk, 4, 5, ACT, knots, 2, 0.1)
    # outside the window: untouched
    for i in range(4):
        assert_equal(chunk[i * ACT], 0.25)
    for i in range(9, H):
        assert_equal(chunk[i * ACT + 1], 0.25)
    # the ends carry the knots, the middle is their mean
    assert_almost_equal(chunk[4 * ACT], 0.35)
    assert_almost_equal(chunk[4 * ACT + 1], 0.15)
    assert_almost_equal(chunk[8 * ACT], 0.15)
    assert_almost_equal(chunk[8 * ACT + 1], 0.25)
    assert_almost_equal(chunk[6 * ACT], 0.25)
    assert_almost_equal(chunk[6 * ACT + 1], 0.20)
    # one knot is a constant offset, and the result is clamped
    var hi = List[Float64](length=H * ACT, fill=0.95)
    var one: List[Float64] = [1.0, -1.0]
    add_knot_residual(hi, 0, H, ACT, one, 1, 0.1)
    for i in range(H):
        assert_equal(hi[i * ACT], 1.0)
        assert_almost_equal(hi[i * ACT + 1], 0.85)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
