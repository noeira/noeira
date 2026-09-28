"""The G1 tracking protocol's native half, without the store or Python.

`examples/g1/bfm_zero_eval_tracking.mojo` gates the whole protocol against
`tools/g1/bfm_zero_tracking_oracle.py` on real trajectories every run, but that
needs the 1.7 GB store and a Python env. These are the formula-level facts,
runnable anywhere, and one of them is discriminating in a way the oracle
comparison is not: `emd` and `distance` MUST disagree under a shuffle.

Run:
    pixi run mojo run -I . tests/deep_agents/test_g1_tracking_protocol.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true

from noeira.envs.robots.g1_tracking_eval import (
    G1_SEG_ROWS, G1_SEG_STRIDE, g1_n_segments, g1_segment_row, g1_segment_pick,
    g1_track_metrics,
)

comptime ACT: Int = 29
comptime T: Int = 8
comptime TOL: Float64 = 1e-12


def _fill(mut ach: List[Float64], mut tgt: List[Float64], off: Float64):
    """A target cloud and an achieved cloud offset by `off` along axis 0."""
    for j in range(T):
        for k in range(ACT):
            var v = 0.1 * Float64(j) + 0.01 * Float64(k)
            tgt[j * ACT + k] = v
            ach[j * ACT + k] = v
        ach[j * ACT] += off


def test_identity() raises:
    print("[1] achieved == target ...")
    var ach = List[Float64](length=T * ACT, fill=0.0)
    var tgt = List[Float64](length=T * ACT, fill=0.0)
    _fill(ach, tgt, 0.0)
    var m = g1_track_metrics[ACT](ach, tgt, T)
    print("      distance", m.distance, " emd", m.emd, " proximity", m.proximity)
    assert_true(m.distance < TOL, "distance " + String(m.distance))
    assert_true(m.emd < TOL, "emd " + String(m.emd))
    assert_true(abs(m.proximity - 1.0) < TOL, "proximity " + String(m.proximity))


def test_proximity_ramp() raises:
    """`bound` 2, `margin` 2: inside 2 scores 1, past 4 scores 0, and 3 is
    exactly half way down the ramp. Pins the VALUE, not just the ordering."""
    print("[2] the proximity ramp ...")
    for i in range(3):
        var off = 1.0 if i == 0 else (3.0 if i == 1 else 5.0)
        var want = 1.0 if i == 0 else (0.5 if i == 1 else 0.0)
        var ach = List[Float64](length=T * ACT, fill=0.0)
        var tgt = List[Float64](length=T * ACT, fill=0.0)
        _fill(ach, tgt, off)
        var m = g1_track_metrics[ACT](ach, tgt, T)
        print("      offset", off, " distance", m.distance, " proximity", m.proximity)
        assert_true(
            abs(m.distance - off) < 1e-9,
            "offset " + String(off) + ": distance " + String(m.distance),
        )
        assert_true(
            abs(m.proximity - want) < 1e-9,
            "offset " + String(off) + ": proximity " + String(m.proximity)
            + " want " + String(want),
        )


def test_emd_is_a_set_metric_and_distance_is_not() raises:
    """THE discriminating gate.

    `distance` pairs row t with row t — reorder the achieved rows and it moves.
    `emd` is an optimal assignment over the two clouds — reorder and it is
    IDENTICAL. Collapsing `emd` into a row-wise distance is the obvious
    "simplification" of `core/assignment.mojo`, it would pass every equality
    check above, and this is what refuses it.
    """
    print("[3] emd is permutation-invariant, distance is not ...")
    var ach = List[Float64](length=T * ACT, fill=0.0)
    var tgt = List[Float64](length=T * ACT, fill=0.0)
    _fill(ach, tgt, 0.0)
    # a real offset so both metrics are non-zero to begin with
    for j in range(T):
        ach[j * ACT + 1] += 0.5 * Float64(j % 3)
    var m0 = g1_track_metrics[ACT](ach, tgt, T)

    # reverse the achieved rows: same cloud, different pairing
    var rev = List[Float64](length=T * ACT, fill=0.0)
    for j in range(T):
        for k in range(ACT):
            rev[j * ACT + k] = ach[(T - 1 - j) * ACT + k]
    var m1 = g1_track_metrics[ACT](rev, tgt, T)

    print("      distance", m0.distance, "->", m1.distance)
    print("      emd     ", m0.emd, "->", m1.emd)
    assert_true(
        abs(m0.emd - m1.emd) < 1e-9,
        "emd moved under a shuffle (" + String(m0.emd) + " -> "
        + String(m1.emd) + ") — it is not an optimal assignment",
    )
    assert_true(
        m1.distance > m0.distance + 1e-6,
        "distance did NOT move under a shuffle (" + String(m0.distance)
        + " -> " + String(m1.distance) + ") — the probe is vacuous, or"
        " distance is being computed as a set metric",
    )


def test_segment_table() raises:
    """Every start must fit its 499 rows, and the next one must not."""
    print("[4] the segment table ...")
    assert_true(g1_n_segments(G1_SEG_ROWS - 1) == 0, "a short clip has no segment")
    assert_true(g1_n_segments(G1_SEG_ROWS) == 1, "exactly 499 rows is one segment")
    for i in range(12):
        var ep_len = 400 + i * 317
        var n = g1_n_segments(ep_len)
        if n == 0:
            continue
        var last = g1_segment_row(0, n - 1)
        assert_true(
            last + G1_SEG_ROWS <= ep_len,
            "ep_len " + String(ep_len) + ": segment " + String(n - 1)
            + " runs past the clip",
        )
        assert_true(
            g1_segment_row(0, n) + G1_SEG_ROWS > ep_len,
            "ep_len " + String(ep_len) + ": segment " + String(n)
            + " would have fitted and was not counted",
        )
    # the offset is the clip's first row
    assert_true(g1_segment_row(1234, 3) == 1234 + 3 * G1_SEG_STRIDE, "offset")


def test_segment_pick() raises:
    """`g1_segment_pick` decides WHICH windows the eval ever sees.

    Taking the first `n_take` of every clip reads ~0.19 low against the
    reference's all-862 number (docs §12.29), so the picks are stratified.
    Four properties, each of which a plausible wrong formula breaks:
      * FULL COVERAGE IS THE IDENTITY — `n_take == n_avail` must score every
        segment, or raising `--eval-segments` would start SKIPPING windows;
      * in range, so no pick runs past the clip;
      * strictly increasing, so no window is scored twice while another is
        never scored (a `k * n_avail // n_take` rule with rounding can tie);
      * at `n_take == 1` it is NOT segment 0 — that is the whole bias being
        removed, and a formula without the midpoint offset returns 0 here.
    """
    print("[5] the stratified segment picker ...")
    for n_avail in range(1, 40):
        # identity at full coverage
        for k in range(n_avail):
            assert_true(
                g1_segment_pick(n_avail, n_avail, k) == k,
                "full coverage must be the identity: n_avail "
                + String(n_avail) + " k " + String(k) + " -> "
                + String(g1_segment_pick(n_avail, n_avail, k)),
            )
        for n_take in range(1, n_avail + 1):
            var prev = -1
            for k in range(n_take):
                var got = g1_segment_pick(n_avail, n_take, k)
                assert_true(
                    got >= 0 and got < n_avail,
                    "pick out of range: " + String(got) + " of "
                    + String(n_avail),
                )
                assert_true(
                    got > prev,
                    "picks not strictly increasing at n_avail "
                    + String(n_avail) + " n_take " + String(n_take),
                )
                prev = got
    # one pick lands mid-clip, not at its opening ten seconds
    assert_true(
        g1_segment_pick(22, 1, 0) == 11,
        "a single pick must be the MIDDLE of the clip, got "
        + String(g1_segment_pick(22, 1, 0)),
    )
    print("      identity at full coverage, in range, strictly increasing,"
          " 1-of-22 -> segment", g1_segment_pick(22, 1, 0))


def main() raises:
    print("=== G1 tracking protocol ===")
    test_identity()
    test_proximity_ramp()
    test_emd_is_a_set_metric_and_distance_is_not()
    test_segment_table()
    test_segment_pick()
    print("=== all passed ===")
