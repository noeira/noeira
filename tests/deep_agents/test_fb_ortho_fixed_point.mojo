"""`L_ortho` must reach its own fixed point — the gate that FOLLOWS THE GRADIENT.

`test_fb_loss.mojo` already checks that the ortho gradient matches the ortho
loss ([2]) and that a rank-1 `B` scores worse than a spread one ([4]). Both
passed for months against a version of `L_ortho` that ran the pairwise dot
across TWO INDEPENDENT `B` batches — an objective whose minimiser is not "B
isotropic" but "let the two batches span orthogonal subspaces", i.e. a
rank-≤d/2 collapse of both. [2] could not see it because the gradient was a
faithful gradient OF THE WRONG LOSS; [4] could not see it because it scores two
hand-built points and never visits the minimiser. It cost the first G1 run
(`docs/BFM_ZERO_G1_REPRODUCTION.md` §12.12-§12.13).

So this gate optimises. `B`'s rows are the free parameters, renormalised to √d
each step exactly as the net's own sphere projection does, and the assertion is
on where descent ARRIVES:

    rank_eff(C) = tr(C)^2 / tr(C^2),  C = (1/n)·BᵀB     must reach d

That is the participation-ratio effective rank of the second-moment matrix: d
when `E[BBᵀ] = I`, and 1 when every row shares a direction. It is the quantity
BFM-Zero's own shipped log pins for 200 M steps
(`references/BFM-Zero-main/released/new_model/train_log.txt`:
`orth_loss_offdiag` = 128.3 = exactly 0.5·d at d = 256).

BATCH/D is 4 here, matching the run's 1024/256.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_ortho_fixed_point.mojo
"""

from std.math import sqrt
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.fb.loss import fb_ortho_loss, fb_rank_eff_from_ortho


comptime D: Int = 32
comptime BATCH: Int = 128
comptime STEPS: Int = 600
comptime LR: Float64 = 2.0
# Descent from a random start must land within 1 % of full rank. The broken
# two-tensor form landed at 11.3 / 32 — a third — so the bar does not need to
# be tight to be decisive, and a loose bar keeps it robust to the step size.
comptime RANK_TOL: Float64 = 0.01


def _renorm(mut b: Tensor):
    """The sphere projection `BFMBNet` ends with: every row to norm √D."""
    for i in range(BATCH):
        var s = Float64(0)
        for k in range(D):
            var v = Float64(b.data[i * D + k])
            s += v * v
        var inv = sqrt(Float64(D)) / sqrt(s + 1e-12)
        for k in range(D):
            b.data[i * D + k] = Scalar[DT](Float64(b.data[i * D + k]) * inv)


def _rand_fill(mut b: Tensor, seed: UInt64):
    var s = seed
    for i in range(BATCH * D):
        s = s * 6364136223846793005 + 1442695040888963407
        b.data[i] = Scalar[DT](
            Float64((s >> 11) & 0xFFFFFFFF) / Float64(0xFFFFFFFF) - 0.5
        )


def _rank_eff(ref b: Tensor) -> Float64:
    """`tr(C)^2 / tr(C^2)` for `C = (1/n)·BᵀB`, computed exactly."""
    var c = Tensor.alloc(D * D)
    for a in range(D):
        for bb in range(D):
            var s = Float64(0)
            for i in range(BATCH):
                s += Float64(b.data[i * D + a]) * Float64(b.data[i * D + bb])
            c.data[a * D + bb] = Scalar[DT](s / Float64(BATCH))
    var tr = Float64(0)
    var tr2 = Float64(0)
    for a in range(D):
        tr += Float64(c.data[a * D + a])
        for bb in range(D):
            var v = Float64(c.data[a * D + bb])
            tr2 += v * v
    return tr * tr / tr2


def test_rank_from_ortho_matches_direct() raises:
    """`fb_rank_eff_from_ortho` must agree with rank computed FROM B.

    The training loop never sees B — it logs `b_rank_eff` from the ortho loss
    alone, because `|B|` is pinned and cannot show a directional collapse
    (§12.12). That conversion depends on `L_ortho`'s exact definition, and
    when the loss moved to the reference's scale (§12.28) the inline copy in
    the driver was missed: it logged 200.8 where the truth was 244.3 — which
    reads exactly like B collapsing. Nothing caught it because nothing
    compared the derived number against one computed independently.

    This does. Two routes to the same quantity, on three B's spanning the
    range: a random spread one, a renormalised rank-1 collapse, and the
    descended near-isotropic one.
    """
    print("[0] rank from L_ortho == rank computed from B ...")
    var g = Tensor.alloc(BATCH * D)
    for ci in range(3):
        var name = String("random")
        if ci == 1:
            name = String("rank-1")
        elif ci == 2:
            name = String("descended")
        var b = Tensor.alloc(BATCH * D)
        if ci == 1:
            # rank-1: every row the same direction, plus a whisper of noise so
            # the Gram is not exactly singular (mirrors [2]'s start)
            var st = UInt64(4242)
            for k in range(D):
                st = st * 6364136223846793005 + 1442695040888963407
                var v = Float64((st >> 33) % 2000) / 1000.0 - 1.0
                for i in range(BATCH):
                    b.data[i * D + k] = Scalar[DT](v)
            for i in range(BATCH * D):
                st = st * 6364136223846793005 + 1442695040888963407
                b.data[i] = Scalar[DT](
                    Float64(b.data[i])
                    + 1e-3 * (Float64((st >> 33) % 2000) / 1000.0 - 1.0)
                )
        else:
            _rand_fill(b, UInt64(777 + ci))
        _renorm(b)
        if ci == 2:
            for _ in range(STEPS):
                _ = fb_ortho_loss[D, BATCH](b, g)
                for i in range(BATCH * D):
                    b.data[i] = Scalar[DT](
                        Float64(b.data[i]) - LR * Float64(g.data[i])
                    )
                _renorm(b)
        var direct = _rank_eff(b)
        var ortho = fb_ortho_loss[D, BATCH](b, g)
        var derived = fb_rank_eff_from_ortho[D, BATCH](ortho)
        var rel = abs(derived - direct) / direct
        print("      ", name, ": direct", direct, " from ortho", derived,
              " rel", rel)
        assert_true(
            rel < 1e-6,
            "rank from L_ortho (" + String(derived) + ") disagrees with rank"
            " computed from B (" + String(direct) + ") on the " + name
            + " case — the conversion has drifted from the loss",
        )
    print("      OK")


def test_ortho_descends_to_full_rank() raises:
    print("[1] L_ortho descends to rank_eff = D ...")
    var b = Tensor.alloc(BATCH * D)
    _rand_fill(b, 12345)
    _renorm(b)
    var g = Tensor.alloc(BATCH * D)

    var start = _rank_eff(b)
    print("      start  rank_eff =", start, " of", D)

    var prev = Float64(0)
    for step in range(STEPS):
        _ = fb_ortho_loss[D, BATCH](b, g)
        for i in range(BATCH * D):
            b.data[i] = Scalar[DT](
                Float64(b.data[i]) - LR * Float64(g.data[i])
            )
        _renorm(b)
        if step == STEPS // 2:
            prev = _rank_eff(b)

    var got = _rank_eff(b)
    print("      final  rank_eff =", got, " of", D)
    assert_true(
        got > Float64(D) * (1.0 - RANK_TOL),
        "L_ortho descended to rank_eff " + String(got) + " of " + String(D)
        + " — it is a COLLAPSE objective, not a regulariser. The known cause"
        " is running the pairwise dot across two independent B batches"
        " instead of one batch against itself; see loss.mojo's header.",
    )
    # and it must STAY there — a term that overshoots is as bad as one that
    # undershoots, and the reference's log is flat for 200 M steps.
    assert_true(
        prev > Float64(D) * (1.0 - RANK_TOL),
        "rank_eff was " + String(prev) + " at the halfway mark and "
        + String(got) + " at the end — not a fixed point",
    )


def test_a_collapsed_start_is_recovered() raises:
    """From a rank-1 start, the term must CLIMB back to full rank.

    Descending to `d` from a random start only shows the term does not break
    something already healthy. The regulariser's actual job is to pull B OUT
    of a collapse, so start it in one.
    """
    print("[2] L_ortho climbs out of a rank-1 start ...")
    var b = Tensor.alloc(BATCH * D)
    # every row the same direction, plus a whisper of noise so the gradient
    # is not exactly zero by symmetry
    var s = UInt64(777)
    for i in range(BATCH):
        for k in range(D):
            s = s * 6364136223846793005 + 1442695040888963407
            var n = Float64((s >> 11) & 0xFFFF) / Float64(0xFFFF) - 0.5
            b.data[i * D + k] = Scalar[DT](
                (1.0 if k == 0 else 0.0) + 1e-3 * n
            )
    _renorm(b)
    var g = Tensor.alloc(BATCH * D)

    var start = _rank_eff(b)
    print("      start  rank_eff =", start, " of", D)
    assert_true(start < 1.5, "the probe did not start collapsed: " + String(start))

    for _ in range(STEPS * 4):
        _ = fb_ortho_loss[D, BATCH](b, g)
        for i in range(BATCH * D):
            b.data[i] = Scalar[DT](
                Float64(b.data[i]) - LR * Float64(g.data[i])
            )
        _renorm(b)

    var got = _rank_eff(b)
    print("      final  rank_eff =", got, " of", D)
    assert_true(
        got > Float64(D) * (1.0 - RANK_TOL),
        "from a rank-1 B, L_ortho only recovered to rank_eff " + String(got)
        + " of " + String(D) + " — it cannot undo the collapse it exists for",
    )


def main() raises:
    print("=== FB ortho fixed point ===")
    test_rank_from_ortho_matches_direct()
    test_ortho_descends_to_full_rank()
    test_a_collapsed_start_is_recovered()
    print("=== all passed ===")
