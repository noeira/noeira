"""PushT sim → LeWM-encoder bridge: render + HWC→CHW correctness (CPU).

Validates `sim_frame_chw_norm` against a direct HWC render: the CHW [0,1]
output must equal HWC[0,255]/255 element-for-element, and the rendered scene
must actually contain the goal-T (green) over a white background (the
renderer is drawing, not blank). No GPU / no world model.

Run:  pixi run mojo run -I . tests/experimental/lewm/test_pusht_sim_bridge.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT
from noeira.envs.pusht.render_swm import render_pusht_swm_at  # what the bridge draws with
from noeira.experimental.lewm.pusht_sim_bridge import sim_frame_chw_norm


comptime OUT = 16
comptime HW = OUT * OUT


def main() raises:
    print("=" * 70)
    print("PushT sim → encoder bridge (render + HWC→CHW)")
    print("=" * 70)

    var bcx = Scalar[DT](256.0)
    var bcy = Scalar[DT](256.0)
    var bang = Scalar[DT](0.5)
    var acx = Scalar[DT](120.0)
    var acy = Scalar[DT](120.0)

    # reference HWC [0,255]
    var hwc = alloc[Scalar[DT]](HW * 3).as_unsafe_any_origin()
    var hwc_t = LayoutTensor[DT, Layout.row_major(OUT, OUT, 3), MutAnyOrigin](hwc)
    render_pusht_swm_at[OUT](bcx, bcy, bang, acx, acy, hwc_t)

    # bridge CHW [0,1]
    var chw = alloc[Scalar[DT]](3 * HW).as_unsafe_any_origin()
    sim_frame_chw_norm[OUT](bcx, bcy, bang, acx, acy, chw)

    # permute + scale must be exact
    var maxd: Scalar[DT] = 0.0
    var all_in_range = True
    for c in range(3):
        for y in range(OUT):
            for x in range(OUT):
                var got = chw[c * HW + y * OUT + x]
                var want = hwc[(y * OUT + x) * 3 + c] / Scalar[DT](255.0)
                var d = (got - want).__abs__()
                if d > maxd:
                    maxd = d
                if got < Scalar[DT](-1e-6) or got > Scalar[DT](1.0 + 1e-6):
                    all_in_range = False
    print("   max|chw - hwc/255| =", maxd)
    assert_true(maxd < Scalar[DT](1e-7), "HWC→CHW permute+scale exact")
    assert_true(all_in_range, "CHW values in [0,1]")

    # scene is actually drawn: some green (goal-T) and some white (bg)
    var has_green = False
    var has_white = False
    for y in range(OUT):
        for x in range(OUT):
            var r = hwc[(y * OUT + x) * 3 + 0]
            var g = hwc[(y * OUT + x) * 3 + 1]
            var b = hwc[(y * OUT + x) * 3 + 2]
            # light green (144,238,144): G high, R/B lower
            if g > Scalar[DT](200.0) and r < Scalar[DT](200.0):
                has_green = True
            if (
                r > Scalar[DT](250.0) and g > Scalar[DT](250.0)
                and b > Scalar[DT](250.0)
            ):
                has_white = True
    assert_true(has_green, "goal-T rendered (green present)")
    assert_true(has_white, "background present (white)")

    hwc.free(); chw.free()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
