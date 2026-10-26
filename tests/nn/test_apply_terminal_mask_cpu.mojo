"""apply_terminal_mask CPU regression (audit L6).

`apply_terminal_mask` writes the full TD target in place:

    y[b] = r[b] + (1 - term[b]) * bootstrap[b]

dropping the bootstrap on natural termination (term=1) and keeping it on
time-limit truncation (term=0). This was a REAL past bug — SAC/DDPG/TD3
never masked the bootstrap, which silently broke Hopper (see
project_nn2_sac_terminal_bootstrap_fix). There was no dedicated unit
test for the helper; this is it.

Gates:
  * terminal sample (term=1)  ⇒ y == r exactly (bootstrap dropped)
  * non-terminal (term=0)     ⇒ y == r + bootstrap exactly

Run: `pixi run mojo run -I . tests/nn/test_apply_terminal_mask_cpu.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.deep_agents.training.terminal_mask import apply_terminal_mask


def main() raises:
    print("=" * 70)
    print("apply_terminal_mask CPU regression (L6)")
    print("=" * 70)

    comptime N = 4
    var r: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N).as_unsafe_any_origin()
    var term: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N).as_unsafe_any_origin()
    var y: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N).as_unsafe_any_origin()

    # Per-sample (reward, termination, bootstrap-entering-y).
    var rew = Array[Scalar[DT], N](fill=0)
    rew[0] = 1.0; rew[1] = 2.0; rew[2] = -1.0; rew[3] = 0.5
    var tm = Array[Scalar[DT], N](fill=0)
    tm[0] = 1.0; tm[1] = 0.0; tm[2] = 1.0; tm[3] = 0.0  # term, non, term, non
    var boot = Array[Scalar[DT], N](fill=0)
    boot[0] = 10.0; boot[1] = -5.0; boot[2] = 3.0; boot[3] = 7.0

    for i in range(N):
        r[i] = rew[i]
        term[i] = tm[i]
        y[i] = boot[i]  # y enters holding ONLY the bootstrap term

    # The helper takes `LayoutTensor` views, not raw pointers.
    var r_t = tile[N](r)
    var term_t = tile[N](term)
    var y_t = tile[N, 1](y)
    apply_terminal_mask["cpu", N](None, r_t, term_t, y_t)

    for i in range(N):
        if tm[i] == Scalar[DT](1.0):
            print("  [term] y[", i, "] =", y[i], " expect r =", rew[i])
            assert_true(
                y[i] == rew[i],
                "terminal sample must drop the bootstrap (y == r)",
            )
        else:
            var expect = rew[i] + boot[i]
            print("  [non ] y[", i, "] =", y[i], " expect r+boot =", expect)
            assert_true(
                y[i] == expect,
                "non-terminal sample must keep the bootstrap (y == r+boot)",
            )

    r.free(); term.free(); y.free()
    print("=" * 70)
    print("PASS — terminal bootstrap masking is correct")
    print("=" * 70)
