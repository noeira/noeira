""""The actor sees `state | last_action | history | z` — and NOT `privileged`.

`released/new_model/config.json` filters the observation dict per net. The
actor's keys are `state, last_action, history_actor`; `f`, `critic` and
`aux_critic` additionally get `privileged_state`, and `b` / `discriminator` get
`state + privileged_state` ONLY (docs §12.34). Our packed row is

    [ state 64 | privileged 463 | last_action 29 | history 372 | z 256 ]

so `BFMActorView` must drop the middle block. The privileged block sits in the
middle precisely so `b` and `discriminator` keep reading `[0, 527)` — a layout
choice that is invisible unless something checks WHICH elements arrive.

⚠ A shape check alone is vacuous here: `Parallel[Slice, Slice]` of the wrong
two ranges has the right width. This gates the VALUES — each element of the
packed row is its own index, so the forwarded vector must read
`[0..64) ++ [527..1184)` exactly, and the 463 privileged indices must be
ABSENT.

Run: pixi run -e apple mojo run -I . tests/fb/test_bfm_actor_view.mojo
"""

from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.deep_agents.fb.bfm_towers import BFMActorView

comptime SD = 64        # state
comptime PRIV = 463     # privileged_state — what the actor must NOT see
comptime EX = 401       # last_action 29 + history 372
comptime OBSF = SD + PRIV + EX      # 928
comptime D = 256
comptime BATCH = 3
comptime WIN = OBSF + D             # 1184
comptime WOUT = SD + EX + D         # 721


def main() raises:
    print("BFMActorView: the actor's keys, values checked")
    comptime View = BFMActorView[OBSF, SD, EX, D]
    var v = View.make["cpu", Deterministic](None)

    var x = Tensor.alloc(BATCH * WIN)
    for b in range(BATCH):
        for i in range(WIN):
            # each element IS its index (offset per row so a row mix-up shows)
            x.data[b * WIN + i] = Scalar[DT](Float64(i) + 1000.0 * Float64(b))
    var y = Tensor()
    y.ensure(BATCH * WOUT)
    v.forward["cpu", BATCH](TensorRefs[1, MutAnyOrigin](x), y, None)

    print("  in", WIN, " out", WOUT, " (state", SD, "+ last+hist", EX,
          "+ z", D, ")")
    var bad = 0
    var saw_priv = 0
    for b in range(BATCH):
        for j in range(WOUT):
            var got = Float64(y.data[b * WOUT + j]) - 1000.0 * Float64(b)
            # expected source index: [0,SD) then [OBSF-EX, OBSF+D)
            var want = Float64(j) if j < SD else Float64(OBSF - EX + (j - SD))
            if abs(got - want) > 1e-6:
                bad += 1
            # any index inside the privileged block is a leak
            if got >= Float64(SD) - 0.5 and got < Float64(SD + PRIV) - 0.5:
                saw_priv += 1
    print("  wrong elements", bad, "of", BATCH * WOUT,
          "   privileged indices leaked", saw_priv)
    assert_true(
        bad == 0,
        "the forwarded vector is not [0,64) ++ [527,1184) — a Parallel of the"
        " wrong two Slices has the right WIDTH",
    )
    assert_true(
        saw_priv == 0,
        "a privileged_state index reached the actor: the reference's actor"
        " filter is `state, last_action, history_actor` and excludes it",
    )
    # NON-VACUITY: the privileged block must be non-empty and reachable, or
    # "no leak" is true of any slicing at all
    assert_true(PRIV > 0 and OBSF > SD + EX, "vacuous: no privileged block")
    print("BFM_ACTOR_VIEW OK")
