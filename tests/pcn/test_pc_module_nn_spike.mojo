"""Phase A spike — PCN on nn storage + nn Adam (CPU).

Gate for the PCN → nn re-architecture's Phase A: prove that PCN's weight
slab can live in an nn `Param` and be trained by nn `Adam` driven from
PCN's own settling loop. A 2-block *linear* PCN (PCIdentity throughout) is
fit to a deterministic linear target `y = x @ W*`; a linear PC net can
represent this exactly, so the readout loss must fall sharply.

Validates:
  - `PCModule[*BLOCKS]` conforms to nn `Module` (Adam.make accepts it; the
    inherited `for_each_param` discovers the `weights` Param).
  - `make_pcn[PCXavier]` (vendored PCN-local initializer, legacy-`nn`-free).
  - `Adam.step` consumes `weights.grd` (filled by `weight_grad`, standard
    +∂E/∂W) and reduces the loss — confirming NO sign flip is needed.

Run:
    pixi run mojo run -I . tests/pcn/test_pc_module_nn_spike.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.nn.optimizer.adam import Adam

from noeira.experimental.pcn.pc_block import PCBlock
from noeira.experimental.pcn.predictive_model import PCIdentity
from noeira.experimental.pcn.pc_module import PCModule
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_module_trainer import pc_module_train_one_batch
from noeira.nn.core.ptr import mptr


def main() raises:
    comptime IN = 4
    comptime H = 8
    comptime OUT = 2
    comptime BATCH = 16
    comptime T_INFER = 20
    comptime N_STEPS = 300

    seed(0)

    # ── Synthetic linear regression data: y = x @ W_true ────────────────
    var x_s = List[Scalar[DT]](capacity=BATCH * IN)
    for _ in range(BATCH * IN):
        x_s.append(Scalar[DT](random_float64() * 2.0 - 1.0))

    var w_true = List[Scalar[DT]](capacity=IN * OUT)
    for _ in range(IN * OUT):
        w_true.append(Scalar[DT](random_float64() * 2.0 - 1.0))

    var y_s = List[Scalar[DT]](capacity=BATCH * OUT)
    for _ in range(BATCH * OUT):
        y_s.append(Scalar[DT](0))
    for b in range(BATCH):
        for j in range(OUT):
            var acc = Scalar[DT](0)
            for i in range(IN):
                acc += x_s[b * IN + i] * w_true[i * OUT + j]
            y_s[b * OUT + j] = acc

    var x_in = tile[BATCH, IN](mptr(x_s))
    var y_target = tile[BATCH, OUT](mptr(y_s))

    # ── PCN on nn storage + nn Adam ───────────────────────────────────
    comptime Net = PCModule[
        PCBlock[IN, H, PCIdentity],
        PCBlock[H, OUT, PCIdentity],
    ]
    var net = Net.make_pcn[PCXavier]()
    var opt = Adam(lr=Scalar[DT](1e-2))

    var loss_first = Float64(0.0)
    var loss_last = Float64(0.0)
    for step in range(N_STEPS):
        var r = pc_module_train_one_batch[BATCH](
            net, opt, x_in, y_target, T_INFER, Scalar[DT](0.1)
        )
        if step == 0:
            loss_first = r.output_loss_final
        if step == N_STEPS - 1:
            loss_last = r.output_loss_final

    print("readout loss: first =", loss_first, " last =", loss_last)

    # Finite + strong decrease (linear net fits a linear target).
    assert_true(loss_last == loss_last, "loss is NaN")
    assert_true(loss_last < loss_first, "loss did not decrease")
    assert_true(
        loss_last < loss_first * 0.1,
        "loss did not fall by ≥10× (Adam-over-PCN not learning)",
    )
    print("PASS: PCN trains on nn storage via nn Adam (Phase A spike)")
