"""The behaviour-cloning term of `SACActorLoss` — `set_bc` on a small CPU graph.

    pixi run mojo run -I . tests/deep_agents/test_sac_bc_term.mojo

With λ = 0 the loss must equal the plain SAC loss; with λ > 0 on the first
`n` rows it must move by exactly λ · mean_b(mask_b · L1_b), where
L1_b = (1/ACT) Σ_j |tanh(mu_j(s_b)) − a_bj| is recomputed here from the
actor's own mean; and the actor must MOVE toward the demo actions when the
term is the only signal (critics frozen at zero: min_q constant).
"""

from std.math import tanh
from std.testing import assert_almost_equal, assert_true

from noeira.nn.constants import DT
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Xavier, Zero
from noeira.nn.optimizer.adam import Adam
from noeira.deep_agents.primitives.stochastic_actor import StochasticActor
from noeira.deep_agents.sac.actor_loss import SACActorLoss

comptime OBS = 3
comptime ACT = 2
comptime SA = OBS + ACT
comptime H = 16
comptime BATCH = 8
comptime N_DEMO = 4
comptime LAMBDA = Scalar[DT](10.0)
comptime ACTOR = StochasticActor[OBS, ACT, LinearReLU[OBS, H], LinearReLU[H, H]]
comptime CRITIC = Sequential[LinearReLU[SA, H], Linear[H, 1]]
comptime BLK = SACActorLoss[ACTOR, CRITIC, BATCH]


def _mean_l1(mut actor: ACTOR, mut mb_s: Tensor, mut mb_a: Tensor) raises -> Float64:
    """mean over the masked rows of (1/ACT) Σ|tanh(mu) − a|, then / BATCH."""
    var out = Tensor.alloc(BATCH * 2 * ACT)
    actor.forward["cpu", BATCH](TensorRefs[1](mb_s), out)
    var s = 0.0
    for b in range(N_DEMO):
        var l1 = 0.0
        for j in range(ACT):
            l1 += abs(Float64(tanh(out.data[b * 2 * ACT + j])) - Float64(mb_a.data[b * ACT + j]))
        s += l1 / Float64(ACT)
    return s / Float64(BATCH)


def main() raises:
    print("SAC BC term (CPU) ...")
    var blk = BLK.make["cpu"](None, action_scale=1.0)
    var actor = ACTOR.make["cpu", Xavier](None)
    var c1 = CRITIC.make["cpu", Zero](None)
    var c2 = CRITIC.make["cpu", Zero](None)
    var opt = Adam(lr=1e-2)
    var mb_s = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        mb_s.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var mb_a = Tensor.alloc(BATCH * ACT)
    for i in range(BATCH * ACT):
        mb_a.data[i] = Scalar[DT](0.6 if (i % 3) == 0 else -0.5)

    # 1. λ = 0: the BC nodes are inert
    var out0 = blk.forward_backward["cpu"](actor, opt, c1, c2, mb_s, mb_a, Scalar[DT](0.0), None)
    blk.set_bc(Scalar[DT](0.0), N_DEMO, None)
    var out0b = blk.forward_backward["cpu"](actor, opt, c1, c2, mb_s, mb_a, Scalar[DT](0.0), None)
    print("  loss with BC nodes at 0:", out0.loss, "then mask on, λ 0:", out0b.loss)
    assert_almost_equal(Float64(out0.loss), Float64(out0b.loss), atol=1e-6, msg="λ 0 is inert")

    # 2. λ > 0 on the first N_DEMO rows: the loss moves by λ·mean(mask·L1)
    blk.set_bc(LAMBDA, N_DEMO, None)
    # the reference BEFORE the call: forward_backward steps the actor's Adam
    var l1 = _mean_l1(actor, mb_s, mb_a)
    var out1 = blk.forward_backward["cpu"](actor, opt, c1, c2, mb_s, mb_a, Scalar[DT](0.0), None)
    var expect = Float64(out0b.loss) + Float64(LAMBDA) * l1
    print("  loss with λ", LAMBDA, ":", out1.loss, " expected", expect, " (mean masked L1 / B =", l1, ")")
    assert_true(l1 > 0.05, "the actor does not already match the demo actions")
    assert_almost_equal(Float64(out1.loss), expect, atol=1e-4, msg="loss = plain + λ·mean(mask·L1)")

    # 3. the gradient pulls tanh(mu) toward the demo actions (critics zero)
    var before = l1
    for _ in range(60):
        _ = blk.forward_backward["cpu"](actor, opt, c1, c2, mb_s, mb_a, Scalar[DT](0.0), None)
    var after = _mean_l1(actor, mb_s, mb_a)
    print("  masked L1 after 60 BC steps:", before, "->", after)
    assert_true(after < 0.5 * before, "BC pulls the actor toward the demo actions")

    # ── 4. q weight 0: the loss is the BC term alone ─────────────────────
    blk.set_q_weight(Scalar[DT](0.0))
    var l1_only = _mean_l1(actor, mb_s, mb_a)
    var out_only = blk.forward_backward["cpu"](actor, opt, c1, c2, mb_s, mb_a, Scalar[DT](0.0), None)
    print("  loss with q weight 0:", out_only.loss, " expected", Float64(LAMBDA) * l1_only)
    assert_almost_equal(Float64(out_only.loss), Float64(LAMBDA) * l1_only, atol=1e-4,
                        msg="q weight 0 => loss = λ·mean(mask·L1) alone")
    # ── 5. set_bc_weight changes λ alone: the mask stays, the loss scales ─
    blk.set_bc_weight(LAMBDA * Scalar[DT](2.0), None)
    var l1_twice = _mean_l1(actor, mb_s, mb_a)
    var out_twice = blk.forward_backward["cpu"](actor, opt, c1, c2, mb_s, mb_a, Scalar[DT](0.0), None)
    print("  loss with λ doubled by set_bc_weight:", out_twice.loss, " expected", 2.0 * Float64(LAMBDA) * l1_twice)
    assert_almost_equal(Float64(out_twice.loss), 2.0 * Float64(LAMBDA) * l1_twice, atol=1e-4,
                        msg="set_bc_weight(2λ) => loss = 2λ·mean(mask·L1)")
    assert_true(Float64(blk.bc_weight) == 2.0 * Float64(LAMBDA), "bc_weight field follows")
    blk.set_bc_weight(LAMBDA, None)
    blk.set_q_weight(Scalar[DT](1.0))
    print("SAC BC TERM OK (q weight + set_bc_weight legs too)")
    print("SAC BC TERM OK")
