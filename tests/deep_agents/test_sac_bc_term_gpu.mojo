"""The behaviour-cloning term of `SACActorLoss` ON THE DEVICE — `set_bc` and
`set_bc_weight` through the graph the GPU actually runs.

    pixi run -e apple mojo run -I . tests/deep_agents/test_sac_bc_term_gpu.mojo

The CPU twin (`test_sac_bc_term.mojo`) checks the arithmetic. This one
checks the WIRING: on the GPU the `bc_w` Scale node reads λ from a device
word handed to it once in `make`, and a setter that replaced that buffer
instead of writing into it left the node reading the original zero for a
whole run — 25k steps with the BC term silently off (eval 360, teacher-forced
L1 0.14 where the run before had 0.005). So: λ = 0 then λ > 0 must MOVE the
device loss, and `set_bc_weight(2λ)` must move it by twice as much. The
actor's Adam runs at 1e-9 so the three forwards see the same actor, and
the entropy term is zeroed so the device loss is the BC term alone.
"""

from std.testing import assert_almost_equal, assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.core.tensor import Tensor
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


def _loss(
    mut blk: BLK, mut actor: ACTOR, mut opt: Adam, mut c1: CRITIC, mut c2: CRITIC,
    mut mb_s: Tensor, mut mb_a: Tensor, ctx: DeviceContext,
) raises -> Float64:
    """One device forward/backward; the window-mean loss the device
    accumulator holds, drained and reset."""
    blk.reset_loss_accum()
    _ = blk.forward_backward["gpu"](actor, opt, c1, c2, mb_s, mb_a, Scalar[DT](0.0), ctx)
    return Float64(blk.read_loss_accum(ctx))


def main() raises:
    print("SAC BC term (GPU wiring) ...")
    with DeviceContext() as ctx:
        var blk = BLK.make["gpu"](ctx, action_scale=1.0)
        # ⚠ On the GPU the `alogp` node is NOT baked from the `alpha` argument
        # (the trainer wires a device α); its make-time multiplier would put
        # the rsample-noisy log-prob into every loss. Zero it: the device loss
        # is then the BC term alone, deterministic (the L1 is on tanh(mu)).
        blk.graph.set_node_attr["alogp", "multiplier"](Scalar[DT](0.0))
        var actor = ACTOR.make["gpu", Xavier](ctx)
        var c1 = CRITIC.make["gpu", Zero](ctx)
        var c2 = CRITIC.make["gpu", Zero](ctx)
        var opt = Adam(lr=1e-9)
        var mb_s = Tensor.alloc(BATCH * OBS)
        for i in range(BATCH * OBS):
            mb_s.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
        mb_s.upload(ctx)
        var mb_a = Tensor.alloc(BATCH * ACT)
        for i in range(BATCH * ACT):
            mb_a.data[i] = Scalar[DT](0.6 if (i % 3) == 0 else -0.5)
        mb_a.upload(ctx)

        # 1. λ = 0 — the plain loss (critics at zero: the entropy term alone)
        blk.set_bc(Scalar[DT](0.0), N_DEMO, ctx)
        var l0 = _loss(blk, actor, opt, c1, c2, mb_s, mb_a, ctx)
        # 2. λ through `set_bc`
        blk.set_bc(LAMBDA, N_DEMO, ctx)
        var l1 = _loss(blk, actor, opt, c1, c2, mb_s, mb_a, ctx)
        # 3. 2λ through `set_bc_weight` — the after-the-first-step path
        blk.set_bc_weight(LAMBDA * Scalar[DT](2.0), ctx)
        var l2 = _loss(blk, actor, opt, c1, c2, mb_s, mb_a, ctx)
        var d1 = l1 - l0
        var d2 = l2 - l0
        print("  device loss  λ=0:", l0, " λ:", l1, " 2λ:", l2, "  increments", d1, d2)
        assert_true(d1 > 0.05, "λ > 0 moves the DEVICE loss (the bc_w node reads the device word)")
        assert_almost_equal(d2 / d1, 2.0, atol=0.02, msg="set_bc_weight(2λ) doubles the increment")
        # 4. and back to 0 through the setter
        blk.set_bc_weight(Scalar[DT](0.0), ctx)
        var l3 = _loss(blk, actor, opt, c1, c2, mb_s, mb_a, ctx)
        assert_almost_equal(l3, l0, atol=1e-4, msg="set_bc_weight(0) restores the plain loss")
    print("SAC BC TERM GPU OK")
