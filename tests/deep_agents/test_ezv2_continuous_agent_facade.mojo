"""EZv2ContinuousAgent GPU facade smoke — the sampled-Gumbel agent gate.

Builds an `EZv2ContinuousAgent` on Pendulum from the `EZV2ContinuousMLPConfig`
bundle, runs a brief GPU sampled-Gumbel self-play train, reports a deterministic
greedy eval, and round-trips a device-aware checkpoint (all five nets). Proves
the continuous agent facade wires up end-to-end — train / eval_greedy / save /
load over the squashed-Gaussian head. NOT a convergence run.

Run (GPU env required):
    pixi run -e apple mojo run -I . \\
        tests/deep_agents/test_ezv2_continuous_agent_facade.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig, EZv2ContinuousAgent,
)
from noeira.envs.pendulum import PendulumEnv


def main() raises:
    print("test_ezv2_continuous_agent_facade ...")
    var ctx = DeviceContext()

    comptime Env = PendulumEnv[DType.float32]
    comptime Cfg = EZV2ContinuousMLPConfig[
        OBS=3, ACT_DIM=1, LATENT=16, HIDDEN=32, BINS=21,
        PROJ=32, PROJ_HID=32, BOTTLENECK=16,
    ]
    comptime Agent = EZv2ContinuousAgent[
        Env,
        Cfg.Rep, Cfg.Dyn, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        Cfg.OBS, Cfg.ACT_DIM, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=8, MAX_NODES=32, K_ROOT=4, K_NON_ROOT=2,
        CAP=5000, B=16, K=3, N=5,
    ]

    var env = Env()
    var agent = Agent(ctx=ctx, lr=Scalar[DT](3e-4))

    # ── short train on device: pipeline connects, loss finite ──
    var loss = agent.train(
        env,
        iterations=400,
        learning_starts=100,
        seed=7,
        max_ep_steps=200,
        verbose=False,
    )
    print("train loss:", loss)
    assert_true(
        loss == loss and loss < 1e30 and loss > -1e30,
        "facade GPU train loss not finite",
    )

    # ── deterministic greedy eval returns a sane (negative) Pendulum return ──
    var g = agent.eval_greedy(env, episodes=2, max_ep_steps=200)
    print("greedy eval:", g)
    assert_true(g == g and g <= 0.0 and g >= -3000.0, "greedy eval out of range")

    # ── device-aware checkpoint round-trip (byte-level) ──
    var path1 = String("/tmp/ezv2_cont_agent_ckpt1.txt")
    var path2 = String("/tmp/ezv2_cont_agent_ckpt2.txt")
    agent.save(path1)

    var agent2 = Agent(ctx=ctx, lr=Scalar[DT](3e-4))
    agent2.load(path1)
    agent2.save(path2)

    # storage agent checkpoint = 5 per-net sidecars (.rep/.dyn/.pred/.proj/.predh).
    # The sidecars are BINARY (checkpoint format), so compare bytes: reading
    # them with `f.read()` decodes UTF-8 and raises on the first non-text byte.
    for suf in [
        String(".rep"), String(".dyn"), String(".pred"),
        String(".proj"), String(".predh"),
    ]:
        var c1: List[UInt8]
        var c2: List[UInt8]
        with open(path1 + suf, "r") as f:
            c1 = f.read_bytes()
        with open(path2 + suf, "r") as f:
            c2 = f.read_bytes()
        # An empty sidecar would compare equal to another empty one.
        assert_true(len(c1) > 0, "empty checkpoint sidecar " + suf)
        assert_true(
            c1 == c2,
            "GPU checkpoint round-trip did not reproduce " + suf,
        )
    print("checkpoint round-trip: byte-identical (5 sidecars)")

    print("EZv2ContinuousAgent GPU facade smoke: OK")
