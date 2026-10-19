"""MuZeroAgent GPU facade smoke — the CPU-search / GPU-train hybrid driver gate.

The GPU twin of `test_mz_agent_facade`: builds a `TARGET="gpu"` MuZeroAgent on
CartPole, runs a brief self-play train (search on the CPU mirror, K-step BPTT
unroll on the device, mirror re-synced after each train step), reports a greedy
eval, and round-trips a device-aware checkpoint. Proves the whole hybrid wires
up end-to-end — the GPU `train` / `eval_greedy` / `save` / `load` branches that
used to `comptime assert False` (#24). NOT a convergence run.

Run (GPU env required):
    pixi run -e apple mojo run -I . tests/deep_agents/test_mz_agent_facade_gpu.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.muzero import MuZeroMLPConfig, MuZeroAgent
from noeira.envs.cartpole import CartPoleEnv


def main() raises:
    print("test_mz_agent_facade_gpu ...")
    var ctx = DeviceContext()

    comptime Env = CartPoleEnv[DType.float64]
    comptime Cfg = MuZeroMLPConfig[OBS=4, ACT=2, LATENT=16, HIDDEN=32, BINS=51]
    comptime Agent = MuZeroAgent[
        "gpu", Env,
        Cfg.Rep, Cfg.Dyn, Cfg.Pred,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=12, MAX_NODES=48, CAP=20000, B=8, K=3, N=3,
    ]

    var env = Env()
    var agent = Agent(ctx=ctx, lr=Scalar[DT](0.01))

    # ── short train on device: pipeline connects, loss finite ──
    var loss = agent.train(
        env,
        iterations=400,
        learning_starts=200,
        seed=7,
        verbose=False,
    )
    print("train loss:", loss)
    assert_true(
        loss == loss and loss < 1e30 and loss > 0.0,
        "facade GPU train loss not finite/positive",
    )

    # ── greedy eval (CPU mirror synced from device) returns sane magnitude ──
    var g = agent.eval_greedy(env, episodes=3)
    print("greedy eval:", g)
    assert_true(g == g and g >= 1.0 and g <= 500.0, "greedy eval out of range")

    # ── device-aware checkpoint round-trip (byte-level) ──
    var path1 = String("/tmp/mz_agent_facade_gpu_ckpt1.txt")
    var path2 = String("/tmp/mz_agent_facade_gpu_ckpt2.txt")
    agent.save(path1)

    var agent2 = Agent(ctx=ctx, lr=Scalar[DT](0.01))
    agent2.load(path1)
    agent2.save(path2)

    # The storage agent checkpoint is 3 per-net sidecars (path + .rep/.dyn/.pred);
    # compare each to prove the save/load round-trip reproduces serialization.
    # The sidecars are BINARY (checkpoint format), so compare bytes: reading
    # them with `f.read()` decodes UTF-8 and raises on the first non-text byte.
    for suf in [String(".rep"), String(".dyn"), String(".pred")]:
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
            "checkpoint round-trip did not reproduce " + suf,
        )
    print("checkpoint round-trip: byte-identical (3 sidecars)")

    print("MuZeroAgent GPU facade smoke: OK")
