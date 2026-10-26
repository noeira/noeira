"""MuZeroAgent facade smoke — construct, short train, greedy eval, save/load.

Exercises the Phase-B #27 facade end-to-end on CartPole (CPU): the agent builds
its three nets, runs a brief self-play train (pipeline connects, loss finite),
reports a greedy eval, and round-trips a checkpoint (load reproduces the saved
greedy return). NOT a convergence run — that is the separate
`muzero_cartpole_v2_cpu` example (#28, already solved).

Run (no GPU):
    pixi run mojo run -I . tests/deep_agents/test_mz_agent_facade.mojo
"""

from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.deep_agents.muzero import MuZeroMLPConfig, MuZeroAgent
from noeira.envs.cartpole import CartPoleEnv


def main() raises:
    comptime Env = CartPoleEnv[DType.float64]
    comptime Cfg = MuZeroMLPConfig[OBS=4, ACT=2, LATENT=16, HIDDEN=32, BINS=51]
    comptime Agent = MuZeroAgent[
        "cpu", Env,
        Cfg.Rep, Cfg.Dyn, Cfg.Pred,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=12, MAX_NODES=48, CAP=20000, B=8, K=3, N=3,
    ]

    var env = Env()
    var agent = Agent(ctx=None, lr=Scalar[DT](0.01))

    # ── short train: pipeline connects, loss finite ──
    var loss = agent.train(
        env,
        iterations=700,
        learning_starts=200,
        seed=7,
        verbose=False,
    )
    print("train loss:", loss)
    assert_true(
        loss == loss and loss < 1e30 and loss > 0.0,
        "facade train loss not finite/positive",
    )

    # ── greedy eval runs + returns a sane CartPole magnitude ──
    var g = agent.eval_greedy(env, episodes=3)
    print("greedy eval:", g)
    assert_true(g == g and g >= 1.0 and g <= 500.0, "greedy eval out of range")

    # ── checkpoint round-trip (byte-level): save → load → save reproduces the
    # exact serialization. (CartPole reset is stochastic, so comparing rollout
    # returns across reloads is not a valid fidelity check — compare the bytes.)
    var path1 = String("/tmp/mz_agent_facade_ckpt1.txt")
    var path2 = String("/tmp/mz_agent_facade_ckpt2.txt")
    agent.save(path1)

    var agent2 = Agent(ctx=None, lr=Scalar[DT](0.01))
    agent2.load(path1)
    agent2.save(path2)

    # The storage agent checkpoint is 3 per-net sidecars (path + .rep/.dyn/.pred).
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

    print("MuZeroAgent facade smoke: OK")
