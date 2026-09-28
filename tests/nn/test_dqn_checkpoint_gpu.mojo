"""DQN GPU checkpoint round-trip test (storage nn).

Validates that a GPU-trained DQN trainer can `save_state` and a fresh GPU
trainer can `load_state` it, restoring Q-net params + state + the ε-greedy
exploration scalars + the cumulative train-step counter across the
device→host→device round-trip.

STORAGE migration note: the storage checkpoint persists the ONLINE Q-net
**params + state + ε + counter**, but NOT the optimizer moments (resume
re-warms Adam — the same design choice as the storage SAC checkpoint). So this
test no longer asserts an Adam m/v/bc round-trip (the legacy invariant 3);
greedy-action agreement + GPU→CPU interchange remain the behavioural guarantees,
and re-save byte-identity proves the param/ε/counter path is exact.

Invariants:
  1. **Re-save byte-identity**: save → load into a fresh trainer → save again
     ⇒ the two files are byte-for-byte identical (every persisted field — Q-net
     params + state + ε + counter — survived the round-trip exactly).
  2. **Greedy-action agreement**: the loaded trainer picks the same greedy
     action as the original on a battery of observations.
  3. **GPU→CPU interchange**: a CPU trainer loads the GPU checkpoint and picks
     the same greedy actions (train-on-GPU → eval-on-CPU).

Guards on a DeviceContext probe so it no-ops on CPU-only CI.

Run: pixi run -e apple mojo run -I . tests/nn/test_dqn_checkpoint_gpu.mojo
"""

from std.random import seed
from max.gpu.host import DeviceContext
from noeira.io.sha256 import sha256_file
from std.testing import assert_true, assert_equal

from noeira.nn.constants import DT
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.deep_agents.dqn.trainer import DQNTrainer
from noeira.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
)
from noeira.deep_agents.training.blocks import (
    UniformSampleGpuStep, UniformSampleCpuStep,
)

from noeira.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500

comptime QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS],
]

comptime CKPT_A = String("/tmp/dqn_gpu_ckpt_a.ckpt")
comptime CKPT_B = String("/tmp/dqn_gpu_ckpt_b.ckpt")


def _read_file(path: String) raises -> String:
    """The file's SHA-256. A v3 checkpoint is binary, so its bytes cannot be
    read into a `String` and compared as text."""
    return sha256_file(path)


def _obs(f: Scalar[DT]) -> List[Scalar[DT]]:
    """Build a 4-D CartPole-shaped obs (length/fill idiom)."""
    var o = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    o[0] = f
    o[1] = -f
    o[2] = f * Scalar[DT](0.5)
    o[3] = -f
    return o^


def _make_trained_gpu() raises -> DQNTrainer[
    "gpu", UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP], QNet,
]:
    seed(42)
    var trainer = DQNTrainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(
        ctx=DeviceContext(),
        lr=Scalar[DT](2.5e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](1.0),
        epsilon_decay=Scalar[DT](0.995),
        epsilon_min=Scalar[DT](0.05),
        learning_starts=WARMUP,
        target_update_freq=500,
        initial_episode_fill=Scalar[DT](0.0),
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS, ctx=DeviceContext(),
        print_every=5000, verbose=False,
    )
    return trainer^


def _cpu_eval() raises -> DQNTrainer[
    "cpu", UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP], QNet,
]:
    return DQNTrainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(lr=Scalar[DT](2.5e-4), learning_starts=WARMUP)


def _fresh_gpu() raises -> DQNTrainer[
    "gpu", UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP], QNet,
]:
    return DQNTrainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
        QNet,
    ].make(
        ctx=DeviceContext(),
        lr=Scalar[DT](2.5e-4),
        learning_starts=WARMUP,
    )


def test_dqn_gpu_checkpoint_roundtrip() raises:
    print("--- DQN GPU checkpoint round-trip (storage) ---")
    try:
        var _probe = DeviceContext()
    except:
        print("  no accelerator — skipping")
        return

    var trainer = _make_trained_gpu()
    trainer.save_state(CKPT_A)
    print("  saved trained GPU trainer ->", CKPT_A)

    var loaded = _fresh_gpu()
    loaded.load_state(CKPT_A)
    loaded.save_state(CKPT_B)
    print("  loaded into fresh GPU trainer, re-saved ->", CKPT_B)

    # Invariant 1: re-save byte-identity (params + state + ε + counter).
    var a = _read_file(CKPT_A)
    var b = _read_file(CKPT_B)
    assert_equal(
        a, b,
        "GPU checkpoint not byte-identical after save→load→save "
        "(param/state/ε round-trip lossy)",
    )
    print("  re-save byte-identity OK (sha256", a, ")")

    # Invariant 2: greedy-action agreement on a battery of observations.
    var n_match = 0
    var n_total = 0
    for i in range(20):
        var obs = _obs(Scalar[DT](i) * Scalar[DT](0.05) - Scalar[DT](0.5))
        var a_orig = trainer.select_greedy_action(obs)
        var a_load = loaded.select_greedy_action(obs)
        n_total += 1
        if a_orig == a_load:
            n_match += 1
    assert_equal(
        n_match, n_total,
        "loaded GPU trainer disagrees on greedy action ("
        + String(n_match) + "/" + String(n_total) + " match)",
    )
    print("  greedy-action agreement OK (", n_match, "/", n_total, ")")

    # Invariant 3: GPU→CPU interchange — the headline use case. A CPU trainer
    # loads the GPU checkpoint and picks the same greedy actions.
    var cpu_eval = _cpu_eval()
    cpu_eval.load_state(CKPT_A)
    var n_x = 0
    for i in range(20):
        var obs = _obs(Scalar[DT](i) * Scalar[DT](0.05) - Scalar[DT](0.5))
        if trainer.select_greedy_action(obs) == cpu_eval.select_greedy_action(
            obs
        ):
            n_x += 1
    assert_equal(n_x, 20, "GPU→CPU interchange: CPU trainer disagrees")
    print("  GPU→CPU interchange OK ( CPU eval agrees", n_x, "/ 20 )")
    print("PASS")


def main() raises:
    test_dqn_gpu_checkpoint_roundtrip()
