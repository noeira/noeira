"""C51 GPU checkpoint round-trip + GPU→CPU interchange (Phase 2).

Mirrors `test_dqn_checkpoint_gpu.mojo` for the distributional C51 trainer
(per-atom logits Q-net, same Adam optimizer). Invariants:
  1. save → load → save is byte-identical.
  2. loaded GPU trainer agrees on greedy (expected-Q argmax) actions.
  3. GPU→CPU interchange: a CPU trainer loads the GPU checkpoint and
     agrees on greedy actions (train-on-GPU → eval-on-CPU).

Guards on DeviceContext construction (no-op on CPU-only CI).
Run: pixi run -e apple mojo run -I . tests/nn/test_c51_checkpoint_gpu.mojo
"""

from std.random import seed
from max.gpu.host import DeviceContext
from noeira.io.sha256 import sha256_file
from std.testing import assert_true, assert_equal

from noeira.nn.constants import DT
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.deep_agents.c51.trainer import C51Trainer
from noeira.deep_agents.training.driver_offpolicy_discrete import (
    run_offpolicy_discrete_train,
)
from noeira.deep_agents.training.blocks import (
    UniformSampleGpuStep, UniformSampleCpuStep,
)

from noeira.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime NA = 51
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 4_096
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500

comptime C51Net = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS * NA],
]

comptime CKPT = String("/tmp/c51_gpu_ckpt.ckpt")


def _read_file(path: String) raises -> String:
    """The file's SHA-256. A v3 checkpoint is binary, so its bytes cannot be
    read into a `String` and compared as text."""
    return sha256_file(path)


def _obs(f: Scalar[DT]) -> List[Scalar[DT]]:
    var o = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    o[0] = f
    o[1] = -f
    o[2] = f * Scalar[DT](0.5)
    o[3] = -f
    return o^


def test_c51_gpu_checkpoint() raises:
    print("--- C51 GPU checkpoint round-trip ---")
    try:
        var _probe = DeviceContext()
    except:
        print("  no accelerator — skipping")
        return

    seed(42)
    var trainer = C51Trainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
        C51Net,
        NA, NUM_ACTIONS,
    ].make(
        ctx=DeviceContext(), lr=Scalar[DT](2.5e-4),
        learning_starts=WARMUP, target_update_freq=500,
    )
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer, env, TOTAL_STEPS, ctx=DeviceContext(),
        print_every=5000, verbose=False,
    )
    trainer.save_state(CKPT)

    var loaded = C51Trainer[
        "gpu",
        UniformSampleGpuStep[OBS_DIM, 1, BATCH, CAP],
        C51Net,
        NA, NUM_ACTIONS,
    ].make(ctx=DeviceContext(), lr=Scalar[DT](2.5e-4), learning_starts=WARMUP)
    loaded.load_state(CKPT)
    loaded.save_state(String("/tmp/c51_gpu_ckpt_b.ckpt"))
    assert_equal(
        _read_file(CKPT), _read_file(String("/tmp/c51_gpu_ckpt_b.ckpt")),
        "C51 GPU checkpoint not byte-identical after save→load→save",
    )
    print("  [1] re-save byte-identity OK")

    var n_match = 0
    for i in range(20):
        var obs = _obs(Scalar[DT](i) * Scalar[DT](0.05) - Scalar[DT](0.5))
        if trainer.select_greedy_action(obs) == loaded.select_greedy_action(obs):
            n_match += 1
    assert_equal(n_match, 20, "loaded C51 GPU trainer disagrees on greedy")
    print("  [2] greedy-action agreement OK ( 20 / 20 )")

    var cpu_eval = C51Trainer[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        C51Net,
        NA, NUM_ACTIONS,
    ].make(lr=Scalar[DT](2.5e-4), learning_starts=WARMUP)
    cpu_eval.load_state(CKPT)
    var n_x = 0
    for i in range(20):
        var obs = _obs(Scalar[DT](i) * Scalar[DT](0.05) - Scalar[DT](0.5))
        if trainer.select_greedy_action(obs) == cpu_eval.select_greedy_action(
            obs
        ):
            n_x += 1
    assert_equal(n_x, 20, "C51 GPU→CPU interchange: greedy disagrees")
    print("  [3] GPU→CPU interchange OK ( CPU eval agrees 20 / 20 )")
    print("PASS")


def main() raises:
    test_c51_gpu_checkpoint()
