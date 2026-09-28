"""Pure-Python Gymnasium CartPole-v1 stepping rate — the baseline for
`cartpole_steps_per_sec.mojo`, which imports `run` from this file so the
Python row is this exact loop.

Random actions, auto-reset on terminated/truncated, no agent. Standalone:

    pixi run python benchmarks/cartpole_sps_gym.py
"""

import random
import time

import gymnasium as gym


def run(num_steps: int, seed: int = 0) -> tuple[float, float]:
    """Step CartPole-v1 `num_steps` times; return (seconds, checksum)."""
    rng = random.Random(seed)
    actions = [rng.randrange(2) for _ in range(num_steps)]
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    checksum = 0.0
    t0 = time.perf_counter()
    for a in actions:
        obs, reward, terminated, truncated, _ = env.step(a)
        checksum += reward + obs[0]
        if terminated or truncated:
            env.reset()
    dt = time.perf_counter() - t0
    env.close()
    return dt, checksum


if __name__ == "__main__":
    n = 1_000_000
    run(10_000)  # warm-up
    best = min(run(n, seed=s)[0] for s in range(5))
    print(f"gymnasium {gym.__version__} CartPole-v1 (pure Python): "
          f"{n / best:,.0f} steps/s (best of 5, {n:,} steps)")
