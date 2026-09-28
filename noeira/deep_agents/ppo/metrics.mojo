"""PPOMetrics — per-train-step bundle for the PPO trainer.

Mirrors `SACMetrics`: one @fieldwise_init struct, one LogScalar[DT] per
metric. `log_bundle[PPOMetrics, L: Logger]` walks via reflection and
emits one `log_scalar` call per field.

Fields correspond to what `PPOTrainer` accumulates over the K-epoch
minibatch SGD inside one `train_step`:
  * `policy_loss`  — mean of PPO clipped-surrogate loss across all minibatches
  * `critic_loss` — mean of value MSE across all minibatches
  * `train_steps` — cumulative minibatch updates since trainer was made
                    (NOT reset on flush)
  * `n_updates`   — total minibatch updates inside this train_step
                    (= N_EPOCHS * N_MINIBATCHES). Cast to Float64
                    for uniform Logger surface.

Entropy / approx_kl / clip_fraction / explained_variance are captured by
a CPU-only diagnostic walk in `PPOTrainer.train_step` (re-runs the actor
on each minibatch + reads the critic's value/return scratches). They read
0.0 on the GPU train path, where the diag walk is skipped."""

from noeira.nn.constants import DT
from noeira.nn.core.metric import LogScalar


@fieldwise_init
struct PPOMetrics(Copyable, Movable, Deinitable):
    var policy_loss:        LogScalar[DT]
    var critic_loss:        LogScalar[DT]
    var train_steps:        LogScalar[DT]
    var n_updates:          LogScalar[DT]
    # Per-minibatch policy/critic diagnostics (CPU diag walk; 0.0 on GPU
    # where the diag pass is skipped). Mirrors the DQN/C51 metrics pattern.
    var entropy:            LogScalar[DT]  # mean Gaussian entropy
    var approx_kl:          LogScalar[DT]  # Schulman-2020 (r-1)-log r
    var clip_fraction:      LogScalar[DT]  # frac |r-1| > clip_eps
    var explained_variance: LogScalar[DT]  # 1 - Var(ret-v)/Var(ret)
