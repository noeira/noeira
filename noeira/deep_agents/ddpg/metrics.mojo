"""DDPGMetrics — per-train-step bundle for the DDPG trainer.

Learning diagnostics mirror the SAC bundle (CPU-only diag walk):
  * `mean_q`      — mean Q(s, a) over the minibatch (critic forward,
                    reads `critic_blk.inner._mb_q`)
  * `mean_target` — mean TD target y over the minibatch (`state.mb_y`)
  * `reward_mean` — mean reward of the minibatch (`state.mb_r`)

`train_steps` is the cumulative count of trainer updates (NOT reset on
flush); `n_updates` is per-chunk. On the storage GPU train path these
diagnostics are populated from device-resident accumulators (read once at
flush), same as SAC — not 0.0."""

from noeira.nn.constants import DT
from noeira.nn.core.metric import LogScalar


@fieldwise_init
struct DDPGMetrics(Copyable, Movable, Deinitable):
    var policy_loss: LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var mean_q:      LogScalar[DT]
    var mean_target: LogScalar[DT]
    var reward_mean: LogScalar[DT]
    var train_steps: LogScalar[DT]
    var n_updates:   LogScalar[DT]
