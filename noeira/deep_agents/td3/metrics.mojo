"""TD3Metrics — per-train-step bundle for the TD3 trainer.

TD3 has separate actor + critic update counts due to policy delay.
Each is its own LogScalar so the logger sees both. `train_steps`
matches `n_critic_updates` cumulatively (the critic fires every
train_step). Per-chunk counts are reset on `flush_metrics`;
`train_steps` is not.

Learning diagnostics mirror the SAC bundle (CPU-only diag walk, run
on the critic cadence → averaged by `n_critic_updates`):
  * `mean_q`      — mean Q1(s, a) over the minibatch (twin critic
                    forward, reads `twin_critic_blk.inner.c1._mb_q`)
  * `mean_target` — mean TD target y over the minibatch (`state.mb_y`)
  * `reward_mean` — mean reward of the minibatch (`state.mb_r`)
  * `mean_done`   — mean done flag of the minibatch (`state.mb_d`)
On the storage GPU train path these diagnostics are populated from
device-resident accumulators (read once at flush), same as SAC — not 0.0."""

from noeira.nn.constants import DT
from noeira.nn.core.metric import LogScalar


@fieldwise_init
struct TD3Metrics(Copyable, Movable, Deinitable):
    var policy_loss:      LogScalar[DT]
    var critic_loss:      LogScalar[DT]
    var mean_q:           LogScalar[DT]
    var mean_target:      LogScalar[DT]
    var reward_mean:      LogScalar[DT]
    var mean_done:        LogScalar[DT]
    var train_steps:      LogScalar[DT]
    var n_actor_updates:  LogScalar[DT]
    var n_critic_updates: LogScalar[DT]
