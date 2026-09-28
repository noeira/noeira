"""SACMetrics — per-train-step bundle for the SAC trainer.

One @fieldwise_init struct, one LogScalar[DT] per metric.
`log_bundle[SACMetrics, L: Logger]` walks via reflection and emits one
`log_scalar(name, value, step)` per field, so a FIELD NAME IS THE METRIC
NAME the dashboard groups by (`noeira-cloud/.../metric-groups.ts`). Until
2026-09-22 three of them kept the legacy GPU-SAC names — `actor_loss`,
`mean_reward`, `mean_abs_action` — which no dashboard group matched.

Naming convention:
  policy_loss      — mean SAC actor loss across the chunk
  critic_loss     — mean SAC critic loss (twin-critic min) across the chunk
  alpha           — point-in-time entropy temperature (= exp(log_alpha))
  mean_q          — mean of online Q1(s, a) over the batch
  mean_target     — mean of target_y (Bellman target) across the chunk
  reward_mean     — mean of batch reward across the chunk
  mean_next_q     — mean of min(Q1_t, Q2_t)(s', a') over the batch
                    (the target-critic next-Q the TD bootstrap is built
                    from; reads the `min_q` ComputeGraph node)
  mean_done       — mean of batch done across the chunk
  action_abs_mean — mean |action| across the chunk
  train_steps     — cumulative training updates so far (NOT reset on flush)
  n_updates       — training updates THIS chunk (reset on flush)

`mean_next_q` reads the `min_q` intermediate of `TargetYBlock`'s
ComputeGraph via `node_out_ptr` (CPU diag walk; the GPU path leaves it
0.0, same convention as the other diagnostics). Wired 2026-05-30 once
the ComputeGraph node-output accessor landed."""

from noeira.nn.constants import DT
from noeira.nn.core.metric import LogScalar


@fieldwise_init
struct SACMetrics(Copyable, Movable, Deinitable):
    var policy_loss:     LogScalar[DT]
    var critic_loss:     LogScalar[DT]
    var alpha:           LogScalar[DT]
    var mean_q:          LogScalar[DT]
    var mean_target:     LogScalar[DT]
    var reward_mean:     LogScalar[DT]
    var mean_next_q:     LogScalar[DT]
    var mean_done:       LogScalar[DT]
    var action_abs_mean: LogScalar[DT]
    var train_steps:     LogScalar[DT]
    var n_updates:       LogScalar[DT]
