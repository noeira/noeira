"""REDQMetrics — per-train-step bundle for the REDQ trainer.

Same shape as `SACMetrics` field-for-field so downstream monitoring
tools work unchanged (REDQ is a generalization of SAC; the metric
semantics are identical, just averaged over N instead of 2 critics).

Field semantics:
  policy_loss      — mean of `α·logp − combined_Q` across the chunk
                    (computed every POLICY_DELAY inner critic steps)
  critic_loss     — mean of Σᵢ MSE(Qᵢ(s,a), y) across all inner steps
                    (one outer chunk = UTD × outer-train-steps inner
                    updates; this is summed-not-averaged across the N
                    critics, matching SAC's `loss1 + loss2` convention)
  alpha           — point-in-time entropy temperature (= exp(log_alpha))
  mean_q          — mean Q over the LAST critic in the ensemble across
                    inner steps (representative sample; matches the
                    legacy REDQ convention)
  mean_target     — mean of the REDQ TD target `y` across inner steps
  reward_mean     — mean of batch reward across inner steps
  mean_next_q     — mean of the combined_Q used to build y (after
                    MIN/AVE/REM reduction) across inner steps
  mean_done       — mean batch done across inner steps
  action_abs_mean — mean |action| in the actor batch (per-policy step)
  train_steps     — cumulative INNER train steps so far (NOT reset on
                    flush). One env step contributes UTD inner steps.
  n_updates       — INNER train steps THIS chunk (reset on flush)

`mean_q` and `mean_next_q` come from the LAST critic in the ensemble's
sweep order rather than a representative one — both SAC and legacy
REDQ pick a single critic so the metric volume stays comparable across
algorithms with different ensemble sizes."""

from noeira.nn.constants import DT
from noeira.nn.core.metric import LogScalar


@fieldwise_init
struct REDQMetrics(Copyable, Movable, Deinitable):
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
