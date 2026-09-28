"""MBPOMetrics — per-train-step bundle for the MBPO trainer.

Mirrors `SACMetrics`: one @fieldwise_init struct, one LogScalar[DT] per
metric. `log_bundle[MBPOMetrics, L: Logger]` walks via reflection and
emits one `log_scalar` call per field.

Fields correspond to the SAC actor/critic updates that run inside one
MBPO train_step (against the mixed real+synthetic batch):
  * `policy_loss`  — mean SAC actor loss
  * `critic_loss` — mean SAC critic loss
  * `alpha`       — current entropy temperature (point-in-time)
  * `mean_q`      — mean Q1(s, a) over the mixed batch (SAC critic forward)
  * `reward_mean` — mean reward of the mixed real+synthetic minibatch
  * `dyn_loss`    — mean dynamics-ensemble Gaussian-NLL loss over the
                    member-steps run since the last flush (the ensemble
                    trains on its own `model_train_freq` cadence, so this
                    is averaged independently of the SAC `n_updates`)
  * `train_steps` — cumulative SAC mini-updates since trainer was made
                    (NOT reset on flush)
  * `n_updates`   — total SAC mini-updates this chunk
                    (typically `sac_updates_per_step`).

`mean_q` / `reward_mean` mirror the SAC bundle (CPU-only diag walk);
`dyn_loss` surfaces the ensemble NLL the legacy MBPO trainer logged.
On the (unreachable) GPU train path these three read 0.0 — same
convention as DQN/C51/PPO."""

from noeira.nn.constants import DT
from noeira.nn.core.metric import LogScalar


@fieldwise_init
struct MBPOMetrics(Copyable, Movable, Deinitable):
    var policy_loss:            LogScalar[DT]
    var critic_loss:            LogScalar[DT]
    var alpha:                  LogScalar[DT]
    var mean_q:                 LogScalar[DT]
    var reward_mean:            LogScalar[DT]
    # Per-update batch stats (legacy parity): mean TD target y = r +
    # γ(1−d)(min Q' − α·logπ'), and the fraction of terminal transitions in
    # the mixed real+synth batch. `mean_target` going strongly negative is the
    # prime tell for synthetic-data Q-degradation.
    var mean_target:            LogScalar[DT]
    var mean_done:              LogScalar[DT]
    # Mean |action| over the mixed batch: proxy for
    # whether the policy is committing (large torques) or staying timid.
    var action_abs_mean:        LogScalar[DT]
    var dyn_loss:               LogScalar[DT]
    # Dynamics holdout suite (refreshed each model-train round, held between
    # rounds):
    #   * `dyn_holdout_mse_mean`   — mean per-member one-step MSE on a
    #     held-out real batch (`eval_member_mse`; it is an MSE, not the NLL
    #     the ensemble trains on — `dyn_loss` is that).
    #   * `dyn_holdout_mse_min/max/spread` — per-member MSE min / max /
    #     (max-min) = ensemble disagreement.
    #   * `dyn_input_std_mean` — mean over DYN_IN of the input scaler std.
    # ⚠ Named for the dashboard's groups (`metric-groups.ts`); until
    # 2026-09-22 these were `dyn_holdout_loss/min/max/spread`, `td_target`,
    # `done_ratio`, `actor_loss`, `mean_reward` and `mean_abs_action`.
    var dyn_holdout_mse_mean:   LogScalar[DT]
    var dyn_holdout_mse_min:    LogScalar[DT]
    var dyn_holdout_mse_max:    LogScalar[DT]
    var dyn_holdout_mse_spread: LogScalar[DT]
    var dyn_input_std_mean:     LogScalar[DT]
    var train_steps:            LogScalar[DT]
    var n_updates:              LogScalar[DT]
