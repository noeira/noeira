"""C51Metrics — per-train-step bundle for the C51 / Rainbow trainer.

Mirrors `DQNMetrics`: one @fieldwise_init struct, one LogScalar[DT] per
metric. `log_bundle[C51Metrics, L: Logger]` walks via reflection and
emits one `log_scalar` call per field.

Fields correspond to what `C51Trainer` accumulates in `_loss_accum`
(mean categorical cross-entropy over `_update_count`), plus current
explore_rate (point-in-time, not averaged). `train_steps` is the cumulative
count of trainer updates (NOT reset on flush); `n_updates` is per-chunk."""

from noeira.nn.constants import DT
from noeira.nn.core.metric import LogScalar


@fieldwise_init
struct C51Metrics(Copyable, Movable, Deinitable):
    var loss:              LogScalar[DT]
    var explore_rate:      LogScalar[DT]
    var mean_q:            LogScalar[DT]
    var mean_target:       LogScalar[DT]
    var dist_entropy_mean: LogScalar[DT]
    var reward_mean:       LogScalar[DT]
    var mean_done:         LogScalar[DT]
    var train_steps:       LogScalar[DT]
    var n_updates:         LogScalar[DT]
