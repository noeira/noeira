"""Wire a `HilSerlConfig` into a set-up SAC trainer."""

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.core.logger import Logger
from noeira.deep_agents.sac.trainer import SACTrainer
from noeira.deep_agents.training.blocks.sample_block import SampleBlock
from noeira.deep_agents.demos.load import load_demos_into_replay
from .config import HilSerlConfig


def apply_hil_serl[
    t: StaticString, S: SampleBlock, A: Module, C: Module, L: Logger
](
    mut trainer: SACTrainer[t, S, A, C],
    cfg: HilSerlConfig,
    mut logger: L,
    who: String = "hil-serl",
) raises:
    """Load and pin the demos, then turn on the BC term (and `--bc-only`).

    ⚠ BEFORE THE FIRST TRAIN STEP. It writes the replay's device demo count,
    uploads the BC mask and sets loss-graph attributes; a CUDA graph captured
    by an earlier `train_step` would replay the old values. Refused after one.

    ⚠ THE BC MASK IS THE BATCH'S FIRST HALF because the mixed sampler draws
    the demo rows there (`_mixed_indices_dev_kernel`, `data/replay_gpu.mojo`)."""
    cfg.validate(who)
    if trainer.total_train_steps() > 0:
        raise Error(
            who + ": apply_hil_serl after " + String(trainer.total_train_steps())
            + " train steps — it must run before the first (CUDA-graph capture)"
        )
    if cfg.active():
        var rep = load_demos_into_replay(
            trainer.sample_blk, cfg.paths(who), cfg.filter, trainer.ctx, who
        )
        print("  demos    :", rep.n_rows, "rows pinned as the replay"
              " prefix (of", rep.n_file_rows, "in the files); mean reward",
              rep.mean_reward)
        # ⚠ CONFIG, NOT METRICS: facts about the run, not curves. They are
        # known only now, after `register`, so they reach `metrics.config.kv`
        # (rewritten at `close`) but not the dashboard's `/runs` config.
        logger.set_config("demo_rows", String(rep.n_rows))
        logger.set_config("demo_mean_reward", String(rep.mean_reward))
        if cfg.bc_weight > Scalar[DT](0):
            var half = SACTrainer[t, S, A, C].BATCH // 2
            trainer.set_bc(cfg.bc_weight, half)
            print("  bc       : weight", cfg.bc_weight, "on the demo half of"
                  " every batch (", half, "rows )")
            if cfg.bc_q_ratio > Scalar[DT](0):
                trainer.set_bc_q_ratio(cfg.bc_q_ratio)
                print("  bc-q     : the weight tracks max(", cfg.bc_weight,
                      ",", cfg.bc_q_ratio, "* mean|Q| ) at every diagnostics"
                      " flush (logged as `bc_weight`)")
    if cfg.bc_only:
        trainer.set_q_weight(Scalar[DT](0))
        print("  bc-only  : the SAC half of the actor loss is OFF —"
              " the actor fits the demo half of every batch, the critics"
              " train on its rollouts")
