"""GUMBEL MuZero on Connect Four with a SPATIAL latent — the value-ceiling test.

Identical arena harness to `connect_four_muzero_gumbel.mojo`, but the learned
model keeps the latent **spatial** (`[C, H, W]` feature map, flat-encoded as
LATENT = C·6·7) and uses conv h/g/f: a conv representation, a convolutional
dynamics with action-plane embedding (`MZDynNetC4Spatial`, a ComputeGraph), and
conv prediction heads. This targets the value-fit ceiling the flat-MLP runs hit
(`value_mse` plateaued ~0.38 regardless of width/sims) — the flat-latent MLP
dynamics can't model C4 tactics, so the n-step value targets stay noisy. The
spatial dynamics gives g weight-sharing over the board, which is what the
EZv2/AlphaZero spatial model buys.

All conv blocks are BatchNorm-FREE (the arena's params-only promotion can't carry
BN running stats). C = latent channels (LATENT = C·6·7; C=64 → 2688), NB = residual
blocks per net (h/g/f), the muzero-general `blocks` knob. The spatial
latent + conv dynamics are heavier than the flat path, so this starts at 64
sims/move (raise NUM_SIMS once you see the wall-clock).

Watch `value_mse`: if it drops below the flat run's ~0.38 floor, the spatial
dynamics broke the ceiling; if it stays pinned, the value target itself is the
limit (not the model).

The best net goes to `runs/<id>/checkpoints/best.ckpt` (project `board-games`),
beside the run's `metrics.csv` and `run.kv`.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_muzero_gumbel_spatial.mojo
"""

from std.memory import Pointer
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.deep_agents.muzero.nets_spatial import (
    MZRepNetC4Spatial,
    MZDynNetC4Spatial,
    MZPredNetC4Spatial,
)
from noeira.deep_agents.muzero.selfplay_arena_gumbel_2p import (
    run_muzero_selfplay_arena_gumbel_2p,
)
from noeira.nn.optimizer.lr_scheduler import LinearWarmupSchedule
from noeira.deep_agents.zero.symmetries import HFlipColumnAugmenter
from noeira.deep_agents.zero.evaluators import (
    RandomOpponent,
    GPUMinimaxConnectFour,
)
from noeira.nn.core.checkpoint import save_params_multi
from noeira.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    print(
        "=== Gumbel MuZero on Connect Four — SPATIAL latent (deep_agents) ==="
    )
    print()

    var run = RunContext(
        project=String("board-games"),
        driver=String("examples/board_games/connect_four_muzero_gumbel_spatial.mojo"),
        slug=String("muzero-gumbel-spatial-connect-four"),
        env=String("builtin:connect_four"),
    )
    print("run:", run.dir)
    var logger = run_logger(run, buffer_size=22)
    logger.set_config("agent", "GumbelMuZero")
    logger.set_config("env", "ConnectFour")
    logger.set_config("network", "MZ spatial conv h/g/f [C=64, 3 blocks, 6x7]")
    logger.set_config("framework", "deep_agents/nn")
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    comptime OBS = 126
    comptime ACT = 7
    comptime CH = 64  # latent channels → LATENT = CH*6*7 = 2688
    comptime NB = 3  # residual blocks per net (muzero-general `blocks`)
    comptime HH = 6
    comptime WW = 7
    comptime LATENT = CH * HH * WW
    comptime BINS = 51  # categorical value/reward support over [-1, 1]
    # 64 sims/move. 128 was WORSE early (eval1 0.43 vs 0.75 at matched steps, 0
    # promotions): deep search over a still-imperfect learned model amplifies its
    # value/dynamics errors. The references' 200-500 sims assume a converged
    # model + PUCT/Dirichlet; early Gumbel does better at 64. Revisit via a
    # sims SCHEDULE (low early → high once the model is strong) if needed.
    comptime NUM_SIMS = 64
    comptime MAX_NODES = 256  # ≫ NUM_SIMS (≤1 node/sim); ample headroom.
    comptime MAX_K = 4  # already maxed for C4 (power of two ≤ ACT=7)
    comptime CAP = 1_000_000
    comptime B = 128
    comptime K = 5
    # N (td_steps) = full game → Monte-Carlo value targets. C4's reward is
    # sparse-terminal, so a short n-step (was 10) bootstraps early-game value
    # through the NOISY learned value — the likely source of the value_mse floor.
    # N=42 makes the value target the actual game outcome (±1), low-variance, the
    # muzero-general C4 recipe (td_steps=42). The unroll length K stays 5.
    comptime N = 42
    comptime MAX_PLIES = 42
    # Linear LR warmup over the first LR_WARMUP optimizer steps (0 → base lr).
    # The CH=64/3-block net is unstable early under the base 2e-3 (the 32/2 run
    # had a clean start; the bigger net crashed to ~0.13 eval1 at ~step 2k).
    # Warmup ramps in the LR so the first few hundred updates don't blow up.
    comptime LR_WARMUP = 1000

    comptime Env = ConnectFourEnv[DType.float64]
    comptime Rep = MZRepNetC4Spatial[CH, HH, WW, NB]
    comptime Dyn = MZDynNetC4Spatial[CH, ACT, BINS, HH, WW, NB]
    comptime Pred = MZPredNetC4Spatial[CH, ACT, BINS, HH, WW, NB]
    comptime Aug = HFlipColumnAugmenter[ROWS=6, COLS=7, PLANES=3]

    var ctx = DeviceContext()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)
    # init_zero (EZv2) is baked into the net definitions: the policy/value/reward
    # head output Linears are `InitWith[Linear[...], Zero]`, so `make` already
    # produced a uniform policy prior + neutral value/reward (stable MCTS targets
    # early + more early exploration). No post-make zeroing pass needed.

    var res = run_muzero_selfplay_arena_gumbel_2p[
        Env,
        Rep,
        Dyn,
        Pred,
        Aug,
        N_ENVS=64,
        OBS=OBS,
        ACT=ACT,
        LATENT=LATENT,
        BINS=BINS,
        NUM_SIMS=NUM_SIMS,
        MAX_NODES=MAX_NODES,
        MAX_K=MAX_K,
        CAP=CAP,
        B=B,
        K=K,
        N=N,
        MAX_PLIES=MAX_PLIES,
        OPP1=GPUMinimaxConnectFour[5],
        OPP2=RandomOpponent,
        L=RunLogger,
        ARENA_GAMES=64,
        EVAL_GAMES=64,
        TEMP_MOVES=20,
        SCHEDULER=LinearWarmupSchedule[LR_WARMUP],
        USE_TRAIN_CUDA_GRAPH=False,
        # USE_MCTS_CUDA_GRAPH stays OFF: the captured MCTS sim-loop replay
        # produces FLAT search targets (target_max_prob ~0.3 vs ~0.55 eager →
        # policy head stops learning). Bug is strictly in the sim-loop
        # capture/replay (the eager refactored search is correct). Not worth
        # chasing for this net: MCTS here is compute-bound (CH=64 conv), so the
        # graph adds only ~6%. See docs/MUZERO_CUDA_GRAPH_PLAN.md.
    ](
        ctx,
        rep,
        dyn,
        pred,
        iterations=40_000,
        learning_starts=2_000,
        train_per_iter=4,
        lr=Scalar[DT](2e-3),
        gamma=Scalar[DT](1.0),
        # value_coef 0.25 (MuZero paper / muzero-general): a DOWN-weighted value
        # loss avoids value overfitting. The earlier bump to 0.5 was ~neutral;
        # with N=42 Monte-Carlo targets the value signal is cleaner anyway.
        value_coef=Scalar[DT](0.25),
        max_grad_norm=Scalar[DT](1.0),
        seed=42,
        arena_every=2_000,
        arena_open_plies=4,
        promote_threshold=0.55,
        report_every=1_000,
        diag_every=50,
        do_eval=True,
        do_eval2=True,
        verbose=True,
        logger=Pointer(to=logger).as_unsafe_any_origin(),
        # Anti-sharpening / sustained exploration: never go greedy in self-play
        # (temp_min=1.0 → sample ∝ visits the whole game, the muzero-general
        # temp=1-always recipe) + 6 random opening plies, paired with the
        # init_zero uniform prior. Counters the column-marked dynamics' sharper
        # search that was collapsing self-play diversity.
        selfplay_open_plies=6,
        # Self-play temperature anneal (muzero-general whole-game recipe). With
        # decay_steps set, the post-opening temperature follows 1.0 → 0.5 (at 50%
        # of `iterations`) → 0.25 (at 75%): diverse early, sharpening — incl. the
        # endgame — in the back half so the full-MC (N=42) value target reflects
        # position quality instead of a temp=1.0 coin-flip outcome. This is the
        # lever for the value_mse / loss_value plateau (the target was the noise,
        # not the value head). `temp_min` is now only consulted when the schedule
        # is OFF (decay_steps=0); kept at 1.0 for that legacy path.
        temperature_decay_steps=40_000,
        temp_min=1.0,
        eval_open_plies=4,
        # Reanalyze: refresh stale (policy, value) targets on stored positions with
        # the lagging target net. A clear win for C4 (faster, higher-ceiling).
        reanalyze_every=4,
        reanalyze_batch=128,
        target_sync_interval=200,
        # Rolling checkpoint of the best net every 2k moves → playable /
        # recoverable mid-run (play it with play_connect_four_muzero_gumbel).
        checkpoint_every=2_000,
        checkpoint_path=run.checkpoint_path(String("best")),
        # Prioritized Experience Replay (device sum-tree). OFF for board games:
        # the MuZero paper uses PER only for Atari and samples BOARD-GAME states
        # UNIFORMLY ("For board games, states are sampled uniformly"). Empirically
        # PER here drags learning (prioritizing by value-error skews the uniform
        # board-game signal); use_per=False is both paper-correct and the best run.
        # (The PER GPU path itself is verified non-corrupting — see
        # tests/deep_agents/test_mz_unroll_overfit_isw_gpu.mojo.)
        use_per=False,
        per_alpha=Scalar[DT](1.0),
        per_beta=Scalar[DT](1.0),
    )

    # Storage checkpoint: the rep/dyn/pred trio goes into ONE file via
    # `save_params_multi` — the same single-file layout the driver's rolling
    # `checkpoint_every` save uses, so `play_connect_four_muzero_gumbel` loads
    # from it too.
    var ckpt = run.checkpoint_path(String("best"))
    save_params_multi["gpu", Rep, Dyn, Pred](
        ckpt, Optional(ctx), False, rep, dyn, pred
    )
    announce_checkpoint(ckpt, artifacts, run.dir)
    finish_run(
        run, logger, artifacts,
        String("last_loss=") + String(res.last_loss) + String(" promotions=") + String(res.promotions),
    )

    print()
    print("last_loss:", res.last_loss, "| promotions:", res.promotions)
    print("saved best net →", ckpt)
    print("=== Done ===")
