"""GUMBEL MuZero on Connect Four — full GPU, the learned-model sibling of
`connect_four_alphazero_gumbel.mojo`.

Same arena harness as the AlphaZero Gumbel example (best/learner gating,
horizontal-flip augmentation, periodic full-strength MCTS eval vs 5-ply minimax +
random, run logger), but the agent plans over a **learned model** instead of
the true game rules: three MLP nets h/g/f (representation / dynamics / prediction)
trained by a K-step BPTT unroll, with the self-play search swapped to two-player
Gumbel MuZero (`run_muzero_selfplay_arena_gumbel_2p`).

The MuZero value/reward heads are categorical over the board outcome support
[-1, +1] (`BINS` atoms); the two-player n-step targets carry the perspective
sign flips. Unlike the AlphaZero example's BatchNorm ResNet, the h/g/f torsos are
plain MLPs (no BatchNorm), so arena promotion is a params-only copy — there are
no running stats to leak, sidestepping the BN-promotion pitfall.

Connect Four is heavy (126D obs = 3×6×7, 7 actions, games up to 42 plies, a
learned model searched 64 sims/move) — this needs an NVIDIA GPU to train at a
useful pace.

`iterations` / `report_every` / `arena_every` are in self-play *moves* (one loop
pass advances all N_ENVS games by one move).

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_muzero_gumbel.mojo

The run (project `board-games`) keeps its checkpoint, `metrics.csv` and
`run.kv` in `runs/<id>/`; with no `NOEIRA_CLOUD_URL` in `.env` the monitor half
is inert and the per-report lines still print to stdout.
"""

from std.memory import Pointer
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.deep_agents.muzero.nets import (
    MZRepNet, MZRepNetC4Conv, MZDynNet, MZPredNet
)
from noeira.deep_agents.muzero.selfplay_arena_gumbel_2p import (
    run_muzero_selfplay_arena_gumbel_2p,
)
from noeira.deep_agents.zero.symmetries import HFlipColumnAugmenter
from noeira.deep_agents.zero.evaluators import (
    RandomOpponent,
    GPUMinimaxConnectFour,
)
from noeira.nn.core.checkpoint import save_params
from noeira.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    print("=== Gumbel MuZero on Connect Four (deep_agents / nn) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var run = RunContext(
        project=String("board-games"),
        driver=String("examples/board_games/connect_four_muzero_gumbel.mojo"),
        slug=String("muzero-gumbel-connect-four"),
        env=String("builtin:connect_four"),
    )
    print("run:", run.dir)
    var logger = run_logger(run, buffer_size=22)
    logger.set_config("agent", "GumbelMuZero")
    logger.set_config("env", "ConnectFour")
    logger.set_config("network", "MZ MLP[LATENT=128,H=128,BINS=51]")
    logger.set_config("framework", "deep_agents/nn")
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    comptime OBS = 126
    comptime ACT = 7
    # 256-wide latent + hidden: the no-cap run's value head plateaued
    # (value_mse flat ~0.38 across 40k while policy CE fell) — the flat-latent
    # MLP dynamics/value path is the ceiling, so grow its capacity. Slower GEMMs
    # + a fresh restart (new architecture, no checkpoint reuse).
    comptime LATENT = 256
    comptime BINS = 51       # categorical value/reward support over [-1, 1]
    comptime H = 256         # MLP hidden width for h/g/f
    # 128 sims/move (was 64): with the value head pinned by the flat-latent
    # dynamics (value_mse flat across the 256-wide run), more search lets MCTS
    # lean on the strong policy prior instead of the weak value — deeper
    # Sequential-Halving budget over the same 4 root candidates. ~linear cost.
    comptime NUM_SIMS = 128
    comptime MAX_NODES = 256
    # MAX_K is already maxed for C4: the planner clips k to the largest power of
    # two ≤ k (and requires k ≤ ACT=7), so 4 is the ceiling — 8 would exceed the
    # 7 actions and 6 silently clips back to 4. More search must come from sims.
    comptime MAX_K = 4       # Gumbel root candidates (power of two, ≤ ACT=7)
    comptime CAP = 1_000_000
    comptime B = 128         # unroll batch
    comptime K = 5           # BPTT unroll length
    comptime N = 10          # n-step value-target horizon
    comptime MAX_PLIES = 42  # full ConnectFour board

    comptime Env = ConnectFourEnv[DType.float64]
    # Representation torso: a BN-free conv ResNet over the 3×6×7 board (the
    # spatial inductive-bias upgrade from the flat-MLP `MZRepNet[OBS, LATENT, H]`,
    # which remains a drop-in A/B swap — the dynamics/prediction nets and the
    # whole driver are agnostic to the rep torso). F = conv filters.
    comptime Rep = MZRepNetC4Conv[LATENT, H, F=64]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    # ConnectFour's only board symmetry is the left↔right column flip.
    comptime Aug = HFlipColumnAugmenter[ROWS=6, COLS=7, PLANES=3]

    var ctx = DeviceContext()

    # The BEST net trio — holds the final (best) weights on return. The driver
    # builds + trains a learner copy internally and promotes it on arena wins.
    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)

    var res = run_muzero_selfplay_arena_gumbel_2p[
        Env, Rep, Dyn, Pred, Aug,
        N_ENVS=64,
        OBS=OBS, ACT=ACT, LATENT=LATENT, BINS=BINS,
        NUM_SIMS=NUM_SIMS, MAX_NODES=MAX_NODES, MAX_K=MAX_K,
        CAP=CAP, B=B, K=K, N=N, MAX_PLIES=MAX_PLIES,
        OPP1=GPUMinimaxConnectFour[5],
        OPP2=RandomOpponent,
        L=RunLogger,
        ARENA_GAMES=64,
        EVAL_GAMES=64,
        TEMP_MOVES=20,
    ](
        ctx,
        rep, dyn, pred,
        iterations=40_000,
        learning_starts=2_000,
        train_per_iter=4,
        lr=Scalar[DT](2e-3),
        gamma=Scalar[DT](1.0),
        # Value head was the weak/unstable head in the no-reanalyze run
        # (value_mse crept up while policy CE fell) — give it more gradient.
        value_coef=Scalar[DT](0.5),
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
        # Diversity levers against the self-play collapse seen without these:
        # more random opening plies + a temperature floor so play stays
        # stochastic past the opening (AlphaZero.jl uses temp≈0.3 throughout)
        # instead of going fully greedy and narrowing the replay distribution.
        selfplay_open_plies=4,
        temp_min=0.35,
        eval_open_plies=4,
        # Reanalyze: every 4 moves, re-target B stored positions with a lagging
        # target net (synced from the learner every 200 grad steps) — the
        # EfficientZero-style high-coverage refresh that keeps the n-step value
        # targets fresh as the learner improves (the sample-efficiency lever for
        # the plateau). reanalyze_batch ≈ B so a meaningful fraction of each
        # train batch carries fresh targets.
        reanalyze_every=4,
        reanalyze_batch=128,
        target_sync_interval=200,
    )

    # Persist the BEST net trio (rep/dyn/pred hold the final promoted weights —
    # the deployable artifact, distinct from the drifting learner). The storage
    # checkpoint is whole-file-per-model, so this writes three per-net files
    # (`.rep` / `.dyn` / `.pred`), the same layout the driver's rolling
    # checkpoint + `MuZeroAgent.save` use.
    var ckpt = run.checkpoint_path(String("best"))
    save_params["gpu", Rep](rep, ckpt + String(".rep"), Optional(ctx), False)
    save_params["gpu", Dyn](dyn, ckpt + String(".dyn"), Optional(ctx), False)
    save_params["gpu", Pred](pred, ckpt + String(".pred"), Optional(ctx), False)
    announce_checkpoint(ckpt + String(".rep"), artifacts, run.dir)
    announce_checkpoint(ckpt + String(".dyn"), artifacts, run.dir)
    announce_checkpoint(ckpt + String(".pred"), artifacts, run.dir)
    finish_run(
        run, logger, artifacts,
        String("last_loss=") + String(res.last_loss) + String(" promotions=") + String(res.promotions),
    )

    print()
    print("last_loss:", res.last_loss, "| promotions:", res.promotions)
    print("saved best net →", ckpt + String(".{rep,dyn,pred}"))
    print("=== Done ===")
