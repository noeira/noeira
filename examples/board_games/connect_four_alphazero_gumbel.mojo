"""GUMBEL AlphaZero on Connect Four — full GPU, directly comparable to v2.

Identical telemetry/arena/eval harness to `connect_four_alphazero_v2.mojo`
(same minimax+random MCTS evals, arena gating, augmentation, logger, optimizer
stability settings) with ONLY the self-play planner swapped to Gumbel AlphaZero
via `train_arena_gumbel`: Gumbel-Top-k roots + Sequential Halving + improved-
policy targets at 64 sims/move (vs PUCT's 500, ~8× less search) — the
low-budget regime where Gumbel's policy-improvement guarantee lives (the TTT
gate beat the PUCT baseline 8-vs-24 losses at equal sims). A 256-sim run
validated the operator post-σ-fix; see the NUM_SIMS note below for the
budget history and the early-indicator criteria for this 64-sim run.

Second-generation port of `connect_four_alphazero.mojo`. Uses the config-free
nn net torsos (`AZConnectFourResNet` — conv stem → 5 identity-skip ResBlocks →
FC policy/value heads, 128 filters, the closest match to the original AlphaZero
backbone) + the `AlphaZeroAgent` facade, and exercises the production telemetry:
two pluggable `GPUEvaluator` opponents (5-ply minimax + random), a per-report
progress print, and a `run_logger` metrics sink. The periodic eval plays the
agent at full **MCTS** strength (temp=0), so the numbers reflect the deployed
agent, not the bare policy head.

Two logging cadences: the expensive MCTS eval + win-rates run every
`report_every` moves (coarse, ~minutes apart), while the cheap per-batch
training diagnostics — policy CE, policy/target entropy, target max-prob, the
policy KL gap, value MSE/mean, value-target stats — flush every `diag_every`
moves for dense training curves (legacy `train_selfplay_gpu` parity).

Connect Four is heavier than TicTacToe (126D obs = 3 planes × 6×7, 7 actions,
games up to 42 plies, a 5-block ResNet) — this needs an NVIDIA GPU to train at a
useful pace. The ResNet torso carries BatchNorm, which the self-play / eval
harness toggles (`set_attr["training"]`) automatically.

Note `iterations` / `report_every` are in self-play *moves* (one loop pass
advances all N_ENVS games by one move), not legacy-style collect+train rounds.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero_gumbel.mojo

The run (project `board-games`) keeps its checkpoint, `metrics.csv` and
`run.kv` in `runs/<id>/`; with no `NOEIRA_CLOUD_URL` in `.env` the monitor half
is inert and the per-report lines still print to stdout.
"""

from std.memory import Pointer
from max.gpu.host import DeviceContext

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.deep_agents.alphazero.nets import AZConnectFourResNet
from noeira.deep_agents.alphazero.agent import AlphaZeroAgent
from noeira.deep_agents.zero.symmetries import HFlipColumnAugmenter
from noeira.deep_agents.zero.evaluators import (
    RandomOpponent,
    GPUMinimaxConnectFour,
)
from noeira.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    print("=== Gumbel AlphaZero on Connect Four (deep_agents / nn) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var run = RunContext(
        project=String("board-games"),
        driver=String("examples/board_games/connect_four_alphazero_gumbel.mojo"),
        slug=String("alphazero-gumbel-connect-four"),
        env=String("builtin:connect_four"),
    )
    print("run:", run.dir)
    var logger = run_logger(run, buffer_size=22)
    logger.set_config("agent", "GumbelAlphaZero")
    logger.set_config("env", "ConnectFour")
    logger.set_config("network", "AZConnectFourResNet[F=128,NB=5,FC=128]")
    logger.set_config("framework", "deep_agents/nn")
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    # logger.set_config("charts", json.dumps([
    #     {
    #         "title": "Eval vs MinMax",
    #         "metrics": ["eval2_win", "eval2_draw", "eval2_loss"],
    #         "type": "stacked-bar",      # or "stacked-area" / "line"
    #         "normalize": True,          # 100%-stacked → win-rate view
    #         "colors": ["#22c55e", "#eab308", "#ef4444"],  # optional
    #     },
    # ]))

    comptime OBS = 126
    comptime ACT = 7
    # ResNet torso: conv stem → 5 ResBlocks → FC heads (128 filters), the
    # closest match to the legacy `AlphaZeroConnectFourFusedResNetConfig`.
    # BN epsilon raised 1e-5 → 1e-2: low-diversity post-promotion self-play
    # batches give tiny per-channel batch variance, so the default eps lets
    # train-mode inv_std (=1/√(var+eps)) reach ~316 and amplify activations
    # until the float32 policy logits overflow to −inf. eps=1e-3 only halved the
    # bad logits (640→512); eps=1e-2 caps inv_std at ~10. Paired with the BN
    # running-stats finite-guard (batch_norm_2d/1d), which stops the train-mode
    # blow-up from polluting the eval-mode running stats (the leak that turned
    # every self-play policy NaN → garbage data → plateau).
    comptime Net = AZConnectFourResNet[F=128, NB=5, FC=128, EPS=1e-2]
    comptime Env = ConnectFourEnv[DType.float64]
    # Connect Four's only board symmetry is the left↔right column flip; the
    # board is not square, so the D4 group does NOT apply (no rotations).
    comptime Aug = HFlipColumnAugmenter[ROWS=6, COLS=7, PLANES=3]

    var ctx = DeviceContext()
    # NOTE: nn GPU Conv2D now uses im2col + tensor-core GEMM (was a naive
    # direct-conv kernel that made conv nets 5-10× slower) — the ResNet torso
    # is no longer the per-eval bottleneck. BATCH_SIMS still batches the MCTS
    # rounds (see the tuning notes below).
    var agent = AlphaZeroAgent[
        "gpu",
        Env,
        Net,
        N_ENVS=64,
        # Sim-budget history. 64 sims first looked "below C4's tactical
        # floor" (flat targets, 0% vs minimax) — but the real culprit was the
        # σ(completed_Q) tree-global normalization bug (fixed in the Gumbel
        # planner: per-node completed-Q rescale, mctx semantics), which made
        # 64 and 256 sims IDENTICALLY inert. Post-fix, 256 sims validated the
        # operator: target entropy 0.94→0.23, arena #1 at 81%, Minimax-D5
        # beaten 128-0 both colors by move 8k. Now back to 64 sims — the
        # low-budget regime where Gumbel's policy-improvement guarantee
        # actually lives (~4× faster per move; 16+ visits/candidate, σ ≈ 8
        # nats, still logit-dominant). Watch target entropy in the first ~2k
        # moves: diving below ~0.5 like the 256 run ⇒ the operator works at
        # this budget; sitting high ⇒ too thin, try 128. (256-sim partial
        # baseline: 12k moves, promo 4, W128 vs minimax from move 6k.)
        NUM_SIMS=64,
        MAX_NODES=256,
        BATCH=128,
        CAP=1_000_000,
        MAX_TRAJ=42,
    ](ctx, lr=0.002)

    # ── Remaining deltas vs the legacy AlphaZero.jl-tuned config ──────────────
    # Optimizer stability is now wired: v2 uses AdamW with `max_grad_norm=1.0`
    # + `weight_decay=1e-4` (set on the train_arena call below), matching the
    # legacy config. The first 10k-move run without clipping reproduced the
    # documented failure exactly — policy CE bottomed ~1.18 at move ~2000 then
    # climbed back toward uniform (entropy ↑, vs-Random winrate 69%→62%), and
    # the arena rejected every challenger after the first promotion. Clipping is
    # the #1 fix; WD regularizes the 5-block ResNet. If CE still climbs, drop
    # lr 2e-3 → 1e-3 next.
    #
    # Remaining (still hardcoded in selfplay_arena.mojo, apply if it stalls):
    #   * batch_sims: v2 runs 1 (sequential). Legacy ran 6 → ~6× MCTS speedup at
    #     500-600 sims, plus within-round virtual-loss diversity.
    #   * Dirichlet alpha: v2 = 0.25; legacy = 1.0 (more uniform root noise for C4).
    #   * temp_min after the schedule: v2 = 0.0 (greedy); legacy = 0.3 (soft).
    #   * invalid_action_penalty: v2 = 0; legacy = 1.0 (penalizes illegal-move
    #     policy mass). c_puct 1.0 here ≈ legacy's 2.0 (legacy used raw Q, v2
    #     MinMax-normalizes Q on the GPU path), so do NOT copy 2.0.
    #
    # Full AlphaZero: best/learner Arena gating + horizontal-flip augmentation,
    # evaluated periodically vs 5-ply minimax (primary) and random (secondary).
    # Metrics flush to the logger; progress prints to stdout. `RESULT_IDX=43` is
    # the `S_GAME_RESULT` slot in the Connect Four state; `MAX_PLIES=42` is a
    # full board.
    var res = agent.train_arena_gumbel[
        AUG=Aug,
        OPP1=GPUMinimaxConnectFour[5],
        OPP2=RandomOpponent,
        L=RunLogger,
        ARENA_GAMES=128,
        RESULT_IDX=43,
        MAX_PLIES=42,
        EVAL_GAMES=64,
        # Legacy/AlphaZero.jl sampled ∝ visits for the first 20 plies (then
        # temp=0.3, not full greedy). 8 was a guess; 20 matches the reference.
        TEMP_MOVES=20,
        # Gumbel root candidates (power of two, <= ACT=7).
        MAX_K=4,
    ](
        iterations=40_000,
        learning_starts=200,
        train_per_iter=4,
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
        # Stability (legacy AlphaZero.jl): clip grad norm to 1.0 + decoupled
        # weight decay 1e-4. Without these the policy head diverged (CE climbed
        # back past uniform after ~2000 moves) on this 5-block ResNet at lr=2e-3.
        max_grad_norm=1.0,
        weight_decay=1e-4,
        # Self-play opening diversity: 2 uniform-random legal plies → 49
        # distinct openings. Counters the post-promotion one-hot policy
        # head making deterministic self-play replay one game.
        selfplay_open_plies=2,
        # Eval opening diversity: without it, temp-0 MCTS vs deterministic
        # minimax collapses all EVAL_GAMES to ONE distinct game per color —
        # results quantize to multiples of 64 (W0/W64/W128) and the winrate
        # curve swings wildly between razor-edge canonical lines. 4 random
        # plies (arena convention, ~2401 lines) makes it a real winrate.
        eval_open_plies=4,
    )

    var ckpt = run.checkpoint_path(String("last"))
    agent.save(ckpt)
    announce_checkpoint(ckpt, artifacts, run.dir)
    finish_run(
        run, logger, artifacts,
        String("last_loss=") + String(res.last_loss) + String(" promotions=") + String(res.promotions),
    )

    print()
    print("last_loss:", res.last_loss, "| promotions:", res.promotions)
    print("saved →", ckpt)
    print("=== Done ===")
