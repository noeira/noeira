"""AlphaZero on TicTacToe (deep_agents / nn) — CPU path, no GPU.

CPU twin of `tictactoe_alphazero_v2.mojo`. Same `AlphaZeroAgent` facade and
full-AlphaZero recipe (best/learner Arena gating + D4 symmetry augmentation +
two pluggable eval opponents + telemetry), but `TARGET="cpu"`: the net is
host-resident, MCTS runs through `GenericCPUMCTS` + the true-rules adapters, and
the agent plays a single game at a time (`N_ENVS` is ignored on this path). No
`DeviceContext` is ever created, so this runs anywhere — useful for debugging
the algorithm without a GPU, at the cost of wall-clock (single-env self-play).

The same `GPUMinimaxTicTacToe` / `RandomOpponent` evaluators are used — they are
dual-conforming (`GPUEvaluator & CPUEvaluator`), so the CPU path drives them
through their CPU surface. The periodic eval plays the agent at full **MCTS**
strength (temp=0); a before/after greedy-policy eval vs random brackets the run
to show the net actually learning.

Note `iterations` / `report_every` are in self-play *moves* (one move per loop
pass), not legacy-style collect+train rounds.

Usage:
    pixi run mojo run -I . examples/board_games/tictactoe_alphazero_v2_cpu.mojo

The run (project `board-games`) keeps its checkpoint, `metrics.csv` and
`run.kv` in `runs/<id>/`; with no `NOEIRA_CLOUD_URL` in `.env` the monitor half
is inert and the per-report lines still print to stdout.
"""

from std.memory import Pointer

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.deep_agents.alphazero.nets import AZMLPNet
from noeira.deep_agents.alphazero.agent import AlphaZeroAgent
from noeira.deep_agents.zero.symmetries import D4SquareAugmenter
from noeira.deep_agents.zero.evaluators import (
    RandomOpponent,
    GPUMinimaxTicTacToe,
)
from noeira.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero on TicTacToe (deep_agents / nn) — CPU ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var run = RunContext(
        project=String("board-games"),
        driver=String("examples/board_games/tictactoe_alphazero_v2_cpu.mojo"),
        slug=String("alphazero-cpu-tictactoe"),
        env=String("builtin:tictactoe"),
    )
    print("run:", run.dir)
    var logger = run_logger(run, buffer_size=22)
    logger.set_config("agent", "AlphaZero")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", "AZMLPNet[27,9,128]")
    logger.set_config("framework", "deep_agents/nn")
    logger.set_config("target", "cpu")
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    comptime OBS = 27
    comptime ACT = 9
    comptime H = 128
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime Aug = D4SquareAugmenter[3, 3]  # 8 D4 board symmetries

    # CPU path: no DeviceContext (ctx=None), host-resident net. `N_ENVS` is a
    # required facade param but unused on CPU (single-env self-play), so it is
    # set to 1.
    # CAP is deliberately small: single-env self-play generates few games, so a
    # GPU-sized buffer (80k) never fills and would keep training on the earliest
    # random-play games forever. A small ring evicts stale data and keeps the
    # recent (stronger) games — the legacy `history_window` idea. Even so, CPU
    # accuracy is bounded by raw game throughput (one game per ~9 moves); to
    # actually draw minimax as *both* colors it needs far more iterations than
    # the GPU example (which fans out over N_ENVS), or just use the GPU path.
    var agent = AlphaZeroAgent[
        "cpu",
        Env,
        Net,
        N_ENVS=1,
        NUM_SIMS=50,
        MAX_NODES=128,
        BATCH=64,
        CAP=16000,
        MAX_TRAJ=16,
    ](None, lr=0.005)

    # Baseline: greedy policy head (search-free) vs random before training.
    # NB: this is the bare policy net; the *deployed* agent adds MCTS on top
    # (the "vs Random/Minimax" lines in the report below are full-strength).
    var before = agent.eval_vs_random_cpu[200, 9](agent_player=0, seed=12345)
    print(
        "BEFORE (greedy policy vs random)  win=",
        before.wins,
        " draw=",
        before.draws,
        " loss=",
        before.losses,
    )
    print()

    # Full AlphaZero: best/learner Arena gating + D4 augmentation, evaluated
    # periodically vs minimax (primary) and random (secondary). Metrics flush
    # to the logger; progress prints to stdout. Iteration count is trimmed vs
    # the GPU example since CPU self-play is single-env (slower wall-clock).
    var res = agent.train_arena[
        AUG=Aug,
        OPP1=GPUMinimaxTicTacToe,
        OPP2=RandomOpponent,
        L=RunLogger,
        ARENA_GAMES=20,
        RESULT_IDX=10,
        MAX_PLIES=9,
        EVAL_GAMES=32,
    ](
        iterations=10_000,
        learning_starts=20,
        train_per_iter=4,
        seed=42,
        arena_every=400,
        arena_open_plies=2,
        promote_threshold=0.55,
        report_every=200,
        diag_every=20,
        do_eval=True,
        do_eval2=True,
        verbose=True,
        logger=Pointer(to=logger).as_unsafe_any_origin(),
    )

    # Endline: greedy net-policy vs random after training (should clearly beat
    # the baseline). `agent.save` checkpoints the net through the storage
    # facade — it threads `self.ctx` (None on this CPU path), so the
    # host-resident net is written via the weights-only `save_params` surface.
    var after = agent.eval_vs_random_cpu[200, 9](agent_player=0, seed=12345)
    var ckpt = run.checkpoint_path(String("last"))
    agent.save(ckpt)
    announce_checkpoint(ckpt, artifacts, run.dir)
    finish_run(
        run, logger, artifacts,
        String("win_vs_random=") + String(after.wins) + String("/200")
        + String(" draw=") + String(after.draws)
        + String(" loss=") + String(after.losses),
    )

    print()
    print(
        "AFTER  (greedy policy vs random)  win=",
        after.wins,
        " draw=",
        after.draws,
        " loss=",
        after.losses,
    )
    print("last_loss:", res.last_loss, "| promotions:", res.promotions)
    print("saved →", ckpt)
    print("=== Done ===")
