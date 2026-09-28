"""AlphaZeroAgent — facade over the self-play driver + eval + checkpointing.

Parameterized on `TARGET` ("cpu" / "gpu"), mirroring the SAC dual-path pattern:
construct with a `DeviceContext` for "gpu" (or `None` for "cpu"); the net is
built host- or device-resident via `NET.make[TARGET]`, and `train` routes to the
matching driver. Packages the loose functions into a usable API:

  * `__init__(ctx, lr)`  — build a fresh agent (Kaiming-init net on `TARGET`).
  * `train(iterations)`  — self-play training; "gpu" runs `N_ENVS` batched games
                           through `GenericGPUMCTS`, "cpu" plays one game at a
                           time through `GenericCPUMCTS` (true-rules adapters).
  * `train_arena(...)`   — full-AlphaZero arena gating + telemetry (GPU).
  * `eval_vs_random` / `eval_vs_random_cpu` / `eval_mcts` — strength checks.
  * `save` / `load`      — one-file v3 `storage-ckpt` checkpoint of the net.

The net's params persist across `train` calls (training continues from the
current weights); the optimizer + replay are session-local (recreated per
`train` call) — fine for the board-game scale here, revisit for long
incremental runs.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.checkpoint import save_params, load_params
from noeira.core import TwoPlayerDiscreteEnv, Saveable
from noeira.core.env_traits import GPUTwoPlayerDiscreteEnv
from noeira.core.logger import Logger, NoOpLogger

from ..zero.evaluators import GPUEvaluator, CPUEvaluator, RandomOpponent
from ..zero.symmetries import BoardAugmenter, IdentityAugmenter
from .selfplay import run_alphazero_selfplay
from .selfplay_gumbel import run_alphazero_gumbel_selfplay
from .selfplay_arena_gumbel import run_alphazero_selfplay_arena_gumbel
from .selfplay_cpu import run_alphazero_selfplay_cpu
from .selfplay_arena import run_alphazero_selfplay_arena, ArenaRunResult
from .selfplay_arena_cpu import run_alphazero_selfplay_arena_cpu
from .eval import (
    eval_policy_vs_random,
    eval_policy_vs_random_cpu,
    eval_policy_vs_opponent,
    eval_mcts_vs_opponent,
    EvalResult,
)


@fieldwise_init
struct AlphaZeroAgent[
    TARGET: StaticString,
    ENV: GPUTwoPlayerDiscreteEnv & TwoPlayerDiscreteEnv & Saveable & Defaultable & Deinitable,
    NET: Module,
    N_ENVS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    BATCH: Int,
    CAP: Int,
    MAX_TRAJ: Int,
](Deinitable, Movable):
    var ctx: Optional[DeviceContext]
    var net: Self.NET
    var lr: Scalar[DT]

    def __init__(
        out self,
        ctx: Optional[DeviceContext],
        lr: Scalar[DT] = Scalar[DT](0.01),
    ) raises:
        self.ctx = ctx
        self.net = Self.NET.make[Self.TARGET, Kaiming](ctx)
        self.lr = lr

    def train_gumbel[
        MAX_K: Int = 4,
    ](
        mut self,
        iterations: Int,
        learning_starts: Int = 20,
        train_per_iter: Int = 2,
        seed: UInt64 = 0,
    ) raises -> Float64:
        """Gumbel AlphaZero self-play (GPU): Gumbel-Top-k roots + Sequential
        Halving + improved-policy targets — beats the PUCT driver at equal
        sims on the TTT gate (losses vs random 8 vs 24 / 200). ``MAX_K`` is
        the root candidate count (power of two, <= ACT). GPU-only: the Gumbel
        planner is device-side (decision D1)."""
        comptime assert Self.TARGET == "gpu", (
            "train_gumbel is GPU-only (Gumbel planner is device-side)"
        )
        return run_alphazero_gumbel_selfplay[
            Self.ENV,
            Self.NET,
            Self.N_ENVS,
            Self.NUM_SIMS,
            Self.MAX_NODES,
            MAX_K,
            Self.BATCH,
            Self.CAP,
            Self.MAX_TRAJ,
        ](
            self.ctx.value(),
            self.net,
            iterations,
            learning_starts,
            train_per_iter,
            self.lr,
            seed,
        )

    def train(
        mut self,
        iterations: Int,
        learning_starts: Int = 20,
        train_per_iter: Int = 2,
        seed: UInt64 = 0,
    ) raises -> Float64:
        """Self-play training. Routes on `TARGET`: the GPU path runs `N_ENVS`
        batched games through `GenericGPUMCTS`; the CPU path plays a single game
        at a time through `GenericCPUMCTS` (true-rules adapters). Both share the
        AZ loss graph, replay, and value-target convention."""
        comptime if Self.TARGET == "gpu":
            return run_alphazero_selfplay[
                Self.ENV,
                Self.NET,
                Self.N_ENVS,
                Self.NUM_SIMS,
                Self.MAX_NODES,
                Self.BATCH,
                Self.CAP,
                Self.MAX_TRAJ,
            ](
                self.ctx.value(),
                self.net,
                iterations,
                learning_starts,
                train_per_iter,
                self.lr,
                seed,
            )
        else:
            return run_alphazero_selfplay_cpu[
                Self.ENV,
                Self.NET,
                Self.NUM_SIMS,
                Self.MAX_NODES,
                Self.BATCH,
                Self.CAP,
                Self.MAX_TRAJ,
            ](
                self.net,
                iterations,
                learning_starts,
                train_per_iter,
                self.lr,
                seed,
            )

    def train_arena_gumbel[
        AUG: BoardAugmenter = IdentityAugmenter,
        OPP1: GPUEvaluator & CPUEvaluator = RandomOpponent,
        OPP2: GPUEvaluator & CPUEvaluator = RandomOpponent,
        L: Logger = NoOpLogger,
        ARENA_GAMES: Int = 32,
        RESULT_IDX: Int = 10,
        MAX_PLIES: Int = 9,
        EVAL_GAMES: Int = 64,
        TEMP_MOVES: Int = 4,
        MAX_K: Int = 4,
    ](
        mut self,
        iterations: Int,
        learning_starts: Int = 20,
        train_per_iter: Int = 2,
        seed: UInt64 = 0,
        arena_every: Int = 100,
        arena_open_plies: Int = 2,
        promote_threshold: Float64 = 0.55,
        report_every: Int = 0,
        diag_every: Int = 0,
        do_eval: Bool = True,
        do_eval2: Bool = False,
        verbose: Bool = True,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        max_grad_norm: Float64 = 0.0,
        weight_decay: Float64 = 0.0,
        selfplay_open_plies: Int = 2,
        eval_open_plies: Int = 0,
    ) raises -> ArenaRunResult:
        """`train_arena` with GUMBEL self-play (GPU-only): Gumbel-Top-k roots
        + Sequential Halving + improved-policy targets, ``MAX_K`` root
        candidates. Arena gating and the periodic MCTS evals keep the exact
        PUCT machinery of `train_arena` — the same yardstick measures both
        drivers, so runs compare directly. Sequential halving earns its
        policy-improvement guarantee at LOW sim budgets: run with NUM_SIMS
        ~32-64, not the PUCT path's 500."""
        comptime assert Self.TARGET == "gpu", (
            "train_arena_gumbel is GPU-only (Gumbel planner is device-side)"
        )
        return run_alphazero_selfplay_arena_gumbel[
            Self.ENV,
            Self.NET,
            AUG,
            Self.N_ENVS,
            Self.NUM_SIMS,
            Self.MAX_NODES,
            Self.BATCH,
            Self.CAP,
            Self.MAX_TRAJ,
            ARENA_GAMES,
            RESULT_IDX,
            MAX_PLIES,
            OPP1,
            OPP2,
            L,
            EVAL_GAMES,
            TEMP_MOVES,
            MAX_K,
        ](
            self.ctx.value(),
            self.net,
            iterations,
            learning_starts,
            train_per_iter,
            self.lr,
            seed,
            arena_every,
            arena_open_plies,
            promote_threshold,
            report_every,
            diag_every,
            do_eval,
            do_eval2,
            verbose,
            logger,
            max_grad_norm,
            weight_decay,
            selfplay_open_plies,
            eval_open_plies,
        )

    def train_arena[
        AUG: BoardAugmenter = IdentityAugmenter,
        OPP1: GPUEvaluator & CPUEvaluator = RandomOpponent,
        OPP2: GPUEvaluator & CPUEvaluator = RandomOpponent,
        L: Logger = NoOpLogger,
        ARENA_GAMES: Int = 32,
        RESULT_IDX: Int = 10,
        MAX_PLIES: Int = 9,
        EVAL_GAMES: Int = 64,
        TEMP_MOVES: Int = 4,
        BATCH_SIMS: Int = 1,
    ](
        mut self,
        iterations: Int,
        learning_starts: Int = 20,
        train_per_iter: Int = 2,
        seed: UInt64 = 0,
        arena_every: Int = 100,
        arena_open_plies: Int = 2,
        promote_threshold: Float64 = 0.55,
        report_every: Int = 0,
        diag_every: Int = 0,
        do_eval: Bool = True,
        do_eval2: Bool = False,
        verbose: Bool = True,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        max_grad_norm: Float64 = 0.0,
        weight_decay: Float64 = 0.0,
        eval_open_plies: Int = 0,  # GPU path only (CPU eval is single-line)
    ) raises -> ArenaRunResult:
        """Full-AlphaZero training: best/learner split + Arena gating +
        symmetry augmentation, with two pluggable eval opponents and a logger.

        Defaults (`AUG=Identity`, `OPP*=Random`, `L=NoOp`, `report_every=0`,
        `diag_every=0`) reduce to a silent arena run. Set `report_every>0` (+ a
        logger and/or `OPP1=GPUMinimaxTicTacToe`, `do_eval2=True`) for per-report
        MCTS-eval+print+metric flush. Set `diag_every>0` for dense per-batch
        training diagnostics (policy CE, entropy, value MSE, target stats —
        cheap, decoupled from the expensive eval), mirroring the legacy
        `train_selfplay_gpu` telemetry. The
        periodic eval plays the agent at **full MCTS strength** (temp=0), not the
        bare policy head — `iterations`/`report_every` are in self-play *moves*.
        Routes on `TARGET`: GPU runs `N_ENVS` batched games; CPU plays one game
        at a time. The evaluators conform to both `GPUEvaluator` and
        `CPUEvaluator`, so the same `OPP1`/`OPP2` work on either path."""
        comptime if Self.TARGET == "gpu":
            return run_alphazero_selfplay_arena[
                Self.ENV,
                Self.NET,
                AUG,
                Self.N_ENVS,
                Self.NUM_SIMS,
                Self.MAX_NODES,
                Self.BATCH,
                Self.CAP,
                Self.MAX_TRAJ,
                ARENA_GAMES,
                RESULT_IDX,
                MAX_PLIES,
                OPP1,
                OPP2,
                L,
                EVAL_GAMES,
                TEMP_MOVES,
                BATCH_SIMS,
            ](
                self.ctx.value(),
                self.net,
                iterations,
                learning_starts,
                train_per_iter,
                self.lr,
                seed,
                arena_every,
                arena_open_plies,
                promote_threshold,
                report_every,
                diag_every,
                do_eval,
                do_eval2,
                verbose,
                logger,
                max_grad_norm,
                weight_decay,
                eval_open_plies,
            )
        else:
            return run_alphazero_selfplay_arena_cpu[
                Self.ENV,
                Self.NET,
                AUG,
                Self.NUM_SIMS,
                Self.MAX_NODES,
                Self.BATCH,
                Self.CAP,
                Self.MAX_TRAJ,
                ARENA_GAMES,
                MAX_PLIES,
                OPP1,
                OPP2,
                L,
                EVAL_GAMES,
                TEMP_MOVES,
                BATCH_SIMS,
            ](
                self.net,
                iterations,
                learning_starts,
                train_per_iter,
                self.lr,
                seed,
                arena_every,
                arena_open_plies,
                promote_threshold,
                report_every,
                diag_every,
                do_eval,
                do_eval2,
                verbose,
                logger,
                max_grad_norm,
                weight_decay,
            )

    def eval_mcts[
        OPP: GPUEvaluator,
        N_EVAL: Int,
        NUM_SIMS_EVAL: Int,
        MAX_NODES_EVAL: Int,
        MAX_PLIES: Int,
    ](mut self, agent_player: Int = 0, seed: UInt64 = 1) raises -> EvalResult:
        """Full-strength eval: agent plays via MCTS (temp=0) vs `OPP`. This is
        the deployed-agent metric — the policy head alone cannot draw perfect
        minimax, MCTS on top can. Pass the agent's own `Self.NUM_SIMS` /
        `Self.MAX_NODES` for parity with training-time search."""
        return eval_mcts_vs_opponent[
            Self.ENV,
            Self.NET,
            OPP,
            N_EVAL,
            NUM_SIMS_EVAL,
            MAX_NODES_EVAL,
            MAX_PLIES,
        ](self.ctx.value(), self.net, agent_player, seed)

    def eval_vs_random[
        N_EVAL: Int, RESULT_IDX: Int, MAX_PLIES: Int
    ](mut self, agent_player: Int = 0, seed: UInt64 = 1) raises -> EvalResult:
        return eval_policy_vs_random[
            Self.ENV, Self.NET, N_EVAL, RESULT_IDX, MAX_PLIES
        ](self.ctx.value(), self.net, agent_player, seed)

    def eval_vs_random_cpu[
        N_EVAL: Int, MAX_PLIES: Int
    ](mut self, agent_player: Int = 0, seed: UInt64 = 1) raises -> EvalResult:
        """CPU greedy-policy eval vs a random opponent (single CPU env). The CPU
        counterpart to `eval_vs_random` for `TARGET="cpu"` agents whose net is
        host-resident."""
        return eval_policy_vs_random_cpu[
            Self.ENV, Self.NET, N_EVAL, MAX_PLIES
        ](self.net, agent_player, seed)

    def eval_vs_opponent[
        OPP: GPUEvaluator, N_EVAL: Int, RESULT_IDX: Int, MAX_PLIES: Int
    ](
        mut self, agent_player: Int = 0, seed: UInt64 = 1, open_plies: Int = 0
    ) raises -> EvalResult:
        """Greedy net-policy vs an arbitrary `GPUEvaluator` (e.g. minimax).

        `open_plies` randomises the first plies for diverse openings — needed
        for deterministic opponents (see `eval_policy_vs_opponent`)."""
        return eval_policy_vs_opponent[
            Self.ENV, Self.NET, OPP, N_EVAL, RESULT_IDX, MAX_PLIES
        ](self.ctx.value(), self.net, agent_player, seed, open_plies)

    def save(mut self, path: String) raises:
        """Weights-only snapshot of the policy/value net (v3 `storage-ckpt`
        envelope, section `net`). Uses the `save`/`load` surface shared by
        every agent facade. NOTE: optimizers are session-local — rebuilt
        fresh inside each `train_*` call — so there is no persistent
        optimizer state to checkpoint; this is the inference / self-play
        artifact, not a training-resume checkpoint. Self-play buffers are
        not included."""
        # Weights-only (no optimizer moments) via the storage checkpoint, which
        # walks `for_each_param` then `for_each_state` (so BatchNorm running
        # stats persist) and owns the file I/O.
        save_params[Self.TARGET, Self.NET](
            self.net, path, self.ctx, save_moments=False
        )

    def load(mut self, path: String) raises:
        """Inverse of `save` — restores the net weights. See `save` for why
        optimizer state is not part of the checkpoint."""
        load_params[Self.TARGET, Self.NET](self.net, path, self.ctx)
