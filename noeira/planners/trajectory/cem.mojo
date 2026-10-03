"""Categorical Cross-Entropy Method (CEM) optimizer.

Host-side per-step categorical CEM for discrete actions. Maintains
``action_dist[BATCH, horizon, ACT_DIM]`` (per batch-row × per timestep ×
per action probability mass), samples candidate plans, scores each via
a ``ScorePlanCallback``, then refits the distribution to the top-K
plans using add-K (Laplace) smoothing.

Promoted from the first LeWM port's ``kernels._run_cem_eval_iter`` (removed).
The original was specialized to LeWM's autoregressive MPC rollout; this
version is agent-agnostic — the scoring step delegates to a callback,
so LeWM, future MBPO-CEM, or any other discrete-action world-model
agent can reuse the same struct.

**Batch semantics**: action distribution is per-batch-row (different
rows can converge to different plans), but the **score is one scalar
per sample, aggregated across the batch**. Elite selection ranks
across the sample axis only — this matches the LeWM convention where
the world-model rollout produces one MSE-to-goal per (sample, batch
row) and the optimizer pools them.

**Why HORIZON is runtime, not comptime.** Agents like LeWM expose the
plan length (``mpc_horizon`` / ``needed_actions``) as a runtime ctor
arg; pushing it to comptime would cascade into every container struct
(``CEMPlanner``, ``LeWMEvalSuite``, ``LeWMTrainer``) and break the
no-rebuild horizon-sweep workflow. The host-side CEM loop is not
perf-critical compared to the GPU rollout it scores, so the lost
constant-folding is negligible.

**Storage.** Host scratch lives in ``List`` (heap-allocated, safe
ownership) rather than ``Pointer`` — no manual ``__deinit__``, no
move-constructor boilerplate. ``TileTensor`` views are built on the
fly inside each method that needs 3-D / 4-D indexed access; the view
construction is free at runtime (just a pointer + Coord-layout
struct).

Usage::

    var planner = CategoricalCEMOptimizer[BATCH, ACT_DIM](
        horizon=4, cem_iters=5, cem_samples=64, cem_topk=8, cem_smoothing=0.5,
    )
    var callback = MyAgentScoreCallback(...)
    var best = planner.optimize(callback, best_plan_out_ptr)

See ``docs/PLANNERS_PACKAGE.md`` Phase 1.
"""

from std.random import random_float64

from layout import TileTensor, Idx, row_major

from noeira.nn.constants import DT as dtype

from .score_callback import ScorePlanCallback, BatchedScorePlanCallback


struct CategoricalCEMOptimizer[BATCH: Int, ACT_DIM: Int](
    Deinitable,
    Movable,
):
    """Per-step categorical CEM optimizer.

    Owns host scratch only — no GPU buffers. GPU work (action upload,
    model rollout, score reduction) lives inside the
    ``ScorePlanCallback`` the caller supplies.

    Comptime params fix the batch + action dims; ``horizon`` and CEM
    hyperparameters come in as runtime ctor args.
    """

    var horizon: Int
    var cem_iters: Int
    var cem_samples: Int
    var cem_topk: Int
    var cem_smoothing: Float64

    # Host scratch (List-owned; freed automatically when the optimizer
    # is destroyed).
    var action_dist: List[Scalar[dtype]]
    """`(BATCH, horizon, ACT_DIM)` — per-batch-row categorical at each step."""

    var sample_actions: List[Scalar[dtype]]
    """`(cem_samples, BATCH, horizon, ACT_DIM)` — sampled one-hot plans."""

    var sample_plan: List[Scalar[dtype]]
    """`(BATCH, horizon, ACT_DIM)` — scratch holding the *current* sample
    plan, layout that ``ScorePlanCallback.score_plan`` expects."""

    var sample_scores: List[Float64]
    """`(cem_samples,)` — one score per sampled plan."""

    var elite_indices: List[Int]
    """`(cem_topk,)` — indices of top-K samples by lowest score."""

    def __init__(
        out self,
        horizon: Int,
        cem_iters: Int,
        cem_samples: Int,
        cem_topk: Int,
        cem_smoothing: Float64 = 0.5,
    ) raises:
        if horizon < 1:
            raise Error("CategoricalCEMOptimizer: horizon must be >= 1")
        if cem_iters < 0:
            raise Error("CategoricalCEMOptimizer: cem_iters must be >= 0")
        if cem_samples < 1:
            raise Error("CategoricalCEMOptimizer: cem_samples must be >= 1")
        if cem_topk < 1 or cem_topk > cem_samples:
            raise Error(
                "CategoricalCEMOptimizer: cem_topk must be in [1, cem_samples]"
            )
        if cem_smoothing < 0.0:
            raise Error("CategoricalCEMOptimizer: cem_smoothing must be >= 0")

        self.horizon = horizon
        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.cem_topk = cem_topk
        self.cem_smoothing = cem_smoothing

        var plan_size = Self.BATCH * horizon * Self.ACT_DIM
        self.action_dist = List[Scalar[dtype]](
            length=plan_size, fill=Scalar[dtype](0)
        )
        self.sample_actions = List[Scalar[dtype]](
            length=cem_samples * plan_size, fill=Scalar[dtype](0)
        )
        self.sample_plan = List[Scalar[dtype]](
            length=plan_size, fill=Scalar[dtype](0)
        )
        self.sample_scores = List[Float64](length=cem_samples, fill=0.0)
        self.elite_indices = List[Int](length=cem_topk, fill=-1)

    def _init_uniform_dist(mut self):
        var inv_act = Scalar[dtype](1.0 / Float64(Self.ACT_DIM))
        var dist = TileTensor(
            self.action_dist,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                for a in range(Self.ACT_DIM):
                    dist[b, t, a] = inv_act

    def _sample_plan(mut self, sample_idx: Int):
        """Sample one-hot plan from `action_dist`. Writes:
        - `sample_actions[sample_idx, :, :, :]` for later elite recall.
        - `sample_plan[:, :, :]` so the caller can hand it to the
          score callback verbatim.
        """
        var dist = TileTensor(
            self.action_dist,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        var all_samples = TileTensor(
            self.sample_actions,
            row_major(
                (
                    self.cem_samples,
                    Idx[Self.BATCH],
                    self.horizon,
                    Idx[Self.ACT_DIM],
                )
            ),
        )
        var plan = TileTensor(
            self.sample_plan,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                var r = random_float64()
                var cumul: Float64 = 0.0
                var picked = Self.ACT_DIM - 1
                for a in range(Self.ACT_DIM):
                    cumul += Float64(dist[b, t, a])
                    if r < cumul:
                        picked = a
                        break
                for a in range(Self.ACT_DIM):
                    var v = Scalar[dtype](1.0) if a == picked else Scalar[
                        dtype
                    ](0.0)
                    all_samples[sample_idx, b, t, a] = v
                    plan[b, t, a] = v

    def _pick_elites(mut self):
        # Mark all elite slots as -1, then greedy-pick the lowest score
        # K times (skipping already-picked indices). O(K * S) — fine for
        # typical K ≤ 16 and S ≤ 256.
        for k in range(self.cem_topk):
            self.elite_indices[k] = -1
        for k in range(self.cem_topk):
            var best_idx: Int = -1
            var best_score: Float64 = 1.0e30
            for s in range(self.cem_samples):
                var already_picked = False
                for kk in range(k):
                    if self.elite_indices[kk] == s:
                        already_picked = True
                        break
                if not already_picked and self.sample_scores[s] < best_score:
                    best_score = self.sample_scores[s]
                    best_idx = s
            self.elite_indices[k] = best_idx

    def _refit_dist(mut self):
        """Refit per-step categorical from elites with add-K smoothing.

        action_dist[b, t, a] = (count[b, t, a] + smoothing)
                                / (topk + ACT_DIM * smoothing)
        Effectively a Dirichlet-conjugate posterior with uniform prior.
        """
        var denom = (
            Float64(self.cem_topk) + Float64(Self.ACT_DIM) * self.cem_smoothing
        )
        var dist = TileTensor(
            self.action_dist,
            row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
        )
        var all_samples = TileTensor(
            self.sample_actions,
            row_major(
                (
                    self.cem_samples,
                    Idx[Self.BATCH],
                    self.horizon,
                    Idx[Self.ACT_DIM],
                )
            ),
        )
        var prior = Scalar[dtype](self.cem_smoothing / denom)
        var vote = Scalar[dtype](1.0 / denom)
        for b in range(Self.BATCH):
            for t in range(self.horizon):
                # Reset to prior (smoothing / denom).
                for a in range(Self.ACT_DIM):
                    dist[b, t, a] = prior
                # Add `vote` for each elite vote at this (b, t).
                for k in range(self.cem_topk):
                    var e = self.elite_indices[k]
                    for a in range(Self.ACT_DIM):
                        if all_samples[e, b, t, a] > Scalar[dtype](0.5):
                            dist[b, t, a] += vote
                            break

    def optimize[
        CB: ScorePlanCallback
    ](
        mut self,
        mut callback: CB,
        best_plan_out: Pointer[Scalar[dtype], MutAnyOrigin],
        verbose: Bool = True,
    ) raises -> Float64:
        """Run `cem_iters` rounds of sample → score → top-K → refit.

        Returns the best score observed over the entire optimization.
        Writes the best-scoring plan into ``best_plan_out`` (shape
        ``(BATCH, horizon, ACT_DIM)``).

        ``best_plan_out`` is still raw-pointer-typed because it's a
        caller-supplied output slot whose ownership lives outside the
        optimizer. The optimizer wraps it in a ``TileTensor`` view
        internally for the writeback.
        """
        self._init_uniform_dist()

        var best_overall: Float64 = 1.0e30
        var best_overall_sample: Int = -1

        for cem_it in range(self.cem_iters):
            # Sample + score.
            #
            # The plan TileTensor view is rebuilt per sample so the
            # callback receives a view typed against the current host
            # ``self.sample_plan`` buffer (the storage doesn't move,
            # but the view is a value type so it's cheap to recreate).
            for s in range(self.cem_samples):
                self._sample_plan(s)
                var plan_view = TileTensor(
                    self.sample_plan,
                    row_major(
                        (
                            Idx[Self.BATCH],
                            self.horizon,
                            Idx[Self.ACT_DIM],
                        )
                    ),
                )
                var score = callback.score_plan(plan_view)
                self.sample_scores[s] = score
                if score < best_overall:
                    best_overall = score
                    best_overall_sample = s

            # Elites + refit.
            self._pick_elites()
            self._refit_dist()

            if verbose:
                var iter_best: Float64 = 1.0e30
                for s in range(self.cem_samples):
                    if self.sample_scores[s] < iter_best:
                        iter_best = self.sample_scores[s]
                print("    cem iter", cem_it, " best=", iter_best)

        # Write the best plan over the entire optimization back to caller.
        # If cem_iters == 0 (no optimization), leave best_plan_out untouched
        # and return +inf to signal "nothing optimized".
        if best_overall_sample >= 0:
            var all_samples = TileTensor(
                self.sample_actions,
                row_major(
                    (
                        self.cem_samples,
                        Idx[Self.BATCH],
                        self.horizon,
                        Idx[Self.ACT_DIM],
                    )
                ),
            )
            var dst = TileTensor(
                best_plan_out,
                row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
            )
            for b in range(Self.BATCH):
                for t in range(self.horizon):
                    for a in range(Self.ACT_DIM):
                        dst[b, t, a] = all_samples[best_overall_sample, b, t, a]

        return best_overall

    def optimize_batched[
        CB: BatchedScorePlanCallback
    ](
        mut self,
        mut callback: CB,
        best_plan_out: Pointer[Scalar[dtype], MutAnyOrigin],
        verbose: Bool = True,
    ) raises -> Float64:
        """Run ``cem_iters`` rounds of sample → score → top-K → refit
        with the K candidates of each iter scored in ONE batched GPU
        call.

        Mirror of ``optimize`` but uses ``BatchedScorePlanCallback`` so
        the callback can amortize host-sync + kernel-launch overhead
        across all ``cem_samples`` plans per iter. Algorithmically
        equivalent — elite picking + refit are still host-side and
        unchanged.

        Returns the best score observed and writes the best-scoring plan
        into ``best_plan_out`` (shape ``(BATCH, horizon, ACT_DIM)``).
        """
        self._init_uniform_dist()

        var best_overall: Float64 = 1.0e30
        var best_overall_sample: Int = -1

        for cem_it in range(self.cem_iters):
            # Sample all K plans into ``sample_actions``. We deliberately
            # do NOT re-fill ``sample_plan`` per sample (the batched
            # callback reads directly from the (K, B, H, A) buffer).
            for s in range(self.cem_samples):
                self._sample_plan(s)

            var all_samples = TileTensor(
                self.sample_actions,
                row_major(
                    (
                        self.cem_samples,
                        Idx[Self.BATCH],
                        self.horizon,
                        Idx[Self.ACT_DIM],
                    )
                ),
            )
            callback.score_plans_batched(all_samples, self.sample_scores)

            for s in range(self.cem_samples):
                var score = self.sample_scores[s]
                if score < best_overall:
                    best_overall = score
                    best_overall_sample = s

            # Elites + refit (host-side, unchanged).
            self._pick_elites()
            self._refit_dist()

            if verbose:
                var iter_best: Float64 = 1.0e30
                for s in range(self.cem_samples):
                    if self.sample_scores[s] < iter_best:
                        iter_best = self.sample_scores[s]
                print("    cem iter", cem_it, " best=", iter_best)

        if best_overall_sample >= 0:
            var all_samples = TileTensor(
                self.sample_actions,
                row_major(
                    (
                        self.cem_samples,
                        Idx[Self.BATCH],
                        self.horizon,
                        Idx[Self.ACT_DIM],
                    )
                ),
            )
            var dst = TileTensor(
                best_plan_out,
                row_major((Idx[Self.BATCH], self.horizon, Idx[Self.ACT_DIM])),
            )
            for b in range(Self.BATCH):
                for t in range(self.horizon):
                    for a in range(Self.ACT_DIM):
                        dst[b, t, a] = all_samples[best_overall_sample, b, t, a]

        return best_overall
