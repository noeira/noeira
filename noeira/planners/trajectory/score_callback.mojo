"""Score-plan callback traits — host-side rollout score contracts.

Two traits live here:

* ``ScorePlanCallback`` — score ONE candidate plan per call.
  ``(BATCH, HORIZON, ACT_DIM)`` host-side plan → one Float64. Used by
  legs that only ever score one plan at a time (e.g. the expert leg of
  LeWM eval) and as the default contract for host-driven CEM/RS.

* ``BatchedScorePlanCallback`` — score K candidate plans in ONE GPU
  pass. ``(K, BATCH, HORIZON, ACT_DIM)`` host-side plan → ``(K,)`` score
  output. Lets host-side optimizers amortize the per-call host sync +
  kernel-launch overhead across the entire candidate budget (~K* fewer
  syncs per optimization round). Implementing this is opt-in; LeWM
  uses it for the random-shooter and CEM iters where K is large
  (``num_samples``, ``cem_samples``).

A given callback can implement one or both. ``LeWMRolloutScoreCallback``
implements ``ScorePlanCallback``; ``LeWMRolloutScoreBatchedCallback``
implements both (with single-plan delegating to a K=1 batched call when
convenient).

Everything GPU-side (action upload, model rollout, score reduction)
lives inside the callback. ``BATCH``, ``HORIZON``, ``ACT_DIM`` are
implicit in the contract — optimizer and callback are constructed
against the same agent config, so the buffer layout is fixed by
construction.

Concrete implementations live next to the agent that owns the world
model — e.g. a LeWM rollout (``noeira/experimental/lewm/ref_rollout.mojo``)
wraps LeWM's autoregressive MPC shot. New planners (MPPI, iLQR) can
reuse the same traits when they need host-side scoring; planners that
keep the rollout fully on-device (the planned ``GPUMCTS``) don't use
these.

Plans are passed as ``TileTensor`` rather than raw pointers. The layout
type is a method-level comptime param (``L: TensorLayout``), so
implementations work for any plan layout the optimizer constructs.
Implementors trust the contract and index the tensor as
``plan[b, t, a]`` (3-D) or ``plans[k, b, t, a]`` (4-D).
"""

from layout import TileTensor, TensorLayout

from noeira.nn.constants import DT as dtype


trait ScorePlanCallback(Deinitable):
    """Score a candidate action plan for trajectory optimization.

    The callback owns whatever scratch is needed to evaluate a plan
    (GPU buffers, world-model views, etc.). The optimizer treats it as
    a black box: pass a tile-tensor plan, receive a scalar.

    Implementations must agree with the caller on the
    ``(BATCH, HORIZON, ACT_DIM)`` shape of ``action_plan`` (3D, flat
    rank == 3). A lower score is better by convention — e.g. CEM picks
    the K plans with the smallest score values.
    """

    def score_plan[
        L: TensorLayout
    ](
        mut self,
        action_plan: TileTensor[dtype, L, MutAnyOrigin],
    ) raises -> Float64:
        """Score `action_plan` of shape (BATCH, HORIZON, ACT_DIM).

        Returns a single scalar — typically sum-of-MSEs or sum-of-rewards
        aggregated across the batch. Lower is better.
        """
        ...


trait BatchedScorePlanCallback(Deinitable):
    """Score K candidate action plans in a single host call.

    Batched analogue of ``ScorePlanCallback``. The optimizer hands the
    callback a ``(K, BATCH, HORIZON, ACT_DIM)`` host tile-tensor and
    expects ``K`` Float64 scores written into ``scores_out``
    (caller-owned ``List[Float64]`` of length ``K``).

    Implementors are encouraged to run a SINGLE rollout (effective batch
    K*BATCH) or a single host-sync-free K-loop on the GPU — the whole
    point of this trait is to amortize per-call sync + launch overhead.

    A lower score is better by convention. Implementations must agree
    with the caller on the ``(K, BATCH, HORIZON, ACT_DIM)`` shape (4D,
    flat rank == 4).
    """

    def score_plans_batched[
        L: TensorLayout
    ](
        mut self,
        action_plans: TileTensor[dtype, L, MutAnyOrigin],
        mut scores_out: List[Float64],
    ) raises:
        """Score ``action_plans`` of shape (K, BATCH, HORIZON, ACT_DIM).

        ``scores_out`` is caller-allocated with length == K (the leading
        dim of ``action_plans``); the callback writes one Float64 per
        candidate plan. Lower is better.
        """
        ...
