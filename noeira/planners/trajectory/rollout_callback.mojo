"""RolloutCallback traits — per-step model contract for trajectory optimizers.

These traits are the Phase-2 generalization of ``ScorePlanCallback``
(which scored a whole plan in one call). MPPI / iLQR / future MPC need
finer access: each candidate trajectory is rolled out step-by-step, with
the optimizer choosing actions and the callback advancing latent state
and producing per-step rewards plus a terminal value bootstrap.

Two traits, not one, because the CPU and GPU worlds have genuinely
different interfaces:

* ``RolloutCallbackCPU`` operates on ``List[Float64]`` views. B = 1 per
  call — the planner loops over samples / timesteps. Used by
  ``MPPICPU`` and by isolated stub-model tests
  (``LinearQuadratic``, etc.) where a List interface keeps the test
  fixture trivial.

* ``RolloutCallbackGPU`` operates on row-major ``LayoutTensor`` views
  with a method-level ``B`` comptime parameter, matching the existing
  TDMPC2 kernel signatures. The planner batches all
  ``N_ENVS × TOTAL_SAMPLES`` trajectories into one call per
  per-horizon-step. Used by ``MPPIGPUBatched`` and TDMPC2 production.

A given implementor can satisfy one or both. ``LinearQuadratic`` stubs
implement only CPU; ``TDMPC2RolloutCallback`` implements only GPU (the
TDMPC2 agent never invokes the CPU planner on a real world model).
CPU↔GPU parity tests use a stub that implements both — these are the
only consumers that need both surfaces.

Three methods per trait — each maps to one of MPPI's three calls into
the world model:

1. ``policy_action`` — produce a deterministic policy mean at ``z``.
   Used by MPPI for the ``NUM_PI_TRAJS`` policy-seeded warm-start
   trajectories, and for the terminal action in ``terminal_value``.
   Stubs without a learned policy can return zero — pi-trajs degenerate
   to "mean-seeded with noise" trajectories, which is harmless.
2. ``rollout_step`` — advance one step: ``(z, a) → (z', r_scalar)``.
   This is the hot path; the callback is responsible for build_za +
   dynamics + reward + categorical decode (where applicable). Returns
   a scalar reward per batch row.
3. ``terminal_value`` — bootstrap value at the end of the horizon.
   Typically ``Q(z, π(z))`` averaged over a random Q-pair (TDMPC2) or
   the LQR closed-form value (LinearQuadratic). Returns a scalar
   per batch row. May read a random seed to support TDMPC2's
   per-MPPI-iter Q-pair resampling.

Performance contract (carried over from the design doc): a faithful
GPU implementor must not add device syncs or host-side round-trips
inside the trait methods — the planner is allowed to enqueue all
``HORIZON`` rollout steps without any intervening sync.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT as dtype
from noeira.core.tiles import Tile


trait RolloutCallbackCPU(Deinitable):
    """CPU-side per-step rollout contract.

    All operations are scalar (B = 1) — the planner loops over samples
    and timesteps and calls the callback once per (sample, step). This
    is intentionally not batched: CPU MPPI is the testing / fallback
    path, not the production path, and a per-sample interface keeps
    stub-model fixtures trivial.

    ``LATENT_DIM`` / ``ACTION_DIM`` are comptime so the callback and
    planner agree on shapes at compile time. Lists passed in/out must
    have exactly these lengths; implementors should not resize.
    """

    comptime LATENT_DIM: Int
    comptime ACTION_DIM: Int

    def policy_action_cpu(
        mut self,
        z: List[Float64],
        mut action_out: List[Float64],
    ) raises:
        """Write the policy's deterministic action at ``z`` into
        ``action_out``. ``z`` has length ``LATENT_DIM``; ``action_out``
        must have length ``ACTION_DIM`` (caller resizes; callee fills).
        Stubs without a learned policy may write zeros.
        """
        ...

    def rollout_step_cpu(
        mut self,
        z: List[Float64],
        a: List[Float64],
        mut z_next_out: List[Float64],
    ) raises -> Float64:
        """Advance one step: write ``z'`` into ``z_next_out`` and return
        the scalar step reward. ``z``/``z_next_out`` have length
        ``LATENT_DIM``; ``a`` has length ``ACTION_DIM``.
        """
        ...

    def terminal_value_cpu(
        mut self,
        z: List[Float64],
    ) raises -> Float64:
        """Return a scalar bootstrap value at the end-of-horizon state
        ``z``. Stubs without a learned Q may return ``0.0``.
        """
        ...


trait RolloutCallbackGPU(Deinitable):
    """GPU-side per-step rollout contract.

    Operates on row-major ``LayoutTensor`` views with a method-level
    batch comptime parameter, matching the existing TDMPC2 kernel
    signatures. The planner enqueues all ``HORIZON`` rollout steps on
    the same queue without intervening sync; implementors must not add
    sync points inside these methods.

    Outputs are written into caller-supplied tensors (TDMPC2 owns its
    own scratch buffers; the planner owns ``z`` / ``z_next`` /
    ``r_out`` / ``v_out``). Inputs are not mutated.

    The ``seed`` argument on ``terminal_value_gpu`` exists so the
    planner can vary the Q-pair subsample across MPPI iterations
    (matches reference TDMPC2's per-call ``randperm``). Stubs that
    don't sample may ignore it.
    """

    comptime LATENT_DIM: Int
    comptime ACTION_DIM: Int

    def policy_action_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        action_out: Tile[dtype, B, Self.ACTION_DIM],
    ) raises:
        """Write the policy's deterministic action mean for every row
        of ``z`` into ``action_out``. Shapes ``(B, LATENT_DIM)`` and
        ``(B, ACTION_DIM)``.
        """
        ...

    def rollout_step_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        a: Tile[dtype, B, Self.ACTION_DIM],
        z_next_out: Tile[dtype, B, Self.LATENT_DIM],
        r_out: Tile[dtype, B],
    ) raises:
        """Advance one step for the full batch: write ``z'`` into
        ``z_next_out`` and the scalar step reward into ``r_out``.
        Reward decoding (categorical bins → scalar, where applicable)
        lives in the callback. The planner will discount and
        accumulate ``r_out`` into its returns buffer downstream.
        """
        ...

    def terminal_value_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        v_out: Tile[dtype, B],
        seed: UInt32,
    ) raises:
        """Bootstrap value at end-of-horizon state ``z``. TDMPC2's impl
        runs ``π(z)`` then averages two random Q-target heads decoded
        from the categorical distribution. ``seed`` selects the
        Q-pair; the planner varies it per iteration so each
        ``terminal_value_gpu`` call sees a fresh subsample.
        """
        ...
