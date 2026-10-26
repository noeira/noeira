"""MPPI (Model Predictive Path Integral) planner.

Two structs in this file:

* ``MPPICPU`` — host-side reference implementation. Used by the
  TDMPC2 eval ``select_action`` path (single env) and by isolated
  property tests on stub models (LinearQuadratic etc.) where a CPU
  implementation gives a tractable closed-form oracle.

* ``MPPIGPUBatched`` — production GPU implementation that plans for
  ``N_ENVS`` environments in one batched kernel grid per
  per-horizon-step. Owns the per-MPPI scratch device buffers (z,
  z_next, all_actions, returns, mean, std, weights) — separation
  from the agent state was a goal of the Phase 2 refactor.

Both call into a ``RolloutCallback{CPU,GPU}`` for the model contract
(``policy_action``, ``rollout_step``, ``terminal_value``). The
algorithmic logic — Gaussian sampling, softmax weighting, mean/std
refit, multinomial action selection — lives entirely in the planner.

Reference: Hansen et al., 2023 — TD-MPC2.
"""

from std.math import exp, sqrt, cos, log
from std.random import random_float64
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT as dtype

comptime TPB = 256  # preserved from legacy nn.constants (nn.TPB == 128)

from .rollout_callback import RolloutCallbackCPU, RolloutCallbackGPU
from .mppi_kernels import (
    mppi_broadcast_z0_zero_returns_batched_kernel,
    mppi_sample_actions_batched_kernel,
    mppi_accum_reward_scalar_kernel,
    mppi_copy_z_kernel,
    mppi_add_terminal_value_kernel,
    mppi_softmax_weights_kernel,
    mppi_weighted_mean_std_kernel,
    mppi_select_action_kernel,
)
from noeira.core.tiles import Tile, tile


# =============================================================================
# Helper functions (free, not methods — small enough to share between CPU
# and any future scalar GPU CPU-fallback)
# =============================================================================


@always_inline
def _gaussian_sample() -> Float64:
    """Box-Muller transform to draw a standard normal sample.

    Uses the global ``std.random`` RNG — callers seed via ``_set_seed``
    before invoking ``MPPICPU.plan`` for determinism.
    """
    var u1 = random_float64()
    var u2 = random_float64()
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2)


@always_inline
def _clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@always_inline
def _weighted_sample(weights: List[Float64], n: Int) -> Int:
    """Inverse-CDF multinomial sampling — one index proportional to
    ``weights``. Matches reference TD-MPC2's
    ``torch.multinomial(score, 1)`` call.
    """
    var u = random_float64()
    var cumsum: Float64 = 0.0
    for i in range(n):
        cumsum += weights[i]
        if u <= cumsum:
            return i
    return n - 1


# =============================================================================
# MPPICPU
# =============================================================================


struct MPPICPU[
    LATENT_DIM: Int,
    ACTION_DIM: Int,
    HORIZON: Int,
    NUM_SAMPLES: Int,
    NUM_PI_TRAJS: Int,
    NUM_ITERATIONS: Int,
    NUM_ELITES: Int,
](Deinitable, Movable):
    """CPU MPPI planner — reference implementation + test path.

    All hyperparameters that affect storage layout
    (``HORIZON``/``NUM_SAMPLES``/``NUM_PI_TRAJS``/``NUM_ITERATIONS``)
    are comptime to match the existing TDMPC2 ``plan()`` signature.
    The scratch lists are sized at ``__init__`` and reused across
    ``plan()`` calls.

    Warm-start state lives on the planner: ``prev_mean`` accumulates
    the previous timestep's optimized mean; ``t0`` flags whether the
    next ``plan()`` call should treat the episode as just-started
    (warm-start disabled, mean reset to zero). Callers reset the
    episode boundary via ``start_episode()``.
    """

    comptime TOTAL_SAMPLES = Self.NUM_SAMPLES + Self.NUM_PI_TRAJS
    comptime STD_MIN: Float64 = 0.05
    comptime STD_MAX: Float64 = 2.0

    # ── Distribution + warm-start state ────────────────────────────
    var mean: List[Float64]
    """`(HORIZON, ACTION_DIM)` — current MPPI mean, refit each iter."""
    var std: List[Float64]
    """`(HORIZON, ACTION_DIM)` — current MPPI std, clamped to
    ``[STD_MIN, STD_MAX]``."""
    var prev_mean: List[Float64]
    """`(HORIZON, ACTION_DIM)` — final mean from previous ``plan()``
    call; consumed for next-call warm-start."""

    # ── Per-call scratch ───────────────────────────────────────────
    var actions: List[Float64]
    """`(TOTAL_SAMPLES, HORIZON, ACTION_DIM)` — candidate action sequences."""
    var returns: List[Float64]
    """`(TOTAL_SAMPLES,)` — per-sample undiscounted-summed reward
    plus terminal bootstrap."""
    var weights: List[Float64]
    """`(TOTAL_SAMPLES,)` — softmax weights over returns."""

    # ── Episode-edge flag ──────────────────────────────────────────
    var t0: Bool

    def __init__(out self) raises:
        if Self.HORIZON < 1:
            raise Error("MPPICPU: HORIZON must be >= 1")
        if Self.NUM_SAMPLES < 1:
            raise Error("MPPICPU: NUM_SAMPLES must be >= 1")
        if Self.NUM_PI_TRAJS < 0:
            raise Error("MPPICPU: NUM_PI_TRAJS must be >= 0")
        if Self.NUM_ITERATIONS < 1:
            raise Error("MPPICPU: NUM_ITERATIONS must be >= 1")

        var mean_size = Self.HORIZON * Self.ACTION_DIM
        var actions_size = Self.TOTAL_SAMPLES * mean_size

        self.mean = List[Float64](length=mean_size, fill=0.0)
        self.std = List[Float64](length=mean_size, fill=0.5)
        self.prev_mean = List[Float64](length=mean_size, fill=0.0)
        self.actions = List[Float64](length=actions_size, fill=0.0)
        self.returns = List[Float64](length=Self.TOTAL_SAMPLES, fill=0.0)
        self.weights = List[Float64](length=Self.TOTAL_SAMPLES, fill=0.0)
        self.t0 = True

    def start_episode(mut self):
        """Mark the start of a new episode — disables warm-start on
        the next ``plan()`` call. Caller invokes once per
        environment reset.
        """
        self.t0 = True

    def plan[
        CB: RolloutCallbackCPU
    ](
        mut self,
        mut callback: CB,
        z0: List[Float64],
        gamma: Float64,
        temperature: Float64,
        action_scale: Float64 = 1.0,
        deterministic: Bool = False,
    ) raises -> List[Float64]:
        """Run one MPPI optimization and return the selected action.

        Args:
            callback: ``RolloutCallbackCPU`` providing policy / step /
                terminal-value access to the world model.
            z0: Initial latent state, length ``LATENT_DIM``.
            gamma: Per-step discount factor.
            temperature: MPPI softmax temperature
                (``w ∝ exp(temperature * (G - max_G))``).
            action_scale: Multiplier applied to the selected action
                before clipping to ``[-action_scale, action_scale]``.
            deterministic: If ``True``, skip the per-action Gaussian
                exploration noise on the returned action — used in
                eval.

        Returns:
            ``List[Float64]`` of length ``ACTION_DIM``.
        """
        if len(z0) != Self.LATENT_DIM:
            raise Error("MPPICPU.plan: z0 length must equal LATENT_DIM")

        # ── 1. Warm-start mean (shift prev_mean by 1 step) ─────────
        var mean_size = Self.HORIZON * Self.ACTION_DIM
        if not self.t0:
            for t in range(Self.HORIZON - 1):
                for a in range(Self.ACTION_DIM):
                    self.mean[t * Self.ACTION_DIM + a] = self.prev_mean[
                        (t + 1) * Self.ACTION_DIM + a
                    ]
            # Last step has no information from the previous plan.
            for a in range(Self.ACTION_DIM):
                self.mean[(Self.HORIZON - 1) * Self.ACTION_DIM + a] = 0.0
        else:
            for i in range(mean_size):
                self.mean[i] = 0.0
        for i in range(mean_size):
            self.std[i] = 0.5

        # ── 2. Main MPPI loop ──────────────────────────────────────
        for _iter in range(Self.NUM_ITERATIONS):
            # 2a. NUM_PI_TRAJS policy-seeded trajectories
            for s in range(Self.NUM_PI_TRAJS):
                var z_curr = List[Float64](length=Self.LATENT_DIM, fill=0.0)
                for i in range(Self.LATENT_DIM):
                    z_curr[i] = z0[i]
                var pi_mean = List[Float64](length=Self.ACTION_DIM, fill=0.0)
                var a_step = List[Float64](length=Self.ACTION_DIM, fill=0.0)
                var z_next = List[Float64](length=Self.LATENT_DIM, fill=0.0)
                for t in range(Self.HORIZON):
                    callback.policy_action_cpu(z_curr, pi_mean)
                    var sample_base = (
                        s * Self.HORIZON * Self.ACTION_DIM + t * Self.ACTION_DIM
                    )
                    for a in range(Self.ACTION_DIM):
                        var noise = _gaussian_sample() * 0.1
                        var act = pi_mean[a] + noise
                        act = _clamp(act, -1.0, 1.0)
                        a_step[a] = act
                        self.actions[sample_base + a] = act
                    _ = callback.rollout_step_cpu(z_curr, a_step, z_next)
                    for i in range(Self.LATENT_DIM):
                        z_curr[i] = z_next[i]

            # 2b. NUM_SAMPLES Gaussian-sampled trajectories
            for s in range(Self.NUM_PI_TRAJS, Self.TOTAL_SAMPLES):
                for t in range(Self.HORIZON):
                    for a in range(Self.ACTION_DIM):
                        var mu = self.mean[t * Self.ACTION_DIM + a]
                        var sigma = self.std[t * Self.ACTION_DIM + a]
                        var noise = _gaussian_sample()
                        var act = mu + sigma * noise
                        act = _clamp(act, -1.0, 1.0)
                        var idx = (
                            s * Self.HORIZON * Self.ACTION_DIM
                            + t * Self.ACTION_DIM
                            + a
                        )
                        self.actions[idx] = act

            # 2c. Score every candidate
            self._score_all_samples(callback, z0, gamma)

            # 2d. Softmax weights
            self._softmax_weights(temperature)

            # 2e. Refit mean / std
            self._refit_distribution()

        # ── 3. Persist final mean for next call's warm-start ───────
        for i in range(mean_size):
            self.prev_mean[i] = self.mean[i]
        self.t0 = False

        # ── 4. Action selection: multinomial + per-step exploration ─
        var selected = _weighted_sample(self.weights, Self.TOTAL_SAMPLES)
        var result = List[Float64](length=Self.ACTION_DIM, fill=0.0)
        for a in range(Self.ACTION_DIM):
            var base = selected * Self.HORIZON * Self.ACTION_DIM
            var act = self.actions[base + a]
            if not deterministic:
                act += _gaussian_sample() * self.std[a]
            act = _clamp(act * action_scale, -action_scale, action_scale)
            result[a] = act
        return result^

    def _score_all_samples[
        CB: RolloutCallbackCPU
    ](mut self, mut callback: CB, z0: List[Float64], gamma: Float64,) raises:
        """Roll out all ``TOTAL_SAMPLES`` candidates and write their
        discounted returns into ``self.returns``.
        """
        var z_curr = List[Float64](length=Self.LATENT_DIM, fill=0.0)
        var z_next = List[Float64](length=Self.LATENT_DIM, fill=0.0)
        var a_step = List[Float64](length=Self.ACTION_DIM, fill=0.0)
        for s in range(Self.TOTAL_SAMPLES):
            for i in range(Self.LATENT_DIM):
                z_curr[i] = z0[i]
            var G: Float64 = 0.0
            var discount: Float64 = 1.0
            for t in range(Self.HORIZON):
                var base = (
                    s * Self.HORIZON * Self.ACTION_DIM + t * Self.ACTION_DIM
                )
                for a in range(Self.ACTION_DIM):
                    a_step[a] = self.actions[base + a]
                var r = callback.rollout_step_cpu(z_curr, a_step, z_next)
                G += discount * r
                discount *= gamma
                for i in range(Self.LATENT_DIM):
                    z_curr[i] = z_next[i]
            G += discount * callback.terminal_value_cpu(z_curr)
            self.returns[s] = G

    def _softmax_weights(mut self, temperature: Float64):
        """Set ``self.weights`` to the **top-K elite** softmax of
        ``self.returns`` with stability shift by max return.

        Only the top ``NUM_ELITES`` samples (by return) get non-zero
        weight; the rest are zeroed. Matches the reference TD-MPC2
        recipe (tdmpc2.py:186) and the GPU
        ``mppi_softmax_weights_kernel``. The legacy CPU ``plan()`` did
        NOT do this filtering — softmaxing over all samples lets the
        bottom-fraction trajectories contribute small-but-nonzero
        weight, biasing the mean refit toward averaging-in bad
        candidates. (See HalfCheetah plateau diagnosed in the GPU
        kernel docstring.)

        Index-tiebreak: when returns are tied, samples with smaller
        index are elite (matches GPU kernel rank computation).
        """
        # Find max return for numerical stability
        var max_return = self.returns[0]
        for s in range(1, Self.TOTAL_SAMPLES):
            if self.returns[s] > max_return:
                max_return = self.returns[s]

        # Per-sample rank = #{k : returns[k] > returns[s] OR
        # (returns[k] == returns[s] AND k < s)}.
        # Sample is elite iff rank < NUM_ELITES.
        var sum_w: Float64 = 0.0
        for s in range(Self.TOTAL_SAMPLES):
            var rank: Int = 0
            for k in range(Self.TOTAL_SAMPLES):
                if k == s:
                    continue
                if self.returns[k] > self.returns[s]:
                    rank += 1
                elif self.returns[k] == self.returns[s] and k < s:
                    rank += 1
            if rank < Self.NUM_ELITES:
                var w = exp(temperature * (self.returns[s] - max_return))
                self.weights[s] = w
                sum_w += w
            else:
                self.weights[s] = 0.0

        if sum_w < 1e-10:
            sum_w = 1e-10
        for s in range(Self.TOTAL_SAMPLES):
            self.weights[s] = self.weights[s] / sum_w

    def _refit_distribution(mut self):
        """Refit ``self.mean`` and ``self.std`` as the weighted mean
        and weighted std of ``self.actions`` under ``self.weights``.
        Std is clamped to ``[STD_MIN, STD_MAX]`` per reference.
        """
        for t in range(Self.HORIZON):
            for a in range(Self.ACTION_DIM):
                var new_mean: Float64 = 0.0
                for s in range(Self.TOTAL_SAMPLES):
                    var base = (
                        s * Self.HORIZON * Self.ACTION_DIM
                        + t * Self.ACTION_DIM
                        + a
                    )
                    new_mean += self.weights[s] * self.actions[base]
                self.mean[t * Self.ACTION_DIM + a] = new_mean
                var new_var: Float64 = 0.0
                for s in range(Self.TOTAL_SAMPLES):
                    var base = (
                        s * Self.HORIZON * Self.ACTION_DIM
                        + t * Self.ACTION_DIM
                        + a
                    )
                    var diff = self.actions[base] - new_mean
                    new_var += self.weights[s] * diff * diff
                var new_std = sqrt(new_var + 1e-8)
                new_std = _clamp(new_std, Self.STD_MIN, Self.STD_MAX)
                self.std[t * Self.ACTION_DIM + a] = new_std


# =============================================================================
# MPPIGPUBatched
# =============================================================================


struct MPPIGPUBatched[
    LATENT_DIM: Int,
    ACTION_DIM: Int,
    HORIZON: Int,
    NUM_SAMPLES: Int,
    NUM_PI_TRAJS: Int,
    NUM_ELITES: Int,
    NUM_ITERATIONS: Int,
    N_ENVS: Int,
](Deinitable, Movable):
    """GPU-batched MPPI planner — plans for all ``N_ENVS`` envs in one
    kernel grid per horizon step.

    All planner-side scratch lives on the struct; the agent only
    holds the callback (which owns its own scratch — network
    workspaces, za, rew/q logits, etc.). Per-env warm-start state
    lives on host (``env_prev_means`` / ``env_t0_flags``) and is
    uploaded into the GPU ``mean_buf`` at the start of every
    ``plan_gpu`` call.

    Kernel sequence per MPPI iteration (one ``ctx.enqueue_function``
    per kernel; no sync between them — the GPU queue serializes):

    1. broadcast_z0_zero_returns
    2. for t in HORIZON:
       a. callback.policy_action_gpu   (writes pol_action_buf)
       b. mppi_sample_actions_batched  (reads pol_action_buf, writes
          act_step_buf + all_actions_buf)
       c. callback.rollout_step_gpu    (reads z + act_step_buf,
          writes z_next + reward_step_buf)
       d. mppi_accum_reward_scalar     (returns += discount * reward)
       e. mppi_copy_z                  (z_buf <- z_next_buf)
    3. callback.terminal_value_gpu     (writes terminal_v_buf)
    4. mppi_add_terminal_value         (returns += discount^H * v)
    5. mppi_softmax_weights            (weights = softmax(temp*returns))
    6. mppi_weighted_mean_std          (refits mean_buf, std_buf)

    After NUM_ITERATIONS iters: mppi_select_action_kernel writes
    selected actions to out_act_dev (one device-side multinomial
    sample per env, plus per-action exploration noise unless
    deterministic).

    Only one host↔device copy per ``plan_gpu`` call: download
    ``mean_buf`` after action selection for next-call warm-start
    state.
    """

    comptime TOTAL_SAMPLES = Self.NUM_SAMPLES + Self.NUM_PI_TRAJS
    comptime BATCH_TOTAL = Self.N_ENVS * Self.TOTAL_SAMPLES
    comptime MPPI_BLOCKS = (Self.BATCH_TOTAL + TPB - 1) // TPB
    comptime MEAN_STD_TOTAL = Self.N_ENVS * Self.HORIZON * Self.ACTION_DIM
    comptime MEAN_STD_BLOCKS = (Self.MEAN_STD_TOTAL + TPB - 1) // TPB

    # ── Planner-owned device scratch ──────────────────────────────
    var z_buf: DeviceBuffer[dtype]
    """(BATCH_TOTAL, LATENT_DIM) — current latent state."""
    var z_next_buf: DeviceBuffer[dtype]
    """(BATCH_TOTAL, LATENT_DIM) — next latent state, copied into
    ``z_buf`` after each step."""
    var act_step_buf: DeviceBuffer[dtype]
    """(BATCH_TOTAL, ACTION_DIM) — current step's sampled action.
    Read by callback's ``rollout_step_gpu``."""
    var all_actions_buf: DeviceBuffer[dtype]
    """(BATCH_TOTAL * HORIZON * ACTION_DIM,) — per-sample full action
    sequence; consumed by refit + action selection."""
    var pol_action_buf: DeviceBuffer[dtype]
    """(BATCH_TOTAL, ACTION_DIM) — policy mean action returned by
    callback.policy_action_gpu (used by sample kernel for pi-trajs)."""
    var reward_step_buf: DeviceBuffer[dtype]
    """(BATCH_TOTAL,) — per-batch-row scalar reward from
    callback.rollout_step_gpu, decoded by the callback."""
    var terminal_v_buf: DeviceBuffer[dtype]
    """(BATCH_TOTAL,) — terminal bootstrap value from
    callback.terminal_value_gpu."""
    var returns_buf: DeviceBuffer[dtype]
    """(BATCH_TOTAL,) — running discounted return per candidate."""
    var mean_buf: DeviceBuffer[dtype]
    """(N_ENVS * HORIZON * ACTION_DIM,) — MPPI per-env distribution
    mean."""
    var std_buf: DeviceBuffer[dtype]
    """(N_ENVS * HORIZON * ACTION_DIM,) — MPPI per-env distribution
    std."""
    var weights_buf: DeviceBuffer[dtype]
    """(BATCH_TOTAL,) — softmax weights over returns."""

    var mean_host: HostBuffer[dtype]
    """Host mirror of mean_buf, downloaded post-iter for warm-start."""
    var std_host: HostBuffer[dtype]
    """Host mirror of std_buf, used for uploading initial std."""

    # ── Per-env warm-start state (host-side) ──────────────────────
    var env_prev_means: List[List[Float64]]
    """`(N_ENVS, HORIZON*ACTION_DIM)` — last call's optimized mean."""
    var env_t0_flags: List[Bool]
    """`(N_ENVS,)` — episode-edge flags; reset via ``start_episode``."""

    def __init__(out self, ctx: DeviceContext) raises:
        if Self.HORIZON < 1:
            raise Error("MPPIGPUBatched: HORIZON must be >= 1")
        if Self.NUM_SAMPLES < 1:
            raise Error("MPPIGPUBatched: NUM_SAMPLES must be >= 1")
        if Self.NUM_PI_TRAJS < 0:
            raise Error("MPPIGPUBatched: NUM_PI_TRAJS must be >= 0")
        if Self.NUM_ITERATIONS < 1:
            raise Error("MPPIGPUBatched: NUM_ITERATIONS must be >= 1")
        if Self.N_ENVS < 1:
            raise Error("MPPIGPUBatched: N_ENVS must be >= 1")

        var bt_latent = Self.BATCH_TOTAL * Self.LATENT_DIM
        var bt_act = Self.BATCH_TOTAL * Self.ACTION_DIM
        var all_actions_size = Self.BATCH_TOTAL * Self.HORIZON * Self.ACTION_DIM

        self.z_buf = ctx.enqueue_create_buffer[dtype](bt_latent)
        self.z_next_buf = ctx.enqueue_create_buffer[dtype](bt_latent)
        self.act_step_buf = ctx.enqueue_create_buffer[dtype](bt_act)
        self.all_actions_buf = ctx.enqueue_create_buffer[dtype](
            all_actions_size
        )
        self.pol_action_buf = ctx.enqueue_create_buffer[dtype](bt_act)
        self.reward_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH_TOTAL
        )
        self.terminal_v_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH_TOTAL)
        self.returns_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH_TOTAL)
        self.mean_buf = ctx.enqueue_create_buffer[dtype](Self.MEAN_STD_TOTAL)
        self.std_buf = ctx.enqueue_create_buffer[dtype](Self.MEAN_STD_TOTAL)
        self.weights_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH_TOTAL)

        self.mean_host = ctx.enqueue_create_host_buffer[dtype](
            Self.MEAN_STD_TOTAL
        )
        self.std_host = ctx.enqueue_create_host_buffer[dtype](
            Self.MEAN_STD_TOTAL
        )

        # Per-env warm-start state
        var ms = Self.HORIZON * Self.ACTION_DIM
        self.env_prev_means = List[List[Float64]](capacity=Self.N_ENVS)
        self.env_t0_flags = List[Bool](capacity=Self.N_ENVS)
        for _ in range(Self.N_ENVS):
            self.env_prev_means.append(List[Float64](length=ms, fill=0.0))
            self.env_t0_flags.append(True)

    def start_episode(mut self, env_idx: Int):
        """Reset warm-start state for ``env_idx`` — call after that
        env's environment resets."""
        if env_idx < 0 or env_idx >= Self.N_ENVS:
            return
        self.env_t0_flags[env_idx] = True

    def plan_gpu[
        CB: RolloutCallbackGPU
    ](
        mut self,
        ctx: DeviceContext,
        mut callback: CB,
        z0_tensor: Tile[dtype, Self.N_ENVS, Self.LATENT_DIM],
        out_act_dev: Tile[dtype, Self.N_ENVS * Self.ACTION_DIM],
        gamma: Float64,
        temperature: Float64,
        action_scale: Float64 = 1.0,
        deterministic: Bool = False,
        rng_base_seed: UInt32 = 42,
        num_iters: Int = 0,
    ) raises:
        """Plan one timestep for all N_ENVS envs.

        Writes selected actions into ``out_act_dev`` on-device — no
        host roundtrip in the production path (only the mean buffer
        comes back for the next call's warm-start state).
        """
        # ── 1. Initialize per-env mean/std on host, upload ────────
        var ms = Self.HORIZON * Self.ACTION_DIM
        for env_idx in range(Self.N_ENVS):
            var base = env_idx * ms
            for i in range(ms):
                self.mean_host[base + i] = Scalar[dtype](0.0)
                self.std_host[base + i] = Scalar[dtype](0.5)
            if (
                not self.env_t0_flags[env_idx]
                and len(self.env_prev_means[env_idx]) == ms
            ):
                # Shift previous mean by one step.
                for t in range(Self.HORIZON - 1):
                    for a in range(Self.ACTION_DIM):
                        self.mean_host[base + t * Self.ACTION_DIM + a] = Scalar[
                            dtype
                        ](
                            self.env_prev_means[env_idx][
                                (t + 1) * Self.ACTION_DIM + a
                            ]
                        )
        ctx.enqueue_copy(self.mean_buf, self.mean_host)
        ctx.enqueue_copy(self.std_buf, self.std_host)

        # ── 2. LayoutTensor views over our buffers ────────────────
        var z_tensor = tile[Self.BATCH_TOTAL, Self.LATENT_DIM](self.z_buf)
        var z_next_tensor = tile[Self.BATCH_TOTAL, Self.LATENT_DIM](self.z_next_buf)
        var act_step_tensor = tile[Self.BATCH_TOTAL, Self.ACTION_DIM](self.act_step_buf)
        var all_actions_tensor = tile[Self.BATCH_TOTAL * Self.HORIZON * Self.ACTION_DIM](self.all_actions_buf)
        var pol_action_tensor = tile[Self.BATCH_TOTAL, Self.ACTION_DIM](self.pol_action_buf)
        var reward_step_tensor = tile[Self.BATCH_TOTAL](self.reward_step_buf)
        var terminal_v_tensor = tile[Self.BATCH_TOTAL](self.terminal_v_buf)
        var returns_tensor = tile[Self.BATCH_TOTAL](self.returns_buf)
        var mean_tensor = tile[Self.MEAN_STD_TOTAL](self.mean_buf)
        var std_tensor = tile[Self.MEAN_STD_TOTAL](self.std_buf)
        var weights_tensor = tile[Self.BATCH_TOTAL](self.weights_buf)

        # ── 3. Main MPPI iterations ──────────────────────────────
        # `num_iters` overrides `NUM_ITERATIONS` at RUNTIME (0 = use the
        # comptime default). Iteration count is the one budget knob that can be
        # runtime: it is purely a loop bound, whereas NUM_SAMPLES /
        # NUM_PI_TRAJS size every device buffer in `__init__` and cannot change
        # without reallocating. Cost is very close to linear in it, so this is
        # the lever for trading plan quality against frame rate.
        var temp_scalar = Scalar[dtype](temperature)
        var n_iters = num_iters if num_iters > 0 else Self.NUM_ITERATIONS
        for mppi_iter in range(n_iters):
            var rng_seed = rng_base_seed + UInt32(
                mppi_iter
                * Self.BATCH_TOTAL
                * Self.HORIZON
                * Self.ACTION_DIM
                * 2
            )
            _run_mppi_iteration[
                Self.LATENT_DIM,
                Self.ACTION_DIM,
                Self.HORIZON,
                Self.NUM_PI_TRAJS,
                Self.BATCH_TOTAL,
                Self.TOTAL_SAMPLES,
                Self.NUM_ELITES,
                Self.N_ENVS,
                Self.MPPI_BLOCKS,
                Self.MEAN_STD_BLOCKS,
                CB,
            ](
                ctx,
                callback,
                z0_tensor,
                z_tensor,
                z_next_tensor,
                act_step_tensor,
                all_actions_tensor,
                pol_action_tensor,
                reward_step_tensor,
                terminal_v_tensor,
                returns_tensor,
                mean_tensor,
                std_tensor,
                weights_tensor,
                Scalar[dtype](gamma),
                temp_scalar,
                rng_seed,
            )

        # ── 4. Action selection on GPU ───────────────────────────
        comptime select_action = mppi_select_action_kernel[
            dtype,
            Self.N_ENVS,
            Self.TOTAL_SAMPLES,
            Self.ACTION_DIM,
            Self.HORIZON,
            TPB,
        ]
        var act_select_seed = rng_base_seed + UInt32(0x5E1EC7ED)
        var det_flag: UInt32 = 1 if deterministic else 0
        ctx.enqueue_function[select_action](
            weights_tensor,
            all_actions_tensor,
            std_tensor,
            out_act_dev,
            Scalar[DType.uint32](act_select_seed),
            Scalar[dtype](action_scale),
            det_flag,
            grid_dim=(Self.N_ENVS,),
            block_dim=(TPB,),
        )

        # ── 5. Download mean for next-call warm-start ────────────
        ctx.enqueue_copy(self.mean_host, self.mean_buf)
        ctx.synchronize()
        for env_idx in range(Self.N_ENVS):
            var base = env_idx * ms
            for i in range(ms):
                self.env_prev_means[env_idx][i] = Float64(
                    self.mean_host[base + i]
                )
            self.env_t0_flags[env_idx] = False


# =============================================================================
# Module-level helper: one MPPI iteration on GPU
# =============================================================================


def _run_mppi_iteration[
    LATENT_DIM: Int,
    ACTION_DIM: Int,
    HORIZON: Int,
    NUM_PI_TRAJS: Int,
    BATCH_TOTAL: Int,
    TOTAL_SAMPLES: Int,
    NUM_ELITES: Int,
    N_ENVS: Int,
    MPPI_BLOCKS: Int,
    MEAN_STD_BLOCKS: Int,
    CB: RolloutCallbackGPU,
](
    ctx: DeviceContext,
    mut callback: CB,
    z0_tensor: Tile[dtype, N_ENVS, LATENT_DIM],
    z_tensor: Tile[dtype, BATCH_TOTAL, LATENT_DIM],
    z_next_tensor: Tile[dtype, BATCH_TOTAL, LATENT_DIM],
    act_step_tensor: Tile[dtype, BATCH_TOTAL, ACTION_DIM],
    all_actions_tensor: Tile[dtype, BATCH_TOTAL * HORIZON * ACTION_DIM],
    pol_action_tensor: Tile[dtype, BATCH_TOTAL, ACTION_DIM],
    reward_step_tensor: Tile[dtype, BATCH_TOTAL],
    terminal_v_tensor: Tile[dtype, BATCH_TOTAL],
    returns_tensor: Tile[dtype, BATCH_TOTAL],
    mean_tensor: Tile[dtype, N_ENVS * HORIZON * ACTION_DIM],
    std_tensor: Tile[dtype, N_ENVS * HORIZON * ACTION_DIM],
    weights_tensor: Tile[dtype, BATCH_TOTAL],
    gamma_scalar: Scalar[dtype],
    temp_scalar: Scalar[dtype],
    rng_seed: UInt32,
) raises:
    """One MPPI iteration. Extracted as a module-level helper per
    ``feedback_lewm_eval_block_compile_explosion.md`` — keeping this
    body separate from ``plan_gpu`` keeps the per-iter call count
    inside the inliner's safe range, and lets future planners reuse
    the same iteration body without copy-paste.
    """
    # ⚠ `mppi_broadcast_z0_zero_returns_batched_kernel` and `mppi_copy_z_kernel`
    # are indexed ONE THREAD PER ELEMENT (see their docstrings), so they need a
    # grid over BATCH_TOTAL * LATENT_DIM, NOT the row-count `MPPI_BLOCKS` the
    # rest of the kernels here use. Launching them with `MPPI_BLOCKS` would
    # silently copy only the first `TPB * MPPI_BLOCKS` elements — a partially
    # rolled latent, which reads as a planner that plans badly rather than as a
    # crash.
    comptime Z_ELEMS = BATCH_TOTAL * LATENT_DIM
    comptime Z_BLOCKS = (Z_ELEMS + TPB - 1) // TPB

    # 1. Broadcast z0 + zero returns
    comptime broadcast_z0_zero = (
        mppi_broadcast_z0_zero_returns_batched_kernel[
            dtype, BATCH_TOTAL, N_ENVS, TOTAL_SAMPLES, LATENT_DIM
        ]
    )
    ctx.enqueue_function[broadcast_z0_zero](
        z0_tensor,
        z_tensor,
        returns_tensor,
        grid_dim=(Z_BLOCKS,),
        block_dim=(TPB,),
    )

    # 2. Horizon rollout
    comptime sample_actions = mppi_sample_actions_batched_kernel[
        dtype,
        BATCH_TOTAL,
        N_ENVS,
        TOTAL_SAMPLES,
        NUM_PI_TRAJS,
        ACTION_DIM,
        HORIZON,
        ACTION_DIM,  # POL_OUT = ACTION_DIM (only mean is passed)
    ]
    comptime accum_reward = mppi_accum_reward_scalar_kernel[dtype, BATCH_TOTAL]
    comptime copy_z = mppi_copy_z_kernel[dtype, BATCH_TOTAL, LATENT_DIM]

    # ── Rebound views for the callback's trait methods ─────────
    # The compiler can't prove LATENT_DIM == CB.LATENT_DIM /
    # ACTION_DIM == CB.ACTION_DIM at parser time (the helper's
    # comptime params and the trait's comptime params are
    # syntactically distinct). Same in-memory representation, so we
    # rebind once and reuse the CB-typed views across the loop.
    comptime CBZTy = Tile[dtype, BATCH_TOTAL, CB.LATENT_DIM]
    comptime CBATy = Tile[dtype, BATCH_TOTAL, CB.ACTION_DIM]
    var z_cb = rebind[CBZTy](z_tensor)
    var z_next_cb = rebind[CBZTy](z_next_tensor)
    var act_step_cb = rebind[CBATy](act_step_tensor)
    var pol_action_cb = rebind[CBATy](pol_action_tensor)

    var discount = Scalar[dtype](1.0)
    for t in range(HORIZON):
        var step_seed = rng_seed + UInt32(t * BATCH_TOTAL * ACTION_DIM + 1)

        # 2a. Policy action (callback)
        callback.policy_action_gpu[BATCH_TOTAL](ctx, z_cb, pol_action_cb)

        # 2b. Sample actions (planner)
        ctx.enqueue_function[sample_actions](
            pol_action_tensor,
            mean_tensor,
            std_tensor,
            act_step_tensor,
            all_actions_tensor,
            # Int32, not `t`: `Int` is not `DevicePassable` (rc2). The kernel
            # param is Int32 and the CALL SITE has to say so.
            Int32(t),
            Scalar[DType.uint32](step_seed),
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # 2c. Rollout step (callback)
        callback.rollout_step_gpu[BATCH_TOTAL](
            ctx,
            z_cb,
            act_step_cb,
            z_next_cb,
            reward_step_tensor,
        )

        # 2d. Accumulate discounted reward (planner)
        ctx.enqueue_function[accum_reward](
            reward_step_tensor,
            returns_tensor,
            discount,
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # 2e. Copy z_next → z (planner). Element-indexed → `Z_BLOCKS`.
        ctx.enqueue_function[copy_z](
            z_tensor,
            z_next_tensor,
            grid_dim=(Z_BLOCKS,),
            block_dim=(TPB,),
        )

        discount = discount * gamma_scalar

    # 3. Terminal value bootstrap (callback)
    # Pass `rng_seed` straight through — the callback owns its own
    # PRNG stream offset (TDMPC2RolloutCallback adds 0xA1B2C3D4 to
    # match the legacy `plan_gpu_batched` Q-pair sampling exactly).
    # Adding the magic on both sides would double-XOR and break
    # bit-parity vs the legacy path.
    callback.terminal_value_gpu[BATCH_TOTAL](
        ctx, z_cb, terminal_v_tensor, rng_seed
    )

    # 4. Add terminal value
    comptime add_terminal = mppi_add_terminal_value_kernel[dtype, BATCH_TOTAL]
    ctx.enqueue_function[add_terminal](
        terminal_v_tensor,
        returns_tensor,
        discount,
        grid_dim=(MPPI_BLOCKS,),
        block_dim=(TPB,),
    )

    # 5. Softmax weights — top-K elite filter via NUM_ELITES.
    # Matches the legacy ``plan_gpu_batched`` recipe; using
    # ``TOTAL_SAMPLES`` here (which would disable the filter) was a
    # parity-breaker for the HalfCheetah training curve — see kernel
    # docstring for why elite filtering matters.
    comptime softmax_weights = mppi_softmax_weights_kernel[
        dtype, N_ENVS, TOTAL_SAMPLES, NUM_ELITES, TPB
    ]
    ctx.enqueue_function[softmax_weights](
        returns_tensor,
        weights_tensor,
        temp_scalar,
        grid_dim=(N_ENVS,),
        block_dim=(TPB,),
    )

    # 6. Weighted mean/std refit
    # ⚠ ONE BLOCK PER OUTPUT, not per thread. `MEAN_STD_BLOCKS` sized the grid
    # from the OUTPUT count (N_ENVS*H*ACT = 144 at the walker's dims), which is
    # smaller than one block — so the old launch was a single block reducing
    # 268 samples serially, 1.4% of GPU time for 144 numbers. The parallelism
    # has to come from the reduction, so the grid is the output count and each
    # block reduces with `TPB` threads. `MEAN_STD_BLOCKS` is now unused here.
    comptime MEAN_STD_DIMS = N_ENVS * HORIZON * ACTION_DIM
    comptime weighted_mean_std = mppi_weighted_mean_std_kernel[
        dtype, N_ENVS, TOTAL_SAMPLES, HORIZON, ACTION_DIM, TPB
    ]
    ctx.enqueue_function[weighted_mean_std](
        weights_tensor,
        all_actions_tensor,
        mean_tensor,
        std_tensor,
        grid_dim=(MEAN_STD_DIMS,),
        block_dim=(TPB,),
    )
