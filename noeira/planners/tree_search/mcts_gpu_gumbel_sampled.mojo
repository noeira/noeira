"""GPU sampled-Gumbel MCTS — continuous-action variant (Phase 3 EZv2 move).

Moved verbatim from `noeira/deep_agents/efficient_zero_v2/gpu_mcts_sampled.mojo`
to the planners package alongside its discrete sibling. The original module
now re-exports from here; see `planners/tree_search/mcts_gpu_gumbel.mojo` for
the discrete counterpart.

Per-env tree layout mirrors the discrete GPU MCTS but keys per-action data
by candidate index rather than discrete action index:

    Discrete: per-node arrays of length ACT (one slot per action).
    Continuous: per-node arrays of length K_PAD (one slot per sampled
    candidate), plus an `actions[N_ENVS, MAX_NODES, K_PAD, ACT_DIM]` buffer
    holding the actual continuous action vector each slot represents.

Each node stores up to `K_ROOT` candidates (root) or `K_NON_ROOT`
(non-root) — physical slots are sized at K_PAD = K_ROOT, with `active_k`
captured implicitly by leaving slots [active_k, K_PAD) at -1 / 0.

Algorithm: paper App. A.
  • Root: sample K_ROOT candidates a_i; half from N(μ, σ), half from
    N(μ, STD_MAG · σ). Score by g_i + log π(a_i|s) + σ(completed_Q(a_i)).
  • Sequential Halving over K_ROOT candidates with `log2(K_ROOT)` phases.
  • Non-root expansion samples K_NON_ROOT candidates from N(μ, σ) of the
    child policy.
  • Visit-balance selection at non-root over K_NON_ROOT candidates.

Squashed-Gaussian parameterization matches the loss kernel
(`ezv2_policy_loss_grad_continuous_kernel`):
    μ = MAX_ACTION · tanh(μ_raw / MAX_ACTION)
    σ = softplus(σ_raw) + MIN_STD
    u ~ N(μ, σ)
    a = MAX_ACTION · tanh(u)
    log π(a) = Σ_d [-0.5·η_d² − log σ_d − 0.5·log(2π) − log(1 − c_d²)]
        where c_d = a_d / MAX_ACTION, u*_d = atanh(c_d), η_d = (u*_d − μ_d)/σ_d.

Output: `chosen_actions[N_ENVS, ACT_DIM]` — the chosen action vector
per env (argmax-visit candidate at root if `deterministic`, else
visit-weighted soft pick using the same Philox stream). Also written to
diagnostics: `root_visits[N_ENVS, K_ROOT]`.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from std.math import sqrt, log, exp, tanh, cos, pi
from noeira.nn.constants import DT as dtype
from noeira.core.tiles import Tile
comptime TPB = 256  # preserved from legacy nn.constants (nn.TPB == 128)


comptime MAX_DEPTH: Int = 32
comptime LOG_2PI_F: Float64 = 1.8378770664093453


# ═════════════════════════════════════════════════════════════════════════
# State container
# ═════════════════════════════════════════════════════════════════════════


struct EZV2GPUSampledMCTSState[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
    K_ROOT: Int,
    K_NON_ROOT: Int,
](Movable):
    """GPU-resident per-env sampled-Gumbel trees + scratch buffers.

    Parameters:
        N_ENVS: Parallel envs (one tree per env).
        MAX_NODES: Hard upper bound on tree size per env.
        ACT_DIM: Real action vector dimension.
        LATENT: Hidden-state dimension.
        BINS: Categorical value/reward bin count.
        K_ROOT: Root candidate count (physical slot width per node).
        K_NON_ROOT: Non-root candidate count (≤ K_ROOT). Sets the active
            slice of slots at non-root nodes.
    """

    comptime PRED_OUT: Int = 2 * Self.ACT_DIM + Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT_DIM
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime K_PAD: Int = Self.K_ROOT  # physical slot width (K_NON_ROOT ≤ K_ROOT)

    # ─── Tree per-candidate arrays [N_ENVS × MAX_NODES × K_PAD] ──────────
    var visit_count: DeviceBuffer[dtype]
    var total_value: DeviceBuffer[dtype]
    var log_prior: DeviceBuffer[dtype]
    var reward: DeviceBuffer[dtype]
    var child_idx: DeviceBuffer[dtype]

    # ─── Per-candidate action vectors [N_ENVS × MAX_NODES × K_PAD × ACT_DIM]
    var actions: DeviceBuffer[dtype]

    # ─── Per-node scalars [N_ENVS × MAX_NODES] ───────────────────────────
    var total_visits: DeviceBuffer[dtype]
    var node_value: DeviceBuffer[dtype]
    var active_k: DeviceBuffer[dtype]  # number of valid slots (K_ROOT or K_NON_ROOT)

    # ─── Hidden-state pool [N_ENVS × MAX_NODES × LATENT] ─────────────────
    var hidden_states: DeviceBuffer[dtype]

    # ─── Per-env scalars [N_ENVS] ────────────────────────────────────────
    var node_count: DeviceBuffer[dtype]
    var min_q: DeviceBuffer[dtype]
    var max_q: DeviceBuffer[dtype]

    # ─── Sequential Halving state [N_ENVS × K_ROOT] ──────────────────────
    var root_gumbels: DeviceBuffer[dtype]
    var root_active: DeviceBuffer[dtype]  # entry = candidate index or -1

    # ─── Selection / expansion scratch (one sim at a time) ───────────────
    var pending_parent: DeviceBuffer[dtype]
    var pending_cand: DeviceBuffer[dtype]
    var path_lengths: DeviceBuffer[dtype]
    var leaf_values: DeviceBuffer[dtype]
    var search_paths: DeviceBuffer[dtype]
    var cand_paths: DeviceBuffer[dtype]

    # ─── Network I/O scratch ─────────────────────────────────────────────
    var root_hidden: DeviceBuffer[dtype]
    var dyn_input: DeviceBuffer[dtype]
    var dyn_output: DeviceBuffer[dtype]
    var pred_input: DeviceBuffer[dtype]
    var pred_output: DeviceBuffer[dtype]

    # ─── Output [N_ENVS × ACT_DIM] + visit diagnostics [N_ENVS × K_ROOT] ─
    var chosen_actions: DeviceBuffer[dtype]
    var root_visits: DeviceBuffer[dtype]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime NK = Self.N_ENVS * Self.MAX_NODES * Self.K_PAD
        comptime NKA = (
            Self.N_ENVS * Self.MAX_NODES * Self.K_PAD * Self.ACT_DIM
        )
        comptime NS = Self.N_ENVS * Self.MAX_NODES
        comptime NH = Self.N_ENVS * Self.MAX_NODES * Self.LATENT

        self.visit_count = ctx.enqueue_create_buffer[dtype](NK)
        self.total_value = ctx.enqueue_create_buffer[dtype](NK)
        self.log_prior = ctx.enqueue_create_buffer[dtype](NK)
        self.reward = ctx.enqueue_create_buffer[dtype](NK)
        self.child_idx = ctx.enqueue_create_buffer[dtype](NK)
        self.actions = ctx.enqueue_create_buffer[dtype](NKA)

        self.total_visits = ctx.enqueue_create_buffer[dtype](NS)
        self.node_value = ctx.enqueue_create_buffer[dtype](NS)
        self.active_k = ctx.enqueue_create_buffer[dtype](NS)
        self.hidden_states = ctx.enqueue_create_buffer[dtype](NH)

        self.node_count = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.min_q = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.max_q = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)

        self.root_gumbels = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.K_ROOT
        )
        self.root_active = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.K_ROOT
        )

        self.pending_parent = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.pending_cand = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.path_lengths = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.leaf_values = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.search_paths = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * MAX_DEPTH
        )
        self.cand_paths = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * MAX_DEPTH
        )

        self.root_hidden = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.LATENT
        )
        self.dyn_input = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.DYN_IN
        )
        self.dyn_output = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.DYN_OUT
        )
        self.pred_input = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.LATENT
        )
        self.pred_output = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.PRED_OUT
        )

        self.chosen_actions = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT_DIM
        )
        self.root_visits = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.K_ROOT
        )

    def __init__(out self, *, deinit move: Self):
        self.visit_count = move.visit_count^
        self.total_value = move.total_value^
        self.log_prior = move.log_prior^
        self.reward = move.reward^
        self.child_idx = move.child_idx^
        self.actions = move.actions^
        self.total_visits = move.total_visits^
        self.node_value = move.node_value^
        self.active_k = move.active_k^
        self.hidden_states = move.hidden_states^
        self.node_count = move.node_count^
        self.min_q = move.min_q^
        self.max_q = move.max_q^
        self.root_gumbels = move.root_gumbels^
        self.root_active = move.root_active^
        self.pending_parent = move.pending_parent^
        self.pending_cand = move.pending_cand^
        self.path_lengths = move.path_lengths^
        self.leaf_values = move.leaf_values^
        self.search_paths = move.search_paths^
        self.cand_paths = move.cand_paths^
        self.root_hidden = move.root_hidden^
        self.dyn_input = move.dyn_input^
        self.dyn_output = move.dyn_output^
        self.pred_input = move.pred_input^
        self.pred_output = move.pred_output^
        self.chosen_actions = move.chosen_actions^
        self.root_visits = move.root_visits^

    def zero_tree(self, ctx: DeviceContext) raises:
        """Zero per-tree arrays + per-env scalars before each search."""
        ctx.enqueue_memset(self.visit_count, 0)
        ctx.enqueue_memset(self.total_value, 0)
        ctx.enqueue_memset(self.log_prior, 0)
        ctx.enqueue_memset(self.reward, 0)
        ctx.enqueue_memset(self.actions, 0)
        ctx.enqueue_memset(self.total_visits, 0)
        ctx.enqueue_memset(self.node_value, 0)
        ctx.enqueue_memset(self.active_k, 0)
        ctx.enqueue_memset(self.chosen_actions, 0)
        ctx.enqueue_memset(self.root_visits, 0)
        self.child_idx.enqueue_fill(Scalar[dtype](-1.0))
        self.root_active.enqueue_fill(Scalar[dtype](-1.0))


# ═════════════════════════════════════════════════════════════════════════
# Kernels
# ═════════════════════════════════════════════════════════════════════════
#
# Layout conventions (per env e, per node n, per candidate i, per dim d):
#   nk_off  = e * MAX_NODES * K_PAD + n * K_PAD + i
#   nka_off = ((e * MAX_NODES + n) * K_PAD + i) * ACT_DIM + d
#   ns_off  = e * MAX_NODES + n
#   h_off   = (e * MAX_NODES + n) * LATENT
#   k_off   = e * K_ROOT


def gs_scatter_root_hidden_kernel[
    N_ENVS: Int, MAX_NODES: Int, LATENT: Int,
    dtype: DType,
](
    root_hidden: Tile[dtype, N_ENVS * LATENT],
    hidden_states: Tile[dtype, N_ENVS * MAX_NODES * LATENT],
) where dtype.is_floating_point():
    """Scatter contiguous rep-forward output into each env's slot-0 hidden."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    var src = e * LATENT
    var dst = e * MAX_NODES * LATENT
    for i in range(LATENT):
        hidden_states[dst + i] = root_hidden[src + i]


def gs_init_root_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT_DIM: Int,
    BINS: Int,
    K_ROOT: Int,
    K_PAD: Int,
    PRED_OUT: Int,
    # Root sampling mode selector (mirrors CPU `SampledGumbelMCTS`):
    #   • `N_POLICY_AT_ROOT == K_ROOT`: legacy magnified — half the
    #     candidates from `N(μ, σ)`, half from `N(μ, std_mag · σ)`.
    #   • `N_POLICY_AT_ROOT < K_ROOT`: reference DMC — first
    #     `N_POLICY_AT_ROOT` from `N(μ, σ)`, rest from
    #     `Uniform(-MAX_ACTION, MAX_ACTION)`. Matches
    #     `cy_mcts.py:127-128` (policy_action_num=4, random_action_num=12
    #     for K_ROOT=16). The uniform tail decouples exploration from a
    #     potentially-biased policy μ.
    # Always evaluate `log_prior` under unmagnified `N(μ, σ)` so
    # Sequential-Halving scoring stays comparable across modes.
    N_POLICY_AT_ROOT: Int,
    dtype: DType,
](
    actions: Tile[dtype, N_ENVS * MAX_NODES * K_PAD * ACT_DIM],
    log_prior: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    node_value: Tile[dtype, N_ENVS * MAX_NODES],
    active_k: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    root_gumbels: Tile[dtype, N_ENVS * K_ROOT],
    root_active: Tile[dtype, N_ENVS * K_ROOT],
    pred_output: Tile[dtype, N_ENVS * PRED_OUT],
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
    max_action: Scalar[dtype],
    min_std: Scalar[dtype],
    std_mag: Scalar[dtype],
    soft_clamp: Scalar[dtype],
    init_std: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Initialize the root node:
        • read (μ_raw, σ_raw) from `pred_output[e, 0:2*ACT_DIM]`,
        • sample K_ROOT candidates (half N(μ, σ), half N(μ, std_mag · σ)),
        • compute log π(a_i | s) per candidate (under unmagnified σ),
        • decode root scalar value into node_value[e, 0],
        • populate root_active[e, i] = i, draw root_gumbels[e, i] = -log(-log(U)).

    One thread per env."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var pred_off = e * PRED_OUT
    var ns_off = e * MAX_NODES  # root = node 0
    var nka_off_root = (e * MAX_NODES) * K_PAD * ACT_DIM
    var nk_off_root = (e * MAX_NODES) * K_PAD
    var k_off = e * K_ROOT

    # Decode root scalar value from pred_output[2*ACT_DIM:2*ACT_DIM+BINS].
    var v_max_logit = rebind[Scalar[dtype]](
        pred_output[pred_off + 2 * ACT_DIM]
    )
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](
            pred_output[pred_off + 2 * ACT_DIM + i]
        )
        if v > v_max_logit:
            v_max_logit = v
    var v_sum = Scalar[dtype](0.0)
    for i in range(BINS):
        v_sum += exp(
            rebind[Scalar[dtype]](pred_output[pred_off + 2 * ACT_DIM + i])
            - v_max_logit
        )
    var step_v = (v_max - v_min) / Scalar[dtype](BINS - 1)
    var v_expected = Scalar[dtype](0.0)
    for i in range(BINS):
        var prob = (
            exp(
                rebind[Scalar[dtype]](
                    pred_output[pred_off + 2 * ACT_DIM + i]
                )
                - v_max_logit
            )
            / v_sum
        )
        v_expected += prob * (v_min + Scalar[dtype](i) * step_v)
    # h⁻¹ inverse scalar transform.
    var sgn = (
        Scalar[dtype](1.0)
        if v_expected >= Scalar[dtype](0.0)
        else Scalar[dtype](-1.0)
    )
    var abs_y = (
        v_expected
        if v_expected >= Scalar[dtype](0.0)
        else -v_expected
    )
    var eps_h = Scalar[dtype](0.001)
    var inner = sqrt(
        Scalar[dtype](1.0)
        + Scalar[dtype](4.0) * eps_h * (abs_y + Scalar[dtype](1.0) + eps_h)
    )
    var f = (inner - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps_h)
    node_value[ns_off] = sgn * (f * f - Scalar[dtype](1.0))

    # Per-env Philox stream — same mixing pattern as the discrete kernel.
    var philox = PhiloxRandom(
        seed=(
            UInt64(rng_seed) * UInt64(0x9E3779B97F4A7C15)
        )
        + UInt64(e * 1664525 + 1013904223),
        offset=0,
    )

    # Compute (μ, σ) per dim — we'll need them inside the per-candidate
    # sampling loop. Cache via Array.
    var mu = Array[Scalar[dtype], ACT_DIM](uninitialized=True)
    var sg = Array[Scalar[dtype], ACT_DIM](uninitialized=True)
    var inv_max = Scalar[dtype](1.0) / max_action
    var inv_soft = Scalar[dtype](1.0) / soft_clamp
    for d in range(ACT_DIM):
        var mu_raw = rebind[Scalar[dtype]](pred_output[pred_off + d])
        var sg_raw = rebind[Scalar[dtype]](
            pred_output[pred_off + ACT_DIM + d]
        )
        # μ = soft_clamp · tanh(μ_raw / soft_clamp), reference Dreamer-v3 soft clamp.
        mu[d] = soft_clamp * tanh(mu_raw * inv_soft)
        # σ = softplus(σ_raw + init_std) + min_std, numerically stable.
        var sg_pre = sg_raw + init_std
        var sp_neg_abs: Scalar[dtype]
        var sp_pos: Scalar[dtype]
        if sg_pre > Scalar[dtype](0.0):
            sp_neg_abs = -sg_pre
            sp_pos = sg_pre
        else:
            sp_neg_abs = sg_pre
            sp_pos = Scalar[dtype](0.0)
        sg[d] = (
            sp_pos
            + log(Scalar[dtype](1.0) + exp(sp_neg_abs))
            + min_std
        )

    # Root candidate sampling. Two modes selected at comptime by the
    # relationship between `N_POLICY_AT_ROOT` and `K_ROOT` (mirrors CPU
    # `mcts_sampled.SampledGumbelMCTS.search`, lines 453-517):
    #
    #   LEGACY_MAGNIFIED (N_POLICY_AT_ROOT == K_ROOT):
    #       slot i < K_ROOT/2 → sample from N(μ, σ)
    #       slot i ≥ K_ROOT/2 → sample from N(μ, std_mag · σ)
    #
    #   REFERENCE_DMC (N_POLICY_AT_ROOT < K_ROOT):
    #       slot i < N_POLICY_AT_ROOT → sample from N(μ, σ)
    #       slot i ≥ N_POLICY_AT_ROOT → uniform random in (-max_action, max_action)
    #
    # `log_prior` is always evaluated under the unmagnified `N(μ, σ)` so
    # Sequential-Halving scoring remains comparable across modes (widened
    # / uniform samples don't get double-penalized for living in the tail
    # of `N(μ, σ)`).
    comptime LEGACY_MAGNIFIED: Bool = N_POLICY_AT_ROOT == K_ROOT
    comptime HALF_K: Int = K_ROOT // 2
    for i in range(K_ROOT):
        var is_policy_sample: Bool
        var is_magnified: Bool
        comptime if LEGACY_MAGNIFIED:
            is_policy_sample = True
            is_magnified = i >= HALF_K
        else:
            is_policy_sample = i < N_POLICY_AT_ROOT
            is_magnified = False

        var lp = Scalar[dtype](0.0)
        for d in range(ACT_DIM):
            # Box-Muller from two uniform draws — used by both the
            # policy-sample path (z ~ N(0, 1)) and consumed unused for
            # uniform samples (we still draw the same two uniforms so the
            # Philox stream advance is identical across modes, easing
            # future RNG accounting).
            var u1 = philox.step_uniform()
            var u2 = philox.step_uniform()

            var a_d: Scalar[dtype]
            if is_policy_sample:
                var u1f = Scalar[dtype](u1[0])
                if u1f < Scalar[dtype](1e-9):
                    u1f = Scalar[dtype](1e-9)
                var z = sqrt(
                    Scalar[dtype](-2.0) * log(u1f)
                ) * cos(
                    Scalar[dtype](2.0)
                    * Scalar[dtype](pi)
                    * Scalar[dtype](u2[0])
                )
                var sg_eff = sg[d]
                if is_magnified:
                    sg_eff = sg_eff * std_mag
                var u_d = mu[d] + sg_eff * z
                a_d = max_action * tanh(u_d)
            else:
                # Uniform draw in (-max_action, max_action). Re-use `u1`
                # as the [0, 1) source — drawing one extra Box-Muller pair
                # per dim keeps the Philox stream alignment identical to
                # the policy-sample path so a future N_POLICY_AT_ROOT
                # change doesn't perturb earlier slots' RNG.
                var uf = Scalar[dtype](u1[0])
                a_d = max_action * (
                    Scalar[dtype](2.0) * uf - Scalar[dtype](1.0)
                )

            actions[nka_off_root + i * ACT_DIM + d] = a_d

            # log_prior under N(μ, σ) (unmagnified), same density for
            # both policy-sample and uniform candidates.
            var c = a_d * inv_max
            var c_lo = Scalar[dtype](-0.999)
            var c_hi = Scalar[dtype](0.999)
            if c > c_hi:
                c = c_hi
            if c < c_lo:
                c = c_lo
            var u_star = Scalar[dtype](0.5) * log(
                (Scalar[dtype](1.0) + c) / (Scalar[dtype](1.0) - c)
            )
            var diff = u_star - mu[d]
            var inv_sg = Scalar[dtype](1.0) / sg[d]
            var eta = diff * inv_sg
            lp = (
                lp
                + Scalar[dtype](-0.5) * eta * eta
                - log(sg[d])
                - Scalar[dtype](0.5) * Scalar[dtype](LOG_2PI_F)
                - log(Scalar[dtype](1.0) - c * c)
            )
        log_prior[nk_off_root + i] = lp

    # Gumbel noise per root candidate.
    for i in range(K_ROOT):
        var u = philox.step_uniform()
        var uv = Scalar[dtype](u[0])
        if uv < Scalar[dtype](1e-9):
            uv = Scalar[dtype](1e-9)
        if uv > Scalar[dtype](1.0) - Scalar[dtype](1e-9):
            uv = Scalar[dtype](1.0) - Scalar[dtype](1e-9)
        root_gumbels[k_off + i] = -log(-log(uv))
        root_active[k_off + i] = Scalar[dtype](i)

    active_k[ns_off] = Scalar[dtype](K_ROOT)
    node_count[e] = Scalar[dtype](1.0)
    min_q[e] = Scalar[dtype](1e18)
    max_q[e] = Scalar[dtype](-1e18)


def gs_select_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT_DIM: Int,
    K_ROOT: Int,
    K_PAD: Int,
    LATENT: Int,
    DYN_IN: Int,
    dtype: DType,
](
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    log_prior: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    actions: Tile[dtype, N_ENVS * MAX_NODES * K_PAD * ACT_DIM],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_value: Tile[dtype, N_ENVS * MAX_NODES],
    active_k: Tile[dtype, N_ENVS * MAX_NODES],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    root_active: Tile[dtype, N_ENVS * K_ROOT],
    hidden_states: Tile[dtype, N_ENVS * MAX_NODES * LATENT],
    dyn_input: Tile[dtype, N_ENVS * DYN_IN],
    pending_parent: Tile[dtype, N_ENVS],
    pending_cand: Tile[dtype, N_ENVS],
    search_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    cand_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS],
    current_slot: Scalar[DType.int32],
    c_visit: Scalar[dtype],
    c_scale: Scalar[dtype],
) where dtype.is_floating_point():
    """One simulation's selection phase, per env. Picks the root candidate
    at slot `current_slot`, descends by visit-balance until an unexpanded
    candidate is hit, and writes (parent, cand, dyn_input) for the
    expansion phase."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var ns_off = e * MAX_NODES
    var k_off_root = e * K_ROOT
    var path_off = e * MAX_DEPTH

    var slot_idx = Int(current_slot)
    if slot_idx < 0:
        slot_idx = 0
    var cand_idx = Int(
        rebind[Scalar[dtype]](root_active[k_off_root + slot_idx])
    )
    if cand_idx < 0:
        cand_idx = 0

    var node_idx = 0
    var depth = 0
    search_paths[path_off] = Scalar[dtype](0.0)
    cand_paths[path_off] = Scalar[dtype](cand_idx)
    var current_cand = cand_idx

    while depth < MAX_DEPTH - 1:
        var nk_base = (
            e * MAX_NODES * K_PAD + node_idx * K_PAD
        )
        var child = rebind[Scalar[dtype]](
            child_idx[nk_base + current_cand]
        )
        if child < Scalar[dtype](0.0):
            break

        node_idx = Int(child)
        depth += 1
        search_paths[path_off + depth] = Scalar[dtype](node_idx)

        # Non-root visit-balance over the K_NON_ROOT active candidates.
        var ns_idx = ns_off + node_idx
        var nk_base_child = (
            e * MAX_NODES * K_PAD + node_idx * K_PAD
        )
        var ak_child = Int(rebind[Scalar[dtype]](active_k[ns_idx]))
        var n_total = rebind[Scalar[dtype]](total_visits[ns_idx])
        var v_self = rebind[Scalar[dtype]](node_value[ns_idx])

        # v_mix
        var max_visited_lp = Scalar[dtype](-1e18)
        var any_visited = False
        for i in range(ak_child):
            var nva = rebind[Scalar[dtype]](
                visit_count[nk_base_child + i]
            )
            if nva > Scalar[dtype](0.5):
                var lp = rebind[Scalar[dtype]](
                    log_prior[nk_base_child + i]
                )
                if lp > max_visited_lp:
                    max_visited_lp = lp
                any_visited = True
        var v_mix = v_self
        if any_visited:
            var sum_w = Scalar[dtype](0.0)
            var weighted_q = Scalar[dtype](0.0)
            for i in range(ak_child):
                var nva = rebind[Scalar[dtype]](
                    visit_count[nk_base_child + i]
                )
                if nva > Scalar[dtype](0.5):
                    var lp = rebind[Scalar[dtype]](
                        log_prior[nk_base_child + i]
                    )
                    var w = exp(lp - max_visited_lp)
                    sum_w += w
                    var qa = (
                        rebind[Scalar[dtype]](
                            total_value[nk_base_child + i]
                        )
                        / nva
                    )
                    weighted_q += w * qa
            if sum_w > Scalar[dtype](1e-12):
                v_mix = (
                    v_self + n_total * (weighted_q / sum_w)
                ) / (Scalar[dtype](1.0) + n_total)

        # σ(completed_Q) per candidate.
        var max_visit = Scalar[dtype](0.0)
        for i in range(ak_child):
            var nva = rebind[Scalar[dtype]](
                visit_count[nk_base_child + i]
            )
            if nva > max_visit:
                max_visit = nva
        var sigma_scale = (c_visit + max_visit) * c_scale
        # Per-NODE completed-Q min/max (mctx qtransform semantics) — the old
        # tree-GLOBAL min_q/max_q normalization compressed σ to near-noise
        # whenever the tree's Q range dwarfed sibling ΔQ (see
        # mcts_gpu_gumbel.mojo gz_select_kernel for the full story).
        var node_mn = Scalar[dtype](1e18)
        var node_mx = Scalar[dtype](-1e18)
        for i in range(ak_child):
            var nva = rebind[Scalar[dtype]](
                visit_count[nk_base_child + i]
            )
            var cq = v_mix
            if nva > Scalar[dtype](0.5):
                cq = (
                    rebind[Scalar[dtype]](
                        total_value[nk_base_child + i]
                    )
                    / nva
                )
            if cq < node_mn:
                node_mn = cq
            if cq > node_mx:
                node_mx = cq
        var q_range = node_mx - node_mn

        var z = Array[Scalar[dtype], K_PAD](uninitialized=True)
        var max_z = Scalar[dtype](-1e18)
        for i in range(K_PAD):
            z[i] = Scalar[dtype](-1e18)
        for i in range(ak_child):
            var nva = rebind[Scalar[dtype]](
                visit_count[nk_base_child + i]
            )
            var qa: Scalar[dtype]
            if nva > Scalar[dtype](0.5):
                qa = (
                    rebind[Scalar[dtype]](
                        total_value[nk_base_child + i]
                    )
                    / nva
                )
            else:
                qa = v_mix
            var qn: Scalar[dtype]
            if q_range > Scalar[dtype](1e-8):
                qn = (qa - node_mn) / q_range
            else:
                qn = Scalar[dtype](0.0)  # all completed Q equal → σ inert
            z[i] = (
                rebind[Scalar[dtype]](log_prior[nk_base_child + i])
                + sigma_scale * qn
            )
            if z[i] > max_z:
                max_z = z[i]

        var sum_e = Scalar[dtype](0.0)
        var probs = Array[Scalar[dtype], K_PAD](uninitialized=True)
        for i in range(K_PAD):
            probs[i] = Scalar[dtype](0.0)
        for i in range(ak_child):
            var ev = exp(z[i] - max_z)
            probs[i] = ev
            sum_e += ev
        # Uniform fallback when sum underflows (z spread > ~700 float64
        # or ~88 float32 → all exp(z[i]-max_z) underflow except the max,
        # whose sum is dominated by a single 1.0; in pathological cases
        # of multiple max-tied candidates this can still net to 0). The
        # CPU implementation (mcts_sampled.mojo:720-723) does the same
        # uniform-fallback; without it the GPU's `probs` stay at 0 →
        # visit-balance degenerates to pure round-robin instead of the
        # uniform-mixture-vs-N rule.
        if sum_e <= Scalar[dtype](1e-12):
            var inv_ak = Scalar[dtype](1.0) / Scalar[dtype](ak_child)
            for i in range(ak_child):
                probs[i] = inv_ak
        else:
            for i in range(ak_child):
                probs[i] = probs[i] / sum_e

        var denom = Scalar[dtype](1.0) + n_total
        var best_i = 0
        var best_s = Scalar[dtype](-1e18)
        for i in range(ak_child):
            var nva = rebind[Scalar[dtype]](
                visit_count[nk_base_child + i]
            )
            var s = probs[i] - nva / denom
            if s > best_s:
                best_s = s
                best_i = i
        current_cand = best_i
        cand_paths[path_off + depth] = Scalar[dtype](best_i)

    # Record leaf info.
    pending_parent[e] = Scalar[dtype](node_idx)
    pending_cand[e] = Scalar[dtype](current_cand)
    path_lengths[e] = Scalar[dtype](depth + 1)

    # Build dyn_input = [parent_hidden ‖ action_vec(parent, current_cand)].
    var d_off = e * DYN_IN
    var h_off = (e * MAX_NODES + node_idx) * LATENT
    var a_off = (
        ((e * MAX_NODES + node_idx) * K_PAD + current_cand) * ACT_DIM
    )
    for i in range(LATENT):
        dyn_input[d_off + i] = hidden_states[h_off + i]
    for d in range(ACT_DIM):
        dyn_input[d_off + LATENT + d] = actions[a_off + d]


def gs_copy_pred_input_kernel[
    N_ENVS: Int, LATENT: Int, DYN_OUT: Int,
    dtype: DType,
](
    pred_input: Tile[dtype, N_ENVS * LATENT],
    dyn_output: Tile[dtype, N_ENVS * DYN_OUT],
) where dtype.is_floating_point():
    """Copy LATENT prefix of dyn_output into pred_input."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    var src = e * DYN_OUT
    var dst = e * LATENT
    for i in range(LATENT):
        pred_input[dst + i] = dyn_output[src + i]


def gs_expand_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT_DIM: Int,
    K_ROOT: Int,
    K_NON_ROOT: Int,
    K_PAD: Int,
    LATENT: Int,
    BINS: Int,
    PRED_OUT: Int,
    DYN_OUT: Int,
    dtype: DType,
](
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    log_prior: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    reward: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    actions: Tile[dtype, N_ENVS * MAX_NODES * K_PAD * ACT_DIM],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_value: Tile[dtype, N_ENVS * MAX_NODES],
    active_k: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    hidden_states: Tile[dtype, N_ENVS * MAX_NODES * LATENT],
    pending_parent: Tile[dtype, N_ENVS],
    pending_cand: Tile[dtype, N_ENVS],
    dyn_output: Tile[dtype, N_ENVS * DYN_OUT],
    pred_output: Tile[dtype, N_ENVS * PRED_OUT],
    leaf_values: Tile[dtype, N_ENVS],
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
    reward_min: Scalar[dtype],
    reward_max: Scalar[dtype],
    max_action: Scalar[dtype],
    min_std: Scalar[dtype],
    soft_clamp: Scalar[dtype],
    init_std: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
    sim_index: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Expand the leaf for each env: write hidden state, decode reward,
    sample K_NON_ROOT child candidates from N(μ, σ) (no magnification),
    populate child node, decode child value into both `node_value` and
    `leaf_values`."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var cand = Int(rebind[Scalar[dtype]](pending_cand[e]))
    var child = Int(rebind[Scalar[dtype]](node_count[e]))
    if child >= MAX_NODES:
        leaf_values[e] = Scalar[dtype](0.0)
        return

    var ns_off = e * MAX_NODES
    var dyn_off = e * DYN_OUT
    var pred_off = e * PRED_OUT
    var child_h_off = (e * MAX_NODES + child) * LATENT
    var nk_base_child = (e * MAX_NODES + child) * K_PAD
    var nk_base_parent = (e * MAX_NODES + parent) * K_PAD
    var nka_base_child = nk_base_child * ACT_DIM

    # Hidden state: copy first LATENT values from dyn_output.
    for i in range(LATENT):
        hidden_states[child_h_off + i] = dyn_output[dyn_off + i]

    # Reward decoding (categorical → scalar with h⁻¹).
    comptime NUM_REW_BINS = DYN_OUT - LATENT
    var rew_decoded: Scalar[dtype]
    if NUM_REW_BINS == 1:
        rew_decoded = rebind[Scalar[dtype]](
            dyn_output[dyn_off + LATENT]
        )
    else:
        var r_max = rebind[Scalar[dtype]](
            dyn_output[dyn_off + LATENT]
        )
        for i in range(1, NUM_REW_BINS):
            var v = rebind[Scalar[dtype]](
                dyn_output[dyn_off + LATENT + i]
            )
            if v > r_max:
                r_max = v
        var r_sum = Scalar[dtype](0.0)
        for i in range(NUM_REW_BINS):
            r_sum += exp(
                rebind[Scalar[dtype]](
                    dyn_output[dyn_off + LATENT + i]
                )
                - r_max
            )
        # Reward decode uses the reward support (separate from the value
        # support — paper `dmc_state.yaml` carries `reward_support:
        # range=[-2, 2]` and `value_support: range=[-299, 299]`).
        var r_step = (reward_max - reward_min) / Scalar[dtype](
            NUM_REW_BINS - 1
        )
        var r_expected = Scalar[dtype](0.0)
        for i in range(NUM_REW_BINS):
            var p = (
                exp(
                    rebind[Scalar[dtype]](
                        dyn_output[dyn_off + LATENT + i]
                    )
                    - r_max
                )
                / r_sum
            )
            r_expected += p * (reward_min + Scalar[dtype](i) * r_step)
        var sgn_r = (
            Scalar[dtype](1.0)
            if r_expected >= Scalar[dtype](0.0)
            else Scalar[dtype](-1.0)
        )
        var abs_r = (
            r_expected
            if r_expected >= Scalar[dtype](0.0)
            else -r_expected
        )
        var eps_r = Scalar[dtype](0.001)
        var inner_r = sqrt(
            Scalar[dtype](1.0)
            + Scalar[dtype](4.0) * eps_r * (
                abs_r + Scalar[dtype](1.0) + eps_r
            )
        )
        var f_r = (inner_r - Scalar[dtype](1.0)) / (
            Scalar[dtype](2.0) * eps_r
        )
        rew_decoded = sgn_r * (f_r * f_r - Scalar[dtype](1.0))
    reward[nk_base_parent + cand] = rew_decoded

    # Sample K_NON_ROOT child candidates from N(μ, σ).
    # Per-env Philox stream — mix in sim_index so each expansion
    # gets uncorrelated noise.
    var philox = PhiloxRandom(
        seed=(
            UInt64(rng_seed) * UInt64(0x9E3779B97F4A7C15)
        )
        + UInt64(e * 1664525 + 1013904223)
        + UInt64(sim_index) * UInt64(0xD1B54A32D192ED03),
        offset=0,
    )

    var inv_max = Scalar[dtype](1.0) / max_action
    var inv_soft = Scalar[dtype](1.0) / soft_clamp
    var mu = Array[Scalar[dtype], ACT_DIM](uninitialized=True)
    var sg = Array[Scalar[dtype], ACT_DIM](uninitialized=True)
    for d in range(ACT_DIM):
        var mu_raw = rebind[Scalar[dtype]](
            pred_output[pred_off + d]
        )
        var sg_raw = rebind[Scalar[dtype]](
            pred_output[pred_off + ACT_DIM + d]
        )
        # μ = soft_clamp · tanh(μ_raw / soft_clamp), reference Dreamer-v3 soft clamp.
        mu[d] = soft_clamp * tanh(mu_raw * inv_soft)
        # σ = softplus(σ_raw + init_std) + min_std, numerically stable.
        var sg_pre = sg_raw + init_std
        var sp_neg_abs: Scalar[dtype]
        var sp_pos: Scalar[dtype]
        if sg_pre > Scalar[dtype](0.0):
            sp_neg_abs = -sg_pre
            sp_pos = sg_pre
        else:
            sp_neg_abs = sg_pre
            sp_pos = Scalar[dtype](0.0)
        sg[d] = (
            sp_pos
            + log(Scalar[dtype](1.0) + exp(sp_neg_abs))
            + min_std
        )

    for i in range(K_NON_ROOT):
        var lp = Scalar[dtype](0.0)
        for d in range(ACT_DIM):
            var u1 = philox.step_uniform()
            var u2 = philox.step_uniform()
            var u1f = Scalar[dtype](u1[0])
            if u1f < Scalar[dtype](1e-9):
                u1f = Scalar[dtype](1e-9)
            var z = sqrt(
                Scalar[dtype](-2.0) * log(u1f)
            ) * cos(
                Scalar[dtype](2.0) * Scalar[dtype](pi) * Scalar[dtype](
                    u2[0]
                )
            )
            var u_d = mu[d] + sg[d] * z
            var a_d = max_action * tanh(u_d)
            actions[nka_base_child + i * ACT_DIM + d] = a_d

            var c = a_d * inv_max
            var c_lo = Scalar[dtype](-0.999)
            var c_hi = Scalar[dtype](0.999)
            if c > c_hi:
                c = c_hi
            if c < c_lo:
                c = c_lo
            var u_star = Scalar[dtype](0.5) * log(
                (Scalar[dtype](1.0) + c) / (Scalar[dtype](1.0) - c)
            )
            var diff = u_star - mu[d]
            var inv_sg = Scalar[dtype](1.0) / sg[d]
            var eta = diff * inv_sg
            lp = (
                lp
                + Scalar[dtype](-0.5) * eta * eta
                - log(sg[d])
                - Scalar[dtype](0.5) * Scalar[dtype](LOG_2PI_F)
                - log(Scalar[dtype](1.0) - c * c)
            )
        log_prior[nk_base_child + i] = lp
        visit_count[nk_base_child + i] = Scalar[dtype](0.0)
        total_value[nk_base_child + i] = Scalar[dtype](0.0)
        reward[nk_base_child + i] = Scalar[dtype](0.0)
        child_idx[nk_base_child + i] = Scalar[dtype](-1.0)

    active_k[ns_off + child] = Scalar[dtype](K_NON_ROOT)
    total_visits[ns_off + child] = Scalar[dtype](0.0)

    # Decode child scalar value.
    var v_max_logit = rebind[Scalar[dtype]](
        pred_output[pred_off + 2 * ACT_DIM]
    )
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](
            pred_output[pred_off + 2 * ACT_DIM + i]
        )
        if v > v_max_logit:
            v_max_logit = v
    var v_sum = Scalar[dtype](0.0)
    for i in range(BINS):
        v_sum += exp(
            rebind[Scalar[dtype]](
                pred_output[pred_off + 2 * ACT_DIM + i]
            )
            - v_max_logit
        )
    var step_v = (v_max - v_min) / Scalar[dtype](BINS - 1)
    var v_expected = Scalar[dtype](0.0)
    for i in range(BINS):
        var prob = (
            exp(
                rebind[Scalar[dtype]](
                    pred_output[pred_off + 2 * ACT_DIM + i]
                )
                - v_max_logit
            )
            / v_sum
        )
        v_expected += prob * (v_min + Scalar[dtype](i) * step_v)
    var sgn = (
        Scalar[dtype](1.0)
        if v_expected >= Scalar[dtype](0.0)
        else Scalar[dtype](-1.0)
    )
    var abs_y = (
        v_expected
        if v_expected >= Scalar[dtype](0.0)
        else -v_expected
    )
    var eps_h = Scalar[dtype](0.001)
    var inner = sqrt(
        Scalar[dtype](1.0)
        + Scalar[dtype](4.0) * eps_h * (abs_y + Scalar[dtype](1.0) + eps_h)
    )
    var f = (inner - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps_h)
    var v_decoded = sgn * (f * f - Scalar[dtype](1.0))
    node_value[ns_off + child] = v_decoded
    leaf_values[e] = v_decoded

    # Link parent → child.
    child_idx[nk_base_parent + cand] = Scalar[dtype](child)
    node_count[e] = Scalar[dtype](child + 1)


def gs_backup_kernel[
    N_ENVS: Int, MAX_NODES: Int, K_PAD: Int,
    dtype: DType,
](
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    reward: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    search_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    cand_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS],
    leaf_values: Tile[dtype, N_ENVS],
    gamma: Scalar[dtype],
) where dtype.is_floating_point():
    """Walk path leaf→root, accumulate discounted return, refresh min/max_Q."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var path_off = e * MAX_DEPTH
    var path_len = Int(rebind[Scalar[dtype]](path_lengths[e]))

    var value = rebind[Scalar[dtype]](leaf_values[e])
    for i in range(path_len):
        var idx = path_len - 1 - i
        var node_idx = Int(
            rebind[Scalar[dtype]](search_paths[path_off + idx])
        )
        var ci = Int(
            rebind[Scalar[dtype]](cand_paths[path_off + idx])
        )
        var nk_off = (e * MAX_NODES + node_idx) * K_PAD + ci
        value = rebind[Scalar[dtype]](reward[nk_off]) + gamma * value
        visit_count[nk_off] = (
            rebind[Scalar[dtype]](visit_count[nk_off])
            + Scalar[dtype](1.0)
        )
        total_value[nk_off] = (
            rebind[Scalar[dtype]](total_value[nk_off]) + value
        )
        var ns_off = e * MAX_NODES + node_idx
        total_visits[ns_off] = (
            rebind[Scalar[dtype]](total_visits[ns_off])
            + Scalar[dtype](1.0)
        )

        var n_a = rebind[Scalar[dtype]](visit_count[nk_off])
        var mean_q = rebind[Scalar[dtype]](total_value[nk_off]) / n_a
        if mean_q < rebind[Scalar[dtype]](min_q[e]):
            min_q[e] = mean_q
        if mean_q > rebind[Scalar[dtype]](max_q[e]):
            max_q[e] = mean_q


def gs_halve_active_kernel[
    N_ENVS: Int, MAX_NODES: Int, K_ROOT: Int, K_PAD: Int,
    dtype: DType,
](
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    log_prior: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_value: Tile[dtype, N_ENVS * MAX_NODES],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    root_gumbels: Tile[dtype, N_ENVS * K_ROOT],
    root_active: Tile[dtype, N_ENVS * K_ROOT],
    old_size: Scalar[DType.int32],
    keep: Scalar[DType.int32],
    c_visit: Scalar[dtype],
    c_scale: Scalar[dtype],
) where dtype.is_floating_point():
    """Sequential-Halving phase boundary: keep top-`keep` active root
    candidates by `g_i + log_prior_i + σ(completed_Q_i)`."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var k_off = e * K_ROOT
    var nk_base = e * MAX_NODES * K_PAD  # root = node 0
    var ns_idx = e * MAX_NODES

    var v_self = rebind[Scalar[dtype]](node_value[ns_idx])
    var n_total = rebind[Scalar[dtype]](total_visits[ns_idx])

    # v_mix
    var max_visited_lp = Scalar[dtype](-1e18)
    var any_visited = False
    for i in range(K_ROOT):
        var nva = rebind[Scalar[dtype]](visit_count[nk_base + i])
        if nva > Scalar[dtype](0.5):
            var lp = rebind[Scalar[dtype]](log_prior[nk_base + i])
            if lp > max_visited_lp:
                max_visited_lp = lp
            any_visited = True
    var v_mix = v_self
    if any_visited:
        var sum_w = Scalar[dtype](0.0)
        var weighted_q = Scalar[dtype](0.0)
        for i in range(K_ROOT):
            var nva = rebind[Scalar[dtype]](visit_count[nk_base + i])
            if nva > Scalar[dtype](0.5):
                var lp = rebind[Scalar[dtype]](log_prior[nk_base + i])
                var w = exp(lp - max_visited_lp)
                sum_w += w
                var qa = (
                    rebind[Scalar[dtype]](total_value[nk_base + i]) / nva
                )
                weighted_q += w * qa
        if sum_w > Scalar[dtype](1e-12):
            v_mix = (
                v_self + n_total * (weighted_q / sum_w)
            ) / (Scalar[dtype](1.0) + n_total)

    var max_visit = Scalar[dtype](0.0)
    for i in range(K_ROOT):
        var nva = rebind[Scalar[dtype]](visit_count[nk_base + i])
        if nva > max_visit:
            max_visit = nva
    var sigma_scale = (c_visit + max_visit) * c_scale
    # Per-NODE (root) completed-Q min/max over the K_ROOT candidates — see
    # the interior-select site above / mcts_gpu_gumbel.mojo for why the old
    # tree-GLOBAL min_q/max_q normalization neutered σ(completed_Q).
    var node_mn = Scalar[dtype](1e18)
    var node_mx = Scalar[dtype](-1e18)
    for i in range(K_ROOT):
        var nva = rebind[Scalar[dtype]](visit_count[nk_base + i])
        var cq = v_mix
        if nva > Scalar[dtype](0.5):
            cq = rebind[Scalar[dtype]](total_value[nk_base + i]) / nva
        if cq < node_mn:
            node_mn = cq
        if cq > node_mx:
            node_mx = cq
    var q_range = node_mx - node_mn

    var old_n = Int(old_size)
    if old_n > K_ROOT:
        old_n = K_ROOT
    var keep_n = Int(keep)
    if keep_n < 1:
        keep_n = 1
    if keep_n > old_n:
        keep_n = old_n

    var scores = Array[Scalar[dtype], K_ROOT](uninitialized=True)
    var active_idx = Array[Int, K_ROOT](uninitialized=True)
    for i in range(K_ROOT):
        scores[i] = Scalar[dtype](-1e18)
        active_idx[i] = -1
    for i in range(old_n):
        var cand = Int(rebind[Scalar[dtype]](root_active[k_off + i]))
        if cand < 0:
            continue
        var nva = rebind[Scalar[dtype]](visit_count[nk_base + cand])
        var qa: Scalar[dtype]
        if nva > Scalar[dtype](0.5):
            qa = rebind[Scalar[dtype]](total_value[nk_base + cand]) / nva
        else:
            qa = v_mix
        var qn: Scalar[dtype]
        if q_range > Scalar[dtype](1e-8):
            qn = (qa - node_mn) / q_range
        else:
            qn = Scalar[dtype](0.0)  # all completed Q equal → σ inert
        var sigma_q = sigma_scale * qn
        var lp = rebind[Scalar[dtype]](log_prior[nk_base + cand])
        var g = rebind[Scalar[dtype]](root_gumbels[k_off + cand])
        scores[i] = g + lp + sigma_q
        active_idx[i] = cand

    for slot in range(keep_n):
        var best = slot
        for j in range(slot + 1, old_n):
            if scores[j] > scores[best]:
                best = j
        if best != slot:
            var ts = scores[slot]
            scores[slot] = scores[best]
            scores[best] = ts
            var ti = active_idx[slot]
            active_idx[slot] = active_idx[best]
            active_idx[best] = ti

    for i in range(K_ROOT):
        root_active[k_off + i] = Scalar[dtype](-1.0)
    for i in range(keep_n):
        root_active[k_off + i] = Scalar[dtype](active_idx[i])


def gs_extract_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT_DIM: Int,
    K_ROOT: Int,
    K_PAD: Int,
    dtype: DType,
](
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * K_PAD],
    actions: Tile[dtype, N_ENVS * MAX_NODES * K_PAD * ACT_DIM],
    chosen_actions: Tile[dtype, N_ENVS * ACT_DIM],
    root_visits_out: Tile[dtype, N_ENVS * K_ROOT],
    deterministic: Scalar[DType.uint8],
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Build per-env outputs from the root visit counts:
        • normalized visit distribution → `root_visits_out[e, i]`,
        • chosen action vector — argmax-visit if deterministic, else
          a Philox draw weighted by visit counts.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var nk_base = e * MAX_NODES * K_PAD  # root = node 0
    var k_off = e * K_ROOT
    var a_off_root = (e * MAX_NODES) * K_PAD * ACT_DIM
    var ca_off = e * ACT_DIM

    # Read visit counts.
    var visits = Array[Scalar[dtype], K_ROOT](uninitialized=True)
    var total = Scalar[dtype](0.0)
    for i in range(K_ROOT):
        var v = rebind[Scalar[dtype]](visit_count[nk_base + i])
        visits[i] = v
        total += v

    # Normalize.
    if total > Scalar[dtype](0.5):
        for i in range(K_ROOT):
            root_visits_out[k_off + i] = visits[i] / total
    else:
        var u = Scalar[dtype](1.0) / Scalar[dtype](K_ROOT)
        for i in range(K_ROOT):
            root_visits_out[k_off + i] = u

    var chosen_idx = 0
    if deterministic > Scalar[DType.uint8](0):
        var best_v = Scalar[dtype](-1.0)
        for i in range(K_ROOT):
            if visits[i] > best_v:
                best_v = visits[i]
                chosen_idx = i
    else:
        # Weighted soft pick via Philox uniform draw.
        var philox = PhiloxRandom(
            seed=(
                UInt64(rng_seed) * UInt64(0x9E3779B97F4A7C15)
            )
            + UInt64(e * 1664525 + 1013904223)
            + UInt64(0xCAFEBABE),  # decorrelate from sampling streams
            offset=0,
        )
        var u = philox.step_uniform()
        var uf = Scalar[dtype](u[0])
        var acc = Scalar[dtype](0.0)
        var picked = False
        for i in range(K_ROOT):
            var p = rebind[Scalar[dtype]](root_visits_out[k_off + i])
            acc += p
            if not picked and uf <= acc:
                chosen_idx = i
                picked = True
        if not picked:
            chosen_idx = K_ROOT - 1

    var src = a_off_root + chosen_idx * ACT_DIM
    for d in range(ACT_DIM):
        chosen_actions[ca_off + d] = actions[src + d]


