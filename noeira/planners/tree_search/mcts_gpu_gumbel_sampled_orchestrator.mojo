"""SampledGumbelGPUMCTS — continuous-action Gumbel-MCTS orchestrator (EZv2).

The agent-side counterpart to ``GumbelGPUMCTS`` for the EfficientZero V2
**sampled / continuous** planner. Owns an ``EZV2GPUSampledMCTSState``
plus a ``root_value_out`` device buffer, exposes
``search_gpu[REP, DYN, PRED]`` that drives the full
encode → predict (root) → init_root (sample K_ROOT candidates with
Gumbel) → Sequential Halving → extract chosen action pipeline.

What's different vs ``GumbelGPUMCTS``:
  • Action space is continuous — per-node arrays are keyed by candidate
    index (slot ``i ∈ [0, K_ROOT)``) rather than discrete action index.
    Each slot also stores the real ``ACT_DIM``-vector that the candidate
    represents in ``state.actions[N, MAX_NODES, K_PAD, ACT_DIM]``.
  • Root sampling: ``K_ROOT`` candidates drawn either as legacy
    magnified (``N_POLICY_AT_ROOT == K_ROOT`` → half ``N(μ, σ)``, half
    ``N(μ, std_mag·σ)``) or DMC-style (``N_POLICY_AT_ROOT < K_ROOT`` →
    first slice from ``N(μ, σ)``, rest uniform on ``±max_action``). The
    comptime template arg selects which mode at compile time.
  • Pred head: emits ``(μ_raw, σ_raw)`` per ACT_DIM plus value bins
    (``PRED_OUT_DIM = 2·ACT_DIM + BINS``). Adapter is
    ``EZv2PredGPUSampled`` (not the discrete ``EZv2PredGPU``).
  • Selection rule (non-root): same Gumbel-MCTS visit-balance
    ``argmax_i [π_improved(i) − N(i)/(1+ΣN(j))]`` over ``K_NON_ROOT``
    slots.
  • Output: writes ``chosen_actions[N_ENVS × ACT_DIM]`` (argmax-visit
    candidate at root if ``deterministic``, else visit-weighted soft
    pick) and ``root_visits[N_ENVS × K_ROOT]`` diagnostics. There is no
    improved-policy distribution to extract because the action space
    is continuous — agent reads ``chosen_actions`` directly.

Output buffers exposed:
  • ``chosen_actions_view()`` → ``[N_ENVS × ACT_DIM]`` continuous action
    pick per env.
  • ``root_visits_view()`` → ``[N_ENVS × K_ROOT]`` visit counts at the
    root, indexed by candidate slot (for diagnostics and the value
    target's policy-weighted Q computation).
  • ``root_value_view()`` → ``[N_ENVS]`` scalar root value (scattered
    from ``state.node_value[e * MAX_NODES]``). Decoded by
    ``gs_init_root_kernel`` at root expansion time.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT as dtype

comptime TPB = 256  # preserved from legacy nn.constants (nn.TPB == 128)

from .model_traits_gpu import (
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
)
from .mcts_gpu_gumbel_sampled import (
    MAX_DEPTH,
    EZV2GPUSampledMCTSState,
    gs_scatter_root_hidden_kernel,
    gs_init_root_kernel,
    gs_select_kernel,
    gs_copy_pred_input_kernel,
    gs_expand_kernel,
    gs_backup_kernel,
    gs_halve_active_kernel,
    gs_extract_kernel,
)
from .mcts_gpu_gumbel_orchestrator import gz_extract_root_value_kernel
from noeira.core.tiles import Tile, tile


# ═════════════════════════════════════════════════════════════════════════
# Helpers (private duplicates of the legacy driver's helpers).
# ═════════════════════════════════════════════════════════════════════════


def _ilog2(n: Int) -> Int:
    var x = n
    var r = 0
    while x > 1:
        x = x // 2
        r += 1
    return r


def _largest_power_of_two_le(n: Int) -> Int:
    if n < 1:
        return 1
    var p = 1
    while p * 2 <= n:
        p *= 2
    return p


# ═════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════


struct SampledGumbelGPUMCTS[
    N_ENVS: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
    MAX_NODES: Int,
    K_ROOT: Int,
    K_NON_ROOT: Int,
    NUM_SIMULATIONS: Int,
    # Root sampling mode selector — see ``gs_init_root_kernel``. Default
    # ``K_ROOT`` preserves legacy magnified behavior so existing template
    # ordering keeps working. Tail position so legacy positional callers
    # stay unchanged.
    N_POLICY_AT_ROOT: Int = K_ROOT,
](Deinitable, Movable):
    """GPU sampled-Gumbel MCTS orchestrator (EZv2 continuous planner).

    Comptime params:
        N_ENVS: Parallel envs / trees.
        ACT_DIM: Continuous action vector dim.
        LATENT: Hidden state dim.
        BINS: Categorical reward / value bins.
        MAX_NODES: Per-tree node arena.
        K_ROOT: Root candidate count (physical slot width). Must be a
            power of two.
        K_NON_ROOT: Non-root candidate count (``≤ K_ROOT``). Active
            slice at non-root nodes.
        NUM_SIMULATIONS: Total sims per ``search_gpu`` call.
        N_POLICY_AT_ROOT: Legacy-vs-DMC root-sampling dispatch (see
            module docstring).

    Runtime ctor args:
        gamma, v_min, v_max — categorical decode + discounting.
        reward_min, reward_max — separate reward-head transformed range
            (paper ``dmc_state.yaml: reward_support: [-2, 2]`` decoded
            via ``h⁻¹`` → ≈ ±0.732). Decoupled from ``v_min/v_max``.
        max_action, min_std, std_magnification, soft_clamp, init_std —
            Squashed-Gaussian policy parameterization. Must match the
            training loss kernel.
        c_visit, c_scale — π-improvement σ(Q) scaling (paper defaults
            50.0 and 0.1).
    """

    comptime PRED_OUT: Int = 2 * Self.ACT_DIM + Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT_DIM
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime ENV_BLOCKS: Int = (Self.N_ENVS + TPB - 1) // TPB
    comptime K_PAD: Int = Self.K_ROOT  # physical slot width

    var state: EZV2GPUSampledMCTSState[
        Self.N_ENVS,
        Self.MAX_NODES,
        Self.ACT_DIM,
        Self.LATENT,
        Self.BINS,
        Self.K_ROOT,
        Self.K_NON_ROOT,
    ]
    """Per-env trees + scratch (candidate actions, log_prior, search
    paths, network I/O staging, ``chosen_actions``, ``root_visits``).
    See ``EZV2GPUSampledMCTSState`` for field-by-field layout."""

    var root_value_out: DeviceBuffer[dtype]
    """``[N_ENVS]`` scalar root value scattered from
    ``state.node_value[e * MAX_NODES]`` (decoded by
    ``gs_init_root_kernel``)."""

    var gamma: Float64
    var v_min: Float64
    var v_max: Float64
    var reward_min: Float64
    var reward_max: Float64
    var max_action: Float64
    var min_std: Float64
    var std_magnification: Float64
    var soft_clamp: Float64
    var init_std: Float64
    var c_visit: Float64
    var c_scale: Float64

    def __init__(
        out self,
        ctx: DeviceContext,
        gamma: Float64 = 0.997,
        v_min: Float64 = -10.0,
        v_max: Float64 = 10.0,
        reward_min: Float64 = -0.732_050_807_568_877_3,
        reward_max: Float64 = 0.732_050_807_568_877_3,
        max_action: Float64 = 1.0,
        min_std: Float64 = 0.1,
        std_magnification: Float64 = 3.0,
        soft_clamp: Float64 = 5.0,
        init_std: Float64 = 1.0,
        c_visit: Float64 = 50.0,
        c_scale: Float64 = 0.1,
    ) raises:
        if Self.K_ROOT < 1:
            raise Error("SampledGumbelGPUMCTS: K_ROOT must be >= 1")
        if Self.K_NON_ROOT < 1:
            raise Error("SampledGumbelGPUMCTS: K_NON_ROOT must be >= 1")
        if Self.K_NON_ROOT > Self.K_ROOT:
            raise Error("SampledGumbelGPUMCTS: K_NON_ROOT must be <= K_ROOT")
        if Self.N_POLICY_AT_ROOT < 1:
            raise Error("SampledGumbelGPUMCTS: N_POLICY_AT_ROOT must be >= 1")
        if Self.N_POLICY_AT_ROOT > Self.K_ROOT:
            raise Error(
                "SampledGumbelGPUMCTS: N_POLICY_AT_ROOT must be <= K_ROOT"
            )
        self.state = EZV2GPUSampledMCTSState[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.ACT_DIM,
            Self.LATENT,
            Self.BINS,
            Self.K_ROOT,
            Self.K_NON_ROOT,
        ](ctx)
        self.root_value_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.max_action = max_action
        self.min_std = min_std
        self.std_magnification = std_magnification
        self.soft_clamp = soft_clamp
        self.init_std = init_std
        self.c_visit = c_visit
        self.c_scale = c_scale

    def __init__(out self, *, deinit move: Self):
        self.state = move.state^
        self.root_value_out = move.root_value_out^
        self.gamma = move.gamma
        self.v_min = move.v_min
        self.v_max = move.v_max
        self.reward_min = move.reward_min
        self.reward_max = move.reward_max
        self.max_action = move.max_action
        self.min_std = move.min_std
        self.std_magnification = move.std_magnification
        self.soft_clamp = move.soft_clamp
        self.init_std = move.init_std
        self.c_visit = move.c_visit
        self.c_scale = move.c_scale

    # ══════════════════════════════════════════════════════════════════════
    # Views
    # ══════════════════════════════════════════════════════════════════════

    def chosen_actions_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS × ACT_DIM]`` continuous action pick per env."""
        return self.state.chosen_actions

    def root_visits_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS × K_ROOT]`` root visit counts per candidate slot."""
        return self.state.root_visits

    def root_value_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS]`` scalar root value (decoded at init_root)."""
        return self.root_value_out

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def search_gpu[
        REP: RepresentationGPU,
        DYN: DynamicsGPU,
        PRED: PredictionGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut rep: REP,
        mut dyn: DYN,
        mut pred: PRED,
        obs: Tile[dtype, Self.N_ENVS, REP.OBS_DIM],
        deterministic: Bool = False,
        rng_seed: UInt32 = UInt32(0),
    ) raises:
        """Run sampled-Gumbel MCTS for all envs.

        Pipeline (mirrors ``run_sampled_gumbel_search_gpu``):
          1. ``rep.encode_gpu`` → root hidden.
          2. ``pred.predict_gpu`` → root pred output (μ_raw, σ_raw, V bins).
          3. Scatter root hidden into ``state.hidden_states[e][0]``.
          4. ``gs_init_root_kernel`` — sample ``K_ROOT`` candidates +
             log_prior + decoded root value + Gumbel ranks.
          5. Sequential Halving: ``log2(K_ROOT)`` phases, each
             ``per_phase_budget // active_size`` sims per slot, then
             ``gs_halve_active_kernel``.
             Each sim: ``gs_select_kernel`` → ``dyn.step_gpu`` →
             ``gs_copy_pred_input_kernel`` → ``pred.predict_gpu`` →
             ``gs_expand_kernel`` → ``gs_backup_kernel``.
          6. Leftover sims on slot 0 of the size-1 survivor.
          7. ``gs_extract_kernel`` → chosen_actions + root_visits.
          8. ``gz_extract_root_value_kernel`` → root_value_out.

        ``deterministic=True`` returns the argmax-visit candidate at
        the root; otherwise visit-weighted soft pick using the same
        Philox stream.
        """

        # ── 0. Reset tree ────────────────────────────────────────────────
        self.state.zero_tree(ctx)

        # ── 1. Rep forward ───────────────────────────────────────────────
        var root_hidden_t = tile[Self.N_ENVS, REP.LATENT_DIM](self.state.root_hidden)
        rep.encode_gpu[Self.N_ENVS](ctx, obs, root_hidden_t)

        # ── 2. Pred forward at root ──────────────────────────────────────
        var pred_root_in = tile[Self.N_ENVS, PRED.LATENT_DIM](self.state.root_hidden)
        var pred_root_out = tile[Self.N_ENVS, PRED.PRED_OUT_DIM](self.state.pred_output)
        pred.predict_gpu[Self.N_ENVS](ctx, pred_root_in, pred_root_out)

        # ── 3. Scatter root hidden into per-env slot-0 ───────────────────
        var rh_flat = tile[Self.N_ENVS * Self.LATENT](self.state.root_hidden)
        var hs_flat = tile[Self.N_ENVS * Self.MAX_NODES * Self.LATENT](self.state.hidden_states)
        comptime run_scatter = gs_scatter_root_hidden_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.LATENT,
            dtype,
        ]
        ctx.enqueue_function[run_scatter](
            rh_flat,
            hs_flat,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 4. Init root ─────────────────────────────────────────────────
        var act_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD * Self.ACT_DIM](self.state.actions)
        var lp_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.log_prior)
        var nv_t = tile[Self.N_ENVS * Self.MAX_NODES](self.state.node_value)
        var ak_t = tile[Self.N_ENVS * Self.MAX_NODES](self.state.active_k)
        var nc_t = tile[Self.N_ENVS](self.state.node_count)
        var miq_t = tile[Self.N_ENVS](self.state.min_q)
        var mxq_t = tile[Self.N_ENVS](self.state.max_q)
        var rg_t = tile[Self.N_ENVS * Self.K_ROOT](self.state.root_gumbels)
        var ra_t = tile[Self.N_ENVS * Self.K_ROOT](self.state.root_active)
        var po_full_t = tile[Self.N_ENVS * Self.PRED_OUT](self.state.pred_output)

        comptime run_init = gs_init_root_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.ACT_DIM,
            Self.BINS,
            Self.K_ROOT,
            Self.K_PAD,
            Self.PRED_OUT,
            Self.N_POLICY_AT_ROOT,
            dtype,
        ]
        ctx.enqueue_function[run_init](
            act_t,
            lp_t,
            nv_t,
            ak_t,
            nc_t,
            miq_t,
            mxq_t,
            rg_t,
            ra_t,
            po_full_t,
            Scalar[dtype](self.v_min),
            Scalar[dtype](self.v_max),
            Scalar[dtype](self.max_action),
            Scalar[dtype](self.min_std),
            Scalar[dtype](self.std_magnification),
            Scalar[dtype](self.soft_clamp),
            Scalar[dtype](self.init_std),
            rng_seed,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 5. Sequential Halving simulation loop ────────────────────────
        var num_phases = _ilog2(_largest_power_of_two_le(Self.K_ROOT))
        if num_phases < 1:
            num_phases = 1
        var per_phase_budget = Self.NUM_SIMULATIONS // num_phases
        if per_phase_budget < 1:
            per_phase_budget = 1

        var sims_used = 0
        var active_size = _largest_power_of_two_le(Self.K_ROOT)
        if active_size < 1:
            active_size = 1
        for phase in range(num_phases):
            var per_action = per_phase_budget // active_size
            if per_action < 1:
                per_action = 1

            for _rep in range(per_action):
                for slot in range(active_size):
                    if sims_used >= Self.NUM_SIMULATIONS:
                        break
                    self._run_one_sim_gpu[REP, DYN, PRED](
                        ctx, dyn, pred, slot, rng_seed, UInt32(sims_used)
                    )
                    sims_used += 1

            # Halve the active set, except in the last phase.
            if phase + 1 < num_phases and active_size > 1:
                var keep = active_size // 2
                if keep < 1:
                    keep = 1
                comptime run_halve = gs_halve_active_kernel[
                    Self.N_ENVS,
                    Self.MAX_NODES,
                    Self.K_ROOT,
                    Self.K_PAD,
                    dtype,
                ]
                var vc_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.visit_count)
                var tv_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.total_value)
                var tvis_t = tile[Self.N_ENVS * Self.MAX_NODES](self.state.total_visits)
                ctx.enqueue_function[run_halve](
                    vc_t,
                    tv_t,
                    lp_t,
                    tvis_t,
                    nv_t,
                    miq_t,
                    mxq_t,
                    rg_t,
                    ra_t,
                    Scalar[DType.int32](active_size),
                    Scalar[DType.int32](keep),
                    Scalar[dtype](self.c_visit),
                    Scalar[dtype](self.c_scale),
                    grid_dim=(Self.ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                active_size = keep

        # Spend leftover sims on slot 0 of the size-1 survivor.
        while sims_used < Self.NUM_SIMULATIONS:
            self._run_one_sim_gpu[REP, DYN, PRED](
                ctx, dyn, pred, 0, rng_seed, UInt32(sims_used)
            )
            sims_used += 1

        # ── 6. Extract chosen action + root visit diagnostics ────────────
        var ca_t = tile[Self.N_ENVS * Self.ACT_DIM](self.state.chosen_actions)
        var rv_t = tile[Self.N_ENVS * Self.K_ROOT](self.state.root_visits)
        var vc_extract_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.visit_count)
        comptime run_extract = gs_extract_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.ACT_DIM,
            Self.K_ROOT,
            Self.K_PAD,
            dtype,
        ]
        ctx.enqueue_function[run_extract](
            vc_extract_t,
            act_t,
            ca_t,
            rv_t,
            Scalar[DType.uint8](1 if deterministic else 0),
            rng_seed,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 7. Extract root scalar value ─────────────────────────────────
        var root_value_t = tile[Self.N_ENVS](self.root_value_out)
        var vc_root_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.visit_count)
        var tv_root_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.total_value)
        comptime run_root_value = gz_extract_root_value_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.K_PAD,
            dtype,
        ]
        ctx.enqueue_function[run_root_value](
            vc_root_t,
            tv_root_t,
            nv_t,
            root_value_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Internal — one MCTS simulation across all envs.
    # ══════════════════════════════════════════════════════════════════════

    def _run_one_sim_gpu[
        REP: RepresentationGPU,
        DYN: DynamicsGPU,
        PRED: PredictionGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut dyn: DYN,
        mut pred: PRED,
        slot: Int,
        rng_seed: UInt32,
        sim_index: UInt32,
    ) raises:
        """One sim across all envs: select → dyn → pred → expand → backup."""

        var vc_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.visit_count)
        var tv_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.total_value)
        var lp_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.log_prior)
        var rw_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.reward)
        var ci_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD](self.state.child_idx)
        var act_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.K_PAD * Self.ACT_DIM](self.state.actions)
        var tvis_t = tile[Self.N_ENVS * Self.MAX_NODES](self.state.total_visits)
        var nv_t = tile[Self.N_ENVS * Self.MAX_NODES](self.state.node_value)
        var ak_t = tile[Self.N_ENVS * Self.MAX_NODES](self.state.active_k)
        var nc_t = tile[Self.N_ENVS](self.state.node_count)
        var miq_t = tile[Self.N_ENVS](self.state.min_q)
        var mxq_t = tile[Self.N_ENVS](self.state.max_q)
        var ra_t = tile[Self.N_ENVS * Self.K_ROOT](self.state.root_active)
        var hs_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.LATENT](self.state.hidden_states)
        var di_t = tile[Self.N_ENVS * Self.DYN_IN](self.state.dyn_input)
        var pp_t = tile[Self.N_ENVS](self.state.pending_parent)
        var pc_t = tile[Self.N_ENVS](self.state.pending_cand)
        var sp_t = tile[Self.N_ENVS * MAX_DEPTH](self.state.search_paths)
        var cp_t = tile[Self.N_ENVS * MAX_DEPTH](self.state.cand_paths)
        var pl_t = tile[Self.N_ENVS](self.state.path_lengths)

        # Selection.
        comptime run_select = gs_select_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.ACT_DIM,
            Self.K_ROOT,
            Self.K_PAD,
            Self.LATENT,
            Self.DYN_IN,
            dtype,
        ]
        ctx.enqueue_function[run_select](
            vc_t,
            tv_t,
            lp_t,
            ci_t,
            act_t,
            tvis_t,
            nv_t,
            ak_t,
            miq_t,
            mxq_t,
            ra_t,
            hs_t,
            di_t,
            pp_t,
            pc_t,
            sp_t,
            cp_t,
            pl_t,
            Scalar[DType.int32](slot),
            Scalar[dtype](self.c_visit),
            Scalar[dtype](self.c_scale),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Dynamics forward (via trait adapter).
        var dyn_in_b = tile[Self.N_ENVS, DYN.DYN_IN_DIM](self.state.dyn_input)
        var dyn_out_b = tile[Self.N_ENVS, DYN.DYN_OUT_DIM](self.state.dyn_output)
        dyn.step_gpu[Self.N_ENVS](ctx, dyn_in_b, dyn_out_b)

        # Copy LATENT prefix of dyn_output into pred_input.
        comptime run_copy = gs_copy_pred_input_kernel[
            Self.N_ENVS,
            Self.LATENT,
            Self.DYN_OUT,
            dtype,
        ]
        var pred_in_flat = tile[Self.N_ENVS * Self.LATENT](self.state.pred_input)
        var dyn_out_flat = tile[Self.N_ENVS * Self.DYN_OUT](self.state.dyn_output)
        ctx.enqueue_function[run_copy](
            pred_in_flat,
            dyn_out_flat,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Prediction forward.
        var pred_in_b = tile[Self.N_ENVS, PRED.LATENT_DIM](self.state.pred_input)
        var pred_out_b = tile[Self.N_ENVS, PRED.PRED_OUT_DIM](self.state.pred_output)
        pred.predict_gpu[Self.N_ENVS](ctx, pred_in_b, pred_out_b)

        # Expand.
        var lv_t = tile[Self.N_ENVS](self.state.leaf_values)
        var po_full_t = tile[Self.N_ENVS * Self.PRED_OUT](self.state.pred_output)
        comptime run_expand = gs_expand_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.ACT_DIM,
            Self.K_ROOT,
            Self.K_NON_ROOT,
            Self.K_PAD,
            Self.LATENT,
            Self.BINS,
            Self.PRED_OUT,
            Self.DYN_OUT,
            dtype,
        ]
        ctx.enqueue_function[run_expand](
            vc_t,
            tv_t,
            lp_t,
            rw_t,
            ci_t,
            act_t,
            tvis_t,
            nv_t,
            ak_t,
            nc_t,
            hs_t,
            pp_t,
            pc_t,
            dyn_out_flat,
            po_full_t,
            lv_t,
            Scalar[dtype](self.v_min),
            Scalar[dtype](self.v_max),
            Scalar[dtype](self.reward_min),
            Scalar[dtype](self.reward_max),
            Scalar[dtype](self.max_action),
            Scalar[dtype](self.min_std),
            Scalar[dtype](self.soft_clamp),
            Scalar[dtype](self.init_std),
            rng_seed,
            sim_index,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Backup.
        comptime run_backup = gs_backup_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.K_PAD,
            dtype,
        ]
        ctx.enqueue_function[run_backup](
            vc_t,
            tv_t,
            rw_t,
            tvis_t,
            miq_t,
            mxq_t,
            sp_t,
            cp_t,
            pl_t,
            lv_t,
            Scalar[dtype](self.gamma),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )
