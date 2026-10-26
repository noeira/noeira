"""Generic GPU MCTS orchestrator — ``GenericGPUMCTS``.

The agent-side counterpart of ``GenericCPUMCTS``. Owns a
``GPUMCTSState`` plus its action / policy / root-value output buffers,
and a ``search_gpu`` method that drives the full kernel-sequence-+-net-
forward pipeline for one MCTS search across ``N_ENVS`` environments in
parallel.

Phase 3b second slice. Replaces the ~600-line inline orchestration in
``muzero.mojo`` for the **MuZero batched single-player path**:

  root setup
    ┌── encode_gpu (REP) → hidden_states[node 0]
    ├── mcts_gpu_scale_hidden_kernel
    ├── predict_gpu (PRED, batch=N_ENVS) → pred_output
    ├── zero_tree
    └── gpu_mcts_init_root_kernel  ← adds Dirichlet noise if NOISE != NoNoise
  for each round (NUM_SIMS / BATCH_SIMS):
    ┌── gpu_mcts_batched_select_and_build_dyn_kernel
    ├── step_gpu (DYN, batch=N_ENVS·BATCH_SIMS)
    ├── mcts_gpu_extract_hidden_kernel
    ├── predict_gpu (PRED, batch=N_ENVS·BATCH_SIMS)
    └── gpu_mcts_batched_expand_backup_muzero_kernel
  readout
    ├── gpu_mcts_extract_actions_kernel
    └── gpu_mcts_extract_root_value_kernel

Other paths (AlphaZero `env.step` expansion, self-play negated backup,
temperature-weighted action sampling, legal-mask variants) are
deliberately not wired here yet — they're separate methods on the
struct, landing as their respective agents get rewired. The single-player
MuZero path is the most-used + easiest to validate against the existing
muzero CartPole training run.

The struct doesn't own a workspace buffer; the adapters do (they know
their network's ``WORKSPACE_SIZE_PER_SAMPLE``). This avoids the
orchestrator having to peek at trait-internal model dimensions.
"""

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT as dtype

from .model_traits_gpu import (
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
    EnvStepGPU,
)
from .strategies import (
    PUCTFormula,
    ExplorationNoise,
    PlayerMode,
)
from .mcts_gpu import (
    TPB,
    MAX_DEPTH,
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_extract_actions_kernel,
    gpu_mcts_extract_actions_masked_kernel,
    gpu_mcts_extract_actions_temp_kernel,
    gpu_mcts_extract_root_value_kernel,
    gpu_mcts_apply_legal_mask_kernel,
    gpu_mcts_apply_legal_mask_with_noise_kernel,
    gpu_mcts_batched_select_and_build_dyn_kernel,
    gpu_mcts_batched_expand_backup_muzero_kernel,
    gpu_mcts_batched_select_and_copy_kernel,
    gpu_mcts_batched_expand_backup_masked_kernel,
    gpu_mcts_copy_root_state_kernel,
    mcts_gpu_scale_hidden_kernel,
    mcts_gpu_scatter_root_hidden_kernel,
    mcts_gpu_extract_hidden_kernel,
)
from noeira.core.tiles import Tile, tile


struct GenericGPUMCTS[
    N_ENVS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_NODES: Int,
    NUM_SIMULATIONS: Int,
    BATCH_SIMS: Int,
    PUCT: PUCTFormula,
    NOISE: ExplorationNoise,
    PLAYER: PlayerMode,
    STATE_SIZE: Int = 0,
    VIRTUAL_LOSS: Int = 3,
    # ⚠ BATCH_SIMS > 1 is KNOWN-BIASED and gated off by default. The batched
    # rounds select every leaf against a FROZEN tree (the CPU search expands
    # during selection, so later sims in a round descend through earlier sims'
    # children); duplicate (parent, action) selections re-expand the same edge,
    # leak the orphaned node, and double-count its value. Measured: MuZero
    # CartPole root values systematically +1.2-1.7 vs CPU + argmax flips
    # (tests/deep_agents/test_mz_search_gpu_cpu_parity.mojo); AZ Connect Four
    # visit-policy targets distorted up to 2× per action
    # (test_az_search_gpu_batched_bias.mojo). BATCH_SIMS=1 is bit-near-exact
    # vs the converged CPU search. For batched GPU search use the Gumbel
    # orchestrators (sequential halving — no virtual loss, immune by
    # construction). Set UNSAFE_BATCHED=True only for diagnostics.
    UNSAFE_BATCHED: Bool = False,
](Deinitable, Movable):
    """GPU MCTS orchestrator for the MuZero batched single-player path.

    Comptime params:
        N_ENVS: Number of parallel envs / trees.
        ACT: Discrete action count.
        LATENT: Hidden state dim.
        BINS: Categorical reward / value bins (= ``NUM_BINS``).
        MAX_NODES: Per-tree node arena size.
        NUM_SIMULATIONS: Sims per ``search_gpu`` call. Must be a
            multiple of ``BATCH_SIMS``.
        BATCH_SIMS: Leaves selected per round (virtual-loss batching).
        PUCT: PUCT formula trait — supplies ``C_BASE`` / ``C_INIT`` to
            the selection kernel. ``MuZeroPUCT`` (log-based) is the
            default for MuZero.
        NOISE: Root exploration noise. ``NOISE_FRACTION`` is forwarded
            to ``gpu_mcts_init_root_kernel``; ``NoNoise`` ⇒ 0.
        PLAYER: ``SinglePlayer`` (MuZero, single-player envs) or
            ``SelfPlay`` (board games — uses negated backup +
            ``search_gpu_selfplay``).
        STATE_SIZE: ``0`` for MuZero / single-player paths (default).
            ``> 0`` for AlphaZero ``search_gpu_alphazero`` — the tree
            allocates ``N_ENVS × MAX_NODES × STATE_SIZE`` cells of
            ``game_states`` plus ``N_ENVS × BATCH_SIMS × STATE_SIZE``
            staging for env-step expansion.

    Runtime ctor args: ``gamma``, ``v_min``, ``v_max``.

    The orchestrator does **not** own a workspace buffer — adapters
    own theirs because they know the network's
    ``WORKSPACE_SIZE_PER_SAMPLE``.
    """

    comptime PRED_OUT: Int = Self.ACT + Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime MCTS_TOTAL: Int = Self.N_ENVS * Self.BATCH_SIMS
    comptime MCTS_ROUNDS: Int = Self.NUM_SIMULATIONS // Self.BATCH_SIMS
    comptime ENV_BLOCKS: Int = (Self.N_ENVS + TPB - 1) // TPB

    var state: GPUMCTSState[
        Self.N_ENVS,
        Self.MAX_NODES,
        Self.ACT,
        Self.LATENT,
        Self.BINS,
        Self.STATE_SIZE,
        Self.BATCH_SIMS,
    ]
    """Tree node arena + per-batch scratch buffers, all device-resident."""

    var actions_out: DeviceBuffer[dtype]
    """``[N_ENVS]`` argmax action per env (output of
    ``gpu_mcts_extract_actions_kernel``)."""

    var policies_out: DeviceBuffer[dtype]
    """``[N_ENVS × ACT]`` visit-count policy per env."""

    var root_value_out: DeviceBuffer[dtype]
    """``[N_ENVS]`` visit-weighted root Q per env."""

    var az_exp_dones: DeviceBuffer[dtype]
    """``[MCTS_TOTAL]`` env.step done flag per pending expansion. Used
    by ``search_gpu_alphazero`` only; sized 1 when ``STATE_SIZE == 0``
    to avoid wasted memory on MuZero-style configs."""

    var az_exp_terminated: DeviceBuffer[dtype]
    """``[MCTS_TOTAL]`` env.step terminated flag — AlphaZero path."""

    var az_exp_obs: DeviceBuffer[dtype]
    """``[MCTS_TOTAL × LATENT]`` env.step obs output (AlphaZero
    adapters treat ``LATENT_DIM`` as ``OBS_DIM``)."""

    var gamma: Float64
    var v_min: Float64
    var v_max: Float64

    def __init__(
        out self,
        ctx: DeviceContext,
        gamma: Float64 = 0.997,
        v_min: Float64 = -10.0,
        v_max: Float64 = 10.0,
    ) raises:
        # NUM_SIMULATIONS must be a clean multiple of BATCH_SIMS.
        # Enforced at construction so a misconfigured agent fails loudly.
        if Self.NUM_SIMULATIONS % Self.BATCH_SIMS != 0:
            raise Error(
                "GenericGPUMCTS: NUM_SIMULATIONS must be divisible by"
                " BATCH_SIMS"
            )
        if Self.MCTS_ROUNDS < 1:
            raise Error(
                "GenericGPUMCTS: NUM_SIMULATIONS >= BATCH_SIMS required"
            )

        self.state = GPUMCTSState[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.ACT,
            Self.LATENT,
            Self.BINS,
            Self.STATE_SIZE,
            Self.BATCH_SIMS,
        ](ctx)
        self.actions_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.policies_out = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )
        self.root_value_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        # AlphaZero env.step scratch — allocate full size only when
        # STATE_SIZE > 0, otherwise placeholder of length 1.
        comptime AZ_SCALAR = (Self.MCTS_TOTAL if Self.STATE_SIZE > 0 else 1)
        comptime AZ_OBS = (
            Self.MCTS_TOTAL * Self.LATENT if Self.STATE_SIZE > 0 else 1
        )
        self.az_exp_dones = ctx.enqueue_create_buffer[dtype](AZ_SCALAR)
        self.az_exp_terminated = ctx.enqueue_create_buffer[dtype](AZ_SCALAR)
        self.az_exp_obs = ctx.enqueue_create_buffer[dtype](AZ_OBS)
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max

    def __init__(out self, *, deinit move: Self):
        self.state = move.state^
        self.actions_out = move.actions_out^
        self.policies_out = move.policies_out^
        self.root_value_out = move.root_value_out^
        self.az_exp_dones = move.az_exp_dones^
        self.az_exp_terminated = move.az_exp_terminated^
        self.az_exp_obs = move.az_exp_obs^
        self.gamma = move.gamma
        self.v_min = move.v_min
        self.v_max = move.v_max

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
        rng_seed: UInt32 = UInt32(0),
    ) raises:
        """Run ``NUM_SIMULATIONS`` MCTS sims for every env in parallel.

        Writes results into ``self.actions_out``, ``self.policies_out``,
        ``self.root_value_out``. Caller reads via the corresponding
        ``…_view()`` methods (or directly off the device buffers).

        Single-player MuZero-style: gamma-discounted backup, categorical
        reward + value decode, MinMax Q-normalization, virtual-loss
        batched leaf selection. Root exploration noise is gated on the
        ``NOISE`` trait — ``NoNoise`` ⇒ no noise added.
        """
        comptime assert Self.BATCH_SIMS == 1 or Self.UNSAFE_BATCHED, (
            "BATCH_SIMS > 1 is known-biased on this orchestrator (frozen-tree"
            " duplicate re-expansion; see the UNSAFE_BATCHED param doc). Use"
            " BATCH_SIMS=1 or the Gumbel orchestrators."
        )

        # ── 1. Root encode → hidden_states[node 0 for each env] ──────
        # Use the trait's ``LATENT_DIM`` for the view so the
        # method-template binding sees one comptime expression, not
        # ``Self.LATENT`` vs ``REP.LATENT_DIM`` (same value, distinct
        # expressions in Mojo's type system).
        var hidden_root = tile[Self.N_ENVS, REP.LATENT_DIM](self.state.hidden_states)
        rep.encode_gpu[Self.N_ENVS](ctx, obs, hidden_root)

        # 1a. Post-encode MinMax scaling. MuZero networks bake this into
        # the model (``MinMaxNorm`` tail layer) so the kernel is a no-op
        # on already-normalized output; for stub / Linear-only test
        # networks it does real work. Always-on is the safer default —
        # the kernel is idempotent on [0, 1] values.
        comptime BATCH_BLOCKS = (Self.N_ENVS + TPB - 1) // TPB
        var hidden_root_flat = tile[Self.N_ENVS * Self.LATENT](self.state.hidden_states)
        comptime run_scale_root = mcts_gpu_scale_hidden_kernel[
            Self.N_ENVS, Self.LATENT, dtype
        ]
        ctx.enqueue_function[run_scale_root](
            hidden_root_flat,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # 1b. Scatter dense root hidden into strided tree slots. The
        # rep forward + post-scale write to the dense ``[N_ENVS, LATENT]``
        # prefix of ``hidden_states`` (env e at offset e*LATENT). But
        # every MCTS kernel that reads env e's node-0 hidden uses the
        # strided offset e*MAX_NODES*LATENT. For e=0 the layouts collide
        # (offset 0), so env 0 gets the correct hidden; for e≥1 the
        # strided slots are uninitialized (typically zero) and the
        # search runs with wrong, identical-across-envs hidden states.
        # See ``tests/deep_agents/test_muzero_gpu_mcts_per_env_isolation.mojo``
        # and ``test_muzero_network_row_symmetry.mojo`` for the bisection
        # that surfaced this.
        var hidden_full_flat = tile[Self.N_ENVS * Self.MAX_NODES * Self.LATENT](self.state.hidden_states)
        comptime SCATTER_TOTAL = Self.N_ENVS * Self.LATENT
        comptime SCATTER_BLOCKS = (SCATTER_TOTAL + TPB - 1) // TPB
        comptime run_scatter_root = mcts_gpu_scatter_root_hidden_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.LATENT, dtype
        ]
        ctx.enqueue_function[run_scatter_root](
            hidden_full_flat,
            grid_dim=(SCATTER_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 2. Root predict → policy logits + value bins ─────────────
        var pred_root_in = tile[Self.N_ENVS, PRED.LATENT_DIM](self.state.hidden_states)
        var pred_root_out = tile[Self.N_ENVS, PRED.PRED_OUT_DIM](self.state.pred_output)
        pred.predict_gpu[Self.N_ENVS](ctx, pred_root_in, pred_root_out)

        # ── 3. Zero tree + init root from pred output ────────────────
        self.state.zero_tree(ctx)

        var vc_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.visit_count)
        var tv_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.total_value)
        var pr_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.prior)
        var rw_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.reward)
        var ci_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.child_idx)
        var tvis_t = tile[Self.N_ENVS * Self.MAX_NODES](self.state.total_visits)
        var nc_t = tile[Self.N_ENVS](self.state.node_count)
        var po_root_t = tile[Self.N_ENVS * Self.PRED_OUT](self.state.pred_output)
        var miq_t = tile[Self.N_ENVS](self.state.min_q)
        var mxq_t = tile[Self.N_ENVS](self.state.max_q)

        comptime run_init = gpu_mcts_init_root_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.ACT,
            Self.LATENT,
            Self.PRED_OUT,
            dtype,
        ]
        # NOISE_FRACTION = 0 for NoNoise; the kernel internally still
        # generates Dirichlet samples and blends them in, but with
        # fraction 0 the original softmax prior is preserved.
        var noise_fraction = Scalar[dtype](Self.NOISE.NOISE_FRACTION)
        ctx.enqueue_function[run_init](
            vc_t,
            tv_t,
            pr_t,
            rw_t,
            ci_t,
            tvis_t,
            nc_t,
            po_root_t,
            miq_t,
            mxq_t,
            noise_fraction,
            rng_seed,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 4. MCTS simulation rounds ────────────────────────────────
        var hs_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.LATENT](self.state.hidden_states)
        var b_pp = tile[Self.MCTS_TOTAL](self.state.pending_parent)
        var b_pa = tile[Self.MCTS_TOTAL](self.state.pending_action)
        var b_sp = tile[Self.MCTS_TOTAL * MAX_DEPTH](self.state.search_paths)
        var b_ap = tile[Self.MCTS_TOTAL * MAX_DEPTH](self.state.action_paths)
        var b_pl = tile[Self.MCTS_TOTAL](self.state.path_lengths)
        var b_di = tile[Self.MCTS_TOTAL * Self.DYN_IN](self.state.dyn_input)

        for _round in range(Self.MCTS_ROUNDS):
            # 4a. Batched select + build dynamics input
            comptime run_sel_dyn = gpu_mcts_batched_select_and_build_dyn_kernel[
                Self.N_ENVS,
                Self.MAX_NODES,
                Self.ACT,
                Self.BATCH_SIMS,
                Self.LATENT,
                Self.DYN_IN,
                dtype,
                VIRTUAL_LOSS_VAL=Self.VIRTUAL_LOSS,
            ]
            ctx.enqueue_function[run_sel_dyn](
                vc_t,
                tv_t,
                pr_t,
                ci_t,
                tvis_t,
                nc_t,
                miq_t,
                mxq_t,
                hs_t,
                b_di,
                b_pp,
                b_pa,
                b_sp,
                b_ap,
                b_pl,
                Scalar[dtype](Self.PUCT.C_BASE),
                Scalar[dtype](Self.PUCT.C_INIT),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # 4b. Batched dynamics forward.
            # Build the LayoutTensors against the trait's comptime
            # dimensions so the method-template binding doesn't see
            # ``Self.DYN_IN`` vs ``DYN.DYN_IN_DIM`` as distinct
            # expressions (Mojo's type system treats unfolded comptime
            # exprs structurally — equal values aren't enough).
            var dyn_in_b = tile[Self.MCTS_TOTAL, DYN.DYN_IN_DIM](self.state.dyn_input)
            var dyn_out_b = tile[Self.MCTS_TOTAL, DYN.DYN_OUT_DIM](self.state.dyn_output)
            dyn.step_gpu[Self.MCTS_TOTAL](ctx, dyn_in_b, dyn_out_b)

            # 4c. Extract hidden slice → pred input
            var pred_in_b = tile[Self.MCTS_TOTAL * Self.LATENT](self.state.pred_input)
            var dyn_out_b_flat = tile[Self.MCTS_TOTAL * Self.DYN_OUT](self.state.dyn_output)
            comptime EXTR_TOTAL = Self.MCTS_TOTAL * Self.LATENT
            comptime EXTR_BLK = (EXTR_TOTAL + TPB - 1) // TPB
            comptime run_extr = mcts_gpu_extract_hidden_kernel[
                Self.MCTS_TOTAL, Self.LATENT, Self.DYN_OUT, dtype
            ]
            ctx.enqueue_function[run_extr](
                pred_in_b,
                dyn_out_b_flat,
                grid_dim=(EXTR_BLK,),
                block_dim=(TPB,),
            )

            # 4d. Batched prediction forward — same trait-dim trick as
            # the dynamics call above.
            var pred_in_net = tile[Self.MCTS_TOTAL, PRED.LATENT_DIM](self.state.pred_input)
            var pred_out_net = tile[Self.MCTS_TOTAL, PRED.PRED_OUT_DIM](self.state.pred_output)
            pred.predict_gpu[Self.MCTS_TOTAL](ctx, pred_in_net, pred_out_net)

            # 4e. Batched expand + backup + remove virtual loss
            var b_do = tile[Self.MCTS_TOTAL * Self.DYN_OUT](self.state.dyn_output)
            var b_po = tile[Self.MCTS_TOTAL * Self.PRED_OUT](self.state.pred_output)
            comptime run_exp_bk = gpu_mcts_batched_expand_backup_muzero_kernel[
                Self.N_ENVS,
                Self.MAX_NODES,
                Self.ACT,
                Self.BATCH_SIMS,
                Self.LATENT,
                Self.PRED_OUT,
                Self.DYN_OUT,
                dtype,
                VIRTUAL_LOSS_VAL=Self.VIRTUAL_LOSS,
            ]
            ctx.enqueue_function[run_exp_bk](
                vc_t,
                tv_t,
                pr_t,
                rw_t,
                ci_t,
                tvis_t,
                nc_t,
                miq_t,
                mxq_t,
                hs_t,
                b_pp,
                b_pa,
                b_do,
                b_po,
                b_sp,
                b_ap,
                b_pl,
                Scalar[dtype](self.v_min),
                Scalar[dtype](self.v_max),
                Scalar[dtype](self.gamma),
                Scalar[DType.bool](Self.PLAYER.NEGATE_BACKUP),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── 5. Extract actions + visit-count policy ──────────────────
        var act_out_t = tile[Self.N_ENVS](self.actions_out)
        var pol_out_t = tile[Self.N_ENVS * Self.ACT](self.policies_out)
        comptime run_act = gpu_mcts_extract_actions_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_act](
            vc_t,
            act_out_t,
            pol_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 6. Extract root value (visit-weighted Q at root) ─────────
        var rv_out_t = tile[Self.N_ENVS](self.root_value_out)
        comptime run_rv = gpu_mcts_extract_root_value_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_rv](
            vc_t,
            tv_t,
            rv_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Self-play variant — legal masks + negated backup
    # ══════════════════════════════════════════════════════════════════════

    def search_gpu_selfplay[
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
        legal_masks: Tile[dtype, Self.N_ENVS * Self.ACT],
        rng_seed: UInt32 = UInt32(0),
    ) raises:
        """Self-play variant of ``search_gpu`` for board games.

        Differences vs ``search_gpu``:
          * After ``init_root``, applies the legal mask to the root prior
            (zeroes illegal actions, renormalizes). With ``DirichletNoise``
            the noisy variant is used so the noise budget lands only on
            legal actions; ``init_root`` is then called with fraction=0
            to avoid double-noise.
          * Backup is negated (``PLAYER.NEGATE_BACKUP=True`` for
            ``SelfPlay``) — already wired in the kernel call.
          * Action extraction uses ``gpu_mcts_extract_actions_masked_kernel``
            so the argmax + visit-count policy only considers legal
            actions.

        The MCTS simulation rounds themselves are identical to
        ``search_gpu``. AlphaZero ``env.step`` expansion is a separate
        follow-up path (``search_gpu_alphazero``) because it replaces
        the dynamics network call.
        """
        comptime assert Self.BATCH_SIMS == 1 or Self.UNSAFE_BATCHED, (
            "BATCH_SIMS > 1 is known-biased on this orchestrator (frozen-tree"
            " duplicate re-expansion; see the UNSAFE_BATCHED param doc). Use"
            " BATCH_SIMS=1 or the Gumbel orchestrators."
        )

        # ── 1. Root encode → hidden[node 0] ──────────────────────────
        var hidden_root = tile[Self.N_ENVS, REP.LATENT_DIM](self.state.hidden_states)
        rep.encode_gpu[Self.N_ENVS](ctx, obs, hidden_root)

        # 1a. Post-encode MinMax scaling.
        comptime BATCH_BLOCKS = (Self.N_ENVS + TPB - 1) // TPB
        var hidden_root_flat = tile[Self.N_ENVS * Self.LATENT](self.state.hidden_states)
        comptime run_scale_root = mcts_gpu_scale_hidden_kernel[
            Self.N_ENVS, Self.LATENT, dtype
        ]
        ctx.enqueue_function[run_scale_root](
            hidden_root_flat,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # 1b. Scatter dense root hidden into strided tree slots. Same
        # fix as ``search_gpu`` — see the comment block there and
        # ``tests/deep_agents/test_muzero_gpu_mcts_per_env_isolation.mojo``
        # for the bisection that surfaced this on the single-player
        # path. Self-play has the identical dense-vs-strided layout
        # mismatch (rep forward writes ``[N_ENVS, LATENT]`` but MCTS
        # kernels read ``[N_ENVS * MAX_NODES * LATENT]``), so the same
        # scatter is required for envs 1..N-1 to see their own root
        # hidden state.
        var hidden_full_flat = tile[Self.N_ENVS * Self.MAX_NODES * Self.LATENT](self.state.hidden_states)
        comptime SCATTER_TOTAL = Self.N_ENVS * Self.LATENT
        comptime SCATTER_BLOCKS = (SCATTER_TOTAL + TPB - 1) // TPB
        comptime run_scatter_root = mcts_gpu_scatter_root_hidden_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.LATENT, dtype
        ]
        ctx.enqueue_function[run_scatter_root](
            hidden_full_flat,
            grid_dim=(SCATTER_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 2. Root predict ──────────────────────────────────────────
        var pred_root_in = tile[Self.N_ENVS, PRED.LATENT_DIM](self.state.hidden_states)
        var pred_root_out = tile[Self.N_ENVS, PRED.PRED_OUT_DIM](self.state.pred_output)
        pred.predict_gpu[Self.N_ENVS](ctx, pred_root_in, pred_root_out)

        # ── 3. Zero tree + init root with noise=0 (legal-mask path
        #       owns noise injection so it can keep the budget on legal
        #       actions). ────────────────────────────────────────────
        self.state.zero_tree(ctx)

        var vc_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.visit_count)
        var tv_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.total_value)
        var pr_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.prior)
        var rw_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.reward)
        var ci_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.child_idx)
        var tvis_t = tile[Self.N_ENVS * Self.MAX_NODES](self.state.total_visits)
        var nc_t = tile[Self.N_ENVS](self.state.node_count)
        var po_root_t = tile[Self.N_ENVS * Self.PRED_OUT](self.state.pred_output)
        var miq_t = tile[Self.N_ENVS](self.state.min_q)
        var mxq_t = tile[Self.N_ENVS](self.state.max_q)

        comptime run_init = gpu_mcts_init_root_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.ACT,
            Self.LATENT,
            Self.PRED_OUT,
            dtype,
        ]
        # Always init_root with fraction=0 here — the legal-mask kernel
        # injects noise restricted to legal actions if NOISE is on.
        ctx.enqueue_function[run_init](
            vc_t,
            tv_t,
            pr_t,
            rw_t,
            ci_t,
            tvis_t,
            nc_t,
            po_root_t,
            miq_t,
            mxq_t,
            Scalar[dtype](0.0),
            rng_seed,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 3a. Apply legal mask (+ Dirichlet noise on legal-only if
        #         NOISE is enabled). ──────────────────────────────────
        comptime if Self.NOISE.NOISE_TYPE == 2:
            comptime run_mask = gpu_mcts_apply_legal_mask_kernel[
                Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
            ]
            ctx.enqueue_function[run_mask](
                pr_t,
                legal_masks,
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            comptime run_mask_noise = (
                gpu_mcts_apply_legal_mask_with_noise_kernel[
                    Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
                ]
            )
            ctx.enqueue_function[run_mask_noise](
                pr_t,
                legal_masks,
                Scalar[dtype](Self.NOISE.NOISE_FRACTION),
                rng_seed,
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── 4. Simulation rounds (identical to single-player path) ──
        var hs_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.LATENT](self.state.hidden_states)
        var b_pp = tile[Self.MCTS_TOTAL](self.state.pending_parent)
        var b_pa = tile[Self.MCTS_TOTAL](self.state.pending_action)
        var b_sp = tile[Self.MCTS_TOTAL * MAX_DEPTH](self.state.search_paths)
        var b_ap = tile[Self.MCTS_TOTAL * MAX_DEPTH](self.state.action_paths)
        var b_pl = tile[Self.MCTS_TOTAL](self.state.path_lengths)
        var b_di = tile[Self.MCTS_TOTAL * Self.DYN_IN](self.state.dyn_input)

        for _round in range(Self.MCTS_ROUNDS):
            comptime run_sel_dyn = gpu_mcts_batched_select_and_build_dyn_kernel[
                Self.N_ENVS,
                Self.MAX_NODES,
                Self.ACT,
                Self.BATCH_SIMS,
                Self.LATENT,
                Self.DYN_IN,
                dtype,
                VIRTUAL_LOSS_VAL=Self.VIRTUAL_LOSS,
            ]
            ctx.enqueue_function[run_sel_dyn](
                vc_t,
                tv_t,
                pr_t,
                ci_t,
                tvis_t,
                nc_t,
                miq_t,
                mxq_t,
                hs_t,
                b_di,
                b_pp,
                b_pa,
                b_sp,
                b_ap,
                b_pl,
                Scalar[dtype](Self.PUCT.C_BASE),
                Scalar[dtype](Self.PUCT.C_INIT),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            var dyn_in_b = tile[Self.MCTS_TOTAL, DYN.DYN_IN_DIM](self.state.dyn_input)
            var dyn_out_b = tile[Self.MCTS_TOTAL, DYN.DYN_OUT_DIM](self.state.dyn_output)
            dyn.step_gpu[Self.MCTS_TOTAL](ctx, dyn_in_b, dyn_out_b)

            var pred_in_b = tile[Self.MCTS_TOTAL * Self.LATENT](self.state.pred_input)
            var dyn_out_b_flat = tile[Self.MCTS_TOTAL * Self.DYN_OUT](self.state.dyn_output)
            comptime EXTR_TOTAL = Self.MCTS_TOTAL * Self.LATENT
            comptime EXTR_BLK = (EXTR_TOTAL + TPB - 1) // TPB
            comptime run_extr = mcts_gpu_extract_hidden_kernel[
                Self.MCTS_TOTAL, Self.LATENT, Self.DYN_OUT, dtype
            ]
            ctx.enqueue_function[run_extr](
                pred_in_b,
                dyn_out_b_flat,
                grid_dim=(EXTR_BLK,),
                block_dim=(TPB,),
            )

            var pred_in_net = tile[Self.MCTS_TOTAL, PRED.LATENT_DIM](self.state.pred_input)
            var pred_out_net = tile[Self.MCTS_TOTAL, PRED.PRED_OUT_DIM](self.state.pred_output)
            pred.predict_gpu[Self.MCTS_TOTAL](ctx, pred_in_net, pred_out_net)

            var b_do = tile[Self.MCTS_TOTAL * Self.DYN_OUT](self.state.dyn_output)
            var b_po = tile[Self.MCTS_TOTAL * Self.PRED_OUT](self.state.pred_output)
            comptime run_exp_bk = gpu_mcts_batched_expand_backup_muzero_kernel[
                Self.N_ENVS,
                Self.MAX_NODES,
                Self.ACT,
                Self.BATCH_SIMS,
                Self.LATENT,
                Self.PRED_OUT,
                Self.DYN_OUT,
                dtype,
                VIRTUAL_LOSS_VAL=Self.VIRTUAL_LOSS,
            ]
            ctx.enqueue_function[run_exp_bk](
                vc_t,
                tv_t,
                pr_t,
                rw_t,
                ci_t,
                tvis_t,
                nc_t,
                miq_t,
                mxq_t,
                hs_t,
                b_pp,
                b_pa,
                b_do,
                b_po,
                b_sp,
                b_ap,
                b_pl,
                Scalar[dtype](self.v_min),
                Scalar[dtype](self.v_max),
                Scalar[dtype](self.gamma),
                Scalar[DType.bool](Self.PLAYER.NEGATE_BACKUP),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── 5. Masked action extraction ──────────────────────────────
        var act_out_t = tile[Self.N_ENVS](self.actions_out)
        var pol_out_t = tile[Self.N_ENVS * Self.ACT](self.policies_out)
        comptime run_act_masked = gpu_mcts_extract_actions_masked_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_act_masked](
            vc_t,
            legal_masks,
            act_out_t,
            pol_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 6. Root value (same kernel, no masking needed since the
        #       backup already encoded negation). ─────────────────────
        var rv_out_t = tile[Self.N_ENVS](self.root_value_out)
        comptime run_rv = gpu_mcts_extract_root_value_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_rv](
            vc_t,
            tv_t,
            rv_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Temperature-weighted action sampling
    # ══════════════════════════════════════════════════════════════════════

    def extract_actions_temp[
        TEMP_THRESHOLD: Int = 0,
    ](
        mut self,
        ctx: DeviceContext,
        ep_steps: Tile[dtype, Self.N_ENVS],
        legal_masks: Tile[dtype, Self.N_ENVS * Self.ACT],
        rng_seed: UInt32 = UInt32(0),
        temp_min: Float64 = 0.0,
    ) raises:
        """Overwrite ``actions_out`` / ``policies_out`` from the tree's
        visit counts using the AlphaZero-style temperature schedule.

        Composable: call this **after** ``search_gpu`` or
        ``search_gpu_selfplay`` — the tree's ``visit_count`` is still
        live, so the temperature kernel re-derives actions/policies
        without rerunning MCTS. The previous argmax/masked-argmax
        output written by ``search_gpu*`` is silently replaced.

        Temperature schedule (per env):
          * For the first ``TEMP_THRESHOLD`` moves of the episode
            (``ep_steps[e] < TEMP_THRESHOLD``) temperature is 1.0 — the
            policy is proportional to visit counts and the action is
            sampled from it.
          * Afterwards temperature is ``temp_min``:
            - ``temp_min = 0`` → greedy (argmax + one-hot policy).
            - ``temp_min > 0`` → sample from ``N^(1/τ)``.

        ``TEMP_THRESHOLD`` is comptime because Mojo's
        ``enqueue_function`` rejects runtime ``Int`` kernel args (see
        ``feedback_gpu_scalar_args`` memory + legacy muzero pattern at
        ``muzero.mojo:4276``).

        ``legal_masks`` is required even when every action is legal —
        callers can pass an all-ones buffer in that case.
        """
        var vc_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.visit_count)
        var act_out_t = tile[Self.N_ENVS](self.actions_out)
        var pol_out_t = tile[Self.N_ENVS * Self.ACT](self.policies_out)

        comptime run_temp = gpu_mcts_extract_actions_temp_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_temp](
            vc_t,
            legal_masks,
            ep_steps,
            act_out_t,
            pol_out_t,
            Int32(TEMP_THRESHOLD),
            rng_seed,
            Scalar[dtype](temp_min),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # AlphaZero variant — env.step expansion (no dynamics network)
    # ══════════════════════════════════════════════════════════════════════

    def search_gpu_alphazero[
        PRED: PredictionGPU,
        ENV: EnvStepGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut pred: PRED,
        mut env: ENV,
        root_obs: Tile[dtype, Self.N_ENVS, PRED.LATENT_DIM],
        root_states: DeviceBuffer[dtype],
        root_legal_masks: Tile[dtype, Self.N_ENVS * Self.ACT],
        rng_seed: UInt64 = UInt64(0),
    ) raises:
        """AlphaZero MCTS path — expansion via true game rules.

        Replaces the dynamics-network half of ``search_gpu_selfplay``
        with ``ENV.step_gpu[B]`` calls. Tree nodes carry full
        ``game_state`` payloads (size ``Self.STATE_SIZE``) instead of
        learned hidden states; the prediction net is fed obs directly
        (so ``PRED.LATENT_DIM == OBS_DIM`` for AlphaZero adapters).

        Construction requires ``STATE_SIZE > 0`` — the orchestrator
        raises at runtime when this is misconfigured.

        Pipeline (per round):

            * select_and_copy: PUCT pick + virtual loss + copy parent
              game state to expansion staging buffer.
            * env.step_gpu (B = N_ENVS · BATCH_SIMS): play the chosen
              action in the staged game state; writes reward, done,
              terminated, obs, legal_mask for each expansion in the
              same buffer.
            * pred.predict_gpu (B = N_ENVS · BATCH_SIMS): forward on
              the post-step obs to get child priors + value.
            * expand_backup_masked: store child game state, set
              masked priors, backup leaf value through the path.

        Backup is negated (``NEGATE_BACKUP=True``) and value-squashed
        (``VALUE_SQUASH=True``) per the legacy AlphaZero kernel
        configuration at ``alphazero.mojo:2784``.

        Args:
            ctx: GPU device context.
            pred: Prediction net adapter (obs → policy + value).
            env: Env-step adapter (game_state, action → next_state,
                reward, done, terminated, obs, legal_mask).
            root_obs: ``[N_ENVS × OBS_DIM]`` observation at the root
                of each env's tree.
            root_states: ``[N_ENVS × STATE_SIZE]`` initial game state
                per env. Copied into ``state.game_states[node 0]``.
            root_legal_masks: ``[N_ENVS × ACT]`` legal-action mask at
                the root, applied to the root prior.
            rng_seed: Forwarded to env.step_gpu (per-round seed
                derived from ``rng_seed + round_index`` if the env
                needs stochasticity).
        """
        comptime assert Self.BATCH_SIMS == 1 or Self.UNSAFE_BATCHED, (
            "BATCH_SIMS > 1 is known-biased on this orchestrator (frozen-tree"
            " duplicate re-expansion; see the UNSAFE_BATCHED param doc). Use"
            " BATCH_SIMS=1 or the Gumbel orchestrators."
        )

        if Self.STATE_SIZE <= 0:
            raise Error(
                "GenericGPUMCTS.search_gpu_alphazero: STATE_SIZE must"
                " be > 0 — construct the orchestrator with"
                " STATE_SIZE = <env state dim>"
            )

        # ── 1. Root predict on obs ───────────────────────────────────
        var pred_root_in = root_obs
        var pred_root_out = tile[Self.N_ENVS, PRED.PRED_OUT_DIM](self.state.pred_output)
        pred.predict_gpu[Self.N_ENVS](ctx, pred_root_in, pred_root_out)

        # ── 2. Zero tree + init root with noise=0 ────────────────────
        self.state.zero_tree(ctx)

        var vc_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.visit_count)
        var tv_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.total_value)
        var pr_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.prior)
        var rw_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.reward)
        var ci_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.ACT](self.state.child_idx)
        var tvis_t = tile[Self.N_ENVS * Self.MAX_NODES](self.state.total_visits)
        var nc_t = tile[Self.N_ENVS](self.state.node_count)
        var po_root_t = tile[Self.N_ENVS * Self.PRED_OUT](self.state.pred_output)
        var miq_t = tile[Self.N_ENVS](self.state.min_q)
        var mxq_t = tile[Self.N_ENVS](self.state.max_q)

        # Note: init_root_kernel expects LATENT as a comptime arg even
        # though the AlphaZero path never touches hidden_states. Pass
        # ``Self.LATENT`` (the orchestrator's, which equals OBS_DIM for
        # AlphaZero adapters anyway).
        comptime run_init = gpu_mcts_init_root_kernel[
            Self.N_ENVS,
            Self.MAX_NODES,
            Self.ACT,
            Self.LATENT,
            Self.PRED_OUT,
            dtype,
        ]
        ctx.enqueue_function[run_init](
            vc_t,
            tv_t,
            pr_t,
            rw_t,
            ci_t,
            tvis_t,
            nc_t,
            po_root_t,
            miq_t,
            mxq_t,
            Scalar[dtype](0.0),
            UInt32(rng_seed & UInt64(0xFFFFFFFF)),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 3. Apply legal mask (with-noise variant if NOISE != None) ─
        comptime if Self.NOISE.NOISE_TYPE == 2:
            comptime run_mask = gpu_mcts_apply_legal_mask_kernel[
                Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
            ]
            ctx.enqueue_function[run_mask](
                pr_t,
                root_legal_masks,
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            comptime run_mask_noise = (
                gpu_mcts_apply_legal_mask_with_noise_kernel[
                    Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
                ]
            )
            ctx.enqueue_function[run_mask_noise](
                pr_t,
                root_legal_masks,
                Scalar[dtype](Self.NOISE.NOISE_FRACTION),
                UInt32(rng_seed & UInt64(0xFFFFFFFF)),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── 4. Copy root game state into game_states[node 0] ─────────
        var gs_t = tile[Self.N_ENVS * Self.MAX_NODES * Self.STATE_SIZE](self.state.game_states)
        var rs_t = rebind[
            Tile[dtype, Self.N_ENVS * Self.STATE_SIZE]
        ](
            tile[Self.N_ENVS * Self.STATE_SIZE](root_states)
        )
        comptime run_cp_root = gpu_mcts_copy_root_state_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.STATE_SIZE, dtype
        ]
        ctx.enqueue_function[run_cp_root](
            gs_t,
            rs_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 5. Simulation rounds ─────────────────────────────────────
        var b_pp = tile[Self.MCTS_TOTAL](self.state.pending_parent)
        var b_pa = tile[Self.MCTS_TOTAL](self.state.pending_action)
        var b_sp = tile[Self.MCTS_TOTAL * MAX_DEPTH](self.state.search_paths)
        var b_ap = tile[Self.MCTS_TOTAL * MAX_DEPTH](self.state.action_paths)
        var b_pl = tile[Self.MCTS_TOTAL](self.state.path_lengths)
        var b_exp_st = tile[Self.MCTS_TOTAL * Self.STATE_SIZE](self.state.expansion_states)

        for _round in range(Self.MCTS_ROUNDS):
            # 5a. select + copy parent state to staging
            comptime run_sel_cp = gpu_mcts_batched_select_and_copy_kernel[
                Self.N_ENVS,
                Self.MAX_NODES,
                Self.ACT,
                Self.BATCH_SIMS,
                Self.STATE_SIZE,
                dtype,
                VIRTUAL_LOSS_VAL=Self.VIRTUAL_LOSS,
            ]
            ctx.enqueue_function[run_sel_cp](
                vc_t,
                tv_t,
                pr_t,
                ci_t,
                tvis_t,
                nc_t,
                miq_t,
                mxq_t,
                gs_t,
                b_pp,
                b_pa,
                b_exp_st,
                b_sp,
                b_ap,
                b_pl,
                Scalar[dtype](Self.PUCT.C_BASE),
                Scalar[dtype](Self.PUCT.C_INIT),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # 5b. env.step_gpu — writes obs + reward + done +
            # terminated + legal_mask in place. Adapter is responsible
            # for the per-sample (state, action) → outputs mapping.
            env.step_gpu[Self.MCTS_TOTAL](
                ctx,
                self.state.expansion_states,
                self.state.pending_action,
                self.state.leaf_values,  # rewards buffer (reused)
                self.az_exp_dones,
                self.az_exp_terminated,
                self.az_exp_obs,
                self.state.expansion_legal_masks,
                rng_seed=rng_seed + UInt64(_round),
            )

            # 5c. predict on the post-step obs
            var p_in = tile[Self.MCTS_TOTAL, PRED.LATENT_DIM](self.az_exp_obs)
            var p_out = tile[Self.MCTS_TOTAL, PRED.PRED_OUT_DIM](self.state.pred_output)
            pred.predict_gpu[Self.MCTS_TOTAL](ctx, p_in, p_out)

            # 5d. expand + backup with legal-mask propagation
            var b_po = tile[Self.MCTS_TOTAL * Self.PRED_OUT](self.state.pred_output)
            var b_rew = tile[Self.MCTS_TOTAL](self.state.leaf_values)
            var b_lm = tile[Self.MCTS_TOTAL * Self.ACT](self.state.expansion_legal_masks)
            var b_dones = tile[Self.MCTS_TOTAL](self.az_exp_dones)

            comptime run_exp = gpu_mcts_batched_expand_backup_masked_kernel[
                Self.N_ENVS,
                Self.MAX_NODES,
                Self.ACT,
                Self.BATCH_SIMS,
                Self.PRED_OUT,
                Self.STATE_SIZE,
                Self.PLAYER.NEGATE_BACKUP,
                True,  # VALUE_SQUASH=True for AlphaZero scalar value head
                dtype,
                VIRTUAL_LOSS_VAL=Self.VIRTUAL_LOSS,
            ]
            ctx.enqueue_function[run_exp](
                vc_t,
                tv_t,
                pr_t,
                rw_t,
                ci_t,
                tvis_t,
                nc_t,
                miq_t,
                mxq_t,
                gs_t,
                b_exp_st,
                b_pp,
                b_pa,
                b_po,
                b_rew,
                b_dones,
                b_sp,
                b_ap,
                b_pl,
                b_lm,
                Scalar[dtype](self.gamma),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── 6. Masked action extraction ──────────────────────────────
        var act_out_t = tile[Self.N_ENVS](self.actions_out)
        var pol_out_t = tile[Self.N_ENVS * Self.ACT](self.policies_out)
        comptime run_act = gpu_mcts_extract_actions_masked_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_act](
            vc_t,
            root_legal_masks,
            act_out_t,
            pol_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 7. Root value ────────────────────────────────────────────
        var rv_out_t = tile[Self.N_ENVS](self.root_value_out)
        comptime run_rv = gpu_mcts_extract_root_value_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_rv](
            vc_t,
            tv_t,
            rv_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Output views
    # ══════════════════════════════════════════════════════════════════════

    def actions_view(
        self,
    ) -> Tile[dtype, Self.N_ENVS]:
        """``[N_ENVS]`` argmax action — caller copies to host to read."""
        # T6: copy the field to a mut local so the view is mut=True and widens
        # into the MutAnyOrigin return type (read-only in practice).
        var b = self.actions_out
        return tile[Self.N_ENVS](b)

    def policies_view(
        self,
    ) -> Tile[dtype, Self.N_ENVS * Self.ACT]:
        """``[N_ENVS × ACT]`` visit-count policy per env (sums to 1)."""
        # T6: mut local copy → mut=True view → widens into MutAnyOrigin return.
        var b = self.policies_out
        return tile[Self.N_ENVS * Self.ACT](b)

    def root_value_view(
        self,
    ) -> Tile[dtype, Self.N_ENVS]:
        """``[N_ENVS]`` visit-weighted Q at the root."""
        # T6: mut local copy → mut=True view → widens into MutAnyOrigin return.
        var b = self.root_value_out
        return tile[Self.N_ENVS](b)
