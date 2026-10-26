"""GPU MCTS — Fully GPU-resident Monte Carlo Tree Search.

Promoted from ``noeira/deep_agents/muzero/gpu_mcts.mojo`` in Phase 3b of
the planners refactor (see ``docs/PLANNERS_PACKAGE.md``). All kernels +
``GPUMCTSState`` live here so MuZero / EZv2 / AlphaZero (and future
tree-search agents) can share them without a downward import into a
specific agent's module.

The kernels themselves are **agent-agnostic**: they take buffer views
(``dyn_output`` / ``pred_output`` / ``game_states``) and shape constants,
they don't know about networks. The agent owns orchestration —
interleaving ``Network[…].forward_gpu[B]`` calls between kernel launches.
A future generic ``GPUMCTS`` struct (see ``RepresentationGPU`` /
``DynamicsGPU`` / ``PredictionGPU`` traits in ``model_traits_gpu.mojo``)
will hide that loop; until then, the agent's ``select_action_gpu``
remains the orchestrator.

The shim ``muzero/gpu_mcts.mojo`` re-exports every public symbol so
existing imports keep working through the strangler migration.

Root parallelization: one independent MCTS tree per environment, all
running in parallel on GPU. Network evaluations (dynamics + prediction)
are batched across all environments.

Architecture (per simulation round, all n_envs in parallel):
  1. GPU kernel: PUCT selection [1 thread/env] → pending expansion buffer
  2. GPU kernel: build dynamics input [n_envs × (LATENT + ACT)]
  3. GPU forward: dynamics [BATCH=n_envs] → next hidden + reward
  4. GPU forward: prediction [BATCH=n_envs] → policy + value
  5. GPU kernel: expand node + set priors [1 thread/env]
  6. GPU kernel: backup [1 thread/env]
  After all simulations:
  7. GPU kernel: visit counts → action selection

No CPU↔GPU sync during MCTS — only the final actions are read back.

References:
  - Root parallelization: Wikipedia MCTS article
  - GPU Monte Carlo: Buzer & Cazenave, 2023 (LION 17) — 420x speedup
  - MuZero: Schrittwieser et al., 2020 (Nature)
"""

from max.gpu import block_dim, block_idx, thread_idx
from std.random.philox import Random as PhiloxRandom
from std.math import sqrt, log, exp
from max.gpu.host import DeviceContext, DeviceBuffer
from noeira.nn.constants import DT as dtype
from noeira.core.tiles import Tile, shared_tile

comptime TPB: Int = 256
comptime MAX_DEPTH: Int = 32  # Maximum tree depth for search paths


# ═══════════════════════════════════════════════════════════════════════════
# Helper kernels — hidden-state scale + dyn→pred copy
# ═══════════════════════════════════════════════════════════════════════════
#
# Promoted from ``muzero/kernels.mojo`` so the GPU MCTS orchestrator
# (``GenericGPUMCTS``) doesn't need a downward import into a specific
# agent's module. These are the only two utility kernels the orchestrator
# wires between the MCTS kernels above and the agent's
# ``Network[…].forward_gpu`` calls — extracting them keeps the
# orchestrator self-contained.


def mcts_gpu_scale_hidden_kernel[
    BATCH: Int,
    LATENT: Int,
    dtype: DType,
](
    hidden: Tile[dtype, BATCH * LATENT],
) where dtype.is_floating_point():
    """Min-max scale ``hidden`` to [0, 1] per sample.

    One thread per sample (= per env / per leaf). Used after the
    representation forward to bring the root hidden into the same
    [0, 1] range the rest of MCTS assumes.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var off = b * LATENT
    var mn = rebind[Scalar[dtype]](hidden[off])
    var mx = mn
    for i in range(1, LATENT):
        var v = rebind[Scalar[dtype]](hidden[off + i])
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    var delta = mx - mn
    if delta > Scalar[dtype](1e-8):
        var inv = Scalar[dtype](1.0) / delta
        for i in range(LATENT):
            var v = rebind[Scalar[dtype]](hidden[off + i])
            hidden[off + i] = (v - mn) * inv


def mcts_gpu_scatter_root_hidden_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    LATENT: Int,
    dtype: DType,
](
    hidden: Tile[dtype, N_ENVS * MAX_NODES * LATENT],
) where dtype.is_floating_point():
    """Scatter dense ``[N_ENVS, LATENT]`` root hidden into strided tree slots.

    The representation forward writes the root hidden for all envs into
    the dense prefix ``hidden[0 .. N_ENVS * LATENT)`` (row e at
    ``e * LATENT``). But every downstream MCTS kernel reads env e's
    node-0 hidden from the strided offset
    ``e * MAX_NODES * LATENT + 0 * LATENT``. For ``e=0`` the two
    coincide (offset 0). For ``e>=1`` the strided slot is uninitialized,
    so envs 1..N-1 all read whatever happens to be in that memory
    (typically zeros) — producing wrong, identical-across-envs MCTS
    trees. See ``tests/deep_agents/test_muzero_gpu_mcts_per_env_isolation.mojo``
    for the rigorous diagnostic that surfaced this.

    Fix: after rep forward + MinMax post-scale, copy each env's dense
    row into its strided root slot. The dense region [0, N_ENVS*LATENT)
    and the strided slots for envs 1..N-1 (each at offset
    e*MAX_NODES*LATENT ≥ MAX_NODES*LATENT ≥ N_ENVS*LATENT when
    MAX_NODES ≥ N_ENVS) don't overlap, so there's no data race.
    One thread per (env, latent_dim) cell. env=0 is a self-copy → skip.
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N_ENVS * LATENT:
        return
    var e = idx // LATENT
    var l = idx % LATENT
    if e == 0:
        return  # self-copy: src offset = dst offset = l
    var src = e * LATENT + l
    var dst = e * MAX_NODES * LATENT + l
    hidden[dst] = hidden[src]


def mcts_gpu_extract_hidden_kernel[
    BATCH: Int,
    LATENT: Int,
    DYN_OUT: Int,
    dtype: DType,
](
    next_hidden: Tile[dtype, BATCH * LATENT],
    dyn_output: Tile[dtype, BATCH * DYN_OUT],
) where dtype.is_floating_point():
    """Slice the first ``LATENT`` elements of each row of ``dyn_output``
    into ``next_hidden`` — turns the dynamics network's split output
    (hidden ++ reward_logits) into a flat batched hidden buffer the
    prediction network can read directly.

    One thread per output scalar (=BATCH × LATENT total threads).
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * LATENT:
        return
    var b = idx // LATENT
    var d = idx % LATENT
    next_hidden[idx] = dyn_output[b * DYN_OUT + d]


# ═══════════════════════════════════════════════════════════════════════════
# GPU Tree Storage Layout
# ═══════════════════════════════════════════════════════════════════════════
#
# All n_envs trees stored in flat device buffers with strided layout:
#   node_visit_count: [n_envs × MAX_NODES × ACT]  (int32 stored as dtype)
#   node_total_value: [n_envs × MAX_NODES × ACT]
#   node_prior:       [n_envs × MAX_NODES × ACT]
#   node_reward:      [n_envs × MAX_NODES × ACT]
#   node_child_idx:   [n_envs × MAX_NODES × ACT]  (-1 = unexpanded)
#   node_total_visits: [n_envs × MAX_NODES]
#   hidden_states:    [n_envs × MAX_NODES × LATENT]
#   node_count:       [n_envs]  (number of nodes per tree)
#
# Indexing: tree_offset = env_idx * MAX_NODES
#           node_act_offset = (tree_offset + node_idx) * ACT + action


# ═══════════════════════════════════════════════════════════════════════════
# GPU MCTS Kernels
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_init_root_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    PRED_OUT: Int,
    dtype: DType,
](
    # Node storage
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    reward: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    # Prediction output (from representation + prediction forward on root obs)
    pred_output: Tile[dtype, N_ENVS * PRED_OUT],
    # MinMax tracking
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    # Dirichlet noise
    noise_fraction: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Initialize root node for each env's tree from prediction output.

    Sets prior from softmax of policy logits, adds Dirichlet noise.
    One thread per environment.

    IMPORTANT: Caller must zero visit_count, total_value, reward, total_visits
    and set child_idx to -1 via unsafe_memset BEFORE launching this kernel.
    Prior is set here from softmax, so it doesn't need zeroing.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT

    # Set root prior from softmax of policy logits
    var pred_off = e * PRED_OUT
    var max_logit = rebind[Scalar[dtype]](pred_output[pred_off])
    for a in range(1, ACT):
        var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
        if v > max_logit:
            max_logit = v
    var sum_exp = Scalar[dtype](0.0)
    for a in range(ACT):
        var v = exp(
            rebind[Scalar[dtype]](pred_output[pred_off + a]) - max_logit
        )
        prior[tree_off + a] = v
        sum_exp += v
    for a in range(ACT):
        prior[tree_off + a] = (
            rebind[Scalar[dtype]](prior[tree_off + a]) / sum_exp
        )

    # Add Dirichlet noise (approximate with Exponential(1) / sum)
    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(e * 137 + 1), offset=0
    )
    var noise_sum = Scalar[dtype](0.0)
    for a in range(ACT):
        var rand_vals = philox.step_uniform()
        var noise_val = -log(Scalar[dtype](rand_vals[0]) + Scalar[dtype](1e-8))
        reward[
            tree_off + a
        ] = noise_val  # Temporarily store noise in reward[root]
        noise_sum += noise_val
    for a in range(ACT):
        var noise_val = rebind[Scalar[dtype]](reward[tree_off + a]) / noise_sum
        prior[tree_off + a] = (Scalar[dtype](1.0) - noise_fraction) * rebind[
            Scalar[dtype]
        ](prior[tree_off + a]) + noise_fraction * noise_val
        reward[tree_off + a] = Scalar[dtype](0.0)  # Reset reward

    # Initialize tree state
    node_count[e] = Scalar[dtype](1.0)  # Root is node 0
    min_q[e] = Scalar[dtype](1e18)
    max_q[e] = Scalar[dtype](-1e18)


def gpu_mcts_select_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType,
](
    # Node storage (read)
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    # MinMax Q normalization
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    # Output: pending expansion info
    pending_parent: Tile[dtype, N_ENVS],
    pending_action: Tile[dtype, N_ENVS],
    # Output: search paths for backup
    search_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    action_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS],
    # PUCT constants
    c_base: Scalar[dtype],
    c_init: Scalar[dtype],
) where dtype.is_floating_point():
    """PUCT selection: traverse each env's tree from root to leaf.

    One thread per environment. Sequential tree traversal within each tree,
    parallel across all n_envs trees.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var path_off = e * MAX_DEPTH

    var node_idx = 0
    var depth = 0
    search_paths[path_off] = Scalar[dtype](0.0)

    var mn = rebind[Scalar[dtype]](min_q[e])
    var mx = rebind[Scalar[dtype]](max_q[e])
    var q_range = mx - mn

    while depth < MAX_DEPTH - 1:
        # PUCT selection
        var n_total = rebind[Scalar[dtype]](total_visits[tv_off + node_idx])
        var sqrt_total = sqrt(n_total + Scalar[dtype](1e-8))
        var c = log((Scalar[dtype](1.0) + n_total + c_base) / c_base) + c_init

        var best_action = 0
        var best_score = Scalar[dtype](-1e18)

        for a in range(ACT):
            var na_off = tree_off + node_idx * ACT + a
            var n_a = rebind[Scalar[dtype]](visit_count[na_off])

            var q_val: Scalar[dtype]
            if n_a > Scalar[dtype](0.5):
                q_val = rebind[Scalar[dtype]](total_value[na_off]) / n_a
                # MinMax normalize
                if q_range > Scalar[dtype](1e-8):
                    q_val = (q_val - mn) / q_range
            else:
                q_val = Scalar[dtype](0.0)

            var p = rebind[Scalar[dtype]](prior[na_off])
            var exploration = c * p * sqrt_total / (Scalar[dtype](1.0) + n_a)
            var score = q_val + exploration

            if score > best_score:
                best_score = score
                best_action = a

        action_paths[path_off + depth] = Scalar[dtype](best_action)

        # Check if child exists
        var child = rebind[Scalar[dtype]](
            child_idx[tree_off + node_idx * ACT + best_action]
        )
        if child < Scalar[dtype](0.0):
            # Leaf — record pending expansion
            pending_parent[e] = Scalar[dtype](node_idx)
            pending_action[e] = Scalar[dtype](best_action)
            path_lengths[e] = Scalar[dtype](depth + 1)
            return

        # Descend to child
        node_idx = Int(child)
        depth += 1
        search_paths[path_off + depth] = Scalar[dtype](node_idx)

    # Max depth reached — use current node as leaf
    pending_parent[e] = Scalar[dtype](node_idx)
    pending_action[e] = Scalar[dtype](0)
    path_lengths[e] = Scalar[dtype](depth + 1)


def gpu_mcts_expand_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    PRED_OUT: Int,
    DYN_OUT: Int,
    dtype: DType,
](
    # Node storage (write)
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    reward_buf: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    # Hidden state pool
    hidden_states: Tile[dtype, N_ENVS * MAX_NODES * LATENT],
    # Pending expansion info
    pending_parent: Tile[dtype, N_ENVS],
    pending_action: Tile[dtype, N_ENVS],
    # Network outputs
    dyn_output: Tile[dtype, N_ENVS * DYN_OUT],
    pred_output: Tile[dtype, N_ENVS * PRED_OUT],
    # Value support
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
    # Output: leaf values for backup
    leaf_values: Tile[dtype, N_ENVS],
) where dtype.is_floating_point():
    """Create child nodes from dynamics/prediction outputs.

    One thread per environment. Each thread:
    1. Extracts hidden state from dynamics output → hidden pool
    2. Scales hidden state [0,1]
    3. Extracts reward from dynamics output
    4. Sets child prior from prediction softmax
    5. Links parent → child
    6. Decodes leaf value from prediction output
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var h_off = e * MAX_NODES * LATENT

    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var action = Int(rebind[Scalar[dtype]](pending_action[e]))
    var child_node_idx = Int(rebind[Scalar[dtype]](node_count[e]))

    # Guard against overflow
    if child_node_idx >= MAX_NODES:
        leaf_values[e] = Scalar[dtype](0.0)
        return

    # 1. Extract hidden state from dynamics output
    var dyn_off = e * DYN_OUT
    var child_h_off = h_off + child_node_idx * LATENT
    for i in range(LATENT):
        hidden_states[child_h_off + i] = dyn_output[dyn_off + i]

    # 2. Scale hidden state to [0,1]
    var h_min = rebind[Scalar[dtype]](hidden_states[child_h_off])
    var h_max = h_min
    for i in range(1, LATENT):
        var v = rebind[Scalar[dtype]](hidden_states[child_h_off + i])
        if v < h_min:
            h_min = v
        if v > h_max:
            h_max = v
    var h_delta = h_max - h_min
    if h_delta > Scalar[dtype](1e-8):
        for i in range(LATENT):
            var v = rebind[Scalar[dtype]](hidden_states[child_h_off + i])
            hidden_states[child_h_off + i] = (v - h_min) / h_delta

    # 3. Extract reward from dynamics output
    comptime NUM_REW_BINS = DYN_OUT - LATENT
    var rew_decoded: Scalar[dtype]
    if NUM_REW_BINS == 1:
        # Scalar reward: read raw value, tanh for bounding
        var raw = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
        var e_pos = exp(raw)
        var e_neg = exp(-raw)
        rew_decoded = (e_pos - e_neg) / (e_pos + e_neg)  # tanh
    else:
        # Categorical reward: softmax + expectation over bins
        var rew_max_val = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
        for i in range(1, NUM_REW_BINS):
            var v = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
            if v > rew_max_val:
                rew_max_val = v
        var rew_sum_exp = Scalar[dtype](0.0)
        for i in range(NUM_REW_BINS):
            rew_sum_exp += exp(
                rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                - rew_max_val
            )
        rew_decoded = Scalar[dtype](0.0)
        var rew_step = (v_max - v_min) / Scalar[dtype](NUM_REW_BINS - 1)
        for i in range(NUM_REW_BINS):
            var prob = (
                exp(
                    rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                    - rew_max_val
                )
                / rew_sum_exp
            )
            rew_decoded += prob * (v_min + Scalar[dtype](i) * rew_step)
        # Inverse scalar transform
        var sign_r = Scalar[dtype](1.0) if rew_decoded >= Scalar[dtype](
            0.0
        ) else Scalar[dtype](-1.0)
        var abs_r = (
            rew_decoded if rew_decoded >= Scalar[dtype](0.0) else -rew_decoded
        )
        var eps_r = Scalar[dtype](0.001)
        var inner_r = sqrt(
            Scalar[dtype](1.0)
            + Scalar[dtype](4.0) * eps_r * (abs_r + Scalar[dtype](1.0) + eps_r)
        )
        var f_r = (inner_r - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps_r)
        rew_decoded = sign_r * (f_r * f_r - Scalar[dtype](1.0))

    # Set parent reward
    reward_buf[tree_off + parent * ACT + action] = rew_decoded

    # 4. Set child prior from prediction softmax
    var pred_off = e * PRED_OUT
    var child_tree_off = tree_off + child_node_idx * ACT
    var p_max = rebind[Scalar[dtype]](pred_output[pred_off])
    for a in range(1, ACT):
        var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
        if v > p_max:
            p_max = v
    var p_sum = Scalar[dtype](0.0)
    for a in range(ACT):
        var v = exp(rebind[Scalar[dtype]](pred_output[pred_off + a]) - p_max)
        prior[child_tree_off + a] = v
        p_sum += v
    for a in range(ACT):
        prior[child_tree_off + a] = (
            rebind[Scalar[dtype]](prior[child_tree_off + a]) / p_sum
        )
        visit_count[child_tree_off + a] = Scalar[dtype](0.0)
        total_value[child_tree_off + a] = Scalar[dtype](0.0)
        reward_buf[child_tree_off + a] = Scalar[dtype](0.0)
        child_idx[child_tree_off + a] = Scalar[dtype](-1.0)
    total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

    # 5. Link parent → child
    child_idx[tree_off + parent * ACT + action] = Scalar[dtype](child_node_idx)
    node_count[e] = Scalar[dtype](child_node_idx + 1)

    # 6. Decode leaf value
    comptime NUM_VAL_BINS = PRED_OUT - ACT
    if NUM_VAL_BINS == 1:
        # Scalar value: read raw output, tanh for [-1, 1] bounding
        var raw_v = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
        if raw_v > Scalar[dtype](10.0):
            raw_v = Scalar[dtype](10.0)
        elif raw_v < Scalar[dtype](-10.0):
            raw_v = Scalar[dtype](-10.0)
        var ev_pos = exp(raw_v)
        var ev_neg = exp(-raw_v)
        leaf_values[e] = (ev_pos - ev_neg) / (ev_pos + ev_neg)  # tanh
    else:
        # Categorical value: softmax + expectation over bins
        var val_max = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
        for i in range(1, NUM_VAL_BINS):
            var v = rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
            if v > val_max:
                val_max = v
        var val_sum_exp = Scalar[dtype](0.0)
        for i in range(NUM_VAL_BINS):
            val_sum_exp += exp(
                rebind[Scalar[dtype]](pred_output[pred_off + ACT + i]) - val_max
            )
        var val_step = (v_max - v_min) / Scalar[dtype](NUM_VAL_BINS - 1)
        var val_decoded = Scalar[dtype](0.0)
        for i in range(NUM_VAL_BINS):
            var prob = (
                exp(
                    rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                    - val_max
                )
                / val_sum_exp
            )
            val_decoded += prob * (v_min + Scalar[dtype](i) * val_step)

        # Inverse scalar transform: h^{-1}(y)
        var sign = Scalar[dtype](1.0) if val_decoded >= Scalar[dtype](
            0.0
        ) else Scalar[dtype](-1.0)
        var abs_y = (
            val_decoded if val_decoded >= Scalar[dtype](0.0) else -val_decoded
        )
        var eps = Scalar[dtype](0.001)
        var inner = sqrt(
            Scalar[dtype](1.0)
            + Scalar[dtype](4.0) * eps * (abs_y + Scalar[dtype](1.0) + eps)
        )
        var f = (inner - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps)
        leaf_values[e] = sign * (f * f - Scalar[dtype](1.0))


def gpu_mcts_backup_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType,
](
    # Node storage (read/write)
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    reward_buf: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    # Search paths
    search_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    action_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS],
    leaf_values: Tile[dtype, N_ENVS],
    gamma: Scalar[dtype],
) where dtype.is_floating_point():
    """Backup leaf values through search paths.

    One thread per environment. Walks backwards from leaf to root,
    accumulating discounted returns and updating visit counts.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var path_off = e * MAX_DEPTH
    var path_len = Int(rebind[Scalar[dtype]](path_lengths[e]))

    var value = rebind[Scalar[dtype]](leaf_values[e])

    # Walk backwards from leaf parent to root
    for i in range(path_len):
        var idx = path_len - 1 - i
        var node_idx = Int(rebind[Scalar[dtype]](search_paths[path_off + idx]))
        var action = Int(rebind[Scalar[dtype]](action_paths[path_off + idx]))

        var na_off = tree_off + node_idx * ACT + action
        value = rebind[Scalar[dtype]](reward_buf[na_off]) + gamma * value

        visit_count[na_off] = rebind[Scalar[dtype]](
            visit_count[na_off]
        ) + Scalar[dtype](1.0)
        total_value[na_off] = rebind[Scalar[dtype]](total_value[na_off]) + value
        total_visits[tv_off + node_idx] = rebind[Scalar[dtype]](
            total_visits[tv_off + node_idx]
        ) + Scalar[dtype](1.0)

        # Update MinMax
        var n_a = rebind[Scalar[dtype]](visit_count[na_off])
        var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
        if mean_q < rebind[Scalar[dtype]](min_q[e]):
            min_q[e] = mean_q
        if mean_q > rebind[Scalar[dtype]](max_q[e]):
            max_q[e] = mean_q


def gpu_mcts_extract_actions_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType,
](
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    actions_out: Tile[dtype, N_ENVS],
    policies_out: Tile[dtype, N_ENVS * ACT],
) where dtype.is_floating_point():
    """Extract action from root visit counts (argmax) and policy (normalized visits).

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT  # Root is always node 0

    # Sum visit counts and find argmax
    var total = Scalar[dtype](0.0)
    var best_action = 0
    var best_count = rebind[Scalar[dtype]](visit_count[root_off])
    for a in range(ACT):
        var count = rebind[Scalar[dtype]](visit_count[root_off + a])
        total += count
        if count > best_count:
            best_count = count
            best_action = a

    actions_out[e] = Scalar[dtype](best_action)

    # Normalized policy
    if total > Scalar[dtype](0.5):
        for a in range(ACT):
            policies_out[e * ACT + a] = (
                rebind[Scalar[dtype]](visit_count[root_off + a]) / total
            )
    else:
        for a in range(ACT):
            policies_out[e * ACT + a] = Scalar[dtype](1.0) / Scalar[dtype](ACT)


def gpu_mcts_extract_root_value_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType,
](
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    values_out: Tile[dtype, N_ENVS],
) where dtype.is_floating_point():
    """Extract MCTS root value as `Σ_a total_value[root,a] / Σ_a visit_count[root,a]`.

    Mirrors muzero-general's `node.value() = value_sum / visit_count` and is
    used as the n-step bootstrap V(s_{t+n}) in `nstep_value_targets_kernel`.
    Without this, the bootstrap would be a hardcoded zero — value targets
    collapse to the sum of K rewards regardless of state, and the value head
    learns a constant. See docs/MUZERO_AUDIT.md F2.

    Total_value is accumulated by the expand+backup kernel in raw (post-h⁻¹)
    scalar space (after F1), so the returned root value is in raw space —
    the same space the n-step return formula expects.

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT  # Root is always node 0

    var visit_sum = Scalar[dtype](0.0)
    var value_sum = Scalar[dtype](0.0)
    for a in range(ACT):
        visit_sum += rebind[Scalar[dtype]](visit_count[root_off + a])
        value_sum += rebind[Scalar[dtype]](total_value[root_off + a])

    if visit_sum > Scalar[dtype](0.5):
        values_out[e] = value_sum / visit_sum
    else:
        values_out[e] = Scalar[dtype](0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Self-Play Kernels (legal masking + negated backup)
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_apply_legal_mask_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType,
](
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    legal_masks: Tile[dtype, N_ENVS * ACT],
) where dtype.is_floating_point():
    """Mask root prior with legal action mask and renormalize.

    For board games: zero out illegal actions in root prior, then
    renormalize so probabilities sum to 1.

    One thread per environment. Call AFTER init_root, BEFORE simulations.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT  # Root is always node 0

    # Apply mask
    var sum_p = Scalar[dtype](0.0)
    for a in range(ACT):
        var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
        if legal < Scalar[dtype](0.5):
            prior[root_off + a] = Scalar[dtype](0.0)
        else:
            sum_p += rebind[Scalar[dtype]](prior[root_off + a])

    # Renormalize
    if sum_p > Scalar[dtype](1e-8):
        for a in range(ACT):
            prior[root_off + a] = (
                rebind[Scalar[dtype]](prior[root_off + a]) / sum_p
            )
    else:
        # All actions masked — uniform over legal actions (fallback)
        var n_legal = Scalar[dtype](0.0)
        for a in range(ACT):
            if rebind[Scalar[dtype]](legal_masks[e * ACT + a]) > Scalar[dtype](
                0.5
            ):
                n_legal += Scalar[dtype](1.0)
        if n_legal > Scalar[dtype](0.5):
            var inv_n = Scalar[dtype](1.0) / n_legal
            for a in range(ACT):
                if rebind[Scalar[dtype]](legal_masks[e * ACT + a]) > Scalar[
                    dtype
                ](0.5):
                    prior[root_off + a] = inv_n


def gpu_mcts_apply_legal_mask_with_noise_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType,
](
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    legal_masks: Tile[dtype, N_ENVS * ACT],
    noise_fraction: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
) where dtype.is_floating_point():
    """Mask root prior, add Dirichlet noise only to legal actions, renormalize.

    Unlike apply_legal_mask_kernel, this generates Dirichlet noise ONLY for
    legal actions, ensuring the full noise budget goes to valid moves.

    One thread per environment. Call AFTER init_root (with noise=0), BEFORE simulations.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT  # Root is always node 0

    # Step 1: Apply legal mask to prior and renormalize (strip any pre-existing noise)
    var sum_p = Scalar[dtype](0.0)
    for a in range(ACT):
        var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
        if legal < Scalar[dtype](0.5):
            prior[root_off + a] = Scalar[dtype](0.0)
        else:
            sum_p += rebind[Scalar[dtype]](prior[root_off + a])

    if sum_p > Scalar[dtype](1e-8):
        for a in range(ACT):
            prior[root_off + a] = (
                rebind[Scalar[dtype]](prior[root_off + a]) / sum_p
            )
    else:
        var n_legal = Scalar[dtype](0.0)
        for a in range(ACT):
            if rebind[Scalar[dtype]](legal_masks[e * ACT + a]) > Scalar[dtype](
                0.5
            ):
                n_legal += Scalar[dtype](1.0)
        if n_legal > Scalar[dtype](0.5):
            var inv_n = Scalar[dtype](1.0) / n_legal
            for a in range(ACT):
                if rebind[Scalar[dtype]](legal_masks[e * ACT + a]) > Scalar[
                    dtype
                ](0.5):
                    prior[root_off + a] = inv_n

    # Step 2: Generate Dirichlet noise only for legal actions
    # Use two passes with deterministic PhiloxRandom (same seed = same sequence)
    if noise_fraction > Scalar[dtype](0.0):
        # Pass 1: compute noise sum over legal actions only
        var philox1 = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(e * 137 + 1), offset=0
        )
        var noise_sum = Scalar[dtype](0.0)
        for a in range(ACT):
            var rand_vals = philox1.step_uniform()
            var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
            if legal > Scalar[dtype](0.5):
                noise_sum += -log(
                    Scalar[dtype](rand_vals[0]) + Scalar[dtype](1e-8)
                )

        # Pass 2: regenerate same noise, normalize, and blend with prior
        if noise_sum > Scalar[dtype](1e-8):
            var philox2 = PhiloxRandom(
                seed=UInt64(rng_seed) + UInt64(e * 137 + 1), offset=0
            )
            for a in range(ACT):
                var rand_vals = philox2.step_uniform()
                var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
                if legal > Scalar[dtype](0.5):
                    var noise_val = -log(
                        Scalar[dtype](rand_vals[0]) + Scalar[dtype](1e-8)
                    ) / noise_sum
                    prior[root_off + a] = (
                        (Scalar[dtype](1.0) - noise_fraction)
                        * rebind[Scalar[dtype]](prior[root_off + a])
                        + noise_fraction * noise_val
                    )


def gpu_mcts_backup_negated_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType,
](
    # Node storage (read/write)
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    reward_buf: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    # Search paths
    search_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    action_paths: Tile[dtype, N_ENVS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS],
    leaf_values: Tile[dtype, N_ENVS],
) where dtype.is_floating_point():
    """Backup with value negation for two-player zero-sum games.

    At each backup level, the value is NEGATED (parent's perspective
    is opposite to child's perspective). No discount (gamma=1 for board games).
    No per-step reward (only terminal reward).

    value = -value  at each level

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var path_off = e * MAX_DEPTH
    var path_len = Int(rebind[Scalar[dtype]](path_lengths[e]))

    var value = rebind[Scalar[dtype]](leaf_values[e])

    # Walk backwards from leaf parent to root, NEGATING at each level
    for i in range(path_len):
        var idx = path_len - 1 - i
        var node_idx = Int(rebind[Scalar[dtype]](search_paths[path_off + idx]))
        var action = Int(rebind[Scalar[dtype]](action_paths[path_off + idx]))

        # Negate value (parent sees opposite of child)
        value = -value

        var na_off = tree_off + node_idx * ACT + action
        visit_count[na_off] = rebind[Scalar[dtype]](
            visit_count[na_off]
        ) + Scalar[dtype](1.0)
        total_value[na_off] = rebind[Scalar[dtype]](total_value[na_off]) + value
        total_visits[tv_off + node_idx] = rebind[Scalar[dtype]](
            total_visits[tv_off + node_idx]
        ) + Scalar[dtype](1.0)

        # Update MinMax
        var n_a = rebind[Scalar[dtype]](visit_count[na_off])
        var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
        if mean_q < rebind[Scalar[dtype]](min_q[e]):
            min_q[e] = mean_q
        if mean_q > rebind[Scalar[dtype]](max_q[e]):
            max_q[e] = mean_q


def gpu_mcts_extract_actions_masked_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType,
](
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    legal_masks: Tile[dtype, N_ENVS * ACT],
    actions_out: Tile[dtype, N_ENVS],
    policies_out: Tile[dtype, N_ENVS * ACT],
) where dtype.is_floating_point():
    """Extract actions respecting legal mask — choose only among legal actions.

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT

    var total = Scalar[dtype](0.0)
    var best_action = -1
    var best_count = Scalar[dtype](-1.0)

    for a in range(ACT):
        var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
        if legal > Scalar[dtype](0.5):
            var count = rebind[Scalar[dtype]](visit_count[root_off + a])
            total += count
            if count > best_count or best_action < 0:
                best_count = count
                best_action = a

    if best_action < 0:
        best_action = 0  # Fallback (shouldn't happen with valid game state)

    actions_out[e] = Scalar[dtype](best_action)

    # Normalized policy (only over legal actions)
    if total > Scalar[dtype](0.5):
        for a in range(ACT):
            var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
            if legal > Scalar[dtype](0.5):
                policies_out[e * ACT + a] = (
                    rebind[Scalar[dtype]](visit_count[root_off + a]) / total
                )
            else:
                policies_out[e * ACT + a] = Scalar[dtype](0.0)
    else:
        for a in range(ACT):
            policies_out[e * ACT + a] = Scalar[dtype](0.0)
        policies_out[e * ACT + best_action] = Scalar[dtype](1.0)


def gpu_mcts_extract_actions_temp_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType,
](
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    legal_masks: Tile[dtype, N_ENVS * ACT],
    ep_steps: Tile[dtype, N_ENVS],
    actions_out: Tile[dtype, N_ENVS],
    policies_out: Tile[dtype, N_ENVS * ACT],
    # ⚠ Int32, NOT Int — `Int`/`UInt` are not `DevicePassable`. A bare
    # `Int` still compiles in `pixi run build` and fails only where the
    # kernel is LAUNCHED; keep the `Int32(...)` casts at the call sites.
    temp_threshold: Int32,
    rng_seed: Scalar[DType.uint32],
    temp_min: Scalar[dtype] = Scalar[dtype](0.0),
) where dtype.is_floating_point():
    """Extract actions with temperature annealing.

    POLICY TARGET (``policies_out``): always the normalized visit
    distribution over legal actions (∝ visits), independent of
    temperature. This matches the CPU MCTS path (``GenericCPUMCTS``)
    and the AlphaZero paper convention where the search result π is
    the training target — softening the late-game target preserves
    the search's uncertainty.

    ACTION SELECTION (``actions_out``): temperature-controlled.
      - First ``temp_threshold`` moves: τ=1, sample ∝ N.
      - After ``temp_threshold``: τ = ``temp_min``.
          τ=0     → greedy argmax.
          τ>0     → sample from N^(1/τ).

    Previously this kernel overwrote ``policies_out`` with a one-hot
    target at τ=0, which deviated from CPU semantics and (with D4
    augmentation on small boards like TTT) caused premature policy
    entropy collapse + non-convergence. Action sampling now reads
    temperature-adjusted weights from local storage so the stored
    policy target is never sharpened.

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT
    var move_count = Int(rebind[Scalar[dtype]](ep_steps[e]))

    # Compute legal-masked visit total + argmax over legal actions.
    var total = Scalar[dtype](0.0)
    var best_action = -1
    var best_count = Scalar[dtype](-1.0)

    for a in range(ACT):
        var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
        if legal > Scalar[dtype](0.5):
            var count = rebind[Scalar[dtype]](visit_count[root_off + a])
            total += count
            if count > best_count or best_action < 0:
                best_count = count
                best_action = a

    if best_action < 0:
        best_action = 0

    # ── Soft policy target: π_a = N_a / Σ N (legal-masked) ──
    if total > Scalar[dtype](0.5):
        for a in range(ACT):
            var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
            if legal > Scalar[dtype](0.5):
                policies_out[e * ACT + a] = (
                    rebind[Scalar[dtype]](visit_count[root_off + a]) / total
                )
            else:
                policies_out[e * ACT + a] = Scalar[dtype](0.0)
    else:
        # No visits — fallback to one-hot on best_action (only used as
        # a degenerate seed; should not occur after MCTS has run).
        for a in range(ACT):
            policies_out[e * ACT + a] = Scalar[dtype](0.0)
        policies_out[e * ACT + best_action] = Scalar[dtype](1.0)

    # ── Temperature-controlled action selection ──
    var temp: Scalar[dtype]
    if move_count < Int(temp_threshold):
        temp = Scalar[dtype](1.0)
    else:
        temp = temp_min

    if temp > Scalar[dtype](0.01) and total > Scalar[dtype](0.5):
        # Sample from temperature-adjusted N^(1/τ) over legal actions.
        # Weights are kept in a local Array so we never mutate
        # the stored soft policy target.
        var inv_temp = Scalar[dtype](1.0) / temp
        var weighted_total = Scalar[dtype](0.0)
        var w_arr = Array[Scalar[dtype], ACT](fill=Scalar[dtype](0.0))
        for a in range(ACT):
            var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
            if legal > Scalar[dtype](0.5):
                var count = rebind[Scalar[dtype]](visit_count[root_off + a])
                if count > Scalar[dtype](0.5):
                    var w = exp(log(count) * inv_temp)
                    w_arr[a] = w
                    weighted_total += w

        if weighted_total <= Scalar[dtype](1e-8):
            actions_out[e] = Scalar[dtype](best_action)
        else:
            var philox = PhiloxRandom(
                seed=UInt64(rng_seed) + UInt64(e * 7919 + move_count * 6271),
                offset=0,
            )
            var rand_vals = philox.step_uniform()
            var rand = Scalar[dtype](rand_vals[0]) * weighted_total

            var cumsum = Scalar[dtype](0.0)
            var sampled = best_action  # Fallback to best
            for a in range(ACT):
                if w_arr[a] > Scalar[dtype](0.0):
                    cumsum += w_arr[a]
                    if rand < cumsum:
                        sampled = a
                        break
            actions_out[e] = Scalar[dtype](sampled)
    else:
        # τ ≤ 0.01: greedy argmax. Policy target stays soft (above).
        actions_out[e] = Scalar[dtype](best_action)


# ═══════════════════════════════════════════════════════════════════════════
# Game State Kernels (AlphaZero mode — true game rules)
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_copy_parent_state_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    STATE_SIZE: Int,
    dtype: DType,
](
    expansion_states: Tile[dtype, N_ENVS * STATE_SIZE],
    game_states: Tile[dtype, N_ENVS * MAX_NODES * STATE_SIZE],
    pending_parent: Tile[dtype, N_ENVS],
) where dtype.is_floating_point():
    """Copy parent game state to expansion staging buffer.

    One thread per env. Copies STATE_SIZE floats from parent node's
    game state slot to the flat expansion buffer for env.step().
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var src_off = e * MAX_NODES * STATE_SIZE + parent * STATE_SIZE
    var dst_off = e * STATE_SIZE

    for i in range(STATE_SIZE):
        expansion_states[dst_off + i] = game_states[src_off + i]


def gpu_mcts_store_child_state_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    STATE_SIZE: Int,
    dtype: DType,
](
    game_states: Tile[dtype, N_ENVS * MAX_NODES * STATE_SIZE],
    expansion_states: Tile[dtype, N_ENVS * STATE_SIZE],
    node_count: Tile[dtype, N_ENVS],
) where dtype.is_floating_point():
    """Copy expansion result (after env.step) to child node's game state slot.

    One thread per env. The child index is node_count (before increment —
    expand kernel will increment it).
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var child_idx = Int(rebind[Scalar[dtype]](node_count[e]))
    if child_idx >= MAX_NODES:
        return

    var src_off = e * STATE_SIZE
    var dst_off = e * MAX_NODES * STATE_SIZE + child_idx * STATE_SIZE

    for i in range(STATE_SIZE):
        game_states[dst_off + i] = expansion_states[src_off + i]


def gpu_mcts_copy_root_state_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    STATE_SIZE: Int,
    dtype: DType,
](
    game_states: Tile[dtype, N_ENVS * MAX_NODES * STATE_SIZE],
    env_states: Tile[dtype, N_ENVS * STATE_SIZE],
) where dtype.is_floating_point():
    """Copy current env states to root node (node 0) of each env's tree.

    Called once at start of each MCTS search.
    One thread per env.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var src_off = e * STATE_SIZE
    var dst_off = e * MAX_NODES * STATE_SIZE  # Root = node 0

    for i in range(STATE_SIZE):
        game_states[dst_off + i] = env_states[src_off + i]


def gpu_mcts_expand_alphazero_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    PRED_OUT: Int,
    dtype: DType,
](
    # Node storage
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    reward_buf: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    # Pending expansion
    pending_parent: Tile[dtype, N_ENVS],
    pending_action: Tile[dtype, N_ENVS],
    # Prediction output [N_ENVS * PRED_OUT] = (policy_logits[ACT], value[1])
    pred_output: Tile[dtype, N_ENVS * PRED_OUT],
    # Reward from env.step [N_ENVS]
    step_rewards: Tile[dtype, N_ENVS],
    # Output
    leaf_values: Tile[dtype, N_ENVS],
) where dtype.is_floating_point():
    """AlphaZero-specific expand: uses env.step reward and scalar value with tanh.

    Unlike MuZero expand, this kernel:
    - Reads reward directly from env.step output (not dynamics network)
    - Reads value as a single scalar with tanh activation (not categorical)
    - Does NOT touch hidden_states (game states handled separately)
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var action = Int(rebind[Scalar[dtype]](pending_action[e]))
    var child_node_idx = Int(rebind[Scalar[dtype]](node_count[e]))

    if child_node_idx >= MAX_NODES:
        leaf_values[e] = Scalar[dtype](0.0)
        return

    # Set parent reward from env.step
    reward_buf[tree_off + parent * ACT + action] = step_rewards[e]

    # Set child prior from softmax of policy logits
    var pred_off = e * PRED_OUT
    var child_off = tree_off + child_node_idx * ACT
    var p_max = rebind[Scalar[dtype]](pred_output[pred_off])
    for a in range(1, ACT):
        var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
        if v > p_max:
            p_max = v
    var p_sum = Scalar[dtype](0.0)
    for a in range(ACT):
        var v = exp(rebind[Scalar[dtype]](pred_output[pred_off + a]) - p_max)
        prior[child_off + a] = v
        p_sum += v
    for a in range(ACT):
        prior[child_off + a] = (
            rebind[Scalar[dtype]](prior[child_off + a]) / p_sum
        )
        visit_count[child_off + a] = Scalar[dtype](0.0)
        total_value[child_off + a] = Scalar[dtype](0.0)
        reward_buf[child_off + a] = Scalar[dtype](0.0)
        child_idx[child_off + a] = Scalar[dtype](-1.0)
    total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

    # Link parent → child
    child_idx[tree_off + parent * ACT + action] = Scalar[dtype](child_node_idx)
    node_count[e] = Scalar[dtype](child_node_idx + 1)

    # Decode leaf value:
    # If env.step returned done (game over), use step reward directly.
    # Otherwise, use tanh(network_output) as value estimate.
    var step_done = rebind[Scalar[dtype]](
        step_rewards[e]
    )  # reward doubles as done signal
    # Actually step_rewards is the reward. We need done flag separately.
    # For now: if reward is +1 or -1 (terminal), use it. If 0, use network.
    var abs_rew = step_done if step_done >= Scalar[dtype](0.0) else -step_done
    if abs_rew > Scalar[dtype](0.5):
        # Terminal state: use game outcome directly
        leaf_values[e] = step_done
    else:
        # Non-terminal: use network value prediction
        var raw_v = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
        if raw_v > Scalar[dtype](10.0):
            raw_v = Scalar[dtype](10.0)
        elif raw_v < Scalar[dtype](-10.0):
            raw_v = Scalar[dtype](-10.0)
        var ev_p = exp(raw_v)
        var ev_n = exp(-raw_v)
        leaf_values[e] = (ev_p - ev_n) / (ev_p + ev_n)


# ═══════════════════════════════════════════════════════════════════════════
# Batched Fused Kernels (BATCH_SIMS leaves per env per round)
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_batched_select_and_copy_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BATCH_SIMS: Int,
    STATE_SIZE: Int,
    dtype: DType,
    VIRTUAL_LOSS_VAL: Int = 3,
](
    # Node storage
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    # Game states (for AlphaZero copy)
    game_states: Tile[dtype, N_ENVS * MAX_NODES * STATE_SIZE],
    # Output: BATCH_SIMS pending expansions per env
    pending_parents: Tile[dtype, N_ENVS * BATCH_SIMS],
    pending_actions: Tile[dtype, N_ENVS * BATCH_SIMS],
    expansion_states: Tile[dtype, N_ENVS * BATCH_SIMS * STATE_SIZE],
    # Search paths for backup (only last sim's path stored for simplicity)
    search_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    action_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS * BATCH_SIMS],
    # PUCT constants
    c_base: Scalar[dtype],
    c_init: Scalar[dtype],
) where dtype.is_floating_point():
    """Fused: select BATCH_SIMS leaves per env with virtual losses + copy parent game states.

    One thread per env. Each thread sequentially selects BATCH_SIMS leaves,
    applying virtual loss (+3 visits) after each selection to encourage diversity.
    Then copies parent game states to the expansion staging buffer.

    Combines gpu_mcts_select_kernel + gpu_mcts_copy_parent_state_kernel.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var mn = rebind[Scalar[dtype]](min_q[e])
    var mx = rebind[Scalar[dtype]](max_q[e])
    var q_range = mx - mn

    comptime VIRTUAL_LOSS: Int = VIRTUAL_LOSS_VAL

    # Use shared memory for root visit counts to prevent NVIDIA compiler
    # from caching global memory reads across loop iterations.
    # Each thread gets ACT+1 slots: [tid*(ACT+1) .. tid*(ACT+1)+ACT) = per-action visits,
    # slot [tid*(ACT+1)+ACT] = total visits sum.
    from max.gpu.memory import AddressSpace
    comptime SLOTS_PER_THREAD = ACT + 1
    var s_root = shared_tile[dtype, TPB * SLOTS_PER_THREAD]()

    var tid = Int(thread_idx.x)
    var s_off = tid * SLOTS_PER_THREAD

    # Load root visit counts into shared memory
    var s_total = Scalar[dtype](0.0)
    for _a in range(ACT):
        var vc_val = rebind[Scalar[dtype]](visit_count[tree_off + _a])
        s_root[s_off + _a] = vc_val
        s_total += vc_val
    s_root[s_off + ACT] = s_total

    for s in range(BATCH_SIMS):
        # ── PUCT Selection ──────────────────────────────────
        var node_idx = 0
        var depth = 0
        var path_off = (e * BATCH_SIMS + s) * MAX_DEPTH
        search_paths[path_off] = Scalar[dtype](0.0)

        while depth < MAX_DEPTH - 1:
            # Use shared memory for root, global for deeper nodes
            var n_total: Scalar[dtype]
            if node_idx == 0:
                n_total = rebind[Scalar[dtype]](s_root[s_off + ACT])
            else:
                n_total = rebind[Scalar[dtype]](total_visits[tv_off + node_idx])
            var sqrt_total = sqrt(n_total + Scalar[dtype](1e-8))
            # PUCT exploration constant:
            # MuZero: c = log((1 + N + c_base) / c_base) + c_init
            # AlphaGo: c = c_init (when c_base = 0)
            var c: Scalar[dtype]
            if c_base > Scalar[dtype](0.5):
                c = log(
                    (Scalar[dtype](1.0) + n_total + c_base) / c_base
                ) + c_init
            else:
                c = c_init  # Constant c_puct (AlphaGo-style)

            var best_action = 0
            var best_score = Scalar[dtype](-1e18)
            for a in range(ACT):
                var na_off = tree_off + node_idx * ACT + a
                # Use shared memory for root actions
                var n_a: Scalar[dtype]
                if node_idx == 0:
                    n_a = rebind[Scalar[dtype]](s_root[s_off + a])
                else:
                    n_a = rebind[Scalar[dtype]](visit_count[na_off])
                var q_val: Scalar[dtype]
                if n_a > Scalar[dtype](0.5):
                    q_val = rebind[Scalar[dtype]](total_value[na_off]) / n_a
                    if q_range > Scalar[dtype](1e-8):
                        q_val = (q_val - mn) / q_range
                else:
                    q_val = Scalar[dtype](0.0)
                var p = rebind[Scalar[dtype]](prior[na_off])
                var score = q_val + c * p * sqrt_total / (
                    Scalar[dtype](1.0) + n_a
                )
                if score > best_score:
                    best_score = score
                    best_action = a

            action_paths[path_off + depth] = Scalar[dtype](best_action)

            # Apply virtual loss to (node_idx, best_action) at EVERY descent
            # step — canonical AlphaGo Zero behavior. Without this, batched
            # sims share the same root→…→leaf-parent path and only diverge
            # at the leaf action, severely under-exploring root-level
            # alternatives. The matching subtraction happens during backup
            # so visit counts stay correct.
            var na_off = tree_off + node_idx * ACT + best_action
            visit_count[na_off] = rebind[Scalar[dtype]](
                visit_count[na_off]
            ) + Scalar[dtype](VIRTUAL_LOSS)
            total_visits[tv_off + node_idx] = rebind[Scalar[dtype]](
                total_visits[tv_off + node_idx]
            ) + Scalar[dtype](VIRTUAL_LOSS)
            if node_idx == 0:
                s_root[s_off + best_action] = rebind[Scalar[dtype]](
                    s_root[s_off + best_action]
                ) + Scalar[dtype](VIRTUAL_LOSS)
                s_root[s_off + ACT] = rebind[Scalar[dtype]](
                    s_root[s_off + ACT]
                ) + Scalar[dtype](VIRTUAL_LOSS)

            var child = rebind[Scalar[dtype]](
                child_idx[tree_off + node_idx * ACT + best_action]
            )
            if child < Scalar[dtype](0.0):
                # Leaf found
                var sim_off = e * BATCH_SIMS + s
                pending_parents[sim_off] = Scalar[dtype](node_idx)
                pending_actions[sim_off] = Scalar[dtype](best_action)
                path_lengths[sim_off] = Scalar[dtype](depth + 1)

                # ── Copy parent game state to expansion buffer ──
                var parent_gs_off = (
                    e * MAX_NODES * STATE_SIZE + node_idx * STATE_SIZE
                )
                var exp_off = sim_off * STATE_SIZE
                for i in range(STATE_SIZE):
                    expansion_states[exp_off + i] = game_states[
                        parent_gs_off + i
                    ]

                break

            node_idx = Int(child)
            depth += 1
            search_paths[path_off + depth] = Scalar[dtype](node_idx)

        # If max depth reached without finding leaf
        if depth >= MAX_DEPTH - 1:
            var sim_off = e * BATCH_SIMS + s
            pending_parents[sim_off] = Scalar[dtype](node_idx)
            pending_actions[sim_off] = Scalar[dtype](0)
            path_lengths[sim_off] = Scalar[dtype](depth + 1)
            var parent_gs_off = (
                e * MAX_NODES * STATE_SIZE + node_idx * STATE_SIZE
            )
            var exp_off = sim_off * STATE_SIZE
            for i in range(STATE_SIZE):
                expansion_states[exp_off + i] = game_states[parent_gs_off + i]


def gpu_mcts_batched_expand_backup_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BATCH_SIMS: Int,
    PRED_OUT: Int,
    STATE_SIZE: Int,
    dtype: DType,
](
    # Node storage
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    reward_buf: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    # Game states
    game_states: Tile[dtype, N_ENVS * MAX_NODES * STATE_SIZE],
    expansion_states: Tile[dtype, N_ENVS * BATCH_SIMS * STATE_SIZE],
    # Pending expansions [N_ENVS * BATCH_SIMS]
    pending_parents: Tile[dtype, N_ENVS * BATCH_SIMS],
    pending_actions: Tile[dtype, N_ENVS * BATCH_SIMS],
    # Prediction output [N_ENVS * BATCH_SIMS * PRED_OUT]
    pred_output: Tile[dtype, N_ENVS * BATCH_SIMS * PRED_OUT],
    # Rewards from env.step [N_ENVS * BATCH_SIMS]
    step_rewards: Tile[dtype, N_ENVS * BATCH_SIMS],
    # Search paths [N_ENVS * BATCH_SIMS * MAX_DEPTH]
    search_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    action_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS * BATCH_SIMS],
) where dtype.is_floating_point():
    """Fused: expand BATCH_SIMS nodes + negated backup + remove virtual losses.

    One thread per env. Processes all BATCH_SIMS expansions sequentially.
    Combines: store_child_state + expand_alphazero + backup_negated + virtual loss removal.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES

    comptime VIRTUAL_LOSS: Int = 3

    for s in range(BATCH_SIMS):
        var sim_off = e * BATCH_SIMS + s
        var parent = Int(rebind[Scalar[dtype]](pending_parents[sim_off]))
        var action = Int(rebind[Scalar[dtype]](pending_actions[sim_off]))
        var child_node_idx = Int(rebind[Scalar[dtype]](node_count[e]))

        if child_node_idx >= MAX_NODES:
            # Remove virtual loss even if we can't expand
            visit_count[tree_off + parent * ACT + action] = rebind[
                Scalar[dtype]
            ](visit_count[tree_off + parent * ACT + action]) - Scalar[dtype](
                VIRTUAL_LOSS
            )
            total_visits[tv_off + parent] = rebind[Scalar[dtype]](
                total_visits[tv_off + parent]
            ) - Scalar[dtype](VIRTUAL_LOSS)
            continue

        # ── Store child game state ──────────────────────────
        var exp_gs_off = sim_off * STATE_SIZE
        var child_gs_off = (
            e * MAX_NODES * STATE_SIZE + child_node_idx * STATE_SIZE
        )
        for i in range(STATE_SIZE):
            game_states[child_gs_off + i] = expansion_states[exp_gs_off + i]

        # ── Set parent reward from env.step ─────────────────
        reward_buf[tree_off + parent * ACT + action] = step_rewards[sim_off]

        # ── Set child prior from softmax ────────────────────
        var pred_off = sim_off * PRED_OUT
        var child_off = tree_off + child_node_idx * ACT
        var p_max = rebind[Scalar[dtype]](pred_output[pred_off])
        for a in range(1, ACT):
            var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
            if v > p_max:
                p_max = v
        var p_sum = Scalar[dtype](0.0)
        for a in range(ACT):
            var v = exp(
                rebind[Scalar[dtype]](pred_output[pred_off + a]) - p_max
            )
            prior[child_off + a] = v
            p_sum += v
        for a in range(ACT):
            prior[child_off + a] = (
                rebind[Scalar[dtype]](prior[child_off + a]) / p_sum
            )
            visit_count[child_off + a] = Scalar[dtype](0.0)
            total_value[child_off + a] = Scalar[dtype](0.0)
            reward_buf[child_off + a] = Scalar[dtype](0.0)
            child_idx[child_off + a] = Scalar[dtype](-1.0)
        total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

        # ── Link parent → child ─────────────────────────────
        child_idx[tree_off + parent * ACT + action] = Scalar[dtype](
            child_node_idx
        )
        node_count[e] = Scalar[dtype](child_node_idx + 1)

        # ── Decode leaf value ───────────────────────────────
        # leaf_value must be from the CHILD's perspective so that the
        # backup loop's first negation converts it to the parent's perspective.
        # Terminal reward is from the PARENT's perspective (player who moved),
        # so we negate it to get the child's (opponent's) perspective.
        var step_rew = rebind[Scalar[dtype]](step_rewards[sim_off])
        var abs_rew = step_rew if step_rew >= Scalar[dtype](0.0) else -step_rew
        var leaf_value: Scalar[dtype]
        if abs_rew > Scalar[dtype](0.5):
            leaf_value = -step_rew  # Terminal: negate (reward is parent's perspective)
        else:
            var raw_v = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
            # Clamp to prevent tanh overflow/saturation
            if raw_v > Scalar[dtype](10.0):
                raw_v = Scalar[dtype](10.0)
            elif raw_v < Scalar[dtype](-10.0):
                raw_v = Scalar[dtype](-10.0)
            var ev_p = exp(raw_v)
            var ev_n = exp(-raw_v)
            leaf_value = (ev_p - ev_n) / (
                ev_p + ev_n
            )  # Non-terminal: network estimate

        # ── Remove virtual loss ─────────────────────────────
        visit_count[tree_off + parent * ACT + action] = rebind[Scalar[dtype]](
            visit_count[tree_off + parent * ACT + action]
        ) - Scalar[dtype](VIRTUAL_LOSS)
        total_visits[tv_off + parent] = rebind[Scalar[dtype]](
            total_visits[tv_off + parent]
        ) - Scalar[dtype](VIRTUAL_LOSS)

        # ── Negated backup ──────────────────────────────────
        var value = leaf_value
        var path_off = sim_off * MAX_DEPTH
        var path_len = Int(rebind[Scalar[dtype]](path_lengths[sim_off]))

        for i in range(path_len):
            var idx = path_len - 1 - i
            var node = Int(rebind[Scalar[dtype]](search_paths[path_off + idx]))
            var act = Int(rebind[Scalar[dtype]](action_paths[path_off + idx]))

            value = -value  # Negate for zero-sum

            var na_off = tree_off + node * ACT + act
            visit_count[na_off] = rebind[Scalar[dtype]](
                visit_count[na_off]
            ) + Scalar[dtype](1.0)
            total_value[na_off] = (
                rebind[Scalar[dtype]](total_value[na_off]) + value
            )
            total_visits[tv_off + node] = rebind[Scalar[dtype]](
                total_visits[tv_off + node]
            ) + Scalar[dtype](1.0)

            var n_a = rebind[Scalar[dtype]](visit_count[na_off])
            var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
            if mean_q < rebind[Scalar[dtype]](min_q[e]):
                min_q[e] = mean_q
            if mean_q > rebind[Scalar[dtype]](max_q[e]):
                max_q[e] = mean_q


def gpu_mcts_batched_expand_backup_masked_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BATCH_SIMS: Int,
    PRED_OUT: Int,
    STATE_SIZE: Int,
    NEGATE_BACKUP: Bool,
    VALUE_SQUASH: Bool,
    dtype: DType,
    VIRTUAL_LOSS_VAL: Int = 3,
](
    # Node storage
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    reward_buf: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    # Game states
    game_states: Tile[dtype, N_ENVS * MAX_NODES * STATE_SIZE],
    expansion_states: Tile[dtype, N_ENVS * BATCH_SIMS * STATE_SIZE],
    # Pending expansions [N_ENVS * BATCH_SIMS]
    pending_parents: Tile[dtype, N_ENVS * BATCH_SIMS],
    pending_actions: Tile[dtype, N_ENVS * BATCH_SIMS],
    # Prediction output [N_ENVS * BATCH_SIMS * PRED_OUT]
    pred_output: Tile[dtype, N_ENVS * BATCH_SIMS * PRED_OUT],
    # Rewards from env.step [N_ENVS * BATCH_SIMS]
    step_rewards: Tile[dtype, N_ENVS * BATCH_SIMS],
    # Done flags from env.step [N_ENVS * BATCH_SIMS]. Both single-player
    # and two-player paths read this. Two-player MUST read it (not infer
    # terminal from |reward| > 0.5) because draw terminals carry reward 0
    # and would otherwise be treated as non-terminal — using network V
    # instead of the true draw value 0 (a major training-signal bug in
    # games with frequent draws like TTT / ConnectFour endgames).
    step_dones: Tile[dtype, N_ENVS * BATCH_SIMS],
    # Search paths [N_ENVS * BATCH_SIMS * MAX_DEPTH]
    search_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    action_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS * BATCH_SIMS],
    # Legal masks for expanded child positions [N_ENVS * BATCH_SIMS * ACT]
    expansion_legal_masks: Tile[dtype, N_ENVS * BATCH_SIMS * ACT],
    # Discount factor (used only by single-player backup; ignored when
    # NEGATE_BACKUP=True).
    gamma: Scalar[dtype],
) where dtype.is_floating_point():
    """Fused expand + backup + remove virtual losses — with child prior masking.

    Like gpu_mcts_batched_expand_backup_kernel but masks child priors with
    legal action masks (matching AlphaZero.jl / alpha-zero-general which mask
    priors at ALL nodes, not just root).

    Comptime branches on game family:

      NEGATE_BACKUP=True  (board games / two-player zero-sum):
        Leaf value:
          - terminal (|step_rew| > 0.5): leaf_value = -step_rew
          - non-terminal: leaf_value = tanh(raw_v) if VALUE_SQUASH else raw_v
        Backup recurrence: value = -value at each parent (perspective flip,
        no per-edge reward — board games are terminal-reward-only).

      NEGATE_BACKUP=False (single-player envs with per-step rewards):
        Leaf value:
          - terminal (step_dones[sim] >= 0.5): leaf_value = step_rew
          - non-terminal: leaf_value = step_rew + γ · V_leaf
            where V_leaf = tanh(raw_v) if VALUE_SQUASH else raw_v.
        Backup recurrence: value = edge_reward + γ · value at each parent,
        where `edge_reward` is read from `reward_buf` (set on prior expansions).

    One thread per env. Processes all BATCH_SIMS expansions sequentially.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES

    comptime VIRTUAL_LOSS: Int = VIRTUAL_LOSS_VAL

    for s in range(BATCH_SIMS):
        var sim_off = e * BATCH_SIMS + s
        var parent = Int(rebind[Scalar[dtype]](pending_parents[sim_off]))
        var action = Int(rebind[Scalar[dtype]](pending_actions[sim_off]))
        var child_node_idx = Int(rebind[Scalar[dtype]](node_count[e]))
        var sim_path_off = sim_off * MAX_DEPTH
        var sim_path_len = Int(
            rebind[Scalar[dtype]](path_lengths[sim_off])
        )

        if child_node_idx >= MAX_NODES:
            # Tree full — undo the virtual loss applied along the entire
            # selection path so the abandoned sim doesn't poison the tree.
            for i in range(sim_path_len):
                var pn = Int(rebind[Scalar[dtype]](
                    search_paths[sim_path_off + i]
                ))
                var pa = Int(rebind[Scalar[dtype]](
                    action_paths[sim_path_off + i]
                ))
                var pna = tree_off + pn * ACT + pa
                visit_count[pna] = rebind[Scalar[dtype]](
                    visit_count[pna]
                ) - Scalar[dtype](VIRTUAL_LOSS)
                total_visits[tv_off + pn] = rebind[Scalar[dtype]](
                    total_visits[tv_off + pn]
                ) - Scalar[dtype](VIRTUAL_LOSS)
            continue

        # ── Store child game state ──────────────────────────
        var exp_gs_off = sim_off * STATE_SIZE
        var child_gs_off = (
            e * MAX_NODES * STATE_SIZE + child_node_idx * STATE_SIZE
        )
        for i in range(STATE_SIZE):
            game_states[child_gs_off + i] = expansion_states[exp_gs_off + i]

        # ── Set parent reward from env.step ─────────────────
        reward_buf[tree_off + parent * ACT + action] = step_rewards[sim_off]

        # ── Set child prior from MASKED softmax ─────────────
        # Only legal actions get prior > 0 (matches AlphaZero.jl/alpha-zero-general)
        var pred_off = sim_off * PRED_OUT
        var mask_off = sim_off * ACT
        var child_off = tree_off + child_node_idx * ACT

        # Masked softmax: find max over legal actions only
        var p_max = Scalar[dtype](-1e18)
        for a in range(ACT):
            var legal = rebind[Scalar[dtype]](
                expansion_legal_masks[mask_off + a]
            )
            if legal > Scalar[dtype](0.5):
                var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
                if v > p_max:
                    p_max = v

        # Fallback if no legal actions (terminal position)
        if p_max < Scalar[dtype](-1e17):
            p_max = rebind[Scalar[dtype]](pred_output[pred_off])
            for a in range(1, ACT):
                var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
                if v > p_max:
                    p_max = v

        # Compute masked softmax + renormalize
        var p_sum = Scalar[dtype](0.0)
        for a in range(ACT):
            var legal = rebind[Scalar[dtype]](
                expansion_legal_masks[mask_off + a]
            )
            if legal > Scalar[dtype](0.5):
                var v = exp(
                    rebind[Scalar[dtype]](pred_output[pred_off + a]) - p_max
                )
                prior[child_off + a] = v
                p_sum += v
            else:
                prior[child_off + a] = Scalar[dtype](0.0)

        if p_sum > Scalar[dtype](1e-8):
            for a in range(ACT):
                prior[child_off + a] = (
                    rebind[Scalar[dtype]](prior[child_off + a]) / p_sum
                )
        else:
            # All masked — uniform over legal (fallback)
            var n_legal = Scalar[dtype](0.0)
            for a in range(ACT):
                if rebind[Scalar[dtype]](
                    expansion_legal_masks[mask_off + a]
                ) > Scalar[dtype](0.5):
                    n_legal += Scalar[dtype](1.0)
            if n_legal > Scalar[dtype](0.5):
                var inv_n = Scalar[dtype](1.0) / n_legal
                for a in range(ACT):
                    if rebind[Scalar[dtype]](
                        expansion_legal_masks[mask_off + a]
                    ) > Scalar[dtype](0.5):
                        prior[child_off + a] = inv_n

        for a in range(ACT):
            visit_count[child_off + a] = Scalar[dtype](0.0)
            total_value[child_off + a] = Scalar[dtype](0.0)
            reward_buf[child_off + a] = Scalar[dtype](0.0)
            child_idx[child_off + a] = Scalar[dtype](-1.0)
        total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

        # ── Link parent → child ─────────────────────────────
        child_idx[tree_off + parent * ACT + action] = Scalar[dtype](
            child_node_idx
        )
        node_count[e] = Scalar[dtype](child_node_idx + 1)

        # ── Decode leaf value ───────────────────────────────
        var step_rew = rebind[Scalar[dtype]](step_rewards[sim_off])
        var was_done = (
            rebind[Scalar[dtype]](step_dones[sim_off])
            > Scalar[dtype](0.5)
        )
        var leaf_value: Scalar[dtype]
        comptime if NEGATE_BACKUP:
            # Two-player: terminal detected via done flag (NOT |reward|>0.5
            # which mishandles draws where reward=0 + done=True). Leaf
            # value is the leaf-player's perspective: -step_rew works for
            # both win (rew=1 → leaf=-1, since the move ending the game
            # was the opponent's) and draw (rew=0 → leaf=0).
            if was_done:
                leaf_value = -step_rew
            else:
                var raw_v = rebind[Scalar[dtype]](
                    pred_output[pred_off + ACT]
                )
                comptime if VALUE_SQUASH:
                    if raw_v > Scalar[dtype](10.0):
                        raw_v = Scalar[dtype](10.0)
                    elif raw_v < Scalar[dtype](-10.0):
                        raw_v = Scalar[dtype](-10.0)
                    var ev_p = exp(raw_v)
                    var ev_n = exp(-raw_v)
                    leaf_value = (ev_p - ev_n) / (ev_p + ev_n)
                else:
                    leaf_value = raw_v
        else:
            # Single-player: terminal from explicit done flag; leaf value
            # accumulates step reward + discounted V_pred.
            if was_done:
                leaf_value = step_rew
            else:
                var raw_v = rebind[Scalar[dtype]](
                    pred_output[pred_off + ACT]
                )
                comptime if VALUE_SQUASH:
                    if raw_v > Scalar[dtype](10.0):
                        raw_v = Scalar[dtype](10.0)
                    elif raw_v < Scalar[dtype](-10.0):
                        raw_v = Scalar[dtype](-10.0)
                    var ev_p = exp(raw_v)
                    var ev_n = exp(-raw_v)
                    var v_pred = (ev_p - ev_n) / (ev_p + ev_n)
                    leaf_value = step_rew + gamma * v_pred
                else:
                    leaf_value = step_rew + gamma * raw_v

        # ── Backup + remove virtual loss along entire path ──
        # Virtual loss was applied at every (node, action) during selection
        # (canonical AlphaGo Zero); undo it at every step here while adding
        # the real +1 real visit. Net change per step = (1 - VIRTUAL_LOSS).
        var value = leaf_value
        var path_off = sim_off * MAX_DEPTH
        var path_len = Int(rebind[Scalar[dtype]](path_lengths[sim_off]))
        comptime VLOSS_DELTA: Int = 1 - VIRTUAL_LOSS

        for i in range(path_len):
            var idx = path_len - 1 - i
            var node = Int(rebind[Scalar[dtype]](search_paths[path_off + idx]))
            var act = Int(rebind[Scalar[dtype]](action_paths[path_off + idx]))

            comptime if NEGATE_BACKUP:
                value = -value

            var na_off = tree_off + node * ACT + act
            visit_count[na_off] = rebind[Scalar[dtype]](
                visit_count[na_off]
            ) + Scalar[dtype](VLOSS_DELTA)
            total_value[na_off] = (
                rebind[Scalar[dtype]](total_value[na_off]) + value
            )
            total_visits[tv_off + node] = rebind[Scalar[dtype]](
                total_visits[tv_off + node]
            ) + Scalar[dtype](VLOSS_DELTA)

            var n_a = rebind[Scalar[dtype]](visit_count[na_off])
            var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
            if mean_q < rebind[Scalar[dtype]](min_q[e]):
                min_q[e] = mean_q
            if mean_q > rebind[Scalar[dtype]](max_q[e]):
                max_q[e] = mean_q

            # Single-player: accumulate this edge's stored reward into the
            # value seen by the next ancestor up the path. (Edge reward was
            # written to reward_buf when this child was first expanded.)
            comptime if not NEGATE_BACKUP:
                if idx > 0:
                    var anc = Int(rebind[Scalar[dtype]](
                        search_paths[path_off + idx - 1]
                    ))
                    var anc_act = Int(rebind[Scalar[dtype]](
                        action_paths[path_off + idx - 1]
                    ))
                    var edge_rew = rebind[Scalar[dtype]](
                        reward_buf[tree_off + anc * ACT + anc_act]
                    )
                    value = edge_rew + gamma * value


# ═══════════════════════════════════════════════════════════════════════════
# MuZero Batched Fused Kernels (hidden states + dynamics network)
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_batched_select_and_build_dyn_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BATCH_SIMS: Int,
    LATENT: Int,
    DYN_IN: Int,
    dtype: DType,
    VIRTUAL_LOSS_VAL: Int = 3,
](
    # Node storage
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    # Hidden states
    hidden_states: Tile[dtype, N_ENVS * MAX_NODES * LATENT],
    # Output: batched dynamics inputs [N_ENVS * BATCH_SIMS * DYN_IN]
    dyn_inputs: Tile[dtype, N_ENVS * BATCH_SIMS * DYN_IN],
    # Output: pending info + paths [N_ENVS * BATCH_SIMS * ...]
    pending_parents: Tile[dtype, N_ENVS * BATCH_SIMS],
    pending_actions: Tile[dtype, N_ENVS * BATCH_SIMS],
    search_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    action_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS * BATCH_SIMS],
    # PUCT
    c_base: Scalar[dtype],
    c_init: Scalar[dtype],
) where dtype.is_floating_point():
    """Fused: select BATCH_SIMS leaves per env + build dynamics input.

    MuZero version: copies parent hidden state + one-hot action into dyn_input.
    Applies virtual losses for diversity.
    One thread per env.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var mn = rebind[Scalar[dtype]](min_q[e])
    var mx = rebind[Scalar[dtype]](max_q[e])
    var q_range = mx - mn

    comptime VIRTUAL_LOSS: Int = VIRTUAL_LOSS_VAL

    for s in range(BATCH_SIMS):
        var node_idx = 0
        var depth = 0
        var path_off = (e * BATCH_SIMS + s) * MAX_DEPTH
        search_paths[path_off] = Scalar[dtype](0.0)

        while depth < MAX_DEPTH - 1:
            var n_total = rebind[Scalar[dtype]](total_visits[tv_off + node_idx])
            var sqrt_total = sqrt(n_total + Scalar[dtype](1e-8))
            # PUCT exploration constant:
            # MuZero: c = log((1 + N + c_base) / c_base) + c_init
            # AlphaGo: c = c_init (when c_base = 0)
            var c: Scalar[dtype]
            if c_base > Scalar[dtype](0.5):
                c = log(
                    (Scalar[dtype](1.0) + n_total + c_base) / c_base
                ) + c_init
            else:
                c = c_init  # Constant c_puct (AlphaGo-style)

            var best_action = 0
            var best_score = Scalar[dtype](-1e18)
            for a in range(ACT):
                var na_off = tree_off + node_idx * ACT + a
                var n_a = rebind[Scalar[dtype]](visit_count[na_off])
                var q_val: Scalar[dtype]
                if n_a > Scalar[dtype](0.5):
                    q_val = rebind[Scalar[dtype]](total_value[na_off]) / n_a
                    if q_range > Scalar[dtype](1e-8):
                        q_val = (q_val - mn) / q_range
                else:
                    q_val = Scalar[dtype](0.0)
                var p = rebind[Scalar[dtype]](prior[na_off])
                var score = q_val + c * p * sqrt_total / (
                    Scalar[dtype](1.0) + n_a
                )
                if score > best_score:
                    best_score = score
                    best_action = a

            action_paths[path_off + depth] = Scalar[dtype](best_action)

            # Apply virtual loss along the WHOLE descent path (canonical
            # AlphaGo Zero), not only at the leaf parent. Removed during
            # backup so visit counts stay correct.
            var na_off = tree_off + node_idx * ACT + best_action
            visit_count[na_off] = rebind[Scalar[dtype]](
                visit_count[na_off]
            ) + Scalar[dtype](VIRTUAL_LOSS)
            total_visits[tv_off + node_idx] = rebind[Scalar[dtype]](
                total_visits[tv_off + node_idx]
            ) + Scalar[dtype](VIRTUAL_LOSS)

            var child = rebind[Scalar[dtype]](
                child_idx[tree_off + node_idx * ACT + best_action]
            )
            if child < Scalar[dtype](0.0):
                var sim_off = e * BATCH_SIMS + s
                pending_parents[sim_off] = Scalar[dtype](node_idx)
                pending_actions[sim_off] = Scalar[dtype](best_action)
                path_lengths[sim_off] = Scalar[dtype](depth + 1)

                # Build dynamics input: [hidden || one_hot_action]
                var dyn_off = sim_off * DYN_IN
                var h_off = e * MAX_NODES * LATENT + node_idx * LATENT
                for i in range(LATENT):
                    dyn_inputs[dyn_off + i] = hidden_states[h_off + i]
                for a in range(ACT):
                    dyn_inputs[dyn_off + LATENT + a] = Scalar[dtype](0.0)
                dyn_inputs[dyn_off + LATENT + best_action] = Scalar[dtype](1.0)
                break

            node_idx = Int(child)
            depth += 1
            search_paths[path_off + depth] = Scalar[dtype](node_idx)

        if depth >= MAX_DEPTH - 1:
            var sim_off = e * BATCH_SIMS + s
            pending_parents[sim_off] = Scalar[dtype](node_idx)
            pending_actions[sim_off] = Scalar[dtype](0)
            path_lengths[sim_off] = Scalar[dtype](depth + 1)
            var dyn_off = sim_off * DYN_IN
            var h_off = e * MAX_NODES * LATENT + node_idx * LATENT
            for i in range(LATENT):
                dyn_inputs[dyn_off + i] = hidden_states[h_off + i]
            for a in range(ACT):
                dyn_inputs[dyn_off + LATENT + a] = Scalar[dtype](0.0)


def gpu_mcts_batched_expand_backup_muzero_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BATCH_SIMS: Int,
    LATENT: Int,
    PRED_OUT: Int,
    DYN_OUT: Int,
    dtype: DType,
    VIRTUAL_LOSS_VAL: Int = 3,
](
    # Node storage
    visit_count: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_value: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    prior: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    reward_buf: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    child_idx: Tile[dtype, N_ENVS * MAX_NODES * ACT],
    total_visits: Tile[dtype, N_ENVS * MAX_NODES],
    node_count: Tile[dtype, N_ENVS],
    min_q: Tile[dtype, N_ENVS],
    max_q: Tile[dtype, N_ENVS],
    # Hidden states
    hidden_states: Tile[dtype, N_ENVS * MAX_NODES * LATENT],
    # Pending [N_ENVS * BATCH_SIMS]
    pending_parents: Tile[dtype, N_ENVS * BATCH_SIMS],
    pending_actions: Tile[dtype, N_ENVS * BATCH_SIMS],
    # Dynamics output [N_ENVS * BATCH_SIMS * DYN_OUT]
    dyn_output: Tile[dtype, N_ENVS * BATCH_SIMS * DYN_OUT],
    # Prediction output [N_ENVS * BATCH_SIMS * PRED_OUT]
    pred_output: Tile[dtype, N_ENVS * BATCH_SIMS * PRED_OUT],
    # Search paths [N_ENVS * BATCH_SIMS * MAX_DEPTH]
    search_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    action_paths: Tile[dtype, N_ENVS * BATCH_SIMS * MAX_DEPTH],
    path_lengths: Tile[dtype, N_ENVS * BATCH_SIMS],
    # Value support
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
    gamma: Scalar[dtype],
    # Whether to negate backup (two-player) — passed as Scalar[DType.bool]
    negate: Scalar[DType.bool],
) where dtype.is_floating_point():
    """Fused: extract hidden + expand + backup for BATCH_SIMS MuZero leaves.

    One thread per env. For each of BATCH_SIMS expansions:
    1. Extract hidden state from dynamics output → hidden_states pool
    2. Scale hidden state (MinMax)
    3. Decode reward from dynamics output
    4. Set child prior from prediction softmax
    5. Decode leaf value (scalar or categorical)
    6. Remove virtual losses
    7. Backup (standard or negated)
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES

    comptime VIRTUAL_LOSS: Int = VIRTUAL_LOSS_VAL
    comptime NUM_VAL_BINS = PRED_OUT - ACT
    comptime NUM_REW_BINS = DYN_OUT - LATENT

    for s in range(BATCH_SIMS):
        var sim_off = e * BATCH_SIMS + s
        var parent = Int(rebind[Scalar[dtype]](pending_parents[sim_off]))
        var action = Int(rebind[Scalar[dtype]](pending_actions[sim_off]))
        var child_node_idx = Int(rebind[Scalar[dtype]](node_count[e]))
        var sim_path_off = sim_off * MAX_DEPTH
        var sim_path_len = Int(
            rebind[Scalar[dtype]](path_lengths[sim_off])
        )

        if child_node_idx >= MAX_NODES:
            # Tree full — undo virtual loss applied along the path.
            for i in range(sim_path_len):
                var pn = Int(rebind[Scalar[dtype]](
                    search_paths[sim_path_off + i]
                ))
                var pa = Int(rebind[Scalar[dtype]](
                    action_paths[sim_path_off + i]
                ))
                var pna = tree_off + pn * ACT + pa
                visit_count[pna] = rebind[Scalar[dtype]](
                    visit_count[pna]
                ) - Scalar[dtype](VIRTUAL_LOSS)
                total_visits[tv_off + pn] = rebind[Scalar[dtype]](
                    total_visits[tv_off + pn]
                ) - Scalar[dtype](VIRTUAL_LOSS)
            continue

        # 1. Extract hidden state from dyn_output
        var dyn_off = sim_off * DYN_OUT
        var child_h_off = e * MAX_NODES * LATENT + child_node_idx * LATENT
        for i in range(LATENT):
            hidden_states[child_h_off + i] = dyn_output[dyn_off + i]

        # 2. Scale hidden state (MinMax)
        var h_min = rebind[Scalar[dtype]](hidden_states[child_h_off])
        var h_max = h_min
        for i in range(1, LATENT):
            var v = rebind[Scalar[dtype]](hidden_states[child_h_off + i])
            if v < h_min:
                h_min = v
            if v > h_max:
                h_max = v
        var h_delta = h_max - h_min
        if h_delta > Scalar[dtype](1e-8):
            for i in range(LATENT):
                var v = rebind[Scalar[dtype]](hidden_states[child_h_off + i])
                hidden_states[child_h_off + i] = (v - h_min) / h_delta

        # 3. Decode reward
        var rew_decoded: Scalar[dtype]
        if NUM_REW_BINS == 1:
            var raw = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
            var e_pos = exp(raw)
            var e_neg = exp(-raw)
            rew_decoded = (e_pos - e_neg) / (e_pos + e_neg)
        else:
            var rew_max = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
            for i in range(1, NUM_REW_BINS):
                var v = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                if v > rew_max:
                    rew_max = v
            var rew_se = Scalar[dtype](0.0)
            for i in range(NUM_REW_BINS):
                rew_se += exp(
                    rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                    - rew_max
                )
            rew_decoded = Scalar[dtype](0.0)
            var rew_step = (v_max - v_min) / Scalar[dtype](NUM_REW_BINS - 1)
            for i in range(NUM_REW_BINS):
                var prob = (
                    exp(
                        rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                        - rew_max
                    )
                    / rew_se
                )
                rew_decoded += prob * (v_min + Scalar[dtype](i) * rew_step)
            # Inverse scalar transform: targets are h(r), so the network
            # output decodes to h(r); the backup recurrence
            # `value = reward + γ · value` is only valid in raw space.
            # h^{-1}(y) = sign(y) * (((sqrt(1+4ε(|y|+1+ε)) - 1)/(2ε))^2 - 1)
            var sign_r = Scalar[dtype](1.0) if rew_decoded >= Scalar[dtype](
                0.0
            ) else Scalar[dtype](-1.0)
            var abs_r = (
                rew_decoded if rew_decoded >= Scalar[dtype](0.0) else -rew_decoded
            )
            var eps_r = Scalar[dtype](0.001)
            var inner_r = sqrt(
                Scalar[dtype](1.0)
                + Scalar[dtype](4.0) * eps_r * (abs_r + Scalar[dtype](1.0) + eps_r)
            )
            var f_r = (inner_r - Scalar[dtype](1.0)) / (
                Scalar[dtype](2.0) * eps_r
            )
            rew_decoded = sign_r * (f_r * f_r - Scalar[dtype](1.0))
        reward_buf[tree_off + parent * ACT + action] = rew_decoded

        # 4. Set child prior from prediction softmax
        var pred_off = sim_off * PRED_OUT
        var child_off = tree_off + child_node_idx * ACT
        var p_max = rebind[Scalar[dtype]](pred_output[pred_off])
        for a in range(1, ACT):
            var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
            if v > p_max:
                p_max = v
        var p_sum = Scalar[dtype](0.0)
        for a in range(ACT):
            var v = exp(
                rebind[Scalar[dtype]](pred_output[pred_off + a]) - p_max
            )
            prior[child_off + a] = v
            p_sum += v
        for a in range(ACT):
            prior[child_off + a] = (
                rebind[Scalar[dtype]](prior[child_off + a]) / p_sum
            )
            visit_count[child_off + a] = Scalar[dtype](0.0)
            total_value[child_off + a] = Scalar[dtype](0.0)
            reward_buf[child_off + a] = Scalar[dtype](0.0)
            child_idx[child_off + a] = Scalar[dtype](-1.0)
        total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

        # Link
        child_idx[tree_off + parent * ACT + action] = Scalar[dtype](
            child_node_idx
        )
        node_count[e] = Scalar[dtype](child_node_idx + 1)

        # 5. Decode leaf value
        var leaf_value: Scalar[dtype]
        if NUM_VAL_BINS == 1:
            var raw_v = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
            if raw_v > Scalar[dtype](10.0):
                raw_v = Scalar[dtype](10.0)
            elif raw_v < Scalar[dtype](-10.0):
                raw_v = Scalar[dtype](-10.0)
            var ev_p = exp(raw_v)
            var ev_n = exp(-raw_v)
            leaf_value = (ev_p - ev_n) / (ev_p + ev_n)
        else:
            var val_max = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
            for i in range(1, NUM_VAL_BINS):
                var v = rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                if v > val_max:
                    val_max = v
            var val_se = Scalar[dtype](0.0)
            for i in range(NUM_VAL_BINS):
                val_se += exp(
                    rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                    - val_max
                )
            var val_step = (v_max - v_min) / Scalar[dtype](NUM_VAL_BINS - 1)
            leaf_value = Scalar[dtype](0.0)
            for i in range(NUM_VAL_BINS):
                var prob = (
                    exp(
                        rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                        - val_max
                    )
                    / val_se
                )
                leaf_value += prob * (v_min + Scalar[dtype](i) * val_step)
            # Inverse scalar transform: pred net is trained on h(V), so
            # the categorical decode recovers h(V) (transformed space).
            # Convert to raw V before entering the backup recurrence.
            var sign_v = Scalar[dtype](1.0) if leaf_value >= Scalar[dtype](
                0.0
            ) else Scalar[dtype](-1.0)
            var abs_v = (
                leaf_value if leaf_value >= Scalar[dtype](0.0) else -leaf_value
            )
            var eps_v = Scalar[dtype](0.001)
            var inner_v = sqrt(
                Scalar[dtype](1.0)
                + Scalar[dtype](4.0) * eps_v * (abs_v + Scalar[dtype](1.0) + eps_v)
            )
            var f_v = (inner_v - Scalar[dtype](1.0)) / (
                Scalar[dtype](2.0) * eps_v
            )
            leaf_value = sign_v * (f_v * f_v - Scalar[dtype](1.0))

        # 6. Backup (negated or standard) + remove virtual loss along path.
        # Virtual loss was applied at every (node, action) during selection;
        # subtract it while adding the +1 real visit. Net per step = (1 -
        # VIRTUAL_LOSS) on visit counts.
        var value = leaf_value
        var path_off_s = sim_path_off
        var path_len = sim_path_len
        comptime VLOSS_DELTA: Int = 1 - VIRTUAL_LOSS

        for i in range(path_len):
            var idx = path_len - 1 - i
            var node = Int(
                rebind[Scalar[dtype]](search_paths[path_off_s + idx])
            )
            var act = Int(rebind[Scalar[dtype]](action_paths[path_off_s + idx]))

            if Bool(negate):
                value = -value
            else:
                value = (
                    rebind[Scalar[dtype]](
                        reward_buf[tree_off + node * ACT + act]
                    )
                    + gamma * value
                )

            var na_off = tree_off + node * ACT + act
            visit_count[na_off] = rebind[Scalar[dtype]](
                visit_count[na_off]
            ) + Scalar[dtype](VLOSS_DELTA)
            total_value[na_off] = (
                rebind[Scalar[dtype]](total_value[na_off]) + value
            )
            total_visits[tv_off + node] = rebind[Scalar[dtype]](
                total_visits[tv_off + node]
            ) + Scalar[dtype](VLOSS_DELTA)

            var n_a = rebind[Scalar[dtype]](visit_count[na_off])
            var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
            if mean_q < rebind[Scalar[dtype]](min_q[e]):
                min_q[e] = mean_q
            if mean_q > rebind[Scalar[dtype]](max_q[e]):
                max_q[e] = mean_q


def gpu_mcts_build_dyn_input_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    DYN_IN: Int,
    dtype: DType,
](
    dyn_input: Tile[dtype, N_ENVS * DYN_IN],
    hidden_states: Tile[dtype, N_ENVS * MAX_NODES * LATENT],
    pending_parent: Tile[dtype, N_ENVS],
    pending_action: Tile[dtype, N_ENVS],
) where dtype.is_floating_point():
    """Build dynamics input [hidden || one_hot_action] for each env's pending expansion.

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var action = Int(rebind[Scalar[dtype]](pending_action[e]))
    var h_off = e * MAX_NODES * LATENT + parent * LATENT
    var d_off = e * DYN_IN

    # Copy hidden state
    for i in range(LATENT):
        dyn_input[d_off + i] = hidden_states[h_off + i]

    # One-hot action
    for a in range(ACT):
        dyn_input[d_off + LATENT + a] = Scalar[dtype](0.0)
    dyn_input[d_off + LATENT + action] = Scalar[dtype](1.0)


def gpu_mcts_copy_pred_input_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    LATENT: Int,
    dtype: DType,
](
    pred_input: Tile[dtype, N_ENVS * LATENT],
    hidden_states: Tile[dtype, N_ENVS * MAX_NODES * LATENT],
    node_count: Tile[dtype, N_ENVS],
    dyn_output: Tile[dtype, N_ENVS * (LATENT + LATENT)],
) where dtype.is_floating_point():
    """Copy newly expanded child hidden states to prediction input buffer.

    The child's hidden state was written to hidden_states by the expand kernel.
    We need to copy it to pred_input for the prediction forward call.
    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    # The child index is node_count - 1 (expand kernel already incremented)
    var child_idx_val = Int(rebind[Scalar[dtype]](node_count[e])) - 1
    if child_idx_val < 0:
        return

    var h_off = e * MAX_NODES * LATENT + child_idx_val * LATENT
    var p_off = e * LATENT

    for i in range(LATENT):
        pred_input[p_off + i] = hidden_states[h_off + i]


# ═══════════════════════════════════════════════════════════════════════════
# GPU MCTS State
# ═══════════════════════════════════════════════════════════════════════════


struct GPUMCTSState[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    STATE_SIZE: Int = 0,
    BATCH_SIMS: Int = 8,
](Movable):
    """GPU-resident MCTS tree storage for n_envs parallel searches.

    All tree data lives on GPU. No CPU↔GPU transfer during search.

    When STATE_SIZE > 0 (AlphaZero mode), stores actual game states
    per tree node for expansion via true game rules.
    """

    comptime PRED_OUT: Int = Self.ACT + Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS

    # Node data [N_ENVS × MAX_NODES × ACT]
    var visit_count: DeviceBuffer[dtype]
    var total_value: DeviceBuffer[dtype]
    var prior: DeviceBuffer[dtype]
    var reward: DeviceBuffer[dtype]
    var child_idx: DeviceBuffer[dtype]

    # Per-node scalar [N_ENVS × MAX_NODES]
    var total_visits: DeviceBuffer[dtype]

    # Hidden state pool [N_ENVS × MAX_NODES × LATENT] (MuZero mode)
    var hidden_states: DeviceBuffer[dtype]

    # Game state pool [N_ENVS × MAX_NODES × STATE_SIZE] (AlphaZero mode)
    # Stores actual game states for expansion via env.step()
    var game_states: DeviceBuffer[dtype]

    # Staging buffer for env.step expansion [N_ENVS × STATE_SIZE]
    var expansion_states: DeviceBuffer[dtype]
    # Obs output from env.step [N_ENVS × OBS_DIM] — reuses pred_input
    # Legal masks from env.step [N_ENVS × ACT]
    var expansion_legal_masks: DeviceBuffer[dtype]

    # Per-env scalars [N_ENVS]
    var node_count: DeviceBuffer[dtype]
    var min_q: DeviceBuffer[dtype]
    var max_q: DeviceBuffer[dtype]

    # Selection output
    var pending_parent: DeviceBuffer[dtype]
    var pending_action: DeviceBuffer[dtype]
    var search_paths: DeviceBuffer[dtype]  # [N_ENVS × MAX_DEPTH]
    var action_paths: DeviceBuffer[dtype]  # [N_ENVS × MAX_DEPTH]
    var path_lengths: DeviceBuffer[dtype]
    var leaf_values: DeviceBuffer[dtype]

    # Batched network I/O
    var dyn_input: DeviceBuffer[dtype]  # [N_ENVS × DYN_IN]
    var dyn_output: DeviceBuffer[dtype]  # [N_ENVS × DYN_OUT]
    var pred_input: DeviceBuffer[dtype]  # [N_ENVS × LATENT]
    var pred_output: DeviceBuffer[dtype]  # [N_ENVS × PRED_OUT]

    # Final output
    var actions_out: DeviceBuffer[dtype]  # [N_ENVS]
    var policies_out: DeviceBuffer[dtype]  # [N_ENVS × ACT]

    # Legal mask (for self-play / board games)
    var legal_masks: DeviceBuffer[dtype]  # [N_ENVS × ACT]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime NODE_ACT_SIZE = Self.N_ENVS * Self.MAX_NODES * Self.ACT
        comptime NODE_SIZE = Self.N_ENVS * Self.MAX_NODES
        comptime HIDDEN_SIZE = Self.N_ENVS * Self.MAX_NODES * Self.LATENT

        self.visit_count = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.total_value = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.prior = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.reward = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.child_idx = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.total_visits = ctx.enqueue_create_buffer[dtype](NODE_SIZE)
        self.hidden_states = ctx.enqueue_create_buffer[dtype](HIDDEN_SIZE)
        self.node_count = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.min_q = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.max_q = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)

        # Batched pending/path buffers [N_ENVS * BATCH_SIMS * ...]
        comptime BS = Self.BATCH_SIMS
        self.pending_parent = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * BS)
        self.pending_action = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * BS)
        self.search_paths = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * MAX_DEPTH
        )
        self.action_paths = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * MAX_DEPTH
        )
        self.path_lengths = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * BS)
        self.leaf_values = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * BS)

        self.dyn_input = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.DYN_IN
        )
        self.dyn_output = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.DYN_OUT
        )
        self.pred_input = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.LATENT
        )
        self.pred_output = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.PRED_OUT
        )

        self.actions_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.policies_out = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )
        self.legal_masks = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )

        # Game state storage (AlphaZero mode: STATE_SIZE > 0)
        comptime GS_SIZE = Self.N_ENVS * Self.MAX_NODES * Self.STATE_SIZE
        self.game_states = ctx.enqueue_create_buffer[dtype](
            GS_SIZE if GS_SIZE > 0 else 1
        )
        comptime EXP_SIZE = Self.N_ENVS * BS * Self.STATE_SIZE
        self.expansion_states = ctx.enqueue_create_buffer[dtype](
            EXP_SIZE if EXP_SIZE > 0 else 1
        )
        self.expansion_legal_masks = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.ACT
        )

    def zero_tree(self, ctx: DeviceContext) raises:
        """Zero all tree node data via GPU unsafe_memset/fill.

        Must be called before gpu_mcts_init_root_kernel to replace the
        serial zeroing loop that was previously in the kernel.
        """
        ctx.enqueue_memset(self.visit_count, 0)
        ctx.enqueue_memset(self.total_value, 0)
        ctx.enqueue_memset(self.reward, 0)
        ctx.enqueue_memset(self.total_visits, 0)
        self.child_idx.enqueue_fill(Scalar[dtype](-1.0))

    def __init__(out self, *, deinit move: Self):
        self.visit_count = move.visit_count^
        self.total_value = move.total_value^
        self.prior = move.prior^
        self.reward = move.reward^
        self.child_idx = move.child_idx^
        self.total_visits = move.total_visits^
        self.hidden_states = move.hidden_states^
        self.game_states = move.game_states^
        self.expansion_states = move.expansion_states^
        self.expansion_legal_masks = move.expansion_legal_masks^
        self.node_count = move.node_count^
        self.min_q = move.min_q^
        self.max_q = move.max_q^
        self.pending_parent = move.pending_parent^
        self.pending_action = move.pending_action^
        self.search_paths = move.search_paths^
        self.action_paths = move.action_paths^
        self.path_lengths = move.path_lengths^
        self.leaf_values = move.leaf_values^
        self.dyn_input = move.dyn_input^
        self.dyn_output = move.dyn_output^
        self.pred_input = move.pred_input^
        self.pred_output = move.pred_output^
        self.actions_out = move.actions_out^
        self.policies_out = move.policies_out^
        self.legal_masks = move.legal_masks^
