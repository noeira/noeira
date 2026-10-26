"""Gumbel AlphaZero self-play training driver (GPU) — improved-policy targets.

The Gumbel-planner sibling of `selfplay.mojo::run_alphazero_selfplay`. Same
host-orchestrated loop (search → record → outcome-z → replay → graph train),
with the planner swapped to `GumbelGPUMCTS.search_gpu_alphazero` (Danihelka et
al. Gumbel AlphaZero): Gumbel-Top-k root candidates replace Dirichlet noise,
Sequential Halving replaces PUCT visit allocation, and the recorded policy
target is the **improved policy** ``softmax(logits + σ(completed_Q))`` rather
than visit counts. Serial sims — structurally immune to the frozen-tree
batched-leaf bias that retired `BATCH_SIMS > 1` on the classic orchestrator.

Actions are sampled host-side ∝ the improved policy (the legal mask is already
folded in), matching the validated EZv2 / MuZero Gumbel drivers. Returns the
last observed mean train loss (0.0 if training never triggered).

Storage surface: the AZ loss runs on the storage ComputeGraph and the replay is
the fully device-resident `GpuMCTSExampleReplay` (obs / target never leave the
GPU; the improved policy is pulled to host only to sample the action, which this
driver does on the host by design).
"""

from noeira.nn.core.param import walk_params
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.initializer import Zero
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import (
    InputSlot, Node, ExternalNode,
)
from noeira.core.env_traits import GPUTwoPlayerDiscreteEnv
from noeira.planners.tree_search import GumbelGPUMCTS, SelfPlay

from .loss_ops import AZLossOp
from ..zero.mcts_adapters import AZPredGPU, AZEnvGPU
from ..zero.gpu_example_replay import GpuMCTSExampleReplay
from noeira.core.tiles import tile


def run_alphazero_gumbel_selfplay[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    N_ENVS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    BATCH: Int,
    CAP: Int,
    MAX_TRAJ: Int,
](
    ctx: DeviceContext,
    mut net: NET,
    iterations: Int,
    learning_starts: Int = 0,
    train_per_iter: Int = 1,
    lr: Scalar[DT] = Scalar[DT](0.01),
    seed: UInt64 = 0,
) raises -> Float64:
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime W = NET.OUT_DIM  # ACT + 1
    comptime STATE = ENV.STATE_SIZE
    comptime MCTS = GumbelGPUMCTS[
        N_ENVS,
        ACT,
        OBS,
        1,
        MAX_NODES,
        MAX_K,
        NUM_SIMS,
        SelfPlay,
        STATE_SIZE=STATE,
    ]
    comptime Graph = ComputeGraph[
        InputSlot["obs", OBS],
        ExternalNode["pred", NET, "obs"],
        InputSlot["tgt", W],
        Node["loss", AZLossOp[ACT], "pred", "tgt"],
    ]

    var octx = Optional[DeviceContext](ctx)
    var opt = Adam(lr=lr)
    var graph = Graph.make["gpu", Zero](octx)
    var mcts = MCTS(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)
    var replay = GpuMCTSExampleReplay[OBS, ACT, CAP, N_ENVS, MAX_TRAJ](ctx)

    # ── Env / MCTS device buffers (category-B; RAII DeviceBuffer) ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var act_dev = ctx.enqueue_create_buffer[DT](N_ENVS)
    var rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done = ctx.enqueue_create_buffer[DT](N_ENVS)
    var term = ctx.enqueue_create_buffer[DT](N_ENVS)
    var obs_next = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)

    # ── Host staging: improved policy (action sampling) + chosen actions ──
    var pol_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var act_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    ctx.synchronize()

    # ── Train-batch storage Tensors (device-resident; the nn surface) ──
    var obs_t = Tensor.alloc_gpu(ctx, BATCH * OBS)
    var tgt_t = Tensor.alloc_gpu(ctx, BATCH * W)
    var loss_t = Tensor.alloc_gpu(ctx, BATCH)
    var grad_t = Tensor.alloc(BATCH)
    for i in range(BATCH):
        grad_t.data[i] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    grad_t.upload(ctx)

    # ── Initialize all games ──
    ENV.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var rng = seed ^ UInt64(0x9E3779B97F4A7C15)
    var last_loss: Float64 = 0.0

    for it in range(iterations):
        # 1. Gumbel AlphaZero search across all envs (net in eval mode).
        net.set_attr["training"](Scalar[DT](0.0))
        var pred = AZPredGPU[OBS, ACT, NET].make(net)
        var env_ad = AZEnvGPU[ENV, STATE, OBS, ACT]()
        var root_obs = tile[N_ENVS, OBS](obs_dev)
        mcts.search_gpu_alphazero[type_of(pred), type_of(env_ad)](
            ctx,
            pred,
            env_ad,
            root_obs,
            states,
            legal_dev,
            k_actual=MAX_K,
            rng_seed=UInt32((seed + UInt64(it)) & 0xFFFFFFFF),
        )

        # 2. Record root obs + IMPROVED policy into the device replay
        #    (device→device), and pull the policy to host for action sampling.
        var pol_view = mcts.policies_view()
        replay.record_step_gpu(obs_dev, pol_view)
        ctx.enqueue_copy(pol_h, pol_view)
        ctx.synchronize()

        # 3. Sample each env's action ∝ improved policy (legal-masked) and step
        #    every game. The improved policy IS the exploration — Gumbel root
        #    sampling already injected it, no temperature knob.
        for e in range(N_ENVS):
            rng = rng ^ (rng << 13)
            rng = rng ^ (rng >> 7)
            rng = rng ^ (rng << 17)
            var r = Float64(rng % UInt64(1_000_000)) / 1_000_000.0
            var cum = 0.0
            var a_sel = -1
            for a in range(ACT):
                var p = Float64(pol_h[e * ACT + a])
                cum += p
                if r <= cum and p > 0.0:
                    a_sel = a
                    break
            if a_sel < 0:  # numeric fallback: argmax
                var bv = -1.0
                for a in range(ACT):
                    var p = Float64(pol_h[e * ACT + a])
                    if p > bv:
                        bv = p
                        a_sel = a
            act_h[e] = Scalar[DT](a_sel)
        ctx.enqueue_copy(act_dev, act_h)
        ENV.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx,
            states,
            act_dev,
            rew,
            done,
            term,
            obs_next,
            legal_next,
        )

        # 4. Flush finished games into the ring with in-kernel value targets z
        #    (strict-alternation parity; same convention as the PUCT driver).
        replay.flush_finished_gpu(done, rew)

        # 5. Reset finished games, refresh obs/legal for the next move.
        ENV.selective_reset_kernel_gpu[N_ENVS, STATE](
            ctx, states, done, rng_seed=seed + UInt64(it)
        )
        ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()

        # 6. Train on (obs, [improved-π | z]) batches.
        if len(replay) >= BATCH and it >= learning_starts:
            net.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                replay.sample_batch_gpu[BATCH](obs_t, tgt_t)
                net.zero_grad["gpu"](octx)
                graph.set_input["obs", BATCH](obs_t, octx)
                graph.set_input["tgt", BATCH](tgt_t, octx)
                graph.forward[BATCH, "gpu"](loss_t, octx, net)
                graph.vjp[BATCH, "gpu"](grad_t, octx, net)
                opt.begin_step()
                walk_params["gpu"](net, opt, octx)
            loss_t.download(ctx)
            var ml: Float64 = 0.0
            for b in range(BATCH):
                ml += Float64(loss_t.data[b])
            last_loss = ml / Float64(BATCH)

    return last_loss
