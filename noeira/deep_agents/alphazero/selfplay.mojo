"""AlphaZero self-play training driver — ties search → record → outcome-z →
replay → train into one host-orchestrated loop (GPU MCTS + GPU net + GPU env).

One `run_alphazero_selfplay` call drives `iterations` self-play *moves* across
`N_ENVS` parallel games:

  1. `search_gpu_alphazero` (true game rules) → action + visit-count policy + root value.
  2. record (canonical_obs, policy) into the per-env in-progress trajectory.
  3. step every game by its chosen action (`env.step_kernel_gpu`).
  4. on a finished game, assign value targets z and flush (obs, [π | z]) to replay:
       * strict-alternation zero-sum ⇒ no absolute player index needed — the last
         mover gets z=+1 on a win, signs alternate backward by step parity
         (`z_k = +1 if (L-1-k) even else -1`); a draw gives z=0 for every step.
         (canonical obs + canonical z = the AlphaZero training convention.)
  5. selective-reset finished games, re-extract obs/legal.
  6. once replay ≥ BATCH, run `train_per_iter` graph train steps (the bit-identical
     CPU/GPU AZ loss graph) and update the net.

Returns the last observed mean train loss (0.0 if training never triggered).

Storage surface: the AZ loss runs on the storage ComputeGraph (net threaded as a
forward/vjp external arg; `set_input` takes a `Tensor`; Adam via `begin_step` +
`for_each_param`). The per-env trajectory is an owned `List` (RAII); the env /
MCTS device buffers stay category-B raw `DeviceBuffer`s. The training batch is
bridged into storage Tensors by `sample_batch_tensors` + `upload`.
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
from noeira.planners.tree_search import (
    GenericGPUMCTS,
    AlphaGoPUCT,
    DirichletNoise,
    SelfPlay,
)

from .loss_ops import AZLossOp
from ..zero.mcts_adapters import AZPredGPU, AZEnvGPU
from ..zero.gpu_example_replay import GpuMCTSExampleReplay
from noeira.core.tiles import tile


def run_alphazero_selfplay[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    N_ENVS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
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
    comptime MCTS = GenericGPUMCTS[
        N_ENVS,
        ACT,
        OBS,
        1,
        MAX_NODES,
        NUM_SIMS,
        1,
        AlphaGoPUCT[1.0],
        DirichletNoise[0.25, 0.25],
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
    # Fully device-resident replay: obs / packed-target / per-env trajectory all
    # stay on the GPU (no per-step obs/policy D2H, no per-train-step batch H2D).
    var replay = GpuMCTSExampleReplay[OBS, ACT, CAP, N_ENVS, MAX_TRAJ](ctx)

    # ── Env / MCTS device buffers (category-B; RAII DeviceBuffer) ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done = ctx.enqueue_create_buffer[DT](N_ENVS)
    var term = ctx.enqueue_create_buffer[DT](N_ENVS)
    var obs_next = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    ctx.synchronize()

    # ── Train-batch storage Tensors (device-resident; the nn surface) ──
    var obs_t = Tensor.alloc_gpu(ctx, BATCH * OBS)
    var tgt_t = Tensor.alloc_gpu(ctx, BATCH * W)
    var loss_t = Tensor.alloc_gpu(ctx, BATCH)
    var grad_t = Tensor.alloc(BATCH)
    for i in range(BATCH):
        grad_t.data[i] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    grad_t.upload(ctx)  # constant 1/BATCH grad seed, uploaded once

    # ── Initialize all games ──
    ENV.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var last_loss: Float64 = 0.0

    for it in range(iterations):
        # 1. MCTS search across all envs. Net runs in eval mode so any
        #    BatchNorm (CNN / ResNet torsos) uses running stats during the
        #    single-position inference of MCTS expansion (no-op for the MLP).
        net.set_attr["training"](Scalar[DT](0.0))
        var pred = AZPredGPU[OBS, ACT, NET].make(net)
        var env_ad = AZEnvGPU[ENV, STATE, OBS, ACT]()
        var root_obs = tile[N_ENVS, OBS](obs_dev)
        var root_legal = tile[N_ENVS * ACT](legal_dev)
        mcts.search_gpu_alphazero[type_of(pred), type_of(env_ad)](
            ctx,
            pred,
            env_ad,
            root_obs,
            states,
            root_legal,
            rng_seed=seed + UInt64(it),
        )

        # 2. Record this move's root obs + visit policy into each env's open
        #    trajectory — device→device, no D2H.
        replay.record_step_gpu(obs_dev, mcts.policies_out)

        # 3. Step every game by its chosen action.
        ENV.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx,
            states,
            mcts.actions_out,
            rew,
            done,
            term,
            obs_next,
            legal_next,
        )

        # 4. Flush finished games into the ring with in-kernel value targets z
        #    (the replay reads back only the tiny done/rew vectors for control
        #    flow; the trajectory→ring copies are device→device).
        replay.flush_finished_gpu(done, rew)

        # 5. Reset finished games, refresh obs/legal for the next move.
        ENV.selective_reset_kernel_gpu[N_ENVS, STATE](
            ctx, states, done, rng_seed=seed + UInt64(it)
        )
        ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()

        # 6. Train. Net back in train mode so BatchNorm uses batch stats and
        #    updates its running averages (no-op for the MLP).
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
