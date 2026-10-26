"""GPU PCN training step + workspace (Phase D).

`pc_module_train_one_batch_gpu` is the GPU mirror of
`pc_module_train_one_batch`: the settling loop runs on-device via
`PCTrainer.compute_grads_only_gpu`, fills `net.weights.grd` (device), then an
nn `Optimizer.step['gpu']` updates `net.weights.val`.

All matmuls in the GPU settling go through `linalg.matmul` (`max_matmul`),
matching nn's convention — PCN's custom 2×2 register-tiled MMA fallback is
dead on the default path (`PCBlock.USE_MAX_KERNELS=True`). See
`pc_block.mojo`'s `predict_gpu` / `pull_back_gpu` / `weight_grad_gpu`.

`PCGpuWorkspace` owns the settling's device working buffers (latents, μ/ε,
a_below, z_below, dx). Allocate it ONCE and reuse it across steps so the
training loop does no per-step device allocation (and stays CUDA-graph
capturable). This is the GPU analogue of moving the working buffers onto the
holder as persistent storage.
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.optimizer.optimizer import Optimizer

from .predictive_model import PCBlockTrait
from .pc_sequential import PCSequential
from .pc_trainer import PCTrainer
from .pc_module import PCModule
from noeira.core.tiles import Tile, tile


@fieldwise_init
struct PCGpuWorkspace[BATCH: Int, *BLOCKS: PCBlockTrait](Movable):
    """Persistent device working buffers for the GPU settling loop. Sized
    once from the net's compile-time dims; reused every training step."""

    comptime NET = PCSequential[*Self.BLOCKS]

    var latents_b: DeviceBuffer[DT]
    var mu_eps_b: DeviceBuffer[DT]
    var a_below_b: DeviceBuffer[DT]
    var z_below_b: DeviceBuffer[DT]
    var dx_b: DeviceBuffer[DT]

    @staticmethod
    def make(ctx: DeviceContext) raises -> Self:
        return Self(
            latents_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.LATENT_DIM
            ),
            mu_eps_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.SCRATCH_OUT_DIM
            ),
            a_below_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.SCRATCH_IN_DIM
            ),
            z_below_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.SCRATCH_IN_DIM
            ),
            dx_b=ctx.enqueue_create_buffer[DT](
                Self.BATCH * Self.NET.LATENT_DIM
            ),
        )


def pc_module_train_one_batch_gpu[
    BATCH: Int, OPT: Optimizer, *BLOCKS: PCBlockTrait
](
    ctx: DeviceContext,
    mut net: PCModule[*BLOCKS],
    mut opt: OPT,
    mut ws: PCGpuWorkspace[BATCH, *BLOCKS],
    x_in: Tile[DT, BATCH, PCSequential[*BLOCKS].IN_DIM],
    y_target: Tile[DT, BATCH, PCSequential[*BLOCKS].OUT_DIM],
    T_infer: Int,
    lr_x: Scalar[DT],
) raises:
    comptime NET = PCSequential[*BLOCKS]

    var params = tile[NET.PARAM_SIZE](net.weights.val.dev.value())
    var grads = tile[NET.PARAM_SIZE](net.weights.grd.dev.value())

    var latents = tile[BATCH, NET.LATENT_DIM](ws.latents_b)
    var mu_eps = tile[BATCH, NET.SCRATCH_OUT_DIM](ws.mu_eps_b)
    var a_below = tile[BATCH, NET.SCRATCH_IN_DIM](ws.a_below_b)
    var z_below = tile[BATCH, NET.SCRATCH_IN_DIM](ws.z_below_b)
    var dx = tile[BATCH, NET.LATENT_DIM](ws.dx_b)

    # 1. GPU settling fills `grads` (= net.weights.grd) with +∂E/∂W.
    PCTrainer[*BLOCKS].compute_grads_only_gpu[BATCH](
        ctx,
        params,
        grads,
        latents,
        mu_eps,
        a_below,
        z_below,
        dx,
        x_in,
        y_target,
        T_infer,
        lr_x,
    )

    # 2. nn optimizer consumes net.weights.grd, updates net.weights.val.
    opt.step["gpu", PCModule[*BLOCKS]](net, ctx)


def pc_module_train_one_batch_gpu[
    BATCH: Int, OPT: Optimizer, *BLOCKS: PCBlockTrait
](
    ctx: DeviceContext,
    mut net: PCModule[*BLOCKS],
    mut opt: OPT,
    x_in: Tile[DT, BATCH, PCSequential[*BLOCKS].IN_DIM],
    y_target: Tile[DT, BATCH, PCSequential[*BLOCKS].OUT_DIM],
    T_infer: Int,
    lr_x: Scalar[DT],
) raises:
    """Convenience overload: allocate a one-shot workspace per call. Prefer
    the workspace-taking overload in a training loop to avoid per-step
    device allocations."""
    var ws = PCGpuWorkspace[BATCH, *BLOCKS].make(ctx)
    pc_module_train_one_batch_gpu[BATCH](
        ctx, net, opt, ws, x_in, y_target, T_infer, lr_x
    )
    ctx.synchronize()
