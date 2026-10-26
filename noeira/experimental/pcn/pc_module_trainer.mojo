"""Drive nn Adam over a PCModule — `pc_module_train_one_batch` (Phase A spike).

One Bogacz-canonical PC training step on the nn storage layer:
  1. PCN's own settling loop fills the weight gradient (`net.weights.grd`)
     via the existing `PCTrainer.compute_grads_only` static path — the
     local error-minimization math is reused verbatim.
  2. nn `Adam.step` consumes that gradient and updates the weights
     (`net.weights.val`). No negation: `weight_grad` already stores the
     standard +∂E/∂W (see `pc_module.mojo` header).

Working buffers (latents, μ/ε, a_below, z_below, dx) are allocated per call
here — moving them onto the holder as `Scratch`/`Cache` is Phase B. The
weight slab is the storage that matters for the optimizer + checkpoint, and
that now lives in `net.weights` (a `Param`).
"""


from noeira.nn.constants import DT
from noeira.nn.optimizer.optimizer import Optimizer

from .predictive_model import PCBlockTrait
from .pc_sequential import PCSequential
from .pc_trainer import PCTrainer, PCTrainResult
from .pc_module import PCModule
from noeira.core.tiles import Tile, tile


def _zeroed(n: Int) -> List[Scalar[DT]]:
    var s = List[Scalar[DT]](capacity=n)
    for _ in range(n):
        s.append(Scalar[DT](0))
    return s^


def pc_module_train_one_batch[
    BATCH: Int, OPT: Optimizer, *BLOCKS: PCBlockTrait
](
    mut net: PCModule[*BLOCKS],
    mut opt: OPT,
    x_in: Tile[DT, BATCH, PCSequential[*BLOCKS].IN_DIM],
    y_target: Tile[DT, BATCH, PCSequential[*BLOCKS].OUT_DIM],
    T_infer: Int,
    lr_x: Scalar[DT],
) raises -> PCTrainResult:
    comptime NET = PCSequential[*BLOCKS]

    # Weight + grad views over the owned Param's CPU storage.
    var params = tile[NET.PARAM_SIZE](net.weights.val.data)
    var grads = tile[NET.PARAM_SIZE](net.weights.grd.data)

    # Per-call working buffers (Phase B → Scratch/Cache on the holder).
    var latents_s = _zeroed(BATCH * NET.LATENT_DIM)
    var mu_eps_s = _zeroed(BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_s = _zeroed(BATCH * NET.SCRATCH_IN_DIM)
    var z_below_s = _zeroed(BATCH * NET.SCRATCH_IN_DIM)
    var dx_s = _zeroed(BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](latents_s)
    var mu_eps = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_s)
    var a_below = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_s)
    var z_below = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_s)
    var dx = tile[BATCH, NET.LATENT_DIM](dx_s)

    # 1. Settling loop fills `grads` (= net.weights.grd) with +∂E/∂W.
    var result = PCTrainer[*BLOCKS].compute_grads_only[BATCH](
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

    # 2. nn optimizer (Adam/AdamW/…) consumes net.weights.grd, updates
    #    net.weights.val.
    opt.step["cpu", PCModule[*BLOCKS]](net)

    return result
