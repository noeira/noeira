"""Shared off-policy action-selection GPU kernels.

ONE copy of the warmup-uniform / obs-copy / action-clamp kernels that every
off-policy trainer (ddpg / td3 / sac / redq / redq_ofe / mbpo) previously
carried as byte-identical private duplicates (docstring-only diffs — audited
2026-07). Two clamp flavours:

  * `offpolicy_noise_clamp_kernel` — deterministic (Tanh) actors adding
    exploration noise then clamping (ddpg, td3).
  * `offpolicy_clamp_action_kernel` — squashed rsample actors whose output
    row carries a trailing log-prob column that is dropped before clamping
    (sac, redq, redq_ofe, mbpo; `ALP` defaults to `ACT + 1`).
"""

from max.gpu import global_idx
from std.random.philox import Random as PhiloxRandom

from noeira.nn.constants import DT
from noeira.core.tiles import Tile


def offpolicy_warmup_uniform_kernel[
    N_ENVS: Int, ACT: Int
](
    action_dest: Tile[DT, N_ENVS, ACT],
    action_scale: Scalar[DT],
    seed: UInt64,
    offset_base: UInt64,
):
    """Per-lane Philox uniform → [N_ENVS, ACT] of Uniform(-scale, +scale)."""
    var i = Int(global_idx.x)
    var total = N_ENVS * ACT
    if i >= total:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var s = Scalar[DT](2.0) * Scalar[DT](u) - Scalar[DT](1.0)
    action_dest[i // ACT, i % ACT] = s * action_scale


def offpolicy_copy2d_kernel[
    N_ENVS: Int, D: Int
](
    src: Tile[DT, N_ENVS, D],
    dst: Tile[DT, N_ENVS, D],
):
    """`dst[e,d] = src[e,d]` — bridge the driver's obs view into the trainer's
    owned device scratch the storage actor.forward consumes."""
    var i = Int(global_idx.x)
    var total = N_ENVS * D
    if i < total:
        dst[i // D, i % D] = rebind[Scalar[DT]](src[i // D, i % D])


def offpolicy_noise_clamp_kernel[
    N_ENVS: Int, ACT: Int
](
    ao: Tile[DT, N_ENVS, ACT],
    noise: Tile[DT, N_ENVS, ACT],
    action_out: Tile[DT, N_ENVS, ACT],
    sigma: Scalar[DT],
    action_scale: Scalar[DT],
):
    """`action_out = clamp(ao + noise·sigma, ±scale)` per lane. `ao` is the
    deterministic actor output (Tanh-bounded, ACT-wide); `noise` the
    contiguous box-muller fill."""
    var i = Int(global_idx.x)
    var total = N_ENVS * ACT
    if i >= total:
        return
    var e = i // ACT
    var j = i % ACT
    var a = rebind[Scalar[DT]](ao[e, j]) + rebind[Scalar[DT]](noise[e, j]) * sigma
    if a > action_scale:
        a = action_scale
    elif a < -action_scale:
        a = -action_scale
    action_out[e, j] = a


def offpolicy_clamp_action_kernel[
    N_ENVS: Int, ACT: Int, ALP: Int = ACT + 1
](
    alp: Tile[DT, N_ENVS, ALP],
    action: Tile[DT, N_ENVS, ACT],
    scale: Scalar[DT],
):
    """`action[e,j] = clamp(alp[e,j], ±scale)` — drop the trailing log-prob
    column of the rsample output and clamp the squashed action."""
    var i = Int(global_idx.x)
    var total = N_ENVS * ACT
    if i < total:
        var e = i // ACT
        var j = i % ACT
        var a = rebind[Scalar[DT]](alp[e, j])
        if a > scale:
            a = scale
        elif a < -scale:
            a = -scale
        action[e, j] = a
