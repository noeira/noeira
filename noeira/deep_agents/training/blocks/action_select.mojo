"""Shared off-policy batched action-selection policy bodies.

ONE copy of the post-warmup policy path of `select_action_batched` that the
off-policy trainers previously carried as near-identical private blocks
(~70 lines × 6 packages, differing only in field names). Two flavours:

  * `select_deterministic_batched` — deterministic (Tanh-bounded) actor +
    Gaussian exploration noise + clamp (ddpg, td3). The actor output is fed
    raw (NOT scaled by action_scale — legacy parity); action_scale only
    bounds the clamp.
  * `select_squashed_batched` — actor → RSample (squash + log-prob) → drop
    the log-prob column + clamp (sac, redq, redq_ofe, mbpo).

Both handle the CPU and GPU targets (`target` comptime param): CPU bridges
the driver's LayoutTensor obs into the owned scratch Tensor the storage
`actor.forward` consumes; GPU copies device→device and stays sync-free.
The warmup-uniform branch stays in each trainer (it precedes this call).
"""

from max.gpu.host import DeviceContext
from std.random import random_float64

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.nn.primitives.rsample import RSample
from noeira.nn.random.box_muller import (
    box_muller_normal,
    box_muller_normal_gpu,
)

from .action_kernels import (
    offpolicy_warmup_uniform_kernel,
    offpolicy_copy2d_kernel,
    offpolicy_noise_clamp_kernel,
    offpolicy_clamp_action_kernel,
)
from noeira.core.tiles import Tile


def warmup_uniform_batched[
    target: StaticString, N_ENVS: Int, ACT: Int
](
    action: Tile[DT, N_ENVS, ACT],
    action_scale: Scalar[DT],
    ctx: Optional[DeviceContext],
    mut warmup_rng_seed: UInt64,
    mut warmup_rng_offset: UInt64,
) raises:
    """Warmup: Uniform(-action_scale, +action_scale) actions. CPU uses the
    host RNG (matching the historical trainers); GPU launches the shared
    Philox warmup kernel and advances the caller's offset."""
    comptime if target == "cpu":
        for env in range(N_ENVS):
            for j in range(ACT):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action[env, j] = u * action_scale
    else:
        var c = ctx.value()
        comptime tot = N_ENVS * ACT
        c.enqueue_function[offpolicy_warmup_uniform_kernel[N_ENVS, ACT]](
            action,
            action_scale,
            warmup_rng_seed,
            warmup_rng_offset,
            grid_dim=(tot + TPB - 1) // TPB,
            block_dim=TPB,
        )
        warmup_rng_offset += UInt64(N_ENVS * ACT * 2)


def select_deterministic_batched[
    A: Module,
    target: StaticString,
    N_ENVS: Int,
    OBS: Int,
    ACT: Int,
](
    mut actor: A,
    mut ob_scr: Tensor,
    mut ao_scr: Tensor,
    mut noise_scr: Tensor,
    obs: Tile[DT, N_ENVS, OBS],
    action: Tile[DT, N_ENVS, ACT],
    sigma: Scalar[DT],
    action_scale: Scalar[DT],
    ctx: Optional[DeviceContext],
    mut noise_rng_seed: UInt64,
    mut noise_rng_offset: UInt64,
) raises:
    """`action = clamp(actor(obs) + N(0, sigma), ±action_scale)`."""
    comptime if target == "cpu":
        # Bridge LayoutTensor obs → owned Tensor (storage actor.forward
        # wants a Tensor).
        ob_scr.ensure(N_ENVS * OBS)
        for env in range(N_ENVS):
            for d in range(OBS):
                ob_scr.data[env * OBS + d] = rebind[Scalar[DT]](obs[env, d])
        ao_scr.ensure(N_ENVS * ACT)
        noise_scr.ensure(N_ENVS * ACT)
        call_forward["cpu", N_ENVS](
            actor, TensorRefs[A.ARITY](ob_scr), ao_scr
        )
        box_muller_normal(noise_scr.data.unsafe_ptr(), N_ENVS * ACT)
        for env in range(N_ENVS):
            for j in range(ACT):
                var a = (
                    ao_scr.data[env * ACT + j]
                    + noise_scr.data[env * ACT + j] * sigma
                )
                if a > action_scale:
                    a = action_scale
                elif a < -action_scale:
                    a = -action_scale
                action[env, j] = a
    else:
        # Bridge the driver's device obs view → owned device scratch, run
        # actor on device, fill device box-muller noise, then noise+clamp
        # into `action`.
        var c = ctx.value()
        ob_scr.ensure_gpu(c, N_ENVS * OBS)
        ao_scr.ensure_gpu(c, N_ENVS * ACT)
        noise_scr.ensure_gpu(c, N_ENVS * ACT)
        comptime tot_obs = N_ENVS * OBS
        c.enqueue_function[offpolicy_copy2d_kernel[N_ENVS, OBS]](
            obs,
            ob_scr.tt["gpu", N_ENVS, OBS](),
            grid_dim=(tot_obs + TPB - 1) // TPB,
            block_dim=TPB,
        )
        call_forward["gpu", N_ENVS](
            actor, TensorRefs[A.ARITY](ob_scr), ao_scr, ctx
        )
        comptime tot_act = N_ENVS * ACT
        # box-muller fills the noise scratch (ACT-packed, flat); take a 1-D
        # device view and pass its concrete-origin `.ptr`
        # (box_muller_normal_gpu rebuilds the view).
        var noise_flat = noise_scr.tt["gpu", tot_act]()
        box_muller_normal_gpu[tot_act](
            c, noise_flat.ptr, noise_rng_seed, noise_rng_offset
        )
        noise_rng_offset += UInt64(((tot_act + 1) // 2) * 2)
        c.enqueue_function[offpolicy_noise_clamp_kernel[N_ENVS, ACT]](
            ao_scr.tt["gpu", N_ENVS, ACT](),
            noise_scr.tt["gpu", N_ENVS, ACT](),
            action,
            sigma,
            action_scale,
            grid_dim=(tot_act + TPB - 1) // TPB,
            block_dim=TPB,
        )


def select_squashed_batched[
    A: Module,
    target: StaticString,
    N_ENVS: Int,
    OBS: Int,
    ACT: Int,
](
    mut actor: A,
    mut sel: RSample[ACT],
    mut ob_scr: Tensor,
    mut ao_scr: Tensor,
    mut alp_scr: Tensor,
    obs: Tile[DT, N_ENVS, OBS],
    action: Tile[DT, N_ENVS, ACT],
    action_scale: Scalar[DT],
    ctx: Optional[DeviceContext],
) raises:
    """`action = clamp(rsample(actor(obs)).action, ±action_scale)` — the
    rsample output row is [action(ACT), log_prob], log-prob dropped."""
    comptime if target == "cpu":
        # Bridge LayoutTensor obs → owned Tensor; storage actor.forward
        # wants a Tensor.
        ob_scr.ensure(N_ENVS * OBS)
        for env in range(N_ENVS):
            for d in range(OBS):
                ob_scr.data[env * OBS + d] = rebind[Scalar[DT]](obs[env, d])
        ao_scr.ensure(N_ENVS * 2 * ACT)
        alp_scr.ensure(N_ENVS * (ACT + 1))
        call_forward["cpu", N_ENVS](
            actor, TensorRefs[A.ARITY](ob_scr), ao_scr
        )
        call_forward["cpu", N_ENVS](sel, TensorRefs[1](ao_scr), alp_scr)
        for env in range(N_ENVS):
            for j in range(ACT):
                var a = alp_scr.data[env * (ACT + 1) + j]
                if a > action_scale:
                    a = action_scale
                elif a < -action_scale:
                    a = -action_scale
                action[env, j] = a
    else:
        # Bridge the driver's device obs view → owned device scratch, run
        # actor → rsample on device, then clamp the squashed action out.
        var c = ctx.value()
        ob_scr.ensure_gpu(c, N_ENVS * OBS)
        ao_scr.ensure_gpu(c, N_ENVS * 2 * ACT)
        alp_scr.ensure_gpu(c, N_ENVS * (ACT + 1))
        comptime tot_obs = N_ENVS * OBS
        c.enqueue_function[offpolicy_copy2d_kernel[N_ENVS, OBS]](
            obs,
            ob_scr.tt["gpu", N_ENVS, OBS](),
            grid_dim=(tot_obs + TPB - 1) // TPB,
            block_dim=TPB,
        )
        call_forward["gpu", N_ENVS](
            actor, TensorRefs[A.ARITY](ob_scr), ao_scr, ctx
        )
        call_forward["gpu", N_ENVS](sel, TensorRefs[1](ao_scr), alp_scr, ctx)
        comptime tot_act = N_ENVS * ACT
        c.enqueue_function[
            offpolicy_clamp_action_kernel[N_ENVS, ACT, ACT + 1]
        ](
            alp_scr.tt["gpu", N_ENVS, ACT + 1](),
            action,
            action_scale,
            grid_dim=(tot_act + TPB - 1) // TPB,
            block_dim=TPB,
        )
