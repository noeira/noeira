"""Batched (N_ENVS at a time) action selection for TD-MPC2.

The single-env `TDMPC2Agent.select_action` stages ONE observation into a
Tensor, runs `encoder → policy → rsample` at batch 1, and downloads one
action. That is the right shape for the single-env driver and the wrong shape
for a batched one: N envs would mean N round trips per env-step, each with its
own kernel launches and its own D2H.

This module is the batched counterpart — one `encoder → policy → rsample`
pass over `[N_ENVS, ·]`, writing the action straight into the env's own action
slab. It is the TD-MPC2 sibling of
`deep_agents/training/blocks/action_select.mojo::select_squashed_batched`
(SAC/DDPG/TD3 share that one; TD-MPC2 needs its own because the actor is
`policy ∘ encoder`, not a single actor module), and it reuses that module's
copy/clamp kernels rather than re-declaring them.

MPC acting is NOT here: `MPPIGPUBatched` is already N_ENVS-batched, so the
driver calls `plan_gpu` directly with an `[N_ENVS, LATENT]` root and lets it
write the whole action slab.
"""

from max.gpu.host import DeviceContext
from std.math import tanh
from max.gpu import global_idx

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.nn.primitives.rsample import RSample

from noeira.deep_agents.training.blocks.action_kernels import (
    offpolicy_copy2d_kernel,
    offpolicy_clamp_action_kernel,
)
from noeira.core.tiles import Tile


def tdmpc2_greedy_action_kernel[
    N_ENVS: Int, ACT: Int, POL: Int
](
    pio: Tile[DT, N_ENVS, POL],
    action_out: Tile[DT, N_ENVS, ACT],
    action_scale: Scalar[DT],
):
    """`action = tanh(pio[:, :ACT]) · action_scale` — the deterministic head.

    `pio` is the policy's `[mean(ACT), log_std(ACT)]` row; greedy acting reads
    the mean only and squashes it, matching `select_action(explore=False)`.
    The trailing log_std columns are deliberately ignored (not zeroed — the
    scratch is overwritten next call)."""
    var i = Int(global_idx.x)
    var total = N_ENVS * ACT
    if i >= total:
        return
    var e = i // ACT
    var j = i % ACT
    action_out[e, j] = tanh(rebind[Scalar[DT]](pio[e, j])) * action_scale


def tdmpc2_select_action_batched[
    ENC_M: Module,
    POL_M: Module,
    target: StaticString,
    N_ENVS: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
](
    mut encoder: ENC_M,
    mut policy: POL_M,
    mut rs: RSample[ACT],
    mut ob_scr: Tensor,
    mut z_scr: Tensor,
    mut pio_scr: Tensor,
    mut alp_scr: Tensor,
    obs: Tile[DT, N_ENVS, OBS],
    action: Tile[DT, N_ENVS, ACT],
    action_scale: Scalar[DT],
    explore: Bool,
    ctx: Optional[DeviceContext],
) raises:
    """MPC-off acting for N envs: `a = π(encode(obs))`, written into `action`.

    `explore=True`  → rsample (squash + noise), the collection policy.
    `explore=False` → `tanh(mean)·scale`, the deterministic eval policy.

    Scratch is caller-owned (`ob_scr` [N,OBS], `z_scr` [N,LATENT], `pio_scr`
    [N,2·ACT], `alp_scr` [N,ACT+1]) so the driver allocates ONCE instead of
    per env-step — `enqueue_create_buffer` in a hot loop is a known way to
    balloon the process.

    `obs` and `action` are views over the ENV's slabs, so both must live on
    `target`; the driver enforces `env_target == train_target`.
    """
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime POL = 2 * ACT

    comptime if target == "cpu":
        ob_scr.ensure(N_ENVS * OBS)
        z_scr.ensure(N_ENVS * LATENT)
        pio_scr.ensure(N_ENVS * POL)
        alp_scr.ensure(N_ENVS * (ACT + 1))
        for e in range(N_ENVS):
            for d in range(OBS):
                ob_scr.data[e * OBS + d] = rebind[Scalar[DT]](obs[e, d])
        call_forward[target, N_ENVS](
        encoder, TensorRefs[ENC_M.ARITY](ob_scr), z_scr, ctx
    )
        call_forward[target, N_ENVS](
            policy, TensorRefs[POL_M.ARITY](z_scr), pio_scr, ctx
        )
        if explore:
            call_forward[target, N_ENVS](
                rs, TensorRefs[1](pio_scr), alp_scr, ctx
            )
            for e in range(N_ENVS):
                for j in range(ACT):
                    var a = alp_scr.data[e * (ACT + 1) + j]
                    if a > action_scale:
                        a = action_scale
                    elif a < -action_scale:
                        a = -action_scale
                    action[e, j] = a
        else:
            for e in range(N_ENVS):
                for j in range(ACT):
                    action[e, j] = (
                        tanh(pio_scr.data[e * POL + j]) * action_scale
                    )
    else:
        var c = ctx.value()
        ob_scr.ensure_gpu(c, N_ENVS * OBS)
        z_scr.ensure_gpu(c, N_ENVS * LATENT)
        pio_scr.ensure_gpu(c, N_ENVS * POL)
        alp_scr.ensure_gpu(c, N_ENVS * (ACT + 1))

        comptime tot_obs = N_ENVS * OBS
        c.enqueue_function[offpolicy_copy2d_kernel[N_ENVS, OBS]](
            obs,
            ob_scr.tt["gpu", N_ENVS, OBS](),
            grid_dim=(tot_obs + TPB - 1) // TPB,
            block_dim=TPB,
        )
        call_forward[target, N_ENVS](
        encoder, TensorRefs[ENC_M.ARITY](ob_scr), z_scr, ctx
    )
        call_forward[target, N_ENVS](
            policy, TensorRefs[POL_M.ARITY](z_scr), pio_scr, ctx
        )

        comptime tot_act = N_ENVS * ACT
        if explore:
            call_forward[target, N_ENVS](
                rs, TensorRefs[1](pio_scr), alp_scr, ctx
            )
            c.enqueue_function[
                offpolicy_clamp_action_kernel[N_ENVS, ACT, ACT + 1]
            ](
                alp_scr.tt["gpu", N_ENVS, ACT + 1](),
                action,
                action_scale,
                grid_dim=(tot_act + TPB - 1) // TPB,
                block_dim=TPB,
            )
        else:
            c.enqueue_function[
                tdmpc2_greedy_action_kernel[N_ENVS, ACT, POL]
            ](
                pio_scr.tt["gpu", N_ENVS, POL](),
                action,
                action_scale,
                grid_dim=(tot_act + TPB - 1) // TPB,
                block_dim=TPB,
            )


def tdmpc2_encode_batched[
    ENC_M: Module,
    target: StaticString,
    N_ENVS: Int,
    OBS: Int,
    LATENT: Int,
](
    mut encoder: ENC_M,
    mut ob_scr: Tensor,
    mut z_scr: Tensor,
    obs: Tile[DT, N_ENVS, OBS],
    ctx: Optional[DeviceContext],
) raises:
    """`z_scr ← encode(obs)` for N envs — the MPPI planner's root latents.

    Split out of `tdmpc2_select_action_batched` because the MPC path needs the
    latent and NOT the policy action (the planner asks the callback for policy
    actions itself, inside the rollout)."""
    comptime assert target == "gpu", (
        "tdmpc2_encode_batched: MPC planning is GPU-only"
    )
    var c = ctx.value()
    ob_scr.ensure_gpu(c, N_ENVS * OBS)
    z_scr.ensure_gpu(c, N_ENVS * LATENT)
    comptime tot_obs = N_ENVS * OBS
    c.enqueue_function[offpolicy_copy2d_kernel[N_ENVS, OBS]](
        obs,
        ob_scr.tt["gpu", N_ENVS, OBS](),
        grid_dim=(tot_obs + TPB - 1) // TPB,
        block_dim=TPB,
    )
    call_forward[target, N_ENVS](
        encoder, TensorRefs[ENC_M.ARITY](ob_scr), z_scr, ctx
    )
