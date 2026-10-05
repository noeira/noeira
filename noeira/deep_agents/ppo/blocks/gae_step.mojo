"""PPOGAEStep — per-env Generalized Advantage Estimation over the rollout.

V of every row's NEXT (pre-reset) observation via one critic forward over
`state.next_obs_buf`, then GAE backward for each env independently (T-major
layout — strided reads at gap N_ENVS), with truncation bootstrapped from the
ending episode's own final state and the recursion cut at every episode end
(see `step`).

GPU path (hybrid N=1+): the critic forward on device, the values down, the
recurrence on host (sequential per env; a per-env parallel scan kernel adds no
value below very large N_ENVS). The device rollout's twin is
`device_rollout._gae_kernel`.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward, call_vjp
from ...training.onpolicy_state import OnPolicyState


struct PPOGAEStep[
    OBS_: Int,
    ROLLOUT_LEN_: Int,
    CRITIC: Module,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ROLLOUT_LEN = Self.ROLLOUT_LEN_

    def __init__(out self):
        pass

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOGAEStep: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def step[
        target: StaticString,
        ACT: Int,
        MINIBATCH: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, ACT, Self.ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        mut critic: Self.CRITIC,
        gamma: Scalar[DT],
        gae_lambda: Scalar[DT],
    ) raises:
        """V of every row's next observation (one critic forward over the
        rollout's `next_obs_buf`, BATCH = ROLLOUT_LEN * N_ENVS), then GAE
        backward per env over the T-major host-side buffers:

            nv_t  = (1 - term_t) * V(next_obs_t)   if the episode ended at t
                                                   or t is the rollout's last
                    val[t + 1]                     otherwise (the same state)
            gae_t = delta_t + gamma * lambda * (1 - done_t) * gae_{t+1}

        A TRUNCATED episode (done, not terminated) bootstraps from its OWN
        final state and a terminated one from 0; neither leaks into the next
        episode's row. (Before 2026-10-05 only `term_buf` was read: a
        truncation bootstrapped from the NEXT episode's first state and the
        recursion ran across the boundary.)

        GPU path: the next obs uploaded to their resident buffer, the critic
        forward on device, its values down; the recurrence runs on host."""
        comptime RN = Self.ROLLOUT_LEN * N_ENVS
        comptime if target == "gpu":
            var ctx = state.ctx.value()
            state.next_obs_buf.upload_resident(ctx)
            call_forward[target, RN, POLICY=POLICY](
                critic, TensorRefs[Self.CRITIC.ARITY](state.next_obs_buf),
                state.next_val_buf, state.ctx,
            )
            state.next_val_buf.download(ctx)
        else:
            call_forward[target, RN, POLICY=POLICY](
                critic, TensorRefs[Self.CRITIC.ARITY](state.next_obs_buf),
                state.next_val_buf, state.ctx,
            )

        # Per-env GAE backward pass over T-major rollout buffers (host-side
        # `.data` Lists, indexed directly — no raw pointers).
        # Layout: buf[t * N_ENVS + e] for time t, env e.
        ref nval = state.next_val_buf.data
        ref rew = state.rew_buf.data
        ref val = state.val_buf.data
        ref done = state.done_buf.data
        ref term = state.term_buf.data
        ref adv = state.adv_buf.data
        ref ret = state.ret_buf.data
        for e in range(N_ENVS):
            var last_gae: Scalar[DT] = 0.0
            for t in range(Self.ROLLOUT_LEN - 1, -1, -1):
                var idx = t * N_ENVS + e
                var ended = done[idx] > Scalar[DT](0.5)
                var nv: Scalar[DT]
                if ended or t == Self.ROLLOUT_LEN - 1:
                    nv = (Scalar[DT](1.0) - term[idx]) * nval[idx]
                else:
                    nv = val[(t + 1) * N_ENVS + e]
                var cont = Scalar[DT](0.0) if ended else Scalar[DT](1.0)
                var delta = rew[idx] + gamma * nv - val[idx]
                last_gae = delta + gamma * gae_lambda * cont * last_gae
                adv[idx] = last_gae
                ret[idx] = last_gae + val[idx]
