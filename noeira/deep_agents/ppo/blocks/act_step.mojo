"""PPOActStep — N_ENVS-batched action selection (actor sample + critic eval).

Per-step entry point. Runs:
  1. actor.forward[BATCH=N_ENVS] on `state.ob1` → `state.ao1`.
  2. box_muller_normal → `state.z` (N_ENVS * ACT noise samples).
  3. For each env, each act dim: clamped log_std, sample, clamp, log_prob.
  4. critic.forward[BATCH=N_ENVS] on `state.ob1` → `state.v1`.
  5. Writes env-ready actions; caches per-env (sample, log_prob, value).

Per-env caches (cached_action / cached_log_prob / cached_value) live as
storage Tensors sized N_ENVS, consumed by the next `PPORecordStep.step`.

Greedy variant: deterministic — uses mu directly, no sampling, doesn't
touch the cache. N=1 greedy path exposed via list-based wrapper for
single-env eval.

STORAGE migration: nets are storage `Module`s (`forward[target, B, POLICY](
TensorRefs[1](ob1), ao1, ctx)`). The obs/output staging works on the storage
tensors' host `.data` (sanctioned host loops). On GPU `ob1.upload(ctx)` stages
H2D, the actor/critic forward runs on device, then `ao1.download`/`v1.download`
read the result back on host for the sampling walk.
"""

from std.math import exp as fexp
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.random.box_muller import box_muller_normal
from ...training.onpolicy_state import OnPolicyState


comptime LOG_2PI: Scalar[DT] = 1.8378770664093453
comptime EPS_STD: Scalar[DT] = 1e-6
comptime LOG_STD_MIN_F: Scalar[DT] = -5.0
comptime LOG_STD_MAX_F: Scalar[DT] = 2.0


def _clamp_log_std(ls: Scalar[DT]) -> Scalar[DT]:
    if ls < LOG_STD_MIN_F:
        return LOG_STD_MIN_F
    elif ls > LOG_STD_MAX_F:
        return LOG_STD_MAX_F
    return ls


struct PPOActStep[
    OBS_: Int,
    ACT_: Int,
    ACTOR: Module,
    CRITIC: Module,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_

    def __init__(out self):
        pass

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOActStep: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error("PPOActStep.make[target='gpu']: ctx required")
        return Self()

    def step[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        mut actor: Self.ACTOR,
        mut critic: Self.CRITIC,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_scale: Scalar[DT],
    ) raises:
        """Sample N_ENVS PPO actions. Reads N_ENVS × OBS from obs_ptr,
        writes N_ENVS × ACT env-ready (clamped, scaled) actions into
        action_ptr, caches per-env (sample, log_prob, value) into state
        for the upcoming PPORecordStep.

        Both pointers are host-side (rollout buffer is host-only on
        GPU train_target — see OnPolicyState docstring). On GPU,
        actor + critic forward run on device via H2D/D2H staging."""
        # Stage N_ENVS × OBS into the host mirror of ob1 (index the storage
        # tensor's `.data` List directly; obs_ptr is the driver trait ABI).
        for e in range(N_ENVS):
            for d in range(Self.OBS):
                state.ob1.data[e * Self.OBS + d] = obs_ptr[unsafe_offset=e * Self.OBS + d]

        comptime if target == "gpu":
            var ctx = state.ctx.value()
            state.ob1.upload(ctx)
            call_forward[target, N_ENVS, POLICY=POLICY](
                actor, TensorRefs[Self.ACTOR.ARITY](state.ob1), state.ao1, state.ctx
            )
            call_forward[target, N_ENVS, POLICY=POLICY](
                critic, TensorRefs[Self.CRITIC.ARITY](state.ob1), state.v1, state.ctx
            )
            state.ao1.download(ctx)
            state.v1.download(ctx)
        else:
            call_forward[target, N_ENVS, POLICY=POLICY](
                actor, TensorRefs[Self.ACTOR.ARITY](state.ob1), state.ao1, state.ctx
            )
            call_forward[target, N_ENVS, POLICY=POLICY](
                critic, TensorRefs[Self.CRITIC.ARITY](state.ob1), state.v1, state.ctx
            )

        # Host RNG fill of the noise buffer (box_muller takes a raw pointer —
        # a genuine pointer-API boundary, the one sanctioned unsafe here).
        box_muller_normal(state.z.data.unsafe_ptr(), N_ENVS * Self.ACT)
        # Sampling walk reads/writes the host `.data` Lists directly.
        ref ao1 = state.ao1.data
        ref v1 = state.v1.data
        ref z = state.z.data
        ref ca = state.cached_action.data
        ref clp = state.cached_log_prob.data
        ref cval = state.cached_value.data
        for e in range(N_ENVS):
            var lp_total: Scalar[DT] = 0.0
            for j in range(Self.ACT):
                var mu = ao1[e * 2 * Self.ACT + j]
                var ls = _clamp_log_std(ao1[e * 2 * Self.ACT + Self.ACT + j])
                var sample = mu + fexp(ls) * z[e * Self.ACT + j]
                ca[e * Self.ACT + j] = sample
                var env_a = sample
                if env_a > action_scale:
                    env_a = action_scale
                elif env_a < -action_scale:
                    env_a = -action_scale
                action_ptr[unsafe_offset=e * Self.ACT + j] = env_a
                var zz = (sample - mu) / (fexp(ls) + EPS_STD)
                lp_total += Scalar[DT](-0.5) * (
                    LOG_2PI + Scalar[DT](2.0) * ls + zz * zz
                )
            clp[e] = lp_total
            cval[e] = v1[e]

    def step_greedy_batched[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        mut actor: Self.ACTOR,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_scale: Scalar[DT],
    ) raises:
        """Deterministic N_ENVS-wide actions for eval — the actor's mean,
        clamped like `step_greedy_n1`. Does not touch the rollout cache."""
        for e in range(N_ENVS):
            for d in range(Self.OBS):
                state.ob1.data[e * Self.OBS + d] = obs_ptr[
                    unsafe_offset = e * Self.OBS + d
                ]
        comptime if target == "gpu":
            var ctx = state.ctx.value()
            state.ob1.upload(ctx)
            call_forward[target, N_ENVS, POLICY=POLICY](
                actor, TensorRefs[Self.ACTOR.ARITY](state.ob1), state.ao1, state.ctx
            )
            state.ao1.download(ctx)
        else:
            call_forward[target, N_ENVS, POLICY=POLICY](
                actor, TensorRefs[Self.ACTOR.ARITY](state.ob1), state.ao1, state.ctx
            )
        ref ao1 = state.ao1.data
        for e in range(N_ENVS):
            for j in range(Self.ACT):
                var env_a = ao1[e * 2 * Self.ACT + j]
                if env_a > action_scale:
                    env_a = action_scale
                elif env_a < -action_scale:
                    env_a = -action_scale
                action_ptr[unsafe_offset = e * Self.ACT + j] = env_a

    def step_greedy_n1[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        mut actor: Self.ACTOR,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        action_scale: Scalar[DT],
    ) raises:
        """Deterministic single-env action for eval — uses mu of env 0
        directly. Does not touch the cache (eval bypasses the rollout
        buffer). Always BATCH=1 even when state was sized for N_ENVS>1
        (writes only the first OBS/ACT slot of ob1/ao1)."""
        for d in range(Self.OBS):
            state.ob1.data[d] = obs[d]
        comptime if target == "gpu":
            var ctx = state.ctx.value()
            state.ob1.upload(ctx)
            call_forward[target, 1, POLICY=POLICY](
                actor, TensorRefs[Self.ACTOR.ARITY](state.ob1), state.ao1, state.ctx
            )
            state.ao1.download(ctx)
        else:
            call_forward[target, 1, POLICY=POLICY](
                actor, TensorRefs[Self.ACTOR.ARITY](state.ob1), state.ao1, state.ctx
            )
        ref ao1 = state.ao1.data
        for j in range(Self.ACT):
            var env_a = ao1[j]
            if env_a > action_scale:
                env_a = action_scale
            elif env_a < -action_scale:
                env_a = -action_scale
            action_out[j] = env_a
