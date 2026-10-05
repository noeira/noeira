"""PPORecordStep — push N_ENVS transitions into the rollout buffer (STORAGE).

Reads N_ENVS-wide (obs, reward, done, next_obs) from the driver + the
per-step caches (cached_action / cached_log_prob / cached_value) filled
by PPOActStep, writes into the [ROLLOUT_LEN, N_ENVS]-sized buffers at
state.rollout_idx (all envs at row `rollout_idx`), then advances the
single shared cursor.

Layout: T-major. Slot at time t, env e:
  obs_buf[ (t*N_ENVS + e) * OBS + d ] = obs_e[d]
  act_buf[ (t*N_ENVS + e) * ACT + j ] = cached_action_e[j]
  olp_buf[ t*N_ENVS + e ] = cached_log_prob_e
  ... etc

STORAGE migration: rollout buffers live host-side only (the GPU train_target
only uploads the gathered minibatch), so every access here indexes the storage
tensors' host `.data` Lists directly (no raw pointers). The driver-supplied
obs/reward/done/next_obs pointers are the (Pointer) trait ABI.

Driver contract:
  - `action_ptr` arg is the env-ready action vector — IGNORED (the
    rollout uses cached unbounded action for PPO math).
  - `done_ptr` is treated as truncation by default. Real terminals
    need `mark_terminal(env_idx)`.
  - `next_obs_ptr` (the PRE-RESET obs after the step) is stored per row in
    `next_obs_buf`: GAE bootstraps a truncated episode from it.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from ...training.onpolicy_state import OnPolicyState


struct PPORecordStep[
    OBS_: Int,
    ACT_: Int,
    ROLLOUT_LEN_: Int,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime ROLLOUT_LEN = Self.ROLLOUT_LEN_

    def __init__(out self):
        pass

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPORecordStep: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def step[
        target: StaticString,
        MINIBATCH: Int,
        N_ENVS: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        done_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Push N_ENVS transitions into rollout buffers at row
        state.rollout_idx, advance the single shared cursor by 1."""
        var t = state.rollout_idx
        if t >= Self.ROLLOUT_LEN:
            return
        # Rollout buffers always live host-side (GPU train_target only
        # uploads the gathered minibatch), so we index the storage tensors'
        # host `.data` Lists directly — no raw pointers. The driver-supplied
        # obs/reward/done pointers are the (Pointer) trait ABI.
        ref obs_buf = state.obs_buf.data
        ref act_buf = state.act_buf.data
        ref olp_buf = state.olp_buf.data
        ref val_buf = state.val_buf.data
        ref rew_buf = state.rew_buf.data
        ref done_buf = state.done_buf.data
        ref next_buf = state.next_obs_buf.data
        ref ca = state.cached_action.data
        ref clp = state.cached_log_prob.data
        ref cval = state.cached_value.data
        # T-major layout: row `t` holds all N_ENVS rows back-to-back.
        var row_base = t * N_ENVS
        for e in range(N_ENVS):
            for d in range(Self.OBS):
                obs_buf[(row_base + e) * Self.OBS + d] = obs_ptr[unsafe_offset=e * Self.OBS + d]
            for j in range(Self.ACT):
                act_buf[(row_base + e) * Self.ACT + j] = ca[e * Self.ACT + j]
            olp_buf[row_base + e] = clp[e]
            val_buf[row_base + e] = cval[e]
            rew_buf[row_base + e] = reward_ptr[unsafe_offset=e]
            done_buf[row_base + e] = done_ptr[unsafe_offset=e]
            # term_buf stays at 0 unless caller marks terminal explicitly.
            for d in range(Self.OBS):
                next_buf[(row_base + e) * Self.OBS + d] = next_obs_ptr[unsafe_offset=e * Self.OBS + d]
        state.rollout_idx += 1

    def mark_terminal[
        target: StaticString,
        MINIBATCH: Int,
        N_ENVS: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        env_idx: Int,
    ) raises:
        """Mark the last-recorded transition for `env_idx` as a real
        terminal (V=0 bootstrap). No-op if the cursor is at 0."""
        if state.rollout_idx > 0:
            var row = (state.rollout_idx - 1) * N_ENVS + env_idx
            state.term_buf.data[row] = Scalar[DT](1.0)

    def reset_rollout[
        target: StaticString,
        MINIBATCH: Int,
        N_ENVS: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
    ) raises:
        """Called by the trainer once the K-epoch update has fired —
        zeros the term buffer and resets the cursor."""
        state.rollout_idx = 0
        ref term = state.term_buf.data
        for k in range(Self.ROLLOUT_LEN * N_ENVS):
            term[k] = Scalar[DT](0.0)
