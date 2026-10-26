"""PPODeviceRollout — the PPO rollout kept on the GPU (`DEVICE_ROLLOUT`).

The host-staged rollout copies the observation to the host every env step,
samples there with the host RNG, and records into host `.data` lists. On the
device path the same steps are kernels over the env's own device buffers, the
SAC way (`training/blocks/action_select.mojo`, `nn/random/box_muller.mojo`):

  - `copy_obs`        env obs → `state.ob1` (device), the forwards' input;
  - `sample_continuous` / `sample_discrete`
                      the policy sample from the actor output, written to the
                      env's action buffer AND the per-env caches (sample,
                      log p, V) — the host walk of `PPOActStep` /
                      `PPODiscreteActStep`, one thread per env;
  - `record`          rollout row `t` from the caches and the env's reward /
                      done / terminated / next-obs buffers (`PPORecordStep`'s
                      loop, `mark_terminal` folded in: term = terminated);
  - `gae`             the per-env backward GAE pass (`PPOGAEStep`), one thread
                      per env, after the trainer's critic forward over every
                      row's next obs (truncations bootstrap from their own
                      final state).

The randomness is Philox with the offset in a DEVICE buffer advanced by a
kernel (`box_muller_normal_gpu_dev` + `advance_rng_offset_kernel`), so the act
step stays capture-safe. It is a different stream from the host RNG the
host-staged path samples with: a device-rollout run is NOT bit-identical to a
host-rollout one (its eager and captured runs are).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import exp as fexp, log as flog
from std.random.philox import Random as PhiloxRandom

from noeira.nn.constants import DT, TPB
from noeira.nn.core.fill import fill_dev
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor
from noeira.nn.random.box_muller import (
    box_muller_normal_gpu_dev,
    advance_rng_offset_kernel,
)
from .act_step import LOG_2PI, EPS_STD, _clamp_log_std
from ...training.onpolicy_state import OnPolicyState
from noeira.core.tiles import Tile, tile


comptime _Ptr = Pointer[Scalar[DT], MutAnyOrigin]


def _copy_kernel[N: Int](
    src: Tile[DT, N],
    dst: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = src[i]


def _sample_continuous_kernel[N_ENVS: Int, ACT: Int](
    ao: Tile[DT, N_ENVS * 2 * ACT],
    v1: Tile[DT, N_ENVS],
    z: Tile[DT, N_ENVS * ACT],
    action: Tile[DT, N_ENVS * ACT],
    ca: Tile[DT, N_ENVS * ACT],
    clp: Tile[DT, N_ENVS],
    cval: Tile[DT, N_ENVS],
    action_scale: Scalar[DT],
):
    """`PPOActStep.step`'s sampling walk for env `e` (one thread)."""
    var e = Int(global_idx.x)
    if e >= N_ENVS:
        return
    var lp_total: Scalar[DT] = 0.0
    for j in range(ACT):
        var mu = rebind[Scalar[DT]](ao[e * 2 * ACT + j])
        var ls = _clamp_log_std(rebind[Scalar[DT]](ao[e * 2 * ACT + ACT + j]))
        var sample = mu + fexp(ls) * rebind[Scalar[DT]](z[e * ACT + j])
        ca[e * ACT + j] = sample
        var env_a = sample
        if env_a > action_scale:
            env_a = action_scale
        elif env_a < -action_scale:
            env_a = -action_scale
        action[e * ACT + j] = env_a
        var zz = (sample - mu) / (fexp(ls) + EPS_STD)
        lp_total += Scalar[DT](-0.5) * (
            LOG_2PI + Scalar[DT](2.0) * ls + zz * zz
        )
    clp[e] = lp_total
    cval[e] = v1[e]


def _sample_discrete_kernel[N_ENVS: Int, N: Int](
    logits: Tile[DT, N_ENVS * N],
    v1: Tile[DT, N_ENVS],
    offset: Tile[DType.uint64, 1],
    seed: UInt64,
    action: Tile[DT, N_ENVS],
    ca: Tile[DT, N_ENVS],
    clp: Tile[DT, N_ENVS],
    cval: Tile[DT, N_ENVS],
):
    """`PPODiscreteActStep.step`'s softmax + inverse-CDF sample for env `e`
    (one thread). The uniform is Philox(seed, subsequence=e, offset)."""
    var e = Int(global_idx.x)
    if e >= N_ENVS:
        return
    var base = e * N
    var max_l = rebind[Scalar[DT]](logits[base])
    for j in range(1, N):
        var lj = rebind[Scalar[DT]](logits[base + j])
        if lj > max_l:
            max_l = lj
    var sum_exp: Scalar[DT] = 0.0
    for j in range(N):
        sum_exp += fexp(rebind[Scalar[DT]](logits[base + j]) - max_l)
    var rng = PhiloxRandom(
        seed=seed,
        subsequence=UInt64(e),
        offset=rebind[UInt64](offset[0]),
    )
    var u = Scalar[DT](rng.step_uniform()[0])
    var cum: Scalar[DT] = 0.0
    var a_idx: Int = N - 1
    for j in range(N):
        var p_j = fexp(rebind[Scalar[DT]](logits[base + j]) - max_l) / sum_exp
        cum += p_j
        if u <= cum:
            a_idx = j
            break
    var log_p_a = (rebind[Scalar[DT]](logits[base + a_idx]) - max_l) - flog(
        sum_exp
    )
    ca[e] = Scalar[DT](a_idx)
    action[e] = Scalar[DT](a_idx)
    clp[e] = log_p_a
    cval[e] = v1[e]


def _record_kernel[N_ENVS: Int, OBS: Int, ACT: Int, RN: Int](
    t: Int32,
    ob1: Tile[DT, N_ENVS * OBS],
    ca: Tile[DT, N_ENVS * ACT],
    clp: Tile[DT, N_ENVS],
    cval: Tile[DT, N_ENVS],
    reward: Tile[DT, N_ENVS],
    done: Tile[DT, N_ENVS],
    terminated: Tile[DT, N_ENVS],
    next_obs: Tile[DT, N_ENVS * OBS],
    obs_buf: Tile[DT, RN * OBS],
    act_buf: Tile[DT, RN * ACT],
    olp_buf: Tile[DT, RN],
    val_buf: Tile[DT, RN],
    rew_buf: Tile[DT, RN],
    done_buf: Tile[DT, RN],
    term_buf: Tile[DT, RN],
    next_buf: Tile[DT, RN * OBS],
):
    """`PPORecordStep.step` for env `e` at row `t` (one thread), with the
    driver's `mark_terminal_env` folded in: term = (terminated > 0.5)."""
    var e = Int(global_idx.x)
    if e >= N_ENVS:
        return
    var row = Int(t) * N_ENVS + e
    for d in range(OBS):
        obs_buf[row * OBS + d] = ob1[e * OBS + d]
        next_buf[row * OBS + d] = next_obs[e * OBS + d]
    for j in range(ACT):
        act_buf[row * ACT + j] = ca[e * ACT + j]
    olp_buf[row] = clp[e]
    val_buf[row] = cval[e]
    rew_buf[row] = reward[e]
    done_buf[row] = done[e]
    term_buf[row] = (
        Scalar[DT](1.0)
        if rebind[Scalar[DT]](terminated[e]) > Scalar[DT](0.5)
        else Scalar[DT](0.0)
    )


def _gae_kernel[N_ENVS: Int, T: Int](
    nval: Tile[DT, T * N_ENVS],
    rew: Tile[DT, T * N_ENVS],
    val: Tile[DT, T * N_ENVS],
    done: Tile[DT, T * N_ENVS],
    term: Tile[DT, T * N_ENVS],
    adv: Tile[DT, T * N_ENVS],
    ret: Tile[DT, T * N_ENVS],
    gamma: Scalar[DT],
    gae_lambda: Scalar[DT],
):
    """`PPOGAEStep`'s per-env backward pass for env `e` (one thread): a row
    that ended (or the rollout's last) bootstraps from V(its own next obs),
    times (1 - term); the recursion stops at every episode end."""
    var e = Int(global_idx.x)
    if e >= N_ENVS:
        return
    var last_gae: Scalar[DT] = 0.0
    for t in range(T - 1, -1, -1):
        var idx = t * N_ENVS + e
        var ended = rebind[Scalar[DT]](done[idx]) > Scalar[DT](0.5)
        var nv: Scalar[DT]
        if ended or t == T - 1:
            nv = (Scalar[DT](1.0) - rebind[Scalar[DT]](term[idx])) * rebind[
                Scalar[DT]
            ](nval[idx])
        else:
            nv = rebind[Scalar[DT]](val[(t + 1) * N_ENVS + e])
        var cont = Scalar[DT](0.0) if ended else Scalar[DT](1.0)
        var vi = rebind[Scalar[DT]](val[idx])
        var delta = rebind[Scalar[DT]](rew[idx]) + gamma * nv - vi
        last_gae = delta + gamma * gae_lambda * cont * last_gae
        adv[idx] = last_gae
        ret[idx] = last_gae + vi


struct PPODeviceRollout(Defaultable & Movable & Deinitable):
    """The device rollout's own state: the Philox seed and its device offset
    (the trainer owns everything else, in `OnPolicyState`)."""

    var seed: UInt64
    var _offset: Optional[DeviceBuffer[DType.uint64]]

    def __init__(out self):
        self.seed = 0
        self._offset = None

    def ready(self) -> Bool:
        return Bool(self._offset)

    def setup(mut self, ctx: DeviceContext, seed: UInt64) raises:
        """Allocate the device RNG offset (0). Once, before the first act."""
        self.seed = seed
        var b = ctx.enqueue_create_buffer[DType.uint64](1)
        b.enqueue_fill(UInt64(0))
        self._offset = b^

    def _offset_lt(
        mut self,
    ) -> Tile[DType.uint64, 1]:
        return tile[1](self._offset.value())

    def copy_obs[
        OBS: Int, ACT: Int, RL: Int, MB: Int, N_ENVS: Int
    ](
        mut self,
        mut state: OnPolicyState[OBS, ACT, RL, MB, N_ENVS],
        obs_ptr: _Ptr,
    ) raises:
        comptime N = N_ENVS * OBS
        var c = state.ctx.value()
        c.enqueue_function[_copy_kernel[N]](
            tile[N](obs_ptr),
            state.ob1.tt["gpu", N](),
            grid_dim=(N + TPB - 1) // TPB,
            block_dim=TPB,
        )

    def sample_continuous[
        OBS: Int, ACT: Int, RL: Int, MB: Int, N_ENVS: Int
    ](
        mut self,
        mut state: OnPolicyState[OBS, ACT, RL, MB, N_ENVS],
        action_ptr: _Ptr,
        action_scale: Scalar[DT],
    ) raises:
        """After the actor forward into `state.ao1` and the critic forward into
        `state.v1`: N(0,1) noise into `state.z` (device Philox), then the
        sample. The offset advances by the draw's even-rounded size."""
        comptime NZ = N_ENVS * ACT
        var c = state.ctx.value()
        box_muller_normal_gpu_dev[NZ](
            c, mptr(state.z.dev.value().unsafe_ptr()), self.seed,
            self._offset_lt(),
        )
        # The same rounding as `fb/kernels.mojo`'s draw (`N + N % 2`).
        c.enqueue_function[advance_rng_offset_kernel[NZ + (NZ % 2)]](
            self._offset_lt(), grid_dim=1, block_dim=1
        )
        c.enqueue_function[_sample_continuous_kernel[N_ENVS, ACT]](
            state.ao1.tt["gpu", N_ENVS * 2 * ACT](),
            state.v1.tt["gpu", N_ENVS](),
            state.z.tt["gpu", NZ](),
            tile[NZ](action_ptr),
            state.cached_action.tt["gpu", NZ](),
            state.cached_log_prob.tt["gpu", N_ENVS](),
            state.cached_value.tt["gpu", N_ENVS](),
            action_scale,
            grid_dim=(N_ENVS + TPB - 1) // TPB,
            block_dim=TPB,
        )

    def sample_discrete[
        N_ACTIONS: Int, OBS: Int, RL: Int, MB: Int, N_ENVS: Int
    ](
        mut self,
        mut state: OnPolicyState[OBS, 1, RL, MB, N_ENVS],
        mut logits: Tensor,
        action_ptr: _Ptr,
    ) raises:
        """After the actor forward into `logits` and the critic forward into
        `state.v1`: the categorical sample, then the offset advances by 1."""
        var c = state.ctx.value()
        c.enqueue_function[_sample_discrete_kernel[N_ENVS, N_ACTIONS]](
            logits.tt["gpu", N_ENVS * N_ACTIONS](),
            state.v1.tt["gpu", N_ENVS](),
            self._offset_lt(),
            self.seed,
            tile[N_ENVS](action_ptr),
            state.cached_action.tt["gpu", N_ENVS](),
            state.cached_log_prob.tt["gpu", N_ENVS](),
            state.cached_value.tt["gpu", N_ENVS](),
            grid_dim=(N_ENVS + TPB - 1) // TPB,
            block_dim=TPB,
        )
        c.enqueue_function[advance_rng_offset_kernel[1]](
            self._offset_lt(), grid_dim=1, block_dim=1
        )

    def record[
        OBS: Int, ACT: Int, RL: Int, MB: Int, N_ENVS: Int
    ](
        mut self,
        mut state: OnPolicyState[OBS, ACT, RL, MB, N_ENVS],
        reward_ptr: _Ptr,
        next_obs_ptr: _Ptr,
        done_ptr: _Ptr,
        terminated_ptr: _Ptr,
    ) raises:
        """Row `state.rollout_idx` (the obs the act step read is `state.ob1`),
        then the cursor advances. A no-op past the end, like the host record."""
        comptime RN = RL * N_ENVS
        var t = state.rollout_idx
        if t >= RL:
            return
        var c = state.ctx.value()
        c.enqueue_function[_record_kernel[N_ENVS, OBS, ACT, RN]](
            Int32(t),
            state.ob1.tt["gpu", N_ENVS * OBS](),
            state.cached_action.tt["gpu", N_ENVS * ACT](),
            state.cached_log_prob.tt["gpu", N_ENVS](),
            state.cached_value.tt["gpu", N_ENVS](),
            tile[N_ENVS](reward_ptr),
            tile[N_ENVS](done_ptr),
            tile[N_ENVS](terminated_ptr),
            tile[N_ENVS * OBS](next_obs_ptr),
            state.obs_buf.tt["gpu", RN * OBS](),
            state.act_buf.tt["gpu", RN * ACT](),
            state.olp_buf.tt["gpu", RN](),
            state.val_buf.tt["gpu", RN](),
            state.rew_buf.tt["gpu", RN](),
            state.done_buf.tt["gpu", RN](),
            state.term_buf.tt["gpu", RN](),
            state.next_obs_buf.tt["gpu", RN * OBS](),
            grid_dim=(N_ENVS + TPB - 1) // TPB,
            block_dim=TPB,
        )
        state.rollout_idx += 1

    def gae[
        OBS: Int, ACT: Int, RL: Int, MB: Int, N_ENVS: Int
    ](
        mut self,
        mut state: OnPolicyState[OBS, ACT, RL, MB, N_ENVS],
        gamma: Scalar[DT],
        gae_lambda: Scalar[DT],
    ) raises:
        """After the critic forward over `state.next_obs_buf` into
        `state.next_val_buf`: GAE into the device `adv_buf` / `ret_buf`."""
        comptime RN = RL * N_ENVS
        var c = state.ctx.value()
        c.enqueue_function[_gae_kernel[N_ENVS, RL]](
            state.next_val_buf.tt["gpu", RN](),
            state.rew_buf.tt["gpu", RN](),
            state.val_buf.tt["gpu", RN](),
            state.done_buf.tt["gpu", RN](),
            state.term_buf.tt["gpu", RN](),
            state.adv_buf.tt["gpu", RN](),
            state.ret_buf.tt["gpu", RN](),
            gamma,
            gae_lambda,
            grid_dim=(N_ENVS + TPB - 1) // TPB,
            block_dim=TPB,
        )

    def reset_rollout[
        OBS: Int, ACT: Int, RL: Int, MB: Int, N_ENVS: Int
    ](
        mut self,
        mut state: OnPolicyState[OBS, ACT, RL, MB, N_ENVS],
    ) raises:
        """The device twin of `PPORecordStep.reset_rollout`: cursor to 0 and
        the device `term_buf` zeroed (the record kernel writes every row's
        term anyway; zeroed so a short rollout cannot read a stale one)."""
        state.rollout_idx = 0
        var c = state.ctx.value()
        fill_dev(state.term_buf.dev.value(), RL * N_ENVS, Scalar[DT](0.0), c)
