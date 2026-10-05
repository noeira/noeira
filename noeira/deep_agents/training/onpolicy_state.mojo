"""OnPolicyState — shared per-step flow object for on-policy block trainers.

Analog to the off-policy `TrainerState`. Carries:
  - the full rollout buffers (obs/act/olp/rew/val/done/term/adv/ret)
    sized [ROLLOUT_LEN, N_ENVS], written by record_step + gae_step,
    read by minibatch_gather_step
  - the per-step caches (action / log_prob / value) sized N_ENVS,
    filled by act_step, drained by record_step
  - the per-step BATCH=N_ENVS scratches (ob1/ao1/v1/z) for batched
    actor + critic forward calls during action selection
  - the minibatch BATCH=MINIBATCH scratches (mb_obs/mb_act/mb_olp/mb_adv/
    mb_ret/mb_v/mb_gv/mb_gi) written by gather, consumed by actor/critic
    train steps
  - rollout cursor (single Int — all envs advance synchronously) + ctx
  - Int32 indices for the Fisher-Yates shuffle, sized ROLLOUT_LEN * N_ENVS

STORAGE migration: each buffer is a storage `Tensor` (host `.data` List + an
optional GPU `DeviceBuffer`), allocated via the `_mk[target]` helper which
ALWAYS allocates the host list and (on GPU) also a device buffer. The hybrid
pattern is unchanged: rollout / per-step / gather work on the host `.data`; the
per-step obs (`ob1`) is `upload`ed and the actor/critic outputs (`ao1`/`v1`)
`download`ed inside PPOActStep; the minibatch tensors are `upload`ed before the
actor/critic train steps. Blocks read raw host pointers via `.data.unsafe_ptr()`
and build device views via `.lt["gpu", layout]()`.

N_ENVS defaults to 1 for single-env callers (host-list driver path);
N_ENVS > 1 is reached via the BatchedEnv driver.
"""

from std.memory import alloc
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor


struct OnPolicyState[
    OBS: Int,
    ACT: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_ENVS: Int = 1,
](Defaultable & Movable & Deinitable):
    # ── Rollout buffers (ROLLOUT_LEN × N_ENVS, T-major) ─────────────
    var obs_buf: Tensor
    var act_buf: Tensor
    var olp_buf: Tensor
    var rew_buf: Tensor
    var val_buf: Tensor
    var done_buf: Tensor
    var term_buf: Tensor
    var adv_buf: Tensor
    var ret_buf: Tensor

    # Each row's NEXT observation — the PRE-RESET obs after its step, so an
    # episode that ends on that row (terminated or truncated) still has its
    # own final state — and V of it, computed once at the update. GAE
    # bootstraps a truncated row from it, a terminated row from 0, and the
    # last row of the rollout from it (`PPOGAEStep`).
    var next_obs_buf: Tensor
    var next_val_buf: Tensor

    # ── Per-step caches per env (filled by act, drained by record) ──
    var cached_action: Tensor
    var cached_log_prob: Tensor
    var cached_value: Tensor

    # ── Per-step BATCH=N_ENVS scratches ─────────────────────────────
    var ob1: Tensor
    var ao1: Tensor
    var v1: Tensor
    var z: Tensor

    # ── Minibatch (BATCH=MINIBATCH) scratches ───────────────────────
    var mb_obs: Tensor
    var mb_act: Tensor
    var mb_olp: Tensor
    var mb_adv: Tensor
    var mb_ret: Tensor
    var mb_v: Tensor
    var mb_gv: Tensor
    var mb_gi: Tensor

    # Int32 shuffle/gather index array (Tensor is DT-only, so raw ptr).
    var indices: Optional[Pointer[Int32, MutUntrackedOrigin]]

    # Rollout cursor.
    var rollout_idx: Int

    # Step bookkeeping.
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.obs_buf = Tensor()
        self.act_buf = Tensor()
        self.olp_buf = Tensor()
        self.rew_buf = Tensor()
        self.val_buf = Tensor()
        self.done_buf = Tensor()
        self.term_buf = Tensor()
        self.adv_buf = Tensor()
        self.ret_buf = Tensor()
        self.next_obs_buf = Tensor()
        self.next_val_buf = Tensor()
        self.cached_action = Tensor()
        self.cached_log_prob = Tensor()
        self.cached_value = Tensor()
        self.ob1 = Tensor()
        self.ao1 = Tensor()
        self.v1 = Tensor()
        self.z = Tensor()
        self.mb_obs = Tensor()
        self.mb_act = Tensor()
        self.mb_olp = Tensor()
        self.mb_adv = Tensor()
        self.mb_ret = Tensor()
        self.mb_v = Tensor()
        self.mb_gv = Tensor()
        self.mb_gi = Tensor()
        self.indices = None
        self.rollout_idx = 0
        self.ctx = None

    @staticmethod
    def _mk[
        target: StaticString
    ](n: Int, ctx: Optional[DeviceContext]) raises -> Tensor:
        """Allocate a buffer with a host `.data` list ALWAYS and (on GPU) a
        device buffer too — so blocks can use `.data` (host work) and
        `.lt["gpu"]` / `upload` / `download` (device work) on any field."""
        var t = Tensor.alloc(n)
        comptime if target == "gpu":
            t.ensure_gpu(ctx.value(), n)
        return t^

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "OnPolicyState: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error("OnPolicyState.make[target='gpu']: ctx required")
        comptime RN = Self.ROLLOUT_LEN * Self.N_ENVS
        var s = Self()
        s.obs_buf = Self._mk[target](RN * Self.OBS, ctx)
        s.act_buf = Self._mk[target](RN * Self.ACT, ctx)
        s.olp_buf = Self._mk[target](RN, ctx)
        s.rew_buf = Self._mk[target](RN, ctx)
        s.val_buf = Self._mk[target](RN, ctx)
        s.done_buf = Self._mk[target](RN, ctx)
        s.term_buf = Self._mk[target](RN, ctx)
        s.adv_buf = Self._mk[target](RN, ctx)
        s.ret_buf = Self._mk[target](RN, ctx)
        s.next_obs_buf = Self._mk[target](RN * Self.OBS, ctx)
        s.next_val_buf = Self._mk[target](RN, ctx)
        s.cached_action = Self._mk[target](Self.N_ENVS * Self.ACT, ctx)
        s.cached_log_prob = Self._mk[target](Self.N_ENVS, ctx)
        s.cached_value = Self._mk[target](Self.N_ENVS, ctx)
        s.ob1 = Self._mk[target](Self.N_ENVS * Self.OBS, ctx)
        s.ao1 = Self._mk[target](Self.N_ENVS * 2 * Self.ACT, ctx)
        s.v1 = Self._mk[target](Self.N_ENVS, ctx)
        s.z = Self._mk[target](Self.N_ENVS * Self.ACT, ctx)
        s.mb_obs = Self._mk[target](Self.MINIBATCH * Self.OBS, ctx)
        s.mb_act = Self._mk[target](Self.MINIBATCH * Self.ACT, ctx)
        s.mb_olp = Self._mk[target](Self.MINIBATCH, ctx)
        s.mb_adv = Self._mk[target](Self.MINIBATCH, ctx)
        s.mb_ret = Self._mk[target](Self.MINIBATCH * 1, ctx)
        s.mb_v = Self._mk[target](Self.MINIBATCH * 1, ctx)
        s.mb_gv = Self._mk[target](Self.MINIBATCH * 1, ctx)
        s.mb_gi = Self._mk[target](Self.MINIBATCH * Self.OBS, ctx)
        s.indices = alloc[Int32]({count = RN}).unsafe_leak()
        s.ctx = ctx
        return s^
