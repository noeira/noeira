"""ScalarAdam — single-scalar Adam (storage surface).

Bespoke (NOT a ParamVisitor — it tunes ONE scalar, not a Module tree). Used for
SAC's auto-tuned entropy temperature `log_alpha`: a single trainable scalar with
its own Adam moments + incremental bias correction.

CPU `step`: pure host scalars (incremental β^t running products → O(1), the SAC
CPU bit-identity path). GPU `step_device` (CUDA-graph capturable): the scalar
state lives in a [6] device buffer `state_dev` = [log_α, m, v, β₁ᵗ, β₂ᵗ, α]; a
1-thread kernel reads the device-resident grad (SAC's reduced `lp_mean`), runs the
same Adam math, and writes α = exp(log_α) back into `state_dev[ALPHA]` — the slot
SAC's `Scale` nodes read via `alpha_dev_ptr` (no host scalar baked per step →
capturable). The host fields are NOT advanced on the GPU path; `read_alpha` D2Hs
the live α at log cadence.
"""

from std.math import sqrt as fsqrt, exp as fexp
from max.gpu import thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import LayoutTensor, Layout

from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import CheckpointScalars


# state_dev layout — single [6] device buffer.
comptime _SA_LOG_ALPHA = 0
comptime _SA_M = 1
comptime _SA_V = 2
comptime _SA_B1POW = 3
comptime _SA_B2POW = 4
comptime _SA_ALPHA = 5
comptime _SA_STATE_N = 6


def _scalar_adam_step_kernel(
    state: LayoutTensor[DT, Layout.row_major(6, 1), MutAnyOrigin],
    lp_mean: LayoutTensor[DT, Layout.row_major(1, 1), ImmutAnyOrigin],
    target_entropy: Scalar[DT],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
):
    """1-thread on-device ScalarAdam step. Reads the entropy grad on-device
    (`grad = -(lp_mean[0] + H_target)`, the host AlphaUpdateStep expression),
    runs the same incremental-bias-correction Adam update, and writes
    `state[ALPHA] = exp(state[LOG_ALPHA])`. lr/β/ε/H_target are capture-safe
    baked scalars."""
    if Int(thread_idx.x) != 0:
        return
    var one: Scalar[DT] = 1.0
    var grad = -(rebind[Scalar[DT]](lp_mean[0, 0]) + target_entropy)
    var m = beta1 * rebind[Scalar[DT]](state[_SA_M, 0]) + (one - beta1) * grad
    var v = beta2 * rebind[Scalar[DT]](state[_SA_V, 0]) + (
        one - beta2
    ) * grad * grad
    var b1pow = rebind[Scalar[DT]](state[_SA_B1POW, 0]) * beta1
    var b2pow = rebind[Scalar[DT]](state[_SA_B2POW, 0]) * beta2
    var m_hat = m / (one - b1pow)
    var v_hat = v / (one - b2pow)
    var la = rebind[Scalar[DT]](state[_SA_LOG_ALPHA, 0]) - lr * m_hat / (
        fsqrt(v_hat) + eps
    )
    state[_SA_M, 0] = m
    state[_SA_V, 0] = v
    state[_SA_B1POW, 0] = b1pow
    state[_SA_B2POW, 0] = b2pow
    state[_SA_LOG_ALPHA, 0] = la
    state[_SA_ALPHA, 0] = fexp(la)


struct ScalarAdam(Movable & Deinitable):
    var value: Scalar[DT]
    var m: Scalar[DT]
    var v: Scalar[DT]
    var t: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var beta1_pow_t: Scalar[DT]  # incremental β₁ᵗ (host path)
    var beta2_pow_t: Scalar[DT]  # incremental β₂ᵗ
    var state_dev: Optional[DeviceBuffer[DT]]  # GPU capture state; None on CPU
    var _ctx: Optional[DeviceContext]

    def __init__(
        out self,
        value: Scalar[DT],
        m: Scalar[DT],
        v: Scalar[DT],
        t: Int,
        lr: Scalar[DT],
        beta1: Scalar[DT],
        beta2: Scalar[DT],
        eps: Scalar[DT],
    ):
        self.value = value
        self.m = m
        self.v = v
        self.t = t
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.beta1_pow_t = Scalar[DT](1.0)
        self.beta2_pow_t = Scalar[DT](1.0)
        self.state_dev = None
        self._ctx = None

    @staticmethod
    def new(initial: Scalar[DT], lr: Scalar[DT]) -> Self:
        return Self(
            value=initial, m=0.0, v=0.0, t=0,
            lr=lr, beta1=0.9, beta2=0.999, eps=1e-8,
        )

    @staticmethod
    def new_device(
        ctx: DeviceContext, initial: Scalar[DT], lr: Scalar[DT]
    ) raises -> Self:
        """GPU factory — `state_dev` = [log_α, m, v, β₁ᵗ, β₂ᵗ, α] seeded to
        [initial, 0, 0, 1, 1, exp(initial)]."""
        var s = Self.new(initial, lr)
        var buf = ctx.enqueue_create_buffer[DT](_SA_STATE_N)
        var host = ctx.enqueue_create_host_buffer[DT](_SA_STATE_N)
        ctx.synchronize()
        host[_SA_LOG_ALPHA] = initial
        host[_SA_M] = Scalar[DT](0.0)
        host[_SA_V] = Scalar[DT](0.0)
        host[_SA_B1POW] = Scalar[DT](1.0)
        host[_SA_B2POW] = Scalar[DT](1.0)
        host[_SA_ALPHA] = fexp(initial)
        ctx.enqueue_copy(buf, host)
        ctx.synchronize()
        s.state_dev = buf^
        s._ctx = ctx
        return s^

    def step(mut self, grad: Scalar[DT]):
        """CPU step — host scalars only (incremental β^t)."""
        self.t += 1
        var one: Scalar[DT] = 1.0
        self.m = self.beta1 * self.m + (one - self.beta1) * grad
        self.v = self.beta2 * self.v + (one - self.beta2) * grad * grad
        self.beta1_pow_t *= self.beta1
        self.beta2_pow_t *= self.beta2
        var m_hat = self.m / (one - self.beta1_pow_t)
        var v_hat = self.v / (one - self.beta2_pow_t)
        self.value = self.value - self.lr * m_hat / (fsqrt(v_hat) + self.eps)

    def step_device(
        mut self,
        ctx: DeviceContext,
        lp_mean: DeviceBuffer[DT],
        target_entropy: Scalar[DT],
    ) raises:
        """GPU step — 1-thread kernel reading the device `lp_mean` grad. No D2H,
        no host scalar update; `state_dev[ALPHA]` is refreshed in place."""
        var state_lt = LayoutTensor[DT, Layout.row_major(6, 1), MutAnyOrigin](
            self.state_dev.value()
        )
        var lp_mean_lt = LayoutTensor[DT, Layout.row_major(1, 1)](lp_mean)
        ctx.enqueue_function[_scalar_adam_step_kernel](
            state_lt,
            lp_mean_lt,
            target_entropy,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            grid_dim=1,
            block_dim=1,
        )

    def alpha_dev_ptr(mut self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to `state_dev[ALPHA]` — the live α read by raw GPU-ABI kernel
        consumers (SAC's `target_y` device-α path takes this as a kernel arg).
        Stable for the buffer lifetime. Module-trait consumers (the actor-loss
        `Scale` node) should prefer `alpha_dev_buffer` (type-safe DeviceBuffer)."""
        return rebind[Pointer[Scalar[DT], MutAnyOrigin]](
            self.state_dev.value().unsafe_ptr()
        ).unsafe_offset(_SA_ALPHA)

    def alpha_dev_buffer(mut self) raises -> DeviceBuffer[DT]:
        """Length-1 device sub-buffer viewing `state_dev[ALPHA]` — the live α
        SAC's `Scale` nodes read via their multiplier source. Memory-sharing
        (`create_sub_buffer`), so in-place α refreshes by `step_device` are
        visible; carries device-residency in the type (no raw pointer)."""
        return self.state_dev.value().create_sub_buffer[DT](_SA_ALPHA, 1)

    def read_alpha(mut self) raises -> Scalar[DT]:
        """D2H the live device α (log cadence only — NOT per step)."""
        var ctx = self._ctx.value()
        var h = ctx.enqueue_create_host_buffer[DT](_SA_STATE_N)
        ctx.enqueue_copy(h, self.state_dev.value())
        ctx.synchronize()
        return h[_SA_ALPHA]

    def put_state(mut self, mut sc: CheckpointScalars, prefix: String) raises:
        """Record the whole optimizer — the tuned value, both moments and both
        bias-correction powers — as `<prefix>.value/.m/.v/.b1_pow/.b2_pow`.

        On the GPU path the host fields are never advanced (see the module
        header), so the live state is read back from `state_dev` first; saving
        the host fields there would checkpoint the INITIAL α."""
        if self.state_dev:
            var ctx = self._ctx.value()
            var h = ctx.enqueue_create_host_buffer[DT](_SA_STATE_N)
            ctx.enqueue_copy(h, self.state_dev.value())
            ctx.synchronize()
            self.value = h[_SA_LOG_ALPHA]
            self.m = h[_SA_M]
            self.v = h[_SA_V]
            self.beta1_pow_t = h[_SA_B1POW]
            self.beta2_pow_t = h[_SA_B2POW]
        sc.set(prefix + ".value", Float64(self.value))
        sc.set(prefix + ".m", Float64(self.m))
        sc.set(prefix + ".v", Float64(self.v))
        sc.set_int(prefix + ".t", self.t)
        sc.set(prefix + ".b1_pow", Float64(self.beta1_pow_t))
        sc.set(prefix + ".b2_pow", Float64(self.beta2_pow_t))

    def take_state(mut self, sc: CheckpointScalars, prefix: String) raises:
        """Inverse of `put_state`, host AND device. A checkpoint without these
        keys leaves the optimizer untouched."""
        if not sc.has(prefix + ".value"):
            return
        self.value = Scalar[DT](sc.get(prefix + ".value", 0.0))
        self.m = Scalar[DT](sc.get(prefix + ".m", 0.0))
        self.v = Scalar[DT](sc.get(prefix + ".v", 0.0))
        self.t = sc.get_int(prefix + ".t", 0)
        self.beta1_pow_t = Scalar[DT](sc.get(prefix + ".b1_pow", 1.0))
        self.beta2_pow_t = Scalar[DT](sc.get(prefix + ".b2_pow", 1.0))
        if self.state_dev:
            var ctx = self._ctx.value()
            var h = ctx.enqueue_create_host_buffer[DT](_SA_STATE_N)
            ctx.synchronize()
            h[_SA_LOG_ALPHA] = self.value
            h[_SA_M] = self.m
            h[_SA_V] = self.v
            h[_SA_B1POW] = self.beta1_pow_t
            h[_SA_B2POW] = self.beta2_pow_t
            h[_SA_ALPHA] = fexp(self.value)
            ctx.enqueue_copy(self.state_dev.value(), h)
            ctx.synchronize()
