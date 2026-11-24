# +--------------------------------------------------------------------------+ #
# | SmolVLA — Real-Time Chunking (RTC): the next chunk joins the committed one
# +--------------------------------------------------------------------------+ #
"""Guided inpainting of the next action chunk while the current one executes.

Black, Galliker, Levine, "Real-Time Execution of Action Chunking Flow
Policies" (arXiv 2506.07339), as LeRobot ships it
(`lerobot/policies/rtc/modeling_rtc.py`, `RTCProcessor.denoise_step`).

## What it is for

A query takes `d` control steps. Without RTC the robot either HOLDS during
them (`--sync`, the reference's evaluation loop: ~400 ms frozen per chunk on
the Orin) or keeps executing the old chunk and then jumps to a new one planned
from a stale observation (pipelining: the 35° oscillation of 20 Sep). RTC
plans the next chunk so that its first `d` actions ARE the old chunk's next
`d` actions — they will run during the query anyway — and blends the rest:

    weights W_i:  1 for i < d,  a decay for d <= i < e,  0 for i >= e

## The guided velocity, in SmolVLA's convention

SmolVLA integrates t from 1 (noise) to 0 (actions), `x_t = t*noise +
(1-t)*x0`, `v = noise - x0`, `x <- x + dt*v` with dt = -1/steps. At each step:

    x1   = x_t - t * v                         # the predicted clean chunk
    err  = W ⊙ (prev - x1)
    corr = err                                  # LeRobot (see below)
         | (∂x1/∂x_t)^T err = err - t * J_v^T err   # the paper, exact
    v'   = v - g(t) * corr
    g(t) = min(beta, (1-tau)/(tau * r²)),  tau = 1 - t,
           r² = (1-tau)² / (tau² + (1-tau)²)

`dt < 0`, so `x <- x + dt*v'` moves `x` by `+|dt|*g*corr`: toward `prev`.

⚠⚠ **LeRobot's correction IS `err`, not the paper's VJP.** It computes `v_t`
BEFORE `x_t.requires_grad_(True)`, so the graph of `v_t` does not reach `x_t`
and `autograd.grad(x1_t, x_t, err)` returns the identity term only. That
costs nothing at run time, which is why it is the default here
(`RTC_CORR_IDENTITY`). The exact term needs a backward pass through the
expert per step (`SmolVLADenoise.backward`, RECORD = True) and is not wired
yet.

⚠ **`prev` is in the MODEL's space** — the normalised, 32-wide chunk the
sampler returned, before `unnormalize_action` — never robot units. LeRobot
does the same (no normalisation inside RTC). The padded dims 6.. are matched
too; they came out of the same model, so pulling them toward themselves is
harmless and keeps the kernel free of an `ADIM_REAL`.

⚠ **`e` (LeRobot's `execution_horizon`) is the END of the overlap in the NEW
chunk's index**, the paper's `H - s`, not the paper's `s`. LeRobot clips it to
the leftover's length and starts the ramp at `min(d, e)`.
"""

from std.math import exp
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.core.tiles import Tile


# ── prefix-weight schedules (LeRobot's `RTCAttentionSchedule`) ─────────────
comptime RTC_ZEROS: Int = 0
"""1 for i < d, else 0: hard inpainting of the frozen prefix only."""
comptime RTC_ONES: Int = 1
"""1 for i < e, else 0 (does not depend on d)."""
comptime RTC_LINEAR: Int = 2
"""1 for i < d; (e - i)/(e - d + 1) for d <= i < e; 0 after. LeRobot's
default."""
comptime RTC_EXP: Int = 3
"""1 for i < d; c(e^c - 1)/(e - 1), c = (e - i)/(e - d + 1), for
d <= i < e; 0 after. The paper's Eq. 5, which it found best."""

comptime RTC_BETA_DEFAULT: Float64 = 10.0
"""LeRobot's `max_guidance_weight`. The paper uses 5 ("conservative")."""


def rtc_schedule_from_name(name: String) raises -> Int:
    if name == "zeros":
        return RTC_ZEROS
    if name == "ones":
        return RTC_ONES
    if name == "linear":
        return RTC_LINEAR
    if name == "exp":
        return RTC_EXP
    raise Error(
        "rtc: unknown schedule '" + name + "' (zeros | ones | linear | exp)"
    )


def rtc_prefix_weights(
    d: Int, e: Int, total: Int, schedule: Int
) raises -> List[Float64]:
    """W_i for i in [0, total). `d` = steps that execute during the query,
    `e` = end of the overlap, both in the NEW chunk's index.

    LeRobot's construction exactly: `start = min(d, e)`, the ramp is
    `linspace(1, 0, n + 2)[1:-1]` over `n = e - start` entries, which is
    `(e - i)/(e - start + 1)` at `i = start + k`."""
    if d < 0 or e < 0 or total < 1:
        raise Error(
            "rtc_prefix_weights: d " + String(d) + ", e " + String(e)
            + ", total " + String(total)
        )
    var end = min(e, total)
    var start = min(d, end)
    var w = List[Float64](length=total, fill=0.0)
    if schedule == RTC_ZEROS:
        for i in range(min(d, total)):
            w[i] = 1.0
        return w^
    if schedule == RTC_ONES:
        for i in range(end):
            w[i] = 1.0
        return w^
    if schedule != RTC_LINEAR and schedule != RTC_EXP:
        raise Error("rtc_prefix_weights: schedule " + String(schedule))
    for i in range(start):
        w[i] = 1.0
    var denom = Float64(end - start + 1)
    for i in range(start, end):
        var c = Float64(end - i) / denom
        if schedule == RTC_LINEAR:
            w[i] = c
        else:
            w[i] = c * (exp(c) - 1.0) / (exp(1.0) - 1.0)
    return w^


def rtc_guidance_weight(time: Float64, beta: Float64) -> Float64:
    """`min(beta, (1-tau)/(tau r²))` at SmolVLA time `time` (tau = 1 - time).

    Simplified: `(t² + (1-t)²) / (t (1-t))` with t = `time`. At time = 1
    (the first step, tau = 0) it is +inf and clips to `beta`, as LeRobot's
    `nan_to_num(posinf=beta)`; at time = 0 it is 0 (never evaluated)."""
    if time >= 1.0:
        return beta
    if time <= 0.0:
        return 0.0
    var g = (time * time + (1.0 - time) * (1.0 - time)) / (
        time * (1.0 - time)
    )
    return min(beta, g)


def _rtc_guide_kernel[
    N: Int, ADIM: Int, CHUNK: Int
](
    x: Tile[DT, N],
    v: Tile[DT, N],
    prev: Tile[DT, N],
    w: Tile[DT, CHUNK],
    time: Scalar[DT],
    gw: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i >= N:
        return
    var tok = (i // ADIM) % CHUNK
    var vi = rebind[Scalar[DT]](v.ptr[unsafe_offset=i])
    var x1 = rebind[Scalar[DT]](x.ptr[unsafe_offset=i]) - time * vi
    var err = rebind[Scalar[DT]](w.ptr[unsafe_offset=tok]) * (
        rebind[Scalar[DT]](prev.ptr[unsafe_offset=i]) - x1
    )
    v.ptr[unsafe_offset=i] = vi - gw * err


struct RtcGuide[CHUNK: Int, ADIM: Int, B: Int = 1](Movable):
    """The committed prefix, its weights and beta — armed per query.

    Off (`active == False`, the default) the sampler is today's sampler
    exactly: `apply` is never called, nothing is uploaded."""

    comptime XN: Int = Self.B * Self.CHUNK * Self.ADIM

    var active: Bool
    var beta: Float64
    var d: Int
    var e: Int
    var prev: Tensor
    """`[B, CHUNK, ADIM]` in model space: the leftover of the chunk being
    executed, shifted so its index 0 is the new chunk's index 0, zero-padded."""
    var w: Tensor
    """`[CHUNK]`, one weight per action step, shared by every dim."""

    def __init__(out self):
        self.active = False
        self.beta = RTC_BETA_DEFAULT
        self.d = 0
        self.e = 0
        self.prev = Tensor()
        self.w = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.active = move.active
        self.beta = move.beta
        self.d = move.d
        self.e = move.e
        self.prev = move.prev^
        self.w = move.w^

    def arm[
        target: StaticString
    ](
        mut self,
        ref last: Tensor,
        k: Int,
        d: Int,
        horizon: Int,
        schedule: Int,
        beta: Float64,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Guide the next `sample` toward `last[k:]`.

        `last` is the previous chunk in model space with its HOST data valid
        (`SmolVLAPolicy.chunk_buf` after `finish_action`). `k` = the index of
        `last` that lines up with the new chunk's index 0 — the action about
        to execute when the observation was taken. `d` = the steps expected
        to execute during the query. `horizon` = LeRobot's
        `execution_horizon`, clipped to the leftover `CHUNK - k`; pass
        `CHUNK` for "the whole leftover"."""
        comptime assert Self.B == 1, (
            "RtcGuide.arm: B = 1 only — the deploy queries one robot"
        )
        if k < 0 or k >= Self.CHUNK:
            raise Error(
                "RtcGuide.arm: k " + String(k) + " outside the chunk of "
                + String(Self.CHUNK) + " — nothing of the old chunk is left"
            )
        if len(last.data) < Self.XN:
            raise Error(
                "RtcGuide.arm: the previous chunk holds "
                + String(len(last.data)) + " host values, need "
                + String(Self.XN) + " (finish_action not called?)"
            )
        var left = Self.CHUNK - k
        var e = min(horizon, left)
        var wts = rtc_prefix_weights(d, e, Self.CHUNK, schedule)
        self.prev.ensure(Self.XN)
        self.w.ensure(Self.CHUNK)
        for j in range(Self.CHUNK):
            for a in range(Self.ADIM):
                var dst = j * Self.ADIM + a
                if j < left:
                    self.prev.data[dst] = last.data[(j + k) * Self.ADIM + a]
                else:
                    self.prev.data[dst] = Scalar[DT](0)
            self.w.data[j] = Scalar[DT](wts[j])
        comptime if target != "cpu":
            var c = ctx.value()
            self.prev.upload_resident(c)
            self.w.upload_resident(c)
        self.beta = beta
        self.d = d
        self.e = e
        self.active = True

    def disarm(mut self):
        self.active = False

    def apply[
        target: StaticString
    ](
        mut self,
        mut x: Tensor,
        mut v: Tensor,
        time: Float64,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`v <- v - g(time) * W ⊙ (prev - (x - time*v))`, in place."""
        var gw = Scalar[DT](rtc_guidance_weight(time, self.beta))
        var tm = Scalar[DT](time)
        comptime if target == "cpu":
            for i in range(Self.XN):
                var tok = (i // Self.ADIM) % Self.CHUNK
                var vi = v.data[i]
                var x1 = x.data[i] - tm * vi
                var err = self.w.data[tok] * (self.prev.data[i] - x1)
                v.data[i] = vi - gw * err
        else:
            var c = ctx.value()
            comptime nb = (Self.XN + TPB - 1) // TPB
            c.enqueue_function[
                _rtc_guide_kernel[Self.XN, Self.ADIM, Self.CHUNK]
            ](
                x.tt["gpu", Self.XN](),
                v.tt["gpu", Self.XN](),
                self.prev.tt["gpu", Self.XN](),
                self.w.tt["gpu", Self.CHUNK](),
                tm,
                gw,
                grid_dim=nb,
                block_dim=TPB,
            )
