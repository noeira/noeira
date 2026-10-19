# +--------------------------------------------------------------------------+ #
# | SmolVLA — the flow-matching training objective
# +--------------------------------------------------------------------------+ #
"""What the model is actually trained on, which is not what it is run on.

Inference integrates a velocity field over ten Euler steps. Training does no
integration at all: it samples ONE time `t`, forms the point `x_t` on the
straight line between the action chunk and a noise sample, asks the network
for the velocity there, and compares it to the velocity the straight line
actually has.

    noise ~ N(0, 1)                         the flow-matching x_1
    t     ~ Beta(1.5, 1) * 0.999 + 0.001    openpi's convention
    x_t   = t*noise + (1 - t)*actions       the interpolant
    u_t   = noise - actions                 its velocity, CONSTANT in t
    v_t   = action_out(expert(embed(x_t, t)))
    L     = mean over the REAL action dims of (v_t - u_t)^2

⚠ **`u_t = noise − actions`, not `actions − noise`.** Both are shape-identical
and the second trains a model that drives the chunk the wrong way. The
endpoints are what pin it: at `t = 1`, `x_t` is the noise; at `t = 0`, `x_t` is
the actions. `test_flow_loss.mojo` asserts both exactly.

⚠ **The loss covers only the first `ADIM_REAL` action dimensions.** The chunk
is padded to `max_action_dim = 32` and the SO-101 has 6, so 26 of every 32
columns of `actions` are zeros. `lerobot` slices `losses[:, :, :original_action_dim]`
before reducing. Skipping that slice does NOT crash and does not look wrong:
it trains the network to predict `u_t = noise − 0 = noise` on 26 columns —
pure noise, unpredictable by construction — and every one of those columns
contributes gradient. The padded columns must receive EXACTLY zero.

⚠ **`x_t` still spans all 32 columns**, because `action_in_proj` takes 32. Only
the LOSS is truncated. Truncating `x_t` too would change the network's input.

⚠ **And the loss is masked per TIMESTEP as well as per column.** A chunk that
runs off the end of its episode is not short — `dataset_reader` CLAMPS the
query index, so those slots hold the last real action repeated, and marks them
in `action_is_pad`. Training on them teaches the model that every episode ends
by holding still. The reference zeroes those timesteps AND removes them from
the denominator (`losses.sum() / num_valid`); doing only the first would leave
the loss scaled by the fraction of valid steps, which changes the effective
learning rate with how close the sample sits to an episode boundary.

`valid` is `[B, CHUNK]`, 1.0 or 0.0, and `n_terms` is the denominator the loss
is averaged over. The caller builds both because the caller is what knows
where the episode ended.

⚠ **`n_terms` is NOT necessarily this call's own count of 1s.** Under gradient
accumulation it is the whole GROUP's — every micro-batch is given the same
total so the accumulated gradient is the mean over the group rather than a sum
of per-micro-batch means. `n_terms` was originally called `n_valid` and
bounded by `B*CHUNK`, which read as a sanity check and was really a
single-call assumption; the first accumulating run hit it at 400 against a
bound of 50. The bound is gone and the name says what the number is.

⚠ Where the reference does `num_valid.clamp_min(1)`, this RAISES. A batch with
no valid timestep at all is a broken sampler, and a loss quietly divided by 1
is a number that looks like a loss.

## The time distribution

`Beta(alpha, 1)` has density `alpha * x^(alpha-1)` and therefore CDF `x^alpha`,
so its inverse is `u^(1/alpha)` and one uniform sample is enough — no
rejection, no Dirichlet, and no dependence on a Beta sampler the stdlib does
not have. That identity holds ONLY for `beta == 1`, which is why this file
asserts it at compile time rather than carrying a `beta` that would be
silently ignored.

⚠ Sampling `t` UNIFORMLY instead would run, train, and converge to something
else: `Beta(1.5, 1)` has mean 0.6 and puts most of its mass near `t = 1`, the
noisy end, which is where the velocity is hardest to predict. The gate
measures the CDF against `x^1.5` and separately demonstrates that a uniform
draw fails it.
"""

from std.math import exp, log, sqrt
from max.gpu import global_idx
from std.random import random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.random.box_muller import box_muller_normal
from noeira.core.tiles import Tile


comptime FM_ALPHA: Float64 = 1.5
comptime FM_BETA: Float64 = 1.0
comptime FM_SCALE: Float64 = 0.999
comptime FM_OFFSET: Float64 = 0.001


def sample_time() -> Float64:
    """One flow-matching timestep, `Beta(FM_ALPHA, 1) * SCALE + OFFSET`.

    ⚠ Inverse-CDF, which is exact only because `FM_BETA == 1`.
    """
    comptime assert FM_BETA == 1.0, (
        "sample_time inverts the CDF x^alpha, which is Beta(alpha, beta) only"
        " for beta == 1. A different beta needs a real Beta sampler, not a"
        " different exponent here."
    )
    var u = random_float64()
    return exp(log(u) / FM_ALPHA) * FM_SCALE + FM_OFFSET


def sample_times(b: Int) -> List[Float64]:
    """One `t` PER BATCH ELEMENT — the reference does `time[:, None, None]`,
    so a batch sees a spread of times, not one shared one."""
    var ts = List[Float64]()
    for _ in range(b):
        ts.append(sample_time())
    return ts^


def sample_noise(mut dst: Tensor, n: Int) raises:
    """`dst[0:n]` ~ iid N(0, 1) on the HOST. Upload if the caller needs it on
    the device — the caller also needs `actions` there, so it owns the
    transfer."""
    dst.ensure(n)
    box_muller_normal(dst.data.unsafe_ptr(), n)


# ── x_t and u_t ──────────────────────────────────────────────────────────


def _xt_ut_kernel[B: Int, ROW: Int](
    noise: Tile[DT, B, ROW],
    actions: Tile[DT, B, ROW],
    times: Tile[DT, B],
    x_t: Tile[DT, B, ROW],
    u_t: Tile[DT, B, ROW],
):
    var i = Int(global_idx.x)
    if i >= B * ROW:
        return
    var b = i // ROW
    var t = rebind[Scalar[DT]](times.ptr[unsafe_offset=b])
    var nz = rebind[Scalar[DT]](noise.ptr[unsafe_offset=i])
    var a = rebind[Scalar[DT]](actions.ptr[unsafe_offset=i])
    x_t.ptr[unsafe_offset=i] = t * nz + (Scalar[DT](1) - t) * a
    u_t.ptr[unsafe_offset=i] = nz - a


def build_xt_ut[
    target: StaticString, B: Int, ROW: Int
](
    mut noise: Tensor, mut actions: Tensor, mut times: Tensor,
    mut x_t: Tensor, mut u_t: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`ROW = CHUNK * ADIM`. `times` is `[B]`, one timestep per sample."""
    comptime if target == "cpu":
        x_t.ensure(B * ROW)
        u_t.ensure(B * ROW)
        for b in range(B):
            var t = times.data[b]
            for j in range(ROW):
                var i = b * ROW + j
                x_t.data[i] = t * noise.data[i] + (Scalar[DT](1) - t) * (
                    actions.data[i]
                )
                u_t.data[i] = noise.data[i] - actions.data[i]
    else:
        var c = ctx.value()
        x_t.ensure_gpu(c, B * ROW)
        u_t.ensure_gpu(c, B * ROW)
        c.enqueue_function[_xt_ut_kernel[B, ROW]](
            noise.tt["gpu", B, ROW](),
            actions.tt["gpu", B, ROW](),
            times.tt["gpu", B](),
            x_t.tt["gpu", B, ROW](),
            u_t.tt["gpu", B, ROW](),
            grid_dim=(B * ROW + TPB - 1) // TPB,
            block_dim=TPB,
        )


# ── the loss ─────────────────────────────────────────────────────────────


def _mse_kernel[B: Int, CHUNK: Int, ADIM: Int, ADIM_REAL: Int](
    v_t: Tile[DT, B, CHUNK * ADIM],
    u_t: Tile[DT, B, CHUNK * ADIM],
    valid: Tile[DT, B, CHUNK],
    grad_v: Tile[DT, B, CHUNK * ADIM],
    err: Tile[DT, B, CHUNK * ADIM],
    two_over_n: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i >= B * CHUNK * ADIM:
        return
    var d = i % ADIM
    var bt = i // ADIM
    # ⚠ A padded column, or a timestep past the end of its episode. EXACTLY
    # zero, not a small number: for a padded column `actions` is 0 so u_t is
    # raw noise, and for a padded timestep the action is the last real one
    # repeated.
    if d >= ADIM_REAL or rebind[Scalar[DT]](
        valid.ptr[unsafe_offset=bt]
    ) == Scalar[DT](0):
        grad_v.ptr[unsafe_offset=i] = Scalar[DT](0)
        err.ptr[unsafe_offset=i] = Scalar[DT](0)
        return
    var e = rebind[Scalar[DT]](v_t.ptr[unsafe_offset=i]) - rebind[Scalar[DT]](
        u_t.ptr[unsafe_offset=i]
    )
    err.ptr[unsafe_offset=i] = e * e
    grad_v.ptr[unsafe_offset=i] = two_over_n * e


def flow_mse[
    target: StaticString, B: Int, CHUNK: Int, ADIM: Int, ADIM_REAL: Int
](
    mut v_t: Tensor, mut u_t: Tensor, mut valid: Tensor, mut grad_v: Tensor,
    mut err: Tensor, n_terms: Int, ctx: Optional[DeviceContext] = None,
) raises:
    """Writes `grad_v` = dL/d(v_t) and `err` = the per-element squared errors.

    `valid` is `[B, CHUNK]` (1.0 / 0.0). `n_terms` is the averaging
    denominator — this call's count of 1s when used alone, the whole group's
    when accumulating. See the header.

    ⚠ Split from the scalar so a training step can skip the reduction: reading
    the loss means bringing `err` back to the host, and that is a full pipeline
    drain. `mean_err` does it when the caller actually wants the number.
    """
    comptime assert ADIM_REAL <= ADIM, (
        "flow_mse: ADIM_REAL is the robot's real action width and cannot"
        " exceed the padded ADIM"
    )
    comptime TOT = B * CHUNK * ADIM
    if n_terms <= 0:
        raise Error(
            "flow_mse: n_terms is 0 — every timestep in the batch is padding."
            " The reference clamps this to 1 and returns a number; that number"
            " is not a loss, so this raises instead."
        )
    var two_over_n = Scalar[DT](2.0 / Float64(n_terms * ADIM_REAL))
    comptime if target == "cpu":
        grad_v.ensure(TOT)
        err.ensure(TOT)
        for i in range(TOT):
            var d = i % ADIM
            var bt = i // ADIM
            if d >= ADIM_REAL or valid.data[bt] == Scalar[DT](0):
                grad_v.data[i] = Scalar[DT](0)
                err.data[i] = Scalar[DT](0)
            else:
                var e = v_t.data[i] - u_t.data[i]
                err.data[i] = e * e
                grad_v.data[i] = two_over_n * e
    else:
        var c = ctx.value()
        grad_v.ensure_gpu(c, TOT)
        err.ensure_gpu(c, TOT)
        c.enqueue_function[
            _mse_kernel[B, CHUNK, ADIM, ADIM_REAL]
        ](
            v_t.tt["gpu", B, CHUNK * ADIM](),
            u_t.tt["gpu", B, CHUNK * ADIM](),
            valid.tt["gpu", B, CHUNK](),
            grad_v.tt["gpu", B, CHUNK * ADIM](),
            err.tt["gpu", B, CHUNK * ADIM](),
            two_over_n,
            grid_dim=(TOT + TPB - 1) // TPB,
            block_dim=TPB,
        )


def mean_err[
    target: StaticString, B: Int, CHUNK: Int, ADIM: Int, ADIM_REAL: Int
](
    mut err: Tensor, n_terms: Int, ctx: Optional[DeviceContext] = None
) raises -> Float64:
    """The scalar loss. ⚠ SYNCHRONISES on GPU — call it when you want to log.

    Divides by `n_terms * ADIM_REAL`, the count of terms that are not
    structurally zero, matching `losses.sum() / num_valid`. Dividing by the
    padded total would report a loss scaled by the fraction of real columns
    and real timesteps — which varies per batch, so two batches' losses would
    not be comparable and neither would two runs.
    """
    comptime TOT = B * CHUNK * ADIM
    if n_terms <= 0:
        raise Error("mean_err: n_terms is 0")
    comptime if target != "cpu":
        err.download(ctx.value())
    var acc = 0.0
    for i in range(TOT):
        acc += Float64(err.data[i])
    return acc / Float64(n_terms * ADIM_REAL)
