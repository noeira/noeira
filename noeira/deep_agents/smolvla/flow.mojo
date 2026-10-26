# +--------------------------------------------------------------------------+ #
# | SmolVLA — flow-matching sampling (forward Euler)
# +--------------------------------------------------------------------------+ #
"""Ten Euler steps from noise to an action chunk.

`euler_integrate` (openpi's, via `lerobot/policies/common/flow_matching.py`):

    dt   = -1 / num_steps
    x_t  = noise                                  # x_1
    for step in range(num_steps):
        time = 1.0 + step * dt                    # 1.0, 0.9, …, 0.1
        v_t  = denoise_fn(x_t, time)
        x_t  = x_t + dt * v_t
    return x_t                                    # x_0

⚠ **`dt` is NEGATIVE.** Integration runs from t=1 (noise) toward t=0 (actions),
so each step SUBTRACTS a tenth of the velocity. Flipping the sign integrates
away from the data manifold and returns finite, plausibly-scaled, entirely wrong
actions.

⚠ **`time` never reaches 0.** It starts at exactly 1.0 and the last step uses
0.1; `num_steps` steps cover `[1.0, dt]`, not `[1.0, 0.0]`. An off-by-one that
starts at `1.0 + dt` or ends at 0 shifts every conditioning value.

The schedule is a separate struct from the model precisely so it can be tested
without one: with a CONSTANT velocity field `v ≡ c`, the sum of the steps is
`Σ dt·c = -c` exactly, so `x_final == x_0 - c` — an analytic check that pins both
the sign and the step count with no reference dump.

## The per-token concat

`embed_suffix` builds `cat([action_emb, time_emb], dim=2)`: **per token**, each
720-wide action embedding followed by the same 720-wide time embedding, giving
the 1440 that `action_time_mlp_in` takes.

⚠ `Concat2[CHUNK*720, CHUNK*720]` would NOT do this. It concatenates whole
flattened rows — every action embedding, then every time embedding — which has
the identical total width and interleaves nothing correctly. The token-major
layout is why this needs its own helper.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.core.tiles import Tile


struct EulerSchedule[STEPS: Int](Movable):
    """The `dt`/`time` schedule, and the update, with no model attached."""

    def __init__(out self):
        comptime assert Self.STEPS >= 1, "EulerSchedule: STEPS must be >= 1"

    @staticmethod
    def dt() -> Float64:
        """Negative: t runs 1 -> 0."""
        return -1.0 / Float64(Self.STEPS)

    @staticmethod
    def time_at(step: Int) raises -> Float64:
        """`1.0 + step*dt`, so `time_at(0) == 1.0` and the last is `-dt`."""
        if step < 0 or step >= Self.STEPS:
            raise Error(
                "EulerSchedule.time_at: step " + String(step) + " out of "
                + String(Self.STEPS)
            )
        return 1.0 + Float64(step) * Self.dt()

    @staticmethod
    def advance[
        target: StaticString, N: Int
    ](
        mut x: Tensor, mut v: Tensor, ctx: Optional[DeviceContext] = None
    ) raises:
        """`x <- x + dt * v`, in place over `N` elements."""
        var d = Scalar[DT](Self.dt())
        comptime if target == "cpu":
            for i in range(N):
                x.data[i] = x.data[i] + d * v.data[i]
        else:
            var c = ctx.value()
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_euler_step_kernel[N]](
                x.tt["gpu", N](),
                v.tt["gpu", N](),
                d,
                grid_dim=nb,
                block_dim=TPB,
            )


def _euler_step_kernel[
    N: Int
](
    x: Tile[DT, N],
    v: Tile[DT, N],
    dt: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i >= N:
        return
    x.ptr[unsafe_offset=i] = rebind[Scalar[DT]](
        x.ptr[unsafe_offset=i]
    ) + dt * rebind[Scalar[DT]](v.ptr[unsafe_offset=i])


def _token_concat_kernel[
    BATCH: Int, SEQ: Int, DA: Int, DB: Int, B_STRIDE: Int
](
    a: Tile[DT, BATCH, SEQ * DA],
    b: Tile[DT, DB],
    dst: Tile[DT, BATCH, SEQ * (DA + DB)],
):
    comptime D = DA + DB
    var idx = Int(global_idx.x)
    if idx >= BATCH * SEQ * D:
        return
    var d = idx % D
    var r = idx // D
    var t = r % SEQ
    var bi = r // SEQ
    if d < DA:
        dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
            a.ptr[unsafe_offset = bi * (SEQ * DA) + t * DA + d]
        )
    else:
        # `b` is ONE vector shared by every token — the reference's
        # `time_emb[:, None, :].expand_as(action_emb)`. B_STRIDE = 0 shares it
        # across the batch too (inference: one timestep per query); B_STRIDE =
        # DB gives each sample its own (training: `time` is [B]).
        dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
            b.ptr[unsafe_offset = bi * B_STRIDE + d - DA]
        )


def token_concat[
    target: StaticString, BATCH: Int, SEQ: Int, DA: Int, DB: Int,
    B_STRIDE: Int = 0,
](
    mut a: Tensor, mut b: Tensor, mut dst: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`dst[b, t] = [a[b, t] ‖ b]`, token by token.

    `b` is a single `DB`-vector broadcast to every token — the time embedding,
    which does not vary along the chunk.

    ⚠ `B_STRIDE` says whether it also broadcasts across the BATCH. Inference
    denoises one query at a shared timestep, so `b` is `[DB]` and the stride is
    0. Training samples `t` PER SAMPLE (`time[:, None, None]`), so `b` is
    `[BATCH, DB]` and the stride is `DB`. One index expression, not two
    routines: the alternative is this layout rule written twice, which is the
    defect this codebase produces most often.
    """
    comptime D = DA + DB
    comptime N = BATCH * SEQ * D
    comptime if target == "cpu":
        dst.ensure(N)
        for bi in range(BATCH):
            for t in range(SEQ):
                var o = bi * (SEQ * D) + t * D
                var ai = bi * (SEQ * DA) + t * DA
                for d in range(DA):
                    dst.data[o + d] = a.data[ai + d]
                for d in range(DB):
                    dst.data[o + DA + d] = b.data[bi * B_STRIDE + d]
    else:
        var c = ctx.value()
        dst.ensure_gpu(c, N)
        comptime nb = (N + TPB - 1) // TPB
        c.enqueue_function[
            _token_concat_kernel[BATCH, SEQ, DA, DB, B_STRIDE]
        ](
            a.tt["gpu", BATCH, SEQ * DA](),
            b.tt["gpu", DB](),
            dst.tt["gpu", BATCH, SEQ * D](),
            grid_dim=nb,
            block_dim=TPB,
        )


def _token_split_a_kernel[
    BATCH: Int, SEQ: Int, DA: Int, DB: Int
](
    src: Tile[DT, BATCH, SEQ * (DA + DB)],
    dst: Tile[DT, BATCH, SEQ * DA],
):
    comptime D = DA + DB
    var idx = Int(global_idx.x)
    if idx >= BATCH * SEQ * DA:
        return
    var d = idx % DA
    var r = idx // DA
    var t = r % SEQ
    var bi = r // SEQ
    dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset = bi * (SEQ * D) + t * D + d]
    )


def token_split_a[
    target: StaticString, BATCH: Int, SEQ: Int, DA: Int, DB: Int
](
    mut src: Tensor, mut dst: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`token_concat`'s adjoint, for the `a` operand only.

    Concatenation is a gather, so its backward is a scatter — but only the
    first `DA` columns of each token go anywhere. The `b` half's gradient
    would be the sum over tokens (and, at `B_STRIDE = 0`, over the batch), and
    it is DELIBERATELY not computed: `b` is the sinusoidal time embedding,
    which is a fixed function of a sampled scalar and owns no parameter. There
    is nothing for that gradient to reach.

    ⚠ Lives beside `token_concat` on purpose. The layout rule — `[a_t ‖ b]`
    per token, not `[all a ‖ all b]` — is stated once here and inverted once
    here, so the two cannot drift apart in different files.
    """
    comptime D = DA + DB
    comptime N = BATCH * SEQ * DA
    comptime if target == "cpu":
        dst.ensure(N)
        for bi in range(BATCH):
            for t in range(SEQ):
                var o = bi * (SEQ * D) + t * D
                var ai = bi * (SEQ * DA) + t * DA
                for d in range(DA):
                    dst.data[ai + d] = src.data[o + d]
    else:
        var c = ctx.value()
        dst.ensure_gpu(c, N)
        comptime nb = (N + TPB - 1) // TPB
        c.enqueue_function[_token_split_a_kernel[BATCH, SEQ, DA, DB]](
            src.tt["gpu", BATCH, SEQ * D](),
            dst.tt["gpu", BATCH, SEQ * DA](),
            grid_dim=nb,
            block_dim=TPB,
        )
