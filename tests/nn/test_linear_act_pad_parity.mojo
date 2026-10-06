"""Does `LinearAct`'s newly-padded GPU path still compute the same layer?

`LinearAct` used to pad its contraction dim to a multiple of 32 with no floor
and never padded N at all, so `LinearReLU[6, 256]` — the first layer of every
SAC / TD3 / DDPG / PPO / DQN trunk here — ran its forward at K=32 (fails
`multi_gemm_cond`'s `k >= 128`) and its grad_input at N=6 (fails `n % 128`).
Both went to the cuBLAS vendor path, which allocates and memsets 32 MB per
call. Both dims are now padded to 128.

Padding is supposed to be arithmetically inert: the appended rows and columns
are exactly zero, so every dot product is unchanged. This test is that claim.

THE CONTROL IS A ZERO-EXTENDED MODULE ON THE SAME DEVICE, NOT THE CPU
---------------------------------------------------------------------
`LinearAct[6, 256]` pads K to 128, so its GEMM is literally `[B,128] @
[128,256]`. `LinearAct[128, 256]` is ALREADY aligned, takes the old unpadded
branch, and issues that same GEMM — so feeding it an input whose columns 6..128
are zero and a weight whose rows 6..128 are zero makes the two mathematically
identical, on the same device, at the same precision. They must agree BIT FOR
BIT, and any difference is the new padding code.

⚠ THE FIRST VERSION USED THE CPU PATH AS THE REFERENCE AND THAT WAS THE WRONG
COMPARISON. It passed bit-exact on Apple and failed on a 5090 with

    out 8.2e-4   grad_in 6.5e-2   grad_w 1.9e-1   grad_b 3.0e-1

which is not a padding bug: a 5090 runs fp32 matmul through TF32 tensor cores,
8.2e-4 IS TF32's precision, and ReLU's derivative is DISCONTINUOUS — a
pre-activation inside that noise band has its sign decided by rounding, so one
arm propagates a gradient through the unit and the other zeroes it. The
difference is the size of whatever gradient was flowing, up to the tensor max,
and it is set by the gradient magnitude rather than by the precision. Raising
the tolerance does not fix that; it just waits for a bigger gradient to land on
the boundary (see `_a_discontinuity_makes_per_element_parity_undecidable` —
this repo has already lost three rounds to exactly that). Removing TF32 from
the comparison, by putting both arms on the same device, does fix it.

⚠ THIS RUNS ON APPLE TOO. `PADDED` is a property of `IN_`/`OUT_`, not of the
device, so the padded branch is live on Metal as well.

    pixi run mojo run -I . tests/nn/test_linear_act_pad_parity.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_linear_act_pad_parity.mojo
"""

from std.math import abs
from noeira.nn.core.cublas_gemm import CUBLAS_BWD
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.linear_act import LinearAct
from noeira.nn.primitives.ops.relu_op import ReLUOp


def cmp[
    ROWS: Int, COLS: Int, A_STRIDE: Int, B_STRIDE: Int
](mut a: Tensor, mut b: Tensor, label: String) raises -> Float64:
    """Max |a - b| over `[ROWS, COLS]`, relative to b's magnitude.

    The two operands have DIFFERENT row strides — one module is the padded
    shape and the other the zero-extended one — so this cannot be a flat loop.

    Vacuity guard included: `max_abs / mag` returns a perfect 0.0 when the
    reference is all zeros, which is indistinguishable from a real pass in the
    output. It has already happened once in this file.
    """
    var max_abs = Float64(0)
    var mag = Float64(0)
    var nz = 0
    for r in range(ROWS):
        for cc in range(COLS):
            var av = Float64(a.data[r * A_STRIDE + cc])
            var bv = Float64(b.data[r * B_STRIDE + cc])
            if abs(bv) > mag:
                mag = abs(bv)
            if bv != 0.0:
                nz += 1
            var d = abs(av - bv)
            if d > max_abs:
                max_abs = d
    if mag == 0.0:
        raise Error(
            label + ": reference is ALL ZEROS over " + String(ROWS * COLS)
            + " elements — this comparison tested nothing"
        )
    if nz * 20 < ROWS * COLS:
        raise Error(
            label + ": only " + String(nz) + " of " + String(ROWS * COLS)
            + " reference elements are non-zero — too sparse to compare"
        )
    return max_abs / mag


def _verdict(label: String, e: Float64) raises:
    # Same device, same precision, mathematically identical GEMMs: the only
    # licence here is fp32 reduction ORDER, and the two arms issue the SAME
    # shape, so even that should not move. Anything above zero is worth
    # looking at; 1e-6 is the bound because a tile-scheduling difference is
    # conceivable and a padding bug is never this small.
    if e > 1e-6:
        raise Error(
            label + " disagrees with the zero-extended control (" + String(e)
            + ") — the padding is NOT arithmetically inert"
        )


def check_k[
    IN: Int, OUT: Int, B: Int
](ctx: DeviceContext, mut n_run: Int) raises:
    """K padding: `LinearAct[IN, OUT]` vs an aligned `LinearAct[K_PAD, OUT]`
    fed zero-extended inputs and weights."""
    comptime P = LinearAct[IN, OUT, ReLUOp]
    comptime KP = P.K_PAD
    comptime U = LinearAct[KP, OUT, ReLUOp]
    comptime assert P.NEEDS_PAD, "check_k needs a K-padded shape"
    comptime assert not U.PADDED, "the control must take the unpadded branch"
    n_run += 1

    var p = P.make["gpu", INIT=Kaiming](ctx=ctx)
    var u = U.make["gpu", INIT=Kaiming](ctx=ctx)

    p.weight.val.ensure_host(ctx, P.W_SIZE)
    p.bias.val.ensure_host(ctx, P.B_SIZE)
    u.weight.val.ensure_host(ctx, U.W_SIZE)
    u.bias.val.ensure_host(ctx, U.B_SIZE)
    # u.weight = [[p.weight], [0]]  (rows IN..K_PAD are zero)
    for i in range(U.W_SIZE):
        u.weight.val.data[i] = Scalar[DT](0)
    for i in range(P.W_SIZE):
        u.weight.val.data[i] = p.weight.val.data[i]
    for j in range(OUT):
        u.bias.val.data[j] = p.bias.val.data[j]
    u.weight.val.upload_resident(ctx)
    u.bias.val.upload_resident(ctx)

    var xp = Tensor.alloc(B * IN)
    var xu = Tensor.alloc(B * KP)
    for i in range(B * KP):
        xu.data[i] = Scalar[DT](0)
    for b in range(B):
        for k in range(IN):
            var v = Scalar[DT](0.01) * Scalar[DT](((b * IN + k) % 37) - 18)
            xp.data[b * IN + k] = v
            xu.data[b * KP + k] = v
    xp.upload(ctx)
    xu.upload(ctx)

    var yp = Tensor.alloc_gpu(ctx, B * OUT)
    var yu = Tensor.alloc_gpu(ctx, B * OUT)
    p.forward["gpu", B](TensorRefs[1](xp), yp, Optional(ctx))
    u.forward["gpu", B](TensorRefs[1](xu), yu, Optional(ctx))

    var gop = Tensor.alloc(B * OUT)
    var gou = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        var v = Scalar[DT](0.003) * Scalar[DT]((i % 23) - 11)
        gop.data[i] = v
        gou.data[i] = v
    gop.upload(ctx)
    gou.upload(ctx)

    var gip = Tensor.alloc_gpu(ctx, B * IN)
    var giu = Tensor.alloc_gpu(ctx, B * KP)
    p.vjp["gpu", B](TensorRefs[1](xp), gop, TensorRefs[1](gip), Optional(ctx))
    u.vjp["gpu", B](TensorRefs[1](xu), gou, TensorRefs[1](giu), Optional(ctx))

    yp.download(ctx); yu.download(ctx)
    gip.download(ctx); giu.download(ctx)
    p.weight.grd.download(ctx); u.weight.grd.download(ctx)
    p.bias.grd.download(ctx); u.bias.grd.download(ctx)
    ctx.synchronize()

    var e_out = cmp[B, OUT, OUT, OUT](yp, yu, "out")
    var e_gi = cmp[B, IN, IN, KP](gip, giu, "grad_in")
    var e_gw = cmp[IN, OUT, OUT, OUT](p.weight.grd, u.weight.grd, "grad_w")
    var e_gb = cmp[1, OUT, OUT, OUT](p.bias.grd, u.bias.grd, "grad_b")
    print(
        "  K-pad  IN=", IN, "->", KP, " OUT=", OUT, " B=", B,
        "    out ", e_out, "  grad_in ", e_gi,
        "  grad_w ", e_gw, "  grad_b ", e_gb, sep="",
    )
    _verdict("out", e_out); _verdict("grad_in", e_gi)
    _verdict("grad_w", e_gw); _verdict("grad_b", e_gb)


def check_n[
    IN: Int, OUT: Int, B: Int
](ctx: DeviceContext, mut n_run: Int) raises:
    """N padding: `LinearAct[IN, OUT]` vs an aligned `LinearAct[IN, N_PAD]`
    whose weight columns, bias and grad_output are zero past `OUT`."""
    comptime P = LinearAct[IN, OUT, ReLUOp]
    comptime NP = P.N_PAD
    comptime U = LinearAct[IN, NP, ReLUOp]
    comptime assert P.NEEDS_N_PAD, "check_n needs an N-padded shape"
    comptime assert not U.PADDED, "the control must take the unpadded branch"
    n_run += 1

    var p = P.make["gpu", INIT=Kaiming](ctx=ctx)
    var u = U.make["gpu", INIT=Kaiming](ctx=ctx)

    p.weight.val.ensure_host(ctx, P.W_SIZE)
    p.bias.val.ensure_host(ctx, P.B_SIZE)
    u.weight.val.ensure_host(ctx, U.W_SIZE)
    u.bias.val.ensure_host(ctx, U.B_SIZE)
    for i in range(U.W_SIZE):
        u.weight.val.data[i] = Scalar[DT](0)
    for r in range(IN):
        for j in range(OUT):
            u.weight.val.data[r * NP + j] = p.weight.val.data[r * OUT + j]
    for j in range(NP):
        u.bias.val.data[j] = (
            p.bias.val.data[j] if j < OUT else Scalar[DT](0)
        )
    u.weight.val.upload_resident(ctx)
    u.bias.val.upload_resident(ctx)

    var xp = Tensor.alloc(B * IN)
    var xu = Tensor.alloc(B * IN)
    for i in range(B * IN):
        var v = Scalar[DT](0.01) * Scalar[DT]((i % 37) - 18)
        xp.data[i] = v
        xu.data[i] = v
    xp.upload(ctx)
    xu.upload(ctx)

    var yp = Tensor.alloc_gpu(ctx, B * OUT)
    var yu = Tensor.alloc_gpu(ctx, B * NP)
    p.forward["gpu", B](TensorRefs[1](xp), yp, Optional(ctx))
    u.forward["gpu", B](TensorRefs[1](xu), yu, Optional(ctx))

    var gop = Tensor.alloc(B * OUT)
    var gou = Tensor.alloc(B * NP)
    for i in range(B * NP):
        gou.data[i] = Scalar[DT](0)
    for b in range(B):
        for j in range(OUT):
            var v = Scalar[DT](0.003) * Scalar[DT](((b * OUT + j) % 23) - 11)
            gop.data[b * OUT + j] = v
            gou.data[b * NP + j] = v
    gop.upload(ctx)
    gou.upload(ctx)

    var gip = Tensor.alloc_gpu(ctx, B * IN)
    var giu = Tensor.alloc_gpu(ctx, B * IN)
    p.vjp["gpu", B](TensorRefs[1](xp), gop, TensorRefs[1](gip), Optional(ctx))
    u.vjp["gpu", B](TensorRefs[1](xu), gou, TensorRefs[1](giu), Optional(ctx))

    yp.download(ctx); yu.download(ctx)
    gip.download(ctx); giu.download(ctx)
    p.weight.grd.download(ctx); u.weight.grd.download(ctx)
    p.bias.grd.download(ctx); u.bias.grd.download(ctx)
    ctx.synchronize()

    var e_out = cmp[B, OUT, OUT, NP](yp, yu, "out")
    var e_gi = cmp[B, IN, IN, IN](gip, giu, "grad_in")
    var e_gw = cmp[IN, OUT, OUT, NP](p.weight.grd, u.weight.grd, "grad_w")
    var e_gb = cmp[1, OUT, OUT, NP](p.bias.grd, u.bias.grd, "grad_b")
    print(
        "  N-pad  IN=", IN, " OUT=", OUT, "->", NP, " B=", B,
        "    out ", e_out, "  grad_in ", e_gi,
        "  grad_w ", e_gw, "  grad_b ", e_gb, sep="",
    )
    _verdict("out", e_out); _verdict("grad_in", e_gi)
    _verdict("grad_w", e_gw); _verdict("grad_b", e_gb)


def main() raises:
    comptime if CUBLAS_BWD:
        # The padded branch does not run on NVIDIA (unless built with
        # `-D NN_GEMM_PATH=max`): a shape that would need padding goes through
        # `cublas_gemm` unpadded (`use_cublas_fwd`), and so does the backward.
        # Nothing here to compare; the GPU-vs-CPU gate for those paths is
        # `test_linear_act_gpu`.
        print("NVIDIA: LinearAct's padded path is not used (cuBLAS); see test_linear_act_gpu")
        return
    with DeviceContext() as ctx:
        print("LinearAct padded vs a ZERO-EXTENDED aligned control, same device")
        var n_run = 0

        print("== K padded (the RL trunk's first layer) ==")
        check_k[6, 256, 64](ctx, n_run)      # SAC/TD3 obs -> hidden
        check_k[17, 256, 64](ctx, n_run)     # a wider obs
        check_k[200, 256, 64](ctx, n_run)    # K padded, N already aligned

        print("== N padded ==")
        check_n[256, 100, 64](ctx, n_run)
        check_n[128, 30, 64](ctx, n_run)     # an action head width

        print()
        print("shapes compared:", n_run, "of 5")
        if n_run < 5:
            raise Error("not every shape ran")
        print("all good")
