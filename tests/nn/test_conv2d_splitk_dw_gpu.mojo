"""Does routing `Conv2D`'s dW GEMM through our own split-K workspace change
the gradient?

`Conv2D`'s backward computes `dW[OC, CPAD] = goᵀ[OC, BS] @ col[BS, CPAD]`,
where `BS = batch * OH * OW`. That contraction is far longer than the
transformer dW's `batch * tokens` — a ResNet18 stem at batch 16 and 240x320
input gives BS = 307,200 — while M and N stay small (an out-channel count and
an im2col column count). Long K, tiny tile grid: exactly the regime split-K
exists for, and the one `select_config` under-partitions worst.

Same in-process A/B as `test_linear_splitk_dw_gpu.mojo`: two identical
`Conv2D`s, same weights and input, one pinned to `_sk_p = 1` so it takes plain
`max_matmul`. Their `grad_w` must agree.

⚠ VACUITY IS THE FAILURE MODE. `Conv2D` has TWO GPU dW sites (fp32 and
bf16-flow), and `Linear`'s integration shipped once with only one of them
routed — five shapes printed `0.0` on a complete no-op and only the partition
count gave it away. So this prints P per shape and RAISES if nothing split.
A green run reading `split shapes: 0` is a red run.

    pixi run -e nvidia mojo run -I . tests/nn/test_conv2d_splitk_dw_gpu.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, LAYOUT_NCHW
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.splitk_gemm import splitk_path_applies
from noeira.nn.core.cublas_gemm import GEMM_PATH
from std.sys import has_nvidia_gpu_accelerator
from noeira.nn.primitives.conv2d import Conv2D


def check[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, B: Int,
    EXPECT_SPLIT: Bool,
    WHY: StaticString,
](ctx: DeviceContext, mut n_split: Int) raises:
    comptime C = Conv2D[IC, OC, K, S, P, H, W]
    comptime BS = B * C.OH * C.OW
    comptime IN_N = B * IC * H * W
    comptime OUT_N = B * OC * C.OH * C.OW

    var ca = C.make["gpu", INIT=Kaiming](ctx=ctx)   # may split
    var cb = C.make["gpu", INIT=Kaiming](ctx=ctx)   # pinned to plain matmul
    cb._sk_p = 1

    ca.weight.val.ensure_host(ctx, C.W_SIZE)
    cb.weight.val.ensure_host(ctx, C.W_SIZE)
    for i in range(C.W_SIZE):
        cb.weight.val.data[i] = ca.weight.val.data[i]
    ca.weight.val.upload_resident(ctx)
    cb.weight.val.upload_resident(ctx)

    var xa = Tensor.alloc(IN_N)
    var xb = Tensor.alloc(IN_N)
    for i in range(IN_N):
        var v = Scalar[DT](0.01) * Scalar[DT]((i % 37) - 18)
        xa.data[i] = v
        xb.data[i] = v
    xa.upload(ctx)
    xb.upload(ctx)

    var ya = Tensor.alloc_gpu(ctx, OUT_N)
    var yb = Tensor.alloc_gpu(ctx, OUT_N)
    ca.forward["gpu", B](TensorRefs[1](xa), ya, Optional(ctx))
    cb.forward["gpu", B](TensorRefs[1](xb), yb, Optional(ctx))

    var goa = Tensor.alloc(OUT_N)
    var gob = Tensor.alloc(OUT_N)
    for i in range(OUT_N):
        var v = Scalar[DT](0.003) * Scalar[DT]((i % 23) - 11)
        goa.data[i] = v
        gob.data[i] = v
    goa.upload(ctx)
    gob.upload(ctx)

    var gia = Tensor.alloc_gpu(ctx, IN_N)
    var gib = Tensor.alloc_gpu(ctx, IN_N)
    ca.vjp["gpu", B](TensorRefs[1](xa), goa, TensorRefs[1](gia), Optional(ctx))
    cb.vjp["gpu", B](TensorRefs[1](xb), gob, TensorRefs[1](gib), Optional(ctx))

    ca.weight.grd.download(ctx)
    cb.weight.grd.download(ctx)
    ctx.synchronize()

    var max_abs = Float64(0)
    var mag = Float64(0)
    for i in range(C.W_SIZE):
        var a = Float64(ca.weight.grd.data[i])
        var b = Float64(cb.weight.grd.data[i])
        if abs(b) > mag:
            mag = abs(b)
        var d = abs(a - b)
        if d > max_abs:
            max_abs = d
    var rel = (max_abs / mag) if mag > 0.0 else 0.0

    if ca._sk_p > 1:
        n_split += 1

    print(
        "  IC=", IC, " OC=", OC, " K=", K, " S=", S, " ", H, "x", W,
        " B=", B, "   BS=", BS, " CPAD=", C.CPAD, " (n%128=", C.CPAD % 128, ")",
        "   P=", ca._sk_p,
        "   rel=", rel,
        "   [", WHY, "]",
        sep="",
    )

    # Assert the ROUTING DECISION per shape, not just that something split.
    # Counting alone let `Linear` ship with one of two dW sites unrouted.
    var did_split = ca._sk_p > 1
    if did_split != EXPECT_SPLIT:
        raise Error(
            "routing decision does not match the shape's own arithmetic:"
            " expected split=" + String(EXPECT_SPLIT)
            + " got P=" + String(ca._sk_p) + " — " + String(WHY)
        )

    # P separate fp32 accumulations, so not bit-identical once P > 1 -- but it
    # must stay at rounding. A dropped K tail shows up here as a relative error
    # equal to the fraction of K lost. K is ~3e5 here, so even ONE lost BK tile
    # is ~5e-5 -- well above this bound.
    if rel > 1e-4:
        raise Error("split-K dW disagrees with the plain GEMM beyond fp32 noise")


def main() raises:
    comptime if has_nvidia_gpu_accelerator() and GEMM_PATH != "max":
        # On NVIDIA these shapes run on cuBLAS / cuDNN (`Conv2D.CUB`,
        # `Conv2D.use_cudnn`), never on MAX's split-K GEMM. Build with
        # `-D NN_GEMM_PATH=max` to test that path; the GPU-vs-CPU gate for the
        # NVIDIA paths is `test_conv2d_gpu`.
        print("NVIDIA: Conv2D does not use MAX's split-K GEMM here (cuBLAS / cuDNN); see test_conv2d_gpu")
        return
    comptime if not splitk_path_applies[DeviceContext.default_device_info]():
        print(
            "split-K path does not apply on this device — Conv2D unchanged"
            " here, nothing to test."
        )
        return

    with DeviceContext() as ctx:
        print("conv dW = goT @ col: M=OC, N=CPAD, K=BS = batch*OH*OW")
        var n_split = 0

        print("== conv dW that REACHES multistage (CPAD % 128 == 0) ==")
        # layer2-shaped: COL = 128*3*3 = 1152, and 1152 % 128 == 0.
        check[128, 256, 3, 1, 1, 30, 40, 16, True,
              "CPAD=1152, BS=19200"](ctx, n_split)
        check[128, 256, 3, 1, 1, 15, 20, 16, True,
              "CPAD=1152, BS=4800"](ctx, n_split)

        print("== previously EXCLUDED, now admitted by PAD_TO=128 ==")
        # These two were the reason PAD_TO moved from 32 to 128. `CPAD` used to
        # round the im2col column count to 32 -- the FORWARD's alignment -- and
        # for the dW GEMM that same number is N, which multistage needs at a
        # multiple of 128. They took the vendor fallback, which allocates and
        # memsets 32 MB per call and cannot be captured.
        check[3, 64, 7, 2, 3, 240, 320, 16, True,
              "COL=147 -> CPAD=256"](ctx, n_split)
        check[64, 64, 3, 1, 1, 60, 80, 16, True,
              "COL=576 -> CPAD=640"](ctx, n_split)

        print("== control: passes multi_gemm_cond, BS below min_k_partition ==")
        check[256, 512, 3, 1, 1, 8, 10, 2, False,
              "CPAD=2304 ok, but BS=160 < 1024"](ctx, n_split)

        print()
        print("split shapes:", n_split, "of 5  (4 split, 1 control)")
        if n_split == 0:
            raise Error(
                "NO shape took the split-K path — this run tested nothing."
                " Conv2D has TWO GPU dW sites and only the fp32 one is routed;"
                " check that before reading the zeros above as a pass."
            )
        print("all good")
