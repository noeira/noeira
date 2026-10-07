"""Does routing `Conv2D`'s FORWARD GEMM through our own split-K workspace
change the output?

The dW got this treatment first, on the assumption that a weight gradient is
where a long contraction lives. That assumption was wrong and it cost a
CUDA-graph capture. `Conv2D`'s forward is

    out[BS, OCPAD] = col[BS, CPAD] @ wᵀ[OCPAD, CPAD]

whose contraction is `CPAD = IC * K * K` — 2304 for a 3x3 at IC=256, 4608 at
IC=512. `select_config` partitions any `K >= 2048`, so ResNet18's layer3 and
layer4 forwards ALL take MAX's split-K path and allocate `P * BS * OCPAD * 4`
bytes per call. On the ACT step that is

    MxNxK: 9600x256x2304, K partitions: 2
    workspace = 2 * 9600 * 256 * 4 = 19,660,800 B = 18.75 MB

which is exactly the allocation that aborted the capture, found with
`-D LOGGING_LEVEL=INFO` after `NOEIRA_ALLOC_TRACE` proved it was not ours.

⚠ The lesson is in the shape, not the gradient: ANY `max_matmul` inside a
captured region can hit MAX's split-K, and `K >= 2048` with `n % 128 == 0` is
the whole test. Do not assume a call is safe because it is a forward.

This is the in-process A/B: two identical `Conv2D`s, same weights, same input,
one pinned to `_sk_p_fwd = 1` so it takes plain `max_matmul`. Their forward
outputs must agree.

    pixi run -e nvidia mojo run -I . tests/nn/test_conv2d_splitk_fwd_gpu.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
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
](ctx: DeviceContext, mut n_split: Int) raises:
    comptime L = Conv2D[IC, OC, K, S, P, H, W]

    var la = L.make["gpu", INIT=Kaiming](ctx=ctx)   # may split
    var lb = L.make["gpu", INIT=Kaiming](ctx=ctx)   # pinned to plain matmul
    lb._sk_p_fwd = 1

    la.weight.val.ensure_host(ctx, L.W_SIZE)
    lb.weight.val.ensure_host(ctx, L.W_SIZE)
    la.bias.val.ensure_host(ctx, L.B_SIZE)
    lb.bias.val.ensure_host(ctx, L.B_SIZE)
    for i in range(L.W_SIZE):
        lb.weight.val.data[i] = la.weight.val.data[i]
    for i in range(L.B_SIZE):
        lb.bias.val.data[i] = la.bias.val.data[i]
    la.weight.val.upload_resident(ctx)
    lb.weight.val.upload_resident(ctx)
    la.bias.val.upload_resident(ctx)
    lb.bias.val.upload_resident(ctx)

    var xa = Tensor.alloc(B * L.IN_FLAT)
    var xb = Tensor.alloc(B * L.IN_FLAT)
    # Sign-changing fill: a split that drops part of K shows up as a relative
    # error equal to the fraction lost, which only reads correctly when the
    # summands do not all share a sign.
    for i in range(B * L.IN_FLAT):
        var v = Scalar[DT](0.01) * Scalar[DT]((i % 37) - 18)
        xa.data[i] = v
        xb.data[i] = v
    xa.upload(ctx)
    xb.upload(ctx)

    var ya = Tensor.alloc_gpu(ctx, B * L.OUT_FLAT)
    var yb = Tensor.alloc_gpu(ctx, B * L.OUT_FLAT)
    la.forward["gpu", B](TensorRefs[1](xa), ya, Optional(ctx))
    lb.forward["gpu", B](TensorRefs[1](xb), yb, Optional(ctx))

    ya.download(ctx)
    yb.download(ctx)
    ctx.synchronize()

    var max_abs = Float64(0)
    var mag = Float64(0)
    for i in range(B * L.OUT_FLAT):
        var a = Float64(ya.data[i])
        var b = Float64(yb.data[i])
        if abs(b) > mag:
            mag = abs(b)
        var d = abs(a - b)
        if d > max_abs:
            max_abs = d
    var rel = (max_abs / mag) if mag > 0.0 else 0.0

    if la._sk_p_fwd > 1:
        n_split += 1
    if (la._sk_p_fwd > 1) != EXPECT_SPLIT:
        raise Error(
            "routing decision does not match the shape's arithmetic: expected"
            " split=" + String(EXPECT_SPLIT)
            + " got P=" + String(la._sk_p_fwd)
            + " (CPAD=" + String(L.CPAD) + ")"
        )

    print(
        "  IC=", IC, " OC=", OC, " ", K, "x", K,
        "   BS=", B * L.SO, " CPAD=", L.CPAD, " OCPAD=", L.OCPAD,
        "   P=", la._sk_p_fwd,
        "   max|split - plain|=", max_abs, "  rel=", rel,
        sep="",
    )
    if rel > 1e-4:
        raise Error("split-K forward disagrees with the plain GEMM")


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
            "split-K path does not apply on this device (Apple / AMD / H100 /"
            " sm_100 Blackwell) — Conv2D's forward is unchanged here."
        )
        return

    with DeviceContext() as ctx:
        print("forward: out[BS, OCPAD] = col[BS, CPAD] @ wT,  K = CPAD")
        var n_split = 0

        print("== CPAD >= 2048: MAX would split, so we must ==")
        # IC=256 3x3 -> CPAD = 2304. This is ResNet18 layer3, the shape that
        # aborted the ACT capture (there at BS=9600).
        check[256, 256, 3, 1, 1, 15, 20, 8, True](ctx, n_split)
        # IC=512 3x3 -> CPAD = 4608. ResNet18 layer4.
        check[512, 512, 3, 1, 1, 8, 10, 4, True](ctx, n_split)

        print("== CPAD < 2048: below min_k_partition, must NOT split ==")
        # IC=64 3x3 -> COL 576 -> CPAD 640. Layer1: MAX runs this unpartitioned
        # and so must we.
        check[64, 64, 3, 1, 1, 30, 40, 8, False](ctx, n_split)
        # IC=128 3x3 -> CPAD 1152. Layer2 — still under the 2048 floor, which
        # is worth pinning: it is the nearest shape ABOVE the ones that split.
        check[128, 128, 3, 1, 1, 15, 20, 8, False](ctx, n_split)
        # 1x1 downsample: CPAD = IC = 256, far under.
        check[256, 512, 1, 2, 0, 15, 20, 8, False](ctx, n_split)

        print()
        print("split shapes:", n_split, "of 5")
        if n_split == 0:
            raise Error(
                "NO shape took the split-K path — this run tested nothing."
                " Check select_config's min_k_partition and the device gate"
                " before reading the zeros above as a pass."
            )
        if n_split < 2:
            raise Error(
                "both CPAD >= 2048 shapes should split; a shortfall means the"
                " forward site is not routed"
            )
        print("all good")
