"""One `Linear.vjp`, trainable vs frozen, at token counts either side of 16.

    pixi run -e apple mojo run -I . benchmarks/peft/bench_linear_vjp_tokens.mojo

On Apple, `bench_prefill_no_dw.mojo` measured freezing at 4.2x on the whole
prefill backward, where removing one GEMM of dX's size would predict about 2x.
The suspect is the dW GEMM's CONTRACTION: dW = xᵀ @ go contracts over the
token count B (140 in the prefill), which nothing pads, while MAX's Metal GEMM
wants K % 16 (`Linear`'s K-alignment note). dX contracts over OUT, which the
N padding already rounds to 128. If that is the cause, the trainable/frozen
ratio falls back towards 2x at B = 144 and stays high at B = 140.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear

comptime REPS = 20


def median_vjp[IN: Int, OUT: Int, B: Int](
    mut l: Linear[IN, OUT], mut x: Tensor, mut go: Tensor, mut gi: Tensor,
    ctx: DeviceContext,
) raises -> Float64:
    var c = Optional(ctx)
    l.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), c)
    ctx.synchronize()
    var ts = List[Float64]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        l.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), c)
        ctx.synchronize()
        ts.append(Float64(perf_counter_ns() - t0) / 1e3)
    sort(ts)
    return ts[len(ts) // 2]


def row[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises:
    var c = Optional(ctx)
    var l = Linear[IN, OUT].make["gpu", Deterministic](c)
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.01) * Scalar[DT]((i * 7 % 37) - 18)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](0.003) * Scalar[DT]((i * 5 % 23) - 11)
    x.upload(ctx)
    go.upload(ctx)
    var y = Tensor.alloc_gpu(ctx, B * OUT)
    var gi = Tensor.alloc_gpu(ctx, B * IN)
    l.forward["gpu", B](TensorRefs[1](x), y, c)
    var t_train = median_vjp[IN, OUT, B](l, x, go, gi, ctx)
    l.freeze()
    var t_frozen = median_vjp[IN, OUT, B](l, x, go, gi, ctx)
    l.unfreeze["gpu"](c)
    var t_train2 = median_vjp[IN, OUT, B](l, x, go, gi, ctx)
    print(
        "  Linear[", IN, ", ", OUT, "]  B=", B, "  B%16=", B % 16,
        "   trainable ", t_train, " / ", t_train2, " us   frozen ", t_frozen,
        " us   ratio ", (t_train + t_train2) / 2.0 / t_frozen, sep="",
    )


def main() raises:
    with DeviceContext() as ctx:
        print("Linear.vjp, Metal, median of", REPS)
        row[960, 2560, 140](ctx)
        row[960, 2560, 144](ctx)
        row[2560, 960, 140](ctx)
        row[2560, 960, 144](ctx)
        row[960, 320, 140](ctx)
        row[960, 320, 144](ctx)
