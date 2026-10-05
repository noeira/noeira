"""ScaledDotProductAttentionQKV — attention on the QKV projection's own
token-major layout — on the GPU against its CPU path.

On the CPU the module is `QKVToMajor` + `ScaledDotProductAttention`'s plain
reference loops; on the GPU, where the fused kernels apply, it reads and
writes the token-major layout directly (`IL=True`), and elsewhere it permutes
and runs the bmm path. Compared in std units on the output and the whole
input gradient, each shape twice (the second run bit-identical):

  - fused: GPT 256x(6x64) causal, CIFAR ViT 64x(6x32), ViT-tiny 257x(3x64),
    causal 50x(4x32);
  - fallback (head dim 8, outside the fused domain): 8x(2x8) causal.

Non-vacuity: the major-layout leaf fed the SAME token-major buffer must
disagree (the layout is actually being read differently).

    pixi run -e apple mojo run -I . tests/nn/test_attention_qkv_layout.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.attention import (
    ScaledDotProductAttention, ScaledDotProductAttentionQKV,
)


comptime TOL = 1e-4


def _err(ref_: List[Scalar[DT]], got: List[Scalar[DT]]) -> Float64:
    var mean = 0.0
    for i in range(len(ref_)):
        mean += Float64(ref_[i])
    mean /= Float64(len(ref_))
    var v = 0.0
    var w = 0.0
    for i in range(len(ref_)):
        v += (Float64(ref_[i]) - mean) ** 2
        w = max(w, abs(Float64(got[i]) - Float64(ref_[i])))
    var sd = sqrt(v / Float64(len(ref_)))
    return w / (sd if sd > 0.0 else 1.0)


def _inputs(n_in: Int, n_out: Int) -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    seed(17)
    var x = List[Scalar[DT]]()
    var g = List[Scalar[DT]]()
    for _ in range(n_in):
        x.append(Scalar[DT](random_float64(-2, 2)))
    for _ in range(n_out):
        g.append(Scalar[DT](random_float64(-1, 1)))
    return (x^, g^)


def _run[
    target: StaticString, DIM: Int, H: Int, S: Int, CAUSAL: Bool, B: Int
](
    mut m: ScaledDotProductAttentionQKV[DIM, H, S, CAUSAL],
    x: List[Scalar[DT]], g: List[Scalar[DT]], ctx: Optional[DeviceContext],
) raises -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    comptime IN = 3 * S * DIM
    comptime OUT = S * DIM
    var xt = Tensor.alloc(B * IN)
    var gt = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        xt.data[i] = x[i]
    for i in range(B * OUT):
        gt.data[i] = g[i]
    comptime if target == "gpu":
        xt.upload(ctx.value())
        gt.upload(ctx.value())
    var y = Tensor()
    var gi = Tensor()
    m.forward[target, B](TensorRefs[1](xt), y, ctx)
    m.vjp[target, B](TensorRefs[1](xt), gt, TensorRefs[1](gi), ctx)
    comptime if target == "gpu":
        ctx.value().synchronize()
        y.download(ctx.value())
        gi.download(ctx.value())
    var yo = List[Scalar[DT]]()
    for i in range(B * OUT):
        yo.append(y.data[i])
    var go = List[Scalar[DT]]()
    for i in range(B * IN):
        go.append(gi.data[i])
    return (yo^, go^)


def _shape[
    DIM: Int, H: Int, S: Int, CAUSAL: Bool, B: Int, FUSED: Bool
](name: String, ctx: DeviceContext) raises -> Int:
    comptime M = ScaledDotProductAttentionQKV[DIM, H, S, CAUSAL]
    comptime assert M.ATTN.USE_FLASH == FUSED, "the shape takes the other path"
    var xg = _inputs(B * 3 * S * DIM, B * S * DIM)
    var mc = M.make["cpu", Kaiming](None)
    var mg = M.make["gpu", Kaiming](Optional(ctx))
    var ref_ = _run["cpu", DIM, H, S, CAUSAL, B](mc, xg[0], xg[1], None)
    var fails = 0
    var first = List[Scalar[DT]]()
    for rep in range(2):
        var got = _run["gpu", DIM, H, S, CAUSAL, B](mg, xg[0], xg[1], Optional(ctx))
        var e0 = _err(ref_[0], got[0])
        var e1 = _err(ref_[1], got[1])
        var bad = e0 > TOL or e1 > TOL
        if rep == 0:
            first = got[1].copy()
        else:
            for k in range(len(first)):
                if first[k] != got[1][k]:
                    bad = True
                    break
        print("  ", name, "run", rep, ": out", e0, " grad_in", e1, " ✗" if bad else "")
        if bad:
            fails += 1
    return fails


def _major_leaf_disagrees(ctx: DeviceContext) raises -> Float64:
    """The token-major buffer read as [Q | K | V] by the plain leaf."""
    comptime DIM = 192
    comptime S = 64
    comptime B = 2
    var xg = _inputs(B * 3 * S * DIM, B * S * DIM)
    var mq = ScaledDotProductAttentionQKV[DIM, 6, S, False].make["cpu", Kaiming](None)
    var ref_ = _run["cpu", DIM, 6, S, False, B](mq, xg[0], xg[1], None)
    var ml = ScaledDotProductAttention[DIM, 6, S, False].make["cpu", Kaiming](None)
    var xt = Tensor.alloc(B * 3 * S * DIM)
    for i in range(B * 3 * S * DIM):
        xt.data[i] = xg[0][i]
    var y = Tensor()
    ml.forward["cpu", B](TensorRefs[1](xt), y, None)
    var yo = List[Scalar[DT]]()
    for i in range(B * S * DIM):
        yo.append(y.data[i])
    return _err(ref_[0], yo)


def main() raises:
    print("ScaledDotProductAttentionQKV GPU vs CPU (std units, tol", TOL, ")")
    var c = DeviceContext()
    var fails = 0
    fails += _shape[384, 6, 256, True, 2, True](String("GPT 256x(6x64) causal "), c)
    fails += _shape[192, 6, 64, False, 3, True](String("CIFAR ViT 64x(6x32)   "), c)
    fails += _shape[192, 3, 257, False, 2, True](String("ViT-tiny 257x(3x64)   "), c)
    fails += _shape[128, 4, 50, True, 3, True](String("causal 50x(4x32)      "), c)
    fails += _shape[16, 2, 8, True, 4, False](String("fallback 8x(2x8)      "), c)
    var e = _major_leaf_disagrees(c)
    print("   vacuity: the major-layout leaf on the token-major buffer, out", e)
    assert_true(e > 100 * TOL, "the layouts are indistinguishable at this input")
    if fails > 0:
        raise Error("FAIL: " + String(fails) + " run(s) off the CPU path")
    print("PASS")
