"""ScaledDotProductAttention's fused (flash) GPU path against the CPU reference.

The fused path (`flash_attention.mojo`) never stores the scores: forward with
an online softmax, backward recomputing P from the per-row log-sum-exp in two
deterministic kernels. Compared with the CPU path (plain loops + BLAS) in std
units (max |gpu − cpu| / std(cpu)) on the forward output and the whole input
gradient (dQ | dK | dV), at the shapes that train:

  - LeWM's ViT-tiny, 257 tokens (a partial last tile), 3 x 64, bidirectional;
  - the CIFAR ViT, 64 tokens, 6 x 32;
  - the TinyShakespeare GPT, 256 tokens, 6 x 64, causal;
  - a causal block whose length is not a multiple of the tile, 50 x (4 x 32);
  - LeWM's predictor, 3 tokens (one partial tile), 16 x 64, causal.

Each shape runs TWICE on the same module (stale scratch shows on the second),
and the second run must be bit-identical to the first (no atomics: the fused
backward is deterministic). Non-vacuity: the compiled path is the fused one
(`USE_FLASH`), and a causal shape fed to a bidirectional module must fail.

    pixi run -e apple mojo run -I . tests/nn/test_flash_attention.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.attention import ScaledDotProductAttention


comptime TOL = 1e-4
"""Both sides float32 in different orders; a semantic error (a mask, a tile
edge, a transpose) is O(1)."""


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


def _run[
    target: StaticString, DIM: Int, H: Int, S: Int, CAUSAL: Bool, FLASH: Bool,
    B: Int,
](
    mut m: ScaledDotProductAttention[DIM, H, S, CAUSAL, FLASH=FLASH],
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


def _inputs(n_in: Int, n_out: Int) -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    seed(11)
    var x = List[Scalar[DT]]()
    var g = List[Scalar[DT]]()
    for _ in range(n_in):
        x.append(Scalar[DT](random_float64(-2, 2)))
    for _ in range(n_out):
        g.append(Scalar[DT](random_float64(-1, 1)))
    return (x^, g^)


def _shape[
    DIM: Int, H: Int, S: Int, CAUSAL: Bool, B: Int
](name: String, ctx: DeviceContext) raises -> Int:
    comptime assert ScaledDotProductAttention[DIM, H, S, CAUSAL].USE_FLASH, (
        "the shape does not take the fused path"
    )
    var xg = _inputs(B * 3 * S * DIM, B * S * DIM)
    var mc = ScaledDotProductAttention[DIM, H, S, CAUSAL, FLASH=True].make[
        "cpu", Kaiming
    ](None)
    var mg = ScaledDotProductAttention[DIM, H, S, CAUSAL, FLASH=True].make[
        "gpu", Kaiming
    ](Optional(ctx))
    var ref_ = _run["cpu", DIM, H, S, CAUSAL, True, B](mc, xg[0], xg[1], None)
    var fails = 0
    var first = List[Scalar[DT]]()
    for rep in range(2):
        var got = _run["gpu", DIM, H, S, CAUSAL, True, B](
            mg, xg[0], xg[1], Optional(ctx)
        )
        var e0 = _err(ref_[0], got[0])
        var e1 = _err(ref_[1], got[1])
        var bad = e0 > TOL or e1 > TOL
        if rep == 0:
            first = got[1].copy()
        else:
            for k in range(len(first)):
                if first[k] != got[1][k]:
                    print("   run 1 differs from run 0 at", k)
                    bad = True
                    break
        print("  ", name, "run", rep, ": out", e0, " grad_in", e1, " ✗" if bad else "")
        if bad:
            fails += 1
    return fails


def _causal_fed_bidirectional(ctx: DeviceContext) raises -> Float64:
    """The causal 50-token reference against a BIDIRECTIONAL fused module."""
    comptime DIM = 128
    comptime S = 50
    comptime B = 2
    var xg = _inputs(B * 3 * S * DIM, B * S * DIM)
    var mc = ScaledDotProductAttention[DIM, 4, S, True].make["cpu", Kaiming](None)
    var mg = ScaledDotProductAttention[DIM, 4, S, False].make["gpu", Kaiming](
        Optional(ctx)
    )
    var ref_ = _run["cpu", DIM, 4, S, True, True, B](mc, xg[0], xg[1], None)
    var got = _run["gpu", DIM, 4, S, False, True, B](mg, xg[0], xg[1], Optional(ctx))
    return _err(ref_[0], got[0])


def main() raises:
    print("ScaledDotProductAttention fused GPU vs CPU (std units, tol", TOL, ")")
    var c = DeviceContext()
    var fails = 0
    fails += _shape[192, 3, 257, False, 2](String("ViT-tiny 257x(3x64)     "), c)
    fails += _shape[192, 6, 64, False, 3](String("CIFAR ViT 64x(6x32)     "), c)
    fails += _shape[384, 6, 256, True, 2](String("GPT 256x(6x64) causal   "), c)
    fails += _shape[128, 4, 50, True, 3](String("causal 50x(4x32)        "), c)
    fails += _shape[1024, 16, 3, True, 4](String("LeWM predictor 3x(16x64)"), c)
    var e = _causal_fed_bidirectional(c)
    print("   vacuity: causal reference vs bidirectional module, out", e)
    assert_true(e > 100 * TOL, "the gate cannot see the causal mask")
    if fails > 0:
        raise Error("FAIL: " + String(fails) + " run(s) off the CPU reference")
    print("PASS")
