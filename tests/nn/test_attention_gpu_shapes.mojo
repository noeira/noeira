"""ScaledDotProductAttention, GPU against CPU, at the shapes that train.

The GPU path is packs + `bmm_tiled` products (Kᵀ / Vᵀ materialised for the two
·ᵀ products) + one-warp-per-row softmax and softmax-JVP; the CPU path is the
plain reference loops. Random inputs; compared in std units
(max |gpu − cpu| / std(cpu)): the forward output, the softmax weights the
forward caches for the vjp, and the input gradient. Each shape runs TWICE on
the same module, so a kernel reading stale scratch from the first call shows
on the second (`docs/CROSS_ATTENTION_OPTIMIZATION.md` §3).

Pinned to the bmm path (`FLASH=False`): it caches the softmax weights this
test compares, and it is the path dropout and heads wider than 64 take. The
fused path is `test_flash_attention.mojo`.

Shapes: LeWM's ViT-tiny (257 tokens, 3 x 64, bidirectional), a causal GPT-ish
block (64 tokens, 4 x 32), LeWM's predictor (3 tokens, 16 x 64, causal).

    pixi run -e apple mojo run -I . tests/nn/test_attention_gpu_shapes.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.attention import ScaledDotProductAttention


comptime TOL = 1e-4
"""Both sides are float32 in different orders. Measured on an M1 Pro: the
current path 3.7e-6 / 2.3e-5 / 5.6e-6 (out / attn / grad_in) at the ViT
shape; the per-(b,h)-block softmax + MAX `bmm` path it replaced 1.8e-6 /
1.1e-5 / 1.9e-6 — the warp reductions and a contracted `s*scale - max` move
the last bits, as they did for CrossAttention (doc §2.2). A semantic error
(a mask ignored, a transpose wrong) is O(1)."""


def _err(cpu: List[Scalar[DT]], gpu: List[Scalar[DT]]) -> Float64:
    var mean = 0.0
    for i in range(len(cpu)):
        mean += Float64(cpu[i])
    mean /= Float64(len(cpu))
    var v = 0.0
    var w = 0.0
    for i in range(len(cpu)):
        v += (Float64(cpu[i]) - mean) ** 2
        w = max(w, abs(Float64(gpu[i]) - Float64(cpu[i])))
    var sd = sqrt(v / Float64(len(cpu)))
    return w / (sd if sd > 0.0 else 1.0)


def _run[
    target: StaticString, DIM: Int, H: Int, S: Int, CAUSAL: Bool, B: Int
](
    mut m: ScaledDotProductAttention[DIM, H, S, CAUSAL, FLASH=False],
    x: List[Scalar[DT]], g: List[Scalar[DT]], ctx: Optional[DeviceContext],
) raises -> List[List[Scalar[DT]]]:
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
        m.cache.download(ctx.value())
    var out = List[List[Scalar[DT]]]()
    var yo = List[Scalar[DT]]()
    for i in range(B * OUT):
        yo.append(y.data[i])
    var at = List[Scalar[DT]]()
    comptime C = m.CACHE_SIZE
    for b in range(B):
        for k in range(H * S * S):
            at.append(m.cache.data[b * C + m.ATTN_OFF + k])
    var go = List[Scalar[DT]]()
    for i in range(B * IN):
        go.append(gi.data[i])
    out.append(yo^)
    out.append(at^)
    out.append(go^)
    return out^


def _shape[
    DIM: Int, H: Int, S: Int, CAUSAL: Bool, B: Int
](name: String, ctx: DeviceContext) raises -> Int:
    seed(11)
    var x = List[Scalar[DT]]()
    var g = List[Scalar[DT]]()
    for _ in range(B * 3 * S * DIM):
        x.append(Scalar[DT](random_float64(-2, 2)))
    for _ in range(B * S * DIM):
        g.append(Scalar[DT](random_float64(-1, 1)))
    var mc = ScaledDotProductAttention[DIM, H, S, CAUSAL, FLASH=False].make["cpu", Kaiming](None)
    var mg = ScaledDotProductAttention[DIM, H, S, CAUSAL, FLASH=False].make["gpu", Kaiming](Optional(ctx))
    var ref_ = _run["cpu", DIM, H, S, CAUSAL, B](mc, x, g, None)
    var fails = 0
    for rep in range(2):
        var got = _run["gpu", DIM, H, S, CAUSAL, B](mg, x, g, Optional(ctx))
        var e0 = _err(ref_[0], got[0])
        var e1 = _err(ref_[1], got[1])
        var e2 = _err(ref_[2], got[2])
        var bad = e0 > TOL or e1 > TOL or e2 > TOL
        print("  ", name, "run", rep, ": out", e0, " attn", e1, " grad_in", e2, " ✗" if bad else "")
        if bad:
            fails += 1
    return fails


def main() raises:
    print("ScaledDotProductAttention GPU vs CPU (std units, tol", TOL, ")")
    var c = DeviceContext()
    var fails = 0
    fails += _shape[192, 3, 257, False, 2](String("ViT-tiny 257x(3x64)    "), c)
    fails += _shape[128, 4, 64, True, 3](String("causal 64x(4x32)       "), c)
    fails += _shape[1024, 16, 3, True, 4](String("LeWM predictor 3x(16x64)"), c)
    if fails > 0:
        raise Error("FAIL: " + String(fails) + " run(s) off the CPU reference")
    print("PASS")
