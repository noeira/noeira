"""The bf16 attention on the tensor cores (`flash_attention_mma.mojo`)
against the fp32 CPU reference — NVIDIA.

bf16 activations: the GPU module flows at bf16, the CPU reference is the fp32
module fed the same bf16-rounded inputs. Compared in std units (max |gpu −
cpu| / std(cpu)) on the forward output and the whole input gradient (dQ | dK |
dV), for both input layouts — `ScaledDotProductAttention` ([Q | K | V]) and
`ScaledDotProductAttentionQKV` (the QKV projection's token-major output) — at
the shapes that train:

  - the TinyShakespeare GPT, 256 tokens, 6 x 64, causal (QKV layout);
  - LeWM's ViT-tiny, 257 tokens (a partial last tile), 3 x 64, bidirectional;
  - the CIFAR ViT, 64 tokens, 6 x 32 (head dim 32);
  - a causal block whose length is not a multiple of the tile, 50 x (4 x 32);
  - LeWM's predictor, 3 tokens (one partial tile), 16 x 64, causal.

The bands: `TOL` on the max error, `TOL_RMS` on the RMS error. The test
prints the floor — the exact answer rounded to bf16, since every output is
stored bf16 — and the fp32 CUDA-core kernels (`simt`) sit EXACTLY on it (max
0.004-0.05 std units, RMS ~0.0016). The tensor-core path adds the bf16
rounding of P and dS as MMA operands (as PyTorch's FA2 does): measured max
<= 1.75x the floor, RMS 0.0019-0.0024 (RTX 5090). `TOL_RMS` is ~2x the
floor's RMS; a semantic error (a mask, a tile edge, a transpose) is O(1). Each shape runs TWICE on the
same module: the second run must be bit-identical (no atomics). Non-vacuity:
a causal reference against a bidirectional module must fail by > 10 x TOL.
`-D NN_ATTN_PATH=simt` runs the same gate on the fp32 CUDA-core kernels (the
path before) for comparison.

    pixi run -e nvidia mojo run -I . tests/nn/test_flash_attention_mma_gpu.mojo
"""

from std.math import sqrt, abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.attention import (
    ScaledDotProductAttention, ScaledDotProductAttentionQKV,
)
from noeira.nn.primitives.flash_attention import ATTN_PATH


comptime BF16 = DType.bfloat16
comptime TOL = 1e-1
comptime TOL_RMS = 5e-3


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


def _rms(ref_: List[Scalar[DT]], got: List[Scalar[DT]]) -> Float64:
    """RMS |gpu − cpu| / std(cpu)."""
    var mean = 0.0
    for i in range(len(ref_)):
        mean += Float64(ref_[i])
    mean /= Float64(len(ref_))
    var v = 0.0
    var w = 0.0
    for i in range(len(ref_)):
        v += (Float64(ref_[i]) - mean) ** 2
        w += (Float64(got[i]) - Float64(ref_[i])) ** 2
    return sqrt(w / v) if v > 0.0 else sqrt(w)


def _rounded(x: List[Scalar[DT]]) -> List[Scalar[DT]]:
    var o = List[Scalar[DT]]()
    for v in x:
        o.append(v.cast[BF16]().cast[DT]())
    return o^


def _inputs(n_in: Int, n_out: Int) -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    """bf16-representable values (so both sides see the same inputs)."""
    seed(11)
    var x = List[Scalar[DT]]()
    var g = List[Scalar[DT]]()
    for _ in range(n_in):
        x.append(Scalar[DT](random_float64(-2, 2)).cast[BF16]().cast[DT]())
    for _ in range(n_out):
        g.append(Scalar[DT](random_float64(-1, 1)).cast[BF16]().cast[DT]())
    return (x^, g^)


def _cpu[
    QKV: Bool, DIM: Int, H: Int, S: Int, CAUSAL: Bool, B: Int
](x: List[Scalar[DT]], g: List[Scalar[DT]]) raises -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    comptime IN = 3 * S * DIM
    comptime OUT = S * DIM
    var xt = Tensor.alloc(B * IN)
    var gt = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        xt.data[i] = x[i]
    for i in range(B * OUT):
        gt.data[i] = g[i]
    var y = Tensor()
    var gi = Tensor()
    comptime if QKV:
        var m = ScaledDotProductAttentionQKV[DIM, H, S, CAUSAL].make["cpu", Kaiming](None)
        m.forward["cpu", B](TensorRefs[1](xt), y, None)
        m.vjp["cpu", B](TensorRefs[1](xt), gt, TensorRefs[1](gi), None)
    else:
        var m = ScaledDotProductAttention[DIM, H, S, CAUSAL].make["cpu", Kaiming](None)
        m.forward["cpu", B](TensorRefs[1](xt), y, None)
        m.vjp["cpu", B](TensorRefs[1](xt), gt, TensorRefs[1](gi), None)
    var yo = List[Scalar[DT]]()
    for i in range(B * OUT):
        yo.append(y.data[i])
    var go = List[Scalar[DT]]()
    for i in range(B * IN):
        go.append(gi.data[i])
    return (yo^, go^)


def _gpu_run[
    QKV: Bool, DIM: Int, H: Int, S: Int, CAUSAL: Bool, B: Int
](
    x: List[Scalar[DT]], g: List[Scalar[DT]], ctx: DeviceContext, runs: Int
) raises -> List[Tuple[List[Scalar[DT]], List[Scalar[DT]]]]:
    """`runs` forward + vjp passes on ONE bf16 module."""
    comptime IN = 3 * S * DIM
    comptime OUT = S * DIM
    var xt = TensorImpl[BF16].alloc(B * IN)
    var gt = TensorImpl[BF16].alloc(B * OUT)
    for i in range(B * IN):
        xt.data[i] = x[i].cast[BF16]()
    for i in range(B * OUT):
        gt.data[i] = g[i].cast[BF16]()
    xt.upload(ctx)
    gt.upload(ctx)
    var res = List[Tuple[List[Scalar[DT]], List[Scalar[DT]]]]()
    comptime if QKV:
        var m = ScaledDotProductAttentionQKV[DIM, H, S, CAUSAL, True, BF16].make[
            "gpu", Kaiming
        ](Optional(ctx))
        for _ in range(runs):
            var y = TensorImpl[BF16]()
            var gi = TensorImpl[BF16]()
            m.forward["gpu", B](TensorRefs[1, ADT=BF16](xt), y, Optional(ctx))
            m.vjp["gpu", B](TensorRefs[1, ADT=BF16](xt), gt, TensorRefs[1, ADT=BF16](gi), Optional(ctx))
            res.append(_dl[B * OUT, B * IN](y, gi, ctx))
    else:
        var m = ScaledDotProductAttention[DIM, H, S, CAUSAL, True, BF16].make[
            "gpu", Kaiming
        ](Optional(ctx))
        for _ in range(runs):
            var y = TensorImpl[BF16]()
            var gi = TensorImpl[BF16]()
            m.forward["gpu", B](TensorRefs[1, ADT=BF16](xt), y, Optional(ctx))
            m.vjp["gpu", B](TensorRefs[1, ADT=BF16](xt), gt, TensorRefs[1, ADT=BF16](gi), Optional(ctx))
            res.append(_dl[B * OUT, B * IN](y, gi, ctx))
    return res^


def _dl[
    NO: Int, NI: Int
](mut y: TensorImpl[BF16], mut gi: TensorImpl[BF16], ctx: DeviceContext) raises -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    ctx.synchronize()
    y.download(ctx)
    gi.download(ctx)
    var yo = List[Scalar[DT]]()
    for i in range(NO):
        yo.append(y.data[i].cast[DT]())
    var go = List[Scalar[DT]]()
    for i in range(NI):
        go.append(gi.data[i].cast[DT]())
    return (yo^, go^)


def _shape[
    QKV: Bool, DIM: Int, H: Int, S: Int, CAUSAL: Bool, B: Int
](name: String, ctx: DeviceContext) raises -> Int:
    var xg = _inputs(B * 3 * S * DIM, B * S * DIM)
    var ref_ = _cpu[QKV, DIM, H, S, CAUSAL, B](xg[0], xg[1])
    var got = _gpu_run[QKV, DIM, H, S, CAUSAL, B](xg[0], xg[1], ctx, 2)
    var y0 = got[0][0].copy()
    var g0 = got[0][1].copy()
    var y1 = got[1][0].copy()
    var g1 = got[1][1].copy()
    var e0 = _err(ref_[0], y0)
    var e1 = _err(ref_[1], g0)
    var q0 = _rms(ref_[0], y0)
    var q1 = _rms(ref_[1], g0)
    # The floor: the exact answer rounded to bf16 (outputs are stored bf16).
    var f0 = _err(ref_[0], _rounded(ref_[0]))
    var f1 = _err(ref_[1], _rounded(ref_[1]))
    var bad = e0 > TOL or e1 > TOL or q0 > TOL_RMS or q1 > TOL_RMS
    var same = True
    for k in range(len(g0)):
        if g0[k] != g1[k]:
            same = False
            break
    for k in range(len(y0)):
        if y0[k] != y1[k]:
            same = False
            break
    if not same:
        bad = True
    print(
        "  ", name, " | max out ", e0, " grad_in ", e1, " (bf16-rounding floor ",
        f0, " / ", f1, ") | rms ", q0, " / ", q1,
        " | run 2 bit-identical" if same else " | RUN 2 DIFFERS",
        " ✗" if bad else "", sep="",
    )
    return 1 if bad else 0


def main() raises:
    print(
        "bf16 attention (path", ATTN_PATH, ") vs fp32 CPU on bf16 inputs (std units, tol",
        TOL, ")",
    )
    var c = DeviceContext()
    var fails = 0
    fails += _shape[True, 384, 6, 256, True, 2]("GPT 256x(6x64) causal, QKV layout ", c)
    fails += _shape[False, 384, 6, 256, True, 2]("GPT 256x(6x64) causal, [Q|K|V]    ", c)
    fails += _shape[False, 192, 3, 257, False, 2]("ViT-tiny 257x(3x64)              ", c)
    fails += _shape[True, 192, 6, 64, False, 3]("CIFAR ViT 64x(6x32), QKV layout   ", c)
    fails += _shape[False, 128, 4, 50, True, 3]("causal 50x(4x32)                 ", c)
    fails += _shape[False, 1024, 16, 3, True, 4]("LeWM predictor 3x(16x64)         ", c)
    # Vacuity: the causal reference against a bidirectional module.
    comptime DIM = 128
    comptime S = 50
    var xg = _inputs(2 * 3 * S * DIM, 2 * S * DIM)
    var ref_ = _cpu[False, DIM, 4, S, True, 2](xg[0], xg[1])
    var got = _gpu_run[False, DIM, 4, S, False, 2](xg[0], xg[1], c, 1)
    var yv = got[0][0].copy()
    var ev = _err(ref_[0], yv)
    print("   vacuity: causal reference vs bidirectional module, out", ev)
    assert_true(ev > 10 * TOL, "the gate cannot see the causal mask")
    if fails > 0:
        raise Error("FAIL: " + String(fails) + " shape(s) off the reference")
    print("PASS")
