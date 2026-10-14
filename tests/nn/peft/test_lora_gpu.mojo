"""LoRA on the GPU: no-op at init, against the CPU, rank padding under the
optimizers, merge, frozen.

    pixi run -e apple  mojo run -I . tests/nn/peft/test_lora_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/peft/test_lora_gpu.mojo

The CPU twin (`test_lora.mojo`) gates the adapter's arithmetic against central
differences; this one gates the GPU path against the CPU path.

  [1] at init (B = 0), `LoRALinear`'s forward and grad_input are BIT-identical
      to a plain `Linear` on the GPU
  [2] with a nonzero B: forward, grad_input, dA and dB against the CPU path,
      at a toy shape (IN 13, OUT 7, R 5, T 4) and at a wider one where no dim
      is a tile multiple (IN 96, OUT 40, R 8 padded to 16, T 37), within fp32
      rounding (2e-6 of the largest value). A trainable `Linear`'s own
      GPU-vs-CPU error at the same shape is printed beside it: 0 at these
      sizes, where the adapter's ~1e-7 is its separate scale and add
  [3] rank padding under the optimizers: 60 steps of per-parameter Adam and of
      the arena (`adopt`, the path SmolVLA trains on), weight decay on: the pad
      columns of A and rows of B end EXACTLY zero, and the loss falls
  [4] merge: the merged forward matches the adapter's. IN = 13 puts the base
      `Linear` on Metal's K-padded path, whose padded weight copy is gated on
      the weight's version: a merge that did not bump it would serve the old
      weight. unmerge ∘ merge restores W
  [5] a frozen adapter's grad_input is BIT-identical to a trainable one's
"""

from std.math import abs
from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.linear import Linear
from noeira.nn.peft.lora import LoRALinear


def _fill(mut t: Tensor, n: Int, mul: Int, mod: Int, scale: Float64):
    for i in range(n):
        t.data[i] = Scalar[DT](Float64(((i * mul) % mod) - mod // 2) * scale)


def _host(mut t: Tensor, n: Int, c: DeviceContext) raises -> List[Scalar[DT]]:
    t.download(c)
    var out = List[Scalar[DT]](capacity=n)
    for i in range(n):
        out.append(t.data[i])
    return out^


def _cpu(ref t: Tensor, n: Int) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]](capacity=n)
    for i in range(n):
        out.append(t.data[i])
    return out^


def _err(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) -> Float64:
    """max |a - b| / max |b|."""
    var d = Float64(0)
    var m = Float64(0)
    for i in range(len(b)):
        d = max(d, abs(Float64(a[i]) - Float64(b[i])))
        m = max(m, abs(Float64(b[i])))
    return d / max(m, 1e-30)


def _bits(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) -> Int:
    var n = 0
    for i in range(len(b)):
        if a[i] != b[i]:
            n += 1
    return n


def _set_b[IN: Int, OUT: Int, R: Int](mut m: LoRALinear[IN, OUT, R], ctx: Optional[DeviceContext]) raises:
    """A nonzero logical B (pad rows stay zero)."""
    for j in range(R):
        for k in range(OUT):
            m.adapter.lora_B.val.data[j * OUT + k] = Scalar[DT](
                Float64(((j * 5 + k * 3) % 7) - 3) * 0.25
            )
    if ctx:
        m.adapter.lora_B.val.upload_resident(ctx.value())


# ── [1] ─────────────────────────────────────────────────────────────────
def leg_noop(c: DeviceContext) raises:
    comptime IN = 13
    comptime OUT = 7
    comptime T = 4
    var ctx = Optional(c)
    var base = Linear[IN, OUT].make["gpu", Deterministic](ctx)
    var lora = LoRALinear[IN, OUT, 5].make["gpu", Deterministic](ctx)
    var x = Tensor.alloc(T * IN)
    _fill(x, T * IN, 7, 11, 0.2)
    x.upload(c)
    var go = Tensor.alloc(T * OUT)
    _fill(go, T * OUT, 5, 9, 0.3)
    go.upload(c)
    var yb = Tensor.alloc_gpu(c, T * OUT)
    var yl = Tensor.alloc_gpu(c, T * OUT)
    var gb = Tensor.alloc_gpu(c, T * IN)
    var gl = Tensor.alloc_gpu(c, T * IN)
    base.forward["gpu", T](TensorRefs[1](x), yb, ctx)
    lora.forward["gpu", T](TensorRefs[1](x), yl, ctx)
    base.vjp["gpu", T](TensorRefs[1](x), go, TensorRefs[1](gb), ctx)
    lora.vjp["gpu", T](TensorRefs[1](x), go, TensorRefs[1](gl), ctx)
    var by = _bits(_host(yl, T * OUT, c), _host(yb, T * OUT, c))
    var bg = _bits(_host(gl, T * IN, c), _host(gb, T * IN, c))
    print("  [1] at init: forward mismatches", by, "of", T * OUT,
          "; grad_input mismatches", bg, "of", T * IN)
    assert_true(by == 0 and bg == 0, "a fresh adapter is not a no-op on the GPU")


# ── [2] ─────────────────────────────────────────────────────────────────
def leg_vs_cpu[IN: Int, OUT: Int, R: Int, T: Int](c: DeviceContext, label: String) raises:
    comptime M = LoRALinear[IN, OUT, R]
    comptime RP = M.Adapter.R_PAD
    var ctx = Optional(c)
    seed(7)
    var mc = M.make["cpu", Deterministic]()
    seed(7)
    var mg = M.make["gpu", Deterministic](ctx)
    _set_b(mc, None)
    _set_b(mg, ctx)
    mc.adapter.alpha = 2.5 * Float64(R)
    mg.adapter.alpha = 2.5 * Float64(R)
    var x = Tensor.alloc(T * IN)
    _fill(x, T * IN, 7, 11, 0.2)
    var go = Tensor.alloc(T * OUT)
    _fill(go, T * OUT, 5, 9, 0.3)
    var yc = Tensor.alloc(T * OUT)
    var gc = Tensor.alloc(T * IN)
    mc.forward["cpu", T](TensorRefs[1](x), yc)
    mc.vjp["cpu", T](TensorRefs[1](x), go, TensorRefs[1](gc))
    x.upload(c)
    go.upload(c)
    var yg = Tensor.alloc_gpu(c, T * OUT)
    var gg = Tensor.alloc_gpu(c, T * IN)
    mg.forward["gpu", T](TensorRefs[1](x), yg, ctx)
    mg.vjp["gpu", T](TensorRefs[1](x), go, TensorRefs[1](gg), ctx)
    var e_y = _err(_host(yg, T * OUT, c), _cpu(yc, T * OUT))
    var e_g = _err(_host(gg, T * IN, c), _cpu(gc, T * IN))
    var e_a = _err(
        _host(mg.adapter.lora_A.grd, IN * RP, c), _cpu(mc.adapter.lora_A.grd, IN * RP)
    )
    var e_b = _err(
        _host(mg.adapter.lora_B.grd, RP * OUT, c), _cpu(mc.adapter.lora_B.grd, RP * OUT)
    )
    # The same comparison for a trainable Linear at this shape.
    var lc = Linear[IN, OUT].make["cpu", Deterministic]()
    var lg = Linear[IN, OUT].make["gpu", Deterministic](ctx)
    var ylc = Tensor.alloc(T * OUT)
    var glc = Tensor.alloc(T * IN)
    lc.forward["cpu", T](TensorRefs[1](x), ylc)
    lc.vjp["cpu", T](TensorRefs[1](x), go, TensorRefs[1](glc))
    var ylg = Tensor.alloc_gpu(c, T * OUT)
    var glg = Tensor.alloc_gpu(c, T * IN)
    lg.forward["gpu", T](TensorRefs[1](x), ylg, ctx)
    lg.vjp["gpu", T](TensorRefs[1](x), go, TensorRefs[1](glg), ctx)
    var l_y = _err(_host(ylg, T * OUT, c), _cpu(ylc, T * OUT))
    var l_g = _err(_host(glg, T * IN, c), _cpu(glc, T * IN))
    var l_w = _err(
        _host(lg.weight.grd, IN * OUT, c), _cpu(lc.weight.grd, IN * OUT)
    )
    var worst = max(max(e_y, e_g), max(e_a, e_b))
    var base_worst = max(max(l_y, l_g), l_w)
    print("  [2]", label, ": GPU vs CPU, max |diff| / max |cpu|:  y", e_y, " gx", e_g,
          " dA", e_a, " dB", e_b, "  | trainable Linear: y", l_y, " gx", l_g, " dW", l_w)
    assert_true(
        worst <= max(4.0 * base_worst, 2e-6),
        "the GPU adapter differs from the CPU path beyond fp32 rounding",
    )


# ── [3] ─────────────────────────────────────────────────────────────────
def leg_padding[ADOPT: Bool](c: DeviceContext, label: String) raises:
    comptime IN = 13
    comptime OUT = 7
    comptime T = 4
    comptime R = 3
    comptime M = LoRALinear[IN, OUT, R]
    comptime RP = M.Adapter.R_PAD
    var ctx = Optional(c)
    seed(1234)
    var m = M.make["gpu", Deterministic](ctx)
    m.adapter.alpha = 6.0
    var opt = Adam(lr=1e-2, wd=0.1)
    comptime if ADOPT:
        opt.adopt["gpu"](m, ctx)
    var x = Tensor.alloc(T * IN)
    var tgt = Tensor.alloc(T * OUT)
    var y = Tensor.alloc_gpu(c, T * OUT)
    var yb = Tensor.alloc_gpu(c, T * OUT)
    var g = Tensor.alloc(T * OUT)
    var gx = Tensor.alloc_gpu(c, T * IN)
    var l0 = Float64(0)
    var l1 = Float64(0)
    for step in range(60):
        for i in range(T * IN):
            x.data[i] = Scalar[DT](Float64(((i * 7 + step) % 11) - 5) * 0.2)
        x.upload(c)
        m.base.forward["gpu", T](TensorRefs[1](x), yb, ctx)
        yb.download(c)
        for t in range(T):
            var xu = Float64(0)
            for i in range(IN):
                xu += Float64(x.data[t * IN + i]) * (Float64(((i * 3) % 5) - 2) * 0.3)
            for o in range(OUT):
                tgt.data[t * OUT + o] = yb.data[t * OUT + o] + Scalar[DT](
                    xu * (Float64(((o * 2) % 3) - 1) * 0.5)
                )
        opt.zero_grad["gpu"](m, ctx)
        m.forward["gpu", T](TensorRefs[1](x), y, ctx)
        y.download(c)
        var l = Float64(0)
        for i in range(T * OUT):
            var d = y.data[i] - tgt.data[i]
            g.data[i] = d
            l += 0.5 * Float64(d) * Float64(d)
        if step == 0:
            l0 = l
        l1 = l
        g.upload(c)
        m.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), ctx)
        opt.step["gpu"](m, ctx)
    var a = _host(m.adapter.lora_A.val, IN * RP, c)
    var b = _host(m.adapter.lora_B.val, RP * OUT, c)
    var pad_nz = 0
    var pad_n = 0
    for i in range(IN):
        for j in range(R, RP):
            pad_n += 1
            if a[i * RP + j] != 0:
                pad_nz += 1
    for j in range(R, RP):
        for o in range(OUT):
            pad_n += 1
            if b[j * OUT + o] != 0:
                pad_nz += 1
    print("  [3]", label, ": 60 steps, loss", l0, "->", l1, "; pad values nonzero",
          pad_nz, "of", pad_n)
    assert_true(pad_nz == 0, "the rank padding drifted from zero under the optimizer")
    assert_true(l1 < 0.3 * l0, "the adapter did not learn: [3] is vacuous")


# ── [4] ─────────────────────────────────────────────────────────────────
def leg_merge(c: DeviceContext) raises:
    comptime IN = 13
    comptime OUT = 7
    comptime T = 4
    comptime R = 5
    comptime M = LoRALinear[IN, OUT, R]
    var ctx = Optional(c)
    var m = M.make["gpu", Deterministic](ctx)
    _set_b(m, ctx)
    m.adapter.alpha = 2.5 * Float64(R)
    var x = Tensor.alloc(T * IN)
    _fill(x, T * IN, 7, 11, 0.2)
    x.upload(c)
    var ya = Tensor.alloc_gpu(c, T * OUT)
    var ym = Tensor.alloc_gpu(c, T * OUT)
    m.forward["gpu", T](TensorRefs[1](x), ya, ctx)
    var w0 = _host(m.base.weight.val, IN * OUT, c)
    m.merge["gpu"](ctx)
    var wm = _host(m.base.weight.val, IN * OUT, c)
    m.forward["gpu", T](TensorRefs[1](x), ym, ctx)
    var e = _err(_host(ym, T * OUT, c), _host(ya, T * OUT, c))
    m.unmerge["gpu"](ctx)
    var w1 = _host(m.base.weight.val, IN * OUT, c)
    var w_err = Float64(0)
    var w_mag = Float64(0)
    for i in range(IN * OUT):
        w_err = max(w_err, abs(Float64(w1[i]) - Float64(w0[i])))
        w_mag = max(w_mag, max(abs(Float64(w0[i])), abs(Float64(wm[i]))))
    var moved = _bits(wm, w0)
    print("  [4] merged vs adapter forward: max |diff| / max |y|", e,
          "; merge moved", moved, "of", IN * OUT, "weights; unmerge(merge(W)) max |diff|", w_err)
    assert_true(moved > 0, "merge changed no weight: [4] is vacuous")
    assert_true(e < 1e-5, "the merged forward differs from the adapter's (a stale padded weight?)")
    assert_true(w_err <= 2 * 1.2e-7 * w_mag, "unmerge did not restore W")


# ── [5] ─────────────────────────────────────────────────────────────────
def leg_frozen(c: DeviceContext) raises:
    comptime IN = 13
    comptime OUT = 7
    comptime T = 4
    comptime M = LoRALinear[IN, OUT, 5]
    var ctx = Optional(c)
    seed(99)
    var t = M.make["gpu", Deterministic](ctx)
    seed(99)
    var f = M.make["gpu", Deterministic](ctx)
    _set_b(t, ctx)
    _set_b(f, ctx)
    f.adapter.freeze()
    var x = Tensor.alloc(T * IN)
    _fill(x, T * IN, 7, 11, 0.2)
    x.upload(c)
    var go = Tensor.alloc(T * OUT)
    _fill(go, T * OUT, 5, 9, 0.3)
    go.upload(c)
    var y = Tensor.alloc_gpu(c, T * OUT)
    var gt = Tensor.alloc_gpu(c, T * IN)
    var gf = Tensor.alloc_gpu(c, T * IN)
    t.forward["gpu", T](TensorRefs[1](x), y, ctx)
    f.forward["gpu", T](TensorRefs[1](x), y, ctx)
    t.vjp["gpu", T](TensorRefs[1](x), go, TensorRefs[1](gt), ctx)
    f.vjp["gpu", T](TensorRefs[1](x), go, TensorRefs[1](gf), ctx)
    var bad = _bits(_host(gf, T * IN, c), _host(gt, T * IN, c))
    print("  [5] frozen vs trainable adapter grad_input mismatches", bad,
          "; frozen gradients empty", f.adapter.is_frozen())
    assert_true(bad == 0, "freezing the adapter changed grad_input")
    assert_true(f.adapter.is_frozen(), "the frozen adapter kept a gradient")


def main() raises:
    print("LoRA (GPU)")
    var c = DeviceContext()
    leg_noop(c)
    leg_vs_cpu[13, 7, 5, 4](c, "IN 13 OUT 7 R 5 T 4")
    leg_vs_cpu[96, 40, 8, 37](c, "IN 96 OUT 40 R 8 T 37")
    leg_padding[False](c, "per-parameter Adam")
    leg_padding[True](c, "arena Adam")
    leg_merge(c)
    leg_frozen(c)
    print("PASSED")
