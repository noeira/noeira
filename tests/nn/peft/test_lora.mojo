"""LoRA on the CPU: no-op at init, gradcheck, rank padding, merge, frozen.

    pixi run mojo run -I . tests/nn/peft/test_lora.mojo

  [1] at init (B = 0) `LoRALinear`'s forward and grad_input are BIT-identical
      to a plain `Linear` holding the same weights
  [2] gradcheck with a nonzero B: dA, dB and dX against central differences of
      L = Σ g ⊙ y, at R = 1 and R = 5, IN = 13, OUT = 7 (no tile multiple).
      y is affine in each of A, B and x separately, so a central difference is
      the exact derivative at any step: what is left is fp32 rounding
  [3] rank padding: 100 Adam steps (weight decay on) of a padded (R = 3 in 16)
      and an unpadded adapter from the same initialisation, learning the
      frozen base plus a rank-1 update: the pad columns of A and rows of B end
      EXACTLY zero, and the two runs agree
  [4] merge: the merged layer's forward matches the adapter's, a merged
      trainable adapter refuses a vjp, and unmerge ∘ merge restores W
  [5] a frozen adapter forms grad_input only, BIT-identical to a trainable
      one's, and keeps no gradient
"""

from std.math import abs
from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.linear import Linear
from noeira.nn.peft.lora import LoRALinear

comptime IN = 13
comptime OUT = 7
comptime T = 4


def _fill(mut t: Tensor, n: Int, mul: Int, mod: Int, scale: Float64):
    for i in range(n):
        t.data[i] = Scalar[DT](Float64(((i * mul) % mod) - mod // 2) * scale)


def _loss[M: Int](ref y: Tensor, ref g: Tensor) -> Float64:
    var l = Float64(0)
    for i in range(M):
        l += Float64(g.data[i]) * Float64(y.data[i])
    return l


def _rel(num: Float64, ana: Float64) -> Float64:
    return abs(num - ana) / max(1.0, abs(ana))


# ── [1] ─────────────────────────────────────────────────────────────────
def leg_noop() raises:
    comptime R = 5
    var base = Linear[IN, OUT].make["cpu", Deterministic]()
    var lora = LoRALinear[IN, OUT, R].make["cpu", Deterministic]()
    var x = Tensor.alloc(T * IN)
    _fill(x, T * IN, 7, 11, 0.2)
    var go = Tensor.alloc(T * OUT)
    _fill(go, T * OUT, 5, 9, 0.3)
    var yb = Tensor.alloc(T * OUT)
    var yl = Tensor.alloc(T * OUT)
    base.forward["cpu", T](TensorRefs[1](x), yb)
    lora.forward["cpu", T](TensorRefs[1](x), yl)
    var gb = Tensor.alloc(T * IN)
    var gl = Tensor.alloc(T * IN)
    base.vjp["cpu", T](TensorRefs[1](x), go, TensorRefs[1](gb))
    lora.vjp["cpu", T](TensorRefs[1](x), go, TensorRefs[1](gl))
    var bad_y = 0
    for i in range(T * OUT):
        if yb.data[i] != yl.data[i]:
            bad_y += 1
    var bad_g = 0
    for i in range(T * IN):
        if gb.data[i] != gl.data[i]:
            bad_g += 1
    var a_mag = Float64(0)
    for i in range(IN * LoRALinear[IN, OUT, R].Adapter.R_PAD):
        a_mag = max(a_mag, abs(Float64(lora.adapter.lora_A.val.data[i])))
    print("  [1] at init: forward mismatches", bad_y, "of", T * OUT,
          "; grad_input mismatches", bad_g, "of", T * IN, " (max |A|", a_mag, ")")
    assert_true(bad_y == 0 and bad_g == 0, "a fresh adapter is not a no-op")
    assert_true(a_mag > 0.05, "A is zero: [1] would hold for a dead adapter")


# ── [2] ─────────────────────────────────────────────────────────────────
def gradcheck[R: Int](label: String) raises:
    comptime M = LoRALinear[IN, OUT, R]
    comptime RP = M.Adapter.R_PAD
    var m = M.make["cpu", Deterministic]()
    # A nonzero B, so dA (which reads B) is not trivially zero.
    for j in range(R):
        for k in range(OUT):
            m.adapter.lora_B.val.data[j * OUT + k] = Scalar[DT](
                Float64(((j * 5 + k * 3) % 7) - 3) * 0.25
            )
    m.adapter.alpha = 2.5 * Float64(R)  # s = 2.5, so a lost scale shows
    var x = Tensor.alloc(T * IN)
    _fill(x, T * IN, 7, 11, 0.2)
    var g = Tensor.alloc(T * OUT)
    _fill(g, T * OUT, 5, 9, 0.3)
    var y = Tensor.alloc(T * OUT)
    var gx = Tensor.alloc(T * IN)
    m.zero_grad["cpu"](None)
    m.forward["cpu", T](TensorRefs[1](x), y)
    m.vjp["cpu", T](TensorRefs[1](x), g, TensorRefs[1](gx))

    comptime EPS = 0.5
    var worst = Float64(0)
    var mag_a = Float64(0)
    var mag_b = Float64(0)
    var mag_x = Float64(0)
    var checked = 0
    # dA over the logical block, dB over the logical block.
    for i in range(IN):
        for j in range(R):
            var k = i * RP + j
            var keep = m.adapter.lora_A.val.data[k]
            m.adapter.lora_A.val.data[k] = keep + Scalar[DT](EPS)
            m.forward["cpu", T](TensorRefs[1](x), y)
            var lp = _loss[T * OUT](y, g)
            m.adapter.lora_A.val.data[k] = keep - Scalar[DT](EPS)
            m.forward["cpu", T](TensorRefs[1](x), y)
            var lm = _loss[T * OUT](y, g)
            m.adapter.lora_A.val.data[k] = keep
            var ana = Float64(m.adapter.lora_A.grd.data[k])
            worst = max(worst, _rel((lp - lm) / (2 * EPS), ana))
            mag_a = max(mag_a, abs(ana))
            checked += 1
    for j in range(R):
        for o in range(OUT):
            var k = j * OUT + o
            var keep = m.adapter.lora_B.val.data[k]
            m.adapter.lora_B.val.data[k] = keep + Scalar[DT](EPS)
            m.forward["cpu", T](TensorRefs[1](x), y)
            var lp = _loss[T * OUT](y, g)
            m.adapter.lora_B.val.data[k] = keep - Scalar[DT](EPS)
            m.forward["cpu", T](TensorRefs[1](x), y)
            var lm = _loss[T * OUT](y, g)
            m.adapter.lora_B.val.data[k] = keep
            var ana = Float64(m.adapter.lora_B.grd.data[k])
            worst = max(worst, _rel((lp - lm) / (2 * EPS), ana))
            mag_b = max(mag_b, abs(ana))
            checked += 1
    for k in range(T * IN):
        var keep = x.data[k]
        x.data[k] = keep + Scalar[DT](EPS)
        m.forward["cpu", T](TensorRefs[1](x), y)
        var lp = _loss[T * OUT](y, g)
        x.data[k] = keep - Scalar[DT](EPS)
        m.forward["cpu", T](TensorRefs[1](x), y)
        var lm = _loss[T * OUT](y, g)
        x.data[k] = keep
        var ana = Float64(gx.data[k])
        worst = max(worst, _rel((lp - lm) / (2 * EPS), ana))
        mag_x = max(mag_x, abs(ana))
        checked += 1
    # The pad of dA and dB is exactly zero (it is never perturbed above: the
    # forward does not read A's pad while B's pad is zero).
    var pad_nz = 0
    for i in range(IN):
        for j in range(R, RP):
            if m.adapter.lora_A.grd.data[i * RP + j] != 0:
                pad_nz += 1
    for j in range(R, RP):
        for o in range(OUT):
            if m.adapter.lora_B.grd.data[j * OUT + o] != 0:
                pad_nz += 1
    print("  [2]", label, ": R_PAD", RP, ",", checked,
          "derivatives, worst |num - ana| / max(1, |ana|)", worst,
          "; max |dA|", mag_a, "|dB|", mag_b, "|dX|", mag_x,
          "; nonzero pad gradients", pad_nz)
    assert_true(worst < 1e-5, "a LoRA derivative disagrees with its central difference")
    assert_true(mag_a > 0.1 and mag_b > 0.1 and mag_x > 0.1, "a gradient is ~0: [2] is vacuous")
    assert_true(pad_nz == 0, "a pad gradient is nonzero")


# ── [3] ─────────────────────────────────────────────────────────────────
struct Run(Movable):
    var a: List[Scalar[DT]]  # logical A, [IN, R] row-major
    var b: List[Scalar[DT]]  # logical B, [R, OUT]
    var loss0: Float64
    var loss1: Float64
    var pad_nz: Int
    var pad_n: Int

    def __init__(out self):
        self.a = List[Scalar[DT]]()
        self.b = List[Scalar[DT]]()
        self.loss0 = 0
        self.loss1 = 0
        self.pad_nz = 0
        self.pad_n = 0


def train[PAD_R: Int]() raises -> Run:
    comptime R = 3
    comptime M = LoRALinear[IN, OUT, R, PAD_R]
    comptime RP = M.Adapter.R_PAD
    seed(1234)
    var m = M.make["cpu", Deterministic]()
    m.adapter.alpha = 6.0
    var opt = Adam(lr=1e-2, wd=0.1)
    var x = Tensor.alloc(T * IN)
    var tgt = Tensor.alloc(T * OUT)
    var y = Tensor.alloc(T * OUT)
    var g = Tensor.alloc(T * OUT)
    var gx = Tensor.alloc(T * IN)
    var yb = Tensor.alloc(T * OUT)
    var r = Run()
    for step in range(100):
        for i in range(T * IN):
            x.data[i] = Scalar[DT](Float64(((i * 7 + step) % 11) - 5) * 0.2)
        # A task the adapter can solve: the frozen base plus a fixed rank-1
        # update D = u vᵀ, on a fresh batch every step.
        m.base.forward["cpu", T](TensorRefs[1](x), yb)
        for t in range(T):
            var xu = Float64(0)
            for i in range(IN):
                xu += Float64(x.data[t * IN + i]) * (Float64(((i * 3) % 5) - 2) * 0.3)
            for o in range(OUT):
                tgt.data[t * OUT + o] = yb.data[t * OUT + o] + Scalar[DT](
                    xu * (Float64(((o * 2) % 3) - 1) * 0.5)
                )
        m.zero_grad["cpu"](None)
        m.forward["cpu", T](TensorRefs[1](x), y)
        var l = Float64(0)
        for i in range(T * OUT):
            var d = y.data[i] - tgt.data[i]
            g.data[i] = d
            l += 0.5 * Float64(d) * Float64(d)
        if step == 0:
            r.loss0 = l
        r.loss1 = l
        m.vjp["cpu", T](TensorRefs[1](x), g, TensorRefs[1](gx))
        opt.step["cpu"](m)
    for i in range(IN):
        for j in range(RP):
            var v = m.adapter.lora_A.val.data[i * RP + j]
            if j < R:
                r.a.append(v)
            else:
                r.pad_n += 1
                if v != 0:
                    r.pad_nz += 1
    for j in range(RP):
        for o in range(OUT):
            var v = m.adapter.lora_B.val.data[j * OUT + o]
            if j < R:
                r.b.append(v)
            else:
                r.pad_n += 1
                if v != 0:
                    r.pad_nz += 1
    return r^


def leg_padding() raises:
    var p = train[16]()
    var u = train[1]()
    var rel = Float64(0)
    var mag = Float64(0)
    for i in range(len(p.a)):
        rel = max(rel, abs(Float64(p.a[i]) - Float64(u.a[i])) / max(abs(Float64(u.a[i])), 1e-3))
    for i in range(len(p.b)):
        rel = max(rel, abs(Float64(p.b[i]) - Float64(u.b[i])) / max(abs(Float64(u.b[i])), 1e-3))
        mag = max(mag, abs(Float64(u.b[i])))
    print("  [3] 100 Adam steps: loss", p.loss0, "->", p.loss1, " (unpadded", u.loss1,
          "); pad values nonzero", p.pad_nz, "of", p.pad_n,
          "; padded vs unpadded A, B max rel", rel, " (max |B|", mag, ")")
    assert_true(p.pad_n > 0 and u.pad_n == 0, "the padding did not pad")
    assert_true(p.pad_nz == 0, "the rank padding drifted from zero under Adam")
    assert_true(p.loss1 < 0.3 * p.loss0, "the adapter did not learn the rank-1 update: [3] is vacuous")
    assert_true(mag > 0.05, "B stayed ~0: [3] compared two untrained adapters")
    assert_true(rel < 1e-4, "the padded and the unpadded adapter trained differently")


# ── [4] ─────────────────────────────────────────────────────────────────
def leg_merge() raises:
    comptime R = 5
    comptime M = LoRALinear[IN, OUT, R]
    var m = M.make["cpu", Deterministic]()
    for j in range(R):
        for k in range(OUT):
            m.adapter.lora_B.val.data[j * OUT + k] = Scalar[DT](
                Float64(((j * 5 + k * 3) % 7) - 3) * 0.25
            )
    m.adapter.alpha = 2.5 * Float64(R)
    var x = Tensor.alloc(T * IN)
    _fill(x, T * IN, 7, 11, 0.2)
    var ya = Tensor.alloc(T * OUT)
    var ym = Tensor.alloc(T * OUT)
    m.forward["cpu", T](TensorRefs[1](x), ya)
    var w0 = List[Scalar[DT]]()
    for i in range(IN * OUT):
        w0.append(m.base.weight.val.data[i])
    var v0 = m.base.weight.val.version
    m.merge["cpu"]()
    m.forward["cpu", T](TensorRefs[1](x), ym)
    var rel = Float64(0)
    var ymag = Float64(0)
    for i in range(T * OUT):
        rel = max(rel, abs(Float64(ya.data[i]) - Float64(ym.data[i])))
        ymag = max(ymag, abs(Float64(ya.data[i])))
    rel /= ymag
    var refused = False
    var g = Tensor.alloc(T * OUT)
    var gx = Tensor.alloc(T * IN)
    try:
        m.vjp["cpu", T](TensorRefs[1](x), g, TensorRefs[1](gx))
    except:
        refused = True
    var bumped = m.base.weight.val.version > v0
    var wm_mag = Float64(0)
    for i in range(IN * OUT):
        wm_mag = max(wm_mag, abs(Float64(m.base.weight.val.data[i])))
    m.unmerge["cpu"]()
    var w_err = Float64(0)
    var w_mag = Float64(0)
    for i in range(IN * OUT):
        w_err = max(w_err, abs(Float64(m.base.weight.val.data[i]) - Float64(w0[i])))
        w_mag = max(w_mag, abs(Float64(w0[i])))
    print("  [4] merged vs adapter forward: max |diff| / max |y|", rel,
          "; merged vjp refused", refused, "; weight version bumped", bumped,
          "; unmerge(merge(W)) max |diff|", w_err, " (max |W|", w_mag, ")")
    assert_true(rel < 1e-6, "the merged forward differs from the adapter's")
    assert_true(refused, "a merged trainable adapter ran a vjp that trains nothing")
    assert_true(bumped, "merge did not bump the weight version (stale w_pad / w_bf)")
    # (W + d) - d: one rounding of W + d, one of the difference.
    assert_true(w_err <= 2 * 1.2e-7 * max(w_mag, wm_mag), "unmerge did not restore W")


# ── [5] ─────────────────────────────────────────────────────────────────
def leg_frozen() raises:
    comptime R = 5
    comptime M = LoRALinear[IN, OUT, R]
    seed(99)
    var t = M.make["cpu", Deterministic]()
    seed(99)
    var f = M.make["cpu", Deterministic]()
    for j in range(R):
        for k in range(OUT):
            var v = Scalar[DT](Float64(((j * 5 + k * 3) % 7) - 3) * 0.25)
            t.adapter.lora_B.val.data[j * OUT + k] = v
            f.adapter.lora_B.val.data[j * OUT + k] = v
    f.adapter.freeze()
    var x = Tensor.alloc(T * IN)
    _fill(x, T * IN, 7, 11, 0.2)
    var g = Tensor.alloc(T * OUT)
    _fill(g, T * OUT, 5, 9, 0.3)
    var y = Tensor.alloc(T * OUT)
    var gt = Tensor.alloc(T * IN)
    var gf = Tensor.alloc(T * IN)
    t.forward["cpu", T](TensorRefs[1](x), y)
    f.forward["cpu", T](TensorRefs[1](x), y)
    t.vjp["cpu", T](TensorRefs[1](x), g, TensorRefs[1](gt))
    f.vjp["cpu", T](TensorRefs[1](x), g, TensorRefs[1](gf))
    var bad = 0
    var diff_base = 0
    var base = Linear[IN, OUT].make["cpu", Deterministic]()
    var gb = Tensor.alloc(T * IN)
    base.vjp["cpu", T](TensorRefs[1](x), g, TensorRefs[1](gb))
    for i in range(T * IN):
        if gt.data[i] != gf.data[i]:
            bad += 1
        if gf.data[i] != gb.data[i]:
            diff_base += 1
    print("  [5] frozen vs trainable adapter grad_input mismatches", bad,
          "; differs from the base alone at", diff_base, "of", T * IN,
          "; frozen gradients empty", f.adapter.is_frozen())
    assert_true(bad == 0, "freezing the adapter changed grad_input")
    assert_true(diff_base > 0, "the adapter adds nothing to grad_input: [5] is vacuous")
    assert_true(f.adapter.is_frozen(), "the frozen adapter kept a gradient")


def main() raises:
    print("LoRA (CPU)")
    leg_noop()
    gradcheck[1]("R = 1")
    gradcheck[5]("R = 5")
    leg_padding()
    leg_merge()
    leg_frozen()
    print("PASSED")
