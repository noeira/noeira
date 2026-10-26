"""More ops for the functional tape: exp / log / sqrt (as `ElementOp`s, so
they ride the activation machinery), clamp, softmax and log-softmax over
the last axis, a fused cross-entropy, and dropout.

Softmax, log-softmax and cross-entropy work on the 2-D view: a row is
everything but the last axis, so `[B, T, C]` logits are `B*T` rows of `C`.
One thread per row (rows are short; this is the simple, deterministic
form, not a tuned reduction).

Dropout takes its `seed` from the caller (JAX's style, not a hidden
counter): a mask is a function of (seed, element index) through one integer
hash that the CPU and the GPU compute alike, so the two draw the SAME mask
and a step is reproducible. Pass a different seed per step to train. Under
CUDA-graph capture the seed is baked into the graph: every replay draws
the captured step's mask.
"""

from std.math import exp, log, sqrt
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.element_op import ElementOp
from .backend import Dev, fill_dev, ensure_gpu
from .opcodes import *
from .tape import TapeData
from .gpu import gpu_reduce


comptime P = Pointer[Scalar[DT], MutAnyOrigin]
comptime TPB = 256


@always_inline
def _grid(n: Int) -> Int:
    return (n + TPB - 1) // TPB


# ── exp / log / sqrt as ElementOps ───────────────────────────────────────
struct ExpOp(ElementOp):
    comptime owns_cache = True  # d exp = exp

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return exp(x)

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        return exp(x)

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        return go * c

    @staticmethod
    def backward_simd[W: Int](c: SIMD[DT, W], go: SIMD[DT, W]) -> SIMD[DT, W]:
        return go * c


struct LogOp(ElementOp):
    comptime owns_cache = False  # d log x = 1 / x

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return log(x)

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        return log(x)

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        return go / c

    @staticmethod
    def backward_simd[W: Int](c: SIMD[DT, W], go: SIMD[DT, W]) -> SIMD[DT, W]:
        return go / c


struct SqrtOp(ElementOp):
    comptime owns_cache = True  # d sqrt x = 1 / (2 sqrt x)

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return sqrt(x)

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        return sqrt(x)

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        return go / (Scalar[DT](2) * c)

    @staticmethod
    def backward_simd[W: Int](c: SIMD[DT, W], go: SIMD[DT, W]) -> SIMD[DT, W]:
        return go / (SIMD[DT, W](2) * c)


# ── clamp ────────────────────────────────────────────────────────────────
def _clamp_kernel(o: P, a: P, n: Int64, lo: Scalar[DT], hi: Scalar[DT]):
    var k = Int(global_idx.x)
    if k < Int(n):
        var x = a[unsafe_offset=k]
        o[unsafe_offset=k] = lo if x < lo else (hi if x > hi else x)


def _clamp_bwd_kernel[ACC: Bool](
    ga: P, go: P, a: P, n: Int64, lo: Scalar[DT], hi: Scalar[DT]
):
    """Gradient 1 inside [lo, hi] (bounds included, as torch), 0 outside."""
    var k = Int(global_idx.x)
    if k >= Int(n):
        return
    var x = a[unsafe_offset=k]
    var g = go[unsafe_offset=k] if (x >= lo and x <= hi) else Scalar[DT](0)
    comptime if ACC:
        ga[unsafe_offset=k] += g
    else:
        ga[unsafe_offset=k] = g


def ex_clamp(mut t: TapeData, a: Int, lo: Scalar[DT], hi: Scalar[DT]) raises -> Int:
    if lo > hi:
        raise Error("clamp: lo > hi")
    var o = t.new_node(t.nodes[a].rows, t.nodes[a].cols, OP_CLAMP, a, -1, t.nodes[a].rg)
    t.like(o, a)
    t.nodes[o].c = lo
    t.nodes[o].c2 = hi
    var n = t.nodes[a].size()
    if t.ctx:
        t.dctx().enqueue_function[_clamp_kernel](
            t.dv(o), t.dv(a), Int64(n), lo, hi, grid_dim=_grid(n), block_dim=TPB
        )
        return o
    var po = t.val_ptr(o)
    var pa = t.val_ptr(a)
    for k in range(n):
        var x = pa[unsafe_offset=k]
        po[unsafe_offset=k] = lo if x < lo else (hi if x > hi else x)
    return o


# ── softmax / log-softmax over the last axis ─────────────────────────────
def _softmax_kernel[LOG: Bool](o: P, a: P, rows: Int64, cols: Int64):
    var C = Int(cols)
    var r = Int(global_idx.x)
    if r >= Int(rows):
        return
    var base = r * C
    var m = a[unsafe_offset=base]
    for j in range(1, C):
        m = max(m, a[unsafe_offset=base + j])
    var s: Scalar[DT] = 0
    for j in range(C):
        s += exp(a[unsafe_offset=base + j] - m)
    comptime if LOG:
        var ls = log(s)
        for j in range(C):
            o[unsafe_offset=base + j] = a[unsafe_offset=base + j] - m - ls
    else:
        for j in range(C):
            o[unsafe_offset=base + j] = exp(a[unsafe_offset=base + j] - m) / s


def _softmax_bwd_kernel[LOG: Bool, ACC: Bool](
    ga: P, go: P, y: P, rows: Int64, cols: Int64
):
    """softmax: dx = y ⊙ (g − Σ g·y); log-softmax: dx = g − softmax · Σ g."""
    var C = Int(cols)
    var r = Int(global_idx.x)
    if r >= Int(rows):
        return
    var base = r * C
    var s: Scalar[DT] = 0
    for j in range(C):
        comptime if LOG:
            s += go[unsafe_offset=base + j]
        else:
            s += go[unsafe_offset=base + j] * y[unsafe_offset=base + j]
    for j in range(C):
        var d: Scalar[DT]
        comptime if LOG:
            d = go[unsafe_offset=base + j] - exp(y[unsafe_offset=base + j]) * s
        else:
            d = y[unsafe_offset=base + j] * (go[unsafe_offset=base + j] - s)
        comptime if ACC:
            ga[unsafe_offset=base + j] += d
        else:
            ga[unsafe_offset=base + j] = d


def ex_softmax(mut t: TapeData, a: Int, log_: Bool) raises -> Int:
    var R = t.nodes[a].rows
    var C = t.nodes[a].cols
    var o = t.new_node(R, C, OP_SOFTMAX, a, -1, t.nodes[a].rg)
    t.like(o, a)
    t.nodes[o].sub = 1 if log_ else 0
    if t.ctx:
        var c = t.dctx()
        if log_:
            c.enqueue_function[_softmax_kernel[True]](t.dv(o), t.dv(a), Int64(R), Int64(C), grid_dim=_grid(R), block_dim=TPB)
        else:
            c.enqueue_function[_softmax_kernel[False]](t.dv(o), t.dv(a), Int64(R), Int64(C), grid_dim=_grid(R), block_dim=TPB)
        return o
    var po = t.val_ptr(o)
    var pa = t.val_ptr(a)
    for r in range(R):
        var base = r * C
        var m = pa[unsafe_offset=base]
        for j in range(1, C):
            m = max(m, pa[unsafe_offset=base + j])
        var s: Scalar[DT] = 0
        for j in range(C):
            s += exp(pa[unsafe_offset=base + j] - m)
        var ls = log(s)
        for j in range(C):
            if log_:
                po[unsafe_offset=base + j] = pa[unsafe_offset=base + j] - m - ls
            else:
                po[unsafe_offset=base + j] = exp(pa[unsafe_offset=base + j] - m) / s
    return o


# ── cross-entropy (fused log-softmax + pick + mean) ──────────────────────
def _xent_rows_kernel(loss: P, z: P, tgt: P, rows: Int64, cols: Int64):
    """Per row: logsumexp(z) − z[target]."""
    var C = Int(cols)
    var r = Int(global_idx.x)
    if r >= Int(rows):
        return
    var base = r * C
    var m = z[unsafe_offset=base]
    for j in range(1, C):
        m = max(m, z[unsafe_offset=base + j])
    var s: Scalar[DT] = 0
    for j in range(C):
        s += exp(z[unsafe_offset=base + j] - m)
    var c = Int(tgt[unsafe_offset=r])
    var zt = z[unsafe_offset=base + c] if (c >= 0 and c < C) else Scalar[DT](0) / Scalar[DT](0)
    loss[unsafe_offset=r] = m + log(s) - zt


def _xent_bwd_kernel[ACC: Bool](
    gz: P, go: P, z: P, tgt: P, rows: Int64, cols: Int64
):
    """d mean CE / dz = (softmax(z) − onehot(target)) · g / rows."""
    var C = Int(cols)
    var R = Int(rows)
    var r = Int(global_idx.x)
    if r >= R:
        return
    var base = r * C
    var m = z[unsafe_offset=base]
    for j in range(1, C):
        m = max(m, z[unsafe_offset=base + j])
    var s: Scalar[DT] = 0
    for j in range(C):
        s += exp(z[unsafe_offset=base + j] - m)
    var g = go[unsafe_offset=0] / Scalar[DT](R)
    var c = Int(tgt[unsafe_offset=r])
    for j in range(C):
        var p = exp(z[unsafe_offset=base + j] - m) / s
        var d = (p - (Scalar[DT](1) if j == c else Scalar[DT](0))) * g
        comptime if ACC:
            gz[unsafe_offset=base + j] += d
        else:
            gz[unsafe_offset=base + j] = d


def ex_cross_entropy(mut t: TapeData, z: Int, tgt: Int) raises -> Int:
    """Mean over rows of −log softmax(z)[target]: z's rows (every axis but
    the last) against `tgt`'s values (class indices held as floats)."""
    var R = t.nodes[z].rows
    var C = t.nodes[z].cols
    if t.nodes[tgt].size() != R:
        raise Error(
            String("cross_entropy: ", R, " rows of logits, ",
                   t.nodes[tgt].size(), " targets")
        )
    var o = t.new_node(1, 1, OP_XENT, z, tgt, t.nodes[z].rg)
    if t.ctx:
        var c = t.dctx()
        ensure_gpu(t.nodes[o].aux, c, R)
        var rows_loss = t.nodes[o].aux.dev.value().create_sub_buffer[DT](0, R)
        c.enqueue_function[_xent_rows_kernel](
            rows_loss, t.dv(z), t.dv(tgt), Int64(R), Int64(C),
            grid_dim=_grid(R), block_dim=TPB,
        )
        gpu_reduce[True](c, t.dv(o), rows_loss, R)
        return o
    var pz = t.val_ptr(z)
    var pt = t.val_ptr(tgt)
    var total: Scalar[DT] = 0
    for r in range(R):
        var base = r * C
        var cls = Int(pt[unsafe_offset=r])
        if cls < 0 or cls >= C or Scalar[DT](cls) != pt[unsafe_offset=r]:
            raise Error(String("cross_entropy: target ", pt[unsafe_offset=r], " not in [0, ", C, ")"))
        var m = pz[unsafe_offset=base]
        for j in range(1, C):
            m = max(m, pz[unsafe_offset=base + j])
        var s: Scalar[DT] = 0
        for j in range(C):
            s += exp(pz[unsafe_offset=base + j] - m)
        total += m + log(s) - pz[unsafe_offset=base + cls]
    t.val_ptr(o)[unsafe_offset=0] = total / Scalar[DT](R)
    return o


# ── dropout ──────────────────────────────────────────────────────────────
@always_inline
def _keep(seed: UInt32, k: Int, p: Scalar[DT]) -> Bool:
    """Element `k` survives: a 32-bit integer hash (lowbias32) of the seed
    and the index, top 24 bits as a uniform in [0, 1), compared with `p`.
    Identical on the CPU and the GPU — pure integer arithmetic."""
    var x = seed ^ (UInt32(k) * UInt32(0x9E3779B9))
    x ^= x >> 16
    x *= UInt32(0x7FEB352D)
    x ^= x >> 15
    x *= UInt32(0x846CA68B)
    x ^= x >> 16
    var u = Scalar[DT](Float32(x >> 8) * Float32(1.0 / 16777216.0))
    return u >= p


def _dropout_kernel(o: P, a: P, n: Int64, p: Scalar[DT], seed: UInt32):
    var k = Int(global_idx.x)
    if k < Int(n):
        var scale = Scalar[DT](1) / (Scalar[DT](1) - p)
        o[unsafe_offset=k] = a[unsafe_offset=k] * scale if _keep(seed, k, p) else Scalar[DT](0)


def _dropout_bwd_kernel[ACC: Bool](ga: P, go: P, n: Int64, p: Scalar[DT], seed: UInt32):
    var k = Int(global_idx.x)
    if k >= Int(n):
        return
    var scale = Scalar[DT](1) / (Scalar[DT](1) - p)
    var g = go[unsafe_offset=k] * scale if _keep(seed, k, p) else Scalar[DT](0)
    comptime if ACC:
        ga[unsafe_offset=k] += g
    else:
        ga[unsafe_offset=k] = g


def ex_dropout(mut t: TapeData, a: Int, p: Scalar[DT], seed: Int) raises -> Int:
    if p < 0 or p >= 1:
        raise Error("dropout: p must be in [0, 1)")
    var o = t.new_node(t.nodes[a].rows, t.nodes[a].cols, OP_DROPOUT, a, -1, t.nodes[a].rg)
    t.like(o, a)
    t.nodes[o].c = p
    t.nodes[o].i0 = seed
    var n = t.nodes[a].size()
    var s = UInt32(seed & 0xFFFFFFFF)
    if t.ctx:
        t.dctx().enqueue_function[_dropout_kernel](
            t.dv(o), t.dv(a), Int64(n), p, s, grid_dim=_grid(n), block_dim=TPB
        )
        return o
    var po = t.val_ptr(o)
    var pa = t.val_ptr(a)
    var scale = Scalar[DT](1) / (Scalar[DT](1) - p)
    for k in range(n):
        po[unsafe_offset=k] = pa[unsafe_offset=k] * scale if _keep(s, k, p) else Scalar[DT](0)
    return o


# ── backward ─────────────────────────────────────────────────────────────
def extra_vjp(mut t: TapeData, o: Int) raises:
    var op = t.nodes[o].op
    var a = t.nodes[o].a
    if not t.nodes[a].rg:
        return
    var n = t.nodes[a].size()
    if t.ctx:
        var c = t.dctx()
        var ga = t.dg(a)
        var acc = t._take(a)
        if op == OP_CLAMP:
            var lo = t.nodes[o].c
            var hi = t.nodes[o].c2
            if acc:
                c.enqueue_function[_clamp_bwd_kernel[True]](ga, t.dg(o), t.dv(a), Int64(n), lo, hi, grid_dim=_grid(n), block_dim=TPB)
            else:
                c.enqueue_function[_clamp_bwd_kernel[False]](ga, t.dg(o), t.dv(a), Int64(n), lo, hi, grid_dim=_grid(n), block_dim=TPB)
        elif op == OP_SOFTMAX:
            var R = t.nodes[a].rows
            var C = t.nodes[a].cols
            var log_ = t.nodes[o].sub == 1
            if log_ and acc:
                c.enqueue_function[_softmax_bwd_kernel[True, True]](ga, t.dg(o), t.dv(o), Int64(R), Int64(C), grid_dim=_grid(R), block_dim=TPB)
            elif log_:
                c.enqueue_function[_softmax_bwd_kernel[True, False]](ga, t.dg(o), t.dv(o), Int64(R), Int64(C), grid_dim=_grid(R), block_dim=TPB)
            elif acc:
                c.enqueue_function[_softmax_bwd_kernel[False, True]](ga, t.dg(o), t.dv(o), Int64(R), Int64(C), grid_dim=_grid(R), block_dim=TPB)
            else:
                c.enqueue_function[_softmax_bwd_kernel[False, False]](ga, t.dg(o), t.dv(o), Int64(R), Int64(C), grid_dim=_grid(R), block_dim=TPB)
        elif op == OP_XENT:
            var R = t.nodes[a].rows
            var C = t.nodes[a].cols
            var tg = t.dv(t.nodes[o].b)
            if acc:
                c.enqueue_function[_xent_bwd_kernel[True]](ga, t.dg(o), t.dv(a), tg, Int64(R), Int64(C), grid_dim=_grid(R), block_dim=TPB)
            else:
                c.enqueue_function[_xent_bwd_kernel[False]](ga, t.dg(o), t.dv(a), tg, Int64(R), Int64(C), grid_dim=_grid(R), block_dim=TPB)
        elif op == OP_DROPOUT:
            var p = t.nodes[o].c
            var s = UInt32(t.nodes[o].i0 & 0xFFFFFFFF)
            if acc:
                c.enqueue_function[_dropout_bwd_kernel[True]](ga, t.dg(o), Int64(n), p, s, grid_dim=_grid(n), block_dim=TPB)
            else:
                c.enqueue_function[_dropout_bwd_kernel[False]](ga, t.dg(o), Int64(n), p, s, grid_dim=_grid(n), block_dim=TPB)
        else:
            raise Error(String("extra backward: unknown op ", op))
        return
    var ga = t.grad_ptr(a)
    var go = t.grad_ptr(o)
    if op == OP_CLAMP:
        var lo = t.nodes[o].c
        var hi = t.nodes[o].c2
        var pa = t.val_ptr(a)
        for k in range(n):
            var x = pa[unsafe_offset=k]
            if x >= lo and x <= hi:
                ga[unsafe_offset=k] += go[unsafe_offset=k]
    elif op == OP_SOFTMAX:
        var R = t.nodes[a].rows
        var C = t.nodes[a].cols
        var log_ = t.nodes[o].sub == 1
        var y = t.val_ptr(o)
        for r in range(R):
            var base = r * C
            var s: Scalar[DT] = 0
            for j in range(C):
                s += go[unsafe_offset=base + j] if log_ else go[unsafe_offset=base + j] * y[unsafe_offset=base + j]
            for j in range(C):
                if log_:
                    ga[unsafe_offset=base + j] += go[unsafe_offset=base + j] - exp(y[unsafe_offset=base + j]) * s
                else:
                    ga[unsafe_offset=base + j] += y[unsafe_offset=base + j] * (go[unsafe_offset=base + j] - s)
    elif op == OP_XENT:
        var R = t.nodes[a].rows
        var C = t.nodes[a].cols
        var pz = t.val_ptr(a)
        var pt = t.val_ptr(t.nodes[o].b)
        var g = go[unsafe_offset=0] / Scalar[DT](R)
        for r in range(R):
            var base = r * C
            var m = pz[unsafe_offset=base]
            for j in range(1, C):
                m = max(m, pz[unsafe_offset=base + j])
            var s: Scalar[DT] = 0
            for j in range(C):
                s += exp(pz[unsafe_offset=base + j] - m)
            var cls = Int(pt[unsafe_offset=r])
            for j in range(C):
                var p = exp(pz[unsafe_offset=base + j] - m) / s
                ga[unsafe_offset=base + j] += (p - (Scalar[DT](1) if j == cls else Scalar[DT](0))) * g
    elif op == OP_DROPOUT:
        var p = t.nodes[o].c
        var s = UInt32(t.nodes[o].i0 & 0xFFFFFFFF)
        var scale = Scalar[DT](1) / (Scalar[DT](1) - p)
        for k in range(n):
            if _keep(s, k, p):
                ga[unsafe_offset=k] += go[unsafe_offset=k] * scale
    else:
        raise Error(String("extra backward: unknown op ", op))
