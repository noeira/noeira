"""N-d ops of the functional tape: broadcasting over four dims, batched
matmul, reshape, permute, reductions along one axis, narrow, concat, gather.

A node's N-d shape is four dims LEFT-padded with 1s (`Node.d0..d3`, rank
<= 4); its 2-D view (`rows` = product of the leading dims, `cols` = the
last) is what every older op reads, so those keep working on N-d data
(`x @ w`, `F.linear`, a module over `[B, T, D]` act on `[B*T, D]`).

Data movement — reshape, permute, narrow, concat, forward and backward — is
ONE kernel, `_scopy_kernel`: iterate a 4-d index space and copy (or add)
`dst[doff + Σ x_k·ds_k] ← src[soff + Σ x_k·ss_k]`. Each op is a choice of
index space and strides.

Kernel arguments pack two 32-bit numbers per `Int64` (`_pk`): dims and
strides are far below 2^31, and it keeps every kernel near ten arguments
(Metal's argument table is finite; see the repo's notes on it).

Every op has a CPU and a GPU path; every backward follows the tape's GPU
rule — the FIRST contribution to an activation's gradient writes it
(`TapeData._take`), later ones add — and the CPU's (gradients zero-filled
before the walk, always add).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.fill import fill_dev
from noeira.nn.core.cpu_gemm import gemm_cpu_ptr
from .opcodes import *
from .tape import TapeData
from .gpu import gpu_matmul, gpu_matmul_bwd_a, gpu_matmul_bwd_b, bin_term
from .extra import extra_vjp


comptime P = Pointer[Scalar[DT], MutAnyOrigin]
comptime DBuf = DeviceBuffer[DT]
comptime TPB = 256


@always_inline
def _grid(n: Int) -> Int:
    return (n + TPB - 1) // TPB


@always_inline
def _pk(a: Int, b: Int) -> Int64:
    return (Int64(a) << 32) | Int64(b)


@always_inline
def _hi(v: Int64) -> Int:
    return Int(v >> 32)


@always_inline
def _lo(v: Int64) -> Int:
    return Int(v & 0xFFFFFFFF)


@fieldwise_init
struct D4(ImplicitlyCopyable):
    """Four padded dims (or four strides)."""

    var a: Int
    var b: Int
    var c: Int
    var d: Int

    def size(self) -> Int:
        return self.a * self.b * self.c * self.d

    def strides(self) -> D4:
        """Row-major strides of these dims."""
        return D4(self.b * self.c * self.d, self.c * self.d, self.d, 1)

    def at(self, p: Int) -> Int:
        if p == 0:
            return self.a
        elif p == 1:
            return self.b
        elif p == 2:
            return self.c
        return self.d

    def with_(self, p: Int, v: Int) -> D4:
        var r = self
        if p == 0:
            r.a = v
        elif p == 1:
            r.b = v
        elif p == 2:
            r.c = v
        else:
            r.d = v
        return r

    def p01(self) -> Int64:
        return _pk(self.a, self.b)

    def p23(self) -> Int64:
        return _pk(self.c, self.d)


def _shape_str(t: TapeData, i: Int) -> String:
    var out = String("[")
    ref n = t.nodes[i]
    for p in range(4 - n.rank, 4):
        if p > 4 - n.rank:
            out += ", "
        out += String(n.dim(p))
    return out + "]"


def _dims(t: TapeData, i: Int) -> D4:
    ref n = t.nodes[i]
    return D4(n.d0, n.d1, n.d2, n.d3)


def _new(mut t: TapeData, d: D4, rank: Int, op: Int, a: Int, b: Int, rg: Bool) raises -> Int:
    var o = t.new_node(d.a * d.b * d.c, d.d, op, a, b, rg)
    t.nodes[o].set_dims(d.a, d.b, d.c, d.d, rank)
    return o


def _axis(t: TapeData, i: Int, axis: Int) raises -> Int:
    """Logical axis (negative counts from the end) -> padded axis 0..3."""
    var r = t.nodes[i].rank
    var ax = axis + r if axis < 0 else axis
    if ax < 0 or ax >= r:
        raise Error(String("axis ", axis, " out of range for rank ", r))
    return ax + 4 - r


# ── the one data-movement kernel ─────────────────────────────────────────
def _scopy_kernel[ACC: Bool](
    dst: P, src: P, it01: Int64, it23: Int64, ds01: Int64, ds23: Int64,
    ss01: Int64, ss23: Int64, offs: Int64,
):
    var i1 = _lo(it01)
    var i2 = _hi(it23)
    var i3 = _lo(it23)
    var t = Int(global_idx.x)
    if t >= _hi(it01) * i1 * i2 * i3:
        return
    var x3 = t % i3
    var r = t // i3
    var x2 = r % i2
    r //= i2
    var x1 = r % i1
    var x0 = r // i1
    var di = _hi(offs) + x0 * _hi(ds01) + x1 * _lo(ds01) + x2 * _hi(ds23) + x3 * _lo(ds23)
    var si = _lo(offs) + x0 * _hi(ss01) + x1 * _lo(ss01) + x2 * _hi(ss23) + x3 * _lo(ss23)
    comptime if ACC:
        dst[unsafe_offset=di] += src[unsafe_offset=si]
    else:
        dst[unsafe_offset=di] = src[unsafe_offset=si]


def _copy(
    mut t: TapeData, dst: Int, dgrad: Bool, src: Int, sgrad: Bool,
    it: D4, ds: D4, doff: Int, ss: D4, soff: Int, acc: Bool,
) raises:
    """Over the index space `it`: dst(·) ← / += src(·), each side its node's
    value or gradient, strides and offset."""
    if t.ctx:
        var c = t.dctx()
        var db = t.dg(dst) if dgrad else t.dv(dst)
        var sb = t.dg(src) if sgrad else t.dv(src)
        if acc:
            c.enqueue_function[_scopy_kernel[True]](
                db, sb, it.p01(), it.p23(), ds.p01(), ds.p23(), ss.p01(),
                ss.p23(), _pk(doff, soff), grid_dim=_grid(it.size()),
                block_dim=TPB,
            )
        else:
            c.enqueue_function[_scopy_kernel[False]](
                db, sb, it.p01(), it.p23(), ds.p01(), ds.p23(), ss.p01(),
                ss.p23(), _pk(doff, soff), grid_dim=_grid(it.size()),
                block_dim=TPB,
            )
        return
    var dp = t.grad_ptr(dst) if dgrad else t.val_ptr(dst)
    var sp = t.grad_ptr(src) if sgrad else t.val_ptr(src)
    for x0 in range(it.a):
        for x1 in range(it.b):
            for x2 in range(it.c):
                for x3 in range(it.d):
                    var di = doff + x0 * ds.a + x1 * ds.b + x2 * ds.c + x3 * ds.d
                    var si = soff + x0 * ss.a + x1 * ss.b + x2 * ss.c + x3 * ss.d
                    if acc:
                        dp[unsafe_offset=di] += sp[unsafe_offset=si]
                    else:
                        dp[unsafe_offset=di] = sp[unsafe_offset=si]


def _zero_grad_if_fresh(mut t: TapeData, i: Int) raises:
    """For a backward that only ADDS into part of `i`'s gradient (narrow,
    max, gather): a fresh GPU gradient is cleared first."""
    if t.ctx and not t._take(i):
        fill_dev(t.dg(i), t.nodes[i].size(), Scalar[DT](0), t.dctx())


# ── broadcasting binary ops ──────────────────────────────────────────────
def _bin4_kernel[OP: Int](
    o: P, a: P, b: P, od01: Int64, od23: Int64, ad01: Int64, ad23: Int64,
    bd01: Int64, bd23: Int64,
):
    var O1 = _lo(od01)
    var O2 = _hi(od23)
    var O3 = _lo(od23)
    var t = Int(global_idx.x)
    if t >= _hi(od01) * O1 * O2 * O3:
        return
    var x3 = t % O3
    var r = t // O3
    var x2 = r % O2
    r //= O2
    var x1 = r % O1
    var x0 = r // O1
    var A0 = _hi(ad01)
    var A1 = _lo(ad01)
    var A2 = _hi(ad23)
    var A3 = _lo(ad23)
    var B0 = _hi(bd01)
    var B1 = _lo(bd01)
    var B2 = _hi(bd23)
    var B3 = _lo(bd23)
    var ka = (((x0 if A0 > 1 else 0) * A1 + (x1 if A1 > 1 else 0)) * A2 + (x2 if A2 > 1 else 0)) * A3 + (x3 if A3 > 1 else 0)
    var kb = (((x0 if B0 > 1 else 0) * B1 + (x1 if B1 > 1 else 0)) * B2 + (x2 if B2 > 1 else 0)) * B3 + (x3 if B3 > 1 else 0)
    o[unsafe_offset=t] = bin_op[OP](a[unsafe_offset=ka], b[unsafe_offset=kb])


def _bin4_bwd_kernel[OP: Int, WHICH: Int, ACC: Bool](
    gx: P, go: P, a: P, b: P, od01: Int64, od23: Int64, ad01: Int64,
    ad23: Int64, bd01: Int64, bd23: Int64,
):
    """One thread per element of input `WHICH`: its gradient summed over
    every output element it was broadcast to."""
    var O0 = _hi(od01)
    var O1 = _lo(od01)
    var O2 = _hi(od23)
    var O3 = _lo(od23)
    var A0 = _hi(ad01)
    var A1 = _lo(ad01)
    var A2 = _hi(ad23)
    var A3 = _lo(ad23)
    var B0 = _hi(bd01)
    var B1 = _lo(bd01)
    var B2 = _hi(bd23)
    var B3 = _lo(bd23)
    var X0 = A0 if WHICH == 0 else B0
    var X1 = A1 if WHICH == 0 else B1
    var X2 = A2 if WHICH == 0 else B2
    var X3 = A3 if WHICH == 0 else B3
    var t = Int(global_idx.x)
    if t >= X0 * X1 * X2 * X3:
        return
    var x3 = t % X3
    var r = t // X3
    var x2 = r % X2
    r //= X2
    var x1 = r % X1
    var x0 = r // X1
    var acc: Scalar[DT] = 0
    for y0 in range(x0 if X0 == O0 else 0, x0 + 1 if X0 == O0 else O0):
        for y1 in range(x1 if X1 == O1 else 0, x1 + 1 if X1 == O1 else O1):
            for y2 in range(x2 if X2 == O2 else 0, x2 + 1 if X2 == O2 else O2):
                for y3 in range(x3 if X3 == O3 else 0, x3 + 1 if X3 == O3 else O3):
                    var oi = ((y0 * O1 + y1) * O2 + y2) * O3 + y3
                    var ka = (((y0 if A0 > 1 else 0) * A1 + (y1 if A1 > 1 else 0)) * A2 + (y2 if A2 > 1 else 0)) * A3 + (y3 if A3 > 1 else 0)
                    var kb = (((y0 if B0 > 1 else 0) * B1 + (y1 if B1 > 1 else 0)) * B2 + (y2 if B2 > 1 else 0)) * B3 + (y3 if B3 > 1 else 0)
                    acc += bin_term[OP, WHICH](
                        go[unsafe_offset=oi], a[unsafe_offset=ka], b[unsafe_offset=kb]
                    )
    comptime if ACC:
        gx[unsafe_offset=t] += acc
    else:
        gx[unsafe_offset=t] = acc


def nd_binary[OP: Int](mut t: TapeData, a: Int, b: Int) raises -> Int:
    var da = _dims(t, a)
    var db = _dims(t, b)
    var od = D4(0, 0, 0, 0)
    for p in range(4):
        var x = da.at(p)
        var y = db.at(p)
        if x == y or y == 1:
            od = od.with_(p, x)
        elif x == 1:
            od = od.with_(p, y)
        else:
            raise Error(
                String("shapes ", _shape_str(t, a), " and ",
                       _shape_str(t, b), " do not broadcast")
            )
    var rank = max(t.nodes[a].rank, t.nodes[b].rank)
    var o = _new(t, od, rank, OP, a, b, t.nodes[a].rg or t.nodes[b].rg)
    if t.ctx:
        var c = t.dctx()
        c.enqueue_function[_bin4_kernel[OP]](
            t.dv(o), t.dv(a), t.dv(b), od.p01(), od.p23(), da.p01(),
            da.p23(), db.p01(), db.p23(), grid_dim=_grid(od.size()),
            block_dim=TPB,
        )
        return o
    var po = t.val_ptr(o)
    var pa = t.val_ptr(a)
    var pb = t.val_ptr(b)
    var k = 0
    for x0 in range(od.a):
        for x1 in range(od.b):
            for x2 in range(od.c):
                for x3 in range(od.d):
                    var ka = (((x0 if da.a > 1 else 0) * da.b + (x1 if da.b > 1 else 0)) * da.c + (x2 if da.c > 1 else 0)) * da.d + (x3 if da.d > 1 else 0)
                    var kb = (((x0 if db.a > 1 else 0) * db.b + (x1 if db.b > 1 else 0)) * db.c + (x2 if db.c > 1 else 0)) * db.d + (x3 if db.d > 1 else 0)
                    po[unsafe_offset=k] = bin_op[OP](pa[unsafe_offset=ka], pb[unsafe_offset=kb])
                    k += 1
    return o


def _bin4_bwd_cpu[OP: Int, WHICH: Int](
    gx: P, go: P, pa: P, pb: P, od: D4, da: D4, db: D4
):
    for y0 in range(od.a):
        for y1 in range(od.b):
            for y2 in range(od.c):
                for y3 in range(od.d):
                    var oi = ((y0 * od.b + y1) * od.c + y2) * od.d + y3
                    var ka = (((y0 if da.a > 1 else 0) * da.b + (y1 if da.b > 1 else 0)) * da.c + (y2 if da.c > 1 else 0)) * da.d + (y3 if da.d > 1 else 0)
                    var kb = (((y0 if db.a > 1 else 0) * db.b + (y1 if db.b > 1 else 0)) * db.c + (y2 if db.c > 1 else 0)) * db.d + (y3 if db.d > 1 else 0)
                    var kx = ka if WHICH == 0 else kb
                    gx[unsafe_offset=kx] += bin_term[OP, WHICH](
                        go[unsafe_offset=oi], pa[unsafe_offset=ka], pb[unsafe_offset=kb]
                    )


def _bin4_bwd[OP: Int](mut t: TapeData, o: Int) raises:
    var a = t.nodes[o].a
    var b = t.nodes[o].b
    var od = _dims(t, o)
    var da = _dims(t, a)
    var db = _dims(t, b)
    comptime for WHICH in range(2):
        var x = a if WHICH == 0 else b
        if t.nodes[x].rg:
            var n = (da if WHICH == 0 else db).size()
            if t.ctx:
                var c = t.dctx()
                var gx = t.dg(x)
                if t._take(x):
                    c.enqueue_function[_bin4_bwd_kernel[OP, WHICH, True]](
                        gx, t.dg(o), t.dv(a), t.dv(b), od.p01(), od.p23(),
                        da.p01(), da.p23(), db.p01(), db.p23(),
                        grid_dim=_grid(n), block_dim=TPB,
                    )
                else:
                    c.enqueue_function[_bin4_bwd_kernel[OP, WHICH, False]](
                        gx, t.dg(o), t.dv(a), t.dv(b), od.p01(), od.p23(),
                        da.p01(), da.p23(), db.p01(), db.p23(),
                        grid_dim=_grid(n), block_dim=TPB,
                    )
            else:
                _bin4_bwd_cpu[OP, WHICH](
                    t.grad_ptr(x), t.grad_ptr(o), t.val_ptr(a), t.val_ptr(b),
                    od, da, db,
                )


# ── batched matmul ───────────────────────────────────────────────────────
def nd_bmm(mut t: TapeData, a: Int, b: Int) raises -> Int:
    """`[..., m, k] @ [..., k, n]` with the SAME leading dims: one GEMM per
    batch (cuBLAS / MAX / cblas, the 2-D path's own)."""
    var da = _dims(t, a)
    var db = _dims(t, b)
    if da.a != db.a or da.b != db.b or da.d != db.c or t.nodes[a].rank != t.nodes[b].rank:
        raise Error(
            String("matmul: ", _shape_str(t, a), " @ ",
                   _shape_str(t, b),
                   " — batched matmul needs equal leading dims")
        )
    var m = da.c
    var k = da.d
    var n = db.d
    var o = _new(
        t, D4(da.a, da.b, m, n), t.nodes[a].rank, OP_MATMUL, a, b,
        t.nodes[a].rg or t.nodes[b].rg,
    )
    for bi in range(da.a * da.b):
        if t.ctx:
            var c = t.dctx()
            var ov = t.nodes[o].val.dev.value().create_sub_buffer[DT](bi * m * n, m * n)
            var av = t.nodes[a].val.dev.value().create_sub_buffer[DT](bi * m * k, m * k)
            var bv = t.nodes[b].val.dev.value().create_sub_buffer[DT](bi * k * n, k * n)
            gpu_matmul(c, ov, av, bv, m, n, k)
        else:
            gemm_cpu_ptr[False, False](
                m, n, k, 1.0,
                rebind[Pointer[Float32, ImmutAnyOrigin]](t.val_ptr(a) + bi * m * k), k,
                rebind[Pointer[Float32, ImmutAnyOrigin]](t.val_ptr(b) + bi * k * n), n,
                0.0,
                rebind[Pointer[Float32, MutAnyOrigin]](t.val_ptr(o) + bi * m * n), n,
            )
    return o


def _bmm_bwd(mut t: TapeData, o: Int) raises:
    var a = t.nodes[o].a
    var b = t.nodes[o].b
    var da = _dims(t, a)
    var m = da.c
    var k = da.d
    var n = t.nodes[b].d3
    var nb = da.a * da.b
    if t.ctx:
        var c = t.dctx()
        if t.nodes[a].rg:
            var acc = t._take(a)
            for bi in range(nb):
                var ga = t.nodes[a].grad.dev.value().create_sub_buffer[DT](bi * m * k, m * k)
                var go = t.nodes[o].grad.dev.value().create_sub_buffer[DT](bi * m * n, m * n)
                var bv = t.nodes[b].val.dev.value().create_sub_buffer[DT](bi * k * n, k * n)
                var tmp = t._scratch2(m * k)
                if acc:
                    gpu_matmul_bwd_a[True](c, ga, go, bv, m, n, k, tmp)
                else:
                    gpu_matmul_bwd_a[False](c, ga, go, bv, m, n, k, tmp)
        if t.nodes[b].rg:
            var acc = t._take(b)
            for bi in range(nb):
                var gb = t.nodes[b].grad.dev.value().create_sub_buffer[DT](bi * k * n, k * n)
                var go = t.nodes[o].grad.dev.value().create_sub_buffer[DT](bi * m * n, m * n)
                var av = t.nodes[a].val.dev.value().create_sub_buffer[DT](bi * m * k, m * k)
                var t1 = t._scratch1(m * k)
                var t2 = t._scratch2(k * n)
                if acc:
                    gpu_matmul_bwd_b[True](c, gb, av, go, m, n, k, t1, t2)
                else:
                    gpu_matmul_bwd_b[False](c, gb, av, go, m, n, k, t1, t2)
        return
    for bi in range(nb):
        var go = rebind[Pointer[Float32, ImmutAnyOrigin]](t.grad_ptr(o) + bi * m * n)
        if t.nodes[a].rg:
            gemm_cpu_ptr[False, True](
                m, k, n, 1.0, go, n,
                rebind[Pointer[Float32, ImmutAnyOrigin]](t.val_ptr(b) + bi * k * n), n,
                1.0,
                rebind[Pointer[Float32, MutAnyOrigin]](t.grad_ptr(a) + bi * m * k), k,
            )
        if t.nodes[b].rg:
            gemm_cpu_ptr[True, False](
                k, n, m, 1.0,
                rebind[Pointer[Float32, ImmutAnyOrigin]](t.val_ptr(a) + bi * m * k), k,
                go, n, 1.0,
                rebind[Pointer[Float32, MutAnyOrigin]](t.grad_ptr(b) + bi * k * n), n,
            )


def nd_binary_vjp(mut t: TapeData, o: Int) raises:
    var op = t.nodes[o].op
    if op == OP_ADD:
        _bin4_bwd[OP_ADD](t, o)
    elif op == OP_SUB:
        _bin4_bwd[OP_SUB](t, o)
    elif op == OP_MUL:
        _bin4_bwd[OP_MUL](t, o)
    elif op == OP_DIV:
        _bin4_bwd[OP_DIV](t, o)
    else:
        _bmm_bwd(t, o)


# ── reshape, permute, narrow, concat ─────────────────────────────────────
def nd_reshape(mut t: TapeData, a: Int, shape: List[Int]) raises -> Int:
    var r = len(shape)
    if r < 1 or r > 4:
        raise Error("reshape: rank 1 to 4")
    var known = 1
    var infer = -1
    for i in range(r):
        if shape[i] == -1:
            if infer >= 0:
                raise Error("reshape: one -1 at most")
            infer = i
        else:
            known *= shape[i]
    var n = t.nodes[a].size()
    var dims = List[Int](length=4, fill=1)
    for i in range(r):
        dims[4 - r + i] = shape[i]
    if infer >= 0:
        if known == 0 or n % known != 0:
            raise Error("reshape: cannot infer -1")
        dims[4 - r + infer] = n // known
    var d = D4(dims[0], dims[1], dims[2], dims[3])
    if d.size() != n:
        raise Error(String("reshape: ", n, " elements into ", d.size()))
    var o = _new(t, d, r, OP_RESHAPE, a, -1, t.nodes[a].rg)
    var flat = D4(1, 1, 1, n)
    _copy(t, o, False, a, False, flat, D4(0, 0, 0, 1), 0, D4(0, 0, 0, 1), 0, False)
    return o


def nd_permute(mut t: TapeData, a: Int, perm: List[Int]) raises -> Int:
    """`perm[i]` = the input axis that becomes output axis `i` (torch's)."""
    var r = t.nodes[a].rank
    if len(perm) != r:
        raise Error(String("permute: ", len(perm), " axes for rank ", r))
    var src = D4(0, 1, 2, 3)  # padded source axis of each padded output axis
    var seen = List[Bool](length=r, fill=False)
    for i in range(r):
        var ax = perm[i] + r if perm[i] < 0 else perm[i]
        if ax < 0 or ax >= r or seen[ax]:
            raise Error("permute: not a permutation")
        seen[ax] = True
        src = src.with_(4 - r + i, 4 - r + ax)
    var da = _dims(t, a)
    var sa = da.strides()
    var od = D4(da.at(src.a), da.at(src.b), da.at(src.c), da.at(src.d))
    var o = _new(t, od, r, OP_PERMUTE, a, -1, t.nodes[a].rg)
    t.nodes[o].i0 = src.a
    t.nodes[o].i1 = src.b
    t.nodes[o].i2 = src.c
    t.nodes[o].i3 = src.d
    var ss = D4(sa.at(src.a), sa.at(src.b), sa.at(src.c), sa.at(src.d))
    _copy(t, o, False, a, False, od, od.strides(), 0, ss, 0, False)
    return o


def nd_narrow(mut t: TapeData, a: Int, axis: Int, start: Int, length: Int) raises -> Int:
    var p = _axis(t, a, axis)
    var da = _dims(t, a)
    if start < 0 or length < 1 or start + length > da.at(p):
        raise Error(String("narrow: [", start, ", ", start + length, ") out of ", da.at(p)))
    var od = da.with_(p, length)
    var o = _new(t, od, t.nodes[a].rank, OP_NARROW, a, -1, t.nodes[a].rg)
    t.nodes[o].i0 = p
    t.nodes[o].i1 = start
    _copy(t, o, False, a, False, od, od.strides(), 0, da.strides(),
          start * da.strides().at(p), False)
    return o


def nd_concat(mut t: TapeData, a: Int, b: Int, axis: Int) raises -> Int:
    var p = _axis(t, a, axis)
    var da = _dims(t, a)
    var db = _dims(t, b)
    if t.nodes[a].rank != t.nodes[b].rank:
        raise Error("concat: ranks differ")
    for q in range(4):
        if q != p and da.at(q) != db.at(q):
            raise Error(
                String("concat: ", _shape_str(t, a), " and ",
                       _shape_str(t, b), " differ off the axis")
            )
    var od = da.with_(p, da.at(p) + db.at(p))
    var o = _new(t, od, t.nodes[a].rank, OP_CONCAT, a, b, t.nodes[a].rg or t.nodes[b].rg)
    t.nodes[o].i0 = p
    var so = od.strides()
    _copy(t, o, False, a, False, da, so, 0, da.strides(), 0, False)
    _copy(t, o, False, b, False, db, so, da.at(p) * so.at(p), db.strides(), 0, False)
    return o


# ── reductions along one axis ────────────────────────────────────────────
def _red_kernel[RED: Int](o: P, a: P, ad01: Int64, ad23: Int64, p_: Int64):
    """One thread per output element (the input's dims with axis p = 1)."""
    var p = Int(p_)
    var A1 = _lo(ad01)
    var A2 = _hi(ad23)
    var A3 = _lo(ad23)
    var n = _hi(ad01) if p == 0 else (A1 if p == 1 else (A2 if p == 2 else A3))
    var O0 = 1 if p == 0 else _hi(ad01)
    var O1 = 1 if p == 1 else A1
    var O2 = 1 if p == 2 else A2
    var O3 = 1 if p == 3 else A3
    var t = Int(global_idx.x)
    if t >= O0 * O1 * O2 * O3:
        return
    var x3 = t % O3
    var r = t // O3
    var x2 = r % O2
    r //= O2
    var x1 = r % O1
    var x0 = r // O1
    var base = ((x0 * A1 + x1) * A2 + x2) * A3 + x3
    var stride = A1 * A2 * A3 if p == 0 else (A2 * A3 if p == 1 else (A3 if p == 2 else 1))
    var s: Scalar[DT] = 0
    comptime if RED == RED_MAX:
        s = a[unsafe_offset=base]
    for j in range(n):
        var v = a[unsafe_offset=base + j * stride]
        comptime if RED == RED_MAX:
            if v > s:
                s = v
        else:
            s += v
    comptime if RED == RED_MEAN:
        s = s / Scalar[DT](n)
    o[unsafe_offset=t] = s


def _red_bwd_sum_kernel[MEAN: Bool, ACC: Bool](
    ga: P, go: P, ad01: Int64, ad23: Int64, p_: Int64
):
    """One thread per INPUT element: the output gradient it fed."""
    var p = Int(p_)
    var A0 = _hi(ad01)
    var A1 = _lo(ad01)
    var A2 = _hi(ad23)
    var A3 = _lo(ad23)
    var t = Int(global_idx.x)
    if t >= A0 * A1 * A2 * A3:
        return
    var x3 = t % A3
    var r = t // A3
    var x2 = r % A2
    r //= A2
    var x1 = r % A1
    var x0 = r // A1
    var O1 = 1 if p == 1 else A1
    var O2 = 1 if p == 2 else A2
    var O3 = 1 if p == 3 else A3
    var oi = (((0 if p == 0 else x0) * O1 + (0 if p == 1 else x1)) * O2 + (0 if p == 2 else x2)) * O3 + (0 if p == 3 else x3)
    var g = go[unsafe_offset=oi]
    comptime if MEAN:
        var n = A0 if p == 0 else (A1 if p == 1 else (A2 if p == 2 else A3))
        g = g / Scalar[DT](n)
    comptime if ACC:
        ga[unsafe_offset=t] += g
    else:
        ga[unsafe_offset=t] = g


def _red_bwd_max_kernel(ga: P, go: P, a: P, ad01: Int64, ad23: Int64, p_: Int64):
    """One thread per OUTPUT element: its gradient to the FIRST maximum."""
    var p = Int(p_)
    var A1 = _lo(ad01)
    var A2 = _hi(ad23)
    var A3 = _lo(ad23)
    var n = _hi(ad01) if p == 0 else (A1 if p == 1 else (A2 if p == 2 else A3))
    var O0 = 1 if p == 0 else _hi(ad01)
    var O1 = 1 if p == 1 else A1
    var O2 = 1 if p == 2 else A2
    var O3 = 1 if p == 3 else A3
    var t = Int(global_idx.x)
    if t >= O0 * O1 * O2 * O3:
        return
    var x3 = t % O3
    var r = t // O3
    var x2 = r % O2
    r //= O2
    var x1 = r % O1
    var x0 = r // O1
    var base = ((x0 * A1 + x1) * A2 + x2) * A3 + x3
    var stride = A1 * A2 * A3 if p == 0 else (A2 * A3 if p == 1 else (A3 if p == 2 else 1))
    var best = a[unsafe_offset=base]
    var arg = 0
    for j in range(1, n):
        var v = a[unsafe_offset=base + j * stride]
        if v > best:
            best = v
            arg = j
    ga[unsafe_offset=base + arg * stride] += go[unsafe_offset=t]


def nd_reduce(mut t: TapeData, a: Int, axis: Int, red: Int, keepdim: Bool) raises -> Int:
    var p = _axis(t, a, axis)
    var da = _dims(t, a)
    var r = t.nodes[a].rank
    var o: Int
    if keepdim or r == 1:
        o = _new(t, da.with_(p, 1), r, OP_REDUCE_AXIS, a, -1, t.nodes[a].rg)
    else:
        # Same data order as keepdim (a size-1 axis moves nothing); only the
        # shape loses the axis.
        var dims = List[Int]()
        for q in range(4 - r, 4):
            if q != p:
                dims.append(da.at(q))
        var nr = len(dims)
        var full = List[Int](length=4, fill=1)
        for i in range(nr):
            full[4 - nr + i] = dims[i]
        o = _new(t, D4(full[0], full[1], full[2], full[3]), nr, OP_REDUCE_AXIS, a, -1, t.nodes[a].rg)
    t.nodes[o].sub = red
    t.nodes[o].i0 = p
    var od = da.with_(p, 1)
    if t.ctx:
        var c = t.dctx()
        if red == RED_SUM:
            c.enqueue_function[_red_kernel[RED_SUM]](t.dv(o), t.dv(a), da.p01(), da.p23(), Int64(p), grid_dim=_grid(od.size()), block_dim=TPB)
        elif red == RED_MEAN:
            c.enqueue_function[_red_kernel[RED_MEAN]](t.dv(o), t.dv(a), da.p01(), da.p23(), Int64(p), grid_dim=_grid(od.size()), block_dim=TPB)
        else:
            c.enqueue_function[_red_kernel[RED_MAX]](t.dv(o), t.dv(a), da.p01(), da.p23(), Int64(p), grid_dim=_grid(od.size()), block_dim=TPB)
        return o
    var po = t.val_ptr(o)
    var pa = t.val_ptr(a)
    var sa = da.strides()
    var n = da.at(p)
    var k = 0
    for x0 in range(od.a):
        for x1 in range(od.b):
            for x2 in range(od.c):
                for x3 in range(od.d):
                    var base = x0 * sa.a + x1 * sa.b + x2 * sa.c + x3 * sa.d
                    var s: Scalar[DT] = pa[unsafe_offset=base] if red == RED_MAX else 0
                    for j in range(n):
                        var v = pa[unsafe_offset=base + j * sa.at(p)]
                        if red == RED_MAX:
                            if v > s:
                                s = v
                        else:
                            s += v
                    if red == RED_MEAN:
                        s = s / Scalar[DT](n)
                    po[unsafe_offset=k] = s
                    k += 1
    return o


def _reduce_bwd(mut t: TapeData, o: Int) raises:
    var a = t.nodes[o].a
    if not t.nodes[a].rg:
        return
    var red = t.nodes[o].sub
    var p = t.nodes[o].i0
    var da = _dims(t, a)
    if t.ctx:
        var c = t.dctx()
        if red == RED_MAX:
            _zero_grad_if_fresh(t, a)
            c.enqueue_function[_red_bwd_max_kernel](
                t.dg(a), t.dg(o), t.dv(a), da.p01(), da.p23(), Int64(p),
                grid_dim=_grid(da.with_(p, 1).size()), block_dim=TPB,
            )
            return
        var acc = t._take(a)
        if red == RED_MEAN:
            if acc:
                c.enqueue_function[_red_bwd_sum_kernel[True, True]](t.dg(a), t.dg(o), da.p01(), da.p23(), Int64(p), grid_dim=_grid(da.size()), block_dim=TPB)
            else:
                c.enqueue_function[_red_bwd_sum_kernel[True, False]](t.dg(a), t.dg(o), da.p01(), da.p23(), Int64(p), grid_dim=_grid(da.size()), block_dim=TPB)
        else:
            if acc:
                c.enqueue_function[_red_bwd_sum_kernel[False, True]](t.dg(a), t.dg(o), da.p01(), da.p23(), Int64(p), grid_dim=_grid(da.size()), block_dim=TPB)
            else:
                c.enqueue_function[_red_bwd_sum_kernel[False, False]](t.dg(a), t.dg(o), da.p01(), da.p23(), Int64(p), grid_dim=_grid(da.size()), block_dim=TPB)
        return
    var ga = t.grad_ptr(a)
    var go = t.grad_ptr(o)
    var pa = t.val_ptr(a)
    var sa = da.strides()
    var n = da.at(p)
    var od = da.with_(p, 1)
    var k = 0
    for x0 in range(od.a):
        for x1 in range(od.b):
            for x2 in range(od.c):
                for x3 in range(od.d):
                    var base = x0 * sa.a + x1 * sa.b + x2 * sa.c + x3 * sa.d
                    var g = go[unsafe_offset=k]
                    if red == RED_MAX:
                        var best = pa[unsafe_offset=base]
                        var arg = 0
                        for j in range(1, n):
                            var v = pa[unsafe_offset=base + j * sa.at(p)]
                            if v > best:
                                best = v
                                arg = j
                        ga[unsafe_offset=base + arg * sa.at(p)] += g
                    else:
                        var gg = g / Scalar[DT](n) if red == RED_MEAN else g
                        for j in range(n):
                            ga[unsafe_offset=base + j * sa.at(p)] += gg
                    k += 1


# ── gather (an embedding lookup) ─────────────────────────────────────────
def _gather_kernel(o: P, table: P, idx: P, n_: Int64, vd: Int64):
    var V = _hi(vd)
    var D = _lo(vd)
    var t = Int(global_idx.x)
    if t >= Int(n_) * D:
        return
    var v = Int(idx[unsafe_offset=t // D])
    if v < 0 or v >= V:
        o[unsafe_offset=t] = Scalar[DT](0) / Scalar[DT](0)  # NaN: out of range
        return
    o[unsafe_offset=t] = table[unsafe_offset=v * D + t % D]


def _gather_bwd_kernel(gt: P, go: P, idx: P, n_: Int64, vd: Int64):
    """One thread per COLUMN d, the rows in index order: no two threads
    write one element, and repeated indices add up deterministically."""
    var V = _hi(vd)
    var D = _lo(vd)
    var d = Int(global_idx.x)
    if d >= D:
        return
    for i in range(Int(n_)):
        var v = Int(idx[unsafe_offset=i])
        if v >= 0 and v < V:
            gt[unsafe_offset=v * D + d] += go[unsafe_offset=i * D + d]


def nd_gather(mut t: TapeData, table: Int, idx: Int) raises -> Int:
    """`table[idx]` along the rows: table `[V, D]`, indices of any shape up
    to rank 3 (integer values held as floats) -> `idx.shape + [D]`."""
    if t.nodes[table].rank != 2:
        raise Error("gather: the table must be rank 2 [V, D]")
    var ri = t.nodes[idx].rank
    if ri > 3:
        raise Error("gather: indices of rank <= 3")
    var V = t.nodes[table].rows
    var D = t.nodes[table].cols
    var N = t.nodes[idx].size()
    var di = _dims(t, idx)
    var od = D4(di.b, di.c, di.d, D)  # idx's dims (rank <= 3: di.a == 1), then D
    var o = _new(t, od, ri + 1, OP_GATHER, table, idx, t.nodes[table].rg)
    if t.ctx:
        var c = t.dctx()
        c.enqueue_function[_gather_kernel](
            t.dv(o), t.dv(table), t.dv(idx), Int64(N), _pk(V, D),
            grid_dim=_grid(N * D), block_dim=TPB,
        )
        return o
    var po = t.val_ptr(o)
    var pt = t.val_ptr(table)
    var pi = t.val_ptr(idx)
    for i in range(N):
        var v = Int(pi[unsafe_offset=i])
        if v < 0 or v >= V or Scalar[DT](v) != pi[unsafe_offset=i]:
            raise Error(String("gather: index ", pi[unsafe_offset=i], " not in [0, ", V, ")"))
        for d in range(D):
            po[unsafe_offset=i * D + d] = pt[unsafe_offset=v * D + d]
    return o


def _gather_bwd(mut t: TapeData, o: Int) raises:
    var table = t.nodes[o].a
    var idx = t.nodes[o].b
    if not t.nodes[table].rg:
        return
    var V = t.nodes[table].rows
    var D = t.nodes[table].cols
    var N = t.nodes[idx].size()
    if t.ctx:
        _zero_grad_if_fresh(t, table)
        t.dctx().enqueue_function[_gather_bwd_kernel](
            t.dg(table), t.dg(o), t.dv(idx), Int64(N), _pk(V, D),
            grid_dim=_grid(D), block_dim=TPB,
        )
        return
    var gt = t.grad_ptr(table)
    var go = t.grad_ptr(o)
    var pi = t.val_ptr(idx)
    for i in range(N):
        var v = Int(pi[unsafe_offset=i])
        for d in range(D):
            gt[unsafe_offset=v * D + d] += go[unsafe_offset=i * D + d]


# ── backward dispatch for the N-d ops ────────────────────────────────────
def nd_vjp(mut t: TapeData, o: Int) raises:
    var op = t.nodes[o].op
    var a = t.nodes[o].a
    if op >= OP_CLAMP:
        extra_vjp(t, o)
    elif op == OP_RESHAPE:
        if t.nodes[a].rg:
            var n = t.nodes[a].size()
            _copy(t, a, True, o, True, D4(1, 1, 1, n), D4(0, 0, 0, 1), 0,
                  D4(0, 0, 0, 1), 0, t._take(a))
    elif op == OP_PERMUTE:
        if t.nodes[a].rg:
            var od = _dims(t, o)
            var sa = _dims(t, a).strides()
            ref nd = t.nodes[o]
            var src = D4(nd.i0, nd.i1, nd.i2, nd.i3)
            var ss = D4(sa.at(src.a), sa.at(src.b), sa.at(src.c), sa.at(src.d))
            # Each input element is ONE output element: write or add.
            _copy(t, a, True, o, True, od, ss, 0, od.strides(), 0, t._take(a))
    elif op == OP_NARROW:
        if t.nodes[a].rg:
            var p = t.nodes[o].i0
            var start = t.nodes[o].i1
            var da = _dims(t, a)
            var od = _dims(t, o)
            _zero_grad_if_fresh(t, a)
            _copy(t, a, True, o, True, od, da.strides(),
                  start * da.strides().at(p), od.strides(), 0, True)
    elif op == OP_CONCAT:
        var b = t.nodes[o].b
        var p = t.nodes[o].i0
        var da = _dims(t, a)
        var db = _dims(t, b)
        var so = _dims(t, o).strides()
        if t.nodes[a].rg:
            _copy(t, a, True, o, True, da, da.strides(), 0, so, 0, t._take(a))
        if t.nodes[b].rg:
            _copy(t, b, True, o, True, db, db.strides(), 0, so,
                  da.at(p) * so.at(p), t._take(b))
    elif op == OP_REDUCE_AXIS:
        _reduce_bwd(t, o)
    elif op == OP_GATHER:
        _gather_bwd(t, o)
    else:
        raise Error(String("nd backward: unknown op ", op))
