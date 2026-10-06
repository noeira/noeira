"""LayerNorm as two MAX custom ops, for the kernel-backed rule (`rules/custom.py`).

- `noeira_layer_norm_fwd`: `x [R, D]`, `gamma [D]`, `beta [D]`, `eps [1]`
  -> `y [R, D]` **and the residuals** `mean [R, 1]`, `rstd [R, 1]`.
- `noeira_layer_norm_bwd`: `dy`, `x`, `gamma`, `mean`, `rstd`
  -> `dx [R, D]`, and `dgamma` and `dbeta` as partial sums `[P, D]` over P
  chunks of rows: the graph's own `sum` finishes them, so the kernel needs
  no scratch memory (nothing to allocate inside a captured graph) and the
  result does not depend on thread timing.

The forward returns what its backward needs, which MAX's own
`mo.reduce.layer_norm` does not, so its composite rule recomputes the mean
and the variance. `rules/custom.py` registers the pairing: the VJP of the
forward op is a call to the backward op on the forward's residuals.

Rows are the leading dims flattened by the caller (`layer_norm_kernel` in
`models/common.py`). On the CPU, rows run in parallel, SIMD within a row.
On a GPU, one block per row reduces in shared memory, and the partials are
one thread per (column, chunk).
"""

import extensibility

from extensibility import InputTensor, OutputTensor
from max.algorithm import parallelize_over_rows, sync_parallelize
from max.gpu import block_dim, block_idx, global_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from std.math import ceildiv, sqrt
from std.memory.alloc import unsafe_alloc
from std.sys import simd_width_of, size_of
from std.utils.index import IndexList

comptime BLOCK = 128
"""GPU threads per row (a power of two: the shared-memory tree reduction).
Sizes reach the kernels as `Int32`: `Int` is not device-passable."""


@always_inline
def _block_sum[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    """The sum of `value` over the block's `BLOCK` threads, in every thread
    (a tree reduction in the dynamic shared memory the launch provides)."""
    var shared = external_memory[
        Scalar[dtype], address_space=.SHARED, alignment=size_of[Scalar[dtype]]()
    ]()
    var tid = Int(thread_idx.x)
    shared[unsafe_offset=tid] = value
    barrier()
    var stride = BLOCK // 2
    while stride > 0:
        if tid < stride:
            shared[unsafe_offset=tid] = shared[unsafe_offset=tid] + shared[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    var total = shared[unsafe_offset=0]
    barrier()
    return total


@extensibility.register("noeira_layer_norm_fwd")
struct LayerNormFwd:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        y: OutputTensor[dtype=dtype, rank=2, ...],
        mean: OutputTensor[dtype=dtype, rank=2, ...],
        rstd: OutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        beta: InputTensor[dtype=dtype, rank=1, ...],
        eps: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var rows = x.shape()[0]
        var d = x.shape()[1]
        comptime if target == "gpu":
            var y_t = y.to_layout_tensor()
            var mean_t = mean.to_layout_tensor()
            var rstd_t = rstd.to_layout_tensor()
            var x_t = x.to_layout_tensor()
            var gamma_t = gamma.to_layout_tensor()
            var beta_t = beta.to_layout_tensor()
            var eps_t = eps.to_layout_tensor()

            @__parameter
            def fwd_gpu(
                y_t: type_of(y_t),
                mean_t: type_of(mean_t),
                rstd_t: type_of(rstd_t),
                x_t: type_of(x_t),
                gamma_t: type_of(gamma_t),
                beta_t: type_of(beta_t),
                eps_t: type_of(eps_t),
                d32: Int32,
            ):
                var d = Int(d32)
                var r = Int(block_idx.x)
                var tid = Int(thread_idx.x)
                var s = Scalar[dtype](0)
                for c in range(tid, d, BLOCK):
                    s += rebind[Scalar[dtype]](x_t[r, c])
                var m = _block_sum[dtype](s) / Scalar[dtype](d)
                var v = Scalar[dtype](0)
                for c in range(tid, d, BLOCK):
                    var t = rebind[Scalar[dtype]](x_t[r, c]) - m
                    v += t * t
                var rs = 1 / sqrt(_block_sum[dtype](v) / Scalar[dtype](d) + rebind[Scalar[dtype]](eps_t[0]))
                if tid == 0:
                    mean_t[r, 0] = m
                    rstd_t[r, 0] = rs
                for c in range(tid, d, BLOCK):
                    y_t[r, c] = (rebind[Scalar[dtype]](x_t[r, c]) - m) * rs * rebind[Scalar[dtype]](gamma_t[c]) + rebind[Scalar[dtype]](beta_t[c])

            ctx.enqueue_function[fwd_gpu](
                y_t, mean_t, rstd_t, x_t, gamma_t, beta_t, eps_t, Int32(d),
                grid_dim=rows,
                block_dim=BLOCK,
                shared_mem_bytes=BLOCK * size_of[Scalar[dtype]](),
            )
        else:
            comptime W = simd_width_of[dtype]()
            var e = eps[0]

            def fwd_rows(start: Int, end: Int) {imm}:
                for r in range(start, end):
                    var acc = SIMD[dtype, W](0)
                    var c = 0
                    while c + W <= d:
                        acc += x.load[W](IndexList[2](r, c))
                        c += W
                    var s = acc.reduce_add()
                    for t in range(c, d):
                        s += x[r, t]
                    var m = s / Scalar[dtype](d)
                    var vacc = SIMD[dtype, W](0)
                    c = 0
                    while c + W <= d:
                        var t = x.load[W](IndexList[2](r, c)) - m
                        vacc += t * t
                        c += W
                    var v = vacc.reduce_add()
                    for t in range(c, d):
                        var u = x[r, t] - m
                        v += u * u
                    var rs = 1 / sqrt(v / Scalar[dtype](d) + e)
                    mean[r, 0] = m
                    rstd[r, 0] = rs
                    c = 0
                    while c + W <= d:
                        var value = (x.load[W](IndexList[2](r, c)) - m) * rs * gamma.load[W](
                            IndexList[1](c)
                        ) + beta.load[W](IndexList[1](c))
                        y.store[W](IndexList[2](r, c), value)
                        c += W
                    for t in range(c, d):
                        y[r, t] = (x[r, t] - m) * rs * gamma[t] + beta[t]

            parallelize_over_rows(fwd_rows, x.shape(), axis=1, grain_size=1)


@extensibility.register("noeira_layer_norm_bwd")
struct LayerNormBwd:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        dx: OutputTensor[dtype=dtype, rank=2, ...],
        dgamma: OutputTensor[dtype=dtype, rank=2, ...],
        dbeta: OutputTensor[dtype=dtype, rank=2, ...],
        dy: InputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        mean: InputTensor[dtype=dtype, rank=2, ...],
        rstd: InputTensor[dtype=dtype, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        var rows = x.shape()[0]
        var d = x.shape()[1]
        var chunks = dgamma.shape()[0]
        comptime if target == "gpu":
            var dx_t = dx.to_layout_tensor()
            var dgamma_t = dgamma.to_layout_tensor()
            var dbeta_t = dbeta.to_layout_tensor()
            var dy_t = dy.to_layout_tensor()
            var x_t = x.to_layout_tensor()
            var gamma_t = gamma.to_layout_tensor()
            var mean_t = mean.to_layout_tensor()
            var rstd_t = rstd.to_layout_tensor()

            @__parameter
            def dx_gpu(
                dx_t: type_of(dx_t),
                dy_t: type_of(dy_t),
                x_t: type_of(x_t),
                gamma_t: type_of(gamma_t),
                mean_t: type_of(mean_t),
                rstd_t: type_of(rstd_t),
                d32: Int32,
            ):
                var d = Int(d32)
                var r = Int(block_idx.x)
                var tid = Int(thread_idx.x)
                var m = rebind[Scalar[dtype]](mean_t[r, 0])
                var rs = rebind[Scalar[dtype]](rstd_t[r, 0])
                var a = Scalar[dtype](0)
                var b = Scalar[dtype](0)
                for c in range(tid, d, BLOCK):
                    var g = rebind[Scalar[dtype]](dy_t[r, c]) * rebind[Scalar[dtype]](gamma_t[c])
                    a += g
                    b += g * (rebind[Scalar[dtype]](x_t[r, c]) - m) * rs
                var ma = _block_sum[dtype](a) / Scalar[dtype](d)
                var mb = _block_sum[dtype](b) / Scalar[dtype](d)
                for c in range(tid, d, BLOCK):
                    var g = rebind[Scalar[dtype]](dy_t[r, c]) * rebind[Scalar[dtype]](gamma_t[c])
                    var xh = (rebind[Scalar[dtype]](x_t[r, c]) - m) * rs
                    dx_t[r, c] = rs * (g - ma - xh * mb)

            @__parameter
            def dparams_gpu(
                dgamma_t: type_of(dgamma_t),
                dbeta_t: type_of(dbeta_t),
                dy_t: type_of(dy_t),
                x_t: type_of(x_t),
                mean_t: type_of(mean_t),
                rstd_t: type_of(rstd_t),
                rows32: Int32,
                d32: Int32,
                chunks32: Int32,
            ):
                var rows = Int(rows32)
                var d = Int(d32)
                var chunks = Int(chunks32)
                var c = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
                var k = Int(block_idx.y)
                if c >= d:
                    return
                var sg = Scalar[dtype](0)
                var sb = Scalar[dtype](0)
                for r in range(k * rows // chunks, (k + 1) * rows // chunks):
                    var g = rebind[Scalar[dtype]](dy_t[r, c])
                    sg += g * (rebind[Scalar[dtype]](x_t[r, c]) - rebind[Scalar[dtype]](mean_t[r, 0])) * rebind[Scalar[dtype]](rstd_t[r, 0])
                    sb += g
                dgamma_t[k, c] = sg
                dbeta_t[k, c] = sb

            ctx.enqueue_function[dx_gpu](
                dx_t, dy_t, x_t, gamma_t, mean_t, rstd_t, Int32(d),
                grid_dim=rows,
                block_dim=BLOCK,
                shared_mem_bytes=BLOCK * size_of[Scalar[dtype]](),
            )
            ctx.enqueue_function[dparams_gpu](
                dgamma_t, dbeta_t, dy_t, x_t, mean_t, rstd_t, Int32(rows), Int32(d), Int32(chunks),
                grid_dim=(ceildiv(d, BLOCK), chunks),
                block_dim=BLOCK,
            )
        else:
            comptime W = simd_width_of[dtype]()
            var partial = unsafe_alloc[Scalar[dtype]](2 * chunks * d)
            for i in range(2 * chunks * d):
                partial[unsafe_offset=i] = 0

            def bwd_chunk(k: Int) {imm}:
                var first = k * rows // chunks
                var last = (k + 1) * rows // chunks
                var pg = partial + 2 * k * d
                var pb = pg + d
                for r in range(first, last):
                    var m = mean[r, 0]
                    var rs = rstd[r, 0]
                    var av = SIMD[dtype, W](0)
                    var bv = SIMD[dtype, W](0)
                    var c = 0
                    while c + W <= d:
                        var at = IndexList[2](r, c)
                        var g = dy.load[W](at)
                        var xh = (x.load[W](at) - m) * rs
                        var gg = g * gamma.load[W](IndexList[1](c))
                        av += gg
                        bv += gg * xh
                        (pg + c).store(0, (pg + c).load[width=W](0) + g * xh)
                        (pb + c).store(0, (pb + c).load[width=W](0) + g)
                        c += W
                    var a = av.reduce_add()
                    var b = bv.reduce_add()
                    for t in range(c, d):
                        var xh = (x[r, t] - m) * rs
                        var gg = dy[r, t] * gamma[t]
                        a += gg
                        b += gg * xh
                        pg[unsafe_offset=t] = pg[unsafe_offset=t] + dy[r, t] * xh
                        pb[unsafe_offset=t] = pb[unsafe_offset=t] + dy[r, t]
                    a /= Scalar[dtype](d)
                    b /= Scalar[dtype](d)
                    c = 0
                    while c + W <= d:
                        var at = IndexList[2](r, c)
                        var xh = (x.load[W](at) - m) * rs
                        var gg = dy.load[W](at) * gamma.load[W](IndexList[1](c))
                        dx.store[W](at, rs * (gg - a - xh * b))
                        c += W
                    for t in range(c, d):
                        var xh = (x[r, t] - m) * rs
                        dx[r, t] = rs * (dy[r, t] * gamma[t] - a - xh * b)

            sync_parallelize(bwd_chunk, chunks)
            for k in range(chunks):
                for c in range(d):
                    dgamma[k, c] = partial[unsafe_offset=2 * k * d + c]
                    dbeta[k, c] = partial[unsafe_offset=2 * k * d + d + c]
            partial.unsafe_free()
