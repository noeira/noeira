"""Causal multi-head attention as two MAX custom ops, for a kernel-backed
rule: noeira's fused (flash) attention on the GPU, plain loops on the CPU.

- `noeira_attention_fwd[B, H, S, HD]`: `qkv [B, S, 3·H·HD]`, the QKV
  projection's own layout (token i holds its q, k and v, heads interleaved)
  -> `o [B, S, H·HD]` (heads merged) **and the residual** `lse [B·H·S]`, each
  query row's log-sum-exp.
- `noeira_attention_bwd[B, H, S, HD]`: `do`, `qkv`, `o`, `lse` -> `dqkv` (the
  input's layout, every element written) and `dvec [B·H·S]` (D = Σ dO·O).
  `dvec` is the backward's workspace, returned as an output so that the
  kernels allocate nothing (the graph owns it).

Every size is a compile-time parameter (`ops.custom(parameters=...)`), as
noeira's kernels take them. The scale is 1/sqrt(HD). The GPU path enqueues noeira's FlashAttention-2
kernels (`flash_attention.mojo`; float32, `lse` in log2 units). The CPU path
is O(S²) loops in any dtype (natural log), for float64 gradchecks. A graph
uses one path, whose forward and backward agree.
"""

import extensibility

from extensibility import InputTensor, OutputTensor
from std.ffi import external_call
from max.algorithm import sync_parallelize
from max.gpu import WARP_SIZE
from max.gpu.host import DeviceContext
from std.math import ceildiv, exp, log, sqrt

from .flash_attention import (
    DQ_TK,
    DQ_TQ,
    DT,
    FWD_TK,
    FWD_TQ,
    KV_TK,
    KV_TQ,
    TPB,
    _flash_bwd_dkdv_kernel,
    _flash_bwd_dq_kernel,
    _flash_d_kernel,
    _flash_fwd_kernel,
)

comptime F32Ptr = Pointer[Scalar[DT], MutAnyOrigin]


# The CPU path is the float64 reference for gradchecks, so it needs exact
# transcendentals: Mojo 1.1's float64 `log` is off by up to 5e-10 (at 0.731)
# and `exp` by 3e-12 relative, against libm's correctly rounded results. In
# float64 they come from libm.


@always_inline
def _exp[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime if dtype == DType.float64:
        return rebind[Scalar[dtype]](external_call["exp", Float64](rebind[Float64](x)))
    else:
        return exp(x)


@always_inline
def _log[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime if dtype == DType.float64:
        return rebind[Scalar[dtype]](external_call["log", Float64](rebind[Float64](x)))
    else:
        return log(x)


@extensibility.register("noeira_attention_fwd")
struct AttentionFwd:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        B: Int,
        H: Int,
        S: Int,
        HD: Int,
        target: StaticString,
    ](
        o: OutputTensor[dtype=dtype, rank=3, ...],
        lse: OutputTensor[dtype=dtype, rank=1, ...],
        qkv: InputTensor[dtype=dtype, rank=3, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        comptime batch = B
        comptime D = H * HD
        comptime if target == "gpu":
            comptime assert dtype == DT, "the GPU path is float32"
            var out = rebind[F32Ptr](o.unsafe_ptr())
            ctx.enqueue_function[_flash_fwd_kernel[DT, H, S, HD, True, FWD_TQ, FWD_TK, True]](
                rebind[F32Ptr](qkv.unsafe_ptr()), out, out, rebind[F32Ptr](lse.unsafe_ptr()),
                grid_dim=(ceildiv(S, FWD_TQ), batch * H),
                block_dim=FWD_TQ * FWD_TK // 16,
            )
        else:
            var scale = Scalar[dtype](1.0 / sqrt(Float64(HD)))

            def head(work: Int) {imm}:
                var b = work // H
                var h = work - b * H
                var q0 = h * HD
                var k0 = D + h * HD
                var v0 = 2 * D + h * HD
                var p = List[Scalar[dtype]](length=S, fill=0)
                for i in range(S):
                    var m = Scalar[dtype](-1e300) if dtype == DType.float64 else Scalar[dtype](-1e30)
                    for j in range(i + 1):
                        var dot = Scalar[dtype](0)
                        for d in range(HD):
                            dot += qkv[b, i, q0 + d] * qkv[b, j, k0 + d]
                        p[j] = dot * scale
                        m = max(m, p[j])
                    var l = Scalar[dtype](0)
                    for j in range(i + 1):
                        p[j] = _exp(p[j] - m)
                        l += p[j]
                    for d in range(HD):
                        var acc = Scalar[dtype](0)
                        for j in range(i + 1):
                            acc += p[j] * qkv[b, j, v0 + d]
                        o[b, i, q0 + d] = acc / l
                    lse[work * S + i] = m + _log(l)

            sync_parallelize(head, batch * H)


@extensibility.register("noeira_attention_bwd")
struct AttentionBwd:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        B: Int,
        H: Int,
        S: Int,
        HD: Int,
        target: StaticString,
    ](
        dqkv: OutputTensor[dtype=dtype, rank=3, ...],
        dvec: OutputTensor[dtype=dtype, rank=1, ...],
        do: InputTensor[dtype=dtype, rank=3, ...],
        qkv: InputTensor[dtype=dtype, rank=3, ...],
        o: InputTensor[dtype=dtype, rank=3, ...],
        lse: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        comptime batch = B
        comptime D = H * HD
        comptime if target == "gpu":
            comptime assert dtype == DT, "the GPU path is float32"
            var inp = rebind[F32Ptr](qkv.unsafe_ptr())
            var dout = rebind[F32Ptr](do.unsafe_ptr())
            var l = rebind[F32Ptr](lse.unsafe_ptr())
            var dv = rebind[F32Ptr](dvec.unsafe_ptr())
            var gin = rebind[F32Ptr](dqkv.unsafe_ptr())
            comptime rows = B * H * S
            ctx.enqueue_function[_flash_d_kernel[DT, H, S, HD, rows]](
                dout, rebind[F32Ptr](o.unsafe_ptr()), dv,
                grid_dim=ceildiv(rows * WARP_SIZE, TPB),
                block_dim=TPB,
            )
            ctx.enqueue_function[_flash_bwd_dq_kernel[DT, H, S, HD, True, DQ_TQ, DQ_TK, True]](
                inp, dout, l, dv, gin,
                grid_dim=(ceildiv(S, DQ_TQ), batch * H),
                block_dim=DQ_TQ * DQ_TK // 16,
            )
            ctx.enqueue_function[_flash_bwd_dkdv_kernel[DT, H, S, HD, True, KV_TQ, KV_TK, True]](
                inp, dout, l, dv, gin,
                grid_dim=(ceildiv(S, KV_TK), batch * H),
                block_dim=KV_TQ * KV_TK // 16,
            )
        else:
            var scale = Scalar[dtype](1.0 / sqrt(Float64(HD)))

            def head(work: Int) {imm}:
                var b = work // H
                var h = work - b * H
                var q0 = h * HD
                var k0 = D + h * HD
                var v0 = 2 * D + h * HD
                # dq, dk, dv of this head, [S, HD] each, summed here and
                # written once (an output tensor cannot be read back).
                var g = List[Scalar[dtype]](length=3 * S * HD, fill=0)
                var dd = List[Scalar[dtype]](length=S, fill=0)
                for i in range(S):
                    var s = Scalar[dtype](0)
                    for d in range(HD):
                        s += do[b, i, q0 + d] * o[b, i, q0 + d]
                    dd[i] = s
                    dvec[work * S + i] = s
                for i in range(S):
                    var li = lse[work * S + i]
                    for j in range(i + 1):
                        var dot = Scalar[dtype](0)
                        var dp = Scalar[dtype](0)
                        for d in range(HD):
                            dot += qkv[b, i, q0 + d] * qkv[b, j, k0 + d]
                            dp += do[b, i, q0 + d] * qkv[b, j, v0 + d]
                        var pij = _exp(dot * scale - li)
                        var ds = pij * (dp - dd[i]) * scale
                        for d in range(HD):
                            g[i * HD + d] += ds * qkv[b, j, k0 + d]
                            g[S * HD + j * HD + d] += ds * qkv[b, i, q0 + d]
                            g[2 * S * HD + j * HD + d] += pij * do[b, i, q0 + d]
                for i in range(S):
                    for d in range(HD):
                        dqkv[b, i, q0 + d] = g[i * HD + d]
                        dqkv[b, i, k0 + d] = g[S * HD + i * HD + d]
                        dqkv[b, i, v0 + d] = g[2 * S * HD + i * HD + d]

            sync_parallelize(head, batch * H)
