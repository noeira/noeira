# +--------------------------------------------------------------------------+ #
# | LFM3 int8 weights — per-row quantization, the decode GEMV, the dequantization
# +--------------------------------------------------------------------------+ #
"""Weight-only int8 for the LFM3 backbone (docs/NN_MODEL_PORTS_PLAN.md §12.7, int8):
each linear's weight `W [n, k]` is stored as `q [n, k]` int8 and `scale [n]`
fp32, `W[r, :] ~ scale[r] * q[r, :]`, `scale[r] = max |W[r, :]| / 127`
(symmetric, round to nearest, the bf16 weights quantized once on the device).

Decoding reads every weight once a token, so the bytes are the time (the Orin's
~100 GB/s): int8 halves them. `q8_gemv_kernel` serves a few rows of
activations (decoding: one); a longer chunk dequantizes the weight to bf16 /
fp32 and takes the GEMM (`q8_dequant_*_kernel`).
"""

from std.math import round, sqrt
from std.memory import stack_allocation
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block, warp

from noeira.nn.constants import DT
from .kernels import TPB, Ptr, HPtr, BPtr, IPtr, _bf, _sigmoid


comptime QPtr = Pointer[Scalar[DType.int8], MutAnyOrigin]
comptime Q8_LANE: Int = 16
"""int8 weights one lane loads at a time (16 bytes); a warp covers 512 of a row."""
comptime Q8_GEMV_MAX: Int = 4
"""Most activation rows the GEMV serves; more take the dequantize + GEMM route."""


def q8_quantize_kernel(w: HPtr, q: QPtr, scale: Ptr, k_n: Int64, levels_n: Int64):
    """Row `r` of the bf16 `w [n, k]`: `scale[r] = max |w| / L` and `q = round(w
    / scale)` (clamped to +-L; a zero row: scale 0, q 0), L = `levels` = 127
    (int8; the gate's mutant passes 7, a 4-bit grid). A block per row."""
    var r = Int(block_idx.x)
    var k = Int(k_n)
    var tid = Int(thread_idx.x)
    var base = r * k
    var m = Scalar[DT](0)
    var j = tid
    while j < k:
        m = max(m, abs(_bf(w[unsafe_offset=base + j])))
        j += TPB
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var lv = Scalar[DT](Int(levels_n))
    var s = top / lv
    if tid == 0:
        scale[unsafe_offset=r] = s
    var inv = Scalar[DT](1) / s if s > Scalar[DT](0) else Scalar[DT](0)
    j = tid
    while j < k:
        var v = round(_bf(w[unsafe_offset=base + j]) * inv)
        v = min(max(v, -lv), lv)
        q[unsafe_offset=base + j] = v.cast[DType.int8]()
        j += TPB


comptime Q8_RPW: Int = 4
"""Weight rows a warp computes: the activations it loads serve all of them
(one load of x per 4 rows of int8, not per row). Every quantized `n` is a
multiple of 4 (checked by `quantize_int8`)."""


def q8_gemv_kernel[NS: Int](x: Ptr, q: QPtr, scale: Ptr, y: Ptr, n_n: Int64, k_n: Int64):
    """`y[s, r] = scale[r] * sum_j q[r, j] x[s, j]` for the `NS` rows of `x [NS, k]`
    (exactly NS: no padding is read). A warp per Q8_RPW weight rows, 8 warps a
    block; each lane takes 16 int8 of each row (one 16-byte load) against the
    same 16 activations; `k` a multiple of 512. Grid `ceil(n / (8 Q8_RPW))`."""
    var tid = Int(thread_idx.x)
    var row0 = (Int(block_idx.x) * 8 + (tid >> 5)) * Q8_RPW
    var lane = tid & 31
    var n = Int(n_n)
    var k = Int(k_n)
    var acc = SIMD[DT, Q8_RPW * NS](0)
    if row0 < n:
        var j = lane * Q8_LANE
        while j < k:
            var xs = SIMD[DT, Q8_LANE * NS](0)  # the NS activation rows' 16 values, side by side
            comptime for c in range(NS):
                xs = xs.insert[offset=c * Q8_LANE](x.unsafe_load[width=Q8_LANE, alignment=16](c * k + j))
            comptime for r in range(Q8_RPW):
                var wv = q.unsafe_load[width=Q8_LANE, alignment=16]((row0 + r) * k + j).cast[DT]()
                comptime for c in range(NS):
                    acc[r * NS + c] += (wv * xs.slice[Q8_LANE, offset=c * Q8_LANE]()).reduce_add()
            j += 32 * Q8_LANE
    comptime for r in range(Q8_RPW):
        comptime for c in range(NS):
            var tot = warp.sum(acc[r * NS + c])
            if lane == 0 and row0 < n:
                y[unsafe_offset=c * n + row0 + r] = tot * scale[unsafe_offset=row0 + r]


def q8_dequant_bf16_kernel(q: QPtr, scale: Ptr, dst: BPtr, k_n: Int64):
    """`dst[r, :] = scale[r] * q[r, :]` in bf16 (a chunk's GEMM operand), 16
    weights a thread. Grid `(ceil(k / (16 TPB)), n)`."""
    var r = Int(block_idx.y)
    var k = Int(k_n)
    var j = (Int(block_idx.x) * TPB + Int(thread_idx.x)) * Q8_LANE
    if j < k:
        var v = q.unsafe_load[width=Q8_LANE, alignment=16](r * k + j).cast[DT]() * scale[unsafe_offset=r]
        dst.unsafe_store[alignment=16](r * k + j, v.cast[DType.bfloat16]())


def q8_dequant_f32_kernel(q: QPtr, scale: Ptr, dst: Ptr, k_n: Int64):
    """`dst[r, :] = scale[r] * q[r, :]` in fp32, 16 weights a thread. Grid `(ceil(k / (16 TPB)), n)`."""
    var r = Int(block_idx.y)
    var k = Int(k_n)
    var j = (Int(block_idx.x) * TPB + Int(thread_idx.x)) * Q8_LANE
    if j < k:
        var v = q.unsafe_load[width=Q8_LANE, alignment=16](r * k + j).cast[DT]() * scale[unsafe_offset=r]
        dst.unsafe_store[alignment=16](r * k + j, v)


def q8_embed_kernel(ids: IPtr, q: QPtr, scale: Ptr, dst: Ptr, d_n: Int64):
    """`dst[t, :] = scale[id] * q[id, :]`, `id = ids[t]`: the embedding lookup
    from the int8 table (a model loaded int8 keeps no bf16 copy of it).
    Grid `(ceil(D / TPB), T)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    var dim = Int(d_n)
    if d < dim:
        var id = Int(ids[unsafe_offset=t])
        dst[unsafe_offset=t * dim + d] = q[unsafe_offset=id * dim + d].cast[DT]() * scale[unsafe_offset=id]


# ── W8A8: int8 activations for the int8 tensor-core GEMM ─────────────────────
#
# A chunk's linear as `y = sx[t] sw[n] * (xq[t, :] . wq[n, :])`: the weights'
# per-row int8 (above), the activations quantized per token (one scale a row,
# `max |x[t, :]| / 127`, symmetric, round to nearest), the exact int32 sums from
# `cublas_gemm_i8`, then both scales. docs/NN_MODEL_PORTS_PLAN.md §12.8.8.


def q8_act_kernel(x: Ptr, q: QPtr, scale: Ptr, k_n: Int64):
    """Row `t` of the fp32 `x [T, k]` to int8: `scale[t] = max |x[t, :]| / 127`,
    `q = round(x / scale)` (a zero row: scale 0, q 0). A block per row."""
    var r = Int(block_idx.x)
    var k = Int(k_n)
    var tid = Int(thread_idx.x)
    var base = r * k
    var m = Scalar[DT](0)
    var j = tid
    while j < k:
        m = max(m, abs(x[unsafe_offset=base + j]))
        j += TPB
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var s = top / Scalar[DT](127)
    if tid == 0:
        scale[unsafe_offset=r] = s
    var inv = Scalar[DT](1) / s if s > Scalar[DT](0) else Scalar[DT](0)
    j = tid
    while j < k:
        var v = min(max(round(x[unsafe_offset=base + j] * inv), Scalar[DT](-127)), Scalar[DT](127))
        q[unsafe_offset=base + j] = v.cast[DType.int8]()
        j += TPB


def q8_rms_kernel(x: Ptr, y: Ptr, q: QPtr, scale: Ptr, w: HPtr, dim_n: Int64, eps: Float32):
    """`rms_kernel` (`y = w * x / rms(x)`), and `y`'s row quantized as
    `q8_act_kernel` does, in the same block: the norm's output is the operand
    of the GEMMs after it (q / k / v / gate, the conv's in_proj, w1 / w3)."""
    var dim = Int(dim_n)
    var row = Int(block_idx.x) * dim
    var tid = Int(thread_idx.x)
    var s = Scalar[DT](0)
    var d = tid
    while d < dim:
        var v = x[unsafe_offset=row + d]
        s += v * v
        d += TPB
    var tot = block.sum[block_size=TPB, broadcast=True](val=s)
    var inv = Scalar[DT](1) / sqrt(tot / Scalar[DT](dim) + Scalar[DT](eps))
    var m = Scalar[DT](0)
    d = tid
    while d < dim:
        var o = _bf(w[unsafe_offset=d]) * (x[unsafe_offset=row + d] * inv)
        y[unsafe_offset=row + d] = o
        m = max(m, abs(o))
        d += TPB
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var sc = top / Scalar[DT](127)
    if tid == 0:
        scale[unsafe_offset=Int(block_idx.x)] = sc
    var qi = Scalar[DT](1) / sc if sc > Scalar[DT](0) else Scalar[DT](0)
    d = tid
    while d < dim:
        var v = min(max(round(y[unsafe_offset=row + d] * qi), Scalar[DT](-127)), Scalar[DT](127))
        q[unsafe_offset=row + d] = v.cast[DType.int8]()
        d += TPB


comptime Q8_OUT_W: Int = 4
"""Outputs a thread of `q8_out_kernel` writes (one 16-byte load and store; `n` a multiple of 4)."""


def q8_out_kernel(acc: IPtr, sx: Ptr, sw: Ptr, y: Ptr, n_n: Int64):
    """`y[t, c] = acc[t, c] * sx[t] * sw[c]`: the int32 sums of `cublas_gemm_i8`
    back to fp32 with the token's and the weight row's scales. Grid
    `(ceil(n / (Q8_OUT_W TPB)), T)`."""
    var t = Int(block_idx.y)
    var n = Int(n_n)
    var c = (Int(block_idx.x) * TPB + Int(thread_idx.x)) * Q8_OUT_W
    if c < n:
        var a = acc.unsafe_load[width=Q8_OUT_W, alignment=16](t * n + c).cast[DT]()
        var w = sw.unsafe_load[width=Q8_OUT_W, alignment=16](c)
        y.unsafe_store[alignment=16](t * n + c, a * sx[unsafe_offset=t] * w)


comptime Q8_FF_MAX: Int = 8192
"""Widest MLP row `q8_swiglu_kernel` keeps in shared memory (32 KB; LFM3's is 7680)."""


def q8_swiglu_kernel(a1: IPtr, a3: IPtr, sx: Ptr, s1: Ptr, s3: Ptr, q: QPtr, qs: Ptr, ff_n: Int64):
    """Row `t` of the MLP between its GEMMs, on the int8 path: the w1 / w3
    int32 sums scaled back (`a sx[t] s[c]`, as `q8_out_kernel`), `h = silu(a1)
    * a3` (as `swiglu_kernel`), and `h` quantized as `q8_act_kernel` does — the
    w2 GEMM's operand. `h` waits in shared memory, never in fp32 in device
    memory. A block per row; `ff` <= Q8_FF_MAX and a multiple of 4."""
    var t = Int(block_idx.x)
    var ff = Int(ff_n)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[Q8_FF_MAX, Scalar[DT], address_space=AddressSpace.SHARED]()
    var x = sx[unsafe_offset=t]
    var m = Scalar[DT](0)
    var j = tid * Q8_OUT_W
    while j < ff:
        var u = a1.unsafe_load[width=Q8_OUT_W, alignment=16](t * ff + j).cast[DT]() * x * s1.unsafe_load[width=Q8_OUT_W, alignment=16](j)
        var v = a3.unsafe_load[width=Q8_OUT_W, alignment=16](t * ff + j).cast[DT]() * x * s3.unsafe_load[width=Q8_OUT_W, alignment=16](j)
        comptime for e in range(Q8_OUT_W):
            var h = u[e] * _sigmoid(u[e]) * v[e]
            sh[j + e] = h
            m = max(m, abs(h))
        j += TPB * Q8_OUT_W
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var sc = top / Scalar[DT](127)
    if tid == 0:
        qs[unsafe_offset=t] = sc
    var inv = Scalar[DT](1) / sc if sc > Scalar[DT](0) else Scalar[DT](0)
    j = tid * Q8_OUT_W
    while j < ff:
        comptime for e in range(Q8_OUT_W):
            var v = min(max(round(sh[j + e] * inv), Scalar[DT](-127)), Scalar[DT](127))
            q[unsafe_offset=t * ff + j + e] = v.cast[DType.int8]()
        j += TPB * Q8_OUT_W
