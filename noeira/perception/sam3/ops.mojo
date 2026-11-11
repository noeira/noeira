# +--------------------------------------------------------------------------+ #
# | SAM 3 — weights on the device, the GEMM, and the kernels its stages share
# +--------------------------------------------------------------------------+ #
"""Plumbing for `noeira.perception.sam3` (docs/NN_MODEL_PORTS_PLAN.md, SAM 3).

`Sam3Weights`: every fp32 tensor of the checkpoint under a prefix, one device
buffer each (the checkpoint is fp32: `facebook/sam3`'s `model.safetensors`, the
VIDEO model — the image detector is `detector_model.*`, the tracker is never
loaded). `linear`: `x W^T + b` through `gemm_tb` (cuBLAS on NVIDIA, fp32 or
`-D SAM3_TF32=1`; MAX's matmul elsewhere).

`-D SAM3_BF16=1` (NVIDIA): every `linear` on the bf16 tensor cores — its
activations rounded to bf16, the weight's bf16 copy (made on first use, kept
beside the fp32 one: +1.7 GB), fp32 accumulation and output, as LFM3's bf16
build; the attention's GEMMs and the 3x3 convs stay TF32.
"""

from std.math import sqrt, erf, exp
from std.memory import unsafe_memcpy
from std.os import listdir, getenv
from std.sys import default_accelerator
from std.sys.defines import is_defined
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from max.gpu.primitives import block, warp
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.io.safetensors import SafeTensors, ST_F32, ST_BF16
from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm, cublas_gemm_lp, _shared_workspace, CUBLAS_WS_BYTES
from linalg.matmul.vendor.blas import _get_global_handle, Backend, _ffi_void_ptr
from _cublas.cublas import (
    _get_dylib_function as _cublas_fn, Algorithm, ComputeType, _convert_to_cublas_datatype,
    _convert_to_cublas_transpose, check_cublas_error, cublasMath_t, cublasSetMathMode, cublasSetWorkspace,
)
from noeira.nn.primitives.flash_attention_tf32 import _tf32_fwd_kernel, TF_TQ, TF_WARPS
from noeira.nn.primitives.flash_attention_mma import _mma_fwd_kernel, MMA_TQ, MMA_WARPS
from max.gpu import WARP_SIZE


comptime SAM3_BF16: Bool = is_defined["SAM3_BF16"]() and default_accelerator().is_nvidia_gpu()
comptime SAM3_TF32: Bool = is_defined["SAM3_TF32"]() or SAM3_BF16
"""TF32 for every GEMM `linear` does not take to bf16 (the attention's, the convs')."""
comptime BPtr = Pointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime TPB: Int = 256
comptime BF16_MIN_K: Int = 1024
"""SAM3_BF16: a `linear` on an fp32 input takes the bf16 GEMM only when its
input is at least this wide (the backbone's): at 256 wide the GEMM is
memory-bound and the input's conversion pass costs what bf16 saves, so the
narrow ones stay TF32. bf16 operands produced for free (an activation pass, the
fused attention's output) take the bf16 GEMM at any width (`linear_into_b`)."""
comptime Ptr = Pointer[Scalar[DT], MutAnyOrigin]


def blocks(n: Int) -> Int:
    return (n + TPB - 1) // TPB


@always_inline
def _d(a: Int, b: Int) -> Int:
    """`a // b` for non-negatives below 2^32, in 32-bit arithmetic: a 64-bit
    divide is a software routine on a GPU (the profile's bias kernel spent most
    of its 0.65 s on `i % cols`)."""
    return Int(UInt32(a) // UInt32(b))


def sam3_checkpoint() raises -> String:
    """`SAM3_CHECKPOINT`, else the Hugging Face cache's `facebook/sam3` snapshot."""
    var env = getenv("SAM3_CHECKPOINT")
    if env.byte_length() > 0:
        return env
    var snaps = getenv("HOME") + "/.cache/huggingface/hub/models--facebook--sam3/snapshots"
    for d in listdir(snaps):
        return snaps + "/" + String(d) + "/model.safetensors"
    raise Error("sam3: no checkpoint (set SAM3_CHECKPOINT, or fetch facebook/sam3)")


struct Sam3Weights(Movable):
    var ctx: DeviceContext
    var index: Dict[String, Int]
    var dev: List[DeviceBuffer[DT]]
    var used: List[Bool]
    var names: List[String]
    var bf16_source: Bool
    """The file held bf16 (`tools/perception/extract_sam3_detector.py`): the
    weights were widened to fp32 here, and they are the fp32 model's rounded."""
    var bf16: Dict[String, DeviceBuffer[DType.bfloat16]]
    """SAM3_BF16: each linear weight's bf16 copy, made on first use."""

    def __init__(out self, var ctx: DeviceContext, path: String, prefix: String) raises:
        """Every tensor whose name starts with `prefix`, the prefix dropped;
        fp32, or bf16 widened to fp32 on the way in."""
        self.ctx = ctx^
        self.index = Dict[String, Int]()
        self.dev = List[DeviceBuffer[DT]]()
        self.used = List[Bool]()
        self.names = List[String]()
        self.bf16_source = False
        self.bf16 = Dict[String, DeviceBuffer[DType.bfloat16]]()
        var st = SafeTensors(path)
        for i in range(st.size()):
            var name = st.names[i]
            if not name.startswith(prefix):
                continue
            var raw = st.read_raw(name)
            var n: Int
            var h: HostBuffer[DT]
            if st.dtypes[i] == ST_F32:
                n = len(raw) // 4
                h = self.ctx.enqueue_create_host_buffer[DT](n)
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=raw.unsafe_ptr(), count=len(raw))
            elif st.dtypes[i] == ST_BF16:
                n = len(raw) // 2
                h = self.ctx.enqueue_create_host_buffer[DT](n)
                var src = raw.unsafe_ptr().unsafe_bitcast[UInt16]()
                var dst = h.unsafe_ptr().unsafe_bitcast[UInt32]()
                for j in range(n):
                    dst[unsafe_offset=j] = UInt32(src[unsafe_offset=j]) << 16
                self.bf16_source = True
            else:
                raise Error("sam3: " + name + " is neither fp32 nor bf16")
            var d = self.ctx.enqueue_create_buffer[DT](n)
            self.ctx.enqueue_copy(d, h)
            self.ctx.synchronize()
            var short = String(name[byte=prefix.byte_length():])
            self.index[short] = len(self.dev)
            self.dev.append(d^)
            self.used.append(False)
            self.names.append(short)
        if len(self.dev) == 0:
            raise Error("sam3: no tensor under '" + prefix + "' in " + path)

    def w(mut self, name: String) raises -> DeviceBuffer[DT]:
        var i = self.index.get(name)
        if not i:
            raise Error("sam3: no tensor '" + name + "'")
        self.used[i.value()] = True
        return self.dev[i.value()]

    def w_bf16(mut self, name: String) raises -> DeviceBuffer[DType.bfloat16]:
        """`name`'s bf16 copy (rounded to nearest even once, then kept)."""
        var c = self.bf16.get(name)
        if c:
            return c.value()
        var f = self.w(name)
        var b = self.ctx.enqueue_create_buffer[DType.bfloat16](len(f))
        self.ctx.enqueue_function[to_bf16_kernel](f, b, Int64(len(f)), grid_dim=blocks(len(f)), block_dim=TPB)
        self.bf16[name] = b
        return b^

    def concat(mut self, name: String, parts: List[String]) raises:
        """`name.weight` / `name.bias`: the parts' stacked along the output axis
        (q, k, v as ONE GEMM whose rows are `[q | k | v]`, the fused attention's
        interleaved input). The parts count as used; their device copies are
        released (the stacked one replaces them)."""
        var suffixes = List[String]()
        suffixes.append(".weight")
        suffixes.append(".bias")
        for suffix in suffixes:
            var total = 0
            for part in parts:
                total += len(self.w(part + suffix))
            var d = self.ctx.enqueue_create_buffer[DT](total)
            var off = 0
            for part in parts:
                var src = self.w(part + suffix)
                self.ctx.enqueue_copy(d.create_sub_buffer[DT](off, len(src)), src)
                off += len(src)
            self.ctx.synchronize()
            for part in parts:
                self.dev[self.index[part + suffix]] = self.ctx.enqueue_create_buffer[DT](1)
            self.index[name + suffix] = len(self.dev)
            self.dev.append(d^)
            self.used.append(False)
            self.names.append(name + suffix)

    def unused(self, under: String = String("")) -> List[String]:
        var out = List[String]()
        for i in range(len(self.used)):
            if not self.used[i] and self.names[i].startswith(under):
                out.append(self.names[i])
        return out^


def gemm_tb(ctx: DeviceContext, dst: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT], m: Int, n: Int, k: Int) raises:
    """`dst[m, n] = a[m, k] @ b[n, k]^T`, fp32 (TF32 with SAM3_TF32 on NVIDIA)."""
    comptime if default_accelerator().is_nvidia_gpu():
        cublas_gemm[False, True, SAM3_TF32](ctx, dst, a, b, m, n, k, 0.0)
    else:
        var out_buf = dst
        var av = TileTensor(a, row_major(m, k))
        var ov = TileTensor(out_buf, row_major(m, n))
        var bv = TileTensor(b, row_major(n, k))
        max_matmul[transpose_b=True, target="gpu"](ov, av, bv, ctx)


def linear(mut wts: Sam3Weights, name: String, x: DeviceBuffer[DT], t: Int, k: Int, bias: Bool = True) raises -> DeviceBuffer[DT]:
    """`x[t, k] @ W^T (+ b)` for `name.weight` [n, k] (and `name.bias`)."""
    var n = len(wts.w(name + ".weight")) // k
    var y = wts.ctx.enqueue_create_buffer[DT](t * n)
    linear_into(wts, name, x, t, k, y, bias)
    return y^


def linear_into(
    mut wts: Sam3Weights, name: String, x: DeviceBuffer[DT], t: Int, k: Int, y: DeviceBuffer[DT], bias: Bool = True,
) raises:
    """`linear`, written into `y` (e.g. a slot of a Q|K|V buffer)."""
    var wb = wts.w(name + ".weight")
    var n = len(wb) // k
    comptime if SAM3_BF16:
        if k >= BF16_MIN_K:
            var xb = wts.ctx.enqueue_create_buffer[DType.bfloat16](t * k)
            wts.ctx.enqueue_function[to_bf16_kernel](x, xb, Int64(t * k), grid_dim=blocks(t * k), block_dim=TPB)
            linear_into_b(wts, name, xb, t, k, y, bias)
            return
    gemm_tb(wts.ctx, y, x, wb, t, n, k)
    if bias:
        wts.ctx.enqueue_function[bias_kernel](
            y, wts.w(name + ".bias"), Int64(n), Int64(t), grid_dim=(t, blocks(n)), block_dim=TPB,
        )

def linear_into_b(
    mut wts: Sam3Weights, name: String, xb: DeviceBuffer[DType.bfloat16], t: Int, k: Int, y: DeviceBuffer[DT],
    bias: Bool = True,
) raises:
    """`linear_into` from an input already in bf16 (SAM3_BF16; fp32 out)."""
    var n = len(wts.w(name + ".weight")) // k
    cublas_gemm_lp[False, True, DType.bfloat16, DT](wts.ctx, y, xb, wts.w_bf16(name + ".weight"), t, n, k, 0.0)
    if bias:
        wts.ctx.enqueue_function[bias_kernel](
            y, wts.w(name + ".bias"), Int64(n), Int64(t), grid_dim=(t, blocks(n)), block_dim=TPB,
        )


def bias_add(mut wts: Sam3Weights, name: String, x: DeviceBuffer[DT], y: DeviceBuffer[DT], rows: Int, cols: Int) raises:
    """`x += y + name.bias` in one pass (a residual add with the linear's bias
    folded in: the same floats as `y + b` then `x + (y + b)`)."""
    wts.ctx.enqueue_function[bias_add_kernel](
        x, y, wts.w(name + ".bias"), Int64(cols), grid_dim=(rows, blocks(cols)), block_dim=TPB,
    )


def linear_residual(
    mut wts: Sam3Weights, name: String, x: DeviceBuffer[DT], t: Int, k: Int, resid: DeviceBuffer[DT], n: Int,
) raises:
    """`resid += x W^T + b` (`resid` [t, n])."""
    var y = wts.ctx.enqueue_create_buffer[DT](t * n)
    linear_into(wts, name, x, t, k, y, bias=False)
    bias_add(wts, name, resid, y, t, n)


def layer_norm_bf16(
    mut wts: Sam3Weights, name: String, x: DeviceBuffer[DT], y: DeviceBuffer[DType.bfloat16], rows: Int, dim: Int,
    eps: Float32, grid: Int = 0, win: Int = 0,
) raises:
    """`layer_norm` written as bf16, row r at `_win_row(r, grid, win)` (`win` 0: in place order)."""
    wts.ctx.enqueue_function[layer_norm_bf16_kernel](
        x, y, wts.w(name + ".weight"), wts.w(name + ".bias"), Int64(dim), eps, Int64(grid), Int64(win),
        grid_dim=rows, block_dim=TPB,
    )


def bias_add_perm(
    mut wts: Sam3Weights, name: String, x: DeviceBuffer[DT], y: DeviceBuffer[DT], rows: Int, cols: Int, grid: Int, win: Int,
) raises:
    """`bias_add` with `y` in window order (`_win_row`)."""
    wts.ctx.enqueue_function[bias_add_perm_kernel](
        x, y, wts.w(name + ".bias"), Int64(cols), Int64(grid), Int64(win), grid_dim=(rows, blocks(cols)), block_dim=TPB,
    )


def mlp_residual_b[ACT: Int](
    mut wts: Sam3Weights, fc1: String, fc2: String, xb: DeviceBuffer[DType.bfloat16], t: Int, d: Int, ff: Int,
    resid: DeviceBuffer[DT],
) raises:
    """`mlp_residual` from an input already in bf16 (SAM3_BF16)."""
    var f1 = wts.ctx.enqueue_create_buffer[DT](t * ff)
    linear_into_b(wts, fc1, xb, t, d, f1, bias=False)
    _mlp_tail[ACT](wts, fc1, fc2, f1, t, d, ff, resid)


def mlp_residual[ACT: Int](
    mut wts: Sam3Weights, fc1: String, fc2: String, x: DeviceBuffer[DT], t: Int, d: Int, ff: Int,
    resid: DeviceBuffer[DT],
) raises:
    """`resid += fc2(act(fc1(x)))`, `ACT` 1 exact GELU, 2 ReLU: fc1's bias rides
    the activation's pass, fc2's the residual add's (the same floats as the
    unfused order); with SAM3_BF16 the activation writes fc2's bf16 operand."""
    var f1 = wts.ctx.enqueue_create_buffer[DT](t * ff)
    linear_into(wts, fc1, x, t, d, f1, bias=False)
    _mlp_tail[ACT](wts, fc1, fc2, f1, t, d, ff, resid)


def _mlp_tail[ACT: Int](
    mut wts: Sam3Weights, fc1: String, fc2: String, f1: DeviceBuffer[DT], t: Int, d: Int, ff: Int,
    resid: DeviceBuffer[DT],
) raises:
    """`resid += fc2(act(f1 + fc1.bias))`."""
    var f2 = wts.ctx.enqueue_create_buffer[DT](t * d)
    comptime if SAM3_BF16:
        var fb = wts.ctx.enqueue_create_buffer[DType.bfloat16](t * ff)
        if ff % 4 == 0:
            wts.ctx.enqueue_function[bias_act_bf16_v4_kernel[ACT]](
                f1, wts.w(fc1 + ".bias"), fb, Int64(ff), grid_dim=(t, (ff // 4 + TPB - 1) // TPB), block_dim=TPB,
            )
        else:
            wts.ctx.enqueue_function[bias_act_bf16_kernel[ACT]](
                f1, wts.w(fc1 + ".bias"), fb, Int64(ff), grid_dim=(t, blocks(ff)), block_dim=TPB,
            )
        cublas_gemm_lp[False, True, DType.bfloat16, DT](wts.ctx, f2, fb, wts.w_bf16(fc2 + ".weight"), t, d, ff, 0.0)
    else:
        wts.ctx.enqueue_function[bias_act_kernel[ACT]](
            f1, wts.w(fc1 + ".bias"), Int64(ff), grid_dim=(t, blocks(ff)), block_dim=TPB,
        )
        gemm_tb(wts.ctx, f2, f1, wts.w(fc2 + ".weight"), t, d, ff)
    bias_add(wts, fc2, resid, f2, t, d)


# ── kernels ──────────────────────────────────────────────────────────────


@always_inline
def _act[ACT: Int, W: Int = 1](v: SIMD[DT, W]) -> SIMD[DT, W]:
    """1: exact GELU (`gelu_kernel`'s); 2: ReLU; else identity — per lane."""
    comptime if ACT == 1:
        return SIMD[DT, W](0.5) * v * (SIMD[DT, W](1) + erf(v * SIMD[DT, W](0.7071067811865476)))
    elif ACT == 2:
        return max(v, SIMD[DT, W](0))
    else:
        return v


def bias_act_kernel[ACT: Int](x: Ptr, b: Ptr, cols_n: Int64):
    """`x[r, c] = act(x[r, c] + b[c])` in place. Grid `(rows, ceil(cols / TPB))`."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var cols = Int(cols_n)
    if c < cols:
        var i = Int(block_idx.x) * cols + c
        x[unsafe_offset=i] = _act[ACT](x[unsafe_offset=i] + b[unsafe_offset=c])


def bias_act_bf16_kernel[ACT: Int](x: Ptr, b: Ptr, dst: BPtr, cols_n: Int64):
    """`dst[r, c] = bf16(act(x[r, c] + b[c]))` (the next GEMM's operand).
    Grid `(rows, ceil(cols / TPB))`."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var cols = Int(cols_n)
    if c < cols:
        var i = Int(block_idx.x) * cols + c
        dst[unsafe_offset=i] = _act[ACT](x[unsafe_offset=i] + b[unsafe_offset=c]).cast[DType.bfloat16]()


def bias_act_bf16_v4_kernel[ACT: Int](x: Ptr, b: Ptr, dst: BPtr, cols_n: Int64):
    """`bias_act_bf16_kernel` four columns a thread (16-byte loads; `cols` a
    multiple of 4): the same per-element operations. Grid `(rows, ceil(cols /
    4 TPB))`."""
    var c = (Int(block_idx.y) * TPB + Int(thread_idx.x)) * 4
    var cols = Int(cols_n)
    if c < cols:
        var i = Int(block_idx.x) * cols + c
        var v = x.unsafe_load[width=4, alignment=16](i) + b.unsafe_load[width=4, alignment=16](c)
        dst.unsafe_store[alignment=8](i, _act[ACT, 4](v).cast[DType.bfloat16]())


@always_inline
def _win_row(n: Int, grid: Int, win: Int) -> Int:
    """Raster row `n` of a `grid x grid` map -> its row in window order (the
    `win x win` windows row-major, their tokens row-major) — `vit.window_kernel`'s
    order; `win` 0: `n`."""
    if win == 0:
        return n
    var r = _d(n, grid)
    var c = n - r * grid
    var wr = _d(r, win)
    var wc = _d(c, win)
    return ((wr * _d(grid, win) + wc) * win + (r - wr * win)) * win + (c - wc * win)


def layer_norm_bf16_kernel(
    x: Ptr, y: BPtr, w: Ptr, b: Ptr, dim_n: Int64, eps: Float32, grid_n: Int64, win_n: Int64,
):
    """`layer_norm_kernel`'s arithmetic, the result rounded to bf16 (the next
    GEMM's operand) into row `_win_row(r)` — the backbone's windowed layers get
    their input in window order without a permutation pass. A block per row."""
    var dim = Int(dim_n)
    var r = Int(block_idx.x)
    var row = r * dim
    var orow = _win_row(r, Int(grid_n), Int(win_n)) * dim
    var tid = Int(thread_idx.x)
    var s = Scalar[DT](0)
    var d = tid
    while d < dim:
        s += x[unsafe_offset=row + d]
        d += TPB
    var mean = block.sum[block_size=TPB, broadcast=True](val=s) / Scalar[DT](dim)
    var q = Scalar[DT](0)
    d = tid
    while d < dim:
        var v = x[unsafe_offset=row + d] - mean
        q += v * v
        d += TPB
    var inv = Scalar[DT](1) / sqrt(block.sum[block_size=TPB, broadcast=True](val=q) / Scalar[DT](dim) + Scalar[DT](eps))
    d = tid
    while d < dim:
        y[unsafe_offset=orow + d] = ((x[unsafe_offset=row + d] - mean) * inv * w[unsafe_offset=d] + b[unsafe_offset=d]).cast[DType.bfloat16]()
        d += TPB


def bias_add_perm_kernel(x: Ptr, y: Ptr, b: Ptr, cols_n: Int64, grid_n: Int64, win_n: Int64):
    """`x[r, c] += y[_win_row(r), c] + b[c]`: a residual add reading a
    window-order result in raster order. Grid `(rows, ceil(cols / TPB))`."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var cols = Int(cols_n)
    if c < cols:
        var r = Int(block_idx.x)
        var i = r * cols + c
        x[unsafe_offset=i] = x[unsafe_offset=i] + (y[unsafe_offset=_win_row(r, Int(grid_n), Int(win_n)) * cols + c] + b[unsafe_offset=c])


def bias_add_kernel(x: Ptr, y: Ptr, b: Ptr, cols_n: Int64):
    """`x[r, c] += y[r, c] + b[c]`. Grid `(rows, ceil(cols / TPB))`."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var cols = Int(cols_n)
    if c < cols:
        var i = Int(block_idx.x) * cols + c
        x[unsafe_offset=i] = x[unsafe_offset=i] + (y[unsafe_offset=i] + b[unsafe_offset=c])


def to_bf16_kernel(src: Ptr, dst: BPtr, n_n: Int64):
    """Round to bf16 (nearest even): a tensor-core GEMM's operand."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        dst[unsafe_offset=i] = src[unsafe_offset=i].cast[DType.bfloat16]()


def bias_kernel(x: Ptr, b: Ptr, cols_n: Int64, rows_n: Int64):
    """`x[r, c] += b[c]`. Grid `(rows, ceil(cols / TPB))`: no division, and the
    rows on x (CUDA caps grid y at 65 535; the pixel decoder's linears have
    82 944)."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var r = Int(block_idx.x)
    var cols = Int(cols_n)
    if c < cols:
        x[unsafe_offset=r * cols + c] = x[unsafe_offset=r * cols + c] + b[unsafe_offset=c]


def add_kernel(dst: Ptr, src: Ptr, n_n: Int64):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        dst[unsafe_offset=i] = dst[unsafe_offset=i] + src[unsafe_offset=i]


def gelu_kernel(x: Ptr, n_n: Int64):
    """Exact GELU in place (torch's default `gelu`)."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var v = x[unsafe_offset=i]
        x[unsafe_offset=i] = Scalar[DT](0.5) * v * (Scalar[DT](1) + erf(v * Scalar[DT](0.7071067811865476)))


def layer_norm_kernel(x: Ptr, y: Ptr, w: Ptr, b: Ptr, dim_n: Int64, eps: Float32):
    """`y[r] = (x[r] - mean) / sqrt(var + eps) * w + b` (biased variance), a
    block per row; `y` may be `x`."""
    var dim = Int(dim_n)
    var row = Int(block_idx.x) * dim
    var tid = Int(thread_idx.x)
    var s = Scalar[DT](0)
    var d = tid
    while d < dim:
        s += x[unsafe_offset=row + d]
        d += TPB
    var mean = block.sum[block_size=TPB, broadcast=True](val=s) / Scalar[DT](dim)
    var q = Scalar[DT](0)
    d = tid
    while d < dim:
        var v = x[unsafe_offset=row + d] - mean
        q += v * v
        d += TPB
    var inv = Scalar[DT](1) / sqrt(block.sum[block_size=TPB, broadcast=True](val=q) / Scalar[DT](dim) + Scalar[DT](eps))
    d = tid
    while d < dim:
        y[unsafe_offset=row + d] = (x[unsafe_offset=row + d] - mean) * inv * w[unsafe_offset=d] + b[unsafe_offset=d]
        d += TPB


def layer_norm(mut wts: Sam3Weights, name: String, x: DeviceBuffer[DT], y: DeviceBuffer[DT], rows: Int, dim: Int, eps: Float32) raises:
    wts.ctx.enqueue_function[layer_norm_kernel](
        x, y, wts.w(name + ".weight"), wts.w(name + ".bias"), Int64(dim), eps, grid_dim=rows, block_dim=TPB,
    )


def rope_pairs_kernel(x: Ptr, cos_t: Ptr, sin_t: Ptr, heads_n: Int64, pairs_n: Int64, stride_n: Int64):
    """PAIRWISE RoPE in place on `x[T, heads, 2 pairs]` (rows `stride` apart:
    `heads 2 pairs`, or `3 x` that inside a `[q | k | v]` row): (2i, 2i+1)
    turned by angle `i` (`cos_t` / `sin_t` [T, pairs]) — transformers'
    `rotate_pairwise`, not the rotate-half of LFM3. Grid `(ceil(heads pairs /
    TPB), T)`."""
    var pairs = Int(pairs_n)
    var heads = Int(heads_n)
    var idx = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    if idx >= heads * pairs:
        return
    var h = _d(idx, pairs)
    var i = idx - h * pairs
    var base = t * Int(stride_n) + h * 2 * pairs + 2 * i
    var c = cos_t[unsafe_offset=t * pairs + i]
    var s = sin_t[unsafe_offset=t * pairs + i]
    var a = x[unsafe_offset=base]
    var b = x[unsafe_offset=base + 1]
    x[unsafe_offset=base] = a * c - b * s
    x[unsafe_offset=base + 1] = b * c + a * s


def qkv_rope_bf16_kernel(x: Ptr, b: Ptr, cos_t: Ptr, sin_t: Ptr, dst: BPtr, heads_n: Int64, pairs_n: Int64):
    """One pass over an interleaved `[q | k | v]` row (`3 heads 2 pairs` wide):
    the bias, `rope_pairs_kernel`'s rotation on q and k, the bf16 rounding (the
    bf16 fused attention's input). Grid `(ceil(3 heads pairs / TPB), T)`."""
    var pairs = Int(pairs_n)
    var dimp = Int(heads_n) * pairs
    var idx = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    if idx >= 3 * dimp:
        return
    var part = _d(idx, dimp)
    var j = idx - part * dimp
    var i = j - _d(j, pairs) * pairs
    var col = part * 2 * dimp + 2 * j
    var base = t * 6 * dimp + col
    var a = x[unsafe_offset=base] + b[unsafe_offset=col]
    var c = x[unsafe_offset=base + 1] + b[unsafe_offset=col + 1]
    if part < 2:
        var cs = cos_t[unsafe_offset=t * pairs + i]
        var sn = sin_t[unsafe_offset=t * pairs + i]
        var ra = a * cs - c * sn
        var rc = c * cs + a * sn
        a = ra
        c = rc
    dst[unsafe_offset=base] = a.cast[DType.bfloat16]()
    dst[unsafe_offset=base + 1] = c.cast[DType.bfloat16]()


def put_head_kernel(src: Ptr, dst: Ptr, h_n: Int64, hd_n: Int64, heads_n: Int64):
    """`dst[t, h, :] = src[t, :]` (one head's attention output into its
    columns). Grid `(ceil(hd / TPB), T)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    var hd = Int(hd_n)
    if d < hd:
        dst[unsafe_offset=(t * Int(heads_n) + Int(h_n)) * hd + d] = src[unsafe_offset=t * hd + d]


# ── attention: any query / key counts, one head at a time, GEMMs ──────────


def gemm_nn(ctx: DeviceContext, dst: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT], m: Int, n: Int, k: Int) raises:
    """`dst[m, n] = a[m, k] @ b[k, n]`."""
    comptime if default_accelerator().is_nvidia_gpu():
        cublas_gemm[False, False, SAM3_TF32](ctx, dst, a, b, m, n, k, 0.0)
    else:
        var out_buf = dst
        var av = TileTensor(a, row_major(m, k))
        var ov = TileTensor(out_buf, row_major(m, n))
        var bv = TileTensor(b, row_major(k, n))
        max_matmul[transpose_b=False, target="gpu"](ov, av, bv, ctx)


def heads_kernel(src: Ptr, dst: Ptr, t_n: Int64, heads_n: Int64, hd_n: Int64):
    """`dst[h, t, d] = src[t, h, d]`. Grid `(ceil(heads hd / TPB), T)`."""
    var col = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    var heads = Int(heads_n)
    var hd = Int(hd_n)
    if col < heads * hd:
        var h = _d(col, hd)
        dst[unsafe_offset=(h * Int(t_n) + t) * hd + (col - h * hd)] = src[unsafe_offset=t * heads * hd + col]


def attn_softmax_kernel(sc: Ptr, bias: Ptr, tk_n: Int64, scale: Float32, key_len_n: Int64, has_bias: Int64):
    """Row `i` of `sc[tq, tk]` to `softmax_j(scale sc + bias[i, j])` over the
    keys `j < key_len` (0 after), in place — torch's `qk^T * scaling +
    attention_mask`. A block per row."""
    var tk = Int(tk_n)
    var row = Int(block_idx.x) * tk
    var tid = Int(thread_idx.x)
    var kl = Int(key_len_n)
    var m = Scalar[DT](-3.0e38)
    var j = tid
    while j < tk:
        var v = Scalar[DT](-3.0e38)
        if j < kl:
            v = sc[unsafe_offset=row + j] * Scalar[DT](scale)
            if has_bias != 0:
                v += bias[unsafe_offset=row + j]
        sc[unsafe_offset=row + j] = v
        m = max(m, v)
        j += TPB
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var s = Scalar[DT](0)
    j = tid
    while j < tk:
        var e = Scalar[DT](0)
        if j < kl:
            e = exp(sc[unsafe_offset=row + j] - top)
        sc[unsafe_offset=row + j] = e
        s += e
        j += TPB
    var inv = Scalar[DT](1) / block.sum[block_size=TPB, broadcast=True](val=s)
    j = tid
    while j < tk:
        sc[unsafe_offset=row + j] = sc[unsafe_offset=row + j] * inv
        j += TPB


def attn_softmax_warp_kernel(
    sc: Ptr, bias: Ptr, rows_n: Int64, tk_n: Int64, scale: Float32, key_len_n: Int64, has_bias: Int64,
):
    """`attn_softmax_kernel` with a WARP per row, 8 rows a block (the batched
    `mha`'s): a block per row left 224 of 256 threads idle on 32 text keys and
    synchronised the block three times a row. Same operations per element.
    Grid `ceil(rows / 8)`, 256 threads."""
    var tid = Int(thread_idx.x)
    var r = Int(block_idx.x) * 8 + (tid >> 5)
    if r >= Int(rows_n):
        return  # a whole warp leaves together (r is per warp)
    var lane = tid & 31
    var tk = Int(tk_n)
    var row = r * tk
    var kl = Int(key_len_n)
    var m = Scalar[DT](-3.0e38)
    var j = lane
    while j < tk:
        var v = Scalar[DT](-3.0e38)
        if j < kl:
            v = sc[unsafe_offset=row + j] * Scalar[DT](scale)
            if has_bias != 0:
                v += bias[unsafe_offset=row + j]
        sc[unsafe_offset=row + j] = v
        m = max(m, v)
        j += 32
    var top = warp.max(m)
    var s = Scalar[DT](0)
    j = lane
    while j < tk:
        var e = Scalar[DT](0)
        if j < kl:
            e = exp(sc[unsafe_offset=row + j] - top)
        sc[unsafe_offset=row + j] = e
        s += e
        j += 32
    var inv = Scalar[DT](1) / warp.sum(s)
    j = lane
    while j < tk:
        sc[unsafe_offset=row + j] = sc[unsafe_offset=row + j] * inv
        j += 32


def attn_softmax_rpb_kernel(
    sc: Ptr, dy: Ptr, dx: Ptr, tq_n: Int64, tk_n: Int64, g_n: Int64, heads_n: Int64, scale: Float32,
):
    """`attn_softmax_kernel` (all keys real) with SAM 3's box relative-position
    bias made per key instead of read from a [heads, tq, tk] tensor:
    `bias[h, qq, r g + c] = dy[((qq - 1) g + r) heads + h] + dx[((qq - 1) g + c)
    heads + h]`, 0 for the presence row qq = 0 — the decoder's `rpb_kernel` sum,
    so the same floats, without writing and reading 33 MB a layer. A block per
    row of `sc [heads, tq, tk]`; grid `heads tq`."""
    var ri = Int(block_idx.x)
    var tq = Int(tq_n)
    var tk = Int(tk_n)
    var g = Int(g_n)
    var heads = Int(heads_n)
    var h = _d(ri, tq)
    var qq = ri - h * tq
    var row = ri * tk
    var tid = Int(thread_idx.x)
    var m = Scalar[DT](-3.0e38)
    var j = tid
    while j < tk:
        var v = sc[unsafe_offset=row + j] * Scalar[DT](scale)
        if qq > 0:
            var r = _d(j, g)
            var c = j - r * g
            v += dy[unsafe_offset=((qq - 1) * g + r) * heads + h] + dx[unsafe_offset=((qq - 1) * g + c) * heads + h]
        sc[unsafe_offset=row + j] = v
        m = max(m, v)
        j += TPB
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var s = Scalar[DT](0)
    j = tid
    while j < tk:
        var e = exp(sc[unsafe_offset=row + j] - top)
        sc[unsafe_offset=row + j] = e
        s += e
        j += TPB
    var inv = Scalar[DT](1) / block.sum[block_size=TPB, broadcast=True](val=s)
    j = tid
    while j < tk:
        sc[unsafe_offset=row + j] = sc[unsafe_offset=row + j] * inv
        j += TPB


def mha_rpb(
    ctx: DeviceContext, q: DeviceBuffer[DT], k: DeviceBuffer[DT], v: DeviceBuffer[DT], tq: Int, tk: Int,
    heads: Int, hd: Int, dy: DeviceBuffer[DT], dx: DeviceBuffer[DT], g: Int,
) raises -> DeviceBuffer[DT]:
    """`mha` (batched, NVIDIA) with the decoder's box bias from its two factors
    (`attn_softmax_rpb_kernel`): `dy`, `dx` [(tq - 1) g, heads], `tk = g g`."""
    comptime assert MHA_BATCHED, "mha_rpb: the batched (NVIDIA) path"
    var scale = Float32(1.0) / Float32(Float64(hd) ** 0.5)
    var ld = heads * hd
    var sc = ctx.enqueue_create_buffer[DT](heads * tq * tk)
    gemm_strided_batched[False, True](ctx, sc, q, k, tq, tk, hd, ld, ld, tk, hd, hd, tq * tk, heads)
    ctx.enqueue_function[attn_softmax_rpb_kernel](
        sc, dy, dx, Int64(tq), Int64(tk), Int64(g), Int64(heads), scale, grid_dim=heads * tq, block_dim=TPB,
    )
    var out = ctx.enqueue_create_buffer[DT](tq * ld)
    gemm_strided_batched[False, False](ctx, out, sc, v, tq, hd, tk, tk, ld, ld, tq * tk, hd, hd, heads)
    return out^


def gemm_strided_batched[TA: Bool, TB: Bool](
    ctx: DeviceContext, dst: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT], m: Int, n: Int, k: Int,
    lda: Int, ldb: Int, ldc: Int, sa: Int, sb: Int, sc: Int, batch: Int,
) raises:
    """`dst_i[m, n] = op(a_i) @ op(b_i)` for i < batch, row-major with explicit
    row strides (`ld*`) and per-batch element strides (`s*`): `a_i` is [m, k]
    (or [k, m]^T with TA), `b_i` [k, n] (or [n, k]^T with TB). One
    `cublasGemmStridedBatchedEx` (MAX does not bind it; same C-ABI trick as
    `nn.core.cudnn_conv`), fp32 (TF32 with SAM3_TF32). Heads interleaved in
    `[t, heads hd]` are a batch with stride `hd` and row stride `heads hd`.
    The row-major mapping is `cublas_gemm`'s: cuBLAS gets (b, a), dims (n, m)."""
    comptime assert default_accelerator().is_nvidia_gpu(), "cuBLAS is NVIDIA-only"
    var ws = _shared_workspace(ctx)
    var handle = _get_global_handle[DT, Backend.CUBLAS](ctx)._get_cublas()
    check_cublas_error(cublasSetWorkspace(handle, _ffi_void_ptr(ws.unsafe_ptr()), CUBLAS_WS_BYTES))
    check_cublas_error(
        cublasSetMathMode(handle, cublasMath_t.CUBLAS_TF32_TENSOR_OP_MATH if SAM3_TF32 else cublasMath_t.CUBLAS_DEFAULT_MATH)
    )
    var alpha = Float32(1.0)
    var beta = Float32(0.0)
    var op_b = _convert_to_cublas_transpose(TB)
    var op_a = _convert_to_cublas_transpose(TA)
    var dt = _convert_to_cublas_datatype[DT]()
    var compute = ComputeType.COMPUTE_32F_FAST_TF32 if SAM3_TF32 else ComputeType.COMPUTE_32F
    var algo = Algorithm.DEFAULT
    var st = _cublas_fn[
        "cublasGemmStridedBatchedEx",
        def(
            type_of(handle), type_of(op_b), type_of(op_a), Int32, Int32, Int32, Int,
            Int, type_of(dt), Int32, Int64,
            Int, type_of(dt), Int32, Int64,
            Int, Int, type_of(dt), Int32, Int64, Int32, type_of(compute), type_of(algo),
        ) thin abi("C") -> Int32,
    ]()(
        handle, op_b, op_a, Int32(n), Int32(m), Int32(k), Int(Pointer(to=alpha)),
        Int(b.unsafe_ptr()), dt, Int32(ldb), Int64(sb),
        Int(a.unsafe_ptr()), dt, Int32(lda), Int64(sa),
        Int(Pointer(to=beta)), Int(dst.unsafe_ptr()), dt, Int32(ldc), Int64(sc), Int32(batch), compute, algo,
    )
    _ = alpha
    _ = beta
    if st != 0:
        raise Error("sam3: cublasGemmStridedBatchedEx failed with status " + String(st))


comptime MHA_BATCHED: Bool = default_accelerator().is_nvidia_gpu()
"""On NVIDIA, `mha` takes every head at once (two strided-batched GEMMs and one
softmax launch, not ~4 launches a head) whenever the scores fit in 64 M floats
— every attention of a word (cross-attentions on 32 text keys, the decoder's
200 queries on 5184 image tokens); the fp32 DETR encoder's 5184^2 self-attention
stays per head."""


def mha(
    ctx: DeviceContext, q: DeviceBuffer[DT], k: DeviceBuffer[DT], v: DeviceBuffer[DT], tq: Int, tk: Int,
    heads: Int, hd: Int, key_len: Int, bias: Optional[DeviceBuffer[DT]] = None,
) raises -> DeviceBuffer[DT]:
    """Multi-head attention `[tq, heads hd]` from projected `q [tq, heads hd]`,
    `k`, `v [tk, heads hd]`: per head `softmax(q k^T / sqrt(hd) + bias_h)` over
    the first `key_len` keys (a padding mask whose real keys come first), then
    `@ v`. `bias`: `[heads, tq, tk]`, added before the softmax (SAM 3's box
    relative-position bias). One head at a time (a 5184 x 5184 head is 107 MB),
    or all at once (`MHA_BATCHED`)."""
    var scale = Float32(1.0) / Float32(Float64(hd) ** 0.5)
    comptime if MHA_BATCHED:
        if heads * tq * tk <= (64 << 20):
            var ld = heads * hd
            var sc = ctx.enqueue_create_buffer[DT](heads * tq * tk)
            gemm_strided_batched[False, True](ctx, sc, q, k, tq, tk, hd, ld, ld, tk, hd, hd, tq * tk, heads)
            var bp = bias.value() if bias else sc
            if tk <= 256:  # short rows (text keys, the 201 queries): a warp a row
                ctx.enqueue_function[attn_softmax_warp_kernel](
                    sc, bp, Int64(heads * tq), Int64(tk), scale, Int64(key_len), Int64(1 if bias else 0),
                    grid_dim=(heads * tq + 7) // 8, block_dim=256,
                )
            else:  # long rows (5184 image keys): a block a row measured faster
                ctx.enqueue_function[attn_softmax_kernel](
                    sc, bp, Int64(tk), scale, Int64(key_len), Int64(1 if bias else 0), grid_dim=heads * tq, block_dim=TPB,
                )
            var out = ctx.enqueue_create_buffer[DT](tq * ld)
            gemm_strided_batched[False, False](ctx, out, sc, v, tq, hd, tk, tk, ld, ld, tq * tk, hd, hd, heads)
            return out^
    var qh = ctx.enqueue_create_buffer[DT](heads * tq * hd)
    var kh = ctx.enqueue_create_buffer[DT](heads * tk * hd)
    var vh = ctx.enqueue_create_buffer[DT](heads * tk * hd)
    ctx.enqueue_function[heads_kernel](q, qh, Int64(tq), Int64(heads), Int64(hd), grid_dim=(blocks(heads * hd), tq), block_dim=TPB)
    ctx.enqueue_function[heads_kernel](k, kh, Int64(tk), Int64(heads), Int64(hd), grid_dim=(blocks(heads * hd), tk), block_dim=TPB)
    ctx.enqueue_function[heads_kernel](v, vh, Int64(tk), Int64(heads), Int64(hd), grid_dim=(blocks(heads * hd), tk), block_dim=TPB)
    var sc = ctx.enqueue_create_buffer[DT](tq * tk)
    var oh = ctx.enqueue_create_buffer[DT](tq * hd)
    var out = ctx.enqueue_create_buffer[DT](tq * heads * hd)
    for h in range(heads):
        gemm_tb(ctx, sc, qh.create_sub_buffer[DT](h * tq * hd, tq * hd), kh.create_sub_buffer[DT](h * tk * hd, tk * hd), tq, tk, hd)
        var bp = sc
        if bias:
            bp = bias.value().create_sub_buffer[DT](h * tq * tk, tq * tk)
        ctx.enqueue_function[attn_softmax_kernel](
            sc, bp, Int64(tk), scale, Int64(key_len), Int64(1 if bias else 0), grid_dim=tq, block_dim=TPB,
        )
        gemm_nn(ctx, oh, sc, vh.create_sub_buffer[DT](h * tk * hd, tk * hd), tq, hd, tk)
        ctx.enqueue_function[put_head_kernel](oh, out, Int64(h), Int64(hd), Int64(heads), grid_dim=(blocks(hd), tq), block_dim=TPB)
    return out^


def relu_kernel(x: Ptr, n_n: Int64):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        x[unsafe_offset=i] = max(x[unsafe_offset=i], Scalar[DT](0))


def add_into_kernel(dst: Ptr, a: Ptr, b: Ptr, n_n: Int64):
    """`dst = a + b`."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        dst[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


comptime USE_FLASH: Bool = SAM3_TF32 and default_accelerator().is_nvidia_gpu()
"""TF32 / bf16 builds on NVIDIA: the self-attentions of S keys (the backbone's
windows and global layers, the DETR encoder) through the tree's TF32 tensor-core
FlashAttention-2 forward (`nn/primitives/flash_attention_tf32.mojo`): no S x S
score matrix. The fp32 build keeps `mha` (the strict gates' path)."""


def flash_self[S: Int, H: Int, HD: Int, IL: Bool = False](
    ctx: DeviceContext, qkv: DeviceBuffer[DT], dst: DeviceBuffer[DT], batch: Int = 1,
) raises:
    """Bidirectional attention, `batch` independent samples: `qkv` holds per
    sample `[Q tokens | K tokens | V tokens]` ([S, H HD] each, heads interleaved),
    or with `IL` S rows of `[q | k | v]` (one GEMM's output);
    `dst` per sample [S, H HD]; scale 1 / sqrt(HD), TF32 products."""
    comptime if default_accelerator().is_nvidia_gpu():
        var lse = ctx.enqueue_create_buffer[DT](batch * H * S)
        ctx.enqueue_function[_tf32_fwd_kernel[H, S, HD, False, IL]](
            qkv, dst, dst, lse,  # the kernel's backward cache of the output IS the output
            grid_dim=((S + TF_TQ - 1) // TF_TQ, batch * H), block_dim=TF_WARPS * WARP_SIZE,
        )
    else:
        raise Error("sam3: flash_self runs on NVIDIA only")


def flash_self_bf16[S: Int, H: Int, HD: Int, IL: Bool = False](
    ctx: DeviceContext, qkv: DeviceBuffer[DType.bfloat16], dst: DeviceBuffer[DType.bfloat16], batch: Int = 1,
) raises:
    """`flash_self` on bf16 tensor cores (`nn/primitives/flash_attention_mma.mojo`,
    PyTorch FA2's numerics: bf16 products, fp32 softmax and accumulators), bf16
    in and out — SAM3_BF16."""
    comptime if default_accelerator().is_nvidia_gpu():
        var lse = ctx.enqueue_create_buffer[DT](batch * H * S)
        ctx.enqueue_function[_mma_fwd_kernel[DType.bfloat16, H, S, HD, False, IL]](
            qkv, dst, dst, lse,  # the backward cache of the output IS the output
            grid_dim=((S + MMA_TQ - 1) // MMA_TQ, batch * H), block_dim=MMA_WARPS * WARP_SIZE,
        )
    else:
        raise Error("sam3: flash_self_bf16 runs on NVIDIA only")
