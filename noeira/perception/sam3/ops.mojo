# +--------------------------------------------------------------------------+ #
# | SAM 3 — weights on the device, the GEMM, and the kernels its stages share
# +--------------------------------------------------------------------------+ #
"""Plumbing for `noeira.perception.sam3` (docs/NN_MODEL_PORTS_PLAN.md, SAM 3).

`Sam3Weights`: every fp32 tensor of the checkpoint under a prefix, one device
buffer each (the checkpoint is fp32: `facebook/sam3`'s `model.safetensors`, the
VIDEO model — the image detector is `detector_model.*`, the tracker is never
loaded). `linear`: `x W^T + b` through `gemm_tb` (cuBLAS on NVIDIA, fp32 or
`-D SAM3_TF32=1`; MAX's matmul elsewhere).
"""

from std.math import sqrt, erf, exp
from std.memory import unsafe_memcpy
from std.os import listdir, getenv
from std.sys import default_accelerator
from std.sys.defines import is_defined
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.primitives import block
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.io.safetensors import SafeTensors, ST_F32
from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm


comptime SAM3_TF32: Bool = is_defined["SAM3_TF32"]()
comptime TPB: Int = 256
comptime Ptr = Pointer[Scalar[DT], MutAnyOrigin]


def blocks(n: Int) -> Int:
    return (n + TPB - 1) // TPB


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

    def __init__(out self, var ctx: DeviceContext, path: String, prefix: String) raises:
        """Every tensor whose name starts with `prefix`, the prefix dropped."""
        self.ctx = ctx^
        self.index = Dict[String, Int]()
        self.dev = List[DeviceBuffer[DT]]()
        self.used = List[Bool]()
        self.names = List[String]()
        var st = SafeTensors(path)
        for i in range(st.size()):
            var name = st.names[i]
            if not name.startswith(prefix):
                continue
            if st.dtypes[i] != ST_F32:
                raise Error("sam3: " + name + " is not fp32")
            var raw = st.read_raw(name)
            var n = len(raw) // 4
            var h = self.ctx.enqueue_create_host_buffer[DT](n)
            unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=raw.unsafe_ptr(), count=len(raw))
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
    var wb = wts.w(name + ".weight")
    var n = len(wb) // k
    var y = wts.ctx.enqueue_create_buffer[DT](t * n)
    gemm_tb(wts.ctx, y, x, wb, t, n, k)
    if bias:
        wts.ctx.enqueue_function[bias_kernel](
            y, wts.w(name + ".bias"), Int64(n), Int64(t * n), grid_dim=blocks(t * n), block_dim=TPB,
        )
    return y^


# ── kernels ──────────────────────────────────────────────────────────────


def bias_kernel(x: Ptr, b: Ptr, cols_n: Int64, n_n: Int64):
    """`x[r, c] += b[c]`."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var cols = Int(cols_n)
        x[unsafe_offset=i] = x[unsafe_offset=i] + b[unsafe_offset=i % cols]


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


def rope_pairs_kernel(x: Ptr, cos_t: Ptr, sin_t: Ptr, heads_n: Int64, pairs_n: Int64):
    """PAIRWISE RoPE in place on `x[T, heads, 2 pairs]`: (2i, 2i+1) turned by
    angle `i` (`cos_t` / `sin_t` [T, pairs]) — transformers' `rotate_pairwise`,
    not the rotate-half of LFM3. Grid `(ceil(heads pairs / TPB), T)`."""
    var pairs = Int(pairs_n)
    var heads = Int(heads_n)
    var idx = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    if idx >= heads * pairs:
        return
    var h = idx // pairs
    var i = idx - h * pairs
    var base = (t * heads + h) * 2 * pairs + 2 * i
    var c = cos_t[unsafe_offset=t * pairs + i]
    var s = sin_t[unsafe_offset=t * pairs + i]
    var a = x[unsafe_offset=base]
    var b = x[unsafe_offset=base + 1]
    x[unsafe_offset=base] = a * c - b * s
    x[unsafe_offset=base + 1] = b * c + a * s


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
        var h = col // hd
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


def mha(
    ctx: DeviceContext, q: DeviceBuffer[DT], k: DeviceBuffer[DT], v: DeviceBuffer[DT], tq: Int, tk: Int,
    heads: Int, hd: Int, key_len: Int, bias: Optional[DeviceBuffer[DT]] = None,
) raises -> DeviceBuffer[DT]:
    """Multi-head attention `[tq, heads hd]` from projected `q [tq, heads hd]`,
    `k`, `v [tk, heads hd]`: per head `softmax(q k^T / sqrt(hd) + bias_h)` over
    the first `key_len` keys (a padding mask whose real keys come first), then
    `@ v`. `bias`: `[heads, tq, tk]`, added before the softmax (SAM 3's box
    relative-position bias). One head at a time (a 5184 x 5184 head is 107 MB)."""
    var scale = Float32(1.0) / Float32(Float64(hd) ** 0.5)
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
