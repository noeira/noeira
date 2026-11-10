# +--------------------------------------------------------------------------+ #
# | Parakeet-TDT — the FastConformer encoder, at a fixed maximum length
# +--------------------------------------------------------------------------+ #
"""Mel frames -> encoder frames, for `ParakeetTDT`'s weights.

    var ws = PkEncoderWork[801]()             # 8 s of audio: 801 mel frames, 101 encoder frames
    # ... fill ws.feats ([801, 128], zero beyond the valid mel frames) ...
    var t = pk_encode[801](model, ws, n_mel)  # valid encoder frames; ws.proj is [101, 640]

## Fixed maximum, runtime length (docs/NN_MODEL_PORTS_PLAN.md §9.3)

Every buffer and every `Linear` / `LayerNorm` call is sized by the COMPILED
maximum (`F_MAX` mel frames, hence `T` encoder frames); the clip's real length is
a runtime number. That is the reference's own padded path — a clip batched with
a longer one — and it stays exact for the valid frames because of three masks,
all reproduced here:

1. **Subsampling**: after every convolution, rows beyond the current valid
   length are zeroed (the length shrinks `n -> (n - 1) // 2 + 1` per stride-2
   conv). Without it the next 3x3 conv would read padding as signal.
2. **Attention**: a valid frame attends to valid frames only.
3. **Conformer conv**: rows beyond the valid length are zeroed before the
   kernel-9 depthwise conv, which would otherwise pull them in.

Feed-forwards and norms are per-frame, so padded rows compute something and
nothing reads it. Outputs beyond the valid length are NOT meaningful.

## Relative positions

Transformer-XL attention: `logit[i, j] = ((q_i + u) . k_j + (q_i + v) . r_{i-j}) / sqrt(128)`
with `r = rel_k(sinusoid)`. The table holds offsets `T-1 .. -(T-1)`, so offset
`i - j` is row `j - i + T - 1`, whatever `T` the table was built for — which is
why a compiled maximum does not change the valid frames' result.

## Two targets, one sequence of steps

`pk_subsample`, `pk_block` and `pk_encode` take `target` ("cpu" by default):
the `Linear` and `LayerNorm` calls go to that target as they are, and every
other step is either the plain loop below (CPU, written for exactness against
the reference) or the kernel of the same name in `kernels.mojo` (device). On a
device the buffers stay there: `pk_subsample` uploads `feats`, `pk_encode`
leaves `proj` there for the device decoder, and a caller that wants `x` or
`proj` on the host downloads it.

## What the device path does differently, and why

Measured on Metal, where a kernel launch costs ~22 us plus ~0.22 us per live
device buffer (`weights.mojo`), so that LAUNCHES and BUFFERS, not arithmetic,
were the whole cost of the first version (162 ms for the 24 blocks):

- the LayerNorms run nn's kernel on TWO scratch buffers of the workspace
  instead of each `LayerNorm` leaf's own pair (240 live buffers less);
- a bias-free `Linear` is the GEMM alone (`_gemm_any`): the leaf's forward
  would add its all-zero bias with one more launch, 10 times per block;
- `rel_k(positions)` does not depend on the clip, so the 24 projected tables
  are computed once (`pk_prepare_positions`) and kept in `relk_all`.
  ⚠ They are a CACHE of the `rel_k` weights and of `pos`: set
  `ws.relk_ready = False` after changing either;
- the block's output norm writes into `h`, and `x` and `h` exchange device
  buffers instead of copying.

## Only the valid rows are computed, where the backend allows it

With launches out of the way the GEMMs are the cost (114 of 127 ms on an M1
Pro), and at the compiled maximum most of their rows are padding (a 2.5 s
command in an 8 s shape: 45 rows of 101). The BUFFERS keep their compiled
size, but `_gemm_any`, the norms and the activations take the valid row count:
Metal's GEMM (runtime extents) and Accelerate's take it as an argument. MAX's
NVIDIA GEMM tiles its rows at compile time, so there — and on a CPU without
Accelerate — the full compiled shape runs. On the Orin that costs almost
nothing: its GEMM time at these sizes is the streaming of the weights, not
the rows (35.3 ms at 51 rows, 37.3 ms at 101; `stt.mojo`). Rows beyond the
valid length hold whatever an earlier clip left: finite, and never read.

`tests/audio/test_parakeet_reference.mojo` gates every stage on the CPU and
`tests/audio/test_parakeet_gpu.mojo` on the device.
"""

from std.math import exp, sin, cos, sqrt
from std.sys import default_accelerator, CompilationTarget
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.nn.constants import DT, TPB
from noeira.nn.core.cpu_gemm import gemm_cpu
from noeira.nn.primitives.layer_norm import (
    LayerNorm, _layer_norm_forward_kernel, LN_TPB, LN_EPS,
)
from noeira.nn.primitives.linear import Linear
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from .config import (
    PK_MELS, PK_D, PK_FF, PK_HEADS, PK_HEAD_DIM, PK_LAYERS, PK_CONV_K,
    PK_SUB_CH, PK_SUB_FREQ, PK_SUB_FLAT, PK_DEC, PK_BN_EPS, pk_sub_len,
)
from .kernels import (
    conv3x3_s2_first_kernel, dw3x3_s2_kernel, relu_rows_kernel,
    flatten_tfc_kernel, silu_kernel, add_scaled_kernel, rel_scores_kernel,
    rel_apply_kernel, glu_rows_kernel, dw1d_bn_silu_kernel,
)
from .weights import ParakeetTDT, PkConformerLayer


def pk_pos_table(t: Int) -> List[Float32]:
    """The relative positional encoding for `t` frames: `[2t - 1, D]`, row `p`
    is offset `t - 1 - p`, sin and cos interleaved. Float32 arithmetic in the
    reference's order (`inv_freq` in float32, then the angle)."""
    var out = List[Float32](length=(2 * t - 1) * PK_D, fill=0.0)
    for p in range(2 * t - 1):
        var position = Float32(t - 1 - p)
        for m in range(PK_D // 2):
            var e = Float32(2 * m) / Float32(PK_D)
            var inv_freq: Float32 = 1.0 / (Float32(10000.0) ** e)
            var angle = position * inv_freq
            out[p * PK_D + 2 * m] = sin(angle)
            out[p * PK_D + 2 * m + 1] = cos(angle)
    return out^


struct PkEncoderWork[F_MAX: Int](Movable):
    """The encoder's buffers at `F_MAX` mel frames. `feats` is the input,
    `x` the hidden state (`[T, 1024]`) after `pk_subsample` and each
    `pk_block`, `proj` the encoder projection (`[T, 640]`)."""

    comptime F1: Int = pk_sub_len(Self.F_MAX)
    comptime F2: Int = pk_sub_len(Self.F1)
    comptime T: Int = pk_sub_len(Self.F2)
    comptime P: Int = 2 * Self.T - 1
    comptime W1: Int = PK_MELS // 2
    comptime W2: Int = PK_MELS // 4

    var feats: Tensor
    var a1: Tensor
    var b1: Tensor
    var c1: Tensor
    var b2: Tensor
    var c2: Tensor
    var flat: Tensor
    var x: Tensor
    var h: Tensor
    var y: Tensor
    var ff: Tensor
    var q: Tensor
    var k: Tensor
    var v: Tensor
    var att: Tensor
    var pos: Tensor
    var relk: Tensor
    var relk_all: Tensor
    """`[LAYERS, P, 1024]`: every layer's `rel_k(pos)` (`pk_prepare_positions`)."""
    var relk_ready: Bool
    var ln_xhat: Tensor
    var ln_inv: Tensor
    """Device: the LayerNorm kernel's two scratch outputs, shared by all norms."""
    var glu: Tensor
    var conv: Tensor
    var proj: Tensor
    var scores: List[Float32]
    """CPU: one query's `[T]` logits."""
    var scores_dev: Tensor
    """Device: all the logits, `[HEADS, T, T]`."""

    def __init__(out self):
        """The CPU workspace (`make["cpu"]`)."""
        comptime assert PK_SUB_FREQ == PK_MELS // 8, "three stride-2 convs leave MELS / 8 bins"
        self.feats = Tensor.alloc(Self.F_MAX * PK_MELS)
        self.a1 = Tensor.alloc(Self.F1 * Self.W1 * PK_SUB_CH)
        self.b1 = Tensor.alloc(Self.F2 * Self.W2 * PK_SUB_CH)
        self.c1 = Tensor.alloc(Self.F2 * Self.W2 * PK_SUB_CH)
        self.b2 = Tensor.alloc(Self.T * PK_SUB_FLAT)
        self.c2 = Tensor.alloc(Self.T * PK_SUB_FLAT)
        self.flat = Tensor.alloc(Self.T * PK_SUB_FLAT)
        self.x = Tensor.alloc(Self.T * PK_D)
        self.h = Tensor.alloc(Self.T * PK_D)
        self.y = Tensor.alloc(Self.T * PK_D)
        self.ff = Tensor.alloc(Self.T * PK_FF)
        self.q = Tensor.alloc(Self.T * PK_D)
        self.k = Tensor.alloc(Self.T * PK_D)
        self.v = Tensor.alloc(Self.T * PK_D)
        self.att = Tensor.alloc(Self.T * PK_D)
        self.pos = Tensor.alloc(Self.P * PK_D)
        var table = pk_pos_table(Self.T)
        for i in range(len(table)):
            self.pos.data[i] = table[i]
        self.relk = Tensor.alloc(Self.P * PK_D)
        self.relk_all = Tensor.alloc(PK_LAYERS * Self.P * PK_D)
        self.relk_ready = False
        self.ln_xhat = Tensor()
        self.ln_inv = Tensor()
        self.glu = Tensor.alloc(Self.T * 2 * PK_D)
        self.conv = Tensor.alloc(Self.T * PK_D)
        self.proj = Tensor.alloc(Self.T * PK_DEC)
        self.scores = List[Float32](length=Self.T, fill=0.0)
        self.scores_dev = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        """The workspace on `target`. On a device every buffer also exists
        there (zeroed), and the positional table is uploaded once."""
        var w = Self()
        comptime if target == "gpu":
            var c = ctx.value()
            w.feats.ensure_gpu(c, Self.F_MAX * PK_MELS)
            w.a1.ensure_gpu(c, Self.F1 * Self.W1 * PK_SUB_CH)
            w.b1.ensure_gpu(c, Self.F2 * Self.W2 * PK_SUB_CH)
            w.c1.ensure_gpu(c, Self.F2 * Self.W2 * PK_SUB_CH)
            w.b2.ensure_gpu(c, Self.T * PK_SUB_FLAT)
            w.c2.ensure_gpu(c, Self.T * PK_SUB_FLAT)
            w.flat.ensure_gpu(c, Self.T * PK_SUB_FLAT)
            w.x.ensure_gpu(c, Self.T * PK_D)
            w.h.ensure_gpu(c, Self.T * PK_D)
            w.y.ensure_gpu(c, Self.T * PK_D)
            w.ff.ensure_gpu(c, Self.T * PK_FF)
            w.q.ensure_gpu(c, Self.T * PK_D)
            w.k.ensure_gpu(c, Self.T * PK_D)
            w.v.ensure_gpu(c, Self.T * PK_D)
            w.att.ensure_gpu(c, Self.T * PK_D)
            w.relk.ensure_gpu(c, Self.P * PK_D)
            w.relk_all.ensure_gpu(c, PK_LAYERS * Self.P * PK_D)
            w.ln_xhat = Tensor.alloc_gpu(c, Self.T * PK_D)
            w.ln_inv = Tensor.alloc_gpu(c, Self.T)
            w.glu.ensure_gpu(c, Self.T * 2 * PK_D)
            w.conv.ensure_gpu(c, Self.T * PK_D)
            w.proj.ensure_gpu(c, Self.T * PK_DEC)
            w.scores_dev = Tensor.alloc_gpu(c, PK_HEADS * Self.T * Self.T)
            w.pos.upload_resident(c)
        return w^


# ── elementwise pieces ─────────────────────────────────────────────────────


def _silu(mut t: Tensor, n: Int):
    for i in range(n):
        var v = t.data[i]
        t.data[i] = v / (1.0 + exp(-v))


def _relu_rows(mut t: Tensor, row: Int, valid: Int, rows: Int):
    """ReLU on the first `valid` rows of `[rows, row]`, zero on the others."""
    for i in range(valid * row):
        if t.data[i] < 0.0:
            t.data[i] = 0.0
    for i in range(valid * row, rows * row):
        t.data[i] = 0.0


def _add_scaled(mut x: Tensor, ref y: Tensor, n: Int, scale: Float32):
    for i in range(n):
        x.data[i] += scale * y.data[i]


# ── subsampling convolutions (channel-last, CPU) ───────────────────────────


def _conv3x3_s2_first(
    ref src: Tensor, src_h: Int, src_w: Int,
    ref w: Tensor, ref b: Tensor,
    mut dst: Tensor, dst_h: Int, dst_w: Int, valid: Int,
):
    """`src [src_h, src_w]` (one channel) -> `dst [dst_h, dst_w, C]`: 3x3,
    stride 2, padding 1, then ReLU. Rows beyond `valid` are zero."""
    comptime C = PK_SUB_CH
    var p = List[Float32](length=9, fill=0.0)
    for t in range(valid):
        for x in range(dst_w):
            for kh in range(3):
                var sy = 2 * t - 1 + kh
                for kw in range(3):
                    var sx = 2 * x - 1 + kw
                    var inside = sy >= 0 and sy < src_h and sx >= 0 and sx < src_w
                    p[kh * 3 + kw] = src.data[sy * src_w + sx] if inside else Float32(0.0)
            var base = (t * dst_w + x) * C
            for c in range(C):
                var acc = b.data[c]
                for j in range(9):
                    acc += p[j] * w.data[c * 9 + j]
                dst.data[base + c] = acc if acc > 0.0 else Float32(0.0)
    for i in range(valid * dst_w * C, dst_h * dst_w * C):
        dst.data[i] = 0.0


def _dw3x3_s2(
    ref src: Tensor, src_h: Int, src_w: Int,
    ref w: Tensor, ref b: Tensor,
    mut dst: Tensor, dst_h: Int, dst_w: Int, valid: Int,
):
    """Depthwise 3x3, stride 2, padding 1, channel-last: `src [src_h, src_w, C]`
    -> `dst [dst_h, dst_w, C]`. Rows beyond `valid` are zero."""
    comptime C = PK_SUB_CH
    for t in range(valid):
        for x in range(dst_w):
            var base = (t * dst_w + x) * C
            for c in range(C):
                dst.data[base + c] = b.data[c]
            for kh in range(3):
                var sy = 2 * t - 1 + kh
                if sy < 0 or sy >= src_h:
                    continue
                for kw in range(3):
                    var sx = 2 * x - 1 + kw
                    if sx < 0 or sx >= src_w:
                        continue
                    var sbase = (sy * src_w + sx) * C
                    var j = kh * 3 + kw
                    for c in range(C):
                        dst.data[base + c] += src.data[sbase + c] * w.data[c * 9 + j]
    for i in range(valid * dst_w * C, dst_h * dst_w * C):
        dst.data[i] = 0.0


def pk_subsample[F_MAX: Int, target: StaticString = "cpu"](
    mut m: ParakeetTDT, mut ws: PkEncoderWork[F_MAX], n_mel: Int,
    ctx: Optional[DeviceContext] = None,
) raises -> Int:
    """`ws.feats` (`[F_MAX, 128]` on the host, `n_mel` valid rows, zero beyond)
    -> `ws.x` (`[T, 1024]`). Returns the valid encoder frames."""
    comptime W = PkEncoderWork[F_MAX]
    comptime C = PK_SUB_CH
    comptime B1 = W.F2 * W.W2
    comptime B2 = W.T * PK_SUB_FREQ
    if n_mel < 1 or n_mel > F_MAX:
        raise Error("pk_subsample: n_mel out of range for the compiled maximum")
    var len1 = pk_sub_len(n_mel)
    var len2 = pk_sub_len(len1)
    var len3 = pk_sub_len(len2)

    comptime if target == "cpu":
        _conv3x3_s2_first(
            ws.feats, F_MAX, PK_MELS, m.sub.conv0.weight.val, m.sub.conv0.bias.val,
            ws.a1, W.F1, W.W1, len1,
        )
        _dw3x3_s2(
            ws.a1, W.F1, W.W1, m.sub.dw1.weight.val, m.sub.dw1.bias.val,
            ws.b1, W.F2, W.W2, len2,
        )
        m.sub.pw1.forward["cpu", B1](TensorRefs[1](ws.b1), ws.c1)
        _relu_rows(ws.c1, W.W2 * C, len2, W.F2)
        _dw3x3_s2(
            ws.c1, W.F2, W.W2, m.sub.dw2.weight.val, m.sub.dw2.bias.val,
            ws.b2, W.T, PK_SUB_FREQ, len3,
        )
        m.sub.pw2.forward["cpu", B2](TensorRefs[1](ws.b2), ws.c2)
        _relu_rows(ws.c2, PK_SUB_FREQ * C, len3, W.T)
        # [T, freq, C] -> [T, C * freq]: the reference flattens channel-major.
        for t in range(W.T):
            for f in range(PK_SUB_FREQ):
                var src = (t * PK_SUB_FREQ + f) * C
                for c in range(C):
                    ws.flat.data[t * PK_SUB_FLAT + c * PK_SUB_FREQ + f] = ws.c2.data[src + c]
        m.sub.linear.forward["cpu", W.T](TensorRefs[1](ws.flat), ws.x)
    else:
        var c = ctx.value()
        ws.feats.upload_resident(c)
        comptime N1 = W.F1 * W.W1 * C
        c.enqueue_function[
            conv3x3_s2_first_kernel[F_MAX, PK_MELS, W.F1, W.W1, C]
        ](
            ws.feats.tt["gpu", F_MAX, PK_MELS](),
            m.sub.conv0.weight.val.tt["gpu", C, 9](),
            m.sub.conv0.bias.val.tt["gpu", C](),
            ws.a1.tt["gpu", W.F1, W.W1, C](),
            Int64(len1),
            grid_dim=(N1 + TPB - 1) // TPB, block_dim=TPB,
        )
        c.enqueue_function[dw3x3_s2_kernel[W.F1, W.W1, W.F2, W.W2, C]](
            ws.a1.tt["gpu", W.F1, W.W1, C](),
            m.sub.dw1.weight.val.tt["gpu", C, 9](),
            m.sub.dw1.bias.val.tt["gpu", C](),
            ws.b1.tt["gpu", W.F2, W.W2, C](),
            Int64(len2),
            grid_dim=(B1 * C + TPB - 1) // TPB, block_dim=TPB,
        )
        m.sub.pw1.forward["gpu", B1](TensorRefs[1](ws.b1), ws.c1, ctx)
        c.enqueue_function[relu_rows_kernel[W.F2, W.W2 * C]](
            ws.c1.tt["gpu", W.F2, W.W2 * C](), Int64(len2),
            grid_dim=(B1 * C + TPB - 1) // TPB, block_dim=TPB,
        )
        c.enqueue_function[dw3x3_s2_kernel[W.F2, W.W2, W.T, PK_SUB_FREQ, C]](
            ws.c1.tt["gpu", W.F2, W.W2, C](),
            m.sub.dw2.weight.val.tt["gpu", C, 9](),
            m.sub.dw2.bias.val.tt["gpu", C](),
            ws.b2.tt["gpu", W.T, PK_SUB_FREQ, C](),
            Int64(len3),
            grid_dim=(B2 * C + TPB - 1) // TPB, block_dim=TPB,
        )
        m.sub.pw2.forward["gpu", B2](TensorRefs[1](ws.b2), ws.c2, ctx)
        c.enqueue_function[relu_rows_kernel[W.T, PK_SUB_FLAT]](
            ws.c2.tt["gpu", W.T, PK_SUB_FLAT](), Int64(len3),
            grid_dim=(B2 * C + TPB - 1) // TPB, block_dim=TPB,
        )
        c.enqueue_function[flatten_tfc_kernel[W.T, PK_SUB_FREQ, C]](
            ws.c2.tt["gpu", W.T, PK_SUB_FREQ, C](),
            ws.flat.tt["gpu", W.T, PK_SUB_FLAT](),
            grid_dim=(B2 * C + TPB - 1) // TPB, block_dim=TPB,
        )
        m.sub.linear.forward["gpu", W.T](TensorRefs[1](ws.flat), ws.x, ctx)
    return len3


# ── one conformer block ────────────────────────────────────────────────────


def _rel_attention(
    ref q: Tensor, ref k: Tensor, ref v: Tensor, ref relk: Tensor,
    ref bias_u: Tensor, ref bias_v: Tensor,
    mut out: Tensor, mut scores: List[Float32], t_max: Int, valid: Int,
    relk_off: Int,
):
    """Relative-position multi-head attention over the first `valid` of
    `t_max` frames; `out` rows beyond `valid` are zero. `relk` holds every
    layer's projected table; this layer's starts at element `relk_off`."""
    comptime D = PK_D
    comptime HD = PK_HEAD_DIM
    var scale = Float32(1.0 / sqrt(Float64(HD)))
    var qu = List[Float32](length=HD, fill=0.0)
    var qv = List[Float32](length=HD, fill=0.0)
    for i in range(t_max * D):
        out.data[i] = 0.0
    for h in range(PK_HEADS):
        var ho = h * HD
        for i in range(valid):
            for d in range(HD):
                var qd = q.data[i * D + ho + d]
                qu[d] = qd + bias_u.data[ho + d]
                qv[d] = qd + bias_v.data[ho + d]
            var top = Float32(-3.0e38)
            for j in range(valid):
                var kb = j * D + ho
                var rb = relk_off + (j - i + t_max - 1) * D + ho
                var ac = Float32(0.0)
                var bd = Float32(0.0)
                for d in range(HD):
                    ac += qu[d] * k.data[kb + d]
                    bd += qv[d] * relk.data[rb + d]
                var s = ac * scale + bd * scale
                scores[j] = s
                if s > top:
                    top = s
            var total = Float32(0.0)
            for j in range(valid):
                scores[j] = exp(scores[j] - top)
                total += scores[j]
            var ob = i * D + ho
            for j in range(valid):
                var wj = scores[j] / total
                var vb = j * D + ho
                for d in range(HD):
                    out.data[ob + d] += wj * v.data[vb + d]


def _glu_rows(ref src: Tensor, mut dst: Tensor, valid: Int, rows: Int):
    """`dst[t, c] = src[t, c] * sigmoid(src[t, D + c])` on the valid rows, zero
    on the others (the mask the depthwise conv needs)."""
    comptime D = PK_D
    for t in range(valid):
        for c in range(D):
            var g = src.data[t * 2 * D + D + c]
            dst.data[t * D + c] = src.data[t * 2 * D + c] / (1.0 + exp(-g))
    for i in range(valid * D, rows * D):
        dst.data[i] = 0.0


def _dw1d_bn_silu(
    ref src: Tensor, ref w: Tensor,
    ref gamma: Tensor, ref beta: Tensor, ref mean: Tensor, ref variance: Tensor,
    mut dst: Tensor, valid: Int, rows: Int,
):
    """Depthwise Conv1d over time (kernel 9, padding 4, no bias), BatchNorm with
    the running statistics, SiLU. `src` rows beyond `valid` are zero; so are
    `dst`'s."""
    comptime D = PK_D
    comptime K = PK_CONV_K
    for t in range(valid):
        var ob = t * D
        for c in range(D):
            dst.data[ob + c] = 0.0
        for j in range(K):
            var tt = t - K // 2 + j
            if tt < 0 or tt >= valid:
                continue
            var sb = tt * D
            for c in range(D):
                dst.data[ob + c] += src.data[sb + c] * w.data[c * K + j]
        for c in range(D):
            var n = (dst.data[ob + c] - mean.data[c]) / sqrt(
                variance.data[c] + Float32(PK_BN_EPS)
            ) * gamma.data[c] + beta.data[c]
            dst.data[ob + c] = n / (1.0 + exp(-n))
    for i in range(valid * D, rows * D):
        dst.data[i] = 0.0


def _silu_any[target: StaticString, N: Int](
    mut t: Tensor, n: Int, ctx: Optional[DeviceContext]
) raises:
    """SiLU in place on the first `n` of `N` elements."""
    comptime if target == "cpu":
        _silu(t, n)
    else:
        ctx.value().enqueue_function[silu_kernel[N]](
            t.tt["gpu", N](), grid_dim=(n + TPB - 1) // TPB, block_dim=TPB,
        )


def _add_scaled_any[target: StaticString, N: Int](
    mut x: Tensor, mut y: Tensor, scale: Float32, n: Int,
    ctx: Optional[DeviceContext],
) raises:
    """`x += scale * y` on the first `n` of `N` elements."""
    comptime if target == "cpu":
        _add_scaled(x, y, n, scale)
    else:
        ctx.value().enqueue_function[add_scaled_kernel[N]](
            x.tt["gpu", N](), y.tt["gpu", N](), Scalar[DT](scale),
            grid_dim=(n + TPB - 1) // TPB, block_dim=TPB,
        )


def _ln_rows_cpu(
    ref x: Tensor, mut dst: Tensor, ref gamma: Tensor, ref beta: Tensor, rows: Int
):
    """`LayerNorm` (biased variance, eps 1e-5) over the first `rows` rows of
    `[*, 1024]`: the leaf's arithmetic with a runtime row count."""
    comptime D = PK_D
    for r in range(rows):
        var base = r * D
        var total = Float32(0.0)
        for d in range(D):
            total += x.data[base + d]
        var mean = total / Float32(D)
        var sq = Float32(0.0)
        for d in range(D):
            var diff = x.data[base + d] - mean
            sq += diff * diff
        var inv = Float32(1.0) / sqrt(sq / Float32(D) + LN_EPS)
        for d in range(D):
            dst.data[base + d] = (x.data[base + d] - mean) * inv * gamma.data[d] + beta.data[d]


def _ln_any[target: StaticString, B: Int](
    mut ln: LayerNorm[PK_D], mut x: Tensor, mut dst: Tensor,
    mut xhat: Tensor, mut inv: Tensor, rows: Int, ctx: Optional[DeviceContext],
) raises:
    """`dst = ln(x)` over the first `rows` rows of `[B, 1024]`. On a device:
    nn's LayerNorm kernel (one block per row) on the workspace's shared
    scratch (module docstring)."""
    comptime if target == "cpu":
        _ln_rows_cpu(x, dst, ln.gamma.val, ln.beta.val, rows)
    else:
        ctx.value().enqueue_function[
            _layer_norm_forward_kernel[B, PK_D, DT, LN_EPS]
        ](
            x.tt["gpu", B, PK_D](),
            dst.tt["gpu", B, PK_D](),
            ln.gamma.val.tt["gpu", PK_D](),
            ln.beta.val.tt["gpu", PK_D](),
            xhat.tt["gpu", B, PK_D](),
            inv.tt["gpu", B](),
            grid_dim=rows, block_dim=LN_TPB,
        )


def _gemm_any[target: StaticString, B: Int, IN: Int, OUT: Int](
    mut lin: Linear[IN, OUT], mut x: Tensor, mut dst: Tensor, rows: Int,
    ctx: Optional[DeviceContext],
) raises:
    """`dst[:rows] = x[:rows] @ W` for a `Linear` whose bias is ZERO
    (`TN_ZEROS` in the name map), `x` and `dst` being `[B, IN]` / `[B, OUT]`.

    Where the GEMM takes its row count at run time — Metal, and Accelerate on
    the macOS CPU — only the `rows` valid rows are computed, and without the
    leaf's zero bias add. Elsewhere (NVIDIA, whose GEMM tiles rows at compile
    time; a CPU without Accelerate) the leaf's own forward over all `B` rows."""
    comptime if target == "gpu" and not default_accelerator().is_nvidia_gpu():
        var c = ctx.value()
        dst.ensure_gpu(c, B * OUT)
        var av = TileTensor(x.dev.value(), row_major(rows, IN))
        var bv = TileTensor(lin.weight.val.dev.value(), row_major(IN, OUT))
        var ov = TileTensor(dst.dev.value(), row_major(rows, OUT))
        max_matmul[target="gpu"](ov, av, bv, c)
    elif target == "cpu" and CompilationTarget.is_macos():
        dst.ensure(B * OUT)
        gemm_cpu[False, False](
            rows, OUT, IN, 1.0, x, IN, lin.weight.val, OUT, 0.0, dst, OUT
        )
    else:
        lin.forward[target, B](TensorRefs[1](x), dst, ctx)


def pk_prepare_positions[F_MAX: Int, target: StaticString = "cpu"](
    mut m: ParakeetTDT, mut ws: PkEncoderWork[F_MAX],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Fill `ws.relk_all` with every layer's `rel_k(pos)`. Runs once (the
    blocks call it while `ws.relk_ready` is False)."""
    comptime P = PkEncoderWork[F_MAX].P
    comptime N = P * PK_D
    for layer in range(PK_LAYERS):
        _gemm_any[target, P, PK_D, PK_D](m.layers[layer].rel_k, ws.pos, ws.relk, P, ctx)
        comptime if target == "cpu":
            for i in range(N):
                ws.relk_all.data[layer * N + i] = ws.relk.data[i]
        else:
            var slot = ws.relk_all.dev.value().create_sub_buffer[DT](layer * N, N)
            ctx.value().enqueue_copy(slot, ws.relk.dev.value())
    ws.relk_ready = True


def _block[F_MAX: Int, target: StaticString](
    mut l: PkConformerLayer, mut ws: PkEncoderWork[F_MAX], layer: Int,
    valid: Int, ctx: Optional[DeviceContext],
) raises:
    comptime T = PkEncoderWork[F_MAX].T
    comptime P = PkEncoderWork[F_MAX].P
    comptime D = PK_D
    comptime N = T * D

    # 1. first feed-forward, half-weighted
    _ln_any[target, T](l.norm_ff1, ws.x, ws.h, ws.ln_xhat, ws.ln_inv, valid, ctx)
    _gemm_any[target, T, D, PK_FF](l.ff1_a, ws.h, ws.ff, valid, ctx)
    _silu_any[target, T * PK_FF](ws.ff, valid * PK_FF, ctx)
    _gemm_any[target, T, PK_FF, D](l.ff1_b, ws.ff, ws.y, valid, ctx)
    _add_scaled_any[target, N](ws.x, ws.y, 0.5, valid * D, ctx)

    # 2. relative-position self-attention
    _ln_any[target, T](l.norm_att, ws.x, ws.h, ws.ln_xhat, ws.ln_inv, valid, ctx)
    _gemm_any[target, T, D, D](l.q, ws.h, ws.q, valid, ctx)
    _gemm_any[target, T, D, D](l.k, ws.h, ws.k, valid, ctx)
    _gemm_any[target, T, D, D](l.v, ws.h, ws.v, valid, ctx)
    comptime if target == "cpu":
        _rel_attention(
            ws.q, ws.k, ws.v, ws.relk_all, l.bias_u.val, l.bias_v.val,
            ws.att, ws.scores, T, valid, layer * P * D,
        )
    else:
        var c = ctx.value()
        comptime NS = PK_HEADS * T * T
        c.enqueue_function[rel_scores_kernel[T, P, D, PK_HEADS, PK_LAYERS]](
            ws.q.tt["gpu", T, D](),
            ws.k.tt["gpu", T, D](),
            ws.relk_all.tt["gpu", PK_LAYERS, P, D](),
            l.bias_u.val.tt["gpu", D](),
            l.bias_v.val.tt["gpu", D](),
            ws.scores_dev.tt["gpu", PK_HEADS, T, T](),
            Int64(valid),
            Int64(layer),
            grid_dim=(NS + TPB - 1) // TPB, block_dim=TPB,
        )
        c.enqueue_function[rel_apply_kernel[T, D, PK_HEADS]](
            ws.scores_dev.tt["gpu", PK_HEADS, T, T](),
            ws.v.tt["gpu", T, D](),
            ws.att.tt["gpu", T, D](),
            Int64(valid),
            grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )
    _gemm_any[target, T, D, D](l.o, ws.att, ws.y, valid, ctx)
    _add_scaled_any[target, N](ws.x, ws.y, 1.0, valid * D, ctx)

    # 3. convolution module
    _ln_any[target, T](l.norm_conv, ws.x, ws.h, ws.ln_xhat, ws.ln_inv, valid, ctx)
    _gemm_any[target, T, D, 2 * D](l.pw1, ws.h, ws.glu, valid, ctx)
    comptime if target == "cpu":
        _glu_rows(ws.glu, ws.h, valid, T)
        _dw1d_bn_silu(
            ws.h, l.dw.weight.val, l.bn.gamma.val, l.bn.beta.val,
            l.bn.running_mean.t, l.bn.running_var.t, ws.conv, valid, T,
        )
    else:
        var c = ctx.value()
        c.enqueue_function[glu_rows_kernel[T, D]](
            ws.glu.tt["gpu", T, 2 * D](), ws.h.tt["gpu", T, D](), Int64(valid),
            grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )
        c.enqueue_function[
            dw1d_bn_silu_kernel[T, D, PK_CONV_K, Scalar[DT](PK_BN_EPS)]
        ](
            ws.h.tt["gpu", T, D](),
            l.dw.weight.val.tt["gpu", D, PK_CONV_K](),
            l.bn.gamma.val.tt["gpu", D](),
            l.bn.beta.val.tt["gpu", D](),
            l.bn.running_mean.t.tt["gpu", D](),
            l.bn.running_var.t.tt["gpu", D](),
            ws.conv.tt["gpu", T, D](),
            Int64(valid),
            grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )
    _gemm_any[target, T, D, D](l.pw2, ws.conv, ws.y, valid, ctx)
    _add_scaled_any[target, N](ws.x, ws.y, 1.0, valid * D, ctx)

    # 4. second feed-forward, half-weighted
    _ln_any[target, T](l.norm_ff2, ws.x, ws.h, ws.ln_xhat, ws.ln_inv, valid, ctx)
    _gemm_any[target, T, D, PK_FF](l.ff2_a, ws.h, ws.ff, valid, ctx)
    _silu_any[target, T * PK_FF](ws.ff, valid * PK_FF, ctx)
    _gemm_any[target, T, PK_FF, D](l.ff2_b, ws.ff, ws.y, valid, ctx)
    _add_scaled_any[target, N](ws.x, ws.y, 0.5, valid * D, ctx)

    # 5. output norm, back into `x`
    _ln_any[target, T](l.norm_out, ws.x, ws.h, ws.ln_xhat, ws.ln_inv, valid, ctx)
    comptime if target == "cpu":
        for i in range(valid * D):
            ws.x.data[i] = ws.h.data[i]
    else:
        # `x` and `h` exchange device buffers: no copy, no launch.
        var hd = ws.h.dev.value()
        ws.h.dev = ws.x.dev.value()
        ws.x.dev = hd


def pk_block[F_MAX: Int, target: StaticString = "cpu"](
    mut m: ParakeetTDT, mut ws: PkEncoderWork[F_MAX], layer: Int, valid: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Apply conformer block `layer` to `ws.x` in place (`valid` of `T` frames)."""
    if not ws.relk_ready:
        pk_prepare_positions[F_MAX, target](m, ws, ctx)
    _block[F_MAX, target](m.layers[layer], ws, layer, valid, ctx)


def pk_encode[F_MAX: Int, target: StaticString = "cpu"](
    mut m: ParakeetTDT, mut ws: PkEncoderWork[F_MAX], n_mel: Int,
    ctx: Optional[DeviceContext] = None,
) raises -> Int:
    """The whole encoder: `ws.feats` -> `ws.x` (`[T, 1024]`) and `ws.proj`
    (`[T, 640]`, what the joint network reads), both on `target`. Returns the
    valid frames."""
    comptime T = PkEncoderWork[F_MAX].T
    if not ws.relk_ready:
        pk_prepare_positions[F_MAX, target](m, ws, ctx)
    var valid = pk_subsample[F_MAX, target](m, ws, n_mel, ctx)
    for layer in range(PK_LAYERS):
        _block[F_MAX, target](m.layers[layer], ws, layer, valid, ctx)
    m.enc_proj.forward[target, T](TensorRefs[1](ws.x), ws.proj, ctx)
    return valid
