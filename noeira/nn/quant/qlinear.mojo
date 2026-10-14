"""QLinear: a frozen Q4_0 linear. A forward and a grad_input; no weight gradient.

    y[T, OUT] = x[T, IN] @ deq(W)ᵀ        W: Q4_0, [OUT, IN] (torch's orientation)
    dx[T, IN] = dy[T, OUT] @ deq(W)

`QLinear[IN, OUT]` is a `Module` with fp32 activations, so it takes the place
of a frozen `Linear[IN, OUT]` in a model whose activations are fp32 (SmolVLA).
It has no bias (the text towers it replaces have none) and no `Param`: an
optimizer walk never sees it, and neither does a v3 checkpoint. A quantized
model is a deployment artifact, saved as Q4_0 blocks (`blocks()`, U8 in a
safetensors file), not in the fp32 checkpoint.

## Paths

| Where | Forward | grad_input |
|---|---|---|
| NVIDIA (`NN_Q4_PATH=auto`) | MAX's int4 tensor-core GEMM, `multistage_gemm_q`, on the repacked blocks | dequantize, fp32 cuBLAS |
| NVIDIA (`NN_Q4_PATH=deq`), Metal | dequantize, fp32 GEMM | dequantize, fp32 GEMM |
| CPU | dequantize, `gemm_cpu` | dequantize, `gemm_cpu` |

MAX's int4 GEMM takes and returns bf16, so the NVIDIA forward casts x to
bf16 (fused with the K padding below) and the output back to fp32 (fused with
the N slice): three launches. Its numbers are therefore a bf16 GEMM's (~4e-3
per element), not the fp32 path's; the gate compares within 1e-2.

## Padding (NVIDIA)

MAX's repack (`gpu_qint4_repack_Q4_0`) instantiates only when N % 128 == 0
and K % 1024 == 0 (its masked tile does not compile on a nested layout), and
a K that is not a multiple of 1024 would also leave the warp-K split of
noeira's configs uneven. So the blocks
are padded once, at load, to [N_PAD, K_PAD] with zero groups (d = 0, which
decode to 0 whatever the nibbles), x is padded with zero columns, and the
extra output columns are never read. SmolVLA's text layer pays +14.7 % bytes.

## Configs

MAX's tuned configs do not compile in 26.6 (Kev's notes), so the tile is
noeira's, picked by `NN_Q4_CFG`:

    small2  64x128x32, warp-K 2, 4 stages   51.2 KB  (default)
    small   64x128x32, warp-K 4, 3 stages   76.8 KB
    small4  64x128x32, warp-K 4, 4 stages  102.4 KB  (Kev's on the Orin; refused on consumer
                                                      Blackwell: 101,376 B per block)
    large   128x128x32, warp-K 1, 5 stages  65.5 KB

small2 is the fastest or tied at every SmolVLA text shape at the prefill's 135
tokens on the Orin (the GEMM alone: 37 / 25 / 82 / 92 us against 42 / 29 /
104 / 94 for small, `benchmarks/quant/bench_qlinear_smolvla.mojo`), and tied
with small on an RTX 5090 (14 us).

## Inference-only builds keep only the repacked blocks

With `-D NN_INFERENCE_ONLY` on NVIDIA nothing can take a vjp, and the forward
reads only the repacked blocks, so `set_blocks` never keeps the logical ones:
no host copy, no device copy. On the Orin that is not a nicety. Memory freed
after load is not handed back to the board (the process keeps it), so a copy
dropped later by `keep_only_repacked` is a copy the process still holds.

## The dequantized weight is a scratch

The dequantize paths write deq(W) [OUT, IN] fp32 into `deq` right before the
GEMM that reads it. Layers run one at a time, so a model shares ONE buffer
across all of them (`share_scratch`), sized for the largest; a QLinear without
one allocates its own on first use. Either way nothing is allocated after the
first call (capture-safe). The NVIDIA forward never touches it: an inference
deploy on the Orin holds the repacked blocks and nothing else
(`keep_only_repacked`).
"""

from std.memory import unsafe_memcpy
from std.sys import has_nvidia_gpu_accelerator
from std.sys.defines import get_defined_string
from std.utils.index import Index
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Coord, Idx, TileTensor, row_major
from linalg.utils_gpu import MatmulConfig
from quantization.qmatmul_gpu import gpu_qint4_repack_Q4_0, multistage_gemm_q

from noeira.nn.constants import DT
from ..core.param import NN_INFERENCE_ONLY
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.mm import mm
from ..core.cublas_gemm import cublas_gemm, cublas_tf32
from ..core.cpu_gemm import gemm_cpu
from .q4_0 import (
    Q4_GROUP,
    Q4_BLOCK,
    q4_0_row_bytes,
    rtn_q4_0_blocks,
    dequantize_q4_0,
    dequant_q4_0_kernel,
)


comptime Q4_PATH = get_defined_string["NN_Q4_PATH", "auto"]()
"""`auto`: MAX's int4 GEMM for the NVIDIA forward. `deq`: the dequantized
fp32 GEMM on every backend (A/B runs, and the int4 path's reference)."""
comptime Q4_CFG = get_defined_string["NN_Q4_CFG", "small2"]()
"""The int4 GEMM's tile config (module docstring)."""
comptime Q4_INT4_GEMM = has_nvidia_gpu_accelerator() and Q4_PATH != "deq"
"""The GPU forward runs MAX's int4 GEMM (else it dequantizes)."""

comptime _QCfg = MatmulConfig[
    DType.bfloat16, DType.uint8, DType.bfloat16, True
]
comptime Q4_SMALL = _QCfg(
    block_tile_shape=Index(64, 128, 32),
    warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=3,
    num_k_partitions=1,
    num_warp_k_partitions=4,
)
comptime Q4_SMALL2 = _QCfg(
    block_tile_shape=Index(64, 128, 32),
    warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=4,
    num_k_partitions=1,
    num_warp_k_partitions=2,
)
comptime Q4_SMALL4 = _QCfg(
    block_tile_shape=Index(64, 128, 32),
    warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=4,
    num_k_partitions=1,
    num_warp_k_partitions=4,
)
comptime Q4_LARGE = _QCfg(
    block_tile_shape=Index(128, 128, 32),
    warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=5,
    num_k_partitions=1,
    num_warp_k_partitions=1,
)


@always_inline
def _round_up(v: Int, to: Int) -> Int:
    return ((v + to - 1) // to) * to


@always_inline
def _blocks(n: Int) -> Int:
    return (n + 255) // 256


def _cast_pad_kernel(
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[BFloat16, MutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    cols_pad_arg: Int64,
):
    """`dst[r, c] = bf16(src[r, c])` for c < cols, 0 for the padded columns:
    fp32 [rows, cols] -> bf16 [rows, cols_pad]."""
    var cols = Int(cols_arg)
    var cols_pad = Int(cols_pad_arg)
    var i = Int(global_idx.x)
    if i < Int(rows_arg) * cols_pad:
        var r = i // cols_pad
        var c = i % cols_pad
        if c < cols:
            dst[unsafe_offset=i] = src[unsafe_offset=r * cols + c].cast[
                DType.bfloat16
            ]()
        else:
            dst[unsafe_offset=i] = BFloat16(0)


def _slice_cast_kernel(
    src: Pointer[BFloat16, MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    cols_pad_arg: Int64,
):
    """`dst[r, c] = fp32(src[r, c])`: bf16 [rows, cols_pad] -> fp32
    [rows, cols], the padded columns dropped."""
    var cols = Int(cols_arg)
    var cols_pad = Int(cols_pad_arg)
    var i = Int(global_idx.x)
    if i < Int(rows_arg) * cols:
        var r = i // cols
        var c = i % cols
        dst[unsafe_offset=i] = src[unsafe_offset=r * cols_pad + c].cast[
            DType.float32
        ]()


def _upload_bytes(
    c: DeviceContext, ref data: List[UInt8], n: Int
) raises -> DeviceBuffer[DType.uint8]:
    """`n` bytes from the host list into a new device buffer, straight from
    the list (no pinned staging copy is kept)."""
    var buf = c.enqueue_create_buffer[DType.uint8](n)
    c.enqueue_copy(
        buf, rebind[Pointer[UInt8, ImmutAnyOrigin]](data.unsafe_ptr())
    )
    c.synchronize()
    return buf


struct QLinear[IN_: Int, OUT_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_
    comptime ROW_BYTES = (Self.IN_ // Q4_GROUP) * Q4_BLOCK
    comptime BYTES = Self.OUT_ * Self.ROW_BYTES
    """The Q4_0 blocks: [OUT, IN / 32 · 18]."""
    comptime N_PAD = _round_up(Self.OUT_, 128)
    comptime K_PAD = _round_up(Self.IN_, 1024)
    comptime PAD_ROW_BYTES = (Self.K_PAD // Q4_GROUP) * Q4_BLOCK
    comptime PAD_BYTES = Self.N_PAD * Self.PAD_ROW_BYTES
    comptime MAX_Q4 = Q4_INT4_GEMM
    """The forward runs MAX's int4 GEMM (NVIDIA, `NN_Q4_PATH=auto`)."""

    @staticmethod
    def display_label() -> String:
        return String("QLinear")

    var q: TensorImpl[DType.uint8]
    """The Q4_0 blocks [OUT, IN / 32 · 18]: on the host (`data`, what a file
    holds) and, on a GPU, on the device for the dequantize paths."""
    var qpk: TensorImpl[DType.uint8]
    """NVIDIA: the blocks padded to [N_PAD, K_PAD] and repacked by MAX for its
    int4 GEMM. Device only."""
    var deq: Tensor
    """deq(W) [OUT, IN] fp32, written before each GEMM that reads it: the
    caller's shared scratch (`share_scratch`) or this layer's own."""
    var xb: TensorImpl[DType.bfloat16]
    """NVIDIA forward: x cast to bf16 and zero-padded, [T, K_PAD]."""
    var yb: TensorImpl[DType.bfloat16]
    """NVIDIA forward: the int4 GEMM's bf16 output, [T, N_PAD]."""
    var _loaded: Bool

    def __init__(out self):
        comptime assert Self.IN_ % Q4_GROUP == 0, (
            "QLinear: IN must be a multiple of 32 (one Q4_0 group)"
        )
        self.q = TensorImpl[DType.uint8]()
        self.qpk = TensorImpl[DType.uint8]()
        self.deq = Tensor()
        self.xb = TensorImpl[DType.bfloat16]()
        self.yb = TensorImpl[DType.bfloat16]()
        self._loaded = False

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """An empty layer: a QLinear's weights come from `set_weight` (an fp32
        weight, quantized here) or `set_blocks` (Q4_0 blocks), never from an
        initializer."""
        return Self()

    # ── weights ────────────────────────────────────────────────────────

    def set_weight[
        target: StaticString
    ](mut self, ref w: List[Float32], ctx: Optional[DeviceContext] = None) raises:
        """Quantize an fp32 weight [OUT, IN] (torch's orientation, row-major)
        to Q4_0 by round-to-nearest (`quantize_rtn_q4_0`) and load it."""
        self.set_blocks[target](rtn_q4_0_blocks(w, Self.OUT_, Self.IN_), ctx)

    def set_blocks[
        target: StaticString
    ](mut self, var blocks: List[UInt8], ctx: Optional[DeviceContext] = None) raises:
        """Load Q4_0 blocks [OUT, IN / 32 · 18]. On a GPU they go to the device,
        and on NVIDIA they are also padded and repacked for MAX's GEMM."""
        if len(blocks) != Self.BYTES:
            raise Error(
                "QLinear[" + String(Self.IN_) + ", " + String(Self.OUT_)
                + "].set_blocks: " + String(len(blocks)) + " bytes, expected "
                + String(Self.BYTES)
            )
        # Inference-only on NVIDIA: the repacked blocks are all that is kept.
        comptime REPACKED_ONLY = (
            target == "gpu" and Self.MAX_Q4 and NN_INFERENCE_ONLY
        )
        comptime if target == "gpu":
            var c = ctx.value()
            comptime if not REPACKED_ONLY:
                self.q.dev = _upload_bytes(c, blocks, Self.BYTES)
            comptime if Self.MAX_Q4:
                var pad = List[UInt8](length=Self.PAD_BYTES, fill=0)
                for r in range(Self.OUT_):
                    unsafe_memcpy(
                        dest=pad.unsafe_ptr().unsafe_offset(
                            r * Self.PAD_ROW_BYTES
                        ),
                        src=blocks.unsafe_ptr().unsafe_offset(
                            r * Self.ROW_BYTES
                        ),
                        count=Self.ROW_BYTES,
                    )
                var pad_d = _upload_bytes(c, pad, Self.PAD_BYTES)
                self.qpk.dev = c.enqueue_create_buffer[DType.uint8](
                    Self.PAD_BYTES
                )
                self.qpk.n = Self.PAD_BYTES
                var src_tt = TileTensor(
                    pad_d,
                    row_major(Coord(Idx[Self.N_PAD], Idx[Self.PAD_ROW_BYTES])),
                )
                var dst_tt = TileTensor(
                    self.qpk.dev.value(),
                    row_major(Coord(Idx[Self.N_PAD], Idx[Self.PAD_ROW_BYTES])),
                )
                gpu_qint4_repack_Q4_0["gpu"](src_tt.as_immut(), dst_tt, c)
                c.synchronize()
        comptime if not REPACKED_ONLY:
            self.q.data = blocks^
            self.q.n = Self.BYTES
        self._loaded = True

    def blocks(self) -> List[UInt8]:
        """A copy of the Q4_0 blocks [OUT, IN / 32 · 18] (for a file)."""
        return self.q.data.copy()

    def dequantized(self) raises -> List[Float32]:
        """deq(W) [OUT, IN] fp32 on the host, from the blocks (for gates)."""
        return dequantize_q4_0(self.q.data, Self.OUT_, Self.IN_)

    def share_scratch(mut self, buf: DeviceBuffer[DT], n: Int) raises:
        """Use `buf` (`n` fp32 values, at least OUT · IN) as the dequantize
        scratch instead of a buffer of this layer's own. Every QLinear of a
        model can share one: they run one at a time."""
        if n < Self.W_SIZE:
            raise Error(
                "QLinear.share_scratch: " + String(n) + " values for a ["
                + String(Self.OUT_) + ", " + String(Self.IN_) + "] weight"
            )
        self.deq.dev = buf
        self.deq.n = n

    def keep_only_repacked(mut self) raises:
        """NVIDIA inference: drop the host blocks and the device blocks, keep
        the repacked ones the int4 GEMM reads. A `vjp`, `blocks()` or
        `dequantized()` afterwards raises."""
        comptime if not Self.MAX_Q4:
            raise Error(
                "QLinear.keep_only_repacked: this build's forward dequantizes"
                " (no NVIDIA, or NN_Q4_PATH=deq); it needs the blocks"
            )
        self.q.data = List[UInt8]()
        self.q.dev = None
        self.q.hbuf = None
        self.q.hcap = 0
        self.q.n = 0
        self.deq.release()

    def _need_blocks(self, what: StaticString) raises:
        if not self._loaded:
            raise Error(
                "QLinear." + String(what) + ": no weights (set_weight or"
                " set_blocks first)"
            )
        if self.q.n == 0:
            raise Error(
                "QLinear." + String(what) + ": no blocks are kept"
                " (keep_only_repacked, or an NN_INFERENCE_ONLY build on"
                " NVIDIA); only the int4 forward is left"
            )

    def _dequantize_dev(mut self, c: DeviceContext) raises:
        """deq(W) into the device scratch, [OUT, IN] fp32."""
        self.deq.ensure_gpu(c, Self.W_SIZE)
        c.enqueue_function[dequant_q4_0_kernel](
            self.q.dev.value(),
            self.deq.dev.value(),
            Int64(Self.OUT_),
            Int64(Self.IN_),
            grid_dim=_blocks(Self.W_SIZE),
            block_dim=256,
        )

    def _gemm_q4[
        B: Int, CFG: _QCfg
    ](mut self, c: DeviceContext) raises:
        """yb [B, N_PAD] = xb [B, K_PAD] @ deq(qpk)ᵀ, MAX's int4 GEMM."""
        var a_tt = TileTensor(
            self.xb.dev.value(), row_major(Coord(Idx[B], Idx[Self.K_PAD]))
        )
        var c_tt = TileTensor(
            self.yb.dev.value(), row_major(Coord(Idx[B], Idx[Self.N_PAD]))
        )
        var b_tt = TileTensor(
            self.qpk.dev.value(),
            row_major(Coord(Idx[Self.N_PAD], Idx[Self.PAD_ROW_BYTES])),
        )
        multistage_gemm_q[group_size=Q4_GROUP, pack_factor=8, config=CFG](
            c_tt.to_layout_tensor(),
            a_tt.as_immut().to_layout_tensor(),
            b_tt.as_immut().to_layout_tensor(),
            CFG,
            c,
        )

    # ── Module ─────────────────────────────────────────────────────────

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = inputs[0]
        comptime if target == "cpu":
            self._need_blocks("forward")
            out.ensure(B * Self.OUT_)
            var w = Tensor()
            w.data = dequantize_q4_0(self.q.data, Self.OUT_, Self.IN_)
            w.n = Self.W_SIZE
            # y = x @ deq(W)ᵀ
            gemm_cpu[False, True](
                B, Self.OUT_, Self.IN_, 1.0, x, Self.IN_, w, Self.IN_, 0.0,
                out, Self.OUT_,
            )
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
            comptime if Self.MAX_Q4:
                if not self.qpk.dev:
                    raise Error("QLinear.forward: no weights")
                self.xb.ensure_gpu(c, B * Self.K_PAD)
                self.yb.ensure_gpu(c, B * Self.N_PAD)
                c.enqueue_function[_cast_pad_kernel](
                    x.dev.value(),
                    self.xb.dev.value(),
                    Int64(B),
                    Int64(Self.IN_),
                    Int64(Self.K_PAD),
                    grid_dim=_blocks(B * Self.K_PAD),
                    block_dim=256,
                )
                comptime if Q4_CFG == "small":
                    self._gemm_q4[B, Q4_SMALL](c)
                elif Q4_CFG == "small4":
                    self._gemm_q4[B, Q4_SMALL4](c)
                elif Q4_CFG == "large":
                    self._gemm_q4[B, Q4_LARGE](c)
                else:
                    comptime assert Q4_CFG == "small2", (
                        "NN_Q4_CFG is small2, small, small4 or large"
                    )
                    self._gemm_q4[B, Q4_SMALL2](c)
                c.enqueue_function[_slice_cast_kernel](
                    self.yb.dev.value(),
                    out.dev.value(),
                    Int64(B),
                    Int64(Self.OUT_),
                    Int64(Self.N_PAD),
                    grid_dim=_blocks(B * Self.OUT_),
                    block_dim=256,
                )
            else:
                self._need_blocks("forward")
                self._dequantize_dev(c)
                comptime if has_nvidia_gpu_accelerator():
                    cublas_gemm[False, True, cublas_tf32(B, Self.OUT_, Self.IN_)](
                        c, out.dev.value(), x.dev.value(), self.deq.dev.value(),
                        B, Self.OUT_, Self.IN_, 0.0,
                    )
                else:
                    mm[
                        transpose_b=True,
                        A0=B, A1=Self.IN_, B0=Self.OUT_, B1=Self.IN_,
                        O0=B, O1=Self.OUT_,
                    ](out.dev.value(), x.dev.value(), self.deq.dev.value(), c)

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """grad_input = grad_output @ deq(W), overwriting it as `Linear.vjp`
        does. Nothing else: the weight is frozen and there is no bias."""
        ref gx = grad_inputs[0]
        self._need_blocks("vjp")
        comptime if target == "cpu":
            gx.ensure(B * Self.IN_)
            var w = Tensor()
            w.data = dequantize_q4_0(self.q.data, Self.OUT_, Self.IN_)
            w.n = Self.W_SIZE
            gemm_cpu[False, False](
                B, Self.IN_, Self.OUT_, 1.0, grad_output, Self.OUT_, w,
                Self.IN_, 0.0, gx, Self.IN_,
            )
        else:
            var c = ctx.value()
            gx.ensure_gpu(c, B * Self.IN_)
            self._dequantize_dev(c)
            comptime if has_nvidia_gpu_accelerator():
                cublas_gemm[False, False, cublas_tf32(B, Self.OUT_, Self.IN_)](
                    c, gx.dev.value(), grad_output.dev.value(),
                    self.deq.dev.value(), B, Self.IN_, Self.OUT_, 0.0,
                )
            else:
                mm[
                    A0=B, A1=Self.OUT_, B0=Self.OUT_, B1=Self.IN_,
                    O0=B, O1=Self.IN_,
                ](gx.dev.value(), grad_output.dev.value(), self.deq.dev.value(), c)

    def release_buffers(mut self):
        """The per-call activations (`xb`, `yb`). The blocks, the repacked
        blocks and the dequantize scratch stay."""
        self.xb.release()
        self.yb.release()
