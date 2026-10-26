"""Transpose2D[A, B] — in-batch 2D transpose (storage surface).

Transformed from legacy `nn.primitives.Transpose2D` (surface-only change). The
CPU loops + the GPU `_transpose2d_kernel` are carried over verbatim.

Reinterprets each sample's `A*B` flat slab as a row-major `(A, B)` matrix and
transposes it to `(B, A)`, written back row-major:

    out[b, j*A + i] = in[b, i*B + j]   for i in [0,A), j in [0,B)

IN_DIM == OUT_DIM == A*B; no params, no cache. Backward is the inverse
permutation (transpose again):

    grad_in[b, i*B + j] = grad_out[b, j*A + i]

Used by ViT `PatchEmbed` to turn Conv2D's channel-major patch layout
`(embed_dim, n_patches)` into the attention op's patch-major
`(n_patches, embed_dim)`.
"""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile, tile, shared_tile

from noeira.nn.constants import DT, TPB

comptime T2D_TILE = 32   # 32x32 shared-mem tile (B1 coalescing rewrite)
comptime T2D_BR = 8      # BLOCK_ROWS: 32x8 block, 4 elems/thread (B1' occupancy)
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── GPU kernel: shared-mem tiled batched transpose (B1 / B1') ──────────
# Per sample, src is row-major (A, B); dst is row-major (B, A):
#   dst[b, j*A + i] = src[b, i*B + j].
# The naive one-thread-per-element map writes dst coalesced but reads src
# with stride B (uncoalesced). Staging a 32×32 block in shared memory makes
# BOTH the read (along j) and the write (along i) coalesced; the +1 column
# pad avoids shared-memory bank conflicts on the transposed read.
# The block is 32×8 (BLOCK_ROWS) with 4 elements/thread (canonical NVIDIA
# transpose) — 256 threads/block gives better occupancy/ILP than a 32×32
# 1024-thread block. NVIDIA (A100-class): 5.8× @ A=192 vs naive (2.1× over
# the 32×32 tile), 1.6–1.95× for larger A.
def _transpose2d_kernel[
    BATCH: Int, A: Int, B: Int
](
    # src laid out (A, B) per sample, dst laid out (B, A) per sample.
    src: Tile[DT, BATCH, A * B],
    dst: Tile[DT, BATCH, A * B],
):
    var tile = shared_tile[DT, T2D_TILE, T2D_TILE + 1]()

    var b = Int(block_idx.z)
    var cy = Int(block_idx.y) * T2D_TILE     # tile origin in i (A dim)
    var cx = Int(block_idx.x) * T2D_TILE     # tile origin in j (B dim)
    var tx = Int(thread_idx.x)               # [0, T2D_TILE)
    var ty = Int(thread_idx.y)               # [0, T2D_BR)

    # load: j=cx+tx → consecutive tx = consecutive j (coalesced read); each
    # thread walks T2D_TILE/T2D_BR rows of i.
    var j = cx + tx
    comptime for r in range(0, T2D_TILE, T2D_BR):
        var i = cy + ty + r
        if i < A and j < B:
            tile[ty + r, tx] = rebind[Scalar[DT]](src[b, i * B + j])
    barrier()

    # write: out col = i (stride 1 in dst row j → coalesced), out row = j
    var i2 = cy + tx
    comptime for r in range(0, T2D_TILE, T2D_BR):
        var j2 = cx + ty + r
        if i2 < A and j2 < B:
            dst[b, j2 * A + i2] = rebind[Scalar[DT]](tile[tx, ty + r])


struct Transpose2D[A_: Int, B_: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.A_ * Self.B_)
    comptime OUT_DIM = Self.A_ * Self.B_

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert Self.A_ > 0 and Self.B_ > 0, (
            "Transpose2D: A,B must be > 0"
        )
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime AB = Self.A_ * Self.B_
        comptime if target == "cpu":
            out.ensure(B * AB)
            var in_v = TileTensor(in0.data, row_major[B, AB]())
            var out_v = TileTensor(out.data, row_major[B, AB]())
            for b in range(B):
                for i in range(Self.A_):
                    for j in range(Self.B_):
                        out_v[b, j * Self.A_ + i] = in_v[b, i * Self.B_ + j]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * AB)
            comptime l2d = RowMajor[B, AB]
            # tile grid: x over j (B_ cols), y over i (A_ rows), z over batch.
            comptime gx = (Self.B_ + T2D_TILE - 1) // T2D_TILE
            comptime gy = (Self.A_ + T2D_TILE - 1) // T2D_TILE
            c.enqueue_function[_transpose2d_kernel[B, Self.A_, Self.B_]](
                in0.tt["gpu", B, AB](),
                out.tt["gpu", B, AB](),
                grid_dim=(gx, gy, B),
                block_dim=(T2D_TILE, T2D_BR),
            )

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
        ref gin = grad_inputs[0]
        comptime AB = Self.A_ * Self.B_
        # Backward of a transpose is the transpose with (A,B) swapped:
        # grad_in[b, i*B + j] = grad_out[b, j*A + i]. Reuse the same kernel
        # with src=grad_out laid out (B,A) → dst=grad_in laid out (A,B).
        comptime if target == "cpu":
            gin.ensure(B * AB)
            var go_v = TileTensor(grad_output.data, row_major[B, AB]())
            var gi_v = TileTensor(gin.data, row_major[B, AB]())
            for b in range(B):
                for i in range(Self.A_):
                    for j in range(Self.B_):
                        gi_v[b, i * Self.B_ + j] = go_v[b, j * Self.A_ + i]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * AB)
            comptime l2d = RowMajor[B, AB]
            # Inverse permutation: treat grad_out as (B,A), write grad_in (A,B).
            # Kernel A,B are swapped → grid x over A_ cols, y over B_ rows.
            comptime gx = (Self.A_ + T2D_TILE - 1) // T2D_TILE
            comptime gy = (Self.B_ + T2D_TILE - 1) // T2D_TILE
            c.enqueue_function[_transpose2d_kernel[B, Self.B_, Self.A_]](
                grad_output.tt["gpu", B, AB](),
                gin.tt["gpu", B, AB](),
                grid_dim=(gx, gy, B),
                block_dim=(T2D_TILE, T2D_BR),
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
