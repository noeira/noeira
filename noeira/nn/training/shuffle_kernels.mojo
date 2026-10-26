"""GPU helper kernels for whole-dataset minibatch shuffling (storage surface).

Pure `LayoutTensor` kernels — permutation indices + RNG seed live in device
memory, so the shuffle → gather → step sequence is CUDA-graph capturable. Ported
verbatim from the legacy `nn/training/shuffle_kernels.mojo` (same shapes, same
PhiloxRandom Fisher-Yates, same parallel gather); kept framework-agnostic so the
storage `Trainer.train_gpu` drives them over `TensorImpl[int32]`/`[uint64]`
shuffle scratch (no legacy import).
"""

from max.gpu import block_dim, block_idx, thread_idx
from std.random.philox import Random as PhiloxRandom
from noeira.core.tiles import Tile


@always_inline
def init_identity_indices_kernel[
    N: Int,
](indices: Tile[DType.int32, N]):
    """Fill indices[i] = i. Parallel over N threads."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < N:
        indices[i] = Int32(i)


@always_inline
def fisher_yates_shuffle_kernel[
    N: Int,
](
    indices: Tile[DType.int32, N],
    seed_buf: Tile[DType.uint64, 1],
):
    """Serial Fisher-Yates shuffle on a single GPU thread.

    N serial iterations on one thread is fast enough at ~once-per-epoch cadence
    (a few ms at N=60k). Parallel shuffles (sort-by-random-key) are faster but
    require a device sort, which this codebase does not have.
    """
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var s = seed_buf.ptr[unsafe_offset=0]
    var philox = PhiloxRandom(seed=s, offset=0)
    for i in range(N - 1, 0, -1):
        var r = philox.step_uniform()
        # Metal does not support Float64 — use Float32 throughout.
        var j = Int(Float32(r[0]) * Float32(i + 1))
        if j > i:
            j = i
        var tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp


@always_inline
def increment_seed_kernel(
    seed_buf: Tile[DType.uint64, 1],
):
    """Bump the device-side RNG seed so each epoch has a different
    permutation."""
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        seed_buf.ptr[unsafe_offset=0] = seed_buf.ptr[unsafe_offset=0] + UInt64(1)


@always_inline
def gather_rows_kernel[
    N_TOTAL: Int,
    BATCH: Int,
    DIM: Int,
    dtype: DType,
](
    batch_out: Tile[dtype, BATCH, DIM],
    full: Tile[dtype, N_TOTAL, DIM],
    indices: Tile[DType.int32, N_TOTAL],
    # ⚠ Int32, NOT Int — `Int`/`UInt` are not `DevicePassable`. A bare
    # `Int` still compiles in `pixi run build` and fails only where the
    # kernel is LAUNCHED; keep the `Int32(...)` casts at the call sites.
    offset: Int32,
):
    """`batch_out[b, d]` = full[indices[offset + b], d]. Parallel over
    BATCH * DIM threads."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH * DIM:
        return
    var b = i // DIM
    var d = i % DIM
    var src = Int(indices[Int(offset) + b])
    batch_out[b, d] = full[src, d]


@always_inline
def gather_rows_at_counter_kernel[
    N_TOTAL: Int,
    BATCH: Int,
    DIM: Int,
    dtype: DType,
](
    batch_out: Tile[dtype, BATCH, DIM],
    full: Tile[dtype, N_TOTAL, DIM],
    indices: Tile[DType.int32, N_TOTAL],
    counter: Tile[DType.int32, 1],
):
    """`gather_rows_kernel` at offset `counter[0] * BATCH`, read from device
    memory: the same launch serves every batch of the epoch, so it can be
    captured once and replayed (`advance_counter_kernel` moves to the next
    batch)."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH * DIM:
        return
    var b = i // DIM
    var d = i % DIM
    var off = Int(counter.ptr[unsafe_offset=0]) * BATCH
    var src = Int(indices[off + b])
    batch_out[b, d] = full[src, d]


@always_inline
def advance_counter_kernel(
    counter: Tile[DType.int32, 1],
):
    """`counter += 1` (one thread)."""
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        counter.ptr[unsafe_offset=0] = counter.ptr[unsafe_offset=0] + 1


@always_inline
def reset_counter_kernel(
    counter: Tile[DType.int32, 1],
):
    """`counter = 0` (one thread)."""
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        counter.ptr[unsafe_offset=0] = 0
