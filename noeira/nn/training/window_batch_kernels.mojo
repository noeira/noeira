"""Next-token window batches built on the device — the GPU twin of
`datasets.make_batch` + the trainer's host one-hot.

The token corpus lives on the device as int32. One kernel draws each row's
window start from a counter hash of (seed, step, row) — both words read from
device memory — and writes the input and target one-hots in full (every
element, so nothing needs clearing first); a one-thread kernel then advances
the step word. Nothing reads the host, so the batch build rides inside a
captured CUDA graph and each replay draws a fresh batch.

Window semantics are `make_batch`'s: start uniform in [0, n_starts) with
n_starts = len(corpus) - SEQ, input = corpus[start + t], target =
corpus[start + t + 1].
"""

from layout import Layout, LayoutTensor
from max.gpu import global_idx
from std.memory import Pointer

from noeira.nn.constants import DT
from noeira.nn.random.hash_mask import hash_u64


@always_inline
def window_start(seed: UInt64, step: UInt64, row: Int, n_starts: Int) -> Int:
    """Row `row`'s window start at step `step`, uniform in [0, n_starts)
    (the modulo bias is n_starts / 2^64)."""
    return Int(hash_u64(seed, step, UInt64(row)) % UInt64(n_starts))


def window_onehot_kernel[
    ADT: DType, BATCH: Int, SEQ: Int, VOCAB: Int
](
    corpus: Pointer[Scalar[DType.int32], MutAnyOrigin],
    n_starts: Int64,
    rng: LayoutTensor[DType.uint64, Layout.row_major(2), MutAnyOrigin],
    inp: LayoutTensor[ADT, Layout.row_major(BATCH * SEQ * VOCAB), MutAnyOrigin],
    tgt: LayoutTensor[DT, Layout.row_major(BATCH * SEQ * VOCAB), MutAnyOrigin],
    starts: LayoutTensor[DType.int32, Layout.row_major(BATCH), MutAnyOrigin],
):
    """One thread per one-hot element `[b, t, v]`: writes `inp` (the net's
    activation dtype) and `tgt` (fp32, the loss target); the `[b, 0, 0]`
    thread records the row's start in `starts`. `rng` = [seed, step]."""
    comptime ROW = SEQ * VOCAB
    comptime TOTAL = BATCH * ROW
    var i = Int(global_idx.x)
    if i >= TOTAL:
        return
    var b = i // ROW
    var r = i - b * ROW
    var t = r // VOCAB
    var v = r - t * VOCAB
    var seed = rng.ptr[unsafe_offset=0]
    var step = rng.ptr[unsafe_offset=1]
    var s = window_start(seed, step, b, Int(n_starts))
    var x = Int(corpus[unsafe_offset=s + t])
    var y = Int(corpus[unsafe_offset=s + t + 1])
    inp[i] = Scalar[ADT](1) if x == v else Scalar[ADT](0)
    tgt[i] = Scalar[DT](1) if y == v else Scalar[DT](0)
    if r == 0:
        starts[b] = Int32(s)


def advance_step_kernel(
    rng: LayoutTensor[DType.uint64, Layout.row_major(2), MutAnyOrigin],
):
    """step += 1 (one thread). Enqueued after `window_onehot_kernel`."""
    if Int(global_idx.x) == 0:
        rng.ptr[unsafe_offset=1] = rng.ptr[unsafe_offset=1] + 1
