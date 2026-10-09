"""Window batches sharded by rank — `window_onehot_kernel` with a row offset.

The single-GPU sampler draws row b's window from `hash(seed, step, b)`. Here
rank r builds its BL local rows as GLOBAL rows `r*BL .. r*BL+BL-1` of the same
hash, so N ranks with BL = B/N rows each draw exactly the windows one rank with
B rows draws. The global batch is the same at every N by construction, which is
what the N-vs-1 equivalence gate needs, and the ranks' rows are disjoint.
"""

from layout import Layout, LayoutTensor
from max.gpu import global_idx

from noeira.nn.constants import DT
from noeira.nn.training.window_batch_kernels import window_start


def window_onehot_shard_kernel[
    ADT: DType, BL: Int, SEQ: Int, VOCAB: Int
](
    corpus: Pointer[Scalar[DType.int32], MutAnyOrigin],
    n_starts: Int64,
    row_off: Int64,
    rng: LayoutTensor[DType.uint64, Layout.row_major(2), MutAnyOrigin],
    inp: LayoutTensor[ADT, Layout.row_major(BL * SEQ * VOCAB), MutAnyOrigin],
    tgt: LayoutTensor[DT, Layout.row_major(BL * SEQ * VOCAB), MutAnyOrigin],
):
    """One thread per one-hot element `[b, t, v]` of this rank's BL rows;
    local row b is global row `row_off + b`. `rng` = [seed, step]."""
    comptime ROW = SEQ * VOCAB
    comptime TOTAL = BL * ROW
    var i = Int(global_idx.x)
    if i >= TOTAL:
        return
    var b = i // ROW
    var r = i - b * ROW
    var t = r // VOCAB
    var v = r - t * VOCAB
    var seed = rng.ptr[unsafe_offset=0]
    var step = rng.ptr[unsafe_offset=1]
    var s = window_start(seed, step, Int(row_off) + b, Int(n_starts))
    var x = Int(corpus[unsafe_offset=s + t])
    var y = Int(corpus[unsafe_offset=s + t + 1])
    inp[i] = Scalar[ADT](1) if x == v else Scalar[ADT](0)
    tgt[i] = Scalar[DT](1) if y == v else Scalar[DT](0)
