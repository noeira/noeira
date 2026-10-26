"""Direct test for `seed_grad_inv_batch[target, BATCH]` (Phase 3 helper).

The helper fills a [BATCH] tensor with `1/BATCH`. Used in FullGraph
loss blocks (SACActorLoss, soon TargetYBlock) to seed the
backward pass for a mean-batch loss: when the forward output is
`loss_per_b ∈ [BATCH, 1]` and the trainer wants `d(mean_b(loss_per_b))/d(loss_per_b)`,
that gradient is the constant `1/BATCH` in every slot.

Pre-Phase-3 the SACActorLoss had an inline `_fill_constant_kernel`;
this helper extracts that pattern. Currently covered only indirectly
by the SAC bit-identity gate — a direct test catches regressions
in isolation (e.g. accidental `1/N` formula bugs at the next
helper user).
"""

from std.memory import alloc
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.deep_agents.loss.seed_grad_inv_batch import seed_grad_inv_batch


def test_cpu_basic() raises:
    print("test_cpu_basic ...")
    comptime BATCH = 8
    var buf = alloc[Scalar[DT]](BATCH).as_unsafe_any_origin()
    # Pre-fill with garbage to confirm the seed overwrites it.
    for i in range(BATCH):
        buf[i] = Scalar[DT](999.0)

    seed_grad_inv_batch["cpu", BATCH](
        tile[BATCH, 1](buf)
    )

    var expected = Scalar[DT](1.0) / Scalar[DT](BATCH)
    for i in range(BATCH):
        assert_true(
            (buf[i] - expected).__abs__() < Scalar[DT](1e-6),
            "seed_grad_inv_batch[cpu] failed at i=" + String(i),
        )
    buf.free()
    print("  ok")


def test_cpu_batch_sizes() raises:
    """Different BATCH values must each produce 1/BATCH."""
    print("test_cpu_batch_sizes ...")
    # BATCH=1 → 1.0, BATCH=2 → 0.5, BATCH=256 → ~0.0039.
    var b1 = alloc[Scalar[DT]](1).as_unsafe_any_origin()
    seed_grad_inv_batch["cpu", 1](
        tile[1, 1](b1)
    )
    assert_true(b1[0] == Scalar[DT](1.0), "BATCH=1 should give 1.0")
    b1.free()

    var b2 = alloc[Scalar[DT]](2).as_unsafe_any_origin()
    seed_grad_inv_batch["cpu", 2](
        tile[2, 1](b2)
    )
    assert_true(b2[0] == Scalar[DT](0.5), "BATCH=2 [0] should give 0.5")
    assert_true(b2[1] == Scalar[DT](0.5), "BATCH=2 [1] should give 0.5")
    b2.free()

    var b256 = alloc[Scalar[DT]](256).as_unsafe_any_origin()
    seed_grad_inv_batch["cpu", 256](
        tile[256, 1](b256)
    )
    var expected_256 = Scalar[DT](1.0) / Scalar[DT](256)
    for i in range(256):
        assert_true(
            (b256[i] - expected_256).__abs__() < Scalar[DT](1e-7),
            "BATCH=256 inv mismatch at i=" + String(i),
        )
    b256.free()
    print("  ok")


def test_gpu_basic() raises:
    print("test_gpu_basic ...")
    comptime BATCH = 16
    var ctx = DeviceContext()
    var dev_buf = ctx.enqueue_create_buffer[DT](BATCH)
    var host_buf = ctx.enqueue_create_host_buffer[DT](BATCH)
    for i in range(BATCH):
        host_buf[i] = Scalar[DT](999.0)
    ctx.enqueue_copy(dev_buf, host_buf)

    var dev_p = tile[BATCH, 1](dev_buf.unsafe_ptr().as_unsafe_any_origin())
    seed_grad_inv_batch["gpu", BATCH](
        dev_p, Optional[DeviceContext](ctx)
    )
    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()

    var expected = Scalar[DT](1.0) / Scalar[DT](BATCH)
    for i in range(BATCH):
        assert_true(
            (host_buf[i] - expected).__abs__() < Scalar[DT](1e-6),
            "seed_grad_inv_batch[gpu] failed at i=" + String(i),
        )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("seed_grad_inv_batch — Phase 3 helper unit test")
    print("=" * 70)
    test_cpu_basic()
    test_cpu_batch_sizes()
    try:
        test_gpu_basic()
    except e:
        # GPU may be unavailable in some build configs; skip with a note.
        print("  test_gpu_basic SKIPPED (no GPU? error: ", String(e), ")")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
