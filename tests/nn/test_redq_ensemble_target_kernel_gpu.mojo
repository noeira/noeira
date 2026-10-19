"""R.5 GPU kernel parity test for `redq_ensemble_target_gpu`.

Mirrors the CPU kernel unit test: at N=2 / N_MIN=2 / MODE=MIN, the
GPU kernel must produce y values bit-equal (within FP rounding tol)
to the CPU kernel for the same inputs. This gates the GPU port of
the combine + α·logp + γ + terminal-mask math.

Apple Metal: this is the "compiles + matches CPU" gate. Real
numeric convergence on NVIDIA is HW-gated (legacy GPU-only kernel
math is well-understood; the port is mechanical).
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true

from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.redq.kernels import (
    redq_ensemble_target_cpu,
    redq_ensemble_target_gpu,
    REDQ_TARGET_MIN,
)


comptime N = 2
comptime N_MIN = 2
comptime BATCH = 8


def test_redq_ensemble_target_gpu_matches_cpu() raises:
    print("--- redq_ensemble_target_gpu vs CPU at N=2 N_MIN=2 MIN ---")
    var ctx = DeviceContext()

    # ── Build inputs on host, run both CPU and GPU, compare.
    var q_buf = Tensor.alloc(N * BATCH)
    # Deterministic values; Q1 and Q2 alternate which is smaller.
    var q1_vals = List[Float64](length=BATCH, fill=0.0)
    var q2_vals = List[Float64](length=BATCH, fill=0.0)
    for b in range(BATCH):
        q1_vals[b] = 0.3 * Float64(b) - 1.0
        q2_vals[b] = -0.2 * Float64(b) + 0.5
        q_buf.data[0 * BATCH + b] = Scalar[DT](q1_vals[b])
        q_buf.data[1 * BATCH + b] = Scalar[DT](q2_vals[b])

    var subset_host = List[UInt32](length=N_MIN, fill=UInt32(0))
    subset_host[0] = UInt32(0)
    subset_host[1] = UInt32(1)

    var subset_cpu_int = List[Int](length=N_MIN, fill=0)
    subset_cpu_int[0] = 0
    subset_cpu_int[1] = 1

    var rewards = Tensor.alloc(BATCH)
    var terms   = Tensor.alloc(BATCH)
    var lps     = Tensor.alloc(BATCH)
    for b in range(BATCH):
        rewards.data[b] = Scalar[DT](-0.1 + 0.2 * Float64(b))
        terms.data[b]   = Scalar[DT](1.0) if b == 3 else Scalar[DT](0.0)
        lps.data[b]     = Scalar[DT](-0.5 + 0.15 * Float64(b))

    var gamma = Scalar[DT](0.97)
    var alpha = Scalar[DT](0.18)

    # CPU reference.
    var y_cpu = Tensor.alloc(BATCH)
    redq_ensemble_target_cpu[N, N_MIN, REDQ_TARGET_MIN, BATCH](
        rewards,
        q_buf,
        terms,
        lps,
        subset_cpu_int,
        gamma, alpha,
        y_cpu,
    )

    # GPU run. The launcher takes the storage `Tensor`s themselves (it reads
    # their device buffers), so upload the same host slabs used above.
    var y_gpu = Tensor.alloc(BATCH)
    var subset_dev = ctx.enqueue_create_buffer[DType.uint32](N_MIN)
    ctx.enqueue_copy(subset_dev, subset_host.unsafe_ptr())
    y_gpu.ensure_gpu(ctx, BATCH)
    rewards.upload(ctx)
    q_buf.upload(ctx)
    terms.upload(ctx)
    lps.upload(ctx)

    redq_ensemble_target_gpu[N, N_MIN, REDQ_TARGET_MIN, BATCH](
        ctx,
        y_gpu,
        rewards,
        q_buf,
        terms,
        lps,
        tile[N_MIN](subset_dev.unsafe_ptr().as_unsafe_any_origin()),
        gamma, alpha,
    )
    y_gpu.download(ctx)
    ctx.synchronize()

    var max_dev: Float64 = 0.0
    for b in range(BATCH):
        var d = Float64(y_gpu.data[b]) - Float64(y_cpu.data[b])
        if d < 0.0:
            d = -d
        if d > max_dev:
            max_dev = d
        print(
            "  b=", b,
            " cpu=", y_cpu.data[b],
            " gpu=", y_gpu.data[b],
        )
    print("  max |gpu - cpu| =", max_dev)
    assert_true(
        max_dev < 1e-5,
        "GPU kernel must match CPU within FP rounding tolerance",
    )
    print("PASS — redq_ensemble_target GPU == CPU at N=2 N_MIN=2 MIN.")


def main() raises:
    test_redq_ensemble_target_gpu_matches_cpu()
