"""Phase D gate — PCN GPU path (max_matmul) ↔ CPU parity (linear PCModule).

A 2-block linear `PCModule` is trained identically on CPU and GPU (same
init, same data, same Adam) and the resulting weights must agree within
float tolerance. The GPU settling routes every matmul through `max_matmul`
(`PCBlock.USE_MAX_KERNELS` path, now used on Apple too — no custom MMA).

Validates:
  - `PCModule.make_pcn_gpu` (device Param) + `Adam.make['gpu']`.
  - `pc_module_train_one_batch_gpu` → `compute_grads_only_gpu` settling via
    `max_matmul` produces gradients matching the CPU path.
  - `Adam.step['gpu']` over the device `Param` tracks the CPU optimizer.

Run (Apple):
    pixi run -e apple mojo run -I . tests/pcn/test_pc_module_gpu_parity.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true
from std.math import abs
from max.gpu.host import DeviceContext
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.nn.optimizer.adam import Adam

from noeira.experimental.pcn.pc_block import PCBlock
from noeira.experimental.pcn.predictive_model import PCIdentity
from noeira.experimental.pcn.pc_sequential import PCSequential
from noeira.experimental.pcn.pc_module import PCModule
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_module_trainer import pc_module_train_one_batch
from noeira.experimental.pcn.pc_module_trainer_gpu import (
    pc_module_train_one_batch_gpu,
    PCGpuWorkspace,
)
from noeira.nn.core.ptr import mptr


def main() raises:
    comptime IN = 4
    comptime H = 8
    comptime OUT = 2
    comptime BATCH = 16
    comptime T_INFER = 20
    comptime N_STEPS = 5
    comptime LR_X = Scalar[DT](0.1)

    comptime Net = PCModule[
        PCBlock[IN, H, PCIdentity],
        PCBlock[H, OUT, PCIdentity],
    ]
    comptime Seq = PCSequential[
        PCBlock[IN, H, PCIdentity],
        PCBlock[H, OUT, PCIdentity],
    ]
    comptime PSIZE = Seq.PARAM_SIZE

    var ctx = DeviceContext()
    seed(0)

    # Data.
    var x_s = List[Scalar[DT]](capacity=BATCH * IN)
    for _ in range(BATCH * IN):
        x_s.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    var w_true = List[Scalar[DT]](capacity=IN * OUT)
    for _ in range(IN * OUT):
        w_true.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    var y_s = List[Scalar[DT]](length=BATCH * OUT, fill=Scalar[DT](0))
    for b in range(BATCH):
        for j in range(OUT):
            var acc = Scalar[DT](0)
            for i in range(IN):
                acc += x_s[b * IN + i] * w_true[i * OUT + j]
            y_s[b * OUT + j] = acc

    var x_cpu = tile[BATCH, IN](mptr(x_s))
    var y_cpu = tile[BATCH, OUT](mptr(y_s))

    # Nets: CPU init, then GPU starts from the SAME weights.
    var cpu_net = Net.make_pcn[PCXavier]()
    var gpu_net = Net.make_pcn_gpu[PCXavier](ctx)
    ctx.enqueue_copy(
        gpu_net.weights.val.dev.value(), mptr(cpu_net.weights.val.data)
    )
    ctx.synchronize()

    var cpu_opt = Adam(lr=Scalar[DT](1e-2))
    var gpu_opt = Adam(lr=Scalar[DT](1e-2))

    # Upload data to device.
    var x_dev_b = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var y_dev_b = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(x_dev_b, mptr(x_s))
    ctx.enqueue_copy(y_dev_b, mptr(y_s))
    ctx.synchronize()
    var x_dev = tile[BATCH, IN](x_dev_b.unsafe_ptr().as_unsafe_any_origin())
    var y_dev = tile[BATCH, OUT](y_dev_b.unsafe_ptr().as_unsafe_any_origin())

    # Persistent GPU workspace — allocated once, reused every step.
    var ws = PCGpuWorkspace[BATCH, PCBlock[IN, H, PCIdentity], PCBlock[
        H, OUT, PCIdentity
    ]].make(ctx)

    # Train both identically.
    for _ in range(N_STEPS):
        _ = pc_module_train_one_batch[BATCH](
            cpu_net, cpu_opt, x_cpu, y_cpu, T_INFER, LR_X
        )
        pc_module_train_one_batch_gpu[BATCH](
            ctx, gpu_net, gpu_opt, ws, x_dev, y_dev, T_INFER, LR_X
        )
    ctx.synchronize()

    # Download GPU weights, compare.
    var gpu_host = List[Scalar[DT]](length=PSIZE, fill=Scalar[DT](0))
    ctx.enqueue_copy(mptr(gpu_host), gpu_net.weights.val.dev.value())
    ctx.synchronize()

    var max_diff = Float64(0.0)
    for k in range(PSIZE):
        var d = abs(Float64(cpu_net.weights.val.data[k]) - Float64(gpu_host[k]))
        if d > max_diff:
            max_diff = d
    print("max |W_cpu - W_gpu| after", N_STEPS, "steps =", max_diff)
    assert_true(max_diff == max_diff, "NaN in weights")
    assert_true(
        max_diff < 1e-3,
        "CPU/GPU weight parity broke (max_matmul GPU settling mismatch)",
    )
    print("PASS: PCN GPU (max_matmul) ↔ CPU parity (Phase D)")
    _ = x_dev_b^
    _ = y_dev_b^
