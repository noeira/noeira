"""Phase D — conv PCModule GPU ↔ CPU parity (CPU).

A conv+linear `PCModule` (`ConvPCBlock` → readout `PCBlock`) is trained
identically on CPU and GPU; the resulting weights must agree within float
tolerance. Confirms the GPU settling (`compute_grads_only_gpu`) drives a
NON-linear block type correctly through `PCModule` + `Adam.step['gpu']`.

NOTE: the linear block's GPU matmuls use `max_matmul` (Phase D gate flip);
the conv block's GPU ops still use PCN's direct-conv custom kernels (NOT
`max_matmul`). Porting conv GPU to im2col+`max_matmul` (to match nn's conv)
is the remaining Phase D optimization — this test validates correctness of
the existing path.

Run (Apple):
    pixi run -e apple mojo run -I . tests/pcn/test_pc_module_conv_gpu_parity.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true
from std.math import abs
from max.gpu.host import DeviceContext
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.nn.optimizer.adam import Adam

from noeira.experimental.pcn.pc_block import PCBlock
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
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
    comptime C_IN = 1
    comptime H = 6
    comptime W = 6
    comptime IN = C_IN * H * W  # 36
    comptime C_OUT = 2
    comptime CONV_OUT = C_OUT * H * W  # 72
    comptime OUT = 3
    comptime BATCH = 8
    comptime T_INFER = 20
    comptime N_STEPS = 5
    comptime LR_X = Scalar[DT](0.1)

    comptime B0 = ConvPCBlock[C_IN, C_OUT, 3, 1, 1, H, W, PCIdentity]
    comptime B1 = PCBlock[CONV_OUT, OUT, PCIdentity]
    comptime Net = PCModule[B0, B1]
    comptime Seq = PCSequential[B0, B1]
    comptime PSIZE = Seq.PARAM_SIZE

    var ctx = DeviceContext()
    seed(0)

    # Data.
    var x_s = List[Scalar[DT]](capacity=BATCH * IN)
    for _ in range(BATCH * IN):
        x_s.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    var y_s = List[Scalar[DT]](capacity=BATCH * OUT)
    for _ in range(BATCH * OUT):
        y_s.append(Scalar[DT](random_float64() * 2.0 - 1.0))

    var x_cpu = tile[BATCH, IN](mptr(x_s))
    var y_cpu = tile[BATCH, OUT](mptr(y_s))

    # CPU init; GPU starts from the SAME weights.
    var cpu_net = Net.make_pcn[PCXavier]()
    var gpu_net = Net.make_pcn_gpu[PCXavier](ctx)
    ctx.enqueue_copy(
        gpu_net.weights.val.dev.value(), mptr(cpu_net.weights.val.data)
    )
    ctx.synchronize()

    var cpu_opt = Adam(lr=Scalar[DT](1e-2))
    var gpu_opt = Adam(lr=Scalar[DT](1e-2))

    # Upload data.
    var x_dev_b = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var y_dev_b = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(x_dev_b, mptr(x_s))
    ctx.enqueue_copy(y_dev_b, mptr(y_s))
    ctx.synchronize()
    var x_dev = tile[BATCH, IN](x_dev_b.unsafe_ptr().as_unsafe_any_origin())
    var y_dev = tile[BATCH, OUT](y_dev_b.unsafe_ptr().as_unsafe_any_origin())

    var ws = PCGpuWorkspace[BATCH, B0, B1].make(ctx)

    for _ in range(N_STEPS):
        _ = pc_module_train_one_batch[BATCH](
            cpu_net, cpu_opt, x_cpu, y_cpu, T_INFER, LR_X
        )
        pc_module_train_one_batch_gpu[BATCH](
            ctx, gpu_net, gpu_opt, ws, x_dev, y_dev, T_INFER, LR_X
        )
    ctx.synchronize()

    var gpu_host = List[Scalar[DT]](length=PSIZE, fill=Scalar[DT](0))
    ctx.enqueue_copy(mptr(gpu_host), gpu_net.weights.val.dev.value())
    ctx.synchronize()

    var max_diff = Float64(0.0)
    for k in range(PSIZE):
        var d = abs(Float64(cpu_net.weights.val.data[k]) - Float64(gpu_host[k]))
        if d > max_diff:
            max_diff = d
    print("conv max |W_cpu - W_gpu| after", N_STEPS, "steps =", max_diff)
    assert_true(max_diff == max_diff, "NaN in weights")
    assert_true(max_diff < 1e-3, "conv CPU/GPU weight parity broke")
    print("PASS: conv PCModule GPU ↔ CPU parity (Phase D)")
    _ = x_dev_b^
    _ = y_dev_b^
