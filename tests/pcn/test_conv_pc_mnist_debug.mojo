"""Debug: full conv-PCN net CPU vs GPU gradient parity on one batch.

Localizes the P3 non-learning bug: if CPU grads (trusted — the CPU end-to-end
test learns) and GPU grads diverge, the GPU inference/grad loop mishandles conv
blocks. Same architecture as the MNIST lighthouse, BATCH=8, one synthetic batch.

Run (Apple):
    pixi run -e apple mojo run -I . tests/pcn/test_conv_pc_mnist_debug.mojo
"""

from std.memory import alloc, memset
from std.math import sin
from noeira.core.tiles import tile
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock

comptime BATCH = 8
comptime T_INFER = 20
comptime LR_X: Float32 = 0.05

comptime NET = PCSequential[
    ConvPCBlock[1, 8, 3, 2, 1, 28, 28, PCIdentity],
    ConvPCBlock[8, 16, 3, 2, 1, 14, 14, PCReLU],
    PCBlock[784, 128, PCReLU],
    PCBlock[128, 10, PCIdentity],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[1, 8, 3, 2, 1, 28, 28, PCIdentity],
    ConvPCBlock[8, 16, 3, 2, 1, 14, 14, PCReLU],
    PCBlock[784, 128, PCReLU],
    PCBlock[128, 10, PCIdentity],
    dtype=dtype,
]


def main() raises:
    print("Full conv-PCN CPU-vs-GPU grads parity\n")
    var ctx = DeviceContext()

    # ── Shared params + synthetic batch (host) ───────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    var params_lt = tile[NET.PARAM_SIZE](params_buf)
    NET.pc_init_params[PCXavier, dtype](params_lt)

    var x_buf = alloc[Scalar[dtype]](BATCH * NET.IN_DIM).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    for i in range(BATCH * NET.IN_DIM):
        x_buf[i] = Scalar[dtype](0.5 + 0.5 * sin(Float32(i) * 0.3))
    for i in range(BATCH * NET.OUT_DIM):
        y_buf[i] = 0
    for b in range(BATCH):
        y_buf[b * NET.OUT_DIM + (b % NET.OUT_DIM)] = 1

    # ── CPU grads ─────────────────────────────────────────────────────────────
    var grads_cpu = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(grads_cpu, 0, NET.PARAM_SIZE)
    var lat_c = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_c = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var ab_c = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var zb_c = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_c = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    memset(lat_c, 0, BATCH * NET.LATENT_DIM)
    memset(mu_c, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(ab_c, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(zb_c, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_c, 0, BATCH * NET.LATENT_DIM)

    var pc = tile[NET.PARAM_SIZE](params_buf)
    var gc = tile[NET.PARAM_SIZE](grads_cpu)
    var latc = tile[BATCH, NET.LATENT_DIM](lat_c)
    var muc = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_c)
    var abc = tile[BATCH, NET.SCRATCH_IN_DIM](ab_c)
    var zbc = tile[BATCH, NET.SCRATCH_IN_DIM](zb_c)
    var dxc = tile[BATCH, NET.LATENT_DIM](dx_c)
    var xc = tile[BATCH, NET.IN_DIM](x_buf)
    var yc = tile[BATCH, NET.OUT_DIM](y_buf)

    var r = TRAINER.compute_grads_only[BATCH](
        pc, gc, latc, muc, abc, zbc, dxc, xc, yc,
        T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
    )
    print("  CPU: E_init=", r.energy_initial, " E_final=", r.energy_final,
          " loss=", r.output_loss_final)

    # ── GPU grads ─────────────────────────────────────────────────────────────
    var params_d = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var grads_d = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var lat_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var mu_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_OUT_DIM)
    var ab_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var zb_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var dx_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var x_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.IN_DIM)
    var y_d = ctx.enqueue_create_buffer[dtype](BATCH * NET.OUT_DIM)

    var ph = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    for i in range(NET.PARAM_SIZE):
        ph.unsafe_ptr()[i] = params_buf[i]
    ctx.enqueue_copy(params_d, ph)
    var xh = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.IN_DIM)
    for i in range(BATCH * NET.IN_DIM):
        xh.unsafe_ptr()[i] = x_buf[i]
    ctx.enqueue_copy(x_d, xh)
    var yh = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.OUT_DIM)
    for i in range(BATCH * NET.OUT_DIM):
        yh.unsafe_ptr()[i] = y_buf[i]
    ctx.enqueue_copy(y_d, yh)

    var pd = tile[NET.PARAM_SIZE](params_d)
    var gd = tile[NET.PARAM_SIZE](grads_d)
    var latd = tile[BATCH, NET.LATENT_DIM](lat_d)
    var mud = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_d)
    var abd = tile[BATCH, NET.SCRATCH_IN_DIM](ab_d)
    var zbd = tile[BATCH, NET.SCRATCH_IN_DIM](zb_d)
    var dxd = tile[BATCH, NET.LATENT_DIM](dx_d)
    var xd = tile[BATCH, NET.IN_DIM](x_d)
    var yd = tile[BATCH, NET.OUT_DIM](y_d)

    TRAINER.compute_grads_only_gpu[BATCH](
        ctx, pd, gd, latd, mud, abd, zbd, dxd, xd, yd,
        T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
    )
    ctx.synchronize()
    var gh = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    ctx.enqueue_copy(gh, grads_d)
    ctx.synchronize()

    # ── Compare per-block ─────────────────────────────────────────────────────
    var max_diff: Float32 = 0.0
    var cpu_norm: Float64 = 0.0
    var gpu_norm: Float64 = 0.0
    for i in range(NET.PARAM_SIZE):
        var d = grads_cpu[i] - gh.unsafe_ptr()[i]
        var ad = d if d >= 0 else -d
        if ad > max_diff:
            max_diff = ad
        cpu_norm += Float64(grads_cpu[i]) * Float64(grads_cpu[i])
        gpu_norm += Float64(gh.unsafe_ptr()[i]) * Float64(gh.unsafe_ptr()[i])

    # Per-block offsets for localization.
    print("  per-block grad |Δ|max and CPU/GPU L2:")
    comptime for bi in range(NET.N):
        comptime off = NET._param_offset[bi]()
        comptime psz = NET.block_types[bi].PARAM_SIZE
        var bd: Float32 = 0.0
        var bc: Float64 = 0.0
        var bg: Float64 = 0.0
        for j in range(psz):
            var d = grads_cpu[off + j] - gh.unsafe_ptr()[off + j]
            var ad = d if d >= 0 else -d
            if ad > bd:
                bd = ad
            bc += Float64(grads_cpu[off + j]) * Float64(grads_cpu[off + j])
            bg += Float64(gh.unsafe_ptr()[off + j]) * Float64(
                gh.unsafe_ptr()[off + j]
            )
        print("    block", bi, " |Δ|max=", bd, " cpuL2=", bc, " gpuL2=", bg)

    print("\n  overall max|Δ| =", max_diff)
    print("  cpu grad L2² =", cpu_norm, "  gpu grad L2² =", gpu_norm)
    if max_diff < 1e-3:
        print("\n✅ PASS — full conv-PCN GPU grads match CPU within 1e-3")
    else:
        print("\n❌ FAIL — GPU grads diverge from CPU (inference/grad bug)")
        raise Error("conv-PCN full-net CPU/GPU grad parity failed")
