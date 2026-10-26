"""`mppi_copy_z_kernel` / `mppi_broadcast_z0_...` must touch EVERY element.

Both were re-indexed one-thread-per-ELEMENT (from one-per-ROW) to fix a 9-block
uncoalesced launch that cost 10.4% of GPU time on an RTX 5090. The failure mode
of getting that wrong is silent: a grid sized for rows leaves the tail of the
latent slab stale, so the planner rolls a partially-updated z and simply plans
worse. Nothing raises. So these gates check the FULL slab element by element,
with a pre-poisoned destination so an untouched tail cannot pass.

    pixi run -e apple mojo run -I . tests/planners/trajectory/test_mppi_z_kernels_grid.mojo
"""

from max.gpu.host import DeviceContext
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.planners.trajectory.mppi_kernels import (
    mppi_copy_z_kernel,
    mppi_broadcast_z0_zero_returns_batched_kernel,
)

comptime TPB = 256
# Deliberately NOT a multiple of TPB, and a row count whose row-grid
# (ceil(536/256) = 3 blocks) is far short of the element-grid (1072 blocks) —
# so the old launch would leave most of the slab untouched.
comptime N_ENVS = 2
comptime TOTAL_SAMPLES = 268
comptime BATCH_TOTAL = N_ENVS * TOTAL_SAMPLES      # 536
comptime LATENT = 512
comptime ELEMS = BATCH_TOTAL * LATENT              # 274_432
comptime Z_BLOCKS = (ELEMS + TPB - 1) // TPB


def _expect(row: Int, k: Int) -> Scalar[DT]:
    return Scalar[DT](row) * Scalar[DT](1000.0) + Scalar[DT](k)


def test_copy_z(ctx: DeviceContext) raises:
    var src_h = ctx.enqueue_create_host_buffer[DT](ELEMS)
    var dst_h = ctx.enqueue_create_host_buffer[DT](ELEMS)
    for r in range(BATCH_TOTAL):
        for k in range(LATENT):
            src_h[r * LATENT + k] = _expect(r, k)
            dst_h[r * LATENT + k] = Scalar[DT](-7.0)   # poison
    var src = ctx.enqueue_create_buffer[DT](ELEMS)
    var dst = ctx.enqueue_create_buffer[DT](ELEMS)
    ctx.enqueue_copy(src, src_h)
    ctx.enqueue_copy(dst, dst_h)

    comptime k = mppi_copy_z_kernel[DT, BATCH_TOTAL, LATENT]
    ctx.enqueue_function[k](
        tile[BATCH_TOTAL, LATENT](dst),
        tile[BATCH_TOTAL, LATENT](src),
        grid_dim=(Z_BLOCKS,),
        block_dim=(TPB,),
    )
    ctx.enqueue_copy(dst_h, dst)
    ctx.synchronize()

    var bad = 0
    var first_bad = -1
    for r in range(BATCH_TOTAL):
        for kk in range(LATENT):
            if dst_h[r * LATENT + kk] != _expect(r, kk):
                bad += 1
                if first_bad < 0:
                    first_bad = r * LATENT + kk
    print("  copy_z    : ", ELEMS - bad, "/", ELEMS, " elements correct", sep="")
    if bad != 0:
        raise Error(
            "copy_z left " + String(bad) + " elements stale (first at flat "
            + String(first_bad) + ")"
        )


def test_broadcast_z0(ctx: DeviceContext) raises:
    var z0_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * LATENT)
    for e in range(N_ENVS):
        for k in range(LATENT):
            z0_h[e * LATENT + k] = _expect(e, k)
    var zall_h = ctx.enqueue_create_host_buffer[DT](ELEMS)
    var ret_h = ctx.enqueue_create_host_buffer[DT](BATCH_TOTAL)
    for i in range(ELEMS):
        zall_h[i] = Scalar[DT](-7.0)
    for i in range(BATCH_TOTAL):
        ret_h[i] = Scalar[DT](-7.0)

    var z0 = ctx.enqueue_create_buffer[DT](N_ENVS * LATENT)
    var zall = ctx.enqueue_create_buffer[DT](ELEMS)
    var ret = ctx.enqueue_create_buffer[DT](BATCH_TOTAL)
    ctx.enqueue_copy(z0, z0_h)
    ctx.enqueue_copy(zall, zall_h)
    ctx.enqueue_copy(ret, ret_h)

    comptime k = mppi_broadcast_z0_zero_returns_batched_kernel[
        DT, BATCH_TOTAL, N_ENVS, TOTAL_SAMPLES, LATENT
    ]
    ctx.enqueue_function[k](
        tile[N_ENVS, LATENT](z0),
        tile[BATCH_TOTAL, LATENT](zall),
        tile[BATCH_TOTAL](ret),
        grid_dim=(Z_BLOCKS,),
        block_dim=(TPB,),
    )
    ctx.enqueue_copy(zall_h, zall)
    ctx.enqueue_copy(ret_h, ret)
    ctx.synchronize()

    var bad = 0
    for r in range(BATCH_TOTAL):
        var e = r // TOTAL_SAMPLES
        for kk in range(LATENT):
            if zall_h[r * LATENT + kk] != _expect(e, kk):
                bad += 1
    var bad_ret = 0
    for r in range(BATCH_TOTAL):
        if ret_h[r] != Scalar[DT](0.0):
            bad_ret += 1
    print(
        "  broadcast : ", ELEMS - bad, "/", ELEMS, " elements correct, ",
        BATCH_TOTAL - bad_ret, "/", BATCH_TOTAL, " returns zeroed", sep="",
    )
    if bad != 0:
        raise Error("broadcast left " + String(bad) + " elements stale")
    if bad_ret != 0:
        raise Error("broadcast left " + String(bad_ret) + " returns un-zeroed")


def main() raises:
    var ctx = DeviceContext()
    print("MPPI z-kernel full-coverage gate —", ctx.name())
    print(
        "  BATCH_TOTAL=", BATCH_TOTAL, " LATENT=", LATENT, " elems=", ELEMS,
        "  row-grid would be ", (BATCH_TOTAL + TPB - 1) // TPB,
        " blocks vs element-grid ", Z_BLOCKS, sep="",
    )
    test_copy_z(ctx)
    test_broadcast_z0(ctx)
    print("ALL PASSED")
