"""PushT pixel rasterizer smoke test (CPU + GPU)."""

from std.math import pi
from noeira.core.tiles import tile
from max.gpu.host import DeviceContext, DeviceBuffer
from noeira.physics2d import dtype, SHAPE_MAX_SIZE
from noeira.envs.pusht import (
    PushTV2,
    PConstants,
    PushTLayout,
    PushTShapeBuf,
    render_pixel_obs_single,
    render_pixel_obs_kernel_gpu,
    IMG_H,
    IMG_W,
    IMG_C,
)
from noeira.nn.core.ptr import mptr


def cpu_test() raises:
    # Render with block at goal pose, agent at (256, 400).
    var data = Array[Scalar[dtype], IMG_H * IMG_W * IMG_C](
        fill=Scalar[dtype](0.0)
    )
    var out_t = tile[IMG_H, IMG_W, IMG_C](data.unsafe_ptr().unsafe_bitcast[Scalar[dtype]]().as_unsafe_any_origin())
    render_pixel_obs_single(
        Scalar[dtype](256.0),
        Scalar[dtype](256.0),
        Scalar[dtype](pi / 4.0),
        Scalar[dtype](256.0),
        Scalar[dtype](400.0),
        out_t,
    )

    # Count category pixels: white (255,255,255), light slate gray block+goal
    # overlay (color = light_slate_gray since block on top of goal), royal blue
    # (agent), and the rest.
    var white = 0
    var slate = 0
    var blue = 0
    var green = 0
    var gray = 0
    var other = 0
    for r in range(IMG_H):
        for c in range(IMG_W):
            var r_v = Int(out_t[r, c, 0])
            var g_v = Int(out_t[r, c, 1])
            var b_v = Int(out_t[r, c, 2])
            if r_v == 255 and g_v == 255 and b_v == 255:
                white += 1
            elif r_v == 119 and g_v == 136 and b_v == 153:
                slate += 1
            elif r_v == 65 and g_v == 105 and b_v == 225:
                blue += 1
            elif r_v == 144 and g_v == 238 and b_v == 144:
                green += 1
            elif r_v == 211 and g_v == 211 and b_v == 211:
                gray += 1
            else:
                other += 1
    print(
        "CPU pixel counts: white=",
        white,
        " slate(block+goal)=",
        slate,
        " blue(agent)=",
        blue,
        " green(goal-only)=",
        green,
        " gray(wall)=",
        gray,
        " other=",
        other,
    )
    if blue == 0:
        raise Error("agent (royal blue) not rendered")
    if slate == 0:
        raise Error("block (slate gray) not rendered")
    if green == 0:
        # Since block is drawn on top of goal at identical pose, the goal
        # is entirely occluded — this is expected. So this case won't fail.
        print("  (note: goal fully occluded by block at identical pose)")
    if white == 0:
        raise Error("background (white) not rendered")
    if other != 0:
        raise Error("unexpected color in output")
    print("CPU pixel rasterizer OK.")


def gpu_test() raises:
    comptime BATCH = 2
    var ctx = DeviceContext()
    var states = ctx.enqueue_create_buffer[dtype](
        BATCH * PushTLayout.STATE_SIZE
    )
    var pixels = ctx.enqueue_create_buffer[dtype](BATCH * IMG_H * IMG_W * IMG_C)
    var workspace = ctx.enqueue_create_buffer[dtype](
        PushTShapeBuf.NUM_SHAPES * SHAPE_MAX_SIZE
    )

    PushTV2[dtype].init_step_workspace_gpu[BATCH](ctx, workspace)
    PushTV2[dtype].reset_kernel_gpu[BATCH, PushTLayout.STATE_SIZE](
        ctx, states, rng_seed=3
    )
    render_pixel_obs_kernel_gpu[BATCH, PushTLayout.STATE_SIZE](
        ctx, states, pixels
    )

    var host = List[Scalar[dtype]](capacity=BATCH * IMG_H * IMG_W * IMG_C)
    for _ in range(BATCH * IMG_H * IMG_W * IMG_C):
        host.append(Scalar[dtype](0.0))
    ctx.enqueue_copy(mptr(host), pixels)
    ctx.synchronize()

    # Count categories in env 0
    var white = 0
    var blue = 0
    var slate = 0
    var green = 0
    var off = 0
    for r in range(IMG_H):
        for c in range(IMG_W):
            var idx = off + (r * IMG_W + c) * IMG_C
            var r_v = Int(host[idx])
            var g_v = Int(host[idx + 1])
            var b_v = Int(host[idx + 2])
            if r_v == 255 and g_v == 255 and b_v == 255:
                white += 1
            elif r_v == 119 and g_v == 136 and b_v == 153:
                slate += 1
            elif r_v == 65 and g_v == 105 and b_v == 225:
                blue += 1
            elif r_v == 144 and g_v == 238 and b_v == 144:
                green += 1
    print(
        "GPU env=0 pixel counts: white=",
        white,
        " slate=",
        slate,
        " blue=",
        blue,
        " green=",
        green,
    )
    if blue == 0:
        raise Error("GPU: agent not rendered")
    if slate == 0:
        raise Error("GPU: block not rendered")
    print("GPU pixel rasterizer OK.")


def main() raises:
    cpu_test()
    gpu_test()
    print("All PushT render tests passed.")
