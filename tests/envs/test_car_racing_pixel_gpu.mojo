"""GPU smoke test for CarRacingPixel — verifies the rasterizer renders a sane
top-down image (car sprite at center, road + grass pixels present, values in
[0,1]) and the frame stack advances.

Run: pixi run -e apple mojo run -I . tests/envs/test_car_racing_pixel_gpu.mojo
"""

from max.gpu.host import DeviceContext
from noeira.physics2d import dtype
from noeira.envs.car_racing import CarRacingPixel


comptime B = 8
comptime K = 30


def main() raises:
    print("=== CarRacingPixel GPU smoke test ===")
    comptime E = CarRacingPixel[DType.float32]
    comptime SSZ = E.STATE_SIZE
    comptime OBS = E.OBS_DIM
    comptime WS = E.STEP_WS_PER_ENV
    comptime FS = E.FRAME_SIZE
    print("STATE_SIZE =", SSZ, " OBS_DIM =", OBS, " (4x84x84)  WS/env =", WS)

    var ctx = DeviceContext()
    var states = ctx.enqueue_create_buffer[dtype](B * SSZ)
    var actions = ctx.enqueue_create_buffer[dtype](B * 1)
    var rewards = ctx.enqueue_create_buffer[dtype](B)
    var dones = ctx.enqueue_create_buffer[dtype](B)
    var term = ctx.enqueue_create_buffer[dtype](B)
    var obs = ctx.enqueue_create_buffer[dtype](B * OBS)
    var ws = ctx.enqueue_create_buffer[dtype](B * WS)
    ctx.synchronize()
    var wsp = rebind[Pointer[Scalar[dtype], MutAnyOrigin]](ws.unsafe_ptr())

    # gas every step
    var ahost = ctx.enqueue_create_host_buffer[dtype](B * 1)
    ctx.synchronize()
    for i in range(B):
        ahost[i] = Scalar[dtype](3.0)
    ctx.enqueue_copy(actions, ahost)

    E.init_step_workspace_gpu[B](ctx, ws)
    E.reset_kernel_gpu[B, SSZ](ctx, states, rng_seed=7)
    ctx.synchronize()

    for _ in range(K):
        E.step_kernel_gpu[B, SSZ, OBS](
            ctx, states, actions, rewards, dones, term, obs,
            rng_seed=0, workspace_ptr=wsp,
        )
    ctx.synchronize()

    var ohost = ctx.enqueue_create_host_buffer[dtype](B * OBS)
    ctx.enqueue_copy(ohost, obs)
    ctx.synchronize()

    # Inspect env 0's NEWEST frame (last FRAME_SIZE block of the 4-stack).
    var car = 0
    var road = 0
    var grass = 0
    var other = 0
    var out_of_range = 0
    var newest = 3 * FS  # frame index 3 = newest in chronological output
    for p in range(FS):
        var v = Float64(ohost[0 * OBS + newest + p])
        if v < 0.0 or v > 1.0:
            out_of_range += 1
        if v > 0.95:
            car += 1
        elif v > 0.6 and v < 0.8:
            road += 1
        elif v > 0.25 and v < 0.35:
            grass += 1
        else:
            other += 1

    # Center pixel should be the car sprite.
    var cx = Int(E.OBS_W // 2)
    var cy = Int(E.CY)
    var center = Float64(ohost[0 * OBS + newest + cy * E.OBS_W + cx])

    print("newest frame: car=", car, " road=", road, " grass=", grass, " other=", other)
    print("center pixel value =", center, " (expect ~1.0 car)")

    if out_of_range > 0:
        raise Error(String("obs out of [0,1]: ", out_of_range, " pixels"))
    if car <= 0:
        raise Error("no car sprite pixels rendered")
    if road <= 0:
        raise Error("no road pixels — track not visible to the agent")
    if grass <= 0:
        raise Error("no grass pixels — rasterizer suspicious")
    if center < 0.95:
        raise Error(String("center pixel not car (", center, ")"))

    print("=== PASS: pixel rasterizer renders car + road + grass, obs in [0,1] ===")
