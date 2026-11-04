"""WindowSource GPU bridge smoke (no neural net — cheap).

Validates the GPU target path of the bridge — sample uint8 window → H2D →
device uint8→fp32 ÷255 — by D2H-ing the result and checking it equals the
stored bytes / 255. Uses a tiny buffer (few frames) and small B/T so it
runs fast on Apple; the 84×84 frame size is fixed by PongOfflineBuffer but
nothing here runs the model.

Run:  pixi run -e apple mojo run -I . tests/data/test_pong_data_gpu.mojo
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.data.window_source import WindowSource
from noeira.envs.arcade_games.pong.offline_buffer import (
    PongOfflineBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)


comptime IMG_DIM = PONG_FRAME_BYTES   # 28224
comptime ACT = PONG_NUM_ACTIONS       # 3
comptime T = 4
comptime B = 2
comptime N_FRAMES = 24
comptime NPIX = B * T * IMG_DIM
comptime NACT = B * T * ACT


def main() raises:
    print("=" * 70)
    print("WindowSource GPU bridge smoke")
    print("=" * 70)
    seed(7)
    var ctx = DeviceContext()

    var buf = PongOfflineBuffer(capacity=N_FRAMES)
    for n in range(N_FRAMES):
        for i in range(PONG_FRAME_BYTES):
            buf.frames[n * PONG_FRAME_BYTES + i] = UInt8((n * 13 + i) % 256)
        buf.actions[n] = UInt8(n % PONG_NUM_ACTIONS)
        buf.dones[n] = UInt8(0)
    buf.n_frames = N_FRAMES

    var src = WindowSource[IMG_DIM, ACT, T, B, "gpu", PongOfflineBuffer].make(buf^, ctx=ctx)
    src.next_batch()

    # D2H the device fp32 pixels + actions and validate against /255.
    var pix_h = ctx.enqueue_create_host_buffer[DT](NPIX)
    var act_h = ctx.enqueue_create_host_buffer[DT](NACT)
    var pix_dev = DeviceBuffer[DT](ctx, src.pix_ptr(), NPIX, owning=False)
    var act_dev = DeviceBuffer[DT](ctx, src.act_ptr(), NACT, owning=False)
    ctx.enqueue_copy(pix_h, pix_dev)
    ctx.enqueue_copy(act_h, act_dev)
    ctx.synchronize()

    var bad_pix = 0
    for k in range(NPIX):
        var v = pix_h.unsafe_ptr()[k]
        if v < Scalar[DT](0.0) or v > Scalar[DT](1.0):
            bad_pix += 1
        var q = v * Scalar[DT](255.0)
        var r = q - Scalar[DT](Int(q + Scalar[DT](0.5)))
        if r.__abs__() >= Scalar[DT](1e-3):
            bad_pix += 1
    print("   pixels checked:", NPIX, " bad:", bad_pix)
    assert_true(bad_pix == 0, "all device pixels are exactly k/255 in [0,1]")

    var bad_act = 0
    for b in range(B):
        for t in range(T):
            var s: Scalar[DT] = 0.0
            for j in range(ACT):
                s += act_h.unsafe_ptr()[(b * T + t) * ACT + j]
            if s != Scalar[DT](1.0):
                bad_act += 1
    print("   action rows checked:", B * T, " bad:", bad_act)
    assert_true(bad_act == 0, "device actions are one-hot")

    _ = src^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
