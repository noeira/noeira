"""OnlinePongSampler CPU smoke (no neural net — cheap).

Validates the online sim-backed data source two ways:

  1. Direct contract: ``sample_batch_uint8`` fills a (B, T) uint8 pixel batch
     + fp32 one-hot action batch. Checks one-hot actions, valid pixel bytes,
     and that frames actually evolve across the window (env is stepping, not
     emitting a frozen frame).
  2. Drop-in through ``WindowSource[..., BUF=OnlinePongSampler[...]]``: the
     same generic bridge that consumes ``PongOfflineBuffer`` consumes the live
     sampler with no other change, producing fp32 pixels in [0, 1] (÷255) and
     one-hot actions on the CPU target.

Run:  pixi run mojo run -I . tests/data/test_online_sampler.mojo
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.data.window_source import WindowSource
from noeira.envs.arcade_games.pong.offline_buffer import (
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)
from noeira.envs.arcade_games.pong.online_sampler import (
    OnlinePongSampler,
    ScriptedPongPolicy,
)


comptime IMG_DIM = PONG_FRAME_BYTES   # 28224
comptime ACT = PONG_NUM_ACTIONS       # 3
comptime T = 4
comptime B = 2
comptime NPIX = B * T * IMG_DIM
comptime NACT = B * T * ACT

comptime Sampler = OnlinePongSampler[ScriptedPongPolicy, B, T]


def _check_one_hot_fp32(
    act: Pointer[Scalar[DType.float32], MutAnyOrigin]
) raises:
    var bad = 0
    for b in range(B):
        for t in range(T):
            var s: Scalar[DType.float32] = 0.0
            var ones = 0
            for j in range(ACT):
                var a = act[(b * T + t) * ACT + j]
                s += a
                if a == Scalar[DType.float32](1.0):
                    ones += 1
                elif a != Scalar[DType.float32](0.0):
                    bad += 1
            if not (ones == 1 and s == Scalar[DType.float32](1.0)):
                bad += 1
    assert_true(bad == 0, "every (b, t) action row is exactly one-hot")


def main() raises:
    print("=" * 70)
    print("OnlinePongSampler CPU smoke")
    print("=" * 70)
    seed(7)

    # ── 1. Direct sample_batch_uint8 contract ───────────────────────────
    var sampler = Sampler.make(ScriptedPongPolicy(eps=0.3))
    var pix = alloc[Scalar[DType.uint8]](NPIX).as_unsafe_any_origin()
    var act = alloc[Scalar[DType.float32]](NACT).as_unsafe_any_origin()
    sampler.sample_batch_uint8(B, T, pix.as_unsafe_any_origin(), act)

    _check_one_hot_fp32(act.as_unsafe_any_origin())
    print("   [direct] actions one-hot OK")

    # Frames carry content (Pong renders paddles + ball → nonzero pixels).
    var nonzero = 0
    for k in range(NPIX):
        if pix[k] != UInt8(0):
            nonzero += 1
    assert_true(nonzero > 0, "rendered frames are not all-zero")
    print("   [direct] nonzero pixels:", nonzero, "/", NPIX)

    # The env is actually stepping: env 0's first vs last frame in the window
    # differ (ball/paddle move under the scripted policy).
    var diff = 0
    var first = 0 * (T * IMG_DIM) + 0 * IMG_DIM
    var last = 0 * (T * IMG_DIM) + (T - 1) * IMG_DIM
    for i in range(IMG_DIM):
        if pix[first + i] != pix[last + i]:
            diff += 1
    assert_true(diff > 0, "frames evolve across the window (env is stepping)")
    print("   [direct] frame[0] vs frame[T-1] differing pixels:", diff)

    pix.free()
    act.free()
    _ = sampler^

    # ── 2. Drop-in through the generic WindowSource (CPU) ───────────────
    comptime Source = WindowSource[IMG_DIM, ACT, T, B, "cpu", BUF=Sampler]
    var src = Source.make(Sampler.make(ScriptedPongPolicy(eps=0.5)))
    src.next_batch()
    var fpix = src.pix_ptr()
    var fact = src.act_ptr()

    var bad_pix = 0
    for k in range(NPIX):
        var v = fpix[k]
        if v < Scalar[DT](0.0) or v > Scalar[DT](1.0):
            bad_pix += 1
        # quantised back must land on an exact k/255
        var q = v * Scalar[DT](255.0)
        var r = q - Scalar[DT](Int(q + Scalar[DT](0.5)))
        if r.__abs__() >= Scalar[DT](1e-3):
            bad_pix += 1
    assert_true(bad_pix == 0, "WindowSource fp32 pixels are exactly k/255 in [0,1]")
    print("   [WindowSource] pixels checked:", NPIX, " bad:", bad_pix)

    _check_one_hot_fp32(fact)
    print("   [WindowSource] actions one-hot OK")

    _ = src^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
