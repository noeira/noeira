"""Pong buffer roundtrip test.

Builds a tiny buffer (10 fake frames with known patterns), exercises
add_step → save → load → sample. Validates pixel values survive the
uint8 round-trip and one-hot action expansion is correct.
"""

from std.memory import alloc
from std.random import seed
from std.os import os

from noeira.envs.arcade_games.pong.offline_buffer import (
    PongOfflineBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)

comptime dtype = DType.float32


def main() raises:
    seed(0xABCD)
    print("=" * 70)
    print("PongOfflineBuffer roundtrip test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Build a buffer with 12 frames:
    #   - 6 frames from "episode 0" (dones[5] = 1, others 0)
    #   - 6 frames from "episode 1" (dones[11] = 1, others 0)
    # Frame `i` is filled with a constant pixel value `i * 20` ∈ [0, 255]
    # so we can verify quantization round-trip.
    # ------------------------------------------------------------------
    var buf = PongOfflineBuffer(capacity=16)
    var scratch = alloc[Scalar[dtype]](PONG_FRAME_BYTES).as_unsafe_any_origin()
    for i in range(12):
        var fill_val = Scalar[dtype](i * 20) / 255.0
        for k in range(PONG_FRAME_BYTES):
            scratch[k] = fill_val
        var done = (i == 5) or (i == 11)
        var act = i % 3
        buf.add_step_fp32(scratch.as_unsafe_any_origin(), act, done)

    print("n_frames:", buf.n_frames)
    assert_eq(buf.n_frames, 12, "n_frames")

    # ------------------------------------------------------------------
    # Save / load roundtrip
    # ------------------------------------------------------------------
    var tmp_path = String("/tmp/test_pong_buffer.bin")
    buf.save(tmp_path)
    var loaded = PongOfflineBuffer.load(tmp_path)
    assert_eq(loaded.n_frames, 12, "loaded n_frames")

    # Spot-check a few raw bytes survived the round-trip.
    for i in range(12):
        var base = i * PONG_FRAME_BYTES
        var expected_byte = UInt8(i * 20)
        if loaded.frames[base] != expected_byte:
            print(
                "  byte mismatch @ frame",
                i,
                "expected",
                Int(expected_byte),
                "got",
                Int(loaded.frames[base]),
            )
            raise Error("PongOfflineBuffer roundtrip: frame bytes diverged")
    print("Frame byte roundtrip OK")

    # Action + done bytes
    for i in range(12):
        if Int(loaded.actions[i]) != (i % 3):
            raise Error("PongOfflineBuffer roundtrip: action mismatch")
    if Int(loaded.dones[5]) != 1 or Int(loaded.dones[11]) != 1:
        raise Error("PongOfflineBuffer roundtrip: done flag mismatch")
    print("Action/done roundtrip OK")

    # ------------------------------------------------------------------
    # Sample batch — B=4, T=4. Episode boundaries at indices 5 and 11.
    # Valid starts (windows of length 4 with no done in first T-1 = 3
    # frames, i.e. no done at start, start+1, start+2):
    #   start ∈ {0, 1, 2, 3, 6, 7, 8} — start=4 has dones[5]=1 ⇒ invalid;
    #   start=5 has dones[5]=1 ⇒ invalid; start=9..11 too close to end
    #   (end is n_frames-T=8 so max start is 8; 9+ rejected upstream).
    # ------------------------------------------------------------------
    comptime B = 4
    comptime T = 4
    # Pixels now come back as raw uint8; the fp32 normalize moved onto the
    # GPU (`pixels_uint8_to_fp32_kernel`), so the buffer hands out bytes.
    var pix_out = alloc[Scalar[DType.uint8]](
        B * T * PONG_FRAME_BYTES
    ).as_unsafe_any_origin()
    var act_out = alloc[Scalar[dtype]](B * T * PONG_NUM_ACTIONS).as_unsafe_any_origin()
    for i in range(B * T * PONG_FRAME_BYTES):
        pix_out[i] = 0
    for i in range(B * T * PONG_NUM_ACTIONS):
        act_out[i] = -1.0

    loaded.sample_batch_uint8(B, T, pix_out, act_out)

    # Validate every action slot is one-hot (sums to 1.0).
    var bad_onehot = 0
    for b in range(B):
        for t in range(T):
            var s: Scalar[dtype] = 0.0
            for k in range(PONG_NUM_ACTIONS):
                var v = act_out[
                    b * T * PONG_NUM_ACTIONS + t * PONG_NUM_ACTIONS + k
                ]
                s += v
                if v != 0.0 and v != 1.0:
                    bad_onehot += 1
            if s != 1.0:
                bad_onehot += 1
    if bad_onehot != 0:
        raise Error("PongOfflineBuffer sample: actions not one-hot")
    print("One-hot actions OK")

    # Validate pixels in [0, 1] and that within a single window, pixels
    # are roughly constant per timestep (each frame fills with a single
    # value).
    var pix_min = Scalar[DType.uint8](255)
    var pix_max = Scalar[DType.uint8](0)
    for i in range(B * T * PONG_FRAME_BYTES):
        var v = pix_out[i]
        if v < pix_min:
            pix_min = v
        if v > pix_max:
            pix_max = v
    print("pixel range:", Float64(pix_min), "to", Float64(pix_max))
    assert_in_range(Scalar[dtype](Float64(pix_min) / 255.0), 0.0, 1.0, "pix_min")
    assert_in_range(Scalar[dtype](Float64(pix_max) / 255.0), 0.0, 1.0, "pix_max")

    pix_out.free()
    act_out.free()
    scratch.free()

    # ------------------------------------------------------------------
    # Window-validity probe: build a buffer where every odd frame is
    # done, T=2 windows starting at even indices should be valid.
    # ------------------------------------------------------------------
    var buf2 = PongOfflineBuffer(capacity=10)
    var scratch2 = alloc[Scalar[dtype]](PONG_FRAME_BYTES).as_unsafe_any_origin()
    for i in range(PONG_FRAME_BYTES):
        scratch2[i] = 0.5
    for i in range(10):
        var done = (i % 2 == 1)
        buf2.add_step_fp32(scratch2.as_unsafe_any_origin(), 0, done)
    scratch2.free()

    # _window_is_valid for T=2 means dones[start] == 0 (the only frame
    # in the [start, start+T-2] = [start, start] window).
    for s in range(9):
        var expected = (s % 2 == 0)
        var actual = buf2._window_is_valid(s, 2)
        if actual != expected:
            print(
                "  window validity mismatch @ start",
                s,
                "expected",
                expected,
                "got",
                actual,
            )
            raise Error("PongOfflineBuffer window validity probe failed")
    print("Window validity probe OK")

    print()
    print("All PongOfflineBuffer tests passed.")


def assert_eq(actual: Int, expected: Int, label: String) raises:
    if actual != expected:
        raise Error(
            String("  ")
            + label
            + ": expected "
            + String(expected)
            + ", got "
            + String(actual)
        )


def assert_in_range(
    v: Scalar[dtype], lo: Scalar[dtype], hi: Scalar[dtype], label: String
) raises:
    if v < lo or v > hi:
        raise Error(label + ": out of range")
