"""The Q4_0 format and the RTN grid (CPU).

    pixi run python tools/quant/q4_0_rtn_reference.py --out tests/fixtures/quant
    pixi run mojo run -I . tests/nn/quant/test_q4_0.mojo

  [1] RTN: `quantize_rtn_q4_0` on the fixture's 25 x 128 weight gives the
      reference's codes and scales BIT for bit. The fixture is built to fail
      a first-max -> last-max change, round half away from zero, a truncating
      bf16 cast and a reciprocal multiply (its generator's docstring)
  [2] pack ∘ unpack is the identity on random byte strings, NaN and
      subnormal scales included; unpack ∘ pack on random codes and scale bits
  [3] dequantize: every value is (q - 8) · d of its own code and scale, and
      the RTN round trip stays within d / 2 of the weight wherever a code
      was not clipped
  [4] U8 safetensors: blocks written with `add_u8` read back byte for byte
"""

from std.math import abs
from std.random import random_ui64, seed
from std.testing import assert_true

from noeira.io.safetensors import SafeTensors, SafeTensorsWriter, dtype_name
from noeira.nn.quant.q4_0 import (
    Q4_GROUP, Q4_BLOCK, q4_0_row_bytes, quantize_rtn_q4_0, pack_q4_0,
    unpack_q4_0, rtn_q4_0_blocks, dequantize_q4_0, f16_bits, f16_value,
)
from std.memory import bitcast

comptime FIX = "tests/fixtures/quant/q4_0_rtn.safetensors"


def rtn_vs_reference() raises:
    var f = SafeTensors(String(FIX))
    var shape = f.shape(String("w"))
    var n = shape[0]
    var k = shape[1]
    var w = f.read_f32(String("w"))
    var q_ref = f.read_raw(String("q"))
    var d_ref = f.read_f32(String("d"))
    var q = List[UInt8]()
    var d = List[Float32]()
    quantize_rtn_q4_0(w, n, k, q, d)
    var bad_q = 0
    for i in range(n * k):
        if q[i] != q_ref[i]:
            bad_q += 1
    var bad_d = 0
    for i in range(len(d_ref)):
        if bitcast[DType.uint32](d[i]) != bitcast[DType.uint32](d_ref[i]):
            bad_d += 1
    print("  [1] RTN vs the reference, [", n, "x", k, "]: codes differing",
          bad_q, "of", n * k, "| scales differing (bits)", bad_d, "of",
          len(d_ref))
    assert_true(bad_q == 0 and bad_d == 0, "RTN differs from the reference")


def round_trips() raises:
    seed(7)
    var trials = 0
    for t in range(6):
        var n = 1 + t
        var k = Q4_GROUP * (1 + 2 * t)
        var nb = n * q4_0_row_bytes(k)
        var blocks = List[UInt8](capacity=nb)
        for _ in range(nb):
            blocks.append(UInt8(Int(random_ui64(0, 255))))
        # A NaN, an infinity and a subnormal scale in the first blocks.
        blocks[0] = 0x01
        blocks[1] = 0x7E      # fp16 NaN 0x7E01
        if nb >= 2 * Q4_BLOCK:
            blocks[Q4_BLOCK] = 0x00
            blocks[Q4_BLOCK + 1] = 0x7C   # +inf
        if nb >= 3 * Q4_BLOCK:
            blocks[2 * Q4_BLOCK] = 0x03
            blocks[2 * Q4_BLOCK + 1] = 0x00   # subnormal
        var q = List[UInt8]()
        var bits = List[UInt16]()
        unpack_q4_0(blocks, n, k, q, bits)
        var again = pack_q4_0(q, bits, n, k)
        for i in range(nb):
            assert_true(again[i] == blocks[i], "pack ∘ unpack changed a byte")
        # The other direction: codes 0..15 and any 16-bit scale.
        var q2 = List[UInt8](capacity=n * k)
        for _ in range(n * k):
            q2.append(UInt8(Int(random_ui64(0, 15))))
        var b2 = List[UInt16](capacity=n * (k // Q4_GROUP))
        for _ in range(n * (k // Q4_GROUP)):
            b2.append(UInt16(Int(random_ui64(0, 65535))))
        var packed = pack_q4_0(q2, b2, n, k)
        var q3 = List[UInt8]()
        var b3 = List[UInt16]()
        unpack_q4_0(packed, n, k, q3, b3)
        for i in range(n * k):
            assert_true(q3[i] == q2[i], "unpack ∘ pack changed a code")
        for i in range(len(b2)):
            assert_true(b3[i] == b2[i], "unpack ∘ pack changed a scale")
        trials += 1
    print("  [2] pack ∘ unpack = identity on", trials,
          "random byte strings (NaN, inf and subnormal scales included);"
          " unpack ∘ pack on random codes and scales")


def dequantize() raises:
    var f = SafeTensors(String(FIX))
    var shape = f.shape(String("w"))
    var n = shape[0]
    var k = shape[1]
    var w = f.read_f32(String("w"))
    var blocks = rtn_q4_0_blocks(w, n, k)
    assert_true(len(blocks) == n * q4_0_row_bytes(k), "block count")
    var deq = dequantize_q4_0(blocks, n, k)
    var q = List[UInt8]()
    var bits = List[UInt16]()
    unpack_q4_0(blocks, n, k, q, bits)
    # The grid's own scales, before fp16 storage.
    var q_rtn = List[UInt8]()
    var d_rtn = List[Float32]()
    quantize_rtn_q4_0(w, n, k, q_rtn, d_rtn)
    var worst = Float64(0)
    var checked = 0
    var clipped = 0
    var moved = 0
    var groups = k // Q4_GROUP
    for gi in range(n * groups):
        var stored = f16_value(bits[gi])
        if stored != d_rtn[gi]:
            # fp16 storage changed the scale: only below fp16's normal range
            # (2^-14), where fp16 has fewer bits than bf16.
            moved += 1
            assert_true(
                abs(d_rtn[gi]) < Float32(6.103515625e-05),
                "fp16 storage changed a normal-range scale",
            )
    for r in range(n):
        for c in range(k):
            var gi = r * groups + c // Q4_GROUP
            var dg = f16_value(bits[gi])
            var want = (Float32(Int(q[r * k + c])) - 8.0) * dg
            assert_true(
                bitcast[DType.uint32](deq[r * k + c]) == bitcast[DType.uint32](want),
                "dequantize is not (q - 8) d",
            )
            if dg == 0.0 or dg != d_rtn[gi]:
                continue
            if q[r * k + c] == 0 or q[r * k + c] == 15:
                clipped += 1
                continue
            # Unclipped, on a scale fp16 stored exactly: within half a step.
            var e = Float64(abs(deq[r * k + c] - w[r * k + c])) / Float64(abs(dg))
            worst = max(worst, e)
            checked += 1
    print("  [3] dequantize = (q - 8) d at all", n * k,
          "values; worst |deq - w| / |d| over", checked,
          "unclipped values:", worst, "(", clipped, "at the code ends;",
          moved, "scales below fp16's normal range, stored inexactly )")
    assert_true(worst <= 0.5 + 1e-6, "an RTN value is more than d / 2 off")


def u8_file() raises:
    var n = 3
    var k = 64
    var blocks = List[UInt8]()
    for i in range(n * q4_0_row_bytes(k)):
        blocks.append(UInt8((i * 37 + 11) % 256))
    var wr = SafeTensorsWriter()
    var shape: List[Int] = [n, q4_0_row_bytes(k)]
    wr.add_u8(String("proj.q4_0"), shape, blocks, len(blocks))
    var path = String("/tmp/noeira_test_q4_0_u8.safetensors")
    wr.save(path)
    var f = SafeTensors(path)
    var back = f.read_raw(String("proj.q4_0"))
    assert_true(dtype_name(f.dtype_of(String("proj.q4_0"))) == "U8", "dtype")
    var s = f.shape(String("proj.q4_0"))
    assert_true(s[0] == n and s[1] == q4_0_row_bytes(k), "shape")
    assert_true(len(back) == len(blocks), "length")
    for i in range(len(blocks)):
        assert_true(back[i] == blocks[i], "a byte changed in the file")
    print("  [4] U8 safetensors:", len(blocks), "bytes back byte for byte")


def main() raises:
    print("Q4_0 format and RTN grid (CPU)")
    rtn_vs_reference()
    round_trips()
    dequantize()
    u8_file()
    print("PASSED")
