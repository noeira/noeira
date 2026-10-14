"""Q4_0: llama.cpp's 4-bit weight format, the one MAX's int4 GEMM consumes.

    group   32 consecutive weights of one output row, along IN
    block   18 bytes per group: the scale d as fp16 (little-endian), then 16
            bytes of nibbles, byte j holding element j in its low nibble and
            element j + 16 in its high one
    value   (q - 8) · d,  q in [0, 15]           (4.5 bits per weight)

## Orientation

A Q4_0 weight is [OUT, IN], grouped along IN: torch's orientation, and the one
MAX's GEMM reads (`b` is [N, K]). noeira's `Linear` stores [IN, OUT]; Q4_0 is
the one place the convention flips. A row of OUT holds IN / 32 blocks.

## The RTN grid

`quantize_rtn_q4_0` is round-to-nearest on the grid of the reference GPTQ
pipeline that produced Kev's int4 weights:

    d = (the group's element of largest magnitude, the FIRST one on a tie) / -8
    d = 1e-8 where that is 0, then d rounded to bf16 (ties to even)
    q = clip(round(w / d) + 8, 0, 15), with round's ties to even

`d` is rounded to bf16 because MAX's GEMM reads the fp16 scale and truncates
it to bf16 (`qmatmul_gpu.mojo`): a bf16 d is exact in fp16 over fp16's normal
range, so the value MAX multiplies by is the one the grid chose. A group whose
d falls below fp16's normal range (all |w| < 4.9e-4) loses bits when the scale
is stored; d = 1e-8 (a zero group) flushes to 0, which still decodes to 0.

The division is `w / d`, not `w · (1 / d)` as in llama.cpp's own quantizer,
because the reference divides: the two differ in the last bit, which can move
a code across a rounding boundary.

## Bytes are the interchange

`pack_q4_0` / `unpack_q4_0` move between blocks and (codes, fp16 scale bits).
The scale stays as its fp16 BITS through an unpack, so pack ∘ unpack is the
identity on any byte string, NaN scales included.
"""

from std.math import floor
from std.memory import bitcast
from max.gpu import global_idx

from noeira.io.safetensors import f32_to_bf16_bits
from noeira.nn.constants import DT


comptime Q4_GROUP: Int = 32
"""Weights per group (and per scale)."""
comptime Q4_BLOCK: Int = 18
"""Bytes per group: a 2-byte fp16 scale and 16 bytes of nibbles."""


@always_inline
def q4_0_row_bytes(k: Int) -> Int:
    """Bytes of one [K]-wide row: K / 32 blocks of 18."""
    return (k // Q4_GROUP) * Q4_BLOCK


@always_inline
def bf16_round(x: Float32) -> Float32:
    """`x` rounded to the nearest bfloat16, ties to even, as an fp32."""
    return bitcast[DType.float32](UInt32(f32_to_bf16_bits(x)) << UInt32(16))


@always_inline
def round_half_even(x: Float32) -> Float32:
    """`x` rounded to the nearest integer, ties to even (numpy's `round`).

    Written out instead of the `x + 1.5 · 2^23 - 1.5 · 2^23` trick, which LLVM
    folds away. `x - floor(x)` is exact for |x| < 2^23, so the tie test is
    exact; the grid only ever rounds |x| <= ~8."""
    var f = floor(x)
    var frac = x - f
    if frac > 0.5:
        return f + 1.0
    if frac < 0.5:
        return f
    var half = f * 0.5
    return f if floor(half) == half else f + 1.0


@always_inline
def f16_bits(x: Float32) -> UInt16:
    """The fp16 bits of `x`, rounded to nearest even (exact for a bf16-rounded
    scale in fp16's normal range)."""
    return bitcast[DType.uint16](x.cast[DType.float16]())


@always_inline
def f16_value(bits: UInt16) -> Float32:
    """The fp32 value of fp16 `bits` (exact: fp32 holds every fp16)."""
    return bitcast[DType.float16](bits).cast[DType.float32]()


@always_inline
def q4_0_group_scale(g: Pointer[Float32, MutAnyOrigin]) -> Float32:
    """The grid's scale for the 32 weights at `g` (module docstring): the
    FIRST element of largest magnitude over -8, 1e-8 if that is 0, rounded
    to bf16. The one definition RTN and GPTQ share."""
    var best = 0
    var best_abs = abs(g[unsafe_offset=0])
    for j in range(1, Q4_GROUP):
        var a = abs(g[unsafe_offset=j])
        if a > best_abs:
            best_abs = a
            best = j
    var d = g[unsafe_offset=best] / Float32(-8.0)
    if d == 0.0:
        d = Float32(1e-8)
    return bf16_round(d)


@always_inline
def q4_0_code(w: Float32, d: Float32) -> UInt8:
    """clip(round(w / d) + 8, 0, 15), ties to even: the code of `w` at
    scale `d`. Its value is (code - 8) · d."""
    var v = round_half_even(w / d) + 8.0
    v = min(max(v, Float32(0.0)), Float32(15.0))
    return UInt8(Int(v))


def quantize_rtn_q4_0(
    ref w: List[Float32],
    n: Int,
    k: Int,
    mut q: List[UInt8],
    mut d: List[Float32],
) raises:
    """Round-to-nearest of a row-major [n, k] fp32 weight on the Q4_0 grid
    (module docstring): `q` gets the [n, k] codes in 0..15, `d` the
    [n, k / 32] scales as fp32 (bf16-rounded). `pack_q4_0` turns them into
    blocks; this split is what the reference comparison checks bit for bit."""
    if k % Q4_GROUP != 0:
        raise Error(
            "quantize_rtn_q4_0: k = " + String(k) + " is not a multiple of 32"
        )
    if len(w) < n * k:
        raise Error(
            "quantize_rtn_q4_0: " + String(len(w)) + " values for a ["
            + String(n) + ", " + String(k) + "] weight"
        )
    var groups = k // Q4_GROUP
    q = List[UInt8](length=n * k, fill=0)
    d = List[Float32](length=n * groups, fill=0)
    var wp = rebind[Pointer[Float32, MutAnyOrigin]](w.unsafe_ptr())
    for r in range(n):
        for g in range(groups):
            var base = r * k + g * Q4_GROUP
            var dg = q4_0_group_scale(wp.unsafe_offset(base))
            d[r * groups + g] = dg
            for j in range(Q4_GROUP):
                q[base + j] = q4_0_code(wp[unsafe_offset=base + j], dg)


def pack_q4_0(
    ref q: List[UInt8], ref d_bits: List[UInt16], n: Int, k: Int
) raises -> List[UInt8]:
    """Codes [n, k] (0..15) and fp16 scale bits [n, k / 32] -> Q4_0 blocks
    [n, k / 32 · 18]."""
    var groups = k // Q4_GROUP
    if k % Q4_GROUP != 0 or len(q) < n * k or len(d_bits) < n * groups:
        raise Error("pack_q4_0: inconsistent sizes")
    var out = List[UInt8](length=n * groups * Q4_BLOCK, fill=0)
    for r in range(n):
        for g in range(groups):
            var blk = (r * groups + g) * Q4_BLOCK
            var h = d_bits[r * groups + g]
            out[blk] = UInt8(Int(h & 0xFF))
            out[blk + 1] = UInt8(Int(h >> 8))
            var base = r * k + g * Q4_GROUP
            for j in range(Q4_GROUP // 2):
                var lo = q[base + j] & 0xF
                var hi = q[base + j + Q4_GROUP // 2] & 0xF
                out[blk + 2 + j] = lo | (hi << 4)
    return out^


def unpack_q4_0(
    ref blocks: List[UInt8],
    n: Int,
    k: Int,
    mut q: List[UInt8],
    mut d_bits: List[UInt16],
) raises:
    """Q4_0 blocks [n, k / 32 · 18] -> codes [n, k] and fp16 scale bits
    [n, k / 32]. The inverse of `pack_q4_0` on every byte string."""
    var groups = k // Q4_GROUP
    if k % Q4_GROUP != 0 or len(blocks) < n * groups * Q4_BLOCK:
        raise Error("unpack_q4_0: inconsistent sizes")
    q = List[UInt8](length=n * k, fill=0)
    d_bits = List[UInt16](length=n * groups, fill=0)
    for r in range(n):
        for g in range(groups):
            var blk = (r * groups + g) * Q4_BLOCK
            d_bits[r * groups + g] = (
                UInt16(Int(blocks[blk])) | (UInt16(Int(blocks[blk + 1])) << 8)
            )
            var base = r * k + g * Q4_GROUP
            for j in range(Q4_GROUP // 2):
                var byte = blocks[blk + 2 + j]
                q[base + j] = byte & 0xF
                q[base + j + Q4_GROUP // 2] = byte >> 4


def rtn_q4_0_blocks(ref w: List[Float32], n: Int, k: Int) raises -> List[UInt8]:
    """`quantize_rtn_q4_0` then `pack_q4_0`: a row-major [n, k] fp32 weight
    (torch's [out, in]) to Q4_0 blocks."""
    var q = List[UInt8]()
    var d = List[Float32]()
    quantize_rtn_q4_0(w, n, k, q, d)
    var bits = List[UInt16](capacity=len(d))
    for i in range(len(d)):
        bits.append(f16_bits(d[i]))
    return pack_q4_0(q, bits, n, k)


def dequantize_q4_0(ref blocks: List[UInt8], n: Int, k: Int) raises -> List[Float32]:
    """Q4_0 blocks -> the [n, k] fp32 values `(q - 8) · d`, d read from its
    fp16 bits. On the host; `dequant_q4_0_kernel` is the device twin."""
    var groups = k // Q4_GROUP
    if k % Q4_GROUP != 0 or len(blocks) < n * groups * Q4_BLOCK:
        raise Error("dequantize_q4_0: inconsistent sizes")
    var out = List[Float32](length=n * k, fill=0)
    for r in range(n):
        for g in range(groups):
            var blk = (r * groups + g) * Q4_BLOCK
            var dg = f16_value(
                UInt16(Int(blocks[blk])) | (UInt16(Int(blocks[blk + 1])) << 8)
            )
            var base = r * k + g * Q4_GROUP
            for j in range(Q4_GROUP // 2):
                var byte = blocks[blk + 2 + j]
                out[base + j] = (Float32(Int(byte & 0xF)) - 8.0) * dg
                out[base + j + Q4_GROUP // 2] = (
                    (Float32(Int(byte >> 4)) - 8.0) * dg
                )
    return out^


def dequant_q4_0_kernel(
    src: Pointer[UInt8, MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
    k_arg: Int64,
):
    """`dst[r, c] = (q - 8) · d` over Q4_0 blocks `src` [n, k / 32 · 18], one
    thread per output element of the row-major [n, k] `dst`, so the writes
    coalesce; the 32 threads of a group read the same two scale bytes."""
    var n = Int(n_arg)
    var k = Int(k_arg)
    var i = Int(global_idx.x)
    if i < n * k:
        var r = i // k
        var c = i % k
        var g = c // Q4_GROUP
        var j = c % Q4_GROUP
        var blk = (r * (k // Q4_GROUP) + g) * Q4_BLOCK
        var bits = UInt16(Int(src[unsafe_offset=blk])) | (
            UInt16(Int(src[unsafe_offset=blk + 1])) << 8
        )
        var dg = bitcast[DType.float16](bits).cast[DType.float32]()
        var byte = src[unsafe_offset=blk + 2 + (j % (Q4_GROUP // 2))]
        var nib = (byte & 0xF) if j < Q4_GROUP // 2 else (byte >> 4)
        dst[unsafe_offset=i] = (Float32(Int(nib)) - 8.0) * dg
