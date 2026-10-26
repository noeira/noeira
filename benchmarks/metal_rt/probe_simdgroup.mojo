"""Can Mojo reach Apple's simdgroup matrix hardware? One simdgroup computes
an 8x8x8 product through the AIR intrinsics the Metal compiler itself emits
for `simdgroup_load` / `simdgroup_multiply_accumulate` / `simdgroup_store`
(names read off `xcrun metal -S -emit-llvm` of a 15-line MSL kernel of our
own: see the README). They are not LLVM intrinsics (`llvm_intrinsic` says
"could not find LLVM intrinsic"): they are external functions the Metal
backend resolves, so `external_call` reaches them. Checked against a CPU
reference, plain
and with B loaded transposed."""

from std.sys import exit
from std.ffi import external_call
from max.gpu import thread_idx
from max.gpu.memory import AddressSpace

from noeira.experimental.metal_rt.runtime import MetalRT

comptime F = DType.float32
comptime M64 = SIMD[F, 64]
comptime I2 = SIMD[DType.int64, 2]
comptime GP = Pointer[Float32, MutAnyOrigin]


def mm8(c: GP, a: GP, b: GP, tb: Int64):
    var pa = a.unsafe_address_space_cast[AddressSpace.GLOBAL]()
    var pb = b.unsafe_address_space_cast[AddressSpace.GLOBAL]()
    var pc = c.unsafe_address_space_cast[AddressSpace.GLOBAL]()
    var A = external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", M64](
        pa, I2(8, 8), I2(1, 8), I2(0, 0)
    )
    var B: M64
    if tb != 0:
        B = external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", M64](
            pb, I2(8, 8), I2(8, 1), I2(0, 0)
        )
    else:
        B = external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", M64](
            pb, I2(8, 8), I2(1, 8), I2(0, 0)
        )
    var C = external_call["air.simdgroup_matrix_8x8_init_filled.v64f32.f32", M64](Float32(0))
    var D = external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32", M64
    ](A, B, C)
    external_call["air.simdgroup_matrix_8x8_store.v64f32.p1f32", NoneType](
        D, pc, I2(8, 8), I2(1, 8), I2(0, 0)
    )


def main() raises:
    var rt = MetalRT()
    var a = rt.alloc[F](64)
    var b = rt.alloc[F](64)
    var c = rt.alloc[F](64)
    for i in range(64):
        a.host_ptr[F]()[unsafe_offset=i] = Float32((i * 7) % 13) - 6.0
        b.host_ptr[F]()[unsafe_offset=i] = Float32((i * 5) % 11) * 0.5 - 2.0
    var bad = 0
    for tb in range(2):
        rt.launch[mm8](c.dev[F](), a.dev[F](), b.dev[F](), Int64(tb), grid=1, block=32)
        rt.sync()
        var worst: Float64 = 0
        for i in range(8):
            for j in range(8):
                var s: Float64 = 0
                for k in range(8):
                    var bv = b.host_ptr[F]()[unsafe_offset=(j * 8 + k) if tb != 0 else (k * 8 + j)]
                    s += Float64(a.host_ptr[F]()[unsafe_offset=i * 8 + k]) * Float64(bv)
                worst = max(worst, abs(s - Float64(c.host_ptr[F]()[unsafe_offset=i * 8 + j])))
        print("8x8x8 simdgroup MMA, B", "transposed" if tb != 0 else "plain", ": worst |err| =", worst, "OK" if worst < 1e-4 else "FAIL")
        if worst >= 1e-4:
            bad += 1
    if bad != 0:
        exit(1)
