"""Kernel-argument ABI and ordering checks for `MetalRT` (exits nonzero on
failure).

1. Mixed arguments (pointer, Int, Float32, pointer) bound by index with
   `setBytes`, a multi-threadgroup grid, `block_idx`/`block_dim`/`thread_idx`.
2. 2000 dependent 1-thread launches under every commit policy and residency
   mode: the counter must read exactly 2000.
"""

from max.gpu import block_dim, block_idx, thread_idx, global_idx
from std.sys import exit

from noeira.experimental.metal_rt.runtime import (
    MetalRT,
    COMMIT_EACH,
    COMMIT_EVERY_N,
    COMMIT_ADAPTIVE,
    RESIDENCY_SET,
    RESIDENCY_PER_CB,
    RESIDENCY_PER_LAUNCH,
    policy_name,
)

comptime P = Pointer[Float32, MutAnyOrigin]


def abi_kernel(dst: P, n: Int, scale: Float32, src: P):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        dst[unsafe_offset=i] = src[unsafe_offset=i] * scale + Float32(
            block_idx.x
        ) * 1000.0


def bump(x: P):
    if global_idx.x == 0:
        x[unsafe_offset=0] = x[unsafe_offset=0] + 1.0


def main() raises:
    var fails = 0
    var rt = MetalRT()
    print("device:", rt.name())
    comptime N = 1000
    var a = rt.alloc[DType.float32](N)
    var b = rt.alloc[DType.float32](N)
    for i in range(N):
        a.host_ptr[DType.float32]()[unsafe_offset=i] = Float32(i)
        b.host_ptr[DType.float32]()[unsafe_offset=i] = -1.0
    rt.launch[abi_kernel](
        b.dev[DType.float32](), N - 3, Float32(0.5), a.dev[DType.float32](),
        grid=(N + 63) // 64, block=64,
    )
    rt.sync()
    var bad = 0
    for i in range(N):
        var want: Float32 = Float32(i) * 0.5 + Float32(i // 64) * 1000.0
        if i >= N - 3:
            want = -1.0
        if b.host_ptr[DType.float32]()[unsafe_offset=i] != want:
            if bad < 5:
                print("  abi mismatch at", i, b.host_ptr[DType.float32]()[unsafe_offset=i], "want", want)
            bad += 1
    print("abi (ptr, Int, Float32, ptr) over", (N + 63) // 64, "groups:", "OK" if bad == 0 else "FAIL")
    if bad != 0:
        fails += 1

    for res in [RESIDENCY_SET, RESIDENCY_PER_CB, RESIDENCY_PER_LAUNCH]:
        for pol in [COMMIT_EACH, COMMIT_EVERY_N, COMMIT_ADAPTIVE]:
            var r2 = MetalRT(policy=pol, batch_n=16, residency=res)
            var c = r2.alloc[DType.float32](1)
            c.host_ptr[DType.float32]()[unsafe_offset=0] = 0.0
            for _ in range(2000):
                r2.launch[bump](c.dev[DType.float32](), grid=1, block=1)
            r2.sync()
            var v = c.host_ptr[DType.float32]()[unsafe_offset=0]
            print("order: residency", res, "policy", policy_name(pol), "count", v,
                  "commits", r2.n_commits, "OK" if v == 2000.0 else "FAIL")
            if v != 2000.0:
                fails += 1
    if fails != 0:
        print("FAILED:", fails)
        exit(1)
    print("ALL OK")
