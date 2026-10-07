# +--------------------------------------------------------------------------+ #
# | Gated DeltaNet recurrence: MAX's kernel vs a noeira kernel, on the Orin
# +--------------------------------------------------------------------------+ #
"""The recurrence of Qwen3.5's linear-attention layers — HALF of Kev-9B's GPU
time on the Jetson Orin, because mlx-lm runs it as a per-token Python loop
off Apple GPUs (4-5 kernels per token per layer). Two fused kernels for the
same op, same inputs, checked against one float64 CPU reference:

  MAX     `state_space.gated_delta.gated_delta_recurrence_fwd_gpu` (MAX 26.6,
          imported from the shipped state_space.mojoc): one CTA per value
          head, one thread per value column holding the whole 128-float state
          column; every thread recomputes both 128-element L2 norms per token.
  noeira  `gd_split_kernel[SPLIT]`: a column is split across SPLIT adjacent
          lanes (KD/SPLIT state floats each); the norms and both dot products
          are partial sums over the lane's slice, completed by log2(SPLIT)
          `shuffle_xor` steps. SPLIT x the threads, 1/SPLIT the chain length.

Shapes are Kev-9B's (Qwen3.5-9B): 32 value heads, 16 key heads, 128 x 128;
sequences of 48 / 218 / 42 tokens are the state and the two question rows of
one local decision (projects/g1 trap work), 308 = the whole request.

    pixi run -e jetson build-jetson noeira_max/gated_delta/bench_gated_delta.mojo -o build/bench_gated_delta
"""

from std.bit import log2_floor
from std.math import rsqrt, sqrt
from std.memory import unsafe_stack_allocation
from std.random import rand, seed
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu import barrier, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import warp
from layout import TensorLayout, TileTensor, row_major

from state_space.gated_delta import gated_delta_recurrence_fwd_gpu


comptime NV = 32  # value heads
comptime NK = 16  # key heads
comptime KD = 128  # key head dim
comptime VD = 128  # value head dim
comptime KEY_DIM = NK * KD
comptime VALUE_DIM = NV * VD
comptime CONV_DIM = 2 * KEY_DIM + VALUE_DIM
comptime STATE = NV * KD * VD
comptime ITERS = 50


# ===----------------------------------------------------------------------=== #
# noeira kernel
# ===----------------------------------------------------------------------=== #


def gd_split_kernel[
    SPLIT: Int,
    out_LT: TensorLayout,
    state_LT: TensorLayout,
    qkv_LT: TensorLayout,
    decay_LT: TensorLayout,
    beta_LT: TensorLayout,
](
    seq_len: Int32,
    y: TileTensor[DType.float32, out_LT, MutUntrackedOrigin],
    state: TileTensor[DType.float32, state_LT, MutUntrackedOrigin],
    qkv: TileTensor[DType.float32, qkv_LT, MutUntrackedOrigin],
    decay: TileTensor[DType.float32, decay_LT, MutUntrackedOrigin],
    beta: TileTensor[DType.float32, beta_LT, MutUntrackedOrigin],
):
    """One CTA per value head, VD * SPLIT threads. Lane `part` of column `col`
    owns state rows [part * SL, (part + 1) * SL) of that column.

    Same arithmetic as MAX's kernel (raw q/k, L2 norms with eps 1e-6, the
    1/sqrt(KD) query scale folded into the readout), reassociated: each sum
    is SPLIT partial sums added by a butterfly — fp32 rounding differs, the
    algebra does not.
    """
    comptime SL = KD // SPLIT
    comptime NT = VD * SPLIT
    comptime assert KD % SPLIT == 0 and SPLIT <= 32, "SPLIT must divide KD and fit a warp"

    var tid = Int(thread_idx.x)
    var vh = Int(block_idx.x)
    var col = tid // SPLIT
    var part = tid % SPLIT
    var kh = vh // (NV // NK)
    var k0 = part * SL

    var q_s = unsafe_stack_allocation[KD, Float32, address_space=.SHARED]()
    var k_s = unsafe_stack_allocation[KD, Float32, address_space=.SHARED]()

    var s = SIMD[DType.float32, SL](0.0)
    comptime for j in range(SL):
        s[j] = state._storage[vh * KD * VD + (k0 + j) * VD + col]

    var q_base = kh * KD
    var k_base = KEY_DIM + kh * KD
    var v_ch = 2 * KEY_DIM + vh * VD + col
    var scale = Float32(1.0) / sqrt(Float32(KD))

    for t in range(Int(seq_len)):
        var row = t * CONV_DIM
        # cooperative load of this token's raw q and k (2 * KD floats)
        var i = tid
        while i < 2 * KD:
            if i < KD:
                q_s[i] = qkv._storage[row + q_base + i]
            else:
                k_s[i - KD] = qkv._storage[row + k_base + i - KD]
            i += NT
        barrier()

        var dec = decay._storage[t * NV + vh]
        var bet = beta._storage[t * NV + vh]
        var v = qkv._storage[row + v_ch]

        var qq = Float32(0.0)
        var kk = Float32(0.0)
        var kv = Float32(0.0)
        comptime for j in range(SL):
            var qj = q_s[k0 + j]
            var kj = k_s[k0 + j]
            qq += qj * qj
            kk += kj * kj
            s[j] = s[j] * dec
            kv += s[j] * kj
        comptime for o in range(log2_floor(SPLIT)):
            qq += warp.shuffle_xor(qq, UInt32(1 << o))
            kk += warp.shuffle_xor(kk, UInt32(1 << o))
            kv += warp.shuffle_xor(kv, UInt32(1 << o))
        var k_inv = rsqrt(kk + Float32(1e-6))
        var q_fac = rsqrt(qq + Float32(1e-6)) * scale
        var upd = bet * (v - kv * k_inv) * k_inv

        var o_acc = Float32(0.0)
        comptime for j in range(SL):
            s[j] = s[j] + k_s[k0 + j] * upd
            o_acc += s[j] * q_s[k0 + j]
        comptime for o in range(log2_floor(SPLIT)):
            o_acc += warp.shuffle_xor(o_acc, UInt32(1 << o))
        if part == 0:
            y._storage[t * VALUE_DIM + vh * VD + col] = o_acc * q_fac
        barrier()  # all reads of q_s/k_s done before the next token's load

    comptime for j in range(SL):
        state._storage[vh * KD * VD + (k0 + j) * VD + col] = s[j]


def gd_pipe_kernel[
    out_LT: TensorLayout,
    state_LT: TensorLayout,
    qkv_LT: TensorLayout,
    decay_LT: TensorLayout,
    beta_LT: TensorLayout,
](
    seq_len: Int32,
    y: TileTensor[DType.float32, out_LT, MutUntrackedOrigin],
    state: TileTensor[DType.float32, state_LT, MutUntrackedOrigin],
    qkv: TileTensor[DType.float32, qkv_LT, MutUntrackedOrigin],
    decay: TileTensor[DType.float32, decay_LT, MutUntrackedOrigin],
    beta: TileTensor[DType.float32, beta_LT, MutUntrackedOrigin],
):
    """v2, one thread per value column (VD threads), latency-oriented:

    1. Token t+1's inputs (this thread's q and k element, its v, the head's
       decay and beta) are LOADED into registers before token t's math, so
       the global latency overlaps compute; q/k go through a double-buffered
       SMEM tile, which needs ONE barrier per token instead of two (a thread
       writing buffer t&1 has passed barrier t-1, which every thread reached
       only after its reads of the same buffer at t-2).
    2. The L2 norms ride in the dot-product loops (k's in the kv loop, q's in
       the readout loop) instead of two extra 128-element passes per thread.
    3. Two accumulators per dot product halve the dependent FMA chain.
    """
    var tid = Int(thread_idx.x)
    var vh = Int(block_idx.x)
    var kh = vh // (NV // NK)
    var T = Int(seq_len)

    var q_s = unsafe_stack_allocation[2 * KD, Float32, address_space=.SHARED]()
    var k_s = unsafe_stack_allocation[2 * KD, Float32, address_space=.SHARED]()

    var s = SIMD[DType.float32, KD](0.0)
    comptime for j in range(KD):
        s[j] = state._storage[vh * KD * VD + j * VD + tid]

    var q_base = kh * KD + tid
    var k_base = KEY_DIM + kh * KD + tid
    var v_ch = 2 * KEY_DIM + vh * VD + tid
    var scale = Float32(1.0) / sqrt(Float32(KD))

    # prologue: token 0 in registers
    var nq = Float32(0.0)
    var nk = Float32(0.0)
    var nv = Float32(0.0)
    var nd = Float32(0.0)
    var nb = Float32(0.0)
    if T > 0:
        nq = qkv._storage[q_base]
        nk = qkv._storage[k_base]
        nv = qkv._storage[v_ch]
        nd = decay._storage[vh]
        nb = beta._storage[vh]

    for t in range(T):
        var buf = (t & 1) * KD
        q_s[buf + tid] = nq
        k_s[buf + tid] = nk
        var v = nv
        var dec = nd
        var bet = nb
        if t + 1 < T:  # issue token t+1's loads now; consumed next iteration
            var row = (t + 1) * CONV_DIM
            nq = qkv._storage[row + q_base]
            nk = qkv._storage[row + k_base]
            nv = qkv._storage[row + v_ch]
            nd = decay._storage[(t + 1) * NV + vh]
            nb = beta._storage[(t + 1) * NV + vh]
        barrier()

        var kv0 = Float32(0.0)
        var kv1 = Float32(0.0)
        var kk0 = Float32(0.0)
        var kk1 = Float32(0.0)
        comptime for j in range(0, KD, 2):
            var ka = k_s[buf + j]
            var kb = k_s[buf + j + 1]
            s[j] = s[j] * dec
            s[j + 1] = s[j + 1] * dec
            kv0 += s[j] * ka
            kv1 += s[j + 1] * kb
            kk0 += ka * ka
            kk1 += kb * kb
        var k_inv = rsqrt(kk0 + kk1 + Float32(1e-6))
        var upd = bet * (v - (kv0 + kv1) * k_inv) * k_inv

        var o0 = Float32(0.0)
        var o1 = Float32(0.0)
        var qq0 = Float32(0.0)
        var qq1 = Float32(0.0)
        comptime for j in range(0, KD, 2):
            var qa = q_s[buf + j]
            var qb = q_s[buf + j + 1]
            s[j] = s[j] + k_s[buf + j] * upd
            s[j + 1] = s[j + 1] + k_s[buf + j + 1] * upd
            o0 += s[j] * qa
            o1 += s[j + 1] * qb
            qq0 += qa * qa
            qq1 += qb * qb
        var q_fac = rsqrt(qq0 + qq1 + Float32(1e-6)) * scale
        y._storage[t * VALUE_DIM + vh * VD + tid] = (o0 + o1) * q_fac

    comptime for j in range(KD):
        state._storage[vh * KD * VD + j * VD + tid] = s[j]


def gd_vec_kernel[
    W: Int,
    out_LT: TensorLayout,
    state_LT: TensorLayout,
    qkv_LT: TensorLayout,
    decay_LT: TensorLayout,
    beta_LT: TensorLayout,
](
    seq_len: Int32,
    y: TileTensor[DType.float32, out_LT, MutUntrackedOrigin],
    state: TileTensor[DType.float32, state_LT, MutUntrackedOrigin],
    qkv: TileTensor[DType.float32, qkv_LT, MutUntrackedOrigin],
    decay: TileTensor[DType.float32, decay_LT, MutUntrackedOrigin],
    beta: TileTensor[DType.float32, beta_LT, MutUntrackedOrigin],
):
    """v3 = v2 with W-wide SMEM loads and W accumulators per dot product.

    v2 was SMEM-INSTRUCTION bound, not latency bound: 32 head-CTAs share the
    Orin's 8 SMs (4 per SM), and every FMA read one float of q or k from SMEM
    (~512 ld.shared per thread per token). One W-wide load feeds W FMAs.
    """
    comptime assert KD % W == 0, "W must divide KD"
    var tid = Int(thread_idx.x)
    var vh = Int(block_idx.x)
    var kh = vh // (NV // NK)
    var T = Int(seq_len)

    var q_s = unsafe_stack_allocation[
        2 * KD, Float32, address_space=.SHARED, alignment=16
    ]()
    var k_s = unsafe_stack_allocation[
        2 * KD, Float32, address_space=.SHARED, alignment=16
    ]()

    var s = SIMD[DType.float32, KD](0.0)
    comptime for j in range(KD):
        s[j] = state._storage[vh * KD * VD + j * VD + tid]

    var q_base = kh * KD + tid
    var k_base = KEY_DIM + kh * KD + tid
    var v_ch = 2 * KEY_DIM + vh * VD + tid
    var scale = Float32(1.0) / sqrt(Float32(KD))

    var nq = Float32(0.0)
    var nk = Float32(0.0)
    var nv = Float32(0.0)
    var nd = Float32(0.0)
    var nb = Float32(0.0)
    if T > 0:
        nq = qkv._storage[q_base]
        nk = qkv._storage[k_base]
        nv = qkv._storage[v_ch]
        nd = decay._storage[vh]
        nb = beta._storage[vh]

    for t in range(T):
        var buf = (t & 1) * KD
        q_s[buf + tid] = nq
        k_s[buf + tid] = nk
        var v = nv
        var dec = nd
        var bet = nb
        if t + 1 < T:
            var row = (t + 1) * CONV_DIM
            nq = qkv._storage[row + q_base]
            nk = qkv._storage[row + k_base]
            nv = qkv._storage[row + v_ch]
            nd = decay._storage[(t + 1) * NV + vh]
            nb = beta._storage[(t + 1) * NV + vh]
        barrier()

        var kv = SIMD[DType.float32, W](0.0)
        var kk = SIMD[DType.float32, W](0.0)
        comptime for j in range(0, KD, W):
            var k4 = k_s.load[width=W](buf + j)
            comptime for i in range(W):
                s[j + i] = s[j + i] * dec
                kv[i] += s[j + i] * k4[i]
            kk += k4 * k4
        var k_inv = rsqrt(kk.reduce_add() + Float32(1e-6))
        var upd = bet * (v - kv.reduce_add() * k_inv) * k_inv

        var o = SIMD[DType.float32, W](0.0)
        var qq = SIMD[DType.float32, W](0.0)
        comptime for j in range(0, KD, W):
            var q4 = q_s.load[width=W](buf + j)
            var k4 = k_s.load[width=W](buf + j)
            comptime for i in range(W):
                s[j + i] = s[j + i] + k4[i] * upd
                o[i] += s[j + i] * q4[i]
            qq += q4 * q4
        var q_fac = rsqrt(qq.reduce_add() + Float32(1e-6)) * scale
        y._storage[t * VALUE_DIM + vh * VD + tid] = o.reduce_add() * q_fac

    comptime for j in range(KD):
        state._storage[vh * KD * VD + j * VD + tid] = s[j]


# ===----------------------------------------------------------------------=== #
# float64 CPU reference (the five-step rule, state [vh, kd, vd])
# ===----------------------------------------------------------------------=== #


def reference(
    T: Int,
    qkv: UnsafePointer[Float32, _],
    decay: UnsafePointer[Float32, _],
    beta: UnsafePointer[Float32, _],
) -> List[Float64]:
    var out = List[Float64](length=T * VALUE_DIM, fill=0.0)
    var st = List[Float64](length=KD, fill=0.0)
    var scale = 1.0 / sqrt(Float64(KD))
    for vh in range(NV):
        var kh = vh // (NV // NK)
        for col in range(VD):
            for k in range(KD):
                st[k] = 0.0
            for t in range(T):
                var row = t * CONV_DIM
                var qq = 0.0
                var kk = 0.0
                for k in range(KD):
                    var qv = Float64(qkv[row + kh * KD + k])
                    var kv_ = Float64(qkv[row + KEY_DIM + kh * KD + k])
                    qq += qv * qv
                    kk += kv_ * kv_
                var q_inv = 1.0 / sqrt(qq + 1e-6) * scale
                var k_inv = 1.0 / sqrt(kk + 1e-6)
                var dec = Float64(decay[t * NV + vh])
                var bet = Float64(beta[t * NV + vh])
                var mem = 0.0
                for k in range(KD):
                    st[k] *= dec
                    mem += st[k] * Float64(qkv[row + KEY_DIM + kh * KD + k]) * k_inv
                var v = Float64(qkv[row + 2 * KEY_DIM + vh * VD + col])
                var delta = bet * (v - mem)
                var o = 0.0
                for k in range(KD):
                    st[k] += Float64(qkv[row + KEY_DIM + kh * KD + k]) * k_inv * delta
                    o += st[k] * Float64(qkv[row + kh * KD + k]) * q_inv
                out[t * VALUE_DIM + vh * VD + col] = o
    return out^


def _f(x: Float64) -> Float64:
    return Float64(Int(x * 100 + 0.5)) / 100


def max_err(got: UnsafePointer[Float32, _], want: List[Float64]) -> Float64:
    var m = 0.0
    var scale = 0.0
    for i in range(len(want)):
        m = max(m, abs(Float64(got[i]) - want[i]))
        scale = max(scale, abs(want[i]))
    return m / scale


# ===----------------------------------------------------------------------=== #
# one sequence length
# ===----------------------------------------------------------------------=== #


def run_case(ctx: DeviceContext, T: Int, iters: Int = ITERS, warm: Int = 3) raises:
    # host inputs: raw q/k/v ~ U(0,1) - 0.5, decay and beta in (0, 1)
    var qkv_h = ctx.enqueue_create_host_buffer[DType.float32](T * CONV_DIM)
    var dec_h = ctx.enqueue_create_host_buffer[DType.float32](T * NV)
    var bet_h = ctx.enqueue_create_host_buffer[DType.float32](T * NV)
    ctx.synchronize()
    rand[DType.float32](qkv_h.unsafe_ptr(), T * CONV_DIM)
    rand[DType.float32](dec_h.unsafe_ptr(), T * NV)
    rand[DType.float32](bet_h.unsafe_ptr(), T * NV)
    for i in range(T * CONV_DIM):
        qkv_h.unsafe_ptr()[i] -= 0.5
    for i in range(T * NV):
        dec_h.unsafe_ptr()[i] = 0.80 + 0.19 * dec_h.unsafe_ptr()[i]  # gentle decay, like a trained gate

    var want = reference(T, qkv_h.unsafe_ptr(), dec_h.unsafe_ptr(), bet_h.unsafe_ptr())

    var qkv_d = ctx.enqueue_create_buffer[DType.float32](T * CONV_DIM)
    var dec_d = ctx.enqueue_create_buffer[DType.float32](T * NV)
    var bet_d = ctx.enqueue_create_buffer[DType.float32](T * NV)
    var out_d = ctx.enqueue_create_buffer[DType.float32](T * VALUE_DIM)
    var st_d = ctx.enqueue_create_buffer[DType.float32](STATE)
    var slot_d = ctx.enqueue_create_buffer[DType.uint32](1)
    var off_d = ctx.enqueue_create_buffer[DType.uint32](2)
    var out_h = ctx.enqueue_create_host_buffer[DType.float32](T * VALUE_DIM)
    var off_h = ctx.enqueue_create_host_buffer[DType.uint32](2)
    ctx.synchronize()
    off_h.unsafe_ptr()[0] = 0
    off_h.unsafe_ptr()[1] = UInt32(T)
    ctx.enqueue_copy(qkv_d, qkv_h)
    ctx.enqueue_copy(dec_d, dec_h)
    ctx.enqueue_copy(bet_d, bet_h)
    ctx.enqueue_copy(off_d, off_h)
    ctx.enqueue_memset(slot_d, UInt32(0))

    var qkv_tt = TileTensor(qkv_d, row_major(T, CONV_DIM))
    var dec_tt = TileTensor(dec_d, row_major(T, NV))
    var bet_tt = TileTensor(bet_d, row_major(T, NV))
    var out_tt = TileTensor(out_d, row_major(T, VALUE_DIM))
    var pool_tt = TileTensor(st_d, row_major(1, NV, KD, VD))
    var st_tt = TileTensor(st_d, row_major(NV, KD, VD))
    var slot_tt = TileTensor(slot_d, row_major(1))
    var off_tt = TileTensor(off_d, row_major(2))

    # ── MAX ─────────────────────────────────────────────────────────────────
    var max_k = ctx.compile_function[
        gated_delta_recurrence_fwd_gpu[
            DType.float32, DType.float32, KD, VD,
            out_tt.LayoutType, qkv_tt.LayoutType, dec_tt.LayoutType,
            bet_tt.LayoutType, pool_tt.LayoutType, slot_tt.LayoutType,
            off_tt.LayoutType,
        ]
    ]()

    @always_inline
    def launch_max() raises capturing:
        ctx.enqueue_function(
            max_k,
            Int32(1), Int32(NV), Int32(NK), Int32(KEY_DIM),
            out_tt, pool_tt, slot_tt, qkv_tt, dec_tt, bet_tt, off_tt,
            UInt32(CONV_DIM), UInt32(1), UInt32(NV), UInt32(1),
            UInt32(STATE), UInt32(KD * VD), UInt32(VD), UInt32(1),
            UInt32(VALUE_DIM), UInt32(1),
            grid_dim=(NV,), block_dim=(VD,),
        )

    ctx.enqueue_memset(st_d, Float32(0))
    launch_max()
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()
    var err_max = max_err(out_h.unsafe_ptr(), want)
    for _ in range(warm):
        launch_max()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(iters):
        launch_max()
    ctx.synchronize()
    var us_max = Float64(perf_counter_ns() - t0) / 1e3 / Float64(max(iters, 1))
    print("T=", T, " MAX        ", _f(us_max), "us/call ", _f(us_max / Float64(T)), "us/token  x24 layers", _f(us_max * 24 / 1e3), "ms  rel err", err_max)

    # ── noeira, SPLIT = 1, 2, 4, 8 ──────────────────────────────────────────
    comptime for si in range(4):
        comptime SPLIT = 1 << si
        comptime kern = gd_split_kernel[
            SPLIT, out_tt.LayoutType, st_tt.LayoutType, qkv_tt.LayoutType,
            dec_tt.LayoutType, bet_tt.LayoutType,
        ]
        var nk = ctx.compile_function[kern]()

        @always_inline
        def launch_n() raises capturing:
            ctx.enqueue_function(
                nk, Int32(T), out_tt, st_tt, qkv_tt, dec_tt, bet_tt,
                grid_dim=(NV,), block_dim=(VD * SPLIT,),
            )

        ctx.enqueue_memset(st_d, Float32(0))
        launch_n()
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        var err = max_err(out_h.unsafe_ptr(), want)
        for _ in range(warm):
            launch_n()
        ctx.synchronize()
        var t1 = perf_counter_ns()
        for _ in range(iters):
            launch_n()
        ctx.synchronize()
        var us = Float64(perf_counter_ns() - t1) / 1e3 / Float64(max(iters, 1))
        print("T=", T, " noeira S=", SPLIT, _f(us), "us/call ", _f(us / Float64(T)), "us/token  x24 layers", _f(us * 24 / 1e3), "ms  rel err", err, " ", _f(us_max / us), "x MAX")


    # ── noeira v2: prefetch + double buffer + fused norms ───────────────────
    comptime pk = gd_pipe_kernel[
        out_tt.LayoutType, st_tt.LayoutType, qkv_tt.LayoutType,
        dec_tt.LayoutType, bet_tt.LayoutType,
    ]
    var pf = ctx.compile_function[pk]()

    @always_inline
    def launch_p() raises capturing:
        ctx.enqueue_function(
            pf, Int32(T), out_tt, st_tt, qkv_tt, dec_tt, bet_tt,
            grid_dim=(NV,), block_dim=(VD,),
        )

    ctx.enqueue_memset(st_d, Float32(0))
    launch_p()
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()
    var err_p = max_err(out_h.unsafe_ptr(), want)
    for _ in range(warm):
        launch_p()
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(iters):
        launch_p()
    ctx.synchronize()
    var us_p = Float64(perf_counter_ns() - t2) / 1e3 / Float64(max(iters, 1))
    print("T=", T, " noeira v2  ", _f(us_p), "us/call ", _f(us_p / Float64(T)), "us/token  x24 layers", _f(us_p * 24 / 1e3), "ms  rel err", err_p, " ", _f(us_max / us_p), "x MAX")

    # ── noeira v3: v2 + W-wide SMEM loads, W = 2, 4 ─────────────────────────
    comptime for wi in range(1, 3):
        comptime W = 1 << wi
        comptime vk = gd_vec_kernel[
            W, out_tt.LayoutType, st_tt.LayoutType, qkv_tt.LayoutType,
            dec_tt.LayoutType, bet_tt.LayoutType,
        ]
        var vf = ctx.compile_function[vk]()

        @always_inline
        def launch_v() raises capturing:
            ctx.enqueue_function(
                vf, Int32(T), out_tt, st_tt, qkv_tt, dec_tt, bet_tt,
                grid_dim=(NV,), block_dim=(VD,),
            )

        ctx.enqueue_memset(st_d, Float32(0))
        launch_v()
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        var err_v = max_err(out_h.unsafe_ptr(), want)
        for _ in range(warm):
            launch_v()
        ctx.synchronize()
        var t3 = perf_counter_ns()
        for _ in range(iters):
            launch_v()
        ctx.synchronize()
        var us_v = Float64(perf_counter_ns() - t3) / 1e3 / Float64(max(iters, 1))
        print("T=", T, " noeira v3 W=", W, _f(us_v), "us/call ", _f(us_v / Float64(T)), "us/token  x24 layers", _f(us_v * 24 / 1e3), "ms  rel err", err_v, " ", _f(us_max / us_v), "x MAX")


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    print("Gated DeltaNet recurrence, Qwen3.5-9B shapes (NV 32, NK 16, 128x128), fp32 state;", ctx.name())
    # `--ncu`: T = 218 only, no warm-up, one timed launch per kernel (for Nsight Compute)
    var ncu = False
    for a in argv():
        if a == "--ncu":
            ncu = True
    if ncu:
        run_case(ctx, 218, 1, 0)
        return
    for T in [48, 218, 42, 308]:
        run_case(ctx, T)
