"""PROTOTYPE — multi-tensor ("grouped") Adam update kernel.

Compile-viability + correctness probe for the real fix to nn's per-update
kernel count. nn stores each parameter tensor as its OWN `DeviceBuffer`
(`Param.make_gpu` → one `enqueue_create_buffer` per tensor), so the params are
NOT a contiguous slab — a single slab-wide Adam kernel can't address them. The
standard remedy is a *fused multi-tensor apply*: hand ONE kernel device-resident
arrays of the per-param value/grad pointer ADDRESSES (+ sizes + moment offsets)
and let a 2-D grid (`grid.y = param index`) update every param in one launch.

The open question this probes: can a Mojo GPU kernel load a pointer address from
a device buffer (`UInt64`) and dereference it (`Pointer(unsafe_from_address=…)`)?
- On CUDA this is just generic-address pointer arithmetic and should work.
- On Apple Metal it may not (Metal dislikes raw pointer arrays in buffers).

If this builds + passes on Apple → the pattern is portable (no platform gate
needed). If it fails to compile/run on Apple but works on NVIDIA → the real
implementation gates the grouped kernel under
`comptime if default_accelerator().is_nvidia_gpu()` and Apple keeps the per-tensor path
(bit-identical math; capture is NVIDIA-only anyway).

This file wires NOTHING into Adam — it's a throwaway probe. Run:
    pixi run -e apple  mojo run -I . tests/nn/test_grouped_adam_prototype.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_grouped_adam_prototype.mojo
"""

from std.math import sqrt
from std.sys import default_accelerator
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT, TPB


# ──────────────────────────────────────────────────────────────────────
# The grouped kernel: one launch updates ALL params.
#   grid_dim = (max_blocks_x, n_params)   block_dim = (TPB,)
#   block (x, p): element e = x*TPB + tid of param p.
# Param/grad live in SEPARATE buffers (addresses passed via device arrays);
# the moment slab is contiguous (one m / one v buffer) addressed by offset[p].
# ──────────────────────────────────────────────────────────────────────
def _grouped_adam_kernel(
    param_addrs: Pointer[UInt64, MutAnyOrigin],
    grad_addrs: Pointer[UInt64, MutAnyOrigin],
    sizes: Pointer[Int32, MutAnyOrigin],
    moment_offs: Pointer[Int32, MutAnyOrigin],
    m_base: Pointer[Scalar[DT], MutAnyOrigin],
    v_base: Pointer[Scalar[DT], MutAnyOrigin],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
):
    var p = Int(block_idx.y)
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if e < Int(sizes[p]):
        # Reconstruct the per-param pointers from their addresses.
        var param = Pointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=Int(param_addrs[p])
        )
        var grad = Pointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=Int(grad_addrs[p])
        )
        var mo = Int(moment_offs[p]) + e
        var one: Scalar[DT] = 1.0
        var g = grad[e]
        var m_new = beta1 * m_base[mo] + (one - beta1) * g
        var v_new = beta2 * v_base[mo] + (one - beta2) * g * g
        m_base[mo] = m_new
        v_base[mo] = v_new
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        param[e] = param[e] - lr * m_hat / (sqrt(v_hat) + eps)


def _u64_dev(
    ctx: DeviceContext, ref h: List[UInt64]
) raises -> DeviceBuffer[DType.uint64]:
    var b = ctx.enqueue_create_buffer[DType.uint64](len(h))
    with b.map_to_host() as hm:
        for i in range(len(h)):
            hm[i] = h[i]
    return b^


def _i32_dev(
    ctx: DeviceContext, ref h: List[Int32]
) raises -> DeviceBuffer[DType.int32]:
    var b = ctx.enqueue_create_buffer[DType.int32](len(h))
    with b.map_to_host() as hm:
        for i in range(len(h)):
            hm[i] = h[i]
    return b^


def main() raises:
    print("=== grouped (multi-tensor) Adam prototype ===")

    # 3 params of different sizes — mimics weight/bias tensors in SEPARATE
    # DeviceBuffers (exactly how nn's Param.make_gpu allocates).
    var sizes_host: List[Int] = [5, 3, 7]
    var n_params = len(sizes_host)
    var total = 0
    for i in range(n_params):
        total += sizes_host[i]

    comptime LR = Scalar[DT](0.01)
    comptime B1 = Scalar[DT](0.9)
    comptime B2 = Scalar[DT](0.999)
    comptime EPS = Scalar[DT](1e-8)
    # Step-1 bias correction (m, v start at 0).
    var bc1 = Scalar[DT](1.0) - B1
    var bc2 = Scalar[DT](1.0) - B2

    with DeviceContext() as ctx:
        # Per-param value + grad buffers (separate allocations).
        var param_bufs = List[DeviceBuffer[DT]]()
        var grad_bufs = List[DeviceBuffer[DT]]()
        var param_addr_host = List[UInt64]()
        var grad_addr_host = List[UInt64]()
        var moment_off_host = List[Int32]()
        var sizes_i32_host = List[Int32]()

        # Host shadow copies of initial params/grads for the reference Adam.
        var init_params = List[Scalar[DT]]()  # flattened in walk order
        var grads_flat = List[Scalar[DT]]()
        var roff = 0
        for p in range(n_params):
            var n = sizes_host[p]
            var pb = ctx.enqueue_create_buffer[DT](n)
            var gb = ctx.enqueue_create_buffer[DT](n)
            # Deterministic init: param[p][j] = 0.1*(p+1)+0.01*j ; grad = (-1)^j*(p+1)
            with pb.map_to_host() as ph:
                with gb.map_to_host() as gh:
                    for j in range(n):
                        var pv = Scalar[DT](
                            0.1 * Float64(p + 1) + 0.01 * Float64(j)
                        )
                        var gv = Scalar[DT](Float64(p + 1))
                        if j % 2 == 1:
                            gv = -gv
                        ph[j] = pv
                        gh[j] = gv
                        init_params.append(pv)
                        grads_flat.append(gv)
            param_addr_host.append(UInt64(Int(pb.unsafe_ptr())))
            grad_addr_host.append(UInt64(Int(gb.unsafe_ptr())))
            moment_off_host.append(Int32(roff))
            sizes_i32_host.append(Int32(n))
            roff += n
            param_bufs.append(pb^)
            grad_bufs.append(gb^)

        # Contiguous moment slab (m, v) — zero-initialized, like Adam's m_dev/v_dev.
        var m_buf = ctx.enqueue_create_buffer[DT](total)
        var v_buf = ctx.enqueue_create_buffer[DT](total)
        m_buf.enqueue_fill(Scalar[DT](0.0))
        v_buf.enqueue_fill(Scalar[DT](0.0))

        # Upload the descriptor arrays.
        var param_addr_dev = _u64_dev(ctx, param_addr_host)
        var grad_addr_dev = _u64_dev(ctx, grad_addr_host)
        var sizes_dev = _i32_dev(ctx, sizes_i32_host)
        var moff_dev = _i32_dev(ctx, moment_off_host)

        # Max blocks across params (grid.x); grid.y = n_params.
        var max_n = 0
        for i in range(n_params):
            if sizes_host[i] > max_n:
                max_n = sizes_host[i]
        var blocks_x = (max_n + TPB - 1) // TPB

        ctx.enqueue_function[_grouped_adam_kernel](
            rebind[Pointer[UInt64, MutAnyOrigin]](
                param_addr_dev.unsafe_ptr()
            ),
            rebind[Pointer[UInt64, MutAnyOrigin]](
                grad_addr_dev.unsafe_ptr()
            ),
            rebind[Pointer[Int32, MutAnyOrigin]](sizes_dev.unsafe_ptr()),
            rebind[Pointer[Int32, MutAnyOrigin]](moff_dev.unsafe_ptr()),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](m_buf.unsafe_ptr()),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](v_buf.unsafe_ptr()),
            bc1, bc2, LR, B1, B2, EPS,
            grid_dim=(blocks_x, n_params),
            block_dim=(TPB,),
        )
        ctx.synchronize()

        # ── Host reference Adam (one step, m=v=0) for every element ──
        var max_err = Float64(0.0)
        var flat = 0
        for p in range(n_params):
            var n = sizes_host[p]
            with param_bufs[p].map_to_host() as ph:
                for j in range(n):
                    var g = Float64(grads_flat[flat])
                    var m_new = (1.0 - Float64(B1)) * g
                    var v_new = (1.0 - Float64(B2)) * g * g
                    var m_hat = m_new / Float64(bc1)
                    var v_hat = v_new / Float64(bc2)
                    var expected = Float64(init_params[flat]) - Float64(
                        LR
                    ) * m_hat / (sqrt(v_hat) + Float64(EPS))
                    var got = Float64(ph[j])
                    var err = abs(got - expected)
                    if err > max_err:
                        max_err = err
                    flat += 1

        print("  n_params =", n_params, " total_elems =", total)
        print("  grid =", "(", blocks_x, ",", n_params, ")  block =", TPB)
        print("  max abs err vs host Adam =", max_err)
        var ok = max_err < 1e-5
        comptime if default_accelerator().is_nvidia_gpu():
            if ok:
                print(
                    "  PASS (NVIDIA) — grouped multi-tensor Adam works:"
                    " reconstructing per-param pointers from device addresses"
                    " and dereferencing them in-kernel is correct on CUDA."
                    " => the real fix can collapse the per-tensor Adam/zero/"
                    "polyak launches into ONE grouped launch per optimizer."
                )
            else:
                print(
                    "  FAIL (NVIDIA) — grouped kernel produced wrong results."
                    " The pointer-array approach is NOT viable even on CUDA;"
                    " do NOT pursue multi-tensor apply this way."
                )
        else:
            # On Apple Metal a host-captured device address is not valid to
            # dereference inside a kernel, so the per-param writes are lost
            # (only the directly-bound moment slab updates). Expected FAIL.
            if ok:
                print(
                    "  UNEXPECTED PASS (Apple) — Metal dereffed the raw"
                    " addresses; pattern may be portable after all."
                )
            else:
                print(
                    "  EXPECTED FAIL (Apple/Metal) — raw device-address deref"
                    " doesn't land (err≈LR ⇒ params untouched). This is the"
                    " documented Metal limitation. The real fix therefore gates"
                    " the grouped kernel under default_accelerator().is_nvidia_gpu() and"
                    " Apple keeps the per-tensor path (bit-identical math;"
                    " CUDA-graph capture is NVIDIA-only anyway)."
                    " ACTION: run this on an NVIDIA box — it must print PASS."
                )
        print("=== done ===")
