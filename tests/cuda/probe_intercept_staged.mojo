"""Staged interceptor probe — which call in `CUDAGraph.__init__` dies?

`test_cuda_graph_minimal` crashes on NVIDIA, so the graph harness is broken
independently of SAC / replay / physics. This walks `CUDAGraph.__init__`'s
interceptor calls ONE AT A TIME, printing before and after each, so a single
run names the failing call instead of leaving us with an unwind whose frames
#3 and #6 are the same libc trampoline.

It does NOT use `CUDAGraph` — it loads the .so directly, so nothing is hidden
behind the wrapper.

⚠ RUN ON A TTY, NOT INTO A FILE. Mojo's `print` is block-buffered when stdout
is a pipe, and a SIGSEGV never flushes it — you would lose exactly the lines
that matter. If you must redirect, expect the tail to be missing.

⚠ THE ARITY HYPOTHESIS WAS TESTED AND REFUTED (NVIDIA, 2026-08-09).
Declaring a 1-argument signature for the zero-argument C function produced
the IDENTICAL crash at the IDENTICAL address, so `_DLCallable.__call__` is
not secretly invoking the function. The stage below measures the resolved
pointer rather than reasoning about it.

⚠ AND THE ADDRESS MATCH MAY BE A COINCIDENCE. Frames #3 and #6 of that stack
are the same libc trampoline, so the unwind is unreliable past #5; `#7` could
be a stack word that merely HOLDS the stream rather than the faulting PC. The
`fn-value bits` print below is what settles it — it needs no unwinder.

SUPERSEDED HYPOTHESIS, kept because it shaped the probe: `graph.mojo` unwraps every symbol with a trailing
`()`, per its own note that `_DLCallable.__call__` is variadic and discards
arguments. That works for symbols WITH arguments, where unwrap and call are
distinguishable. `intercept_get_mojo_stream` is the only ZERO-ARGUMENT one:

    var get_stream = lib.get_function[_CUptr](name)
    self._mojo_stream = get_stream()

For a 0-arg signature both spellings are `()`, so if the first one CALLS
rather than unwraps, `get_stream` holds the STREAM VALUE typed as a function
pointer and the second `()` jumps to it. The reported `#7` equalled the
stream address the interceptor printed, which is what that would look like.

READ THE OUTPUT LIKE THIS:
  crash before "get_stream unwrapped"   -> dlopen/dlsym, not the ABI
  crash between unwrap and "stream ="   -> the 0-arg ambiguity above; the
                                           `_ARG` variant below is the fix
  "stream = 0x..." matches the value the
  interceptor printed                   -> unwrap is fine, look further down
  crash at stream_create / begin_capture-> the ambiguity is NOT the bug;
                                           suspect the interceptor vs the
                                           current Mojo runtime (stream
                                           discovery, cuGetProcAddress)
"""

from std.ffi import OwnedDLHandle, c_int
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.sys import default_accelerator
from std.memory import alloc
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT

comptime _CUptr = Pointer[NoneType, MutUntrackedOrigin]


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print("non-NVIDIA: the interceptor path does not exist here. Skipped.")
        return

    var ctx = DeviceContext()

    @always_inline
    def _bump(buf: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin]):
        if Int(thread_idx.x) != 0:
            return
        buf[0] = buf[0] + Scalar[DT](1.0)

    var buf = ctx.enqueue_create_buffer[DT](1)
    var lt = LayoutTensor[DT, Layout.row_major(1)](buf)

    # Stream discovery needs a prior launch (see CUDAGraph.__init__).
    print("[probe] warmup launch ...")
    ctx.enqueue_function[_bump](lt, grid_dim=1, block_dim=1)
    ctx.synchronize()
    print("[probe] warmup done")

    print("[probe] dlopen libcuda_intercept.so ...")
    var lib = OwnedDLHandle("./noeira/cuda/libcuda_intercept.so")
    print("[probe] dlopen ok")

    # ── 1. WHAT DID THE UNWRAP ACTUALLY PRODUCE? ─────────────────────────
    #
    # The arity hypothesis is DEAD: declaring a 1-arg signature for the 0-arg
    # C function changed nothing (same crash, same address), so
    # `_DLCallable.__call__` is not silently invoking the function.
    #
    # So measure the pointer instead of reasoning about it. A `thin` function
    # value is one word, so reading the variable's own bytes as an Int gives
    # the address the call will jump to. Three outcomes, all decisive:
    #
    #   bits == the stream address the interceptor printed
    #        -> the unwrap really is handing back the stream. Wrong VALUE.
    #   bits inside libcuda_intercept.so's mapped range (see /proc/self/maps)
    #        -> the pointer is CORRECT and the CALL is what faults: an ABI or
    #           calling-convention problem with `thin`, not a lookup problem.
    #   bits == 0
    #        -> dlsym never resolved the symbol; check the .so's exports.
    #
    # `begin_capture` is the control: a symbol resolved the same way that is
    # known to have worked. If its bits look like code and `get_stream`'s do
    # not, the difference is the symbol, not the machinery.
    print("[probe] resolving intercept_get_mojo_stream (0-arg) ...")
    var get_stream = lib.get_function[_CUptr]("intercept_get_mojo_stream")
    var gs_bits = Pointer(to=get_stream).bitcast[Int]()[]
    print("[probe] get_stream  fn-value bits =", gs_bits)

    print("[probe] resolving intercept_stream_begin_capture (1-arg, CONTROL)")
    var begin_cap = lib.get_function[c_int]("intercept_stream_begin_capture")
    var bc_bits = Pointer(to=begin_cap).bitcast[Int]()[]
    print("[probe] begin_cap   fn-value bits =", bc_bits)
    print(
        "[probe] the two differ by", bc_bits - gs_bits,
        "(a few KB apart => both are code in the same .so)",
    )

    # ── 2. only now, call it ──────────────────────────────────────────────
    print("[probe] calling get_stream() ...")
    var stream1 = get_stream()
    print("[probe] stream =", Int(stream1))

    if Int(stream1) == 0:
        print("[probe] STREAM NOT DISCOVERED — the interceptor never saw a")
        print("        launch. That is the dlsym-hook-vs-Mojo-runtime break,")
        print("        not an ABI problem.")
        return

    # ── 3. replay-stream creation ─────────────────────────────────────────
    print("[probe] intercept_stream_create ...")
    var stream_create = lib.get_function[c_int]("intercept_stream_create")
    var sbuf = alloc[_CUptr](1)
    var rc_create = stream_create(sbuf)
    print("[probe] stream_create rc =", Int(rc_create),
          " stream =", Int(sbuf[]))
    sbuf.free()

    # ── 4. begin / end capture around one launch ──────────────────────────
    print("[probe] intercept_stream_begin_capture ...")
    var rc_begin = begin_cap(stream1)
    print("[probe] begin_capture rc =", Int(rc_begin))
    if Int(rc_begin) != 0:
        print("[probe] begin_capture FAILED — stop here.")
        return

    ctx.enqueue_function[_bump](lt, grid_dim=1, block_dim=1)
    print("[probe] kernel enqueued under capture")

    var graph_buf = alloc[_CUptr](1)
    var end_cap = lib.get_function[c_int]("intercept_stream_end_capture")
    var rc_end = end_cap(stream1, graph_buf)
    print("[probe] end_capture rc =", Int(rc_end),
          " graph =", Int(graph_buf[]))
    graph_buf.free()
    print("[probe] SURVIVED the whole sequence.")
