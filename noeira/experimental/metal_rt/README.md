# metal_rt — a Metal kernel launcher in pure Mojo (experimental)

A small launcher that runs Mojo GPU kernels on Apple GPUs without MAX's
runtime: Mojo compiles the kernel to a metallib (`_compile_code`, the same
compiler `DeviceContext` uses), and this package loads and launches it through
the Objective-C runtime (`objc_msgSend`) with Apple's public Metal API. No C,
C++ or Objective-C source files.

It was built clean-room from Apple's public headers (Metal.framework and
objc in the macOS 27 SDK), the Mojo standard library and noeira's own code.
No MAX source was read. Statements below about MAX are measurements of its
public API (`DeviceContext.enqueue_function`), not knowledge of how it works.

The question: does a noeira-owned launcher make a realistic small training
step much faster than MAX's launch path, with bit-identical results?
Short answer: yes, 1.2x to 5x on an M1 Pro, bit for bit identical (660 of
660 runs). The gain is largest for small batches, where launch overhead is
most of the step.

## Files

| File | What |
|---|---|
| `objc.mojo` | ObjC bridge. `objc_msgSend` is looked up once and called through typed thin function pointers, one per arity. |
| `runtime.mojo` | `MetalRT`: device, shared buffers, a pipeline cache per kernel, `launch[kernel](args..., grid=, block=)`, commit policies, ordering, residency, `flush()` / `sync()`. Also the `Launcher` trait. |
| `metal4.mojo` | `Metal4RT`: the same API on Metal 4 (`MTL4CommandQueue`, `MTL4CommandAllocator`, `MTL4ArgumentTable`). |
| `test_abi.mojo` | Checks the argument ABI, and launch ordering under every policy, residency mode, ordering mode and both APIs. Exits 1 on failure. |
| `bench_launch.mojo` | Micro-benchmarks: msgSend caching, encode cost, batching and ordering, the synced round trip, Metal 4, live buffers. |
| `mlp_kernels.mojo` | Naive kernels for one 2-layer MLP training step (14 launches). |
| `bench_mlp.mojo` | The MLP step through MAX and through both launchers under each policy, plus the bit-identity gate. Exits 1 on failure. |
| `probe_own_launch.mojo` | The seed probe this started from. |

## Run

From the repo root (one Mojo compile at a time):

```bash
pixi run -e apple mojo run -I . noeira/experimental/metal_rt/test_abi.mojo
pixi run -e apple mojo run -I . noeira/experimental/metal_rt/bench_launch.mojo
pixi run -e apple mojo run -I . noeira/experimental/metal_rt/bench_mlp.mojo          # ~5 min
pixi run -e apple mojo run -I . noeira/experimental/metal_rt/bench_mlp.mojo --quick  # ~1 min
```

```mojo
from noeira.experimental.metal_rt.runtime import MetalRT, COMMIT_ADAPTIVE

var rt = MetalRT(policy=COMMIT_ADAPTIVE, batch_n=16, idle_age_us=20)
var x = rt.alloc[DType.float32](n)                # shared storage
# write x.host_ptr[DType.float32]() on the host ...
rt.launch[my_kernel](x.dev[DType.float32](), Int64(n), grid=(n + 255) // 256, block=256)
rt.sync()                                          # commit + wait; then read host_ptr
```

A kernel argument is bound by value, so a device pointer argument takes
`buf.dev[dt]()` (the buffer's GPU address), not `host_ptr`.

## How it works

- **Arguments.** Each kernel argument is its own Metal buffer index, in
  declaration order. Its bytes go in with `setBytes:length:atIndex:`. A
  pointer is passed as the buffer's 8-byte `gpuAddress`. Scalars are passed as
  their own bytes (`Int64` is 8 bytes, `Float32` is 4). `test_abi.mojo`
  checks this with a mixed (pointer, Int, Float32, pointer) kernel over 16
  threadgroups.
- **One serial encoder per command buffer.** Every launch between two commits
  goes into one `computeCommandEncoder`. Apple's header says a serial
  encoder runs its dispatches "in dispatched order", and that
  `memoryBarrierWithScope:` on a serial encoder "is allowed, but ignored". So
  dispatches inside one command buffer are already ordered.
- **Ordering between command buffers: one `MTLFence`.** Every encoder except
  the first waits on the fence at its start, and every encoder updates it at
  its end. This is needed (see pitfalls), and it costs nothing measurable.
- **Completion without blocks.** The host retains the last committed command
  buffer and reads its `status` to tell whether the GPU is idle. When
  `sync()` commits a command buffer, that buffer also signals a shared event,
  and `sync()` spins on `signaledValue`. The host sees the event about 80 us
  before `status` turns Completed.
- **Residency.** One `MTLResidencySet` holds every live buffer and is attached
  to the queue once (macOS 15+). Nothing is declared per command buffer or per
  launch. `RESIDENCY_PER_CB` (`useResources:count:usage:` once per command
  buffer) and `RESIDENCY_PER_LAUNCH` are kept for comparison.
- **Commit policies.**
  - `COMMIT_EACH` commits every launch.
  - `COMMIT_EVERY_N` commits once `batch_n` launches are pending.
  - `COMMIT_ADAPTIVE` commits at once if the GPU has nothing in flight, and
    otherwise batches up to `batch_n`. With `idle_age_us=T`, "at once"
    becomes "once the oldest pending launch is T us old".
  - `sync()` always commits what is pending, and so does `flush()`.

## Results (Apple M1 Pro, macOS 27, Mojo 1.2.0rc2 / MAX 26.7.0rc2)

### Launch micro-benchmarks (`bench_launch.mojo`)

The ObjC call itself, in ns per call:

| cached fn pointer | `get_function` per call (dlsym) | cached + `sel_registerName` per call |
|---|---|---|
| 1.7 | 458 | 18.7 |

The seed probe resolved `objc_msgSend` on every call. That costs about 0.46
us per send, and a launch makes about 5 sends, so about 2.3 us per launch.
With the cache, the host cost of encoding a launch (set pipeline if it
changed, `setBytes` per argument, dispatch) is **0.15 us**.

2000 dependent 1-thread launches, in us per launch. "-1" means updates were
lost.

| launches / cb | fence + status (default) | blocking sync | event wait chain | tracked + useResources | fence + event signal per cb | no ordering |
|---|---|---|---|---|---|---|
| 1 | 21.4 | 21.7 | 110.7 | 21.8 | 42.0 | -1 (1979/2000) |
| 4 | 6.0 | 6.1 | 27.4 | 6.1 | 11.1 | -1 (1726) |
| 16 | 2.8 | 3.0 | 9.5 | 3.0 | 2.8 | -1 (1585) |
| 64 | 2.5 | 2.5 | 4.0 | 2.5 | 2.6 | -1 (1405) |
| 2000 | 2.3 | 2.2 | 2.2 | 2.2 | 2.2 | 2.2 |
| adaptive N=16 / N=64 | 3.0 / 2.4 | | | | | |

MAX's `enqueue_function` costs 22.0 us per launch on the same loop. At one
launch per command buffer the GPU-side cost of a command buffer is about
21 us, whoever submits it. All of the per-launch gain comes from putting
several launches in one command buffer.

Synced round trip: one 1-thread launch, then wait for it. Median of 400, in
us.

| MAX enqueue + synchronize | ours, spin on `status` | ours, `waitUntilCompleted` | ours, spin on event (default) | ours, spin on the value the kernel wrote |
|---|---|---|---|---|
| 215 | 177 | 176 | **101** | 78 |

The last column is a floor, not an API: the host spins on the GPU's write in
shared memory.

Live buffers: 2000 dependent launches with 0, 256 or 1024 extra live buffers,
in us per launch.

| extra live buffers | MAX | ours, residency set, 1/cb | set, 16/cb | `useResources` per cb, 1/cb | per cb, 16/cb | per launch, 16/cb |
|---|---|---|---|---|---|---|
| 0 | 22.0 | 21.9 | 2.8 | 22.4 | 2.8 | 2.8 |
| 256 | 46.8 | 22.7 | 3.0 | 34.0 | 2.8 | 2.6 |
| 1024 | 128.5 | **21.8** | **2.8** | 90.6 | 5.3 | 6.8 |

With the residency set, our cost stays flat as the number of live buffers
grows. MAX's cost grows by about 0.1 us per live buffer per launch.
Declaring every buffer per command buffer grows too, but the cost is shared
by the launches in that command buffer.

Metal 4 (`Metal4RT`), 2000 dependent launches, in us per launch:

| launches / cb | with barriers | no barriers |
|---|---|---|
| 1 | 42.3 | -1 (1995/2000) |
| 4 | 10.2 | -1 (498) |
| 16 | 3.1 | -1 (124) |
| 64 | 2.6 | -1 (119) |
| 2000 | 4.4 | -1 (128) |

Encoding costs 2.4 us of host time per launch, and an intra-pass barrier
accounts for 1.5 us of that. The synced round trip is 94 to 130 us; it was
noisy across runs.

### The 2-layer MLP training step (`bench_mlp.mojo`)

The step is I=16, O=4, 14 launches: forward pass, MSE loss, full backward
pass, and Adam on W1, b1, W2, b2. Each cell is 500 timed steps after 50
warm-up steps, in us per step, median of 5 repeats. The MAX column uses
`DeviceContext` + `DeviceBuffer` + `enqueue_function`. Its loss read-back is
an `enqueue_copy` to a host buffer plus `synchronize()`. Ours reads the
shared buffer after `sync()`.

**Loss read back every 20 steps:**

| H | B | MAX | each | every-16 | every-64 | adaptive-16 | adaptive-16, age 20us | Metal 4 every-16 | best ours vs MAX |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 1 | 367.6 | 341.6 | 95.4 | 73.7 | 89.2 | 87.7 | 88.5 | 5.0x |
| 64 | 64 | 379.0 | 346.3 | 160.7 | 142.1 | 161.2 | 160.3 | 160.7 | 2.7x |
| 64 | 256 | 514.6 | 512.4 | 378.0 | 353.7 | 376.9 | 381.9 | 376.9 | 1.5x |
| 64 | 1024 | 1451.3 | 1441.0 | 1230.9 | 1205.1 | 1229.8 | 1233.3 | 1230.3 | 1.2x |
| 256 | 1 | 371.6 | 325.8 | 122.7 | 106.0 | 124.0 | 122.3 | 124.2 | 3.5x |
| 256 | 64 | 380.6 | 340.0 | 220.6 | 196.5 | 219.6 | 215.4 | 215.2 | 1.9x |
| 256 | 256 | 667.8 | 668.3 | 514.2 | 493.0 | 511.5 | 521.0 | 515.2 | 1.4x |
| 256 | 1024 | 1960.4 | 1950.6 | 1677.8 | 1672.8 | 1679.4 | 1670.5 | 1680.5 | 1.2x |

**Loss read back every step** (the RL-style loop, where the host waits for
the GPU every step):

| H | B | MAX | each | every-16 | every-64 | adaptive-16 | adaptive-16, age 20us | Metal 4 every-16 | best ours vs MAX |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 1 | 617.8 | 572.9 | 340.1 | 337.1 | 327.3 | 296.3 | 308.8 | 2.1x |
| 64 | 64 | 635.5 | 772.4 | 477.8 | 507.0 | 465.0 | 472.0 | 491.7 | 1.4x |
| 64 | 256 | 875.8 | 1039.2 | 568.7 | 563.6 | 562.8 | 549.8 | 541.2 | 1.6x |
| 64 | 1024 | 1769.2 | 1649.6 | 1370.9 | 1380.3 | 1393.6 | 1370.8 | 1360.0 | 1.3x |
| 256 | 1 | 635.2 | 686.3 | 441.1 | 474.0 | 457.4 | 422.3 | 426.7 | 1.5x |
| 256 | 64 | 656.5 | 927.1 | 551.5 | 529.6 | 532.1 | 517.3 | 517.6 | 1.3x |
| 256 | 256 | 976.8 | 1107.3 | 649.9 | 648.8 | 673.6 | 644.0 | 652.1 | 1.5x |
| 256 | 1024 | 2274.9 | 2172.9 | 1837.4 | 1846.1 | 1850.7 | 1832.0 | 1824.0 | 1.2x |

How to read these tables:

- **Small B.** The gain is launch overhead, and it is large. With 16 to 64
  launches per command buffer the step costs 74 to 95 us, against 368 for
  MAX and 342 for our own one-launch-per-command-buffer policy.
- **B=1024.** The naive kernels make the step GPU-bound, so what is left is
  roughly MAX's per-launch overhead, about 15 to 20 us per launch.
- **Every-step loop.** Each step pays one synced round trip. That is about
  100 us on our event path (215 us for MAX's synchronize), plus the time the
  GPU takes to come back from idle, so even the best policy stays above
  about 300 us per step.
- **Noise.** The every-step numbers move ±20% from run to run, likely
  because of GPU power states, so compare medians only.

**The adaptive policy in the every-step loop.** The rule as specified
("commit at once when the GPU has nothing in flight") commits **2 command
buffers per step**: the step's first launch goes alone, because the GPU is
idle after the previous sync, and the other 13 follow at the sync. That is
one extra command buffer per step (about 20 us of GPU time) and buys
nothing, because the host encodes the next launch 0.15 us later. With
`idle_age_us=20`, a lone launch is committed to an idle GPU only once it
has waited 20 us. That gives 1.0 command buffer per step (idle commits
0–1%), and it was the fastest or tied at most sizes. When the loss is read
back every 20 steps, both variants run at about 0.9–1.0 command buffers per
step, because the GPU is never idle.

**The tail problem.** Here the host does CPU work between a step's launches
and its loss read-back, work that does not need the GPU's result (an env
step, say). H=64, B=64, loss read back every step, us per step, median of 5:

| host CPU us/step | MAX | each | every-16 | adaptive-16 | adaptive-16, age 20us | every-16 + `flush()` per step |
|---|---|---|---|---|---|---|
| 0 | 684.5 | 772.0 | 461.0 | 468.6 | 441.2 | **395.0** |
| 100 | 678.3 | 797.4 | 620.1 | 484.8 | 596.3 | **406.6** |
| 300 | 824.3 | 761.7 | 834.2 | 809.7 | 829.7 | **452.7** |
| 600 | 1159.0 | 788.4 | 1172.5 | 1130.1 | 1168.7 | **626.0** |

- **Batching policies.** Every policy that batches holds the step's last
  launches on the host while the CPU works, so the GPU sits idle and the step
  costs CPU time plus GPU time. Adaptive cannot help: the last launches of a
  step always arrive while the GPU is busy with the first ones, so it batches
  them.
- **`flush()`.** One `flush()` per step, right before the CPU work, fixes it.
  The step then costs about max(CPU, GPU) plus the round trip.
- **COMMIT_EACH.** It also overlaps, but pays about 21 us per launch.
- **No automatic fix.** No launch-time rule can fix this. It would take a
  background thread that commits stale work, which is not implemented.

**Gate:** 660 of 660 runs are bit-identical to the MAX run of the same size:
the final W1, b1, W2, b2 and the loss of every one of the 550 steps, compared
as `uint32` bits. That covers every size, both loop shapes, every policy,
both APIs, the MAX re-runs, and the CPU-work table. The losses fall from
about 2.0 to about 0.001–0.008 (to about 1e-17 at B=1), so the gate does not
compare a run that is doing nothing. Negative control: the same MLP with
`ORDER_NONE` diverges from MAX, and the gate catches it.

## Pitfalls hit

1. **`objc_msgSend` must be called through a pointer with the method's exact
   prototype.** It is not variadic at the call site on arm64. One `Int`-only
   prototype per arity covers every method used here.
2. **Struct arguments.** `MTLSize` is 24 bytes, and arm64 passes structs
   larger than 16 bytes by reference. Pass the address of three NSUIntegers.
3. **BOOL returns.** Only the low byte of w0 is defined. Mask with `& 0xFF`.
4. **A Mojo `String` is freed at its last use.** For `Int(s.unsafe_ptr())`,
   that last use is the `unsafe_ptr()` call, before the C function runs.
   Symptom: `-[MTLResidencySetDescriptor ]: unrecognized selector`, an EMPTY
   selector name. Every helper ends the string's life after the call
   (`_ = s^`).
5. **`external_call["dlopen"/"dlsym"]` clashes with the stdlib's own
   declarations** ("existing function with conflicting signature"). Instead,
   `dlsym(RTLD_DEFAULT = -2, name)` is called through
   `OwnedDLHandle.get_function`, and the returned address is cast to a thin
   `abi("C")` function type with `Pointer.unsafe_bitcast`.
6. **`_compile_code` cannot run at compile time** ("failed to fold operation
   kgen.compile_offload"), so it runs on every launch. Its `function_name`
   points to static storage, so the address of that string is the key of the
   pipeline cache: a pointer compare, with no hashing.
7. **MAX's `enqueue_function` rejects `Int` arguments** ("Int and UInt do
   not conform to DevicePassable"). The kernels take `Int64`.
8. **The host address is not the GPU address.** A shared buffer's
   `contents` and its `gpuAddress` are different numbers.
9. **Command buffers on one queue overlap.** With hazard-untracked buffers
   that are reached only through GPU addresses, consecutive command buffers
   run concurrently on the GPU, and 2000 dependent increments lose 1–30% of
   their updates. A serial encoder orders only its own dispatches. The
   fixes, by cost:
   - an `MTLFence` (free);
   - tracked buffers declared with `useResources` (cost grows with the
     number of live buffers);
   - a shared-event wait chain (110 us per command buffer, against 22).
10. **A shared-event signal costs about 20 us of GPU time per command
    buffer** (42 vs 22 us/launch at one launch per command buffer). So the
    event is signalled only by the command buffer that `sync()` or `flush()`
    commits.
11. **`status` lags the event by about 80 us.** Right after a `sync()` that
    waited on the event, `status` still says "not completed". The adaptive
    policy's "is the GPU idle?" was reading the stale `status`, which
    silently turned it into every-N. `gpu_idle()` now reads the event when
    the last command buffer signalled it.
12. **Autorelease pools.** The command buffer and its encoder are
    autoreleased. There is one pool per command buffer (pushed at begin,
    popped after commit), and the last command buffer is `retain`ed so that
    its `status` can be read later.
13. **A residency set needs `commit` after `addAllocation:`.** Otherwise the
    new buffer is not resident. The set is committed lazily when the next
    command buffer begins.
14. **Metal 4.**
    - There is no `setBytes`, so argument bytes go into a shared ring
      buffer, and the argument table gets their addresses. The table is
      snapshotted at each dispatch, so one table can be reused.
    - Compute encoders are concurrent. Without barriers, 2000 dependent
      increments end at about 125.
    - The intra-pass barrier costs 1.5 us of host time per call. That makes
      the Metal 4 path slower than the classic one for long chains of
      dependent dispatches (4.4 vs 2.3 us/launch at 2000 per command
      buffer). Metal 4 pays off only if barriers are placed where real
      dependencies are, and `launch()` cannot see dependencies.
    - Allocators may be `reset` only after their command buffers complete.
      A ring of 16 allocators is recycled by the event value.
15. **"Context leak detected, CoreAnalytics returned false"** is sometimes
    printed during the MLP benchmark. It is harmless.

## Limits

- **No argument type check.** `launch` binds whatever bytes it is given.
  Pass the wrong argument type and you get garbage, not a compile error.
- **Grids are 1-D only** (`grid=`, `block=` are threadgroups and threads per
  group).
- **Objects are never released.** The benchmarks never release the device,
  queue, pipeline or buffer objects; that is acceptable for a prototype.
  `MetalRT.free(buf)` exists, and it syncs first.
- **The tail problem needs an explicit `flush()`.** There is no background
  committer thread.
- **Metal 4 has no dependency tracking.** It puts a barrier between every
  pair of dispatches.
