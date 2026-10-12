"""State memory per GPU: DDP, ZeRO-1, ZeRO-2, ZeRO-3, on the nanoGPT-size GPT.

Builds each wrapper for the 6 x 384 char-GPT (seq 256, tied head) at N = 2,
4 and 8 ranks on the simulator backend, runs its setup (packing, broadcast,
sharding, binding) and prints the bytes each one allocated per rank for
weights, gradients, optimizer state and collective scratch: the largest
rank, from the allocations themselves, not a formula. No step runs, so
activations (the same for every wrapper at a given per-rank batch) and the
Signal payload (sized by the caller) are not included.

The decay mask (one fp32 per element, AdamW's per-element weight-decay gate)
is reported separately: DDP and ZeRO-1 keep it for the whole arena, ZeRO-2/3
for their shard.

Run (Mac, ~1 GB peak):
  pixi run -e apple mojo build -I . examples/nn/distributed/zero_memory.mojo -o $B/zm && $B/zm
"""

from std.random import seed
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Normal
from noeira.nn.distributed.process_group import ProcessGroup
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.zero import Zero1
from noeira.nn.distributed.zero_sharded import ZeroSharded
from noeira.nn.distributed.gpt_marked import GPTMarked


comptime NET = GPTMarked[
    65, 256, 384, 6, 6, 4, True, 0.0, UInt64(0xC0FFEE), True, True
]


def _mb(b: Int) -> String:
    var x = Float64(b) / 1e6
    return String(Float64(Int(x * 10.0 + 0.5)) / 10.0)


def _ddp[N: Int](ctx: DeviceContext) raises -> Tuple[Int, Int]:
    seed(1)
    var dp = DataParallel[NET, N].make[Normal[0.0, 0.02]](
        ProcessGroup[N].shared(ctx), lr=1e-3
    )
    dp.synchronize()
    # val + grd + red + m + v; the decay mask is one more arena
    return (dp.state_bytes_per_rank(), dp.total * 4)


def _zero1[N: Int](ctx: DeviceContext) raises -> Tuple[Int, Int]:
    seed(1)
    var z = Zero1[NET, N].make[Normal[0.0, 0.02]](
        ProcessGroup[N].shared(ctx), lr=1e-3
    )
    z.synchronize()
    var b = 0
    for r in range(N):
        b = max(b, z.state_bytes_per_rank(r))
    return (b, z.total * 4)


def _sharded[N: Int, STAGE: Int](
    ctx: DeviceContext, slots: Int
) raises -> Tuple[Int, Int, String]:
    seed(1)
    var z = ZeroSharded[NET, N, STAGE].make[Normal[0.0, 0.02]](
        ProcessGroup[N].shared(ctx), lr=1e-3, slots=slots
    )
    z.sync_params()
    var b = 0
    var dmask = 0
    for r in range(N):
        b = max(b, z.state_bytes_per_rank(r))
        dmask = max(dmask, z.rs[r].decay.n * 4)
    # state_bytes includes the decay shard; report it apart, as for DDP
    return (b - dmask, dmask, z.layout_summary())


def _row[N: Int](ctx: DeviceContext) raises:
    var d = _ddp[N](ctx)
    var z1 = _zero1[N](ctx)
    var z2 = _sharded[N, 2](ctx, 1)
    var z3 = _sharded[N, 3](ctx, 1)
    var z3b = _sharded[N, 3](ctx, 2)
    if N == 2:
        print("  layout:", z3[2])
    print(
        "  | " + String(N) + " | " + _mb(d[0]) + " | " + _mb(z1[0]) + " | "
        + _mb(z2[0]) + " | " + _mb(z3[0]) + " | " + _mb(z3b[0]) + " | "
        + _mb(d[1]) + " / " + _mb(z2[1]) + " |"
    )


def main() raises:
    var ctx = DeviceContext()
    seed(1)
    var probe = DataParallel[NET, 1].make[Normal[0.0, 0.02]](
        ProcessGroup[1].shared(ctx), lr=1e-3
    )
    var psi = probe.total
    print("GPT 6 x 384, seq 256: arena Psi =", psi, "fp32 elements =",
          _mb(psi * 4), "MB")
    print("  MB per rank: weights + grads + optimizer state + scratch"
          " (decay mask apart)")
    print("  | N | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 (1 slot) | ZeRO-3 (2 slots)"
          " | decay mask DDP / ZeRO-2,3 |")
    _row[2](ctx)
    _row[4](ctx)
    _row[8](ctx)
