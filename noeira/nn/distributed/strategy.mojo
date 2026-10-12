"""ShardingStrategy — one interface for DDP and the three ZeRO stages.

A strategy owns one replica of the model per rank, the optimizer state and
the collectives that keep the replicas in step. A trainer drives it through
this trait and never names a stage, so the four are one type apart:

    DataParallel[M, N]       DDP: every rank holds weights, gradients and Adam
                             state; the gradients are all-reduced
    Zero1[M, N]              ZeRO-1: Adam state sharded over the ranks
    ZeroSharded[M, N, 2]     ZeRO-2: the gradients sharded too
    ZeroSharded[M, N, 3]     ZeRO-3: the weights sharded too, each block
                             gathered when the forward or backward needs it

(FSDP's names for three of them: NO_SHARD, SHARD_GRAD_OP, FULL_SHARD.)

The step, for every strategy:

    dist.set_lr(lr)
    dist.forward_backward(bodies)   # bodies[r].run_rank(r): rank r's forward,
                                    # loss and vjp; on return every rank holds
                                    # its part of the mean gradient
    dist.clip_grads_device(max_norm)
    dist.step()

A body is the trainer's work for one rank (`RankStep`, fibers.mojo). DDP and
ZeRO-1 run the bodies rank by rank and reduce once every backward is
enqueued. ZeRO-2/3 run them on `RankFibers`, so each block's collective sits
inside the backward, between the ranks' vjps of that block.

A body reaches its replica by address (`Pointer(to=dist.net(r))`). Every
strategy keeps its replicas in a list that is never resized after `make`,
so a replica does not move when the strategy does.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer

from .process_group import ProcessGroup
from .fibers import RankStep


trait ShardingStrategy(Deinitable, Movable):
    comptime Net: Module
    """The model each rank holds a replica of."""
    comptime RANKS: Int
    comptime REPLICATED: Bool
    """Whether every rank holds the full weights between steps (all but
    ZeRO-3), so that one rank can run a forward alone."""

    @staticmethod
    def make[
        INIT: Initializer
    ](
        var pg: ProcessGroup[Self.RANKS],
        lr: Scalar[DT],
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
        wd: Scalar[DT] = 0.0,
    ) raises -> Self:
        """One replica per rank, each initialized by `INIT`, plus Adam(W)
        state. The replicas are not synchronized yet: do any per-replica
        surgery on `net(r)` (weight tying, scaled init), then
        `sync_params`."""
        ...

    @staticmethod
    def name() -> String:
        """"DDP", "ZeRO-1", "ZeRO-2" or "ZeRO-3"."""
        ...

    def ctx(self, r: Int) -> DeviceContext:
        """Rank r's device context."""
        ...

    def net(mut self, r: Int) -> ref[MutAnyOrigin] Self.Net:
        """Rank r's replica. The reference is untracked: it stays valid as
        long as the strategy lives (its replicas never move)."""
        ...

    def sync_params(mut self) raises:
        """Rank 0's weights to every rank. Once, after any surgery."""
        ...

    def forward_backward[B: RankStep](mut self, mut bodies: List[B]) raises:
        """One step's gradients: `bodies[r].run_rank(r)` for every rank (its
        forward, loss and vjp), with the collectives the stage needs. On
        return each rank holds its part of the mean gradient over ranks."""
        ...

    def forward_only[B: RankStep](mut self, mut bodies: List[B]) raises:
        """`bodies[r].run_rank(r)` for every rank, for bodies that run a
        forward only (eval, generation). Under ZeRO-3 the forward is itself
        collective: it gathers each block's weights on the way."""
        ...

    def clip_grads_device(mut self, max_norm: Scalar[DT]) raises:
        """Global grad-norm clip, on the device (no host read)."""
        ...

    def set_lr(mut self, lr: Scalar[DT]):
        ...

    def step(mut self) raises:
        """The optimizer step on every rank, and whatever the stage needs to
        bring the weights back in step (ZeRO's all-gathers)."""
        ...

    def synchronize(self) raises:
        ...

    def download_params(mut self, r: Int) raises -> List[Scalar[DT]]:
        """Rank r's weights on the host, in `DataParallel`'s arena layout
        (walk order, every parameter on a `PARAM_ALIGN` boundary). ZeRO-3
        gathers them from every rank's shard."""
        ...

    def state_bytes_per_rank(self, r: Int) -> Int:
        """Bytes of weights, gradients and optimizer state held on rank r
        between steps (no activations)."""
        ...


def check_bodies(n_bodies: Int, ranks: Int, who: String) raises:
    """One body per rank, as fibers.mojo's contract asks."""
    if n_bodies != ranks:
        raise Error(
            who + ": " + String(n_bodies) + " bodies for " + String(ranks)
            + " ranks"
        )
