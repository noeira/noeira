"""DQNTargetYBlock — target-Y compute for DQN (standard + Double) (STORAGE).

Forward-only block — produces `y = r + γ · max_a Q_target(s', a) · (1 − d)`.
No gradient flows through this path; the trainer only uses `y` as a target for
the critic update.

Linear chain (Q_target.forward → ReduceMax.forward → finalize kernel), not a
ComputeGraph — no fan-out, no Slice/Min, so a full graph would only add
overhead. The win is owned-scratch ownership, not DAG topology.

Double DQN (`DOUBLE=True`):

    y = r + γ · Q_target(s', argmax_a Q_online(s', a)) · (1 − d)

uses an additional `Q_online.forward(sp) → q_on`, an inline argmax helper
kernel that emits Scalar[DT] indices, then `GatherCols[NA]` to pick
`q_t[b, argmax]`. Tail finalize kernel is identical to the standard branch.

STORAGE migration (Stage 5): `Scratch`/`TargetStorage`/`init_scratch_auto`/
TileTensor gone — scratch are owned `nn.storage.Tensor`s; the Q nets forward
through the storage `Module` surface over `TensorRefs`; the GatherCols inputs
(Q_target(sp) + argmax idx) live in a block-owned `TensorPack[2]` so they share
ONE origin (§B0). CPU + GPU.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward
from noeira.nn.core.initializer import Zero
from noeira.nn.primitives.reduce_max import ReduceMax
from noeira.nn.primitives.gather_cols import GatherCols
from noeira.core.tiles import Tile


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — argmax-index emission and the finalize fuse.
# ──────────────────────────────────────────────────────────────────────


def _argmax_idx_kernel[
    BATCH: Int,
    NA: Int,
](
    q: Tile[DT, BATCH, NA],
    idx_out: Tile[DT, BATCH, 1],
):
    """`idx_out[b, 0] = argmax_a q[b, a]`, emitted as Scalar[DT]."""
    var b = Int(global_idx.x)
    if b < BATCH:
        var best_a: Int = 0
        var best_q: Scalar[DT] = rebind[Scalar[DT]](q[b, 0])
        for a in range(1, NA):
            var v = rebind[Scalar[DT]](q[b, a])
            if v > best_q:
                best_q = v
                best_a = a
        idx_out[b, 0] = Scalar[DT](best_a)


def _target_y_finalize_kernel[
    BATCH: Int,
](
    max_q: Tile[DT, BATCH, 1],
    r: Tile[DT, BATCH],
    d: Tile[DT, BATCH],
    y_out: Tile[DT, BATCH],
    gamma: Scalar[DT],
):
    """`y_out[b] = r[b] + γ · max_q[b, 0] · (1 − d[b])`."""
    var b = Int(global_idx.x)
    if b < BATCH:
        var nonterm = Scalar[DT](1.0) - rebind[Scalar[DT]](d[b])
        var mq = rebind[Scalar[DT]](max_q[b, 0])
        y_out[b] = rebind[Scalar[DT]](r[b]) + gamma * mq * nonterm


struct DQNTargetYBlock[
    Q_NET: Module,
    BATCH: Int,
    OBS: Int,
    NA: Int,
    DOUBLE: Bool = False,
](Defaultable & Movable & Deinitable):
    """Owns:
    - `ReduceMax[NA]` (standard branch reducer)
    - `GatherCols[NA]` (Double branch gather; constructed but unused on the
      standard path — keeps the field set uniform across DOUBLE configs)
    - `_gather_in: TensorPack[2]` — slot[0] = Q_target(sp) ([B, NA], also the
      ReduceMax input on the standard path); slot[1] = argmax idx ([B], Double)
    - `_q_on` ([B*NA]; Double only), `_max_q` ([B]).
    """

    var reduce_max: ReduceMax[Self.NA]
    var gather_cols: GatherCols[Self.NA]

    var _gather_in: TensorPack[2]
    var _q_on: Tensor       # [B*NA] — Q_online(sp) (Double only)
    var _max_q: Tensor      # [B] — bootstrap value

    var gamma: Scalar[DT]

    def __init__(out self):
        self.reduce_max = ReduceMax[Self.NA]()
        self.gather_cols = GatherCols[Self.NA]()
        self._gather_in = TensorPack[2]()
        self._q_on = Tensor()
        self._max_q = Tensor()
        self.gamma = Scalar[DT](0.99)

    @staticmethod
    def make[
        target: StaticString
    ](
        gamma: Scalar[DT] = Scalar[DT](0.99),
        nstep: Int = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory.

        `nstep` (default 1) bakes `γ^nstep` into the effective discount used by
        the finalize fuse. For n-step DQN the sample block emits compressed
        transitions where `mb_r` is the n-step return `Σ_{i<N} γ^i r_i` and
        `mb_sp` is the state at `t+N`; the bootstrap therefore uses `γ^N`. For
        uniform / 1-step replay, `nstep=1` keeps the discount at γ."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "DQNTargetYBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS
        ), "DQNTargetYBlock: Q_NET.IN_DIM must equal OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NA
        ), "DQNTargetYBlock: Q_NET.OUT_DIM must equal NA"
        var b = Self()
        # Bake γ^nstep so the finalize kernel multiplies by the right discount
        # regardless of replay flavor (nstep is small → runtime pow is fine).
        var gamma_n: Scalar[DT] = Scalar[DT](1.0)
        for _ in range(nstep):
            gamma_n = gamma_n * gamma
        b.gamma = gamma_n
        b.reduce_max = ReduceMax[Self.NA].make[target, INIT=Zero](ctx=ctx)
        b.gather_cols = GatherCols[Self.NA].make[target, INIT=Zero](ctx=ctx)
        comptime if target == "cpu":
            b._gather_in[0].ensure(Self.BATCH * Self.NA)
            b._gather_in[1].ensure(Self.BATCH)
            b._q_on = Tensor.alloc(Self.BATCH * Self.NA)
            b._max_q = Tensor.alloc(Self.BATCH)
        else:
            var c = ctx.value()
            b._gather_in[0].ensure_gpu(c, Self.BATCH * Self.NA)
            b._gather_in[1].ensure_gpu(c, Self.BATCH)
            b._q_on = Tensor.alloc_gpu(c, Self.BATCH * Self.NA)
            b._max_q = Tensor.alloc_gpu(c, Self.BATCH)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut q_target: Self.Q_NET,
        mut q_online: Self.Q_NET,
        mut mb_sp: Tensor,
        mut mb_r: Tensor,
        mut mb_d: Tensor,
        mut mb_y: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Standard: `mb_y[b] = mb_r[b] + γ·max_a Q_target(sp)[b,a]·(1−mb_d[b])`.
        Double:    `mb_y[b] = mb_r[b] + γ·Q_target(sp)[b, argmax_a Q_online(sp)[b,a]]·(1−mb_d[b])`.

        `q_online` is ignored on the standard path (DOUBLE=False). For Double,
        both nets are forwarded on `sp`."""
        # Step 1: Q_target(sp) → _gather_in[0] ([B, NA]).
        call_forward[target, Self.BATCH, POLICY=POLICY](
            q_target, TensorRefs[Self.Q_NET.ARITY](mb_sp), self._gather_in[0], ctx
        )

        # Step 2: best_q = either max or gather-by-online-argmax → _max_q.
        comptime if Self.DOUBLE:
            # Q_online(sp) → _q_on, argmax → _gather_in[1], gather Q_target → _max_q.
            call_forward[target, Self.BATCH, POLICY=POLICY](
                q_online, TensorRefs[Self.Q_NET.ARITY](mb_sp), self._q_on, ctx
            )
            comptime if target == "cpu":
                for bb in range(Self.BATCH):
                    var best_a: Int = 0
                    var best_q: Scalar[DT] = self._q_on.data[bb * Self.NA]
                    for a in range(1, Self.NA):
                        var v = self._q_on.data[bb * Self.NA + a]
                        if v > best_q:
                            best_q = v
                            best_a = a
                    self._gather_in[1].data[bb] = Scalar[DT](best_a)
            else:
                comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
                ctx.value().enqueue_function[
                    _argmax_idx_kernel[Self.BATCH, Self.NA]
                ](
                    self._q_on.tt["gpu", Self.BATCH, Self.NA](),
                    self._gather_in[1].tt["gpu", Self.BATCH, 1](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
            self.gather_cols.forward[target, Self.BATCH, POLICY=POLICY](
                TensorRefs[2](self._gather_in[0], self._gather_in[1]),
                self._max_q,
                ctx,
            )
        else:
            # Standard DQN: max_a Q_target(sp). ReduceMax is single-input.
            self.reduce_max.forward[target, Self.BATCH, POLICY=POLICY](
                TensorRefs[1](self._gather_in[0]),
                self._max_q,
                ctx,
            )

        # Step 3: y = r + γ·max_q·(1 − d).
        comptime if target == "cpu":
            for bb in range(Self.BATCH):
                var nonterm = Scalar[DT](1.0) - mb_d.data[bb]
                mb_y.data[bb] = (
                    mb_r.data[bb] + self.gamma * self._max_q.data[bb] * nonterm
                )
        else:
            comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
            ctx.value().enqueue_function[_target_y_finalize_kernel[Self.BATCH]](
                self._max_q.tt["gpu", Self.BATCH, 1](),
                mb_r.tt["gpu", Self.BATCH](),
                mb_d.tt["gpu", Self.BATCH](),
                mb_y.tt["gpu", Self.BATCH](),
                self.gamma,
                grid_dim=n_blocks,
                block_dim=TPB,
            )
