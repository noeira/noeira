"""TD-MPC2 MPPI rollout callbacks (storage framework) — bridge the nn world
model into the shared `noeira.planners.trajectory` MPPI planners.

`TDMPC2RolloutCallbackCPU` (B=1, List[Float64] views) and
`TDMPC2RolloutCallbackGPU` (row-major LayoutTensor views, batched) hold raw
pointers to the trainer-owned storage modules (dynamics / reward / policy /
target-Q ensemble) + an owned TwoHotDecode. The three trait methods map to
the world model's forward passes:

  * policy_action(z) → tanh(mean) of π(z)  — normalized [-1,1] seed.
  * rollout_step(z, a) → (z' = dynamics(z, a·scale),
    reward = two-hot-decode(reward(z, a·scale))).
  * terminal_value(z) → avg of 2 (random) target-Q heads at π(z), decoded.

Storage migration: net `forward` takes `TensorRefs[1](Tensor)`. The CPU path
stages List[Float64] into owned scratch Tensors; the GPU path bridges the
planner's device LayoutTensor views into owned scratch Tensors via copy
kernels (the storage forward consumes owned Tensors). The GPU Q ensemble is
threaded as distinct fields q0..q4 with a comptime-guarded dispatch (two `mut`
List subscripts can't alias in one call).

Used only by `select_action_mpc` (GPU MPC planning); the default acting path
is MPC-off (`select_action`).
"""

from std.math import tanh
from std.random.philox import Random as PhiloxRandom
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Zero
from noeira.planners.trajectory.rollout_callback import (
    RolloutCallbackCPU, RolloutCallbackGPU,
)

from .nets import TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy
from .losses import TwoHotDecode
from .wm_graph import NQ


@fieldwise_init
struct TDMPC2RolloutCallbackCPU[
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    QP: Float64 = 0.0,
](RolloutCallbackCPU):
    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime ZA = Self.LATENT + Self.ACT

    var dyn: Pointer[Self.DynT, MutUntrackedOrigin]
    var rew: Pointer[Self.RewT, MutUntrackedOrigin]
    var pol: Pointer[Self.PolicyT, MutUntrackedOrigin]
    var qt: Pointer[List[Self.QNetT], MutUntrackedOrigin]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]
    var action_scale: Float64
    var qi: Int
    var qj: Int

    @staticmethod
    def make(
        mut dyn: Self.DynT,
        mut rew: Self.RewT,
        mut pol: Self.PolicyT,
        mut qt: List[Self.QNetT],
        action_scale: Float64,
        qi: Int,
        qj: Int,
    ) raises -> Self:
        return Self(
            dyn=rebind[Pointer[Self.DynT, MutUntrackedOrigin]](Pointer(to=dyn)),
            rew=rebind[Pointer[Self.RewT, MutUntrackedOrigin]](Pointer(to=rew)),
            pol=rebind[Pointer[Self.PolicyT, MutUntrackedOrigin]](Pointer(to=pol)),
            qt=rebind[Pointer[List[Self.QNetT], MutUntrackedOrigin]](Pointer(to=qt)),
            decode=TwoHotDecode[
                Self.BINS, Self.VMIN, Self.VMAX
            ].make["cpu", INIT=Zero](),
            action_scale=action_scale,
            qi=qi,
            qj=qj,
        )

    def policy_action_cpu(
        mut self, z: List[Float64], mut action_out: List[Float64],
    ) raises:
        var zt = Tensor.alloc(Self.LATENT)
        for i in range(Self.LATENT):
            zt.data[i] = Scalar[DT](z[i])
        var pio = Tensor.alloc(2 * Self.ACT)
        self.pol[].forward["cpu", 1](TensorRefs[1](zt), pio)
        for j in range(Self.ACT):
            action_out[j] = Float64(tanh(pio.data[j]))   # normalized [-1,1] mean

    def _za_tensor(
        self, z: List[Float64], a: List[Float64]
    ) raises -> Tensor:
        var p = Tensor.alloc(Self.ZA)
        for i in range(Self.LATENT):
            p.data[i] = Scalar[DT](z[i])
        for j in range(Self.ACT):
            p.data[Self.LATENT + j] = Scalar[DT](a[j] * self.action_scale)
        return p^

    def rollout_step_cpu(
        mut self,
        z: List[Float64],
        a: List[Float64],
        mut z_next_out: List[Float64],
    ) raises -> Float64:
        var za = self._za_tensor(z, a)
        # dynamics → z'
        var zn = Tensor.alloc(Self.LATENT)
        self.dyn[].forward["cpu", 1](TensorRefs[1](za), zn)
        for i in range(Self.LATENT):
            z_next_out[i] = Float64(zn.data[i])
        # reward → two-hot decode → scalar
        var rl = Tensor.alloc(Self.BINS)
        self.rew[].forward["cpu", 1](TensorRefs[1](za), rl)
        var rs = Tensor.alloc(1)
        self.decode.forward["cpu", 1](TensorRefs[1](rl), rs)
        return Float64(rs.data[0])

    def terminal_value_cpu(mut self, z: List[Float64]) raises -> Float64:
        # action = tanh(mean) of π(z), then za = [z, action·scale]
        var act = List[Float64](length=Self.ACT, fill=0.0)
        self.policy_action_cpu(z, act)
        var za = self._za_tensor(z, act)
        var ql = Tensor.alloc(Self.BINS)
        var qs = Tensor.alloc(1)
        self.qt[][self.qi].forward["cpu", 1](TensorRefs[1](za), ql)
        self.decode.forward["cpu", 1](TensorRefs[1](ql), qs)
        var qa = Float64(qs.data[0])
        self.qt[][self.qj].forward["cpu", 1](TensorRefs[1](za), ql)
        self.decode.forward["cpu", 1](TensorRefs[1](ql), qs)
        var qb = Float64(qs.data[0])
        return (qa + qb) * 0.5


# ──────────────────────────────────────────────────────────────────────
# GPU batched MPPI callback — the practical MPC path (MPPIGPUBatched plans
# all N_ENVS×TOTAL_SAMPLES trajectories per horizon step in one grid).
# ──────────────────────────────────────────────────────────────────────


def _copy_in_k[N: Int](
    src: Tile[DT, N],
    dst: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i])


def _extract_tanh_mean_k[B_: Int, ACT_: Int, POL_: Int](
    pio: Tile[DT, B_ * POL_],
    action: Tile[DT, B_ * ACT_],
):
    """action[b,j] = tanh(pio[b, j])  (mean = first ACT cols of [mean|log_std])."""
    var i = Int(global_idx.x)
    if i < B_ * ACT_:
        var b = i // ACT_
        var j = i % ACT_
        action[i] = tanh(rebind[Scalar[DT]](pio[b * POL_ + j]))


def _build_za_scaled_k[B_: Int, LATENT_: Int, ACT_: Int, ZA_: Int](
    z: Tile[DT, B_ * LATENT_],
    a: Tile[DT, B_ * ACT_],
    za: Tile[DT, B_ * ZA_],
    scale: Scalar[DT],
):
    """za[b] = [z[b] | a[b]·scale]."""
    var i = Int(global_idx.x)
    if i < B_ * ZA_:
        var b = i // ZA_
        var k = i % ZA_
        if k < LATENT_:
            za[i] = rebind[Scalar[DT]](z[b * LATENT_ + k])
        else:
            za[i] = rebind[Scalar[DT]](a[b * ACT_ + (k - LATENT_)]) * scale


def _avg2_k[B_: Int](
    qa: Tile[DT, B_],
    qb: Tile[DT, B_],
    out_v: Tile[DT, B_],
):
    var b = Int(global_idx.x)
    if b < B_:
        out_v[b] = Scalar[DT](0.5) * (
            rebind[Scalar[DT]](qa[b]) + rebind[Scalar[DT]](qb[b])
        )


def _flat[
    N: Int, src_layout: TensorLayout
](
    t: TileTensor[DT, src_layout, MutAnyOrigin]
) -> Tile[DT, N]:
    """Flat 1D view over a contiguous LayoutTensor — the kernel-ABI bridge for
    the planner's 2D `(B, DIM)` views into the 1D-indexed callback kernels
    below. Origin is the incoming MutAnyOrigin (explicit, NOT inferred) so GPU
    codegen stays correct — this restores the pre-storage-migration `_ltg`
    flatten that the 1D kernels still require."""
    return tile[N](t.ptr)


@fieldwise_init
struct TDMPC2RolloutCallbackGPU[
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    NUM_Q: Int,
    BT: Int,   # BATCH_TOTAL = N_ENVS × TOTAL_SAMPLES
    QP: Float64 = 0.0,
](RolloutCallbackGPU):
    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime POL: Int = 2 * Self.ACT
    comptime ZA: Int = Self.LATENT + Self.ACT
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]

    var dyn: Pointer[Self.DynT, MutUntrackedOrigin]
    var rew: Pointer[Self.RewT, MutUntrackedOrigin]
    var pol: Pointer[Self.PolicyT, MutUntrackedOrigin]
    # 5 DISTINCT target-Q head pointers (NUM_Q fixed = 5; a List pointer can't
    # be split into two non-aliasing `mut` borrows, and storage Modules aren't
    # Copyable so a temporary List can't be built either).
    var qt0: Pointer[Self.QNetT, MutUntrackedOrigin]
    var qt1: Pointer[Self.QNetT, MutUntrackedOrigin]
    var qt2: Pointer[Self.QNetT, MutUntrackedOrigin]
    var qt3: Pointer[Self.QNetT, MutUntrackedOrigin]
    var qt4: Pointer[Self.QNetT, MutUntrackedOrigin]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]
    var action_scale: Scalar[DT]
    # device scratch Tensors (sized BT)
    var zin: Tensor
    var pio: Tensor
    var action: Tensor
    var za: Tensor
    var zout: Tensor
    var rlog: Tensor
    var qlog: Tensor
    var qa: Tensor
    var qb: Tensor

    @staticmethod
    def make(
        mut dyn: Self.DynT,
        mut rew: Self.RewT,
        mut pol: Self.PolicyT,
        mut qt0: Self.QNetT,
        mut qt1: Self.QNetT,
        mut qt2: Self.QNetT,
        mut qt3: Self.QNetT,
        mut qt4: Self.QNetT,
        action_scale: Scalar[DT],
        ctx: DeviceContext,
    ) raises -> Self:
        return Self(
            dyn=rebind[Pointer[Self.DynT, MutUntrackedOrigin]](Pointer(to=dyn)),
            rew=rebind[Pointer[Self.RewT, MutUntrackedOrigin]](Pointer(to=rew)),
            pol=rebind[Pointer[Self.PolicyT, MutUntrackedOrigin]](Pointer(to=pol)),
            qt0=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](Pointer(to=qt0)),
            qt1=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](Pointer(to=qt1)),
            qt2=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](Pointer(to=qt2)),
            qt3=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](Pointer(to=qt3)),
            qt4=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](Pointer(to=qt4)),
            decode=TwoHotDecode[
                Self.BINS, Self.VMIN, Self.VMAX
            ].make["gpu", INIT=Zero](ctx=ctx),
            action_scale=action_scale,
            zin=Tensor.alloc_gpu(ctx, Self.BT * Self.LATENT),
            pio=Tensor.alloc_gpu(ctx, Self.BT * Self.POL),
            action=Tensor.alloc_gpu(ctx, Self.BT * Self.ACT),
            za=Tensor.alloc_gpu(ctx, Self.BT * Self.ZA),
            zout=Tensor.alloc_gpu(ctx, Self.BT * Self.LATENT),
            rlog=Tensor.alloc_gpu(ctx, Self.BT * Self.BINS),
            qlog=Tensor.alloc_gpu(ctx, Self.BT * Self.BINS),
            qa=Tensor.alloc_gpu(ctx, Self.BT),
            qb=Tensor.alloc_gpu(ctx, Self.BT),
        )

    def policy_action_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        z: Tile[DT, B, Self.LATENT_DIM],
        action_out: Tile[DT, B, Self.ACTION_DIM],
    ) raises:
        comptime assert B == Self.BT, "callback B must equal BT"
        comptime nbc = (B * Self.LATENT + TPB - 1) // TPB
        ctx.enqueue_function[_copy_in_k[B * Self.LATENT]](
            _flat[B * Self.LATENT](z),
            self.zin.tt["gpu", B * Self.LATENT](),
            grid_dim=nbc, block_dim=TPB,
        )
        self.pol[].forward["gpu", B](
            TensorRefs[1](self.zin), self.pio, Optional(ctx)
        )
        comptime k = _extract_tanh_mean_k[B, Self.ACT, Self.POL]
        comptime nb = (B * Self.ACT + TPB - 1) // TPB
        ctx.enqueue_function[k](
            self.pio.tt["gpu", B * Self.POL](),
            _flat[B * Self.ACT](action_out),
            grid_dim=nb, block_dim=TPB,
        )

    def rollout_step_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        z: Tile[DT, B, Self.LATENT_DIM],
        a: Tile[DT, B, Self.ACTION_DIM],
        z_next_out: Tile[DT, B, Self.LATENT_DIM],
        r_out: Tile[DT, B],
    ) raises:
        comptime assert B == Self.BT, "callback B must equal BT"
        # za = [z | a·scale] — read the planner views directly.
        comptime bza = _build_za_scaled_k[B, Self.LATENT, Self.ACT, Self.ZA]
        comptime nbz = (B * Self.ZA + TPB - 1) // TPB
        ctx.enqueue_function[bza](
            _flat[B * Self.LATENT](z), _flat[B * Self.ACT](a),
            self.za.tt["gpu", B * Self.ZA](),
            self.action_scale, grid_dim=nbz, block_dim=TPB,
        )
        # dynamics → zout, copy into the planner's z_next_out
        self.dyn[].forward["gpu", B](
            TensorRefs[1](self.za), self.zout, Optional(ctx)
        )
        comptime nbl = (B * Self.LATENT + TPB - 1) // TPB
        ctx.enqueue_function[_copy_in_k[B * Self.LATENT]](
            self.zout.tt["gpu", B * Self.LATENT](),
            _flat[B * Self.LATENT](z_next_out), grid_dim=nbl, block_dim=TPB,
        )
        # reward → two-hot decode → r_out (build a Tensor view of r_out target)
        self.rew[].forward["gpu", B](
            TensorRefs[1](self.za), self.rlog, Optional(ctx)
        )
        self.decode.forward["gpu", B](
            TensorRefs[1](self.rlog), self.qa, Optional(ctx)
        )
        comptime nbb = (B + TPB - 1) // TPB
        ctx.enqueue_function[_copy_in_k[B]](
            self.qa.tt["gpu", B](),
            r_out, grid_dim=nbb, block_dim=TPB,
        )

    def terminal_value_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        z: Tile[DT, B, Self.LATENT_DIM],
        v_out: Tile[DT, B],
        seed: UInt32,
    ) raises:
        comptime assert B == Self.BT, "callback B must equal BT"
        # host Philox → 2 distinct Q indices (matches legacy recipe).
        var rng = PhiloxRandom(seed=UInt64(seed) + UInt64(0xA1B2C3D4), offset=0)
        var u = rng.step_uniform()
        var qi = Int(Float64(u[0]) * Float64(Self.NUM_Q)) % Self.NUM_Q
        var qj = (
            qi + 1 + Int(Float64(u[1]) * Float64(Self.NUM_Q - 1))
            % (Self.NUM_Q - 1)
        ) % Self.NUM_Q

        # action = tanh(mean) of π(z)
        comptime nbc = (B * Self.LATENT + TPB - 1) // TPB
        ctx.enqueue_function[_copy_in_k[B * Self.LATENT]](
            _flat[B * Self.LATENT](z),
            self.zin.tt["gpu", B * Self.LATENT](),
            grid_dim=nbc, block_dim=TPB,
        )
        self.pol[].forward["gpu", B](
            TensorRefs[1](self.zin), self.pio, Optional(ctx)
        )
        comptime ek = _extract_tanh_mean_k[B, Self.ACT, Self.POL]
        comptime nba = (B * Self.ACT + TPB - 1) // TPB
        ctx.enqueue_function[ek](
            self.pio.tt["gpu", B * Self.POL](),
            self.action.tt["gpu", B * Self.ACT](),
            grid_dim=nba, block_dim=TPB,
        )
        # za = [z, action·scale]
        comptime bza = _build_za_scaled_k[B, Self.LATENT, Self.ACT, Self.ZA]
        comptime nbz = (B * Self.ZA + TPB - 1) // TPB
        ctx.enqueue_function[bza](
            _flat[B * Self.LATENT](z),
            self.action.tt["gpu", B * Self.ACT](),
            self.za.tt["gpu", B * Self.ZA](),
            self.action_scale, grid_dim=nbz, block_dim=TPB,
        )
        # avg of 2 (random) target-Q, two-hot decoded. Comptime-guarded
        # dispatch picks two DISTINCT head fields (qi, qj are runtime; unrolled
        # NQ instantiations select the matching head).
        self._q_decode_into[target_qa=True](qi, ctx)
        self._q_decode_into[target_qa=False](qj, ctx)
        comptime ak = _avg2_k[B]
        comptime nbb = (B + TPB - 1) // TPB
        ctx.enqueue_function[ak](
            self.qa.tt["gpu", B](),
            self.qb.tt["gpu", B](),
            v_out, grid_dim=nbb, block_dim=TPB,
        )

    def _q_decode_into[target_qa: Bool](
        mut self, head: Int, ctx: DeviceContext
    ) raises:
        """forward the `head`-th target-Q (distinct field) on self.za, decode
        into self.qa/qb. Runtime `head` → comptime-unrolled field select."""
        # forward the matching distinct head into self.qlog.
        if head == 0:
            self.qt0[].forward["gpu", Self.BT](
                TensorRefs[1](self.za), self.qlog, Optional(ctx)
            )
        elif head == 1:
            self.qt1[].forward["gpu", Self.BT](
                TensorRefs[1](self.za), self.qlog, Optional(ctx)
            )
        elif head == 2:
            self.qt2[].forward["gpu", Self.BT](
                TensorRefs[1](self.za), self.qlog, Optional(ctx)
            )
        elif head == 3:
            self.qt3[].forward["gpu", Self.BT](
                TensorRefs[1](self.za), self.qlog, Optional(ctx)
            )
        else:
            self.qt4[].forward["gpu", Self.BT](
                TensorRefs[1](self.za), self.qlog, Optional(ctx)
            )
        comptime if target_qa:
            self.decode.forward["gpu", Self.BT](
                TensorRefs[1](self.qlog), self.qa, Optional(ctx)
            )
        else:
            self.decode.forward["gpu", Self.BT](
                TensorRefs[1](self.qlog), self.qb, Optional(ctx)
            )
