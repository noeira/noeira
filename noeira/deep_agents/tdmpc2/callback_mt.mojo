"""GPU MPPI rollout callback for MULTI-TASK TD-MPC2 — the planner's world-model
access, task-conditioned.

The multi-task sibling of `TDMPC2RolloutCallbackGPU`. `MPPIGPUBatched` is
task-agnostic (it plans in latent space and asks a callback to roll out), so
enabling MPC on the multi-task agent needs no planner change at all — only this
callback, which splices the task embedding into every net input.

⚠ THE CONCAT ORDER IS FIXED BY TRAINING, NOT BY TASTE. `wm_graph_mt.mojo`
declares `za = Concat[LATENT, MAX_ACT, TASK_EMB]("z","a","task_emb")` and feeds
that ONE tensor to dynamics, reward, every Q head and termination; the policy is
fed `[z | tem]` by `agent_mt.select_action`. So:

    dynamics / reward / Q / termination :  [ z | a·scale | tem ]
    policy                              :  [ z | tem ]

A callback that appends in any other order loads a checkpoint cleanly and
plans against scrambled features — the failure is silent and looks like "the
world model is bad", so the layout is asserted by construction here (the widths
`ZAMT` / `PIN` come from the same expressions the graph uses) and gated by
`tests/deep_agents/test_tdmpc2_multitask_mpc_gpu.mojo`.

⚠ The MT dynamics outputs a PLAIN `LATENT`-wide next latent (`wm_graph_mt`'s
`znext`), NOT the augmented `[z|tem]`. The rollout is therefore not closed over
an augmented latent: `tem` is re-appended at every horizon step. That is why
the tempting shortcut — reusing the single-task callback with
`LATENT := LATENT + TASK_EMB` — does not work.

`tem` is `[BT, TASK_EMB]`: one row per planning candidate, all identical
(a plan is for ONE env on ONE task). The agent fills it before planning via
`set_task_embedding`.
"""

from max.gpu import global_idx
from std.random.philox import Random as PhiloxRandom
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Zero
from noeira.planners.trajectory.rollout_callback import RolloutCallbackGPU

from .losses import TwoHotDecode
from .nets_mt import (
    TDMPC2DynamicsMT, TDMPC2RewardMT, TDMPC2QNetMT, TDMPC2PolicyMT,
)
from .callback import _copy_in_k, _extract_tanh_mean_k, _avg2_k, _flat
from noeira.core.tiles import Tile


def _build_pin_mt_k[
    B_: Int, LATENT_: Int, EMB_: Int, PIN_: Int
](
    z: Tile[DT, B_ * LATENT_],
    tem: Tile[DT, B_ * EMB_],
    pin: Tile[DT, B_ * PIN_],
):
    """`pin[b] = [z[b] | tem[b]]` — the multi-task policy input."""
    var i = Int(global_idx.x)
    if i < B_ * PIN_:
        var b = i // PIN_
        var k = i % PIN_
        if k < LATENT_:
            pin[i] = rebind[Scalar[DT]](z[b * LATENT_ + k])
        else:
            pin[i] = rebind[Scalar[DT]](tem[b * EMB_ + (k - LATENT_)])


def _build_za_mt_k[
    B_: Int, LATENT_: Int, ACT_: Int, EMB_: Int, ZA_: Int
](
    z: Tile[DT, B_ * LATENT_],
    a: Tile[DT, B_ * ACT_],
    tem: Tile[DT, B_ * EMB_],
    za: Tile[DT, B_ * ZA_],
    scale: Scalar[DT],
):
    """`za[b] = [z[b] | a[b]·scale | tem[b]]` — the order `wm_graph_mt`'s
    `Concat[LATENT, MAX_ACT, TASK_EMB]` produces during training."""
    var i = Int(global_idx.x)
    if i < B_ * ZA_:
        var b = i // ZA_
        var k = i % ZA_
        if k < LATENT_:
            za[i] = rebind[Scalar[DT]](z[b * LATENT_ + k])
        elif k < LATENT_ + ACT_:
            za[i] = rebind[Scalar[DT]](a[b * ACT_ + (k - LATENT_)]) * scale
        else:
            za[i] = rebind[Scalar[DT]](
                tem[b * EMB_ + (k - LATENT_ - ACT_)]
            )


@fieldwise_init
struct TDMPC2RolloutCallbackGPUMT[
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    EMB: Int,
    NUM_Q: Int,
    BT: Int,   # BATCH_TOTAL = N_ENVS × TOTAL_SAMPLES
    QP: Float64 = 0.0,
](RolloutCallbackGPU):
    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime POL: Int = 2 * Self.ACT
    comptime PIN: Int = Self.LATENT + Self.EMB
    comptime ZAMT: Int = Self.LATENT + Self.ACT + Self.EMB
    comptime DynT = TDMPC2DynamicsMT[
        Self.LATENT, Self.ACT, Self.MLP, Self.SN, Self.EMB
    ]
    comptime RewT = TDMPC2RewardMT[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.EMB
    ]
    comptime QNetT = TDMPC2QNetMT[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.EMB, Self.QP
    ]
    comptime PolicyT = TDMPC2PolicyMT[
        Self.LATENT, Self.ACT, Self.MLP, Self.EMB
    ]

    var dyn: Pointer[Self.DynT, MutUntrackedOrigin]
    var rew: Pointer[Self.RewT, MutUntrackedOrigin]
    var pol: Pointer[Self.PolicyT, MutUntrackedOrigin]
    var qt0: Pointer[Self.QNetT, MutUntrackedOrigin]
    var qt1: Pointer[Self.QNetT, MutUntrackedOrigin]
    var qt2: Pointer[Self.QNetT, MutUntrackedOrigin]
    var qt3: Pointer[Self.QNetT, MutUntrackedOrigin]
    var qt4: Pointer[Self.QNetT, MutUntrackedOrigin]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]
    var action_scale: Scalar[DT]
    # device scratch (sized BT)
    var pin: Tensor
    var pio: Tensor
    var action: Tensor
    var za: Tensor
    var zout: Tensor
    var rlog: Tensor
    var qlog: Tensor
    var qa: Tensor
    var qb: Tensor
    # [BT, EMB] — the planned task's embedding, broadcast over candidates.
    var tem: Tensor

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
            pol=rebind[Pointer[Self.PolicyT, MutUntrackedOrigin]](
                Pointer(to=pol)
            ),
            qt0=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](
                Pointer(to=qt0)
            ),
            qt1=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](
                Pointer(to=qt1)
            ),
            qt2=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](
                Pointer(to=qt2)
            ),
            qt3=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](
                Pointer(to=qt3)
            ),
            qt4=rebind[Pointer[Self.QNetT, MutUntrackedOrigin]](
                Pointer(to=qt4)
            ),
            decode=TwoHotDecode[
                Self.BINS, Self.VMIN, Self.VMAX
            ].make["gpu", INIT=Zero](ctx=ctx),
            action_scale=action_scale,
            pin=Tensor.alloc_gpu(ctx, Self.BT * Self.PIN),
            pio=Tensor.alloc_gpu(ctx, Self.BT * Self.POL),
            action=Tensor.alloc_gpu(ctx, Self.BT * Self.ACT),
            za=Tensor.alloc_gpu(ctx, Self.BT * Self.ZAMT),
            zout=Tensor.alloc_gpu(ctx, Self.BT * Self.LATENT),
            rlog=Tensor.alloc_gpu(ctx, Self.BT * Self.BINS),
            qlog=Tensor.alloc_gpu(ctx, Self.BT * Self.BINS),
            qa=Tensor.alloc_gpu(ctx, Self.BT),
            qb=Tensor.alloc_gpu(ctx, Self.BT),
            tem=Tensor.alloc_gpu(ctx, Self.BT * Self.EMB),
        )

    def set_task_embedding(
        mut self, mut row: Tensor, ctx: DeviceContext
    ) raises:
        """Broadcast a ONE-row `[EMB]` embedding across all `BT` candidates.

        Called before every `plan_gpu`, not once per episode: the table is
        trained, so a row captured earlier plans with an embedding the world
        model has moved away from."""
        comptime nb = (Self.BT * Self.EMB + TPB - 1) // TPB
        ctx.enqueue_function[_bcast_emb_k[Self.BT, Self.EMB]](
            row.tt["gpu", Self.EMB](),
            self.tem.tt["gpu", Self.BT * Self.EMB](),
            grid_dim=nb, block_dim=TPB,
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
        # pin = [z | tem]
        comptime nbp = (B * Self.PIN + TPB - 1) // TPB
        ctx.enqueue_function[
            _build_pin_mt_k[B, Self.LATENT, Self.EMB, Self.PIN]
        ](
            _flat[B * Self.LATENT](z),
            self.tem.tt["gpu", B * Self.EMB](),
            self.pin.tt["gpu", B * Self.PIN](),
            grid_dim=nbp, block_dim=TPB,
        )
        self.pol[].forward["gpu", B](
            TensorRefs[1](self.pin), self.pio, Optional(ctx)
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
        # za = [z | a·scale | tem]
        comptime bza = _build_za_mt_k[
            B, Self.LATENT, Self.ACT, Self.EMB, Self.ZAMT
        ]
        comptime nbz = (B * Self.ZAMT + TPB - 1) // TPB
        ctx.enqueue_function[bza](
            _flat[B * Self.LATENT](z), _flat[B * Self.ACT](a),
            self.tem.tt["gpu", B * Self.EMB](),
            self.za.tt["gpu", B * Self.ZAMT](),
            self.action_scale, grid_dim=nbz, block_dim=TPB,
        )
        # dynamics → zout (PLAIN LATENT wide) → planner's z_next_out
        self.dyn[].forward["gpu", B](
            TensorRefs[1](self.za), self.zout, Optional(ctx)
        )
        comptime nbl = (B * Self.LATENT + TPB - 1) // TPB
        ctx.enqueue_function[_copy_in_k[B * Self.LATENT]](
            self.zout.tt["gpu", B * Self.LATENT](),
            _flat[B * Self.LATENT](z_next_out), grid_dim=nbl, block_dim=TPB,
        )
        # reward → two-hot decode → r_out
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
        # Same host-Philox 2-of-NUM_Q pick as the single-task callback.
        var rng = PhiloxRandom(seed=UInt64(seed) + UInt64(0xA1B2C3D4), offset=0)
        var u = rng.step_uniform()
        var qi = Int(Float64(u[0]) * Float64(Self.NUM_Q)) % Self.NUM_Q
        var qj = (
            qi + 1 + Int(Float64(u[1]) * Float64(Self.NUM_Q - 1))
            % (Self.NUM_Q - 1)
        ) % Self.NUM_Q

        # action = tanh(mean) of π([z|tem])
        comptime nbp = (B * Self.PIN + TPB - 1) // TPB
        ctx.enqueue_function[
            _build_pin_mt_k[B, Self.LATENT, Self.EMB, Self.PIN]
        ](
            _flat[B * Self.LATENT](z),
            self.tem.tt["gpu", B * Self.EMB](),
            self.pin.tt["gpu", B * Self.PIN](),
            grid_dim=nbp, block_dim=TPB,
        )
        self.pol[].forward["gpu", B](
            TensorRefs[1](self.pin), self.pio, Optional(ctx)
        )
        comptime ek = _extract_tanh_mean_k[B, Self.ACT, Self.POL]
        comptime nba = (B * Self.ACT + TPB - 1) // TPB
        ctx.enqueue_function[ek](
            self.pio.tt["gpu", B * Self.POL](),
            self.action.tt["gpu", B * Self.ACT](),
            grid_dim=nba, block_dim=TPB,
        )
        # za = [z | action·scale | tem]
        comptime bza = _build_za_mt_k[
            B, Self.LATENT, Self.ACT, Self.EMB, Self.ZAMT
        ]
        comptime nbz = (B * Self.ZAMT + TPB - 1) // TPB
        ctx.enqueue_function[bza](
            _flat[B * Self.LATENT](z),
            self.action.tt["gpu", B * Self.ACT](),
            self.tem.tt["gpu", B * Self.EMB](),
            self.za.tt["gpu", B * Self.ZAMT](),
            self.action_scale, grid_dim=nbz, block_dim=TPB,
        )
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
        """Forward the `head`-th target-Q (a DISTINCT field — two `mut` List
        subscripts can't alias) on `self.za`, decode into qa/qb."""
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


def _bcast_emb_k[B_: Int, EMB_: Int](
    row: Tile[DT, EMB_],
    dst: Tile[DT, B_ * EMB_],
):
    """`dst[b, e] = row[e]` for every planning candidate `b`."""
    var i = Int(global_idx.x)
    if i < B_ * EMB_:
        dst[i] = rebind[Scalar[DT]](row[i % EMB_])
