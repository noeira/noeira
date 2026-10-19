"""TD-MPC2 multi-task TD-target step (storage framework; CPU + GPU) — §14.3.

Clone of `td_target_step.mojo` with the task embedding concatenated into the
encoder input (`[obs|tem]`), the policy input (`[next_z|tem]`), and the Q input
(`za = [next_z|action|tem]`). Forward-only / stop-grad, so it contributes NO
gradient to the embedding table (matches the reference: targets are detached).

Storage migration: all buffers are storage `Tensor`s (host `.data` / device
`.dev`); the two target-Q heads are passed as DISTINCT `mut q_a, mut q_b` fields
(the agent's comptime dispatch picks the random pair, mirroring the single-task
`TDTargetStep`). `task_ids`/`obs`/`reward`/`done`/`td_out` are `Tensor`s.
See `td_target_step.mojo` for the bootstrap math.
"""

from std.math import min
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Zero
from noeira.nn.primitives.rsample import RSample

from .nets_mt import TDMPC2EncoderMT, TDMPC2PolicyMT, TDMPC2QNetMT
from .losses import TwoHotDecode
from .td_target_step import _td_combine_k
from .task_embedding import TaskEmbedding
from noeira.core.tiles import Tile, tile


def _cat2_k[B_: Int, A_: Int, BD_: Int](
    a: Tile[DT, B_ * A_],
    b: Tile[DT, B_ * BD_],
    dst: Tile[DT, B_ * (A_ + BD_)],
):
    """dst[row] = [a[row] | b[row]] (both per-row, widths A_ and BD_)."""
    var i = Int(global_idx.x)
    var W = A_ + BD_
    if i < B_ * W:
        var row = i // W
        var c = i % W
        if c < A_:
            dst[i] = rebind[Scalar[DT]](a[row * A_ + c])
        else:
            dst[i] = rebind[Scalar[DT]](b[row * BD_ + (c - A_)])


def _build_za3_k[B_: Int, LAT_: Int, A_: Int, ALP_: Int, EMB_: Int](
    nz: Tile[DT, B_ * LAT_],
    alp: Tile[DT, B_ * ALP_],
    tem: Tile[DT, B_ * EMB_],
    za: Tile[DT, B_ * (LAT_ + A_ + EMB_)],
):
    """za[row] = [nz | action(alp[:A_]) | tem]."""
    var i = Int(global_idx.x)
    var W = LAT_ + A_ + EMB_
    if i < B_ * W:
        var row = i // W
        var c = i % W
        if c < LAT_:
            za[i] = rebind[Scalar[DT]](nz[row * LAT_ + c])
        elif c < LAT_ + A_:
            za[i] = rebind[Scalar[DT]](alp[row * ALP_ + (c - LAT_)])
        else:
            za[i] = rebind[Scalar[DT]](tem[row * EMB_ + (c - LAT_ - A_)])


struct TDTargetStepMT[
    MAX_OBS: Int,
    ENC: Int,
    MAX_ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
    NUM_TASKS: Int,
    TASK_EMB: Int,
    QP: Float64 = 0.0,
](Movable & Deinitable):
    comptime AOBS = Self.MAX_OBS + Self.TASK_EMB
    comptime PIN = Self.LATENT + Self.TASK_EMB          # policy input width
    comptime ZA = Self.LATENT + Self.MAX_ACT + Self.TASK_EMB
    comptime EncT = TDMPC2EncoderMT[
        Self.MAX_OBS, Self.ENC, Self.LATENT, Self.SN, Self.TASK_EMB
    ]
    comptime PolicyT = TDMPC2PolicyMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime QNetT = TDMPC2QNetMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB, Self.QP
    ]
    comptime EmbT = TaskEmbedding[Self.NUM_TASKS, Self.TASK_EMB]

    var rsample: RSample[Self.MAX_ACT]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]

    # Persistent scratch Tensors (allocated once in make, reused every step).
    var tem: Tensor       # [B*TASK_EMB] gathered embeddings
    var ein: Tensor       # [B*AOBS]  encoder input [obs|tem]
    var obs_step: Tensor  # [B*MAX_OBS] obs window
    var nz: Tensor        # [B*LATENT]
    var pin: Tensor       # [B*PIN]  policy input [nz|tem]
    var pio: Tensor       # [B*2*MAX_ACT]
    var alp: Tensor       # [B*(MAX_ACT+1)]
    var za: Tensor        # [B*ZA]
    var qlog1: Tensor
    var qlog2: Tensor
    var qa: Tensor
    var qb: Tensor

    def __init__(out self):
        self.rsample = RSample[Self.MAX_ACT]()
        self.decode = TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]()
        self.tem = Tensor()
        self.ein = Tensor()
        self.obs_step = Tensor()
        self.nz = Tensor()
        self.pin = Tensor()
        self.pio = Tensor()
        self.alp = Tensor()
        self.za = Tensor()
        self.qlog1 = Tensor()
        self.qlog2 = Tensor()
        self.qa = Tensor()
        self.qb = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TDTargetStepMT: target must be 'cpu' or 'gpu'"
        )
        comptime LAT = Self.LATENT
        comptime A = Self.MAX_ACT
        comptime MO = Self.MAX_OBS
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime PIN = Self.PIN
        comptime ZA = Self.ZA
        comptime BB = Self.B
        var s = Self()
        s.rsample = RSample[Self.MAX_ACT].make[target, INIT=Zero](ctx=ctx)
        s.decode = TwoHotDecode[
            Self.BINS, Self.VMIN, Self.VMAX
        ].make[target, INIT=Zero](ctx=ctx)
        s.tem = Tensor.make[target](BB * EMB, ctx)
        s.ein = Tensor.make[target](BB * AOBS, ctx)
        s.obs_step = Tensor.make[target](BB * MO, ctx)
        s.nz = Tensor.make[target](BB * LAT, ctx)
        s.pin = Tensor.make[target](BB * PIN, ctx)
        s.pio = Tensor.make[target](BB * 2 * A, ctx)
        s.alp = Tensor.make[target](BB * (A + 1), ctx)
        s.za = Tensor.make[target](BB * ZA, ctx)
        s.qlog1 = Tensor.make[target](BB * Self.BINS, ctx)
        s.qlog2 = Tensor.make[target](BB * Self.BINS, ctx)
        s.qa = Tensor.make[target](BB, ctx)
        s.qb = Tensor.make[target](BB, ctx)
        return s^

    def step[target: StaticString](
        mut self,
        mut enc: Self.EncT,
        mut policy: Self.PolicyT,
        mut q_a: Self.QNetT,
        mut q_b: Self.QNetT,
        mut task_emb: Self.EmbT,
        mut task_ids: Tensor,  # [B] per-window DT ids
        mut obs: Tensor,       # [(H+1),B,MAX_OBS]
        mut reward: Tensor,    # [H,B]
        mut done: Tensor,      # [H,B]
        mut td_out: Tensor,    # [H,B] (written)
        gamma: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime LAT = Self.LATENT
        comptime A = Self.MAX_ACT
        comptime MO = Self.MAX_OBS
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime PIN = Self.PIN
        comptime ZA = Self.ZA
        comptime BB = Self.B

        # gather per-window embeddings (constant across the H steps).
        task_emb.gather[target, BB](task_ids, self.tem, ctx)

        for t in range(Self.H):
            # ein = [obs[t+1] | tem]
            Self._copy_window[target](
                obs, (t + 1) * BB * MO, self.obs_step, BB * MO, ctx
            )
            Self._cat2[target, BB, MO, EMB](
                self.obs_step, self.tem, self.ein, ctx
            )
            enc.forward[target, BB](TensorRefs[1](self.ein), self.nz, ctx)
            # pin = [nz | tem]
            Self._cat2[target, BB, LAT, EMB](self.nz, self.tem, self.pin, ctx)
            policy.forward[target, BB](TensorRefs[1](self.pin), self.pio, ctx)
            self.rsample.forward[target, BB](
                TensorRefs[1](self.pio), self.alp, ctx
            )
            # za = [nz | action | tem]
            self._build_za[target](ctx)
            # Q = min of 2 target heads, two-hot decoded.
            q_a.forward[target, BB](TensorRefs[1](self.za), self.qlog1, ctx)
            self.decode.forward[target, BB](
                TensorRefs[1](self.qlog1), self.qa, ctx
            )
            q_b.forward[target, BB](TensorRefs[1](self.za), self.qlog2, ctx)
            self.decode.forward[target, BB](
                TensorRefs[1](self.qlog2), self.qb, ctx
            )
            self._td_combine[target](reward, done, td_out, t, gamma, ctx)

    @staticmethod
    def _copy_window[target: StaticString](
        mut src: Tensor,
        off: Int,
        mut dst: Tensor,
        n: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "cpu":
            for i in range(n):
                dst.data[i] = src.data[off + i]
        else:
            var c = ctx.value()
            var sub = src.dev.value().create_sub_buffer[DT](off, n)
            c.enqueue_copy(dst.dev.value(), sub)

    @staticmethod
    def _cat2[target: StaticString, BB: Int, AW: Int, BW: Int](
        mut a: Tensor,
        mut b: Tensor,
        mut dst: Tensor,
        ctx: Optional[DeviceContext],
    ) raises:
        """dst[row] = [a[row] (width AW) | b[row] (width BW)] for `BB` rows.
        Staticmethod so the three distinct `self.` field args don't alias `self`
        (exclusivity)."""
        comptime if target == "cpu":
            comptime W = AW + BW
            for row in range(BB):
                for c in range(AW):
                    dst.data[row * W + c] = a.data[row * AW + c]
                for c in range(BW):
                    dst.data[row * W + AW + c] = b.data[row * BW + c]
        else:
            var c = ctx.value()
            comptime nb = (BB * (AW + BW) + TPB - 1) // TPB
            c.enqueue_function[_cat2_k[BB, AW, BW]](
                a.tt["gpu", BB * AW](),
                b.tt["gpu", BB * BW](),
                dst.tt["gpu", BB * (AW + BW)](),
                grid_dim=nb, block_dim=TPB,
            )

    def _build_za[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        comptime LAT = Self.LATENT
        comptime A = Self.MAX_ACT
        comptime EMB = Self.TASK_EMB
        comptime ZA = Self.ZA
        comptime BB = Self.B
        comptime if target == "cpu":
            for b in range(BB):
                for k in range(LAT):
                    self.za.data[b * ZA + k] = self.nz.data[b * LAT + k]
                for k in range(A):
                    self.za.data[b * ZA + LAT + k] = self.alp.data[
                        b * (A + 1) + k
                    ]
                for e in range(EMB):
                    self.za.data[b * ZA + LAT + A + e] = self.tem.data[
                        b * EMB + e
                    ]
        else:
            var c = ctx.value()
            comptime nb = (BB * ZA + TPB - 1) // TPB
            c.enqueue_function[_build_za3_k[BB, LAT, A, A + 1, EMB]](
                self.nz.tt["gpu", BB * LAT](),
                self.alp.tt["gpu", BB * (A + 1)](),
                self.tem.tt["gpu", BB * EMB](),
                self.za.tt["gpu", BB * ZA](),
                grid_dim=nb, block_dim=TPB,
            )

    def _td_combine[target: StaticString](
        mut self,
        mut reward: Tensor,
        mut done: Tensor,
        mut td_out: Tensor,
        t: Int,
        gamma: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        comptime BB = Self.B
        comptime if target == "cpu":
            for b in range(BB):
                var qmin = min(self.qa.data[b], self.qb.data[b])
                var d = done.data[t * BB + b]
                td_out.data[t * BB + b] = reward.data[t * BB + b] + gamma * (
                    Scalar[DT](1.0) - d
                ) * qmin
        else:
            var c = ctx.value()
            comptime nb = (BB + TPB - 1) // TPB
            var rew_sub = reward.dev.value().create_sub_buffer[DT](t * BB, BB)
            var done_sub = done.dev.value().create_sub_buffer[DT](t * BB, BB)
            var td_sub = td_out.dev.value().create_sub_buffer[DT](t * BB, BB)
            c.enqueue_function[_td_combine_k[BB]](
                tile[BB](rew_sub),
                tile[BB](done_sub),
                self.qa.tt["gpu", BB](),
                self.qb.tt["gpu", BB](),
                tile[BB](td_sub),
                gamma, grid_dim=nb, block_dim=TPB,
            )
