"""TD-MPC2 TD-target step — stop-grad value targets for the WM value loss.

Reference `_td_target` + the `encode(obs[1:])` in `_update`
(`references/tdmpc2-main/tdmpc2/tdmpc2.py:242-264`):

    next_z = encode(obs[t+1])                      # stop-grad
    a ~ π(next_z)                                  # stochastic sample
    Q = min over 2 random TARGET-Q heads, two-hot decoded to scalar
    td[t] = reward[t] + γ·(1 − done[t])·Q

Forward-only (no autograd): calls the encoder, policy, RSample, the two
target-Q heads, and TwoHotDecode directly. Storage migration: inputs are
`Tensor`s; the two target-Q heads are passed as DISTINCT `mut q_a, mut q_b`
fields (the agent's comptime dispatch picks the random pair). Output `td`
[H, B] feeds `WMStep` as the stop-grad `td` input.

CPU + GPU.
"""

from std.math import min
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Zero
from noeira.nn.primitives.rsample import RSample

from .nets import TDMPC2Encoder, TDMPC2Policy, TDMPC2QNet
from .losses import TwoHotDecode
from noeira.core.tiles import Tile, tile


def _build_za_k[B_: Int, LAT_: Int, A_: Int, ALP_: Int](
    nz: Tile[DT, B_ * LAT_],
    alp: Tile[DT, B_ * ALP_],
    za: Tile[DT, B_ * (LAT_ + A_)],
):
    """za[b] = [nz[b] | action[b]] where action = alp[b, :A_]."""
    var i = Int(global_idx.x)
    var ZA = LAT_ + A_
    if i < B_ * ZA:
        var b = i // ZA
        var k = i % ZA
        if k < LAT_:
            za[i] = rebind[Scalar[DT]](nz[b * LAT_ + k])
        else:
            za[i] = rebind[Scalar[DT]](alp[b * ALP_ + (k - LAT_)])


def _td_combine_k[B_: Int](
    rew: Tile[DT, B_],
    done: Tile[DT, B_],
    qa: Tile[DT, B_],
    qb: Tile[DT, B_],
    td_out: Tile[DT, B_],
    gamma: Scalar[DT],
):
    """td[b] = rew[b] + gamma·(1−done[b])·min(qa,qb)."""
    var b = Int(global_idx.x)
    if b < B_:
        var a = rebind[Scalar[DT]](qa[b])
        var bb = rebind[Scalar[DT]](qb[b])
        var qm = a if a < bb else bb
        td_out[b] = rebind[Scalar[DT]](rew[b]) + gamma * (
            Scalar[DT](1.0) - rebind[Scalar[DT]](done[b])
        ) * qm


struct TDTargetStep[
    OBS: Int,
    ENC: Int,
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
    QP: Float64 = 0.0,
](Movable & Deinitable):
    comptime EncT = TDMPC2Encoder[Self.OBS, Self.ENC, Self.LATENT, Self.SN]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]

    var rsample: RSample[Self.ACT]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]

    # Persistent scratch Tensors (allocated once in make, reused every step).
    var nz: Tensor
    var pio: Tensor
    var alp: Tensor
    var za: Tensor
    var qlog1: Tensor
    var qlog2: Tensor
    var qa: Tensor
    var qb: Tensor
    var obs_step: Tensor  # [B*OBS] encoder input window

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self.decode = TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]()
        self.nz = Tensor()
        self.pio = Tensor()
        self.alp = Tensor()
        self.za = Tensor()
        self.qlog1 = Tensor()
        self.qlog2 = Tensor()
        self.qa = Tensor()
        self.qb = Tensor()
        self.obs_step = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TDTargetStep: target must be 'cpu' or 'gpu'"
        )
        comptime LAT = Self.LATENT
        comptime A = Self.ACT
        comptime ZA = LAT + A
        comptime BB = Self.B
        var s = Self()
        s.rsample = RSample[Self.ACT].make[target, INIT=Zero](ctx=ctx)
        s.decode = TwoHotDecode[
            Self.BINS, Self.VMIN, Self.VMAX
        ].make[target, INIT=Zero](ctx=ctx)
        s.nz = Tensor.make[target](BB * LAT, ctx)
        s.pio = Tensor.make[target](BB * 2 * A, ctx)
        s.alp = Tensor.make[target](BB * (A + 1), ctx)
        s.za = Tensor.make[target](BB * ZA, ctx)
        s.qlog1 = Tensor.make[target](BB * Self.BINS, ctx)
        s.qlog2 = Tensor.make[target](BB * Self.BINS, ctx)
        s.qa = Tensor.make[target](BB, ctx)
        s.qb = Tensor.make[target](BB, ctx)
        s.obs_step = Tensor.make[target](BB * Self.OBS, ctx)
        return s^

    def step[target: StaticString](
        mut self,
        mut enc: Self.EncT,
        mut policy: Self.PolicyT,
        mut q_a: Self.QNetT,
        mut q_b: Self.QNetT,
        mut obs: Tensor,     # [(H+1),B,OBS]
        mut reward: Tensor,  # [H,B]
        mut done: Tensor,    # [H,B]
        mut td_out: Tensor,  # [H,B] (written)
        gamma: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime LAT = Self.LATENT
        comptime A = Self.ACT
        comptime ZA = LAT + A
        comptime BB = Self.B

        for t in range(Self.H):
            # next_z = encode(obs[t+1])  (stop-grad)
            Self._copy_window[target](
                obs, (t + 1) * BB * Self.OBS, self.obs_step, BB * Self.OBS, ctx
            )
            enc.forward[target, BB](TensorRefs[1](self.obs_step), self.nz, ctx)
            # a ~ π(next_z)
            policy.forward[target, BB](TensorRefs[1](self.nz), self.pio, ctx)
            self.rsample.forward[target, BB](
                TensorRefs[1](self.pio), self.alp, ctx
            )
            # za = [next_z | action]
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

    def _build_za[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        comptime LAT = Self.LATENT
        comptime A = Self.ACT
        comptime ZA = LAT + A
        comptime BB = Self.B
        comptime if target == "cpu":
            for b in range(BB):
                for k in range(LAT):
                    self.za.data[b * ZA + k] = self.nz.data[b * LAT + k]
                for k in range(A):
                    self.za.data[b * ZA + LAT + k] = self.alp.data[b * (A + 1) + k]
        else:
            var c = ctx.value()
            comptime nb = (BB * ZA + TPB - 1) // TPB
            c.enqueue_function[_build_za_k[BB, LAT, A, A + 1]](
                self.nz.tt["gpu", BB * LAT](),
                self.alp.tt["gpu", BB * (A + 1)](),
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
