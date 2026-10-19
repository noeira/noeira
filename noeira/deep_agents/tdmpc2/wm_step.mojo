"""TD-MPC2 world-model BPTT step (storage framework; CPU + GPU).

Mirrors DreamerV3's `WMStep` carry-passthrough BPTT and the validated
`tests/nn/spike_wm_bptt.mojo` scan. The world model's dynamics, reward, and
5 Q heads are **trainer-owned standalone Modules** referenced by the WM
ComputeGraph as `ExternalNode`s (so the policy graph + MPPI planner can call
the same instances). This block binds them, runs the forward scan, reverse
BPTT scan, encoder backward, and steps one Adam per module.

  z_0 = encode(obs[0])
  for t in 0..H-1:
      out_t = WMGraph(z=carry_t, a=a_t, z_enc_next=sg·encode(obs[t+1]),
                      r=r_t, td=td_t)            # dyn/rew/Q threaded externals
      carry_{t+1} = out_t[:, 8:]                 # znext (dynamics output)
  reverse scan seeds loss cols (coef·ρ^t/norm) + znext cols (carry grad),
  vjp accumulates into the external modules, grad_input["z"] threads back.

Storage migration (Stage 5): inputs are storage `Tensor`s (`.data` host /
`.dev` device). The 5 online Q heads are passed as DISTINCT fields q0..q4
(storage threads externals into one `forward`/`vjp` call in node order;
two `mut` subscripts of one List can't alias). The legacy `set_external` /
`graph.forward(out_only)` / `node_out_ptr` / `grad_input_ptr` are replaced by
threaded externals + `node_output` / `grad_input`.

Inputs (t-major, contiguous per step):
  obs [(H+1),B,OBS], act [H,B,ACT], r [H,B], td [H,B] (stop-grad targets).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.optimizer.adam import Adam

from .nets import (
    TDMPC2Encoder, TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Termination,
)
from .wm_graph import TDMPC2WMGraph, NQ, NLOSS, TERM_COL
from noeira.core.tiles import Tile, tile


@fieldwise_init
struct WMLossOut(Copyable):
    """Per-component world-model losses (already coef·ρ^t/norm weighted).
    `termination` is the BCE head loss (item B); 0 unless bce_coef > 0."""
    var consistency: Scalar[DT]
    var reward: Scalar[DT]
    var value: Scalar[DT]
    var termination: Scalar[DT]

    @always_inline
    def total(self) -> Scalar[DT]:
        return self.consistency + self.reward + self.value + self.termination


# ── GPU kernels (operate over storage Tensor `.lt` views) ───────────────
def _copy_slice_k[N: Int](
    src: Tile[DT, N],
    dst: Tile[DT, N],
):
    """dst[i] = src[i] over a contiguous N-window (sub-buffer views)."""
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i])


def _extract_carry_k[B_: Int, LAT_: Int, OW_: Int](
    ob: Tile[DT, B_ * OW_],
    carry_next: Tile[DT, B_ * LAT_],
):
    """carry_next[b,k] = out[b, NLOSS+k]  (znext passthrough columns)."""
    var i = Int(global_idx.x)
    if i < B_ * LAT_:
        var b = i // LAT_
        var k = i % LAT_
        carry_next[i] = rebind[Scalar[DT]](ob[b * OW_ + NLOSS + k])


def _seed_wm_k[B_: Int, OW_: Int, LAT_: Int, NQ_: Int](
    seed: Tile[DT, B_ * OW_],
    gz: Tile[DT, B_ * LAT_],
    sc_cons: Scalar[DT],
    sc_rew: Scalar[DT],
    sc_val: Scalar[DT],
    sc_term: Scalar[DT],
):
    """seed[b] = [sc_cons, sc_rew, sc_val×NQ, sc_term, gz[b]]."""
    var b = Int(global_idx.x)
    if b < B_:
        seed[b * OW_ + 0] = sc_cons
        seed[b * OW_ + 1] = sc_rew
        for q in range(NQ_):
            seed[b * OW_ + 2 + q] = sc_val
        seed[b * OW_ + TERM_COL] = sc_term
        for k in range(LAT_):
            seed[b * OW_ + NLOSS + k] = rebind[Scalar[DT]](gz[b * LAT_ + k])


def _accum_metric_k[B_: Int, OW_: Int, NQ_: Int](
    ob: Tile[DT, B_ * OW_],
    acc: Tile[DT, 4],
    sc_cons: Scalar[DT],
    sc_rew: Scalar[DT],
    sc_val: Scalar[DT],
    sc_term: Scalar[DT],
):
    """acc += [Σ_b sc_cons·cons, Σ_b sc_rew·rew, Σ_b sc_val·Σ_q v_q,
    Σ_b sc_term·tloss]. One thread."""
    var t = Int(global_idx.x)
    if t == 0:
        var sc: Scalar[DT] = 0.0
        var sr: Scalar[DT] = 0.0
        var sv: Scalar[DT] = 0.0
        var st: Scalar[DT] = 0.0
        for b in range(B_):
            sc += sc_cons * rebind[Scalar[DT]](ob[b * OW_ + 0])
            sr += sc_rew * rebind[Scalar[DT]](ob[b * OW_ + 1])
            var v: Scalar[DT] = 0.0
            for q in range(NQ_):
                v += rebind[Scalar[DT]](ob[b * OW_ + 2 + q])
            sv += sc_val * v
            st += sc_term * rebind[Scalar[DT]](ob[b * OW_ + TERM_COL])
        acc[0] = rebind[Scalar[DT]](acc[0]) + sc
        acc[1] = rebind[Scalar[DT]](acc[1]) + sr
        acc[2] = rebind[Scalar[DT]](acc[2]) + sv
        acc[3] = rebind[Scalar[DT]](acc[3]) + st


struct WMStep[
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
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime TermT = TDMPC2Termination[Self.LATENT, Self.ACT, Self.MLP]
    comptime GraphT = TDMPC2WMGraph[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, Self.QP,
    ]
    comptime OUTW = NLOSS + Self.LATENT

    var consistency_coef: Scalar[DT]
    var reward_coef: Scalar[DT]
    var value_coef: Scalar[DT]
    # Termination BCE coefficient (item B). 0.0 → non-episodic: the term head
    # gets zero gradient (Adam no-op) → other nets bit-identical. Reference
    # `termination_coef` (e.g. 1.0) for episodic envs.
    var termination_coef: Scalar[DT]
    var rho: Scalar[DT]

    # Persistent scratch Tensors (allocated once in make). On CPU they back the
    # host `.data`; on GPU they back the device buffers. Reused across steps to
    # avoid per-step (re)allocation.
    var carry: Tensor    # [(H+1)*B*LATENT]
    var zen: Tensor      # [H*B*LATENT]
    var out_t: Tensor    # [B*OUTW]
    var scratch: Tensor  # [B*OUTW]
    var seed: Tensor     # [B*OUTW]
    var gz: Tensor       # [B*LATENT]
    var gobs: Tensor     # [B*OBS]
    var acc: Tensor      # [4] metric accumulator
    # per-step input scratch (one window each, copied from the full input).
    var in_z: Tensor     # [B*LATENT]
    var in_a: Tensor     # [B*ACT]
    var in_zen: Tensor   # [B*LATENT]
    var in_r: Tensor     # [B]
    var in_td: Tensor    # [B]
    var in_done: Tensor  # [B]
    var obs_step: Tensor  # [B*OBS] encoder input window

    def __init__(out self):
        self.consistency_coef = Scalar[DT](20.0)
        self.reward_coef = Scalar[DT](0.1)
        self.value_coef = Scalar[DT](0.1)
        self.termination_coef = Scalar[DT](0.0)
        self.rho = Scalar[DT](0.5)
        self.carry = Tensor()
        self.zen = Tensor()
        self.out_t = Tensor()
        self.scratch = Tensor()
        self.seed = Tensor()
        self.gz = Tensor()
        self.gobs = Tensor()
        self.acc = Tensor()
        self.in_z = Tensor()
        self.in_a = Tensor()
        self.in_zen = Tensor()
        self.in_r = Tensor()
        self.in_td = Tensor()
        self.in_done = Tensor()
        self.obs_step = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        termination_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "WMStep: target must be 'cpu' or 'gpu'"
        )
        comptime LAT = Self.LATENT
        comptime OW = Self.OUTW
        comptime BB = Self.B
        var s = Self()
        s.termination_coef = termination_coef
        s.carry = Tensor.make[target]((Self.H + 1) * BB * LAT, ctx)
        s.zen = Tensor.make[target](Self.H * BB * LAT, ctx)
        s.out_t = Tensor.make[target](BB * OW, ctx)
        s.scratch = Tensor.make[target](BB * OW, ctx)
        s.seed = Tensor.make[target](BB * OW, ctx)
        s.gz = Tensor.make[target](BB * LAT, ctx)
        s.gobs = Tensor.make[target](BB * Self.OBS, ctx)
        s.acc = Tensor.make[target](4, ctx)
        s.in_z = Tensor.make[target](BB * LAT, ctx)
        s.in_a = Tensor.make[target](BB * Self.ACT, ctx)
        s.in_zen = Tensor.make[target](BB * LAT, ctx)
        s.in_r = Tensor.make[target](BB, ctx)
        s.in_td = Tensor.make[target](BB, ctx)
        s.in_done = Tensor.make[target](BB, ctx)
        s.obs_step = Tensor.make[target](BB * Self.OBS, ctx)
        return s^

    # ── copy a contiguous [n]-window from src (offset off) into dst[0:n] ──
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
            # device-to-device copy of the window via a sub-buffer view.
            var sub = src.dev.value().create_sub_buffer[DT](off, n)
            c.enqueue_copy(dst.dev.value(), sub)

    def step[target: StaticString](
        mut self,
        mut graph: Self.GraphT,
        mut enc: Self.EncT,
        mut dyn: Self.DynT,
        mut rew_net: Self.RewT,
        mut q0: Self.QNetT,
        mut q1: Self.QNetT,
        mut q2: Self.QNetT,
        mut q3: Self.QNetT,
        mut q4: Self.QNetT,
        mut term_net: Self.TermT,
        mut enc_opt: Adam,
        mut dyn_opt: Adam,
        mut rew_opt: Adam,
        mut qo0: Adam,
        mut qo1: Adam,
        mut qo2: Adam,
        mut qo3: Adam,
        mut qo4: Adam,
        mut term_opt: Adam,
        mut obs: Tensor,   # [(H+1),B,OBS]
        mut act: Tensor,   # [H,B,ACT]
        mut rew: Tensor,   # [H,B]
        mut td: Tensor,    # [H,B]
        mut done: Tensor,  # [H,B] BCE target
        ctx: Optional[DeviceContext] = None,
    ) raises -> WMLossOut:
        comptime LAT = Self.LATENT
        comptime OW = Self.OUTW
        comptime BB = Self.B

        var inv_b = Scalar[DT](1.0) / Scalar[DT](BB)
        var inv_h = Scalar[DT](1.0) / Scalar[DT](Self.H)
        var inv_lat = Scalar[DT](1.0) / Scalar[DT](LAT)
        var inv_nq = Scalar[DT](1.0) / Scalar[DT](NQ)

        # ── 1. consistency targets enc(obs[t+1]) → zen[t] (stop-grad) ─────
        for t in range(Self.H):
            Self._copy_window[target](
                obs, (t + 1) * BB * Self.OBS, self.obs_step, BB * Self.OBS, ctx
            )
            enc.forward[target, BB](
                TensorRefs[1](self.obs_step), self.in_zen, ctx
            )
            # in_zen → zen[t] window
            Self._copy_window_into[target](
                self.in_zen, self.zen, t * BB * LAT, BB * LAT, ctx
            )

        # ── 2. z_0 = enc(obs[0]) → carry[0] ───────────────────────────────
        Self._copy_window[target](obs, 0, self.obs_step, BB * Self.OBS, ctx)
        enc.forward[target, BB](
            TensorRefs[1](self.obs_step), self.in_z, ctx
        )
        Self._copy_window_into[target](self.in_z, self.carry, 0, BB * LAT, ctx)

        # ── 3. forward scan: roll latents, accumulate weighted metric ──────
        comptime if target == "gpu":
            self.acc.dev.value().enqueue_fill(Scalar[DT](0))
        else:
            for i in range(4):
                self.acc.data[i] = 0

        var rho_t = Scalar[DT](1.0)
        for t in range(Self.H):
            self._seed_step[target](graph, t, act, rew, td, done, ctx)
            graph.forward[BB, target](
                self.out_t, ctx, dyn, rew_net, q0, q1, q2, q3, q4, term_net
            )
            # carry[t+1] = out[:, NLOSS:]
            self._extract_carry[target]((t + 1) * BB * LAT, ctx)
            self._accum_metric[target](
                self.consistency_coef * rho_t * inv_b * inv_lat * inv_h,
                self.reward_coef * rho_t * inv_b * inv_h,
                self.value_coef * rho_t * inv_b * inv_nq * inv_h,
                self.termination_coef * rho_t * inv_b * inv_h,
                ctx,
            )
            rho_t *= self.rho

        # ── 4. zero grads + gz ─────────────────────────────────────────────
        enc_opt.zero_grad[target, M=Self.EncT](enc, ctx)
        dyn_opt.zero_grad[target, M=Self.DynT](dyn, ctx)
        rew_opt.zero_grad[target, M=Self.RewT](rew_net, ctx)
        qo0.zero_grad[target, M=Self.QNetT](q0, ctx)
        qo1.zero_grad[target, M=Self.QNetT](q1, ctx)
        qo2.zero_grad[target, M=Self.QNetT](q2, ctx)
        qo3.zero_grad[target, M=Self.QNetT](q3, ctx)
        qo4.zero_grad[target, M=Self.QNetT](q4, ctx)
        term_opt.zero_grad[target, M=Self.TermT](term_net, ctx)
        comptime if target == "gpu":
            self.gz.dev.value().enqueue_fill(Scalar[DT](0))
        else:
            for i in range(BB * LAT):
                self.gz.data[i] = 0

        # ── 5. reverse-scan BPTT ───────────────────────────────────────────
        var rho_rev = Scalar[DT](1.0)
        for _ in range(Self.H - 1):
            rho_rev *= self.rho
        for rev in range(Self.H):
            var t = Self.H - 1 - rev
            self._seed_step[target](graph, t, act, rew, td, done, ctx)
            graph.forward[BB, target](
                self.scratch, ctx, dyn, rew_net, q0, q1, q2, q3, q4, term_net
            )
            self._build_seed[target](
                self.consistency_coef * rho_rev * inv_b * inv_lat * inv_h,
                self.reward_coef * rho_rev * inv_b * inv_h,
                self.value_coef * rho_rev * inv_b * inv_nq * inv_h,
                self.termination_coef * rho_rev * inv_b * inv_h,
                ctx,
            )
            graph.vjp[BB, target](
                self.seed, ctx, dyn, rew_net, q0, q1, q2, q3, q4, term_net
            )
            # gz = grad_input["z"]
            self._copy_grad_z[target](graph, ctx)
            rho_rev /= self.rho

        # ── 6. encoder backward from t=0 carry grad ────────────────────────
        Self._copy_window[target](obs, 0, self.obs_step, BB * Self.OBS, ctx)
        enc.forward[target, BB](
            TensorRefs[1](self.obs_step), self.in_z, ctx
        )
        enc.vjp[target, BB](
            TensorRefs[1](self.obs_step), self.gz, TensorRefs[1](self.gobs), ctx
        )

        # ── 7. optimizer steps ─────────────────────────────────────────────
        enc_opt.step[target, M=Self.EncT](enc, ctx)
        dyn_opt.step[target, M=Self.DynT](dyn, ctx)
        rew_opt.step[target, M=Self.RewT](rew_net, ctx)
        qo0.step[target, M=Self.QNetT](q0, ctx)
        qo1.step[target, M=Self.QNetT](q1, ctx)
        qo2.step[target, M=Self.QNetT](q2, ctx)
        qo3.step[target, M=Self.QNetT](q3, ctx)
        qo4.step[target, M=Self.QNetT](q4, ctx)
        term_opt.step[target, M=Self.TermT](term_net, ctx)

        # ── read the metric accumulator ────────────────────────────────────
        comptime if target == "gpu":
            self.acc.download(ctx.value())
        return WMLossOut(
            self.acc.data[0], self.acc.data[1], self.acc.data[2],
            self.acc.data[3],
        )

    # ── copy in_t[0:n] into dst at offset off ──────────────────────────────
    @staticmethod
    def _copy_window_into[target: StaticString](
        mut src: Tensor,
        mut dst: Tensor,
        off: Int,
        n: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "cpu":
            for i in range(n):
                dst.data[off + i] = src.data[i]
        else:
            var c = ctx.value()
            var sub = dst.dev.value().create_sub_buffer[DT](off, n)
            c.enqueue_copy(sub, src.dev.value())

    def _seed_step[target: StaticString](
        mut self,
        mut graph: Self.GraphT,
        t: Int,
        mut act: Tensor,
        mut rew: Tensor,
        mut td: Tensor,
        mut done: Tensor,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime LAT = Self.LATENT
        comptime BB = Self.B
        # z = carry[t]
        Self._copy_window[target](self.carry, t * BB * LAT, self.in_z, BB * LAT, ctx)
        Self._copy_window[target](act, t * BB * Self.ACT, self.in_a, BB * Self.ACT, ctx)
        Self._copy_window[target](self.zen, t * BB * LAT, self.in_zen, BB * LAT, ctx)
        Self._copy_window[target](rew, t * BB, self.in_r, BB, ctx)
        Self._copy_window[target](td, t * BB, self.in_td, BB, ctx)
        Self._copy_window[target](done, t * BB, self.in_done, BB, ctx)
        graph.set_input["z", BB](self.in_z, ctx)
        graph.set_input["a", BB](self.in_a, ctx)
        graph.set_input["z_enc_next", BB](self.in_zen, ctx)
        graph.set_input["r", BB](self.in_r, ctx)
        graph.set_input["td", BB](self.in_td, ctx)
        graph.set_input["done", BB](self.in_done, ctx)

    def _extract_carry[target: StaticString](
        mut self, off: Int, ctx: Optional[DeviceContext]
    ) raises:
        comptime LAT = Self.LATENT
        comptime OW = Self.OUTW
        comptime BB = Self.B
        comptime if target == "cpu":
            for b in range(BB):
                for k in range(LAT):
                    self.carry.data[off + b * LAT + k] = self.out_t.data[
                        b * OW + NLOSS + k
                    ]
        else:
            var c = ctx.value()
            comptime nb = (BB * LAT + TPB - 1) // TPB
            var carry_sub = self.carry.dev.value().create_sub_buffer[DT](
                off, BB * LAT
            )
            c.enqueue_function[_extract_carry_k[BB, LAT, OW]](
                self.out_t.tt["gpu", BB * OW](),
                tile[BB * LAT](carry_sub),
                grid_dim=nb, block_dim=TPB,
            )

    def _accum_metric[target: StaticString](
        mut self,
        sc_cons: Scalar[DT],
        sc_rew: Scalar[DT],
        sc_val: Scalar[DT],
        sc_term: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        comptime OW = Self.OUTW
        comptime BB = Self.B
        comptime if target == "cpu":
            var sc: Scalar[DT] = 0.0
            var sr: Scalar[DT] = 0.0
            var sv: Scalar[DT] = 0.0
            var st: Scalar[DT] = 0.0
            for b in range(BB):
                sc += sc_cons * self.out_t.data[b * OW + 0]
                sr += sc_rew * self.out_t.data[b * OW + 1]
                var v: Scalar[DT] = 0.0
                for q in range(NQ):
                    v += self.out_t.data[b * OW + 2 + q]
                sv += sc_val * v
                st += sc_term * self.out_t.data[b * OW + TERM_COL]
            self.acc.data[0] += sc
            self.acc.data[1] += sr
            self.acc.data[2] += sv
            self.acc.data[3] += st
        else:
            var c = ctx.value()
            c.enqueue_function[_accum_metric_k[BB, OW, NQ]](
                self.out_t.tt["gpu", BB * OW](),
                self.acc.tt["gpu", 4](),
                sc_cons, sc_rew, sc_val, sc_term,
                grid_dim=1, block_dim=1,
            )

    def _build_seed[target: StaticString](
        mut self,
        sc_cons: Scalar[DT],
        sc_rew: Scalar[DT],
        sc_val: Scalar[DT],
        sc_term: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        comptime LAT = Self.LATENT
        comptime OW = Self.OUTW
        comptime BB = Self.B
        comptime if target == "cpu":
            for b in range(BB):
                self.seed.data[b * OW + 0] = sc_cons
                self.seed.data[b * OW + 1] = sc_rew
                for q in range(NQ):
                    self.seed.data[b * OW + 2 + q] = sc_val
                self.seed.data[b * OW + TERM_COL] = sc_term
                for k in range(LAT):
                    self.seed.data[b * OW + NLOSS + k] = self.gz.data[b * LAT + k]
        else:
            var c = ctx.value()
            comptime nb = (BB + TPB - 1) // TPB
            c.enqueue_function[_seed_wm_k[BB, OW, LAT, NQ]](
                self.seed.tt["gpu", BB * OW](),
                self.gz.tt["gpu", BB * LAT](),
                sc_cons, sc_rew, sc_val, sc_term,
                grid_dim=nb, block_dim=TPB,
            )

    def _copy_grad_z[target: StaticString](
        mut self, mut graph: Self.GraphT, ctx: Optional[DeviceContext]
    ) raises:
        comptime LAT = Self.LATENT
        comptime BB = Self.B
        comptime if target == "cpu":
            ref gzin = graph.grad_input["z"]()
            for i in range(BB * LAT):
                self.gz.data[i] = gzin.data[i]
        else:
            var c = ctx.value()
            comptime nb = (BB * LAT + TPB - 1) // TPB
            c.enqueue_function[_copy_slice_k[BB * LAT]](
                graph.grad_input["z"]().tt["gpu", BB * LAT](),
                self.gz.tt["gpu", BB * LAT](),
                grid_dim=nb, block_dim=TPB,
            )
