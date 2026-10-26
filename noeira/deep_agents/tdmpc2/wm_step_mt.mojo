"""TD-MPC2 multi-task world-model BPTT step (storage; CPU + GPU) — §14.3.

Clone of `wm_step.mojo` adapted for the task-conditioned world model:

  * Encoder input is `MAX_OBS + TASK_EMB` (obs zero-padded to MAX_OBS by the env
    wrapper, then the gathered task embedding concatenated) — built once per step
    into an augmented obs buffer (`oaug`).
  * The WM graph gains a `task_emb` input (set per step) feeding the 3-way `za`
    concat → dynamics/reward/Q/term.
  * Embedding gradient collection (sites 1 + 2 of three): after EACH reverse-scan
    `vjp` the `task_emb` input-slot grad (`grad_input["task_emb"]`) is scatter-
    added into the table by window task; after the encoder backward the encoder
    input-grad's last `TASK_EMB` columns are likewise accumulated. (The policy
    graph is site 3, in `policy_step_mt`.) The table's `zero_grad`/`step` are
    driven by the agent around the whole train step.

Storage migration: all buffers are storage `Tensor`s; the 5 online Q heads are
distinct fields q0..q4 (threaded into one forward/vjp in node order, mirroring
the single-task `WMStep`). `WMLossOut` and the metric/seed/carry GPU kernels are
reused from `wm_step.mojo` (the graph output layout `8 + LATENT` is unchanged).
See `wm_step.mojo` for the forward/reverse scan documentation.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.optimizer.adam import Adam

from .nets_mt import (
    TDMPC2EncoderMT, TDMPC2DynamicsMT, TDMPC2RewardMT, TDMPC2QNetMT,
    TDMPC2TerminationMT,
)
from .wm_graph_mt import TDMPC2WMGraphMT
from .wm_graph import NQ, NLOSS, TERM_COL
from .wm_step import (
    WMLossOut, _extract_carry_k, _seed_wm_k, _accum_metric_k, _copy_slice_k,
)
from .task_embedding import TaskEmbedding
from noeira.core.tiles import Tile, tile


def _build_oaug_k[H1B: Int, MAXOBS: Int, EMB: Int, B: Int](
    obs: Tile[DT, H1B * MAXOBS],
    tem: Tile[DT, B * EMB],
    oaug: Tile[DT, H1B * (MAXOBS + EMB)],
):
    """oaug[row] = [obs[row] | tem[row % B]] — broadcast the per-window task
    embedding across the (H+1) frames (row = t*B + b)."""
    var i = Int(global_idx.x)
    var AOBS = MAXOBS + EMB
    if i < H1B * AOBS:
        var row = i // AOBS
        var c = i % AOBS
        if c < MAXOBS:
            oaug[i] = rebind[Scalar[DT]](obs[row * MAXOBS + c])
        else:
            var b = row % B
            oaug[i] = rebind[Scalar[DT]](tem[b * EMB + (c - MAXOBS)])


struct WMStepMT[
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
    comptime EncT = TDMPC2EncoderMT[
        Self.MAX_OBS, Self.ENC, Self.LATENT, Self.SN, Self.TASK_EMB
    ]
    comptime DynT = TDMPC2DynamicsMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.SN, Self.TASK_EMB
    ]
    comptime RewT = TDMPC2RewardMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB
    ]
    comptime QNetT = TDMPC2QNetMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB, Self.QP
    ]
    comptime TermT = TDMPC2TerminationMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime GraphT = TDMPC2WMGraphMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, Self.TASK_EMB, Self.QP,
    ]
    comptime EmbT = TaskEmbedding[Self.NUM_TASKS, Self.TASK_EMB]
    comptime OUTW = NLOSS + Self.LATENT

    var consistency_coef: Scalar[DT]
    var reward_coef: Scalar[DT]
    var value_coef: Scalar[DT]
    # Termination BCE coefficient (item B). 0.0 → non-episodic.
    var termination_coef: Scalar[DT]
    var rho: Scalar[DT]

    # Persistent scratch Tensors (allocated once in make, reused every step).
    var tem: Tensor      # [B*TASK_EMB] gathered embeddings
    var oaug: Tensor     # [(H+1)*B*AOBS] augmented obs [obs|tem]
    var carry: Tensor    # [(H+1)*B*LATENT]
    var zen: Tensor      # [H*B*LATENT]
    var out_t: Tensor    # [B*OUTW]
    var scratch: Tensor  # [B*OUTW]
    var seed: Tensor     # [B*OUTW]
    var gz: Tensor       # [B*LATENT]
    var gobs: Tensor     # [B*AOBS] encoder input-grad
    var acc: Tensor      # [4] metric accumulator
    # per-step input scratch (one window each).
    var in_z: Tensor     # [B*LATENT]
    var in_a: Tensor     # [B*MAX_ACT]
    var in_zen: Tensor   # [B*LATENT]
    var in_r: Tensor     # [B]
    var in_td: Tensor    # [B]
    var in_done: Tensor  # [B]
    var ein_step: Tensor  # [B*AOBS] encoder input window

    def __init__(out self):
        self.consistency_coef = Scalar[DT](20.0)
        self.reward_coef = Scalar[DT](0.1)
        self.value_coef = Scalar[DT](0.1)
        self.termination_coef = Scalar[DT](0.0)
        self.rho = Scalar[DT](0.5)
        self.tem = Tensor()
        self.oaug = Tensor()
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
        self.ein_step = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        termination_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "WMStepMT: target must be 'cpu' or 'gpu'"
        )
        comptime LAT = Self.LATENT
        comptime OW = Self.OUTW
        comptime BB = Self.B
        comptime AOBS = Self.AOBS
        comptime EMB = Self.TASK_EMB
        var s = Self()
        s.termination_coef = termination_coef
        s.tem = Tensor.make[target](BB * EMB, ctx)
        s.oaug = Tensor.make[target]((Self.H + 1) * BB * AOBS, ctx)
        s.carry = Tensor.make[target]((Self.H + 1) * BB * LAT, ctx)
        s.zen = Tensor.make[target](Self.H * BB * LAT, ctx)
        s.out_t = Tensor.make[target](BB * OW, ctx)
        s.scratch = Tensor.make[target](BB * OW, ctx)
        s.seed = Tensor.make[target](BB * OW, ctx)
        s.gz = Tensor.make[target](BB * LAT, ctx)
        s.gobs = Tensor.make[target](BB * AOBS, ctx)
        s.acc = Tensor.make[target](4, ctx)
        s.in_z = Tensor.make[target](BB * LAT, ctx)
        s.in_a = Tensor.make[target](BB * Self.MAX_ACT, ctx)
        s.in_zen = Tensor.make[target](BB * LAT, ctx)
        s.in_r = Tensor.make[target](BB, ctx)
        s.in_td = Tensor.make[target](BB, ctx)
        s.in_done = Tensor.make[target](BB, ctx)
        s.ein_step = Tensor.make[target](BB * AOBS, ctx)
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
            var sub = src.dev.value().create_sub_buffer[DT](off, n)
            c.enqueue_copy(dst.dev.value(), sub)

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

    def _build_oaug[target: StaticString](
        mut self,
        mut obs: Tensor,   # [(H+1),B,MAX_OBS]
        ctx: Optional[DeviceContext],
    ) raises:
        comptime MO = Self.MAX_OBS
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime BB = Self.B
        comptime H1B = (Self.H + 1) * BB
        comptime if target == "cpu":
            for row in range(H1B):
                var b = row % BB
                for i in range(MO):
                    self.oaug.data[row * AOBS + i] = obs.data[row * MO + i]
                for e in range(EMB):
                    self.oaug.data[row * AOBS + MO + e] = self.tem.data[
                        b * EMB + e
                    ]
        else:
            var c = ctx.value()
            comptime nb = (H1B * AOBS + TPB - 1) // TPB
            c.enqueue_function[_build_oaug_k[H1B, MO, EMB, BB]](
                obs.tt["gpu", H1B * MO](),
                self.tem.tt["gpu", BB * EMB](),
                self.oaug.tt["gpu", H1B * AOBS](),
                grid_dim=nb, block_dim=TPB,
            )

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
        mut task_emb: Self.EmbT,
        mut enc_opt: Adam,
        mut dyn_opt: Adam,
        mut rew_opt: Adam,
        mut qo0: Adam,
        mut qo1: Adam,
        mut qo2: Adam,
        mut qo3: Adam,
        mut qo4: Adam,
        mut term_opt: Adam,
        mut task_ids: Tensor,  # [B] per-window DT ids
        mut obs: Tensor,   # [(H+1),B,MAX_OBS]
        mut act: Tensor,   # [H,B,MAX_ACT]
        mut rew: Tensor,   # [H,B]
        mut td: Tensor,    # [H,B]
        mut done: Tensor,  # [H,B] BCE target
        ctx: Optional[DeviceContext] = None,
    ) raises -> WMLossOut:
        comptime LAT = Self.LATENT
        comptime BB = Self.B
        comptime MO = Self.MAX_OBS
        comptime AOBS = Self.AOBS
        comptime EMB = Self.TASK_EMB

        var inv_b = Scalar[DT](1.0) / Scalar[DT](BB)
        var inv_h = Scalar[DT](1.0) / Scalar[DT](Self.H)
        var inv_lat = Scalar[DT](1.0) / Scalar[DT](LAT)
        var inv_nq = Scalar[DT](1.0) / Scalar[DT](NQ)

        # ── 0. gather embeddings + build augmented obs ────────────────────
        task_emb.gather[target, BB](task_ids, self.tem, ctx)
        self._build_oaug[target](obs, ctx)

        # ── 1. consistency targets enc(oaug[t+1]) → zen[t] (stop-grad) ────
        for t in range(Self.H):
            Self._copy_window[target](
                self.oaug, (t + 1) * BB * AOBS, self.ein_step, BB * AOBS, ctx
            )
            enc.forward[target, BB](
                TensorRefs[1](self.ein_step), self.in_zen, ctx
            )
            Self._copy_window_into[target](
                self.in_zen, self.zen, t * BB * LAT, BB * LAT, ctx
            )

        # ── 2. z_0 = enc(oaug[0]) → carry[0] ──────────────────────────────
        Self._copy_window[target](self.oaug, 0, self.ein_step, BB * AOBS, ctx)
        enc.forward[target, BB](
            TensorRefs[1](self.ein_step), self.in_z, ctx
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
            # site 1: accumulate the task_emb input-slot grad into the table.
            task_emb.accumulate[target, BB, EMB, 0](
                task_ids, graph.grad_input["task_emb"](), ctx
            )
            rho_rev /= self.rho

        # ── 6. encoder backward from t=0 carry grad → gobs [B, AOBS] ───────
        Self._copy_window[target](self.oaug, 0, self.ein_step, BB * AOBS, ctx)
        enc.forward[target, BB](
            TensorRefs[1](self.ein_step), self.in_z, ctx
        )
        enc.vjp[target, BB](
            TensorRefs[1](self.ein_step), self.gz,
            TensorRefs[1](self.gobs), ctx,
        )
        # site 2: accumulate encoder input-grad tail (last TASK_EMB cols).
        task_emb.accumulate[target, BB, AOBS, MO](task_ids, self.gobs, ctx)

        # ── 7. optimizer steps (nets only; table step is the agent's job) ──
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
        comptime A = Self.MAX_ACT
        comptime BB = Self.B
        Self._copy_window[target](self.carry, t * BB * LAT, self.in_z, BB * LAT, ctx)
        Self._copy_window[target](act, t * BB * A, self.in_a, BB * A, ctx)
        Self._copy_window[target](self.zen, t * BB * LAT, self.in_zen, BB * LAT, ctx)
        Self._copy_window[target](rew, t * BB, self.in_r, BB, ctx)
        Self._copy_window[target](td, t * BB, self.in_td, BB, ctx)
        Self._copy_window[target](done, t * BB, self.in_done, BB, ctx)
        graph.set_input["z", BB](self.in_z, ctx)
        graph.set_input["a", BB](self.in_a, ctx)
        graph.set_input["task_emb", BB](self.tem, ctx)
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
