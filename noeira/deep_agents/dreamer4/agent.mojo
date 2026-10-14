"""Dreamer4Agent — behavior-cloning agent facade (Phase 3.7).

Wires the four trained components into one BC training step:

  TaskEmbedder ──embed_into──▶ agent_in (task token per sequence, broadcast)
        │                          │
        │                          ▼
        │      Dreamer4Dynamics.set_agent_in → shortcut-forcing forwards
        │      (video-prediction loss on noisy reps) → main-pass h_t
        │                          │
        │                          ▼
        │              MTP policy + reward heads → BC loss (eq. 9) → grad_h
        │                          │
        ▼                          ▼
  accumulate_grad ◀── dyn.grad_agent_in ◀── dyn.vjp(grad_zhat, set_grad_h)

The video-prediction (shortcut-forcing) loss and the BC loss are trained
jointly: the MAIN forward of the shortcut loss produces h_t (agent tokens are
injected into every pass), the BC loss backprops into h_t, and `dyn.vjp`
carries BOTH the video grad (spatial flow columns) and the BC grad (agent
columns) through the transformer in one pass — then the agent-input grad feeds
the TaskEmbedder.

`Dreamer4Agent` is itself a (degenerate) `Module`: its forward/vjp are unused
(the real entry point is `bc_train_step`), but conforming lets a SINGLE Adam
cover every parameter (dynamics + task embedder + both heads) through the
delegated `for_each_param`/`zero_grad`.

v1 uses an UNCONDITIONAL dynamics (ADIM=0) — the action-conditioned video
prediction (ADIM>0, already built) layers in for the real-env lighthouse.
"""

from max.gpu.host import DeviceContext, HostBuffer

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import ParamVisitor, ParamVisitorRef
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.core.walkers import join_name
from noeira.nn.core.checkpoint import (
    BinaryCheckpointWriter, BinaryCheckpointReader,
    _read_file_bytes,
)

from .dynamics import Dreamer4Dynamics
from .task_embedder import TaskEmbedder
from .heads import (
    Dreamer4PolicyHead, Dreamer4RewardHead, Dreamer4ValueHead,
    Dreamer4ContinueHead,
)
from .shortcut_loss import dynamics_pretrain_loss, _mao
from .bc_loss import bc_mtp_loss, bc_policy_only_loss
from .imag_rollout import imagine_rollout, _fwd_window
from ..dreamerv3.dists_discrete import cat_sample, UNIMIX
from .imag_rl_loss import (
    lambda_returns,
    value_td_loss_cpu,
    value_td_loss_backward,
    pmpo_policy_loss_cpu,
    pmpo_policy_loss_backward,
    continue_pred,
)
from ..dreamerv3.polyak import polyak_module


def _ilog2(n: Int) -> Int:
    var k = 0
    var v = n
    while v > 1:
        v //= 2
        k += 1
    return k


struct Dreamer4Agent[
    DSP: Int, NSP: Int, D: Int, NH: Int, T: Int, NREG: Int, HID: Int,
    DEPTH: Int, KMAX: Int,            # dynamics backbone
    NAGENT: Int, NTASK: Int,          # agent tokens + task table
    HHID: Int, NACT: Int, NBINS: Int, NMTP: Int,   # heads
    B: Int, B_SELF: Int,              # sequences per batch + self rows
    USE_MAX: Bool = True,
    ADIM: Int = 0,                    # action dim (0 ⇒ unconditional; Phase-4
    AHID: Int = 0,                    #  imagination needs ADIM = NACT one-hot)
    K_IMAG: Int = 0,                  # ODE steps for imagination (0 ⇒ KMAX)
    NCTX: Int = 1,                    # clean context frames for the rollout
    DYN_TARGET: StaticString = "cpu", # "gpu" ⇒ dynamics on device (the heavy
                                      #  imagination compute); heads stay CPU
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=1)
    comptime OUT_DIM = 1

    comptime AGD: Int = Self.NAGENT * Self.D          # agent-token width = h_t dim
    comptime ND: Int = Self.NSP * Self.DSP            # packed latent width
    comptime BF: Int = Self.B * Self.T                # nn batch (B·T)
    comptime PLOG: Int = Self.NMTP * Self.NACT
    comptime RLOG: Int = Self.NMTP * Self.NBINS
    comptime EMAX: Int = _ilog2(Self.KMAX)            # clean-BC step index
    comptime KI: Int = Self.K_IMAG if Self.K_IMAG > 0 else Self.KMAX
    comptime TM1: Int = Self.T - 1                    # imagined states with a return

    comptime DYN = Dreamer4Dynamics[
        Self.DSP, Self.NSP, Self.D, Self.NH, Self.T, Self.NREG, Self.HID,
        Self.DEPTH, Self.KMAX, Self.USE_MAX, Self.ADIM, Self.AHID, Self.NAGENT,
    ]
    comptime TE = TaskEmbedder[Self.D, Self.NTASK, Self.NAGENT]
    comptime PH = Dreamer4PolicyHead[Self.AGD, Self.HHID, Self.NACT, Self.NMTP]
    comptime RH = Dreamer4RewardHead[Self.AGD, Self.HHID, Self.NBINS, Self.NMTP]
    comptime VH = Dreamer4ValueHead[Self.AGD, Self.HHID, Self.NBINS]
    comptime CH = Dreamer4ContinueHead[Self.AGD, Self.HHID]

    var dyn: Self.DYN
    var te: Self.TE
    var ph: Self.PH
    var rh: Self.RH
    var vh: Self.VH                  # value head (Phase 4; untrained during BC)
    var ph_prior: Self.PH           # frozen behavioral prior (PMPO reverse-KL)
    var ch: Self.CH                 # continue/termination head (opt-in via
                                    #  use_continue; frozen during imagination)

    # CPU scratch (owned; sized at make)
    var agent_in: List[Scalar[DT]]      # [BF, AGD]
    var grad_zhat: List[Scalar[DT]]     # [BF, ND]
    var zhat: List[Scalar[DT]]          # [BF, ND]
    var ztil: List[Scalar[DT]]          # [BF, ND] main-pass input (= vjp fwd_in)
    var grad_zt: List[Scalar[DT]]       # [BF, ND] (grad wrt z̃; discarded)
    var grad_h: List[Scalar[DT]]        # [BF, AGD]
    var grad_h_tmp: List[Scalar[DT]]    # [BF, AGD]
    var plog: List[Scalar[DT]]          # [BF, PLOG]
    var rlog: List[Scalar[DT]]          # [BF, RLOG]
    var gpl: List[Scalar[DT]]           # [BF, PLOG]
    var grl: List[Scalar[DT]]           # [BF, RLOG]
    # clean-BC scratch: a dedicated near-clean forward at the HIGHEST TRAINED
    # signal level (σ=(KMAX-1)/KMAX, sig=KMAX-1) gives h_t a near-un-noised
    # frame so the policy can read the observation. σ=1 (sig=KMAX) is avoided:
    # the video loss never samples it, so that embedding row is untrained ⇒ OOD.
    var clean_sig: List[Scalar[DT]]     # [BF] = KMAX-1 (highest trained σ)
    var clean_step: List[Scalar[DT]]    # [BF] = EMAX
    var bc_in: List[Scalar[DT]]         # [BF, ND] noised BC input at σ_bc
    var gzero: List[Scalar[DT]]         # [BF, ND] zero flow-grad for the BC vjp
    # ── action-conditioned WM scratch (acwm_train_step) ─────────────────
    var ac_tok: List[Scalar[DT]]        # [BF, ADIM] shifted action one-hot tokens
    var ac_mask: List[Scalar[DT]]       # [ADIM] all-ones (no masking)
    var rew_shift: List[Scalar[DT]]     # [BF] transition-into reward (r[f-1])
    # ── imagination-RL scratch (Phase 4) ────────────────────────────────
    var im_h: List[Scalar[DT]]          # [BF, AGD]  rollout agent tokens
    var im_act: List[Scalar[DT]]        # [B, T-1]   sampled action class
    var im_rew: List[Scalar[DT]]        # [BF]       reward-head pred per state
    var im_val: List[Scalar[DT]]        # [BF]       value-head pred per state
    var im_con: List[Scalar[DT]]        # [BF]       continue (= γ)
    var im_ret: List[Scalar[DT]]        # [B, T-1]   λ-returns
    var im_adv: List[Scalar[DT]]        # [B, T-1]   advantages
    var im_actbt: List[Scalar[DT]]      # [BF]       actions on the [B,T] grid
    var im_vlog: List[Scalar[DT]]       # [BF, NBINS] value logits (grad re-run)
    var im_gvlog: List[Scalar[DT]]      # [BF, NBINS] value-logit grad
    var im_vloss: List[Scalar[DT]]      # [B, T-1]
    var im_plog: List[Scalar[DT]]       # [BF, PLOG]  policy logits (grad re-run)
    var im_prior: List[Scalar[DT]]      # [BF, PLOG]  frozen prior logits
    var im_plog0: List[Scalar[DT]]      # [BF, NACT]  dist-0 policy block
    var im_prior0: List[Scalar[DT]]     # [BF, NACT]  dist-0 prior block
    var im_gplog0: List[Scalar[DT]]     # [BF, NACT]  dist-0 grad
    var im_gplog: List[Scalar[DT]]      # [BF, PLOG]  full policy-logit grad
    var im_clog: List[Scalar[DT]]       # [BF]        continue logits (use_continue)
    var im_chat: List[Scalar[DT]]       # [BF]        continue preds ĉ

    def __init__(out self):
        self.dyn = Self.DYN()
        self.te = Self.TE()
        self.ph = Self.PH()
        self.rh = Self.RH()
        self.vh = Self.VH()
        self.ph_prior = Self.PH()
        self.ch = Self.CH()
        self.im_h = List[Scalar[DT]]()
        self.im_act = List[Scalar[DT]]()
        self.im_rew = List[Scalar[DT]]()
        self.im_val = List[Scalar[DT]]()
        self.im_con = List[Scalar[DT]]()
        self.im_ret = List[Scalar[DT]]()
        self.im_adv = List[Scalar[DT]]()
        self.im_actbt = List[Scalar[DT]]()
        self.im_vlog = List[Scalar[DT]]()
        self.im_gvlog = List[Scalar[DT]]()
        self.im_vloss = List[Scalar[DT]]()
        self.im_plog = List[Scalar[DT]]()
        self.im_prior = List[Scalar[DT]]()
        self.im_plog0 = List[Scalar[DT]]()
        self.im_prior0 = List[Scalar[DT]]()
        self.im_gplog0 = List[Scalar[DT]]()
        self.im_gplog = List[Scalar[DT]]()
        self.im_clog = List[Scalar[DT]]()
        self.im_chat = List[Scalar[DT]]()
        self.agent_in = List[Scalar[DT]]()
        self.grad_zhat = List[Scalar[DT]]()
        self.zhat = List[Scalar[DT]]()
        self.ztil = List[Scalar[DT]]()
        self.grad_zt = List[Scalar[DT]]()
        self.grad_h = List[Scalar[DT]]()
        self.grad_h_tmp = List[Scalar[DT]]()
        self.plog = List[Scalar[DT]]()
        self.rlog = List[Scalar[DT]]()
        self.gpl = List[Scalar[DT]]()
        self.grl = List[Scalar[DT]]()
        self.clean_sig = List[Scalar[DT]]()
        self.clean_step = List[Scalar[DT]]()
        self.bc_in = List[Scalar[DT]]()
        self.gzero = List[Scalar[DT]]()
        self.ac_tok = List[Scalar[DT]]()
        self.ac_mask = List[Scalar[DT]]()
        self.rew_shift = List[Scalar[DT]]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", (
            "Dreamer4Agent heads/te/value are CPU; set DYN_TARGET=\"gpu\" to put"
            " the dynamics on device (the heavy imagination compute)."
        )
        comptime assert Self.DYN_TARGET == "cpu" or Self.DYN_TARGET == "gpu", (
            "DYN_TARGET must be 'cpu' or 'gpu'"
        )
        var m = Self()
        # dynamics on DYN_TARGET (GPU for the device rollout); everything else
        # (task embedder, heads, value, prior) stays on host.
        m.dyn = Self.DYN.make[target = Self.DYN_TARGET, INIT=INIT](ctx)
        m.te = Self.TE.make[target=target, INIT=INIT](ctx)
        m.ph = Self.PH.make[target=target, INIT=INIT](ctx)
        m.rh = Self.RH.make[target=target, INIT=INIT](ctx)
        # vh + ph_prior made AFTER rh so the dyn/te/ph/rh RNG draws (hence the
        # BC path) are byte-for-byte unchanged from the Phase-3 agent.
        m.vh = Self.VH.make[target=target, INIT=INIT](ctx)
        m.ph_prior = Self.PH.make[target=target, INIT=INIT](ctx)
        # continue head made LAST so all earlier RNG draws (BC path + Phase-4
        # heads) are byte-for-byte unchanged.
        m.ch = Self.CH.make[target=target, INIT=INIT](ctx)
        m.im_h.resize(Self.BF * Self.AGD, Scalar[DT](0.0))
        m.im_act.resize(Self.B * Self.TM1, Scalar[DT](0.0))
        m.im_rew.resize(Self.BF, Scalar[DT](0.0))
        m.im_val.resize(Self.BF, Scalar[DT](0.0))
        m.im_con.resize(Self.BF, Scalar[DT](0.0))
        m.im_ret.resize(Self.B * Self.TM1, Scalar[DT](0.0))
        m.im_adv.resize(Self.B * Self.TM1, Scalar[DT](0.0))
        m.im_actbt.resize(Self.BF, Scalar[DT](0.0))
        m.im_vlog.resize(Self.BF * Self.NBINS, Scalar[DT](0.0))
        m.im_gvlog.resize(Self.BF * Self.NBINS, Scalar[DT](0.0))
        m.im_vloss.resize(Self.B * Self.TM1, Scalar[DT](0.0))
        m.im_plog.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.im_prior.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.im_plog0.resize(Self.BF * Self.NACT, Scalar[DT](0.0))
        m.im_prior0.resize(Self.BF * Self.NACT, Scalar[DT](0.0))
        m.im_gplog0.resize(Self.BF * Self.NACT, Scalar[DT](0.0))
        m.im_gplog.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.im_clog.resize(Self.BF, Scalar[DT](0.0))
        m.im_chat.resize(Self.BF, Scalar[DT](0.0))
        m.agent_in.resize(Self.BF * Self.AGD, Scalar[DT](0.0))
        m.grad_zhat.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.zhat.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.ztil.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.grad_zt.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.grad_h.resize(Self.BF * Self.AGD, Scalar[DT](0.0))
        m.grad_h_tmp.resize(Self.BF * Self.AGD, Scalar[DT](0.0))
        m.plog.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.rlog.resize(Self.BF * Self.RLOG, Scalar[DT](0.0))
        m.gpl.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.grl.resize(Self.BF * Self.RLOG, Scalar[DT](0.0))
        m.clean_sig.resize(Self.BF, Scalar[DT](Float64(Self.KMAX - 1)))  # cleanest
        # sig_idx (KMAX-1); the BC forward feeds CLEAN content (sig_bc=1.0) here,
        # matching how imagination/ode_sampler place clean latents at this index.
        m.clean_step.resize(Self.BF, Scalar[DT](Float64(Self.EMAX)))
        m.bc_in.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.gzero.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.ac_tok.resize(Self.BF * Self.ADIM, Scalar[DT](0.0))
        m.ac_mask.resize(Self.ADIM, Scalar[DT](1.0))
        m.rew_shift.resize(Self.BF, Scalar[DT](0.0))
        return m^

    @staticmethod
    def display_label() -> String:
        return String("Dreamer4Agent")

    # ── Module conformance: forward/vjp unused (entry point is bc_train_step) ─
    def forward[
        target: StaticString, BATCH: Int, o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut output: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error("Dreamer4Agent.forward is unused; call bc_train_step")

    def vjp[
        target: StaticString, BATCH: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error("Dreamer4Agent.vjp is unused; call bc_train_step")

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        comptime assert Self.DYN_TARGET == "cpu", (
            "whole-agent for_each_param spans one target; with DYN_TARGET=\"gpu\""
            " the dynamics is on device — optimize submodules separately"
            " (agent.dyn on GPU, agent.ph/agent.vh on CPU)."
        )
        self.dyn.for_each_param[target, V](visitor, ctx, join_name(prefix, "dyn"))
        self.te.for_each_param[target, V](visitor, ctx, join_name(prefix, "te"))
        self.ph.for_each_param[target, V](visitor, ctx, join_name(prefix, "ph"))
        self.rh.for_each_param[target, V](visitor, ctx, join_name(prefix, "rh"))
        self.vh.for_each_param[target, V](visitor, ctx, join_name(prefix, "vh"))
        self.ch.for_each_param[target, V](visitor, ctx, join_name(prefix, "ch"))
        # `ph_prior` is the behavioral prior: BC-trained (in acwm), FROZEN during
        # imagination (the reverse-KL anchor). It IS optimized by the acwm step, so
        # it joins the whole-agent param walk (CPU path). It is NOT touched by the
        # imagination optimizer (which walks vh + ph only).
        self.ph_prior.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "ph_prior")
        )

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        comptime assert Self.DYN_TARGET == "cpu", (
            "whole-agent zero_grad spans one target; with DYN_TARGET=\"gpu\""
            " zero submodule grads separately."
        )
        self.dyn.zero_grad[target](ctx)
        self.te.zero_grad[target](ctx)
        self.ph.zero_grad[target](ctx)
        self.rh.zero_grad[target](ctx)
        self.vh.zero_grad[target](ctx)
        self.ch.zero_grad[target](ctx)
        self.ph_prior.zero_grad[target](ctx)   # BC-trained anchor (acwm)

    # ── checkpoint (params only; the frozen ph_prior + Adam moments are not
    #    saved). ONE combined file `<base>.ckpt` holding the tokenizer + agent
    #    (dyn, te, ph, rh, vh, ch) as consecutive name-validated sections (the
    #    save_params_multi format, but with PER-MODULE targets so the mixed
    #    DYN_TARGET — tok+dyn on device, heads/te on host — round-trips through
    #    one writer). `load` walks the SAME modules in the SAME order. `dctx`
    #    required when DYN_TARGET="gpu" (D2H/H2D the device tok+dyn params). te is
    #    a bespoke non-Module → param-only section (no for_each_state). ──
    @staticmethod
    def _wsec[M: Module, tgt: StaticString](
        mut w: BinaryCheckpointWriter, mut m: M, ctx: Optional[DeviceContext]
    ) raises:
        w.mode = 0
        var _ref1 = ParamVisitorRef.of[type_of(w), tgt](w)
        m.for_each_param[tgt](_ref1, ctx)
        w.mode = 1
        var _sref1 = ParamVisitorRef.of[type_of(w), tgt](w)
        m.for_each_state[tgt](_sref1, ctx)

    @staticmethod
    def _rsec[M: Module, tgt: StaticString](
        mut r: BinaryCheckpointReader, mut m: M, ctx: Optional[DeviceContext]
    ) raises:
        r.mode = 0
        var _ref2 = ParamVisitorRef.of[type_of(r), tgt](r)
        m.for_each_param[tgt](_ref2, ctx)
        r.mode = 1
        var _sref2 = ParamVisitorRef.of[type_of(r), tgt](r)
        m.for_each_state[tgt](_sref2, ctx)

    def save[
        TOK: Module
    ](
        mut self, mut tok: TOK, base: String,
        dctx: Optional[DeviceContext] = None,
    ) raises:
        var w = BinaryCheckpointWriter.to_file(base + ".ckpt", False)
        comptime if Self.DYN_TARGET == "gpu":
            Self._wsec[TOK, "gpu"](w, tok, dctx)
            Self._wsec[Self.DYN, "gpu"](w, self.dyn, dctx)
        else:
            Self._wsec[TOK, "cpu"](w, tok, None)
            Self._wsec[Self.DYN, "cpu"](w, self.dyn, None)
        w.mode = 0                          # te: param-only (no state section)
        var _ref3 = ParamVisitorRef.of[type_of(w), "cpu"](w)
        self.te.for_each_param["cpu"](_ref3, None)
        Self._wsec[Self.PH, "cpu"](w, self.ph, None)
        Self._wsec[Self.RH, "cpu"](w, self.rh, None)
        Self._wsec[Self.VH, "cpu"](w, self.vh, None)
        Self._wsec[Self.CH, "cpu"](w, self.ch, None)
        w.commit()

    def load[
        TOK: Module
    ](
        mut self, mut tok: TOK, base: String,
        dctx: Optional[DeviceContext] = None,
    ) raises:
        var bytes = _read_file_bytes(base + ".ckpt")
        var r = BinaryCheckpointReader(bytes^)
        comptime if Self.DYN_TARGET == "gpu":
            Self._rsec[TOK, "gpu"](r, tok, dctx)
            Self._rsec[Self.DYN, "gpu"](r, self.dyn, dctx)
        else:
            Self._rsec[TOK, "cpu"](r, tok, None)
            Self._rsec[Self.DYN, "cpu"](r, self.dyn, None)
        r.mode = 0                          # te: param-only
        var _ref4 = ParamVisitorRef.of[type_of(r), "cpu"](r)
        self.te.for_each_param["cpu"](_ref4, None)
        Self._rsec[Self.PH, "cpu"](r, self.ph, None)
        Self._rsec[Self.RH, "cpu"](r, self.rh, None)
        Self._rsec[Self.VH, "cpu"](r, self.vh, None)
        Self._rsec[Self.CH, "cpu"](r, self.ch, None)

    # ── storage-module call bridges (List host scratch ↔ boundary Tensor) ──
    # @staticmethod so the caller can pass three DISTINCT fields of `self`
    # (the module + an input List + an output List) without a `mut self`
    # whole-struct borrow aliasing them. CPU-only host world-model path.
    @staticmethod
    def _head_fwd[M: Module, NB: Int](
        mut m: M, imm inp: List[Scalar[DT]], mut out: List[Scalar[DT]]
    ) raises:
        comptime IN = M.IN_DIMS[0]
        comptime OUT = M.OUT_DIM
        var in_t = Tensor.alloc(NB * IN)
        for i in range(NB * IN):
            in_t.data[i] = inp[i]
        var out_t = Tensor.alloc(NB * OUT)
        call_forward["cpu", NB](m, TensorRefs[M.ARITY](in_t), out_t, None)
        for i in range(NB * OUT):
            out[i] = out_t.data[i]

    @staticmethod
    def _head_vjp[M: Module, NB: Int](
        mut m: M, imm fin: List[Scalar[DT]], imm go: List[Scalar[DT]]
    ) raises:
        # grad wrt input discarded (heads-only imagination training).
        comptime IN = M.IN_DIMS[0]
        comptime OUT = M.OUT_DIM
        var fin_t = Tensor.alloc(NB * IN)
        for i in range(NB * IN):
            fin_t.data[i] = fin[i]
        var go_t = Tensor.alloc(NB * OUT)
        for i in range(NB * OUT):
            go_t.data[i] = go[i]
        var gi_t = Tensor.alloc(NB * IN)
        call_vjp["cpu", NB](
            m, TensorRefs[M.ARITY](fin_t), go_t, TensorRefs[M.ARITY](gi_t), None
        )

    @staticmethod
    def _dyn_fwd[NB: Int](
        mut dyn: Self.DYN, imm inp: List[Scalar[DT]], mut out: List[Scalar[DT]]
    ) raises:
        comptime ND = Self.ND
        var in_t = Tensor.alloc(NB * ND)
        for i in range(NB * ND):
            in_t.data[i] = inp[i]
        var out_t = Tensor.alloc(NB * ND)
        dyn.forward["cpu", NB](TensorRefs[1](in_t), out_t, None)
        for i in range(NB * ND):
            out[i] = out_t.data[i]

    @staticmethod
    def _dyn_vjp[NB: Int](
        mut dyn: Self.DYN, imm fin: List[Scalar[DT]], imm go: List[Scalar[DT]]
    ) raises:
        # forward_input = `fin` (the input of the preceding dyn forward, so the
        # spatial-proj grad recomputes correctly); dyn reuses the grid/tf_out/
        # cache_sig it cached during that forward. grad wrt input discarded.
        comptime ND = Self.ND
        var fin_t = Tensor.alloc(NB * ND)
        for i in range(NB * ND):
            fin_t.data[i] = fin[i]
        var go_t = Tensor.alloc(NB * ND)
        for i in range(NB * ND):
            go_t.data[i] = go[i]
        var gi_t = Tensor.alloc(NB * ND)
        dyn.vjp["cpu", NB](
            TensorRefs[1](fin_t), go_t, TensorRefs[1](gi_t), None
        )

    @staticmethod
    def _dyn_fwd_gpu[NB: Int](
        mut dyn: Self.DYN, imm inp: List[Scalar[DT]], mut out: List[Scalar[DT]],
        c: DeviceContext,
    ) raises:
        # GPU sibling of _dyn_fwd: upload input → device forward → download the
        # flow output. Leaves dyn's device caches (grid/tf_out/agent_out) for a
        # following _dyn_vjp_gpu / an h_t D2H read.
        comptime ND = Self.ND
        var in_t = Tensor.alloc(NB * ND)
        for i in range(NB * ND):
            in_t.data[i] = inp[i]
        in_t.upload(c)
        var out_t = Tensor.alloc_gpu(c, NB * ND)
        dyn.forward["gpu", NB](TensorRefs[1](in_t), out_t, Optional(c))
        out_t.download(c)
        for i in range(NB * ND):
            out[i] = out_t.data[i]

    @staticmethod
    def _dyn_vjp_gpu[NB: Int](
        mut dyn: Self.DYN, imm fin: List[Scalar[DT]], imm go: List[Scalar[DT]],
        c: DeviceContext,
    ) raises:
        # GPU sibling of _dyn_vjp: forward_input `fin` + grad_output `go` uploaded
        # to device, dyn.vjp["gpu"] accumulates the dynamics param grads on device
        # (grad wrt input discarded). Matches the CPU forward_input semantics.
        comptime ND = Self.ND
        var fin_t = Tensor.alloc(NB * ND)
        for i in range(NB * ND):
            fin_t.data[i] = fin[i]
        fin_t.upload(c)
        var go_t = Tensor.alloc(NB * ND)
        for i in range(NB * ND):
            go_t.data[i] = go[i]
        go_t.upload(c)
        var gi_t = Tensor.alloc_gpu(c, NB * ND)
        dyn.vjp["gpu", NB](
            TensorRefs[1](fin_t), go_t, TensorRefs[1](gi_t), Optional(c)
        )

    # ── eval accessors ──────────────────────────────────────────────────
    def agent_out_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Return h_t from the last forward (for inspection / eval heads)."""
        return self.dyn.agent_out_ptr_cpu()

    def policy_logits_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Return the [BF, NMTP·NACT] policy logits from the last bc_train_step
        (distance n at columns [n·NACT, (n+1)·NACT)) — greedy action = argmax of
        the distance-0 block."""
        return _mao(self.plog.unsafe_ptr())

    # Imagination internals from the last imag_train_step (diagnostics): the
    # reward-head prediction, value-head prediction, and λ-return per imagined
    # state — used to check whether imagination sees phantom rewards/values.
    def im_rew_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:  # [BF]
        return _mao(self.im_rew.unsafe_ptr())

    def im_val_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:  # [BF]
        return _mao(self.im_val.unsafe_ptr())

    def im_ret_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:  # [B*(T-1)]
        return _mao(self.im_ret.unsafe_ptr())

    # ── online acting (single-step inference) ────────────────────────────
    def act_from_latents(
        mut self,
        z_window: Pointer[Scalar[DT], MutAnyOrigin],     # [n_ctx * ND]
        n_ctx: Int,
        action_hist: Pointer[Scalar[DT], MutAnyOrigin],  # [T * ADIM]
        task_id: Int,
        explore: Bool,
        u01: Float64,
        dctx: Optional[DeviceContext] = None,   # required when DYN_TARGET="gpu"
    ) raises -> Int:
        """Pick an action from a window of recent CLEAN world-model latents.

        Places the `n_ctx` latents into frames [0, n_ctx-1] of a [B=1, T] window
        (clean: σ=KMAX-1, step=EMAX; later frames zero — causal time attention
        means they cannot affect the read), sets the action tokens from
        `action_hist` (the one-hots leading INTO frames 1..n_ctx-1, the
        `imagine_rollout` convention), runs ONE frozen-dynamics forward, reads
        the agent token h at the current frame `n_ctx-1`, runs the policy head,
        and returns the action class in [0, NACT) from the dist-0 block —
        `explore=True` → categorical sample with `u01`; else argmax.

        Allocates act-LOCAL scratch (B=1 ⇒ BF=T); does not touch the training
        scratch. FWD path follows `DYN_TARGET` exactly like `imagine_rollout`."""
        comptime assert Self.ADIM == Self.NACT, (
            "act_from_latents needs ADIM = NACT (one-hot discrete actions)"
        )
        comptime T = Self.T
        comptime ND = Self.ND
        comptime AGD = Self.AGD
        comptime ADIM = Self.ADIM
        comptime PLOG = Self.PLOG
        comptime BF = Self.T               # B = 1

        var nc = n_ctx
        if nc < 1:
            nc = 1
        if nc > T:
            nc = T
        var cur = nc - 1                    # current state / acting frame

        # act-local scratch (B=1)
        var sig = List[Scalar[DT]](length=BF, fill=Scalar[DT](0.0))
        var step = List[Scalar[DT]](length=BF, fill=Scalar[DT](0.0))
        var act_oh = List[Scalar[DT]](length=BF * ADIM, fill=Scalar[DT](0.0))
        var act_mask = List[Scalar[DT]](length=BF * ADIM, fill=Scalar[DT](1.0))
        var packed = List[Scalar[DT]](length=BF * ND, fill=Scalar[DT](0.0))
        var zhat = List[Scalar[DT]](length=BF * ND, fill=Scalar[DT](0.0))
        var h_host = List[Scalar[DT]](length=BF * AGD, fill=Scalar[DT](0.0))
        var agp_l = List[Scalar[DT]](length=BF * AGD, fill=Scalar[DT](0.0))
        var task_l = List[Scalar[DT]](length=1, fill=Scalar[DT](Float64(task_id)))

        for f in range(nc):
            sig[f] = Scalar[DT](Float64(Self.KMAX - 1))
            step[f] = Scalar[DT](Float64(Self.EMAX))
            for i in range(ND):
                packed[f * ND + i] = z_window[unsafe_offset=f * ND + i]
        for i in range(BF * ADIM):
            act_oh[i] = action_hist[unsafe_offset=i]

        # task embedding → agent tokens [B=1, T]
        self.te.embed_into["cpu", 1, T](
            _mao(task_l.unsafe_ptr()), _mao(agp_l.unsafe_ptr()), None
        )

        # boundary tensors: `_fwd_window` writes host `.data` then (gpu) uploads /
        # downloads, so in_t/out_t need BOTH a host buffer AND (gpu) a device one.
        # `Tensor.make["gpu"]` is device-ONLY → alloc host, then add the device
        # buffer for the gpu path.
        var in_t = Tensor.alloc(BF * ND)
        var out_t = Tensor.alloc(BF * ND)
        var h_ag = Optional[HostBuffer[DT]](None)
        comptime if Self.DYN_TARGET == "gpu":
            var dc = dctx.value()
            in_t.ensure_gpu(dc, BF * ND)
            out_t.ensure_gpu(dc, BF * ND)
            h_ag = dc.enqueue_create_host_buffer[DT](BF * AGD)

        # one frozen-dynamics forward → h_host [BF, AGD]
        _fwd_window[Self.DYN, Self.DYN_TARGET, BF, ND, AGD](
            self.dyn,
            _mao(sig.unsafe_ptr()), _mao(step.unsafe_ptr()),
            _mao(act_oh.unsafe_ptr()), _mao(act_mask.unsafe_ptr()),
            _mao(agp_l.unsafe_ptr()),
            _mao(packed.unsafe_ptr()), _mao(zhat.unsafe_ptr()),
            _mao(h_host.unsafe_ptr()),
            in_t, out_t, dctx, h_ag,
        )

        # policy head on every frame's h → [BF, PLOG]; act on frame `cur`
        var plog = List[Scalar[DT]](length=BF * PLOG, fill=Scalar[DT](0.0))
        Self._head_fwd[Self.PH, BF](self.ph, h_host, plog)

        if explore:
            return cat_sample[Self.NACT](
                _mao(plog.unsafe_ptr()), cur * PLOG, UNIMIX, Scalar[DT](u01)
            )
        var best = 0
        var best_v = Float64(plog[cur * PLOG + 0])
        for a in range(1, Self.NACT):
            var v = Float64(plog[cur * PLOG + a])
            if v > best_v:
                best_v = v
                best = a
        return best

    def _run_bc_loss(
        mut self,
        ht: Pointer[Scalar[DT], MutAnyOrigin],
        actions: Pointer[Scalar[DT], MutAnyOrigin],
        rewards: Pointer[Scalar[DT], MutAnyOrigin],
        bins: Pointer[Scalar[DT], MutAnyOrigin],
        policy_weight: Scalar[DT],
        reward_weight: Scalar[DT],
    ) raises -> Float64:
        """BC loss on h_t → fills grad_h + head param grads. Returns the loss."""
        return bc_mtp_loss[
            Self.PH, Self.RH, Self.B, Self.T, Self.NMTP, Self.NACT,
            Self.NBINS, Self.AGD,
        ](
            self.ph, self.rh, ht, actions, rewards, bins,
            _mao(self.plog.unsafe_ptr()),
            _mao(self.rlog.unsafe_ptr()),
            _mao(self.gpl.unsafe_ptr()),
            _mao(self.grl.unsafe_ptr()),
            _mao(self.grad_h.unsafe_ptr()),
            _mao(self.grad_h_tmp.unsafe_ptr()),
            policy_weight=policy_weight,
            reward_weight=reward_weight,
        )

    def _train_prior_bc(
        mut self,
        ht: Pointer[Scalar[DT], MutAnyOrigin],       # [B*T, AGD] clean h_t
        actions: Pointer[Scalar[DT], MutAnyOrigin],  # [B*T] class ids
        policy_weight: Scalar[DT] = Scalar[DT](1.0),
    ) raises:
        """BC-train the behavioral prior `ph_prior` (policy-only) on the SAME
        clean h_t + real actions as the `ph` BC. Fills `ph_prior`'s param grads
        (the caller's acwm optimizer steps it). This makes `ph_prior` a diverse
        BC anchor — never touched by PMPO — so the imagination reverse-KL keeps
        the learned policy `ph` from mode-collapsing (the old self-snapshot prior
        collapsed together with `ph`, providing no resistance). Self-contained
        scratch (own Tensors); the grad-wrt-h is a throwaway (does NOT feed the
        dynamics)."""
        var plog = Tensor.alloc(Self.BF * Self.PLOG)
        var gpl = Tensor.alloc(Self.BF * Self.PLOG)
        var gh = Tensor.alloc(Self.BF * Self.AGD)   # throwaway grad-wrt-h
        _ = bc_policy_only_loss[
            Self.PH, Self.B, Self.T, Self.NMTP, Self.NACT, Self.AGD
        ](
            self.ph_prior, ht, actions,
            _mao(plog.data.unsafe_ptr()), _mao(gpl.data.unsafe_ptr()),
            _mao(gh.data.unsafe_ptr()),
            Scalar[DT](0.0), policy_weight,
        )

    # ── one joint BC + video-prediction training step (fills all grads) ──
    def bc_train_step(
        mut self,
        z1: Pointer[Scalar[DT], MutAnyOrigin],         # [BF, ND] latents
        z0: Pointer[Scalar[DT], MutAnyOrigin],         # [BF, ND] noise
        sigma: Pointer[Scalar[DT], MutAnyOrigin],      # [BF]
        sigma_idx: Pointer[Scalar[DT], MutAnyOrigin],  # [BF]
        step_idx: Pointer[Scalar[DT], MutAnyOrigin],   # [BF]
        do_boot: Bool,
        task_ids: Pointer[Scalar[DT], MutAnyOrigin],   # [B]
        actions: Pointer[Scalar[DT], MutAnyOrigin],    # [BF] class ids
        rewards: Pointer[Scalar[DT], MutAnyOrigin],    # [BF]
        bins: Pointer[Scalar[DT], MutAnyOrigin],       # [NBINS]
        policy_weight: Scalar[DT] = Scalar[DT](1.0),
        reward_weight: Scalar[DT] = Scalar[DT](1.0),
        clean_bc: Bool = True,
    ) raises -> Tuple[Float64, Float64]:
        """Returns (video_loss, bc_loss). Fills the param grads of all four
        components (caller then runs one `optim.step`). Assumes the caller has
        already `optim.zero_grad`'d this agent.

        `clean_bc=True` (default) DECOUPLES the two losses: the shortcut-forcing
        video loss trains the world model on noised reps, then a SEPARATE clean
        (σ=1) forward gives h_t the un-noised frame so the policy can read the
        observation — the straightforward BC. Both backprop into the shared
        transformer (grads accumulate over the two vjp calls). `clean_bc=False`
        is the paper's coupled form (BC reads the noised main-pass h_t); it only
        clones cleanly when the WM is a strong denoiser (large-scale regime)."""
        comptime assert Self.DYN_TARGET == "cpu", (
            "bc_train_step is the CPU world-model path; a DYN_TARGET=\"gpu\""
            " agent trains the dynamics via dynamics_pretrain_loss[FWD=\"gpu\"]"
            " directly (see the imagination lighthouse stage 1)."
        )
        var agp = _mao(self.agent_in.unsafe_ptr())
        var gzh = _mao(self.grad_zhat.unsafe_ptr())
        var zh = _mao(self.zhat.unsafe_ptr())
        var ghp = _mao(self.grad_h.unsafe_ptr())

        # 1. task embeddings → agent token input for every (b,t)
        self.te.embed_into["cpu", Self.B, Self.T](task_ids, agp, None)

        # 2. shortcut-forcing video-prediction loss (injects agent tokens into
        #    every pass; the MAIN pass leaves h_t in dyn.agent_out)
        var loss_v = dynamics_pretrain_loss[
            Self.DYN, Self.B, Self.T, Self.B_SELF, Self.NSP, Self.DSP,
            Self.KMAX, "cpu", 0, Self.AGD,
        ](
            self.dyn, z1, z0, sigma, sigma_idx, step_idx, do_boot, gzh, zh,
            agent_in=agp,
        )

        # main-pass input z̃ = (1−σ)·z0 + σ·z1 (the storage dyn.vjp recomputes
        # the spatial-proj forward from it; identical to the loss's internal z̃).
        for bt in range(Self.BF):
            var s = Float64(sigma[unsafe_offset=bt])
            for i in range(Self.ND):
                var idx = bt * Self.ND + i
                self.ztil[idx] = Scalar[DT](
                    (1.0 - s) * Float64(z0[unsafe_offset=idx]) + s * Float64(z1[unsafe_offset=idx])
                )

        var loss_bc: Float64 = 0.0
        if not clean_bc:
            # COUPLED: BC reads the noised main-pass h_t; one combined vjp.
            loss_bc = self._run_bc_loss(
                self.dyn.agent_out_ptr_cpu(), actions, rewards, bins,
                policy_weight, reward_weight,
            )
            self.dyn.set_grad_h(ghp, Self.BF)
            Self._dyn_vjp[Self.BF](self.dyn, self.ztil, self.grad_zhat)
            self.te.accumulate_grad["cpu", Self.B, Self.T](
                self.dyn.grad_agent_in_ptr_cpu(), None
            )
            return Tuple(loss_v, loss_bc)

        # DECOUPLED clean BC.
        # 3. video vjp ONLY (zero the agent-token grad), using the video caches.
        for i in range(Self.BF * Self.AGD):
            self.grad_h[i] = Scalar[DT](0.0)
        self.dyn.set_grad_h(ghp, Self.BF)
        Self._dyn_vjp[Self.BF](self.dyn, self.ztil, self.grad_zhat)

        # 4. dedicated CLEAN forward on z1 (σ=1) → un-noised h_t.
        self.dyn.set_indices(
            _mao(self.clean_sig.unsafe_ptr()),
            _mao(self.clean_step.unsafe_ptr()),
            Self.BF,
        )
        self.dyn.set_agent_in(agp, Self.BF)
        # near-clean input z̃ = σ_bc·z1 + (1−σ_bc)·z0 at the highest TRAINED σ
        var sig_bc = Float64(Self.KMAX - 1) / Float64(Self.KMAX)   # 0.75
        for i in range(Self.BF * Self.ND):
            self.bc_in[i] = Scalar[DT](
                sig_bc * Float64(z1[unsafe_offset=i]) + (1.0 - sig_bc) * Float64(z0[unsafe_offset=i])
            )
        Self._dyn_fwd[Self.BF](self.dyn, self.bc_in, self.zhat)

        # 5. BC loss on the clean h_t → grad_h + head grads
        loss_bc = self._run_bc_loss(
            self.dyn.agent_out_ptr_cpu(), actions, rewards, bins,
            policy_weight, reward_weight,
        )

        # 6. BC vjp through the clean forward (zero flow grad), accumulating
        #    into the dynamics params; then the task-embedder grad.
        self.dyn.set_grad_h(ghp, Self.BF)
        Self._dyn_vjp[Self.BF](self.dyn, self.bc_in, self.gzero)
        self.te.accumulate_grad["cpu", Self.B, Self.T](
            self.dyn.grad_agent_in_ptr_cpu(), None
        )

        return Tuple(loss_v, loss_bc)

    # ── action-conditioned world-model + reward + BC step ────────────────
    def acwm_train_step(
        mut self,
        z1: Pointer[Scalar[DT], MutAnyOrigin],         # [BF, ND] latents
        z0: Pointer[Scalar[DT], MutAnyOrigin],         # [BF, ND] noise
        sigma: Pointer[Scalar[DT], MutAnyOrigin],      # [BF]
        sigma_idx: Pointer[Scalar[DT], MutAnyOrigin],  # [BF]
        step_idx: Pointer[Scalar[DT], MutAnyOrigin],   # [BF]
        do_boot: Bool,
        task_ids: Pointer[Scalar[DT], MutAnyOrigin],   # [B]
        actions: Pointer[Scalar[DT], MutAnyOrigin],    # [BF] class ids
        rewards: Pointer[Scalar[DT], MutAnyOrigin],    # [BF] transition reward
        bins: Pointer[Scalar[DT], MutAnyOrigin],       # [NBINS]
        policy_weight: Scalar[DT] = Scalar[DT](1.0),
        reward_weight: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Tuple[Float64, Float64]:
        """ACTION-CONDITIONED counterpart of `bc_train_step`. Trains the world
        model so the action token MOVES the transition (and hence the reward),
        which `imag_train_step` then exploits to improve the policy by
        imagination. Returns (video_loss, bc_loss); fills all four components'
        grads — caller `zero_grad`s + `step`s one optimizer.

        Conventions (matched to `imagine_rollout`):
          • the action token at frame f conditions the transition INTO f, so it
            holds the dataset action taken at f−1 (frame 0 → no action);
          • the reward head at frame f predicts the reward of that transition,
            i.e. the dataset reward earned at f−1 (so the λ-return's r_{t+1}
            term equals the reward of the action sampled at state t);
          • the policy head at frame f clones the SAME-frame action a_f (it is
            what `imagine_rollout` samples at state f).
        `actions`/`rewards` are passed UNSHIFTED ([BF] per (b, window-pos)); the
        shifts are built here. Always clean-decoupled (a dedicated near-clean
        forward gives h_t an un-noised frame). CPU world-model path only."""
        comptime assert Self.DYN_TARGET == "cpu", (
            "acwm_train_step is the CPU world-model path (DYN_TARGET=\"cpu\")"
        )
        comptime assert Self.ADIM == Self.NACT, (
            "acwm_train_step needs ADIM = NACT (one-hot action conditioning)"
        )
        var agp = _mao(self.agent_in.unsafe_ptr())
        var gzh = _mao(self.grad_zhat.unsafe_ptr())
        var zh = _mao(self.zhat.unsafe_ptr())
        var ghp = _mao(self.grad_h.unsafe_ptr())
        var atk = _mao(self.ac_tok.unsafe_ptr())
        var amk = _mao(self.ac_mask.unsafe_ptr())
        var rsh = _mao(self.rew_shift.unsafe_ptr())

        # 0. build the SHIFTED action tokens + transition rewards (frame f ← f−1;
        #    frame 0 = no preceding in-window action ⇒ zeros / 0 reward).
        for i in range(Self.BF * Self.ADIM):
            self.ac_tok[i] = Scalar[DT](0.0)
        for b in range(Self.B):
            self.rew_shift[b * Self.T + 0] = Scalar[DT](0.0)
            for f in range(1, Self.T):
                var a_prev = Int(Float64(actions[unsafe_offset=b * Self.T + f - 1]) + 0.5)
                self.ac_tok[(b * Self.T + f) * Self.ADIM + a_prev] = Scalar[DT](
                    1.0
                )
                self.rew_shift[b * Self.T + f] = rewards[unsafe_offset=b * Self.T + f - 1]

        # 1. task embeddings → agent token input
        self.te.embed_into["cpu", Self.B, Self.T](task_ids, agp, None)

        # 2. ACTION-CONDITIONED shortcut-forcing video loss (ADIM=NACT): the
        #    action token moves the predicted transition. MAIN pass leaves h_t.
        var loss_v = dynamics_pretrain_loss[
            Self.DYN, Self.B, Self.T, Self.B_SELF, Self.NSP, Self.DSP,
            Self.KMAX, "cpu", Self.ADIM, Self.AGD,
        ](
            self.dyn, z1, z0, sigma, sigma_idx, step_idx, do_boot, gzh, zh,
            actions=atk, act_mask=amk, agent_in=agp,
        )

        # main-pass input z̃ (= the storage dyn.vjp forward_input for the video
        # vjp). The dyn cached its grid/tf_out/cache_sig from this same z̃.
        for bt in range(Self.BF):
            var s = Float64(sigma[unsafe_offset=bt])
            for i in range(Self.ND):
                var idx = bt * Self.ND + i
                self.ztil[idx] = Scalar[DT](
                    (1.0 - s) * Float64(z0[unsafe_offset=idx]) + s * Float64(z1[unsafe_offset=idx])
                )

        # 3. video vjp ONLY (zero the agent-token grad) using the video caches;
        #    accumulates the action-MLP + transition param grads.
        for i in range(Self.BF * Self.AGD):
            self.grad_h[i] = Scalar[DT](0.0)
        self.dyn.set_grad_h(ghp, Self.BF)
        Self._dyn_vjp[Self.BF](self.dyn, self.ztil, self.grad_zhat)

        # 4. dedicated CLEAN forward WITH the action tokens → an action-
        #    conditioned, un-noised h_t for the heads.
        #
        #    sig_bc MUST be 1.0 (pure z1, no noise). The heads (policy/reward/
        #    value) are QUERIED in imagination on FULLY-clean latents — real z1
        #    context + the flow head's x̂1 prediction — placed at sig_idx=KMAX-1
        #    (imag_rollout.mojo; and the WM's own video sampler ode_sampler.mojo:73
        #    likewise conditions on CLEAN context at KMAX-1). Training the heads'
        #    h_t on a 0.75·z1+0.25·z0 (σ=0.75) frame instead — the earlier value —
        #    was a train/inference distribution shift LOCALIZED TO THE HEADS: the
        #    reward head then never reached the tile-crossing (+3) regime inside
        #    imagination → dead imagined-reward stream → the value collapses to a
        #    constant and PMPO gets no advantage signal. sig_idx stays KMAX-1 (the
        #    cleanest index); only the CONTENT is now the clean z1 imagination uses.
        self.dyn.set_indices(
            _mao(self.clean_sig.unsafe_ptr()),
            _mao(self.clean_step.unsafe_ptr()),
            Self.BF,
        )
        self.dyn.set_actions(atk, amk, Self.BF)
        self.dyn.set_agent_in(agp, Self.BF)
        var sig_bc = 1.0                       # clean latent (match imagination)
        for i in range(Self.BF * Self.ND):
            self.bc_in[i] = Scalar[DT](
                sig_bc * Float64(z1[unsafe_offset=i]) + (1.0 - sig_bc) * Float64(z0[unsafe_offset=i])
            )
        Self._dyn_fwd[Self.BF](self.dyn, self.bc_in, self.zhat)

        # 5. BC loss on the clean h_t: policy clones SAME-frame actions; reward
        #    head fits the SHIFTED (transition-into) reward.
        var loss_bc = self._run_bc_loss(
            self.dyn.agent_out_ptr_cpu(), actions, rsh, bins,
            policy_weight, reward_weight,
        )
        # 5b. BC-train the behavioral prior on the SAME clean h_t (policy-only).
        self._train_prior_bc(
            self.dyn.agent_out_ptr_cpu(), actions, policy_weight
        )

        # 6. BC vjp through the clean forward (zero flow grad) → dyn + act-MLP
        #    grads; then the task-embedder grad.
        self.dyn.set_grad_h(ghp, Self.BF)
        Self._dyn_vjp[Self.BF](self.dyn, self.bc_in, self.gzero)
        self.te.accumulate_grad["cpu", Self.B, Self.T](
            self.dyn.grad_agent_in_ptr_cpu(), None
        )

        return Tuple(loss_v, loss_bc)

    def acwm_train_step_gpu(
        mut self,
        z1: Pointer[Scalar[DT], MutAnyOrigin],
        z0: Pointer[Scalar[DT], MutAnyOrigin],
        sigma: Pointer[Scalar[DT], MutAnyOrigin],
        sigma_idx: Pointer[Scalar[DT], MutAnyOrigin],
        step_idx: Pointer[Scalar[DT], MutAnyOrigin],
        do_boot: Bool,
        task_ids: Pointer[Scalar[DT], MutAnyOrigin],
        actions: Pointer[Scalar[DT], MutAnyOrigin],
        rewards: Pointer[Scalar[DT], MutAnyOrigin],
        bins: Pointer[Scalar[DT], MutAnyOrigin],
        dctx: DeviceContext,
        policy_weight: Scalar[DT] = Scalar[DT](1.0),
        reward_weight: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Tuple[Float64, Float64]:
        """GPU-dynamics counterpart of `acwm_train_step` (DYN_TARGET=\"gpu\"): the
        three shortcut-forcing forwards + both vjps run the transformer ON DEVICE;
        the heads (ph/rh) + task embedder stay on host, reading h_t / feeding the
        agent-token grad across the device boundary (D2H). Grads are bit-parity
        with the CPU path (see test_dreamer4_acwm_gpu_parity)."""
        comptime assert Self.DYN_TARGET == "gpu", (
            "acwm_train_step_gpu is the GPU world-model path (DYN_TARGET=\"gpu\")"
        )
        comptime assert Self.ADIM == Self.NACT, (
            "acwm_train_step_gpu needs ADIM = NACT (one-hot action conditioning)"
        )
        # Pre-size ALL dynamics GPU scratch to the FULL batch BF up front. With
        # do_boot=True the bootstrap half-passes run at the smaller self-row batch
        # BS=B_SELF·T first; the device/host staging buffers only grow (never
        # shrink), so sizing to BF now prevents a later BF copy from overflowing a
        # BS-sized staging buffer ("Copy size exceeds Metal buffer length").
        self.dyn._ensure_scratch_gpu(Self.BF, dctx)
        var agp = _mao(self.agent_in.unsafe_ptr())
        var gzh = _mao(self.grad_zhat.unsafe_ptr())
        var zh = _mao(self.zhat.unsafe_ptr())
        var ghp = _mao(self.grad_h.unsafe_ptr())
        var atk = _mao(self.ac_tok.unsafe_ptr())
        var amk = _mao(self.ac_mask.unsafe_ptr())
        var rsh = _mao(self.rew_shift.unsafe_ptr())

        # 0. shifted action tokens + transition rewards (identical to CPU)
        for i in range(Self.BF * Self.ADIM):
            self.ac_tok[i] = Scalar[DT](0.0)
        for b in range(Self.B):
            self.rew_shift[b * Self.T + 0] = Scalar[DT](0.0)
            for f in range(1, Self.T):
                var a_prev = Int(Float64(actions[unsafe_offset=b * Self.T + f - 1]) + 0.5)
                self.ac_tok[(b * Self.T + f) * Self.ADIM + a_prev] = Scalar[DT](
                    1.0
                )
                self.rew_shift[b * Self.T + f] = rewards[unsafe_offset=b * Self.T + f - 1]

        # 1. task embeddings → agent token input (host)
        self.te.embed_into["cpu", Self.B, Self.T](task_ids, agp, None)

        # 2. ACTION-CONDITIONED shortcut-forcing video loss ON DEVICE (FWD="gpu")
        var loss_v = dynamics_pretrain_loss[
            Self.DYN, Self.B, Self.T, Self.B_SELF, Self.NSP, Self.DSP,
            Self.KMAX, "gpu", Self.ADIM, Self.AGD,
        ](
            self.dyn, z1, z0, sigma, sigma_idx, step_idx, do_boot, gzh, zh,
            ctx=Optional(dctx), actions=atk, act_mask=amk, agent_in=agp,
        )

        # main-pass input z̃ (the video vjp forward_input)
        for bt in range(Self.BF):
            var s = Float64(sigma[unsafe_offset=bt])
            for i in range(Self.ND):
                var idx = bt * Self.ND + i
                self.ztil[idx] = Scalar[DT](
                    (1.0 - s) * Float64(z0[unsafe_offset=idx]) + s * Float64(z1[unsafe_offset=idx])
                )

        # 3. video vjp ONLY (zero the agent-token grad), on device
        for i in range(Self.BF * Self.AGD):
            self.grad_h[i] = Scalar[DT](0.0)
        self.dyn.set_grad_h(ghp, Self.BF)
        Self._dyn_vjp_gpu[Self.BF](self.dyn, self.ztil, self.grad_zhat, dctx)

        # 4. dedicated CLEAN forward WITH action tokens, on device. sig_bc MUST
        #    match the CPU path (=1.0, clean z1): the heads are queried in
        #    imagination on fully-clean latents at sig_idx=KMAX-1, so training
        #    their h_t on a σ<1 corrupted frame is a train/inference shift that
        #    kills the imagined reward stream. See the CPU-path note above.
        self.dyn.set_indices(
            _mao(self.clean_sig.unsafe_ptr()),
            _mao(self.clean_step.unsafe_ptr()),
            Self.BF,
        )
        self.dyn.set_actions(atk, amk, Self.BF)
        self.dyn.set_agent_in(agp, Self.BF)
        var sig_bc = 1.0                       # clean latent (match imagination + CPU)
        for i in range(Self.BF * Self.ND):
            self.bc_in[i] = Scalar[DT](
                sig_bc * Float64(z1[unsafe_offset=i]) + (1.0 - sig_bc) * Float64(z0[unsafe_offset=i])
            )
        Self._dyn_fwd_gpu[Self.BF](self.dyn, self.bc_in, self.zhat, dctx)

        # 5. BC loss on the clean h_t — D2H the agent tokens for the CPU heads.
        self.dyn.sync_agent_out(dctx)
        var loss_bc = self._run_bc_loss(
            self.dyn.agent_out_ptr_cpu(), actions, rsh, bins,
            policy_weight, reward_weight,
        )
        # 5b. BC-train the behavioral prior on the SAME clean h_t (policy-only).
        self._train_prior_bc(
            self.dyn.agent_out_ptr_cpu(), actions, policy_weight
        )

        # 6. BC vjp through the clean forward (zero flow grad) → dyn + act-MLP
        #    grads; then the task-embedder grad (D2H the agent-input grad).
        self.dyn.set_grad_h(ghp, Self.BF)
        Self._dyn_vjp_gpu[Self.BF](self.dyn, self.bc_in, self.gzero, dctx)
        self.dyn.sync_grad_agent_in(dctx)
        self.te.accumulate_grad["cpu", Self.B, Self.T](
            self.dyn.grad_agent_in_ptr_cpu(), None
        )

        return Tuple(loss_v, loss_bc)

    # ── imagination RL (Phase 4) ─────────────────────────────────────────
    def snapshot_prior(mut self) raises:
        """Freeze the current policy head as the behavioral prior π_prior (the
        PMPO reverse-KL anchor). Call once before imagination training starts."""
        polyak_module["cpu", Self.PH](self.ph, self.ph_prior, Scalar[DT](1.0))

    def imag_policy_logits_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """Per-state policy logits [BF, PLOG] from the last `imag_train_step`
        (greedy action = argmax of the dist-0 block)."""
        return _mao(self.im_plog.unsafe_ptr())

    def imag_train_step(
        mut self,
        ctx: Pointer[Scalar[DT], MutAnyOrigin],       # [B, NCTX, ND]
        u01: Pointer[Scalar[DT], MutAnyOrigin],       # [B, T] action rng
        znoise: Pointer[Scalar[DT], MutAnyOrigin],    # [B, T, ND] ODE seeds
        task_ids: Pointer[Scalar[DT], MutAnyOrigin],  # [B]
        bins: Pointer[Scalar[DT], MutAnyOrigin],      # [NBINS]
        gamma: Scalar[DT] = Scalar[DT](0.997),
        lam: Scalar[DT] = Scalar[DT](0.95),
        alpha: Scalar[DT] = Scalar[DT](0.5),
        beta: Scalar[DT] = Scalar[DT](0.3),
        policy_weight: Scalar[DT] = Scalar[DT](1.0),
        value_weight: Scalar[DT] = Scalar[DT](1.0),
        use_continue: Bool = False,             # discount with the continue head
        dctx: Optional[DeviceContext] = None,   # required when DYN_TARGET="gpu"
    ) raises -> Tuple[Float64, Float64]:
        """One imagination-RL step (paper §3.3). Generates an on-policy rollout
        inside the FROZEN world model, then trains ONLY the policy + value heads:
          • value head — TD-λ twohot CE vs sg(R_t^λ)   (eq. 10)
          • policy head — PMPO, sign-of-advantage + reverse-KL prior (eq. 11).
        Fills the grads of `ph` and `vh` ONLY (dyn / te / rh / ph_prior get no
        grad ⇒ frozen under a fresh heads-only optimizer). Returns
        (value_loss, policy_loss). Caller `zero_grad`s + `step`s the optimizer.
        """
        comptime assert Self.ADIM == Self.NACT, (
            "imag_train_step needs ADIM = NACT (one-hot action conditioning)"
        )
        comptime assert Self.NMTP >= 1, "need at least the dist-0 MTP block"
        var agp = _mao(self.agent_in.unsafe_ptr())
        var im_h_p = _mao(self.im_h.unsafe_ptr())
        var im_act_p = _mao(self.im_act.unsafe_ptr())
        var im_rew_p = _mao(self.im_rew.unsafe_ptr())
        var im_val_p = _mao(self.im_val.unsafe_ptr())
        var im_con_p = _mao(self.im_con.unsafe_ptr())
        var im_ret_p = _mao(self.im_ret.unsafe_ptr())

        # 1. task embeddings → agent tokens
        self.te.embed_into["cpu", Self.B, Self.T](task_ids, agp, None)

        # 2. imagined rollout (frozen transformer + heads, forward-only). The
        #    dynamics forward runs on DYN_TARGET (GPU = the heavy compute); the
        #    heads + everything below stay on host.
        comptime if Self.DYN_TARGET == "cpu":
            imagine_rollout[
                Self.DYN, Self.PH, Self.VH, Self.RH, Self.B, Self.T, Self.NSP,
                Self.DSP, Self.KMAX, Self.KI, Self.NCTX, Self.AGD, Self.NACT,
                Self.NBINS, Self.NMTP, "cpu",
            ](
                self.dyn, self.ph, self.vh, self.rh, ctx, agp, u01, znoise, bins,
                im_h_p, im_act_p, im_rew_p, im_val_p,
            )
        else:
            imagine_rollout[
                Self.DYN, Self.PH, Self.VH, Self.RH, Self.B, Self.T, Self.NSP,
                Self.DSP, Self.KMAX, Self.KI, Self.NCTX, Self.AGD, Self.NACT,
                Self.NBINS, Self.NMTP, "gpu",
            ](
                self.dyn, self.ph, self.vh, self.rh, ctx, agp, u01, znoise, bins,
                im_h_p, im_act_p, im_rew_p, im_val_p, dctx=dctx,
            )

        # 3. continue factor con_t. use_continue=False ⇒ constant γ (no
        #    termination). use_continue=True ⇒ con_t = γ·ĉ_t where ĉ_t is the
        #    (frozen) continue head's sigmoid read off the rollout's h_t, so the
        #    λ-return truncates at predicted terminal states. Then λ-returns.
        if use_continue:
            Self._head_fwd[Self.CH, Self.BF](self.ch, self.im_h, self.im_clog)
            var clog_p = _mao(self.im_clog.unsafe_ptr())
            var chat_p = _mao(self.im_chat.unsafe_ptr())
            continue_pred[Self.BF](clog_p, chat_p)
            for i in range(Self.BF):
                self.im_con[i] = gamma * self.im_chat[i]
        else:
            for i in range(Self.BF):
                self.im_con[i] = gamma
        lambda_returns[Self.B, Self.T](
            im_rew_p, im_val_p, im_con_p, lam, im_ret_p
        )

        # 4. advantages A_t = R_t^λ − v_t (states 0..T-2); actions on the [B,T] grid
        for b in range(Self.B):
            for t in range(Self.TM1):
                self.im_adv[b * Self.TM1 + t] = (
                    self.im_ret[b * Self.TM1 + t] - self.im_val[b * Self.T + t]
                )
            for t in range(Self.T):
                self.im_actbt[b * Self.T + t] = Scalar[DT](0.0)
            for t in range(Self.TM1):
                self.im_actbt[b * Self.T + t] = self.im_act[b * Self.TM1 + t]

        # 5. value loss + backward → vh param grads
        Self._head_fwd[Self.VH, Self.BF](self.vh, self.im_h, self.im_vlog)
        var vlog_p = _mao(self.im_vlog.unsafe_ptr())
        var vloss_p = _mao(self.im_vloss.unsafe_ptr())
        value_td_loss_cpu[Self.B, Self.T, Self.NBINS](
            vlog_p, bins, im_ret_p, vloss_p
        )
        var vloss: Float64 = 0.0
        for i in range(Self.B * Self.TM1):
            vloss += Float64(self.im_vloss[i])
        # d_loss = value_weight (reuse the loss buffer as the cotangent)
        for i in range(Self.B * Self.TM1):
            self.im_vloss[i] = value_weight
        var gvlog_p = _mao(self.im_gvlog.unsafe_ptr())
        value_td_loss_backward[Self.B, Self.T, Self.NBINS](
            vlog_p, bins, im_ret_p, vloss_p, gvlog_p
        )
        # vh backward (grad wrt h discarded — heads-only training)
        Self._head_vjp[Self.VH, Self.BF](self.vh, self.im_h, self.im_gvlog)

        # 6. policy: current + frozen-prior logits → PMPO (dist-0 block)
        Self._head_fwd[Self.PH, Self.BF](self.ph, self.im_h, self.im_plog)
        Self._head_fwd[Self.PH, Self.BF](self.ph_prior, self.im_h, self.im_prior)
        var plog_p = _mao(self.im_plog.unsafe_ptr())
        var prior_p = _mao(self.im_prior.unsafe_ptr())
        # extract dist-0 logits [BF, NACT]
        for s in range(Self.BF):
            for a in range(Self.NACT):
                self.im_plog0[s * Self.NACT + a] = self.im_plog[s * Self.PLOG + a]
                self.im_prior0[s * Self.NACT + a] = self.im_prior[
                    s * Self.PLOG + a
                ]
        var plog0_p = _mao(self.im_plog0.unsafe_ptr())
        var prior0_p = _mao(self.im_prior0.unsafe_ptr())
        var actbt_p = _mao(self.im_actbt.unsafe_ptr())
        var adv_p = _mao(self.im_adv.unsafe_ptr())
        var ploss = pmpo_policy_loss_cpu[Self.B, Self.T, Self.NACT](
            plog0_p, prior0_p, actbt_p, adv_p, alpha, beta
        )
        var gplog0_p = _mao(self.im_gplog0.unsafe_ptr())
        pmpo_policy_loss_backward[Self.B, Self.T, Self.NACT](
            plog0_p, prior0_p, actbt_p, adv_p, alpha, beta, policy_weight,
            gplog0_p,
        )
        # scatter dist-0 grad into the full [BF, PLOG] grad (other blocks = 0)
        for i in range(Self.BF * Self.PLOG):
            self.im_gplog[i] = Scalar[DT](0.0)
        for s in range(Self.BF):
            for a in range(Self.NACT):
                self.im_gplog[s * Self.PLOG + a] = self.im_gplog0[
                    s * Self.NACT + a
                ]
        # ph backward (grad wrt h discarded — heads-only training)
        Self._head_vjp[Self.PH, Self.BF](self.ph, self.im_h, self.im_gplog)

        return Tuple(vloss, ploss)
