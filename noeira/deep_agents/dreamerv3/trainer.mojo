"""DreamerV3Trainer — SAC-style block-composed trainer (CPU; GPU-ready).

Composes the `blocks.mojo` units (`WMStep` / `ParamSyncStep` / `ACStep`)
over a shared `DreamerState`, mirroring `deep_agents/sac/trainer.mojo`:
`train_target: StaticString` is comptime, `ctx: Optional[DeviceContext]` is
threaded through `make`/`step`. v1 instantiates `train_target="cpu"` (the
custom RSSM ops are CPU-only until PR5c Step 5 adds their GPU kernels; the
composition + plumbing are already GPU-shaped, so the GPU enable is localized
to each block's `comptime if target == "gpu"` branch + the ops' kernels).

`train_step` = sample a length-T window → `WMStep` (WM-BPTT) → `ParamSyncStep`
(core/prior → imagine) → `ACStep` (imagination AC + Polyak slowvalue).

Validated CPU: `test_dreamerv3_trainer_smoke.mojo` (synthetic) +
`test_dreamerv3_pendulum_smoke.mojo` (real Pendulum env loop).

Activation = SiLU (size1m/dmc config `act: silu`) — production type members
pass `SwishOp`; the validation spikes keep the GELU default for fixture parity.

v1 simplifications: random-action data collection (running-carry policy
action lives on `DreamerV3Agent`); imagination from the final posterior carry.
"""

from std.random import random_float64
from std.math import exp
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer, Kaiming, Zero
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs, child_refs
from noeira.nn.core.checkpoint import (
    CheckpointReader, _split_lines,
    BinaryCheckpointWriter, BinaryCheckpointReader,
    _write_file_bytes, _read_file_bytes, _is_v3_header,
)
from noeira.nn.core.param import ParamVisitor, walk_params, ParamVisitorRef
from noeira.nn.primitives.ops.swish_op import SwishOp
from noeira.nn.optimizer.dreamer_opt import DreamerOpt
from noeira.nn.optimizer.schedules import LinearWarmupSchedule
from noeira.deep_agents.data.any_sequence_replay import AnySequenceReplay
from noeira.cuda import CUDAGraph, maybe_capture_replay
from noeira.deep_agents.dreamerv3.wm import (
    WMCoreGraph, WMImagineGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from noeira.deep_agents.dreamerv3.nets import (
    DreamerEncoder, DreamerDecoder, DreamerValue, DreamerPolicyHead,
)
from noeira.nn.core.module import Module
from noeira.deep_agents.dreamerv3.twohot import (
    symexp_twohot_bins,
    twohot_pred,
    DREAMER_REWARD_GRID_LO,
)
from noeira.deep_agents.dreamerv3.normalize import PercentileNormalize
from noeira.deep_agents.dreamerv3.blocks import (
    DreamerState, WMStep, ParamSyncStep, ACStep,
)


@always_inline
def _hp(mut t: Tensor) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Sanctioned host-pointer view of a storage Tensor's CPU `data` — for the
    raw-pointer helpers (`twohot_pred`, sample_batch). CPU only."""
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](t.data.unsafe_ptr())


@always_inline
def _symexp(x: Scalar[DT]) -> Scalar[DT]:
    """Inverse of symlog: sign(x)·(exp(|x|)−1). Decoder targets are symlog(obs),
    so symexp maps a decoder output back to raw observation space."""
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * (exp(a) - Scalar[DT](1.0))


@always_inline
def _recon_decode[SIGMOID: Bool](x: Scalar[DT]) -> Scalar[DT]:
    """Map a raw decoder output back to observation space. Must match the recon
    loss: sigmoid (bounded [0,1] pixels) when RECON_SIGMOID, else symexp."""
    comptime if SIGMOID:
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))
    else:
        return _symexp(x)


@fieldwise_init
struct DreamerV3Trainer[
    train_target: StaticString,
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, VU: Int, PU: Int,
    BINS: Int, B: Int, T: Int, T_IMAG: Int, CAP: Int, DISCRETE: Bool = False,
    # Encoder / decoder Module TYPES (default = MLP; pixel obs pass the CNN
    # nets from nets_cnn.mojo). The decoder is threaded into DecLossGraph; the
    # encoder is run standalone to produce `tokens[TOKEN]`. OBS stays the flat
    # obs dim (C*H*W for images — the conv index math reads it as [C,H,W]).
    ENC: Module = DreamerEncoder[OBS, TOKEN, SwishOp],
    DEC: Module = DreamerDecoder[STOCH * CLASSES + DETER, OBS, DEC_U, SwishOp],
    # RECON_SIGMOID=True → reference pixel recon: decoder logits -> sigmoid,
    # plain MSE vs raw [0,1] obs (decode = sigmoid). False (default) keeps the
    # symlog recon, correct for unbounded vector obs (CartPole/Pendulum).
    RECON_SIGMOID: Bool = False,
    # Reward + value/slowvalue OUTPUT-layer initializer, declared structurally
    # (InitWith in nets.mojo) — replaces the runtime `out_init_scale` post-hoc
    # scaling (silent-on-miss name paths). Zero = the paper's zero-init (p.6,
    # best for negative-reward tasks); Kaiming = full pre-zero-init optimism
    # (helps POSITIVE-reward tasks like CartPole explore/solve faster). The
    # policy head's reference outscale 0.01 is FIXED inside DreamerPolicyHead.
    OUT_INIT: Initializer = Zero,
    # Hidden-layer weight init for ALL networks. Kaiming (He-uniform,
    # gain sqrt(6/fan_in)) is the historical default; the reference uses
    # trunc_normal_in (sigma = sqrt(1/fan_in), sqrt(2) smaller) — pass
    # `TruncNormalIn` for reference parity.
    NET_INIT: Initializer = Kaiming,
](Movable & Deinitable):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime FEAT = Self.DETER + Self.SC

    comptime EncT = Self.ENC
    comptime CoreT = WMCoreGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN, SwishOp,
    ]
    comptime DecT = DecLossGraph[
        Self.SC, Self.DETER, Self.OBS, Self.DEC_U, SwishOp, Self.DEC,
        Self.RECON_SIGMOID,
    ]
    comptime RewT = RewLossGraph[
        Self.DETER, Self.SC, Self.HU, Self.BINS, SwishOp, Self.OUT_INIT
    ]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU, SwishOp]
    comptime ValT = DreamerValue[
        Self.FEAT, Self.VU, Self.BINS, SwishOp, Self.OUT_INIT
    ]
    # discrete (categorical) actor → ACT logits; continuous → 2·ACT (mean,std)
    comptime PolT = DreamerPolicyHead[
        Self.FEAT, Self.PU, Self.ACT, Self.DISCRETE, SwishOp
    ]
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT, SwishOp,
    ]
    # Sequence replay backend, selected by `train_target` via the carry-both
    # `AnySequenceReplay` shim (Mojo has no type-ternary). GPU resolves to the
    # device-resident `GPUSequenceReplay` (device circular storage + device
    # Philox window draws → no per-step host B×T sampling, capture-safe); CPU
    # to the host `SequenceReplay`.
    comptime RepT = AnySequenceReplay[
        Self.train_target, Self.OBS, Self.ACT, Self.CAP
    ]
    comptime StateT = DreamerState[
        Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T,
        Self.T_IMAG,
    ]
    comptime WMBlk = WMStep[
        Self.OBS, Self.ACT, Self.DETER, Self.H, Self.STOCH, Self.CLASSES,
        Self.BLOCKS, Self.TOKEN, Self.DEC_U, Self.HU, Self.BINS, Self.B, Self.T,
        Self.DISCRETE, Self.ENC, Self.DEC, Self.RECON_SIGMOID, Self.OUT_INIT,
    ]
    comptime SyncBlk = ParamSyncStep[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN,
    ]
    comptime ACBlk = ACStep[
        Self.OBS, Self.ACT, Self.DETER, Self.H, Self.STOCH, Self.CLASSES,
        Self.BLOCKS, Self.TOKEN, Self.HU, Self.VU, Self.PU, Self.BINS, Self.B,
        Self.T, Self.T_IMAG, Self.DISCRETE, Self.OUT_INIT,
    ]

    var enc: Self.EncT
    var core: Self.CoreT
    var dec: Self.DecT
    var rew: Self.RewT
    var con: Self.ConT
    var value: Self.ValT
    var slowvalue: Self.ValT
    var policy: Self.PolT
    var imagine: Self.ImagT

    var oe: DreamerOpt
    var ocore: DreamerOpt
    var odec: DreamerOpt
    var orew: DreamerOpt
    var ocon: DreamerOpt
    var oval: DreamerOpt
    var opol: DreamerOpt

    var wm_blk: Self.WMBlk
    var sync_blk: Self.SyncBlk
    var ac_blk: Self.ACBlk

    var replay: Self.RepT
    var retnorm: PercentileNormalize
    var bins: List[Scalar[DT]]
    var state: Self.StateT
    var ctx: Optional[DeviceContext]
    var learning_starts: Int
    var train_steps: Int
    # WM-maturation gate: actor-critic updates (sync + ACStep) are skipped
    # until train_steps >= ac_start. The policy stays at its near-uniform init
    # (outscale 0.01) → diverse on-policy replay while the WM matures, and the
    # actor's first updates see dreams that contain real reward structure.
    # Rationale (Pong forensics): the value head builds a self-fulfilling ridge
    # over the policy's own imagined states within ~1.5-3k updates — long
    # before the WM can even render the agent's paddle — and REINFORCE
    # collapses onto it absorbingly. ac_start lets the WM win that race.
    var ac_start: Int
    var warmup: LinearWarmupSchedule   # reference LR ramp 0→lr over warmup_steps
    # Lazily-captured discrete-GPU train-step graph (Stage 3 P5; `None` until the
    # first capture). Moved into a disjoint local for the `maybe_capture_replay`
    # call (mbpo idiom) so the capture closure can borrow `self` mutably.
    var _train_graph: Optional[CUDAGraph]
    # P4 follow-up: discrete-GPU imagination-noise source. True (default) →
    # on-device Philox in the eager prologue (no host gen/upload). False →
    # host-seeded gen + upload (the CPU↔GPU parity gate, so both targets read
    # IDENTICAL noise — host RNG ≠ device Philox).
    var device_noise: Bool

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](4e-5),
        learning_starts: Int = 200,
        warmup_steps: Int = 1000,
        actent: Scalar[DT] = Scalar[DT](3e-4),
        slowtar: Bool = False,
        device_noise: Bool = True,
        ac_start: Int = 0,
        online: Bool = False,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "DreamerV3Trainer: train_target must be 'cpu' or 'gpu'"
        var enc = Self.EncT.make[Self.train_target, INIT = Self.NET_INIT](ctx=ctx)
        var core = Self.CoreT.make[Self.train_target, INIT = Self.NET_INIT](ctx=ctx)
        var dec = Self.DecT.make[Self.train_target, INIT = Self.NET_INIT](ctx=ctx)
        var rew = Self.RewT.make[Self.train_target, INIT = Self.NET_INIT](ctx=ctx)
        var con = Self.ConT.make[Self.train_target, INIT = Self.NET_INIT](ctx=ctx)
        var value = Self.ValT.make[Self.train_target, INIT = Self.NET_INIT](ctx=ctx)
        var slowvalue = Self.ValT.make[Self.train_target, INIT = Self.NET_INIT](ctx=ctx)
        var policy = Self.PolT.make[Self.train_target, INIT = Self.NET_INIT](ctx=ctx)
        var imagine = Self.ImagT.make[Self.train_target, INIT = Self.NET_INIT](ctx=ctx)

        # Output-head inits are declared STRUCTURALLY in nets.mojo (InitWith):
        # reward + value/slowvalue built with `OUT_INIT` (paper p.6 zero-init
        # by default; Kaiming restores positive-reward optimism for CartPole),
        # and the policy output with the reference's fixed outscale 0.01
        # (ScaledKaiming[1,100]). No post-hoc name-path scaling — a mismatch
        # is now a compile error instead of a silent no-op. slowvalue shares
        # ValT, so it starts neutral too (the value loss regularizes TOWARD
        # slowvalue with slowreg=1, so a non-neutral slowvalue would pull it).

        # Storage DreamerOpt is driven INSIDE the blocks (graph.for_each_param /
        # opt.step[target, M]); the trainer just constructs them with the lr.
        var oe = DreamerOpt(lr=lr)
        var ocore = DreamerOpt(lr=lr)
        var odec = DreamerOpt(lr=lr)
        var orew = DreamerOpt(lr=lr)
        var ocon = DreamerOpt(lr=lr)
        var oval = DreamerOpt(lr=lr)
        var opol = DreamerOpt(lr=lr)

        var bins = List[Scalar[DT]](length=Self.BINS, fill=Scalar[DT](0.0))
        # Reward/value twohot bins, narrowed (max bin ≈ 8102 vs the reference's
        # symexp(20) ≈ 4.85e8). Two reasons for the narrow grid:
        #   (1) huge bins amplify float noise in `Σ softmax·bins` (the e-tail
        #       over a 4.85e8 bin), which breaks CPU↔GPU AC parity;
        #   (2) they turn off-distribution head errors into 1e5–1e6 predictions.
        # This grid MUST equal the grid the reward head is TRAINED on
        # (`TwoHotLoss.make` in wm_loss_ops.mojo). Both now read the SAME
        # `DREAMER_REWARD_GRID_LO` constant (S4) so they can no longer diverge —
        # a past -9-vs-(-20)-default split decoded reward ~5× small, starving
        # imagined returns.
        symexp_twohot_bins[Self.BINS](
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](bins.unsafe_ptr()),
            lo=Scalar[DT](DREAMER_REWARD_GRID_LO),
        )
        var retnorm = PercentileNormalize.make(
            String("perc"), Scalar[DT](0.01), Scalar[DT](5.0),
            Scalar[DT](95.0), Scalar[DT](1.0), False,
        )

        var s = Self(
            enc=enc^, core=core^, dec=dec^, rew=rew^, con=con^, value=value^,
            slowvalue=slowvalue^, policy=policy^, imagine=imagine^,
            oe=oe^, ocore=ocore^, odec=odec^, orew=orew^, ocon=ocon^,
            oval=oval^, opol=opol^,
            wm_blk=Self.WMBlk.make[Self.train_target](ctx=ctx),
            sync_blk=Self.SyncBlk.make[Self.train_target](ctx=ctx),
            ac_blk=Self.ACBlk.make[Self.train_target](
                ctx=ctx, actent=actent, slowtar=slowtar
            ),
            # GPU: device-resident GPUSequenceReplay (sampled straight into the
            # WM device buffers); CPU: host SequenceReplay. The shim picks the
            # backend off `train_target`; the GPU side requires `ctx`.
            replay=Self.RepT.make(ctx=ctx),
            retnorm=retnorm^,
            bins=bins^,
            state=Self.StateT.make[Self.train_target](ctx=ctx),
            ctx=ctx,
            learning_starts=learning_starts,
            train_steps=0,
            ac_start=ac_start,
            warmup=LinearWarmupSchedule.make(lr, warmup_steps),
            _train_graph=None,
            device_noise=device_noise,
        )
        # One-time upload of the CONSTANT twohot bins grid into the AC's device
        # buffer (Stage 3 P5). The grid never changes, so doing it here (not
        # per-step in `_ac_gpu_disc`/`_ac_gpu_cont`) keeps the captured WM+AC
        # region free of H2D copies — which are illegal inside a CUDA-graph
        # capture. Both discrete and continuous device-resident ACs read bins_d.
        comptime if Self.train_target == "gpu":
            for c in range(Self.BINS):
                s.ac_blk.bins_d.data[c] = s.bins[c]
            s.ac_blk.bins_d.upload(ctx.value())
        # Reference replay `online: True`: every fresh T-window is guaranteed
        # into a batch row exactly once, promptly (recency coverage on top of
        # uniform sampling).
        if online:
            s.replay.set_online(Self.T)
        return s^

    def record(
        mut self,
        obs: Pointer[Scalar[DT], MutAnyOrigin],
        act: Pointer[Scalar[DT], MutAnyOrigin],
        reward: Scalar[DT],
        done: Scalar[DT],
    ) raises:
        self.replay.record(obs, act, reward, done)

    def record_terminal(
        mut self, obs: Pointer[Scalar[DT], MutAnyOrigin]
    ) raises:
        """Store a genuine terminal observation (call right after `record(done=1)`).
        Lets the WM continue head learn `latent(terminal)→0`."""
        self.replay.record_terminal(obs)

    def can_train(self) -> Bool:
        return self.replay.count() >= Self.T + 1

    def last_wm_loss(self) -> Scalar[DT]:
        return self.state.last_wm_loss

    def last_ac_loss(self) -> Scalar[DT]:
        return self.state.last_ac_loss

    def dbg_real_rew(self) -> Scalar[DT]:
        return self.state.dbg_real_rew

    def dbg_rew_pred(self) -> Scalar[DT]:
        return self.state.dbg_rew_pred

    def dbg_ret_mean(self) -> Scalar[DT]:
        return self.state.dbg_ret_mean

    def dbg_ret_std(self) -> Scalar[DT]:
        return self.state.dbg_ret_std

    def dbg_pmean_abs(self) -> Scalar[DT]:
        return self.state.dbg_pmean_abs

    def dbg_val_mean(self) -> Scalar[DT]:
        return self.state.dbg_val_mean

    def dbg_pstd(self) -> Scalar[DT]:
        return self.state.dbg_pstd

    def dbg_rscale(self) -> Scalar[DT]:
        return self.state.dbg_rscale

    def dbg_con_mean(self) -> Scalar[DT]:
        return self.state.dbg_con_mean

    def dbg_con_min(self) -> Scalar[DT]:
        return self.state.dbg_con_min

    def dbg_val_std(self) -> Scalar[DT]:
        return self.state.dbg_val_std

    def dbg_feat_std(self) -> Scalar[DT]:
        return self.state.dbg_feat_std

    def dbg_dyn_kl(self) -> Scalar[DT]:
        return self.state.dbg_dyn_kl

    def dbg_rep_kl(self) -> Scalar[DT]:
        return self.state.dbg_rep_kl

    def dbg_obs_loss(self) -> Scalar[DT]:
        return self.state.dbg_obs_loss

    def dbg_rew_loss(self) -> Scalar[DT]:
        return self.state.dbg_rew_loss

    def dbg_con_loss(self) -> Scalar[DT]:
        return self.state.dbg_con_loss

    def dbg_pol_loss(self) -> Scalar[DT]:
        return self.state.dbg_pol_loss

    def dbg_val_loss(self) -> Scalar[DT]:
        return self.state.dbg_val_loss

    def dbg_adv_act(self, a: Int) -> Scalar[DT]:
        """Per-action mean imagination advantage E[adv | sampled action a]
        (discrete; want_diag-refreshed). A steady inter-action gap while eval
        is flat = the collapse driver (model exploitation)."""
        return self.state.dbg_adv_act[a]

    def train_steps_done(self) -> Int:
        return self.train_steps

    def train_step(mut self, want_diag: Bool = True) raises -> Bool:
        if not self.can_train():
            return False
        self.train_prologue(want_diag)
        self._device_step(want_diag)
        self.train_steps += 1
        return True

    def _prof_sync(mut self) raises:
        comptime if Self.train_target == "gpu":
            self.ctx.value().synchronize()

    def profile_sections(mut self) raises -> Array[Float64, 5]:
        """EAGER per-section wall times for ONE train step, in ms:
        [0] draw+noise prologue, [1] WM-BPTT (fwd+bwd+opt), [2] core→imagine
        sync, [3] imagination AC (+Polyak), [4] total. GPU: `synchronize()`
        brackets each section so the wall time is real device time (the Pong
        run is compute-bound — capture ≈ 0 gain — so eager ≈ captured steady
        state). Trains normally (advances `train_steps`); the AC-gate is
        ignored (both sections always run) so profile with ac_start=0."""
        var out = Array[Float64, 5](fill=0.0)
        if not self.can_train():
            return out^
        self._prof_sync()
        var t0 = perf_counter_ns()
        self.train_prologue(want_diag=False)
        self._prof_sync()
        var t1 = perf_counter_ns()
        self.wm_blk.step[Self.train_target, Self.T_IMAG](
            self.state, self.enc, self.core, self.dec, self.rew, self.con,
            self.oe, self.ocore, self.odec, self.orew, self.ocon,
            want_diag=False,
        )
        self._prof_sync()
        var t2 = perf_counter_ns()
        self.sync_blk.step[Self.train_target](
            self.core, self.imagine, ctx=self.ctx
        )
        self._prof_sync()
        var t3 = perf_counter_ns()
        self.ac_blk.step[Self.train_target](
            self.state, self.imagine, self.value, self.slowvalue, self.policy,
            self.rew, self.con, self.oval, self.opol, self.retnorm,
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                self.bins.unsafe_ptr()
            ),
            False,
        )
        self._prof_sync()
        var t4 = perf_counter_ns()
        self.train_steps += 1
        out[0] = Float64(t1 - t0) / 1e6
        out[1] = Float64(t2 - t1) / 1e6
        out[2] = Float64(t3 - t2) / 1e6
        out[3] = Float64(t4 - t3) / 1e6
        out[4] = Float64(t4 - t0) / 1e6
        return out^

    def train_prologue(mut self, want_diag: Bool) raises:
        """Eager per-step work that is NOT part of the captured device-kernel
        sequence (Stage 3 P5): LR refresh + minibatch draw + imagination-noise
        fill/upload. Must run before `_device_step` / `train_device_kernels`
        each step so the (fixed) device input buffers — `wm_blk.mb*_d`,
        `ac_blk.noise_d`, `state.d_*` — hold this step's fresh data before a
        graph replay reads them."""
        # reference LR warmup: ramp 0→lr over warmup_steps (all modules).
        var clr = self.warmup.lr_at(self.train_steps)
        self.oe.lr = clr; self.ocore.lr = clr; self.odec.lr = clr
        self.orew.lr = clr; self.ocon.lr = clr; self.oval.lr = clr
        self.opol.lr = clr
        self._draw_minibatch(want_diag)
        self._fill_noise()

    def _draw_minibatch(mut self, want_diag: Bool) raises:
        """Sample one length-T window into the minibatch buffers the WM/AC
        blocks consume.

        CPU: host `SequenceReplay.sample_batch_fst` → `state.mb_*` (raw host
        pointers).

        GPU: device `GPUSequenceReplay.sample_batch_fst_dev` straight into the
        WM device buffers (`wm_blk.mb*_d`) — no host obs/act gather, no upload
        (the windows are drawn with the buffer's device Philox RNG). rew/dne are
        copied device→device into the shared `state.d_rew/d_cont`, which BOTH the
        discrete (`_ac_gpu_disc`) and continuous (`_ac_gpu_cont`) device-resident
        ACs read directly — no D2H/H2D round-trip. `dbg_real_rew` is the only host
        readout, `want_diag`-gated → the non-diag draw is host-free (capturable)."""
        comptime if Self.train_target == "gpu":
            var sctx = self.ctx.value()
            self.replay.sample_batch_fst_dev[Self.B, Self.T](
                sctx,
                self.wm_blk.mbobs_d.dev.value(),
                self.wm_blk.mbact_d.dev.value(),
                self.wm_blk.mbrew_d.dev.value(),
                self.wm_blk.mbdne_d.dev.value(),
                self.wm_blk.mbfst_d.dev.value(),
            )
            # rew/dne stay on device (st.d_rew / st.d_cont carries dne) for the
            # device-resident AC — BOTH discrete (`_ac_gpu_disc`) and continuous
            # (`_ac_gpu_cont`) read them device-direct, so no D2H/H2D round-trip.
            # `dbg_real_rew` is the only host readout, and it's `want_diag`-gated
            # → the non-diag draw is host-free (capture-safe).
            sctx.enqueue_copy(
                self.state.d_rew.dev.value(),
                self.wm_blk.mbrew_d.dev.value(),
            )
            sctx.enqueue_copy(
                self.state.d_cont.dev.value(),
                self.wm_blk.mbdne_d.dev.value(),
            )
            if want_diag:
                self.wm_blk.mbrew_d.download(sctx)
                var rr: Scalar[DT] = 0.0
                for i in range(Self.B * Self.T):
                    rr += self.wm_blk.mbrew_d.data[i]
                self.state.dbg_real_rew = rr / Scalar[DT](Self.B * Self.T)
        else:
            self.replay.sample_batch_fst[Self.B, Self.T](
                _hp(self.state.mb_obs), _hp(self.state.mb_act),
                _hp(self.state.mb_rew), _hp(self.state.mb_dne),
                _hp(self.state.mb_fst),
            )
            var rr: Scalar[DT] = 0.0
            for i in range(Self.B * Self.T):
                rr += self.state.mb_rew.data[i]
            self.state.dbg_real_rew = rr / Scalar[DT](Self.B * Self.T)

    def load_minibatch(
        mut self,
        obs: Pointer[Scalar[DT], MutAnyOrigin],   # [B*(T+1)*OBS]
        act: Pointer[Scalar[DT], MutAnyOrigin],   # [B*T*ACT]
        rew: Pointer[Scalar[DT], MutAnyOrigin],   # [B*T]
        dne: Pointer[Scalar[DT], MutAnyOrigin],   # [B*T]
        fst: Pointer[Scalar[DT], MutAnyOrigin],   # [B*(T+1)]
    ) raises:
        """Load a FIXED minibatch into the WM/AC buffers, bypassing the replay
        draw. The CPU↔GPU parity gate uses this to feed both backends a
        byte-identical window — their samplers now genuinely differ (host RNG
        vs device Philox), so `train_step`'s own draw can no longer be compared
        directly. Same buffer layout as `_draw_minibatch` fills."""
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        for i in range(Self.B * Self.T):
            self.state.mb_rew.data[i] = rew[unsafe_offset=i]
            self.state.mb_dne.data[i] = dne[unsafe_offset=i]
        comptime if Self.train_target == "gpu":
            var sctx = self.ctx.value()
            for i in range(Self.B * (Self.T + 1) * OBSD):
                self.wm_blk.mbobs_d.data[i] = obs[unsafe_offset=i]
            for i in range(Self.B * Self.T * ACTD):
                self.wm_blk.mbact_d.data[i] = act[unsafe_offset=i]
            for i in range(Self.B * Self.T):
                self.wm_blk.mbrew_d.data[i] = rew[unsafe_offset=i]
                self.wm_blk.mbdne_d.data[i] = dne[unsafe_offset=i]
            for i in range(Self.B * (Self.T + 1)):
                self.wm_blk.mbfst_d.data[i] = fst[unsafe_offset=i]
            # `upload_resident` (stable pointer) so a CUDA-graph that captured
            # these WM input buffers stays valid across replays — only contents
            # change. `upload` recreates the buffer (new pointer) → captured WM
            # reads a stale capture-time minibatch (Stage 3 P5).
            self.wm_blk.mbobs_d.upload_resident(sctx)
            self.wm_blk.mbact_d.upload_resident(sctx)
            self.wm_blk.mbrew_d.upload_resident(sctx)
            self.wm_blk.mbdne_d.upload_resident(sctx)
            self.wm_blk.mbfst_d.upload_resident(sctx)
            # mirror _draw_minibatch: hand rew/dne to the device-resident AC via
            # the shared state buffers (st.d_cont carries dne) — both discrete and
            # continuous read them device-direct.
            sctx.enqueue_copy(
                self.state.d_rew.dev.value(),
                self.wm_blk.mbrew_d.dev.value(),
            )
            sctx.enqueue_copy(
                self.state.d_cont.dev.value(),
                self.wm_blk.mbdne_d.dev.value(),
            )
        else:
            for i in range(Self.B * (Self.T + 1) * OBSD):
                self.state.mb_obs.data[i] = obs[unsafe_offset=i]
            for i in range(Self.B * Self.T * ACTD):
                self.state.mb_act.data[i] = act[unsafe_offset=i]
            for i in range(Self.B * (Self.T + 1)):
                self.state.mb_fst.data[i] = fst[unsafe_offset=i]
        var rr: Scalar[DT] = 0.0
        for i in range(Self.B * Self.T):
            rr += self.state.mb_rew.data[i]
        self.state.dbg_real_rew = rr / Scalar[DT](Self.B * Self.T)

    def _fill_noise(mut self) raises:
        """Imagination sampling noise (eager prologue): NS = T*B starts ×
        T_IMAG steps × ACT.

        Discrete GPU + `device_noise` (default, production): generated ON-DEVICE
        via Philox straight into `ac_blk.noise_d` — no host `random_float64` gen,
        no H2D upload (Stage 3 P4 follow-up). `noise_d` is a fixed buffer the
        captured AC reads, refreshed by this eager kernel each step.

        Else (CPU `_ac_cpu`, GPU-continuous, and the discrete-GPU parity gate
        with `device_noise=False`): host-filled so `_ac_cpu` and the uploaded GPU
        noise read the SAME noise[(t*NS+b)*ACT+a] → CPU↔GPU bit-match; on discrete
        GPU it's `upload_resident`-copied (stable pointer → capture-safe)."""
        comptime if Self.train_target == "gpu":
            if self.device_noise:
                self.ac_blk.gen_noise_device(self.ctx.value())
                return
        for i in range(Self.T_IMAG * Self.T * Self.B * Self.ACT):
            self.state.noise.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        comptime if Self.train_target == "gpu":
            var nctx = self.ctx.value()
            for i in range(Self.T_IMAG * Self.T * Self.B * Self.ACT):
                self.ac_blk.noise_d.data[i] = self.state.noise.data[i]
            self.ac_blk.noise_d.upload_resident(nctx)

    def _device_step(mut self, want_diag: Bool) raises:
        """WM-BPTT → ParamSync → imagination AC over the CURRENTLY-LOADED
        minibatch + noise (filled by `train_prologue` / `load_minibatch`). This
        is the pure compute sequence; on discrete GPU with `want_diag=False` it
        is fully device-resident + sync-free (Stage 3 P1–P4) → the body the
        CUDA-graph capture closure replays. Does NOT advance `train_steps` (the
        caller does, once per logical update, via `train_step` /
        `note_train_update`)."""
        # WM-BPTT → fills the carry (device for discrete, host for continuous)
        # + state.last_wm_loss/dbg (want_diag-gated; the optimizer steps run
        # regardless, so non-diag steps still train — just no loss readout).
        self.wm_blk.step[Self.train_target, Self.T_IMAG](
            self.state, self.enc, self.core, self.dec, self.rew, self.con,
            self.oe, self.ocore, self.odec, self.orew, self.ocon,
            want_diag=want_diag,
        )
        # WM-maturation gate: skip the actor-critic while train_steps <
        # ac_start (policy stays at near-uniform init → diverse replay; the
        # value ridge can't form on an immature WM). The capture path stays
        # EAGER until this gate opens (see train_step_captured), so the
        # captured kernel sequence always includes the AC.
        if self.train_steps < self.ac_start:
            return
        # core/prior → imagine mirror
        self.sync_blk.step[Self.train_target](
            self.core, self.imagine, ctx=self.ctx
        )
        # imagination AC + Polyak → state.last_ac_loss
        self.ac_blk.step[Self.train_target](
            self.state, self.imagine, self.value, self.slowvalue, self.policy,
            self.rew, self.con, self.oval, self.opol, self.retnorm,
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                self.bins.unsafe_ptr()
            ),
            want_diag,
        )

    def _run_minibatch(mut self, want_diag: Bool) raises:
        """Noise fill + device step over an ALREADY-LOADED minibatch, advancing
        the step counter. `train_step` uses `train_prologue` + `_device_step`
        directly; this wrapper is the entry the CPU↔GPU parity gate drives after
        `load_minibatch` (no draw, no LR refresh — just noise + compute)."""
        self._fill_noise()
        self._device_step(want_diag)
        self.train_steps += 1

    # ─── CUDA-graph capture surface (Stage 3 P5) ───────────────────────────
    #
    # The discrete-GPU non-diag step is sync/D2H-free (P1–P4), so its WM+AC
    # device-kernel sequence is capturable. `train_device_kernels` is that
    # sequence with NO host work and NO counter advance — the body of the
    # driver's capture closure. The eager prologue (sample + noise, which read
    # host RNG / use host ring indices) is NOT captured: the caller runs
    # `train_prologue` each step before replay so the fixed device input
    # buffers are refreshed. Mirrors `SACAgent.train_device_kernels`.
    def train_device_kernels(mut self) raises:
        comptime assert Self.train_target == "gpu", (
            "train_device_kernels is the GPU CUDA-graph capture path"
            " (discrete `_ac_gpu_disc` + continuous `_ac_gpu_cont` both qualify —"
            " the non-diag WM+AC step is sync/D2H-free for both)"
        )
        self._device_step(want_diag=False)

    def note_train_update(mut self):
        """Advance one logical update's host counter under graph replay (the
        device work is replayed by the captured graph; loss/dbg scalars are
        only refreshed on the eager want_diag steps)."""
        self.train_steps += 1

    def learning_starts_count(self) -> Int:
        return self.learning_starts

    def train_step_captured(mut self, want_diag: Bool = False) raises -> Bool:
        """CUDA-graph train step for the discrete-GPU path. `want_diag` steps
        take the eager `train_step` (the diagnostic readout — D2H + host sum —
        can't be captured); non-diag steps run the eager prologue (sample +
        noise → fixed device buffers) then capture-once / replay the WM+AC
        device-kernel sequence (`train_device_kernels`). Host counter advances
        via `note_train_update`. On non-NVIDIA `maybe_capture_replay` runs the
        closure directly → identical to `train_step(want_diag=False)` (the
        Apple transparency path; real capture/replay = NVIDIA).

        The `_train_graph` field is moved into a disjoint local for the capture
        call (mbpo idiom) so the closure can borrow `self` mutably without
        overlapping a mut borrow of the field."""
        comptime assert Self.train_target == "gpu", (
            "train_step_captured is the GPU CUDA-graph path (discrete + continuous)"
        )
        if not self.can_train():
            return False
        if want_diag:
            return self.train_step(want_diag=True)
        # CRITICAL (Stage 3 P5): the DreamerOpt update kernel takes `lr` as a
        # HOST scalar arg, so a captured graph FREEZES the lr at capture time.
        # During the LR warmup the per-step lr ramps 0→target, so capturing
        # mid-warmup would pin every replay at the near-zero capture-time lr →
        # the WM/AC barely train → divergence (high WM loss, exploding return
        # scale). Train EAGERLY until warmup completes (lr constant), then
        # capture once the frozen lr == the steady-state lr.
        # Also stay eager through the AC-delay phase (ac_start): the captured
        # graph must include the AC kernel sequence, so capture only once the
        # AC gate is open (and the lr is constant).
        if (
            self.train_steps < self.warmup.warmup_steps
            or self.train_steps < self.ac_start
        ):
            return self.train_step(want_diag=False)
        var ctx = self.ctx.value()
        # Eager prologue: refresh the FIXED device input buffers this step.
        self.train_prologue(want_diag=False)
        var g = self._train_graph^
        self._train_graph = None

        def _captured() capturing raises -> None:
            self.train_device_kernels()

        maybe_capture_replay[_captured](g, ctx)
        self._train_graph = g^
        self.note_train_update()
        return True

    # ─── Checkpoint (ONE file: the whole world model + actor-critic) ───────
    def save_state(mut self, path: String) raises:
        """Write the full DreamerV3 network set into a SINGLE v3 binary file,
        sections name-prefixed per module. `imagine` is NOT saved — it is a
        read-only mirror of `core` re-synced every train_step (and unused by the
        greedy/inference path). Optimizer moments are NOT persisted (resume
        re-warms), matching the SAC checkpoint convention. v3 binary is REQUIRED
        at size200m: the v2 text format was ~5 GB per save and silently
        truncated at the 2 GiB single-write(2) cap."""
        var w = BinaryCheckpointWriter(save_moments=False)
        w.mode = 0
        walk_params[Self.train_target](self.enc, w, self.ctx, "enc")
        walk_params[Self.train_target](self.core, w, self.ctx, "core")
        walk_params[Self.train_target](self.dec, w, self.ctx, "dec")
        walk_params[Self.train_target](self.rew, w, self.ctx, "rew")
        walk_params[Self.train_target](self.con, w, self.ctx, "con")
        walk_params[Self.train_target](self.value, w, self.ctx, "value")
        walk_params[Self.train_target](self.slowvalue, w, self.ctx, "slowvalue")
        walk_params[Self.train_target](self.policy, w, self.ctx, "policy")
        w.mode = 1
        var _sref1 = ParamVisitorRef.of[type_of(w), Self.train_target](w)
        self.enc.for_each_state[Self.train_target](_sref1, self.ctx, "enc")
        var _sref2 = ParamVisitorRef.of[type_of(w), Self.train_target](w)
        self.core.for_each_state[Self.train_target](_sref2, self.ctx, "core")
        var _sref3 = ParamVisitorRef.of[type_of(w), Self.train_target](w)
        self.dec.for_each_state[Self.train_target](_sref3, self.ctx, "dec")
        var _sref4 = ParamVisitorRef.of[type_of(w), Self.train_target](w)
        self.rew.for_each_state[Self.train_target](_sref4, self.ctx, "rew")
        var _sref5 = ParamVisitorRef.of[type_of(w), Self.train_target](w)
        self.con.for_each_state[Self.train_target](_sref5, self.ctx, "con")
        var _sref6 = ParamVisitorRef.of[type_of(w), Self.train_target](w)
        self.value.for_each_state[Self.train_target](_sref6, self.ctx, "value")
        var _sref7 = ParamVisitorRef.of[type_of(w), Self.train_target](w)
        self.slowvalue.for_each_state[Self.train_target](_sref7, self.ctx, "slowvalue")
        var _sref8 = ParamVisitorRef.of[type_of(w), Self.train_target](w)
        self.policy.for_each_state[Self.train_target](_sref8, self.ctx, "policy")
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Restore the full network set from the single envelope (same walk
        order as `save_state`; v3 binary or legacy v2 text). `imagine` is
        re-synced from `core` on the next `train_step`."""
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            rb.mode = 0
            self._load_walk_params(rb)
            rb.mode = 1
            self._load_walk_states(rb)
        else:
            var content: String
            with open(path, "r") as f:
                content = String(f.read())
            var lines = _split_lines(content)
            var body = List[String]()
            for li in range(len(lines)):
                if lines[li].startswith("storage-ckpt"):
                    continue
                body.append(lines[li])
            var r = CheckpointReader(body^)
            r.mode = 0
            self._load_walk_params(r)
            r.mode = 1
            self._load_walk_states(r)
        # Mirror the restored core into `imagine` IMMEDIATELY. `imagine` is not
        # checkpointed; without this, a load followed by any prior rollout
        # (openloop probes, imagination GIFs) before the first train_step would
        # run the dynamics on the random init.
        self.sync_blk.step[Self.train_target](
            self.core, self.imagine, ctx=self.ctx
        )

    def _load_walk_params[V: ParamVisitor](mut self, mut r: V) raises:
        """`load_state` Param walk (same order as `save_state`), shared by the
        v3 and legacy-v2 readers. Caller sets the visitor's mode."""
        self.enc.for_each_param[Self.train_target](r, self.ctx, "enc")
        self.core.for_each_param[Self.train_target](r, self.ctx, "core")
        self.dec.for_each_param[Self.train_target](r, self.ctx, "dec")
        self.rew.for_each_param[Self.train_target](r, self.ctx, "rew")
        self.con.for_each_param[Self.train_target](r, self.ctx, "con")
        self.value.for_each_param[Self.train_target](r, self.ctx, "value")
        self.slowvalue.for_each_param[Self.train_target](r, self.ctx, "slowvalue")
        self.policy.for_each_param[Self.train_target](r, self.ctx, "policy")

    def _load_walk_states[V: ParamVisitor](mut self, mut r: V) raises:
        """`load_state` State walk (same order as `save_state`)."""
        self.enc.for_each_state[Self.train_target](r, self.ctx, "enc")
        self.core.for_each_state[Self.train_target](r, self.ctx, "core")
        self.dec.for_each_state[Self.train_target](r, self.ctx, "dec")
        self.rew.for_each_state[Self.train_target](r, self.ctx, "rew")
        self.con.for_each_state[Self.train_target](r, self.ctx, "con")
        self.value.for_each_state[Self.train_target](r, self.ctx, "value")
        self.slowvalue.for_each_state[Self.train_target](r, self.ctx, "slowvalue")
        self.policy.for_each_state[Self.train_target](r, self.ctx, "policy")

    def openloop_report(
        mut self,
        real_obs: Pointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor+1)*OBS]
        real_act: Pointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor)*ACT] NORMALIZED [-1,1]
        real_rew: Pointer[Scalar[DT], MutAnyOrigin],   # [ctx+hor]
        ctx_len: Int,
        hor: Int,
        out_ol_obs: Pointer[Scalar[DT], MutAnyOrigin],  # [hor] open-loop obs MSE (raw space)
        out_tf_obs: Pointer[Scalar[DT], MutAnyOrigin],  # [hor] teacher-forced obs MSE
        out_ol_rew: Pointer[Scalar[DT], MutAnyOrigin],  # [hor] open-loop |reward err|
        out_tf_rew: Pointer[Scalar[DT], MutAnyOrigin],  # [hor] teacher-forced |reward err|
    ) raises:
        """Open-loop WM-accuracy probe (Finding-4 follow-up diagnostic).

        Builds the posterior belief by OBSERVING `ctx_len` real steps, then rolls
        `hor` steps forward TWO ways and compares decoded predictions to reality:
          * open-loop  — `imagine` (PRIOR dynamics, no obs) fed the real recorded
            actions. Tests whether multi-step dynamics stay accurate.
          * teacher-forced — `core` re-observes each real obs (POSTERIOR). The 1-step
            recon floor (decoder + posterior quality).
        If `out_ol_obs` grows steeply with horizon while `out_tf_obs` stays small,
        the world model's *dynamics* are the bottleneck (capacity / latent
        resolution) — not a code bug, not the decoder. Reward `out_ol_rew` shows
        whether the predicted reward tracks reality along the rollout.

        Uses `imagine` (synced from `core` every train_step) for the prior path, so
        call AFTER training. CPU-only — run with a CPU agent; the WM forward is
        bit-identical CPU↔GPU (parity test), so the diagnosis transfers to GPU runs.
        """
        comptime assert Self.train_target == "cpu", (
            "openloop_report is CPU-only — run the diagnostic with a CPU agent"
        )
        # `imagine` is NOT in the checkpoint (normally re-synced from `core`
        # each train_step) — after a bare `load_state` it still holds its random
        # init, which silently corrupts the open-loop panel. Sync here so the
        # probe is self-contained.
        self.sync_blk.step[Self.train_target](
            self.core, self.imagine, ctx=self.ctx
        )
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime FEATl = Self.FEAT
        comptime BINSl = Self.BINS
        comptime CARRY = 2 + D + SCl
        var bins = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
            self.bins.unsafe_ptr()
        )

        # Belief carry (host Tensors, reused as set_input sources).
        var bd = Tensor.alloc(D)
        var bs = Tensor.alloc(SCl)
        var tok = Tensor.alloc(TOK)
        var pa = Tensor.alloc(ACTD)
        var obt = Tensor.alloc(OBSD)       # obs frame staging (enc input)
        var dummyO = Tensor.alloc(OBSD)    # recon target (unused)
        var dummy1 = Tensor.alloc(1)       # reward target (unused)
        # reusable mutable output Tensors (forward writes the loss/feat slot).
        var carry_t = Tensor.alloc(CARRY)
        var feat_t = Tensor.alloc(FEATl)
        var loss_t = Tensor.alloc(1)

        # ── context: observe ctx_len real steps (prev-action convention) ──
        for t in range(ctx_len):
            if t == 0:
                for k in range(ACTD):
                    pa.data[k] = 0.0
            else:
                for k in range(ACTD):
                    pa.data[k] = real_act[unsafe_offset=(t - 1) * ACTD + k]
            for k in range(OBSD):
                obt.data[k] = real_obs[unsafe_offset=t * OBSD + k]
            self.enc.forward["cpu", 1](
                child_refs[Self.EncT.ARITY, Self.EncT.ACT_DT](obt),
                rebind[TensorImpl[Self.EncT.ACT_DT]](tok),
                None,
            )
            self.core.set_input["deter", 1](bd, None)
            self.core.set_input["stoch", 1](bs, None)
            self.core.set_input["action", 1](pa, None)
            self.core.set_input["tokens", 1](tok, None)
            self.core.forward[1, "cpu"](carry_t, None)
            ref nd = self.core.node_output["nd"]()
            ref sn = self.core.node_output["stoch_new"]()
            for k in range(D):
                bd.data[k] = nd.data[k]
            for k in range(SCl):
                bs.data[k] = sn.data[k]

        # open-loop + teacher-forced belief chains, both seeded from `bd/bs`.
        var old = Tensor.alloc(D); var ols = Tensor.alloc(SCl)
        var tfd = Tensor.alloc(D); var tfs = Tensor.alloc(SCl)
        # staging Tensors for the per-step posterior carry fed back into graphs.
        var ond_t = Tensor.alloc(D); var osn_t = Tensor.alloc(SCl)
        var tnd_t = Tensor.alloc(D); var tsn_t = Tensor.alloc(SCl)
        for k in range(D):
            old.data[k] = bd.data[k]; tfd.data[k] = bd.data[k]
        for k in range(SCl):
            ols.data[k] = bs.data[k]; tfs.data[k] = bs.data[k]

        for h in range(hor):
            var idx = ctx_len - 1 + h
            for k in range(ACTD):
                pa.data[k] = real_act[unsafe_offset=idx * ACTD + k]
            # ── open-loop: prior dynamics (imagine), no observation ──
            self.imagine.set_input["deter", 1](old, None)
            self.imagine.set_input["stoch", 1](ols, None)
            self.imagine.set_input["action", 1](pa, None)
            self.imagine.forward[1, "cpu"](feat_t, None)
            ref ond = self.imagine.node_output["nd"]()
            ref osn = self.imagine.node_output["stoch_new"]()
            for k in range(D):
                ond_t.data[k] = ond.data[k]
            for k in range(SCl):
                osn_t.data[k] = osn.data[k]
            self.dec.set_input["stoch_new", 1](osn_t, None)
            self.dec.set_input["nd", 1](ond_t, None)
            self.dec.set_input["rtgt", 1](dummyO, None)
            self.dec.forward[1, "cpu"](loss_t, None)
            ref pred = self.dec.node_output["dec"]()
            var ms: Scalar[DT] = 0.0
            for k in range(OBSD):
                var dv = _recon_decode[Self.RECON_SIGMOID](pred.data[k]) - real_obs[unsafe_offset=(idx + 1) * OBSD + k]
                ms += dv * dv
            out_ol_obs[unsafe_offset=h] = ms / Scalar[DT](OBSD)
            self.rew.set_input["nd", 1](ond_t, None)
            self.rew.set_input["stoch_new", 1](osn_t, None)
            self.rew.set_input["rtgt", 1](dummy1, None)
            self.rew.forward[1, "cpu"](loss_t, None)
            ref rl = self.rew.node_output["rew"]()
            var re = twohot_pred[BINSl](_hp(rl), 0, bins) - real_rew[unsafe_offset=idx]
            out_ol_rew[unsafe_offset=h] = re if re >= Scalar[DT](0.0) else -re
            for k in range(D):
                old.data[k] = ond_t.data[k]
            for k in range(SCl):
                ols.data[k] = osn_t.data[k]

            # ── teacher-forced: re-observe the real obs (posterior path) ──
            for k in range(OBSD):
                obt.data[k] = real_obs[unsafe_offset=(idx + 1) * OBSD + k]
            self.enc.forward["cpu", 1](
                child_refs[Self.EncT.ARITY, Self.EncT.ACT_DT](obt),
                rebind[TensorImpl[Self.EncT.ACT_DT]](tok),
                None,
            )
            self.core.set_input["deter", 1](tfd, None)
            self.core.set_input["stoch", 1](tfs, None)
            self.core.set_input["action", 1](pa, None)
            self.core.set_input["tokens", 1](tok, None)
            self.core.forward[1, "cpu"](carry_t, None)
            ref tnd = self.core.node_output["nd"]()
            ref tsn = self.core.node_output["stoch_new"]()
            for k in range(D):
                tnd_t.data[k] = tnd.data[k]
            for k in range(SCl):
                tsn_t.data[k] = tsn.data[k]
            self.dec.set_input["stoch_new", 1](tsn_t, None)
            self.dec.set_input["nd", 1](tnd_t, None)
            self.dec.set_input["rtgt", 1](dummyO, None)
            self.dec.forward[1, "cpu"](loss_t, None)
            ref tpred = self.dec.node_output["dec"]()
            var tms: Scalar[DT] = 0.0
            for k in range(OBSD):
                var dv = _recon_decode[Self.RECON_SIGMOID](tpred.data[k]) - real_obs[unsafe_offset=(idx + 1) * OBSD + k]
                tms += dv * dv
            out_tf_obs[unsafe_offset=h] = tms / Scalar[DT](OBSD)
            # teacher-forced reward: head prediction on the REAL posterior state
            # vs real reward — isolates head calibration from dynamics drift.
            self.rew.set_input["nd", 1](tnd_t, None)
            self.rew.set_input["stoch_new", 1](tsn_t, None)
            self.rew.set_input["rtgt", 1](dummy1, None)
            self.rew.forward[1, "cpu"](loss_t, None)
            ref trl = self.rew.node_output["rew"]()
            var tre = twohot_pred[BINSl](_hp(trl), 0, bins) - real_rew[unsafe_offset=idx]
            out_tf_rew[unsafe_offset=h] = tre if tre >= Scalar[DT](0.0) else -tre
            for k in range(D):
                tfd.data[k] = tnd_t.data[k]
            for k in range(SCl):
                tfs.data[k] = tsn_t.data[k]

    def openloop_trace(
        mut self,
        real_obs: Pointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor+1)*OBS]
        real_act: Pointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor)*ACT]
        ctx_len: Int,
        hor: Int,
        out_ol_obs: Pointer[Scalar[DT], MutAnyOrigin],  # [hor*OBS] open-loop decoded obs (raw space)
        out_tf_obs: Pointer[Scalar[DT], MutAnyOrigin],  # [hor*OBS] teacher-forced decoded obs (raw space)
        out_ol_con: Pointer[Scalar[DT], MutAnyOrigin],  # [hor] open-loop continue prob = sigmoid(con logit)
        out_tf_con: Pointer[Scalar[DT], MutAnyOrigin],  # [hor] teacher-forced continue prob
    ) raises:
        """Per-component open-loop WM trace (CartPole fidelity diagnostic).

        Same belief/rollout machinery as `openloop_report`, but instead of an
        aggregate obs MSE it writes the RAW decoded observation vectors (so the
        caller can read individual components, e.g. CartPole's pole angle) plus
        the CONTINUE-head probability along both the open-loop (prior, imagined)
        and teacher-forced (posterior) rollouts. If the open-loop pole angle
        stops tracking the real fall and `ol_con` stays ≈1, imagination does not
        reproduce termination → the model-exploitation gap is a dynamics-fidelity
        problem. CPU-only.
        """
        comptime assert Self.train_target == "cpu", (
            "openloop_trace is CPU-only — run the diagnostic with a CPU agent"
        )
        # Self-contained probe: sync the `imagine` mirror from the trained core
        # (it is not checkpointed — see openloop_report).
        self.sync_blk.step[Self.train_target](
            self.core, self.imagine, ctx=self.ctx
        )
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime FEATl = Self.FEAT
        comptime CARRY = 2 + D + SCl

        var bd = Tensor.alloc(D)
        var bs = Tensor.alloc(SCl)
        var tok = Tensor.alloc(TOK)
        var pa = Tensor.alloc(ACTD)
        var obt = Tensor.alloc(OBSD)
        var dummyO = Tensor.alloc(OBSD)
        var dummy1 = Tensor.alloc(1)
        var carry_t = Tensor.alloc(CARRY)
        var feat_t = Tensor.alloc(FEATl)
        var loss_t = Tensor.alloc(1)

        # ── context: observe ctx_len real steps (prev-action convention) ──
        for t in range(ctx_len):
            if t == 0:
                for k in range(ACTD):
                    pa.data[k] = 0.0
            else:
                for k in range(ACTD):
                    pa.data[k] = real_act[unsafe_offset=(t - 1) * ACTD + k]
            for k in range(OBSD):
                obt.data[k] = real_obs[unsafe_offset=t * OBSD + k]
            self.enc.forward["cpu", 1](
                child_refs[Self.EncT.ARITY, Self.EncT.ACT_DT](obt),
                rebind[TensorImpl[Self.EncT.ACT_DT]](tok),
                None,
            )
            self.core.set_input["deter", 1](bd, None)
            self.core.set_input["stoch", 1](bs, None)
            self.core.set_input["action", 1](pa, None)
            self.core.set_input["tokens", 1](tok, None)
            self.core.forward[1, "cpu"](carry_t, None)
            ref nd = self.core.node_output["nd"]()
            ref sn = self.core.node_output["stoch_new"]()
            for k in range(D):
                bd.data[k] = nd.data[k]
            for k in range(SCl):
                bs.data[k] = sn.data[k]

        var old = Tensor.alloc(D); var ols = Tensor.alloc(SCl)
        var tfd = Tensor.alloc(D); var tfs = Tensor.alloc(SCl)
        var ond_t = Tensor.alloc(D); var osn_t = Tensor.alloc(SCl)
        var tnd_t = Tensor.alloc(D); var tsn_t = Tensor.alloc(SCl)
        for k in range(D):
            old.data[k] = bd.data[k]; tfd.data[k] = bd.data[k]
        for k in range(SCl):
            ols.data[k] = bs.data[k]; tfs.data[k] = bs.data[k]

        for h in range(hor):
            var idx = ctx_len - 1 + h
            for k in range(ACTD):
                pa.data[k] = real_act[unsafe_offset=idx * ACTD + k]
            # ── open-loop: prior dynamics (imagine), no observation ──
            self.imagine.set_input["deter", 1](old, None)
            self.imagine.set_input["stoch", 1](ols, None)
            self.imagine.set_input["action", 1](pa, None)
            self.imagine.forward[1, "cpu"](feat_t, None)
            ref ond = self.imagine.node_output["nd"]()
            ref osn = self.imagine.node_output["stoch_new"]()
            for k in range(D):
                ond_t.data[k] = ond.data[k]
            for k in range(SCl):
                osn_t.data[k] = osn.data[k]
            self.dec.set_input["stoch_new", 1](osn_t, None)
            self.dec.set_input["nd", 1](ond_t, None)
            self.dec.set_input["rtgt", 1](dummyO, None)
            self.dec.forward[1, "cpu"](loss_t, None)
            ref pred = self.dec.node_output["dec"]()
            for k in range(OBSD):
                out_ol_obs[unsafe_offset=h * OBSD + k] = _recon_decode[Self.RECON_SIGMOID](pred.data[k])
            self.con.set_input["nd", 1](ond_t, None)
            self.con.set_input["stoch_new", 1](osn_t, None)
            self.con.set_input["ctgt", 1](dummy1, None)
            self.con.forward[1, "cpu"](loss_t, None)
            ref ocon = self.con.node_output["con"]()
            out_ol_con[unsafe_offset=h] = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-ocon.data[0]))
            for k in range(D):
                old.data[k] = ond_t.data[k]
            for k in range(SCl):
                ols.data[k] = osn_t.data[k]

            # ── teacher-forced: re-observe the real obs (posterior path) ──
            for k in range(OBSD):
                obt.data[k] = real_obs[unsafe_offset=(idx + 1) * OBSD + k]
            self.enc.forward["cpu", 1](
                child_refs[Self.EncT.ARITY, Self.EncT.ACT_DT](obt),
                rebind[TensorImpl[Self.EncT.ACT_DT]](tok),
                None,
            )
            self.core.set_input["deter", 1](tfd, None)
            self.core.set_input["stoch", 1](tfs, None)
            self.core.set_input["action", 1](pa, None)
            self.core.set_input["tokens", 1](tok, None)
            self.core.forward[1, "cpu"](carry_t, None)
            ref tnd = self.core.node_output["nd"]()
            ref tsn = self.core.node_output["stoch_new"]()
            for k in range(D):
                tnd_t.data[k] = tnd.data[k]
            for k in range(SCl):
                tsn_t.data[k] = tsn.data[k]
            self.dec.set_input["stoch_new", 1](tsn_t, None)
            self.dec.set_input["nd", 1](tnd_t, None)
            self.dec.set_input["rtgt", 1](dummyO, None)
            self.dec.forward[1, "cpu"](loss_t, None)
            ref tpred = self.dec.node_output["dec"]()
            for k in range(OBSD):
                out_tf_obs[unsafe_offset=h * OBSD + k] = _recon_decode[Self.RECON_SIGMOID](tpred.data[k])
            self.con.set_input["nd", 1](tnd_t, None)
            self.con.set_input["stoch_new", 1](tsn_t, None)
            self.con.set_input["ctgt", 1](dummy1, None)
            self.con.forward[1, "cpu"](loss_t, None)
            ref tcon = self.con.node_output["con"]()
            out_tf_con[unsafe_offset=h] = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-tcon.data[0]))
            for k in range(D):
                tfd.data[k] = tnd_t.data[k]
            for k in range(SCl):
                tfs.data[k] = tsn_t.data[k]

    def reset_ac(mut self) raises:
        """Re-initialize the ACTOR-CRITIC — value / slowvalue / policy (fresh
        params, hence fresh optimizer moments: storage moments live in the
        Params), fresh opt step counters, fresh return normalizer — while
        KEEPING the trained world model.

        Purpose: pay the WM warmup wall-time ONCE. Load a WM-mature checkpoint,
        call reset_ac(), and train with ac_start=0 → actor-from-step-1
        experiments on a mature WM without re-running ~hours of WM-only
        maturation per iteration (the saved value/policy in such a checkpoint
        are collapsed and must not be reused).

        Call BEFORE any CUDA-graph capture: this swaps module params (new
        device buffers) — a previously captured train graph would replay onto
        stale pointers. In the standard flow (load → reset_ac → train) the
        first capture happens after lr-warmup, well past this call."""
        self.value = Self.ValT.make[Self.train_target, INIT = Self.NET_INIT](
            ctx=self.ctx
        )
        self.slowvalue = Self.ValT.make[Self.train_target, INIT = Self.NET_INIT](
            ctx=self.ctx
        )
        self.policy = Self.PolT.make[Self.train_target, INIT = Self.NET_INIT](
            ctx=self.ctx
        )
        self.oval = DreamerOpt(lr=self.oval.lr)
        self.opol = DreamerOpt(lr=self.opol.lr)
        self.retnorm = PercentileNormalize.make(
            String("perc"), Scalar[DT](0.01), Scalar[DT](5.0),
            Scalar[DT](95.0), Scalar[DT](1.0), False,
        )
        comptime if Self.train_target == "gpu":
            # device-resident retnorm EMA state back to the make() init
            self.ac_blk.retstate_d.data[0] = Scalar[DT](0.0)
            self.ac_blk.retstate_d.data[1] = Scalar[DT](0.0)
            self.ac_blk.retstate_d.upload(self.ctx.value())

    def openloop_heads_gpu(
        mut self,
        real_obs: Pointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor+1)*OBS]
        real_act: Pointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor)*ACT]
        ctx_len: Int,
        hor: Int,
        out_ol_obs: Pointer[Scalar[DT], MutAnyOrigin],   # [hor*OBS] decoded obs
        out_heads: Pointer[Scalar[DT], MutAnyOrigin],    # [hor*3] = [rew_hat, val_hat, con_prob] per step
    ) raises:
        """Open-loop PRIOR rollout with HEAD decodes (exploit forensics).

        Same belief-seeding + prior rollout as `openloop_decode_gpu`'s open-loop
        half (no teacher-forced panel), but per imagined step also decodes what
        the actor actually trains on: the reward head (twohot), the value head
        (twohot), and the continue prob. Rolling this under counterfactual
        held actions exposes WHICH head hands out a spurious advantage
        (value ramp / reward blip / fake terminal) on the collapsed action's
        branch. GPU-only; B=1 marshalling identical to openloop_decode_gpu.
        """
        comptime assert Self.train_target == "gpu", (
            "openloop_heads_gpu is GPU-only"
        )
        # imagine mirror sync (not checkpointed; see openloop_decode_gpu).
        self.sync_blk.step[Self.train_target](
            self.core, self.imagine, ctx=self.ctx
        )
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime FEATl = Self.FEAT
        comptime BINSl = Self.BINS
        comptime CARRY = 2 + D + SCl
        var ctx = self.ctx.value()
        var bins = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
            self.bins.unsafe_ptr()
        )

        var bd = Tensor.make["gpu"](D, self.ctx); bd.ensure(D)
        var bs = Tensor.make["gpu"](SCl, self.ctx); bs.ensure(SCl)
        var pa = Tensor.make["gpu"](ACTD, self.ctx); pa.ensure(ACTD)
        var obt = Tensor.make["gpu"](OBSD, self.ctx); obt.ensure(OBSD)
        var dummyO = Tensor.make["gpu"](OBSD, self.ctx); dummyO.ensure(OBSD)
        var dummy1 = Tensor.make["gpu"](1, self.ctx); dummy1.ensure(1)
        var tok = Tensor.make["gpu"](TOK, self.ctx)
        var carry_t = Tensor.make["gpu"](CARRY, self.ctx)
        var feat_t = Tensor.make["gpu"](FEATl, self.ctx); feat_t.ensure(FEATl)
        var loss_t = Tensor.make["gpu"](1, self.ctx)
        var vscr = Tensor.make["gpu"](BINSl, self.ctx)
        for k in range(OBSD):
            dummyO.data[k] = 0.0
        dummyO.upload(ctx)
        dummy1.data[0] = 0.0
        dummy1.upload(ctx)
        for k in range(D):
            bd.data[k] = 0.0
        for k in range(SCl):
            bs.data[k] = 0.0

        # ── context: observe ctx_len real steps (prev-action convention) ──
        for t in range(ctx_len):
            if t == 0:
                for k in range(ACTD):
                    pa.data[k] = 0.0
            else:
                for k in range(ACTD):
                    pa.data[k] = real_act[unsafe_offset=(t - 1) * ACTD + k]
            for k in range(OBSD):
                obt.data[k] = real_obs[unsafe_offset=t * OBSD + k]
            obt.upload(ctx); bd.upload(ctx); bs.upload(ctx); pa.upload(ctx)
            self.enc.forward["gpu", 1](
                child_refs[Self.EncT.ARITY, Self.EncT.ACT_DT](obt),
                rebind[TensorImpl[Self.EncT.ACT_DT]](tok),
                self.ctx,
            )
            self.core.set_input["deter", 1](bd, self.ctx)
            self.core.set_input["stoch", 1](bs, self.ctx)
            self.core.set_input["action", 1](pa, self.ctx)
            self.core.set_input["tokens", 1](tok, self.ctx)
            self.core.forward[1, "gpu"](carry_t, self.ctx)
            ref nd = self.core.node_output["nd"]()
            ref sn = self.core.node_output["stoch_new"]()
            nd.download(ctx); sn.download(ctx)
            for k in range(D):
                bd.data[k] = nd.data[k]
            for k in range(SCl):
                bs.data[k] = sn.data[k]

        var old = Tensor.make["gpu"](D, self.ctx); old.ensure(D)
        var ols = Tensor.make["gpu"](SCl, self.ctx); ols.ensure(SCl)
        var ond_t = Tensor.make["gpu"](D, self.ctx); ond_t.ensure(D)
        var osn_t = Tensor.make["gpu"](SCl, self.ctx); osn_t.ensure(SCl)
        for k in range(D):
            old.data[k] = bd.data[k]
        for k in range(SCl):
            ols.data[k] = bs.data[k]

        for h in range(hor):
            var idx = ctx_len - 1 + h
            for k in range(ACTD):
                pa.data[k] = real_act[unsafe_offset=idx * ACTD + k]
            pa.upload(ctx)
            old.upload(ctx); ols.upload(ctx)
            self.imagine.set_input["deter", 1](old, self.ctx)
            self.imagine.set_input["stoch", 1](ols, self.ctx)
            self.imagine.set_input["action", 1](pa, self.ctx)
            self.imagine.forward[1, "gpu"](feat_t, self.ctx)
            ref ond = self.imagine.node_output["nd"]()
            ref osn = self.imagine.node_output["stoch_new"]()
            ond.download(ctx); osn.download(ctx)
            for k in range(D):
                ond_t.data[k] = ond.data[k]
            for k in range(SCl):
                osn_t.data[k] = osn.data[k]
            ond_t.upload(ctx); osn_t.upload(ctx)
            # decoder → pixels
            self.dec.set_input["stoch_new", 1](osn_t, self.ctx)
            self.dec.set_input["nd", 1](ond_t, self.ctx)
            self.dec.set_input["rtgt", 1](dummyO, self.ctx)
            self.dec.forward[1, "gpu"](loss_t, self.ctx)
            ref pred = self.dec.node_output["dec"]()
            pred.download(ctx)
            for k in range(OBSD):
                out_ol_obs[unsafe_offset=h * OBSD + k] = _recon_decode[Self.RECON_SIGMOID](
                    pred.data[k]
                )
            # reward head (twohot decode)
            self.rew.set_input["nd", 1](ond_t, self.ctx)
            self.rew.set_input["stoch_new", 1](osn_t, self.ctx)
            self.rew.set_input["rtgt", 1](dummy1, self.ctx)
            self.rew.forward[1, "gpu"](loss_t, self.ctx)
            ref rl = self.rew.node_output["rew"]()
            rl.download(ctx)
            out_heads[unsafe_offset=h * 3 + 0] = twohot_pred[BINSl](_hp(rl), 0, bins)
            # value head (twohot decode) on feat = [nd, stoch_new]
            for k in range(D):
                feat_t.data[k] = ond_t.data[k]
            for k in range(SCl):
                feat_t.data[D + k] = osn_t.data[k]
            feat_t.upload(ctx)
            self.value.forward["gpu", 1](
                TensorRefs[1](feat_t), vscr, self.ctx
            )
            vscr.download(ctx)
            out_heads[unsafe_offset=h * 3 + 1] = twohot_pred[BINSl](_hp(vscr), 0, bins)
            # continue head → prob
            self.con.set_input["nd", 1](ond_t, self.ctx)
            self.con.set_input["stoch_new", 1](osn_t, self.ctx)
            self.con.set_input["ctgt", 1](dummy1, self.ctx)
            self.con.forward[1, "gpu"](loss_t, self.ctx)
            ref ocon = self.con.node_output["con"]()
            ocon.download(ctx)
            out_heads[unsafe_offset=h * 3 + 2] = Scalar[DT](1.0) / (
                Scalar[DT](1.0) + exp(-ocon.data[0])
            )
            for k in range(D):
                old.data[k] = ond_t.data[k]
            for k in range(SCl):
                ols.data[k] = osn_t.data[k]

    def openloop_decode_gpu(
        mut self,
        real_obs: Pointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor+1)*OBS]
        real_act: Pointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor)*ACT]
        ctx_len: Int,
        hor: Int,
        out_ol_obs: Pointer[Scalar[DT], MutAnyOrigin],  # [hor*OBS] open-loop decoded obs (raw space)
        out_tf_obs: Pointer[Scalar[DT], MutAnyOrigin],  # [hor*OBS] teacher-forced decoded obs (raw space)
    ) raises:
        """GPU twin of `openloop_trace`'s decode path (imagination-GIF probe).

        Seeds the posterior belief from `ctx_len` real frames (encode→core), then
        rolls `hor` steps two ways and writes the RAW decoded observation vectors:

          * `out_ol_obs` — OPEN-LOOP: prior dynamics (`imagine`) fed the real
            recorded actions but NO observations — what imagination actually
            decodes to ("Dreamer's dream").
          * `out_tf_obs` — TEACHER-FORCED: re-observes each real obs (posterior).
            The decode upper bound (separates decoder fidelity from dynamics drift).

        Unlike `openloop_trace` (CPU host reads) this runs the enc/core/imagine/dec
        forwards on-device (reusing the LIVE training GPU graphs, B=1 — buffers are
        grow-only so B=1 shares the B/NS training instances, exactly like
        `select_action`), and D2Hs only the small per-step decoded frame into the
        host out buffers. GPU-only; `select_action` marshalling idiom throughout.
        """
        comptime assert Self.train_target == "gpu", (
            "openloop_decode_gpu is GPU-only — use openloop_trace on CPU"
        )
        # Self-contained probe: sync the `imagine` mirror from the trained core.
        # `imagine` is NOT checkpointed (normally re-synced each train_step), so
        # a bare `load_state` → openloop_decode_gpu would roll the prior with a
        # RANDOM dynamics net — the GIF's "IMAGINED" panel dissolves to the mean
        # image while RECON stays sharp, misdiagnosing the world model.
        self.sync_blk.step[Self.train_target](
            self.core, self.imagine, ctx=self.ctx
        )
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime FEATl = Self.FEAT
        comptime CARRY = 2 + D + SCl
        var ctx = self.ctx.value()

        # device staging tensors (B=1). `.ensure(n)` sizes the host `.data` List
        # for the ones we fill host-side before upload; pure outputs skip it.
        var bd = Tensor.make["gpu"](D, self.ctx); bd.ensure(D)
        var bs = Tensor.make["gpu"](SCl, self.ctx); bs.ensure(SCl)
        var pa = Tensor.make["gpu"](ACTD, self.ctx); pa.ensure(ACTD)
        var obt = Tensor.make["gpu"](OBSD, self.ctx); obt.ensure(OBSD)
        var dummyO = Tensor.make["gpu"](OBSD, self.ctx); dummyO.ensure(OBSD)
        var tok = Tensor.make["gpu"](TOK, self.ctx)
        var carry_t = Tensor.make["gpu"](CARRY, self.ctx)
        var feat_t = Tensor.make["gpu"](FEATl, self.ctx)
        var loss_t = Tensor.make["gpu"](1, self.ctx)

        # rtgt is required by the dec loss graph but the "dec" node (prediction)
        # does not depend on it — zero it once.
        for k in range(OBSD):
            dummyO.data[k] = 0.0
        dummyO.upload(ctx)
        for k in range(D):
            bd.data[k] = 0.0
        for k in range(SCl):
            bs.data[k] = 0.0

        # ── context: observe ctx_len real steps (prev-action convention) ──
        for t in range(ctx_len):
            if t == 0:
                for k in range(ACTD):
                    pa.data[k] = 0.0
            else:
                for k in range(ACTD):
                    pa.data[k] = real_act[unsafe_offset=(t - 1) * ACTD + k]
            for k in range(OBSD):
                obt.data[k] = real_obs[unsafe_offset=t * OBSD + k]
            obt.upload(ctx); bd.upload(ctx); bs.upload(ctx); pa.upload(ctx)
            self.enc.forward["gpu", 1](
                child_refs[Self.EncT.ARITY, Self.EncT.ACT_DT](obt),
                rebind[TensorImpl[Self.EncT.ACT_DT]](tok),
                self.ctx,
            )
            self.core.set_input["deter", 1](bd, self.ctx)
            self.core.set_input["stoch", 1](bs, self.ctx)
            self.core.set_input["action", 1](pa, self.ctx)
            self.core.set_input["tokens", 1](tok, self.ctx)
            self.core.forward[1, "gpu"](carry_t, self.ctx)
            ref nd = self.core.node_output["nd"]()
            ref sn = self.core.node_output["stoch_new"]()
            nd.download(ctx); sn.download(ctx)
            for k in range(D):
                bd.data[k] = nd.data[k]
            for k in range(SCl):
                bs.data[k] = sn.data[k]

        var old = Tensor.make["gpu"](D, self.ctx); old.ensure(D)
        var ols = Tensor.make["gpu"](SCl, self.ctx); ols.ensure(SCl)
        var tfd = Tensor.make["gpu"](D, self.ctx); tfd.ensure(D)
        var tfs = Tensor.make["gpu"](SCl, self.ctx); tfs.ensure(SCl)
        var ond_t = Tensor.make["gpu"](D, self.ctx); ond_t.ensure(D)
        var osn_t = Tensor.make["gpu"](SCl, self.ctx); osn_t.ensure(SCl)
        var tnd_t = Tensor.make["gpu"](D, self.ctx); tnd_t.ensure(D)
        var tsn_t = Tensor.make["gpu"](SCl, self.ctx); tsn_t.ensure(SCl)
        for k in range(D):
            old.data[k] = bd.data[k]; tfd.data[k] = bd.data[k]
        for k in range(SCl):
            ols.data[k] = bs.data[k]; tfs.data[k] = bs.data[k]

        for h in range(hor):
            var idx = ctx_len - 1 + h
            for k in range(ACTD):
                pa.data[k] = real_act[unsafe_offset=idx * ACTD + k]
            pa.upload(ctx)
            # ── open-loop: prior dynamics (imagine), no observation ──
            old.upload(ctx); ols.upload(ctx)
            self.imagine.set_input["deter", 1](old, self.ctx)
            self.imagine.set_input["stoch", 1](ols, self.ctx)
            self.imagine.set_input["action", 1](pa, self.ctx)
            self.imagine.forward[1, "gpu"](feat_t, self.ctx)
            ref ond = self.imagine.node_output["nd"]()
            ref osn = self.imagine.node_output["stoch_new"]()
            ond.download(ctx); osn.download(ctx)
            for k in range(D):
                ond_t.data[k] = ond.data[k]
            for k in range(SCl):
                osn_t.data[k] = osn.data[k]
            ond_t.upload(ctx); osn_t.upload(ctx)
            self.dec.set_input["stoch_new", 1](osn_t, self.ctx)
            self.dec.set_input["nd", 1](ond_t, self.ctx)
            self.dec.set_input["rtgt", 1](dummyO, self.ctx)
            self.dec.forward[1, "gpu"](loss_t, self.ctx)
            ref pred = self.dec.node_output["dec"]()
            pred.download(ctx)
            for k in range(OBSD):
                out_ol_obs[unsafe_offset=h * OBSD + k] = _recon_decode[Self.RECON_SIGMOID](pred.data[k])
            for k in range(D):
                old.data[k] = ond_t.data[k]
            for k in range(SCl):
                ols.data[k] = osn_t.data[k]

            # ── teacher-forced: re-observe the real obs (posterior path) ──
            for k in range(OBSD):
                obt.data[k] = real_obs[unsafe_offset=(idx + 1) * OBSD + k]
            obt.upload(ctx)
            self.enc.forward["gpu", 1](
                child_refs[Self.EncT.ARITY, Self.EncT.ACT_DT](obt),
                rebind[TensorImpl[Self.EncT.ACT_DT]](tok),
                self.ctx,
            )
            tfd.upload(ctx); tfs.upload(ctx)
            self.core.set_input["deter", 1](tfd, self.ctx)
            self.core.set_input["stoch", 1](tfs, self.ctx)
            self.core.set_input["action", 1](pa, self.ctx)
            self.core.set_input["tokens", 1](tok, self.ctx)
            self.core.forward[1, "gpu"](carry_t, self.ctx)
            ref tnd = self.core.node_output["nd"]()
            ref tsn = self.core.node_output["stoch_new"]()
            tnd.download(ctx); tsn.download(ctx)
            for k in range(D):
                tnd_t.data[k] = tnd.data[k]
            for k in range(SCl):
                tsn_t.data[k] = tsn.data[k]
            tnd_t.upload(ctx); tsn_t.upload(ctx)
            self.dec.set_input["stoch_new", 1](tsn_t, self.ctx)
            self.dec.set_input["nd", 1](tnd_t, self.ctx)
            self.dec.set_input["rtgt", 1](dummyO, self.ctx)
            self.dec.forward[1, "gpu"](loss_t, self.ctx)
            ref tpred = self.dec.node_output["dec"]()
            tpred.download(ctx)
            for k in range(OBSD):
                out_tf_obs[unsafe_offset=h * OBSD + k] = _recon_decode[Self.RECON_SIGMOID](tpred.data[k])
            for k in range(D):
                tfd.data[k] = tnd_t.data[k]
            for k in range(SCl):
                tfs.data[k] = tsn_t.data[k]
