"""Dreamer 4 ONLINE training driver (facade-takes-tokenizer).

The offline pipeline (`examples/dreamer4/pong_reward_end2end.mojo`) trains the
tokenizer + agent on a pre-collected buffer. This driver instead collects the
data ONLINE from a live `BoxDiscreteActionEnv`, in three stages:

  Stage 0 — warmup collect : random actions fill the ring buffer to `learn_start`.
  Stage 1 — tokenizer pretrain : the Phase-A MAE loop on the warmup buffer, then
            FREEZE the tokenizer (`set_mae_p(0,0)`).
  Stage 2 — online RL loop : act with the policy (a clean-latent forward), step
            the env, append to the ring buffer; every `train_every` steps train
            the action-conditioned world model + reward/continue heads
            (`acwm_train_step` + continue BCE), and every `imag_every` steps run
            an imagination-RL update (`imag_train_step`) on the frozen WM.

`run_online_dreamer4` is a free function (not a method) so it can be parameterized
by the full agent + tokenizer + image dims without bloating `Dreamer4Agent`. The
tokenizer is passed as an argument (its dims would explode the agent's comptime
param list). `DYN_TARGET` selects the WM compute target: "cpu" (default) runs the
whole agent on host; "gpu" runs the dynamics transformer on device (via
`acwm_train_step_gpu` + the GPU imagination rollout), stepping the device dynamics
and the host heads/task-embedder as separate submodules — pass a `DeviceContext`
as `dctx`. Tokenizer + perceptual backbone stay on host either way.

This is the driver the online CarRacing lighthouse calls; a stub-env smoke gate
lives in `tests/nn/test_dreamer4_train_online.mojo`.
"""

from noeira.nn.core.param import walk_params
from std.os.path import exists
from std.math import sqrt, log, cos
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.call import call_forward
from noeira.nn.optimizer.adam import Adam
from noeira.core.env_traits import BoxDiscreteActionEnv
from noeira.core.logger import Logger

from .agent import Dreamer4Agent
from .tokenizer import Dreamer4Tokenizer
from .frame_buffer import Dreamer4FrameBuffer
from .recon_loss import masked_recon_loss, masked_recon_grad_gpu
from .perceptual_loss import (
    masked_recon_plus_perceptual_loss, masked_recon_plus_perceptual_loss_gpu,
)
from .patchify import downscale_box, temporal_patchify
from .imag_rl_loss import continue_bce_backward
from .shortcut_loss import _mao, _ilog2
from ..dreamerv3.twohot import symexp_twohot_bins
from ...nn.models.cifar_feature_net import CifarBackbone


struct OnlineRng(Copyable, Movable):
    """Deterministic xorshift64* + Box-Muller (mirrors the end2end example)."""

    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed | 1

    def u64(mut self) -> UInt64:
        var x = self.s
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.s = x
        return x * 0x2545F4914F6CDD1D

    def uniform(mut self) -> Float64:
        return Float64(self.u64() >> 11) * (1.0 / 9007199254740992.0)

    def gauss(mut self) -> Float64:
        var u1 = self.uniform()
        var u2 = self.uniform()
        if u1 < 1e-12:
            u1 = 1e-12
        return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)


def _window_to_patches[
    B: Int, T: Int, IN_CH: Int, IMG: Int, TGT: Int, PATCH: Int
](
    pix: Pointer[Scalar[DT], MutAnyOrigin],     # [B*T*(IN_CH*IMG*IMG)]
    frames: Pointer[Scalar[DT], MutAnyOrigin],  # [B*T*TGT*TGT] scratch
    patches: Pointer[Scalar[DT], MutAnyOrigin], # [B*T*NP*DP]
) raises:
    """Downscale the LATEST stacked frame of each (b,t) obs to TGT×TGT grayscale
    and patchify (mirrors end2end `_window_to_patches`)."""
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime BATCH = B * T
    for b in range(B):
        for t in range(T):
            var bt = b * T + t
            var fsrc = pix.unsafe_offset(bt * IMG_DIM + (IN_CH - 1) * IMG * IMG)
            downscale_box[IMG, IMG, TGT, TGT](fsrc, frames.unsafe_offset(bt * TGT * TGT))
    temporal_patchify[BATCH, 1, TGT, TGT, PATCH](frames, patches)


def _frame_to_patches[
    IN_CH: Int, IMG: Int, TGT: Int, PATCH: Int
](
    obs: Pointer[Scalar[DT], MutAnyOrigin],     # [IN_CH*IMG*IMG] single obs
    frame: Pointer[Scalar[DT], MutAnyOrigin],   # [TGT*TGT] scratch
    patches: Pointer[Scalar[DT], MutAnyOrigin], # [NP*DP] out
) raises:
    """Downscale + patchify ONE obs frame (latest stacked channel)."""
    var fsrc = obs.unsafe_offset((IN_CH - 1) * IMG * IMG)
    downscale_box[IMG, IMG, TGT, TGT](fsrc, frame)
    temporal_patchify[1, 1, TGT, TGT, PATCH](frame, patches)


def _push_frame[
    IN_CH: Int, IMG: Int, TGT: Int, PATCH: Int, T: Int, NP: Int, DP: Int
](
    cur: Pointer[Scalar[DT], MutAnyOrigin],   # [IN_CH*IMG*IMG]
    mut fr1: Tensor,                                # [TGT*TGT] scratch
    mut pa1: Tensor,                                # [NP*DP] scratch
    mut win_patch: Tensor,                          # [T*NP*DP]
    mut win_act: List[Int],                         # [T]
    win_n: Int,
    a_prev: Int,
) raises -> Int:
    """Downscale+patchify `cur` into the front-aligned rolling window (slide left
    when full). Returns the new window length."""
    _frame_to_patches[IN_CH, IMG, TGT, PATCH](
        cur, _mao(fr1.data.unsafe_ptr()), _mao(pa1.data.unsafe_ptr())
    )
    var f: Int
    var new_n = win_n
    if win_n < T:
        f = win_n
        new_n = win_n + 1
    else:
        for i in range(T - 1):
            for k in range(NP * DP):
                win_patch.data[i * NP * DP + k] = win_patch.data[
                    (i + 1) * NP * DP + k
                ]
            win_act[i] = win_act[i + 1]
        f = T - 1
    for k in range(NP * DP):
        win_patch.data[f * NP * DP + k] = pa1.data[k]
    win_act[f] = a_prev
    return new_n


def _default_collect[
    E: BoxDiscreteActionEnv
](mut env: E, step: Int) capturing raises -> Int:
    """No-op collect action (never called when SCRIPTED=False); the default so
    the `COLLECT_ACTION` param is optional. Overridden with a real driver in
    offline mode."""
    return 0


def _step_repeat[
    E: BoxDiscreteActionEnv, IMG_DIM: Int
](
    mut env: E, action: Int, repeat: Int, mut out_obs: List[Scalar[DT]]
) raises -> Tuple[Float64, Bool]:
    """Apply `action` for `repeat` env steps (action repeat / frame skip),
    accumulating reward and stopping early on `done`. Fills `out_obs` with the
    LAST observation and returns (summed_reward, done). One agent decision → one
    buffer transition covering `repeat` env steps (standard CarRacing setup)."""
    var r_sum: Float64 = 0.0
    var d = False
    var reps = repeat if repeat > 0 else 1
    for _ in range(reps):
        var res = env.step_obs(action)
        r_sum += Float64(res[1])
        d = res[2]
        for i in range(IMG_DIM):
            out_obs[i] = Scalar[DT](Float64(res[0][i]))
        if d:
            break
    return (r_sum, d)


def _mmm(
    p: Pointer[Scalar[DT], MutAnyOrigin], n: Int
) -> Tuple[Float64, Float64, Float64]:
    """(mean, min, max) over p[0:n]."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    var s: Float64 = 0.0
    var mn = Float64(p[unsafe_offset=0])
    var mx = mn
    for i in range(n):
        var v = Float64(p[unsafe_offset=i])
        s += v
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    return (s / Float64(n), mn, mx)


def _build_shortcut_schedule[
    B: Int, T: Int, B_SELF: Int, KMAX: Int, EMAX: Int, ND: Int
](
    mut rng: OnlineRng,
    sigma: Pointer[Scalar[DT], MutAnyOrigin],    # [B*T]
    sig_idx: Pointer[Scalar[DT], MutAnyOrigin],  # [B*T]
    step_idx: Pointer[Scalar[DT], MutAnyOrigin], # [B*T]
    z0: Pointer[Scalar[DT], MutAnyOrigin],       # [B*T*ND]
):
    """The shortcut-forcing per-(b,t) signal-level / step-size schedule + the
    τ=0 Gaussian noise seed (lifted from end2end Phase B)."""
    comptime B_EMP = B - B_SELF
    for b in range(B):
        var is_self = b >= B_EMP
        for t in range(T):
            var bt = b * T + t
            var stp = EMAX
            if is_self:
                stp = Int(rng.uniform() * Float64(EMAX))
            var K = 1 << stp
            var j = Int(rng.uniform() * Float64(K))
            if j >= K:
                j = K - 1
            var scale = KMAX // K
            sigma[unsafe_offset=bt] = Scalar[DT](Float64(j) / Float64(K))
            sig_idx[unsafe_offset=bt] = Scalar[DT](Float64(j * scale))
            step_idx[unsafe_offset=bt] = Scalar[DT](Float64(stp))
    for i in range(B * T * ND):
        z0[unsafe_offset=i] = Scalar[DT](rng.gauss())


@always_inline
def _dptr(mut t: Tensor) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Raw device pointer of a device-resident Tensor (recon-grad kernel ABI)."""
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        t.dev.value().unsafe_ptr()
    )


def _encode[
    M: Module, //, TB: Int, TARGET: StaticString
](
    mut enc: M,
    mut patches: Tensor,     # [TB * IN] host-filled
    mut z_out: Tensor,       # [TB * OUT] latents (left on HOST for the agent)
    dctx: Optional[DeviceContext],
) raises:
    """Encode patches → latents on TARGET. CPU: direct forward. GPU: upload the
    host patches, run the device encoder, download the latents back to host (the
    agent methods — act/acwm/imag — consume host-resident z)."""
    comptime IN = M.IN_DIMS[0]
    comptime OUT = M.OUT_DIM
    comptime if TARGET == "cpu":
        call_forward["cpu", TB](enc, TensorRefs[M.ARITY](patches), z_out, None)
    else:
        var c = dctx.value()
        patches.ensure_gpu(c, TB * IN)
        patches.upload(c)
        z_out.ensure_gpu(c, TB * OUT)
        call_forward["gpu", TB](
            enc, TensorRefs[M.ARITY](patches), z_out, Optional(c)
        )
        z_out.download(c)


def run_online_dreamer4[
    # ── agent dims (same order as Dreamer4Agent) ──
    DSP: Int, NSP: Int, D: Int, NH: Int, T: Int, NREG: Int, HID: Int,
    DEPTH: Int, KMAX: Int, NAGENT: Int, NTASK: Int, HHID: Int, NACT: Int,
    NBINS: Int, NMTP: Int, B: Int, B_SELF: Int, USE_MAX: Bool, ADIM: Int,
    AHID: Int, K_IMAG: Int, NCTX: Int,
    # ── env / logger ──
    E: BoxDiscreteActionEnv, L: Logger,
    # ── image / tokenizer / buffer dims ──
    IN_CH: Int, IMG: Int, TGT: Int, PATCH: Int, TNP: Int, CAP: Int,
    TOK_D: Int, TOK_NH: Int, TOK_HID: Int, TOK_DEPTH: Int,
    TOK_PMIN: Float64, TOK_PMAX: Float64, TOK_SEED: UInt64,
    # "cpu" (default) or "gpu" — runs the dynamics transformer on device (the
    # heavy WM compute); heads + tokenizer + backbone stay on host regardless.
    DYN_TARGET: StaticString = "cpu",
    # OFFLINE-validation mode: when True, BOTH warmup and ongoing collection pick
    # actions via `collect_action` (e.g. a scripted decent driver) instead of
    # random / the learned policy, so the WM + BC heads train on GOOD data. The
    # policy is still trained ONLY in imagination and greedy-eval'd. This is the
    # paper's offline setting (fixed decent dataset), the setting Dreamer 4 is
    # designed for — vs online-from-scratch where random rewards make PMPO's
    # sign-of-advantage signal pure noise.
    SCRIPTED: Bool = False,
    # Action provider for SCRIPTED mode (comptime fn param, like other drivers).
    # Typed over env E; (env, step) → discrete action. Ignored when SCRIPTED=False.
    COLLECT_ACTION: def (mut E, Int) capturing raises -> Int = _default_collect[E],
](
    mut agent: Dreamer4Agent[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, NAGENT, NTASK, HHID,
        NACT, NBINS, NMTP, B, B_SELF, USE_MAX, ADIM, AHID, K_IMAG, NCTX,
        DYN_TARGET,
    ],
    mut tok: Dreamer4Tokenizer[
        PATCH * PATCH, TOK_D, TOK_NH, T, NSP,
        TNP, DSP, TOK_HID, TOK_DEPTH,
        TOK_PMIN, TOK_PMAX, TOK_SEED,
    ],
    # Frozen perceptual backbone for the tokenizer's MSE+w·perceptual recon loss
    # (paper eq. 5). Unused when `perc_weight == 0` — pass any instance then.
    mut backbone: CifarBackbone[TGT, TGT],
    mut env: E,
    mut logger: L,
    warmup_steps: Int,
    tok_pretrain_steps: Int,
    total_env_steps: Int,
    train_every: Int,
    imag_every: Int,
    eval_every: Int,
    lr_tok: Scalar[DT] = Scalar[DT](2e-3),
    lr_agent: Scalar[DT] = Scalar[DT](1e-3),
    lr_cont: Scalar[DT] = Scalar[DT](3e-3),
    lr_imag: Scalar[DT] = Scalar[DT](3e-4),   # value + policy heads in imagination.
                            # Was 1e-2 (10× lr_agent) — far above the DreamerV3
                            # critic range (~3e-4); the value head is the ONLY
                            # imagination-trained head, regressed via twohot-CE onto
                            # its OWN bootstrapped λ-return with no slow-target/
                            # normalization anchor, so 1e-2 drove it into an input-
                            # insensitive constant collapse (dead advantage → no
                            # policy learning). 3e-4 = standard critic lr.
    perc_weight: Float64 = 0.0,
    eval_max_steps: Int = 1000,
    num_eval_episodes: Int = 1,   # average the greedy-eval return over this many
                            # episodes. CarRacing draws a NEW random track each
                            # reset, so a single greedy episode is dominated by
                            # track-seed noise; average 5-10 to see the real trend.
    imag_gamma: Scalar[DT] = Scalar[DT](0.997),
    value_bin_lo: Scalar[DT] = Scalar[DT](-9.0),   # symexp value/reward grid lower
                            # bound: bins span ±symexp(|lo|). Narrow it (e.g. -5/-6)
                            # for small-bounded-reward envs so the 41 bins get
                            # resolution where values live AND the value is
                            # physically bounded (prevents the TD critic drifting
                            # onto far bins with a short imagination horizon).
    log_every: Int = 500,   # cadence for stdout progress + metric logging
    frame_repeat: Int = 1,  # repeat each chosen action this many env steps
    diag: Bool = False,     # print reward/value/return sanity stats at log cadence
    save_ckpt: String = "", # if non-empty, base path for a params checkpoint of
                            # the tokenizer + agent (written every eval + at the
                            # end): `<base>.tok.ckpt` + `<base>.{dyn,te,ph,rh,vh,
                            # ch}.ckpt`. Load with the matching dims via
                            # `tok`/`agent` `load` (see the imagination-GIF example).
    seed: UInt64 = 20260701,
    dctx: Optional[DeviceContext] = None,   # required when DYN_TARGET="gpu"
) raises -> Tuple[Float64, Float64, Float64, Float64, Float64]:
    """Returns (last_tok_recon, last_wm_video, last_wm_bc, last_imag_value,
    last_eval_return) for a finiteness check / logging. `last_eval_return` is the
    most recent greedy-eval episode return (0 if `eval_every <= 0`)."""
    comptime assert TNP == (TGT // PATCH) * (TGT // PATCH), (
        "TNP must equal (TGT//PATCH)^2 (the tokenizer's patch count)"
    )
    comptime BATCH = B * T
    comptime NP = TNP
    comptime DP = PATCH * PATCH
    comptime ND = NSP * DSP
    comptime ZN = BATCH * ND
    comptime AGD = NAGENT * D
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime EMAX = _ilog2(KMAX)

    comptime assert ADIM == NACT, "run_online_dreamer4 needs ADIM == NACT"

    var rng = OnlineRng(seed)
    var buf = Dreamer4FrameBuffer[IN_CH, IMG, IMG, NACT, CAP]()

    var topt = Adam(lr=lr_tok)
    var aopt = Adam(lr=lr_agent)          # CPU path: whole-agent WM step
    var copt = Adam(lr=lr_cont)           # continue head (both paths)
    var iopt = Adam(lr=lr_imag)           # imagination (value + policy heads)
    # GPU path: the whole-agent walk asserts DYN_TARGET=="cpu", so the dynamics
    # (on device) and the host heads/task-embedder are stepped as SEPARATE
    # submodules. Adam moments live in each Param, so extra Adam instances are
    # just lr/beta config (unused on the CPU path).
    var dopt = Adam(lr=lr_agent)          # GPU path: dynamics (device)
    var hopt = Adam(lr=lr_agent)          # GPU path: ph/rh/te (host)

    # ── shared scratch ──
    var frames = Tensor.alloc(BATCH * TGT * TGT)
    var patches = Tensor.alloc(BATCH * NP * DP)
    var pred = Tensor.alloc(BATCH * NP * DP)
    var gpred = Tensor.alloc(BATCH * NP * DP)
    var gin = Tensor.alloc(BATCH * NP * DP)
    var gperc = Tensor.alloc(BATCH * NP * DP)   # perceptual-grad scratch (Stage 1)
    var z1 = Tensor.alloc(ZN)
    var z0n = Tensor.alloc(ZN)
    var sigma = Tensor.alloc(BATCH)
    var sig_idx = Tensor.alloc(BATCH)
    var step_idx = Tensor.alloc(BATCH)
    var task_ids = Tensor.alloc(B)
    var act_idx = Tensor.alloc(BATCH)
    var rew = Tensor.alloc(BATCH)
    var done_b = Tensor.alloc(BATCH)
    var cont_tgt = Tensor.alloc(BATCH)
    var bins = Tensor.alloc(NBINS)
    symexp_twohot_bins[NBINS](_mao(bins.data.unsafe_ptr()), lo=value_bin_lo)
    for b in range(B):
        task_ids.data[b] = Scalar[DT](0.0)

    var pix = Tensor.alloc(BATCH * IMG_DIM)
    var act_oh = Tensor.alloc(BATCH * NACT)

    # current observation (DT copy of the env's obs list) + next-obs scratch
    var cur = List[Scalar[DT]](length=IMG_DIM, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=IMG_DIM, fill=Scalar[DT](0.0))
    var ob0 = env.reset_obs_list()
    for i in range(IMG_DIM):
        cur[i] = Scalar[DT](Float64(ob0[i]))

    # ── Stage 0: warmup collect (random, or scripted in offline mode) ────
    print("[dreamer4-online] Stage 0: warmup collect —", warmup_steps,
          "env steps", " (scripted)" if SCRIPTED else " (random)")
    var collected = 0
    while collected < warmup_steps:
        var a: Int
        comptime if SCRIPTED:
            a = COLLECT_ACTION(env, collected)
        else:
            a = Int(rng.uniform() * Float64(NACT))
            if a >= NACT:
                a = NACT - 1
        var rd = _step_repeat[E, IMG_DIM](env, a, frame_repeat, nxt)
        var r = Scalar[DT](rd[0])
        var d = rd[1]
        buf.add_step_fp32_list(cur, a, d, r)
        collected += 1
        if collected % 500 == 0:
            print("  [warmup]", collected, "/", warmup_steps, " buf=", buf.count())
        if d:
            var no = env.reset_obs_list()
            for i in range(IMG_DIM):
                cur[i] = Scalar[DT](Float64(no[i]))
        else:
            for i in range(IMG_DIM):
                cur[i] = nxt[i]
    logger.log_scalar(String("warmup_frames"), Float64(buf.count()), 0)

    # ── Stage 1: tokenizer pretrain → freeze (OR reuse a checkpoint) ─────
    # If `save_ckpt`.ckpt already exists, LOAD it (tok + agent) and SKIP the
    # pretrain loop — so you pretrain the (expensive) tokenizer ONCE, then iterate
    # on the RL downstream by re-running against the same checkpoint. Delete the
    # file (or point save_ckpt elsewhere) to force a fresh pretrain.
    var reused = False
    if save_ckpt != String(""):
        if exists(save_ckpt + ".ckpt"):
            print("[dreamer4-online] reusing checkpoint (skip tokenizer",
                  "pretrain):", save_ckpt + ".ckpt")
            agent.load(tok, save_ckpt, dctx)
            tok.set_mae_p(0.0, 0.0)   # frozen: MAE masking off
            reused = True
    var last_tok_loss: Float64 = 0.0
    if not reused:
        print("[dreamer4-online] Stage 1: tokenizer pretrain —",
              tok_pretrain_steps, "steps (perc_weight=", perc_weight, ")")
    for s in range(0 if reused else tok_pretrain_steps):
        buf.sample_reward_window_batch[B, T](
            pix.data.unsafe_ptr(), act_oh.data.unsafe_ptr(),
            rew.data.unsafe_ptr(), done_b.data.unsafe_ptr(),
        )
        _window_to_patches[B, T, IN_CH, IMG, TGT, PATCH](
            _mao(pix.data.unsafe_ptr()), _mao(frames.data.unsafe_ptr()),
            _mao(patches.data.unsafe_ptr()),
        )
        comptime if DYN_TARGET == "cpu":
            topt.zero_grad["cpu"](tok, None)
            tok.forward["cpu", BATCH](TensorRefs[1](patches), pred, None)
            if perc_weight > 0.0:
                # paper eq. 5: MSE + w·perceptual (frozen CIFAR backbone, BN-eval).
                var lv = masked_recon_plus_perceptual_loss[
                    BATCH, 1, TGT, TGT, PATCH
                ](
                    _mao(pred.data.unsafe_ptr()),
                    _mao(patches.data.unsafe_ptr()),
                    tok.mae_mask_ptr(), backbone, perc_weight,
                    _mao(gpred.data.unsafe_ptr()), _mao(gperc.data.unsafe_ptr()),
                )
                last_tok_loss = lv[0] + perc_weight * lv[1]
            else:
                last_tok_loss = masked_recon_loss[NP, DP, BATCH](
                    _mao(pred.data.unsafe_ptr()),
                    _mao(patches.data.unsafe_ptr()),
                    tok.mae_mask_ptr(), _mao(gpred.data.unsafe_ptr()),
                )
            tok.vjp["cpu", BATCH](
                TensorRefs[1](patches), gpred, TensorRefs[1](gin), None
            )
            topt.step["cpu"](tok, None)
        else:
            # GPU tokenizer: masked-MSE recon (perceptual = Part 2, CPU-only for
            # now — perc_weight is ignored on this path). Upload patches, device
            # forward, device recon grad (keep flags come off mae_mask_ptr(),
            # which returns the DEVICE ptr on GPU), device vjp, device step.
            var c = dctx.value()
            patches.ensure_gpu(c, BATCH * NP * DP)
            patches.upload(c)
            pred.ensure_gpu(c, BATCH * NP * DP)
            gpred.ensure_gpu(c, BATCH * NP * DP)
            gin.ensure_gpu(c, BATCH * NP * DP)
            topt.zero_grad["gpu"](tok, dctx)
            tok.forward["gpu", BATCH](
                TensorRefs[1](patches), pred, Optional(c)
            )
            if perc_weight > 0.0:
                # MSE + w·perceptual, backbone on device (GPU-resident backbone).
                var lv = masked_recon_plus_perceptual_loss_gpu[
                    BATCH, 1, TGT, TGT, PATCH
                ](
                    pred, patches, tok.mae_mask_ptr(), backbone, perc_weight,
                    gpred, gperc, c,
                )
                last_tok_loss = lv[0] + perc_weight * lv[1]
            else:
                masked_recon_grad_gpu[NP, DP, BATCH](
                    _dptr(pred), _dptr(patches), tok.mae_mask_ptr(),
                    _dptr(gpred), c,
                )
                if s % 10 == 0:                 # proxy scalar for logging only
                    pred.download(c)
                    var sse: Float64 = 0.0
                    for i in range(BATCH * NP * DP):
                        var d = Float64(pred.data[i]) - Float64(patches.data[i])
                        sse += d * d
                    last_tok_loss = sse / Float64(BATCH * NP * DP)
            tok.vjp["gpu", BATCH](
                TensorRefs[1](patches), gpred, TensorRefs[1](gin), Optional(c)
            )
            topt.step["gpu"](tok, dctx)
        tok.advance_rng()
        if s % 10 == 0:
            print("  [tok]", s, "/", tok_pretrain_steps, " recon=", last_tok_loss)
    if not reused:
        tok.set_mae_p(0.0, 0.0)  # FREEZE
        print("  tokenizer frozen (recon=", last_tok_loss, ")")
        logger.log_scalar(String("tokenizer_loss"), last_tok_loss, 0)
        # Checkpoint right after tokenizer pretrain so the imagination-GIF example
        # can eyeball RECON quality (the tokenizer autoencode) WITHOUT running any
        # RL — the tokenizer is the gate: if RECON is noise, nothing downstream
        # can work. This is also the file a later run reuses to skip pretraining.
        if save_ckpt != String(""):
            agent.save(tok, save_ckpt, dctx)
            print("  [ckpt] post-tokenizer-pretrain checkpoint saved:",
                  save_ckpt + ".ckpt", "(RECON is now GIF-able)")

    # ── Stage 2: online RL loop ─────────────────────────────────────────
    # rolling window of the last ≤T encoded obs frames (front-aligned) +
    # the actions leading into each frame.
    var win_patch = Tensor.alloc(T * NP * DP)
    var win_z = Tensor.alloc(T * ND)
    var fr1 = Tensor.alloc(TGT * TGT)
    var pa1 = Tensor.alloc(NP * DP)
    var win_act = List[Int](length=T, fill=-1)
    var act_hist = List[Scalar[DT]](length=T * ADIM, fill=Scalar[DT](0.0))
    var win_n = 0
    var last_action = -1

    var ctx = Tensor.alloc(B * NCTX * ND)
    var u01 = Tensor.alloc(B * T)
    var znoise = Tensor.alloc(B * T * ND)

    var ob1 = env.reset_obs_list()
    for i in range(IMG_DIM):
        cur[i] = Scalar[DT](Float64(ob1[i]))
    var did_imag = False
    var last_v: Float64 = 0.0
    var last_p: Float64 = 0.0
    var last_video: Float64 = 0.0
    var last_bc: Float64 = 0.0
    var ep_ret: Float64 = 0.0            # training-episode return (explore=True)
    var last_eval_return: Float64 = 0.0
    # greedy-eval scratch (its own window; an eval episode interrupts the
    # in-progress training episode, so we reset both env + training window after).
    var ewin_patch = Tensor.alloc(T * NP * DP)
    var ewin_z = Tensor.alloc(T * ND)
    var efr1 = Tensor.alloc(TGT * TGT)
    var epa1 = Tensor.alloc(NP * DP)
    var ewin_act = List[Int](length=T, fill=-1)
    var eact_hist = List[Scalar[DT]](length=T * ADIM, fill=Scalar[DT](0.0))
    var ecur = List[Scalar[DT]](length=IMG_DIM, fill=Scalar[DT](0.0))
    print("[dreamer4-online] Stage 2: online RL —", total_env_steps, "env steps")
    for step in range(total_env_steps):
        if step > 0 and step % log_every == 0:
            print("  [rl]", step, "/", total_env_steps,
                  " wm_video=", last_video, " wm_bc=", last_bc,
                  " imag_v=", last_v, " eval=", last_eval_return)
            # Reward/value calibration means (cheap _mmm) — computed here so they
            # go to BOTH the CSV logger and the diag print. `imag_value`/
            # `imag_policy` above are the value/policy LOSSES (twohot CE / PMPO),
            # NOT the value magnitude — these means are the magnitudes. If
            # imag_rew_mean ≫ real_rew_mean (or imag_val_mean is wildly optimistic
            # vs eval_return), the reward/value heads hallucinate → the WM/heads
            # are the bottleneck; if they match but the policy doesn't improve, the
            # bug is the policy update (PMPO advantage/weighting).
            var rr = _mmm(_mao(rew.data.unsafe_ptr()), BATCH)
            var ir = _mmm(agent.im_rew_ptr(), BATCH)
            var iv = _mmm(agent.im_val_ptr(), BATCH)
            var lr = _mmm(agent.im_ret_ptr(), B * (T - 1))
            # metric logging at the SAME cadence (not per train/imag step — that
            # floods the remote logger with thousands of points).
            if logger.is_active():
                logger.log_scalar(String("wm_video_loss"), last_video, step)
                logger.log_scalar(String("wm_bc_loss"), last_bc, step)
                logger.log_scalar(String("value_loss"), last_v, step)
                logger.log_scalar(String("policy_loss"), last_p, step)
                # NEW — reward/value calibration (the decisive columns):
                logger.log_scalar(String("reward_mean"), rr[0], step)
                logger.log_scalar(String("imagined_reward_mean"), ir[0], step)
                logger.log_scalar(String("imagined_value_mean"), iv[0], step)
                logger.log_scalar(String("lambda_return_mean"), lr[0], step)
            if diag:
                print("  [diag] real_rew(mean/min/max)=", rr[0], rr[1], rr[2],
                      " imag_rew=", ir[0], ir[1], ir[2],
                      " imag_val=", iv[0], iv[1], iv[2],
                      " lambda_ret=", lr[0], lr[1], lr[2])
        var a: Int
        comptime if SCRIPTED:
            # Offline collection: act with the decent driver (data stays good);
            # the learned policy is trained ONLY in imagination + greedy-eval'd.
            a = COLLECT_ACTION(env, step)
        else:
            win_n = _push_frame[IN_CH, IMG, TGT, PATCH, T, NP, DP](
                _mao(cur.unsafe_ptr()), fr1, pa1, win_patch, win_act,
                win_n, last_action,
            )
            # encode the window (B'=1, T frames) → win_z [T*ND]
            # (zero the unused tail frames so the encode is deterministic)
            for i in range(win_n * NP * DP, T * NP * DP):
                win_patch.data[i] = Scalar[DT](0.0)
            _encode[T, DYN_TARGET](tok.enc, win_patch, win_z, dctx)
            # action history one-hots leading into frames 1..win_n-1
            for i in range(T * ADIM):
                act_hist[i] = Scalar[DT](0.0)
            for fr in range(1, win_n):
                var ap = win_act[fr]
                if ap >= 0 and ap < ADIM:
                    act_hist[fr * ADIM + ap] = Scalar[DT](1.0)
            a = agent.act_from_latents(
                _mao(win_z.data.unsafe_ptr()), win_n,
                _mao(act_hist.unsafe_ptr()), 0, True, rng.uniform(), dctx,
            )

        var rd = _step_repeat[E, IMG_DIM](env, a, frame_repeat, nxt)
        var r = Scalar[DT](rd[0])
        var d = rd[1]
        buf.add_step_fp32_list(cur, a, d, r)
        last_action = a
        ep_ret += Float64(rd[0])
        if d:
            if logger.is_active():
                logger.log_scalar(String("episode_reward"), ep_ret, step)
            ep_ret = 0.0
            var no = env.reset_obs_list()
            for i in range(IMG_DIM):
                cur[i] = Scalar[DT](Float64(no[i]))
            win_n = 0
            last_action = -1
        else:
            for i in range(IMG_DIM):
                cur[i] = nxt[i]

        # ── world-model + heads update ──
        if step % train_every == 0 and buf.count() >= BATCH:
            buf.sample_reward_window_batch[B, T](
                pix.data.unsafe_ptr(), act_oh.data.unsafe_ptr(),
                rew.data.unsafe_ptr(), done_b.data.unsafe_ptr(),
            )
            _window_to_patches[B, T, IN_CH, IMG, TGT, PATCH](
                _mao(pix.data.unsafe_ptr()), _mao(frames.data.unsafe_ptr()),
                _mao(patches.data.unsafe_ptr()),
            )
            _encode[BATCH, DYN_TARGET](tok.enc, patches, z1, dctx)
            _build_shortcut_schedule[B, T, B_SELF, KMAX, EMAX, ND](
                rng, _mao(sigma.data.unsafe_ptr()),
                _mao(sig_idx.data.unsafe_ptr()),
                _mao(step_idx.data.unsafe_ptr()), _mao(z0n.data.unsafe_ptr()),
            )
            # dataset action ids from the sampled one-hots
            for bt in range(BATCH):
                var best = 0
                var bv = act_oh.data[bt * NACT]
                for c in range(1, NACT):
                    if act_oh.data[bt * NACT + c] > bv:
                        bv = act_oh.data[bt * NACT + c]
                        best = c
                act_idx.data[bt] = Scalar[DT](Float64(best))
            var do_boot = step >= 2 * train_every
            comptime if DYN_TARGET == "cpu":
                aopt.zero_grad["cpu"](agent, None)
                var losses = agent.acwm_train_step(
                    _mao(z1.data.unsafe_ptr()), _mao(z0n.data.unsafe_ptr()),
                    _mao(sigma.data.unsafe_ptr()), _mao(sig_idx.data.unsafe_ptr()),
                    _mao(step_idx.data.unsafe_ptr()), do_boot,
                    _mao(task_ids.data.unsafe_ptr()),
                    _mao(act_idx.data.unsafe_ptr()),
                    _mao(rew.data.unsafe_ptr()), _mao(bins.data.unsafe_ptr()),
                )
                aopt.step["cpu"](agent, None)
                last_video = losses[0]
                last_bc = losses[1]
            else:
                var c = dctx.value()
                # zero grads: dynamics on device, heads + task-embedder on host.
                dopt.zero_grad["gpu"](agent.dyn, dctx)
                agent.ph.zero_grad["cpu"](None)
                agent.rh.zero_grad["cpu"](None)
                agent.te.zero_grad["cpu"](None)   # te is not a Module
                agent.ph_prior.zero_grad["cpu"](None)  # BC-trained anchor
                var losses = agent.acwm_train_step_gpu(
                    _mao(z1.data.unsafe_ptr()), _mao(z0n.data.unsafe_ptr()),
                    _mao(sigma.data.unsafe_ptr()), _mao(sig_idx.data.unsafe_ptr()),
                    _mao(step_idx.data.unsafe_ptr()), do_boot,
                    _mao(task_ids.data.unsafe_ptr()),
                    _mao(act_idx.data.unsafe_ptr()),
                    _mao(rew.data.unsafe_ptr()), _mao(bins.data.unsafe_ptr()), c,
                )
                # step: dynamics via its Adam; the host heads + te under a SINGLE
                # hopt advance (calling Adam.step per-submodule would over-bump the
                # bias-correction step counter). te isn't a Module → walk its
                # params with the Adam visitor directly.
                dopt.step["gpu"](agent.dyn, dctx)
                hopt.begin_step()
                walk_params["cpu"](agent.ph, hopt, None)
                walk_params["cpu"](agent.rh, hopt, None)
                walk_params["cpu"](agent.te, hopt, None)
                walk_params["cpu"](agent.ph_prior, hopt, None)  # BC anchor
                last_video = losses[0]
                last_bc = losses[1]

            # continue head off the clean h_t left by acwm's clean forward
            var ht = agent.agent_out_ptr()
            var ht_t = Tensor.alloc(BATCH * AGD)
            for i in range(BATCH * AGD):
                ht_t.data[i] = ht[unsafe_offset=i]
            var clog = Tensor.alloc(BATCH)
            var gcl = Tensor.alloc(BATCH)
            var gci = Tensor.alloc(BATCH * AGD)
            for i in range(BATCH):
                cont_tgt.data[i] = Scalar[DT](1.0) - done_b.data[i]
            copt.zero_grad["cpu"](agent.ch, None)
            agent.ch.forward["cpu", BATCH](TensorRefs[1](ht_t), clog, None)
            continue_bce_backward[BATCH](
                _mao(clog.data.unsafe_ptr()), _mao(cont_tgt.data.unsafe_ptr()),
                Scalar[DT](1.0), _mao(gcl.data.unsafe_ptr()),
            )
            agent.ch.vjp["cpu", BATCH](
                TensorRefs[1](ht_t), gcl, TensorRefs[1](gci), None
            )
            copt.step["cpu"](agent.ch, None)

        # ── imagination-RL update (frozen WM) ──
        if imag_every > 0 and step % imag_every == 0 and buf.count() >= BATCH:
            # ph_prior is now a BC-trained anchor (updated in acwm), NOT a
            # self-snapshot — so DON'T copy ph into it here. The reverse-KL to
            # this diverse BC prior is what prevents PMPO policy mode-collapse.
            buf.sample_reward_window_batch[B, T](
                pix.data.unsafe_ptr(), act_oh.data.unsafe_ptr(),
                rew.data.unsafe_ptr(), done_b.data.unsafe_ptr(),
            )
            _window_to_patches[B, T, IN_CH, IMG, TGT, PATCH](
                _mao(pix.data.unsafe_ptr()), _mao(frames.data.unsafe_ptr()),
                _mao(patches.data.unsafe_ptr()),
            )
            _encode[BATCH, DYN_TARGET](tok.enc, patches, z1, dctx)
            for b in range(B):
                for i in range(ND):
                    ctx.data[b * NCTX * ND + i] = z1.data[(b * T + 0) * ND + i]
            for i in range(B * T):
                u01.data[i] = Scalar[DT](rng.uniform())
            for i in range(B * T * ND):
                znoise.data[i] = Scalar[DT](rng.gauss())
            # Imagination trains only the value + policy heads (transformer
            # frozen); the rollout runs the dynamics forward on DYN_TARGET.
            comptime if DYN_TARGET == "cpu":
                # Zero + step ONLY the value/policy heads (mirrors the GPU
                # branch below). A whole-agent `iopt.step(agent)` here
                # silently UN-FROZE the WM: every dynamics/tokenizer param
                # rode the imagination optimizer at lr_imag.
                agent.vh.zero_grad["cpu"](None)
                agent.ph.zero_grad["cpu"](None)
                var il = agent.imag_train_step(
                    _mao(ctx.data.unsafe_ptr()), _mao(u01.data.unsafe_ptr()),
                    _mao(znoise.data.unsafe_ptr()),
                    _mao(task_ids.data.unsafe_ptr()),
                    _mao(bins.data.unsafe_ptr()), use_continue=True,
                    gamma=imag_gamma,
                )
                # value + policy heads under a SINGLE iopt advance (see WM
                # note in the gpu branch).
                iopt.begin_step()
                walk_params["cpu"](agent.vh, iopt, None)
                walk_params["cpu"](agent.ph, iopt, None)
                last_v = il[0]
                last_p = il[1]
            else:
                agent.vh.zero_grad["cpu"](None)
                agent.ph.zero_grad["cpu"](None)
                var il = agent.imag_train_step(
                    _mao(ctx.data.unsafe_ptr()), _mao(u01.data.unsafe_ptr()),
                    _mao(znoise.data.unsafe_ptr()),
                    _mao(task_ids.data.unsafe_ptr()),
                    _mao(bins.data.unsafe_ptr()), use_continue=True,
                    gamma=imag_gamma, dctx=dctx,
                )
                # value + policy heads under a SINGLE iopt advance (see WM note).
                iopt.begin_step()
                walk_params["cpu"](agent.vh, iopt, None)
                walk_params["cpu"](agent.ph, iopt, None)
                last_v = il[0]
                last_p = il[1]
            did_imag = True

        # ── greedy-eval (measurable return; interrupts training ep). Averaged
        # over num_eval_episodes to cancel CarRacing's per-reset track-seed noise ──
        if eval_every > 0 and step % eval_every == 0 and step > 0:
            var eret_sum: Float64 = 0.0
            var eret_min: Float64 = 0.0
            var eret_max: Float64 = 0.0
            var n_ep = num_eval_episodes if num_eval_episodes > 0 else 1
            for _ep in range(n_ep):
                var eob = env.reset_obs_list()
                for i in range(IMG_DIM):
                    ecur[i] = Scalar[DT](Float64(eob[i]))
                var ewin_n = 0
                var elast = -1
                var eret: Float64 = 0.0
                for _es in range(eval_max_steps):
                    ewin_n = _push_frame[IN_CH, IMG, TGT, PATCH, T, NP, DP](
                        _mao(ecur.unsafe_ptr()), efr1, epa1, ewin_patch, ewin_act,
                        ewin_n, elast,
                    )
                    for i in range(ewin_n * NP * DP, T * NP * DP):
                        ewin_patch.data[i] = Scalar[DT](0.0)
                    _encode[T, DYN_TARGET](tok.enc, ewin_patch, ewin_z, dctx)
                    for i in range(T * ADIM):
                        eact_hist[i] = Scalar[DT](0.0)
                    for fr in range(1, ewin_n):
                        var ap = ewin_act[fr]
                        if ap >= 0 and ap < ADIM:
                            eact_hist[fr * ADIM + ap] = Scalar[DT](1.0)
                    var ea = agent.act_from_latents(
                        _mao(ewin_z.data.unsafe_ptr()), ewin_n,
                        _mao(eact_hist.unsafe_ptr()), 0, False, 0.0, dctx,
                    )
                    var erd = _step_repeat[E, IMG_DIM](env, ea, frame_repeat, ecur)
                    eret += erd[0]
                    elast = ea
                    if erd[1]:
                        break
                eret_sum += eret
                if _ep == 0 or eret < eret_min:
                    eret_min = eret
                if _ep == 0 or eret > eret_max:
                    eret_max = eret
            last_eval_return = eret_sum / Float64(n_ep)
            if n_ep > 1:
                print("  [eval] step", step, " greedy return (mean/min/max over",
                      n_ep, ") =", last_eval_return, eret_min, eret_max)
            else:
                print("  [eval] step", step, " greedy return =", last_eval_return)
            if logger.is_active():
                logger.log_scalar(String("eval_return"),
                                  last_eval_return, step)
            if save_ckpt != String(""):
                agent.save(tok, save_ckpt, dctx)
                print("  [ckpt] saved", save_ckpt + ".ckpt", "at step", step)
            # resume training on a fresh episode (eval consumed the env)
            var rob = env.reset_obs_list()
            for i in range(IMG_DIM):
                cur[i] = Scalar[DT](Float64(rob[i]))
            win_n = 0
            last_action = -1
            ep_ret = 0.0

    if save_ckpt != String(""):
        agent.save(tok, save_ckpt, dctx)
        print("  [ckpt] final checkpoint saved:", save_ckpt + ".ckpt")

    logger.flush()
    _ = did_imag
    _ = last_p
    return (last_tok_loss, last_video, last_bc, last_v, last_eval_return)
