"""Dreamer 4 dynamics lighthouse — learn to predict the next Pong frame (CPU).

    pixi run mojo run -I . examples/dreamer4/pong_dynamics_lighthouse.mojo

End-to-end Phase 2 validation of the interactive dynamics + shortcut-forcing
flow-matching loss on the *frozen* tokenizer:

  Phase A  train the causal tokenizer on Pong frames (as in the tokenizer
           lighthouse), then FREEZE it (mae p=0 ⇒ clean latents).
  Phase B  each step: stream a Pong window, encode all T frames → clean
           latents z1 (packed k=1 ⇒ n_spatial=L, d_spatial=D_BOT); train the
           dynamics with `dynamics_pretrain_loss` (empirical flow + bootstrap).
  Eval     autoregressive one-step rollout: take frames 0..T-2 as clean
           context, ODE-sample (K steps) the T-1-th frame's latents, and
           score them against the true last-frame latents (latent MSE) — plus
           splice+decode for an informational pixel PSNR.

GATE = **latent-MSE of the ODE-sampled frame** (sampled vs ground-truth
latents). At init the zero-init flow head drives the ODE integration to ~0
(MSE ≈ mean(z²)); a trained dynamics predicts the true next-frame latents, so
the MSE collapses. This also validates the K-step ODE SAMPLER, a distinct
code path from the single-step training loss.

Pixel PSNR is reported but NOT gated: Pong frames are mostly-black and
near-identical frame-to-frame, so one-step next-frame PSNR saturates at the
(frozen) tokenizer's recon ceiling (~23 dB) regardless of the predicted frame
— a property of the sparse content + tokenizer, not the dynamics. Pure CPU.
"""

from std.math import sqrt, log, log10, cos
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.envs.arcade_games.pong.online_sampler import (
    OnlinePongSampler, ScriptedPongPolicy,
)
from noeira.data.window_source import WindowSource
from noeira.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from noeira.deep_agents.dreamer4.dynamics import Dreamer4Dynamics
from noeira.deep_agents.dreamer4.shortcut_loss import dynamics_pretrain_loss, _mao
from noeira.deep_agents.dreamer4.ode_sampler import sample_one_timestep
from noeira.deep_agents.dreamer4.recon_loss import (
    masked_recon_loss, full_recon_psnr,
)
from noeira.deep_agents.dreamer4.patchify import downscale_box, temporal_patchify


# ── tiny deterministic RNG (xorshift64* + Box-Muller) ──────────────────
struct Rng(Copyable, Movable):
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


def main() raises:
    print("=" * 70)
    print("Dreamer 4 dynamics lighthouse — next-frame prediction (CPU)")
    print("=" * 70)

    comptime IN_CH = 4
    comptime IMG = 84
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime ACT = 3
    comptime T = 4
    comptime B = 4
    comptime BATCH = B * T

    comptime TGT = 32
    comptime PATCH = 8
    comptime NP = (TGT // PATCH) * (TGT // PATCH)   # 16
    comptime DP = PATCH * PATCH                      # 64
    comptime D = 64
    comptime NH = 4
    comptime L = 8
    comptime D_BOT = 16
    comptime HID = 256
    comptime DEPTH = 2
    comptime DROP = 0.5

    comptime NSP = L                  # packing k=1
    comptime DSP = D_BOT
    comptime ND = NSP * DSP
    comptime KMAX = 4
    comptime KEVAL = 4
    comptime NREG = 2
    comptime D_DYN = 64
    comptime HID_DYN = 128
    comptime DEPTH_DYN = 2
    comptime B_SELF = 2
    comptime B_EMP = B - B_SELF
    comptime EMAX = 2                 # log2(KMAX)

    comptime STEPS_TOK = 150
    comptime STEPS_DYN = 220
    comptime EVAL_EVERY = 40
    comptime LR_TOK = Scalar[DT](2e-3)
    comptime LR_DYN = Scalar[DT](1e-3)

    comptime FRAME_N = BATCH * TGT * TGT
    comptime PATCH_N = BATCH * NP * DP
    comptime ZN = BATCH * ND

    comptime OnlineBuf = OnlinePongSampler[ScriptedPongPolicy, B, T]
    var src = WindowSource[IMG_DIM, ACT, T, B, "cpu", OnlineBuf].make(
        OnlineBuf.make(ScriptedPongPolicy(eps=0.3))
    )

    var tok = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, DROP, DROP, 7
    ].make["cpu", Xavier](None)
    var topt = Adam(lr=LR_TOK)

    var dyn = Dreamer4Dynamics[
        DSP, NSP, D_DYN, NH, T, NREG, HID_DYN, DEPTH_DYN, KMAX
    ].make["cpu", Xavier](None)
    var dopt = Adam(lr=LR_DYN)

    # Storage scratch. Buffers fed to forward/vjp are `Tensor`s; the loss/sampler
    # helpers read their underlying host `data` via `_mao(...)`.
    var frames = Tensor.alloc(FRAME_N)
    var patches = Tensor.alloc(PATCH_N)
    var pred = Tensor.alloc(PATCH_N)
    var gpred = Tensor.alloc(PATCH_N)
    var gin = Tensor.alloc(PATCH_N)
    var z1 = Tensor.alloc(ZN)
    var gz = Tensor.alloc(ZN)
    var z0n = Tensor.alloc(ZN)
    var sigma = Tensor.alloc(BATCH)
    var sig_idx = Tensor.alloc(BATCH)
    var step_idx = Tensor.alloc(BATCH)
    var grad_zhat = Tensor.alloc(ZN)
    var zhat = Tensor.alloc(ZN)
    var ztil = Tensor.alloc(ZN)       # main-pass input z̃ (= storage dyn.vjp fwd_in)
    var ctx_lat = Tensor.alloc(B * (T - 1) * ND)
    var z_init = Tensor.alloc(B * ND)
    var pred_last = Tensor.alloc(B * ND)
    var zwin = Tensor.alloc(ZN)
    var rec = Tensor.alloc(PATCH_N)

    var rng = Rng(20260606)

    # ── Phase A: train tokenizer ────────────────────────────────────────
    print("- Phase A: tokenizer")
    for step in range(STEPS_TOK):
        src.next_batch()
        var pix = src.pix_ptr()
        for b in range(B):
            for t in range(T):
                var bt = b * T + t
                var fsrc = pix + bt * IMG_DIM + 3 * IMG * IMG
                downscale_box[IMG, IMG, TGT, TGT](
                    fsrc, _mao(frames.data.unsafe_ptr()) + bt * TGT * TGT
                )
        temporal_patchify[BATCH, 1, TGT, TGT, PATCH](
            _mao(frames.data.unsafe_ptr()), _mao(patches.data.unsafe_ptr())
        )

        topt.zero_grad["cpu"](tok, None)
        tok.forward["cpu", BATCH](TensorRefs[1](patches), pred, None)
        var mask = tok.mae_mask_ptr()
        var loss = masked_recon_loss[NP, DP, BATCH](
            _mao(pred.data.unsafe_ptr()),
            _mao(patches.data.unsafe_ptr()),
            mask,
            _mao(gpred.data.unsafe_ptr()),
        )
        tok.vjp["cpu", BATCH](TensorRefs[1](patches), gpred, TensorRefs[1](gin), None)
        topt.step["cpu"](tok, None)
        tok.advance_rng()
        if step % 50 == 0:
            tok.set_mae_p(0.0, 0.0)
            tok.forward["cpu", BATCH](TensorRefs[1](patches), pred, None)
            print("   tok step", step, " recon PSNR =",
                  full_recon_psnr[NP, DP, BATCH](
                      _mao(pred.data.unsafe_ptr()),
                      _mao(patches.data.unsafe_ptr())
                  ), "dB")
            tok.set_mae_p(DROP, DROP)
    tok.set_mae_p(0.0, 0.0)            # FREEZE: clean latents from here on

    # ── Phase B: train dynamics on frozen-tokenizer latents ─────────────
    print("- Phase B: dynamics (shortcut forcing)")
    var first_psnr: Float64 = 0.0
    var last_psnr: Float64 = 0.0
    var first_lmse: Float64 = 0.0
    var last_lmse: Float64 = 0.0
    for step in range(-1, STEPS_DYN):
        var is_eval = step < 0 or step % EVAL_EVERY == 0 or step == STEPS_DYN - 1

        # stream a fresh Pong window + patchify
        src.next_batch()
        var pix = src.pix_ptr()
        for b in range(B):
            for t in range(T):
                var bt = b * T + t
                var fsrc = pix + bt * IMG_DIM + 3 * IMG * IMG
                downscale_box[IMG, IMG, TGT, TGT](
                    fsrc, _mao(frames.data.unsafe_ptr()) + bt * TGT * TGT
                )
        temporal_patchify[BATCH, 1, TGT, TGT, PATCH](
            _mao(frames.data.unsafe_ptr()), _mao(patches.data.unsafe_ptr())
        )

        # encode clean latents (tokenizer frozen, p=0)
        tok.enc.forward["cpu", BATCH](TensorRefs[1](patches), z1, None)

        if step >= 0:
            # sample per-(b,t) step/sigma + noise, then one shortcut step
            for b in range(B):
                var is_self = b >= B_EMP
                for t in range(T):
                    var bt = b * T + t
                    var stp = EMAX
                    if is_self:
                        stp = Int(rng.uniform() * Float64(EMAX))   # [0,EMAX)
                    var K = 1 << stp
                    var j = Int(rng.uniform() * Float64(K))
                    if j >= K:
                        j = K - 1
                    var scale = KMAX // K
                    sigma.data[bt] = Scalar[DT](Float64(j) / Float64(K))
                    sig_idx.data[bt] = Scalar[DT](Float64(j * scale))
                    step_idx.data[bt] = Scalar[DT](Float64(stp))
            for i in range(ZN):
                z0n.data[i] = Scalar[DT](rng.gauss())

            var do_boot = step >= 30
            dopt.zero_grad["cpu"](dyn, None)
            var loss = dynamics_pretrain_loss[
                type_of(dyn), B, T, B_SELF, NSP, DSP, KMAX
            ](
                dyn,
                _mao(z1.data.unsafe_ptr()),
                _mao(z0n.data.unsafe_ptr()),
                _mao(sigma.data.unsafe_ptr()),
                _mao(sig_idx.data.unsafe_ptr()),
                _mao(step_idx.data.unsafe_ptr()),
                do_boot,
                _mao(grad_zhat.data.unsafe_ptr()),
                _mao(zhat.data.unsafe_ptr()),
            )
            # Reconstruct the main-pass input z̃ = (1−σ)·z0 + σ·z1: the storage
            # dyn.vjp recomputes the spatial-proj forward from it (identical to
            # the loss's internal z̃).
            for bt in range(BATCH):
                var s = Float64(sigma.data[bt])
                for i in range(ND):
                    var idx = bt * ND + i
                    ztil.data[idx] = Scalar[DT](
                        (1.0 - s) * Float64(z0n.data[idx]) + s * Float64(z1.data[idx])
                    )
            dyn.vjp["cpu", BATCH](
                TensorRefs[1](ztil), grad_zhat, TensorRefs[1](gz), None
            )
            dopt.step["cpu"](dyn, None)
            if step % EVAL_EVERY == 0:
                print("   dyn step", step, " loss =", loss)

        # ── rollout eval (uses the just-encoded window) ─────────────────
        if is_eval:
            for b in range(B):
                for t in range(T - 1):
                    for i in range(ND):
                        ctx_lat.data[(b * (T - 1) + t) * ND + i] = (
                            z1.data[(b * T + t) * ND + i]
                        )
            for b in range(B):
                for i in range(ND):
                    z_init.data[b * ND + i] = Scalar[DT](rng.gauss())
            sample_one_timestep[type_of(dyn), B, T, NSP, DSP, KMAX, KEVAL](
                dyn,
                _mao(ctx_lat.data.unsafe_ptr()),
                _mao(z_init.data.unsafe_ptr()),
                _mao(pred_last.data.unsafe_ptr()),
            )
            # latent-MSE: ODE-sampled last frame vs true last-frame latents
            var lse: Float64 = 0.0
            for b in range(B):
                for i in range(ND):
                    var dl = (
                        Float64(pred_last.data[b * ND + i])
                        - Float64(z1.data[(b * T + (T - 1)) * ND + i])
                    )
                    lse += dl * dl
            var lmse = lse / Float64(B * ND)
            for b in range(B):
                for t in range(T):
                    for i in range(ND):
                        var wi = (b * T + t) * ND + i
                        if t < T - 1:
                            zwin.data[wi] = z1.data[wi]
                        else:
                            zwin.data[wi] = pred_last.data[b * ND + i]
            tok.dec.forward["cpu", BATCH](TensorRefs[1](zwin), rec, None)
            var sse: Float64 = 0.0
            for b in range(B):
                var base = (b * T + (T - 1)) * NP * DP
                for k in range(NP * DP):
                    var dd = (
                        Float64(rec.data[base + k]) - Float64(patches.data[base + k])
                    )
                    sse += dd * dd
            var mse = sse / Float64(B * NP * DP)
            var psnr = 120.0 if mse <= 1e-12 else -10.0 * log10(mse)
            if step < 0:
                first_psnr = psnr
                first_lmse = lmse
                print("   untrained dynamics: latent MSE =", lmse,
                      " (rollout PSNR =", psnr, "dB, saturated)")
            else:
                last_psnr = psnr
                last_lmse = lmse
                print("   dyn step", step, " latent MSE =", lmse,
                      " (rollout PSNR =", psnr, "dB)")

    print("-" * 70)
    print("  latent MSE   first =", first_lmse, "  final =", last_lmse)
    print("  rollout PSNR first =", first_psnr, "dB  final =", last_psnr,
          "dB  (informational — saturated at tokenizer ceiling)")
    assert_true(last_lmse < 0.3 * first_lmse,
                "ODE-sampled latent MSE must collapse (dynamics learns)")
    print("=" * 70)
    print("DYNAMICS LIGHTHOUSE PASSED — shortcut-forcing dynamics + ODE")
    print("sampler predict next-frame latents on frozen-tokenizer Pong (CPU)")
    print("=" * 70)
