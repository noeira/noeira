"""Dreamer 4 ONLINE on CarRacing (discrete, pixel obs) — the P2 lighthouse.

Drives Dreamer 4 with its OWN experience (unlike the paper's offline setting):
`run_online_dreamer4` collects from a live `CarRacingMB` (pixel obs, 5 discrete
actions), pretrains + freezes the tokenizer, then interleaves acting with world
model + agent training and periodically reports a GREEDY-EVAL return.

Tokenizer recon uses MSE + 0.2·perceptual (paper eq. 5), the perceptual term
against the frozen CIFAR ResNet-20 backbone trained by
`examples/dreamer4/train_perceptual_backbone_cifar_gpu.mojo` — this file loads
`dreamer4_perceptual_backbone.ckpt` from the working directory, or the
backbone run's `checkpoints/backbone.ckpt` with `--ckpt <run_id>`.

DYN_TARGET="gpu": the dynamics transformer AND the tokenizer run on device (the
heavy compute); the heads, task embedder, perceptual backbone, and the env stay on
host. The dims below are a modest starting point — scale D_DYN/HID/DEPTH/T/B up on
a bigger box.

Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/car_racing/dreamer4_car_racing_online.mojo
Run (Apple):  pixi run -e apple  mojo run -I . examples/car_racing/dreamer4_car_racing_online.mojo
Backbone from a run: ... dreamer4_car_racing_online.mojo --ckpt <run_id>
"""

from std.random import seed

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Xavier
from noeira.nn.core.checkpoint import load_params
from noeira.nn.models.cifar_feature_net import CifarBackbone
from noeira.core.dotenv import load_dotenv
from noeira.core.logger import RemoteLogger
from noeira.deep_agents.dreamer4.agent import Dreamer4Agent
from noeira.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from noeira.deep_agents.dreamer4.online import run_online_dreamer4
from noeira.envs.car_racing.car_racing_mb import CarRacingMB


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    # ── image / patch dims ──
    comptime IN_CH = 4
    comptime IMG = 84                       # CarRacing PIX_RES
    comptime TGT = 64                       # tokenizer target resolution
    comptime PATCH = 8
    comptime NP = (TGT // PATCH) * (TGT // PATCH)   # 64 patches
    comptime DP = PATCH * PATCH                     # 64
    comptime CAP = 100_000                  # ring-buffer capacity

    # ── sequence / batch ──
    # T=10 (was 6): the imagination horizon is H≈T-1, and the de-noised 8-track
    # eval on the T=6 run was FLAT (−30…−57, no climb) with the value head
    # collapsed to a state-independent constant (imag_val min≈max across the
    # batch) — H≈5 is too short to accumulate distinct returns, so PMPO's
    # advantage sign is noise. T=10 → H≈9 restores return variance. Cost: WM
    # attention is O(T²) → ~2.8× the T=6 WM compute + more GPU memory; if this
    # OOMs on the box, drop B 8→4 (don't change both at once — it muddies the
    # horizon measurement).
    comptime T = 10
    comptime B = 8
    comptime B_SELF = 2

    # ── tokenizer ──
    comptime L = 16
    comptime D_BOT = 16
    comptime TOK_D = 128
    comptime TOK_NH = 4
    comptime TOK_HID = 256
    comptime TOK_DEPTH = 2
    comptime DROP = 0.5

    # ── agent / dynamics ──
    comptime NSP = L
    comptime DSP = D_BOT
    comptime ND = NSP * DSP                 # 256 (must equal tokenizer L·D_BOT)
    comptime D_DYN = 128
    comptime NH = 4
    comptime NREG = 2
    comptime HID_DYN = 256
    comptime DEPTH_DYN = 2
    comptime KMAX = 4

    # ── heads / imagination ──
    comptime NAGENT = 1
    comptime NTASK = 1
    comptime HHID = 128
    comptime NACT = 5                       # noop/left/right/gas/brake
    comptime NBINS = 41
    comptime NMTP = 2
    comptime ADIM = NACT
    comptime AHID = 2 * D_DYN
    comptime K_IMAG = 2
    comptime NCTX = 1

    seed(42)
    print("=" * 70)
    print("Dreamer 4 — ONLINE CarRacing (discrete pixel) lighthouse")
    print("=" * 70)

    var ctx = DeviceContext()
    var env = CarRacingMB[DT, PIXEL_OBS=True, PIX_RES=IMG]()

    # DYN_TARGET="gpu": the dynamics transformer (the heavy WM compute) is
    # device-resident; heads + task-embedder + backbone stay on host. The agent
    # make target is "cpu" (heads on host) but the ctx is threaded so agent.dyn is
    # built on device. The tokenizer is made on GPU separately (below).
    var agent = Dreamer4Agent[
        DSP, NSP, D_DYN, NH, T, NREG, HID_DYN, DEPTH_DYN, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
        True, ADIM, AHID, K_IMAG, NCTX, "gpu",
    ].make["cpu", Xavier](Optional(ctx))

    var tok = Dreamer4Tokenizer[
        DP, TOK_D, TOK_NH, T, NSP, NP, DSP, TOK_HID, TOK_DEPTH, DROP, DROP, 7
    ].make["gpu", Xavier](Optional(ctx))       # tokenizer runs on device too

    # Frozen perceptual backbone (CIFAR ResNet-20). Trained separately; loaded
    # from the working directory, or from the run/file given by `--ckpt`.
    var backbone = CifarBackbone[TGT, TGT].make["gpu", Xavier](Optional(ctx))
    var backbone_ckpt = resolve_checkpoint(
        _flag(String("--ckpt"), String("dreamer4_perceptual_backbone.ckpt")),
        String("backbone"),
    )
    load_params["gpu"](backbone, backbone_ckpt, Optional(ctx))
    print("loaded perceptual backbone (GPU)")

    # Remote logger config from a .env (NOEIRA_CLOUD_URL / NOEIRA_CLOUD_API_KEY), the
    # same one the SAC examples use.
    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("NOEIRA_CLOUD_URL", ""),
        run_name="Dreamer4 CarRacing (online)",
        buffer_size=64,
        api_key=env_vars.get("NOEIRA_CLOUD_API_KEY", ""),
    )
    logger.set_config("algorithm", "Dreamer4")
    logger.set_config("env", "CarRacing")
    logger.set_config("obs", "pixel")

    print("starting online training...")
    # ── TRAINING config. The reward/value diagnostic (diag=True, still on as cheap
    # telemetry) confirmed the critic is now stable: with imag_gamma=0.95 +
    # value_bin_lo=-6 the value stays bounded (±~20) with state structure and
    # tracks the λ-returns (the earlier ±2000 divergence is gone). This run is the
    # real learning curve — multiple eval points over 200k steps.
    #
    # T RAISED 6→10 (see the comptime T comment): the T=6 de-noised eval was flat
    # with a state-collapsed value head (H≈5 too short). Watch the diag `imag_val`
    # min/max spread — if it now opens up (state-dependent value) AND the 8-track
    # eval mean climbs, the horizon was the binding constraint; scale
    # total_env_steps to 1_000_000+. If eval is STILL flat with T=10, the limiter
    # is the imagined reward head under-dispersing (it rarely predicts the +3
    # tile-crossing real reward shows) — a WM-fidelity problem, not horizon.
    var summary = run_online_dreamer4[
        IN_CH=IN_CH, IMG=IMG, TGT=TGT, PATCH=PATCH, TNP=NP, CAP=CAP,
        TOK_D=TOK_D, TOK_NH=TOK_NH, TOK_HID=TOK_HID, TOK_DEPTH=TOK_DEPTH,
        TOK_PMIN=DROP, TOK_PMAX=DROP, TOK_SEED=7, DYN_TARGET="gpu",
    ](
        agent, tok, backbone, env, logger,
        warmup_steps=5_000,
        tok_pretrain_steps=4_000,  # With perc_weight=0.0 (pure masked-MSE) the
                                  # tokenizer reconstructs REAL CarRacing frames to
                                  # MSE ~0.001 (PSNR ~30) by ~step 400 — isolated
                                  # diagnostics (scratch tok_diag_*) confirm the
                                  # encoder/decoder/MAE/vjp are all correct on CPU
                                  # AND GPU. The old 20_000 + "tokenizer is the
                                  # GATE / needs capacity" story was WRONG: the
                                  # 0.22 plateau came entirely from perc_weight>0
                                  # (see below), not undertraining or capacity.
                                  # 4_000 is generous headroom over the ~400 needed.
        total_env_steps=200_000,
        train_every=4,
        imag_every=8,
        eval_every=10_000,
        perc_weight=0.0,          # paper eq. 5 is MSE + 0.2·LPIPS, but our LPIPS
                                  # SURROGATE (frozen CIFAR ResNet-20) is net
                                  # HARMFUL here: CarRacing frames are far OOD for a
                                  # CIFAR net, so its feature-MSE gradient points
                                  # away from pixel reconstruction and DOMINATES the
                                  # true MSE grad. Isolated diagnostic (real frames,
                                  # components logged separately): pure MSE → 0.001;
                                  # +0.2·perceptual → spikes to 0.22 at step 40 with
                                  # pred pinned ≈0 (THE plateau), then only recovers
                                  # to 0.023 (23× worse). Pure MSE gives sharp recon
                                  # on this simple env; revisit only with real LPIPS.
        eval_max_steps=1_000,
        num_eval_episodes=8,      # average greedy return over 8 random tracks —
                                  # single-episode eval is dominated by which track
                                  # it draws (a +42 vs -66 swing can be the SAME
                                  # policy on an easy vs hard track).
        imag_gamma=Scalar[DT](0.95),   # H≈4-step imagination ⇒ 0.997 is mismatched
                                        # (return = ~pure bootstrap, value collapses
                                        # to a constant, PMPO advantage sign → noise).
                                        # 0.95 (eff. horizon ~20) restores state
                                        # structure. Raise T (→16) if still weak.
        value_bin_lo=Scalar[DT](-6.0),  # value grid ±symexp(6)≈±402 (vs default
                                        # ±8100): matches CarRacing's ±~60 scale so
                                        # the 41 bins have resolution where values
                                        # live and the TD critic can't drift to the
                                        # ±2000 excursions seen with the wide grid.
        frame_repeat=4,           # action repeat (standard CarRacing)
        diag=True,                # print reward/value/return sanity stats
        save_ckpt="dreamer4_carracing_online",  # params ckpt (tok + agent) every
                                  # eval + at end; loaded by the imagination-GIF
                                  # example dreamer4_car_racing_imagination_gif.mojo
        dctx=Optional(ctx),       # GPU dynamics
    )
    logger.close()
    _ = logger

    print("-" * 70)
    print("done. summary (tok, wm_video, wm_bc, imag_value, eval_return):")
    print(" ", summary[0], summary[1], summary[2], summary[3], summary[4])
    print("  last greedy-eval return =", summary[4])
