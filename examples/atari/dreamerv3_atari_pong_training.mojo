"""DreamerV3 on Atari Pong (GPU) via the `DreamerV3Agent` facade — discrete,
pixel observations.

The discrete-pixel corner of the DreamerV3 matrix (CartPole = discrete+vector,
CarRacing = continuous+pixel, Pong = discrete+pixel) and the canonical DreamerV3
Atari benchmark. The world model + actor-critic train on-device
(`train_target="gpu"`, the discrete device-resident AC `_ac_gpu_disc`); the CPU
6502 emulator (`AtariEnv`) steps the ROM and obs are marshalled H2D inside
`select_action`. The env is NOT the bottleneck — DreamerV3 is sample-efficient
(replays each frame ~train_ratio×), so the GPU train_step dominates.

REFERENCE-ALIGNED `atari100k` run (configs.yaml `atari100k` preset, Pong
published ≈ +18 at 110k steps): train_ratio 256, actor from step 1 (NO gate),
size200m ladder (deter 8192 / hidden 1024 / classes 64 / depth 64), single
frame (no stack — the RSSM carries motion), sticky OFF, noop 30, reward
unclipped, 110k agent steps (= 440k frames, the atari100k budget + ref's 10%
margin). This makes the run a binary reproduction test of the reference.

Phase B-1 (replay `online: True`) and Phase B-2 (reference conv geometry)
are IMPLEMENTED and enabled below:
  B-1: every fresh T-window is queued on insert and served into a batch row
       exactly once, promptly (`online=True`).
  B-2: k5-s1 convs + 2×2 max-pool (encoder) / nearest-×2 upsample (decoder),
       channels BASE·{2,3,4,4} (TOKEN = 9216 = reference), the decoder's
       bspace-8 stem (BlockLinear(deter) + MLP(stoch)), and winit
       trunc_normal_in on every hidden layer (`NET_INIT=TruncNormalIn`).

Remaining NAMED deltas vs the reference (Phase B-3 if still collapsing):
  1. replay_context 1 — ref stores per-step RSSM latents in replay and
     rebuilds each window's initial carry from them (dyn.truncate); ours
     starts windows with a zero carry + fst mask. DEFERRED deliberately:
     ~6.5 GB extra VRAM at deter 8192 + replay-writeback infra for a 1-step
     carry burn-in that only improves WM signal — already our strongest
     component (obs_loss ≈0.3 by 3k updates).
  2. obs — ref atari100k is 64×64 RGB; ours 96×96 grayscale (Pong is
     near-monochrome and we have MORE pixels; ranked last).

Observation: **1×96×96 grayscale single frame** (OBS = 9216, values in [0,1]) —
`AtariEnv[OBS_MODE=3]`. The CNN encoder centers to [-0.5, 0.5] (CenterHalfOp)
and reconstructs with sigmoid + plain MSE (RECON_SIGMOID=True). Action: Pong
minimal set (6 discrete, ref `actions: needed`). Frame-skip=4 + max-pool is
done inside the env, so no agent-side action repeat.

The facade owns the whole loop (warmup → select_action → step → record
(+record_terminal on done) → train_step → periodic greedy eval, which SAMPLES the
actor) and logs the SAME KNOWN_GROUPS metrics as CartPole/CarRacing.

⚠️ Requires `roms/pong.bin` (`pixi run setup-roms`). Convergence/tuning is P2 —
run on NVIDIA.

Run:
    pixi run -e apple  mojo run -I . examples/atari/dreamerv3_atari_pong_training.mojo
    pixi run -e nvidia mojo run -I . examples/atari/dreamerv3_atari_pong_training.mojo
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.nn.primitives.ops.swish_op import SwishOp
from noeira.nn.core.initializer import Zero, TruncNormalIn
from noeira.deep_agents.dreamerv3.agent import DreamerV3Agent
from noeira.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNNPool,
    DreamerDecoderCNNPool,
)
from noeira.envs.atari import AtariEnv
from noeira.envs.atari.games.registry import AtariGame

# =============================================================================
# Architecture
# =============================================================================
comptime C = 1  # single grayscale frame (reference parity — no stacking)
comptime IMG = 96  # 96×96 (16-divisible → conv minres 6)
# Reference size LADDER (configs.yaml; deter/8 = hidden = units, classes =
# hidden/16, depth = conv base). "200m" is the un-overridden atari100k default
# behind the published Pong +18; "50m" is the SAME recipe one coherent tier
# down — ~4× cheaper per update (size was falsified as the collapse driver;
# WM fidelity on Pong was historically achieved well below even 12m).
# ⚠️ Checkpoints are tier-specific: probes/GIF scripts must use the same TIER.
comptime TIER = "50m"  # "200m" | "50m"
comptime BASE = 64 if TIER == "200m" else 32  # conv depth (channels BASE·{2,3,4,4})
comptime OBS = C * IMG * IMG  # 9216
comptime ACT = 6  # Pong minimal action set (NOOP/FIRE/RIGHT/LEFT/RIGHTFIRE/LEFTFIRE)
comptime DETER = 8192 if TIER == "200m" else 4096
comptime H = 1024 if TIER == "200m" else 512
comptime STOCH = 32
comptime CLASSES = 64 if TIER == "200m" else 32
comptime BLOCKS = 8
# RAW conv tokens (reference parity): the posterior consumes the flattened
# final conv map directly — no Linear bottleneck. With the Phase B-2 pool
# geometry (channels BASE·{2,3,4,4}, final depth 4·BASE) this is
# 4·BASE·(IMG/16)² = 256·36 = 9216 — the reference's exact token width.
comptime TOKEN = 4 * BASE * (IMG // 16) * (IMG // 16)  # 9216 (200m) / 4608 (50m)
comptime UNITS = H  # decoder bspace-stem MLP width (reference `units` = hidden)
comptime DEC_U = H  # unused by the CNN decoder (BASE drives it)
comptime HU = H  # head widths = reference units (= hidden, per tier)
comptime VU = H
comptime PU = H
comptime BINS = 255
comptime B = 16
comptime T = 64  # reference batch_length
comptime T_IMAG = 15
# GPU-resident pixel replay: CAP×OBS×4 B of VRAM (≈7.4 GB either way below).
# Bigger CAP protects rare-state coverage (paddle pinned at an edge, ball-at-
# paddle events) from circular eviction — the reference keeps 5M transitions
# uniform. At C=1 the same VRAM budget affords 4× the horizon.
comptime CAP = 200_000 if C == 1 else 50_000

comptime FEATIN = STOCH * CLASSES + DETER
# Phase B-2 reference geometry: k5-s1 convs + 2×2 max-pool (encoder) /
# nearest-×2 upsample (decoder), channels BASE·{2,3,4,4}, and the decoder's
# bspace-8 two-branch input stem (BlockLinear(deter) + MLP(stoch)).
comptime ENC = DreamerEncoderCNNPool[C, IMG, IMG, BASE, SwishOp]
comptime DEC = DreamerDecoderCNNPool[
    FEATIN, DETER, C, IMG, IMG, BASE, UNITS, SwishOp
]

comptime Ag = DreamerV3Agent[
    "gpu",
    OBS,
    ACT,
    DETER,
    H,
    STOCH,
    CLASSES,
    BLOCKS,
    TOKEN,
    DEC_U,
    HU,
    VU,
    PU,
    BINS,
    B,
    T,
    T_IMAG,
    CAP,
    True,  # DISCRETE=True (categorical actor)
    ENC,
    DEC,
    True,  # RECON_SIGMOID — reference pixel recon (sigmoid + plain MSE on [0,1])
    Zero,  # OUT_INIT — reference rew/value outscale 0
    TruncNormalIn,  # NET_INIT — reference winit trunc_normal_in (σ=√(1/fan_in))
]
comptime Env = AtariEnv[3, DT]  # OBS_MODE=3 (gray-96 single frame)

comptime NUM_STEPS = 110_000  # agent decisions (×4 ROM frames = 440k frames):
# the atari100k budget (400k frames) + the reference's own 10% margin (1.1e5).
comptime LEARN_START = 1024
# Replay ratio = B·T / TRAIN_EVERY replayed frames per env step. atari100k
# preset: train_ratio 256 → with T=64 (1024 frames per update) TRAIN_EVERY=4
# (27.5k updates over the run — the reference solves Pong in this many).
comptime TRAIN_EVERY = 4
# Actor-critic from step 1 (reference parity, benchmark-honest). The gate
# (DreamerV3Trainer.ac_start) remains available as DIAGNOSTIC scaffolding
# only: the collapse history (self-fulfilling value ridge, see the
# openloop_heads probe) must be beaten ungated to count as a reproduction.
comptime AC_START = 0
# WM-checkpoint reuse: set to a checkpoint path from a PREVIOUS run of THIS
# config (raw-token arch) to skip the WM warmup wall-time entirely — loads the
# full checkpoint, then `reset_ac()` re-initializes value/slowvalue/policy
# (the saved ones are collapsed) and the run trains the actor FROM STEP 1 on
# the mature WM (ac_start=0). This is the benchmark-honest experiment AND the
# fast iteration loop for AC-side changes. Empty = train from scratch.
comptime WM_CKPT = ""
comptime LOG_EVERY = 1000  # WM/AC loss curves (cheap; no greedy eval) — frequent
comptime EVAL_EVERY = 10_000  # greedy eval + episode returns (expensive, ~3 min)
comptime EVAL_EPISODES = 5
comptime EP_LEN = 2000  # eval-episode cap (agent decisions)
comptime CHECKPOINT_EVERY = 10_000  # frequent WM assets for WM_CKPT/reset_ac reuse


def main() raises:
    seed(42)
    print("=" * 70)
    print("DreamerV3 (facade) — Atari Pong PIXEL GPU (discrete)")
    print("=" * 70)
    print("  OBS / ACT          =", OBS, "(", C, "x", IMG, "x", IMG, ") /", ACT)
    print("  DETER/STOCH/CLASSES=", DETER, "/", STOCH, "/", CLASSES)
    print("  BASE / T / T_IMAG  =", BASE, "/", T, "/", T_IMAG)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Run + logger (same KNOWN_GROUPS metrics as the other envs) ────
        var run = RunContext(
            project=String("atari"),
            driver=String("examples/atari/dreamerv3_atari_pong_training.mojo"),
            slug=String("dreamerv3-atari-pong"),
            env=String("builtin:atari/pong"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run:", run.dir)
        var logger = run_logger(run, buffer_size=200)
        logger.set_config("algorithm", "DreamerV3")
        logger.set_config("env", "AtariPong")
        logger.set_config("target", "gpu")
        logger.set_config("t_imag", String(T_IMAG))
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        # ─── Agent (GPU) + env (CPU 6502; obs marshalled H2D in select_action) ──
        var agent = Ag.make(
            ctx=ctx,
            lr=Scalar[DT](4e-5),
            learning_starts=LEARN_START,
            warmup_steps=1000,
            actent=Scalar[DT](3e-4),
            # reference imag_loss config: slowtar False (bootstrap from the
            # ONLINE value). True also lets a value ridge persist via the
            # Polyak-lagged slow value once formed.
            slowtar=False,
            # WM-ckpt reuse → actor from step 1 (no gate); scratch → gated.
            ac_start=0 if WM_CKPT.byte_length() > 0 else AC_START,
            # Reference replay `online: True` (Phase B-1): every fresh
            # T-window is guaranteed into a batch row exactly once, promptly.
            online=True,
        )
        comptime if WM_CKPT.byte_length() > 0:
            print("loading WM checkpoint", WM_CKPT, "+ reset_ac()...")
            agent.load(String(WM_CKPT))
            agent.reset_ac()
        # atari100k env protocol: sticky OFF (unlike full-atari Machado),
        # random no-op starts 30, reward unclipped (symlog/twohot). Loads
        # roms/pong.bin.
        var env = Env(AtariGame.PONG, sticky_prob=0.0, noop_max=30)

        # ─── Single train() call — auto-eval + auto-log + auto-checkpoint ──
        print("Starting GPU training (heavy pixel config; warmup is slow)...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        var final_ret = agent.train_single[
            Env, L=RunLogger, USE_TRAIN_CUDA_GRAPH=True
        ](
            env,
            NUM_STEPS,
            learn_start=LEARN_START,
            train_every=TRAIN_EVERY,
            eval_every=EVAL_EVERY,
            eval_episodes=EVAL_EPISODES,
            ep_len=EP_LEN,
            print_every=LOG_EVERY,
            log_every=LOG_EVERY,
            verbose=True,
            logger=logger_ptr,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run.dir,
            checkpoint_every=CHECKPOINT_EVERY,
        )
        var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
        var sent = logger.b.total_logged()
        finish_run(
            run, logger, artifacts,
            String("eval_return=") + String(final_ret),
        )
        _ = logger  # lifetime extender for logger_ptr

        # ─── Summary ─────────────────────────────────────────────────────
        print("-" * 70)
        print("=" * 70)
        print("Training complete")
        print("  total env_steps   =", NUM_STEPS)
        print("  elapsed           =", elapsed_s, "s")
        print("  FINAL mean_ret(", EVAL_EPISODES, ")  =", final_ret)
        print("  remote points sent=", sent)
        print("  (Pong: -21 = shutout loss, +21 = shutout win)")
        print("=" * 70)
