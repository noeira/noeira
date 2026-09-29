"""BFM-Zero on the Unitree G1 — the privileged arm, trained the reference's
way: FB-CPR online, 1024 lanes, reference-state init, expert-tracking
rollouts, 16 updates per batched step. G3.3 of
`docs/BFM_ZERO_G1_REPRODUCTION.md` §12.

    pixi run -e nvidia mojo run -I . examples/g1/bfm_zero_train_gpu.mojo --smoke
    pixi run -e nvidia mojo run -I . examples/g1/bfm_zero_train_gpu.mojo --steps 204800   # 200 batched steps
    pixi run -e nvidia mojo run -I . examples/g1/bfm_zero_train_gpu.mojo --steps 192000000 --tag g3_priv

    # resume the 192 M run from the checkpoint a dead box left behind
    pixi run -e nvidia mojo run -I . examples/g1/bfm_zero_train_gpu.mojo \\
        --resume runs/<id>/checkpoints/step_50000.ckpt --start-at 51200000

`--smoke` is a PRESET (40 batched steps, print every 10, reset diagnostics on
every reset), not an override — pass `--steps` and it wins, so
`--smoke --steps 204800` is a long smoke rather than a silent 40 steps.

THE RUN IS AN OBJECT ON DISK, not a terminal. `RunContext` names the directory;
`runs/<id>/metrics.csv` carries every curve as the run goes (`CsvLogger`), the
dashboard gets the same points when `.env` names one (`RemoteLogger`, inert
without a URL), `run.kv` records `status`/`outcome` WRITTEN and never inferred,
and each checkpoint is offered to the artifact sink so a box that dies at hour
40 does not take the weights with it. A 30-minute check-up is therefore
`tail -3 runs/<id>/metrics.csv` plus `run.kv`, from anywhere — read `eval_emd`
and `b_rank_eff` first. `eval_*` is the TRACKING EVAL, run in the loop over
the checkpoint just written (`--eval-segments`, default 5 = five segments per
clip STRATIFIED across it, ~200 of the store's ~862 windows, ~3 min; the old
default of 1 scored the opening ten seconds of each clip and read ~0.19 low
against the reference, §12.29); run 1 had to be scored by hand hours after it ended, which is how
a divergence at 4 M steps went unnoticed until 8 M (§12.14). `norm/B` is sqrt(d) = 16 BY CONSTRUCTION and cannot
move, so it can never warn you about anything; `norm/B_rank_eff` is the
effective rank of `E[B B^T]` and must sit at d = 256. Run 1 died with it at
162 (§12.12-§12.13).

`--resume PATH` restores the online nets + `.norm`; `--start-at N` is the env
steps that checkpoint already represents, so `--steps` stays the TOTAL and this
run does the remainder, with printed/logged steps and checkpoint names carrying
the offset. ⚠ The targets are hard-copied from the online nets, the Adam
moments re-warm, and the REPLAY RING STARTS EMPTY (~32 min to refill at 1024
lanes) — a resume is not bit-identical to an uninterrupted run.

⚠ NVIDIA ONLY. The G1 batched env does not compile for Metal (§12.2), so this
file is written on the laptop and built on the box — every laptop-side edit
reaches the GPU as its first compile. Everything it composes is gated on its
own: the env (G0), the store (G1), the released actor in the env (G2), the
527-D observation (G3.0), the reset table (G3.1 host half) and the injection
kernel itself (`tests/robots/test_g1_rsi_inject_kernel_gpu.mojo`), the towers,
the normaliser and the agent at these dims (G3.2). What only the box can
check — the GPU obs hook, the tracking-z pipeline, the capture of 16 updates —
has a diagnostic here (`--smoke`); step 3 of the run sheet passed it on
2026-09-10 (§12.9): 37 808 nodes captured and replayed, 480 updates.

THE LOOP (`train.py::train_online`, `fb/agent.py::maybe_update_rollout_context`,
`legged_robot_motions.py` resets), per batched step `s` of `N_ENVS` env steps:

    every T_EPISODE steps    reset every lane: `reset_batch` (the engine's
                             own bookkeeping) then `rsi_inject_kernel`: a
                             motion uniform, a row uniform inside it, the
                             lie-down transform with probability 0.3 and
                             a sign shared by the batch; then FK, body
                             velocities, observation. No termination;
                             `truncated` is not a terminal, so `done` is 0.
    every TRACK_LEN steps    draw N_TRACK lanes (with replacement) and
                             N_TRACK expert windows of TRACK_LEN + 1 rows;
                             z_track[n, t] = project(mean_{j<8} B(row t+1+j))
                             with the CURRENT B on NORMALISED rows
                             (`_sample_tracking_z`); those lanes are PINNED
                             to z_track[n, s mod TRACK_LEN] every step
                             (`FBOnlineAgent.enable_z_pin`), the others keep
                             the agent's own rule: hold 100 steps, redraw
                             from the z-buffer (`zbuf_frac 1.0`: the
                             reference draws from the buffer only, once it
                             is non-empty).
    every step               obs → prev_obs; action (uniform for the first
                             SEED_STEPS env steps, else π_z + N(0, 0.05²));
                             `step_batch`; record (prev_obs, a, obs, z, 0).
    after SEED_STEPS         16 updates per batched step, one captured
                             CUDA graph replayed (`maybe_capture_replay`),
                             each update = `FBCPROnlineAgent._train_kernels`
                             with the running normaliser inside.

Config = the released `config.json` at Fig. 13's 60 M size: h 1024, L 3
for F / Q_D / actor, B 256, D 1024 × 3, z 256, batch 1024, γ 0.98, τ_FB
0.01, τ_Q 0.005, lr 3e-4 (B and D 1e-5), ortho 100, reg 0.05, GP 10, actor
std 0.05, relabel 80 %, mixture 0.2 goal / 0.6 expert / 0.2 sphere,
z-buffer 8192, seq 8. No aux critic (G4), no noise, no DR, no history.

Ring: CAP 2 M rows × (527 + 29 + 527 + 256 + 1) floats = 10.7 GB; expert
table 441 k × 527 = 0.93 GB; tracking rows 512 × 250 × 527 = 0.27 GB.

Checkpoints `runs/<id>/checkpoints/step_<batched step>.ckpt` (FB file + `.cpr`
+ `.norm` sidecars), scored
by the G3.4 eval, never one alone.
"""

from std.math import sqrt
from std.random import seed, random_float64
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu import global_idx

from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.cuda import CUDAGraph, maybe_capture_replay
from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.nn.core.ptr import mptr
from noeira.core.run import RunContext, register_run, run_id_of_checkpoint
from noeira.core.dotenv import load_dotenv
from noeira.core.logger import CsvLogger, RemoteLogger, CompositeLogger
from noeira.io.artifact_sink import sink_for_run
from noeira.core.run_session import finish_run
from noeira.deep_agents.training.checkpoint import (
    announce_checkpoint, checkpoint_retain,
)
from noeira.io.fileio import remove_file, file_size
from noeira.io.proc import run_capture
from noeira.data.store import TrajectoryStore
from noeira.data.resident import IDX_DT
from noeira.deep_agents.fb import FBCPROnlineAgent
from noeira.deep_agents.fb.loss import fb_rank_eff_from_ortho
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA, G1_HIST_DIM, G1_N_ACT,
    G1_HIST_STEP, G1_HIST_MAX_AGE,
    g1_pack_full_obs_kernel, g1_hist_push_kernel, g1_hist_reset_kernel,
    g1_build_tail_spec, g1_scale_clip_kernel,
)
from noeira.envs.robots.unitree_g1_pd import G1_NORMALIZE_TO, G1_ACTION_CLIP
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMBNetFiltered, BFMActorTowerFiltered, BFMDNetFiltered,
)
from noeira.deep_agents.fb.kernels import (
    gather_rows_into_kernel,
    gather_rows_kernel, project_sphere_kernel, ensure_t, _blocks,
)
from noeira.deep_agents.fb.kernels import uniform01_kernel
from noeira.envs.robots import UnitreeG1Batched
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM, UNITREE_G1_PRIV_DIM,
)
from noeira.envs.robots import UnitreeG1
from noeira.deep_agents.fb.obs_norm import ObsNorm
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.envs.robots.g1_motion_priority import (
    G1_PRIO_REFRESH, g1_motion_priority, g1_fill_motion_table,
    g1_fill_window_table,
)
from noeira.envs.robots.g1_tracking_eval import (
    G1_D, G1_H, G1_L, G1_HB, G1_HD,
    G1_SEG_ROWS, G1TrackScore, g1_n_segments, g1_segment_row, g1_segment_pick,
    g1_score_segment,
)
from noeira.envs.robots.unitree_g1_rsi import (
    G1RsiTable, rsi_inject_kernel, G1_RSI_NQ, G1_RSI_NV, G1_LIE_DOWN_PROB,
    lie_down_selected,
)


# ── the recipe ────────────────────────────────────────────────────────────
comptime N_ENVS: Int = 1024
# ⚠ TWO widths (docs §12.34-12.36). `SP` is what the ENV produces and what
# `b` / `discriminator` consume — `state 64 | privileged 463`. `OBS` is the
# packed row `f`, `critic` and (filtered) the actor see; its last 401 are
# `last_action 29 | history 372`, maintained per lane during the rollout and
# DERIVED from the ring during training, never stored.
comptime SP: Int = UNITREE_G1_OBS_DIM         # 527
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM   # 928
comptime ACT: Int = UnitreeG1Model.ACTION_DIM  # 29
comptime D: Int = G1_D
comptime H: Int = G1_H
comptime L: Int = G1_L
comptime HB: Int = G1_HB
comptime HD: Int = G1_HD
comptime BATCH: Int = 1024
# ⚠ THE RING IS THE LARGEST THING ON THE CARD, not the model. Measured from
# `NOEIRA_ALLOC_TRACE=1` at H=2048/L=6 (docs §12.22): `r_obs` and `r_nxt` were
# 4020.7 MB EACH (CAP x 527 x 4) and `r_z` another 1953 MB — 9.76 GiB of a
# 25.9 GiB Tensor peak, against a 28.5 GiB pool that still OOM'd, so there is
# >= 2.6 GiB of non-Tensor overhead (MAX workspaces, the graph, cuBLAS) on top.
#
# `r_nxt` IS NOW GONE (docs §12.23): it was `r_obs` shifted by one step within
# a lane, so it is derived instead of stored and the ring costs 3256 B per
# transition rather than 5360. At CAP 2 M that frees 3.92 GiB, which is what
# pays for the CAP raise below.
#
# ⚠ CAP IS 1 M BECAUSE THE CARD SAYS SO (docs §12.42, MEASURED).
#
# The 2048/6 smoke peaked at 31 436 MiB of 32 607 — 30.70 GiB of 31.84, with
# 1.14 GiB free. Non-ring at this tower is 24.63 GiB MEASURED, against the
# 15.3 predicted from §12.21's table plus the 1024/3 anchor: that table was
# built at OBS 527 and does not carry the CPR critic, and the 928-wide inputs
# widen every embedding. The prediction was wrong by 9.7 GiB — the sixth VRAM
# estimate in this track and the fifth to miss.
#
#     CAP 2.00 M  ring 6.07 GiB  total 30.70  free 1.14   <- OOM band
#     CAP 1.50 M  ring 4.55 GiB  total 29.18  free 2.66
#     CAP 1.00 M  ring 3.04 GiB  total 27.66  free 4.18   <- here
#
# §12.22 OOM'd this card at ~29 GiB, so 1.5 M is not a margin. 1 M leaves
# 4.18 GiB.
#
# ⚠⚠ THIS CONFOUNDS THE NEXT COMPARISON. `g3_hist` was 1024/3 at CAP 2 M; this
# is 2048/6 at CAP 1 M — two axes. A WIN is still unambiguous (better with half
# the buffer). A LOSS is not, and needs 1024/3 + CAP 1 M as the control before
# anything is concluded. That control is also the CAP probe §12.23 has been
# asking for since the ring rewrite and which has never been run.
comptime CAP: Int = 1_000_000
comptime SEQ: Int = 8
comptime ZBUF: Int = 8192
comptime T_EPISODE: Int = 500
comptime Z_HOLD: Int = 100
comptime TRACK_LEN: Int = 250
comptime N_TRACK: Int = N_ENVS // 2
comptime SEED_STEPS: Int = 10_240
comptime UPDATES_PER_STEP: Int = 16
comptime EXPL_STD: Float64 = 0.05
comptime B_CHUNK: Int = 4096                  # rows per B forward while encoding tracking z

comptime EnvT = UnitreeG1Batched[N_ENVS]
comptime FNet = BFMFTower[OBS, ACT, D, H, L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, HB]
comptime ANet = BFMActorTowerFiltered[
    OBS, UNITREE_G1_STATE_DIM, G1_ACTOR_EXTRA, D, H, L, ACT
]
comptime DNet = BFMDNetFiltered[OBS, SP, D, HD]
comptime QNet = BFMFTower[OBS, ACT, D, H, L, 1]
comptime Agent = FBCPROnlineAgent[
    FNet, BNet, ANet, DNet, QNet, OBS, ACT, D, BATCH, CAP, N_ENVS, SEQ, ZBUF,
    G1_ACTOR_EXTRA,   # DERIVED_TAIL — the 401 the ring does NOT store (§12.36)
]
comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
# The in-loop tracking eval runs the SAME protocol as
# `bfm_zero_eval_tracking.mojo`, on the CPU, over the checkpoint that was just
# written — so the number in `metrics.csv` is the number those exact bytes
# score, and the checkpoint round-trip is exercised every time.
comptime EvalTrainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, 64, "cpu"]


# ── kernels of this driver ────────────────────────────────────────────────
def zero_kernel[N: Int](p: Pointer[Scalar[DT], MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < N:
        p[unsafe_offset=i] = Scalar[DT](0)


def draw_index_kernel[N: Int](
    u: Pointer[Scalar[DT], MutAnyOrigin],       # N uniforms
    table: Pointer[Scalar[DT], MutAnyOrigin],   # a table of candidates (as float)
    n_table: Int32,
    picks: Pointer[Scalar[DT], MutAnyOrigin],   # N picks (as float)
):
    """`picks[i] = table[floor(u[i] · n_table)]`."""
    var i = Int(global_idx.x)
    if i >= N:
        return
    var k = Int(u[unsafe_offset=i] * Scalar[DT](Int(n_table)))
    if k >= Int(n_table):
        k = Int(n_table) - 1
    picks[unsafe_offset=i] = table[unsafe_offset=k]


def track_row_index_kernel[N: Int, T: Int](
    starts: Pointer[Scalar[DT], MutAnyOrigin],  # N window starts (as float)
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],         # N * T row ids: start + 1 + t
):
    var i = Int(global_idx.x)
    if i >= N * T:
        return
    var n = i // T
    var t = i % T
    idx[unsafe_offset=i] = Scalar[IDX_DT](Int(starts[unsafe_offset=n]) + 1 + t)


def chunk_index_kernel[ROWS: Int](
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],   # the full row-id table
    off: Int32,
    dst: Pointer[Scalar[IDX_DT], MutAnyOrigin],   # ROWS ids from `off`
):
    var i = Int(global_idx.x)
    if i < ROWS:
        dst[unsafe_offset=i] = idx[unsafe_offset=Int(off) + i]


def copy_rows_kernel[W: Int, ROWS: Int](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    dst_off_rows: Int32,
):
    var i = Int(global_idx.x)
    if i < ROWS * W:
        dst[unsafe_offset=Int(dst_off_rows) * W + i] = src[unsafe_offset=i]


def track_mean_project_kernel[Dz: Int, N: Int, T: Int, S: Int](
    b: Pointer[Scalar[DT], MutAnyOrigin],      # N * T * Dz: B(row t+1) per window step
    z: Pointer[Scalar[DT], MutAnyOrigin],      # N * T * Dz: the tracking z
    radius: Scalar[DT],
):
    """`z[n, t] = project(mean_{j < min(S, T − t)} b[n, t + j])` — the
    reference's `_sample_tracking_z` mean over the next `seq_length` rows,
    then onto the radius-sqrt(d) sphere. One thread per (n, t)."""
    var i = Int(global_idx.x)
    if i >= N * T:
        return
    var n = i // T
    var t = i % T
    var m = S
    if T - t < S:
        m = T - t
    var acc = Array[Scalar[DT], Dz](fill=Scalar[DT](0))
    for j in range(m):
        var base = ((n * T) + t + j) * Dz
        for k in range(Dz):
            acc[k] += b[unsafe_offset=base + k]
    var nrm2 = Scalar[DT](0)
    for k in range(Dz):
        acc[k] = acc[k] / Scalar[DT](m)
        nrm2 += acc[k] * acc[k]
    var scale = radius / sqrt(nrm2 + Scalar[DT](1e-12))
    var out_base = (n * T + t) * Dz
    for k in range(Dz):
        z[unsafe_offset=out_base + k] = acc[k] * scale


def pin_step_kernel[Dz: Int, N: Int, T: Int](
    lanes: Pointer[Scalar[DT], MutAnyOrigin],   # N lane ids (as float)
    z_track: Pointer[Scalar[DT], MutAnyOrigin], # N * T * Dz
    t: Int32,
    z_pin: Pointer[Scalar[DT], MutAnyOrigin],   # LANES * Dz
    mask: Pointer[Scalar[DT], MutAnyOrigin],    # LANES
):
    """`z_pin[lane_n] = z_track[n, t]`, `mask[lane_n] = 1`. A lane drawn
    twice takes whichever writer lands last — either is a valid pin."""
    var i = Int(global_idx.x)
    if i >= N * Dz:
        return
    var n = i // Dz
    var k = i % Dz
    var lane = Int(lanes[unsafe_offset=n])
    z_pin[unsafe_offset=lane * Dz + k] = z_track[unsafe_offset=(n * T + Int(t)) * Dz + k]
    if k == 0:
        mask[unsafe_offset=lane] = Scalar[DT](1)


# ── helpers ───────────────────────────────────────────────────────────────
def _flag(name: String, default: String) -> String:
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == name and i + 1 < len(args):
            return String(args[i + 1])
    return default


def _has(name: String) -> Bool:
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == name:
            return True
    return False


def _clip_ranges(
    mut store: TrajectoryStore, window: Int,
    mut begin: List[Int], mut end: List[Int],
) raises:
    """The `[begin, end)` slice of `_valid_starts`'s output belonging to each
    clip. `_valid_starts` walks episodes in order and appends contiguously, so
    this describes the layout that table ALREADY has — motion prioritization
    expands it clip by clip."""
    begin.clear()
    end.clear()
    var at = 0
    for e in range(store.n_episodes()):
        var n = store.episodes.length_of(e)
        var count = n - window
        if count < 0:
            count = 0
        begin.append(at)
        at += count
        end.append(at)


def _valid_starts(mut store: TrajectoryStore, window: Int) raises -> List[Int]:
    """Window starts whose rows `start .. start + window` stay inside one
    episode (`start + window` is the last NEXT row the window touches)."""
    var out = List[Int]()
    for e in range(store.n_episodes()):
        var off = store.episodes.start_of(e)
        var n = store.episodes.length_of(e)
        var last = off + n - 1 - window
        var s = off
        while s <= last:
            out.append(s)
            s += 1
    return out^


def _upload_ints(ctx: DeviceContext, xs: List[Int]) raises -> DeviceBuffer[IDX_DT]:
    var h = ctx.enqueue_create_host_buffer[IDX_DT](len(xs))
    for i in range(len(xs)):
        h[i] = Scalar[IDX_DT](xs[i])
    var d = ctx.enqueue_create_buffer[IDX_DT](len(xs))
    ctx.enqueue_copy(d, h)
    ctx.synchronize()
    return d^


def _upload_floats(ctx: DeviceContext, xs: List[Int]) raises -> Tensor:
    var t = Tensor()
    ensure_t["gpu"](t, len(xs), Optional(ctx))
    for i in range(len(xs)):
        t.data[i] = Scalar[DT](xs[i])
    t.upload_resident(ctx)
    return t^


def _score_tracking(
    mut t: EvalTrainer,
    mut env: UnitreeG1[DType.float64],
    ref rsi: G1RsiTable,
    ref st: List[Scalar[DType.float32]],
    ref pv: List[Scalar[DType.float32]],
    ref qpos_col: List[Scalar[DType.float32]],
    ckpt: String,
    max_segments: Int,
    mut ach: List[Float64],
    mut tgt: List[Float64],
    mut b_in: Tensor,
    mut b_out: Tensor,
    mut z_seg: Tensor,
    mut obs_t: Tensor,
    mut z1: Tensor,
    mut act_out: Tensor,
    mut per_clip: List[Float64],
) raises -> G1TrackScore:
    """The tracking protocol over the checkpoint just written.

    `per_clip` is filled with each motion's mean EMD — that is what motion
    PRIORITIZATION reweights the sampling by, and the reference refreshes it
    from this same eval (`train.py:350`). A clip with no scored segment keeps
    its entry at 0, which `g1_motion_priority` clamps to the floor weight.

    Loading the FILE rather than reading the live GPU nets is deliberate: the
    number logged is then the number those exact bytes score, and a checkpoint
    that fails to round-trip shows up here instead of hours later on the
    laptop. `<ckpt>.norm` is read beside it exactly as the standalone eval
    does.
    """
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")
    var sum_d = 0.0
    var sum_e = 0.0
    var sum_p = 0.0
    var n = 0
    per_clip.clear()
    for _ in range(rsi.n_ep):
        per_clip.append(0.0)
    for clip in range(rsi.n_ep):
        var clip_e = 0.0
        var clip_n = 0
        var n_avail = g1_n_segments(Int(rsi.ep_len.data[clip]))
        var n_seg = n_avail
        if n_seg > max_segments:
            n_seg = max_segments
        for k in range(n_seg):
            # spread across the clip, not the first `n_seg` — see
            # `g1_segment_pick`; taking the opening windows reads ~0.19 low
            var seg = g1_segment_pick(n_avail, n_seg, k)
            var r0 = g1_segment_row(Int(rsi.ep_offset.data[clip]), seg)
            var sc = g1_score_segment[FNet, BNet, ANet, OBS, ACT, D, 64](
                t, env, rsi, st, pv, qpos_col, norm, r0,
                ach, tgt, b_in, b_out, z_seg, obs_t, z1, act_out,
            )
            sum_d += sc.distance
            sum_e += sc.emd
            sum_p += sc.proximity
            clip_e += sc.emd
            clip_n += 1
            n += 1
        if clip_n > 0:
            per_clip[clip] = clip_e / Float64(clip_n)
    var f = Float64(n if n > 0 else 1)
    return G1TrackScore(sum_d / f, sum_e / f, sum_p / f, n)



def _free_gb(path: String) raises -> Float64:
    """Free space on `path`'s filesystem, GB. -1 when it cannot be read.

    ⚠ VIA `df`, not `statvfs`. The struct `statvfs` fills differs between
    macOS and Linux and would have to be declared per platform; `df -k` prints
    the same column on both and this runs once per checkpoint, not per step.
    """
    try:
        var out = run_capture(String("df -k ") + path + String(" 2>/dev/null"))
        var lines = out.split("\n")
        if len(lines) < 2:
            return -1.0
        var f = lines[1].split()
        if len(f) < 4:
            return -1.0
        return Float64(String(f[3])) * 1024.0 / 1073741824.0
    except:
        return -1.0


def _drop_checkpoint(path: String):
    """Remove a checkpoint's THREE files. A checkpoint is not one file:
    `<p>` (the FB nets), `<p>.cpr` (the CPR critic) and `<p>.norm` (the
    normaliser sidecar) — 1.17 GB together at 2048/6. Deleting only `<p>`
    would leave the two largest behind and the retention policy would appear
    to do nothing."""
    var sfx = List[String]()
    sfx.append(String(""))
    sfx.append(String(".cpr"))
    sfx.append(String(".norm"))
    for i in range(len(sfx)):
        try:
            remove_file(path + sfx[i])
        except:
            pass


def main() raises:
    var smoke = _has("--smoke")
    var no_graph = _has("--no-graph")
    var total_env_steps = atol(_flag(String("--steps"), String(192_000_000)))
    # ⚠ `--smoke` is a PRESET, not an override: it pins the length only when
    # `--steps` was not given. It used to win unconditionally, so
    # `--smoke --steps 200000` silently ran 40 batched steps and looked like
    # `--steps` was ignored. An explicit flag beats a preset.
    if smoke and not _has("--steps"):
        total_env_steps = N_ENVS * 40
    var ups = atol(_flag(String("--ups"), String(UPDATES_PER_STEP)))
    var tag = _flag(String("--tag"), String("g3_priv"))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var ckpt_every = atol(_flag(String("--ckpt-every"), String(2000)))
    # ⚠ RETENTION, because the 2048/6 run DIED OF A FULL DISK at 38 k steps.
    # A checkpoint here is 1.17 GB across three files and the cadence was
    # every 2000 batched steps, so 19 of them filled a 60 GB box and
    # `step_38000.ckpt.cpr` was cut off mid-write. The run did not fail on the
    # GPU, on the maths or on the data — it ran out of somewhere to put the
    # next file, 20 h and $16 in.
    #
    # Keep the last `--ckpt-keep`, the BEST-scoring one, and a sparse
    # milestone ladder. At the defaults a 64 h / 120 M-frame run holds
    # 3 + 1 + ~5 = 9 checkpoints ~ 10.5 GB, against 192 ~ 225 GB unpruned.
    var ckpt_keep = atol(_flag(String("--ckpt-keep"), String(3)))
    var ckpt_milestone = atol(_flag(String("--ckpt-milestone"), String(25000)))
    # Below this, SKIP the write rather than start one that cannot finish. A
    # skipped checkpoint costs one interval; a truncated one is what ended the
    # last run, and it is indistinguishable from a good file until it is read.
    var ckpt_min_free = Float64(
        String(_flag(String("--ckpt-min-free-gb"), String("4.0")))
    )
    var ckpt_written = List[Int]()
    var ckpt_best_step = -1
    var ckpt_best_emd = 1e18
    var ckpt_skipped = 0   # batched steps
    var print_every = atol(_flag(String("--print-every"), String(100)))
    # ── resume ───────────────────────────────────────────────────────────
    # `--resume PATH` restores the online nets + the `.norm` sidecar from a
    # checkpoint; `--start-at N` is how many ENV steps that checkpoint already
    # represents, so `--steps` stays the TOTAL target and this run does the
    # remainder. Both the printed step and the logged step carry the offset,
    # so a resumed run's curves continue the first one's instead of restarting
    # at zero, and its checkpoints keep climbing the same ladder.
    #
    # ⚠ WHAT A RESUME DOES NOT CARRY. `save_state` writes the ONLINE nets only
    # (`fb/trainer.mojo`): the targets are hard-copied from them on load, the
    # Adam moments re-warm, and THE REPLAY RING STARTS EMPTY. The ring is a
    # 2 M-transition rolling window, i.e. ~32 min of collection at 1024 lanes,
    # so a resume costs about that before the batch distribution is back to
    # normal. That is proportionate to a 33-min checkpoint cadence; it is not
    # free, and it is why this is not bit-identical to an uninterrupted run.
    #
    # ⚠ THE SEED PHASE RE-RUNS ON A RESUME, AND THAT IS CORRECT. `env_steps`
    # counts THIS execution, so the `env_steps >= SEED_STEPS` gate below holds
    # the updates off for the first 10 batched steps — which is what an empty
    # ring needs. Offsetting it would start training against ~0 transitions.
    var resume_path = _flag(String("--resume"), String(""))
    var start_at = atol(_flag(String("--start-at"), String(0)))
    var s_off = start_at // N_ENVS  # the batched-step offset
    var seed_v = atol(_flag(String("--seed"), String(20260909)))
    var lie_prob = atof(_flag(String("--lie-prob"), String(G1_LIE_DOWN_PROB)))
    var track_on = not _has("--no-track")
    # 0 turns the in-loop tracking eval off. 1 is one segment per clip (40
    # segments, ~35 s) — ~3 % of a 20-minute checkpoint interval, and the only
    # number in the file that measures the thing the run is FOR.
    # 5 per clip = ~200 of the store's ~862 windows, STRATIFIED across each
    # clip (`g1_segment_pick`). The old default of 1 scored 40 — the opening
    # ten seconds of every motion, which reads ~0.19 low against the
    # reference's all-862 number (docs §12.29). 5 is the coverage the mid and
    # ring runs already paid for, so the cost is known.
    var eval_segments = atol(_flag(String("--eval-segments"), String(5)))
    # The eval used to run ONLY at checkpoints, so the curve was three points
    # and "the peak is at 4000" partly meant "4000 was the best of the three
    # we sampled". `--eval-every` (batched steps) decouples the two; it
    # defaults to the checkpoint cadence, so nothing moves unless asked.
    var eval_every = atol(_flag(String("--eval-every"), String(0)))
    # Motion prioritization: resample the motions we track WORST more often
    # (`g1_motion_priority.mojo`). The reference refreshes it from the tracking
    # eval every `eval_every_steps` = 9.6 M env steps, so it does nothing at
    # all below that — and matters above it, which is where a long run lives.
    var prio_on = not _has("--no-prio")
    seed(seed_v)

    # ⚠⚠ `agent.save_state(tag + "." + String(s))` WAS THE SECOND HAND-ROLLED
    # TAG MECHANISM IN THIS TREE, beside the one `examples/fb/fb_train_gpu.mojo`
    # carried. Two copies of a rule is the shape `_a_rule_written_inline_twice_drifts`
    # names as the most frequent defect here, so both are replaced by
    # `RunContext` in one change — removing only one would have made the
    # survivor the third copy.
    #
    # ⚠ CHECKPOINTS MOVE. They used to land as `g3_priv.2000` in the working
    # directory; they now land in `runs/<id>/checkpoints/step_2000.ckpt`, with
    # the `.cpr` and `.norm` sidecars beside them. `bfm_zero_eval_tracking`
    # already takes `--ckpt <path>` and needs no change; the paths this run
    # writes are printed below so the next command is copy-pasteable.
    var run = RunContext(
        project=String("g1"),
        driver=String("examples/g1/bfm_zero_train_gpu.mojo"),
        slug=String("bfm-zero") + ("-" + tag if tag.byte_length() > 0 else ""),
        env=String("builtin:unitree_g1"),
        dataset=store_path,
        seed=seed_v,
        resumed_from=run_id_of_checkpoint(resume_path),
    )
    run.set_tag(tag)
    # `project-resume <run_id>` continues this run from its last checkpoint,
    # at the env step it saved (the replay re-warms from the store).
    run.set_resume_args(String("--resume {ckpt}"))
    print("run:", run.dir)

    # ── the run's own record: a CSV that outlives the ssh session ─────────
    # ⚠ A 55-HOUR RUN THAT ONLY PRINTS TO STDOUT HAS NO RECORD. The terminal
    # is not an artefact: a dropped connection, a full scrollback or a box
    # that dies takes the whole curve with it. `CsvLogger` writes
    # `step,wall_time_ms,name,value` into the run directory as the run goes.
    # ⚠ `RemoteLogger` WITH NO URL IS INERT — its POST sink is built lazily on
    # the first payload — so this costs nothing with no `.env`, and its header
    # states the rule that matters here: a dashboard that cannot be reached
    # must never take the training run with it.
    var env_vars = load_dotenv()
    var remote = RemoteLogger(
        server_url=env_vars.get("NOEIRA_CLOUD_URL", ""),
        run_name=run.name(),
        run_id=run.id,
        buffer_size=64,
        api_key=env_vars.get("NOEIRA_CLOUD_API_KEY", ""),
    )
    # ⚠ THE CONFIG GOES TO BOTH HALVES: `/runs` for the dashboard and
    # `metrics.config.kv` beside the CSV, so a CSV read a week later still says
    # what produced it. Hence the composite is built BEFORE `set_config`.
    var logger = CompositeLogger(CsvLogger(run.metrics_path()), remote^)
    logger.set_config("algorithm", "BFM-Zero FB-CPR")
    logger.set_config("env", "unitree_g1")
    logger.set_config("target", "gpu")
    logger.set_config("lanes", String(N_ENVS))
    logger.set_config("obs", String(OBS))
    logger.set_config("act", String(ACT))
    logger.set_config("d", String(D))
    logger.set_config("h", String(H))
    logger.set_config("layers", String(L))
    logger.set_config("updates_per_step", String(ups))
    logger.set_config("seed_steps", String(SEED_STEPS))
    logger.set_config("store", store_path)
    logger.set_config("resume_from", resume_path)
    logger.set_config("start_at", String(start_at))
    logger.set_config("t_episode", String(T_EPISODE))
    logger.set_config("track_len", String(TRACK_LEN))
    logger.set_config("lie_prob", String(lie_prob))
    # ⚠ AFTER the config and before step 0 — `register_run` seeds the
    # dashboard from the run (id, project, commit, seed, host) and POSTs
    # `/runs`. A run that dies before step 0 otherwise never appears at all.
    register_run(run, logger)

    # ⚠⚠ THE ARTIFACT UPLINK. A checkpoint leaves the box WHILE the run is
    # going, so a rented box that dies at hour 40 does not take 40 hours of
    # weights with it. `sink_for_run` returns None when `.env` names no
    # monitor — a box with no credentials must still train — and every
    # `announce_checkpoint` below is a no-op on a None, so there is no branch.
    var artifacts = sink_for_run(run.id, run.dir)

    var ctx = DeviceContext()
    print("BFM-Zero G1 privileged arm: lanes", N_ENVS, " obs", OBS, " act", ACT, " d", D, " h", H, " L", L)

    # ── the store: expert rows, window starts, the reset table ────────
    var store = TrajectoryStore(store_path)
    var n_rows = store.n_rows()
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    # ⚠ `upload` REALLOCATES the device buffer on every call (by design — it
    # is the resize path), so `ensure_t["gpu"]` + `upload` allocates the table
    # TWICE and leaves both live across the copy. At 441 131 x 527 that is
    # 886.8 MB of pure transient, and the alloc trace showed it: one id, two
    # allocations. `upload_resident` fills the buffer `ensure_t` already made.
    # Same pattern applied to every one-shot table below, and to the
    # prioritization refresh — where `upload` would ALSO hand the RSI kernel a
    # new pointer every 9.6 M steps.
    var eobs = Tensor()
    # ⚠ SP-wide, not OBS-wide. The expert rows are only ever read by `b` (the
    # window encoding) and the discriminator, whose reference filters are both
    # `state + privileged_state` — so the derived tail would be 0.7 GB of
    # columns nothing reads. The B forward takes an OBS-wide row, so the
    # CHUNK is widened and its tail zeroed once, not the table.
    ensure_t["gpu"](eobs, n_rows * SP, Optional(ctx))
    for r in range(n_rows):
        for i in range(UNITREE_G1_STATE_DIM):
            eobs.data[r * SP + i] = Scalar[DT](st[r * UNITREE_G1_STATE_DIM + i])
        for i in range(UNITREE_G1_PRIV_DIM):
            eobs.data[r * SP + UNITREE_G1_STATE_DIM + i] = Scalar[DT](pv[r * UNITREE_G1_PRIV_DIM + i])
    eobs.upload_resident(ctx)
    var starts8 = _valid_starts(store, SEQ)
    var starts250 = _valid_starts(store, TRACK_LEN + 1)
    # motion prioritization rebuilds these two tables clip by clip
    var beg8 = List[Int]()
    var end8 = List[Int]()
    var beg250 = List[Int]()
    var end250 = List[Int]()
    _clip_ranges(store, SEQ, beg8, end8)
    _clip_ranges(store, TRACK_LEN + 1, beg250, end250)
    var starts8_dev = _upload_ints(ctx, starts8)
    var starts250_t = _upload_floats(ctx, starts250)
    var rsi = G1RsiTable.from_store(store)
    ensure_t["gpu"](rsi.rows, rsi.n_rows * (G1_RSI_NQ + G1_RSI_NV), Optional(ctx))
    ensure_t["gpu"](rsi.ep_offset, rsi.n_ep, Optional(ctx))
    ensure_t["gpu"](rsi.ep_len, rsi.n_ep, Optional(ctx))
    rsi.rows.upload_resident(ctx)
    rsi.ep_offset.upload_resident(ctx)
    rsi.ep_len.upload_resident(ctx)
    print("  store:", n_rows, "rows,", store.n_episodes(), "clips;", len(starts8), "expert windows,", len(starts250), "tracking windows")

    # ── motion prioritization tables ──────────────────────────────────
    # FIXED LENGTH, rewritten IN PLACE. `_gather_expert_windows` reads the
    # expert start buffer from inside the captured graph, so the graph bakes
    # in that buffer's POINTER — handing the agent a NEW, longer table at a
    # refresh would leave it reading freed memory 9.6 M steps into a run.
    comptime PRIO_MOTION_LEN = 4096
    var prio_motion = Tensor()
    ensure_t["gpu"](prio_motion, PRIO_MOTION_LEN, Optional(ctx))
    var n_prio_motion = 0
    var prio_tbl = List[Int]()
    var cur8 = List[Int]()
    var cur250 = List[Int]()
    for _ in range(store.n_episodes()):
        cur8.append(0)
        cur250.append(0)
    var next_prio = G1_PRIO_REFRESH
    var prio_emd = List[Float64]()

    # ── the in-loop tracking eval ─────────────────────────────────────
    # ⚠ HOST RAM, AND IT SCALES WITH THE TOWER. The CPU `FBTrainer` is the
    # nets + targets + Adam moments again on the HOST: 4 copies x 4 B x the
    # F/B/actor parameter count. At `G1_H = 2048, G1_L = 6` that is ~2.7 GB
    # (it was ~0.35 GB at 1024/3), and it is allocated WHETHER OR NOT the eval
    # runs — `--eval-segments 0` skips the scoring and the ~63 MB `qpos`
    # column, not that allocation. Say so rather than let the flag imply
    # otherwise.
    var eval_on = eval_segments > 0
    var eval_env = UnitreeG1[DType.float64]()
    var eval_t = EvalTrainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    # The target is read from the `qpos` COLUMN, not from `rsi.rows` — they
    # hold the same numbers, and that is the point: a corrupted RSI table
    # would shift the reset AND the target together and the score would still
    # look fine. Two reads, one of them independent.
    var eval_qpos = List[Scalar[DType.float32]]()
    if eval_on:
        _ = eval_env.reset()
        eval_qpos = store.load_column[DType.float32](String("qpos"))
    var eval_b_in = Tensor.alloc(G1_SEG_ROWS * OBS if eval_on else 1)
    var eval_b_out = Tensor()
    var eval_z_seg = Tensor.alloc(G1_SEG_ROWS * D if eval_on else 1)
    var eval_obs = Tensor.alloc(OBS if eval_on else 1)
    var eval_z1 = Tensor.alloc(D if eval_on else 1)
    var eval_act = Tensor.alloc(ACT if eval_on else 1)
    var eval_ach = List[Float64](
        length=(G1_SEG_ROWS * ACT if eval_on else 1), fill=0.0
    )
    var eval_tgt = List[Float64](
        length=(G1_SEG_ROWS * ACT if eval_on else 1), fill=0.0
    )

    # ── env and agent ─────────────────────────────────────────────────
    var env = EnvT(ctx)
    var agent = Agent.make(
        ctx, lr=3e-4, lr_b=1e-5, lr_d=1e-5, lr_q=3e-4, gamma=0.98, tau=0.01,
        tau_q=0.005, ortho_weight=100.0, max_grad_norm=0.0, bc_weight=0.0,
        act_l2_weight=0.0, reg_coeff=0.05, gp_coef=10.0,
        learning_starts=SEED_STEPS, action_scale=1.0, expl_std=EXPL_STD,
        z_hold=Z_HOLD, zbuf_frac=1.0, keep_frac=0.2, p_goal=0.2, p_expert=0.6,
        seed=UInt64(seed_v), normalize_obs=True,
    )
    # ⚠ The agent derives a tail it cannot interpret (§12.36): the env hands
    # it the layout, and the scaling its PD chain applies to a stored action.
    var tail_spec = List[Int32]()
    g1_build_tail_spec(tail_spec)
    agent.attach_tail_spec(tail_spec)
    agent.set_action_norm(G1_NORMALIZE_TO, G1_ACTION_CLIP)
    agent.attach_expert_windows(eobs^, starts8_dev^, len(starts8))
    if track_on:
        agent.base.enable_z_pin()

    # ⚠⚠ A FAILED `--resume` MUST NOT FALL BACK TO TRAINING FROM SCRATCH.
    # The two runs look identical in the log until the curve starts from zero
    # hours later, and on a 55-hour run the wasted box time IS the cost of the
    # mistake. Refuse instead. (Same rule as
    # `examples/so101/sac_so_arm101_reach_training_gpu.mojo`.)
    if resume_path.byte_length() > 0:
        try:
            agent.load_state(resume_path)
            print("resumed from", resume_path, " at env step", start_at)
            print(
                "  ⚠ online nets + normaliser restored; the targets are"
                " hard-copied from them,\n    Adam moments re-warm, and the"
                " replay ring starts EMPTY (~32 min to refill)."
            )
        except e:
            print("ERROR: --resume given but", resume_path, "did not load:")
            print("   ", e)
            print("Refusing to silently train from scratch. Drop --resume to")
            print("start fresh, or pass a path that exists.")
            return

    # ── rollout buffers ───────────────────────────────────────────────
    # `prev_obs` is the PACKED row the action was chosen from — that is what
    # the ring must store the stored head of, and what the eval reproduces.
    var prev_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var prev_env_obs = ctx.enqueue_create_buffer[DT](N_ENVS * SP)
    # ── the actor's other 401, per lane (docs §12.34-12.36) ──────────
    var full_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_last = ctx.enqueue_create_buffer[DT](N_ENVS * G1_N_ACT)
    var h_hist = ctx.enqueue_create_buffer[DT](N_ENVS * G1_HIST_DIM)
    var h_live = ctx.enqueue_create_buffer[DT](N_ENVS)
    var h_all = ctx.enqueue_create_buffer[DT](N_ENVS)
    h_last.enqueue_fill(Scalar[DT](0.0))
    h_hist.enqueue_fill(Scalar[DT](0.0))
    h_live.enqueue_fill(Scalar[DT](0.0))   # the reset row is never pushed
    h_all.enqueue_fill(Scalar[DT](1.0))    # a mask selecting every lane

    def _hist_reset() capturing raises:
        """Zero `last_action` and the history, and skip the next push.

        Every lane resets together under `_rsi_reset`, so the mask is all
        ones. Clearing `live` is what implements "the reset observation is
        never pushed"."""
        ctx.enqueue_function[g1_hist_reset_kernel[N_ENVS]](
            mptr(h_last.unsafe_ptr()), mptr(h_hist.unsafe_ptr()),
            mptr(h_live.unsafe_ptr()), mptr(h_all.unsafe_ptr()),
            grid_dim=_blocks(N_ENVS * (G1_N_ACT + G1_HIST_DIM)), block_dim=TPB,
        )

    def _pack_obs() capturing raises:
        """`full_obs = [env._obs 527 | last_action 29 | history 372]`."""
        ctx.enqueue_function[g1_pack_full_obs_kernel[N_ENVS]](
            mptr(env._obs.unsafe_ptr()), mptr(h_last.unsafe_ptr()),
            mptr(h_hist.unsafe_ptr()), mptr(full_obs.unsafe_ptr()),
            grid_dim=_blocks(N_ENVS * OBS), block_dim=TPB,
        )

    def _hist_advance() capturing raises:
        """Push this step's state with the PREVIOUS action, then take the new
        one — the reference's order (`_push` then `last_action =`), which the
        history gate caught an off-by-one in once."""
        ctx.enqueue_function[g1_hist_push_kernel[N_ENVS]](
            mptr(env._obs.unsafe_ptr()), mptr(h_last.unsafe_ptr()),
            mptr(h_hist.unsafe_ptr()), mptr(h_live.unsafe_ptr()),
            grid_dim=_blocks(N_ENVS * G1_HIST_STEP), block_dim=TPB,
        )
        h_live.enqueue_fill(Scalar[DT](1.0))
        ctx.enqueue_function[g1_scale_clip_kernel[N_ENVS * G1_N_ACT]](
            mptr(h_last.unsafe_ptr()), mptr(env._action.unsafe_ptr()),
            Scalar[DT](G1_NORMALIZE_TO), Scalar[DT](G1_ACTION_CLIP),
            grid_dim=_blocks(N_ENVS * G1_N_ACT), block_dim=TPB,
        )
    var reward0 = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done0 = ctx.enqueue_create_buffer[DT](N_ENVS)
    var ao = ctx.enqueue_create_buffer[DT](N_ENVS * 2 * ACT)
    var alp = ctx.enqueue_create_buffer[DT](N_ENVS * (ACT + 1))
    ctx.enqueue_function[zero_kernel[N_ENVS]](mptr(reward0.unsafe_ptr()), grid_dim=_blocks(N_ENVS), block_dim=TPB)
    ctx.enqueue_function[zero_kernel[N_ENVS]](mptr(done0.unsafe_ptr()), grid_dim=_blocks(N_ENVS), block_dim=TPB)
    var u_rsi = ctx.enqueue_create_buffer[DT](N_ENVS * 3)
    var row_got = ctx.enqueue_create_buffer[DT](N_ENVS)
    var h_row_got = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    # The lie-down diagnostic reads the SAME uniforms the inject kernel drew,
    # and decides with the same `lie_down_selected` — no second copy of the rule.
    var h_u_rsi = ctx.enqueue_create_host_buffer[DT](N_ENVS * 3)
    var rng_seed = UInt64(seed_v) + 99
    var rng_off = UInt64(0)

    # tracking-z pipeline buffers
    var u_track = ctx.enqueue_create_buffer[DT](N_TRACK * 2)
    var track_starts = ctx.enqueue_create_buffer[DT](N_TRACK)
    var track_lanes = ctx.enqueue_create_buffer[DT](N_TRACK)
    var lane_table = Tensor()
    ensure_t["gpu"](lane_table, N_ENVS, Optional(ctx))
    for l in range(N_ENVS):
        lane_table.data[l] = Scalar[DT](l)
    lane_table.upload_resident(ctx)
    # the B forward runs in whole chunks, so the row-id table and the B
    # output are PADDED to N_CHUNKS * B_CHUNK: the tail rows past
    # N_TRACK * TRACK_LEN gather row 0 and are never read by the mean
    comptime N_CHUNKS = (N_TRACK * TRACK_LEN + B_CHUNK - 1) // B_CHUNK
    comptime N_PAD = N_CHUNKS * B_CHUNK
    var track_idx = ctx.enqueue_create_buffer[IDX_DT](N_PAD)
    ctx.enqueue_memset(track_idx, 0)
    var chunk_idx = ctx.enqueue_create_buffer[IDX_DT](B_CHUNK)
    var chunk_in = Tensor()
    var chunk_out = Tensor()
    ensure_t["gpu"](chunk_in, B_CHUNK * OBS, Optional(ctx))
    # the derived tail of these rows is never gathered into; zero it ONCE so
    # `b` is not fed whatever the allocator left behind
    chunk_in.dev.value().enqueue_fill(Scalar[DT](0.0))
    ensure_t["gpu"](chunk_out, B_CHUNK * D, Optional(ctx))
    var track_b = ctx.enqueue_create_buffer[DT](N_PAD * D)
    var track_z = ctx.enqueue_create_buffer[DT](N_TRACK * TRACK_LEN * D)

    def _rsi_reset(lie_sign: Float64) capturing raises:
        """`reset_batch`, then every lane injected from the store, then the
        FK / velocity / observation refresh — what `set_state` does on CPU."""
        env.reset_batch[N_ENVS](Optional(ctx), rng_seed + rng_off)
        comptime NU = N_ENVS * 3
        ctx.enqueue_function[uniform01_kernel[NU]](
            mptr(u_rsi.unsafe_ptr()), rng_seed + 1, rng_off,
            grid_dim=_blocks(NU), block_dim=TPB,
        )
        rng_off += UInt64(2 * NU)
        ctx.enqueue_function[rsi_inject_kernel[N_ENVS, NQ, NV]](
            mptr(rsi.rows.dev.value().unsafe_ptr()),
            mptr(rsi.ep_offset.dev.value().unsafe_ptr()),
            mptr(rsi.ep_len.dev.value().unsafe_ptr()),
            Int32(rsi.n_ep),
            mptr(u_rsi.unsafe_ptr()),
            Scalar[DT](lie_prob), Scalar[DT](lie_sign),
            mptr(env.d.qpos.dev.value().unsafe_ptr()),
            mptr(env.d.qvel.dev.value().unsafe_ptr()),
            mptr(row_got.unsafe_ptr()),
            mptr(prio_motion.dev.value().unsafe_ptr()), Int32(n_prio_motion),
            grid_dim=_blocks(N_ENVS), block_dim=TPB,
        )
        env._run_fields_fk(ctx)
        env._run_fields_vel(ctx)
        env._extract_obs_only(ctx)

    def _draw_tracking() capturing raises:
        """N_TRACK lanes and windows; B on the windows' next rows with the
        current normaliser and B; the mean-of-8 projected tracking z."""
        comptime NU2 = N_TRACK * 2
        ctx.enqueue_function[uniform01_kernel[NU2]](
            mptr(u_track.unsafe_ptr()), rng_seed + 2, rng_off,
            grid_dim=_blocks(NU2), block_dim=TPB,
        )
        rng_off += UInt64(2 * NU2)
        ctx.enqueue_function[draw_index_kernel[N_TRACK]](
            mptr(u_track.unsafe_ptr()), mptr(starts250_t.dev.value().unsafe_ptr()),
            Int32(len(starts250)), mptr(track_starts.unsafe_ptr()),
            grid_dim=_blocks(N_TRACK), block_dim=TPB,
        )
        ctx.enqueue_function[draw_index_kernel[N_TRACK]](
            mptr(u_track.unsafe_ptr()).unsafe_offset(N_TRACK),
            mptr(lane_table.dev.value().unsafe_ptr()),
            Int32(N_ENVS), mptr(track_lanes.unsafe_ptr()),
            grid_dim=_blocks(N_TRACK), block_dim=TPB,
        )
        ctx.enqueue_function[track_row_index_kernel[N_TRACK, TRACK_LEN]](
            mptr(track_starts.unsafe_ptr()), mptr(track_idx.unsafe_ptr()),
            grid_dim=_blocks(N_TRACK * TRACK_LEN), block_dim=TPB,
        )
        for c in range(N_CHUNKS):
            ctx.enqueue_function[chunk_index_kernel[B_CHUNK]](
                mptr(track_idx.unsafe_ptr()), Int32(c * B_CHUNK), mptr(chunk_idx.unsafe_ptr()),
                grid_dim=_blocks(B_CHUNK), block_dim=TPB,
            )
            # SP-wide source into OBS-wide rows; the tail was zeroed once at
            # allocation and `b` filters it out anyway (§12.34)
            ctx.enqueue_function[gather_rows_into_kernel[SP, OBS, B_CHUNK]](
                mptr(agent.exp_obs.dev.value().unsafe_ptr()), mptr(chunk_idx.unsafe_ptr()),
                mptr(chunk_in.dev.value().unsafe_ptr()),
                grid_dim=_blocks(B_CHUNK * SP), block_dim=TPB,
            )
            agent.base.obs_ema.apply[B_CHUNK](chunk_in)
            call_forward["gpu", B_CHUNK](
                agent.base.t.bnet.online, TensorRefs[1, MutAnyOrigin](chunk_in),
                chunk_out, Optional(ctx),
            )
            ctx.enqueue_function[copy_rows_kernel[D, B_CHUNK]](
                mptr(chunk_out.dev.value().unsafe_ptr()), mptr(track_b.unsafe_ptr()),
                Int32(c * B_CHUNK),
                grid_dim=_blocks(B_CHUNK * D), block_dim=TPB,
            )
        ctx.enqueue_function[track_mean_project_kernel[D, N_TRACK, TRACK_LEN, SEQ]](
            mptr(track_b.unsafe_ptr()), mptr(track_z.unsafe_ptr()),
            Scalar[DT](sqrt(Float64(D))),
            grid_dim=_blocks(N_TRACK * TRACK_LEN), block_dim=TPB,
        )
        ctx.enqueue_function[zero_kernel[N_ENVS]](
            mptr(agent.base.z_pin_mask.dev.value().unsafe_ptr()),
            grid_dim=_blocks(N_ENVS), block_dim=TPB,
        )

    def _pin_step(t: Int) capturing raises:
        ctx.enqueue_function[pin_step_kernel[D, N_TRACK, TRACK_LEN]](
            mptr(track_lanes.unsafe_ptr()), mptr(track_z.unsafe_ptr()), Int32(t),
            mptr(agent.base.z_pin.dev.value().unsafe_ptr()),
            mptr(agent.base.z_pin_mask.dev.value().unsafe_ptr()),
            grid_dim=_blocks(N_TRACK * D), block_dim=TPB,
        )

    def _captured_updates() capturing raises -> None:
        for _ in range(ups):
            agent.train_device_kernels()

    var train_graph: Optional[CUDAGraph] = None
    # The REMAINDER when resuming: `--steps` is the total target and
    # `--start-at` is what a previous run already did.
    var n_batched = (total_env_steps - start_at) // N_ENVS
    if n_batched <= 0:
        print("nothing to do: --start-at", start_at, ">= --steps", total_env_steps)
        finish_run(run, logger, artifacts, String("noop_start_at_ge_steps"))
        return
    var last_measure = 0.0
    var last_rate = 0.0
    var lie_frac = 0.0
    var t0 = perf_counter_ns()
    var lie_total = 0   # lanes given the lie-down transform, over diagnostic resets
    var lie_resets = 0  # diagnostic resets counted, the denominator for the above
    print("  batched steps", n_batched, " seed steps", SEED_STEPS, " updates/step", ups, " tracking", track_on)
    for s in range(n_batched):
        var env_steps = s * N_ENVS
        if s % T_EPISODE == 0:
            var sign = 1.0 if random_float64() < 0.5 else -1.0
            _rsi_reset(sign)
            _hist_reset()
            if smoke or s % (T_EPISODE * 20) == 0:
                # diagnostics: which rows the lanes got, and how many lie down
                ctx.enqueue_copy(h_row_got, row_got)
                ctx.enqueue_copy(h_u_rsi, u_rsi)
                ctx.synchronize()
                var lo = 1e30
                var hi = -1e30
                var lie_n = 0
                for l in range(N_ENVS):
                    var v = Float64(h_row_got[l])
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
                    if lie_down_selected(h_u_rsi[l * 3 + 2], Scalar[DT](lie_prob)):
                        lie_n += 1
                lie_total += lie_n
                lie_resets += 1
                lie_frac = Float64(lie_n) / Float64(N_ENVS)
                print(
                    "  [reset @", s, "] rows in [", lo, ",", hi, "] of", n_rows,
                    " lie-down", Float64(lie_n) / Float64(N_ENVS),
                    "(want", lie_prob, ", cumulative",
                    Float64(lie_total) / Float64(lie_resets * N_ENVS), ")",
                )
        if track_on and s % TRACK_LEN == 0:
            _draw_tracking()
        if track_on:
            _pin_step(s % TRACK_LEN)

        # read: pack the actor's view BEFORE acting
        ctx.enqueue_copy(prev_env_obs, env._obs)   # the STORED head, pre-step
        _pack_obs()
        ctx.enqueue_copy(prev_obs, full_obs)
        agent.select_action_batched[N_ENVS](
            LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin](full_obs),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin](env._action),
            LayoutTensor[DT, Layout.row_major(N_ENVS, 2 * ACT), MutAnyOrigin](ao),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT + 1), MutAnyOrigin](alp),
            env_steps,
        )
        # push(state_t, a_{t-1}) then last_action = a_t, before the env moves
        _hist_advance()
        env.step_batch[N_ENVS](Optional(ctx), UInt64(seed_v) + UInt64(s))
        # steps since the reset, for the DERIVED tail's back-step bound
        var age = s % T_EPISODE
        agent.set_age(age if age < G1_HIST_MAX_AGE else G1_HIST_MAX_AGE)
        # The reset runs at the START of a step, so it is THIS transition
        # whose successor row will hold a post-reset observation — the ring
        # derives `s'` from the next row and must be told to skip this one.
        agent.set_boundary((s + 1) % T_EPISODE == 0)
        # ⚠ `prev_obs` is the PACKED 928 row; `record_batch_gpu` stores
        # `STORE_OBS` = 527 columns per row and would MIS-STRIDE a packed one.
        # `prev_env_obs` is the same row's stored head, captured before the
        # step — identical to `prev_obs[0:527]` by construction, and correctly
        # strided.
        agent.record_batch_gpu[N_ENVS](ctx, prev_env_obs, env._action, reward0, env._obs, done0)

        if env_steps >= SEED_STEPS:
            if no_graph:
                # `--no-graph`: the 16 updates as plain launches, so
                # `MODULAR_DEBUG=device-sync-mode` can name a faulting kernel
                # (a graph replay reports its fault only at the next check)
                _captured_updates()
            else:
                maybe_capture_replay[_captured_updates](train_graph, ctx)
            for _ in range(ups):
                agent.note_train_update()

        if s % print_every == 0 or (smoke and s % 10 == 0):
            ctx.synchronize()
            var el = Float64(perf_counter_ns() - t0) * 1e-9
            var measure = 0.0
            var ortho = 0.0
            # the two halves of `measure`, HALVED into the reference's scale so
            # they read directly against its shipped 200 M-step
            # `train_log.txt` at the same timestep (§12.15)
            var fb_quad = 0.0
            var fb_anchor = 0.0
            var m_mean = 0.0
            # `scale_reg`'s weight, mean|Q_fb|. Against `loss/actor`
            # (= -mean Q_fb) the RATIO is the factor the CPR style term was
            # under-weighted by before §12.15 — 1.0 means it never mattered.
            var q_fb_abs = 0.0
            var actor = 0.0
            var f_norm = 0.0
            var b_norm = 0.0
            if env_steps >= SEED_STEPS:
                agent.base.peek_losses(
                measure, ortho, actor, f_norm, b_norm,
                fb_quad, fb_anchor, m_mean, q_fb_abs,
            )
            var rate = Float64(env_steps) / (el + 1e-9)
            last_measure = measure
            last_rate = rate
            print(
                "  step", s + s_off, " env", env_steps + start_at, " ", rate, "env st/s",
                " ring", agent.base.size, " updates", agent.total_train_steps(),
                " measure", measure, " ortho", ortho, " actor", actor, " |F|", f_norm, " |B|", b_norm,
            )
            # ⚠ THE STEP IS THE GLOBAL ONE so a resumed run's curve continues
            # the first one's instead of overwriting it from zero.
            # ⚠ `env/st_s` IS ALSO A LEARNING SIGNAL, not only a speed one:
            # collision cost tracks how much robot is touching the ground
            # (physics3d/PERFORMANCE.md §13.51), so a policy that learns to
            # stand makes the run go FASTER.
            var mn = List[String]()
            var mv = List[Float64]()
            mn.append(String("fb_measure_loss")); mv.append(measure)
            mn.append(String("fb_ortho_loss")); mv.append(ortho)
            mn.append(String("policy_loss")); mv.append(actor)
            mn.append(String("fb_offdiag")); mv.append(fb_quad)
            mn.append(String("fb_diag")); mv.append(fb_anchor)
            mn.append(String("fb_m1")); mv.append(m_mean)
            mn.append(String("fb_q_abs_mean")); mv.append(q_fb_abs)
            # ── the CPR half ──────────────────────────────────────────
            # Runs 1-3 turned at the same step under three materially
            # different FB losses, and NOTHING logged inflected at the turn.
            # The FB half was the only half being logged: the discriminator
            # and Q_D — the terms that hold pi on the expert manifold, i.e.
            # the terms tracking quality IS — were completely unobserved.
            # ⚠ `d_pos` / `d_neg` are BCE LOSSES (`bce_logits_const_t` with
            # targets 1.0 and 0.0), NOT discriminator scores. CHANCE IS
            # ln(2) = 0.693 ON BOTH; small is healthy, and the DIFFERENCE
            # between them means nothing. A first reading of them as scores
            # produced a confident "the discriminator is dead" that the
            # reference's own log refuted: at 4.6 M it runs 0.044 expert /
            # 0.110 policy where we run 0.074 / 0.086 — same regime, both far
            # below chance. Named for what they are so the misreading is
            # harder to repeat.
            var d_pos = 0.0
            var d_neg = 0.0
            var r_d = 0.0
            var q_d = 0.0
            var q_loss = 0.0
            var q_pi = 0.0
            agent.head.read_diag(d_pos, d_neg, r_d, q_d, q_loss, q_pi)
            mn.append(String("disc_expert_loss")); mv.append(d_pos)
            mn.append(String("disc_policy_loss")); mv.append(d_neg)
            mn.append(String("disc_reward_mean")); mv.append(r_d)
            mn.append(String("mean_q")); mv.append(q_d)
            mn.append(String("critic_loss")); mv.append(q_loss)
            mn.append(String("policy_q_mean")); mv.append(q_pi)
            mn.append(String("f_norm")); mv.append(f_norm)
            mn.append(String("b_norm")); mv.append(b_norm)
            # `|B|` is pinned to sqrt(d) by the net's sphere projection, so it
            # is structurally incapable of showing a DIRECTIONAL collapse —
            # which is what killed run 1 (§12.12). `ortho` can, once unpacked.
            #
            # The participation-ratio effective rank of `E[B B^T]`, derived
            # from `L_ortho`. d = 256 means isotropic; the reference holds
            # ~255.7 (§12.13). THIS is the number to watch, not `b_norm`.
            #
            # ⚠ The conversion lives in `loss.mojo` BESIDE the loss it depends
            # on. It used to be inline here and was missed when `L_ortho`
            # moved to the reference's scale (§12.28), logging 200.8 where the
            # truth was 244.3 — the shape of a B collapse, from a metric bug.
            mn.append(String("b_rank_eff"))
            mv.append(fb_rank_eff_from_ortho[D, BATCH](ortho))
            mn.append(String("steps_per_s")); mv.append(rate)
            mn.append(String("buffer_size")); mv.append(Float64(agent.base.size))
            mn.append(String("train_steps")); mv.append(Float64(agent.total_train_steps()))
            mn.append(String("lie_down_frac")); mv.append(lie_frac)
            mn.append(String("wall_s")); mv.append(el)
            logger.log_scalars(mn, mv, env_steps + start_at)
        var ee = eval_every if eval_every > 0 else ckpt_every
        var do_ckpt = ckpt_every > 0 and s > 0 and s % ckpt_every == 0
        var do_eval = eval_on and ee > 0 and s > 0 and s % ee == 0
        if do_ckpt or do_eval:
            # A kept checkpoint when one is due, otherwise ONE scratch file
            # overwritten in place — the eval scores a FILE either way, so the
            # finer cadence costs a 465 MB write and no disk growth. The
            # scratch path is never announced to the artifact sink.
            var p = run.checkpoint_path(
                String("step_") + String(s + s_off) if do_ckpt
                else String("_eval_scratch")
            )
            # ⚠ THE GUARD IS BEFORE THE WRITE, not after. A checkpoint is
            # 1.17 GB across three files and `save_state` has no way to
            # un-write a partial one; the last run's `step_38000.ckpt.cpr`
            # stopped mid-stream and the process died on it. Refusing the
            # write costs one interval and leaves a resumable run behind.
            var free_gb = _free_gb(run.dir)
            var room = free_gb < 0.0 or free_gb >= ckpt_min_free
            if not room:
                ckpt_skipped += 1
                print(
                    "  ⚠ SKIPPING checkpoint —", free_gb, "GB free on",
                    run.dir, "is under --ckpt-min-free-gb", ckpt_min_free,
                    " (skipped", ckpt_skipped, "so far; the run continues)",
                )
            if room:
                agent.save_state(p)
            if room and do_ckpt:
                announce_checkpoint(p, artifacts, run.dir)
                print("  checkpoint", p)
                ckpt_written.append(s + s_off)
            if do_eval and room:
                var sc = _score_tracking(
                    eval_t, eval_env, rsi, st, pv, eval_qpos, p, eval_segments,
                    eval_ach, eval_tgt, eval_b_in, eval_b_out, eval_z_seg,
                    eval_obs, eval_z1, eval_act, prio_emd,
                )
                print(
                    "  eval: emd", sc.emd, " distance", sc.distance,
                    " proximity", sc.proximity, " over", sc.n, "segments",
                )
                var en = List[String]()
                var ev = List[Float64]()
                en.append(String("eval_emd")); ev.append(sc.emd)
                en.append(String("eval_distance")); ev.append(sc.distance)
                en.append(String("eval_proximity")); ev.append(sc.proximity)
                en.append(String("eval_segments")); ev.append(Float64(sc.n))

                # ── motion prioritization refresh ─────────────────────
                if prio_on and env_steps + start_at >= next_prio:
                    next_prio += G1_PRIO_REFRESH
                    # 1. the RSI reset draw: which motion a lane starts in
                    g1_fill_motion_table(
                        prio_emd, PRIO_MOTION_LEN, prio_tbl
                    )
                    for i in range(len(prio_tbl)):
                        prio_motion.data[i] = Scalar[DT](prio_tbl[i])
                    prio_motion.upload_resident(ctx)
                    n_prio_motion = len(prio_tbl)
                    # 2. the tracking-z windows, same length, in place
                    g1_fill_window_table(
                        starts250, beg250, end250, prio_emd, cur250,
                        len(starts250), prio_tbl,
                    )
                    for i in range(len(prio_tbl)):
                        starts250_t.data[i] = Scalar[DT](prio_tbl[i])
                    starts250_t.upload_resident(ctx)
                    # 3. the expert windows the discriminator certifies —
                    #    SAME buffer, SAME length. `attach_expert_windows`
                    #    would hand the agent a new one and the captured
                    #    graph still points at the old.
                    g1_fill_window_table(
                        starts8, beg8, end8, prio_emd, cur8,
                        len(starts8), prio_tbl,
                    )
                    var h8 = ctx.enqueue_create_host_buffer[IDX_DT](
                        len(prio_tbl)
                    )
                    for i in range(len(prio_tbl)):
                        h8[i] = Scalar[IDX_DT](prio_tbl[i])
                    ctx.enqueue_copy(agent.starts_dev.value(), h8)
                    ctx.synchronize()
                    var lo = prio_emd[0]
                    var hi = prio_emd[0]
                    for i in range(len(prio_emd)):
                        if prio_emd[i] < lo:
                            lo = prio_emd[i]
                        if prio_emd[i] > hi:
                            hi = prio_emd[i]
                    print(
                        "  motion priorities refreshed: EMD", lo, "..", hi,
                        " weight spread",
                        g1_motion_priority(hi) / g1_motion_priority(lo),
                    )
                    en.append(String("prio_emd_min")); ev.append(lo)
                    en.append(String("prio_emd_max")); ev.append(hi)
                    en.append(String("prio_spread"))
                    ev.append(
                        g1_motion_priority(hi) / g1_motion_priority(lo)
                    )
                logger.log_scalars(en, ev, env_steps + start_at)

                # best-so-far, and ONLY when a real checkpoint backs it. The
                # finer eval cadence scores `_eval_scratch`, which is
                # overwritten every time — recording it as "best" would
                # protect a path whose bytes are already gone. §12.44 hit
                # exactly that: the run's best number, 1.1899 at step 33000,
                # has no checkpoint because that step was eval-only.
                if do_ckpt and sc.emd < ckpt_best_emd:
                    ckpt_best_emd = sc.emd
                    ckpt_best_step = s + s_off

            # ── retention, AFTER the eval so `best` is current ─────────
            if do_ckpt and room and ckpt_keep > 0:
                var dropped = 0
                for i in range(len(ckpt_written)):
                    var st_i = ckpt_written[i]
                    if not checkpoint_retain(
                        st_i, ckpt_written, ckpt_keep, ckpt_milestone,
                        ckpt_best_step,
                    ):
                        _drop_checkpoint(
                            run.checkpoint_path(String("step_") + String(st_i))
                        )
                        dropped += 1
                if dropped > 0:
                    var alive = List[Int]()
                    for i in range(len(ckpt_written)):
                        var st_i = ckpt_written[i]
                        if checkpoint_retain(
                            st_i, ckpt_written, ckpt_keep, ckpt_milestone,
                            ckpt_best_step,
                        ):
                            alive.append(st_i)
                    ckpt_written = alive^
                    print(
                        "  retention: dropped", dropped, "· keeping",
                        len(ckpt_written), "(last", ckpt_keep,
                        "+ best step", ckpt_best_step,
                        "+ every", ckpt_milestone, ") ·",
                        _free_gb(run.dir), "GB free",
                    )
            logger.flush()  # the CSV is the record; do not lose it to a crash
    ctx.synchronize()
    # ⚠ STOP THE CLOCK BEFORE THE CHECKPOINT. `el` used to be taken AFTER
    # `save_state`, so the headline rate measured a file write: on the 40-step
    # smoke that one write was ~17 s of a 49 s run and dragged 1030 env st/s
    # down to 830 (docs/BFM_ZERO_G1_REPRODUCTION.md §12.9).
    var el = Float64(perf_counter_ns() - t0) * 1e-9
    var final_ckpt = run.checkpoint_path(String("step_") + String(n_batched + s_off))
    agent.save_state(final_ckpt)
    announce_checkpoint(final_ckpt, artifacts, run.dir)
    print("done:", n_batched, "batched steps,", agent.total_train_steps(), "updates in", el, "s;", Float64(n_batched * N_ENVS) / el, "env st/s")
    print("final checkpoint:", final_ckpt)
    print("run record      :", run.kv_path())
    print("metrics         :", run.metrics_path())
    # ⚠ `status` AND `outcome` ARE WRITTEN, NEVER INFERRED (core/run.mojo). A
    # `run.kv` still saying `running` with an hour-old `started` IS a crashed
    # run — which is exactly what a 55-hour run needs a reader to be able to
    # tell without the terminal it was launched from.
    finish_run(
        run, logger, artifacts,
        String("env_steps=") + String(start_at + n_batched * N_ENVS)
        + " updates=" + String(agent.total_train_steps())
        + " measure=" + String(last_measure)
        + " env_st_s=" + String(last_rate),
    )
