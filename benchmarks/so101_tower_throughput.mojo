"""What the so101_tower scene costs per env step — physics and cameras — at RL lane counts.

    pixi run -e nvidia mojo build -I . benchmarks/so101_tower_throughput.mojo -o bench_tower
    pixi run -e nvidia ./bench_tower [--only physics|camera] [--warmup 50] [--steps 600] [--png DIR]

The question it answers: could pixel RL (SAC / PPO from a camera, the Squint
recipe: ~1024 lanes, one small wrist image) run on this scene at a useful
rate? Two halves, each at 32 (the sim eval's width), 256, 1024 and 4096 lanes.

## PHYSICS — `bench_half_cheetah_batch.mojo`'s protocol

WARMUP untimed steps, STEPS timed between two synchronisations, the action
upload and the selective reset inside the loop, EAGER and GRAPH (the physics
`step_batch` captured once and replayed — run it through `pixi run` so the
CUDA interceptor is preloaded), so the rows line up with half cheetah's.

⚠⚠ ONE STEP HERE IS A CONTROL STEP: FRAME_SKIP (16) PHYSICS STEPS. Set the
`physics_steps_per_s` column beside half cheetah's (5 per env step) and
MuJoCo Warp's. The series-3 sim eval's "physics 257 s" was 76 800 control
steps = 1.23M physics steps at 32 lanes, EAGER, with three downloads and a
synchronise per step — a correctness loop, not this number.

⚠⚠ THE ARM MOVES AND THE PROPS ARE PLACED, OR THIS MEASURES AN EMPTY DESK. A
constant action parks the arm in the air and the solve sees only the props
resting on the mat. Every lane draws uniform joint targets over the whole
`ctrlrange` (gripper included) and holds each HOLD steps — an exploring
policy's motion: the arm sweeps the desk, hits the brick and the bowl. The
task's `meta` words and region table go in before the reset (the eval's own
set-up), so the device reset places the props per lane as training would; the
row prints the brick's spread across lanes to prove it, and the contacts and
Newton iterations per solve over the timed window
(`META_IDX_SOLVER_ACC_NCON` / `_ITER`, differenced) — this scene's cost is
contacts, and a row near the props-only contact floor is a parked arm.

## CAMERAS — the rig's tracer, three ways

- `rig`    320x240, 4 samples, overhead + wrist: what the store and the sim
           eval render (`TowerRenderer`). Kernel only.
- `rl128`  128x128, 1 sample, wrist, then overhead: a pixel-RL observation
           (Squint renders 128x128 and downsamples to 16x16). Kernel only.
- `evalpath` (32 lanes only) the sim eval's whole `_render`: host FK of every
           lane, three uploads, and per camera the launch, the copy of `rgb`
           to the host, a synchronise and the uint8 pack. Its gap to `rig` is
           what the eval spends outside the tracer.

The lanes hold DIFFERENT poses (the props placed per lane, then 40 steps of
the random policy above), not one pose copied — `camera_tracer_lane_sweep.mojo`
explains why identical lanes are a best case. Each camera row prints the
fraction of pixels that hit geometry; `--png DIR` writes lane 0 of every leg.

## RESULTS — RTX 5090, 2026-09-26 (ffbc16d94)

    envs   control steps/s   physics steps/s   ms / control step   graph
      32         604               9.7k               53            = eager
     256       3 075              49k                83            = eager
    1024       9 752             156k               105            = eager
    4096      21 647             346k               189            = eager

~9 contacts and 1.9 Newton iterations per solve at every width: the solve is
light, and the CUDA graph buys nothing (357 nodes per control step) — this is
not launch-bound. Per physics step it is 16x (32 envs) to 22-25x (1024-4096)
half cheetah's; whether that is the scene or the engine needs MuJoCo Warp on
THIS scene beside it.

    frames/s        32      256     1024    4096
    wrist 128²    13.3k   16.9k   17.4k   17.0k
    overhead 128²  4.9k    7.2k    7.2k    7.4k
    rig pair        172     199     198      -     (320x240x4, both cameras)

The tracer saturates by 256 lanes. The arm's visual meshes are ~343k
triangles over 14 distinct STLs (dm_control's Jaco: 8k), and per pixel this
is ~3x (wrist) to ~7.5x (overhead) slower than `camera_tracer_lane_sweep`'s
lift_brick without shadows. The eval's render at 32 lanes: 289 ms per step,
186 ms in the tracer, 103 ms on the host (FK, copies, uint8 pack) — its log
said 335 ms, against 107 ms of physics: the eval was render-bound.

Pixel RL at 1024 lanes with the 128² wrist camera: 105 + 59 = 164 ms per
step, ~6.2k control steps/s before the learner; 4096 lanes ~9.5k/s.

The MESH SHARE (`rl128-nomesh`, M1 Pro, not yet on the 5090): 0.68-0.70
(wrist) and 0.77-0.79 (overhead) at 32 and 256 lanes, with 35 visual geoms, 18
meshes, 319 922 triangles — while the jaws cover ~6 % of the wrist image. The
levers are read in `noeira-docs/SO101_RENDER_SPEED.md`.

**After 07d5a165a + 65ecd9a5d (5090, same binary, new vs old):** 128² wrist
38.5k fps at 1024 lanes (2.28x), overhead 17.7k (2.16x), the rig pair 502/s
(2.27x); the screen cull alone +11 % wrist, -9 % overhead; mesh share still
0.73-0.83; 64x64x1 wrist 144k fps. Physics unchanged (153k / 354k physics
steps/s at 1024 / 4096).

**MuJoCo Warp on the same scene** (`so101_tower_throughput_mjwarp.py`, same
5090, MuJoCo 3.14 / Warp 1.17, same protocol, CUDA graph per control step;
9-10 contacts and 1.7 solver iterations per world against our 9 and 1.9):

    physics steps/s     32      256     1024    4096
    noeira             8.9k    50.6k   153k    354k
    MuJoCo Warp       28.5k   136.5k   428k    1.21M
    ratio              0.31    0.37    0.36    0.29

    frames/s (1024)   wrist 128²  overhead 128²  wrist 64²  overhead 64²
    noeira              38.5k       17.7k         144k        59k
    MuJoCo Warp         76.2k       38.2k         231k       105k
    ratio               0.51        0.46          0.62       0.56

(MuJoCo Warp rendered groups 0/2/4/5 — the calibrated look's set — and hit
0.64-0.68 of the wrist pixels against our 0.66-0.69.) nsys on our physics at
4096 lanes: the Newton solver kernels ~58 % of GPU time, the collision
kernels ~38 %, the rest ~3 %.

**After the 26-27 Sep Newton + collision commits (a684443d2, 5090, 2026-09-28,
same box for all three rows; the old row is abf28444a rebuilt there):**

    physics steps/s     32      256     1024    4096
    noeira abf28444a   8.9k    50.7k   154k    354k
    noeira a684443d2  39.0k    211k    618k    1.32M    (3.7-4.4x)
    MuJoCo Warp       27.3k    136k    431k    1.21M
    ratio              1.42    1.55    1.43    1.09     (was 0.29-0.37)

Same load: 9.2-9.3 contacts and 1.9 Newton iterations per solve (MuJoCo Warp
9.8-10.8 and 1.7). Cameras unchanged (no renderer commit): 0.47-0.62x.

⚠ THE POSE IS HOST FK OF THE ENV'S `qpos`, AS IN THE EVAL
(`so101_tower_rig.mojo`'s header): the env leaves `SYNC_FK_AFTER_STEP` off, so
its device `xpos` is one substep stale. A pixel-RL loop would need a device FK
after the step (`forward_kinematics["gpu", ...]`) — not timed here.
"""

from std.os import makedirs
from std.random import random_float64, seed as seed_rng
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from noeira.cuda import CUDAGraph, maybe_capture_replay
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.io.png import save_png
from noeira.nn.core.ptr import mptr
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, MODEL_CURRICULUM_SIZE,
    META_IDX_SOLVER_ACC_NCON, META_IDX_SOLVER_ACC_ITER,
    MODEL_MESH_META_SIZE, MESH_META_IDX_TRINUM,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.flat_model import FlatModelDef
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.posed_reset import task_meta_words
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, RIG_CAM_W, RIG_CAM_H, RIG_N_CAMS, RIG_NPIX,
    RIG_CAM_ELEMS, RIG_IMG_ELEMS, RIG_SAMPLES, TowerRendererSized,
    make_tower_model, make_tower_renderer, tower_cameras, pack_camera_u8,
    rig_byte,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family


comptime DT = RIG_DT
comptime CFG = So101TowerConfig
comptime FRAME_SKIP = CFG.FRAME_SKIP
comptime NQ = So101TowerModel.NQ
comptime NB = So101TowerModel.NBODY
comptime TASK = "so101_tower_cube_in_bowl"
comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime HOLD = 16
"""Control steps a random target is held (~0.5 s at 31.25 Hz)."""
comptime N_TABLES = 8
"""Pre-drawn action tables, cycled — no host RNG inside the timed loop."""
comptime POSE_STEPS = 40
"""Random-policy steps before the camera legs, so the lanes' arms differ."""
comptime RL_W = 128
comptime RL_H = 128

# ⚠ SET FALSE ON A SMALL BOARD: 4 096 lanes allocate the tower's whole `Data`
# (contact buffers included) 4 096 times, and the camera half a second one.
comptime LANES_4096: Bool = True

# Every timed camera leg gets the same wall-clock window, not the same rep
# count — `camera_tracer_lane_sweep.mojo`'s `_reps_for`, for its reason.
comptime MIN_WINDOW_MS = 300.0
comptime MIN_REPS = 3
comptime MAX_REPS = 2000


struct TaskWords(Copyable, Movable):
    """The region table and the task's `meta` words — the eval's set-up."""

    var curriculum: List[Float64]
    var meta_idx: List[Int]
    var meta_val: List[Float64]
    var brick: Int

    def __init__(out self, fmd: FlatModelDef) raises:
        var f = load_family(String(FAMILY_PATH))
        var rsites = region_sites(f, fmd.site_names)
        var rects = region_rects(f)
        var rheights = region_half_heights(f)
        var cw = region_table_words(
            rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
            rheights[0],
        )
        self.curriculum = List[Float64]()
        for k in range(MODEL_CURRICULUM_SIZE):
            self.curriculum.append(Float64(cw[k]))
        var mw = task_meta_words(
            String(TASK), String(FAMILY), CFG.SHAPE_W_GOAL,
            CFG.SHAPE_W_REACH, CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
        )
        self.meta_idx = mw[0].copy()
        self.meta_val = List[Float64]()
        for k in range(len(mw[1])):
            self.meta_val.append(Float64(mw[1][k]))
        self.brick = -1
        for b in range(len(fmd.body_names)):
            if String(fmd.body_names[b]) == "brick_brick":
                self.brick = b
        if self.brick < 0:
            raise Error("brick_brick not found in the composed scene")


struct Opts(Copyable, Movable):
    var physics: Bool
    var camera: Bool
    var warmup: Int
    var steps: Int
    var png: String

    def __init__(out self) raises:
        self.physics = True
        self.camera = True
        self.warmup = 50
        self.steps = 600
        self.png = String("")
        var a = argv()
        var i = 1
        while i < len(a):
            var k = String(a[i])
            if i + 1 >= len(a):
                raise Error("missing value after " + k)
            var v = String(a[i + 1])
            if k == "--only":
                self.physics = v == "physics"
                self.camera = v == "camera"
                if not (self.physics or self.camera):
                    raise Error("--only physics|camera")
            elif k == "--warmup":
                self.warmup = Int(v)
            elif k == "--steps":
                self.steps = Int(v)
            elif k == "--png":
                self.png = v
            else:
                raise Error("unknown flag " + k)
            i += 2


comptime TowerEnv[N: Int] = Phyics3dBatchedEnv[So101TowerModel, CFG, N]


def _install_task_and_reset[
    N: Int
](ctx: DeviceContext, mut env: TowerEnv[N], tw: TaskWords) raises:
    """The task's words, then the device reset — which places the props."""
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DT](tw.curriculum[k])
    env.mf.curriculum.upload(ctx)
    for e in range(N):
        var mb = e * METADATA_SIZE
        for k in range(METADATA_SIZE):
            env.d.meta.data[mb + k] = Scalar[DT](0)
        for k in range(len(tw.meta_idx)):
            env.d.meta.data[mb + tw.meta_idx[k]] = Scalar[DT](tw.meta_val[k])
    env.d.meta.upload(ctx)
    ctx.synchronize()
    env.reset_batch[N](ctx=ctx, rng_seed=UInt64(7))
    ctx.synchronize()


def _brick_sd_mm[
    N: Int
](ctx: DeviceContext, mut env: TowerEnv[N], brick: Int) raises -> String:
    """The brick's xy standard deviation across lanes, in mm. 0,0 = every lane
    got the same pose = the reset did not place."""
    env.d.xpos.download(ctx)
    ctx.synchronize()
    var s = List[Float64](length=4, fill=0.0)
    for e in range(N):
        var x = Float64(env.d.xpos.data[e * NB * 3 + brick * 3])
        var y = Float64(env.d.xpos.data[e * NB * 3 + brick * 3 + 1])
        s[0] += x
        s[1] += x * x
        s[2] += y
        s[3] += y * y
    var nf = Float64(N)
    var vx = s[1] / nf - (s[0] / nf) * (s[0] / nf)
    var vy = s[3] / nf - (s[2] / nf) * (s[2] / nf)
    var sx = vx ** 0.5 if vx > 0.0 else 0.0
    var sy = vy ** 0.5 if vy > 0.0 else 0.0
    return String(Int(sx * 1000.0)) + "," + String(Int(sy * 1000.0))


def _action_tables[
    N: Int
](ctx: DeviceContext) raises -> List[HostBuffer[DT]]:
    comptime ACT = TowerEnv[N].ACT_DIM
    seed_rng(11)
    var tables = List[HostBuffer[DT]]()
    for _t in range(N_TABLES):
        var hb = ctx.enqueue_create_host_buffer[DT](N * ACT)
        tables.append(hb^)
    ctx.synchronize()
    for t in range(N_TABLES):
        var p = tables[t].unsafe_ptr()
        for k in range(N * ACT):
            p[unsafe_offset=k] = Scalar[DT](random_float64(-1.0, 1.0))
    return tables^


def _solver_sums[
    N: Int
](ctx: DeviceContext, mut env: TowerEnv[N]) raises -> Tuple[Float64, Float64]:
    """Summed over lanes: contacts handed to the solve, Newton iterations."""
    env.d.meta.download(ctx)
    ctx.synchronize()
    var ncon = 0.0
    var iters = 0.0
    for e in range(N):
        ncon += Float64(env.d.meta.data[e * METADATA_SIZE + META_IDX_SOLVER_ACC_NCON])
        iters += Float64(env.d.meta.data[e * METADATA_SIZE + META_IDX_SOLVER_ACC_ITER])
    return (ncon, iters)


# ═══ PHYSICS ═════════════════════════════════════════════════════════════


def bench_physics[
    N: Int, USE_GRAPH: Bool
](ctx: DeviceContext, tw: TaskWords, warmup: Int, steps: Int) raises:
    comptime ACT = TowerEnv[N].ACT_DIM
    var env = TowerEnv[N](ctx)
    _install_task_and_reset[N](ctx, env, tw)
    var sd = _brick_sd_mm[N](ctx, env, tw.brick)
    var tables = _action_tables[N](ctx)
    var act_dev = DeviceBuffer[DT](ctx, env.action_ptr(), N * ACT, owning=False)

    var graph: Optional[CUDAGraph] = None

    @always_inline
    @parameter
    def physics() raises capturing:
        env.step_batch[N](ctx=ctx, rng_seed=UInt64(1))

    @always_inline
    def one_step(it: Int) raises capturing:
        ctx.enqueue_copy(act_dev, tables[(it // HOLD) % N_TABLES])
        comptime if USE_GRAPH:
            maybe_capture_replay[physics](graph, ctx)
        else:
            physics()
        env.selective_reset_batch[N](ctx=ctx, rng_seed=UInt64(it + 1) * 7)

    for it in range(warmup):
        one_step(it)
    ctx.synchronize()
    var s0 = _solver_sums[N](ctx, env)
    var t0 = perf_counter_ns()
    for it in range(steps):
        one_step(warmup + it)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    var s1 = _solver_sums[N](ctx, env)
    var env_sps = Float64(N * steps) / dt
    var solves = Float64(N * steps * FRAME_SKIP)

    var mode = String("graph") if USE_GRAPH else String("eager")
    if USE_GRAPH and graph and graph.value().is_disabled():
        mode = String("graph-DISABLED(ran eager)")
    print(
        "RESULT side=noeira-gpu leg=physics mode=" + mode,
        "model=so101_tower_cube_in_bowl n_envs=" + String(N),
        "steps=" + String(steps),
        "wall_s=" + String(dt),
        "us_per_batch_step=" + String(dt / Float64(steps) * 1e6),
        "env_steps_per_s=" + String(Int(env_sps)),
        "physics_steps_per_s=" + String(Int(env_sps * Float64(FRAME_SKIP))),
        "contacts_per_solve=" + String((s1[0] - s0[0]) / solves),
        "newton_iters_per_solve=" + String((s1[1] - s0[1]) / solves),
        "brick_sd_mm=" + sd,
    )


# ═══ CAMERAS ═════════════════════════════════════════════════════════════


def _reps_for(probe_ms: Float64) -> Int:
    if probe_ms <= 0.0:
        return MAX_REPS
    var n = Int(MIN_WINDOW_MS / probe_ms) + 1
    if n < MIN_REPS:
        return MIN_REPS
    if n > MAX_REPS:
        return MAX_REPS
    return n


def _posed_rig_data[
    N: Int
](
    ctx: DeviceContext, tw: TaskWords, mut rm: Model[RIG_DT, TOWER_MD],
    mut rd: Data[RIG_DT, TOWER_MD, N],
) raises -> String:
    """Place the props per lane, run the random policy POSE_STEPS steps, and
    write host FK of every lane's `qpos` into `rd` (the eval's pose path).
    The env is local: it is gone before the camera legs allocate."""
    comptime ACT = TowerEnv[N].ACT_DIM
    var env = TowerEnv[N](ctx)
    _install_task_and_reset[N](ctx, env, tw)
    var tables = _action_tables[N](ctx)
    var act_dev = DeviceBuffer[DT](ctx, env.action_ptr(), N * ACT, owning=False)
    for it in range(POSE_STEPS):
        ctx.enqueue_copy(act_dev, tables[(it // HOLD) % N_TABLES])
        env.step_batch[N](ctx=ctx, rng_seed=UInt64(1))
    ctx.synchronize()
    var sd = _brick_sd_mm[N](ctx, env, tw.brick)
    env.d.qpos.download(ctx)
    ctx.synchronize()
    for e in range(N):
        for k in range(NQ):
            rd.qpos.data[e * NQ + k] = env.d.qpos.data[e * NQ + k]
    forward_kinematics["cpu", RIG_DT, TOWER_MD, N](rd, rm)
    rd.qpos.upload_resident(ctx)
    rd.xpos.upload_resident(ctx)
    rd.xquat.upload_resident(ctx)
    ctx.synchronize()
    return sd


def _time_cam[
    N: Int, W: Int, H: Int, S: Int, R: Bool = False
](
    ctx: DeviceContext, mut r: TowerRendererSized[N, W, H, S, R],
    mut rd: Data[RIG_DT, TOWER_MD, N], mut rm: Model[RIG_DT, TOWER_MD],
    cam: Int,
) raises -> Float64:
    """ms per launch (every lane, one camera), over a MIN_WINDOW_MS window."""
    r.render(ctx, rd, rm, cam)
    ctx.synchronize()
    var t = perf_counter_ns()
    r.render(ctx, rd, rm, cam)
    ctx.synchronize()
    var n = _reps_for(Float64(perf_counter_ns() - t) / 1.0e6)
    t = perf_counter_ns()
    for _i in range(n):
        r.render(ctx, rd, rm, cam)
    ctx.synchronize()
    return Float64(perf_counter_ns() - t) / Float64(n) / 1.0e6


def _hits_and_png[
    N: Int, W: Int, H: Int, S: Int, R: Bool = False
](
    ctx: DeviceContext, mut r: TowerRendererSized[N, W, H, S, R],
    mut rd: Data[RIG_DT, TOWER_MD, N], mut rm: Model[RIG_DT, TOWER_MD],
    cam: Int, png: String, name: String,
) raises -> Float64:
    """Render once; the fraction of pixels (all lanes) that hit geometry, and
    lane 0 as a PNG when `png` is set. A camera looking at nothing renders
    fast and measures nothing — the fraction is printed beside the time."""
    comptime NPIX = W * H
    r.render(ctx, rd, rm, cam)
    var seg = ctx.enqueue_create_host_buffer[RIG_DT](N * NPIX)
    ctx.enqueue_copy(seg, r.seg)
    var rgb = ctx.enqueue_create_host_buffer[RIG_DT](NPIX * 3)
    ctx.enqueue_copy(rgb, r.rgb.create_sub_buffer[RIG_DT](0, NPIX * 3))
    ctx.synchronize()
    var hit = 0
    var sp = seg.unsafe_ptr()
    for i in range(N * NPIX):
        if Int(sp[unsafe_offset=i]) >= 0:
            hit += 1
    if png.byte_length() > 0:
        makedirs(png, exist_ok=True)
        var hwc = List[UInt8](length=NPIX * 3, fill=UInt8(0))
        var p = rgb.unsafe_ptr()
        for q in range(NPIX * 3):
            hwc[q] = rig_byte(Float64(p[unsafe_offset=q]))
        save_png(png + "/" + name + "_lanes" + String(N) + ".png", hwc, W, H, 3)
    return Float64(hit) / Float64(N * NPIX)


def _eval_render[
    N: Int
](
    ctx: DeviceContext,
    mut rr: TowerRendererSized[N, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES],
    mut rd: Data[RIG_DT, TOWER_MD, N], mut rm: Model[RIG_DT, TOWER_MD],
    cams: List[Int], host_q: List[Scalar[RIG_DT]],
    mut h_rgb: HostBuffer[RIG_DT], mut img_u8: List[Scalar[DType.uint8]],
) raises:
    """`so101_tower_act_eval.TowerActEval._render`, step for step: host FK of
    every lane, three uploads, and per camera the launch, the copy to the
    host, a synchronise and the uint8 pack."""
    for k in range(N * NQ):
        rd.qpos.data[k] = host_q[k]
    forward_kinematics["cpu", RIG_DT, TOWER_MD, N](rd, rm)
    rd.qpos.upload_resident(ctx)
    rd.xpos.upload_resident(ctx)
    rd.xquat.upload_resident(ctx)
    for slot in range(RIG_N_CAMS):
        rr.render(ctx, rd, rm, cams[slot])
        ctx.enqueue_copy(h_rgb, rr.rgb)
        ctx.synchronize()
        for e in range(N):
            _ = pack_camera_u8(
                mptr(h_rgb.unsafe_ptr()), e, mptr(img_u8),
                e * RIG_IMG_ELEMS + slot * RIG_CAM_ELEMS,
            )


def _r2(x: Float64) -> String:
    return String(Float64(Int(x * 100.0 + 0.5)) / 100.0)


def bench_camera[
    N: Int
](ctx: DeviceContext, fmd: FlatModelDef, tw: TaskWords, png: String) raises:
    var cams = tower_cameras(fmd)
    var cam_over = cams[0]
    var cam_wrist = cams[1]
    var rm = make_tower_model(ctx)
    var rd = Data[RIG_DT, TOWER_MD, N]()
    rd.upload_all(ctx)
    var sd = _posed_rig_data[N](ctx, tw, rm, rd)

    # ── rl128: the pixel-RL observation ──────────────────────────────────
    var hit_w = 0.0
    var hit_o = 0.0
    var ms_w = 0.0
    var ms_o = 0.0
    var r128 = make_tower_renderer[N, RL_W, RL_H, 1](ctx, fmd, rm)
    ms_w = _time_cam[N, RL_W, RL_H, 1](ctx, r128, rd, rm, cam_wrist)
    ms_o = _time_cam[N, RL_W, RL_H, 1](ctx, r128, rd, rm, cam_over)
    hit_w = _hits_and_png[N, RL_W, RL_H, 1](ctx, r128, rd, rm, cam_wrist, png, "rl128_wrist")
    hit_o = _hits_and_png[N, RL_W, RL_H, 1](ctx, r128, rd, rm, cam_over, png, "rl128_overhead")
    print(
        "RESULT leg=camera cfg=rl128 n_envs=" + String(N),
        "res=" + String(RL_W) + "x" + String(RL_H), "samples=1",
        "wrist_ms=" + _r2(ms_w), "wrist_fps=" + String(Int(Float64(N) / (ms_w / 1000.0))),
        "wrist_hit=" + _r2(hit_w),
        "overhead_ms=" + _r2(ms_o), "overhead_fps=" + String(Int(Float64(N) / (ms_o / 1000.0))),
        "overhead_hit=" + _r2(hit_o),
        "brick_sd_mm=" + sd,
        "vis_geoms=" + String(r128.vis.ngeom) + "+" + String(r128.vis.ncond),
    )

    # ── rl128, the OLD configuration: REFLECT compiled in, median trees ──
    # ⚠ The best-hit cut and the tie rule are in both legs (not switchable):
    # compare against the ffbc16d94 rows for their share.
    # `tests/tasks/test_so101_tower_render_variants.mojo` holds the two
    # configurations to the same bytes.
    var r128o = make_tower_renderer[N, RL_W, RL_H, 1, True](ctx, fmd, rm, bvh_sah=False)
    r128o.cull_enabled = False
    var ow = _time_cam[N, RL_W, RL_H, 1, True](ctx, r128o, rd, rm, cam_wrist)
    var oo = _time_cam[N, RL_W, RL_H, 1, True](ctx, r128o, rd, rm, cam_over)
    # The screen-rectangle cull alone: the new renderer with it switched off.
    r128.cull_enabled = False
    var cw = _time_cam[N, RL_W, RL_H, 1](ctx, r128, rd, rm, cam_wrist)
    var co = _time_cam[N, RL_W, RL_H, 1](ctx, r128, rd, rm, cam_over)
    r128.cull_enabled = True
    print(
        "RESULT leg=camera cfg=rl128-nocull n_envs=" + String(N),
        "wrist_ms=" + _r2(cw), "overhead_ms=" + _r2(co),
        "cull_speedup_wrist=" + _r2(cw / ms_w),
        "cull_speedup_overhead=" + _r2(co / ms_o),
    )
    print(
        "RESULT leg=camera cfg=rl128-old(reflect+median+nocull) n_envs=" + String(N),
        "wrist_ms=" + _r2(ow), "overhead_ms=" + _r2(oo),
        "speedup_wrist=" + _r2(ow / ms_w), "speedup_overhead=" + _r2(oo / ms_o),
    )

    # ── cheaper RL pixels: fewer rays for the same anti-aliased 16x16 ────
    var r64 = make_tower_renderer[N, 64, 64, 1](ctx, fmd, rm)
    var w64 = _time_cam[N, 64, 64, 1](ctx, r64, rd, rm, cam_wrist)
    var o64 = _time_cam[N, 64, 64, 1](ctx, r64, rd, rm, cam_over)
    _ = _hits_and_png[N, 64, 64, 1](ctx, r64, rd, rm, cam_wrist, png, "rl64_wrist")
    var r32 = make_tower_renderer[N, 32, 32, 4](ctx, fmd, rm)
    var w32 = _time_cam[N, 32, 32, 4](ctx, r32, rd, rm, cam_wrist)
    var o32 = _time_cam[N, 32, 32, 4](ctx, r32, rd, rm, cam_over)
    _ = _hits_and_png[N, 32, 32, 4](ctx, r32, rd, rm, cam_wrist, png, "rl32x4_wrist")
    print(
        "RESULT leg=camera cfg=rl-lowres n_envs=" + String(N),
        "64x64x1_wrist_fps=" + String(Int(Float64(N) / (w64 / 1000.0))),
        "64x64x1_overhead_fps=" + String(Int(Float64(N) / (o64 / 1000.0))),
        "32x32x4_wrist_fps=" + String(Int(Float64(N) / (w32 / 1000.0))),
        "32x32x4_overhead_fps=" + String(Int(Float64(N) / (o32 / 1000.0))),
    )

    # ── rl128 with every mesh's triangles removed: the mesh share ────────
    # `TRINUM = 0` makes the mesh test return NO HIT without touching a geom,
    # a pose or the kernel (`camera_tracer_lane_sweep.mojo`'s control), so
    # the time left is the per-ray geom loop, the primitives and the shading.
    # ⚠⚠ THE RENDERER'S OWN TABLE, `vis.mesh_meta`, NOT `rm.mesh_meta`: the
    # kernel reads the VisualModel's copy (built from the parse by
    # `set_visual`), and zeroing the Model's changed nothing — the first run of
    # this leg printed a mesh share of 0.00 and was a no-op.
    # ⚠ The arm is INVISIBLE in this leg; its pictures are not written.
    var nmesh = r128.vis.nmesh
    var tri_backup = List[Scalar[RIG_DT]]()
    for mi in range(nmesh):
        tri_backup.append(r128.vis.mesh_meta.data[mi * MODEL_MESH_META_SIZE + MESH_META_IDX_TRINUM])
        r128.vis.mesh_meta.data[mi * MODEL_MESH_META_SIZE + MESH_META_IDX_TRINUM] = Scalar[RIG_DT](0)
    r128.vis.mesh_meta.upload(ctx)
    ctx.synchronize()
    var nm_w = _time_cam[N, RL_W, RL_H, 1](ctx, r128, rd, rm, cam_wrist)
    var nm_o = _time_cam[N, RL_W, RL_H, 1](ctx, r128, rd, rm, cam_over)
    var nm_hit = _hits_and_png[N, RL_W, RL_H, 1](ctx, r128, rd, rm, cam_wrist, String(""), String(""))
    for mi in range(nmesh):
        r128.vis.mesh_meta.data[mi * MODEL_MESH_META_SIZE + MESH_META_IDX_TRINUM] = tri_backup[mi]
    r128.vis.mesh_meta.upload(ctx)
    ctx.synchronize()
    print(
        "RESULT leg=camera cfg=rl128-nomesh n_envs=" + String(N),
        "wrist_ms=" + _r2(nm_w), "wrist_mesh_share=" + _r2(1.0 - nm_w / ms_w),
        "overhead_ms=" + _r2(nm_o), "overhead_mesh_share=" + _r2(1.0 - nm_o / ms_o),
        "wrist_hit=" + _r2(nm_hit) + "(arm gone)",
        "meshes=" + String(nmesh), "tris=" + String(r128.vis.ntri),
    )

    # ── rig: the store's / the eval's pixels (skipped at 4096: ~6 GB) ────
    comptime if N <= 1024:
        var rr = make_tower_renderer[N](ctx, fmd, rm)
        var mo = _time_cam[N, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES](ctx, rr, rd, rm, cam_over)
        var mw = _time_cam[N, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES](ctx, rr, rd, rm, cam_wrist)
        var hw = _hits_and_png[N, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES](ctx, rr, rd, rm, cam_wrist, png, "rig_wrist")
        _ = _hits_and_png[N, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES](ctx, rr, rd, rm, cam_over, png, "rig_overhead")
        var rro = make_tower_renderer[N, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES, True](
            ctx, fmd, rm, bvh_sah=False
        )
        rro.cull_enabled = False
        var omo = _time_cam[N, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES, True](ctx, rro, rd, rm, cam_over)
        var omw = _time_cam[N, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES, True](ctx, rro, rd, rm, cam_wrist)
        print(
            "RESULT leg=camera cfg=rig-old(reflect+median+nocull) n_envs=" + String(N),
            "overhead_ms=" + _r2(omo), "wrist_ms=" + _r2(omw),
            "speedup_pair=" + _r2((omo + omw) / (mo + mw)),
        )
        print(
            "RESULT leg=camera cfg=rig n_envs=" + String(N),
            "res=" + String(RIG_CAM_W) + "x" + String(RIG_CAM_H),
            "samples=" + String(RIG_SAMPLES),
            "overhead_ms=" + _r2(mo), "wrist_ms=" + _r2(mw),
            "pair_fps=" + String(Int(Float64(N) / ((mo + mw) / 1000.0))),
            "wrist_hit=" + _r2(hw),
        )

        # ── evalpath: the eval's whole `_render`, at the eval's width ────
        comptime if N == 32:
            var h_rgb = ctx.enqueue_create_host_buffer[RIG_DT](N * RIG_NPIX * 3)
            var img_u8 = List[Scalar[DType.uint8]](length=N * RIG_IMG_ELEMS, fill=0)
            var host_q = List[Scalar[RIG_DT]](length=N * NQ, fill=0)
            for k in range(N * NQ):
                host_q[k] = rd.qpos.data[k]
            ctx.synchronize()

            _eval_render[N](ctx, rr, rd, rm, cams, host_q, h_rgb, img_u8)
            var t = perf_counter_ns()
            _eval_render[N](ctx, rr, rd, rm, cams, host_q, h_rgb, img_u8)
            var n = _reps_for(Float64(perf_counter_ns() - t) / 1.0e6)
            t = perf_counter_ns()
            for _i in range(n):
                _eval_render[N](ctx, rr, rd, rm, cams, host_q, h_rgb, img_u8)
            var ms = Float64(perf_counter_ns() - t) / Float64(n) / 1.0e6
            print(
                "RESULT leg=camera cfg=evalpath n_envs=" + String(N),
                "ms_per_step=" + _r2(ms),
                "kernels_ms=" + _r2(mo + mw),
                "outside_tracer_ms=" + _r2(ms - mo - mw),
                "(eval log: 803 s / 2400 steps = 334.6 ms)",
            )


def main() raises:
    var o = Opts()
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var tw = TaskWords(fmd)
    var ctx = DeviceContext()
    print("device:", ctx.name(), "| warmup", o.warmup, "| steps", o.steps,
          "| frame skip", FRAME_SKIP, "| hold", HOLD)
    if o.physics:
        bench_physics[32, False](ctx, tw, o.warmup, o.steps)
        bench_physics[32, True](ctx, tw, o.warmup, o.steps)
        bench_physics[256, False](ctx, tw, o.warmup, o.steps)
        bench_physics[256, True](ctx, tw, o.warmup, o.steps)
        bench_physics[1024, False](ctx, tw, o.warmup, o.steps)
        bench_physics[1024, True](ctx, tw, o.warmup, o.steps)
        comptime if LANES_4096:
            bench_physics[4096, False](ctx, tw, o.warmup, o.steps)
            bench_physics[4096, True](ctx, tw, o.warmup, o.steps)
    if o.camera:
        bench_camera[32](ctx, fmd, tw, o.png)
        bench_camera[256](ctx, fmd, tw, o.png)
        bench_camera[1024](ctx, fmd, tw, o.png)
        comptime if LANES_4096:
            bench_camera[4096](ctx, fmd, tw, o.png)
