"""A PIXEL student for `so101_tower`, taught by a STATE PPO teacher (DAgger).

    pixi run -e nvidia mojo build -I . examples/tasks/dagger_tower_pixels.mojo -o dagger_px
    ./dagger_px so101_tower_lift_real_layout --teacher RUN_DIR --steps 20000000

Step 1 of `noeira-docs/SO101_PIXEL_RL_PLAN.md`, the route agreed on 27 Sep:
the PPO policies that solve the task FROM STATE (lift 82.7 %, cube in bowl
62.2 % greedy) label the states a camera policy visits, and the camera policy
regresses onto the labels (Ross, Gordon & Bagnell 2011). Much cheaper than
pixel RL from scratch, and the student is corrected where IT drifts, not only
on the teacher's own trajectories.

## The loop, all lanes in lockstep (`N_ENVS`, the PPO driver's define)

Every control step:

1. the pose is made current on the device — `qpos` copied into the rig's own
   `Data` and `forward_kinematics["gpu"]` run on it (the env leaves
   `SYNC_FK_AFTER_STEP` off: its `xpos` is one substep stale, and an image of
   the wrong state is an off-by-one in the MDP nobody would see);
2. each camera is traced at `RENDER` x `RENDER` (1 sample) and box-averaged
   on the device to `OBS_PX` x `OBS_PX` ("resolution squinting", Squint) —
   `_pack_camera_kernel` writes it straight into this step's rows of the
   device REPLAY; `_pack_proprio_kernel` adds the six actuated joints;
3. the TEACHER acts greedily on the normalised STATE observation (its own
   `obs_norm.txt`); its action is this step's LABEL for every lane;
4. the STUDENT acts on the rows just written; each lane executes the
   student's action or the teacher's, fixed per EPISODE (teacher with
   probability `beta`, which decays from 1 to 0 over `--beta-steps`);
5. the env steps, finished lanes reset.

Every `ITER_STEPS` steps, `--updates` Adam steps of MSE on minibatches drawn
uniformly from the whole replay (DAgger's aggregate dataset, a ring of
`--replay` rows).

## The student's input — one tensor, no combinator

`C_IN = 3 * N_CAMS + ACT_DIM` planes of `OBS_PX` x `OBS_PX`, NCHW: each camera's
RGB minus 0.5, then each joint angle (x 0.5) BROADCAST over a whole plane.
The planes let a plain `Conv2D` stack take the proprioception without a
split/concat module; the first conv reads them like a bias per location.

⚠ THE LABEL IS THE TEACHER'S CLAMPED MEAN in the delta action space
(`_delta_to_env` of the PPO driver turns either policy's output into targets),
so the student's output is executed exactly as the teacher's would be.

⚠ `N_CAMS` IS A BUILD CHOICE: 2 (overhead + wrist, the rig's recorded pair)
by default, `-D DAGGER_WRIST_ONLY` for Squint's wrist-only observation.
"""

from std.math import abs, sqrt
from std.random import random_float64, random_ui64, seed as seed_rng
from std.sys import is_defined
from std.time import perf_counter_ns

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Layout, LayoutTensor

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.io.artifact_sink import sink_for_run
from noeira.io.png import save_png
from std.os import makedirs
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT, TPB
from noeira.nn.core.checkpoint import save_params, load_params
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.conv2d import Conv2D
from noeira.nn.primitives.flatten import Flatten
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, MODEL_CURRICULUM_SIZE, META_IDX_GOAL_HELD,
    META_IDX_STEP_COUNT, META_IDX_REWARD_MODE, META_IDX_TASK_PARAM_0,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.posed_reset import task_meta_words
from noeira.tasks.ppo_family_driver import (
    AgentT, RunningMeanStd, N_ENVS, ACT_DIM, OBS_CLIP, OBS_BOUND, GAMMA,
    _delta_to_env, _arg,
)
from noeira.tasks.shaping import reward_mode_words
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, TowerRendererSized, make_tower_model,
    make_tower_renderer, tower_cameras,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime M = So101TowerModel
comptime C = So101TowerConfig
comptime EnvT = Phyics3dBatchedEnv[M, C, N_ENVS, TERMINATE_ON_UNHEALTHY=False]
comptime OBS = EnvT.OBS_DIM
comptime NQ = M.NQ
comptime NB = M.NBODY

comptime N_CAMS = 1 if is_defined["DAGGER_WRIST_ONLY"]() else 2
comptime RENDER = 64
"""The traced resolution (1 sample): 144k frames/s wrist at 1024 lanes."""
comptime OBS_PX = 32 if is_defined["DAGGER_PX_32"]() else 16
"""The student's resolution: 16 (Squint's) by default, `-D DAGGER_PX_32` for
32 — the traced 64x64 averaged over 4x4 or 2x2 blocks."""
comptime PLANE = OBS_PX * OBS_PX
comptime IMG = 3 * N_CAMS * PLANE
comptime ROW = IMG + ACT_DIM
"""One replay row: the cameras' planes, then the six joints."""
comptime C_IN = 3 * N_CAMS + ACT_DIM
comptime IN_DIM = C_IN * PLANE
comptime BATCH = 1024
comptime ITER_STEPS = 32
comptime HID = 256

comptime StudentNet = Sequential[
    Conv2D[C_IN, 32, 3, 1, 1, OBS_PX, OBS_PX], ReLU[32 * PLANE],
    Conv2D[32, 64, 3, 2, 1, OBS_PX, OBS_PX], ReLU[64 * (PLANE // 4)],
    Conv2D[64, 64, 3, 2, 1, OBS_PX // 2, OBS_PX // 2], ReLU[64 * (PLANE // 16)],
    Flatten[64 * (PLANE // 16)],
    LinearReLU[64 * (PLANE // 16), HID],
    LinearReLU[HID, HID],
    Linear[HID, ACT_DIM],
]

comptime RigData = Data[RIG_DT, TOWER_MD, N_ENVS]
comptime Renderer = TowerRendererSized[N_ENVS, RENDER, RENDER, 1]


# ── device kernels ───────────────────────────────────────────────────────


def _pack_camera_kernel[
    N: Int, R: Int, P: Int
](
    rgb: LayoutTensor[DT, Layout.row_major(N * R * R * 3), MutAnyOrigin],
    ring: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    base_row: Int64,
    ch0: Int64,
):
    """One camera's `rgb` ([N, R*R*3], HWC, top row first) box-averaged to
    P x P, minus 0.5, into channels ch0..ch0+2 of rows base_row.. (CHW)."""
    comptime F = R // P
    var i = Int(global_idx.x)
    if i >= N * 3 * P * P:
        return
    var lane = i // (3 * P * P)
    var r = i % (3 * P * P)
    var c = r // (P * P)
    var p = r % (P * P)
    var oy = p // P
    var ox = p % P
    var acc = Scalar[DT](0)
    var base = lane * R * R * 3
    for dy in range(F):
        for dx in range(F):
            var px = (oy * F + dy) * R + ox * F + dx
            acc += rebind[Scalar[DT]](rgb[base + px * 3 + c])
    var row = Int(base_row) + lane
    ring[row * ROW + (Int(ch0) + c) * P * P + p] = (
        acc / Scalar[DT](F * F) - Scalar[DT](0.5)
    )


def _pack_proprio_kernel[
    N: Int, NQ_: Int
](
    qpos: LayoutTensor[DT, Layout.row_major(N * NQ_), MutAnyOrigin],
    qa: LayoutTensor[DT, Layout.row_major(ACT_DIM), MutAnyOrigin],
    ring: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    base_row: Int64,
):
    """The actuated joints (qpos addresses `qa`) into the rows' tail."""
    var i = Int(global_idx.x)
    if i >= N * ACT_DIM:
        return
    var lane = i // ACT_DIM
    var j = i % ACT_DIM
    var a = Int(rebind[Scalar[DT]](qa[j]))
    ring[(Int(base_row) + lane) * ROW + IMG + j] = rebind[Scalar[DT]](
        qpos[lane * NQ_ + a]
    )


def _gather_kernel[
    B: Int
](
    ring: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    g: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    x: LayoutTensor[DT, Layout.row_major(B * IN_DIM), MutAnyOrigin],
    blank: Int64,
):
    """Rows `g` of the replay to the student's input: the image planes as
    stored (ZERO when `blank` — the control: joints alone), then each joint
    (x 0.5) broadcast over a plane."""
    var i = Int(global_idx.x)
    if i >= B * IN_DIM:
        return
    var b = i // IN_DIM
    var k = i % IN_DIM
    var row = Int(rebind[Scalar[DT]](g[b]))
    if k < IMG:
        if blank != 0:
            x[i] = Scalar[DT](0)
        else:
            x[i] = rebind[Scalar[DT]](ring[row * ROW + k])
    else:
        var j = (k - IMG) // PLANE
        x[i] = rebind[Scalar[DT]](ring[row * ROW + IMG + j]) * Scalar[DT](0.5)


struct PixelObs(Movable):
    """The rig's own pose + renderer, and the device replay the cameras
    write into. `observe` makes rows base..base+N_ENVS-1 the pictures and
    joints of the env's CURRENT state."""

    var rm: Model[RIG_DT, TOWER_MD]
    var rd: RigData
    var r: Renderer
    var cams: List[Int]
    var qa: Tensor
    var ring: Tensor
    var cap: Int
    var blank: Bool
    """`--blank-images`: the student sees zero image planes — the CONTROL
    that says how much of its score the joints alone would get."""

    def __init__(
        out self, ctx: DeviceContext, fmd_path: String, a_qa: List[Int],
        cap: Int,
    ) raises:
        var fmd = parse_model_runtime(fmd_path)
        self.rm = make_tower_model(ctx)
        self.rd = RigData()
        self.rd.upload_all(ctx)
        self.r = make_tower_renderer[N_ENVS, RENDER, RENDER, 1](ctx, fmd, self.rm)
        var both = tower_cameras(fmd)  # [overhead, wrist]
        self.cams = List[Int]()
        comptime if N_CAMS == 2:
            self.cams.append(both[0])
        self.cams.append(both[1])
        self.qa = Tensor.alloc(ACT_DIM)
        for j in range(ACT_DIM):
            self.qa.data[j] = Scalar[DT](a_qa[j])
        self.qa.upload(ctx)
        # ⚠ THE RING IS INDEXED THROUGH A 32-BIT LayoutTensor OFFSET
        # (`_a_layouttensor_index_is_int32_regardless_of_your_arithmetic`).
        if cap * ROW >= (1 << 31):
            raise Error("pixel dagger: --replay " + String(cap) + " x "
                        + String(ROW) + " words overflows a 32-bit index")
        if cap % N_ENVS != 0:
            raise Error("pixel dagger: --replay must be a multiple of "
                        + String(N_ENVS))
        self.cap = cap
        self.ring = Tensor.alloc_gpu(ctx, cap * ROW)
        self.blank = False

    def observe(
        mut self, ctx: DeviceContext, env_qpos: DeviceBuffer[DT], base_row: Int
    ) raises:
        ctx.enqueue_copy(self.rd.qpos.dev.value(), env_qpos)
        forward_kinematics["gpu", RIG_DT, TOWER_MD, N_ENVS](
            self.rd, self.rm, Optional(ctx)
        )
        comptime n_img = (N_ENVS * 3 * PLANE + TPB - 1) // TPB
        for k in range(len(self.cams)):
            self.r.render(ctx, self.rd, self.rm, self.cams[k])
            # ⚠ ONE non-owning view per kernel (two miscompile on Metal —
            # `Tensor.view_gpu`); the ring is an owned Tensor.
            var rgb = Tensor.view_gpu(
                ctx, mptr(self.r.rgb.unsafe_ptr()),
                N_ENVS * RENDER * RENDER * 3,
            )
            ctx.enqueue_function[_pack_camera_kernel[N_ENVS, RENDER, OBS_PX]](
                rgb.lt["gpu", Layout.row_major(N_ENVS * RENDER * RENDER * 3)](),
                self.ring.lt["gpu", Layout.row_major(1)](),
                Int64(base_row),
                Int64(3 * k),
                grid_dim=n_img,
                block_dim=TPB,
            )
        comptime n_q = (N_ENVS * ACT_DIM + TPB - 1) // TPB
        ctx.enqueue_function[_pack_proprio_kernel[N_ENVS, NQ]](
            self.rd.qpos.lt["gpu", Layout.row_major(N_ENVS * NQ)](),
            self.qa.lt["gpu", Layout.row_major(ACT_DIM)](),
            self.ring.lt["gpu", Layout.row_major(1)](),
            Int64(base_row),
            grid_dim=n_q,
            block_dim=TPB,
        )

    def gather[B: Int](
        mut self, ctx: DeviceContext, mut g: Tensor, mut x: Tensor
    ) raises:
        comptime n = (B * IN_DIM + TPB - 1) // TPB
        ctx.enqueue_function[_gather_kernel[B]](
            self.ring.lt["gpu", Layout.row_major(1)](),
            g.lt["gpu", Layout.row_major(B)](),
            x.lt["gpu", Layout.row_major(B * IN_DIM)](),
            Int64(1 if self.blank else 0),
            grid_dim=n,
            block_dim=TPB,
        )


def run_pixel_dagger(args: List[String], driver: String) raises:
    # ── flags ────────────────────────────────────────────────────────────
    var task = String("so101_tower_lift_real_layout")
    if len(args) > 1 and not args[1].startswith("--"):
        task = args[1]
    var teacher_dir = _arg(args, "--teacher", "")
    if teacher_dir.byte_length() == 0:
        raise Error("pixel dagger: --teacher RUN_DIR (a PPO run: "
                    "checkpoints/last.ckpt + obs_norm.txt) is required")
    var total_steps = Int(_arg(args, "--steps", "20000000"))
    var beta_steps = Int(_arg(args, "--beta-steps", "2000000"))
    var updates = Int(_arg(args, "--updates", "32"))
    var lr = Float64(_arg(args, "--lr", "0.0003"))
    # the default ring: 512 steps of every lane, capped so `cap * ROW` stays
    # under the 32-bit LayoutTensor offset (a 32x32 row is 6150 words)
    var cap_max = ((1 << 31) - 1) // ROW // N_ENVS * N_ENVS
    var cap = Int(_arg(args, "--replay", String(min(N_ENVS * 512, cap_max))))
    var seed = Int(_arg(args, "--seed", "1"))
    var eval_rounds = Int(_arg(args, "--eval-rounds", "2"))
    var init_student = _arg(args, "--init-student", "")
    var png_dir = _arg(args, "--png", "")
    var blank = _arg(args, "--blank-images", "0") == "1"
    var grip_sign = _arg(args, "--gripper-sign", "0") == "1"
    seed_rng(seed)
    var family = String("so101_tower")
    var family_path = String("noeira/tasks/families/so101_tower.family")

    print("=" * 70)
    print("PIXEL DAgger on", family, "—", task)
    print("  lanes", N_ENVS, "| cameras", N_CAMS, "| traced", RENDER, "->",
          OBS_PX, "| input", C_IN, "x", OBS_PX, "x", OBS_PX)
    print("  teacher", teacher_dir)
    print("  steps", total_steps, "| beta 1 -> 0 over", beta_steps,
          "| updates", updates, "x batch", BATCH, "every", ITER_STEPS,
          "steps | lr", lr, "| replay", cap, "rows")
    print("=" * 70)

    var f = load_family(family_path)
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    var mw = task_meta_words(
        task, family, C.SHAPE_W_GOAL, C.SHAPE_W_REACH, C.GOAL_MARGIN,
        C.REACH_MARGIN,
    )
    var rw = reward_mode_words(True, 0.0)
    var jadr = List[Int]()
    var acc = 0
    for i in range(len(fmd.joints)):
        jadr.append(acc)
        acc += fmd.joints[i].nq
    var a_qa = List[Int]()
    var a_lo = List[Float64]()
    var a_hi = List[Float64]()
    for i in range(ACT_DIM):
        a_qa.append(jadr[fmd.actuators[i].joint_id])
        a_lo.append(fmd.actuators[i].ctrl_min)
        a_hi.append(fmd.actuators[i].ctrl_max)

    var run = RunContext(
        project=String("so101-tower"), driver=driver,
        slug=String("dagger-px-") + task, env=String("family:") + family,
        task=task, seed=seed, device="gpu",
    )
    print("  run", run.dir)
    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "DAgger (pixel student, state PPO teacher)")
    logger.set_config("task", task)
    logger.set_config("teacher", teacher_dir)
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("n_cams", String(N_CAMS))
    logger.set_config("render", String(RENDER))
    logger.set_config("obs_px", String(OBS_PX))
    logger.set_config("beta_steps", String(beta_steps))
    logger.set_config("updates", String(updates))
    logger.set_config("batch", String(BATCH))
    logger.set_config("lr", String(lr))
    logger.set_config("replay", String(cap))
    logger.set_config("blank_images", String(blank))
    logger.set_config("gripper_sign", String(grip_sign))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    with DeviceContext() as ctx:
        # ── the teacher: the PPO actor, frozen, and its obs statistics ───
        var teacher = AgentT[OBS](
            ctx=ctx, actor_lr=Scalar[DT](0.0), critic_lr=Scalar[DT](0.0),
            gamma=Scalar[DT](GAMMA), gae_lambda=Scalar[DT](0.95),
            clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.0),
            action_scale=Scalar[DT](1.0), log_std_init=Scalar[DT](-1.0),
            window_size=100, initial_episode_fill=Scalar[DT](0.0),
            max_grad_norm=Scalar[DT](0.5),
        )
        teacher.trainer.load_state(teacher_dir + "/checkpoints/last.ckpt")
        var obs_rms = RunningMeanStd(OBS)
        obs_rms.load(teacher_dir + "/obs_norm.txt")

        # ── the student ──────────────────────────────────────────────────
        var student = StudentNet.make["gpu", Kaiming](Optional(ctx))
        if init_student.byte_length() > 0:
            load_params["gpu"](student, init_student, Optional(ctx))
            print("  init: student from", init_student)
        var opt = Adam(lr=Scalar[DT](lr))

        # ── the env, as the PPO driver sets it up ────────────────────────
        var env = EnvT(ctx)
        for i in range(MODEL_CURRICULUM_SIZE):
            env.mf.curriculum.data[i] = Scalar[DT](cw[i])
        env.mf.curriculum.upload(ctx)
        for e in range(N_ENVS):
            var mb = e * METADATA_SIZE
            for k in range(METADATA_SIZE):
                env.d.meta.data[mb + k] = Scalar[DT](0)
            for k in range(len(mw[0])):
                env.d.meta.data[mb + mw[0][k]] = Scalar[DT](mw[1][k])
            for k in range(len(rw)):
                env.d.meta.data[mb + META_IDX_REWARD_MODE + k] = Scalar[DT](rw[k])
        env.d.meta.upload(ctx)
        ctx.synchronize()
        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(seed))
        ctx.synchronize()
        env.d.meta.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            env.d.meta.data[e * METADATA_SIZE + META_IDX_STEP_COUNT] = Scalar[DT](
                Int(random_float64() * Float64(C.MAX_STEPS))
            )
        env.d.meta.upload(ctx)
        ctx.synchronize()

        var px = PixelObs(ctx, scene_path(f), a_qa, cap)
        px.blank = blank
        if blank:
            print("  ⚠ --blank-images 1: the student sees ZERO image planes"
                  " (the joints-only control)")

        # ── buffers ──────────────────────────────────────────────────────
        var raw_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
        var cur_n = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
        var act_t = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
        var env_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
        var done_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        ctx.synchronize()
        var obs_dev = DeviceBuffer[DT](ctx, env.obs_ptr(), N_ENVS * OBS, owning=False)
        var act_dev = DeviceBuffer[DT](ctx, env.action_ptr(), N_ENVS * ACT_DIM, owning=False)
        var done_dev = DeviceBuffer[DT](ctx, env.done_ptr(), N_ENVS, owning=False)
        var qpos_dev = DeviceBuffer[DT](
            ctx, env.d.qpos.dev.value().unsafe_ptr(), N_ENVS * NQ, owning=False
        )
        var labels = List[Scalar[DT]](length=cap * ACT_DIM, fill=Scalar[DT](0))
        var arm_q = List[Float64](length=N_ENVS * ACT_DIM, fill=0.0)
        # the student's acting batch (N_ENVS rows) and training batch
        var g_act = Tensor.alloc(N_ENVS)
        var x_act = Tensor.alloc(N_ENVS * IN_DIM)
        var y_act = Tensor.alloc(N_ENVS * ACT_DIM)
        var g_tr = Tensor.alloc(BATCH)
        var x_tr = Tensor.alloc(BATCH * IN_DIM)
        var pred = Tensor.alloc(BATCH * ACT_DIM)
        var gout = Tensor.alloc(BATCH * ACT_DIM)
        var gin = Tensor.alloc(BATCH * IN_DIM)
        g_act.upload(ctx)
        x_act.upload(ctx)
        y_act.upload(ctx)
        g_tr.upload(ctx)
        x_tr.upload(ctx)
        pred.upload(ctx)
        gout.upload(ctx)
        gin.upload(ctx)
        ctx.synchronize()

        var stud_lane = List[Bool](length=N_ENVS, fill=False)
        var succ = List[Bool](length=N_ENVS, fill=False)
        var hist_s = List[Bool]()  # student-executed episodes' success
        var hist_t = List[Bool]()  # teacher-executed
        var n_filled = 0
        var step = 0
        var it = 0
        var loss_acc = 0.0
        var loss_n = 0
        var t0 = perf_counter_ns()
        var ckpt = run.dir + "/student.ckpt"

        ctx.enqueue_copy(raw_h, obs_dev)
        ctx.synchronize()
        if png_dir.byte_length() > 0:
            _dump_obs_png(ctx, px, png_dir, qpos_dev)

        while step < total_steps:
            var beta = 1.0 - Float64(step) / Float64(max(beta_steps, 1))
            if beta < 0.0:
                beta = 0.0
            var base = (it % (cap // N_ENVS)) * N_ENVS
            # 1-2. the pictures and joints of the CURRENT state
            px.observe(ctx, qpos_dev, base)
            # 3. the teacher, on the state
            var rp = mptr(raw_h.unsafe_ptr())
            for e in range(N_ENVS):
                for j in range(ACT_DIM):
                    arm_q[e * ACT_DIM + j] = Float64(
                        rp[unsafe_offset = e * OBS + a_qa[j]]
                    )
            obs_rms.normalize_into(
                rp, mptr(cur_n.unsafe_ptr()), N_ENVS, OBS, OBS_CLIP
            )
            teacher.trainer.select_greedy_action_batched(
                mptr(cur_n.unsafe_ptr()), mptr(act_t.unsafe_ptr())
            )
            var at = mptr(act_t.unsafe_ptr())
            for e in range(N_ENVS):
                for j in range(ACT_DIM):
                    labels[(base + e) * ACT_DIM + j] = at[
                        unsafe_offset = e * ACT_DIM + j
                    ]
            n_filled = min(n_filled + N_ENVS, cap)
            # 4. the student, on the rows just written
            var any_student = False
            for e in range(N_ENVS):
                if stud_lane[e]:
                    any_student = True
            if any_student:
                for e in range(N_ENVS):
                    g_act.data[e] = Scalar[DT](base + e)
                g_act.upload_resident(ctx)
                px.gather[N_ENVS](ctx, g_act, x_act)
                student.forward["gpu", N_ENVS](
                    TensorRefs[1](x_act), y_act, Optional(ctx)
                )
                y_act.download(ctx)
                ctx.synchronize()
                for e in range(N_ENVS):
                    if stud_lane[e]:
                        for j in range(ACT_DIM):
                            at[unsafe_offset = e * ACT_DIM + j] = _student_act(
                                y_act.data[e * ACT_DIM + j], j, grip_sign
                            )
            _delta_to_env(
                mptr(act_t.unsafe_ptr()), mptr(env_act.unsafe_ptr()),
                arm_q, a_lo, a_hi,
            )
            ctx.enqueue_copy(act_dev, env_act)
            # 5. step, tally, reset
            env.step_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(it + 1))
            ctx.enqueue_copy(raw_h, obs_dev)
            ctx.enqueue_copy(done_h, done_dev)
            env.d.meta.download(ctx)
            ctx.synchronize()
            var rq = mptr(raw_h.unsafe_ptr())
            var dh = mptr(done_h.unsafe_ptr())
            var forced = False
            for e in range(N_ENVS):
                for k in range(OBS):
                    var v = Float64(rq[unsafe_offset = e * OBS + k])
                    if not (v == v) or abs(v) > OBS_BOUND:
                        dh[unsafe_offset=e] = Scalar[DT](1)
                        forced = True
                        break
                if env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD] > Scalar[DT](0.5):
                    succ[e] = True
                if dh[unsafe_offset=e] > Scalar[DT](0.5):
                    if stud_lane[e]:
                        hist_s.append(succ[e])
                    else:
                        hist_t.append(succ[e])
                    succ[e] = False
                    stud_lane[e] = random_float64() >= beta
            if forced:
                ctx.enqueue_copy(done_dev, done_h)
            env.selective_reset_batch[N_ENVS](
                ctx=ctx, rng_seed=UInt64(seed * 7919 + it + 1)
            )
            ctx.enqueue_copy(raw_h, obs_dev)
            ctx.synchronize()
            step += N_ENVS
            it += 1

            # ── the regression, on the aggregate dataset ─────────────────
            if it % ITER_STEPS == 0:
                for _u in range(updates):
                    for b in range(BATCH):
                        g_tr.data[b] = Scalar[DT](
                            Int(random_ui64(0, UInt64(n_filled - 1)))
                        )
                    g_tr.upload_resident(ctx)
                    px.gather[BATCH](ctx, g_tr, x_tr)
                    student.forward["gpu", BATCH](
                        TensorRefs[1](x_tr), pred, Optional(ctx)
                    )
                    pred.download(ctx)
                    ctx.synchronize()
                    var loss = 0.0
                    for b in range(BATCH):
                        var row = Int(g_tr.data[b])
                        for j in range(ACT_DIM):
                            var e2 = Float64(pred.data[b * ACT_DIM + j]) - Float64(
                                labels[row * ACT_DIM + j]
                            )
                            loss += e2 * e2
                            gout.data[b * ACT_DIM + j] = Scalar[DT](
                                2.0 * e2 / Float64(BATCH * ACT_DIM)
                            )
                    loss_acc += loss / Float64(BATCH * ACT_DIM)
                    loss_n += 1
                    gout.upload_resident(ctx)
                    student.zero_grad["gpu"](Optional(ctx))
                    student.vjp["gpu", BATCH](
                        TensorRefs[1](x_tr), gout, TensorRefs[1](gin),
                        Optional(ctx),
                    )
                    opt.step["gpu"](student, Optional(ctx))
                if it % (ITER_STEPS * 10) == 0:
                    var secs = Float64(perf_counter_ns() - t0) / 1e9
                    var rs = _rate(hist_s)
                    var rt = _rate(hist_t)
                    var mse = loss_acc / Float64(max(loss_n, 1))
                    print("  step", step, "| beta", beta, "| mse", mse,
                          "| student success", rs, "(", len(hist_s),
                          "ep) | teacher", rt, "(", len(hist_t), "ep) |",
                          Int(Float64(step) / secs), "steps/s")
                    logger.log_scalar("mse", mse, step)
                    logger.log_scalar("beta", beta, step)
                    logger.log_scalar("student_success", rs, step)
                    logger.log_scalar("teacher_success", rt, step)
                    logger.log_scalar("sps", Float64(step) / secs, step)
                    loss_acc = 0.0
                    loss_n = 0
                if it % (ITER_STEPS * 100) == 0:
                    save_params["gpu"](student, ckpt, Optional(ctx))
        save_params["gpu"](student, ckpt, Optional(ctx))
        print("  student", ckpt)

        # ── greedy evaluation of the STUDENT, held-out placements ────────
        var ok_all = 0
        var n_all = 0
        for rnd in range(eval_rounds):
            env.reset_batch[N_ENVS](
                ctx=ctx, rng_seed=UInt64(1_000_003 + seed * 101 + rnd)
            )
            ctx.enqueue_copy(raw_h, obs_dev)
            ctx.synchronize()
            var held = List[Bool](length=N_ENVS, fill=False)
            var held_end = List[Bool](length=N_ENVS, fill=False)
            var z0 = List[Float64](length=N_ENVS, fill=0.0)
            var rise = List[Float64](length=N_ENVS, fill=0.0)
            var over = List[Bool](length=N_ENVS, fill=False)
            for t in range(C.MAX_STEPS - 1):
                px.observe(ctx, qpos_dev, 0)
                var rq = mptr(raw_h.unsafe_ptr())
                for e in range(N_ENVS):
                    for j in range(ACT_DIM):
                        arm_q[e * ACT_DIM + j] = Float64(
                            rq[unsafe_offset = e * OBS + a_qa[j]]
                        )
                    g_act.data[e] = Scalar[DT](e)
                g_act.upload_resident(ctx)
                px.gather[N_ENVS](ctx, g_act, x_act)
                student.forward["gpu", N_ENVS](
                    TensorRefs[1](x_act), y_act, Optional(ctx)
                )
                y_act.download(ctx)
                ctx.synchronize()
                var at = mptr(act_t.unsafe_ptr())
                for k in range(N_ENVS * ACT_DIM):
                    at[unsafe_offset=k] = _student_act(
                        y_act.data[k], k % ACT_DIM, grip_sign
                    )
                _delta_to_env(
                    mptr(act_t.unsafe_ptr()), mptr(env_act.unsafe_ptr()),
                    arm_q, a_lo, a_hi,
                )
                ctx.enqueue_copy(act_dev, env_act)
                env.step_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(t + 1))
                ctx.enqueue_copy(raw_h, obs_dev)
                env.d.meta.download(ctx)
                env.d.xpos.download(ctx)
                ctx.synchronize()
                for e in range(N_ENVS):
                    var hb = env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD] > Scalar[DT](0.5)
                    if hb:
                        held[e] = True
                    held_end[e] = hb
                    var mb = e * METADATA_SIZE + META_IDX_TASK_PARAM_0
                    var ia = Int(env.d.meta.data[mb + 1])
                    var ib = Int(env.d.meta.data[mb + 2])
                    var xb = e * NB * 3
                    var za = Float64(env.d.xpos.data[xb + ia * 3 + 2])
                    if t == 0:
                        z0[e] = za
                    if za - z0[e] > rise[e]:
                        rise[e] = za - z0[e]
                    # over term 0's second body while lifted (the bowl, on
                    # cube in bowl; the desk's origin on lift — read loosely)
                    var ex = Float64(env.d.xpos.data[xb + ia * 3] - env.d.xpos.data[xb + ib * 3])
                    var ey = Float64(env.d.xpos.data[xb + ia * 3 + 1] - env.d.xpos.data[xb + ib * 3 + 1])
                    if za - z0[e] > 0.02 and sqrt(ex * ex + ey * ey) < 0.045:
                        over[e] = True
            var ok = 0
            var n_end = 0
            var n_lift = 0
            var n_over = 0
            for e in range(N_ENVS):
                if over[e]:
                    n_over += 1
                if held[e]:
                    ok += 1
                if held_end[e]:
                    n_end += 1
                if rise[e] > 0.02:
                    n_lift += 1
            print("  student greedy eval round", rnd, ":", ok, "/", N_ENVS,
                  "| held at the end", n_end, "| lifted >2cm", n_lift,
                  "| over b while lifted", n_over)
            ok_all += ok
            n_all += N_ENVS
        var rate = Float64(ok_all) / Float64(max(n_all, 1))
        if n_all > 0:
            print("  STUDENT GREEDY SUCCESS", ok_all, "/", n_all, "=", rate)
            logger.log_scalar("eval_success_rate", rate, step)
        print("=" * 70)
        finish_run(run, logger, artifacts,
                   String("eval_success_rate=") + String(rate))
        _ = logger


def _dump_obs_png(
    ctx: DeviceContext, mut px: PixelObs, dir: String,
    qpos_dev: DeviceBuffer[DT],
) raises:
    """Lanes 0..3's first observation, each camera upscaled x8, as PNGs — to
    SEE what the student sees (a transposed or blank plane trains silently)."""
    makedirs(dir, exist_ok=True)
    px.observe(ctx, qpos_dev, 0)
    var h = ctx.enqueue_create_host_buffer[DT](4 * ROW)
    ctx.enqueue_copy(h, px.ring.dev.value().create_sub_buffer[DT](0, 4 * ROW))
    ctx.synchronize()
    var hp = h.unsafe_ptr()
    comptime S = 8
    comptime W = OBS_PX * S
    for lane in range(4):
        for cam in range(N_CAMS):
            var img = List[UInt8](length=W * W * 3, fill=UInt8(0))
            for y in range(W):
                for x in range(W):
                    for c in range(3):
                        var v = Float64(hp[
                            lane * ROW + (3 * cam + c) * PLANE
                            + (y // S) * OBS_PX + x // S
                        ]) + 0.5
                        var b = Int(v * 255.0 + 0.5)
                        img[(y * W + x) * 3 + c] = UInt8(max(0, min(255, b)))
            save_png(dir + "/lane" + String(lane) + "_cam" + String(cam) + ".png",
                     img, W, W, 3)
    print("  png: lanes 0-3 x", N_CAMS, "cameras ->", dir)


comptime GRIPPER_ACT = 5
"""The gripper's action index (the rig's `RIG_GRIPPER`)."""


@always_inline
def _student_act(v: Scalar[DT], j: Int, grip_sign: Bool) -> Scalar[DT]:
    """The student's output as executed: clamped to [-1, 1]; with
    `--gripper-sign 1` the GRIPPER word snapped to +-1. The release is a few
    steps per episode, so a regression under-weights it and can leave the
    jaws opening too slowly to drop the brick (cube in bowl, 27 Sep: over
    the bowl 59 %, success 12.7 %)."""
    if grip_sign and j == GRIPPER_ACT:
        return Scalar[DT](1) if v > Scalar[DT](0) else Scalar[DT](-1)
    if v > Scalar[DT](1):
        return Scalar[DT](1)
    if v < Scalar[DT](-1):
        return Scalar[DT](-1)
    return v


def _rate(h: List[Bool]) -> Float64:
    """Success over the last 1024 episodes of a history."""
    var n = len(h)
    var lo = n - 1024 if n > 1024 else 0
    var s = 0
    for k in range(lo, n):
        if h[k]:
            s += 1
    return Float64(s) / Float64(n - lo) if n - lo > 0 else 0.0
