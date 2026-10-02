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
    _delta_to_env, _arg, _lag_reset, _augment, _hist_push, _hist_clear,
    _delta_targets, _targets_to_env, _target_targets, _target_reset,
)
from noeira.tasks.delta_action import ServoLag, DELTA_ARM, DELTA_GRIPPER, TARGET_OBS
from noeira.tasks.shaping import reward_mode_words
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, TowerRendererSized, make_tower_model,
    make_tower_renderer, tower_cameras, scale_tower_camera_dr, RIG_DR_TARGET,
)
from noeira.physics3d.raytrace.randomize import (
    DomainRandConfig, VisualRandomizer, geom_labels,
    so101_tower_surface_groups,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.tasks.pixel_student import (
    StudentNet, N_CAMS, RENDER, OBS_PX, PLANE, IMG, C_IN, IN_DIM, ACT,
    OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, WINDOWED, camera_names,
    window_in_render,
    PROPRIO, JOINT_SCALE, JOINT_VEL_SCALE, student_act, write_pixel_manifest,
    PROPRIO_STATE, HIST_WORDS, TARGET_LEAD_SCALE,
)

comptime M = So101TowerModel
comptime C = So101TowerConfig
comptime EnvT = Phyics3dBatchedEnv[M, C, N_ENVS, TERMINATE_ON_UNHEALTHY=False]
comptime OBS = EnvT.OBS_DIM
comptime T_OBS = OBS + HIST_WORDS + TARGET_OBS
"""The teacher's observation: the env's, then (`-D TASK_PPO_ACT_HIST=K`) the
last K executed actions — the same words the student sees as planes."""
comptime EXW = HIST_WORDS + TARGET_OBS
"""The host-made joint words per lane: the action history, then (target
mode) the target's lead over the joints."""
comptime HW1 = EXW if EXW > 0 else 1
comptime NQ = M.NQ
comptime NB = M.NBODY

comptime ROW = IMG + PROPRIO
"""One replay row: the cameras' planes, then the joints (and, with
`DAGGER_JOINT_VEL`, their velocities; with `TASK_PPO_ACT_HIST`, the last
executed actions), RAW — scaled in the gather."""
comptime NV = M.NV
comptime BATCH = 1024
comptime ITER_STEPS = 32

comptime RigData = Data[RIG_DT, TOWER_MD, N_ENVS]
comptime Renderer = TowerRendererSized[N_ENVS, RENDER, RENDER, 1]
"""The wrist's (and, without DAGGER_WINDOW, the overhead's) square trace."""
comptime OverheadRenderer = TowerRendererSized[
    N_ENVS, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, 1
]
"""The overhead's: the full 4:3 frame with DAGGER_WINDOW."""


# ── device kernels ───────────────────────────────────────────────────────


def _pack_camera_kernel[
    N: Int, RW: Int, RH: Int, P: Int
](
    rgb: LayoutTensor[DT, Layout.row_major(N * RW * RH * 3), MutAnyOrigin],
    ring: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    base_row: Int64,
    ch0: Int64,
    wx0: Scalar[DT],
    wy0: Scalar[DT],
    wx1: Scalar[DT],
    wy1: Scalar[DT],
):
    """One camera's `rgb` ([N, RW*RH*3], HWC, top row first) — its WINDOW
    (wx0, wy0)-(wx1, wy1) in render pixels, AREA-averaged to P x P, minus
    0.5 — into channels ch0..ch0+2 of rows base_row.. (CHW). The device twin
    of `pixel_student.render_to_planes` (float32: Metal has no float64); a
    window on whole pixels is the plain block mean."""
    var i = Int(global_idx.x)
    if i >= N * 3 * P * P:
        return
    var lane = i // (3 * P * P)
    var r = i % (3 * P * P)
    var c = r // (P * P)
    var p = r % (P * P)
    var oy = p // P
    var ox = p % P
    var cw = (wx1 - wx0) / Scalar[DT](P)
    var chh = (wy1 - wy0) / Scalar[DT](P)
    var ax = wx0 + Scalar[DT](ox) * cw
    var bx = ax + cw
    var ay = wy0 + Scalar[DT](oy) * chh
    var by = ay + chh
    var acc = Scalar[DT](0)
    var wsum = Scalar[DT](0)
    var base = lane * RW * RH * 3
    var py_end = min(Int(by) + 1, RH)
    var px_end = min(Int(bx) + 1, RW)
    for py in range(Int(ay), py_end):
        var lo_y = ay if ay > Scalar[DT](py) else Scalar[DT](py)
        var hi_y = by if by < Scalar[DT](py + 1) else Scalar[DT](py + 1)
        var wy = hi_y - lo_y
        if wy <= Scalar[DT](0):
            continue
        for px in range(Int(ax), px_end):
            var lo_x = ax if ax > Scalar[DT](px) else Scalar[DT](px)
            var hi_x = bx if bx < Scalar[DT](px + 1) else Scalar[DT](px + 1)
            var wx = hi_x - lo_x
            if wx <= Scalar[DT](0):
                continue
            acc += wx * wy * rebind[Scalar[DT]](rgb[base + (py * RW + px) * 3 + c])
            wsum += wx * wy
    var row = Int(base_row) + lane
    ring[row * ROW + (Int(ch0) + c) * P * P + p] = acc / wsum - Scalar[DT](0.5)


def _pack_proprio_kernel[
    N: Int, NQ_: Int, NV_: Int
](
    qpos: LayoutTensor[DT, Layout.row_major(N * NQ_), MutAnyOrigin],
    qvel: LayoutTensor[DT, Layout.row_major(N * NV_), MutAnyOrigin],
    qa: LayoutTensor[DT, Layout.row_major(ACT_DIM), MutAnyOrigin],
    da: LayoutTensor[DT, Layout.row_major(ACT_DIM), MutAnyOrigin],
    hist: LayoutTensor[DT, Layout.row_major(N * HW1), MutAnyOrigin],
    ring: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    base_row: Int64,
):
    """The actuated joints (qpos addresses `qa`) — with `DAGGER_JOINT_VEL`
    their velocities (dof addresses `da`), with `TASK_PPO_ACT_HIST` the lane's
    action history (`hist`, [N, HIST_WORDS]) — into the rows' tail, raw."""
    var i = Int(global_idx.x)
    if i >= N * PROPRIO:
        return
    var lane = i // PROPRIO
    var j = i % PROPRIO
    var v: Scalar[DT]
    if j < ACT_DIM:
        v = rebind[Scalar[DT]](qpos[lane * NQ_ + Int(rebind[Scalar[DT]](qa[j]))])
    elif j < PROPRIO_STATE:
        v = rebind[Scalar[DT]](
            qvel[lane * NV_ + Int(rebind[Scalar[DT]](da[j - ACT_DIM]))]
        )
    else:
        v = rebind[Scalar[DT]](hist[lane * HW1 + j - PROPRIO_STATE])
    ring[(Int(base_row) + lane) * ROW + IMG + j] = v


comptime AUG_WORDS = 7
"""Per training sample: brightness, contrast, R/G/B gains, shift x, shift y."""


def _gather_kernel[
    B: Int
](
    ring: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    g: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    x: LayoutTensor[DT, Layout.row_major(B * IN_DIM), MutAnyOrigin],
    aug: LayoutTensor[DT, Layout.row_major(B * AUG_WORDS), MutAnyOrigin],
    blank: Int64,
    use_aug: Int64,
):
    """Rows `g` of the replay to the student's input: the image planes as
    stored (ZERO when `blank` — the control: joints alone), then the joint
    planes (x their scales). With `use_aug`, each sample's image planes are
    photometrically jittered and shifted by its `aug` words (TRAINING ONLY —
    the acting gather passes 0): v' = clamp((v x contrast x gain_c) +
    brightness), read at (y + sy, x + sx) clamped to the plane."""
    var i = Int(global_idx.x)
    if i >= B * IN_DIM:
        return
    var b = i // IN_DIM
    var k = i % IN_DIM
    var row = Int(rebind[Scalar[DT]](g[b]))
    if k < IMG:
        if blank != 0:
            x[i] = Scalar[DT](0)
        elif use_aug != 0:
            var plane = k // PLANE
            var p = k % PLANE
            var oy = p // OBS_PX
            var ox = p % OBS_PX
            var ab = b * AUG_WORDS
            var sx = Int(rebind[Scalar[DT]](aug[ab + 5]))
            var sy = Int(rebind[Scalar[DT]](aug[ab + 6]))
            var yy = min(max(oy + sy, 0), OBS_PX - 1)
            var xx = min(max(ox + sx, 0), OBS_PX - 1)
            var v = rebind[Scalar[DT]](ring[row * ROW + plane * PLANE + yy * OBS_PX + xx])
            var c = plane % 3
            v = v * rebind[Scalar[DT]](aug[ab + 1]) * rebind[Scalar[DT]](aug[ab + 2 + c])
            v += rebind[Scalar[DT]](aug[ab])
            if v > Scalar[DT](0.5):
                v = Scalar[DT](0.5)
            elif v < Scalar[DT](-0.5):
                v = Scalar[DT](-0.5)
            x[i] = v
        else:
            x[i] = rebind[Scalar[DT]](ring[row * ROW + k])
    else:
        var j = (k - IMG) // PLANE
        var sc = Scalar[DT](JOINT_SCALE) if j < ACT_DIM else (
            Scalar[DT](JOINT_VEL_SCALE) if j < PROPRIO_STATE else (
                Scalar[DT](1) if j < PROPRIO_STATE + HIST_WORDS
                else Scalar[DT](TARGET_LEAD_SCALE)
            )
        )
        x[i] = rebind[Scalar[DT]](ring[row * ROW + IMG + j]) * sc


struct PixelObs(Movable):
    """The rig's own pose + renderer, and the device replay the cameras
    write into. `observe` makes rows base..base+N_ENVS-1 the pictures and
    joints of the env's CURRENT state."""

    var rm: Model[RIG_DT, TOWER_MD]
    var rd: RigData
    var r: Renderer
    var r_o: OverheadRenderer
    var dr_w: VisualRandomizer[RIG_DT]
    var dr_o: VisualRandomizer[RIG_DT]
    var clean_w: VisualRandomizer[RIG_DT]
    var clean_o: VisualRandomizer[RIG_DT]
    """`off` randomizers: their `apply` RESTORES the base look (the rig's
    calibrated one) — the clean evaluation."""
    var dr_on: Bool
    var aug_on: Bool
    var aug: Tensor
    var win: List[Float64]
    """Per camera slot, its window in its render's pixels (x0 y0 x1 y1)."""
    var cams: List[Int]
    var qa: Tensor
    var da: Tensor
    var hist_t: Tensor
    """[N_ENVS, HW1]: the lanes' action histories, uploaded per `observe`."""
    var ring: Tensor
    var cap: Int
    var blank: Bool
    """`--blank-images`: the student sees zero image planes — the CONTROL
    that says how much of its score the joints alone would get."""

    def __init__(
        out self, ctx: DeviceContext, fmd_path: String, a_qa: List[Int],
        a_da: List[Int], cap: Int, dr_name: String, dr_seed: Int,
    ) raises:
        var fmd = parse_model_runtime(fmd_path)
        self.rm = make_tower_model(ctx)
        self.rd = RigData()
        self.rd.upload_all(ctx)
        self.r = make_tower_renderer[N_ENVS, RENDER, RENDER, 1](ctx, fmd, self.rm)
        self.r_o = make_tower_renderer[
            N_ENVS, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, 1
        ](ctx, fmd, self.rm)
        self.win = List[Float64]()
        var names = camera_names()
        for k in range(N_CAMS):
            var is_o = N_CAMS == 2 and k == 0
            var w = window_in_render(
                names[k], OVERHEAD_RENDER_W if is_o else RENDER,
                OVERHEAD_RENDER_H if is_o else RENDER,
            )
            self.win.append(w[0])
            self.win.append(w[1])
            self.win.append(w[2])
            self.win.append(w[3])
        var both = tower_cameras(fmd)  # [overhead, wrist]
        self.cams = List[Int]()
        comptime if N_CAMS == 2:
            self.cams.append(both[0])
        self.cams.append(both[1])
        # ⚠ DOMAIN RANDOMISATION: one randomizer per renderer (each holds its
        # own visual tables), same config and seed, so a draw is ONE look
        # across both cameras; both write the model's camera rows the same
        # way. Built AFTER `make_tower_renderer` applied the rig's look, so the
        # draws jitter around it (`apply_tower_look`'s rule).
        var labels = geom_labels(fmd)
        var dcfg = DomainRandConfig.parse(dr_name, UInt64(dr_seed))
        self.dr_w = VisualRandomizer[RIG_DT](
            dcfg, so101_tower_surface_groups(dr_name == "room"), self.r.vis, self.rm, labels,
            self.cams.copy(), self.r.background, RIG_DR_TARGET,
        )
        scale_tower_camera_dr(self.dr_w, fmd)
        self.dr_o = VisualRandomizer[RIG_DT](
            dcfg, so101_tower_surface_groups(dr_name == "room"), self.r_o.vis, self.rm, labels,
            self.cams.copy(), self.r_o.background, RIG_DR_TARGET,
        )
        scale_tower_camera_dr(self.dr_o, fmd)
        var off = DomainRandConfig.off()
        self.clean_w = VisualRandomizer[RIG_DT](
            off, so101_tower_surface_groups(), self.r.vis, self.rm, labels,
            self.cams.copy(), self.r.background, RIG_DR_TARGET,
        )
        self.clean_o = VisualRandomizer[RIG_DT](
            off, so101_tower_surface_groups(), self.r_o.vis, self.rm, labels,
            self.cams.copy(), self.r_o.background, RIG_DR_TARGET,
        )
        self.dr_on = dr_name != "off" and dr_name != ""
        self.aug_on = False
        # sized for the larger of the training batch and the acting batch:
        # the acting gather passes it too (and never reads it)
        self.aug = Tensor.alloc(max(BATCH, N_ENVS) * AUG_WORDS)
        self.aug.upload(ctx)
        self.qa = Tensor.alloc(ACT_DIM)
        for j in range(ACT_DIM):
            self.qa.data[j] = Scalar[DT](a_qa[j])
        self.qa.upload(ctx)
        self.da = Tensor.alloc(ACT_DIM)
        for j in range(ACT_DIM):
            self.da.data[j] = Scalar[DT](a_da[j])
        self.da.upload(ctx)
        self.hist_t = Tensor.alloc(N_ENVS * HW1)
        self.hist_t.upload(ctx)
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
        mut self, ctx: DeviceContext, env_qpos: DeviceBuffer[DT],
        env_qvel: DeviceBuffer[DT], base_row: Int, ref hist: List[Float64],
        ref lead: List[Float64],
    ) raises:
        """`hist`: the lanes' action histories ([N_ENVS, HIST_WORDS], the
        PPO driver's `_hist_push` layout; empty without TASK_PPO_ACT_HIST)."""
        comptime if EXW > 0:
            # per lane: its HIST_WORDS history words, then its TARGET_OBS
            # target leads (`lead`: [N_ENVS, TARGET_OBS], target - q)
            for e in range(N_ENVS):
                for k in range(HIST_WORDS):
                    self.hist_t.data[e * HW1 + k] = Scalar[DT](hist[e * HIST_WORDS + k])
                for k in range(TARGET_OBS):
                    self.hist_t.data[e * HW1 + HIST_WORDS + k] = Scalar[DT](
                        lead[e * TARGET_OBS + k]
                    )
            self.hist_t.upload_resident(ctx)
        ctx.enqueue_copy(self.rd.qpos.dev.value(), env_qpos)
        ctx.enqueue_copy(self.rd.qvel.dev.value(), env_qvel)
        forward_kinematics["gpu", RIG_DT, TOWER_MD, N_ENVS](
            self.rd, self.rm, Optional(ctx)
        )
        comptime n_img = (N_ENVS * 3 * PLANE + TPB - 1) // TPB
        for k in range(len(self.cams)):
            if N_CAMS == 2 and k == 0:
                self.r_o.render(ctx, self.rd, self.rm, self.cams[k])
                # ⚠ ONE non-owning view per kernel (two miscompile on Metal —
                # `Tensor.view_gpu`); the ring is an owned Tensor.
                var rgb = Tensor.view_gpu(
                    ctx, mptr(self.r_o.rgb.unsafe_ptr()),
                    N_ENVS * OVERHEAD_RENDER_W * OVERHEAD_RENDER_H * 3,
                )
                ctx.enqueue_function[_pack_camera_kernel[
                    N_ENVS, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, OBS_PX
                ]](
                    rgb.lt["gpu", Layout.row_major(
                        N_ENVS * OVERHEAD_RENDER_W * OVERHEAD_RENDER_H * 3
                    )](),
                    self.ring.lt["gpu", Layout.row_major(1)](),
                    Int64(base_row),
                    Int64(3 * k),
                    Scalar[DT](self.win[4 * k]), Scalar[DT](self.win[4 * k + 1]),
                    Scalar[DT](self.win[4 * k + 2]), Scalar[DT](self.win[4 * k + 3]),
                    grid_dim=n_img,
                    block_dim=TPB,
                )
            else:
                self.r.render(ctx, self.rd, self.rm, self.cams[k])
                var rgb = Tensor.view_gpu(
                    ctx, mptr(self.r.rgb.unsafe_ptr()),
                    N_ENVS * RENDER * RENDER * 3,
                )
                ctx.enqueue_function[_pack_camera_kernel[
                    N_ENVS, RENDER, RENDER, OBS_PX
                ]](
                    rgb.lt["gpu", Layout.row_major(N_ENVS * RENDER * RENDER * 3)](),
                    self.ring.lt["gpu", Layout.row_major(1)](),
                    Int64(base_row),
                    Int64(3 * k),
                    Scalar[DT](self.win[4 * k]), Scalar[DT](self.win[4 * k + 1]),
                    Scalar[DT](self.win[4 * k + 2]), Scalar[DT](self.win[4 * k + 3]),
                    grid_dim=n_img,
                    block_dim=TPB,
                )
        comptime n_q = (N_ENVS * PROPRIO + TPB - 1) // TPB
        ctx.enqueue_function[_pack_proprio_kernel[N_ENVS, NQ, NV]](
            self.rd.qpos.lt["gpu", Layout.row_major(N_ENVS * NQ)](),
            self.rd.qvel.lt["gpu", Layout.row_major(N_ENVS * NV)](),
            self.qa.lt["gpu", Layout.row_major(ACT_DIM)](),
            self.da.lt["gpu", Layout.row_major(ACT_DIM)](),
            self.hist_t.lt["gpu", Layout.row_major(N_ENVS * HW1)](),
            self.ring.lt["gpu", Layout.row_major(1)](),
            Int64(base_row),
            grid_dim=n_q,
            block_dim=TPB,
        )

    def redraw(mut self, ctx: DeviceContext, draw: Int) raises:
        """Draw `draw` of the look, on both cameras' renderers."""
        self.r.background = self.dr_w.apply(draw, self.r.vis, self.rm)
        self.dr_w.upload(ctx, self.r.vis, self.rm)
        self.r_o.background = self.dr_o.apply(draw, self.r_o.vis, self.rm)
        self.dr_o.upload(ctx, self.r_o.vis, self.rm)

    def restore(mut self, ctx: DeviceContext) raises:
        """The base (calibrated) look back on both renderers."""
        self.r.background = self.clean_w.apply(0, self.r.vis, self.rm)
        self.clean_w.upload(ctx, self.r.vis, self.rm)
        self.r_o.background = self.clean_o.apply(0, self.r_o.vis, self.rm)
        self.clean_o.upload(ctx, self.r_o.vis, self.rm)

    def draw_aug(mut self, ctx: DeviceContext) raises:
        """Fresh per-sample photometric jitter + shift for the next TRAINING
        batch: brightness U(+-0.1), contrast U(0.8, 1.2), per-channel gain
        U(0.9, 1.1), shift in {-1, 0, 1} per axis."""
        for b in range(BATCH):
            var o = b * AUG_WORDS
            self.aug.data[o] = Scalar[DT](random_float64(-0.1, 0.1))
            self.aug.data[o + 1] = Scalar[DT](random_float64(0.8, 1.2))
            for c in range(3):
                self.aug.data[o + 2 + c] = Scalar[DT](random_float64(0.9, 1.1))
            self.aug.data[o + 5] = Scalar[DT](Int(random_ui64(0, 2)) - 1)
            self.aug.data[o + 6] = Scalar[DT](Int(random_ui64(0, 2)) - 1)
        self.aug.upload_resident(ctx)

    def gather[B: Int](
        mut self, ctx: DeviceContext, mut g: Tensor, mut x: Tensor,
        augment: Bool = False,
    ) raises:
        comptime n = (B * IN_DIM + TPB - 1) // TPB
        ctx.enqueue_function[_gather_kernel[B]](
            self.ring.lt["gpu", Layout.row_major(1)](),
            g.lt["gpu", Layout.row_major(B)](),
            x.lt["gpu", Layout.row_major(B * IN_DIM)](),
            self.aug.lt["gpu", Layout.row_major(B * AUG_WORDS)](),
            Int64(1 if self.blank else 0),
            Int64(1 if (augment and self.aug_on and B == BATCH) else 0),
            grid_dim=n,
            block_dim=TPB,
        )


def run_pixel_dagger(args: List[String], driver: String) raises:
    comptime assert ACT == ACT_DIM, "the student acts in the teacher's space"
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
    var eval_teacher = _arg(args, "--eval-teacher", "0") == "1"
    # `--act-gain k`: the student's output x k before the clamp, in the EVAL
    # only. A regression onto sign-noise labels learns their small mean; if
    # the student is merely too TIMID, a gain restores it without retraining.
    var act_gain = Float64(_arg(args, "--act-gain", "1"))
    # domain randomisation: the rig's look redrawn every `--dr-every` control
    # steps (`off` | `light` | `full`); `--aug 1` the per-sample photometric
    # jitter + shift in the training batches; the eval is CLEAN (the
    # calibrated look) unless `--eval-dr 1`
    var dr_name = _arg(args, "--dr", "off")
    var dr_every = max(Int(_arg(args, "--dr-every", "4")), 1)
    var dr_seed = Int(_arg(args, "--dr-seed", "11"))
    var use_aug = _arg(args, "--aug", "0") == "1"
    var eval_dr = _arg(args, "--eval-dr", "0") == "1"
    # the real servos' response, as in the PPO driver (`--lag-tau lo,hi` ms,
    # `--lag-delay lo,hi` ticks); the student is trained AND evaluated under it
    var lag_tau = _arg(args, "--lag-tau", "")
    var lag_delay = _arg(args, "--lag-delay", "")
    # the real arm's speed cap and elbow stop (`ServoLag.set_limits`)
    var lag_vmax = _arg(args, "--lag-vmax", "")
    var elbow_max = Float64(_arg(args, "--elbow-max", "0"))
    # per-joint servo dynamics, offsets, control-period jitter, as in the PPO
    # driver (`ServoLag.set_per_joint` / `set_offset` / `set_period`):
    # "lo,hi;..." x 6 joints (offsets in rad), "lo,hi" ms
    var lag_tau_j = _arg(args, "--lag-tau-j", "")
    var lag_delay_j = _arg(args, "--lag-delay-j", "")
    var lag_vmax_j = _arg(args, "--lag-vmax-j", "")
    var lag_off_j = _arg(args, "--lag-offset-j", "")
    var lag_period = _arg(args, "--lag-period", "")
    # ⚠ THE TEACHER'S SCALES: its labels are actions in its own units, so the
    # student must execute them with the same (checked below against the
    # teacher's run config when it recorded them)
    var d_arm = Float64(_arg(args, "--delta-arm", String(DELTA_ARM)))
    var d_grip = Float64(_arg(args, "--delta-gripper", String(DELTA_GRIPPER)))
    # the teacher's cadence (`ppo_family_driver --repeat`): it acts every
    # `repeat` ticks, targets held between — the student, its labels and its
    # replay rows are on the same cadence
    var repeat = max(Int(_arg(args, "--repeat", "1")), 1)
    if C.MAX_STEPS % repeat != 0:
        raise Error("pixel dagger: --repeat must divide the horizon")
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
    var jdadr = List[Int]()
    var dacc = 0
    for i in range(len(fmd.joints)):
        jdadr.append(dacc)
        dacc += fmd.joints[i].nv
    var a_qa = List[Int]()
    var a_da = List[Int]()
    var a_lo = List[Float64]()
    var a_hi = List[Float64]()
    for i in range(ACT_DIM):
        a_qa.append(jadr[fmd.actuators[i].joint_id])
        a_da.append(jdadr[fmd.actuators[i].joint_id])
        a_lo.append(fmd.actuators[i].ctrl_min)
        a_hi.append(fmd.actuators[i].ctrl_max)

    var run = RunContext(
        project=String("so101-tower"), driver=driver,
        slug=String("eval-dagger-px-" if total_steps == 0 else "dagger-px-") + task,
        env=String("family:") + family,
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
    logger.set_config("proprio", String("q+qd" if PROPRIO_STATE > ACT else "q"))
    logger.set_config("act_hist", String(HIST_WORDS // ACT_DIM))
    logger.set_config("window", String("workspace" if WINDOWED else "centre-square"))
    logger.set_config("dr", dr_name)
    logger.set_config("dr_every", String(dr_every))
    logger.set_config("aug", String(use_aug))
    logger.set_config("lag_tau_ms", lag_tau)
    logger.set_config("lag_delay_ticks", lag_delay)
    logger.set_config("lag_tau_j", lag_tau_j)
    logger.set_config("lag_delay_j", lag_delay_j)
    logger.set_config("lag_vmax_j", lag_vmax_j)
    logger.set_config("lag_offset_j", lag_off_j)
    logger.set_config("lag_period_ms", lag_period)
    logger.set_config("delta_arm", String(d_arm))
    logger.set_config("delta_gripper", String(d_grip))
    logger.set_config("gripper_sign", String(grip_sign))
    logger.set_config("act_gain", String(act_gain))
    logger.set_config("repeat", String(repeat))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    with DeviceContext() as ctx:
        # ── the teacher: the PPO actor, frozen, and its obs statistics ───
        var teacher = AgentT[T_OBS](
            ctx=ctx, actor_lr=Scalar[DT](0.0), critic_lr=Scalar[DT](0.0),
            gamma=Scalar[DT](GAMMA), gae_lambda=Scalar[DT](0.95),
            clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.0),
            action_scale=Scalar[DT](1.0), log_std_init=Scalar[DT](-1.0),
            window_size=100, initial_episode_fill=Scalar[DT](0.0),
            max_grad_norm=Scalar[DT](0.5),
        )
        teacher.trainer.load_state(teacher_dir + "/checkpoints/last.ckpt")
        var t_cfg = teacher_dir + "/metrics.config.kv"
        var t_action = String("delta")
        var target_lead = 0.0
        try:
            with open(t_cfg, "r") as fh:
                for ln in fh.read().split("\n"):
                    var sl = String(ln)
                    # ⚠ a `--action target` teacher's actions are steps
                    # from its PREVIOUS target (`delta_action.target_step`):
                    # executed here the same way, with the same lead bound,
                    # and only by a TARGET_OBS build (its input has the leads)
                    if sl.startswith("action="):
                        t_action = String(sl[byte = 7 :])
                    if sl.startswith("target_lead="):
                        target_lead = Float64(String(sl[byte = 12 :]))
                    if sl.startswith("delta_arm="):
                        var tv = Float64(String(sl[byte = 10 :]))
                        if abs(tv - d_arm) > 1e-9:
                            raise Error("pixel dagger: the teacher acts with"
                                        " --delta-arm " + String(tv)
                                        + ", this run with " + String(d_arm))
        except e:
            if String(e).find("pixel dagger:") >= 0:
                raise e^
        if (t_action == "target") != (TARGET_OBS > 0):
            raise Error("pixel dagger: the teacher acts with --action "
                        + t_action + "; a target teacher needs (and only it"
                        + " may use) a -D TASK_PPO_TARGET_OBS build")
        print("  teacher action", t_action, "| target lead", target_lead)
        var obs_rms = RunningMeanStd(T_OBS)
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
                Int(random_float64() * Float64(C.MAX_STEPS // repeat)) * repeat
            )
        env.d.meta.upload(ctx)
        ctx.synchronize()

        var px = PixelObs(ctx, scene_path(f), a_qa, a_da, cap, dr_name, dr_seed)
        px.blank = blank
        px.aug_on = use_aug
        var dr_draw = 0
        if px.dr_on or use_aug:
            print("  domain randomisation:", dr_name, "every", dr_every,
                  "steps | per-sample aug:", use_aug, "| eval",
                  "randomised" if eval_dr else "clean")
        if blank:
            print("  ⚠ --blank-images 1: the student sees ZERO image planes"
                  " (the joints-only control)")

        # ── buffers ──────────────────────────────────────────────────────
        var raw_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
        var cur_n = ctx.enqueue_create_host_buffer[DT](N_ENVS * T_OBS)
        var aug_o = List[Scalar[DT]](length=N_ENVS * T_OBS, fill=Scalar[DT](0))
        # the lanes' executed actions, most recent first (zero at a reset) —
        # the teacher's extra words and the student's extra planes
        var hist = List[Float64](length=N_ENVS * HIST_WORDS, fill=0.0)
        # target mode: each lane's last commanded target, and its lead over
        # the joints at the observation (the student's lead planes)
        var tprev = List[Float64](length=N_ENVS * ACT_DIM, fill=0.0)
        var lead = List[Float64](length=N_ENVS * TARGET_OBS, fill=0.0)
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
        var qvel_dev = DeviceBuffer[DT](
            ctx, env.d.qvel.dev.value().unsafe_ptr(), N_ENVS * NV, owning=False
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
        # the policy step's held targets; per lane, whether it ended in them
        var tg = List[Float64](length=N_ENVS * ACT_DIM, fill=0.0)
        var dmac = List[Bool](length=N_ENVS, fill=False)
        var lag = ServoLag.parse(
            N_ENVS, lag_tau, lag_delay, Float64(C.FRAME_SKIP) * M.TIMESTEP
        )
        lag.set_limits(lag_vmax, elbow_max)
        lag.set_per_joint(lag_tau_j, lag_delay_j, lag_vmax_j)
        lag.set_offset(lag_off_j)
        lag.set_period(lag_period)
        var lag_pending = List[Bool](length=N_ENVS, fill=True)
        if lag.on:
            print("  servo lag: tau", lag_tau, "ms, delay", lag_delay,
                  "ticks | arm speed cap", lag_vmax, "rad/s | elbow max", elbow_max)
        if lag.per_joint or lag.has_off or lag.dt_hi > 0.0:
            print("  servo per joint: tau", lag_tau_j, "ms | delay", lag_delay_j,
                  "| cap", lag_vmax_j, "rad/s | offset", lag_off_j,
                  "rad | period", lag_period, "ms")
        var succ = List[Bool](length=N_ENVS, fill=False)
        var hist_s = List[Bool]()  # student-executed episodes' success
        var hist_t = List[Bool]()  # teacher-executed
        var n_filled = 0
        var step = 0
        var it = 0
        var loss_acc = 0.0
        var loss_n = 0
        var t0 = perf_counter_ns()
        # ⚠ `checkpoints/last.ckpt` + `checkpoints/norm.json` (the student's
        # manifest): the layout `project-promote <run> last --as <role>`
        # takes, so a student reaches the real-arm deploy as a ROLE.
        makedirs(run.dir + "/checkpoints", exist_ok=True)
        var ckpt = run.dir + "/checkpoints/last.ckpt"
        write_pixel_manifest(
            run.dir + "/checkpoints/norm.json", task, teacher_dir, grip_sign,
            Float64(C.FRAME_SKIP) * M.TIMESTEP, d_arm, d_grip, lag_tau, lag_delay, repeat,
            t_action, target_lead,
        )

        ctx.enqueue_copy(raw_h, obs_dev)
        ctx.synchronize()
        if png_dir.byte_length() > 0:
            _dump_obs_png(ctx, px, png_dir, qpos_dev, qvel_dev, hist)

        while step < total_steps:
            var beta = 1.0 - Float64(step) / Float64(max(beta_steps, 1))
            if beta < 0.0:
                beta = 0.0
            var base = (it % (cap // N_ENVS)) * N_ENVS
            # 1-2. the pictures and joints of the CURRENT state (under a new
            # draw of the look every `dr_every` steps)
            if px.dr_on and it % dr_every == 0:
                px.redraw(ctx, dr_draw)
                dr_draw += 1
            # the joints, and a new episode's servo model and target
            var rp = mptr(raw_h.unsafe_ptr())
            for e in range(N_ENVS):
                for j in range(ACT_DIM):
                    arm_q[e * ACT_DIM + j] = Float64(
                        rp[unsafe_offset = e * OBS + a_qa[j]]
                    )
                if lag_pending[e]:
                    _lag_reset(lag, arm_q, e)
                    _target_reset(tprev, arm_q, e)
                    lag_pending[e] = False
                for j in range(TARGET_OBS):
                    lead[e * TARGET_OBS + j] = (
                        tprev[e * ACT_DIM + j] - arm_q[e * ACT_DIM + j]
                    )
            px.observe(ctx, qpos_dev, qvel_dev, base, hist, lead)
            # 3. the teacher, on the state
            _augment[OBS](rp, hist, tprev, a_qa, mptr(aug_o.unsafe_ptr()))
            obs_rms.normalize_into(
                mptr(aug_o.unsafe_ptr()), mptr(cur_n.unsafe_ptr()), N_ENVS,
                T_OBS, OBS_CLIP,
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
                            at[unsafe_offset = e * ACT_DIM + j] = student_act(
                                y_act.data[e * ACT_DIM + j], j, grip_sign
                            )
            _hist_push(hist, mptr(act_t.unsafe_ptr()))
            if t_action == "target":
                _target_targets(
                    mptr(act_t.unsafe_ptr()), tg, tprev, arm_q, a_lo, a_hi,
                    d_arm, d_grip, target_lead,
                )
            else:
                _delta_targets(
                    mptr(act_t.unsafe_ptr()), tg, arm_q, a_lo, a_hi, d_arm, d_grip,
                )
            # 5. `repeat` ticks under the held targets, tally, reset
            for e in range(N_ENVS):
                dmac[e] = False
            for tick in range(repeat):
                _targets_to_env(tg, mptr(env_act.unsafe_ptr()), a_lo, a_hi, lag)
                ctx.enqueue_copy(act_dev, env_act)
                env.step_batch[N_ENVS](
                    ctx=ctx, rng_seed=UInt64(it * repeat + tick + 1)
                )
                ctx.enqueue_copy(raw_h, obs_dev)
                ctx.enqueue_copy(done_h, done_dev)
                env.d.meta.download(ctx)
                ctx.synchronize()
                var rq_t = mptr(raw_h.unsafe_ptr())
                var dh_t = mptr(done_h.unsafe_ptr())
                for e in range(N_ENVS):
                    if dmac[e]:
                        continue
                    for k in range(OBS):
                        var v = Float64(rq_t[unsafe_offset = e * OBS + k])
                        if not (v == v) or abs(v) > OBS_BOUND:
                            dmac[e] = True
                            break
                    if env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD] > Scalar[DT](0.5):
                        succ[e] = True
                    if dh_t[unsafe_offset=e] > Scalar[DT](0.5):
                        dmac[e] = True
            var dh = mptr(done_h.unsafe_ptr())
            var forced = False
            for e in range(N_ENVS):
                var was = dh[unsafe_offset=e] > Scalar[DT](0.5)
                if dmac[e] != was:
                    forced = True
                dh[unsafe_offset=e] = Scalar[DT](1) if dmac[e] else Scalar[DT](0)
                if dh[unsafe_offset=e] > Scalar[DT](0.5):
                    if stud_lane[e]:
                        hist_s.append(succ[e])
                    else:
                        hist_t.append(succ[e])
                    succ[e] = False
                    stud_lane[e] = random_float64() >= beta
                    lag_pending[e] = True
                    _hist_clear(hist, e)
            if forced:
                ctx.enqueue_copy(done_dev, done_h)
            env.selective_reset_batch[N_ENVS](
                ctx=ctx, rng_seed=UInt64(seed * 7919 + it + 1)
            )
            ctx.enqueue_copy(raw_h, obs_dev)
            ctx.synchronize()
            step += N_ENVS * repeat
            it += 1

            # ── the regression, on the aggregate dataset ─────────────────
            if it % ITER_STEPS == 0:
                for _u in range(updates):
                    for b in range(BATCH):
                        g_tr.data[b] = Scalar[DT](
                            Int(random_ui64(0, UInt64(n_filled - 1)))
                        )
                    g_tr.upload_resident(ctx)
                    if use_aug:
                        px.draw_aug(ctx)
                    px.gather[BATCH](ctx, g_tr, x_tr, augment=True)
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
        # ⚠ THE EVAL'S LOOK: clean (the calibrated base) unless `--eval-dr 1`
        # — a trained student is scored on the picture the training was
        # randomised AROUND, and separately on held-out draws.
        if px.dr_on and not eval_dr:
            px.restore(ctx)
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
            # ⚠ WHERE THE BRICK COMES DOWN: the brick's horizontal gap to the
            # bowl at the step it first falls back (rise < 1 cm after having
            # been > 2 cm up) — "released over the bowl and bounced out" and
            # "released beside it" are different failures with different fixes.
            var was_up = List[Bool](length=N_ENVS, fill=False)
            var drop_h = List[Float64](length=N_ENVS, fill=-1.0)
            var dz_end = List[Float64](length=N_ENVS, fill=0.0)
            var h_end = List[Float64](length=N_ENVS, fill=0.0)
            # ⚠ WHAT THE TEACHER WOULD DO WHERE THE STUDENT HOVERS: in the
            # student's run, the teacher's greedy gripper word on the same
            # state (the DAgger label), at every step the brick is up and
            # over the bowl. Teacher "open" + student "closed" is a learning
            # failure; teacher "closed" too is a state the teacher would
            # change first (lower, centre) — a perception question.
            var hov = 0
            var hov_t_open = 0
            var hov_s_open = 0
            var hov_both = 0
            var lab = List[Scalar[DT]](length=N_ENVS * ACT_DIM, fill=Scalar[DT](0))
            # per joint over the hover steps: the teacher's and the student's
            # mean action, and their mean |difference|
            var hv_t = List[Float64](length=ACT_DIM, fill=0.0)
            var hv_s = List[Float64](length=ACT_DIM, fill=0.0)
            var hv_d = List[Float64](length=ACT_DIM, fill=0.0)
            for e in range(N_ENVS):
                _hist_clear(hist, e)
            for t in range(C.MAX_STEPS - 1):
                # the policy acts every `repeat` ticks (observe, label, act);
                # its targets are held between, the servo model per tick
                var at = mptr(act_t.unsafe_ptr())
                if t % repeat == 0:
                    if px.dr_on and eval_dr and t % dr_every == 0:
                        px.redraw(ctx, 5_000_000 + rnd * 1000 + t)  # held-out draws
                    var rq = mptr(raw_h.unsafe_ptr())
                    for e in range(N_ENVS):
                        for j in range(ACT_DIM):
                            arm_q[e * ACT_DIM + j] = Float64(
                                rq[unsafe_offset = e * OBS + a_qa[j]]
                            )
                        g_act.data[e] = Scalar[DT](e)
                        if t == 0:
                            _target_reset(tprev, arm_q, e)
                        for j in range(TARGET_OBS):
                            lead[e * TARGET_OBS + j] = (
                                tprev[e * ACT_DIM + j] - arm_q[e * ACT_DIM + j]
                            )
                    px.observe(ctx, qpos_dev, qvel_dev, 0, hist, lead)
                    if eval_teacher:
                        # the TEACHER through the same loop — the reference the
                        # student's stages are read against
                        _augment[OBS](rq, hist, tprev, a_qa, mptr(aug_o.unsafe_ptr()))
                        obs_rms.normalize_into(
                            mptr(aug_o.unsafe_ptr()), mptr(cur_n.unsafe_ptr()),
                            N_ENVS, T_OBS, OBS_CLIP,
                        )
                        teacher.trainer.select_greedy_action_batched(
                            mptr(cur_n.unsafe_ptr()), mptr(act_t.unsafe_ptr())
                        )
                    else:
                        _augment[OBS](rq, hist, tprev, a_qa, mptr(aug_o.unsafe_ptr()))
                        obs_rms.normalize_into(
                            mptr(aug_o.unsafe_ptr()), mptr(cur_n.unsafe_ptr()),
                            N_ENVS, T_OBS, OBS_CLIP,
                        )
                        teacher.trainer.select_greedy_action_batched(
                            mptr(cur_n.unsafe_ptr()), mptr(lab.unsafe_ptr())
                        )
                        g_act.upload_resident(ctx)
                        px.gather[N_ENVS](ctx, g_act, x_act)
                        student.forward["gpu", N_ENVS](
                            TensorRefs[1](x_act), y_act, Optional(ctx)
                        )
                        y_act.download(ctx)
                        ctx.synchronize()
                        for k in range(N_ENVS * ACT_DIM):
                            at[unsafe_offset=k] = student_act(
                                y_act.data[k] * Scalar[DT](act_gain), k % ACT_DIM,
                                grip_sign,
                            )
                    if t == 0:
                        for e in range(N_ENVS):
                            _lag_reset(lag, arm_q, e)
                    _hist_push(hist, mptr(act_t.unsafe_ptr()))
                    if t_action == "target":
                        _target_targets(
                            mptr(act_t.unsafe_ptr()), tg, tprev, arm_q, a_lo,
                            a_hi, d_arm, d_grip, target_lead,
                        )
                    else:
                        _delta_targets(
                            mptr(act_t.unsafe_ptr()), tg, arm_q, a_lo, a_hi,
                            d_arm, d_grip,
                        )
                _targets_to_env(tg, mptr(env_act.unsafe_ptr()), a_lo, a_hi, lag)
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
                    var hh = sqrt(ex * ex + ey * ey)
                    var dz = za - z0[e]
                    if not eval_teacher and dz > 0.02 and hh < 0.045:
                        # the state BEFORE this step's action: the labels and
                        # actions above were computed on it (a one-step lag
                        # in the hover test, harmless for a count)
                        hov += 1
                        var t_open = lab[e * ACT_DIM + ACT_DIM - 1] > Scalar[DT](0)
                        var s_open = at[unsafe_offset = e * ACT_DIM + ACT_DIM - 1] > Scalar[DT](0)
                        if t_open:
                            hov_t_open += 1
                        if s_open:
                            hov_s_open += 1
                        if t_open and s_open:
                            hov_both += 1
                        for j in range(ACT_DIM):
                            var tv = Float64(lab[e * ACT_DIM + j])
                            var sv = Float64(at[unsafe_offset = e * ACT_DIM + j])
                            hv_t[j] += tv
                            hv_s[j] += sv
                            hv_d[j] += abs(tv - sv)
                    if dz > 0.02 and hh < 0.045:
                        over[e] = True
                    if dz > 0.02:
                        was_up[e] = True
                    elif was_up[e] and dz < 0.01 and drop_h[e] < 0.0:
                        drop_h[e] = hh
                    dz_end[e] = dz
                    h_end[e] = hh
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
            var who = "teacher" if eval_teacher else "student"
            print("  " + who + " greedy eval round", rnd, ":", ok, "/", N_ENVS,
                  "| held at the end", n_end, "| lifted >2cm", n_lift,
                  "| over b while lifted", n_over)
            # the drops, by where they happened, and the endings of the rest
            var d_in = 0
            var d_rim = 0
            var d_out = 0
            var up_end = 0
            var rim_end = 0
            var beside_end = 0
            var else_end = 0
            for e in range(N_ENVS):
                if drop_h[e] >= 0.0:
                    if drop_h[e] < 0.045:
                        d_in += 1
                    elif drop_h[e] < 0.08:
                        d_rim += 1
                    else:
                        d_out += 1
                if held_end[e]:
                    continue
                if dz_end[e] > 0.02:
                    up_end += 1
                elif h_end[e] < 0.052:
                    rim_end += 1
                elif h_end[e] < 0.10:
                    beside_end += 1
                else:
                    else_end += 1
            if not eval_teacher:
                print("    hover steps (up, over the bowl):", hov,
                      "| teacher says OPEN", hov_t_open, "| student opens",
                      hov_s_open, "| both", hov_both)
                var lt = String("      teacher mean a:")
                var ls = String("      student mean a:")
                var ld = String("      mean |t - s|  :")
                var nh = Float64(max(hov, 1))
                for j in range(ACT_DIM):
                    lt += " " + String(Float64(Int(hv_t[j] / nh * 1000.0)) / 1000.0)
                    ls += " " + String(Float64(Int(hv_s[j] / nh * 1000.0)) / 1000.0)
                    ld += " " + String(Float64(Int(hv_d[j] / nh * 1000.0)) / 1000.0)
                print(lt)
                print(ls)
                print(ld)
            print("    drops: over the bowl (<4.5 cm)", d_in, "| at the rim (4.5-8)",
                  d_rim, "| away (>8)", d_out,
                  "|| not held at the end: brick up", up_end,
                  "| inside the rim, not Near", rim_end, "| beside (<10 cm)",
                  beside_end, "| elsewhere", else_end)
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
    qpos_dev: DeviceBuffer[DT], qvel_dev: DeviceBuffer[DT],
    ref hist: List[Float64],
) raises:
    """Lanes 0-1's first observation, each camera upscaled, as PNGs — to SEE
    what the student sees (a transposed or blank plane trains silently). With
    domain randomisation on, under the clean look and four draws
    (`_draw<d>`), so the randomisation's range can be judged by eye."""
    makedirs(dir, exist_ok=True)
    var n_draw = 5 if px.dr_on else 1
    for d in range(n_draw):
        if px.dr_on:
            if d == 0:
                px.restore(ctx)
            else:
                px.redraw(ctx, 900_000 + d)
        var lead0 = List[Float64](length=N_ENVS * TARGET_OBS, fill=0.0)
        px.observe(ctx, qpos_dev, qvel_dev, 0, hist, lead0)
        var h = ctx.enqueue_create_host_buffer[DT](2 * ROW)
        ctx.enqueue_copy(h, px.ring.dev.value().create_sub_buffer[DT](0, 2 * ROW))
        ctx.synchronize()
        var hp = h.unsafe_ptr()
        comptime S = 256 // OBS_PX
        comptime W = OBS_PX * S
        for lane in range(2):
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
                var tag = String("_clean") if (px.dr_on and d == 0) else (
                    String("_draw") + String(d) if px.dr_on else String("")
                )
                save_png(dir + "/lane" + String(lane) + "_cam" + String(cam)
                         + tag + ".png", img, W, W, 3)
    if px.dr_on:
        px.restore(ctx)
    print("  png: lanes 0-1 x", N_CAMS, "cameras x", n_draw, "looks ->", dir)


def _rate(h: List[Bool]) -> Float64:
    """Success over the last 1024 episodes of a history."""
    var n = len(h)
    var lo = n - 1024 if n > 1024 else 0
    var s = 0
    for k in range(lo, n):
        if h[k]:
            s += 1
    return Float64(s) / Float64(n - lo) if n - lo > 0 else 0.0
