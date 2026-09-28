"""The so101_tower vision student, closed loop — the library half of
`examples/so101/tower_act_eval.mojo` (the example parses the command line and
calls `TowerActEval.run`; the protocol, the success rule and the failure
buckets are documented there).

## ⚠⚠ WHY IT COMPILES NOW — AND WHAT DID NOT MATTER

The example was one 470-line `main` holding the batched env, the rig's
renderer and model, the randomizer and the ACT trainer, the trainer as an
`Optional[ACTTrainer[...]]`. It was OOM-killed at 57 GB of compiler memory on
a 62 GB box with 1, 4 or 16 compiler threads, while its parts compiled alone
in 1 GB (the env), 1 GB (the renderer) and 6 GB (ACT make + load + predict).
Bisected with `--mlir-timing` probes and an RSS sampler (5090 box,
2026-09-23): the trainer held in an `Optional` alone took ACT's compile from
6 to 14 GB and added a 31 s `ReorderParamOps` pass; the trainer's type
expression carries the whole loss graph, and the wrapper makes the compiler
process it again wherever it appears. Holding it directly (`act` below), this
module plus the thin example compile in 15 GB and 258 s, and reproduce the
old binary's result on the same seeds exactly (41/128, same buckets).

Moving the loop into `mut self` methods was tried FIRST, on the hypothesis
that each raising call's unwind path destroyed the heavy locals
(`COMPILE_TIME_PROFILING` §3.6): alone it changed nothing measurable (still
killed past 30 GB). The struct stays because it is the better shape — short
methods, one owner for the device objects — not because it fixed the build.
Do not wrap `act` (or another graph-typed field) in an `Optional`.
"""

from std.math import sqrt
from std.os import makedirs
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, HostBuffer

from noeira.nn.core.ptr import mptr
from noeira.deep_agents.act.config import (
    SO101_ADIM, SO101_IMG_H, SO101_IMG_W, SO101_N_CAM, SO101_QPOS,
    RUN_K, RUN_DIM, RUN_HEADS, RUN_FF, RUN_LATENT, RUN_ENC_LAYERS,
    RUN_DEC_LAYERS, ACT_TEMPORAL_ENSEMBLE_M,
)
from noeira.deep_agents.act.trainer import ACTTrainer
from noeira.deep_agents.act.norm_file import ACTNorm
from noeira.deep_agents.act.inference import (
    TemporalEnsemble, normalize_camera_chw, denormalize,
)
from noeira.deep_agents.demos.file import DemoSet, write_demo_file
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.io.png import save_png
from noeira.math3d import Vec3 as Vec3Generic
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_GOAL_HELD, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.physics3d.raytrace.randomize import (
    DomainRandConfig, VisualRandomizer, geom_labels,
    so101_tower_surface_groups,
)
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig, So101TowerTeleopConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, RIG_CAM_W, RIG_CAM_H, RIG_N_CAMS, RIG_NPIX,
    RIG_CAM_ELEMS, RIG_IMG_ELEMS, RIG_ACT, RIG_DR_TARGET, RIG_JOINT_ZERO_NONE,
    TowerRenderer, make_tower_model, make_tower_renderer, tower_cameras,
    pack_camera_u8, So101TowerUnits,
    RIG_LOOK_CALIBRATED, scale_tower_camera_dr,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family


comptime TOWER_EVAL_HOLD_STEPS = 31
"""`tower_expert_record.mojo`'s success rule: the goal held this many steps."""
comptime TOWER_EVAL_SETTLE_STEPS = 5
comptime TOWER_EVAL_DEFAULT_SEED0 = 30000
comptime TOWER_EVAL_DEFAULT_STEPS = 600
"""19 s at 31.25 Hz. The expert's cube-in-bowl episode is ~236 steps + the
hold; a student is allowed 2.5x that."""
comptime TOWER_EVAL_LIFT_DZ = 0.03
comptime TOWER_EVAL_LAND_DZ = 0.01
comptime TOWER_EVAL_NEAR_BOWL = 0.08
comptime _FAMILY = "so101_tower"
comptime _FAMILY_PATH = "noeira/tasks/families/so101_tower.family"


def _clamp1(x: Float64) -> Float64:
    return 1.0 if x > 1.0 else (-1.0 if x < -1.0 else x)


def tower_eval_pct(n: Int, d: Int) -> String:
    if d == 0:
        return String("-")
    return String(Float64(Int(1000.0 * Float64(n) / Float64(d) + 0.5)) / 10.0) + "%"


struct TowerEvalConfig(Copyable, Movable):
    """What one evaluation run is. Paths already resolved (the example turns
    `--ckpt RUN_ID|DIR|FILE` into `ckpt_path` + `norm_path`)."""

    var use_act: Bool
    var ckpt_path: String
    var norm_path: String
    var n_episodes: Int
    var seed0: Int
    var steps: Int
    var exec_n: Int
    """0 = temporal ensemble (`ens_m`), N = open-loop chunks of N steps."""
    var ens_m: Float64
    var task: String
    var dr_name: String
    var dr_seed: Int
    var dr_draw0: Int
    var demo_out: String
    var snap_dir: String
    var joint_zero: String
    var look: String
    """The rig's look (`so101_tower_rig.apply_tower_look`) — the training
    store's, recorded in its provenance line."""

    def __init__(out self):
        self.use_act = True
        self.ckpt_path = String("")
        self.norm_path = String("")
        self.n_episodes = 128
        self.seed0 = TOWER_EVAL_DEFAULT_SEED0
        self.steps = TOWER_EVAL_DEFAULT_STEPS
        self.exec_n = 0
        self.ens_m = ACT_TEMPORAL_ENSEMBLE_M
        self.task = String("so101_tower_cube_in_bowl")
        self.dr_name = String("off")
        self.dr_seed = 0
        self.dr_draw0 = 0
        self.demo_out = String("")
        self.snap_dir = String("")
        self.joint_zero = String(RIG_JOINT_ZERO_NONE)
        self.look = String(RIG_LOOK_CALIBRATED)


struct TowerEvalTally(Copyable, Movable):
    """The outcome buckets and the counters, summed over rounds."""

    var n: Int
    var ok: Int
    var no_grasp: Int
    var dropped: Int
    var missed: Int
    var goal_not_held: Int
    var held_end: Int
    var steps_to_success: Float64
    var act_words: Int
    var saturated: Int
    var nonfinite: Int
    var ns_render: Int
    var ns_forward: Int
    var ns_physics: Int

    def __init__(out self):
        self.n = 0
        self.ok = 0
        self.no_grasp = 0
        self.dropped = 0
        self.missed = 0
        self.goal_not_held = 0
        self.held_end = 0
        self.steps_to_success = 0.0
        self.act_words = 0
        self.saturated = 0
        self.nonfinite = 0
        self.ns_render = 0
        self.ns_forward = 0
        self.ns_physics = 0

    def report(self, use_act: Bool):
        var n = self.n
        print("-" * 78)
        print("  SUCCESS        ", self.ok, "/", n, "=", tower_eval_pct(self.ok, n),
              ("| mean steps " + String(Int(self.steps_to_success / Float64(self.ok))))
              if self.ok > 0 else String(""))
        print("  no grasp       ", self.no_grasp, "(", tower_eval_pct(self.no_grasp, n), ")")
        print("  dropped        ", self.dropped, "(", tower_eval_pct(self.dropped, n),
              ") — landed >", TOWER_EVAL_NEAR_BOWL, "m from the bowl")
        print("  missed bowl    ", self.missed, "(", tower_eval_pct(self.missed, n),
              ") — landed near it")
        print("  goal not held  ", self.goal_not_held, "(",
              tower_eval_pct(self.goal_not_held, n), ")")
        print("  held to end    ", self.held_end, "(", tower_eval_pct(self.held_end, n), ")")
        if use_act:
            print("  actions        ", self.act_words, "words |", self.saturated,
                  "saturated (", tower_eval_pct(self.saturated, self.act_words), ") |",
                  self.nonfinite, "non-finite")
            print("  time           render", Float64(self.ns_render) / 1e9,
                  "s | forward", Float64(self.ns_forward) / 1e9, "s | physics",
                  Float64(self.ns_physics) / 1e9, "s")

    def check(self) raises:
        var s = (self.ok + self.no_grasp + self.dropped + self.missed
                 + self.goal_not_held + self.held_end)
        if s != self.n:
            raise Error("tower act eval: the outcome buckets do not sum to the episodes")
        if self.nonfinite > 0:
            raise Error("tower act eval: " + String(self.nonfinite)
                        + " non-finite action words")


struct TowerActEval[LANES: Int](Movable):
    """`LANES` episodes per round on the batched GPU env."""

    comptime DT = RIG_DT
    comptime CFG = So101TowerTeleopConfig
    """The recorder's config (the tower's, with a 1200-step horizon), so the
    env never truncates inside `steps`."""
    comptime E = Phyics3dBatchedEnv[So101TowerModel, Self.CFG, Self.LANES]
    comptime NQ = So101TowerModel.NQ
    comptime NV = So101TowerModel.NV
    comptime OD = Self.E.OBS_DIM
    comptime AD = Self.E.ACT_DIM
    comptime QPOS = SO101_QPOS
    comptime ADIM = SO101_ADIM
    comptime K = RUN_K
    comptime T = ACTTrainer[
        SO101_QPOS, SO101_ADIM, SO101_N_CAM, SO101_IMG_H, SO101_IMG_W, RUN_K,
        RUN_DIM, RUN_HEADS, RUN_FF, RUN_LATENT, RUN_ENC_LAYERS, RUN_DEC_LAYERS,
        Self.LANES, target="gpu",
    ]
    comptime Vec3 = Vec3Generic[Self.DT]

    var cfg: TowerEvalConfig
    var ctx: DeviceContext
    var n_rounds: Int
    var brick: Int
    var bowl: Int
    var meta_idx: List[Int]
    var meta_val: List[Float64]
    var units: So101TowerUnits
    var cams: List[Int]
    var norm: Optional[ACTNorm]
    # ── the heavy fields, built last ─────────────────────────────────────
    var env: Self.E
    var rm: Model[RIG_DT, TOWER_MD]
    var rd: Data[RIG_DT, TOWER_MD, Self.LANES]
    var r: TowerRenderer[Self.LANES]
    var dr_cfg: DomainRandConfig
    var dr: VisualRandomizer[RIG_DT]
    var act: Self.T
    """⚠ NOT `Optional[Self.T]`. The trainer's type expression carries the
    whole loss graph; wrapping it in an `Optional` added a 31 s
    `ReorderParamOps` pass and took ACT make + load + predict alone from 6 to
    14 GB of compiler memory (5090 box, 2026-09-23), and the full eval, which
    held it that way, did not compile in 62 GB. So it is always built (about
    200 MB of device memory) and loaded only when a checkpoint is given."""
    # ── buffers ──────────────────────────────────────────────────────────
    var h_rgb: HostBuffer[RIG_DT]
    var act_h: HostBuffer[RIG_DT]
    var obs_h: HostBuffer[RIG_DT]
    var rew_h: HostBuffer[RIG_DT]
    var img_u8: List[Scalar[DType.uint8]]
    var qpos_n: List[Scalar[RIG_DT]]
    var images_n: List[Scalar[RIG_DT]]
    var dummy_a: List[Scalar[RIG_DT]]
    var dummy_v: List[Scalar[RIG_DT]]
    var chunk: List[Scalar[RIG_DT]]
    var pred_n: List[Scalar[RIG_DT]]
    var pred: List[Scalar[RIG_DT]]
    var ens: List[TemporalEnsemble[SO101_ADIM, RUN_K]]
    # ── per-round state ──────────────────────────────────────────────────
    var rest_z: List[Float64]
    var held: List[Int]
    var done: List[Bool]
    var succ_step: List[Int]
    var lifted: List[Bool]
    var land_d: List[Float64]
    var goal_ever: List[Bool]
    var rows_obs: List[List[Float32]]
    var rows_act: List[List[Float32]]
    var rows_rew: List[List[Float32]]
    var rows_nobs: List[List[Float32]]
    var t_query: Int
    var demos: DemoSet
    var tally: TowerEvalTally

    def __init__(out self, var cfg: TowerEvalConfig, ctx: DeviceContext) raises:
        comptime assert Self.AD == RIG_ACT, "the env's action is the six joint targets"
        comptime assert SO101_N_CAM == RIG_N_CAMS
        comptime assert SO101_IMG_H == RIG_CAM_H and SO101_IMG_W == RIG_CAM_W
        # ── the cheap, raising setup: locals only ────────────────────────
        var f = load_family(String(_FAMILY_PATH))
        var fmd = parse_model_runtime(scene_path(f))
        var rsites = region_sites(f, fmd.site_names)
        var rects = region_rects(f)
        var rheights = region_half_heights(f)
        var cw = region_table_words(
            rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
            rheights[0],
        )
        var mw = task_meta_words(
            cfg.task, String(_FAMILY), Self.CFG.SHAPE_W_GOAL,
            Self.CFG.SHAPE_W_REACH, Self.CFG.GOAL_MARGIN, Self.CFG.REACH_MARGIN,
        )
        var brick = -1
        var bowl = -1
        for b in range(len(fmd.body_names)):
            if String(fmd.body_names[b]) == "brick_brick":
                brick = b
            if String(fmd.body_names[b]) == "bowl_bowl":
                bowl = b
        if brick < 0 or bowl < 0:
            raise Error("brick_brick / bowl_bowl not found in the composed scene")
        var units = So101TowerUnits(cfg.joint_zero)
        var cams = tower_cameras(fmd)
        var labels = geom_labels(fmd)
        var dr_cfg = DomainRandConfig.parse(cfg.dr_name, UInt64(cfg.dr_seed))
        var norm = Optional[ACTNorm](None)
        if cfg.use_act:
            norm = ACTNorm.load(cfg.norm_path, Self.QPOS, Self.ADIM)

        # ── the heavy fields ─────────────────────────────────────────────
        self.ctx = ctx
        self.env = Self.E(ctx)
        for k in range(MODEL_CURRICULUM_SIZE):
            self.env.mf.curriculum.data[k] = Scalar[Self.DT](cw[k])
        self.env.mf.curriculum.upload(ctx)
        self.rm = make_tower_model(ctx)
        self.rd = Data[RIG_DT, TOWER_MD, Self.LANES]()
        self.rd.upload_all(ctx)
        self.r = make_tower_renderer[Self.LANES](ctx, fmd, self.rm, cfg.look)
        self.dr = VisualRandomizer[RIG_DT](
            dr_cfg, so101_tower_surface_groups(), self.r.vis, self.rm, labels,
            cams.copy(), self.r.background, RIG_DR_TARGET,
        )
        scale_tower_camera_dr(self.dr, fmd)
        self.act = Self.T.make(ctx=Optional[DeviceContext](ctx))
        if cfg.use_act:
            self.act.load(cfg.ckpt_path)

        # ── the rest: no raising calls below this line ───────────────────
        self.n_rounds = (cfg.n_episodes + Self.LANES - 1) // Self.LANES
        self.brick = brick
        self.bowl = bowl
        self.meta_idx = mw[0].copy()
        self.meta_val = List[Float64]()
        for k in range(len(mw[1])):
            self.meta_val.append(Float64(mw[1][k]))
        self.units = units^
        self.cams = cams^
        self.norm = norm^
        self.dr_cfg = dr_cfg^
        self.h_rgb = ctx.enqueue_create_host_buffer[RIG_DT](Self.LANES * RIG_NPIX * 3)
        self.act_h = ctx.enqueue_create_host_buffer[RIG_DT](Self.LANES * Self.AD)
        self.obs_h = ctx.enqueue_create_host_buffer[RIG_DT](Self.LANES * Self.OD)
        self.rew_h = ctx.enqueue_create_host_buffer[RIG_DT](Self.LANES)
        self.img_u8 = List[Scalar[DType.uint8]](length=Self.LANES * RIG_IMG_ELEMS, fill=0)
        self.qpos_n = List[Scalar[RIG_DT]](length=Self.LANES * Self.QPOS, fill=0)
        self.images_n = List[Scalar[RIG_DT]](length=Self.LANES * RIG_IMG_ELEMS, fill=0)
        self.dummy_a = List[Scalar[RIG_DT]](length=Self.LANES * Self.K * Self.ADIM, fill=0)
        self.dummy_v = List[Scalar[RIG_DT]](length=Self.LANES * Self.K, fill=1)
        self.chunk = List[Scalar[RIG_DT]](length=Self.LANES * Self.K * Self.ADIM, fill=0)
        self.pred_n = List[Scalar[RIG_DT]](length=Self.ADIM, fill=0)
        self.pred = List[Scalar[RIG_DT]](length=Self.ADIM, fill=0)
        self.ens = List[TemporalEnsemble[SO101_ADIM, RUN_K]]()
        for _ in range(Self.LANES):
            self.ens.append(TemporalEnsemble[SO101_ADIM, RUN_K](m=cfg.ens_m))
        self.rest_z = List[Float64](length=Self.LANES, fill=0.0)
        self.held = List[Int](length=Self.LANES, fill=0)
        self.done = List[Bool](length=Self.LANES, fill=False)
        self.succ_step = List[Int](length=Self.LANES, fill=-1)
        self.lifted = List[Bool](length=Self.LANES, fill=False)
        self.land_d = List[Float64](length=Self.LANES, fill=-1.0)
        self.goal_ever = List[Bool](length=Self.LANES, fill=False)
        self.rows_obs = List[List[Float32]]()
        self.rows_act = List[List[Float32]]()
        self.rows_rew = List[List[Float32]]()
        self.rows_nobs = List[List[Float32]]()
        self.t_query = 0
        self.demos = DemoSet(Self.OD, Self.AD)
        self.tally = TowerEvalTally()
        self.cfg = cfg^

    def describe(self) -> String:
        return (
            String("  lanes  : ") + String(Self.LANES) + " | " + String(self.n_rounds)
            + " rounds = " + String(self.n_rounds * Self.LANES) + " episodes | seeds "
            + String(self.cfg.seed0) + " .. "
            + String(self.cfg.seed0 + self.n_rounds * Self.LANES - 1) + " | steps "
            + String(self.cfg.steps) + " | success = goal held "
            + String(TOWER_EVAL_HOLD_STEPS) + "\n  dr     : " + String(self.dr_cfg)
            + " (one draw per round, from " + String(self.cfg.dr_draw0) + ")"
            + "\n  units  : " + self.units.describe()
            + " — must be the training store's"
            + "\n  look   : " + self.cfg.look + " — must be the training store's"
        )

    # ── one round ───────────────────────────────────────────────────────

    def _reset_round(mut self, rnd: Int) raises:
        """The task's words, the device reset, the expert's placements, the
        look for this round, the settle; then the per-round bookkeeping."""
        comptime L = Self.LANES
        for e in range(L):
            var mb = e * METADATA_SIZE
            for k in range(METADATA_SIZE):
                self.env.d.meta.data[mb + k] = Scalar[Self.DT](0)
            for k in range(len(self.meta_idx)):
                self.env.d.meta.data[mb + self.meta_idx[k]] = Scalar[Self.DT](
                    self.meta_val[k]
                )
        self.env.d.meta.upload(self.ctx)
        self.ctx.synchronize()
        self.env.reset_batch[L](self.ctx, UInt64(self.cfg.seed0 + rnd))
        self.ctx.synchronize()
        self.env.d.qpos.download(self.ctx)
        self.env.d.qvel.download(self.ctx)
        self.ctx.synchronize()
        for e in range(L):
            var q0 = posed_qpos[So101TowerPlacement](
                self.cfg.task, String(_FAMILY), So101TowerConfig.SLOT_RADIUS,
                UInt64(self.cfg.seed0 + rnd * L + e),
            )
            for k in range(Self.NQ):
                self.env.d.qpos.data[e * Self.NQ + k] = Scalar[Self.DT](q0[k])
            for k in range(Self.NV):
                self.env.d.qvel.data[e * Self.NV + k] = Scalar[Self.DT](0)
        self.env.d.qpos.upload(self.ctx)
        self.env.d.qvel.upload(self.ctx)
        if self.dr_cfg.enabled:
            self.r.background = self.dr.apply(self.cfg.dr_draw0 + rnd, self.r.vis, self.rm)
            self.dr.upload(self.ctx, self.r.vis, self.rm)
        for _ in range(TOWER_EVAL_SETTLE_STEPS):
            self._hold_actions()
            self.ctx.enqueue_copy(self.env._action, self.act_h)
            self.env.step_batch[L](self.ctx, UInt64(1))
            self.env.d.qpos.download(self.ctx)
            self.ctx.synchronize()
        self.env.d.xpos.download(self.ctx)
        self.ctx.synchronize()
        comptime NB = So101TowerModel.NBODY
        for e in range(L):
            self.rest_z[e] = Float64(self.env.d.xpos.data[e * NB * 3 + self.brick * 3 + 2])
            self.held[e] = 0
            self.done[e] = False
            self.succ_step[e] = -1
            self.lifted[e] = False
            self.land_d[e] = -1.0
            self.goal_ever[e] = False
            self.ens[e].reset()
        self.rows_obs = List[List[Float32]]()
        self.rows_act = List[List[Float32]]()
        self.rows_rew = List[List[Float32]]()
        self.rows_nobs = List[List[Float32]]()
        for _ in range(L):
            self.rows_obs.append(List[Float32]())
            self.rows_act.append(List[Float32]())
            self.rows_rew.append(List[Float32]())
            self.rows_nobs.append(List[Float32]())
        self.t_query = 0
        if self.cfg.demo_out.byte_length() > 0:
            self.ctx.enqueue_copy(self.obs_h, self.env._obs)
            self.ctx.synchronize()

    def _hold_actions(mut self):
        """The null action: every joint's target is where it is."""
        for e in range(Self.LANES):
            for k in range(Self.AD):
                self.act_h[e * Self.AD + k] = Scalar[Self.DT](
                    _clamp1(self.units.joint_to_action(
                        k, Float64(self.env.d.qpos.data[e * Self.NQ + k])
                    ))
                )

    def _render(mut self, rnd: Int, t: Int) raises:
        """Host FK of every lane's current qpos -> the rig's tracer, both
        cameras -> uint8 CHW (the store's frame r = FK(qpos r))."""
        comptime L = Self.LANES
        for e in range(L):
            for k in range(Self.NQ):
                self.rd.qpos.data[e * Self.NQ + k] = self.env.d.qpos.data[e * Self.NQ + k]
        forward_kinematics["cpu", RIG_DT, TOWER_MD, L](self.rd, self.rm)
        self.rd.qpos.upload_resident(self.ctx)
        self.rd.xpos.upload_resident(self.ctx)
        self.rd.xquat.upload_resident(self.ctx)
        for slot in range(RIG_N_CAMS):
            self.r.render(self.ctx, self.rd, self.rm, self.cams[slot])
            self.ctx.enqueue_copy(self.h_rgb, self.r.rgb)
            self.ctx.synchronize()
            for e in range(L):
                _ = pack_camera_u8(
                    mptr(self.h_rgb.unsafe_ptr()), e, mptr(self.img_u8),
                    e * RIG_IMG_ELEMS + slot * RIG_CAM_ELEMS,
                )
        if self.cfg.snap_dir.byte_length() > 0 and t == 0:
            self._snap(rnd)

    def _snap(mut self, rnd: Int) raises:
        makedirs(self.cfg.snap_dir, exist_ok=True)
        for slot in range(RIG_N_CAMS):
            var hwc = List[UInt8](length=RIG_CAM_ELEMS, fill=UInt8(0))
            for q in range(RIG_NPIX):
                for c in range(3):
                    hwc[q * 3 + c] = self.img_u8[slot * RIG_CAM_ELEMS + c * RIG_NPIX + q]
            save_png(
                self.cfg.snap_dir + "/round" + String(rnd) + "_lane0_"
                + ("overhead" if slot == 0 else "wrist") + ".png",
                hwc, RIG_CAM_W, RIG_CAM_H, 3,
            )

    def _query(mut self, rnd: Int, t: Int) raises:
        """Render, normalise, one ACT forward at batch LANES -> `chunk`."""
        var tr0 = perf_counter_ns()
        self._render(rnd, t)
        ref nm = self.norm.value()
        for e in range(Self.LANES):
            for c in range(RIG_N_CAMS):
                var o = e * RIG_IMG_ELEMS + c * RIG_CAM_ELEMS
                normalize_camera_chw[RIG_CAM_H, RIG_CAM_W](self.img_u8, o, self.images_n, o)
            for k in range(Self.QPOS):
                var lr = self.units.joint_to_lerobot(
                    k, Float64(self.env.d.qpos.data[e * Self.NQ + k])
                )
                self.qpos_n[e * Self.QPOS + k] = (
                    Scalar[Self.DT](lr) - nm.qpos_mean[k]
                ) / nm.qpos_std[k]
        self.tally.ns_render += perf_counter_ns() - tr0
        var tf0 = perf_counter_ns()
        self.act.predict(
            self.qpos_n, self.images_n, self.dummy_a, self.dummy_v, self.chunk
        )
        self.tally.ns_forward += perf_counter_ns() - tf0
        self.t_query = t

    def _policy_actions(mut self, t: Int) raises:
        """The chunk (ensembled, or open-loop) -> LeRobot target -> joint ->
        the env's action word, clamped; saturated / non-finite counted."""
        ref nm = self.norm.value()
        for e in range(Self.LANES):
            if self.cfg.exec_n == 0:
                self.ens[e].push(t, self.chunk, e * Self.K * Self.ADIM)
                self.ens[e].action_at(t, self.pred_n, 0)
            else:
                var idx = t - self.t_query
                for k in range(Self.ADIM):
                    self.pred_n[k] = self.chunk[e * Self.K * Self.ADIM + idx * Self.ADIM + k]
            denormalize(self.pred_n, 0, nm.action_mean, nm.action_std, self.pred, 0, Self.ADIM)
            for k in range(Self.AD):
                var lr = Float64(self.pred[k])
                var a = self.units.joint_to_action(k, self.units.lerobot_to_joint(k, lr))
                if not (a == a):
                    self.tally.nonfinite += 1
                    a = 0.0
                if a > 1.0 or a < -1.0:
                    self.tally.saturated += 1
                self.tally.act_words += 1
                self.act_h[e * Self.AD + k] = Scalar[Self.DT](_clamp1(a))

    def _step(mut self, t: Int) raises:
        var tp0 = perf_counter_ns()
        var recording = self.cfg.demo_out.byte_length() > 0
        self.ctx.enqueue_copy(self.env._action, self.act_h)
        self.env.step_batch[Self.LANES](self.ctx, UInt64(t + 2))
        self.env.d.qpos.download(self.ctx)
        self.env.d.meta.download(self.ctx)
        self.env.d.xpos.download(self.ctx)
        if recording:
            self.ctx.enqueue_copy(self.rew_h, self.env._reward)
        self.ctx.synchronize()
        self.tally.ns_physics += perf_counter_ns() - tp0
        if recording:
            self._record_before()
            self.ctx.enqueue_copy(self.obs_h, self.env._obs)
            self.ctx.synchronize()
            self._record_after()

    def _record_before(mut self):
        for e in range(Self.LANES):
            if self.done[e]:
                continue
            for k in range(Self.OD):
                self.rows_obs[e].append(Float32(self.obs_h[e * Self.OD + k]))
            for k in range(Self.AD):
                self.rows_act[e].append(Float32(self.act_h[e * Self.AD + k]))
            self.rows_rew[e].append(Float32(self.rew_h[e]))

    def _record_after(mut self):
        for e in range(Self.LANES):
            if self.done[e]:
                continue
            for k in range(Self.OD):
                self.rows_nobs[e].append(Float32(self.obs_h[e * Self.OD + k]))

    def _score_step(mut self, t: Int):
        """The success rule and the failure bookkeeping, per lane."""
        comptime NB = So101TowerModel.NBODY
        for e in range(Self.LANES):
            if self.done[e]:
                continue
            var goal = Float64(
                self.env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD]
            ) > 0.5
            if goal:
                self.goal_ever[e] = True
                self.held[e] += 1
            else:
                self.held[e] = 0
            var bb = e * NB * 3
            var bz = Float64(self.env.d.xpos.data[bb + self.brick * 3 + 2])
            if bz > self.rest_z[e] + TOWER_EVAL_LIFT_DZ:
                self.lifted[e] = True
            elif (self.lifted[e] and self.land_d[e] < 0.0
                  and bz < self.rest_z[e] + TOWER_EVAL_LAND_DZ):
                var dx = Float64(self.env.d.xpos.data[bb + self.brick * 3]) - Float64(
                    self.env.d.xpos.data[bb + self.bowl * 3]
                )
                var dy = Float64(self.env.d.xpos.data[bb + self.brick * 3 + 1]) - Float64(
                    self.env.d.xpos.data[bb + self.bowl * 3 + 1]
                )
                self.land_d[e] = sqrt(dx * dx + dy * dy)
            if self.held[e] >= TOWER_EVAL_HOLD_STEPS:
                self.done[e] = True
                self.succ_step[e] = t + 1

    def _all_done(self) -> Bool:
        for e in range(Self.LANES):
            if not self.done[e]:
                return False
        return True

    def _finish_round(mut self) raises -> Int:
        """Bucket every lane; append the recorded episodes. Returns the
        round's successes."""
        var r_ok = 0
        for e in range(Self.LANES):
            self.tally.n += 1
            if self.succ_step[e] >= 0:
                self.tally.ok += 1
                r_ok += 1
                self.tally.steps_to_success += Float64(self.succ_step[e])
            elif not self.lifted[e]:
                self.tally.no_grasp += 1
            elif self.goal_ever[e]:
                self.tally.goal_not_held += 1
            elif self.land_d[e] < 0.0:
                self.tally.held_end += 1
            elif self.land_d[e] > TOWER_EVAL_NEAR_BOWL:
                self.tally.dropped += 1
            else:
                self.tally.missed += 1
            if self.cfg.demo_out.byte_length() > 0 and len(self.rows_act[e]) > 0:
                self._append_episode(e)
        return r_ok

    def _append_episode(mut self, e: Int) raises:
        comptime OD = Self.OD
        comptime AD = Self.AD
        self.demos.begin_episode()
        var n = len(self.rows_act[e]) // AD
        var o = List[Scalar[Self.DT]](length=OD, fill=Scalar[Self.DT](0))
        var no = List[Scalar[Self.DT]](length=OD, fill=Scalar[Self.DT](0))
        var a = List[Scalar[Self.DT]](length=AD, fill=Scalar[Self.DT](0))
        for rr in range(n):
            for k in range(OD):
                o[k] = Scalar[Self.DT](self.rows_obs[e][rr * OD + k])
                no[k] = Scalar[Self.DT](self.rows_nobs[e][rr * OD + k])
            for k in range(AD):
                a[k] = Scalar[Self.DT](self.rows_act[e][rr * AD + k])
            self.demos.add(o, a, Float64(self.rows_rew[e][rr]), no, 0.0)
        self.demos.end_episode(success=self.succ_step[e] >= 0)

    # ── the run ─────────────────────────────────────────────────────────

    def run(mut self) raises -> TowerEvalTally:
        """Every round; prints one line per round, returns the tally. Only
        `mut self` method calls here — see the module header."""
        var t0 = perf_counter_ns()
        for rnd in range(self.n_rounds):
            self._reset_round(rnd)
            for t in range(self.cfg.steps):
                if self._all_done():
                    break
                if self.cfg.use_act:
                    if self.cfg.exec_n == 0 or t % self.cfg.exec_n == 0:
                        self._query(rnd, t)
                    self._policy_actions(t)
                else:
                    self._hold_actions()
                self._step(t)
                self._score_step(t)
            var r_ok = self._finish_round()
            var secs = Float64(perf_counter_ns() - t0) / 1e9
            print("  round", rnd, ":", r_ok, "/", Self.LANES, "| running",
                  self.tally.ok, "/", (rnd + 1) * Self.LANES, "|", Int(secs), "s")
        if self.cfg.demo_out.byte_length() > 0:
            write_demo_file(self.cfg.demo_out, self.demos)
        return self.tally.copy()
