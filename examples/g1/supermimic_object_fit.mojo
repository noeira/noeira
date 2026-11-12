# +--------------------------------------------------------------------------+ #
# | SuperMimic with an object: fit the object network on the blind tracker's replay
# +--------------------------------------------------------------------------+ #
"""Step O1.3's fit (`docs/OBJECT_INTERACTION_PLAN.md`): a FROZEN tracker
checkpoint plays carry clips in the simulator that also holds the box, without
seeing it, with the training env's noise and randomisation; the object network
(`object_model.SMObjectModel`) is fitted on that replay beside the tracker's
frozen world model, then scored on the replay of HELD-OUT clips.

pixi run -e nvidia mojo build -I . -D SM_PAPER=1 examples/g1/supermimic_object_fit.mojo -o sm_obj_fit
./sm_obj_fit --ckpt tracker.ckpt --store carry_50hz.h5 --object-dir DIR --clips clips.tsv \\
    [--iters 3000] [--eval-steps 400] [--updates 1] [--lr 2e-4] [--seed 0] [--out object.ckpt]

    --store       the clips' library store (one episode per clip, `ref` column)
    --object-dir  one `<clip name>.csv` per clip: the box's pose at 50 Hz
                  (`x,y,z,qw,qx,qy,qz`, a header row; frame f of the clip is row f)
    --clips       a TSV with a header whose first two columns are `name` and
                  `split` (`train` or `eval`); clips it does not name are left out
    --iters       env steps of the fit (each: LANES transitions, `--updates` updates)
    --eval-steps  env steps collected on each split before scoring

Comptime: `-D SM_PAPER=1` the tracker's paper widths (a checkpoint loads only
into its own), `-D SM_LANES=` (1024), `-D SM_OBJ_BATCH=` (2048), `-D SM_OBJ_H=`
(1024, the object network's width).

WHAT IT PRINTS AT THE END, on windows of k = 8 steps from the held-out clips'
replay, open loop, for the fitted network and for the SAME network at its zero
start (which predicts the tracker's world model for the robot and a constant
velocity for the box):

    box position error after 1 and k steps (m), over all windows and over the
    windows where the logged box moved more than 2 cm (most windows hold a box
    at rest, where any model is right: read the `moving` line)
    the robot's k-step error, fitted over zero start (O1.3: not above 1.1)

⚠ NVIDIA only (the batched G1 does not build for Metal). mjlab's G1 only.
"""

from std.math import ceildiv
from std.random import random_ui64, seed
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.core.bytes import string_from_bytes
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.agent import sm_cosine_lr
from noeira.deep_agents.supermimic.agent_gpu import _k_ref_root
from noeira.deep_agents.supermimic.blocks import ObjectModelUpdateStep
from noeira.deep_agents.supermimic.env import SMStepInfo
from noeira.deep_agents.supermimic.env_gpu import SMReplayGpuOf
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import talloc, tp
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_env_gpu import SMObjReplayGpu, SMObjTrackingEnvGpu, sm_object_track
from noeira.deep_agents.supermimic.object_model import SMObjLoss, SMObjWindowBatch, SMObjectModel, sm_obj_window_loss
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX
from noeira.deep_agents.supermimic.robot import G1MjlabBoxBatchedSim, G1MjlabBoxRobot, SMDims
from noeira.deep_agents.supermimic.world_model import SMWindowBatch
from noeira.io.fileio import read_file_bytes
from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import CheckpointScalars, save_params_multi
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl


comptime P = get_defined_int["SM_PAPER", 0]() != 0
comptime LANES = get_defined_int["SM_LANES", 1024]()
comptime B = get_defined_int["SM_OBJ_BATCH", 2048]()
comptime HO = get_defined_int["SM_OBJ_H", 1024]()
comptime Robot = G1MjlabBoxRobot
comptime Sim = G1MjlabBoxBatchedSim[LANES]
comptime D = SMDims[Robot]
# the tracker's widths (`supermimic_train.mojo`'s paper and smoke sizes); the shapes are this driver's
comptime C = SMConfig[
    "gpu", Robot,
    2048 if P else 128, 64, 2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32,
    2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32, 256 if P else 64, 128 if P else 32,
    LANES, B, 16, 32,
]
comptime K = C.DEF_WM_K
comptime Env = SMObjTrackingEnvGpu[Robot, Sim]
comptime OM = SMObjectModel[Robot, HO]
comptime _TPB: Int = 64


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _object_poses(path: String, frames: Int) raises -> List[Float64]:
    """The box's pose rows of one clip, cut or padded (its last row repeated)
    to the clip's `frames`: [frames · 7]."""
    var rows = List[Float64]()
    var first = True
    for line in string_from_bytes(read_file_bytes(path)).split("\n"):
        var t = String(String(line).strip())
        if t.byte_length() == 0:
            continue
        if first:               # the header
            first = False
            continue
        var n = 0
        for f in t.split(","):
            rows.append(Float64(String(String(f).strip())))
            n += 1
        if n != 7:
            raise Error(path + ": a row of " + String(n) + " values, expected 7")
    var have = len(rows) // 7
    if have < 2:
        raise Error(path + ": fewer than two rows")
    var out = List[Float64](length=frames * 7, fill=0.0)
    for f in range(frames):
        var s = min(f, have - 1) * 7
        for k in range(7):
            out[f * 7 + k] = rows[s + k]
    return out^


def _splits(path: String) raises -> Tuple[List[String], List[String]]:
    """(train names, eval names) from the TSV's `name` and `split` columns."""
    var tr = List[String]()
    var ev = List[String]()
    var first = True
    for line in string_from_bytes(read_file_bytes(path)).split("\n"):
        var t = String(String(line).strip())
        if t.byte_length() == 0:
            continue
        var cols = List[String]()
        for c in t.split("\t"):
            cols.append(String(String(c).strip()))
        if first:
            first = False
            if len(cols) < 2 or cols[0] != "name" or cols[1] != "split":
                raise Error(path + ": the header's first two columns must be `name` and `split`")
            continue
        if len(cols) < 2:
            raise Error(path + ": a row without a split: " + t)
        if cols[1] == "train":
            tr.append(cols[0].copy())
        elif cols[1] == "eval":
            ev.append(cols[0].copy())
        else:
            raise Error(path + ": split `" + cols[1] + "` for " + cols[0] + " (train or eval)")
    return (tr^, ev^)


struct Clips(Movable):
    """A split: its library rows, clip lengths and the box's pose rows."""
    var rows: List[Scalar[DT]]
    var lens: List[Int]
    var poses: List[Float64]
    var short: Int               # clips whose object track had fewer rows than the clip

    def __init__(out self):
        self.rows = List[Scalar[DT]]()
        self.lens = List[Int]()
        self.poses = List[Float64]()
        self.short = 0

    def __init__(out self, *, deinit move: Self):
        self.rows = move.rows^
        self.lens = move.lens^
        self.poses = move.poses^
        self.short = move.short


def _has(names: List[String], x: String) -> Bool:
    for i in range(len(names)):
        if names[i] == x:
            return True
    return False


def _load(store: String, object_dir: String, want: List[String]) raises -> Clips:
    """The store's episodes named in `want`, in the store's order."""
    var st = TrajectoryStore(store.copy())
    var refc = st.load_column[DType.float32](String("ref"))
    var out = Clips()
    var o = 0
    for e in range(st.n_episodes()):
        var l = Int(st.episodes.ep_len[e])
        var name = st.manifest.task_text(e)
        if _has(want, name):
            for i in range(l * D.S):
                out.rows.append(refc[o * D.S + i])
            out.lens.append(l)
            var p = _object_poses(object_dir + "/" + name + ".csv", l)
            for i in range(len(p)):
                out.poses.append(p[i])
        o += l
    if len(out.lens) != len(want):
        raise Error("the store holds " + String(len(out.lens)) + " of the " + String(len(want)) + " clips named")
    return out^


def _q0() -> List[Float64]:
    var q0 = List[Float64](length=Robot.NDOF, fill=0.0)
    for j in range(Robot.NDOF):
        q0[j] = Robot.default_joint_pos(j)
    return q0^


def _make_env(ctx: DeviceContext, c: Clips, rng_seed: Int) raises -> Env:
    var track = sm_object_track(c.poses, c.lens)
    return Env(ctx, SMMotionLibrary[Robot](c.rows.copy(), c.lens, _q0()), track, Sim.create(ctx), rng_seed=rng_seed)


struct Windows(Movable):
    """One batch of k-step windows, robot and object, device-resident."""
    var wb: SMWindowBatch[Robot]
    var wo: SMObjWindowBatch
    var idx: DeviceBuffer[DType.int32]
    var frame: DeviceBuffer[DType.int32]
    var aclean: DeviceBuffer[DT]
    var reward: DeviceBuffer[DT]

    def __init__(out self, ctx: DeviceContext) raises:
        self.wb = SMWindowBatch[Robot](K, B)
        self.wb.states.ensure_gpu(ctx, (K + 1) * B * D.S)
        self.wb.a_exec.ensure_gpu(ctx, K * B * Robot.NDOF)
        self.wb.xi.ensure_gpu(ctx, B * D.X)
        self.wb.done.ensure_gpu(ctx, K * B)
        self.wb.ref_root.ensure_gpu(ctx, K * B * 7)
        self.wo = SMObjWindowBatch(K, B)
        self.wo.states.ensure_gpu(ctx, (K + 1) * B * SM_OBJ)
        self.wo.ctx.ensure_gpu(ctx, B * SM_OBJ_CTX)
        self.idx = ctx.enqueue_create_buffer[DType.int32](B)
        self.frame = ctx.enqueue_create_buffer[DType.int32](K * B)
        self.aclean = ctx.enqueue_create_buffer[DT](K * B * Robot.NDOF)
        self.reward = ctx.enqueue_create_buffer[DT](K * B)
        ctx.synchronize()

    def __init__(out self, *, deinit move: Self):
        self.wb = move.wb^
        self.wo = move.wo^
        self.idx = move.idx^
        self.frame = move.frame^
        self.aclean = move.aclean^
        self.reward = move.reward^

    def draw(mut self, ctx: DeviceContext, mut env: Env, mut rb: SMReplayGpuOf[Robot], mut orb: SMObjReplayGpu, check: Bool) raises:
        """B windows from the two rings, the same rows of both."""
        var ip = mptr(self.idx.unsafe_ptr())
        rb.sample_windows(ctx, K, B, random_ui64(0, 0xFFFFFFFFFFFF), ip)
        if check and rb.eligible_total(ctx) == 0:
            raise Error("no window of " + String(K) + " steps in the replay yet")
        rb.windows_at(
            ctx, ip, B, K, tp["gpu"](self.wb.states), tp["gpu"](self.wb.a_exec), mptr(self.aclean.unsafe_ptr()),
            tp["gpu"](self.wb.xi), mptr(self.reward.unsafe_ptr()), tp["gpu"](self.wb.done), mptr(self.frame.unsafe_ptr()),
        )
        ctx.enqueue_function[_k_ref_root[D.S]](
            Int64(K * B), mptr(env.env.lib.rows_dev.value().unsafe_ptr()), mptr(self.frame.unsafe_ptr()), tp["gpu"](self.wb.ref_root),
            grid_dim=ceildiv(K * B, _TPB), block_dim=_TPB,
        )
        orb.windows_at(ctx, ip, B, K, tp["gpu"](self.wo.states), tp["gpu"](self.wo.ctx))


def _zero_loss() -> SMObjLoss:
    return SMObjLoss(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _env_step(
    ctx: DeviceContext, mut trainer: SuperMimicTrainer[C], mut env: Env, mut goals: TensorImpl[DT],
    mut rb: SMReplayGpuOf[Robot], mut orb: SMObjReplayGpu,
) raises -> SMStepInfo:
    env.env.policy_inputs(ctx, goals)
    var a = trainer.act[LANES](env.env.hist, goals, resident=True)
    return env.step(ctx, a, rb, orb)


def _score(
    ctx: DeviceContext, label: String, mut trainer: SuperMimicTrainer[C], mut om: OM, mut om0: OM, mut win: Windows,
    mut env: Env, mut rb: SMReplayGpuOf[Robot], mut orb: SMObjReplayGpu, batches: Int,
) raises -> Tuple[SMObjLoss, SMObjLoss]:
    """`batches` window batches scored by the fitted network and by the zero
    start, on the SAME windows."""
    var fit = _zero_loss()
    var zero = _zero_loss()
    var w = 1.0 / Float64(batches)
    for i in range(batches):
        win.draw(ctx, env, rb, orb, i == 0)
        fit.add_scaled(sm_obj_window_loss[Robot, C.WM_H, C.TERM_H, HO, B, "gpu"](trainer.wm, om, win.wb, win.wo, False, ctx=Optional(ctx)), w)
        zero.add_scaled(sm_obj_window_loss[Robot, C.WM_H, C.TERM_H, HO, B, "gpu"](trainer.wm, om0, win.wb, win.wo, False, ctx=Optional(ctx)), w)
    print(label, "(", batches * B, "windows of", K, "steps ):")
    print("  box position error, 1 step  : fitted", fit.obj_pos_1step, "m | zero start", zero.obj_pos_1step, "m")
    print("  box position error, k steps : fitted", fit.obj_pos_kstep, "m | zero start", zero.obj_pos_kstep, "m")
    print(
        "  … where the box moved > 2 cm:  fitted", fit.obj_pos_kstep_moving, "m | zero start", zero.obj_pos_kstep_moving,
        "m | share of the windows", fit.moving_share,
    )
    print("  box error (normalised), k    : fitted", fit.obj_nse_kstep, "| zero start", zero.obj_nse_kstep)
    print(
        "  robot error (normalised), k  : fitted", fit.nse_kstep, "| zero start", zero.nse_kstep, "| ratio",
        fit.nse_kstep / max(zero.nse_kstep, 1e-30),
    )
    return (fit^, zero^)


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var ckpt = _opt(args, String("--ckpt"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var object_dir = _opt(args, String("--object-dir"), String(""))
    var clips = _opt(args, String("--clips"), String(""))
    var out = _opt(args, String("--out"), String(""))
    var iters = Int(_opt(args, String("--iters"), String("3000")))
    var eval_steps = Int(_opt(args, String("--eval-steps"), String("400")))
    var updates = Int(_opt(args, String("--updates"), String("1")))
    var lr = Float64(_opt(args, String("--lr"), String("2e-4")))
    var rng_seed = Int(_opt(args, String("--seed"), String("0")))
    var cap = Int(_opt(args, String("--replay"), String("1000")))
    var log_every = Int(_opt(args, String("--log-every"), String("100")))
    if ckpt == "" or store == "" or object_dir == "" or clips == "":
        raise Error("give --ckpt PATH --store STORE.h5 --object-dir DIR --clips CLIPS.tsv")
    seed(rng_seed)

    var sp = _splits(clips)
    var train = _load(store, object_dir, sp[0])
    var held = _load(store, object_dir, sp[1])
    print("clips: train", len(train.lens), "| held out", len(held.lens), "| lanes", LANES, "| batch", B, "| object width", HO)

    var ctx = DeviceContext()
    var trainer = SuperMimicTrainer[C].for_robot(Robot.SIM.create(), Optional(ctx))
    var ctr = trainer.load_state(ckpt)
    print("tracker checkpoint at iteration", ctr[0], "(", ctr[1], "updates ), frozen")
    var om = OM.make[Kaiming, "gpu"](Optional(ctx))
    var om0 = OM.make[Kaiming, "gpu"](Optional(ctx))      # never stepped: the zero start
    var blk = ObjectModelUpdateStep[C, HO]()
    var win = Windows(ctx)
    var goals = talloc["gpu"](LANES * D.GOAL, Optional(ctx))

    # ── the fit, on the train clips ──
    var env = _make_env(ctx, train, rng_seed)
    var rb = SMReplayGpuOf[Robot](ctx, LANES, cap)
    var orb = SMObjReplayGpu(ctx, LANES, cap)
    var acc = _zero_loss()
    var n_acc = 0
    var r_acc = 0.0
    var term = 0
    var trunc = 0
    var n_upd = 0
    var t0 = perf_counter_ns()
    for it in range(1, iters + 1):
        var si = _env_step(ctx, trainer, env, goals, rb, orb)
        r_acc += si.reward_mean
        term += si.n_terminated
        trunc += si.n_truncated
        if it > C.DEF_WARMUP:
            blk.opt.lr = Scalar[DT](sm_cosine_lr(it, iters, lr))
            for u in range(updates):
                win.draw(ctx, env, rb, orb, n_upd == 0)
                var r = blk.step["gpu", B](trainer.wm, om, win.wb, win.wo, Optional(ctx))
                acc.add_scaled(r[0], 1.0)
                n_acc += 1
                n_upd += 1
        if it % log_every == 0 or it == iters:
            var w = 1.0 / Float64(max(n_acc, 1))
            print(
                "it", it, "| reward/step", r_acc / Float64(log_every), "| ended: fell", term, "other", trunc,
                "| loss", acc.loss * w, "robot", acc.state_loss * w, "object", acc.obj_loss * w,
                "| object nse 1-step", acc.obj_nse_1step * w, "k-step", acc.obj_nse_kstep * w,
                "| robot nse k-step", acc.nse_kstep * w, "| updates", n_upd, "|", Float64(perf_counter_ns() - t0) * 1e-9, "s",
            )
            acc = _zero_loss()
            n_acc = 0
            r_acc = 0.0
            term = 0
            trunc = 0
    print("")
    _ = _score(ctx, String("TRAIN clips' replay"), trainer, om, om0, win, env, rb, orb, 8)

    # ── the held-out clips: a fresh env and replay, no update ──
    var henv = _make_env(ctx, held, rng_seed + 1)
    var hrb = SMReplayGpuOf[Robot](ctx, LANES, cap)
    var horb = SMObjReplayGpu(ctx, LANES, cap)
    for _ in range(eval_steps):
        _ = _env_step(ctx, trainer, henv, goals, hrb, horb)
    var sc = _score(ctx, String("HELD-OUT clips' replay"), trainer, om, om0, win, henv, hrb, horb, 8)
    var ratio = sc[0].nse_kstep / max(sc[1].nse_kstep, 1e-30)
    print("")
    print(
        "O1.3 gate: box position after k =", K, "steps", sc[0].obj_pos_kstep, "m (bound 0.02), on the moving windows",
        sc[0].obj_pos_kstep_moving, "m; robot error ratio", ratio, "(bound 1.1)",
    )
    if out != "":
        var scal = CheckpointScalars()
        scal.set_int("iteration", iters)
        scal.set_int("updates", n_upd)
        save_params_multi["gpu"](out, Optional(ctx), False, om, scalars=scal)
        print("wrote", out)
