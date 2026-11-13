"""SuperMimic's object interface: train the object encoder on carry clips.

A frozen tracker, the world model's object network (O1.3's fit, which keeps
fitting here) and a new object encoder whose output is added to the motion
latent: `docs/OBJECT_INTERACTION_PLAN.md` §4, step O1.4. The simulator is the
G1 with one free box; the object's pose is the simulator's (no perception).

    pixi run -e nvidia mojo build -I . -D SM_PAPER=1 examples/g1/supermimic_object_train.mojo -o /tmp/sm_obj_train
    /tmp/sm_obj_train --ckpt TRACKER.ckpt --object-ckpt OBJECT_FIT.ckpt \\
        --store CARRY_50HZ.h5 --object-dir OBJECT_CSV_DIR --clips CLIPS.tsv \\
        [--iters 20000] [--log-every 200] [--out INTERFACE.ckpt] [--checkpoint-every 5000]
        [--seed 0] [--utd 0.4] [--warmup 40] [--lr 2e-4] [--om-lr 5e-5] [--gain 0.1] [--replay 1000]
        [--w-pos 1.0] [--std-pos 0.2] [--w-ori 0.5] [--std-ori 0.4] [--w-hands 1.0] [--std-hands 0.1]
        [--no-noise 1] [--resume INTERFACE.ckpt] [--squeeze-copies 0.10] [--squeeze-force 40] [--squeeze-until 6000]

    --ckpt         the tracker's checkpoint (its world model, encoder, decoder: none of them is written)
    --object-ckpt  the object network alone, as `supermimic_object_fit.mojo --out` saves it;
                   without it the object network starts at zero (a box at constant velocity)
    --clips        a TSV with `name` and `split` columns; the `train` rows are used
    the six reward numbers are data: a revision is a run, not a build
    --squeeze-copies D  the library is the train clips TWICE: as they are, and with the hands aimed D metres
                   into the box while the reference box is lifted (`object_task.sm_squeeze_reference`). The
                   tracker presses on the box on the second half's lanes, so the replay and the object network
                   see held boxes. The box's reference and the hands' reference grasp are the ORIGINAL clips'
                   on both halves. An exploration prior: evaluation uses the clips as they are.
    --squeeze-force F   another exploration prior, on the EXECUTED actions (`object_agent_gpu.SMObjSqueeze`): while
                   the reference box is lifted, lane l's hands are pushed toward the simulated box with
                   F · (l mod 8) / 7 newtons, so an eighth of the lanes are plain and the rest show the
                   object network how the box's slip depends on the squeeze. The policy's own actions are
                   what the imagined rollout uses.
    --squeeze-until N   the squeeze's forces fade linearly to zero at iteration N (0: they stay). Once the
                   interface squeezes by itself the scripted squeeze on top is too much.

Compile-time sizes: `-D SM_PAPER=1` the tracker's paper widths (a paper
checkpoint needs it), `-D SM_LANES=1024`, `-D SM_OBJ_BATCH=2048` (windows per
object-network batch), `-D SM_PI_BATCH=2048` (imagined starts), `-D
SM_OBJ_H=1024` (the object network's width: the fit's).

Each log line: the simulator's reward per step (the tracker's eight terms),
falls and episode ends; the box against its reference in the simulator (mean
distance, mean angle, share of lanes within 10 cm); the imagined objective and
the box's three reward terms per imagined step; the object network's k-step
errors on the interface's own replay.

⚠ NVIDIA only (the batched G1 does not build for Metal). mjlab's G1 only.
"""

from std.random import seed
from std.sys import argv, get_defined_int

from max.gpu.host import DeviceContext

from noeira.core.bytes import string_from_bytes
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.agent import sm_schedule
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_agent_gpu import SM_OBJ_OM_LR, SMObjAgentGpu, SMObjSqueeze
from noeira.deep_agents.supermimic.object_env_gpu import sm_object_track
from noeira.deep_agents.supermimic.object_policy import SMObjConfig
from noeira.deep_agents.supermimic.object_task import (
    SM_OSTD_HANDS, SM_OSTD_ORI, SM_OSTD_POS, SM_OW_HANDS, SM_OW_ORI, SM_OW_POS, sm_object_ref_rows, sm_obj_reward_params,
    sm_squeeze_reference,
)
from noeira.deep_agents.supermimic.object_spec import SM_OBJ
from noeira.deep_agents.supermimic.object_trainer import SMObjTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabBoxBatchedSim, G1MjlabBoxRobot, SMDims
from noeira.envs.robots.unitree_g1_mjlab_pd import g1m_action_scale, g1m_kp
from noeira.io.fileio import read_file_bytes
from noeira.nn.constants import DT


comptime P = get_defined_int["SM_PAPER", 0]() != 0
comptime LANES = get_defined_int["SM_LANES", 1024]()
comptime WB = get_defined_int["SM_OBJ_BATCH", 2048]()
comptime PB = get_defined_int["SM_PI_BATCH", 2048]()
comptime HO = get_defined_int["SM_OBJ_H", 1024]()
comptime Robot = G1MjlabBoxRobot
comptime Sim = G1MjlabBoxBatchedSim[LANES]
comptime D = SMDims[Robot]
comptime ARM0 = 15          # mjlab's G1: joints 15..28 are the two arms (`--squeeze-copies`)
# the tracker's widths (`supermimic_train.mojo`'s paper and smoke sizes); the shapes are this driver's
comptime C = SMConfig[
    "gpu", Robot,
    2048 if P else 128, 64, 2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32,
    2048 if P else 128, 1024 if P else 64, 512 if P else 64, 256 if P else 32, 256 if P else 64, 128 if P else 32,
    LANES, WB, PB, 32,
]
# the interface: the object encoder at the prompt interface's widths (Table A.4's MLP (256, 256)), its critic (256, 128)
comptime OC = SMObjConfig[C, HO, 256 if P else 64, 256 if P else 64, 256 if P else 64, 128 if P else 32]


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _object_poses(path: String, frames: Int) raises -> List[Float64]:
    """The box's pose rows of one clip, cut or padded (its last row repeated)
    to the clip's `frames`: [frames · 7]. (`supermimic_object_fit.mojo`'s.)"""
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


def _train_names(path: String) raises -> List[String]:
    """The TSV's `train` rows (`name` and `split` columns first)."""
    var tr = List[String]()
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
        if len(cols) >= 2 and cols[1] == "train":
            tr.append(cols[0].copy())
    return tr^


def _has(names: List[String], x: String) -> Bool:
    for i in range(len(names)):
        if names[i] == x:
            return True
    return False


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var ckpt = _opt(args, String("--ckpt"), String(""))
    var obj_ckpt = _opt(args, String("--object-ckpt"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var object_dir = _opt(args, String("--object-dir"), String(""))
    var clips = _opt(args, String("--clips"), String(""))
    var out = _opt(args, String("--out"), String(""))
    var resume = _opt(args, String("--resume"), String(""))
    var iters = Int(_opt(args, String("--iters"), String("20000")))
    var log_every = Int(_opt(args, String("--log-every"), String("200")))
    var ckpt_every = Int(_opt(args, String("--checkpoint-every"), String("5000")))
    var rng_seed = Int(_opt(args, String("--seed"), String("0")))
    var utd = Float64(_opt(args, String("--utd"), String(C.DEF_UTD)))
    var warmup = Int(_opt(args, String("--warmup"), String(C.DEF_WARMUP)))
    var lr = Float64(_opt(args, String("--lr"), String(OC.DEF_LR)))
    var om_lr = Float64(_opt(args, String("--om-lr"), String(SM_OBJ_OM_LR)))
    var gain = Float64(_opt(args, String("--gain"), String(OC.DEF_GAIN)))
    var cap = Int(_opt(args, String("--replay"), String(C.DEF_REPLAY_CAP)))
    var noise = _opt(args, String("--no-noise"), String("0")) == "0"
    var sq_depth = Float64(_opt(args, String("--squeeze-copies"), String("0")))
    var sq_force = Float64(_opt(args, String("--squeeze-force"), String("0")))
    var sq_until = Int(_opt(args, String("--squeeze-until"), String("0")))
    var rw = sm_obj_reward_params(
        Float64(_opt(args, String("--w-pos"), String(SM_OW_POS))), Float64(_opt(args, String("--std-pos"), String(SM_OSTD_POS))),
        Float64(_opt(args, String("--w-ori"), String(SM_OW_ORI))), Float64(_opt(args, String("--std-ori"), String(SM_OSTD_ORI))),
        Float64(_opt(args, String("--w-hands"), String(SM_OW_HANDS))), Float64(_opt(args, String("--std-hands"), String(SM_OSTD_HANDS))),
    )
    if ckpt == "" or store == "" or object_dir == "" or clips == "":
        raise Error("give --ckpt TRACKER.ckpt --store STORE.h5 --object-dir DIR --clips CLIPS.tsv [--object-ckpt FIT.ckpt]")
    seed(rng_seed)

    # ── the train clips: library rows and the box's poses ──
    var want = _train_names(clips)
    var st = TrajectoryStore(store.copy())
    var refc = st.load_column[DType.float32](String("ref"))
    var rows = List[Scalar[DT]]()
    var lens = List[Int]()
    var poses = List[Float64]()
    var o = 0
    for e in range(st.n_episodes()):
        var l = Int(st.episodes.ep_len[e])
        var name = st.manifest.task_text(e)
        if _has(want, name):
            for i in range(l * D.S):
                rows.append(refc[o * D.S + i])
            lens.append(l)
            var p = _object_poses(object_dir + "/" + name + ".csv", l)
            for i in range(len(p)):
                poses.append(p[i])
        o += l
    if len(lens) != len(want):
        raise Error("the store holds " + String(len(lens)) + " of the " + String(len(want)) + " train clips named")
    var track = sm_object_track(poses, lens)
    print(
        "clips: train", len(lens), "(", len(rows) // D.S, "frames ) | lanes", LANES, "| object-network batch", WB, "| imagined starts", PB,
        "| horizon", C.HORIZON, "| object width", HO,
    )

    var ctx = DeviceContext()
    var base = SuperMimicTrainer[C].for_robot(Robot.SIM.create(), Optional(ctx))
    var ctr = base.load_state(ckpt)
    print("tracker checkpoint at iteration", ctr[0], "(", ctr[1], "updates ): its world model, encoder and decoder stay as they are")
    var rows_ref = rows.copy()
    var track_ref = track.copy()
    var orefs = sm_object_ref_rows[Robot](base.tables, rows_ref, track_ref)
    if sq_depth > 0.0:
        var squeezed = sm_squeeze_reference[Robot](base.tables, rows, lens, track, SM_OBJ, sq_depth, ARM0)
        var moved = 0.0
        for i in range(len(rows)):
            moved = max(moved, abs(Float64(squeezed[i]) - Float64(rows[i])))
        var nc = len(lens)
        for c in range(nc):
            lens.append(lens[c])
        var nt = len(track)
        for i in range(nt):
            track.append(track[i])
        var no = len(orefs)
        for i in range(no):
            orefs.append(orefs[i])
        for i in range(len(squeezed)):
            rows.append(squeezed[i])
        print(
            "squeezed copies: the library is", len(lens), "clips, the second half with the hands", sq_depth,
            "m into the box while it is lifted (largest joint change", moved, "rad); the box's and the hands' references are the originals'",
        )
    var tr = SMObjTrainer[OC](base^)
    if obj_ckpt != "":
        tr.load_object(obj_ckpt)
        print("object network from", obj_ckpt)
    else:
        print("object network at its zero start (no --object-ckpt): the box moves at constant velocity until it has fitted")
    tr.gain = gain
    var q0 = List[Float64](length=Robot.NDOF, fill=0.0)
    for j in range(Robot.NDOF):
        q0[j] = Robot.default_joint_pos(j)
    var sched = sm_schedule[C](iters, noise, rng_seed, utd, warmup, lr, cap)
    var sq = SMObjSqueeze[Robot](ctx)
    if sq_force > 0.0:
        var forces = List[Float64](length=LANES, fill=0.0)
        for l in range(LANES):
            forces[l] = sq_force * Float64(l % 8) / 7.0
        var ks = List[Float64](length=Robot.NDOF, fill=1.0)
        for j in range(Robot.NDOF):
            ks[j] = g1m_kp(j) * g1m_action_scale(j)
        sq = SMObjSqueeze[Robot](ctx, LANES, lens, track, SM_OBJ, forces, ks, ARM0)
        print("squeeze on the executed actions: lane l's hands pushed toward the box with", sq_force, "· (l mod 8) / 7 newtons while the reference box is lifted")
    var agent = SMObjAgentGpu[OC, Sim](ctx, SMMotionLibrary[Robot](rows^, lens, q0), track, orefs, Sim.create(ctx), tr^, sched^)
    agent.om_lr = om_lr
    agent.squeeze = sq^
    agent.squeeze_until = sq_until
    if sq_force > 0.0 and sq_until > 0:
        print("the squeeze fades linearly to zero at iteration", sq_until)
    agent.set_reward(rw)
    if resume != "":
        var rs = agent.trainer.load_state(resume)
        agent.resume(rs[0], rs[1])
        print("resumed from", resume, "at iteration", rs[0], "(", rs[1], "updates )")
    print(
        "gain", gain, "| lr", lr, "| object-network lr", om_lr, "| utd", utd, "| noise and randomisation", noise,
        "| reward (dt·w, 1/σ²) pos, ori, hands:", rw[0], rw[1], rw[2], rw[3], rw[4], rw[5],
    )
    agent.train(iters, log_every, ckpt_every, out)
    if out != "":
        print("wrote", out)
