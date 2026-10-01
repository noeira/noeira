# +--------------------------------------------------------------------------+ #
# | Build the G1 command bank — compounds, gated, CEM-refined (§12.52)
# +--------------------------------------------------------------------------+ #
"""The bank is a list of COMPOUNDS, not a list of privileged offsets.

    pixi run mojo build -I . -Xlinker -ld_classic \
        examples/g1/bfm_zero_bank_build.mojo -o build/g1bank
    ./build/g1bank --ckpt runs/<id>/checkpoints/step_36000.ckpt \
        --out g1_command_bank.txt

## What this produces

One file of named whole-body commands, each carrying its compound, the pool
support behind it, the control that proves its goal term does work, what the
robot actually achieved, and the CEM-refined `z` the viewer loads. Consumers
load `z` and never recompute anything.

## ⚠ THE BUILDER REFUSES ENTRIES

The failure mode this whole arc has been about is a bank that fills up with
commands that look fine and do nothing. §12.50 passed its numeric gate 4 of 4
and failed on sight. So every entry faces three gates and a rejected entry is
NOT written:

  G1 support    the product's ESS >= `--min-ess` rows. A product is an AND of
                sets; §12.51's squat compound retains 15 rows of a 4096 pool.
  G2 scaffold   the goal quantity under the full compound must differ from the
                SCAFFOLD (the same compound with the goal term deleted) by at
                least 15 % of that quantity's pool spread. Most of these
                compounds are mostly "stand upright and still"; one that reads
                well because its scaffold does has measured nothing.
  G3 behaviour  the hard compound must actually hold on the rollout, at least
                `--min-hard` of the scored window.

Rejections are printed with their numbers. They belong in the record — a
command LAFAN cannot express is a fact about the dataset worth keeping.

## Latent search, and why the objective is smooth

CEM over `z` is worth 37.6 % on pose reaching (§12.46) and 49.3 % on the
joystick's velocity prompts (§12.49), network frozen. Here it optimises the
compound's own satisfaction on the ROLLOUT — the honest objective, since it
is what the prompt is a proxy for, and it is what removes the residual motion
a prompt alone leaves (§12.51's arm poses hold at 0.38 m/s).

⚠ The search climbs the SMOOTH form of the compound (`g1_term_soft`). A
product of hard indicators is zero almost everywhere near a bad prompt, so
every candidate would score 0, every elite set would be arbitrary, and CEM
would return its own starting point while appearing to run. The HARD
satisfaction rate is what gets published and gated.
"""

from std.math import exp, sqrt, abs, cos as _cos64, log as _log64
from std.random import random_float64, seed
from std.sys import argv
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.module import Module
from noeira.core.cont_action import ContAction
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.obs_norm import ObsNorm
from noeira.io.fileio import write_text_atomic, read_file_bytes
from noeira.core.bytes import string_from_bytes
from noeira.deep_agents.fb.z_sampler import z_from_reward
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTowerFiltered, BFMBNetFiltered,
)
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_rsi import G1RsiTable, G1_RSI_NQ, G1_RSI_NV
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA, G1ActorObs,
)
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
    UNITREE_G1_PRIV_DIM,
)
from noeira.envs.robots.g1_tracking_eval import (
    G1_D, G1_H, G1_L, G1_HB, g1_project_z,
)
from noeira.envs.robots.g1_spec import (
    G1Pool, g1_spec_load_candidates,
)
from noeira.envs.robots.g1_reward_vocab import (
    G1_NVOC, G1Term, g1_quantities, g1_vocab_name, g1_term_value,
    g1_term_soft, g1_term_str, g1_ess_rows, g1_quantile,
    OP_GT, OP_LT, OP_BAND, OP_SOFT,
    QV_BODY_H, QV_HEAD_H, QV_LHAND_H, QV_RHAND_H, QV_LHAND_LAT,
    QV_RHAND_LAT, QV_LFOOT_H, QV_RFOOT_H, QV_UPRIGHT, QV_SPEED_FWD,
    QV_SPEED_LAT, QV_SPEED, QV_YAW_RATE, QV_TORSO_YAW,
)

comptime SP: Int = UNITREE_G1_OBS_DIM
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = G1_D
comptime BATCH: Int = 64
comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime HOLD: Int = 40
comptime CHUNK: Int = 4096

comptime FNet = BFMFTower[OBS, ACT, D, G1_H, G1_L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, G1_HB]
comptime ANet = BFMActorTowerFiltered[
    OBS, UNITREE_G1_STATE_DIM, G1_ACTOR_EXTRA, D, G1_H, G1_L, ACT
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]


def _has(name: String) -> Bool:
    var av = argv()
    for i in range(len(av)):
        if String(av[i]) == name:
            return True
    return False


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _f3(v: Float64) -> String:
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 1000.0 + 0.5)
    var f = String(h % 1000)
    while f.byte_length() < 3:
        f = String("0") + f
    var body = String(h // 1000) + String(".") + f
    return (String("-") + body) if neg else body


def _pad(s: String, w: Int) -> String:
    var o = s
    while o.byte_length() < w:
        o += String(" ")
    return o


def _lpad(s: String, w: Int) -> String:
    var o = s
    while o.byte_length() < w:
        o = String(" ") + o
    return o


def _gauss() -> Float64:
    var u1 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    return sqrt(-2.0 * _log64(u1)) * _cos64(6.283185307179586 * random_float64())


struct Roll(Copyable, Movable):
    var soft: Float64
    var hard: Float64
    var q: List[Float64]
    # ⚠ THE PER-STEP RANGE, not just the mean. `run` reported a mean of
    # 1.798 m/s inside its own [1.50, 3.00] band and a hold of 0.000 — the
    # mean sits in the band while the stride carries the instantaneous speed
    # out of it twice a cycle. A mean cannot show that; a range can.
    var qmin: List[Float64]
    var qmax: List[Float64]

    def __init__(out self, soft: Float64, hard: Float64, q: List[Float64],
                 qmin: List[Float64], qmax: List[Float64]):
        self.soft = soft
        self.hard = hard
        self.q = q.copy()
        self.qmin = qmin.copy()
        self.qmax = qmax.copy()


def _roll[
    FNET: Module, BNET: Module, ANET: Module
](
    mut t: FBTrainer[FNET, BNET, ANET, OBS, ACT, D, BATCH, "cpu"],
    mut env: UnitreeG1[DType.float64],
    ref rsi: G1RsiTable,
    ref norm: Optional[ObsNorm[OBS]],
    start_row: Int,
    ref z: List[Scalar[DT]],
    z_off: Int,
    ref terms: List[G1Term],
    ref los: List[Float64],
    ref his: List[Float64],
    ref wid: List[Float64],
    skip_from: Int,
    horizon: Int,
    render: Bool,
    frame_delay_ms: Int,
    mut obs_t: Tensor,
    mut z1: Tensor,
    mut act_out: Tensor,
    mut qp: List[Float64],
    mut qv: List[Float64],
) raises -> Roll:
    """Roll `z` and score the compound on what the robot DID.

    `skip_from` deletes every term from that index on — the scaffold arm.
    Goal terms are always appended LAST, so one index removes all of them.
    -1 keeps the whole compound.

    ⚠ Identical reset for every candidate (`set_state` + `G1ActorObs.reset()`)
    or the search ranks start states rather than prompts — the actor's 401-dim
    history is part of the state. Same rule as §12.49 to §12.51.
    """
    var base = start_row * (G1_RSI_NQ + G1_RSI_NV)
    for i in range(NQ):
        qp[i] = Float64(rsi.rows.data[base + i])
    for i in range(NV):
        qv[i] = Float64(rsi.rows.data[base + G1_RSI_NQ + i])
    env.set_state(qp, qv)
    var aobs = G1ActorObs()
    for k in range(D):
        z1.data[k] = z[z_off + k]

    var acc = List[Float64](length=G1_NVOC, fill=0.0)
    var qmin = List[Float64](length=G1_NVOC, fill=1e30)
    var qmax = List[Float64](length=G1_NVOC, fill=-1e30)
    var one = List[Float64](length=G1_NVOC, fill=0.0)
    var s_soft = 0.0
    var s_hard = 0.0
    var n = 0
    for step in range(horizon):
        var o = env.get_obs_list()
        aobs.fill[OBS=OBS](o, obs_t)
        if norm:
            norm.value().apply_row(obs_t)
        t.act[1](obs_t, z1, act_out)
        var a = ContAction[ACT]()
        for k in range(ACT):
            var v = Float64(act_out.data[k])
            if v > 1.0:
                v = 1.0
            elif v < -1.0:
                v = -1.0
            a.data[k] = v
        aobs.push(o, act_out)
        _ = env.step(a)
        if render:
            env.render_frame()
            if frame_delay_ms > 0:
                env.renderer_delay(frame_delay_ms)
        if step >= horizon - HOLD:
            var o2 = env.get_obs_list()
            g1_quantities(o2, 0, one, 0)
            for c in range(G1_NVOC):
                acc[c] += one[c]
                if one[c] < qmin[c]:
                    qmin[c] = one[c]
                if one[c] > qmax[c]:
                    qmax[c] = one[c]
            var ps = 1.0
            var ph = 1.0
            for ti in range(len(terms)):
                if skip_from >= 0 and ti >= skip_from:
                    continue
                var tm = terms[ti].copy()
                var x = one[tm.q]
                ps *= g1_term_soft(tm, los[ti], his[ti], wid[ti], x)
                ph *= g1_term_value(tm, los[ti], his[ti], x)
            s_soft += ps
            s_hard += ph
            n += 1
    for c in range(G1_NVOC):
        acc[c] = acc[c] / Float64(n)
    return Roll(s_soft / Float64(n), s_hard / Float64(n), acc, qmin, qmax)


def _add(
    mut names: List[String], mut groups: List[String], mut ngoal: List[Int],
    mut terms: List[List[G1Term]],
    name: String, group: String, p: Int, ref ts: List[G1Term],
):
    names.append(name)
    groups.append(group)
    ngoal.append(p)
    terms.append(ts.copy())


def _scaffold_stand() -> List[G1Term]:
    """Upright, head at full height, near-still. Motivo's "standing" is
    locomotion with the speed target set to zero, and every one of their
    static tasks carries it."""
    var v = List[G1Term]()
    v.append(G1Term(QV_UPRIGHT, OP_GT, 0.90, 0.0, False))
    v.append(G1Term(QV_HEAD_H, OP_GT, 0.50, 0.0, True))
    v.append(G1Term(QV_SPEED, OP_LT, 0.0, 0.30, False))
    return v^


def _scaffold_low() -> List[G1Term]:
    """Upright and still, with NO head-height term.

    ⚠ `_scaffold_stand` cannot wrap a low posture: its `head_height > p50`
    is 1.14 m and a squat puts the head far below that, so the product is
    empty — squat and crouch both came back at ESS 0 with it. Motivo keeps
    high and low tasks in separate categories for this reason.
    """
    var v = List[G1Term]()
    v.append(G1Term(QV_UPRIGHT, OP_GT, 0.90, 0.0, False))
    v.append(G1Term(QV_SPEED, OP_LT, 0.0, 0.30, False))
    return v^


def _scaffold_move() -> List[G1Term]:
    """Off the ground, and NOT upright.

    ⚠ NO UPRIGHTNESS TERM. `upright > 0.9` contradicts fast locomotion —
    running pitches the torso forward, and with it in place `run` reached
    3.87 m/s against a goal of > 1.5 and still scored a hard 0.000, because
    the goal held and the scaffold did not. Motivo's locomotion category
    constrains only the head z: "in high locomotion tasks, we constrain the
    head z-coordinate to be above a threshold".
    """
    var v = List[G1Term]()
    v.append(G1Term(QV_HEAD_H, OP_GT, 0.10, 0.0, True))
    return v^


def _scaffold_rotate() -> List[G1Term]:
    """Upright and off the ground — for turning in place. Motivo's rotation
    category keeps the alignment term, which they describe as "crucial to
    prevent unwanted movement in other directions", plus a minimum pelvis
    height. A spin does not lean the way a sprint does."""
    var v = List[G1Term]()
    v.append(G1Term(QV_UPRIGHT, OP_GT, 0.90, 0.0, False))
    v.append(G1Term(QV_BODY_H, OP_GT, 0.50, 0.0, True))
    return v^


def main() raises:
    seed(31)
    var ckpt = _flag(String("--ckpt"), String(""))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var out_path = _flag(String("--out"), String("g1_command_bank.txt"))
    var horizon = atol(_flag(String("--horizon"), String(150)))
    var start_clip = atol(_flag(String("--start-clip"), String(13)))
    var n_pool = atol(_flag(String("--pool"), String(65536)))
    # ⚠⚠ 20 x 24, NOT 8 x 16, BECAUSE THAT IS WHAT BUILT THE SHIPPED BANK —
    # its own header records `cem 20x24`. A re-run at the smaller default
    # REJECTED `run` and `back` at hold 0.000, and §12.52's title is "`run`
    # and `back` recovered": they are the two entries that NEED the larger
    # search. `run`'s row shows the shape of it — soft gain **806 %** with
    # hardCEM still 0.000, so CEM was climbing hard and simply ran out of
    # iterations before the indicator held.
    #
    # A default that does not reproduce the artefact in the tree is a trap:
    # the rebuild looks like it worked, the count comes out the same, and two
    # commands are quietly different.
    var iters = atol(_flag(String("--cem-iters"), String(20)))
    var pop = atol(_flag(String("--cem-pop"), String(24)))
    var elites = atol(_flag(String("--cem-elites"), String(4)))
    # ⚠⚠ 0.60, NOT 0.30, AND THIS IS THE FIX FOR `run` (§12.63). At 0.30 the
    # search NEVER VISITS the region where `run`'s compound holds: `hard` was
    # 0.000 for all 480 candidates, which is why adding it to the objective
    # changed nothing. Measured on `--only run`, everything else equal:
    #
    #     sigma 0.15   hold 0.000   soft gain  334 %
    #     sigma 0.30   hold 0.000   soft gain 1111 %   <- the old default
    #     sigma 0.60   hold 1.000   soft gain 1712 %   shipped 2.494
    #     sigma 1.00   hold 1.000   soft gain 1712 %   shipped 1.968
    #
    # `z` is on the radius-sqrt(D) sphere and sigma is per-component, so the
    # step norm is about sigma * sqrt(256) = 16 * sigma — 0.60 is 60 % of the
    # radius. 1.00 is a full-radius step, essentially a random direction, and
    # it lands nearer the band edge (1.968 against 0.60's 2.494), so 0.60 is
    # the choice rather than "as large as possible".
    #
    # A 24-command regression at 20x24 confirmed no cost: ACCEPTED 23 of 24
    # with every accepted row holding at 1.000, `back` 0.850 -> 1.000 and
    # `right_foot_height_hi` 0.675 -> 1.000, and nothing regressed.
    var sigma0 = Float64(String(_flag(String("--cem-sigma"), String("0.60"))))
    var min_ess = Float64(String(_flag(String("--min-ess"), String("100"))))
    var min_hard = Float64(String(_flag(String("--min-hard"), String("0.25"))))
    # ⚠ `--only NAME` builds ONE command. Iterating on a single rejected
    # entry cost 25 minutes a try without it, which is how a threshold ends
    # up tuned by patience rather than by evidence.
    var only = _flag(String("--only"), String(""))
    # ⚠ PROMOTION RUNS THE SAME FOUR GATES AS EVERY HAND-WRITTEN ENTRY. A
    # promoted row that skipped the scaffold-control arm or the hold would be
    # a bank entry with none of the properties a bank entry is relied on for,
    # and `g1_decide` offers them all to the model as equivalent. The only
    # difference is where the terms came from.
    var promote = _flag(String("--promote"), String(""))
    # see the guards at the write site — this is the only way past them
    var force_shrink = _has(String("--force-shrink"))
    # ⚠ `--append` (with `--promote`): evaluate ONLY the candidates and ADD the
    # accepted rows to the existing `--out` file, whose rows are left byte for
    # byte. Without it a promotion rebuilds every hand-written command too —
    # hours of CEM to add one row — and a bank that is not this file's own
    # (a project's copy) could not grow at all without being regenerated.
    var append_mode = _has(String("--append"))
    if append_mode and promote == "":
        raise Error("--append adds PROMOTED rows; pass --promote <candidates>")
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")

    print("=" * 104)
    print("BFM-Zero G1 — building the command bank (§12.52)")
    print("=" * 104)
    print("  gates: ESS >=", min_ess, "rows | scaffold delta >= 15% of pool spread | hard hold >=", min_hard)
    print("  CEM", iters, "x", pop, ",", elites, "elites, sigma", sigma0, "— NETWORK FROZEN")

    var t = Trainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")
    if not norm:
        print("  ⚠ no .norm sidecar beside the checkpoint: RAW inputs.")

    var store = TrajectoryStore(store_path)
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    var rsi = G1RsiTable.from_store(store)
    var n_rows = len(st) // UNITREE_G1_STATE_DIM
    if n_pool > n_rows:
        n_pool = n_rows
    var stride = n_rows // n_pool

    # ── the pool, encoded in CHUNKs ───────────────────────────────────
    # ⚠ `backward_embed` takes its row count as a COMPTIME parameter, which is
    # why §12.50 ran 4096 rows (0.9 % of the store) and why its products
    # starved. Chunking is what makes the pool size a runtime flag.
    var b_list = List[Scalar[DT]](length=n_pool * D, fill=Scalar[DT](0))
    var qv = List[Float64](length=n_pool * G1_NVOC, fill=0.0)
    var row = List[Float64](length=SP, fill=0.0)
    var chunk_t = Tensor.alloc(CHUNK * OBS)
    var b_chunk = Tensor()
    var done = 0
    while done < n_pool:
        var m = CHUNK if done + CHUNK <= n_pool else n_pool - done
        for i in range(CHUNK * OBS):
            chunk_t.data[i] = Scalar[DT](0.0)
        for j in range(m):
            var r = (done + j) * stride
            for k in range(UNITREE_G1_STATE_DIM):
                row[k] = Float64(st[r * UNITREE_G1_STATE_DIM + k])
            for k in range(UNITREE_G1_PRIV_DIM):
                row[UNITREE_G1_STATE_DIM + k] = Float64(
                    pv[r * UNITREE_G1_PRIV_DIM + k]
                )
            # RAW, before the normaliser — the reward is about physical units
            g1_quantities(row, 0, qv, (done + j) * G1_NVOC)
            for k in range(SP):
                chunk_t.data[j * OBS + k] = Scalar[DT](row[k])
        if norm:
            norm.value().apply_rows(chunk_t, CHUNK)
        t.backward_embed[CHUNK](chunk_t, b_chunk)
        for j in range(m):
            for k in range(D):
                b_list[(done + j) * D + k] = b_chunk.data[j * D + k]
        done += m
    print("  pool", n_pool, "rows (stride", stride, "of", n_rows, ") — B encoded")

    # per-quantity spread, for the scaffold gate and the soft widths
    var qlo = List[Float64](length=G1_NVOC, fill=0.0)
    var qhi = List[Float64](length=G1_NVOC, fill=0.0)
    var col = List[Float64](length=n_pool, fill=0.0)
    for q in range(G1_NVOC):
        for i in range(n_pool):
            col[i] = qv[i * G1_NVOC + q]
        qlo[q] = g1_quantile(col, 0.10)
        qhi[q] = g1_quantile(col, 0.90)

    # ── the bank ──────────────────────────────────────────────────────
    var names = List[String]()
    var groups = List[String]()
    var ngoal = List[Int]()
    var terms = List[List[G1Term]]()

    # locomotion — literal m/s and rad/s bands, so that ESS answers whether
    # LAFAN contains the gait at all rather than being guaranteed support
    var v = _scaffold_move()
    v.append(G1Term(QV_SPEED_FWD, OP_BAND, 0.50, 1.40, False))
    _add(names, groups, ngoal, terms, String("walk"), String("locomotion"), 1, v)
    v = _scaffold_move()
    v.append(G1Term(QV_SPEED_FWD, OP_BAND, 1.50, 3.00, False))
    _add(names, groups, ngoal, terms, String("run"), String("locomotion"), 1, v)
    v = _scaffold_move()
    v.append(G1Term(QV_SPEED_FWD, OP_BAND, -2.50, -0.50, False))
    _add(names, groups, ngoal, terms, String("back"), String("locomotion"), 1, v)
    v = _scaffold_rotate()
    v.append(G1Term(QV_YAW_RATE, OP_GT, 1.00, 0.0, False))
    _add(names, groups, ngoal, terms, String("spin_left"), String("locomotion"), 1, v)
    v = _scaffold_rotate()
    v.append(G1Term(QV_YAW_RATE, OP_LT, 0.0, -1.00, False))
    _add(names, groups, ngoal, terms, String("spin_right"), String("locomotion"), 1, v)
    v = _scaffold_move()
    v.append(G1Term(QV_SPEED_LAT, OP_GT, 0.40, 0.0, False))
    _add(names, groups, ngoal, terms, String("strafe_left"), String("locomotion"), 1, v)
    v = _scaffold_move()
    v.append(G1Term(QV_SPEED_LAT, OP_LT, 0.0, -0.40, False))
    _add(names, groups, ngoal, terms, String("strafe_right"), String("locomotion"), 1, v)
    v = _scaffold_move()
    v.append(G1Term(QV_SPEED_FWD, OP_GT, 0.30, 0.0, False))
    v.append(G1Term(QV_SPEED_LAT, OP_GT, 0.30, 0.0, False))
    _add(names, groups, ngoal, terms, String("diagonal"), String("locomotion"), 2, v)
    # ⚠ `stand` cannot use `_scaffold_stand`: its GOAL is that scaffold's own
    # stillness term, so G2 would compare a term against itself and reject a
    # command that works. Its scaffold is uprightness and head height.
    v = List[G1Term]()
    v.append(G1Term(QV_UPRIGHT, OP_GT, 0.90, 0.0, False))
    v.append(G1Term(QV_HEAD_H, OP_GT, 0.50, 0.0, True))
    v.append(G1Term(QV_SPEED, OP_LT, 0.0, 0.15, False))
    _add(names, groups, ngoal, terms, String("stand"), String("posture"), 1, v)

    # posture
    v = _scaffold_low()
    v.append(G1Term(QV_BODY_H, OP_BAND, 0.40, 0.62, False))
    _add(names, groups, ngoal, terms, String("squat"), String("posture"), 1, v)
    v = _scaffold_low()
    v.append(G1Term(QV_BODY_H, OP_BAND, 0.62, 0.72, False))
    _add(names, groups, ngoal, terms, String("crouch"), String("posture"), 1, v)
    v = _scaffold_stand()
    v.append(G1Term(QV_BODY_H, OP_GT, 0.90, 0.0, True))
    _add(names, groups, ngoal, terms, String("stand_tall"), String("posture"), 1, v)

    # arms — both wrists ALWAYS constrained, which is the §12.51 lesson and
    # Motivo's own design (nine tasks, 3 left x 3 right, never one free)
    v = _scaffold_stand()
    v.append(G1Term(QV_RHAND_H, OP_GT, 0.90, 0.0, True))
    v.append(G1Term(QV_LHAND_H, OP_LT, 0.0, 0.50, True))
    _add(names, groups, ngoal, terms, String("right_hand_up"), String("arms"), 2, v)
    v = _scaffold_stand()
    v.append(G1Term(QV_LHAND_H, OP_GT, 0.90, 0.0, True))
    v.append(G1Term(QV_RHAND_H, OP_LT, 0.0, 0.50, True))
    _add(names, groups, ngoal, terms, String("left_hand_up"), String("arms"), 2, v)
    v = _scaffold_stand()
    v.append(G1Term(QV_LHAND_H, OP_GT, 0.85, 0.0, True))
    v.append(G1Term(QV_RHAND_H, OP_GT, 0.85, 0.0, True))
    _add(names, groups, ngoal, terms, String("both_hands_up"), String("arms"), 2, v)
    v = _scaffold_stand()
    v.append(G1Term(QV_LHAND_LAT, OP_GT, 0.90, 0.0, True))
    v.append(G1Term(QV_RHAND_LAT, OP_GT, 0.90, 0.0, True))
    _add(names, groups, ngoal, terms, String("arms_wide"), String("arms"), 2, v)

    # torso
    v = _scaffold_stand()
    v.append(G1Term(QV_TORSO_YAW, OP_GT, 0.90, 0.0, True))
    _add(names, groups, ngoal, terms, String("look_left"), String("torso"), 1, v)
    v = _scaffold_stand()
    v.append(G1Term(QV_TORSO_YAW, OP_LT, 0.0, 0.10, True))
    _add(names, groups, ngoal, terms, String("look_right"), String("torso"), 1, v)

    # ⚠ combos: Motivo's conflicting-objective category — locomotion wants
    # motion, arm-raising wants stillness. FB-CPR scores 74 % of single-task
    # TD3 on exactly this pairing, so a partial trade here is the expected
    # shape rather than a failure.
    v = _scaffold_move()
    v.append(G1Term(QV_SPEED_FWD, OP_GT, 0.40, 0.0, False))
    v.append(G1Term(QV_RHAND_H, OP_GT, 0.90, 0.0, True))
    _add(names, groups, ngoal, terms, String("walk_right_hand_up"), String("combo"), 2, v)
    v = _scaffold_move()
    v.append(G1Term(QV_SPEED_FWD, OP_GT, 0.40, 0.0, False))
    v.append(G1Term(QV_LHAND_H, OP_GT, 0.85, 0.0, True))
    v.append(G1Term(QV_RHAND_H, OP_GT, 0.85, 0.0, True))
    _add(names, groups, ngoal, terms, String("walk_both_hands_up"), String("combo"), 3, v)
    v = _scaffold_rotate()
    v.append(G1Term(QV_YAW_RATE, OP_GT, 1.00, 0.0, False))
    v.append(G1Term(QV_LHAND_LAT, OP_LT, 0.0, 0.25, True))
    v.append(G1Term(QV_RHAND_LAT, OP_LT, 0.0, 0.25, True))
    _add(names, groups, ngoal, terms, String("spin_arms_in"), String("combo"), 3, v)

    # ── promoted candidates: the demo's own specs, same gates ─────────
    # ⚠ APPENDED TO THE SAME LIST, so they go through G1 (pool support), G2
    # (the scaffold-control arm), the CEM and G3 (the rollout hold) exactly
    # as the hand-written entries above do. The ONLY difference is where the
    # terms came from — which is the point: a promoted row has to earn its
    # place, because `g1_decide` offers every bank entry to the model as
    # though they were equivalent.
    var existing = List[String]()
    # ⚠ THE HAND-WRITTEN ENTRIES STAY IN THE LIST under `--append` and are
    # SKIPPED in the loop, not removed: CEM is seeded `31 + c` per command, so
    # a list holding only the candidates would shift every index and give each
    # candidate a different seed — results that match neither a full run nor
    # another `--append` run, and nothing would say so.
    var n_hand = len(names)
    if append_mode:
        var prev_txt = string_from_bytes(read_file_bytes(out_path))
        var prev_ls = prev_txt.split("\n")
        for i in range(len(prev_ls)):
            var pl = String(prev_ls[i])
            if pl.startswith("name "):
                existing.append(String(pl.split(" ")[1]))
        print("  --append: only the candidates; ", len(existing), "rows already in", out_path)
    if promote != "":
        var cands = g1_spec_load_candidates(promote)
        print("  promoting", len(cands), "candidates from", promote)
        for i in range(len(cands)):
            # ⚠ a name that already exists would be a SECOND row with the
            # same name, and `bank.find` returns the FIRST — so the robot
            # would run whichever happened to be written first, silently.
            var clash = False
            for j in range(len(names)):
                if names[j] == cands[i].name:
                    clash = True
            for j in range(len(existing)):
                if existing[j] == cands[i].name:
                    clash = True
            if clash:
                print("    skipped `" + cands[i].name
                      + "` — that name is already in the bank")
                continue
            print("    +", cands[i].name, "(", cands[i].group, ",",
                  cands[i].n_goal, "goal terms )")
            _add(names, groups, ngoal, terms, cands[i].name.copy(),
                 cands[i].group.copy(), cands[i].n_goal, cands[i].terms)

    var ncmd = len(names)
    print("  ", ncmd, "candidate commands")

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var qp = List[Float64](length=NQ, fill=0.0)
    var qvel = List[Float64](length=NV, fill=0.0)
    var start_row = Int(rsi.ep_offset.data[start_clip])

    var prod = List[Float64](length=n_pool, fill=0.0)
    var scaf = List[Float64](length=n_pool, fill=0.0)
    var rew = List[Scalar[DT]](length=n_pool, fill=Scalar[DT](0))
    var zc = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
    var zk = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
    var zb = List[Scalar[DT]](length=D, fill=Scalar[DT](0))

    var entries = String("")
    var n_ok = 0
    # ⚠ WHICH commands were accepted, not just HOW MANY. The write guard
    # compares the SET against what the bank already holds, because a COUNT
    # guard watched 20 -> 20 while `run` and `back` were being replaced by
    # two new entries. A count is not coverage.
    var ok_flag = List[Bool](length=len(names), fill=False)
    var t0 = perf_counter_ns()
    print("-" * 104)
    print("  " + _pad(String("command"), 21) + _pad(String("group"), 12)
          + _lpad(String("ESS"), 7) + _lpad(String("scaf->cmd"), 20)
          + _lpad(String("hard0"), 8) + _lpad(String("hardCEM"), 9)
          + _lpad(String("soft gain"), 11) + _lpad(String("shipped"), 9) + "  verdict")

    for c in range(ncmd):
        if only != "" and names[c] != only:
            continue
        if append_mode and c < n_hand:
            continue
        # ⚠⚠ RESEED PER COMMAND, OR `--only` MEASURES A DIFFERENT THING THAN
        # THE RUN IT IS MEANT TO STAND IN FOR. CEM draws from one seeded
        # stream, so with a single `seed(31)` at the top a command's random
        # sequence depends on HOW MANY COMMANDS CONSUMED DRAWS BEFORE IT.
        # Measured on `right_foot_height_hi` at the same 8x16 budget:
        #
        #     --only      hold 0.500  ->  REJECT
        #     full run    hold 0.600  ->  ok (cem)
        #
        # Two full runs agreed exactly, so the stream is deterministic; it is
        # the POSITION in it that moved. Every per-command measurement taken
        # with `--only` was therefore measuring a path the real run never
        # takes — which is the whole reason `--only` exists (iterating one
        # rejected entry cost 25 minutes a try without it).
        #
        # Seeding from the command INDEX makes a command's result independent
        # of what ran before it, so `--only NAME` and a full rebuild agree,
        # and a full rebuild stays reproducible.
        seed(31 + c)
        var nt = len(terms[c])
        var los = List[Float64](length=nt, fill=0.0)
        var his = List[Float64](length=nt, fill=0.0)
        var wid = List[Float64](length=nt, fill=0.0)
        for i in range(n_pool):
            prod[i] = 1.0
            scaf[i] = 1.0
        for ti in range(nt):
            var tm = terms[c][ti].copy()
            for i in range(n_pool):
                col[i] = qv[i * G1_NVOC + tm.q]
            los[ti] = g1_quantile(col, tm.lo) if tm.as_pct else tm.lo
            his[ti] = g1_quantile(col, tm.hi) if tm.as_pct else tm.hi
            # ⚠ THE WIDTH MUST MATCH THE EXCURSION, NOT THE DISTRIBUTION.
            # At 0.15 of the pool spread this was 0.225 for a forward speed,
            # and `run`'s policy came out 2 m/s outside its band — where
            # exp(-2.1/0.225) is 1e-4 for every candidate alike. Monotone is
            # not enough; the objective has to still VARY where the search
            # actually is. A band's own width is the right scale for a band,
            # and half the pool spread for a one-sided threshold.
            var sp = qhi[tm.q] - qlo[tm.q]
            if sp < 0.0:
                sp = -sp
            if tm.op == OP_BAND:
                var bw = his[ti] - los[ti]
                wid[ti] = bw if bw > 0.0 else -bw
            else:
                wid[ti] = 0.5 * sp
            if wid[ti] < 1e-6:
                wid[ti] = 1e-6
            for i in range(n_pool):
                var w = g1_term_value(tm, los[ti], his[ti], col[i])
                prod[i] *= w
                if ti < nt - ngoal[c]:
                    scaf[i] *= w
        var ess = g1_ess_rows(prod, n_pool)
        var gq = terms[c][nt - ngoal[c]].q   # the FIRST goal term's quantity
        var spread = qhi[gq] - qlo[gq]
        if spread < 0.0:
            spread = -spread

        var line = String("  ") + _pad(names[c], 21) + _pad(groups[c], 12) \
                   + _lpad(String(Int(ess)), 7)
        # ── G1: support ───────────────────────────────────────────────
        if ess < min_ess:
            print(line + _lpad(String("-"), 20) + _lpad(String("-"), 8)
                  + _lpad(String("-"), 9) + _lpad(String("-"), 11)
                  + "  REJECT: ESS " + String(Int(ess)) + " < " + String(Int(min_ess)))
            continue
        for i in range(n_pool):
            rew[i] = Scalar[DT](prod[i])
        var zcl = z_from_reward[D](b_list, rew, n_pool)
        for k in range(D):
            zc[k] = zcl[k]
        for i in range(n_pool):
            rew[i] = Scalar[DT](scaf[i])
        var zkl = z_from_reward[D](b_list, rew, n_pool)
        for k in range(D):
            zk[k] = zkl[k]

        var r_c = _roll[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, zc, 0, terms[c], los, his, wid, -1,
            horizon, False, 0, obs_t, z1, act_out, qp, qvel,
        )
        # ⚠ the scaffold is rolled with its GOAL TERM DELETED from the score
        # too, or it would be judged against a compound it was never asked to
        # satisfy. Only its achieved QUANTITY is compared.
        var r_k = _roll[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, zk, 0, terms[c], los, his, wid, nt - ngoal[c],
            horizon, False, 0, obs_t, z1, act_out, qp, qvel,
        )
        var delta = r_c.q[gq] - r_k.q[gq]
        var adelta = delta if delta > 0.0 else -delta
        var scol = _f3(r_k.q[gq]) + String(" -> ") + _f3(r_c.q[gq])
        line += _lpad(scol, 20)
        # ── G2: the goal term must do work ────────────────────────────
        if adelta < 0.15 * spread:
            print(line + _lpad(_f3(r_c.hard), 8) + _lpad(String("-"), 9)
                  + _lpad(String("-"), 11) + "  REJECT: scaffold delta "
                  + _f3(adelta) + " < " + _f3(0.15 * spread))
            continue

        # ── CEM over z, on the SMOOTH compound ────────────────────────
        var mean = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
        for k in range(D):
            mean[k] = zc[k]
            zb[k] = zc[k]
        var best = r_c.soft
        var cand = List[Scalar[DT]](length=pop * D, fill=Scalar[DT](0))
        var score = List[Float64](length=pop, fill=0.0)
        var sigma = sigma0
        for _it in range(iters):
            var ct = Tensor.alloc(pop * D)
            for p2 in range(pop):
                for k in range(D):
                    ct.data[p2 * D + k] = Scalar[DT](
                        Float64(mean[k]) + sigma * _gauss()
                    )
                g1_project_z[D](ct, p2)
            for p2 in range(pop):
                for k in range(D):
                    cand[p2 * D + k] = ct.data[p2 * D + k]
                var rr = _roll[FNet, BNet, ANet](
                    t, env, rsi, norm, start_row, cand, p2 * D,
                    terms[c], los, his, wid, -1, horizon, False, 0,
                    obs_t, z1, act_out, qp, qvel,
                )
                score[p2] = rr.soft
                if rr.soft > best:
                    best = rr.soft
                    for k in range(D):
                        zb[k] = cand[p2 * D + k]
            # elites = the top `elites` by score
            var used = List[Bool](length=pop, fill=False)
            var acc = List[Float64](length=D, fill=0.0)
            for _e in range(elites):
                var bi = -1
                var bs = -1e30
                for p2 in range(pop):
                    if not used[p2] and score[p2] > bs:
                        bs = score[p2]
                        bi = p2
                if bi < 0:
                    break
                used[bi] = True
                for k in range(D):
                    acc[k] += Float64(cand[bi * D + k])
            for k in range(D):
                mean[k] = Scalar[DT](acc[k] / Float64(elites))
            sigma *= 0.85

        var r_b = _roll[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, zb, 0, terms[c], los, his, wid, -1,
            horizon, False, 0, obs_t, z1, act_out, qp, qvel,
        )
        var gain = 0.0
        if r_c.soft > 1e-12:
            gain = 100.0 * (r_b.soft - r_c.soft) / r_c.soft
        var kept = String("cem")
        var hard_ship = r_b.hard
        if r_c.hard > r_b.hard:
            for k in range(D):
                zb[k] = zc[k]
            hard_ship = r_c.hard
            kept = String("zero-shot")
        line += _lpad(_f3(r_c.hard), 8) + _lpad(_f3(hard_ship), 9) \
                + _lpad(_f3(gain) + String("%"), 11)
        # ── G3: it must actually hold ─────────────────────────────────
        if hard_ship < min_hard:
            print(line + "  REJECT: holds " + _f3(hard_ship) + " < " + _f3(min_hard))
            if only != "":
                for ti in range(nt):
                    var td = terms[c][ti].copy()
                    print("      " + g1_vocab_name(td.q) + " ranged "
                          + _f3(r_b.qmin[td.q]) + " .. " + _f3(r_b.qmax[td.q])
                          + "  (want " + g1_term_str(td, los[ti], his[ti]) + ")")
            continue
        # ⚠ G4: `hard` is the fraction of STEPS that satisfy the compound, and
        # a command can clear a fractional bar while its MEAN behaviour breaks
        # its own definition — `back` shipped at -2.65 m/s against a band of
        # [-2.50, -0.50], in band a quarter of the time and out of it on
        # average. A bank that refuses placebos has to refuse that too.
        var ship = r_b.copy() if kept == "cem" else r_c.copy()
        var mean_ok = True
        var why = String("")
        for ti in range(nt):
            var tm2 = terms[c][ti].copy()
            if tm2.op == OP_SOFT:
                continue
            var x = ship.q[tm2.q]
            var okt = True
            if tm2.op == OP_GT:
                okt = x > los[ti]
            elif tm2.op == OP_LT:
                okt = x < his[ti]
            else:
                okt = x > los[ti] and x < his[ti]
            if not okt:
                mean_ok = False
                why = g1_vocab_name(tm2.q) + String(" = ") + _f3(x)
        if not mean_ok:
            print(line + "  REJECT: the shipped MEAN breaks its own compound (" + why + ")")
            continue
        print(line + _lpad(_f3(ship.q[gq]), 9) + "  ok (" + kept + ")")
        if only != "":
            print("      per-step range of " + g1_vocab_name(gq) + ": "
                  + _f3(ship.qmin[gq]) + " .. " + _f3(ship.qmax[gq]))
        n_ok += 1
        ok_flag[c] = True

        entries += String("name ") + names[c] + String("\n")
        entries += String("group ") + groups[c] + String("\n")
        entries += String("terms ") + String(nt) + String("\n")
        for ti in range(nt):
            var tm = terms[c][ti].copy()
            entries += String("term ") + String(tm.q) + String(" ") + String(tm.op) \
                       + String(" ") + String(los[ti]) + String(" ") + String(his[ti]) \
                       + String("  # ") + g1_term_str(tm, los[ti], his[ti]) \
                       + (String("  <- GOAL") if ti >= nt - ngoal[c] else String("")) \
                       + String("\n")
        entries += String("ess ") + String(Int(ess)) + String("\n")
        entries += String("scaffold ") + String(gq) + String(" ") + String(r_k.q[gq]) \
                   + String(" ") + String(r_c.q[gq]) + String("\n")
        entries += String("hard ") + String(r_c.hard) + String(" ") + String(hard_ship) \
                   + String(" ") + kept + String("\n")
        entries += String("achieved")
        for q in range(G1_NVOC):
            entries += String(" ") + String(r_b.q[q])
        entries += String("\n")
        entries += String("z")
        for k in range(D):
            entries += String(" ") + String(Float64(zb[k]))
        entries += String("\n")

    print("-" * 104)
    print("  ACCEPTED", n_ok, "of", ncmd, "— elapsed",
          Float64(perf_counter_ns() - t0) / 1e9, "s")
    var head = String("# g1 command bank v1\n")
    head += String("# ckpt ") + ckpt + String("\n")
    head += String("# pool ") + String(n_pool) + String(" stride ") + String(stride) + String("\n")
    # ⚠ THE HEADER MUST RECORD SIGMA. It recorded iters, pop and elites but
    # not sigma, and sigma is the one that decides whether `run` is in the
    # bank at all. §12.62's lesson was that a default which cannot rebuild
    # the artefact beside it is a trap; an artefact that does not record what
    # built it is the same trap from the other side.
    head += String("# sigma ") + String(sigma0) + String("\n")
    head += String("# cem ") + String(iters) + String("x") + String(pop) \
            + String(" elites ") + String(elites) + String("\n")
    head += String("# gates ess>=") + String(Int(min_ess)) \
            + String(" scaffold>=15% hard>=") + String(min_hard) + String("\n")
    head += String("count ") + String(n_ok) + String(" ") + String(D) + String("\n")
    # ⚠ THE START POSE TRAVELS WITH THE BANK. Without it a consumer has to
    # open the 1.69 GB LAFAN store for ONE row, which cost the viewer ~14 s
    # of startup — long enough that commands sent in the meantime were
    # dropped before the loop began polling.
    head += String("start_clip ") + String(start_clip) + String("\n")
    head += String("qpos")
    for i in range(NQ):
        head += String(" ") + String(Float64(rsi.rows.data[start_row * (G1_RSI_NQ + G1_RSI_NV) + i]))
    head += String("\nqvel")
    for i in range(NV):
        head += String(" ") + String(Float64(rsi.rows.data[start_row * (G1_RSI_NQ + G1_RSI_NV) + G1_RSI_NQ + i]))
    head += String("\n")
    # ⚠⚠ THIS WRITE DESTROYED THE BANK ONCE AND MUST NOT BE ABLE TO AGAIN.
    # `--only right_foot_height_hi --promote ...` evaluated ONE command, the
    # other 23 were skipped, and the writer emitted only the accepted ones —
    # so a 20-command bank gated at hold 1.000 was replaced by a 9-line file
    # with NO entries. It was recoverable from git; the viewer, `g1say` and
    # the probe all read this file, and `G1CommandBank.load` raises on no
    # `count` header, so every one of them would have failed to start.
    #
    # Two guards, and the second is the one that generalises:
    if only != "":
        print("  NOT written —", out_path, "is unchanged. `--only` evaluates"
              " ONE command, so writing would delete every other entry."
              " Re-run without `--only` to rebuild the bank.")
    elif append_mode:
        # the existing rows stay byte for byte; only `count` changes
        if n_ok == 0:
            print("  NOT written — no candidate was accepted;", out_path, "is unchanged")
        else:
            var prev_txt2 = string_from_bytes(read_file_bytes(out_path))
            var prev_ls2 = prev_txt2.split("\n")
            var out_txt = String("")
            var n_prev = 0
            for i in range(len(prev_ls2)):
                var pl = String(prev_ls2[i])
                if pl.startswith("count "):
                    var pp = pl.split(" ")
                    n_prev = atol(String(pp[1]))
                    if len(pp) < 3 or atol(String(pp[2])) != D:
                        raise Error("--append: " + out_path + " is not a "
                                    + String(D) + "-wide bank")
                    pl = String("count ") + String(n_prev + n_ok) + String(" ") + String(D)
                if i + 1 < len(prev_ls2) or pl.byte_length() > 0:
                    out_txt += pl + String("\n")
            out_txt += String("# appended by --append from ") + promote + String(": cem ") \
                       + String(iters) + String("x") + String(pop) + String(" elites ") \
                       + String(elites) + String(", same gates\n")
            write_text_atomic(out_path, out_txt + entries)
            print("  appended", n_ok, "rows to", out_path, "— now", n_prev + n_ok)
    else:
        # ⚠ NEVER SHRINK SILENTLY. A rebuild that accepts fewer commands than
        # the file already holds is far more likely to be a mistake (a bad
        # flag, a missing artefact, a threshold typo) than an intention, and
        # the cost of being wrong is asymmetric: a refused write loses a
        # 25-minute run, an accepted one loses commands that took four gates
        # each to earn.
        # ⚠⚠ THE SET, NOT THE COUNT — and the count version was MEASURED
        # FAILING. A promote run rejected `run` and `back` while accepting
        # two new foot commands, so 20 went to 20, the count guard saw
        # nothing, and the bank silently LOST the command that §12.56's
        # "plus vite" resolves to. A count is not coverage.
        var prior = List[String]()
        try:
            var prev = string_from_bytes(read_file_bytes(out_path))
            var pls = prev.split("\n")
            for i in range(len(pls)):
                var pl = String(pls[i])
                if pl.startswith("name "):
                    var pp = pl.split(" ")
                    if len(pp) >= 2:
                        prior.append(String(pp[1]))
        except:
            prior = List[String]()
        var lost = String("")
        for i in range(len(prior)):
            var still = False
            for c in range(ncmd):
                if names[c] == prior[i] and ok_flag[c]:
                    still = True
            if not still:
                if lost != "":
                    lost += String(" ")
                lost += prior[i]
        var had = len(prior)
        if lost != "" and not force_shrink:
            print("  NOT written —", out_path, "holds", had,
                  "commands and this run would DROP:", lost)
            print("    A command already in the bank passed four gates to get"
                  " there, so losing one is far more likely to be a smaller"
                  " CEM budget or a missing artefact than an intention.")
            print("    This run used CEM", iters, "x", pop,
                  "; the shipped bank's own header records what built it.")
            print("    Pass --force-shrink if dropping them is really what"
                  " you want.")
        else:
            write_text_atomic(out_path, head + entries)
            print("  wrote", out_path, "—", n_ok, "commands (was", had, ")")
