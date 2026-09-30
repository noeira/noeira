# +--------------------------------------------------------------------------+ #
# | Motivo-style COMPOUND rewards: is it the methodology or the data? (§12.51)
# +--------------------------------------------------------------------------+ #
""""Raise your right hand" raised both hands and walked away. Why?

    pixi run mojo build -I . -Xlinker -ld_classic \
        examples/g1/bfm_zero_reward_vocab.mojo -o build/g1vocab
    ./build/g1vocab --ckpt runs/<id>/checkpoints/step_36000.ckpt
    ./build/g1vocab --ckpt ... --render        # watch single vs compound

## The diagnosis this probe tests

§12.50 prompted ONE privileged column per command and passed its own numeric
gate 4 of 4 — and the rendered result was wrong in three specific ways: the
raised arm brought the other arm up with it, the robot walked instead of
standing, and the squat was a forward pitch rather than a squat.

That is not a model failure. `z = E_rho[r(s) B(s)]` with an indicator `r` is
EXACTLY the conditional mean of `B` over the states that satisfy it, so

    every DoF left unconstrained comes back as the dataset's conditional
    average GIVEN the ones that were constrained.

The LAFAN states with a high right wrist are dance and fight frames, where
the left arm is also up and the body is moving. The robot answered the
question that was asked. Meta Motivo's paper says the same of its own `Run`
task: they specify only high speed and get forward running "probably because
the majority of run motions in M show this behavior" (arXiv 2412.10062,
App. D.3.2).

## What Motivo's rewards actually look like

Not one of them is a single term. Their "most complex task", verbatim:

    r = I[up > 0.9] * I[z_head > 1.4] * exp(-sqrt(vx^2+vy^2)) * I[z_rankle > b]

Three of those four terms exist only to suppress the failure modes above.
Their arm-raising category constrains BOTH wrists independently (nine tasks,
3 left x 3 right) and adds "maintain a standing position"; their rotation
category carries a body-alignment term they describe as "crucial to prevent
unwanted movement in other directions". Bands and indicators throughout, not
Gaussians. Their action-penalty term is inert for us — `B` sees only states,
which their own footnote 12 makes explicit — so it is not reproduced here.

## ⚠ The confound that must be controlled before blaming LAFAN

A product of indicators is an AND of sets, and the effective sample size
collapses fast. §12.50 ran a pool of **4096 rows, stride 107 — 0.9 % of the
store.** At 1 % effective that is FORTY rows behind the prompt. The same 1 %
of 65 536 rows is 655.

So every compound is evaluated at BOTH pool sizes and the ESS of every term
AND of the product is printed. Blaming a 2.45 h dataset for what a 1-in-107
subsample caused would be the easiest available mistake.

    ESS healthy + behaviour fixed  -> methodology. Build the vocabulary.
    ESS ~ 0 at 65 536              -> the data. LAFAN has no such frame,
                                      and we know it for a nameable reason.

## ⚠ One definition, read twice

`_quantities` is used BOTH to score the pool (which builds the reward) and to
measure the rollout (which judges it). If the reward said "right wrist" and
the metric read a neighbouring body, every compound would look like it
worked. One function, two call sites — never a second transcription.
"""

from std.math import exp, sqrt, abs, acos, cos as _cos64, log as _log64
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
from noeira.io.fileio import write_text_atomic
from noeira.deep_agents.fb.z_sampler import z_from_reward
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTowerFiltered, BFMBNetFiltered,
)
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_rsi import G1RsiTable, G1_RSI_NQ, G1_RSI_NV
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA, G1ActorObs,
)
from noeira.envs.robots.unitree_g1_priv_obs import (
    G1_PRIV_OFF_HEIGHT, G1_PRIV_OFF_POS, G1_PRIV_OFF_ROT,
    G1_PRIV_OFF_VEL, G1_PRIV_OFF_ANGVEL,
)
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
    UNITREE_G1_PRIV_DIM,
)
from noeira.envs.robots.g1_tracking_eval import (
    G1_D, G1_H, G1_L, G1_HB, g1_project_z,
)

comptime SP: Int = UNITREE_G1_OBS_DIM
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = G1_D
comptime BATCH: Int = 64
comptime NQ = UnitreeG1Model.NQ
comptime NV = UnitreeG1Model.NV
comptime HOLD: Int = 40
# ⚠ `backward_embed` takes its row count as a COMPTIME parameter, so the pool
# is encoded in fixed-size CHUNKs and accumulated. That is what lets the pool
# size itself be a runtime flag — which is the whole point of this probe.
comptime CHUNK: Int = 4096

# ── the named vocabulary ──────────────────────────────────────────────────
# `g1_skeleton_body` order: 0 pelvis, 1-6 left leg, 7-12 right leg,
# 13-15 waist/torso, 16-22 left arm, 23-29 right arm, 30 the virtual head.
# ⚠ THE POS BLOCK DROPS THE ROOT: body s is at `1 + (s-1)*3` there, and at
# `s*6` / `s*3` in ROT / VEL / ANGVEL. Reading a neighbouring body returns a
# perfectly plausible wrong `z`, so each index is derived from the skeleton
# constant rather than written out.
comptime SK_ANKLE_L: Int = 6        # left_ankle_roll_link   (model body 7)
comptime SK_ANKLE_R: Int = 12       # right_ankle_roll_link  (model body 17)
comptime SK_TORSO: Int = 15         # torso_link             (model body 24)
comptime SK_WRIST_L: Int = 22       # left_wrist_yaw_link    (model body 31)
comptime SK_WRIST_R: Int = 29       # right_wrist_yaw_link   (model body 39)
comptime SK_HEAD: Int = 30          # the virtual head       (torso + 0.35 z)

comptime P_L: Int = G1_PRIV_OFF_POS   # POS base, body s at P_L + (s-1)*3
comptime R_L: Int = G1_PRIV_OFF_ROT   # ROT base, body s at R_L + s*6
comptime V_L: Int = G1_PRIV_OFF_VEL
comptime W_L: Int = G1_PRIV_OFF_ANGVEL

comptime NVOC: Int = 13
comptime Q_BODY_H: Int = 0
comptime Q_HEAD_H: Int = 1
comptime Q_LHAND_H: Int = 2
comptime Q_RHAND_H: Int = 3
comptime Q_LHAND_LAT: Int = 4
comptime Q_RHAND_LAT: Int = 5
comptime Q_LFOOT_H: Int = 6
comptime Q_RFOOT_H: Int = 7
comptime Q_UPRIGHT: Int = 8
comptime Q_SPEED_FWD: Int = 9
comptime Q_SPEED: Int = 10
comptime Q_YAW_RATE: Int = 11
comptime Q_TORSO_YAW: Int = 12

# term ops
comptime OP_GT: Int = 0        # I[x > lo]
comptime OP_LT: Int = 1        # I[x < hi]
comptime OP_BAND: Int = 2      # I[lo < x < hi]
comptime OP_SOFT: Int = 3      # exp(-lo * |x|)   — Motivo's stillness term

comptime FNet = BFMFTower[OBS, ACT, D, G1_H, G1_L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, G1_HB]
comptime ANet = BFMActorTowerFiltered[
    OBS, UNITREE_G1_STATE_DIM, G1_ACTOR_EXTRA, D, G1_H, G1_L, ACT
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]


def _vocab_name(q: Int) -> String:
    if q == Q_BODY_H:
        return String("body_height")
    if q == Q_HEAD_H:
        return String("head_height")
    if q == Q_LHAND_H:
        return String("left_hand_height")
    if q == Q_RHAND_H:
        return String("right_hand_height")
    if q == Q_LHAND_LAT:
        return String("left_hand_lateral")
    if q == Q_RHAND_LAT:
        return String("right_hand_lateral")
    if q == Q_LFOOT_H:
        return String("left_foot_height")
    if q == Q_RFOOT_H:
        return String("right_foot_height")
    if q == Q_UPRIGHT:
        return String("upright")
    if q == Q_SPEED_FWD:
        return String("body_speed_forward")
    if q == Q_SPEED:
        return String("body_speed")
    if q == Q_YAW_RATE:
        return String("body_angular_velocity_yaw")
    return String("torso_yaw")


@always_inline
def _quantities(ref o: List[Float64], base: Int, mut out: List[Float64], off: Int):
    """The 13 named quantities from one [state 64 | privileged 463] row.

    ⚠ THE ONE DEFINITION. The pool scorer and the rollout metric both call
    this. Heights are ABSOLUTE: the privileged POS block is root-relative in
    the heading frame, and the heading frame is a yaw-only rotation, so z adds
    straight onto the root height. Lateral distances stay in the heading
    frame, which is what makes them mean "out to the side of the robot"
    rather than "along the world y axis".
    """
    var p = base + UNITREE_G1_STATE_DIM
    var root_z = o[p + G1_PRIV_OFF_HEIGHT]
    out[off + Q_BODY_H] = root_z
    out[off + Q_HEAD_H] = root_z + o[p + P_L + (SK_HEAD - 1) * 3 + 2]
    out[off + Q_LHAND_H] = root_z + o[p + P_L + (SK_WRIST_L - 1) * 3 + 2]
    out[off + Q_RHAND_H] = root_z + o[p + P_L + (SK_WRIST_R - 1) * 3 + 2]
    var ly = o[p + P_L + (SK_WRIST_L - 1) * 3 + 1]
    var ry = o[p + P_L + (SK_WRIST_R - 1) * 3 + 1]
    out[off + Q_LHAND_LAT] = ly if ly > 0.0 else -ly
    out[off + Q_RHAND_LAT] = ry if ry > 0.0 else -ry
    out[off + Q_LFOOT_H] = root_z + o[p + P_L + (SK_ANKLE_L - 1) * 3 + 2]
    out[off + Q_RFOOT_H] = root_z + o[p + P_L + (SK_ANKLE_R - 1) * 3 + 2]
    out[off + Q_UPRIGHT] = o[p + R_L + SK_TORSO * 6 + 5]      # torso normal z
    var vx = o[p + V_L + 0]
    var vy = o[p + V_L + 1]
    out[off + Q_SPEED_FWD] = vx
    out[off + Q_SPEED] = sqrt(vx * vx + vy * vy)
    out[off + Q_YAW_RATE] = o[p + W_L + 2]
    out[off + Q_TORSO_YAW] = o[p + R_L + SK_TORSO * 6 + 1]    # torso tangent y


struct Term(Copyable, Movable):
    """One predicate. `as_pct` resolves `lo`/`hi` against the POOL's own
    quantile of that quantity — a literal threshold nobody in the dataset
    satisfies gives a reward of 0 everywhere, and `z_from_reward` still
    returns a unit-norm vector. That failure is silent, so thresholds that
    should follow the data say so."""
    var q: Int
    var op: Int
    var lo: Float64
    var hi: Float64
    var as_pct: Bool

    def __init__(out self, q: Int, op: Int, lo: Float64, hi: Float64, as_pct: Bool):
        self.q = q
        self.op = op
        self.lo = lo
        self.hi = hi
        self.as_pct = as_pct


def _term_str(ref t: Term, lo: Float64, hi: Float64) -> String:
    var n = _vocab_name(t.q)
    if t.op == OP_GT:
        return n + String(" > ") + _f3(lo)
    if t.op == OP_LT:
        return n + String(" < ") + _f3(hi)
    if t.op == OP_BAND:
        return _f3(lo) + String(" < ") + n + String(" < ") + _f3(hi)
    return String("exp(-") + _f3(t.lo) + String(" * |") + n + String("|)")


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


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _has(name: String) raises -> Bool:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            return True
    return False


def _term_value(ref t: Term, lo: Float64, hi: Float64, x: Float64) -> Float64:
    if t.op == OP_GT:
        return 1.0 if x > lo else 0.0
    if t.op == OP_LT:
        return 1.0 if x < hi else 0.0
    if t.op == OP_BAND:
        return 1.0 if (x > lo and x < hi) else 0.0
    var a = x if x > 0.0 else -x
    return exp(-t.lo * a)


def _ess_pct(ref w: List[Float64], n: Int) -> Float64:
    """(sum w)^2 / sum w^2, as a percentage of `n`. The number of pool rows
    actually standing behind the prompt."""
    var s1 = 0.0
    var s2 = 0.0
    for i in range(n):
        s1 += w[i]
        s2 += w[i] * w[i]
    if s2 <= 0.0:
        return 0.0
    return 100.0 * (s1 * s1 / s2) / Float64(n)


def _pct_of(ref col: List[Float64], q: Float64) -> Float64:
    var v = List[Float64](length=len(col), fill=0.0)
    for i in range(len(col)):
        v[i] = col[i]
    sort(v)
    var idx = Int(q * Float64(len(v) - 1) + 0.5)
    if idx < 0:
        idx = 0
    if idx >= len(v):
        idx = len(v) - 1
    return v[idx]


struct Roll(Copyable, Movable):
    var q: List[Float64]

    def __init__(out self, q: List[Float64]):
        self.q = q.copy()


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
    horizon: Int,
    render: Bool,
    frame_delay_ms: Int,
    mut obs_t: Tensor,
    mut z1: Tensor,
    mut act_out: Tensor,
    mut qp: List[Float64],
    mut qv: List[Float64],
) raises -> Roll:
    """Mean of every named quantity over the last HOLD steps.

    ⚠ Identical reset for every prompt (`set_state` + `G1ActorObs.reset()`) or
    the comparison ranks start states rather than prompts — the actor's
    401-dim history is part of the state. Same rule as §12.49 and §12.50.
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

    var acc = List[Float64](length=NVOC, fill=0.0)
    var one = List[Float64](length=NVOC, fill=0.0)
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
            _quantities(o2, 0, one, 0)
            for c in range(NVOC):
                acc[c] += one[c]
            n += 1
    for c in range(NVOC):
        acc[c] = acc[c] / Float64(n)
    return Roll(acc)


def main() raises:
    seed(31)
    var ckpt = _flag(String("--ckpt"), String(""))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var horizon = atol(_flag(String("--horizon"), String(150)))
    var start_clip = atol(_flag(String("--start-clip"), String(13)))
    var small = atol(_flag(String("--pool-small"), String(4096)))
    var big = atol(_flag(String("--pool-big"), String(65536)))
    var render = _has(String("--render"))
    var fps = atol(_flag(String("--fps"), String(50)))
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")

    print("=" * 92)
    print("BFM-Zero G1 — COMPOUND rewards, Motivo style: methodology or data? (§12.51)")
    print("=" * 92)

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
    if big > n_rows:
        big = n_rows
    print("  store", n_rows, "rows; pools", small, "(stride", n_rows // small,
          ") and", big, "(stride", n_rows // big, ")")

    # ── the compounds ─────────────────────────────────────────────────
    # Each carries its Motivo ancestor. `as_pct` thresholds follow the pool;
    # `upright > 0.9` and the stillness constant are Motivo's own literals.
    var names = List[String]()
    var says = List[String]()
    var prim = List[Int]()
    var terms = List[List[Term]]()

    var c0 = List[Term]()
    c0.append(Term(Q_UPRIGHT, OP_GT, 0.90, 0.0, False))
    c0.append(Term(Q_HEAD_H, OP_GT, 0.50, 0.0, True))
    c0.append(Term(Q_SPEED, OP_SOFT, 3.0, 0.0, False))
    c0.append(Term(Q_RHAND_H, OP_GT, 0.90, 0.0, True))
    c0.append(Term(Q_LHAND_H, OP_LT, 0.0, 0.50, True))
    names.append(String("raise_right_hand"))
    says.append(String("Motivo raisearms-l-h + standing"))
    prim.append(3)                      # the right-hand term is the goal
    terms.append(c0^)

    var c1 = List[Term]()
    c1.append(Term(Q_UPRIGHT, OP_GT, 0.90, 0.0, False))
    c1.append(Term(Q_BODY_H, OP_BAND, 0.40, 0.62, False))
    c1.append(Term(Q_SPEED, OP_SOFT, 3.0, 0.0, False))
    names.append(String("squat"))
    says.append(String("Motivo crouch: low root, still UPRIGHT"))
    prim.append(1)
    terms.append(c1^)

    var c2 = List[Term]()
    c2.append(Term(Q_UPRIGHT, OP_GT, 0.90, 0.0, False))
    c2.append(Term(Q_BODY_H, OP_GT, 0.50, 0.0, True))
    c2.append(Term(Q_SPEED, OP_SOFT, 3.0, 0.0, False))
    c2.append(Term(Q_TORSO_YAW, OP_GT, 0.90, 0.0, True))
    names.append(String("look_left"))
    says.append(String("Motivo rotate + its alignment term"))
    prim.append(3)
    terms.append(c2^)

    # ⚠ the composite category: locomotion and arm-raising have CONFLICTING
    # objectives (one wants motion, the other stillness). Motivo reports
    # FB-CPR at 74 % of single-task TD3 on exactly this pairing, so a partial
    # result here is the expected shape, not a failure.
    var c3 = List[Term]()
    c3.append(Term(Q_UPRIGHT, OP_GT, 0.90, 0.0, False))
    c3.append(Term(Q_SPEED_FWD, OP_BAND, 0.60, 0.90, True))
    c3.append(Term(Q_LHAND_H, OP_GT, 0.90, 0.0, True))
    names.append(String("walk_left_hand_up"))
    says.append(String("Motivo WALK-LAM: band on speed x hand high"))
    prim.append(2)
    terms.append(c3^)

    # ⚠ THE SOFT STILLNESS TERM IS WEAK. `exp(-3|v|)` never reaches zero, so
    # a fast state still contributes; the scaffold above runs at 0.65 m/s with
    # it in place. Motivo's standing category is locomotion with the speed
    # TARGET set to 0 — a band, not a decay. This arm swaps one for the other
    # and is otherwise identical to `raise_right_hand`, so the difference is
    # attributable to that single term.
    var c4 = List[Term]()
    c4.append(Term(Q_UPRIGHT, OP_GT, 0.90, 0.0, False))
    c4.append(Term(Q_HEAD_H, OP_GT, 0.50, 0.0, True))
    c4.append(Term(Q_SPEED, OP_LT, 0.0, 0.30, False))
    c4.append(Term(Q_RHAND_H, OP_GT, 0.90, 0.0, True))
    c4.append(Term(Q_LHAND_H, OP_LT, 0.0, 0.50, True))
    names.append(String("raise_right_hand_hard"))
    says.append(String("same, with I[speed < 0.3] instead of exp(-3|v|)"))
    prim.append(3)
    terms.append(c4^)

    var ncmd = len(names)

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var delay_ms = 1000 // fps if fps > 0 else 0
    if render:
        if not env.init_renderer(show_velocity=False):
            print("  ⚠ no renderer available — running headless.")
            render = False
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var qp = List[Float64](length=NQ, fill=0.0)
    var qvel = List[Float64](length=NV, fill=0.0)
    var start_row = Int(rsi.ep_offset.data[start_clip])

    var pools = List[Int]()
    pools.append(small)
    pools.append(big)
    # ess[p][c][term] and the product at index len(terms)
    var ess_all = List[Float64](length=2 * ncmd * 8, fill=-1.0)
    var zc_all = List[Scalar[DT]](length=2 * ncmd * D, fill=Scalar[DT](0))
    var zs_all = List[Scalar[DT]](length=2 * ncmd * D, fill=Scalar[DT](0))
    # ⚠ THE SCAFFOLD ARM: the compound with its GOAL term deleted. Three of
    # these four compounds are mostly "stand upright and still", and a
    # compound that reads well because the scaffold alone reads well has
    # measured nothing. This is the arm that says the goal term does work.
    var zk_all = List[Scalar[DT]](length=2 * ncmd * D, fill=Scalar[DT](0))
    var t0 = perf_counter_ns()

    for pi in range(2):
        var n_pool = pools[pi]
        var stride = n_rows // n_pool
        print("-" * 92)
        print("POOL", n_pool, "rows (stride", stride, ")")

        # ── chunked encode: B over the pool, CHUNK rows at a time ──────
        var b_list = List[Scalar[DT]](length=n_pool * D, fill=Scalar[DT](0))
        var qv = List[Float64](length=n_pool * NVOC, fill=0.0)
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
                # ⚠ RAW, before the normaliser: the reward is about physical
                # quantities, and `_quantities` is the same call the rollout
                # metric makes.
                _quantities(row, 0, qv, (done + j) * NVOC)
                for k in range(SP):
                    chunk_t.data[j * OBS + k] = Scalar[DT](row[k])
            if norm:
                norm.value().apply_rows(chunk_t, CHUNK)
            t.backward_embed[CHUNK](chunk_t, b_chunk)
            for j in range(m):
                for k in range(D):
                    b_list[(done + j) * D + k] = b_chunk.data[j * D + k]
            done += m

        # per-quantity columns, for the percentile thresholds
        var col = List[Float64](length=n_pool, fill=0.0)
        var w = List[Float64](length=n_pool, fill=0.0)
        var prod = List[Float64](length=n_pool, fill=0.0)
        var rew = List[Scalar[DT]](length=n_pool, fill=Scalar[DT](0))

        # ── what the dataset actually CONTAINS ────────────────────────
        # The data half of the question. A command whose band falls outside
        # this table is not a methodology problem and no amount of compounding
        # will reach it — LAFAN is 2.45 h and eight activity words.
        if pi == 1:
            print("  vocabulary over the pool (the reachable range):")
            print("    " + _pad(String("quantity"), 28)
                  + _lpad(String("p02"), 9) + _lpad(String("p10"), 9)
                  + _lpad(String("p50"), 9) + _lpad(String("p90"), 9)
                  + _lpad(String("p98"), 9))
            for q in range(NVOC):
                for i in range(n_pool):
                    col[i] = qv[i * NVOC + q]
                print("    " + _pad(_vocab_name(q), 28)
                      + _lpad(_f3(_pct_of(col, 0.02)), 9)
                      + _lpad(_f3(_pct_of(col, 0.10)), 9)
                      + _lpad(_f3(_pct_of(col, 0.50)), 9)
                      + _lpad(_f3(_pct_of(col, 0.90)), 9)
                      + _lpad(_f3(_pct_of(col, 0.98)), 9))

        for c in range(ncmd):
            print("  " + _pad(names[c], 26) + says[c])
            print("    " + _pad(String("term"), 46) + _lpad(String("ESS%"), 9)
                  + _lpad(String("rows"), 9))
            for i in range(n_pool):
                prod[i] = 1.0
            var scaf = List[Float64](length=n_pool, fill=1.0)
            var nt = len(terms[c])
            for ti in range(nt):
                var tm = terms[c][ti].copy()
                for i in range(n_pool):
                    col[i] = qv[i * NVOC + tm.q]
                var lo = _pct_of(col, tm.lo) if tm.as_pct else tm.lo
                var hi = _pct_of(col, tm.hi) if tm.as_pct else tm.hi
                for i in range(n_pool):
                    w[i] = _term_value(tm, lo, hi, col[i])
                    prod[i] *= w[i]
                    if ti != prim[c]:
                        scaf[i] *= w[i]
                var e = _ess_pct(w, n_pool)
                print("    " + _pad(_term_str(tm, lo, hi), 46) + _lpad(_f3(e), 9)
                      + _lpad(String(Int(e * Float64(n_pool) / 100.0)), 9))
                # the SINGLE-term arm is the primary term on its own
                if ti == prim[c]:
                    for i in range(n_pool):
                        rew[i] = Scalar[DT](w[i])
                    var zsg = z_from_reward[D](b_list, rew, n_pool)
                    for k in range(D):
                        zs_all[(pi * ncmd + c) * D + k] = zsg[k]
            var ep = _ess_pct(prod, n_pool)
            var nrows = Int(ep * Float64(n_pool) / 100.0)
            print("    " + _pad(String("PRODUCT"), 46) + _lpad(_f3(ep), 9)
                  + _lpad(String(nrows), 9)
                  + (String("   ⚠ STARVED") if nrows < 20 else String("")))
            ess_all[(pi * ncmd + c) * 8] = ep
            for i in range(n_pool):
                rew[i] = Scalar[DT](prod[i])
            if ep > 0.0:
                var zcp = z_from_reward[D](b_list, rew, n_pool)
                for k in range(D):
                    zc_all[(pi * ncmd + c) * D + k] = zcp[k]
            var ek = _ess_pct(scaf, n_pool)
            print("    " + _pad(String("scaffold (goal term removed)"), 46)
                  + _lpad(_f3(ek), 9)
                  + _lpad(String(Int(ek * Float64(n_pool) / 100.0)), 9))
            if ek > 0.0:
                for i in range(n_pool):
                    rew[i] = Scalar[DT](scaf[i])
                var zkp = z_from_reward[D](b_list, rew, n_pool)
                for k in range(D):
                    zk_all[(pi * ncmd + c) * D + k] = zkp[k]

    # ── the behaviour, at the BIG pool ────────────────────────────────
    print("=" * 92)
    print("BEHAVIOUR at pool", pools[1], "— single term vs compound")
    var hdr = String("  ") + _pad(String("command / arm"), 26)
    var show = List[Int]()
    show.append(Q_RHAND_H)
    show.append(Q_LHAND_H)
    show.append(Q_BODY_H)
    show.append(Q_SPEED)
    show.append(Q_UPRIGHT)
    show.append(Q_TORSO_YAW)
    for i in range(len(show)):
        hdr += _lpad(_vocab_name(show[i]), 13)
    print(hdr)
    for c in range(ncmd):
        var rs = _roll[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, zs_all, (1 * ncmd + c) * D,
            horizon, False, 0, obs_t, z1, act_out, qp, qvel,
        )
        var rc = _roll[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, zc_all, (1 * ncmd + c) * D,
            horizon, False, 0, obs_t, z1, act_out, qp, qvel,
        )
        var rk = _roll[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, zk_all, (1 * ncmd + c) * D,
            horizon, False, 0, obs_t, z1, act_out, qp, qvel,
        )
        var l1 = String("  ") + _pad(names[c] + String(" single"), 26)
        var starved = ess_all[(1 * ncmd + c) * 8] <= 0.0
        var l2 = String("  ") + _pad(
            names[c] + (String(" COMPOUND") if not starved else String(" COMPOUND(VOID)")), 26
        )
        for i in range(len(show)):
            l1 += _lpad(_f3(rs.q[show[i]]), 13)
            l2 += _lpad(_f3(rc.q[show[i]]), 13)
        var l0 = String("  ") + _pad(names[c] + String(" scaffold"), 26)
        for i in range(len(show)):
            l0 += _lpad(_f3(rk.q[show[i]]), 13)
        print(l1)
        print(l0)
        # ⚠ a starved product never built a `z`, so the vector is all zeros and
        # the rollout below it is a zero-`z` rollout wearing a result's clothes.
        if starved:
            l2 = String("  ") + _pad(names[c] + String(" COMPOUND"), 26) \
                 + String("   NO PROMPT — the product selected 0 pool rows")
        print(l2)
    print("  elapsed", Float64(perf_counter_ns() - t0) / 1e9, "s")

    if render:
        print("-" * 92)
        print("Rendering: single then COMPOUND for each. Escape to skip on.")
        for c in range(ncmd):
            for arm in range(2):
                if env.check_renderer_quit():
                    break
                print("  ", names[c], "SINGLE" if arm == 0 else "COMPOUND")
                if arm == 0:
                    _ = _roll[FNet, BNet, ANet](
                        t, env, rsi, norm, start_row, zs_all, (1 * ncmd + c) * D,
                        horizon * 2, True, delay_ms, obs_t, z1, act_out, qp, qvel,
                    )
                else:
                    _ = _roll[FNet, BNet, ANet](
                        t, env, rsi, norm, start_row, zc_all, (1 * ncmd + c) * D,
                        horizon * 2, True, delay_ms, obs_t, z1, act_out, qp, qvel,
                    )
        env.close()
