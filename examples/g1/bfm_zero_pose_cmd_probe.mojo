# +--------------------------------------------------------------------------+ #
# | Can we command a NON-VELOCITY whole-body quantity? (§12.50)
# +--------------------------------------------------------------------------+ #
""""Raise your right hand", "squat", "look left" — promptable or not?

    pixi run mojo build -I . -Xlinker -ld_classic \
        examples/g1/bfm_zero_pose_cmd_probe.mojo -o build/g1poseprobe
    ./build/g1poseprobe --ckpt runs/<id>/checkpoints/step_36000.ckpt
    ./build/g1poseprobe --ckpt ... --render        # watch the p90 end of each

## The question this decides

`docs/BFM_ZERO_NEXT_LEVEL.md` argues the command vocabulary is analytic: the
privileged vector carries root height plus per-body position, 6-D orientation,
linear and angular velocity for 31 bodies, so `z = E_rho[r(s) B(s)]` should
address any of them with no training. The joystick only ever uses two of those
columns (`local_body_vel` and `local_body_ang_vel` of body 0). Everything in
the ladder above L1 rests on the rest of the table being reachable too.

If the three commands separate here, the vocabulary is open. If they collapse
toward `stand`, the bottleneck is pre-training diversity and the next move is
a longer run on a richer motion set, not a language layer.

## ⚠ Why the angle probe is NOT the gate

The joystick prints the angle between prompts at startup and that is a real
vacuity check — but it is only *necessary*. Two prompts can sit 60 degrees
apart on the sphere and still drive identical behaviour, because nothing
forces `pi_z` to distinguish directions of `z` the pre-training never had to
use. The gate is BEHAVIOURAL, and it is the same one §12.49 used for
velocity: command a quantity, then measure that quantity on the rollout.

So every command is prompted at three levels taken from the POOL'S OWN
percentiles (p10 / p50 / p90) and the decisive number is

    cmd-gain = (achieved@p90 - achieved@p10) / (p90 - p10)

1.0 is perfect tracking, 0.0 is a placebo. Targets are percentiles rather
than round numbers on purpose: a hand-picked target that no pool state is
near produces a reward that is ~0 everywhere, and `z_from_reward` still
returns a unit-norm vector. That failure is silent, so the targets are drawn
from the distribution and the effective sample size is printed beside them.

## ⚠ Two confounds that would fake a pass

1. **A fall is a successful squat.** Root height drops just as well when the
   robot collapses. Every row therefore carries `upright`, the torso's normal
   z in the heading frame (1.0 standing, 0.0 on its face), averaged over the
   same window. A squat with `upright` near zero is a fall.
2. **A quantity the pool barely varies over** gives p10 == p90, so cmd-gain
   divides by ~0. The spread is printed raw next to the ratio.

## The fourth command is a control on the METHOD

`left_knee_z` is not a command anybody wants. It is there because walking
lifts knees, so the pool certainly varies over it: if even the knee reads
vacuous, the fault is in this probe or in the prompt path, not in the
vocabulary. Read that row first.
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
comptime POOL: Int = 4096
comptime HOLD: Int = 40
comptime NCMD: Int = 4
comptime NLEV: Int = 3

# ── the privileged columns, by name ───────────────────────────────────────
# `g1_skeleton_body` order: 0 pelvis, 1-6 left leg, 7-12 right leg,
# 13-15 waist/torso, 16-22 left arm, 23-29 right arm, 30 the virtual head.
# ⚠ THE POS BLOCK DROPS THE ROOT, so body s sits at `1 + (s-1)*3` there and
# at `s*6` / `s*3` in the other three. Getting that wrong reads a NEIGHBOURING
# BODY's column and returns a perfectly plausible wrong `z`.
comptime SK_KNEE_L: Int = 4         # left_knee_link      (model body 5)
comptime SK_TORSO: Int = 15         # torso_link          (model body 24)
comptime SK_WRIST_R: Int = 29       # right_wrist_yaw_link (model body 39)

comptime COL_ROOT_Z: Int = G1_PRIV_OFF_HEIGHT
comptime COL_HAND_R_Z: Int = G1_PRIV_OFF_POS + (SK_WRIST_R - 1) * 3 + 2
comptime COL_KNEE_L_Z: Int = G1_PRIV_OFF_POS + (SK_KNEE_L - 1) * 3 + 2
# rotation is [tangent 3 | normal 3]: tangent y is the torso's yaw against the
# heading frame (the heading frame IS the root yaw, so this is torso-vs-root).
comptime COL_TORSO_TANY: Int = G1_PRIV_OFF_ROT + SK_TORSO * 6 + 1
# normal z: 1.0 upright, 0.0 horizontal — the fall detector.
comptime COL_TORSO_UP: Int = G1_PRIV_OFF_ROT + SK_TORSO * 6 + 5

# the joystick's own two, for the reference prompts the angles are taken against
comptime OFF_VX: Int = UNITREE_G1_STATE_DIM + G1_PRIV_OFF_VEL
comptime OFF_WZ: Int = UNITREE_G1_STATE_DIM + G1_PRIV_OFF_ANGVEL + 2
comptime SIGMA_V: Float64 = 0.5
comptime SIGMA_W: Float64 = 0.5

comptime FNet = BFMFTower[OBS, ACT, D, G1_H, G1_L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, G1_HB]
comptime ANet = BFMActorTowerFiltered[
    OBS, UNITREE_G1_STATE_DIM, G1_ACTOR_EXTRA, D, G1_H, G1_L, ACT
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]


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


def _pct(ref xs: List[Float64], q: Float64) -> Float64:
    """`q`-quantile of a COPY — the caller's column order is the pool order and
    several commands read it again."""
    var v = List[Float64](length=len(xs), fill=0.0)
    for i in range(len(xs)):
        v[i] = xs[i]
    sort(v)
    var idx = Int(q * Float64(len(v) - 1) + 0.5)
    if idx < 0:
        idx = 0
    if idx >= len(v):
        idx = len(v) - 1
    return v[idx]


def _ang(dot: Float64) -> Float64:
    var c = dot / Float64(D)
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0
    return 57.29577951308232 * acos(c)


def _angle_same(ref a: List[Scalar[DT]], ao: Int, bo: Int) -> Float64:
    """Angle between two rows of the SAME list.

    ⚠ This exists because `_angle_deg(zs, i, zs, j)` does not compile: Mojo
    refuses two `ref` arguments that alias. §12.47's `g1_slerp_z` carries the
    same signature for the same reason — one container, two indices.
    """
    var dot = 0.0
    for k in range(D):
        dot += Float64(a[ao + k]) * Float64(a[bo + k])
    return _ang(dot)


def _angle_deg(ref a: List[Scalar[DT]], ao: Int, ref b: List[Scalar[DT]], bo: Int) -> Float64:
    """Angle between rows of two DIFFERENT lists. Both live on the
    radius-sqrt(D) sphere, so the cosine is the dot over D."""
    var dot = 0.0
    for k in range(D):
        dot += Float64(a[ao + k]) * Float64(b[bo + k])
    return _ang(dot)


struct Roll(Copyable, Movable):
    """What one rollout reports: ALL FOUR quantities and the fall detector.

    ⚠ Measuring only the commanded quantity is how a probe passes while the
    commands are not separable. `raise your right hand` and `squat` both score
    if the policy simply does something energetic. Reading all four columns
    from every rollout costs nothing and yields the cross-command matrix,
    which is the only thing that shows the prompts address DIFFERENT
    behaviours rather than one shared "move about" direction.
    """
    var q: List[Float64]
    var upright: Float64

    def __init__(out self, q: List[Float64], upright: Float64):
        # four floats — a copy here is not worth a transfer convention
        self.q = q.copy()
        self.upright = upright


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
    ref cols: List[Int],
    horizon: Int,
    render: Bool,
    frame_delay_ms: Int,
    mut obs_t: Tensor,
    mut z1: Tensor,
    mut act_out: Tensor,
    mut qp: List[Float64],
    mut qv: List[Float64],
) raises -> Roll:
    """Mean of EVERY command column in `cols` and of the torso's upright over
    the last HOLD steps.

    ⚠ Identical reset for every candidate (`set_state` + `G1ActorObs.reset()`),
    or the comparison ranks start states rather than prompts — the actor's
    401-dim history is part of the state. Same rule as §12.49's `_roll`.
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

    var acc = List[Float64](length=len(cols), fill=0.0)
    var acc_up = 0.0
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
            for c in range(len(cols)):
                acc[c] += Float64(o2[UNITREE_G1_STATE_DIM + cols[c]])
            acc_up += Float64(o2[UNITREE_G1_STATE_DIM + COL_TORSO_UP])
            n += 1
    for c in range(len(acc)):
        acc[c] = acc[c] / Float64(n)
    return Roll(acc, acc_up / Float64(n))


def main() raises:
    seed(31)
    var ckpt = _flag(String("--ckpt"), String(""))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var horizon = atol(_flag(String("--horizon"), String(150)))
    var sigma_frac = Float64(String(_flag(String("--sigma-frac"), String("0.25"))))
    var start_clip = atol(_flag(String("--start-clip"), String(13)))
    var render = _has(String("--render"))
    var fps = atol(_flag(String("--fps"), String(50)))
    var emit = _flag(String("--emit"), String(""))
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")

    print("=" * 84)
    print("BFM-Zero G1 — is a NON-VELOCITY quantity promptable? (4 commands x 3 levels)")
    print("=" * 84)
    print("  gate: cmd-gain = (achieved@p90 - achieved@p10) / (p90 - p10)")
    print("  window: last", HOLD, "of", horizon, "steps — NETWORK FROZEN, no training")

    var t = Trainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")
    if not norm:
        print("  ⚠ no .norm sidecar beside the checkpoint: RAW inputs.")

    # ── the pool, built EXACTLY as the joystick builds it ─────────────
    var store = TrajectoryStore(store_path)
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    var rsi = G1RsiTable.from_store(store)
    var n_rows = len(st) // UNITREE_G1_STATE_DIM
    if n_rows < POOL:
        raise Error("store has fewer rows than POOL")
    var stride = n_rows // POOL

    var pool = Tensor.alloc(POOL * OBS)
    # ⚠ the 928-wide row with a zeroed tail: `B` SLICES the first 527. §12.38.
    for i in range(POOL * OBS):
        pool.data[i] = Scalar[DT](0.0)

    var cols = List[Int]()
    cols.append(COL_HAND_R_Z)
    cols.append(COL_ROOT_Z)
    cols.append(COL_TORSO_TANY)
    cols.append(COL_KNEE_L_Z)
    var names = List[String]()
    names.append(String("right_hand_z"))
    names.append(String("root_height"))
    names.append(String("torso_yaw"))
    names.append(String("left_knee_z"))
    var says = List[String]()
    says.append(String("raise your right hand"))
    says.append(String("squat"))
    says.append(String("look left"))
    says.append(String("(method control: walking lifts knees)"))

    # the command features, kept RAW — the reward is about physical quantities,
    # so they are read BEFORE the normaliser touches the pool.
    var feat = List[Float64](length=NCMD * POOL, fill=0.0)
    var pvx = List[Float64](length=POOL, fill=0.0)
    var pvy = List[Float64](length=POOL, fill=0.0)
    var pwz = List[Float64](length=POOL, fill=0.0)
    for i in range(POOL):
        var r = i * stride
        for k in range(UNITREE_G1_STATE_DIM):
            pool.data[i * OBS + k] = Scalar[DT](st[r * UNITREE_G1_STATE_DIM + k])
        for k in range(UNITREE_G1_PRIV_DIM):
            pool.data[i * OBS + UNITREE_G1_STATE_DIM + k] = Scalar[DT](
                pv[r * UNITREE_G1_PRIV_DIM + k]
            )
        for c in range(NCMD):
            feat[c * POOL + i] = Float64(
                pool.data[i * OBS + UNITREE_G1_STATE_DIM + cols[c]]
            )
        pvx[i] = Float64(pool.data[i * OBS + OFF_VX + 0])
        pvy[i] = Float64(pool.data[i * OBS + OFF_VX + 1])
        pwz[i] = Float64(pool.data[i * OBS + OFF_WZ])
    if norm:
        norm.value().apply_rows(pool, POOL)

    var b_pool = Tensor()
    t.backward_embed[POOL](pool, b_pool)
    var b_list = List[Scalar[DT]](length=POOL * D, fill=Scalar[DT](0))
    for i in range(POOL * D):
        b_list[i] = b_pool.data[i]
    print("  pool", POOL, "rows (stride", stride, "of", n_rows, ") — B encoded")

    # ── the three velocity prompts the angles are measured against ────
    var rewards = List[Scalar[DT]](length=POOL, fill=Scalar[DT](0))
    var ref_z = List[Scalar[DT]](length=3 * D, fill=Scalar[DT](0))
    var ref_vx = List[Float64](length=3, fill=0.0)
    var ref_wz = List[Float64](length=3, fill=0.0)
    ref_vx[1] = 1.0
    ref_wz[2] = 1.2
    for c in range(3):
        for i in range(POOL):
            var dx = pvx[i] - ref_vx[c]
            var dy = pvy[i]
            var dw = pwz[i] - ref_wz[c]
            rewards[i] = Scalar[DT](
                exp(-(dx * dx + dy * dy) / (SIGMA_V * SIGMA_V))
                * exp(-(dw * dw) / (SIGMA_W * SIGMA_W))
            )
        var zc = z_from_reward[D](b_list, rewards, POOL)
        for k in range(D):
            ref_z[c * D + k] = zc[k]
    var ref_names = List[String]()
    ref_names.append(String("stand"))
    ref_names.append(String("forward"))
    ref_names.append(String("turn"))

    # ── the env ───────────────────────────────────────────────────────
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
    var qv = List[Float64](length=NV, fill=0.0)
    var start_row = Int(rsi.ep_offset.data[start_clip])

    var qs = List[Float64]()
    qs.append(0.10)
    qs.append(0.50)
    qs.append(0.90)

    var t0 = perf_counter_ns()
    var zs = List[Scalar[DT]](length=NCMD * NLEV * D, fill=Scalar[DT](0))
    var tgt = List[Float64](length=NCMD * NLEV, fill=0.0)
    var ess = List[Float64](length=NCMD * NLEV, fill=0.0)

    # ── PART 1: the prompts, their pool support and their geometry ────
    print("-" * 84)
    print("PART 1 — pool support and prompt geometry")
    print("  " + _pad(String("command"), 14) + _lpad(String("col"), 5)
          + _lpad(String("p10"), 9) + _lpad(String("p50"), 9) + _lpad(String("p90"), 9)
          + _lpad(String("sigma"), 8) + _lpad(String("ESS%"), 7)
          + _lpad(String("ang(10,90)"), 12) + _lpad(String("min-ang-vel"), 13))
    for c in range(NCMD):
        var colv = List[Float64](length=POOL, fill=0.0)
        for i in range(POOL):
            colv[i] = feat[c * POOL + i]
        var q10 = _pct(colv, 0.10)
        var q50 = _pct(colv, 0.50)
        var q90 = _pct(colv, 0.90)
        var sig = sigma_frac * (q90 - q10)
        if sig < 1e-6:
            sig = 1e-6
        var ess_s = String("")
        for l in range(NLEV):
            var target = q10 if l == 0 else (q50 if l == 1 else q90)
            tgt[c * NLEV + l] = target
            var s1 = 0.0
            var s2 = 0.0
            for i in range(POOL):
                var d = (colv[i] - target) / sig
                var w = exp(-d * d)
                rewards[i] = Scalar[DT](w)
                s1 += w
                s2 += w * w
            var e = (s1 * s1 / s2) if s2 > 0.0 else 0.0
            ess[c * NLEV + l] = 100.0 * e / Float64(POOL)
            var zc = z_from_reward[D](b_list, rewards, POOL)
            for k in range(D):
                zs[(c * NLEV + l) * D + k] = zc[k]
            if l > 0:
                ess_s += String("/")
            ess_s += _f3(ess[c * NLEV + l])
        var a1090 = _angle_same(zs, (c * NLEV + 0) * D, (c * NLEV + 2) * D)
        var minang = 1e9
        for l in range(NLEV):
            for rr in range(3):
                var av = _angle_deg(zs, (c * NLEV + l) * D, ref_z, rr * D)
                if av < minang:
                    minang = av
        print("  " + _pad(names[c], 14) + _lpad(String(cols[c]), 5)
              + _lpad(_f3(q10), 9) + _lpad(_f3(q50), 9) + _lpad(_f3(q90), 9)
              + _lpad(_f3(sig), 8) + _lpad(ess_s, 7)
              + _lpad(_f3(a1090), 12) + _lpad(_f3(minang), 13))
    print("  (ESS% is the effective sample size of the reward weights at each")
    print("   level: 100% = every pool row counts, ~0% = the target is off the")
    print("   distribution and `z` is noise with a plausible norm.)")

    # ── PART 2: the behaviour. This is the gate. ──────────────────────
    print("-" * 84)
    print("PART 2 — what the robot ACTUALLY does (the gate)")
    print("  " + _pad(String("command"), 14) + _lpad(String("ach@p10"), 9)
          + _lpad(String("ach@p50"), 9) + _lpad(String("ach@p90"), 9)
          + _lpad(String("spread"), 9) + _lpad(String("cmd-gain"), 10)
          + _lpad(String("stand"), 9) + _lpad(String("rand+-sd"), 14)
          + _lpad(String("upright"), 11))
    var out = String(NCMD * NLEV) + " " + String(D) + "\n"

    # the two controls FIRST, because every row is read against them.
    var r_stand = _roll[FNet, BNet, ANet](
        t, env, rsi, norm, start_row, ref_z, 0,
        cols, horizon, False, 0, obs_t, z1, act_out, qp, qv,
    )
    # ⚠ EIGHT random draws, not one. A single `z` is not a control for an
    # EXTREMAL quantity: a random prompt makes the robot flail, and flailing
    # scores well on "how high is the hand" by accident. The mean and spread
    # of eight is the honest bar, and the first run of this probe read a
    # single draw ABOVE the commanded p90 on two of the four columns.
    comptime NRAND: Int = 8
    var rmean = List[Float64](length=NCMD, fill=0.0)
    var rsd = List[Float64](length=NCMD, fill=0.0)
    var rup = 0.0
    for _r in range(NRAND):
        var zt = Tensor.alloc(D)
        for k in range(D):
            zt.data[k] = Scalar[DT](_gauss())
        g1_project_z[D](zt, 0)
        var zr = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
        for k in range(D):
            zr[k] = zt.data[k]
        var rr = _roll[FNet, BNet, ANet](
            t, env, rsi, norm, start_row, zr, 0,
            cols, horizon, False, 0, obs_t, z1, act_out, qp, qv,
        )
        for c in range(NCMD):
            rmean[c] += rr.q[c]
            rsd[c] += rr.q[c] * rr.q[c]
        rup += rr.upright
    for c in range(NCMD):
        rmean[c] = rmean[c] / Float64(NRAND)
        var v = rsd[c] / Float64(NRAND) - rmean[c] * rmean[c]
        rsd[c] = sqrt(v) if v > 0.0 else 0.0
    rup = rup / Float64(NRAND)

    # every level of every command, keeping the p90 rollouts for the matrix
    var p90q = List[Float64](length=NCMD * NCMD, fill=0.0)
    var p10q = List[Float64](length=NCMD * NCMD, fill=0.0)
    var p90up = List[Float64](length=NCMD, fill=0.0)
    var n_open = 0
    for c in range(NCMD):
        var ach = List[Float64](length=NLEV, fill=0.0)
        var up = List[Float64](length=NLEV, fill=0.0)
        for l in range(NLEV):
            var r = _roll[FNet, BNet, ANet](
                t, env, rsi, norm, start_row, zs, (c * NLEV + l) * D,
                cols, horizon, False, 0, obs_t, z1, act_out, qp, qv,
            )
            ach[l] = r.q[c]
            up[l] = r.upright
            if l == 0:
                for c2 in range(NCMD):
                    p10q[c * NCMD + c2] = r.q[c2]
            if l == NLEV - 1:
                for c2 in range(NCMD):
                    p90q[c * NCMD + c2] = r.q[c2]
                p90up[c] = r.upright
            out += names[c] + " p" + String(Int(qs[l] * 100.0)) + " " + _f3(tgt[c * NLEV + l]) + "\n"
            for k in range(D):
                out += String(Float64(zs[(c * NLEV + l) * D + k]))
                out += " " if k + 1 < D else "\n"
        var spread = ach[2] - ach[0]
        var tspread = tgt[c * NLEV + 2] - tgt[c * NLEV + 0]
        var gain = spread / tspread if (tspread > 1e-9 or tspread < -1e-9) else 0.0
        # ⚠ a TOLERANCE, not a strict inequality: `ach` is a 40-step mean, and
        # the first run of this probe flagged `left_knee_z` non-monotonic on a
        # 0.003 inversion against a spread of 0.157. The bar is 10 % of the
        # command's own excursion.
        var tol = 0.10 * (spread if spread > 0.0 else -spread)
        var mono = (ach[1] >= ach[0] - tol and ach[2] >= ach[1] - tol) if tspread > 0.0 else (ach[1] <= ach[0] + tol and ach[2] <= ach[1] + tol)
        var ups = _f3(up[0]) + String("/") + _f3(up[2])
        print("  " + _pad(names[c], 14) + _lpad(_f3(ach[0]), 9)
              + _lpad(_f3(ach[1]), 9) + _lpad(_f3(ach[2]), 9)
              + _lpad(_f3(spread), 9) + _lpad(_f3(gain), 10)
              + _lpad(_f3(r_stand.q[c]), 9)
              + _lpad(_f3(rmean[c]) + String("+-") + _f3(rsd[c]), 14)
              + _lpad(ups + (String(" ") if mono else String("!")), 11))
        if gain > 0.30 and up[0] > 0.5 and up[2] > 0.5 and mono:
            n_open += 1
    print("  (upright = torso normal z at p10/p90: 1.0 standing, 0.0 on its")
    print("   face. ⚠ A SQUAT AND A FALL LOOK THE SAME IN ROOT HEIGHT.")
    print("   A trailing ! marks a NON-MONOTONIC p10 -> p50 -> p90.)")
    print("   random-z upright, mean of", NRAND, ":", _f3(rup))

    # ── PART 2b: are the four commands DIFFERENT behaviours? ──────────
    # ⚠ THE REAL GATE. Four prompts can each move their own quantity and
    # still be one shared "do something" direction, in which case commanding
    # `squat` would move the hand exactly as much as commanding the hand
    # does. Rows are what was COMMANDED, columns what was MEASURED, and every
    # entry is an EXCURSION between that row's own p10 and p90 prompts,
    # normalised by the column command's own excursion:
    #     M[r][c] = (q_c at r@p90 - q_c at r@p10) / (q_c at c@p90 - q_c at c@p10)
    # so the diagonal is 1.0 by construction and an off-diagonal near 1.0
    # means the two commands are the same behaviour wearing two names.
    #
    # ⚠ NOT normalised against `stand`. The first version of this matrix was,
    # and `root_height` blew up to -355: standing IS the 90th percentile of
    # root height, so `q_root at root@p90 - q_root at stand` is 0.001 and the
    # whole column divided by noise. An excursion between two prompts of the
    # SAME command is the well-conditioned denominator.
    print("-" * 84)
    print("PART 2b — cross-command matrix: commanded (row) vs measured (col)")
    var hdr = String("  ") + _pad(String("commanded \\ measured"), 22)
    for c in range(NCMD):
        hdr += _lpad(names[c], 14)
    print(hdr)
    var worst_off = 0.0
    for r in range(NCMD):
        var line = String("  ") + _pad(names[r], 22)
        for c in range(NCMD):
            var den = p90q[c * NCMD + c] - p10q[c * NCMD + c]
            var v = 0.0
            if den > 1e-9 or den < -1e-9:
                v = (p90q[r * NCMD + c] - p10q[r * NCMD + c]) / den
            line += _lpad(_f3(v), 14)
            if r != c:
                var a = v if v > 0.0 else -v
                if a > worst_off:
                    worst_off = a
        print(line)
    print("  worst off-diagonal magnitude:", _f3(worst_off),
          "(<1.0 means every command moves its OWN quantity most)")

    print("-" * 84)
    print("  commands that separate (cmd-gain > 0.30, monotonic, upright):",
          n_open, "of", NCMD)
    for c in range(NCMD):
        print("    " + _pad(names[c], 14) + " = " + says[c])
    print("  elapsed", Float64(perf_counter_ns() - t0) / 1e9, "s")

    if emit != "":
        write_text_atomic(emit, out)
        print("  wrote", NCMD * NLEV, "prompts to", emit)

    # ── PART 3: look at it ────────────────────────────────────────────
    if render:
        print("-" * 84)
        print("PART 3 — rendering the p90 end of each command. Escape to skip on.")
        for c in range(NCMD):
            print("  ", names[c], "@ p90 =", _f3(tgt[c * NLEV + 2]), "—", says[c])
            if env.check_renderer_quit():
                break
            _ = _roll[FNet, BNet, ANet](
                t, env, rsi, norm, start_row, zs, (c * NLEV + 2) * D,
                cols, horizon * 2, True, delay_ms, obs_t, z1, act_out, qp, qv,
            )
        env.close()
