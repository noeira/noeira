"""The G1 walker under RoboParty's recipe, before training — G1_WALKER_PLAN §10.

    pixi run mojo run -I . tests/robots/test_unitree_g1_walk_rp.mojo

CPU env, through the GPU hooks' own functions (`unitree_g1_walk_rp.mojo`):

  1. Stand (zero action, nominal gains, zero command), first 25 steps:
     both feet on the floor; the solver's vertical contact forces sum to
     the robot's weight (±10 %) — the force plumbing against physics; sole
     heights ~0; feet and knee spacing inside RoboParty's bands; the
     standing terms paid.
  2. The zero-action hold falls (BFM's own env does, ~1.1 s) and the
     torso / hip contact terminates it.
  3. A scripted left-leg lift under a walking command: the swing foot
     clears 2 cm and pays `feet_height`, the single stance pays
     `feet_air_time`.
  4. An ankle driven past its range pays `dof_pos_limits`.
  5. A robot on its side terminates.
  6. Random resets: inside their ranges; per-joint gain scales inside
     [0.9, 1.1], not all equal, and exactly 1 at key 0.
  7. `feet_stumble` on synthetic records (it cannot fire on a flat floor).
  8. Every other term is non-zero somewhere in 1-5.
"""

from std.math import abs, sqrt, sin, cos

from layout import Layout

from noeira.core.cont_action import ContAction
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.gpu.constants import (
    BODY_IDX_MASS, MODEL_BODY_SIZE, CONTACT_SIZE, METADATA_SIZE,
    META_IDX_NUM_CONTACTS, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    CONTACT_IDX_FORCE_N, CONTACT_IDX_FORCE_T1, CONTACT_IDX_NX, CONTACT_IDX_NZ,
)
from noeira.envs.robots.unitree_g1_pd import (
    g1_pos_lower, g1_pos_upper, g1_default_pos, g1_kp, g1_kd,
)
from noeira.envs.robots.unitree_g1_walk_rp import (
    UnitreeG1WalkRP,
    UnitreeG1WalkRPModel,
    g1r_host_reset,
    g1r_host_pre_step,
    g1r_host_terms,
)
from noeira.envs.robots.unitree_g1_walk_rp_config import (
    G1R_N_TERMS,
    G1R_CMD_VX,
    G1R_CMD_VY,
    G1R_CMD_WZ,
    G1R_CMD_TIMER,
    G1R_KEY,
    G1R_O_CONTACT,
    G1R_O_FORCE,
    G1R_O_HEIGHT,
    G1R_FORCE_OBS_SCALE,
    R_FEET_STILL,
    R_STAND_STILL,
    R_FEET_DIST,
    R_KNEE_DIST,
    R_FEET_HEIGHT,
    R_FEET_AIR,
    R_POS_LIMITS,
    R_TERMINATION,
    R_UPWARD,
    R_FEET_STUMBLE,
    R_FEET_ORIENT,
    G1RContacts,
    g1r_contacts,
    g1r_stumble,
    g1r_gain_scales,
    g1_mirror_joint,
    g1r_mirror_env_obs,
    G1R_OBS_DIM,
    G1R_OBS_ACTOR,
    g1r_term_name,
)

comptime NQ = UnitreeG1WalkRPModel.NQ
comptime NV = UnitreeG1WalkRPModel.NV
comptime ACT = UnitreeG1WalkRPModel.ACTION_DIM
comptime NB = UnitreeG1WalkRPModel.NBODY
comptime E = UnitreeG1WalkRP[False]


def _fail(msg: String) raises:
    print("  FAIL:", msg)
    raise Error(msg)


struct Hits(Movable):
    var nz: List[Int]

    def __init__(out self):
        self.nz = List[Int](length=G1R_N_TERMS, fill=0)

    def add(mut self, terms: Array[Float64, G1R_N_TERMS]):
        for t in range(G1R_N_TERMS):
            if terms[t] != 0.0:
                self.nz[t] += 1


def _refresh(mut env: E):
    var q = List[Float64]()
    for i in range(NQ):
        q.append(Float64(env.d.qpos.data[i]))
    var v = List[Float64]()
    for i in range(NV):
        v.append(Float64(env.d.qvel.data[i]))
    env.set_state(q, v)


def _start(mut env: E, vx: Float64):
    _ = env.reset()
    g1r_host_reset(env.d, 1, False)
    _refresh(env)
    env.d.meta.data[G1R_CMD_VX] = vx
    env.d.meta.data[G1R_CMD_VY] = 0
    env.d.meta.data[G1R_CMD_WZ] = 0
    env.d.meta.data[G1R_CMD_TIMER] = -1


def _step(
    mut env: E, a: List[Float64], mut terms: Array[Float64, G1R_N_TERMS]
) raises -> Tuple[Bool, List[Float64]]:
    g1r_host_pre_step[False](env.d)
    var act = ContAction[ACT]()
    for j in range(ACT):
        act[j] = a[j]
    var r = env.step(act)
    var done = g1r_host_terms(env.d, a, terms)
    if r[2] != done and env.current_step < 1000:
        _fail("CPU env termination disagrees with the host terms")
    var o = List[Float64]()
    for k in range(UnitreeG1WalkRPModel.OBS_DIM):
        o.append(Float64(r[0].data[k]))
    return (done, o^)


def check_stand(mut hits: Hits) raises:
    var env = E()
    _start(env, 0.0)
    var mass = 0.0
    for b in range(NB):
        mass += Float64(env.mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
    var weight = mass * 9.81
    var zero = List[Float64](length=ACT, fill=0.0)
    var terms = Array[Float64, G1R_N_TERMS](fill=0.0)
    var fz_sum = 0.0
    var hmax = 0.0
    var n = 0
    var both = 0
    var fell_at = -1
    for k in range(200):
        var r = _step(env, zero, terms)
        hits.add(terms)
        if k >= 5 and k < 25:
            var o = r[1].copy()
            if o[G1R_O_CONTACT] > 0.5 and o[G1R_O_CONTACT + 1] > 0.5:
                both += 1
            fz_sum += (o[G1R_O_FORCE + 2] + o[G1R_O_FORCE + 5]) / G1R_FORCE_OBS_SCALE
            hmax = max(hmax, max(abs(o[G1R_O_HEIGHT]), abs(o[G1R_O_HEIGHT + 1])))
            n += 1
            if terms[R_FEET_STILL] <= 0.0 or terms[R_STAND_STILL] <= 0.0:
                _fail("standing terms not paid while standing (step " + String(k) + ")")
            if terms[R_FEET_DIST] < 0.99 or terms[R_KNEE_DIST] < 0.99:
                _fail("stance width outside the bands: feet " + String(terms[R_FEET_DIST])
                      + " knees " + String(terms[R_KNEE_DIST]))
            if terms[R_UPWARD] < 0.95:
                _fail("upward not ~1 while standing")
        if r[0]:
            fell_at = k
            break
    var fz = fz_sum / Float64(n)
    print("  stand: mass", mass, "kg, weight", weight, "N | mean vertical contact force",
          fz, "N | both feet", both, "/", n, "| max |sole height|", hmax,
          "| zero-action hold terminated at step", fell_at)
    if abs(fz - weight) > 0.1 * weight:
        _fail("vertical contact forces do not carry the robot's weight")
    if both < n:
        _fail("feet not both on the floor while standing")
    if hmax > 0.01:
        _fail("sole height not ~0 while standing")
    if fell_at < 0:
        _fail("the zero-action hold never terminated: the torso / hip contact missed the fall")


def check_lift(mut hits: Hits) raises:
    var env = E()
    _start(env, 0.5)
    var a = List[Float64](length=ACT, fill=0.0)
    var terms = Array[Float64, G1R_N_TERMS](fill=0.0)
    var height_paid = 0
    var air_paid = 0.0
    for k in range(60):
        for j in range(ACT):
            a[j] = 0.0
        if k >= 5 and k < 30:
            a[0] = -0.4 / 0.25
            a[3] = 0.8 / 0.25
            a[4] = -0.4 / 0.25
        var r = _step(env, a, terms)
        hits.add(terms)
        if terms[R_FEET_HEIGHT] > 0.0:
            height_paid += 1
        air_paid = max(air_paid, terms[R_FEET_AIR])
        if r[0]:
            break
    print("  lift: feet_height paid", height_paid, "steps, best feet_air_time", air_paid)
    if height_paid == 0 or air_paid <= 0.0:
        _fail("a lifted foot under a walking command paid no feet_height / feet_air_time")


def check_limits(mut hits: Hits) raises:
    var env = E()
    _start(env, 0.0)
    var a = List[Float64](length=ACT, fill=0.0)
    a[5] = 4.0      # left ankle roll: target +1 rad, range +-0.26
    var terms = Array[Float64, G1R_N_TERMS](fill=0.0)
    var paid = 0
    for _ in range(15):
        var r = _step(env, a, terms)
        hits.add(terms)
        if terms[R_POS_LIMITS] > 0.0:
            paid += 1
        if r[0]:
            break
    print("  limits: dof_pos_limits paid", paid, "steps")
    if paid == 0:
        _fail("an ankle driven past its range paid no dof_pos_limits")


def check_side(mut hits: Hits) raises:
    var env = E()
    _start(env, 0.0)
    env.d.qpos.data[2] = 0.3
    env.d.qpos.data[3] = 0.7071067811865476
    env.d.qpos.data[4] = 0.7071067811865476
    _refresh(env)
    var zero = List[Float64](length=ACT, fill=0.0)
    var terms = Array[Float64, G1R_N_TERMS](fill=0.0)
    var done = False
    for _ in range(10):
        var r = _step(env, zero, terms)
        hits.add(terms)
        if r[0]:
            done = True
            break
    print("  side: terminated", done)
    if not done or terms[R_TERMINATION] != 1.0:
        _fail("a robot on its side did not terminate")


def check_foot_orientation() raises:
    """The whole robot pitched by 0.3 rad (joints default): each foot is
    tilted 0.3 rad, so `feet_orientation_l2` = 2 sin^2(0.3). Catches a body
    quaternion read in the wrong component order — exact at the stand, wrong
    at any tilt."""
    var env = E()
    _start(env, 0.0)
    var q = List[Float64]()
    for i in range(NQ):
        q.append(Float64(env.d.qpos.data[i]))
    var v = List[Float64](length=NV, fill=0.0)
    q[3] = cos(0.15)
    q[5] = sin(0.15)
    env.set_state(q, v)
    var zero = List[Float64](length=ACT, fill=0.0)
    var terms = Array[Float64, G1R_N_TERMS](fill=0.0)
    _ = g1r_host_terms(env.d, zero, terms)
    var want = 2.0 * sin(0.3) ** 2
    print("  pitched 0.3 rad: feet_orientation_l2", terms[R_FEET_ORIENT], "want", want)
    if abs(terms[R_FEET_ORIENT] - want) > 1e-6:
        _fail("feet_orientation_l2 is not the feet's tilt")
    # ⚠ A PITCH ALONE IS BLIND to a (w, x, y, z) / (x, y, z, w) mix-up (the
    # mix-up gives the same squared tilt); a YAW is not: flat feet must read
    # 0, the mix-up reads 2 sin^2(2 x 0.3).
    q[3] = cos(0.3)
    q[4] = 0.0
    q[5] = 0.0
    q[6] = sin(0.3)
    env.set_state(q, v)
    _ = g1r_host_terms(env.d, zero, terms)
    print("  yawed 0.6 rad: feet_orientation_l2", terms[R_FEET_ORIENT], "want 0")
    if abs(terms[R_FEET_ORIENT]) > 1e-6:
        _fail("feet_orientation_l2 sees tilt in a yawed, flat foot: quaternion order")


def check_resets() raises:
    var env = E()
    _ = env.reset()
    var lo = 10.0
    var hi = -10.0
    for s in range(20):
        g1r_host_reset(env.d, 100 + s, True)
        var key = Int(env.d.meta.data[G1R_KEY])
        if key <= 0:
            _fail("a random reset left key 0 (nominal gains)")
        if abs(Float64(env.d.qpos.data[0])) > 0.5 or abs(Float64(env.d.qpos.data[1])) > 0.5:
            _fail("reset xy outside +-0.5")
        for i in range(29):
            var sc = g1r_gain_scales(key, i)
            lo = min(lo, min(sc[0], sc[1]))
            hi = max(hi, max(sc[0], sc[1]))
    var one = g1r_gain_scales(0, 3)
    print("  resets: gain scales over 20 lanes x 29 joints in [", lo, ",", hi, "], key 0 ->", one[0], one[1])
    if lo < 0.9 or hi > 1.1 or hi - lo < 0.15:
        _fail("gain scales outside [0.9, 1.1] or degenerate")
    if one[0] != 1.0 or one[1] != 1.0:
        _fail("key 0 is not nominal")


def check_stumble() raises:
    """`feet_stumble` cannot fire on a flat floor (friction ~ mu N); its
    condition is checked on records instead: a left foot (body 8) against a
    wall-like normal (+x) and against the floor (+z)."""
    var t = TensorImpl[DType.float64].alloc(64 * CONTACT_SIZE)
    var m = TensorImpl[DType.float64].alloc(METADATA_SIZE)
    m.data[META_IDX_NUM_CONTACTS] = 1
    t.data[CONTACT_IDX_BODY_A] = -1
    t.data[CONTACT_IDX_BODY_B] = 8
    t.data[CONTACT_IDX_FORCE_N] = 100.0
    t.data[CONTACT_IDX_NX] = 1.0
    var fc = g1r_contacts[DType.float64, 1, 64](
        t.lt["cpu", Layout.row_major(1, 64 * CONTACT_SIZE)](),
        m.lt["cpu", Layout.row_major(1, METADATA_SIZE)](), 0,
    )
    var wall = g1r_stumble(fc)
    t.data[CONTACT_IDX_NX] = 0.0
    t.data[CONTACT_IDX_NZ] = 1.0
    t.data[CONTACT_IDX_FORCE_T1] = 80.0      # friction at mu 0.8
    fc = g1r_contacts[DType.float64, 1, 64](
        t.lt["cpu", Layout.row_major(1, 64 * CONTACT_SIZE)](),
        m.lt["cpu", Layout.row_major(1, METADATA_SIZE)](), 0,
    )
    var floor = g1r_stumble(fc)
    print("  stumble: wall-normal contact", wall, "| floor contact with friction", floor)
    if not wall or floor:
        _fail("feet_stumble's condition is wrong on a synthetic record")


def check_mirror_model() raises:
    """The joint mirror against the model: ranges reflect, defaults reflect,
    gains match, and mirroring twice is the identity."""
    var worst = 0.0
    for i in range(29):
        var m = g1_mirror_joint(i)
        var j = m[0]
        var sg = m[1]
        var back = g1_mirror_joint(j)
        if back[0] != i or back[1] != sg:
            _fail("the joint mirror is not an involution at " + String(i))
        var lo = sg * g1_pos_lower(i) if sg > 0 else -g1_pos_upper(i)
        var hi = sg * g1_pos_upper(i) if sg > 0 else -g1_pos_lower(i)
        worst = max(worst, abs(lo - g1_pos_lower(j)))
        worst = max(worst, abs(hi - g1_pos_upper(j)))
        worst = max(worst, abs(sg * g1_default_pos(i) - g1_default_pos(j)))
        worst = max(worst, abs(g1_kp(i) - g1_kp(j)) + abs(g1_kd(i) - g1_kd(j)))
    print("  joint mirror vs the model: worst range / default / gain mismatch", worst)
    if worst > 1e-3:
        _fail("the joint mirror disagrees with the model's ranges, defaults or gains")


def _mirror_state(mut env: E) -> Tuple[List[Float64], List[Float64]]:
    var q = List[Float64]()
    for i in range(NQ):
        q.append(Float64(env.d.qpos.data[i]))
    var v = List[Float64]()
    for i in range(NV):
        v.append(Float64(env.d.qvel.data[i]))
    var qm = q.copy()
    var vm = v.copy()
    qm[1] = -q[1]
    qm[4] = -q[4]          # quaternion (w, x, y, z) -> (w, -x, y, -z)
    qm[6] = -q[6]
    vm[1] = -v[1]          # world linear (vx, -vy, vz)
    vm[3] = -v[3]          # body angular (-wx, wy, -wz)
    vm[5] = -v[5]
    for i in range(29):
        var m = g1_mirror_joint(i)
        qm[7 + m[0]] = m[1] * q[7 + i]
        vm[6 + m[0]] = m[1] * v[6 + i]
    return (qm^, vm^)


def check_mirror_physics() raises:
    """Step a lifted, turning state A with an asymmetric action, and its
    mirror image B with the mirrored action: B's observation must be A's
    mirrored. Physics agrees only as far as the model is symmetric, so the
    bar is loose; a wrong sign is an O(1) error."""
    var a_env = E()
    _start(a_env, 0.5)
    a_env.d.meta.data[G1R_CMD_VY] = 0.2
    a_env.d.meta.data[G1R_CMD_WZ] = 0.4
    var a = List[Float64](length=ACT, fill=0.0)
    var terms = Array[Float64, G1R_N_TERMS](fill=0.0)
    for _ in range(12):
        for j in range(ACT):
            a[j] = 0.0
        a[0] = -1.6
        a[3] = 2.4
        a[13] = 0.8          # waist roll
        a[16] = 1.2          # left shoulder roll
        _ = _step(a_env, a, terms)
    var ms = _mirror_state(a_env)
    var b_env = E()
    _start(b_env, 0.5)
    b_env.set_state(ms[0], ms[1])
    b_env.d.meta.data[G1R_CMD_VX] = a_env.d.meta.data[G1R_CMD_VX]
    b_env.d.meta.data[G1R_CMD_VY] = -a_env.d.meta.data[G1R_CMD_VY]
    b_env.d.meta.data[G1R_CMD_WZ] = -a_env.d.meta.data[G1R_CMD_WZ]
    var am = List[Float64](length=ACT, fill=0.0)
    for j in range(ACT):
        var m = g1_mirror_joint(j)
        am[m[0]] = m[1] * a[j]
    var ra = _step(a_env, a, terms)
    var rb = _step(b_env, am, terms)
    var mo = g1r_mirror_env_obs()
    var midx = mo[0].copy()
    var msgn = mo[1].copy()
    var oa = ra[1].copy()
    var ob = rb[1].copy()
    var worst_act = 0.0
    var worst_k = -1
    var worst_priv = 0.0
    var worst_pk = -1
    var worst_rel = 0.0
    for k in range(G1R_OBS_DIM):
        var want = msgn[k] * oa[midx[k]]
        var d = abs(ob[k] - want)
        if k < G1R_OBS_ACTOR:
            if d > worst_act:
                worst_act = d
                worst_k = k
        else:
            if d > worst_priv:
                worst_priv = d
                worst_pk = k
            # relative to the word's own size (forces are in N)
            worst_rel = max(worst_rel, d / max(1.0, abs(want)))
    print("  mirror physics after one step: actor obs worst", worst_act, "at word", worst_k,
          "| privileged worst", worst_priv, "at word", worst_pk, "(",
          oa[midx[worst_pk]] * msgn[worst_pk], "vs", ob[worst_pk], "), relative", worst_rel)
    if worst_act > 0.05:
        _fail("the observation mirror disagrees with the mirrored physics")
    if worst_rel > 0.05:
        _fail("the privileged mirror disagrees with the mirrored physics")


def main() raises:
    var hits = Hits()
    print("1-2. stand, then the zero-action fall")
    check_stand(hits)
    print("3. scripted lift")
    check_lift(hits)
    print("4. joint limits")
    check_limits(hits)
    print("5. on its side")
    check_side(hits)
    print("5b. foot orientation under a known tilt")
    check_foot_orientation()
    print("6. random resets, gain DR")
    check_resets()
    print("6b. mirror maps: against the model, against physics")
    check_mirror_model()
    check_mirror_physics()
    print("7. feet_stumble on synthetic records")
    check_stumble()
    print("8. term liveness (feet_stumble excused: flat floor, see 7)")
    var dead = 0
    for t in range(G1R_N_TERMS):
        print("   ", g1r_term_name(t), hits.nz[t])
        if hits.nz[t] == 0 and t != R_FEET_STUMBLE:
            dead += 1
    if dead > 0:
        _fail(String(dead) + " term(s) never live")
    print("[PASS] unitree_g1_walk_rp")
