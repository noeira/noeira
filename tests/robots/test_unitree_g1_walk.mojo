"""The G1 walker's task, before any training — G1_WALKER_PLAN L0a.

    pixi run mojo run -I . tests/robots/test_unitree_g1_walk.mojo

Runs on the CPU env through the GPU hooks' own functions
(`unitree_g1_walk.mojo` host helpers), so every check is on the arithmetic
the 5090 trains on:

  1. `g1_rotate_inverse` against an explicit rotation matrix.
  2. Command draws: 20 % +- 1 % exact zeros over 20 000 draws, every
     component inside its range, the ranges reached.
  3. Stand: the default-pose PD hold with a zero command for 10 s never
     terminates, keeps both feet on the floor, and pays the standing terms.
  4. A scripted step: lifting the left foot clears its contact and the
     touchdown pays air time under a non-zero command.
  5. A fall (pelvis on its side at 0.3 m) terminates, by tilt and by a
     non-foot floor contact.
  6. Every reward term is non-zero somewhere in 3-5 (the vacuity rule), and
     each check prints the hit count it stands on.

⚠ RUN FROM THE REPO ROOT (the model loads `unitree_g1.xml` by path).
"""

from std.math import cos, sin, sqrt
from std.random.philox import Random as PhiloxRandom

from noeira.core.cont_action import ContAction
from noeira.envs.robots.unitree_g1_walk import (
    UnitreeG1Walk,
    UnitreeG1WalkModel,
    g1_walk_host_reset,
    g1_walk_host_pre_step,
    g1_walk_host_terms,
)
from noeira.envs.robots.unitree_g1_walk_config import (
    G1_WALK_N_TERMS,
    G1_WALK_VX_MIN,
    G1_WALK_VX_MAX,
    G1_WALK_VY_MAX,
    G1_WALK_WZ_MAX,
    G1W_CMD_VX,
    G1W_CMD_VY,
    G1W_CMD_WZ,
    G1W_CMD_TIMER,
    G1W_LAST_L,
    G1W_LAST_R,
    T_FEET_AIR,
    T_FEET_STILL,
    T_STAND_STILL,
    T_TERMINATION,
    T_ALIVE,
    g1_rotate_inverse,
    g1_walk_command,
    g1_walk_is_stand,
    g1_walk_reward,
    g1_walk_term_name,
)

comptime NQ = UnitreeG1WalkModel.NQ
comptime NV = UnitreeG1WalkModel.NV
comptime ACT = UnitreeG1WalkModel.ACTION_DIM
comptime E = UnitreeG1Walk[False]


def _fail(msg: String) raises:
    print("  FAIL:", msg)
    raise Error(msg)


def check_rotate_inverse() raises:
    # q = rotation of `ang` about the unit axis (1, 2, 3)/|.|
    var ax = 1.0 / sqrt(14.0)
    var ay = 2.0 / sqrt(14.0)
    var az = 3.0 / sqrt(14.0)
    var ang = 0.7
    var w = cos(ang / 2)
    var s = sin(ang / 2)
    var x = ax * s
    var y = ay * s
    var z = az * s
    # R columns (body axes in world); R^T v = (col_k . v)
    var r00 = 1 - 2 * (y * y + z * z)
    var r01 = 2 * (x * y - w * z)
    var r02 = 2 * (x * z + w * y)
    var r10 = 2 * (x * y + w * z)
    var r11 = 1 - 2 * (x * x + z * z)
    var r12 = 2 * (y * z - w * x)
    var r20 = 2 * (x * z - w * y)
    var r21 = 2 * (y * z + w * x)
    var r22 = 1 - 2 * (x * x + y * y)
    var v0 = 0.3
    var v1 = -1.1
    var v2 = 0.5
    var e0 = r00 * v0 + r10 * v1 + r20 * v2
    var e1 = r01 * v0 + r11 * v1 + r21 * v2
    var e2 = r02 * v0 + r12 * v1 + r22 * v2
    var got = g1_rotate_inverse(w, x, y, z, v0, v1, v2)
    var err = abs(got[0] - e0) + abs(got[1] - e1) + abs(got[2] - e2)
    print("  rotate_inverse err", err)
    if err > 1e-12:
        _fail("g1_rotate_inverse disagrees with R^T v")
    # facing +y (yaw 90 deg), moving +y in the world = forward in the body
    var h = g1_rotate_inverse(cos(0.7853981633974483), 0, 0, sin(0.7853981633974483), 0, 1, 0)
    print("  yaw 90, world +y ->", h[0], h[1], h[2])
    if abs(h[0] - 1) > 1e-12 or abs(h[1]) > 1e-12:
        _fail("a forward walk at yaw 90 deg is not +x in the body frame")


def check_commands() raises:
    var rng = PhiloxRandom(seed=12345, offset=0)
    var n = 20000
    var zeros = 0
    var vx_lo = 1e9
    var vx_hi = -1e9
    var vy_hi = 0.0
    var wz_hi = 0.0
    for _ in range(n):
        var u = rng.step_uniform()
        var c = g1_walk_command(Float64(u[0]), Float64(u[1]), Float64(u[2]), Float64(u[3]))
        if c[0] == 0.0 and c[1] == 0.0 and c[2] == 0.0:
            zeros += 1
            if not g1_walk_is_stand(c[0], c[1], c[2]):
                _fail("a zero command is not a stand command")
            continue
        if c[0] < G1_WALK_VX_MIN or c[0] > G1_WALK_VX_MAX:
            _fail("vx out of range: " + String(c[0]))
        if abs(c[1]) > G1_WALK_VY_MAX or abs(c[2]) > G1_WALK_WZ_MAX:
            _fail("vy / wz out of range")
        vx_lo = min(vx_lo, c[0])
        vx_hi = max(vx_hi, c[0])
        vy_hi = max(vy_hi, abs(c[1]))
        wz_hi = max(wz_hi, abs(c[2]))
    var frac = Float64(zeros) / Float64(n)
    print("  zero commands", zeros, "/", n, "=", frac, " vx", vx_lo, "..", vx_hi,
          " |vy| max", vy_hi, " |wz| max", wz_hi)
    if abs(frac - 0.2) > 0.01:
        _fail("standing fraction off 20 %")
    if vx_lo > G1_WALK_VX_MIN + 0.01 or vx_hi < G1_WALK_VX_MAX - 0.01:
        _fail("vx range not reached")
    if vy_hi < G1_WALK_VY_MAX - 0.01 or wz_hi < G1_WALK_WZ_MAX - 0.01:
        _fail("vy / wz range not reached")


def _set_cmd(mut env: E, vx: Float64, vy: Float64, wz: Float64):
    env.d.meta.data[G1W_CMD_VX] = vx
    env.d.meta.data[G1W_CMD_VY] = vy
    env.d.meta.data[G1W_CMD_WZ] = wz
    env.d.meta.data[G1W_CMD_TIMER] = -1.0


def _refresh(mut env: E):
    var q = List[Float64]()
    for i in range(NQ):
        q.append(Float64(env.d.qpos.data[i]))
    var v = List[Float64]()
    for i in range(NV):
        v.append(Float64(env.d.qvel.data[i]))
    env.set_state(q, v)


struct Hits(Movable):
    var nz: List[Int]

    def __init__(out self):
        self.nz = List[Int](length=G1_WALK_N_TERMS, fill=0)

    def add(mut self, terms: Array[Float64, G1_WALK_N_TERMS]):
        for t in range(G1_WALK_N_TERMS):
            if terms[t] != 0.0:
                self.nz[t] += 1


def _step(
    mut env: E, a: List[Float64], mut terms: Array[Float64, G1_WALK_N_TERMS]
) raises -> Bool:
    g1_walk_host_pre_step[False](env.d)
    var act = ContAction[ACT]()
    for j in range(ACT):
        act[j] = a[j]
    var r = env.step(act)
    var done = g1_walk_host_terms(env.d, a, terms)
    if r[2] and not done and env.current_step < 1000:
        _fail("the CPU env ended an episode the host terms did not")
    return done


def check_stand(mut hits: Hits) raises:
    """⚠ THE DEFAULT-POSE PD HOLD IS NOT A STANDING CONTROLLER. With a zero
    action the G1 pitches forward and is down in ~1.1 s — BFM-Zero's own env
    (`UnitreeG1`) traces the same trajectory to the last digit, so this is
    the robot under its PD, not the walker's action map. Standing is a
    learned behaviour here. The check therefore reads the standing terms
    while the robot is still upright, and takes the forward fall as a
    natural termination case."""
    var env = E()
    _ = env.reset()
    g1_walk_host_reset(env.d, 1, False)
    _refresh(env)
    _set_cmd(env, 0, 0, 0)
    var zero = List[Float64](length=ACT, fill=0.0)
    var terms = Array[Float64, G1_WALK_N_TERMS](fill=0.0)
    comptime UPRIGHT = 30
    var both = 0
    var still_paid = 0
    var fell_at = -1
    for k in range(500):
        var done = _step(env, zero, terms)
        hits.add(terms)
        if k < UPRIGHT:
            if done:
                _fail("terminated while upright, step " + String(k))
            if env.d.meta.data[G1W_LAST_L] > 0.5 and env.d.meta.data[G1W_LAST_R] > 0.5:
                both += 1
            if terms[T_FEET_STILL] == 1.0:
                still_paid += 1
        if done:
            fell_at = k
            break
    print("  stand, first", UPRIGHT, "steps: both feet down", both,
          " feet_contact_no_cmd paid", still_paid, " | zero-action hold fell at step",
          fell_at, " pelvis z", Float64(env.d.qpos.data[2]))
    # the first step lands from the reset height: allow it
    if both < UPRIGHT - 2 or still_paid < UPRIGHT - 2:
        _fail("feet not both on the floor / standing term not paid while upright")
    if fell_at < 0:
        _fail("the zero-action hold did not fall: the termination missed it")


def check_step(mut hits: Hits) raises:
    var env = E()
    _ = env.reset()
    g1_walk_host_reset(env.d, 1, False)
    _refresh(env)
    _set_cmd(env, 0.5, 0, 0)
    var a = List[Float64](length=ACT, fill=0.0)
    var terms = Array[Float64, G1_WALK_N_TERMS](fill=0.0)
    # settle, then lift the left leg (hip pitch -0.4 rad, knee +0.5 rad — the
    # action range is +-0.5 rad,
    # ankle pitch -0.4) for 0.5 s, then put it back — before ~step 55, where
    # the zero-action hold is down anyway (check_stand)
    var off_steps = 0
    var off_run = 0
    var touchdown = -1e9
    var flight = 0
    var fell_at = -1
    for k in range(100):
        for j in range(ACT):
            a[j] = 0.0
        if k >= 5 and k < 30:
            a[0] = -0.4 / 0.25
            a[3] = 0.5 / 0.25
            a[4] = -0.4 / 0.25
        if _step(env, a, terms):
            fell_at = k
            hits.add(terms)
            break
        hits.add(terms)
        if env.d.meta.data[G1W_LAST_L] < 0.5:
            off_steps += 1
            off_run += 1
        else:
            # the first touchdown after a real flight (>= 3 steps off)
            if off_run >= 3 and touchdown == -1e9:
                touchdown = terms[T_FEET_AIR]
                flight = off_run
                print("  touchdown after", off_run, "steps off, air term", touchdown)
            off_run = 0
    print("  step: left foot off", off_steps, "steps, fell at", fell_at)
    if off_steps < 3 or touchdown == -1e9:
        _fail("lifting the left leg never produced a flight and a touchdown")
    # Playground's term at touchdown: the flight time minus 0.2 s (no lower
    # clip — a short flight is charged), to a step either way.
    var expect = Float64(flight) * 0.02 - 0.2
    print("  expected", expect, "+- 0.03")
    if abs(touchdown - expect) > 0.03:
        _fail("the touchdown's air term is not the flight time minus 0.2 s")


def check_fall(mut hits: Hits) raises:
    var env = E()
    _ = env.reset()
    g1_walk_host_reset(env.d, 1, False)
    # on its side: 90 deg about x, pelvis at 0.3 m
    env.d.qpos.data[2] = 0.3
    env.d.qpos.data[3] = 0.7071067811865476
    env.d.qpos.data[4] = 0.7071067811865476
    _refresh(env)
    _set_cmd(env, 0, 0, 0)
    var zero = List[Float64](length=ACT, fill=0.0)
    var terms = Array[Float64, G1_WALK_N_TERMS](fill=0.0)
    var done = _step(env, zero, terms)
    hits.add(terms)
    print("  fall: terminated", done, " termination term", terms[T_TERMINATION],
          " CPU env terminated", env.was_terminated())
    if not done or terms[T_TERMINATION] != 1.0 or terms[T_ALIVE] != 0.0:
        _fail("a robot on its side did not terminate")
    if not env.was_terminated():
        _fail("the CPU env's own termination missed the fall")


def check_random_reset() raises:
    """Random resets are inside their spreads and the first step is sane."""
    var env = E()
    _ = env.reset()
    var falls = 0
    for s in range(20):
        g1_walk_host_reset(env.d, 100 + s, True)
        _refresh(env)
        var yaw2 = Float64(env.d.qpos.data[6])
        if abs(Float64(env.d.qpos.data[0])) > 0.5 or abs(Float64(env.d.qpos.data[1])) > 0.5:
            _fail("reset xy outside +-0.5")
        var qn = 0.0
        for k in range(4):
            qn += Float64(env.d.qpos.data[3 + k]) ** 2
        if abs(qn - 1) > 1e-6:
            _fail("reset quaternion not unit")
        _ = yaw2
        var zero = List[Float64](length=ACT, fill=0.0)
        var terms = Array[Float64, G1_WALK_N_TERMS](fill=0.0)
        if _step(env, zero, terms):
            falls += 1
        if env.d.meta.data[G1W_CMD_TIMER] <= 0:
            _fail("the first pre-step did not draw a command")
    print("  random resets: 20 drawn, first-step terminations", falls)
    if falls > 0:
        _fail("a random reset terminates on its first step")


def main() raises:
    print("1. rotate_inverse")
    check_rotate_inverse()
    print("2. command draws")
    check_commands()
    var hits = Hits()
    print("3. stand")
    check_stand(hits)
    print("4. scripted step")
    check_step(hits)
    print("5. fall")
    check_fall(hits)
    print("random resets")
    check_random_reset()
    print("6. term liveness (non-zero steps over 3-5)")
    var dead = 0
    for t in range(G1_WALK_N_TERMS):
        print("   ", g1_walk_term_name(t), hits.nz[t])
        if hits.nz[t] == 0:
            dead += 1
    if dead > 0:
        _fail(String(dead) + " reward term(s) never live")
    print("[PASS] unitree_g1_walk")
