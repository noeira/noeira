"""Multi-body car dynamics validation (CPU) against Box2D ground truth.

CarDynamicsMB models the car as 5 bodies + 4 revolute joints solved with the
physics2d sequential-impulse pipeline. This test checks the two canonical
scenarios whose Box2D ground truth we measured (gen_car_reference.py), on grass:

  A: full gas, no steer -> car accelerates DEAD STRAIGHT (x~0, omega~0).
     Box2D reference at step 120: vy ~= 64.6, x ~= 0.
  B: full gas + full right steer -> bounded donut (Box2D omega ~5.7), the car
     does NOT fly off or blow up.

The legacy single-body CarDynamics spun in place off-track; this is the
regression guard that the multi-body port stays faithful + stable.
"""

from std.math import sqrt
from layout import Layout, LayoutTensor
from noeira.physics2d import dtype
from noeira.physics2d.constants import IDX_X, IDX_Y, IDX_ANGLE, IDX_VX, IDX_VY, IDX_OMEGA
from noeira.physics2d.car import CarDynamicsMB
from noeira.physics2d.car.constants import (
    FRICTION_LIMIT, GRASS_FRICTION, STEERING_LIMIT,
    CTRL_STEERING, CTRL_GAS, CTRL_BRAKE,
)
from noeira.physics2d.constants import BODY_STATE_SIZE, JOINT_DATA_SIZE
from noeira.nn.core.ptr import mptr

# Compact validation layout (one car).
comptime BOFF = 0
comptime FOFF = BOFF + CarDynamicsMB.NUM_BODIES * BODY_STATE_SIZE
comptime JOFF = FOFF + CarDynamicsMB.NUM_BODIES * 3
comptime ROFF = JOFF + CarDynamicsMB.NUM_JOINTS * JOINT_DATA_SIZE
comptime COFF = ROFF + CarDynamicsMB.NUM_WHEELS
comptime SSZ = COFF + 3
comptime DT = Scalar[dtype](0.02)


def fail(name: String, msg: String) raises:
    raise Error(String("[", name, "] ", msg))


def run_scenario(
    steer: Float64, gas: Float64, nsteps: Int
) raises -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """Run one car from (0,0,0) on grass; return final hull (x,y,angle,vx,vy,omega).
    """
    var sbuf = List[Scalar[dtype]](capacity=SSZ)
    for _ in range(SSZ):
        sbuf.append(Scalar[dtype](0.0))
    var state = LayoutTensor[dtype, Layout.row_major(1, SSZ), MutAnyOrigin](
        mptr(sbuf)
    )

    CarDynamicsMB.init_env[1, SSZ, BOFF, FOFF, JOFF, ROFF](
        0, state, Scalar[dtype](0.0), Scalar[dtype](0.0), Scalar[dtype](0.0)
    )
    state[0, COFF + CTRL_STEERING] = Scalar[dtype](steer)
    state[0, COFF + CTRL_GAS] = Scalar[dtype](gas)
    state[0, COFF + CTRL_BRAKE] = Scalar[dtype](0.0)

    var fric = Scalar[dtype](FRICTION_LIMIT) * Scalar[dtype](GRASS_FRICTION)
    for _ in range(nsteps):
        CarDynamicsMB.step_single_env[1, SSZ, BOFF, FOFF, JOFF, ROFF, COFF](
            0, state, fric, DT
        )

    return (
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_X])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_Y])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_ANGLE])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_VX])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_VY])),
        Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_OMEGA])),
    )


def run_spin_check(nsteps: Int) raises -> Tuple[Float64, Float64]:
    """Full gas + full lock: the car spins fast, so the hull turns past +/-pi
    (angles are not wrapped, as in Box2D). Returns (max |front-wheel steering
    angle|, turned-past-pi flag). The steering revolute joint must stay within
    +/-STEERING_LIMIT despite the hull crossing +/-pi — the regression guard
    for the angle-wrap-vs-joint-limit bug (a glitched relative angle let the
    motor slam the wheel a full turn past its limit)."""
    var sbuf = List[Scalar[dtype]](capacity=SSZ)
    for _ in range(SSZ):
        sbuf.append(Scalar[dtype](0.0))
    var state = LayoutTensor[dtype, Layout.row_major(1, SSZ), MutAnyOrigin](
        mptr(sbuf)
    )
    CarDynamicsMB.init_env[1, SSZ, BOFF, FOFF, JOFF, ROFF](
        0, state, Scalar[dtype](0.0), Scalar[dtype](0.0), Scalar[dtype](0.0)
    )
    state[0, COFF + CTRL_STEERING] = Scalar[dtype](1.0)  # full lock
    state[0, COFF + CTRL_GAS] = Scalar[dtype](1.0)
    state[0, COFF + CTRL_BRAKE] = Scalar[dtype](0.0)

    var fric = Scalar[dtype](FRICTION_LIMIT) * Scalar[dtype](GRASS_FRICTION)
    var pi_v = 3.141592653589793
    var two_pi = 2.0 * pi_v
    var bsz = Int(BODY_STATE_SIZE)
    var max_rel = 0.0
    var crossed = False
    for _ in range(nsteps):
        CarDynamicsMB.step_single_env[1, SSZ, BOFF, FOFF, JOFF, ROFF, COFF](
            0, state, fric, DT
        )
        var hull = Float64(rebind[Scalar[dtype]](state[0, BOFF + IDX_ANGLE]))
        # The unwrapped hull angle turned past +/-pi.
        var ahull = hull if hull >= 0.0 else -hull
        if ahull > pi_v:
            crossed = True
        # Front wheels are bodies 1 and 2; steering angle = wheel - hull.
        for wb in range(1, 3):
            var wa = Float64(
                rebind[Scalar[dtype]](state[0, BOFF + wb * bsz + IDX_ANGLE])
            )
            var rel = wa - hull
            if rel > pi_v:
                rel -= two_pi
            elif rel < -pi_v:
                rel += two_pi
            var arel = rel if rel >= 0.0 else -rel
            if arel > max_rel:
                max_rel = arel
    var crossed_f = 1.0 if crossed else 0.0
    return (max_rel, crossed_f)


def main() raises:
    print("=== CarDynamicsMB validation (vs Box2D ground truth) ===")

    # --- Scenario A: full gas, no steer -> dead straight ------------------
    var a = run_scenario(0.0, 1.0, 120)
    var ax = a[0]
    var ay = a[1]
    var avy = a[4]
    var aom = a[5]
    print("A @120: x=", ax, " y=", ay, " vy=", avy, " omega=", aom)
    var ax_abs = ax if ax >= 0.0 else -ax
    var aom_abs = aom if aom >= 0.0 else -aom
    if ax_abs > 1.0:
        fail("A", String("car drifted sideways (|x|=", ax_abs, "), expected ~0"))
    if aom_abs > 0.5:
        fail("A", String("car yawed (|omega|=", aom_abs, "), expected ~0"))
    # Box2D reference vy@120 ~= 64.6; allow generous band.
    if avy < 55.0 or avy > 72.0:
        fail("A", String("forward speed ", avy, " off Box2D reference ~64.6"))

    # --- Scenario B: full gas + full right steer -> bounded donut ---------
    var b = run_scenario(1.0, 1.0, 120)
    var bx = b[0]
    var by = b[1]
    var bom = b[5]
    print("B @120: x=", bx, " y=", by, " omega=", bom)
    var bx_abs = bx if bx >= 0.0 else -bx
    var by_abs = by if by >= 0.0 else -by
    var bom_abs = bom if bom >= 0.0 else -bom
    # Box2D donut stays near origin with bounded yaw (~5.7); must not blow up.
    if bom_abs > 12.0:
        fail("B", String("yaw rate ", bom_abs, " unbounded (Box2D ~5.7)"))
    if bx_abs > 40.0 or by_abs > 40.0:
        fail("B", String("car flew off: (", bx, ",", by, ")"))

    # --- Scenario C: full-lock spin -> steering joint respects its limit ---
    var c = run_spin_check(300)
    var c_maxrel = c[0]
    var c_crossed = c[1] > 0.5
    print(
        "C spin: max |steer angle|=", c_maxrel,
        " (limit ", STEERING_LIMIT, ")  crossed +/-pi:", c_crossed,
    )
    if not c_crossed:
        fail("C", "car never spun past +/-pi — spin precondition not met")
    # With the wrap fix the steering joint holds near +/-0.4; the bug let it
    # glitch toward ~2*pi. 0.8 cleanly separates the two.
    if c_maxrel > 0.8:
        fail(
            "C",
            String(
                "steering joint escaped +/-", STEERING_LIMIT,
                " (max |rel|=", c_maxrel, ") — angle-wrap vs joint-limit bug",
            ),
        )

    print("=== PASS: multi-body car matches Box2D (straight, donut, spin-limit) ===")
