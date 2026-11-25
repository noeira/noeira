"""`TrackingSim.set_box_load` and `joint_torque` on the CPU G1-with-a-box
simulator (`G1MjlabBoxSim`): known answers, no checkpoint, no store.

`set_box_load(kg)` is a constant vertical force on the box, the weight of `kg`
more kilograms, through the simulator's external wrench (`-D SM_XFRC=1`,
`noeira/physics3d/dynamics/xfrc.mojo`). The box is dropped from 2 m, away from
the robot, for 10 control steps (0.2 s) under three loads; a body in free
fall under the Euler integrator has `v = a t` to rounding, so:

  [1] no load: the box falls at g
  [2] a load of minus its own mass: it does not fall (the force is on the
      box, upward, of exactly its weight)
  [3] a load of plus its own mass: it falls at 2 g (weight without inertia)
  [4] the robot's state is the same in the three runs, bit for bit: the
      force is on the box alone
  [5] `joint_torque`: within each joint's limit, and not all zero (the robot
      holds its pose)

Built WITHOUT the define, the same file checks that `set_box_load` refuses
and that the fall is g.

Run: pixi run mojo build -D SM_XFRC=1 -I . -o build/t tests/deep_agents/supermimic/test_supermimic_box_load.mojo && pixi run ./build/t
     (and once without -D)
"""

from std.math import abs

from noeira.deep_agents.supermimic.robot import G1MjlabBoxSim, SM_XFRC
from noeira.envs.robots.unitree_g1_mjlab_pd import G1M_N_DOF, g1m_effort
from noeira.physics3d.gpu.constants import BODY_IDX_MASS, MODEL_META_IDX_GRAVITY_Z

comptime STEPS = 10
comptime NQ_ROBOT = 36
comptime NV_ROBOT = 35


struct Tally:
    var ok: Int
    var total: Int

    def __init__(out self):
        self.ok = 0
        self.total = 0

    def check(mut self, cond: Bool, msg: String):
        self.total += 1
        if cond:
            self.ok += 1
            print("  ok:", msg)
        else:
            print("  FAIL:", msg)


def _drop(load_in_box_masses: Float64, mut robot: List[Float64], mut torque: List[Float64]) raises -> Float64:
    """The box dropped from 2 m under a load of that many times its own mass;
    its vertical velocity after STEPS control steps. Appends the robot's
    (qpos, qvel) and the joints' last torques."""
    var sim = G1MjlabBoxSim.create()
    var nb = type_of(sim.env).NBODY
    var mass = sim.body_field(nb - 1, BODY_IDX_MASS)
    var bq: List[Float64] = [2.0, 0.5, 2.0, 1.0, 0.0, 0.0, 0.0]
    var bv: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sim.set_box(bq, bv)
    if load_in_box_masses != 0.0:
        sim.set_box_load(load_in_box_masses * mass)
    var a = List[Float64](length=G1M_N_DOF, fill=0.0)
    var bias = List[Float64](length=G1M_N_DOF, fill=0.0)
    for _ in range(STEPS):
        sim.control_step(a, bias)
    for i in range(NQ_ROBOT):
        robot.append(sim.qpos(i))
    for i in range(NV_ROBOT):
        robot.append(sim.qvel(i))
    for j in range(G1M_N_DOF):
        torque.append(sim.joint_torque(j))
    return sim.box_qvel(2)


def main() raises:
    var tl = Tally()
    var probe = G1MjlabBoxSim.create()
    var g = Float64(probe.env.mf.meta.data[MODEL_META_IDX_GRAVITY_Z])
    var dt = Float64(type_of(probe.env).CONFIG.get_timestep()) * 4.0
    var t = dt * Float64(STEPS)
    print("gravity", g, " control step", dt, " box mass", probe.body_field(type_of(probe.env).NBODY - 1, BODY_IDX_MASS), " built with SM_XFRC:", SM_XFRC)
    var r0 = List[Float64]()
    var tq = List[Float64]()
    var v0 = _drop(0.0, r0, tq)
    tl.check(abs(v0 - g * t) < 1e-9 * abs(g * t), "no load: the box falls at g (vz " + String(v0) + ", g t " + String(g * t) + ")")
    comptime if SM_XFRC:
        var r1 = List[Float64]()
        var r2 = List[Float64]()
        var t1 = List[Float64]()
        var t2 = List[Float64]()
        var vm = _drop(-1.0, r1, t1)
        var vp = _drop(1.0, r2, t2)
        tl.check(abs(vm) < 1e-9, "a load of minus its mass: it does not fall (vz " + String(vm) + ")")
        tl.check(abs(vp - 2.0 * g * t) < 1e-9 * abs(g * t), "a load of plus its mass: it falls at 2 g (vz " + String(vp) + ", 2 g t " + String(2.0 * g * t) + ")")
        var diff = 0
        for i in range(len(r0)):
            if r1[i] != r0[i] or r2[i] != r0[i]:
                diff += 1
        tl.check(len(r0) == NQ_ROBOT + NV_ROBOT and diff == 0, "the robot's " + String(len(r0)) + " state values are the same in the three runs (" + String(diff) + " differ)")
    else:
        var refused = False
        try:
            var rr = List[Float64]()
            var tt = List[Float64]()
            _ = _drop(1.0, rr, tt)
        except e:
            refused = True
        tl.check(refused, "without -D SM_XFRC=1, set_box_load refuses")
    var over = 0
    var biggest = 0.0
    for j in range(G1M_N_DOF):
        if abs(tq[j]) > g1m_effort(j) * (1.0 + 1e-12):
            over += 1
        biggest = max(biggest, abs(tq[j]))
    tl.check(over == 0 and biggest > 0.1, "joint torques: " + String(over) + " over their limit, the largest " + String(biggest) + " N.m")
    print("\n===", tl.ok, "/", tl.total, "passed ===")
    if tl.ok != tl.total:
        raise Error("test_supermimic_box_load: " + String(tl.total - tl.ok) + " failed")
