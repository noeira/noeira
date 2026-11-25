"""A per-lane external wrench on the device (`Data.xfrc_applied`, `XFRC=True`),
on the batched G1 (mjlab's model, the one SuperMimic trains on), 2 lanes.

What `mj_xfrcAccumulate` is and its float64 gate against MuJoCo:
`noeira/physics3d/dynamics/xfrc.mojo`, `tests/physics3d/test_xfrc_vs_mujoco.mojo`.
This file asks the device questions: does the kernel give the CPU pass's
answer, per lane, through the same Euler step; and does the batched env's own
step read the buffer.

The reference for [1] and [2] is the SAME integrator type stepped on the CPU
in float32 on the env's own model, from the same state and wrenches. The two
targets differ before any wrench (float32 rounding, contact solves), so the
bars are relative, as in the float64 gate (written before the first run):

    e0 = device vs CPU with no wrench, e = with the wrenches,
    effect = what the wrenches do (CPU with vs without)
    e <= max(10 e0, 1e-5)   and   1000 e <= effect

  [1] in the air (root raised 0.5 m, no contact), after 1 and 5 steps: each
      lane has its own wrenches (a force at a point of each wrist, a force
      and a torque on the torso; lane 1 other values, another arm pose)
  [2] feet in the floor, after 1 and 5 steps
  [3] lanes are independent and zero is nothing: with wrenches on lane 0
      only, lane 1 equals the no-wrench run's lane 1 BIT FOR BIT, and lane 0
      equals its values in the run where lane 1 also had wrenches
  [4] the env's own `step_batch` reads the buffer: 3 control steps at a zero
      action, wrenches on lane 0 only: lane 0 departs from the no-wrench run,
      lane 1 does not, bit for bit

Not measured here: a device `XFRC=True` build with an all-zero buffer against
a device `XFRC=False` build (it would compile a second G1 env; the float64
gate measures it for the same function, and [3] measures the skip per lane).

NVIDIA ONLY, and NOT RUN YET (2026-11-25): the batched G1 does not build for
Metal (its action hook computes in float64), so this prints SKIP on Apple, as
`test_supermimic_env_gpu` does. On the Mac it type-checks and lowers up to
Metal's IR verifier, which then refuses that hook; no value below has been
read. The same four questions on Metal, on other models:
`tests/physics3d/test_xfrc_gpu.mojo`. Read the printed values before
trusting the bars.

Run: pixi run -e nvidia mojo run -I . tests/physics3d/test_xfrc_g1_gpu.mojo
"""

from std.math import abs
from std.sys import default_accelerator

from max.gpu.host import DeviceContext

from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.robots.unitree_g1_mjlab_config import UnitreeG1MjlabConfig
from noeira.envs.robots.unitree_g1_mjlab_xml import UnitreeG1MjlabModel
from noeira.nn.constants import DT
from noeira.physics3d.dynamics.xfrc import XFRC_STRIDE, xfrc_at_point
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics


comptime NL = 2
comptime GEnv = Phyics3dBatchedEnv[
    UnitreeG1MjlabModel, UnitreeG1MjlabConfig, NL, TERMINATE_ON_UNHEALTHY=False, LANE_DR=True, XFRC=True,
]
comptime GData = Data[DT, GEnv.MD, NL]
comptime GModel = Model[DT, GEnv.MD]
comptime NQ = GEnv.NQ
comptime NV = GEnv.NV
comptime NA = GEnv.ACT_DIM
comptime NB = GEnv.NBODY
comptime ROW = NQ + NV
# MuJoCo's body ids in mjlab's g1.xml (world = 0); the float64 gate resolves
# the same three by name on the same file.
comptime B_LEFT = 23
comptime B_RIGHT = 30
comptime B_TORSO = 16
comptime STEPS_A = 1
comptime STEPS_B = 5
comptime ENV_STEPS = 3
# which lanes carry wrenches
comptime W_NONE = 0
comptime W_BOTH = 1
comptime W_LANE0 = 2


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


def _state(base: List[Scalar[DT]], dz: Float64, mut d: GData):
    """Both lanes at `base` raised by `dz`, at rest; lane 1 with another arm pose."""
    for l in range(NL):
        for k in range(NQ):
            d.qpos.data[l * NQ + k] = base[k]
        d.qpos.data[l * NQ + 2] = base[2] + Scalar[DT](dz)
        for k in range(NV):
            d.qvel.data[l * NV + k] = Scalar[DT](0)
            d.qacc.data[l * NV + k] = Scalar[DT](0)
            d.qacc_warmstart.data[l * NV + k] = Scalar[DT](0)
            d.qfrc.data[l * NV + k] = Scalar[DT](0)
    # lane 1: left shoulder pitch, left elbow, right shoulder roll (qpos 7 + joint)
    d.qpos.data[NQ + 7 + 15] = d.qpos.data[NQ + 7 + 15] + Scalar[DT](0.3)
    d.qpos.data[NQ + 7 + 18] = d.qpos.data[NQ + 7 + 18] + Scalar[DT](-0.25)
    d.qpos.data[NQ + 7 + 23] = d.qpos.data[NQ + 7 + 23] + Scalar[DT](-0.2)


def _wrenches(mut d: GData, mut mf: GModel, which: Int) raises:
    """`d.xfrc_applied` for the state in `d` (host): none, both lanes, or lane 0 only."""
    for i in range(d.xfrc_applied.n):
        d.xfrc_applied.data[i] = Scalar[DT](0)
    if which == W_NONE:
        return
    forward_kinematics["cpu", DT, BATCH=NL, LANE_DR=True](d, mf, None)
    for l in range(NL):
        if which == W_LANE0 and l > 0:
            continue
        var s = 1.0 if l == 0 else -0.6
        # a force at a point 6 cm from each wrist body's CoM
        var pl = (
            Float64(d.xipos.data[(l * NB + B_LEFT) * 3 + 0]) + 0.05,
            Float64(d.xipos.data[(l * NB + B_LEFT) * 3 + 1]) + 0.01,
            Float64(d.xipos.data[(l * NB + B_LEFT) * 3 + 2]) - 0.03,
        )
        var pr = (
            Float64(d.xipos.data[(l * NB + B_RIGHT) * 3 + 0]) + 0.05,
            Float64(d.xipos.data[(l * NB + B_RIGHT) * 3 + 1]) - 0.01,
            Float64(d.xipos.data[(l * NB + B_RIGHT) * 3 + 2]) - 0.03,
        )
        xfrc_at_point(d, l, B_LEFT, 5.0 * s, -8.0, -20.0 * s, pl[0], pl[1], pl[2])
        xfrc_at_point(d, l, B_RIGHT, -3.0, 6.0 * s, -15.0, pr[0], pr[1], pr[2])
        var o = (l * NB + B_TORSO) * XFRC_STRIDE
        d.xfrc_applied.data[o + 0] = Scalar[DT](10.0 * s)
        d.xfrc_applied.data[o + 1] = Scalar[DT](0.0)
        d.xfrc_applied.data[o + 2] = Scalar[DT](4.0)
        d.xfrc_applied.data[o + 3] = Scalar[DT](0.0)
        d.xfrc_applied.data[o + 4] = Scalar[DT](2.0)
        d.xfrc_applied.data[o + 5] = Scalar[DT](-1.0 * s)


def _rows(d: GData, mut out: List[Scalar[DT]]):
    for l in range(NL):
        for k in range(NQ):
            out.append(d.qpos.data[l * NQ + k])
        for k in range(NV):
            out.append(d.qvel.data[l * NV + k])


def _cpu(mut env: GEnv, base: List[Scalar[DT]], dz: Float64, which: Int, mut ncon: List[Int]) raises -> List[Scalar[DT]]:
    """The env's integrator type on the CPU, on the env's model: the lanes'
    (qpos, qvel) after STEPS_A then STEPS_B steps; lane 0's contact count at
    the end is appended to `ncon`."""
    var d = GData()
    _state(base, dz, d)
    _wrenches(d, env.mf, which)
    var integ = GEnv.IntegEuler()
    var out = List[Scalar[DT]]()
    for t in range(1, STEPS_B + 1):
        integ.step["cpu"](d, env.mf)
        if t == STEPS_A or t == STEPS_B:
            _rows(d, out)
    ncon.append(Int(d.meta.data[META_IDX_NUM_CONTACTS]))
    return out^


def _to_device(mut env: GEnv, ctx: DeviceContext, base: List[Scalar[DT]], dz: Float64, which: Int) raises:
    """A fresh episode, then the state and the wrenches on the device."""
    env.reset_batch[NL](Optional(ctx), 0)
    _state(base, dz, env.d)
    _wrenches(env.d, env.mf, which)
    env.d.qpos.upload_resident(ctx)
    env.d.qvel.upload_resident(ctx)
    env.d.qacc.upload_resident(ctx)
    env.d.qacc_warmstart.upload_resident(ctx)
    env.d.qfrc.upload_resident(ctx)
    env.d.xfrc_applied.upload_resident(ctx)
    env._run_fields_fk(ctx)
    env._run_fields_vel(ctx)


def _gpu(mut env: GEnv, ctx: DeviceContext, base: List[Scalar[DT]], dz: Float64, which: Int) raises -> List[Scalar[DT]]:
    """The env's own integrator on the device, the same steps."""
    _to_device(env, ctx, base, dz, which)
    var out = List[Scalar[DT]]()
    for t in range(1, STEPS_B + 1):
        env.integ_euler.step["gpu"](env.d, env.mf, ctx)
        if t == STEPS_A or t == STEPS_B:
            env.d.qpos.download(ctx)
            env.d.qvel.download(ctx)
            _rows(env.d, out)
    return out^


def _env_steps(mut env: GEnv, ctx: DeviceContext, base: List[Scalar[DT]], dz: Float64, which: Int) raises -> List[Scalar[DT]]:
    """`step_batch` at a zero action, ENV_STEPS control steps."""
    _to_device(env, ctx, base, dz, which)
    var ha = ctx.enqueue_create_host_buffer[DT](NL * NA)
    ctx.synchronize()
    for i in range(NL * NA):
        ha[i] = Scalar[DT](0)
    for t in range(ENV_STEPS):
        ctx.enqueue_copy(env._action, ha)
        env.step_batch[NL](Optional(ctx), UInt64(t))
    env.d.qpos.download(ctx)
    env.d.qvel.download(ctx)
    var out = List[Scalar[DT]]()
    _rows(env.d, out)
    return out^


def _dist(a: List[Scalar[DT]], b: List[Scalar[DT]], lo: Int, hi: Int) -> Tuple[Int, Float64]:
    """(entries that differ, max |a − b|) over [lo, hi)."""
    var n = 0
    var m = 0.0
    for i in range(lo, hi):
        if a[i] != b[i]:
            n += 1
            m = max(m, abs(Float64(a[i]) - Float64(b[i])))
    return (n, m)


def _leg(mut env: GEnv, ctx: DeviceContext, base: List[Scalar[DT]], dz: Float64, label: String, contact: Bool, mut tl: Tally) raises:
    var ncon = List[Int]()
    var c0 = _cpu(env, base, dz, W_NONE, ncon)
    var c1 = _cpu(env, base, dz, W_BOTH, ncon)
    var g0 = _gpu(env, ctx, base, dz, W_NONE)
    var g1 = _gpu(env, ctx, base, dz, W_BOTH)
    tl.check((ncon[0] > 0) == contact, label + ": lane 0 has " + String(ncon[0]) + " contacts with no wrench at the end (contact expected: " + String(contact) + ")")
    var nan = 0
    for i in range(len(g1)):
        if g1[i] != g1[i] or c1[i] != c1[i]:
            nan += 1
    tl.check(nan == 0 and len(g1) == 2 * NL * ROW, label + ": " + String(len(g1)) + " values, " + String(nan) + " NaN")
    for part in range(2):
        var steps = STEPS_A if part == 0 else STEPS_B
        for l in range(NL):
            var lo = part * NL * ROW + l * ROW
            var e0 = _dist(g0, c0, lo, lo + ROW)[1]
            var e = _dist(g1, c1, lo, lo + ROW)[1]
            var eff = _dist(c1, c0, lo, lo + ROW)[1]
            var bar = max(10.0 * e0, 1e-5)
            tl.check(
                e <= bar and e * 1000.0 <= eff,
                label + ", " + String(steps) + " step(s), lane " + String(l) + ": device vs CPU " + String(e) + " (no wrench " + String(e0)
                + ", bar " + String(bar) + "), the wrenches' effect " + String(eff),
            )


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        # the batched G1 is NVIDIA only; never instantiated here
        print("SKIP test_xfrc_g1_gpu: needs an NVIDIA accelerator (the batched G1)")
        return
    else:
        var ctx = DeviceContext()
        var tl = Tally()
        var env = GEnv(ctx)
        env.reset_batch[NL](Optional(ctx), 0)
        env.d.qpos.download(ctx)
        var base = List[Scalar[DT]](length=NQ, fill=0)
        for k in range(NQ):
            base[k] = env.d.qpos.data[k]
        print("the reset pose's root height:", base[2], " bodies:", NB, " nq:", NQ, " nv:", NV)
        _ = env.xfrc_ptr()

        print("[1] in the air: the device kernel against the CPU pass, per lane")
        _leg(env, ctx, base, 0.5, String("air"), False, tl)

        # the floor leg's height: the first drop from the reset pose (2 mm steps) with contacts after a CPU step
        var fdz = 0.0
        var found = False
        for i in range(40):
            fdz = -0.002 * Float64(i)
            var d = GData()
            _state(base, fdz, d)
            var integ = GEnv.IntegEuler()
            integ.step["cpu"](d, env.mf)
            if Int(d.meta.data[META_IDX_NUM_CONTACTS]) >= 2:
                found = True
                break
        if not found:
            raise Error("no height within 8 cm under the reset pose puts the feet in the floor")
        print("[2] feet in the floor (root lowered by", -fdz, "m)")
        _leg(env, ctx, base, fdz, String("floor"), True, tl)

        print("[3] lanes are independent, and zero is nothing")
        var g0 = _gpu(env, ctx, base, 0.5, W_NONE)
        var gb = _gpu(env, ctx, base, 0.5, W_BOTH)
        var gl = _gpu(env, ctx, base, 0.5, W_LANE0)
        var again = _gpu(env, ctx, base, 0.5, W_NONE)
        var rep = _dist(g0, again, 0, len(g0))
        tl.check(rep[0] == 0, "the no-wrench run repeats bit for bit on the same env (" + String(rep[0]) + " values differ)")
        var same1 = 0
        var same0 = 0
        var moved = 0.0
        for part in range(2):
            var lo = part * NL * ROW
            same0 += _dist(gl, gb, lo, lo + ROW)[0]
            same1 += _dist(gl, g0, lo + ROW, lo + 2 * ROW)[0]
            moved = max(moved, _dist(gl, g0, lo, lo + ROW)[1])
        tl.check(same1 == 0, "wrenches on lane 0 only: lane 1 equals the no-wrench run's (" + String(same1) + " values differ)")
        tl.check(same0 == 0, "and lane 0 equals its values when lane 1 also had wrenches (" + String(same0) + " values differ)")
        tl.check(moved > 1e-2, "while lane 0 moved by " + String(moved) + " from the no-wrench run")

        print("[4] the env's own step reads the buffer")
        var e0 = _env_steps(env, ctx, base, 0.0, W_NONE)
        var e1 = _env_steps(env, ctx, base, 0.0, W_LANE0)
        var d0 = _dist(e1, e0, 0, ROW)
        var d1 = _dist(e1, e0, ROW, 2 * ROW)
        tl.check(d0[1] > 1e-3, String(ENV_STEPS) + " control steps: lane 0 moved by " + String(d0[1]) + " from the no-wrench run")
        tl.check(d1[0] == 0, "lane 1, with no wrench, did not (" + String(d1[0]) + " values differ)")

        print("\n===", tl.ok, "/", tl.total, "passed ===")
        if tl.ok != tl.total:
            raise Error("test_xfrc_g1_gpu: " + String(tl.total - tl.ok) + " failed")
