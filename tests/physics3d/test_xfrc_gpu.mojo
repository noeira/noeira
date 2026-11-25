"""A per-lane external wrench on the device (`Data.xfrc_applied`, `XFRC=True`):
the kernel against the CPU pass, lane by lane, and the batched env's step.

What `mj_xfrcAccumulate` is and its float64 gate against MuJoCo:
`noeira/physics3d/dynamics/xfrc.mojo`, `tests/physics3d/test_xfrc_vs_mujoco.mojo`.
This file asks the device questions on models that build for Metal and
NVIDIA alike; the batched G1 (NVIDIA only) has its own file,
`test_xfrc_g1_gpu.mojo`.

[1]-[3]: the Ant (a free-jointed torso, eight hinges, feet on a plane), 2
lanes, the Euler integrator a batched env would build, with `XFRC=True`. The
reference is the SAME integrator type stepped on the CPU in float32 from the
same state and wrenches. The two targets differ before any wrench (float32
rounding, contact solves), so the bars are relative, as in the float64 gate
(written before the first run):

    e0 = device vs CPU with no wrench, e = with the wrenches,
    effect = what the wrenches do (CPU with vs without)
    e <= max(10 e0, 1e-5)   and   1000 e <= effect

  [1] in the air, after 1 and 5 steps: each lane its own wrenches (a force
      and a torque on the torso, a torque on a middle body, a force at a
      point of the last body; lane 1 other values and another attitude)
  [2] feet in the floor, after 1 and 5 steps
  [3] lanes are independent and zero is nothing: with wrenches on lane 0
      only, lane 1 equals the no-wrench run's lane 1 BIT FOR BIT, and lane 0
      equals its values in the run where lane 1 also had wrenches

[4]: the env's own step reads the buffer. `Phyics3dBatchedEnv[...,
XFRC=True]` on HalfCheetah (an Euler env without RNE_POST), 5 control steps
at a zero action, a wrench on lane 0's torso only: lane 0 departs from the
no-wrench env, lane 1 does not, bit for bit.

Not measured: a device `XFRC=True` build with an all-zero buffer against a
device `XFRC=False` build. The float64 gate measures it for the same
function, and [3] measures the skip per lane.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_xfrc_gpu.mojo  (or -e nvidia)
"""

from std.math import abs

from max.gpu.host import DeviceContext

from noeira.envs.ant.ant_xml import AntModel
from noeira.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from noeira.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.nn.constants import DT
from noeira.physics3d.dynamics.xfrc import XFRC_STRIDE, xfrc_at_point
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.model.model_dims import ModelDims


comptime NL = 2
comptime AM = AntModel
comptime MD = ModelDims[AM]
comptime NQ = AM.NQ
comptime NV = AM.NV
comptime ROW = NQ + NV
comptime AData = Data[DT, MD, NL]
comptime AModel = Model[DT, MD]
# the integrator `Phyics3dBatchedEnv` builds for an Euler config, plus XFRC
comptime AInteg = EulerIntegrator[
    DT, MD, AM.CONE_TYPE, NL, SOLVER="newton", PARALLEL_GPU=True, CRBA_TREEWALK=True,
    MAX_CONDIM=AM.MAX_CONDIM, NOSLIP_ITER=AM.NOSLIP_ITER, XFRC=True,
]
comptime STEPS_A = 1
comptime STEPS_B = 5
comptime AIR_Z = 1.5
comptime W_NONE = 0
comptime W_BOTH = 1
comptime W_LANE0 = 2

comptime HEnv = Phyics3dBatchedEnv[HalfCheetahModel, HalfCheetahConfig, NL, TERMINATE_ON_UNHEALTHY=False, XFRC=True]
comptime H_STEPS = 5


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


def _state(z: Float64, mut d: AData):
    """Both lanes at rest, the torso at height z, the ankles inside their
    ranges; lane 1 tilted and with other hip angles."""
    var ank: List[Float64] = [0.87, -0.87, -0.87, 0.87]
    for l in range(NL):
        for k in range(NQ):
            d.qpos.data[l * NQ + k] = Scalar[DT](0)
        for k in range(NV):
            d.qvel.data[l * NV + k] = Scalar[DT](0)
        d.qpos.data[l * NQ + 2] = Scalar[DT](z)
        d.qpos.data[l * NQ + 3] = Scalar[DT](1.0)
        for j in range(4):
            d.qpos.data[l * NQ + 7 + 2 * j] = Scalar[DT](0.1 * Float64(j - 1) * Float64(1 - 2 * l))
            d.qpos.data[l * NQ + 7 + 2 * j + 1] = Scalar[DT](ank[j])
    # lane 1: a rotation of 0.2 rad about (0.6, 0.8, 0)
    d.qpos.data[NQ + 3] = Scalar[DT](0.9950041652780258)
    d.qpos.data[NQ + 4] = Scalar[DT](0.6 * 0.09983341664682815)
    d.qpos.data[NQ + 5] = Scalar[DT](0.8 * 0.09983341664682815)
    d.qpos.data[NQ + 6] = Scalar[DT](0.0)


def _wrenches(mut d: AData, mut mf: AModel, which: Int) raises:
    """`d.xfrc_applied` for the state in `d` (host): none, both lanes, or lane 0 only."""
    for i in range(d.xfrc_applied.n):
        d.xfrc_applied.data[i] = Scalar[DT](0)
    if which == W_NONE:
        return
    forward_kinematics["cpu", DT, BATCH=NL](d, mf, None)
    var nb = d.dims.get_nbody()
    var torso = 1
    var mid = nb // 2
    var last = nb - 1
    for l in range(NL):
        if which == W_LANE0 and l > 0:
            continue
        var s = 1.0 if l == 0 else -0.6
        var o = (l * nb + torso) * XFRC_STRIDE
        d.xfrc_applied.data[o + 0] = Scalar[DT](3.0 * s)
        d.xfrc_applied.data[o + 1] = Scalar[DT](-2.0)
        d.xfrc_applied.data[o + 2] = Scalar[DT](4.0)
        d.xfrc_applied.data[o + 3] = Scalar[DT](0.3)
        d.xfrc_applied.data[o + 4] = Scalar[DT](-0.2 * s)
        d.xfrc_applied.data[o + 5] = Scalar[DT](0.5)
        var om = (l * nb + mid) * XFRC_STRIDE
        d.xfrc_applied.data[om + 3] = Scalar[DT](0.2 * s)
        d.xfrc_applied.data[om + 4] = Scalar[DT](0.4)
        d.xfrc_applied.data[om + 5] = Scalar[DT](-0.3)
        xfrc_at_point(
            d, l, last, 1.5, -2.5 * s, 2.0,
            Float64(d.xipos.data[(l * nb + last) * 3 + 0]) + 0.05,
            Float64(d.xipos.data[(l * nb + last) * 3 + 1]) - 0.04,
            Float64(d.xipos.data[(l * nb + last) * 3 + 2]) + 0.03,
        )


def _rows(d: AData, mut out: List[Scalar[DT]]):
    for l in range(NL):
        for k in range(NQ):
            out.append(d.qpos.data[l * NQ + k])
        for k in range(NV):
            out.append(d.qvel.data[l * NV + k])


def _cpu(mut mf: AModel, z: Float64, which: Int, mut ncon: List[Int]) raises -> List[Scalar[DT]]:
    """The lanes' (qpos, qvel) after STEPS_A then STEPS_B CPU steps; lane 0's
    contact count at the end is appended to `ncon`."""
    var d = AData()
    _state(z, d)
    _wrenches(d, mf, which)
    var integ = AInteg()
    var out = List[Scalar[DT]]()
    for t in range(1, STEPS_B + 1):
        integ.step["cpu"](d, mf)
        if t == STEPS_A or t == STEPS_B:
            _rows(d, out)
    ncon.append(Int(d.meta.data[META_IDX_NUM_CONTACTS]))
    return out^


def _gpu(ctx: DeviceContext, mut mf: AModel, z: Float64, which: Int) raises -> List[Scalar[DT]]:
    """The same on the device, on a fresh `Data` and integrator."""
    var d = AData()
    _state(z, d)
    _wrenches(d, mf, which)
    d.upload_all(ctx)
    var integ = AInteg()
    integ.prepare_gpu(ctx)
    var out = List[Scalar[DT]]()
    for t in range(1, STEPS_B + 1):
        integ.step["gpu"](d, mf, ctx)
        if t == STEPS_A or t == STEPS_B:
            d.qpos.download(ctx)
            d.qvel.download(ctx)
            _rows(d, out)
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


def _leg(ctx: DeviceContext, mut mf: AModel, z: Float64, label: String, contact: Bool, mut tl: Tally) raises:
    var ncon = List[Int]()
    var c0 = _cpu(mf, z, W_NONE, ncon)
    var c1 = _cpu(mf, z, W_BOTH, ncon)
    var g0 = _gpu(ctx, mf, z, W_NONE)
    var g1 = _gpu(ctx, mf, z, W_BOTH)
    tl.check((ncon[0] > 0) == contact, label + ": lane 0 has " + String(ncon[0]) + " contacts with no wrench at the end")
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


def _cheetah(ctx: DeviceContext, wrench: Bool) raises -> List[Scalar[DT]]:
    """A fresh HalfCheetah env, the same seed, a wrench on lane 0's torso or
    none; every lane's (qpos, qvel) after H_STEPS control steps at a zero action."""
    comptime HQ = HEnv.NQ
    comptime HV = HEnv.NV
    comptime HA = HEnv.ACT_DIM
    var env = HEnv(ctx)
    env.reset_batch[NL](Optional(ctx), 0)
    _ = env.xfrc_ptr()
    if wrench:
        var o = 1 * XFRC_STRIDE      # lane 0, body 1 (the torso)
        env.d.xfrc_applied.data[o + 0] = Scalar[DT](30.0)
        env.d.xfrc_applied.data[o + 2] = Scalar[DT](20.0)
        env.d.xfrc_applied.data[o + 4] = Scalar[DT](3.0)
        env.d.xfrc_applied.upload_resident(ctx)
    var ha = ctx.enqueue_create_host_buffer[DT](NL * HA)
    ctx.synchronize()
    for i in range(NL * HA):
        ha[i] = Scalar[DT](0)
    for t in range(H_STEPS):
        ctx.enqueue_copy(env._action, ha)
        env.step_batch[NL](Optional(ctx), UInt64(t))
    env.d.qpos.download(ctx)
    env.d.qvel.download(ctx)
    var out = List[Scalar[DT]]()
    for l in range(NL):
        for k in range(HQ):
            out.append(env.d.qpos.data[l * HQ + k])
        for k in range(HV):
            out.append(env.d.qvel.data[l * HV + k])
    return out^


def main() raises:
    var ctx = DeviceContext()
    var tl = Tally()
    var mf = AModel()
    AM.init_fields[DT](ctx, mf)
    print("Ant: nq", NQ, "nv", NV)

    print("[1] in the air: the device kernel against the CPU pass, per lane")
    _leg(ctx, mf, AIR_Z, String("air"), False, tl)

    # the floor leg's height: the first drop (1 cm steps) with contacts after a CPU step
    var fz = 0.0
    var found = False
    for i in range(60):
        fz = 0.80 - 0.01 * Float64(i)
        var d = AData()
        _state(fz, d)
        var integ = AInteg()
        integ.step["cpu"](d, mf)
        if Int(d.meta.data[META_IDX_NUM_CONTACTS]) >= 2:
            found = True
            break
    if not found:
        raise Error("no torso height from 0.80 to 0.21 m puts the Ant's feet in the floor")
    print("[2] feet in the floor (torso at", fz, "m)")
    _leg(ctx, mf, fz, String("floor"), True, tl)

    print("[3] lanes are independent, and zero is nothing")
    var g0 = _gpu(ctx, mf, AIR_Z, W_NONE)
    var gb = _gpu(ctx, mf, AIR_Z, W_BOTH)
    var gl = _gpu(ctx, mf, AIR_Z, W_LANE0)
    var again = _gpu(ctx, mf, AIR_Z, W_NONE)
    var rep = _dist(g0, again, 0, len(g0))
    tl.check(rep[0] == 0, "the no-wrench run repeats bit for bit (" + String(rep[0]) + " values differ)")
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

    print("[4] the batched env's own step reads the buffer (HalfCheetah)")
    var h0 = _cheetah(ctx, False)
    var h1 = _cheetah(ctx, True)
    var hrow = len(h0) // NL
    var d0 = _dist(h1, h0, 0, hrow)
    var d1 = _dist(h1, h0, hrow, 2 * hrow)
    tl.check(d0[1] > 1e-3, String(H_STEPS) + " control steps: lane 0 moved by " + String(d0[1]) + " from the no-wrench env")
    tl.check(d1[0] == 0, "lane 1, with no wrench, did not (" + String(d1[0]) + " values differ)")

    print("\n===", tl.ok, "/", tl.total, "passed ===")
    if tl.ok != tl.total:
        raise Error("test_xfrc_gpu: " + String(tl.total - tl.ok) + " failed")
