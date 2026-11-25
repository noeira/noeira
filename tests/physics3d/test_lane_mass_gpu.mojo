"""A per-lane mass for free bodies on the device (`Data.lane_mass`,
`LANE_MASS=True`; `noeira/physics3d/dynamics/lane_mass.mojo`), float32, 2
lanes, on `test_lane_mass.mojo`'s fixture (the same XML, compiled here as a
static model: keep the two strings alike).

The float64 CPU gate shows that a lane's scale IS the model's mass. This
file asks it of the device kernels, with the sharpest reference a device
has: the same step on the device with the mass written in the MODEL.

  [1] scales that are powers of two: each device lane equals its device twin
      (a plain build, `LANE_MASS=False`, the model edited) BIT FOR BIT after
      40 steps; and is not the unscaled model's
  [2] no scale: the plain device build, bit for bit
  [3] the solver's weights are live on the device: a twin without them
      differs
  [4] `sync_lane_mass_weights`: scales written without their weights, then
      the device kernel, give the run whose weights the host setter wrote
  [5] the device against the CPU in float32, the same integrator type, with
      a scale of 3 in lane 1: e <= max(10 e0, 1e-5), e0 the same distance
      with no scale (the two targets differ before any scale)

On Metal the solve is the per-env Newton kernel; the blocked kernel runs on
NVIDIA only.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_lane_mass_gpu.mojo  (or -e nvidia)
"""

from std.math import abs

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.physics3d.dynamics.lane_mass import lane_mass_reset, set_lane_mass_scale, sync_lane_mass_weights
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.gpu.constants import (
    BODY_IDX_INV_IXX, BODY_IDX_INV_MASS, BODY_IDX_IXX, BODY_IDX_MASS, META_IDX_NUM_CONTACTS, METADATA_SIZE, MODEL_BODY_SIZE,
)
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.types import ConeType

comptime XML = """<mujoco model="lane mass">
  <option cone="pyramidal" timestep="0.002" gravity="0.6 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="4 4 1"/>
    <body name="arm" pos="0.45 0 0.3">
      <joint name="h" type="hinge" axis="0 1 0"/>
      <geom name="ga" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03" mass="0.6"/>
    </body>
    <body name="crate" pos="0.6 0 0.05">
      <freejoint/>
      <geom name="gc" type="box" size="0.08 0.06 0.05" mass="1"/>
    </body>
    <body name="box" pos="0.62 0.01 0.151">
      <freejoint/>
      <geom name="gt" type="box" size="0.04 0.04 0.05" mass="0.5"/>
    </body>
    <body name="cart" pos="-0.6 0 0.06">
      <freejoint/>
      <geom name="gk" type="box" size="0.06 0.06 0.06" mass="1.5"/>
      <body name="flag" pos="0 0 0.1">
        <joint name="f" type="hinge" axis="1 0 0"/>
        <geom name="gf" type="capsule" fromto="0 0 0 0 0 0.1" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

comptime _tp = parse_xml(XML)
comptime LMM = ModelDefFromXML[
    xml=XML,
    nbody=_tp.NBODY, njoint=_tp.NJOINT, nq=_tp.NQ, nv=_tp.NV,
    ngeom=_tp.NGEOM, nact=_tp.NACT, ntex=_tp.NTEX, nmat=_tp.NMAT,
    nlight=_tp.NLIGHT, ncam=_tp.NCAM, nsite=_tp.NSITE,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=32,
    obs_dim_override=4,
    obs_qpos_skip=0,
    timestep=_tp.TIMESTEP,
]
comptime MD = ModelDims[LMM]
comptime NL = 2
comptime NQ = LMM.NQ
comptime NV = LMM.NV
comptime ROW = NQ + NV
comptime STEPS = 40
comptime CRATE = 2
comptime BOX = 3
comptime LData = Data[DT, MD, NL]
comptime LModel = Model[DT, MD]
comptime Integ[LM: Bool] = EulerIntegrator[
    DT, MD, ConeType.PYRAMIDAL, NL, SOLVER="newton", PARALLEL_GPU=True, CRBA_TREEWALK=True,
    MAX_CONDIM=LMM.MAX_CONDIM, NOSLIP_ITER=LMM.NOSLIP_ITER, LANE_MASS=LM,
]
# how a run's weights are written
comptime W_SETTER = 0        # the host setter writes the scales and the weights
comptime W_KERNEL = 1        # the scales alone, then `sync_lane_mass_weights` on the device


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


def _model(ctx: DeviceContext, scales: List[Float64], weights: Bool) raises -> LModel:
    """The fixture's model, with `scales` (rows: body, scale) written in it:
    mass, inertia, their inverses, and with `weights` the solver's weight."""
    var m = LModel()
    LMM.init_fields[DT](ctx, m)
    for r in range(len(scales) // 2):
        var b = Int(scales[r * 2])
        var s = Scalar[DT](scales[r * 2 + 1])
        var o = b * MODEL_BODY_SIZE
        m.bodies.data[o + BODY_IDX_MASS] = m.bodies.data[o + BODY_IDX_MASS] * s
        m.bodies.data[o + BODY_IDX_INV_MASS] = m.bodies.data[o + BODY_IDX_INV_MASS] / s
        for k in range(3):
            m.bodies.data[o + BODY_IDX_IXX + k] = m.bodies.data[o + BODY_IDX_IXX + k] * s
            m.bodies.data[o + BODY_IDX_INV_IXX + k] = m.bodies.data[o + BODY_IDX_INV_IXX + k] / s
        if weights:
            for k in range(2):
                m.body_invweight0.data[b * 2 + k] = m.body_invweight0.data[b * 2 + k] / s
    if len(scales) > 0:
        m.bodies.upload_resident(ctx)
        m.body_invweight0.upload_resident(ctx)
    return m^


def _state(mut d: LData):
    """The XML's start, at rest but for the crate sliding (lane 1 slower)."""
    var q0: List[Float64] = [
        0.0,
        0.6, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0,
        0.62, 0.01, 0.151, 1.0, 0.0, 0.0, 0.0,
        -0.6, 0.0, 0.06, 1.0, 0.0, 0.0, 0.0,
        0.0,
    ]
    for l in range(NL):
        for i in range(NQ):
            d.qpos.data[l * NQ + i] = Scalar[DT](q0[i])
        for i in range(NV):
            d.qvel.data[l * NV + i] = Scalar[DT](0)
        d.qvel.data[l * NV + 1] = Scalar[DT](0.3 - 0.1 * Float64(l))


def _rows(d: LData, mut out: List[Scalar[DT]]):
    for l in range(NL):
        for k in range(NQ):
            out.append(d.qpos.data[l * NQ + k])
        for k in range(NV):
            out.append(d.qvel.data[l * NV + k])


def _gpu[LM: Bool](
    ctx: DeviceContext, mut m: LModel, lane_scales: List[Float64], how: Int, mut ncon: List[Int]
) raises -> List[Scalar[DT]]:
    """STEPS device steps; every lane's (qpos, qvel) at the end.
    `lane_scales`: rows (lane, body, scale), for an LM build."""
    var d = LData()
    _state(d)
    comptime if LM:
        lane_mass_reset(d, m)
        for r in range(len(lane_scales) // 3):
            var l = Int(lane_scales[r * 3])
            var b = Int(lane_scales[r * 3 + 1])
            if how == W_SETTER:
                set_lane_mass_scale(d, m, l, b, lane_scales[r * 3 + 2])
            else:
                d.lane_mass.data[l * LMM.NBODY + b] = Scalar[DT](lane_scales[r * 3 + 2])
    d.upload_all(ctx)
    comptime if LM:
        if how == W_KERNEL:
            sync_lane_mass_weights(d, m, ctx)
    var integ = Integ[LM]()
    integ.prepare_gpu(ctx)
    for _ in range(STEPS):
        integ.step["gpu"](d, m, ctx)
    d.qpos.download(ctx)
    d.qvel.download(ctx)
    d.meta.download(ctx)
    var most = 0
    for l in range(NL):
        most = max(most, Int(d.meta.data[l * METADATA_SIZE + META_IDX_NUM_CONTACTS]))
    ncon.append(most)
    var out = List[Scalar[DT]]()
    _rows(d, out)
    return out^


def _cpu(mut m: LModel, lane_scales: List[Float64]) raises -> List[Scalar[DT]]:
    """The same steps of the LANE_MASS integrator on the CPU."""
    var d = LData()
    _state(d)
    lane_mass_reset(d, m)
    for r in range(len(lane_scales) // 3):
        set_lane_mass_scale(d, m, Int(lane_scales[r * 3]), Int(lane_scales[r * 3 + 1]), lane_scales[r * 3 + 2])
    var integ = Integ[True]()
    for _ in range(STEPS):
        integ.step["cpu"](d, m)
    var out = List[Scalar[DT]]()
    _rows(d, out)
    return out^


def _dist(a: List[Scalar[DT]], b: List[Scalar[DT]], lo: Int, hi: Int) -> Tuple[Int, Float64]:
    var n = 0
    var w = 0.0
    for i in range(lo, hi):
        if a[i] != b[i]:
            n += 1
            w = max(w, abs(Float64(a[i]) - Float64(b[i])))
    return (n, w)


def main() raises:
    var ctx = DeviceContext()
    var tl = Tally()
    var nc = List[Int]()
    var none = List[Float64]()
    var base = _model(ctx, none, True)
    print("fixture: nq", NQ, "nv", NV, "bodies", LMM.NBODY)

    print("[1] device lanes against device twins (the mass in the model)")
    var lanes: List[Float64] = [0.0, Float64(CRATE), 2.0, 0.0, Float64(BOX), 4.0, 1.0, Float64(CRATE), 4.0, 1.0, Float64(BOX), 0.5]
    var got = _gpu[True](ctx, base, lanes, W_SETTER, nc)
    var tw0: List[Float64] = [Float64(CRATE), 2.0, Float64(BOX), 4.0]
    var tw1: List[Float64] = [Float64(CRATE), 4.0, Float64(BOX), 0.5]
    var m0 = _model(ctx, tw0, True)
    var m1 = _model(ctx, tw1, True)
    var twin0 = _gpu[False](ctx, m0, none, W_SETTER, nc)
    var twin1 = _gpu[False](ctx, m1, none, W_SETTER, nc)
    var plain = _gpu[False](ctx, base, none, W_SETTER, nc)
    var d0 = _dist(got, twin0, 0, ROW)
    var d1 = _dist(got, twin1, ROW, 2 * ROW)
    var e0 = _dist(got, plain, 0, ROW)
    var e1 = _dist(got, plain, ROW, 2 * ROW)
    tl.check(nc[0] >= 4, "contacts in a lane at the end: " + String(nc[0]))
    tl.check(d0[0] == 0, "lane 0 (crate x2, box x4) after " + String(STEPS) + " steps: " + String(d0[0]) + " of " + String(ROW) + " values differ from its twin (max " + String(d0[1]) + ")")
    tl.check(d1[0] == 0, "lane 1 (crate x4, box x0.5): " + String(d1[0]) + " differ (max " + String(d1[1]) + ")")
    tl.check(e0[1] > 1e-4 and e1[1] > 1e-4, "against the unscaled model: max " + String(e0[1]) + " and " + String(e1[1]))

    print("[2] no scale is the plain build")
    var zero = _gpu[True](ctx, base, none, W_SETTER, nc)
    var dz = _dist(zero, plain, 0, 2 * ROW)
    tl.check(dz[0] == 0, "LANE_MASS with nothing scaled: " + String(dz[0]) + " values differ from the plain build")

    print("[3] the solver's weights are live")
    var m0n = _model(ctx, tw0, False)
    var twin0n = _gpu[False](ctx, m0n, none, W_SETTER, nc)
    var dn = _dist(got, twin0n, 0, ROW)
    tl.check(dn[0] > 0 and dn[1] > 1e-6, "lane 0 against a twin without the weights: " + String(dn[0]) + " differ, max " + String(dn[1]))

    print("[4] the weights from the device kernel")
    var by_kernel = _gpu[True](ctx, base, lanes, W_KERNEL, nc)
    var dk = _dist(by_kernel, got, 0, 2 * ROW)
    tl.check(dk[0] == 0, "scales written alone, then sync_lane_mass_weights: " + String(dk[0]) + " values differ from the host setter's run")

    print("[5] the device against the CPU")
    var l3: List[Float64] = [1.0, Float64(CRATE), 3.0, 0.0, Float64(BOX), 2.0]
    var g3 = _gpu[True](ctx, base, l3, W_SETTER, nc)
    var c3 = _cpu(base, l3)
    var c0 = _cpu(base, none)
    for l in range(NL):
        var ee0 = _dist(zero, c0, l * ROW, (l + 1) * ROW)[1]
        var ee = _dist(g3, c3, l * ROW, (l + 1) * ROW)[1]
        var eff = _dist(c3, c0, l * ROW, (l + 1) * ROW)[1]
        var bar = max(10.0 * ee0, 1e-5)
        tl.check(ee <= bar, "lane " + String(l) + ": device vs CPU " + String(ee) + " (no scale " + String(ee0) + ", bar " + String(bar) + "), the scales' effect " + String(eff))

    print("\n===", tl.ok, "/", tl.total, "passed ===")
    if tl.ok != tl.total:
        raise Error("test_lane_mass_gpu: " + String(tl.total - tl.ok) + " failed")
