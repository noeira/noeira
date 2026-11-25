"""A per-lane mass for free bodies (`Data.lane_mass`, an Euler integrator
built with `LANE_MASS=True`; `noeira/physics3d/dynamics/lane_mass.mojo`)
against the same mass written in the MODEL, float64, CPU, 2 lanes.

    pixi run mojo run -I . tests/physics3d/test_lane_mass.mojo

The fixture: a tilted gravity, a hinged arm that falls onto a crate (a free
box), a smaller free box resting on the crate, and a free cart that carries a
hinged flag. The crate slides; contacts join the floor, the crate, the box
on it and the arm, so the mass matrix's blocks, the bias forces and the
contact rows' weights of the scaled bodies are all in play.

The TWIN of a lane is a plain batch (`LANE_MASS=False`) whose model holds
that lane's masses: each scaled body's mass, inertia and their inverses, and
its `body_invweight0`. (Not the model's mean inertia, which a lane keeps:
see the module's header.) Bars, written before the first run:

  [1] scales that are powers of two (lane 0: crate x2, box x4; lane 1:
      crate x4, box x0.5): each lane equals its twin BIT FOR BIT after 1 and
      40 steps. Vacuity: there are contacts, and after 40 steps the lanes are
      not the unscaled model's.
  [2] a scale of 3 on the crate: within 1e-9 of its twin after 40 steps, and
      not the unscaled model's.

The first run (2026-11-25) asked the vacuity of [1] and the bound of [2]
after ONE step, where no mass can show: from rest, free bodies under gravity
and resting contacts accelerate alike whatever they weigh (the scaled lanes
equalled the unscaled model to 0.0 there). Both are read after 40 steps
now, once the arm has hit the crate; [2]'s bound went from 1e-12 at one step
to 1e-9 at 40 (measured 9e-16).
  [3] no scale, and a scale of exactly 1: the plain build, bit for bit.
  [4] the solver's weights are live: a twin whose model holds the mass and
      inertia but NOT the weights differs from the lane.
  [5] the setter refuses the world, a hinged body, a free body with a child,
      that child, a scale of 0 or below, and a lane outside the batch.
"""

from std.math import abs
from std.testing import assert_true, TestSuite

from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.expander import expand_mjcf
from noeira.physics3d.parser.runtime_load import dims_from_flat, build_model_runtime, spec_fields_runtime
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.dynamics.lane_mass import lane_mass_reset, set_lane_mass_scale
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.studio.stepping import STUDIO_MAX_CONDIM, STUDIO_NOSLIP, studio_cone_of, studio_uses_implicit
from noeira.physics3d.types import ConeType
from noeira.physics3d.gpu.constants import (
    BODY_IDX_INV_IXX, BODY_IDX_INV_MASS, BODY_IDX_IXX, BODY_IDX_MASS, META_IDX_NUM_CONTACTS, METADATA_SIZE, MODEL_BODY_SIZE,
)

comptime DT = DType.float64
comptime NL = 2
comptime STEPS_A = 1
comptime STEPS_B = 40
# bodies of the fixture (world = 0)
comptime ARM = 1
comptime CRATE = 2
comptime BOX = 3
comptime CART = 4
comptime FLAG = 5

comptime XML = String(
    """<mujoco model="lane mass">
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
)


def _scale_model(mut m: Model[DT, DynDims], body: Int, s: Float64, weights: Bool):
    """The model holds `body`'s mass times `s`: mass, inertia, their inverses,
    and with `weights` its solver weight."""
    var o = body * MODEL_BODY_SIZE
    m.bodies.data[o + BODY_IDX_MASS] = m.bodies.data[o + BODY_IDX_MASS] * s
    m.bodies.data[o + BODY_IDX_INV_MASS] = m.bodies.data[o + BODY_IDX_INV_MASS] / s
    for k in range(3):
        m.bodies.data[o + BODY_IDX_IXX + k] = m.bodies.data[o + BODY_IDX_IXX + k] * s
        m.bodies.data[o + BODY_IDX_INV_IXX + k] = m.bodies.data[o + BODY_IDX_INV_IXX + k] / s
    if weights:
        for k in range(2):
            m.body_invweight0.data[body * 2 + k] = m.body_invweight0.data[body * 2 + k] / s


def _roll[LM: Bool](
    lane_scales: List[Float64], model_scales: List[Float64], model_weights: Bool, mut ncon: List[Int],
) raises -> List[Float64]:
    """2 lanes from the same start (lane 1's crate a little slower).
    `lane_scales`: rows (lane, body, scale) given to `set_lane_mass_scale`
    (LM builds only); `model_scales`: rows (body, scale) written in the
    model. Per lane, (qpos, qvel) after STEPS_A then STEPS_B steps; the
    largest contact count seen is appended to `ncon`."""
    var fmd = parse_xml_full(expand_mjcf(XML, String("")), String(""))
    assert_true(not studio_uses_implicit(fmd) and studio_cone_of(fmd) == ConeType.PYRAMIDAL, "the fixture: Euler, pyramidal")
    assert_true(String(fmd.body_names[CRATE]) == "crate" and String(fmd.body_names[CART]) == "cart", "the fixture's body order")
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    for r in range(len(model_scales) // 2):
        _scale_model(m, Int(model_scales[r * 2]), model_scales[r * 2 + 1], model_weights)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, NL](dims)
    for l in range(NL):
        for i in range(nq):
            d.qpos.data[l * nq + i] = sf.qpos0.data[i]
        for i in range(nv):
            d.qvel.data[l * nv + i] = Scalar[DT](0)
            d.qfrc.data[l * nv + i] = Scalar[DT](0)
        d.qvel.data[l * nv + 1] = Scalar[DT](0.3 - 0.1 * Float64(l))      # the crate's x velocity
    comptime if LM:
        lane_mass_reset(d, m)
        for r in range(len(lane_scales) // 3):
            set_lane_mass_scale(d, m, Int(lane_scales[r * 3]), Int(lane_scales[r * 3 + 1]), lane_scales[r * 3 + 2])
    var integ = EulerIntegrator[
        DT, DynDims, ConeType.PYRAMIDAL, NL, "newton",
        MAX_CONDIM=STUDIO_MAX_CONDIM, NOSLIP_ITER=STUDIO_NOSLIP, CRBA_TREEWALK=True, LANE_MASS=LM,
    ](dims)
    var out = List[Float64]()
    var most = 0
    for t in range(1, STEPS_B + 1):
        integ.step["cpu"](d, m)
        for l in range(NL):
            most = max(most, Int(d.meta.data[l * METADATA_SIZE + META_IDX_NUM_CONTACTS]))
        if t == STEPS_A or t == STEPS_B:
            for l in range(NL):
                for i in range(nq):
                    out.append(Float64(d.qpos.data[l * nq + i]))
                for i in range(nv):
                    out.append(Float64(d.qvel.data[l * nv + i]))
    ncon.append(most)
    return out^


def _lane(a: List[Float64], part: Int, lane: Int) -> List[Float64]:
    """Lane `lane`'s values of part 0 (after STEPS_A) or 1 (after STEPS_B)."""
    var row = len(a) // (2 * NL)
    var out = List[Float64]()
    for i in range(row):
        out.append(a[(part * NL + lane) * row + i])
    return out^


def _diff(a: List[Float64], b: List[Float64]) -> Tuple[Int, Float64]:
    var n = 0
    var w = 0.0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
            w = max(w, abs(a[i] - b[i]))
    return (n, w)


def test_powers_of_two_equal_the_model_bit_for_bit() raises:
    print("=== lane scales against the same masses in the model ===")
    var nc = List[Int]()
    var none = List[Float64]()
    var lanes: List[Float64] = [0.0, Float64(CRATE), 2.0, 0.0, Float64(BOX), 4.0, 1.0, Float64(CRATE), 4.0, 1.0, Float64(BOX), 0.5]
    var got = _roll[True](lanes, none, True, nc)
    var tw0: List[Float64] = [Float64(CRATE), 2.0, Float64(BOX), 4.0]
    var tw1: List[Float64] = [Float64(CRATE), 4.0, Float64(BOX), 0.5]
    var twin0 = _roll[False](none, tw0, True, nc)
    var twin1 = _roll[False](none, tw1, True, nc)
    var plain = _roll[False](none, none, True, nc)
    print("  largest contact count in a lane:", nc[0], " values per lane and part:", len(got) // (2 * NL))
    assert_true(nc[0] >= 4, "the fixture has too few contacts to test the contact rows")
    for part in range(2):
        var d0 = _diff(_lane(got, part, 0), _lane(twin0, part, 0))
        var d1 = _diff(_lane(got, part, 1), _lane(twin1, part, 1))
        var e0 = _diff(_lane(got, part, 0), _lane(plain, part, 0))
        var e1 = _diff(_lane(got, part, 1), _lane(plain, part, 1))
        print("  part", part, ": lane 0 vs its twin", d0[0], "differ (max", d0[1], "); lane 1", d1[0], "(max", d1[1], "); against the unscaled model: max", e0[1], "and", e1[1])
        assert_true(d0[0] == 0 and d1[0] == 0, "a lane differs from its model-edited twin")
        if part == 1:
            assert_true(e0[1] > 1e-6 and e1[1] > 1e-6, "the scales do nothing readable after " + String(STEPS_B) + " steps")
    print("  PASS")


def test_a_scale_of_three() raises:
    print("=== a scale that is not a power of two ===")
    var nc = List[Int]()
    var none = List[Float64]()
    var lanes: List[Float64] = [0.0, Float64(CRATE), 3.0, 1.0, Float64(CRATE), 3.0]
    var tw: List[Float64] = [Float64(CRATE), 3.0]
    var got = _roll[True](lanes, none, True, nc)
    var twin = _roll[False](none, tw, True, nc)
    var plain = _roll[False](none, none, True, nc)
    for part in range(2):
        var d0 = _diff(_lane(got, part, 0), _lane(twin, part, 0))
        var d1 = _diff(_lane(got, part, 1), _lane(twin, part, 1))
        var e0 = _diff(_lane(got, part, 0), _lane(plain, part, 0))
        print("  part", part, ": lane 0 max |d|", d0[1], " lane 1", d1[1], " against the unscaled model:", e0[1])
        if part == 1:
            assert_true(d0[1] < 1e-9 and d1[1] < 1e-9, String(STEPS_B) + " steps at a scale of 3 are not within rounding of the model")
            assert_true(e0[1] > 1e-6, "a scale of 3 does nothing readable")
    print("  PASS")


def test_no_scale_is_the_plain_build() raises:
    print("=== no scale, and a scale of 1 ===")
    var nc = List[Int]()
    var none = List[Float64]()
    var ones: List[Float64] = [0.0, Float64(CRATE), 1.0, 1.0, Float64(BOX), 1.0]
    var a = _roll[True](none, none, True, nc)
    var b = _roll[True](ones, none, True, nc)
    var plain = _roll[False](none, none, True, nc)
    var da = _diff(a, plain)
    var db = _diff(b, plain)
    print("  values", len(plain), " differing with no scale:", da[0], " with scales of 1:", db[0])
    assert_true(len(plain) > 0 and da[0] == 0 and db[0] == 0, "a LANE_MASS build with nothing scaled is not the plain build")
    print("  PASS")


def test_the_weights_are_live() raises:
    print("=== the solver's weights: a twin without them differs ===")
    var nc = List[Int]()
    var none = List[Float64]()
    var lanes: List[Float64] = [0.0, Float64(CRATE), 2.0, 0.0, Float64(BOX), 4.0, 1.0, Float64(CRATE), 2.0, 1.0, Float64(BOX), 4.0]
    var tw: List[Float64] = [Float64(CRATE), 2.0, Float64(BOX), 4.0]
    var got = _roll[True](lanes, none, True, nc)
    var with_w = _roll[False](none, tw, True, nc)
    var without = _roll[False](none, tw, False, nc)
    var dw = _diff(got, with_w)
    var dn = _diff(got, without)
    print("  against the twin with the weights:", dw[0], "differ; without them:", dn[0], "differ, max", dn[1])
    assert_true(dw[0] == 0, "the lanes differ from the twin that holds the weights")
    assert_true(dn[0] > 0 and dn[1] > 1e-9, "the weights change nothing here: the gate above did not test them")
    print("  PASS")


def test_the_setter_refuses() raises:
    print("=== set_lane_mass_scale refuses what the argument does not hold for ===")
    var fmd = parse_xml_full(expand_mjcf(XML, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, NL](dims)
    lane_mass_reset(d, m)
    # (lane, body, scale): the world, a hinged body, a free body with a child, that child, bad scales, a bad lane
    var bad: List[Float64] = [
        0.0, 0.0, 2.0, 0.0, Float64(ARM), 2.0, 0.0, Float64(CART), 2.0, 0.0, Float64(FLAG), 2.0,
        0.0, Float64(CRATE), 0.0, 0.0, Float64(CRATE), -1.0, 2.0, Float64(CRATE), 2.0, 0.0, 6.0, 2.0,
    ]
    var refused = 0
    for r in range(len(bad) // 3):
        try:
            set_lane_mass_scale(d, m, Int(bad[r * 3]), Int(bad[r * 3 + 1]), bad[r * 3 + 2])
            print("  NOT refused: lane", bad[r * 3], "body", bad[r * 3 + 1], "scale", bad[r * 3 + 2])
        except e:
            refused += 1
    var touched = 0
    for i in range(d.lane_mass.n):
        if d.lane_mass.data[i] != 0:
            touched += 1
    print("  refused", refused, "of", len(bad) // 3, "; scales written by the refused calls:", touched)
    assert_true(refused == len(bad) // 3 and touched == 0, "a call that must be refused was accepted")
    set_lane_mass_scale(d, m, 1, CRATE, 2.5)
    assert_true(Float64(d.lane_mass.data[dims.get_nbody() + CRATE]) == 2.5, "an accepted call did not write its scale")
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
