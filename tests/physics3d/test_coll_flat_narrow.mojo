"""The flat narrow phase emits the block kernel's contacts, bit for bit.

    pixi run -e nvidia mojo run -I . tests/physics3d/test_coll_flat_narrow.mojo
    pixi run -e apple  mojo run -I . tests/physics3d/test_coll_flat_narrow.mojo

`COLL_FLAT_NARROW` (`collision/ccd_workspace.mojo`) replaces the SAP block
kernel's phase 2 — an env's candidates on the lanes of one warp — with four
launches: the block kernel's listing into `Data.coll_flat` and its global
queues, a narrow phase over them (a HOT pair alone in a warp, cold pairs 32 to
a warp) and the block kernel's phase 3 on the result. Only WHICH
warp and CCD row runs a pair may change; every candidate goes through the same
`_sap_block_candidate` into the same staging window and out through the same
`_sap_block_output`. So the gate is equality: both paths on one `Data`, the
same poses, every field of every contact and every `ncon` identical.

Which pairs run HOT is decided from `coll_flat`'s per-pair cost slots, which
the narrow phase itself writes on NVIDIA. Left to that, a single call would
schedule everything cold (the slots start at 0) and the hot path would never
run — so each case SEEDS the slots before the call:

- all 0: every candidate cold (32 to a warp, across env boundaries);
- all huge: every candidate hot, in bucket 3 (one warp each);
- a spread over the four buckets and cold: the queues' bucket ordering.

and asserts the queues it expects were non-empty — vacuity is the default
failure. The CCD workspace is re-uploaded (zeroed) before every call, so the
hill climb's cross-step warm slots start equal on both paths.

The fixture is `test_mesh_manifold_gpu_parity`'s five box/mesh and mesh/mesh
groups — real hulls, penetrating, near face-to-face so the clipper runs —
plus a box resting on a floor plane for plane candidates, over 8 envs with
different poses so the queues cross env boundaries.
"""

from std.math import sqrt, sin, cos
from std.testing import assert_true, assert_equal, TestSuite
from max.gpu.host import DeviceContext

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.broadphase_sap import detect_contacts_sap
from noeira.physics3d.collision.ccd_workspace import (
    CF_ROW, CF_COST, CF_NHOT, CF_NCOLD, COLL_FLAT_HOT_NS, HILL_WARM_SLOTS,
)
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE,
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
)


comptime DTYPE = DType.float32

comptime FLAT_XML = """
<mujoco model="flat narrow">
  <option timestep="0.002"/>
  <asset>
    <mesh name="cube" file="tests/physics3d/assets/mc_cube.stl"/>
    <mesh name="hex" file="tests/physics3d/assets/mc_hex.stl"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 .1"/>
    <body name="a0" pos="0 0 0.5">
      <geom name="g0a" type="mesh" mesh="cube"/>
    </body>
    <body name="b0" pos="0 0 0.5">
      <joint name="j0" type="free"/>
      <geom name="g0b" type="box" size=".05 .04 .06"/>
    </body>
    <body name="a1" pos="2 0 0.5">
      <geom name="g1a" type="box" size=".05 .04 .06"/>
    </body>
    <body name="b1" pos="2 0 0.5">
      <joint name="j1" type="free"/>
      <geom name="g1b" type="mesh" mesh="cube"/>
    </body>
    <body name="a2" pos="4 0 0.5">
      <geom name="g2a" type="mesh" mesh="cube"/>
    </body>
    <body name="b2" pos="4 0 0.5">
      <joint name="j2" type="free"/>
      <geom name="g2b" type="mesh" mesh="cube"/>
    </body>
    <body name="a3" pos="6 0 0.5">
      <geom name="g3a" type="mesh" mesh="hex"/>
    </body>
    <body name="b3" pos="6 0 0.5">
      <joint name="j3" type="free"/>
      <geom name="g3b" type="box" size=".05 .04 .06"/>
    </body>
    <body name="a4" pos="8 0 0.5">
      <geom name="g4a" type="mesh" mesh="hex"/>
    </body>
    <body name="b4" pos="8 0 0.5">
      <joint name="j4" type="free"/>
      <geom name="g4b" type="mesh" mesh="cube"/>
    </body>
    <body name="c" pos="10 0 0.05">
      <joint name="jc" type="free"/>
      <geom name="gc" type="box" size=".05 .05 .05"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime fm = parse_xml(FLAT_XML)
comptime FM = ModelDefFromXML[
    xml=FLAT_XML,
    nbody=fm.NBODY, njoint=fm.NJOINT, nq=fm.NQ, nv=fm.NV,
    ngeom=fm.NGEOM, nact=fm.NACT, ntex=fm.NTEX, nmat=fm.NMAT,
    nlight=fm.NLIGHT, ncam=fm.NCAM, nsite=fm.NSITE,
    max_tendon=fm.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=64,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=fm.TIMESTEP,
]

comptime NQ: Int = FM.NQ
comptime MD = ModelDims[FM, 64]
comptime BATCH: Int = 8
comptime NGROUP: Int = 5
comptime MCON: Int = FM.MAX_CONTACTS

comptime Dat = Data[DTYPE, MD, BATCH]
comptime Mod = Model[DTYPE, MD]


def _stack_z(g: Int) -> Float64:
    """Face-to-face separation of group `g`'s two geoms when aligned."""
    if g == 0:
        return 0.05 + 0.06
    if g == 1:
        return 0.06 + 0.05
    if g == 2:
        return 0.05 + 0.05
    if g == 3:
        return 0.08 + 0.06
    return 0.08 + 0.05


struct Lcg(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64((self.s >> 11) & 0x1FFFFFFFFFFFFF) / 9007199254740992.0

    def sym(mut self, a: Float64) -> Float64:
        return (self.next() * 2.0 - 1.0) * a


# ⚠ NO HELPER TAKES THE `Data`. A `def` with a `mut d: Dat` parameter crashes
# the compiler for this model (reproduced on a clean tree, 2026-09-27: a one-
# line body is enough), so the helpers build plain lists and `_case` copies
# them in.


def _free_qpos(
    mut q: List[Float64], qo: Int, px: Float64, py: Float64, pz: Float64,
    ang: Float64, mut rng: Lcg,
):
    var ax = rng.sym(1.0)
    var ay = rng.sym(1.0)
    var az = rng.sym(1.0)
    var an = sqrt(ax * ax + ay * ay + az * az)
    if an < 1e-9:
        ax = 1.0
        ay = 0.0
        az = 0.0
        an = 1.0
    var s = sin(0.5 * ang) / an
    # free-joint qpos is (x, y, z, qw, qx, qy, qz)
    q[qo + 0] = px
    q[qo + 1] = py
    q[qo + 2] = pz
    q[qo + 3] = cos(0.5 * ang)
    q[qo + 4] = ax * s
    q[qo + 5] = ay * s
    q[qo + 6] = az * s


def _poses(mut rng: Lcg) -> List[Float64]:
    """Every env its own pose: the five groups interpenetrating by 2-5 mm,
    aligned, tilted a little or tilted more (so some pairs clip a face and
    some do not), and the lone box 1-3 mm into the floor. `[env * NQ + k]`."""
    var q = List[Float64](length=BATCH * NQ, fill=0.0)
    for e in range(BATCH):
        for g in range(NGROUP):
            var regime = (e + g) % 3
            var ang: Float64
            if regime == 0:
                ang = 0.0
            elif regime == 1:
                ang = rng.sym(0.008)
            else:
                ang = rng.sym(0.09)
            var pen = 0.002 + 0.003 * rng.next()
            var px = Float64(g) * 2.0 + rng.sym(0.01)
            var py = rng.sym(0.01)
            _free_qpos(q, e * NQ + g * 7, px, py, 0.5 + _stack_z(g) - pen, ang, rng)
        var cx = 10.0 + rng.sym(0.01)
        var cy = rng.sym(0.01)
        var cz = 0.05 - 0.001 - 0.002 * rng.next()
        var ca = rng.sym(0.02)
        _free_qpos(q, e * NQ + NGROUP * 7, cx, cy, cz, ca, rng)
    return q^


def _costs(mode: Int, mut rng: Lcg) -> List[Float64]:
    """The cost slots, `[env * HILL_WARM_SLOTS + slot]`. 0: all cold; 1: all
    bucket 3; 2: spread over cold and the four buckets."""
    var v = List[Float64](length=BATCH * HILL_WARM_SLOTS, fill=0.0)
    for i in range(len(v)):
        if mode == 1:
            v[i] = 1.0e9
        elif mode == 2:
            var r = Int(rng.next() * 5.0)
            if r > 0:
                # bucket r - 1: [2^(r-1), 2^r) x HOT_NS
                v[i] = Float64(COLL_FLAT_HOT_NS) * Float64(1 << (r - 1)) * 1.5
    return v^


def _case(mode: Int, name: String) raises:
    var ctx = DeviceContext()
    var mf = Mod()
    FM.init_fields[DTYPE](ctx, mf)
    var d = Dat()
    var rng = Lcg(UInt64(0x9E3779B97F4A7C15) + UInt64(mode))
    var q = _poses(rng)
    for i in range(BATCH * NQ):
        d.qpos.data[i] = Scalar[DTYPE](q[i])
    d.upload_all(ctx)
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)

    # The block kernel, from a zeroed CCD workspace.
    d.ccd_ws.upload(ctx)
    detect_contacts_sap["gpu", DTYPE, BATCH=BATCH, FLAT=False](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)
    ctx.synchronize()
    var blk = List[Float64]()
    for e in range(BATCH):
        var n = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        blk.append(Float64(n))
        for k in range(n * CONTACT_SIZE):
            blk.append(Float64(d.contacts.data[e * MCON * CONTACT_SIZE + k]))

    # The flat path, from the same zeroed workspace and the seeded costs.
    # (`ccd_ws`'s host copy was never downloaded, so it is still zeros.)
    var cost = _costs(mode, rng)
    for e in range(BATCH):
        for k in range(HILL_WARM_SLOTS):
            d.coll_flat.data[e * CF_ROW + CF_COST + k] = Scalar[DTYPE](
                cost[e * HILL_WARM_SLOTS + k]
            )
    d.ccd_ws.upload(ctx)
    d.coll_flat.upload(ctx)
    detect_contacts_sap["gpu", DTYPE, BATCH=BATCH, FLAT=True](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)
    d.coll_flat.download(ctx)
    ctx.synchronize()
    var got = List[Float64]()
    for e in range(BATCH):
        var n = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        got.append(Float64(n))
        for k in range(n * CONTACT_SIZE):
            got.append(Float64(d.contacts.data[e * MCON * CONTACT_SIZE + k]))

    # The queues this case was meant to fill, from the envs' own counts (the
    # global counters are zeroed by the output kernel once consumed).
    var nh = 0
    var nc = 0
    for e in range(BATCH):
        for k in range(4):
            nh += Int(d.coll_flat.data[e * CF_ROW + CF_NHOT + k])
        nc += Int(d.coll_flat.data[e * CF_ROW + CF_NCOLD])
    print("  ", name, ": hot tasks", nh, " cold tasks", nc)
    if mode == 0:
        assert_true(nh == 0 and nc > 0, name + ": expected every task cold")
    elif mode == 1:
        assert_true(nh > 0, name + ": expected hot tasks")
    else:
        assert_true(nh > 0 and nc > 0, name + ": expected both queues")

    var total = 0
    var multi = 0
    var i = 0
    for e in range(BATCH):
        var nr = Int(blk[i])
        assert_equal(
            Int(got[i]), nr, name + ": env " + String(e) + " ncon flat vs block"
        )
        total += nr
        if nr > NGROUP + 1:
            multi += 1
        i += 1
        for k in range(nr * CONTACT_SIZE):
            assert_true(
                got[i + k] == blk[i + k],
                name + ": env " + String(e) + " contact "
                + String(k // CONTACT_SIZE) + " field "
                + String(k % CONTACT_SIZE) + ": flat " + String(got[i + k])
                + " vs block " + String(blk[i + k]),
            )
        i += nr * CONTACT_SIZE
    print("  ", name, ": contacts", total, " envs with a manifold", multi)
    assert_true(
        total >= BATCH * (NGROUP + 1),
        name + ": too few contacts to mean anything",
    )
    assert_true(multi > 0, name + ": no env ran the clipper")


def test_all_cold() raises:
    _case(0, "all cold")


def test_all_hot() raises:
    _case(1, "all hot")


def test_mixed_buckets() raises:
    _case(2, "mixed buckets")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
