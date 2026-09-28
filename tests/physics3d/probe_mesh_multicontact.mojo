"""INSTRUMENTATION probe for the native mesh multicontact path (not a gate).

One exactly-aligned pose per pair ordering, through the LIVE routine — not a
reconstruction of it. Reconstructing a collision call's inputs by hand already
produced one false negative in this repo (defect 27, where the rebuilt
`clcorner` was 0 and the live one was 1), so this drives
`gjk_epa_witness` + `native_multicontact_contacts` exactly as the narrow phase
does and prints what they decide, next to MuJoCo's answer for the same pose.

Set `MC_DEBUG = True` in `collision/native_multicontact.mojo` to see the branch
trace. Delete this file once the path is correct and gated.

Run: pixi run mojo run -I . tests/physics3d/probe_mesh_multicontact.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from max.gpu.host import DeviceContext
from layout import Layout

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.physics3d.fields import Data, Model, Dims, AsStatic
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.ccd_workspace import L_CCD_WS1
from noeira.physics3d.collision.ccd_workspace_host import ccd_ws_alloc
from noeira.physics3d.collision.gjk import gjk_epa_witness
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.collision.native_multicontact import (
    native_multicontact_contacts,
)
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_MESH_META_SIZE,
    MODEL_MESH_POLY_SIZE,
    MAX_GPU_MESHES,
    MESH_META_IDX_POLYADR,
    MESH_META_IDX_POLYNUM,
    mesh_max_poly,
    mesh_max_polyvert,
    mesh_max_edge,
)
from noeira.physics3d.gpu.constants import (
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_MESH_ID,
    GEOM_IDX_RBOUND,
)

comptime DTYPE = DType.float64

comptime PX_XML = """
<mujoco model="mc probe">
  <option timestep="0.002"/>
  <asset>
    <mesh name="cube" file="tests/physics3d/assets/mc_cube.stl"/>
  </asset>
  <worldbody>
    <body name="a" pos="0 0 0.5">
      <geom name="ga" type="mesh" mesh="cube"/>
    </body>
    <body name="b" pos="0 0 0.5">
      <joint name="jb" type="free"/>
      <geom name="gb" type="mesh" mesh="cube"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime px = parse_xml(PX_XML)
comptime PXM = ModelDefFromXML[
    xml=PX_XML,
    nbody=px.NBODY, njoint=px.NJOINT, nq=px.NQ, nv=px.NV,
    ngeom=px.NGEOM, nact=px.NACT, ntex=px.NTEX, nmat=px.NMAT,
    nlight=px.NLIGHT, ncam=px.NCAM, nsite=px.NSITE,
    max_tendon=px.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=px.TIMESTEP,
]

comptime NMESHV: Int = 32
comptime NQ: Int = PXM.NQ
comptime NV: Int = PXM.NV
comptime NBODY: Int = PXM.NBODY
comptime MC: Int = PXM.MAX_CONTACTS
comptime NGEOM: Int = PXM.NGEOM
comptime MD = ModelDims[PXM, 32]

comptime Dat = Data[DTYPE, MD, 1]
comptime Mod = Model[DTYPE, MD]


def main() raises:
    var sf = PXM.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Mod()
    PXM.init_fields[DTYPE](ctx, mf)
    var d = Dat()

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(PX_XML))
    var dat = mujoco.MjData(m)

    # Exactly aligned, 5 mm of penetration: cube half 0.05 each.
    var pen = 0.005
    var pz = 0.5 + 0.10 - pen
    PXM.reset_data(sf, d)
    d.qpos.data[0] = Scalar[DTYPE](0.0)
    d.qpos.data[1] = Scalar[DTYPE](0.0)
    d.qpos.data[2] = Scalar[DTYPE](pz)
    d.qpos.data[3] = Scalar[DTYPE](1.0)
    forward_kinematics["cpu"](d, mf)

    dat.qpos[0] = 0.0
    dat.qpos[1] = 0.0
    dat.qpos[2] = pz
    dat.qpos[3] = 1.0
    dat.qpos[4] = 0.0
    dat.qpos[5] = 0.0
    dat.qpos[6] = 0.0
    mujoco.mj_forward(m, dat)

    print("=== MuJoCo, mesh(cube) x mesh(cube), aligned, pen =", pen)
    print("  ncon =", Int(py=dat.ncon))
    for k in range(Int(py=dat.ncon)):
        var c = dat.contact[k]
        print(
            "   g", Int(py=c.geom1), "/", Int(py=c.geom2),
            " dist", Float64(py=c.dist),
            " pos", Float64(py=c.pos[0]), Float64(py=c.pos[1]),
            Float64(py=c.pos[2]),
            " n", Float64(py=c.frame[0]), Float64(py=c.frame[1]),
            Float64(py=c.frame[2]),
        )

    # ---- the live call ----------------------------------------------------
    comptime L_GEOM = Layout.row_major(NGEOM, MODEL_GEOM_SIZE)
    comptime L_MESH_META = Layout.row_major(
        MAX_GPU_MESHES, MODEL_MESH_META_SIZE
    )
    comptime L_MESH_VERT = Layout.row_major(NMESHV, 3)
    comptime L_MESH_POLY = Layout.row_major(
        mesh_max_poly(NMESHV), MODEL_MESH_POLY_SIZE
    )
    comptime L_MESH_PV = Layout.row_major(mesh_max_polyvert(NMESHV))
    comptime L_MESH_VEADR = Layout.row_major(NMESHV)
    comptime L_MESH_EDGE = Layout.row_major(mesh_max_edge(NMESHV))
    comptime L_MESH_VPM = Layout.row_major(NMESHV, 2)
    comptime L_CONTACTS = Layout.row_major(1, MC * CONTACT_SIZE)
    comptime L_B3 = Layout.row_major(1, NBODY * 3)
    comptime L_B4 = Layout.row_major(1, NBODY * 4)

    var geoms_v = mf.geoms.lt["cpu", L_GEOM]()
    var mesh_meta_v = mf.mesh_meta.lt["cpu", L_MESH_META]()
    var mesh_verts_v = mf.mesh_verts.lt["cpu", L_MESH_VERT]()
    var mesh_veadr_v = mf.mesh_vert_edgeadr.lt["cpu", L_MESH_VEADR]()
    var mesh_edges_v = mf.mesh_edges.lt["cpu", L_MESH_EDGE]()
    var mesh_polys_v = mf.mesh_polys.lt["cpu", L_MESH_POLY]()
    var mesh_pv_v = mf.mesh_polyvert.lt["cpu", L_MESH_PV]()
    var mesh_pm_v = mf.mesh_polymap.lt["cpu", L_MESH_PV]()
    var mesh_vpm_v = mf.mesh_vert_polymap.lt["cpu", L_MESH_VPM]()
    var contacts_v = d.contacts.lt["cpu", L_CONTACTS]()
    var xpos_v = d.xpos.lt["cpu", L_B3]()
    var xquat_v = d.xquat.lt["cpu", L_B4]()

    # geom world poses straight out of FK (both geoms sit at their body origin)
    var gi = 0
    var gj = 1
    var bi = Int(mf.geoms.data[gi * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
    var bj = Int(mf.geoms.data[gj * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
    var pix = Float64(d.xpos.data[bi * 3 + 0])
    var piy = Float64(d.xpos.data[bi * 3 + 1])
    var piz = Float64(d.xpos.data[bi * 3 + 2])
    var pjx = Float64(d.xpos.data[bj * 3 + 0])
    var pjy = Float64(d.xpos.data[bj * 3 + 1])
    var pjz = Float64(d.xpos.data[bj * 3 + 2])
    print("  our geom centres:", pix, piy, piz, " | ", pjx, pjy, pjz)

    var mi_id = Int(mf.geoms.data[gi * MODEL_GEOM_SIZE + GEOM_IDX_MESH_ID])
    var mj_id = Int(mf.geoms.data[gj * MODEL_GEOM_SIZE + GEOM_IDX_MESH_ID])
    var va1 = Int(mf.mesh_meta.data[mi_id * MODEL_MESH_META_SIZE + 0])
    var mnv1 = Int(mf.mesh_meta.data[mi_id * MODEL_MESH_META_SIZE + 1])
    var va2 = Int(mf.mesh_meta.data[mj_id * MODEL_MESH_META_SIZE + 0])
    var mnv2 = Int(mf.mesh_meta.data[mj_id * MODEL_MESH_META_SIZE + 1])
    var pa1 = Int(
        mf.mesh_meta.data[mi_id * MODEL_MESH_META_SIZE + MESH_META_IDX_POLYADR]
    )
    var pn1 = Int(
        mf.mesh_meta.data[mi_id * MODEL_MESH_META_SIZE + MESH_META_IDX_POLYNUM]
    )
    var pa2 = Int(
        mf.mesh_meta.data[mj_id * MODEL_MESH_META_SIZE + MESH_META_IDX_POLYADR]
    )
    var pn2 = Int(
        mf.mesh_meta.data[mj_id * MODEL_MESH_META_SIZE + MESH_META_IDX_POLYNUM]
    )
    var rb_i = Scalar[DTYPE](
        mf.geoms.data[gi * MODEL_GEOM_SIZE + GEOM_IDX_RBOUND]
    )
    var rb_j = Scalar[DTYPE](
        mf.geoms.data[gj * MODEL_GEOM_SIZE + GEOM_IDX_RBOUND]
    )
    var gi_type = Int(mf.geoms.data[gi * MODEL_GEOM_SIZE + GEOM_IDX_TYPE])
    var gj_type = Int(mf.geoms.data[gj * MODEL_GEOM_SIZE + GEOM_IDX_TYPE])

    var wf1 = Array[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var wf2 = Array[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var wxx = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    var wf_ok = 0
    var wfi = Array[Int, 6](fill=-1)
    var ws = ccd_ws_alloc[DTYPE]()
    var z = Scalar[DTYPE](0)
    var one = Scalar[DTYPE](1)
    var r = gjk_epa_witness[DTYPE](
        gi_type,
        Scalar[DTYPE](pix), Scalar[DTYPE](piy), Scalar[DTYPE](piz),
        z, z, z, one,
        z, z, z, z, z,
        mesh_verts_v, mesh_veadr_v, mesh_edges_v, va1, mnv1,
        gj_type,
        Scalar[DTYPE](pjx), Scalar[DTYPE](pjy), Scalar[DTYPE](pjz),
        z, z, z, one,
        z, z, z, z, z,
        va2, mnv2,
        wf1, wf2, wxx, wf_ok, wfi,
        ws.lt["cpu", L_CCD_WS1](), 0,
    )
    print("  EPA: dist", r[0], " n(gi->gj)", r[4], r[5], r[6], " ok", wf_ok)
    print("   x1", wxx[0], wxx[1], wxx[2], "  x2", wxx[3], wxx[4], wxx[5])
    for t in range(3):
        print("   wf1[", t, "]", wf1[t * 3], wf1[t * 3 + 1], wf1[t * 3 + 2],
              "   wf2[", t, "]", wf2[t * 3], wf2[t * 3 + 1], wf2[t * 3 + 2])

    var nc = 0
    var got = native_multicontact_contacts[
        DTYPE](
        0, bi, bj,
        gi_type,
        Scalar[DTYPE](pix), Scalar[DTYPE](piy), Scalar[DTYPE](piz),
        z, z, z, one, z, z, z, rb_i, va1, mnv1, pa1, pn1,
        gj_type,
        Scalar[DTYPE](pjx), Scalar[DTYPE](pjy), Scalar[DTYPE](pjz),
        z, z, z, one, z, z, z, rb_j, va2, mnv2, pa2, pn2,
        AsStatic[MD](),
        mesh_verts_v, mesh_polys_v, mesh_pv_v, mesh_pm_v, mesh_vpm_v,
        wf1, wf2, wxx, wfi,
        r[0],
        Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0.005),
        Scalar[DTYPE](0.0001), 3,
        # ⚠ THE OPERANDS ARE IN GEOM ORDER HERE, NOT MuJoCo'S. This probe
        # queries the pair as the model declares it, so the normal needs no
        # flip; the narrow phase passes `True` on the leg where it sorted the
        # operands to `type1 <= type2`. See `native_multicontact_contacts`.
        False,
        contacts_v, ws.lt["cpu", L_CCD_WS1](), 0, nc,
    )
    print("  OURS: multicontact wrote", got, "rows")
    for k in range(got):
        var o = k * CONTACT_SIZE
        print(
            "   pos", Float64(d.contacts.data[o + CONTACT_IDX_POS_X]),
            Float64(d.contacts.data[o + CONTACT_IDX_POS_Y]),
            Float64(d.contacts.data[o + CONTACT_IDX_POS_Z]),
            " n", Float64(d.contacts.data[o + CONTACT_IDX_NX]),
            Float64(d.contacts.data[o + CONTACT_IDX_NY]),
            Float64(d.contacts.data[o + CONTACT_IDX_NZ]),
        )
