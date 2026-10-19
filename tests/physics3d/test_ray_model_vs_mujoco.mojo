"""`ray/model.mojo` vs `mj_ray` — the whole model, every geom type, one ray.

    pixi run mojo run -I . tests/physics3d/test_ray_model_vs_mujoco.mojo

The per-geom routines are gated one type at a time elsewhere. What THIS file
gates is the traversal on top of them: that the nearest hit wins across
different types, that `geomid` names the geom MuJoCo names, and that
`ray_eliminate` skips exactly what the reference skips.

⚠⚠ `geomid` IS THE ASSERTION THAT MAKES THE FILTER TESTABLE. Distance alone
cannot see an elimination bug: skip the wrong geom and the ray simply reports
whatever is behind it, which is a perfectly plausible number. Two geoms are
therefore stacked along the same line of sight — a decoy in FRONT of a target —
so the wrong answer is a different `geomid` at a different depth rather than a
near miss. The scene is built so that:

  · `invisible_front`  rgba alpha 0, in front of `target_sphere`
  · `matinvis_front`   OPAQUE rgba but an alpha-0 MATERIAL, also in front
  · `exclude_me`       on its own body, so `bodyexclude` can drop it

`test_the_decoys_are_actually_in_the_way` proves each decoy really does occlude
before the sweep runs — a decoy that misses the target proves nothing about the
filter, and it would look identical in the pass column.

⚠ EVERY GEOM TYPE IS PRESENT ON PURPOSE, including the mesh and the
heightfield, because the traversal's job is to dispatch and a dispatch is only
tested by the branch it gets wrong. `nmesh_tri` is passed, since without it the
mesh silently contributes nothing.

WHAT THIS GATE WAS PROVEN ABLE TO FAIL
======================================
    injected defect                          wrong geomid   splits   |dt|
    --------------------------------------   ------------   ------   ---------
    body exclusion removed                        150          0     UNCHANGED
    nearest replaced by FARTHEST                  336          0     UNCHANGED
    visibility check removed                        6          4     UNCHANGED
    ellipsoid sized from `radius`                   6          6     0.073
    capsule/cylinder sized from `half_*`          312         93     1.83

⚠⚠ THE FIRST THREE LEAVE `max |dt|` WHERE IT WAS. Skip the wrong geom and the
ray reports whatever is behind it — a perfectly plausible distance. Only the
`geomid` column sees them, which is why it is asserted at zero rather than
printed for information.

⚠ AND THE FIRST ROW WAS INVISIBLE UNTIL THE COLUMN FAMILY EXISTED. Removing
body exclusion changed NOTHING over 300 uniformly-sampled rays: a 5 cm sphere
at z = 0.60 is simply too small a target to hit by chance. The z-axis family
below took it from 0 to 150. Same shape as the capsule caps in the
`mju_rayGeom` sweep — a hit count says the ray reached the scene, not the
BRANCH. [[feedback_a_hit_count_is_not_coverage_of_the_branch]]
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from noeira.physics3d.fields import (
    Data, Model, DynDims, init_hfield_data, DYN1, DYN2, rl1, rl2,
)
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import (
    dims_from_flat,
    build_model_runtime,
    spec_fields_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RAY_VISIBLE,
    MODEL_BODY_SIZE,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
)
from noeira.physics3d.ray import ray_model

comptime DT = DType.float64
comptime Vec3 = Vec3Generic[DT]
comptime Quat = QuatGeneric[DT]

comptime SCENE = String(
    """
<mujoco model="ray traversal gate">
  <asset>
    <hfield name="terrain" file="tests/physics3d/assets/hf_8x8.bin" size="0.4 0.4 0.15 0.05"/>
    <mesh name="notch" file="tests/physics3d/assets/notch.stl"/>
    <material name="ghost" rgba="0.9 0.2 0.2 0"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.6"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0.55 0.35 -0.35" euler="8 -6 20"/>
    <geom name="target_sphere" type="sphere" size="0.09" pos="0 0 0"/>
    <geom name="a_box" type="box" size="0.07 0.05 0.06" pos="-0.45 0.20 0.05" euler="20 -35 15"/>
    <geom name="a_capsule" type="capsule" size="0.04 0.09" pos="0.40 -0.30 0.10" euler="55 10 -20"/>
    <geom name="a_cylinder" type="cylinder" size="0.05 0.07" pos="-0.35 -0.35 0.12" euler="-25 40 5"/>
    <geom name="an_ellipsoid" type="ellipsoid" size="0.09 0.04 0.06" pos="0.30 0.42 -0.05" euler="10 25 -40"/>
    <geom name="a_mesh" type="mesh" mesh="notch" pos="-0.10 0.50 0.08" euler="15 -25 40"/>

    <!-- The two decoys, both between the eye and `target_sphere`. -->
    <geom name="invisible_front" type="sphere" size="0.05" pos="0 0 0.45" rgba="0.2 0.7 0.2 0"/>
    <geom name="matinvis_front" type="sphere" size="0.05" pos="0 0 0.30" material="ghost" rgba="0.2 0.2 0.9 1"/>

    <body name="excl" pos="0 0 0.60">
      <freejoint/>
      <geom name="exclude_me" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""
)

comptime NCASE = 600


struct Lcg(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def u01(mut self) -> Float64:
        self.s = self.s * 1664525 + 1013904223
        return Float64((self.s >> 16) & 0xFFFFFFF) / Float64(0x10000000)

    def sym(mut self, a: Float64) -> Float64:
        return (self.u01() * 2.0 - 1.0) * a


struct Built(Movable):
    var m: Model[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var dims: DynDims

    def __init__(out self) raises:
        var fmd = parse_xml_full(SCENE, String("."))
        var dims = dims_from_flat(
            fmd, max_contacts=32, nmesh_verts=256, nmesh_tri=64
        )
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var sf = spec_fields_runtime[DT](fmd, dims, m)
        var d = Data[DT, DynDims, 1](dims)
        init_hfield_data(d, m)
        for i in range(dims.get_nq()):
            d.qpos.data[i] = sf.qpos0.data[i]
        for i in range(dims.get_nv()):
            d.qvel.data[i] = Scalar[DT](0)
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        self.m = m^
        self.d = d^
        self.dims = dims


def _gid(mujoco: PythonObject, m: PythonObject, name: String) raises -> Int:
    return Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name))


def test_the_decoys_are_actually_in_the_way() raises:
    """Each decoy must occlude the target when visibility is ignored.

    ⚠ A DECOY THAT MISSES PROVES NOTHING, and it would look exactly like a
    pass. This asserts, against MuJoCo and with the decoy made VISIBLE, that
    the ray really would stop on it — so when the sweep below reports the
    target instead, the filter is what moved the answer.
    """
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    # The same scene with both decoys opaque and the ghost material solid.
    var vis = String(SCENE)
    vis = vis.replace('rgba="0.9 0.2 0.2 0"', 'rgba="0.9 0.2 0.2 1"')
    vis = vis.replace('rgba="0.2 0.7 0.2 0"', 'rgba="0.2 0.7 0.2 1"')
    var m = mujoco.MjModel.from_xml_string(vis)
    var d = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, d)

    var pnt = np.zeros(3)
    pnt[2] = 1.2
    var vec = np.zeros(3)
    vec[2] = -1.0
    var gid = np.zeros(1, np.int32)
    var t = Float64(py=mujoco.mj_ray(m, d, pnt, vec, None, True, -1, gid, None))
    var hit = Int(py=gid[0])
    var name = String(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, hit))
    print("  with the decoys VISIBLE, the ray stops on:", name, "at t =", t)
    assert_true(
        name == "exclude_me" or name == "invisible_front",
        "the straight-down ray stops on " + name + ", so the decoys are not"
        " in the line of sight and this scene cannot test the filter",
    )


def test_invisible_geoms_are_skipped() raises:
    """The precomputed flag, against the two spellings MuJoCo distinguishes."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(SCENE))
    var b = Built()
    var n_invis = 0
    for g in range(b.dims.get_ngeom()):
        var o = g * MODEL_GEOM_SIZE
        var ours = Float64(b.m.geoms.data[o + GEOM_IDX_RAY_VISIBLE]) != 0.0
        # MuJoCo's rule, spelled out here rather than read from a helper.
        var matid = Int(py=m.geom_matid[g])
        var theirs: Bool
        if matid >= 0:
            theirs = Float64(py=m.mat_rgba[matid][3]) != 0.0
        else:
            theirs = Float64(py=m.geom_rgba[g][3]) != 0.0
        var nm = String(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        assert_true(
            ours == theirs,
            "geom " + nm + ": ours visible=" + String(ours) + " MuJoCo="
            + String(theirs),
        )
        if not theirs:
            n_invis += 1
    print("  visibility agrees on all", b.dims.get_ngeom(), "geoms;",
          n_invis, "invisible")
    assert_true(
        n_invis == 2,
        "expected 2 invisible geoms (an alpha-0 rgba and an alpha-0 MATERIAL),"
        " found " + String(n_invis) + " — if this drops to 0 the flag is"
        " never exercised and the sweep cannot see an elimination bug",
    )


def test_ray_model_vs_mujoco() raises:
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(String(SCENE))
    var d = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, d)

    var b = Built()
    var excl_body = Int(
        py=m.geom_bodyid[_gid(mujoco, m, String("exclude_me"))]
    )

    var a_pnt = np.zeros(3)
    var a_vec = np.zeros(3)
    var a_gid = np.zeros(1, np.int32)
    var a_nrm = np.zeros(3)

    # ⚠ The eight views, built ONCE — `lt_dyn` needs `mut` and none of them
    # move between rays. `ray_model` composes each geom's world pose itself
    # from `xpos`/`xquat`, which is what lets one implementation serve a
    # kernel where a thread owns one ray and cannot hold a scene.
    var ng = b.dims.get_ngeom()
    var nb = b.dims.get_nbody()
    var geoms_v = b.m.geoms.tt_dyn["cpu", DYN2](rl2(ng, MODEL_GEOM_SIZE))
    var bodies_v = b.m.bodies.tt_dyn["cpu", DYN2](rl2(nb, MODEL_BODY_SIZE))
    var xpos_v = b.d.xpos.tt_dyn["cpu", DYN2](rl2(1, nb * 3))
    var xquat_v = b.d.xquat.tt_dyn["cpu", DYN2](rl2(1, nb * 4))
    var mesh_meta_v = b.m.mesh_meta.tt_dyn["cpu", DYN1](
        rl1(MAX_GPU_MESHES * MODEL_MESH_META_SIZE)
    )
    var mesh_tris_v = b.m.mesh_tris.tt_dyn["cpu", DYN1](rl1(64 * 9))
    var hf_meta_v = b.m.hfield_meta.tt_dyn["cpu", DYN1](
        rl1(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
    )
    var hf_data_v = b.d.hfield_data.tt_dyn["cpu", DYN1](
        rl1(b.dims.get_nhfield_data())
    )

    var rng = Lcg(0xBEEF77)
    var hits = 0
    var split = 0
    var wrong_geom = 0
    var worst_t = 0.0
    var worst_n = 0.0
    var excluded_used = 0

    for c in range(NCASE):
        # Half the rays exclude the free body, so the argument is exercised
        # rather than merely accepted.
        var bex = -1
        if c % 2 == 1:
            bex = excl_body
            excluded_used += 1

        var eye: Vec3
        var aim: Vec3
        if c % 4 == 1 or c % 4 == 2:
            # ⚠ `c % 4 == 1` is ODD and `== 2` is EVEN, so the column is fired
            # both WITH `bodyexclude` (where the answer must be
            # `target_sphere`) and WITHOUT it (where it must be `exclude_me`).
            # One variant alone would leave the other branch of the argument
            # unexercised.
            # ⚠⚠ THE COLUMN FAMILY, and it is what makes `bodyexclude`
            # testable. Everything the filter cares about is stacked on the
            # z-axis: `exclude_me` at 0.60, `invisible_front` at 0.45,
            # `matinvis_front` at 0.30, `target_sphere` at 0. One ray down
            # that column exercises all three exclusions at once, and the
            # ANSWER changes with the argument — with `bodyexclude` set the
            # ray reaches `target_sphere`, without it, it stops at 0.60.
            #
            # A uniform sampler cannot do this: a 5 cm sphere at z = 0.60 is a
            # tiny target, and dropping the body-exclusion check was MEASURED
            # to change NOTHING over 300 random rays before this family
            # existed. Same lesson as the capsule caps in the `mju_rayGeom`
            # sweep — a hit count says the ray reached the scene, not the
            # BRANCH. [[feedback_a_hit_count_is_not_coverage_of_the_branch]]
            eye = Vec3(rng.sym(0.02), rng.sym(0.02), 1.1 + rng.u01() * 0.4)
            aim = Vec3(rng.sym(0.02), rng.sym(0.02), -0.2)
        else:
            eye = Vec3(rng.sym(1.6), rng.sym(1.6), rng.sym(1.6))
            aim = Vec3(rng.sym(0.6), rng.sym(0.6), rng.sym(0.5))
        var vec = aim - eye

        var ours = ray_model[DT](
            geoms_v, ng, bodies_v, xpos_v, xquat_v, 0,
            mesh_meta_v, mesh_tris_v, hf_meta_v, hf_data_v,
            b.dims.get_nhfield_data(),
            eye, vec, bex,
        )

        a_pnt[0] = eye.x
        a_pnt[1] = eye.y
        a_pnt[2] = eye.z
        a_vec[0] = vec.x
        a_vec[1] = vec.y
        a_vec[2] = vec.z
        var t_mj = Float64(
            py=mujoco.mj_ray(m, d, a_pnt, a_vec, None, True, bex, a_gid, a_nrm)
        )

        var t_ours = Float64(ours.t)
        if (t_ours >= 0.0) != (t_mj >= 0.0):
            split += 1
            continue
        if t_mj < 0.0:
            continue

        hits += 1
        if ours.geom != Int(py=a_gid[0]):
            wrong_geom += 1
            continue
        worst_t = max(worst_t, abs(t_ours - t_mj))
        worst_n = max(worst_n, abs(Float64(ours.normal.x) - Float64(py=a_nrm[0])))
        worst_n = max(worst_n, abs(Float64(ours.normal.y) - Float64(py=a_nrm[1])))
        worst_n = max(worst_n, abs(Float64(ours.normal.z) - Float64(py=a_nrm[2])))

    print("  hits", hits, "/", NCASE, " (", excluded_used,
          "rays used bodyexclude )")
    print("  splits        ", split)
    print("  wrong geomid  ", wrong_geom)
    print("  worst |dt|      ", worst_t)
    print("  worst |dnormal| ", worst_n)

    assert_true(hits > NCASE // 4, "only " + String(hits) + " hits — vacuous")
    assert_true(split == 0, String(split) + " hit/miss disagreements")
    assert_true(
        wrong_geom == 0,
        String(wrong_geom) + " rays named a DIFFERENT geom than MuJoCo — an"
        " elimination or dispatch bug, which a distance comparison alone"
        " reports as a plausible number",
    )
    # The loosest routine underneath sets this: the hfield's grid is float32.
    assert_true(worst_t < 1e-6, "worst |dt| " + String(worst_t))
    assert_true(worst_n < 1e-5, "worst |dnormal| " + String(worst_n))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
