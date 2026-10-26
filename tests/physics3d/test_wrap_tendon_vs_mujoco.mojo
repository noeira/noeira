"""Geom-wrapped spatial tendons, END TO END, against MuJoCo 3.10.0.

`test_mju_wrap_vs_mujoco` gates the GEOMETRY — one segment, one object, 18
poses. This file gates the ROUTING around it, which is a separate thing and
has its own ways to be wrong:

  * consuming `site-geom-site` as ONE step (`j += 2`), not two
  * advancing by 2 even when `mju_wrap` DECLINED — a pulley the tendon clears
    is skipped, and treating its entry as a waypoint would route the tendon
    through the object's centre
  * the arc's two ends belonging to the WRAP GEOM's body, so a tendon sliding
    over a fixed pulley contributes no moment there
  * summing `|x0-w0| + wlen + |w1-x1|`, with the ARC and not its chord

TWO FIXTURES, and they answer different questions:

  `assets/wrap_arm.xml`  — ours, one hinge, two tendons, swept through 12
      angles. The sweep CROSSES the wrap/no-wrap transition in both objects
      (16 of the 24 tendon-poses wrap, 8 run straight), which a fixed pose
      cannot do. ⚠ A gate built only from our own XML is blind to whatever we
      never thought to write, hence the second fixture.

  `iit_softfoot`         — Menagerie's, five tendons of 39 waypoints each
      threaded through 18 cylinders. This is the model the feature exists for
      and it was unopenable before it.

Regenerate the goldens: pixi run python scripts/dump_mujoco_wrap.py
Run: pixi run mojo run -I . tests/physics3d/test_wrap_tendon_vs_mujoco.mojo
"""

from noeira.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime,
)
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.expander import expand_mjcf
from noeira.physics3d.fields import (
    Data, Model, DynamicsScratch, DynDims, rl1, rl2, DYN1, DYN2,
)
from noeira.physics3d.fields.scratch import Scratch
from noeira.physics3d.dynamics.tendon import spatial_tendon_length_jac
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.gpu.constants import (
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
)
from tests.physics3d.wrap_goldens import (
    arm_sweep_goldens, softfoot_goldens, ARM_COLS, SOFTFOOT_COLS,
)


comptime DTYPE = DType.float64
comptime ARM = String("tests/physics3d/assets/wrap_arm.xml")
comptime SOFTFOOT = String(
    "references/mujoco_menagerie-main/iit_softfoot/scene.xml"
)


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def close(mut self, got: Float64, want: Float64, tol: Float64, msg: String):
        self.checks += 1
        var dd = got - want
        if dd < 0:
            dd = -dd
        if dd <= tol:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want, "|d|", dd)

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _dir_of(p: String) -> String:
    var i = p.rfind("/")
    return String(p[byte=0:i]) if i > 0 else String(".")


def _tendon_lengths(
    path: String,
    explicit: List[Float64],
    offset: Float64,
    mut out: List[Float64],
) raises:
    """Every spatial tendon's length, in model order.

    `explicit` sets qpos outright when non-empty; otherwise qpos is
    `qpos0 + offset`, uniformly.

    ⚠⚠ `expand_mjcf`, NOT `parse_model_runtime`. `iit_softfoot`'s scene
    composes the robot with `<attach>`, and `parse_model_runtime` resolves
    only `<include>` — it returns a ONE-BODY model with NO diagnostic, which
    is what this test hit first. Recorded here rather than worked around
    silently; the loader gap is real and separate from tendon wrapping.
    """
    var f = open(path, "r")
    var raw = f.read()
    f.close()
    var base = _dir_of(path)
    var fmd = parse_xml_full(expand_mjcf(raw, base), base)
    # ⚠ `nmesh_verts` IS A WORKSPACE BUDGET, NOT A MODEL PROPERTY, and it is
    # non-zero here only because softfoot's meshes now RESOLVE. While the
    # asset paths were broken they loaded as nothing, so no collidable hull
    # asked for room and the default 0 was enough — a gate passing because a
    # separate bug kept the work from happening. The studio discovers this by
    # doubling; a fixed budget with headroom is right for a fixture whose
    # meshes never change. Tendon routing does not read hulls at all.
    var dims = dims_from_flat(fmd, nmesh_verts=8192)
    var mf = Model[DTYPE, DynDims](dims)
    build_model_runtime[DTYPE](fmd, dims, mf)
    var d = Data[DTYPE, DynDims, 1](dims)
    var sc = DynamicsScratch[DTYPE, DynDims, 1](dims)

    var nq = dims.get_nq()
    var nv = dims.get_nv()
    for i in range(nq):
        var v: Float64
        if len(explicit) > 0:
            v = explicit[i]
        else:
            v = Float64(fmd.qpos0[i]) + offset
        d.qpos.data[i] = Scalar[DTYPE](v)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DTYPE](0)

    forward_kinematics["cpu", DTYPE, DynDims, 1](d, mf)
    compute_cdof["cpu", DTYPE, DynDims, 1](d, mf, sc)

    var rl_TEN = rl2(dims.get_ntendon(), MODEL_TENDON_SIZE)
    var rl_SITE = rl2(dims.get_nsite(), MODEL_SITE_SIZE)
    var rl_GEOM = rl2(dims.get_ngeom(), MODEL_GEOM_SIZE)
    var rl_BODY = rl2(dims.get_nbody(), MODEL_BODY_SIZE)
    var rl_JOINT = rl2(dims.get_njoint(), MODEL_JOINT_SIZE)
    var rl_META = rl1(MODEL_META_SIZE)
    var rl_B3 = rl2(1, dims.get_nbody() * 3)
    var rl_B4 = rl2(1, dims.get_nbody() * 4)
    var rl_CDOF = rl2(1, nv * 6)

    var ten_v = mf.tendons.tt_dyn["cpu", DYN2](rl_TEN)
    var site_v = mf.sites.tt_dyn["cpu", DYN2](rl_SITE)
    var geom_v = mf.geoms.tt_dyn["cpu", DYN2](rl_GEOM)
    var body_v = mf.bodies.tt_dyn["cpu", DYN2](rl_BODY)
    var joint_v = mf.joints.tt_dyn["cpu", DYN2](rl_JOINT)
    var meta_v = mf.meta.tt_dyn["cpu", DYN1](rl_META)
    var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
    var xpos_v = d.xpos.tt_dyn["cpu", DYN2](rl_B3)
    var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl_B4)
    var cdof_v = sc.cdof.tt_dyn["cpu", DYN2](rl_CDOF)

    var J = Scratch[Scalar[DTYPE], 0](nv, fill=Scalar[DTYPE](0))
    out.clear()
    for t in range(dims.get_ntendon()):
        var L = spatial_tendon_length_jac[DTYPE, 0, 1](
            0, t, dims, ten_v, site_v, geom_v, body_v, joint_v, meta_v,
            stcom_v, cdof_v, xpos_v, xquat_v, J,
        )
        # ⚠⚠ THE LENGTH-ONLY PATH MUST RETURN THE SAME NUMBER, BIT FOR BIT.
        # `flg_jac=False` skips the two `_contact_jacobian_row` calls per
        # sub-segment — and nothing else. It is the path
        # `dynamics/tendon_lengths.compute_tendon_lengths` takes for the
        # `<tendonpos>` sensor, and it is the reason that pass may hand in a
        # STALE `cdof`: if the flag ever started changing the length, a
        # sensor would silently report a tendon measured against last step's
        # dof frames. Checked HERE rather than in the sensor gate because
        # this file is the one that routes tendons round geoms — softfoot's
        # five 39-waypoint tendons included — where the two paths have the
        # most sub-segments to disagree over.
        var Jf = Scratch[Scalar[DTYPE], 0](nv, fill=Scalar[DTYPE](0))
        var Lf = spatial_tendon_length_jac[DTYPE, 0, 1](
            0, t, dims, ten_v, site_v, geom_v, body_v, joint_v, meta_v,
            stcom_v, cdof_v, xpos_v, xquat_v, Jf,
            flg_jac=False,
        )
        if Lf != L:
            raise Error(
                "flg_jac=False changed tendon " + String(t) + "'s length: "
                + String(Float64(Lf)) + " vs " + String(Float64(L))
                + " — the length-only path is not the same walk"
            )
        # And it must leave the moment arm alone, or a caller that reused a
        # scratch would pick up a half-built row.
        for i in range(nv):
            if Jf[i] != Scalar[DTYPE](0):
                raise Error(
                    "flg_jac=False wrote J_row[" + String(i) + "] = "
                    + String(Float64(Jf[i])) + "; it must leave it zero"
                )
        out.append(Float64(L))


def main() raises:
    var t = Tally()
    print("=== geom-wrapped spatial tendons vs MuJoCo 3.10.0 ===")

    # ⚠ f64 AND 1e-9. The wrap solve is iterative and the goldens carry
    # MuJoCo's full precision; a float32 pass here would be gated against its
    # own rounding rather than against MuJoCo.
    comptime TOL = 1e-9

    # ── fixture 1: the hinge sweep ────────────────────────────────────────
    print("--- assets/wrap_arm.xml, hinge swept through both transitions ---")
    var g = arm_sweep_goldens()
    var npose = len(g) // ARM_COLS
    var nwrapped = 0
    var nstraight = 0
    for p in range(npose):
        var b = p * ARM_COLS
        var angle = g[b + 0]
        var qp = List[Float64]()
        qp.append(angle)
        var got = List[Float64]()
        _tendon_lengths(ARM, qp, 0.0, got)
        t.truth(len(got) == 2, String("angle ", angle, ": two tendons"))
        t.close(got[0], g[b + 1], TOL, String("angle ", angle, " cylinder"))
        t.close(got[1], g[b + 3], TOL, String("angle ", angle, " sphere"))
        # `ten_wrapnum` 4 = the object wrapped, 2 = the tendon ran straight.
        if g[b + 2] > 3:
            nwrapped += 1
        else:
            nstraight += 1
        if g[b + 4] > 3:
            nwrapped += 1
        else:
            nstraight += 1

    # ⚠⚠ NON-VACUITY, AND IT IS THE POINT OF THE SWEEP. If every pose wrapped,
    # a routing loop that ALWAYS wrapped would pass; if none did, one that
    # NEVER wrapped would. Only a table containing both catches either.
    print("--- the sweep crosses the transition ---")
    t.truth(nwrapped >= 12, String("tendon-poses that WRAP: ", nwrapped))
    t.truth(nstraight >= 6, String("tendon-poses that run STRAIGHT: ",
                                   nstraight))

    # ── fixture 2: iit_softfoot ───────────────────────────────────────────
    # 39 waypoints per tendon, 18 wrap cylinders. Before this feature the
    # parser refused the model outright.
    print("--- iit_softfoot (Menagerie), 5 tendons x 18 wrap cylinders ---")
    var sg = softfoot_goldens()
    var nsf = len(sg) // SOFTFOOT_COLS
    for p in range(nsf):
        var b = p * SOFTFOOT_COLS
        var amp = sg[b + 0]
        var qp = List[Float64]()
        var got = List[Float64]()
        _tendon_lengths(SOFTFOOT, qp, amp, got)
        t.truth(len(got) == 5, String("offset ", amp, ": five tendons"))
        for k in range(5):
            t.close(got[k], sg[b + 1 + k], TOL,
                    String("offset ", amp, " tendon ", k))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_wrap_tendon_vs_mujoco: " + String(t.fails) + " failed"
        )
