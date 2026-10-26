"""`jac_site` — the full 6 x nv site Jacobian — against MuJoCo's `mj_jacSite`.

STEP 1 OF THE dm_control PHASE 7 RESET PATH. The manipulation tasks place the
gripper with damped-least-squares IK over this matrix, so every later step (the
quaternion error chain, the DLS loop, the rejection sampler) sits on top of it.
Gating it alone, first, keeps a non-converging IK from being a three-way guess
between the Jacobian, the error term and the solver.

⚠ BOTH BLOCKS ARE COMPARED, SEPARATELY. The rotational half is where a frame
convention hides: it is the block the constraint code never needed, because
`_angular_jacobian_row` only ever dotted it with a torsion axis. A gate on
`jacp` alone would pass with `jacr` transposed, negated, or left-handed.

⚠ LAYOUT IS PINNED, NOT ASSUMED. MuJoCo returns 3 x nv row-major (`jacp[i +
k*nv]`) and so do we. Reading either as nv x 3 transposes it while preserving
the Frobenius norm, so the comparison is element-by-element and the poses are
deliberately asymmetric — at a symmetric or zero pose a transpose changes
nothing.

DOF-BRANCH COVERAGE, and what is NOT covered. `jac_point` walks `num_dof` per
joint: 1 for hinge/slide, 3 for ball, 6 for free.

  * quadruped fetch covers FREE (torso + ball) and HINGE, and carries one site
    welded to the WORLD — the `body_weldid == 0` early return.
  * ball_in_cup covers SLIDE, where `cdof`'s angular half is identically zero,
    so `jacr` must come back exactly zero and `jacp` a bare axis.
  * ⚠ BALL JOINTS (`num_dof == 3`) ARE NOT COVERED BY EITHER, because nothing
    in the tree has one — the same gap `envs/dm_control/gpu_reset.mojo`
    records. That branch is written but unmeasured; a model with a ball joint
    must extend this test rather than assume it.

⚠ ball_in_cup IS THE WEAKER OF THE TWO AND CANNOT SUBSTITUTE FOR quadruped.
Measured, not reasoned: flipping the sign of the cross-product term in `jacp`
as a negative control moved quadruped's worst error from 1.7e-15 to 2.22, and
left ball_in_cup at EXACTLY 0.0. With only slide joints the angular `cdof` is
zero, so the whole `cdof_ang x offset` correction degenerates and a bug in it
is invisible. If quadruped is ever dropped from this file, the remaining test
still passes while covering none of that arithmetic.

⚠ THE WORLD-WELDED SITE IS EXERCISED BUT NOT DISCRIMINATING. Its block comes
back all-zero and MuJoCo agrees, but deleting our `wbody == 0` early return
would also give zero — the ancestor walk finds no joints either way. It is a
faithfulness check on the transcription, not a trap that would catch its
removal.

Both sides are built from the SAME XML string (ours, via `materialize`), so a
disagreement is arithmetic, not a model difference.

Run with:
    pixi run mojo run -I . tests/physics3d/test_jac_site_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.collections import Array
from max.gpu.host import DeviceContext
from noeira.core.tiles import RowMajor

from noeira.envs.dm_control.ball_in_cup import (
    DMBallInCupModel,
)
from noeira.envs.dm_control.quadruped import (
    DMQuadrupedFetchModel,
)
from noeira.physics3d.fields import Model, Data, DynamicsScratch, Dims, DimsLike
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.dynamics.subtree_com import compute_subtree_com
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.jac_point import jac_site
from noeira.physics3d.fields.scratch import Scratch, cap
from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_SITE_SIZE,
    BODY_IDX_WELDID,
    JOINT_IDX_TYPE,
    SITE_IDX_BODY,
)
from noeira.physics3d.joint_types import JNT_FREE, JNT_SLIDE
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float64

# Both sides build the Jacobian from the same float64 FK products by the same
# multiply-adds, so the only freedom is the order of a handful of additions.
# Measured worst is printed below; 1e-14 is machine precision at these
# magnitudes, not a number fitted to the result.
comptime JAC_TOL: Float64 = 1e-14

# `site_xpos` must already agree or the two Jacobians are of DIFFERENT POINTS
# and any disagreement gets attributed to the wrong thing.
comptime POS_TOL: Float64 = 1e-12

comptime N_POSES: Int = 4


def _pose(p: Int, i: Int) -> Float64:
    """A spread of asymmetric qpos values — no symmetry, no zeros, and nowhere
    near qpos0, where repeated entries let a transpose or a sign error
    survive."""
    return (
        0.13 * Float64(p + 1)
        - 0.07 * Float64(i + 1)
        + 0.31 * Float64((p * 7 + i * 3) % 5)
        - 0.6
    )


def _sweep[D: DimsLike](
    label: String,
    mujoco: PythonObject,
    np: PythonObject,
    mm: PythonObject,
    dat: PythonObject,
    mut mf: Model[DTYPE, D],
    mut d: Data[DTYPE, D, 1],
    want_free: Bool,
    want_slide: Bool,
    want_world_site: Bool,
) raises:
    """Precondition-check one model, then compare every site at every pose."""
    assert_true(Int(py=mm.nv) == D.NV, label + ": nv mismatch")
    assert_true(Int(py=mm.nsite) == D.NSITE, label + ": nsite mismatch")

    comptime L_NB3 = RowMajor[1, D.NBODY * 3]
    comptime L_JNT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_BOD = RowMajor[D.NBODY, MODEL_BODY_SIZE]
    comptime L_MET = RowMajor[MODEL_META_SIZE]
    comptime L_CDOF = RowMajor[1, D.NV * 6]
    comptime L_SITE = RowMajor[D.NSITE, MODEL_SITE_SIZE]
    comptime L_SX = RowMajor[1, D.NSITE * 3]

    var bodies_v = mf.bodies.tt["cpu", D.NBODY, MODEL_BODY_SIZE]()
    var joints_v = mf.joints.tt["cpu", D.NJOINT, MODEL_JOINT_SIZE]()
    var sites_v = mf.sites.tt["cpu", D.NSITE, MODEL_SITE_SIZE]()
    var mmeta_v = mf.meta.tt["cpu", MODEL_META_SIZE]()

    # ── preconditions: this model really does carry what it is here for ──
    var n_free = 0
    var n_slide = 0
    for j in range(D.NJOINT):
        var jt = Int(rebind[Scalar[DTYPE]](joints_v[j, JOINT_IDX_TYPE]))
        if jt == JNT_FREE:
            n_free += 1
        elif jt == JNT_SLIDE:
            n_slide += 1
    var n_world_site = 0
    for s in range(D.NSITE):
        var sb = Int(rebind[Scalar[DTYPE]](sites_v[s, SITE_IDX_BODY]))
        if Int(rebind[Scalar[DTYPE]](bodies_v[sb, BODY_IDX_WELDID])) == 0:
            n_world_site += 1
    print(
        "  [" + label + "] nv", D.NV, " nsite", D.NSITE,
        " free", n_free, " slide", n_slide, " world-welded sites",
        n_world_site,
    )
    if want_free:
        assert_true(
            n_free > 0,
            label + ": no FREE joint — the 6-DOF branch is never entered and"
            " this model does not cover what the docstring claims it does",
        )
    if want_slide:
        assert_true(
            n_slide > 0,
            label + ": no SLIDE joint — the zero-angular-block case is never"
            " entered",
        )
    if want_world_site:
        assert_true(
            n_world_site > 0,
            label + ": no site welded to the world — the body_weldid == 0"
            " early return is never exercised",
        )

    # ── sweep ────────────────────────────────────────────────────────────
    var worst_p = 0.0
    var worst_r = 0.0
    var worst_where = String("none")
    var n_checked = 0
    var n_zero_blocks = 0

    for p in range(N_POSES):
        for i in range(D.NQ):
            dat.qpos[i] = _pose(p, i)
        for i in range(D.NV):
            dat.qvel[i] = 0.0
        # Free-joint quaternions must be unit, or MuJoCo renormalises and its
        # qpos then describes a different pose than the one we mirror in.
        mujoco.mj_normalizeQuat(mm, dat.qpos)
        mujoco.mj_forward(mm, dat)

        for i in range(D.NQ):
            d.qpos.data[i] = Scalar[DTYPE](Float64(py=dat.qpos[i]))
        for i in range(D.NV):
            d.qvel.data[i] = Scalar[DTYPE](0)

        # FK -> subtree_com -> cdof, explicitly. NOT via `step`: `step`
        # integrates, so its scratch would describe the POST-step pose while
        # `site_xpos` describes the pre-step one.
        forward_kinematics["cpu"](d, mf)
        var scratch = DynamicsScratch[DTYPE, D, 1]()
        compute_subtree_com["cpu"](d, mf)
        compute_cdof["cpu"](d, mf, scratch)

        var subtree_v = d.subtree_com.tt["cpu", 1, D.NBODY * 3]()
        var cdof_v = scratch.cdof.tt["cpu", 1, D.NV * 6]()
        var sxpos_v = d.site_xpos.tt["cpu", 1, D.NSITE * 3]()

        for s in range(D.NSITE):
            var dp = 0.0
            for k in range(3):
                var e = abs(
                    Float64(d.site_xpos.data[s * 3 + k])
                    - Float64(py=dat.site_xpos[s][k])
                )
                if e > dp:
                    dp = e
            assert_true(
                dp < POS_TOL,
                label + ": site_xpos already differs at pose " + String(p)
                + " site " + String(s) + " (" + String(dp) + ") — the"
                " Jacobians below would be of two different points",
            )

            var jp = Scratch[Scalar[DTYPE], 3 * cap[D.NV]()](
                3 * D.NV, fill=Scalar[DTYPE](0)
            )
            var jr = Scratch[Scalar[DTYPE], 3 * cap[D.NV]()](
                3 * D.NV, fill=Scalar[DTYPE](0)
            )
            jac_site[DTYPE, cap[D.NV]()](
                0, subtree_v, joints_v, bodies_v, mmeta_v, cdof_v,
                sites_v, sxpos_v, s, jp, jr, D.NV,
            )

            var mjp = np.zeros(3 * D.NV).reshape(3, D.NV)
            var mjr = np.zeros(3 * D.NV).reshape(3, D.NV)
            mujoco.mj_jacSite(mm, dat, mjp, mjr, s)
            var fp = mjp.flatten().tolist()
            var fr = mjr.flatten().tolist()

            var all_zero = True
            for k in range(3 * D.NV):
                var mp = Float64(py=fp[k])
                var mr = Float64(py=fr[k])
                if mp != 0.0 or mr != 0.0:
                    all_zero = False
                var ep = abs(Float64(jp[k]) - mp)
                var er = abs(Float64(jr[k]) - mr)
                if ep > worst_p:
                    worst_p = ep
                    worst_where = (
                        "pose " + String(p) + " site " + String(s)
                    )
                if er > worst_r:
                    worst_r = er
                    worst_where = (
                        "pose " + String(p) + " site " + String(s)
                    )
            if all_zero:
                n_zero_blocks += 1
            n_checked += 1

    print(
        "  [" + label + "] site-poses", n_checked,
        " worst |d(jacp)|", worst_p, " worst |d(jacr)|", worst_r,
        " at", worst_where, " (all-zero blocks:", n_zero_blocks, ")",
    )

    assert_true(
        n_checked == N_POSES * D.NSITE,
        label + ": not every site-pose was compared — the loop fell through",
    )
    if want_world_site:
        assert_true(
            n_zero_blocks >= N_POSES,
            label + ": no all-zero Jacobian block was seen even though a site"
            " is welded to the world — the early return never fired, so it is"
            " untested",
        )
    assert_true(
        worst_p <= JAC_TOL,
        label + ": translational site Jacobian differs from mj_jacSite",
    )
    assert_true(
        worst_r <= JAC_TOL,
        label + ": ROTATIONAL site Jacobian differs from mj_jacSite — the"
        " block the constraint-row code never exercised, so it has no prior"
        " coverage anywhere in the tree",
    )


def test_jac_site_quadruped_fetch() raises:
    """FREE (torso + ball) + HINGE DOFs, and the `target` site welded to the
    world. ⚠ The FETCH variant, not walk: walk is built with
    `walls_and_ball=False`, which deletes the ball (the second free joint) and
    the target site (the only world-welded one), so it covers neither."""
    print("--- quadruped fetch: jac_site vs mj_jacSite ---")
    comptime M = DMQuadrupedFetchModel
    comptime MD = ModelDims[M]
    var sf = M.make_spec_fields[DTYPE]()
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var mm = mujoco.MjModel.from_xml_path("noeira/envs/dm_control/assets/quadruped_fetch.xml")
    var dat = mujoco.MjData(mm)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)

    _sweep(
        "quadruped_fetch", mujoco, np, mm, dat, mf, d,
        want_free=True, want_slide=False, want_world_site=True,
    )


def test_jac_site_ball_in_cup() raises:
    """SLIDE DOFs — `cdof`'s angular half is identically zero here, so `jacr`
    must come back exactly zero and `jacp` a bare axis."""
    print("--- ball_in_cup: jac_site vs mj_jacSite ---")
    comptime M = DMBallInCupModel
    comptime MD_2 = ModelDims[M]
    var sf = M.make_spec_fields[DTYPE]()
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var mm = mujoco.MjModel.from_xml_path("noeira/envs/dm_control/assets/ball_in_cup.xml")
    var dat = mujoco.MjData(mm)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_2]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_2, 1]()
    M.reset_data[DTYPE](sf, d)

    _sweep(
        "ball_in_cup", mujoco, np, mm, dat, mf, d,
        want_free=False, want_slide=True, want_world_site=False,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
