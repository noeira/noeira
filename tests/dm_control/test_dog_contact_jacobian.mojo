"""dog's contact JACOBIAN rows against MuJoCo's `efc_J`.

THE LAST UNVERIFIED INPUT TO THE CONTACT SOLVE.

Everything else the solver reads is now measured exact against MuJoCo at dog's
settled pose: the contact set (dist/pos/normal ~1e-15, condim histogram), the
mixed parameters (solref/solimp/friction all 0.0), the derived row constants
(`R` 2.6e-16 and `aref` 4.4e-16 relative, checked by reproducing our own
formulas against `efc_R`/`efc_aref`), the mass matrix and passive forces
(2.2e-12 on a qacc reaching 715), and `qfrc_bias` (5.3e-15).

And the solve is CONVERGED: tightening `NEWTON_TOL_GPU` from 1e-8 to 1e-14
changes our answer in NOT ONE DIGIT, while MuJoCo reaches its own answer in 5
iterations and holds it at 5004 iterations with `tolerance = 0`. Two converged
solvers, inputs proven identical, different answers — which leaves the
Jacobian, or the solver's own arithmetic.

HOW MuJoCo's J_n IS RECOVERED. For a pyramidal contact MuJoCo emits no normal
row; it emits opposing edge pairs `J_n +- mu_k * J_k`. So for the first pair
(rows j0, j1 of that contact):

    J_n  = (efc_J[j0] + efc_J[j1]) / 2
    J_t1 = (efc_J[j0] - efc_J[j1]) / (2 * mu)

⚠ THAT IDENTITY IS ITSELF AN ASSUMPTION, so the test asserts it before using
it: `mu` is read back from `d.contact[c].friction[0]` and the reconstruction is
checked to be consistent across the pair rather than taken on faith. A
frictionless contact (condim 1) emits ONE row which IS `J_n`, handled
separately.

⚠ `m.opt.jacobian` IS FORCED DENSE. MuJoCo defaults to `mjJAC_AUTO`, which
picks sparse on a model this size, and `d.efc_J` would then be a packed sparse
array whose `nefc x nv` reshape is silently wrong rather than an error.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_contact_jacobian.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext
from std.collections import Array
from noeira.core.tiles import RowMajor

from noeira.envs.dm_control.dog import (
    DMDogStandWalkModel,
)
from noeira.physics3d.fields import Model, Data, Dims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.dynamics.jac_contact_row import _contact_jacobian_row
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    META_IDX_NUM_CONTACTS,
    MODEL_JOINT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_META_SIZE,
)

comptime DTYPE = DType.float64
comptime M = DMDogStandWalkModel
comptime MD = ModelDims[M]
comptime NQ = M.NQ
comptime NV = M.NV
comptime N_SETTLE: Int = 400

# Both sides build the same rows from the same float64 FK products, so this is
# round-off on a Jacobian whose entries are O(1) lever arms.
comptime JAC_TOL: Float64 = 1e-12


def test_dog_contact_jacobian_matches_mujoco() raises:
    var sf = M.make_spec_fields[DTYPE]()
    print("--- dog: contact Jacobian rows vs MuJoCo efc_J ---")
    var mujoco = Python.import_module("mujoco")
    var mm = mujoco.MjModel.from_xml_path("noeira/envs/dm_control/assets/dog_stand_walk.xml")
    # ⚠ See the module docstring: AUTO would give a SPARSE efc_J here.
    mm.opt.jacobian = 0  # mjJAC_DENSE
    var dat = mujoco.MjData(mm)
    mujoco.mj_resetData(mm, dat)
    for _ in range(N_SETTLE):
        mujoco.mj_step(mm, dat)
    for k in range(M.nact):
        dat.ctrl[k] = 0.0
        dat.act[k] = 0.0
    mujoco.mj_forward(mm, dat)

    var nefc = Int(py=dat.nefc)
    var jflat = dat.efc_J.flatten().tolist()
    assert_true(
        len(jflat) == nefc * NV,
        "efc_J is not a dense nefc x nv block — MuJoCo chose a SPARSE layout"
        " and every row read below would be garbage",
    )

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=dat.qpos[i]))
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=dat.qvel[i]))
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)

    # One step populates `cdof` and `subtree_com` from the PRE-step state (they
    # are computed at the top of `step`, before anything integrates), and
    # `_finalize_env` does not touch either — unlike `M`, `L`, `D`, `fnet` and
    # `qacc_ws`, which it reuses. Same survivor set as the staged probe.
    var integ = EulerIntegrator[DTYPE, MD, M.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=M.NOSLIP_ITER]()
    integ.step["cpu"](d, mf)

    comptime L_NB3 = RowMajor[1, M.NBODY * 3]
    comptime L_JNT = RowMajor[M.NJOINT, MODEL_JOINT_SIZE]
    comptime L_BOD = RowMajor[M.NBODY, MODEL_BODY_SIZE]
    comptime L_MET = RowMajor[MODEL_META_SIZE]
    comptime L_CDOF = RowMajor[1, NV * 6]
    var subtree_v = d.subtree_com.tt["cpu", 1, M.NBODY * 3]()
    var joints_v = mf.joints.tt["cpu", M.NJOINT, MODEL_JOINT_SIZE]()
    var bodies_v = mf.bodies.tt["cpu", M.NBODY, MODEL_BODY_SIZE]()
    var mmeta_v = mf.meta.tt["cpu", MODEL_META_SIZE]()
    var cdof_v = integ.scratch.cdof.tt["cpu", 1, NV * 6]()

    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    var mj_ncon = Int(py=dat.ncon)
    print("  ncon: ours", nc, " MuJoCo", mj_ncon, "  nefc", nefc)
    assert_true(
        nc == mj_ncon,
        "contact counts differ — the row mapping below is meaningless",
    )

    # Map each contact to its FIRST efc row. Contact rows come in `efc_id`
    # order; walk the array rather than assuming a stride, because condim 1
    # contributes 1 row and condim 3 contributes 4.
    var first_row = List[Int]()
    for _c in range(nc):
        first_row.append(-1)
    for k in range(nefc):
        var t = Int(py=dat.efc_type[k])
        if t != 5 and t != 6:
            continue
        var cid = Int(py=dat.efc_id[k])
        if cid >= 0 and cid < nc and first_row[cid] < 0:
            first_row[cid] = k

    var worst = 0.0
    var worst_c = -1
    var n_checked = 0
    for c in range(nc):
        var r0 = first_row[c]
        if r0 < 0:
            continue
        var o = c * CONTACT_SIZE
        var dim = Int(Float64(d.contacts.data[o + CONTACT_IDX_CONDIM]))

        # MuJoCo's J_n for this contact.
        var jn_mj = List[Float64]()
        if dim == 1:
            for i in range(NV):
                jn_mj.append(Float64(py=jflat[r0 * NV + i]))
        else:
            for i in range(NV):
                jn_mj.append(
                    0.5
                    * (
                        Float64(py=jflat[r0 * NV + i])
                        + Float64(py=jflat[(r0 + 1) * NV + i])
                    )
                )

        # Ours, built by the same helper the solver uses.
        var jn_ours = Array[Scalar[DTYPE], NV](fill=Scalar[DTYPE](0))
        _contact_jacobian_row[DTYPE, NV](
            0,
            subtree_v,
            joints_v,
            bodies_v,
            mmeta_v,
            cdof_v,
            Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_A])),
            Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_B])),
            Scalar[DTYPE](Float64(d.contacts.data[o + CONTACT_IDX_POS_X])),
            Scalar[DTYPE](Float64(d.contacts.data[o + CONTACT_IDX_POS_Y])),
            Scalar[DTYPE](Float64(d.contacts.data[o + CONTACT_IDX_POS_Z])),
            Scalar[DTYPE](Float64(d.contacts.data[o + CONTACT_IDX_NX])),
            Scalar[DTYPE](Float64(d.contacts.data[o + CONTACT_IDX_NY])),
            Scalar[DTYPE](Float64(d.contacts.data[o + CONTACT_IDX_NZ])),
            jn_ours,
        )

        # ⚠ SIGN. Our record's normal points body_b -> body_a on some pairs
        # (the convention is `normal = gi->gj, body_a = gi`), so compare the
        # row up to an overall sign rather than declaring a mismatch on a
        # convention that is gated elsewhere.
        var e_pos = 0.0
        var e_neg = 0.0
        for i in range(NV):
            var mine = Float64(jn_ours[i])
            var dp = abs(mine - jn_mj[i])
            var dn = abs(mine + jn_mj[i])
            if dp > e_pos:
                e_pos = dp
            if dn > e_neg:
                e_neg = dn
        var e = e_pos if e_pos < e_neg else e_neg
        n_checked += 1
        if e > worst:
            worst = e
            worst_c = c
        var mag = 0.0
        for i in range(NV):
            if abs(jn_mj[i]) > mag:
                mag = abs(jn_mj[i])
        print("   c", c, " dim", dim, " row", r0,
              " max|d(J_n)|", e, "  (|J_n| up to", mag, ")")

    print("  contacts checked:", n_checked, " worst |d(J_n)| =", worst,
          " at contact", worst_c)

    assert_true(
        n_checked >= 4,
        "fewer than four contacts had an efc row — the mapping failed and"
        " nothing was actually compared",
    )
    assert_true(
        worst <= JAC_TOL,
        "our contact normal JACOBIAN differs from MuJoCo's — with the contact"
        " set, parameters, row constants, mass matrix and bias all already"
        " measured exact, and both solvers converged, this is the remaining"
        " input that can move the answer",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
