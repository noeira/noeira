"""With NO constraint rows the Newton solve must be an IDENTITY. PN1's gate.

WHY THIS TEST EXISTS
====================
`mj_fwdConstraint` returns before solving when there is nothing to solve
(`references/mujoco-3.10.0/src/engine/engine_forward.c:884`):

    // no constraints: copy unconstrained acc, clear forces, return
    if (!nefc) { mju_copy(d->qacc, d->qacc_smooth, nv); ... return; }

Our three Newton loops had no such guard. They ran the full iteration — build
`H = M`, cooperative Cholesky, line search — on a problem with no constraints
in it. P0 priced that at **32.3 of 46.9 ms per step** on the k=9 park scene:
69% of GPU time, and 78% of the entire cost of a parked slot
(`docs/BLOCK_DIAGONAL_MASS_MATRIX_IMPLEMENTATION.md` §0.0.1).

⚠⚠ THE GUARD IS ONLY SOUND IF SKIPPING CANNOT CHANGE THE ANSWER, and that is
what this file gates. The argument is that with no rows the warmstart block is
already skipped (each of the three sites guards it on its own row count), so
`qacc` still holds `qacc_smooth`, `Ma == f_smooth`, `qfrc == 0`, and the
gradient `Ma - f_smooth - qfrc` is identically zero — the search direction is
zero and the iterate cannot move. An argument is not a measurement, so:

    ARM 1  a contact-free scene: `qacc_constrained` is BIT-IDENTICAL across
           the solve, on all three solver paths.
    ARM 2  ⚠ THE POSITIVE CONTROL, and the test is worthless without it: the
           SAME scene, dropped onto the floor so contacts exist, must CHANGE
           `qacc_constrained`. Otherwise arm 1 passes on a harness that could
           not detect a change at all — "0 differences" over a comparison that
           never moves looks exactly like a pass.
    ARM 3  ⚠ NON-VACUITY ON THE PREMISE. Arm 1 asserts the contact count is
           really 0 and arm 2 asserts it is really > 0. Without these, a scene
           that quietly grew a contact would flip which branch is being tested
           while both arms still printed "ok".

⚠ ALL THREE SOLVER PATHS, because the guard is written three times — the
PYRAMIDAL and ELLIPTIC branches of `_newton_solve_env` and the cooperative
`_newton_blocked_fields_kernel`. A rule written three times drifts, and this
tree has already paid for that exact shape in this exact function: the blocked
kernel accepted `NOSLIP_ITER` and never read it, so the two branches of one
solver computed different physics from identical inputs
(`newton_solve.mojo`'s noslip comment).

Run: pixi run mojo run -I . tests/physics3d/test_newton_no_constraint_rows.mojo
"""

from max.gpu.host import DeviceContext

from noeira.physics3d.fields import (
    AsStatic, Data, Model, DynamicsScratch, ContactScratch, Dims, DimsLike,
)
from noeira.physics3d.types import ConeType
from noeira.physics3d.solver.je_budget import je_ws_size
from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.integrator.euler import (
    _armature_env, _fnet_passive_env, _qacc_writeback_env,
)
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics, compute_body_velocities,
)
from noeira.physics3d.dynamics.subtree_com import compute_subtree_com
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.mass_matrix import compute_mass_matrix
from noeira.physics3d.dynamics.ldl import (
    ldl_factor, ldl_solve, compute_m_inv,
)
from noeira.physics3d.dynamics.rne import compute_bias_forces_rne
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.solver.newton_solve import (
    solve_newton, solve_newton_blocked,
)
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, METADATA_SIZE, MODEL_JOINT_SIZE,
)
from noeira.core.tiles import RowMajor
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float64
comptime BATCH = 2

# A sphere on a slider pair over a plane. `pos` decides whether it touches:
# the same model gives arm 1 its contact-free scene and arm 2 its contact.
comptime BOX_XML = """
<mujoco model="drop">
  <option cone="pyramidal" timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="4 4 1"/>
    <body name="box" pos="0 0 1.0">
      <joint name="sx" type="slide" axis="1 0 0"/>
      <joint name="sz" type="slide" axis="0 0 1"/>
      <geom name="gb" type="sphere" size=".05"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime bp = parse_xml(BOX_XML)
comptime BoxModel = ModelDefFromXML[
    xml=BOX_XML,
    nbody=bp.NBODY, njoint=bp.NJOINT, nq=bp.NQ, nv=bp.NV,
    ngeom=bp.NGEOM, nact=bp.NACT, ntex=bp.NTEX, nmat=bp.NMAT,
    nlight=bp.NLIGHT, ncam=bp.NCAM, nsite=bp.NSITE,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=4,
    obs_qpos_skip=0,
    timestep=bp.TIMESTEP,
]
# ⚠ `ModelDims`, NOT the model def. `Data`/`DynamicsScratch`/`ContactScratch`
# take a `DimsLike`, and a `ModelDefLike` is not one — the same wrapper
# `test_friction_dof_rows_vs_mujoco` uses (`:307`).
comptime MD = ModelDims[BoxModel]
comptime JE_WS = je_ws_size[DTYPE, MD.NV, MD.NJOINT, MD.NTENDON, MD.NEQUALITY, MD.MAX_CONTACTS, 3]()
comptime NQ = BoxModel.NQ
comptime NV = BoxModel.NV
comptime NJOINT = BoxModel.NJOINT


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _prep(
    mut d: Data[DTYPE, MD, BATCH],
    mut mf: Model[DTYPE, MD],
    mut sc: DynamicsScratch[DTYPE, MD, BATCH],
) raises:
    """The dynamics chain an integrator runs before the solve, CPU leg.

    Leaves `sc.qacc_constrained` holding `qacc_smooth` — the unconstrained
    acceleration — which is exactly the quantity MuJoCo copies through when
    `nefc == 0`, and the one the arms below compare across the solve."""
    comptime L_JOINT = RowMajor[NJOINT, MODEL_JOINT_SIZE]
    comptime L_M = RowMajor[BATCH, NV * NV]
    comptime L_NV = RowMajor[BATCH, NV]
    comptime L_QPOS = RowMajor[BATCH, NQ]

    forward_kinematics["cpu", DTYPE, BATCH=BATCH](d, mf, None)
    compute_body_velocities["cpu", DTYPE, BATCH=BATCH](d, mf, None)
    compute_subtree_com["cpu", DTYPE, BATCH=BATCH](d, mf, None)
    compute_cdof["cpu", DTYPE, BATCH=BATCH](d, mf, sc, None)
    compute_mass_matrix["cpu", DTYPE, BATCH=BATCH](d, mf, sc, None)

    var joints_v = mf.joints.tt["cpu", NJOINT, MODEL_JOINT_SIZE]()
    var M_v = sc.M.tt["cpu", BATCH, NV * NV]()
    for e in range(BATCH):
        _armature_env[DTYPE](e, AsStatic[MD](), joints_v, M_v)
    ldl_factor["cpu", DTYPE, BATCH=BATCH](mf, sc, None)
    compute_m_inv["cpu", DTYPE, BATCH=BATCH](mf, sc, None)
    compute_bias_forces_rne["cpu", DTYPE, BATCH=BATCH](d, mf, sc, None)

    var qpos_v = d.qpos.tt["cpu", BATCH, NQ]()
    var qvel_v = d.qvel.tt["cpu", BATCH, NV]()
    var qfrc_v = d.qfrc.tt["cpu", BATCH, NV]()
    var bias_v = sc.bias.tt["cpu", BATCH, NV]()
    var fnet_v = sc.fnet.tt["cpu", BATCH, NV]()
    for e in range(BATCH):
        _fnet_passive_env[DTYPE](
            e, AsStatic[MD](), qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v
        )
    ldl_solve["cpu", DTYPE, BATCH=BATCH](mf, sc, None)
    var qacc_ws_v = sc.qacc_ws.tt["cpu", BATCH, NV]()
    var qacc_v = d.qacc.tt["cpu", BATCH, NV]()
    var qacc_c_v = sc.qacc_constrained.tt["cpu", BATCH, NV]()
    for e in range(BATCH):
        _qacc_writeback_env[DTYPE](
            e, AsStatic[MD](), qacc_ws_v, qacc_v, qacc_c_v
        )
    detect_contacts["cpu", DTYPE, BATCH=BATCH](d, mf, None)


comptime BODY_Z = 1.0     # <body name="box" pos="0 0 1.0">
comptime BALL_R = 0.05    # <geom name="gb" type="sphere" size=".05">


def _seed(mut d: Data[DTYPE, MD, BATCH], centre_z: Float64):
    """Put the sphere's CENTRE at `centre_z` in world coordinates.

    ⚠ THE SLIDE JOINT IS AN OFFSET FROM THE BODY'S `pos`, NOT AN ABSOLUTE
    HEIGHT, and the first version of this file wrote the absolute value
    straight into `qpos`. `0.04` then meant 1.04 — a metre in the air — so the
    grounded control had no contacts and asserted that a contact-free solve
    fails to move `qacc`, which is the thing arm 1 already proves. It failed
    loudly only because the control checks its own premise."""
    for e in range(BATCH):
        d.qpos.data[e * NQ + 0] = Scalar[DTYPE](0.1 * Float64(e))
        d.qpos.data[e * NQ + 1] = Scalar[DTYPE](centre_z - BODY_Z)
        d.qvel.data[e * NV + 0] = Scalar[DTYPE](0.3)
        d.qvel.data[e * NV + 1] = Scalar[DTYPE](0.0)


def _ncon(d: Data[DTYPE, MD, BATCH]) -> Int:
    var n = 0
    for e in range(BATCH):
        n += Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    return n


def _snapshot(sc: DynamicsScratch[DTYPE, MD, BATCH]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(BATCH * NV):
        out.append(Float64(sc.qacc_constrained.data[i]))
    return out^


def _bit_diffs(a: List[Float64], sc: DynamicsScratch[DTYPE, MD, BATCH]) -> Int:
    """Count of entries whose BITS changed. Not a tolerance: the claim is that
    the solve is an identity, and `!=` on the value is the exact test."""
    var n = 0
    for i in range(BATCH * NV):
        if a[i] != Float64(sc.qacc_constrained.data[i]):
            n += 1
    return n


def main() raises:
    var t = Tally()
    print("=== Newton with no constraint rows is an identity (PN1) ===")
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    BoxModel.init_fields[DTYPE](ctx, mf)

    # ── ARM 1: contact-free. The solve must not move qacc, on all 3 paths. ──
    print("--- arm 1: airborne (no contacts) ---")
    var compared = 0
    for path in range(3):
        var d = Data[DTYPE, MD, BATCH]()
        var sc = DynamicsScratch[DTYPE, MD, BATCH]()
        var cs = ContactScratch[DTYPE, MD, BATCH, JE_WS]()
        _seed(d, 1.0)                    # a metre up: nothing touches
        _prep(d, mf, sc)
        var nc = _ncon(d)
        # ⚠ THE PREMISE, CHECKED. Arm 1 tests the guarded branch only if the
        # scene really has no rows; a scene that grew a contact would test the
        # other branch while still printing "ok".
        t.truth(nc == 0, String("path ", path, ": airborne contacts = ", nc,
                                " (must be 0 or this arm is vacuous)"))
        var before = _snapshot(sc)
        if path == 0:
            solve_newton["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL,
                         BATCH=BATCH](d, mf, sc, cs, None)
        elif path == 1:
            solve_newton["cpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC,
                         BATCH=BATCH](d, mf, sc, cs, None)
        else:
            solve_newton_blocked["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL,
                                 BATCH=BATCH, JE_WS=JE_WS](d, mf, sc, cs, None)
        var diffs = _bit_diffs(before, sc)
        compared += BATCH * NV
        t.truth(diffs == 0,
                String("path ", path, ": qacc_constrained unchanged over ",
                       BATCH * NV, " entries (", diffs, " moved)"))

    # ── ARM 2: the positive control. With contacts the solve MUST move it. ──
    print("--- arm 2: resting on the floor (contacts) — the control ---")
    var moved_any = 0
    for path in range(3):
        var d = Data[DTYPE, MD, BATCH]()
        var sc = DynamicsScratch[DTYPE, MD, BATCH]()
        var cs = ContactScratch[DTYPE, MD, BATCH, JE_WS]()
        # Centre 0.03 against a radius of 0.05: 0.02 of penetration.
        _seed(d, BALL_R - 0.02)
        _prep(d, mf, sc)
        var nc = _ncon(d)
        t.truth(nc > 0, String("path ", path, ": grounded contacts = ", nc,
                               " (must be > 0 or the control is vacuous)"))
        var before = _snapshot(sc)
        if path == 0:
            solve_newton["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL,
                         BATCH=BATCH](d, mf, sc, cs, None)
        elif path == 1:
            solve_newton["cpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC,
                         BATCH=BATCH](d, mf, sc, cs, None)
        else:
            solve_newton_blocked["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL,
                                 BATCH=BATCH, JE_WS=JE_WS](d, mf, sc, cs, None)
        var diffs = _bit_diffs(before, sc)
        moved_any += diffs
        t.truth(diffs > 0,
                String("path ", path, ": qacc_constrained MOVED (", diffs,
                       " of ", BATCH * NV, ") — the harness can see a change"))

    print("--- the comparison was not empty ---")
    t.truth(compared == 3 * BATCH * NV,
            String("entries compared in arm 1: ", compared))
    t.truth(moved_any > 0,
            String("entries the control moved: ", moved_any))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_newton_no_constraint_rows: " + String(t.fails) + " failed"
        )
