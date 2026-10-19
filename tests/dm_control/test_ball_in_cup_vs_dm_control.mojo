"""dm_control `ball_in_cup-catch` parity: our env vs MuJoCo + the reference task.

Same four layers as the other domain tests (model / physics / observation /
reward), plus the one this domain exists to prove: a SPATIAL tendon and its
LIMIT.

WHAT THIS EXERCISES THAT NO EARLIER DOMAIN DID
----------------------------------------------
  - `<spatial>` tendons. Every previously ported model with a `<tendon>` uses
    the FIXED kind (`fish`, `humanoid`, and the `point_mass` tendons that were
    rewritten as joint motors). Nothing routed a tendon through sites, so
    `dynamics/tendon.mojo` had no gate at all before this file.
  - `mjCNSTR_LIMIT_TENDON`. No earlier model sets `limited` on a tendon.
  - `tendon_invweight0`. This is the tendon limit's `diagApprox`, i.e. the
    same slot whose mishandling was bug 20 (a 64x-too-soft contact that hid
    for weeks). It is diffed against MuJoCo directly below, per the habit that
    bug established: check every newly ported model's inverse weights.

THE POPULATION SPLIT
--------------------
`test_ball_in_cup_matches_mujoco` does NOT report one aggregate error. It
splits every physics substep by WHICH CONSTRAINT ROWS MUJOCO HAS LIVE:

    limit only     — string taut, ball in mid-air
    contact only   — ball resting/bouncing on the cup, string slack
    limit+contact  — both at once, on shared dofs
    neither        — free flight

That split is the measurement that made the finger bug legible (commit
04a7c508): aggregated, a coupling bug reads as "somewhat worse than we'd
like"; split, it reads as "the uncoupled populations are exact and the
coupled one is 39x off", which is a different claim entirely. ball_in_cup is
the natural gate for the tendon half of that work, because a caught ball rests
on the cup capsules WHILE the string is taut.

The three rollouts below were chosen (by sweeping control amplitudes and
phases against MuJoCo) to populate all four buckets; the counts are printed so
a future change that empties the coupled bucket is visible rather than silent.

Run with:
    pixi run mojo run -I . tests/dm_control/test_ball_in_cup_vs_dm_control.mojo
"""

from std.math import abs, sin, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext
from std.collections import Array

from noeira.envs.dm_control.ball_in_cup import (
    DMBallInCupCatch,
    DMBallInCupModel,
    BALL_BODY_IDX,
    CUP_SITE_IDX,
    TARGET_SITE_IDX,
    BALL_SITE_IDX,
    BALL_GEOM_IDX,
    TARGET_HALF_X,
    TARGET_HALF_Z,
    BALL_RADIUS,
)
from noeira.physics3d.fields import Data, Model, DynamicsScratch, Dims, DimsLike, AsStatic
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.dynamics.subtree_com import compute_subtree_com
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.tendon import spatial_tendon_length_jac
from noeira.physics3d.fields.scratch import Scratch
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.gpu.constants import (
    MODEL_META_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_SITE_SIZE,
    MODEL_TENDON_SIZE,
    BODY_IDX_MASS,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_DAMPING,
    GEOM_IDX_BODY,
    SITE_IDX_BODY,
    TENDON_IDX_KIND,
    TENDON_IDX_LIMITED,
    TENDON_IDX_RANGE_MIN,
    TENDON_IDX_RANGE_MAX,
    TENDON_IDX_INVWEIGHT0,
    TENDON_KIND_SPATIAL,
)


comptime DTYPE = DType.float64
comptime Env = DMBallInCupCatch[DTYPE]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/ball_in_cup.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = DMBallInCupModel.NQ  # 4
comptime NV: Int = DMBallInCupModel.NV  # 4
comptime NBODY: Int = DMBallInCupModel.NBODY  # 3
comptime NGEOM: Int = DMBallInCupModel.NGEOM  # 7
comptime NSITE: Int = DMBallInCupModel.NSITE  # 3
comptime NJOINT: Int = DMBallInCupModel.NJOINT  # 4
comptime NTEN: Int = DMBallInCupModel.MAX_TENDON  # 1
comptime NACT: Int = DMBallInCupModel.nact  # 2
comptime MAXC: Int = DMBallInCupModel.MAX_CONTACTS
comptime MD = ModelDims[DMBallInCupModel]
comptime FRAME_SKIP: Int = 10

comptime N_STEPS: Int = 150

# Gate. Set from the measured worst case per bucket (printed below), not
# guessed. The coupled bucket carries its own budget because it is the one the
# tendon-limit row exists to get right.
# Every bucket measures at <= 8.9e-16 (machine precision for these
# magnitudes), so 1e-13 is a real gate with headroom, not a fitted number.
comptime TOL_UNCOUPLED: Float64 = 1e-13
comptime TOL_COUPLED: Float64 = 1e-13
comptime TOL_OBS: Float64 = 1e-13
comptime TOL_REWARD: Float64 = 0.0  # sparse; must agree exactly


def _build_model() raises -> Model[DTYPE, MD]:
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    DMBallInCupModel.init_fields[DTYPE](ctx, mf)
    return mf^


def test_ball_in_cup_model_matches_mujoco() raises:
    """Dims, masses, springs, element ordering, and the tendon constants."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(String(REF_XML))

    assert_true(Int(py=m.nbody) == NBODY, "nbody mismatch")
    assert_true(Int(py=m.njnt) == NJOINT, "njnt mismatch")
    assert_true(Int(py=m.nq) == NQ, "nq mismatch")
    assert_true(Int(py=m.nv) == NV, "nv mismatch")
    assert_true(Int(py=m.ngeom) == NGEOM, "ngeom mismatch")
    assert_true(Int(py=m.nsite) == NSITE, "nsite mismatch")
    assert_true(Int(py=m.nu) == NACT, "nu mismatch")
    assert_true(Int(py=m.ntendon) == NTEN, "ntendon mismatch")

    var mf = _build_model()

    # --- ordering, pinned by NAME on the MuJoCo side ----------------------
    # Our parser numbers by XML text order; MuJoCo sorts geoms/sites by body.
    # They agree for this model (every world geom precedes the first body),
    # which is exactly the assumption point_mass proved can fail — so pin it.
    var ref_ball_geom = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    )
    var ref_target_site = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
    )
    var ref_cup_site = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cup")
    )
    var ref_ball_site = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ball")
    )
    var ref_ball_body = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    )
    assert_true(ref_ball_geom == BALL_GEOM_IDX, "MuJoCo's ball geom moved")
    assert_true(ref_target_site == TARGET_SITE_IDX, "target site moved")
    assert_true(ref_cup_site == CUP_SITE_IDX, "cup site moved")
    assert_true(ref_ball_site == BALL_SITE_IDX, "ball site moved")
    assert_true(ref_ball_body == BALL_BODY_IDX, "ball body moved")

    assert_true(
        Int(mf.geoms.data[BALL_GEOM_IDX * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
        == BALL_BODY_IDX,
        "our ball geom is not on the ball body",
    )
    assert_true(
        Int(mf.sites.data[TARGET_SITE_IDX * MODEL_SITE_SIZE + SITE_IDX_BODY])
        == 1,
        "our target site is not on the cup body",
    )
    assert_true(
        Int(mf.sites.data[BALL_SITE_IDX * MODEL_SITE_SIZE + SITE_IDX_BODY])
        == BALL_BODY_IDX,
        "our ball site is not on the ball body",
    )

    # --- masses -----------------------------------------------------------
    var worst_mass = 0.0
    for b in range(NBODY):
        var dm = abs(
            Float64(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
            - Float64(py=m.body_mass[b])
        )
        if dm > worst_mass:
            worst_mass = dm
    print("ball_in_cup model: max |d(mass)| =", worst_mass)
    assert_true(worst_mass <= 1e-15, "masses differ from MuJoCo")

    # --- joint springs + damping -----------------------------------------
    # The cup's `stiffness="20"` is what holds it up against the string; a
    # dropped spring would look like a physics bug much later.
    var worst_stiff = 0.0
    var worst_damp = 0.0
    for j in range(NJOINT):
        var ds = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_STIFFNESS])
            - Float64(py=m.jnt_stiffness[j])
        )
        if ds > worst_stiff:
            worst_stiff = ds
        var dd = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_DAMPING])
            - Float64(py=m.dof_damping[j])
        )
        if dd > worst_damp:
            worst_damp = dd
    print("  max |d(jnt_stiffness)| =", worst_stiff)
    print("  max |d(dof_damping)|   =", worst_damp)
    assert_true(worst_stiff <= 1e-15, "joint stiffness differs")
    assert_true(worst_damp <= 1e-15, "dof damping differs")

    # --- the target/ball sizes the sparse reward differences --------------
    assert_true(
        abs(Float64(py=m.site_size[ref_target_site][0]) - TARGET_HALF_X)
        <= 1e-15,
        "TARGET_HALF_X out of sync with ball_in_cup.xml",
    )
    assert_true(
        abs(Float64(py=m.site_size[ref_target_site][2]) - TARGET_HALF_Z)
        <= 1e-15,
        "TARGET_HALF_Z out of sync with ball_in_cup.xml",
    )
    assert_true(
        abs(Float64(py=m.geom_size[ref_ball_geom][0]) - BALL_RADIUS) <= 1e-15,
        "BALL_RADIUS out of sync with ball_in_cup.xml",
    )

    # --- tendon record ----------------------------------------------------
    assert_true(
        Int(mf.tendons.data[TENDON_IDX_KIND]) == TENDON_KIND_SPATIAL,
        "the string did not parse as a SPATIAL tendon",
    )
    assert_true(
        Int(mf.tendons.data[TENDON_IDX_LIMITED]) == 1,
        "the string's `limited` was dropped",
    )
    assert_true(
        abs(
            Float64(mf.tendons.data[TENDON_IDX_RANGE_MIN])
            - Float64(py=m.tendon_range[0][0])
        ) <= 1e-15
        and abs(
            Float64(mf.tendons.data[TENDON_IDX_RANGE_MAX])
            - Float64(py=m.tendon_range[0][1])
        ) <= 1e-15,
        "tendon_range differs from MuJoCo",
    )

    # tendon_invweight0 — the limit row's diagApprox. See the module docstring
    # on why this specific number gets its own assertion.
    var iw_ours = Float64(mf.tendons.data[TENDON_IDX_INVWEIGHT0])
    var iw_ref = Float64(py=m.tendon_invweight0[0])
    print("  tendon_invweight0 ours =", iw_ours, " MuJoCo =", iw_ref)
    assert_true(
        abs(iw_ours - iw_ref) <= 1e-12,
        "tendon_invweight0 differs from MuJoCo — the limit row's whole force"
        " scale is wrong (this is the bug-20 failure mode)",
    )

    # --- ten_length / ten_J at qpos0 --------------------------------------
    var mj_d = mujoco.MjData(m)
    mujoco.mj_forward(m, mj_d)

    var d = Data[DTYPE, MD, 1]()
    var sc = DynamicsScratch[DTYPE, MD, 1]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](0)
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu", DTYPE, BATCH=1](d, mf, None)
    compute_subtree_com["cpu", DTYPE, BATCH=1](d, mf, None)
    compute_cdof["cpu", DTYPE, BATCH=1](d, mf, sc, None)

    # site_xpos, which the tendon is routed through.
    var worst_site = 0.0
    for s in range(NSITE):
        for k in range(3):
            var ds = abs(
                Float64(d.site_xpos.data[s * 3 + k])
                - Float64(py=mj_d.site_xpos[s][k])
            )
            if ds > worst_site:
                worst_site = ds
    print("  max |d(site_xpos)| =", worst_site)
    assert_true(worst_site <= 1e-14, "site_xpos differs from MuJoCo")

    # ⚠ `Scratch`, not `Array` — this call site had been stale since
    # `spatial_tendon_length_jac` took `Scratch` (the test tree is only
    # partially swept, `project_rc2_test_tree_compile_sweep`). Unrelated to
    # the wrap work; fixed here because it is one line and the file could not
    # otherwise build.
    var tJ = Scratch[Scalar[DTYPE], NV](NV, fill=Scalar[DTYPE](0))
    var L = spatial_tendon_length_jac[
        DTYPE, NV, 1
    ](
        0, 0, AsStatic[MD](),
        mf.tendons.tt["cpu", NTEN, MODEL_TENDON_SIZE](),
        mf.sites.tt["cpu", NSITE, MODEL_SITE_SIZE](),
        mf.geoms.tt["cpu", NGEOM, MODEL_GEOM_SIZE](),
        mf.bodies.tt["cpu", NBODY, MODEL_BODY_SIZE](),
        mf.joints.tt["cpu", NJOINT, MODEL_JOINT_SIZE](),
        mf.meta.tt["cpu", MODEL_META_SIZE](),
        d.subtree_com.tt["cpu", 1, NBODY * 3](),
        sc.cdof.tt["cpu", 1, NV * 6](),
        d.xpos.tt["cpu", 1, NBODY * 3](),
        d.xquat.tt["cpu", 1, NBODY * 4](),
        tJ,
    )
    var dL = abs(Float64(L) - Float64(py=mj_d.ten_length[0]))
    print("  |d(ten_length)| =", dL)
    assert_true(dL <= 1e-15, "ten_length differs from MuJoCo")

    var worst_J = 0.0
    for i in range(NV):
        var dj = abs(Float64(tJ[i]) - Float64(py=mj_d.ten_J[i]))
        if dj > worst_J:
            worst_J = dj
    print("  max |d(ten_J)| =", worst_J)
    assert_true(worst_J <= 1e-15, "ten_J differs from MuJoCo")


def test_ball_in_cup_matches_mujoco() raises:
    """Physics lockstep, split by which constraint rows MuJoCo has live."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)

    # (ball_x, ball_z, amp_x, freq_x, phase_x, amp_z, freq_z, phase_z).
    # Chosen by sweeping against MuJoCo so that all four buckets fill; see
    # the module docstring.
    # All three start COLLISION-FREE, as `BallInCup.initialize_episode`
    # guarantees; a penetrating start is a state the task never produces and
    # would gate the engine on a regime it is never asked to handle.
    var runs = [
        [-0.0664, 0.2017, 0.6158, 0.2326, 0.3252, 0.8828, 0.1567, 1.7046],
        [0.0426, 0.2593, 0.8492, 0.1464, 0.8094, 0.7241, 0.0786, 0.318],
        [0.1278, 0.4977, 0.8058, 0.4487, 2.9527, 0.8703, 0.2112, 0.6637],
    ]

    var n_both = 0
    var n_lim = 0
    var n_con = 0
    var n_none = 0
    var worst_both = 0.0
    var worst_lim = 0.0
    var worst_con = 0.0
    var worst_none = 0.0
    var max_obs = 0.0
    var max_r = 0.0
    var n_reward_one = 0

    for r in runs:
        mujoco.mj_resetData(model, data)
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = r[0]
        data.qpos[3] = r[1]
        mujoco.mj_forward(model, data)

        var env = Env()
        _ = env.reset()

        for step in range(N_STEPS):
            # RESYNC: our engine starts every control step from MuJoCo's exact
            # state, so what is measured is ONE step of divergence rather than
            # accumulated drift. Without this the buckets are meaningless —
            # once two trajectories separate, the free-flight bucket looks as
            # bad as the contact bucket and attribution is impossible. (That
            # is not a hypothetical: the first version of this test reported
            # 2.86 on `neither`, a bucket with no constraints in it at all.)
            var qs = List[Float64]()
            var vs = List[Float64]()
            for i in range(NQ):
                qs.append(Float64(py=data.qpos[i]))
            for i in range(NV):
                vs.append(Float64(py=data.qvel[i]))
            env.set_state(qs, vs)

            var act = Env.ActionType()
            var cx = r[2] * sin(r[3] * Float64(step) + r[4])
            var cz = r[5] * sin(r[6] * Float64(step) + r[7])
            data.ctrl[0] = cx
            data.ctrl[1] = cz
            act.data[0] = cx
            act.data[1] = cz

            # Step MuJoCo one physics substep at a time so each substep can be
            # classified; our env advances FRAME_SKIP substeps in one call, so
            # the comparison itself still happens once per control step.
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
                var nefc = Int(py=data.nefc)
                var has_lim = False
                for e in range(nefc):
                    if Int(py=data.efc_type[e]) == 4:  # mjCNSTR_LIMIT_TENDON
                        has_lim = True
                        break
                var has_con = Int(py=data.ncon) > 0
                if has_lim and has_con:
                    n_both += 1
                elif has_lim:
                    n_lim += 1
                elif has_con:
                    n_con += 1
                else:
                    n_none += 1
            mujoco.mj_forward(model, data)

            # Which bucket this control step belongs to, for the error split:
            # the strictest bucket touched during its substeps.
            var out = env.step(act)

            var worst_here = 0.0
            for i in range(NQ):
                var dq = abs(
                    Float64(py=data.qpos[i]) - Float64(env.d.qpos.data[i])
                )
                if dq > worst_here:
                    worst_here = dq
            for i in range(NV):
                var dv = abs(
                    Float64(py=data.qvel[i]) - Float64(env.d.qvel.data[i])
                )
                if dv > worst_here:
                    worst_here = dv

            # Attribute this step's error to the regime MuJoCo was in at the
            # END of the step (the substep counters above give the population
            # sizes; this gives the per-regime worst case).
            var nefc_e = Int(py=data.nefc)
            var lim_e = False
            for e in range(nefc_e):
                if Int(py=data.efc_type[e]) == 4:
                    lim_e = True
                    break
            var con_e = Int(py=data.ncon) > 0
            if lim_e and con_e:
                if worst_here > worst_both:
                    worst_both = worst_here
            elif lim_e:
                if worst_here > worst_lim:
                    worst_lim = worst_here
            elif con_e:
                if worst_here > worst_con:
                    worst_con = worst_here
            else:
                if worst_here > worst_none:
                    worst_none = worst_here

            # observation: qpos then qvel
            var obs = out[0]
            for i in range(NQ):
                var d_o = abs(Float64(py=data.qpos[i]) - Float64(obs.data[i]))
                if d_o > max_obs:
                    max_obs = d_o
            for i in range(NV):
                var d_o = abs(
                    Float64(py=data.qvel[i]) - Float64(obs.data[NQ + i])
                )
                if d_o > max_obs:
                    max_obs = d_o

            # reward: `Physics.in_target()`, sparse
            var tx = Float64(py=data.site_xpos[TARGET_SITE_IDX][0])
            var tz = Float64(py=data.site_xpos[TARGET_SITE_IDX][2])
            var bx = Float64(py=data.xpos[BALL_BODY_IDX][0])
            var bz = Float64(py=data.xpos[BALL_BODY_IDX][2])
            var ref_r = 0.0
            if (
                abs(tx - bx) < TARGET_HALF_X - BALL_RADIUS
                and abs(tz - bz) < TARGET_HALF_Z - BALL_RADIUS
            ):
                ref_r = 1.0
                n_reward_one += 1
            var dr = abs(ref_r - Float64(out[1]))
            if dr > max_r:
                max_r = dr

    print("ball_in_cup substep populations (MuJoCo's live rows):")
    print("  limit only    :", n_lim, " worst |d(state)| =", worst_lim)
    print("  contact only  :", n_con, " worst |d(state)| =", worst_con)
    print("  limit+contact :", n_both, " worst |d(state)| =", worst_both)
    print("  neither       :", n_none, " worst |d(state)| =", worst_none)
    print("  max |d(obs)|    =", max_obs)
    print("  max |d(reward)| =", max_r, " (steps in target:", n_reward_one, ")")

    # The coupled bucket must be non-empty or this test proves nothing about
    # the tendon-limit row's interaction with contacts — the exact failure
    # mode described in feedback_sweep_model_must_express_defect.
    assert_true(
        n_both > 0,
        "no substep had a tendon limit and a contact live at once — this"
        " rollout no longer exercises the coupling it was chosen for",
    )
    assert_true(n_lim > 0, "no substep loaded the tendon limit alone")
    assert_true(n_con > 0, "no substep had contacts without the limit")

    assert_true(
        worst_lim <= TOL_UNCOUPLED and worst_none <= TOL_UNCOUPLED,
        "uncoupled substeps diverge from MuJoCo",
    )
    assert_true(
        worst_con <= TOL_UNCOUPLED, "contact-only substeps diverge from MuJoCo"
    )
    assert_true(
        worst_both <= TOL_COUPLED,
        "coupled (tendon limit + contact) substeps diverge from MuJoCo",
    )
    assert_true(max_obs <= TOL_OBS, "observation differs from MuJoCo")
    assert_true(
        max_r <= TOL_REWARD, "sparse reward disagrees with Physics.in_target()"
    )
    # The positive branch is NOT asserted here: none of these rollouts
    # actually catches the ball, and contriving one that does would make the
    # physics gate hostage to a lucky trajectory. It gets its own test below,
    # driven from explicit qpos.


def test_ball_in_cup_reward_matches_mujoco() raises:
    """`Physics.in_target()` on BOTH branches, from explicit poses.

    The rollout test above never catches the ball, so without this the sparse
    reward would only ever be observed returning 0 — a reward function stuck
    at zero would pass it. Each pose below is checked against the reference
    formula evaluated on MuJoCo's own `site_xpos`/`xpos`, not against a
    hardcoded expectation.

    Poses are collision-free so that `mj_forward` alone settles nothing.
    """
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)

    # (cup_x, cup_z, ball_x, ball_z, expected in_target)
    #   target site sits at cup + (0, 0, -.05); the ball body at (0, 0, .2)
    #   plus its slides. The box half-extent after subtracting the ball radius
    #   is .025 in x and z.
    var poses = [
        [0.0, 0.0, 0.0, 0.35, 1.0],  # dead centre of the target
        [0.0, 0.0, 0.02, 0.34, 1.0],  # inside, off-centre in both axes
        [0.0, 0.0, 0.0, 0.20, 0.0],  # hanging well below the cup
        [0.0, 0.0, 0.03, 0.35, 0.0],  # x just outside (.03 > .025)
        [0.0, 0.0, 0.0, 0.32, 0.0],  # z just outside
        [0.1, 0.0, 0.1, 0.35, 1.0],  # cup translated: target moves with it
    ]

    var n_pos = 0
    var n_neg = 0
    for p in poses:
        mujoco.mj_resetData(model, data)
        data.qpos[0] = p[0]
        data.qpos[1] = p[1]
        data.qpos[2] = p[2]
        data.qpos[3] = p[3]
        mujoco.mj_forward(model, data)

        # The reference computation, on MuJoCo's own tables.
        var tx = Float64(py=data.site_xpos[TARGET_SITE_IDX][0])
        var tz = Float64(py=data.site_xpos[TARGET_SITE_IDX][2])
        var bx = Float64(py=data.xpos[BALL_BODY_IDX][0])
        var bz = Float64(py=data.xpos[BALL_BODY_IDX][2])
        var ref_r = 0.0
        if (
            abs(tx - bx) < TARGET_HALF_X - BALL_RADIUS
            and abs(tz - bz) < TARGET_HALF_Z - BALL_RADIUS
        ):
            ref_r = 1.0
        assert_true(
            ref_r == p[4],
            "the reference formula disagrees with this pose's intent — the"
            " pose table is wrong, not the engine",
        )

        var env = Env()
        _ = env.reset()
        var qs = List[Float64]()
        var vs = List[Float64]()
        for i in range(NQ):
            qs.append(p[i])
        for _ in range(NV):
            vs.append(0.0)
        env.set_state(qs, vs)

        # One zero-control step so the env evaluates its reward hook. The pose
        # is collision-free and nearly static, so the reward cannot flip.
        var act = Env.ActionType()
        for k in range(NACT):
            act.data[k] = 0.0
            data.ctrl[k] = 0.0
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        var out = env.step(act)

        var tx2 = Float64(py=data.site_xpos[TARGET_SITE_IDX][0])
        var tz2 = Float64(py=data.site_xpos[TARGET_SITE_IDX][2])
        var bx2 = Float64(py=data.xpos[BALL_BODY_IDX][0])
        var bz2 = Float64(py=data.xpos[BALL_BODY_IDX][2])
        var ref_r2 = 0.0
        if (
            abs(tx2 - bx2) < TARGET_HALF_X - BALL_RADIUS
            and abs(tz2 - bz2) < TARGET_HALF_Z - BALL_RADIUS
        ):
            ref_r2 = 1.0
            n_pos += 1
        else:
            n_neg += 1

        assert_true(
            abs(ref_r2 - Float64(out[1])) <= 1e-15,
            "reward disagrees with Physics.in_target() after a step",
        )

    print("ball_in_cup reward: ", n_pos, "in-target poses,", n_neg, "outside")
    assert_true(n_pos > 0, "no pose exercised the positive branch")
    assert_true(n_neg > 0, "no pose exercised the negative branch")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
