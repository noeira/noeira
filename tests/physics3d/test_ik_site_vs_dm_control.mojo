"""`qpos_from_site_pose` against dm_control's OWN site IK, on the real Jaco.

Step 3 of the dm_control Phase 7 reset path, and the first gate that runs the
whole chain — `jac_site` + `quat2vel` + `integrate_pos` + the damped
least-squares loop — end to end on the model the manipulation tasks actually
use (`reach_site_features`, nq/nv 9, 17 bodies, 12 sites).

⚠ THE REFERENCE IS dm_control's REAL `inverse_kinematics.qpos_from_site_pose`,
reached through `manipulation_ref.ik_reference`, NOT a transcription of it. A
transcription on both sides would agree with itself and prove nothing.

⚠ THE ARGUMENTS COME FROM `set_site_to_xpos`, NOT FROM THE IK FUNCTION'S OWN
DEFAULTS — `rot_weight=2` (its default is 1) and the joint set restricted to
the six ARM joints, with the hand's three finger joints held. Both sides are
given the same, and the reference helper hardcodes the same call.

WHAT IS COMPARED, AND WHY NOT qpos

The contract of this routine is "put the site at the target pose", not "return
these joint angles". Both sides run the same iteration but with different
inner linear algebra (our Cholesky vs LAPACK's `lstsq`), and a Newton loop
amplifies roundoff, so demanding qpos parity would be demanding bit-identical
LAPACK. What must agree is:

  * the `success` flag — ⚠ NOT implied by the returned qpos looking sane. The
    progress guard breaks out with `success=False` while leaving qpos at the
    last accepted step, so a gate that only looked at qpos would pass on a run
    that never converged.
  * the ACHIEVED SITE POSITION, when converged — that is the contract, and it
    is checked against the TARGET, so it cannot be satisfied by agreeing with
    a reference that also failed.

`|d qpos|` is printed for information; a gross divergence would show up there
even though it is not the gate.

⚠ SITE INDICES ARE MAPPED, NOT ASSUMED. Our parser numbers sites in XML text
order; MuJoCo sorts them by body. The two agree for some models and not
others, and a silent mismatch here would compare the wrong site. The mapping
is established by matching site POSITIONS after forward kinematics and
asserted to be a bijection — which incidentally is the first gate on Jaco's
forward kinematics at all.

Run with:
    pixi run mojo run -I . tests/physics3d/test_ik_site_vs_dm_control.mojo
"""

from std.math import abs, sqrt
from std.python import Python
from std.testing import assert_true, TestSuite
from std.collections import Array
from max.gpu.host import DeviceContext

from noeira.physics3d.fields import Model, Data, Dims, DimsLike
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.fields_build import build_model_fields_from_flat
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.dynamics.ik_site import (
    qpos_from_site_pose,
    set_site_to_xpos,
)
from noeira.envs.dm_control.manipulation_reset import (
    set_grasp,
    sample_bbox_uniform,
)

comptime DTYPE = DType.float64

# Jaco's `reach_site_features`, measured on the runtime (see the mesh-inertia
# gate, which pins the same numbers).
comptime NBODY = 17
comptime NQ = 9
comptime NV = 9
comptime NJOINT = 9
comptime NGEOM = 21
comptime NSITE = 12
comptime NEXCLUDE = 4
comptime NMESH_VERTS = 60000
comptime MAXC = 64
comptime MD = Dims[
    nq=NQ,
    nv=NV,
    nbody=NBODY,
    njoint=NJOINT,
    ngeom=NGEOM,
    nsite=NSITE,
    max_contacts=MAXC,
    nequality=0,
    ntendon=0,
    nexclude=NEXCLUDE,
    nmesh_verts=NMESH_VERTS,
    npair=0,
]

# The six ARM DOFs. Jaco's joints are all hinges in model order: joint_1..6
# then the hand's finger_1..3, so the arm is dofs 0..5. Asserted below against
# the reference's own `arm_joint_names`, not trusted.
comptime NDOF = 6

comptime N_TRIALS: Int = 12

# The contract: a converged solve puts the site ON the target.
comptime SITE_TOL: Float64 = 1e-9


def _read(path: String) raises -> String:
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "r")
    var txt = String(f.read())
    _ = f.close()
    return txt


def test_ik_site_matches_dm_control() raises:
    print("=== Jaco reach_site_features: site IK vs dm_control ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var tempfile = Python.import_module("tempfile")
    var os = Python.import_module("os")
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var refmod = Python.import_module("manipulation_ref")

    var tmp = String(tempfile.mkdtemp(prefix="jaco_ik_"))
    var xml_path = String(refmod.bake("reach_site_features", tmp))
    var cwd = String(os.getcwd())
    _ = os.chdir(tmp)
    var mm = mujoco.MjModel.from_xml_path(xml_path)
    var dat = mujoco.MjData(mm)
    var fmd = parse_xml_full(_read(xml_path))

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    build_model_fields_from_flat[DTYPE](fmd, mf)
    _ = os.chdir(cwd)

    var d = Data[DTYPE, MD, 1]()

    # ── the arm DOF set, from the reference rather than from belief ──────
    var arm_names = refmod.arm_joint_names()
    assert_true(
        Int(py=Python.evaluate("len")(arm_names)) == NDOF,
        "the reference's arm joint set is not "
        + String(NDOF)
        + " joints — the DOF restriction this port assumes is wrong",
    )
    var dof_idx = Array[Int, NDOF](fill=0)
    for a in range(NDOF):
        var jid = Int(
            py=mujoco.mj_name2id(
                mm, mujoco.mjtObj.mjOBJ_JOINT, arm_names[a]
            )
        )
        assert_true(jid >= 0, "arm joint not found in the model")
        dof_idx[a] = Int(py=mm.jnt_dofadr[jid])
    print("  arm dofs:", dof_idx[0], dof_idx[1], dof_idx[2], dof_idx[3],
          dof_idx[4], dof_idx[5])

    # ── site index mapping, ours <-> MuJoCo, by position after FK ────────
    var q_probe = Array[Float64, NQ](fill=0.0)
    for i in range(NQ):
        q_probe[i] = 0.11 * Float64(i + 1) - 0.4
    for i in range(NQ):
        dat.qpos[i] = q_probe[i]
        d.qpos.data[i] = Scalar[DTYPE](q_probe[i])
    mujoco.mj_forward(mm, dat)
    forward_kinematics["cpu"](d, mf)

    var mj_tcp = Int(
        py=mujoco.mj_name2id(
            mm, mujoco.mjtObj.mjOBJ_SITE, refmod.TCP_SITE
        )
    )
    assert_true(mj_tcp >= 0, "TCP site not found in the reference model")

    var our_tcp = -1
    var n_matched = 0
    var worst_site = 0.0
    for s_mj in range(NSITE):
        var best = -1
        var best_e = 1e30
        for s_ours in range(NSITE):
            var e = 0.0
            for k in range(3):
                var dd = abs(
                    Float64(d.site_xpos.data[s_ours * 3 + k])
                    - Float64(py=dat.site_xpos[s_mj][k])
                )
                if dd > e:
                    e = dd
            if e < best_e:
                best_e = e
                best = s_ours
        if best_e < 1e-9:
            n_matched += 1
            if best_e > worst_site:
                worst_site = best_e
            if s_mj == mj_tcp:
                our_tcp = best
    print("  sites matched by position:", n_matched, "/", NSITE,
          " worst |d(site_xpos)|:", worst_site)

    if n_matched != NSITE:
        # Separate "our body FK is wrong" from "our site records are wrong":
        # bodies first, then the site table with its parsed body/local pos.
        var worst_body = 0.0
        for b in range(NBODY):
            for k in range(3):
                var e = abs(
                    Float64(d.xpos.data[b * 3 + k])
                    - Float64(py=dat.xpos[b][k])
                )
                if e > worst_body:
                    worst_body = e
        print("  DIAG worst |d(xpos)| over all", NBODY, "bodies:", worst_body)
        print("  DIAG ours: idx body   local pos            world pos")
        for s in range(len(fmd.sites)):
            var sd = fmd.sites[s]
            print(
                "   ", s, " b", sd.body_id,
                " lp", sd.pos_x, sd.pos_y, sd.pos_z,
                " wp", Float64(d.site_xpos.data[s * 3 + 0]),
                Float64(d.site_xpos.data[s * 3 + 1]),
                Float64(d.site_xpos.data[s * 3 + 2]),
            )
        print("  DIAG mujoco: idx body  world pos")
        for s in range(NSITE):
            print(
                "   ", s, " b", Int(py=mm.site_bodyid[s]),
                " wp", Float64(py=dat.site_xpos[s][0]),
                Float64(py=dat.site_xpos[s][1]),
                Float64(py=dat.site_xpos[s][2]),
            )
    assert_true(
        n_matched == NSITE,
        "not every MuJoCo site had a matching one of ours after FK — either"
        " Jaco's forward kinematics disagrees or the site sets differ, and"
        " either way the IK comparison below would be meaningless",
    )
    assert_true(our_tcp >= 0, "could not locate our index for the TCP site")
    print("  TCP site: MuJoCo", mj_tcp, " ours", our_tcp)

    # dm_control's DOWN_QUATERNION is (w, x, y, z); ours is (x, y, z, w).
    var down = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    down[0] = Scalar[DTYPE](0.70710678118)
    down[1] = Scalar[DTYPE](0.70710678118)
    down[2] = Scalar[DTYPE](0.0)
    down[3] = Scalar[DTYPE](0.0)

    # ── sweep ────────────────────────────────────────────────────────────
    var rng = np.random.default_rng(11)
    var n_agree = 0
    var n_both_ok = 0
    var worst_site_err = 0.0
    var worst_dq = 0.0
    var max_dsteps = 0

    for t in range(N_TRIALS):
        var q0 = np.zeros(NQ)
        for i in range(NDOF):
            q0[i] = Float64(py=rng.uniform(-2.0, 2.0))
        var tgt = np.zeros(3)
        tgt[0] = Float64(py=rng.uniform(-0.10, 0.10))
        tgt[1] = Float64(py=rng.uniform(-0.10, 0.10))
        tgt[2] = Float64(py=rng.uniform(0.28, 0.42))

        var r = refmod.ik_reference("reach_site_features", q0, tgt)
        var ref_ok = Bool(py=r[3])
        var ref_steps = Int(py=r[2])

        for i in range(NQ):
            d.qpos.data[i] = Scalar[DTYPE](Float64(py=q0[i]))
        var tp = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        for k in range(3):
            tp[k] = Scalar[DTYPE](Float64(py=tgt[k]))

        var res = qpos_from_site_pose[DTYPE, NDOF](d, mf, our_tcp, tp, down, dof_idx)

        assert_true(
            not res.rank_deficient,
            "the SPD solve hit a non-positive pivot — the rank-deficient case"
            " this port cannot reproduce (see ik_site.mojo); it has never been"
            " observed on this model",
        )

        if res.success == ref_ok:
            n_agree += 1
        if Int(res.steps) - ref_steps > max_dsteps:
            max_dsteps = Int(res.steps) - ref_steps
        if ref_steps - Int(res.steps) > max_dsteps:
            max_dsteps = ref_steps - Int(res.steps)

        var dq = 0.0
        for i in range(NQ):
            var e = abs(Float64(d.qpos.data[i]) - Float64(py=r[0][i]))
            if e > dq:
                dq = e
        if dq > worst_dq:
            worst_dq = dq

        if res.success and ref_ok:
            n_both_ok += 1
            # ⚠ Against the TARGET, not against the reference's answer — so a
            # run where BOTH sides failed the same way cannot pass this.
            forward_kinematics["cpu"](d, mf)
            var se = 0.0
            for k in range(3):
                var e = abs(
                    Float64(d.site_xpos.data[our_tcp * 3 + k])
                    - Float64(py=tgt[k])
                )
                if e > se:
                    se = e
            if se > worst_site_err:
                worst_site_err = se

        print("   t", t, " ours ok", res.success, "steps", res.steps,
              " ref ok", ref_ok, "steps", ref_steps, " |dq|", dq)

    print("  success flag agrees:", n_agree, "/", N_TRIALS,
          "   both converged:", n_both_ok)
    print("  worst |site - target| where both converged:", worst_site_err)
    print("  worst |d qpos| (informational):", worst_dq,
          "   max |d steps|:", max_dsteps)

    assert_true(
        n_both_ok >= 4,
        "fewer than four trials converged on BOTH sides — the site-position"
        " gate below would then be nearly vacuous",
    )
    assert_true(
        n_agree == N_TRIALS,
        "our `success` disagrees with dm_control's on at least one trial",
    )
    assert_true(
        worst_site_err <= SITE_TOL,
        "a converged solve did not put the site on the target — this is the"
        " routine's actual contract",
    )


def test_set_site_to_xpos_matches_dm_control() raises:
    """The full `set_site_to_xpos`: IK, canonicalise, retry on failure.

    ⚠ THE RETRY DRAWS ARE PINNED, so this is a trajectory comparison and not
    a statistical one. The reference re-randomises the arm with
    `random_state.uniform(lower, upper)` once per failed attempt; the test
    reproduces that exact stream (same `RandomState`, same seed, same call
    order) via `manipulation_ref.retry_pose_draws` and injects it into ours.
    Both sides therefore visit the SAME sequence of start poses, and qpos can
    be compared directly — which it cannot be for IK alone.

    ⚠ If the two ever disagree about whether attempt k succeeded, the
    injected draws desynchronise from the reference's and every later attempt
    diverges. That is a feature: the qpos comparison below turns a
    single-attempt disagreement into a loud failure rather than a near-miss.
    """
    print("=== Jaco: set_site_to_xpos (IK + canonicalise + retry) ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var tempfile = Python.import_module("tempfile")
    var os = Python.import_module("os")
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var refmod = Python.import_module("manipulation_ref")

    var tmp = String(tempfile.mkdtemp(prefix="jaco_sstx_"))
    var xml_path = String(refmod.bake("reach_site_features", tmp))
    var cwd = String(os.getcwd())
    _ = os.chdir(tmp)
    var mm = mujoco.MjModel.from_xml_path(xml_path)
    var dat = mujoco.MjData(mm)
    var fmd = parse_xml_full(_read(xml_path))
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    build_model_fields_from_flat[DTYPE](fmd, mf)
    _ = os.chdir(cwd)
    var d = Data[DTYPE, MD, 1]()

    # Locate our index for the TCP site by position, as above.
    for i in range(NQ):
        var qv = 0.11 * Float64(i + 1) - 0.4
        dat.qpos[i] = qv
        d.qpos.data[i] = Scalar[DTYPE](qv)
    mujoco.mj_forward(mm, dat)
    forward_kinematics["cpu"](d, mf)
    var mj_tcp = Int(
        py=mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_SITE, refmod.TCP_SITE)
    )
    var our_tcp = -1
    var best_e = 1e30
    for s_ours in range(NSITE):
        var e = 0.0
        for k in range(3):
            var dd = abs(
                Float64(d.site_xpos.data[s_ours * 3 + k])
                - Float64(py=dat.site_xpos[mj_tcp][k])
            )
            if dd > e:
                e = dd
        if e < best_e:
            best_e = e
            our_tcp = s_ours
    assert_true(best_e < 1e-9, "could not locate our TCP site by position")

    # Arm DOFs, qpos addresses and sampling bounds — all from the reference.
    var arm_names = refmod.arm_joint_names()
    var bounds = refmod.arm_joint_bounds()
    var adr_py = refmod.arm_qpos_adr()
    var dof_idx = Array[Int, NDOF](fill=0)
    var qpos_adr = Array[Int, NDOF](fill=0)
    var lower = Array[Float64, NDOF](fill=0.0)
    var upper = Array[Float64, NDOF](fill=0.0)
    for a in range(NDOF):
        var jid = Int(
            py=mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, arm_names[a])
        )
        dof_idx[a] = Int(py=mm.jnt_dofadr[jid])
        qpos_adr[a] = Int(py=adr_py[a])
        lower[a] = Float64(py=bounds[0][a])
        upper[a] = Float64(py=bounds[1][a])
    print("  bounds[0]:", lower[0], lower[1], lower[2], lower[3], lower[4],
          lower[5])
    print("  bounds[1]:", upper[0], upper[1], upper[2], upper[3], upper[4],
          upper[5])
    # An unlimited hinge must come back as [0, 2pi] — if the reference ever
    # stopped doing that, the canonicalisation below would be wrapping into
    # the wrong window and still look plausible.
    assert_true(
        abs(upper[0] - 6.283185307179586) < 1e-9 and abs(lower[0]) < 1e-12,
        "the first arm joint is unlimited and should sample over [0, 2*pi]",
    )

    var down = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    down[0] = Scalar[DTYPE](0.70710678118)
    down[1] = Scalar[DTYPE](0.70710678118)

    comptime MAX_ATT: Int = 10
    var rng = np.random.default_rng(23)
    var n_agree = 0
    var n_ok = 0
    var worst_dq = 0.0
    var worst_oob = 0.0
    var max_attempts_used = 0

    for t in range(N_TRIALS):
        var q0 = np.zeros(NQ)
        for i in range(NDOF):
            q0[i] = Float64(py=rng.uniform(-2.0, 2.0))
        var tgt = np.zeros(3)
        tgt[0] = Float64(py=rng.uniform(-0.10, 0.10))
        tgt[1] = Float64(py=rng.uniform(-0.10, 0.10))
        tgt[2] = Float64(py=rng.uniform(0.28, 0.42))
        var rng_seed = 100 + t

        var rr = refmod.set_site_to_xpos_reference(
            "reach_site_features", q0, tgt, rng_seed
        )
        var ref_ok = Bool(py=rr[0])

        # The SAME draws the reference will make, in the same order.
        var draws = refmod.retry_pose_draws(MAX_ATT - 1, rng_seed)
        var retry = List[Scalar[DTYPE]]()
        for k in range((MAX_ATT - 1) * NDOF):
            retry.append(Scalar[DTYPE](Float64(py=draws[k])))

        for i in range(NQ):
            d.qpos.data[i] = Scalar[DTYPE](Float64(py=q0[i]))
        var tp = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        for k in range(3):
            tp[k] = Scalar[DTYPE](Float64(py=tgt[k]))

        var sres = set_site_to_xpos[DTYPE, NDOF](
            d, mf, our_tcp, tp, down, dof_idx, qpos_adr, lower, upper,
            retry, MAX_ATT,
        )
        var ok = sres.success
        if sres.attempts > max_attempts_used:
            max_attempts_used = sres.attempts

        if ok == ref_ok:
            n_agree += 1
        var dq = 0.0
        for i in range(NQ):
            var e = abs(Float64(d.qpos.data[i]) - Float64(py=rr[1][i]))
            if e > dq:
                dq = e
        if dq > worst_dq:
            worst_dq = dq
        if ok:
            n_ok += 1
            # A success must leave every arm joint INSIDE its bounds — that
            # is what the canonicalisation is for, so check it rather than
            # trusting the flag.
            for a in range(NDOF):
                var v = Float64(d.qpos.data[qpos_adr[a]])
                var oob = 0.0
                if v < lower[a]:
                    oob = lower[a] - v
                elif v > upper[a]:
                    oob = v - upper[a]
                if oob > worst_oob:
                    worst_oob = oob
        print("   t", t, " ours", ok, " attempts", sres.attempts,
              " ref", ref_ok, " |dq|", dq)

    print("  success agrees:", n_agree, "/", N_TRIALS, "  ours succeeded:",
          n_ok)
    print("  worst |d qpos|:", worst_dq, "  worst bound violation:",
          worst_oob, "  max attempts used:", max_attempts_used)

    assert_true(
        n_ok >= 4,
        "fewer than four trials succeeded — the bound and qpos checks would"
        " then be nearly vacuous",
    )
    assert_true(
        n_agree == N_TRIALS,
        "our set_site_to_xpos disagrees with dm_control's about success",
    )
    assert_true(
        worst_oob <= 1e-12,
        "a SUCCESSFUL result left an arm joint outside its sampling bounds —"
        " the canonicalisation did not do its job",
    )
    assert_true(
        max_attempts_used > 1,
        "every trial succeeded on the FIRST attempt, so the re-randomisation"
        " path and the injected draw sequence never ran — the trajectory"
        " pinning this test claims to verify is untested",
    )
    assert_true(
        worst_dq <= 1e-9,
        "qpos diverged from dm_control's despite identical injected retry"
        " draws, so the two took different trajectories",
    )


def test_set_grasp_and_target_sample_match_dm_control() raises:
    """`set_grasp` and the target-site sampler — the other two statements of
    `Reach.initialize_episode`.

    Needs no model build: both are pure arithmetic over joint ranges and a
    bounding box, so this compares them directly against the reference.

    ⚠ ONLY THE HAND JOINTS ARE COMPARED. `set_grasp_reference` returns the
    FULL qpos of a CACHED reference env whose arm joints still hold whatever
    the previous test left there, so comparing all of qpos would be comparing
    unrelated state.
    """
    print("=== set_grasp + target sample vs dm_control ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var np = Python.import_module("numpy")
    var refmod = Python.import_module("manipulation_ref")

    comptime NHAND = 3
    var info = refmod.hand_joint_info()
    assert_true(
        Int(py=Python.evaluate("len")(info[0])) == NHAND,
        "the reference hand does not have " + String(NHAND) + " joints",
    )
    var hadr = Array[Int, NHAND](fill=0)
    var hlo = Array[Float64, NHAND](fill=0.0)
    var hhi = Array[Float64, NHAND](fill=0.0)
    for i in range(NHAND):
        hadr[i] = Int(py=info[1][i])
        hlo[i] = Float64(py=info[2][i])
        hhi[i] = Float64(py=info[3][i])
    print("  hand qpos adr:", hadr[0], hadr[1], hadr[2],
          " range:", hlo[0], "..", hhi[0])
    # A degenerate range would make set_grasp a constant and the comparison
    # below vacuous for any close factor.
    assert_true(
        hhi[0] - hlo[0] > 1e-6,
        "the finger joint range is degenerate — set_grasp would be constant"
        " and this test could not distinguish any two close factors",
    )

    var worst_grasp = 0.0
    var n_g = 0
    for t in range(11):
        var f = 0.1 * Float64(t)
        var refq = refmod.set_grasp_reference(f)
        var qpos = List[Scalar[DTYPE]](length=NQ, fill=Scalar[DTYPE](0))
        var factors = Array[Float64, NHAND](fill=f)
        set_grasp[DTYPE, NHAND](qpos, hadr, hlo, hhi, factors)
        for i in range(NHAND):
            var e = abs(
                Float64(qpos[hadr[i]]) - Float64(py=refq[hadr[i]])
            )
            if e > worst_grasp:
                worst_grasp = e
        n_g += 1
    print("  set_grasp: close factors checked", n_g,
          " worst |d qpos|", worst_grasp)

    # ── target site sampler ──────────────────────────────────────────────
    var ws = refmod.reach_workspace()
    var lo = Array[Float64, 3](fill=0.0)
    var hi = Array[Float64, 3](fill=0.0)
    for k in range(3):
        lo[k] = Float64(py=ws[1][0][k])
        hi[k] = Float64(py=ws[1][1][k])
    print("  target_bbox:", lo[0], lo[1], lo[2], " ..", hi[0], hi[1], hi[2])

    var worst_pos = 0.0
    for sd in range(20):
        var rr = refmod.target_placer_reference(sd)
        var u = Array[Float64, 3](fill=0.0)
        for k in range(3):
            u[k] = Float64(py=rr[1][k])
        var got = sample_bbox_uniform[DTYPE](lo, hi, u)
        for k in range(3):
            var e = abs(Float64(got[k]) - Float64(py=rr[0][k]))
            if e > worst_pos:
                worst_pos = e
    print("  target sample: 20 seeds, worst |d pos|", worst_pos)

    assert_true(
        worst_grasp <= 1e-15, "set_grasp differs from the reference"
    )
    assert_true(
        worst_pos <= 1e-15,
        "the target sampler differs from distributions.Uniform(*target_bbox)"
        " given the same underlying draws",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
