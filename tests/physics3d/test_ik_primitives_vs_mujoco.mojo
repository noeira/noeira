"""`quat2vel` and `integrate_pos` against `mju_quat2Vel` / `mj_integratePos`.

STEP 2 OF THE dm_control PHASE 7 RESET PATH, after `jac_site`. These are the
two primitives `qpos_from_site_pose` needs that the tree did not already have:
the rotational error term is `quat2Vel(target (x) conj(site_quat))`, and each
Newton step is applied with `mj_integratePos(m, qpos, update, 1)`.

⚠ QUATERNION ORDER DIFFERS BETWEEN THE TWO SIDES. MuJoCo stores `(w, x, y, z)`;
`kinematics/quat_math` takes and returns `(x, y, z, w)`. Every conversion in
this file is explicit for that reason — a test that got the order wrong would
still produce plausible-looking rotations and would fail only on asymmetric
inputs, which is exactly why the sweep below uses random quaternions rather
than axis-aligned ones.

WHAT EACH TEST IS FOR

`quat2vel`: the danger is not the formula but the `speed > pi` wrap, which is
what makes the function sign-invariant. The IK caller depends on that: it lets
the error chain use the site's world quaternion composed directly
(`xquat[body] * site_quat`, which `kinematics/site_frame` already provides)
instead of porting `mju_mat2Quat` to round-trip through `site_xmat`. The round
trip can only differ by an overall sign. So the sweep feeds BOTH `q` and `-q`
and requires the pair to agree, in addition to matching MuJoCo — the property
being relied on is tested directly, not inferred from the fact that the
formula matches.

`integrate_pos`: the danger is the quaternion joints. A free joint's qpos and
qvel have different widths (7 vs 6) and its rotation is an exponential map,
not an addition. The sweep therefore uses LARGE updates (IK allows an update
norm up to 2.0 rad, far outside the small-angle regime a physics timestep
stays in) — a first-order quaternion update would pass at `dt=1e-3` and fail
here. ⚠ The BALL branch is NOT covered: nothing in the tree has a ball joint.

Run with:
    pixi run mojo run -I . tests/physics3d/test_ik_primitives_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext
from noeira.core.tiles import RowMajor

from noeira.envs.dm_control.fish import DMFishUprightModel
from noeira.physics3d.fields import Model, Data, Dims, AsStatic
from noeira.physics3d.kinematics.quat_math import quat2vel
from noeira.physics3d.kinematics.integrate_pos import integrate_pos
from noeira.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
)
from noeira.physics3d.joint_types import JNT_FREE

comptime DTYPE = DType.float64

# Both sides run the same float64 formula; the only freedom is the order of a
# handful of operations inside atan2/sqrt. Measured worst is printed.
comptime VEL_TOL: Float64 = 1e-14
comptime POS_TOL: Float64 = 1e-14

comptime N_QUATS: Int = 2000
comptime N_POSE_TRIALS: Int = 200


def test_quat2vel_matches_mujoco() raises:
    """Random quaternions, both signs, several `dt`."""
    print("--- quat2vel vs mju_quat2Vel ---")
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var rng = np.random.default_rng(12345)

    var worst = 0.0
    var worst_sign = 0.0
    var n = 0
    var n_wrapped = 0

    # A mix: unit-random (mostly generic), plus deliberately near-identity and
    # near-pi rotations, where atan2's two arguments go to zero in turn.
    for trial in range(N_QUATS):
        var q = rng.normal(0.0, 1.0, 4)
        if trial % 7 == 0:
            # near identity: w ~ 1, tiny vector part
            q = np.array([1.0, 1e-9, -2e-9, 3e-10])
        elif trial % 11 == 0:
            # near pi: w ~ 0
            q = np.array([1e-10, 0.3, -0.5, 0.81])
        q = q / np.linalg.norm(q)

        var dt = 1.0
        if trial % 3 == 1:
            dt = 0.5
        elif trial % 3 == 2:
            dt = 2.0

        # MuJoCo order is (w, x, y, z).
        var qw = Float64(py=q[0])
        var qx = Float64(py=q[1])
        var qy = Float64(py=q[2])
        var qz = Float64(py=q[3])

        var mj_v = np.zeros(3)
        mujoco.mju_quat2Vel(mj_v, q, dt)
        var mj_vneg = np.zeros(3)
        mujoco.mju_quat2Vel(mj_vneg, -q, dt)

        var got = quat2vel[DTYPE](
            Scalar[DTYPE](qx),
            Scalar[DTYPE](qy),
            Scalar[DTYPE](qz),
            Scalar[DTYPE](qw),
            Scalar[DTYPE](dt),
        )
        var got_neg = quat2vel[DTYPE](
            Scalar[DTYPE](-qx),
            Scalar[DTYPE](-qy),
            Scalar[DTYPE](-qz),
            Scalar[DTYPE](-qw),
            Scalar[DTYPE](dt),
        )

        # The wrap only fires for w < 0; count it so a sweep that never
        # exercised it cannot pass silently.
        if qw < 0.0:
            n_wrapped += 1

        # ⚠ Tuple slots need comptime indices, so this is unrolled.
        var e0 = abs(Float64(got[0]) - Float64(py=mj_v[0]))
        var e1 = abs(Float64(got[1]) - Float64(py=mj_v[1]))
        var e2 = abs(Float64(got[2]) - Float64(py=mj_v[2]))
        if e0 > worst:
            worst = e0
        if e1 > worst:
            worst = e1
        if e2 > worst:
            worst = e2

        # ⚠ The property the IK chain relies on: q and -q are the same
        # rotation and MUST give the same velocity.
        var s0 = abs(Float64(got[0]) - Float64(got_neg[0]))
        var s1 = abs(Float64(got[1]) - Float64(got_neg[1]))
        var s2 = abs(Float64(got[2]) - Float64(got_neg[2]))
        if s0 > worst_sign:
            worst_sign = s0
        if s1 > worst_sign:
            worst_sign = s1
        if s2 > worst_sign:
            worst_sign = s2
        n += 1

    print("  quats checked:", n, " (with w<0, exercising the wrap:",
          n_wrapped, ")")
    print("  worst |d(quat2vel)| vs MuJoCo:", worst)
    print("  worst |quat2vel(q) - quat2vel(-q)|:", worst_sign)

    assert_true(n == N_QUATS, "the sweep fell through")
    assert_true(
        n_wrapped > 100,
        "almost no quaternion had w < 0, so the `speed > pi` wrap — the whole"
        " reason this is not `quat_to_axis_angle` — was never exercised",
    )
    assert_true(
        worst <= VEL_TOL, "quat2vel differs from mju_quat2Vel"
    )
    assert_true(
        worst_sign <= VEL_TOL,
        "quat2vel(q) != quat2vel(-q) — the IK error chain composes the site"
        " quaternion directly instead of round-tripping through mj_mat2Quat,"
        " and that substitution is only safe if this holds",
    )


def test_integrate_pos_matches_mujoco() raises:
    """Fish — a free joint (7 qpos / 6 qvel) plus hinges."""
    print("--- integrate_pos vs mj_integratePos (fish) ---")
    comptime M = DMFishUprightModel
    comptime NQ: Int = M.NQ
    comptime NV: Int = M.NV
    comptime NJOINT: Int = M.NJOINT
    comptime MD = Dims[
        nq=NQ,
        nv=NV,
        nbody=M.NBODY,
        njoint=NJOINT,
        ngeom=M.NGEOM,
        nsite=M.NSITE,
        max_contacts=M.MAX_CONTACTS,
        nequality=M.MAX_EQUALITY,
        ntendon=M.MAX_TENDON,
        nexclude=M.NEXCLUDE,
        nmesh_verts=0,
        npair=M.NPAIR,
        nact=M.NACT,
        nten=M.NTEN_F,
        nkey=M.NKEY,
    ]
    var sf = M.make_spec_fields[DTYPE]()

    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var rng = np.random.default_rng(7)
    var mm = mujoco.MjModel.from_xml_path("noeira/envs/dm_control/assets/fish.xml")
    var dat = mujoco.MjData(mm)

    assert_true(Int(py=mm.nq) == NQ, "nq mismatch")
    assert_true(Int(py=mm.nv) == NV, "nv mismatch")

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)

    comptime L_JNT = RowMajor[NJOINT, MODEL_JOINT_SIZE]
    comptime L_QPOS = RowMajor[1, NQ]
    comptime L_QVEL = RowMajor[1, NV]
    var joints_v = mf.joints.tt["cpu", NJOINT, MODEL_JOINT_SIZE]()
    var qpos_v = d.qpos.tt["cpu", 1, NQ]()
    var qvel_v = d.qvel.tt["cpu", 1, NV]()

    var n_free = 0
    for j in range(NJOINT):
        if Int(rebind[Scalar[DTYPE]](joints_v[j, JOINT_IDX_TYPE])) == JNT_FREE:
            n_free += 1
    print("  nq", NQ, " nv", NV, " njoint", NJOINT, " free joints", n_free)
    assert_true(
        n_free > 0,
        "fish has no FREE joint — the quaternion path of integrate_pos, the"
        " only part that is not a scalar addition, is then never entered",
    )

    var worst = 0.0
    var worst_trial = -1

    for t in range(N_POSE_TRIALS):
        # ⚠ LARGE updates on purpose: IK permits |update| up to 2.0 rad, so a
        # first-order quaternion step must fail here rather than squeak by in
        # the small-angle regime a physics timestep lives in.
        var scale = 0.05 + 1.2 * Float64(t % 5)
        var qp = rng.normal(0.0, 1.0, NQ)
        var dq = rng.normal(0.0, scale, NV)
        var dt = 1.0
        if t % 4 == 1:
            dt = 0.37
        elif t % 4 == 2:
            dt = 2.5

        for i in range(NQ):
            dat.qpos[i] = Float64(py=qp[i])
        # Normalise the free-joint quaternion; MuJoCo assumes a unit quat here
        # and a non-unit one would make the two sides integrate different
        # rotations.
        mujoco.mj_normalizeQuat(mm, dat.qpos)
        for i in range(NV):
            dat.qvel[i] = Float64(py=dq[i])

        for i in range(NQ):
            d.qpos.data[i] = Scalar[DTYPE](Float64(py=dat.qpos[i]))
        for i in range(NV):
            d.qvel.data[i] = Scalar[DTYPE](Float64(py=dat.qvel[i]))

        mujoco.mj_integratePos(mm, dat.qpos, dat.qvel, dt)
        integrate_pos(
            0, AsStatic[MD](), qpos_v, qvel_v, joints_v, Scalar[DTYPE](dt)
        )

        for i in range(NQ):
            var e = abs(
                Float64(d.qpos.data[i]) - Float64(py=dat.qpos[i])
            )
            if e > worst:
                worst = e
                worst_trial = t

    print("  trials:", N_POSE_TRIALS, " worst |d(qpos)|:", worst,
          " at trial", worst_trial)
    assert_true(
        worst <= POS_TOL, "integrate_pos differs from mj_integratePos"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
