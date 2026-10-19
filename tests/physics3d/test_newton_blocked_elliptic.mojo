"""The blocked Newton kernel's ELLIPTIC leg against the per-env elliptic leg
— on the same device, bit for bit.

WHY (PERFORMANCE.md §13.53, 2026-09-15). `solve_newton` routed the ELLIPTIC
cone to the one-thread-per-env kernel on every device; on LIBERO (RTX 5090,
256 lanes) that kernel was 60% of the physics step at 23.6 ms per substep
for 1.27 Newton iterations a solve — the per-solve SETUP on one thread. The
blocked kernel (one env per block, cooperative rows / Hessian / factor)
existed for the pyramidal cone only. This file gates its elliptic leg.

WHAT IS GATED, AND IN WHAT SHAPE. The reference is `_newton_solve_env`, the
per-env elliptic leg, which `test_elliptic_linesearch_evals_vs_mujoco`,
`test_elliptic_condim46_vs_mujoco` and `test_noslip_elliptic_vs_mujoco` gate
against MuJoCo. Comparing the two LEGS rather than re-deriving MuJoCo here is
the same argument `test_noslip_blocked_kernel` makes: they are different
code over different memory (per-thread `Scratch` against cooperative
threadgroup arrays), so they cannot be wrong in the same way by
construction, and the defect a port can have is exactly a divergence
between them.

  1. NON-VACUITY, first. Every lane has contacts; the solve ITERATED (the
     published `META_IDX_NEWTON_ITER` is at least 1 somewhere) and the line
     search BRACKETED (`META_IDX_LS_EVAL` above the two-per-solve floor);
     a cone is ACTIVE (a tangential force is non-zero); and a condim-4
     contact carries a TORSIONAL force — the tangential-row mapping
     (`row0 + 1 + t` -> T1, T2, TORSION) is what a port gets wrong first.
  2. THE SAME DEVICE, BIT FOR BIT: blocked-GPU against per-env-GPU on
     `qacc_constrained`, every contact force slot, and the two counters.
     `newton_ell_coop.mojo` keeps the per-env leg's summation orders for
     exactly this assertion.
  3. A COARSE CROSS-DEVICE BOUND: blocked-GPU against per-env-CPU at
     float32 — a regression bound, not the gate (see
     `test_noslip_blocked_kernel`'s `MAX_GAP_FRACTION` for why).

⚠ THIS RUNS THE BLOCKED KERNEL ON METAL by calling `solve_newton_blocked`
directly; `solve_newton` only routes to it on NVIDIA. On NVIDIA the
`gpu_perenv` leg reaches the blocked kernel too and assertion 2 degrades
to an identity — the CPU bound still runs there. The fixture is
`test_noslip_blocked_kernel`'s slam chain with LIBERO's cone options
(`cone="elliptic" impratio="20"`), one capsule at `condim="4"`, and no
`noslip_iterations` (the elliptic leg has no `noslip_elliptic` port and
`solve_newton` keeps such models on the per-env kernel).

Run:
    pixi run -e apple mojo run -I . tests/physics3d/test_newton_blocked_elliptic.mojo
"""

from std.math import abs
from std.sys import default_accelerator
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext
from layout import Layout

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.fields import (
    AsStatic,
    Data, Model, DynamicsScratch, ContactScratch, Dims,
)
from noeira.physics3d.types import ConeType
from noeira.physics3d.solver.je_budget import je_ws_size
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from noeira.physics3d.dynamics.subtree_com import compute_subtree_com
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.mass_matrix import compute_mass_matrix
from noeira.physics3d.dynamics.ldl import (
    ldl_factor, ldl_solve, compute_m_inv,
)
from noeira.physics3d.dynamics.rne import compute_bias_forces_rne
from noeira.physics3d.integrator.euler import (
    _armature_kernel,
    _fnet_passive_kernel,
    _qacc_writeback_kernel,
    _armature_env,
    _fnet_passive_env,
    _qacc_writeback_env,
)
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.solver.newton_solve import (
    solve_newton_blocked, solve_newton,
)
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    META_IDX_NEWTON_ITER,
    META_IDX_LS_EVAL,
    METADATA_SIZE,
    MODEL_JOINT_SIZE,
    CONTACT_SIZE,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FORCE_TORSION,
    CONTACT_IDX_FORCE_ROLL1,
    CONTACT_IDX_FORCE_ROLL2,
)

# float32: the production GPU instantiation, and the precision the
# bit-equality assertion is about.
comptime DTYPE = DType.float32
comptime BATCH = 2

# `test_noslip_blocked_kernel`'s slam chain under LIBERO's `<option>`:
# elliptic cone, `impratio="20"` (robosuite's `base.xml`). `g1` is
# `condim="4"` so one contact carries a torsional row — the LIBERO gripper
# fingers are condim 4 — and its torsional friction is large enough for that
# row to carry force under the tumble below. No `noslip_iterations`.
comptime CHAIN_XML = String(
    """
<mujoco model="slamchain_ell">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic"
          impratio="20"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="l1" pos="0 0 .3">
      <joint type="free" name="root"/>
      <geom name="g1" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
            condim="4" friction=".7 .3 .05"/>
      <body name="l2" pos=".3 0 0">
        <joint type="hinge" name="j2" axis="0 1 0" range="-60 60"
               limited="true" frictionloss="0.05"/>
        <geom name="g2" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
              condim="3" friction=".7 .05 .05"/>
        <body name="l3" pos=".3 0 0">
          <joint type="hinge" name="j3" axis="0 1 0" range="-60 60"
                 limited="true" frictionloss="0.05"/>
          <geom name="g3" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
                condim="3" friction=".7 .05 .05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
)

comptime pc = parse_xml(CHAIN_XML)

comptime MD = ModelDefFromXML[
    xml=CHAIN_XML,
    nbody=pc.NBODY, njoint=pc.NJOINT, nq=pc.NQ, nv=pc.NV,
    ngeom=pc.NGEOM, nact=pc.NACT, ntex=pc.NTEX, nmat=pc.NMAT,
    nlight=pc.NLIGHT, ncam=pc.NCAM, nsite=pc.NSITE, neq=pc.NEQ,
    nexclude=pc.NEXCLUDE, npair=pc.NPAIR, max_tendon=pc.NTENDON,
    cone_type=ConeType.ELLIPTIC,
    max_contacts=32,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pc.TIMESTEP,
    max_condim=pc.MAX_CONDIM,
    noslip_iter=0,
]

comptime NQ = MD.NQ
comptime NV = MD.NV
comptime NJOINT = MD.NJOINT
comptime MC = MD.MAX_CONTACTS
comptime MDIMS = ModelDims[MD]
comptime JE_WS = je_ws_size[
    DTYPE, MDIMS.NV, MDIMS.NJOINT, MDIMS.NTENDON, MDIMS.NEQUALITY,
    MDIMS.MAX_CONTACTS, MD.MAX_CONDIM, CONE_TYPE=ConeType.ELLIPTIC,
]()

# The blocked-GPU vs per-env-CPU bound, float32 across two devices on a
# stiff slam — a regression bound. `test_noslip_blocked_kernel` measured
# 6.26e-04 for the pyramidal pair on this fixture's shape.
comptime CROSS_DEVICE_TOL: Float64 = 5e-3


struct Solved(Movable):
    var qacc: List[Float64]
    var forces: List[Float64]  # six slots per contact, env-major
    var condim: List[Int]      # per contact, env-major
    var ncon: List[Int]        # per env
    var iters: List[Int]       # per env
    var lsev: List[Int]        # per env

    def __init__(out self):
        self.qacc = List[Float64]()
        self.forces = List[Float64]()
        self.condim = List[Int]()
        self.ncon = List[Int]()
        self.iters = List[Int]()
        self.lsev = List[Int]()


def _prep[
    target: StaticString
](
    mut d: Data[DTYPE, MDIMS, BATCH],
    mut mf: Model[DTYPE, MDIMS],
    mut scratch: DynamicsScratch[DTYPE, MDIMS, BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth dynamics + detection up to the constraint seam — verbatim from
    `test_noslip_blocked_kernel._prep`."""
    forward_kinematics[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_cdof[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    compute_mass_matrix[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)

    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_QPOS = Layout.row_major(BATCH, NQ)

    comptime if target == "cpu":
        var joints_v = mf.joints.lt["cpu", L_JOINT]()
        var M_v = scratch.M.lt["cpu", L_M]()
        for e in range(BATCH):
            _armature_env[DTYPE](e, AsStatic[MDIMS](), joints_v, M_v)
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var qfrc_v = d.qfrc.lt["cpu", L_NV]()
        var bias_v = scratch.bias.lt["cpu", L_NV]()
        var fnet_v = scratch.fnet.lt["cpu", L_NV]()
        for e in range(BATCH):
            _fnet_passive_env[DTYPE](
                e, AsStatic[MDIMS](), qpos_v, qvel_v, qfrc_v, joints_v,
                bias_v, fnet_v,
            )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        var qacc_ws_v = scratch.qacc_ws.lt["cpu", L_NV]()
        var qacc_v = d.qacc.lt["cpu", L_NV]()
        var qacc_c_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        for e in range(BATCH):
            _qacc_writeback_env[DTYPE](
                e, AsStatic[MDIMS](), qacc_ws_v, qacc_v, qacc_c_v
            )
    else:
        ctx.value().enqueue_function[
            _armature_kernel[DTYPE, NV, NJOINT, BATCH]
        ](
            mf.joints.lt["gpu", L_JOINT](),
            scratch.M.lt["gpu", L_M](),
            grid_dim=(BATCH,), block_dim=(1,),
        )
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        ctx.value().enqueue_function[
            _fnet_passive_kernel[DTYPE, NQ, NV, NJOINT, BATCH]
        ](
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            d.qfrc.lt["gpu", L_NV](),
            mf.joints.lt["gpu", L_JOINT](),
            scratch.bias.lt["gpu", L_NV](),
            scratch.fnet.lt["gpu", L_NV](),
            grid_dim=(BATCH,), block_dim=(1,),
        )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        ctx.value().enqueue_function[
            _qacc_writeback_kernel[DTYPE, NV, BATCH]
        ](
            scratch.qacc_ws.lt["gpu", L_NV](),
            d.qacc.lt["gpu", L_NV](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            grid_dim=(BATCH,), block_dim=(1,),
        )

    detect_contacts[target, DTYPE, BATCH=BATCH](d, mf, ctx)


def _slam_state(mut d: Data[DTYPE, MDIMS, BATCH]):
    """The chain driven INTO the floor while sliding and tumbling — two
    lanes at two speeds, so the batch exercises `env` indexing and two
    different active sets. Same shape as `test_noslip_blocked_kernel`'s
    state; the spin about z is what loads `g1`'s torsional row."""
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = 0
        d.qpos.data[e * NQ + 2] = Scalar[DTYPE](0.045)  # below the .05 radius
        d.qpos.data[e * NQ + 3] = 1  # quat w
        for i in range(NV):
            d.qvel.data[e * NV + i] = 0
            d.qfrc.data[e * NV + i] = 0
        var K = Scalar[DTYPE](0.2) if e == 0 else Scalar[DTYPE](0.35)
        d.qvel.data[e * NV + 0] = K * Scalar[DTYPE](6.0)   # sliding +x
        d.qvel.data[e * NV + 1] = K * Scalar[DTYPE](2.0)   # sliding +y
        d.qvel.data[e * NV + 2] = K * Scalar[DTYPE](-40.0)  # slamming down
        d.qvel.data[e * NV + 4] = K * Scalar[DTYPE](3.0)   # tumbling
        d.qvel.data[e * NV + 5] = K * Scalar[DTYPE](12.0)  # spin about z
        if NV > 6:
            d.qvel.data[e * NV + 6] = K * Scalar[DTYPE](2.5)
        if NV > 7:
            d.qvel.data[e * NV + 7] = K * Scalar[DTYPE](-2.5)


def _solve[target: StaticString](ctx: DeviceContext) raises -> Solved:
    """One elliptic Newton solve on the slam state: `gpu_perenv` is
    `solve_newton["gpu"]` (the per-env kernel on Metal; the blocked one on
    NVIDIA), `gpu` is `solve_newton_blocked["gpu"]`, `cpu` its per-env
    CPU branch."""
    var mf = Model[DTYPE, MDIMS]()
    MD.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MDIMS, BATCH]()
    _slam_state(d)
    var scratch = DynamicsScratch[DTYPE, MDIMS, BATCH]()
    var cscratch = ContactScratch[DTYPE, MDIMS, BATCH, JE_WS]()

    comptime if target == "gpu_perenv":
        d.upload_all(ctx)
        scratch.upload_all(ctx)
        cscratch.upload_all(ctx)
        _prep["gpu"](d, mf, scratch, ctx)
        solve_newton["gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH, MAX_CONDIM = MD.MAX_CONDIM, NOSLIP_ITER = 0, JE_WS=JE_WS](d, mf, scratch, cscratch, ctx)
        scratch.qacc_constrained.download(ctx)
        d.meta.download(ctx)
        d.contacts.download(ctx)
    elif target == "gpu":
        d.upload_all(ctx)
        scratch.upload_all(ctx)
        cscratch.upload_all(ctx)
        _prep["gpu"](d, mf, scratch, ctx)
        solve_newton_blocked["gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH, MAX_CONDIM = MD.MAX_CONDIM, NOSLIP_ITER = 0, JE_WS=JE_WS](d, mf, scratch, cscratch, ctx)
        scratch.qacc_constrained.download(ctx)
        d.meta.download(ctx)
        d.contacts.download(ctx)
    else:
        _prep["cpu"](d, mf, scratch, None)
        solve_newton_blocked["cpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH, MAX_CONDIM = MD.MAX_CONDIM, NOSLIP_ITER = 0, JE_WS=JE_WS](d, mf, scratch, cscratch, None)

    var out = Solved()
    for i in range(BATCH * NV):
        out.qacc.append(Float64(scratch.qacc_constrained.data[i]))
    for e in range(BATCH):
        var mb = e * METADATA_SIZE
        var n = Int(d.meta.data[mb + META_IDX_NUM_CONTACTS])
        if n > MC:
            n = MC
        out.ncon.append(n)
        out.iters.append(Int(d.meta.data[mb + META_IDX_NEWTON_ITER]))
        out.lsev.append(Int(d.meta.data[mb + META_IDX_LS_EVAL]))
        for c in range(n):
            var cb = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
            out.condim.append(Int(d.contacts.data[cb + CONTACT_IDX_CONDIM]))
            out.forces.append(Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_N]))
            out.forces.append(Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_T1]))
            out.forces.append(Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_T2]))
            out.forces.append(
                Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_TORSION])
            )
            out.forces.append(
                Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_ROLL1])
            )
            out.forces.append(
                Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_ROLL2])
            )
    return out^


def _worst_rel(a: List[Float64], b: List[Float64]) -> Float64:
    var worst = Float64(0)
    for i in range(len(a)):
        var e = abs(a[i] - b[i]) / (1.0 + abs(a[i]))
        if e > worst:
            worst = e
    return worst


def _bit_equal(a: List[Float64], b: List[Float64]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def test_blocked_elliptic_matches_per_env() raises:
    print("=== the blocked Newton kernel's ELLIPTIC leg vs the per-env leg ===")
    var ctx = DeviceContext()
    var blk = _solve["gpu"](ctx)
    var pe = _solve["gpu_perenv"](ctx)
    var cpu = _solve["cpu"](ctx)

    print("  contacts per lane, blocked :", blk.ncon[0], blk.ncon[1])
    print("  Newton iterations per lane :", blk.iters[0], blk.iters[1],
          "| per-env", pe.iters[0], pe.iters[1])
    print("  line-search evals per lane :", blk.lsev[0], blk.lsev[1],
          "| per-env", pe.lsev[0], pe.lsev[1])

    # ── 1. non-vacuity ────────────────────────────────────────────────────
    for e in range(BATCH):
        assert_true(
            blk.ncon[e] > 0,
            "lane " + String(e) + " produced NO contacts — the slam state"
            " gates nothing; check the capsule height against the .05 radius",
        )
    var iters_max = 0
    var lsev_sum = 0
    for e in range(BATCH):
        if blk.iters[e] > iters_max:
            iters_max = blk.iters[e]
        lsev_sum += blk.lsev[e]
    assert_true(
        iters_max >= 1,
        "no lane took a Newton step — `META_IDX_NEWTON_ITER` is 0 on both,"
        " so the solve did nothing and the equality below is vacuous",
    )
    assert_true(
        lsev_sum > 2 * BATCH,
        "the line search spent " + String(lsev_sum) + " evaluations over "
        + String(BATCH) + " lanes, i.e. the two-per-solve floor — nothing"
        " bracketed, and the elliptic `PrimalEval` transcription is untested",
    )
    var tangential_live = False
    var torsion_live = False
    var k = 0
    for c in range(len(blk.condim)):
        var f_norm = blk.forces[6 * c + 0]
        var f_tan = abs(blk.forces[6 * c + 1]) + abs(blk.forces[6 * c + 2])
        if f_norm > 0.0 and f_tan > 1e-6 * f_norm:
            tangential_live = True
        if blk.condim[c] >= 4 and abs(blk.forces[6 * c + 3]) > 0.0:
            torsion_live = True
        k += 1
    assert_true(
        tangential_live,
        "no contact carries a tangential force — every cone is at its"
        " apex, and the elliptic coupling this file exists to gate is"
        " inert on this state",
    )
    assert_true(
        torsion_live,
        "no condim-4 contact carries a TORSIONAL force. `g1` spins about the"
        " normal with friction .7 .3 .05, so its t = 2 row must load;"
        " a zero here is the tangential-row mapping (`row0 + 1 + t` ->"
        " `CONTACT_IDX_FORCE_TORSION`) or the load of `ws_ell_ntc`",
    )

    # ── 2. the same device, bit for bit ───────────────────────────────────
    var gap_q = _worst_rel(blk.qacc, pe.qacc)
    var gap_f = _worst_rel(blk.forces, pe.forces)
    print("  blocked-GPU vs per-env-GPU  qacc worst rel:", gap_q,
          " forces:", gap_f)
    if default_accelerator().is_nvidia_gpu():
        print("  (NVIDIA: `solve_newton` routes ELLIPTIC to the blocked"
              " kernel too — the same-device leg is an identity here)")
    assert_true(
        _bit_equal(blk.qacc, pe.qacc) and _bit_equal(blk.forces, pe.forces),
        "the blocked kernel's ELLIPTIC leg and the per-env elliptic leg"
        " disagree ON THE SAME DEVICE — qacc " + String(gap_q) + ", forces "
        + String(gap_f) + ". `newton_ell_coop.mojo` keeps the per-env"
        " leg's summation orders so these are two spellings of one"
        " algorithm; a difference is a transcription defect, not float32",
    )
    for e in range(BATCH):
        assert_true(
            blk.iters[e] == pe.iters[e] and blk.lsev[e] == pe.lsev[e],
            "lane " + String(e) + ": the two legs took a different number of"
            " Newton iterations or line-search evaluations (blocked "
            + String(blk.iters[e]) + "/" + String(blk.lsev[e]) + ", per-env "
            + String(pe.iters[e]) + "/" + String(pe.lsev[e]) + ") while"
            " agreeing on the answer — an exit test or the p0 evaluation"
            " guard drifted",
        )

    # ── 3. the coarse cross-device bound ──────────────────────────────────
    var gap_c = _worst_rel(blk.qacc, cpu.qacc)
    print("  blocked-GPU vs per-env-CPU  qacc worst rel:", gap_c)
    assert_true(
        gap_c < CROSS_DEVICE_TOL,
        "blocked-GPU against the per-env CPU leg differs by " + String(gap_c)
        + " (bound " + String(CROSS_DEVICE_TOL) + ") — beyond the float32"
        " cross-device band the pyramidal pair sits in on this fixture",
    )
    print("  PASS")


def main() raises:
    var suite = TestSuite()
    suite.test[test_blocked_elliptic_matches_per_env]()
    suite^.run()
