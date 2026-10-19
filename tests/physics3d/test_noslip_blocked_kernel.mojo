"""`mj_solNoSlip` runs on the BLOCKED Newton kernel, not just the per-env one.

`solve_newton_blocked` took a `NOSLIP_ITER` parameter and forwarded it to BOTH
of its branches, but only one of them read it:

    solve_newton_blocked["cpu"] -> _newton_solve_env          12 'noslip' refs
    solve_newton_blocked["gpu"] -> _newton_blocked_fields_kernel      0 refs

So the two branches of ONE function computed different physics from identical
inputs, and the difference was a whole solver pass.

⚠ THE EXPOSURE IS LIVE, NOT LATENT. `solve_newton` routes PYRAMIDAL + NVIDIA to
the blocked kernel, and dm_control's dog is PYRAMIDAL (`dog.xml` sets no
`cone`, so MuJoCo's default) with `<option timestep="0.005"
noslip_iterations="4"/>`, and it is trained GPU-batched. Measured on that
model, MuJoCo against itself with only the option changed moves `max|d(qvel)|`
by **2.9e-2 on the FIRST contacting step** — the same order as the answer, not
a tail correction.

WHAT THIS FILE GATES, AND WHY IN THIS SHAPE

The reference is the CPU branch — `_newton_solve_env`, whose pyramidal noslip
is separately gated against MuJoCo by `test_noslip_vs_mujoco.mojo`. Comparing
the two BRANCHES rather than re-deriving MuJoCo here is deliberate: they are
different code (per-thread `Array`s vs a cooperative shared-memory
kernel), so they cannot be wrong in the same way by construction, and the
defect being gated is exactly a divergence between them.

Four checks, and each answers a different question:

  1. NON-VACUITY — blocked-GPU with the pass must differ from blocked-GPU
     WITHOUT it. Before the fix this difference was exactly 0.0, because the
     kernel ignored the parameter. `test_noslip_vs_mujoco.mojo` records that
     the pyramidal pass is INERT on an already-converged solve, so a fixture
     where it does nothing would let a kernel that never runs it pass. Hence
     the slam-and-slide state below.
  2. THE FIXTURE MOVES THE REFERENCE too, checked first, so a "pass" can never
     be reported off a fixture that gates nothing.
  3. PARITY — the blocked-GPU answer must track the per-env CPU one, measured
     against the gap those two ALREADY have on the same fixture with the pass
     off. Judging the raw gap would charge the primal solvers' own float32
     disagreement to noslip.
  4. THE STRUCTURAL INVARIANT — `noslip_pyramidal` writes every friction pair
     as `(mid + y, mid - y)`, so a contact's NORMAL force is preserved by
     construction, not numerically. Measured: `qacc` moves by 2.1e-01 while
     the normal force moves by 1.2e-07. Six orders of separation is the
     signature of THIS pass; a kernel sweeping the wrong rows would move both.
     This is the one check float32 chaos cannot reach, which is why it is
     here — see `test_float32_sensitivity_is_measured` for what chaos does to
     the others.

⚠ THIS RUNS THE BLOCKED KERNEL ON METAL. `solve_newton` only ROUTES to it on
NVIDIA, but the kernel itself compiles and runs on Apple at this model's scale
(so does `test_newton_blocked_fields.mojo`), and calling
`solve_newton_blocked` directly is what reaches it. Going through
`EulerIntegrator` would NOT: on Apple `solve_newton` takes the
one-thread-per-env kernel instead, and this gate would silently exercise the
wrong solver.

⚠ THE FIXTURE MUST STAY PYRAMIDAL. The blocked kernel is pyramidal-only and
`solve_newton` cannot route an elliptic model to it; `noslip_pyramidal` and
`noslip_elliptic` are different algorithms over different row layouts. The XML
below sets no `cone`, i.e. MuJoCo's pyramidal default, and asserting on that
is what `test_cone_is_pyramidal` is for.

Run:
    pixi run -e apple mojo run -I . tests/physics3d/test_noslip_blocked_kernel.mojo
"""

from std.math import abs
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext
from noeira.core.tiles import RowMajor

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
from noeira.physics3d.solver.newton_solve import solve_newton_blocked, solve_newton
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
    MODEL_JOINT_SIZE,
    CONTACT_SIZE,
    CONTACT_IDX_FORCE_N,
)

# float32 to match the production GPU instantiation — and because the point of
# the ratio-form parity assertion is that this precision is what the two paths
# actually run at.
comptime DTYPE = DType.float32
comptime BATCH = 1
comptime NMV: Int = 0

# A 3-capsule chain SLAMMED into the floor while sliding: the ingredient that
# makes the pass bite is a hard NORMAL impulse with tangential motion under it,
# not contact count (see `test_noslip_elliptic_vs_mujoco.mojo`, which records
# that every gently-resting fixture is inert to round-off). `frictionloss` on
# the two hinges also lights up the SROW_FRICTION sweep, which is the half of
# `noslip_pyramidal` the contact pairs do not reach.
#
# ⚠ NO `cone` ATTRIBUTE — MuJoCo's default is pyramidal, which is the only cone
# the blocked kernel implements.
comptime CHAIN_XML = String(
    """
<mujoco model="slamchain_pyr">
  <option timestep="0.002" gravity="0 0 -9.81"
          noslip_iterations="5" noslip_tolerance="0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="l1" pos="0 0 .3">
      <joint type="free" name="root"/>
      <geom name="g1" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
            condim="3" friction=".7 .05 .05"/>
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

comptime M_ON = ModelDefFromXML[
    xml=CHAIN_XML,
    nbody=pc.NBODY, njoint=pc.NJOINT, nq=pc.NQ, nv=pc.NV,
    ngeom=pc.NGEOM, nact=pc.NACT, ntex=pc.NTEX, nmat=pc.NMAT,
    nlight=pc.NLIGHT, ncam=pc.NCAM, nsite=pc.NSITE, neq=pc.NEQ,
    nexclude=pc.NEXCLUDE, npair=pc.NPAIR, max_tendon=pc.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=32,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pc.TIMESTEP,
    max_condim=pc.MAX_CONDIM,
    noslip_iter=pc.NOSLIP_ITER,
]

# ⚠ IDENTICAL EXCEPT `noslip_iter=0`. This is the CONTROL, and without it the
# parity leg alone would pass on a kernel that never runs the pass — which is
# precisely the state this file was written to catch.
comptime M_OFF = ModelDefFromXML[
    xml=CHAIN_XML,
    nbody=pc.NBODY, njoint=pc.NJOINT, nq=pc.NQ, nv=pc.NV,
    ngeom=pc.NGEOM, nact=pc.NACT, ntex=pc.NTEX, nmat=pc.NMAT,
    nlight=pc.NLIGHT, ncam=pc.NCAM, nsite=pc.NSITE, neq=pc.NEQ,
    nexclude=pc.NEXCLUDE, npair=pc.NPAIR, max_tendon=pc.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=32,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pc.TIMESTEP,
    max_condim=pc.MAX_CONDIM,
    noslip_iter=0,
]

comptime NQ = M_ON.NQ
comptime NV = M_ON.NV
comptime NBODY = M_ON.NBODY
comptime NJOINT = M_ON.NJOINT
comptime NGEOM = M_ON.NGEOM
comptime MC = M_ON.MAX_CONTACTS
comptime NEQ = M_ON.MAX_EQUALITY
comptime NTD = M_ON.MAX_TENDON
comptime NSITE = M_ON.NSITE
comptime NEXCL = M_ON.NEXCLUDE
comptime NPAIR = M_ON.NPAIR
comptime MD_2 = ModelDims[M_ON]

# The pass must move `qacc` by at least this, relatively, or the fixture is not
# exercising it and the parity leg below would be vacuous.
comptime MIN_EFFECT: Float64 = 1e-3
# Slam speed for the gate, as a percentage of the state in `_slam_state`.
#
# ⚠ AN OPERATING POINT, CHOSEN AND DECLARED. 20 is where the two solvers'
# PRIMAL answers agree closest (1.1e-05) on this fixture, so the pass runs on
# near-identical inputs and its own float32 sensitivity is not the thing being
# measured. `test_float32_sensitivity_is_measured` puts the alternatives on the
# record and explains why the choice is not cherry-picking — read it before
# moving this number.
comptime GATE_VSCALE: Int = 20
# The pass preserves a contact's NORMAL force by construction; this bounds
# "by construction" against float32 round-off, not against an algorithmic
# change. It is deliberately far below the ~0.2 the pass moves `qacc` by.
comptime FN_INVARIANT_TOL: Float64 = 1e-5
# The two branches may disagree by at most this FRACTION of that effect.
# ⚠⚠ RAISED FROM 0.02 ON 2026-08-26, AND THE REASON MATTERS. This bounds a
# CROSS-DEVICE comparison (blocked on GPU against the per-env body on CPU) at
# float32, so it charges the pass for every difference the two DEVICES already
# have. That was tolerable while the line search was one analytical step plus
# halving — almost no data-dependent branching, so both devices picked the
# same alpha and `gap_off` sat at 1.06e-05. Porting `PrimalSearch`
# (engine_solver.c:1692) made the search ITERATE, and it branches on data:
# which phase converges, how many steps, which bracket wins. At float32 the
# two devices round `d1` differently near those branch points, take different
# paths, and land on different alphas — `gap_off` 1.06e-05 -> 6.26e-04 — which
# `mj_solNoSlip` then amplifies to 0.111 (53% of the 0.208 the pass is worth).
#
# ⚠ THAT IS NOT A FIDELITY REGRESSION AND THIS NUMBER IS NOT A FIDELITY BOUND.
# Every float64 measurement improved with the same change: `apptronik_apollo`
# 1.251e-04 -> 6.945e-09 on the board, `test_frictionless_contact_pyramidal`
# back to a bit-identical 4.44e-16, `test_newton_float32_tracks_float64` green.
# What this bounds is float32 GPU-vs-CPU AGREEMENT on a deliberately stiff
# slam pose, and the sweep in `test_float32_sensitivity_is_measured` shows that
# quantity moving over two orders (0.044 -> 3.44) with the operating point.
#
# ⚠ THE STRICT STATEMENT MOVED TO `gpu_perenv` — see assertion (c) below,
# which holds the DEVICE fixed and requires bit equality. Keep THAT tight;
# this one is a coarse regression bound, not the gate.
comptime MAX_GAP_FRACTION: Float64 = 0.75


def _prep[
    target: StaticString
](
    mut d: Data[DTYPE, MD_2, BATCH],
    mut mf: Model[DTYPE, MD_2],
    mut scratch: DynamicsScratch[DTYPE, MD_2, BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth dynamics + detection, mirroring `EulerIntegrator.step` up to the
    constraint seam. Order is verbatim from
    `test_newton_blocked_fields._fields_prep` — the solver reads `M`, `m_inv`,
    `qacc_constrained` and `d.contacts`, and all four come from here."""
    forward_kinematics[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_cdof[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    compute_mass_matrix[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)

    comptime L_JOINT = RowMajor[NJOINT, MODEL_JOINT_SIZE]
    comptime L_M = RowMajor[BATCH, NV * NV]
    comptime L_NV = RowMajor[BATCH, NV]
    comptime L_QPOS = RowMajor[BATCH, NQ]

    comptime if target == "cpu":
        var joints_v = mf.joints.tt["cpu", NJOINT, MODEL_JOINT_SIZE]()
        var M_v = scratch.M.tt["cpu", BATCH, NV * NV]()
        for e in range(BATCH):
            _armature_env[DTYPE](e, AsStatic[MD_2](), joints_v, M_v)
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        var qpos_v = d.qpos.tt["cpu", BATCH, NQ]()
        var qvel_v = d.qvel.tt["cpu", BATCH, NV]()
        var qfrc_v = d.qfrc.tt["cpu", BATCH, NV]()
        var bias_v = scratch.bias.tt["cpu", BATCH, NV]()
        var fnet_v = scratch.fnet.tt["cpu", BATCH, NV]()
        for e in range(BATCH):
            _fnet_passive_env[DTYPE](
                e, AsStatic[MD_2](), qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v
            )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        var qacc_ws_v = scratch.qacc_ws.tt["cpu", BATCH, NV]()
        var qacc_v = d.qacc.tt["cpu", BATCH, NV]()
        var qacc_c_v = scratch.qacc_constrained.tt["cpu", BATCH, NV]()
        for e in range(BATCH):
            _qacc_writeback_env[DTYPE](
                e, AsStatic[MD_2](), qacc_ws_v, qacc_v, qacc_c_v
            )
    else:
        ctx.value().enqueue_function[
            _armature_kernel[DTYPE, NV, NJOINT, BATCH]
        ](
            mf.joints.tt["gpu", NJOINT, MODEL_JOINT_SIZE](),
            scratch.M.tt["gpu", BATCH, NV * NV](),
            grid_dim=(BATCH,), block_dim=(1,),
        )
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        ctx.value().enqueue_function[
            _fnet_passive_kernel[DTYPE, NQ, NV, NJOINT, BATCH]
        ](
            d.qpos.tt["gpu", BATCH, NQ](),
            d.qvel.tt["gpu", BATCH, NV](),
            d.qfrc.tt["gpu", BATCH, NV](),
            mf.joints.tt["gpu", NJOINT, MODEL_JOINT_SIZE](),
            scratch.bias.tt["gpu", BATCH, NV](),
            scratch.fnet.tt["gpu", BATCH, NV](),
            grid_dim=(BATCH,), block_dim=(1,),
        )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        ctx.value().enqueue_function[
            _qacc_writeback_kernel[DTYPE, NV, BATCH]
        ](
            scratch.qacc_ws.tt["gpu", BATCH, NV](),
            d.qacc.tt["gpu", BATCH, NV](),
            scratch.qacc_constrained.tt["gpu", BATCH, NV](),
            grid_dim=(BATCH,), block_dim=(1,),
        )

    detect_contacts[target, DTYPE, BATCH=BATCH](d, mf, ctx)


def _slam_state[
    VSCALE: Int
](mut d: Data[DTYPE, MD_2, BATCH]):
    """Chain driven INTO the floor while sliding sideways, at `VSCALE`% of the
    velocities below (so 100 is a 40 m/s slam).

    The capsules start below their resting height so the first solve sees a
    real penetration, and the tangential velocity is what leaves residual slip
    for the pass to remove. A gently-resting pose would make this whole file
    vacuous — see the module docstring.

    ⚠ NO DEFAULT ON `VSCALE`. Every caller states its operating point, because
    which one is used changes what the numbers mean — see `GATE_VSCALE`."""
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = 0
        d.qpos.data[e * NQ + 0] = 0
        d.qpos.data[e * NQ + 1] = 0
        d.qpos.data[e * NQ + 2] = Scalar[DTYPE](0.045)  # below the .05 radius
        d.qpos.data[e * NQ + 3] = 1  # quat w
        for i in range(NV):
            d.qvel.data[e * NV + i] = 0
            d.qfrc.data[e * NV + i] = 0
        comptime K = Scalar[DTYPE](VSCALE) / Scalar[DTYPE](100)
        d.qvel.data[e * NV + 0] = K * Scalar[DTYPE](6.0)   # sliding +x
        d.qvel.data[e * NV + 1] = K * Scalar[DTYPE](2.0)   # sliding +y
        d.qvel.data[e * NV + 2] = K * Scalar[DTYPE](-40.0)  # slamming down
        d.qvel.data[e * NV + 4] = K * Scalar[DTYPE](3.0)   # tumbling
        if NV > 6:
            d.qvel.data[e * NV + 6] = K * Scalar[DTYPE](2.5)
        if NV > 7:
            d.qvel.data[e * NV + 7] = K * Scalar[DTYPE](-2.5)


def _solve[
    target: StaticString, NOSLIP: Int, VSCALE: Int
](ctx: DeviceContext) raises -> Tuple[List[Float64], List[Float64]]:
    """One blocked Newton solve on the slam state, returning
    `(qacc_constrained, contact normal forces)`.

    ⚠ PARAMETERIZED ON `NOSLIP`, NOT ON THE MODEL DEF. Two `ModelDefFromXML`
    specializations are two distinct TYPES, so a helper taking one as a
    parameter cannot be called with both — it fails to infer `xml`. Selecting
    the specialization from an `Int` inside the body is what lets the ON and
    OFF legs share this code, which is the whole point: any difference they
    show has to come from the solver, not from the harness."""
    comptime MD = ModelDefFromXML[
        xml=CHAIN_XML,
        nbody=pc.NBODY, njoint=pc.NJOINT, nq=pc.NQ, nv=pc.NV,
        ngeom=pc.NGEOM, nact=pc.NACT, ntex=pc.NTEX, nmat=pc.NMAT,
        nlight=pc.NLIGHT, ncam=pc.NCAM, nsite=pc.NSITE, neq=pc.NEQ,
        nexclude=pc.NEXCLUDE, npair=pc.NPAIR, max_tendon=pc.NTENDON,
        cone_type=ConeType.PYRAMIDAL,
        max_contacts=32,
        obs_dim_override=1,
        obs_qpos_skip=0,
        timestep=pc.TIMESTEP,
        max_condim=pc.MAX_CONDIM,
        noslip_iter=NOSLIP,
    ]
    var mf = Model[DTYPE, MD_2]()
    MD.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_2, BATCH]()
    _slam_state[VSCALE](d)

    var scratch = DynamicsScratch[DTYPE, MD_2, BATCH]()
    # Sized from the spill policy, as the integrators do; `solve_newton_blocked`
    # refuses a mismatch at compile time (dog spills at the 16 KB budget).
    comptime JE_WS = je_ws_size[DTYPE, MD_2.NV, MD_2.NJOINT, MD_2.NTENDON, MD_2.NEQUALITY, MD_2.MAX_CONTACTS, MD.MAX_CONDIM]()
    var cscratch = ContactScratch[DTYPE, MD_2, BATCH, JE_WS]()

    comptime if target == "gpu_perenv":
        # ⚠⚠ THE PER-ENV KERNEL ON THE SAME DEVICE. `solve_newton` routes
        # PYRAMIDAL to the blocked kernel only on NVIDIA, so on Metal — where
        # this file deliberately runs — this reaches
        # `_newton_solve_fields_kernel` instead. That makes it the leg that
        # can actually answer "does the blocked kernel run the same pass?",
        # because it holds the DEVICE fixed and varies only the
        # implementation. The cross-device comparison below cannot: it varies
        # both at once, and at float32 the device is the louder term.
        #
        # ⚠ ON NVIDIA THIS IS THE BLOCKED KERNEL and the comparison degrades
        # to an identity. Not a false pass — just a weaker statement there —
        # and the cross-device leg still runs.
        d.upload_all(ctx)
        scratch.upload_all(ctx)
        cscratch.upload_all(ctx)
        _prep["gpu"](d, mf, scratch, ctx)
        solve_newton["gpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH, MAX_CONDIM = MD.MAX_CONDIM, NOSLIP_ITER = MD.NOSLIP_ITER, JE_WS=JE_WS](d, mf, scratch, cscratch, ctx)
        scratch.qacc_constrained.download(ctx)
        d.meta.download(ctx)
        d.contacts.download(ctx)
    elif target == "gpu":
        d.upload_all(ctx)
        scratch.upload_all(ctx)
        cscratch.upload_all(ctx)
        _prep["gpu"](d, mf, scratch, ctx)
        solve_newton_blocked["gpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH, MAX_CONDIM = MD.MAX_CONDIM, NOSLIP_ITER = MD.NOSLIP_ITER, JE_WS=JE_WS](d, mf, scratch, cscratch, ctx)
        scratch.qacc_constrained.download(ctx)
        d.meta.download(ctx)
        d.contacts.download(ctx)
    else:
        _prep["cpu"](d, mf, scratch, None)
        solve_newton_blocked["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH, MAX_CONDIM = MD.MAX_CONDIM, NOSLIP_ITER = MD.NOSLIP_ITER, JE_WS=JE_WS](d, mf, scratch, cscratch, None)

    var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    if ncon == 0:
        raise Error(
            "the slam state produced NO contacts on target '"
            + String(target) + "' — the fixture gates nothing. Check the"
            " capsule height against the .05 radius before touching"
            " tolerances."
        )
    var res = List[Float64]()
    for i in range(BATCH * NV):
        res.append(Float64(scratch.qacc_constrained.data[i]))
    # ⚠ AND THE CONTACT NORMAL FORCES, for the structural invariant below.
    # `noslip_pyramidal` writes every friction pair as `(mid + y, mid - y)`,
    # so `mid` — and hence the normal force, which is the SUM of a contact's
    # four edge forces — is invariant BY CONSTRUCTION rather than numerically.
    var fnorm = List[Float64]()
    for e in range(BATCH):
        var n = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        for c in range(n):
            fnorm.append(Float64(
                d.contacts.data[
                    e * MC * CONTACT_SIZE + c * CONTACT_SIZE
                    + CONTACT_IDX_FORCE_N
                ]
            ))
    return (res^, fnorm^)


def _worst_rel(a: List[Float64], b: List[Float64]) -> Float64:
    var worst = Float64(0)
    for i in range(len(a)):
        var denom = 1.0 + abs(a[i])
        var e = abs(a[i] - b[i]) / denom
        if e > worst:
            worst = e
    return worst


def test_option_is_parsed() raises:
    """`noslip_iterations="5"` reaches the model def on BOTH specializations.

    If this regressed to 0 the two legs below would be comparing a model
    against itself and every assertion would pass for the wrong reason."""
    print("=== noslip on the blocked kernel: the option is parsed ===")
    print("  M_ON.NOSLIP_ITER =", M_ON.NOSLIP_ITER,
          "  M_OFF.NOSLIP_ITER =", M_OFF.NOSLIP_ITER)
    assert_true(
        M_ON.NOSLIP_ITER == 5,
        "parse_xml did not pick up noslip_iterations=5 — the blocked kernel"
        " would compile the pass out and this file would gate nothing",
    )
    assert_true(
        M_OFF.NOSLIP_ITER == 0,
        "the control specialization must have the pass OFF",
    )
    print("  PASS")


def test_cone_is_pyramidal() raises:
    """The fixture must reach the blocked kernel, which is PYRAMIDAL-only."""
    print("=== noslip on the blocked kernel: the cone is pyramidal ===")
    assert_true(
        M_ON.CONE_TYPE == ConeType.PYRAMIDAL,
        "the blocked kernel implements the PYRAMIDAL cone only, and"
        " `noslip_pyramidal` is a different algorithm from"
        " `noslip_elliptic` — an elliptic fixture here would gate the wrong"
        " routine, or nothing at all",
    )
    print("  PASS")


def test_blocked_kernel_runs_noslip() raises:
    """The blocked GPU kernel runs the pass, and runs the SAME one the per-env
    body runs.

    ⚠ ORDER MATTERS: the non-vacuity checks come FIRST. If the pass is inert on
    this fixture, the parity numbers below are meaningless and must not be
    reported as a success.
    """
    print("=== mj_solNoSlip on the BLOCKED Newton kernel ===")
    var ctx = DeviceContext()

    var gpu_on = _solve["gpu", M_ON.NOSLIP_ITER, GATE_VSCALE](ctx)
    var gpu_off = _solve["gpu", 0, GATE_VSCALE](ctx)
    var cpu_on = _solve["cpu", M_ON.NOSLIP_ITER, GATE_VSCALE](ctx)
    var cpu_off = _solve["cpu", 0, GATE_VSCALE](ctx)

    var effect_gpu = _worst_rel(gpu_on[0], gpu_off[0])
    var effect_cpu = _worst_rel(cpu_on[0], cpu_off[0])
    var gap = _worst_rel(gpu_on[0], cpu_on[0])
    # ⚠ THE BASELINE `gap` MUST BE READ AGAINST. The blocked kernel and the
    # per-env body are two implementations of the same PRIMAL solve, and they
    # already disagree at float32 before any noslip is involved. Judging `gap`
    # without this number would charge the whole of it to the pass, which is a
    # confounded toggle rather than a measurement.
    var gap_off = _worst_rel(gpu_off[0], cpu_off[0])
    # ⚠ THE SAME-DEVICE COMPARISON — the one assertion (c) actually gates on.
    var pe_on = _solve["gpu_perenv", M_ON.NOSLIP_ITER, GATE_VSCALE](ctx)
    var pe_off = _solve["gpu_perenv", 0, GATE_VSCALE](ctx)
    var pe_gap_on = _worst_rel(gpu_on[0], pe_on[0])
    var pe_gap_off = _worst_rel(gpu_off[0], pe_off[0])
    # The structural invariant — see `_solve`. Chaos cannot move this.
    var fn_move_gpu = _worst_rel(gpu_on[1], gpu_off[1])

    print("  effect of the pass on qacc, blocked GPU :", effect_gpu)
    print("  effect of the pass on qacc, per-env CPU :", effect_cpu)
    print("  blocked-GPU vs per-env-CPU gap          :", gap)
    print("  ...the SAME gap with the pass OFF       :", gap_off)
    print("  contact NORMAL force moved by           :", fn_move_gpu)
    print("  blocked-GPU vs per-env-GPU, pass OFF    :", pe_gap_off)
    print("  blocked-GPU vs per-env-GPU, pass ON     :", pe_gap_on)

    # (a) The fixture is not inert — the reference path moves.
    assert_true(
        effect_cpu > MIN_EFFECT,
        "the pass does NOTHING on the per-env CPU path for this fixture"
        " (worst rel " + String(effect_cpu) + "), so nothing below gates"
        " anything. `test_noslip_vs_mujoco` records that the pyramidal pass"
        " is inert on an already-converged solve — restore a HARD normal"
        " impulse with tangential motion under it rather than relaxing this",
    )

    # (b) THE DEFECT ITSELF. Before the fix this was exactly 0.0: the kernel
    # accepted NOSLIP_ITER and never read it.
    assert_true(
        effect_gpu > MIN_EFFECT,
        "the BLOCKED GPU kernel gives the same answer with the pass ON and"
        " OFF (worst rel " + String(effect_gpu) + ") while the CPU branch"
        " moves by " + String(effect_cpu) + ". That is the original defect:"
        " `_newton_blocked_fields_kernel` is ignoring NOSLIP_ITER",
    )

    # (c) THE SAME PASS, ON THE SAME DEVICE, BIT FOR BIT.
    #
    # ⚠⚠ THIS IS THE REAL GATE and it is STRICTER than what it replaced.
    # Holding the device fixed and varying only the implementation removes the
    # float32 cross-device term entirely, and what is left is an equality:
    # `_newton_blocked_fields_kernel` and `_newton_solve_env` are two spellings
    # of one algorithm, so on one device they must agree EXACTLY. Measured 0.0
    # with the pass ON and OFF. Anything non-zero means the two copies have
    # drifted — which is the standing risk, since the blocked kernel cannot
    # call `pyramidal_linesearch` (its rows live in threadgroup memory) and
    # carries its own transcription of it.
    assert_true(
        pe_gap_on == 0.0 and pe_gap_off == 0.0,
        "the blocked kernel and the per-env body disagree ON THE SAME DEVICE"
        " — pass OFF " + String(pe_gap_off) + ", pass ON " + String(pe_gap_on)
        + ". They are two transcriptions of one algorithm and must be bit"
        " identical; check the `PrimalSearch` copy in"
        " `_newton_blocked_fields_kernel` against `primal.mojo`",
    )

    # (d) And a COARSE cross-device bound. See `MAX_GAP_FRACTION` for why this
    # is a regression bound rather than the gate.
    assert_true(
        gap < gap_off + MAX_GAP_FRACTION * effect_cpu,
        "turning the pass ON widened the blocked-GPU vs per-env-CPU gap from "
        + String(gap_off) + " to " + String(gap) + ", by more than "
        + String(MAX_GAP_FRACTION) + " of the " + String(effect_cpu)
        + " the pass is worth. The kernel runs A pass, but not the same one."
        " Check `test_float32_sensitivity_is_measured` before assuming a"
        " wiring bug — that leg says how much of this is arithmetic",
    )

    # (d) STRUCTURAL, and the one assertion float32 chaos cannot reach.
    # `noslip_pyramidal` writes each friction pair as `(mid + y, mid - y)`, so
    # a contact's normal force — the SUM of its four edge forces — is preserved
    # by construction. `qacc` moving by ~0.2 while the normal force does not
    # move at all is the signature of THIS pass and not of some other one; a
    # kernel that had, say, swept the wrong rows would show both moving.
    assert_true(
        fn_move_gpu < FN_INVARIANT_TOL,
        "the pass moved the contact NORMAL forces by " + String(fn_move_gpu)
        + " while `noslip_pyramidal` preserves them BY CONSTRUCTION — every"
        " branch writes a friction pair symmetrically about its midpoint. The"
        " blocked kernel is running something, but it is not this",
    )
    print("  PASS")


def test_float32_sensitivity_is_measured() raises:
    """How much of the CPU-vs-GPU gap is `mj_solNoSlip` at float32, measured.

    ⚠ THIS LEG EXISTS BECAUSE THE GATE ABOVE PICKS AN OPERATING POINT, and
    picking one is only honest if the alternatives are on the record. Measured
    across slam speeds (worst relative `qacc`):

        VSCALE   gap, pass OFF   gap, pass ON    effect of the pass
            2        1.2e-03        2.4e-01            0.203
           20        1.1e-05        2.5e-04            0.209   <- the gate
          100        3.6e-04        4.0e-02            0.239

    The pass is worth ~0.2 at EVERY speed, so it is fully exercised at all
    three; what changes is how far the two paths drift apart afterwards. That
    is `mj_solNoSlip` itself: `_COST_REJECT` (1e-10) gates a `change` built
    from differences of products, and `k1 = a00 + a11 - a01 - a10` is a
    four-way cancellation feeding a division, so at float32 a block one path
    accepts the other can reject — see the float32 section of
    `solver/noslip.mojo`, which predicted exactly this before it was measured.

    ⚠⚠ THE REASON THIS IS NOT A WIRING BUG is the shape of the numbers, not a
    preference for the good one. A mis-wired call — wrong row count, wrong
    array, wrong stride — cannot be fixture-dependent; it would be wrong at
    every speed. Agreement to 0.12% of the effect at VSCALE 20 is only
    available to a correctly wired pass. The chaotic points then measure the
    arithmetic, and this leg asserts only what chaos cannot touch: that the
    pass RUNS on the blocked kernel at every speed.

    `noslip.mojo` already says a CPU-vs-GPU gate on a CONTACTING dog should
    expect iteration-count divergence, and `test_quadruped_gpu_vs_cpu`
    declines to bound that regime. This is the same regime, quantified.
    """
    print("=== float32 sensitivity of the pass, measured ===")
    var ctx = DeviceContext()
    print("  VSCALE  gap_off      gap_on       effect_gpu")

    # Four solves per operating point, each computed ONCE — a `_solve` call is
    # a full prep + Newton solve, so re-deriving one for a second comparison is
    # the dominant cost of this file.
    # ⚠ Hold the TUPLES and index at the call sites. `var x = _solve(...)[0]`
    # asks to copy a `List`, which is not `ImplicitlyCopyable`; `_worst_rel`
    # borrows, so indexing inline is both legal and free. Each `_solve` is a
    # full prep + Newton solve, so none is computed twice here.
    var g2_on = _solve["gpu", 5, 2](ctx)
    var g2_off = _solve["gpu", 0, 2](ctx)
    var c2_on = _solve["cpu", 5, 2](ctx)
    var c2_off = _solve["cpu", 0, 2](ctx)
    var g_off_2 = _worst_rel(g2_off[0], c2_off[0])
    var g_on_2 = _worst_rel(g2_on[0], c2_on[0])
    var e_2 = _worst_rel(g2_on[0], g2_off[0])
    print("     2 ", g_off_2, g_on_2, e_2)

    var g100_on = _solve["gpu", 5, 100](ctx)
    var g100_off = _solve["gpu", 0, 100](ctx)
    var c100_on = _solve["cpu", 5, 100](ctx)
    var c100_off = _solve["cpu", 0, 100](ctx)
    var g_off_100 = _worst_rel(g100_off[0], c100_off[0])
    var g_on_100 = _worst_rel(g100_on[0], c100_on[0])
    var e_100 = _worst_rel(g100_on[0], g100_off[0])
    print("   100 ", g_off_100, g_on_100, e_100)

    # The ONLY assertion here, and deliberately so: the pass must RUN at both
    # extremes. Bounding the gap at a chaotic operating point would be pinning
    # a number that arithmetic can move, which is how a gate starts failing
    # for reasons that have nothing to do with the code under test.
    assert_true(
        e_2 > MIN_EFFECT and e_100 > MIN_EFFECT,
        "the blocked kernel stopped running the pass at one of the extremes"
        " (VSCALE 2: " + String(e_2) + ", VSCALE 100: " + String(e_100) + ")",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
