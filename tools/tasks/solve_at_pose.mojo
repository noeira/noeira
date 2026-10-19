"""ONE LIBERO STATE, ONE ELLIPTIC SOLVE, THREE LEGS — which one is wrong?

    # the state comes from the family driver:
    #   libero_family_batched --dump-at-step 1 --dump-state build/diag/lr3_step1.txt
    pixi run mojo run -I . tools/tasks/solve_at_pose.mojo build/diag/lr3_step1.txt 0
    #   <states.txt> <lane_a> <step> <lane_b>
    pixi run mojo run -I . tools/tasks/solve_at_pose.mojo build/diag/lr3_all.txt 1 1 2

## ⚠⚠ THE ANSWER FOR ONE SOLVE — AND READ THE SCOPE BELOW

At this state the blocked elliptic kernel is NOT the worse of the two legs.
On the reproduction state (`libero_living_room_scene3`, step 1):

    lane 0   per-env 10 iters / 147 line-search evals   blocked 10 / 148
    worst |qacc| between the two float32 legs           5.72e-06
    worst |qacc - CPU float64 reference|   per-env 9.109e-03   blocked 9.103e-03
    reference max |qacc| 37.2 m/s^2   =>   both 2.4e-04 RELATIVE, ratio 0.9994

The two legs differ from each other by ~1600x LESS than they each differ from
the truth, and the blocked leg is the marginally nearer one. The gap they
share is float32 against float64 on a converged solve — all three legs
converge, in 10 iterations, well inside the model's budget of 100.

⚠ LANE 1 IS THE CONTROL AND IT WAS NOT PLANNED. There the two float32 legs
are BIT-IDENTICAL (both 8.980136837440256e-03 from the reference) and the gap
to float64 is the SAME size. A difference that survives when the accused
kernel agrees with its accuser is not that kernel's difference.

## ⚠ THE SCOPE OF THE BLOCK ABOVE: ONE SOLVE, NOT A ROLLOUT

On its own the block above says only that at THIS state, for ONE solve, the
blocked leg is not the worse of the two. The acquittal comes from the float64
sweep in the next section, not from here.

## ⚠⚠ THE ROLLOUT ARM COMPARISON IS AN ACCURACY COMPARISON — IT COUNTS

`libero_family_batched`'s CPU leg is `comptime H = DType.float64`, so the 5090
run that put blocked at 0.04190795940899783 and per-env at 0.0018 measured both
against a FLOAT64 reference. Both numbers are `alphabet_soup_1_joint0[2]` —
that body's z — at step 4. So over a rollout the blocked leg is genuinely ~23x
further from the truth than the per-env leg, and this file's single-solve
result does NOT excuse that.

⚠ AN EARLIER VERSION OF THIS HEADER SAID THOSE ARMS WERE SCORED AGAINST A
FLOAT32 CPU LEG and therefore ranked by summation order rather than accuracy.
That was wrong — it confused this file's `_solve64`/float32 pair with the
family gate's float64 leg. The order-of-summation argument applies to THIS
file's two GPU legs, not to the family gate's verdict.

## ⚠⚠ ANSWERED ON CUDA 2026-09-18: IT IS ROUNDING. THE BLOCKED KERNEL IS SOUND.

The float64 sweep ran on a 5090 (`scripts/libero_f64_sweep_cuda.sh`) and the
offset TRACKS MACHINE EPSILON:

    float32   worst |qacc| diff 5.722046e-06  / scale 37.22 = 1.29 ULP
    float64   worst |qacc| diff 7.105427e-15  / scale 37.22 = 0.86 ULP
    the diffs fell 8.05e8 while eps fell 5.37e8 — the same order
    bias |mean|/mean|.|   float32 0.975   ->   float64 0.330

About one ULP of the answer's own magnitude in BOTH precisions. A logic
difference keeps its RELATIVE size when the precision rises; this did not. So
the difference is non-associativity — the two-stage `JH` in
`ell_add_contact_hessian` against the per-entry recompute in
`_ell_entry_contact_term`, and/or FMA contraction applied in one leg and not the
other — and `solve_newton_blocked`'s elliptic leg is NOT defective.

⚠ THE 0.975 BIAS RATIO WAS REAL BUT IT WAS NOT EVIDENCE OF A DEFECT. A rounding
difference CAN be almost perfectly one-sided: at float32 a single systematic
rounding decision dominates all 90 dofs, and at float64 the same mechanism is
5e8 times smaller so ordinary rounding takes over and the ratio falls to 0.330.
"Systematic" narrows the mechanism; it does not by itself convict a kernel.

⚠ AND BIT-IDENTITY AT FLOAT64 WAS THE WRONG CRITERION. The first version of
this file's verdict demanded `worst == 0.0` at float64 and would have called
this run a LOGIC difference — it printed exactly that before being corrected.
Rounding does not go to zero with more precision, it goes to ~1 ULP. Compare
ULPs ACROSS precisions, never against zero.

## WHAT THIS MEANS FOR THE LIBERO PORT

`libero_family_batched`'s 1e-3 window bound compares a GPU float32 rollout
against a CPU FLOAT64 one, so it varies precision AND implementation at once and
is dominated by float32 discretisation amplified through contact chaos. No
kernel fix can meet it. Either add the fixed-axis leg (CPU float32, which
isolates the implementation) or justify the bound against what float32 can
actually deliver on a scene of boxes resting at the contact margin.

## ⚠ HOW BOTH RESULTS CAN BE TRUE AT ONCE

One solve: the legs are 5.7e-06 apart and both 9.1e-03 from float64.
One rollout: the legs are 0.042 and 0.0018 from float64.

Two readings survive that, and they call for different fixes:

1. **A knife edge.** The 5.7e-06 decides a DISCRETE event over 125 substeps —
   a box on the table tips or does not — and which side you land on is not an
   accuracy property. Then a one-ULP perturbation of the PER-ENV leg would
   also throw it to ~0.04, and the scene, not the kernel, is the problem.
2. **A BIAS.** If the blocked leg's per-solve error is systematic rather than
   random — say it consistently over-supports a resting contact — then 5.7e-06
   a solve integrates into 0.04 over the window while per-env's unbiased error
   does not. Then the blocked kernel IS wrong and the single solve above was
   too blunt to see it, because it reports `worst |.|` and a bias needs a
   SIGNED mean.

The signed-mean column below settles that the difference IS systematic
(|mean|/mean|.| = 0.975 over 90 dofs, where random rounding sits near 0.1).
What it does NOT settle is whether that systematic offset is a ROUNDING bias —
two assembly orders, or FMA contraction applied in one leg and not the other,
both of which are one-directional — or a genuine logic difference.

⚠⚠ THE TEST FOR THAT IS NOT RUNNABLE ON METAL, AND THE VACUOUS VERSION OF IT
LOOKS LIKE A PASS. See `_solve64`: `solve_newton_blocked["cpu", ...]` falls
back to the per-env body, so a CPU float64 leg-vs-leg sweep compares a function
with itself and prints 0.0 at every iteration. It needs a CUDA box, where
`solve_newton_blocked["gpu", DType.float64]` is a real second implementation.

## WHY A STEPPING RUN CANNOT ANSWER THIS EITHER

`libero_family_batched --solver-log` shows the two legs bit-identical through
step 1 and diverging at step 2, the line-search work first (cumulative
evaluations 2997 against 3006) while the state still matches to six digits.
After that every difference is downstream of the first, so a rollout says only
THAT they part. This replays ONE state through both, with identical inputs.

`tests/physics3d/test_newton_blocked_elliptic.mojo` does the same for a slam
chain of capsules and asserts bit equality — and PASSES. LIBERO is boxes
resting at the contact margin, 18 contacts a lane, which that fixture never
produces. The pre-solve pipeline below is its `_prep`, verbatim, for that
reason: the two files must feed their solves the same way.

## ⚠ THE DIVERGENCE FOLLOWS THE STATE, NOT THE LANE SLOT

Commit 4412f5b81's message says it "needs two lanes". THAT IS WRONG and this
file is the correction: lane 1's state placed in slot 0 reproduces it at
BATCH 1. Two lanes are still carried below, for the separate reason that a
per-env indexing mistake in a one-block-per-env kernel cannot show at BATCH 1
— and lane 1 earned its keep as the control above.

## HOW THE ITERATION SWEEP LOCALISED IT

Capping the Newton budget at k and comparing the k-th iterate: bit-identical
through k=6; at k=7 BOTH legs do identical work (7 iterations, 127 evaluations
each) and the answers already differ by 2.384e-07 — one float32 ULP. The extra
evaluation at k=8 is a CONSEQUENCE of that ULP, not its cause, which is what
retires the line search as a suspect.

## ⚠ max_contacts IS LOWERED TO FIT METAL

`libero_living_room_scene3` at 64 asks for 32 852 B of threadgroup memory and
Metal's limit is 32 768 — 84 bytes over. At 56 it fits, and the family's
measured contact peak is 28 (`contact_budget.kv`), so nothing is truncated.
Both legs run at the SAME cap, which is what the comparison needs.
"""

from std.math import abs
from std.sys import argv
from max.gpu.host import DeviceContext
from layout import Layout

from noeira.physics3d.fields import (
    AsStatic, Data, Model, DynamicsScratch, ContactScratch, Dims,
)
from noeira.physics3d.types import ConeType
from noeira.physics3d.solver.je_budget import je_ws_size
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics, compute_body_velocities,
)
from noeira.physics3d.dynamics.subtree_com import compute_subtree_com
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.mass_matrix import compute_mass_matrix
from noeira.physics3d.dynamics.ldl import ldl_factor, ldl_solve, compute_m_inv
from noeira.physics3d.dynamics.rne import compute_bias_forces_rne
from noeira.physics3d.integrator.euler import (
    _armature_kernel, _fnet_passive_kernel, _qacc_writeback_kernel,
    _armature_env, _fnet_passive_env, _qacc_writeback_env,
)
from noeira.physics3d.collision.broadphase_sap import detect_contacts_sap
from noeira.physics3d.solver.newton_solve import (
    solve_newton_blocked, solve_newton, NEWTON_FORCE_PER_ENV,
)
from std.sys import default_accelerator
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, META_IDX_NEWTON_ITER, META_IDX_LS_EVAL,
    METADATA_SIZE, MODEL_JOINT_SIZE, CONTACT_SIZE, CONTACT_IDX_CONDIM,
    CONTACT_IDX_DIST, CONTACT_IDX_FORCE_N, CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2, CONTACT_IDX_FORCE_TORSION,
    MODEL_META_IDX_SOLVER_ITERATIONS,
)
from noeira.envs.libero.models.libero_living_room_scene3_xml import (
    LiberoLivingRoomScene3Model,
)


comptime DTYPE = DType.float32
comptime BATCH = 2
"""⚠⚠ TWO LANES, AND THAT IS DELIBERATE. The blocked kernel is one BLOCK PER
ENV; a per-env indexing mistake cannot show at BATCH 1, which is exactly the
shape a single-lane harness would call "bit-identical" forever. The two lanes
are filled with DIFFERENT dumped states."""
comptime M = LiberoLivingRoomScene3Model
"""⚠ ONE FAMILY PER BUILD — `sed` this line, as in the other family tools."""
comptime MC = 56
comptime NQ = M.NQ
comptime NV = M.NV
comptime NJOINT = M.NJOINT
comptime MDIMS = Dims[
    nq=M.NQ, nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM,
    nsite=M.NSITE, max_contacts=MC, nequality=M.MAX_EQUALITY,
    ntendon=M.MAX_TENDON, nexclude=M.NEXCLUDE, nmesh_verts=65536,
    npair=M.NPAIR, nact=M.NACT, nten=M.NTEN_F, nkey=M.NKEY,
]
comptime JE_WS = je_ws_size[
    DTYPE, MDIMS.NV, MDIMS.NJOINT, MDIMS.NTENDON, MDIMS.NEQUALITY,
    MDIMS.MAX_CONTACTS, M.MAX_CONDIM, CONE_TYPE=ConeType.ELLIPTIC,
]()

comptime DTYPE64 = DType.float64
"""⚠⚠ THE THIRD COLUMN. Two float32 legs that disagree by an ULP cannot say
which one is RIGHT — non-associativity alone produces that, and a cooperative
reduction is entitled to a different rounding from a serial loop. The CPU
float64 solve is the reference both are scored against: if the two float32
distances to it are the same size, there is no defect to fix and the rollout
divergence is chaos amplifying a legitimate rounding difference; if the
blocked leg is the far one, it is wrong."""
comptime JE_WS64 = je_ws_size[
    DTYPE64, MDIMS.NV, MDIMS.NJOINT, MDIMS.NTENDON, MDIMS.NEQUALITY,
    MDIMS.MAX_CONTACTS, M.MAX_CONDIM, CONE_TYPE=ConeType.ELLIPTIC,
]()

comptime F64_GPU_SWEEP: Bool = False
"""⚠⚠ THE TEST THAT SEPARATES ROUNDING FROM LOGIC. `sed` this to True ON A
CUDA BOX. Off, the float64 GPU kernels are never instantiated, which is what
keeps this file building on Metal — Metal has no float64 at all.

⚠ DO NOT FLIP `DTYPE` TO float64 TO GET THIS. That drags the whole GPU prep to
float64 and does not compile: `cos` has no float64 lowering on NVIDIA, so
`gpu_axis_angle_to_quat` inside `forward_kinematics` fails with "DType.float64
is not supported for cos on NVIDIA GPU". The arm below runs the PREP ON THE CPU
at float64 (`_prep64`) and puts only the two SOLVES on the device, which is the
only part under test.

⚠⚠ AND IT NEEDS `NEWTON_FORCE_PER_ENV = True` IN `newton_solve.mojo`. On NVIDIA
`solve_newton` DISPATCHES TO `solve_newton_blocked` (see its routing at
`newton_solve.mojo:4396`), so without that flag both arms are the blocked
kernel and the sweep is an identity — the same vacuity that made the CPU
float64 sweep worthless. `_solve64_gpu` REFUSES to run rather than print it."""


struct Solved(Movable):
    var qacc: List[Float64]
    var forces: List[Float64]
    var dist: List[Float64]
    var condim: List[Int]
    var ncon: Int
    var iters: Int
    var lsev: Int
    var per_lane_ncon: List[Int]
    var per_lane_iters: List[Int]
    var per_lane_lsev: List[Int]

    def __init__(out self):
        self.qacc = List[Float64]()
        self.forces = List[Float64]()
        self.dist = List[Float64]()
        self.condim = List[Int]()
        self.ncon = 0
        self.iters = 0
        self.lsev = 0
        self.per_lane_ncon = List[Int]()
        self.per_lane_iters = List[Int]()
        self.per_lane_lsev = List[Int]()


def _prep(
    mut d: Data[DTYPE, MDIMS, BATCH],
    mut mf: Model[DTYPE, MDIMS],
    mut scratch: DynamicsScratch[DTYPE, MDIMS, BATCH],
    ctx: DeviceContext,
) raises:
    """Smooth dynamics + detection up to the constraint seam — `test_newton_
    blocked_elliptic._prep`'s GPU branch, so both files feed the solve the
    same way. ⚠ SAP, not the O(N^2) loop: it is what the batched env runs."""
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_QPOS = Layout.row_major(BATCH, NQ)
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_cdof["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    compute_mass_matrix["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    ctx.enqueue_function[_armature_kernel[DTYPE, NV, NJOINT, BATCH]](
        mf.joints.lt["gpu", L_JOINT](), scratch.M.lt["gpu", L_M](),
        grid_dim=(BATCH,), block_dim=(1,),
    )
    ldl_factor["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
    compute_m_inv["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
    compute_bias_forces_rne["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    ctx.enqueue_function[_fnet_passive_kernel[DTYPE, NQ, NV, NJOINT, BATCH]](
        d.qpos.lt["gpu", L_QPOS](), d.qvel.lt["gpu", L_NV](),
        d.qfrc.lt["gpu", L_NV](), mf.joints.lt["gpu", L_JOINT](),
        scratch.bias.lt["gpu", L_NV](), scratch.fnet.lt["gpu", L_NV](),
        grid_dim=(BATCH,), block_dim=(1,),
    )
    ldl_solve["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
    ctx.enqueue_function[_qacc_writeback_kernel[DTYPE, NV, BATCH]](
        scratch.qacc_ws.lt["gpu", L_NV](), d.qacc.lt["gpu", L_NV](),
        scratch.qacc_constrained.lt["gpu", L_NV](),
        grid_dim=(BATCH,), block_dim=(1,),
    )
    detect_contacts_sap["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)


def _prep64(
    mut d: Data[DTYPE64, MDIMS, BATCH],
    mut mf: Model[DTYPE64, MDIMS],
    mut scratch: DynamicsScratch[DTYPE64, MDIMS, BATCH],
) raises:
    """`_prep`'s CPU float64 twin — `test_newton_blocked_elliptic._prep`'s
    `"cpu"` branch, which is where this shape comes from."""
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_QPOS = Layout.row_major(BATCH, NQ)
    var none = Optional[DeviceContext](None)
    forward_kinematics["cpu", DTYPE64, BATCH=BATCH](d, mf, none)
    compute_body_velocities["cpu", DTYPE64, BATCH=BATCH](d, mf, none)
    compute_subtree_com["cpu", DTYPE64, BATCH=BATCH](d, mf, none)
    compute_cdof["cpu", DTYPE64, BATCH=BATCH](d, mf, scratch, none)
    compute_mass_matrix["cpu", DTYPE64, BATCH=BATCH](d, mf, scratch, none)
    var joints_v = mf.joints.lt["cpu", L_JOINT]()
    var M_v = scratch.M.lt["cpu", L_M]()
    for e in range(BATCH):
        _armature_env[DTYPE64](e, AsStatic[MDIMS](), joints_v, M_v)
    ldl_factor["cpu", DTYPE64, BATCH=BATCH](mf, scratch, none)
    compute_m_inv["cpu", DTYPE64, BATCH=BATCH](mf, scratch, none)
    compute_bias_forces_rne["cpu", DTYPE64, BATCH=BATCH](d, mf, scratch, none)
    var qpos_v = d.qpos.lt["cpu", L_QPOS]()
    var qvel_v = d.qvel.lt["cpu", L_NV]()
    var qfrc_v = d.qfrc.lt["cpu", L_NV]()
    var bias_v = scratch.bias.lt["cpu", L_NV]()
    var fnet_v = scratch.fnet.lt["cpu", L_NV]()
    for e in range(BATCH):
        _fnet_passive_env[DTYPE64](
            e, AsStatic[MDIMS](), qpos_v, qvel_v, qfrc_v, joints_v, bias_v,
            fnet_v,
        )
    ldl_solve["cpu", DTYPE64, BATCH=BATCH](mf, scratch, none)
    var qacc_ws_v = scratch.qacc_ws.lt["cpu", L_NV]()
    var qacc_v = d.qacc.lt["cpu", L_NV]()
    var qacc_c_v = scratch.qacc_constrained.lt["cpu", L_NV]()
    for e in range(BATCH):
        _qacc_writeback_env[DTYPE64](
            e, AsStatic[MDIMS](), qacc_ws_v, qacc_v, qacc_c_v
        )
    detect_contacts_sap["cpu", DTYPE64, BATCH=BATCH](d, mf, none)


def _solve64(
    ctx: DeviceContext, q: List[Float64], v: List[Float64],
    blocked: Bool = False, niter_cap: Int = -1,
) raises -> Solved:
    """The reference: the same state, the same pipeline, at float64 on the CPU.

    ⚠ ONE LEG ONLY. At float64 the two legs' rounding difference is ~1e-16 and
    the question the reference answers is which float32 leg it is nearer to,
    so a second float64 arm would add a column that cannot disagree."""
    var mf = Model[DTYPE64, MDIMS]()
    # ⚠ THE CONTEXT IS STILL REQUIRED: `init_fields` builds the model and
    # uploads it whatever the solve's target is. Nothing below reads the
    # device copy.
    M.init_fields[DTYPE64](ctx, mf)
    if niter_cap > 0:
        mf.meta.data[MODEL_META_IDX_SOLVER_ITERATIONS] = Scalar[DTYPE64](
            niter_cap
        )
    var d = Data[DTYPE64, MDIMS, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = Scalar[DTYPE64](q[e * NQ + i])
        for i in range(NV):
            d.qvel.data[e * NV + i] = Scalar[DTYPE64](v[e * NV + i])
            d.qfrc.data[e * NV + i] = 0
    var scratch = DynamicsScratch[DTYPE64, MDIMS, BATCH]()
    var cscratch = ContactScratch[DTYPE64, MDIMS, BATCH, JE_WS64]()
    _prep64(d, mf, scratch)
    # ⚠⚠ THE SAME TWO LEGS, AT FLOAT64, ON THE CPU. This is what separates
    # a rounding difference from a logic one: if the two legs' float32 gap is
    # assembly order (`ell_add_contact_hessian`'s two-stage `JH` against
    # `_ell_entry_contact_term`'s per-entry recompute, or any other
    # non-associativity), it collapses to ~1e-16 here. If it SURVIVES at
    # float64, the two legs are computing different things and the gap is a
    # defect in one of them.
    # ⚠⚠ `blocked` IS REFUSED, NOT HONOURED. On CPU `solve_newton_blocked`
    # dispatches to `_newton_solve_env` — the per-env body — so a "blocked"
    # float64 arm here is the per-env arm under another name, and comparing
    # the two printed a perfect 0.0 at every iteration. The parameter is kept
    # so the refusal is visible to the next caller who reaches for it.
    if blocked:
        raise Error(
            "solve_at_pose: there is no CPU blocked leg — "
            "`solve_newton_blocked[\"cpu\"]` falls back to the per-env body,"
            " so a CPU float64 leg-vs-leg comparison is vacuous. Run the"
            " float64 sweep on CUDA, where the blocked GPU kernel is real."
        )
    solve_newton[
        "cpu", DTYPE64, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH,
        MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER=0, JE_WS=JE_WS64,
    ](d, mf, scratch, cscratch, Optional[DeviceContext](None))
    var out = Solved()
    for i in range(BATCH * NV):
        out.qacc.append(Float64(scratch.qacc_constrained.data[i]))
    for e in range(BATCH):
        var ne = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        if ne > MC:
            ne = MC
        out.per_lane_ncon.append(ne)
        out.per_lane_iters.append(
            Int(d.meta.data[e * METADATA_SIZE + META_IDX_NEWTON_ITER])
        )
        out.per_lane_lsev.append(
            Int(d.meta.data[e * METADATA_SIZE + META_IDX_LS_EVAL])
        )
        out.ncon += ne
    out.iters = out.per_lane_iters[0]
    out.lsev = out.per_lane_lsev[0]
    return out^


def _solve64_gpu(
    ctx: DeviceContext, q: List[Float64], v: List[Float64], blocked: Bool,
    niter_cap: Int = -1,
) raises -> Solved:
    """The two GPU elliptic kernels at FLOAT64, prep on the CPU. CUDA only.

    A rounding difference between the legs shrinks with the precision; a
    different computation does not. This is the arm that decides which the
    0.975-systematic float32 offset is.

    ⚠ TWO WARNINGS ON METAL ARE EXPECTED AND CORRECT, do not "fix" them:
    `NEWTON_FORCE_PER_ENV` is a comptime False there, so `if not
    NEWTON_FORCE_PER_ENV` is always True and everything below it is dead —
    which is the intent. They disappear in the CUDA build that sets the flag."""
    if not default_accelerator().is_nvidia_gpu():
        raise Error(
            "solve_at_pose: the float64 GPU sweep is CUDA-only — Metal has no"
            " float64. Leave F64_GPU_SWEEP at False on this machine."
        )
    if not NEWTON_FORCE_PER_ENV:
        # ⚠⚠ THE ANTI-VACUITY GATE. Without the flag `solve_newton` routes to
        # `solve_newton_blocked` on NVIDIA and both arms are ONE kernel — which
        # prints a perfect 0.0 at every iteration and means nothing. This is
        # exactly how the CPU float64 sweep fooled me; it refuses instead.
        raise Error(
            "solve_at_pose: NEWTON_FORCE_PER_ENV is False, so on NVIDIA"
            " `solve_newton` dispatches to the BLOCKED kernel and both arms of"
            " this sweep would be the same kernel — a vacuous 0.0 at every"
            " iteration. Set `comptime NEWTON_FORCE_PER_ENV: Bool = True` in"
            " noeira/physics3d/solver/newton_solve.mojo and rebuild."
        )
    var mf = Model[DTYPE64, MDIMS]()
    M.init_fields[DTYPE64](ctx, mf)
    if niter_cap > 0:
        mf.meta.data[MODEL_META_IDX_SOLVER_ITERATIONS] = Scalar[DTYPE64](
            niter_cap
        )
    var d = Data[DTYPE64, MDIMS, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = Scalar[DTYPE64](q[e * NQ + i])
        for i in range(NV):
            d.qvel.data[e * NV + i] = Scalar[DTYPE64](v[e * NV + i])
            d.qfrc.data[e * NV + i] = 0
    var scratch = DynamicsScratch[DTYPE64, MDIMS, BATCH]()
    var cscratch = ContactScratch[DTYPE64, MDIMS, BATCH, JE_WS64]()
    # ⚠ THE PREP IS THE CPU'S, and that is the point: `forward_kinematics` at
    # float64 on NVIDIA does not compile (`cos`). Both solves then read the
    # SAME float64 smooth dynamics and contact set.
    _prep64(d, mf, scratch)
    mf.upload_all(ctx)
    d.upload_all(ctx)
    scratch.upload_all(ctx)
    cscratch.upload_all(ctx)
    if blocked:
        solve_newton_blocked[
            "gpu", DTYPE64, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH,
            MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER=0, JE_WS=JE_WS64,
        ](d, mf, scratch, cscratch, ctx)
    else:
        solve_newton[
            "gpu", DTYPE64, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH,
            MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER=0, JE_WS=JE_WS64,
        ](d, mf, scratch, cscratch, ctx)
    scratch.qacc_constrained.download(ctx)
    d.meta.download(ctx)
    ctx.synchronize()
    var out = Solved()
    for i in range(BATCH * NV):
        out.qacc.append(Float64(scratch.qacc_constrained.data[i]))
    for e in range(BATCH):
        var ne = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        if ne > MC:
            ne = MC
        out.per_lane_ncon.append(ne)
        out.per_lane_iters.append(
            Int(d.meta.data[e * METADATA_SIZE + META_IDX_NEWTON_ITER])
        )
        out.per_lane_lsev.append(
            Int(d.meta.data[e * METADATA_SIZE + META_IDX_LS_EVAL])
        )
        out.ncon += ne
    out.iters = out.per_lane_iters[0]
    out.lsev = out.per_lane_lsev[0]
    return out^


def _solve(
    ctx: DeviceContext, q: List[Float64], v: List[Float64], blocked: Bool,
    niter_cap: Int = -1,
) raises -> Solved:
    """`q`/`v` hold BATCH lanes' words, lane-major.

    `niter_cap > 0` overrides the model's Newton budget, which is how the
    sweep in `main` localises the divergence to ONE iteration. ⚠ IT IS THE
    MODEL'S WORD, NOT A KERNEL KNOB: both legs read
    `MODEL_META_IDX_SOLVER_ITERATIONS` at the same place, so capping it is the
    only way to stop both at the same point without touching either kernel —
    and an in-kernel `print` is not available here, it takes the per-env
    elliptic kernel over Metal's per-thread stack ("Compute function exceeds
    available stack space")."""
    var mf = Model[DTYPE, MDIMS]()
    M.init_fields[DTYPE](ctx, mf)
    if niter_cap > 0:
        mf.meta.data[MODEL_META_IDX_SOLVER_ITERATIONS] = Scalar[DTYPE](
            niter_cap
        )
        mf.meta.upload(ctx)
    # ⚠ NO `reset_data`: it is a single-env helper (its `Data` parameter is
    # batch 1) and every word it would set is overwritten below or starts at
    # the zero a fresh `Data` already has.
    var d = Data[DTYPE, MDIMS, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = Scalar[DTYPE](q[e * NQ + i])
        for i in range(NV):
            d.qvel.data[e * NV + i] = Scalar[DTYPE](v[e * NV + i])
            d.qfrc.data[e * NV + i] = 0
    var scratch = DynamicsScratch[DTYPE, MDIMS, BATCH]()
    var cscratch = ContactScratch[DTYPE, MDIMS, BATCH, JE_WS]()
    d.upload_all(ctx)
    scratch.upload_all(ctx)
    cscratch.upload_all(ctx)
    _prep(d, mf, scratch, ctx)
    if blocked:
        solve_newton_blocked[
            "gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH,
            MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER=0, JE_WS=JE_WS,
        ](d, mf, scratch, cscratch, ctx)
    else:
        # ⚠ `solve_newton` is the PER-ENV leg on Metal and the BLOCKED one on
        # NVIDIA. On a CUDA box run this arm with NEWTON_FORCE_PER_ENV = True,
        # or both arms are the same kernel and the comparison is an identity.
        solve_newton[
            "gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH,
            MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER=0, JE_WS=JE_WS,
        ](d, mf, scratch, cscratch, ctx)
    scratch.qacc_constrained.download(ctx)
    d.meta.download(ctx)
    d.contacts.download(ctx)
    ctx.synchronize()

    var out = Solved()
    for i in range(BATCH * NV):
        out.qacc.append(Float64(scratch.qacc_constrained.data[i]))
    var n = 0
    for e in range(BATCH):
        var ne = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        if ne > MC:
            ne = MC
        out.per_lane_ncon.append(ne)
        out.per_lane_iters.append(
            Int(d.meta.data[e * METADATA_SIZE + META_IDX_NEWTON_ITER])
        )
        out.per_lane_lsev.append(
            Int(d.meta.data[e * METADATA_SIZE + META_IDX_LS_EVAL])
        )
        n += ne
    out.ncon = n
    out.iters = out.per_lane_iters[0]
    out.lsev = out.per_lane_lsev[0]
    for c in range(MC * BATCH):
        var cb = c * CONTACT_SIZE
        out.condim.append(Int(d.contacts.data[cb + CONTACT_IDX_CONDIM]))
        out.dist.append(Float64(d.contacts.data[cb + CONTACT_IDX_DIST]))
        out.forces.append(Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_N]))
        out.forces.append(Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_T1]))
        out.forces.append(Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_T2]))
        out.forces.append(
            Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_TORSION])
        )
    return out^


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error(
            "usage: solve_at_pose.mojo <dumped-states.txt> [lane]   (the dump"
            " comes from libero_family_batched --dump-at-step N --dump-state)\n"
            "       solve_at_pose.mojo <states.txt> <lane> <step>"
        )
    var lane_a = Int(String(a[2])) if len(a) > 2 else 0
    var want_step = Int(String(a[3])) if len(a) > 3 else -1
    var lane_b = Int(String(a[4])) if len(a) > 4 else lane_a + 1
    var text: String
    with open(String(a[1]), "r") as fh:
        text = fh.read()
    var q = List[Float64]()
    var v = List[Float64]()
    var lines = text.split("\n")
    var lanes = List[Int]()
    lanes.append(lane_a)
    lanes.append(lane_b)
    for li in range(BATCH):
        var got_q = False
        var got_v = False
        for i in range(len(lines)):
            var l = String(String(lines[i]).strip())
            if l.byte_length() == 0:
                continue
            var t = l.split(" ")
            if len(t) < 5:
                continue
            if Int(String(t[2])) != lanes[li]:
                continue
            if want_step >= 0 and Int(String(t[4])) != want_step:
                continue
            if l.startswith("QPOS") and not got_q:
                got_q = True
                for k in range(5, len(t)):
                    q.append(Float64(String(t[k])))
            elif l.startswith("QVEL") and not got_v:
                got_v = True
                for k in range(5, len(t)):
                    v.append(Float64(String(t[k])))
    if len(q) != NQ * BATCH or len(v) != NV * BATCH:
        raise Error(
            "the dump gave " + String(len(q)) + " qpos and " + String(len(v))
            + " qvel words for lanes " + String(lane_a) + "," + String(lane_b)
            + "; this model is nq " + String(NQ) + " nv " + String(NV)
            + " over " + String(BATCH) + " lanes"
        )
    print("=" * 78)
    print("one elliptic solve, both legs — lanes", lane_a, lane_b, "step",
          want_step, "| nq", NQ, "nv", NV, "| max_contacts", MC)
    print("=" * 78)
    var ctx = DeviceContext()
    comptime if F64_GPU_SWEEP:
        # ⚠⚠ THE FLOAT32 ARMS ARE NOT BUILT IN THIS CONFIGURATION, and that
        # is a COMPILE-TIME decision, not tidiness. Each elliptic Newton
        # instantiation is ~15 minutes on a rented 5090 (see
        # feedback_a_rented_gpu_box_prices_every_gate_in_compile_minutes).
        # With both flows live this file asks for FIVE of them — blocked and
        # per-env at float32, blocked and per-env at float64, and the per-env
        # CPU float64 reference — which is the 30+ minute build. `comptime if`
        # elaborates only the branch taken, so the float64 configuration pays
        # for TWO. The float32 numbers are already recorded in this header and
        # do not need re-measuring on the box to answer the float64 question.
        _f64_sweep(ctx, q, v)
    else:
        _f32_flow(ctx, q, v)


def _f64_sweep(
    ctx: DeviceContext, q: List[Float64], v: List[Float64]
) raises:
    """The CUDA-only float64 leg-vs-leg sweep — see `F64_GPU_SWEEP`."""
        # ⚠⚠ THE REAL SWEEP: BOTH GPU KERNELS AT FLOAT64, prep on the CPU.
        # `_solve64_gpu` refuses unless NEWTON_FORCE_PER_ENV is set, so the two
        # arms here cannot silently be one kernel.
        print("  --- rounding vs logic: BOTH GPU LEGS AT FLOAT64, capped at k ---")
        print("   k | per-env64 iters/lsev | blocked64 iters/lsev | worst |qacc| diff")
        var any_diff64 = False
        for k in range(1, 11):
            var p64 = _solve64_gpu(ctx, q, v, False, k)
            var b64 = _solve64_gpu(ctx, q, v, True, k)
            var w64 = 0.0
            for i in range(BATCH * NV):
                var dv = abs(p64.qacc[i] - b64.qacc[i])
                if dv > w64:
                    w64 = dv
            if w64 != 0.0:
                any_diff64 = True
            print("  ", k, "|", p64.per_lane_iters[0], "/",
                  p64.per_lane_lsev[0], "|", b64.per_lane_iters[0], "/",
                  b64.per_lane_lsev[0], "|", w64)
        # ⚠ AND THE SIGNED MEAN AT FLOAT64, uncapped — the same bias statistic
        # as the float32 block, so the two are read side by side.
        var pf = _solve64_gpu(ctx, q, v, False)
        var bf = _solve64_gpu(ctx, q, v, True)
        var s64 = 0.0
        var a64 = 0.0
        var w64f = 0.0
        for i in range(BATCH * NV):
            var e64 = bf.qacc[i] - pf.qacc[i]
            s64 += e64
            a64 += abs(e64)
            if abs(e64) > w64f:
                w64f = abs(e64)
        var n64 = Float64(BATCH * NV)
        print("   uncapped: worst |qacc| diff", w64f, " signed mean",
              s64 / n64, " mean |.|", a64 / n64, " ratio",
              abs(s64) / a64 if a64 > 0.0 else 0.0)

        # ⚠⚠ BIT-IDENTITY IS THE WRONG CRITERION AND THIS IS THE CORRECTION.
        # An earlier version of this verdict demanded `worst == 0.0` at float64
        # and called anything else a logic difference. That is wrong:
        # non-associativity does NOT vanish when the precision rises, it shrinks
        # WITH the precision — to about one ULP of the answer's own magnitude,
        # which at float64 on a 37 m/s^2 qacc is ~8e-15, not 0. Demanding zero
        # converts every rounding difference into a false "logic" verdict.
        #
        # The decisive quantity is the difference measured in ULPs of the
        # answer's scale, compared ACROSS the two precisions. Measured here:
        #     float32  5.722046e-06 / 37.22 = 1.29 ULP
        #     float64  7.105427e-15 / 37.22 = 0.86 ULP
        #     the diffs fell 8.05e8 while eps fell 5.37e8 — the same order.
        # A LOGIC difference keeps its RELATIVE size when eps shrinks; a
        # rounding one tracks eps. This tracks eps.
        var scale64 = 0.0
        for i in range(BATCH * NV):
            if abs(pf.qacc[i]) > scale64:
                scale64 = abs(pf.qacc[i])
        comptime EPS64 = 2.220446049250313e-16
        var ulps = (w64f / scale64 / EPS64) if scale64 > 0.0 else 0.0
        print("   scale (max |qacc|)", scale64, " => worst diff is", ulps,
              "ULP of float64")
        print()
        # A handful of ULPs is what a different summation order costs; a logic
        # difference would sit orders above it, at the float32 RELATIVE size
        # (1.5e-07 of scale, i.e. ~7e8 ULP of float64).
        if ulps <= 64.0:
            print("   => VERDICT: ROUNDING, not logic. The float64 gap is",
                  ulps, "ULP of the answer's own")
            print("      scale — the difference tracked machine epsilon down")
            print("      instead of holding its relative size, which is what a")
            print("      real logic difference would have done. The blocked")
            print("      kernel is SOUND: the float32 offset is assembly order")
            print("      or FMA contraction in one leg and not the other.")
            print("      What needs revisiting is the 1e-3 family-gate bound,")
            print("      which compares GPU float32 against CPU float64 and so")
            print("      measures float32 discretisation plus contact chaos —")
            print("      no kernel fix can meet it.")
        else:
            print("   => VERDICT: LOGIC, not rounding. The gap is", ulps,
                  "ULP of scale at float64, far")
            print("      above the couple of ULPs a summation order costs, so")
            print("      it held its relative size as eps shrank. One of the")
            print("      two kernels is wrong. Bisect the cone Hessian")
            print("      assembly next (two-stage `JH` in")
            print("      `ell_add_contact_hessian` against the per-entry")
            print("      recompute in `_ell_entry_contact_term`).")
        if any_diff64:
            print("   (the legs are not bit-identical at float64, which is")
            print("    EXPECTED and not evidence of anything on its own)")


def _f32_flow(
    ctx: DeviceContext, q: List[Float64], v: List[Float64]
) raises:
    """The float32 GPU legs, the cap sweep, the CPU float64 reference and the
    bias test — everything that does NOT need a float64 device kernel."""
    var per_env = _solve(ctx, q, v, False)
    var blocked = _solve(ctx, q, v, True)

    # ── THE ITERATION SWEEP ───────────────────────────────────────────────
    # ⚠ WHICH NEWTON ITERATION FIRST DISAGREES. Both legs converge here, so
    # the final `qacc` is one number and says nothing about where the two
    # paths parted. Capping the budget at k and comparing the k-th iterate
    # bisects that: the first k whose `qacc` differs is the iteration the
    # divergence is BORN in, and the line-search counts beside it say whether
    # the search took a different number of steps in that same iteration or
    # only reacted to an input that already differed.
    print()
    print("  --- capped at k Newton iterations (lane 0) ---")
    print("   k | per-env iters/lsev | blocked iters/lsev | worst |qacc| diff")
    for k in range(1, 11):
        var pe = _solve(ctx, q, v, False, k)
        var bl = _solve(ctx, q, v, True, k)
        var w = 0.0
        for i in range(BATCH * NV):
            var dv = abs(pe.qacc[i] - bl.qacc[i])
            if dv > w:
                w = dv
        print("  ", k, "|", pe.per_lane_iters[0], "/", pe.per_lane_lsev[0],
              "|", bl.per_lane_iters[0], "/", bl.per_lane_lsev[0], "|", w)
    print()

    for e in range(BATCH):
        print("  lane", e, "per-env: ncon", per_env.per_lane_ncon[e], "iters",
              per_env.per_lane_iters[e], "lsev", per_env.per_lane_lsev[e],
              "| blocked: ncon", blocked.per_lane_ncon[e], "iters",
              blocked.per_lane_iters[e], "lsev", blocked.per_lane_lsev[e])
    if per_env.ncon != blocked.ncon:
        # ⚠ THE SOLVE DOES NOT CHANGE THE CONTACT SET: a difference here is the
        # detection, run twice on the same words, and would mean the two arms
        # were not fed the same thing.
        print("  ⚠ THE INPUTS DIFFER — the contact sets are not the same;"
              " nothing below is a solver comparison")
    var worst_qacc = 0.0
    var worst_i = -1
    for i in range(BATCH * NV):
        var dv = abs(per_env.qacc[i] - blocked.qacc[i])
        if dv > worst_qacc:
            worst_qacc = dv
            worst_i = i
    print("  worst |qacc| difference", worst_qacc, "at dof", worst_i,
          "(lane", worst_i // NV, ")")
    var nf = len(per_env.forces) if len(per_env.forces) < len(blocked.forces) else len(blocked.forces)
    var worst_f = 0.0
    var worst_c = -1
    for i in range(nf):
        var dv = abs(per_env.forces[i] - blocked.forces[i])
        if dv > worst_f:
            worst_f = dv
            worst_c = i // 4
    print("  worst |contact force| difference", worst_f, "at contact", worst_c)
    if worst_c >= 0:
        print("     that contact: dist", per_env.dist[worst_c], " condim",
              per_env.condim[worst_c], " forces per-env",
              per_env.forces[worst_c * 4], per_env.forces[worst_c * 4 + 1],
              per_env.forces[worst_c * 4 + 2], per_env.forces[worst_c * 4 + 3],
              " blocked", blocked.forces[worst_c * 4],
              blocked.forces[worst_c * 4 + 1], blocked.forces[worst_c * 4 + 2],
              blocked.forces[worst_c * 4 + 3])
    # ── THE REFERENCE ─────────────────────────────────────────────────────
    var ref64 = _solve64(ctx, q, v)

    # ── ⚠⚠ THE FLOAT64 LEG-vs-LEG SWEEP IS NOT AVAILABLE HERE, AND THE
    # VACUOUS VERSION OF IT IS KEPT AS A WARNING ─────────────────────────
    # The test that would separate ROUNDING from LOGIC is the same cap sweep
    # with both legs at float64: a rounding difference shrinks with the
    # precision, a different computation does not. It cannot be run on this
    # target.
    #
    # `solve_newton_blocked`'s own docstring: "Only the GPU (blocked) launch is
    # meaningful; the CPU branch falls back to the single-source per-env body
    # (`_newton_solve_env`)". So `solve_newton_blocked["cpu", ...]` IS
    # `solve_newton["cpu", ...]`, and a CPU float64 leg-vs-leg sweep compares a
    # function with ITSELF: it printed 0.0 at every k, which reads exactly like
    # the answer you want and proves nothing. The tell was in the counts —
    # 71/71 line-search evaluations where the GPU legs give 147/148; two real
    # implementations do not match evaluation for evaluation.
    #
    # Metal has no float64, so the blocked kernel cannot be run at float64 on
    # this machine at all. THE SWEEP NEEDS A CUDA BOX, where
    # `solve_newton_blocked["gpu", DType.float64]` is a real second
    # implementation. Until then the bias below is measured but NOT attributed.
    print()
    print("  --- rounding vs logic: NOT DECIDED ON THIS TARGET ---")
    print("   the blocked kernel has no CPU implementation (the CPU branch")
    print("   is the per-env body) and Metal has no float64, so the sweep")
    print("   that would separate them needs a CUDA box: set")
    print("   F64_GPU_SWEEP = True here and NEWTON_FORCE_PER_ENV = True in")
    print("   newton_solve.mojo.")
    print()
    print()
    print("  --- against the CPU float64 reference ---")
    print("   reference: ncon", ref64.per_lane_ncon[0], "iters",
          ref64.per_lane_iters[0], "lsev", ref64.per_lane_lsev[0])
    if ref64.per_lane_ncon[0] != per_env.per_lane_ncon[0]:
        # ⚠ A DIFFERENT CONTACT SET IS A DIFFERENT PROBLEM, and the distances
        # below would be comparing two answers to two questions.
        print("  ⚠ the reference sees a DIFFERENT contact set — the distances"
              " below are not a solver comparison")
    var d_pe = 0.0
    var d_bl = 0.0
    var scale = 0.0
    for i in range(BATCH * NV):
        var a = abs(per_env.qacc[i] - ref64.qacc[i])
        var b = abs(blocked.qacc[i] - ref64.qacc[i])
        if a > d_pe:
            d_pe = a
        if b > d_bl:
            d_bl = b
        if abs(ref64.qacc[i]) > scale:
            scale = abs(ref64.qacc[i])
    print("   worst |qacc - reference|:  per-env", d_pe, " blocked", d_bl)
    # ⚠ AN ABSOLUTE qacc DISTANCE IS NOT READABLE WITHOUT THE SCALE: these are
    # m/s^2 on a scene under gravity with an arm holding itself up, so the
    # reference's own largest component is what says whether 1e-2 is a wrong
    # answer or the last digit of a right one.
    print("   reference max |qacc|", scale, " => relative: per-env",
          d_pe / scale if scale > 0.0 else 0.0, " blocked",
          d_bl / scale if scale > 0.0 else 0.0)
    for e in range(BATCH):
        var lpe = 0.0
        var lbl = 0.0
        var lsc = 0.0
        for i in range(e * NV, (e + 1) * NV):
            var a = abs(per_env.qacc[i] - ref64.qacc[i])
            var b = abs(blocked.qacc[i] - ref64.qacc[i])
            if a > lpe:
                lpe = a
            if b > lbl:
                lbl = b
            if abs(ref64.qacc[i]) > lsc:
                lsc = abs(ref64.qacc[i])
        print("     lane", e, ": per-env", lpe, " blocked", lbl,
              " (max |qacc| ", lsc, ", lsev ref", ref64.per_lane_lsev[e], ")")
    if d_pe > 0.0 or d_bl > 0.0:
        var ratio = d_bl / d_pe if d_pe > 0.0 else 0.0
        print("   blocked / per-env =", ratio,
              "— near 1 means the two float32 legs are EQUALLY far from the"
              " truth, so neither is the worse ANSWER at this state")

    # ── THE BIAS TEST ─────────────────────────────────────────────────────
    # ⚠⚠ `worst |.|` CANNOT SEE A BIAS, and a bias is the only way a 5.7e-06
    # per-solve difference becomes 0.04 over 125 substeps. A leg whose error
    # is random has a SIGNED mean near zero and a mean |.| much larger; a leg
    # that consistently over- or under-supports has the two the same size.
    # The ratio |mean| / mean|.| is that comparison in one number: near 0 is
    # rounding, near 1 is a systematic offset that will integrate.
    var s_pe = 0.0
    var s_bl = 0.0
    var s_dif = 0.0
    var a_pe = 0.0
    var a_bl = 0.0
    var a_dif = 0.0
    for i in range(BATCH * NV):
        var ep = per_env.qacc[i] - ref64.qacc[i]
        var eb = blocked.qacc[i] - ref64.qacc[i]
        var ed = blocked.qacc[i] - per_env.qacc[i]
        s_pe += ep
        s_bl += eb
        s_dif += ed
        a_pe += abs(ep)
        a_bl += abs(eb)
        a_dif += abs(ed)
    var n = Float64(BATCH * NV)
    print("   signed mean error vs reference:  per-env", s_pe / n,
          " blocked", s_bl / n)
    print("   mean |error|      vs reference:  per-env", a_pe / n,
          " blocked", a_bl / n)
    if a_pe > 0.0 and a_bl > 0.0:
        print("   |mean|/mean|.| (0 = random, 1 = systematic):  per-env",
              abs(s_pe) / a_pe, " blocked", abs(s_bl) / a_bl)
    print("   blocked MINUS per-env: signed mean", s_dif / n, " mean |.|",
          a_dif / n,
          " ratio", abs(s_dif) / a_dif if a_dif > 0.0 else 0.0)

    print()
    if worst_qacc == 0.0 and worst_f == 0.0:
        print("=== the two legs agree BIT FOR BIT on this state ===")
    elif worst_qacc <= d_pe:
        # ⚠⚠ THE VERDICT IS A COMPARISON, NOT A THRESHOLD. The two legs
        # differing is not news; what matters is whether they differ by LESS
        # than they each differ from the truth. When they do, neither is the
        # worse ANSWER here — but that is a statement about this state's
        # qacc, NOT about the kernel: read the bias ratio above, and note
        # that whether a systematic offset is rounding or logic is decided by
        # the float64 sweep this target cannot run.
        print("=== neither leg is the worse ANSWER at this state: they differ"
              " by", worst_qacc)
        print("=== and are both", d_pe, "from the float64 reference.")
        print("=== The offset between them is ROUNDING, settled on CUDA"
              " 2026-09-18: at float64")
        print("=== it falls to 0.86 ULP of scale from 1.29 ULP at float32,"
              " tracking machine")
        print("=== epsilon instead of holding its relative size. See this"
              " file's header. ===")
    else:
        print("=== ⚠ ONE LEG IS OUT: the legs differ by", worst_qacc,
              "which EXCEEDS the", d_pe, "gap to the float64 reference ===")
