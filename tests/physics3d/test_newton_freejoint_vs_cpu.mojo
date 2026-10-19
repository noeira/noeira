"""NVIDIA validation: the fields Newton contact solve on FREE-JOINT models
(Ant + Humanoid) vs the trusted fields-CPU per-env oracle.

WHY: before the physics3d sunset we must be sure the PYRAMIDAL blocked Newton
solver (the one-env-per-block cooperative kernel that runs ONLY on NVIDIA for
free-joint models) computes correct physics. It was validated bit-exact vs
per-env only on Walker2D (slide/hinge). This closes the free-joint gap.

HOW: `solve_newton` routes PYRAMIDAL+NVIDIA -> BLOCKED, everything else
-> per-env. So:
  * on NVIDIA: solve_newton["gpu"] == BLOCKED (production path);
    solve_newton["cpu"] == per-env oracle (trusted).  Comparing them
    VALIDATES THE BLOCKED SOLVER against ground truth.
  * on Apple: solve_newton["gpu"] == per-env (blocked never launches);
    this just checks per-env GPU vs CPU (safe — no heavy cooperative kernel).
Same script, both meaningful. Gentle floor-contact poses (light penetration)
keep the Newton solve well-conditioned — a deep-penetration pose ill-conditions
it and can NaN/crash, which is a test artifact, not a solver bug.

Verdict: GPU-vs-CPU qacc_constrained relative error < 1e-2 => solver correct.

Run on NVIDIA: pixi run -e nvidia mojo run -I . tests/physics3d/test_newton_freejoint_vs_cpu.mojo
Run on Apple : pixi run -e apple  mojo run -I . tests/physics3d/test_newton_freejoint_vs_cpu.mojo
"""

from std.math import abs
from std.sys import default_accelerator
from max.gpu.host import DeviceContext
from layout import Layout

from noeira.physics3d.fields import (
    AsStatic,
    AsStatic,
    AsStatic,
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
 DimsLike,)
from noeira.physics3d.model.model_def import ModelDefLike
from noeira.tasks.so101_tabletop_xml import So101TabletopModel
from noeira.physics3d.types import ConeType
from noeira.physics3d.solver.je_budget import je_ws_size
from noeira.physics3d.integrator.euler import (
    _armature_kernel,
    _fnet_passive_kernel,
    _qacc_writeback_kernel,
    _armature_env,
    _fnet_passive_env,
    _qacc_writeback_env,
)
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from noeira.physics3d.dynamics.subtree_com import (
    compute_subtree_com,
)
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix,
)
from noeira.physics3d.dynamics.ldl import (
    ldl_factor,
    ldl_solve,
    compute_m_inv,
)
from noeira.physics3d.dynamics.rne import (
    compute_bias_forces_rne,
)
from noeira.physics3d.collision.contact_detection import (
    detect_contacts,
)
from noeira.physics3d.solver.newton_solve import (
    solve_newton, solve_newton_blocked,
)
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_NTREE,
    METADATA_SIZE,
    MODEL_JOINT_SIZE,
)
from noeira.envs.ant.ant_xml import AntModel
from noeira.envs.humanoid.humanoid_xml import HumanoidModel
from noeira.physics3d.parser import parse_xml, ModelDefFromXML

comptime DTYPE = DType.float32
comptime BATCH = 2
comptime CONE_T = ConeType.PYRAMIDAL  # forces the blocked branch on NVIDIA

# Set True ONLY when running on NVIDIA. Referencing solve_newton for
# humanoid forces Metal to compile the blocked-humanoid kernel, whose
# threadgroup memory exceeds Metal's 32KB limit -> Apple compile failure. The
# comptime guard below excludes it entirely when False, so Apple compiles
# (Ant-only). On NVIDIA the blocked-humanoid kernel is the production path.
comptime INCLUDE_HUMANOID = False


# ⚠⚠ A MULTI-TREE MODEL, AND WITHOUT ONE THIS FILE GATES NOTHING ABOUT THE
# BLOCK WORK. `Ant`, `Humanoid` and `Walker2d` — every model any GPU solver
# test uses — are SINGLE-TREE, so `build_dof_segments` returns one segment
# spanning `[0, nv)` and every loop restriction PN2c/d/e and P2 added is a
# NO-OP on them. They would all pass against a completely broken partition.
#
# Three trees here: a slider body and two free boxes. The XML places the boxes
# on the floor and TOUCHING EACH OTHER, so a contact row spans two trees and
# the segment MERGE path runs too, not just the trivial all-separate case.
comptime TREES_XML = """
<mujoco model="three trees">
  <option cone="pyramidal" timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="4 4 1"/>
    <body name="slider" pos="0 0 0.30">
      <joint name="sx" type="slide" axis="1 0 0"/>
      <joint name="sz" type="slide" axis="0 0 1"/>
      <geom name="gs" type="sphere" size="0.05"/>
    </body>
    <body name="boxA" pos="0.60 0 0.05">
      <freejoint/>
      <geom name="ga" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="boxB" pos="0.69 0 0.05">
      <freejoint/>
      <geom name="gb" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime _tp = parse_xml(TREES_XML)
comptime ThreeTreesModel = ModelDefFromXML[
    xml=TREES_XML,
    nbody=_tp.NBODY, njoint=_tp.NJOINT, nq=_tp.NQ, nv=_tp.NV,
    ngeom=_tp.NGEOM, nact=_tp.NACT, ntex=_tp.NTEX, nmat=_tp.NMAT,
    nlight=_tp.NLIGHT, ncam=_tp.NCAM, nsite=_tp.NSITE,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=4,
    obs_qpos_skip=0,
    timestep=_tp.TIMESTEP,
]


def _prep[target: StaticString, D: DimsLike](
    mut d: Data[DTYPE, D, BATCH],
    mut mf: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth-dynamics prep + auto detection (mirrors the integrator seam)."""
    forward_kinematics[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_cdof[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    compute_mass_matrix[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)

    comptime L_JOINT = Layout.row_major(D.NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(BATCH, D.NV * D.NV)
    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_QPOS = Layout.row_major(BATCH, D.NQ)

    comptime if target == "cpu":
        var joints_v = mf.joints.lt["cpu", L_JOINT]()
        var M_v = scratch.M.lt["cpu", L_M]()
        for e in range(BATCH):
            _armature_env[DTYPE](e, AsStatic[D](), joints_v, M_v)
        ldl_factor["cpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv["cpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne["cpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var qfrc_v = d.qfrc.lt["cpu", L_NV]()
        var bias_v = scratch.bias.lt["cpu", L_NV]()
        var fnet_v = scratch.fnet.lt["cpu", L_NV]()
        for e in range(BATCH):
            _fnet_passive_env[DTYPE](
                e, AsStatic[D](), qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v
            )
        ldl_solve["cpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
        var qacc_ws_v = scratch.qacc_ws.lt["cpu", L_NV]()
        var qacc_v = d.qacc.lt["cpu", L_NV]()
        var qacc_c_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        for e in range(BATCH):
            _qacc_writeback_env[DTYPE](
                e, AsStatic[D](), qacc_ws_v, qacc_v, qacc_c_v
            )
    else:
        ctx.value().enqueue_function[_armature_kernel[DTYPE, D.NV, D.NJOINT, BATCH]](
            mf.joints.lt["gpu", L_JOINT](),
            scratch.M.lt["gpu", L_M](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_factor["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        ctx.value().enqueue_function[
            _fnet_passive_kernel[DTYPE, D.NQ, D.NV, D.NJOINT, BATCH]
        ](
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            d.qfrc.lt["gpu", L_NV](),
            mf.joints.lt["gpu", L_JOINT](),
            scratch.bias.lt["gpu", L_NV](),
            scratch.fnet.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_solve["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
        ctx.value().enqueue_function[_qacc_writeback_kernel[DTYPE, D.NV, BATCH]](
            scratch.qacc_ws.lt["gpu", L_NV](),
            d.qacc.lt["gpu", L_NV](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    detect_contacts[target, DTYPE, BATCH=BATCH](d, mf, ctx)


def _validate[
    MODEL: ModelDefLike,
    # ⚠⚠ FORCE THE BLOCKED KERNEL, WHICH `solve_newton` WOULD NOT LAUNCH HERE.
    # Its routing is `if default_accelerator().is_nvidia_gpu()` — a RUNTIME test, so the
    # blocked kernel is COMPILED on Metal and simply never reached. That left a
    # hole exactly where the block work lives: the only multi-tree model in the
    # tree reached the blocked kernel on NVIDIA and nowhere else, so a change to
    # its per-block code could only be gated on the 5090. Calling
    # `solve_newton_blocked` directly closes it — Metal runs the same kernel.
    FORCE_BLOCKED: Bool = False,
    # Mesh-collision capacity. 0 for the hand-written fixtures; a real robot
    # model needs its hull vertices or `init_fields` raises.
    NMESH_VERTS: Int = 0,
](
    ctx: DeviceContext,
    name: String,
    torso_z: Float64,
    min_trees: Int = 1,
) raises -> Bool:
    """Returns True if GPU==CPU within tolerance (solver correct)."""
    # Dims as local comptime aliases OFF the model spec so the mf type
    # structurally matches init_fields' signature (a literal/explicit param
    # would not unify with MODEL.NV etc — the Stage-E threading pattern).
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime NBODY = MODEL.NBODY
    comptime NJOINT = MODEL.NJOINT
    comptime NGEOM = MODEL.NGEOM
    comptime MC = MODEL.MAX_CONTACTS
    comptime NEQ = MODEL.MAX_EQUALITY
    comptime NTEN = MODEL.MAX_TENDON
    comptime NSITE = MODEL.NSITE
    comptime NEXCL = MODEL.NEXCLUDE
    comptime MD = Dims[
        nq=NQ,
        nv=NV,
        nbody=NBODY,
        njoint=NJOINT,
        ngeom=NGEOM,
        nsite=NSITE,
        max_contacts=MC,
        nequality=NEQ,
        ntendon=NTEN,
        nexclude=NEXCL,
        nmesh_verts=NMESH_VERTS,
    ]
    print("--- ", name, " (NV=", NV, ") gentle floor contact ---")
    # Offset-free build straight from the compile-time model spec — no slab,
    # no init_model_gpu / load_from_slab.
    var mf = Model[DTYPE, MD]()
    MODEL.init_fields[DTYPE](ctx, mf)

    # ⚠⚠ THE ARM'S OWN PREMISE, CHECKED. `build_dof_segments` returns ONE
    # segment on a single-tree model, so every loop restriction PN2c/d/e and P2
    # added is a NO-OP there and this comparison would pass against a
    # completely broken partition. Printing `ntree` tells a reader which arms
    # actually exercise the block work; `min_trees` makes the multi-tree arm
    # FAIL rather than quietly degrade if its XML ever collapses to one tree.
    var ntree = Int(mf.meta.data[MODEL_META_IDX_NTREE])
    print("    kinematic trees:", ntree,
          "(1 = the block restriction is a no-op here)" if ntree <= 1 else
          "(multi-tree: the block restriction is LIVE)")
    if ntree < min_trees:
        print("    FAIL:", name, "has", ntree, "trees, needs >=", min_trees,
              "- this arm is VACUOUS for the block work")
        return False

    # Gentle pose: torso lowered so feet lightly touch (not deep penetration).
    var d_g = Data[DTYPE, MD, BATCH]()
    var d_c = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        d_g.qpos.data[e * NQ + 2] = Scalar[DTYPE](torso_z + 0.01 * Float64(e))
        d_g.qpos.data[e * NQ + 3] = Scalar[DTYPE](1.0)  # quat w
        d_c.qpos.data[e * NQ + 2] = Scalar[DTYPE](torso_z + 0.01 * Float64(e))
        d_c.qpos.data[e * NQ + 3] = Scalar[DTYPE](1.0)
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 40.0  # mild
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 8.0
            d_g.qvel.data[e * NV + i] = qv
            d_g.qfrc.data[e * NV + i] = qf
            d_c.qvel.data[e * NV + i] = qv
            d_c.qfrc.data[e * NV + i] = qf
    d_g.upload_all(ctx)

    # ⚠ SIZED FROM THE SPILL POLICY, AS THE INTEGRATORS DO. At the default
    # `JE_WS = 0` a model whose `Je` spills has the blocked kernel writing
    # its rows into a one-scalar buffer: SO101Tabletop read a relative error
    # of 722 the day the budget moved to 16 KB. `solve_newton_blocked` now
    # refuses the mismatch at compile time.
    comptime JE_WS = je_ws_size[
        DTYPE, MD.NV, MD.NJOINT, MD.NTENDON, MD.NEQUALITY, MD.MAX_CONTACTS, 3
    ]()
    var sg = DynamicsScratch[DTYPE, MD, BATCH]()
    var cg = ContactScratch[DTYPE, MD, BATCH, JE_WS]()
    sg.upload_all(ctx)
    cg.upload_all(ctx)
    var sc = DynamicsScratch[DTYPE, MD, BATCH]()
    var cc = ContactScratch[DTYPE, MD, BATCH, JE_WS]()

    # GPU path (blocked on NVIDIA, per-env on Apple).
    _prep["gpu"](
        d_g, mf, sg, ctx
    )
    comptime if FORCE_BLOCKED:
        solve_newton_blocked[
            "gpu", DTYPE, CONE_TYPE=CONE_T, BATCH=BATCH, JE_WS=JE_WS
        ](d_g, mf, sg, cg, ctx)
    else:
        solve_newton["gpu", DTYPE, CONE_TYPE=CONE_T, BATCH=BATCH](d_g, mf, sg, cg, ctx)

    # CPU oracle (per-env).
    _prep["cpu"](
        d_c, mf, sc, None
    )
    solve_newton["cpu", DTYPE, CONE_TYPE=CONE_T, BATCH=BATCH](d_c, mf, sc, cc, None)

    sg.qacc_constrained.download(ctx)
    d_g.meta.download(ctx)
    var ncon = 0
    for e in range(BATCH):
        ncon += Int(d_g.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    print("    contacts:", ncon, "(",
          "FORCED blocked" if FORCE_BLOCKED else
          ("NVIDIA=blocked" if default_accelerator().is_nvidia_gpu()
           else "Apple=per-env"), ")")
    if ncon == 0:
        print("    WARNING: 0 contacts (vacuous). Lower this model's torso_z\n"
              "    arg in main() by ~0.05 and re-run until contacts > 0.")

    var worst = Float64(0)
    for i in range(BATCH * NV):
        var g = Float64(sg.qacc_constrained.data[i])
        var c = Float64(sc.qacc_constrained.data[i])
        var err = abs(g - c) / (1.0 + abs(c))
        if err > worst:
            worst = err
    print("    qacc_constrained GPU-vs-CPU-oracle worst rel err:", worst)
    if worst < 1e-2:
        print("    PASS:", name, "GPU solver matches CPU oracle")
        return True
    print("    FAIL:", name, "GPU solver DIVERGES from CPU oracle")
    return False


def main() raises:
    print("=" * 66)
    print("Free-joint Newton solve: GPU (blocked on NVIDIA) vs CPU oracle")
    print("=" * 66)
    var ctx = DeviceContext()

    var all_ok = True

    # ── Ant (free joint, NV=14) ────────────────────────────────────────────
    all_ok = _validate[AntModel](ctx, "Ant", 0.28) and all_ok

    # ── ⚠ THE MULTI-TREE ARM. Every other model here is single-tree, so
    # without this one the file says nothing about the block restrictions.
    # `torso_z` is 0: the XML already places the bodies.
    all_ok = _validate[ThreeTreesModel](
        ctx, "ThreeTrees", 0.0, min_trees=3
    ) and all_ok

    # ── ⚠⚠ THE SAME MULTI-TREE MODEL, THROUGH THE BLOCKED KERNEL ITSELF.
    # The arm above runs per-env on Apple, so on this machine it gates the
    # partition but NOT the blocked kernel's per-block code — and F3b made that
    # code thread-parallel (one diagonal block per thread). With three trees
    # this is the arm where three threads solve three systems concurrently;
    # every other blocked model in the tree is single-tree, where the loop
    # degenerates to "thread 0 does block 0" and would pass whatever the
    # mapping did.
    all_ok = _validate[ThreeTreesModel, FORCE_BLOCKED=True](
        ctx, "ThreeTrees/blocked", 0.0, min_trees=3
    ) and all_ok

    # ── ⚠ AN INDEPENDENTLY-AUTHORED MULTI-TREE MODEL, and that is the point.
    # ThreeTrees is mine, so it can only fail in ways I thought to build into
    # it. This one comes from the task layer: 4 trees, real contacts against 30
    # collision MESHES rather than box-on-box, and `max_contacts = 32` where
    # ThreeTrees is 8 and the park scenes are 16 — so `ME = 4*MC + 2*NJOINT +
    # NV` moves on both axes at once.
    all_ok = _validate[
        So101TabletopModel, FORCE_BLOCKED=True, NMESH_VERTS=30000
    ](ctx, "SO101Tabletop/blocked", 0.0, min_trees=2) and all_ok

    # ── Humanoid (free joint, NV=23) — the production blocked model ─────────
    # On NVIDIA this exercises the exact blocked kernel humanoid training uses.
    # Comptime-excluded on Apple (blocked-humanoid overflows Metal's 32KB at
    # COMPILE time) — set INCLUDE_HUMANOID=True on the NVIDIA box.
    comptime if INCLUDE_HUMANOID:
        all_ok = _validate[HumanoidModel](ctx, "Humanoid", 1.0) and all_ok
    else:
        print("--- Humanoid: SKIPPED on Apple (NVIDIA-only blocked path) ---")

    print("=" * 66)
    if not all_ok:
        raise Error("free-joint Newton solver diverges from CPU oracle")
    print("ALL PASS: free-joint blocked/per-env solver == CPU oracle")
