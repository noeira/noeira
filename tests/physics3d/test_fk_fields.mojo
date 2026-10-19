"""Regression gate (GOLDEN-frozen): per-field single-source FK + the full
dynamics prep chain (subtree_com, cdof, CRBA mass matrix, LDL factor/solve, RNE
bias) on Walker2D, plus a synthetic sites model.

Originally validated BIT-EXACT against the legacy flat-slab kernels (and, for FK,
the legacy CPU forward_kinematics). Those legacy references were frozen into the
GOLDEN fingerprints below during Phase-0 of the physics3d sunset, so this gate
survives deletion of the legacy slab/kernels. It checks:
  * fields-CPU == fields-GPU at every chained stage (independent CPU oracle;
    the real correctness gate, legacy-free), and
  * combined order-sensitive checksums of the fields-GPU outputs reproduce the
    frozen (legacy-validated) fingerprint (Apple-gated).

A. Walker2D (NQ=9, NBODY=8), BATCH=3 distinct qpos configs.
B. Synthetic 2-body hinge model with NSITE=2 (covers the sites FK variant +
   site_xpos), BATCH=2 — records written DIRECTLY into the per-field tensors.

Model build is offset-free (Part A = init_fields; Part B = direct per-field
record writes) — no slab, no load_from_slab. Regenerate goldens after an
INTENTIONAL physics change: HARVEST=True, run on Apple, paste, HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_fk_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.joint_types import JNT_HINGE
from noeira.physics3d.fields import Data, Model, DynamicsScratch, Dims, DimsLike
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
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
)
from noeira.physics3d.dynamics.rne import (
    compute_bias_forces_rne,
)
from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_SITE_SIZE,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    BODY_IDX_PARENT,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_IPOS_Z,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_Y,
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Z,
)
from noeira.envs.walker2d.walker2d_xml import Walker2dModel
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float32
comptime POS_TOL: Float64 = 1e-4
comptime QUAT_TOL: Float64 = 1e-4

# ── Walker2D dims ──────────────────────────────────────────────────────────
comptime NQ = Walker2dModel.NQ  # 9
comptime NV = Walker2dModel.NV  # 9
comptime NBODY = Walker2dModel.NBODY  # 8
comptime NJOINT = Walker2dModel.NJOINT  # 9
comptime NGEOM = Walker2dModel.NGEOM  # 8
comptime MAX_CONTACTS = Walker2dModel.MAX_CONTACTS  # 20
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE
comptime MD = ModelDims[Walker2dModel]
comptime BATCH = 3

# ── Synthetic sites model dims ─────────────────────────────────────────────
comptime S_NQ = 1
comptime S_NV = 1
comptime S_NBODY = 2
comptime S_NJOINT = 1
comptime S_NSITE = 2
comptime S_MC = 1
comptime MD_2 = Dims[
    nq=S_NQ,
    nv=S_NV,
    nbody=S_NBODY,
    njoint=S_NJOINT,
    ngeom=0,
    nsite=S_NSITE,
    max_contacts=S_MC,
    nequality=0,
    ntendon=0,
]
comptime S_BATCH = 2

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_A = 3539856.921795104  # walker2d combined chained-output checksum
comptime GOLD_B = 16603.795657545328  # synthetic sites combined checksum


def _quat_err(
    ax: Float64, ay: Float64, az: Float64, aw: Float64,
    bx: Float64, by: Float64, bz: Float64, bw: Float64,
) -> Float64:
    var dp = abs(ax - bx) + abs(ay - by) + abs(az - bz) + abs(aw - bw)
    var dn = abs(ax + bx) + abs(ay + by) + abs(az + bz) + abs(aw + bw)
    return dp if dp < dn else dn


def _fold(mut acc: Float64, tag: Int, data: List[Scalar[DTYPE]], n: Int):
    for i in range(n):
        acc += Float64(data[i]) * Float64((i + 1) * 131 + tag)


def _gold_check(name: String, got: Float64, gold: Float64) raises:
    if HARVEST:
        print("  HARVEST", name, "=", got)
    else:
        var denom = abs(gold) if abs(gold) > 1e-9 else 1.0
        if abs(got - gold) / denom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                name + " fingerprint " + String(got) + " != golden "
                + String(gold)
            )
        print("  PASS:", name, "matches golden fingerprint")


def test_walker2d() raises:
    print("--- A. Walker2D fields FK + dynamics chain, BATCH=", BATCH, "---")
    var ctx = DeviceContext()

    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    var qcfg = List[List[Float64]]()
    var q1 = List[Float64](length=NQ, fill=0.0)
    q1[1] = 1.25
    qcfg.append(q1^)
    var q2 = List[Float64](length=NQ, fill=0.0)
    q2[1] = 1.25
    q2[3] = 0.5
    q2[4] = -0.8
    q2[5] = 0.3
    qcfg.append(q2^)
    var q3 = List[Float64](length=NQ, fill=0.0)
    q3[1] = 1.25
    q3[2] = 0.5
    q3[3] = 1.0
    q3[4] = -1.2
    q3[5] = 0.6
    q3[6] = -1.0
    q3[7] = 1.2
    q3[8] = -0.6
    qcfg.append(q3^)

    var d = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
            dc.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
    d.upload_all(ctx)

    # 1. FK.
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    d.download_all(ctx)
    forward_kinematics["cpu", DTYPE, BATCH=BATCH](dc, mf)
    var worst_fk = Float64(0)
    for e in range(BATCH):
        for b in range(NBODY):
            for k in range(3):
                var err = abs(
                    Float64(dc.xpos.data[e * NBODY * 3 + b * 3 + k])
                    - Float64(d.xpos.data[e * NBODY * 3 + b * 3 + k])
                )
                if err > worst_fk:
                    worst_fk = err
            var qe = _quat_err(
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 0]),
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 1]),
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 2]),
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 3]),
                Float64(d.xquat.data[e * NBODY * 4 + b * 4 + 0]),
                Float64(d.xquat.data[e * NBODY * 4 + b * 4 + 1]),
                Float64(d.xquat.data[e * NBODY * 4 + b * 4 + 2]),
                Float64(d.xquat.data[e * NBODY * 4 + b * 4 + 3]),
            )
            if qe > worst_fk:
                worst_fk = qe
    print("  FK fields-CPU vs fields-GPU worst err:", worst_fk)
    if worst_fk > QUAT_TOL:
        raise Error("walker2d FK fields-CPU tolerance exceeded")

    # 2. subtree_com.
    compute_subtree_com["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    d.subtree_com.download(ctx)
    compute_subtree_com["cpu", DTYPE, BATCH=BATCH](dc, mf)
    var worst_st = Float64(0)
    for i in range(BATCH * NBODY * 3):
        var err = abs(
            Float64(dc.subtree_com.data[i]) - Float64(d.subtree_com.data[i])
        )
        if err > worst_st:
            worst_st = err
    print("  subtree_com fields-CPU vs fields-GPU worst err:", worst_st)
    if worst_st > POS_TOL:
        raise Error("walker2d subtree_com fields-CPU tolerance exceeded")

    # 3. cdof.
    var scratch = DynamicsScratch[DTYPE, MD, BATCH]()
    scratch.upload_all(ctx)
    var scratch_c = DynamicsScratch[DTYPE, MD, BATCH]()
    compute_cdof["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    scratch.cdof.download(ctx)
    compute_cdof["cpu", DTYPE, BATCH=BATCH](dc, mf, scratch_c)
    var worst_cd = Float64(0)
    for i in range(BATCH * NV * 6):
        var err = abs(
            Float64(scratch_c.cdof.data[i]) - Float64(scratch.cdof.data[i])
        )
        if err > worst_cd:
            worst_cd = err
    print("  cdof fields-CPU vs fields-GPU worst err:", worst_cd)
    if worst_cd > POS_TOL:
        raise Error("walker2d cdof fields-CPU tolerance exceeded")

    # 4. CRBA mass matrix.
    compute_mass_matrix["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    scratch.M.download(ctx)
    compute_mass_matrix["cpu", DTYPE, BATCH=BATCH](dc, mf, scratch_c)
    var worst_mm = Float64(0)
    for i in range(BATCH * NV * NV):
        var err = abs(Float64(scratch_c.M.data[i]) - Float64(scratch.M.data[i]))
        if err > worst_mm:
            worst_mm = err
    print("  mass matrix fields-CPU vs fields-GPU worst err:", worst_mm)
    if worst_mm > 1e-3:
        raise Error("walker2d mass matrix fields-CPU tolerance exceeded")

    # 5. LDL factor + solve (synthetic per-DOF fnet).
    for e in range(BATCH):
        for i in range(NV):
            var f = Scalar[DTYPE]((e * 31 + i * 7) % 11 - 5) / 3.0
            scratch.fnet.data[e * NV + i] = f
            scratch_c.fnet.data[e * NV + i] = f
    scratch.fnet.upload(ctx)
    ldl_factor["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
    ldl_solve["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
    scratch.qacc_ws.download(ctx)
    ldl_factor["cpu", DTYPE, BATCH=BATCH](mf, scratch_c)
    ldl_solve["cpu", DTYPE, BATCH=BATCH](mf, scratch_c)
    var worst_ld = Float64(0)
    for i in range(BATCH * NV):
        var err = abs(
            Float64(scratch_c.qacc_ws.data[i])
            - Float64(scratch.qacc_ws.data[i])
        )
        if err > worst_ld:
            worst_ld = err
    print("  LDL qacc fields-CPU vs fields-GPU worst err:", worst_ld)
    if worst_ld > 1e-3:
        raise Error("walker2d LDL fields-CPU tolerance exceeded")

    # 6. RNE bias forces (synthetic qvel).
    for e in range(BATCH):
        for i in range(NV):
            var v = Scalar[DTYPE]((e * 17 + i * 13) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = v
            dc.qvel.data[e * NV + i] = v
    d.qvel.upload(ctx)
    compute_bias_forces_rne["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    scratch.bias.download(ctx)
    compute_bias_forces_rne["cpu", DTYPE, BATCH=BATCH](dc, mf, scratch_c)
    var worst_rne = Float64(0)
    for i in range(BATCH * NV):
        var err = abs(
            Float64(scratch_c.bias.data[i]) - Float64(scratch.bias.data[i])
        )
        if err > worst_rne:
            worst_rne = err
    print("  RNE bias fields-CPU vs fields-GPU worst err:", worst_rne)
    if worst_rne > 1e-3:
        raise Error("walker2d RNE fields-CPU tolerance exceeded")
    print("  PASS: all stages fields-CPU within tolerance of fields-GPU")

    # --- combined golden fingerprint of the fields-GPU chained outputs ---
    var fp = Float64(0)
    _fold(fp, 1, d.xpos.data, BATCH * NBODY * 3)
    _fold(fp, 2, d.xquat.data, BATCH * NBODY * 4)
    _fold(fp, 3, d.subtree_com.data, BATCH * NBODY * 3)
    _fold(fp, 4, scratch.cdof.data, BATCH * NV * 6)
    _fold(fp, 5, scratch.M.data, BATCH * NV * NV)
    _fold(fp, 6, scratch.qacc_ws.data, BATCH * NV)
    _fold(fp, 7, scratch.bias.data, BATCH * NV)
    _gold_check("walker2d chain", fp, GOLD_A)


def test_synthetic_sites() raises:
    print("--- B. Synthetic hinge model with NSITE=2, BATCH=", S_BATCH, "---")
    var ctx = DeviceContext()

    # Hand-built fields model, records written DIRECTLY into the per-field
    # tensors (no offset slab, no load_from_slab): worldbody + 1 hinge body,
    # 2 sites. Each record family packs as `record_idx * MODEL_<KIND>_SIZE +
    # <KIND>_IDX_*`; meta is a standalone tensor.
    var mf = Model[DTYPE, MD_2]()
    mf.bodies.data[0 * MODEL_BODY_SIZE + BODY_IDX_QUAT_W] = 1.0
    var b1 = 1 * MODEL_BODY_SIZE
    mf.bodies.data[b1 + BODY_IDX_PARENT] = 0.0
    mf.bodies.data[b1 + BODY_IDX_POS_X] = 0.1
    mf.bodies.data[b1 + BODY_IDX_POS_Y] = 0.2
    mf.bodies.data[b1 + BODY_IDX_POS_Z] = 1.0
    mf.bodies.data[b1 + BODY_IDX_QUAT_W] = 1.0
    mf.bodies.data[b1 + BODY_IDX_IPOS_Z] = -0.25
    var j0 = 0 * MODEL_JOINT_SIZE
    mf.joints.data[j0 + JOINT_IDX_TYPE] = Scalar[DTYPE](JNT_HINGE)
    mf.joints.data[j0 + JOINT_IDX_BODY_ID] = 1.0
    mf.joints.data[j0 + JOINT_IDX_QPOS_ADR] = 0.0
    mf.joints.data[j0 + JOINT_IDX_POS_Z] = 0.5
    mf.joints.data[j0 + JOINT_IDX_AXIS_Y] = 1.0
    mf.meta.data[MODEL_META_IDX_NBODY] = Scalar[DTYPE](S_NBODY)
    mf.meta.data[MODEL_META_IDX_NJOINT] = Scalar[DTYPE](S_NJOINT)
    mf.sites.data[0 * MODEL_SITE_SIZE + SITE_IDX_BODY] = 1.0
    mf.sites.data[0 * MODEL_SITE_SIZE + SITE_IDX_POS_Z] = -0.5
    mf.sites.data[1 * MODEL_SITE_SIZE + SITE_IDX_BODY] = 0.0
    mf.sites.data[1 * MODEL_SITE_SIZE + SITE_IDX_POS_X] = 1.0
    mf.upload_all(ctx)

    var angles = List[Float64]()
    angles.append(0.7)
    angles.append(-1.2)

    var d = Data[DTYPE, MD_2, S_BATCH]()
    var dc = Data[DTYPE, MD_2, S_BATCH]()
    for e in range(S_BATCH):
        d.qpos.data[e * S_NQ + 0] = Scalar[DTYPE](angles[e])
        dc.qpos.data[e * S_NQ + 0] = Scalar[DTYPE](angles[e])
    d.upload_all(ctx)

    forward_kinematics["gpu", DTYPE, BATCH=S_BATCH](d, mf, ctx)
    d.download_all(ctx)
    forward_kinematics["cpu", DTYPE, BATCH=S_BATCH](dc, mf)

    var worst = Float64(0)
    for i in range(S_BATCH * S_NBODY * 3):
        var err = abs(Float64(dc.xpos.data[i]) - Float64(d.xpos.data[i]))
        if err > worst:
            worst = err
    for i in range(S_BATCH * S_NSITE * 3):
        var err = abs(
            Float64(dc.site_xpos.data[i]) - Float64(d.site_xpos.data[i])
        )
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU worst err:", worst)
    if worst > POS_TOL:
        raise Error("synthetic fields-CPU tolerance exceeded")
    print("  PASS: fields-CPU within 1e-4 (incl. site_xpos)")

    var fp = Float64(0)
    _fold(fp, 1, d.xpos.data, S_BATCH * S_NBODY * 3)
    _fold(fp, 2, d.xquat.data, S_BATCH * S_NBODY * 4)
    _fold(fp, 3, d.xipos.data, S_BATCH * S_NBODY * 3)
    _fold(fp, 4, d.site_xpos.data, S_BATCH * S_NSITE * 3)
    _gold_check("synthetic sites", fp, GOLD_B)


def main() raises:
    test_walker2d()
    test_synthetic_sites()
    print("test_fk_fields: ALL PASS")
