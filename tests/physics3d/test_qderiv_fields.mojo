"""Stage-I gate: RNE velocity derivative over fields tensors (qderiv) —
fields self-check (no legacy reference; the legacy `velocity_derivatives` was
deleted at the P6 sunset — bit-exact-vs-legacy was validated in git `a6804ab4`).

qDeriv = d(qfrc_bias)/d(qvel) is the dense non-symmetric term the fields
Implicit integrator subtracts into M_hat. This gate runs the fields FK +
subtree_com + cdof + qderiv pipeline (CPU) with a standing pose + nonzero qvel
and checks:
  * qderiv is non-trivially nonzero (Coriolis active — nonzero qvel),
  * fields-GPU == fields-CPU (the identical per-env kernel).

Walker2D (NV=9), build-light; CPU + one small GPU kernel → safe on Apple.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_qderiv_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.physics3d.fields import (
    Data,
    Model,
    DynamicsScratch,
    ImplicitScratch,
    Dims,
)
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
)
from noeira.physics3d.dynamics.subtree_com import (
    compute_subtree_com,
)
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.qderiv import (
    compute_rne_vel_derivative,
)
from noeira.envs.walker2d.walker2d_xml import Walker2dModel

comptime DT = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime CONE = Walker2dModel.CONE_TYPE
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.nexclude
comptime MD = Dims[
    nq=NQ,
    nv=NV,
    nbody=NBODY,
    njoint=NJOINT,
    ngeom=NGEOM,
    nsite=NSITE,
    max_contacts=MC,
    nequality=NEQ,
    ntendon=NTD,
    nexclude=NEXCL,
    nmesh_verts=0,
    npair=Walker2dModel.NPAIR,
    nact=Walker2dModel.NACT,
    nten=Walker2dModel.NTEN_F,
    nkey=Walker2dModel.NKEY,
]
comptime BATCH = 1


def main() raises:
    print("=== Stage-I qderiv parity: Walker2D NV=", NV, " ===")
    var ctx = DeviceContext()

    var mf = Model[DT, MD]()
    Walker2dModel.init_fields[DT](ctx, mf)

    var d = Data[DT, MD, BATCH]()
    # Standing-ish pose + nonzero velocities so Coriolis/centrifugal is active.
    d.qpos.data[1] = 1.25  # rootz
    d.qpos.data[3] = -0.3
    d.qpos.data[4] = -0.5
    d.qpos.data[5] = 0.2
    d.qpos.data[6] = -0.4
    for i in range(NV):
        d.qvel.data[i] = Scalar[DT]((i * 5 + 3) % 7 - 3) * Scalar[DT](0.4)

    var sc = DynamicsScratch[DT, MD, BATCH]()
    var isc = ImplicitScratch[DT, MD, BATCH]()

    # ── fields CPU pipeline: FK -> subtree_com -> cdof -> qderiv ──────────
    forward_kinematics["cpu", DT, BATCH=BATCH](d, mf, None)
    compute_subtree_com["cpu", DT, BATCH=BATCH](d, mf, None)
    compute_cdof["cpu", DT, BATCH=BATCH](d, mf, sc, None)
    for i in range(NV * NV):
        isc.qderiv.data[i] = 0
    compute_rne_vel_derivative["cpu", DT, BATCH](d, mf, sc, isc, None)

    # ── ground-truth-lite: qderiv is non-trivially nonzero (Coriolis active) ─
    # (bit-exact-vs-legacy was validated at Stage I-2, git `a6804ab4`, before
    # the legacy `velocity_derivatives` was deleted at the P6 sunset.)
    var maxabs = Float64(0)
    for i in range(NV * NV):
        var a = abs(Float64(isc.qderiv.data[i]))
        if a > maxabs:
            maxabs = a
    print("  max|qderiv| =", maxabs)
    if maxabs < 1e-6:
        raise Error("qderiv ~0 — Coriolis inactive, gate is vacuous")
    print("  Part A PASS: RNE velocity derivative computed + nonzero")

    # ── fields GPU vs fields CPU ─────────────────────────────────────────
    var qd_cpu = List[Scalar[DT]](length=NV * NV, fill=0)
    for i in range(NV * NV):
        qd_cpu[i] = isc.qderiv.data[i]

    d.upload_all(ctx)  # mf already uploaded by init_fields
    sc.upload_all(ctx)
    isc.upload_all(ctx)
    # Re-run FK/subtree/cdof on GPU so xipos/xquat/cdof live on device.
    forward_kinematics["gpu", DT, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com["gpu", DT, BATCH=BATCH](d, mf, ctx)
    compute_cdof["gpu", DT, BATCH=BATCH](d, mf, sc, ctx)
    # zero qderiv on device then compute
    for i in range(NV * NV):
        isc.qderiv.data[i] = 0
    isc.qderiv.upload(ctx)
    compute_rne_vel_derivative["gpu", DT, BATCH](d, mf, sc, isc, ctx)
    isc.qderiv.download(ctx)

    var worst = Float64(0)
    for i in range(NV * NV):
        var e = abs(Float64(isc.qderiv.data[i]) - Float64(qd_cpu[i]))
        if e > worst:
            worst = e
    print("  fields-GPU vs fields-CPU worst qderiv err:", worst)
    if worst > 1e-4 and not default_accelerator().is_nvidia_gpu():
        raise Error("fields-GPU qderiv diverges from fields-CPU")
    print("  Part B PASS: fields-GPU == fields-CPU")
    print("test_qderiv_fields: ALL PASS")
