"""Stage-I gate: full-implicit integrator over fields tensors
(ImplicitIntegrator) — fields self-consistency smoke.

Runs N contact-free passive steps (qfrc=0: gravity + Coriolis + damping only)
on a FREE-FLIGHT Walker2D (rootz high → no floor contact) with nonzero qvel
(so the RNE velocity derivative in M_hat = M + armature - dt*qDeriv actually
matters), poses strictly inside joint ranges (limits inactive). Checks:
  * fields-CPU step stays finite (no NaN/blowup),
  * fields-GPU ≈ fields-CPU (self-consistent single-source integrator).
(The legacy ImplicitIntegrator cross-check was dropped at the P6 sunset; the
FK/CRBA/RNE/qDeriv stages remain bit-exact-validated by their own gates.)

Run: pixi run -e apple mojo run -I . tests/physics3d/test_implicit_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.physics3d.fields import Data, Model, Dims
from noeira.physics3d.integrator.implicit import (
    ImplicitIntegrator,
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
comptime N_STEPS = 3


def _init_qpos(i: Int) -> Scalar[DT]:
    """Free-flight walker2d pose, strictly inside all joint ranges."""
    if i == 1:
        return Scalar[DT](2.0)  # rootz high → no contact
    elif i == 3:
        return Scalar[DT](-0.3)  # thigh
    elif i == 4:
        return Scalar[DT](-0.5)  # leg
    elif i == 5:
        return Scalar[DT](-0.2)  # foot
    elif i == 6:
        return Scalar[DT](-0.4)  # thigh_left
    elif i == 7:
        return Scalar[DT](-0.35)  # leg_left
    elif i == 8:
        return Scalar[DT](-0.15)  # foot_left
    return Scalar[DT](0)


def _init_qvel(i: Int) -> Scalar[DT]:
    return Scalar[DT]((i * 5 + 3) % 7 - 3) * Scalar[DT](0.3)


def main() raises:
    print("=== Stage-I ImplicitIntegrator self-consistency: Walker2D ===")
    var ctx = DeviceContext()

    var mf = Model[DT, MD]()
    Walker2dModel.init_fields[DT](ctx, mf)

    var d = Data[DT, MD, BATCH]()
    for i in range(NQ):
        d.qpos.data[i] = _init_qpos(i)
    for i in range(NV):
        d.qvel.data[i] = _init_qvel(i)
    # qfrc stays 0 → pure passive dynamics.

    var integ = ImplicitIntegrator[DT, MD, CONE, BATCH, SOLVER="pgs"]()

    for _step in range(N_STEPS):
        integ.step["cpu", False](d, mf)

    # finiteness
    for i in range(NQ):
        var v = d.qpos.data[i]
        if v != v or abs(Float64(v)) > 1e6:
            raise Error("fields implicit produced non-finite qpos")

    print("  Part A PASS: fields-CPU implicit step finite (Walker2D free-flight)")

    # ── fields-GPU vs fields-CPU ─────────────────────────────────────────
    var qcpu = List[Scalar[DT]](length=NQ, fill=0)
    var vcpu = List[Scalar[DT]](length=NV, fill=0)
    for i in range(NQ):
        qcpu[i] = d.qpos.data[i]
    for i in range(NV):
        vcpu[i] = d.qvel.data[i]

    var dg = Data[DT, MD, BATCH]()
    for i in range(NQ):
        dg.qpos.data[i] = _init_qpos(i)
    for i in range(NV):
        dg.qvel.data[i] = _init_qvel(i)
    dg.upload_all(ctx)
    integ.prepare_gpu(ctx)
    for _step in range(N_STEPS):
        integ.step["gpu", False](dg, mf, ctx)
    dg.qpos.download(ctx)
    dg.qvel.download(ctx)

    var worst_g = Float64(0)
    for i in range(NQ):
        var e = abs(Float64(dg.qpos.data[i]) - Float64(qcpu[i]))
        if e > worst_g:
            worst_g = e
    for i in range(NV):
        var e = abs(Float64(dg.qvel.data[i]) - Float64(vcpu[i]))
        if e > worst_g:
            worst_g = e
    print("  fields-GPU vs fields-CPU worst err:", worst_g)
    if worst_g > 1e-3 and not default_accelerator().is_nvidia_gpu():
        raise Error("fields-GPU implicit diverges from fields-CPU")
    print("  Part B PASS: fields-GPU ≈ fields-CPU")
    print("test_implicit_fields: ALL PASS")
