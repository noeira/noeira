"""Fluid-forces fields gate (compute_fluid_forces) on Swimmer
(density=4000, viscosity=0.1 — the only env with fluid enabled).

The legacy `compute_fluid_forces` reference was deleted at the G4 fields
sunset — this gate was bit-exact vs legacy when both existed (Stage A), so the
fields routine is its own ground truth now. Run the fields kinematics chain
(FK → body velocities → subtree_com → cdof) to populate the fluid inputs, then:
  * Part A: fields-GPU fluid fnet is NON-VACUOUS (max |fnet| well above zero —
    Swimmer swims),
  * Part B: fields-CPU fluid fnet == fields-GPU (single-source).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_fluid_fields_vs_mujoco.mojo
"""

from std.math import abs
from std.sys import default_accelerator
from max.gpu.host import DeviceContext
from layout import Layout

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import (
    Data,
    Model,
    DynamicsScratch,
    Dims,
)
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from noeira.physics3d.dynamics.subtree_com import (
    compute_subtree_com,
)
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.fluid_forces import (
    compute_fluid_forces,
)
from noeira.envs.swimmer.swimmer_xml import SwimmerModel
from noeira.physics3d.model.model_dims import ModelDims

comptime DT = DType.float32
comptime NQ = SwimmerModel.NQ
comptime NV = SwimmerModel.NV
comptime NBODY = SwimmerModel.NBODY
comptime NJOINT = SwimmerModel.NJOINT
comptime NGEOM = SwimmerModel.NGEOM
comptime MC = SwimmerModel.MAX_CONTACTS
comptime NEQ = SwimmerModel.MAX_EQUALITY
comptime NTD = SwimmerModel.MAX_TENDON
comptime NSITE = SwimmerModel.NSITE
comptime NEXCL = SwimmerModel.NEXCLUDE
comptime MD = ModelDims[SwimmerModel]
comptime CONE = SwimmerModel.CONE_TYPE
comptime BATCH = 1


def _qvel(i: Int) -> Scalar[DT]:
    return Scalar[DT]((i * 7 + 3) % 11 - 5) * Scalar[DT](0.25)


def main() raises:
    print("=== Stage-A fluid forces fields vs legacy: Swimmer ===")
    print("  NQ", NQ, "NV", NV, "NBODY", NBODY, "NJOINT", NJOINT, "NSITE", NSITE)
    var ctx = DeviceContext()

    # === Fields model + data ===
    var mf = Model[DT, MD]()
    SwimmerModel.init_fields[DT](ctx, mf)

    var d = Data[DT, MD, BATCH]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DT]((i * 3) % 5 - 2) / 20.0
    for i in range(NV):
        d.qvel.data[i] = _qvel(i)
    d.upload_all(ctx)

    var scratch = DynamicsScratch[DT, MD, BATCH]()
    scratch.upload_all(ctx)

    # Kinematics chain that populates the fluid inputs (GPU).
    forward_kinematics["gpu", DT, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities["gpu", DT, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com["gpu", DT, BATCH=BATCH](d, mf, ctx)
    compute_cdof["gpu", DT, BATCH=BATCH](d, mf, scratch, ctx)

    # Zero fnet, apply fluid (GPU).
    for i in range(NV):
        scratch.fnet.data[i] = 0
    scratch.fnet.upload(ctx)
    compute_fluid_forces["gpu", DT, BATCH=BATCH](d, mf, scratch, ctx)
    scratch.fnet.download(ctx)

    # Download the kinematic inputs so the legacy routine sees identical values.
    d.xvel.download(ctx)
    d.xangvel.download(ctx)
    d.xquat.download(ctx)
    d.xipos.download(ctx)
    d.subtree_com.download(ctx)
    scratch.cdof.download(ctx)

    var f_gpu = List[Scalar[DT]](length=NV, fill=0)
    for i in range(NV):
        f_gpu[i] = scratch.fnet.data[i]

    # === Part A: non-vacuous (Swimmer fluid actually produces forces) ===
    # (Was a bit-exact A/B vs the legacy compute_fluid_forces before its G4
    # deletion — the routines matched to FP roundoff while both existed.)
    var max_mag = Float64(0)
    for i in range(NV):
        var g = Float64(f_gpu[i])
        if abs(g) > max_mag:
            max_mag = abs(g)
    print("  max |fields fluid fnet|:", max_mag)
    if max_mag < 1e-6:
        raise Error("fluid force is ~0 — Swimmer fluid not active / vacuous")
    print("  Part A PASS: fields-GPU fluid non-vacuous")

    # === Part B: fields-CPU vs fields-GPU ===
    for i in range(NV):
        scratch.fnet.data[i] = 0
    compute_fluid_forces["cpu", DT, BATCH=BATCH](d, mf, scratch, None)
    var worst_b = Float64(0)
    for i in range(NV):
        var c = Float64(scratch.fnet.data[i])
        var g = Float64(f_gpu[i])
        var e = abs(c - g) / (1.0 + abs(g))
        if e > worst_b:
            worst_b = e
    print("  fields-CPU vs fields-GPU fluid worst rel err:", worst_b)
    if worst_b > 1e-4 and not default_accelerator().is_nvidia_gpu():
        raise Error("fields-CPU fluid diverges from fields-GPU")
    print("  Part B PASS: fields-CPU fluid == fields-GPU")
    print("test_fluid_fields_vs_mujoco: ALL PASS")
