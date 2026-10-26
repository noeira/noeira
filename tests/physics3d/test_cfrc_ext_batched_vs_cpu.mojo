"""`gpu/cfrc_ext_gpu.compute_cfrc_ext` (batched) vs the CPU `rne_post` path.

WHY THIS EXISTS. `compute_cfrc_ext` is a SECOND implementation of `cfrc_ext`,
with different arithmetic from the CPU one — it rebuilds each body's subtree CoM
from `xipos` inside the kernel instead of reading `Data.subtree_com`. It runs on
every batched step and every batched reset, and **nothing reads its output**: a
grep of every env config finds `cfrc_ext` exactly once per file, always as a
hook PARAMETER declaration, never in an expression.

An unread second implementation is a correctness liability rather than a
performance one. It cannot be caught by any existing gate, it can rot silently,
and whoever eventually wires it up (Ant-v5's contact cost, or a touch-based
reward) will reasonably trust it. So it gets gated rather than deleted — the
capability is real and the arithmetic was validated against MuJoCo once.

WHY AGAINST THE CPU PATH AND NOT MUJOCO. `test_rne_post_sensors_vs_mujoco`
already pins the CPU `cfrc_ext` against MuJoCo at 6.5e-11 on this exact model
and pose. Diffing the batched one against the CPU one therefore gates it
against MuJoCo TRANSITIVELY, with no MuJoCo in this file at all — and it
compares the two implementations directly, which is the thing that can drift.

⚠ The two do NOT have to agree bit-for-bit: the CPU path reads
`Data.subtree_com` (accumulated by `compute_subtree_com`) while the kernel
rebuilds it from `xipos` and `body_mass`. Same quantity, different summation
order. The tolerance below is set from the measurement, not guessed.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_cfrc_ext_batched_vs_cpu.mojo
"""

from std.math import abs
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.physics3d.fields import Data, Model, Dims, AsStatic
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.gpu import compute_cfrc_ext
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.envs.dm_control.quadruped.quadruped_xml import (
    DMQuadrupedWalkModel as Mdl,
)
from noeira.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from noeira.physics3d.model.model_dims import ModelDims


comptime DTYPE = DType.float32
comptime NQ = Mdl.NQ
comptime NV = Mdl.NV
comptime NBODY = Mdl.NBODY
comptime MC = Mdl.MAX_CONTACTS
comptime NSITE = Mdl.NSITE
comptime MD = ModelDims[Mdl]

comptime Integ = EulerIntegrator[DTYPE, MD, Mdl.CONE_TYPE, 1, SOLVER="newton", RNE_POST=True]
comptime Dat = Data[DTYPE, MD, 1]
comptime Mod = Model[DTYPE, MD]

# FLOAT32, because the batched path is a Metal kernel and Metal has no float64.
#
# MEASURED 2026-08-01 on the standing pose: **0.0**, i.e. the two
# implementations agree bit-for-bit there, with |cfrc_ext| up to 30.8 N across
# four loaded toes. They are not REQUIRED to — one reads `Data.subtree_com`
# and the other rebuilds it from `xipos`, so the summation orders differ — but
# quadruped's subtree structure makes them coincide. 1e-6 leaves room for a
# model where they do not, while still catching any real drift: the failure
# this gate exists for is a kernel that stops tracking the CPU path, which
# shows up far above float32 noise.
comptime TOL: Float64 = 1e-6


def test_batched_cfrc_ext_matches_cpu() raises:
    """Standing quadruped: four toes loaded, so the gate is not vacuous."""
    var sf = Mdl.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Mod()
    Mdl.init_fields[DTYPE](ctx, mf)
    var d = Dat()
    Mdl.reset_data(sf, d)

    # Standing on the floor. The height is SEARCHED, not hardcoded — the same
    # thing `test_rne_post_sensors_vs_mujoco::_find_standing_z` does, because
    # the toe geometry decides it and a fixed number silently becomes a
    # free-flight pose (zero contacts, zero cfrc_ext, vacuous gate) the moment
    # the model changes.
    var zero = List[Float64]()
    for _ in range(Mdl.ACTION_DIM):
        zero.append(0.0)
    var act = List[Scalar[DTYPE]]()
    for _ in range(Mdl.NA if Mdl.NA > 0 else 1):
        act.append(Scalar[DTYPE](0))
    for i in range(NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
    Mdl.apply_actions(sf, d, zero, act)

    var integ = Integ()
    var ncon = 0
    var z = 0.80
    while z > 0.30:
        Mdl.reset_data(sf, d)
        d.qpos.data[2] = Scalar[DTYPE](z)
        d.qpos.data[3] = Scalar[DTYPE](1.0)
        for i in range(NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        Mdl.apply_actions(sf, d, zero, act)
        integ.step["cpu"](d, mf)
        ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
        if ncon > 0:
            break
        z -= 0.01
    print("  standing z =", z)
    print("  contacts:", ncon)
    assert_true(
        ncon > 0,
        "no contacts — cfrc_ext would be all zeros and this gates nothing",
    )

    # CPU cfrc_ext, written by the RNE_POST stage of the step above.
    var cpu = List[Float64]()
    var cpu_scale = Float64(0)
    for i in range(NBODY * 6):
        var v = Float64(d.cfrc_ext.data[i])
        cpu.append(v)
        if abs(v) > cpu_scale:
            cpu_scale = abs(v)
    print("  |cfrc_ext|max (CPU) =", cpu_scale)
    assert_true(
        cpu_scale > 1e-6,
        "CPU cfrc_ext is ~zero at a standing pose — the gate is vacuous",
    )

    # Batched kernel, fed the SAME post-step state.
    for i in range(NBODY * 6):
        d.cfrc_ext.data[i] = Scalar[DTYPE](0)
    d.upload_all(ctx)
    mf.upload_all(ctx)
    compute_cfrc_ext[DTYPE, 1](
        ctx,
        AsStatic[MD](),
        d.xipos.tt_dyn["gpu", type_of(d).L_B3](type_of(d).L_B3()),
        d.contacts.tt_dyn["gpu", type_of(d).L_CONTACTS](type_of(d).L_CONTACTS()),
        d.meta.tt_dyn["gpu", type_of(d).L_META](type_of(d).L_META()),
        d.cfrc_ext.tt_dyn["gpu", type_of(d).L_B6](type_of(d).L_B6()),
        mf.bodies.tt_dyn["gpu", type_of(mf).L_BODY](type_of(mf).L_BODY()),
    )
    ctx.synchronize()
    d.cfrc_ext.download(ctx)

    var worst = Float64(0)
    var wi = 0
    for i in range(NBODY * 6):
        var e = abs(Float64(d.cfrc_ext.data[i]) - cpu[i])
        if e > worst:
            worst = e
            wi = i
    var rel = worst / (1e-15 + cpu_scale)
    print(
        "  worst |batched - cpu| =", worst, " at", wi,
        " rel =", rel,
    )
    assert_true(
        rel <= TOL,
        "the batched cfrc_ext kernel has drifted from the CPU path, which is"
        " the one gated against MuJoCo — they are two implementations of the"
        " same quantity and only this test compares them",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
