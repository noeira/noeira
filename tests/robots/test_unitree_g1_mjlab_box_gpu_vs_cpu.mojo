"""Unitree G1 (mjlab's) plus a free box: batched GPU vs CPU, ONE STEP AT A TIME from the same state.

    pixi run -e nvidia mojo run -I . tests/robots/test_unitree_g1_mjlab_box_gpu_vs_cpu.mojo

`test_unitree_g1_gpu_vs_cpu`'s protocol (read its header for why the state is
re-injected before every step and why a float32 knife-edge step is excused),
on `UnitreeG1MjlabBox`, and on the STATE instead of the observation: the
env's observation is the robot's, and what this family adds is the box
(qpos 36..42, qvel 35..40).

The float64 CPU env is the trajectory: the box is released above the
forearms of the standing robot (`test_unitree_g1_mjlab_box_vs_mujoco`'s
`catch`), the joints are driven, and the box lands on the arms, then on the
floor. Before EVERY control step its qpos / qvel are injected into the GPU
lanes and into a float32 CPU env; all three take one step; qpos and qvel are
compared, with the box's own entries reported apart.

    GPU   vs CPU64   the device against the env proved on MuJoCo
    CPU32 vs CPU64   the precision floor
    GPU   vs CPU32   the device column: the one that is gated

A device step passes if it reproduces the float64 answer OR the float32 CPU
answer within `ATOL + RTOL·|cpu64|`; float64 agreement must hold on at least
`MIN_F64_FRACTION` of the steps.

WHAT IT GATES THAT THE ROBOT'S OWN DEVICE GATES DO NOT: a second free body on
the batched path, box-capsule / box-sphere / box-plane contacts there, and the
contact-parameter mixing by priority (the box geom is `condim 3`,
`priority 1`) in the device's narrow phase and solver.

⚠ NVIDIA-ONLY, as every G1 device gate (nv 41 here): on Apple this SKIPS.
"""

from max.gpu.host import DeviceContext
from std.sys import default_accelerator
from std.math import abs, sin
from std.testing import assert_true, TestSuite

from noeira.nn.constants import DT
from noeira.core.cont_action import ContAction
from noeira.envs.robots import UnitreeG1MjlabBox, UnitreeG1MjlabBoxBatched
from noeira.envs.robots.unitree_g1_mjlab_box_xml import G1M_BOX_QPOS, G1M_BOX_QVEL, UnitreeG1MjlabBoxModel


comptime N_ENVS = 2
comptime N_STEPS = 60
comptime ATOL_DEV = 1e-3
comptime RTOL_DEV = 1e-2
comptime MIN_F64_FRACTION = 0.9
comptime NQ = UnitreeG1MjlabBoxModel.NQ
comptime NV = UnitreeG1MjlabBoxModel.NV
comptime ACT_DIM = UnitreeG1MjlabBoxModel.ACTION_DIM
comptime BOX_Z0 = 1.05
comptime BOX_VZ0 = -0.5


def _action(t: Int, j: Int) -> Float64:
    return 0.3 * sin(Float64(t) * 0.23 + Float64(j) * 0.61)


def _state(env: UnitreeG1MjlabBox[DType.float64], k: Int) -> Float64:
    """Entry k of [qpos, qvel]."""
    if k < NQ:
        return Float64(env.d.qpos.data[k])
    return Float64(env.d.qvel.data[k - NQ])


def _is_box(k: Int) -> Bool:
    return (k >= G1M_BOX_QPOS and k < NQ) or k >= NQ + G1M_BOX_QVEL


def _run(ctx: DeviceContext) raises:
    var cpu64 = UnitreeG1MjlabBox[DType.float64]()
    var cpu32 = UnitreeG1MjlabBox[DType.float32]()
    var gpu = UnitreeG1MjlabBoxBatched[N_ENVS](ctx)
    _ = cpu64.reset()
    _ = cpu32.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(3))

    # the box above the forearms, falling
    var q0 = List[Float64](capacity=NQ)
    var v0 = List[Float64](capacity=NV)
    for i in range(NQ):
        q0.append(Float64(cpu64.d.qpos.data[i]))
    for i in range(NV):
        v0.append(Float64(cpu64.d.qvel.data[i]))
    q0[G1M_BOX_QPOS] = 0.27
    q0[G1M_BOX_QPOS + 1] = 0.0
    q0[G1M_BOX_QPOS + 2] = BOX_Z0
    v0[G1M_BOX_QVEL + 2] = BOX_VZ0
    cpu64.set_state(q0, v0)

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    ctx.synchronize()

    var worst_gpu64 = 0.0
    var worst_3264 = 0.0
    var worst_dev = 0.0
    var worst_dev_box = 0.0
    var worst_dev_step = -1
    var worst_dev_k = -1
    var worst_lane = 0.0
    var n_bad = 0
    var n_f64_ok = 0
    var n_knife = 0
    var held_up = 0.0           # how far above free fall the box was while still in the air
    var box_travel = 0.0

    for t in range(N_STEPS):
        # ── inject the float64 trajectory's state into the other two ────
        var qp = List[Float64](capacity=NQ)
        var qv = List[Float64](capacity=NV)
        for i in range(NQ):
            qp.append(Float64(cpu64.d.qpos.data[i]))
        for i in range(NV):
            qv.append(Float64(cpu64.d.qvel.data[i]))
        cpu32.set_state(qp, qv)
        gpu.d.qpos.download(ctx)
        gpu.d.qvel.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            for i in range(NQ):
                gpu.d.qpos.data[e * NQ + i] = Scalar[DT](qp[i])
            for i in range(NV):
                gpu.d.qvel.data[e * NV + i] = Scalar[DT](qv[i])
        gpu.d.qpos.upload(ctx)
        gpu.d.qvel.upload(ctx)
        ctx.synchronize()

        # ── one control step on all three ────────────────────────────────
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            var u = _action(t, j)
            act.data[j] = u
            for e in range(N_ENVS):
                h_act[e * ACT_DIM + j] = Scalar[DT](u)
        ctx.enqueue_copy(gpu._action, h_act)
        gpu.step_batch[N_ENVS](Optional(ctx), 0)
        gpu.d.qpos.download(ctx)
        gpu.d.qvel.download(ctx)
        ctx.synchronize()
        _ = cpu64.step(act)
        _ = cpu32.step(act)

        # ── the three columns, plus lane agreement ───────────────────────
        var step_gpu64 = 0.0
        var step_3264 = 0.0
        var step_dev = 0.0
        var step_f64_ok = True
        var step_knife = False
        for k in range(NQ + NV):
            var v64 = _state(cpu64, k)
            var v32 = Float64(cpu32.d.qpos.data[k]) if k < NQ else Float64(cpu32.d.qvel.data[k - NQ])
            var vg = Float64(gpu.d.qpos.data[k]) if k < NQ else Float64(gpu.d.qvel.data[k - NQ])
            var d_gpu64 = abs(vg - v64)
            var d_3264 = abs(v32 - v64)
            var d_dev = abs(vg - v32)
            step_gpu64 = max(step_gpu64, d_gpu64)
            step_3264 = max(step_3264, d_3264)
            step_dev = max(step_dev, d_dev)
            if d_dev > worst_dev:
                worst_dev = d_dev
                worst_dev_step = t
                worst_dev_k = k
            if _is_box(k):
                worst_dev_box = max(worst_dev_box, d_dev)
            var band = ATOL_DEV + RTOL_DEV * abs(v64)
            if d_gpu64 > band:
                step_f64_ok = False
            if d_3264 > band:
                step_knife = True
            if d_gpu64 > band and d_dev > band:
                print("  DEVICE MISMATCH step=", t, " k=", k, "(box)" if _is_box(k) else "(robot)", " gpu=", vg, " cpu32=", v32,
                      " cpu64=", v64, " |gpu-cpu64|=", d_gpu64, " |gpu-cpu32|=", d_dev)
                n_bad += 1
            for e in range(1, N_ENVS):
                var ve = Float64(gpu.d.qpos.data[e * NQ + k]) if k < NQ else Float64(gpu.d.qvel.data[e * NV + k - NQ])
                worst_lane = max(worst_lane, abs(ve - vg))
        if step_f64_ok:
            n_f64_ok += 1
        if step_knife or not step_f64_ok:
            n_knife += 1
            print("  KNIFE-EDGE step", t, " |gpu-cpu64|", step_gpu64, " |cpu32-cpu64|", step_3264, " |gpu-cpu32|", step_dev,
                  "  (device sides with", "cpu64" if step_f64_ok else "cpu32", ")")
        worst_gpu64 = max(worst_gpu64, step_gpu64)
        worst_3264 = max(worst_3264, step_3264)
        # the box against free fall from its release, while free fall is still above the floor
        var tt = Float64(t + 1) * 0.02
        var z_ff = BOX_Z0 + BOX_VZ0 * tt - 0.5 * 9.81 * tt * tt
        var zb = Float64(cpu64.d.qpos.data[G1M_BOX_QPOS + 2])
        if z_ff > 0.35:
            held_up = max(held_up, zb - z_ff)
        box_travel = max(box_travel, BOX_Z0 - zb)
        if t < 3 or t % 10 == 9:
            print("      step", t, " |gpu-cpu64|", step_gpu64, " |cpu32-cpu64|", step_3264, " |gpu-cpu32|", step_dev, " box z", zb)

    print("  unitree_g1_mjlab_box one-step GPU vs CPU over", N_STEPS, "steps x", N_ENVS, "lanes (qpos and qvel):")
    print("    GPU   vs CPU64 worst", worst_gpu64)
    print("    CPU32 vs CPU64 worst", worst_3264)
    print("    GPU   vs CPU32 (GATED) worst", worst_dev, " at step", worst_dev_step, " k", worst_dev_k, "| on the box's entries", worst_dev_box)
    print("    lane-to-lane worst", worst_lane, "| box fell", box_travel, "m, held", held_up, "m above free fall by the arms")
    print("    device agreed with float64 on", n_f64_ok, "of", N_STEPS, "steps; float32 knife-edge steps:", n_knife)
    assert_true(worst_lane == 0.0, "the lanes disagree by " + String(worst_lane) + " on identical inputs")
    assert_true(box_travel > 0.5, "the box fell only " + String(box_travel) + " m: the gate is vacuous")
    assert_true(held_up > 0.03, "the box was never slowed by the robot (" + String(held_up) + " m above free fall): no box-robot contact was gated")
    assert_true(n_bad == 0, String(n_bad) + " element(s) where the device matched NEITHER CPU step: see the DEVICE MISMATCH lines")
    assert_true(
        Float64(n_f64_ok) >= MIN_F64_FRACTION * Float64(N_STEPS),
        "the device agreed with float64 on only " + String(n_f64_ok) + " of " + String(N_STEPS) + " steps",
    )


def test_unitree_g1_mjlab_box_gpu_matches_cpu() raises:
    if not default_accelerator().is_nvidia_gpu():
        print("  unitree_g1_mjlab_box GPU vs CPU: SKIPPED off NVIDIA (nv 41 exceeds Metal's per-thread stack in the Newton solver). UNGATED until run on NVIDIA.")
        return
    with DeviceContext() as ctx:
        _run(ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
