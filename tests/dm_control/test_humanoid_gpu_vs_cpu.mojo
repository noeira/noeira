"""dm_control humanoid + humanoid_CMU: batched GPU vs CPU, per step.

Fourth of the GPU-vs-CPU gates; read `test_pendulum_gpu_vs_cpu.mojo`'s header
for why the comparison has to be per-step.

These are the largest ports in tranche 1 — a 67-D engineered observation built
from a torso-frame rotation, four egocentric limb offsets and a mass-weighted
CoM velocity — and the first with a FREE ROOT, so `init_qpos_gpu` goes through
`randomize_limited_and_rotational_joints_gpu`'s quaternion branch.

⚠ FOUR CONFIGURATIONS, none redundant:
  * humanoid-stand      — MOVE_SPEED == 0: `dont_move`, a MEAN over the two
                          horizontal CoM components scored separately
  * humanoid-walk       — MOVE_SPEED != 0: `move`, the NORM of the same two.
                          Different formula, not a different constant.
  * humanoid-run-pure   — PURE_STATE: the raw qpos+qvel observation, which
                          skips the entire extremities/xmat path
  * humanoidCMU-walk    — the THORAX/ZY variant. This domain differs from
                          `humanoid` in exactly two places and both are easy to
                          copy wrongly: the subtree root and extremity frame
                          are the thorax, and `thorax_upright()` reads xmat ZY
                          where humanoid reads ZZ. A gate on humanoid alone
                          would say nothing about either.

⚠⚠ humanoidCMU IS SKIPPED ON APPLE, and the reason is not this port. Metal
refuses its PHYSICS kernel outright — "Compute function exceeds available stack
space", raised while creating the compute pipeline for the integrator, before
any hook of ours runs. Bisected with `_step_impl[..., DEBUG=True]`: `pre_step`
and `apply_actions` both pass, the integrator stage does not. The cause is
size, not correctness — humanoid_CMU is NQ 63 / NV 62 / NBODY 32 against
humanoid's 28 / 27 / 17, and the Newton solver's per-thread working set scales
with NV^2, so ~5x humanoid's stack. Its GPU hooks are therefore written and
compiled but NOT VALUE-GATED; that has to happen on NVIDIA, where the per-thread
stack is far larger. Do not read a green run on Apple as covering it.

Run with:
    pixi run -e apple  mojo run -I . tests/dm_control/test_humanoid_gpu_vs_cpu.mojo
    pixi run -e nvidia mojo run -I . tests/dm_control/test_humanoid_gpu_vs_cpu.mojo
"""

from max.gpu.host import DeviceContext
from std.sys import default_accelerator
from std.math import abs, sin
from std.testing import assert_true, TestSuite

from noeira.nn.constants import DT
from noeira.core.cont_action import ContAction
from noeira.physics3d.model.model_def import ModelDefLike
from noeira.envs.phyics3d_env_config import Phyics3dEnvConfig
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.dm_control.humanoid import (
    DMHumanoidModel,
    DMHumanoidPureModel,
    DMHumanoidConfig,
    WALK_SPEED,
    RUN_SPEED,
)
from noeira.envs.dm_control.humanoid_cmu import (
    DMHumanoidCMUModel,
    DMHumanoidCMUConfig,
)

comptime N_ENVS = 2

# Short window: humanoid is FRAME_SKIP=5 and humanoid_CMU FRAME_SKIP=10, both
# with a full contact set, so a step here is 1-2 orders more physics than
# pendulum's. 20 control steps is still 100-200 substeps of accumulation.
comptime N_STEPS = 20

# Mixed absolute + relative — see `test_locomotion_gpu_vs_cpu.mojo` for why an
# absolute-only bound is the wrong instrument on an observation vector that
# mixes O(1) rotation entries with O(10) joint velocities. Looser than the
# locomotion gate because these models have far more contacting geoms, so the
# contact set itself can diverge a substep earlier on one path.
comptime ATOL: Float64 = 3e-2
comptime RTOL: Float64 = 1e-2


def _run[
    MODEL: ModelDefLike,
    CFG: Phyics3dEnvConfig,
    label: StaticString,
](ctx: DeviceContext, mut worst: Float64) raises:
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime OBS_DIM = MODEL.OBS_DIM
    comptime ACT_DIM = MODEL.ACTION_DIM

    var cpu = Phyics3dEnv[MODEL, CFG, DType.float64, False]()
    var gpu = Phyics3dBatchedEnv[
        MODEL, CFG, N_ENVS, TERMINATE_ON_UNHEALTHY=False
    ](ctx)

    _ = cpu.reset()
    gpu.reset_batch[N_ENVS](Optional(ctx), UInt64(3))

    # Shared start: upright, identity root quaternion, small forward CoM
    # velocity. Chosen so `standing` (margin STAND_HEIGHT/4) and `upright` are
    # both strictly inside their margins rather than saturated at 0 — a
    # collapsed humanoid scores ~0 whatever the CoM velocity is, which would
    # let a broken `subtree_linvel` pass unnoticed.
    var qpos0 = List[Float64](length=NQ, fill=0.0)
    var qvel0 = List[Float64](length=NV, fill=0.0)
    qpos0[2] = 1.5          # root z — the XML's spawn height
    qpos0[3] = 1.0          # root quaternion, W-FIRST: identity
    for i in range(7, NQ):
        qpos0[i] = 0.02 * Float64((i % 7) - 3)
    qvel0[0] = 0.5          # forward, so the CoM velocity is nonzero at step 0
    for i in range(6, NV):
        qvel0[i] = 0.02 * Float64((i % 5) - 2)
    cpu.set_state(qpos0, qvel0)

    gpu.d.qpos.download(ctx)
    gpu.d.qvel.download(ctx)
    ctx.synchronize()
    for e in range(N_ENVS):
        for i in range(NQ):
            gpu.d.qpos.data[e * NQ + i] = Scalar[DT](qpos0[i])
        for i in range(NV):
            gpu.d.qvel.data[e * NV + i] = Scalar[DT](qvel0[i])
    gpu.d.qpos.upload(ctx)
    gpu.d.qvel.upload(ctx)
    ctx.synchronize()

    var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
    var h_rew = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    ctx.synchronize()

    var max_obs = 0.0
    var max_rew = 0.0
    var max_rel = 0.0
    var worst_step = -1
    var worst_k = -1
    var n_bad = 0
    var rew_lo = 1e30
    var rew_hi = -1e30

    for t in range(N_STEPS):
        var act = ContAction[ACT_DIM]()
        for j in range(ACT_DIM):
            # Per-actuator phase. A shared torque would drive a symmetric pose
            # and leave the left/right limb offsets mirror images of each
            # other — which is exactly what an extremity-ORDER error looks
            # like, so it would hide the bug this gate exists to catch.
            var u = 0.4 * sin(Float64(t) * 0.31 + Float64(j) * 0.7)
            act.data[j] = u
            for e in range(N_ENVS):
                h_act[e * ACT_DIM + j] = Scalar[DT](u)
        ctx.enqueue_copy(gpu._action, h_act)
        gpu.step_batch[N_ENVS](Optional(ctx), 0)
        ctx.enqueue_copy(h_obs, gpu._obs)
        ctx.enqueue_copy(h_rew, gpu._reward)
        ctx.synchronize()

        var res = cpu.step(act)
        var cpu_rew = Float64(res[1])
        if cpu_rew < rew_lo:
            rew_lo = cpu_rew
        if cpu_rew > rew_hi:
            rew_hi = cpu_rew

        for e in range(N_ENVS):
            for k in range(OBS_DIM):
                var cpu_v = res[0].data[k]
                var d = abs(Float64(h_obs[e * OBS_DIM + k]) - cpu_v)
                if d > max_obs:
                    max_obs = d
                var rel = d / (abs(cpu_v) if abs(cpu_v) > 1e-12 else 1.0)
                if rel > max_rel:
                    max_rel = rel
                    worst_step = t
                    worst_k = k
                if d > ATOL + RTOL * abs(cpu_v):
                    print(
                        label, " OBS MISMATCH step=", t, " env=", e, " k=", k,
                        " gpu=", h_obs[e * OBS_DIM + k],
                        " cpu=", cpu_v, " diff=", d, " rel=", rel,
                    )
                    n_bad += 1
            var dr = abs(Float64(h_rew[e]) - cpu_rew)
            if dr > max_rew:
                max_rew = dr
            if dr > ATOL + RTOL * abs(cpu_rew):
                print(
                    label, " REWARD MISMATCH step=", t, " env=", e,
                    " gpu=", h_rew[e], " cpu=", cpu_rew, " diff=", dr,
                )
                n_bad += 1

    print(
        "  ", label, ": max |obs diff| = ", max_obs,
        ", max rel = ", max_rel, " (step ", worst_step, ", k ", worst_k, ")",
        ", max |rew diff| = ", max_rew,
        "  [cpu reward ", rew_lo, " .. ", rew_hi, "]",
    )
    assert_true(
        n_bad == 0,
        String(label) + ": " + String(n_bad)
        + " element(s) outside atol+rtol*|cpu| — see the MISMATCH lines above",
    )
    # Non-vacuity: these rewards are products of `tolerance` terms that
    # saturate at 0, so a badly chosen start state would make the diff pass for
    # free. Require the CPU reward to have actually moved.
    assert_true(
        rew_hi - rew_lo > 1e-6,
        String(label)
        + ": CPU reward never moved — the gate is vacuous, pick a start state"
        " where the reward is not saturated",
    )
    if max_obs > worst:
        worst = max_obs
    if max_rew > worst:
        worst = max_rew


def test_humanoid_gpu_matches_cpu() raises:
    with DeviceContext() as ctx:
        var worst = 0.0
        _run[DMHumanoidModel, DMHumanoidConfig[0.0, False],
             "humanoid-stand   "](ctx, worst)
        _run[DMHumanoidModel, DMHumanoidConfig[WALK_SPEED, False],
             "humanoid-walk    "](ctx, worst)
        _run[DMHumanoidPureModel, DMHumanoidConfig[RUN_SPEED, True],
             "humanoid-run-pure"](ctx, worst)
        # See the ⚠⚠ note in the module docstring: Metal cannot build
        # humanoid_CMU's integrator kernel at all. Skipping keeps the gate
        # honest on Apple instead of turning a hardware limit into a red test
        # that nobody can fix here.
        if default_accelerator().is_nvidia_gpu():
            _run[DMHumanoidCMUModel, DMHumanoidCMUConfig[WALK_SPEED],
                 "humanoidCMU-walk "](ctx, worst)
        else:
            print(
                "   humanoidCMU-walk  : SKIPPED on Apple — Metal rejects its"
                " physics kernel ('exceeds available stack space', NV=62)."
                " UNGATED until run on NVIDIA."
            )
        print(
            "humanoid GPU vs CPU: ", 4 if default_accelerator().is_nvidia_gpu() else 3,
            " configs x ", N_STEPS, " steps x ",
            N_ENVS, " lanes — worst abs diff = ", worst,
            " (bound ", ATOL, " + ", RTOL, "*|cpu|)",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
