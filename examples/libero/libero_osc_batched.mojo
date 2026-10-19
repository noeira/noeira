"""THE BATCHED LIBERO ENV, DRIVEN BY OSC_POSE — L6's wiring, running.

    pixi run -e nvidia mojo run -I . examples/libero/libero_osc_batched.mojo
    pixi run -e nvidia mojo run -I . examples/libero/libero_osc_batched.mojo 20   # control steps; 16 lanes (comptime)

⚠⚠ CORRECTED 2026-09-14: IT BUILDS AND STEPS ON APPLE TOO. The failure below
was not the nv = 37 stack: it was eight `ScratchPool`-backed scratches in the
elliptic Newton kernel (`cap[]` is 0 for a model with no tendons/equalities),
the same defect NVIDIA's ptxas reported as an unresolved
`KGEN_CompilerRT_GetOrCreateGlobal`. The paragraph is kept for the record.

(Historical:) RUN THIS ON NVIDIA, AND ON APPLE IT DOES NOT FAIL AT RUNTIME — IT
FAILS TO BUILD. `libero_goal` is nv = 37 and the Newton solver's per-thread arrays are
sized by it; Metal's backend answers "Metal Compiler failed to compile
metallib" and no binary is produced. That is not a bug to catch: the P0 park
probe died the same way at nv = 24, and `examples/tasks/task_eval_frozen.mojo`
carries the same warning for the SO-101 family.

⚠ WHICH MAKES THIS FILE THE TYPE CHECK ON A LAPTOP. Elaboration runs to
completion before Metal codegen, so a build that reaches the metallib error has
type-checked every comptime branch the controller adds — the layouts
`osc_control_step` binds, the `DynamicsScratch` in its `List`, the
`apply_actions_kernel_gpu` call fed from `ctrl` instead of `actions`, and the
`ACTION_DIM == 7` assert. `tests/libero/test_libero_osc_env.mojo` keeps the
NVIDIA leg behind `comptime if default_accelerator().is_nvidia_gpu()` so the suite still
runs on Apple; this one does not, deliberately.

## WHAT IT PRINTS, AND WHAT THAT DOES AND DOES NOT SHOW

Per control step: the mean and max end-effector speed, the arm's mean |torque|,
and how many lanes reported a SINGULAR operational-space inertia. With a zero
action the arm should hold — so the speeds say whether the controller is
actually closing its loop through the integrator, which is the one thing no
CPU-side gate can show.

⚠ THE NUMBERS ARE NOT A FIDELITY CLAIM. What the controller computes is gated
by `tests/libero/test_osc_control_batched.mojo` (eight lanes against `OscPose`,
1.7e-14 of the torque scale) and `tools/libero/libero_demo_replay.py` (one lane
against MuJoCo, 1.5e-5 m over a recorded demo). This shows the WIRING carries
those numbers into the physics; a lane-by-lane comparison against the CPU loop
in `examples/libero/libero_eval.mojo` is the gate that closes it, and it is owed
on the box.

⚠ THE REWARD IS ZERO UNTIL A TAPE IS WRITTEN. `LiberoGoalOscConfig`'s reward
IS the goal tape (`eval_tape_gpu` over `meta`), and this driver writes no tape —
so `_reward` is 0 by construction here, exactly as the config's docstring says.
Placing the free slots and writing the tape per lane is the task layer's, and
`tasks/gpu_eval.mojo` is where it lives.
"""

from std.sys import argv
from std.random import seed as seed_rng
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.physics3d.dynamics.osc_pose import ARM_DOF, OscPoseConfig
from noeira.physics3d.dynamics.osc_pose_gpu import (
    OSC_ACTION_DIM, build_osc_refs,
)
from noeira.tasks.spec import load_family
from noeira.tasks.family import scene_path
from noeira.envs.libero.models.libero_goal_dims import LIBERO_GOAL_DIMS
from noeira.envs.libero.models.libero_goal_xml import LIBERO_GOAL_OBS_DIM
from noeira.envs.libero.libero_goal_config import (
    LiberoGoalOscConfig, LiberoGoalOscEnv, LIBERO_GOAL_FRAME_SKIP,
)


comptime FAMILY = "noeira/envs/libero/families/libero_goal.family"
comptime N_ENVS = 16
comptime NB = LIBERO_GOAL_DIMS.NBODY


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("no '" + want + "' in the composed scene")


def main() raises:
    var args = argv()
    var n_steps = 10
    if len(args) > 1:
        n_steps = Int(String(args[1]))
    seed_rng(0)

    print("=" * 74)
    print("libero_goal on the batched env under OSC_POSE —", N_ENVS, "lanes,",
          n_steps, "control steps")
    print("=" * 74)

    var f = load_family(FAMILY)
    var fmd = parse_model_runtime(scene_path(f))

    # The record, from the parse — the same walk `libero_eval.mojo` does.
    var qadr_all = List[Int]()
    var dadr_all = List[Int]()
    var qa = 0
    var da = 0
    for i in range(len(fmd.joints)):
        qadr_all.append(qa)
        dadr_all.append(da)
        qa += fmd.joints[i].nq
        da += fmd.joints[i].nv
    var dof = List[Int]()
    var qadr = List[Int]()
    var jidx = List[Int]()
    var act_idx = List[Int]()
    var tmin = List[Float64]()
    var tmax = List[Float64]()
    for j in range(ARM_DOF):
        var ji = _index(fmd.joint_names, String("robot_joint") + String(j + 1))
        dof.append(dadr_all[ji])
        qadr.append(qadr_all[ji])
        jidx.append(ji)
        var ai = _index(fmd.actuator_names, String("robot_torq_j") + String(j + 1))
        act_idx.append(ai)
        tmin.append(fmd.actuators[ai].ctrl_min)
        tmax.append(fmd.actuators[ai].ctrl_max)
    var site = _index(fmd.site_names, String("robot_grip_site"))
    var site_body = fmd.sites[site].body_id
    var ga1 = _index(fmd.actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd.actuator_names, String("robot_gripper_finger_joint2"))
    var cfg = OscPoseConfig()
    var dof_j = dof.copy()
    var refs = build_osc_refs(
        dof^, qadr^, jidx^, tmin^, tmax^, act_idx.copy(), site, site_body,
        ga1, ga2,
        fmd.actuators[ga1].ctrl_min, fmd.actuators[ga1].ctrl_max,
        fmd.actuators[ga2].ctrl_min, fmd.actuators[ga2].ctrl_max,
        cfg.kp, cfg.damping_ratio, cfg.output_max_pos, cfg.output_max_ori,
        cfg.nullspace_kp, cfg.gripper_speed,
    )
    print("  scene :", scene_path(f), "| nq", LIBERO_GOAL_DIMS.NQ,
          "nv", LIBERO_GOAL_DIMS.NV)
    print("  control: frame skip", LIBERO_GOAL_FRAME_SKIP, "| action dim",
          OSC_ACTION_DIM, "| horizon", LiberoGoalOscConfig.MAX_STEPS)
    print("  grip site", site, "on body", site_body)
    print()

    var ctx = DeviceContext()
    var env = LiberoGoalOscEnv[N_ENVS](ctx)
    env.set_osc_refs(refs, ctx)
    env.reset_batch[N_ENVS](ctx, 1)
    ctx.synchronize()

    # ⚠ A ZERO ACTION IS `metric.py`'s `dummy`, and it is not "no torque": OSC
    # HOLDS the pose it was reset on, which takes gravity compensation plus the
    # nullspace term. So a near-zero speed here is the controller WORKING.
    # ⚠ THE ACTION BUFFER IS THE ABI'S, AND IT IS ALREADY ZERO. The env memsets
    # it at construction; writing zeros again would only prove the copy works.
    _ = env.action_ptr()

    # The observation STARTS with `qpos ++ qvel` (see LIBERO_GOAL_OBS_DIM), so
    # the arm's joint speeds are readable without any new accessor.
    #
    # ⚠ THE ROW WIDTH IS `LIBERO_GOAL_OBS_DIM`, NOT `NQ + NV`. It was `NQ + NV`
    # until the task hooks widened the observation to 91; after that this
    # buffer was 13 words per lane SHORT of the `_obs` copied into it, and every
    # lane past the first was read at the wrong stride.
    comptime OBS = LIBERO_GOAL_OBS_DIM
    var obs_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
    print("  step   mean |qvel_arm|   max |qvel_arm|   singular lanes")
    for s in range(n_steps):
        env.step_batch[N_ENVS](ctx, UInt64(s + 1))
        ctx.enqueue_copy(obs_h, env._obs)
        ctx.synchronize()
        var op = obs_h.unsafe_ptr()
        var tot = 0.0
        var mx = 0.0
        for e in range(N_ENVS):
            for j in range(ARM_DOF):
                var w = abs(Float64(
                    op[unsafe_offset = e * OBS + LIBERO_GOAL_DIMS.NQ + dof_j[j]]
                ))
                tot += w
                if w > mx:
                    mx = w
        var sing = env.osc_singular_lanes(ctx)
        print("  ", s, "   ", tot / Float64(N_ENVS * ARM_DOF), "   ", mx,
              "   ", sing)
        if sing > 0:
            # ⚠ NOT AN ERROR, AND NOT IGNORABLE. `osc_run_gpu` leaves a flagged
            # lane's torques at ZERO — a limp arm. A training driver should
            # reset that lane; a report that never printed this would show a
            # batch in which some lanes were not being controlled.
            print("     ⚠", sing, "of", N_ENVS,
                  "lanes have a singular operational-space inertia; their"
                  " torques are zero")
    print()
    print("=== " + String(n_steps) + " control steps x "
          + String(LIBERO_GOAL_FRAME_SKIP) + " substeps on " + String(N_ENVS)
          + " lanes ===")
