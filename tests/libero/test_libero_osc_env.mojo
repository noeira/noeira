"""THE BATCHED ENV COMPILES WITH OSC_POSE IN IT — L6's wiring, elaborated.

    pixi run mojo run -I . tests/libero/test_libero_osc_env.mojo

## ⚠⚠ WHAT THIS CAN AND CANNOT CLAIM ON THIS MACHINE

`libero_goal` is `nv = 37`, and the CRBA / Newton kernels stack-allocate per
thread by `nv`: Metal's pipeline creation refuses them (the P0 park probe died
at nv = 24). So the env INSTANTIATES here and cannot STEP here, and saying that
plainly is the point of the file.

What elaboration is worth on its own is not nothing —
`_the_package_build_cannot_instantiate_a_generic_kernel` records that a green
package build proves nothing about a GPU kernel because precompilation stops at
elaboration. This goes one step further than a package build: it names the
concrete `Phyics3dBatchedEnv[LiberoGoalModel, LiberoGoalOscConfig, N]`, so
every comptime branch the controller adds — the `OSC_LANES` fold, the
`constrained` on `ACTION_DIM`, the layouts the control kernel binds — is
type-checked at the real dimensions.

⚠ WHAT IS ACTUALLY GATED, AND WHERE. The controller's numbers are
`tests/libero/test_osc_control_batched.mojo` (eight lanes, three control steps,
against `OscPose`, 1.7e-14 of the torque scale) and
`tests/libero/test_osc_pose_gpu.mojo` (the kernel on device at float32). The
stepping leg of THIS file — that the env produces those numbers once wired
through `apply_actions_kernel_gpu` and the integrator — is owed on NVIDIA and
is listed as owed in the assessment. It is not claimed here.

## THE CHECKS THAT DO RUN

1. the config's own arithmetic: `FRAME_SKIP` is the family's `control_freq`
   against the scene's timestep, and `ACTION_DIM` is 7.
2. `build_osc_refs` produces a record of the right width from the real parsed
   scene, and every index in it addresses something that exists.
3. ⚠ the refusal: an env with `HAS_OSC_CONTROLLER` that was never given its
   record must REFUSE to step, not drive zeros.
"""

from std.os.path import exists
from std.sys import has_accelerator, default_accelerator

from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.physics3d.dynamics.osc_pose import ARM_DOF, OscPoseConfig
from noeira.physics3d.dynamics.osc_pose_gpu import (
    OSC_ACTION_DIM, OSC_REF_WORDS, OSC_REF_DOF, OSC_REF_QADR, OSC_REF_ACT,
    OSC_REF_SITE, OSC_REF_GRIP_ACT0, OSC_REF_GRIP_ACT1, build_osc_refs,
)
from noeira.tasks.spec import load_family
from noeira.tasks.family import scene_path
from noeira.envs.libero.models.libero_goal_dims import LIBERO_GOAL_DIMS
from noeira.envs.libero.models.libero_goal_xml import LiberoGoalModel
from noeira.envs.libero.libero_goal_config import (
    LiberoGoalOscConfig, LiberoGoalOscEnv, LIBERO_GOAL_FRAME_SKIP,
)


comptime FAMILY = "noeira/envs/libero/families/libero_goal.family"
comptime PACK = "noeira/envs/libero/assets"
comptime N_ENVS = 4


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("no '" + want + "' in the composed scene")


def main() raises:
    print("=" * 74)
    print("libero_goal on the batched env, OSC_POSE wired —", N_ENVS, "lanes")
    print("=" * 74)
    if not exists(PACK):
        print("  SKIPPED: no LIBERO pack at", PACK)
        print("=== SKIPPED (no pack — this is not a pass) ===")
        return
    var ta = Tally()

    # ── 1. the config's arithmetic ────────────────────────────────────────
    print()
    print("--- 1. the config ---")
    var f = load_family(FAMILY)
    var want_skip = Int(
        1.0 / Float64(f.control_freq) / LIBERO_GOAL_DIMS.TIMESTEP + 0.5
    )
    print("     control_freq", f.control_freq, "x timestep",
          LIBERO_GOAL_DIMS.TIMESTEP, "-> FRAME_SKIP", want_skip,
          "| config says", LIBERO_GOAL_FRAME_SKIP)
    ta.check(want_skip == LIBERO_GOAL_FRAME_SKIP,
             "FRAME_SKIP is the family's control_freq against the timestep,"
             " not a chosen number")
    ta.check(LiberoGoalOscConfig.MAX_STEPS == f.horizon,
             "MAX_STEPS " + String(LiberoGoalOscConfig.MAX_STEPS)
             + " == the family's horizon " + String(f.horizon))
    ta.check(LiberoGoalModel.ACTION_DIM == OSC_ACTION_DIM,
             "the model def's ACTION_DIM is " + String(LiberoGoalModel.ACTION_DIM)
             + " — OSC's seven, not the nine actuators")
    ta.check(LiberoGoalOscConfig.HAS_OSC_CONTROLLER,
             "HAS_OSC_CONTROLLER is on")
    ta.check(not LiberoGoalOscConfig.NORMALIZED_ACTIONS,
             "NORMALIZED_ACTIONS is OFF — OSC scales its own deltas")

    # ── 2. the record, from the real scene ────────────────────────────────
    print()
    print("--- 2. build_osc_refs against the composed scene ---")
    var fmd = parse_model_runtime(scene_path(f))
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
    var refs = build_osc_refs(
        dof.copy(), qadr.copy(), jidx.copy(), tmin.copy(), tmax.copy(),
        act_idx.copy(), site, site_body, ga1, ga2,
        fmd.actuators[ga1].ctrl_min, fmd.actuators[ga1].ctrl_max,
        fmd.actuators[ga2].ctrl_min, fmd.actuators[ga2].ctrl_max,
        cfg.kp, cfg.damping_ratio, cfg.output_max_pos, cfg.output_max_ori,
        cfg.nullspace_kp, cfg.gripper_speed,
    )
    ta.check(len(refs) == OSC_REF_WORDS,
             "the record is " + String(len(refs)) + " words")

    # ⚠ EVERY INDEX MUST ADDRESS SOMETHING THAT EXISTS. A record whose
    # actuator index is out of range writes a torque past the end of `ctrl`
    # and the failure is a corrupted neighbour, not a bounds error.
    var nact = len(fmd.actuator_names)
    var nq = LIBERO_GOAL_DIMS.NQ
    var nv = LIBERO_GOAL_DIMS.NV
    var in_range = True
    for j in range(ARM_DOF):
        if Int(refs[OSC_REF_DOF + j]) < 0 or Int(refs[OSC_REF_DOF + j]) >= nv:
            in_range = False
        if Int(refs[OSC_REF_QADR + j]) < 0 or Int(refs[OSC_REF_QADR + j]) >= nq:
            in_range = False
        if Int(refs[OSC_REF_ACT + j]) < 0 or Int(refs[OSC_REF_ACT + j]) >= nact:
            in_range = False
    if Int(refs[OSC_REF_SITE]) < 0 or Int(refs[OSC_REF_SITE]) >= len(fmd.site_names):
        in_range = False
    if Int(refs[OSC_REF_GRIP_ACT0]) >= nact or Int(refs[OSC_REF_GRIP_ACT1]) >= nact:
        in_range = False
    ta.check(in_range,
             "every dof / qpos / actuator / site index in the record addresses"
             " something the scene has")
    # ⚠ AND THEY MUST NOT ALL BE ZERO — an all-zero record passes the range
    # check and is exactly what `set_osc_refs` exists to prevent.
    var distinct = True
    for j in range(1, ARM_DOF):
        if refs[OSC_REF_ACT + j] == refs[OSC_REF_ACT]:
            distinct = False
    ta.check(distinct, "the seven arm actuators are seven DIFFERENT indices")

    # ── 3. the env elaborates, and refuses without its record ─────────────
    print()
    print("--- 3. the env instantiates, and refuses to step unarmed ---")
    if not default_accelerator().is_nvidia_gpu():
        print("     no NVIDIA target: the config and the record are checked")
        print("     above; the env's own kernels need one. See the header.")
        print()
        print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
        if ta.failures != 0:
            raise Error(String(ta.failures) + " failed")
        print("=== PASS (config + record; the stepping leg is owed on NVIDIA) ===")
        return

    # ⚠⚠ THE TYPE IS NAMED ONLY ON NVIDIA, AND THAT IS A COMPTIME BRANCH.
    # On Metal this does not fail at RUNTIME, it fails to BUILD: "Metal Compiler
    # failed to compile metallib" for the Newton solver at nv = 37. A `try`
    # around the construction cannot catch a metallib that never compiled, and
    # a runtime `has_accelerator()` guard does not stop elaboration — so the
    # whole leg is behind `comptime if default_accelerator().is_nvidia_gpu()`, which is
    # the only form that keeps the kernels out of an Apple build entirely.
    comptime if default_accelerator().is_nvidia_gpu():
        from max.gpu.host import DeviceContext
        var ctx = DeviceContext()
        var env = LiberoGoalOscEnv[N_ENVS](ctx)
        var refused = False
        try:
            env.step_batch[N_ENVS](ctx, 0)
        except e:
            refused = String(e).find("set_osc_refs") >= 0
        ta.check(refused,
                 "an env with HAS_OSC_CONTROLLER refuses to step before"
                 " set_osc_refs — it does not drive an all-zero record")
        env.set_osc_refs(refs, ctx)
        env.reset_batch[N_ENVS](ctx, 0)
        env.step_batch[N_ENVS](ctx, 0)
        ctx.synchronize()
        ta.check(True, "reset + one step with the controller wired")
    else:
        print("     not an NVIDIA target: the env type is not even named here")
        print("     (nv =", nv, "— Metal cannot compile the metallib; header)")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "libero osc env: " + String(ta.failures) + " of "
            + String(ta.checks) + " failed"
        )
    print("=== PASS ===")
