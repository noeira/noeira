"""A pixel student in the SIM, headless, printing what the real deploy prints.

    pixi run mojo build -I . -D DAGGER_PX_32 -D DAGGER_JOINT_VEL -o /tmp/px_probe \\
        examples/so101/pixel_student_probe_sim.mojo
    /tmp/px_probe --ckpt projects/so101-tower/policies/pixel_bowl.ckpt [--task T] [--seed S] [--episodes N]
        [--record ticks.csv]      # the deploy's --record columns, per episode
        [--lag-tau 140,140 --lag-delay 2,2]   # the real servos' response (ServoLag)
        [--lag-vmax-j "...;0.8,1.2" --lag-offset-j ... --lag-period 32,38]   # per joint

The reference a real run is read against. It starts every episode where the
real deploy's ramp leaves the arm — the task's reset pose, props placed by
the host sampler at `--seed` — and drives the student exactly as the deploy
does (`pixel_student.render_to_planes` of the trainer's traces instead of a
camera frame; joints from `qpos`, velocities from `qvel`; the delta rule), on
the CPU env. ⚠ The joints and the scene come from the env's `Data`, not its
observation: without the task's `meta` words (this probe writes none) the
family's observation hook ZEROES the props' slots, and a render of that
"qpos" has no brick in it (the first version of this probe failed 0 / 3 so).
Every 15 steps it prints `t=` and the six executed action words
in the deploy's format, so a real log and a sim log line up; at the end, the
brick's rise, its horizontal gap to the bowl, and whether `Near` held.

⚠ Built with the student's defines (the manifest beside `--ckpt` is checked).
"""

from std.os.path import exists
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.core.cont_action import ContAction
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import load_params
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.physics3d.fields import Data
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.delta_action import delta_target, target_step, ServoLag, TARGET_OBS
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.pixel_student import (
    StudentNet, N_CAMS, IN_DIM, ACT, RENDER, OVERHEAD_RENDER_W,
    OVERHEAD_RENDER_H, OBS_PX, JOINT_VEL, WINDOWED, check_pixel_manifest,
    render_to_planes, joints_to_planes, joint_vels_to_planes, student_act,
    HIST_WORDS, act_hist_push, act_hist_to_planes, target_lead_to_planes,
)
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, make_tower_model, make_tower_renderer, tower_cameras,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.utils.fmt import col, fixed, pad_left

comptime E = Phyics3dEnv[So101TowerModel, So101TowerConfig, DT, False]
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime NB = So101TowerModel.NBODY


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var ckpt = _arg(args, "--ckpt", "projects/so101-tower/policies/pixel_bowl.ckpt")
    var task = _arg(args, "--task", "so101_tower_cube_in_bowl")
    var seed0 = Int(_arg(args, "--seed", "1"))
    var episodes = Int(_arg(args, "--episodes", "3"))
    var rec_path = _arg(args, "--record", "")
    # ⚠ A REAL SCENE, REPRODUCED: `--arm-pose` (a deploy snap's pose.txt,
    # model radians) replaces the reset's arm joints, `--brick x,y` /
    # `--bowl x,y` (world metres, e.g. from a real overhead frame through
    # `tools/so101/sim_prop_pixels.mojo`'s homography) the props' positions
    var arm_pose = _arg(args, "--arm-pose", "")
    var brick_xy = _arg(args, "--brick", "")
    var bowl_xy = _arg(args, "--bowl", "")

    var rec_csv = String("ep,t_s,q0,q1,q2,q3,q4,q5,qd0,qd1,qd2,qd3,qd4,qd5,a0,a1,a2,a3,a4,a5,tgt0,tgt1,tgt2,tgt3,tgt4,tgt5\n")
    var man_path = String(ckpt[byte = 0 : ckpt.byte_length() - 5]) + ".norm.json"
    if not exists(man_path):
        man_path = ckpt[byte = 0 : ckpt.rfind("/")] + "/norm.json"
    var man = check_pixel_manifest(man_path)
    var lag_tau_s = _arg(args, "--lag-tau", "")
    # added to the GRIPPER angle the student sees (the deploy's --grip-offset):
    # -0.05 plays the real jaws, which read ~0.05 rad more closed on the cube
    var grip_off = Float64(_arg(args, "--grip-offset", "0"))
    # the deploy's --act-ema: arm action words executed as (1 - A) new + A previous
    var act_ema = Float64(_arg(args, "--act-ema", "0"))
    # overrides of the manifest's action scales — for what-ifs only
    var da_o = _arg(args, "--delta-arm", "")
    var dg_o = _arg(args, "--delta-gripper", "")
    if da_o.byte_length() > 0:
        man.delta_arm = Float64(da_o)
    if dg_o.byte_length() > 0:
        man.delta_gripper = Float64(dg_o)
    print("probe: action scale arm", man.delta_arm, "gripper", man.delta_gripper)
    var lag_delay_s = _arg(args, "--lag-delay", "")
    print("probe:", ckpt, "|", OBS_PX, "px", "| q+qd" if JOINT_VEL else "| q",
          "| window" if WINDOWED else "| centre square", "| task", task)

    var ctx = DeviceContext()
    var f = load_family("noeira/tasks/families/so101_tower.family")
    var fmd = parse_model_runtime(scene_path(f))
    var cams = tower_cameras(fmd)
    var rm = make_tower_model(ctx)
    var rd = Data[RIG_DT, TOWER_MD, 1]()
    var r_w = make_tower_renderer[1, RENDER, RENDER, 1](ctx, fmd, rm)
    var r_o = make_tower_renderer[1, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, 1](ctx, fmd, rm)
    var jadr = List[Int]()
    var jdadr = List[Int]()
    var a1 = 0
    var a2 = 0
    for i in range(len(fmd.joints)):
        jadr.append(a1)
        jdadr.append(a2)
        a1 += fmd.joints[i].nq
        a2 += fmd.joints[i].nv
    var qa = List[Int]()
    var da = List[Int]()
    var lo = List[Float64]()
    var hi = List[Float64]()
    for i in range(ACT):
        qa.append(jadr[fmd.actuators[i].joint_id])
        da.append(jdadr[fmd.actuators[i].joint_id])
        lo.append(fmd.actuators[i].ctrl_min)
        hi.append(fmd.actuators[i].ctrl_max)
    var brick = -1
    var bowl = -1
    for b in range(len(fmd.body_names)):
        if String(fmd.body_names[b]) == "brick_brick":
            brick = b
        if String(fmd.body_names[b]) == "bowl_bowl":
            bowl = b

    var net = StudentNet.make["cpu", Kaiming](None)
    load_params["cpu"](net, ckpt, None)
    var x = Tensor.alloc(IN_DIM)
    var y = Tensor.alloc(ACT)
    var xs = List[Scalar[DT]](length=IN_DIM, fill=Scalar[DT](0))
    var env = E(ctx)
    var lag = ServoLag.parse(1, lag_tau_s, lag_delay_s, man.control_period_s)
    # the real arm's speed cap (rad/s, "v" or "lo,hi": the probe takes the
    # middle) and elbow stop — `ServoLag.set_limits`
    lag.set_limits(_arg(args, "--lag-vmax", ""), Float64(_arg(args, "--elbow-max", "0")))
    # per-joint ranges ("lo,hi;..." x6, the gripper last), signed offsets and
    # the control period's range — `ServoLag.set_per_joint` / `set_offset` /
    # `set_period`, as in the state probe
    lag.set_per_joint(
        _arg(args, "--lag-tau-j", ""), _arg(args, "--lag-delay-j", ""),
        _arg(args, "--lag-vmax-j", ""),
    )
    lag.set_offset(_arg(args, "--lag-offset-j", ""))
    lag.set_period(_arg(args, "--lag-period", ""))
    if lag.on:
        print("probe: servo lag tau", lag_tau_s, "ms, delay", lag_delay_s,
              "ticks | arm speed cap", lag.vmax_lo, "-", lag.vmax_hi,
              "rad/s | elbow max", lag.elbow_max)
    var n_ok = 0
    for ep in range(episodes):
        _ = env.reset()
        var q0 = posed_qpos[So101TowerPlacement](
            task, String("so101_tower"), So101TowerConfig.SLOT_RADIUS,
            seed=UInt64(seed0 + ep),
        )
        if arm_pose.byte_length() > 0:
            var txt = String("")
            with open(arm_pose, "r") as fh:
                txt = fh.read()
            var parts = String(txt.strip()).split(" ")
            for j in range(ACT):
                q0[qa[j]] = Float64(String(parts[j]))
        # free slots: 0 = bowl, 1 = brick (the family's slot order)
        if bowl_xy.byte_length() > 0:
            var p = bowl_xy.split(",")
            q0[So101TowerPlacement.free_qadr(0)] = Float64(String(p[0]))
            q0[So101TowerPlacement.free_qadr(0) + 1] = Float64(String(p[1]))
        if brick_xy.byte_length() > 0:
            var p = brick_xy.split(",")
            q0[So101TowerPlacement.free_qadr(1)] = Float64(String(p[0]))
            q0[So101TowerPlacement.free_qadr(1) + 1] = Float64(String(p[1]))
        var v0 = List[Float64](length=NV, fill=0.0)
        var obs = env.obs_at(q0, v0)
        var z0 = Float64(env.d.xpos.data[brick * 3 + 2]) if brick >= 0 else 0.0
        var rise = 0.0
        var held = False
        print("── episode", ep, "(placement seed", seed0 + ep, ")")
        var line0 = String("   start q:")
        for j in range(ACT):
            line0 += " " + col(Float64(obs.data[qa[j]]), 7, 3)
        print(line0)
        var at_lim = 0
        var q_start = List[Float64](length=ACT, fill=0.0)
        for j in range(ACT):
            q_start[j] = Float64(env.d.qpos.data[qa[j]])
        lag.reset_lane(0, q_start, 0, 0.5, 0.5)
        var hist = List[Float64](length=HIST_WORDS, fill=0.0)
        var q = List[Float64](length=ACT, fill=0.0)
        var qd = List[Float64](length=ACT, fill=0.0)
        var a_ex = List[Float64](length=ACT, fill=0.0)
        var tgt_hold = List[Float64](length=ACT, fill=0.0)
        # target mode (`man.action == "target"`): the last commanded target,
        # starting on the joints; its lead over them is the student's input
        var tprev = q_start.copy()
        var lead = List[Float64](length=TARGET_OBS, fill=0.0)
        var line = String("")
        var ra = String("")
        for t in range(So101TowerConfig.MAX_STEPS):
            # the policy acts every `repeat` ticks (the manifest's): render,
            # forward, new targets; between, the targets are HELD and the
            # servo model runs per tick
            if t % man.repeat == 0:
                for k in range(NQ):
                    rd.qpos.data[k] = Scalar[RIG_DT](env.d.qpos.data[k])
                forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
                for k in range(N_CAMS):
                    var rgb = List[Scalar[RIG_DT]]()
                    var dd = List[Scalar[RIG_DT]]()
                    var ss = List[Scalar[RIG_DT]]()
                    if N_CAMS == 2 and k == 0:
                        r_o.cam = cams[k]
                        r_o.render_cpu(rd, rm, rgb, dd, ss)
                        render_to_planes(rgb, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, k, xs)
                    else:
                        r_w.cam = cams[k]
                        r_w.render_cpu(rd, rm, rgb, dd, ss)
                        render_to_planes(rgb, RENDER, RENDER, k, xs)
                for j in range(ACT):
                    # ⚠ FROM THE ENV'S Data, as the trainer packs them (the
                    # observation's words are what the STATE policy reads)
                    q[j] = Float64(env.d.qpos.data[qa[j]])
                    qd[j] = Float64(env.d.qvel.data[da[j]])
                if t < 2:
                    var dq = 0.0
                    var dv = 0.0
                    for k in range(NQ):
                        dq = max(dq, abs(Float64(obs.data[k]) - Float64(env.d.qpos.data[k])))
                    for k in range(NV):
                        dv = max(dv, abs(Float64(obs.data[NQ + k]) - Float64(env.d.qvel.data[k])))
                    var lv = String("   t=") + String(t) + " max|obs-qpos| " + fixed(dq, 5) + " max|obs-qvel| " + fixed(dv, 5) + "  qd:"
                    for j in range(ACT):
                        lv += " " + col(qd[j], 6, 2)
                    print(lv)
                    if t == 0:
                        var lo_ = String("   obs  [0:NQ]:")
                        var ld_ = String("   qpos [0:NQ]:")
                        for k in range(NQ):
                            lo_ += " " + fixed(Float64(obs.data[k]), 3)
                            ld_ += " " + fixed(Float64(env.d.qpos.data[k]), 3)
                        print(lo_)
                        print(ld_)
                var q_pol = q.copy()
                q_pol[ACT - 1] += grip_off
                joints_to_planes(q_pol, xs)
                joint_vels_to_planes(qd, xs)
                act_hist_to_planes(hist, xs)
                for j in range(TARGET_OBS):
                    lead[j] = tprev[j] - q[j]
                target_lead_to_planes(lead, xs)
                for k in range(IN_DIM):
                    x.data[k] = xs[k]
                net.forward["cpu", 1](TensorRefs[1](x), y, None)
                line = String("")
                ra = String("")
                for j in range(ACT):
                    var a = Float64(student_act(y.data[j], j, man.gripper_sign))
                    if act_ema > 0.0 and j < ACT - 1:
                        a = (1.0 - act_ema) * a + act_ema * a_ex[j]
                    a_ex[j] = a
                    if man.action == "target":
                        tgt_hold[j] = target_step(
                            tprev[j], q[j], a, j, lo[j], hi[j], man.delta_arm,
                            man.delta_gripper, man.target_lead,
                        )
                        tprev[j] = tgt_hold[j]
                    else:
                        tgt_hold[j] = delta_target(q[j], a, j, lo[j], hi[j], man.delta_arm, man.delta_gripper)
                    ra += "," + String(a)
                    line += " " + col(a, 6, 2)
                act_hist_push(hist, a_ex)
            var act = ContAction[ACT]()
            var rt = String("")
            for j in range(ACT):
                var tgt = lag.apply(0, j, tgt_hold[j])
                # the COMMANDED target, as the deploy's --record logs it (the
                # lagged one is what the actuator gets)
                rt += "," + String(tgt_hold[j])
                if tgt <= lo[j] or tgt >= hi[j]:
                    at_lim += 1
                var mid = 0.5 * (lo[j] + hi[j])
                var half = 0.5 * (hi[j] - lo[j])
                act.data[j] = (tgt - mid) / half
            if rec_path.byte_length() > 0:
                var row = String(ep) + "," + String(Float64(t) * man.control_period_s)
                for j in range(ACT):
                    row += "," + String(Float64(env.d.qpos.data[qa[j]]))
                for j in range(ACT):
                    row += "," + String(Float64(env.d.qvel.data[da[j]]))
                rec_csv += row + ra + rt + "\n"
            if t % 15 == 0:
                print("  t=" + pad_left(fixed(Float64(t) * man.control_period_s, 1), 5)
                      + "s  a:" + line)
            lag.advance()
            var res = env.step(act)
            obs = res[0].copy()
            if brick >= 0:
                var zb = Float64(env.d.xpos.data[brick * 3 + 2])
                if zb - z0 > rise:
                    rise = zb - z0
                if bowl >= 0:
                    var ex = Float64(env.d.xpos.data[brick * 3] - env.d.xpos.data[bowl * 3])
                    var ey = Float64(env.d.xpos.data[brick * 3 + 1] - env.d.xpos.data[bowl * 3 + 1])
                    var ez = Float64(env.d.xpos.data[brick * 3 + 2] - env.d.xpos.data[bowl * 3 + 2])
                    if (ex * ex + ey * ey + ez * ez) ** 0.5 < 0.045:
                        held = True
        if held:
            n_ok += 1
        print("   brick max rise", fixed(rise * 1000.0, 1), "mm | Near held:", held,
              "| targets at a joint limit", at_lim)
    print("probe:", n_ok, "/", episodes, "episodes reached the goal")
    if rec_path.byte_length() > 0:
        with open(rec_path, "w") as f:
            f.write(rec_csv)
        print("probe: wrote", rec_path)
