"""A STATE PPO policy in the SIM, headless, on a placement or a rebuilt real scene.

    pixi run mojo build -I . -D TASK_PPO_ACT_HIST=3 -D TASK_PPO_TARGET_OBS \\
        examples/so101/ppo_state_probe_sim.mojo -o build/ppo_state_probe
    build/ppo_state_probe --run runs/<ppo run dir> [--task T] [--seed S] [--episodes N]
        [--arm-pose "q0 q1 q2 q3 q4 q5" | FILE]  [--brick x,y] [--bowl x,y]
        [--lag-tau 50,50 --lag-delay 2,2] [--lag-vmax 1.1,1.1] [--elbow-max 1.56]
        [--lag-tau-j "lo,hi;..." --lag-delay-j ... --lag-vmax-j ...] [--lag-offset-j "lo,hi;..."] [--lag-period lo,hi]

The state teacher's counterpart of `pixel_student_probe_sim.mojo`, for the
same gate: the real scene rebuilt (`--arm-pose` from the run's pose_0,
`--brick` / `--bowl` from the overhead frame through the desk-plane
homography), several fixed servo-lag variants as the repeats. The policy
reads the EXACT sim state — this measures the policy under the real arm's
dynamics on the real layout, not perception (the pose estimator and its
noise come later).

Everything the action needs is read from the run's `metrics.config.kv`
(`action` delta|target, `delta_arm`, `delta_gripper`, `target_lead`,
`repeat`), and the build must match the run's observation (`act_hist`,
`target_obs`): it refuses otherwise, since a policy fed a differently laid
out observation still acts — wrongly, and silently.

⚠ THE OBSERVATION IS THE DRIVER'S. The task's `meta` words (goal tape,
active mask, shaping) and the family's region table are written into the env
before the episode, as `ppo_family_driver.run_ppo` does per lane — without
them the observation hook ZEROES the props' slots (the pixel probe's first
version failed 0 / 3 so). Checked at the start of every episode: the brick
and bowl words of the observation must equal their `qpos`.
"""

from std.os.path import exists
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.core.cont_action import ContAction
from noeira.deep_agents.ppo import PPOAgent
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.delta_action import (
    DELTA_ACT, ACT_HIST, TARGET_OBS, ServoLag, delta_target, target_step,
)
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.host_reward import family_reward_host
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.ppo_family_driver import ActorNet, CriticNet, RunningMeanStd, OBS_CLIP
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.utils.fmt import col, fixed

comptime E = Phyics3dEnv[So101TowerModel, So101TowerConfig, DT, False]
comptime ACT = DELTA_ACT
comptime E_OBS = So101TowerModel.OBS_DIM
comptime W = ACT_HIST * ACT
comptime OBS = E_OBS + W + TARGET_OBS
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime TeacherT = PPOAgent["cpu", ActorNet[OBS], CriticNet[OBS], OBS, ACT, 16, 16, 1, 1]


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def _cfg(path: String, key: String, default: String) raises -> String:
    """`key=` from a run's `metrics.config.kv`, or `default`."""
    with open(path, "r") as f:
        for ln in f.read().split("\n"):
            var s = String(ln)
            if s.startswith(key + "="):
                return String(s[byte = key.byte_length() + 1 :])
    return default


def _pose_words(spec: String) raises -> List[Float64]:
    """Six model-radian joints, inline ("q0 q1 ..." or comma-separated) or
    from a file (a deploy snap's pose.txt)."""
    var txt = spec
    if exists(spec):
        with open(spec, "r") as fh:
            txt = fh.read()
    var out = List[Float64]()
    for p in String(txt.strip()).replace(",", " ").split(" "):
        var s = String(p.strip())
        if s.byte_length() > 0:
            out.append(Float64(s))
    if len(out) != ACT:
        raise Error("probe: --arm-pose needs 6 joints, got " + String(len(out)))
    return out^


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var run = _arg(args, "--run", "")
    if run.byte_length() == 0:
        raise Error("probe: --run <ppo run dir> is required")
    var task = _arg(args, "--task", "so101_tower_cube_in_bowl")
    var seed0 = Int(_arg(args, "--seed", "1"))
    var episodes = Int(_arg(args, "--episodes", "4"))
    var arm_pose = _arg(args, "--arm-pose", "")
    var brick_xy = _arg(args, "--brick", "")
    var bowl_xy = _arg(args, "--bowl", "")
    var quiet = _arg(args, "--quiet", "0") == "1"
    var debug = _arg(args, "--debug", "0") == "1"

    # ── the run: its action and its observation layout ──────────────────
    var cfg = run + "/metrics.config.kv"
    var mode = _cfg(cfg, "action", "absolute")
    var d_arm = Float64(_cfg(cfg, "delta_arm", "0.05"))
    var d_grip = Float64(_cfg(cfg, "delta_gripper", "0.2"))
    var lead = Float64(_cfg(cfg, "target_lead", "0"))
    var repeat = Int(_cfg(cfg, "repeat", "1"))
    var r_hist = Int(_cfg(cfg, "act_hist", "0"))
    var r_tobs = Int(_cfg(cfg, "target_obs", "0"))
    if mode != "delta" and mode != "target":
        raise Error("probe: the run's action is " + mode + "; delta|target only")
    if r_hist != ACT_HIST or r_tobs != TARGET_OBS:
        raise Error(
            "probe: the run sees " + String(r_hist) + " past actions + "
            + String(r_tobs) + " target words; this build " + String(ACT_HIST)
            + " + " + String(TARGET_OBS) + " (-D TASK_PPO_ACT_HIST / -D TASK_PPO_TARGET_OBS)"
        )
    var period = Float64(So101TowerConfig.FRAME_SKIP) * So101TowerModel.TIMESTEP
    print("probe:", run, "| task", task, "| action", mode, "| scales", d_arm,
          "/", d_grip, "| lead", lead, "| repeat", repeat, "| obs", OBS)

    var teacher = TeacherT()
    teacher.trainer.load_state(run + "/checkpoints/last.ckpt")
    var obs_rms = RunningMeanStd(OBS)
    obs_rms.load(run + "/obs_norm.txt")

    # ── the family: actuators, props, region table, the task's meta words ─
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var jadr = List[Int]()
    var acc = 0
    for i in range(len(fmd.joints)):
        jadr.append(acc)
        acc += fmd.joints[i].nq
    var qa = List[Int]()
    var lo = List[Float64]()
    var hi = List[Float64]()
    for i in range(ACT):
        qa.append(jadr[fmd.actuators[i].joint_id])
        lo.append(fmd.actuators[i].ctrl_min)
        hi.append(fmd.actuators[i].ctrl_max)
    var brick = -1
    for b in range(len(fmd.body_names)):
        if String(fmd.body_names[b]) == "brick_brick":
            brick = b
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    var mw = task_meta_words(
        task, String(FAMILY), So101TowerConfig.SHAPE_W_GOAL,
        So101TowerConfig.SHAPE_W_REACH, So101TowerConfig.GOAL_MARGIN,
        So101TowerConfig.REACH_MARGIN,
    )

    var ctx = DeviceContext()
    var env = E(ctx)
    for i in range(len(cw)):
        env.mf.curriculum.data[i] = Scalar[DT](cw[i])
    var lag = ServoLag.parse(1, _arg(args, "--lag-tau", ""), _arg(args, "--lag-delay", ""), period)
    lag.set_limits(_arg(args, "--lag-vmax", ""), Float64(_arg(args, "--elbow-max", "0")))
    lag.set_per_joint(
        _arg(args, "--lag-tau-j", ""), _arg(args, "--lag-delay-j", ""),
        _arg(args, "--lag-vmax-j", ""),
    )
    lag.set_offset(_arg(args, "--lag-offset-j", ""))
    lag.set_period(_arg(args, "--lag-period", ""))
    if lag.on:
        print("probe: servo lag tau", _arg(args, "--lag-tau", ""), "ms, delay",
              _arg(args, "--lag-delay", ""), "ticks | arm speed cap",
              lag.vmax_lo, "-", lag.vmax_hi, "rad/s | elbow max", lag.elbow_max)

    var raw = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var nrm = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var ao = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    var n_ok = 0
    var n_empty_first = 0
    for ep in range(episodes):
        _ = env.reset()
        for k in range(len(mw[0])):
            env.d.meta.data[mw[0][k]] = Scalar[DT](mw[1][k])
        var q0 = posed_qpos[So101TowerPlacement](
            task, String(FAMILY), So101TowerConfig.SLOT_RADIUS,
            seed=UInt64(seed0 + ep),
        )
        if arm_pose.byte_length() > 0:
            var p = _pose_words(arm_pose)
            for j in range(ACT):
                q0[qa[j]] = p[j]
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
        # ⚠ the props must be IN the observation (see the header)
        for s in range(2):
            var a0 = So101TowerPlacement.free_qadr(s)
            for k in range(2):
                var o = Float64(obs.data[a0 + k])
                var qv = Float64(env.d.qpos.data[a0 + k])
                if abs(o - qv) > 1e-4 or abs(qv) < 1e-6:
                    raise Error(
                        "probe: obs[" + String(a0 + k) + "] = " + String(o)
                        + " but qpos " + String(qv) + " — the props are not"
                        " in the observation (meta words?)"
                    )
        var q = List[Float64](length=ACT, fill=0.0)
        for j in range(ACT):
            q[j] = Float64(env.d.qpos.data[qa[j]])
        lag.reset_lane(0, q, 0, 0.5, 0.5)
        var tprev = q.copy()
        var tg = q.copy()
        var hist = List[Float64](length=W, fill=0.0)
        var z0 = Float64(env.d.xpos.data[brick * 3 + 2])
        var rise = 0.0
        var held = False
        var t_held = -1
        var flips = 0
        # ⚠ EMPTY CLOSES: the jaw fully shut (< -0.10 rad: on the brick it
        # stalls near +0.10) after having opened (> 0.20), counted once per
        # close; "first close empty" = one happened before the brick rose 2 cm
        var jaw_open = False
        var n_empty = 0
        var t_first_empty = -1
        var lifted = False
        var a_last = List[Float64](length=ACT, fill=0.0)
        if not quiet:
            var l0 = String("── episode ") + String(ep) + " start q:"
            for j in range(ACT):
                l0 += " " + col(q[j], 7, 3)
            print(l0)
        for t in range(So101TowerConfig.MAX_STEPS):
            if t % repeat == 0:
                for j in range(ACT):
                    q[j] = Float64(env.d.qpos.data[qa[j]])
                for k in range(E_OBS):
                    raw[k] = Scalar[DT](obs.data[k])
                for k in range(W):
                    raw[E_OBS + k] = Scalar[DT](hist[k])
                comptime if TARGET_OBS > 0:
                    for j in range(ACT):
                        raw[E_OBS + W + j] = Scalar[DT](tprev[j] - q[j])
                obs_rms.normalize_into(
                    mptr(raw.unsafe_ptr()), mptr(nrm.unsafe_ptr()), 1, OBS, OBS_CLIP
                )
                if debug and t == 0:
                    for k in range(OBS):
                        print("   obs", k, "raw", fixed(Float64(raw[k]), 4),
                              "mean", fixed(obs_rms.mean[k], 4), "std",
                              fixed(obs_rms.var_[k] ** 0.5, 4), "norm",
                              fixed(Float64(nrm[k]), 2))
                teacher.trainer.select_greedy_action(nrm, ao)
                var line = String("")
                for j in range(ACT):
                    var a = Float64(ao[j])
                    a = 1.0 if a > 1.0 else (-1.0 if a < -1.0 else a)
                    if j < ACT - 1 and t > 0 and a * a_last[j] < 0.0:
                        flips += 1
                    a_last[j] = a
                    if mode == "target":
                        tg[j] = target_step(tprev[j], q[j], a, j, lo[j], hi[j], d_arm, d_grip, lead)
                        tprev[j] = tg[j]
                    else:
                        tg[j] = delta_target(q[j], a, j, lo[j], hi[j], d_arm, d_grip)
                    line += " " + col(a, 6, 2)
                # the history, as the driver's `_hist_push`: newest first
                for k in range(W - 1, ACT - 1, -1):
                    hist[k] = hist[k - ACT]
                comptime if W > 0:
                    for j in range(ACT):
                        hist[j] = a_last[j]
                if not quiet and t % 30 == 0:
                    print("  t=" + fixed(Float64(t) * period, 1) + "s  a:" + line)
            var act = ContAction[ACT]()
            var act_l = List[Float64](length=ACT, fill=0.0)
            for j in range(ACT):
                var u = lag.apply(0, j, tg[j])
                var mid = 0.5 * (lo[j] + hi[j])
                var half = 0.5 * (hi[j] - lo[j])
                act.data[j] = (u - mid) / half
                act_l[j] = Float64(act.data[j])
            lag.advance()
            var res = env.step(act)
            obs = res[0].copy()
            # ⚠ the CPU env's own reward hook is a constant zero (the family's
            # reward is a GPU kernel over the tape): the goal is the KERNEL's,
            # evaluated on the host (`tasks/host_reward.mojo`)
            var rdone = family_reward_host[So101TowerConfig, DT, E.MD, ACT](
                env.d, env.mf, act_l, t + 1, So101TowerConfig.FRAME_SKIP,
                So101TowerModel.TIMESTEP,
            )
            var zb = Float64(env.d.xpos.data[brick * 3 + 2])
            if zb - z0 > rise:
                rise = zb - z0
            if rise > 0.02:
                lifted = True
            var jaw = Float64(env.d.qpos.data[qa[ACT - 1]])
            if jaw > 0.20:
                jaw_open = True
            elif jaw < -0.10 and jaw_open:
                jaw_open = False
                n_empty += 1
                if t_first_empty < 0 and not lifted:
                    t_first_empty = t
            # the task's own goal word, as the driver's success count reads it
            if rdone[1]:
                if not held:
                    t_held = t
                held = True
        if held:
            n_ok += 1
        if t_first_empty >= 0:
            n_empty_first += 1
        var steps = So101TowerConfig.MAX_STEPS // repeat
        print("   episode", ep, "| brick max rise", fixed(rise * 1000.0, 1),
              "mm | goal held:", held, "at", fixed(Float64(t_held) * period, 2),
              "s | empty closes", n_empty, "(first at tick", t_first_empty,
              ") | arm sign flips per step", fixed(Float64(flips) / Float64(steps), 3))
    print("probe:", n_ok, "/", episodes, "episodes reached the goal |",
          n_empty_first, "closed EMPTY before the first lift")
