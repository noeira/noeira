# +--------------------------------------------------------------------------+ #
# | SuperMimic on Berkeley Humanoid Lite — the torque a checkpoint asks for
# +--------------------------------------------------------------------------+ #
"""Roll a checkpoint over library clips on the CPU env and record, per joint,
the torque BHL's PD law applies: what the tracking benchmark does not report.

    pixi run mojo build -I . -D SM_LIGHT=1 examples/bhl/supermimic_torque_log.mojo -o bhl_torque
    pixi run ./bhl_torque --ckpt <last.ckpt> --store <store> --every 10 --out torque.csv

Build it with the checkpoint's `-D SM_LIGHT` / `-D SM_PAPER` and the SAME
`-D BHL_EFFORT_PCT` / `-D BHL_SPEED_PCT` it was trained under.

The protocol is the benchmark's (Appendix F: each clip from its first frame
to its end, no noise, no domain randomisation). At every control step the
torque is `bhl_pd_torque` on the state BEFORE the step and the action the
policy gives: the first of the step's 10 substeps. Per joint, over all the
steps of all the clips run:

    rms          root mean square torque, N·m
    peak         largest |torque|, N·m
    at_effort    share of steps with |torque| within 1 % of the effort limit
    over_speed   share of steps with |joint speed| above the speed limit
    rms_speed    root mean square joint speed, rad/s

and per clip (`<out>.clips.csv`): steps, normalised return, failed, the
clip's own RMS torque over the leg joints.

⚠ CPU PHYSICS, float64, one lane: the checkpoint trained on the float32
batched env. Take tracking numbers from `supermimic_eval.mojo`; this file is
for the torques. ⚠ A store made from BONES-SEED is confidential.

Options
-------
--ckpt PATH   the checkpoint
--store PATH  the library store (or a directory / comma-separated list)
--every K     run every K-th clip (default 1: all)
--max-clips N stop after N clips run (0 = no cap)
--out PATH    the per-joint CSV (default torque.csv)
"""

from std.builtin.sort import sort
from std.math import abs, sqrt
from std.os import listdir
from std.os.path import isdir
from std.sys import argv, get_defined_int

from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.robot import BhlRobot, SMDims, TrackingSim
from noeira.envs.robots.berkeley_humanoid_lite_config import (
    BHL_EFFORT_PCT, BHL_SPEED_PCT, bhl_effort_limit, bhl_pd_target, bhl_pd_torque, bhl_speed_limit,
)
from noeira.envs.robots.berkeley_humanoid_lite_pd import BHL_N_DOF, bhl_dof_name
from noeira.io.fileio import write_text_atomic
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime P = get_defined_int["SM_PAPER", 0]() != 0
comptime LIGHT = get_defined_int["SM_LIGHT", 0]() != 0 and not P
comptime Robot = BhlRobot
# the training example's widths (a checkpoint loads only into its own), one lane
comptime C = SMConfig[
    "cpu", Robot,
    2048 if P else (1024 if LIGHT else 128), 64,
    2048 if P else (1024 if LIGHT else 128), 1024 if P else (512 if LIGHT else 64),
    512 if P else (256 if LIGHT else 64), 256 if P else (128 if LIGHT else 32),
    2048 if P else (1024 if LIGHT else 128), 1024 if P else (512 if LIGHT else 64),
    512 if P else (256 if LIGHT else 64), 256 if P else (128 if LIGHT else 32),
    256 if P else (128 if LIGHT else 64), 128 if P else (64 if LIGHT else 32),
    1, 64, 16, 32,
]
comptime D = SMDims[Robot]
comptime S = SMDims[Robot].S
comptime N = BHL_N_DOF
comptime FIRST_LEG = 10          # the 10 arm joints come first in BHL's order


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _store_paths(spec: String) raises -> List[String]:
    var out = List[String]()
    if isdir(spec):
        var names = List[String]()
        for n in listdir(spec):
            if String(n).endswith(".h5"):
                names.append(String(n))
        sort(names)
        for k in range(len(names)):
            out.append(spec + "/" + names[k])
    else:
        for part in spec.split(","):
            out.append(String(part))
    return out^


def _library(spec: String, upto: Int) raises -> SMMotionLibrary[Robot]:
    """Clips 0 .. upto of the stores, in order (enough for the chosen ones)."""
    var paths = _store_paths(spec)
    var lens = List[Int]()
    var rows = List[Scalar[DT]]()
    for p in range(len(paths)):
        if len(lens) > upto:
            break
        var st = TrajectoryStore(paths[p])
        var refc = st.load_column[DType.float32](String("ref"))
        var n_ep = min(st.n_episodes(), upto + 1 - len(lens))
        var rows_n = 0
        for e in range(n_ep):
            var l = Int(st.episodes.ep_len[e])
            lens.append(l)
            rows_n += l
        var o = len(rows)
        rows.resize(o + rows_n * S, 0)
        for i in range(rows_n * S):
            rows[o + i] = refc[i]
    var q0 = List[Float64](length=Robot.NDOF, fill=0.0)
    for j in range(Robot.NDOF):
        q0[j] = Robot.default_joint_pos(j)
    return SMMotionLibrary[Robot](rows^, lens, q0)


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var ckpt = _opt(args, String("--ckpt"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var every = max(Int(_opt(args, String("--every"), String("1"))), 1)
    var max_clips = Int(_opt(args, String("--max-clips"), String("0")))
    var out = _opt(args, String("--out"), String("torque.csv"))
    if ckpt == "" or store == "":
        raise Error("give --ckpt PATH and --store PATH")
    var trainer = SuperMimicTrainer[C].for_robot(Robot.SIM.create())
    var ctr = trainer.load_state(ckpt)
    print("checkpoint at iteration", ctr[0], "(", ctr[1], "updates ); effort", BHL_EFFORT_PCT, "%, speed", BHL_SPEED_PCT, "% of BHL's limits")
    var env = SMTrackingEnv[Robot](
        1, _library(store, 1 << 30), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, max_episode_steps=0,
    )
    var clips = List[Int]()
    var c = 0
    while c < env.lib.num_clips and (max_clips == 0 or len(clips) < max_clips):
        clips.append(c)
        c += every
    print("library:", env.lib.num_clips, "clips; running", len(clips))

    var rb = SMReplayOf[Robot](1, 64)
    var hist = TensorImpl[DT].alloc(D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(D.GOAL)
    var sum2 = List[Float64](length=N, fill=0.0)
    var peak = List[Float64](length=N, fill=0.0)
    var at_effort = List[Int](length=N, fill=0)
    var over_speed = List[Int](length=N, fill=0)
    var v2 = List[Float64](length=N, fill=0.0)
    var total_steps = 0
    var failed_n = 0
    var ret_sum = 0.0
    var per_clip = String("clip,steps,norm_return,failed,leg_rms_torque\n")

    var first: List[Int] = [env.lib.clip_start[clips[0]]]
    env.restart(first)
    for k in range(len(clips)):
        # the lane's NEXT reset: the following clip (or this one again at the end)
        env.next_frame[0] = env.lib.clip_start[clips[k + 1]] if k + 1 < len(clips) else env.lib.clip_start[clips[k]]
        var ret = 0.0
        var steps = 0
        var leg2 = 0.0
        while True:
            env.policy_inputs(hist, goals)
            var a = trainer.act[1](hist, goals)
            for j in range(N):
                var q = env.lanes[0].sim.qpos(7 + j)
                var qd = env.lanes[0].sim.qvel(6 + j)
                var tau = bhl_pd_torque(j, bhl_pd_target(j, Float64(a.data[j])), q, qd)
                sum2[j] += tau * tau
                v2[j] += qd * qd
                peak[j] = max(peak[j], abs(tau))
                if abs(tau) >= 0.99 * bhl_effort_limit(j):
                    at_effort[j] += 1
                if bhl_speed_limit(j) > 0.0 and abs(qd) > bhl_speed_limit(j):
                    over_speed[j] += 1
                if j >= FIRST_LEG:
                    leg2 += tau * tau
            _ = env.step(a.data, rb)
            ret += env.last_reward[0]
            steps += 1
            if env.lanes[0].ep_len == 0:        # the clip ended (the lane was reset to the next)
                break
        var failed = env.last_terminated[0]
        total_steps += steps
        failed_n += 1 if failed else 0
        var nr = ret / (0.1 * Float64(steps))
        ret_sum += nr
        per_clip += (
            String(clips[k]) + "," + String(steps) + "," + String(nr) + "," + ("1" if failed else "0") + ","
            + String(sqrt(leg2 / Float64(steps * (N - FIRST_LEG)))) + "\n"
        )
        if (k + 1) % 50 == 0 or k == len(clips) - 1:
            print("  [", k + 1, "/", len(clips), "] mean normalised return", ret_sum / Float64(k + 1), "failed", failed_n)

    var csv = String("joint,effort_limit,speed_limit,rms,peak,at_effort,over_speed,rms_speed\n")
    var leg_rms2 = 0.0
    var worst_sat = 0.0
    for j in range(N):
        var rms = sqrt(sum2[j] / Float64(total_steps))
        var sat = Float64(at_effort[j]) / Float64(total_steps)
        csv += (
            String(bhl_dof_name(j)) + "," + String(bhl_effort_limit(j)) + "," + String(bhl_speed_limit(j)) + ","
            + String(rms) + "," + String(peak[j]) + "," + String(sat) + ","
            + String(Float64(over_speed[j]) / Float64(total_steps)) + "," + String(sqrt(v2[j] / Float64(total_steps))) + "\n"
        )
        worst_sat = max(worst_sat, sat)
        if j >= FIRST_LEG:
            leg_rms2 += sum2[j] / Float64(total_steps)
    write_text_atomic(out.copy(), csv)
    write_text_atomic(out + ".clips.csv", per_clip)
    print(
        len(clips), "clips,", total_steps, "steps: mean normalised return", ret_sum / Float64(len(clips)), ",", failed_n,
        "failed | leg RMS torque", sqrt(leg_rms2 / Float64(N - FIRST_LEG)), "N.m | most saturated joint at its effort limit",
        worst_sat * 100.0, "% of the steps",
    )
    print("wrote", out, "and", out + ".clips.csv")
