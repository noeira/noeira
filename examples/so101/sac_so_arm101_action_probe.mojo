"""Is the actor SATURATED? Run this after any training run.

    pixi run mojo run -I . examples/so101/sac_so_arm101_action_probe.mojo --ckpt <run_id>

A greedy action is `tanh(mu) * action_scale`, and where it sits in that range
is the difference between "a smooth policy that happens to oscillate" and "an
actor railed at the tanh limits and flipping between them". The reward cannot
see the difference; nothing in a return curve can.

⚠⚠ THIS IS THE SCRIPT THAT FOUND THE DEFECT BEHIND THREE FAILED REWARD
SHAPES. With the old `ACTION_SCALE = 2.0` against joint ranges of 1.66 to
2.84, a trained policy commanded an out-of-range pose on **24% to 100% of
control steps**, `elbow_flex` sat at the rail 49% of the time, and the gripper
— asymmetric -0.17..1.75 against a symmetric +-2.0 — was out of range on
EVERY step. Everything past the clamp is a DEAD GRADIENT BAND: many actor
outputs map to one pose, so the actor drifts across it for free and flips to
the far rail for free. The reward was never reaching the policy.

The action space is now NORMALIZED — [-1, 1] per joint, mapped affinely onto
each `ctrlrange` by the env (`SoArmReachConfig.NORMALIZED_ACTIONS`), so
`SCALE = 1.0` and **`outside range` should read 0% for every joint**. A
non-zero column here means either the flag is off, or a script is still
building its agent with the old scale.

⚠ A high `at rail` is no longer automatically a defect once the rails ARE the
joint limits — a reach that genuinely wants a joint at its stop will sit
there. Read it together with `outside range`, which should be 0.
"""
from std.random import seed
from std.sys import argv
from std.math import abs
from max.gpu.host import DeviceContext
from noeira.nn.constants import DT
from noeira.deep_agents.data.any_replay import AnyReplay
from noeira.deep_agents.sac import SACAgent, SACActorNet, SACCriticNet
from noeira.deep_agents.training.blocks import ReplaySampleStep
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.robots.so_arm101_xml import SoArm101Model
from noeira.envs.robots.so_arm101 import SoArm101ReachConfig
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from noeira.robot.so101 import joint_name
from noeira.utils.fmt import col, fixed, pad_right
from noeira.core.run import resolve_checkpoint

comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime OBS_DIM = EnvT.OBS_DIM
comptime ACT_DIM = EnvT.ACTION_DIM
comptime SCALE = 1.0
comptime AgentT = SACAgent[
    "cpu",
    ReplaySampleStep[AnyReplay["cpu", OBS_DIM, ACT_DIM, 1000], 256],
    SACActorNet[OBS_DIM, ACT_DIM, 256],
    SACCriticNet[OBS_DIM, ACT_DIM, 256],
]
comptime N_EP = 8
comptime STEPS = 500
comptime SETTLE = 200
comptime DEFAULT_CKPT = "sac_so_arm101_reach.ckpt"


def ckpt_from_argv(default: String) raises -> String:
    """`--ckpt <run_id|path>`, else `default`. A RUN ID resolves to its
    `checkpoints/last.ckpt` — what `sac_so_arm101_reach_training_gpu.mojo`
    writes into its run directory."""
    var args = argv()
    for i in range(1, len(args) - 1):
        if String(args[i]) == "--ckpt":
            return resolve_checkpoint(String(args[i + 1]), String("last"))
    return default


def main() raises:
    var agent = AgentT(action_scale=SCALE)
    var ckpt = ckpt_from_argv(String(DEFAULT_CKPT))
    print("  checkpoint:", ckpt)
    agent.load(ckpt)
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    var sf = SoArm101Model.make_spec_fields[DType.float64]()
    var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, ACT_DIM)
    var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, ACT_DIM)

    seed(7)
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0))
    var rail = List[Float64](length=ACT_DIM, fill=0.0)
    var outside = List[Float64](length=ACT_DIM, fill=0.0)
    var absmean = List[Float64](length=ACT_DIM, fill=0.0)
    var n = 0.0
    for _ in range(N_EP):
        var s0 = env.reset()
        for i in range(OBS_DIM):
            obs[i] = Scalar[DT](s0.data[i])
        for t in range(STEPS):
            agent.select_greedy_action(obs, act)
            var a = EnvT.ActionType()
            for j in range(ACT_DIM):
                var v = Float64(act[j])
                a.data[j] = v
                if t >= SETTLE:
                    # |a| > 0.99 * scale means tanh is within 1% of its rail
                    if abs(v) > 0.99 * SCALE:
                        rail[j] += 1.0
                    # ⚠⚠ "OUTSIDE" IS OUTSIDE THE ACTION BOX, NOT THE
                    # ctrlrange. Under the old radians action space those were
                    # the same test; under `NORMALIZED_ACTIONS` they are not,
                    # and comparing a normalized action against a range in
                    # RADIANS reported the gripper — range -0.17..1.75, so it
                    # does not contain [-1, 1] — as 78% out of range when it
                    # was perfectly in bounds. A units error in the CHECK,
                    # read for a moment as a defect in the policy.
                    if abs(v) > SCALE:
                        outside[j] += 1.0
                    absmean[j] += abs(v)
            if t >= SETTLE:
                n += 1.0
            var out = env.step(a)
            for i in range(OBS_DIM):
                obs[i] = Scalar[DT](out[0].data[i])
    print("=" * 72)
    print("GREEDY ACTION, after arrival —", Int(n), "control steps")
    print("  action_scale =", SCALE, " (tanh rails at +/-", SCALE, ")")
    print("=" * 72)
    print("  joint          ctrlrange        mean|a|   at rail   outside +/-1")
    for j in range(ACT_DIM):
        print(
            "  " + pad_right(String(joint_name(j)), 13),
            col(Float64(lo_col[j]), 7, 2), col(Float64(hi_col[j]), 7, 2),
            col(absmean[j] / n, 11, 2),
            col(100.0 * rail[j] / n, 9, 0) + "%",
            col(100.0 * outside[j] / n, 12, 0) + "%",
        )
    print("=" * 72)
    print("A high 'at rail' means the actor is SATURATED: it is commanding a")
    print("pose the joint cannot reach, the env clamps it, and flipping rail")
    print("to rail is free. That is an optimisation failure, not a reward one.")
