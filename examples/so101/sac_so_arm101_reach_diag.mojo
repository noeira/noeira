"""Did it REACH, does it HOLD, and how hard is it shaking? (CPU, no window)

    pixi run mojo run -I . examples/so101/sac_so_arm101_reach_diag.mojo --ckpt <run_id>

⚠⚠ **THE RETURN DOES NOT ANSWER "DID IT REACH."** The reward is dm_control's
`tolerance` with `TARGET_RADIUS = 0.02 m` and `REWARD_MARGIN = 0.25 m`, and
that margin is twelve times the radius, so the falloff is nearly flat across
the whole neighbourhood of the target:

    distance   0 mm   20 mm   30 mm   40 mm   60 mm  100 mm  150 mm
    reward     1.000  1.000   0.996   0.985   0.943   0.790   0.537

**Hovering 40 mm away for the whole episode scores 492.7 / 500** — higher than
the best episode a trained policy has actually produced here. So a 475 mean,
which lands in `sac_so_arm101_reach_eval_cpu.mojo`'s "EXCELLENT" band, is
consistent with an arm that never once touches the target. That band reads the
only number it has; this script measures the ones that matter:

  * **reached** — did the closest approach get inside 20 mm, per episode;
  * **held** — where is the jaw at the END, which is a different question and
    usually a much worse number (mean closest ~20 mm against mean final
    ~38 mm: it passes THROUGH the target and drifts off);
  * **chatter** — how far the COMMANDED POSITION moves per control step, in
    servo ticks, and how often it reverses direction. This is the number that
    decides whether the policy is safe to put on hardware, and the reward is
    blind to it: shake costs nothing when the falloff is that flat.

The second table sweeps a low-pass filter on the command, which is what
`deploy_reach_real.mojo` applies (`SMOOTH`). It is a DEVIATION FROM TRAINING,
so it is measured here rather than assumed safe — same targets for every
alpha, or the comparison would be reading noise.

⚠ The two tables use different target draws (the sweep re-seeds per alpha so
every row sees the same targets, which is what makes the ROWS comparable to
each other, not to the table above). Reach rates around half are what this
checkpoint does; do not read one table's rate as more precise than the other.
"""

from std.random import seed
from std.sys import argv
from std.math import sqrt, asin, abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.data.any_replay import AnyReplay
from noeira.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from noeira.deep_agents.training.blocks import ReplaySampleStep
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.robots.so_arm101_xml import SoArm101Model
from noeira.envs.robots.so_arm101 import SoArm101ReachConfig
from noeira.utils.fmt import col, fixed, pad_left
from noeira.core.run import resolve_checkpoint

comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime OBS_DIM = EnvT.OBS_DIM
comptime ACT_DIM = EnvT.ACTION_DIM
comptime HIDDEN = 256
comptime BATCH = 256
comptime CAP = 1000
comptime CKPT = "sac_so_arm101_reach.ckpt"
comptime ACTION_SCALE = 1.0

comptime N_EP = 24
comptime STEPS = 500  # SoArmReachConfig.MAX_STEPS
comptime SETTLE = 200
"""Chatter is read only from here on. Before it the arm is still travelling,
and counting the approach as shake would blame the policy for doing its job."""
comptime SUCCESS_M = 0.02  # SoArmReachConfig.TARGET_RADIUS
comptime TICKS_PER_RAD = 4096.0 / 6.283185307179586
comptime RANGE_RAD = 3.4
"""Mean joint span, radians — `ctrlrange` runs 3.32 (wrist_flex) to 5.58
(wrist_roll), mean ~3.4.

⚠⚠ THE ACTION IS NORMALIZED NOW, so a delta in action units is NOT a delta in
radians and multiplying it by `TICKS_PER_RAD` alone under-reports the command
motion by half a joint span. `a in [-1, 1]` maps onto `[lo, hi]`, so
`d(ctrl) = d(a) * (hi - lo) / 2`. Using ONE mean span rather than the per-joint
one keeps this a single comparable number across joints and versions; it is a
scale, not a physical claim. The same units error made the action probe report
the gripper 78% "out of range" when it was in bounds — third instance of it
from one action-space change, which is what a change of UNITS does to every
derived number that was written before it."""
comptime BASE_Z = 0.05  # SoArmReachConfig.BASE_Z

comptime AgentT = SACAgent[
    "cpu",
    ReplaySampleStep[AnyReplay["cpu", OBS_DIM, ACT_DIM, CAP], BATCH],
    SACActorNet[OBS_DIM, ACT_DIM, HIDDEN],
    SACCriticNet[OBS_DIM, ACT_DIM, HIDDEN],
]


def ckpt_from_argv(default: String) raises -> String:
    """`--ckpt <run_id|path>`, else `default`. A RUN ID resolves to its
    `checkpoints/last.ckpt` — what `sac_so_arm101_reach_training_gpu.mojo`
    writes into its run directory."""
    var args = argv()
    for i in range(1, len(args) - 1):
        if String(args[i]) == "--ckpt":
            return resolve_checkpoint(String(args[i + 1]), String("last"))
    return default


def episodes(
    mut env: EnvT, mut agent: AgentT, alpha: Float64, rows: Bool
) raises -> Tuple[Int, Float64, Float64, Float64, Float64]:
    """`(reached, mean_min_mm, mean_end_mm, chatter_ticks, mean_return)`.

    ⚠ RE-SEEDS. Every alpha must see the SAME targets or the sweep compares
    target luck. The seed is set here and not in `main` because building the
    agent draws from the same stream.
    """
    seed(7)
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0))
    var cmd = List[Float64](length=ACT_DIM, fill=0.0)
    var prev = List[Float64](length=ACT_DIM, fill=0.0)
    var prev_d = List[Float64](length=ACT_DIM, fill=0.0)

    var reached = 0
    var s_min = 0.0
    var s_end = 0.0
    var s_chat = 0.0
    var s_ret = 0.0
    if rows:
        print("  ep      r     el |  closest     final |  ticks/step   rev%")
        print("  " + "-" * 62)
    for ep in range(N_EP):
        var s0 = env.reset()
        for i in range(OBS_DIM):
            obs[i] = Scalar[DT](s0.data[i])
        # The target in the sampler's own polar frame, so a failure can be
        # read against the shell it was drawn from.
        var tx = Float64(obs[15])
        var ty = Float64(obs[16])
        var tz = Float64(obs[17]) - BASE_Z
        var r = sqrt(tx * tx + ty * ty + tz * tz)
        var el = asin(tz / r) if r > 1e-9 else 0.0
        for j in range(ACT_DIM):
            cmd[j] = Float64(obs[j])  # seed the filter on the measured pose
            prev[j] = cmd[j]
            prev_d[j] = 0.0

        var min_d = 1.0e9
        var end_d = 0.0
        var chat = 0.0
        var revs = 0.0
        var moves = 0.0
        var ret = 0.0
        for t in range(STEPS):
            agent.select_greedy_action(obs, act)
            var a = EnvT.ActionType()
            for j in range(ACT_DIM):
                cmd[j] = (1.0 - alpha) * cmd[j] + alpha * Float64(act[j])
                a.data[j] = cmd[j]
                if t >= SETTLE:
                    var d = cmd[j] - prev[j]
                    chat += abs(d) * 0.5 * RANGE_RAD * TICKS_PER_RAD
                    if d * prev_d[j] < 0.0:
                        revs += 1.0
                    moves += 1.0
                    prev_d[j] = d
                prev[j] = cmd[j]
            var out = env.step(a)
            for i in range(OBS_DIM):
                obs[i] = Scalar[DT](out[0].data[i])
            ret += Float64(out[1])
            var dx = Float64(obs[18])
            var dy = Float64(obs[19])
            var dz = Float64(obs[20])
            var dist = sqrt(dx * dx + dy * dy + dz * dz)
            if dist < min_d:
                min_d = dist
            end_d = dist

        var per_step = chat / (moves / Float64(ACT_DIM)) if moves > 0 else 0.0
        var rev_pct = 100.0 * revs / moves if moves > 0 else 0.0
        if min_d <= SUCCESS_M:
            reached += 1
        s_min += min_d * 1000.0
        s_end += end_d * 1000.0
        s_chat += per_step
        s_ret += ret
        if rows:
            print(
                pad_left(String(ep), 4), col(r, 7, 3), col(el, 6, 2), " |",
                col(min_d * 1000.0, 9, 1),
                col(end_d * 1000.0, 10, 1), " |",
                col(per_step, 12, 1), col(rev_pct, 7, 0),
                " <" if min_d <= SUCCESS_M else "",
            )
    var f = Float64(N_EP)
    return (reached, s_min / f, s_end / f, s_chat / f, s_ret / f)


def main() raises:
    print("=" * 72)
    print("SO-ARM101 reach — reached / held / chatter")
    print("=" * 72)
    var agent = AgentT(action_scale=ACTION_SCALE)
    var ckpt = ckpt_from_argv(String(CKPT))
    try:
        agent.load(ckpt)
    except e:
        print("ERROR loading", ckpt, "-", e)
        return
    print("  checkpoint:", ckpt, " episodes:", N_EP, "x", STEPS, "steps")
    var ctx = DeviceContext()
    var env = EnvT(ctx)

    print("\nRAW POLICY (what `deploy_reach_real.mojo` would send at SMOOTH=1)")
    var base = episodes(env, agent, 1.0, True)
    print("  " + "-" * 62)
    print(
        "  reached <=", fixed(SUCCESS_M * 1000.0, 0), "mm:", base[0], "/",
        N_EP, "  mean closest", fixed(base[1], 1),
        "mm   mean final", fixed(base[2], 1), "mm",
    )
    print(
        "  mean return", fixed(base[4], 1),
        "/ 500  <- note how little this moves across the rows above",
    )

    print("\nCOMMAND LOW-PASS SWEEP — same targets in every row")
    print("  alpha | reached | closest | final |  ticks/step | return")
    print("  " + "-" * 62)
    var alphas = List[Float64]()
    alphas.append(1.0)
    alphas.append(0.5)
    alphas.append(0.3)
    alphas.append(0.2)
    alphas.append(0.15)
    alphas.append(0.1)
    alphas.append(0.05)
    for i in range(len(alphas)):
        var res = episodes(env, agent, alphas[i], False)
        print(
            col(alphas[i], 7, 2), " |",
            pad_left(String(res[0]) + "/" + String(N_EP), 8), " |",
            col(res[1], 8, 1), " |", col(res[2], 6, 1), " |",
            col(res[3], 12, 1), " |", col(res[4], 7, 1),
        )
    print("  " + "-" * 62)
    print(
        "  alpha 1.0 is the raw policy. `deploy_reach_real.mojo` ships 0.05,"
        "\n  which on the current checkpoint holds the reach rate exactly"
        " while cutting\n  command motion ~48x (4438 -> 93 ticks/step)."
        "\n  ⚠ A VELOCITY PENALTY CANNOT SEE OSCILLATION, only speed: across"
        " five\n  checkpoints mean |qvel| fell 1.21 -> 0.92 while the"
        " REVERSAL rate rose\n  59% -> 92%. The policy paid for lower speed"
        " with higher frequency,\n  which is exactly what the metric asked"
        " for. Penalising the ACTION RATE\n  is the fix that targets the"
        " oscillation itself, and it needs the previous\n  action in the"
        " observation to stay Markov."
    )
    print("=" * 72)
