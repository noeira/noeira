"""SAC ON A TASK FAMILY — the ONE driver, generic over the family's model + config.

    examples/tasks/sac_task_gpu.mojo    the so101_tabletop entry (project so101)
    examples/tasks/sac_tower_gpu.mojo   the so101_tower entry (project so101-tower)

Each entry is a dozen lines: it names the family, the project the runs file
under, and calls `run_sac[MODEL, CONFIG](argv, ...)`. ONE instantiation per
binary, on purpose — the family's GPU kernels are minutes of compile each,
and a runtime dispatch over two families would compile both every time.
Everything below is the driver as it was written for the tabletop family;
`So101TabletopConfig` became `C`, `So101TabletopModel` became `M`, nothing
else moved. The history and every measured number in this docstring are the
tabletop's.

SAC ON THE TASK FAMILY — the first policy this layer has ever carried.

    pixi run -e nvidia mojo run -I . examples/tasks/sac_task_gpu.mojo so101_reach_clear
    ... --steps 200000        # lanes are comptime (N_ENVS below), not a flag

⚠⚠ NVIDIA ONLY. The family is `nv = 24`, where the P0 park probe died on
Metal ("Compute function exceeds available stack space" — the physics kernels
stack-allocate per-thread arrays sized by `nv`). It COMPILES on Apple and
cannot launch, exactly like `task_batched_gpu.mojo` beside it.

## ⚠⚠ EVERY NUMBER THIS LAYER HAS REPORTED SO FAR IS A HARNESS NUMBER

P1-P5 gated the spec, the composer, the goal language, the sampler, the tape,
the active mask, the init table and the LIBERO importer. Not one of them put a
POLICY on the family, and the paths a training run exercises are not the paths
a gate does: the observation's scale, a sparse reward's cadence, the reset rate
under success-termination, and whether `TERMINATE_ON_UNHEALTHY` was spelled.
This file is that run.

## ⚠⚠ THE TASK IS `so101_reach_clear`, AND THE REASON IS A REAL GAP

`so101_reach_brick` cannot train today. An ACTIVE free slot must be PLACED at
every episode reset, and on the GPU path nothing places it: `_reset_env_lane`
restores the composed scene's `qpos0` — the PARK pose, 50 m up — and only
INACTIVE slots are pinned there afterwards by `pre_step_full_gpu`. So an
active prop starts every episode in the sky, falls for the whole horizon, and
its qpos and qvel go into the observation. The HOST places props
(`sampler.sample_placements` + `reset.reset_slots`), which is why the eval and
viewer paths are fine and why nothing saw this until something trained.

`so101_reach_clear` has the same goal and NO active free slot, so it needs no
placement. See its header for what that costs.

## ⚠⚠ THREE WIRING FACTS THAT ARE EACH A SILENT FAILURE IF MISSED

1. **`TERMINATE_ON_UNHEALTHY=True`.** The config's reward hook returns
   `(reward, holds)` and asks to terminate on success — and the ask is
   DISCARDED unless the env is instantiated with this flag
   (`phyics3d_batched_env.mojo:1161`). Without it a solved lane keeps running
   and banks +1 per step to the horizon, so a return stops being a success
   indicator and becomes "how early did it succeed". `task_eval_frozen.mojo`
   already reported 0/128 on a task that holds at reset for the same reason,
   from the other side.

2. **`ACTION_SCALE = 1.0`.** `So101TabletopConfig.NORMALIZED_ACTIONS` is True,
   so the action IS [-1, 1] per joint and the env maps it affinely onto each
   actuator's own ctrlrange. A scale of 2.0 maps [-2, 2] onto the range and
   puts the useful band back inside the tanh rails — undoing the fix while
   still looking configured. The measurement behind that flag is in
   `Phyics3dEnvConfig.NORMALIZED_ACTIONS`, on this same robot.

3. **`initial_episode_fill = 0.0`.** SAC seeds its return window with
   `-1250.0`, a HalfCheetah-flavoured value. Here a return is in {0, 1}, so
   the default makes every reading before the window fills a blend of real
   returns and sentinels and reads as a broken reward.

## ⚠⚠ THE MEASURED BASELINES — READ THESE BEFORE ANY CURVE

Per task, 20k env-steps at N_ENVS=64 with `--warmup >= --steps` (uniform
random) and the driver's greedy eval on an UNTRAINED actor:

    task                  random   untrained greedy   constant action
    so101_lift_brick        0.00        0.00           never met
    so101_gather_bricks    <0.02        0.00           never met
    so101_reach_clear       0.25        1.00           SWEPT THROUGH
    so101_settle_brick      1.00        1.00           met at every step

⚠ `gather` IS NOT EXACTLY ZERO under random actions — two 20k warmup-only runs
gave 0.000 and 0.0156 (one lane of 64). Random does occasionally push the
blocks together, so 0.02 is the number a rate has to beat, and at a 100-
episode window its standard error is 0.014. The verdict prints that band
because the first long `gather` run oscillated 0.00 .. 0.05 for 125k steps and
every one of those values sits inside it.

⚠ `reach` IS NOT A REACHING TASK AND THAT IS WHY THE DEFAULT IS `lift`.
`examples/tasks/task_null_action.mojo` measures it: a CONSTANT action of +0.3
meets `AtRegion(robot_gripperframe, table_top)` on 77 consecutive steps, and
the run ENDS at step 97 of 300 — the arm sweeps the gripper across the region
on its way somewhere else. An instantaneous predicate over a CONTROLLED end
effector, with a per-step reward and first-hit termination, asks "did the
gripper ever pass through here". A predicate over an OBJECT's pose does not
have that failure: a sweep does not lift a brick.

⚠ `settle` IS THE PROBE and scores 1.00 by construction — its goal holds at
reset. Two GPU gates need a true lane; see its own header. Training it is
meaningless and its 20032 episodes in 20k steps (one per step, because
success terminates) is what a correctly wired success-termination looks like.

⚠⚠ SO A FLOOR OF 0.00 IS WHAT `lift` AND `gather` ACTUALLY HAVE, and any
sustained rate above it is learning.

## ⚠⚠ THE REWARD IS SHAPED NOW, AND THE RETURN IS NO LONGER THE SUCCESS RATE

The first 125k-step `gather` run was flat, and the reason was arithmetic:
at ~1.5% success that is about SIX rewarding transitions in 125,000 — 5e-05
of the replay buffer, which a batch of 256 contains 1.4% of the time. The
critic almost never saw a success. `So101TabletopConfig` now subtracts two
dense terms, `SHAPE_W_GOAL` on the goal's own distance (generic, from the
tape) and `SHAPE_W_REACH` on the gripper's distance to the body the goal
names — the second because goal distance alone has NO gradient until the arm
touches something, which on `gather` is the whole difficulty.

    tolerance reward, random actions, gather, 32 envs   +70.7
    success rate, greedy, untrained                       0.00

⚠⚠ THE REWARD IS POSITIVE AND BOUNDED NOW — two `tolerance` terms in [0, 1]
weighted 1.0 and 0.5, so an episode return lives in [0, 450] and random sits
at 70.7: 16% of the ceiling with the rest reachable. The old clipped linear
PENALTY put random at -31.5 against a ceiling of 0 — the same information with
no room above it and no saturation below.

⚠ EARLIER FIGURES ARE NOT COMPARABLE, and are recorded because the reasoning
around them is: -31.5 (32 envs) and -19.98 (64 envs) were the linear form at
weights 0.50/0.25, and -3.996 was 0.10/0.05. A return is comparable only
within one reward SHAPE, one weight pair and one lane count. All three now
travel with the run in its config (`metrics.config.kv`).

⚠ THE SUCCESS-RATE baselines are lane-count independent — a rate is per
episode either way — so only the RETURN figures moved.

⚠⚠ THE CONFIGURATION THAT TRAINS, AND THE TASK IS SOLVED MORE THAN HALF THE
TIME. `so101_gather_bricks`, 1M steps, weights 1.0/0.21, everything else
default, 3328 episodes in 34 minutes on a 5090:

    SUCCESS RATE                          0.5625   ⚠ PRE-SHRINK, superseded — see `baselines_for`
      against a random baseline of          0.02   2-sigma band ends 0.069
    eval return                  35 -> 207, best 238   ceiling 363
    avg_reward                   44 -> 224, monotone, best = last
    mean_reward               0.186 -> 0.451, monotone over 1M
    mean_q       48.9 against a fixed point of 48.8            1.00x
    next_q - mean_q                                          +0.112
    critic_loss                                                0.83

Eighteen of thirty-two lanes bring the blocks within 6 cm, from a policy that
was handed nothing but a `.task` file. `mean_q` landing ON its fixed point is
what a correct critic looks like — read that before any return.

⚠ THE RETURN AND THE RATE ARE DIFFERENT CLAIMS and only the rate is the task.
Both reward terms pay for PROXIMITY, so a high return can mean "hovering near"
rather than "solved" — which is exactly what 990k steps at 64% of the ceiling
turned out to be worth: 0.56 success, not 0.64.

⚠⚠ THE LAST CHANGE WAS `TERMINATE_ON_UNHEALTHY: True -> False`, ALONE. Run 13
had every other setting identical — same weights, same tau, same entropy, same
tolerance margins, confirmed by its own logged config — and its `mean_q` sat 12x
above its fixed point with the eval swinging 25..69 around a baseline of 48.

Why it matters is a value CLIFF. A lane that succeeds gets `done = 1`, so the
critic masks the bootstrap and its target is the one-step reward alone —
about 0.25 — while every neighbouring state carries Q near 24. Successes are
order 1.5% of episodes, so a handful of transitions in the buffer disagree by
sixty-fold with everything around them, and the critic chases that.

⚠ FIVE REWARD HYPOTHESES CAME FIRST AND EACH EXPLAINED PART OF IT: magnitude,
shape, the reach/goal ratio, the entropy target, the tracking rate. Only the
tracking rate and this were real. The rest fitted the runs after the fact —
which is what a table of partial correlations looks like, and the reason to
prefer a STRUCTURAL difference from a working reference over another curve.

⚠ AND A HEALTHY RUN HAS A SHAPE. What to read FIRST is `mean_q`: it should
converge toward `mean_reward / (1 - gamma)` with `mean_next_q - mean_q` under
a tenth. Run 3's gap was +5 and its `mean_q` ran to 508; run 10's was +35 at
5743. That is a critic chasing itself, and no return moves under one.

⚠ SO THIS FILE PRINTS TWO NUMBERS. `mean_return` is what SAC optimises and
moves smoothly; the SUCCESS RATE is measured separately by
`greedy_success_rate` — one greedy episode per lane, counting `reward > 0.5`
— reading `META_IDX_GOAL_HELD`, the word the reward hook writes. `tests/tasks/test_goal_distance.mojo` asserts that bound.

⚠ SET `SHAPE_W_GOAL` AND `SHAPE_W_REACH` TO 0.0 to get the sparse reward
back. Every success-rate baseline above was measured there, and a shaped run
is not comparable with a sparse one on RETURN — only on the rate.

## ⚠ THE RETURN *IS* THE SUCCESS RATE, WHICH IS WHY THIS IS READABLE AT ALL

The reward is sparse — +1 on the step the goal holds — and the episode
TERMINATES on that step. So an episode return is exactly 1 if solved and 0 if
not, and `agent.mean_return()` over the last 100 episodes is the success rate
directly. No band table, no shaped-reward calibration: the criterion is
"does it move off zero".

⚠ AND THE FLOOR IS ZERO HERE, unlike the shaped SO-ARM101 reach whose
untrained baseline is 46 of 500. `examples/tasks/task_reachability.mojo`
measures 3.2% of uniform arm poses inside the goal region, so a random policy
scores somewhere near but not at zero; anything sustained above ~0.1 is
learning.
"""

from std.pathlib import Path
from std.sys import is_defined
from std.random import seed as seed_rng
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer

from layout import Layout, LayoutTensor

from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.core.dotenv import load_dotenv
from noeira.core.logger import CsvLogger, RemoteLogger, CompositeLogger
from noeira.core.run import RunContext, register_run
from noeira.io.artifact_sink import close_sink, sink_for_run
from noeira.deep_agents.primitives.stochastic_actor import StochasticActor
from noeira.deep_agents.sac import SACAgent
from noeira.deep_agents.hil_serl import HilSerlConfig, apply_hil_serl
from noeira.tasks.sac_family_policy import HIDDEN
from noeira.deep_agents.training.blocks import UniformSampleGpuStep
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
    META_IDX_INIT_REGION_0, META_IDX_GOAL_HELD,
    META_IDX_SHAPE_W_GOAL, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime

from noeira.tasks.spec import (
    TaskSpec,
    load_family, load_task, validate_task_against_family, SLOT_FREE,
)
from noeira.tasks.family import scene_path, task_path
from noeira.envs.phyics3d_env import Phyics3dEnvConfig
from noeira.physics3d.model import ModelDefLike
from noeira.tasks.predicates import parse_goal, bind_goal, require_tier_a
from noeira.tasks.eval import (
    region_sites, region_rects, region_half_heights,
)
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.tasks.gpu_eval import region_table_words, require_gpu_regions
from noeira.tasks.active import active_mask, init_region_words
from noeira.tasks.lanes import lane_task, lanes_for_task
from noeira.tasks.shaping import shaping_words, SHAPING_WORDS
from noeira.tasks.critic_health import critic_health


# ⚠⚠ 32, MATCHING THE TWO EXAMPLES THAT TRAIN ON THIS STACK.
# `sac_so_arm101_reach_training_gpu.mojo` and
# `sac_half_cheetah_training_gpu.mojo` both run 32 with
# `updates_per_step = N_ENVS`, which is UTD 1 at a target tracking rate of
# 14.8%. This file ran 64 with 64 — also UTD 1, but 27.4% tracking — and the
# critic diverged in four consecutive runs before the tracking rate was the
# thing anybody looked at.
#
# ⚠ COMPTIME, so this is an edit and not a flag: `N_ENVS` sizes `Layout`
# parameters and the greedy eval's lane count. `--updates-per-step` is the
# flag, and it is what varies the tracking rate WITHOUT changing how much
# data an iteration collects.
#
# ⚠ `-D TASK_SAC_LANES_256` / `-D TASK_SAC_LANES_1024` select a wider batch
# at BUILD time (Mojo has no integer define here), for the Squint-scale runs
# of `noeira-docs/SO101_PIXEL_RL_PLAN.md` step 0. The default stays 32. A
# wide run MUST pass `--updates-per-step` and `--tau` explicitly: the default
# `updates_per_step = N_ENVS` would be 1024 updates per iteration, and the
# target's tracking rate is `1 - (1 - tau)^updates_per_step` (below) — at 256
# updates, `--tau 0.0003` is the ~7.4 % the 32-lane history found stable.
comptime N_ENVS = (
    1024 if is_defined["TASK_SAC_LANES_1024"]()
    else (256 if is_defined["TASK_SAC_LANES_256"]() else 32)
)
# ⚠⚠ `lift`, NOT `reach`. `reach` and `reach_clear` are both
# `AtRegion(robot_gripperframe, table_top)` and
# `examples/tasks/task_null_action.mojo` measured what that is worth: a
# CONSTANT action of +0.3 meets it on 76 consecutive steps, and the longest
# run ends at step 96 of 300 — the arm SWEEPS the gripper across the region on
# its way somewhere else. An instantaneous predicate over a controlled end
# effector plus first-hit termination asks "did the gripper ever pass through
# here", which most large joint motions satisfy without aiming.
#
# A predicate over an OBJECT's pose is not that shape: a sweep does not lift a
# brick. `--task` takes any of them.

comptime EVAL_ROUNDS_N: Int = 4
"""Rounds of `N_ENVS` greedy episodes the final evaluation pools.

⚠ EVERY MESSAGE ABOUT THE RATE READS THIS. The band, the rule-of-three bound
and the printed lane count were all written as a bare 32 — the lane count —
back when one round was the whole evaluation. A second round would have made
each of them wrong by a factor, silently, in the direction of claiming more
resolution than the number has."""

# ⚠⚠ `False`, MATCHING BOTH EXAMPLES THAT TRAIN ON THIS STACK. It was True —
# the config's reward hook asks to terminate on success and that ask is
# discarded without this flag — and True is a STRUCTURAL difference from
# `sac_so_arm101_reach_training_gpu.mojo` and
# `sac_half_cheetah_training_gpu.mojo`, both of which run False.
#
# ⚠ WHAT SUCCESS-TERMINATION DOES TO A VALUE FUNCTION. A lane that succeeds
# gets `done = 1`, so the critic masks the bootstrap and its target for that
# transition is the one-step reward alone — about 0.25 here — while its
# neighbours in state space carry Q near the fixed point of 16. The critic has
# to fit a cliff at the success boundary, and it is a RARE cliff: successes
# are order 1.5% of episodes, so a handful of transitions in the buffer
# disagree by 60x with everything around them.
#
# That is a candidate for the divergence that five reward hypotheses each
# explained partly and none explained fully — and unlike them it is
# structural, not fitted to the runs after the fact.
#
# ⚠ IT COSTS THE EPISODE BOUNDARY ON SUCCESS. A solved lane now keeps running
# to `MAX_STEPS`, accruing reward that is HIGHEST near the goal — so holding
# the blocks together pays, which is the behaviour the task wants anyway and
# what both working examples do. `greedy_success_rate` is unaffected: it reads
# `META_IDX_GOAL_HELD` and asks whether the goal held at ANY step.
comptime EnvT[M: ModelDefLike, C: Phyics3dEnvConfig] = Phyics3dBatchedEnv[
    M, C, N_ENVS, TERMINATE_ON_UNHEALTHY=False,
]

comptime OBS_DIM[M: ModelDefLike, C: Phyics3dEnvConfig] = EnvT[M, C].OBS_DIM
# tabletop: 54 = NQ(27) + NV(24) + N_FREE(3); tower: 49 = 20 + 18 + 2 + 9
comptime ACT_DIM = 6
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000

comptime NUM_STEPS = 300_000
comptime WARMUP_STEPS = 10_000
comptime PRINT_EVERY = 10_000
comptime DIAG_EVERY = 2_000
comptime CHECKPOINT_EVERY = 50_000
comptime EVAL_EVERY = 25_000

def split_csv(v: String) raises -> List[String]:
    """`"a,b"` -> `["a", "b"]`; a bare `"a"` -> `["a"]`.

    ⚠ EMPTY PIECES ARE REFUSED. `"gather,"` is a typo that would otherwise
    become a second task named "" and fail much later, inside `load_task`,
    naming a path instead of the flag that produced it.
    """
    var out = List[String]()
    var parts = v.split(",")
    for i in range(len(parts)):
        var piece = String(String(parts[i]).strip())
        if piece.byte_length() == 0:
            raise Error(
                "sac task: empty element in '" + v + "'. A trailing or"
                " doubled comma is a typo, not an empty task."
            )
        out.append(piece)
    return out^


def per_task(v: String, n: Int, what: String) raises -> List[Float64]:
    """One value broadcast to `n` tasks, or exactly `n` values.

    ⚠⚠ A SHAPING PARAMETER IS PER TASK AND THAT IS THE WHOLE POINT. The
    margins live in per-lane `meta` because what a weight is worth depends on
    the task's own distance scale — `gather`'s `Near` starts at 0.139 m,
    `settle`'s `On` at 0.000. One margin across a two-task batch hands its
    lanes a bimodal reward, which on this family is how a critic gets
    destabilised. Broadcasting is allowed because it is sometimes right; a
    WRONG-LENGTH list is refused rather than recycled.
    """
    var parts = split_csv(v)
    if len(parts) == 1:
        var out = List[Float64]()
        for _ in range(n):
            out.append(Float64(parts[0]))
        return out^
    if len(parts) != n:
        raise Error(
            "sac task: " + what + " has " + String(len(parts)) + " values for "
            + String(n) + " tasks. Give one value for all of them, or one per"
            " task in the same order."
        )
    var out = List[Float64]()
    for i in range(len(parts)):
        out.append(Float64(parts[i]))
    return out^


def baselines_for(task: String) -> Tuple[Float64, Float64, Float64, Bool]:
    """`(random_any, untrained_greedy, random_FINAL, measured)` for a task.

    ⚠⚠ THE THIRD NUMBER EXISTS BECAUSE THE FIRST SATURATES ON SOME TASKS.
    `so101_settle_brick`'s goal HOLDS AT RESET, so its any-step rate is 1.0
    for any policy including no policy — judging a run on it reports FLAT
    whether the task is solved or broken. The driver switches to the
    held-at-END rate for such a task and needs that criterion's OWN baseline;
    reusing the any-step 1.00 would put the band at 1.0 and call every run
    flat forever.

    ⚠ FOR A TASK WHOSE ANY-STEP BASELINE IS 0, THE FINAL-STEP ONE IS 0 TOO,
    exactly and not by measurement: a goal met at the final step was met at
    SOME step, so the final rate can never exceed the any-step rate.

    ⚠⚠ PER TASK, AND IT WAS A PAIR OF CONSTANTS. This file trained one task
    when those were written; `--task` made them a lie, and the first `gather`
    run on a 5090 printed `lift`'s floors under `gather`'s rate. A verdict
    that names the wrong baseline is worse than none, because it reads as
    though somebody checked.

    ⚠ THE THIRD FIELD IS "HAS THIS BEEN MEASURED". A task nobody has run a
    baseline for gets `False` and the verdict says so, rather than defaulting
    to 0.0 — which is a real claim, and the flattering one.
    """
    if task == "so101_lift_brick":
        # ⚠⚠ AND IT HAS NOT BEEN TRAINED PAST NOISE. Two 1M-step runs, both
        # with a textbook critic (`mean_q` 0.91x and 0.92x its fixed point):
        #
        #   margin 0.02   return  19.9 ->  56.2  (2.8x)   success 0.00000
        #   margin 0.05   return 153.0 -> 197.6  (1.29x)  success 0.03125
        #
        # 1 lane of 32 is 0.031 against a rule-of-three band of 0.094, so the
        # second is NOT significant. Decomposing the second run's reward: the
        # gripper closed from 0.096 m to 0.050 m while the GOAL distance
        # stayed at 0.030 m — its random value. The arm approaches the brick
        # and never lifts it.
        #
        # ⚠ THAT IS A TASK PROPERTY, NOT A TUNING ONE. `Above(brick, table,
        # 0.06)` only pays once the brick RISES, and the brick only rises once
        # it is GRASPED — a discrete contact event with no partial credit. The
        # reach term can walk the gripper in; nothing in a distance-shaped
        # reward can manufacture a grasp. `Grasped` and `Touching` are Tier B
        # (they read the contact array) and `predicates.require_tier_a` refuses
        # them as goals, so the fix is not a weight.
        return (0.00, 0.00, 0.00, True)
    if task == "so101_gather_bricks":
        # ⚠⚠ RE-MEASURED AFTER THE PROP SHRANK, AND IT MOVED. This was 0.02,
        # from two 20k warmup-only runs on the 4 cm cube that gave 0.000 and
        # 0.0156. `cube.xml` went to 2.4 cm (the jaw cannot close on 4 cm)
        # and a 6.9 g cube is not a 32 g one under a flailing arm, so the
        # number was measured on a task that no longer exists — and the
        # trainer went on printing a verdict against it. Eight 24k
        # warmup-only runs post-shrink, seeds 101..108:
        #
        #     0 successes in 256 episodes -> rule-of-three bound 0.0117
        #
        # ⚠ A HARD-CODED BASELINE IS INVALIDATED BY A GEOMETRY CHANGE AND
        # NOTHING TELLS YOU. Re-measure whenever the family's assets, regions
        # or slot radius move; the provenance above is what makes that
        # checkable.
        #
        # ⚠ THESE ARE SUCCESS RATES AND ARE LANE-COUNT INDEPENDENT, unlike the
        # shaped RETURN — a rate is per episode either way. The return
        # baselines in the header are not, and mixing the two cost two rounds.
        # ⚠⚠ THE 0.5625 TRAINED REFERENCE IS PRE-SHRINK and is NOT a target
        # on the current geometry. It was one draw at margin 0.10 on the 4 cm
        # prop, it did not reproduce at the same config (the repeat diverged
        # to 273x), and the prop has changed since.
        #
        # THE ESTABLISHED REFERENCE, on the current geometry, is two draws at
        # margins 0.10/0.20, weights 1.0/0.21, 16 updates and tau 0.00125 —
        # each 1M steps with a textbook critic (peak 1.015x and 1.023x their
        # fixed points), evaluated over 128 episodes:
        #
        #     seed   any-step        held at END     critic peak
        #     2      0.203 (26/128)  0.148 (19/128)  1.015x
        #     3      0.133 (17/128)  0.117 (15/128)  1.023x
        #     pooled 0.168 (43/256)  0.133 (34/256)
        #
        # The two seeds are one population (Fisher p = 0.09) and the pooled
        # rate against a baseline of 0 in 256 is not close — 43 successes
        # against an expectation of 3. A run far below 0.13 at this config is
        # configured differently or has a sick critic; check `metrics.config.kv` and the
        # peak ratio before tuning anything.
        #
        # ⚠⚠ AND THE MULTI-TASK BATCH BEATS IT. One policy on
        # `so101_gather_bricks,so101_settle_brick`, 2M steps so each task's 16
        # lanes see 1M of their own transitions — the same count the
        # single-task reference had:
        #
        #     gather  0.375 (24/64) any   0.344 (22/64) held   critic 0.995x
        #     settle  1.000 (64/64) any   1.000 (64/64) held
        #
        # p = 4.6e-4 and 1.7e-4 against the single-task rows above.
        #
        # ⚠ THIS IS NOT YET A TRANSFER RESULT. The 2M batch ran 1,000,000
        # gradient updates against the 1M single-task run's 500,000 — gather's
        # own TRANSITIONS match, the network's UPDATES do not. The control
        # that separates them is `gather` alone at 2M steps; until that is
        # run, the claim is "two tasks in one policy, each at or above its
        # single-task strength", not "multi-task helps".
        return (0.00, 0.00, 0.00, True)
    if task == "so101_reach_clear" or task == "so101_reach_brick":
        # ⚠ THE FINAL-STEP FIGURE IS UNMEASURED AND UNUSED HERE: the switch
        # to it only fires when the any-step baseline is 1.0, and this is
        # 0.25. Left at the any-step value rather than a flattering 0.
        return (0.25, 1.00, 0.25, True)
    if task == "so101_settle_brick":
        # ⚠⚠ THE ANY-STEP RATE IS 1.00 BY CONSTRUCTION and says nothing. Its
        # goal holds at reset — two GPU gates need that — so the number to
        # read is the END-OF-EPISODE one, and the shaped return.
        #
        # Measured, 1M steps at 1.0/0.21 with margins 0.05/0.20:
        #
        #   held at the END   0.96875   (31 of 32 — one lane lost the brick)
        #   shaped return     314.2 -> 354.3 of a 363 ceiling, 82% of the
        #                     available headroom, critic at 1.02x fixed point
        #
        # Decomposed, the gain is almost all the GOAL term — 0.896 -> 1.022
        # per step, worth 38 of return — against 2.4 from the reach term. The
        # policy learned to KEEP the brick on the table, which is what settle
        # asks, and barely moved the gripper (0.096 m -> 0.089 m).
        # ⚠⚠ AND THE ANY-STEP 1.00 IS NOT A CRITERION — the goal HOLDS AT
        # RESET, so that rate is 1.0 for any policy including no policy. The
        # held-at-END rate is what a run is judged on and it needs its own
        # baseline: four 24k warmup-only runs, seeds 201..204, gave 0, 0, 0
        # and 1 of 128 — 1 in 512. 0.008 is the ceiling of what was seen,
        # the convention `gather`'s baseline already used.
        return (1.00, 1.00, 0.008, True)
    return (0.0, 0.0, 0.0, False)
# ⚠⚠ THE PREFIXES ARE GONE, AND THE PAIN THEY FIXED IS WORTH KEEPING WRITTEN
# DOWN. Both were fixed strings from when this file trained one task, so the
# first `gather` run on a 5090 wrote `sac_task_reach.ckpt` — and a `lift` run
# after it would have OVERWRITTEN that checkpoint with weights for a different
# task, silently, under a name naming a third. Adding the task to the name
# fixed the collision BETWEEN tasks and left the one that matters more: two
# runs of the SAME task still overwrote each other, which is every sweep arm
# this file has ever produced.
#
# `RunContext` (`core/run.mojo`) is the general form. Every path below comes
# from it, so no two runs can collide and each one carries a record saying
# what it was and whether it worked.

# See wiring fact 2 in the header. NORMALIZED_ACTIONS is True on this config.
comptime ACTION_SCALE = Scalar[DT](1.0)

comptime ActorNet[M: ModelDefLike, C: Phyics3dEnvConfig] = StochasticActor[
    OBS_DIM[M, C],
    ACT_DIM,
    LinearReLU[OBS_DIM[M, C], HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet[M: ModelDefLike, C: Phyics3dEnvConfig] = Sequential[
    LinearReLU[OBS_DIM[M, C] + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]

comptime AgentT[M: ModelDefLike, C: Phyics3dEnvConfig] = SACAgent[
    "gpu",
    UniformSampleGpuStep[OBS_DIM[M, C], ACT_DIM, BATCH, REPLAY_CAPACITY],
    ActorNet[M, C],
    CriticNet[M, C],
]


def greedy_success_rate[M: ModelDefLike, C: Phyics3dEnvConfig](
    mut agent: AgentT[M, C], mut env: EnvT[M, C], ctx: DeviceContext, n_tasks: Int
) raises -> Tuple[List[Float64], List[Float64]]:
    """`(met at ANY step, met at the FINAL step)` over one greedy episode.

    ## ⚠⚠ "AT ANY STEP" IS VACUOUS FOR A GOAL THAT HOLDS AT RESET

    `so101_settle_brick`'s goal is true at step 0 by construction — two GPU
    gates depend on it — so an any-step rate reports 1.0 for that task
    whatever the policy does, including one that knocks the brick straight off
    the table. A criterion a task satisfies before the first action is not a
    criterion.

    ⚠ AND IT IS WEAKER THAN IT LOOKS ON THE OTHERS TOO. `so101_gather_bricks`
    scored 0.5625 at any-step: the blocks came within 6 cm at some point in
    the episode. Whether they were STILL there at the end is a different
    claim, and the one a person means by "solved". The two are returned
    together so the gap between them is visible rather than assumed away.

    ## ⚠⚠ WHY THIS EXISTS: THE RETURN STOPPED BEING THE SUCCESS RATE

    Sparse, with termination on success, an episode return was exactly 0 or 1
    and `agent.mean_return()` WAS the rate — no band table, no decoding. The
    shaped terms in `So101TabletopConfig.custom_reward_gpu` subtract a
    per-step penalty from that same scalar, so a return is now dominated by
    integrated distance and says nothing directly about success. The driver's
    own greedy eval returns that shaped mean too.

    ⚠⚠ SUCCESS COMES FROM `META_IDX_GOAL_HELD`, NOT FROM THE REWARD. It used
    to be `reward > 0.5`, which held only while the reward was `+1 if holds`
    minus a penalty capped below 0.5. The reward is two `tolerance` terms in
    [0, 1] now and carries no success bonus, so that test would count a lane
    hovering near the blocks as solved. The hook writes the bit; every reader
    reads the bit.

    ⚠ NO `selective_reset_batch` IN THE LOOP, deliberately. A lane that
    succeeds TERMINATES and then keeps stepping with its done flag set; what
    is being counted is "did this lane ever meet its goal in one episode", so
    resetting mid-window would let one lane contribute twice and inflate the
    rate above what an episode is worth.
    """
    comptime AO = 2 * ACT_DIM
    var ao = ctx.enqueue_create_buffer[DT](N_ENVS * AO)

    # ⚠⚠ FOUR ROUNDS, BECAUSE 32 EPISODES CANNOT RESOLVE THE RATES THIS
    # FAMILY PRODUCES. One `reset_batch` gives 32 episodes and a granularity
    # of 1/32 = 0.031, and the criterion the whole family is judged by was
    # being read off that. `gather` scored 2/32 = 0.0625 against a baseline
    # whose 95% upper bound is 0.0117 — p ~ 0.054, decided by whether ONE
    # lane fell either way. 128 episodes costs 1200 greedy steps against the
    # run's 1,000,000 training ones.
    #
    # ⚠ THE SEEDS ARE FIXED AND CONSECUTIVE, so the init set is still frozen
    # and two runs remain comparable — that is the whole point of a fixed
    # eval seed, and randomising per run would trade the resolution back for
    # noise.
    #
    # ⚠ A FULL `reset_batch` PER ROUND, not `selective_reset_batch` mid-loop.
    # Resetting inside the step window lets a fast lane contribute twice and
    # inflates the denominator; a round is a clean set of N_ENVS episodes.
    comptime EVAL_ROUNDS = EVAL_ROUNDS_N
    # ⚠⚠ A RATE PER TASK, NOT ONE POOLED NUMBER. A two-task batch of `gather`
    # (0.2) and `settle` (1.0) pools to 0.6 — a number neither task has, and
    # one that moves when the lane split changes. The per-task curves ARE the
    # multi-task claim; a mean over tasks is exactly what hides it.
    var n = List[Int](length=n_tasks, fill=0)
    var nf = List[Int](length=n_tasks, fill=0)
    for rnd in range(EVAL_ROUNDS):
        var r = round_rate(
            agent, env, ctx, ao, UInt64(20260907 + rnd), n_tasks
        )
        for i in range(n_tasks):
            n[i] += r[0][i]
            nf[i] += r[1][i]
    var any = List[Float64]()
    var fin = List[Float64]()
    for i in range(n_tasks):
        # ⚠ THE DENOMINATOR IS THE TASK'S OWN LANE COUNT. 32 lanes over 3
        # tasks is 11/11/10; dividing all three by 32/3 would report the last
        # one low while the numbers still looked consistent.
        var d = Float64(lanes_for_task(i, N_ENVS, n_tasks) * EVAL_ROUNDS)
        any.append(Float64(n[i]) / d)
        fin.append(Float64(nf[i]) / d)
    return (any^, fin^)


def round_rate[M: ModelDefLike, C: Phyics3dEnvConfig](
    mut agent: AgentT[M, C], mut env: EnvT[M, C], ctx: DeviceContext,
    ao: DeviceBuffer[DT], seed: UInt64, n_tasks: Int,
) raises -> Tuple[List[Int], List[Int]]:
    """One round of `N_ENVS` greedy episodes: `(met at ANY step, met at the
    FINAL step)` as COUNTS PER TASK, so the caller can pool rounds without
    averaging averages and without pooling tasks."""
    comptime AO = 2 * ACT_DIM
    env.reset_batch[N_ENVS](ctx, seed)

    var solved = List[Bool](length=N_ENVS, fill=False)

    for step in range(C.MAX_STEPS):
        agent.trainer.select_greedy_action_batched[N_ENVS](
            ctx,
            LayoutTensor[DT, Layout.row_major(N_ENVS, OBS_DIM[M, C]), MutAnyOrigin](
                env.obs_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT_DIM), MutAnyOrigin](
                env.action_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, AO), MutAnyOrigin](
                mptr(ao.unsafe_ptr())
            ),
        )
        env.step_batch[N_ENVS](ctx, UInt64(step + 1))
        # ⚠⚠ THE GOAL BIT, NOT THE REWARD. This read `reward > 0.5`, which
        # was the same signal only while the reward was `+1 if holds` minus a
        # bounded penalty. The reward is two `tolerance` terms now and carries
        # no success bonus at all, so that test would count a lane hovering
        # near the blocks as a success.
        env.d.meta.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            if Float64(
                env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD]
            ) > 0.5:
                solved[e] = True

    # ⚠ THE FINAL-STEP READ IS THE `meta` STILL ON THE HOST FROM THE LAST
    # ITERATION — the loop downloads it every step, so this is the last step's
    # bit and needs no extra transfer.
    var n = List[Int](length=n_tasks, fill=0)
    var nf = List[Int](length=n_tasks, fill=0)
    for e in range(N_ENVS):
        # ⚠ THE SAME `lane_task` THE DRIVER WROTE WITH. Reading a lane back
        # under a different mapping reports each task's rate under another
        # task's name, and every total still adds up.
        var lt = lane_task(e, n_tasks)
        if solved[e]:
            n[lt] += 1
        if Float64(
            env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD]
        ) > 0.5:
            nf[lt] += 1
    return (n^, nf^)


def run_sac[M: ModelDefLike, C: Phyics3dEnvConfig](
    args: List[String],
    family_path: String,
    project: String,
    driver: String,
    default_task: String,
    shape_w_goal: Float64,
    shape_w_reach: Float64,
    goal_margin_default: Float64,
    reach_margin_default: Float64,
) raises:
    """The whole run: argv -> family -> env -> SAC -> eval -> the monitor.

    `args` is the entry's argv (index 0 the program), `family_path` its
    `.family`, `project` where its runs file, `driver` the entry's own path
    (recorded in `run.kv`), `default_task` what runs with no task named.
    The four shaping numbers are the concrete config's `SHAPE_W_GOAL`,
    `SHAPE_W_REACH`, `GOAL_MARGIN`, `REACH_MARGIN` — SO-101-family members
    the env-config trait does not carry — as the flags' defaults."""
    comptime EnvL = EnvT[M, C]
    comptime AgentL = AgentT[M, C]
    comptime OBS = OBS_DIM[M, C]
    # ⚠⚠ THE SEED IS SET AFTER THE PARSE, NOT HERE. `--seed` cannot be read
    # by a call that runs before the flags are read, and this line used to be
    # `seed_rng(42)` on the first line of `main`.
    var seed = 42

    # ⚠ `--warmup` EXISTS FOR THE BASELINE, not for tuning. Setting it at or
    # above `--steps` runs the whole loop on UNIFORM RANDOM actions with no
    # gradient step, which is the only way to measure what this task pays a
    # policy that has learned nothing — and the number every later rate has to
    # beat. `task_reachability.mojo` predicts it from geometry (3.2% of
    # uniform arm POSES meet the goal); this measures it through the dynamics.
    var num_steps = NUM_STEPS
    var warmup = WARMUP_STEPS
    var eval_every = EVAL_EVERY
    # ⚠ A RUN THAT PLATEAUS AT ITS FIRST EVAL AND IS STOPPED THERE HAS LEFT
    # NO CHECKPOINT to diagnose (50k default, first eval 25k): `--checkpoint-
    # every` sets the cadence so a stopped run still has a policy to load.
    var checkpoint_every = CHECKPOINT_EVERY
    # ⚠ `--demos` / `--demo-filter` / `--bc-weight` / `--bc-q-ratio` /
    # `--bc-only` are
    # HIL-SERL's (`deep_agents/hil_serl`, which documents each); `--init CKPT`
    # loads a `--bc-only` checkpoint (actor + twin critics; targets
    # hard-copied) into a normal run — the second half of a warm start.
    var hil = HilSerlConfig()
    var init_ckpt = String("")
    var task_name = String(default_task)
    # ⚠⚠ FLAGS BECAUSE THESE TWO ARE WHAT A FLAT RUN ACTUALLY NEEDS SWEPT.
    # Measured on a 230k-step `gather` run: `mean_q` reached **1151** while the
    # true episode return was **-7.5** — wrong sign, and 11x beyond the most
    # the task can pay even if every one of 300 steps scored the +1 success
    # bonus. The critic had diverged, and it was tracking the ENTROPY term,
    # not the task: corr(alpha, mean_q) = 0.88 over 111 points, with
    # mean_q ~ 2500 * alpha throughout.
    #
    # The task's own contribution to Q is r/(1-gamma) = -0.025 * 95 = -2.4, so
    # at alpha = 0.26 the task was about 0.2% of the value function the actor
    # was maximising. That is a REWARD SCALE problem — SAC's per-step reward
    # here is 0.025 where HalfCheetah's is O(1..10) — and the two levers that
    # reach it without changing what a reward MEANS are the entropy target and
    # the initial temperature.
    #
    # ⚠ `-ACT_DIM` IS THE STANDARD HEURISTIC AND IT IS WHAT FORCES ALPHA UP.
    # Making it more negative (-12, -24) drives alpha down and shrinks the
    # entropy contribution to Q. Falsifiable in a short run: if the mechanism
    # is right, `mean_q` should fall roughly in proportion.
    var target_entropy = -Scalar[DT](ACT_DIM)
    var init_alpha = Scalar[DT](0.2)
    # ⚠⚠ 0.50/0.25 IS THE ONLY PAIR THE CRITIC HAS SURVIVED. Measured, all at
    # 32 envs / 32 updates / tau 0.0025 — the SAME 7.7% tracking rate:
    #
    #     0.50 / 0.25   mean_q -> -6.2 converging, alpha -> 0.0014, 290k steps
    #     0.10 / 0.70   mean_q -> 5743, alpha -> 2.2, critic_loss 17640
    #
    # So the tracking rate is NECESSARY AND NOT SUFFICIENT, and the weights
    # are the other axis. Hypothesis fitted to two points, recorded as such:
    # the reach term is `|gripper - subject|` and moves as fast as the arm,
    # while the goal term is a separation between two props that barely moves,
    # so weighting the fast one at 7x the slow one raises the target's
    # step-to-step variance. What is MEASURED is only that 0.10/0.70 diverges
    # where 0.50/0.25 does not, at identical everything else.
    #
    # ⚠⚠ THE REWARD SCALE IS THE OPEN QUESTION ON THIS FAMILY, so it is a flag.
    # Three 190k-step runs at 0.10/0.05 held `mean_reward` at -0.024 from the
    # first diagnostic sample to the last — through an alpha fix and an
    # observation widening — while `mean_q` ran to 508. With `mean_done` at
    # 8e-05 nothing anchors the value function except the reward, and 0.024
    # per step is 200x too small to. `SoArm101ReachConfig`, which DOES train
    # on this robot, pays a `tolerance` in [0, 1] every step.
    # ⚠ TEXT, NOT A FLOAT, so a comma list survives the parse — these are
    # PER TASK in a multi-task batch (`per_task` broadcasts or splits).
    var shape_goal = String(shape_w_goal)
    var shape_reach = String(shape_w_reach)
    # ⚠⚠ THE MARGINS ARE FLAGS TOO, AND THEY HAVE TO BE. They became per-lane
    # `meta` words precisely because they are a per-TASK quantity, and a
    # per-task quantity that needs a rebuild to change is not one. Measured
    # goal distances under a random policy — `task_shaping_probe.mojo <task>`:
    #
    #     gather   Near    0.139 m     lift  Above  0.030 m     settle  On  0
    #
    # so one margin cannot serve them. `gather` trained at 0.10 against 0.139,
    # a margin/distance ratio of 0.72 and a tolerance of 0.012 at random —
    # the goal term contributing almost nothing at first and the reach term
    # doing the early work. Reproducing that ratio on `lift` is a margin of
    # about 0.02.
    var goal_margin = String(goal_margin_default)
    var reach_margin = String(reach_margin_default)
    # ⚠⚠ THE TARGET NETWORK'S TRACKING RATE, WHICH `N_ENVS` SETS BY ACCIDENT.
    # `updates_per_step = N_ENVS` keeps UTD at 1 — 64 transitions collected,
    # 64 gradient steps — and that is the number people quote. It is not the
    # number that governs the CRITIC's stability. Polyak runs ONCE PER UPDATE
    # at `tau`, so a driver iteration moves the target by
    #
    #     1 - (1 - tau)^updates_per_step
    #
    # which at tau 0.005 is 27% for 64 updates and 15% for 32. A target that
    # moves a quarter of the way to the online net between env steps is barely
    # a target, and chasing it is what a runaway critic looks like:
    # `mean_next_q` sat about +5 above `mean_q` at EVERY sample of run 3.
    #
    # ⚠ THIS IS WHY 64 ENVS VS 32 IS NOT THE NO-OP I SAID IT WAS. The UTD
    # argument was right and the conclusion was wrong — the two examples that
    # DO train on this stack (`sac_so_arm101_reach_training_gpu.mojo` and
    # `sac_half_cheetah_training_gpu.mojo`) both run 32.
    # ⚠⚠ `N_ENVS` AGAIN, WHICH IS UTD 1 — BUT AT 32 ENVS, NOT 64. The
    # measured history, same task and same everything else:
    #
    #                    64 env / 64 upd     64 env / 16 upd
    #                    (27.4% tracking)    (7.7% tracking)
    #     mean_q          0 -> 14820          0 -> -6.63, converging
    #     next_q - q      about +5            +0.048
    #     critic_loss     190800              0.0035
    #     mean_reward     -0.115, FLAT        -0.112 -> -0.0889, improving on
    #                                         70 of 79 samples
    #     eval return     -25 .. -51          -28.7 -> -20.1
    #
    # The second column trains and costs sample efficiency: UTD 16/64 = 0.25.
    # 32 envs with 32 updates is UTD 1 at 14.8% tracking — between the two,
    # and exactly what the working references use. ⚠ THAT COMBINATION IS NOT
    # YET MEASURED ON THIS FAMILY; the 7.7% column is. If the critic diverges
    # again, `--updates-per-step 16` is the configuration known to hold.
    var updates_per_step = N_ENVS
    # ⚠⚠ 0.0025, NOT SAC'S USUAL 0.005, BECAUSE 32 UPDATES OF 0.005 IS 14.8%
    # AND THIS FAMILY DIVERGES THERE. Lowering `tau` rather than the update
    # count is what buys the safe tracking rate WITHOUT paying UTD:
    #
    #     32 upd, tau 0.005    14.8%   UTD 1.00   mean_q -> 1152, diverged
    #     16 upd, tau 0.005     7.7%   UTD 0.50   mean_q -> -6.63, converged
    #     32 upd, tau 0.0025    7.7%   UTD 1.00   mean_q -> -3.10, converged
    #
    # ⚠ THE THIRD ROW IS WHAT SETTLED IT. Rows one and two differ in TWO
    # things, so neither could say whether the tracking rate or the UTD was
    # the axis; row three holds UTD at 1 and moves only the rate, and the
    # critic came back healthy — `next_q - q` +0.18 against +5.85 at 14.8%,
    # `critic_loss` 0.013 against 1667. The tracking rate is the axis.
    var tau = Scalar[DT](0.0025)
    # ⚠⚠ AN UNRECOGNISED ARGUMENT IS A HARD ERROR, AND IT DID NOT USED TO BE.
    # The task was settable ONLY by `--task <name>` while every other tool in
    # `examples/tasks/` takes it POSITIONALLY, so
    #
    #     ... sac_task_gpu.mojo so101_gather_bricks --steps 1000000 ...
    #
    # matched no branch, was dropped in silence, and ran `DEFAULT_TASK`. Two
    # 17-minute `gather` runs on a 5090 were `lift` runs, and both printed
    # `so101_lift_brick` in a banner that nobody reads when they know what
    # they launched. The earlier `lift` runs were right BY ACCIDENT — lift is
    # the default — which is exactly why it went unnoticed.
    #
    # A silently ignored argument makes every typo a full-length run of the
    # wrong experiment, so the loop below refuses anything it does not know.
    var positional = List[String]()
    for i in range(1, len(args)):
        var a = String(args[i])
        # skip a flag's VALUE — it is consumed by the flag, not positional
        if i > 1:
            var prev = String(args[i - 1])
            if prev.startswith("--"):
                continue
        if not a.startswith("--"):
            positional.append(a)
    if len(positional) > 1:
        raise Error(
            "sac task: more than one positional argument ("
            + String(len(positional)) + "). The first is the task name;"
            " everything else must be a flag."
        )
    if len(positional) == 1:
        task_name = positional[0]

    for i in range(1, len(args)):
        var a = String(args[i])
        if not a.startswith("--"):
            continue                      # a positional, or a flag's value
        if a == "--steps" and i + 1 < len(args):
            num_steps = Int(String(args[i + 1]))
        elif a == "--warmup" and i + 1 < len(args):
            warmup = Int(String(args[i + 1]))
        elif a == "--eval-every" and i + 1 < len(args):
            eval_every = Int(String(args[i + 1]))
        elif a == "--checkpoint-every" and i + 1 < len(args):
            checkpoint_every = Int(String(args[i + 1]))
        elif a == "--task" and i + 1 < len(args):
            task_name = String(args[i + 1])
        elif a == "--target-entropy" and i + 1 < len(args):
            target_entropy = Scalar[DT](Float64(String(args[i + 1])))
        elif a == "--alpha" and i + 1 < len(args):
            init_alpha = Scalar[DT](Float64(String(args[i + 1])))
        elif a == "--shape-goal" and i + 1 < len(args):
            shape_goal = String(args[i + 1])
        elif a == "--shape-reach" and i + 1 < len(args):
            shape_reach = String(args[i + 1])
        elif a == "--goal-margin" and i + 1 < len(args):
            goal_margin = String(args[i + 1])
        elif a == "--reach-margin" and i + 1 < len(args):
            reach_margin = String(args[i + 1])
        elif a == "--updates-per-step" and i + 1 < len(args):
            updates_per_step = Int(String(args[i + 1]))
        elif a == "--tau" and i + 1 < len(args):
            tau = Scalar[DT](Float64(String(args[i + 1])))
        elif a == "--seed" and i + 1 < len(args):
            seed = Int(String(args[i + 1]))
        elif a == "--init" and i + 1 < len(args):
            init_ckpt = String(args[i + 1])
        elif hil.try_parse(
            a, String(args[i + 1]) if i + 1 < len(args) else String(""),
            i + 1 < len(args), "sac task",
        ):
            # ⚠ `--bc-only` IS A FLAG WITHOUT A VALUE — the one exception to
            # "every flag takes a value" below; `try_parse` matches it first.
            pass
        else:
            # ⚠ INCLUDES A KNOWN FLAG WITH NO VALUE, which falls through the
            # `i + 1 < len(args)` guards above and would otherwise be dropped
            # exactly as silently as a misspelling.
            raise Error(
                "sac task: unrecognised or valueless argument '" + a + "'."
                " The task name is POSITIONAL (or `--task <name>`); every"
                " flag takes a value. Refusing rather than running"
                " " + default_task + " for an hour."
            )

    # `--bc-weight` without `--demos`, `--bc-only` without `--bc-weight`:
    # refused here, before an env is built.
    hil.validate("sac task")

    # ⚠⚠ ONE SEED FOR BOTH RNGs — the host's (uniform warmup actions, network
    # init) and the env's per-lane device stream. They were two separate 42s,
    # so there was no single knob to turn and every run was the same draw.
    #
    # ⚠ A RUN OF THIS CONFIGURATION IS ONE SAMPLE. Two `gather` runs at an
    # identical logged config reached 0.5625 and 0.0, the second by a critic
    # that peaked 273x its fixed point and decayed back. Read `critic_health`
    # at the end of BOTH before comparing their rates.
    seed_rng(seed)

    print("=" * 72)
    print("SAC on the task family —", task_name, "(GPU)")
    print("=" * 72)

    # ── the task, on the host ─────────────────────────────────────────────
    var f = load_family(family_path)
    # ⚠⚠ ONE TASK OR SEVERAL, BY THE SAME PATH. `task_name` may be a comma
    # list; a single task is `n_tasks == 1` and every loop below still runs,
    # so the multi-task batch is not a second code path that can rot while
    # the single-task one is the only one exercised.
    var task_names = split_csv(task_name)
    var n_tasks = len(task_names)
    var tasks = List[TaskSpec]()
    for i in range(n_tasks):
        var ti = load_task(task_path(f, task_names[i]))
        validate_task_against_family(ti, f)
        # ⚠ EVERY TASK MUST BE THIS FAMILY'S. The env is monomorphised on one
        # family, so a task from another has the wrong slot table and its
        # body ids address the wrong props — which evaluates, and is wrong.
        if ti.family != f.name:
            raise Error(
                "sac task: '" + task_names[i] + "' belongs to family '"
                + ti.family + "' but the env is built for '" + f.name
                + "'. A batch shares one family by construction."
            )
        tasks.append(ti^)
    ref t = tasks[0]
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)

    # ⚠ AFTER `load_family`, so the env reference is the family's OWN name
    # rather than a second copy of it written by hand here — the typed
    # `family:<name>` form the project layer resolves.
    var run = RunContext(
        project=project,
        driver=driver,
        # ⚠ COMMAS DO NOT BELONG IN A PATH. A two-task run's slug was
        # `sac-so101_gather_bricks,so101_settle_brick`, which is a directory
        # name every shell, tar and CSV consumer has an opinion about.
        slug=String("sac-") + task_name.replace(",", "+"),
        env=String("family:") + f.name,
        task=task_name,
        seed=seed,
        device=String("gpu"),
    )
    var ckpt_path = run.checkpoint_path(String("last"))
    var csv_path = run.metrics_path()
    print("  run:", run.dir)

    # ── the per-lane words, one set per TASK ──────────────────────────────
    #
    # ⚠⚠ EVERY ONE OF THESE IS ALREADY A PER-LANE `meta` FIELD. The tape, the
    # active mask, the init-region words and the shaping words were made per
    # lane precisely so a batch could carry more than one task; until now the
    # driver wrote the same values into all of them. Nothing about the env or
    # the reward kernel changes here — only which words land in which lane.
    var tapes = List[List[Float64]]()
    var masks = List[Float64]()
    for i in range(n_tasks):
        ref ti = tasks[i]
        var gi = bind_goal(
            parse_goal(ti.goal), f, fmd.body_names, fmd.site_names
        )
        require_tier_a(gi, ti.name)
        # ⚠ ONE region table on device and a term's region index is ignored
        # there.
        require_gpu_regions(gi, ti.name)
        tapes.append(encode_goal(gi))
        masks.append(active_mask(ti, f))

    print("  task     :", t.name)
    print("  language :", t.language)
    print("  goal     :", t.goal)
    print("  OBS  :", OBS, " ACT_DIM:", ACT_DIM, " N_ENVS:", N_ENVS)
    print("  steps    :", num_steps, " warmup:", warmup,
          "(baseline run)" if warmup >= num_steps else "")
    print("  action_scale:", ACTION_SCALE, "(NORMALIZED_ACTIONS is True)")
    print("  target_entropy:", target_entropy, " init_alpha:", init_alpha)
    hil.print_banner(warmup, WARMUP_STEPS)
    print("  shape weights: goal", shape_goal, " reach", shape_reach,
          " (tolerance margins", goal_margin, "/", reach_margin, "m)")
    # ⚠ THE NUMBER THAT ACTUALLY GOVERNS CRITIC STABILITY, printed because it
    # is derived and nobody sets it directly.
    var track = 1.0 - (1.0 - Float64(tau)) ** Float64(updates_per_step)
    print("  updates/step:", updates_per_step, " tau:", tau,
          " -> the target moves", track, "per iteration")
    print("  UTD:", Float64(updates_per_step) / Float64(N_ENVS))
    # ⚠⚠ THE KNOWN-BAD BAND, MEASURED ON THIS FAMILY. Three runs, same task:
    #
    #     27.4%  (64 env / 64 upd)   mean_q -> 14820, mean_reward FLAT
    #     14.8%  (32 env / 32 upd)   mean_q -> 1152,  next_q-q +5.9, no trend
    #      7.7%  (64 env / 16 upd)   mean_q -> -6.63 converging, reward moving
    #
    # So the threshold is somewhere in (7.7%, 14.8%] and it is BELOW what the
    # working references use — `sac_so_arm101_reach_training_gpu.mojo` trains
    # at 14.8%, this family does not. Warned rather than refused, because the
    # rate is a legitimate sweep axis and the band is three points wide, not a
    # law.
    #
    # ⚠ TO KEEP UTD AT 1 AND STILL LAND AT 7.7%, LOWER `tau`, NOT THE UPDATE
    # COUNT: 32 updates at tau 0.0025 is 7.70%, which is the tracking rate
    # that held and the sample efficiency that 16-updates gave away.
    if track > 0.10:
        print()
        print("  ⚠⚠ THE TARGET MOVES", track, "PER ITERATION, AND THIS FAMILY")
        print("  HAS DIVERGED ABOVE 0.10 IN EVERY RUN SO FAR (14.8% and")
        print("  27.4%; 7.7% converged). Watch `mean_q` against")
        print("  `mean_reward / (1 - gamma)` — if it passes zero the run is")
        print("  already lost. `--tau 0.0025` at", updates_per_step,
              "updates gives 7.7% with UTD unchanged.")
        print()

    # ⚠⚠ AN ACTIVE FREE SLOT WOULD FALL FOR THE WHOLE EPISODE. Refused here
    # rather than trained around — see the header. The failure is not a crash:
    # it is a policy learning from an observation with a prop falling through
    # it, and the curve looks like a hard task.
    # ⚠ THE REFUSAL THAT USED TO BE HERE IS GONE, AND THE INIT WORDS ARE WHY.
    # This file refused any task activating a FREE slot, because nothing
    # placed one at a GPU reset and a prop would start every episode 50 m up
    # and fall. `C.init_qpos_gpu` now samples them per lane
    # from `META_IDX_INIT_REGION_*`, gated against the host sampler coordinate
    # for coordinate by `tests/tasks/test_device_placement.mojo`.
    # ⚠ PER TASK: `--goal-margin 0.10,0.21` gives each its own, because what
    # a margin is worth depends on the task's own distance scale. One value
    # broadcasts.
    var wg = per_task(shape_goal, n_tasks, String("--shape-goal"))
    var wr = per_task(shape_reach, n_tasks, String("--shape-reach"))
    var mg = per_task(goal_margin, n_tasks, String("--goal-margin"))
    var mr = per_task(reach_margin, n_tasks, String("--reach-margin"))
    var iws = List[List[Float64]]()
    var sws = List[List[Float64]]()
    for i in range(n_tasks):
        ref ti = tasks[i]
        iws.append(init_region_words(ti, f))
        # ⚠ VALIDATED HERE, not at the write. `shaping_words` refuses a
        # nonzero weight with a zero margin — `tolerance` with margin 0 is a
        # HARD indicator, so the term goes sparse while the run still looks
        # shaped.
        sws.append(shaping_words(wg[i], wr[i], mg[i], mr[i]))
        var n_active_free = 0
        ref iwi = iws[i]
        for j in range(len(iwi)):
            if iwi[j] > 0.0:   # the word is region_index + 1; 0 = not placed
                n_active_free += 1
        print("  ", ti.name, ": lanes", lanes_for_task(i, N_ENVS, n_tasks),
              " free slots placed at reset", n_active_free, "of", len(iwi),
              " margins", mg[i], "/", mr[i], " weights", wg[i], "/", wr[i])

    with DeviceContext() as ctx:
        # ── the logger: CSV always, the dashboard when it is configured ──
        #
        # ⚠⚠ A SPARSE 0/1 RETURN TELLS YOU ALMOST NOTHING WHILE IT IS ZERO,
        # which is exactly the regime this task sits in. `diag_every` flushes
        # the SAC bundle — `mean_q`, `critic_loss`, `actor_loss`, `alpha`,
        # `mean_reward`, `train_steps` — and those move long before the return
        # does: a critic whose `mean_q` is drifting up has found SOMETHING to
        # predict, and an `alpha` pinned at its ceiling says the actor is
        # still being paid to be random. Without them a flat return is
        # indistinguishable from a broken reward.
        #
        # ⚠ `RemoteLogger` WITH NO URL IS INERT — its POST sink is built
        # lazily on the first payload — so this is safe with no `.env` and
        # costs nothing. The CSV is the local artefact that survives the
        # dashboard being down and is what a later run gets diffed against.
        var env_vars = load_dotenv()
        var remote = RemoteLogger(
            server_url=env_vars.get("NOEIRA_CLOUD_URL", ""),
            run_name=run.name(),
            run_id=run.id,
            buffer_size=64,
            api_key=env_vars.get("NOEIRA_CLOUD_API_KEY", ""),
            spool_path=run.dir + "/remote.spool",
        )
        # ⚠ THE CONFIG GOES TO BOTH HALVES: `/runs` for the dashboard and
        # `metrics.config.kv` beside the CSV, so a CSV read later still says
        # what produced it. Hence the composite is built BEFORE `set_config`.
        var logger = CompositeLogger(CsvLogger(csv_path), remote^)
        logger.set_config("algorithm", "SAC")
        logger.set_config("family", f.name)
        logger.set_config("task", task_name)
        logger.set_config("goal", t.goal)
        logger.set_config("language", t.language)
        logger.set_config("target", "gpu")
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("warmup", String(warmup))
        logger.set_config("horizon", String(C.MAX_STEPS))
        logger.set_config("action_scale", String(ACTION_SCALE))
        logger.set_config("target_entropy", String(target_entropy))
        logger.set_config("init_alpha", String(init_alpha))
        logger.set_config("shape_w_goal", String(shape_goal))
        logger.set_config("shape_w_reach", String(shape_reach))
        logger.set_config("updates_per_step", String(updates_per_step))
        logger.set_config("tau", String(tau))
        hil.log_config(logger)
        logger.set_config("target_track_per_iter", String(track))
        # ⚠ THE MEASURED FLOOR TRAVELS WITH THE RUN. A rate on a dashboard is
        # unreadable without it — 0.05 is nothing on `reach` and would be real
        # on `lift` — and a config field is the only part of a run that is
        # still there when somebody opens the chart a week later.
        var bl0 = baselines_for(task_name)
        logger.set_config("baseline_random", String(bl0[0]))
        logger.set_config("baseline_untrained_greedy", String(bl0[1]))
        logger.set_config("baseline_measured", String(bl0[2]))
        # ⚠⚠ PER TASK, KEYED BY TASK NAME. The weights and margins were one
        # value each, which is right for one task and a lie for two — a batch
        # whose tasks carry different margins would have recorded one of
        # them. It cost a real conclusion once: two runs' shaped costs were
        # decomposed under ASSUMED weights, the decompositions disagreed (a
        # goal distance of -0.030 m), and nothing in the files could say
        # which weight had actually run.
        logger.set_config("n_tasks", String(n_tasks))
        for i in range(n_tasks):
            var k = task_names[i] + String(".")
            logger.set_config(k + String("shape_w_goal"), String(wg[i]))
            logger.set_config(k + String("shape_w_reach"), String(wr[i]))
            logger.set_config(k + String("goal_margin"), String(mg[i]))
            logger.set_config(k + String("reach_margin"), String(mr[i]))
            logger.set_config(
                k + String("lanes"), String(lanes_for_task(i, N_ENVS, n_tasks))
            )
        logger.set_config("obs_dim", String(OBS))
        # ⚠⚠ THE ONE SETTING THAT WAS NOT RECORDED WAS THE DECISIVE ONE.
        # `TERMINATE_ON_UNHEALTHY` is a comptime env parameter, not a flag, so
        # it was in no config — and it is what separated fourteen runs of a
        # diverging critic from the first one that trained.
        logger.set_config(
            "terminate_on_unhealthy", String(EnvL.TERMINATE_ON_UNHEALTHY)
        )
        # ⚠ AFTER the config and before step 0 — `register_run` seeds the
        # dashboard's config from the run (id, project, task, commit, seed,
        # host) and then POSTs `/runs`. Registering any earlier would ship an
        # empty config; registering lazily on the first metric batch — which is
        # what `flush` still does for drivers that never call this — means a run
        # that dies before step 0 never appears at all.
        register_run(run, logger)

        # ⚠⚠ THE ARTIFACT UPLINK. §7: a checkpoint leaves the box WHILE the run
        # is going, so a box that dies at 3am does not take the weights with
        # it. `sink_for_run` returns None when `.env` names no monitor — a box
        # with no credentials must still train — and every announce downstream
        # is a no-op on a None, so there is no branch to write here.
        var artifacts = sink_for_run(run.id, run.dir)


        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = AgentL(
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=tau,
            action_scale=ACTION_SCALE,
            init_alpha=init_alpha,
            target_entropy=target_entropy,
            learning_starts=warmup,
            window_size=100,
            # See wiring fact 3 in the header: a return here is 0 or 1.
            initial_episode_fill=0.0,
        )
        var env = EnvL(ctx)

        # ── the demonstrations, into the replay's pinned prefix ───────────
        #
        # HIL-SERL / RLPD (`deep_agents/hil_serl`): demos pinned as the demo
        # half of every batch, then the BC term / `--bc-only`. Before the
        # first train step, which a CUDA graph may capture.
        apply_hil_serl(agent.trainer, hil, logger, "sac task")
        if init_ckpt.byte_length() > 0:
            if not Path(init_ckpt).exists():
                raise Error("sac task: --init: no such checkpoint: " + init_ckpt)
            agent.load(init_ckpt)
            print("  init     : actor + critics loaded from", init_ckpt)

        # ── the region table, once; the tape and mask, once per lane ──────
        #
        # ⚠ WRITTEN BEFORE THE LOOP AND NEVER AGAIN, which is only safe
        # because `_reset_env_lane` writes just `META_IDX_STEP_COUNT` and
        # leaves the rest of `meta` alone (`gpu/constants.mojo`). Every lane
        # runs the SAME task here, so there is no per-lane variation to
        # maintain — a multi-task run would still write these once, with
        # different words per lane.
        # ⚠ THE HALF-HEIGHT IS THE FIFTH NUMBER AND IT IS REQUIRED. Without
        # it the device would use `IN_HALF_HEIGHT` while `eval.eval_goal` used
        # the region's own band — a CPU/GPU disagreement inside the reward.
        var cw = region_table_words(
            rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
            rheights[0],
        )
        for i in range(MODEL_CURRICULUM_SIZE):
            env.mf.curriculum.data[i] = Scalar[DT](cw[i])
        env.mf.curriculum.upload(ctx)

        for e in range(N_ENVS):
            # ⚠⚠ THE LANE'S OWN TASK. `lane_task` is the single definition of
            # this mapping and the greedy evaluation reads a lane's success
            # back through the SAME call — an offset between the two would
            # report each task's rate under another task's name, sum
            # correctly, and name no error.
            var lt = lane_task(e, n_tasks)
            ref tape = tapes[lt]
            ref iw = iws[lt]
            ref sw = sws[lt]
            var mask = masks[lt]
            for w in range(TAPE_WORDS):
                env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + w] \
                    = Scalar[DT](tape[w])
            env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_ACTIVE] = \
                Scalar[DT](mask)
            # ⚠⚠ THE INIT WORDS SURVIVE EVERY RESET, which is the whole point:
            # `_reset_env_lane` writes only META_IDX_STEP_COUNT, so writing
            # these once before the loop makes EVERY later reset place the
            # props. A driver that wrote them per episode would be doing the
            # host round-trip this exists to remove.
            for j in range(len(iw)):
                env.d.meta.data[
                    e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j
                ] = Scalar[DT](iw[j])
            # ⚠⚠ PER LANE, EVEN THOUGH EVERY LANE RUNS THE SAME TASK HERE.
            # They are per-lane words because a multi-task batch needs them
            # to be — writing them the same way whether or not the run
            # happens to be single-task is what stops the multi-task driver
            # being a different code path.
            for j in range(SHAPING_WORDS):
                env.d.meta.data[
                    e * METADATA_SIZE + META_IDX_SHAPE_W_GOAL + j
                ] = Scalar[DT](sw[j])
        env.d.meta.upload(ctx)

        # ── a SECOND env, for greedy eval ─────────────────────────────────
        #
        # ⚠⚠ IT NEEDS THE SAME THREE UPLOADS, AND A MISSING ONE IS SILENT.
        # `curriculum` and `meta` are per-ENV-INSTANCE device buffers, so an
        # eval env constructed without them evaluates a tape of zeros — op 0
        # is `OP_IN`, with body 0 against region 0 — which is a real, wrong
        # predicate that returns a plausible rate rather than an error. The
        # driver's `eval_env` is an ISOLATED BatchedEnv by design (it must
        # never share the training env's state), and isolation is exactly what
        # makes this easy to forget.
        var eval_env = EnvL(ctx)
        for i in range(MODEL_CURRICULUM_SIZE):
            eval_env.mf.curriculum.data[i] = Scalar[DT](cw[i])
        eval_env.mf.curriculum.upload(ctx)
        for e in range(N_ENVS):
            # ⚠⚠ THE LANE'S OWN TASK. `lane_task` is the single definition of
            # this mapping and the greedy evaluation reads a lane's success
            # back through the SAME call — an offset between the two would
            # report each task's rate under another task's name, sum
            # correctly, and name no error.
            var lt = lane_task(e, n_tasks)
            ref tape = tapes[lt]
            ref iw = iws[lt]
            ref sw = sws[lt]
            var mask = masks[lt]
            for w in range(TAPE_WORDS):
                eval_env.d.meta.data[
                    e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + w
                ] = Scalar[DT](tape[w])
            eval_env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_ACTIVE] = \
                Scalar[DT](mask)
            for j in range(len(iw)):
                eval_env.d.meta.data[
                    e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j
                ] = Scalar[DT](iw[j])
            for j in range(SHAPING_WORDS):
                eval_env.d.meta.data[
                    e * METADATA_SIZE + META_IDX_SHAPE_W_GOAL + j
                ] = Scalar[DT](sw[j])
        eval_env.d.meta.upload(ctx)
        print("  ok: region table, tape and active mask uploaded to BOTH envs")

        print("-" * 72)
        var t0 = perf_counter_ns()
        _ = agent.train[
            EnvL,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=True,
            USE_ENV_CUDA_GRAPH=False,
            L=CompositeLogger[CsvLogger, RemoteLogger],
        ](
            env,
            num_steps,
            rng_seed=UInt64(seed),
            updates_per_step=updates_per_step,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            episode_sync_every=32,
            checkpoint_every=checkpoint_every,
            checkpoint_path=ckpt_path,
            artifacts=artifacts,
            run_dir=run.dir,
            # ⚠ GREEDY, on a SEPARATE env, at a FIXED eval seed — the
            # criterion number. `mean_return()` below is measured under SAC's
            # stochastic policy and understates what the actor has learned;
            # this is the actor mean, with no sampling. Both are printed
            # because a large gap between them is itself a reading (an actor
            # that is good and an alpha that is still high).
            eval_env=Pointer(to=eval_env).as_unsafe_any_origin(),
            eval_every=eval_every,
            eval_episodes=N_ENVS,   # the PERIODIC eval; the final one pools rounds
            eval_max_steps=C.MAX_STEPS + 1,
        )
        var secs = Float64(perf_counter_ns() - t0) / 1e9

        # ⚠⚠ TWO DIFFERENT NUMBERS NOW, AND THEY USED TO BE ONE. `mean_return`
        # is the SHAPED return — dominated by integrated distance, and what
        # SAC actually optimises. The success rate has to be measured, and
        # `greedy_success_rate` is that measurement: one greedy episode per
        # lane, counting `META_IDX_GOAL_HELD`.
        var shaped = Float64(agent.mean_return())
        var rates = greedy_success_rate(agent, eval_env, ctx, n_tasks)
        ref any_rates = rates[0]
        ref fin_rates = rates[1]
        # ⚠ `rate` IS THE FIRST TASK'S, and it is only meaningful as "the"
        # rate when there is one task. Everything below that judges a single
        # number uses it; the per-task block above is what a batch is read
        # from.
        var rate = any_rates[0]
        var rate_final = fin_rates[0]
        print("-" * 72)
        print("  env steps          :", num_steps)
        print("  elapsed            :", secs, "s")
        print("  episodes           :", agent.ep_count())
        print("  shaped mean return :", shaped,
              "(last 100 episodes, POOLED over tasks)" if n_tasks > 1
              else "(last 100 episodes)")
        # ⚠⚠ PER TASK, AND NEVER AVERAGED ACROSS THEM. `gather` at 0.20 beside
        # `settle` at 1.0 has a mean of 0.60 — a number neither task has, that
        # moves with the lane split, and that would let one task's collapse
        # hide inside the other's success. The per-task rates ARE the
        # multi-task claim.
        for i in range(n_tasks):
            var lanes_i = lanes_for_task(i, N_ENVS, n_tasks)
            print("  ", task_names[i], ":  ANY", any_rates[i],
                  "  held at END", fin_rates[i],
                  "  (", lanes_i * EVAL_ROUNDS_N, "episodes =", lanes_i,
                  "lanes x", EVAL_ROUNDS_N, "rounds )")
        if n_tasks == 1:
            print("  SUCCESS RATE       :", rate, "(greedy,",
                  N_ENVS * EVAL_ROUNDS_N, "episodes =", N_ENVS, "lanes x",
                  EVAL_ROUNDS_N, "rounds, met at ANY step)")
            print("  held at the END    :", rate_final,
                  "— the stronger claim; for a goal that holds at RESET the"
                  " any-step rate is 1.0 by construction")
        # ⚠ LOGGED, NOT ONLY PRINTED. It is the criterion the whole family is
        # judged by and it was reaching stdout and nothing else, so no chart
        # ever carried it and no two runs could be compared on it.
        # ⚠ ONE SERIES PER TASK, KEYED BY NAME, so two tasks in one batch
        # give two curves rather than one average. The unsuffixed keys stay
        # for a single-task run, because every chart and every comparison
        # made so far reads them.
        for i in range(n_tasks):
            logger.log_scalar(
                String("eval_success_rate_") + task_names[i],
                any_rates[i], num_steps,
            )
            logger.log_scalar(
                String("eval_success_rate_final_") + task_names[i],
                fin_rates[i], num_steps,
            )
        if n_tasks == 1:
            logger.log_scalar(String("eval_success_rate"), rate, num_steps)
            logger.log_scalar(
                String("eval_success_rate_final"), rate_final, num_steps
            )
        logger.log_scalar(String("eval_shaped_return"), shaped, num_steps)

        # ⚠⚠ `close()` GOES **AFTER** THE LAST `log_scalar`, AND IT DID NOT.
        # It sat above the greedy eval, so the two metrics this run exists to
        # produce were logged to a CLOSED logger and dropped — `close()`
        # drains the queue and joins the POST thread, and anything queued
        # afterwards has nothing to carry it. The 990k-step run's
        # `eval/success_rate` (now `eval_success_rate`) is missing from its CSV for exactly this reason
        # and NOT because the export was early, which is what I assumed.
        #
        # ⚠ IT IS SILENT. `log_scalar` after `close` neither raises nor warns;
        # the metric simply never appears, and a missing key reads as "the run
        # did not get that far".
        #
        # ⚠⚠ AND THE VERDICT GOES TO THE MONITOR BEFORE `close()`. `close()`
        # sends `/finish` with `done` and an EMPTY outcome for any run that did
        # not `finish` first, and the first finish wins — so setting the
        # outcome on `run` below, after this, reached run.kv and never the
        # dashboard. Every SAC-family run on the monitor had a blank outcome.
        var outcome = (
            String("success_rate=") + String(rate)
            + " success_rate_final=" + String(rate_final)
            + " shaped_return=" + String(shaped)
        )
        logger.finish(String("done"), outcome)
        logger.close()
        _ = logger        # keeps `logger_ptr` alive to here

        # ⚠ CLOSED BEFORE THE VERDICT IS WRITTEN, NOT AFTER. `close_sink`
        # drains what is queued and JOINS, so a checkpoint saved in the last
        # iteration is on the far side before the process can exit — which is
        # the entire point. It also prints the transfer accounting, including
        # what was abandoned: those are artifacts still only on this box.
        close_sink(artifacts)

        # ⚠⚠ THE RUN RECORDS ITS OWN VERDICT. "I forgot if the checkpoint was
        # successful" is pain 1, and no naming convention fixes it — a written
        # `outcome=` does. ⚠ `success_rate` is NOT the return: this family's
        # `RETURN != SUCCESS`, and it is the rate the whole family is judged by.
        run.set_outcome(outcome)

        print("  csv                :", csv_path)
        print("  checkpoint         :", ckpt_path)
        print("  run record         :", run.kv_path())

        # ⚠⚠ READ AFTER `close()`, WHICH IS THE ONLY POINT THE CSV IS WHOLE.
        # `CsvLogger` streams rows as they happen but the last of them are
        # only guaranteed on disk once the queue is drained, and `close()` is
        # what drains it. This also means the check costs one file read of a
        # file the run wrote anyway — no per-iteration hook in the shared
        # driver, which is where these metrics are actually produced.
        var health = critic_health(csv_path, 0.99)
        var peak_q = health[0]
        var peak_loss = health[1]
        var q_fp = health[2]
        print("  Q fixed point      :", q_fp, "(this run's own"
              " mean_reward / (1 - gamma))")
        print("  peak mean_q        :", peak_q,
              "(" + String(peak_q / q_fp) + "x the fixed point)"
              if q_fp > 0.0 else "")
        print("  peak critic_loss   :", peak_loss)
        # ⚠ TWO BANDS, BECAUSE THE MIDDLE OF THE RANGE TURNED OUT TO BE
        # POPULATED. This said 10x sat "in the middle of an empty decade" on
        # the strength of two runs, 1.00x and 89x. A later `lift` run peaked
        # at 4.19x: it did not diverge, it overshot and decayed — and it
        # learned nothing at all (`mean_reward` 0.3931 -> 0.3949 over 1M
        # steps). That run passed silently under a single 10x gate while
        # being neither healthy nor diverged, which is the reading the gate
        # exists to prevent.
        if q_fp > 0.0 and peak_q > 2.0 * q_fp and peak_q <= 10.0 * q_fp:
            print("  ⚠ THE CRITIC OVERSHOT.", peak_q / q_fp, "x its fixed"
                  " point at the peak — not a divergence, but the runs that")
            print("     learned on this family peaked INSIDE 1.01x. Treat the")
            print("     rate as provisional and read the return against this")
            print("     run's own warmup baseline before believing it.")
        if q_fp > 0.0 and peak_q > 10.0 * q_fp:
            print("  ⚠⚠ THE CRITIC DIVERGED. `mean_q` reached", peak_q,
                  "against a fixed point of", q_fp, "— the policy was")
            print("     trained against that critic for however long it took")
            print("     to decay back, so the rate above is NOT a measurement")
            print("     of the task. Re-run; if it recurs, the configuration")
            print("     has no stability margin — `--updates-per-step 16`")
            print("     halves the target tracking rate and is the setting")
            print("     this file's own table records as converging.")

        # ⚠⚠ THE ANTI-VACUITY CHECK, AND IT IS NOT THE SUCCESS CRITERION.
        # Zero completed episodes reports `mean_return` as the fill value and
        # prints a plausible rate — which is what a run whose episodes never
        # terminate looks like, i.e. exactly the `TERMINATE_ON_UNHEALTHY`
        # failure this file's header is about. The horizon is 300 steps and
        # truncation ends every episode regardless, so a run of this length
        # with no episodes is a broken loop, not a hard task.
        if agent.ep_count() == 0:
            raise Error(
                "sac task reach: NOT ONE episode completed in "
                + String(num_steps) + " env-steps against a 300-step horizon."
                " `mean_return` is then the fill value and the rate above is"
                " meaningless. Check that the driver is stepping and that"
                " truncation is reaching the tracker."
            )
        print()
        # ⚠⚠ THE VERDICT IS AGAINST A BASELINE, NOT AGAINST ZERO. "The rate
        # moved off zero" was the criterion this run was built to answer and
        # it is the WRONG one for this task: uniform random already scores
        # 0.27 and the UNTRAINED greedy actor scores 1.00 (see the header).
        # Printing "moved off zero" here would have reported a trivial task as
        # a trained one.
        # ⚠⚠ THE VERDICT RUNS PER TASK. It read `baselines_for(task_name)`
        # with `task_name` being the whole comma list, so a two-task batch
        # found no entry and printed "NO BASELINE MEASURED" for a pair of
        # tasks that both have one. Worse, it judged ONE rate — the first
        # task's — and would have called a batch flat or not on the strength
        # of half of it.
        for bi in range(n_tasks):
            ref bname = task_names[bi]
            print("-" * 72)
            var bl = baselines_for(bname)
            # ⚠⚠ A CRITERION RANDOM ALREADY SATURATES CANNOT DISCRIMINATE.
            # `so101_settle_brick`'s goal HOLDS AT RESET, so its any-step rate
            # is 1.0 for any policy including no policy — its baseline says
            # exactly that (random 1.00). Judging that rate reports FLAT for a
            # task that is solved and FLAT for one that is broken, which is
            # the definition of a vacuous check. When the any-step baseline is
            # already 1, the held-at-END rate is the one carrying information.
            var rate = any_rates[bi]
            var which = String("met at ANY step")
            var p_base = bl[0]
            if bl[3] and bl[0] >= 1.0:
                rate = fin_rates[bi]
                p_base = bl[2]
                which = String("held at the END — the any-step rate is 1.0"
                               " at random, so it cannot discriminate")
            print("  judging:", which)
            if not bl[3]:
                print("  ⚠⚠ NO BASELINE MEASURED for", bname, "— run it with")
                print("  `--warmup >= --steps` first. A rate with nothing to")
                print("  compare it to is not a result.")
                continue
            print("  baselines for", bname, "— random", bl[0],
                  " untrained greedy", bl[1])

            # ⚠⚠ A RATE OVER n EPISODES HAS A STANDARD ERROR, AND AT THESE n IT IS
            # THE SAME SIZE AS THE EFFECT. The window is 100 episodes, so a
            # baseline of 0.02 carries se = sqrt(p(1-p)/n) = 0.014 — and a reading
            # of 0.05 is 2 se above it, which is suggestive and is NOT a result.
            # Printing the interval is what stops a noisy tick being read as a
            # curve; the first `gather` run oscillated 0.00 .. 0.05 for 125k steps
            # and every one of those values sits inside this band.
            # ⚠ THE DENOMINATOR IS THE GREEDY EVAL'S LANE COUNT, not the training
            # window — `rate` above comes from N_ENVS greedy episodes and the band
            # has to be the band for THAT n.
            # ⚠⚠ THE DENOMINATOR IS THIS TASK'S OWN EPISODE COUNT. In a batch a
            # task gets its share of the lanes, so a band computed from N_ENVS
            # would be the band for a sample four or eight times larger than the
            # one the rate came from — narrower, and wrong in the direction that
            # calls noise a result.
            var n = Float64(lanes_for_task(bi, N_ENVS, n_tasks) * EVAL_ROUNDS_N)
            var p = p_base
            var se = 0.0
            if n > 0.0:
                se = (p * (1.0 - p) / n) ** 0.5
            # ⚠⚠ A ZERO BASELINE GIVES A ZERO STANDARD ERROR, and then ANY nonzero
            # rate clears the band — one lucky lane out of 32 would read as
            # learning. `so101_lift_brick`'s measured random rate is 0.0 and this
            # printed a 2-sigma band ending at 0.0.
            #
            # The rule of three: having seen 0 successes in n trials, the 95%
            # upper bound on the true rate is about 3/n. That is the band a zero
            # baseline deserves — at 32 episodes it is 0.094, so three must
            # succeed before the number means anything; at 128 it is 0.023.
            var band = p + 2.0 * se
            if p <= 0.0:
                band = 3.0 / n
            if p <= 0.0:
                print("  baseline is 0 over", Int(n), "greedy episodes -> the",
                      "rule-of-three 95% upper bound is", band)
            else:
                print("  baseline se over", Int(n), "greedy episodes:", se,
                      " -> 2-sigma band ends at", band)

            if rate <= band:
                print("  FLAT — the rate is inside the random baseline's band.")
                print("  ⚠ READ THE SHAPED RETURN AND `mean_q` BEFORE CONCLUDING")
                print("  ANYTHING. The return is dense and moves long before the")
                print("  rate does, and this run's own warmup carries its")
                print("  baseline: while `episodes < 100` the printed avg_reward")
                print("  is `true_mean * episodes / 100`.")
                print("  If the return ROSE and then plateaued with a healthy")
                print("  critic, the shaping ran out of gradient — decompose it")
                print("  with `task_shaping_probe.mojo` and check the goal term's")
                print("  tolerance at the measured distance. `so101_lift_brick`")
                print("  plateaued at exactly that: margin 0.02 against a 0.030 m")
                print("  distance is a tolerance of 0.006 and a gradient of 1.9")
                print("  per metre, against 24 per metre at margin 0.05.")
            else:
                print("  the rate is ABOVE the baseline's 2-sigma band:", rate)
                # ⚠⚠ NO "FRACTION OF THE REFERENCE" LINE. This divided by 0.5625
                # and reported the run as a fraction of it — a number measured on
                # the 4 cm prop, at margin 0.10, on a single draw that never
                # reproduced at its own config (the repeat diverged to 273x). The
                # comment in `baselines_for` was corrected to say so and THIS
                # line, three hundred lines away, went on printing it: the same
                # constant written twice, drifting the moment one copy moved.
                #
                # A run that clears its baseline has said what it can say. The
                # comparison that means something is against ANOTHER RUN on the
                # SAME geometry, and that belongs in the log, not in a verdict
                # that implies a target nobody has reproduced.
        # ⚠ `status=done` IS WRITTEN HERE AND NOWHERE ELSE. A record still
        # saying `running` with an old `started` IS a crashed run, which is
        # information no directory listing has ever carried here — so nothing
        # may infer the status on this run's behalf.
        run.close()
        print("=" * 72)
