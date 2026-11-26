"""RL beside a FROZEN chunked policy, under an inference delay, sparse reward.

    # the base alone, at a delay (no RL): what the delay costs
    build/tower_async_rl_chunk --base CKPT --arm base --delay 9 --episodes 200

    # one training run of an arm
    build/tower_async_rl_chunk --base CKPT --arm C --delay 9 --seed 1 \\
        --train-episodes 600 --out RESULT.csv
    build/tower_async_rl_step  --base CKPT --arm A --delay 9 --seed 1 ...

Build once per arm KIND (the SAC shapes are compile-time):

    pixi run mojo build -I . examples/so101/tower_async_rl.mojo \\
        -o build/tower_async_rl_chunk
    pixi run mojo build -I . -D ARM_STEP=1 examples/so101/tower_async_rl.mojo \\
        -o build/tower_async_rl_step

## What it asks

Three 2026 papers train a small RL policy beside a frozen VLA, and differ in
where it acts (docs/VLA_ONLINE_RL_REREAD_2026_11_25.md). None says what an
inference delay of the Orin's size does to each shape. Here the frozen
policy is `TowerChunkNet` (state -> 32 actions, `tower_chunk_bc_fit.mojo`)
on `so101_tower_lift_brick`, its answers are delayed by `--delay` control
steps (`deep_agents/bc/async_chunk.mojo`), and the reward is 1 at success
(the goal held 31 steps) and 0 otherwise.

## The arms

Every arm adds a BOUNDED residual (`--cap`, in normalised action units) to
the base's actions. The scene (the 49-word state, which holds the brick's
pose) is seen only at a query, as an image would be; the arm's own joints
(12 words) are free to read at any step.

| arm | acts | state it reads |
|---|---|---|
| A | each step, 6 words | the executing chunk's query state, fresh joints, the base action |
| B | each chunk, when the query STARTS | the query state, the reference actions |
| C | as B | B + the actions still committed while the query is in flight |
| D | each chunk, when it LANDS | the query state, fresh joints, the reference actions |

A is the per-command residual (ForceRFT, PLD). B is a chunk actor fed stale
state only. C adds ARLI's first state addition. D is what a chunk actor that
runs after the VLA can read (RL Token's shape under latency, and the
analogue of ARLI's second addition).

⚠ THE CHUNK RESIDUAL IS `KNOTS` KNOTS, NOT ONE VALUE PER ACTION: piecewise
linear over the executed part. RL Token's actor outputs every action of the
chunk (140 words); ARLI's outputs one vector repeated over the chunk. Two
knots of six words sit between the two and keep SAC's action at 12 words.

## The learner

`SAC` on the CPU, as the family driver configures it, with the discount of
one decision (`0.99` per control step, so `0.99^16` per chunk decision) and
the same number of gradient updates per control step in every arm. A
success is a terminal; the 300-step limit is a truncation.

## What a run writes

`--out F`: one line per evaluation (greedy, on `--eval-episodes` fixed
placements) and a `final` line on `--final-episodes` other placements. A run
with a `final` line is complete; a queue skips it.
"""

from std.math import sqrt
from std.random import seed as seed_rng, random_float64
from std.sys import argv
from std.sys.defines import is_defined
from std.pathlib import Path
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.io.fileio import write_text_atomic
from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.checkpoint import load_params
from noeira.nn.core.initializer import Kaiming
from noeira.core.cont_action import ContAction
from noeira.deep_agents.bc.async_chunk import AsyncChunkExec, add_knot_residual
from noeira.deep_agents.bc.policy import BcNorm, load_bc_norm
from noeira.deep_agents.demos.file import DemoSet, read_demo_file
from noeira.deep_agents.sac import SAC, SACPresetAgent
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig, So101TowerTeleopConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.host_reward import family_reward_host
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.so101_tower_chunk import (
    TOWER_ACT, TOWER_CHUNK_H, TOWER_CHUNK_EXEC, TOWER_CHUNK_OUT, TowerChunkNet,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.utils.fmt import fixed

comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime TASK = "so101_tower_lift_brick"
comptime CFG = So101TowerTeleopConfig
comptime E = Phyics3dEnv[So101TowerModel, CFG, DType.float64, False]
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime OBS_DIM = So101TowerModel.OBS_DIM
comptime ACT = TOWER_ACT
comptime H = TOWER_CHUNK_H
comptime N_EXEC = TOWER_CHUNK_EXEC

comptime STEP_ARM = is_defined["ARM_STEP"]()
"""This build's arm kind: per-step (arm A) or per-chunk (arms B, C, D)."""
comptime KNOTS: Int = 2
comptime DMAX: Int = H - N_EXEC
"""The largest delay, and the width of the committed-actions block."""
comptime PROPRIO: Int = 2 * ACT
"""The arm's own joint angles and velocities."""

# ── the RL observation, per arm kind ───────────────────────────────────────
#   step : [ scene OBS_DIM | joints 12 | base action 6 | fraction 1 ]
#   chunk: [ scene OBS_DIM | joints 12 | reference N_EXEC*6 | committed DMAX*6 ]
# A block an arm does not read is zero for the whole run, so B, C and D share
# one network shape and differ only in what they are shown.
comptime O_SCENE: Int = 0
comptime O_JOINTS: Int = OBS_DIM
comptime O_REF: Int = OBS_DIM + PROPRIO
comptime O_COM: Int = O_REF + N_EXEC * ACT
comptime OBS_RL: Int = (
    OBS_DIM + PROPRIO + ACT + 1 if STEP_ARM
    else OBS_DIM + PROPRIO + N_EXEC * ACT + DMAX * ACT
)
comptime ACT_RL: Int = ACT if STEP_ARM else KNOTS * ACT
comptime BATCH: Int = 256
comptime CAP: Int = 300_000 if STEP_ARM else 30_000
comptime HIDDEN: Int = 256
comptime LEARNING_STARTS: Int = 4096 if STEP_ARM else 256
"""About the same number of control steps in both kinds (256 chunks of 16)."""
comptime UPDATES_PER_DECISION: Int = 1 if STEP_ARM else N_EXEC
comptime GAMMA_STEP: Float64 = 0.99

comptime Agent = SACPresetAgent["cpu", OBS_RL, ACT_RL, BATCH, CAP, HIDDEN]

comptime GB = So101TowerConfig.OBS_GOAL_BASE
comptime HOLD_STEPS: Int = 31
"""The goal held this many consecutive steps is a success — the recorders'
rule (`tower_expert_record.mojo`)."""
comptime SETTLE_STEPS: Int = 5

comptime ARM_BASE: Int = -1
comptime ARM_A: Int = 0
comptime ARM_B: Int = 1
comptime ARM_C: Int = 2
comptime ARM_D: Int = 3

comptime MODE_NONE: Int = 0
comptime MODE_TRAIN: Int = 1
comptime MODE_GREEDY: Int = 2
comptime MODE_RANDOM: Int = 3

# Placement seeds. The expert files hold seeds below 20000; these three
# ranges are disjoint from them and from each other.
comptime TRAIN_SEED0: Int = 1_000_000
comptime EVAL_SEED0: Int = 200_000
comptime FINAL_SEED0: Int = 300_000


struct Rig(Movable):
    """What an episode needs besides the env: the actuator ranges and the
    task's `meta` words."""

    var lo: List[Float64]
    var hi: List[Float64]
    var meta_idx: List[Int]
    var meta_val: List[Float64]
    var frame_skip: Int
    var timestep: Float64

    def __init__(out self):
        self.lo = List[Float64]()
        self.hi = List[Float64]()
        self.meta_idx = List[Int]()
        self.meta_val = List[Float64]()
        self.frame_skip = 1
        self.timestep = 0.0


struct Tally(Movable):
    var env_steps: Int
    var decisions: Int
    var updates: Int
    var transitions: Int
    var terminals: Int
    # the last episode, for `--verbose`
    var ep_max_held: Int
    var ep_reach_min_mm: Float64
    var ep_z0_max: Float64
    var ep_z1_max: Float64
    var ep_queries: Int
    var ep_z_max: Float64
    """The largest |z-score| of an observation word at a query: how far the
    base was asked to answer from its training distribution."""
    var ep_z_word: Int
    var ep_z_first: Float64
    var ep_z_first_word: Int
    var debug: Bool

    def __init__(out self):
        self.env_steps = 0
        self.decisions = 0
        self.updates = 0
        self.transitions = 0
        self.terminals = 0
        self.ep_max_held = 0
        self.ep_reach_min_mm = 0.0
        self.ep_z0_max = 0.0
        self.ep_z1_max = 0.0
        self.ep_queries = 0
        self.ep_z_max = 0.0
        self.ep_z_word = -1
        self.ep_z_first = 0.0
        self.ep_z_first_word = -1
        self.debug = False


def _usage():
    print("usage: tower_async_rl --base CKPT --arm base|A|B|C|D --delay D"
          " [--episodes N] [--seed0 S] [--random-cap X] [--exec N]"
          " [--verbose 1]")
    print("       tower_async_rl --base CKPT --replay-demo F [--episodes N]")
    print("       tower_async_rl --base CKPT --arm A|B|C|D --delay D --seed S"
          " --out F [--train-episodes N] [--eval-every N] [--eval-episodes N]"
          " [--final-episodes N] [--cap X] [--max-steps N]"
          " [--zero-head 1] [--save-actor F]")


def base_chunk(
    mut net: TowerChunkNet, ref norm: BcNorm, mut bx: Tensor, mut by: Tensor,
    ref obs: List[Float64], mut chunk: List[Float64],
) raises:
    """The frozen policy: one state in, `H` actions out, clamped to [-1, 1]
    (the fit has no output squash)."""
    for j in range(OBS_DIM):
        bx.data[j] = Scalar[DT]((obs[j] - norm.mu[j]) / norm.sd[j])
    net.forward["cpu", 1](TensorRefs[1](bx), by, None)
    for k in range(TOWER_CHUNK_OUT):
        var a = Float64(by.data[k])
        if a > 1.0:
            a = 1.0
        if a < -1.0:
            a = -1.0
        chunk[k] = a


def zero_mean_head(mut agent: Agent):
    """Zero the actor's mean head (weight and bias), so that the greedy
    residual of an untrained agent is zero and a run starts from the frozen
    base exactly. The log-std head keeps its initial values."""
    for k in range(len(agent.trainer.actor.heads.branches[0].weight.val.data)):
        agent.trainer.actor.heads.branches[0].weight.val.data[k] = 0
    for k in range(len(agent.trainer.actor.heads.branches[0].bias.val.data)):
        agent.trainer.actor.heads.branches[0].bias.val.data[k] = 0
    agent.trainer.actor.heads.branches[0].weight.val.version += 1
    agent.trainer.actor.heads.branches[0].bias.val.version += 1


def decide(
    mut agent: Agent, ref o: List[Scalar[DT]], mut a: List[Scalar[DT]],
    mode: Int, idx: Int,
) raises:
    """The residual's raw action in [-1, 1]: sampled (uniform during the
    warm-up) when training, the mean when evaluating, uniform in
    `MODE_RANDOM`."""
    if mode == MODE_TRAIN:
        agent.select_action(o, a, idx)
    elif mode == MODE_GREEDY:
        agent.select_greedy_action(o, a)
    else:
        for j in range(ACT_RL):
            a[j] = Scalar[DT](2.0 * random_float64() - 1.0)


def train_updates(mut agent: Agent, mut tally: Tally, n: Int) raises:
    for _ in range(n):
        if agent.trainer.train_step(tally.decisions):
            tally.updates += 1


def run_episode(
    mut env: E, mut net: TowerChunkNet, ref norm: BcNorm, mut bx: Tensor,
    mut by: Tensor, mut agent: Agent, ref rig: Rig, arm: Int, mode: Int,
    delay: Int, cap: Float64, seed: Int, max_steps: Int, mut tally: Tally,
    n_exec: Int = N_EXEC,
) raises -> Tuple[Bool, Int]:
    """One episode from the placement `seed`. Returns (success, steps).

    `n_exec` other than `N_EXEC` is for the base alone: the RL observation's
    layout is built on `N_EXEC`."""
    if n_exec != N_EXEC and mode != MODE_NONE:
        raise Error("run_episode: an RL arm executes N_EXEC actions per chunk")
    var use_rl = mode != MODE_NONE and arm != ARM_BASE
    var training = mode == MODE_TRAIN
    var ex = AsyncChunkExec(H, ACT, n_exec, delay)

    # ── reset: the task's meta words, the placement, the recorder's settle ─
    _ = env.reset()
    for k in range(len(rig.meta_idx)):
        env.d.meta.data[rig.meta_idx[k]] = Scalar[DType.float64](
            rig.meta_val[k]
        )
    var q0 = posed_qpos[So101TowerPlacement](
        String(TASK), String(FAMILY), So101TowerConfig.SLOT_RADIUS,
        UInt64(seed),
    )
    var v0 = List[Float64](length=NV, fill=0.0)
    var s0 = env.obs_at(q0, v0)
    var obs = List[Float64](length=OBS_DIM, fill=0.0)
    for k in range(OBS_DIM):
        obs[k] = Float64(s0.data[k])
    var act_l = List[Float64](length=ACT, fill=0.0)
    for _ in range(SETTLE_STEPS):
        var hold = ContAction[ACT]()
        for k in range(ACT):
            var span = rig.hi[k] - rig.lo[k]
            var q = Float64(env.d.qpos.data[k])
            hold.data[k] = (
                2.0 * (q - rig.lo[k]) / span - 1.0 if span != 0.0 else 0.0
            )
        var o = env.step(hold)
        for k in range(OBS_DIM):
            obs[k] = Float64(o[0].data[k])

    var chunk = List[Float64](length=TOWER_CHUNK_OUT, fill=0.0)
    var com = List[Float64](length=DMAX * ACT, fill=0.0)
    var knots = List[Float64](length=KNOTS * ACT, fill=0.0)
    var pend_scene = List[Float64](length=OBS_DIM, fill=0.0)
    var cur_scene = List[Float64](length=OBS_DIM, fill=0.0)
    var rl_obs = List[Scalar[DT]](length=OBS_RL, fill=Scalar[DT](0))
    var rl_act = List[Scalar[DT]](length=ACT_RL, fill=Scalar[DT](0))
    # the chunk arms' decisions: the one whose chunk is executing, and the
    # one made for the chunk that has not landed yet
    var exec_obs = List[Scalar[DT]](length=OBS_RL, fill=Scalar[DT](0))
    var exec_act = List[Scalar[DT]](length=ACT_RL, fill=Scalar[DT](0))
    var have_exec = False
    var pend_obs = List[Scalar[DT]](length=OBS_RL, fill=Scalar[DT](0))
    var pend_act = List[Scalar[DT]](length=ACT_RL, fill=Scalar[DT](0))
    var have_pend = False
    # the step arm's previous decision
    var prev_obs = List[Scalar[DT]](length=OBS_RL, fill=Scalar[DT](0))
    var prev_act = List[Scalar[DT]](length=ACT_RL, fill=Scalar[DT](0))
    var have_prev = False
    var zero = Scalar[DT](0)
    var one = Scalar[DT](1)
    var held = 0
    var adr0 = So101TowerPlacement.free_qadr(0)
    var adr1 = So101TowerPlacement.free_qadr(1)
    tally.ep_max_held = 0
    tally.ep_reach_min_mm = 1.0e9
    tally.ep_z0_max = -1.0
    tally.ep_z1_max = -1.0
    tally.ep_queries = 0
    tally.ep_z_max = 0.0
    tally.ep_z_word = -1
    tally.ep_z_first = 0.0
    tally.ep_z_first_word = -1

    for t in range(max_steps):
        # ── 1. a query starts from this step's observation ────────────────
        if ex.needs_query():
            base_chunk(net, norm, bx, by, obs, chunk)
            for k in range(OBS_DIM):
                var z = abs((obs[k] - norm.mu[k]) / norm.sd[k])
                if z > tally.ep_z_max:
                    tally.ep_z_max = z
                    tally.ep_z_word = k
                if t == 0 and z > tally.ep_z_first:
                    tally.ep_z_first = z
                    tally.ep_z_first_word = k
            if tally.debug and (t == 0 or t == N_EXEC or t == 4 * N_EXEC):
                var line = String("      query at step ") + String(t) + " obs:"
                for k in range(OBS_DIM):
                    line += " " + fixed(obs[k], 3)
                print(line)
                var l2 = String("      chunk actions 0 / 15 / 31:")
                for idx in [0, 15, 31]:
                    l2 += " |"
                    for j in range(ACT):
                        l2 += " " + fixed(chunk[idx * ACT + j], 3)
                print(l2)
            var start = ex.next_start()
            for k in range(OBS_DIM):
                pend_scene[k] = obs[k]
            if use_rl and (arm == ARM_B or arm == ARM_C):
                for k in range(OBS_RL):
                    rl_obs[k] = zero
                for k in range(OBS_DIM):
                    rl_obs[O_SCENE + k] = Scalar[DT](obs[k])
                for k in range(N_EXEC * ACT):
                    rl_obs[O_REF + k] = Scalar[DT](chunk[start * ACT + k])
                if arm == ARM_C:
                    ex.committed(com, DMAX)
                    for k in range(DMAX * ACT):
                        rl_obs[O_COM + k] = Scalar[DT](com[k])
                decide(agent, rl_obs, rl_act, mode, tally.decisions)
                if training:
                    tally.decisions += 1
                for k in range(KNOTS * ACT):
                    knots[k] = Float64(rl_act[k])
                add_knot_residual(chunk, start, N_EXEC, ACT, knots, KNOTS, cap)
                for k in range(OBS_RL):
                    pend_obs[k] = rl_obs[k]
                for k in range(ACT_RL):
                    pend_act[k] = rl_act[k]
                have_pend = True
            ex.submit(chunk)

        # ── 2. a chunk lands ──────────────────────────────────────────────
        if ex.lands_now():
            if use_rl and arm == ARM_D:
                for k in range(OBS_RL):
                    rl_obs[k] = zero
                for k in range(OBS_DIM):
                    rl_obs[O_SCENE + k] = Scalar[DT](pend_scene[k])
                for k in range(ACT):
                    rl_obs[O_JOINTS + k] = Scalar[DT](obs[k])
                    rl_obs[O_JOINTS + ACT + k] = Scalar[DT](obs[NQ + k])
                var ps = ex.pend_start
                for k in range(N_EXEC * ACT):
                    rl_obs[O_REF + k] = Scalar[DT](ex.pend[ps * ACT + k])
                decide(agent, rl_obs, rl_act, mode, tally.decisions)
                if training:
                    tally.decisions += 1
                for k in range(KNOTS * ACT):
                    knots[k] = Float64(rl_act[k])
                add_knot_residual(ex.pend, ps, N_EXEC, ACT, knots, KNOTS, cap)
                for k in range(OBS_RL):
                    pend_obs[k] = rl_obs[k]
                for k in range(ACT_RL):
                    pend_act[k] = rl_act[k]
                have_pend = True
            if have_pend:
                # the executing chunk ran out without a terminal: its
                # decision's transition ends at the next decision
                if training and have_exec:
                    agent.trainer.record(exec_obs, exec_act, zero, pend_obs, zero)
                    tally.transitions += 1
                    train_updates(agent, tally, UPDATES_PER_DECISION)
                for k in range(OBS_RL):
                    exec_obs[k] = pend_obs[k]
                for k in range(ACT_RL):
                    exec_act[k] = pend_act[k]
                have_exec = True
                have_pend = False
            ex.land()
            for k in range(OBS_DIM):
                cur_scene[k] = pend_scene[k]

        # ── 3. this step's action ─────────────────────────────────────────
        if ex.has_action():
            for j in range(ACT):
                act_l[j] = ex.peek(j)
            if use_rl and arm == ARM_A:
                for k in range(OBS_DIM):
                    rl_obs[O_SCENE + k] = Scalar[DT](cur_scene[k])
                for k in range(ACT):
                    rl_obs[O_JOINTS + k] = Scalar[DT](obs[k])
                    rl_obs[O_JOINTS + ACT + k] = Scalar[DT](obs[NQ + k])
                    rl_obs[OBS_DIM + PROPRIO + k] = Scalar[DT](act_l[k])
                rl_obs[OBS_DIM + PROPRIO + ACT] = Scalar[DT](ex.exec_fraction())
                if training and have_prev:
                    agent.trainer.record(prev_obs, prev_act, zero, rl_obs, zero)
                    tally.transitions += 1
                    train_updates(agent, tally, UPDATES_PER_DECISION)
                decide(agent, rl_obs, rl_act, mode, tally.decisions)
                if training:
                    tally.decisions += 1
                for j in range(ACT):
                    var a = act_l[j] + cap * Float64(rl_act[j])
                    if a > 1.0:
                        a = 1.0
                    if a < -1.0:
                        a = -1.0
                    act_l[j] = a
                for k in range(OBS_RL):
                    prev_obs[k] = rl_obs[k]
                for k in range(ACT_RL):
                    prev_act[k] = rl_act[k]
                have_prev = True
        else:
            # before the first chunk lands: hold the pose
            for k in range(ACT):
                var span = rig.hi[k] - rig.lo[k]
                var q = Float64(env.d.qpos.data[k])
                act_l[k] = (
                    2.0 * (q - rig.lo[k]) / span - 1.0 if span != 0.0 else 0.0
                )

        # ── 4. the environment ────────────────────────────────────────────
        var action = ContAction[ACT]()
        for k in range(ACT):
            action.data[k] = act_l[k]
        var o = env.step(action)
        for k in range(OBS_DIM):
            obs[k] = Float64(o[0].data[k])
        var rd = family_reward_host[CFG, DType.float64, E.MD, ACT](
            env.d, env.mf, act_l, t + 1, rig.frame_skip, rig.timestep,
        )
        if rd[1]:
            held += 1
        else:
            held = 0
        ex.advance()
        if held > tally.ep_max_held:
            tally.ep_max_held = held
        var reach = sqrt(
            obs[GB + 3] * obs[GB + 3] + obs[GB + 4] * obs[GB + 4]
            + obs[GB + 5] * obs[GB + 5]
        ) * 1000.0
        if reach < tally.ep_reach_min_mm:
            tally.ep_reach_min_mm = reach
        if obs[adr0 + 2] > tally.ep_z0_max:
            tally.ep_z0_max = obs[adr0 + 2]
        if obs[adr1 + 2] > tally.ep_z1_max:
            tally.ep_z1_max = obs[adr1 + 2]
        tally.ep_queries = ex.n_queries
        if training:
            tally.env_steps += 1

        # ── 5. success is the only reward, and a terminal ─────────────────
        # (`rl_obs` stands in as the next observation: a terminal's is never
        # read, and `record` refuses the same list twice.)
        if held >= HOLD_STEPS:
            if training:
                if arm == ARM_A and have_prev:
                    agent.trainer.record(prev_obs, prev_act, one, rl_obs, one)
                    tally.transitions += 1
                    tally.terminals += 1
                    train_updates(agent, tally, UPDATES_PER_DECISION)
                elif arm != ARM_A and have_exec:
                    agent.trainer.record(exec_obs, exec_act, one, rl_obs, one)
                    tally.transitions += 1
                    tally.terminals += 1
                    train_updates(agent, tally, UPDATES_PER_DECISION)
            return (True, t + 1)
    return (False, max_steps)


def evaluate(
    mut env: E, mut net: TowerChunkNet, ref norm: BcNorm, mut bx: Tensor,
    mut by: Tensor, mut agent: Agent, ref rig: Rig, arm: Int, mode: Int,
    delay: Int, cap: Float64, seed0: Int, n: Int, max_steps: Int,
    mut tally: Tally, n_exec: Int = N_EXEC, verbose: Bool = False,
) raises -> Tuple[Int, Float64]:
    """`n` episodes from `seed0`: (successes, mean steps of the successes)."""
    var ok = 0
    var steps_ok = 0
    for ep in range(n):
        var r = run_episode(
            env, net, norm, bx, by, agent, rig, arm, mode, delay, cap,
            seed0 + ep, max_steps, tally, n_exec,
        )
        if r[0]:
            ok += 1
            steps_ok += r[1]
        if verbose:
            print("    ep", ep, "seed", seed0 + ep, "ok" if r[0] else "--",
                  "steps", r[1], "| held max", tally.ep_max_held,
                  "| reach min", fixed(tally.ep_reach_min_mm, 1), "mm",
                  "| free-slot z max", fixed(tally.ep_z0_max, 3),
                  fixed(tally.ep_z1_max, 3), "| queries", tally.ep_queries,
                  "| |z| first query", fixed(tally.ep_z_first, 1), "word",
                  tally.ep_z_first_word, "max", fixed(tally.ep_z_max, 1),
                  "word", tally.ep_z_word)
    var mean_steps = Float64(steps_ok) / Float64(ok) if ok > 0 else 0.0
    return (ok, mean_steps)


def replay_demo(
    mut env: E, mut net: TowerChunkNet, ref norm: BcNorm, mut bx: Tensor,
    mut by: Tensor, ref rig: Rig, ref ds: DemoSet, n: Int,
) raises:
    """Two checks on the recording the base was fitted to, per episode:

    1. OPEN-LOOP REPLAY: the env set to the first recorded state, then the
       recorded actions. Does this loop's success rule fire, and how far does
       the state drift from the recorded one? A recording that does not
       replay says the env, the reset or the rule differ from the recorder's.
    2. TEACHER FORCING: the base's chunk at each recorded state against the
       actions recorded from there on, as a mean absolute error at chunk
       indices 0, 8, 16 and 31."""
    var o32 = List[Float32](length=OBS_DIM, fill=0.0)
    var n32 = List[Float32](length=OBS_DIM, fill=0.0)
    var a32 = List[Float32](length=ACT, fill=0.0)
    var obs = List[Float64](length=OBS_DIM, fill=0.0)
    var chunk = List[Float64](length=TOWER_CHUNK_OUT, fill=0.0)
    var act_l = List[Float64](length=ACT, fill=0.0)
    var probes: List[Int] = [0, 8, 16, 31]
    var tf_err = List[Float64](length=4, fill=0.0)
    var tf_n = 0
    var n_ok = 0
    var done_eps = 0
    for e in range(ds.n_episodes()):
        if done_eps >= n:
            break
        if not ds.ep_success[e]:
            continue
        done_eps += 1
        var start = ds.ep_start[e]
        var ln = ds.ep_len[e]
        _ = env.reset()
        for k in range(len(rig.meta_idx)):
            env.d.meta.data[rig.meta_idx[k]] = Scalar[DType.float64](
                rig.meta_val[k]
            )
        ds.row_obs[DType.float32](start, o32)
        var q0 = List[Float64](length=NQ, fill=0.0)
        var v0 = List[Float64](length=NV, fill=0.0)
        for k in range(NQ):
            q0[k] = Float64(o32[k])
        for k in range(NV):
            v0[k] = Float64(o32[NQ + k])
        _ = env.obs_at(q0, v0)
        var held = 0
        var max_held = 0
        var drift_max = 0.0
        var drift_end = 0.0
        for k in range(ln + HOLD_STEPS + 5):
            var r = start + (k if k < ln else ln - 1)
            ds.row_act[DType.float32](r, a32)
            if k < ln:
                ds.row_obs[DType.float32](r, o32)
                for j in range(OBS_DIM):
                    obs[j] = Float64(o32[j])
                base_chunk(net, norm, bx, by, obs, chunk)
                for pi in range(4):
                    var idx = probes[pi]
                    var src = k + idx
                    if src > ln - 1:
                        src = ln - 1
                    ds.row_act[DType.float32](start + src, a32)
                    var e1 = 0.0
                    for j in range(ACT):
                        e1 += abs(chunk[idx * ACT + j] - Float64(a32[j]))
                    tf_err[pi] += e1 / Float64(ACT)
                tf_n += 1
                ds.row_act[DType.float32](r, a32)
            var action = ContAction[ACT]()
            for j in range(ACT):
                action.data[j] = Float64(a32[j])
                act_l[j] = Float64(a32[j])
            var o = env.step(action)
            var rd = family_reward_host[CFG, DType.float64, E.MD, ACT](
                env.d, env.mf, act_l, k + 1, rig.frame_skip, rig.timestep,
            )
            if rd[1]:
                held += 1
            else:
                held = 0
            if held > max_held:
                max_held = held
            if k < ln:
                ds.row_next_obs[DType.float32](r, n32)
                var dmax = 0.0
                for j in range(NQ):
                    var dv = abs(Float64(o[0].data[j]) - Float64(n32[j]))
                    if dv > dmax:
                        dmax = dv
                if dmax > drift_max:
                    drift_max = dmax
                drift_end = dmax
        if max_held >= HOLD_STEPS:
            n_ok += 1
        print("    demo ep", e, "rows", ln, "| held max", max_held,
              "| qpos drift max", fixed(drift_max, 4), "end",
              fixed(drift_end, 4))
    print("  REPLAY open loop:", n_ok, "of", done_eps,
          "episodes reach the success rule")
    if tf_n > 0:
        print("  TEACHER FORCING, mean |error| per action word at chunk index"
              " 0 / 8 / 16 / 31:", fixed(tf_err[0] / Float64(tf_n), 4),
              fixed(tf_err[1] / Float64(tf_n), 4),
              fixed(tf_err[2] / Float64(tf_n), 4),
              fixed(tf_err[3] / Float64(tf_n), 4), "over", tf_n, "rows")


def main() raises:
    var args = argv()
    var base = String("")
    var arm_s = String("base")
    var out = String("")
    var delay = 0
    var seed = 1
    var episodes = 200
    var seed0 = EVAL_SEED0
    var random_cap = 0.0
    var train_episodes = 600
    var eval_every = 100
    var eval_episodes = 50
    var final_episodes = 200
    var cap = 0.02
    var max_steps = 300
    var n_exec = N_EXEC
    var verbose = False
    var debug = False
    var replay = String("")
    var zero_head = True
    var save_actor = String("")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--help" or a == "-h":
            _usage()
            return
        if i + 1 >= len(args):
            _usage()
            raise Error("missing value for " + a)
        var v = String(args[i + 1])
        if a == "--base":
            base = v
        elif a == "--arm":
            arm_s = v
        elif a == "--out":
            out = v
        elif a == "--delay":
            delay = Int(v)
        elif a == "--seed":
            seed = Int(v)
        elif a == "--episodes":
            episodes = Int(v)
        elif a == "--seed0":
            seed0 = Int(v)
        elif a == "--random-cap":
            random_cap = Float64(v)
        elif a == "--train-episodes":
            train_episodes = Int(v)
        elif a == "--eval-every":
            eval_every = Int(v)
        elif a == "--eval-episodes":
            eval_episodes = Int(v)
        elif a == "--final-episodes":
            final_episodes = Int(v)
        elif a == "--cap":
            cap = Float64(v)
        elif a == "--max-steps":
            max_steps = Int(v)
        elif a == "--exec":
            n_exec = Int(v)
        elif a == "--verbose":
            verbose = Int(v) != 0
            debug = Int(v) >= 2
        elif a == "--replay-demo":
            replay = v
        elif a == "--zero-head":
            zero_head = Int(v) != 0
        elif a == "--save-actor":
            save_actor = v
        else:
            _usage()
            raise Error("unrecognised argument: " + a)
        i += 2
    if base.byte_length() == 0 or not Path(base).exists():
        _usage()
        raise Error("--base: no such checkpoint: " + base)
    if delay < 0 or delay > DMAX:
        raise Error("--delay must be in [0, " + String(DMAX) + "]")
    var arm = ARM_BASE
    if arm_s == "A":
        arm = ARM_A
    elif arm_s == "B":
        arm = ARM_B
    elif arm_s == "C":
        arm = ARM_C
    elif arm_s == "D":
        arm = ARM_D
    elif arm_s != "base":
        _usage()
        raise Error("--arm: base, A, B, C or D")
    if STEP_ARM and (arm == ARM_B or arm == ARM_C or arm == ARM_D):
        raise Error("this build is the per-step kind (-D ARM_STEP=1): arm A"
                    " or base only")
    if (not STEP_ARM) and arm == ARM_A:
        raise Error("arm A needs the per-step build (-D ARM_STEP=1)")

    # ── the frozen base ───────────────────────────────────────────────────
    var net = TowerChunkNet.make["cpu", Kaiming](None)
    _ = load_params["cpu"](net, base, None)
    var norm = load_bc_norm(
        base + ".norm", OBS_DIM, TOWER_CHUNK_OUT,
        "Re-run tower_chunk_bc_fit.mojo.",
    )
    var bx = Tensor.alloc(OBS_DIM)
    var by = Tensor.alloc(TOWER_CHUNK_OUT)

    # ── the env and the task's words (the probe's own preamble) ───────────
    var ctx = DeviceContext()
    var env = E(ctx)
    var rig = Rig()
    var sf = So101TowerModel.make_spec_fields[DType.float64]()
    var lo = actuator_column(sf, ACT_IDX_CTRL_MIN, ACT)
    var hi = actuator_column(sf, ACT_IDX_CTRL_MAX, ACT)
    for k in range(ACT):
        rig.lo.append(Float64(lo[k]))
        rig.hi.append(Float64(hi[k]))
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DType.float64](cw[k])
    var mw = task_meta_words(
        String(TASK), String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    for k in range(len(mw[0])):
        rig.meta_idx.append(mw[0][k])
        rig.meta_val.append(mw[1][k])
    rig.frame_skip = CFG.FRAME_SKIP
    rig.timestep = So101TowerModel.TIMESTEP

    seed_rng(seed)
    var gamma = GAMMA_STEP
    if not STEP_ARM:
        gamma = GAMMA_STEP ** Float64(N_EXEC)
    var agent: Agent = SAC["cpu", OBS_RL, ACT_RL, BATCH, CAP, HIDDEN](
        gamma=Scalar[DT](gamma),
        action_scale=1.0,
        init_alpha=0.01,
        target_entropy=Scalar[DT](-Float64(ACT_RL)),
        learning_starts=LEARNING_STARTS,
    )
    if zero_head:
        zero_mean_head(agent)
    var tally = Tally()
    tally.debug = debug

    print("=" * 70)
    print("so101_tower — RL beside a frozen chunk policy | arm", arm_s,
          "| delay", delay, "steps | base", base)
    print("  chunk", H, "x", ACT, ", executes", N_EXEC, "| build:",
          "per-step" if STEP_ARM else "per-chunk", "| RL obs", OBS_RL,
          "act", ACT_RL)
    print("=" * 70)

    if replay.byte_length() > 0:
        var ds = read_demo_file(replay)
        replay_demo(env, net, norm, bx, by, rig, ds, episodes)
        return

    # ── the base alone (or with a uniform random residual) ────────────────
    if arm == ARM_BASE or random_cap > 0.0:
        var mode = MODE_NONE
        var the_arm = arm
        var the_cap = 0.0
        if random_cap > 0.0:
            if arm == ARM_BASE:
                raise Error("--random-cap needs an arm (A, B, C or D)")
            mode = MODE_RANDOM
            the_cap = random_cap
        var t0 = perf_counter_ns()
        var r = evaluate(
            env, net, norm, bx, by, agent, rig, the_arm, mode, delay,
            the_cap, seed0, episodes, max_steps, tally, n_exec, verbose,
        )
        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        print("  RESULT arm", arm_s, "delay", delay, "exec", n_exec, "random_cap",
              fixed(random_cap, 3), "success", r[0], "of", episodes,
              "mean_steps_ok", fixed(r[1], 1), "|", fixed(secs, 1), "s")
        return

    # ── one training run ──────────────────────────────────────────────────
    if out.byte_length() == 0:
        raise Error("--out is required for a training run")
    var text = String(
        "# arm,delay,seed,cap,train_episodes,env_steps,decisions,updates,"
        "transitions,terminals,train_success_window,eval_success,eval_n,"
        "eval_mean_steps_ok,kind\n"
    )
    var window_ok = 0
    var window_n = 0
    var t0 = perf_counter_ns()
    # with `--zero-head 0` the untrained policy is NOT the base: its mean
    # residual is a random function of the state, and the first evaluation
    # says what that costs. With the zeroed head it is the base exactly.
    for ep in range(train_episodes + 1):
        if ep % eval_every == 0 or ep == train_episodes:
            var is_final = ep == train_episodes
            var n = final_episodes if is_final else eval_episodes
            var s0 = FINAL_SEED0 if is_final else EVAL_SEED0
            if is_final and save_actor.byte_length() > 0:
                agent.save(save_actor)
            var r = evaluate(
                env, net, norm, bx, by, agent, rig, arm, MODE_GREEDY, delay,
                cap, s0, n, max_steps, tally,
            )
            var win = (
                Float64(window_ok) / Float64(window_n) if window_n > 0 else 0.0
            )
            var kind = String("final") if is_final else String("eval")
            text += (
                arm_s + "," + String(delay) + "," + String(seed) + ","
                + fixed(cap, 3) + "," + String(ep) + ","
                + String(tally.env_steps) + "," + String(tally.decisions) + ","
                + String(tally.updates) + "," + String(tally.transitions) + ","
                + String(tally.terminals) + "," + fixed(win, 3) + ","
                + String(r[0]) + "," + String(n) + "," + fixed(r[1], 1) + ","
                + kind + "\n"
            )
            write_text_atomic(out, text)
            var secs = Float64(perf_counter_ns() - t0) / 1.0e9
            print("  ", kind, "after", ep, "episodes | env steps",
                  tally.env_steps, "| updates", tally.updates,
                  "| train window", fixed(win, 3), "| greedy", r[0], "of", n,
                  "| mean steps", fixed(r[1], 1), "|", fixed(secs, 0), "s")
            window_ok = 0
            window_n = 0
        if ep == train_episodes:
            break
        var tr = run_episode(
            env, net, norm, bx, by, agent, rig, arm, MODE_TRAIN, delay, cap,
            TRAIN_SEED0 + seed * 100_000 + ep, max_steps, tally,
        )
        window_n += 1
        if tr[0]:
            window_ok += 1
    print("  done:", out)
