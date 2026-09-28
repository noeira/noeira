"""ONE balanced FB dataset from ALL THREE walker ladders — stand, walk, run.

Supersedes `collect_walker_sac.mojo` (single task, uniform rung weighting).
Written after reading the first real walk ladder, which came out like this:

    rung  1        289          50 k rows
    rungs 2-3      657 -> 687  100 k
    rung  4        950          50 k
    rungs 5-20     946 - 970   800 k     <-- 80% of the dataset

The whole behaviour gradient lives in rungs 1-4; rungs 5-20 are one policy
sampled sixteen times. A dataset that is 80% one gait teaches `B` to resolve
fine structure inside a single behaviour, which is not what the successor
measure needs. §13's failure was the same quantity — coverage — from the other
end.

Three changes, all cheap:

1. **All three tasks in one store.** The dynamics are IDENTICAL across
   stand/walk/run — `MOVE_SPEED` only reaches `compute_reward_*`, never the
   physics — so one store of generalized coordinates is legitimately scorable
   under all three rewards via `reward_at`. Three ladders give three genuinely
   different behaviour modes over one body, which is the entire premise of
   zero-shot inference. This is the biggest single coverage win available and
   it costs nothing but rollout time.

2. **A uniform-random rung 0.** ⚠ It is NOT recoverable from the ladder: the
   first checkpoint lands at 32 k env-steps, by which point walk already
   returns 289 against a random baseline near 25. The 25 -> 289 -> 657
   transition — the falls, tumbles and recoveries — was never written to disk.
   Rolling uniform actions recovers the BOTTOM of that range. It does not
   recover the middle, and no script can; re-running training with a
   subdivided first segment is the only way to get that back.

   ⚠ Rung 0 is uniform random ACTIONS, not a freshly-initialised actor. An
   untrained tanh MLP has small weights, so it emits near-zero actions — a
   degenerate constant policy, not exploration. That distinction is easy to
   get backwards and would quietly produce a rung with no coverage at all.

3. **Front-weighted episode counts.** Episodes are spent where behaviour
   CHANGES, not where it has converged. Per task:

       rung 0        (random)      EP_RANDOM  episodes
       rungs 1-4     (transition)  EP_EARLY   each
       rungs 5-20    (converged)   EP_LATE    each

   The converged rungs are deliberately kept, not dropped — they are the
   high-quality tail FB-CPR's discriminator needs (§15.5), and HIL's ablation
   is blunt about what a discriminator trained on poor positives does. They
   just do not need 800 k rows to say what they say.

## Columns

    qpos, qvel, action    generalized coordinates + the action that produced them
    policy_step           ladder rung (0 = uniform random)
    ep_return             that episode's return UNDER ITS OWN TASK's reward
    task                  0 = stand, 1 = walk, 2 = run

`ep_return` is scored under the generating task because its job is a QUALITY
tag: a good stand policy measured under the walk reward reads as a bad policy,
and the FB-CPR positive-selection filter would then discard exactly the rows it
wants. `task` records which POLICY produced the row — not which reward applies,
since after `reward_at` any of the three can.

⚠ Requires all three ladders. Train them first, editing TASK in
`examples/dm_control/sac_dm_walker_training_gpu.mojo` between runs.

Run:
    pixi run mojo run -I . examples/fb/collect_walker_all.mojo
    pixi run mojo run -I . examples/fb/collect_walker_all.mojo --run-stand <id> --run-walk <id> --run-run <id>
"""

from std.random import random_float64, seed

from noeira.nn.constants import DT
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStoreWriter
from noeira.deep_agents.sac import SAC
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.phyics3d_env_config import Phyics3dEnvConfig
from noeira.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig

from max.gpu.host import DeviceContext
from std.sys import argv
from std.os.path import exists
from noeira.core.project import projects_root
from noeira.io.proc import quote_arg, run_capture


# ══ LADDER GEOMETRY — MUST MATCH `sac_dm_walker_training_gpu.mojo` ═══════
comptime SEGMENT_STEPS = 32_000  # trainer's EPISODE_LEN * N_ENVS
comptime N_SEGMENTS = 20
comptime HIDDEN = 256
comptime CKPT_PREFIX = "sac_dm_walker_"

# ── dataset composition ──────────────────────────────────────────────────
# Per task: 40 + 4x60 + 16x14 = 504 episodes = 504 000 rows.
# Three tasks: 1 512 000 rows (~145 MB at 24 fp32 + 3 tag columns).
#
# Resulting balance, against the walk-only store's 5% / 15% / 80%:
#     random        8%
#     transition   48%
#     converged    44%
#
# At BATCH=1024 x 2 M steps that is ~1355 epochs over the dataset —
# comfortably under `fb_train_gpu.mojo`'s 5000-epoch refusal.
comptime EARLY_RUNGS = 4  # rungs 1..4 carry the behaviour gradient
comptime EP_RANDOM = 40
comptime EP_EARLY = 60
comptime EP_LATE = 14
comptime EP_LEN = 1000  # dm_control's own MAX_STEPS
comptime SEED = 20260810

comptime OUT_PATH = "fb_walker_all_sac.h5"

# Only the ARCHITECTURE has to match the checkpoint; BATCH/CAP size the
# unused replay buffer, so keep them small.
comptime BATCH = 256
comptime CAP = 1000

comptime NQ = DMWalkerModel.NQ
comptime NV = DMWalkerModel.NV
comptime NACT = DMWalkerModel.ACTION_DIM
comptime OBS_DIM = DMWalkerModel.OBS_DIM


def _stamped(prefix: String, step: Int) raises -> String:
    """Mirror of the training script's `_stamped` — the two must agree."""
    var s = String(step)
    var pad = String("")
    for _ in range(8 - s.byte_length()):
        pad += "0"
    return prefix + ".ckpt." + pad + s


def _episodes_for(rung: Int) -> Int:
    if rung == 0:
        return EP_RANDOM
    if rung <= EARLY_RUNGS:
        return EP_EARLY
    return EP_LATE


struct TaskStats(Movable & Deinitable):
    var rows: Int
    var episodes: Int
    var missing: Int

    def __init__(out self, rows: Int, episodes: Int, missing: Int):
        self.rows = rows
        self.episodes = episodes
        self.missing = missing

    def __init__(out self, *, deinit move: Self):
        self.rows = move.rows
        self.episodes = move.episodes
        self.missing = move.missing


def collect_task[
    CONFIG: Phyics3dEnvConfig
](
    mut w: TrajectoryStoreWriter,
    ctx: DeviceContext,
    task_name: String,
    task_id: Int,
    run_dir: String,
) raises -> TaskStats:
    """Roll out one task's whole ladder, rung 0 (random) through N_SEGMENTS.

    `run_dir` is the training run holding that ladder (its `checkpoints/`), or
    "" for the old layout — rungs in the CWD."""
    comptime EnvT = Phyics3dEnv[DMWalkerModel, CONFIG, DType.float64, False]

    var prefix = String(CKPT_PREFIX) + task_name
    if run_dir.byte_length() > 0:
        prefix = run_dir + "/checkpoints/" + prefix
    var env = EnvT(ctx)

    # Flushed once per episode so `ep_return` — known only at the end — can be
    # written on EVERY row of the episode that produced it.
    var qbuf = List[Float32](length=EP_LEN * NQ, fill=Float32(0))
    var vbuf = List[Float32](length=EP_LEN * NV, fill=Float32(0))
    var abuf = List[Float32](length=EP_LEN * NACT, fill=Float32(0))
    var sbuf = List[Int32](length=EP_LEN, fill=Int32(0))
    var rbuf = List[Float32](length=EP_LEN, fill=Float32(0))
    var tbuf = List[Int32](length=EP_LEN, fill=Int32(task_id))

    var act_out = List[Scalar[DT]](length=NACT, fill=Scalar[DT](0))
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))

    var rows = 0
    var eps = 0
    var missing = 0

    for rung in range(N_SEGMENTS + 1):  # 0 = uniform random
        var at = rung * SEGMENT_STEPS
        var n_ep = _episodes_for(rung)

        # `learning_starts=0` so `select_action` always takes the POLICY path
        # rather than the uniform-random warmup path — otherwise every rung
        # would collect random actions and the ladder would be worthless.
        var agent = SAC["cpu", OBS_DIM, NACT, BATCH, CAP, HIDDEN](
            action_scale=1.0,
            learning_starts=0,
        )
        var have_policy = False
        if rung > 0:
            var path = _stamped(prefix, at)
            try:
                agent.load(path)
                have_policy = True
            except e:
                print("    [rung", rung, "] MISSING", path, "—", e)
                missing += 1
                continue

        var rung_ret = Float64(0)
        for _ep in range(n_ep):
            var o64 = env.reset_obs_list()
            for i in range(OBS_DIM):
                obs[i] = Scalar[DT](Float64(o64[i]))

            var ep_ret = Float64(0)
            var n = 0
            for t in range(EP_LEN):
                var alist = List[Scalar[DT]](capacity=NACT)
                if have_policy:
                    agent.select_action(obs, act_out, t + 1)
                    for k in range(NACT):
                        alist.append(act_out[k])
                else:
                    # ⚠ UNIFORM actions, not an untrained actor — see the
                    # module docstring. An untrained tanh MLP emits ~0.
                    for k in range(NACT):
                        alist.append(
                            Scalar[DT](random_float64() * 2.0 - 1.0)
                        )
                for k in range(NACT):
                    abuf[t * NACT + k] = Float32(Float64(alist[k]))

                var res = env.step_continuous_vec[DT](alist)
                for i in range(OBS_DIM):
                    obs[i] = res[0][i]
                ep_ret += Float64(res[1])

                # ⚠ State AFTER the step, paired with the action that produced
                # it — the convention `tests/dm_control/test_reward_relabel.mojo`
                # gates. The PRE-step state would leave every relabelled reward
                # off by one control step.
                for i in range(NQ):
                    qbuf[t * NQ + i] = Float32(Float64(env.d.qpos.data[i]))
                for i in range(NV):
                    vbuf[t * NV + i] = Float32(Float64(env.d.qvel.data[i]))
                sbuf[t] = Int32(at)
                n += 1
                if res[2]:
                    break

            for t in range(n):
                rbuf[t] = Float32(ep_ret)

            w.append[DType.float32](
                String("qpos"), qbuf.unsafe_ptr().as_unsafe_any_origin(), n
            )
            w.append[DType.float32](
                String("qvel"), vbuf.unsafe_ptr().as_unsafe_any_origin(), n
            )
            w.append[DType.float32](
                String("action"), abuf.unsafe_ptr().as_unsafe_any_origin(), n
            )
            w.append[DType.int32](
                String("policy_step"),
                sbuf.unsafe_ptr().as_unsafe_any_origin(),
                n,
            )
            w.append[DType.float32](
                String("ep_return"), rbuf.unsafe_ptr().as_unsafe_any_origin(), n
            )
            w.append[DType.int32](
                String("task"), tbuf.unsafe_ptr().as_unsafe_any_origin(), n
            )
            w.end_episode()

            rows += n
            eps += 1
            rung_ret += ep_ret

        var label = String("random") if rung == 0 else String("step ") + String(at)
        print(
            "    [rung", rung, "/", N_SEGMENTS, "]", label,
            " eps", n_ep, " mean_ret", rung_ret / Float64(n_ep),
            " rows", rows,
        )

    return TaskStats(rows, eps, missing)


def _run_dir(handle: String) raises -> String:
    """`handle` as a run directory: a path to one, or a run id under `runs/` or
    any `projects/*/runs/`. Raises when none of them holds a `checkpoints/`."""
    if exists(handle + "/checkpoints"):
        return handle
    if exists(String("runs/") + handle + "/checkpoints"):
        return String("runs/") + handle
    var root = projects_root()
    var listing = run_capture(
        String("ls -d ") + quote_arg(root) + "/*/runs/" + quote_arg(handle)
        + " 2>/dev/null; true",
        1 << 16,
    )
    for line in listing.split("\n"):
        var d = String(String(line).strip())
        if d.byte_length() > 0 and exists(d + "/checkpoints"):
            return d
    raise Error(
        "--run '" + handle + "' is neither a run directory nor a run id under"
        " runs/ or " + root + "/*/runs/"
    )


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _task_run_dir(flag: String) raises -> String:
    var h = _flag(flag, String(""))
    return _run_dir(h) if h.byte_length() > 0 else String("")


def main() raises:
    seed(SEED)
    var per_task = (
        EP_RANDOM + EARLY_RUNGS * EP_EARLY
        + (N_SEGMENTS - EARLY_RUNGS) * EP_LATE
    )
    print("=" * 70)
    print("FB dataset — ALL THREE walker ladders (stand / walk / run)")
    print("=" * 70)
    print("  rungs / task       =", N_SEGMENTS, "+ 1 random")
    print("  episodes / task    =", per_task)
    print("    rung 0  (random) =", EP_RANDOM)
    print("    rungs 1 -", EARLY_RUNGS, "        =", EP_EARLY, "each")
    print("    rungs", EARLY_RUNGS + 1, "-", N_SEGMENTS, "      =", EP_LATE, "each")
    print("  episode length     =", EP_LEN)
    print("  target rows        =", 3 * per_task * EP_LEN)
    print("  NQ / NV / NACT     =", NQ, "/", NV, "/", NACT)
    print("  out                =", OUT_PATH)
    print("=" * 70)

    # One training run per task (`--run-stand/--run-walk/--run-run`); a task
    # without one reads its rungs from the CWD, the old layout.
    var dir_stand = _task_run_dir(String("--run-stand"))
    var dir_walk = _task_run_dir(String("--run-walk"))
    var dir_run = _task_run_dir(String("--run-run"))

    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, NQ))
    cols.append(ColumnSpec(String("qvel"), DType.float32, NV))
    cols.append(ColumnSpec(String("action"), DType.float32, NACT))
    cols.append(ColumnSpec(String("policy_step"), DType.int32, 1))
    cols.append(ColumnSpec(String("ep_return"), DType.float32, 1))
    cols.append(ColumnSpec(String("task"), DType.int32, 1))

    var w = TrajectoryStoreWriter(
        String(OUT_PATH),
        cols^,
        env_id=String("dm_control/walker-all-sac-ladder"),
        seed=SEED,
        chunk_rows=4096,
    )

    # Host staging for the fields model bridge; the agents below are CPU.
    var ctx = DeviceContext()

    print("[stand] ------------------------------------------------------")
    var s_stand = collect_task[DMWalkerConfig[0.0]](
        w, ctx, String("stand"), 0, dir_stand
    )
    print("[walk]  ------------------------------------------------------")
    var s_walk = collect_task[DMWalkerConfig[1.0]](w, ctx, String("walk"), 1, dir_walk)
    print("[run]   ------------------------------------------------------")
    var s_run = collect_task[DMWalkerConfig[8.0]](w, ctx, String("run"), 2, dir_run)

    w.close()

    var rows = s_stand.rows + s_walk.rows + s_run.rows
    var eps = s_stand.episodes + s_walk.episodes + s_run.episodes
    var missing = s_stand.missing + s_walk.missing + s_run.missing

    print("")
    print("=" * 70)
    print("Collection complete —", OUT_PATH)
    print("  rows                =", rows)
    print("    stand / walk / run=", s_stand.rows, "/", s_walk.rows, "/", s_run.rows)
    print("  episodes            =", eps)
    print("  rungs missing       =", missing)
    print("=" * 70)

    if missing >= 3 * N_SEGMENTS:
        print("⚠⚠ NO rung loaded for ANY task. Train the ladders first —")
        print("   edit TASK in sac_dm_walker_training_gpu.mojo between runs:")
        print(
            "   pixi run -e nvidia mojo run -I ."
            " examples/dm_control/sac_dm_walker_training_gpu.mojo"
        )
    elif missing > 0:
        print(
            "⚠ ", missing, "rungs missing. The store is USABLE but has holes."
            " Check WHICH — a missing EARLY rung costs coverage, a missing"
            " late one costs almost nothing."
        )
    else:
        print("What to check before training FB on this:")
        print("  * rung 0's mean_ret should be near the random baseline")
        print("    (walker ~25). If it is not, the uniform branch did not run.")
        print("  * each task's rungs 1-4 should SPAN — that span is the")
        print("    behaviour gradient the whole dataset exists to provide.")
        print("  * stand / walk / run converged returns should DIFFER in")
        print("    character, not just value. Three ladders that all learned")
        print("    the same gait give one behaviour mode, not three, and")
        print("    zero-shot transfer has nothing to interpolate between.")
        print("")
        print("Then point `fb_train_gpu.mojo`'s STORE_PATH at", OUT_PATH)
