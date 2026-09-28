"""Collect the FB dataset from a SAC POLICY LADDER — the fix for §13's limit.

`docs/BFM_ZERO_SHOT_RL.md` §13 measured why M2 failed, and it was not the FB
code: a random-policy walker dataset contains no trajectory that ever stands or
walks, so `F`'s argmax over it is arbitrary. §6 component 1 ranks dumping the
states visited across SAC training above every other collection lever.

`examples/dm_control/sac_dm_walker_training_gpu.mojo` writes that ladder — one
step-stamped checkpoint per 25 k env-steps, random through expert. This script
rolls out EVERY rung and writes ONE store spanning the whole gradient of
behaviour.

## What is stored, and the two consumers

    qpos, qvel        generalized coordinates — NOT the observation
    action            the action that produced the row
    policy_step       the ladder rung the row came from
    ep_return         that episode's undiscounted return

`qpos`/`qvel` rather than observations because dm_control rewards read `Data`
(FK products, `xmat`, `subtree_linvel`), so only generalized coordinates let
`Phyics3dEnv.reward_at` relabel a row under a reward invented afterwards. That
argument is the header of `deep_agents/fb/collect.mojo` and it is why the SAC
replay buffer itself — which holds observations — is not what gets exported.

`policy_step` and `ep_return` exist because §15.5 found that the same rows
serve two consumers that want DIFFERENT subsets:

  * **FB** reads everything. The falls from rung 1 are the coverage; without
    them `B` sees only the gait manifold and the successor measure is
    degenerate off it.
  * **FB-CPR's discriminator** must read only a high-quality tail. HIL's
    §6.1 is explicit — *"it is essential to use high-quality motions as the
    positive samples for the adversarial motion priors"* — and their ablation
    shows a mis-specified discriminator is WORSE than none at all (skill
    accuracy 0.38 vs 0.53).

⚠ Neither tag is recoverable after collection. A store written without them
forces a full re-collect the day FB-CPR needs positives, which is exactly the
cost §15.5 said to pay now.

## The rollout policy is STOCHASTIC, deliberately

`select_action`, not `select_greedy_action`. SAC's entropy term is tuned
against `target_entropy`, so the stochastic actor explores a calibrated
neighbourhood of the learned gait — that spread is dataset coverage. A greedy
rollout of 24 rungs would give 24 near-deterministic trajectories and a dataset
with no local structure for `B` to embed.

Run (after training the ladder for the same TASK):
    pixi run mojo run -I . examples/fb/collect_walker_sac.mojo
    pixi run mojo run -I . examples/fb/collect_walker_sac.mojo --run <run_id>
"""

from std.random import seed

from noeira.nn.constants import DT
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStoreWriter
from noeira.deep_agents.sac import SAC
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig

from max.gpu.host import DeviceContext
from std.sys import argv
from std.os.path import exists
from noeira.core.project import projects_root
from noeira.io.proc import quote_arg, run_capture


# ── must match the training script for this TASK ─────────────────────────
comptime TASK: StaticString = "walk"  # "stand" | "walk" | "run"
comptime MOVE_SPEED: Float64 = 0.0 if TASK == "stand" else (
    1.0 if TASK == "walk" else 8.0
)
# ══ LADDER GEOMETRY — MUST MATCH `sac_dm_walker_training_gpu.mojo` ═══════
# Rung filenames are reconstructed as `(k+1) * SEGMENT_STEPS`. If the trainer
# used different values every rung reports MISSING below — each with the path
# it tried, so compare that against `ls sac_dm_walker_*.ckpt.*`.
comptime SEGMENT_STEPS = 32_000  # trainer's EPISODE_LEN * N_ENVS
comptime N_SEGMENTS = 20
comptime HIDDEN = 256
comptime CKPT_PREFIX = "sac_dm_walker_"

# ── dataset geometry ─────────────────────────────────────────────────────
# 20 rungs x 50 episodes x 1000 steps = 1 000 000 rows (~110 MB at 24 fp32 +
# 2 tag columns). §13's guard refuses a run above 5000 epochs over the
# dataset, which at BATCH=1024 x 2 M steps needs >= ~400 k rows — so this is
# comfortably above the floor that made the first M2 launch memorise.
comptime EPISODES_PER_RUNG = 50
comptime EP_LEN = 1000  # dm_control's own episode length (MAX_STEPS)
comptime SEED = 20260810

comptime OUT_PATH = "fb_walker_" + TASK + "_sac.h5"

# SAC preset dims — only the architecture has to match the checkpoint; BATCH
# and CAP size the (unused) replay buffer, so keep them small here.
comptime BATCH = 256
comptime CAP = 1000

comptime WalkerCfg = DMWalkerConfig[MOVE_SPEED]
comptime EnvT = Phyics3dEnv[DMWalkerModel, WalkerCfg, DType.float64, False]

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


def main() raises:
    # ⚠ In a function body — a top-level `comptime assert` does not parse.
    # Without it a typo'd TASK falls through the ternary above to 8.0 (run)
    # and silently collects the wrong task.
    comptime assert (
        TASK == "stand" or TASK == "walk" or TASK == "run"
    ), "TASK must be 'stand', 'walk' or 'run'"

    seed(SEED)
    print("=" * 70)
    print("FB dataset from the SAC ladder — dm_control walker", TASK)
    print("=" * 70)
    print("  rungs              =", N_SEGMENTS)
    print("  episodes / rung    =", EPISODES_PER_RUNG)
    print("  episode length     =", EP_LEN)
    print("  target rows        =", N_SEGMENTS * EPISODES_PER_RUNG * EP_LEN)
    print("  NQ / NV / NACT     =", NQ, "/", NV, "/", NACT)
    print("  out                =", OUT_PATH)
    print("=" * 70)

    # Rungs live in the CWD (old layout) or, with `--run`, in that training
    # run's `checkpoints/` (`runs/<id>/checkpoints/sac_dm_walker_<task>.ckpt.*`).
    var run_arg = _flag(String("--run"), String(""))
    var prefix = String(CKPT_PREFIX) + String(TASK)
    if run_arg.byte_length() > 0:
        prefix = _run_dir(run_arg) + "/checkpoints/" + prefix
    print("  ladder prefix      =", prefix)

    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, NQ))
    cols.append(ColumnSpec(String("qvel"), DType.float32, NV))
    cols.append(ColumnSpec(String("action"), DType.float32, NACT))
    cols.append(ColumnSpec(String("policy_step"), DType.int32, 1))
    cols.append(ColumnSpec(String("ep_return"), DType.float32, 1))

    var w = TrajectoryStoreWriter(
        String(OUT_PATH),
        cols^,
        env_id=String("dm_control/walker-") + String(TASK) + "-sac-ladder",
        seed=SEED,
        chunk_rows=4096,
    )

    # `ctx` is the fields facade's host staging for the model bridge — the
    # agent below is a CPU agent; nothing here runs on device.
    var ctx = DeviceContext()
    var env = EnvT(ctx)

    # Per-episode buffers, flushed once per episode so `ep_return` can be
    # written on EVERY row of the episode that produced it (the return is only
    # known at the end, and a column cannot be patched after the fact).
    var qbuf = List[Float32](length=EP_LEN * NQ, fill=Float32(0))
    var vbuf = List[Float32](length=EP_LEN * NV, fill=Float32(0))
    var abuf = List[Float32](length=EP_LEN * NACT, fill=Float32(0))
    var sbuf = List[Int32](length=EP_LEN, fill=Int32(0))
    var rbuf = List[Float32](length=EP_LEN, fill=Float32(0))

    var act_out = List[Scalar[DT]](length=NACT, fill=Scalar[DT](0))
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))

    var total_rows = 0
    var total_eps = 0
    var missing = 0

    for seg in range(N_SEGMENTS):
        var at = (seg + 1) * SEGMENT_STEPS
        var path = _stamped(prefix, at)

        # `learning_starts=0` so `select_action` always takes the POLICY path
        # rather than the uniform-random warmup path — otherwise every rung
        # would collect random actions and the ladder would be worthless.
        var agent = SAC["cpu", OBS_DIM, NACT, BATCH, CAP, HIDDEN](
            action_scale=1.0,
            learning_starts=0,
        )
        try:
            agent.load(path)
        except e:
            print("  [rung", seg + 1, "] MISSING", path, "—", e)
            missing += 1
            continue

        var rung_ret = Float64(0)
        for _ep in range(EPISODES_PER_RUNG):
            var o64 = env.reset_obs_list()
            for i in range(OBS_DIM):
                obs[i] = Scalar[DT](Float64(o64[i]))

            var ep_ret = Float64(0)
            var n = 0
            for t in range(EP_LEN):
                # step_idx is past learning_starts=0, so this is the policy.
                agent.select_action(obs, act_out, t + 1)

                var alist = List[Scalar[DT]](capacity=NACT)
                for k in range(NACT):
                    alist.append(act_out[k])
                    abuf[t * NACT + k] = Float32(Float64(act_out[k]))

                var res = env.step_continuous_vec[DT](alist)
                for i in range(OBS_DIM):
                    obs[i] = res[0][i]
                ep_ret += Float64(res[1])

                # ⚠ State AFTER the step, paired with the action that produced
                # it — the convention `tests/dm_control/test_reward_relabel.mojo`
                # gates. Storing the PRE-step state would leave every relabelled
                # reward off by one control step.
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
            w.end_episode()

            total_rows += n
            total_eps += 1
            rung_ret += ep_ret

        print(
            "  [rung", seg + 1, "/", N_SEGMENTS, "]  step", at,
            "  mean_ret", rung_ret / Float64(EPISODES_PER_RUNG),
            "  rows", total_rows,
        )

    w.close()

    print("")
    print("=" * 70)
    print("Collection complete —", OUT_PATH)
    print("  rows                =", total_rows)
    print("  episodes            =", total_eps)
    print("  rungs missing       =", missing)
    print("=" * 70)

    if missing == N_SEGMENTS:
        print("⚠⚠ NO rung loaded. Train the ladder first, for THIS task:")
        print(
            "   pixi run -e nvidia mojo run -I ."
            " examples/dm_control/sac_dm_walker_training_gpu.mojo"
        )
        print("   (and check TASK matches in both scripts)")
    elif missing > 0:
        print(
            "⚠ ", missing, "rungs were missing. The dataset is USABLE but its"
            " behaviour gradient has holes — check whether the missing rungs"
            " are the EARLY ones, because those carry the coverage."
        )
    else:
        print("Next — the numbers to check BEFORE training FB on this:")
        print("  * mean_ret must RISE across rungs. A flat ladder means SAC")
        print("    never learned, and the dataset is random data with extra")
        print("    steps — the exact M2 failure.")
        print("  * rung 1 should be near-random. If it is already competent,")
        print("    WARMUP_STEPS was too small and the coverage is missing.")
        print("")
        print("Then point `fb_train_gpu.mojo` at", OUT_PATH, "and re-run the")
        print("walker eval against THIS store — §13's last eval computed z")
        print("from a 10 k store while the checkpoint had trained on 1 M.")
