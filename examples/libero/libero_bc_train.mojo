"""BEHAVIOUR CLONING ON LIBERO'S DEMONSTRATIONS — the port's first policy.

    pixi run libero-bc-train
    pixi run libero-bc-train --epochs 40 --lr 3e-4     # the batch is comptime: BATCH
    pixi run libero-bc-train --demos 10 --epochs 5        # a smoke run

Reads `build/demos/<family>.lowdim.h5` (`data/libero_demos.mojo`'s store: one
episode per demonstration, `state` in OUR joint order, `action` the recorded
seven OSC_POSE words), rebuilds each frame's OBSERVATION exactly as the batched
env writes it, fits an MLP, and writes a checkpoint
`examples/libero/libero_eval_batched.mojo` can load into `_policy_action`:

    pixi run mojo run -I . examples/libero/libero_eval_batched.mojo --policy <run_id>

The checkpoint lives in the run's own directory (`RunContext`, project
`libero`): `runs/<id>/checkpoints/last.ckpt` with its `.norm` sidecar beside it
(`last.ckpt.norm`) — the network at the END of the fit, the only one
`fit_bc` writes. `--out PATH` still overrides the location. The run's
`run.kv`, `metrics.csv` and monitor row carry the same id.

## ⚠⚠ THE OBSERVATION IS REBUILT, NOT STORED — AND THAT IS THE POINT

The store holds `state` (qpos ++ qvel), not the policy input. The input is what
`task_hooks.write_task_obs` writes on the device — qpos, qvel, one active word
per free slot, then the nine goal words measured from `robot_grip_site` — and
`write_task_obs_host` is the SAME implementation
(`tests/libero/test_libero_task_hooks.mojo` gates the two word for word on every
LIBERO task). So each row here is: load `state` into a CPU `Data`, run FK, put
the row's task tape and mask into `meta`, and call the host writer. A policy
trained on anything else would be trained on a vector the env never produces.

⚠ FK PER ROW IS THE COST. 63 728 rows of `libero_goal` take about a minute.

## ⚠ THE SPLIT IS BY DEMONSTRATION, NOT BY ROW

Consecutive frames of one demo are nearly identical; a row-wise split puts a
frame's neighbours on both sides and reports a validation error that means
nothing. `--val-demos K` holds out the LAST K demos of every task.

## ⚠⚠ THE BASELINE IS PRINTED BESIDE THE RESULT

A validation MSE on its own is unreadable. Two references are printed with it:
the error of predicting ZERO (LIBERO's null action — what the eval harness
scores 0 with) and of predicting the TRAINING MEAN action. A policy that does
not beat both has learned nothing, and the run says so rather than reporting a
number that looks like progress.

## ⚠ WHAT THIS IS NOT

It is not a LIBERO result. A success rate comes from
`libero_eval_batched.mojo` on the frozen inits, on the box; this file only
fits actions to observations. Open-loop imitation of 50 demos per task is a
BASELINE — the thing a real policy (ACT, a diffusion head, an image policy)
has to beat, and the thing that says whether the low-dimensional observation
carries enough to imitate at all.
"""

from std.os import listdir
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns


from noeira.core.run import RunContext, register_run
from noeira.core.run_session import finish_run, run_logger
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import sink_for_run
from noeira.data.store import TrajectoryStore
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.gpu.constants import (
    META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
)
from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family,
)
from noeira.tasks.family import scene_path
from noeira.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses,
)
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.tasks.active import active_mask
from noeira.deep_agents.bc.policy import BcNet, BC_HID
from noeira.deep_agents.bc.dataset import BcDataset
from noeira.deep_agents.bc.fit import fit_bc
from noeira.tasks.task_hooks import write_task_obs_host
from noeira.envs.libero.placement.libero_goal import LiberoGoalPlacement
from noeira.envs.libero.models.libero_goal_xml import (
    LIBERO_GOAL_OBS_DIM, LIBERO_GOAL_MAX_CONTACTS,
)


comptime H = DType.float64
comptime FAMILY = "libero_goal"
comptime FAMILY_DIR = "noeira/envs/libero/families/"
comptime TASK_DIR = "noeira/envs/libero/tasks/"
comptime STORE = "build/demos/libero_goal.lowdim.h5"
comptime OBS = LIBERO_GOAL_OBS_DIM
comptime ACT = 7
comptime HID = BC_HID
comptime BATCH = 256
"""⚠ COMPTIME: `forward`/`vjp` take the batch as a parameter, so one build has
one batch size. `--batch` would be a second kernel instantiation, not a flag."""

comptime NET = BcNet[OBS, ACT]
"""⚠ THE SHAPE LIVES IN `deep_agents/bc/policy.mojo`, so the driver that RUNS this
checkpoint builds the same network from the same declaration rather than a
second spelling of it."""


def _task_names(family: String) raises -> List[String]:
    var out = List[String]()
    var want = family + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def main() raises:
    var args = argv()
    var epochs = 20
    var lr = 1.0e-3
    var val_demos = 5
    var max_demos = 0
    var out_path = String("")
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--epochs" and i + 1 < len(args):
            epochs = Int(String(args[i + 1]))
            i += 1
        elif s == "--lr" and i + 1 < len(args):
            lr = Float64(String(args[i + 1]))
            i += 1
        elif s == "--val-demos" and i + 1 < len(args):
            val_demos = Int(String(args[i + 1]))
            i += 1
        elif s == "--demos" and i + 1 < len(args):
            max_demos = Int(String(args[i + 1]))
            i += 1
        elif s == "--out" and i + 1 < len(args):
            out_path = String(args[i + 1])
            i += 1
        else:
            raise Error(
                "libero bc train: unknown argument '" + s + "' (--epochs N,"
                " --lr X, --val-demos K, --demos N, --out PATH)"
            )
        i += 1

    print("=" * 78)
    print("Behaviour cloning on LIBERO's demonstrations —", FAMILY)
    print("=" * 78)
    if not exists(String(STORE)):
        # ⚠ SKIPPED, LOUDLY, AND NOT A PASS — the store is gitignored.
        print("  SKIPPED: no demo store at", STORE)
        print("  Build it:  pixi run libero-demo-import --no-images")
        print("=== SKIPPED (no demonstrations — this is not a pass) ===")
        return

    # ── the scene, for the observation ────────────────────────────────────
    var f = load_family(String(FAMILY_DIR) + String(FAMILY) + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
    )
    var m = Model[H, DynDims](dims)
    while True:
        try:
            build_model_runtime[H](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(
                fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[H, DynDims](dims)
    var d = Data[H, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()

    # ── the tasks: one tape and mask per task index ───────────────────────
    var names = _task_names(String(FAMILY))
    var nqs = List[Int]()
    for k in range(len(fmd.joints)):
        nqs.append(fmd.joints[k].nq)
    var jadr = joint_qpos_addresses(nqs)
    var tapes = List[List[Float64]]()
    var masks = List[Float64]()
    for ti in range(len(names)):
        var t = load_task(String(TASK_DIR) + names[ti] + ".task")
        validate_task_against_family(t, f)
        var g = bind_goal(
            parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
            fmd.joint_names, jadr,
        )
        require_tier_a(g, t.name)
        tapes.append(encode_goal(g))
        masks.append(active_mask(t, f))

    # ── the store ─────────────────────────────────────────────────────────
    var store = TrajectoryStore(String(STORE))
    var n_rows_all = store.n_rows()
    var n_eps = store.n_episodes()
    print("  store :", STORE, "|", n_rows_all, "rows |", n_eps, "episodes")
    var state_col = store.load_column[H](String("state"))
    var act_col = store.load_column[DType.float32](String("action"))
    var task_col = store.load_column[DType.int32](String("task_index"))
    var state_dim = nq + nv
    if len(state_col) != n_rows_all * state_dim:
        raise Error(
            "libero bc train: state column is " + String(len(state_col))
            + " words, expected " + String(n_rows_all * state_dim)
            + " — the store was written for a different scene"
        )

    # ⚠ THE SPLIT IS BY DEMONSTRATION. Episodes run task by task in the
    # importer's order, so "the last `val_demos` of each task" is the last
    # `val_demos` episodes of each task's run of episodes.
    var ep_task = List[Int](length=n_eps, fill=0)
    var per_task_eps = List[List[Int]]()
    for _ in range(len(names)):
        per_task_eps.append(List[Int]())
    for e in range(n_eps):
        var off = Int(store.episodes.ep_offset[e])
        var ti = Int(task_col[off])
        ep_task[e] = ti
        per_task_eps[ti].append(e)
    var is_val = List[Bool](length=n_eps, fill=False)
    var use_ep = List[Bool](length=n_eps, fill=True)
    for ti in range(len(names)):
        ref eps = per_task_eps[ti]
        # ⚠ THE CAP COMES FIRST AND THE SPLIT IS TAKEN INSIDE IT. `--demos 4
        # --val-demos 2` must hold out 2 OF THOSE 4, not demos 48-49 that the
        # cap already dropped — which is how the first smoke run got an empty
        # validation set and a fit with nothing to score.
        var n_use = len(eps)
        if max_demos > 0 and max_demos < n_use:
            n_use = max_demos
        if val_demos >= n_use:
            raise Error(
                "libero bc train: --val-demos " + String(val_demos) + " of "
                + String(n_use) + " usable demos leaves no training data"
            )
        for k in range(len(eps)):
            if k >= n_use:
                use_ep[eps[k]] = False
            elif k >= n_use - val_demos:
                is_val[eps[k]] = True

    # ── the observations, rebuilt frame by frame ──────────────────────────
    var t0 = perf_counter_ns()
    var data = BcDataset(OBS, ACT)
    var arow = List[Scalar[DType.float32]](length=ACT, fill=0)
    for e in range(n_eps):
        if not use_ep[e]:
            continue
        var off = Int(store.episodes.ep_offset[e])
        var ln = Int(store.episodes.ep_len[e])
        var ti = ep_task[e]
        for k in range(TAPE_WORDS):
            d.meta.data[META_IDX_TASK_PARAM_0 + k] = Scalar[H](tapes[ti][k])
        d.meta.data[META_IDX_TASK_ACTIVE] = Scalar[H](masks[ti])
        for r in range(off, off + ln):
            for j in range(nq):
                d.qpos.data[j] = state_col[r * state_dim + j]
            for j in range(nv):
                d.qvel.data[j] = state_col[r * state_dim + nq + j]
            forward_kinematics["cpu", H, DynDims, 1](d, m)
            var row = List[Scalar[H]]()
            write_task_obs_host[LiberoGoalPlacement, H, DynDims](d, row)
            if len(row) != OBS:
                raise Error(
                    "libero bc train: the host writer produced "
                    + String(len(row)) + " words, the model def says "
                    + String(OBS)
                )
            for j in range(ACT):
                arow[j] = act_col[r * ACT + j]
            data.add(row, arow, is_val[e])
    var n_tr = data.n_tr
    print("  frames:", n_tr, "train /", data.n_va, "val (held out the last",
          val_demos, "demos of each task ) in",
          Float64(perf_counter_ns() - t0) / 1e9, "s")
    # ⚠ ANTI-VACUITY ON THE REBUILT OBSERVATION. A writer that produced a
    # constant row — a stale `Data`, an FK that never ran, a goal-word block
    # left at zero — trains a policy on nothing and reports a small MSE for it.
    # The last nine words are the GOAL block (gripper xyz, subject - gripper,
    # target - subject); they move frame to frame in a demonstration.
    var moved = 0
    var goal_nonzero = 0
    for j in range(OBS):
        if data.x_tr[j] != data.x_tr[(n_tr - 1) * OBS + j]:
            moved += 1
    for j in range(OBS - 9, OBS):
        if Float64(data.x_tr[j]) != 0.0:
            goal_nonzero += 1
    print("         first vs last training row:", moved, "of", OBS,
          "words differ |", goal_nonzero, "of the 9 goal words nonzero")
    if moved == 0 or goal_nonzero == 0:
        raise Error(
            "libero bc train: the rebuilt observation is constant or has an"
            " empty goal block — the frames were not written"
        )
    # ── the run: checkpoint, metrics and record under runs/<id>/ ──────────
    var run = RunContext(
        project=String("libero"),
        driver=String("examples/libero/libero_bc_train.mojo"),
        slug=String("bc-libero-goal"),
        env=String("builtin:libero_goal"),
        dataset=String(STORE),
    )
    if out_path.byte_length() == 0:
        out_path = run.checkpoint_path(String("last"))
    print("  run   :", run.dir)
    print("  ckpt  :", out_path)
    var logger = run_logger(run)
    logger.set_config("algorithm", "BC")
    logger.set_config("suite", "libero_goal")
    logger.set_config("store", String(STORE))
    logger.set_config("epochs", String(epochs))
    logger.set_config("lr", String(lr))
    logger.set_config("batch", String(BATCH))
    logger.set_config("val_demos", String(val_demos))
    logger.set_config("train_rows", String(n_tr))
    logger.set_config("val_rows", String(data.n_va))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    # ── normalise, baseline, fit, save, re-score (`deep_agents/bc/fit.mojo`) ─
    var rep = fit_bc[OBS, ACT, BATCH](
        data, epochs, lr, out_path, String("libero bc train")
    )
    announce_checkpoint(out_path, artifacts, run.dir)
    announce_checkpoint(out_path + ".norm", artifacts, run.dir)
    print("=== best val MSE", rep.best_val, "against zero", rep.mse_zero,
          "and mean", rep.mse_mean, "===")
    finish_run(
        run, logger, artifacts,
        String("best_val_mse=") + String(rep.best_val)
        + " reloaded_val_mse=" + String(rep.reloaded_val)
        + " mse_zero=" + String(rep.mse_zero)
        + " mse_mean=" + String(rep.mse_mean),
    )
    print("  run record:", run.kv_path())
