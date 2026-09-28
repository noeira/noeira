# +--------------------------------------------------------------------------+ #
# | ACT on the SO-ARM101 — open-loop evaluation on a held-out episode
# +--------------------------------------------------------------------------+ #
"""Walk a held-out episode step by step, query the policy at EVERY step, combine
the overlapping chunks with temporal ensembling, and compare the resulting
action stream against what was actually recorded.

    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/act_eval \\
        examples/so101/act_so101_openloop_eval.mojo && /tmp/act_eval
    ACT_CKPT=<run_id> /tmp/act_eval    # a training run's best.ckpt (or a .ckpt path)

This is the whole inference path — chunked query, ring buffer, exponential
weighting, denormalization back to lerobot units — exercised exactly as a
deployment would drive it, without touching hardware.

## ⚠ Open-loop is not closed-loop

Observations come from the RECORDING, not from where the policy's own actions
would have taken the arm. So this measures "does the policy predict the
demonstrated action from the demonstrated observation", which is a necessary
condition for the policy working and not a sufficient one. Compounding error —
the thing action chunking exists to fight — is invisible here by construction.
Closed-loop needs the arm.

## The baselines are the point

An error number on its own says nothing. Two references are reported beside it:

* **hold** — predict the CURRENT joint positions for every future step. This is
  the trivial policy, and on a slow demonstration it is a strong one; a model
  that cannot beat it has learned nothing useful.

  ⚠⚠ IT IS REPORTED TWICE, AND ONLY ONE OF THE TWO IS ACT'S BASELINE.
  `hold 1-step` compares each step's action with the position measured AT
  that step: a horizon of zero. ACT's action for step `t` is the temporal
  ensemble of every chunk queried in the last K steps, weighted toward the
  OLDEST (`m = 0.01`), so what it scores was on average predicted ~K/2 steps
  earlier. `hold ens.` is hold run through the SAME ensemble — query `i`
  predicts `q_i` for every future step, and step `t` averages those with
  ACT's own weights — and it is the column ACT is judged against. Before
  2026-09-22 only the 1-step column existed, and the first checkpoint with a
  working decoder read as "does NOT beat hold" (9.94 vs 2.00) while it beat
  the horizon-matched hold (14.4 by the weighted per-horizon error) by ~30%.
* **mean** — predict the dataset's mean action. Beating only this is not
  evidence of anything.

Also reported per joint, because a model can look fine in aggregate while
ignoring the gripper — the one dimension that decides whether a grasp happens.
"""

from std.python import Python, PythonObject

from std.math import exp
from noeira.nn.constants import DT
from noeira.deep_agents.act.config import (
    ACT_TEMPORAL_ENSEMBLE_M,
    RUN_DEC_LAYERS,
    RUN_DIM,
    RUN_ENC_LAYERS,
    RUN_FF,
    RUN_HEADS,
    RUN_K,
    RUN_LATENT,
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from noeira.deep_agents.act.data import ACTDataset
from noeira.deep_agents.act.inference import TemporalEnsemble, denormalize
from noeira.deep_agents.act.trainer import ACTTrainer
from noeira.core.run import resolve_checkpoint


comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W
# ⚠ FROM `act.config`, not restated here. These dims ARE the checkpoint's
# parameter shapes: this file used to carry its own K=20/dim=64 copy while the
# training example moved to K=60/dim=256, so the checkpoint the trainer told
# you to evaluate could not be loaded by the evaluator.
comptime K = RUN_K
comptime DIM = RUN_DIM
comptime HEADS = RUN_HEADS
comptime FF = RUN_FF
comptime LATENT = RUN_LATENT
comptime N_ENC = RUN_ENC_LAYERS
comptime N_DEC = RUN_DEC_LAYERS
comptime BATCH = 1  # one step at a time, as a deployment would

comptime T = ACTTrainer[
    QPOS,
    ADIM,
    N_CAM,
    IMG_H,
    IMG_W,
    K,
    DIM,
    HEADS,
    FF,
    LATENT,
    N_ENC,
    N_DEC,
    BATCH,
]
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W
comptime DEFAULT_CKPT = "act_so101_best_gpu.ckpt"
"""The GPU run's best checkpoint. `ACT_CKPT` overrides — a RUN ID (its
`runs/<id>/checkpoints/best.ckpt`, what the trainers now write) or a checkpoint
FILE, e.g. a run's `last.ckpt` to evaluate a run that was killed."""


# Joint names, in the dataset's own order (meta/info.json `action.names`).
def joint_names() -> List[String]:
    var v = List[String]()
    v.append(String("shoulder_pan"))
    v.append(String("shoulder_lift"))
    v.append(String("elbow_flex"))
    v.append(String("wrist_flex"))
    v.append(String("wrist_roll"))
    v.append(String("gripper"))
    return v^


def store_path() raises -> String:
    """`$ACT_STORE` if set, else the recording the header names.

    The default is a specific recording, not a pattern: an example that
    silently picked up whichever store happened to be newest in the cache
    would report numbers nobody could attribute to a dataset. Point
    `ACT_STORE` at another store to train on it.
    """
    var os = Python.import_module("os")
    var env = String(
        os.environ.get(PythonObject("ACT_STORE"), PythonObject(""))
    )
    if env.byte_length() > 0:
        return env
    var home = String(os.path.expanduser(PythonObject("~")))
    return (
        home
        + "/.cache/noeira/act_so101/"
        + "DenisLabs__record-test_20260825_094319_"
        + String(IMG_H)
        + "x"
        + String(IMG_W)
        + ".h5"
    )


def main() raises:
    var os = Python.import_module("os")
    var path = store_path()
    if not Bool(os.path.exists(PythonObject(path))):
        print("MISSING STORE: " + path)
        raise Error("store not found")
    var ckpt = String(
        os.environ.get(PythonObject("ACT_CKPT"), PythonObject(""))
    )
    if ckpt.byte_length() == 0:
        ckpt = String(DEFAULT_CKPT)
    else:
        # a RUN ID resolves to its checkpoints/best.ckpt; a file is used as is
        ckpt = resolve_checkpoint(ckpt, String("best"))
    if not Bool(os.path.exists(PythonObject(ckpt))):
        print("MISSING CHECKPOINT: " + ckpt)
        print(
            "run examples/so101/act_so101_train_gpu.mojo first, or set"
            " ACT_CKPT to a checkpoint written at the act.config run dims"
        )
        raise Error("checkpoint not found")

    var ds = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](String(path), seed=7)
    # ⚠ seed 7 — the SAME seed the training example used, so the split is the
    # same and this episode is genuinely held out. A different seed here would
    # silently evaluate on training data.
    var ep = ds.val_eps[0]
    var ep_len = ds.store.episodes.length_of(ep)

    var tr = T.make()
    tr.load(String(ckpt))

    print("ACT / SO-ARM101 — open-loop evaluation")
    print("  checkpoint " + ckpt)
    print(
        "  episode "
        + String(ep)
        + " (held out), "
        + String(ep_len)
        + " steps, chunk "
        + String(K)
        + ", m = "
        + String(ACT_TEMPORAL_ENSEMBLE_M)
    )
    print("")

    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)
    var chunk = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var pred_n = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0.0))
    var pred = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0.0))

    var te = TemporalEnsemble[ADIM, K](m=ACT_TEMPORAL_ENSEMBLE_M)

    var sum_abs = List[Float64](length=ADIM, fill=0.0)
    var sum_hold = List[Float64](length=ADIM, fill=0.0)
    var sum_hold_ens = List[Float64](length=ADIM, fill=0.0)
    var sum_mean = List[Float64](length=ADIM, fill=0.0)
    var n = 0
    var g0 = ds.store.episodes.start_of(ep)

    for t in range(ep_len):
        ds.fill_at[K](0, ep, t, qpos, images, actions, valid)
        tr.predict(qpos, images, actions, valid, chunk)
        te.push(t, chunk, 0)
        te.action_at(t, pred_n)
        denormalize(pred_n, 0, ds.action_mean, ds.action_std, pred, 0, ADIM)

        for j in range(ADIM):
            var truth = Float64(ds.action_raw[(g0 + t) * ADIM + j])
            sum_abs[j] += abs(Float64(pred[j]) - truth)
            # `hold 1-step`: keep the current measured joint position.
            sum_hold[j] += abs(
                Float64(ds.qpos_raw[(g0 + t) * QPOS + j]) - truth
            )
            # `hold ens.`: hold through ACT's ensemble — the queries that
            # contribute to step t (the window [t-K+1, t]) each predict the
            # position they measured, weighted exp(-m * (i - i_min)) exactly
            # as `TemporalEnsemble.action_at` weights ACT's chunks.
            var i_min = t - K + 1 if t - K + 1 > 0 else 0
            var wsum = 0.0
            var acc = 0.0
            for i in range(i_min, t + 1):
                var w = exp(-ACT_TEMPORAL_ENSEMBLE_M * Float64(i - i_min))
                wsum += w
                acc += w * Float64(ds.qpos_raw[(g0 + i) * QPOS + j])
            sum_hold_ens[j] += abs(acc / wsum - truth)
            # `mean`: the dataset's average action.
            sum_mean[j] += abs(Float64(ds.action_mean[j]) - truth)
        n += 1

        if t % 50 == 0:
            print(
                "    t="
                + String(t)
                + "  contributors "
                + String(te.n_contributors(t))
            )

    print("")
    print("  mean |error| in lerobot units (degrees; gripper 0-100)")
    print("    joint            ACT      hold ens.   hold 1-step   mean")
    var names = joint_names()
    var tot_act = Float64(0.0)
    var tot_hold = Float64(0.0)
    var tot_hold_ens = Float64(0.0)
    var tot_mean = Float64(0.0)
    for j in range(ADIM):
        var a = sum_abs[j] / Float64(n)
        var h = sum_hold[j] / Float64(n)
        var he = sum_hold_ens[j] / Float64(n)
        var m = sum_mean[j] / Float64(n)
        tot_act += a
        tot_hold += h
        tot_hold_ens += he
        tot_mean += m
        var nm = names[j]
        while nm.byte_length() < 14:
            nm += " "
        print(
            "    "
            + nm
            + "  "
            + String(a)
            + "   "
            + String(he)
            + "   "
            + String(h)
            + "   "
            + String(m)
        )
    print("")
    print(
        "    ALL             "
        + String(tot_act / Float64(ADIM))
        + "   "
        + String(tot_hold_ens / Float64(ADIM))
        + "   "
        + String(tot_hold / Float64(ADIM))
        + "   "
        + String(tot_mean / Float64(ADIM))
    )
    print("")
    if tot_act < tot_hold_ens:
        print("  ACT beats `hold ens.` (hold at ACT's own horizons) — the policy is"
              " using the observation. `hold 1-step` is a zero-horizon"
              " reference, not ACT's baseline.")
    else:
        print(
            "  ⚠ ACT does NOT beat `hold ens.` — holding the arm still, run through"
            " the same ensemble, predicts the demonstration as well. The policy"
            " has not learned anything the closed loop could use; check its"
            " chunks with `pixi run act-so101-inspect` (a flat chunk is the"
            " pre-2026-09-21 decoder) before blaming the data."
        )
