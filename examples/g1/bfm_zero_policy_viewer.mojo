""""Watch a BFM-Zero G1 checkpoint track a LAFAN1 clip — the scored rollout, drawn.

    pixi run mojo build -I . -Xlinker -ld_classic examples/g1/bfm_zero_policy_viewer.mojo -o /tmp/g1view
    pixi run /tmp/g1view --ckpt runs/<id>/checkpoints/step_36000.ckpt
    pixi run /tmp/g1view --ckpt <path> --clips 7 25 --segments 3 --fps 50

## ⚠ THIS IS `g1_score_segment` WITH A WINDOW OPEN

It does not re-implement the rollout. `g1_score_segment(..., render=True)` is
the SAME function the EMD comes out of, so what you watch is what the number
measured. A viewer with its own copy of the loop would be a second home for the
z schedule (`z_t = project(B(row t+1))`), the `last_action`/history rules and
the `x5 -> clip -> x0.25*effort/kp` action chain — three things this track has
already got wrong once each — and it would then be showing something no metric
ever scored (`_a_rule_written_inline_twice_drifts`).

Each segment prints its own `distance` / `emd` / `proximity` as it finishes, so
the picture and the number are on screen together. A clip whose EMD is 3.2 and
one whose EMD is 1.1 look completely different, and after §12.33 we know the
aggregate hides that: 17 % of windows FAIL outright (proximity <= 0.5) while
the rest track within 14 % of the reference. Watching is how you tell which
kind of failure a bad number is — a fall, a drift, or a wrong limb.

## What you are looking at

The robot is driven by the policy; there is no target ghost rendered. The
reference motion is the store rows the segment came from, and the number that
compares them is `emd` / `distance` in the printout. Judge the MOTION (does it
look like walking / a fall recovery) and read the numbers for how close.

⚠ 50 Hz is the control rate. `--fps` paces the WINDOW, not the physics: the
rollout is deterministic and identical at any pace, so slowing it down is free
and is the only way to see a 10-second fall recovery properly.

## ⚠ THE DIMS COME FROM `g1_tracking_eval.mojo`

`G1_H` / `G1_L` / `G1_D` and the observation widths are imported, not restated,
so this viewer cannot drift from the trainer the way two hand-kept copies do
(§12.22 moved them into one place for exactly this reason). A checkpoint from a
different tower fails loudly on a shape mismatch, which is what you want.

⚠ CPU PHYSICS. The env is the float64 single-env path; the checkpoint trained
against the GPU batched path. §11 measured those two agreeing to three decimals
on 37 of 39 segments with the RELEASED actor, so treat small differences as
path noise.
"""

from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.obs_norm import ObsNorm
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTowerFiltered, BFMBNetFiltered,
)
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_rsi import G1RsiTable
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA,
)
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
)
from noeira.envs.robots.g1_tracking_eval import (
    G1_SEG_ROWS, G1_D, G1_H, G1_L, G1_HB,
    g1_n_segments, g1_segment_row, g1_segment_pick, g1_score_segment,
)

comptime SP: Int = UNITREE_G1_OBS_DIM
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = G1_D
comptime H: Int = G1_H
comptime L: Int = G1_L
comptime HB: Int = G1_HB
comptime BATCH: Int = 64          # the trainer's batch — unused here, kept small
comptime NQ = UnitreeG1Model.NQ
comptime SEG_ROWS: Int = G1_SEG_ROWS

comptime FNet = BFMFTower[OBS, ACT, D, H, L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, HB]
comptime ANet = BFMActorTowerFiltered[
    OBS, UNITREE_G1_STATE_DIM, G1_ACTOR_EXTRA, D, H, L, ACT
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _flag_ints(name: String) raises -> List[Int]:
    """Every integer after `--name`, until the next `--flag`."""
    var out = List[Int]()
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            var j = i + 1
            while j < len(av) and not String(av[j]).startswith("--"):
                out.append(atol(String(av[j])))
                j += 1
            break
    return out^


def main() raises:
    var ckpt = _flag(String("--ckpt"), String(""))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var max_segments = atol(_flag(String("--segments"), String(2)))
    var fps = atol(_flag(String("--fps"), String("50")))
    var clips = _flag_ints(String("--clips"))
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")
    var delay_ms = 1000 // fps if fps > 0 else 0

    print("=" * 66)
    print("BFM-Zero G1 — the scored tracking rollout, rendered")
    print("=" * 66)
    print("  dims   obs", OBS, "(stored", SP, "+ derived", G1_ACTOR_EXTRA, ")",
          " act", ACT, " d", D, " h", H, " L", L)
    print("  ckpt  ", ckpt)
    print("  pacing", fps, "fps ->", delay_ms, "ms/frame (the ROLLOUT is"
          " unaffected; only the window is paced)")

    var t = Trainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")
    if norm:
        print("  normaliser sidecar applied")
    else:
        print("  ⚠ no .norm sidecar beside the checkpoint: RAW inputs. If the"
              " run trained with `normalize_obs=True` the policy will look"
              " far worse than its EMD — this is the first thing to check when"
              " a viewer disagrees with a metric.")

    var store = TrajectoryStore(store_path)
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    var qpos_col = store.load_column[DType.float32](String("qpos"))
    var rsi = G1RsiTable.from_store(store)
    if len(clips) == 0:
        for c in range(rsi.n_ep):
            clips.append(c)

    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    if not env.init_renderer():
        print("  ⚠ no renderer available — scoring headless instead.")
    var obs_t = Tensor.alloc(OBS)
    var z1 = Tensor.alloc(D)
    var act_out = Tensor.alloc(ACT)
    var b_in = Tensor.alloc(SEG_ROWS * OBS)
    var b_out = Tensor()
    var z_seg = Tensor.alloc(SEG_ROWS * D)
    var ach = List[Float64](length=SEG_ROWS * ACT, fill=0.0)
    var tgt = List[Float64](length=SEG_ROWS * ACT, fill=0.0)

    var n = 0
    var sum_emd = 0.0
    var sum_prox = 0.0
    var t0 = perf_counter_ns()
    for ci in range(len(clips)):
        if env.check_renderer_quit():
            break
        var clip = clips[ci]
        var n_avail = g1_n_segments(Int(rsi.ep_len.data[clip]))
        var n_seg = n_avail if n_avail < max_segments else max_segments
        # the clip's NAME lives in the store's task table and is only reachable
        # through the Python oracle (`proto.keys`); the index is enough to line
        # a segment up with an eval CSV row, and keeping the oracle out of this
        # file means the viewer needs no h5py/numpy to open a window
        print("  clip", clip, ":", n_seg, "of", n_avail, "segments")
        for k in range(n_seg):
            if env.check_renderer_quit():
                break
            # spread the picks over the clip — the OPENING ten seconds are the
            # easy part and score ~0.19 low (§12.29), so a viewer that always
            # showed segment 0 would flatter the policy
            var seg = g1_segment_pick(n_avail, n_seg, k)
            var r0 = g1_segment_row(Int(rsi.ep_offset.data[clip]), seg)
            var sc = g1_score_segment[FNet, BNet, ANet, OBS, ACT, D, BATCH](
                t, env, rsi, st, pv, qpos_col, norm, r0,
                ach, tgt, b_in, b_out, z_seg, obs_t, z1, act_out,
                render=True, frame_delay_ms=delay_ms,
            )
            n += 1
            sum_emd += sc.emd
            sum_prox += sc.proximity
            print("    seg", seg, " distance", sc.distance, " emd", sc.emd,
                  " proximity", sc.proximity,
                  "   <-- FAILED" if sc.proximity <= 0.5 else "")
    env.close()
    var el = Float64(perf_counter_ns() - t0) * 1e-9
    print("-" * 66)
    if n > 0:
        print("  segments", n, " mean emd", sum_emd / Float64(n),
              " mean proximity", sum_prox / Float64(n), " in", el, "s")
        print("  ⚠ proximity <= 0.5 is the FAILURE mode §12.33 found: 17 % of"
              " windows lose the motion outright while the rest track within"
              " 14 % of the reference. A mean hides which of the two you are"
              " looking at.")
    else:
        print("  nothing scored")
