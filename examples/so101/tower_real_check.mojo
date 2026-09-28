"""THE VISION STUDENT ON RECORDED FRAMES — teacher-forced action error, any store.

    pixi run -e apple mojo run -I . examples/so101/tower_real_check.mojo \\
        --ckpt <run_id> --student-zero none \\
        --store ~/.cache/noeira/act_so101/so101-tower__cube-in-bowl_240x320_undist.h5 \\
        --store-zero follower                                   # the REAL rig
    pixi run -e apple mojo run -I . examples/so101/tower_real_check.mojo \\
        --ckpt <run_id> --student-zero none \\
        --store projects/so101-tower/demos/<its store>.rendered.h5 \\
        --store-zero none --split val                           # its own sim val

`DOMAIN_RANDOMIZATION_PLAN.md` Phase 4: the only real data without a desk
session is the rig's LeRobot recording, imported with `--undistort` (the real
fisheye frames brought to the sim's pinhole, 320x240). This walks its episodes
with the student querying at EVERY step on the RECORDED observation (both
cameras + the six joints), the chunks temporally ensembled exactly as the
closed-loop eval does (`tasks/so101_tower_act_eval.mojo`), and reports the
mean |predicted - recorded| action per joint, in the STORE's units (degrees;
gripper 0..100). Run on a sim store's held-out episodes (`--split val`) it
gives the sim number the real one is compared with.

## ⚠⚠ THE UNITS ARE TWO CHOICES, BOTH REQUIRED

A student speaks the joint zero of the store it was TRAINED on
(`tower_demo_rerender.mojo --joint-zero`, absent = `none`); a real recording
speaks the real arm's, which is `follower` by definition
(`robot/so101/sim_map.mojo`: pan -10.7, lift -3.6, elbow -7.3 deg). Every
joint crosses between them through `So101TowerUnits` (store -> model radians
-> student, and back for the prediction). Mixing them up moves the pan by
10.7 degrees and nothing raises, so neither has a default.

## ⚠ The normalisation is the STUDENT's

`norm.json` beside the checkpoint, never the scored store's own moments: a sim
student fed real joints standardised by the real store's statistics would be
shown a different input than the one it was trained on.

## What the numbers are, and are not

* `hold ens.` is the ACT-horizon baseline of `act_so101_openloop_eval.mojo`
  (hold the measured joints, through the same ensemble); `mean` predicts the
  student's training mean action. A student that does not beat `hold ens.` on
  real frames is not using what it sees there.
* `--moving-only` scores only the rows with the arm off its folded rest.
  Every real episode idles at rest ~2 s before moving and folds back to rest
  for ~3 s after (the operator's reaction time and the recording's tail —
  ~40% of the rows); the same frame then carries "hold" and "move", and no
  sim expert demonstrates the idle. The student is still queried every step,
  so its ensemble is the one a deployment runs.
* The real recording is TELEOP of a different scene (another cube and bowl,
  the leader arm and the operator in the overhead view): the demonstrator is
  not the sim expert. The number RANKS students on the same frames (the plan's
  G4: DR vs no-DR); it does not certify one
  (`_a_fit_to_the_demo_manifold_says_nothing_one_step_off_it`).
"""

from std.os.path import exists, isdir
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from std.math import exp
from max.gpu.host import DeviceContext

from noeira.nn.core.ptr import mptr
from noeira.deep_agents.act.config import (
    SO101_ADIM, SO101_IMG_H, SO101_IMG_W, SO101_N_CAM, SO101_QPOS,
    RUN_K, RUN_DIM, RUN_HEADS, RUN_FF, RUN_LATENT, RUN_ENC_LAYERS,
    RUN_DEC_LAYERS, ACT_TEMPORAL_ENSEMBLE_M,
)
from noeira.deep_agents.act.data import ACTDataset
from noeira.deep_agents.act.trainer import ACTTrainer
from noeira.deep_agents.act.norm_file import ACTNorm
from noeira.deep_agents.act.inference import (
    TemporalEnsemble, normalize_camera_chw, denormalize,
)
from noeira.core.run import resolve_checkpoint
from noeira.io.png import save_png
from noeira.tasks.so101_tower_rig import (
    RIG_DT, RIG_IMG_ELEMS, RIG_CAM_ELEMS, RIG_CAM_H, RIG_CAM_W, RIG_N_CAMS,
    RIG_NPIX, So101TowerUnits,
)
from noeira.utils.fmt import col, pad_right


comptime LANES = 8
"""Episodes walked side by side: one ACT forward at batch LANES per step."""
comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime K = RUN_K
comptime DT = RIG_DT
comptime T = ACTTrainer[
    SO101_QPOS, SO101_ADIM, SO101_N_CAM, SO101_IMG_H, SO101_IMG_W, RUN_K,
    RUN_DIM, RUN_HEADS, RUN_FF, RUN_LATENT, RUN_ENC_LAYERS, RUN_DEC_LAYERS,
    LANES, target="gpu",
]
comptime DS = ACTDataset[QPOS, ADIM, SO101_N_CAM, SO101_IMG_H, SO101_IMG_W]
comptime REST_LIFT_RAD: Float64 = -93.6 * 3.141592653589793 / 180.0
"""`--moving-only`: a row whose recorded shoulder_lift, in MODEL radians, is
below this is the arm folded at rest — -90 LeRobot deg on the rig's follower
(zero -3.6); the rest itself sits at -109.4. `tools/soarm/grasp_posture.py`'s
"off rest" rule."""
comptime DEFAULT_SPLIT_SEED = 7
"""The trainer's default `ACT_SEED`: the episode split `--split val` reproduces."""


def _usage() -> String:
    return String(
        "usage: tower_real_check.mojo --ckpt RUN_ID|DIR|FILE [--ckpt-name best|last]"
        " [--norm FILE] --student-zero none|follower|follower-v1 --store FILE.h5"
        " --store-zero none|follower|follower-v1 [--split all|val] [--split-seed S]"
        " [--episodes N] [--m M] [--dump FILE.csv] [--moving-only]"
        " [--mask-overhead x0,y0,x1,y1] [--mask-wrist x0,y0,x1,y1]"
        " [--mask-rgb r,g,b] [--mask-preview F.png]"
    )


def _joint_name(j: Int) -> String:
    var v: List[String] = [
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
        "wrist_roll", "gripper",
    ]
    return v[j]


def main() raises:
    comptime if not has_accelerator():
        print("  SKIPPED: no accelerator — ACT runs on the device here")
        print("=== SKIPPED (this is not a pass) ===")
        return
    comptime assert SO101_N_CAM == RIG_N_CAMS
    comptime assert SO101_IMG_H == RIG_CAM_H and SO101_IMG_W == RIG_CAM_W

    # ── args ──────────────────────────────────────────────────────────────
    var args = argv()
    var ckpt_arg = String("")
    var ckpt_name = String("best")
    var norm_path = String("")
    var student_zero = String("")
    var store_path = String("")
    var store_zero = String("")
    var split = String("all")
    var split_seed = DEFAULT_SPLIT_SEED
    var max_eps = 0
    var ens_m = ACT_TEMPORAL_ENSEMBLE_M
    var dump_path = String("")
    var moving_only = False
    var mask = List[Int]()
    var mask_w = List[Int]()
    var mask_rgb: List[Int] = [205, 205, 200]
    var mask_preview = String("")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--moving-only":
            moving_only = True
            i += 1
            continue
        if not a.startswith("--") or i + 1 >= len(args):
            raise Error("bad argument " + a + "\n" + _usage())
        var v = String(args[i + 1])
        if a == "--ckpt":
            ckpt_arg = v
        elif a == "--ckpt-name":
            ckpt_name = v
        elif a == "--norm":
            norm_path = v
        elif a == "--student-zero":
            student_zero = v
        elif a == "--store":
            store_path = v
        elif a == "--store-zero":
            store_zero = v
        elif a == "--split":
            split = v
        elif a == "--split-seed":
            split_seed = Int(v)
        elif a == "--episodes":
            max_eps = Int(v)
        elif a == "--m":
            ens_m = Float64(v)
        elif a == "--dump":
            dump_path = v
        elif a == "--mask-overhead":
            for w in v.split(","):
                mask.append(Int(String(w).strip()))
            if len(mask) != 4:
                raise Error("--mask-overhead takes x0,y0,x1,y1")
        elif a == "--mask-wrist":
            for w in v.split(","):
                mask_w.append(Int(String(w).strip()))
            if len(mask_w) != 4:
                raise Error("--mask-wrist takes x0,y0,x1,y1")
        elif a == "--mask-rgb":
            mask_rgb = List[Int]()
            for w in v.split(","):
                mask_rgb.append(Int(String(w).strip()))
        elif a == "--mask-preview":
            mask_preview = v
        else:
            raise Error("unknown option " + a + "\n" + _usage())
        i += 2
    if ckpt_arg.byte_length() == 0 or store_path.byte_length() == 0:
        raise Error("--ckpt and --store are required\n" + _usage())
    if student_zero.byte_length() == 0 or store_zero.byte_length() == 0:
        raise Error(
            "--student-zero and --store-zero are both required: a real"
            " recording is `follower`, a rendered store is what"
            " tower_demo_rerender was given (absent = none)\n" + _usage()
        )
    if split != "all" and split != "val":
        raise Error("--split is all or val, not " + split)
    var ckpt_path: String
    if isdir(ckpt_arg):
        ckpt_path = ckpt_arg + "/" + ckpt_name + ".ckpt"
        if norm_path.byte_length() == 0:
            norm_path = ckpt_arg + "/norm.json"
    elif exists(ckpt_arg):
        ckpt_path = ckpt_arg
    else:
        ckpt_path = resolve_checkpoint(ckpt_arg, ckpt_name)
        if norm_path.byte_length() == 0:
            norm_path = String(ckpt_path[byte=0 : ckpt_path.rfind("/")]) + "/norm.json"
    if norm_path.byte_length() == 0:
        raise Error("--norm is required when --ckpt names a file")
    for pth in [ckpt_path, norm_path, store_path]:
        if not exists(pth):
            raise Error("no such file: " + pth)

    var u_student = So101TowerUnits(student_zero)
    var u_store = So101TowerUnits(store_zero)
    var nm = ACTNorm.load(norm_path, QPOS, ADIM)
    # the store: its joint columns and the trainer's split, images streamed
    var ds = DS(String(store_path), seed=UInt64(split_seed), max_image_bytes=0)
    var eps = List[Int]()
    if split == "val":
        for e in ds.val_eps:
            eps.append(e)
    else:
        for e in range(ds.store.n_episodes()):
            eps.append(e)
    if max_eps > 0 and len(eps) > max_eps:
        var kept = List[Int]()
        for k in range(max_eps):
            kept.append(eps[k])
        eps = kept^

    print("=" * 78)
    print("so101_tower — vision student, TEACHER-FORCED on recorded frames")
    print("=" * 78)
    print("  student:", ckpt_path)
    print("  norm   :", norm_path, "| trained on", nm.store)
    print("  units  : student", u_student.describe(), "| store", u_store.describe())
    print("  store  :", store_path, "|", ds.store.n_episodes(), "episodes,",
          ds.store.n_rows(), "rows | scoring", len(eps), "(" + split + ")")
    print("  ens    : m =", ens_m, "| chunk", K, "| lanes", LANES)
    if len(mask_w) == 4:
        print("  mask   : wrist x", mask_w[0], "..", mask_w[2], " y", mask_w[1], "..",
              mask_w[3], "painted", mask_rgb[0], mask_rgb[1], mask_rgb[2])
    if len(mask) == 4:
        print("  mask   : overhead x", mask[0], "..", mask[2], " y", mask[1], "..",
              mask[3], "painted", mask_rgb[0], mask_rgb[1], mask_rgb[2])
    if moving_only:
        print("  rows   : MOVING ONLY — rows with the arm folded at rest (idle"
              " before / after the task) are queried but not scored")

    var ctx = DeviceContext()
    var act = T.make(ctx=Optional[DeviceContext](ctx))
    act.load(ckpt_path)

    # ── buffers ───────────────────────────────────────────────────────────
    var qpos_n = List[Scalar[DT]](length=LANES * QPOS, fill=0)
    var images_n = List[Scalar[DT]](length=LANES * RIG_IMG_ELEMS, fill=0)
    var dummy_a = List[Scalar[DT]](length=LANES * K * ADIM, fill=0)
    var dummy_v = List[Scalar[DT]](length=LANES * K, fill=1)
    var chunk = List[Scalar[DT]](length=LANES * K * ADIM, fill=0)
    var pred_n = List[Scalar[DT]](length=ADIM, fill=0)
    var pred = List[Scalar[DT]](length=ADIM, fill=0)
    var img_u8 = List[Scalar[DType.uint8]](length=LANES * RIG_IMG_ELEMS, fill=0)

    var s_act = List[Float64](length=ADIM, fill=0.0)
    var s_hold = List[Float64](length=ADIM, fill=0.0)
    var s_hold_ens = List[Float64](length=ADIM, fill=0.0)
    var s_mean = List[Float64](length=ADIM, fill=0.0)
    var n_scored = 0
    # the student's mean action, in the store's units (the `mean` baseline)
    var mean_store = List[Float64](length=ADIM, fill=0.0)
    for j in range(ADIM):
        mean_store[j] = u_store.joint_to_lerobot(
            j, u_student.lerobot_to_joint(j, Float64(nm.action_mean[j]))
        )
    var t0 = perf_counter_ns()
    var ns_io = 0
    # `--dump`: one line per scored row, store units, for a per-phase look
    var dump = String("")
    if dump_path.byte_length() > 0:
        dump = String("episode,t")
        for j in range(ADIM):
            dump += ",pred" + String(j)
        for j in range(ADIM):
            dump += ",act" + String(j)
        for j in range(QPOS):
            dump += ",qpos" + String(j)
        dump += "\n"

    var r0 = 0
    while r0 < len(eps):
        var n_l = min(LANES, len(eps) - r0)
        var starts = List[Int]()
        var lens = List[Int]()
        var slabs = List[List[Scalar[DType.uint8]]]()
        var t_io = perf_counter_ns()
        for l in range(n_l):
            var e = eps[r0 + l]
            var s = ds.store.episodes.start_of(e)
            var ln = ds.store.episodes.length_of(e)
            starts.append(s)
            lens.append(ln)
            var slab = List[Scalar[DType.uint8]](unsafe_uninit_length=ln * RIG_IMG_ELEMS)
            ds.store.read_range[DType.uint8](
                String("images"), s, s + ln, mptr(slab.unsafe_ptr())
            )
            slabs.append(slab^)
        ns_io += perf_counter_ns() - t_io
        var t_max = 0
        for l in range(n_l):
            t_max = max(t_max, lens[l])
        var ens = List[TemporalEnsemble[ADIM, K]]()
        for _ in range(LANES):
            ens.append(TemporalEnsemble[ADIM, K](m=ens_m))

        for t in range(t_max):
            # ── observations: a finished lane repeats its last row ───────
            for l in range(LANES):
                var src_l = l if l < n_l else 0
                var tt = min(t, lens[src_l] - 1)
                var row = starts[src_l] + tt
                var o = l * RIG_IMG_ELEMS
                for k in range(RIG_IMG_ELEMS):
                    img_u8[o + k] = slabs[src_l][tt * RIG_IMG_ELEMS + k]
                if len(mask) == 4:
                    # slot 0 = overhead, CHW; the rectangle painted flat
                    for c in range(3):
                        for y in range(max(0, mask[1]), min(RIG_CAM_H, mask[3])):
                            for x in range(max(0, mask[0]), min(RIG_CAM_W, mask[2])):
                                img_u8[o + c * RIG_NPIX + y * RIG_CAM_W + x] = UInt8(mask_rgb[c])
                if len(mask_w) == 4:
                    # slot 1 = wrist
                    for c in range(3):
                        for y in range(max(0, mask_w[1]), min(RIG_CAM_H, mask_w[3])):
                            for x in range(max(0, mask_w[0]), min(RIG_CAM_W, mask_w[2])):
                                img_u8[o + RIG_CAM_ELEMS + c * RIG_NPIX + y * RIG_CAM_W + x] = UInt8(mask_rgb[c])
                if len(mask) == 4:
                    if mask_preview.byte_length() > 0 and l == 0 and t == 0 and r0 == 0:
                        var hwc = List[UInt8](length=RIG_NPIX * 3, fill=0)
                        for q in range(RIG_NPIX):
                            for c in range(3):
                                hwc[q * 3 + c] = img_u8[o + c * RIG_NPIX + q]
                        save_png(mask_preview, hwc, RIG_CAM_W, RIG_CAM_H, 3)
                for c in range(RIG_N_CAMS):
                    var oc = o + c * RIG_CAM_ELEMS
                    normalize_camera_chw[RIG_CAM_H, RIG_CAM_W](img_u8, oc, images_n, oc)
                for k in range(QPOS):
                    var v_store = Float64(ds.qpos_raw[row * QPOS + k])
                    var v_st = u_student.joint_to_lerobot(
                        k, u_store.lerobot_to_joint(k, v_store)
                    )
                    qpos_n[l * QPOS + k] = (Scalar[DT](v_st) - nm.qpos_mean[k]) / nm.qpos_std[k]
            act.predict(qpos_n, images_n, dummy_a, dummy_v, chunk)
            # ── score each live lane at step t ───────────────────────────
            for l in range(n_l):
                ens[l].push(t, chunk, l * K * ADIM)
                if t >= lens[l]:
                    continue
                var row = starts[l] + t
                if moving_only and u_store.lerobot_to_joint(
                    1, Float64(ds.qpos_raw[row * QPOS + 1])
                ) < REST_LIFT_RAD:
                    continue
                ens[l].action_at(t, pred_n, 0)
                denormalize(pred_n, 0, nm.action_mean, nm.action_std, pred, 0, ADIM)
                var i_min = t - K + 1 if t - K + 1 > 0 else 0
                if dump_path.byte_length() > 0:
                    var line = String(eps[r0 + l]) + "," + String(t)
                    for j in range(ADIM):
                        line += "," + String(u_store.joint_to_lerobot(
                            j, u_student.lerobot_to_joint(j, Float64(pred[j]))
                        ))
                    for j in range(ADIM):
                        line += "," + String(Float64(ds.action_raw[row * ADIM + j]))
                    for j in range(QPOS):
                        line += "," + String(Float64(ds.qpos_raw[row * QPOS + j]))
                    dump += line + "\n"
                for j in range(ADIM):
                    var p = u_store.joint_to_lerobot(
                        j, u_student.lerobot_to_joint(j, Float64(pred[j]))
                    )
                    var truth = Float64(ds.action_raw[row * ADIM + j])
                    s_act[j] += abs(p - truth)
                    s_hold[j] += abs(Float64(ds.qpos_raw[row * QPOS + j]) - truth)
                    var wsum = 0.0
                    var acc = 0.0
                    for q in range(i_min, t + 1):
                        var w = exp(-ens_m * Float64(q - i_min))
                        wsum += w
                        acc += w * Float64(ds.qpos_raw[(starts[l] + q) * QPOS + j])
                    s_hold_ens[j] += abs(acc / wsum - truth)
                    s_mean[j] += abs(mean_store[j] - truth)
                n_scored += 1
        r0 += n_l
        print("   ", r0, "/", len(eps), "episodes |", n_scored, "rows scored")

    var secs = Float64(perf_counter_ns() - t0) / 1e9
    print("")
    print("  mean |predicted - recorded| action, store units (deg; gripper 0..100)")
    print("    " + pad_right(String("joint"), 15) + "  student  hold ens.  hold 1-step     mean")
    var tot = List[Float64](length=4, fill=0.0)
    for j in range(ADIM):
        var a = s_act[j] / Float64(n_scored)
        var he = s_hold_ens[j] / Float64(n_scored)
        var h = s_hold[j] / Float64(n_scored)
        var m = s_mean[j] / Float64(n_scored)
        tot[0] += a
        tot[1] += he
        tot[2] += h
        tot[3] += m
        print("    " + pad_right(_joint_name(j), 15) + col(a, 9, 2) + col(he, 10, 2)
              + col(h, 13, 2) + col(m, 9, 2))
    var n6 = Float64(ADIM)
    print("    " + pad_right(String("ALL"), 15) + col(tot[0] / n6, 9, 2)
          + col(tot[1] / n6, 10, 2) + col(tot[2] / n6, 13, 2) + col(tot[3] / n6, 9, 2))
    print("")
    print("  " + String(n_scored) + " rows in " + String(Int(secs)) + " s (image io "
          + String(Int(Float64(ns_io) / 1e9)) + " s)")
    if dump_path.byte_length() > 0:
        with open(dump_path, "w") as fh:
            fh.write(dump)
        print("  dump   :", dump_path, "(" + String(n_scored) + " rows)")
    print("RESULT ckpt=" + ckpt_path + " store=" + store_path + " split=" + split
          + " student_zero=" + student_zero + " store_zero=" + store_zero
          + " moving_only=" + String(moving_only)
          + " mask=" + (String(mask[0]) + "," + String(mask[1]) + "," + String(mask[2])
                        + "," + String(mask[3]) if len(mask) == 4 else String("none"))
          + " mask_wrist=" + (String(mask_w[0]) + "," + String(mask_w[1]) + ","
                              + String(mask_w[2]) + "," + String(mask_w[3])
                              if len(mask_w) == 4 else String("none"))
          + " rows=" + String(n_scored) + " l1_all=" + String(tot[0] / n6)
          + " hold_ens=" + String(tot[1] / n6) + " mean=" + String(tot[3] / n6))
    if tot[0] < tot[1]:
        print("  the student beats `hold ens.` on these frames")
    else:
        print("  ⚠ the student does NOT beat `hold ens.` on these frames")
    print("=== DONE ===")
