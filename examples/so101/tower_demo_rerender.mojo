"""THE TOWER'S DEMONSTRATIONS, RENDERED THROUGH ITS TWO CAMERAS — the image store
a vision student trains on, in the REAL dataset's layout and units.

    pixi run -e apple mojo run -I . examples/so101/tower_demo_rerender.mojo \\
        --demos projects/so101-tower/demos/expert_cube_in_bowl_clean_600.demo \\
        --episodes 20 --out /tmp/tower_smoke.rendered.h5

Reads one or more `.demo` files (the scripted expert's, a DAgger round's, a
teleop session's — `deep_agents/demos/file.mojo`), poses the sim at every
recorded row, and renders the rig's two cameras with the device tracer. It
writes a `TrajectoryStore`

    <out>.rendered.h5
        qpos        (6)   f32   the arm's joints in LEROBOT UNITS: five body
                                joints in DEGREES, the gripper 0..100
        action      (6)   f32   the recorded joint TARGET, same units
        images      (2 x 3 x 240 x 320) u8   CHW, top row first,
                                slot 0 = overhead, slot 1 = wrist
        state       (nq+nv) f64 the sim state the frame was rendered from
        action_sim  (6)   f32   the recorded action as the env takes it
                                (normalised onto each actuator's ctrlrange)

## ⚠⚠ THE REAL DATASET'S LAYOUT, ON PURPOSE

`qpos` / `action` / `images` are what `act_so101_import_dataset.mojo` writes
from the rig's LeRobot recordings (`projects/so101-tower/datasets/cube-in-bowl`:
`observation.state` and `action` are 6-vectors in LeRobot units, and the video
keys are sorted by name, so the OVERHEAD camera is slot 0 and the WRIST is
slot 1). A student trained on this store is the SAME network, with the same
normalisation statistics' meaning, as one trained on the real recordings, and
`act_so101_train_gpu.mojo` / `act_so101_deploy_real.mojo` take it unchanged.
The sim-only columns (`state`, `action_sim`) ride along for the sim eval and
for DAgger; the ACT loader never reads them.

The unit map is `tasks/so101_tower_rig.So101TowerUnits` (sign +1, the gripper
by FRACTION of its range): body joint degrees = (radians - zero) x 180/pi;
gripper = 100 x (q - lo) / (hi - lo) over the actuator's ctrlrange, which is
the joint's range on this model. `--joint-zero` picks `zero`: `none` (the
default, 0, what every store before 2026-09-23 was written with) or
`follower`, the real follower's measured zero, so the store's degrees are
what the REAL arm reports at that pose — the choice a student meant for the
real arm needs. ⚠ `tower_act_eval.mojo` must be given the SAME choice (the
rig module says why); the manifest's provenance line records it. The action is the target the expert
commanded — the `.demo` word denormalised onto the ctrlrange — which is what a
leader arm's position is on the real recording.

## ⚠⚠ ROW r IS THE PICTURE OF obs[r], PAIRED WITH act[r]

A `.demo` row is (obs, act, reward, next_obs). The frame is rendered from
`obs[r]` — the state the demonstrator was IN when it chose `act[r]` — so a
closed-loop student sees exactly the pairing it will be asked to act on.
(`libero_demo_rerender.mojo` documents what the other pairing costs.)

## ⚠ HOST FK, DEVICE PIXELS, AND ONE ROW CHECKED AGAINST THE HOST TRACER

Nothing here steps physics: each row's `qpos` is written into a lane, forward
kinematics runs on the host, and the device tracer renders `LANES` rows per
launch. Before the store is written, the FIRST row is also rendered by the
host leg (`raytrace/host_render.render_lane_cpu`) on a model built through
the RUNTIME parser at the same float32 — an independent path to the same
pose — and the run refuses below `MIN_PSNR_HOST` dB on either camera. It
catches a camera read from the wrong row, a stale pose upload, a wrong group
mask, a lane mix-up; it cannot catch a scene that is wrong in both builds.
The first row's frames are also written as PNGs beside the store. The check
runs on the FIRST launch, so a broken pipeline fails in seconds, and it
carries a negative control: the device overhead frame against the host
WRIST frame must score low, or the match proves nothing (the first run
printed 99 dB, byte-identical, on both cameras).

⚠ THE STORE IS BIG. A row is 460 800 image bytes; deflate (on by default)
shrinks the sim's flat colours several-fold. Check free disk before a full
file: the tool prints the running size.

## `--look calibrated|legacy` — the rig's lights and colours

`calibrated` (the default since 2026-09-24) renders the scene as composed:
lights and albedos fitted to the rig's recorded frames, so the desk and the
props come out at the real cameras' grey levels and hues instead of clipped
white / yellow / cyan (`so101_tower.family`). `legacy` puts back the look
every earlier store was rendered with (`so101_tower_rig.apply_tower_look`),
so the same demos can be rendered both ways and the students compared on the
look alone. Recorded in the provenance line; `tower_act_eval.mojo --look`
must match.

## `--dr off|light|full` — render-time domain randomization

`physics3d/raytrace/randomize.mojo` (DR plan, level L2). ONE appearance draw
per LAUNCH: before each launch the lights, the surface groups' colours, both
cameras' pose/fovy and the background are re-drawn from `(--dr-seed, launch
index)` and the four tables re-uploaded in place; both cameras of a launch
share the draw. A launch is at most `LANES` rows of ONE episode, so an
episode's look changes every `LANES` rows — for ACT (one frame per sample)
that is only more variety. With DR on the store gains a `dr_draw` (int32)
column naming the draw each row was rendered under, so any row can be
re-rendered; with `--dr off` (the default) every column and every frame is
byte for byte what this tool wrote before DR existed (checked on 521 rows);
only the manifest's provenance line gains "; dr off".

The host check runs UNDER THE SAME DRAW: the runtime-built pair gets its own
randomizer with the same config and applies launch 0's draw, so the check
covers the randomized tables too. `--dr-preview K` also writes the overhead
and wrist frames of the first row of the first K launches as PNGs — look at
them before rendering hours of data.
"""

from std.math import pi, log10
from std.memory.alloc import unsafe_alloc
from std.os import makedirs
from std.os.path import exists, dirname, getsize
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStoreWriter
from noeira.deep_agents.demos.file import DemoSet, read_demo_file
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.io.png import save_png
from noeira.math3d import Vec3 as Vec3Generic
from noeira.physics3d.fields import Data, Model, DynDims, actuator_column
from noeira.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.raytrace import BatchedCameraRenderer, RGB_CHANNELS
from noeira.physics3d.raytrace.host_render import render_lane_cpu
from noeira.physics3d.raytrace.visual import build_visual_model
from noeira.physics3d.raytrace.randomize import (
    DomainRandConfig,
    VisualRandomizer,
    geom_labels,
    so101_tower_surface_groups,
)
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.so101_tower_xml import (
    So101TowerModel, SO101_TOWER_MAX_CONTACTS, SO101_TOWER_NMESH_VERTS,
)
from noeira.nn.core.ptr import mptr
from noeira.tasks.so101_tower_rig import (
    TOWER_MD, RIG_CAM_W, RIG_CAM_H, RIG_N_CAMS, RIG_SAMPLES,
    RIG_VISUAL_GROUP_MASK, RIG_DR_TARGET, TowerRenderer, make_tower_model,
    make_tower_renderer, tower_cameras, pack_camera_u8, rig_byte,
    So101TowerUnits, RIG_JOINT_ZERO_NONE, RIG_LOOK_CALIBRATED, apply_tower_look,
    scale_tower_camera_dr, tower_camera_dr_describe,
    rig_visual_group_mask,
)
from noeira.tasks.spec import load_family, load_task


comptime DT = DType.float32
"""The device's type — the batched env's `Data` is float32."""
comptime Vec3 = Vec3Generic[DT]
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime TASK_DIR = "noeira/tasks/tasks/"
comptime DEFAULT_TASK = "so101_tower_cube_in_bowl"
comptime MD = TOWER_MD
"""The env's own model shape — `tasks/so101_tower_rig.mojo`, shared with the
closed-loop eval (`tower_act_eval.mojo`) so both render the same pixels."""
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime STATE_DIM = NQ + NV
comptime ACT = 6
comptime GRIPPER = 5
comptime LANES = 32
"""Rows rendered per launch. Comptime: the kernel is instantiated per value."""
comptime CAM_W = RIG_CAM_W
comptime CAM_H = RIG_CAM_H
comptime N_CAMS = RIG_N_CAMS
comptime NPIX = CAM_W * CAM_H
comptime CAM_ELEMS = 3 * NPIX
comptime SAMPLES = RIG_SAMPLES
comptime Renderer = TowerRenderer[LANES]
comptime VISUAL_GROUP_MASK: Int = RIG_VISUAL_GROUP_MASK
comptime MIN_PSNR_HOST = 30.0
"""Device vs host tracer, same pose, same float32. Anything below is a
pipeline defect (pose, camera, mask), not shading."""
comptime MAX_PSNR_SWAPPED = 20.0
"""The device overhead frame against the host WRIST frame must score under
this, or the host check is blind."""
comptime DEFAULT_DEFLATE = 4
comptime DR_TARGET = RIG_DR_TARGET


def _to_byte(x: Float64) -> UInt8:
    return rig_byte(x)


def _usage() -> String:
    return String(
        "usage: tower_demo_rerender.mojo --demos a.demo[,b.demo] [--out F]"
        " [--task NAME] [--episodes N] [--per-file a,b] [--all-episodes]"
        " [--deflate 0-9]"
        " [--no-host-check] [--dr off|light|full] [--dr-seed N]"
        " [--dr-preview K] [--joint-zero none|follower|follower-v1]"
        " [--look calibrated|legacy]"
    )


def main() raises:
    comptime if not has_accelerator():
        print("  SKIPPED: no accelerator — the tracer is a device kernel")
        print("=== SKIPPED (this is not a pass) ===")
        return

    var args = argv()
    var demos_arg = String("")
    var out_path = String("")
    var task_name = String(DEFAULT_TASK)
    var max_eps = 0
    var per_file_arg = String("")
    var success_only = True
    var deflate = DEFAULT_DEFLATE
    var host_check = True
    var dr_name = String("off")
    var dr_seed = 0
    var dr_preview = 0
    var joint_zero = String(RIG_JOINT_ZERO_NONE)
    var look = String(RIG_LOOK_CALIBRATED)
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--all-episodes":
            success_only = False
        elif a == "--no-host-check":
            host_check = False
        elif a.startswith("--"):
            if i + 1 >= len(args):
                raise Error(a + " takes a value\n" + _usage())
            var v = String(args[i + 1])
            if a == "--demos":
                demos_arg = v
            elif a == "--out":
                out_path = v
            elif a == "--task":
                task_name = v
            elif a == "--episodes":
                max_eps = Int(v)
            elif a == "--per-file":
                per_file_arg = v
            elif a == "--deflate":
                deflate = Int(v)
            elif a == "--dr":
                dr_name = v
            elif a == "--dr-seed":
                dr_seed = Int(v)
            elif a == "--dr-preview":
                dr_preview = Int(v)
            elif a == "--joint-zero":
                joint_zero = v
            elif a == "--look":
                look = v
            else:
                raise Error("unknown option " + a + "\n" + _usage())
            i += 1
        else:
            raise Error("unexpected argument " + a + "\n" + _usage())
        i += 1
    if demos_arg.byte_length() == 0:
        raise Error("--demos is required\n" + _usage())
    var demo_paths = List[String]()
    for p in demos_arg.split(","):
        var s = String(String(p).strip())
        if s.byte_length() == 0:
            raise Error("--demos: empty element in '" + demos_arg + "'")
        if not exists(s):
            raise Error("--demos: no such file " + s)
        demo_paths.append(s^)
    # `--per-file 120,80`: at most that many (successful) episodes from each
    # file, in `--demos` order. ⚠ THE ACT GPU PATH HOLDS EVERY IMAGE ON THE
    # DEVICE: 460 800 bytes a row, so a 32 GB card trains on ~50k rows at
    # most. This is how a mixed training store is cut to fit.
    var per_file = List[Int]()
    if per_file_arg.byte_length() > 0:
        for p in per_file_arg.split(","):
            per_file.append(Int(String(String(p).strip())))
        if len(per_file) != len(demo_paths):
            raise Error("--per-file names " + String(len(per_file))
                        + " caps for " + String(len(demo_paths)) + " files")
    if out_path.byte_length() == 0:
        var first = demo_paths[0]
        out_path = String(first[byte = 0 : first.byte_length() - 5]) + ".rendered.h5"

    print("=" * 78)
    print("so101_tower demonstrations rendered through the rig's cameras")
    print("=" * 78)

    # ── the scene, the env's way (device) and the runtime way (host check) ─
    var fam = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(fam))
    var task = load_task(String(TASK_DIR) + task_name + ".task")
    var ctx = DeviceContext()
    var m = make_tower_model(ctx)
    var d = Data[DT, MD, LANES]()
    d.upload_all(ctx)
    ctx.synchronize()
    var cams = tower_cameras(fmd)  # [overhead, wrist] — the real dataset's key order
    var cam_over = cams[0]
    var cam_wrist = cams[1]
    var r = make_tower_renderer[LANES](ctx, fmd, m, look)
    print("  device :", ctx.name(), "|", LANES, "lanes |", CAM_W, "x", CAM_H,
          "|", SAMPLES, "samples | overhead", cam_over, "wrist", cam_wrist)
    print("  " + r.vis.describe())
    var dr_cfg = DomainRandConfig.parse(dr_name, UInt64(dr_seed))
    var dr_on = dr_cfg.enabled
    var labels = geom_labels(fmd)
    var dr = VisualRandomizer[DT](
        dr_cfg, so101_tower_surface_groups(), r.vis, m, labels, cams.copy(),
        r.background, DR_TARGET,
    )
    scale_tower_camera_dr(dr, fmd)
    print("  dr     :", String(dr_cfg))
    if dr_on:
        print("  dr cams:", tower_camera_dr_describe())

    # the actuators' ctrlrange: the action map and the gripper's LeRobot unit
    var units = So101TowerUnits(joint_zero)
    print("  units  :", units.describe())
    print("  look   :", look)

    # ── the demonstrations: checked up front, read again one at a time ────
    var total_eps = 0
    var total_rows = 0
    for k in range(len(demo_paths)):
        var ds = read_demo_file(demo_paths[k])
        if ds.obs_dim < STATE_DIM or ds.act_dim != ACT:
            raise Error(
                demo_paths[k] + ": obs " + String(ds.obs_dim) + " act "
                + String(ds.act_dim) + " — not a so101_tower file (needs obs"
                " >= " + String(STATE_DIM) + ", act " + String(ACT) + ")"
            )
        print("  demos  :", demo_paths[k], "|", len(ds.ep_len), "episodes |",
              ds.count(), "rows | obs", ds.obs_dim)

    # ── the store ─────────────────────────────────────────────────────────
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, ACT))
    cols.append(ColumnSpec(String("action"), DType.float32, ACT))
    cols.append(ColumnSpec(String("images"), DType.uint8, N_CAMS * CAM_ELEMS))
    cols.append(ColumnSpec(String("state"), DType.float64, STATE_DIM))
    cols.append(ColumnSpec(String("action_sim"), DType.float32, ACT))
    if dr_on:
        cols.append(ColumnSpec(String("dr_draw"), DType.int32, 1))
    var dd = dirname(out_path)
    if dd.byte_length() > 0:
        makedirs(dd, exist_ok=True)
    var w = TrajectoryStoreWriter(
        String(out_path), cols^,
        env_id=String("so101_tower_rerender:") + task_name,
        seed=0,
        source_commit=String("so101_tower .demo rows rendered by")
            + " noeira/physics3d/raytrace (batch.mojo) at " + String(CAM_W)
            + "x" + String(CAM_H) + " " + String(SAMPLES) + "x MSAA, groups"
            " 0+2, row 0 = top, slot 0 overhead / 1 wrist, frame r = obs r;"
            " qpos/action in LeRobot units (deg, gripper 0..100), "
            + units.describe() + "; look " + look + "; dr "
            + String(dr_cfg)
            + (" (one draw per launch, see dr_draw; " + tower_camera_dr_describe()
               + ")" if dr_on else ""),
        deflate=deflate,
    )
    w.add_task(0, String(task.language))

    # ── buffers ───────────────────────────────────────────────────────────
    var h_rgb = ctx.enqueue_create_host_buffer[DT](LANES * NPIX * RGB_CHANNELS)
    var im = unsafe_alloc[Scalar[DType.uint8]](
        LANES * N_CAMS * CAM_ELEMS
    ).as_unsafe_any_origin()
    var qb = unsafe_alloc[Scalar[DType.float32]](LANES * ACT).as_unsafe_any_origin()
    var ab = unsafe_alloc[Scalar[DType.float32]](LANES * ACT).as_unsafe_any_origin()
    var asb = unsafe_alloc[Scalar[DType.float32]](LANES * ACT).as_unsafe_any_origin()
    var sb = unsafe_alloc[Scalar[DType.float64]](LANES * STATE_DIM).as_unsafe_any_origin()
    var drb = unsafe_alloc[Scalar[DType.int32]](LANES).as_unsafe_any_origin()
    var launch = 0
    var first_frames = List[UInt8]()   # row 0, both cameras, for the host check
    var first_state = List[Float64]()
    var worst = 99.0
    var swapped = 0.0
    var constant_pictures = 0
    var t_fk = 0
    var t_render = 0
    var t_io = 0
    var t0 = perf_counter_ns()

    for si in range(len(demo_paths)):
        var ds = read_demo_file(demo_paths[si])
        var file_eps = 0
        for e in range(len(ds.ep_len)):
            if max_eps > 0 and total_eps >= max_eps:
                break
            if len(per_file) > 0 and file_eps >= per_file[si]:
                break
            if success_only and not ds.ep_success[e]:
                continue
            file_eps += 1
            var off = ds.ep_start[e]
            var T = ds.ep_len[e]
            var done = 0
            while done < T:
                var n = T - done
                if n > LANES:
                    n = LANES
                # ── host: obs -> qpos -> FK ──────────────────────────────
                var tf = perf_counter_ns()
                for l in range(LANES):
                    var rr = off + done + (l if l < n else 0)
                    for k in range(NQ):
                        d.qpos.data[l * NQ + k] = Scalar[DT](
                            ds.obs[rr * ds.obs_dim + k]
                        )
                forward_kinematics["cpu", DT, MD, LANES](d, m)
                d.qpos.upload_resident(ctx)
                d.xpos.upload_resident(ctx)
                d.xquat.upload_resident(ctx)
                t_fk += perf_counter_ns() - tf
                # ── the launch's appearance draw ─────────────────────────
                if dr_on:
                    r.background = dr.apply(launch, r.vis, m)
                    dr.upload(ctx, r.vis, m)
                    for l in range(n):
                        drb[unsafe_offset=l] = Int32(launch)
                # ── device: both cameras ─────────────────────────────────
                var tr = perf_counter_ns()
                for slot in range(N_CAMS):
                    r.render(ctx, d, m, cams[slot])
                    ctx.enqueue_copy(h_rgb, r.rgb)
                    ctx.synchronize()
                    var p = h_rgb.unsafe_ptr()
                    for l in range(n):
                        var dst = l * N_CAMS * CAM_ELEMS + slot * CAM_ELEMS
                        if pack_camera_u8(mptr(p), l, im, dst):
                            constant_pictures += 1
                    if dr_on and launch < dr_preview:
                        var hwc = List[UInt8](length=CAM_ELEMS, fill=UInt8(0))
                        for q in range(NPIX):
                            for c in range(3):
                                hwc[q * 3 + c] = _to_byte(
                                    Float64(p[unsafe_offset = q * 3 + c])
                                )
                        save_png(
                            out_path + ".dr" + String(launch) + "."
                            + ("overhead" if slot == 0 else "wrist") + ".png",
                            hwc, CAM_W, CAM_H, 3,
                        )
                t_render += perf_counter_ns() - tr
                # ── the rows ─────────────────────────────────────────────
                for l in range(n):
                    var rr = off + done + l
                    for k in range(ACT):
                        var qk = Float64(ds.obs[rr * ds.obs_dim + k])
                        var ak = Float64(ds.act[rr * ACT + k])
                        var tgt = units.action_to_joint(k, ak)
                        qb[unsafe_offset = l * ACT + k] = Float32(
                            units.joint_to_lerobot(k, qk)
                        )
                        ab[unsafe_offset = l * ACT + k] = Float32(
                            units.joint_to_lerobot(k, tgt)
                        )
                        asb[unsafe_offset = l * ACT + k] = Float32(ak)
                    for k in range(STATE_DIM):
                        sb[unsafe_offset = l * STATE_DIM + k] = Float64(
                            ds.obs[rr * ds.obs_dim + k]
                        )
                if len(first_frames) == 0:
                    for k in range(N_CAMS * CAM_ELEMS):
                        first_frames.append(im[unsafe_offset=k])
                    for k in range(STATE_DIM):
                        first_state.append(Float64(ds.obs[off * ds.obs_dim + k]))
                    # ── the host check, on the FIRST launch, before hours of
                    # rendering: row 0 through the runtime model and the host
                    # leg, with a negative control (see the module header).
                    if host_check:
                        var dims = dims_from_flat(
                            fmd, max_contacts=SO101_TOWER_MAX_CONTACTS,
                            nmesh_verts=SO101_TOWER_NMESH_VERTS,
                        )
                        var mh = Model[DT, DynDims](dims)
                        build_model_runtime[DT](fmd, dims, mh)
                        var dh = Data[DT, DynDims, 1](dims)
                        if dims.get_nq() != NQ:
                            raise Error("host model nq " + String(dims.get_nq()) + " != "
                                        + String(NQ))
                        for k in range(NQ):
                            dh.qpos.data[k] = Scalar[DT](first_state[k])
                        forward_kinematics["cpu", DT, DynDims, 1](dh, mh)
                        var vis_h = build_visual_model[DT, DynDims](
                            fmd, mh, group_mask=rig_visual_group_mask(look)
                        )
                        apply_tower_look(vis_h, fmd, look)
                        # The same draw (launch 0) on the host pair, from its
                        # own pristine base — so the check covers the
                        # randomized tables, not only the pose.
                        var bg_h = r.background
                        if dr_on:
                            var dr_h = VisualRandomizer[DT](
                                dr_cfg, so101_tower_surface_groups(), vis_h,
                                mh, labels, cams.copy(), dr.base_background,
                                DR_TARGET,
                            )
                            scale_tower_camera_dr(dr_h, fmd)
                            bg_h = dr_h.apply(0, vis_h, mh)
                        var rgb = List[Scalar[DT]]()
                        var depth = List[Scalar[DT]]()
                        var seg = List[Scalar[DT]]()
                        var refl = List[Scalar[DT]]()
                        # both host frames first, then the matched pairs AND the swapped pair:
                        # ⚠ A MATCH ONLY MEANS SOMETHING IF A MISMATCH SCORES LOW. The first
                        # run of this check printed 99 dB (byte-identical) on both cameras;
                        # the swapped pair (device overhead vs host wrist) is the negative
                        # control that proves the comparison can tell two frames apart.
                        var host = List[UInt8](length=N_CAMS * CAM_ELEMS, fill=UInt8(0))
                        for slot in range(N_CAMS):
                            render_lane_cpu[DT, DynDims, 1, False, True, SAMPLES](
                                dh, mh, vis_h, cams[slot], 0, CAM_W, CAM_H, bg_h,
                                rgb, depth, seg, refl,
                            )
                            for q in range(NPIX):
                                for c in range(3):
                                    host[slot * CAM_ELEMS + c * NPIX + q] = _to_byte(
                                        Float64(rgb[q * 3 + c])
                                    )
                        for pair in range(N_CAMS + 1):
                            var ds_ = pair if pair < N_CAMS else 0   # device slot
                            var hs_ = pair if pair < N_CAMS else 1   # host slot
                            var se = 0.0
                            for k in range(CAM_ELEMS):
                                var diff = Float64(Int(host[hs_ * CAM_ELEMS + k])) - Float64(
                                    Int(first_frames[ds_ * CAM_ELEMS + k])
                                )
                                se += diff * diff
                            var mse = se / Float64(CAM_ELEMS)
                            var psnr = 99.0 if mse <= 0.0 else 10.0 * log10(255.0 * 255.0 / mse)
                            if pair < N_CAMS:
                                var name = String("overhead") if pair == 0 else String("wrist")
                                print("  host check, row 0,", name, ":", psnr, "dB (device vs host)")
                                if psnr < worst:
                                    worst = psnr
                                var hwc = List[UInt8](length=CAM_ELEMS, fill=UInt8(0))
                                for q in range(NPIX):
                                    for c in range(3):
                                        hwc[q * 3 + c] = first_frames[pair * CAM_ELEMS + c * NPIX + q]
                                save_png(out_path + "." + name + ".row0.png", hwc, CAM_W, CAM_H, 3)
                            else:
                                swapped = psnr
                                print("  negative control, device overhead vs host wrist:",
                                      psnr, "dB (must be low)")

                        if swapped >= MAX_PSNR_SWAPPED:
                            raise Error(
                                "tower rerender: the negative control scored "
                                + String(swapped) + " dB — the check cannot"
                                " tell the two cameras apart, so its match"
                                " proves nothing"
                            )
                        if worst < MIN_PSNR_HOST:
                            raise Error(
                                "tower rerender: device vs host " + String(worst)
                                + " dB < " + String(MIN_PSNR_HOST) + " — the"
                                " pose, the camera or the group mask differs"
                                " between the two paths; the store is not"
                                " trusted"
                            )
                var tio = perf_counter_ns()
                w.append[DType.float32](String("qpos"), qb, n)
                w.append[DType.float32](String("action"), ab, n)
                w.append[DType.uint8](String("images"), im, n)
                w.append[DType.float64](String("state"), sb, n)
                w.append[DType.float32](String("action_sim"), asb, n)
                if dr_on:
                    w.append[DType.int32](String("dr_draw"), drb, n)
                t_io += perf_counter_ns() - tio
                done += n
                launch += 1
            w.end_episode()
            total_eps += 1
            total_rows += T
            if total_eps % 10 == 0:
                var secs = Float64(perf_counter_ns() - t0) / 1e9
                print("  ", total_eps, "episodes |", total_rows, "rows |",
                      Int(Float64(total_rows) / secs), "rows/s")
        if max_eps > 0 and total_eps >= max_eps:
            break
    w.close()
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    if total_rows == 0:
        raise Error("tower rerender: no rows written (no successful episode?)")

    print("-" * 78)
    print("  wrote", out_path, "|", total_eps, "episodes |", total_rows, "rows |",
          getsize(out_path) // (1024 * 1024), "MiB")
    print("  time   :", secs, "s |", Float64(total_rows) / secs, "rows/s | fk",
          Float64(t_fk) / 1e9, "s, render", Float64(t_render) / 1e9, "s, io",
          Float64(t_io) / 1e9, "s")
    if constant_pictures > 0:
        raise Error(
            "tower rerender: " + String(constant_pictures) + " frames are one"
            " flat colour — a camera inside a mesh or a pose that never"
            " reached the device"
        )
    print("=== OK ===")
