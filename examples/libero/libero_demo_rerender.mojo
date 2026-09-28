"""LIBERO'S DEMONSTRATIONS, RE-RENDERED THROUGH OUR CAMERAS — the image store a
policy evaluated in OUR sim should train on. L7a.

    pixi run -e apple  libero-demo-rerender --demos 2            # a Mac smoke run
    pixi run -e nvidia libero-demo-rerender                      # the whole suite
    pixi run -e nvidia libero-demo-rerender --tasks turn_on_the_stove --out /tmp/stove.h5

Reads the low-dim demo store (`build/demos/<suite>.lowdim.h5`: `state` in OUR
joint order, `action`, `task_index`, one episode per demonstration), puts every
`state` row into the batched `Data` the eval env uses, runs forward kinematics
on the host, and renders BOTH of LIBERO's cameras (`agentview`, `eye_in_hand`)
with the device tracer at LIBERO's own settings — 128x128, 4x MSAA, the visual
geom group, the stove burner rule, the demo's own fixture draw. It writes

    build/demos/<suite>.rendered.h5
        action      (7)   f32    the recorded OSC_POSE words, as before
        state       (nq+nv) f64  as before
        qpos        (18+T) f32   the 7 arm joints + 2 finger joints OF `state`,
                                 their one-step difference (zero on a demo's
                                 first row), then a one-hot of the task over
                                 the suite's T
        task_index  (1)   i32
        images      (2 x 3 x 128 x 128) u8  OURS, CHW, top row first
        psnr        (2)   f32    ours vs the recording, per camera, per row
                                 (-1 where the recording has no frame)

## ⚠⚠ WHY A SECOND IMAGE STORE EXISTS AT ALL — THE PIXEL DOMAIN

`libero-demo-import` writes the RECORDED frames: robosuite 1.4's OpenGL, whose
pixel domain our tracer reaches to 32 dB (`libero-camera-gate`), not to
identity. A policy trained on those frames and evaluated on ours crosses that
gap at test time, and the success rate then measures the gap as much as the
policy. This store closes it: every training frame is the frame the eval env
produces for that state, from the same kernel. Training on the recorded store
and evaluating here remains a legitimate ARM — it is how the gap is priced in
success points rather than in dB — and both stores carry the same `action`,
`state` and `qpos`, so the two arms differ in pixels only.

## ⚠⚠ ROW `r` IS THE PICTURE OF `state[r]`, PAIRED WITH `action[r]`

LIBERO's own dataset pairs `obs[j]` — the frame AFTER `env.step(actions[j])`,
i.e. `states[j + 1]` — with `actions[j]` (`scripts/create_dataset.py:175-221`),
so their policies learn to emit action `j` from the picture action `j`
produced. A closed-loop policy in our sim sees the picture of the state it is
IN and must emit the action to take FROM it; that is the pairing written here,
and `qpos` is taken from the same `state[r]` (the recorded `joint_states[r]`
is, like the frame, the post-step value and is deliberately NOT used). The
recorded store keeps LIBERO's pairing because it is LIBERO's data.

## ⚠⚠ THE ALIGNMENT IS ASSERTED, NOT ASSUMED

For the first demonstration of every task the rows are scored against three
readings of the recording — `obs[r-1]` flipped (the correct one: the frame of
`state[r]`, stored bottom row first), `obs[r]` flipped (one step late) and
`obs[r-1]` as stored (upside down) — and the run refuses unless the first wins
on both cameras. `libero_camera_gate.mojo` measured the three at 41.8 / 35.0 /
11.5 dB; a tool that silently paired frames one step off would train a policy
on a 50 ms lag and report a fine loss.

## ⚠ THE FIXTURES ARE THE DEMO'S, FROM THE DEMO-SUCCESS DUMP

`envs/libero/fixtures.mojo`: worth ~7 dB, read from
`references/libero_demos/_dumps/<suite>/<task>.dump` when it exists. A missing
dump is WARNED per task, not fatal — `libero-demo-dump` writes it.

## ⚠ HOST FK, DEVICE PIXELS — AND THE THROUGHPUT LINE SAYS WHICH IS WHICH

Forward kinematics runs on the host (`libero_bc_train` measured ~1 min for
63 728 rows) because nothing here steps physics; the tracer runs on the device
over `LANES` rows at a time and is timed separately, so the number printed for
the tracer is the tracer's. It is also the first measurement of the camera
kernel on a LIBERO scene (240 geoms, 126 598 Panda triangles + the objects),
which the closed-loop eval pays once per camera per control step.

⚠ NVIDIA OR METAL. The camera kernel has no per-thread array sized by `nv`, so
unlike the physics kernels it builds on Metal at nv=37; the Mac smoke run above
is real pixels. The lane count is a comptime constant (`LANES`).
"""

from std.sys import argv, has_accelerator
from std.math import log10
from std.memory.alloc import unsafe_alloc
from std.os import listdir, makedirs
from std.os.path import exists, dirname
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.hdf5.reader import H5File
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStore, TrajectoryStoreWriter
from noeira.envs.libero.demos import (
    CAM_H, CAM_W, N_CAMS, CAM_ELEMS, ACTION_DIM, QPOS_PROPRIO, QPOS_WORDS,
    COL_ACTION, COL_STATE, COL_QPOS, COL_TASK, COL_IMAGES,
)
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.physics3d.parser.fields_build import build_model_fields_from_flat
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.raytrace import BatchedCameraRenderer, RGB_CHANNELS
from noeira.physics3d.raytrace.visual import build_visual_model
from noeira.tasks.spec import load_family, load_task
from noeira.tasks.family import scene_path
from noeira.tasks.predicates import joint_qpos_addresses
from noeira.envs.libero.models.libero_goal_xml import LiberoGoalModel
from noeira.envs.libero.osc_config import LiberoOscConfig
from noeira.envs.libero.placement.libero_goal import LiberoGoalPlacement
from noeira.envs.libero.visual import libero_site_conditions
from noeira.envs.libero.fixtures import patch_fixtures, fixtures_dump_path


comptime DT = DType.float32
"""The device's type — the eval env's `Data` is float32 too."""
comptime SUITE = "libero_goal"
comptime FAMILY_DIR = "noeira/envs/libero/families/"
comptime TASK_DIR = "noeira/envs/libero/tasks/"
comptime DEMO_DIR = "references/libero_demos/"
comptime LANES = 64
"""Rows rendered per launch. Comptime: the kernel is instantiated per value."""
comptime MD = ModelDims[
    LiberoGoalModel,
    nmesh_verts = LiberoOscConfig[LiberoGoalPlacement].NMESH_VERTS,
]
"""⚠ THE EVAL ENV'S OWN `Data` SHAPE, so the renderer alias below is the one
the closed-loop driver instantiates over `env.d` — one kernel, two callers."""
comptime NQ = MD.NQ
comptime NV = MD.NV
comptime STATE_DIM = NQ + NV
comptime VISUAL_GROUP_MASK: Int = 1 << 1
"""Group 1 only: robosuite renders with `render_collision_mesh=False`."""
comptime SAMPLES = 4
"""MuJoCo's `offsamples`; one ray per pixel scores 14 dB lower on the gate."""
comptime Renderer = BatchedCameraRenderer[
    DT, MD, LANES, CAM_W, CAM_H, False, True, SAMPLES
]
"""No shadow ray (the camera gate's setting), the reflection pass on."""
comptime NPIX = CAM_W * CAM_H
comptime PSNR_COL: StaticString = "psnr"
comptime MIN_PSNR_AGENTVIEW = 24.0
comptime MIN_PSNR_EYE_IN_HAND = 22.0
"""The refusal lines. `libero-camera-video` measured demo 0 of two tasks at
32.0 (agentview) and 29.7 dB (eye_in_hand) with the fixture dump; a suite-wide
mean below these is a misalignment or a missing asset, not shading."""
comptime ALIGN_ROWS = 6
"""Rows per first-demo alignment check, from row 1 (row 0 has no frame)."""


def _to_byte(x: Float64) -> UInt8:
    var v = Int(x * 255.0 + 0.5)
    if v < 0:
        v = 0
    if v > 255:
        v = 255
    return UInt8(v)


def _f(x: Float64, n: Int) -> String:
    """`x` printed and cut at `n` bytes — a table cell, not a number."""
    var out = String(x)
    if out.byte_length() > n:
        return String(out[byte = 0 : n])
    return out^


def _pad(s: String, n: Int) -> String:
    var out = String(s)
    if out.byte_length() > n:
        return String(out[byte = 0 : n])
    while out.byte_length() < n:
        out += " "
    return out^


def _task_stems(suite: String) raises -> List[String]:
    """`libero_demo_import.task_stems`'s order — `task_index` indexes it."""
    var out = List[String]()
    var want = suite + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = want.byte_length() : n.byte_length() - 5]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("libero rerender: no '" + want + "' in the scene")


def _psnr_chw_vs_hwc(
    ours: Pointer[Scalar[DType.uint8], MutAnyOrigin], ours_off: Int,
    rec: Pointer[Scalar[DType.uint8], MutAnyOrigin], rec_off: Int,
    flip: Bool,
) -> Float64:
    """Ours (CHW, top row first) against one recorded frame (HWC); `flip`
    reads the recording bottom row first, which is how LIBERO stored it."""
    var se = 0.0
    for y in range(CAM_H):
        var ry = CAM_H - 1 - y if flip else y
        for x in range(CAM_W):
            for c in range(3):
                var a = Float64(Int(ours[unsafe_offset = ours_off + c * NPIX + y * CAM_W + x]))
                var b = Float64(Int(rec[unsafe_offset = rec_off + (ry * CAM_W + x) * 3 + c]))
                se += (a - b) * (a - b)
    var mse = se / Float64(NPIX * 3)
    return 99.0 if mse <= 0.0 else 10.0 * log10(255.0 * 255.0 / mse)


def _usage() -> String:
    return String(
        "usage: libero_demo_rerender.mojo [--store F] [--out F] [--demos N]"
        " [--tasks a,b] [--demos-dir DIR] [--no-compare]"
    )


def main() raises:
    comptime if not has_accelerator():
        print("  SKIPPED: no accelerator — the tracer is a device kernel")
        print("=== SKIPPED (this is not a pass) ===")
        return

    var args = argv()
    var suite = String(SUITE)
    var store_path = String("build/demos/") + suite + ".lowdim.h5"
    var out_path = String("build/demos/") + suite + ".rendered.h5"
    var demos_dir = String(DEMO_DIR)
    var max_demos = 0
    var only_tasks = List[String]()
    var compare = True
    var no_dq = False  # --no-dq: the ablation store, difference words ZERO
    var i = 1
    while i < len(args):
        var a = String(args[i])
        var has_val = i + 1 < len(args)
        if a == "--no-dq":
            no_dq = True
        elif a == "--no-compare":
            compare = False
        elif a.startswith("--"):
            if not has_val:
                raise Error(a + " takes a value\n" + _usage())
            var v = String(args[i + 1])
            if a == "--store":
                store_path = v
            elif a == "--out":
                out_path = v
            elif a == "--demos":
                max_demos = Int(v)
            elif a == "--demos-dir":
                demos_dir = v
            elif a == "--tasks":
                for t in v.split(","):
                    only_tasks.append(String(t))
            else:
                raise Error("unknown option " + a + "\n" + _usage())
            i += 1
        else:
            raise Error("unexpected argument " + a + "\n" + _usage())
        i += 1

    print("=" * 78)
    print("LIBERO demonstrations re-rendered through our cameras —", suite)
    print("=" * 78)
    if not exists(store_path):
        print("  SKIPPED: no low-dim store at", store_path)
        print("  Build it:  pixi run libero-demo-import --no-images")
        print("=== SKIPPED (no demonstrations — this is not a pass) ===")
        return

    # ── the scene, built the way the eval env builds it ───────────────────
    var fam = load_family(String(FAMILY_DIR) + suite + ".family")
    var fmd = parse_model_runtime(scene_path(fam))
    var ctx = DeviceContext()
    var m = Model[DT, MD]()
    build_model_fields_from_flat[DT](fmd, m)
    m.upload_all(ctx)
    var d = Data[DT, MD, LANES]()
    d.upload_all(ctx)
    ctx.synchronize()
    var cam_av = _index(fmd.camera_names, String("arena_agentview"))
    var cam_eih = _index(fmd.camera_names, String("robot_eye_in_hand"))
    var r = Renderer(ctx, m, cam_av)
    r.set_visual(
        ctx,
        build_visual_model[DT, MD](
            fmd, m, group_mask=VISUAL_GROUP_MASK,
            conditions=libero_site_conditions(fam),
        ),
    )
    print("  device :", ctx.name(), "|", LANES, "lanes |", CAM_W, "x", CAM_H,
          "|", SAMPLES, "samples | cameras", cam_av, "(agentview)", cam_eih,
          "(eye_in_hand)")
    print("  " + r.vis.describe())

    # the nine proprio words, by NAME
    var nqs = List[Int]()
    for k in range(len(fmd.joints)):
        nqs.append(fmd.joints[k].nq)
    var jadr = joint_qpos_addresses(nqs)
    var qadr = List[Int]()
    for j in range(7):
        qadr.append(jadr[_index(fmd.joint_names, String("robot_joint") + String(j + 1))])
    qadr.append(jadr[_index(fmd.joint_names, String("robot_finger_joint1"))])
    qadr.append(jadr[_index(fmd.joint_names, String("robot_finger_joint2"))])

    # ── the store in ──────────────────────────────────────────────────────
    var st = TrajectoryStore(store_path)
    var n_rows_all = st.n_rows()
    var n_eps = st.n_episodes()
    var state_col = st.load_column[DType.float64](String(COL_STATE))
    var act_col = st.load_column[DType.float32](String(COL_ACTION))
    var task_col = st.load_column[DType.int32](String(COL_TASK))
    if len(state_col) != n_rows_all * STATE_DIM:
        raise Error(
            "libero rerender: the state column is " + String(len(state_col))
            + " words for " + String(n_rows_all) + " rows; this build's scene"
            " has nq + nv = " + String(STATE_DIM)
        )
    var stems = _task_stems(suite)
    print("  store  :", store_path, "|", n_rows_all, "rows |", n_eps,
          "episodes |", len(stems), "tasks")

    # episode -> (task, demo index within the task), in the importer's order
    var ep_task = List[Int]()
    var ep_demo = List[Int]()
    var seen = List[Int](length=len(stems), fill=0)
    for e in range(n_eps):
        var ti = Int(task_col[st.episodes.start_of(e)])
        if ti < 0 or ti >= len(stems):
            raise Error("episode " + String(e) + " names task " + String(ti)
                        + ", the suite has " + String(len(stems)))
        ep_task.append(ti)
        ep_demo.append(seen[ti])
        seen[ti] += 1
    var use_task = List[Bool](length=len(stems), fill=len(only_tasks) == 0)
    for k in range(len(only_tasks)):
        var hit = False
        for ti in range(len(stems)):
            if stems[ti] == only_tasks[k]:
                use_task[ti] = True
                hit = True
        if not hit:
            raise Error("--tasks names '" + only_tasks[k] + "', not a "
                        + suite + " task")

    # ── the store out ─────────────────────────────────────────────────────
    # ⚠ THE TASK RIDES IN `qpos` AS A ONE-HOT — the picture cannot carry it,
    # every libero_goal task being the same scene (`libero_demos.mojo`).
    var QPOS_DIM = QPOS_WORDS + len(stems)
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String(COL_ACTION), DType.float32, ACTION_DIM))
    cols.append(ColumnSpec(String(COL_STATE), DType.float64, STATE_DIM))
    cols.append(ColumnSpec(String(COL_QPOS), DType.float32, QPOS_DIM))
    cols.append(ColumnSpec(String(COL_TASK), DType.int32, 1))
    cols.append(ColumnSpec(String(COL_IMAGES), DType.uint8, N_CAMS * CAM_ELEMS))
    cols.append(ColumnSpec(String(PSNR_COL), DType.float32, N_CAMS))
    var dd = dirname(out_path)
    if dd.byte_length() > 0:
        makedirs(dd, exist_ok=True)
    var w = TrajectoryStoreWriter(
        String(out_path), cols^,
        env_id=String("libero_rerender:") + suite,
        seed=0,
        source_commit=String("LIBERO-v1 demonstrations, states remapped by ")
            + "noeira/envs/libero/tables/state_remap_" + suite + ".kv, images by"
            " noeira/physics3d/raytrace (batch.mojo) at " + String(CAM_W)
            + "x" + String(CAM_H) + " " + String(SAMPLES) + "x MSAA, group 1,"
            " row 0 = top, frame r = state r",
    )
    for ti in range(len(stems)):
        var t = load_task(String(TASK_DIR) + suite + "__" + stems[ti] + ".task")
        w.add_task(ti, String(t.language))

    # ── buffers ───────────────────────────────────────────────────────────
    var h_rgb = ctx.enqueue_create_host_buffer[DT](LANES * NPIX * RGB_CHANNELS)
    var im = unsafe_alloc[Scalar[DType.uint8]](
        LANES * N_CAMS * CAM_ELEMS
    ).as_unsafe_any_origin()
    var ab = unsafe_alloc[Scalar[DType.float32]](LANES * ACTION_DIM).as_unsafe_any_origin()
    var sb = unsafe_alloc[Scalar[DType.float64]](LANES * STATE_DIM).as_unsafe_any_origin()
    var qb = unsafe_alloc[Scalar[DType.float32]](LANES * QPOS_DIM).as_unsafe_any_origin()
    var tb = unsafe_alloc[Scalar[DType.int32]](LANES).as_unsafe_any_origin()
    var pb = unsafe_alloc[Scalar[DType.float32]](LANES * N_CAMS).as_unsafe_any_origin()
    var rec = List[Pointer[Scalar[DType.uint8], MutAnyOrigin]]()
    var rec_cap = 0

    # per-task accounting
    var task_rows = List[Int](length=len(stems), fill=0)
    var task_eps = List[Int](length=len(stems), fill=0)
    var task_psnr = List[Float64](length=len(stems) * N_CAMS, fill=0.0)
    var task_scored = List[Int](length=len(stems), fill=0)
    # -1 = no dump; else fixtures placed on the LAST demo (all demos alike)
    var task_fix = List[Int](length=len(stems), fill=-1)
    # alignment: per task, per camera, the three candidates summed over rows
    var align = List[Float64](length=len(stems) * N_CAMS * 3, fill=0.0)
    var align_n = List[Int](length=len(stems), fill=0)
    var constant_pictures = 0
    var t_fk = 0
    var t_render = 0
    var t_io = 0
    var rendered_frames = 0
    var total_rows = 0
    var total_eps = 0
    var t0 = perf_counter_ns()

    for e in range(n_eps):
        var ti = ep_task[e]
        if not use_task[ti]:
            continue
        var di = ep_demo[e]
        if max_demos > 0 and di >= max_demos:
            continue
        var off = st.episodes.start_of(e)
        var T = st.episodes.length_of(e)

        # the demo's fixture draw, into the host body records FK reads
        var dump = fixtures_dump_path(demos_dir, suite, stems[ti])
        if exists(dump):
            task_fix[ti] = patch_fixtures[DT, MD](dump, di, fmd.body_names, m)
        # the recording, for the comparison
        var have_rec = False
        if compare:
            var tio = perf_counter_ns()
            var h5 = demos_dir + suite + "/" + stems[ti] + "_demo.hdf5"
            if exists(h5):
                var f5 = H5File(h5)
                var base = String("data/demo_") + String(di) + "/obs/"
                if len(rec) == 0:
                    rec.append(unsafe_alloc[Scalar[DType.uint8]](1).as_unsafe_any_origin())
                    rec.append(unsafe_alloc[Scalar[DType.uint8]](1).as_unsafe_any_origin())
                if T * CAM_ELEMS > rec_cap:
                    rec[0].unsafe_free()
                    rec[1].unsafe_free()
                    rec_cap = T * CAM_ELEMS
                    rec[0] = unsafe_alloc[Scalar[DType.uint8]](rec_cap).as_unsafe_any_origin()
                    rec[1] = unsafe_alloc[Scalar[DType.uint8]](rec_cap).as_unsafe_any_origin()
                for cam in range(N_CAMS):
                    var key = base + ("agentview_rgb" if cam == 0 else "eye_in_hand_rgb")
                    var ds = f5.open_dataset(key)
                    if (ds.ndim() != 4 or Int(ds.dims[0]) != T
                            or Int(ds.dims[1]) != CAM_H or Int(ds.dims[2]) != CAM_W):
                        raise Error(h5 + ": " + key + " is not (" + String(T)
                                    + ", " + String(CAM_H) + ", " + String(CAM_W)
                                    + ", 3) — the store and the recording disagree")
                    ds.read_all[DType.uint8](rec[cam])
                have_rec = True
            t_io += perf_counter_ns() - tio

        var done = 0
        while done < T:
            var n = T - done
            if n > LANES:
                n = LANES
            # ── host: state -> qpos -> FK ────────────────────────────────
            var tf = perf_counter_ns()
            for l in range(LANES):
                var rr = off + done + (l if l < n else 0)
                for k in range(NQ):
                    d.qpos.data[l * NQ + k] = Scalar[DT](state_col[rr * STATE_DIM + k])
            forward_kinematics["cpu", DT, MD, LANES](d, m)
            d.qpos.upload_resident(ctx)
            d.xpos.upload_resident(ctx)
            d.xquat.upload_resident(ctx)
            t_fk += perf_counter_ns() - tf
            # ── device: both cameras ─────────────────────────────────────
            var tr = perf_counter_ns()
            for cam in range(N_CAMS):
                r.render(ctx, d, m, cam_av if cam == 0 else cam_eih)
                ctx.enqueue_copy(h_rgb, r.rgb)
                ctx.synchronize()
                var p = h_rgb.unsafe_ptr()
                for l in range(n):
                    var dst = l * N_CAMS * CAM_ELEMS + cam * CAM_ELEMS
                    var src = l * NPIX * RGB_CHANNELS
                    var first = _to_byte(Float64(p[unsafe_offset=src]))
                    var all_same = True
                    for q in range(NPIX):
                        for c in range(3):
                            var b = _to_byte(Float64(p[unsafe_offset = src + q * 3 + c]))
                            im[unsafe_offset = dst + c * NPIX + q] = b
                            if b != first:
                                all_same = False
                    if all_same:
                        constant_pictures += 1
            rendered_frames += n * N_CAMS
            t_render += perf_counter_ns() - tr
            # ── the rows ─────────────────────────────────────────────────
            for l in range(n):
                var rr = off + done + l
                var t = done + l
                for k in range(ACTION_DIM):
                    ab[unsafe_offset = l * ACTION_DIM + k] = act_col[rr * ACTION_DIM + k]
                for k in range(STATE_DIM):
                    sb[unsafe_offset = l * STATE_DIM + k] = state_col[rr * STATE_DIM + k]
                for k in range(QPOS_PROPRIO):
                    var cur = Scalar[DType.float32](state_col[rr * STATE_DIM + qadr[k]])
                    var prev = (
                        Scalar[DType.float32](state_col[(rr - 1) * STATE_DIM + qadr[k]])
                        if t > 0 else cur
                    )
                    qb[unsafe_offset = l * QPOS_DIM + k] = cur
                    qb[unsafe_offset = l * QPOS_DIM + QPOS_PROPRIO + k] = (
                        Scalar[DType.float32](0) if no_dq else cur - prev
                    )
                for k in range(len(stems)):
                    qb[unsafe_offset = l * QPOS_DIM + QPOS_WORDS + k] = Scalar[
                        DType.float32
                    ](1.0 if k == ti else 0.0)
                tb[unsafe_offset=l] = Int32(ti)
                for cam in range(N_CAMS):
                    var ps = -1.0
                    # ⚠ the recorded frame of `state[t]` is `obs[t - 1]`
                    if have_rec and t >= 1:
                        ps = _psnr_chw_vs_hwc(
                            im, l * N_CAMS * CAM_ELEMS + cam * CAM_ELEMS,
                            rec[cam], (t - 1) * CAM_ELEMS, True,
                        )
                        task_psnr[ti * N_CAMS + cam] += ps
                        if cam == N_CAMS - 1:
                            task_scored[ti] += 1
                        # the alignment check, first demo of the task
                        if di == 0 and t <= ALIGN_ROWS and t < T - 1:
                            var b0 = (ti * N_CAMS + cam) * 3
                            align[b0 + 0] += ps
                            align[b0 + 1] += _psnr_chw_vs_hwc(
                                im, l * N_CAMS * CAM_ELEMS + cam * CAM_ELEMS,
                                rec[cam], t * CAM_ELEMS, True,
                            )
                            align[b0 + 2] += _psnr_chw_vs_hwc(
                                im, l * N_CAMS * CAM_ELEMS + cam * CAM_ELEMS,
                                rec[cam], (t - 1) * CAM_ELEMS, False,
                            )
                            if cam == N_CAMS - 1:
                                align_n[ti] += 1
                    pb[unsafe_offset = l * N_CAMS + cam] = Scalar[DType.float32](ps)
            var tw = perf_counter_ns()
            w.append[DType.float32](String(COL_ACTION), ab, n)
            w.append[DType.float64](String(COL_STATE), sb, n)
            w.append[DType.float32](String(COL_QPOS), qb, n)
            w.append[DType.int32](String(COL_TASK), tb, n)
            w.append[DType.uint8](String(COL_IMAGES), im, n)
            w.append[DType.float32](String(PSNR_COL), pb, n)
            t_io += perf_counter_ns() - tw
            done += n
        w.end_episode()
        task_rows[ti] += T
        task_eps[ti] += 1
        total_rows += T
        total_eps += 1
        if di == 0 or (di + 1) % 10 == 0:
            print("  " + _pad(stems[ti], 52) + " demo " + String(di) + "  "
                  + String(T) + " rows  ("
                  + String(Float64(perf_counter_ns() - t0) / 1e9) + " s)",
                  flush=True)
    w.close()
    var elapsed = Float64(perf_counter_ns() - t0) / 1e9

    # ── the report ────────────────────────────────────────────────────────
    print()
    print("  wrote", out_path, "—", total_eps, "episodes,", total_rows,
          "rows, frames of state r paired with action r")
    print("  host FK+upload", Float64(t_fk) / 1e9, "s | tracer",
          Float64(t_render) / 1e9, "s for", rendered_frames, "frames =",
          Float64(rendered_frames) / (Float64(t_render) / 1e9) if t_render > 0
          else 0.0, "frames/s (both cameras, incl. the copy back) | hdf5",
          Float64(t_io) / 1e9, "s | wall", elapsed, "s")
    print()
    print("  " + _pad("task", 52) + " demos   rows   fixtures   agentview  eye_in_hand")
    var sum_av = 0.0
    var sum_eih = 0.0
    var n_scored = 0
    var align_bad = List[String]()
    var align_checked = 0
    for ti in range(len(stems)):
        if task_eps[ti] == 0:
            continue
        var av = task_psnr[ti * N_CAMS] / Float64(task_scored[ti]) if task_scored[ti] > 0 else -1.0
        var eih = task_psnr[ti * N_CAMS + 1] / Float64(task_scored[ti]) if task_scored[ti] > 0 else -1.0
        sum_av += task_psnr[ti * N_CAMS]
        sum_eih += task_psnr[ti * N_CAMS + 1]
        n_scored += task_scored[ti]
        var fx = String("no dump") if task_fix[ti] < 0 else String(task_fix[ti])
        print("  " + _pad(stems[ti], 52) + " " + _pad(String(task_eps[ti]), 7)
              + " " + _pad(String(task_rows[ti]), 6) + " " + _pad(fx, 10)
              + " " + _pad(_f(av, 6), 10) + " " + _f(eih, 6))
        if align_n[ti] > 0:
            align_checked += 1
            for cam in range(N_CAMS):
                var b0 = (ti * N_CAMS + cam) * 3
                var a = align[b0] / Float64(align_n[ti])
                var late = align[b0 + 1] / Float64(align_n[ti])
                var upside = align[b0 + 2] / Float64(align_n[ti])
                if not (a > late and a > upside):
                    align_bad.append(
                        stems[ti] + " cam " + String(cam) + ": state r vs obs[r-1]"
                        " flipped " + String(a) + " dB, obs[r] flipped "
                        + String(late) + ", obs[r-1] unflipped " + String(upside)
                    )
    var mean_av = sum_av / Float64(n_scored) if n_scored > 0 else -1.0
    var mean_eih = sum_eih / Float64(n_scored) if n_scored > 0 else -1.0
    print()
    print("  mean PSNR ours vs the recording over", n_scored, "rows: agentview",
          mean_av, "dB | eye_in_hand", mean_eih, "dB")
    print("  alignment (first demo of", align_checked, "tasks, rows 1..",
          ALIGN_ROWS, "): state r == obs[r-1] flipped wins on",
          align_checked * N_CAMS - len(align_bad), "of", align_checked * N_CAMS,
          "camera-tasks")
    for k in range(len(align_bad)):
        print("    ✗", align_bad[k])

    # ── the verdict ───────────────────────────────────────────────────────
    var fails = List[String]()
    if total_rows == 0:
        fails.append("no rows rendered")
    if constant_pictures > 0:
        fails.append(String(constant_pictures) + " frames are a single colour")
    if compare:
        if n_scored == 0:
            fails.append("no row was compared with the recording (are the"
                         " demos under " + demos_dir + suite + "?)")
        else:
            if mean_av < MIN_PSNR_AGENTVIEW:
                fails.append("agentview " + String(mean_av) + " dB < "
                             + String(MIN_PSNR_AGENTVIEW))
            if mean_eih < MIN_PSNR_EYE_IN_HAND:
                fails.append("eye_in_hand " + String(mean_eih) + " dB < "
                             + String(MIN_PSNR_EYE_IN_HAND))
        if align_checked == 0:
            fails.append("the alignment was never checked")
        if len(align_bad) > 0:
            fails.append(String(len(align_bad)) + " camera-task(s) where"
                         " state r is not best matched by obs[r-1] flipped")
    print()
    if len(fails) > 0:
        for k in range(len(fails)):
            print("  FAIL:", fails[k])
        raise Error("libero rerender: " + String(len(fails)) + " check(s) failed")
    print("=== PASS —", total_rows, "rows of", suite, "rendered through our"
          " cameras" + (", " + _f(mean_av, 5) + " / "
          + _f(mean_eih, 5) + " dB vs the recording" if compare
          else " (not compared)"), "===")
