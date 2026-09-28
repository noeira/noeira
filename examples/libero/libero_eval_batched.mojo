"""LIBERO'S OWN EVAL PROTOCOL, ON THE BATCH — the benchmark number's driver.

    pixi run -e nvidia mojo run -I . examples/libero/libero_eval_batched.mojo
    pixi run -e nvidia mojo run -I . examples/libero/libero_eval_batched.mojo --inits 20 --steps 600
    pixi run -e apple  mojo run -I . examples/libero/libero_eval_batched.mojo --steps 40   # a small family
    pixi run -e nvidia mojo run -I . examples/libero/libero_eval_batched.mojo --policy <run_id>   # a libero-bc-train run

`examples/libero/libero_eval.mojo` runs `lifelong/metric.py`'s loop one env at a
time on the CPU. This is the same loop on the batched GPU env, lane per
episode:

    reset_batch                  -> the family's base pose; OSC anchored THERE
    write the frozen row         -> qpos/qvel wholesale, controller NOT re-anchored
    5 settle steps of zeros(7)   -> the props land (LIBERO's own drop)
    up to `--steps` policy steps -> success at ANY step, per lane
    dones[k] |= done[k]

## ⚠⚠ THE ANCHOR, AND WHY THE ORDER IS THE PROTOCOL

robosuite builds the OSC controller in `Robot.reset`, so its nullspace target
`q0` is the configuration AT CONSTRUCTION — the family's rest pose — and
`set_init_state` overwrites `qpos` without rebuilding it. `reset_batch`
anchors after its own reset (`_osc_anchor`, the rest pose), and this driver
then writes the row and does NOT re-anchor. That is deliberate:
`libero_eval.mojo`'s header measured the difference (row 0 of
`open_the_middle_drawer` starts at joint2 = -0.188 against a rest pose of
-0.161), and `libero_demo_batched` takes the OTHER order on purpose because a
demo replay is not an eval.

## ⚠⚠ WHAT MAKES A NUMBER COMPARABLE, AND WHAT DOES NOT

* **The frozen rows.** `build/init/<family>.init.h5`, LIBERO's `.pruned_init`
  through `libero-init-dump` + `libero-init-freeze`. A success rate over states
  the run sampled for itself is not comparable with anything, so the frozen
  table is the only mode that prints a BENCHMARK number. It exists for
  `libero_goal` today.
* ⚠ **`--sampled` IS A SMOKE MODE, NOT A BENCHMARK.** Without a table (22
  families) the lanes are reset by the DEVICE sampler and the header says so
  on every line of the report. It exercises the protocol, the tape, the mask
  and the success word; it cannot be quoted against LIBERO's tables.
* **The mask comes from the TABLE, the goal from the `.task`.** `init_table`'s
  header explains: an eval months later must not change its answer because
  someone edited an `active=` line, while the GOAL is the benchmark's
  definition and belongs under version control.

## ⚠ THE POLICY IS `zeros(7)` UNTIL ONE EXISTS IN THE TREE

That is `metric.py`'s own `dummy`, so this is a real instance of the protocol
rather than a stand-in — and the L6 gate is that the rate is 0: a task solved
by the null action is a goal defect, not a policy result. `_policy_action` is
the one place a policy plugs in; it reads the env's observation rows, which
`task_hooks.write_task_obs` has already written for every lane.

## `--act DIR` — THE IMAGE POLICY, WITH THE CAMERAS IN THE LOOP (L7c)

`DIR` is a `libero-act-train` checkpoint directory (`best.ckpt` + `norm.json`);
`--act-ckpt last` evaluates `last.ckpt` instead — the recipe's checkpoint when
the validation minimum is the noise floor (see `ACT_PATIENCE` in the trainer).
`--act-latent sample` draws `z ~ N(0, I)` at every query instead of the
paper's `z = 0`: a commitment to one style per query where the prior mean is
the flat conditional median. Only a fit whose latent stayed informative
(`ACT_KL` below 10) can differ here. `--act-m M` sets the temporal ensemble's weight (default `ACT_TEMPORAL_ENSEMBLE_M`
= 0.01, the paper's): `w = exp(-M * rank)` with rank 0 the OLDEST query in the
window, so the paper's positive M favours stale predictions (0.01 over 40
chunks: 1.0 down to 0.68, near-uniform over two seconds); a NEGATIVE M favours
the newest. The long-schedule drawer fit (`e01d6782`, 2026-09-20) reached the
demonstrators' action scale and still missed the handle by 4-5 cm on the
approach under the default ensemble, while exec 10 on the multi-task fit hit
it — the averaging of two seconds of predictions across a fast sideways move
is the suspect, and this is the knob that tests it without a refit.
Every policy step then renders BOTH of LIBERO's cameras for every lane with the
batched tracer — `raytrace/batch.mojo` over `env.d`, 128x128, 4x MSAA, the
visual group, after the step's FK sync — packs the pixels the way the store
holds them (uint8 CHW, top row first) and normalises them with the SAME
`normalize_camera_chw` the trainer's sampler used, reads the nine proprio words
from `qpos` and standardises them with the checkpoint's own `norm.json`, runs
one forward at batch `LANES`, pushes each lane's chunk into its own
`TemporalEnsemble`, and writes the ensembled, denormalised, clamped action.

⚠ THE OBSERVATION IS THE STATE THE LANE IS IN. `SYNC_FK_AFTER_STEP` is on for
the LIBERO config, so `xpos`/`xquat` on the device describe the integrated
`qpos` when the camera reads them; a stale FK here would be a one-step lag no
picture would show. The rendered store pairs frame `r` with `action[r]` for the
same reason (`libero_demo_rerender.mojo`'s header).

⚠⚠ `--act-exec N` — HOW MUCH OF EACH CHUNK IS EXECUTED, AND WHY IT IS A FLAG.
`0` (the default) is the paper's temporal ensemble: query every step, execute
the weighted mean of every chunk that covers it. `N >= 1` is LeRobot's
`n_action_steps`: query, execute the chunk's first N actions open-loop, query
again. The first 5090 run (2026-09-18, ensemble) scored 0/200 with a mean
|action| of 0.107 against the demonstrations' 0.294 — at `m = 0.01` the 40
overlapping predictions are weighted almost uniformly and their mean shrinks
towards the dataset's; `--act-exec 1` isolates that (query every step, take
the newest chunk's first action) and `--act-exec 10` is the half-second
open-loop setting LeRobot's LIBERO configs run. The per-word |action| table
at the end is printed beside the store's own mean and spread for this reason.

⚠⚠ `--check-obs [STORE]` — IS THE OBSERVATION THE ONE THE POLICY TRAINED ON?
Every gate so far compares physics or the store against the recording; none
compares the picture THIS driver renders inside the loop against the picture
the store holds. At the first policy step of the first chunk each lane's two
rendered frames are scored (PSNR, bytes) against the rendered store's frame 0
of the demonstration its init row came from — the frozen inits ARE the demos'
initial states, in order — and against the NEXT demo's frame 0 as a control.
A consistent pipeline reads 30 dB+ on the demo and clearly less on the
control; the five settle steps and the fixture draw (the store carries each
demo's, the env the band centre) cost a few dB, not twenty. Three image
policies at 0-5/200 with no such check is how a wrong picture hides.

`--trace-lane L` — every step of the first chunk, lane L's seven action words
as written to the controller and its finger opening (joint 8), one line per
step. The video showed the gripper centred on the drawer handle at step ~75
and descending past it without closing; the trace says what was commanded
there.

`--video FILE.mp4 [--video-lane L]` — lane L of the FIRST chunk, both cameras
side by side (agentview | eye_in_hand, x3 nearest-neighbour), every step
including the settle, at the control rate. The policy starts at the
demonstrations' action scale and stops within seconds (measured 2026-09-19);
what the arm does in between is a thing to watch, not to infer. Needs
`ffmpeg` on PATH.

`--knn [STORE] [--knn-k N]` — THE VISION-FREE CONTROL. No network: at each
query every lane's nine joint values (standardised by the store's own spread)
are matched against every row of the SAME task in the low-dim store
(`build/demos/<family>.lowdim.h5` unless given), and the chunk executed is the
per-step, per-word MEDIAN of the k nearest rows' next K recorded actions —
the same conditional-median target the L1 fit converges to, resolved from
proprioception alone. `--act-exec` applies as for `--act` (0 = the temporal
ensemble). It exists because the validation L1 cannot separate a policy that
reads the picture from one that does not: on this store a k=5 median chunk
from qpos+task scores 0.436, one from the FULL state 0.424, the task+phase
oracle 0.386, and the fitted ACT 0.405 (2026-09-19). What the closed loop does
with the proprioception-only median is therefore the number that says whether
the picture is what is missing. `--knn-vel` adds the nine joints' one-step
differences to the match (the eval keeps each lane's previous nine values;
the store's row t uses row t-1 of its episode, zero at t=0): on the held-out
drawer demos the joint-only neighbours pull sideways at the hook moment in 2
of 7, the joint+velocity neighbours in 6 of 7 (2026-09-19). Both scored
0/200 on the 5090 the same day with the drawer joint never moving on lane 0;
the joint-only trace latched onto a demonstrator's RETRY rows (lift + back,
44 rows of 7027) and the velocity one onto episode-end rows, so the traced
lane now carries its seven arm joints, the grip site's xyz, and the match
distance of the nearest row (std units) — where the arm IS beside what it
was told — and every chunk ends with each drawer lane's largest opening.

⚠ The four kNN box runs of 2026-09-19/20 (0/200 each) are VOID: the query
was standardised without the mean the rows had subtracted, 6-7 std off
(the trace's `nn` column said so). A load-time self-test now feeds a row's
own words back and requires distance 0.

`--demo-init [STORE]` — start each lane from its paired demo's OWN frame-0
state (`state` column, our order; `_demo_state0`, the `--check-obs` leg)
instead of the frozen init row, then the settle steps as usual. The
recorded-action replay that opened 19 drawers in 20 (§6t) started there; a
policy that opens them from here and not from the frozen inits is a policy
whose first chunk is wrong, not its hook.

⚠ `norm.json` NAMES THE STORE THE CHECKPOINT WAS FITTED ON, and this driver
prints it: a checkpoint from the RECORDED store crosses the pixel-domain gap
here (robosuite's OpenGL -> our tracer), one from the RENDERED store does not,
and the two rates are only comparable when the table says which is which.

## ⚠ THE SUCCESS WORD IS THE DEVICE'S, CHECKED AGAINST THE HOST ON A SAMPLE

`META_IDX_GOAL_HELD` per lane per step (the config's reward hook wrote it from
`eval_tape_gpu`). `--check-lanes K` re-evaluates K lanes per step with the HOST
evaluator on that lane's downloaded state, the way `libero_demo_batched` does;
a disagreement raises. Checking every lane every step is most of the run's
cost at 200 lanes, which is why it is a sample.

## ⚠ LANES ARE A COMPILE-TIME CONSTANT; THE TABLE IS RUN IN CHUNKS

`LANES` is `N_ENVS` of the env (one kernel instantiation per value, `sed` it —
the `libero_demo_batched` idiom). A table of 200 rows on 20 lanes runs as ten
chunks, and every row is recorded exactly once — `SuccessReport` refuses a
report with an unseen lane rather than counting it as a failure.
"""

from std.os import listdir
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.checkpoint import load_params
from noeira.core.run import resolve_checkpoint
from noeira.nn.core.initializer import Kaiming
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.model.model_def import ModelDefLike
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.dynamics.osc_pose import ARM_DOF, OscPoseConfig
from noeira.physics3d.dynamics.osc_pose_gpu import (
    OSC_ACTION_DIM, build_osc_refs,
)
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
    META_IDX_GOAL_HELD, META_IDX_NUM_CONTACTS, META_IDX_INIT_REGION_0,
    META_INIT_SLOTS, META_IDX_JINIT_0, META_JINIT_SLOTS, META_JINIT_WORDS,
    META_IDX_SHAPE_W_GOAL, META_IDX_SHAPE_W_REACH, MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
)
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family, FamilySpec, TaskSpec,
)
from noeira.tasks.family import scene_path
from noeira.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses, BoundGoal,
)
from noeira.tasks.eval import (
    eval_goal, HostState, region_sites, region_contact_bodies,
)
from noeira.tasks.eval_report import SuccessReport
from noeira.tasks.init_table import load_init_table, InitTable
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.tasks.gpu_eval import region_table_words, require_gpu_regions
from noeira.tasks.active import active_mask, init_region_words
from noeira.deep_agents.bc.policy import BcNet, BcNorm, load_bc_norm
from noeira.physics3d.raytrace import BatchedCameraRenderer, RGB_CHANNELS
from noeira.physics3d.fields.dims import DimsLike
from max.gpu.host import HostBuffer
from noeira.physics3d.raytrace.visual import build_visual_model
from noeira.envs.libero.visual import libero_site_conditions
from noeira.envs.libero.act import (
    LiberoActTrainer, LIBERO_ACT_QPOS, LIBERO_ACT_PROPRIO, LIBERO_ACT_ADIM,
    LIBERO_ACT_K,
    LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W, LIBERO_ACT_IMG_ELEMS, LIBERO_ACT_N_CAM,
)
from noeira.deep_agents.act.norm_file import ACTNorm
from noeira.deep_agents.act.inference import (
    TemporalEnsemble, normalize_camera_chw, denormalize,
)
from noeira.deep_agents.act.config import ACT_TEMPORAL_ENSEMBLE_M
from noeira.data.store import TrajectoryStore
from noeira.render.video_recorder import VideoRecorder
from noeira.envs.libero.act import LIBERO_ACT_STORE_RENDERED
from std.math import log10, sqrt
from std.memory.alloc import unsafe_alloc
from noeira.tasks.placement.table import PlacementTable
from noeira.tasks.placement.check import (
    joint_init_words, require_device_placement,
)
from noeira.envs.libero.osc_config import LiberoOscConfig, LIBERO_CONTROL_FREQ
from noeira.envs.libero.placement.libero_goal import LiberoGoalPlacement
from noeira.envs.libero.models.libero_goal_xml import LiberoGoalModel
from noeira.envs.libero.placement.libero_object import LiberoObjectPlacement
from noeira.envs.libero.models.libero_object_xml import LiberoObjectModel
from noeira.envs.libero.placement.libero_spatial import LiberoSpatialPlacement
from noeira.envs.libero.models.libero_spatial_xml import LiberoSpatialModel
from noeira.envs.libero.placement.libero_kitchen_scene3 import LiberoKitchenScene3Placement
from noeira.envs.libero.models.libero_kitchen_scene3_xml import LiberoKitchenScene3Model
from noeira.envs.libero.placement.libero_kitchen_scene5 import LiberoKitchenScene5Placement
from noeira.envs.libero.models.libero_kitchen_scene5_xml import LiberoKitchenScene5Model


comptime H = DType.float64
comptime FAMILY = "libero_goal"
"""The family this build evaluates. ⚠ `sed` it — `LANES` too, see the header.

⚠ ONLY THE FAMILIES BELOW ARE IMPORTED. A batched env is a kernel
instantiation per model and this driver is run on one family at a time;
`libero_family_batched.mojo` carries all 23 for the smoke gate. Add a branch
in `main` when a family gains an init table."""
comptime LANES = 20
comptime N_ENVS = LANES
comptime FAMILY_DIR = "noeira/envs/libero/families/"
comptime TASK_DIR = "noeira/envs/libero/tasks/"
comptime SETTLE_STEPS = 5
"""`metric.py`: `for _ in range(5): obs, _, _, _ = env.step(dummy)`."""
comptime LIBERO_MAX_STEPS = 600
"""`cfg.eval.max_steps`."""
comptime LIBERO_N_EVAL = 20
"""Frozen inits per task — the number every published LIBERO rate is over."""
comptime SAMPLED_SEED = 11


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("libero eval batched: no '" + want + "' in the scene")


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


def _pad(s: String, n: Int) -> String:
    var out = String(s)
    if out.byte_length() > n:
        return String(out[byte = 0 : n])
    while out.byte_length() < n:
        out += " "
    return out^


def _clamp(x: Float64) -> Float64:
    """OSC_POSE's action box. ⚠ THE FIT HAS NO OUTPUT SQUASH (`bc_policy`), so
    a regression head can step outside [-1, 1]; robosuite clips there too."""
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def _byte(x: Float64) -> Scalar[DType.uint8]:
    """A tracer float in [0, 1] to the byte the store holds — the same
    rounding `libero_demo_rerender.mojo` wrote the training frames with."""
    var v = Int(x * 255.0 + 0.5)
    if v < 0:
        v = 0
    if v > 255:
        v = 255
    return Scalar[DType.uint8](v)


def _fd(x: Float64, d: Int) -> String:
    """`x` rounded to `d` decimals (the width helper `_f` TRUNCATES characters:
    `_f(-2.404, 3)` is "-2." — a trace read as joints at -2 rad was that)."""
    var scale = 1.0
    for _ in range(d):
        scale *= 10.0
    var r = Float64(Int(x * scale + (0.5 if x >= 0 else -0.5))) / scale
    var out = String(r)
    var dot = out.find(".")
    if dot < 0:
        out += "."
        dot = out.byte_length() - 1
    while out.byte_length() - dot - 1 < d:
        out += "0"
    return out^


def _f(x: Float64, n: Int) -> String:
    var out = String(x)
    if out.byte_length() > n:
        return String(out[byte = 0 : n])
    return out^


def _mean(xs: List[Float64]) -> Float64:
    var t = 0.0
    for k in range(len(xs)):
        t += xs[k]
    return t / Float64(len(xs)) if len(xs) > 0 else 0.0


def _psnr_u8_pp(
    a: Pointer[Scalar[DType.uint8], MutAnyOrigin], ao: Int,
    b: Pointer[Scalar[DType.uint8], MutAnyOrigin], bo: Int, n: Int,
) -> Float64:
    var se = 0.0
    for i in range(n):
        var d = Float64(Int(a[unsafe_offset = ao + i])) - Float64(Int(b[unsafe_offset = bo + i]))
        se += d * d
    var mse = se / Float64(n)
    return 99.0 if mse <= 0.0 else 10.0 * log10(255.0 * 255.0 / mse)


def _psnr_u8(
    a: List[Scalar[DType.uint8]], ao: Int,
    b: Pointer[Scalar[DType.uint8], MutAnyOrigin], bo: Int, n: Int,
) -> Float64:
    var se = 0.0
    for i in range(n):
        var d = Float64(Int(a[ao + i])) - Float64(Int(b[unsafe_offset = bo + i]))
        se += d * d
    var mse = se / Float64(n)
    return 99.0 if mse <= 0.0 else 10.0 * log10(255.0 * 255.0 / mse)


def _render_pack[
    D: DimsLike, LANES_: Int, W: Int, H: Int, SH: Bool, RF: Bool, SM: Int,
    AIMG: Int, ACAM: Int, ANPIX: Int,
](
    mut r: BatchedCameraRenderer[DT, D, LANES_, W, H, SH, RF, SM],
    ctx: DeviceContext,
    mut d: Data[DT, D, LANES_],
    mut m: Model[DT, D],
    ref cam_idx: List[Int],
    h_rgb: HostBuffer[DT],
    mut act_u8: List[Scalar[DType.uint8]],
) raises:
    """Both cameras for every lane, packed as the store holds them — uint8
    CHW, top row first, camera slots in `cam_idx` order — the ONE spelling
    the policy step and `--check-obs` share."""
    for cam in range(len(cam_idx)):
        r.render(ctx, d, m, cam_idx[cam])
        ctx.enqueue_copy(h_rgb, r.rgb)
        ctx.synchronize()
        var p = h_rgb.unsafe_ptr()
        for e in range(LANES_):
            var src = e * ANPIX * RGB_CHANNELS
            var dst = e * AIMG + cam * ACAM
            for q in range(ANPIX):
                for c in range(3):
                    act_u8[dst + c * ANPIX + q] = _byte(
                        Float64(p[unsafe_offset = src + q * 3 + c])
                    )


def _demo_state0(
    store_path: String, ref lane_row: List[Int], ref row_task: List[Int],
    n_inits: Int, nq: Int, nv: Int,
) raises -> List[Float64]:
    """Each lane's paired demo's frame-0 `state` (qpos ++ qvel, OUR order)
    from the rendered store — `[LANES, nq + nv]`, zeros for a lane without
    a demo. The pre-settle leg of `--check-obs` renders THIS, not the init
    row: the frozen inits are not the demos' first states (measured on the
    Mac: 0.04-0.11 rad on the arm, the bowl 7 cm higher), so a picture of the
    init row can never match the store's frame 0 whatever the pipeline
    does. Same state on both sides leaves only the fixture draw."""
    var out = List[Float64](length=len(lane_row) * (nq + nv), fill=0.0)
    if not exists(store_path):
        return out^
    var st = TrajectoryStore(store_path)
    var task_col = st.load_column[DType.int32](String("task_index"))
    var sspec = st.column(String("state"))
    if sspec.row_dim() != nq + nv:
        raise Error("--check-obs: the store's state is " + String(sspec.row_dim())
                    + " wide, the env's is " + String(nq + nv))
    var per_task = List[List[Int]]()
    var n_tasks = 0
    for e in range(st.n_episodes()):
        var ti = Int(task_col[st.episodes.start_of(e)])
        while n_tasks <= ti:
            per_task.append(List[Int]())
            n_tasks += 1
        per_task[ti].append(e)
    var buf = unsafe_alloc[Scalar[DType.float64]](nq + nv).as_unsafe_any_origin()
    for l in range(len(lane_row)):
        var r = lane_row[l]
        if r < 0:
            continue
        var ti = row_task[r]
        var di = r % n_inits
        if ti >= n_tasks or di >= len(per_task[ti]):
            continue
        var r0 = st.episodes.start_of(per_task[ti][di])
        st.read_range[DType.float64](String("state"), r0, r0 + 1, buf)
        for k in range(nq + nv):
            out[l * (nq + nv) + k] = Float64(buf[unsafe_offset=k])
    buf.unsafe_free()
    return out^


def _check_obs[AIMG: Int, ACAM: Int, ANPIX: Int](
    store_path: String, ref act_u8: List[Scalar[DType.uint8]],
    ref lane_row: List[Int], ref row_task: List[Int], n_inits: Int,
    ref lane_qpos9: List[Float64],
    mut out_own: List[Float64], mut out_ctrl: List[Float64],
) raises:
    """Each lane's packed observation vs the store's frame 0 of its own demo
    (init row i of task t == demo i of task t) and of the next demo.

    Per printed lane, two more numbers that split "a different picture of the
    same state" from "a different state": the largest difference between the
    lane's nine joint values and the store's `qpos[0:9]` at that frame, and
    the store's OWN frame-0 PSNR between the two demos (what one demo to the
    next costs with the same pipeline on both sides)."""
    if not exists(store_path):
        print("  ⚠ --check-obs: no store at", store_path, "— not checked")
        return
    var st = TrajectoryStore(store_path)
    var task_col = st.load_column[DType.int32](String("task_index"))
    var spec = st.column(String("images"))
    if spec.row_dim() != AIMG:
        raise Error("--check-obs: the store's images are " + String(spec.row_dim())
                    + " bytes per row, the policy's are " + String(AIMG))
    # episodes per task, in store order
    var per_task = List[List[Int]]()
    var n_tasks = 0
    for e in range(st.n_episodes()):
        var ti = Int(task_col[st.episodes.start_of(e)])
        while n_tasks <= ti:
            per_task.append(List[Int]())
            n_tasks += 1
        per_task[ti].append(e)
    var buf = unsafe_alloc[Scalar[DType.uint8]](AIMG).as_unsafe_any_origin()
    var buf2 = unsafe_alloc[Scalar[DType.uint8]](AIMG).as_unsafe_any_origin()
    var qspec = st.column(String("qpos"))
    var qw = qspec.row_dim()
    var qbuf = unsafe_alloc[Scalar[DType.float32]](qw).as_unsafe_any_origin()
    var lanes = len(lane_row)
    for l in range(lanes):
        var r = lane_row[l]
        if r < 0:
            continue
        var ti = row_task[r]
        var di = r % n_inits
        if ti >= n_tasks or di + 1 >= len(per_task[ti]):
            continue
        var dq = 0.0
        for which in range(2):
            var e = per_task[ti][di + which]
            var r0 = st.episodes.start_of(e)
            st.read_range[DType.uint8](String("images"), r0, r0 + 1,
                                       buf if which == 0 else buf2)
            for cam in range(AIMG // ACAM):
                var p = _psnr_u8(act_u8, l * AIMG + cam * ACAM,
                                 buf if which == 0 else buf2, cam * ACAM, ACAM)
                if which == 0:
                    out_own.append(p)
                else:
                    out_ctrl.append(p)
            if which == 0:
                st.read_range[DType.float32](String("qpos"), r0, r0 + 1, qbuf)
                for k in range(9):
                    var d = abs(lane_qpos9[l * 9 + k] - Float64(qbuf[unsafe_offset=k]))
                    if d > dq:
                        dq = d
        if l < 4:
            # store vs store: demo di's frame 0 against demo di+1's
            var s_av = _psnr_u8_pp(buf, 0, buf2, 0, ACAM)
            var s_eih = _psnr_u8_pp(buf, ACAM, buf2, ACAM, ACAM)
            print("    lane", l, "task", ti, "demo", di, ": agentview",
                  out_own[len(out_own) - 2], "/ ctrl", out_ctrl[len(out_ctrl) - 2],
                  "| eye_in_hand", out_own[len(out_own) - 1], "/ ctrl",
                  out_ctrl[len(out_ctrl) - 1], "dB | max |dq| arm+fingers vs the"
                  " store's frame 0:", dq, "rad | store demo vs next demo:",
                  s_av, "/", s_eih, "dB")
    buf.unsafe_free()
    buf2.unsafe_free()
    qbuf.unsafe_free()


def _knn_chunk[AK: Int, AA: Int](
    ref q9: List[Float64], KF: Int,
    ref knn_q: List[Float64], ref knn_std: List[Float64], ref knn_mean: List[Float64],
    ref knn_act: List[Float64], ref knn_left: List[Int],
    ref rows: List[Int], k: Int,
    mut out: List[Scalar[DT]], obase: Int,
) raises -> Float64:
    """The k nearest rows of `rows` to this lane's nine joints, then the
    per-step, per-word median of their next AK recorded actions, RAW units.
    Past a neighbour's episode end its contribution is the zero action."""
    var q = List[Float64](length=KF, fill=0.0)
    var best_d = List[Float64](length=k, fill=1.0e300)
    var best_g = List[Int](length=k, fill=-1)
    # the lane's words, standardised EXACTLY as the store's rows were
    for j in range(KF):
        q[j] = (q9[j] - knn_mean[j]) / knn_std[j]
    for i in range(len(rows)):
        var g = rows[i]
        var d = 0.0
        for j in range(KF):
            var t = q[j] - knn_q[g * KF + j]
            d += t * t
        if d < best_d[k - 1]:
            var p = k - 1
            while p > 0 and best_d[p - 1] > d:
                best_d[p] = best_d[p - 1]
                best_g[p] = best_g[p - 1]
                p -= 1
            best_d[p] = d
            best_g[p] = g
    var n = 0
    for i in range(k):
        if best_g[i] >= 0:
            n += 1
    if n == 0:
        raise Error("--knn: no rows for this lane's task")
    var vals = List[Float64](length=n, fill=0.0)
    for t in range(AK):
        for j in range(AA):
            for i in range(n):
                var g = best_g[i]
                vals[i] = knn_act[(g + t) * AA + j] if t < knn_left[g] else 0.0
            # insertion sort, n <= 64
            for a in range(1, n):
                var v = vals[a]
                var b = a
                while b > 0 and vals[b - 1] > v:
                    vals[b] = vals[b - 1]
                    b -= 1
                vals[b] = v
            var med = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
            out[obase + t * AA + j] = Scalar[DT](med)
    return sqrt(best_d[0])


def run[T: PlacementTable, M: ModelDefLike](
    n_inits: Int, max_steps: Int, check_lanes: Int, sampled: Bool,
    policy_path: String, act_dir: String, act_exec: Int, obs_store: String,
    video_path: String, video_lane: Int, trace_lane: Int,
    knn_store: String, knn_k: Int, knn_vel: Bool, demo_init: String,
    only_task: Int, act_ckpt: String, act_m: Float64, act_latent: String,
    act_no_dq: Bool,
) raises:
    comptime E = Phyics3dBatchedEnv[
        M, LiberoOscConfig[T], LANES, CRBA_TREEWALK=True
    ]
    comptime NQ = M.NQ
    comptime NV = M.NV
    comptime NB = M.NBODY
    comptime NS = M.NSITE
    comptime MC = M.MAX_CONTACTS
    comptime OD = M.OBS_DIM
    # ── the image policy's types: the tracer over THIS env's Data, ACT at
    # a batch of LANES. Both are compiled whether or not `--act` is given;
    # neither is constructed unless it is.
    comptime Renderer = BatchedCameraRenderer[
        DT, E.MD, LANES, LIBERO_ACT_IMG_W, LIBERO_ACT_IMG_H, False, True, 4
    ]
    comptime ACT_T = LiberoActTrainer[LANES, "gpu"]
    comptime AQ = LIBERO_ACT_QPOS
    comptime AQP = LIBERO_ACT_PROPRIO
    comptime AA = LIBERO_ACT_ADIM
    comptime AK = LIBERO_ACT_K
    comptime AIMG = LIBERO_ACT_IMG_ELEMS
    comptime ANPIX = LIBERO_ACT_IMG_H * LIBERO_ACT_IMG_W
    comptime ACAM = 3 * ANPIX
    var family = String(FAMILY)

    print("=" * 78)
    print("LIBERO's eval protocol on the batch —", family, "|", LANES, "lanes")
    print("=" * 78)

    var f = load_family(String(FAMILY_DIR) + family + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var names = _task_names(family)
    var n_tasks = len(names)

    # ── the tasks: the goal (from the .task) and its words ────────────────
    var nqs = List[Int]()
    var jt = List[Int]()
    var jvn = List[Int]()
    for k in range(len(fmd.joints)):
        nqs.append(fmd.joints[k].nq)
        jt.append(fmd.joints[k].jnt_type)
        jvn.append(fmd.joints[k].nv)
    var jadr = joint_qpos_addresses(nqs)
    var goals = List[BoundGoal]()
    var tapes = List[List[Float64]]()
    var masks = List[Float64]()
    var iwords = List[List[Float64]]()
    var jwords = List[List[Float64]]()
    var languages = List[String]()
    for ti in range(n_tasks):
        var t = load_task(String(TASK_DIR) + names[ti] + ".task")
        validate_task_against_family(t, f)
        var g = bind_goal(
            parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
            fmd.joint_names, jadr,
        )
        require_tier_a(g, t.name)
        require_gpu_regions(g, t.name)
        require_device_placement[T](t, f)
        tapes.append(encode_goal(g))
        masks.append(active_mask(t, f))
        iwords.append(init_region_words(t, f))
        jwords.append(joint_init_words[T](t))
        languages.append(t.language)
        goals.append(g^)
    var rsites = region_sites(f, fmd.site_names)
    var rcontact = region_contact_bodies(f, fmd.body_names)
    var site_body_tab = List[Int]()
    var site_quat_tab = List[Float64]()
    for k in range(len(fmd.sites)):
        site_body_tab.append(fmd.sites[k].body_id)
        site_quat_tab.append(fmd.sites[k].quat_x)
        site_quat_tab.append(fmd.sites[k].quat_y)
        site_quat_tab.append(fmd.sites[k].quat_z)
        site_quat_tab.append(fmd.sites[k].quat_w)
    var body_parent_tab = List[Int]()
    body_parent_tab.append(-1)
    for k in range(len(fmd.bodies)):
        body_parent_tab.append(fmd.bodies[k].parent)

    # ── the rows: LIBERO's frozen fifty, or the sampler ───────────────────
    var table_path = String("build/init/") + family + ".init.h5"
    var have_table = exists(table_path) and not sampled
    var row_task = List[Int]()
    var row_mask = List[Float64]()
    var row_qpos = List[List[Float64]]()
    var row_qvel = List[List[Float64]]()
    var report_opt = List[SuccessReport]()
    if have_table:
        var full = load_init_table(table_path, f.name, NQ, NV)
        var tbl = full.prefix_per_task(n_inits)
        if only_task >= 0:
            # ⚠ ONE TASK'S ROWS, REPORTED OVER ONE TASK: a per-task fit scored
            # on the ten-task table spends 17 min on 180 episodes it was not
            # fitted for. The report's denominators say which rows ran.
            tbl = full.prefix_per_task(n_inits).only_task(only_task)
            print("  task  : ONLY task", only_task, "(--task) —", tbl.n_rows(),
                  "rows; NOT the ten-task benchmark number")
        # ⚠ THE TABLE'S LABEL AGAINST THE `.task`'s `language=`, per row: the
        # table's `task_index` is an ORDER, and `libero_eval.mojo` refuses a
        # table whose order is not this driver's task list.
        for r in range(tbl.n_rows()):
            var ti = Int(tbl.task_index[r])
            if ti < 0 or ti >= n_tasks:
                raise Error(
                    table_path + ": row " + String(r) + " names task index "
                    + String(ti) + ", the family has " + String(n_tasks)
                )
            if tbl.task_label(r) != languages[ti]:
                raise Error(
                    table_path + ": row " + String(r) + " is labelled '"
                    + tbl.task_label(r) + "', task " + String(ti) + " ("
                    + names[ti] + ") says '" + languages[ti] + "'"
                )
            var q = List[Float64](length=NQ, fill=0.0)
            var v = List[Float64](length=NV, fill=0.0)
            tbl.apply(r, q, v)
            row_task.append(ti)
            row_mask.append(tbl.mask[r])
            row_qpos.append(q^)
            row_qvel.append(v^)
        report_opt.append(SuccessReport(tbl))
        print("  inits :", table_path, "|", full.n_rows(), "rows ->",
              tbl.n_rows(), "(", n_inits, "per task ) — LIBERO's OWN")
    else:
        # ⚠ SMOKE MODE. Every line below says so; it is not a benchmark rate.
        for ti in range(n_tasks):
            for _ in range(n_inits):
                row_task.append(ti)
                row_mask.append(masks[ti])
                row_qpos.append(List[Float64]())
                row_qvel.append(List[Float64]())
        if sampled:
            print("  inits : SAMPLED by the device (--sampled) — NOT a"
                  " benchmark number")
        else:
            print("  inits : no table at", table_path,
                  "— SAMPLED by the device, NOT a benchmark number")
            print("          build it: pixi run libero-init-dump && pixi run"
                  " libero-init-freeze")
    var n_rows = len(row_task)
    print("  tasks :", n_tasks, "| rows", n_rows, "| horizon", SETTLE_STEPS,
          "settle +", max_steps, "steps", "(LIBERO's own)" if max_steps
          == LIBERO_MAX_STEPS else "(REDUCED)")
    # ⚠⚠ THE POLICY IS BUILT FROM `deep_agents/bc/policy.BcNet`, the SAME
    # declaration `libero_bc_train` fitted, and `load_params` validates every
    # layer's name and size — a checkpoint of a different shape raises here
    # rather than loading the layers that happen to match.
    comptime POLICY = BcNet[OD, OSC_ACTION_DIM]
    var have_bc = policy_path != ""
    var have_act = act_dir != ""
    var have_knn = knn_store != ""
    if (have_bc and have_act) or (have_bc and have_knn) or (have_act and have_knn):
        raise Error("libero eval batched: --policy, --act and --knn are three"
                    " policies; give one")
    var have_policy = have_bc or have_act or have_knn
    var net = POLICY.make["cpu", Kaiming](None)
    var norm = BcNorm()
    var act_norm = ACTNorm()
    if have_bc:
        load_params["cpu"](net, policy_path, None)
        norm = load_bc_norm(
            policy_path + ".norm", OD, OSC_ACTION_DIM,
            "Re-run libero-bc-train.",
        )
        print("  policy:", policy_path, "| obs", OD, "-> 7, clamped to [-1, 1]")
    elif have_act:
        if not exists(act_dir + "/" + act_ckpt + ".ckpt") or not exists(act_dir + "/norm.json"):
            raise Error("libero eval batched: --act " + act_dir + " has no "
                        + act_ckpt + ".ckpt + norm.json (a libero-act-train"
                        " checkpoint directory; --act-ckpt best|last)")
        act_norm = ACTNorm.load(act_dir + "/norm.json", AQ, AA)
        print("  policy: ACT", act_dir + "/" + act_ckpt + ".ckpt", "| qpos", AQ, "+",
              LIBERO_ACT_N_CAM, "cameras", LIBERO_ACT_IMG_W, "x",
              LIBERO_ACT_IMG_H, "-> chunk", AK, "x", AA,
              ", temporal ensemble m =", act_m,
              "(paper's 0.01; NEGATIVE favours the newest query)" if act_m != Float64(ACT_TEMPORAL_ENSEMBLE_M) else "")
        print("          fitted on", act_norm.store)
        if act_no_dq:
            print("          proprio: the nine DIFFERENCE words fed as ZERO (--act-no-dq; the"
                  " ablation, for a checkpoint fitted on a --no-dq store)")
        print("          latent :", "z ~ N(0, I), a prior DRAW per query (--act-latent sample)"
              if act_latent == "sample" else "z = 0, the prior mean (the paper's)")
        if act_exec == 0:
            print("          chunk use: TEMPORAL ENSEMBLE (query every step)")
        else:
            print("          chunk use: execute", act_exec, "of", AK,
                  "open-loop, then re-query (--act-exec)")
    elif have_knn:
        if not exists(knn_store):
            raise Error("libero eval batched: --knn store not found: " + knn_store)
        print("  policy: kNN median chunk, k =", knn_k, "from", knn_store,
              "| qpos", AQP, "+ dq" if knn_vel else "", "(no cameras) -> chunk",
              AK, "x", AA)
        if act_exec == 0:
            print("          chunk use: TEMPORAL ENSEMBLE (query every step)")
        else:
            print("          chunk use: execute", act_exec, "of", AK,
                  "open-loop, then re-query (--act-exec)")
    else:
        print("  policy: ZERO ACTION —", "the L6 gate is that the rate is 0")
    if demo_init != "":
        if not exists(demo_init):
            raise Error("libero eval batched: --demo-init store not found: " + demo_init)
        if not have_table:
            raise Error("libero eval batched: --demo-init pairs init row i of task"
                        " t with demo i of task t and needs the frozen table")
        print("  inits : REPLACED by each lane's paired demo's frame-0 state from",
              demo_init, "(--demo-init) — NOT a benchmark number")
    if not have_table:
        # ⚠ AND ON THE BOX THE TABLE IS SIMPLY NOT THERE: it is a gitignored
        # build artifact (642 KB), so a machine that pulled the repo has the
        # code and not the inits. Copy it or rebuild it.
        print("          (copy build/init/" + family + ".init.h5 from a"
              " machine that has it, or rebuild it there)")

    # ── the controller record ─────────────────────────────────────────────
    var qadr_all = List[Int]()
    var dadr_all = List[Int]()
    var qa = 0
    var da = 0
    for k in range(len(fmd.joints)):
        qadr_all.append(qa)
        dadr_all.append(da)
        qa += fmd.joints[k].nq
        da += fmd.joints[k].nv
    var ctrl_min = List[Float64]()
    var ctrl_max = List[Float64]()
    for k in range(len(fmd.actuators)):
        ctrl_min.append(fmd.actuators[k].ctrl_min)
        ctrl_max.append(fmd.actuators[k].ctrl_max)
    var dof = List[Int]()
    var qadr = List[Int]()
    var jidx = List[Int]()
    var act_idx = List[Int]()
    var tmin = List[Float64]()
    var tmax = List[Float64]()
    for j in range(ARM_DOF):
        var ji = _index(fmd.joint_names, String("robot_joint") + String(j + 1))
        dof.append(dadr_all[ji])
        qadr.append(qadr_all[ji])
        jidx.append(ji)
        var ai = _index(fmd.actuator_names, String("robot_torq_j") + String(j + 1))
        act_idx.append(ai)
        tmin.append(ctrl_min[ai])
        tmax.append(ctrl_max[ai])
    var site = _index(fmd.site_names, String("robot_grip_site"))
    var ga1 = _index(fmd.actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd.actuator_names, String("robot_gripper_finger_joint2"))
    var cfg = OscPoseConfig()
    var refs = build_osc_refs(
        dof^, qadr^, jidx^, tmin^, tmax^, act_idx^, site,
        fmd.sites[site].body_id, ga1, ga2,
        ctrl_min[ga1], ctrl_max[ga1], ctrl_min[ga2], ctrl_max[ga2],
        cfg.kp, cfg.damping_ratio, cfg.output_max_pos, cfg.output_max_ori,
        cfg.nullspace_kp, cfg.gripper_speed,
    )

    # ══ THE BATCH ══════════════════════════════════════════════════════════
    var ctx = DeviceContext()
    var env = E(ctx)
    env.set_osc_refs(refs, ctx)
    var cw = region_table_words(f, rsites, rcontact)
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DT](cw[k])
    env.mf.curriculum.upload(ctx)
    # ── the image policy, constructed only for --act ─────────────────────
    var ren_opt = List[Renderer]()
    var act_opt = List[ACT_T]()
    var cam_idx = List[Int]()
    var qadr9 = List[Int]()
    var ens = List[TemporalEnsemble[AA, AK]]()
    var h_rgb = ctx.enqueue_create_host_buffer[DT](1)
    var act_u8 = List[Scalar[DType.uint8]]()
    var act_images = List[Scalar[DT]]()
    var act_qpos = List[Scalar[DT]]()
    var act_prev = List[Float64](length=LANES * AQP, fill=0.0)  # last step's joints
    var act_dummy = List[Scalar[DT]]()
    var act_valid = List[Scalar[DT]]()
    var act_chunk = List[Scalar[DT]]()
    var act_pred = List[Scalar[DT]](length=AA, fill=Scalar[DT](0))
    var act_out = List[Scalar[DT]](length=AA, fill=Scalar[DT](0))
    var render_ns = 0
    var forward_ns = 0
    var physics_ns = 0
    var obs_checked = False
    var recorder = VideoRecorder()
    var recording = (have_act or have_knn) and video_path.byte_length() > 0
    comptime VSCALE = 3
    comptime VW = LIBERO_ACT_IMG_W * VSCALE * 2
    comptime VH = LIBERO_ACT_IMG_H * VSCALE
    var vframe = List[UInt8](length=VW * VH * 4, fill=UInt8(255))
    if recording:
        if video_lane < 0 or video_lane >= LANES:
            raise Error("--video-lane must be in [0, " + String(LANES) + ")")
        recorder.start(video_path, fps=LIBERO_CONTROL_FREQ)
        print("  video : lane", video_lane, "of chunk 0 ->", video_path)
    var obs_psnr = List[Float64]()
    var obs_psnr_ctrl = List[Float64]()
    var obs_psnr_pre = List[Float64]()
    var obs_psnr_pre_ctrl = List[Float64]()
    var word_abs = List[Float64](length=OSC_ACTION_DIM, fill=0.0)
    var word_n = 0
    var drawer_qadr = -1
    var drawer_top_qadr = -1
    for j in range(len(fmd.joint_names)):
        if fmd.joint_names[j] == "wooden_cabinet_1_middle_level":
            drawer_qadr = qadr_all[j]
        if fmd.joint_names[j] == "wooden_cabinet_1_top_level":
            drawer_top_qadr = qadr_all[j]
    var drawer_max = List[Float64](length=LANES * 2, fill=0.0)  # |top|, |mid| per lane
    var knn_dist = List[Float64](length=LANES, fill=0.0)
    var grip_site = -1
    for j in range(len(fmd.site_names)):
        if fmd.site_names[j] == "robot_grip_site":
            grip_site = j
    for j in range(ARM_DOF):
        qadr9.append(qadr_all[_index(fmd.joint_names, String("robot_joint") + String(j + 1))])
    qadr9.append(qadr_all[_index(fmd.joint_names, String("robot_finger_joint1"))])
    qadr9.append(qadr_all[_index(fmd.joint_names, String("robot_finger_joint2"))])
    if have_act or recording:
        cam_idx.append(_index(fmd.camera_names, String("arena_agentview")))
        cam_idx.append(_index(fmd.camera_names, String("robot_eye_in_hand")))
        # ⚠ THE SAME VISUAL MODEL THE RENDERED STORE WAS DRAWN WITH: group 1,
        # the stove burner rule, LIBERO's lights and textures.
        ren_opt.append(Renderer(ctx, env.mf, cam_idx[0]))
        ren_opt[0].set_visual(
            ctx,
            build_visual_model[DT, E.MD](
                fmd, env.mf, group_mask=1 << 1,
                conditions=libero_site_conditions(f),
            ),
        )
        print("  camera:", ren_opt[0].vis.describe())
        h_rgb = ctx.enqueue_create_host_buffer[DT](LANES * ANPIX * RGB_CHANNELS)
        act_u8 = List[Scalar[DType.uint8]](length=LANES * AIMG, fill=0)
    if have_act:
        if AQ != 2 * AQP + n_tasks:
            raise Error("libero eval batched: the ACT declaration carries "
                        + String(AQ - 2 * AQP) + " task words, the family has "
                        + String(n_tasks) + " tasks")
        act_opt.append(ACT_T.make(ctx=ctx))
        act_opt[0].load(act_dir + "/" + act_ckpt + ".ckpt")
        act_images = List[Scalar[DT]](length=LANES * AIMG, fill=Scalar[DT](0))
        act_qpos = List[Scalar[DT]](length=LANES * AQ, fill=Scalar[DT](0))
        act_dummy = List[Scalar[DT]](length=LANES * AK * AA, fill=Scalar[DT](0))
        act_valid = List[Scalar[DT]](length=LANES * AK, fill=Scalar[DT](1))
    if have_act or have_knn:
        for _ in range(LANES):
            ens.append(TemporalEnsemble[AA, AK](m=act_m))
        act_chunk = List[Scalar[DT]](length=LANES * AK * AA, fill=Scalar[DT](0))
    # ── the vision-free control: the store's rows, standardised ──────────
    var KF = 2 * AQP if knn_vel else AQP  # match words per row
    var knn_q = List[Float64]()      # [n_rows_store, KF]
    var knn_task = List[Int]()       # [n_rows_store]
    var knn_act = List[Float64]()    # [n_rows_store, AA]
    var knn_left = List[Int]()       # steps left in the row's episode
    var knn_std = List[Float64](length=KF, fill=1.0)
    var knn_mean = List[Float64](length=KF, fill=0.0)
    var knn_prev = List[Float64](length=LANES * AQP, fill=0.0)  # last step's joints
    var knn_rows_of = List[List[Int]]()  # rows per task, in store order
    if have_knn:
        var st = TrajectoryStore(knn_store)
        var js = st.load_column[DType.float32](String("joint_states"))
        var gs = st.load_column[DType.float32](String("gripper_states"))
        var tk = st.load_column[DType.int32](String("task_index"))
        var ac = st.load_column[DType.float32](String("action"))
        var n_st = st.n_rows()
        if len(js) != n_st * ARM_DOF or len(gs) != n_st * 2 or len(ac) != n_st * AA:
            raise Error("--knn: the store's joint_states/gripper_states/action"
                        " are not 7/2/" + String(AA) + " wide")
        knn_left = List[Int](length=n_st, fill=0)
        for ep in range(st.n_episodes()):
            var s0 = st.episodes.start_of(ep)
            var ln = st.episodes.length_of(ep)
            for t in range(ln):
                knn_left[s0 + t] = ln - t
        for _ in range(n_tasks):
            knn_rows_of.append(List[Int]())
        var mean = List[Float64](length=KF, fill=0.0)
        var first = List[Bool](length=n_st, fill=False)
        for ep in range(st.n_episodes()):
            first[st.episodes.start_of(ep)] = True
        for g in range(n_st):
            for k in range(AQP):
                var v = Float64(js[g * ARM_DOF + k]) if k < ARM_DOF else Float64(gs[g * 2 + k - ARM_DOF])
                knn_q.append(v)
                mean[k] += v
            if knn_vel:
                for k in range(AQP):
                    var v = knn_q[g * KF + k] - (knn_q[(g - 1) * KF + k] if not first[g] else knn_q[g * KF + k])
                    knn_q.append(v)
                    mean[AQP + k] += v
            var ti = Int(tk[g])
            if ti < 0 or ti >= n_tasks:
                raise Error("--knn: store task_index " + String(ti) + " outside the family's " + String(n_tasks))
            knn_task.append(ti)
            knn_rows_of[ti].append(g)
            for j in range(AA):
                knn_act.append(Float64(ac[g * AA + j]))
        for k in range(KF):
            mean[k] /= Float64(n_st)
            var ss = 0.0
            for g in range(n_st):
                var d = knn_q[g * KF + k] - mean[k]
                ss += d * d
            knn_std[k] = sqrt(ss / Float64(n_st))
            if knn_std[k] < 1.0e-9:
                knn_std[k] = 1.0
        for g in range(n_st):
            for k in range(KF):
                knn_q[g * KF + k] = (knn_q[g * KF + k] - mean[k]) / knn_std[k]
        for k in range(KF):
            knn_mean[k] = mean[k]
        # ⚠ SELF-TEST, AT LOAD, BEFORE ANY KERNEL: a store row's own raw
        # words fed back as a query must find ITSELF at distance 0. The
        # first four box runs of this policy (2026-09-19/20, 0/200 each)
        # standardised the rows with the mean and the query without it, so
        # every lane was matched 6-7 std away from where it was; the trace's
        # `nn` column read it, the rate could not.
        var probe = List[Float64](length=KF, fill=0.0)
        var scratch = List[Scalar[DT]](length=AK * AA, fill=Scalar[DT](0))
        for ti in range(n_tasks):
            if len(knn_rows_of[ti]) == 0:
                continue
            var g = knn_rows_of[ti][len(knn_rows_of[ti]) // 2]
            for k in range(KF):
                probe[k] = knn_q[g * KF + k] * knn_std[k] + knn_mean[k]
            var d0 = _knn_chunk[AK, AA](
                probe, KF, knn_q, knn_std, knn_mean, knn_act, knn_left,
                knn_rows_of[ti], 1, scratch, 0,
            )
            var same = True
            for j in range(AA):
                if abs(Float64(scratch[j]) - knn_act[g * AA + j]) > 1.0e-6:
                    same = False
            if d0 > 1.0e-6 or not same:
                raise Error("--knn self-test: store row " + String(g)
                            + " queried with its own words is " + String(d0)
                            + " std from its nearest row (must be 0)")
        # the shared denormalise below is the identity for a raw chunk
        act_norm.action_mean = List[Scalar[DT]](length=AA, fill=Scalar[DT](0))
        act_norm.action_std = List[Scalar[DT]](length=AA, fill=Scalar[DT](1))
        print("          store rows", n_st, "| episodes", st.n_episodes(),
              "| qpos spread", _f(knn_std[0], 3), _f(knn_std[1], 3), "...",
              _f(knn_std[AQP - 1], 4))

    var act_h = ctx.enqueue_create_host_buffer[DT](LANES * OSC_ACTION_DIM)
    # ⚠ `env._obs` IS A DEVICE BUFFER, copied into a host one — the same
    # `enqueue_copy` `libero_osc_batched` does, at `M.OBS_DIM` per lane.
    var obs_h = ctx.enqueue_create_host_buffer[DT](LANES * OD)
    var pol_x = Tensor.alloc(LANES * OD)
    var pol_y = Tensor.alloc(LANES * OSC_ACTION_DIM)
    var solved = List[Bool](length=n_rows, fill=False)
    var first_step = List[Int](length=n_rows, fill=-1)
    var at_settle = List[Bool](length=n_rows, fill=False)
    # ⚠ ANTI-VACUITY ON THE POLICY ITSELF: a checkpoint that emits zeros — a
    # dead head, a normalisation that flattened its input, a load that filled
    # nothing — reports exactly like the null run, including its 0 successes.
    var act_abs = 0.0
    var act_words = 0
    var eval_cmp = 0
    var eval_bad = 0
    var saturated = 0
    var nonfinite = 0
    var singular_steps = 0
    var t_start = perf_counter_ns()

    var chunk = 0
    while chunk * LANES < n_rows:
        var base = chunk * LANES
        # ⚠ A SHORT LAST CHUNK RUNS THE SPARE LANES ON ROW 0 AND DISCARDS
        # THEM: the env's lane count is comptime, so the batch is always full.
        # `lane_row[e] < 0` marks a lane whose result is not recorded.
        var lane_row = List[Int]()
        for e in range(LANES):
            var r = base + e
            lane_row.append(r if r < n_rows else -1)

        # the words, then the device reset, then the frozen row
        for e in range(LANES):
            var r = lane_row[e] if lane_row[e] >= 0 else 0
            var ti = row_task[r]
            var mb = e * METADATA_SIZE
            for k in range(METADATA_SIZE):
                env.d.meta.data[mb + k] = Scalar[DT](0)
            for k in range(TAPE_WORDS):
                env.d.meta.data[mb + META_IDX_TASK_PARAM_0 + k] = Scalar[DT](
                    tapes[ti][k]
                )
            # ⚠ THE MASK IS THE TABLE'S ROW, NOT `active_mask(t, f)` — see the
            # header. In sampled mode they are the same value.
            env.d.meta.data[mb + META_IDX_TASK_ACTIVE] = Scalar[DT](row_mask[r])
            for k in range(META_INIT_SLOTS):
                env.d.meta.data[mb + META_IDX_INIT_REGION_0 + k] = Scalar[DT](0)
            for k in range(META_JINIT_SLOTS * META_JINIT_WORDS):
                env.d.meta.data[mb + META_IDX_JINIT_0 + k] = Scalar[DT](0)
            if not have_table:
                # the device sampler places this lane (smoke mode)
                for k in range(len(iwords[ti])):
                    env.d.meta.data[mb + META_IDX_INIT_REGION_0 + k] = Scalar[DT](
                        iwords[ti][k]
                    )
                for k in range(len(jwords[ti])):
                    env.d.meta.data[mb + META_IDX_JINIT_0 + k] = Scalar[DT](
                        jwords[ti][k]
                    )
            env.d.meta.data[mb + META_IDX_SHAPE_W_GOAL] = Scalar[DT](0)
            env.d.meta.data[mb + META_IDX_SHAPE_W_REACH] = Scalar[DT](0)
        env.d.meta.upload(ctx)
        ctx.synchronize()
        # ⚠ THE SEED IS THE CHUNK'S, so two chunks of the SAME sampled task do
        # not run the identical episode (`reset_batch` seeds by lane).
        env.reset_batch[LANES](ctx, UInt64(SAMPLED_SEED + chunk))
        ctx.synchronize()
        if have_table:
            env.d.qpos.download(ctx)
            env.d.qvel.download(ctx)
            ctx.synchronize()
            for e in range(LANES):
                var r = lane_row[e] if lane_row[e] >= 0 else 0
                for k in range(NQ):
                    env.d.qpos.data[e * NQ + k] = Scalar[DT](row_qpos[r][k])
                for k in range(NV):
                    env.d.qvel.data[e * NV + k] = Scalar[DT](row_qvel[r][k])
            env.d.qpos.upload(ctx)
            env.d.qvel.upload(ctx)
            ctx.synchronize()
            if have_act and obs_store.byte_length() > 0 and chunk == 0:
                # ⚠ THE PRE-SETTLE LEG OF --check-obs: the paired demo's OWN
                # frame-0 state written into the lanes (see `_demo_state0`),
                # forward kinematics, nothing stepped, so the ONLY thing that
                # differs from the store's frame 0 is the fixture draw (the
                # store carries each demo's, the env the band centre). A low
                # number HERE is the pipeline. The init rows are restored
                # before the settle steps.
                var ds0 = _demo_state0(obs_store, lane_row, row_task, n_inits, NQ, NV)
                for e in range(LANES):
                    for k in range(NQ):
                        env.d.qpos.data[e * NQ + k] = Scalar[DT](ds0[e * (NQ + NV) + k])
                    for k in range(NV):
                        env.d.qvel.data[e * NV + k] = Scalar[DT](ds0[e * (NQ + NV) + NQ + k])
                env.d.qpos.upload(ctx)
                env.d.qvel.upload(ctx)
                ctx.synchronize()
                env._run_fields_fk(ctx)
                ctx.synchronize()
                _render_pack[E.MD, LANES, LIBERO_ACT_IMG_W, LIBERO_ACT_IMG_H,
                             False, True, 4, AIMG, ACAM, ANPIX](
                    ren_opt[0], ctx, env.d, env.mf, cam_idx, h_rgb, act_u8,
                )
                var q9 = List[Float64]()
                for e in range(LANES):
                    for k in range(AQP):
                        q9.append(Float64(env.d.qpos.data[e * NQ + qadr9[k]]))
                print("  --check-obs, the paired demo's frame-0 STATE rendered"
                      " by this driver (FK only; fixtures at the band centre):")
                _check_obs[AIMG, ACAM, ANPIX](
                    obs_store, act_u8, lane_row, row_task, n_inits, q9,
                    obs_psnr_pre, obs_psnr_pre_ctrl,
                )
                # the frozen rows back, for the protocol
                for e in range(LANES):
                    var r = lane_row[e] if lane_row[e] >= 0 else 0
                    for k in range(NQ):
                        env.d.qpos.data[e * NQ + k] = Scalar[DT](row_qpos[r][k])
                    for k in range(NV):
                        env.d.qvel.data[e * NV + k] = Scalar[DT](row_qvel[r][k])
                env.d.qpos.upload(ctx)
                env.d.qvel.upload(ctx)
                ctx.synchronize()
                env._run_fields_fk(ctx)
                ctx.synchronize()
            # ⚠ NO `_osc_anchor` HERE. See the header: the controller keeps the
            # rest-pose anchor `reset_batch` gave it, which is robosuite's own
            # order and NOT the replay gate's.

        if demo_init != "":
            var ds0 = _demo_state0(demo_init, lane_row, row_task, n_inits, NQ, NV)
            for e in range(LANES):
                if lane_row[e] < 0:
                    continue
                for k in range(NQ):
                    env.d.qpos.data[e * NQ + k] = Scalar[DT](ds0[e * (NQ + NV) + k])
                for k in range(NV):
                    env.d.qvel.data[e * NV + k] = Scalar[DT](ds0[e * (NQ + NV) + NQ + k])
            env.d.qpos.upload(ctx)
            env.d.qvel.upload(ctx)
            ctx.synchronize()
            env._run_fields_fk(ctx)
            ctx.synchronize()
        for e in range(LANES * 2):
            drawer_max[e] = 0.0
        var obs_row = List[Float64]()
        var _action = List[Float64](length=OSC_ACTION_DIM, fill=0.0)
        for e in range(len(ens)):
            ens[e].reset()
        for step in range(SETTLE_STEPS + max_steps):
            if recording and chunk == 0:
                # the picture BEFORE this step's action, every step
                _render_pack[E.MD, LANES, LIBERO_ACT_IMG_W, LIBERO_ACT_IMG_H,
                             False, True, 4, AIMG, ACAM, ANPIX](
                    ren_opt[0], ctx, env.d, env.mf, cam_idx, h_rgb, act_u8,
                )
                for cam in range(2):
                    var src = video_lane * AIMG + cam * ACAM
                    for y in range(VH):
                        var sy = y // VSCALE
                        for x in range(LIBERO_ACT_IMG_W * VSCALE):
                            var sx = x // VSCALE
                            var q = sy * LIBERO_ACT_IMG_W + sx
                            var o = (y * VW + cam * LIBERO_ACT_IMG_W * VSCALE + x) * 4
                            vframe[o] = UInt8(act_u8[src + 2 * ANPIX + q])
                            vframe[o + 1] = UInt8(act_u8[src + ANPIX + q])
                            vframe[o + 2] = UInt8(act_u8[src + q])
                            vframe[o + 3] = UInt8(255)
                recorder.add_frame_bgra(Int(vframe.unsafe_ptr()), VW, VH)
            var ap = act_h.unsafe_ptr()
            # ⚠ THE SETTLE STEPS ARE ZEROS EVEN WITH A POLICY — `metric.py`
            # steps its `dummy` through them, and the props are still falling.
            if step < SETTLE_STEPS or not have_policy:
                for k in range(LANES * OSC_ACTION_DIM):
                    ap[unsafe_offset=k] = Scalar[DT](0)
            elif have_knn:
                var t_pol = step - SETTLE_STEPS
                var query = act_exec == 0 or t_pol % act_exec == 0
                if query:
                    var tf0 = perf_counter_ns()
                    for e in range(LANES):
                        var r_task = row_task[lane_row[e] if lane_row[e] >= 0 else 0]
                        var q9 = List[Float64](length=KF, fill=0.0)
                        for k in range(AQP):
                            q9[k] = Float64(env.d.qpos.data[e * NQ + qadr9[k]])
                        if knn_vel:
                            for k in range(AQP):
                                q9[AQP + k] = q9[k] - (knn_prev[e * AQP + k] if t_pol > 0 else q9[k])
                        knn_dist[e] = _knn_chunk[AK, AA](
                            q9, KF, knn_q, knn_std, knn_mean,
                            knn_act, knn_left, knn_rows_of[r_task], knn_k,
                            act_chunk, e * AK * AA,
                        )
                    forward_ns += perf_counter_ns() - tf0
                for e in range(LANES):
                    for k in range(AQP):
                        knn_prev[e * AQP + k] = Float64(env.d.qpos.data[e * NQ + qadr9[k]])
                    if act_exec == 0:
                        ens[e].push(t_pol, act_chunk, e * AK * AA)
                        ens[e].action_at(t_pol, act_pred, 0)
                    else:
                        var pos = t_pol % act_exec
                        for k in range(AA):
                            act_pred[k] = act_chunk[e * AK * AA + pos * AA + k]
                    for k in range(OSC_ACTION_DIM):
                        var a = _clamp(Float64(act_pred[k]))
                        ap[unsafe_offset = e * OSC_ACTION_DIM + k] = Scalar[DT](a)
                        act_abs += abs(a)
                        act_words += 1
                        word_abs[k] += abs(a)
                    word_n += 1
            elif have_act:
                var t_pol = step - SETTLE_STEPS
                var query = act_exec == 0 or t_pol % act_exec == 0
                # 1. both cameras, every lane, from the state the lanes are in
                var tr0 = perf_counter_ns()
                if query:
                    _render_pack[E.MD, LANES, LIBERO_ACT_IMG_W, LIBERO_ACT_IMG_H,
                                 False, True, 4, AIMG, ACAM, ANPIX](
                        ren_opt[0], ctx, env.d, env.mf, cam_idx, h_rgb, act_u8,
                    )
                    for e in range(LANES):
                        for cam in range(LIBERO_ACT_N_CAM):
                            var dst = e * AIMG + cam * ACAM
                            normalize_camera_chw[LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W](
                                act_u8, dst, act_images, dst
                            )
                if (obs_store.byte_length() > 0 and not obs_checked
                        and query and chunk == 0):
                    obs_checked = True
                    print("  --check-obs, at the FIRST POLICY STEP (after the"
                          " settle steps):")
                    var q9 = List[Float64]()
                    for e in range(LANES):
                        for k in range(AQP):
                            q9.append(Float64(env.d.qpos.data[e * NQ + qadr9[k]]))
                    _check_obs[AIMG, ACAM, ANPIX](
                        obs_store, act_u8, lane_row, row_task, n_inits, q9,
                        obs_psnr, obs_psnr_ctrl,
                    )
                # 2. the nine proprio words and the lane's task one-hot,
                #    standardised as the fit was (`env.d.qpos` was downloaded
                #    after the previous step; the task is the row's)
                #    then the nine one-step differences (this step's joints
                #    minus the previous policy step's, zero at the first,
                #    as the store's episode-first row is), then the one-hot
                for e in range(LANES):
                    var r_task = row_task[lane_row[e] if lane_row[e] >= 0 else 0]
                    for k in range(AQ):
                        var raw: Scalar[DT]
                        if k < AQP:
                            raw = Scalar[DT](env.d.qpos.data[e * NQ + qadr9[k]])
                        elif k < 2 * AQP:
                            var cur = Float64(env.d.qpos.data[e * NQ + qadr9[k - AQP]])
                            raw = Scalar[DT](
                                cur - act_prev[e * AQP + k - AQP]
                                if (t_pol > 0 and not act_no_dq) else 0.0
                            )
                        else:
                            raw = Scalar[DT](1.0 if k - 2 * AQP == r_task else 0.0)
                        act_qpos[e * AQ + k] = (
                            raw - act_norm.qpos_mean[k]
                        ) / act_norm.qpos_std[k]
                    for k in range(AQP):
                        act_prev[e * AQP + k] = Float64(env.d.qpos.data[e * NQ + qadr9[k]])
                render_ns += perf_counter_ns() - tr0
                # 3. one forward at LANES when a query is due, then either
                #    each lane's ensemble or the chunk's next action
                if query:
                    var tf0 = perf_counter_ns()
                    if act_latent == "sample":
                        act_opt[0].predict_prior_sample(act_qpos, act_images, act_dummy, act_valid, act_chunk)
                    else:
                        act_opt[0].predict(act_qpos, act_images, act_dummy, act_valid, act_chunk)
                    forward_ns += perf_counter_ns() - tf0
                for e in range(LANES):
                    if act_exec == 0:
                        ens[e].push(t_pol, act_chunk, e * AK * AA)
                        ens[e].action_at(t_pol, act_pred, 0)
                    else:
                        var pos = t_pol % act_exec
                        for k in range(AA):
                            act_pred[k] = act_chunk[e * AK * AA + pos * AA + k]
                    denormalize(act_pred, 0, act_norm.action_mean,
                                act_norm.action_std, act_out, 0, AA)
                    for k in range(OSC_ACTION_DIM):
                        var a = _clamp(Float64(act_out[k]))
                        ap[unsafe_offset = e * OSC_ACTION_DIM + k] = Scalar[DT](a)
                        act_abs += abs(a)
                        act_words += 1
                        word_abs[k] += abs(a)
                    word_n += 1
            else:
                # the policy, lane by lane, on the env's own observation rows
                ctx.enqueue_copy(obs_h, env._obs)
                ctx.synchronize()
                var op = obs_h.unsafe_ptr()
                # every lane in ONE forward: the net's batch is LANES
                for e in range(LANES):
                    obs_row.clear()
                    for k in range(OD):
                        obs_row.append(Float64(op[unsafe_offset = e * OD + k]))
                    var z = List[Float64]()
                    norm.apply(obs_row, z)
                    for k in range(OD):
                        pol_x.data[e * OD + k] = Scalar[DT](z[k])
                net.forward["cpu", LANES](
                    TensorRefs[1](pol_x), pol_y, None
                )
                for e in range(LANES):
                    for k in range(OSC_ACTION_DIM):
                        var a = _clamp(
                            Float64(pol_y.data[e * OSC_ACTION_DIM + k])
                        )
                        ap[unsafe_offset = e * OSC_ACTION_DIM + k] = Scalar[DT](a)
                        act_abs += abs(a)
                        act_words += 1
            if trace_lane >= 0 and chunk == 0:
                var line = String("    t=") + _pad(String(step - SETTLE_STEPS), 4) + " a="
                for k in range(OSC_ACTION_DIM):
                    line += _pad(_fd(Float64(ap[unsafe_offset = trace_lane * OSC_ACTION_DIM + k]), 3), 7)
                line += " | fingers " + _f(Float64(env.d.qpos.data[trace_lane * NQ + qadr9[7]]), 6)
                line += " " + _f(Float64(env.d.qpos.data[trace_lane * NQ + qadr9[8]]), 6)
                if drawer_qadr >= 0:
                    line += " | drawer_mid " + _fd(Float64(env.d.qpos.data[trace_lane * NQ + drawer_qadr]), 3)
                line += " | q"
                for k in range(ARM_DOF):
                    line += " " + _fd(Float64(env.d.qpos.data[trace_lane * NQ + qadr9[k]]), 2)
                if grip_site >= 0:
                    env.d.site_xpos.download(ctx)
                    ctx.synchronize()
                    line += " | grip"
                    for k in range(3):
                        line += " " + _fd(Float64(env.d.site_xpos.data[trace_lane * NS * 3 + grip_site * 3 + k]), 3)
                if have_knn:
                    line += " | nn " + _fd(knn_dist[trace_lane], 2)
                print(line)
            ctx.enqueue_copy(env._action, act_h)
            var tp0 = perf_counter_ns()
            env.step_batch[LANES](ctx, UInt64(step + 1))
            ctx.synchronize()
            physics_ns += perf_counter_ns() - tp0
            env.d.meta.download(ctx)
            env.d.qpos.download(ctx)
            ctx.synchronize()
            if env.osc_singular_lanes(ctx) > 0:
                singular_steps += 1
            if drawer_qadr >= 0 and drawer_top_qadr >= 0:
                for e in range(LANES):
                    var top = abs(Float64(env.d.qpos.data[e * NQ + drawer_top_qadr]))
                    var mid = abs(Float64(env.d.qpos.data[e * NQ + drawer_qadr]))
                    if top > drawer_max[e * 2]:
                        drawer_max[e * 2] = top
                    if mid > drawer_max[e * 2 + 1]:
                        drawer_max[e * 2 + 1] = mid

            var want_host = check_lanes > 0 and (
                step < SETTLE_STEPS + 2 or step % 25 == 0
            )
            if want_host:
                env.d.xpos.download(ctx)
                env.d.xquat.download(ctx)
                env.d.site_xpos.download(ctx)
                env.d.contacts.download(ctx)
                ctx.synchronize()
            for e in range(LANES):
                var r = lane_row[e]
                if r < 0:
                    continue
                var mb = e * METADATA_SIZE
                var nc = Int(env.d.meta.data[mb + META_IDX_NUM_CONTACTS])
                if nc >= MC:
                    saturated += 1
                for k in range(NQ):
                    var q = Float64(env.d.qpos.data[e * NQ + k])
                    if q != q or q > 1.0e6 or q < -1.0e6:
                        nonfinite += 1
                var dev = Float64(env.d.meta.data[mb + META_IDX_GOAL_HELD]) > 0.5
                # ⚠⚠ SUCCESS AT ANY STEP, AND ONLY AFTER A STEP — `dones[k] =
                # dones[k] or done[k]`. The settle steps count: LIBERO's loop
                # runs them through `env.step` too. A goal met at step 3 and
                # abandoned at step 4 is a SUCCESS.
                if dev:
                    if not solved[r]:
                        first_step[r] = step
                    solved[r] = True
                    if step < SETTLE_STEPS:
                        at_settle[r] = True
                if want_host and (e % max(1, LANES // check_lanes) == 0):
                    var st = HostState(
                        List[Float64](), List[Float64](), List[Float64]()
                    )
                    for k in range(NB * 3):
                        st.xpos.append(Float64(env.d.xpos.data[e * NB * 3 + k]))
                    for k in range(NB * 4):
                        st.xquat.append(Float64(env.d.xquat.data[e * NB * 4 + k]))
                    for k in range(NS * 3):
                        st.site_xpos.append(
                            Float64(env.d.site_xpos.data[e * NS * 3 + k])
                        )
                    for k in range(NQ):
                        st.qpos.append(Float64(env.d.qpos.data[e * NQ + k]))
                    st.site_body = site_body_tab.copy()
                    st.site_quat = site_quat_tab.copy()
                    st.body_parent = body_parent_tab.copy()
                    var ncon = nc if nc < MC else MC
                    st.ncon = ncon
                    for c in range(ncon):
                        var cb = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
                        st.con_a.append(
                            Int(env.d.contacts.data[cb + CONTACT_IDX_BODY_A])
                        )
                        st.con_b.append(
                            Int(env.d.contacts.data[cb + CONTACT_IDX_BODY_B])
                        )
                    var host = eval_goal(
                        goals[row_task[r]], f, st, rsites, rcontact
                    )
                    eval_cmp += 1
                    if host != dev:
                        eval_bad += 1
                        if eval_bad <= 10:
                            print("   EVAL MISMATCH row", r, names[row_task[r]],
                                  "step", step, ": device", dev, "host", host)
        if drawer_qadr >= 0 and drawer_top_qadr >= 0 and have_policy:
            var dline = String("    drawer lanes (task 0/1), largest |top| |mid| opening:")
            var any_drawer = False
            for e in range(LANES):
                var r = lane_row[e]
                if r < 0 or (row_task[r] != 0 and row_task[r] != 1):
                    continue
                any_drawer = True
                dline += " [" + String(e) + ": " + _fd(drawer_max[e * 2], 3) + " " + _fd(drawer_max[e * 2 + 1], 3) + "]"
            if any_drawer:
                print(dline)
        print("  chunk", chunk, "done — rows", base, "..",
              (base + LANES - 1) if base + LANES <= n_rows else n_rows - 1,
              flush=True)
        chunk += 1

    if recording:
        recorder.stop()
        print("  video :", recorder.frames_written(), "frames ->", video_path)

    # ── the report ────────────────────────────────────────────────────────
    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
    print()
    if have_table:
        ref report = report_opt[0]
        for r in range(n_rows):
            report.record(r, solved[r])
        report.show(
            String("LIBERO success — ") + family
            + (", ACT " + act_dir if have_act else
               (", policy " + policy_path if have_bc else ", null policy"))
        )
    else:
        var per_task = List[Int](length=n_tasks, fill=0)
        for r in range(n_rows):
            if solved[r]:
                per_task[row_task[r]] += 1
        print("  SAMPLED success (NOT a benchmark number):")
        for ti in range(n_tasks):
            print("   ", _pad(names[ti], 58), per_task[ti], "/", n_inits)
    var n_solved = 0
    var n_settle = 0
    for r in range(n_rows):
        if solved[r]:
            n_solved += 1
        if at_settle[r]:
            n_settle += 1
    print()
    print("  rows", n_rows, "| solved", n_solved, "| of those, during the",
          SETTLE_STEPS, "settle steps:", n_settle)
    print("  success word: ", eval_cmp, "host comparisons,", eval_bad,
          "disagreeing")
    if have_policy:
        print("  policy: mean |action|",
              act_abs / Float64(act_words) if act_words > 0 else 0.0, "over",
              act_words, "words")
    if have_act:
        print("  act   : fitted on", act_norm.store, "| cameras + qpos",
              Float64(render_ns) / 1e9, "s | forward", Float64(forward_ns) / 1e9,
              "s | physics (step_batch)", Float64(physics_ns) / 1e9,
              "s over the run")
        if len(obs_psnr_pre) > 0:
            print("  obs   : the demo's OWN frame-0 state, rendered here, vs the"
                  " store's frame 0 — own demo", _mean(obs_psnr_pre),
                  "dB | next demo (control)",
                  _mean(obs_psnr_pre_ctrl), "dB over", len(obs_psnr_pre),
                  "lane-cameras")
        if len(obs_psnr) > 0:
            print("  obs   : first policy step vs the store's frame 0 — own demo",
                  _mean(obs_psnr), "dB | next demo (control)",
                  _mean(obs_psnr_ctrl), "dB over", len(obs_psnr),
                  "lane-cameras (" + obs_store + ")")
        if word_n > 0:
            print("  per word            policy mean|a|   store mean     store std")
            for k in range(OSC_ACTION_DIM):
                print("    " + _pad(String(k), 8) + " " + _pad(String(word_abs[k] / Float64(word_n)), 16)
                      + " " + _pad(String(Float64(act_norm.action_mean[k])), 14)
                      + " " + String(Float64(act_norm.action_std[k])))
    print("  contacts saturated lane-steps", saturated, "| non-finite", nonfinite,
          "| singular steps", singular_steps)
    print("  wall", elapsed, "s for", n_rows, "episodes of", SETTLE_STEPS
          + max_steps, "steps")

    # ── the verdict ───────────────────────────────────────────────────────
    var fails = List[String]()
    if n_rows == 0:
        fails.append("no rows evaluated")
    if eval_cmp == 0 and check_lanes > 0:
        fails.append("the host never checked a success word")
    if eval_bad > 0:
        fails.append(String(eval_bad) + " success words disagree with the host")
    if have_policy and (act_words == 0 or act_abs == 0.0):
        fails.append(
            "the policy emitted only zeros — it is the null run wearing a"
            " checkpoint"
        )
    if nonfinite > 0:
        fails.append(String(nonfinite) + " non-finite qpos words")
    if saturated > 0:
        fails.append(String(saturated) + " lane-steps saturated max_contacts")
    # ⚠⚠ THE L6 GATE APPLIES TO THE NULL RUN ONLY: the null action must solve
    # nothing (a task above 0 is a goal defect, not a policy result). With a
    # policy the solved count is THE RESULT, and gating it at 0 would fail the
    # run for succeeding.
    if n_solved > 0 and not have_policy:
        fails.append(
            String(n_solved) + " episodes solved by the NULL action"
        )
    print()
    if len(fails) > 0:
        for i in range(len(fails)):
            print("  FAIL:", fails[i])
        raise Error(family + ": " + String(len(fails)) + " check(s) failed")
    # ⚠ THE MODE IS ON THE VERDICT LINE. A sampled run's PASS, pasted on its
    # own, is indistinguishable from the benchmark's otherwise — and the whole
    # point of the frozen table is that a rate over states the run chose for
    # itself is not comparable with anything.
    print("=== PASS —", family,
          "(LIBERO's frozen inits)" if have_table
          else "(SAMPLED inits — NOT a benchmark number)",
          ", success " + String(n_solved) + " / " + String(n_rows) if have_policy
          else ", null rate 0", "===")


def main() raises:
    var args = argv()
    var n_inits = LIBERO_N_EVAL
    var max_steps = LIBERO_MAX_STEPS
    var check_lanes = 2
    var sampled = False
    var policy_path = String("")
    var act_dir = String("")
    var act_exec = 0
    var obs_store = String("")
    var video_path = String("")
    var video_lane = 0
    var trace_lane = -1
    var knn_store = String("")
    var knn_k = 5
    var knn_vel = False
    var demo_init = String("")
    var only_task = -1
    var act_ckpt = String("best")
    var act_m = Float64(ACT_TEMPORAL_ENSEMBLE_M)
    var act_latent = String("zero")
    var act_no_dq = False
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--inits" and i + 1 < len(args):
            n_inits = Int(String(args[i + 1]))
            i += 1
        elif s == "--steps" and i + 1 < len(args):
            max_steps = Int(String(args[i + 1]))
            i += 1
        elif s == "--check-lanes" and i + 1 < len(args):
            check_lanes = Int(String(args[i + 1]))
            i += 1
        elif s == "--policy" and i + 1 < len(args):
            policy_path = String(args[i + 1])
            i += 1
        elif s == "--act" and i + 1 < len(args):
            act_dir = String(args[i + 1])
            i += 1
        elif s == "--act-exec" and i + 1 < len(args):
            act_exec = Int(String(args[i + 1]))
            i += 1
        elif s == "--video" and i + 1 < len(args):
            video_path = String(args[i + 1])
            i += 1
        elif s == "--video-lane" and i + 1 < len(args):
            video_lane = Int(String(args[i + 1]))
            i += 1
        elif s == "--trace-lane" and i + 1 < len(args):
            trace_lane = Int(String(args[i + 1]))
            i += 1
        elif s == "--knn":
            knn_store = String("build/demos/" + FAMILY + ".lowdim.h5")
            if i + 1 < len(args) and not String(args[i + 1]).startswith("--"):
                knn_store = String(args[i + 1])
                i += 1
        elif s == "--knn-vel":
            knn_vel = True
        elif s == "--act-no-dq":
            act_no_dq = True
        elif s == "--act-latent" and i + 1 < len(args):
            act_latent = String(args[i + 1])
            if act_latent != "zero" and act_latent != "sample":
                raise Error("--act-latent must be zero (the paper) or sample (a prior draw)")
            i += 1
        elif s == "--act-m" and i + 1 < len(args):
            act_m = Float64(String(args[i + 1]))
            i += 1
        elif s == "--act-ckpt" and i + 1 < len(args):
            act_ckpt = String(args[i + 1])
            i += 1
        elif s == "--task" and i + 1 < len(args):
            only_task = Int(String(args[i + 1]))
            i += 1
        elif s == "--demo-init":
            demo_init = String("build/demos/" + FAMILY + ".lowdim.h5")
            if i + 1 < len(args) and not String(args[i + 1]).startswith("--"):
                demo_init = String(args[i + 1])
                i += 1
        elif s == "--knn-k" and i + 1 < len(args):
            knn_k = Int(String(args[i + 1]))
            i += 1
        elif s == "--check-obs":
            obs_store = String(LIBERO_ACT_STORE_RENDERED)
            if i + 1 < len(args) and not String(args[i + 1]).startswith("--"):
                obs_store = String(args[i + 1])
                i += 1
        elif s == "--sampled":
            sampled = True
        else:
            raise Error(
                "libero eval batched: unknown argument '" + s + "' (--inits N,"
                " --steps N, --check-lanes K, --sampled, --policy RUN_ID|PATH,"
                " --act DIR, --act-exec N, --check-obs [STORE], --video F.mp4,"
                " --video-lane L, --trace-lane L, --knn [STORE], --knn-k N, --knn-vel,"
                " --demo-init [STORE], --task T, --act-ckpt best|last, --act-m M,"
                " --act-latent zero|sample, --act-no-dq)"
            )
        i += 1

    if policy_path != "":
        # a RUN ID -> its `checkpoints/last.ckpt` (what libero_bc_train
        # writes, `.norm` sidecar beside it); a file is used as is.
        policy_path = resolve_checkpoint(policy_path, String("last"))
    if knn_k < 1 or knn_k > 64:
        raise Error("--knn-k must be in [1, 64]")
    if act_exec < 0 or act_exec > LIBERO_ACT_K:
        raise Error("--act-exec must be in [0, " + String(LIBERO_ACT_K)
                    + "] (0 = the temporal ensemble)")
    comptime if FAMILY == "libero_goal":
        run[LiberoGoalPlacement, LiberoGoalModel](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store, video_path, video_lane, trace_lane, knn_store, knn_k,
            knn_vel, demo_init, only_task, act_ckpt, act_m, act_latent,
            act_no_dq,
        )
    elif FAMILY == "libero_object":
        run[LiberoObjectPlacement, LiberoObjectModel](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store, video_path, video_lane, trace_lane, knn_store, knn_k,
            knn_vel, demo_init, only_task, act_ckpt, act_m, act_latent,
            act_no_dq,
        )
    elif FAMILY == "libero_spatial":
        run[LiberoSpatialPlacement, LiberoSpatialModel](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store, video_path, video_lane, trace_lane, knn_store, knn_k,
            knn_vel, demo_init, only_task, act_ckpt, act_m, act_latent,
            act_no_dq,
        )
    elif FAMILY == "libero_kitchen_scene3":
        run[LiberoKitchenScene3Placement, LiberoKitchenScene3Model](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store, video_path, video_lane, trace_lane, knn_store, knn_k,
            knn_vel, demo_init, only_task, act_ckpt, act_m, act_latent,
            act_no_dq,
        )
    elif FAMILY == "libero_kitchen_scene5":
        run[LiberoKitchenScene5Placement, LiberoKitchenScene5Model](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store, video_path, video_lane, trace_lane, knn_store, knn_k,
            knn_vel, demo_init, only_task, act_ckpt, act_m, act_latent,
            act_no_dq,
        )
    else:
        comptime assert False, (
            "libero_eval_batched: FAMILY is not one of the imported families —"
            " add its branch (see the FAMILY docstring)"
        )
