# +--------------------------------------------------------------------------+ #
# | SmolVLA on the PHYSICAL SO-101 — chunk-at-a-time, from two live cameras
# +--------------------------------------------------------------------------+ #
"""Drive the follower arm from a fine-tuned SmolVLA, on the real rig.

    pixi run build-opencv                     # ONCE
    pixi run build-serial                     # ONCE

    # SAFE BY DEFAULT: reads the arm and the cameras, runs the policy, prints
    # every command it WOULD have sent, and never energises anything.
    pixi run -e jetson smolvla-deploy-jetson -- --project so101-tower \\
        --devices /dev/soarm_cam_overhead,/dev/soarm_cam_wrist

    # a fine-tune run's own weights: a RUN ID (its checkpoints/last.ckpt) or a
    # .ckpt path; absent it, $SMOLVLA_CKPT, then the promoted policies/smolvla.ckpt
    ... --ckpt <run_id> --stats <meta/stats.json>

    # --arm is what actually moves the robot. Be at the desk, hand on the power.
    ... --arm --seconds 30

⚠⚠ **`--arm` MOVES THE FOLLOWER, AND NOTHING ELSE DOES.** The opt-in is a scar
`record.mojo` earned on 2026-08-31; `docs/SO101_SERIAL_LAYER.md` §safety.

⚠⚠ **A `finally` DOES NOT COVER AN ABORT OR A SIGNAL.** If this dies hard the
follower is left holding its pose — recovery is `pixi run soarm-torque-off`
and the power switch.

## Why this loop is shaped differently from ACT's

`act_so101_deploy_real.mojo` queries EVERY 30 Hz step and blends ~58
overlapping chunks with a temporal ensemble, because its forward is 27 ms and
it can. SmolVLA cannot and does not need to:

  * one query is hundreds of milliseconds (`smolvla_so101_latency_probe.mojo`
    is the prerequisite measurement — run it on the board before believing
    any of this);
  * one CHUNK is 50 steps at 30 Hz = **1.667 s of motion**, and
    `n_action_steps` is the whole chunk. SmolVLA is DESIGNED to be executed
    open-loop chunk-to-chunk.

So the shape is: query once, EXECUTE the chunk on the 30 Hz grid, query again
before it runs out. There is no temporal ensemble — the chunks do not overlap.

⚠ THE 30 Hz ACTION GRID IS KEPT, AND IT IS WHAT MAKES THE LATENCY HONEST. A
chunk requested for the observation at grid step `t_obs` describes steps
`t_obs .. t_obs+49`, exactly as in training. The query takes real time, so when
it returns the clock has moved on; the loop then indexes the chunk at
`now - t_obs` and the steps that elapsed during inference are SKIPPED, not
replayed late. The latency is not hidden, it is INDEXED — the same rule as
ACT's `t_cmd`, and the report's `skipped at handover` line is what it costs.

⚠ `--exec-steps` TRADES REACTIVITY AGAINST QUERY RATE, and both ends are
defensible. At 50 the arm runs a full 1.667 s on one observation — maximum
open-loop, minimum queries. Lower re-queries sooner on fresher pixels, at more
inference. It is a knob because the right value depends on a latency this file
refuses to guess.

## What is checked BEFORE anything is armed

The same four as the ACT deployment, for the same reasons:

1. **the arm is somewhere the demonstrations went** — the state box from
   `meta/stats.json`, printed per joint against the follower's present pose;
2. **the commanded action is clamped to the demonstrated action box** — the
   policy has never been asked what to do outside it and its answer there is
   not evidence of anything;
3. **the cameras are the right way round** — slot order is part of the
   checkpoint, and a swap feeds the policy a world it has never seen;
4. **the first query does not happen inside the loop.** Warm-up queries are
   run before arming: the first one compiles kernels, and on the Orin that was
   9.9 s for ACT. Paid inside the loop it is one command arriving seconds late
   with the arm already energised.

⚠ SAFETY IS NOT REIMPLEMENTED HERE. The step clamp and the calibrated range
live in `SO101Arm.write_goals`; the shutdown that ramps home and refuses to
drop the arm is `robot/so101/deploy_shutdown.mojo`, shared with ACT.
"""

from std.math import cos, log, sin, sqrt
from std.os import getenv
from std.os.path import exists
from std.sys import argv
from std.sys.defines import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.deep_agents.act.config import SO101_FPS
from noeira.deep_agents.act.inference import TemporalEnsemble
from noeira.deep_agents.smolvla.finetune import load_trainables
from noeira.deep_agents.smolvla.heads import (
    SMOLVLA_ACTION_DIM,
    SMOLVLA_EXPERT_W,
    SMOLVLA_STATE_DIM,
)
from noeira.deep_agents.smolvla.expert import EXPERT_FF
from noeira.deep_agents.smolvla.text import (
    SMOLLM_DIM,
    SMOLLM_KV_W,
    SMOLLM_LAYERS,
)
from noeira.deep_agents.smolvla.observation import (
    fill_camera_images, fill_siglip_frames, siglip_frames_into_slot,
)
from noeira.deep_agents.smolvla.policy import SmolVLAPolicy
from noeira.deep_agents.smolvla.query_worker import (
    QW_DROPPED,
    QW_FAILED,
    QW_N_CELLS,
    QW_POLLS,
    QW_QUERY_US,
    QW_READY,
    QW_SERVED,
    QW_STATE,
    QW_SUBMIT_US,
    SmolVLAQueryWorker,
)
from noeira.core.concurrent.block import SharedBlock
from noeira.core.concurrent.ring import SharedRing
from noeira.core.concurrent.worker import BackgroundThread
from noeira.core.run import resolve_checkpoint
from noeira.deep_agents.smolvla.recording import SO101_N_LANG, SO101_TASKS
from noeira.deep_agents.smolvla.tasks import TaskTokens
from noeira.io.fileio import StdinReader, stdin_is_tty
from noeira.io.hf import hf_download_file, HF_MODEL
from noeira.io.json import JsonDoc, load_json
from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.robot.so101 import SO101Arm, SO101_N, joint_name
from noeira.robot.so101.deploy_shutdown import return_and_release
from noeira.robot.so101.ports import follower_port, port_refusal
from noeira.utils.fmt import col, fixed, pad_left, pad_right
from noeira.vision.camera_thread import CameraReader, parse_camera_specs
from noeira.vision.resize_pad import SIGLIP_INPUT


comptime TARGET: StaticString = "gpu" if is_defined["SMOLVLA_GPU"]() else "cpu"
"""⚠⚠ THE INFERENCE DEVICE, CHOSEN AT BUILD TIME — `-DSMOLVLA_GPU=1`.

⚠ ONE TARGET PER BINARY, AND HERE THAT IS NOT MERELY A COMPILE-TIME
PREFERENCE: every layer is instantiated per target and the weights are 3.2 GB
host+device. Building both would be 6.4 GB — the mistake that took a 16 GiB
laptop down once already (`smolvla_so101_latency_probe.mojo`)."""

comptime BASE_REPO = "lerobot/smolvla_base"
comptime N_CAM = 2
comptime N_LANG = SO101_N_LANG
comptime CHUNK = 50
comptime STEPS = 10
"""Euler denoising steps at INFERENCE. Training denoises once; this does not.

⚠ FOUR WAS MEASURED BETTER ON THE ARM and is not the default, because
"better" here means latency and nothing about success. At 4 the query is
491 ms against 663, the chunk lands 0.17 s fresher (skip 16.8 grid steps
against 21.9), and the trajectory kept its shape — net displacement 41 deg,
within the run-to-run spread of 37-84 at 10 steps. What was NOT measured is
whether a coarser integration of the same flow costs accuracy where it
matters, which would take a success rate over many runs to see. Change it
here and rebuild to test it:

    sed -i 's/^comptime STEPS = 10$/comptime STEPS = 4/' <this file>

⚠⚠ THE BEST CONFIGURATION MEASURED, 19 Sep, all four together:

    --threaded --smooth 9 --ensemble   (with STEPS = 4)

    handover jump   63.6 -> 15.1 deg        ensemble 1.82 chunks/command
    control rate    17.4 -> 27.2 Hz         query    2329 -> 491 ms
    motion          matched the demonstrations (step 1.65 vs 1.59 deg)

    It touched the cube repeatedly and did not grasp it. Every control-side
    hypothesis is now measured and closed; what remains is the policy."""
comptime RDIM = 6
"""The robot's real joint count. ⚠ NOT `SMOLVLA_ACTION_DIM` (32), which is the
padding the checkpoint was trained with — `select_action` drops the padding and
returns `CHUNK x stats.action_dim()`."""

comptime Pol = SmolVLAPolicy[N_CAM, N_LANG, CHUNK, STEPS, 1]
comptime QWorker = SmolVLAQueryWorker[
    N_CAM, N_LANG, CHUNK, STEPS, RDIM, TARGET
]
"""The same policy, built and queried on its own thread under `--threaded`."""

comptime CAM_W = 640
comptime CAM_H = 480
"""⚠⚠ THE CAMERA'S NATIVE SIZE, AND IT MUST BE THE ONE THE DEMONSTRATIONS WERE
RECORDED AT. The only resize is SmolVLA's own `resize_with_pad` from this size
to 512x512 (`camera_frame_to_siglip`), which is why the recording is imported
at 480x640 for SmolVLA and 240x320 for ACT. Feeding it a pre-shrunk frame would
resample twice. Since 20 Sep that resize runs ON THE CAMERA THREAD
(`CameraReader(..., siglip=512)`), the same function on the same frame; the
control loop takes the newest finished block."""

comptime WARMUP_QUERIES = 2
"""⚠ FEWER THAN ACT'S FIVE, because one query here costs hundreds of ms rather
than 27. Two still separates the first (which compiles kernels) from the
steady state, which is the whole point."""

comptime MAX_STEP_TICKS = 80
comptime TRACK_STEP_TICKS = 512
"""The same two-phase clamp the recorders and the ACT deployment use, so the
arm a policy drives moves like the arm that recorded its demonstrations."""

comptime ACTION_BOX_MARGIN = 0.05
"""5% of each joint's demonstrated span, added around the action box. The box
edges are real poses the arm reached, so clamping exactly at them would fight
the policy at the extremes of a legitimate motion."""

comptime DEFAULT_TASKS = SO101_TASKS


def _stats_vec(
    ref doc: JsonDoc, key: String, field: String, mut out: List[Float64]
) raises -> Bool:
    """One vector out of `meta/stats.json`, False when the key is absent.

    ⚠ ABSENT IS NOT AN ERROR HERE, unlike mean/std. `min`/`max` are what the
    safety box is built from and LeRobot writes them, but a stats file without
    them is legible — so this reports the absence and the caller decides,
    rather than refusing to start a DRY RUN over a missing guard.
    """
    var root = doc.root()
    var node = doc.field(root, key)
    if node < 0:
        return False
    var arr = doc.field(node, field)
    if arr < 0:
        return False
    out.clear()
    for i in range(doc.size(arr)):
        out.append(doc.number(doc.at(arr, i)))
    return len(out) > 0


def main() raises:
    var arm_it = False
    var force = False
    var project = String("so101-tower")
    var ckpt = String("")
    var stats_path = String("")
    var tasks_path = String(DEFAULT_TASKS)
    var task_index = 0
    var seconds = 30
    var exec_steps = CHUNK
    var step_ticks = MAX_STEP_TICKS
    var devices = List[String]()
    var cam_fourcc = String("")
    var port_arg = String("")
    var do_return = True
    # ⚠ OPT-IN WHILE IT IS NEW. Without it this program keeps the shape that
    # has been on the arm all week: the query on the control thread.
    var threaded = False
    # ⚠⚠ A LOW-PASS OVER THE CHUNK'S WAYPOINTS, OFF BY DEFAULT. Measured on
    # the board, the policy's chunk covers 324.9 deg of path to reach a net
    # displacement of 68.1 — while the demonstrations cover 84.9 to reach
    # 62.0. It arrives in the RIGHT PLACE (the normalisation is fine) along a
    # path 3.8x too long, which is a tremor of a few degrees at every
    # waypoint. A hand shaking +-3 deg as it closes on a cube misses it.
    #
    # ⚠ THIS CHANGES WHAT THE POLICY COMMANDS, so it stays opt-in and the
    # report prints the path and net BEFORE and AFTER: a filter that also
    # shortens the net displacement is eating the motion, not the tremor.
    #
    # MEASURED ON THE BOARD, 19 Sep, against the demonstrations' own figures
    # (`tools/so101/demo_step_stats.mojo`: step 1.59 deg, wiggle 1.37x):
    #
    #   --smooth    step within chunk   wiggle   net    handover jump
    #   (off)       6.92 deg            4.77x    68.1   57.9 deg
    #   5           3.27                1.66x    82.5   31.5
    #   9           1.65                1.43x    52.6   22.7
    #
    # Nine reproduces the demonstrated motion statistics almost exactly and
    # costs 5% of the net displacement. It did not make the grasp succeed,
    # which is the useful part of the result: the arm now moves like the
    # demonstrations and still misses, so what remains is WHERE the policy
    # aims — a fine-tune that was still improving at 2000 steps, and 50
    # episodes — not how it moves.
    var smooth_n = 1
    # ⚠⚠ ACT'S TEMPORAL ENSEMBLE, ON SMOLVLA'S CHUNKS. Off by default.
    #
    # Smoothing fixed the tremor WITHIN a chunk; the handover between chunks
    # is a separate discontinuity (22.7 deg mean even at --smooth 9), because
    # two chunks predicted from observations ~0.9 s apart disagree about the
    # same instant. Averaging them is what the ACT deployment already does on
    # this rig, and `TemporalEnsemble` is that code — reused rather than
    # written twice.
    #
    # ⚠ THE WINDOW IS SPARSE HERE. ACT queries every ~3 grid steps; this
    # queries every ~28, so about TWO chunks cover any instant instead of
    # dozens. The report prints the measured figure — a `chunks per command`
    # near 1.0 means the ensemble is doing nothing and the handover is
    # unchanged.
    var ensemble = False
    var fused_vision = True
    """The SigLIP towers' attention through `CrossAttention`'s fused
    inference path (no scores, no cache, no packs) — the default since
    20 Sep. On NVIDIA at SigLIP's shape that path is MAX's own FA2 flash
    attention (21 Sep): 1.26 ms per layer against 4.1 for our kernel and 8.4
    for the two-pass path, with TF32 tensor-core matmuls — 3.7e-3 std units
    against float64 per layer, the band the GEMMs already run in on CUDA.
    `--no-fused-vision` runs the two-pass fp32 path instead, for an A/B on
    the query cost or on the chunk; `smolvla_so101_latency_probe.mojo`
    prints the chunk delta between the two paths in the action's units."""
    var sync = False
    """⚠⚠ `--sync` IS THE REFERENCE'S EVALUATION LOOP, and the measurement
    that asked for it is the two 20 Sep runs with the accum-64 checkpoints.
    Pipelined, both aimed at the cube within 3 s and then OSCILLATED with
    growing amplitude — `shoulder_lift` alternating -77 / +25 / -77 / +27 /
    -60 … -102 at exactly the query period (0.7 s), `step at handover`
    35 deg mean, 80 worst. That is a delayed-feedback loop, not a policy
    preference: the observation for chunk k+1 is taken while the arm is
    mid-way through chunk k, the answer lands 700 ms later with the arm
    somewhere else, and it is executed from step ~21 — the part of the plan
    that assumes the arm followed THAT chunk's first 20 steps, which it never
    did. Each handover is a jump the servo chases; the next observation sees
    an arm that overshot, and the correction flips sign. Gain ~1, delay
    0.7 s, period 1.4 s, growing.

    LeRobot's `record`-with-policy loop never pipelines: `select_action`
    runs inference when its queue is empty, the robot HOLDS its last command
    for the duration, and the 50 actions then execute one per tick from step
    0. Every chunk starts from the exact state it was predicted for. The
    "fast, freeze, fast" pattern in every SO-101 SmolVLA video is that loop,
    and it is what the fine-tune's episodes look like too — the recording
    never had a 35-degree jump in it.

    So `--sync`: start a query only when the chunk is exhausted, block on it
    (the servo holds the last waypoint), re-base the new chunk to NOW and
    execute it from step 0. No skip, no lead, no overlap — `--ensemble` is
    ignored with a note. `--exec-steps` still applies: fewer steps per
    chunk means fresher pixels per pause."""

    var args = argv()
    for i in range(len(args)):
        var a = String(args[i])
        if a == "--arm" or a == "--live":
            arm_it = True
        elif a == "--force":
            force = True
        elif a == "--threaded":
            threaded = True
        elif a == "--smooth" and i + 1 < len(args):
            smooth_n = Int(String(args[i + 1]))
        elif a == "--ensemble":
            ensemble = True
        elif a == "--sync":
            sync = True
        elif a == "--no-fused-vision":
            fused_vision = False
        elif a == "--project" and i + 1 < len(args):
            project = String(args[i + 1])
        elif a == "--ckpt" and i + 1 < len(args):
            ckpt = String(args[i + 1])
        elif a == "--stats" and i + 1 < len(args):
            stats_path = String(args[i + 1])
        elif a == "--tasks" and i + 1 < len(args):
            tasks_path = String(args[i + 1])
        elif a == "--task-index" and i + 1 < len(args):
            task_index = Int(String(args[i + 1]))
        elif a == "--seconds" and i + 1 < len(args):
            seconds = Int(String(args[i + 1]))
        elif a == "--exec-steps" and i + 1 < len(args):
            exec_steps = Int(String(args[i + 1]))
        elif a == "--step" and i + 1 < len(args):
            step_ticks = Int(String(args[i + 1]))
        elif a == "--port" and i + 1 < len(args):
            port_arg = String(args[i + 1])
        elif a == "--fourcc" and i + 1 < len(args):
            cam_fourcc = String(args[i + 1])
        elif a == "--no-return":
            do_return = False
        elif a == "--devices" and i + 1 < len(args):
            devices = parse_camera_specs(String(args[i + 1]))

    if len(devices) == 0:
        devices.append(String("0"))
        devices.append(String("1"))
    if len(devices) != N_CAM:
        raise Error(
            "smolvla deploy: the policy takes " + String(N_CAM)
            + " cameras but " + String(len(devices)) + " device(s) were given"
        )
    if exec_steps < 1 or exec_steps > CHUNK:
        raise Error(
            "smolvla deploy: --exec-steps must be 1.." + String(CHUNK)
            + " (the chunk is " + String(CHUNK) + " steps = "
            + fixed(Float64(CHUNK) / Float64(SO101_FPS), 2) + " s of motion)"
        )
    if sync and ensemble:
        # No two chunks ever cover the same instant under --sync, so the
        # blend would have one contributor everywhere — `ensemble = 1.00`.
        print("⚠ --sync: chunks do not overlap; --ensemble ignored")
        ensemble = False
    if ckpt == "":
        ckpt = getenv("SMOLVLA_CKPT", String(""))
    if ckpt != "":
        # A RUN ID -> its `checkpoints/last.ckpt` (the fine-tune's deploy
        # checkpoint, see its header); a file is used as is.
        ckpt = resolve_checkpoint(ckpt, String("last"))
    if ckpt == "":
        var promoted = "projects/" + project + "/policies/smolvla.ckpt"
        if exists(promoted):
            ckpt = promoted
    if ckpt == "":
        raise Error(
            "smolvla deploy: no fine-tuned weights. Pass --ckpt <run_id|file>, set"
            " $SMOLVLA_CKPT, or promote one into projects/" + project
            + "/policies/smolvla.ckpt.\n  ⚠ The BASE checkpoint alone has never"
            " seen this robot; running it would not be a test of anything."
        )
    if not exists(ckpt):
        raise Error("smolvla deploy: no such checkpoint: " + ckpt)
    if stats_path == "":
        stats_path = getenv("SMOLVLA_STATS", String(""))
    if stats_path == "" or not exists(stats_path):
        raise Error(
            "smolvla deploy: --stats <meta/stats.json> is required.\n"
            "  ⚠⚠ IT MUST BE THE SAME FILE THE FINE-TUNE USED. lerobot's"
            " population std and this repo's sample std differ by"
            " sqrt(N/(N-1)); normalizing with one while the weights were fit"
            " with the other is a silent scale error on every joint."
        )
    if not exists(tasks_path):
        raise Error(
            "smolvla deploy: no task table at " + tasks_path + " — it carries"
            " the PRE-TOKENISED instruction (there is no tokenizer at"
            " runtime). Pass --tasks <file>."
        )

    print("=" * 74)
    if arm_it:
        print("SmolVLA / SO-101 — CLOSED LOOP ON THE REAL ARM   [ARMED]")
    else:
        print("SmolVLA / SO-101 — DRY RUN (no torque, no goals written)")
        print("  pass --arm to actually move the follower")
    print("=" * 74)

    # ── the instruction ───────────────────────────────────────────────────
    var tasks = TaskTokens(tasks_path)
    var ids = tasks.for_index(task_index)
    if len(ids) != N_LANG:
        raise Error(
            "smolvla deploy: task " + String(task_index) + " is "
            + String(len(ids)) + " tokens, the policy was built for "
            + String(N_LANG)
        )
    print('instruction  "' + tasks.texts[task_index] + '"')
    print("             " + String(len(ids)) + " tokens, prefix P = "
          + String(Pol.P))

    # ── the policy ────────────────────────────────────────────────────────
    # ⚠⚠ ONE OF TWO SHAPES, AND THEY DO NOT OVERLAP. `--threaded` builds the
    # policy on the QUERY THREAD and this thread never touches the GPU at all
    # (no context, no 3.2 GB of layers here); otherwise the policy is built
    # here and queried inline, which is what the arm has run all week.
    var dev_ctx = Optional[DeviceContext](None)
    var pol_opt = Optional[Pol](None)
    var req_ring = SharedRing(capacity=2, slot_bytes=QWorker.REQ_BYTES)
    var rsp_ring = SharedRing(capacity=2, slot_bytes=QWorker.RSP_BYTES)
    var qcells = SharedBlock(QW_N_CELLS)
    var worker = Optional[BackgroundThread[QWorker]](None)

    if sync:
        print("loop         --sync: hold during the query, execute each chunk"
              " from step 0 (the reference's evaluation loop)")
    print(
        "vision       SigLIP attention "
        + ("FUSED (MAX FA2, TF32 matmuls, on NVIDIA at this shape; no cache)"
           if fused_vision else "two-pass fp32 (--no-fused-vision)")
    )
    if threaded:
        print("device       the query runs on its own thread (--threaded)")
        worker = BackgroundThread(
            QWorker(
                req_ring, rsp_ring, qcells, String(BASE_REPO), ckpt,
                stats_path, tasks_path, task_index, WARMUP_QUERIES,
                fused_vision,
            )
        )
        # ⚠ WAIT FOR IT, AND FOR ITS WARM-UP. The worker downloads, builds
        # 3.2 GB of layers and runs the kernel-compiling first query before it
        # publishes READY. Arming while it is still starting would command the
        # arm from an empty chunk.
        while qcells.acquire_load(QW_STATE) == 0:
            pass
        if qcells.acquire_load(QW_STATE) == QW_FAILED:
            raise Error(
                "smolvla deploy: the query thread failed to start — its own"
                " message is above this line"
            )

    comptime if TARGET != "cpu":
        if not threaded:
            dev_ctx = DeviceContext()
            print("device       " + String(dev_ctx.value().name())
                  + "  (-DSMOLVLA_GPU=1)")
    else:
        if not threaded:
            print("device       CPU")

    print("fine-tuned   " + ckpt)
    print("statistics   " + stats_path)
    if not threaded:
        print("base         " + String(BASE_REPO) + "  (~907 MB, cached)")
        var base = hf_download_file(
            String(BASE_REPO), String("model.safetensors"), HF_MODEL
        )
        print("             building every layer (3.2 GB host+device) ...")
        var pol = Pol.make[TARGET, Deterministic](dev_ctx)
        pol.load[TARGET](base, dev_ctx)

        # ⚠⚠ THE FINE-TUNE IS NOT SELF-CONTAINED. It holds the trainable set
        # only — the expert and the four action projections — so the base must
        # be loaded FIRST and this applied over it. Loading only this would
        # leave the vision tower and the language model at their
        # initialisation.
        var sp_frozen = Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM].make[
            TARGET, Deterministic
        ](dev_ctx)
        load_trainables[
            TARGET, SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
            SMOLLM_KV_W, SMOLVLA_ACTION_DIM,
        ](
            ckpt, pol.expert, pol.action_in, pol.time_mlp_in,
            pol.time_mlp_out, pol.action_out, sp_frozen, dev_ctx,
        )

        pol.load_stats(stats_path)
        if pol.stats.action_dim() != RDIM or pol.stats.state_dim() != RDIM:
            raise Error(
                "smolvla deploy: " + stats_path + " describes a "
                + String(pol.stats.state_dim()) + "-state / "
                + String(pol.stats.action_dim())
                + "-action robot, this build is "
                + String(RDIM) + "/" + String(RDIM)
            )
        # Inference only from here: the towers' attention fused (see
        # `SmolVLAPolicy.set_fused_vision_attention`) unless asked not to.
        pol.set_fused_vision_attention(fused_vision)
        pol_opt = pol^

    # ── the safety boxes, from the SAME stats file ────────────────────────
    var doc = load_json(stats_path)
    var a_lo = List[Float64]()
    var a_hi = List[Float64]()
    var q_lo = List[Float64]()
    var q_hi = List[Float64]()
    var have_box = (
        _stats_vec(doc, String("action"), String("min"), a_lo)
        and _stats_vec(doc, String("action"), String("max"), a_hi)
        and _stats_vec(doc, String("observation.state"), String("min"), q_lo)
        and _stats_vec(doc, String("observation.state"), String("max"), q_hi)
    )
    if have_box:
        for j in range(RDIM):
            var pad = ACTION_BOX_MARGIN * (a_hi[j] - a_lo[j])
            a_lo[j] -= pad
            a_hi[j] += pad
    elif arm_it and not force:
        raise Error(
            "smolvla deploy: " + stats_path + " has no min/max, so there is no"
            " action box and nothing would clamp an extrapolated command."
            "\n  Refusing to arm. --force overrides, or use a stats file that"
            " carries min/max."
        )

    # ── warm up BEFORE anything is armed ─────────────────────────────────
    var images = Tensor()
    var scratch = List[Float32]()
    comptime XN = CHUNK * SMOLVLA_ACTION_DIM
    var noise = Tensor.alloc(XN)
    var act = List[Float32]()
    var pose = List[Float32](length=RDIM, fill=Float32(0.0))
    var warm_frames = List[List[UInt8]]()
    for _ in range(N_CAM):
        warm_frames.append(
            List[UInt8](length=CAM_W * CAM_H * 3, fill=UInt8(0))
        )
    var widths = List[Int](length=N_CAM, fill=CAM_W)
    var heights = List[Int](length=N_CAM, fill=CAM_H)
    var first_ms = 0.0
    var warm_ms = 0.0
    if threaded:
        # The worker warmed up on its own thread before publishing READY, and
        # printed its own timings; this is the number it measured.
        warm_ms = Float64(qcells.acquire_load(QW_QUERY_US)) / 1000.0
        # Its `images` buffer is the worker's; this one is only ever a staging
        # area for the request ring, so it is sized on the host alone.
        images.ensure(N_CAM * 3 * SIGLIP_INPUT * SIGLIP_INPUT)
        for j in range(RDIM):
            pose[j] = Float32(0.0)
    else:
        for j in range(RDIM):
            pose[j] = Float32(pol_opt.value().stats.state_mean[j])
        fill_camera_images[TARGET, N_CAM, SIGLIP_INPUT](
            warm_frames, widths, heights, True, images, scratch, dev_ctx
        )
        _fill_noise(noise, XN, 12345, dev_ctx)

        for w in range(WARMUP_QUERIES + 1):
            var t0 = perf_counter_ns()
            pol_opt.value().select_action[TARGET](
                images, ids, pose, noise, act, dev_ctx
            )
            comptime if TARGET != "cpu":
                dev_ctx.value().synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1e6
            if w == 0:
                first_ms = dt
            else:
                warm_ms += dt
        warm_ms /= Float64(WARMUP_QUERIES)
    var chunk_s = Float64(exec_steps) / Float64(SO101_FPS)
    if not threaded:
        print(
            "query        " + fixed(warm_ms, 1) + " ms warm (first "
            + fixed(first_ms, 1) + " ms, " + String(TARGET) + ")"
        )
    print(
        "             chunk " + String(CHUNK) + " steps, executing "
        + String(exec_steps) + " = " + fixed(chunk_s, 2) + " s of motion"
        " per query"
    )
    # ⚠⚠ THE COMPARISON THE WHOLE LOOP SHAPE RESTS ON. If a query costs more
    # than the motion it buys, the arm finishes its chunk before the next one
    # exists and STALLS on its last waypoint every cycle. That is not a slow
    # loop, it is a stuttering arm.
    if warm_ms / 1000.0 >= chunk_s:
        print(
            "   ⚠⚠ ONE QUERY (" + fixed(warm_ms / 1000.0, 2) + " s) COSTS MORE"
            " THAN THE " + fixed(chunk_s, 2) + " s IT BUYS. The arm will stall"
            " between chunks.\n      Raise --exec-steps (up to "
            + String(CHUNK) + "), or pass --threaded, which runs the query off"
            " this thread."
        )
    elif warm_ms / 1000.0 > chunk_s * 0.5:
        print(
            "   ⚠ a query costs " + fixed(warm_ms / 1000.0 / chunk_s, 2)
            + "x the motion it buys — over half the chunk is spent thinking"
            " about the next one."
        )

    # ── the cameras ───────────────────────────────────────────────────────
    print("")
    var cams = List[CameraReader]()
    for i in range(N_CAM):
        print("camera slot " + String(i) + " <- " + devices[i])
        # ⚠ siglip=512, NOT out_w/out_h: the thread runs SmolVLA's OWN
        # `resize_with_pad` from the native frame to the 512x512 [-1, 1]
        # block, the one function the fine-tune's frames went through. ACT's
        # CHW resize would be a different filter and a double resample. It
        # runs there because under --sync the arm holds for the observation
        # build: 38-67 ms measured on the Orin for the two resizes on this
        # thread, now one memcpy of the newest block. rgb=False: the frame is
        # still BGR when the thread swaps it, in the pass that already
        # touches every byte.
        var c = CameraReader.from_spec(
            devices[i], CAM_W, CAM_H, Float64(SO101_FPS), rgb=False,
            fourcc=cam_fourcc, siglip=SIGLIP_INPUT,
        )
        c.start(wait_ms=8000)
        var where = c.resolved_node()
        var got = c.negotiated_fourcc()
        var nfps = c.negotiated_fps()
        print(
            "            " + (where + "  " if where.byte_length() > 0
                               else String(""))
            + ("format " + got + "  " if got.byte_length() > 0
               else String(""))
            + (fixed(nfps, 1) + " fps" if nfps > 0.0 else String(""))
        )
        cams.append(c^)
    print(
        "            " + String(CAM_W) + "x" + String(CAM_H) + " native ->"
        " 512x512 resize_with_pad (SmolVLA's own, not ours), ON THE"
        " CAMERA THREAD"
    )

    var frames = List[List[UInt8]]()
    for i in range(N_CAM):
        frames.append(
            List[UInt8](unsafe_uninit_length = cams[i].frame_bytes())
        )

    # ── the arm ───────────────────────────────────────────────────────────
    print("")
    var f_port = follower_port(port_arg)
    print("follower     " + f_port)
    var why_port = port_refusal(f_port, String("follower"))
    if why_port.byte_length() > 0:
        raise Error("smolvla deploy: " + why_port)
    var follower = SO101Arm(
        f_port,
        max_step_ticks=step_ticks,
        track_step_ticks=TRACK_STEP_TICKS,
    )
    follower.bus.timeout_ms = 20

    var raw = Array[Int32, SO101_N](fill=0)
    if follower.read_positions(Span(raw)) != SO101_N:
        raise Error(
            "smolvla deploy: the follower did not report " + String(SO101_N)
            + " positions — not arming"
        )

    print("")
    if have_box:
        print("   joint          present   demonstrated range        action clamp")
        var outside = 0
        for i in range(SO101_N):
            var p = follower.cal.degrees(i, raw[i])
            var note = String("")
            if p < q_lo[i] or p > q_hi[i]:
                outside += 1
                note = " ⚠ OUTSIDE"
            print(
                "   " + pad_right(joint_name(i), 14)
                + col(p, 8, 1) + "   [" + col(q_lo[i], 7, 1) + ","
                + col(q_hi[i], 7, 1) + " ]   [" + col(a_lo[i], 7, 1) + ","
                + col(a_hi[i], 7, 1) + " ]" + note
            )
        if outside > 0:
            print(
                "   ⚠ " + String(outside) + " joint(s) start outside the"
                " demonstrated pose box. The policy has never seen this"
                " observation;\n     its first command is an extrapolation and"
                " the step clamp is what makes that survivable."
            )
    else:
        print("   ⚠ no min/max in the stats file — no box to check against.")

    # ── go ────────────────────────────────────────────────────────────────
    var stdin = StdinReader()
    var interactive = stdin_is_tty()
    print("")
    if arm_it:
        print(
            "⚠⚠ THE FOLLOWER WILL BE ENERGISED AND WILL MOVE FOR "
            + String(seconds) + " s."
        )
    else:
        print("dry run — torque stays OFF and the arm is backdrivable.")
    if interactive:
        print("press Enter to start (q = quit), and Enter again to stop early")
    stdin.discard_pending()
    var answer = stdin.line()
    if answer == "q" or answer == "Q":
        for i in range(N_CAM):
            try:
                cams[i].stop()
            except:
                pass
        print("nothing was armed.")
        return

    var start_pose = List[Int32]()
    for i in range(SO101_N):
        start_pose.append(raw[i])
    var goals = Array[Int32, SO101_N](fill=0)
    for i in range(SO101_N):
        goals[i] = raw[i]

    if arm_it:
        # ⚠ PARK THE GOAL ON THE PRESENT POSE BEFORE ARMING, so torque engages
        # holding where the arm already is instead of snapping to whatever
        # `Goal_Position` happened to hold.
        follower.write_goals(Span(goals))
        follower.set_torque(True)
        print("\nfollower torque ON\n")
    else:
        print("\ndry run — nothing energised\n")

    var queries = 0
    var commands = 0
    var clamped = 0
    var bus_skipped = 0
    var skipped_at_handover = 0
    var stalled_handovers = 0
    var sync_rebases = 0
    var sum_q = 0.0
    var worst_q = 0.0
    var sum_cam = 0.0
    var t_obs = -10_000
    var t_obs_pending = 0
    var pending = False
    var q_t0 = 0
    # Grid steps of warning to give the next query: only the part of it that
    # runs AFTER submission returns can overlap the arm's motion, so this is
    # the measured overlap and nothing more (see where it is updated). Seeded
    # at 0 — the first query has no measurement, and a lead that is too small
    # costs one handover while a lead that is too large costs every one.
    var lead = 0
    var sum_wait = 0.0
    var worst_wait = 0.0
    var iterations = 0
    var sum_enqueue = 0.0
    var worst_enqueue = 0.0
    # ⚠ NOT `queries`. The last query submitted is still in flight when the
    # run ends, so it is in `sum_enqueue` and not in `queries` — dividing by
    # the wrong one reported a mean ABOVE the worst.
    var submissions = 0
    var sum_write = 0.0
    var worst_write = 0.0
    var sum_body = 0.0
    var worst_body = 0.0
    var sum_obs_gap = 0
    var loop_ns = 0
    var cmd = List[Float64](length=RDIM, fill=0.0)
    var prev_cmd = List[Float64](length=RDIM, fill=0.0)
    var have_prev = False
    var sum_step = 0.0
    var max_step = 0.0
    # ⚠ THE TWO STEPS ARE DIFFERENT ANIMALS AND ONE MEAN HIDES IT. Inside a
    # chunk, consecutive commands are consecutive waypoints of ONE predicted
    # trajectory — demonstration speed. At a handover the arm is sent to where
    # a NEWER trajectory says it should already be, having followed the older
    # one instead, and that discontinuity is what makes a correctly scaled
    # policy look like a fast one.
    var sum_step_in = 0.0
    var max_step_in = 0.0
    var n_step_in = 0
    var sum_step_ho = 0.0
    var max_step_ho = 0.0
    var n_step_ho = 0
    var swapped = False
    # ⚠ THE CHUNK'S OWN SHAPE, measured where it arrives rather than where it
    # is executed. Path length across all CHUNK waypoints against the straight
    # line between the first and the last: a jittery prediction has a long
    # path and a short net, a mis-scaled one has both too long. The
    # demonstrations' figures come from `tools/so101/demo_step_stats.mojo`,
    # which prints this over 50-step windows of the recording.
    var sum_chunk_path = 0.0
    var sum_chunk_net = 0.0
    var n_chunk = 0
    var sum_sm_path = 0.0
    var sum_sm_net = 0.0
    var sum_rev = 0.0
    var sum_rev_of = 0.0
    var ens = TemporalEnsemble[RDIM, CHUNK]()
    var ens_out = List[Scalar[DT]](length=RDIM, fill=Scalar[DT](0))
    var sum_contrib = 0.0
    var n_contrib = 0

    var loop_t0 = perf_counter_ns()
    var deadline = loop_t0 + seconds * 1_000_000_000
    try:
        while perf_counter_ns() < deadline:
            if interactive and stdin.has_input():
                print("stopping early")
                break

            var now_ns = perf_counter_ns()
            iterations += 1
            var t_now = Int(
                Float64(now_ns - loop_t0) * Float64(SO101_FPS) / 1e9
            )

            # ── COLLECT a query that was started earlier ─────────────
            # Only when the current chunk is used up: until then the arm has
            # real waypoints to execute and there is nothing to wait for.
            # ⚠⚠ WHEN TO TAKE THE NEXT CHUNK, AND WHY THE ENSEMBLE NEEDS A
            # DIFFERENT ANSWER. Waiting for the current chunk to be exhausted
            # makes the request cadence exactly `exec_steps - skip`, so the
            # old chunk's coverage ENDS where the new one begins and no
            # instant is covered twice — measured: `ensemble = 1.00 chunks per
            # command`, the blend had nothing to blend.
            #
            # Overlap needs `cadence + skip <= CHUNK - 1`, and with one query
            # in flight the cadence cannot fall below the latency itself
            # (~22 steps here). Keeping a query ALWAYS in flight gives the
            # smallest cadence available and about 5 steps of overlap — which
            # is where the handover jump lives.
            if pending and (
                t_obs < 0
                or t_now - t_obs >= exec_steps
                or (ensemble and rsp_ring.begin_pop().ok())
            ):
                var t_w = perf_counter_ns()
                if threaded and queries == 0:
                    print("  [first query] waiting for the first chunk ...")
                if threaded:
                    # ⚠ THE ONLY PLACE THIS THREAD WAITS FOR THE GPU, and it
                    # waits on a RING rather than on the device: the worker
                    # publishes the chunk when it has it. A long wait here is
                    # the query genuinely outlasting the chunk, not submission
                    # overhead.
                    var got = rsp_ring.begin_pop()
                    var said = False
                    while not got.ok():
                        if qcells.acquire_load(QW_STATE) == QW_FAILED:
                            raise Error(
                                "smolvla deploy: the query thread died — its"
                                " own message is above this line"
                            )
                        # ⚠ A WAIT THAT NEVER ENDS MUST SAY WHY. The worker's
                        # own counters distinguish the three ways this hangs:
                        # a thread that exited, a thread that never polls, and
                        # a thread polling happily while its ring stays empty
                        # (which would mean the two sides hold DIFFERENT rings).
                        if (
                            not said
                            and perf_counter_ns() - t_w > 3_000_000_000
                        ):
                            said = True
                            print(
                                "  ⚠ 3 s with no chunk. query thread: started="
                                + String(worker.value().started())
                                + " exited=" + String(worker.value().exited())
                                + " drive_polls="
                                + String(worker.value().polls())
                                + " body_polls="
                                + String(Int(qcells.acquire_load(QW_POLLS)))
                                + " work=" + String(worker.value().work_polls())
                                + " served="
                                + String(Int(qcells.acquire_load(QW_SERVED)))
                            )
                        got = rsp_ring.begin_pop()
                    var gp = got.data().unsafe_bitcast[Float32]()
                    if len(act) < CHUNK * RDIM:
                        act = List[Float32](
                            length=CHUNK * RDIM, fill=Float32(0)
                        )
                    for i in range(CHUNK * RDIM):
                        act[i] = gp[unsafe_offset=i]
                    rsp_ring.end_pop()
                else:
                    comptime if TARGET != "cpu":
                        dev_ctx.value().synchronize()
                var wait_ms = Float64(perf_counter_ns() - t_w) / 1e6
                sum_wait += wait_ms
                if wait_ms > worst_wait:
                    worst_wait = wait_ms
                if not threaded:
                    pol_opt.value().finish_action[TARGET](act)
                pending = False
                var q_ms = Float64(perf_counter_ns() - q_t0) / 1e6
                sum_q += q_ms
                if q_ms > worst_q:
                    worst_q = q_ms
                queries += 1
                # reversal rate, before any filtering
                var rev = 0
                var rev_of = 0
                for j in range(RDIM):
                    for t in range(1, CHUNK - 1):
                        var d0 = Float64(act[t * RDIM + j]) - Float64(
                            act[(t - 1) * RDIM + j]
                        )
                        var d1 = Float64(act[(t + 1) * RDIM + j]) - Float64(
                            act[t * RDIM + j]
                        )
                        if d0 * d1 < 0.0:
                            rev += 1
                        rev_of += 1
                sum_rev += Float64(rev)
                sum_rev_of += Float64(rev_of)

                var cpath = 0.0
                for t in range(CHUNK - 1):
                    var acc = 0.0
                    for j in range(RDIM):
                        var d = Float64(act[(t + 1) * RDIM + j]) - Float64(
                            act[t * RDIM + j]
                        )
                        acc += d * d
                    cpath += sqrt(acc)
                var cnet = 0.0
                for j in range(RDIM):
                    var d = Float64(act[(CHUNK - 1) * RDIM + j]) - Float64(
                        act[j]
                    )
                    cnet += d * d
                sum_chunk_path += cpath
                sum_chunk_net += sqrt(cnet)
                n_chunk += 1

                if smooth_n > 1:
                    # Centred moving average over waypoints, per joint, with
                    # the window shrinking at both ends so the first and last
                    # waypoints keep their values — those two are what the
                    # handover and the chunk's end depend on.
                    var src = act.copy()
                    var half = smooth_n // 2
                    for t in range(CHUNK):
                        var lo = t - half
                        var hi = t + half
                        if lo < 0:
                            lo = 0
                        if hi > CHUNK - 1:
                            hi = CHUNK - 1
                        var w = Float32(hi - lo + 1)
                        for j in range(RDIM):
                            var acc = Float32(0)
                            for u in range(lo, hi + 1):
                                acc += src[u * RDIM + j]
                            act[t * RDIM + j] = acc / w
                    var spath = 0.0
                    for t in range(CHUNK - 1):
                        var acc2 = 0.0
                        for j in range(RDIM):
                            var d = Float64(act[(t + 1) * RDIM + j]) - Float64(
                                act[t * RDIM + j]
                            )
                            acc2 += d * d
                        spath += sqrt(acc2)
                    var snet = 0.0
                    for j in range(RDIM):
                        var d = Float64(act[(CHUNK - 1) * RDIM + j]) - Float64(
                            act[j]
                        )
                        snet += d * d
                    sum_sm_path += spath
                    sum_sm_net += sqrt(snet)

                t_obs = t_obs_pending
                t_now = Int(
                    Float64(perf_counter_ns() - loop_t0) * Float64(SO101_FPS)
                    / 1e9
                )
                # ⚠⚠ THE SKIP RULE INVERTS WHEN A QUERY OUTLASTS ITS CHUNK:
                # "skip what went stale" would discard the WHOLE chunk and
                # command nothing but its clamped final waypoint — the arm
                # would teleport between end poses and never execute a
                # trajectory. When that happens the arm has been stalled on the
                # pose the observation was taken at, so the chunk still starts
                # where the arm is: re-base the grid instead of skipping.
                if sync:
                    # ⚠ --sync: the arm HELD the pose the observation was
                    # taken at, so the chunk starts where the arm is. Step 0,
                    # now — nothing is stale and nothing is skipped.
                    sync_rebases += 1
                    t_obs = t_now
                elif t_now - t_obs >= CHUNK:
                    stalled_handovers += 1
                    t_obs = t_now
                else:
                    skipped_at_handover += t_now - t_obs
                swapped = True
                if ensemble:
                    # ⚠ PUSHED AT `t_obs`, NOT AT ARRIVAL. The ensemble indexes
                    # a chunk by the step it was PREDICTED FOR; stamping it at
                    # arrival would shift every waypoint by the query latency
                    # and blend poses that describe different instants.
                    ens.push(t_obs, act)

            # ── START the next query while this chunk still has steps ────
            # ⚠⚠ THIS IS THE WHOLE POINT OF THE SPLIT. The query is ~660 ms on
            # an Orin and the chunk is 1.67 s of motion; issued `lead` steps
            # before the chunk runs out, it completes just as the arm needs the
            # next one, and the control loop never stops commanding. Issued at
            # the handover instead — the obvious shape — the arm freezes for
            # two thirds of a second per chunk and then sprints through the
            # catch-up, which is what the first armed run did.
            #
            # ⚠ `images` and `noise` ARE THE GPU'S INPUTS until the collect
            # above, so they are only rebuilt here, with nothing in flight.
            # ⚠ --sync starts a query ONLY when the chunk is exhausted; the
            # collect above then blocks on it next iteration while the servo
            # holds the last waypoint. The pipelined shape starts `lead`
            # steps early, or continuously under --ensemble.
            if not pending and (
                t_obs < 0
                or (sync and t_now - t_obs >= exec_steps)
                or (
                    not sync
                    and (ensemble or exec_steps - (t_now - t_obs) <= lead)
                )
            ):
                var t_c0 = perf_counter_ns()
                # ⚠ FIRST REQUEST ONLY. The loop is silent by design, but the
                # first pass through it is the one that hangs when a new shape
                # is wrong, and "nothing after the banner" names no line.
                var trace = submissions == 0
                if trace:
                    print("  [first query] taking a frame from each camera ...")
                for i in range(N_CAM):
                    if cams[i].take_latest(frames[i]) == 0:
                        if not cams[i].take_blocking(frames[i]):
                            raise Error(
                                "smolvla deploy: camera " + devices[i]
                                + " stopped delivering frames"
                            )
                if trace:
                    print("  [first query] reading the follower's pose ...")
                if follower.read_positions(Span(raw)) != SO101_N:
                    bus_skipped += 1
                    continue
                for j in range(RDIM):
                    pose[j] = Float32(follower.cal.degrees(j, raw[j]))
                if trace:
                    print(
                        "  [first query] frames arrive as 512x512 blocks"
                        " (resized on the camera thread); staging ..."
                    )
                # ⚠ NO RESIZE HERE ANY MORE — `frames[i]` IS the float block
                # the camera thread built. The worker path copies it straight
                # into the request slot below; only the inline path needs it
                # in `images`, on the device.
                if not threaded:
                    fill_siglip_frames[TARGET, N_CAM, SIGLIP_INPUT](
                        frames, images, dev_ctx
                    )
                var obs_ms = Float64(perf_counter_ns() - t_c0) / 1e6
                sum_cam += obs_ms

                # ⚠ FRESH NOISE EVERY QUERY (flow matching integrates FROM a
                # sample of x_1; one reused sample makes every chunk a
                # deterministic function of the observation). Drawn where the
                # query runs: inline below, or on the worker thread.

                # ⚠ THE OBSERVATION'S GRID STEP IS STAMPED BEFORE THE QUERY,
                # not after. The chunk describes the world as it was when the
                # cameras were read, and indexing it from `t_obs` is what makes
                # the inference latency a SKIP rather than a lag.
                q_t0 = perf_counter_ns()
                t_obs_pending = Int(
                    Float64(q_t0 - loop_t0) * Float64(SO101_FPS) / 1e9
                )
                # ⚠⚠ TIMED BECAUSE "ENQUEUE AND RETURN" IS AN ASSUMPTION. A
                # query is ~6970 kernel launches; once the driver's pending-
                # launch queue fills, `cuLaunchKernel` STOPS being asynchronous
                # and blocks until slots free — so the submission can cost most
                # of the GPU time and the control loop gets nothing back.
                if trace:
                    print("  [first query] handing it to the query thread ...")
                if threaded:
                    # ⚠ THE WHOLE REQUEST IN ONE SLOT, pose first. Pushing
                    # costs a 6.3 MB copy (~0.3 ms) and returns: the driver's
                    # launch queue is the worker's problem now.
                    var claim = req_ring.begin_push()
                    if not claim.ok():
                        raise Error(
                            "smolvla deploy: the request ring is full, which"
                            " cannot happen with one query in flight"
                        )
                    var cp = claim.data().unsafe_bitcast[Float32]()
                    for j in range(RDIM):
                        cp[unsafe_offset=j] = pose[j]
                    # The frames ARE the float blocks: N_CAM memcpys after
                    # the pose, no per-element loop over 1.5 M floats.
                    siglip_frames_into_slot[N_CAM, SIGLIP_INPUT](
                        frames, claim.data(), RDIM * 4
                    )
                    req_ring.end_push(QWorker.REQ_BYTES)
                else:
                    _fill_noise(noise, XN, queries * 7919 + 13, dev_ctx)
                    pol_opt.value().start_action[TARGET](
                        images, ids, pose, noise, dev_ctx
                    )
                var enq_ms = Float64(perf_counter_ns() - q_t0) / 1e6
                sum_enqueue += enq_ms
                submissions += 1
                if enq_ms > worst_enqueue:
                    worst_enqueue = enq_ms
                pending = True

                # The observation build itself commands nothing — two grid
                # steps of it, against nineteen for a blocking query.
                sum_obs_gap += Int(obs_ms * Float64(SO101_FPS) / 1000.0)

                # ⚠⚠ THE LEAD IS THE OVERLAP, NOT THE QUERY. Submitting a
                # query is ~6970 launches and the driver stops accepting them
                # once its queue is full, so `start_action` BLOCKS for most of
                # the query (measured: 638 ms of 787 on an Orin). Only the
                # remainder — the part that runs after submission returns —
                # can overlap the arm's motion, and that is all the warning
                # worth taking.
                #
                # Starting earlier than the overlap does not hide the cost, it
                # PAYS IT MORE OFTEN: leading by the whole query length made
                # the loop query every 0.81 s instead of every 1.67 s, and the
                # command rate fell from 17.4 Hz to 8.0. When submission
                # becomes cheap — a CUDA graph, or the query on its own thread
                # — this number grows by itself and the loop pipelines without
                # another edit.
                if queries > 0:
                    # ⚠ THE TWO SHAPES NEED DIFFERENT LEADS, and giving the
                    # threaded one the inline formula cost 4.5 Hz.
                    #
                    # INLINE: submission blocks this thread, so only the
                    # remainder can overlap the arm's motion. Leading by more
                    # does not hide the cost, it pays it more often (measured:
                    # 8.0 Hz against 17.9).
                    #
                    # THREADED: nothing here blocks, so the lead is the WHOLE
                    # round trip — the observation build plus the worker's
                    # query — because that is how long the chunk takes to come
                    # back. Leading by the overlap alone asked 2.5 steps too
                    # late and the loop waited 124 ms per handover.
                    var want = 0
                    if threaded:
                        want = Int(
                            (sum_q / Float64(queries) + obs_ms)
                            * Float64(SO101_FPS) / 1000.0
                        ) + 1
                    else:
                        var overlap_ms = (
                            sum_q / Float64(queries)
                            - sum_enqueue / Float64(submissions)
                        )
                        if overlap_ms < 0.0:
                            overlap_ms = 0.0
                        want = Int(
                            overlap_ms * Float64(SO101_FPS) / 1000.0
                        )
                    if want > exec_steps - 1:
                        want = exec_steps - 1
                    lead = want

            # Nothing to command until the first chunk lands.
            if t_obs < 0:
                continue

            var idx = t_now - t_obs
            if idx < 0:
                idx = 0
            if idx >= CHUNK:
                idx = CHUNK - 1
            var use_ens = False
            if ensemble:
                var nc = ens.n_contributors(t_now)
                if nc > 0:
                    ens.action_at(t_now, ens_out)
                    sum_contrib += Float64(nc)
                    n_contrib += 1
                    use_ens = True
            for j in range(RDIM):
                var v = (
                    Float64(ens_out[j]) if use_ens
                    else Float64(act[idx * RDIM + j])
                )
                if have_box:
                    if v < a_lo[j]:
                        v = a_lo[j]
                        clamped += 1
                    elif v > a_hi[j]:
                        v = a_hi[j]
                        clamped += 1
                cmd[j] = v

            if have_prev:
                var s = 0.0
                for j in range(RDIM):
                    var d = cmd[j] - prev_cmd[j]
                    s += d * d
                var st = sqrt(s)
                sum_step += st
                if st > max_step:
                    max_step = st
                if swapped:
                    sum_step_ho += st
                    n_step_ho += 1
                    if st > max_step_ho:
                        max_step_ho = st
                else:
                    sum_step_in += st
                    n_step_in += 1
                    if st > max_step_in:
                        max_step_in = st
            swapped = False
            for j in range(RDIM):
                prev_cmd[j] = cmd[j]
            have_prev = True

            if arm_it:
                for j in range(RDIM):
                    goals[j] = follower.cal.raw_from_degrees(j, cmd[j])
                # ⚠ TIMED, BECAUSE THE BUS IS SHARED WITH THE CAMERAS. Both
                # MJPG streams and the arm's serial adapter sit on one USB 2.0
                # hub on this rig. A blocking query left the bus idle while it
                # ran; commanding through it every grid step does not, and a
                # write that is suddenly slow would show up only here.
                var t_wr = perf_counter_ns()
                follower.write_goals(Span(goals))
                var wr_ms = Float64(perf_counter_ns() - t_wr) / 1e6
                sum_write += wr_ms
                if wr_ms > worst_write:
                    worst_write = wr_ms
            commands += 1
            var body_ms = Float64(perf_counter_ns() - now_ns) / 1e6
            sum_body += body_ms
            if body_ms > worst_body:
                worst_body = body_ms

            if commands % 10 == 0:
                var line = String("  t=") + pad_left(
                    fixed(Float64(t_now) / Float64(SO101_FPS), 1), 5
                ) + "s  i" + pad_left(String(idx), 3) + " "
                for j in range(RDIM):
                    line += " " + col(cmd[j], 7, 1)
                print(line)

            # Pace to the 30 Hz grid: the chunk's entries are 1/30 s apart and
            # commanding them faster would replay the demonstration in fast
            # forward.
            var next_ns = loop_t0 + (t_now + 1) * 1_000_000_000 // SO101_FPS
            while perf_counter_ns() < next_ns:
                pass
    finally:
        # ⚠ SEE THE ACT DEPLOYMENT: Mojo warns that this assignment is never
        # used and the warning is wrong; the value does propagate.
        loop_ns = perf_counter_ns() - loop_t0
        var released = return_and_release(
            follower, start_pose, arm_it, do_return, stdin, interactive
        )
        if not released:
            print(
                "⚠ the follower is STILL ENERGISED — that is deliberate, see"
                " above."
            )
        for i in range(N_CAM):
            try:
                cams[i].stop()
            except:
                pass

    var elapsed = Float64(loop_ns) / 1e9
    print("=" * 74)
    print("SmolVLA closed-loop run")
    print(
        "  commands          = " + String(commands) + " in "
        + fixed(elapsed, 1) + " s = "
        + fixed(Float64(commands) / elapsed if elapsed > 0.0 else 0.0, 1)
        + " Hz  (the 30 Hz action grid)"
    )
    print(
        "  queries           = " + String(queries) + " = one per "
        + fixed(Float64(commands) / Float64(queries) if queries > 0 else 0.0, 1)
        + " commanded steps"
    )
    print(
        "  query cost        = "
        + fixed(sum_q / Float64(queries) if queries > 0 else 0.0, 1)
        + " ms mean, " + fixed(worst_q, 1) + " ms worst"
    )
    # ⚠ THE LATENCY IS HIDDEN, NOT REMOVED. The query runs while the previous
    # chunk is still executing, so the loop never stops commanding — but a
    # chunk predicted from an observation at time t cannot be executed before
    # t + query, so its first steps are already in the past when it lands, and
    # those are skipped. That floor is the QUERY COST; what the pipeline
    # removed is the freeze that used to sit on top of it.
    print(
        "  skipped at handover = " + String(skipped_at_handover)
        + " grid steps total, "
        + fixed(
            Float64(skipped_at_handover) / Float64(queries)
            if queries > 0 else 0.0, 1
        )
        + " per query (the chunk's first steps, already stale on arrival)"
    )
    # ⚠ NOT A DIAGNOSTIC — A VERDICT ON THE LOOP SHAPE. A stalled handover is
    # a query that outlasted the motion it was buying, so the arm held still
    # waiting for it. Any number here above zero means this machine cannot run
    # this policy closed-loop at this chunk size, whatever the rest of the
    # report says.
    if sync:
        print(
            "  --sync            = " + String(sync_rebases) + " chunks, each"
            " from step 0 after the arm held for the query (the reference's"
            " loop; nothing skipped)"
        )
    elif stalled_handovers > 0:
        print(
            "  ⚠⚠ stalled handovers = " + String(stalled_handovers) + " of "
            + String(queries) + " queries — the arm HELD STILL waiting for the"
            " next chunk.\n     The query outlasts the motion it buys; the"
            " forward belongs off the control thread."
        )
    # ⚠ HOW LATE THE PIPELINE WAS. Zero means every chunk was ready before the
    # arm needed it, which is the whole claim: the query cost is then invisible
    # to the control loop. A large number means `lead` is too small — the query
    # is being started too close to the handover.
    print(
        "  handover wait     = "
        + fixed(sum_wait / Float64(queries) if queries > 0 else 0.0, 1)
        + " ms mean, " + fixed(worst_wait, 1) + " ms worst"
    )
    # ⚠ THE CLAIM THE WHOLE PIPELINE RESTS ON. If submitting the query costs
    # a large fraction of the query itself, the control loop is blocked inside
    # `start_action` and "the query runs while the arm executes" is false —
    # whatever `handover wait` says.
    print(
        "  query submit      = "
        + fixed(
            sum_enqueue / Float64(submissions) if submissions > 0 else 0.0, 1
        )
        + " ms mean, " + fixed(worst_enqueue, 1) + " ms worst"
        + "   (of a " + fixed(sum_q / Float64(queries) if queries > 0 else 0.0, 1)
        + " ms query)"
    )
    if threaded:
        # ⚠ THE SUBMISSION IS STILL ~620 ms — it just is not paid HERE any
        # more. This row is the worker's own measurement, so the two together
        # say where the cost went rather than that it vanished.
        print(
            "  worker query      = "
            + fixed(Float64(qcells.acquire_load(QW_QUERY_US)) / 1000.0, 1)
            + " ms, of which submit "
            + fixed(Float64(qcells.acquire_load(QW_SUBMIT_US)) / 1000.0, 1)
            + " ms   (on the query thread)"
        )
        print(
            "  worker served     = "
            + String(Int(qcells.acquire_load(QW_SERVED))) + " chunks, "
            + String(Int(qcells.acquire_load(QW_DROPPED)))
            + " dropped (nowhere to put them — the loop over-requested)"
        )
    print(
        "  query lead        = " + String(lead)
        + " grid steps of warning (measured from the last query)"
    )
    # The observation build runs on the control thread with nothing in
    # flight — it IS the query's input — so the arm holds for its duration.
    # It was ~2 grid steps (38-67 ms of resize) until the resize moved onto
    # the camera thread; now it is the newest block's copy plus the bus read.
    print(
        "  observation gap   = " + String(sum_obs_gap)
        + " grid steps total (the build commands nothing)"
    )
    print("  observation build = "
          + fixed(sum_cam / Float64(queries) if queries > 0 else 0.0, 1)
          + " ms mean (newest 512x512 block per camera + pose read;"
          " the resize is on the camera thread)")
    # ⚠ WHETHER THE CAMERA THREADS KEPT UP. Each frame costs its thread a
    # resize_with_pad (~15-30 ms on the Orin's cores); at 30 fps that is
    # most of a frame period, and a thread that falls behind delivers
    # staler blocks with no other symptom. Below the requested rate = the
    # observation is up to one dropped period older than the capture.
    for i in range(N_CAM):
        var nf = cams[i].frames_delivered()
        print(
            "  camera " + String(i) + " thread   = " + String(nf)
            + " blocks in " + fixed(elapsed, 1) + " s = "
            + fixed(Float64(nf) / elapsed if elapsed > 0.0 else 0.0, 1)
            + " fps preprocessed (" + String(SO101_FPS) + " requested)"
        )
    # ⚠ THE LOOP'S OWN RATE, not the policy's. `iterations` counts every pass
    # of the control loop; if it is far below the elapsed grid steps then the
    # loop body — not the query — is what fails to keep 30 Hz, and `body` and
    # `bus write` say which part.
    print(
        "  loop iterations   = " + String(iterations) + " in "
        + fixed(elapsed, 1) + " s = "
        + fixed(Float64(iterations) / elapsed if elapsed > 0.0 else 0.0, 1)
        + " Hz  (the grid would be " + String(SO101_FPS) + ")"
    )
    print(
        "  loop body         = "
        + fixed(sum_body / Float64(commands) if commands > 0 else 0.0, 2)
        + " ms mean, " + fixed(worst_body, 1) + " ms worst"
    )
    print(
        "  bus write         = "
        + fixed(sum_write / Float64(commands) if commands > 0 else 0.0, 2)
        + " ms mean, " + fixed(worst_write, 1) + " ms worst"
        + ("" if arm_it else "   (dry run — nothing written)")
    )
    print(
        "  step within chunk = "
        + fixed(sum_step_in / Float64(n_step_in) if n_step_in > 0 else 0.0, 2)
        + " deg mean, " + fixed(max_step_in, 1) + " worst   ("
        + String(n_step_in) + " commands)"
    )
    if n_chunk > 0:
        var mp = sum_chunk_path / Float64(n_chunk)
        var mn = sum_chunk_net / Float64(n_chunk)
        print(
            "  chunk shape       = path " + fixed(mp, 1) + " deg, net "
            + fixed(mn, 1) + " deg, wiggle "
            + fixed(mp / mn if mn > 0.0 else 0.0, 2) + "x over "
            + String(CHUNK) + " waypoints   (" + String(n_chunk) + " chunks)"
        )
        if sum_rev_of > 0.0:
            print(
                "  chunk reversals   = "
                + fixed(100.0 * sum_rev / sum_rev_of, 1)
                + "% of waypoints   (a tremor is ~50%; the demonstrations'"
                " figure comes from tools/so101/demo_step_stats.mojo)"
            )
        if smooth_n > 1:
            var sp = sum_sm_path / Float64(n_chunk)
            var sn = sum_sm_net / Float64(n_chunk)
            print(
                "  after --smooth " + String(smooth_n) + "    = path "
                + fixed(sp, 1) + " deg, net " + fixed(sn, 1) + " deg, wiggle "
                + fixed(sp / sn if sn > 0.0 else 0.0, 2) + "x"
                + "   <- net must SURVIVE; path is what should fall"
            )
    if ensemble:
        print(
            "  ensemble          = "
            + fixed(
                sum_contrib / Float64(n_contrib) if n_contrib > 0 else 0.0, 2
            )
            + " chunks per command   (1.0 means it changed nothing)"
        )
    print(
        "  step at handover  = "
        + fixed(sum_step_ho / Float64(n_step_ho) if n_step_ho > 0 else 0.0, 2)
        + " deg mean, " + fixed(max_step_ho, 1) + " worst   ("
        + String(n_step_ho) + " commands)"
        + "   <- the jump the servo chases at its clamp"
    )
    print("  bus-skipped ticks = " + String(bus_skipped))
    print(
        "  action clamped    = " + String(clamped) + " of "
        + String(commands * RDIM) + " joint-commands"
    )
    print(
        "  commanded step    = mean "
        + fixed(sum_step / Float64(commands) if commands > 0 else 0.0, 2)
        + " deg, max " + fixed(max_step, 2) + " deg"
        + ("" if arm_it else "   <- dry run, the arm never moved")
    )
    if not arm_it:
        print("  ⚠ DRY RUN — nothing was written to the arm. Add --arm.")
    print("=" * 74)


def _fill_noise(
    mut noise: Tensor, n: Int, seed: Int, ctx: Optional[DeviceContext]
) raises:
    """x_1 ~ N(0,1) for the flow-matching sampler, freshly drawn.

    ⚠ A BOX-MULLER FROM A LOCAL LCG, not a shared RNG: this runs inside a
    control loop and must not be perturbed by, or perturb, anything else's
    stream. The quality bar is "plausibly Gaussian and different every query",
    which this clears.
    """
    var s = UInt64(seed * 2 + 1)
    for i in range(0, n, 2):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u1 = Float64((s >> 11) & UInt64(0x1FFFFFFFFFFFFF)) / 9.007199254740992e15
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u2 = Float64((s >> 11) & UInt64(0x1FFFFFFFFFFFFF)) / 9.007199254740992e15
        if u1 < 1e-12:
            u1 = 1e-12
        var r = sqrt(-2.0 * log(u1))
        var a = 6.283185307179586 * u2
        noise.data[i] = Scalar[DT](r * cos(a))
        if i + 1 < n:
            noise.data[i + 1] = Scalar[DT](r * sin(a))
    comptime if TARGET != "cpu":
        noise.upload(ctx.value())
