# +--------------------------------------------------------------------------+ #
# | How long does ONE SmolVLA query take? The number the control loop needs.
# +--------------------------------------------------------------------------+ #
"""Time one `select_action` on the published weights, phase by phase.

    pixi run -e apple mojo build -I . -o /tmp/svla_probe \\
        examples/so101/smolvla_so101_latency_probe.mojo
    /tmp/svla_probe

⚠⚠ **THIS IS A PREREQUISITE, NOT A CURIOSITY.** The ACT deploy program's whole
architecture — a 30 Hz action grid, a temporal ensemble, a chunk index that
means what it meant in training — is built on ONE measured number: 95 ms per
forward on this machine. Writing SmolVLA's control loop before measuring the
same number would be picking an architecture by guesswork.

The decisive comparison is between two durations:

  * **one chunk of motion** = `chunk_size / fps` = 50 / 30 = **1.667 s**. That is
    how much wall-clock the arm can execute from a single query.
  * **one query** = what this measures.

If a query is comfortably shorter than a chunk, the loop can be open-loop
chunk-to-chunk: query, execute 50 waypoints at 30 Hz, query again while the tail
is still playing. SmolVLA is DESIGNED for that — `n_action_steps` is 50, the
whole chunk, not ACT's handful. If a query is longer, the arm stalls between
chunks and the loop needs the forward off the control thread, which is a
different and much larger program.

⚠ **The per-phase figures do NOT sum to the end-to-end figure, and must not be
reported as if they did.** GPU work is enqueued asynchronously; timing a phase
means synchronising after it, and those extra synchronisations change the
pipeline being measured. End-to-end is the honest total. The breakdown is for
deciding WHAT to optimise, not for adding up.

⚠ Needs the published weights (~907 MB, cached) and instantiates every layer:
402,737,376 params x 4 bytes x 2 (host AND device) = **3.2 GB**. One policy.

## Optional: a fine-tune's weights, and the memory the deploy keeps

    SMOLVLA_CKPT=<run's best.ckpt or a merged LoRA checkpoint> /tmp/svla_probe
    SMOLVLA_FREEZE=1 /tmp/svla_probe        # freeze_all after load, as the deploy does

`SMOLVLA_CKPT` loads a fine-tune's trainable set over the published weights
(`load_trainables`, the deploy's `--ckpt`): the latency of a fine-tuned or
merged-LoRA policy against the base's, on one build. `SMOLVLA_FREEZE=1` frees
every gradient buffer after load (`SmolVLAPolicy.freeze_all`). Built with
`-D NN_INFERENCE_ONLY` none is ever allocated. On Linux the probe prints the
system's MemAvailable at each stage: on the Orin, CPU and GPU share that one
pool, so a stage's drop is what the policy actually holds.
"""

from std.ffi import OwnedDLHandle, c_int
from std.math import sqrt
from std.os import getenv
from std.os.path import exists
from std.time import perf_counter_ns
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.io.hf import hf_download_file, HF_MODEL
from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.smolvla.policy import SmolVLAPolicy
from noeira.deep_agents.smolvla.normalize import SmolVLAStats, normalize_state
from noeira.deep_agents.smolvla.heads import SMOLVLA_CONNECTOR_IN
from noeira.deep_agents.smolvla.observation import fill_camera_images
from noeira.deep_agents.smolvla.tasks import TaskTokens
from noeira.deep_agents.smolvla.finetune import load_trainables
from noeira.deep_agents.smolvla.heads import (
    SMOLVLA_EXPERT_W, SMOLVLA_STATE_DIM, SMOLVLA_ACTION_DIM,
)
from noeira.deep_agents.smolvla.text import SMOLLM_DIM, SMOLLM_LAYERS, SMOLLM_KV_W
from noeira.deep_agents.smolvla.expert import EXPERT_FF
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.vision.resize_pad import SIGLIP_INPUT

comptime TARGET = "gpu"
"""⚠ ONE target per binary, deliberately. Building both would instantiate two
policies and cost 6.4 GB — the mistake that took a 16 GiB laptop down once
already. To measure the CPU path, change this and rebuild.

Worth measuring both: ACT found Metal SLOWER than CPU at batch 1, because a few
hundred tiny kernels each pay Metal's ~20 us command-buffer retirement floor
(`_the_metal_launch_floor_is_command_buffer_retirement`). SmolVLA's kernels are
far bigger — 1024-token attention in the vision tower — so that result may
invert here. It is a measurement either way, not an inference from ACT's."""

comptime REPO = String("lerobot/smolvla_base")
comptime TABLE = "tools/vla/smolvla_tasks_record-test_20260828_092736.tsv"
comptime N_CAM = 2
comptime N_LANG = 6
comptime CHUNK = 50
comptime STEPS = 10
comptime FPS = 30
comptime Pol = SmolVLAPolicy[N_CAM, N_LANG, CHUNK, STEPS, 1]
comptime CAM_W = 640
comptime CAM_H = 480
comptime RDIM = 6
comptime REPS = 5
comptime WARMUP = 2



def _launch_count() -> Int:
    """Kernels launched so far, from the LD_PRELOAD interceptor, or -1.

    ⚠⚠ THE NUMBER THAT SEPARATES "SLOW KERNELS" FROM "TOO MANY KERNELS", and
    they need opposite fixes. The prefix runs at ~9% of this board's fp32 peak
    (425 GFLOP in 2.0 s against a 2.4 TFLOPS machine), so it is NOT
    compute-bound — either the kernels are inefficient or there are thousands
    of tiny ones each paying launch overhead. Dividing the query's kernel
    count by its duration says which, before anyone reaches for fp16 or a
    profiler.

    ⚠ NOT AN ERROR WHEN ABSENT. The interceptor is a property of the PROCESS
    (LD_PRELOAD), not of this binary, so a run without it is normal and simply
    has no count to report.
    """
    try:
        var lib = OwnedDLHandle("./noeira/cuda/libcuda_intercept.so")
        return Int(lib.get_function[c_int]("intercept_get_launch_count")())
    except:
        return -1


def ms(ns: Int) -> Float64:
    return Float64(ns) / 1.0e6


def mem_available_mb() -> Int:
    """MemAvailable from /proc/meminfo in MB, or -1 off Linux. On the Orin
    the CPU and the GPU allocate from this one pool."""
    try:
        if not exists("/proc/meminfo"):
            return -1
        var f = open(String("/proc/meminfo"), "r")
        var text = f.read()
        f.close()
        for line in text.splitlines():
            if line.startswith("MemAvailable:"):
                var parts = line.split()
                return Int(String(parts[1])) // 1024
        return -1
    except:
        return -1


def mem_line(label: String, start_mb: Int):
    var now = mem_available_mb()
    if now >= 0:
        print("  memory  " + label + ": MemAvailable " + String(now)
              + " MB  (held since start " + String(start_mb - now) + " MB)")


def main() raises:
    print("=" * 70)
    print("SmolVLA query latency —", TARGET, ", batch 1,", N_CAM, "cameras")
    print("=" * 70)

    var tasks = TaskTokens(String(TABLE))
    var ids = tasks.for_index(0)
    print("  instruction:", tasks.texts[0], "->", len(ids), "tokens, P =",
          Pol.P)

    var path = hf_download_file(REPO, String("model.safetensors"), HF_MODEL)
    var mem0 = mem_available_mb()
    mem_line(String("start"), mem0)
    var d = DeviceContext()
    mem_line(String("after the device context"), mem0)
    print("  building every layer (3.2 GB) …")
    var pol = Pol.make[TARGET](Optional(d))
    mem_line(String("after make (values + gradients)"), mem0)
    pol.load[TARGET](path, Optional(d))
    var ckpt = getenv("SMOLVLA_CKPT")
    if ckpt.byte_length() > 0:
        var sp_unused = Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM].make[
            TARGET, Deterministic
        ](Optional(d))
        _ = load_trainables[
            TARGET, SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
            SMOLLM_KV_W, SMOLVLA_ACTION_DIM,
        ](
            ckpt, pol.expert, pol.action_in, pol.time_mlp_in,
            pol.time_mlp_out, pol.action_out, sp_unused, Optional(d),
        )
        print("  weights: lerobot/smolvla_base + " + ckpt)
    else:
        print("  weights: lerobot/smolvla_base")
    mem_line(String("after load"), mem0)
    if getenv("SMOLVLA_FREEZE").byte_length() > 0:
        var n_frozen = pol.freeze_all[TARGET](Optional(d))
        print("  froze " + String(n_frozen) + " values (freeze_all, as the deploy does)")
        mem_line(String("after freeze_all"), mem0)

    # The recording's own scale. Not the point of this probe, but a policy
    # without stats refuses to run.
    var m: List[Float32] = [16.64, -29.97, 31.07, 73.73, 41.12, 26.27]
    var sd: List[Float32] = [21.00, 54.38, 51.43, 17.93, 18.72, 9.21]
    for i in range(RDIM):
        pol.stats.state_mean.append(m[i])
        pol.stats.state_std.append(sd[i])
        pol.stats.action_mean.append(m[i])
        pol.stats.action_std.append(sd[i])

    # A plausible observation: two 640x480 frames through the real chain.
    var frames = List[List[UInt8]]()
    for c in range(N_CAM):
        var f = List[UInt8](unsafe_uninit_length=CAM_W * CAM_H * 3)
        for i in range(len(f)):
            f[i] = UInt8((i * 7 + c * 53) % 256)
        frames.append(f^)
    var widths: List[Int] = [CAM_W, CAM_W]
    var heights: List[Int] = [CAM_H, CAM_H]
    var scratch = List[Float32]()
    var images = Tensor()

    # ⚠ SYNCHRONISE BEFORE STOPPING THE CLOCK. Without this the timer measures
    # the enqueue and not the work — the first draft of this file reported
    # 17.6 ms here and 127.6 ms for the same call further down, and 17.6 was
    # fiction. This file's own header warns about exactly that; the warning did
    # not stop me writing it.
    var t_pre = perf_counter_ns()
    fill_camera_images[TARGET, N_CAM, SIGLIP_INPUT](
        frames, widths, heights, False, images, scratch, Optional(d)
    )
    comptime if TARGET != "cpu":
        d.synchronize()
    var preproc_ns = perf_counter_ns() - t_pre

    var pose: List[Float32] = [12.0, -40.0, 22.0, 70.0, 35.0, 20.0]
    comptime XN = CHUNK * 32
    var noise = Tensor.alloc(XN)
    for i in range(XN):
        noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
    comptime if TARGET != "cpu":
        noise.upload(d)
    var act = List[Float32]()

    # ⚠ Warm up first. The first call pays lazy buffer allocation and, on GPU,
    # kernel specialisation — real costs, but paid ONCE, so folding them into a
    # steady-state figure would overstate every subsequent query.
    for _ in range(WARMUP):
        pol.select_action[TARGET](
            images, ids, pose, noise, act, Optional(d)
        )
    comptime if TARGET != "cpu":
        d.synchronize()
    mem_line(String("after the warm-up queries"), mem0)

    # ── the fused vision attention against the two-pass path ─────────────
    # The deploy runs the vision towers' attention FUSED (no softmax cache;
    # on NVIDIA at SigLIP's shape that is MAX's FA2 kernel, whose matmuls are
    # TF32). Same observation through both paths, and the chunk they
    # produce compared in the action's own units — the number that says
    # whether a kernel's precision band reaches the arm. The timed queries
    # below run the fused path, as the deploy does.
    comptime if TARGET != "cpu":
        var act_two = List[Float32]()
        pol.set_fused_vision_attention(False)
        pol.select_action[TARGET](
            images, ids, pose, noise, act_two, Optional(d)
        )
        d.synchronize()
        var act_fused = List[Float32]()
        pol.set_fused_vision_attention(True)
        for _ in range(WARMUP):
            pol.select_action[TARGET](
                images, ids, pose, noise, act_fused, Optional(d)
            )
        d.synchronize()
        var worst = Float64(0.0)
        var ss = Float64(0.0)
        var n_cmp = min(len(act_two), len(act_fused))
        for i in range(n_cmp):
            var dlt = Float64(act_two[i]) - Float64(act_fused[i])
            ss += dlt * dlt
            if abs(dlt) > worst:
                worst = abs(dlt)
        print()
        print("  vision attention FUSED vs two-pass, same observation:")
        print("     max |delta action| =", worst, "  rms =",
              sqrt(ss / Float64(max(n_cmp, 1))), "  over", n_cmp,
              "chunk values (the action's units)")

    # ── end to end, the honest total ─────────────────────────────────────
    var launches_before = _launch_count()
    var best = 0
    var total = 0
    for r in range(REPS):
        var t0 = perf_counter_ns()
        pol.select_action[TARGET](
            images, ids, pose, noise, act, Optional(d)
        )
        comptime if TARGET != "cpu":
            d.synchronize()
        var dt = perf_counter_ns() - t0
        total += dt
        # ⚠ MIN, not mean: a baseline that drifts upward across a session is
        # this project's recorded failure mode, and the minimum is the figure
        # least polluted by whatever else the machine was doing.
        if best == 0 or dt < best:
            best = dt
        print("      rep", r, ":", ms(dt), "ms")

    var launches_after = _launch_count()
    print()
    if launches_before >= 0 and launches_after > launches_before:
        var per_query = (launches_after - launches_before) // REPS
        print("  kernel launches per query:", per_query)
        # ⚠ ~5 us of launch overhead each is the rule of thumb on this class of
        # part. If that product is a large share of the query, the fix is
        # FEWER launches (fusion, or the CUDA-graph capture this repo already
        # has in `noeira/cuda/graph.mojo`) — not lower precision.
        var overhead_ms = Float64(per_query) * 0.005
        print("     at ~5 us each that is", overhead_ms, "ms =",
              100.0 * overhead_ms / ms(best), "% of the query")
    print()
    print("  camera preprocessing (CPU, 2 frames):", ms(preproc_ns), "ms")
    print("  query, min of", REPS, ":", ms(best), "ms  (vision attention fused)")
    print("  query, mean     :", ms(total // REPS), "ms")
    print("  queries/s (min) :", 1000.0 / ms(best))

    # ── the comparison the loop design turns on ──────────────────────────
    var chunk_ms = 1000.0 * Float64(CHUNK) / Float64(FPS)
    var query_ms = ms(best) + ms(preproc_ns)
    print()
    print("  one chunk of motion:", chunk_ms, "ms (", CHUNK, "steps at", FPS,
          "fps )")
    print("  one query + preproc:", query_ms, "ms")
    print("  duty cycle         :", query_ms / chunk_ms)
    if query_ms < chunk_ms:
        print()
        print("  => a query fits inside a chunk. Open-loop chunk-to-chunk is")
        print("     viable: execute 50 waypoints while the next query runs.")
        print("     Headroom:", chunk_ms - query_ms, "ms")
    else:
        print()
        print("  => A QUERY IS LONGER THAN THE MOTION IT PRODUCES. The arm")
        print("     would stall between chunks. The loop needs the forward off")
        print("     the control thread, or fewer denoising steps, or a")
        print("     smaller vision path — decide with the breakdown below,")
        print("     NOT by shortening the chunk.")

    # ── per phase, for deciding what to optimise ─────────────────────────
    # ⚠ These are measured with a synchronisation after each phase, so they do
    # NOT sum to the figure above. Read them as proportions.
    print()
    print("  per-phase (syncs added — does NOT sum to the total above):")

    var st = List[Float32]()
    normalize_state(pol.stats, pose, st, 32)
    pol.state_buf.ensure(32)
    for i in range(32):
        pol.state_buf.data[i] = Scalar[DT](st[i])
    comptime if TARGET != "cpu":
        pol.state_buf.upload_resident(d)

    var t_pfx = perf_counter_ns()
    pol.prefix.run[TARGET, Pol.VOCAB, SMOLVLA_CONNECTOR_IN, 32](
        pol.vision, pol.connector, pol.embed.weight.val, pol.state_proj,
        images, ids, pol.state_buf, pol.prefix_buf, Optional(d),
    )
    comptime if TARGET != "cpu":
        d.synchronize()
    var pfx_ns = perf_counter_ns() - t_pfx

    var t_fill = perf_counter_ns()
    pol.cache.reset()
    pol.prefill.run[TARGET](
        pol.tower, pol.cache, pol.prefix_buf, pol.prefill_out, Optional(d)
    )
    comptime if TARGET != "cpu":
        d.synchronize()
    var fill_ns = perf_counter_ns() - t_fill

    var t_den = perf_counter_ns()
    pol.sampler.sample[TARGET, Pol.P](
        pol.expert, pol.cache, pol.denoiser, pol.action_in, pol.time_mlp_in,
        pol.time_mlp_out, pol.action_out, noise, pol.chunk_buf, Optional(d),
    )
    comptime if TARGET != "cpu":
        d.synchronize()
    var den_ns = perf_counter_ns() - t_den

    var sum_ns = preproc_ns + pfx_ns + fill_ns + den_ns
    print("      preprocess 2 frames  :", ms(preproc_ns), "ms")
    print("      prefix (2x SigLIP +  :", ms(pfx_ns), "ms   <- 2 vision towers")
    print("        connector, gather, state_proj)")
    print("      prefill 16 layers    :", ms(fill_ns), "ms")
    print("      denoise", STEPS, "steps     :", ms(den_ns), "ms   (",
          ms(den_ns // STEPS), "ms/step )")
    print("      ---")
    print("      sum of phases        :", ms(sum_ns), "ms  vs end-to-end",
          ms(best), "ms")
    print()
    print("  share of the sum:")
    print("      preprocess :", 100.0 * Float64(preproc_ns) / Float64(sum_ns), "%")
    print("      prefix     :", 100.0 * Float64(pfx_ns) / Float64(sum_ns), "%")
    print("      prefill    :", 100.0 * Float64(fill_ns) / Float64(sum_ns), "%")
    print("      denoise    :", 100.0 * Float64(den_ns) / Float64(sum_ns), "%")

    print()
    print("Measured on this machine. Do not carry these numbers to another.")
