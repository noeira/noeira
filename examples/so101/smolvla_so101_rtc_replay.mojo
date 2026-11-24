# +--------------------------------------------------------------------------+ #
# | Real-Time Chunking, offline: do consecutive chunks join at the handover?
# +--------------------------------------------------------------------------+ #
"""Replay recorded SO-101 episodes through the deploy's policy, pipelined, and
score every handover with and without RTC. No arm, no cameras.

    # on the Orin (CUDA), one policy, ~3.2 GB:
    mojo build -I . --target-accelerator sm_87 -DSMOLVLA_GPU=1 \\
        -o /tmp/svla_rtc examples/so101/smolvla_so101_rtc_replay.mojo
    /tmp/svla_rtc --root projects/so101-tower/datasets/cube-in-bowl \\
        --ckpt ~/ckpt/smolvla_cube_a64_last.ckpt \\
        --stats ~/ckpt/smolvla_stats_cube-in-bowl.json \\
        --episodes 0,7,23 --exec-steps 25 --delays 6,11,16

## The schedule (the paper's, and the deploy's pipelined loop)

Each chunk executes `s = --exec-steps` actions, its indices `d .. d+s-1`.
Observations are `s` steps apart, so the next one is taken at the executing
chunk's index `k = s`, the query lasts `d` control steps, and the new chunk is
executed from ITS index `d` — the deploy's skip rule (`now - t_obs`). During
the query the robot runs the old chunk's `k..k+d-1`; `s + d <= CHUNK`.

Two chains, the same observation and the same noise at every query:

  * NAIVE — today's pipelined loop: every chunk sampled plain.
  * RTC   — each chunk guided toward its predecessor's leftover `old[k:]`
            (`rtc.mojo`), frozen weight 1 on the `d` steps that run during
            the query.

## What is printed, per delay `d`, in degrees (L2 over the 6 joints)

  * **handover step** `|new[d] - old[k+d-1]|` — the jump the servo sees when
    the new chunk takes over. The deploy reports the same quantity as `step
    at handover`; 35° mean was the oscillating 20 Sep run.
  * **in-chunk step** — `|old[i+1] - old[i]|` inside chunks, the demo-speed
    reference a smooth handover should approach.
  * **prefix gap** (RTC only) `mean_i<d |new[i] - old[k+i]|` — how well the
    frozen actions were inpainted.
  * **plan bend** (RTC only) `mean |R[i] - U[i]|` over `i >= d`, R the guided
    chunk and U a plain chunk from the SAME observation and noise — how far
    RTC moved the part of the plan that will actually be executed. Large
    means RTC is overriding what the policy sees, not just joining.
  * **query ms**, plain vs armed, synchronised each — RTC's run-time cost.

⚠⚠ OPEN LOOP. The images and the pose come from the recording at the
observation's frame, not from an arm that followed the chunks. So this
measures whether the chunks JOIN, which is RTC's claim, and nothing about
success; the arm A/B (`smolvla_so101_deploy_real.mojo --rtc`) is what does.
"""

from std.math import abs, sqrt
from std.sys import argv
from std.sys.defines import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.data.lerobot import (
    LeRobotInfo, EpisodeIndex, FrameTable, CameraStream,
)
from noeira.deep_agents.smolvla.expert import EXPERT_FF
from noeira.deep_agents.smolvla.finetune import load_trainables
from noeira.deep_agents.smolvla.heads import (
    SMOLVLA_ACTION_DIM, SMOLVLA_EXPERT_W, SMOLVLA_STATE_DIM,
)
from noeira.deep_agents.smolvla.observation import fill_camera_images
from noeira.deep_agents.smolvla.policy import SmolVLAPolicy
from noeira.deep_agents.smolvla.query_worker import _qw_fill_noise
from noeira.deep_agents.smolvla.recording import SO101_N_LANG, SO101_TASKS
from noeira.deep_agents.smolvla.rtc import (
    RTC_BETA_DEFAULT, rtc_schedule_from_name,
)
from noeira.deep_agents.smolvla.tasks import TaskTokens
from noeira.deep_agents.smolvla.text import (
    SMOLLM_DIM, SMOLLM_KV_W, SMOLLM_LAYERS,
)
from noeira.io.hf import hf_download_file, HF_MODEL
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.tensor import Tensor
from noeira.nn.primitives.linear import Linear
from noeira.utils.fmt import fixed
from noeira.vision.resize_pad import SIGLIP_INPUT

comptime TARGET: StaticString = "gpu" if is_defined["SMOLVLA_GPU"]() else "cpu"
comptime N_CAM = 2
comptime N_LANG = SO101_N_LANG
comptime CHUNK = 50
comptime STEPS = 10
comptime RDIM = 6
comptime XN = CHUNK * SMOLVLA_ACTION_DIM
comptime Pol = SmolVLAPolicy[N_CAM, N_LANG, CHUNK, STEPS, 1]


def _ints(s: String) raises -> List[Int]:
    var out = List[Int]()
    for p in s.split(","):
        out.append(Int(String(p)))
    return out^


def _opt(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def _dist(a: List[Float32], ia: Int, b: List[Float32], ib: Int) -> Float64:
    """L2 over the joints between row `ia` of `a` and row `ib` of `b`."""
    var s = 0.0
    for j in range(RDIM):
        var d = Float64(a[ia * RDIM + j]) - Float64(b[ib * RDIM + j])
        s += d * d
    return sqrt(s)


struct Acc(Copyable, Movable):
    var sum: Float64
    var worst: Float64
    var n: Int

    def __init__(out self):
        self.sum = 0.0
        self.worst = 0.0
        self.n = 0

    def add(mut self, v: Float64):
        self.sum += v
        self.n += 1
        if v > self.worst:
            self.worst = v

    def mean(self) -> Float64:
        return self.sum / Float64(self.n) if self.n > 0 else 0.0

    def show(self) -> String:
        return (
            fixed(self.mean(), 2) + " mean  " + fixed(self.worst, 2)
            + " worst  (n " + String(self.n) + ")"
        )


def _query(
    mut pol: Pol,
    mut images: Tensor,
    ref ids: List[Int],
    ref pose: List[Float32],
    mut noise: Tensor,
    mut act: List[Float32],
    mut keep: Tensor,
    ctx: Optional[DeviceContext],
) raises -> Float64:
    """One synchronised query; `keep` <- the model-space chunk. Returns ms."""
    var t0 = perf_counter_ns()
    pol.start_action[TARGET](images, ids, pose, noise, ctx)
    comptime if TARGET != "cpu":
        ctx.value().synchronize()
    pol.finish_action[TARGET](act)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    keep.ensure(XN)
    for i in range(XN):
        keep.data[i] = pol.chunk_buf.data[i]
    return ms


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var root = _opt(args, "--root", "projects/so101-tower/datasets/cube-in-bowl")
    var ckpt = _opt(args, "--ckpt", "")
    var stats_path = _opt(args, "--stats", "")
    var tasks_path = _opt(args, "--tasks", String(SO101_TASKS))
    var task_index = Int(_opt(args, "--task-index", "0"))
    var episodes = _ints(_opt(args, "--episodes", "0"))
    var s_exec = Int(_opt(args, "--exec-steps", "25"))
    var delays = _ints(_opt(args, "--delays", "11"))
    var beta = Float64(_opt(args, "--beta", String(RTC_BETA_DEFAULT)))
    var sched_name = _opt(args, "--schedule", "exp")
    var schedule = rtc_schedule_from_name(sched_name)
    var max_queries = Int(_opt(args, "--max-queries", "1000000"))
    if ckpt == "" or stats_path == "":
        raise Error("rtc replay: --ckpt and --stats are required")
    for d in delays:
        if d < 1 or d + s_exec > CHUNK:
            raise Error(
                "rtc replay: each delay must be >= 1 with delay + --exec-steps"
                " <= " + String(CHUNK) + " — the old chunk must still cover"
                " the query"
            )

    print("=" * 74)
    print("SmolVLA — Real-Time Chunking, offline replay (open loop)")
    print("=" * 74)
    print(
        "dataset " + root + "  episodes " + String(len(episodes))
        + "  exec " + String(s_exec) + "  schedule " + sched_name
        + "  beta " + fixed(beta, 1)
    )

    # ── the policy, exactly as the query worker builds it ────────────────
    var ctx = Optional[DeviceContext](None)
    comptime if TARGET != "cpu":
        ctx = DeviceContext()
    var base = hf_download_file(
        String("lerobot/smolvla_base"), String("model.safetensors"), HF_MODEL
    )
    var pol = Pol.make[TARGET, Deterministic](ctx)
    pol.load[TARGET](base, ctx)
    var sp_frozen = Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM].make[
        TARGET, Deterministic
    ](ctx)
    _ = load_trainables[
        TARGET, SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF,
        SMOLLM_DIM, SMOLLM_KV_W, SMOLVLA_ACTION_DIM,
    ](
        ckpt, pol.expert, pol.action_in, pol.time_mlp_in, pol.time_mlp_out,
        pol.action_out, sp_frozen, ctx,
    )
    pol.load_stats(stats_path)
    pol.set_fused_vision_attention(True)
    _ = pol.freeze_all[TARGET](ctx)
    var tasks = TaskTokens(tasks_path)
    var ids = tasks.for_index(task_index)
    if len(ids) != N_LANG:
        raise Error("rtc replay: task is " + String(len(ids)) + " tokens")
    print('instruction "' + tasks.texts[task_index] + '"')

    # ── the recording ────────────────────────────────────────────────────
    var info = LeRobotInfo(root)
    var index = EpisodeIndex(root, info.cameras)
    var table = FrameTable(root, info.state_dim, info.action_dim)
    if len(info.cameras) != N_CAM:
        raise Error("rtc replay: the dataset has " + String(len(info.cameras))
                    + " cameras")
    print("cameras " + info.cameras[0] + ", " + info.cameras[1])

    var images = Tensor()
    var scratch = List[Float32]()
    var noise = Tensor.alloc(XN)
    var pose = List[Float32](length=RDIM, fill=Float32(0))
    var act_n = List[Float32]()
    var act_r = List[Float32]()
    var act_u = List[Float32]()
    var prev_n = List[Float32]()
    var prev_r = List[Float32]()
    var keep_n = Tensor()
    var keep_r = Tensor()
    var keep_u = Tensor()

    var t_plain = Acc()
    var t_armed = Acc()
    var n_queries = 0
    for di in range(len(delays)):
        var d = delays[di]
        var k = s_exec
        var ho_n = Acc()
        var ho_r = Acc()
        var in_n = Acc()
        var in_r = Acc()
        var gap = Acc()
        var bend = Acc()
        print("\n── d = " + String(d) + " steps (" + fixed(
            Float64(d) * 1000.0 / Float64(info.fps), 0) + " ms), k = "
            + String(k) + " ──")
        for ep in episodes:
            var length = index.length[ep]
            var first = index.from_index[ep]
            var cams = List[CameraStream]()
            var fr0 = List[Int]()
            for c in range(N_CAM):
                cams.append(CameraStream(info.cameras[c], root))
                fr0.append(
                    Int(round(index.vid_from_ts[c][ep] * Float64(info.fps)))
                )
            # Observation frames: 0, then every `s_exec` steps, taken `k`
            # steps into the executing chunk.
            var t_obs = 0
            var qi = 0
            while t_obs < length and n_queries < max_queries:
                # decode forward to frame t_obs
                var frames = List[List[UInt8]]()
                var widths = List[Int]()
                var heights = List[Int]()
                for c in range(N_CAM):
                    # forward-only: skip to the observation's frame, decode it
                    cams[c].open_at(
                        index.vid_chunk[c][ep], index.vid_file[c][ep],
                        fr0[c] + t_obs,
                    )
                    cams[c].next_native()
                    frames.append(cams[c].raw.copy())
                    widths.append(cams[c].width)
                    heights.append(cams[c].height)
                for j in range(RDIM):
                    pose[j] = table.qpos[(first + t_obs) * RDIM + j]
                # ⚠ ffmpeg decodes RGB24, so no swap (the deploy's OpenCV
                # frames are BGR and pass swap_rb = True).
                fill_camera_images[TARGET, N_CAM, SIGLIP_INPUT](
                    frames, widths, heights, False, images, scratch, ctx
                )
                var seed = n_queries * 7919 + 13

                # NAIVE chain: plain sample.
                pol.rtc_disarm()
                _qw_fill_noise(noise, XN, seed, ctx)
                t_plain.add(_query(pol, images, ids, pose, noise, act_n, keep_n, ctx))
                # RTC chain: plain reference U, then guided R, same noise.
                _qw_fill_noise(noise, XN, seed, ctx)
                _ = _query(pol, images, ids, pose, noise, act_u, keep_u, ctx)
                if qi > 0:
                    pol.sampler.rtc.arm[TARGET](
                        keep_r, k, d, CHUNK, schedule, beta, ctx
                    )
                    _qw_fill_noise(noise, XN, seed, ctx)
                    t_armed.add(
                        _query(pol, images, ids, pose, noise, act_r, keep_r, ctx)
                    )
                    pol.rtc_disarm()
                else:
                    act_r = act_u.copy()
                    keep_r.ensure(XN)
                    for i in range(XN):
                        keep_r.data[i] = keep_u.data[i]
                n_queries += 1

                if qi > 0:
                    # handover: old executed up to k+d-1, new from d
                    ho_n.add(_dist(act_n, d, prev_n, k + d - 1))
                    ho_r.add(_dist(act_r, d, prev_r, k + d - 1))
                    var g = 0.0
                    for i in range(d):
                        g += _dist(act_r, i, prev_r, k + i)
                    gap.add(g / Float64(d))
                    var b = 0.0
                    for i in range(d, CHUNK):
                        b += _dist(act_r, i, act_u, i)
                    bend.add(b / Float64(CHUNK - d))
                for i in range(d, d + s_exec - 1):
                    if i + 1 < CHUNK:
                        in_n.add(_dist(act_n, i + 1, act_n, i))
                        in_r.add(_dist(act_r, i + 1, act_r, i))
                prev_n = act_n.copy()
                prev_r = act_r.copy()
                qi += 1
                t_obs += s_exec
            for c in range(N_CAM):
                cams[c].close()
        print("  handover step  NAIVE " + ho_n.show())
        print("                 RTC   " + ho_r.show())
        print("  in-chunk step  NAIVE " + in_n.show())
        print("                 RTC   " + in_r.show())
        print("  prefix gap     RTC   " + gap.show())
        print("  plan bend      RTC   " + bend.show())
    print("\nquery ms  plain " + t_plain.show())
    print("          armed " + t_armed.show())
