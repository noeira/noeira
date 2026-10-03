"""Column M — the published LeWM PushT weights through OUR whole stack, on the
50 pairs column R played (docs/LEWM_REOPEN_PLAN.md P5).

Column R (box session A) ran le-wm-main's `eval.py` on stable-worldmodel
0.0.6: 98 % (49 / 50). This runs the same protocol on the same 50 pairs with
nothing of theirs but the weights:

  * the model: `ref_model.mojo` loaded from the converted checkpoint
    (gated against torch: G2a / G2b);
  * the planner: `ref_rollout.cem_step` — CEM 300 x 30, top 30, horizon 5,
    variance 1, from mean 0 at every replan (gated: G2d / G3);
  * the env: our PushT, frictionless with the T turning about its cog
    (gated against pymunk: G4a), started like swm's `_set_state` (recorded
    agent velocity + one settling substep);
  * the frames: the DATASET's start frame for the first plan (swm's
    `evaluate_from_dataset` overwrites the first observation with it), then
    our `render_swm` of the env (gated: G4b); the goal is the DATASET's frame
    25 steps later, every step;
  * actions: the (5 x 10) plan mean, block-major to (25 x 2), un-normalised
    with the EVAL StandardScaler (population std), executed as
    agent + 100 * action; all 25 env steps run, then replan; budget 50;
  * success: ‖Δ(agent, block)‖ < 20 px and |Δ angle| < π/9 against the
    dataset's goal state, latched over the budget (swm `eval_state`).

    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_column_m.mojo \\
        [--dump /tmp/lewm_ref] [--fixture ~/.cache/noeira/lewm_pusht/session_a/out/fixture] \\
        [--episodes 50] [--seed 0] [--video DIR]

`--video DIR`: one mp4 per pair, `pair_<e>_success.mp4` / `_fail.mp4`, every
env step (10 fps) drawn on swm's 512 canvas with the PAIR's goal — the dataset
state the success test compares against: the T in green, the agent's goal
position as a faint disc.
"""

from std.sys import argv
from std.os import getenv
from std.math import sqrt, cos, sin, log, pi, abs
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.act.refload import RefDump
from noeira.envs.pusht import PushTEnv, PushTAction
from noeira.envs.pusht.render_swm import render_pusht_swm_at, render_pusht_swm_canvas_goal
from noeira.io.video.encoder import VideoEncoder
from noeira.io.fileio import rename_over
from std.os import makedirs
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.ref_rollout import (
    LeWMRefRollout, RefEncoder, encode_ref, cem_step, REF_EMB, REF_ACT,
)


comptime TARGET = "gpu"
comptime S = 300        # CEM samples
comptime K = 30         # elites
comptime ITERS = 30     # CEM iterations
comptime HORIZON = 5    # planned blocks (receding horizon 5: all execute)
comptime FRAMESKIP = 5
comptime BUDGET = 50
comptime IMG = 224
comptime HW = IMG * IMG
comptime A = HORIZON * REF_ACT
comptime F = DType.float32


def _imagenet_from_hwc255(hwc: List[Scalar[DT]], off: Int) -> List[Scalar[DT]]:
    """(224, 224, 3) 0..255 -> CHW ImageNet-normalised (swm ToImage + Normalize)."""
    var mean: List[Float64] = [0.485, 0.456, 0.406]
    var std: List[Float64] = [0.229, 0.224, 0.225]
    var out = List[Scalar[DT]](length=3 * HW, fill=Scalar[DT](0))
    for i in range(HW):
        for c in range(3):
            out[c * HW + i] = Scalar[DT](
                (Float64(hwc[off + i * 3 + c]) / 255.0 - mean[c]) / std[c]
            )
    return out^


def _render_frame(mut env: PushTEnv[F]) raises -> List[Scalar[DT]]:
    """Our env's current state, drawn like swm 0.0.6, as (224, 224, 3) 0..255."""
    var ag = env.agent_pos()
    var bp = env.block_pose()
    var buf = List[Scalar[DT]](length=HW * 3, fill=Scalar[DT](0))
    var pix = LayoutTensor[DT, Layout.row_major(IMG, IMG, 3), MutAnyOrigin](
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr())
    )
    render_pusht_swm_at[IMG](
        rebind[Scalar[DT]](bp[0]), rebind[Scalar[DT]](bp[1]), rebind[Scalar[DT]](bp[2]),
        rebind[Scalar[DT]](ag[0]), rebind[Scalar[DT]](ag[1]), pix,
    )
    return buf^


def _gauss(seed: UInt64, n: Int, offset: UInt64) -> List[Scalar[DT]]:
    """n standard normals (Box-Muller over Philox)."""
    var out = List[Scalar[DT]](capacity=n)
    var i = 0
    var ctr = offset
    while i < n:
        var rng = PhiloxRandom(seed=seed, offset=ctr)
        var u = rng.step_uniform()
        ctr += 1
        var u1 = max(Float64(u[0]), 1e-12)
        var u2 = Float64(u[1])
        var r = sqrt(-2.0 * log(u1))
        out.append(Scalar[DT](r * cos(2.0 * pi * u2)))
        if i + 1 < n:
            out.append(Scalar[DT](r * sin(2.0 * pi * u2)))
        i += 2
    return out^


def _success(mut env: PushTEnv[F], goal: List[Scalar[DT]], g: Int) -> Bool:
    """swm `eval_state`: ‖goal[:4] - cur[:4]‖ < 20 and wrapped |Δ angle| < π/9."""
    var ag = env.agent_pos()
    var bp = env.block_pose()
    var cur: List[Float64] = [
        Float64(ag[0]), Float64(ag[1]), Float64(bp[0]), Float64(bp[1]), Float64(bp[2])
    ]
    var d = 0.0
    for j in range(4):
        d += (cur[j] - Float64(goal[g * 7 + j])) ** 2
    var a = abs(Float64(goal[g * 7 + 4]) - cur[4])
    while a > 2.0 * pi:
        a -= 2.0 * pi
    a = min(a, 2.0 * pi - a)
    return sqrt(d) < 20.0 and a < pi / 9.0


def _video_frame(mut env: PushTEnv[F], goal: List[Scalar[DT]], e: Int) -> List[UInt8]:
    var ag = env.agent_pos()
    var bp = env.block_pose()
    return render_pusht_swm_canvas_goal(
        Float64(bp[0]), Float64(bp[1]), Float64(bp[2]), Float64(ag[0]), Float64(ag[1]),
        Float64(goal[e * 7 + 2]), Float64(goal[e * 7 + 3]), Float64(goal[e * 7 + 4]),
        Float64(goal[e * 7 + 0]), Float64(goal[e * 7 + 1]),
    )


def main() raises:
    var dump = String("/tmp/lewm_ref")
    var fixture = getenv("HOME") + "/.cache/noeira/lewm_pusht/session_a/out/fixture"
    var n_eps = 50
    var seed: UInt64 = 0
    var video = String("")
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--dump":
            dump = String(args[i + 1]); i += 1
        elif a == "--fixture":
            fixture = String(args[i + 1]); i += 1
        elif a == "--episodes":
            n_eps = Int(String(args[i + 1])); i += 1
        elif a == "--video":
            video = String(args[i + 1]); i += 1
        elif a == "--seed":
            seed = UInt64(Int(String(args[i + 1]))); i += 1
        i += 1

    var fx = RefDump(fixture)
    var start_state = fx.get(String("pairs.start_state"))   # (50, 7)
    var goal_state = fx.get(String("pairs.goal_state"))
    var start_pix = fx.get(String("pairs.start_pixels"))    # (50, 224, 224, 3)
    var goal_pix = fx.get(String("pairs.goal_pixels"))
    var a_mean = fx.get(String("stats.action_mean"))         # (2,)
    var a_scale = fx.get(String("stats.action_scale_eval"))  # (2,) population std
    var n_pairs = len(start_state) // 7
    n_eps = min(n_eps, n_pairs)
    print("Column M — published LeWM weights, our stack:", n_eps, "pairs; CEM", S, "x", ITERS, "top", K)
    print("  eval action stats: mean", a_mean[0], a_mean[1], " scale", a_scale[0], a_scale[1])

    var c = DeviceContext()
    var ctx = Optional(c)
    var enc = RefEncoder.make[TARGET, Kaiming](ctx)
    _ = load_ref[TARGET](enc, dump, String("emb.0."), ctx)
    var roll = LeWMRefRollout[TARGET, S, HORIZON](dump, ctx)

    var n_success = 0
    var successes = List[Bool]()
    var t_all = perf_counter_ns()
    for e in range(n_eps):
        var t0 = perf_counter_ns()
        var env = PushTEnv[F](seed=UInt64(e))
        _ = env.set_state(
            Scalar[F](start_state[e * 7 + 0]), Scalar[F](start_state[e * 7 + 1]),
            Scalar[F](start_state[e * 7 + 2]), Scalar[F](start_state[e * 7 + 3]),
            Scalar[F](start_state[e * 7 + 4]),
            agent_vx=Scalar[F](start_state[e * 7 + 5]),
            agent_vy=Scalar[F](start_state[e * 7 + 6]),
            settle=True,
        )
        var clip = List[List[UInt8]]()  # --video: the episode's frames
        if video.byte_length() > 0:
            clip.append(_video_frame(env, goal_state, e))
        var goal_emb = encode_ref[TARGET, 1](
            enc, _imagenet_from_hwc255(goal_pix, e * HW * 3), ctx
        )
        var ok = False
        var step = 0
        var replan = 0
        while step < BUDGET:
            # observation: the dataset's start frame first (swm overwrites
            # the first info with it), our render of the env afterwards
            var frame = _imagenet_from_hwc255(start_pix, e * HW * 3) if step == 0 else _imagenet_from_hwc255(_render_frame(env), 0)
            var start_emb = encode_ref[TARGET, 1](enc, frame, ctx)
            var mean = List[Scalar[DT]](length=A, fill=Scalar[DT](0))
            var std = List[Scalar[DT]](length=A, fill=Scalar[DT](1))
            for it in range(ITERS):
                var noise = _gauss(seed * 1000003 + UInt64(e), S * A, UInt64((replan * ITERS + it) * S * A))
                var st = cem_step[TARGET, S, HORIZON, K](roll, start_emb, goal_emb, mean, std, noise)
                mean = st.mean.copy()
                std = st.std.copy()
            replan += 1
            # execute all HORIZON blocks (receding horizon = horizon)
            for blk in range(HORIZON):
                for k in range(FRAMESKIP):
                    if step >= BUDGET:
                        break
                    var ax = Float64(mean[blk * REF_ACT + 2 * k + 0]) * Float64(a_scale[0]) + Float64(a_mean[0])
                    var ay = Float64(mean[blk * REF_ACT + 2 * k + 1]) * Float64(a_scale[1]) + Float64(a_mean[1])
                    var ag = env.agent_pos()
                    _ = env.step(PushTAction[F](
                        Scalar[F](Float64(ag[0]) + 100.0 * ax),
                        Scalar[F](Float64(ag[1]) + 100.0 * ay),
                    ))
                    step += 1
                    if video.byte_length() > 0:
                        clip.append(_video_frame(env, goal_state, e))
                    if _success(env, goal_state, e):
                        ok = True
        if ok:
            n_success += 1
        successes.append(ok)
        if video.byte_length() > 0:
            makedirs(video, exist_ok=True)
            var tmp = video + "/pair_" + String(e) + "_tmp.mp4"
            var venc = VideoEncoder(tmp, 512, 512, fps=10)
            for ref fr in clip:
                venc.add_frame_list(fr)
            _ = venc.close()
            rename_over(tmp, video + "/pair_" + String(e) + ("_success" if ok else "_fail") + ".mp4")
        print("  pair", e, "success" if ok else "FAIL", " (", replan, "replans,",
              Float64(perf_counter_ns() - t0) / 1e9, "s )  running", n_success, "/", e + 1)
    print("Column M:", n_success, "/", n_eps, "=", 100.0 * Float64(n_success) / Float64(n_eps), "%",
          "  total", Float64(perf_counter_ns() - t_all) / 1e9, "s")
    var line = String("  per pair: ")
    for ok in successes:
        line += "1" if ok else "0"
    print(line)
