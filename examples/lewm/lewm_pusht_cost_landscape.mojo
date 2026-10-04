"""Does LeWM's planning cost still rank states under a visual shift?

docs/LEWM_REOPEN_PLAN.md P7 (E2). Under `dark:0.5` adapting cut the
prediction loss 71 % and planning stayed at 1-2 / 50: a good one-step
prediction is not a good cost landscape. This probe asks the cost directly,
encoder only, no planner and no predictor:

  for each fixture pair, K states around its goal — the goal itself (k = 0)
  and K - 1 perturbations of the block (position, angle) and the agent at
  random scales — rendered like the planner sees them (`render_frame`);
  cost_k = ‖enc(x_k) − enc(goal frame)‖²  (LeWM's criterion, goal = the
  dataset's goal frame), true distance_k = `pair_margin` (< 1 = success).

  Spearman(cost, distance) per pair, averaged: 1 = the cost orders states as
  the task does; the goal state's cost rank (0 = cheapest of the K); and the
  share of pairs whose cheapest state is a success.

Each `--shift` is applied to every frame and to the goal frame, as in the
planner. Run on the laptop (Metal) in minutes:

    pixi run -e apple mojo run -I . examples/lewm/lewm_pusht_cost_landscape.mojo \\
        --dump <run>/epoch_7 --fixture <session_a>/out/fixture --pairs 20
"""

from std.sys import argv
from std.os import getenv
from std.math import pi
from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.ref_rollout import RefEncoder, encode_ref, REF_EMB
from noeira.experimental.lewm.paper_pairs import (
    PairEnv, PAIR_HW, imagenet_from_hwc255, render_frame, pair_margin,
    VisualShift, shift_frame,
)


comptime TARGET = "gpu"
comptime K = 32
comptime FRAME = PAIR_HW * 3


def _ranks(v: List[Float64]) -> List[Float64]:
    """Ranks 0..n-1 (ties broken by index — continuous values, no ties)."""
    var n = len(v)
    var r = List[Float64](length=n, fill=0.0)
    for i in range(n):
        var below = 0
        for j in range(n):
            if v[j] < v[i] or (v[j] == v[i] and j < i):
                below += 1
        r[i] = Float64(below)
    return r^


def _spearman(a: List[Float64], b: List[Float64]) -> Float64:
    var ra = _ranks(a)
    var rb = _ranks(b)
    var n = Float64(len(a))
    var ma = 0.0
    var mb = 0.0
    for i in range(len(a)):
        ma += ra[i]
        mb += rb[i]
    ma /= n
    mb /= n
    var sab = 0.0
    var saa = 0.0
    var sbb = 0.0
    for i in range(len(a)):
        sab += (ra[i] - ma) * (rb[i] - mb)
        saa += (ra[i] - ma) ** 2
        sbb += (rb[i] - mb) ** 2
    return sab / (saa * sbb) ** 0.5


def main() raises:
    var dump = String("/workspace/lewm_train/epoch_7")
    var fixture = getenv("HOME") + "/.cache/noeira/lewm_pusht/session_a/out/fixture"
    var n_pairs = 20
    var shifts: List[String] = ["none", "dark:0.5", "noise:0.1", "swap"]
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--dump":
            dump = String(args[i + 1]); i += 1
        elif a == "--fixture":
            fixture = String(args[i + 1]); i += 1
        elif a == "--pairs":
            n_pairs = Int(String(args[i + 1])); i += 1
        else:
            raise Error("unknown argument " + a)
        i += 1

    var fx = RefDump(fixture)
    var goal_state = fx.get(String("pairs.goal_state"))
    var goal_pix = fx.get(String("pairs.goal_pixels"))
    n_pairs = min(n_pairs, len(goal_state) // 7)

    var c = DeviceContext()
    var ctx = Optional(c)
    var enc = RefEncoder.make[TARGET, Kaiming](ctx)
    _ = load_ref[TARGET](enc, dump, String("emb.0."), ctx)
    print("cost landscape on", dump, ":", n_pairs, "pairs x", K, "states (k = 0 the goal itself)")

    # the states and their true distances, once for every shift
    seed(7)
    var frames = List[List[Scalar[DT]]]()   # per pair: K frames (HWC 0..255)
    var dists = List[List[Float64]]()
    for e in range(n_pairs):
        var fr = List[Scalar[DT]](capacity=K * FRAME)
        var di = List[Float64]()
        for k in range(K):
            var s = 0.0 if k == 0 else random_float64(0.05, 1.0)
            var g = e * 7
            var bx = Float64(goal_state[g + 2]) + s * random_float64(-90, 90)
            var by = Float64(goal_state[g + 3]) + s * random_float64(-90, 90)
            var bt = Float64(goal_state[g + 4]) + s * random_float64(-pi / 2, pi / 2)
            var ax = Float64(goal_state[g + 0]) + s * random_float64(-90, 90)
            var ay = Float64(goal_state[g + 1]) + s * random_float64(-90, 90)
            var env = PairEnv(seed=UInt64(e))
            _ = env.set_state(
                Scalar[DType.float32](min(480.0, max(32.0, ax))), Scalar[DType.float32](min(480.0, max(32.0, ay))),
                Scalar[DType.float32](min(450.0, max(62.0, bx))), Scalar[DType.float32](min(450.0, max(62.0, by))),
                Scalar[DType.float32](bt),
            )
            di.append(pair_margin(env, goal_state, e))
            var f = render_frame(env)
            for v in f:
                fr.append(v)
        frames.append(fr^)
        dists.append(di^)

    for spec in shifts:
        var shift = VisualShift.parse(spec)
        var rho_sum = 0.0
        var rank_sum = 0.0
        var cheapest_ok = 0
        for e in range(n_pairs):
            var goal = List[Scalar[DT]](capacity=FRAME)
            for q in range(FRAME):
                goal.append(goal_pix[e * FRAME + q])
            shift_frame(goal, 0, shift, UInt64(e) * 7919 + 2)
            var ge = encode_ref[TARGET, 1](enc, imagenet_from_hwc255(goal, 0), ctx)
            var batch = List[Scalar[DT]](capacity=K * FRAME)
            for k in range(K):
                var f = List[Scalar[DT]](capacity=FRAME)
                for q in range(FRAME):
                    f.append(frames[e][k * FRAME + q])
                shift_frame(f, 0, shift, UInt64(e) * 7919 + UInt64(k) * 13 + 3)
                for v in imagenet_from_hwc255(f, 0):
                    batch.append(v)
            var xe = encode_ref[TARGET, K](enc, batch, ctx)
            var cost = List[Float64]()
            for k in range(K):
                var acc = 0.0
                for d in range(REF_EMB):
                    acc += (Float64(xe[k * REF_EMB + d]) - Float64(ge[d])) ** 2
                cost.append(acc)
            rho_sum += _spearman(cost, dists[e])
            var r = _ranks(cost)
            rank_sum += r[0]
            var best = 0
            for k in range(K):
                if cost[k] < cost[best]:
                    best = k
            if dists[e][best] < 1.0:
                cheapest_ok += 1
        print("  shift", spec, "| Spearman(cost, true distance)", Float32(rho_sum / Float64(n_pairs)),
              "| goal state's cost rank", Float32(rank_sum / Float64(n_pairs)), "of", K,
              "| cheapest state a success:", cheapest_ok, "/", n_pairs)
