"""G2d — the planning rollout and its cost against torch's `LeWM.get_cost`.

docs/LEWM_REOPEN_PLAN.md P3. The dump's `rollout` section ran
stable_worldmodel 0.1.1 `get_cost` on S = 8 candidate sequences of 5 z-scored
action blocks from one start frame (history 1, growing to the predictor's 3)
toward one goal frame. Ours (`ref_rollout.LeWMRefRollout`, the left-aligned
causal context on the fixed 3-token predictor) must give:

  * the start and goal embeddings (encoder, BN eval);
  * every predicted embedding, steps 1..5 (context lengths 1, 2, 3, 3, 3 —
    steps 1 and 2 are exactly the variable-length cases the old port faked by
    replicating the latent);
  * the cost of every candidate.

Unit: max |ours - torch| / std(torch) for embeddings; relative for costs.

Run (after `pixi run dump-lewm-ref`):
    pixi run -e apple mojo run -I . tests/experimental/lewm/ref/test_ref_rollout.mojo
"""

from std.sys import argv
from std.math import abs, sqrt
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.ref_rollout import (
    LeWMRefRollout, RefEncoder, encode_ref, REF_EMB,
)


comptime TOL = 1e-4
comptime COST_TOL = 1e-5
comptime S = 8
comptime HORIZON = 5
comptime D = REF_EMB


def _std_err(got: List[Scalar[DT]], want: List[Scalar[DT]], off_g: Int, off_w: Int, n: Int) -> Float64:
    var mean = 0.0
    for i in range(n):
        mean += Float64(want[off_w + i])
    mean /= Float64(n)
    var var_ = 0.0
    var worst = 0.0
    for i in range(n):
        var w = Float64(want[off_w + i])
        var_ += (w - mean) * (w - mean)
        worst = max(worst, abs(Float64(got[off_g + i]) - w))
    var sd = sqrt(var_ / Float64(n))
    return worst / (sd if sd > 0.0 else 1.0)


def _run[target: StaticString](dump: String, ctx: Optional[DeviceContext]) raises -> Int:
    print("  --", target)
    var fails = 0
    var rd = RefDump(dump)
    var enc = RefEncoder.make[target, Kaiming](ctx)
    _ = load_ref[target](enc, dump, String("emb.0."), ctx)
    var pix = rd.get(String("roll.start_pixels"))
    var gpx = rd.get(String("roll.goal_pixels"))
    for i in range(len(gpx)):
        pix.append(gpx[i])
    var se = encode_ref[target, 2](enc, pix, ctx)  # [start | goal]
    var start = List[Scalar[DT]](capacity=D)
    var goal = List[Scalar[DT]](capacity=D)
    for d in range(D):
        start.append(se[d])
        goal.append(se[D + d])

    var want_pred = rd.get(String("roll.predicted_emb"))  # (1, S, 6, D)
    var want_goal = rd.get(String("roll.goal_emb"))
    var e_start = _std_err(start, want_pred, 0, 0, D)
    var e_goal = _std_err(goal, want_goal, 0, 0, D)
    print("     start emb", e_start, "  goal emb", e_goal)
    if e_start > TOL or e_goal > TOL:
        fails += 1

    var roll = LeWMRefRollout[target, S, HORIZON](dump, ctx)
    var embs = roll.rollout(start, rd.get(String("roll.candidates")))
    for t in range(1, HORIZON + 1):
        var worst = 0.0
        for s in range(S):
            var off = (s * (HORIZON + 1) + t) * D
            worst = max(worst, _std_err(embs, want_pred, off, off, D))
        var L = min(t, 3)
        var flag = String("  ") if worst <= TOL else String("✗ ")
        print("     ", flag, "step ", t, " (context ", L, ")  ", worst, sep="")
        if worst > TOL:
            fails += 1

    var cost = roll.cost(embs, goal)
    var want_cost = rd.get(String("roll.cost"))
    var crel = 0.0
    for s in range(S):
        crel = max(crel, abs(Float64(cost[s]) - Float64(want_cost[s])) / abs(Float64(want_cost[s])))
    print("     cost (8 candidates) max rel", crel, "  e.g. ours", cost[0], "torch", want_cost[0])
    if crel > COST_TOL:
        fails += 1
    return fails


def main() raises:
    var dump = String("/tmp/lewm_ref")
    var args = argv()
    if len(args) > 1:
        dump = String(args[1])
    print("G2d  LeWM planning rollout + cost vs torch get_cost")
    var fails = _run["cpu"](dump, None)
    var c = DeviceContext()
    fails += _run["gpu"](dump, Optional(c))
    if fails > 0:
        raise Error("FAIL G2d: " + String(fails) + " check(s) above tolerance")
    print("PASS")
