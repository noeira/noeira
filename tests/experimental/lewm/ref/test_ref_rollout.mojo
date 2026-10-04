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
  * the cost of every candidate;
  * `PlanCost`: one-hot weights on the last step reproduce `cost` exactly,
    `staged` is `last` before replan HORIZON, and `all:2` weights sum to 1
    and rise 2x per step.

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
from noeira.experimental.lewm.ref_load import load_ref, tf32_gemm
from noeira.experimental.lewm.ref_rollout import (
    LeWMRefRollout, RefEncoder, encode_ref, PlanCost, REF_EMB,
)


comptime TOL = 1e-4
comptime COST_TOL = 1e-5
comptime TOL_TF32 = 5e-2       # CUDA GPU leg: measured 2.2e-2 at step 5 (5090)
comptime COST_TOL_TF32 = 3e-2  # measured 7.6e-3 relative
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
    var tol = TOL_TF32 if tf32_gemm[target]() else TOL
    var cost_tol = COST_TOL_TF32 if tf32_gemm[target]() else COST_TOL
    if tf32_gemm[target]():
        print("  --", target, "(TF32 GEMMs: bands", tol, cost_tol, ")")
    else:
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
    if e_start > tol or e_goal > tol:
        fails += 1

    var roll = LeWMRefRollout[target, S, HORIZON](dump, ctx)
    var embs = roll.rollout(start, rd.get(String("roll.candidates")))
    for t in range(1, HORIZON + 1):
        var worst = 0.0
        for s in range(S):
            var off = (s * (HORIZON + 1) + t) * D
            worst = max(worst, _std_err(embs, want_pred, off, off, D))
        var L = min(t, 3)
        var flag = String("  ") if worst <= tol else String("✗ ")
        print("     ", flag, "step ", t, " (context ", L, ")  ", worst, sep="")
        if worst > tol:
            fails += 1

    var cost = roll.cost(embs, goal)
    var want_cost = rd.get(String("roll.cost"))
    var crel = 0.0
    for s in range(S):
        crel = max(crel, abs(Float64(cost[s]) - Float64(want_cost[s])) / abs(Float64(want_cost[s])))
    print("     cost (8 candidates) max rel", crel, "  e.g. ours", cost[0], "torch", want_cost[0])
    if crel > cost_tol:
        fails += 1

    var onehot = List[Float64](length=HORIZON, fill=0.0)
    onehot[HORIZON - 1] = 1.0
    var c1 = roll.cost(embs, goal, onehot)
    var same = True
    for s in range(S):
        if c1[s] != cost[s]:
            same = False
    var staged = PlanCost.parse(String("staged"))
    var w_all = PlanCost.parse(String("all:2")).weights(HORIZON, 0)
    var tot = 0.0
    var rising = True
    for t in range(HORIZON):
        tot += w_all[t]
        if t > 0 and abs(w_all[t] / w_all[t - 1] - 2.0) > 1e-12:
            rising = False
    var ok_pc = same and len(staged.weights(HORIZON, HORIZON - 1)) == 0 \
        and len(staged.weights(HORIZON, HORIZON)) == HORIZON and abs(tot - 1.0) < 1e-12 and rising
    print("     PlanCost: one-hot last == cost", same, "| staged / all:2 weights ok", ok_pc)
    if not ok_pc:
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
