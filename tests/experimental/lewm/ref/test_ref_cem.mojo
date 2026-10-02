"""G3 — CEM planning against stable_worldmodel 0.1.1's CEMSolver.

docs/LEWM_REOPEN_PLAN.md P3. The dump's `cem` section ran CEMSolver.solve's
loop (64 samples, 4 iterations, top 8, horizon 5; the eval budget is 300 x 30
top 30 — same semantics) and recorded every iteration. Each iteration is
re-run HERE FROM TORCH'S STATE (incoming mean / std and torch's noise draw),
so one float32 vs float64 near-tie at the top-K cut cannot cascade:

  * costs of all 64 candidates (rel 1e-5);
  * the elite set — any mismatch must be a TIE: both candidates within 1e-5
    (relative) of torch's K-th cost, else it fails;
  * the updated mean and std (abs 1e-5) when the elite sets agree.

Then the 4 iterations are chained on OUR OWN state and the final plan is
compared with torch's (informational when a tie occurred, gated otherwise).

Run (after `pixi run dump-lewm-ref`):
    pixi run -e apple mojo run -I . tests/experimental/lewm/ref/test_ref_cem.mojo
"""

from std.sys import argv
from std.math import abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.ref_rollout import (
    LeWMRefRollout, RefEncoder, encode_ref, cem_step, REF_EMB, REF_ACT,
)


comptime S = 64
comptime HORIZON = 5
comptime K = 8
comptime N_STEPS = 4
comptime A = HORIZON * REF_ACT
comptime D = REF_EMB


def _max_abs(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Float64:
    var w = 0.0
    for i in range(len(a)):
        w = max(w, abs(Float64(a[i]) - Float64(b[i])))
    return w


def _run[target: StaticString](dump: String, ctx: Optional[DeviceContext]) raises -> Int:
    print("  --", target)
    var rd = RefDump(dump)
    var enc = RefEncoder.make[target, Kaiming](ctx)
    _ = load_ref[target](enc, dump, String("emb.0."), ctx)
    var pix = rd.get(String("cem.start_pixels"))
    var gpx = rd.get(String("cem.goal_pixels"))
    for i in range(len(gpx)):
        pix.append(gpx[i])
    var se = encode_ref[target, 2](enc, pix, ctx)
    var start = List[Scalar[DT]](capacity=D)
    var goal = List[Scalar[DT]](capacity=D)
    for d in range(D):
        start.append(se[d])
        goal.append(se[D + d])
    var roll = LeWMRefRollout[target, S, HORIZON](dump, ctx)

    var fails = 0
    var ties = 0
    for k in range(N_STEPS):
        var p = String("cem.") + String(k) + "."
        var st = cem_step[target, S, HORIZON, K](
            roll, start, goal, rd.get(p + "mean_in"), rd.get(p + "var_in"), rd.get(p + "noise")
        )
        var want_cost = rd.get(p + "costs")
        var crel = 0.0
        for s in range(S):
            crel = max(crel, abs(Float64(st.costs[s]) - Float64(want_cost[s])) / abs(Float64(want_cost[s])))
        var want_idx = rd.get(p + "topk_inds")
        var kth = Float64(want_cost[Int(want_idx[K - 1])])
        var same = True
        var tie_ok = True
        for i in range(K):
            var found = False
            for j in range(K):
                if Int(want_idx[j]) == st.elite[i]:
                    found = True
            if not found:
                same = False
                # ours picked a candidate torch did not: both must sit at the cut
                if abs(Float64(want_cost[st.elite[i]]) - kth) / abs(kth) > 1e-5:
                    tie_ok = False
        var line = String("     iter ") + String(k) + "  cost rel " + String(crel)
        if same:
            var em = _max_abs(st.mean, rd.get(p + "mean_out"))
            var ev = _max_abs(st.std, rd.get(p + "var_out"))
            line += "  elites SAME  mean " + String(em) + "  std " + String(ev)
            if em > 1e-5 or ev > 1e-5:
                fails += 1
                line += "  ✗"
        elif tie_ok:
            ties += 1
            line += "  elites differ AT A TIE (accepted)"
        else:
            fails += 1
            line += "  ✗ elites differ, not a tie"
        if crel > 1e-5:
            fails += 1
            line += "  ✗ cost"
        print(line)

    # the chain on our own state
    var mean = rd.get(String("cem.0.mean_in"))
    var std = rd.get(String("cem.0.var_in"))
    for k in range(N_STEPS):
        var st = cem_step[target, S, HORIZON, K](
            roll, start, goal, mean, std, rd.get(String("cem.") + String(k) + ".noise")
        )
        mean = st.mean.copy()
        std = st.std.copy()
    var chain = _max_abs(mean, rd.get(String("cem.actions")))
    print("     chained 4 iterations on our own state: final plan max |ours - torch|", chain,
          "(gated only without ties)" if ties > 0 else "")
    if ties == 0 and chain > 1e-4:
        fails += 1
    return fails


def main() raises:
    var dump = String("/tmp/lewm_ref")
    var args = argv()
    if len(args) > 1:
        dump = String(args[1])
    print("G3  CEM vs stable_worldmodel 0.1.1 CEMSolver (", S, "samples, top", K, ",", N_STEPS, "iters )")
    var fails = _run["cpu"](dump, None)
    var c = DeviceContext()
    fails += _run["gpu"](dump, Optional(c))
    if fails > 0:
        raise Error("FAIL G3: " + String(fails) + " check(s)")
    print("PASS")
