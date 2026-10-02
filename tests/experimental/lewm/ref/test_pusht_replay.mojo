"""G4a — our native PushT against the dataset's recorded pymunk trajectories.

docs/LEWM_REOPEN_PLAN.md P4. The fixture (box session A) holds 2000 dataset
episodes' `state` + `action`; the oracle (`tools/lewm/pusht_replay_oracle.py`,
run on the box with stable-worldmodel 0.0.6) replayed 400 starts in pymunk
from `_set_state` for 50 steps. Here our `PushTEnv` (float32 — see `F`) replays the SAME
starts with the SAME raw actions (relative: target = agent + 100 * action, as
swm's PushT does), and the errors are reported against:

  * the RECORDED states — the number that matters for the paper protocol;
  * pymunk's replay — what an exact engine reaches from this initialisation
    (the dataset has no block velocity, so pymunk itself sits ~1 px off the
    recording: the FLOOR).

Position error = |Δ(agent_x, agent_y, block_x, block_y)| in px (the protocol's
success test is < 20 px), angle error wrapped to [0, π] (success: < π/9).

GATE: with swm's `_set_state` semantics (agent velocity + one free settling
substep), ours vs pymunk after 25 and 50 steps: median < 0.5 px, p90 < 2 px,
angle median < 0.01 rad. It found two defects in our PushT (2026-10-02): the
reference is FRICTIONLESS (`body.friction = 1` is a pymunk no-op) and the T
rotates about its centre of gravity (0, 45), not its origin. Before: 13 px
median off the recording after 25 steps, 144 / 400 starts beyond the 20 px
success radius; after: 1.08 px (pymunk's own floor 1.10), 2 / 400.

Run:
    pixi run mojo run -I . tests/experimental/lewm/ref/test_pusht_replay.mojo \
        [~/.cache/noeira/lewm_pusht/session_a]
"""

from std.sys import argv
from std.math import sqrt, pi, abs
from std.os import getenv

from noeira.deep_agents.act.refload import RefDump
from noeira.envs.pusht import PushTEnv, PushTAction


comptime F = DType.float32
"""⚠ `PushTEnv[DTYPE]`'s physics is float32 whatever DTYPE says: `set_state`,
`_substep` and the kernels take the module-level `dtype` (float32). pymunk
runs in float64."""
comptime K = 50


def _pose(mut env: PushTEnv[F]) -> List[Float64]:
    """[agent_x, agent_y, block_x, block_y, block_angle] — the block's ORIGIN
    pose, the dataset's convention (`block_pose` converts from the cog the
    physics integrates)."""
    var ag = env.agent_pos()
    var bp = env.block_pose()
    return [Float64(ag[0]), Float64(ag[1]), Float64(bp[0]), Float64(bp[1]), Float64(bp[2])]


def _ang(a: Float64, b: Float64) -> Float64:
    var d = abs(a - b)
    while d > 2.0 * pi:
        d -= 2.0 * pi
    return min(d, 2.0 * pi - d)


def _pct(mut v: List[Float64], q: Float64) -> Float64:
    sort(v)
    return v[min(len(v) - 1, Int(q * Float64(len(v))))]


def main() raises:
    var root = getenv("HOME") + "/.cache/noeira/lewm_pusht/session_a"
    var args = argv()
    if len(args) > 1:
        root = String(args[1])
    var fx = RefDump(root + "/out/fixture")
    var orc = RefDump(root + "/oracle")
    var state = fx.get(String("dyn.state"))     # (N, 7)
    var action = fx.get(String("dyn.action"))   # (N, 2)
    var starts = orc.get(String("oracle.starts"))  # (S, 2): episode, row
    var replay = orc.get(String("oracle.replay"))  # (S, K+1, 7)
    var S = len(starts) // 2
    print("G4a  PushT replay:", S, "starts x", K, "steps (our physics: float32)")

    var fails = 0
    for mode in range(2):
        fails += _replay(mode, state, action, starts, replay, S)
    if fails > 0:
        raise Error("FAIL G4a: our PushT departs from pymunk (" + String(fails) + " check(s))")
    print("PASS")


def _replay(
    mode: Int, state: List[Float32], action: List[Float32],
    starts: List[Float32], replay: List[Float32], S: Int,
) raises -> Int:
    """mode 0: `set_state` as it was (zero velocities, no physics step);
    mode 1: swm's `_set_state` (agent velocity from the state + one free
    settling substep)."""
    print("  --", "set_state: zero velocities" if mode == 0 else "set_state: agent velocity + settle substep (= swm 0.0.6)")
    # errors[k][metric] over starts; metrics: 0 pos vs rec, 1 ang vs rec,
    # 2 pos vs pymunk, 3 ang vs pymunk, 4 pymunk pos vs rec (the floor),
    # 5 agent-only vs rec, 6 block-only vs rec
    var err = List[List[List[Float64]]]()
    for _ in range(K + 1):
        var per = List[List[Float64]]()
        for _ in range(7):
            per.append(List[Float64]())
        err.append(per^)

    var env = PushTEnv[F](seed=0)
    for i in range(S):
        var r = Int(starts[2 * i + 1])
        _ = env.set_state(
            Scalar[F](state[r * 7 + 0]), Scalar[F](state[r * 7 + 1]),
            Scalar[F](state[r * 7 + 2]), Scalar[F](state[r * 7 + 3]),
            Scalar[F](state[r * 7 + 4]),
            agent_vx=Scalar[F](state[r * 7 + 5]) if mode == 1 else Scalar[F](0),
            agent_vy=Scalar[F](state[r * 7 + 6]) if mode == 1 else Scalar[F](0),
            settle=mode == 1,
        )
        for k in range(K):
            var p = _pose(env)
            var tx = p[0] + 100.0 * Float64(action[(r + k) * 2 + 0])
            var ty = p[1] + 100.0 * Float64(action[(r + k) * 2 + 1])
            _ = env.step(PushTAction[F](Scalar[F](tx), Scalar[F](ty)))
            var q = _pose(env)
            var rec = (r + k + 1) * 7
            var pym = (i * (K + 1) + k + 1) * 7
            var dr = 0.0
            var dp = 0.0
            var df = 0.0
            var da = 0.0
            var db = 0.0
            for j in range(4):
                var e = (q[j] - Float64(state[rec + j])) ** 2
                dr += e
                if j < 2:
                    da += e
                else:
                    db += e
                dp += (q[j] - Float64(replay[pym + j])) ** 2
                df += (Float64(replay[pym + j]) - Float64(state[rec + j])) ** 2
            err[k + 1][0].append(sqrt(dr))
            err[k + 1][1].append(_ang(q[4], Float64(state[rec + 4])))
            err[k + 1][2].append(sqrt(dp))
            err[k + 1][3].append(_ang(q[4], Float64(replay[pym + 4])))
            err[k + 1][4].append(sqrt(df))
            err[k + 1][5].append(sqrt(da))
            err[k + 1][6].append(sqrt(db))

    print("  steps | ours vs recorded: pos med / p90 (px), angle med (rad) | agent med, block med | ours vs pymunk: pos med / p90 | pymunk vs recorded (floor): pos med / p90")
    var report: List[Int] = [1, 5, 25, 50]
    for kk in range(len(report)):
        var k = report[kk]
        var a0 = err[k][0].copy()
        var a1 = err[k][1].copy()
        var a2 = err[k][2].copy()
        var a4 = err[k][4].copy()
        var a5 = err[k][5].copy()
        var a6 = err[k][6].copy()
        print(
            "  ", k, " | ", _pct(a0, 0.5), " / ", _pct(a0, 0.9), ", ", _pct(a1, 0.5),
            " | ", _pct(a5, 0.5), ", ", _pct(a6, 0.5),
            " | ", _pct(a2, 0.5), " / ", _pct(a2, 0.9),
            " | ", _pct(a4, 0.5), " / ", _pct(a4, 0.9), sep="",
        )
    # success-relevant tail: fraction of starts whose 25-step error already
    # exceeds the protocol's 20 px success radius
    var over = 0
    for v in err[25][0]:
        if v > 20.0:
            over += 1
    print("  starts > 20 px off the recording after 25 steps:", over, "/", S)
    if mode == 0:
        return 0  # the old initialisation: reported, not gated
    # GATE (swm's `_set_state`): an exact engine from the same start. Measured
    # 2026-10-02: 0.07 / 0.36 px at 25 steps, 0.10 / 0.41 at 50, angle 0.006.
    # The two defects this caught: friction 1.0 alone -> 13-26 px median;
    # rotating about the origin alone -> 8 px.
    var fails = 0
    for k in [25, 50]:
        var pos = err[k][2].copy()
        var ang = err[k][3].copy()
        var med = _pct(pos, 0.5)
        var p90 = _pct(pos, 0.9)
        var amed = _pct(ang, 0.5)
        if med > 0.5 or p90 > 2.0 or amed > 0.01:
            fails += 1
            print("  ✗ vs pymunk after", k, "steps: pos median", med, "p90", p90, "angle median", amed)
    return fails
