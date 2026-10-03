"""The LeWM paper protocol on PushT, shared by its drivers.

docs/LEWM_REOPEN_PLAN.md P5 / P7. The column-M runner
(`examples/lewm/lewm_pusht_column_m.mojo`) and the AdaJEPA driver
(`examples/lewm/lewm_pusht_adajepa.mojo`) play the same 50 (start, goal)
pairs of box session A's fixture and must observe, plan and score them the
same way — kept here once, not twice:

  * `imagenet_from_hwc255`: (224, 224, 3) 0..255 -> CHW ImageNet-normalised
    (swm `ToImage` + `Normalize`);
  * `render_frame`: our env's state drawn like swm 0.0.6 (`render_swm`);
  * `gauss`: the CEM's standard normals (Box-Muller over Philox);
  * `pair_success`: swm `eval_state` — ‖Δ(agent, block)‖ < 20 px and
    wrapped |Δ angle| < π/9 against the pair's goal state;
  * `video_frame`: the 512 canvas with the pair's own goal (for people);
  * `shift_frame`: AdaJEPA's E2 visual shifts — test-time perturbations of
    every observed frame, goal included;
  * `tta_windows`: an episode's newest training windows, for adaptation.
"""

from std.math import sqrt, cos, sin, log, pi, abs
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT
from noeira.envs.pusht import PushTEnv
from noeira.envs.pusht.render_swm import render_pusht_swm_at, render_pusht_swm_canvas_goal


comptime PAIR_IMG = 224
comptime PAIR_HW = PAIR_IMG * PAIR_IMG
comptime PairEnv = PushTEnv[DType.float32]


def imagenet_from_hwc255(hwc: List[Scalar[DT]], off: Int) -> List[Scalar[DT]]:
    """(224, 224, 3) 0..255 -> CHW ImageNet-normalised (swm ToImage + Normalize)."""
    var mean: List[Float64] = [0.485, 0.456, 0.406]
    var std: List[Float64] = [0.229, 0.224, 0.225]
    var out = List[Scalar[DT]](length=3 * PAIR_HW, fill=Scalar[DT](0))
    for i in range(PAIR_HW):
        for c in range(3):
            out[c * PAIR_HW + i] = Scalar[DT](
                (Float64(hwc[off + i * 3 + c]) / 255.0 - mean[c]) / std[c]
            )
    return out^


def render_frame(mut env: PairEnv) raises -> List[Scalar[DT]]:
    """Our env's current state, drawn like swm 0.0.6, as (224, 224, 3) 0..255."""
    var ag = env.agent_pos()
    var bp = env.block_pose()
    var buf = List[Scalar[DT]](length=PAIR_HW * 3, fill=Scalar[DT](0))
    var pix = LayoutTensor[DT, Layout.row_major(PAIR_IMG, PAIR_IMG, 3), MutAnyOrigin](
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr())
    )
    render_pusht_swm_at[PAIR_IMG](
        rebind[Scalar[DT]](bp[0]), rebind[Scalar[DT]](bp[1]), rebind[Scalar[DT]](bp[2]),
        rebind[Scalar[DT]](ag[0]), rebind[Scalar[DT]](ag[1]), pix,
    )
    return buf^


def gauss(seed: UInt64, n: Int, offset: UInt64) -> List[Scalar[DT]]:
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


def pair_success(mut env: PairEnv, goal: List[Scalar[DT]], g: Int) -> Bool:
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


def video_frame(mut env: PairEnv, goal: List[Scalar[DT]], e: Int) -> List[UInt8]:
    """The 512 canvas of the env with pair `e`'s goal state drawn."""
    var ag = env.agent_pos()
    var bp = env.block_pose()
    return render_pusht_swm_canvas_goal(
        Float64(bp[0]), Float64(bp[1]), Float64(bp[2]), Float64(ag[0]), Float64(ag[1]),
        Float64(goal[e * 7 + 2]), Float64(goal[e * 7 + 3]), Float64(goal[e * 7 + 4]),
        Float64(goal[e * 7 + 0]), Float64(goal[e * 7 + 1]),
    )


# ── E2: visual shifts ─────────────────────────────────────────────────────

comptime SHIFT_NONE = 0
comptime SHIFT_NOISE = 1
"""Additive Gaussian pixel noise, σ = strength x 255, clipped to [0, 255]."""
comptime SHIFT_DARK = 2
"""Lighting: every channel x strength (e.g. 0.5)."""
comptime SHIFT_SWAP = 3
"""Colour: RGB -> BRG (the blue agent turns green, the green goal red...)."""


@fieldwise_init
struct VisualShift(Copyable, Movable, Writable):
    var kind: Int
    var strength: Float64

    @staticmethod
    def parse(spec: String) raises -> Self:
        """`none`, `noise:<σ as a fraction of 255>`, `dark:<gain>`, `swap`."""
        if spec == "none":
            return Self(SHIFT_NONE, 0.0)
        if spec == "swap":
            return Self(SHIFT_SWAP, 0.0)
        var parts = spec.split(":")
        if len(parts) == 2:
            var v = Float64(String(parts[1]))
            if parts[0] == "noise":
                return Self(SHIFT_NOISE, v)
            if parts[0] == "dark":
                return Self(SHIFT_DARK, v)
        raise Error("unknown --shift " + spec + " (none | noise:σ | dark:gain | swap)")


def shift_frame(
    mut hwc: List[Scalar[DT]], off: Int, shift: VisualShift, seed: UInt64
):
    """Apply `shift` in place to the (224, 224, 3) 0..255 frame at `off`.
    Noise is drawn from Philox(seed) — pass a distinct seed per frame."""
    if shift.kind == SHIFT_NONE:
        return
    var n = PAIR_HW * 3
    if shift.kind == SHIFT_DARK:
        for i in range(n):
            hwc[off + i] = hwc[off + i] * Scalar[DT](shift.strength)
    elif shift.kind == SHIFT_SWAP:
        for p in range(PAIR_HW):
            var r = hwc[off + p * 3]
            var g = hwc[off + p * 3 + 1]
            var b = hwc[off + p * 3 + 2]
            hwc[off + p * 3] = b
            hwc[off + p * 3 + 1] = r
            hwc[off + p * 3 + 2] = g
    elif shift.kind == SHIFT_NOISE:
        var z = gauss(seed, n, 0)
        for i in range(n):
            var v = Float64(hwc[off + i]) + shift.strength * 255.0 * Float64(z[i])
            hwc[off + i] = Scalar[DT](min(255.0, max(0.0, v)))


# ── test-time adaptation windows ──────────────────────────────────────────

comptime TTA_ACT = 10
"""One executed action block: frameskip 5 x 2 dims."""


def tta_windows[TB: Int, T: Int](
    frames: List[List[Scalar[DT]]], acts: List[List[Scalar[DT]]],
) -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    """An episode's newest TB windows of T consecutive frames, for the
    pretraining loss: pixels (TB, T, 3, 224, 224) ImageNet-normalised and
    actions (TB, T, 10).

    `frames[j]` is the observation after j executed blocks; `acts[j]` the
    block executed FROM frame j (it produced frame j + 1) — the dataset's
    alignment (`LewmPushTExpert`: action t is the dense span starting at frame
    t). Window b ends at frame n − 1 − b; with fewer than TB windows they are
    cycled. A window's LAST action is the one after its last frame: not yet
    taken for the newest window, so zero — the loss never reads it (the
    predictor sees the first T − 1 actions). Needs len(frames) >= T."""
    var n_win = len(frames) - T + 1
    var pix = List[Scalar[DT]](capacity=TB * T * 3 * PAIR_HW)
    var act = List[Scalar[DT]](capacity=TB * T * TTA_ACT)
    for b in range(TB):
        var k = len(frames) - T - (b % n_win)
        for t in range(T):
            var x = imagenet_from_hwc255(frames[k + t], 0)
            for v in x:
                pix.append(v)
            for j in range(TTA_ACT):
                act.append(acts[k + t][j] if k + t < len(acts) else Scalar[DT](0))
    return (pix^, act^)
