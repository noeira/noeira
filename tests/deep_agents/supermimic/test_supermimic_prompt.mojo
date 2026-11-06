"""SuperMimic's prompt interface: the task reward (`deep_agents/supermimic/prompt`).

    uv run --no-project --python 3.12 --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/prompt_oracle.py --out /tmp/sm_prompt
    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_prompt.mojo /tmp/sm_prompt

The oracle is a second implementation: numpy formulas from the paper's Table
A.6 and the module's docstring, MuJoCo 3.12.0 for the kinematics.

WHAT EACH SECTION CAN CATCH
===========================
[1] the hands' FK rows, resolved by name on both sides.
[2] float64 forward against the oracle: the 12 atoms, the reward, the fall
    logit, on upright, tilted and arbitrary states, past the soft limits too.
[3] float32 forward against the same values: the kernels run in float32.
[4] float64 VJP against central differences of our forward, 8 random
    directions of (s, a, a_prev) per lane, under random cotangents of (reward,
    logit, observation).
[5] a whole-robot yaw leaves every atom and the reward unchanged: the heading
    frame's point, and a property no fixed lane set shows by itself.
[6] a spec whose targets are the state's own atoms pays exactly Σ w (the
    kernels are 1), and dropping a group's weight removes exactly its term.
[7] `sm_prompt_fell` is Table A.6's rule on a sweep past both thresholds.

Planted mutants this gate was run against are listed in the plan's progress
notes: without them it says nothing about its own power.
"""

from std.math import cos, sin
from std.random import random_float64, seed
from std.sys import argv

from noeira.deep_agents.supermimic.heads import SMHeadLayout, sm_observation_lane
from noeira.deep_agents.supermimic.kinematics import FP, Q4, sm_fk_lane
from noeira.deep_agents.supermimic.prompt import (
    SMPromptScratch, SMSpec, sm_prompt_atoms_lane, sm_prompt_fall_logit, sm_prompt_fell, sm_prompt_reward_lane,
    sm_prompt_reward_vjp_lane,
)
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, G1MjlabSim, SMDims
from noeira.deep_agents.supermimic.tables import SMTables, sm_tables
from noeira.io.npy import load_npy_2d_f64


comptime R = G1MjlabRobot
comptime D = SMDims[R]
comptime SC = SMPromptScratch[R]
comptime F64 = DType.float64
comptime F32 = DType.float32


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


@always_inline
def _p[dt: DType](mut l: List[Scalar[dt]]) -> Pointer[Scalar[dt], MutAnyOrigin]:
    return l.unsafe_ptr().as_unsafe_any_origin()


def _row[dt: DType](src: List[Float64], b: Int, n: Int) -> List[Scalar[dt]]:
    var o = List[Scalar[dt]](length=n, fill=Scalar[dt](0))
    for i in range(n):
        o[i] = Scalar[dt](src[b * n + i])
    return o^


def _objective(
    mut tb: SMTables[F64], mut s: List[Float64], mut a: List[Float64], mut ap: List[Float64], mut xi: List[Float64],
    mut spec: List[Float64], gr: Float64, gl: Float64, go: List[Float64], mut sc: List[Float64], mut obs: List[Float64],
) -> Float64:
    """gr · reward + gl · logit + <go, observation>."""
    var r = sm_prompt_reward_lane[F64, R](tb.tree_ptr(), tb.head_ptr(), _p(s), _p(a), _p(ap), _p(spec), _p(sc))
    sm_observation_lane[F64, R](tb.head_ptr(), _p(s), _p(a), _p(xi), _p(obs))
    var f = gr * Float64(r) + gl * Float64(sm_prompt_fall_logit[F64](_p(s)))
    for k in range(D.OBS):
        f += go[k] * obs[k]
    return f


def _yaw_quat(w: Float64, x: Float64, y: Float64, z: Float64, th: Float64) -> List[Float64]:
    """(cos(th/2), 0, 0, sin(th/2)) ⊗ q."""
    var c = cos(th / 2)
    var sn = sin(th / 2)
    return [c * w - sn * z, c * x - sn * y, c * y + sn * x, c * z + sn * w]


def main() raises:
    var av = argv()
    if len(av) < 2:
        raise Error("give the oracle's dump directory")
    var dump = String(av[1])
    var t = Tally()
    seed(11)
    var ss = load_npy_2d_f64(dump + "/p_s.npy", D.S)
    var B = len(ss.values) // D.S
    var w_a = load_npy_2d_f64(dump + "/p_a.npy", D.NDOF).values.copy()
    var w_ap = load_npy_2d_f64(dump + "/p_ap.npy", D.NDOF).values.copy()
    var w_xi = load_npy_2d_f64(dump + "/p_xi.npy", D.X).values.copy()
    var w_spec = load_npy_2d_f64(dump + "/p_spec.npy", SMSpec.SIZE).values.copy()
    var w_atoms = load_npy_2d_f64(dump + "/p_atoms.npy", SMSpec.ATOMS).values.copy()
    var w_r = load_npy_2d_f64(dump + "/p_r.npy", 1).values.copy()
    var w_lg = load_npy_2d_f64(dump + "/p_logit.npy", 1).values.copy()
    var w_h = load_npy_2d_f64(dump + "/p_hands.npy", 2).values.copy()
    var sim = R.SIM.create()
    var q0 = List[Float64](length=D.NDOF, fill=0.0)
    for j in range(D.NDOF):
        q0[j] = R.default_joint_pos(j)
    var tb = sm_tables[F64, R](sim, q0)
    var tb32 = sm_tables[F32, R](sim, q0)
    comptime L = SMHeadLayout[R]

    print("\n[1] the hands' FK rows")
    t.truth(Float64(tb.head[L.HANDS]) == w_h[0] and Float64(tb.head[L.HANDS + 1]) == w_h[1], "left and right, by name")

    print("\n[2] float64 forward,", B, "lanes")
    var e_a = 0.0
    var e_r = 0.0
    var e_l = 0.0
    var sc = List[Float64](length=SC.SIZE, fill=0.0)
    for b in range(B):
        var s = _row[F64](ss.values, b, D.S)
        var a = _row[F64](w_a, b, D.NDOF)
        var ap = _row[F64](w_ap, b, D.NDOF)
        var spec = _row[F64](w_spec, b, SMSpec.SIZE)
        var r = sm_prompt_reward_lane[F64, R](tb.tree_ptr(), tb.head_ptr(), _p(s), _p(a), _p(ap), _p(spec), _p(sc))
        for k in range(SMSpec.ATOMS):
            e_a = max(e_a, abs(sc[SC.ATOMS + k] - w_atoms[b * SMSpec.ATOMS + k]))
        e_r = max(e_r, abs(Float64(r) - w_r[b]) / max(abs(w_r[b]), 1.0))
        e_l = max(e_l, abs(Float64(sm_prompt_fall_logit[F64](_p(s))) - w_lg[b]) / max(abs(w_lg[b]), 1.0))
    print("  atoms", e_a, "reward", e_r, "logit", e_l)
    t.truth(e_a < 1e-9, "the 12 atoms against numpy + MuJoCo")
    t.truth(e_r < 1e-9, "the reward against Table A.6")
    t.truth(e_l < 1e-9, "the fall logit")

    print("\n[3] float32 forward")
    var e32 = 0.0
    var sc32 = List[Float32](length=SC.SIZE, fill=0)
    for b in range(B):
        var s = _row[F32](ss.values, b, D.S)
        var a = _row[F32](w_a, b, D.NDOF)
        var ap = _row[F32](w_ap, b, D.NDOF)
        var spec = _row[F32](w_spec, b, SMSpec.SIZE)
        var r = sm_prompt_reward_lane[F32, R](tb32.tree_ptr(), tb32.head_ptr(), _p(s), _p(a), _p(ap), _p(spec), _p(sc32))
        e32 = max(e32, abs(Float64(r) - w_r[b]) / max(abs(w_r[b]), 1.0))
    print("  reward", e32)
    t.truth(e32 < 2e-4, "the reward in float32")

    print("\n[4] float64 VJP against central differences")
    var worst = 0.0
    var n_dir = 8
    var h = 1e-6
    var obs = List[Float64](length=D.OBS, fill=0.0)
    for b in range(0, B, 3):
        for shift in range(2):       # one lane of each kind (upright, tilted, anywhere) per block
            var lane = min(b + shift, B - 1)
            var s = _row[F64](ss.values, lane, D.S)
            var a = _row[F64](w_a, lane, D.NDOF)
            var ap = _row[F64](w_ap, lane, D.NDOF)
            var xi = _row[F64](w_xi, lane, D.X)
            var spec = _row[F64](w_spec, lane, SMSpec.SIZE)
            var gr = random_float64(-3, 3)
            var gl = random_float64(-1e-3, 1e-3)
            var go = List[Float64](length=D.OBS, fill=0.0)
            for k in range(D.OBS):
                go[k] = random_float64(-1, 1) * 1e-2
            var gs = List[Float64](length=D.S, fill=0.0)
            var ga = List[Float64](length=D.NDOF, fill=0.0)
            var gap = List[Float64](length=D.NDOF, fill=0.0)
            _ = sm_prompt_reward_lane[F64, R](tb.tree_ptr(), tb.head_ptr(), _p(s), _p(a), _p(ap), _p(spec), _p(sc))
            sm_prompt_reward_vjp_lane[F64, R](
                tb.tree_ptr(), tb.head_ptr(), _p(s), _p(a), _p(ap), _p(spec), gr, gl, _p(go), _p(sc), _p(gs), _p(ga), _p(gap),
            )
            for _ in range(n_dir):
                var ds = List[Float64](length=D.S, fill=0.0)
                var da = List[Float64](length=D.NDOF, fill=0.0)
                var dap = List[Float64](length=D.NDOF, fill=0.0)
                var pred = 0.0
                for i in range(D.S):
                    ds[i] = random_float64(-1, 1)
                    pred += ds[i] * gs[i]
                for i in range(D.NDOF):
                    da[i] = random_float64(-1, 1)
                    dap[i] = random_float64(-1, 1)
                    pred += da[i] * ga[i] + dap[i] * gap[i]
                var sp = s.copy()
                var sm = s.copy()
                var apl = a.copy()
                var ami = a.copy()
                var appl = ap.copy()
                var apmi = ap.copy()
                for i in range(D.S):
                    sp[i] += h * ds[i]
                    sm[i] -= h * ds[i]
                for i in range(D.NDOF):
                    apl[i] += h * da[i]
                    ami[i] -= h * da[i]
                    appl[i] += h * dap[i]
                    apmi[i] -= h * dap[i]
                var fp = _objective(tb, sp, apl, appl, xi, spec, gr, gl, go, sc, obs)
                var fm = _objective(tb, sm, ami, apmi, xi, spec, gr, gl, go, sc, obs)
                var fd = (fp - fm) / (2 * h)
                worst = max(worst, abs(fd - pred) / max(abs(fd), 1e-3))
    print("  worst relative error", worst)
    t.truth(worst < 2e-6, "∂(reward, logit, obs) / ∂(s, a, a_prev)")

    print("\n[5] a whole-robot yaw")
    var e_y = 0.0
    for b in range(0, B, 7):
        var s = _row[F64](ss.values, b, D.S)
        var a = _row[F64](w_a, b, D.NDOF)
        var ap = _row[F64](w_ap, b, D.NDOF)
        var spec = _row[F64](w_spec, b, SMSpec.SIZE)
        var r0 = sm_prompt_reward_lane[F64, R](tb.tree_ptr(), tb.head_ptr(), _p(s), _p(a), _p(ap), _p(spec), _p(sc))
        var at0 = List[Float64](length=SMSpec.ATOMS, fill=0.0)
        for k in range(SMSpec.ATOMS):
            at0[k] = sc[SC.ATOMS + k]
        var th = random_float64(-3, 3)
        var c = cos(th)
        var sn = sin(th)
        var s2 = s.copy()
        s2[0] = c * s[0] - sn * s[1]
        s2[1] = sn * s[0] + c * s[1]
        var q = _yaw_quat(s[3], s[4], s[5], s[6], th)
        for k in range(4):
            s2[3 + k] = q[k]
        s2[D.NQ] = c * s[D.NQ] - sn * s[D.NQ + 1]
        s2[D.NQ + 1] = sn * s[D.NQ] + c * s[D.NQ + 1]
        var r1 = sm_prompt_reward_lane[F64, R](tb.tree_ptr(), tb.head_ptr(), _p(s2), _p(a), _p(ap), _p(spec), _p(sc))
        e_y = max(e_y, abs(Float64(r1) - Float64(r0)))
        for k in range(SMSpec.ATOMS):
            e_y = max(e_y, abs(sc[SC.ATOMS + k] - at0[k]))
    print("  worst change", e_y)
    t.truth(e_y < 1e-9, "atoms and reward are heading-invariant")

    print("\n[6] a spec met exactly")
    var s6 = _row[F64](ss.values, 0, D.S)
    var a6 = _row[F64](w_a, 0, D.NDOF)
    var ap6 = a6.copy()              # no action-rate term
    var spec6 = List[Float64](length=SMSpec.SIZE, fill=0.0)
    var r_none = sm_prompt_reward_lane[F64, R](tb.tree_ptr(), tb.head_ptr(), _p(s6), _p(a6), _p(ap6), _p(spec6), _p(sc))
    var wsum = 0.0
    for k in range(SMSpec.ATOMS):
        spec6[SMSpec.T + k] = sc[SC.ATOMS + k]
        spec6[SMSpec.IW + k] = 50.0
    var r_zero = sm_prompt_reward_lane[F64, R](tb.tree_ptr(), tb.head_ptr(), _p(s6), _p(a6), _p(ap6), _p(spec6), _p(sc))
    t.truth(abs(Float64(r_zero) - Float64(r_none)) < 1e-12, "all weights 0: the task pays nothing, whatever the targets")
    for g in range(SMSpec.GROUPS):
        spec6[SMSpec.W + g] = 1.0 + Float64(g)
        wsum += 1.0 + Float64(g)
    var r_all = sm_prompt_reward_lane[F64, R](tb.tree_ptr(), tb.head_ptr(), _p(s6), _p(a6), _p(ap6), _p(spec6), _p(sc))
    t.truth(abs((Float64(r_all) - Float64(r_none)) - 0.02 * wsum) < 1e-12, "targets = atoms: dt · Σ w, every group at once")
    spec6[SMSpec.W + SMSpec.G_RHAND] = 0.0
    var r_drop = sm_prompt_reward_lane[F64, R](tb.tree_ptr(), tb.head_ptr(), _p(s6), _p(a6), _p(ap6), _p(spec6), _p(sc))
    t.truth(abs((Float64(r_all) - Float64(r_drop)) - 0.02 * 6.0) < 1e-12, "a group's weight removes exactly its term")

    print("\n[7] the fall rule")
    var s7 = List[Float64](length=D.S, fill=0.0)
    s7[3] = 1.0
    s7[2] = 0.78
    var up_ok = not sm_prompt_fell[F64](_p(s7))
    s7[2] = 0.34
    var low_ok = sm_prompt_fell[F64](_p(s7))
    s7[2] = 0.36
    var above_ok = not sm_prompt_fell[F64](_p(s7))
    # tilt about x: 1 + u_z = 1 − cos θ, the rule fires past acos(0.3) ≈ 72.5°
    s7[2] = 0.78
    var a0 = 72.0 * 3.141592653589793 / 180.0
    s7[3] = cos(a0 / 2)
    s7[4] = sin(a0 / 2)
    var t_in = not sm_prompt_fell[F64](_p(s7))
    var a1 = 73.0 * 3.141592653589793 / 180.0
    s7[3] = cos(a1 / 2)
    s7[4] = sin(a1 / 2)
    var t_out = sm_prompt_fell[F64](_p(s7))
    t.truth(up_ok and low_ok and above_ok, "height: 0.35 m")
    t.truth(t_in and t_out, "tilt: 1 + u_z = 0.7, between 72° and 73°")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_prompt: " + String(t.fails) + " failed")
