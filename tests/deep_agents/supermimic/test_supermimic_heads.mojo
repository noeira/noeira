"""SuperMimic M3b: the analytic heads and their VJP (`deep_agents/supermimic/heads`).

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/heads_oracle.py --out /tmp/sm_heads
    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_heads.mojo /tmp/sm_heads

The oracle is the reproduction's `AnalyticHeads.reward`,
`termination_features` and `analytic_observation` run verbatim (torch,
float64) on our MjModel, with mjlab's quaternion helpers at the pinned commit.
Its lanes are a third near the reference, a third far, and a third within
~1e-9 of it, where mjlab's `axis_angle_from_quat` takes its Taylor branch.

WHAT EACH SECTION CAN CATCH
===========================
[1] the per-model table: the 14 tracked bodies resolve to the reproduction's
    FK rows on our model (by NAME on its side, by `g1_skeleton_body` on
    ours), and mjlab's soft joint limits (range × 0.9 about its midpoint).
[2] float64 forward: the 8 weighted terms, the total, the 2 termination
    features and the 93-d observation, 1e-12.
[3] float64 VJP against torch autograd: d/ds (71), d/da, d/da_prev, 1e-9
    relative, on every lane kind (Taylor lanes included).
[4] float64 VJP against central differences of our own forward along 16
    random directions of (s, a, a_prev), near and far lanes: 1e-6 relative.
[5] float32 forward against the float64 oracle: 1e-5 relative.
[6] the quaternion error on its own, both branches, both signs of w: its
    gradient against central differences scaled to the input (in the Taylor
    region the error is ~1e-14, so [3] cannot see a wrong derivative there),
    and continuity of value and gradient across the |angle| = 1e-6 switch.

WHAT IT CANNOT CATCH (measured with planted mutants)
====================================================
A sign error in the Taylor branch's ∂e/∂w survives every section. That
component is O(|xyz|³·angle) ≈ 1e-20 where the branch applies, 13 orders
below the others, so it moves no computed value by more than rounding and no
float64 central difference resolves it. Three other planted VJP errors fail
[3] and [4]: the yaw-quaternion partial's sign, the body-orientation chain
taking q̄ for q, and the observation's gravity using the NORMALISED root.
"""

from std.math import cos, sin
from std.random import random_float64, seed
from std.sys import argv

from noeira.deep_agents.supermimic.g1_fk import G1FKTree, g1_fk_tree
from noeira.deep_agents.supermimic.heads import (
    G1HeadsTable, HT_HI, HT_LO, HT_TRACKED, SM_N_DOF, SM_N_TERMS, SM_N_TRACKED, SM_OBS_DIM,
    SM_REF_DIM, SM_SCRATCH, SM_STATE_DIM, SM_XI_DIM, g1_heads_lane, g1_heads_table, g1_heads_vjp_lane,
    sq_quat_err, sq_quat_err_vjp,
)
from noeira.deep_agents.supermimic.g1_fk import Q4
from noeira.envs.robots import UnitreeG1
from noeira.io.npy import load_npy_2d_f64


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


struct Lane[dt: DType](Movable):
    """One lane's inputs and forward outputs."""
    var s: List[Scalar[Self.dt]]
    var a: List[Scalar[Self.dt]]
    var ap: List[Scalar[Self.dt]]
    var xi: List[Scalar[Self.dt]]
    var rf: List[Scalar[Self.dt]]
    var terms: List[Scalar[Self.dt]]
    var feat: List[Scalar[Self.dt]]
    var obs: List[Scalar[Self.dt]]
    var scratch: List[Scalar[Self.dt]]
    var total: Scalar[Self.dt]

    def __init__(out self, d: Dump, b: Int):
        self.s = _row[Self.dt](d.s, b, SM_STATE_DIM)
        self.a = _row[Self.dt](d.a, b, SM_N_DOF)
        self.ap = _row[Self.dt](d.ap, b, SM_N_DOF)
        self.xi = _row[Self.dt](d.xi, b, SM_XI_DIM)
        self.rf = _row[Self.dt](d.rf, b, SM_REF_DIM)
        self.terms = List[Scalar[Self.dt]](length=SM_N_TERMS, fill=Scalar[Self.dt](0))
        self.feat = List[Scalar[Self.dt]](length=2, fill=Scalar[Self.dt](0))
        self.obs = List[Scalar[Self.dt]](length=SM_OBS_DIM, fill=Scalar[Self.dt](0))
        self.scratch = List[Scalar[Self.dt]](length=SM_SCRATCH, fill=Scalar[Self.dt](0))
        self.total = 0

    def __init__(out self, *, deinit move: Self):
        self.s = move.s^
        self.a = move.a^
        self.ap = move.ap^
        self.xi = move.xi^
        self.rf = move.rf^
        self.terms = move.terms^
        self.feat = move.feat^
        self.obs = move.obs^
        self.scratch = move.scratch^
        self.total = move.total

    def forward(mut self, mut tree: G1FKTree[Self.dt], mut table: G1HeadsTable[Self.dt]) where Self.dt.is_floating_point():
        self.total = g1_heads_lane[Self.dt](
            tree.ptr(), table.ptr(), _p(self.s), _p(self.a), _p(self.ap), _p(self.xi), _p(self.rf),
            _p(self.terms), _p(self.feat), _p(self.obs), _p(self.scratch),
        )


struct Dump(Movable):
    var B: Int
    var s: List[Float64]
    var a: List[Float64]
    var ap: List[Float64]
    var xi: List[Float64]
    var rf: List[Float64]

    def __init__(out self, dump: String) raises:
        var x = load_npy_2d_f64(dump + "/h_s.npy", SM_STATE_DIM)
        self.B = x.rows
        self.s = x.values.copy()
        self.a = load_npy_2d_f64(dump + "/h_a.npy", SM_N_DOF).values.copy()
        self.ap = load_npy_2d_f64(dump + "/h_a_prev.npy", SM_N_DOF).values.copy()
        self.xi = load_npy_2d_f64(dump + "/h_xi.npy", SM_XI_DIM).values.copy()
        self.rf = load_npy_2d_f64(dump + "/h_ref.npy", SM_REF_DIM).values.copy()

    def __init__(out self, *, deinit move: Self):
        self.B = move.B
        self.s = move.s^
        self.a = move.a^
        self.ap = move.ap^
        self.xi = move.xi^
        self.rf = move.rf^


def _rel(got: Float64, want: Float64) -> Float64:
    return abs(got - want) / max(1.0, abs(want))


def _objective(mut l: Lane[DType.float64], gr: Float64, gf: List[Float64], go: List[Float64], b: Int) -> Float64:
    var v = gr * l.total
    for k in range(2):
        v += gf[b * 2 + k] * l.feat[k]
    for k in range(SM_OBS_DIM):
        v += go[b * SM_OBS_DIM + k] * l.obs[k]
    return v


def main() raises:
    var av = argv()
    var dump = String(av[1]) if len(av) > 1 else String("/tmp/sm_heads")
    var t = Tally()
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var d = Dump(dump)
    var B = d.B
    var third = B // 3

    print("\n[1] the per-model table")
    var q0 = load_npy_2d_f64(dump + "/h_q0.npy", SM_N_DOF).values.copy()
    var tree = g1_fk_tree[DType.float64](env)
    var table = g1_heads_table[DType.float64](env, q0)
    var tr = load_npy_2d_f64(dump + "/h_tracked.npy", SM_N_TRACKED)
    var bad = 0
    for k in range(SM_N_TRACKED):
        if Int(table.data[HT_TRACKED + k]) != Int(tr.values[k]):
            bad += 1
    t.truth(bad == 0, "14 tracked bodies = the reproduction's rows by name")
    var soft = load_npy_2d_f64(dump + "/h_soft.npy", 2)
    var e_soft = 0.0
    for j in range(SM_N_DOF):
        e_soft = max(e_soft, abs(Float64(table.data[HT_LO + j]) - soft.values[j * 2]))
        e_soft = max(e_soft, abs(Float64(table.data[HT_HI + j]) - soft.values[j * 2 + 1]))
    t.truth(e_soft <= 1e-12, "soft joint limits (x0.9), max err " + String(e_soft))

    print("\n[2] float64 forward,", B, "lanes")
    var w_terms = load_npy_2d_f64(dump + "/h_terms.npy", SM_N_TERMS).values.copy()
    var w_total = load_npy_2d_f64(dump + "/h_total.npy", 1).values.copy()
    var w_feat = load_npy_2d_f64(dump + "/h_feat.npy", 2).values.copy()
    var w_obs = load_npy_2d_f64(dump + "/h_obs.npy", SM_OBS_DIM).values.copy()
    var e_t = 0.0
    var e_f = 0.0
    var e_o = 0.0
    var lanes = List[Lane[DType.float64]]()
    for b in range(B):
        var l = Lane[DType.float64](d, b)
        l.forward(tree, table)
        for k in range(SM_N_TERMS):
            e_t = max(e_t, _rel(Float64(l.terms[k]), w_terms[b * SM_N_TERMS + k]))
        e_t = max(e_t, _rel(Float64(l.total), w_total[b]))
        for k in range(2):
            e_f = max(e_f, _rel(Float64(l.feat[k]), w_feat[b * 2 + k]))
        for k in range(SM_OBS_DIM):
            e_o = max(e_o, _rel(Float64(l.obs[k]), w_obs[b * SM_OBS_DIM + k]))
        lanes.append(l^)
    t.truth(e_t <= 1e-12, "8 terms + total: max err " + String(e_t))
    t.truth(e_f <= 1e-12, "termination features: max err " + String(e_f))
    t.truth(e_o <= 1e-12, "observation: max err " + String(e_o))

    print("\n[3] float64 VJP against torch autograd")
    var g_r = load_npy_2d_f64(dump + "/h_g_r.npy", 1).values.copy()
    var g_f = load_npy_2d_f64(dump + "/h_g_feat.npy", 2).values.copy()
    var g_o = load_npy_2d_f64(dump + "/h_g_obs.npy", SM_OBS_DIM).values.copy()
    var w_gs = load_npy_2d_f64(dump + "/h_g_s.npy", SM_STATE_DIM).values.copy()
    var w_ga = load_npy_2d_f64(dump + "/h_g_a.npy", SM_N_DOF).values.copy()
    var w_gap = load_npy_2d_f64(dump + "/h_g_a_prev.npy", SM_N_DOF).values.copy()
    var e_kind = List[Float64](length=3, fill=0.0)
    var gs_all = List[Float64](length=B * SM_STATE_DIM, fill=0.0)
    var ga_all = List[Float64](length=B * SM_N_DOF, fill=0.0)
    var gap_all = List[Float64](length=B * SM_N_DOF, fill=0.0)
    for b in range(B):
        var gf = _row[DType.float64](g_f, b, 2)
        var go = _row[DType.float64](g_o, b, SM_OBS_DIM)
        var gs = List[Float64](length=SM_STATE_DIM, fill=0.0)
        var ga = List[Float64](length=SM_N_DOF, fill=0.0)
        var gap = List[Float64](length=SM_N_DOF, fill=0.0)
        g1_heads_vjp_lane[DType.float64](
            tree.ptr(), table.ptr(), _p(lanes[b].s), _p(lanes[b].a), _p(lanes[b].ap), _p(lanes[b].rf),
            g_r[b], _p(gf), _p(go), _p(lanes[b].scratch), _p(gs), _p(ga), _p(gap),
        )
        var kind = 0 if b < third else (1 if b < 2 * third else 2)
        for k in range(SM_STATE_DIM):
            gs_all[b * SM_STATE_DIM + k] = gs[k]
            e_kind[kind] = max(e_kind[kind], _rel(gs[k], w_gs[b * SM_STATE_DIM + k]))
        for k in range(SM_N_DOF):
            ga_all[b * SM_N_DOF + k] = ga[k]
            gap_all[b * SM_N_DOF + k] = gap[k]
            e_kind[kind] = max(e_kind[kind], _rel(ga[k], w_ga[b * SM_N_DOF + k]))
            e_kind[kind] = max(e_kind[kind], _rel(gap[k], w_gap[b * SM_N_DOF + k]))
    t.truth(e_kind[0] <= 1e-9, "near lanes: max rel err " + String(e_kind[0]))
    t.truth(e_kind[1] <= 1e-9, "far lanes: max rel err " + String(e_kind[1]))
    t.truth(e_kind[2] <= 1e-9, "Taylor-branch lanes: max rel err " + String(e_kind[2]))

    print("\n[4] float64 VJP against central differences of our forward")
    seed(9)
    var h = 1e-6
    var e_fd = 0.0
    for n in range(16):
        var b = n % (2 * third)
        var ds = List[Float64](length=SM_STATE_DIM, fill=0.0)
        var da = List[Float64](length=SM_N_DOF, fill=0.0)
        var dap = List[Float64](length=SM_N_DOF, fill=0.0)
        for k in range(SM_STATE_DIM):
            ds[k] = random_float64(-1.0, 1.0)
        for k in range(SM_N_DOF):
            da[k] = random_float64(-1.0, 1.0)
            dap[k] = random_float64(-1.0, 1.0)
        var v = List[Float64]()
        for sgn in range(2):
            var sc = h if sgn == 0 else -h
            var l = Lane[DType.float64](d, b)
            for k in range(SM_STATE_DIM):
                l.s[k] += sc * ds[k]
            for k in range(SM_N_DOF):
                l.a[k] += sc * da[k]
                l.ap[k] += sc * dap[k]
            l.forward(tree, table)
            v.append(_objective(l, g_r[b], g_f, g_o, b))
        var fd = (v[0] - v[1]) / (2.0 * h)
        var an = 0.0
        for k in range(SM_STATE_DIM):
            an += gs_all[b * SM_STATE_DIM + k] * ds[k]
        for k in range(SM_N_DOF):
            an += ga_all[b * SM_N_DOF + k] * da[k] + gap_all[b * SM_N_DOF + k] * dap[k]
        e_fd = max(e_fd, abs(fd - an) / max(1.0, abs(an)))
    t.truth(e_fd <= 1e-6, "16 directions over near and far lanes: max rel err " + String(e_fd))

    print("\n[5] float32 forward")
    var tree32 = g1_fk_tree[DType.float32](env)
    var table32 = g1_heads_table[DType.float32](env, q0)
    var e32 = 0.0
    for b in range(B):
        var l = Lane[DType.float32](d, b)
        l.forward(tree32, table32)
        for k in range(SM_N_TERMS):
            e32 = max(e32, _rel(Float64(l.terms[k]), w_terms[b * SM_N_TERMS + k]))
        for k in range(2):
            e32 = max(e32, _rel(Float64(l.feat[k]), w_feat[b * 2 + k]))
        for k in range(SM_OBS_DIM):
            e32 = max(e32, _rel(Float64(l.obs[k]), w_obs[b * SM_OBS_DIM + k]))
    t.truth(e32 <= 1e-5, "terms, features, observation: max rel err " + String(e32))


    print("\n[6] the quaternion error, both branches")
    var e_q = 0.0
    var cont = 0.0
    var angles: List[Float64] = [0.7, 2.9, 3e-4, 9.0e-7, 4.0e-7, 1e-9]
    for ia in range(len(angles)):
        for sw in range(2):
            var ang = angles[ia]
            var ax = Q4[DType.float64](0.48, -0.6, 0.64, 0)    # unit
            var sg = 1.0 if sw == 0 else -1.0
            var n = 1.0 + 0.01 * Float64(ia)                   # slightly non-unit too
            var dq = Q4[DType.float64](sg * n * cos(ang / 2), sg * n * sin(ang / 2) * ax[0], sg * n * sin(ang / 2) * ax[1], sg * n * sin(ang / 2) * ax[2])
            var g = sq_quat_err_vjp(dq)
            var hstep = 1e-6 * max(ang, 1e-6)
            for c in range(4):
                var up = dq
                var dn = dq
                up[c] += hstep
                dn[c] -= hstep
                var fd = (sq_quat_err(up) - sq_quat_err(dn)) / (2.0 * hstep)
                var scale = max(abs(g[0]) + abs(g[1]) + abs(g[2]) + abs(g[3]), 1e-300)
                e_q = max(e_q, abs(fd - g[c]) / scale)
    # across the switch: angle 1e-6 (1 ± 1e-9) sits on either side
    for side in range(2):
        var lo = Q4[DType.float64](cos(0.5e-6 * (1 - 1e-9)), sin(0.5e-6 * (1 - 1e-9)), 0, 0)
        var hi = Q4[DType.float64](cos(0.5e-6 * (1 + 1e-9)), sin(0.5e-6 * (1 + 1e-9)), 0, 0)
        if side == 1:
            lo = -lo
            hi = -hi
        cont = max(cont, abs(sq_quat_err(lo) - sq_quat_err(hi)) / sq_quat_err(hi))
        var gl = sq_quat_err_vjp(lo)
        var gh = sq_quat_err_vjp(hi)
        for c in range(4):
            cont = max(cont, abs(gl[c] - gh[c]) / max(abs(gh[1]), 1e-300))
    t.truth(e_q <= 1e-6, "gradient vs central differences, 6 angles x 2 signs: max rel err " + String(e_q))
    t.truth(cont <= 1e-6, "value and gradient continuous across the 1e-6 switch: max rel jump " + String(cont))

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_supermimic_heads: " + String(t.fails) + " failed")
