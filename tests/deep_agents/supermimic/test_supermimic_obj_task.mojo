"""The object interface's task terms, on the CPU.

`object_task.mojo`: the reward, the encoder's features, the reference rows.
Float64, no reference files:
known answers and central differences. (The independent transcription is the
rollout's oracle, `test_supermimic_obj_imag`.)

[1] the reward at the reference: object on its row, hands where the row has
    them: each term is dt·w. One term off by exactly its width σ: that term
    is dt·w·e⁻¹ and the other two keep their value. The half-turn caveat: a
    box turned π about its z scores the orientation term as far.
[2] the reward's VJP against central differences, over the object's 7 pose
    entries and the two hands' positions, 12 random lanes.
[3] the features: the first 15 are `sm_obj_feat_lane(s, o)`, the next 15 the
    same of the reference, the last 3 their positions' difference over 0.1;
    their VJP against central differences over the robot's root pose and the
    object's state.
[4] the reference rows on the G1 with its box: with the robot at a library
    row and the object on its track row, the hands term is at its maximum in
    every frame; the first 13 numbers are the track's.

Run: pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_obj_task.mojo
"""

from std.math import cos, exp, sin, sqrt
from std.random import random_float64, seed

from noeira.deep_agents.supermimic.heads import SMHeadLayout
from noeira.deep_agents.supermimic.kinematics import FP, sm_fk_lane
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_FEAT, sm_obj_feat_lane
from noeira.deep_agents.supermimic.object_task import (
    OT_HANDS, OT_ORI, OT_POS, SM_OBJ_TERMS, SM_OE_FEAT, SM_OE_SCRATCH, SM_OREF, SM_ORW, SM_OSTD_HANDS, SM_OSTD_ORI,
    SM_OSTD_POS, SM_OW_HANDS, SM_OW_ORI, SM_OW_POS, sm_object_ref_rows, sm_obj_reward_lane, sm_obj_reward_params,
    sm_obj_reward_vjp_lane, sm_oe_feat_lane, sm_oe_feat_vjp_lane,
)
from noeira.deep_agents.supermimic.robot import G1MjlabBoxRobot, SMDims
from noeira.deep_agents.supermimic.tables import sm_tables
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr


comptime R = G1MjlabBoxRobot
comptime D = SMDims[R]
comptime L = SMHeadLayout[R]
comptime F64 = DType.float64
comptime HL = 5              # the two hand rows of the made table
comptime HR = 11
comptime EPS = 1e-6
# [1]'s band: the reward's data (dt·w, 1/σ²) is float32 (`sm_obj_reward_params`), so dt·w is 0.02 to 4.5e-10
comptime TOL = 1e-8


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
def _p(mut l: List[Float64], off: Int = 0) -> FP[F64]:
    return l.unsafe_ptr().unsafe_offset(off).as_unsafe_any_origin()


def _rand(n: Int, lo: Float64, hi: Float64) -> List[Float64]:
    var out = List[Float64]()
    for _ in range(n):
        out.append(random_float64(lo, hi))
    return out^


def _unit4(mut v: List[Float64], off: Int):
    var n = 0.0
    for k in range(4):
        n += v[off + k] * v[off + k]
    n = sqrt(n)
    for k in range(4):
        v[off + k] = v[off + k] / n


def _table() -> List[Float64]:
    var t = List[Float64](length=L.SIZE, fill=0.0)
    t[L.HANDS] = Float64(HL)
    t[L.HANDS + 1] = Float64(HR)
    return t^


def _rw() -> List[Float64]:
    var p = sm_obj_reward_params()
    var out = List[Float64]()
    for k in range(SM_ORW):
        out.append(Float64(p[k]))
    return out^


def _random_object() -> List[Float64]:
    var o = _rand(SM_OBJ, -0.5, 0.5)
    o[2] += 1.0
    _unit4(o, 3)
    return o^


def _reward(mut table: List[Float64], mut pos: List[Float64], mut o: List[Float64], mut oref: List[Float64], mut rw: List[Float64]) -> List[Float64]:
    """[the three terms, their sum]."""
    var terms = List[Float64](length=SM_OBJ_TERMS + 1, fill=0.0)
    terms[SM_OBJ_TERMS] = sm_obj_reward_lane[F64, R](_p(table), _p(pos), _p(o), _p(oref), _p(rw), _p(terms))
    return terms^


def _rotate(q: List[Float64], off: Int, v: List[Float64]) -> List[Float64]:
    """R(q) v, by the rotation matrix (not the code under test)."""
    var w = q[off]
    var x = q[off + 1]
    var y = q[off + 2]
    var z = q[off + 3]
    return [
        (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - z * w) * v[1] + 2 * (x * z + y * w) * v[2],
        2 * (x * y + z * w) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - x * w) * v[2],
        2 * (x * z - y * w) * v[0] + 2 * (y * z + x * w) * v[1] + (1 - 2 * (x * x + y * y)) * v[2],
    ]


def test_known(mut t: Tally) raises:
    print("\n[1] the reward at the reference, and one width away from it")
    seed(1)
    var table = _table()
    var rw = _rw()
    var dt = 0.02
    var oref = _rand(SM_OREF, -0.4, 0.4)
    oref[2] += 1.0
    _unit4(oref, 3)
    var o = List[Float64]()
    for k in range(SM_OBJ):
        o.append(oref[k])
    # the hands where the row has them: p_hand = p + R(q) h
    var pos = _rand(D.NB * 3, -1.0, 1.0)
    var hb: List[Int] = [HL, HR]
    for i in range(2):
        var h: List[Float64] = [oref[SM_OBJ + 3 * i], oref[SM_OBJ + 3 * i + 1], oref[SM_OBJ + 3 * i + 2]]
        var w = _rotate(o, 3, h)
        for c in range(3):
            pos[hb[i] * 3 + c] = o[c] + w[c]
    var r0 = _reward(table, pos, o, oref, rw)
    var e0 = max(abs(r0[OT_POS] - dt * SM_OW_POS), max(abs(r0[OT_ORI] - dt * SM_OW_ORI), abs(r0[OT_HANDS] - dt * SM_OW_HANDS)))
    t.truth(e0 <= TOL and abs(r0[3] - dt * (SM_OW_POS + SM_OW_ORI + SM_OW_HANDS)) <= TOL, "at the reference each term is dt·w: max |d| " + String(e0))
    # position: the object AND the hands moved by σ along x (the hands keep their place on it)
    var o1 = o.copy()
    var pos1 = pos.copy()
    o1[0] += SM_OSTD_POS
    for i in range(2):
        pos1[hb[i] * 3] += SM_OSTD_POS
    var r1 = _reward(table, pos1, o1, oref, rw)
    t.truth(
        abs(r1[OT_POS] - dt * SM_OW_POS * exp(-1.0)) <= TOL and abs(r1[OT_ORI] - r0[OT_ORI]) <= TOL and abs(r1[OT_HANDS] - r0[OT_HANDS]) <= TOL,
        "the object and its hands one σ away: position term dt·w/e (" + String(r1[OT_POS]) + "), the other two unchanged",
    )
    # hands: both off by σ in the object's frame along its x → e_h = ½ (σ² + σ²) = σ²
    var pos2 = pos.copy()
    var ex: List[Float64] = [SM_OSTD_HANDS, 0.0, 0.0]
    var wx = _rotate(o, 3, ex)
    for i in range(2):
        for c in range(3):
            pos2[hb[i] * 3 + c] += wx[c]
    var r2 = _reward(table, pos2, o, oref, rw)
    t.truth(
        abs(r2[OT_HANDS] - dt * SM_OW_HANDS * exp(-1.0)) <= TOL and abs(r2[OT_POS] - r0[OT_POS]) <= TOL and abs(r2[OT_ORI] - r0[OT_ORI]) <= TOL,
        "both hands one σ off in the object's frame: hands term dt·w/e (" + String(r2[OT_HANDS]) + "), the other two unchanged",
    )
    # orientation: turned by σ rad about the object's own z, hands turned with it
    var o3 = o.copy()
    var hz = 0.5 * SM_OSTD_ORI
    var qz: List[Float64] = [cos(hz), 0.0, 0.0, sin(hz)]
    # q' = q ⊗ qz
    var w0 = o[3]
    var x0 = o[4]
    var y0 = o[5]
    var z0 = o[6]
    o3[3] = w0 * qz[0] - z0 * qz[3]
    o3[4] = x0 * qz[0] + y0 * qz[3]
    o3[5] = y0 * qz[0] - x0 * qz[3]
    o3[6] = z0 * qz[0] + w0 * qz[3]
    var pos3 = pos.copy()
    for i in range(2):
        var h: List[Float64] = [oref[SM_OBJ + 3 * i], oref[SM_OBJ + 3 * i + 1], oref[SM_OBJ + 3 * i + 2]]
        var w = _rotate(o3, 3, h)
        for c in range(3):
            pos3[hb[i] * 3 + c] = o3[c] + w[c]
    var r3 = _reward(table, pos3, o3, oref, rw)
    t.truth(
        abs(r3[OT_ORI] - dt * SM_OW_ORI * exp(-1.0)) <= TOL and abs(r3[OT_HANDS] - r0[OT_HANDS]) <= TOL and abs(r3[OT_POS] - r0[OT_POS]) <= TOL,
        "turned σ rad about its own z with its hands: orientation term dt·w/e (" + String(r3[OT_ORI]) + "), the other two unchanged",
    )
    # the caveat: half a turn about z is the same cuboid and scores as π away
    var o4 = o.copy()
    o4[3] = -z0
    o4[4] = y0
    o4[5] = -x0
    o4[6] = w0
    var r4 = _reward(table, pos, o4, oref, rw)
    var want = dt * SM_OW_ORI * exp(-(3.141592653589793 * 3.141592653589793) / (SM_OSTD_ORI * SM_OSTD_ORI))
    t.truth(abs(r4[OT_ORI] - want) <= TOL, "half a turn about its z scores as π away (the docstring's caveat): " + String(r4[OT_ORI]))


def test_reward_vjp(mut t: Tally) raises:
    print("\n[2] the reward's VJP against central differences")
    seed(2)
    var table = _table()
    # wider kernels than the defaults, so random lanes are not all in the tails
    var p = sm_obj_reward_params(1.0, 0.8, 0.5, 2.0, 1.0, 1.2)
    var rw = List[Float64]()
    for k in range(SM_ORW):
        rw.append(Float64(p[k]))
    var worst_o = 0.0
    var worst_h = 0.0
    var big = 0.0
    var hb: List[Int] = [HL, HR]
    for _ in range(12):
        var o = _random_object()
        var oref = _rand(SM_OREF, -0.4, 0.4)
        oref[2] += 1.0
        _unit4(oref, 3)
        var pos = _rand(D.NB * 3, -0.6, 0.6)
        for i in range(2):
            pos[hb[i] * 3 + 2] += 1.0
        var g_r = random_float64(0.5, 2.0)
        var gpos = List[Float64](length=D.NB * 3, fill=0.0)
        var g_o = List[Float64](length=SM_OBJ, fill=0.0)
        sm_obj_reward_vjp_lane[F64, R](_p(table), _p(pos), _p(o), _p(oref), _p(rw), g_r, _p(gpos), _p(g_o))
        for k in range(7):
            var op = o.copy()
            var om = o.copy()
            op[k] += EPS
            om[k] -= EPS
            var fd = g_r * (_reward(table, pos, op, oref, rw)[3] - _reward(table, pos, om, oref, rw)[3]) / (2 * EPS)
            worst_o = max(worst_o, abs(fd - g_o[k]))
            big = max(big, abs(fd))
        for k in range(7, SM_OBJ):
            worst_o = max(worst_o, abs(g_o[k]))             # the velocities are not read
        for i in range(2):
            for c in range(3):
                var pp = pos.copy()
                var pm = pos.copy()
                pp[hb[i] * 3 + c] += EPS
                pm[hb[i] * 3 + c] -= EPS
                var fd = g_r * (_reward(table, pp, o, oref, rw)[3] - _reward(table, pm, o, oref, rw)[3]) / (2 * EPS)
                worst_h = max(worst_h, abs(fd - gpos[hb[i] * 3 + c]))
                big = max(big, abs(fd))
        var other = 0.0
        for b in range(D.NB):
            if b != HL and b != HR:
                for c in range(3):
                    other = max(other, abs(gpos[b * 3 + c]))
        worst_h = max(worst_h, other)
    t.truth(worst_o <= 1e-8 and big > 1e-3, "object pose (7) x 12 lanes: max |d| " + String(worst_o) + " (largest derivative " + String(big) + ")")
    t.truth(worst_h <= 1e-8, "the two hands' positions, and nothing on any other body: max |d| " + String(worst_h))


def _feat(mut s: List[Float64], mut o: List[Float64], mut oref: List[Float64]) -> List[Float64]:
    var f = List[Float64](length=SM_OE_FEAT, fill=0.0)
    sm_oe_feat_lane[F64, R](_p(s), _p(o), _p(oref), _p(f))
    return f^


def test_features(mut t: Tally) raises:
    print("\n[3] the encoder's features and their VJP")
    seed(3)
    var worst_def = 0.0
    var worst_s = 0.0
    var worst_o = 0.0
    var leak = 0.0
    for _ in range(12):
        var s = _rand(D.S, -0.5, 0.5)
        _unit4(s, 3)
        var o = _random_object()
        var oref = _rand(SM_OREF, -0.4, 0.4)
        _unit4(oref, 3)
        var f = _feat(s, o, oref)
        var fa = List[Float64](length=SM_OBJ_FEAT, fill=0.0)
        var fb = List[Float64](length=SM_OBJ_FEAT, fill=0.0)
        sm_obj_feat_lane[F64, R](_p(s), _p(o), _p(fa))
        sm_obj_feat_lane[F64, R](_p(s), _p(oref), _p(fb))
        for k in range(SM_OBJ_FEAT):
            worst_def = max(worst_def, max(abs(f[k] - fa[k]), abs(f[SM_OBJ_FEAT + k] - fb[k])))
        for k in range(3):
            worst_def = max(worst_def, abs(f[2 * SM_OBJ_FEAT + k] - (fb[k] - fa[k]) / 0.1))
        var g = _rand(SM_OE_FEAT, -1.0, 1.0)
        var g_s = List[Float64](length=D.S, fill=0.0)
        var g_o = List[Float64](length=SM_OBJ, fill=0.0)
        var tmp = List[Float64](length=SM_OE_SCRATCH, fill=0.0)
        var oref0 = oref.copy()
        sm_oe_feat_vjp_lane[F64, R](_p(s), _p(o), _p(oref), _p(g), _p(tmp), _p(g_s), _p(g_o))
        for k in range(SM_OREF):
            leak = max(leak, abs(oref[k] - oref0[k]))
        for k in range(7):
            var sp = s.copy()
            var sm = s.copy()
            sp[k] += EPS
            sm[k] -= EPS
            var fp = _feat(sp, o, oref)
            var fm = _feat(sm, o, oref)
            var fd = 0.0
            for i in range(SM_OE_FEAT):
                fd += g[i] * (fp[i] - fm[i]) / (2 * EPS)
            worst_s = max(worst_s, abs(fd - g_s[k]))
        for k in range(7, D.S):
            worst_s = max(worst_s, abs(g_s[k]))             # only the root pose is read
        for k in range(SM_OBJ):
            var op = o.copy()
            var om = o.copy()
            op[k] += EPS
            om[k] -= EPS
            var fp = _feat(s, op, oref)
            var fm = _feat(s, om, oref)
            var fd = 0.0
            for i in range(SM_OE_FEAT):
                fd += g[i] * (fp[i] - fm[i]) / (2 * EPS)
            worst_o = max(worst_o, abs(fd - g_o[k]))
    t.truth(worst_def <= 1e-14, "the 33 are [feat(s, o), feat(s, o_ref), Δp / 0.1]: max |d| " + String(worst_def))
    t.truth(worst_s <= 1e-7, "VJP into the robot's root pose (and nothing else of s): max |d| " + String(worst_s))
    t.truth(worst_o <= 1e-7 and leak == 0.0, "VJP into the object's state, the reference row untouched: max |d| " + String(worst_o))


def test_ref_rows(mut t: Tally) raises:
    print("\n[4] the reference rows on the G1 with its box")
    seed(4)
    var sim = R.SIM.create()
    var q0 = List[Float64](length=R.NDOF, fill=0.0)
    for j in range(R.NDOF):
        q0[j] = R.default_joint_pos(j)
    var tb = sm_tables[DT, R](sim, q0)
    comptime FRAMES = 6
    var rows = List[Scalar[DT]](length=FRAMES * D.S, fill=0)
    var track = List[Scalar[DT]](length=FRAMES * SM_OBJ, fill=0)
    for f in range(FRAMES):
        rows[f * D.S + 2] = Scalar[DT](0.76)
        var qn = 0.0
        var q = List[Float64]()
        for _ in range(4):
            q.append(random_float64(-1.0, 1.0))
        for k in range(4):
            qn += q[k] * q[k]
        for k in range(4):
            rows[f * D.S + 3 + k] = Scalar[DT](q[k] / sqrt(qn))
        for j in range(R.NDOF):
            rows[f * D.S + 7 + j] = Scalar[DT](q0[j] + random_float64(-0.3, 0.3))
        for k in range(3):
            track[f * SM_OBJ + k] = Scalar[DT](random_float64(-0.3, 0.3))
        track[f * SM_OBJ + 2] += Scalar[DT](0.9)
        var on = 0.0
        var oq = List[Float64]()
        for _ in range(4):
            oq.append(random_float64(-1.0, 1.0))
        for k in range(4):
            on += oq[k] * oq[k]
        for k in range(4):
            track[f * SM_OBJ + 3 + k] = Scalar[DT](oq[k] / sqrt(on))
    var refs = sm_object_ref_rows[R](tb, rows, track)
    t.truth(len(refs) == FRAMES * SM_OREF, String(FRAMES) + " frames x " + String(SM_OREF))
    var e_track = 0.0
    var e_hands = 0.0
    var far = 0.0
    var rwp = sm_obj_reward_params()
    var fk = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var terms = List[Scalar[DT]](length=SM_OBJ_TERMS, fill=0)
    var pos = mptr(fk)
    for f in range(FRAMES):
        for k in range(SM_OBJ):
            e_track = max(e_track, abs(Float64(refs[f * SM_OREF + k]) - Float64(track[f * SM_OBJ + k])))
        var s = mptr(rows).unsafe_offset(f * D.S)
        sm_fk_lane[DT, D.NB](tb.tree_p["cpu"](), s, s.unsafe_offset(D.NQ), pos, pos.unsafe_offset(D.NB * 3), pos.unsafe_offset(D.NB * 7), pos.unsafe_offset(D.NB * 10))
        _ = sm_obj_reward_lane[DT, R](
            tb.head_p["cpu"](), pos, mptr(track).unsafe_offset(f * SM_OBJ), mptr(refs).unsafe_offset(f * SM_OREF), mptr(rwp), mptr(terms),
        )
        e_hands = max(e_hands, abs(Float64(terms[OT_HANDS]) - 0.02 * SM_OW_HANDS))
        for k in range(6):
            far = max(far, abs(Float64(refs[f * SM_OREF + SM_OBJ + k])))
    t.truth(e_track == 0.0, "the first 13 of each row are the track's")
    t.truth(e_hands <= 1e-6 and far > 0.1, "robot at the library row, object on its track: the hands term is dt·w in every frame: max |d| " + String(e_hands))


def main() raises:
    var t = Tally()
    test_known(t)
    test_reward_vjp(t)
    test_features(t)
    test_ref_rows(t)
    print("\n=== " + String(t.checks - t.fails) + " / " + String(t.checks) + " passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_obj_task: " + String(t.fails) + " failed")
