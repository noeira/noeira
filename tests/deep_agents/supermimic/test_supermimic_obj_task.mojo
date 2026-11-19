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
[5] the lead-in (`sm_lead_in`): two clips of 5 and 3 frames and one more not
    named, 4 frames of lead-in: the named clips are 9 and 7 frames; each
    starts with 4 copies of its first pose at zero velocity and its first box
    pose; its own frames follow bit for bit; the clip not named keeps its
    rows. 0 frames changes nothing.
[6] a run's account of the carry (`SMObjCarryTally`): a clip of 14 frames
    whose reference carries on 10. Held all the way: at the bar and carried.
    Pushed before the lift, picked up 20 cm off and brought back by the
    fifth carry step: under the bar, carried, 4 cm once in hand. Dropped
    after three steps: neither. A fall after four carry steps: no second
    half, not carried. Steps whose reference is under the lift count nowhere.
    The lean (`sm_obj_lean_deg`, the fourth reading): a box held with its
    centre in place and hanging at 40 degrees is at the bar, carried and NOT
    level; the lean is 0 against itself and for any turn about the world's
    vertical, 30 for a pitch of 30 and between tilts of 20 and 50.
[7] the tracking kernels over a subset of the bodies
    (`heads.sm_tracking_kernels_lane`): with every body in the mask they are
    `sm_heads_lane`'s first six terms on a random state and reference (each kernel between 5
    and 98 % of its maximum); the
    G1's trunk-and-legs mask keeps 8 bodies and drops the 6 arm ones; with
    the ARMS alone moved off the reference the trunk-and-legs kernels are at
    their maxima while the all-body position and orientation kernels are
    not; with a KNEE moved they drop too.
[8] the force teacher (`sm_force_alloc`, `sm_force_teacher_lane`). The split
    between two contacts on known cases: a weight of 10 N held by two hands
    0.4 m apart at friction 0.5 is 5 N up and 10 N inward on each; a yaw
    moment of 1 N·m is a pair of 2.5 N; a moment about the line that joins
    the hands asks no force; the ceiling caps the squeeze. On 200 random
    cases the forces sum to the force asked, their moment misses the one
    asked only along the hands' line, and both are inside the friction cone.
    The lane on the G1: nothing at weight 0 or out of reach; with the point
    at the hand body's origin, no gains and no mass it is `sm_squeeze_lane`
    (the box centred between the hands); the joint offsets do the forces'
    virtual work on a random arm displacement, the hands' moments included
    (the box 6 cm ahead of the palms' line: its weight has a moment about
    that line, which the forces cannot give and the two moments do); with
    the point at the palm the wrist's last joint is asked for torque, at the
    body's origin it is not; with a mass alone the two forces sum to its
    weight; with the target seen from the robot's root, a reference robot
    standing elsewhere and turned 0.7 rad whose box is where this robot's
    is, seen from each, asks the weight alone, and without it asks more; the
    pull on the hand bodies alone (the wrench's weight 0): a reference that
    has each hand 4 cm off where it is, in the turned box's frame, pulls it
    with 300 N/m toward that place, capped, and the offsets do its work.

Run: pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_obj_task.mojo
"""

from std.math import cos, exp, sin, sqrt
from std.random import random_float64, seed

from noeira.deep_agents.supermimic.heads import (
    SM_N_TERMS, SMHeadLayout, T_BODY_ANG, T_BODY_LIN, T_BODY_ORI, T_BODY_POS, T_ROOT_ORI, T_ROOT_POS, sm_heads_lane, sm_tracking_kernels_lane,
)
from noeira.deep_agents.supermimic.kinematics import FP, Q4, cross3, dot3, sm_fk_lane
from noeira.deep_agents.supermimic.motion_library import SM_REF_SCRATCH, sm_reference_row_lane
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_FEAT, sm_obj_feat_lane
from noeira.deep_agents.supermimic.object_task import (
    OT_HANDS, OT_ORI, OT_POS, SM_OBJ_TERMS, SM_OE_FEAT, SM_OE_SCRATCH, SM_OREF, SM_ORW, SM_OSTD_HANDS, SM_OSTD_ORI,
    SM_OSTD_POS, SM_OW_HANDS, SM_OW_ORI, SM_OW_POS, sm_object_ref_rows, sm_obj_reward_lane, sm_obj_reward_params,
    sm_lead_in, sm_obj_carry_steps, sm_obj_carry_tally, sm_obj_reward_vjp_lane, sm_oe_feat_lane, sm_oe_feat_vjp_lane, sm_trunk_legs_mask,
    FT_FMAX, FT_HMAX, FT_KD, FT_KH, FT_KP, FT_KR, FT_KW, FT_MASS, FT_MMAX, FT_MU, FT_POINT, FT_REL, FT_SQUEEZE, FT_SQUEEZE_MAX, SM_FT_OUT, SM_FT_PARAMS,
    sm_force_alloc, sm_force_teacher_lane,
    sm_squeeze_lane, sm_obj_lean_deg,
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


def test_lead_in(mut t: Tally) raises:
    print("\n[5] the lead-in")
    seed(5)
    comptime LEAD = 4
    var lens: List[Int] = [5, 3]
    var extra = 2                      # one more clip in `rows`, not named in `lens`
    var rows = List[Scalar[DT]]()
    for _ in range((5 + 3 + extra) * D.S):
        rows.append(Scalar[DT](random_float64(-1.0, 1.0)))
    var poses = List[Float64]()
    for _ in range((5 + 3) * 7):
        poses.append(random_float64(-1.0, 1.0))
    var rows0 = rows.copy()
    var poses0 = poses.copy()
    var same_r = rows.copy()
    var same_l = lens.copy()
    var same_p = poses.copy()
    sm_lead_in[R](same_r, same_l, same_p, 0)
    var d0 = 0.0
    for i in range(len(rows0)):
        d0 = max(d0, abs(Float64(same_r[i]) - Float64(rows0[i])))
    t.truth(d0 == 0.0 and len(same_r) == len(rows0) and same_l[0] == 5 and len(same_p) == len(poses0), "0 frames: nothing changes")
    sm_lead_in[R](rows, lens, poses, LEAD)
    t.truth(
        lens[0] == 5 + LEAD and lens[1] == 3 + LEAD and len(rows) == (8 + extra + 2 * LEAD) * D.S and len(poses) == (8 + 2 * LEAD) * 7,
        "two clips of 5 and 3 frames become " + String(lens[0]) + " and " + String(lens[1]) + "; rows and box poses grow by 2 x " + String(LEAD) + " frames",
    )
    var lead_q = 0.0        # the lead-in's qpos against the clip's first frame
    var lead_v = 0.0        # its velocities
    var lead_p = 0.0        # its box poses
    var body = 0.0          # the clip's own frames, shifted
    var o_old = 0
    var o_new = 0
    var olds: List[Int] = [5, 3]
    for c in range(2):
        for f in range(LEAD):
            for k in range(D.S):
                var v = Float64(rows[(o_new + f) * D.S + k])
                if k < D.NQ:
                    lead_q = max(lead_q, abs(v - Float64(rows0[o_old * D.S + k])))
                else:
                    lead_v = max(lead_v, abs(v))
            for k in range(7):
                lead_p = max(lead_p, abs(poses[(o_new + f) * 7 + k] - poses0[o_old * 7 + k]))
        for f in range(olds[c]):
            for k in range(D.S):
                body = max(body, abs(Float64(rows[(o_new + LEAD + f) * D.S + k]) - Float64(rows0[(o_old + f) * D.S + k])))
            for k in range(7):
                body = max(body, abs(poses[(o_new + LEAD + f) * 7 + k] - poses0[(o_old + f) * 7 + k]))
        o_old += olds[c]
        o_new += olds[c] + LEAD
    var tail = 0.0
    for i in range(extra * D.S):
        tail = max(tail, abs(Float64(rows[o_new * D.S + i]) - Float64(rows0[o_old * D.S + i])))
    t.truth(lead_q == 0.0 and lead_v == 0.0 and lead_p == 0.0, "each lead-in frame is the clip's first pose at zero velocity, with the box's first pose")
    t.truth(body == 0.0 and tail == 0.0, "the clips' own frames follow bit for bit, and the clip not named keeps its rows")


def test_carry_tally(mut t: Tally) raises:
    print("\n[6] a run's account of the carry")
    comptime FRAMES = 14
    # the reference box: on the floor for 2 frames, up for 10 (one of them exactly at the lift), down for 2
    var rise: List[Float64] = [0.0, 0.02, 0.05, 0.2, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.3, 0.06, 0.04, 0.0]
    var orefs = List[Scalar[DT]](length=(FRAMES + 3) * SM_OREF, fill=0)
    for f in range(FRAMES):
        orefs[(3 + f) * SM_OREF + 2] = Scalar[DT](0.25 + rise[f])       # the clip starts at library frame 3
    var total = sm_obj_carry_steps(orefs, 3, FRAMES)
    t.truth(total == 10, "the clip's carry frames: " + String(total) + " of " + String(FRAMES))

    var a = sm_obj_carry_tally(total)
    for f in range(FRAMES):
        a.note(rise[f], rise[f], 0.02, 4.0)
    t.truth(
        a.carry == 10 and a.held == 10 and a.at_bar() and a.carried() and a.late == 5 and a.late_up == 5 and abs(a.mean_err_late() - 0.02) < 1e-12
        and a.level() and abs(a.mean_lean_late() - 4.0) < 1e-12,
        "held all the way, leaning 4 deg: at the bar, carried, 2 cm once in hand, level",
    )
    var h = sm_obj_carry_tally(total)
    for f in range(FRAMES):
        h.note(rise[f], rise[f], 0.02, 40.0)
    t.truth(
        h.at_bar() and h.carried() and (not h.level()) and abs(h.mean_lean_late() - 40.0) < 1e-12,
        "held all the way with its centre in place, hanging at 40 deg: at the bar, carried, NOT level",
    )
    var b = sm_obj_carry_tally(total)
    var n = 0
    for f in range(FRAMES):
        if rise[f] >= 0.05:
            n += 1
        b.note(rise[f], rise[f], 0.2 if n <= 4 else 0.04, 30.0 if n <= 4 else 10.0)
    t.truth(
        b.carry == 10 and b.held == 6 and (not b.at_bar()) and b.carried() and abs(b.mean_err_late() - 0.04) < 1e-12 and abs(b.emax - 0.2) < 1e-12
        and abs(b.mean_err() - 0.104) < 1e-12 and b.level() and abs(b.mean_lean_late() - 10.0) < 1e-12,
        "pushed, picked up 20 cm off and tipped 30 deg, brought back: under the bar, carried, 4 cm once in hand, level at 10 deg (" + String(b.held) + " of " + String(b.carry) + " held)",
    )
    var c = sm_obj_carry_tally(total)
    n = 0
    for f in range(FRAMES):
        if rise[f] >= 0.05:
            n += 1
        c.note(rise[f], rise[f] if n <= 3 else 0.0, 0.02 if n <= 3 else 0.5, 0.0)
    t.truth(
        c.carry == 10 and c.held == 3 and (not c.at_bar()) and (not c.carried()) and c.late == 5 and c.late_up == 0 and (not c.level()),
        "dropped after three steps, lying flat: neither, and not level (level is of a carried box)",
    )
    var d = sm_obj_carry_tally(total)
    for f in range(6):                  # the run ends (a fall) after four carry steps
        d.note(rise[f], rise[f], 0.02, 0.0)
    t.truth(d.carry == 4 and d.held == 4 and d.late == 0 and (not d.carried()), "a fall after four carry steps: no second half, not carried")
    var e = sm_obj_carry_tally(total)
    e.note(0.049, 0.3, 0.9, 80.0)
    e.note(0.0, 0.0, 0.9, 80.0)
    t.truth(e.carry == 0 and e.emax == 0.0 and (not e.at_bar()) and (not e.carried()), "steps whose reference is under the lift count nowhere")

    # the lean: the angle between the world's up seen from the box and seen from the reference box
    comptime RAD = 0.017453292519943295
    var same = sm_obj_lean_deg(0.9, 0.1, -0.3, 0.2, 0.9, 0.1, -0.3, 0.2)
    var pitch = sm_obj_lean_deg(cos(15.0 * RAD), 0.0, sin(15.0 * RAD), 0.0, 1.0, 0.0, 0.0, 0.0)
    var yaw = sm_obj_lean_deg(cos(35.0 * RAD), 0.0, 0.0, sin(35.0 * RAD), 1.0, 0.0, 0.0, 0.0)
    # a reference tilted 20 deg about x; the box the same, then turned 50 deg about the WORLD's vertical; and the box tilted 50 deg about x
    var cr = cos(10.0 * RAD)
    var sr = sin(10.0 * RAD)
    var cz = cos(25.0 * RAD)
    var sz = sin(25.0 * RAD)
    var turned = sm_obj_lean_deg(cz * cr, cz * sr, sz * sr, sz * cr, cr, sr, 0.0, 0.0)
    var more = sm_obj_lean_deg(2.0 * cos(25.0 * RAD), 2.0 * sin(25.0 * RAD), 0.0, 0.0, cr, sr, 0.0, 0.0)
    # the same tilt about the box's OWN other axis is another way of hanging: acos(cos² 20°)
    var other = sm_obj_lean_deg(cr, 0.0, sr, 0.0, cr, sr, 0.0, 0.0)
    t.truth(
        same <= 1e-6 and abs(pitch - 30.0) <= 1e-9 and yaw <= 1e-6 and turned <= 1e-6 and abs(more - 30.0) <= 1e-9 and abs(other - 27.99089071778283) <= 1e-9,
        "the lean: 0 against itself, 30 deg for a pitch of 30, 0 for a turn about the vertical (" + String(yaw) + ") and for a tilted box turned about it ("
        + String(turned) + "), 30 between tilts of 20 and 50 (a quaternion of norm 2), " + String(other) + " between the same tilt about two axes of the box",
    )


def test_tracking_kernels(mut t: Tally) raises:
    print("\n[7] the tracking kernels over a subset of the bodies")
    seed(7)
    var sim = R.SIM.create()
    var q0 = List[Float64](length=R.NDOF, fill=0.0)
    for j in range(R.NDOF):
        q0[j] = R.default_joint_pos(j)
    var tb = sm_tables[DT, R](sim, q0)
    var tree = tb.tree_p["cpu"]()
    var table = tb.head_p["cpu"]()
    var mask_tl = sm_trunk_legs_mask[R]()
    var kept = 0
    var arms_last = True
    for k in range(R.N_TRACKED):
        var on = Float64(mask_tl[k]) != 0.0
        if on:
            kept += 1
        if (k < 8) != on:
            arms_last = False
    t.truth(len(mask_tl) == R.N_TRACKED and kept == 8 and arms_last, "the G1's trunk-and-legs mask keeps " + String(kept) + " of " + String(R.N_TRACKED) + " bodies, the six arm ones out")
    var mask_all = List[Scalar[DT]](length=R.N_TRACKED, fill=1)

    # a reference row (library layout) and a state
    var row = List[Scalar[DT]](length=D.S, fill=0)
    var s = List[Scalar[DT]](length=D.S, fill=0)
    row[2] = Scalar[DT](0.76)
    s[2] = Scalar[DT](0.71)
    s[0] = Scalar[DT](0.05)
    var qa = List[Float64]()
    var qb = List[Float64]()
    var na = 0.0
    var nb = 0.0
    for _ in range(4):
        qa.append(random_float64(-0.06, 0.06))
        qb.append(random_float64(-0.06, 0.06))
    qa[0] += 1.0
    qb[0] += 1.0
    for k in range(4):
        na += qa[k] * qa[k]
        nb += qb[k] * qb[k]
    for k in range(4):
        row[3 + k] = Scalar[DT](qa[k] / sqrt(na))
        s[3 + k] = Scalar[DT](qb[k] / sqrt(nb))
    for j in range(R.NDOF):
        row[7 + j] = Scalar[DT](q0[j] + random_float64(-0.15, 0.15))
        s[7 + j] = Scalar[DT](q0[j] + random_float64(-0.15, 0.15))
    for k in range(R.NV):
        row[D.NQ + k] = Scalar[DT](random_float64(-0.5, 0.5))
        s[D.NQ + k] = Scalar[DT](random_float64(-0.5, 0.5))
    var rf = List[Scalar[DT]](length=D.REF, fill=0)
    var rsc = List[Scalar[DT]](length=SM_REF_SCRATCH[R], fill=0)
    sm_reference_row_lane[R](mptr(row), 0, tree, table, mptr(rsc), mptr(rf))
    var act = List[Scalar[DT]](length=R.NDOF, fill=0)
    var xi = List[Scalar[DT]](length=D.X, fill=0)
    var terms = List[Scalar[DT]](length=SM_N_TERMS, fill=0)
    var feat = List[Scalar[DT]](length=2, fill=0)
    var obs = List[Scalar[DT]](length=D.OBS, fill=0)
    var sc = List[Scalar[DT]](length=L.SCRATCH, fill=0)
    _ = sm_heads_lane[DT, R](tree, table, mptr(s), mptr(act), mptr(act), mptr(xi), mptr(rf), mptr(terms), mptr(feat), mptr(obs), mptr(sc))
    var k6 = List[Scalar[DT]](length=6, fill=0)
    var sum6 = Float64(sm_tracking_kernels_lane[DT, R](table, mptr(sc), mptr(rf), mptr(mask_all), mptr(k6)))   # `sc` starts with the FK block
    var e_all = 0.0
    var want = 0.0
    var below = True                    # every kernel is mid-range: the comparison is of six values that move with their errors
    var wmax: List[Float64] = [0.01, 0.01, 0.02, 0.02, 0.02, 0.02]
    for k in range(6):
        e_all = max(e_all, abs(Float64(k6[k]) - Float64(terms[k])))
        want += Float64(terms[k])
        if Float64(terms[k]) > 0.98 * wmax[k] or Float64(terms[k]) < 0.05 * wmax[k]:
            below = False
    t.truth(e_all <= 1e-7 and abs(sum6 - want) <= 1e-7 and below, "every body in the mask: `sm_heads_lane`'s first six terms, max |d| " + String(e_all) + ", each between 5 and 98 % of its maximum")
    var k6t = List[Scalar[DT]](length=6, fill=0)
    _ = sm_tracking_kernels_lane[DT, R](table, mptr(sc), mptr(rf), mptr(mask_tl), mptr(k6t))
    t.truth(
        Float64(k6t[T_ROOT_POS]) == Float64(k6[T_ROOT_POS]) and Float64(k6t[T_ROOT_ORI]) == Float64(k6[T_ROOT_ORI])
        and abs(Float64(k6t[T_BODY_POS]) - Float64(k6[T_BODY_POS])) > 1e-5 and abs(Float64(k6t[T_BODY_ORI]) - Float64(k6[T_BODY_ORI])) > 1e-5,
        "the mask moves the body kernels and leaves the two root terms: root " + String(Float64(k6t[T_ROOT_POS])) + " " + String(Float64(k6[T_ROOT_POS]))
        + " | " + String(Float64(k6t[T_ROOT_ORI])) + " " + String(Float64(k6[T_ROOT_ORI])) + " | position " + String(Float64(k6t[T_BODY_POS])) + " "
        + String(Float64(k6[T_BODY_POS])) + " | orientation " + String(Float64(k6t[T_BODY_ORI])) + " " + String(Float64(k6[T_BODY_ORI])),
    )

    # the state AT the reference (zero velocities), then the arms alone off it, then a knee
    var fk = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var pos = mptr(fk)
    for k in range(R.NV):
        row[D.NQ + k] = Scalar[DT](0)
    sm_reference_row_lane[R](mptr(row), 0, tree, table, mptr(rsc), mptr(rf))
    var s2 = row.copy()
    for j in range(15, R.NDOF):         # joints 15.. are the two arms
        s2[7 + j] += Scalar[DT](0.4)
    var sp = mptr(s2)
    sm_fk_lane[DT, D.NB](tree, sp, sp.unsafe_offset(D.NQ), pos, pos.unsafe_offset(D.NB * 3), pos.unsafe_offset(D.NB * 7), pos.unsafe_offset(D.NB * 10))
    _ = sm_tracking_kernels_lane[DT, R](table, pos, mptr(rf), mptr(mask_all), mptr(k6))
    _ = sm_tracking_kernels_lane[DT, R](table, pos, mptr(rf), mptr(mask_tl), mptr(k6t))
    var e_max = 0.0
    for k in range(6):
        e_max = max(e_max, abs(Float64(k6t[k]) - wmax[k]))
    t.truth(
        e_max <= 1e-6 and Float64(k6[T_BODY_POS]) < 0.02 - 1e-4 and Float64(k6[T_BODY_ORI]) < 0.02 - 1e-3 and abs(Float64(k6[T_BODY_LIN]) - 0.02) <= 1e-6
        and abs(Float64(k6[T_BODY_ANG]) - 0.02) <= 1e-6,
        "the arms alone 0.4 rad off: trunk and legs at their maxima (max |d| " + String(e_max) + "), all bodies position " + String(Float64(k6[T_BODY_POS]))
        + " orientation " + String(Float64(k6[T_BODY_ORI])) + " of 0.02",
    )
    var s3 = row.copy()
    s3[7 + 3] += Scalar[DT](0.4)        # the left knee
    var sp3 = mptr(s3)
    sm_fk_lane[DT, D.NB](tree, sp3, sp3.unsafe_offset(D.NQ), pos, pos.unsafe_offset(D.NB * 3), pos.unsafe_offset(D.NB * 7), pos.unsafe_offset(D.NB * 10))
    _ = sm_tracking_kernels_lane[DT, R](table, pos, mptr(rf), mptr(mask_tl), mptr(k6t))
    t.truth(Float64(k6t[T_BODY_POS]) < 0.02 - 1e-5 and Float64(k6t[T_BODY_ORI]) < 0.02 - 1e-4, "a knee 0.4 rad off: the trunk-and-legs kernels drop, position " + String(Float64(k6t[T_BODY_POS])) + " orientation " + String(Float64(k6t[T_BODY_ORI])))


def _v(x: Float64, y: Float64, z: Float64) -> Q4[F64]:
    return Q4[F64](x, y, z, 0)


def _n3(v: Q4[F64]) -> Float64:
    return sqrt(dot3(v, v))


def _palm(fk: List[Scalar[DT]], hb: Int, lx: Float64, ly: Float64, lz: Float64) -> Q4[F64]:
    """The point (lx, ly, lz) of body `hb`, in the world, from an FK block."""
    var q = List[Float64]()
    for k in range(4):
        q.append(Float64(fk[D.NB * 3 + hb * 4 + k]))
    var r = _rotate(q, 0, [lx, ly, lz])
    return _v(Float64(fk[hb * 3]) + r[0], Float64(fk[hb * 3 + 1]) + r[1], Float64(fk[hb * 3 + 2]) + r[2])


def _turn(fa: List[Scalar[DT]], fb: List[Scalar[DT]], hb: Int) -> Q4[F64]:
    """Body `hb`'s rotation from the FK block `fb` to `fa`, as 2 · vec(q_a ⊗ conj(q_b)), world."""
    var a = List[Float64]()
    var b = List[Float64]()
    for k in range(4):
        a.append(Float64(fa[D.NB * 3 + hb * 4 + k]))
        b.append(Float64(fb[D.NB * 3 + hb * 4 + k]))
    var w = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    var sg = 2.0 if w >= 0.0 else -2.0
    return _v(
        sg * (-a[0] * b[1] + a[1] * b[0] - a[2] * b[3] + a[3] * b[2]),
        sg * (-a[0] * b[2] + a[1] * b[3] + a[2] * b[0] - a[3] * b[1]),
        sg * (-a[0] * b[3] - a[1] * b[2] + a[2] * b[1] + a[3] * b[0]),
    )


def test_force_teacher(mut t: Tally) raises:
    print("\n[8] the force teacher")
    seed(8)
    # the split, known cases: hands at y = +0.2 and −0.2, the body at the origin
    var c0 = _v(0.0, 0.2, 0.0)
    var c1 = _v(0.0, -0.2, 0.0)
    var z = _v(0.0, 0.0, 0.0)
    var a = sm_force_alloc[F64](c0, c1, z, _v(0.0, 0.0, 10.0), z, 0.0, 100.0, 0.5)
    t.truth(
        _n3(a[0] - _v(0.0, -10.0, 5.0)) <= 1e-12 and _n3(a[1] - _v(0.0, 10.0, 5.0)) <= 1e-12 and abs(a[2] - 10.0) <= 1e-12,
        "a weight of 10 N at friction 0.5: each hand 5 N up and 10 N inward (squeeze " + String(a[2]) + ")",
    )
    a = sm_force_alloc[F64](c0, c1, z, _v(0.0, 0.0, 10.0), z, 0.0, 4.0, 0.5)
    t.truth(abs(a[2] - 4.0) <= 1e-12 and abs(a[0][1] + 4.0) <= 1e-12, "the ceiling caps the squeeze at 4 N")
    a = sm_force_alloc[F64](c0, c1, z, z, _v(0.0, 0.0, 1.0), 0.0, 100.0, 0.5)
    var mz = cross3(c0, a[0]) + cross3(c1, a[1])
    t.truth(
        abs(a[0][0] + 2.5) <= 1e-12 and abs(a[1][0] - 2.5) <= 1e-12 and _n3(mz - _v(0.0, 0.0, 1.0)) <= 1e-12 and abs(a[2] - 5.0) <= 1e-12,
        "a yaw moment of 1 N·m: a pair of 2.5 N, 0.4 m apart, squeezed by 5 N",
    )
    a = sm_force_alloc[F64](c0, c1, z, z, _v(0.0, 1.0, 0.0), 3.0, 100.0, 0.5)
    t.truth(
        _n3(a[0] - _v(0.0, -3.0, 0.0)) <= 1e-12 and _n3(a[1] - _v(0.0, 3.0, 0.0)) <= 1e-12 and abs(a[3] + 1.0) <= 1e-12,
        "a moment about the hands' line asks no force: the squeeze's floor alone, and 1 N·m left over",
    )
    var e_sum = 0.0
    var e_mom = 0.0
    var e_cone = 0.0
    var along = 0.0
    for _ in range(200):
        var k0 = _v(random_float64(-0.3, 0.3), random_float64(0.1, 0.4), random_float64(-0.3, 0.3))
        var k1 = _v(random_float64(-0.3, 0.3), random_float64(-0.4, -0.1), random_float64(-0.3, 0.3))
        var pc = _v(random_float64(-0.2, 0.2), random_float64(-0.1, 0.1), random_float64(-0.2, 0.2))
        var fw = _v(random_float64(-20.0, 20.0), random_float64(-20.0, 20.0), random_float64(-20.0, 20.0))
        var mw = _v(random_float64(-3.0, 3.0), random_float64(-3.0, 3.0), random_float64(-3.0, 3.0))
        var mu = random_float64(0.3, 0.9)
        var r = sm_force_alloc[F64](k0, k1, pc, fw, mw, 2.0, 1.0e6, mu)
        var u = (k1 - k0) / _n3(k1 - k0)
        e_sum = max(e_sum, _n3(r[0] + r[1] - fw))
        var miss = cross3(k0 - pc, r[0]) + cross3(k1 - pc, r[1]) - mw
        e_mom = max(e_mom, _n3(cross3(miss, u)))
        along = max(along, abs(dot3(miss, u)))
        e_mom = max(e_mom, abs(dot3(miss, u) + r[3]))
        var n0 = dot3(r[0], u)
        var n1 = -dot3(r[1], u)
        e_cone = max(e_cone, _n3(r[0] - u * n0) - mu * n0)
        e_cone = max(e_cone, _n3(r[1] + u * n1) - mu * n1)
        e_cone = max(e_cone, 2.0 - r[2])
    t.truth(
        e_sum <= 1e-9 and e_mom <= 1e-9 and e_cone <= 1e-9 and along > 0.1,
        "200 random cases: the forces sum to the one asked (" + String(e_sum) + "), their moment misses the one asked only along the hands' line, by the moment left over (across "
        + String(e_mom) + ", along up to " + String(along) + " N·m), both inside the cone and the squeeze at its floor or above (" + String(e_cone) + ")",
    )

    # the lane on the G1
    var sim = R.SIM.create()
    var q0 = List[Float64](length=R.NDOF, fill=0.0)
    for j in range(R.NDOF):
        q0[j] = R.default_joint_pos(j)
    var tb = sm_tables[DT, R](sim, q0)
    var tree = tb.tree_p["cpu"]()
    var table = tb.head_p["cpu"]()
    var hl = Int(table[unsafe_offset=L.HANDS])
    var hr = Int(table[unsafe_offset=L.HANDS + 1])
    var s = List[Scalar[DT]](length=D.S, fill=0)
    s[2] = Scalar[DT](0.76)
    s[3] = Scalar[DT](1)
    for j in range(R.NDOF):
        s[7 + j] = Scalar[DT](q0[j] + random_float64(-0.2, 0.2))
    var fk = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var pos = mptr(fk)
    var sp = mptr(s)
    sm_fk_lane[DT, D.NB](tree, sp, sp.unsafe_offset(D.NQ), pos, pos.unsafe_offset(D.NB * 3), pos.unsafe_offset(D.NB * 7), pos.unsafe_offset(D.NB * 10))
    comptime ARM0 = 15
    comptime KS = 100.0
    var ks = List[Scalar[DT]](length=R.NDOF, fill=Scalar[DT](KS))
    var scratch = List[Scalar[DT]](length=D.S + 2 * D.FK_BLOCK + SM_FT_OUT, fill=0)
    comptime FO = D.S + 2 * D.FK_BLOCK
    var tp = List[Scalar[DT]](length=SM_FT_PARAMS, fill=0)
    tp[FT_SQUEEZE] = Scalar[DT](20)
    tp[FT_SQUEEZE_MAX] = Scalar[DT](100)
    tp[FT_MU] = Scalar[DT](0.5)
    tp[FT_FMAX] = Scalar[DT](1000)
    # the box centred between the two hand BODIES, at its reference
    var o = List[Scalar[DT]](length=SM_OBJ, fill=0)
    for k in range(3):
        o[k] = (fk[hl * 3 + k] + fk[hr * 3 + k]) / 2
    o[3] = Scalar[DT](1)
    var oref = List[Scalar[DT]](length=SM_OREF, fill=0)
    for k in range(SM_OBJ):
        oref[k] = o[k]
    var act = List[Scalar[DT]](length=R.NDOF, fill=0)
    sm_force_teacher_lane[R](ARM0, Scalar[DT](0), Scalar[DT](0), mptr(tp), tree, table, mptr(ks), sp, sp, mptr(o), mptr(oref), mptr(scratch), mptr(act))
    var moved = 0.0
    for j in range(R.NDOF):
        moved = max(moved, abs(Float64(act[j])))
    var far = o.copy()
    far[0] += Scalar[DT](1.0)
    sm_force_teacher_lane[R](ARM0, Scalar[DT](1), Scalar[DT](0), mptr(tp), tree, table, mptr(ks), sp, sp, mptr(far), mptr(oref), mptr(scratch), mptr(act))
    for j in range(R.NDOF):
        moved = max(moved, abs(Float64(act[j])))
    for k in range(SM_FT_OUT):
        moved = max(moved, abs(Float64(scratch[FO + k])))
    t.truth(moved == 0.0, "weight 0, and a box a metre away: no offset and no force")

    sm_force_teacher_lane[R](ARM0, Scalar[DT](1), Scalar[DT](0), mptr(tp), tree, table, mptr(ks), sp, sp, mptr(o), mptr(oref), mptr(scratch), mptr(act))
    var sq = List[Scalar[DT]](length=R.NDOF, fill=0)
    var scratch2 = List[Scalar[DT]](length=D.S + 2 * D.FK_BLOCK + 6, fill=0)
    sm_squeeze_lane[R](ARM0, Scalar[DT](20), tree, table, mptr(ks), sp, mptr(o), mptr(scratch2), mptr(sq))
    var d_sq = 0.0
    var big = 0.0
    var legs = 0.0
    for j in range(R.NDOF):
        d_sq = max(d_sq, abs(Float64(act[j]) - Float64(sq[j])))
        big = max(big, abs(Float64(act[j])))
        if j < ARM0:
            legs = max(legs, abs(Float64(act[j])))
    var yaw_origin = abs(Float64(act[21])) + abs(Float64(act[28]))
    t.truth(
        d_sq <= 1e-4 and big > 0.01 and legs == 0.0,
        "the point at the hand body's origin, no gains, no mass: `sm_squeeze_lane` at 20 N (max |d| " + String(d_sq) + " of " + String(big)
        + " action units), the legs and the waist untouched",
    )

    # the point at the palm: virtual work on a random arm displacement, and the wrist's last joint
    tp[FT_POINT] = Scalar[DT](0.11)
    tp[FT_POINT + 1] = Scalar[DT](-0.01)
    tp[FT_KP] = Scalar[DT](150)
    tp[FT_KR] = Scalar[DT](4)
    tp[FT_MASS] = Scalar[DT](1)
    tp[FT_MMAX] = Scalar[DT](5)
    var pl = _palm(fk, hl, 0.11, -0.01, 0.0)
    var pr = _palm(fk, hr, 0.11, 0.01, 0.0)
    for k in range(3):
        o[k] = Scalar[DT]((pl[k] + pr[k]) / 2.0)
    o[0] += Scalar[DT](0.06)            # ahead of the palms' line
    for k in range(3):
        oref[k] = o[k] + Scalar[DT](random_float64(-0.05, 0.05))
    var act2 = List[Scalar[DT]](length=R.NDOF, fill=0)
    sm_force_teacher_lane[R](ARM0, Scalar[DT](1), Scalar[DT](0), mptr(tp), tree, table, mptr(ks), sp, sp, mptr(o), mptr(oref), mptr(scratch), mptr(act2))
    var f0 = _v(Float64(scratch[FO]), Float64(scratch[FO + 1]), Float64(scratch[FO + 2]))
    var f1 = _v(Float64(scratch[FO + 3]), Float64(scratch[FO + 4]), Float64(scratch[FO + 5]))
    var mh = Float64(scratch[FO + 6])
    var ul = (pr - pl) / _n3(pr - pl)
    var pc = _v(Float64(o[0]), Float64(o[1]), Float64(o[2]))
    var want = (
        _v(Float64(oref[0]) - Float64(o[0]), Float64(oref[1]) - Float64(o[1]), Float64(oref[2]) - Float64(o[2])) * 150.0 + _v(0.0, 0.0, 9.81)
    )
    var total_m = cross3(pl - pc, f0) + cross3(pr - pc, f1) + ul * (2.0 * mh)
    t.truth(
        _n3(f0 + f1 - want) <= 1e-3 and _n3(total_m) <= 1e-3 and abs(mh) > 0.1 and abs(Float64(scratch[FO + 7]) - mh) == 0.0,
        "the box 6 cm ahead of the palms' line: the forces sum to the force asked, and with a moment of " + String(mh)
        + " N·m from each hand the whole moment is the one asked (off by " + String(_n3(total_m)) + " N·m)",
    )
    var dq = List[Float64](length=R.NDOF, fill=0.0)
    var work_q = 0.0
    for j in range(ARM0, R.NDOF):
        dq[j] = random_float64(-0.004, 0.004)
        work_q += Float64(act2[j]) * KS * dq[j]
    var sa = s.copy()
    var sb = s.copy()
    for j in range(ARM0, R.NDOF):
        sa[7 + j] += Scalar[DT](dq[j])
        sb[7 + j] -= Scalar[DT](dq[j])
    var fa = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var fb = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var pa = mptr(fa)
    var pb = mptr(fb)
    var spa = mptr(sa)
    var spb = mptr(sb)
    sm_fk_lane[DT, D.NB](tree, spa, spa.unsafe_offset(D.NQ), pa, pa.unsafe_offset(D.NB * 3), pa.unsafe_offset(D.NB * 7), pa.unsafe_offset(D.NB * 10))
    sm_fk_lane[DT, D.NB](tree, spb, spb.unsafe_offset(D.NQ), pb, pb.unsafe_offset(D.NB * 3), pb.unsafe_offset(D.NB * 7), pb.unsafe_offset(D.NB * 10))
    var work_x = (
        dot3(_palm(fa, hl, 0.11, -0.01, 0.0) - _palm(fb, hl, 0.11, -0.01, 0.0), f0)
        + dot3(_palm(fa, hr, 0.11, 0.01, 0.0) - _palm(fb, hr, 0.11, 0.01, 0.0), f1)
        + dot3(_turn(fa, fb, hl) + _turn(fa, fb, hr), ul * mh)
    ) / 2.0
    var yaw_palm = abs(Float64(act2[21])) + abs(Float64(act2[28]))
    t.truth(
        abs(work_q - work_x) <= 0.05 * abs(work_x) and abs(work_x) > 1e-3,
        "the joint offsets do the forces' and the moments' virtual work on a random arm displacement: " + String(work_q) + " against " + String(work_x) + " J",
    )
    t.truth(
        yaw_origin * KS <= 1e-3 and yaw_palm * KS > 0.3,
        "the wrists' last joints: " + String(yaw_origin * KS) + " N·m asked with the point at the body's origin, " + String(yaw_palm * KS) + " at the palm",
    )
    tp[FT_KP] = Scalar[DT](0)
    tp[FT_KR] = Scalar[DT](0)
    tp[FT_SQUEEZE] = Scalar[DT](0)
    var act3 = List[Scalar[DT]](length=R.NDOF, fill=0)
    sm_force_teacher_lane[R](ARM0, Scalar[DT](1), Scalar[DT](0), mptr(tp), tree, table, mptr(ks), sp, sp, mptr(o), mptr(oref), mptr(scratch), mptr(act3))
    var fsum = _v(
        Float64(scratch[FO]) + Float64(scratch[FO + 3]), Float64(scratch[FO + 1]) + Float64(scratch[FO + 4]), Float64(scratch[FO + 2]) + Float64(scratch[FO + 5])
    )
    t.truth(_n3(fsum - _v(0.0, 0.0, 9.81)) <= 1e-4, "a mass of 1 kg alone: the two forces sum to its weight (" + String(fsum[2]) + " N up)")

    # the target seen from the robot's root: a reference robot elsewhere, turned 0.7 rad, its box where this robot's is seen from each
    tp[FT_KP] = Scalar[DT](150)
    tp[FT_KR] = Scalar[DT](4)
    tp[FT_MMAX] = Scalar[DT](0)
    comptime PSI = 0.7
    var rs = s.copy()
    rs[0] = Scalar[DT](0.3)
    rs[1] = Scalar[DT](-0.2)
    rs[2] = Scalar[DT](0.81)
    rs[3] = Scalar[DT](cos(PSI / 2.0))
    rs[6] = Scalar[DT](sin(PSI / 2.0))
    var rel = [Float64(o[0]) - Float64(s[0]), Float64(o[1]) - Float64(s[1]), Float64(o[2]) - Float64(s[2])]
    oref[0] = rs[0] + Scalar[DT](cos(PSI) * rel[0] - sin(PSI) * rel[1])
    oref[1] = rs[1] + Scalar[DT](sin(PSI) * rel[0] + cos(PSI) * rel[1])
    oref[2] = rs[2] + Scalar[DT](rel[2])
    oref[3] = rs[3]
    oref[6] = rs[6]
    var sums = List[Float64]()
    for mode in range(2):
        tp[FT_REL] = Scalar[DT](1 - mode)
        var act4 = List[Scalar[DT]](length=R.NDOF, fill=0)
        sm_force_teacher_lane[R](ARM0, Scalar[DT](1), Scalar[DT](0), mptr(tp), tree, table, mptr(ks), sp, mptr(rs), mptr(o), mptr(oref), mptr(scratch), mptr(act4))
        sums.append(_n3(_v(
            Float64(scratch[FO]) + Float64(scratch[FO + 3]), Float64(scratch[FO + 1]) + Float64(scratch[FO + 4]),
            Float64(scratch[FO + 2]) + Float64(scratch[FO + 5]) - 9.81,
        )))
    t.truth(
        sums[0] <= 1e-3 and sums[1] > 5.0,
        "the target seen from the root: a reference robot elsewhere and turned 0.7 rad asks the weight alone (off by " + String(sums[0])
        + " N); the same target in the world asks " + String(sums[1]) + " N more",
    )

    # the pull on the hand bodies alone: the box turned 0.5 rad about z, each hand's reference place 4 cm off where the hand is
    comptime TH = 0.5
    o[3] = Scalar[DT](cos(TH / 2.0))
    o[6] = Scalar[DT](sin(TH / 2.0))
    var off: List[Float64] = [0.04, 0.0, 0.0, 0.0, -0.024, 0.032]       # in the box's frame, 4 cm each
    var hands: List[Int] = [hl, hr]
    for i in range(2):
        var dx = Float64(fk[hands[i] * 3]) - Float64(o[0])
        var dy = Float64(fk[hands[i] * 3 + 1]) - Float64(o[1])
        oref[SM_OBJ + 3 * i] = Scalar[DT](cos(TH) * dx + sin(TH) * dy + off[3 * i])
        oref[SM_OBJ + 3 * i + 1] = Scalar[DT](-sin(TH) * dx + cos(TH) * dy + off[3 * i + 1])
        oref[SM_OBJ + 3 * i + 2] = Scalar[DT](Float64(fk[hands[i] * 3 + 2]) - Float64(o[2]) + off[3 * i + 2])
    tp[FT_KH] = Scalar[DT](300)
    tp[FT_HMAX] = Scalar[DT](100)
    var act5 = List[Scalar[DT]](length=R.NDOF, fill=0)
    sm_force_teacher_lane[R](ARM0, Scalar[DT](0), Scalar[DT](1), mptr(tp), tree, table, mptr(ks), sp, sp, mptr(o), mptr(oref), mptr(scratch), mptr(act5))
    var pull0 = _v(Float64(scratch[FO + 8]), Float64(scratch[FO + 9]), Float64(scratch[FO + 10]))
    var pull1 = _v(Float64(scratch[FO + 11]), Float64(scratch[FO + 12]), Float64(scratch[FO + 13]))
    var exp0 = _v(300.0 * 0.04 * cos(TH), 300.0 * 0.04 * sin(TH), 0.0)
    var exp1 = _v(300.0 * 0.024 * sin(TH), -300.0 * 0.024 * cos(TH), 300.0 * 0.032)
    var wrench = 0.0
    for k in range(8):
        wrench = max(wrench, abs(Float64(scratch[FO + k])))
    var wq = 0.0
    for j in range(ARM0, R.NDOF):
        wq += Float64(act5[j]) * KS * dq[j]
    var hand_a0 = _v(Float64(fa[hl * 3]) - Float64(fb[hl * 3]), Float64(fa[hl * 3 + 1]) - Float64(fb[hl * 3 + 1]), Float64(fa[hl * 3 + 2]) - Float64(fb[hl * 3 + 2]))
    var hand_a1 = _v(Float64(fa[hr * 3]) - Float64(fb[hr * 3]), Float64(fa[hr * 3 + 1]) - Float64(fb[hr * 3 + 1]), Float64(fa[hr * 3 + 2]) - Float64(fb[hr * 3 + 2]))
    var wx = (dot3(hand_a0, pull0) + dot3(hand_a1, pull1)) / 2.0
    tp[FT_HMAX] = Scalar[DT](5)
    var act6 = List[Scalar[DT]](length=R.NDOF, fill=0)
    sm_force_teacher_lane[R](ARM0, Scalar[DT](0), Scalar[DT](1), mptr(tp), tree, table, mptr(ks), sp, sp, mptr(o), mptr(oref), mptr(scratch), mptr(act6))
    var capped = _v(Float64(scratch[FO + 8]), Float64(scratch[FO + 9]), Float64(scratch[FO + 10]))
    t.truth(
        _n3(pull0 - exp0) <= 2e-3 and _n3(pull1 - exp1) <= 2e-3 and wrench == 0.0 and abs(wq - wx) <= 0.05 * abs(wx) and abs(wx) > 1e-4
        and abs(_n3(capped) - 5.0) <= 1e-4,
        "the pull alone: 12 N on each hand toward its place on the turned box (off by " + String(_n3(pull0 - exp0)) + " and " + String(_n3(pull1 - exp1))
        + " N), no wrench, its virtual work " + String(wq) + " against " + String(wx) + " J, capped at 5 N",
    )


def main() raises:
    var t = Tally()
    test_known(t)
    test_reward_vjp(t)
    test_features(t)
    test_ref_rows(t)
    test_lead_in(t)
    test_carry_tally(t)
    test_tracking_kernels(t)
    test_force_teacher(t)
    print("\n=== " + String(t.checks - t.fails) + " / " + String(t.checks) + " passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_obj_task: " + String(t.fails) + " failed")
