# +--------------------------------------------------------------------------+ #
# | SuperMimic with an object: the task — reward terms, the encoder's features
# +--------------------------------------------------------------------------+ #
"""What the object interface adds to the tracker's analytic heads, per lane,
each function with its VJP. `docs/OBJECT_INTERACTION_PLAN.md` §4.3 and §4.4,
step O1.4. There is no reference implementation: the oracle is
`tools/supermimic/obj_imag_oracle.py`, an independent torch transcription.

    oref (19)   = [the object's reference state (13, `object_spec`'s o),
                   the two hands in the REFERENCE object's frame (3 + 3)]
                  one row per library frame, built once (`sm_object_ref_rows`)

    reward      r_obj = Σ of three terms, added to the tracker's eight:
        position     dt·w·exp(−|p − p_ref|² / σ²)                    world, absolute
        orientation  dt·w·exp(−|log(q_ref ⊗ q̄)|² / σ²)               `heads.sq_quat_err`
        hands        dt·w·exp(−½ Σ_i |R(q)⁻¹ (p_hand,i − p) − h_ref,i|² / σ²)
                     each hand, seen from the object, where the reference has
                     it seen from the reference object
      (dt·w, 1/σ²) per term are DATA (`rw`, 6 numbers), so a revision of the
      weights is a run, not a build.

    features (33) = [feat(s, o) 15, feat(s, o_ref) 15, (their positions' difference) / 0.1]
                  `object_spec.sm_obj_feat_lane`: the object, and where it
                  should be, seen from the robot's heading frame

WHY THE HANDS TERM IS RELATIVE TO THE REFERENCE and not a distance to a face of
the box: the plan's "kernel on the distance from each hand body to its face"
needs to know which face each hand holds and when; the reference already says
both. While the reference carries, the term asks for the reference's grasp;
while its hands are away from the box, the term asks for the same and is
satisfied by tracking the body. It stands in for a contact reward: the world
model's state holds no contact force.

⚠ The orientation term compares body frames: a cuboid turned half a turn about
an axis looks the same and scores as wrong. The simulated box starts each
episode AT its reference pose, so the frames agree unless it tumbles.

Pointer-in / pointer-out, no allocation, no Float64 at runtime: the bodies of
GPU kernels (`lanes.mojo`'s rules).
"""

from std.math import acos, exp, sqrt

from noeira.deep_agents.supermimic.heads import SMHeadLayout, qapply_inv, sq_quat_err, sq_quat_err_vjp
from noeira.deep_agents.supermimic.kinematics import FP, Q4, cross3, dot3, qapply, qconj, qmul, sm_fk_lane
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_FEAT, sm_obj_feat_lane, sm_obj_feat_vjp_lane
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.state_spec import qapply_inv_vjp, qnormalize, qnormalize_vjp
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr


comptime SM_OREF: Int = SM_OBJ + 6
"""One reference row: the object's state, then the two hands in its frame."""
comptime SM_OBJ_TERMS: Int = 3
comptime OT_POS: Int = 0
comptime OT_ORI: Int = 1
comptime OT_HANDS: Int = 2
comptime SM_ORW: Int = 2 * SM_OBJ_TERMS
"""The reward's data: (dt·w, 1/σ²) for position, orientation, hands."""
comptime SM_OE_FEAT: Int = 2 * SM_OBJ_FEAT + 3
comptime SM_OE_DP: Float64 = 0.1
"""The scale of the features' position difference (m)."""
comptime SM_OE_SCRATCH: Int = 2 * SM_OBJ_FEAT + SM_OBJ
"""`sm_oe_feat_vjp_lane`'s per-lane scratch."""

# starting weights and widths (the plan's §4.3: "starting choices")
comptime SM_OW_POS: Float64 = 1.0
comptime SM_OSTD_POS: Float64 = 0.2
comptime SM_OW_ORI: Float64 = 0.5
comptime SM_OSTD_ORI: Float64 = 0.4
comptime SM_OW_HANDS: Float64 = 1.0
comptime SM_OSTD_HANDS: Float64 = 0.1
comptime _DT: Float64 = 0.02


def sm_obj_reward_params(
    w_pos: Float64 = SM_OW_POS, std_pos: Float64 = SM_OSTD_POS, w_ori: Float64 = SM_OW_ORI, std_ori: Float64 = SM_OSTD_ORI,
    w_hands: Float64 = SM_OW_HANDS, std_hands: Float64 = SM_OSTD_HANDS,
) -> List[Scalar[DT]]:
    """`rw`: (dt·w, 1/σ²) per term, from weights and widths."""
    return [
        Scalar[DT](_DT * w_pos), Scalar[DT](1.0 / (std_pos * std_pos)),
        Scalar[DT](_DT * w_ori), Scalar[DT](1.0 / (std_ori * std_ori)),
        Scalar[DT](_DT * w_hands), Scalar[DT](1.0 / (std_hands * std_hands)),
    ]


@always_inline
def _v3[dt: DType](p: FP[dt], i: Int) -> Q4[dt]:
    return Q4[dt](p[unsafe_offset=i], p[unsafe_offset=i + 1], p[unsafe_offset=i + 2], 0)


@always_inline
def _q4[dt: DType](p: FP[dt], i: Int) -> Q4[dt]:
    return Q4[dt](p[unsafe_offset=i], p[unsafe_offset=i + 1], p[unsafe_offset=i + 2], p[unsafe_offset=i + 3])


@always_inline
def _add3[dt: DType](p: FP[dt], i: Int, v: Q4[dt]):
    p[unsafe_offset=i] += v[0]
    p[unsafe_offset=i + 1] += v[1]
    p[unsafe_offset=i + 2] += v[2]


# ── the reward ──────────────────────────────────────────────────────────────
def sm_obj_reward_lane[dt: DType, R: TrackingRobot](
    table: FP[dt], pos: FP[dt], o: FP[dt], oref: FP[dt], rw: FP[dt], terms: FP[dt]
) -> Scalar[dt] where dt.is_floating_point():
    """Writes the three terms of the object `o` against the reference row
    `oref`, the hands read from the FK's position block `pos` (the heads'
    scratch after `sm_heads_lane`); returns their sum."""
    comptime L = SMHeadLayout[R]
    var p = _v3(o, 0)
    var qo = qnormalize(_q4(o, 3))[0]
    var qr = qnormalize(_q4(oref, 3))[0]
    var dp = p - _v3(oref, 0)
    var e_p = dot3(dp, dp)
    var e_q = sq_quat_err(qmul(qr, qconj(qo)))
    var e_h = Scalar[dt](0)
    comptime for i in range(2):
        var hb = Int(table[unsafe_offset=L.HANDS + i])
        var d = qapply_inv(qo, _v3(pos, hb * 3) - p) - _v3(oref, SM_OBJ + 3 * i)
        e_h += dot3(d, d)
    e_h = e_h / 2
    terms[unsafe_offset=OT_POS] = rw[unsafe_offset=0] * exp(-e_p * rw[unsafe_offset=1])
    terms[unsafe_offset=OT_ORI] = rw[unsafe_offset=2] * exp(-e_q * rw[unsafe_offset=3])
    terms[unsafe_offset=OT_HANDS] = rw[unsafe_offset=4] * exp(-e_h * rw[unsafe_offset=5])
    return terms[unsafe_offset=OT_POS] + terms[unsafe_offset=OT_ORI] + terms[unsafe_offset=OT_HANDS]


def sm_obj_reward_vjp_lane[dt: DType, R: TrackingRobot](
    table: FP[dt], pos: FP[dt], o: FP[dt], oref: FP[dt], rw: FP[dt], g_r: Scalar[dt], gpos: FP[dt], g_o: FP[dt]
) where dt.is_floating_point():
    """For the cotangent `g_r` of the three terms' sum: `gpos += ∂/∂` the FK's
    positions (the two hand rows), `g_o += ∂/∂o` (both ADD)."""
    comptime L = SMHeadLayout[R]
    var p = _v3(o, 0)
    var on = qnormalize(_q4(o, 3))
    var qo = on[0]
    var qr = qnormalize(_q4(oref, 3))[0]
    var dp = p - _v3(oref, 0)
    var d_q = qmul(qr, qconj(qo))
    var e_h = Scalar[dt](0)
    comptime for i in range(2):
        var hb = Int(table[unsafe_offset=L.HANDS + i])
        var d = qapply_inv(qo, _v3(pos, hb * 3) - p) - _v3(oref, SM_OBJ + 3 * i)
        e_h += dot3(d, d)
    e_h = e_h / 2
    # ∂term/∂e = −term / σ²
    var ge_p = -g_r * rw[unsafe_offset=0] * exp(-dot3(dp, dp) * rw[unsafe_offset=1]) * rw[unsafe_offset=1]
    var ge_q = -g_r * rw[unsafe_offset=2] * exp(-sq_quat_err(d_q) * rw[unsafe_offset=3]) * rw[unsafe_offset=3]
    var ge_h = -g_r * rw[unsafe_offset=4] * exp(-e_h * rw[unsafe_offset=5]) * rw[unsafe_offset=5] / 2
    var g_p = dp * (ge_p * 2)
    # e_q = sqerr(q_ref ⊗ q̄): ∂/∂q = conj(conj(q_ref) ⊗ g_d)
    var g_qo = qconj(qmul(qconj(qr), sq_quat_err_vjp(d_q) * ge_q))
    comptime for i in range(2):
        var hb = Int(table[unsafe_offset=L.HANDS + i])
        var v = _v3(pos, hb * 3) - p
        var d = qapply_inv(qo, v) - _v3(oref, SM_OBJ + 3 * i)
        var a = qapply_inv_vjp(qo, v, d * (ge_h * 2))
        g_qo += a[0]
        _add3(gpos, hb * 3, a[1])
        g_p -= a[1]
    _add3(g_o, 0, g_p)
    var gq = qnormalize_vjp(qo, on[1], g_qo)
    comptime for k in range(4):
        g_o[unsafe_offset=3 + k] += gq[k]


# ── the features of the object encoder and of the interface's critic ────────
def sm_oe_feat_lane[dt: DType, R: TrackingRobot](s: FP[dt], o: FP[dt], oref: FP[dt], f: FP[dt]) where dt.is_floating_point():
    """f (33) = [feat(s, o), feat(s, o_ref), (feat(s, o_ref) − feat(s, o))[:3] / 0.1]."""
    sm_obj_feat_lane[dt, R](s, o, f)
    sm_obj_feat_lane[dt, R](s, oref, f.unsafe_offset(SM_OBJ_FEAT))
    comptime for k in range(3):
        f[unsafe_offset=2 * SM_OBJ_FEAT + k] = (f[unsafe_offset=SM_OBJ_FEAT + k] - f[unsafe_offset=k]) / Scalar[dt](SM_OE_DP)


def sm_oe_feat_vjp_lane[dt: DType, R: TrackingRobot](
    s: FP[dt], o: FP[dt], oref: FP[dt], g_f: FP[dt], tmp: FP[dt], g_s: FP[dt], g_o: FP[dt]
) where dt.is_floating_point():
    """`g_s += ∂/∂s`, `g_o += ∂/∂o` of <g_f, features(s, o, o_ref)> (both
    ADD; the reference is data). `tmp` holds `SM_OE_SCRATCH` values."""
    comptime for k in range(2 * SM_OBJ_FEAT):
        tmp[unsafe_offset=k] = g_f[unsafe_offset=k]
    comptime for k in range(3):
        var gd = g_f[unsafe_offset=2 * SM_OBJ_FEAT + k] / Scalar[dt](SM_OE_DP)
        tmp[unsafe_offset=k] -= gd
        tmp[unsafe_offset=SM_OBJ_FEAT + k] += gd
    var sink = tmp.unsafe_offset(2 * SM_OBJ_FEAT)
    comptime for k in range(SM_OBJ):
        sink[unsafe_offset=k] = 0
    sm_obj_feat_vjp_lane[dt, R](s, o, tmp, g_s, g_o)
    sm_obj_feat_vjp_lane[dt, R](s, oref, tmp.unsafe_offset(SM_OBJ_FEAT), g_s, sink)


# ── the reference rows ──────────────────────────────────────────────────────
def sm_object_ref_rows[R: TrackingRobot](
    mut tb: SMTables[DT], mut rows: List[Scalar[DT]], mut track: List[Scalar[DT]]
) raises -> List[Scalar[DT]]:
    """`SM_OREF` numbers per library frame: the object's track row
    (`object_env_gpu.sm_object_track`), then each hand of the library's robot
    pose (`rows`, S per frame) in that object's frame."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    var frames = len(rows) // D.S
    if len(track) != frames * SM_OBJ:
        raise Error("sm_object_ref_rows: " + String(frames) + " library frames, a track of " + String(len(track) // SM_OBJ))
    var out = List[Scalar[DT]](length=frames * SM_OREF, fill=0)
    var fk = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var tree = tb.tree_p["cpu"]()
    var table = tb.head_p["cpu"]()
    var pos = mptr(fk)
    var rp = mptr(rows)
    var tp_ = mptr(track)
    for f in range(frames):
        var s = rp.unsafe_offset(f * D.S)
        sm_fk_lane[DT, D.NB](tree, s, s.unsafe_offset(D.NQ), pos, pos.unsafe_offset(D.NB * 3), pos.unsafe_offset(D.NB * 7), pos.unsafe_offset(D.NB * 10))
        var o = tp_.unsafe_offset(f * SM_OBJ)
        for k in range(SM_OBJ):
            out[f * SM_OREF + k] = o[unsafe_offset=k]
        var qo = qnormalize(_q4(o, 3))[0]
        comptime for i in range(2):
            var hb = Int(table[unsafe_offset=L.HANDS + i])
            var h = qapply_inv(qo, _v3(pos, hb * 3) - _v3(o, 0))
            comptime for c in range(3):
                out[f * SM_OREF + SM_OBJ + 3 * i + c] = h[c]
    return out^


# ── a lead-in before each clip ──────────────────────────────────────────────
def sm_lead_in[R: TrackingRobot](
    mut rows: List[Scalar[DT]], mut lens: List[Int], mut poses: List[Float64], frames: Int
) raises:
    """Each clip of `lens` gets `frames` more frames at its start: the robot
    standing at the clip's first pose with zero velocity, the box at its first
    pose. In place: `rows` (S a frame, the clips of `lens` first; later rows
    keep their place after them), `lens`, and `poses` (the box's 7 numbers a
    frame, as the drivers read them, BEFORE `sm_object_track` turns them into
    a track, which then gives the box zero velocity over the lead-in).

    Why (`docs/OBJECT_INTERACTION_PLAN.md` §6, 16 Nov): half the carry clips
    lift the box less than a second after their first frame, some after a
    sixth of a second. A start perturbed there (the training's reset noise
    includes a velocity of up to 0.5 m/s) has no time to settle before the
    grasp, and those are the trials that lose the box. With a lead-in the
    clip's first frame is a stand, as a carry begins in practice.

    The reference's velocity is zero through the lead-in and the clip's own
    from its first frame on: a step in velocity there, as at any clip's start."""
    comptime D = SMDims[R]
    var total = 0
    for c in range(len(lens)):
        total += lens[c]
    if frames < 0 or total * D.S > len(rows) or total * 7 != len(poses):
        raise Error("sm_lead_in: " + String(total) + " frames named, " + String(len(rows) // D.S) + " rows, " + String(len(poses) // 7) + " box poses")
    if frames == 0:
        return
    var new_rows = List[Scalar[DT]](capacity=len(rows) + len(lens) * frames * D.S)
    var new_poses = List[Float64](capacity=len(poses) + len(lens) * frames * 7)
    var off = 0
    for c in range(len(lens)):
        for _ in range(frames):
            for k in range(D.S):
                new_rows.append(rows[off * D.S + k] if k < D.NQ else Scalar[DT](0))
            for k in range(7):
                new_poses.append(poses[off * 7 + k])
        for i in range(lens[c] * D.S):
            new_rows.append(rows[off * D.S + i])
        for i in range(lens[c] * 7):
            new_poses.append(poses[off * 7 + i])
        off += lens[c]
        lens[c] += frames
    for i in range(off * D.S, len(rows)):            # rows of clips not named
        new_rows.append(rows[i])
    rows = new_rows^
    poses = new_poses^


# ── a run's account of the carry (host; the benchmark's one rule) ───────────
comptime SM_OBJ_NEAR: Float64 = 0.10
"""A lane "holds" its object when it is within this of its reference (m):
O1.5's distance."""
comptime SM_OBJ_LIFT: Float64 = 0.05
"""A step is a CARRY step when the reference box is at least this above its
clip's first height (m); the simulated box counts as UP by the same rule."""
comptime SM_OBJ_HELD_SHARE: Float64 = 0.95
"""The bar: the share of a run's carry steps that must be held."""
comptime SM_OBJ_LEVEL: Float64 = 15.0
"""A box in hand is LEVEL when its mean lean over the carry's second half is
at most this (degrees). A starting choice: the reward's orientation width is
0.4 rad, 23 degrees."""


def sm_obj_lean_deg(
    qw: Float64, qx: Float64, qy: Float64, qz: Float64, rw: Float64, rx: Float64, ry: Float64, rz: Float64
) -> Float64:
    """The object's LEAN against its reference, degrees: the angle between
    the world's up as the object (quaternion q, WXYZ) sees it and as the
    reference object (r) sees it. A turn of either about the vertical leaves
    it as it is: it reads how the object hangs, not where it faces."""
    var nq = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    var nr = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
    if nq < 1e-12 or nr < 1e-12:
        return 0.0
    var a = qw / nq
    var b = qx / nq
    var c = qy / nq
    var d = qz / nq
    var e = rw / nr
    var f = rx / nr
    var g = ry / nr
    var h = rz / nr
    # R(q)ᵀ ẑ: the rotation matrix's third row
    var dot = (
        (2.0 * (b * d - a * c)) * (2.0 * (f * h - e * g)) + (2.0 * (c * d + a * b)) * (2.0 * (g * h + e * f))
        + (1.0 - 2.0 * (b * b + c * c)) * (1.0 - 2.0 * (f * f + g * g))
    )
    return acos(max(-1.0, min(1.0, dot))) * 57.29577951308232


@fieldwise_init
struct SMObjCarryTally(Copyable, Movable):
    """One run's carry, step by step (`note`). Four readings of it:
    `at_bar` (held on `SM_OBJ_HELD_SHARE` of the carry steps), `carried`
    (the simulated box is up on more than half of the carry's SECOND half:
    it was picked up and is still in the hands late in the carry, wherever
    it is against the reference), the mean distance over that second half
    (the error once in hand), and the mean LEAN over it (`sm_obj_lean_deg`;
    `level`: at most `SM_OBJ_LEVEL`). The second half is the carry
    steps after the first `carry_total // 2`, `carry_total` being the clip's
    count (`sm_obj_carry_steps`), known before the run: a run that falls
    early has no second half and is not carried.

    Why three: a box pushed on the floor before the lift, picked up where it
    lies and brought back to its reference misses the bar for the time the
    way back takes, and is a carry (`docs/OBJECT_INTERACTION_PLAN.md` §6,
    17 Nov). Why the lean: the bar, the carry and the error read the box's
    CENTRE; a box that hangs from the hands at 40 degrees with its centre in
    place passes all three (the same, 18 Nov)."""
    var carry_total: Int
    var carry: Int
    var held: Int
    var emax: Float64
    var esum: Float64
    var late: Int
    var late_up: Int
    var late_esum: Float64
    var late_lean: Float64

    def note(mut self, ref_rise: Float64, box_rise: Float64, err: Float64, lean: Float64):
        """One step: the reference box's and the simulated box's height above
        the clip's first reference height, their distance (m), and the
        simulated box's lean against the reference's (degrees)."""
        if ref_rise < SM_OBJ_LIFT:
            return
        self.carry += 1
        self.esum += err
        self.emax = max(self.emax, err)
        if err <= SM_OBJ_NEAR:
            self.held += 1
        if self.carry > self.carry_total // 2:
            self.late += 1
            self.late_esum += err
            self.late_lean += lean
            if box_rise >= SM_OBJ_LIFT:
                self.late_up += 1

    def at_bar(self) -> Bool:
        return self.carry > 0 and Float64(self.held) >= SM_OBJ_HELD_SHARE * Float64(self.carry)

    def carried(self) -> Bool:
        return self.late > 0 and 2 * self.late_up > self.late

    def mean_err(self) -> Float64:
        return self.esum / Float64(max(self.carry, 1))

    def mean_err_late(self) -> Float64:
        return self.late_esum / Float64(max(self.late, 1))

    def mean_lean_late(self) -> Float64:
        return self.late_lean / Float64(max(self.late, 1))

    def level(self) -> Bool:
        """Carried, and level over the carry's second half."""
        return self.carried() and self.mean_lean_late() <= SM_OBJ_LEVEL


def sm_obj_carry_tally(carry_total: Int) -> SMObjCarryTally:
    return SMObjCarryTally(carry_total, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0)


def sm_obj_carry_steps(orefs: List[Scalar[DT]], first_frame: Int, frames: Int) -> Int:
    """The carry frames of the clip of `frames` frames starting at library
    frame `first_frame` (`orefs`: `sm_object_ref_rows`)."""
    var z0 = Float64(orefs[first_frame * SM_OREF + 2])
    var n = 0
    for f in range(first_frame, first_frame + frames):
        if Float64(orefs[f * SM_OREF + 2]) - z0 >= SM_OBJ_LIFT:
            n += 1
    return n


def sm_trunk_legs_mask[R: TrackingRobot]() -> List[Scalar[DT]]:
    """`heads.sm_tracking_kernels_lane`'s mask that leaves the arms out: 1 per
    tracked body, 0 for those named as a shoulder, an elbow or a wrist."""
    var names = R.tracked_bodies()
    var m = List[Scalar[DT]](length=len(names), fill=1)
    for k in range(len(names)):
        if names[k].find("shoulder") >= 0 or names[k].find("elbow") >= 0 or names[k].find("wrist") >= 0:
            m[k] = Scalar[DT](0)
    return m^


# ── a squeezed copy of the reference ────────────────────────────────────────
comptime SM_SQ_LEAD: Int = 10         # frames before the reference lifts at which the squeeze starts (and its ramp)
comptime SM_SQ_REACH: Float64 = 0.45  # a hand farther than this from the box's centre is left alone (m)
comptime SM_SQ_LIFT: Float64 = 0.01   # the reference box counts as lifted above its clip's first height by this (m)
comptime SM_SQ_EPS: Float64 = 0.01    # rad, the finite difference of the FK
comptime SM_SQ_DAMP: Float64 = 0.0004 # λ² of the damped least squares (m²)
comptime SM_SQ_MAX_ACTION: Float64 = 2.0  # the squeeze's largest offset on one joint, action units


def sm_squeeze_weights(lens: List[Int], track: List[Scalar[DT]], stride: Int, lead: Int = SM_SQ_LEAD) raises -> List[Float64]:
    """One weight in [0, 1] per frame of the clips of `lens`: 1 while the
    reference box is lifted (`SM_SQ_LIFT` above its clip's first height) or
    will be within `lead` frames, with a ramp of `SM_SQ_LEAD` frames on
    both sides. `track`: `stride` numbers a frame, the box's position first."""
    var total = 0
    for c in range(len(lens)):
        total += lens[c]
    if total * stride > len(track) or stride < 3:
        raise Error("sm_squeeze_weights: " + String(total) + " frames named, a track of " + String(len(track) // max(stride, 1)))
    var out = List[Float64](length=total, fill=0.0)
    var off = 0
    for c in range(len(lens)):
        var n = lens[c]
        var z0 = Float64(track[off * stride + 2])
        var wgt = List[Float64](length=n, fill=0.0)
        for f in range(n):
            for g in range(f, min(f + lead, n - 1) + 1):
                if Float64(track[(off + g) * stride + 2]) - z0 > SM_SQ_LIFT:
                    wgt[f] = 1.0
        for f in range(n):
            var acc = 0.0
            var cn = 0
            for g in range(max(f - SM_SQ_LEAD // 2, 0), min(f + SM_SQ_LEAD // 2, n - 1) + 1):
                acc += wgt[g]
                cn += 1
            out[off + f] = acc / Float64(cn)
        off += n
    return out^


def sm_squeeze_reference[R: TrackingRobot](
    mut tb: SMTables[DT], rows: List[Scalar[DT]], lens: List[Int], track: List[Scalar[DT]], stride: Int, depth: Float64, arm0: Int,
    lower: Float64 = 0.0,
) raises -> List[Scalar[DT]]:
    """A copy of the library rows whose hands aim `depth` metres INTO the box
    while the reference box is lifted: for the clips of `lens` (the first
    Σ lens frames of `rows`; later rows are copied as they are), each hand
    within `SM_SQ_REACH` of the reference box's centre is moved toward that
    centre by damped least squares on the joints `arm0 ..` (the hand's
    Jacobian by finite differences of the FK), ramped over `SM_SQ_LEAD`
    frames around the lift. Velocities are left as they are.

    `track`: the box's reference, `stride` numbers a frame, its position
    first (`sm_object_track`'s rows, or raw pose rows).

    `lower` in [0, 1]: each hand is ALSO moved down by that share of its
    height above the box's centre (1: to the centre's height), `depth` then
    being horizontal only. A probe of the grasp's height; the arms alone
    answer it, so a hand the arms cannot bring lower stays where it is.

    Why: the reference's hands touch the box and do not press on it. A
    tracker following this copy presses, so its lanes show the object network
    held boxes: an exploration prior, not a policy (the tracker lifts the box
    on a minority of clips with it, `docs/OBJECT_INTERACTION_PLAN.md` §6)."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    var na = R.NDOF - arm0
    var total = 0
    for c in range(len(lens)):
        total += lens[c]
    if total * D.S > len(rows) or total * stride > len(track) or na <= 0 or stride < 3:
        raise Error("sm_squeeze_reference: " + String(total) + " frames named, " + String(len(rows) // D.S) + " rows, a track of " + String(len(track) // max(stride, 1)))
    var out = rows.copy()
    var e0 = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var e1 = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var q0p = mptr(e0)
    var q1p = mptr(e1)
    var tree = tb.tree_p["cpu"]()
    var table = tb.head_p["cpu"]()
    var hb = List[Int](length=2, fill=0)
    for i in range(2):
        hb[i] = Int(table[unsafe_offset=L.HANDS + i])
    var op = mptr(out)
    var sm = sm_squeeze_weights(lens, track, stride)
    var off = 0
    for c in range(len(lens)):
        var n = lens[c]
        for f in range(n):
            if sm[off + f] <= 0.0:
                continue
            var a = (off + f) * D.S
            var row = op.unsafe_offset(a)
            sm_fk_lane[DT, D.NB](tree, row, row.unsafe_offset(D.NQ), q0p, q0p.unsafe_offset(D.NB * 3), q0p.unsafe_offset(D.NB * 7), q0p.unsafe_offset(D.NB * 10))
            var jac = List[Float64](length=2 * 3 * na, fill=0.0)
            for j in range(na):
                var keep = out[a + 7 + arm0 + j]
                out[a + 7 + arm0 + j] = keep + Scalar[DT](SM_SQ_EPS)
                sm_fk_lane[DT, D.NB](tree, row, row.unsafe_offset(D.NQ), q1p, q1p.unsafe_offset(D.NB * 3), q1p.unsafe_offset(D.NB * 7), q1p.unsafe_offset(D.NB * 10))
                out[a + 7 + arm0 + j] = keep
                for i in range(2):
                    for k in range(3):
                        jac[(i * 3 + k) * na + j] = (Float64(e1[hb[i] * 3 + k]) - Float64(e0[hb[i] * 3 + k])) / SM_SQ_EPS
            var dq = List[Float64](length=na, fill=0.0)
            for i in range(2):
                var dx = List[Float64](length=3, fill=0.0)
                var dn = 0.0
                for k in range(3):
                    dx[k] = Float64(track[(off + f) * stride + k]) - Float64(e0[hb[i] * 3 + k])
                    dn += dx[k] * dx[k]
                dn = sqrt(dn)
                if dn > SM_SQ_REACH or dn < 1e-6:
                    continue
                if lower > 0.0:
                    var hz = -dx[2]                       # the hand above the box's centre
                    var hn = sqrt(dx[0] * dx[0] + dx[1] * dx[1])
                    dx[0] = dx[0] / max(hn, 1e-6) * depth * sm[off + f]
                    dx[1] = dx[1] / max(hn, 1e-6) * depth * sm[off + f]
                    dx[2] = -lower * hz * sm[off + f]
                else:
                    for k in range(3):
                        dx[k] = dx[k] / dn * depth * sm[off + f]
                # dq += Jᵀ (J Jᵀ + λ² I)⁻¹ dx, the 3 × 3 system by its adjugate
                var m = List[Float64](length=9, fill=0.0)
                for a_ in range(3):
                    for b_ in range(3):
                        var v = 0.0
                        for j in range(na):
                            v += jac[(i * 3 + a_) * na + j] * jac[(i * 3 + b_) * na + j]
                        m[a_ * 3 + b_] = v + (SM_SQ_DAMP if a_ == b_ else 0.0)
                var det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6])
                var y: List[Float64] = [
                    ((m[4] * m[8] - m[5] * m[7]) * dx[0] + (m[2] * m[7] - m[1] * m[8]) * dx[1] + (m[1] * m[5] - m[2] * m[4]) * dx[2]) / det,
                    ((m[5] * m[6] - m[3] * m[8]) * dx[0] + (m[0] * m[8] - m[2] * m[6]) * dx[1] + (m[2] * m[3] - m[0] * m[5]) * dx[2]) / det,
                    ((m[3] * m[7] - m[4] * m[6]) * dx[0] + (m[1] * m[6] - m[0] * m[7]) * dx[1] + (m[0] * m[4] - m[1] * m[3]) * dx[2]) / det,
                ]
                for j in range(na):
                    for k in range(3):
                        dq[j] += jac[(i * 3 + k) * na + j] * y[k]
            for j in range(na):
                out[a + 7 + arm0 + j] += Scalar[DT](dq[j])
        off += n
    return out^


def sm_squeeze_lane[R: TrackingRobot](
    arm0: Int, f: Scalar[DT], tree: FP[DT], table: FP[DT], ks: FP[DT], s: FP[DT], o: FP[DT], scratch: FP[DT], actions: FP[DT],
):
    """actions (NDOF) += the joint-target offsets that push each hand toward
    the object `o`'s centre with `f` newtons: Jᵀ F over `ks` (N·m per action
    unit), the hand's Jacobian by finite differences of the FK, on the joints
    `arm0 ..`. scratch: S + 2 · FK_BLOCK + 6. One state `s`.

    The device kernel `object_agent_gpu._k_squeeze` calls this per lane."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    comptime n = R.NDOF
    if f <= 0:
        return
    var sb = scratch
    var p0 = sb.unsafe_offset(D.S)
    var p1 = p0.unsafe_offset(D.FK_BLOCK)
    var pu = p1.unsafe_offset(D.FK_BLOCK)
    for k in range(D.S):
        sb[unsafe_offset=k] = s[unsafe_offset=k]
    sm_fk_lane[DT, D.NB](tree, sb, sb.unsafe_offset(D.NQ), p0, p0.unsafe_offset(D.NB * 3), p0.unsafe_offset(D.NB * 7), p0.unsafe_offset(D.NB * 10))
    var hit = False
    for i in range(2):
        var hb = Int(table[unsafe_offset=L.HANDS + i])
        var d2 = Scalar[DT](0)
        for k in range(3):
            var d = o[unsafe_offset=k] - p0[unsafe_offset=hb * 3 + k]
            pu[unsafe_offset=i * 3 + k] = d
            d2 += d * d
        var dn = sqrt(d2)
        var g = f / dn if (dn < Scalar[DT](SM_SQ_REACH) and dn > Scalar[DT](1e-6)) else Scalar[DT](0)
        if g > 0:
            hit = True
        for k in range(3):
            pu[unsafe_offset=i * 3 + k] = pu[unsafe_offset=i * 3 + k] * g
    if not hit:
        return
    var eps = Scalar[DT](SM_SQ_EPS)
    for j in range(arm0, n):
        var keep = sb[unsafe_offset=7 + j]
        sb[unsafe_offset=7 + j] = keep + eps
        sm_fk_lane[DT, D.NB](tree, sb, sb.unsafe_offset(D.NQ), p1, p1.unsafe_offset(D.NB * 3), p1.unsafe_offset(D.NB * 7), p1.unsafe_offset(D.NB * 10))
        sb[unsafe_offset=7 + j] = keep
        var tau = Scalar[DT](0)
        for i in range(2):
            var hb = Int(table[unsafe_offset=L.HANDS + i])
            for k in range(3):
                tau += (p1[unsafe_offset=hb * 3 + k] - p0[unsafe_offset=hb * 3 + k]) / eps * pu[unsafe_offset=i * 3 + k]
        actions[unsafe_offset=j] += max(Scalar[DT](-SM_SQ_MAX_ACTION), min(Scalar[DT](SM_SQ_MAX_ACTION), tau / ks[unsafe_offset=j]))


# ── a force teacher: the wrench the object needs, shared between the hands ──
comptime SM_FT_PARAMS: Int = 16
"""`sm_force_teacher_lane`'s data, in this order."""
comptime FT_SQUEEZE: Int = 0     # the squeeze's floor, N on each hand along the line that joins them
comptime FT_SQUEEZE_MAX: Int = 1 # its ceiling, N
comptime FT_MU: Int = 2          # the friction coefficient the allocation assumes (below the contact's: its margin)
comptime FT_MASS: Int = 3        # the object's mass, kg: the weight the hands are told to carry
comptime FT_KP: Int = 4          # N/m on the object's position error
comptime FT_KD: Int = 5          # N·s/m on its linear velocity error
comptime FT_KR: Int = 6          # N·m/rad on its orientation error
comptime FT_KW: Int = 7          # N·m·s/rad on its angular velocity error
comptime FT_POINT: Int = 8       # (3) where the LEFT hand's force acts, in that hand body's frame; the right hand's is its mirror in y
comptime FT_FMAX: Int = 11       # the largest force on one hand, N
comptime FT_REL: Int = 12        # not 0: the object's target is the reference's SEEN FROM THE ROBOT'S ROOT (position and heading)
comptime FT_MMAX: Int = 13       # the largest moment one hand gives about the hands' line, N·m (0: forces alone)
comptime FT_KH: Int = 14         # N/m pulling each hand BODY to where the reference has it, seen from the object
comptime FT_HMAX: Int = 15       # that pull's largest force on one hand, N
comptime SM_FT_G: Float64 = 9.81
comptime SM_FT_OUT: Int = 14
"""What the lane leaves at the end of its scratch: the two forces at the
points, the two moments about the hands' line, the two pulls on the hand
bodies."""


@always_inline
def sm_force_alloc[dt: DType](
    c0: Q4[dt], c1: Q4[dt], p: Q4[dt], fw: Q4[dt], mw: Q4[dt], s0: Scalar[dt], s1: Scalar[dt], mu: Scalar[dt]
) -> Tuple[Q4[dt], Q4[dt], Scalar[dt], Scalar[dt]] where dt.is_floating_point():
    """The forces two point contacts at `c0` and `c1` put on a body centred at
    `p`: (f0, f1, squeeze, the moment left over).

        f0 + f1 = fw                                        the force asked
        (c0 − p) × f0 + (c1 − p) × f1 = mw − its part along c1 − c0
            two point forces have no moment about the line that joins them;
            of the pairs that give the rest, the smallest
        f0 += s u,  f1 −= s u,  u = the unit of c1 − c0     an internal squeeze
            s = the least that puts both forces inside a friction cone of
            coefficient `mu` about u, kept between `s0` and `s1`
        the moment left over = (mw − (the contacts' midpoint − p) × fw) · u,
            N·m about u: what the hands have to give as moments, if at all

    Ctrl-F's analytic force model (arXiv 2611.05507, G f ≈ W*, f ∈ F) for two
    contacts, in closed form."""
    var d = c1 - c0
    var d2 = max(dot3(d, d), Scalar[dt](1e-8))
    var u = d / sqrt(d2)
    var mc = (c0 + c1) * Scalar[dt](0.5) - p
    var mres = mw - cross3(mc, fw)
    var g = cross3(mres, d) / d2
    var f0 = fw * Scalar[dt](0.5) - g
    var f1 = fw * Scalar[dt](0.5) + g
    var n0 = dot3(f0, u)
    var n1 = -dot3(f1, u)
    var t0 = f0 - u * n0
    var t1 = f1 + u * n1
    var need = max(sqrt(dot3(t0, t0)) / mu - n0, sqrt(dot3(t1, t1)) / mu - n1)
    var sq = min(max(s0, need), max(s0, s1))
    return (f0 + u * sq, f1 - u * sq, sq, dot3(mres, u))


@always_inline
def _heading[dt: DType](q: Q4[dt]) -> Q4[dt] where dt.is_floating_point():
    """The rotation about the world's z nearest to `q` (mjlab's `yaw_quat`)."""
    var nrm = sqrt(q[0] * q[0] + q[3] * q[3])
    if nrm < Scalar[dt](1e-6):
        return Q4[dt](1, 0, 0, 0)
    return Q4[dt](q[0] / nrm, 0, 0, q[3] / nrm)


def sm_force_teacher_lane[R: TrackingRobot](
    arm0: Int, w: Scalar[DT], wh: Scalar[DT], tp: FP[DT], tree: FP[DT], table: FP[DT], ks: FP[DT], s: FP[DT], ref_s: FP[DT], o: FP[DT],
    oref: FP[DT], scratch: FP[DT], actions: FP[DT],
):
    """actions (NDOF) += the joint-target offsets by which the two hands put
    on the object `o` the wrench that takes it to its target:

        force   w · [m g ẑ + kp (p* − p) + kd (v* − v)]
        moment  w · [kr e_R + kw (ω* − ω)],  e_R = 2 · vec(q* ⊗ q̄), world
        shared between the hands by `sm_force_alloc`, each force capped at
        `FT_FMAX`, acting at `FT_POINT` of its hand body; the moment about
        the hands' line, which no pair of forces gives, is asked of the two
        hands as moments, half each, capped at `FT_MMAX`; then Jᵀ over `ks`
        (N·m per action unit) on the joints `arm0 ..`, both Jacobians by
        finite differences of the FK

    and, weighted by `wh` (the reach: a weight that starts earlier than `w`),
    a pull on each hand BODY of `FT_KH` N/m toward where the reference has
    it seen from the object (`oref`'s hand rows, carried to the simulated
    object), capped at `FT_HMAX`: where the hands land on the object decides
    what the forces can do, and the forces do not choose it.

    The target (p*, q*, v*, ω*) is the reference row `oref`, or with `FT_REL`
    the same SEEN FROM THE ROBOT: carried from the reference robot's root
    (`ref_s`, a library row) to the simulated one's (`s`), position and
    heading. Why: a hand that pushes the object toward a point of the world
    pushes the robot away from it, so a robot behind its reference is pushed
    farther behind; where the robot stands is its legs' business.

    `w` in [0, 1] gates the wrench (`sm_squeeze_weights`: the reference
    lifts), the squeeze's floor included; nothing is added unless BOTH points
    are within `SM_SQ_REACH` of the object's centre. `tp`: `SM_FT_PARAMS`
    numbers. scratch: S + 2 · FK_BLOCK + `SM_FT_OUT`; its last numbers are
    left holding the forces, the moments and the pulls (0 when not added).

    Against `sm_squeeze_lane`: that one pushes each hand BODY (the wrist's
    origin) toward the object's centre with a constant force. A force there
    asks nothing of the wrist's last joint and little of the two before it,
    whose gains are a few N·m/rad, so the palm, 11 cm farther, gives way and
    the object is held by the wrist's end of the hand
    (`docs/OBJECT_INTERACTION_PLAN.md` §6). Here the point is the palm, and
    the forces follow the object's error."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    comptime n = R.NDOF
    var sb = scratch
    var p0 = sb.unsafe_offset(D.S)
    var p1 = p0.unsafe_offset(D.FK_BLOCK)
    var pu = p1.unsafe_offset(D.FK_BLOCK)
    for k in range(SM_FT_OUT):
        pu[unsafe_offset=k] = 0
    if w <= 0 and wh <= 0:
        return
    for k in range(D.S):
        sb[unsafe_offset=k] = s[unsafe_offset=k]
    sm_fk_lane[DT, D.NB](tree, sb, sb.unsafe_offset(D.NQ), p0, p0.unsafe_offset(D.NB * 3), p0.unsafe_offset(D.NB * 7), p0.unsafe_offset(D.NB * 10))
    var h0 = Int(table[unsafe_offset=L.HANDS])
    var h1 = Int(table[unsafe_offset=L.HANDS + 1])
    var l0 = Q4[DT](tp[unsafe_offset=FT_POINT], tp[unsafe_offset=FT_POINT + 1], tp[unsafe_offset=FT_POINT + 2], 0)
    var l1 = Q4[DT](tp[unsafe_offset=FT_POINT], -tp[unsafe_offset=FT_POINT + 1], tp[unsafe_offset=FT_POINT + 2], 0)
    var q0 = _q4(p0, D.NB * 3 + h0 * 4)
    var q1 = _q4(p0, D.NB * 3 + h1 * 4)
    var c0 = _v3(p0, h0 * 3) + qapply(q0, l0)
    var c1 = _v3(p0, h1 * 3) + qapply(q1, l1)
    var p = _v3(o, 0)
    var reach = Scalar[DT](SM_SQ_REACH * SM_SQ_REACH)
    if dot3(c0 - p, c0 - p) > reach or dot3(c1 - p, c1 - p) > reach:
        return
    var qo = qnormalize(_q4(o, 3))[0]
    var qr = qnormalize(_q4(oref, 3))[0]
    # the pull of each hand body to its place on the object
    var hmax = tp[unsafe_offset=FT_HMAX]
    var g0 = (p + qapply(qo, _v3(oref, SM_OBJ)) - _v3(p0, h0 * 3)) * (tp[unsafe_offset=FT_KH] * wh)
    var g1 = (p + qapply(qo, _v3(oref, SM_OBJ + 3)) - _v3(p0, h1 * 3)) * (tp[unsafe_offset=FT_KH] * wh)
    var b0 = sqrt(dot3(g0, g0))
    var b1 = sqrt(dot3(g1, g1))
    if b0 > hmax:
        g0 = g0 * (hmax / b0)
    if b1 > hmax:
        g1 = g1 * (hmax / b1)
    # the target: the reference row, or that row carried from the reference robot's root to this one's
    var pt = _v3(oref, 0)
    var vt = _v3(oref, 7)
    var wt = qapply(qr, _v3(oref, 10))
    var v_here = _v3(o, 7)
    if tp[unsafe_offset=FT_REL] != 0:
        var turn = qmul(_heading(_q4(s, 3)), qconj(_heading(_q4(ref_s, 3))))
        pt = _v3(s, 0) + qapply(turn, pt - _v3(ref_s, 0))
        vt = qapply(turn, vt - _v3(ref_s, D.NQ))
        v_here = v_here - _v3(s, D.NQ)
        wt = qapply(turn, wt)
        qr = qmul(turn, qr)
    var fw = (
        Q4[DT](0, 0, tp[unsafe_offset=FT_MASS] * Scalar[DT](SM_FT_G), 0) + (pt - p) * tp[unsafe_offset=FT_KP] + (vt - v_here) * tp[unsafe_offset=FT_KD]
    ) * w
    var dq = qmul(qr, qconj(qo))
    var sgn = Scalar[DT](2) if dq[0] >= 0 else Scalar[DT](-2)
    var mw = (Q4[DT](dq[1], dq[2], dq[3], 0) * (sgn * tp[unsafe_offset=FT_KR]) + (wt - qapply(qo, _v3(o, 10))) * tp[unsafe_offset=FT_KW]) * w
    var al = sm_force_alloc(c0, c1, p, fw, mw, tp[unsafe_offset=FT_SQUEEZE] * w, tp[unsafe_offset=FT_SQUEEZE_MAX], tp[unsafe_offset=FT_MU])
    var f0 = al[0]
    var f1 = al[1]
    var fmax = tp[unsafe_offset=FT_FMAX]
    var a0 = sqrt(dot3(f0, f0))
    var a1 = sqrt(dot3(f1, f1))
    if a0 > fmax:
        f0 = f0 * (fmax / a0)
    if a1 > fmax:
        f1 = f1 * (fmax / a1)
    var mmax = tp[unsafe_offset=FT_MMAX]
    var mh = max(-mmax, min(mmax, al[3] * Scalar[DT](0.5)))
    var d = c1 - c0
    var mv = d * (mh / sqrt(max(dot3(d, d), Scalar[DT](1e-8))))     # each hand's moment on the object, world
    comptime for k in range(3):
        pu[unsafe_offset=k] = f0[k]
        pu[unsafe_offset=3 + k] = f1[k]
        pu[unsafe_offset=8 + k] = g0[k]
        pu[unsafe_offset=11 + k] = g1[k]
    pu[unsafe_offset=6] = mh
    pu[unsafe_offset=7] = mh
    var eps = Scalar[DT](SM_SQ_EPS)
    for j in range(arm0, n):
        var keep = sb[unsafe_offset=7 + j]
        sb[unsafe_offset=7 + j] = keep + eps
        sm_fk_lane[DT, D.NB](tree, sb, sb.unsafe_offset(D.NQ), p1, p1.unsafe_offset(D.NB * 3), p1.unsafe_offset(D.NB * 7), p1.unsafe_offset(D.NB * 10))
        sb[unsafe_offset=7 + j] = keep
        var r0 = _q4(p1, D.NB * 3 + h0 * 4)
        var r1 = _q4(p1, D.NB * 3 + h1 * 4)
        var e0 = _v3(p1, h0 * 3) + qapply(r0, l0) - c0
        var e1 = _v3(p1, h1 * 3) + qapply(r1, l1) - c1
        var tau = dot3(e0, f0) + dot3(e1, f1) + dot3(_v3(p1, h0 * 3) - _v3(p0, h0 * 3), g0) + dot3(_v3(p1, h1 * 3) - _v3(p0, h1 * 3), g1)
        if mh != 0:
            # each hand body's turn for this joint's step: 2 · vec(q' ⊗ q̄), world
            var t0 = qmul(r0, qconj(q0))
            var t1 = qmul(r1, qconj(q1))
            var s0 = Scalar[DT](2) if t0[0] >= 0 else Scalar[DT](-2)
            var s1 = Scalar[DT](2) if t1[0] >= 0 else Scalar[DT](-2)
            tau += dot3(Q4[DT](t0[1], t0[2], t0[3], 0), mv) * s0 + dot3(Q4[DT](t1[1], t1[2], t1[3], 0), mv) * s1
        tau = tau / eps
        actions[unsafe_offset=j] += max(Scalar[DT](-SM_SQ_MAX_ACTION), min(Scalar[DT](SM_SQ_MAX_ACTION), tau / ks[unsafe_offset=j]))
