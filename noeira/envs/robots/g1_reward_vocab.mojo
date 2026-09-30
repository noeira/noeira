# +--------------------------------------------------------------------------+ #
# | The G1 reward vocabulary — named quantities, bands, products
# +--------------------------------------------------------------------------+ #
"""What a BFM-Zero prompt is allowed to say, in one place.

`z = E_rho[r(s) B(s)]` turns a reward into a policy with no training. This
module is the layer above that: the named physical quantities a reward may
mention, the predicate forms it may take, and how predicates compose.

WHY THIS IS A LIBRARY AND NOT A BLOCK IN THE EXAMPLE
====================================================
Two consumers need the identical definition of "right hand height":
`examples/g1/bfm_zero_reward_vocab.mojo` (which scores the POOL, building
the reward) and `examples/g1/bfm_zero_bank_build.mojo` (which scores the
ROLLOUT, judging it). If those two ever disagreed — one body index off, one
frame convention different — every command would appear to work while
measuring something else. `g1_quantities` is called by both. It is never
transcribed.

⚠ THE INDEX RULES THAT MAKE THAT DANGEROUS
==========================================
The privileged vector's four blocks do not agree on their body indexing:

    HEIGHT  offset 0     scalar, the root's world z
    POS     offset 1     30 bodies, THE ROOT IS DROPPED -> body s at 1+(s-1)*3
    ROT     offset 91    31 bodies, 6 each (tangent 3 | normal 3) -> s*6
    VEL     offset 277   31 bodies, 3 each -> s*3
    ANGVEL  offset 370   31 bodies, 3 each -> s*3

Reading a neighbouring body returns a perfectly plausible wrong `z`, never
an error. Every index here is derived from a named skeleton constant.

FRAMES
======
POS / ROT / VEL / ANGVEL are all in the HEADING frame: the root's yaw
rotation only. Two consequences used throughout:

- the heading frame preserves z, so an ABSOLUTE height is `root_z + lpos_z`;
- a lateral offset in this frame means "out to the robot's side", not "along
  the world y axis", which is what makes `*_lateral` meaningful while the
  robot turns.

PREDICATE FORMS
===============
After Meta Motivo (arXiv 2412.10062, App. C.3.1), rewards are indicators and
bands composed by PRODUCT, never Gaussians on a single column. Their most
complex task is

    I[up > 0.9] * I[z_head > 1.4] * exp(-|v|) * I[z_rankle > b]

where three of the four terms exist only to suppress failure modes — the
other arm coming up with the raised one, the robot walking instead of
standing, a "low root" reward satisfied by lying down. §12.50 and §12.51 of
`docs/BFM_ZERO_G1_REPRODUCTION.md` measured all three on this robot.

⚠ ONE-SIDED THRESHOLDS ON A HEIGHT ARE A TRAP. `body_height < x` rewards
every state below it, and the lowest states in LAFAN are lying down. Motivo
says it outright for low tasks: "the agent is encouraged to keep the pelvis
z-coordinate inside a predefined range". Use `OP_BAND`.
"""

from std.math import exp, sqrt

from noeira.envs.robots.unitree_g1_priv_obs import (
    G1_PRIV_OFF_HEIGHT, G1_PRIV_OFF_POS, G1_PRIV_OFF_ROT,
    G1_PRIV_OFF_VEL, G1_PRIV_OFF_ANGVEL,
)
from noeira.envs.robots.unitree_g1_xml import UNITREE_G1_STATE_DIM

# ── skeleton bodies, in `g1_skeleton_body` order ──────────────────────────
# 0 pelvis | 1-6 left leg | 7-12 right leg | 13-15 waist/torso
# 16-22 left arm | 23-29 right arm | 30 the virtual head (torso + 0.35 z)
comptime G1_SK_KNEE_L: Int = 4       # left_knee_link         (model body 5)
comptime G1_SK_ANKLE_L: Int = 6      # left_ankle_roll_link   (model body 7)
comptime G1_SK_KNEE_R: Int = 10      # right_knee_link        (model body 15)
comptime G1_SK_ANKLE_R: Int = 12     # right_ankle_roll_link  (model body 17)
comptime G1_SK_TORSO: Int = 15       # torso_link             (model body 24)
comptime G1_SK_WRIST_L: Int = 22     # left_wrist_yaw_link    (model body 31)
comptime G1_SK_WRIST_R: Int = 29     # right_wrist_yaw_link   (model body 39)
comptime G1_SK_HEAD: Int = 30        # the virtual head

comptime _P: Int = G1_PRIV_OFF_POS
comptime _R: Int = G1_PRIV_OFF_ROT
comptime _V: Int = G1_PRIV_OFF_VEL
comptime _W: Int = G1_PRIV_OFF_ANGVEL

# ── the named quantities ──────────────────────────────────────────────────
comptime G1_NVOC: Int = 14
comptime QV_BODY_H: Int = 0
comptime QV_HEAD_H: Int = 1
comptime QV_LHAND_H: Int = 2
comptime QV_RHAND_H: Int = 3
comptime QV_LHAND_LAT: Int = 4
comptime QV_RHAND_LAT: Int = 5
comptime QV_LFOOT_H: Int = 6
comptime QV_RFOOT_H: Int = 7
comptime QV_UPRIGHT: Int = 8
comptime QV_SPEED_FWD: Int = 9
comptime QV_SPEED_LAT: Int = 10
comptime QV_SPEED: Int = 11
comptime QV_YAW_RATE: Int = 12
comptime QV_TORSO_YAW: Int = 13

# ── predicate forms ───────────────────────────────────────────────────────
comptime OP_GT: Int = 0        # I[x > lo]
comptime OP_LT: Int = 1        # I[x < hi]
comptime OP_BAND: Int = 2      # I[lo < x < hi]
comptime OP_SOFT: Int = 3      # exp(-lo * |x|)


def g1_vocab_name(q: Int) -> String:
    if q == QV_BODY_H:
        return String("body_height")
    if q == QV_HEAD_H:
        return String("head_height")
    if q == QV_LHAND_H:
        return String("left_hand_height")
    if q == QV_RHAND_H:
        return String("right_hand_height")
    if q == QV_LHAND_LAT:
        return String("left_hand_lateral")
    if q == QV_RHAND_LAT:
        return String("right_hand_lateral")
    if q == QV_LFOOT_H:
        return String("left_foot_height")
    if q == QV_RFOOT_H:
        return String("right_foot_height")
    if q == QV_UPRIGHT:
        return String("upright")
    if q == QV_SPEED_FWD:
        return String("body_speed_forward")
    if q == QV_SPEED_LAT:
        return String("body_speed_lateral")
    if q == QV_SPEED:
        return String("body_speed")
    if q == QV_YAW_RATE:
        return String("body_angular_velocity_yaw")
    return String("torso_yaw")


@always_inline
def g1_quantities(ref o: List[Float64], base: Int, mut out: List[Float64], off: Int):
    """The 14 named quantities from one `[state 64 | privileged 463]` row.

    ⚠ THE ONE DEFINITION — see the module header. `o` is either a store row
    assembled by hand or `env.get_obs_list()`; both are laid out the same way,
    which is what lets the pool scorer and the rollout metric share this.
    """
    var p = base + UNITREE_G1_STATE_DIM
    var root_z = o[p + G1_PRIV_OFF_HEIGHT]
    out[off + QV_BODY_H] = root_z
    # ⚠ POS drops the root: body s lives at `_P + (s-1)*3`, NOT `_P + s*3`.
    out[off + QV_HEAD_H] = root_z + o[p + _P + (G1_SK_HEAD - 1) * 3 + 2]
    out[off + QV_LHAND_H] = root_z + o[p + _P + (G1_SK_WRIST_L - 1) * 3 + 2]
    out[off + QV_RHAND_H] = root_z + o[p + _P + (G1_SK_WRIST_R - 1) * 3 + 2]
    var ly = o[p + _P + (G1_SK_WRIST_L - 1) * 3 + 1]
    var ry = o[p + _P + (G1_SK_WRIST_R - 1) * 3 + 1]
    out[off + QV_LHAND_LAT] = ly if ly > 0.0 else -ly
    out[off + QV_RHAND_LAT] = ry if ry > 0.0 else -ry
    out[off + QV_LFOOT_H] = root_z + o[p + _P + (G1_SK_ANKLE_L - 1) * 3 + 2]
    out[off + QV_RFOOT_H] = root_z + o[p + _P + (G1_SK_ANKLE_R - 1) * 3 + 2]
    # ROT is [tangent 3 | normal 3]: normal z is 1 upright, 0 face-down;
    # tangent y is the torso's yaw against the heading frame.
    out[off + QV_UPRIGHT] = o[p + _R + G1_SK_TORSO * 6 + 5]
    out[off + QV_TORSO_YAW] = o[p + _R + G1_SK_TORSO * 6 + 1]
    var vx = o[p + _V + 0]
    var vy = o[p + _V + 1]
    out[off + QV_SPEED_FWD] = vx
    out[off + QV_SPEED_LAT] = vy
    out[off + QV_SPEED] = sqrt(vx * vx + vy * vy)
    out[off + QV_YAW_RATE] = o[p + _W + 2]


struct G1Term(Copyable, Movable):
    """One predicate over one named quantity.

    `as_pct` resolves `lo`/`hi` against the POOL's own quantile of that
    quantity. A literal threshold nobody in the dataset satisfies gives a
    reward of zero everywhere, and `z_from_reward` still returns a unit-norm
    vector — a silent failure. Thresholds that should follow the data say so;
    thresholds that are physics (uprightness, a speed in m/s) stay literal,
    so that the effective sample size can answer whether the dataset contains
    the behaviour at all.
    """
    var q: Int
    var op: Int
    var lo: Float64
    var hi: Float64
    var as_pct: Bool

    def __init__(out self, q: Int, op: Int, lo: Float64, hi: Float64, as_pct: Bool):
        self.q = q
        self.op = op
        self.lo = lo
        self.hi = hi
        self.as_pct = as_pct


@always_inline
def g1_term_value(ref t: G1Term, lo: Float64, hi: Float64, x: Float64) -> Float64:
    """The HARD predicate — what builds the prompt."""
    if t.op == OP_GT:
        return 1.0 if x > lo else 0.0
    if t.op == OP_LT:
        return 1.0 if x < hi else 0.0
    if t.op == OP_BAND:
        return 1.0 if (x > lo and x < hi) else 0.0
    var a = x if x > 0.0 else -x
    return exp(-t.lo * a)


@always_inline
def g1_term_soft(ref t: G1Term, lo: Float64, hi: Float64, w: Float64, x: Float64) -> Float64:
    """The SMOOTH predicate — what a latent search optimises against.

    ⚠ CEM CANNOT CLIMB A HARD INDICATOR. A product of indicators is zero
    almost everywhere near a bad prompt, so every candidate scores 0, every
    elite set is arbitrary, and the search returns its own starting point
    while looking like it ran. `w` is a width taken from the quantity's own
    pool spread, so one constant works across metres, m/s and rad/s.

    The hard form stays the published number; this one only steers.
    """
    if t.op == OP_SOFT:
        var a = x if x > 0.0 else -x
        return exp(-t.lo * a)
    var ww = w if w > 1e-9 else 1e-9
    if t.op == OP_GT:
        return 1.0 / (1.0 + exp(-(x - lo) / ww))
    if t.op == OP_LT:
        return 1.0 / (1.0 + exp(-(hi - x) / ww))
    return (1.0 / (1.0 + exp(-(x - lo) / ww))) * (1.0 / (1.0 + exp(-(hi - x) / ww)))


def g1_term_str(ref t: G1Term, lo: Float64, hi: Float64) -> String:
    var n = g1_vocab_name(t.q)
    if t.op == OP_GT:
        return n + String(" > ") + String(lo)
    if t.op == OP_LT:
        return n + String(" < ") + String(hi)
    if t.op == OP_BAND:
        return String(lo) + String(" < ") + n + String(" < ") + String(hi)
    return String("exp(-") + String(t.lo) + String(" * |") + n + String("|)")


def g1_ess_rows(ref w: List[Float64], n: Int) -> Float64:
    """`(sum w)^2 / sum w^2` — the number of pool rows actually standing
    behind a prompt. A PRODUCT of predicates is an AND of sets and starves
    fast: §12.51's squat compound retains 15 rows of a 4096 pool and 232 of
    a 65 536 one. Print this before believing any compound."""
    var s1 = 0.0
    var s2 = 0.0
    for i in range(n):
        s1 += w[i]
        s2 += w[i] * w[i]
    if s2 <= 0.0:
        return 0.0
    return s1 * s1 / s2


def g1_quantile(ref col: List[Float64], q: Float64) -> Float64:
    var v = List[Float64](length=len(col), fill=0.0)
    for i in range(len(col)):
        v[i] = col[i]
    sort(v)
    var idx = Int(q * Float64(len(v) - 1) + 0.5)
    if idx < 0:
        idx = 0
    if idx >= len(v):
        idx = len(v) - 1
    return v[idx]
