# +--------------------------------------------------------------------------+ #
# | Reward specs at run time — the cache-miss path (§12.57-12.59, §10.2)
# +--------------------------------------------------------------------------+ #
"""A command the bank does not have, turned into a `z` and admitted or refused.

    var pool = G1Pool.load("g1_pool.bin")          # 67 MB, no ckpt, no store
    var q = g1_spec_questions(True)
    var a = jev.decide_text("instruction: " + heard, q)
    var terms = List[G1Term]()
    if g1_spec_from_answers(a, pool, bank, terms):
        var v = g1_spec_admit(pool, bank, terms, z)
        if v.ok: ...                               # a novel z to run
        elif v.nearest >= 0: ...                   # it IS a bank entry

## Where this sits

The bank's `command` choice is a CACHE LOOKUP keyed by the model, and
`P(none)` over it is a **calibrated cache miss** (§10.2). On a miss the
instruction gets a reward spec instead of a name: three budgeted slots naming
`(quantity, direction)` pairs, resolved against the pool's own quantiles.
Measured end to end at **14 of 16** with `(quantity, direction)` precision
1.00 (§12.58).

⚠ **THE BANK OWNS LEXICAL DISTINCTIONS AND THIS PATH DOES NOT** (§12.59).
`run` is a gait, not fast walking; `walk` and `run` are both "forward speed
high" and no magnitude question separates them — a `score` over ordered rungs
was measured and returns 1.0 for both "marche" and "cours". So this is not a
better bank, it is a different mechanism for a different job: novel
COMPOSITIONS, where the quantities are the content. Anything the bank has a
word for should never reach here, and `P(none)` is what keeps it out.

## The pool sidecar, and why it exists

`z = E_rho[r(s) B(s)]` needs `B` over the pool and nothing else — no actor,
no critic, no simulator. So the 65 536 encoded rows are cached to a file and
this whole path costs **neither the 640 MB checkpoint nor the 1.7 GB store**.
Any writer on the channel can compute a `z`. Built by
`examples/g1/bfm_zero_spec_probe.mojo --emit-pool`.

⚠ 65 536 ROWS, NOT FEWER. A product is an AND of sets: §12.51 measured the
squat compound retaining **15** effective rows at a 4096-row pool against 232
at 65 536. The sidecar is 67 MB because the pool cannot be shrunk.

## Two online gates, and they catch different things

- **ESS** — the effective row count behind the reward. Catches an EMPTY
  product: §12.57's `squat` band collapsed to a point and went to 0.
- **The angle probe** — against the bank's own baseline rows. Catches a
  VACUOUS product, which ESS cannot: §12.58's under-constrained spec retained
  **58 981 of 65 536** rows, so its ESS was near maximal and it meant nothing.

Both are matvecs over a resident matrix. Neither needs a rollout, which is
what makes them usable in a conversation — the CEM refinement that pays
37-49 % is 128 rollouts and stays offline.
"""

from std.math import acos, sqrt
from std.builtin.sort import sort

from noeira.core.bytes import string_from_bytes
from noeira.io.fileio import read_file_bytes, write_file_atomic
from noeira.ai.jev import JevQuestions, JevAnswers
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_reward_vocab import (
    G1_NVOC, G1Term, g1_vocab_name, g1_term_value, g1_ess_rows,
    OP_GT, OP_LT, OP_BAND, OP_SOFT,
    QV_BODY_H, QV_HEAD_H, QV_LHAND_H, QV_RHAND_H, QV_LHAND_LAT,
    QV_RHAND_LAT, QV_LFOOT_H, QV_RFOOT_H, QV_UPRIGHT, QV_SPEED_FWD,
    QV_SPEED_LAT, QV_SPEED, QV_YAW_RATE, QV_TORSO_YAW,
)

comptime G1_SPEC_D: Int = 256
comptime G1_SPEC_SLOTS: Int = 3
comptime G1_POOL_MAGIC: String = "G1POOL1"

# ── the admission thresholds, all measured ────────────────────────────────
# ⚠ 100 ROWS is §12.52's bank gate, reused: below it the compound is not
# describing a behaviour the dataset contains.
comptime G1_SPEC_MIN_ESS: Float64 = 100.0
# ⚠ 33 DEGREES IS THE BANK'S OWN CONFUSION RADIUS (§12.57). Its 20 `z` rows
# sit at a median 84 deg apart with a MINIMUM of 32.9 (`strafe_left` /
# `diagonal`), and the four tightest pairs are all semantically adjacent. So
# a novel `z` closer than this to an existing entry is not a new command: it
# is that entry, and the entry's CEM-refined row is the better answer.
# ⚠ RAISED FROM 33 AND JOINED BY A MARGIN — see `G1SpecVerdict.margin`. 33
# is the bank's own confusion radius and it does NOT transfer to
# spec-vs-bank: a coarse generated spec sits ~70 deg from the tuned compound
# it means, which is FURTHER than a genuinely novel spec sits from its
# incidental nearest neighbour. 90 is a loose upper bound; the margin does
# the discriminating.
comptime G1_SPEC_DUP_DEG: Float64 = 90.0
comptime G1_SPEC_DUP_MARGIN: Float64 = 12.0
# ⚠ AND A FLOOR AGAINST `stand`. A spec that constrains nothing collapses
# onto the scaffold, which is most of the pool — the vacuity ESS cannot see.
comptime G1_SPEC_MIN_DEG: Float64 = 20.0

# the level a word resolves to. ⚠ SLOT-DEPENDENT, and that is §12.58's
# measurement: `low` as a GOAL needs p10 ("recule" scored ESS 27407 and
# retrieved `stand` at p50), `low` as a SUPPRESSION needs p50 (p10 took
# "lève le bras droit" from ESS 233 to **0**, because the pool has almost no
# frames with one hand above p90 while the other is below p10). The bank had
# it right from §12.52: `RHAND_H > p90` AND `LHAND_H < p50`.
comptime G1_SPEC_HIGH_P: Float64 = 0.90
comptime G1_SPEC_GOAL_LOW_P: Float64 = 0.10
comptime G1_SPEC_SUPPRESS_P: Float64 = 0.50
# ⚠ A `low` GOAL ON A HEIGHT IS A BAND, NOT A ONE-SIDED THRESHOLD, and
# `g1_reward_vocab`'s own docstring called this before it was measured here:
# "ONE-SIDED THRESHOLDS ON A HEIGHT ARE A TRAP. `body_height < x` rewards
# every state below it, and the lowest states in LAFAN are lying down."
# Measured: "make yourself as short as you possibly can" resolved to
# `body_height < p10`, scored ESS 640, and landed **73.3 deg from `squat`** —
# it had found lying down. Motivo says the same of its own low tasks: "the
# agent is encouraged to keep the pelvis z-coordinate inside a predefined
# range".
#
# The floor is p02 rather than 0 so the two edges are DISTINCT by
# construction — §12.57's band collapse (`[0.61, 0.61]`, ESS 140 -> 0) is
# designed out here rather than gated against.
comptime G1_SPEC_LOW_FLOOR_P: Float64 = 0.02


def g1_spec_is_height(q: Int) -> Bool:
    """Quantities whose extreme low tail is a DIFFERENT BEHAVIOUR, not more
    of the same. A speed at p02 is fast backwards, which is what "backwards"
    means; a pelvis height at p02 is lying on the floor, which is not what
    "crouch" means."""
    return (
        q == QV_BODY_H or q == QV_HEAD_H or q == QV_LHAND_H
        or q == QV_RHAND_H or q == QV_LFOOT_H or q == QV_RFOOT_H
    )


struct G1Pool(Movable):
    """`B` over the state pool, plus the 14 quantities per row."""

    var n: Int
    var b: List[Float64]
    """`[n, 256]`, row-major."""
    var qv: List[Float64]
    """`[n, G1_NVOC]`, row-major — the RAW physical quantities."""
    var _sorted: List[Float64]
    """`[G1_NVOC, n]`, each quantity's column sorted, for `quantile`."""

    def __init__(out self, n: Int, var b: List[Float64], var qv: List[Float64]):
        self.n = n
        self.b = b^
        self.qv = qv^
        self._sorted = List[Float64](length=G1_NVOC * n, fill=0.0)
        var col = List[Float64](length=n, fill=0.0)
        for q in range(G1_NVOC):
            for i in range(n):
                col[i] = self.qv[i * G1_NVOC + q]
            sort(col)
            for i in range(n):
                self._sorted[q * n + i] = col[i]

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.b = move.b^
        self.qv = move.qv^
        self._sorted = move._sorted^

    def quantile(self, q: Int, p: Float64) -> Float64:
        var idx = Int(p * Float64(self.n - 1) + 0.5)
        if idx < 0:
            idx = 0
        if idx >= self.n:
            idx = self.n - 1
        return self._sorted[q * self.n + idx]

    @staticmethod
    def load(path: String) raises -> G1Pool:
        var raw = read_file_bytes(path)
        # header: "G1POOL1 <n> <d> <nvoc>\n", then float32 B then float32 qv
        var nl = 0
        while nl < len(raw) and raw[nl] != 10:
            nl += 1
        if nl >= len(raw):
            raise Error("g1 pool: no header in " + path)
        var hdr = List[UInt8](capacity=nl)
        for i in range(nl):
            hdr.append(raw[i])
        var htxt = string_from_bytes(hdr)
        # ⚠ THE HEADER IS SPACE-PADDED TO A 4-BYTE BOUNDARY BY THE WRITER,
        # so `split(" ")` returns trailing EMPTY tokens — the first version
        # of this checked `len(p) != 4` and refused its own output. Collect
        # the non-empty tokens instead of counting the split.
        var raw_p = htxt.split(" ")
        var p = List[String]()
        for i in range(len(raw_p)):
            var tok = String(raw_p[i])
            if tok.byte_length() > 0:
                p.append(tok)
        if len(p) != 4 or p[0] != G1_POOL_MAGIC:
            raise Error("g1 pool: bad header '" + htxt + "' in " + path)
        var n = atol(p[1])
        var d = atol(p[2])
        var nv = atol(p[3])
        if d != G1_SPEC_D or nv != G1_NVOC:
            raise Error(
                "g1 pool: built for D=" + String(d) + " nvoc=" + String(nv)
                + ", this binary needs D=" + String(G1_SPEC_D) + " nvoc="
                + String(G1_NVOC)
            )
        var want = nl + 1 + (n * d + n * nv) * 4
        if len(raw) != want:
            raise Error(
                "g1 pool: " + path + " is " + String(len(raw))
                + " bytes, expected " + String(want)
            )
        var off = nl + 1
        var b = List[Float64](length=n * d, fill=0.0)
        var qv = List[Float64](length=n * nv, fill=0.0)
        var ptr = raw.unsafe_ptr().bitcast[Float32]()
        var base = off // 4
        # ⚠ the header is padded to a 4-byte boundary by the writer, so this
        # bitcast is aligned — `g1_pool_save` pads and this checks it.
        if off % 4 != 0:
            raise Error("g1 pool: payload not 4-byte aligned in " + path)
        for i in range(n * d):
            b[i] = Float64(ptr[base + i])
        for i in range(n * nv):
            qv[i] = Float64(ptr[base + n * d + i])
        return G1Pool(n, b^, qv^)


def g1_pool_save(
    path: String, n: Int, ref b: List[Float64], ref qv: List[Float64]
) raises:
    """Write the sidecar. `b` is `[n, 256]` and `qv` is `[n, G1_NVOC]`.

    ⚠ THE HEADER IS PADDED TO FOUR BYTES so the reader can bitcast the
    payload instead of reassembling it a byte at a time — 16.7 M floats
    through a byte loop is a startup cost nobody would accept, and an
    unaligned bitcast is undefined rather than slow.
    """
    var hdr = String(G1_POOL_MAGIC) + " " + String(n) + " " + String(
        G1_SPEC_D) + " " + String(G1_NVOC)
    while (hdr.byte_length() + 1) % 4 != 0:
        hdr += " "
    hdr += "\n"
    var out = List[UInt8](capacity=hdr.byte_length() + (n * G1_SPEC_D + n * G1_NVOC) * 4)
    for i in range(hdr.byte_length()):
        out.append(hdr.as_bytes()[i])
    for i in range(n * G1_SPEC_D):
        var v = Float32(b[i])
        var by = UnsafePointer(to=v).bitcast[UInt8]()
        for k in range(4):
            out.append(by[unsafe_offset=k])
    for i in range(n * G1_NVOC):
        var v = Float32(qv[i])
        var by = UnsafePointer(to=v).bitcast[UInt8]()
        for k in range(4):
            out.append(by[unsafe_offset=k])
    write_file_atomic(path, out)


# ── the spec -> z path ────────────────────────────────────────────────────


def g1_spec_z(
    ref pool: G1Pool, ref terms: List[G1Term], mut z: List[Float64], off: Int
) raises -> Float64:
    """The compound over the pool into `z[off:off+256]`. Returns the ESS.

    `z` comes back on the radius-sqrt(D) sphere, as every prompt must.
    """
    var acc = List[Float64](length=G1_SPEC_D, fill=0.0)
    var prod = List[Float64](length=pool.n, fill=0.0)
    var wsum = 0.0
    for i in range(pool.n):
        var r = 1.0
        for t in range(len(terms)):
            var x = pool.qv[i * G1_NVOC + terms[t].q]
            r *= g1_term_value(terms[t], terms[t].lo, terms[t].hi, x)
            if r == 0.0:
                break
        prod[i] = r
        if r != 0.0:
            wsum += r
            for k in range(G1_SPEC_D):
                acc[k] += r * pool.b[i * G1_SPEC_D + k]
    var ess = g1_ess_rows(prod, pool.n)
    if wsum <= 0.0:
        for k in range(G1_SPEC_D):
            z[off + k] = 0.0
        return ess
    var nrm = 0.0
    for k in range(G1_SPEC_D):
        acc[k] /= wsum
        nrm += acc[k] * acc[k]
    nrm = sqrt(nrm)
    var scale = sqrt(Float64(G1_SPEC_D)) / nrm if nrm > 1e-12 else 0.0
    for k in range(G1_SPEC_D):
        z[off + k] = acc[k] * scale
    return ess


def g1_spec_angle(
    ref a: List[Float64], ao: Int, ref b: List[Float64], bo: Int
) -> Float64:
    """Degrees between two `z` rows. ⚠ TWO LISTS — Mojo rejects two aliasing
    `ref` arguments, so a same-list comparison needs its own call site."""
    var d = 0.0
    var na = 0.0
    var nb = 0.0
    for k in range(G1_SPEC_D):
        d += a[ao + k] * b[bo + k]
        na += a[ao + k] * a[ao + k]
        nb += b[bo + k] * b[bo + k]
    if na <= 1e-18 or nb <= 1e-18:
        return 180.0
    var c = d / (sqrt(na) * sqrt(nb))
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0
    return acos(c) * 57.29577951308232


def g1_spec_bank_baseline(
    ref pool: G1Pool, ref bank: G1CommandBank, mut zb: List[Float64]
) raises:
    """Each bank entry's BASELINE `z`, rebuilt from its own recorded terms.

    ⚠ NOT THE SHIPPED ROW, and §12.57 measured why. CEM moved 11 of the 20
    entries by 23.7-27.9 deg and left 9 at exactly 0.0, so comparing a fresh
    baseline candidate against the shipped rows handicaps it by an uneven
    amount: `arms_wide` sits 0.6 deg from its own baseline and still
    retrieves `both_hands_up` from the shipped bank. Duplicate detection
    compares like with like; what gets RUN is still the refined row.

    Needs no change to the bank file — the terms are already in it.
    """
    for c in range(bank.count()):
        var ts = List[G1Term]()
        var s0 = bank.t_start[c]
        for j in range(bank.t_count[c]):
            var k = s0 + j
            ts.append(G1Term(bank.t_q[k], bank.t_op[k], bank.t_lo[k],
                             bank.t_hi[k], False))
        _ = g1_spec_z(pool, ts, zb, c * G1_SPEC_D)


def _dir_of(op: Int, lo: Float64, hi: Float64, med: Float64) -> String:
    """A term's DIRECTION as a word.

    ⚠ A BAND'S DIRECTION IS NOT IN ITS OPERATOR (§12.58). `squat` is
    `body_height BAND [0.40, 0.62]`, which means low; `run` is
    `speed_forward BAND [1.50, 3.00]`, which means high. Both are OP_BAND and
    only the pool's median separates them.
    """
    if op == OP_GT:
        return String("high")
    if op == OP_LT:
        return String("low")
    if op == OP_BAND:
        return String("high") if (lo + hi) * 0.5 > med else String("low")
    return String("soft")


def g1_spec_goal_key(
    ref pool: G1Pool, ref terms: List[G1Term], n_scaffold: Int
) raises -> String:
    """A spec's goal terms as a canonical `q:dir` key, quantity order."""
    var out = String("")
    for v in range(G1_NVOC):
        for t in range(n_scaffold, len(terms)):
            if terms[t].q != v:
                continue
            if out.byte_length() > 0:
                out += String(",")
            out += g1_vocab_name(v) + String(":") + _dir_of(
                terms[t].op, terms[t].lo, terms[t].hi, pool.quantile(v, 0.50))
    return out


def g1_bank_goal_key(
    ref pool: G1Pool, ref bank: G1CommandBank, i: Int
) raises -> String:
    """The same key for a bank entry, from ITS OWN recorded goal terms."""
    var out = String("")
    var s0 = bank.t_start[i]
    for v in range(G1_NVOC):
        for j in range(bank.t_count[i]):
            var k = s0 + j
            if not bank.t_goal[k] or bank.t_op[k] == OP_SOFT:
                continue
            if bank.t_q[k] != v:
                continue
            if out.byte_length() > 0:
                out += String(",")
            out += g1_vocab_name(v) + String(":") + _dir_of(
                bank.t_op[k], bank.t_lo[k], bank.t_hi[k],
                pool.quantile(v, 0.50))
    return out


@fieldwise_init
struct G1SpecVerdict(Copyable, Movable):
    var ok: Bool
    """True when `z` is a novel command worth running."""
    var ess: Float64
    var angle: Float64
    """Degrees to the nearest bank baseline."""
    var nearest: Int
    """The nearest bank entry, or -1. When `ok` is False and this is >= 0,
    the spec IS that entry and the caller should run it from the bank."""
    var second: Int
    """The runner-up entry, or -1."""
    var margin: Float64
    """Degrees from the nearest to the runner-up.

    ⚠ THE MARGIN, NOT THE ANGLE, IS WHAT A DUPLICATE TEST NEEDS — and the
    absolute angle was measured failing at it. "make yourself as short as you
    possibly can" is `squat` and sits **70.7 deg** from it; "lift your right
    foot off the ground" is genuinely novel and sits **56.9 deg** from
    `look_right`, which it has nothing to do with. The duplicate is FURTHER
    than the novelty, because 20 entries in a 256-dimensional space leave a
    nearest neighbour for everything. A generated spec is a coarse version of
    a tuned compound, so it never lands inside the bank's own 33 deg
    confusion radius (§12.57) — that number bounds bank-to-bank distance and
    does not transfer."""
    var reason: String
    """Empty when admitted; otherwise why not, in words for a HUD."""


def g1_spec_admit(
    ref pool: G1Pool,
    ref bank: G1CommandBank,
    ref zb: List[Float64],
    ref terms: List[G1Term],
    mut z: List[Float64],
    n_scaffold: Int,
    min_ess: Float64 = G1_SPEC_MIN_ESS,
    dup_deg: Float64 = G1_SPEC_DUP_DEG,
    dup_margin: Float64 = G1_SPEC_DUP_MARGIN,
    min_deg: Float64 = G1_SPEC_MIN_DEG,
) raises -> G1SpecVerdict:
    """Both online gates, in the order that makes the reason useful."""
    var ess = g1_spec_z(pool, terms, z, 0)
    var best = 1e9
    var bi = -1
    var second = 1e9
    var si = -1
    for c in range(bank.count()):
        var a = g1_spec_angle(z, 0, zb, c * G1_SPEC_D)
        if a < best:
            second = best
            si = bi
            best = a
            bi = c
        elif a < second:
            second = a
            si = c
    var margin = second - best
    if len(terms) == 0:
        return G1SpecVerdict(False, ess, best, -1, si, margin,
                             String("the spec is empty"))
    # ⚠ ESS FIRST, because an empty product makes the angle meaningless —
    # `z` is all zeros and `g1_spec_angle` returns its 180 sentinel.
    if ess < min_ess:
        return G1SpecVerdict(
            False, ess, best, -1, si, margin,
            String("no support in the data (") + String(Int(ess))
            + String(" rows) — the terms contradict each other"),
        )
    # ⚠ THE DUPLICATE CASE IS A SUCCESS WITH A DIFFERENT HANDLER, not a
    # refusal: the instruction named something the bank already holds, and
    # the bank's row went through four gates and CEM that this one has not.
    # ⚠ THE DUPLICATE TEST IS ON THE GOAL TERMS, NOT ON `z` — and TWO
    # angle-based criteria were measured failing at it first.
    #
    # The absolute angle fails because a coarse generated spec sits ~70 deg
    # from the tuned compound it MEANS while a genuinely novel spec sits ~57
    # from an entry it has nothing to do with: "make yourself as short as you
    # possibly can" is `squat` at 70.7, "lift your right foot off the ground"
    # is novel at 56.9 from `look_right`. The duplicate is FURTHER than the
    # novelty. 20 entries in a 256-dimensional space leave a nearest
    # neighbour for everything.
    #
    # The MARGIN fails too: `right_foot_height high` had a 24.0 deg margin on
    # `look_right` and `left_hand_lateral low` a 14.0 on `left_hand_up` —
    # both would have been claimed as duplicates of entries they share no
    # quantity with.
    #
    # What actually separates them is the TERMS. `body_height low` and
    # `squat`'s `body_height BAND [0.40, 0.62]` are the same quantity in the
    # same direction; `right_foot_height high` and `look_right`'s `torso_yaw
    # low` are not. So the key decides, and the angle is only a TIE-BREAK
    # among entries the key already matched — which is exactly the
    # `walk`/`run` and `squat`/`crouch` collision of §12.59, where two
    # entries share one key and the bank is the more expressive mechanism.
    var key = g1_spec_goal_key(pool, terms, n_scaffold)
    var kbest = -1
    var kang = 1e9
    for c in range(bank.count()):
        if g1_bank_goal_key(pool, bank, c) != key:
            continue
        var a = g1_spec_angle(z, 0, zb, c * G1_SPEC_D)
        if a < kang:
            kang = a
            kbest = c
    if kbest >= 0:
        return G1SpecVerdict(
            False, ess, best, kbest, si, margin,
            String("already in the bank as `") + bank.name_at(kbest)
            + String("` — same goal terms (") + key + String(")"),
        )
    if False:
        return G1SpecVerdict(
            False, ess, best, bi, si, margin,
            String("already in the bank as `") + bank.name_at(bi)
            + String("` (margin ") + _round2(margin) + String(" deg)"),
        )
    var stand = bank.find(String("stand"))
    if stand >= 0:
        var a_stand = g1_spec_angle(z, 0, zb, stand * G1_SPEC_D)
        if a_stand < min_deg:
            return G1SpecVerdict(
                False, ess, best, -1, si, margin,
                String("says nothing — it collapses onto standing"),
            )
    return G1SpecVerdict(True, ess, best, bi, si, margin, String(""))


# ── the questions, and ONE copy of the resolver ───────────────────────────


def _round2(v: Float64) -> String:
    var h = Int(v * 100.0 + 0.5)
    var f = String(h % 100)
    if h % 100 < 10:
        f = String("0") + f
    return String(h // 100) + String(".") + f


def g1_spec_dir_option(q: Int, hi: Bool) -> String:
    return g1_vocab_name(q) + (String("_high") if hi else String("_low"))


def g1_spec_dir_desc(q: Int, hi: Bool) -> String:
    """What a `(quantity, direction)` pair MEANS, including its sign.

    ⚠ THE SIGN LIVES HERE, IN THE OPTION, and §12.58 measured why it cannot
    live anywhere else. An earlier form asked "which quantity?" with the
    conventions on those options and then a SEPARATE "high or low?" whose
    options were bare — and `step to your left` and `step to your right`
    came back as BYTE-IDENTICAL specs. Jev answers every question
    independently against one state, so a convention attached to one
    question's options never reaches another's.
    """
    if q == QV_SPEED_LAT:
        return String("stepping to the robot's LEFT") if hi else String(
            "stepping to the robot's RIGHT")
    if q == QV_YAW_RATE:
        return String("turning on the spot to the LEFT") if hi else String(
            "turning on the spot to the RIGHT")
    if q == QV_TORSO_YAW:
        return String(
            "torso twisted to the LEFT, as when looking left without turning"
        ) if hi else String(
            "torso twisted to the RIGHT, as when looking right without turning"
        )
    if q == QV_SPEED_FWD:
        return String("moving FORWARDS over the ground") if hi else String(
            "moving BACKWARDS over the ground")
    if q == QV_SPEED:
        return String("moving fast, in any direction") if hi else String(
            "barely moving at all — standing still")
    if q == QV_BODY_H:
        return String("pelvis high — standing at full height") if hi else \
            String("pelvis LOW — crouched, squatting, close to the ground")
    if q == QV_HEAD_H:
        return String("head high — upright at full height") if hi else \
            String("head LOW — bent or crouched down")
    if q == QV_UPRIGHT:
        return String("torso vertical") if hi else String(
            "torso far from vertical — leaning, pitched or fallen")
    if q == QV_LHAND_H:
        return String("LEFT hand raised high") if hi else String(
            "LEFT hand down, low by the side")
    if q == QV_RHAND_H:
        return String("RIGHT hand raised high") if hi else String(
            "RIGHT hand down, low by the side")
    if q == QV_LHAND_LAT:
        return String("LEFT hand held out wide, away from the body") if hi \
            else String("LEFT hand held in, close to the body")
    if q == QV_RHAND_LAT:
        return String("RIGHT hand held out wide, away from the body") if hi \
            else String("RIGHT hand held in, close to the body")
    if q == QV_LFOOT_H:
        return String("LEFT foot lifted off the ground") if hi else String(
            "LEFT foot flat on the ground")
    return String("RIGHT foot lifted off the ground") if hi else String(
        "RIGHT foot flat on the ground")


def g1_spec_cat_donor(i: Int) -> String:
    """The bank entry whose NON-goal terms are category `i`'s scaffold.

    ⚠ THE SCAFFOLD COMES FROM THE BANK, NEVER FROM A COPY OF IT. The four
    scaffolds are `bfm_zero_bank_build.mojo`'s and they contradict each other
    on purpose — `_scaffold_low` has no head-height term because a squat's
    head is below it, `_scaffold_move` has no uprightness term because
    running pitches the torso forward and `run` scored a hard 0.000 with it
    in place. A second copy of those rules would drift, and §12.51 is the
    section that drift created.
    """
    if i == 0:
        return String("right_hand_up")     # _scaffold_stand
    if i == 1:
        return String("squat")             # _scaffold_low
    if i == 2:
        return String("walk")              # _scaffold_move
    return String("spin_left")             # _scaffold_rotate


def g1_spec_cat_name(i: Int) -> String:
    if i == 0:
        return String("standing")
    if i == 1:
        return String("low")
    if i == 2:
        return String("moving")
    return String("turning")


def _slot_names(base: List[String], s: Int) -> List[String]:
    var out = base.copy()
    out.append(String("none"))
    return out^


def _slot_descs(base: List[String], s: Int) -> List[String]:
    """⚠ `none` MEANS SOMETHING DIFFERENT ON SLOT 1, and the first version
    shipped the wrong word there. Its description was "no FURTHER quantity is
    constrained", which is nonsense on the first slot and reads as an
    invitation — §12.58's defect (1), which is itself §12.54's defect
    relearned. On slot 1 `none` is the honest abstention for an instruction
    the vocabulary cannot express at all, and it has to read as the
    exception."""
    var out = base.copy()
    if s == 0:
        out.append(String(
            "NOTHING in this list describes what was asked — pick this ONLY"
            " when no quantity above is even approximately relevant"
        ))
    else:
        out.append(String("no further quantity is constrained"))
    return out^


def g1_spec_prompt(instruction: String) -> String:
    """The state the spec questions are asked against — THE INSTRUCTION ALONE.

    ⚠ NOT `g1_command_state`, AND THIS IS MEASURED. The demo first asked the
    sketch against the full conversational state, which carries `doing` — and
    the posture CATEGORY is answered from that state, so the scaffold came
    out of what the robot happened to be doing rather than what was asked.
    "Lève ton pied droit" twice in one session:

        doing: walk   -> category `moving`   -> ESS 5089
        doing: stand  -> category `standing` -> ESS  301

    Same instruction, same goal terms, same `describe` output, same duplicate
    key — 17x the support and a different behaviour. The looser one is
    §12.51's trap exactly: `_scaffold_move` constrains only head height, so
    the product retains walking frames and the conditional mean comes back as
    walking. **An implicit scaffold change is invisible**, which is the defect
    class this whole track keeps paying for.

    A reward spec describes a TARGET BEHAVIOUR, not a transition into one, so
    it must be a function of the instruction and nothing else. `doing` belongs
    to the bank lookup, where it resolves "plus vite" (§12.56); it has no
    business choosing a scaffold.

    ⚠ It is also what the §12.58 measurement used — `decide_text("instruction:
    " + ...)`. The demo had diverged from the number it was built on.
    """
    return String("instruction: ") + instruction


def g1_spec_questions(warn: Bool = True) raises -> JevQuestions:
    """The reward sketch: a posture category plus three budgeted slots.

    ⚠ THREE SLOTS, NOT ONE QUESTION PER QUANTITY. §12.58 measured the wide
    form: asked about all 14 independently the model constrained 8 to 14 of
    them per instruction, because each question is answered in isolation and
    asking about `torso_yaw` at all implies it might matter. Precision 0.13.
    A slot list forces a budget, and it is §12.55's chain pattern — the one
    that already works for `command`/`command2`/`command3`.
    """
    var cats = List[String]()
    var cdesc = List[String]()
    cats.append(String("standing"))
    cdesc.append(String("upright and still, at full height"))
    cats.append(String("low"))
    cdesc.append(String("upright and still, but crouched or close to the ground"))
    cats.append(String("moving"))
    cdesc.append(String("travelling across the ground"))
    cats.append(String("turning"))
    cdesc.append(String("rotating on the spot"))

    var qnames = List[String]()
    var qdescs = List[String]()
    for v in range(G1_NVOC):
        qnames.append(g1_spec_dir_option(v, True))
        qdescs.append(g1_spec_dir_desc(v, True))
        qnames.append(g1_spec_dir_option(v, False))
        qdescs.append(g1_spec_dir_desc(v, False))

    var q = JevQuestions()
    q.choice(
        String("spec_cat"),
        String(
            "A humanoid robot is given this instruction. Which of these best"
            " describes the POSTURE the robot should be in while carrying it"
            " out?"
        ),
        cats, cdesc,
    )
    var ordinals = List[String]()
    ordinals.append(String("FIRST and most important"))
    ordinals.append(String("SECOND"))
    ordinals.append(String("THIRD"))
    for s in range(G1_SPEC_SLOTS):
        q.choice(
            String("spec_q") + String(s),
            String(
                "A humanoid robot is carrying out this instruction. The"
                " robot's body is measured by the quantities listed. Which of"
                " these is the " + ordinals[s] + " thing the instruction asks"
                " for?"
            )
            + (
                # ⚠ ONE SENTENCE, WORTH 4 RETRIEVALS AND ALL THE REMAINING
                # PRECISION (§12.58). It states how `z = E_rho[r B]` behaves
                # — a fact about the SYSTEM, not the answer to any
                # instruction — and with it the model reproduces §12.51's
                # both-wrists rule unprompted, landing on the gated
                # compound's exact ESS.
                String(
                    " ⚠ Any quantity you do NOT name comes back as the"
                    " robot's average behaviour, so name the ones that must"
                    " NOT change as well: if one hand goes up, the other must"
                    " be named as staying low."
                ) if warn else String("")
            )
            + (
                String(" Pick `none` when the instruction constrains nothing"
                       " further.") if s > 0 else String("")
            ),
            _slot_names(qnames, s), _slot_descs(qdescs, s),
        )
    return q^


def g1_spec_from_answers(
    ref ans: JevAnswers,
    ref pool: G1Pool,
    ref bank: G1CommandBank,
    mut terms: List[G1Term],
) raises -> Int:
    """The model's answer into a term list.

    Returns **the number of SCAFFOLD terms** it appended first, or **-1** when
    the model named no quantity at all. ⚠ The caller needs that count: the
    scaffold lands ahead of the goals and a caller that guessed would
    describe boilerplate back to the user as though it were the instruction.

    The category's scaffold comes first, then each named slot, resolved
    against the pool's quantiles by the slot-dependent ladder.
    """
    var pick = ans.choice(String("spec_cat"))
    var ci = 0
    for j in range(4):
        if g1_spec_cat_name(j) == pick:
            ci = j
    var donor = bank.find(g1_spec_cat_donor(ci))
    if donor < 0:
        raise Error(
            "g1 spec: the bank has no `" + g1_spec_cat_donor(ci)
            + "`, which is where category `" + g1_spec_cat_name(ci)
            + "` takes its scaffold from"
        )
    # ⚠ SCAFFOLD AND GOALS ARE BUILT SEPARATELY and concatenated, so a goal
    # can DISPLACE a scaffold term without any in-place shuffling.
    var scaf = List[G1Term]()
    var d0 = bank.t_start[donor]
    for j in range(bank.t_count[donor]):
        var k = d0 + j
        if bank.t_goal[k]:
            continue
        scaf.append(G1Term(bank.t_q[k], bank.t_op[k], bank.t_lo[k],
                           bank.t_hi[k], False))

    var goals = List[G1Term]()
    var named = 0
    for s in range(G1_SPEC_SLOTS):
        var p = ans.choice(String("spec_q") + String(s))
        # ⚠ SAY WHAT EACH SLOT RETURNED. "the model named no quantity" is not
        # a diagnosis — it cannot distinguish an instruction the vocabulary
        # genuinely cannot express from a wording defect in the question,
        # which is exactly the pair that cost §12.58 four iterations. It is
        # also what found the scaffold-collision bug below in one run.
        print("    slot", s, "->", p, _round2(
            ans.probability(String("spec_q") + String(s), p)))
        var vi = -1
        var hi = True
        for v in range(G1_NVOC):
            if g1_spec_dir_option(v, True) == p:
                vi = v
                hi = True
            elif g1_spec_dir_option(v, False) == p:
                vi = v
                hi = False
        # ⚠ STOP AT THE FIRST `none`, exactly as `g1_decide_chain` does: a
        # third slot without a second is a misread, not a spec.
        if vi < 0:
            break
        var nt = G1Term(vi, OP_GT, pool.quantile(vi, G1_SPEC_HIGH_P), 0.0, False)
        if not hi:
            # ⚠ goal vs suppression is by the MODEL's slot, not by position
            # in the list: `low` first is a goal (p10), later is suppression
            # (p50). §12.58 measured both, and p10 on a suppression term took
            # "lève le bras droit" from ESS 233 to 0.
            var pq = G1_SPEC_GOAL_LOW_P if named == 0 else G1_SPEC_SUPPRESS_P
            if named == 0 and g1_spec_is_height(vi):
                # a BAND, per the note beside G1_SPEC_LOW_FLOOR_P
                nt = G1Term(
                    vi, OP_BAND, pool.quantile(vi, G1_SPEC_LOW_FLOOR_P),
                    pool.quantile(vi, pq), False,
                )
            else:
                nt = G1Term(vi, OP_LT, 0.0, pool.quantile(vi, pq), False)
        # ⚠ A GOAL OVERRIDES A SCAFFOLD TERM ON THE SAME QUANTITY — it is NOT
        # a duplicate to drop. The first version dropped it, and "lean your
        # chest forward" was REFUSED as "named no quantity" while the trace
        # showed slot 0 returning `upright_low` at 0.98: the standing
        # scaffold carries `upright > 0.9`, so the one term that expressed
        # the instruction was discarded for colliding with the term it
        # needed to contradict.
        #
        # That is §12.51 from a new direction. The bank avoids it by CHOOSING
        # a scaffold that does not contradict its goal — `_scaffold_move` has
        # no uprightness term precisely because running pitches the torso
        # forward. A generated spec cannot choose, so the goal wins.
        var rep = -1
        for t in range(len(scaf)):
            if scaf[t].q == vi:
                rep = t
        if rep >= 0:
            var kept = List[G1Term]()
            for t in range(len(scaf)):
                if t != rep:
                    kept.append(scaf[t].copy())
            scaf = kept^
        var rep2 = -1
        for t in range(len(goals)):
            if goals[t].q == vi:
                rep2 = t
        if rep2 >= 0:
            goals[rep2] = nt^         # a later slot restates it: last wins
        else:
            goals.append(nt^)
            named += 1

    if named == 0:
        return -1
    var n_scaffold = len(scaf)
    for t in range(len(scaf)):
        terms.append(scaf[t].copy())
    for t in range(len(goals)):
        terms.append(goals[t].copy())
    return n_scaffold


def g1_spec_describe(ref terms: List[G1Term], n_scaffold: Int) -> String:
    """The GOAL terms in words, for a HUD or an ack. The scaffold is left
    out: it is the same boilerplate on most specs and says nothing about
    what was asked."""
    var out = String("")
    for t in range(n_scaffold, len(terms)):
        if out.byte_length() > 0:
            out += String(", ")
        out += g1_vocab_name(terms[t].q)
        out += String(" high") if terms[t].op == OP_GT else String(" low")
        if terms[t].op == OP_BAND:
            out += String(" (band)")
    return out
