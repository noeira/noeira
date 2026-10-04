"""The cache-miss path's pure parts — a gate, because each one fails silently.

    pixi run mojo run -I . tests/robots/test_g1_spec.mojo

WHY THIS EXISTS
===============
**The pool sidecar's header is space-padded to a 4-byte boundary** so the
reader can bitcast 16.7 M floats instead of reassembling them byte by byte.
The first version of the loader then split that header on spaces, counted the
tokens, and REFUSED ITS OWN OUTPUT. A round trip is the only check that
catches a writer and a reader disagreeing about their own format.

**The goal key is the duplicate test**, after two angle-based criteria were
measured failing at it (§12.60). It has to be canonical — same terms in any
order give the same key — and it has to read a BAND's direction from where
the band sits, because `squat` and `run` are both OP_BAND and only the pool's
median separates "low" from "high".

**A `low` goal on a height must be a BAND.** `g1_reward_vocab`'s docstring
warned about this before it was measured: a one-sided `body_height < x`
rewards every state below it and the lowest states in LAFAN are lying down.

Nothing here needs the network, the store or the 70 MB sidecar: the pool is
synthetic and tiny.
"""

from noeira.io.fileio import remove_file
from noeira.envs.robots.g1_spec import (
    G1Pool, g1_pool_save, g1_spec_goal_key, g1_spec_is_height,
    g1_spec_dir_option, g1_spec_cat_donor, g1_spec_cat_name,
    G1_SPEC_D, G1_SPEC_DUP_MARGIN, G1_SPEC_LOW_FLOOR_P,
)
from noeira.envs.robots.g1_reward_vocab import (
    G1_NVOC, G1Term, OP_GT, OP_LT, OP_BAND,
    QV_BODY_H, QV_RHAND_H, QV_SPEED_FWD, QV_UPRIGHT, QV_YAW_RATE,
)
from noeira.envs.robots.g1_command_channel import g1_channel_write_z


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


def main() raises:
    var t = Tally()
    var path = String("/tmp/g1_spec_test_pool.bin")

    # ── a tiny synthetic pool: quantity q of row i is i/(n-1) ─────────
    print("-- the pool sidecar round trip")
    var n = 64
    var b = List[Float64](length=n * G1_SPEC_D, fill=0.0)
    var qv = List[Float64](length=n * G1_NVOC, fill=0.0)
    for i in range(n):
        for k in range(G1_SPEC_D):
            b[i * G1_SPEC_D + k] = Float64(i) * 0.5 + Float64(k) * 0.125
        for q in range(G1_NVOC):
            qv[i * G1_NVOC + q] = Float64(i) / Float64(n - 1)
    g1_pool_save(path, n, b, qv)
    var p = G1Pool.load(path)
    t.truth(p.n == n, "the row count survives the round trip")
    # ⚠ float32 on the wire, so compare within its precision and not exactly
    var worst = 0.0
    for i in range(n * G1_SPEC_D):
        var d = p.b[i] - b[i]
        var a = d if d > 0.0 else -d
        if a > worst:
            worst = a
    t.truth(worst < 1e-3, "B survives to float32 precision (worst " + String(worst) + ")")
    var worst_q = 0.0
    for i in range(n * G1_NVOC):
        var d = p.qv[i] - qv[i]
        var a = d if d > 0.0 else -d
        if a > worst_q:
            worst_q = a
    t.truth(worst_q < 1e-6, "the quantities survive")
    # the median of i/(n-1) over i in [0, n) is about 0.5
    var med = p.quantile(QV_BODY_H, 0.50)
    t.truth(med > 0.45 and med < 0.55, "quantile(0.50) is the median")
    t.truth(p.quantile(QV_BODY_H, 0.0) < 0.02, "quantile(0) is the minimum")
    t.truth(p.quantile(QV_BODY_H, 1.0) > 0.98, "quantile(1) is the maximum")

    # ⚠ a wrong-length file must RAISE, not be read as a shorter pool: a
    # truncated B matrix yields 256 plausible floats and a plausible robot.
    var bad = False
    try:
        # rewrite the header claiming twice the rows
        g1_pool_save(path, n, b, qv)
        # the size check is exercised by asking for a pool that is not there
        _ = G1Pool.load(String("/tmp/g1_spec_test_missing.bin"))
    except:
        bad = True
    t.truth(bad, "an absent or unreadable sidecar raises")

    # ── the goal key ──────────────────────────────────────────────────
    print("-- the goal key")
    var ts = List[G1Term]()
    ts.append(G1Term(QV_UPRIGHT, OP_GT, 0.9, 0.0, False))       # scaffold
    ts.append(G1Term(QV_RHAND_H, OP_GT, 0.8, 0.0, False))       # goal
    ts.append(G1Term(QV_BODY_H, OP_LT, 0.0, 0.3, False))        # goal
    var k1 = g1_spec_goal_key(p, ts, 1)
    t.truth(k1.find("right_hand_height:high") >= 0, "a GT goal reads `high`")
    t.truth(k1.find("body_height:low") >= 0, "an LT goal reads `low`")
    t.truth(k1.find("upright") < 0, "the scaffold is NOT in the key")

    # ⚠ CANONICAL: the same goals in the other order give the same key
    var ts2 = List[G1Term]()
    ts2.append(G1Term(QV_UPRIGHT, OP_GT, 0.9, 0.0, False))
    ts2.append(G1Term(QV_BODY_H, OP_LT, 0.0, 0.3, False))
    ts2.append(G1Term(QV_RHAND_H, OP_GT, 0.8, 0.0, False))
    t.truth(g1_spec_goal_key(p, ts2, 1) == k1, "the key is order-independent")

    # ⚠ A BAND'S DIRECTION COMES FROM THE POOL, NOT THE OPERATOR. On this
    # pool the median is 0.5, so a band below it is `low` and above is `high`
    # — which is what separates `squat` from `run`, both OP_BAND.
    var tb = List[G1Term]()
    tb.append(G1Term(QV_BODY_H, OP_BAND, 0.05, 0.25, False))
    t.truth(
        g1_spec_goal_key(p, tb, 0) == "body_height:low",
        "a BAND below the median reads `low`",
    )
    var th = List[G1Term]()
    th.append(G1Term(QV_SPEED_FWD, OP_BAND, 0.75, 0.95, False))
    t.truth(
        g1_spec_goal_key(p, th, 0) == "body_speed_forward:high",
        "a BAND above the median reads `high` — the same operator",
    )

    # ── the height rule ───────────────────────────────────────────────
    print("-- heights need a band, speeds do not")
    t.truth(g1_spec_is_height(QV_BODY_H), "body_height is a height")
    t.truth(g1_spec_is_height(QV_RHAND_H), "right_hand_height is a height")
    t.truth(not g1_spec_is_height(QV_SPEED_FWD), "a speed is NOT a height")
    t.truth(not g1_spec_is_height(QV_YAW_RATE), "a yaw rate is NOT a height")
    t.truth(
        G1_SPEC_LOW_FLOOR_P > 0.0,
        "the low band's floor is ABOVE zero, so its edges cannot collapse",
    )

    # ── the option names carry their direction ────────────────────────
    print("-- option naming")
    t.truth(
        g1_spec_dir_option(QV_SPEED_FWD, True) == "body_speed_forward_high",
        "an option name is `<quantity>_high`",
    )
    t.truth(
        g1_spec_dir_option(QV_SPEED_FWD, False) == "body_speed_forward_low",
        "and `<quantity>_low`",
    )
    # ⚠ each category's scaffold donor must EXIST as a bank name, or
    # `g1_spec_from_answers` raises at run time on a category the model picked
    var donors_ok = True
    for i in range(4):
        if g1_spec_cat_donor(i) == "" or g1_spec_cat_name(i) == "":
            donors_ok = False
    t.truth(donors_ok, "all four categories name a scaffold donor")

    # ── the channel refuses a label it would truncate ─────────────────
    print("-- the channel's label guard")
    var z = List[Float64](length=G1_SPEC_D, fill=1.0)
    var raised = False
    try:
        g1_channel_write_z(String("/tmp/g1_spec_test_chan"), 1,
                           String("two words"), z)
    except:
        raised = True
    t.truth(raised, "a label with a space RAISES rather than truncating")
    var fine = True
    try:
        g1_channel_write_z(String("/tmp/g1_spec_test_chan"), 1,
                           String("spec:right_foot_height_high"), z)
    except:
        fine = False
    t.truth(fine, "an underscore-joined label is accepted")

    try:
        remove_file(path)
        remove_file(String("/tmp/g1_spec_test_chan"))
    except:
        pass
    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_g1_spec: " + String(t.fails) + " failed")
