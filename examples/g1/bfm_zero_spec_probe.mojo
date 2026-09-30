# +--------------------------------------------------------------------------+ #
# | Can a GENERATED reward spec recover a command we already gated? (§10.2)
# +--------------------------------------------------------------------------+ #
"""The decisive experiment before any model writes a reward.

    pixi run mojo build -I . -Xlinker -ld_classic \
        examples/g1/bfm_zero_spec_probe.mojo -o build/g1spec
    ./build/g1spec --ckpt runs/<id>/checkpoints/step_36000.ckpt

## The question

`docs/BFM_ZERO_NEXT_LEVEL.md` §10.2 proposes that the 20-command bank is a
CACHE and the real interface is the term list — so a model could emit a
reward spec for a command that is not in the bank. Before building any of
that, one thing has to be true: **a spec has to be able to reproduce a
command we ALREADY gated.** If it cannot recover `squat`, whose answer we
know, generating novel commands is guesswork with extra steps.

⚠ AND IT HAS TO BE A RETRIEVAL TEST, NOT A DISTANCE. "Within 20 degrees"
means nothing on its own. The bank's own `z` rows sit at a median 84 deg
apart with a MINIMUM of 32.9 (`strafe_left` / `diagonal`), so 33 deg is the
confusion radius: a candidate at 60 deg is inside several commands'
neighbourhoods at once. The metric here is therefore **which bank entry the
candidate's `z` is nearest to**, with the margin to the runner-up.

## The three phases, each a control for the next

- **A — the harness.** Rebuild each command's `z` from the bank's OWN
  recorded terms. `angle(ref, shipped)` is what CEM moved, and it proves
  this file's reward -> z path matches the builder's. A large angle here
  invalidates everything below it.
- **B — the numeric question, and it needs NO model.** §10.2 wants the model
  to emit a direction and a COARSE LEVEL, with the number resolved from the
  pool's own quantile (`G1Term.as_pct` exists for this). So quantise every
  threshold to the nearest decile of its quantity's pool distribution and
  ask whether the command survives. If exact thresholds are load-bearing,
  the coarse-level design is dead however good the model is.
- **C — the vacuity control.** Retrieval accuracy is only meaningful if it
  CAN fail. Each command is also scored with its goal terms replaced by
  another command's, which must retrieve the wrong entry.

Nothing here rolls the policy out: it is `reward -> z` over the pool, which
is a matvec. That is what makes it minutes rather than the builder's 25 per
entry, and it is also the honest scope — this measures whether a spec names
the right BEHAVIOUR, not whether the robot then holds it.
"""

from std.math import acos, sqrt
from std.sys import argv
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.data.store import TrajectoryStore
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.obs_norm import ObsNorm
from noeira.deep_agents.fb.z_sampler import z_from_reward
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMActorTowerFiltered, BFMBNetFiltered,
)
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA,
)
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
    UNITREE_G1_PRIV_DIM,
)
from noeira.envs.robots.g1_tracking_eval import G1_D, G1_H, G1_L, G1_HB
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_reward_vocab import (
    G1_NVOC, G1Term, g1_quantities, g1_vocab_name, g1_term_value,
    g1_ess_rows, g1_quantile, OP_GT, OP_LT, OP_BAND, OP_SOFT,
    QV_SPEED_FWD, QV_SPEED_LAT, QV_SPEED, QV_YAW_RATE, QV_UPRIGHT,
)

comptime SP: Int = UNITREE_G1_OBS_DIM
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM
comptime ACT: Int = UnitreeG1Model.ACTION_DIM
comptime D: Int = G1_D
comptime BATCH: Int = 64
comptime CHUNK: Int = 4096

comptime FNet = BFMFTower[OBS, ACT, D, G1_H, G1_L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, G1_HB]
comptime ANet = BFMActorTowerFiltered[
    OBS, UNITREE_G1_STATE_DIM, G1_ACTOR_EXTRA, D, G1_H, G1_L, ACT
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]


def _flag(name: String, dflt: String) -> String:
    var av = argv()
    for i in range(len(av)):
        if String(av[i]) == name and i + 1 < len(av):
            return String(av[i + 1])
    return dflt


def _f1(v: Float64) -> String:
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 10.0 + 0.5)
    var b = String(h // 10) + String(".") + String(h % 10)
    return String("-") + b if neg else b


def _f2(v: Float64) -> String:
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 100.0 + 0.5)
    var f = String(h % 100)
    if h % 100 < 10:
        f = String("0") + f
    var b = String(h // 100) + String(".") + f
    return String("-") + b if neg else b


def _lpad(s: String, w: Int) -> String:
    var out = String("")
    for _ in range(w - s.byte_length()):
        out += " "
    return out + s


def _rpad(s: String, w: Int) -> String:
    var out = s.copy()
    for _ in range(w - s.byte_length()):
        out += " "
    return out


def _angle(ref a: List[Float64], ao: Int, ref b: List[Float64], bo: Int) -> Float64:
    """Degrees between two rows. ⚠ TWO SEPARATE LISTS — Mojo rejects two
    aliasing `ref` arguments, so every comparison here is candidate-vs-bank
    and never bank-vs-bank (that matrix is measured separately)."""
    var d = 0.0
    var na = 0.0
    var nb = 0.0
    for k in range(D):
        d += a[ao + k] * b[bo + k]
        na += a[ao + k] * a[ao + k]
        nb += b[bo + k] * b[bo + k]
    var c = d / (sqrt(na) * sqrt(nb))
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0
    return acos(c) * 57.29577951308232


def _z_of(
    ref terms: List[G1Term],
    ref qv: List[Float64],
    ref b_list: List[Scalar[DT]],
    n_pool: Int,
    mut rew: List[Scalar[DT]],
    mut out: List[Float64],
    off: Int,
) raises -> Float64:
    """The compound over the pool -> `z`. Returns the ESS, because a spec
    whose product is empty still yields a unit-norm `z` and a plausible
    robot — that is the silent failure this whole track keeps paying for."""
    var prod = List[Float64](length=n_pool, fill=0.0)
    for i in range(n_pool):
        var p = 1.0
        for t in range(len(terms)):
            var x = qv[i * G1_NVOC + terms[t].q]
            p *= g1_term_value(terms[t], terms[t].lo, terms[t].hi, x)
            if p == 0.0:
                break
        prod[i] = p
        rew[i] = Scalar[DT](p)
    var ess = g1_ess_rows(prod, n_pool)
    var zl = z_from_reward[D](b_list, rew, n_pool)
    for k in range(D):
        out[off + k] = Float64(zl[k])
    return ess


def main() raises:
    var ckpt = _flag(String("--ckpt"), String(""))
    var store_path = _flag(String("--store"), String("lafan_g1_50hz.h5"))
    var bank_path = _flag(String("--bank"), String("g1_command_bank.txt"))
    var n_pool = atol(_flag(String("--pool"), String(65536)))
    # ⚠ the decile ladder of phase B. 10 levels is what a `score` question
    # with 5 levels plus a direction can address; finer than the model can
    # plausibly emit, so it is a GENEROUS test of the coarse-level design.
    var step = Float64(String(_flag(String("--quantise"), String("0.10"))))
    if ckpt == "":
        raise Error("pass --ckpt <path/to/step_NNNN.ckpt>")

    print("=" * 96)
    print("BFM-Zero G1 — can a generated spec recover a gated command? (§10.2)")
    print("=" * 96)

    var bank = G1CommandBank.load(bank_path)
    print("  bank:", bank.count(), "commands from", bank_path)

    var t = Trainer.make(
        lr=3e-4, gamma=0.98, tau=0.01, ortho_weight=100.0, ctx=None,
        seed=UInt64(7),
    )
    t.load_state(ckpt)
    var norm = ObsNorm[OBS].try_load(ckpt + ".norm")
    if not norm:
        print("  ⚠ no .norm sidecar beside the checkpoint: RAW inputs.")

    var store = TrajectoryStore(store_path)
    var st = store.load_column[DType.float32](String("state"))
    var pv = store.load_column[DType.float32](String("privileged"))
    var n_rows = len(st) // UNITREE_G1_STATE_DIM
    if n_pool > n_rows:
        n_pool = n_rows
    var stride = n_rows // n_pool

    # ── the pool, identically to the builder ──────────────────────────
    var t0 = perf_counter_ns()
    var b_list = List[Scalar[DT]](length=n_pool * D, fill=Scalar[DT](0))
    var qv = List[Float64](length=n_pool * G1_NVOC, fill=0.0)
    var row = List[Float64](length=SP, fill=0.0)
    var chunk_t = Tensor.alloc(CHUNK * OBS)
    var b_chunk = Tensor()
    var done = 0
    while done < n_pool:
        var m = CHUNK if done + CHUNK <= n_pool else n_pool - done
        for i in range(CHUNK * OBS):
            chunk_t.data[i] = Scalar[DT](0.0)
        for j in range(m):
            var r = (done + j) * stride
            for k in range(UNITREE_G1_STATE_DIM):
                row[k] = Float64(st[r * UNITREE_G1_STATE_DIM + k])
            for k in range(UNITREE_G1_PRIV_DIM):
                row[UNITREE_G1_STATE_DIM + k] = Float64(
                    pv[r * UNITREE_G1_PRIV_DIM + k]
                )
            g1_quantities(row, 0, qv, (done + j) * G1_NVOC)
            for k in range(SP):
                chunk_t.data[j * OBS + k] = Scalar[DT](row[k])
        if norm:
            norm.value().apply_rows(chunk_t, CHUNK)
        t.backward_embed[CHUNK](chunk_t, b_chunk)
        for j in range(m):
            for k in range(D):
                b_list[(done + j) * D + k] = b_chunk.data[j * D + k]
        done += m
    print("  pool", n_pool, "rows (stride", stride, "of", n_rows,
          ") — B encoded in", Int(Float64(perf_counter_ns() - t0) / 1e9), "s")

    # ── each quantity's sorted column, for the percentile round trip ──
    var cols = List[Float64](length=G1_NVOC * n_pool, fill=0.0)
    for q in range(G1_NVOC):
        for i in range(n_pool):
            cols[q * n_pool + i] = qv[i * G1_NVOC + q]

    var n = bank.count()
    var rew = List[Scalar[DT]](length=n_pool, fill=Scalar[DT](0))
    var z_ship = List[Float64](length=n * D, fill=0.0)
    var z_ref = List[Float64](length=n * D, fill=0.0)
    var z_crs = List[Float64](length=n * D, fill=0.0)
    var z_ctl = List[Float64](length=n * D, fill=0.0)
    var ess_ref = List[Float64](length=n, fill=0.0)
    var ess_crs = List[Float64](length=n, fill=0.0)

    for c in range(n):
        for k in range(D):
            z_ship[c * D + k] = bank.z_at(c, k)

    # ── phase A: the bank's own terms -> z ────────────────────────────
    print()
    print("-" * 96)
    print("PHASE A — the harness: the bank's OWN terms rebuilt into a `z`")
    print("-" * 96)
    for c in range(n):
        var ts = List[G1Term]()
        var s0 = bank.t_start[c]
        for j in range(bank.t_count[c]):
            var k = s0 + j
            ts.append(G1Term(bank.t_q[k], bank.t_op[k], bank.t_lo[k],
                             bank.t_hi[k], False))
        ess_ref[c] = _z_of(ts, qv, b_list, n_pool, rew, z_ref, c * D)

    # ── phase B: every threshold quantised to the decile ladder ───────
    for c in range(n):
        var ts = List[G1Term]()
        var s0 = bank.t_start[c]
        for j in range(bank.t_count[c]):
            var k = s0 + j
            var q = bank.t_q[k]
            var lo = bank.t_lo[k]
            var hi = bank.t_hi[k]
            # ⚠ OP_SOFT's `lo` is a RATE, not a threshold on the quantity —
            # quantising it against the quantity's own distribution would be
            # a category error. Left alone.
            if bank.t_op[k] != OP_SOFT:
                if bank.t_op[k] == OP_GT or bank.t_op[k] == OP_BAND:
                    lo = _snap(cols, q, n_pool, lo, step)
                if bank.t_op[k] == OP_LT or bank.t_op[k] == OP_BAND:
                    hi = _snap(cols, q, n_pool, hi, step)
            ts.append(G1Term(q, bank.t_op[k], lo, hi, False))
        ess_crs[c] = _z_of(ts, qv, b_list, n_pool, rew, z_crs, c * D)

    # ── phase B2: quantise everything EXCEPT the physics quantities ───
    # ⚠ THIS IS THE DESIGN AS WRITTEN, not a new idea. `g1_reward_vocab`'s
    # own docstring says "thresholds that should follow the data say so;
    # thresholds that are physics (uprightness, a speed in m/s) stay
    # literal". B quantised both and B2 quantises only the first kind, so
    # the gap between them prices that sentence.
    var z_b2 = List[Float64](length=n * D, fill=0.0)
    var ess_b2 = List[Float64](length=n, fill=0.0)
    print()
    print("  term-level, where quantisation moved a threshold at all:")
    for c in range(n):
        var ts = List[G1Term]()
        var s0 = bank.t_start[c]
        for j in range(bank.t_count[c]):
            var k = s0 + j
            var q = bank.t_q[k]
            var lo = bank.t_lo[k]
            var hi = bank.t_hi[k]
            var phys = (
                q == QV_SPEED_FWD or q == QV_SPEED_LAT or q == QV_SPEED
                or q == QV_YAW_RATE or q == QV_UPRIGHT
            )
            if bank.t_op[k] != OP_SOFT and not phys:
                if bank.t_op[k] == OP_GT or bank.t_op[k] == OP_BAND:
                    lo = _snap(cols, q, n_pool, lo, step)
                if bank.t_op[k] == OP_LT or bank.t_op[k] == OP_BAND:
                    hi = _snap(cols, q, n_pool, hi, step)
            ts.append(G1Term(q, bank.t_op[k], lo, hi, False))
            # the trace uses phase B's rule (everything quantised), because
            # that is the leg with the failures to explain
            var qlo = bank.t_lo[k]
            var qhi = bank.t_hi[k]
            if bank.t_op[k] != OP_SOFT:
                if bank.t_op[k] == OP_GT or bank.t_op[k] == OP_BAND:
                    qlo = _snap(cols, q, n_pool, bank.t_lo[k], step)
                if bank.t_op[k] == OP_LT or bank.t_op[k] == OP_BAND:
                    qhi = _snap(cols, q, n_pool, bank.t_hi[k], step)
            var dlo = qlo - bank.t_lo[k]
            var dhi = qhi - bank.t_hi[k]
            var adlo = dlo if dlo > 0.0 else -dlo
            var adhi = dhi if dhi > 0.0 else -dhi
            if adlo > 0.02 or adhi > 0.02:
                print("    " + _rpad(bank.name_at(c), 20)
                      + _rpad(g1_vocab_name(q), 26)
                      + " [" + _f2(bank.t_lo[k]) + ", " + _f2(bank.t_hi[k])
                      + "] -> [" + _f2(qlo) + ", " + _f2(qhi) + "]"
                      + (String("   PHYSICS") if phys else String("")))
        ess_b2[c] = _z_of(ts, qv, b_list, n_pool, rew, z_b2, c * D)

    # ── phase C: the goal terms swapped for the next command's ────────
    for c in range(n):
        var ts = List[G1Term]()
        var s0 = bank.t_start[c]
        for j in range(bank.t_count[c]):
            var k = s0 + j
            if bank.t_goal[k]:
                continue                      # keep only the scaffold
            ts.append(G1Term(bank.t_q[k], bank.t_op[k], bank.t_lo[k],
                             bank.t_hi[k], False))
        var o = (c + 1) % n                   # a DIFFERENT command's goal
        var s1 = bank.t_start[o]
        for j in range(bank.t_count[o]):
            var k = s1 + j
            if not bank.t_goal[k]:
                continue
            ts.append(G1Term(bank.t_q[k], bank.t_op[k], bank.t_lo[k],
                             bank.t_hi[k], False))
        _ = _z_of(ts, qv, b_list, n_pool, rew, z_ctl, c * D)

    # ── retrieval, against the SHIPPED bank ───────────────────────────
    # ⚠ TWO RETRIEVAL TARGETS, AND THE FIRST RUN CONFLATED THEM. Retrieving
    # a BASELINE candidate against the SHIPPED bank handicaps it by whatever
    # CEM moved that entry — measured below at 0.0 deg on 9 entries and
    # 23.7-27.9 on the other 11. `arms_wide` failed that way on the first
    # run: its coarse spec sits 0.6 deg from its own baseline and was still
    # scored a miss, because shipped `arms_wide` had been moved 25.9 deg
    # while shipped `both_hands_up` sat 36 deg away in the other direction.
    # So `vs_base` isolates the quantisation and `vs_ship` is what a cache
    # keyed on the shipped rows would actually do. They are different
    # questions and the design conclusion is in the gap between them.
    print(_rpad(String("command"), 20) + _lpad(String("A:cem"), 7)
          + _lpad(String("ESSref"), 8) + _lpad(String("B:crs^ref"), 11)
          + _lpad(String("ESScrs"), 8)
          + _lpad(String("retr(base)"), 21) + _lpad(String("marg"), 7)
          + _lpad(String("retr(ship)"), 21) + _lpad(String("B2^ref"), 8))
    var hitB = 0
    var hitS = 0
    var hitC = 0
    var hit2 = 0
    var ess2_dead = String("")
    var ctl_names = String("")
    var ess_dead = String("")
    for c in range(n):
        var a_ref = _angle(z_ref, c * D, z_ship, c * D)
        var a_crs = _angle(z_crs, c * D, z_ref, c * D)
        # nearest BASELINE entry — isolates the quantisation
        var b1 = 1e9
        var b1i = -1
        var b2 = 1e9
        for o in range(n):
            var a = _angle(z_crs, c * D, z_ref, o * D)
            if a < b1:
                b2 = b1
                b1 = a
                b1i = o
            elif a < b2:
                b2 = a
        if b1i == c:
            hitB += 1
        # nearest SHIPPED entry — what a cache on the bank rows would do
        var s1 = 1e9
        var s1i = -1
        for o in range(n):
            var a = _angle(z_crs, c * D, z_ship, o * D)
            if a < s1:
                s1 = a
                s1i = o
        if s1i == c:
            hitS += 1
        # the control must land somewhere else
        var c1 = 1e9
        var c1i = -1
        for o in range(n):
            var a = _angle(z_ctl, c * D, z_ref, o * D)
            if a < c1:
                c1 = a
                c1i = o
        if c1i == c:
            hitC += 1
            ctl_names += bank.name_at(c) + String(" ")
        var d1 = 1e9
        var d1i = -1
        for o in range(n):
            var a = _angle(z_b2, c * D, z_ref, o * D)
            if a < d1:
                d1 = a
                d1i = o
        if d1i == c:
            hit2 += 1
        if ess_b2[c] < 100.0:
            ess2_dead += bank.name_at(c) + String("(") + String(
                Int(ess_b2[c])) + String(") ")
        if ess_crs[c] < 100.0:
            ess_dead += bank.name_at(c) + String("(") + String(
                Int(ess_crs[c])) + String(") ")
        var mk = String("  ") if b1i == c else String(" X")
        var ms = String("  ") if s1i == c else String(" X")
        print(_rpad(bank.name_at(c), 20) + _lpad(_f1(a_ref), 7)
              + _lpad(String(Int(ess_ref[c])), 8) + _lpad(_f1(a_crs), 11)
              + _lpad(String(Int(ess_crs[c])), 8)
              + _lpad(bank.name_at(b1i) + mk, 21) + _lpad(_f1(b2 - b1), 7)
              + _lpad(bank.name_at(s1i) + ms, 21)
              + _lpad(_f1(_angle(z_b2, c * D, z_ref, c * D)), 8))

    print()
    print("-" * 96)
    print("  B, vs BASELINE :", hitB, "/", n,
          "— quantisation alone; this is the design question")
    print("  B, vs SHIPPED  :", hitS, "/", n,
          "— what a cache keyed on the bank's CEM-refined rows would do")
    print("  B2, phys LITERAL:", hit2, "/", n,
          "— quantise only the percentile quantities (the design as written)")
    if ess2_dead != "":
        print("    ⚠ ESS still collapsed under B2:", ess2_dead)
    print("  C control      :", hitC, "/", n,
          "— a SWAPPED goal must retrieve the WRONG entry; 0 is the pass")
    if ctl_names != "":
        print("    ⚠ control retrieved itself for:", ctl_names,
              "— the scaffold is doing the work there (§12.51)")
    if ess_dead != "":
        print("    ⚠ ESS collapsed under quantisation:", ess_dead,
              "— refused online, not silently wrong")
    print("-" * 96)


def _snap(
    ref cols: List[Float64], q: Int, n_pool: Int, x: Float64, step: Float64
) raises -> Float64:
    """`x` -> the nearest decile of quantity `q`'s pool distribution.

    ⚠ THIS IS THE WHOLE POINT OF PHASE B. §10.2 wants the model to say
    "high" and the number to come from the data (`G1Term.as_pct`). If the
    round trip value -> percentile -> value moves the `z` out of its own
    neighbourhood, then the thresholds are load-bearing at a precision no
    language model will ever emit, and the design fails on arithmetic rather
    than on semantics.
    """
    var below = 0
    for i in range(n_pool):
        if cols[q * n_pool + i] < x:
            below += 1
    var p = Float64(below) / Float64(n_pool)
    var lvl = Float64(Int(p / step + 0.5)) * step
    if lvl < 0.05:
        lvl = 0.05
    if lvl > 0.95:
        lvl = 0.95
    var col = List[Float64](length=n_pool, fill=0.0)
    for i in range(n_pool):
        col[i] = cols[q * n_pool + i]
    return g1_quantile(col, lvl)
