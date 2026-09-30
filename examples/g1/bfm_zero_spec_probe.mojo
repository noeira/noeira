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
from noeira.ai.jev import JevClient, JevQuestions
from noeira.core.bytes import string_from_bytes
from noeira.io.fileio import read_file_bytes
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_spec import g1_pool_save
from noeira.envs.robots.g1_reward_vocab import (
    G1_NVOC, G1Term, g1_quantities, g1_vocab_name, g1_term_value,
    g1_ess_rows, g1_quantile, OP_GT, OP_LT, OP_BAND, OP_SOFT,
    QV_SPEED_FWD, QV_SPEED_LAT, QV_SPEED, QV_YAW_RATE, QV_UPRIGHT,
    QV_BODY_H, QV_HEAD_H, QV_LHAND_H, QV_RHAND_H, QV_LHAND_LAT,
    QV_RHAND_LAT, QV_LFOOT_H, QV_RFOOT_H, QV_TORSO_YAW,
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


def _has(name: String) -> Bool:
    var av = argv()
    for i in range(len(av)):
        if String(av[i]) == name:
            return True
    return False


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
    # ⚠ THE LADDER IS A FLAG, NOT A CONSTANT. §12.57 priced hand height as
    # the precision-critical quantity (3-4 cm = up to 20 deg), so the level
    # a word resolves to is exactly the kind of number that must be
    # measured rather than chosen. The bank itself uses p90 for a raised
    # hand and p50 for the other one.
    var high_p = Float64(String(_flag(String("--high-pct"), String("0.90"))))
    # goal `low` (slot 1) and suppression `low` (later slots) are separate
    # numbers — see the note at the resolver.
    var low_p = Float64(String(_flag(String("--low-pct"), String("0.10"))))
    var sup_p = Float64(String(_flag(String("--sup-pct"), String("0.50"))))
    # what "a moderate amount" and "as far as it can" resolve to
    var mag_base = Float64(String(_flag(String("--mag-base"), String("0.88"))))
    var mag_ext = Float64(String(_flag(String("--mag-ext"), String("0.95"))))
    var mid_lo = Float64(String(_flag(String("--mid-lo"), String("0.40"))))
    var mid_hi = Float64(String(_flag(String("--mid-hi"), String("0.60"))))
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

    # ── the sidecar: this pool, for consumers with no checkpoint ───────
    # ⚠ THIS IS WHAT MAKES THE CACHE-MISS PATH DEPLOYABLE. `z = E_rho[r B]`
    # needs `B` over the pool and nothing else — no actor, no critic, no
    # simulator — so the encoded rows go to a 67 MB file and `g1say` can
    # compute a novel `z` without the 640 MB checkpoint or the 1.7 GB store.
    var emit = _flag(String("--emit-pool"), String(""))
    if emit != "":
        var bf = List[Float64](length=n_pool * D, fill=0.0)
        for i in range(n_pool * D):
            bf[i] = Float64(b_list[i])
        g1_pool_save(emit, n_pool, bf, qv)
        print("  wrote", emit, "—", n_pool, "rows x", D, "+", G1_NVOC,
              "quantities")

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

    # ── the SEMANTIC leg: does a model name the right quantities? ─────
    var ask_path = _flag(String("--ask"), String(""))
    if ask_path != "":
        # ⚠ TWO ARMS, because "the model cannot" and "the model was not
        # told" are different findings. Arm B states how `z = E_rho[r B]`
        # behaves — that an unconstrained DoF returns the dataset average —
        # which is a fact about the SYSTEM, not the answer to any particular
        # instruction. §12.51 is the section that fact came from.
        _ask_leg(
            ask_path, bank, qv, b_list, n_pool, cols, z_ref, high_p, low_p,
            mid_lo, mid_hi, sup_p, _has(String("--warn")),
            # ⚠ OPT-IN, BECAUSE IT WAS MEASURED AND IT DOES NOT PAY (§12.59).
            # The `score` magnitude neither separated `walk` from `run` — the
            # model returns 1.0 for BOTH "marche" and "cours" — nor improved
            # retrieval (13-14 of 16 either way, inside run-to-run noise),
            # and it costs +345 input tokens per instruction. With a loose
            # base it actively hurts, 11 of 16, by overriding thresholds the
            # two-level ladder had right.
            _has(String("--magnitude")), mag_base, mag_ext, high_p,
        )

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


# ⚠ THREE OFFERED, FOUR NAMED. `middle` stays in the enum because
# `_truth_dir` needs a value for a term that is neither, but it is NOT
# offered to the model — see the note beside the option list.
comptime N_LVL: Int = 4
comptime LVL_HIGH: Int = 0
comptime LVL_LOW: Int = 1
comptime LVL_MID: Int = 2
comptime LVL_NONE: Int = 3


def _vocab_desc(q: Int) -> String:
    """What each quantity MEANS, including its sign.

    ⚠ THE SIGN CONVENTION WAS MISSING AND THAT ALONE MADE TWO COMMANDS
    UNANSWERABLE. `step to your left` and `step to your right` came back as
    BYTE-IDENTICAL specs, because nothing told the model that a positive
    lateral speed means left. That is not a model failure and no amount of
    prompt tuning fixes it: the information was not in the question. Every
    signed quantity now states its direction, which is a fact about the
    state representation in the same way the option list is.
    """
    if q == QV_BODY_H:
        return String("height of the pelvis above the ground, in metres")
    if q == QV_HEAD_H:
        return String("height of the head above the ground, in metres")
    if q == QV_LHAND_H:
        return String("height of the LEFT hand above the ground, in metres")
    if q == QV_RHAND_H:
        return String("height of the RIGHT hand above the ground, in metres")
    if q == QV_LHAND_LAT:
        return String(
            "how far the LEFT hand is out to the side, away from the body"
        )
    if q == QV_RHAND_LAT:
        return String(
            "how far the RIGHT hand is out to the side, away from the body"
        )
    if q == QV_LFOOT_H:
        return String("height of the LEFT foot above the ground")
    if q == QV_RFOOT_H:
        return String("height of the RIGHT foot above the ground")
    if q == QV_UPRIGHT:
        return String(
            "how upright the torso is: 1 = straight up, 0 = horizontal"
        )
    if q == QV_SPEED_FWD:
        return String(
            "speed along the direction the robot faces, in m/s."
            " POSITIVE = FORWARDS, NEGATIVE = BACKWARDS"
        )
    if q == QV_SPEED_LAT:
        return String(
            "sideways speed, in m/s. POSITIVE = TO THE ROBOT'S LEFT,"
            " NEGATIVE = TO ITS RIGHT"
        )
    if q == QV_SPEED:
        return String(
            "overall speed over the ground, in m/s, never negative"
        )
    if q == QV_YAW_RATE:
        return String(
            "how fast the whole body turns about the vertical axis, in rad/s."
            " POSITIVE = TURNING LEFT, NEGATIVE = TURNING RIGHT"
        )
    return String(
        "how far the torso is twisted about the vertical axis relative to the"
        " direction of travel. POSITIVE = TWISTED LEFT, NEGATIVE = RIGHT"
    )


def _dir_option(q: Int, hi: Bool) -> String:
    """The option NAME for one (quantity, direction) pair."""
    return g1_vocab_name(q) + (String("_high") if hi else String("_low"))


def _dir_desc(q: Int, hi: Bool) -> String:
    """What that pair MEANS, in the robot's own terms.

    ⚠ THE DIRECTION AND ITS MEANING MUST BE IN THE SAME OPTION. The previous
    form asked "which quantity?" with the sign convention on those options,
    then a SEPARATE "high or low?" whose options were bare. Jev answers every
    question independently against one state, so the convention never reached
    the question that needed it — and `step to your left` came back
    `lateral:low` while `step to your right` came back `lateral:high`. LEFT
    AND RIGHT INVERTED, on a question that could not have been answered from
    what it was given.
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


def _lvl_name(i: Int) -> String:
    if i == LVL_HIGH:
        return String("high")
    if i == LVL_LOW:
        return String("low")
    if i == LVL_MID:
        return String("middle")     # retained for the ground-truth mapping
    return String("unconstrained")


def _cat_name(i: Int) -> String:
    """The four scaffold categories, after Motivo's own task taxonomy."""
    if i == 0:
        return String("standing")
    if i == 1:
        return String("low")
    if i == 2:
        return String("moving")
    return String("turning")


def _cat_donor(i: Int) -> String:
    """⚠ THE SCAFFOLD COMES FROM THE BANK, NOT FROM A COPY OF IT. Each
    category names an entry whose NON-goal terms are that scaffold, so this
    file never transcribes `_scaffold_stand` and friends out of
    `bfm_zero_bank_build.mojo`. A second copy of the scaffold rules is
    exactly the defect shape §12.51 was created by."""
    if i == 0:
        return String("right_hand_up")     # _scaffold_stand
    if i == 1:
        return String("squat")             # _scaffold_low
    if i == 2:
        return String("walk")              # _scaffold_move
    return String("spin_left")             # _scaffold_rotate


def _truth_dir(op: Int, lo: Float64, hi: Float64, med: Float64) -> Int:
    """The ground-truth LEVEL of a bank goal term.

    ⚠ A BAND'S DIRECTION IS NOT IN ITS OPERATOR. `squat` is
    `body_height BAND [0.40, 0.62]`, which means LOW; `run` is
    `speed_forward BAND [1.50, 3.00]`, which means HIGH. Both are OP_BAND.
    The only thing that separates them is where the band sits in the
    quantity's own distribution, so the pool's median decides — which is
    why this probe needs the pool even for the semantic leg.
    """
    if op == OP_GT:
        return LVL_HIGH
    if op == OP_LT:
        return LVL_LOW
    if op == OP_BAND:
        var c = (lo + hi) * 0.5
        if c > med:
            return LVL_HIGH
        return LVL_LOW
    return LVL_NONE


def _pct_for(m: Float64, hi: Bool, base: Float64, ext: Float64) -> Float64:
    """A magnitude score (0..3, fractional) -> the percentile to threshold at.

    `base` is what a "moderate" instruction means and `ext` what an emphatic
    one does; the score interpolates between them, which is why a `score`
    rather than a `choice` is what breaks the magnitude ceiling.
    """
    var t = m / 3.0
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    if hi:
        return base + (ext - base) * t
    return (1.0 - base) + ((1.0 - ext) - (1.0 - base)) * t


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


def _ask_leg(
    path: String,
    ref bank: G1CommandBank,
    ref qv: List[Float64],
    ref b_list: List[Scalar[DT]],
    n_pool: Int,
    ref cols: List[Float64],
    ref z_ref: List[Float64],
    high_p: Float64,
    low_p: Float64,
    mid_lo: Float64,
    mid_hi: Float64,
    sup_p: Float64,
    warn: Bool,
    mag: Bool,
    mag_base: Float64,
    mag_ext: Float64,
    qh_p: Float64,
) raises:
    """Ask a model for the reward, score it two ways.

    The numeric half is settled (§12.57): coarse, quantile-resolved levels
    carry a command. This is the other half — given an instruction and
    nothing else, does the model name the right QUANTITIES with the right
    DIRECTIONS? Scored against each bank entry's own goal terms, and then
    end to end by building its answer into a `z` and retrieving.
    """
    print()
    print("=" * 96)
    print("SEMANTIC LEG — a model writes the reward (§10.2's other half)")
    print("=" * 96)

    # per-quantity medians, for the band-direction rule and the resolver
    var med = List[Float64](length=G1_NVOC, fill=0.0)
    var qh = List[Float64](length=G1_NVOC, fill=0.0)
    var ql = List[Float64](length=G1_NVOC, fill=0.0)
    var qs = List[Float64](length=G1_NVOC, fill=0.0)
    var qml = List[Float64](length=G1_NVOC, fill=0.0)
    var qmh = List[Float64](length=G1_NVOC, fill=0.0)
    var col = List[Float64](length=n_pool, fill=0.0)
    for q in range(G1_NVOC):
        for i in range(n_pool):
            col[i] = cols[q * n_pool + i]
        med[q] = g1_quantile(col, 0.50)
        qh[q] = g1_quantile(col, high_p)
        ql[q] = g1_quantile(col, low_p)
        qs[q] = g1_quantile(col, sup_p)
        qml[q] = g1_quantile(col, mid_lo)
        qmh[q] = g1_quantile(col, mid_hi)

    var raw = string_from_bytes(read_file_bytes(path))
    var lines = raw.split("\n")
    var cmds = List[String]()
    var insts = List[String]()
    for i in range(len(lines)):
        var l = String(lines[i])
        if l.byte_length() == 0 or l.startswith("#"):
            continue
        var eq = l.find("=")
        if eq <= 0:
            continue
        cmds.append(String(l[byte=0:eq]))
        insts.append(String(l[byte=eq + 1:l.byte_length()]))
    print("  instructions:", len(cmds), "from", path)
    print("  ladder: high p" + _f2(high_p) + " | goal low p" + _f2(low_p)
          + " | suppression low p" + _f2(sup_p))
    print("  arm:", String("B — told that an unconstrained DoF returns the"
          " dataset average") if warn else String("A — no hint"))
    if mag:
        print("  magnitude: `score` over 4 rungs, moderate -> p"
              + _f2(mag_base) + ", extreme -> p" + _f2(mag_ext))
    else:
        print("  magnitude: OFF — the two-level ladder of §12.58")

    # ── the questions: one category + one level per quantity ──────────
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

    var lvls = List[String]()
    var ldesc = List[String]()
    lvls.append(String("high"))
    ldesc.append(String(
        "as large as it gets — for a signed quantity, strongly POSITIVE"
    ))
    lvls.append(String("low"))
    ldesc.append(String(
        "small — for a signed quantity, strongly NEGATIVE"
    ))

    # ⚠ BUDGETED SLOTS, NOT ONE QUESTION PER QUANTITY. Asking about all 14
    # independently was the third self-inflicted failure: each question is
    # answered in isolation, so the model has no way to say "only this one
    # matters" and it constrained 8-14 of them per instruction. Precision
    # 0.13. And the direction errors followed from the same thing — asked
    # about `torso_yaw` for "marche", the model has to answer something.
    #
    # A slot list forces the budget, and it is the CHAIN PATTERN that
    # §12.55 already measured working for command/command2/command3. The
    # bank's own entries carry 1-3 goal terms, so three slots is the right
    # size and `none` on slot 2 must stay cheap.
    var qnames = List[String]()
    var qdescs = List[String]()
    for v in range(G1_NVOC):
        qnames.append(_dir_option(v, True))
        qdescs.append(_dir_desc(v, True))
        qnames.append(_dir_option(v, False))
        qdescs.append(_dir_desc(v, False))
    qnames.append(String("none"))
    qdescs.append(String("no further quantity is constrained"))

    # ⚠ FOUR RUNGS, WORDED AS DEGREES AND NOT AS NUMBERS. Jev's own docs
    # list numeric comparison as a failure mode, so the rungs are words and
    # the percentile they resolve to is ours — the same rule that made
    # `doing_since` a word in §12.56.
    var mlvls = List[String]()
    mlvls.append(String("only slightly — barely more than the robot's usual"))
    mlvls.append(String("a moderate amount — a normal, unremarkable version"))
    mlvls.append(String("a lot — clearly more than usual"))
    mlvls.append(String("as far as the robot physically can"))

    var jev = JevClient.from_env()
    var rew = List[Scalar[DT]](length=n_pool, fill=Scalar[DT](0))
    var z_ask = List[Float64](length=D, fill=0.0)

    var tp = 0
    var fp = 0
    var fneg = 0
    var hit = 0
    var toks = 0
    print()
    print(_rpad(String("instruction"), 22) + _rpad(String("truth"), 34)
          + _rpad(String("model"), 34) + _lpad(String("ESS"), 7)
          + _lpad(String("retrieves"), 21))

    for r in range(len(cmds)):
        var c = bank.find(cmds[r])
        if c < 0:
            print("  ⚠ not in the bank, skipped:", cmds[r])
            continue
        # ground truth: the entry's own GOAL terms
        var t_q = List[Int]()
        var t_d = List[Int]()
        var s0 = bank.t_start[c]
        for j in range(bank.t_count[c]):
            var k = s0 + j
            if not bank.t_goal[k] or bank.t_op[k] == OP_SOFT:
                continue
            t_q.append(bank.t_q[k])
            t_d.append(_truth_dir(bank.t_op[k], bank.t_lo[k], bank.t_hi[k],
                                  med[bank.t_q[k]]))

        var q = JevQuestions()
        q.choice(
            String("category"),
            String(
                "A humanoid robot is given this instruction. Which of these"
                " best describes the POSTURE the robot should be in while"
                " carrying it out?"
            ),
            cats, cdesc,
        )
        var ordinals = List[String]()
        ordinals.append(String("FIRST and most important"))
        ordinals.append(String("SECOND"))
        ordinals.append(String("THIRD"))
        for s in range(3):
            q.choice(
                String("q") + String(s),
                String(
                    "A humanoid robot is carrying out this instruction. The"
                    " robot's body is measured by the quantities listed."
                    " Which of these is the " + ordinals[s] + " thing the"
                    " instruction asks for?"
                    + (
                        String(
                            " ⚠ Any quantity you do NOT name comes back as the"
                            " robot's average behaviour, so name the ones that"
                            " must NOT change as well: if one hand goes up,"
                            " the other must be named as staying low."
                        ) if warn else String("")
                    )
                    + (
                        String(" Pick `none` when the instruction constrains"
                               " nothing further.") if s > 0 else String("")
                    )
                ),
                qnames, qdescs,
            )
            # ⚠ A `score`, NOT A `choice`, AND THAT IS THE WHOLE POINT.
            # §12.58's ceiling was 18/20 because a two-level ladder cannot
            # separate commands differing only in MAGNITUDE on one quantity:
            # `walk` [0.5, 1.4] and `run` [1.5, 3.0] are both "forward speed
            # high", `squat` [0.40, 0.62] and `crouch` [0.62, 0.72] both
            # "body height low". `score` takes ORDERED levels and returns the
            # EXPECTED level as a float, so "quite fast" lands between two
            # rungs instead of being forced onto one. The `extent` question
            # already uses it for exactly this reason.
            #
            # ⚠ It re-derives which quantity it is talking about from the
            # instruction, the same way `command2` re-derives the second
            # step (§12.55). It cannot see its own answer to `q<s>`.
            if mag:
                q.score(
                    String("m") + String(s),
                    String(
                        "Think about the " + ordinals[s] + " thing this"
                        " instruction asks the robot for. HOW FAR in that"
                        " direction should the robot go? Judge it from the"
                        " words: a plain instruction is a moderate amount,"
                        " and only an emphatic one is extreme."
                    ),
                    mlvls,
                )
        var a = jev.decide_text(String("instruction: ") + insts[r], q)
        toks += a.input_tokens

        # ── the model's spec: the donor scaffold + its named goals ────
        var cat_pick = a.choice(String("category"))
        var ci = 0
        for j in range(4):
            if _cat_name(j) == cat_pick:
                ci = j
        var donor = bank.find(_cat_donor(ci))
        var ts = List[G1Term]()
        var d0 = bank.t_start[donor]
        for j in range(bank.t_count[donor]):
            var k = d0 + j
            if bank.t_goal[k]:
                continue
            ts.append(G1Term(bank.t_q[k], bank.t_op[k], bank.t_lo[k],
                             bank.t_hi[k], False))

        var m_q = List[Int]()
        var m_d = List[Int]()
        var m_m = List[Float64]()
        for s in range(3):
            var pick = a.choice(String("q") + String(s))
            var vi = -1
            var li = LVL_HIGH
            for v in range(G1_NVOC):
                if _dir_option(v, True) == pick:
                    vi = v
                    li = LVL_HIGH
                elif _dir_option(v, False) == pick:
                    vi = v
                    li = LVL_LOW
            # ⚠ STOP AT THE FIRST `none`, exactly as `g1_decide_chain` does:
            # a third slot without a second is a misread, not a spec.
            if vi < 0:
                break
            var already = False
            for j in range(len(m_q)):
                if m_q[j] == vi:
                    already = True
            if already:
                continue
            m_q.append(vi)
            m_d.append(li)
            m_m.append(a.score(String("m") + String(s)) if mag else -1.0)
            # ⚠ THE LADDER IS SLOT-DEPENDENT, AND THIS IS MEASURED, NOT
            # CHOSEN. `low` means two different things. As the FIRST thing
            # asked for it is a GOAL — "move backwards", "look right" — and
            # must be firmly negative: at p50 `recule` scored ESS 27407 and
            # retrieved `stand`, at p10 it scored 6072 and retrieved `back`.
            # In a LATER slot it is almost always SUPPRESSION — "the other
            # hand stays down" — and p10 empties the product outright:
            # `lève le bras droit` went from ESS 233 to ESS **0**, because
            # the pool has essentially no frames with one hand above p90
            # while the other is below p10.
            #
            # The bank had this right all along and it is visible in its own
            # terms: `right_hand_up` is `RHAND_H > p90` AND `LHAND_H < p50`.
            # A goal threshold and a suppression threshold, in one compound.
            # the column, for a percentile the ladder chose at run time
            for i2 in range(n_pool):
                col[i2] = cols[vi * n_pool + i2]
            if li == LVL_HIGH:
                var p = qh_p if not mag else _pct_for(
                    a.score(String("m") + String(s)), True, mag_base, mag_ext
                )
                ts.append(G1Term(vi, OP_GT, g1_quantile(col, p), 0.0, False))
            elif s == 0:
                var p = low_p if not mag else _pct_for(
                    a.score(String("m") + String(s)), False, mag_base, mag_ext
                )
                ts.append(G1Term(vi, OP_LT, 0.0, g1_quantile(col, p), False))
            else:
                # ⚠ SUPPRESSION IGNORES THE MAGNITUDE, and that is §12.58's
                # measurement rather than a simplification: a later slot is
                # almost always "the other hand stays down", and honouring
                # an emphatic "as low as it can" there took ESS 233 -> 0.
                ts.append(G1Term(vi, OP_LT, 0.0, qs[vi], False))

        var ess = _z_of(ts, qv, b_list, n_pool, rew, z_ask, 0)

        # set scores on (quantity, direction)
        for j in range(len(t_q)):
            var found = False
            for i2 in range(len(m_q)):
                if m_q[i2] == t_q[j] and m_d[i2] == t_d[j]:
                    found = True
            if found:
                tp += 1
            else:
                fneg += 1
        for i2 in range(len(m_q)):
            var found = False
            for j in range(len(t_q)):
                if m_q[i2] == t_q[j] and m_d[i2] == t_d[j]:
                    found = True
            if not found:
                fp += 1

        # retrieval against the BASELINE rows (§12.57's conclusion 2)
        var b1 = 1e9
        var b1i = -1
        for o in range(bank.count()):
            var ang = _angle(z_ask, 0, z_ref, o * D)
            if ang < b1:
                b1 = ang
                b1i = o
        if b1i == c:
            hit += 1

        var ts_t = String("")
        for j in range(len(t_q)):
            if ts_t != "":
                ts_t += String(",")
            ts_t += g1_vocab_name(t_q[j]) + String(":") + _lvl_name(t_d[j])
        var ts_m = String("")
        for j in range(len(m_q)):
            if ts_m != "":
                ts_m += String(",")
            ts_m += g1_vocab_name(m_q[j]) + String(":") + _lvl_name(m_d[j])
            # ⚠ SHOW THE MAGNITUDE. A wrong retrieval with the right
            # quantity is a LADDER failure and one with the wrong quantity
            # is a SEMANTIC failure; without the score printed the two are
            # indistinguishable, which is the defect shape this whole
            # section keeps paying for.
            if mag and m_m[j] >= 0.0:
                ts_m += String("@") + _f1(m_m[j])
        if ts_m == "":
            ts_m = String("(nothing)")
        var mk = String("  ") if b1i == c else String(" X")
        print(_rpad(insts[r], 22) + _rpad(ts_t, 34) + _rpad(ts_m, 34)
              + _lpad(String(Int(ess)), 7)
              + _lpad(bank.name_at(b1i) + mk, 21))

    print()
    print("-" * 96)
    var prec = Float64(tp) / Float64(tp + fp) if tp + fp > 0 else 0.0
    var rec = Float64(tp) / Float64(tp + fneg) if tp + fneg > 0 else 0.0
    print("  (quantity, direction) precision", _f2(prec), " recall", _f2(rec),
          " [tp", tp, "fp", fp, "fn", fneg, "]")
    print("  END TO END retrieval:", hit, "/", len(cmds),
          "— the model's own spec names its own command")
    print("  input tokens:", toks, "total,", toks // len(cmds), "per instruction")
    print("-" * 96)
