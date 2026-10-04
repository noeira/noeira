"""P7 — the test-time-adaptation machinery, before any experiment uses it.

docs/LEWM_REOPEN_PLAN.md P7. A TTA loop's likeliest failure is to run and
adapt NOTHING (the planner keeps its own stale copy) or the wrong thing (a
masked weight drifting on momentum or decay, BN statistics moving under a
frozen encoder). On the published weights (`pixi run dump-lewm-ref`), Apple
GPU:

  1. MASK: 2 steps adapting `pred_raw.5.` + `pred.` with BN in eval mode —
     every tensor outside the subset, BN running stats included, is
     bit-identical; some inside moved; the predictor qkv bias is still 0;
  2. BN TRAIN mode does move the running statistics (the switch is wired);
  3. RESTORE: the base snapshot comes back bit-exact (params + state);
  4. PLANNER SYNC: a planner re-synced from the adapted trainer rolls out
     bit-identically to one loaded independently from a dump of that
     trainer, and differently from the unadapted planner;
  5. WINDOWS: `tta_windows` aligns frames and the actions taken FROM them;
  6. STOP-GRAD TARGET (`set_stop_grad_target`): adapting `pred_raw.5.` +
     `emb.0.6.` with clipping off, detaching the target leaves the predictor's
     update bit-identical (its gradient never flows through the target) and
     changes the encoder projector's (it loses the target-side gradient).

    pixi run -e apple mojo run -I . tests/experimental/lewm/ref/test_ref_tta.mojo
"""

from std.math import abs
from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_trainer import (
    LeWMRefTrainer, TrainerSnapshot, FillFromSnapshot,
)
from noeira.experimental.lewm.ref_rollout import LeWMRefRollout, REF_EMB, REF_ACT
from noeira.experimental.lewm.paper_pairs import tta_windows, PAIR_HW


comptime TB = 4
comptime TMP = "/tmp/lewm_tta_test_dump"


def _diff(a: TrainerSnapshot, b: TrainerSnapshot, keep: List[String]) raises -> Tuple[Int, Int, Int]:
    """(tensors outside `keep` that differ, inside that differ, inside total)."""
    var out_diff = 0
    var in_diff = 0
    var in_tot = 0
    for i in range(len(a.names)):
        var k = b.find(a.names[i])
        if k < 0:
            raise Error("snapshot lacks " + a.names[i])
        var same = True
        for j in range(len(a.values[i])):
            if a.values[i][j] != b.values[k][j]:
                same = False
                break
        var inside = False
        for p in keep:
            if a.names[i].startswith(p):
                inside = True
        if inside:
            in_tot += 1
            if not same:
                in_diff += 1
        elif not same:
            out_diff += 1
            print("     moved outside the subset:", a.names[i])
    return (out_diff, in_diff, in_tot)


def main() raises:
    var dump = String("/tmp/lewm_ref")
    var rd = RefDump(dump)
    var c = DeviceContext()
    var ctx = Optional(c)
    var fails = 0
    print("P7 TTA machinery (published weights, Apple GPU)")
    var keep: List[String] = ["pred_raw.5.", "pred."]
    var tr = LeWMRefTrainer["gpu", TB](ctx, lr=5e-4, wd=0.0, max_norm=1.0, dropout=False)
    _ = tr.load(dump)
    tr.set_keep(keep)
    tr.reset_optimizer()
    var base = tr.export_params(List[String](), True)
    var pix = rd.get(String("steps.0.pixels"))
    var act = rd.get(String("steps.0.action"))

    # 1. mask, BN eval
    tr.set_bn_training(False)
    _ = tr.train_step(pix, act)
    _ = tr.train_step(rd.get(String("steps.1.pixels")), rd.get(String("steps.1.action")))
    var after = tr.export_params(List[String](), True)
    var d = _diff(base, after, keep)
    var qkv_zero = True
    for i in range(len(after.names)):
        if after.names[i].startswith("pred_raw.") and after.names[i].endswith(".attn.0.0.bias"):
            for v in after.values[i]:
                if v != Scalar[DT](0):
                    qkv_zero = False
    print("  1. mask (BN eval): outside moved", d[0], "| inside moved", d[1], "of", d[2],
          "| qkv bias still 0:", qkv_zero)
    if d[0] != 0 or d[1] == 0 or not qkv_zero:
        fails += 1

    # 4. planner sync (on the adapted trainer, before anything else moves it)
    seed(3)
    var start = List[Scalar[DT]]()
    for _ in range(REF_EMB):
        start.append(Scalar[DT](random_float64(-1, 1)))
    var cand = List[Scalar[DT]]()
    for _ in range(8 * 5 * REF_ACT):
        cand.append(Scalar[DT](random_float64(-1, 1)))
    var roll_base = LeWMRefRollout["gpu", 8, 5](dump, ctx)
    var r_base = roll_base.rollout(start, cand)
    _ = tr.save_dump(String(TMP))
    var roll_ind = LeWMRefRollout["gpu", 8, 5](String(TMP), ctx)
    var r_ind = roll_ind.rollout(start, cand)
    var roll_sync = LeWMRefRollout["gpu", 8, 5](dump, ctx)
    var snap = tr.export_params(keep, False)
    var fp = FillFromSnapshot(snap, String("pred_raw."))
    var fa = FillFromSnapshot(snap, String("act_emb."))
    var fe = FillFromSnapshot(snap, String("x_pe."))
    var fl = FillFromSnapshot(snap, String("pred_ln."))
    var fpp = FillFromSnapshot(snap, String("pred."))
    roll_sync.sync_from(fp, fa, fe, fl, fpp)
    var r_sync = roll_sync.rollout(start, cand)
    var n_eq = 0
    var n_base = 0
    for i in range(len(r_ind)):
        if r_sync[i] == r_ind[i]:
            n_eq += 1
        if r_sync[i] != r_base[i]:
            n_base += 1
    print("  4. planner sync: re-synced == independently loaded on", n_eq, "of", len(r_ind),
          "values; differs from the unadapted planner on", n_base, "(", fp.n + fpp.n, "tensors synced )")
    if n_eq != len(r_ind) or n_base == 0:
        fails += 1

    # 3. restore
    _ = tr.restore(base)
    var back = tr.export_params(List[String](), True)
    var none = List[String]()
    var dr = _diff(base, back, none)
    print("  3. restore: tensors differing from the base", dr[0])
    if dr[0] != 0:
        fails += 1

    # 2. BN train mode moves the running statistics
    tr.reset_optimizer()
    tr.set_bn_training(True)
    _ = tr.train_step(pix, act)
    tr.set_bn_training(False)
    var after_t = tr.export_params(List[String](), True)
    var bn_moved = 0
    for i in range(len(base.names)):
        if "running_" in base.names[i]:
            var k = after_t.find(base.names[i])
            for j in range(len(base.values[i])):
                if base.values[i][j] != after_t.values[k][j]:
                    bn_moved += 1
                    break
    print("  2. BN train mode: running-stat tensors moved", bn_moved, "of 4")
    if bn_moved != 4:
        fails += 1

    # 6. stop-grad target: predictor update unchanged, encoder projector's not
    var keep6: List[String] = ["pred_raw.5.", "emb.0.6."]
    tr.set_keep(keep6)
    tr.max_norm = 1e9  # clipping couples every tensor through the global norm
    var upd = List[TrainerSnapshot]()
    for sg in range(2):
        _ = tr.restore(base)
        tr.reset_optimizer()
        tr.set_stop_grad_target(sg == 1)
        _ = tr.train_step(pix, act)
        upd.append(tr.export_params(keep6, False))
    tr.set_stop_grad_target(False)
    var pred_same = True
    var enc_moved = 0
    var pred_moved = 0
    for i in range(len(upd[0].names)):
        var k = upd[1].find(upd[0].names[i])
        var same = True
        for j in range(len(upd[0].values[i])):
            if upd[0].values[i][j] != upd[1].values[k][j]:
                same = False
                break
        var b = base.find(upd[0].names[i])
        var moved = False
        for j in range(len(upd[0].values[i])):
            if upd[0].values[i][j] != base.values[b][j]:
                moved = True
                break
        if upd[0].names[i].startswith("pred_raw."):
            if not same:
                pred_same = False
            if moved:
                pred_moved += 1
        elif not same:
            enc_moved += 1
    print("  6. stop-grad target: predictor update identical", pred_same, "(", pred_moved,
          "tensors moved ) | encoder-projector tensors that differ", enc_moved)
    if not pred_same or pred_moved == 0 or enc_moved == 0:
        fails += 1

    # 5. windows: frame j is all j, action block j is j*100 + i
    var frames = List[List[Scalar[DT]]]()
    var acts = List[List[Scalar[DT]]]()
    for j in range(6):
        frames.append(List[Scalar[DT]](length=PAIR_HW * 3, fill=Scalar[DT](j)))
        if j < 5:
            var blk = List[Scalar[DT]]()
            for i in range(10):
                blk.append(Scalar[DT](j * 100 + i))
            acts.append(blk^)
    var w = tta_windows[4, 4](frames, acts)
    var ok_w = True
    var want_start: List[Int] = [2, 1, 0, 2]  # 3 windows, the 4th cycles
    for b in range(4):
        for t in range(4):
            var j = want_start[b] + t
            # pixel 0, channel R: (j/255 - 0.485)/0.229
            var px = Float64(w[0][((b * 4 + t) * 3) * PAIR_HW])
            var jj = Int((px * 0.229 + 0.485) * 255.0 + 0.5)
            var a0 = Float64(w[1][(b * 4 + t) * 10])
            var want_a = Float64(j * 100) if j < 5 else 0.0
            if jj != j or a0 != want_a:
                ok_w = False
    print("  5. windows: frames and actions aligned:", ok_w)
    if not ok_w:
        fails += 1
    if fails > 0:
        raise Error("FAIL: " + String(fails) + " check(s)")
    print("PASS")
