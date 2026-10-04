"""P7 — AdaJEPA test-time adaptation on the reference LeWM (PushT pairs).

docs/LEWM_REOPEN_PLAN.md P7, docs/ADAJEPA_LEWM_TTA_PLAN.md. AdaJEPA (Wang,
Bounou, LeCun, Ren 2026) wraps the planner in a test-time-adaptation loop:
after each executed action chunk, ONE gradient step of the pretraining loss on
the episode's latest transitions, on a parameter subset, then replan with the
updated model; every episode starts from the pretrained weights.

Two arms per pair, on identical pairs and identical CEM noise:

  frozen  the checkpoint as is (`lewm_pusht_column_m.mojo`'s protocol);
  adapt   after each executed block: the latest windows of 4 frames (history
          3 + 1, frameskip 5) with the executed actions (z-scored with the
          TRAINING normaliser — the model's space, not the planner's
          population-std one), `--tta-steps` steps of `LeWMRefTrainer`
          (`set_keep` subset, wd 0, Adam per `--adam`, dropout off, BN per
          `--tta-bn`), then the planner's predictor side (and encoder, when
          adapted) re-synced, and the goal re-encoded.

Both arms log the PREDICTION loss on those windows at every replan (the
frozen arm measures without updating): the direct test of whether adapting
makes the model better on the episode's new data. The SIGReg term dominates
the total loss (~0.33 vs a prediction loss ~0.004), so the total hides it.

The official implementation (references/adajepa-main, DINO-WM based) sets the
defaults: one block per replan, warm start, budget 100 (`max_iter: 20`), the
predictor's last block + its final norm and the encoder's last submodule
(`predlast_enclast`), a fresh Adam at EVERY adaptation (`_make_optimizer`
per `finetune` call). It also adapts with the predictor in train mode
(dropout 0.1), and its loss is the 1-step latent MSE alone. Here dropout stays
off and the loss is LeWM's own (prediction + λ·SIGReg, `--lambda 0` = MSE).

Knobs (defaults = the official AdaJEPA's): `--receding R` blocks executed per replan (1 =
AdaJEPA's one chunk per replan; 5 = the paper protocol of column R / M, which
leaves 2 replans in a budget of 50 — too few to adapt); `--cold` restarts every
replan's CEM at mean 0 instead of swm's default warm start (`warm_start=True`:
the previous plan's unexecuted blocks, zero-padded — moot at R = 5, where
nothing is left; cold at R = 1 executes the first block of an unrelated plan
each time: epoch_0 frozen 34/50 at R 5 fell to 11/50 cold, 16/50 warm — R 1 at
budget 50 stays the weaker planner either way); `--cost staged[:b] | all[:b] |
last` the planning cost (`PlanCost`; default AdaJEPA's `staged` — LeWM's own
last-step cost PROCRASTINATES at R 1: epoch_7 frozen, 9 pairs, budget 50,
last 3/9, all:2 8/9, staged 9/9, R 5 9/9); `--budget` env steps;
`--subset predlast_enclast | pred | all`; `--adam per-adapt | per-episode`;
`--tta-lr` (5e-5, LeWM's training LR — AdaJEPA's rule);
`--tta-bn eval | train`; `--lambda` SIGReg weight in the adaptation loss;
`--shift none | noise:σ | dark:gain | swap` (E2: applied to EVERY observed
frame, the dataset start and goal frames included); `--arms both | frozen |
adapt`.

    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_adajepa.mojo \\
        --dump <run>/epoch_7 --fixture <session_a>/out/fixture --episodes 50 \\
        --receding 1 --shift dark:0.5
"""

from std.sys import argv
from std.os import getenv
from std.math import isnan
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.act.refload import RefDump
from noeira.envs.pusht import PushTAction
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.ref_rollout import (
    LeWMRefRollout, RefEncoder, encode_ref, cem_step, PlanCost, REF_EMB, REF_ACT,
)
from noeira.experimental.lewm.ref_trainer import (
    LeWMRefTrainer, TrainerSnapshot, FillFromSnapshot, REF_T,
)
from noeira.experimental.lewm.paper_pairs import (
    PairEnv, PAIR_HW, imagenet_from_hwc255, render_frame, gauss,
    VisualShift, shift_frame, tta_windows, pair_margin,
)


comptime TARGET = "gpu"
comptime S = 300
comptime K = 30
comptime ITERS = 30
comptime HORIZON = 5
comptime FRAMESKIP = 5
comptime A = HORIZON * REF_ACT
comptime TB = 4
"""Adaptation batch: the latest TB windows of the episode (cycled while the
episode has fewer)."""
comptime FRAME = PAIR_HW * 3


@fieldwise_init
struct Cfg(Copyable, Movable):
    var receding: Int
    var warm_start: Bool
    var cost: PlanCost
    var budget: Int
    var tta_steps: Int
    var adam_per_adapt: Bool
    var tta_bn_train: Bool
    var keep: List[String]
    var touches_encoder: Bool
    var shift: VisualShift
    var seed: UInt64


@fieldwise_init
struct Outcome(Copyable, Movable):
    var ok: Bool
    var replans: Int
    var adapts: Int
    var pre_loss: Float64
    """Mean total loss on the newest windows just BEFORE each adaptation
    step (0 when nothing was adapted)."""
    var pred_loss: Float64
    """Mean PREDICTION loss on the newest windows at each replan, before
    any update — measured in both arms."""
    var measured: Int
    var margin: Float64
    """min over the episode of `pair_margin` (< 1 = success)."""


def _keep_for(subset: String) raises -> List[String]:
    if subset == "pred":
        return ["pred_raw.", "pred_ln.", "pred.", "x_pe.", "act_emb."]
    if subset == "predlast_enclast":
        return ["pred_raw.5.", "pred_ln.", "pred.", "emb.0.6."]
    if subset == "all":
        return List[String]()
    raise Error("unknown --subset " + subset + " (pred | predlast_enclast | all)")


def _sync_planner(
    mut tr: LeWMRefTrainer[TARGET, TB], mut enc: RefEncoder,
    mut roll: LeWMRefRollout[TARGET, S, HORIZON], cfg: Cfg, ctx: Optional[DeviceContext],
    full: Bool,
) raises -> Int:
    """The planner's copies <- the trainer: the adapted subset (all tensors
    when `full`), with every BN running statistic when BN adapts in train
    mode (or `full`)."""
    var keep = List[String]() if full else cfg.keep.copy()
    var snap = tr.export_params(keep, cfg.tta_bn_train or full)
    var fp = FillFromSnapshot(snap, String("pred_raw."))
    var fa = FillFromSnapshot(snap, String("act_emb."))
    var fe = FillFromSnapshot(snap, String("x_pe."))
    var fl = FillFromSnapshot(snap, String("pred_ln."))
    var fpp = FillFromSnapshot(snap, String("pred."))
    roll.sync_from(fp, fa, fe, fl, fpp)
    var n = fp.n + fa.n + fe.n + fl.n + fpp.n
    if full or cfg.touches_encoder or cfg.tta_bn_train:
        var fenc = FillFromSnapshot(snap, String("emb.0."))
        enc.for_each_param[TARGET](fenc, ctx)
        enc.for_each_state[TARGET](fenc, ctx)
        n += fenc.n
    return n


def _episode(
    e: Int, adapt: Bool, cfg: Cfg,
    mut env: PairEnv, mut enc: RefEncoder, mut roll: LeWMRefRollout[TARGET, S, HORIZON],
    mut tr: LeWMRefTrainer[TARGET, TB], ctx: Optional[DeviceContext],
    start_pix: List[Scalar[DT]], goal_pix: List[Scalar[DT]], goal_state: List[Scalar[DT]],
    a_mean: List[Scalar[DT]], a_scale: List[Scalar[DT]], a_std_train: List[Scalar[DT]],
) raises -> Outcome:
    # observations: the dataset start frame, then our renders — all shifted
    var first = List[Scalar[DT]](capacity=FRAME)
    for i in range(FRAME):
        first.append(start_pix[e * FRAME + i])
    shift_frame(first, 0, cfg.shift, UInt64(e) * 7919 + 1)
    var goal = List[Scalar[DT]](capacity=FRAME)
    for i in range(FRAME):
        goal.append(goal_pix[e * FRAME + i])
    shift_frame(goal, 0, cfg.shift, UInt64(e) * 7919 + 2)
    var goal_emb = encode_ref[TARGET, 1](enc, imagenet_from_hwc255(goal, 0), ctx)
    var frames = List[List[Scalar[DT]]]()
    frames.append(first^)
    var acts = List[List[Scalar[DT]]]()
    var out = Outcome(False, 0, 0, 0.0, 0.0, 0, 1e9)
    var step = 0
    var init = List[Scalar[DT]](length=A, fill=Scalar[DT](0))
    while step < cfg.budget:
        var start_emb = encode_ref[TARGET, 1](enc, imagenet_from_hwc255(frames[len(frames) - 1], 0), ctx)
        var mean = init.copy()
        var std = List[Scalar[DT]](length=A, fill=Scalar[DT](1))
        var step_w = cfg.cost.weights(HORIZON, out.replans)
        for it in range(ITERS):
            var noise = gauss(cfg.seed * 1000003 + UInt64(e), S * A, UInt64((out.replans * ITERS + it) * S * A))
            var st = cem_step[TARGET, S, HORIZON, K](roll, start_emb, goal_emb, mean, std, noise, step_w)
            mean = st.mean.copy()
            std = st.std.copy()
        out.replans += 1
        # swm warm start: the next replan's mean = this plan's unexecuted blocks, zero-padded
        var kept = A - cfg.receding * REF_ACT
        for j in range(A):
            init[j] = mean[cfg.receding * REF_ACT + j] if cfg.warm_start and j < kept else Scalar[DT](0)
        for blk in range(cfg.receding):
            var block = List[Scalar[DT]](capacity=REF_ACT)
            for k in range(FRAMESKIP):
                if step >= cfg.budget:
                    break
                var ax = Float64(mean[blk * REF_ACT + 2 * k + 0]) * Float64(a_scale[0]) + Float64(a_mean[0])
                var ay = Float64(mean[blk * REF_ACT + 2 * k + 1]) * Float64(a_scale[1]) + Float64(a_mean[1])
                # the executed action in the model's (training) normalisation
                block.append(Scalar[DT]((ax - Float64(a_mean[0])) / Float64(a_std_train[0])))
                block.append(Scalar[DT]((ay - Float64(a_mean[1])) / Float64(a_std_train[1])))
                var ag = env.agent_pos()
                _ = env.step(PushTAction[DType.float32](
                    Scalar[DType.float32](Float64(ag[0]) + 100.0 * ax),
                    Scalar[DType.float32](Float64(ag[1]) + 100.0 * ay),
                ))
                step += 1
                var mg = pair_margin(env, goal_state, e)
                out.margin = min(out.margin, mg)
                if mg < 1.0:
                    out.ok = True
            if len(block) < REF_ACT:
                break  # a block cut by the budget is not a model step
            acts.append(block^)
            var fr = render_frame(env)
            shift_frame(fr, 0, cfg.shift, UInt64(e) * 7919 + UInt64(len(frames)) * 13 + 3)
            frames.append(fr^)
        # measure (both arms), adapt (adapt arm), then plan with the updated model
        if len(frames) >= REF_T and step < cfg.budget:
            var bt = tta_windows[TB, REF_T](frames, acts)
            var pre = tr.loss_of(bt[0], bt[1])
            out.pred_loss += pre.pred_loss
            out.measured += 1
            if not adapt:
                continue
            out.pre_loss += pre.loss
            if cfg.adam_per_adapt:
                tr.reset_optimizer()
            tr.set_bn_training(cfg.tta_bn_train)
            for _ in range(cfg.tta_steps):
                _ = tr.train_step(bt[0], bt[1])
            tr.set_bn_training(False)
            _ = _sync_planner(tr, enc, roll, cfg, ctx, False)
            if cfg.touches_encoder or cfg.tta_bn_train:
                goal_emb = encode_ref[TARGET, 1](enc, imagenet_from_hwc255(goal, 0), ctx)
            out.adapts += 1
    if out.adapts > 0:
        out.pre_loss /= Float64(out.adapts)
    if out.measured > 0:
        out.pred_loss /= Float64(out.measured)
    return out^


def main() raises:
    var dump = String("/workspace/lewm_train/epoch_7")
    var fixture = getenv("HOME") + "/.cache/noeira/lewm_pusht/session_a/out/fixture"
    var n_eps = 50
    var first_pair = 0
    var seed: UInt64 = 0
    var receding = 1
    var warm_start = True
    var cost_spec = String("staged")
    var budget = 100
    var arms = String("both")
    var subset = String("predlast_enclast")
    var tta_lr = 5e-5
    var tta_steps = 1
    var adam = String("per-adapt")
    var tta_bn = String("eval")
    var lam = 0.09
    var shift_spec = String("none")
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--dump":
            dump = String(args[i + 1]); i += 1
        elif a == "--fixture":
            fixture = String(args[i + 1]); i += 1
        elif a == "--episodes":
            n_eps = Int(String(args[i + 1])); i += 1
        elif a == "--first":
            first_pair = Int(String(args[i + 1])); i += 1
        elif a == "--seed":
            seed = UInt64(Int(String(args[i + 1]))); i += 1
        elif a == "--receding":
            receding = Int(String(args[i + 1])); i += 1
        elif a == "--cold":
            warm_start = False
        elif a == "--cost":
            cost_spec = String(args[i + 1]); i += 1
        elif a == "--budget":
            budget = Int(String(args[i + 1])); i += 1
        elif a == "--arms":
            arms = String(args[i + 1]); i += 1
        elif a == "--subset":
            subset = String(args[i + 1]); i += 1
        elif a == "--tta-lr":
            tta_lr = Float64(String(args[i + 1])); i += 1
        elif a == "--adam":
            adam = String(args[i + 1]); i += 1
        elif a == "--tta-steps":
            tta_steps = Int(String(args[i + 1])); i += 1
        elif a == "--tta-bn":
            tta_bn = String(args[i + 1]); i += 1
        elif a == "--lambda":
            lam = Float64(String(args[i + 1])); i += 1
        elif a == "--shift":
            shift_spec = String(args[i + 1]); i += 1
        else:
            raise Error("unknown argument " + a)
        i += 1
    if receding < 1 or receding > HORIZON:
        raise Error("--receding must be 1.." + String(HORIZON))
    if tta_bn != "eval" and tta_bn != "train":
        raise Error("--tta-bn eval | train")
    if adam != "per-adapt" and adam != "per-episode":
        raise Error("--adam per-adapt | per-episode")
    var keep = _keep_for(subset)
    var touches_enc = len(keep) == 0
    for k in keep:
        if k.startswith("emb."):
            touches_enc = True
    var cfg = Cfg(receding, warm_start, PlanCost.parse(cost_spec), budget, tta_steps, adam == "per-adapt", tta_bn == "train", keep^, touches_enc,
                  VisualShift.parse(shift_spec), seed)
    var run_frozen = arms == "both" or arms == "frozen"
    var run_adapt = arms == "both" or arms == "adapt"

    var fx = RefDump(fixture)
    var start_state = fx.get(String("pairs.start_state"))
    var goal_state = fx.get(String("pairs.goal_state"))
    var start_pix = fx.get(String("pairs.start_pixels"))
    var goal_pix = fx.get(String("pairs.goal_pixels"))
    var a_mean = fx.get(String("stats.action_mean"))
    var a_scale = fx.get(String("stats.action_scale_eval"))
    var a_std_train = fx.get(String("stats.action_std_train"))
    n_eps = min(n_eps, len(start_state) // 7 - first_pair)

    var c = DeviceContext()
    var ctx = Optional(c)
    var enc = RefEncoder.make[TARGET, Kaiming](ctx)
    _ = load_ref[TARGET](enc, dump, String("emb.0."), ctx)
    var roll = LeWMRefRollout[TARGET, S, HORIZON](dump, ctx)
    var tr = LeWMRefTrainer[TARGET, TB](ctx, lr=tta_lr, wd=0.0, max_norm=1.0,
                                       sigreg_lambda=lam, dropout=False)
    _ = tr.load(dump)
    tr.set_keep(cfg.keep)
    tr.set_bn_training(False)  # measuring (both arms) must not move BN's running stats
    var base = tr.export_params(List[String](), True)
    print("AdaJEPA on", dump, ":", n_eps, "pairs from", first_pair, "; receding", receding, "warm" if warm_start else "cold", "cost", cost_spec,
          "budget", budget, "; subset", subset, "(", len(cfg.keep), "prefixes ) lr", tta_lr,
          "steps", tta_steps, "Adam", adam, "BN", tta_bn, "lambda", lam, "; shift", shift_spec,
          "; base snapshot", len(base.names), "tensors")

    var n_ok = List[Int](length=2, fill=0)
    var pred_sum = List[Float64](length=2, fill=0.0)
    var margin_sum = List[Float64](length=2, fill=0.0)
    var t_all = perf_counter_ns()
    for ei in range(n_eps):
        var e = first_pair + ei
        var line = String("  pair ") + String(e)
        for arm in range(2):
            if (arm == 0 and not run_frozen) or (arm == 1 and not run_adapt):
                continue
            var env = PairEnv(seed=UInt64(e))
            _ = env.set_state(
                Scalar[DType.float32](start_state[e * 7 + 0]), Scalar[DType.float32](start_state[e * 7 + 1]),
                Scalar[DType.float32](start_state[e * 7 + 2]), Scalar[DType.float32](start_state[e * 7 + 3]),
                Scalar[DType.float32](start_state[e * 7 + 4]),
                agent_vx=Scalar[DType.float32](start_state[e * 7 + 5]),
                agent_vy=Scalar[DType.float32](start_state[e * 7 + 6]),
                settle=True,
            )
            if arm == 1:
                tr.reset_optimizer()
            var t0 = perf_counter_ns()
            var o = _episode(e, arm == 1, cfg, env, enc, roll, tr, ctx, start_pix, goal_pix,
                             goal_state, a_mean, a_scale, a_std_train)
            if o.ok:
                n_ok[arm] += 1
            pred_sum[arm] += o.pred_loss
            margin_sum[arm] += o.margin
            line += "  " + (String("frozen ") if arm == 0 else String("adapt ")) + ("ok  " if o.ok else "FAIL")
            line += " (margin " + String(Float32(o.margin)) + ", " + String(o.replans) + " replans, pred loss " + String(Float32(o.pred_loss))
            if arm == 1:
                line += ", " + String(o.adapts) + " adapts, pre-adapt loss " + String(Float32(o.pre_loss))
            line += ", " + String(Float32(Float64(perf_counter_ns() - t0) / 1e9)) + " s)"
            if arm == 1 and o.adapts > 0:
                # back to the pretrained weights for the next pair
                _ = tr.restore(base)
                _ = _sync_planner(tr, enc, roll, cfg, ctx, True)
        print(line)
    print("frozen:", n_ok[0], "/", n_eps, "   adapt:", n_ok[1], "/", n_eps,
          "   total", Float32(Float64(perf_counter_ns() - t_all) / 1e9), "s")
    print("mean best margin (< 1 = success) — frozen:", Float32(margin_sum[0] / Float64(n_eps)),
          "  adapt:", Float32(margin_sum[1] / Float64(n_eps)))
    print("mean pred loss on the episode's windows — frozen:", Float32(pred_sum[0] / Float64(n_eps)),
          "  adapt (before each update):", Float32(pred_sum[1] / Float64(n_eps)))
