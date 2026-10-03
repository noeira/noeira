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
          (`set_keep` subset, wd 0, a fresh Adam per episode, dropout off,
          BN per `--tta-bn`), then the planner's predictor side (and encoder,
          when adapted) re-synced, and the goal re-encoded.

Knobs (defaults = AdaJEPA's): `--receding R` blocks executed per replan (1 =
AdaJEPA's one chunk per replan; 5 = the paper protocol of column R / M, which
leaves 2 replans in a budget of 50 — too few to adapt); `--budget` env steps;
`--subset pred | predlast_enclast | all`; `--tta-lr` (5e-5, the training LR);
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
    LeWMRefRollout, RefEncoder, encode_ref, cem_step, REF_EMB, REF_ACT,
)
from noeira.experimental.lewm.ref_trainer import (
    LeWMRefTrainer, TrainerSnapshot, FillFromSnapshot, REF_T,
)
from noeira.experimental.lewm.paper_pairs import (
    PairEnv, PAIR_HW, imagenet_from_hwc255, render_frame, gauss, pair_success,
    VisualShift, shift_frame, tta_windows,
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
    var budget: Int
    var tta_steps: Int
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
    """Mean loss on the newest windows just BEFORE each adaptation step (the
    adapt arm's health signal; 0 when nothing was adapted)."""


def _keep_for(subset: String) raises -> List[String]:
    if subset == "pred":
        return ["pred_raw.", "pred_ln.", "pred.", "x_pe.", "act_emb."]
    if subset == "predlast_enclast":
        return ["pred_raw.5.", "pred.", "emb.0.6."]
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
    var out = Outcome(False, 0, 0, 0.0)
    var step = 0
    while step < cfg.budget:
        var start_emb = encode_ref[TARGET, 1](enc, imagenet_from_hwc255(frames[len(frames) - 1], 0), ctx)
        var mean = List[Scalar[DT]](length=A, fill=Scalar[DT](0))
        var std = List[Scalar[DT]](length=A, fill=Scalar[DT](1))
        for it in range(ITERS):
            var noise = gauss(cfg.seed * 1000003 + UInt64(e), S * A, UInt64((out.replans * ITERS + it) * S * A))
            var st = cem_step[TARGET, S, HORIZON, K](roll, start_emb, goal_emb, mean, std, noise)
            mean = st.mean.copy()
            std = st.std.copy()
        out.replans += 1
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
                if pair_success(env, goal_state, e):
                    out.ok = True
            if len(block) < REF_ACT:
                break  # a block cut by the budget is not a model step
            acts.append(block^)
            var fr = render_frame(env)
            shift_frame(fr, 0, cfg.shift, UInt64(e) * 7919 + UInt64(len(frames)) * 13 + 3)
            frames.append(fr^)
        # adapt on the episode so far, then plan with the updated model
        if adapt and len(frames) >= REF_T and step < cfg.budget:
            var bt = tta_windows[TB, REF_T](frames, acts)
            tr.set_bn_training(cfg.tta_bn_train)
            var pre = tr.loss_of(bt[0], bt[1])
            out.pre_loss += pre.loss
            for _ in range(cfg.tta_steps):
                _ = tr.train_step(bt[0], bt[1])
            tr.set_bn_training(False)
            _ = _sync_planner(tr, enc, roll, cfg, ctx, False)
            if cfg.touches_encoder or cfg.tta_bn_train:
                goal_emb = encode_ref[TARGET, 1](enc, imagenet_from_hwc255(goal, 0), ctx)
            out.adapts += 1
    if out.adapts > 0:
        out.pre_loss /= Float64(out.adapts)
    return out^


def main() raises:
    var dump = String("/workspace/lewm_train/epoch_7")
    var fixture = getenv("HOME") + "/.cache/noeira/lewm_pusht/session_a/out/fixture"
    var n_eps = 50
    var first_pair = 0
    var seed: UInt64 = 0
    var receding = 1
    var budget = 50
    var arms = String("both")
    var subset = String("pred")
    var tta_lr = 5e-5
    var tta_steps = 1
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
        elif a == "--budget":
            budget = Int(String(args[i + 1])); i += 1
        elif a == "--arms":
            arms = String(args[i + 1]); i += 1
        elif a == "--subset":
            subset = String(args[i + 1]); i += 1
        elif a == "--tta-lr":
            tta_lr = Float64(String(args[i + 1])); i += 1
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
    var keep = _keep_for(subset)
    var touches_enc = len(keep) == 0
    for k in keep:
        if k.startswith("emb."):
            touches_enc = True
    var cfg = Cfg(receding, budget, tta_steps, tta_bn == "train", keep^, touches_enc,
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
    var base = tr.export_params(List[String](), True)
    print("AdaJEPA on", dump, ":", n_eps, "pairs from", first_pair, "; receding", receding,
          "budget", budget, "; subset", subset, "(", len(cfg.keep), "prefixes ) lr", tta_lr,
          "steps", tta_steps, "BN", tta_bn, "lambda", lam, "; shift", shift_spec,
          "; base snapshot", len(base.names), "tensors")

    var n_ok = List[Int](length=2, fill=0)
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
            line += "  " + (String("frozen ") if arm == 0 else String("adapt ")) + ("ok  " if o.ok else "FAIL")
            line += " (" + String(o.replans) + " replans"
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
