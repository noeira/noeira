"""A4 — FB-CPR on GPU, d = 128, from the collected walker store (24-D obs).

`docs/BFM_ZERO_SHOT_RL.md` §18.3 A4. This is `fb_train_gpu.mojo` with the
three CPR pieces on top and NOTHING else changed: the same store, the same
24-D observation through `obs_at`, the same sampler, the same `z` sphere,
the same FB trainer inside (`FBCPRTrainer` owns an unchanged `FBTrainer`;
`test_fb_cpr_smoke.mojo` [3] holds the FB half bit-identical at
`reg_coeff 0`). So an arm of this script against the 24-D `pair` base
differs by the CPR terms and by nothing else.

    pixi run -e nvidia mojo run -I . examples/fb/fb_train_cpr_gpu.mojo --tag cpr

## What CPR adds here

* **Expert set** = the high-return tail of the SAME store (§15.5): per
  task, the episodes whose `ep_return` is in the top `--expert-frac`
  (default 0.2, i.e. ~100 of 504 episodes per task). The store tags every
  row with `policy_step`, `ep_return` and `task`, so the split is a filter,
  not a second file. The thresholds are printed.
* **Expert windows**: `SEQ = 8` consecutive rows of one expert episode; the
  window's `z_e = project(mean_j B(s'_{t+j}))` (`encode_expert`), the
  reference's sequence encoding. Window starts are drawn from a table of
  VALID starts built from the episode index (a window never crosses an
  episode end), one draw per window, expanded on device.
* **z mixture** (BFM-Zero `sample_mixed_z`): `--p-goal 0.2` rows are
  `B(s+)`, `--p-expert 0.6` are expert window encodings, the rest uniform on
  the sphere. Offline every row is relabelled, so this IS the batch's `z`.
* **D(s, z)**: `Linear(OBS+D → 1024) → LayerNorm → Tanh → 2×(Linear → ReLU)
  → Linear(1)`, BCE-with-logits, WGAN-GP `--gp 10` on interpolations of
  `[s | z]`, Adam `--lr-d 1e-5`.
* **Q_D** twins, `[s | a | z] → 1`, TD on `r_D = clamp(logit D)`, Adam
  `--lr-q 1e-4`, Polyak 0.005.
* **actor**: `−F·z − reg · |mean F·z| · Q_D`, `--reg 0.01`.

⚠ `--bc` defaults to 1.0 — the base carries BC (§18.6: load-bearing), and
the first arm varies ONE axis. `--bc 0` is the "CPR replaces BC" arm.

Sweep flags: `--steps --ortho --lr-b --bc --seed --tag` as the FB script,
plus `--reg --gp --lr-d --lr-q --expert-frac --p-goal --p-expert`.
Checkpoints: `fb_walker_<tag>_envobs.ckpt.<step>` (the FB nets, loadable
by `fb_eval_walker_online.mojo` unchanged) + `.cpr` sidecar (D, Q_D).
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.ptr import mptr
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.nn.random.box_muller import box_muller_normal_gpu

from noeira.data.store import TrajectoryStore
from noeira.data.resident import ResidentColumn, IDX_DT
from noeira.data.sampler import UniformDeviceSampler

from noeira.core.dotenv import load_dotenv
from noeira.core.logger import CsvLogger, RemoteLogger, CompositeLogger
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import finish_run
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.cuda import CUDAGraph, maybe_capture_replay
from noeira.deep_agents.fb.cpr import FBCPRTrainer, FBCPRLosses
from noeira.deep_agents.fb.trainer import FBLosses
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig
from noeira.deep_agents.fb.kernels import (
    gather_rows_kernel,
    gather_idx_kernel,
    expand_windows_kernel,
    z_mixture3_kernel,
    project_sphere_kernel,
    uniform01_kernel,
    ensure_t,
    _blocks,
)


# ── the dataset ──────────────────────────────────────────────────────────
comptime STORE_PATH: StaticString = "fb_walker_all_sac.h5"
comptime NQ: Int = 9
comptime NV: Int = 9
comptime NACT: Int = 6
# 24-D env observation ONLY (§18.7.5/6: the base). No 18-D switch here.
comptime OBS: Int = DMWalkerModel.OBS_DIM
comptime ScorerEnv = Phyics3dEnv[
    DMWalkerModel, DMWalkerConfig[1.0], DType.float64, False
]

# ── the run ──────────────────────────────────────────────────────────────
comptime D: Int = 128
comptime BATCH: Int = 1024
comptime SEQ: Int = 8
comptime NW: Int = BATCH // SEQ
comptime HID: Int = 1024
comptime D_HID: Int = 1024
comptime TRAIN_STEPS: Int = 300_000
comptime LOG_EVERY: Int = 2000
comptime CKPT_EVERY: Int = 50_000
comptime MAX_GRAD_NORM: Float64 = 1.0
comptime BC_WEIGHT: Float64 = 1.0
comptime ORTHO_WEIGHT: Float64 = 100.0
comptime LR_B: Float64 = 1e-5
comptime REG_COEFF: Float64 = 0.01
comptime GP_COEF: Float64 = 10.0
comptime LR_D: Float64 = 1e-5
comptime LR_Q: Float64 = 1e-4
comptime EXPERT_FRAC: Float64 = 0.2
comptime P_GOAL: Float64 = 0.2
comptime P_EXPERT: Float64 = 0.6
comptime USE_TRAIN_CUDA_GRAPH: Bool = True
# ⚠ NO PATH CONSTANTS HERE ANY MORE. Every path this driver writes comes
# from `RunContext` in `main`, so two runs cannot collide — see
# `core/run.mojo` and docs/PROJECT_LAYER_PLAN.md P0d.
comptime SEED: Int = 20260805

comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D
comptime D_IN = OBS + D

comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
# BFM-Zero `Discriminator`: hidden_dim 1024, hidden_layers 3.
comptime DNet = Sequential[
    Linear[D_IN, D_HID], LayerNorm[D_HID], Tanh[D_HID],
    Linear[D_HID, D_HID], ReLU[D_HID],
    Linear[D_HID, D_HID], ReLU[D_HID],
    Linear[D_HID, 1],
]
comptime QNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, 1]]
comptime Trainer = FBCPRTrainer[
    FNet, BNet, ANet, DNet, QNet, OBS, NACT, D, BATCH, SEQ, "gpu"
]


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _quantile_threshold(mut v: List[Float64], top_frac: Float64) -> Float64:
    """The value above which the top `top_frac` of `v` lies (inclusive)."""
    # insertion sort — a few hundred episodes per task
    for i in range(1, len(v)):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    var k = Int(Float64(len(v)) * (1.0 - top_frac))
    if k >= len(v):
        k = len(v) - 1
    if k < 0:
        k = 0
    return v[k]


def main() raises:
    var train_steps = atol(_flag(String("--steps"), String(TRAIN_STEPS)))
    var ortho_w = atof(_flag(String("--ortho"), String(ORTHO_WEIGHT)))
    var lr_b = atof(_flag(String("--lr-b"), String(LR_B)))
    var bc_w = atof(_flag(String("--bc"), String(BC_WEIGHT)))
    var reg = atof(_flag(String("--reg"), String(REG_COEFF)))
    var gp = atof(_flag(String("--gp"), String(GP_COEF)))
    var lr_d = atof(_flag(String("--lr-d"), String(LR_D)))
    var lr_q = atof(_flag(String("--lr-q"), String(LR_Q)))
    var expert_frac = atof(_flag(String("--expert-frac"), String(EXPERT_FRAC)))
    var p_goal = atof(_flag(String("--p-goal"), String(P_GOAL)))
    var p_expert = atof(_flag(String("--p-expert"), String(P_EXPERT)))
    var tag = _flag(String("--tag"), String("cpr"))
    var seed_v = atol(_flag(String("--seed"), String(SEED)))
    tag = tag + "_envobs"
    # ⚠⚠ ONE OF **FIVE** COPIES OF THIS BLOCK IN THE FB FAMILY, all replaced by
    # `RunContext` together. Deriving three paths from a `--tag` a human has to
    # remember to vary is one forgotten flag away from a run silently
    # overwriting the previous one. The tag survives as the SLUG so sweep arms
    # stay legible in a directory listing; uniqueness comes from the id now.
    var run = RunContext(
        project=String("fb"),
        driver=String("examples/fb/fb_train_cpr_gpu.mojo"),
        slug=String("fb-cpr-") + tag,
        env=String("builtin:dm_control/walker-all"),
        seed=seed_v,
    )
    run.set_tag(tag)
    var csv_path = run.metrics_path()
    print("run:", run.dir)
    if p_goal + p_expert > 1.0 or p_goal < 0.0 or p_expert < 0.0:
        raise Error("--p-goal + --p-expert must lie in [0, 1]")
    if expert_frac <= 0.0 or expert_frac > 1.0:
        raise Error("--expert-frac must lie in (0, 1]")
    print(
        "[0] arm: steps", train_steps, " ortho", ortho_w, " lr_b", lr_b,
        " bc", bc_w, " reg", reg, " gp", gp, " lr_d", lr_d, " lr_q", lr_q,
        "\n         expert_frac", expert_frac, " p_goal", p_goal,
        " p_expert", p_expert, " seed", seed_v, " tag '", tag, "'",
    )

    var ctx = DeviceContext()
    print("[1] loading", STORE_PATH, "...")
    var store = TrajectoryStore(String(STORE_PATH))
    var n_rows = store.n_rows()
    var qpos = ResidentColumn[DType.float32].load(store, String("qpos"))
    var qvel = ResidentColumn[DType.float32].load(store, String("qvel"))
    var action = ResidentColumn[DType.float32].load(store, String("action"))
    var ep_return = ResidentColumn[DType.float32].load(store, String("ep_return"))
    var task_col = ResidentColumn[DType.int32].load(store, String("task"))
    print("      ", n_rows, "rows")
    if n_rows < BATCH * 4:
        raise Error("store too small for BATCH=" + String(BATCH))
    var epochs = Float64(train_steps) * Float64(BATCH) / Float64(n_rows)
    print("       each transition will be seen ~", epochs, "times")
    if epochs > 5000.0:
        raise Error("dataset far too small — see fb_train_gpu.mojo")

    # ── 24-D observation table via obs_at (the eval's producer) ─────────
    var obs_host = Tensor()
    obs_host.ensure(n_rows * OBS)
    var scorer = ScorerEnv()
    _ = scorer.reset()
    var q = List[Float64](length=NQ, fill=0.0)
    var v = List[Float64](length=NV, fill=0.0)
    for r in range(n_rows):
        for k in range(NQ):
            q[k] = Float64(qpos.host[r * NQ + k])
        for k in range(NV):
            v[k] = Float64(qvel.host[r * NV + k])
        var o = scorer.obs_at(q, v)
        for k in range(OBS):
            obs_host.data[r * OBS + k] = Scalar[DT](Float64(o.data[k]))
    var moving = 0
    for k in range(OBS):
        var mn = Float64(1e30)
        var mx = Float64(-1e30)
        for r in range(n_rows):
            var x = Float64(obs_host.data[r * OBS + k])
            if x < mn:
                mn = x
            if x > mx:
                mx = x
        if mx - mn > 1e-6:
            moving += 1
    print("       ENV_OBS: 24-D env observation via obs_at;", moving, "/", OBS, "dims vary")
    if moving < OBS - 2:
        raise Error("ENV_OBS obs table: too few varying dims")
    obs_host.upload(ctx)

    var act_host = Tensor()
    act_host.ensure(n_rows * NACT)
    for i in range(n_rows * NACT):
        act_host.data[i] = Scalar[DT](Float64(action.host[i]))
    act_host.upload(ctx)

    # ── episode-safe next_row (as fb_train_gpu.mojo) ─────────────────────
    var nxt = ctx.enqueue_create_host_buffer[IDX_DT](n_rows)
    for r in range(n_rows):
        var n = r + 1
        if n >= n_rows:
            n = r
        nxt[r] = Scalar[IDX_DT](n)
    var n_eps = store.episodes.n_episodes()
    var marked = 0
    for e in range(n_eps):
        var off = Int(store.episodes.ep_offset[e])
        var ln = Int(store.episodes.ep_len[e])
        if ln <= 0:
            continue
        var last = off + ln - 1
        if last < n_rows:
            nxt[last] = Scalar[IDX_DT](last)
            marked += 1
    if marked != n_eps:
        raise Error("episode index inconsistent with the row count")
    var nxt_dev = ctx.enqueue_create_buffer[IDX_DT](n_rows)
    ctx.enqueue_copy(nxt_dev, nxt)

    # ── the expert set: per-task top `expert_frac` by episode return ────
    # ⚠ Per TASK, not global: `run`'s returns top out near 740 against
    # `stand`'s 1000, so a global quantile would be a stand/walk set with
    # no running in it, and D would certify standing under every z.
    var n_tasks = 0
    for e in range(n_eps):
        var off = Int(store.episodes.ep_offset[e])
        var t_id = Int(task_col.host[off])
        if t_id + 1 > n_tasks:
            n_tasks = t_id + 1
    var thresholds = List[Float64](length=n_tasks, fill=0.0)
    for t_id in range(n_tasks):
        var rets = List[Float64]()
        for e in range(n_eps):
            var off = Int(store.episodes.ep_offset[e])
            if Int(task_col.host[off]) == t_id:
                rets.append(Float64(ep_return.host[off]))
        if len(rets) == 0:
            continue
        thresholds[t_id] = _quantile_threshold(rets, expert_frac)
    var starts_host = List[Scalar[IDX_DT]]()
    var expert_eps = 0
    var expert_rows = 0
    for e in range(n_eps):
        var off = Int(store.episodes.ep_offset[e])
        var ln = Int(store.episodes.ep_len[e])
        var t_id = Int(task_col.host[off])
        if Float64(ep_return.host[off]) < thresholds[t_id]:
            continue
        expert_eps += 1
        expert_rows += ln
        # rows start..start+SEQ-1 and their next rows must all lie in [off, off+ln)
        var last_start = off + ln - SEQ - 1
        for s0 in range(off, last_start + 1):
            starts_host.append(Scalar[IDX_DT](s0))
    var n_starts = len(starts_host)
    print(
        "      expert set: top", expert_frac, "per task ->", expert_eps, "/", n_eps,
        "episodes,", expert_rows, "rows,", n_starts, "valid window starts",
    )
    for t_id in range(n_tasks):
        print("        task", t_id, "return threshold", thresholds[t_id])
    if n_starts < NW * 4:
        raise Error("too few expert window starts: " + String(n_starts))
    var starts_h = ctx.enqueue_create_host_buffer[IDX_DT](n_starts)
    for i in range(n_starts):
        starts_h[i] = starts_host[i]
    var starts_dev = ctx.enqueue_create_buffer[IDX_DT](n_starts)
    ctx.enqueue_copy(starts_dev, starts_h)
    ctx.synchronize()
    print("      uploaded obs/action/next_row/window starts to device")

    # ── device scratch ───────────────────────────────────────────────────
    var idx_s = ctx.enqueue_create_buffer[IDX_DT](BATCH)
    var idx_sn = ctx.enqueue_create_buffer[IDX_DT](BATCH)
    var idx_sp = ctx.enqueue_create_buffer[IDX_DT](BATCH)
    var idx_w = ctx.enqueue_create_buffer[IDX_DT](NW)
    var win_start = ctx.enqueue_create_buffer[IDX_DT](NW)
    var idx_e = ctx.enqueue_create_buffer[IDX_DT](BATCH)
    var idx_en = ctx.enqueue_create_buffer[IDX_DT](BATCH)
    var samp_a = UniformDeviceSampler(n_rows, seed=UInt64(seed_v))
    var samp_b = UniformDeviceSampler(n_rows, seed=UInt64(seed_v) + 977)
    var samp_w = UniformDeviceSampler(n_starts, seed=UInt64(seed_v) + 4242)

    var t = Trainer.make(
        ctx,
        lr=3e-4, lr_b=lr_b, lr_d=lr_d, lr_q=lr_q,
        gamma=0.98, tau=0.01, tau_q=0.005,
        ortho_weight=ortho_w, max_grad_norm=MAX_GRAD_NORM, bc_weight=bc_w,
        reg_coeff=reg, gp_coef=gp, seed=UInt64(seed_v) + 13,
    )
    t.ensure_sized()

    var gauss = Tensor()
    var pick = Tensor()
    ensure_t["gpu"](gauss, BATCH * D, ctx)
    ensure_t["gpu"](pick, BATCH * 2, ctx)
    var rng_off = UInt64(1)

    # ── logging ──────────────────────────────────────────────────────────
    var env_vars = load_dotenv()
    var logger = CompositeLogger(
        CsvLogger(csv_path, buffer_size=64),
        RemoteLogger(
            server_url=env_vars.get("NOEIRA_CLOUD_URL", ""),
            run_name=run.name(),
            run_id=run.id,
            buffer_size=64,
            api_key=env_vars.get("NOEIRA_CLOUD_API_KEY", ""),
        ),
    )
    logger.set_config("algorithm", "FB-CPR")
    logger.set_config("env", "dm_control/walker-all")
    logger.set_config("store", String(STORE_PATH))
    logger.set_config("rows", String(n_rows))
    logger.set_config("d", String(D))
    logger.set_config("batch", String(BATCH))
    logger.set_config("seq", String(SEQ))
    logger.set_config("hidden", String(HID))
    logger.set_config("train_steps", String(train_steps))
    logger.set_config("max_grad_norm", String(MAX_GRAD_NORM))
    logger.set_config("bc_weight", String(bc_w))
    logger.set_config("ortho_weight", String(ortho_w))
    logger.set_config("lr_b", String(lr_b if lr_b >= 0.0 else 3e-4))
    logger.set_config("reg_coeff", String(reg))
    logger.set_config("gp_coef", String(gp))
    logger.set_config("lr_d", String(lr_d))
    logger.set_config("lr_q", String(lr_q))
    logger.set_config("expert_frac", String(expert_frac))
    logger.set_config("expert_episodes", String(expert_eps))
    logger.set_config("p_goal", String(p_goal))
    logger.set_config("p_expert", String(p_expert))
    logger.set_config("env_obs", "True")
    logger.set_config("tag", tag)
    logger.set_config("seed", String(seed_v))
    logger.set_config("cuda_graph", String(USE_TRAIN_CUDA_GRAPH))
    logger.set_config("epochs_over_dataset", String(epochs))

    var train_graph = Optional[CUDAGraph](None)
    comptime SQRT_D = sqrt(Float64(D))
    # ⚠ AFTER the config, before step 0 — see `core/run.register_run`.
    register_run(run, logger)
    # The uplink: `final` is uploaded (step_* only on request); None without
    # a monitor in .env, and every call below is then a no-op.
    var artifacts = sink_for_run(run.id, run.dir)

    var t_log = perf_counter_ns()
    var last_log_step = 0
    var gn_f1 = Float64(0)
    var gn_f2 = Float64(0)
    var gn_b = Float64(0)
    var g_value = Float64(0)
    var g_total = Float64(0)

    print("[2] training", train_steps, "steps  (d =", D, ", batch =", BATCH, ", seq =", SEQ, ")")
    print("      USE_TRAIN_CUDA_GRAPH =", USE_TRAIN_CUDA_GRAPH)
    for step in range(train_steps):
        # ── train batch: two independent draws + next rows ──────────────
        samp_a.draw_into_device(ctx, idx_s, BATCH)
        samp_b.draw_into_device(ctx, idx_sp, BATCH)
        ctx.enqueue_function[gather_idx_kernel[BATCH]](
            nxt_dev.unsafe_ptr(), idx_s.unsafe_ptr(), idx_sn.unsafe_ptr(),
            grid_dim=_blocks(BATCH), block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(), idx_s.unsafe_ptr(),
            t.t.bs.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS), block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(), idx_sn.unsafe_ptr(),
            t.t.bsn.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS), block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(), idx_sp.unsafe_ptr(),
            t.t.bsp.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS), block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[NACT, BATCH]](
            act_host.dev.value().unsafe_ptr(), idx_s.unsafe_ptr(),
            t.t.ba.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * NACT), block_dim=TPB,
        )

        # ── expert windows: NW starts -> BATCH rows of s and s' ──────────
        samp_w.draw_into_device(ctx, idx_w, NW)
        ctx.enqueue_function[gather_idx_kernel[NW]](
            starts_dev.unsafe_ptr(), idx_w.unsafe_ptr(), win_start.unsafe_ptr(),
            grid_dim=_blocks(NW), block_dim=TPB,
        )
        ctx.enqueue_function[expand_windows_kernel[NW, SEQ]](
            win_start.unsafe_ptr(), idx_e.unsafe_ptr(), idx_en.unsafe_ptr(),
            grid_dim=_blocks(BATCH), block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(), idx_e.unsafe_ptr(),
            t.head.es.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS), block_dim=TPB,
        )
        ctx.enqueue_function[gather_rows_kernel[OBS, BATCH]](
            obs_host.dev.value().unsafe_ptr(), idx_en.unsafe_ptr(),
            t.head.esn.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OBS), block_dim=TPB,
        )

        # ── z: goal / expert / uniform mixture, then the sphere ──────────
        t.t.embed_sp()
        t.encode_expert()
        box_muller_normal_gpu[BATCH * D](
            ctx, mptr(gauss.dev.value().unsafe_ptr()), UInt64(seed_v), rng_off
        )
        rng_off += UInt64(BATCH * D)
        ctx.enqueue_function[uniform01_kernel[BATCH * 2]](
            mptr(pick.dev.value().unsafe_ptr()), UInt64(seed_v) + 31, rng_off,
            grid_dim=_blocks(BATCH * 2), block_dim=TPB,
        )
        rng_off += UInt64(2 * BATCH * 2)
        ctx.enqueue_function[z_mixture3_kernel[D, BATCH]](
            t.t.bz.dev.value().unsafe_ptr(),
            gauss.dev.value().unsafe_ptr(),
            t.t.b_sp.dev.value().unsafe_ptr(),
            t.head.ez.dev.value().unsafe_ptr(),
            pick.dev.value().unsafe_ptr(),
            Scalar[DT](p_goal), Scalar[DT](p_expert),
            Int32(BATCH), Int32(BATCH),
            grid_dim=_blocks(BATCH), block_dim=TPB,
        )
        ctx.enqueue_function[project_sphere_kernel[D, BATCH]](
            t.t.bz.dev.value().unsafe_ptr(), Scalar[DT](SQRT_D),
            grid_dim=_blocks(BATCH), block_dim=TPB,
        )

        var want = (step % LOG_EVERY) == 0 or step == train_steps - 1
        var have = False
        var l = _zero_losses()
        comptime if USE_TRAIN_CUDA_GRAPH:
            if want:
                l = t.train_step(want_loss=True)
                have = True
            else:
                def _captured_step() capturing raises -> None:
                    t.train_device_kernels()

                maybe_capture_replay[_captured_step](train_graph, ctx)
        else:
            l = t.train_step(want_loss=want)
            have = want
        if want and have:
            t.t.read_grad_norms(gn_f1, gn_f2, gn_b)
            t.t.read_actor_grad_split(g_value, g_total)
            var now = perf_counter_ns()
            var since = step - last_log_step
            var sps = 0.0
            if since > 0:
                sps = Float64(since) * 1e9 / Float64(now - t_log)
            t_log = now
            last_log_step = step

            var names = List[String]()
            var vals = List[Float64]()
            names.append(String("fb_measure_loss")); vals.append(l.fb.measure)
            names.append(String("fb_ortho_loss")); vals.append(l.fb.ortho)
            names.append(String("policy_loss")); vals.append(l.fb.actor)
            names.append(String("f_norm")); vals.append(l.fb.f_norm)
            names.append(String("b_norm")); vals.append(l.fb.b_norm)
            names.append(String("b_norm_deficit")); vals.append(SQRT_D - l.fb.b_norm)
            names.append(String("fb_ortho_q")); vals.append(l.fb.ortho + 2.0 * l.fb.b_norm * l.fb.b_norm)
            names.append(String("f1_grad_norm")); vals.append(gn_f1)
            names.append(String("f2_grad_norm")); vals.append(gn_f2)
            names.append(String("b_grad_norm")); vals.append(gn_b)
            names.append(String("actor_grad_value")); vals.append(g_value)
            names.append(String("actor_grad_total")); vals.append(g_total)
            names.append(String("disc_expert_loss")); vals.append(l.d_pos)
            names.append(String("disc_policy_loss")); vals.append(l.d_neg)
            names.append(String("disc_gp_loss")); vals.append(l.d_gp)
            names.append(String("disc_reward_mean")); vals.append(l.r_mean)
            names.append(String("mean_q")); vals.append(l.q_mean)
            names.append(String("critic_loss")); vals.append(l.q_loss)
            names.append(String("policy_q_mean")); vals.append(l.q_pi)
            names.append(String("steps_per_s")); vals.append(sps)
            logger.log_scalars(names, vals, step)
            print(
                "   step", step,
                " measure", l.fb.measure, " ortho", l.fb.ortho, " actor", l.fb.actor,
                " |F|", l.fb.f_norm, " |B|", l.fb.b_norm,
                "\n         D+", l.d_pos, " D-", l.d_neg, " gp", l.d_gp,
                " r", l.r_mean, " Q", l.q_mean, " Qloss", l.q_loss, " Qpi", l.q_pi,
                " gv/gt", g_value, g_total, " ", sps, "st/s",
            )
            # ⚠ Reading D: `D+` and `D-` both near log 2 = D cannot separate
            # (the z-coupling carries no signal yet, or GP dominates); both
            # near 0 = D has WON and r_D saturates at ±16 — the critic then
            # trains on a constant and Q_D says nothing. The healthy band
            # is in between and MOVING; `gp` should settle near its floor.
        if step > 0 and (step % CKPT_EVERY) == 0:
            var p = run.checkpoint_path(String("step_") + String(step))
            t.save_state(p)
            print("      checkpoint ->", p, "(+ .cpr)")
    var pf = run.checkpoint_path(String("final"))
    t.save_state(pf)
    announce_checkpoint(pf, artifacts, run.dir)
    finish_run(run, logger, artifacts)
    print("[3] done. final checkpoint ->", pf, "(+ .cpr)")
    print("      metrics CSV ->", csv_path)
    print("      run record  ->", run.kv_path())


def _zero_losses() raises -> FBCPRLosses:
    return FBCPRLosses(FBLosses(0.0, 0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
