"""TD-MPC2 (deep_agents) — Hopper training (GPU, MPC-off) — item B lighthouse.

Hopper is the lighthouse for the episodic **termination head** (item B,
`docs/TDMPC2_DEEP_AGENTS_PORT.md` §14.2): unlike HalfCheetah it terminates
early when the hopper falls (`TERMINATE_ON_UNHEALTHY=True`), so:

  * `done = env.was_terminated()` — TRUE termination only, NOT the time-limit
    truncation. The value bootstrap is masked by (1−done) (already correct in
    td_target_step), and the same flag is the BCE target for the termination
    head.
  * `bce_coef > 0` enables the head (Kaiming-init + live BCE loss column in the
    WM graph). With `bce_coef=0` Hopper would still learn via the bootstrap
    mask, but the head is the point of this run.

This is the convergence gate for item B. Acting is MPC-off (a = π(encode(obs)));
flip to MPC once the head/value are healthy. Reference-ish dims (latent 512 /
mlp 512 / enc 256 / bins 101 / num_q 5 / horizon 3).

Run:
    pixi run -e nvidia mojo run -I . examples/hopper/tdmpc2_hopper_gpu.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.tdmpc2.agent import TDMPC2Agent
from noeira.deep_agents.tdmpc2.config import TDMPC2
from noeira.envs.hopper import Hopper, HopperConfig

comptime TARGET = "gpu"
comptime OBS = HopperConfig.OBS_DIM        # 11
comptime ACT = HopperConfig.ACTION_DIM     #  3
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 256
comptime H = 3
comptime CAP = 1_000_000

comptime LR = 3e-4
comptime ACTION_SCALE = 1.0
comptime BCE_COEF = 1.0          # item B: enable the termination head
comptime LEARN_START = 5_000
comptime TRAIN_EVERY = 1
comptime TOTAL = 1_000_000
comptime EVAL_EVERY = 20_000
comptime DIAG_EVERY = 1_000
comptime CHECKPOINT_EVERY = 50_000
comptime EVAL_EPS = 5
comptime EP_LEN = 1_000

comptime Env = Hopper[DT, TERMINATE_ON_UNHEALTHY=True]
comptime Ag = TDMPC2Agent[
    TARGET, OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP,
]


def _greedy_eval(mut ag: Ag, mut env: Env) raises -> Scalar[DT]:
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var total: Scalar[DT] = 0.0
    for _ep in range(EVAL_EPS):
        var obs = env.reset_obs_list()
        for _s in range(EP_LEN):
            for i in range(OBS):
                obsbuf[i] = obs[i]
            ag.select_greedy_action(obsbuf, actbuf)
            var al = List[Scalar[DT]]()
            for j in range(ACT):
                al.append(actbuf[j])
            var r = env.step_continuous_vec[DT](al)
            total += r[1]
            obs = r[0].copy()
            if r[2]:
                break
    obsbuf.free(); actbuf.free()
    return total / Scalar[DT](EVAL_EPS)


def main() raises:
    print("=" * 70)
    print("TD-MPC2 (deep_agents) — Hopper", TARGET, "(MPC-off, episodic head)")
    print("  OBS=", OBS, " ACT=", ACT, " latent=", LATENT, " B=", B, " H=", H)
    print("  bce_coef=", BCE_COEF, " (termination head ON)")
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()
    var env = Env()
    # Build via the Design-F preset; bce_coef>0 turns on the termination head.
    var ag = TDMPC2[
        TARGET, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
    ](
        ctx=ctx, lr=Scalar[DT](LR), action_scale=Scalar[DT](ACTION_SCALE),
        learning_starts=LEARN_START, bce_coef=Scalar[DT](BCE_COEF),
    )

    # ─── Run + logger ───────────────────────────────────────────────────────
    var run = RunContext(
        project=String("mujoco"),
        driver=String("examples/hopper/tdmpc2_hopper_gpu.mojo"),
        slug=String("tdmpc2-hopper"),
        env=String("builtin:mujoco/hopper"),
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("  Run:", run.dir)
    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "TD-MPC2")
    logger.set_config("env", "Hopper")
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var best: Scalar[DT] = -1.0e9
    var t_start = perf_counter_ns()

    for step in range(TOTAL):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            for j in range(ACT):
                actbuf[j] = Scalar[DT](random_float64() * 2.0 - 1.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var al = List[Scalar[DT]]()
        for j in range(ACT):
            al.append(actbuf[j])
        var res = env.step_continuous_vec[DT](al)
        # Episodic: record the REAL termination flag (fall), not truncation —
        # this is both the value-bootstrap mask and the BCE target for the head.
        var term = Scalar[DT](1.0) if env.was_terminated() else Scalar[DT](0.0)
        ag.record(obsbuf, actbuf, res[1], term)
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            _ = ag.train_step()
        if step > 0 and step % DIAG_EVERY == 0:
            ag.flush_metrics_through_logger[RunLogger](logger_ptr, step)
            logger.flush()
        if step > 0 and step % CHECKPOINT_EVERY == 0:
            ag.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run.dir)
        if step > 0 and step % EVAL_EVERY == 0:
            var ret = _greedy_eval(ag, env)
            if ret > best:
                best = ret
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            logger.log_scalar("avg_reward", Float64(ret), step)
            logger.log_scalar("best_reward", Float64(best), step)
            print(
                "  step", step, " eval_return=", ret, " best=", best,
                " wm=", ag.last_wm_loss(), " term=", ag.last_termination_loss(),
                " (", elapsed, "s )",
            )

    ag.save_state(checkpoint_path)
    announce_checkpoint(checkpoint_path, artifacts, run.dir)
    finish_run(
        run, logger, artifacts, String("best_eval_return=") + String(best)
    )
    _ = logger
    print("=" * 70)
    print("  FINAL best eval return =", best)
    print("  ( Hopper: >1500 strong, >3000 excellent )")
    print("  run:", run.dir)
    print("=" * 70)
    obsbuf.free(); actbuf.free()
