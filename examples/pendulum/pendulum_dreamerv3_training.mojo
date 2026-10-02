"""DreamerV3 (nn) — Pendulum lighthouse driver.

PR5c Step 7. End-to-end DreamerV3 world-model + actor-critic training on
PendulumV2 (CPU) using the block-composed `DreamerV3Agent`: on-policy
running-carry action selection, sequence replay, WM-BPTT + imagination AC,
periodic greedy eval.

⚠️ This is the lighthouse SCAFFOLD — convergence/tuning is user-in-loop (see
`docs/DREAMERV3_PR5C_RUNBOOK.md` §"Decision points"). Pass = mean_ret(10) ≥
−200 @ ~1M steps; CI early signal at 30k: mean_ret > −1250 (beats random).

act = SiLU (SwishOp), as in the size1m/dmc config: the trainer/blocks/agent
thread `SwishOp` through every WM + AC net (`blocks.mojo` / `trainer.mojo`).
The `GELUTanhOp` default on the `nets.mojo` / `wm.mojo` aliases is ONLY for the
JAX-fixture validation spikes — it is overridden here. (Same for the GPU
driver `pendulum_dreamerv3_nn_gpu.mojo`.)

Known v1 caveats before expecting convergence:
  - CPU only; the full size1m config (DETER=512, B=16, T=64, BINS=255) is
    SLOW on CPU — each train_step is a 64-step BPTT. Use the GPU driver
    `pendulum_dreamerv3_nn_gpu.mojo` for the real 1M run, or the smaller
    config below for a CPU CI gate.
  - imagination rolls out from ALL NS=T·B posterior carries (the reference).
  - convergence is still open (off-distribution imagined-reward optimism —
    a WM-quality/scale issue). Tune lr / warmup / free_nats / loss_scales /
    train_every, and see the GPU-phase levers in the runbook.

Run (CPU):
  pixi run mojo run -I . examples/pendulum/pendulum_dreamerv3_training.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed

from noeira.nn.constants import DT
from noeira.deep_agents.dreamerv3.agent import DreamerV3Agent
from noeira.envs.pendulum import PendulumV2

# ── config ─────────────────────────────────────────────────────────────
# size1m target (runbook): DETER=512 H=64 CLASSES=4 STOCH=32 BLOCKS=8
# units=64 BINS=255 B=16 T=64 T_IMAG=15. CPU-light default below; bump to
# size1m for the real (GPU) run.
comptime OBS = 3
comptime ACT = 1
comptime DETER = 256
comptime H = 64
comptime STOCH = 32
comptime CLASSES = 4
comptime BLOCKS = 8
comptime TOKEN = 64
comptime DEC_U = 64
comptime HU = 64
comptime VU = 64
comptime PU = 64
comptime BINS = 255
comptime B = 16
comptime T = 32
comptime T_IMAG = 15
comptime CAP = 1_000_000

comptime Ag = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP,
]

comptime TOTAL_STEPS = 1_100_000
comptime LEARN_START = 1024
comptime TRAIN_EVERY = 4         # env steps per train_step. 16→4 = 4× more
                                 # gradient (replay ratio ~32→~128), to fix the
                                 # value undertracking (val_m lagging ret_m).
                                 # Reference ratio is ~512 (TRAIN_EVERY=1) but
                                 # that's ~16× the CPU cost; 4 + lr=1e-4 together
                                 # ≈ 10× more effective learning. Costs ~4× CPU
                                 # per env-step — dial back to 8 if too slow.
comptime EVAL_EVERY = 5000
comptime EVAL_EPISODES = 10
comptime EP_LEN = 200
# Env torque range. The agent acts in normalized [-1,1]; the driver scales
# to the env's [-ACTION_SCALE, ACTION_SCALE] at env.step. Recorded actions
# stay normalized (WM grounding consistency — see agent.mojo).
comptime ACTION_SCALE = Scalar[DT](2.0)


def _greedy_eval(
    mut ag: Ag, mut env: PendulumV2[DT]
) raises -> Tuple[Scalar[DT], Scalar[DT]]:
    """Returns (mean_return, mean_|greedy_action|) over EVAL_EPISODES."""
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var total: Scalar[DT] = 0.0
    var act_abs: Scalar[DT] = 0.0
    var nsteps: Int = 0
    for _e in range(EVAL_EPISODES):
        ag.reset_belief()
        var o = env.reset_obs_list()
        for _s in range(EP_LEN):
            for i in range(OBS):
                obsbuf[i] = o[i]
            ag.select_greedy_action(obsbuf, actbuf)
            var aa = actbuf[0] if actbuf[0] >= 0 else -actbuf[0]
            act_abs += aa
            nsteps += 1
            var al = List[Scalar[DT]]()
            al.append(ACTION_SCALE * actbuf[0])   # normalized → env range
            var r = env.step_continuous_vec[DT](al)
            total += r[1]
            o = r[0].copy()
            if r[2]:
                break
    obsbuf.free(); actbuf.free()
    return Tuple(
        total / Scalar[DT](EVAL_EPISODES),
        act_abs / Scalar[DT](nsteps),
    )


def main() raises:
    print("=" * 70)
    print("DreamerV3 (nn) Pendulum lighthouse —", TOTAL_STEPS, "steps")
    print("=" * 70)
    seed(42)
    var env = PendulumV2[DT]()
    # slowtar=True: λ-return bootstraps from the EMA slowvalue (target network)
    # instead of the online value — this breaks the value→return→value
    # self-feedback loop that made lr=1e-4 diverge (val_m shot to +46). With the
    # critic now damped, we re-enable lr=1e-4 + TRAIN_EVERY=4 (~10× more
    # learning) to actually close the value-undertracking gap the probes found.
    # Watch: val_m should track ret_m NEGATIVE and STABLE (no positive runaway),
    # and pstd should drop below 1.0 as the advantage firms up. If val_m still
    # diverges positive, drop lr→4e-5 (slowtar alone, slower but stable).
    # actent 3e-4→1e-3: slowtar FIXED the critic (val_m now tracks ret_m
    # stably, no runaway) and the actor finally commits (pstd dropped 1.0→0.53)
    # — but it commits to a BAD attractor (near-max constant torque, pmean→3.1),
    # ret degrades −865→−1600 and the value descends toward the bad-policy
    # asymptote. Regime flipped vs before: earlier pstd was pinned at 1.0 (under-
    # trained value, raising entropy = more random = worse); NOW the actor
    # commits too fast to a local optimum, so MORE entropy keeps it exploring
    # long enough to find the swing-up. Try 3e-3 if 1e-3 still locks in.
    var ag = Ag.make(
        lr=Scalar[DT](1e-4), learning_starts=LEARN_START,
        action_scale=Scalar[DT](2.0), actent=Scalar[DT](1e-3),
        slowtar=True,
    )

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)

    for step in range(TOTAL_STEPS):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            # warmup explores the NORMALIZED [-1,1] space (recorded as-is)
            actbuf[0] = Scalar[DT](random_float64() * 2.0 - 1.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var al = List[Scalar[DT]]()
        al.append(ACTION_SCALE * actbuf[0])   # normalized → env range
        var res = env.step_continuous_vec[DT](al)
        ag.record(
            obsbuf, actbuf, res[1],            # record the NORMALIZED action
            Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
        )
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
            ag.reset_belief()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            _ = ag.train_step()

        if step > 0 and step % EVAL_EVERY == 0:
            # eval mutates env episode state; use a fresh env for eval
            var eval_env = PendulumV2[DT]()
            var ev = _greedy_eval(ag, eval_env)
            print(
                "  step", step, " ret=", ev[0], " |act|=", ev[1],
                " real_rew=", ag.dbg_real_rew(), " rew_pred=", ag.dbg_rew_pred(),
                " ret_m=", ag.dbg_ret_mean(), " ret_sd=", ag.dbg_ret_std(),
                " pmean=", ag.dbg_pmean_abs(),
                # divergence probes: val_m vs ret_m (critic fit/divergence),
                # pstd (exploration collapse → minstd 0.1), rscale (adv denom)
                " val_m=", ag.dbg_val_mean(), " pstd=", ag.dbg_pstd(),
                " rscale=", ag.dbg_rscale(),
                " WM=", ag.last_wm_loss(), " AC=", ag.last_ac_loss(),
            )
            # restore collection env episode
            obs = env.reset_obs_list()
            ag.reset_belief()

    var fe = _greedy_eval(ag, env)
    print("=" * 70)
    print("FINAL mean_ret(", EVAL_EPISODES, ") =", fe[0],
          "  (lighthouse pass: >= -200)")
    print("=" * 70)
    obsbuf.free(); actbuf.free()
