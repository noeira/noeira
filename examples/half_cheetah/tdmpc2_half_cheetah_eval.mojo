"""TD-MPC2 HalfCheetah — policy-vs-MPC eval on a loaded checkpoint (diagnostic).

Loads a trained checkpoint and evaluates the SAME model two ways via the
`agent.evaluate[...]` facade:
  * policy:  a = π(encode(obs))                 (USE_MPC=False)
  * MPC:     a = MPPI plan over the world model  (USE_MPC=True)

Decisively separates a planning bug from the collection feedback loop:
  - MPC_return ≳ policy_return  → planning works; MPC's training underperformance
    was the data feedback loop / planning budget (try heavier MPC_* or warm up
    the model with MPC-off first).
  - MPC_return ≪ policy_return  → the planner's model-value ranking is off
    (planning bug to fix), independent of training dynamics.

Point CKPT at a GOOD checkpoint — e.g. tdmpc2_half_cheetah_mpcoff.ckpt from a
strong MPC-off run. Bump MPC_* toward the 512/24/64/6 reference to test budget.

Run: `pixi run -e nvidia mojo run -I . examples/half_cheetah/tdmpc2_half_cheetah_eval.mojo`
     (`... --ckpt <run_id>` reads a training run's `checkpoints/last.ckpt`)
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.tdmpc2.config import TDMPC2
from noeira.envs.half_cheetah import HalfCheetah, HalfCheetahConfig

comptime CKPT = "tdmpc2_half_cheetah_mpcoff.ckpt"
comptime OBS = HalfCheetahConfig.OBS_DIM
comptime ACT = HalfCheetahConfig.ACTION_DIM
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 8          # eval-only: training batch unused
comptime H = 3
comptime CAP = 4096     # eval-only: replay unused

# MPC planning budget for the eval (match the run you're diagnosing, or bump
# toward the 512/24/64/6 reference to test whether budget is the issue).
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

comptime EP_LEN = 1_000
comptime N_EPS = 5

comptime Env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    var ckpt = resolve_checkpoint(_flag(String("--ckpt"), String(CKPT)), String("last"))
    print("=" * 70)
    print("TD-MPC2 HalfCheetah — policy vs MPC eval @", ckpt)
    print("  MPC budget:", MPC_SAMPLES, "/", MPC_PI_TRAJS, "/", MPC_ELITES,
          "/", MPC_ITERS, " (samples/pi/elites/iters)")
    print("=" * 70)
    var ctx = DeviceContext()
    var env = Env()
    # Build through the Design-F preset (config.mojo) — returns the agent.
    var ag = TDMPC2[
        "gpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
    ](ctx=ctx, action_scale=Scalar[DT](1.0), learning_starts=0)
    ag.load_state(ckpt)

    var pol = ag.evaluate[Env, USE_MPC=False](
        env, episodes=N_EPS, max_steps=EP_LEN
    )
    print("  policy eval return =", pol)
    var mpc = ag.evaluate[Env, USE_MPC=True](
        env, episodes=N_EPS, max_steps=EP_LEN
    )
    print("  MPC    eval return =", mpc)
    print("=" * 70)
    if mpc >= pol:
        print("  MPC >= policy → planning works; MPC training gap was the data",
              "loop / budget.")
    else:
        print("  MPC < policy → planner mis-ranks vs the direct policy",
              "(planning bug to investigate).")
    print("=" * 70)
