"""Smoke: Gumbel AlphaZero GPU search on real TicTacToe boards.

Drives the new `GumbelGPUMCTS.search_gpu_alphazero` — Gumbel-Top-k roots +
Sequential Halving + visit-balance in-tree selection (the published Gumbel
AlphaZero planner), with true-game-rules expansion (`AZEnvGPU.step_gpu`),
legal-masked child priors, scalar tanh value (`BINS=1`) and negated SelfPlay
backup. Serial sims — structurally immune to the frozen-tree batched-leaf
bias that killed `GenericGPUMCTS` at `BATCH_SIMS > 1`.

Asserts: improved policy is normalized, puts (near-)zero mass on illegal
actions, root values finite and in [-1, 1]; argmax is legal.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_gumbel_search_smoke.mojo
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.alphazero.nets import AZMLPNet
from noeira.deep_agents.zero.mcts_adapters import AZPredGPU, AZEnvGPU
from noeira.planners.tree_search import GumbelGPUMCTS, SelfPlay
from noeira.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime N_ENVS = 4
    comptime ACT = 9
    comptime OBS = 27           # planner LATENT == OBS for AlphaZero
    comptime BINS = 1           # scalar tanh value head → PRED_OUT = ACT + 1
    comptime STATE = 12
    comptime H = 64
    comptime MAX_NODES = 64
    comptime MAX_K = 4          # Gumbel root candidates (power of two)
    comptime NUM_SIMS = 32

    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime MCTS = GumbelGPUMCTS[
        N_ENVS, ACT, OBS, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay,
        STATE_SIZE=STATE,
    ]

    var ctx = DeviceContext()
    var net = Net.make["gpu", Kaiming](Optional(ctx))
    var pred = AZPredGPU[OBS, ACT, Net].make(net)
    var env = AZEnvGPU[Env, STATE, OBS, ACT]()
    var mcts = MCTS(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)

    # ── N_ENVS boards: env 0 fresh, others advanced by e scripted plies ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var actions = ctx.enqueue_create_buffer[DT](N_ENVS)
    var rewards = ctx.enqueue_create_buffer[DT](N_ENVS)
    var dones = ctx.enqueue_create_buffer[DT](N_ENVS)
    var terminated = ctx.enqueue_create_buffer[DT](N_ENVS)
    var act_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    Env.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    ctx.synchronize()
    for ply in range(2):
        for e in range(N_ENVS):
            act_h.unsafe_ptr()[e] = Scalar[DT]((e + ply * 4) % ACT)
        ctx.enqueue_copy(actions, act_h)
        Env.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, actions, rewards, dones, terminated, obs, legal
        )
    Env.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](ctx, states, obs, legal)
    ctx.synchronize()

    var root_obs = tile[N_ENVS, OBS](obs.unsafe_ptr().as_unsafe_any_origin())

    # ── Run the full Gumbel AlphaZero search ──
    mcts.search_gpu_alphazero[type_of(pred), type_of(env)](
        ctx, pred, env, root_obs, states, legal,
        k_actual=MAX_K, rng_seed=UInt32(42),
    )
    ctx.synchronize()

    var pol_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var rv_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var legal_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    ctx.enqueue_copy(pol_h, mcts.policies_view())
    ctx.enqueue_copy(rv_h, mcts.root_value_view())
    ctx.enqueue_copy(legal_h, legal)
    ctx.synchronize()

    for e in range(N_ENVS):
        var psum = 0.0
        var illegal_mass = 0.0
        var best = 0
        for a in range(ACT):
            var p = Float64(pol_h.unsafe_ptr()[e * ACT + a])
            assert_true(p == p, "policy NaN")
            assert_true(p >= 0.0, "policy negative")
            psum += p
            if Float64(legal_h.unsafe_ptr()[e * ACT + a]) < 0.5:
                illegal_mass += p
            if p > Float64(pol_h.unsafe_ptr()[e * ACT + best]):
                best = a
        var v = Float64(rv_h.unsafe_ptr()[e])
        print("env", e, "argmax", best, "root_v", v,
              "psum", psum, "illegal_mass", illegal_mass)
        assert_true(psum > 0.99 and psum < 1.01, "policy does not sum to 1")
        assert_true(illegal_mass < 1e-3, "policy mass on illegal actions")
        assert_true(
            Float64(legal_h.unsafe_ptr()[e * ACT + best]) > 0.5,
            "argmax action is illegal",
        )
        assert_true(v == v and v >= -1.001 and v <= 1.001,
                    "root value out of [-1, 1]")

    _ = net^  # keepalive for pred's non-owning pointer
    print("Gumbel AlphaZero TicTacToe search smoke: OK")
