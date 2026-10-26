"""End-to-end smoke: nn AlphaZero prediction net + true-game-rules MCTS on the
real TicTacToe GPU env via the planner's ``search_gpu_alphazero``.

This is the Phase-A de-risk milestone: it exercises the WHOLE AlphaZero GPU
search loop reusing ``GenericGPUMCTS`` verbatim, with the only nn-new pieces
being the ``AZPredGPU`` (prediction) and ``AZEnvGPU`` (env-step) adapters.

Pipeline per env: root predict → masked Dirichlet prior → {select+copy parent
state → env.step expansion → predict child → negated+squashed backup} × sims →
masked visit-count action + root value.

Asserts the selected action is legal and the policy/value are finite at the
opening position (4 fresh TicTacToe boards).

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_search_tictactoe_smoke.mojo
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.alphazero.nets import AZMLPNet
from noeira.deep_agents.zero.mcts_adapters import AZPredGPU, AZEnvGPU
from noeira.planners.tree_search import (
    GenericGPUMCTS,
    AlphaGoPUCT,
    DirichletNoise,
    SelfPlay,
)
from noeira.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime N_ENVS = 4
    comptime ACT = 9
    comptime OBS = 27           # planner LATENT == OBS for AlphaZero
    comptime BINS = 1           # scalar value head → PRED_OUT = ACT + 1
    comptime STATE = 12
    comptime H = 64
    comptime MAX_NODES = 64
    comptime NUM_SIMS = 16
    comptime BATCH_SIMS = 1

    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime MCTS = GenericGPUMCTS[
        N_ENVS, ACT, OBS, BINS, MAX_NODES, NUM_SIMS, BATCH_SIMS,
        AlphaGoPUCT[2.5], DirichletNoise[0.25, 0.25], SelfPlay,
        STATE_SIZE=STATE,
    ]

    var ctx = DeviceContext()
    var net = Net.make["gpu", Kaiming](Optional(ctx))
    var pred = AZPredGPU[OBS, ACT, Net].make(net)
    var env = AZEnvGPU[Env, STATE, OBS, ACT]()
    var mcts = MCTS(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)

    # ── Set up N_ENVS fresh TicTacToe boards on device ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    Env.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    Env.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](ctx, states, obs, legal)
    ctx.synchronize()

    var root_obs = tile[N_ENVS, OBS](obs.unsafe_ptr().as_unsafe_any_origin())
    var root_legal = tile[N_ENVS * ACT](legal.unsafe_ptr().as_unsafe_any_origin())

    # ── Run the full AlphaZero MCTS search ──
    mcts.search_gpu_alphazero[type_of(pred), type_of(env)](
        ctx, pred, env, root_obs, states, root_legal, rng_seed=UInt64(42)
    )
    ctx.synchronize()

    # ── Read results to host ──
    var act_host = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var pol_host = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var rv_host = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var legal_host = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    ctx.enqueue_copy(act_host, mcts.actions_out)
    ctx.enqueue_copy(pol_host, mcts.policies_out)
    ctx.enqueue_copy(rv_host, mcts.root_value_out)
    ctx.enqueue_copy(legal_host, legal)
    ctx.synchronize()

    for e in range(N_ENVS):
        var a = Int(act_host.unsafe_ptr()[e])
        assert_true(a >= 0 and a < ACT, "action out of range")
        assert_true(
            Float64(legal_host.unsafe_ptr()[e * ACT + a]) > 0.5,
            "selected action is illegal",
        )
        var psum: Float64 = 0.0
        for j in range(ACT):
            var p = Float64(pol_host.unsafe_ptr()[e * ACT + j])
            assert_true(p == p, "policy NaN")
            psum += p
        assert_true(psum > 0.99 and psum < 1.01, "policy does not sum to 1")
        var v = Float64(rv_host.unsafe_ptr()[e])
        assert_true(v == v, "root value NaN")
        print("env", e, "action", a, "root_value", v, "psum", psum)

    _ = net^  # keepalive for pred's non-owning pointer
    print("AZ search TicTacToe smoke: OK")
