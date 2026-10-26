"""AZ GPU search: serial vs batched-leaf A/B — the Connect Four BATCH_SIMS probe.

The MuZero GPU batched-leaf path (BATCH_SIMS>1 + virtual loss) was proven
value-biased vs the CPU search (test_mz_search_gpu_cpu_parity): rounds select
every leaf against a FROZEN tree, so duplicate (parent, action) selections
re-expand the same edge and double-count its value. The AlphaZero batched
kernel (`gpu_mcts_batched_expand_backup_kernel`) has the SAME structure — no
duplicate detection — and every validated AZ run (TTT convergence ×3 nets,
arena, C4 head-to-head) used BATCH_SIMS=1. The only BATCH_SIMS>1 user is the
`connect_four_alphazero_v2.mojo` example (BATCH_SIMS=5, added for ~5× MCTS
speed, never convergence-validated).

This test quantifies the divergence at exactly the example's planner settings
(AlphaGoPUCT[1.0], SelfPlay negate+squash, VIRTUAL_LOSS=3) on real Connect
Four boards: identical net, identical states, NoNoise, serial (BATCH_SIMS=1)
vs batched (BATCH_SIMS=5), same NUM_SIMS. Serial is the configuration all the
convergence evidence rests on; large root-value / argmax divergence here means
the example's batched setting rides an unvalidated, structurally-suspect path.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_search_gpu_batched_bias.mojo
"""

from max.gpu.host import DeviceContext
from std.math import abs
from std.testing import assert_true
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.alphazero.nets import AZMLPNet
from noeira.deep_agents.zero.mcts_adapters import AZPredGPU, AZEnvGPU
from noeira.planners.tree_search import (
    GenericGPUMCTS,
    AlphaGoPUCT,
    NoNoise,
    SelfPlay,
)
from noeira.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    comptime N_ENVS = 4
    comptime ACT = 7
    comptime OBS = 126          # 3 planes × 6×7; planner LATENT == OBS for AZ
    comptime BINS = 1           # scalar value head → PRED_OUT = ACT + 1
    comptime Env = ConnectFourEnv[DType.float64]
    comptime STATE = Env.STATE_SIZE
    comptime H = 64
    comptime MAX_NODES = 256
    comptime NUM_SIMS = 100     # divisible by 1 and 5

    comptime Net = AZMLPNet[OBS, ACT, H]
    # Serial reference — the configuration every green AZ convergence run used.
    comptime MCTS1 = GenericGPUMCTS[
        N_ENVS, ACT, OBS, BINS, MAX_NODES, NUM_SIMS, 1,
        AlphaGoPUCT[1.0], NoNoise, SelfPlay, STATE_SIZE=STATE,
    ]
    # Batched — the former C4-example setting (default VIRTUAL_LOSS=3).
    comptime MCTS5 = GenericGPUMCTS[
        N_ENVS, ACT, OBS, BINS, MAX_NODES, NUM_SIMS, 5,
        AlphaGoPUCT[1.0], NoNoise, SelfPlay, STATE_SIZE=STATE,
        UNSAFE_BATCHED=True,    # diagnostic: measuring the known bias
    ]

    var ctx = DeviceContext()
    var net = Net.make["gpu", Kaiming](Optional(ctx))
    var pred = AZPredGPU[OBS, ACT, Net].make(net)
    var env = AZEnvGPU[Env, STATE, OBS, ACT]()
    var mcts1 = MCTS1(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)
    var mcts5 = MCTS5(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)

    # ── N_ENVS Connect Four boards, diversified by 2 scripted plies each ──
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
            act_h.unsafe_ptr()[e] = Scalar[DT]((e * 2 + ply * 3) % ACT)
        ctx.enqueue_copy(actions, act_h)
        Env.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, actions, rewards, dones, terminated, obs, legal
        )
    Env.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](ctx, states, obs, legal)
    ctx.synchronize()

    var root_obs = tile[N_ENVS, OBS](obs.unsafe_ptr().as_unsafe_any_origin())
    var root_legal = tile[N_ENVS * ACT](legal.unsafe_ptr().as_unsafe_any_origin())

    # ── Serial search (search leaves `states` unmodified) ──
    mcts1.search_gpu_alphazero[type_of(pred), type_of(env)](
        ctx, pred, env, root_obs, states, root_legal, rng_seed=UInt64(42)
    )
    ctx.synchronize()
    var p1 = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var v1 = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    ctx.enqueue_copy(p1, mcts1.policies_out)
    ctx.enqueue_copy(v1, mcts1.root_value_out)
    ctx.synchronize()

    # ── Batched-5 search on the same boards ──
    mcts5.search_gpu_alphazero[type_of(pred), type_of(env)](
        ctx, pred, env, root_obs, states, root_legal, rng_seed=UInt64(42)
    )
    ctx.synchronize()
    var p5 = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var v5 = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    ctx.enqueue_copy(p5, mcts5.policies_out)
    ctx.enqueue_copy(v5, mcts5.root_value_out)
    ctx.synchronize()

    var argmax_flips = 0
    var max_dv = 0.0
    for e in range(N_ENVS):
        var a1 = 0
        var a5 = 0
        var psum1 = 0.0
        var psum5 = 0.0
        for a in range(ACT):
            var q1 = Float64(p1.unsafe_ptr()[e * ACT + a])
            var q5 = Float64(p5.unsafe_ptr()[e * ACT + a])
            psum1 += q1
            psum5 += q5
            if q1 > Float64(p1.unsafe_ptr()[e * ACT + a1]):
                a1 = a
            if q5 > Float64(p5.unsafe_ptr()[e * ACT + a5]):
                a5 = a
        var rv1 = Float64(v1.unsafe_ptr()[e])
        var rv5 = Float64(v5.unsafe_ptr()[e])
        var dv = abs(rv1 - rv5)
        if dv > max_dv:
            max_dv = dv
        if a1 != a5:
            argmax_flips += 1
        print("env", e, "| serial  v", rv1, "argmax", a1, "psum", psum1)
        print("      | batched5 v", rv5, "argmax", a5, "psum", psum5)
        print("      | |dv|", dv)
        for a in range(ACT):
            print(
                "        a", a,
                "p1", Float64(p1.unsafe_ptr()[e * ACT + a]),
                "p5", Float64(p5.unsafe_ptr()[e * ACT + a]),
            )
        assert_true(rv1 == rv1 and rv5 == rv5, "root value NaN")
        assert_true(psum1 > 0.99 and psum1 < 1.01, "serial policy not normed")
        assert_true(psum5 > 0.99 and psum5 < 1.01, "batched policy not normed")

    print("argmax flips:", argmax_flips, "/", N_ENVS, "| max |dv|:", max_dv)
    _ = net^
    print("AZ serial-vs-batched A/B complete (diagnostic — see numbers above)")
