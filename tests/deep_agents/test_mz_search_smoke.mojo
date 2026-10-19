"""End-to-end smoke: nn MuZero h/g/f nets + learned-dynamics GPU MCTS via the
planner's ``search_gpu[REP, DYN, PRED]``. Single-player (CartPole shape).

The MuZero counterpart of `test_az_search_tictactoe_smoke`: it drives the WHOLE
learned-model GPU search loop reusing ``GenericGPUMCTS`` verbatim, with the only
nn-new pieces being the three adapters (`MZRepGPU` / `MZDynGPU` / `MZPredGPU`).
No env is needed — the dynamics are learned, so the search consumes only a batch
of root observations and unrolls the nets in the tree.

Pipeline per env: rep encode root → predict → init root (softmax prior, optional
Dirichlet) → {select+build dyn input → dyn step → predict child → categorical
decode (reward/value h⁻¹) + min-max scale + backup} × sims → visit-count action
+ root value.

Asserts the selected action is in range and policy/value are finite for a batch
of arbitrary root observations.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_mz_search_smoke.mojo
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from noeira.deep_agents.zero.mcts_adapters_mz import (
    MZRepGPU, MZDynGPU, MZPredGPU,
)
from noeira.planners.tree_search import (
    GenericGPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    SinglePlayer,
)


def main() raises:
    comptime N_ENVS = 4
    comptime ACT = 2            # CartPole
    comptime OBS = 4
    comptime LATENT = 16
    comptime BINS = 51          # categorical reward/value
    comptime H = 32
    comptime MAX_NODES = 64
    comptime NUM_SIMS = 24
    comptime BATCH_SIMS = 1

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    comptime MCTS = GenericGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS, BATCH_SIMS,
        MuZeroPUCT[1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
    ]

    var ctx = DeviceContext()
    var rep_net = Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn_net = Dyn.make["gpu", Kaiming](Optional(ctx))
    var pred_net = Pred.make["gpu", Kaiming](Optional(ctx))
    var rep = MZRepGPU[OBS, LATENT, Rep].make(rep_net)
    var dyn = MZDynGPU[LATENT, ACT, BINS, Dyn].make(dyn_net)
    var pred = MZPredGPU[LATENT, ACT, BINS, Pred].make(pred_net)
    # v_min/v_max are the h-space support — must match the two-hot targets used
    # in training. gamma is the MuZero discount.
    var mcts = MCTS(ctx, gamma=0.997, v_min=-10.0, v_max=10.0)

    # ── A batch of arbitrary root observations ──
    var obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var obs_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
    ctx.synchronize()
    for e in range(N_ENVS):
        obs_h.unsafe_ptr()[e * OBS + 0] = Scalar[DT](0.02) * Scalar[DT](e)
        obs_h.unsafe_ptr()[e * OBS + 1] = Scalar[DT](-0.1)
        obs_h.unsafe_ptr()[e * OBS + 2] = Scalar[DT](0.05) * Scalar[DT](e)
        obs_h.unsafe_ptr()[e * OBS + 3] = Scalar[DT](0.0)
    ctx.enqueue_copy(obs, obs_h)
    ctx.synchronize()

    var root_obs = tile[N_ENVS, OBS](obs)

    # ── Run the full MuZero learned-dynamics MCTS search ──
    mcts.search_gpu[type_of(rep), type_of(dyn), type_of(pred)](
        ctx, rep, dyn, pred, root_obs, rng_seed=UInt32(42)
    )
    ctx.synchronize()

    var act_host = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var pol_host = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var rv_host = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    ctx.enqueue_copy(act_host, mcts.actions_out)
    ctx.enqueue_copy(pol_host, mcts.policies_out)
    ctx.enqueue_copy(rv_host, mcts.root_value_out)
    ctx.synchronize()

    for e in range(N_ENVS):
        var a = Int(act_host.unsafe_ptr()[e])
        assert_true(a >= 0 and a < ACT, "action out of range")
        var psum: Float64 = 0.0
        for j in range(ACT):
            var p = Float64(pol_host.unsafe_ptr()[e * ACT + j])
            assert_true(p == p, "policy NaN")
            psum += p
        assert_true(psum > 0.99 and psum < 1.01, "policy does not sum to 1")
        var v = Float64(rv_host.unsafe_ptr()[e])
        assert_true(v == v, "root value NaN")
        assert_true(v > -1e6 and v < 1e6, "root value not finite")
        print("env", e, "action", a, "root_value", v, "psum", psum)

    _ = rep_net^   # keepalive for the adapters' non-owning pointers
    _ = dyn_net^
    _ = pred_net^
    print("MuZero learned-dynamics search smoke: OK")
