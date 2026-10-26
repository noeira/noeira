"""EZv2-Atari value-prefix INTEGRATION smoke (GPU) — search + train compose.

Proves the two new value-prefix integration points compose on the device with
the REAL planner, at the real Atari config (FRAMES=4, ACT=18, BINS=601, spatial
latent [64,6,6]), tiny planner dims:

  1. SEARCH drop-in: the fused `EZDynVPNetAtari` (z'-only dyn + zero-(h,c) LSTM
     reward head) plugged into `GumbelGPUMCTS` via the unchanged `MZDynGPU`
     adapter — one real Gumbel search must yield finite root policy + value.
     This is the decision-B1.1 claim: value prefix needs NO orchestrator change.
  2. TRAIN: `ezv2_unroll_train_step_gpu_vp` driving the SAME fused module's
     `dyn.dynz` / `dyn.rew` sub-modules (+ cumulative value-prefix targets) must
     reduce a fixed-batch loss — confirming the search net and the train step
     share weights and integrate.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_value_prefix_integration_gpu.mojo
"""

from std.math import isnan, isinf
from noeira.core.tiles import tile
from max.gpu.host import DeviceContext
from std.testing import assert_true

from noeira.nn.constants import DT, LAYOUT_NCHW
from noeira.nn.core.initializer import Kaiming
from noeira.nn.optimizer.adam import Adam
from noeira.deep_agents.efficient_zero_v2.config_atari import EZV2AtariConfig
from noeira.deep_agents.efficient_zero_v2.nets_atari import (
    EZDynVPNetAtari, ez_atari_init_zero_pred, ez_atari_init_zero_reward,
    EZ_LSTM_HIDDEN, EZ_LSTM_HORIZON,
)
from noeira.deep_agents.efficient_zero_v2.blocks import (
    ezv2_unroll_train_step_gpu_vp,
)
from noeira.deep_agents.zero import value_prefix_from_rewards
from noeira.deep_agents.zero.mcts_adapters_mz import (
    MZRepGPU, MZDynGPU, MZPredGPU,
)
from noeira.planners.tree_search import GumbelGPUMCTS, SinglePlayer


def main() raises:
    comptime FRAMES = 4
    comptime ACT = 18
    comptime BINS = 601
    comptime Cfg = EZV2AtariConfig[FRAMES, ACT, LAYOUT=LAYOUT_NCHW]
    comptime OBS = Cfg.OBS
    comptime LATENT = Cfg.LATENT
    comptime HID = EZ_LSTM_HIDDEN
    comptime HORIZON = EZ_LSTM_HORIZON

    comptime N_ENVS = 2          # exercise fused EZDynVPNetAtari.forward at B>1
    comptime NUM_SIMS = 4
    comptime MAX_NODES = 16
    comptime MAX_K = 4

    comptime Rep = Cfg.Rep
    comptime Dyn = EZDynVPNetAtari[ACT, BINS]   # fused VP dynamics (drop-in)
    comptime Pred = Cfg.Pred
    comptime Proj = Cfg.Proj
    comptime Predh = Cfg.Predh

    print("=" * 70)
    print("EZv2-Atari value-prefix integration smoke (GPU)")
    print("=" * 70)

    var ctx = DeviceContext()
    var rep = Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn = Dyn.make["gpu", Kaiming](Optional(ctx))
    var pred = Pred.make["gpu", Kaiming](Optional(ctx))
    var proj = Proj.make["gpu", Kaiming](Optional(ctx))
    var predh = Predh.make["gpu", Kaiming](Optional(ctx))
    ez_atari_init_zero_pred["gpu", ACT, BINS](pred, ctx)
    ez_atari_init_zero_reward["gpu", BINS](dyn.rew, ctx)   # zero the LSTM head out
    rep.set_attr["training"](Scalar[DT](1.0))
    dyn.set_attr["training"](Scalar[DT](1.0))
    pred.set_attr["training"](Scalar[DT](1.0))
    proj.set_attr["training"](Scalar[DT](1.0))
    ctx.synchronize()

    # ── 1. SEARCH: real Gumbel search through the fused VP dynamics ──
    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](ctx, gamma=0.997, v_min=-300.0, v_max=300.0, qnorm_per_node=False)
    var rep_a = MZRepGPU[OBS, LATENT, Rep].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, Dyn].make(dyn)   # wraps EZDynVPNetAtari
    var pred_a = MZPredGPU[LATENT, ACT, BINS, Pred].make(pred)

    var d_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_obs = List[Scalar[DT]](length=N_ENVS * OBS, fill=0)
    var xs = UInt64(0x9E3779B97F4A7C15)
    for i in range(N_ENVS * OBS):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        h_obs[i] = Scalar[DT](Int(xs % 256)) / Scalar[DT](255.0)
    ctx.enqueue_copy(d_obs, h_obs.unsafe_ptr())
    var obs_t = tile[N_ENVS, OBS](d_obs.unsafe_ptr().as_unsafe_any_origin())
    planner.search_gpu[type_of(rep_a), type_of(dyn_a), type_of(pred_a)](
        ctx, rep_a, dyn_a, pred_a, obs_t,
        apply_legal=False, k_actual=MAX_K, rng_seed=UInt32(1))
    var h_pol = List[Scalar[DT]](length=N_ENVS * ACT, fill=0)
    var h_val = List[Scalar[DT]](length=N_ENVS, fill=0)
    ctx.enqueue_copy(h_pol.unsafe_ptr(), planner.policies_view())
    ctx.enqueue_copy(h_val.unsafe_ptr(), planner.root_value_view())
    ctx.synchronize()
    var psum = Float64(0.0)
    for a in range(ACT):
        assert_true(not isnan(h_pol[a]) and not isinf(h_pol[a]), "policy finite")
        psum += Float64(h_pol[a])
    assert_true(not isnan(h_val[0]) and not isinf(h_val[0]), "root value finite")
    assert_true(psum > 0.5 and psum < 1.5, "root policy ~normalized")
    print("  [1] Gumbel search through EZDynVPNetAtari OK — root value",
          h_val[0], "policy sum", Scalar[DT](psum))

    # ── 2. TRAIN: VP unroll on the SAME module's dyn.dynz / dyn.rew ──
    comptime B = 2
    comptime K = 2
    var orep = Adam(lr=Scalar[DT](0.02))
    var odynz = Adam(lr=Scalar[DT](0.02))
    var orew = Adam(lr=Scalar[DT](0.02))
    var opred = Adam(lr=Scalar[DT](0.02))
    var oproj = Adam(lr=Scalar[DT](0.02))
    var opredh = Adam(lr=Scalar[DT](0.02))

    var obs_seq = List[Scalar[DT]](length=(K + 1) * B * OBS, fill=0)
    for i in range((K + 1) * B * OBS):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        obs_seq[i] = Scalar[DT](Int(xs % 256)) / Scalar[DT](255.0)
    var actions = List[Scalar[DT]](length=K * B, fill=0)
    for i in range(K * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        actions[i] = Scalar[DT](Int(xs % ACT))
    var policy_tgt = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    for k in range(K + 1):
        for b in range(B):
            xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
            policy_tgt[k * B * ACT + b * ACT + Int(xs % ACT)] = Scalar[DT](1.0)
    var value_tgt = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    for i in range((K + 1) * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        value_tgt[i] = Scalar[DT](Int(xs % 200)) / Scalar[DT](100.0) - Scalar[DT](1.0)
    var reward_tgt = List[Scalar[DT]](length=K * B, fill=0)
    for i in range(K * B):
        xs = xs ^ (xs << 13); xs = xs ^ (xs >> 7); xs = xs ^ (xs << 17)
        reward_tgt[i] = Scalar[DT](Int(xs % 100)) / Scalar[DT](100.0)
    value_prefix_from_rewards[K, HORIZON](reward_tgt, B)

    var first = Scalar[DT](0.0)
    var last = Scalar[DT](0.0)
    for it in range(5):
        var l = ezv2_unroll_train_step_gpu_vp[
            Rep, type_of(dyn.dynz), Pred, Proj, Predh,
            B, K, OBS, ACT, LATENT, BINS, HID, HORIZON,
        ](
            ctx, rep, dyn.dynz, dyn.rew, pred, proj, predh,
            orep, odynz, orew, opred, oproj, opredh,
            obs_seq, actions, policy_tgt, value_tgt, reward_tgt,
            Scalar[DT](-300.0), Scalar[DT](300.0),
            consistency_coef=Scalar[DT](2.0),
        )
        if it == 0:
            first = l
        last = l
        assert_true(l == l, "VP train loss NaN")
    print("  [2] VP train on dyn.dynz/dyn.rew OK — loss", first, "→", last)
    assert_true(last < first, "VP integration train failed to reduce loss")

    _ = rep^; _ = dyn^; _ = pred^; _ = proj^; _ = predh^
    print("=" * 70)
    print("PASSED — value-prefix search + train integrate on GPU")
    print("=" * 70)
