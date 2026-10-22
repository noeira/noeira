"""`FBMebeOnlineAgent` smoke gate — FB-MEBE's online loop on a batched GPU env.

`noeira-docs/FB_MEBE_G1_PLAN.md` §2. Pendulum (`BatchedGpuEnv[PendulumV2]`,
obs `[cos θ, sin θ, θ̇]`), four lanes, tiny nets: WIRING, not learning.

  [1] a pool refresh writes `zsrc`: every row on the radius-sqrt(D) sphere,
      different from the uniform rows it started with, and every pooled
      observation IS a row of the ring (the gather read the ring, not junk).
  [2] the draw is INVERSE-density: with β = 2 the pooled states sit in the
      tails of θ̇ more often than the ring does (share of |θ̇| above the
      ring's median), and with β = 0 they do not.
  [3] the validity box is honoured: with a filter `θ̇ <= 0`, no pooled
      state has θ̇ > 0.
  [4] boundaries follow the truncation schedule (`episode_len` — the
      driver hands the agent `terminated` only), and the regularity reward
      column is what the header says, row by row:
      `−w_acc·((θ̇' − θ̇)/dt)² − w_ar·(a − a_prev)²`, recomputed on the host
      from the ring's own obs and actions (a_prev = 0 after a reset), with
      r = 0 on the boundary rows themselves.
  [5] the driver runs end to end with the training step captured: train
      steps happen, refreshes happen, the reg critic's terms are finite.
  [6] prior rows (`attach_prior`, plan §13.5): rows marked by θ̇ >= 100,
      which the ring never reaches, and by their reward. After a training
      step, batch rows [0, PRIOR_ROWS) are prior rows and the rest ring
      rows, and `br` is each row's OWN reward on both sides (the prior's
      from its table, the ring's from `r_rew[idx_s]`); a pool refresh
      draws some prior rows and nothing that is in neither table; the
      mixed eval embedding returns a prior row's raw obs.

Run (Apple Metal or NVIDIA):
    pixi run -e apple mojo run -I . tests/deep_agents/test_fb_mebe_online_smoke.mojo
"""

from std.math import abs, sqrt
from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.data.resident import IDX_DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.deep_agents.fb.online_mebe import (
    FBMebeOnlineAgent,
    MebeConfig,
    RegRewardConfig,
)
from noeira.deep_agents.fb.density import InverseDensityConfig
from noeira.deep_agents.fb.prior import PriorPrice, prior_rows
from noeira.deep_agents.training.batched_env import BatchedGpuEnv
from noeira.deep_agents.training.driver_offpolicy import (
    OffPolicyAgentGpu,
    run_offpolicy_train_batched,
)
from noeira.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS = 3
comptime ACT = 1
comptime D = 16
comptime BATCH = 64
comptime CAP = 2048
comptime LANES = 4
comptime ZBUF = 256
comptime HID = 32
comptime SEED = 20261012
comptime LEARNING_STARTS = 256
comptime SQRT_D = sqrt(Float64(D))
comptime DT_CTRL = 0.05
comptime W_ACC = 1e-3
comptime W_AR = 0.1
comptime EP_LEN = 200    # Pendulum's time limit; every lane truncates together

comptime FNet = Sequential[Linear[OBS + ACT + D, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[OBS + D, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]
comptime QNet = Sequential[Linear[OBS + ACT + D, HID], ReLU[HID], Linear[HID, 1]]
comptime Agent = FBMebeOnlineAgent[
    FNet, BNet, ANet, QNet, OBS, ACT, D, BATCH, CAP, LANES, ZBUF
]
comptime EnvT = BatchedGpuEnv[PendulumV2[DT], LANES, OBS, ACT]
comptime PRIOR = 16
comptime AgentP = FBMebeOnlineAgent[
    FNet, BNet, ANet, QNet, OBS, ACT, D, BATCH, CAP, LANES, ZBUF, 0, DT, PRIOR
]
comptime P_EPS = 6       # prior episodes ...
comptime P_LEN = 11      # ... of 11 rows: 10 transitions + the terminal row
comptime TH_MARK = 100.0


def _dl(ctx: DeviceContext, ref src: Tensor, n: Int) raises -> Tensor:
    var h = Tensor.alloc(n)
    h.ensure_gpu(ctx, n)
    ctx.enqueue_copy(h.dev.value(), src.dev.value())
    h.download(ctx)
    return h^


def _cfg(beta: Float64, filter_neg: Bool) -> MebeConfig:
    # φ = θ̇ alone: a 1-D density whose rare region is unambiguous (the tails),
    # so leg [2] tests the weighting rather than a guess about a 2-D shape
    var c = MebeConfig([2])
    c.density = InverseDensityConfig(beta=beta, eps_rel=0.1, bw_scale=1.0)
    c.n_cand = 1024
    c.n_ref = 512
    c.refresh_every = 8
    if filter_neg:
        c.add_filter(2, -1e9, 0.0)
    return c^


def _make(ctx: DeviceContext, beta: Float64, filter_neg: Bool) raises -> Agent:
    return Agent.make(
        ctx, _cfg(beta, filter_neg),
        RegRewardConfig(qd_off=2, qd_n=1, dt=DT_CTRL, w_acc=W_ACC, w_ar=W_AR, w_env=0.0),
        mebe=True, use_reg=True, reg_coeff=20.0, lr=1e-3, lr_q=1e-3,
        learning_starts=LEARNING_STARTS, action_scale=2.0, z_hold=20,
        window_size=8, seed=UInt64(SEED), episode_len=EP_LEN,
    )


def _fill[AT: OffPolicyAgentGpu](ctx: DeviceContext, mut a: AT, steps: Int) raises:
    var env = EnvT(ctx)
    _ = run_offpolicy_train_batched[
        AT, EnvT, N_ENVS=LANES,
        USE_TRAIN_CUDA_GRAPH=False, USE_ENV_CUDA_GRAPH=False,
    ](
        Optional(ctx), a, env, steps,
        rng_seed=UInt64(SEED), updates_per_step=1,
        print_every=1_000_000, verbose=False,
    )
    ctx.synchronize()


def _tail_share(ref obs: Tensor, ref rows: List[Int], med: Float64) -> Float64:
    var c = 0
    for i in range(len(rows)):
        if abs(Float64(obs.data[rows[i] * OBS + 2])) > med:
            c += 1
    return Float64(c) / Float64(len(rows))


def _median_abs_thdot(ref ring: Tensor, n: Int) -> Float64:
    var v = List[Float64]()
    for r in range(n):
        v.append(abs(Float64(ring.data[r * OBS + 2])))
    sort(v)
    return v[n // 2]


def _pool_rows_in_ring(
    ref pool: Tensor, ref ring: Tensor, n_ring: Int, mut rows: List[Int]
) -> Int:
    """Rows of the ring each pooled obs equals (exactly); count misses."""
    var miss = 0
    for i in range(BATCH):
        var found = -1
        for r in range(n_ring):
            var eq = True
            for k in range(OBS):
                if pool.data[i * OBS + k] != ring.data[r * OBS + k]:
                    eq = False
                    break
            if eq:
                found = r
                break
        if found < 0:
            miss += 1
        else:
            rows.append(found)
    return miss


def test_pool(ctx: DeviceContext) raises:
    print("[1]+[2] refresh: zsrc on sphere, pooled obs are ring rows, inverse density ...")
    seed(SEED)
    var a = _make(ctx, 2.0, False)
    var z0 = _dl(ctx, a.base.zsrc, BATCH * D)
    _fill(ctx, a, 1600)
    a.refresh_pool()
    ctx.synchronize()
    var z1 = _dl(ctx, a.base.zsrc, BATCH * D)
    var on = 0
    var moved = 0
    for i in range(BATCH):
        var s = Float64(0)
        var dz = Float64(0)
        for k in range(D):
            var v = Float64(z1.data[i * D + k])
            s += v * v
            dz += abs(v - Float64(z0.data[i * D + k]))
        if abs(sqrt(s) - SQRT_D) < 1e-3:
            on += 1
        if dz > 1e-3:
            moved += 1
    print("      zsrc on sphere", on, "/", BATCH, "  changed", moved, "/", BATCH)
    assert_true(on == BATCH, "a pooled z is off the sphere")
    assert_true(moved == BATCH, "a pooled z still holds its init value")
    var n = a.base.size
    var ring = a.base.ring_obs_host()
    var pool = _dl(ctx, a._pool_obs, BATCH * OBS)
    var rows2 = List[Int]()
    var miss = _pool_rows_in_ring(pool, ring, n, rows2)
    print("      pooled obs not found in the ring:", miss)
    assert_true(miss == 0, "a pooled observation is not a ring row")
    var med = _median_abs_thdot(ring, n)
    var share2 = _tail_share(ring, rows2, med)
    print("      β=2: share of pooled states with |θ̇| above the ring's median", share2,
          " (ess", a.last_report.ess_frac, ")")

    seed(SEED)
    var b = _make(ctx, 0.0, False)
    _fill(ctx, b, 1600)
    b.refresh_pool()
    ctx.synchronize()
    var ring0 = b.base.ring_obs_host()
    var pool0 = _dl(ctx, b._pool_obs, BATCH * OBS)
    var rows0 = List[Int]()
    _ = _pool_rows_in_ring(pool0, ring0, b.base.size, rows0)
    var share0 = _tail_share(ring0, rows0, _median_abs_thdot(ring0, b.base.size))
    print("      β=0: same share", share0)
    assert_true(share2 > 0.6, "β=2 did not favour the tails of θ̇")
    assert_true(share2 > share0 + 0.1, "β=2 not above β=0")


def test_filter(ctx: DeviceContext) raises:
    print("[3] filter θ̇ <= 0 honoured ...")
    seed(SEED)
    var a = _make(ctx, 2.0, True)
    _fill(ctx, a, 1600)
    a.refresh_pool()
    ctx.synchronize()
    var pool = _dl(ctx, a._pool_obs, BATCH * OBS)
    var bad = 0
    for i in range(BATCH):
        if Float64(pool.data[i * OBS + 2]) > 0.0:
            bad += 1
    print("      pooled states with θ̇ > 0:", bad, " valid cands",
          a.last_report.n_valid_cand, "/", a.cfg.n_cand)
    assert_true(bad == 0, "the validity box was not honoured")


def test_reward_column(ctx: DeviceContext) raises:
    print("[4] boundaries on the truncation schedule; r_reg == host recomputation ...")
    seed(SEED)
    var a = _make(ctx, 2.0, False)
    _fill(ctx, a, 1200)          # < CAP: no wrap, rows 0..size-1 in order
    var n = a.base.size
    var ring = a.base.ring_obs_host()
    var act = _dl(ctx, a.base.r_act, CAP * ACT)
    var bnd = _dl(ctx, a.base.r_bnd, CAP)
    var rr = _dl(ctx, a.r_rew, CAP)
    # (a) a row is a boundary iff it is the last step of an episode — and the
    # observation really jumps there (θ̇ resets), so the schedule is the env's
    var wrong_bnd = 0
    for r in range(n - LANES):
        var last = ((r // LANES) + 1) % EP_LEN == 0
        if last != (Float64(bnd.data[r]) != 0.0):
            wrong_bnd += 1
    print("      boundary flags off-schedule:", wrong_bnd)
    assert_true(wrong_bnd == 0, "boundary flags do not follow the truncation schedule")
    # (b) the reward of every in-episode row
    var checked = 0
    var worst = Float64(0)
    var checked_reset = 0
    var zero_at_bnd = 0
    # row r = step (r // LANES), lane (r % LANES); its successor is r + LANES
    for r in range(LANES, n - LANES):
        if Float64(bnd.data[r]) != 0.0:
            if Float64(rr.data[r]) == 0.0:
                zero_at_bnd += 1
            continue
        var prev_crosses = Float64(bnd.data[r - LANES]) != 0.0
        var a_prev = 0.0 if prev_crosses else Float64(act.data[r - LANES])
        var dq = (Float64(ring.data[(r + LANES) * OBS + 2])
                  - Float64(ring.data[r * OBS + 2])) / DT_CTRL
        var da = Float64(act.data[r]) - a_prev
        var want = -W_ACC * dq * dq - W_AR * da * da
        var err = abs(Float64(rr.data[r]) - want)
        if err > worst:
            worst = err
        checked += 1
        if prev_crosses:
            checked_reset += 1
    print("      rows checked", checked, " (first-after-reset", checked_reset,
          ")  worst |Δ|", worst, "  boundary rows with r = 0:", zero_at_bnd)
    assert_true(checked > 500, "too few rows checked")
    assert_true(checked_reset >= LANES, "no post-reset row checked — a_prev reset untested")
    assert_true(zero_at_bnd >= LANES, "boundary rows do not carry r = 0")
    assert_true(worst < 1e-4, "r_reg differs from the host recomputation")


def test_driver(ctx: DeviceContext) raises:
    print("[5] driver end to end with the step captured ...")
    seed(SEED)
    var a = _make(ctx, 2.0, False)
    var env = EnvT(ctx)
    _ = run_offpolicy_train_batched[
        Agent, EnvT, N_ENVS=LANES,
        USE_TRAIN_CUDA_GRAPH=True, USE_ENV_CUDA_GRAPH=False,
    ](
        Optional(ctx), a, env, 4096,
        rng_seed=UInt64(SEED), updates_per_step=2,
        print_every=2048, verbose=True,
    )
    ctx.synchronize()
    var r_mean = Float64(0)
    var q_mean = Float64(0)
    var q_loss = Float64(0)
    var q_pi = Float64(0)
    a.head.read_diag(r_mean, q_mean, q_loss, q_pi)
    print("      train steps", a.total_train_steps(), " refreshes", a._n_refresh,
          " r", r_mean, " Q", q_mean, " Qloss", q_loss, " Qpi", q_pi)
    assert_true(a.total_train_steps() > 1000, "too few train steps")
    assert_true(a._n_refresh > 50, "the pool was not refreshed")
    for x in [r_mean, q_mean, q_loss, q_pi]:
        assert_true(x == x and abs(x) < 1e9, "non-finite reg term")


def _dl_idx(ctx: DeviceContext, ref src: DeviceBuffer[IDX_DT], n: Int) raises -> List[Int]:
    var h = ctx.enqueue_create_host_buffer[IDX_DT](n)
    ctx.enqueue_copy(h, src)
    ctx.synchronize()
    var out = List[Int](length=n, fill=0)
    for i in range(n):
        out[i] = Int(h[i])
    return out^


def test_prior(ctx: DeviceContext) raises:
    print("[6] prior rows: batch split, own rewards, pool and eval draw them ...")
    seed(SEED)
    var a = AgentP.make(
        ctx, _cfg(2.0, False),
        RegRewardConfig(qd_off=2, qd_n=1, dt=DT_CTRL, w_acc=W_ACC, w_ar=W_AR, w_env=0.0),
        mebe=True, use_reg=True, reg_coeff=20.0, lr=1e-3, lr_q=1e-3,
        learning_starts=LEARNING_STARTS, action_scale=2.0, z_hold=20,
        window_size=8, seed=UInt64(SEED), episode_len=EP_LEN,
    )
    # the store: P_EPS episodes, θ̇ = TH_MARK + store row (a unique marker)
    var n = P_EPS * P_LEN
    var obs = List[Float32](length=n * OBS, fill=0)
    var act = List[Float32](length=n * ACT, fill=0)
    var r_env = List[Float32](length=n, fill=0)
    var off = List[Int]()
    var ln = List[Int]()
    for e in range(P_EPS):
        off.append(e * P_LEN)
        ln.append(P_LEN)
    for r in range(n):
        obs[r * OBS] = Float32(0.6)
        obs[r * OBS + 1] = Float32(0.8)
        obs[r * OBS + 2] = Float32(TH_MARK + Float64(r))
        act[r] = Float32(0.01 * Float64(r % 7))
    var pr = prior_rows(
        obs, act, r_env, off, ln, OBS, ACT,
        RegRewardConfig(qd_off=2, qd_n=1, dt=DT_CTRL, w_acc=W_ACC, w_ar=W_AR, w_env=0.0),
        PriorPrice.none(),
    )
    var p_obs = List[Scalar[DT]](length=(pr.n_valid + pr.n_term) * OBS, fill=0)
    for i in range(len(p_obs)):
        p_obs[i] = pr.obs.data[i]
    var p_rew = List[Scalar[DT]](length=pr.n_valid, fill=0)
    for i in range(pr.n_valid):
        p_rew[i] = pr.rew.data[i]
    var nxt = pr.upload(ctx)
    var nv = pr.n_valid
    a.attach_prior(pr.take_obs(), pr.take_act(), nxt^, pr.take_rew(), nv)
    _fill(ctx, a, 1600)
    _ = a.train_step(1_000_000)
    ctx.synchronize()
    var bs = _dl(ctx, a.base.t.bs, BATCH * OBS)
    var br = _dl(ctx, a.head.br, BATCH)
    var rr = _dl(ctx, a.r_rew, CAP)
    var ie = _dl_idx(ctx, a.base.idx_e.value(), PRIOR)
    var isr = _dl_idx(ctx, a.base.idx_s.value(), BATCH)
    var wrong_side = 0
    var wrong_rew = 0
    for i in range(BATCH):
        var th = Float64(bs.data[i * OBS + 2])
        if i < PRIOR:
            if th < TH_MARK:
                wrong_side += 1
            # the row's own table row, found by its marker
            var row = ie[i]
            if Float64(p_obs[row * OBS + 2]) != th or br.data[i] != p_rew[row]:
                wrong_rew += 1
        else:
            if th >= TH_MARK:
                wrong_side += 1
            if br.data[i] != rr.data[isr[i - PRIOR]]:
                wrong_rew += 1
    var live = 0
    for i in range(PRIOR):
        if abs(Float64(br.data[i])) > 1e-6:
            live += 1
    print("      batch rows on the wrong side", wrong_side, "  rows whose br is not their own",
          wrong_rew, "  prior rows with r != 0", live, "/", PRIOR)
    assert_true(wrong_side == 0, "prior rows are not exactly batch rows [0, PRIOR_ROWS)")
    assert_true(wrong_rew == 0, "a batch row carries another row's regularity reward")
    assert_true(live > PRIOR // 2, "the prior rows' rewards are all zero — the check is vacuous")

    a.refresh_pool()
    ctx.synchronize()
    var pool = _dl(ctx, a._pool_obs, BATCH * OBS)
    var ring = a.base.ring_obs_host()
    var from_p = 0
    var nowhere = 0
    for i in range(BATCH):
        var th = Float64(pool.data[i * OBS + 2])
        if th >= TH_MARK:
            var r = Int(th - TH_MARK + 0.5)
            # a transition row of the table holds this marker
            var hit = False
            for j in range(nv):
                if p_obs[j * OBS + 2] == pool.data[i * OBS + 2]:
                    hit = True
            if hit and r < n:
                from_p += 1
            else:
                nowhere += 1
        else:
            var hit = False
            for r in range(a.base.size):
                if ring.data[r * OBS + 2] == pool.data[i * OBS + 2]:
                    hit = True
                    break
            if not hit:
                nowhere += 1
    print("      pooled rows from the prior", from_p, "/", BATCH, " (share reported",
          a.last_pool_prior, ")  in neither table", nowhere)
    assert_true(from_p > 0, "the pool never draws a prior row")
    assert_true(nowhere == 0, "a pooled row is in neither table")
    assert_true(abs(a.last_pool_prior - Float64(from_p) / Float64(BATCH)) < 1e-9,
                "the reported pool share is not the pool's")

    var rows: List[Int] = [3, 0, 5]
    var fp: List[Bool] = [True, False, True]
    var raw = List[Scalar[DT]]()
    var bb = List[Scalar[DT]]()
    a.embed_rows_mixed(rows, fp, raw, bb)
    var bad = 0
    for k in range(OBS):
        if raw[k] != p_obs[3 * OBS + k] or raw[2 * OBS + k] != p_obs[5 * OBS + k]:
            bad += 1
        if raw[OBS + k] != ring.data[k]:
            bad += 1
    print("      mixed eval embedding: entries off", bad)
    assert_true(bad == 0, "embed_rows_mixed did not read the row's own table")


def main() raises:
    var ctx = DeviceContext()
    test_pool(ctx)
    test_filter(ctx)
    test_reward_column(ctx)
    test_driver(ctx)
    test_prior(ctx)
    print("ALL PASS — fb mebe online smoke")
