"""`FBOnlineAgent` smoke gate — the online FB loop on a batched GPU env.

`docs/BFM_ZERO_SHOT_RL.md` §18.3 A3. Pendulum (`BatchedGpuEnv[PendulumV2]`),
four lanes, tiny nets: this gates WIRING, not learning. Every check below is
a property the loss curve could not show:

  [1] every lane's `z` sits on the radius-sqrt(D) sphere at init and after
      `z_hold` resamples, and every lane has CHANGED at least once by then —
      a lane whose z never moves trains a single-task policy.
  [2] the replay ring WRAPS: after 3·CAP env steps the fill is CAP, the write
      head is back at 0, and every stored `z` is on the sphere.
  [3] `relabel_ratio` does what it says: `keep_frac = 1` keeps every stored
      row, `0` keeps none, `0.2` keeps a binomial share — checked on a marker
      z, and every relabelled row is back on the sphere.
  [4] the driver runs end to end: train steps happen, the flush prints finite
      diagnostics, and `B`'s rows have not collapsed onto one direction.
  [5] capture safety: two `train_device_kernels` calls draw DIFFERENT batch z
      and move `B` differently — the device counters advance in-sequence.
  [6] a checkpoint written by the online agent loads into the plain CPU
      `FBTrainer` the eval scripts build, and `B` agrees with the GPU copy.
  [7] the driver's episode readback COUNTS at `episode_sync_every = 32` —
      the setting the walker runs use and the one the first online run
      reported `episodes 0` under (§18.7). Pendulum truncates at 200, so
      4096 steps over 4 lanes must complete ~20 episodes.
  [8] the expert slice (A3.5): store rows first, ring rows after.
  [9] an fp16 ring (`RING_DT = float16`) stores EXACTLY the fp32 ring's
      values rounded to fp16, saturates 1e6 to 65504 instead of +inf, and
      its gathered `s` / `z` batch is the fp32 agent's batch rounded the same
      way, element for element — with some element actually rounded, so the
      comparison is not fp32 against itself.

Run (Apple Metal or NVIDIA):
    pixi run -e apple mojo run -I . tests/deep_agents/test_fb_online_smoke.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.data.resident import IDX_DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.deep_agents.fb.online import FBOnlineAgent
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.training.batched_env import BatchedGpuEnv
from noeira.deep_agents.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)
from noeira.envs.pendulum.pendulum_v2 import PendulumV2


comptime OBS = 3
comptime ACT = 1
comptime D = 16
comptime BATCH = 64
comptime CAP = 1024
comptime LANES = 4
comptime ZBUF = 256
comptime HID = 32
comptime SEED = 20260907
comptime Z_HOLD = 20
comptime LEARNING_STARTS = 256
comptime SQRT_D = sqrt(Float64(D))

comptime FNet = Sequential[Linear[OBS + ACT + D, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[OBS + D, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]
comptime Agent = FBOnlineAgent[
    FNet, BNet, ANet, OBS, ACT, D, BATCH, CAP, LANES, ZBUF
]
comptime CpuTrainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]
comptime EnvT = BatchedGpuEnv[PendulumV2[DT], LANES, OBS, ACT]
comptime EXPERT = 32
comptime AgentX = FBOnlineAgent[
    FNet, BNet, ANet, OBS, ACT, D, BATCH, CAP, LANES, ZBUF, EXPERT
]
comptime Agent16 = FBOnlineAgent[
    FNet, BNet, ANet, OBS, ACT, D, BATCH, CAP, LANES, ZBUF, 0, 0, DType.float16
]


def _download_rows(
    ctx: DeviceContext, ref src: Tensor, n: Int
) raises -> Tensor:
    """Host copy of a device-only tensor (the ring has no host mirror)."""
    var h = Tensor.alloc(n)
    h.ensure_gpu(ctx, n)
    ctx.enqueue_copy(h.dev.value(), src.dev.value())
    h.download(ctx)
    return h^


def _rows_on_sphere(ref t: Tensor, rows: Int, tol: Float64) -> Int:
    """Rows whose norm is within `tol` of sqrt(D)."""
    var ok = 0
    for i in range(rows):
        var s = Float64(0)
        for k in range(D):
            var v = Float64(t.data[i * D + k])
            s += v * v
        if abs(sqrt(s) - SQRT_D) < tol:
            ok += 1
    return ok


def _rt(n: Int, ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    t.upload(ctx)
    return t^


def _make(ctx: DeviceContext) raises -> Agent:
    return Agent.make(
        ctx,
        lr=1e-3,
        learning_starts=LEARNING_STARTS,
        action_scale=2.0,
        z_hold=Z_HOLD,
        window_size=8,
        seed=UInt64(SEED),
    )


def test_lane_z(ctx: DeviceContext) raises:
    print("[1] per-lane z: on the sphere, and resampled within z_hold ...")
    var a = _make(ctx)
    var z0 = _download_rows(ctx, a.z_lane, LANES * D)
    var on0 = _rows_on_sphere(z0, LANES, 1e-4)
    print("      init: on sphere", on0, "/", LANES)
    assert_true(on0 == LANES, "a lane's initial z is off the sphere")

    var changed = List[Bool](length=LANES, fill=False)
    for _ in range(Z_HOLD + 1):
        a._resample_lanes(force=False)
        a._act_iter += 1
        var z1 = _download_rows(ctx, a.z_lane, LANES * D)
        assert_true(
            _rows_on_sphere(z1, LANES, 1e-4) == LANES,
            "a lane left the sphere after a resample",
        )
        for e in range(LANES):
            var d = Float64(0)
            for k in range(D):
                d += abs(Float64(z1.data[e * D + k]) - Float64(z0.data[e * D + k]))
            if d > 1e-6:
                changed[e] = True
    var n_changed = 0
    for e in range(LANES):
        if changed[e]:
            n_changed += 1
    print("      lanes resampled within", Z_HOLD + 1, "iters:", n_changed, "/", LANES)
    assert_true(
        n_changed == LANES,
        "a lane's z never changed across z_hold+1 iterations — the stagger"
        " or the hold test is wrong and that lane trains one task forever",
    )
    print("      OK")


def _kept(
    ctx: DeviceContext, mut ag: Agent, keep: Float64,
    mut s: Tensor, mut ac: Tensor, mut sn: Tensor, mut sp: Tensor,
    mut zm: Tensor,
) raises -> Int:
    """Rows still equal to the marker after one relabel at `keep`."""
    ag.keep_frac = keep
    ag.t.load_batch(s, ac, sn, sp, zm)
    ag._relabel_z()
    var out = _download_rows(ctx, ag.t.bz, BATCH * D)
    assert_true(
        _rows_on_sphere(out, BATCH, 1e-4) == BATCH,
        "a relabelled z row is off the sphere",
    )
    var kept = 0
    for i in range(BATCH):
        var d = Float64(0)
        for k in range(D):
            d += abs(Float64(out.data[i * D + k]) - Float64(zm.data[i * D + k]))
        if d < 1e-6:
            kept += 1
    return kept


def test_relabel(ctx: DeviceContext) raises:
    print("[3] relabel: keep_frac 1 keeps all, 0 keeps none, 0.2 a share ...")
    var a = _make(ctx)
    # Marker z: sqrt(D) on axis 0 — already on the sphere, so a KEPT row is
    # exactly the marker and a relabelled one is not.
    var zm = Tensor.alloc(BATCH * D)
    for i in range(BATCH):
        for k in range(D):
            zm.data[i * D + k] = Scalar[DT](SQRT_D if k == 0 else 0.0)
    zm.upload(ctx)
    var s = _rt(BATCH * OBS, ctx)
    var ac = _rt(BATCH * ACT, ctx)
    var sn = _rt(BATCH * OBS, ctx)
    var sp = _rt(BATCH * OBS, ctx)

    var k1 = _kept(ctx, a, 1.0, s, ac, sn, sp, zm)
    var k0 = _kept(ctx, a, 0.0, s, ac, sn, sp, zm)
    var k2 = _kept(ctx, a, 0.2, s, ac, sn, sp, zm)
    print("      kept: keep_frac=1.0 ->", k1, " 0.0 ->", k0, " 0.2 ->", k2, "of", BATCH)
    assert_true(k1 == BATCH, "keep_frac=1.0 relabelled a row")
    assert_true(k0 == 0, "keep_frac=0.0 kept a row")
    # Binomial(64, 0.2): mean 12.8, sd 3.2. ±4 sd.
    assert_true(k2 >= 1 and k2 <= 26, "keep_frac=0.2 kept an implausible share")
    print("      OK")


def test_driver_ring_and_capture(ctx: DeviceContext) raises:
    print("[2]+[4]+[5]+[6] driver end to end, ring wrap, capture, checkpoint ...")
    seed(SEED)
    var a = _make(ctx)
    var env = EnvT(ctx)
    comptime TOTAL = 3 * CAP
    _ = run_offpolicy_train_batched[
        Agent, EnvT, N_ENVS=LANES,
        USE_TRAIN_CUDA_GRAPH=False, USE_ENV_CUDA_GRAPH=False,
    ](
        Optional(ctx), a, env, TOTAL,
        rng_seed=UInt64(SEED), updates_per_step=1,
        print_every=CAP, verbose=True,
    )
    ctx.synchronize()

    print("      [2] ring: size", a.size, " pos", a.pos, " (CAP", CAP, ")")
    assert_true(a.size == CAP, "ring did not fill to CAP after 3*CAP steps")
    assert_true(a.pos == 0, "write head did not wrap back to 0")
    var rz = a.ring_z_host()
    var on = _rows_on_sphere(rz, CAP, 1e-3)
    print("      [2] stored z on sphere:", on, "/", CAP)
    assert_true(on == CAP, "a stored z row is off the sphere")

    print("      [4] train steps:", a.total_train_steps())
    var want_min = (TOTAL - LEARNING_STARTS) // LANES - 2
    assert_true(
        a.total_train_steps() >= want_min,
        "too few train steps (" + String(a.total_train_steps()) + " < "
        + String(want_min) + ") — the warmup gate or the size gate is wrong",
    )
    var measure = Float64(0)
    var ortho = Float64(0)
    var actor = Float64(0)
    var fnv = Float64(0)
    var bn = Float64(0)
    var fbq = Float64(0)
    var fba = Float64(0)
    var mm = Float64(0)
    var qa = Float64(0)
    a.peek_losses(measure, ortho, actor, fnv, bn, fbq, fba, mm, qa)
    print("      [4] losses: measure", measure, " ortho", ortho, " actor", actor,
          " |F|", fnv, " |B|", bn)
    assert_true(measure == measure and ortho == ortho, "loss is NaN")
    assert_true(abs(bn - SQRT_D) < 1e-2, "|B| off the sqrt(D) pin: " + String(bn))

    # B row spread on a probe: collapsed B => every row the same direction.
    var probe = _rt(BATCH * OBS, ctx)
    var b0 = Tensor()
    a.t.backward_embed[BATCH](probe, b0)
    b0.download(ctx)
    var spread = Float64(0)
    var pairs = 0
    for i in range(0, BATCH, 4):
        for j in range(i + 1, BATCH, 4):
            var d = Float64(0)
            for k in range(D):
                d += abs(Float64(b0.data[i * D + k]) - Float64(b0.data[j * D + k]))
            spread += d
            pairs += 1
    spread /= Float64(pairs)
    print("      [4] B row spread (mean L1 between probe rows):", spread)
    assert_true(spread > 0.1, "B rows collapsed onto one direction")

    print("      [5] two captured-path steps draw different z and move B ...")
    a.train_device_kernels()
    var z1 = _download_rows(ctx, a.t.bz, BATCH * D)
    var b1 = Tensor()
    a.t.backward_embed[BATCH](probe, b1)
    b1.download(ctx)
    a.train_device_kernels()
    var z2 = _download_rows(ctx, a.t.bz, BATCH * D)
    var b2 = Tensor()
    a.t.backward_embed[BATCH](probe, b2)
    b2.download(ctx)
    var dz = Float64(0)
    var db = Float64(0)
    for i in range(BATCH * D):
        var x = abs(Float64(z1.data[i]) - Float64(z2.data[i]))
        var y = abs(Float64(b1.data[i]) - Float64(b2.data[i]))
        if x > dz:
            dz = x
        if y > db:
            db = y
    print("      [5] max|Δz|", dz, " max|ΔB|", db)
    assert_true(dz > 1e-4, "the batch z did not change between two device steps")
    assert_true(db > 1e-9, "B did not move on the second device step")

    print("      [6] checkpoint loads into the CPU FBTrainer ...")
    var path = String("/tmp/fb_online_smoke.ckpt")
    a.save_state(path)
    var tc = CpuTrainer.make(lr=1e-3, ctx=None)
    tc.load_state(path)
    var probe_h = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        probe_h.data[i] = probe.data[i]
    var bg = Tensor()
    a.t.backward_embed[BATCH](probe, bg)
    bg.download(ctx)
    var bc = Tensor()
    tc.backward_embed[BATCH](probe_h, bc)
    var worst = Float64(0)
    for i in range(BATCH * D):
        var d = abs(Float64(bg.data[i]) - Float64(bc.data[i]))
        if d > worst:
            worst = d
    print("      [6] max |B_gpu - B_cpu| on the probe:", worst)
    assert_true(worst < 1e-3, "checkpoint round trip disagrees: " + String(worst))
    print("      OK")


def test_episode_readback_at_sync_32(ctx: DeviceContext) raises:
    print("[7] episode readback counts at episode_sync_every = 32 ...")
    seed(SEED)
    var a = _make(ctx)
    var env = EnvT(ctx)
    _ = run_offpolicy_train_batched[
        Agent, EnvT, N_ENVS=LANES,
        USE_TRAIN_CUDA_GRAPH=False, USE_ENV_CUDA_GRAPH=False,
    ](
        Optional(ctx), a, env, 4096,
        rng_seed=UInt64(SEED), updates_per_step=1,
        print_every=100_000, verbose=False, episode_sync_every=32,
    )
    # Pendulum: 200-step episodes, 4 lanes, 4096 steps -> 1024 iterations
    # -> 5 episodes per lane -> 20 total.
    print("      ep_count", a.ep_count(), " mean_ret", a.mean_return(), " (expect 20)")
    assert_true(
        a.ep_count() >= 16,
        "the driver counted " + String(a.ep_count()) + " episodes at"
        " episode_sync_every=32 — the deferred readback is dropping them",
    )
    print("      OK")


def test_expert_slice(ctx: DeviceContext) raises:
    """[8] A3.5: with an expert store attached, batch rows [0, EXPERT) carry
    the store's actions and rows [EXPERT, BATCH) the ring's; the BC mask is 1
    on the store rows and 0 on the ring's; and the captured step runs."""
    print("[8] expert slice: store rows first, ring rows after, mask split ...")
    seed(SEED)
    var a = AgentX.make(
        ctx, lr=1e-3, learning_starts=LEARNING_STARTS, action_scale=2.0,
        z_hold=Z_HOLD, window_size=8, seed=UInt64(SEED),
    )
    # Expert: 300 rows, random obs, CONSTANT action 0.37, identity next-row.
    comptime NE = 300
    var eobs = _rt(NE * OBS, ctx)
    var eact = Tensor.alloc(NE * ACT)
    for i in range(NE * ACT):
        eact.data[i] = Scalar[DT](0.37)
    eact.upload(ctx)
    var nh = ctx.enqueue_create_host_buffer[IDX_DT](NE)
    for r in range(NE):
        nh[r] = Scalar[IDX_DT](r)
    var nd = ctx.enqueue_create_buffer[IDX_DT](NE)
    ctx.enqueue_copy(nd, nh)
    a.attach_expert(eobs^, eact^, nd^, NE)
    # Ring: one record of LANES transitions with CONSTANT action 0.11.
    var ob = ctx.enqueue_create_buffer[DT](LANES * OBS)
    ob.enqueue_fill(Scalar[DT](0.5))
    var ac = ctx.enqueue_create_buffer[DT](LANES * ACT)
    ac.enqueue_fill(Scalar[DT](0.11))
    var rw = ctx.enqueue_create_buffer[DT](LANES)
    rw.enqueue_fill(Scalar[DT](0.0))
    var dn = ctx.enqueue_create_buffer[DT](LANES)
    dn.enqueue_fill(Scalar[DT](0.0))
    # THREE records, the first flagged as an episode boundary. The ring
    # derives `s'` from the row one env step later (§12.23), so:
    #   * the newest LANES rows have no successor yet and are excluded from
    #     the draw — one record would leave the window empty and fall into
    #     the sampler's degenerate clamp, a path the run never takes;
    #   * the flagged rows 0..LANES-1 must be remapped away, which with this
    #     fill means EVERY ring draw has to land at or above LANES.
    # This is also the only place `set_boundary` is instantiated before the
    # G1 driver's 30-minute build on the box.
    a.set_boundary(True)
    a.record_batch_gpu[LANES](ctx, ob, ac, rw, ob, dn)
    a.record_batch_gpu[LANES](ctx, ob, ac, rw, ob, dn)
    a.record_batch_gpu[LANES](ctx, ob, ac, rw, ob, dn)
    a._sample_batch()
    ctx.synchronize()
    var h_bnd = ctx.enqueue_create_host_buffer[DT](CAP)
    ctx.enqueue_copy(h_bnd, a.r_bnd.dev.value())
    var h_is = ctx.enqueue_create_host_buffer[IDX_DT](BATCH)
    ctx.enqueue_copy(h_is, a.idx_s.value())
    ctx.synchronize()
    # Only the 3 * LANES rows this section wrote: the rest of the ring is
    # still at its init fill of 1.0, which is deliberate (an unwritten row
    # must not read as a transition) and would swamp the count.
    var n_flag = 0
    var flag_outside = 0
    for r in range(3 * LANES):
        if Float64(h_bnd[r]) != 0.0:
            n_flag += 1
            if r >= LANES:
                flag_outside += 1
    var below = 0
    for i in range(BATCH - EXPERT):
        if Int(h_is[i]) < LANES:
            below += 1
    print("      [8] boundary rows flagged:", n_flag, "of the", 3 * LANES,
          "written (want", LANES, ", outside the first batch", flag_outside,
          ")  ring draws still on one:", below)
    assert_true(
        n_flag == LANES and flag_outside == 0,
        "`set_boundary` did not reach the ring, or reached the wrong rows:"
        " exactly the first record's LANES rows should be flagged",
    )
    assert_true(
        below == 0,
        "a ring draw returned a flagged row: its successor is the next"
        " episode's first observation, not its own next state",
    )
    var ba = _download_rows(ctx, a.t.ba, BATCH * ACT)
    var ok_e = 0
    var ok_r = 0
    for i in range(BATCH):
        var v = Float64(ba.data[i * ACT])
        if i < EXPERT and abs(v - 0.37) < 1e-6:
            ok_e += 1
        if i >= EXPERT and abs(v - 0.11) < 1e-6:
            ok_r += 1
    a.t.bc_mask.download(ctx)
    var ok_m = 0
    for i in range(BATCH):
        var want = 1.0 if i < EXPERT else 0.0
        if abs(Float64(a.t.bc_mask.data[i]) - want) < 1e-9:
            ok_m += 1
    print("      expert rows with the store's action:", ok_e, "/", EXPERT,
          "  ring rows with the ring's:", ok_r, "/", BATCH - EXPERT,
          "  mask rows right:", ok_m, "/", BATCH)
    assert_true(ok_e == EXPERT, "an expert row did not carry the store's action")
    assert_true(ok_r == BATCH - EXPERT, "a ring row did not carry the ring's action")
    assert_true(ok_m == BATCH, "the BC mask does not split at EXPERT_ROWS")
    # The captured-path step runs on the mixed batch, twice.
    a.train_device_kernels()
    a.train_device_kernels()
    ctx.synchronize()
    print("      OK")


def _f16(x: Scalar[DT]) -> Float64:
    """What the fp16 ring must hold for `x`: round-to-nearest fp16, with
    |x| above the fp16 max SATURATED (the host cast would give inf)."""
    if Float64(x) > 65504.0:
        return 65504.0
    if Float64(x) < -65504.0:
        return -65504.0
    return Float64(x.cast[DType.float16]().cast[DT]())


def _count_off(ref got: Tensor, ref ref32: Tensor, n: Int) -> Tuple[Int, Int]:
    """(elements of `got` != fp16(`ref32`), elements where fp16 changed the
    fp32 value at all) over the first `n`."""
    var bad = 0
    var rounded = 0
    for i in range(n):
        var want = _f16(ref32.data[i])
        if Float64(got.data[i]) != want:
            bad += 1
        if want != Float64(ref32.data[i]):
            rounded += 1
    return (bad, rounded)


def test_ring_fp16(ctx: DeviceContext) raises:
    """[9] the fp16 ring against the fp32 ring, same seed, same records."""
    print("[9] fp16 ring: stores fp16(x), saturates, gathers the same batch ...")
    seed(SEED)
    var a = _make(ctx)
    seed(SEED)
    var h = Agent16.make(
        ctx, lr=1e-3, learning_starts=LEARNING_STARTS, action_scale=2.0,
        z_hold=Z_HOLD, window_size=8, seed=UInt64(SEED),
    )
    var ob = ctx.enqueue_create_buffer[DT](LANES * OBS)
    var hob = ctx.enqueue_create_host_buffer[DT](LANES * OBS)
    var ac = ctx.enqueue_create_buffer[DT](LANES * ACT)
    ac.enqueue_fill(Scalar[DT](0.11))
    var rw = ctx.enqueue_create_buffer[DT](LANES)
    rw.enqueue_fill(Scalar[DT](0.0))
    var dn = ctx.enqueue_create_buffer[DT](LANES)
    dn.enqueue_fill(Scalar[DT](0.0))
    comptime NREC = 6
    for r in range(NREC):
        for i in range(LANES * OBS):
            # not fp16-representable (0.1-ish steps), one row out of range
            hob[i] = Scalar[DT](0.1 + 0.0123 * Float64(i) - 0.37 * Float64(r))
        if r == 2:
            hob[0] = Scalar[DT](1.0e6)
            hob[1] = Scalar[DT](-1.0e6)
        ctx.enqueue_copy(ob, hob)
        a.record_batch_gpu[LANES](ctx, ob, ac, rw, ob, dn)
        h.record_batch_gpu[LANES](ctx, ob, ac, rw, ob, dn)
    ctx.synchronize()
    comptime NW = NREC * LANES
    var o32 = a.ring_obs_host()
    var o16 = h.ring_obs_host()
    var z32 = a.ring_z_host()
    var z16 = h.ring_z_host()
    var so = _count_off(o16, o32, NW * OBS)
    var sz = _count_off(z16, z32, NW * D)
    var sat = Float64(o16.data[2 * LANES * OBS])
    var satn = Float64(o16.data[2 * LANES * OBS + 1])
    print("      [9] ring obs: off", so[0], "of", NW * OBS, "(rounded", so[1],
          ")  z: off", sz[0], "of", NW * D, "(rounded", sz[1], ")  1e6 ->", sat,
          " -1e6 ->", satn)
    assert_true(so[0] == 0, "the fp16 ring's obs is not fp16(the fp32 ring's)")
    assert_true(sz[0] == 0, "the fp16 ring's z is not fp16(the fp32 ring's)")
    assert_true(so[1] > 0 and sz[1] > 0, "nothing was rounded: [9] compared fp32 to fp32")
    assert_true(sat == 65504.0 and satn == -65504.0, "1e6 did not saturate to ±65504")

    a._gather_batch()
    h._gather_batch()
    ctx.synchronize()
    var bs32 = _download_rows(ctx, a.t.bs, BATCH * OBS)
    var bs16 = _download_rows(ctx, h.t.bs, BATCH * OBS)
    var bz32 = _download_rows(ctx, a.t.bz, BATCH * D)
    var bz16 = _download_rows(ctx, h.t.bz, BATCH * D)
    var bsn32 = _download_rows(ctx, a.t.bsn, BATCH * OBS)
    var bsn16 = _download_rows(ctx, h.t.bsn, BATCH * OBS)
    var gs = _count_off(bs16, bs32, BATCH * OBS)
    var gz = _count_off(bz16, bz32, BATCH * D)
    var gn = _count_off(bsn16, bsn32, BATCH * OBS)
    print("      [9] batch s: off", gs[0], "(rounded", gs[1], ")  s':", gn[0],
          "(rounded", gn[1], ")  z: off", gz[0], "(rounded", gz[1], ")")
    assert_true(gs[0] == 0 and gn[0] == 0, "the fp16 agent's gathered s / s' differ from fp16(fp32's)")
    assert_true(gz[0] == 0, "the fp16 agent's gathered z differs from fp16(fp32's)")
    assert_true(gs[1] > 0 and gz[1] > 0, "the gathered batch was never rounded")
    print("      OK")


def main() raises:
    print("=" * 70)
    print("FB online agent smoke — Pendulum, 4 lanes")
    print("=" * 70)
    var ctx = DeviceContext()
    test_lane_z(ctx)
    test_relabel(ctx)
    test_driver_ring_and_capture(ctx)
    test_episode_readback_at_sync_32(ctx)
    test_expert_slice(ctx)
    test_ring_fp16(ctx)
    print("\n[PASS] FB online smoke")
