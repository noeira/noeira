"""`FBCPROnlineAgent` smoke gate on a batched GPU Pendulum — the online
composition wires D's inputs from the right places.

  [1] the driver runs end to end through the CPR agent: train steps counted,
      FB losses finite, |B| pinned, the CPR terms finite.
  [2] D's NEGATIVES carry the STORED z: after one device step, `head.z_neg`
      equals the ring's `r_z` at the sampled rows, and `t.bz` (relabelled)
      differs from it on most rows — the snapshot preceded the relabel.
  [3] the expert windows are what the table says: `head.es[w·SEQ+j]` is the
      store row `start_w + j`, `head.esn` the row after, for every window,
      and every drawn start lies in the valid-start table.
  [4] two captured-path steps draw different z and MOVE D.
  [5] checkpoint: the FB file loads into the CPU `FBTrainer` the evals build,
      the `.cpr` sidecar restores D's logits.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_fb_cpr_online_smoke.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.data.resident import IDX_DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.deep_agents.fb.online_cpr import FBCPROnlineAgent
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
comptime SEQ = 4
comptime NW = BATCH // SEQ
comptime CAP = 1024
comptime LANES = 4
comptime ZBUF = 256
comptime HID = 32
comptime SEED = 20260908
comptime LEARNING_STARTS = 256
comptime SQRT_D = sqrt(Float64(D))
# synthetic expert store: EP-step episodes
comptime EXP_ROWS = 640
comptime EP = 64

comptime FNet = Sequential[Linear[OBS + ACT + D, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[OBS + D, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]
comptime DNet = Sequential[
    Linear[OBS + D, HID], LayerNorm[HID], Tanh[HID],
    Linear[HID, HID], ReLU[HID], Linear[HID, 1],
]
comptime QNet = Sequential[Linear[OBS + ACT + D, HID], ReLU[HID], Linear[HID, 1]]
comptime Agent = FBCPROnlineAgent[
    FNet, BNet, ANet, DNet, QNet, OBS, ACT, D, BATCH, CAP, LANES, SEQ, ZBUF
]
comptime CpuTrainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH, "cpu"]
comptime EnvT = BatchedGpuEnv[PendulumV2[DT], LANES, OBS, ACT]


def _download(ctx: DeviceContext, ref src: Tensor, n: Int) raises -> Tensor:
    var t = Tensor.alloc(n)
    t.ensure_gpu(ctx, n)
    ctx.enqueue_copy(t.dev.value(), src.dev.value())
    t.download(ctx)
    return t^


def _download_idx(
    ctx: DeviceContext, ref src: DeviceBuffer[IDX_DT], n: Int
) raises -> List[Int]:
    var h = ctx.enqueue_create_host_buffer[IDX_DT](n)
    ctx.enqueue_copy(h, src)
    ctx.synchronize()
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(Int(h[i]))
    return out^


def _make(ctx: DeviceContext) raises -> Agent:
    var a = Agent.make(
        ctx, lr=1e-3, lr_b=1e-3, lr_d=1e-3, lr_q=1e-3, ortho_weight=1.0,
        learning_starts=LEARNING_STARTS, z_hold=20, reg_coeff=0.1,
        gp_coef=10.0, seed=UInt64(SEED),
    )
    # a synthetic expert store: random rows, EP-step episodes, every valid
    # start (start + SEQ inside its episode)
    seed(SEED + 3)
    var obs = Tensor.alloc(EXP_ROWS * OBS)
    for i in range(EXP_ROWS * OBS):
        obs.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    obs.upload(ctx)
    var starts = List[Scalar[IDX_DT]]()
    for e in range(EXP_ROWS // EP):
        var off = e * EP
        for s0 in range(off, off + EP - SEQ):
            starts.append(Scalar[IDX_DT](s0))
    var n = len(starts)
    var sh = ctx.enqueue_create_host_buffer[IDX_DT](n)
    for i in range(n):
        sh[i] = starts[i]
    var sd = ctx.enqueue_create_buffer[IDX_DT](n)
    ctx.enqueue_copy(sd, sh)
    ctx.synchronize()
    a.attach_expert_windows(obs^, sd^, n)
    return a^


def main() raises:
    print("=" * 70)
    print("FBCPROnlineAgent smoke (batched GPU Pendulum)")
    print("=" * 70)
    var ctx = DeviceContext()

    print("[1] driver end to end through the CPR agent ...")
    seed(SEED)
    var a = _make(ctx)
    var env = EnvT(ctx)
    # The G1 driver resets on a schedule and marks the transition before each
    # reset (§12.23); this driver resets on `done`, which the store kernel
    # ORs in by itself. Called once so the forwarder is instantiated HERE and
    # not first on the box, where a compile error costs 30 minutes.
    a.set_boundary(False)
    comptime TOTAL = 2 * CAP
    _ = run_offpolicy_train_batched[
        Agent, EnvT, N_ENVS=LANES,
        USE_TRAIN_CUDA_GRAPH=False, USE_ENV_CUDA_GRAPH=False,
    ](
        Optional(ctx), a, env, TOTAL,
        rng_seed=UInt64(SEED), updates_per_step=1,
        print_every=CAP, verbose=True,
    )
    ctx.synchronize()
    var want_min = (TOTAL - LEARNING_STARTS) // LANES - 2
    print("      train steps:", a.total_train_steps(), " replay", a.replay_size())
    assert_true(a.total_train_steps() >= want_min, "too few train steps")
    var measure = Float64(0)
    var ortho = Float64(0)
    var actor = Float64(0)
    var fnv = Float64(0)
    var bn = Float64(0)
    var fbq = Float64(0)
    var fba = Float64(0)
    var mm = Float64(0)
    var qa = Float64(0)
    a.base.peek_losses(measure, ortho, actor, fnv, bn, fbq, fba, mm, qa)
    var d_pos = Float64(0)
    var d_neg = Float64(0)
    var r_mean = Float64(0)
    var q_mean = Float64(0)
    var q_loss = Float64(0)
    var q_pi = Float64(0)
    a.head.read_diag(d_pos, d_neg, r_mean, q_mean, q_loss, q_pi)
    print("      fb: measure", measure, " ortho", ortho, " actor", actor, " |B|", bn)
    print("      cpr: D+", d_pos, " D-", d_neg, " r", r_mean, " Q", q_mean,
          " Qloss", q_loss, " Qpi", q_pi)
    assert_true(measure == measure and ortho == ortho, "FB loss is NaN")
    assert_true(abs(bn - SQRT_D) < 1e-2, "|B| off the sqrt(D) pin: " + String(bn))
    assert_true(d_pos == d_pos and d_neg == d_neg and q_loss == q_loss, "CPR term is NaN")
    assert_true(d_pos > 0.0 and d_neg > 0.0, "BCE halves are not positive")

    print("[2] D's negatives carry the STORED z; the FB batch the relabelled one ...")
    a.train_device_kernels()
    ctx.synchronize()
    var idx = _download_idx(ctx, a.base.idx_s.value(), BATCH)
    var rz = a.base.ring_z_host()
    var zneg = _download(ctx, a.head.z_neg, BATCH * D)
    var bz = _download(ctx, a.base.t.bz, BATCH * D)
    var worst = Float64(0)
    var changed = 0
    for i in range(BATCH):
        var row = idx[i]
        var diff = Float64(0)
        for k in range(D):
            var e = abs(Float64(zneg.data[i * D + k]) - Float64(rz.data[row * D + k]))
            if e > worst:
                worst = e
            diff += abs(Float64(bz.data[i * D + k]) - Float64(zneg.data[i * D + k]))
        if diff > 1e-4:
            changed += 1
    print("      worst |z_neg − r_z[idx]| =", worst, "  relabelled rows:", changed, "/", BATCH)
    assert_true(worst < 1e-6, "z_neg is not the stored z of the sampled rows")
    assert_true(changed >= BATCH // 2, "relabel changed too few rows (keep_frac 0.2)")

    print("[3] expert windows match the store and the start table ...")
    var ws = _download_idx(ctx, a.win_start.value(), NW)
    var es = _download(ctx, a.head.es, BATCH * OBS)
    var esn = _download(ctx, a.head.esn, BATCH * OBS)
    var eobs = _download(ctx, a.exp_obs, EXP_ROWS * OBS)
    var we = Float64(0)
    for w in range(NW):
        var s0 = ws[w]
        assert_true(s0 >= 0 and s0 + SEQ < EXP_ROWS, "window start out of range")
        assert_true((s0 % EP) + SEQ < EP, "window crosses an episode end")
        for j in range(SEQ):
            for k in range(OBS):
                var e1 = abs(Float64(es.data[(w * SEQ + j) * OBS + k]) - Float64(eobs.data[(s0 + j) * OBS + k]))
                var e2 = abs(Float64(esn.data[(w * SEQ + j) * OBS + k]) - Float64(eobs.data[(s0 + j + 1) * OBS + k]))
                if e1 > we:
                    we = e1
                if e2 > we:
                    we = e2
    print("      worst |window row − store row| =", we, " over", NW, "windows")
    assert_true(we == 0.0, "expert window rows do not match the store")

    print("[4] two captured-path steps: different z, D moves ...")
    var probe_s = Tensor.alloc(BATCH * OBS)
    var probe_z = Tensor.alloc(BATCH * D)
    seed(SEED + 9)
    for i in range(BATCH * OBS):
        probe_s.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    for i in range(BATCH * D):
        probe_z.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    probe_s.upload(ctx)
    probe_z.upload(ctx)
    var l0 = Tensor()
    a.head.discriminate[BATCH](probe_s, probe_z, l0)
    l0.download(ctx)
    var z1 = _download(ctx, a.base.t.bz, BATCH * D)
    a.train_device_kernels()
    a.train_device_kernels()
    var z2 = _download(ctx, a.base.t.bz, BATCH * D)
    var l1 = Tensor()
    a.head.discriminate[BATCH](probe_s, probe_z, l1)
    l1.download(ctx)
    var dz = Float64(0)
    var dl = Float64(0)
    for i in range(BATCH * D):
        var x = abs(Float64(z1.data[i]) - Float64(z2.data[i]))
        if x > dz:
            dz = x
    for i in range(BATCH):
        var y = abs(Float64(l0.data[i]) - Float64(l1.data[i]))
        if y > dl:
            dl = y
    print("      max|Δz|", dz, " max|ΔD logit|", dl)
    assert_true(dz > 1e-3, "captured steps drew the same z")
    assert_true(dl > 1e-6, "D did not move over two device steps")

    print("[5] checkpoint: FB file into the CPU trainer, sidecar restores D ...")
    var path = String("/tmp/test_fb_cpr_online.ckpt")
    a.save_state(path)
    var b = _make(ctx)
    b.load_state(path)
    var l2 = Tensor()
    b.head.discriminate[BATCH](probe_s, probe_z, l2)
    l2.download(ctx)
    var wl = Float64(0)
    for i in range(BATCH):
        var y = abs(Float64(l1.data[i]) - Float64(l2.data[i]))
        if y > wl:
            wl = y
    print("      worst |D logit saved − loaded| =", wl)
    assert_true(wl < 1e-5, "sidecar did not restore D")
    var cpu = CpuTrainer.make(lr=1e-3)
    cpu.load_state(path)
    var sh = Tensor.alloc(BATCH * OBS)
    var zh = Tensor.alloc(BATCH * D)
    for i in range(BATCH * OBS):
        sh.data[i] = probe_s.data[i]
    for i in range(BATCH * D):
        zh.data[i] = probe_z.data[i]
    var ac = Tensor()
    cpu.act[BATCH](sh, zh, ac)
    var ag = Tensor()
    a.base.t.act[BATCH](probe_s, probe_z, ag)
    ag.download(ctx)
    var wa = Float64(0)
    for i in range(BATCH * ACT):
        var y = abs(Float64(ac.data[i]) - Float64(ag.data[i]))
        if y > wa:
            wa = y
    print("      worst |actor cpu-loaded − gpu| =", wa)
    assert_true(wa < 1e-4, "the CPU trainer reads a different actor from the FB file")
    print("\n[PASS] FBCPROnlineAgent smoke")
