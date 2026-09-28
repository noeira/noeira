"""The FB-CPR online agent at the G1's dimensions with BFM-Zero's towers and
the running normaliser — a smoke that runs on the laptop (G3.2, part 3).

    pixi run mojo build -I . -Xlinker -ld_classic tests/fb/test_bfm_agent_g1_dims_smoke.mojo -o /tmp/t && pixi run /tmp/t

⚠ `-Xlinker -ld_classic` ON macOS. Apple's current `ld` asserts on the
length of a mangled symbol name (`ld: Assertion failed: (name.size() <=
maxLength), SymbolString.cpp:74`), and the online CPR agent's nested
generics — the towers inside the trainer inside the head inside the agent
— exceed it. HEAD's `fb_online_cpr_walker_gpu.mojo` fails the same way on
this Mac (checked 2026-09-09); the classic linker takes it, with a
deprecation warning. Linux (the 5090 box) has no such limit.

No environment: the G1 batched env is NVIDIA-only, so this drives
`FBCPROnlineAgent` the way the driver does — `select_action_batched`,
`record_batch_gpu`, `train_step` — on random 527-D observations, with an
expert table of random rows attached as windows, and asks only what a
smoke can: the composition INSTANTIATES (`FTower`/`ActorTower`/`BNet`/`DNet`
inside the trainer's generics, on the flat `[s | a | z]` rows), the ring
fills, the normaliser's statistics move, the lane z stay on the radius-16
sphere, and after `N_TRAIN` updates every loss the head reports is finite.
Hidden width is 64 (the run uses 1024): the shapes that can go wrong do
not depend on it, and the laptop's Metal build stays under a minute.

Not a numerical gate — the towers (`test_bfm_towers_vs_torch`), the
normaliser (`test_obs_ema_vs_batchnorm`) and the FB step (the walker-era
FB gates) each have their own.
"""

from std.math import sqrt, abs, isnan, isinf
from std.testing import assert_true, TestSuite

from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.data.resident import IDX_DT
from noeira.deep_agents.fb import FBCPROnlineAgent
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMBNetFiltered, BFMActorTowerFiltered, BFMDNetFiltered,
)
from noeira.deep_agents.fb.kernels import ensure_t
from noeira.envs.robots.unitree_g1_history import g1_build_tail_spec


# ⚠ TWO widths now (§12.34-12.36). `SP` is `state + privileged_state`, the
# 527 that `b` and `discriminator` consume — unchanged. `OBS` is the PACKED
# row `f`, `critic` and (filtered) the actor see: `SP + last_action 29 +
# history 372`. The env still produces SP; the 401 is assembled by the driver
# or derived from the ring.
comptime SP = 527
comptime SD = 64
comptime EX = 401
comptime OBS = SP + EX
comptime ACT = 29
comptime D = 256
comptime H = 64
comptime L = 3
comptime HB = 256
comptime HD = 128
comptime BATCH = 64
comptime LANES = 16
comptime CAP = 4096
comptime SEQ = 8
comptime ZBUF = 256
comptime N_EXPERT = 512
comptime N_TRAIN = 6

comptime FNet = BFMFTower[OBS, ACT, D, H, L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, HB]
comptime ANet = BFMActorTowerFiltered[OBS, SD, EX, D, H, L, ACT]
comptime DNet = BFMDNetFiltered[OBS, SP, D, HD]
comptime QNet = BFMFTower[OBS, ACT, D, H, L, 1]
comptime Agent = FBCPROnlineAgent[
    FNet, BNet, ANet, DNet, QNet, OBS, ACT, D, BATCH, CAP, LANES, SEQ, ZBUF,
    EX,   # DERIVED_TAIL: the 401 the ring does NOT store (docs §12.36)
]


def _fill(mut t: Tensor, n: Int, seed: Int, scale: Float64):
    for i in range(n):
        var k = (i * 7919 + seed * 104729) % 1000
        t.data[i] = Scalar[DT]((Float64(k) / 999.0 * 2.0 - 1.0) * scale)


def test_agent_instantiates_and_trains_finite() raises:
    var ctx = DeviceContext()
    var agent = Agent.make(
        ctx, learning_starts=BATCH, z_hold=100, zbuf_frac=0.5,
        keep_frac=0.2, p_goal=0.2, p_expert=0.6, expl_std=0.05,
        reg_coeff=0.05, gp_coef=10.0, normalize_obs=True,
    )

    # expert table: N_EXPERT random 527-D rows, one episode, every window
    # start that keeps `start + SEQ` inside it
    var eobs = Tensor()
    ensure_t["gpu"](eobs, N_EXPERT * SP, Optional(ctx))
    _fill(eobs, N_EXPERT * SP, 3, 2.0)
    eobs.upload(ctx)
    var n_starts = N_EXPERT - SEQ
    var h_starts = ctx.enqueue_create_host_buffer[IDX_DT](n_starts)
    for i in range(n_starts):
        h_starts[i] = Scalar[IDX_DT](i)
    var starts = ctx.enqueue_create_buffer[IDX_DT](n_starts)
    ctx.enqueue_copy(starts, h_starts)
    ctx.synchronize()
    agent.attach_expert_windows(eobs^, starts^, n_starts)

    # ⚠ With DERIVED_TAIL the ring stores SP columns and the batch row is OBS
    # wide; the last EX are derived from the SAME lane's earlier rows
    # (§12.36). Two things the agent cannot infer and the env must hand it:
    # the spec table, and the scaling the G1's PD chain applies to a stored
    # action (`clip(a * 5, +-5)`) — the tail must reproduce what the policy
    # SAW, not what the net emitted.
    var spec = List[Int32]()
    g1_build_tail_spec(spec)
    agent.attach_tail_spec(spec)
    agent.set_action_norm(5.0, 5.0)

    # rollout buffers
    var obs = Tensor()
    var prev = Tensor()
    var act = Tensor()
    var rew = Tensor()
    var done = Tensor()
    ensure_t["gpu"](obs, LANES * SP, Optional(ctx))
    ensure_t["gpu"](prev, LANES * SP, Optional(ctx))
    ensure_t["gpu"](act, LANES * ACT, Optional(ctx))
    ensure_t["gpu"](rew, LANES, Optional(ctx))
    ensure_t["gpu"](done, LANES, Optional(ctx))
    var ao = ctx.enqueue_create_buffer[DT](LANES * 2 * ACT)
    var alp = ctx.enqueue_create_buffer[DT](LANES * (ACT + 1))
    for i in range(LANES):
        rew.data[i] = Scalar[DT](0)
        done.data[i] = Scalar[DT](0)
    rew.upload(ctx)
    done.upload(ctx)

    var n_steps = (BATCH // LANES) * 2 + N_TRAIN * 2
    var n_trained = 0
    for t in range(n_steps):
        agent.set_age(t if t < 6 else 6)
        _fill(prev, LANES * SP, 10 + t, 3.0)
        _fill(obs, LANES * SP, 11 + t, 3.0)
        prev.upload(ctx)
        obs.upload(ctx)
        agent.select_action_batched[LANES](
            LayoutTensor[DT, Layout.row_major(LANES, OBS), MutAnyOrigin](prev.dev.value()),
            LayoutTensor[DT, Layout.row_major(LANES, ACT), MutAnyOrigin](act.dev.value()),
            LayoutTensor[DT, Layout.row_major(LANES, 2 * ACT), MutAnyOrigin](ao),
            LayoutTensor[DT, Layout.row_major(LANES, ACT + 1), MutAnyOrigin](alp),
            t * LANES,
        )
        agent.record_batch_gpu[LANES](
            ctx, prev.dev.value(), act.dev.value(), rew.dev.value(),
            obs.dev.value(), done.dev.value(),
        )
        if agent.train_step(t * LANES):
            n_trained += 1
            if n_trained >= N_TRAIN:
                break
    ctx.synchronize()
    print("  ring size", agent.base.size, " train steps", n_trained)
    assert_true(n_trained >= N_TRAIN, "no training steps ran")

    # the lane z on the sphere
    agent.base.z_lane.download(ctx)
    ctx.synchronize()
    var worst_norm = 0.0
    for l in range(LANES):
        var s = 0.0
        for k in range(D):
            var v = Float64(agent.base.z_lane.data[l * D + k])
            s += v * v
        var e = abs(sqrt(s) - sqrt(Float64(D)))
        if e > worst_norm:
            worst_norm = e
    print("  lane z: worst | |z| - 16 |", worst_norm)
    assert_true(worst_norm < 1e-3, "a lane z left the sphere")

    # the normaliser moved
    agent.base.obs_ema.sync_host()
    var moved = 0.0
    for d in range(OBS):
        var m = abs(Float64(agent.base.obs_ema.mean.data[d]))
        if m > moved:
            moved = m
    print("  normaliser: updates", agent.base.obs_ema.n_updates, " max |mean|", moved)
    assert_true(agent.base.obs_ema.n_updates == 2 * N_TRAIN, "normaliser did not update twice per train step")
    assert_true(moved > 0.0, "normaliser statistics never moved")

    # the losses are finite
    var measure = 0.0
    var ortho = 0.0
    var actor = 0.0
    var f_norm = 0.0
    var b_norm = 0.0
    var fbq = Float64(0)
    var fba = Float64(0)
    var mm = Float64(0)
    var qa = Float64(0)
    agent.base.peek_losses(
        measure, ortho, actor, f_norm, b_norm, fbq, fba, mm, qa
    )
    print("  losses: measure", measure, " ortho", ortho, " actor", actor, " |F|", f_norm, " |B|", b_norm)
    var vals = List[Float64]()
    vals.append(measure)
    vals.append(ortho)
    vals.append(actor)
    vals.append(f_norm)
    vals.append(b_norm)
    for i in range(len(vals)):
        assert_true(not isnan(vals[i]) and not isinf(vals[i]), "a loss is not finite")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
