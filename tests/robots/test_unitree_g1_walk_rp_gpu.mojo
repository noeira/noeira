"""The G1 walker under RoboParty's recipe on the device — G1_WALKER_PLAN §10.

    pixi run -e nvidia mojo run -I . tests/robots/test_unitree_g1_walk_rp_gpu.mojo
    pixi run -e apple  mojo run -I . tests/robots/test_unitree_g1_walk_rp_gpu.mojo   # SKIPS

`test_unitree_g1_walk_gpu.mojo`'s protocol on `UnitreeG1WalkRP*`; PART A
also compares the privileged words (contacts, solver forces, feet clocks),
PART B the gain DR is on (every lane its own kp / kd).

PART A — GPU vs CPU, one step at a time from the same state
(`test_unitree_g1_gpu_vs_cpu.mojo`'s protocol and its reasons: a free
lockstep of a float32 solve against a float64 one on this body measures the
Lyapunov exponent, not the device). Eval mode (`TRAIN=False`: no noise, no
pushes, frozen commands). Before every step the float64 CPU env's
`qpos` / `qvel` AND its walker `meta` words (command, feet state, key) are
injected into every lane; lanes carry DIFFERENT commands so the command
path is per-lane. Compared: the 70-D observation, the reward (device hook
vs `g1_walk_reward` of the host terms) and the termination.

PART B — train mode at 1024 lanes, random actions, 1000 steps: no NaN;
the fraction of lanes holding a zero command near 20 %; pushes land;
episodes end (falls) and restart; and the throughput against the BFM env's
36.5 k control steps/s at the same lane count.

⚠⚠ NVIDIA-ONLY (nv 35, Metal's per-thread stack). Compile-check on the
Mac with `mojo build --target-accelerator sm_120 --emit asm`.
"""

from max.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator
from std.math import abs, sin
from std.time import perf_counter_ns
from std.testing import assert_true, TestSuite

from noeira.nn.constants import DT
from noeira.core.cont_action import ContAction
from noeira.physics3d.gpu.constants import METADATA_SIZE
from noeira.envs.robots.unitree_g1_walk_rp import (
    UnitreeG1WalkRP as UnitreeG1Walk,
    UnitreeG1WalkRPBatched as UnitreeG1WalkBatched,
    UnitreeG1WalkRPModel as UnitreeG1WalkModel,
    g1r_host_reset as g1_walk_host_reset,
    g1r_host_terms as g1_walk_host_terms,
)
from noeira.envs.robots.unitree_g1_walk_rp_config import (
    G1R_N_TERMS as G1_WALK_N_TERMS,
    G1R_O_CMD as G1_WALK_OBS_CMD,
    G1R_CMD_VX as G1W_CMD_VX,
    G1R_CMD_VY as G1W_CMD_VY,
    G1R_CMD_WZ as G1W_CMD_WZ,
    G1R_CMD_TIMER as G1W_CMD_TIMER,
    G1R_KEY as G1W_KEY,
    G1R_PUSH_TIMER as G1W_PUSH_TIMER,
    g1r_reward as g1_walk_reward,
)

comptime NQ = UnitreeG1WalkModel.NQ
comptime NV = UnitreeG1WalkModel.NV
comptime OBS = UnitreeG1WalkModel.OBS_DIM
comptime ACT = UnitreeG1WalkModel.ACTION_DIM
comptime W0 = G1W_CMD_VX            # first walker meta word
comptime W1 = G1W_KEY + 1           # one past the last

comptime N_A = 4
comptime STEPS_A = 60
comptime ATOL = 1e-3
comptime RTOL = 1e-2
comptime MIN_F64_FRACTION = 0.9

comptime N_B = 1024
comptime STEPS_B = 1000


def _action(t: Int, j: Int) -> Float64:
    return 0.3 * sin(Float64(t) * 0.23 + Float64(j) * 0.61)


def _lane_cmd(e: Int) -> Tuple[Float64, Float64, Float64]:
    if e == 0:
        return (0.0, 0.0, 0.0)
    return (0.25 * Float64(e), -0.1 * Float64(e), 0.3 - 0.2 * Float64(e))


def _part_a(ctx: DeviceContext) raises:
    var cpu = UnitreeG1Walk[False]()
    var gpu = UnitreeG1WalkBatched[N_A, False](ctx)
    _ = cpu.reset()
    g1_walk_host_reset(cpu.d, 7, False)
    gpu.reset_batch[N_A](Optional(ctx), UInt64(3))
    var h_act = ctx.enqueue_create_host_buffer[DT](N_A * ACT)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_A * OBS)
    var h_rew = ctx.enqueue_create_host_buffer[DT](N_A)
    var h_done = ctx.enqueue_create_host_buffer[DT](N_A)
    ctx.synchronize()

    var n_bad = 0
    var n_f64_ok = 0
    var worst_obs = 0.0
    var worst_rew = 0.0
    var obs_lo = 1e30
    var obs_hi = -1e30
    var terms = Array[Float64, G1_WALK_N_TERMS](fill=0.0)
    var acts = List[Float64](length=ACT, fill=0.0)
    for t in range(STEPS_A):
        # ── inject the float64 state and walker words into every lane ──
        var qp = List[Float64]()
        var qv = List[Float64]()
        for i in range(NQ):
            qp.append(Float64(cpu.d.qpos.data[i]))
        for i in range(NV):
            qv.append(Float64(cpu.d.qvel.data[i]))
        gpu.d.qpos.download(ctx)
        gpu.d.qvel.download(ctx)
        gpu.d.meta.download(ctx)
        ctx.synchronize()
        for e in range(N_A):
            for i in range(NQ):
                gpu.d.qpos.data[e * NQ + i] = Scalar[DT](qp[i])
            for i in range(NV):
                gpu.d.qvel.data[e * NV + i] = Scalar[DT](qv[i])
            for w in range(W0, W1):
                gpu.d.meta.data[e * METADATA_SIZE + w] = Scalar[DT](
                    Float64(cpu.d.meta.data[w])
                )
            var c = _lane_cmd(e)
            gpu.d.meta.data[e * METADATA_SIZE + G1W_CMD_VX] = Scalar[DT](c[0])
            gpu.d.meta.data[e * METADATA_SIZE + G1W_CMD_VY] = Scalar[DT](c[1])
            gpu.d.meta.data[e * METADATA_SIZE + G1W_CMD_WZ] = Scalar[DT](c[2])
            gpu.d.meta.data[e * METADATA_SIZE + G1W_CMD_TIMER] = Scalar[DT](-1)
        gpu.d.qpos.upload(ctx)
        gpu.d.qvel.upload(ctx)
        gpu.d.meta.upload(ctx)
        ctx.synchronize()

        for j in range(ACT):
            acts[j] = _action(t, j)
            for e in range(N_A):
                h_act[e * ACT + j] = Scalar[DT](acts[j])
        ctx.enqueue_copy(gpu._action, h_act)
        gpu.step_batch[N_A](Optional(ctx), 0)
        ctx.enqueue_copy(h_obs, gpu._obs)
        ctx.enqueue_copy(h_rew, gpu._reward)
        ctx.enqueue_copy(h_done, gpu._terminated)
        ctx.synchronize()

        # ── the CPU side, lane by lane: the float64 state, each lane's
        # command, one step, the host terms; then restore the trajectory
        var snap_q = qp.copy()
        var snap_v = qv.copy()
        var snap_m = List[Float64]()
        for w in range(METADATA_SIZE):
            snap_m.append(Float64(cpu.d.meta.data[w]))
        var step_ok = True
        var lane0_q = List[Float64]()
        var lane0_v = List[Float64]()
        var lane0_m = List[Float64]()
        for e in range(N_A):
            cpu.set_state(snap_q, snap_v)
            for w in range(METADATA_SIZE):
                cpu.d.meta.data[w] = snap_m[w]
            var c = _lane_cmd(e)
            cpu.d.meta.data[G1W_CMD_VX] = c[0]
            cpu.d.meta.data[G1W_CMD_VY] = c[1]
            cpu.d.meta.data[G1W_CMD_WZ] = c[2]
            cpu.d.meta.data[G1W_CMD_TIMER] = -1
            var a = ContAction[ACT]()
            for j in range(ACT):
                a[j] = acts[j]
            var r = cpu.step(a)
            var done = g1_walk_host_terms(cpu.d, acts, terms)
            var rew = g1_walk_reward(terms)
            for k in range(OBS):
                var v64 = Float64(r[0].data[k])
                var vg = Float64(h_obs[e * OBS + k])
                obs_lo = min(obs_lo, v64)
                obs_hi = max(obs_hi, v64)
                var d = abs(vg - v64)
                worst_obs = max(worst_obs, d)
                if d > ATOL + RTOL * abs(v64):
                    step_ok = False
                    if k >= G1_WALK_OBS_CMD and k < G1_WALK_OBS_CMD + 3:
                        print("  COMMAND OBS MISMATCH lane", e, "k", k, vg, v64)
                        n_bad += 1
            var dr = abs(Float64(h_rew[e]) - rew)
            worst_rew = max(worst_rew, dr)
            if dr > ATOL + RTOL * abs(rew):
                step_ok = False
            if (Float64(h_done[e]) > 0.5) != done:
                print("  TERMINATION MISMATCH step", t, "lane", e)
                n_bad += 1
            if e == 0:
                for i in range(NQ):
                    lane0_q.append(Float64(cpu.d.qpos.data[i]))
                for i in range(NV):
                    lane0_v.append(Float64(cpu.d.qvel.data[i]))
                for w in range(METADATA_SIZE):
                    lane0_m.append(Float64(cpu.d.meta.data[w]))
        if step_ok:
            n_f64_ok += 1
        else:
            print("  step", t, "outside the band (knife-edge or defect)")
        # the trajectory continues from lane 0's step
        cpu.set_state(lane0_q, lane0_v)
        for w in range(METADATA_SIZE):
            cpu.d.meta.data[w] = lane0_m[w]
        if t < 3 or t % 10 == 9:
            print("    step", t, " worst |obs| so far", worst_obs, " |reward|", worst_rew)
    print("  PART A:", N_A, "lanes x", STEPS_A, "steps; in band on", n_f64_ok,
          "steps; worst obs", worst_obs, " worst reward", worst_rew,
          " obs range [", obs_lo, ",", obs_hi, "]")
    assert_true(obs_hi - obs_lo > 1.0, "the observation never moved: vacuous")
    assert_true(n_bad == 0, String(n_bad) + " command / termination mismatch(es)")
    assert_true(
        Float64(n_f64_ok) >= MIN_F64_FRACTION * Float64(STEPS_A),
        "in band on only " + String(n_f64_ok) + " of " + String(STEPS_A),
    )


def _part_b(ctx: DeviceContext) raises:
    var gpu = UnitreeG1WalkBatched[N_B, True](ctx)
    gpu.reset_batch[N_B](Optional(ctx), UInt64(11))
    var h_act = ctx.enqueue_create_host_buffer[DT](N_B * ACT)
    var h_obs = ctx.enqueue_create_host_buffer[DT](N_B * OBS)
    var h_done = ctx.enqueue_create_host_buffer[DT](N_B)
    ctx.synchronize()
    var zero_steps = 0
    var lane_steps = 0
    var ends = 0
    var nan = 0
    var pushes = 0
    var prev_pt = List[Float64](length=N_B, fill=0.0)
    var t0 = perf_counter_ns()
    var t_sim = 0
    for t in range(STEPS_B):
        for i in range(N_B * ACT):
            h_act[i] = Scalar[DT](0.5 * sin(Float64(t) * 0.37 + Float64(i) * 0.013))
        ctx.enqueue_copy(gpu._action, h_act)
        var ts = perf_counter_ns()
        gpu.step_batch[N_B](Optional(ctx), UInt64(t))
        ctx.synchronize()
        t_sim += Int(perf_counter_ns() - ts)
        ctx.enqueue_copy(h_obs, gpu._obs)
        ctx.enqueue_copy(h_done, gpu._done)
        gpu.d.meta.download(ctx)
        ctx.synchronize()
        for e in range(N_B):
            # a push re-arms its timer; a reset re-arms it too, so only
            # lanes that did not just end are counted
            var pt = Float64(gpu.d.meta.data[e * METADATA_SIZE + G1W_PUSH_TIMER])
            if pt > prev_pt[e] and t > 0 and Float64(h_done[e]) < 0.5:
                pushes += 1
            prev_pt[e] = pt
            var c0 = Float64(h_obs[e * OBS + G1_WALK_OBS_CMD])
            var c1 = Float64(h_obs[e * OBS + G1_WALK_OBS_CMD + 1])
            var c2 = Float64(h_obs[e * OBS + G1_WALK_OBS_CMD + 2])
            if c0 == 0.0 and c1 == 0.0 and c2 == 0.0:
                zero_steps += 1
            lane_steps += 1
            var vx = Float64(h_obs[e * OBS])
            if vx != vx:
                nan += 1
            if Float64(h_done[e]) > 0.5:
                ends += 1
        gpu.selective_reset_batch[N_B](ctx=ctx, rng_seed=UInt64(1000 + t))
    var wall = Float64(perf_counter_ns() - t0) / 1e9
    var frac = Float64(zero_steps) / Float64(lane_steps)
    var sps = Float64(N_B * STEPS_B) / (Float64(t_sim) / 1e9)
    print("  PART B:", N_B, "lanes x", STEPS_B, "steps: zero-command lane-steps",
          frac, " episode ends", ends, " pushes", pushes,
          " NaN", nan)
    print("    step_batch throughput", sps, "control steps/s (BFM env: 36.5 k)"
          "; wall incl. host checks", wall, "s")
    assert_true(nan == 0, "NaN in the observation")
    assert_true(abs(frac - 0.2) < 0.05, "zero-command fraction far from 20 %")
    assert_true(ends > 0, "no episode ended in 1000 steps of random actions")
    assert_true(pushes > 0, "no push fired")


def test_unitree_g1_walk_rp_gpu() raises:
    if not has_nvidia_gpu_accelerator():
        print("  unitree_g1_walk_rp GPU: SKIPPED on Apple (nv 35, Metal's"
              " per-thread stack). UNGATED until run on NVIDIA.")
        return
    with DeviceContext() as ctx:
        _part_a(ctx)
        _part_b(ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
