"""SuperMimic's device env (`env_gpu.SMTrackingEnvGpu` over `G1BatchedSim`).

NVIDIA ONLY (the batched G1 does not build for Metal). Same construction as
test_supermimic_env: a reference library from our G1's pose after 2 s of zero
action (a pose the PD holds), so answers are known.

[1] holding the pose: zero action, no noise, no DR — every lane's reward
    within 2 % of its maximum (0.1) for 30 steps, no termination (a lane
    reset near its clip's end may time out: that is not a failure).
[2] the CPU env, noise off, on a library whose frames are all the same
    pose (so the two envs' reset draws, which consume the host RNG
    differently, cannot matter): after the reset the GPU lanes' states and
    histories equal the CPU lanes' (float32 vs float64: 1e-4); after one step
    of the same random actions, rewards and states agree to the bands
    printed (one control step of float32 physics).
[3] termination: a reference 0.3 m above the robot ends every lane at the
    first step: each ring holds a transition row then a terminal row
    labelled 1, its episode counter advanced, and the lane restarted. (The
    reset puts the robot AT the raised reference, so the test drops every
    lane back to the floor pose first, as the CPU test does.)
[4] the clip's end: a 60-frame clip, zero action: every lane times out by
    step 62, flagged at its clip's end, never terminated.
[5] bookkeeping under noise and DR, 60 random steps: every ring holds one
    row per step plus one per ended episode, no NaN in any state, every
    frame inside the library; PER-LANE DR: each lane's ξ friction in
    [0.3, 1.2], not all lanes equal, and each lane's foot-body friction
    overrides (`lane_dr`) equal to its ξ.
[6] prompt lanes: a tracking error ends a tracker lane and not a prompt lane, a
    fall does end one, its task reward comes back, and it never reaches the
    reset sampler.

Run (box): pixi run -e nvidia mojo run -I . tests/deep_agents/supermimic/test_supermimic_env_gpu.mojo
"""

from std.math import ceildiv, isnan
from std.random import random_float64, seed
from std.sys import default_accelerator, get_defined_int

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.core.cont_action import ContAction
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.env_gpu import ST_CLIP_END, ST_ENDING, ST_TERMINATED, SMReplayGpuOf, SMTrackingEnvGpu
from noeira.deep_agents.supermimic.lanes import talloc, tp
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.prompt import SMSpec, sm_spec_velocity
from noeira.deep_agents.supermimic.robot import G1BatchedSim, G1MjlabBatchedSim, G1MjlabRobot, G1Robot, SMDims
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl


# `-D SM_MJLAB=1`: the same gate on mjlab's G1.
comptime TEST_MJLAB = get_defined_int["SM_MJLAB", 0]() != 0
comptime R = G1MjlabRobot if TEST_MJLAB else G1Robot
comptime D = SMDims[R]
comptime n = R.NDOF
comptime NL = 8
comptime BS = G1MjlabBatchedSim[NL] if TEST_MJLAB else G1BatchedSim[NL]
comptime EnvG = SMTrackingEnvGpu[R, BS]


struct Tally:
    var ok: Int
    var total: Int

    def __init__(out self):
        self.ok = 0
        self.total = 0

    def check(mut self, cond: Bool, msg: String):
        self.total += 1
        if cond:
            self.ok += 1
            print("  ok:", msg)
        else:
            print("  FAIL:", msg)


def _standing_row() raises -> List[Scalar[DT]]:
    """The robot under test settled for 2 s under zero action (no bias)
    from its `stand` start, through its own CPU tracking sim."""
    var sim = R.SIM.create()
    var a = List[Float64](length=G1_N_DOF, fill=0.0)
    for _ in range(100):
        sim.control_step(a, a)
    var row = List[Scalar[DT]](length=D.S, fill=0)
    for k in range(R.NQ):
        row[k] = Scalar[DT](sim.qpos(k))
    return row^


def _library(stand: List[Scalar[DT]], lens: List[Int], dz: Float64) raises -> SMMotionLibrary[R]:
    var total = 0
    for c in range(len(lens)):
        total += lens[c]
    var rows = List[Scalar[DT]](length=total * D.S, fill=0)
    for f in range(total):
        for k in range(D.S):
            rows[f * D.S + k] = stand[k]
        rows[f * D.S + 2] += Scalar[DT](dz)
    var q0 = List[Float64](length=n, fill=0.0)
    for j in range(n):
        q0[j] = R.default_joint_pos(j)
    return SMMotionLibrary[R](rows^, lens, q0)


def _k_bcast(n_lanes: Int64, w: Int64, src: Pointer[Scalar[DT], MutAnyOrigin], dst: Pointer[Scalar[DT], MutAnyOrigin]):
    """dst[l · w + k] = src[k] for every lane l (n_lanes = 1: a plain copy)."""
    var i = Int(global_idx.x)
    if i < Int(n_lanes) * Int(w):
        dst[unsafe_offset=i] = src[unsafe_offset=i % Int(w)]


def _dl(ctx: DeviceContext, mut t: TensorImpl[DT]) raises -> List[Scalar[DT]]:
    t.download(ctx)
    var l = List[Scalar[DT]](length=t.n, fill=0)
    for i in range(t.n):
        l[i] = t.data[i]
    return l^


def _actions(ctx: DeviceContext, vals: List[Scalar[DT]]) raises -> TensorImpl[DT]:
    var a = TensorImpl[DT].alloc(NL * n)
    for i in range(NL * n):
        a.data[i] = vals[i]
    a.upload_resident(ctx)
    return a^


def _zeros() -> List[Scalar[DT]]:
    return List[Scalar[DT]](length=NL * n, fill=0)


def test_hold(ctx: DeviceContext, stand: List[Scalar[DT]], mut tl: Tally) raises:
    print("[1] holding the pose (zero action, no noise, no DR)")
    seed(1)
    var env = EnvG(ctx, _library(stand, [600], 0.0), BS.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0)
    var rb = SMReplayGpuOf[R](ctx, NL, 64)
    var a = _actions(ctx, _zeros())
    var worst = 1.0
    var term = 0
    var trunc = 0
    for _ in range(30):
        var si = env.step(ctx, a, rb)
        term += si.n_terminated
        trunc += si.n_truncated
        for l in range(NL):
            worst = min(worst, env.last_reward[l] / 0.1)
    tl.check(worst >= 0.98 and term == 0, "30 steps x 8 lanes: worst reward " + String(worst) + " of the maximum, " + String(term) + " terminations (" + String(trunc) + " clip-end time-outs)")


def test_vs_cpu(ctx: DeviceContext, stand: List[Scalar[DT]], mut tl: Tally) raises:
    print("[2] the CPU env, same seed and draws, noise off")
    var lens: List[Int] = [300]
    seed(2)
    var cpu = SMTrackingEnv[R](NL, _library(stand, lens, 0.0), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, rng_seed=2)
    seed(2)
    var gpu = EnvG(ctx, _library(stand, lens, 0.0), BS.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, rng_seed=2)
    var gs = _dl(ctx, gpu.s)
    var gh = _dl(ctx, gpu.hist)
    var es = 0.0
    var eh = 0.0
    for l in range(NL):
        for k in range(D.S):
            es = max(es, abs(Float64(gs[l * D.S + k]) - Float64(cpu.lanes[l].s[k])))
        for k in range(D.HIST * D.OBS):
            eh = max(eh, abs(Float64(gh[l * D.HIST * D.OBS + k]) - Float64(cpu.lanes[l].hist[k])))
    tl.check(es <= 1e-4 and eh <= 1e-4, "after the reset: states max |d| " + String(es) + ", histories " + String(eh))
    var acts = List[Scalar[DT]](length=NL * n, fill=0)
    for i in range(NL * n):
        acts[i] = Scalar[DT](random_float64(-0.5, 0.5))
    var rbc = SMReplayOf[R](NL, 64)
    var rbg = SMReplayGpuOf[R](ctx, NL, 64)
    _ = cpu.step(acts, rbc)
    var a = _actions(ctx, acts)
    _ = gpu.step(ctx, a, rbg)
    gs = _dl(ctx, gpu.s)
    var er = 0.0
    var eq = 0.0
    var ev = 0.0
    for l in range(NL):
        er = max(er, abs(gpu.last_reward[l] - cpu.last_reward[l]))
        for k in range(R.NQ):
            eq = max(eq, abs(Float64(gs[l * D.S + k]) - Float64(cpu.lanes[l].s[k])))
        for k in range(R.NV):
            ev = max(ev, abs(Float64(gs[l * D.S + R.NQ + k]) - Float64(cpu.lanes[l].s[R.NQ + k])))
    tl.check(er <= 2e-3 and eq <= 5e-3 and ev <= 0.2, "one step: rewards max |d| " + String(er) + ", qpos " + String(eq) + ", qvel " + String(ev))


def test_termination(ctx: DeviceContext, stand: List[Scalar[DT]], mut tl: Tally) raises:
    print("[3] termination: the reference 0.3 m above")
    seed(3)
    var env = EnvG(ctx, _library(stand, [400], 0.3), BS.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0)
    # the reset put every lane AT the raised reference: back to the floor pose
    # (through the batched-sim interface's device pointers, so any robot's sim works)
    var hq = TensorImpl[DT].alloc(R.NQ)
    var hv = TensorImpl[DT].alloc(R.NV)
    for k in range(R.NQ):
        hq.data[k] = stand[k]
    for k in range(R.NV):
        hv.data[k] = 0
    hq.upload_resident(ctx)
    hv.upload_resident(ctx)
    ctx.enqueue_function[_k_bcast](
        Int64(NL), Int64(R.NQ), mptr(hq.dev.value().unsafe_ptr()), env.sim.qpos_ptr(),
        grid_dim=ceildiv(NL * R.NQ, 128), block_dim=128,
    )
    ctx.enqueue_function[_k_bcast](
        Int64(NL), Int64(R.NV), mptr(hv.dev.value().unsafe_ptr()), env.sim.qvel_ptr(),
        grid_dim=ceildiv(NL * R.NV, 128), block_dim=128,
    )
    env.sim.refresh(ctx)
    env._observe(ctx)          # the env's state (and next rows) read back from the simulator
    var rb = SMReplayGpuOf[R](ctx, NL, 64)
    var a = _actions(ctx, _zeros())
    var si = env.step(ctx, a, rb)
    var hp = ctx.enqueue_create_host_buffer[DType.int32](NL)
    var he = ctx.enqueue_create_host_buffer[DType.int32](NL)
    var hfl = ctx.enqueue_create_host_buffer[DType.int32](NL * 64)
    var hd = ctx.enqueue_create_host_buffer[DT](NL * 64)
    ctx.enqueue_copy(hp, rb.ptr)
    ctx.enqueue_copy(he, rb.ep_counter)
    ctx.enqueue_copy(hfl, rb.flags)
    ctx.enqueue_copy(hd, rb.done)
    ctx.synchronize()
    var ok = si.n_terminated == NL
    for l in range(NL):
        ok = ok and Int(hp[l]) == 2 and Int(he[l]) == 1
        ok = ok and (hfl[l * 64] & 2) == 0 and (hfl[l * 64 + 1] & 2) != 0       # transition, then terminal
        ok = ok and hd[l * 64] == 1 and hd[l * 64 + 1] == 1
    tl.check(ok, String(si.n_terminated) + " of 8 lanes terminated; each ring: a transition and a terminal row labelled 1, episode counter 1")


def test_clip_end(ctx: DeviceContext, stand: List[Scalar[DT]], mut tl: Tally) raises:
    print("[4] the clip's end")
    seed(4)
    var env = EnvG(ctx, _library(stand, [60], 0.0), BS.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0)
    var rb = SMReplayGpuOf[R](ctx, NL, 128)
    var a = _actions(ctx, _zeros())
    var ended = List[Bool](length=NL, fill=False)
    var term = 0
    var hi = ctx.enqueue_create_host_buffer[DType.int32](NL * 2)
    for _ in range(62):
        var si = env.step(ctx, a, rb)
        term += si.n_terminated
        ctx.enqueue_copy(hi, env.out_i)
        ctx.synchronize()
        for l in range(NL):
            var st = hi[2 * l + 1]
            if (st & ST_ENDING) != 0 and (st & ST_CLIP_END) != 0:
                ended[l] = True
    var all = True
    for l in range(NL):
        all = all and ended[l]
    tl.check(all and term == 0, "every lane timed out at its clip's end within 62 steps, " + String(term) + " terminations")


def test_bookkeeping(ctx: DeviceContext, stand: List[Scalar[DT]], mut tl: Tally) raises:
    print("[5] bookkeeping under noise and DR")
    seed(5)
    var env = EnvG(ctx, _library(stand, [120, 90, 200], 0.0), BS.create(ctx), friction_period=25)
    var rb = SMReplayGpuOf[R](ctx, NL, 256)
    var ends = 0
    for _ in range(60):
        var acts = List[Scalar[DT]](length=NL * n, fill=0)
        for i in range(NL * n):
            acts[i] = Scalar[DT](random_float64(-1, 1))
        var a = _actions(ctx, acts)
        var si = env.step(ctx, a, rb)
        ends += si.n_terminated + si.n_truncated
    var hp = ctx.enqueue_create_host_buffer[DType.int32](NL)
    var hf = ctx.enqueue_create_host_buffer[DType.int32](NL)
    ctx.enqueue_copy(hp, rb.ptr)
    ctx.enqueue_copy(hf, env.frame)
    ctx.synchronize()
    var rows = 0
    var frames_ok = True
    for l in range(NL):
        rows += Int(hp[l])
        frames_ok = frames_ok and Int(hf[l]) >= 0 and Int(hf[l]) < env.lib.num_frames
    var s = _dl(ctx, env.s)
    var nan = 0
    for i in range(len(s)):
        nan += 1 if isnan(s[i]) else 0
    var x = _dl(ctx, env.xi)
    comptime NDR = NL * (R.NBODY + 1) * 4
    var drt = TensorImpl[DT].alloc(NDR)
    drt.upload_resident(ctx)
    ctx.enqueue_function[_k_bcast](
        Int64(1), Int64(NDR), env.sim.lane_dr_ptr(), mptr(drt.dev.value().unsafe_ptr()),
        grid_dim=ceildiv(NDR, 128), block_dim=128,
    )
    var dr = _dl(ctx, drt)
    var hfeet = ctx.enqueue_create_host_buffer[DType.int32](max(env.n_feet, 1))
    ctx.enqueue_copy(hfeet, env.feet)
    ctx.synchronize()
    var mu_ok = env.n_feet > 0
    var distinct = False
    for l in range(NL):
        var mu = Float64(x[l * D.X])
        mu_ok = mu_ok and mu >= 0.3 and mu <= 1.2
        distinct = distinct or x[l * D.X] != x[0]
        for k in range(env.n_feet):
            mu_ok = mu_ok and dr[(l * (R.NBODY + 1) + Int(hfeet[k])) * 4] == x[l * D.X]
    tl.check(rows == 60 * NL + ends, String(rows) + " rows = 60 x 8 + " + String(ends) + " terminal rows")
    tl.check(nan == 0 and frames_ok and mu_ok and distinct, "no NaN, frames inside the library, per-lane friction on " + String(env.n_feet) + " foot bodies (lane 0 " + String(x[0]) + ", lane 1 " + String(x[D.X]) + ")")


def _put_all(ctx: DeviceContext, mut env: EnvG, qp: List[Scalar[DT]]) raises:
    """Every lane at `qp`, at rest (the reference stays where it is)."""
    var hq = TensorImpl[DT].alloc(R.NQ)
    var hv = TensorImpl[DT].alloc(R.NV)
    for k in range(R.NQ):
        hq.data[k] = qp[k]
    hq.upload_resident(ctx)
    hv.upload_resident(ctx)
    ctx.enqueue_function[_k_bcast](
        Int64(NL), Int64(R.NQ), mptr(hq.dev.value().unsafe_ptr()), env.sim.qpos_ptr(), grid_dim=ceildiv(NL * R.NQ, 128), block_dim=128,
    )
    ctx.enqueue_function[_k_bcast](
        Int64(NL), Int64(R.NV), mptr(hv.dev.value().unsafe_ptr()), env.sim.qvel_ptr(), grid_dim=ceildiv(NL * R.NV, 128), block_dim=128,
    )
    env.sim.refresh(ctx)
    env._observe(ctx)


def test_prompt_lanes(ctx: DeviceContext, stand: List[Scalar[DT]], mut tl: Tally) raises:
    print("[6] prompt lanes (`env.mojo`'s rules, on the device)")
    # ⚠ not `stand`: that settled row is LYING (see `_standing_row`), a fall to
    # this rule. The fresh keyframe is upright.
    var sim0 = R.SIM.create()
    var up = List[Scalar[DT]](length=D.S, fill=0)
    for k in range(R.NQ):
        up[k] = Scalar[DT](sim0.qpos(k))
    var lens: List[Int] = [400]
    var env = EnvG(ctx, _library(up, lens, 0.3), BS.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0)
    env.prompt_from = 4
    _put_all(ctx, env, up)
    var specs = talloc["gpu"](NL * SMSpec.SIZE, Optional(ctx))
    for l in range(NL):
        sm_spec_velocity[DT](specs.data.unsafe_ptr().unsafe_offset(l * SMSpec.SIZE).as_unsafe_any_origin(), 0, 0, 0)
    specs.upload_resident(ctx)
    var rb = SMReplayGpuOf[R](ctx, NL, 64)
    var a = _actions(ctx, _zeros())
    var g0 = 0.0
    for b in range(len(env.sampler.bin_return)):
        g0 += env.sampler.bin_return[b]
    _ = env.step(ctx, a, rb, tp["gpu"](specs))
    var tracker_end = True
    var prompt_run = True
    for l in range(NL):
        var ended = (env.h_out_i[2 * l + 1] & ST_TERMINATED) != 0
        if l < 4:
            tracker_end = tracker_end and ended
        else:
            prompt_run = prompt_run and not ended
    tl.check(tracker_end, "a tracking error ends the tracker lanes")
    tl.check(prompt_run, "it does not end the prompt lanes")
    env.prompt_r.download(ctx)
    var r_ok = True
    for l in range(4, NL):
        r_ok = r_ok and Float64(env.prompt_r.data[l]) > 0.05 and Float64(env.prompt_r.data[l]) < 0.0801
    tl.check(r_ok, "their task reward: about dt · Σ w = 0.08, got " + String(env.prompt_r.data[4]))
    var g1 = 0.0
    for b in range(len(env.sampler.bin_return)):
        g1 += env.sampler.bin_return[b]
    tl.check(g1 != g0, "the tracker lanes reached the sampler (the next check has teeth)")
    # every lane a prompt lane, lying on its back: Table A.6's fall, and no estimate moves
    var env2 = EnvG(ctx, _library(up, lens, 0.0), BS.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0)
    env2.prompt_from = 0
    var lying = up.copy()
    lying[2] = 0.4
    lying[3] = 0.7071068
    lying[4] = 0
    lying[5] = 0.7071068
    lying[6] = 0
    _put_all(ctx, env2, lying)
    var rb2 = SMReplayGpuOf[R](ctx, NL, 64)
    var a2 = _actions(ctx, _zeros())
    var g2 = 0.0
    for b in range(len(env2.sampler.bin_return)):
        g2 += env2.sampler.bin_return[b]
    var si = env2.step(ctx, a2, rb2, tp["gpu"](specs))
    var g3 = 0.0
    for b in range(len(env2.sampler.bin_return)):
        g3 += env2.sampler.bin_return[b]
    tl.check(si.n_terminated == NL, "a fall ends a prompt lane: " + String(si.n_terminated) + " of 8")
    tl.check(g2 == g3, "prompt lanes never reach the reset sampler")


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        # the batched G1 is NVIDIA only; never instantiated here
        print("SKIP test_supermimic_env_gpu: needs an NVIDIA accelerator (the batched G1)")
        return
    else:
        var ctx = DeviceContext()
        var tl = Tally()
        var stand = _standing_row()
        test_hold(ctx, stand, tl)
        test_vs_cpu(ctx, stand, tl)
        test_termination(ctx, stand, tl)
        test_clip_end(ctx, stand, tl)
        test_bookkeeping(ctx, stand, tl)
        test_prompt_lanes(ctx, stand, tl)
        print("\n===", tl.ok, "/", tl.total, "passed ===")
        if tl.ok != tl.total:
            raise Error("test_supermimic_env_gpu: " + String(tl.total - tl.ok) + " failed")
