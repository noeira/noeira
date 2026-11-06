"""SuperMimic's prompt interface: its data and its lanes (`prompt_bank`, `env`).

    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_prompt_data.mojo

Answers known by construction, no reference implementation.

WHAT EACH SECTION CAN CATCH
===========================
[1] the window mean: the value on a ramp, a shrinking window at a clip's ends,
    nothing crossing into the next clip, no float32 drift over a long clip.
[2] a frame's atoms read the reference row: the heading-frame velocity by a
    yaw written in, the world yaw rate from a row whose ω is world-frame (the
    conversion to the state's body-frame ω and back), the height.
[3] z: a spec that selects one cluster of frames returns that cluster's latent
    minus the pool's mean, with its effective sample size; kernels MULTIPLY (a
    frame that meets one of two kernels does not count); a spec that selects
    nothing leaves z alone and reports it. Sharpening, by hand: a fixed beta and
    an effective sample size target retrieve the walking frames from a pool
    that mostly stands; a target the pool cannot reach leaves r alone, and a
    rare reward keeps its frames.
[4] the family: weights share Σ w = 4, a tilt prompt drops the lean price,
    walking with a hand placed is held out, and standing with one is not.
[5] a prompt lane on the CPU env: a tracking error that ends a tracker lane does
    not end it; a fall does; it never reaches the reset sampler's estimates.
[6] commands: a lane holds its command `DEF_HOLD` steps, draws again after, and
    both sources appear with the right share.
[7] resume: the counts go on from the checkpoint and the warm-up runs again.
"""

from std.math import cos, exp, sin
from std.random import random_float64, seed

from noeira.deep_agents.supermimic.agent import sm_schedule
from noeira.deep_agents.supermimic.config import SMConfig
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.prompt import SMSpec, sm_spec_velocity
from noeira.deep_agents.supermimic.prompt_agent import SMPromptAgent
from noeira.deep_agents.supermimic.prompt_bank import (
    sm_draw_spec, sm_frame_atoms, sm_prompt_data, sm_spec_held_out, sm_window_mean, sm_z_of_spec,
)
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfig
from noeira.deep_agents.supermimic.prompt_trainer import SMPromptTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, SMDims
from noeira.deep_agents.supermimic.tables import sm_tables
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime R = G1MjlabRobot
comptime D = SMDims[R]
comptime C = SMConfig["cpu", R, 32, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 8, 8, 8, 8]
comptime P = SMPromptConfig[C, 3, 32, 32, 32, 16]
comptime F = SMSpec.ATOMS


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _g(env: SMTrackingEnv[R]) -> Float64:
    """The sampler's estimates, summed: any fold moves it."""
    var s = 0.0
    for b in range(len(env.sampler.bin_return)):
        s += env.sampler.bin_return[b]
    return s


def _q0() -> List[Float64]:
    var q0 = List[Float64](length=D.NDOF, fill=0.0)
    for j in range(D.NDOF):
        q0[j] = R.default_joint_pos(j)
    return q0^


def _library(lens: List[Int], yaw: Float64, vx: Float64, wz: Float64, h: Float64) raises -> SMMotionLibrary[R]:
    """Clips of an upright robot at height `h`, yawed, moving at `vx` along its
    heading, turning at `wz` (world frame, as the rows hold it)."""
    var total = 0
    for k in range(len(lens)):
        total += lens[k]
    var rows = List[Scalar[DT]](length=total * D.S, fill=0)
    var q0 = _q0()
    for f in range(total):
        var o = f * D.S
        rows[o + 2] = Scalar[DT](h)
        rows[o + 3] = Scalar[DT](cos(yaw / 2))
        rows[o + 6] = Scalar[DT](sin(yaw / 2))
        for j in range(D.NDOF):
            rows[o + 7 + j] = Scalar[DT](q0[j])
        rows[o + D.NQ] = Scalar[DT](vx * cos(yaw))
        rows[o + D.NQ + 1] = Scalar[DT](vx * sin(yaw))
        rows[o + D.NQ + 5] = Scalar[DT](wz)
    return SMMotionLibrary[R](rows^, lens, q0)


def main() raises:
    var t = Tally()
    seed(3)

    print("\n[1] the window mean")
    var lens: List[Int] = [100, 40]
    var lib = _library(lens, 0.0, 0.0, 0.0, 0.78)
    var src = List[Scalar[DT]](length=140 * 2, fill=0)
    for f in range(140):
        src[f * 2] = Scalar[DT](f)
        src[f * 2 + 1] = 7
    var w = sm_window_mean[R](lib, src, 2, 0, 2)
    t.truth(w[50 * 2] == 50, "the middle of a ramp is its own mean")
    t.truth(w[0] == 6, "the first frame averages frames 0 .. 12 (the window shrinks)")
    t.truth(w[99 * 2] == 93, "the clip's last frame averages 87 .. 99: nothing from the next clip")
    t.truth(w[100 * 2] == 106, "the next clip's first frame averages 100 .. 112: nothing from the last")
    t.truth(w[50 * 2 + 1] == 7, "the second column")

    print("\n[2] a frame's atoms")
    var sim = R.SIM.create()
    var tb = sm_tables[DT, R](sim, _q0())
    var lib2 = _library(lens, 1.1, 0.8, -0.6, 0.72)
    var at = sm_frame_atoms[R](lib2, tb)
    var e = abs(Float64(at[SMSpec.A_VX]) - 0.8) + abs(Float64(at[SMSpec.A_VY]))
    print("  vx", at[SMSpec.A_VX], "vy", at[SMSpec.A_VY], "wz", at[SMSpec.A_WZ], "h", at[SMSpec.A_H], "gx", at[SMSpec.A_GX])
    t.truth(e < 1e-5, "the heading-frame velocity undoes the yaw")
    t.truth(abs(Float64(at[SMSpec.A_WZ]) + 0.6) < 1e-5, "the yaw rate comes back through the body frame")
    t.truth(abs(Float64(at[SMSpec.A_H]) - 0.72) < 1e-6, "the height")
    t.truth(abs(Float64(at[SMSpec.A_GX])) < 1e-6 and abs(Float64(at[SMSpec.A_GY])) < 1e-6, "upright: no gravity in the plane")
    # both hands stand at the same height and either side of the root on a symmetric pose
    t.truth(abs(Float64(at[SMSpec.A_LX + 2]) - Float64(at[SMSpec.A_RX + 2])) < 1e-4, "the hands' heights match on a symmetric pose")
    t.truth(Float64(at[SMSpec.A_LX + 1]) > 0.05 and Float64(at[SMSpec.A_RX + 1]) < -0.05, "left is +y in the heading frame, right −y")

    print("\n[3] z")
    comptime NP = 300
    comptime L = 4
    var pa = List[Scalar[DT]](length=NP * F, fill=0)
    var pl = List[Scalar[DT]](length=NP * L, fill=0)
    # a third of the frames at vx 1, h 0.6, latent (1, 0, 0, 0); a third at vx 1,
    # h 0.8, latent (0, 1, 0, 0); a third at vx 0, h 0.8, latent (0, 0, 1, 0)
    var lmean = List[Scalar[DT]](length=L, fill=0)
    for f in range(NP):
        var c = f % 3
        pa[f * F + SMSpec.A_VX] = 1 if c < 2 else 0
        pa[f * F + SMSpec.A_H] = Scalar[DT](0.6) if c == 0 else Scalar[DT](0.8)
        pl[f * L + c] = 1
        lmean[c] += Scalar[DT](1.0 / Float64(NP))
    var spec = List[Scalar[DT]](length=SMSpec.SIZE, fill=0)
    spec[SMSpec.T + SMSpec.A_VX] = 1
    spec[SMSpec.IW + SMSpec.A_VX] = 100
    spec[SMSpec.T + SMSpec.A_H] = 0.6
    spec[SMSpec.IW + SMSpec.A_H] = 400
    var z = List[Scalar[DT]](length=L, fill=0)
    var ess = sm_z_of_spec(
        spec.unsafe_ptr().as_unsafe_any_origin(), pa.unsafe_ptr().as_unsafe_any_origin(), pl.unsafe_ptr().as_unsafe_any_origin(),
        lmean.unsafe_ptr().as_unsafe_any_origin(), NP, L, z.unsafe_ptr().as_unsafe_any_origin(),
    )
    print("  ess", ess, "z", z[0], z[1], z[2], z[3])
    t.truth(abs(ess - 100.0) < 1e-3, "the cluster's size is the effective sample size")
    t.truth(abs(Float64(z[0]) - 2.0 / 3.0) < 1e-5 and abs(Float64(z[1]) + 1.0 / 3.0) < 1e-5, "z is the cluster's latent minus the pool's mean")
    spec[SMSpec.T + SMSpec.A_VX] = 0                  # no frame has vx 0 at h 0.6
    var z2 = List[Scalar[DT]](length=L, fill=9)
    var ess2 = sm_z_of_spec(
        spec.unsafe_ptr().as_unsafe_any_origin(), pa.unsafe_ptr().as_unsafe_any_origin(), pl.unsafe_ptr().as_unsafe_any_origin(),
        lmean.unsafe_ptr().as_unsafe_any_origin(), NP, L, z2.unsafe_ptr().as_unsafe_any_origin(),
    )
    t.truth(ess2 == 0.0 and z2[0] == 9, "kernels multiply: meeting one of two selects nothing, and z is left alone")

    # sharpening: 2000 frames stand (latent e0), 40 walk at vx 0.5 (latent e1), 2 at
    # vx 3 (latent e2). A wide kernel at vx 0.5 is swamped by the standing frames.
    comptime NS = 2042
    var sa = List[Scalar[DT]](length=NS * F, fill=0)
    var sl = List[Scalar[DT]](length=NS * L, fill=0)
    var sbar = List[Scalar[DT]](length=L, fill=0)
    for f in range(NS):
        var c = 0 if f < 2000 else (1 if f < 2040 else 2)
        sa[f * F + SMSpec.A_VX] = Scalar[DT](0.0 if c == 0 else (0.5 if c == 1 else 3.0))
        sl[f * L + c] = 1
    var sq = List[Scalar[DT]](length=SMSpec.SIZE, fill=0)
    sq[SMSpec.T + SMSpec.A_VX] = 0.5
    sq[SMSpec.IW + SMSpec.A_VX] = 4
    var sqp = sq.unsafe_ptr().as_unsafe_any_origin()
    var sap = sa.unsafe_ptr().as_unsafe_any_origin()
    var slp = sl.unsafe_ptr().as_unsafe_any_origin()
    var sbp = sbar.unsafe_ptr().as_unsafe_any_origin()
    var z0 = List[Scalar[DT]](length=L, fill=0)
    var z8 = List[Scalar[DT]](length=L, fill=0)
    var zt = List[Scalar[DT]](length=L, fill=0)
    var e0 = sm_z_of_spec(sqp, sap, slp, sbp, NS, L, z0.unsafe_ptr().as_unsafe_any_origin())
    var e8 = sm_z_of_spec(sqp, sap, slp, sbp, NS, L, z8.unsafe_ptr().as_unsafe_any_origin(), 8.0)
    var et = sm_z_of_spec(sqp, sap, slp, sbp, NS, L, zt.unsafe_ptr().as_unsafe_any_origin(), 1.0, 100.0)
    print("  ess", e0, e8, et, "walking share", z0[1], z8[1], zt[1])
    # by hand: 2000 frames at e^-1 and 40 at 1, then the same at r^8
    var w8 = exp(-8.0)
    t.truth(abs(Float64(z0[1]) - 40.0 / (2000.0 * exp(-1.0) + 40.0)) < 1e-5, "r: the walking frames' share, by hand")
    t.truth(abs(Float64(z8[1]) - 40.0 / (2000.0 * w8 + 40.0)) < 1e-5, "r^8: their share, by hand")
    t.truth(Float64(z0[1]) < 0.06 and Float64(z8[1]) > 0.9, "sharper retrieval leaves the standing frames behind")
    # ESS 100 of 40 frames at 1 and 2000 at w: (40 + 2000 w)² = 100 (40 + 2000 w²), w = 0.01173
    t.truth(et >= 100.0 and et < 106.0, "a target of 100 lands just above 100 (the bisection keeps the sharper side's ESS above it)")
    t.truth(abs(Float64(zt[1]) - 40.0 / (40.0 + 2000.0 * 0.01173)) < 0.03, "and on the share that size implies, by hand")
    var zn = List[Scalar[DT]](length=L, fill=0)
    var en = sm_z_of_spec(sqp, sap, slp, sbp, NS, L, zn.unsafe_ptr().as_unsafe_any_origin(), 1.0, 1e6)
    var same = abs(en - e0) < 1e-6
    for k in range(L):
        same = same and abs(Float64(zn[k]) - Float64(z0[k])) < 1e-6
    t.truth(same, "a target the pool cannot reach leaves r alone")
    sq[SMSpec.T + SMSpec.A_VX] = 3
    var zr = List[Scalar[DT]](length=L, fill=0)
    var er = sm_z_of_spec(sqp, sap, slp, sbp, NS, L, zr.unsafe_ptr().as_unsafe_any_origin(), 1.0, 100.0)
    t.truth(abs(er - 2.0) < 1e-6 and abs(Float64(zr[2]) - 1.0) < 1e-5, "a rare reward keeps its two frames: nothing is lost to the target")

    print("\n[4] the family")
    var sp = spec.unsafe_ptr().as_unsafe_any_origin()
    var sum_ok = True
    var lean_ok = True
    var seen_held = 0
    var seen_kept = 0
    for _ in range(2000):
        sm_draw_spec(sp, pa.unsafe_ptr().as_unsafe_any_origin(), NP)
        var sw = 0.0
        for g in range(SMSpec.GROUPS):
            sw += Float64(spec[SMSpec.W + g])
        sum_ok = sum_ok and abs(sw - 4.0) < 1e-5
        if spec[SMSpec.W + SMSpec.G_TILT] > 0:
            lean_ok = lean_ok and spec[SMSpec.LEAN] == 0
        else:
            lean_ok = lean_ok and spec[SMSpec.LEAN] == 4
        if sm_spec_held_out(sp):
            seen_held += 1
        else:
            seen_kept += 1
    print("  held out", seen_held, "kept", seen_kept)
    t.truth(sum_ok, "the group weights share Σ w = 4, Table A.6's")
    t.truth(lean_ok, "the lean price stands aside for a tilt prompt, only then")
    t.truth(seen_held > 0 and seen_kept > seen_held * 4, "a held-out region exists, and it is not most of the family")
    for k in range(SMSpec.SIZE):
        spec[k] = 0
    spec[SMSpec.W + SMSpec.G_VEL] = 1
    spec[SMSpec.W + SMSpec.G_LHAND] = 1
    spec[SMSpec.T + SMSpec.A_VX] = 1
    t.truth(sm_spec_held_out(sp), "walking with a hand placed is held out")
    spec[SMSpec.T + SMSpec.A_VX] = 0
    t.truth(not sm_spec_held_out(sp), "standing with a hand placed is not")

    print("\n[5] a prompt lane on the CPU env")
    # the reference floats 0.5 m above the start pose: a tracking failure at once
    var libr = _library(lens, 0.0, 0.0, 0.0, 0.78)
    for f in range(libr.num_frames):
        libr.rows[f * D.S + 2] = 1.3
    var env = SMTrackingEnv[R](2, libr^, dr=False, obs_noise=False, reset_noise=False, action_noise=0.0)
    env.prompt_from = 1
    var rb = SMReplayOf[R](2, 64)
    # start both lanes from a standing state, not from the floating reference
    for l in range(2):
        var qp = List[Float64](length=R.NQ, fill=0.0)
        var qv = List[Float64](length=R.NV, fill=0.0)
        qp[2] = 0.78
        qp[3] = 1.0
        for j in range(D.NDOF):
            qp[7 + j] = R.default_joint_pos(j)
        env.lanes[l].sim.set_state(qp, qv)
        env.lanes[l].read_state()
    var acts = List[Scalar[DT]](length=2 * D.NDOF, fill=0)
    var specs = List[Scalar[DT]](length=2 * SMSpec.SIZE, fill=0)
    sm_spec_velocity[DT](specs.unsafe_ptr().unsafe_offset(SMSpec.SIZE).as_unsafe_any_origin(), 0, 0, 0)
    var g0 = _g(env)
    _ = env.step(acts, rb, specs.unsafe_ptr().as_unsafe_any_origin())
    t.truth(env.last_terminated[0], "a tracking error ends a tracker lane")
    t.truth(not env.last_terminated[1], "it does not end a prompt lane")
    print("  task reward on the prompt lane", env.last_prompt_r[1])
    t.truth(env.last_prompt_r[1] > 0.05 and env.last_prompt_r[1] < 0.0801, "the prompt lane's reward: both kernels near 1, dt · Σ w = 0.08 at most")
    t.truth(abs(_g(env) - g0) > 0, "the tracker lane's episode reached the sampler (the next check has teeth)")
    # a lane on the floor ends, and only tracker lanes move the estimates
    var env2 = SMTrackingEnv[R](2, _library(lens, 0.0, 0.0, 0.0, 0.78), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0)
    env2.prompt_from = 0
    var rb2 = SMReplayOf[R](2, 64)
    var qp = List[Float64](length=R.NQ, fill=0.0)
    var qv = List[Float64](length=R.NV, fill=0.0)
    # lying on its back (a low straight-legged pose would be deep in the floor,
    # and contact pops it back above 0.35 m within the step)
    qp[2] = 0.4
    qp[3] = 0.7071068
    qp[5] = 0.7071068
    for l in range(2):
        env2.lanes[l].sim.set_state(qp, qv)
        env2.lanes[l].read_state()
    var g1 = _g(env2)
    _ = env2.step(acts, rb2, specs.unsafe_ptr().as_unsafe_any_origin())
    t.truth(env2.last_terminated[0] and env2.last_terminated[1], "a fall ends a prompt lane")
    t.truth(_g(env2) == g1, "prompt lanes never reach the reset sampler")

    print("\n[6] commands")
    var base = SuperMimicTrainer[C].for_robot(R.SIM.create())
    var lib6 = _library(lens, 0.0, 0.5, 0.0, 0.78)
    var data = sm_prompt_data[C, 3](base, lib6, 32)
    var agent = SMPromptAgent[P](lib6^, SMPromptTrainer[P](base^), data^, sm_schedule[C](100, False, 0), 0)
    var cmds = TensorImpl[DT].alloc(C.LANES * 3)
    agent._commands(cmds)
    var first = agent.src.copy()
    var held = True
    for _ in range(50):
        for l in range(C.LANES):
            agent.env.lanes[l].ep_len = 1
        agent._commands(cmds)
    for l in range(C.LANES):
        held = held and agent.src[l] == first[l]
    t.truth(held, "no new draw inside the hold")
    for _ in range(P.DEF_HOLD):
        for l in range(C.LANES):
            agent.env.lanes[l].ep_len = 1
        agent._commands(cmds)
    var changed = False
    for l in range(C.LANES):
        changed = changed or agent.src[l] != first[l]
    t.truth(changed, "a new draw after it")
    var nref = 0
    var ntot = 0
    for _ in range(40):
        for l in range(C.LANES):
            agent.env.lanes[l].ep_len = 0
        agent._commands(cmds)
        for l in range(C.LANES):
            nref += 1 if agent.src[l] < 0 else 0
            ntot += 1
    var share = Float64(nref) / Float64(ntot)
    print("  reference share", share)
    t.truth(abs(share - P.DEF_REF_SHARE) < 0.08, "both sources, in the configured share")
    # a reference command is the windowed velocity, normalised
    agent.src[0] = -1
    agent.hold[0] = 5
    agent.env.lanes[0].ep_len = 1
    agent.env.lanes[0].frame = 50
    agent._commands(cmds)
    t.truth(abs(Float64(cmds.data[0]) - 0.25) < 1e-5, "a reference command reads the frame's velocity, 0.5 m/s over the range's 2")
    t.truth(agent.specs[SMSpec.T + SMSpec.A_VX] == 0.5, "and its spec targets that velocity")

    print("\n[7] resume")
    agent.resume(1000, 400)
    agent.train(1002, 0)
    t.truth(agent.iteration == 1002, "a resumed run counts on to its total, not by it")
    t.truth(agent.updates == 400, "and waits for the warm-up again: its replay is empty")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_prompt_data: " + String(t.fails) + " failed")
