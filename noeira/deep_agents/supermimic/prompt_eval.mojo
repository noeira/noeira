# +--------------------------------------------------------------------------+ #
# | SuperMimic's prompt interface: the gates, measured
# +--------------------------------------------------------------------------+ #
"""What `noeira-docs/SUPERMIMIC_BFM_HYBRID_PLAN.md` §0 gates the interface on:

SUITE (G2, G3, G4). FB-MEBE's 22 base commands on the same reward function
(`fb/g1_mebe.mojo`'s `g1_cmd_suite` and `g1_cmd_reward`, imported, not copied:
one definition, or the comparison with BFM-Zero's numbers drifts), from a stand
with every joint moved U(−0.1, 0.1), 250 steps, return in [0, 250], a sticky
fall below 0.5 m. The command is computed zero-shot from the task: no search.
A lane the env ended (Table A.6's fall) scores nothing more: the env restarts
it, and a stand it did not earn must not pay.
⚠ Those 22 kernels are tight and instantaneous, and BFM-Zero's scores are on
another body: the numbers compare, they do not rank to the decimal.

Held-out rewards score the same way on their own spec's product of kernels:
walking with a hand placed, which training never saw (`sm_spec_held_out`).

Style (G4): `projects/g1/scripts/g1_style.mojo`'s `arm_dev`, `arm_qd`, `rock`,
after the same 50 steps, read off the state.

STEP TEST (G0). The paper's Appendix G.2: from standing, a fixed command for 5 s,
4 trials per command, 672 commands on a 15 × 15 grid of three planes. The
relative error is |mean velocity error over the last 2 s| / |command| in the
heading frame; a command's value is the median over the trials that did not
fall. The paper reads 0.16 (RL encoder) and 0.49 (distillation).

HONESTY (G1). The same lanes, the same commands, from the same states: H steps
in the world model against H steps in the simulator.
"""

from std.builtin.sort import sort
from std.math import exp, sqrt
from std.random import random_float64

from max.gpu.host import DeviceContext

from noeira.deep_agents.fb.g1_mebe import G1Cmd, g1_cmd_reward, g1_cmd_suite
from noeira.deep_agents.supermimic.kinematics import FP, sm_fk_lane
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.prompt import SMSpec, SM_P_LEAN_DEFAULT, sm_prompt_atoms_lane, sm_spec_velocity
from noeira.deep_agents.supermimic.prompt_bank import (
    SM_B_IW_H, SM_B_IW_HAND, SM_B_IW_TILT, SM_B_TASK_WEIGHT, SM_B_VX_HI, SM_B_VX_LO, SM_B_VY, SM_B_WZ, SMPromptData,
)
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.nn.constants import DT


comptime SM_E_SUITE_STEPS: Int = 250
comptime SM_E_STEP_STEPS: Int = 250           # 5 s
comptime SM_E_STEP_TAIL: Int = 100            # the last 2 s
comptime SM_E_SKIP: Int = 50                  # g1_style's start-up transient
comptime SM_E_FALL_H: Float64 = 0.5
comptime SM_E_FAMILY_NAMES = ("velocity", "posture", "composite", "held-out")


@fieldwise_init
struct SMEvalTask(Copyable, Movable):
    var name: String
    var family: Int                # 0 velocity, 1 posture, 2 composite, 3 held-out
    var cmd: List[Scalar[DT]]      # the command the interface receives
    var spec: List[Scalar[DT]]     # the reward it was trained on
    var suite: List[Float64]       # `G1Cmd.params()`; empty: score the spec's product of kernels


def sm_spec_of_cmd(c: G1Cmd, spec: FP[DT]):
    """A suite task as a spec of the training family: its kernels, the family's
    widths. A velocity task asks for the velocity; a posture task for its height
    and tilt; a composite for both."""
    for k in range(SMSpec.SIZE):
        spec[unsafe_offset=k] = 0
    spec[unsafe_offset=SMSpec.LEAN] = Scalar[DT](SM_P_LEAN_DEFAULT)
    var ng = 0
    if c.use_vw:
        spec[unsafe_offset=SMSpec.T + SMSpec.A_VX] = Scalar[DT](c.vx)
        spec[unsafe_offset=SMSpec.T + SMSpec.A_VY] = Scalar[DT](c.vy)
        spec[unsafe_offset=SMSpec.T + SMSpec.A_WZ] = Scalar[DT](c.wz)
        spec[unsafe_offset=SMSpec.IW + SMSpec.A_VX] = 4
        spec[unsafe_offset=SMSpec.IW + SMSpec.A_VY] = 4
        spec[unsafe_offset=SMSpec.IW + SMSpec.A_WZ] = 2
        spec[unsafe_offset=SMSpec.W + SMSpec.G_VEL] = 1
        spec[unsafe_offset=SMSpec.W + SMSpec.G_YAW] = 1
        ng += 2
    if c.use_h:
        spec[unsafe_offset=SMSpec.T + SMSpec.A_H] = Scalar[DT](c.h)
        spec[unsafe_offset=SMSpec.IW + SMSpec.A_H] = Scalar[DT](SM_B_IW_H)
        spec[unsafe_offset=SMSpec.W + SMSpec.G_H] = 1
        ng += 1
    if not c.use_vw or c.gx != 0.0 or c.gy != 0.0:
        spec[unsafe_offset=SMSpec.T + SMSpec.A_GX] = Scalar[DT](c.gx)
        spec[unsafe_offset=SMSpec.T + SMSpec.A_GY] = Scalar[DT](c.gy)
        spec[unsafe_offset=SMSpec.IW + SMSpec.A_GX] = Scalar[DT](SM_B_IW_TILT)
        spec[unsafe_offset=SMSpec.IW + SMSpec.A_GY] = Scalar[DT](SM_B_IW_TILT)
        spec[unsafe_offset=SMSpec.W + SMSpec.G_TILT] = 1
        spec[unsafe_offset=SMSpec.LEAN] = 0
        ng += 1
    for g in range(SMSpec.GROUPS):
        if spec[unsafe_offset=SMSpec.W + g] > 0:
            spec[unsafe_offset=SMSpec.W + g] = Scalar[DT](SM_B_TASK_WEIGHT / Float64(max(ng, 1)))


def _spec_held_out(spec: FP[DT], vx: Float64, hand_z: Float64, left: Bool):
    """Walking at `vx` with a hand at height `hand_z` above the pelvis."""
    for k in range(SMSpec.SIZE):
        spec[unsafe_offset=k] = 0
    spec[unsafe_offset=SMSpec.LEAN] = Scalar[DT](SM_P_LEAN_DEFAULT)
    spec[unsafe_offset=SMSpec.T + SMSpec.A_VX] = Scalar[DT](vx)
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_VX] = 4
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_VY] = 4
    spec[unsafe_offset=SMSpec.IW + SMSpec.A_WZ] = 2
    var az = (SMSpec.A_LX if left else SMSpec.A_RX) + 2
    spec[unsafe_offset=SMSpec.T + az] = Scalar[DT](hand_z)
    spec[unsafe_offset=SMSpec.IW + az] = Scalar[DT](SM_B_IW_HAND)
    var w = Scalar[DT](SM_B_TASK_WEIGHT / 3.0)
    spec[unsafe_offset=SMSpec.W + SMSpec.G_VEL] = w
    spec[unsafe_offset=SMSpec.W + SMSpec.G_YAW] = w
    spec[unsafe_offset=SMSpec.W + (SMSpec.G_LHAND if left else SMSpec.G_RHAND)] = w


def _velocity_cmd[CMD: Int](vx: Float64, vy: Float64, wz: Float64) -> List[Scalar[DT]]:
    var c = List[Scalar[DT]](length=CMD, fill=0)
    c[0] = Scalar[DT](vx / SM_B_VX_HI)
    c[1] = Scalar[DT](vy / SM_B_VY)
    c[2] = Scalar[DT](wz / SM_B_WZ)
    return c^


def sm_eval_tasks[CMD: Int](mut data: SMPromptData[CMD]) -> List[SMEvalTask]:
    """The suite, then the held-out rewards. With a z (CMD > 3) every task is
    prompted from its reward; a velocity interface can only be asked velocities."""
    var out = List[SMEvalTask]()
    var suite = g1_cmd_suite()
    for i in range(len(suite)):
        ref c = suite[i]
        var spec = List[Scalar[DT]](length=SMSpec.SIZE, fill=0)
        sm_spec_of_cmd(c, spec.unsafe_ptr().as_unsafe_any_origin())
        var cmd = List[Scalar[DT]](length=CMD, fill=0)
        comptime if CMD == 3:
            if not c.use_vw:
                continue
            cmd = _velocity_cmd[CMD](c.vx, c.vy, c.wz)
        else:
            var ess = data.z_of(spec.unsafe_ptr().as_unsafe_any_origin(), cmd.unsafe_ptr().as_unsafe_any_origin())
            if ess < 1.0:
                print("  no pool frame for", c.name, ": z = 0")
        out.append(SMEvalTask(c.name.copy(), c.family, cmd^, spec^, c.params()))
    comptime if CMD != 3:
        var vs: List[Float64] = [0.5, 1.0]
        var zs: List[Float64] = [0.3, 0.45]
        for iv in range(len(vs)):
            for iz in range(len(zs)):
                for side in range(2):
                    var spec = List[Scalar[DT]](length=SMSpec.SIZE, fill=0)
                    _spec_held_out(spec.unsafe_ptr().as_unsafe_any_origin(), vs[iv], zs[iz], side == 0)
                    var cmd = List[Scalar[DT]](length=CMD, fill=0)
                    var ess = data.z_of(spec.unsafe_ptr().as_unsafe_any_origin(), cmd.unsafe_ptr().as_unsafe_any_origin())
                    var name = String("walk") + String(vs[iv]) + (String("_lhand") if side == 0 else String("_rhand")) + String(zs[iz])
                    if ess < 1.0:
                        print("  no pool frame for", name, ": skipped")
                        continue
                    out.append(SMEvalTask(name, 3, cmd^, spec^, List[Float64]()))
    return out^


# ── the stand the suite starts from ─────────────────────────────────────────
def sm_stand_library[R: TrackingRobot](clips: Int, frames: Int, amp: Float64, root_z: Float64) raises -> SMMotionLibrary[R]:
    """Static clips of the default pose with every joint moved U(−amp, amp),
    root at `root_z`: restarting a lane on clip k is the suite's start."""
    comptime D = SMDims[R]
    var rows = List[Scalar[DT]](length=clips * frames * D.S, fill=0)
    var lens = List[Int](length=clips, fill=frames)
    var q0 = List[Float64](length=D.NDOF, fill=0.0)
    for j in range(D.NDOF):
        q0[j] = R.default_joint_pos(j)
    for c in range(clips):
        var row = List[Scalar[DT]](length=D.S, fill=0)
        row[2] = Scalar[DT](root_z)
        row[3] = 1
        for j in range(D.NDOF):
            row[7 + j] = Scalar[DT](q0[j] + random_float64(-amp, amp))
        for f in range(frames):
            for k in range(D.S):
                rows[(c * frames + f) * D.S + k] = row[k]
    return SMMotionLibrary[R](rows^, lens, q0)


# ── scoring ─────────────────────────────────────────────────────────────────
struct SMEvalTally(Movable):
    """Per-lane accumulators over a suite rollout."""
    var ret: List[Float64]
    var ended: List[Bool]
    var fell: List[Bool]
    var dev: List[Float64]
    var qd: List[Float64]
    var rock: List[Float64]
    var n: List[Int]
    var atoms: List[Scalar[DT]]
    var fk: List[Scalar[DT]]

    def __init__(out self, lanes: Int, fk_block: Int):
        self.ret = List[Float64](length=lanes, fill=0.0)
        self.ended = List[Bool](length=lanes, fill=False)
        self.fell = List[Bool](length=lanes, fill=False)
        self.dev = List[Float64](length=lanes, fill=0.0)
        self.qd = List[Float64](length=lanes, fill=0.0)
        self.rock = List[Float64](length=lanes, fill=0.0)
        self.n = List[Int](length=lanes, fill=0)
        self.atoms = List[Scalar[DT]](length=SMSpec.ATOMS, fill=0)
        self.fk = List[Scalar[DT]](length=fk_block, fill=0)

    def __init__(out self, *, deinit move: Self):
        self.ret = move.ret^
        self.ended = move.ended^
        self.fell = move.fell^
        self.dev = move.dev^
        self.qd = move.qd^
        self.rock = move.rock^
        self.n = move.n^
        self.atoms = move.atoms^
        self.fk = move.fk^

    def step[R: TrackingRobot](mut self, mut tb: SMTables[DT], l: Int, t: Int, s: FP[DT], ended_now: Bool, task: SMEvalTask):
        """Lane l's state after step t; `ended_now`: the env ended the lane."""
        comptime D = SMDims[R]
        if self.ended[l]:
            return
        if ended_now:
            self.ended[l] = True
            self.fell[l] = True
            return
        var pos = self.fk.unsafe_ptr().as_unsafe_any_origin()
        var quat = pos.unsafe_offset(D.NB * 3)
        var lin = pos.unsafe_offset(D.NB * 7)
        var ang = pos.unsafe_offset(D.NB * 10)
        sm_fk_lane[DT, D.NB](tb.tree_ptr(), s, s.unsafe_offset(D.NQ), pos, quat, lin, ang)
        var ap = self.atoms.unsafe_ptr().as_unsafe_any_origin()
        sm_prompt_atoms_lane[DT, R](tb.head_ptr(), s, pos, ap)
        var vx = Float64(ap[unsafe_offset=SMSpec.A_VX])
        var vy = Float64(ap[unsafe_offset=SMSpec.A_VY])
        var wz = Float64(ap[unsafe_offset=SMSpec.A_WZ])
        var h = Float64(ap[unsafe_offset=SMSpec.A_H])
        var gx = Float64(ap[unsafe_offset=SMSpec.A_GX])
        var gy = Float64(ap[unsafe_offset=SMSpec.A_GY])
        var r: Float64
        if len(task.suite) > 0:
            ref p = task.suite
            r = g1_cmd_reward(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], vx, vy, wz, gx, gy, h)
        else:
            var e = 0.0
            for a in range(SMSpec.ATOMS):
                var iw = Float64(task.spec[SMSpec.IW + a])
                if iw > 0:
                    var d = Float64(ap[unsafe_offset=a]) - Float64(task.spec[SMSpec.T + a])
                    e += iw * d * d
            r = exp(-e)
        self.ret[l] += r
        if h < SM_E_FALL_H:
            self.fell[l] = True
        if t >= SM_E_SKIP:
            var d0 = 0.0
            var d1 = 0.0
            for j in range(15, D.NDOF):       # the arms (`g1_style`: legs 0..11, waist 12..14)
                var e = Float64(s[unsafe_offset=7 + j]) - R.default_joint_pos(j)
                d0 += e * e
                var v = Float64(s[unsafe_offset=D.NQ + 6 + j])
                d1 += v * v
            self.dev[l] += d0 / Float64(D.NDOF - 15)
            self.qd[l] += d1 / Float64(D.NDOF - 15)
            var w0 = Float64(s[unsafe_offset=D.NQ + 3])
            var w1 = Float64(s[unsafe_offset=D.NQ + 4])
            self.rock[l] += 0.5 * (w0 * w0 + w1 * w1)
            self.n[l] += 1

    def report(self, tasks: List[SMEvalTask], task_of: List[Int]):
        """Per task and per family, as FB-MEBE's `[eval]` lines read."""
        var nt = len(tasks)
        var ret = List[Float64](length=nt, fill=0.0)
        var fall = List[Float64](length=nt, fill=0.0)
        var cnt = List[Int](length=nt, fill=0)
        var dv = List[Float64](length=nt, fill=0.0)
        var qd = List[Float64](length=nt, fill=0.0)
        var rk = List[Float64](length=nt, fill=0.0)
        var ns = List[Int](length=nt, fill=0)
        for l in range(len(self.ret)):
            var k = task_of[l]
            ret[k] += self.ret[l]
            fall[k] += 1.0 if self.fell[l] else 0.0
            cnt[k] += 1
            dv[k] += self.dev[l]
            qd[k] += self.qd[l]
            rk[k] += self.rock[l]
            ns[k] += self.n[l]
        print("task | family | return | fall | arm_dev arm_qd rock")
        var fam_ret = List[Float64](length=4, fill=0.0)
        var fam_fall = List[Float64](length=4, fill=0.0)
        var fam_n = List[Int](length=4, fill=0)
        var sd = 0.0
        var sq = 0.0
        var sr = 0.0
        var sn = 0
        for k in range(nt):
            var m = ret[k] / Float64(max(cnt[k], 1))
            var f = fall[k] / Float64(max(cnt[k], 1))
            var d = Float64(max(ns[k], 1))
            print(tasks[k].name, "|", tasks[k].family, "|", m, "|", f, "|", sqrt(dv[k] / d), sqrt(qd[k] / d), sqrt(rk[k] / d))
            fam_ret[tasks[k].family] += m
            fam_fall[tasks[k].family] += f
            fam_n[tasks[k].family] += 1
            if tasks[k].family == 0:
                sd += dv[k]
                sq += qd[k]
                sr += rk[k]
                sn += ns[k]
        print("--- per family: mean return, fall share")
        for g in range(4):
            if fam_n[g] > 0:
                print(" ", g, ":", fam_ret[g] / Float64(fam_n[g]), "|", fam_fall[g] / Float64(fam_n[g]), "(", fam_n[g], "tasks )")
        var dn = Float64(max(sn, 1))
        print("--- style on the velocity tasks: arm_dev", sqrt(sd / dn), "arm_qd", sqrt(sq / dn), "rock", sqrt(sr / dn), "(teacher .37 .41 .15, garm2 .66 .69 .20)")


# ── the step test ───────────────────────────────────────────────────────────
@fieldwise_init
struct SMStepCmd(Copyable, Movable):
    var vx: Float64
    var vy: Float64
    var wz: Float64


def sm_step_commands() -> List[SMStepCmd]:
    """Appendix G.2's 672: a 15 × 15 grid on each of three planes of the
    uniform ranges, the third component zero, the standing command left out."""
    var out = List[SMStepCmd]()
    for i in range(15):
        for j in range(15):
            var u = Float64(i) / 14.0
            var v = Float64(j) / 14.0
            var vx = SM_B_VX_LO + u * (SM_B_VX_HI - SM_B_VX_LO)
            var vy_v = -SM_B_VY + v * 2.0 * SM_B_VY
            var vy_u = -SM_B_VY + u * 2.0 * SM_B_VY
            var wz = -SM_B_WZ + v * 2.0 * SM_B_WZ
            var plane: List[SMStepCmd] = [SMStepCmd(vx, vy_v, 0.0), SMStepCmd(vx, 0.0, wz), SMStepCmd(0.0, vy_u, wz)]
            for k in range(3):
                ref c = plane[k]
                if abs(c.vx) + abs(c.vy) + abs(c.wz) > 1e-6:
                    out.append(c.copy())
    return out^


def sm_median(mut v: List[Float64]) -> Float64:
    if len(v) == 0:
        return -1.0
    sort(v)
    return v[len(v) // 2]
