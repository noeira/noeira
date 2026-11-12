"""The device tracking env that carries an object (`object_env_gpu.
SMObjTrackingEnvGpu` over `G1MjlabBoxBatchedSim`), and the object's ring.

NVIDIA ONLY (the batched G1 does not build for Metal). The library is the
robot's own settled pose (`test_supermimic_env_gpu`'s construction); the
object's track is made here, so every answer is known.

[1] `sm_object_track` on a track of constant linear and angular velocity:
    the closed form, a clip's last frame taking the frame before it, unit
    quaternions.
[2] a reset: every lane restarted at its own frame, no noise: the object in
    the simulator is its track's row of that frame, the context is (the
    model's mass, the model's friction), and the ROBOT's state is the
    library's row in EVERY lane (the simulator's lanes are 43 wide, the
    robot's rows 36: a wrong stride breaks every lane after the first).
[3] a box at rest out of reach, zero action, no noise: 30 steps hold the
    robot's pose as the robot-only env does (reward within 2 % of its
    maximum, no termination) and the box does not move.
[4] the two rings in step, under noise, randomisation and episode ends (short
    clips, a ring that wraps): a host mirror of the ring's row rule, fed with
    the object's state downloaded before and after every step, equals the
    object's ring on every row it wrote, its pointers equal the ROBOT ring's,
    and the robot ring's state at each mirrored transition row is the robot's
    state downloaded with it. A lane that did not end keeps its object from
    one step to the next; one that ended has it at the track's row of its
    reset frame. Per-lane friction: in [0.5, 1.3], not all equal, and equal
    to the box body's override in `lane_dr`.
[5] `SMObjReplayGpu.windows_at` against a host gather from the mirror.

Run (box): pixi run -e nvidia mojo run -I . tests/deep_agents/supermimic/test_supermimic_obj_env_gpu.mojo
"""

from std.math import ceildiv, cos, isnan, sin, sqrt
from std.random import random_float64, seed
from std.sys import default_accelerator

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.env_gpu import ST_ENDING, SMReplayGpuOf
from noeira.deep_agents.supermimic.lanes import tp
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_env_gpu import (
    SM_OBJ_ROW, SMObjReplayGpu, SMObjTrackingEnvGpu, sm_object_track,
)
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX
from noeira.deep_agents.supermimic.robot import G1MjlabBoxBatchedSim, G1MjlabBoxRobot, SMDims
from noeira.envs.robots.unitree_g1_mjlab_box_xml import (
    G1M_BOX_BODY, G1M_BOX_QPOS, G1M_BOX_QVEL, G1M_BOX_REST_Z,
)
from noeira.nn.constants import DT
from noeira.physics3d.constraints.lane_dr import LANE_DR_FRICTION, LANE_DR_STRIDE
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl


comptime R = G1MjlabBoxRobot
comptime D = SMDims[R]
comptime n = R.NDOF
comptime NL = 8
comptime BS = G1MjlabBoxBatchedSim[NL]
comptime EnvO = SMObjTrackingEnvGpu[R, BS]
comptime SETTLE_STEPS = 100
comptime FPS = 50.0
# the made track: per frame, this much further along x and up, and this much more yaw
comptime DX = 0.01
comptime DZ = 0.004
comptime DYAW = 0.02
# [3]'s box. Zero action does not keep this robot standing: it goes down FORWARD within 2 s and comes to
# rest on the floor (root at 0.13 m), which is the pose the PD holds. The model's own rest pose for the box,
# 0.6 m ahead, is where the robot lands (measured on the CPU lane, 2026-11-12: the box pushed 7 cm, the
# robot still moving at 0.5 rad/s after 2 s), so the box of this gate rests 3 m ahead instead.
comptime FAR_X = 3.0


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
    """The robot settled for 2 s under zero action, through its CPU lane,
    that lane's box first moved `FAR_X` ahead (see `FAR_X`)."""
    var sim = R.SIM.create()
    var far: List[Float64] = [FAR_X, 0.0, G1M_BOX_REST_Z, 1.0, 0.0, 0.0, 0.0]
    sim.set_box(far, List[Float64](length=6, fill=0.0))
    var a = List[Float64](length=n, fill=0.0)
    for _ in range(SETTLE_STEPS):
        sim.control_step(a, a)
    var row = List[Scalar[DT]](length=D.S, fill=0)
    for k in range(R.NQ):
        row[k] = Scalar[DT](sim.qpos(k))
    return row^


def _library(stand: List[Scalar[DT]], lens: List[Int]) raises -> SMMotionLibrary[R]:
    var total = 0
    for c in range(len(lens)):
        total += lens[c]
    var rows = List[Scalar[DT]](length=total * D.S, fill=0)
    for f in range(total):
        for k in range(D.S):
            rows[f * D.S + k] = stand[k]
    var q0 = List[Float64](length=n, fill=0.0)
    for j in range(n):
        q0[j] = R.default_joint_pos(j)
    return SMMotionLibrary[R](rows^, lens, q0)


def _moving_poses(lens: List[Int]) -> List[Float64]:
    """Clip c, frame f: the box at (2 + DX·f, c, 1 + DZ·f), yawed DYAW·f —
    a constant velocity, well away from the robot."""
    var out = List[Float64]()
    for c in range(len(lens)):
        for f in range(lens[c]):
            var h = 0.5 * DYAW * Float64(f)
            out.append(2.0 + DX * Float64(f))
            out.append(Float64(c))
            out.append(1.0 + DZ * Float64(f))
            out.append(cos(h))
            out.append(0.0)
            out.append(0.0)
            out.append(sin(h))
    return out^


def _resting_poses(lens: List[Int]) -> List[Float64]:
    """Every frame: the box at rest on the floor, `FAR_X` ahead."""
    var out = List[Float64]()
    for c in range(len(lens)):
        for _ in range(lens[c]):
            out.append(FAR_X)
            out.append(0.0)
            out.append(G1M_BOX_REST_Z)
            out.append(1.0)
            out.append(0.0)
            out.append(0.0)
            out.append(0.0)
    return out^


def _k_copy(nn: Int64, src: Pointer[Scalar[DT], MutAnyOrigin], dst: Pointer[Scalar[DT], MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < Int(nn):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


def _dl(ctx: DeviceContext, mut t: TensorImpl[DT]) raises -> List[Scalar[DT]]:
    t.download(ctx)
    var l = List[Scalar[DT]](length=t.n, fill=0)
    for i in range(t.n):
        l[i] = t.data[i]
    return l^


def _dl_ptr(ctx: DeviceContext, src: Pointer[Scalar[DT], MutAnyOrigin], count: Int) raises -> List[Scalar[DT]]:
    """`count` floats behind a device pointer."""
    var t = TensorImpl[DT].alloc(count)
    t.upload_resident(ctx)
    ctx.enqueue_function[_k_copy](Int64(count), src, mptr(t.dev.value().unsafe_ptr()), grid_dim=ceildiv(count, 128), block_dim=128)
    return _dl(ctx, t)


def _actions(ctx: DeviceContext, vals: List[Scalar[DT]]) raises -> TensorImpl[DT]:
    var a = TensorImpl[DT].alloc(NL * n)
    for i in range(NL * n):
        a.data[i] = vals[i]
    a.upload_resident(ctx)
    return a^


def test_track(mut tl: Tally) raises:
    print("[1] the object's reference track from its poses")
    var lens: List[Int] = [5, 9]
    var tr = sm_object_track(_moving_poses(lens), lens, FPS)
    tl.check(len(tr) == 14 * SM_OBJ, "14 frames x 13")
    var ev = 0.0
    var ew = 0.0
    var eq = 0.0
    var ep = 0.0
    var f0 = 0
    for c in range(len(lens)):
        for f in range(lens[c]):
            var o = (f0 + f) * SM_OBJ
            ep = max(ep, abs(Float64(tr[o]) - (2.0 + DX * Float64(f))))
            ep = max(ep, abs(Float64(tr[o + 1]) - Float64(c)))
            ep = max(ep, abs(Float64(tr[o + 2]) - (1.0 + DZ * Float64(f))))
            var qn = 0.0
            for k in range(4):
                qn += Float64(tr[o + 3 + k]) * Float64(tr[o + 3 + k])
            eq = max(eq, abs(sqrt(qn) - 1.0))
            # every frame, the last one included: (DX, 0, DZ)·fps, and DYAW·fps about the box's own z
            ev = max(ev, abs(Float64(tr[o + 7]) - DX * FPS))
            ev = max(ev, abs(Float64(tr[o + 8])))
            ev = max(ev, abs(Float64(tr[o + 9]) - DZ * FPS))
            ew = max(ew, abs(Float64(tr[o + 10])))
            ew = max(ew, abs(Float64(tr[o + 11])))
            ew = max(ew, abs(Float64(tr[o + 12]) - DYAW * FPS))
        f0 += lens[c]
    tl.check(ep <= 1e-6 and eq <= 1e-6, "poses kept: position max |d| " + String(ep) + ", quaternion norm off by " + String(eq))
    tl.check(ev <= 1e-5, "linear velocity (0.5, 0, 0.2) m/s in the world on every frame: max |d| " + String(ev))
    tl.check(ew <= 1e-5, "angular velocity 1 rad/s about the box's own z on every frame: max |d| " + String(ew))


def test_reset(ctx: DeviceContext, stand: List[Scalar[DT]], mut tl: Tally) raises:
    print("[2] a reset: the object at its track's row, the robot at the library's, in every lane")
    seed(2)
    var lens: List[Int] = [40, 60]
    var track = sm_object_track(_moving_poses(lens), lens, FPS)
    var env = EnvO(
        ctx, _library(stand, lens), track, BS.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0,
        friction_dr=False,
    )
    var frames = List[Int]()
    for l in range(NL):
        frames.append((0 if l % 2 == 0 else 40) + 3 * l + 1)
    env.restart(ctx, frames)
    var po = _dl(ctx, env.prev_o)
    var eo = 0.0
    for l in range(NL):
        for k in range(SM_OBJ):
            eo = max(eo, abs(Float64(po[l * SM_OBJ + k]) - Float64(track[frames[l] * SM_OBJ + k])))
    tl.check(eo <= 1e-6, "the object's state read back from the simulator is the track's row of each lane's frame: max |d| " + String(eo))
    # the same, straight from the simulator's arrays at the model's own offsets
    var q = _dl_ptr(ctx, env.env.sim.qpos_ptr(), NL * BS.SIM_NQ)
    var v = _dl_ptr(ctx, env.env.sim.qvel_ptr(), NL * BS.SIM_NV)
    var es = 0.0
    var er = 0.0
    for l in range(NL):
        for k in range(7):
            es = max(es, abs(Float64(q[l * BS.SIM_NQ + G1M_BOX_QPOS + k]) - Float64(track[frames[l] * SM_OBJ + k])))
        for k in range(6):
            es = max(es, abs(Float64(v[l * BS.SIM_NV + G1M_BOX_QVEL + k]) - Float64(track[frames[l] * SM_OBJ + 7 + k])))
        for k in range(R.NQ):
            er = max(er, abs(Float64(q[l * BS.SIM_NQ + k]) - Float64(stand[k])))
        for k in range(R.NV):
            er = max(er, abs(Float64(v[l * BS.SIM_NV + k])))
    tl.check(es <= 1e-6, "qpos[" + String(G1M_BOX_QPOS) + "..] / qvel[" + String(G1M_BOX_QVEL) + "..] of every lane hold it: max |d| " + String(es))
    tl.check(er <= 1e-4, "the robot's qpos / qvel of every lane are the library row's: max |d| " + String(er))
    tl.check(BS.SIM_NQ == 43 and BS.SIM_NV == 41 and EnvO.OB == G1M_BOX_BODY, "the simulator's lanes are 43 / 41 wide, the object is body " + String(EnvO.OB))
    var c = _dl(ctx, env.octx)
    var ec = 0.0
    for l in range(NL):
        ec = max(ec, abs(Float64(c[l * SM_OBJ_CTX]) - 1.0))
        ec = max(ec, abs(Float64(c[l * SM_OBJ_CTX + 1]) - 0.9))
    tl.check(ec <= 1e-6 and abs(env.mass - 1.0) <= 1e-9, "the context is the model's (1 kg, friction 0.9): max |d| " + String(ec))


def test_hold(ctx: DeviceContext, stand: List[Scalar[DT]], mut tl: Tally) raises:
    print("[3] a box at rest out of reach: the robot holds its pose, the box stays")
    seed(3)
    var lens: List[Int] = [600]
    var track = sm_object_track(_resting_poses(lens), lens, FPS)
    var env = EnvO(
        ctx, _library(stand, lens), track, BS.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0,
        friction_dr=False,
    )
    var rb = SMReplayGpuOf[R](ctx, NL, 64)
    var orb = SMObjReplayGpu(ctx, NL, 64)
    var a = _actions(ctx, List[Scalar[DT]](length=NL * n, fill=0))
    var worst = 1.0
    var term = 0
    for _ in range(30):
        var si = env.step(ctx, a, rb, orb)
        term += si.n_terminated
        for l in range(NL):
            worst = min(worst, env.env.last_reward[l] / 0.1)
    tl.check(worst >= 0.98 and term == 0, "30 steps x 8 lanes: worst reward " + String(worst) + " of the maximum, " + String(term) + " terminations")
    var o = _dl(ctx, env.o)
    var moved = 0.0
    var speed = 0.0
    for l in range(NL):
        moved = max(moved, abs(Float64(o[l * SM_OBJ]) - FAR_X))
        moved = max(moved, abs(Float64(o[l * SM_OBJ + 1])))
        moved = max(moved, abs(Float64(o[l * SM_OBJ + 2]) - G1M_BOX_REST_Z))
        for k in range(6):
            speed = max(speed, abs(Float64(o[l * SM_OBJ + 7 + k])))
    tl.check(moved <= 1e-3 and speed <= 1e-2, "the box moved " + String(moved) + " m at most, its largest velocity entry " + String(speed))


def test_rings(ctx: DeviceContext, stand: List[Scalar[DT]], mut tl: Tally) raises:
    print("[4] the object's ring in step with the robot's, under noise, randomisation and episode ends")
    seed(4)
    comptime T = 32
    comptime STEPS = 48
    var lens: List[Int] = [12, 20]
    var track = sm_object_track(_moving_poses(lens), lens, FPS)
    var env = EnvO(ctx, _library(stand, lens), track, BS.create(ctx), rng_seed=4)
    var rb = SMReplayGpuOf[R](ctx, NL, T)
    var orb = SMObjReplayGpu(ctx, NL, T)
    # the host mirror: the ring's row rule, written out a second time
    var mirror = List[Scalar[DT]](length=NL * T * SM_OBJ_ROW, fill=0)
    var written = List[Bool](length=NL * T, fill=False)
    var is_transition = List[Bool](length=NL * T, fill=False)
    var robot_at = List[Scalar[DT]](length=NL * T * D.S, fill=0)
    var hptr = List[Int](length=NL, fill=0)
    var endings = 0
    var carry_bad = 0.0          # a lane that did not end: its object, one step to the next
    var reset_bad = 0.0          # a lane that ended: its object against the track at its reset frame
    var reset_n = 0
    var nan = 0
    var hf = ctx.enqueue_create_host_buffer[DType.int32](NL)
    for _ in range(STEPS):
        var acts = List[Scalar[DT]](length=NL * n, fill=0)
        for i in range(NL * n):
            acts[i] = Scalar[DT](random_float64(-0.3, 0.3))
        var a = _actions(ctx, acts)
        var o0 = _dl(ctx, env.prev_o)
        var c0 = _dl(ctx, env.prev_octx)
        var s0 = _dl(ctx, env.env.prev_s)
        var si = env.step(ctx, a, rb, orb)
        var o1 = _dl(ctx, env.o)
        var onext = _dl(ctx, env.prev_o)
        ctx.enqueue_copy(hf, env.env.reset_f)
        ctx.synchronize()
        var any_end = si.n_terminated + si.n_truncated > 0
        for l in range(NL):
            var ending = (env.env.h_out_i[2 * l + 1] & ST_ENDING) != 0
            var r = l * T + hptr[l]
            for k in range(SM_OBJ):
                mirror[r * SM_OBJ_ROW + k] = o0[l * SM_OBJ + k]
                nan += 1 if isnan(o1[l * SM_OBJ + k]) else 0
            for k in range(SM_OBJ_CTX):
                mirror[r * SM_OBJ_ROW + SM_OBJ + k] = c0[l * SM_OBJ_CTX + k]
            for k in range(D.S):
                robot_at[r * D.S + k] = s0[l * D.S + k]
            written[r] = True
            is_transition[r] = True
            hptr[l] = (hptr[l] + 1) % T
            if ending:
                endings += 1
                var r1 = l * T + hptr[l]
                for k in range(SM_OBJ):
                    mirror[r1 * SM_OBJ_ROW + k] = o1[l * SM_OBJ + k]
                for k in range(SM_OBJ_CTX):
                    mirror[r1 * SM_OBJ_ROW + SM_OBJ + k] = c0[l * SM_OBJ_CTX + k]
                written[r1] = True
                is_transition[r1] = False
                hptr[l] = (hptr[l] + 1) % T
                # the lane restarted: the object is at the track's row of its reset frame
                var f = Int(hf[l]) if any_end else -1
                if f < 0:
                    reset_bad = 1e9
                else:
                    reset_n += 1
                    for k in range(SM_OBJ):
                        reset_bad = max(reset_bad, abs(Float64(onext[l * SM_OBJ + k]) - Float64(track[f * SM_OBJ + k])))
            else:
                for k in range(SM_OBJ):
                    carry_bad = max(carry_bad, abs(Float64(onext[l * SM_OBJ + k]) - Float64(o1[l * SM_OBJ + k])))
                if any_end and Int(hf[l]) >= 0:
                    reset_bad = 1e9          # a reset frame for a lane that did not end
    tl.check(endings >= NL and nan == 0, String(STEPS) + " steps: " + String(endings) + " episodes ended, " + String(nan) + " NaN in the object's states")
    tl.check(carry_bad <= 1e-6, "a lane that did not end keeps its object from one step to the next: max |d| " + String(carry_bad))
    tl.check(reset_n == endings and reset_bad <= 1e-6, String(reset_n) + " restarted lanes have the object at the track's row of their reset frame: max |d| " + String(reset_bad))
    # the pointers: the robot ring's own
    var hp = ctx.enqueue_create_host_buffer[DType.int32](NL)
    ctx.enqueue_copy(hp, rb.ptr)
    ctx.synchronize()
    var ptr_ok = True
    var wrapped = 0
    for l in range(NL):
        ptr_ok = ptr_ok and Int(hp[l]) == hptr[l]
    for l in range(NL):
        var cnt = 0
        for t in range(T):
            cnt += 1 if written[l * T + t] else 0
        wrapped += 1 if cnt == T else 0
    tl.check(ptr_ok, "the mirror's pointers are the robot ring's in all " + String(NL) + " lanes")
    tl.check(wrapped == NL, "every lane's ring wrapped (" + String(STEPS) + " steps into " + String(T) + " rows)")
    var ring = _dl_ptr(ctx, mptr(orb.ring.unsafe_ptr()), NL * T * SM_OBJ_ROW)
    var e_ring = 0.0
    for r in range(NL * T):
        if written[r]:
            for k in range(SM_OBJ_ROW):
                e_ring = max(e_ring, abs(Float64(ring[r * SM_OBJ_ROW + k]) - Float64(mirror[r * SM_OBJ_ROW + k])))
    tl.check(e_ring == 0.0, "the object's ring equals the mirror on every row: max |d| " + String(e_ring))
    var rs = _dl_ptr(ctx, mptr(rb.s.unsafe_ptr()), NL * T * D.S)
    var e_rob = 0.0
    var n_tr = 0
    for r in range(NL * T):
        if written[r] and is_transition[r]:
            n_tr += 1
            for k in range(D.S):
                e_rob = max(e_rob, abs(Float64(rs[r * D.S + k]) - Float64(robot_at[r * D.S + k])))
    tl.check(e_rob == 0.0 and n_tr > 0, "the robot ring's state on the " + String(n_tr) + " mirrored transition rows is the robot's of the same step: max |d| " + String(e_rob))
    # per-lane friction
    var c = _dl(ctx, env.octx)
    var dr = _dl_ptr(ctx, env.env.sim.lane_dr_ptr(), NL * BS.SIM_NBODY * LANE_DR_STRIDE)
    var mu_ok = True
    var distinct = False
    for l in range(NL):
        var mu = Float64(c[l * SM_OBJ_CTX + 1])
        mu_ok = mu_ok and mu >= 0.5 and mu <= 1.3 and abs(Float64(c[l * SM_OBJ_CTX]) - 1.0) <= 1e-6
        mu_ok = mu_ok and dr[(l * BS.SIM_NBODY + G1M_BOX_BODY) * LANE_DR_STRIDE + LANE_DR_FRICTION] == c[l * SM_OBJ_CTX + 1]
        distinct = distinct or c[l * SM_OBJ_CTX + 1] != c[1]
    tl.check(mu_ok and distinct, "each lane's box friction is in [0.5, 1.3], is its box body's override, and the lanes differ")

    print("[5] windows from the object's ring against a host gather")
    comptime K = 6
    comptime NW = 16
    var hidx = ctx.enqueue_create_host_buffer[DType.int32](NW)
    var idx = ctx.enqueue_create_buffer[DType.int32](NW)
    var rows0 = List[Int]()
    for w in range(NW):
        rows0.append((w % NL) * T + (5 * w + 3) % T)
        hidx[w] = Int32(rows0[w])
    ctx.enqueue_copy(idx, hidx)
    var ws = TensorImpl[DT].alloc((K + 1) * NW * SM_OBJ)
    ws.upload_resident(ctx)
    var wc = TensorImpl[DT].alloc(NW * SM_OBJ_CTX)
    wc.upload_resident(ctx)
    orb.windows_at(ctx, mptr(idx.unsafe_ptr()), NW, K, tp["gpu"](ws), tp["gpu"](wc))
    var gs = _dl(ctx, ws)
    var gc = _dl(ctx, wc)
    var e_w = 0.0
    for w in range(NW):
        var e = rows0[w] // T
        var t = rows0[w] % T
        for k in range(SM_OBJ_CTX):
            e_w = max(e_w, abs(Float64(gc[w * SM_OBJ_CTX + k]) - Float64(mirror[rows0[w] * SM_OBJ_ROW + SM_OBJ + k])))
        for j in range(K + 1):
            var i = e * T + (t + j) % T
            for k in range(SM_OBJ):
                e_w = max(e_w, abs(Float64(gs[(j * NW + w) * SM_OBJ + k]) - Float64(mirror[i * SM_OBJ_ROW + k])))
    tl.check(e_w == 0.0, String(NW) + " windows of " + String(K) + " steps, across the wrap: max |d| " + String(e_w))


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print("SKIP test_supermimic_obj_env_gpu: needs an NVIDIA accelerator (the batched G1)")
        return
    else:
        var ctx = DeviceContext()
        var tl = Tally()
        test_track(tl)
        var stand = _standing_row()
        test_reset(ctx, stand, tl)
        test_hold(ctx, stand, tl)
        test_rings(ctx, stand, tl)
        print("\n===", tl.ok, "/", tl.total, "passed ===")
        if tl.ok != tl.total:
            raise Error("test_supermimic_obj_env_gpu: " + String(tl.total - tl.ok) + " failed")
