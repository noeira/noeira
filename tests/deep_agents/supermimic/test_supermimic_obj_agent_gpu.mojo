"""The object interface's device agent (`object_agent_gpu.SMObjAgentGpu`).

NVIDIA ONLY (the batched G1 does not build for Metal). A fresh, small tracker
(random weights), the G1 with its box, a library of the robot's settled pose
and a made object track: wiring, not results.

[1] the zero start: with the object encoder as made, the interface's action is
    the tracker's own on the same inputs, exactly.
[2] each lane's object reference row is the row of the lane's frame.
[3] 80 iterations with noise, randomisation and updates: every logged number
    is finite, the box's error is measured, the imagined box terms lie in
    [0, dt·w], updates happened.
[4] a start batch: the object's reference rows are the rows of frames
    min(frame + t, clip end); the object's start state is finite with a unit
    quaternion; its context is (1 kg, a friction in [0.5, 1.3]).
[5] after the updates: the object encoder moved (the interface's action on
    the saved inputs differs from the tracker's), and the TRACKER did not:
    its action on the saved inputs is bit-identical to before.
[6] gain 0 after the updates: the interface's action is the tracker's again.

Run (box): pixi run -e nvidia mojo run -I . tests/deep_agents/supermimic/test_supermimic_obj_agent_gpu.mojo
"""

from std.math import cos, isnan, sin, sqrt
from std.random import seed

from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.deep_agents.supermimic import SMConfig, SuperMimicTrainer
from noeira.deep_agents.supermimic.agent import sm_schedule
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_agent_gpu import SMObjAgentGpu
from noeira.deep_agents.supermimic.object_env_gpu import sm_object_track
from noeira.deep_agents.supermimic.object_policy import SMObjConfig
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX
from noeira.deep_agents.supermimic.object_task import SM_OREF, SM_OW_HANDS, SM_OW_ORI, SM_OW_POS, sm_object_ref_rows
from noeira.deep_agents.supermimic.object_trainer import SMObjTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabBoxBatchedSim, G1MjlabBoxRobot, SMDims
from noeira.envs.robots.unitree_g1_mjlab_box_xml import G1M_BOX_REST_Z
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime R = G1MjlabBoxRobot
comptime D = SMDims[R]
comptime n = R.NDOF
comptime NL = 16
comptime H = 32
comptime BS = G1MjlabBoxBatchedSim[NL]
comptime TC = SMConfig["gpu", R, 64, 32, 64, 48, 32, 24, 64, 48, 32, 24, 32, 16, NL, 32, 16, H]
comptime OC = SMObjConfig[TC, 48, 32, 24, 32, 16]
comptime Agent = SMObjAgentGpu[OC, BS]
comptime BP = TC.PI_BATCH
comptime FAR_X = 3.0          # `test_supermimic_obj_env_gpu`: zero action drops this robot forward, onto a box 0.6 m ahead


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
    var sim = R.SIM.create()
    var far: List[Float64] = [FAR_X, 0.0, G1M_BOX_REST_Z, 1.0, 0.0, 0.0, 0.0]
    sim.set_box(far, List[Float64](length=6, fill=0.0))
    var a = List[Float64](length=n, fill=0.0)
    for _ in range(100):
        sim.control_step(a, a)
    var row = List[Scalar[DT]](length=D.S, fill=0)
    for k in range(R.NQ):
        row[k] = Scalar[DT](sim.qpos(k))
    return row^


def _poses(lens: List[Int]) -> List[Float64]:
    """Clip c, frame f: the box in the air 0.5 m ahead of the lying robot,
    drifting and yawing slowly (it falls in the simulator: wiring only)."""
    var out = List[Float64]()
    for c in range(len(lens)):
        for f in range(lens[c]):
            var h = 0.01 * Float64(f)
            out.append(0.5 + 0.002 * Float64(f))
            out.append(0.1 * Float64(c))
            out.append(0.5)
            out.append(cos(h))
            out.append(0.0)
            out.append(0.0)
            out.append(sin(h))
    return out^


def _host(mut t: TensorImpl[DT], ctx: DeviceContext) raises -> TensorImpl[DT]:
    """A host copy of a device tensor."""
    t.download(ctx)
    var c = TensorImpl[DT].alloc(t.n)
    for i in range(t.n):
        c.data[i] = t.data[i]
    return c^


def _ints(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], count: Int) raises -> List[Int]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](count)
    ctx.enqueue_copy(h, buf)
    ctx.synchronize()
    var out = List[Int]()
    for i in range(count):
        out.append(Int(h[i]))
    return out^


def _max_diff(mut a: TensorImpl[DT], mut b: TensorImpl[DT]) -> Float64:
    var e = 0.0
    for i in range(a.n):
        e = max(e, abs(Float64(a.data[i]) - Float64(b.data[i])))
    return e


def main() raises:
    var tl = Tally()
    var ctx = DeviceContext()
    seed(5)
    var stand = _standing_row()
    var lens: List[Int] = [40, 60]
    var total = 100
    var rows = List[Scalar[DT]](length=total * D.S, fill=0)
    for f in range(total):
        for k in range(D.S):
            rows[f * D.S + k] = stand[k]
    var last = List[Int]()
    for f in range(total):
        last.append(39 if f < 40 else 99)
    var track = sm_object_track(_poses(lens), lens)
    var q0 = List[Float64](length=n, fill=0.0)
    for j in range(n):
        q0[j] = R.default_joint_pos(j)

    var base = SuperMimicTrainer[TC].for_robot(R.SIM.create(), Optional(ctx))
    var rows_ref = rows.copy()
    var track_ref = track.copy()
    var orefs = sm_object_ref_rows[R](base.tables, rows_ref, track_ref)
    var tr = SMObjTrainer[OC](base^)
    var sched = sm_schedule[TC](200, True, 5, 0.5, 20, 2e-4, 64)
    var agent = Agent(ctx, SMMotionLibrary[R](rows^, lens, q0), track, orefs.copy(), BS.create(ctx), tr^, sched^)

    # ── [1], [2] ──
    print("[1] the zero start: the interface's action is the tracker's")
    agent.env.env.policy_inputs(ctx, agent.goals)
    agent._lane_refs()
    var h0 = _host(agent.env.env.hist, ctx)
    var g0 = _host(agent.goals, ctx)
    var s0 = _host(agent.env.env.s, ctx)
    var o0 = _host(agent.env.prev_o, ctx)
    var r0 = _host(agent.oref_l, ctx)
    var a_trk = agent.trainer.base.act[NL](h0, g0)
    var a_obj = agent.trainer.act[NL](h0, g0, s0, o0, r0)
    var big = 0.0
    for i in range(a_trk.n):
        big = max(big, abs(Float64(a_trk.data[i])))
    tl.check(_max_diff(a_obj, a_trk) == 0.0 and big > 0.0, String(NL) + " lanes x " + String(n) + " actions identical (largest " + String(big) + ")")
    print("[2] each lane's object reference row is its frame's")
    var fr = _ints(ctx, agent.env.env.frame, NL)
    var e2 = 0.0
    for l in range(NL):
        for k in range(SM_OREF):
            e2 = max(e2, abs(Float64(r0.data[l * SM_OREF + k]) - Float64(orefs[fr[l] * SM_OREF + k])))
    tl.check(e2 == 0.0, "max |d| " + String(e2))

    # ── [3] ──
    print("[3] 80 iterations with noise, randomisation and updates")
    var bad = 0
    var upd = 0
    var logged = 0
    var terms_ok = True
    var perr = -1.0
    for it in range(80):
        var lg = it >= 40       # every iteration of the second half: a sparse log can miss every update (0.4 a step, summed in floats)
        var r = agent.iterate(lg)
        if isnan(r.sim_reward):
            bad += 1
        if r.updated:
            upd += 1
            if isnan(r.pi.imag_objective) or isnan(r.pi.policy_grad_norm) or isnan(r.pi.critic_loss) or isnan(r.om.loss):
                bad += 1
            if lg:
                logged += 1
                if not (r.pi.r_obj_pos >= 0.0 and r.pi.r_obj_pos <= 0.02 * SM_OW_POS + 1e-6 and r.pi.r_obj_ori >= 0.0
                        and r.pi.r_obj_ori <= 0.02 * SM_OW_ORI + 1e-6 and r.pi.r_obj_hands >= 0.0 and r.pi.r_obj_hands <= 0.02 * SM_OW_HANDS + 1e-6):
                    terms_ok = False
        if lg:
            if isnan(r.obj_pos_err) or r.obj_pos_err < 0.0 or r.obj_near < 0.0 or r.obj_near > 1.0 or isnan(r.obj_ang_err):
                bad += 1
            perr = r.obj_pos_err
    tl.check(bad == 0 and upd >= 20 and agent.updates == upd, String(upd) + " updates, no NaN in any logged number")
    tl.check(logged >= 3 and terms_ok and perr >= 0.0, "the box's error is measured (last " + String(perr) + " m); imagined box terms within [0, dt·w] on " + String(logged) + " logged updates")

    # ── [4] ──
    print("[4] a start batch")
    agent._start_batch()
    var sf = _ints(ctx, agent.s_frame, BP)
    agent.ib.orefs.download(ctx)
    agent.ib.obj0.download(ctx)
    agent.ib.octx.download(ctx)
    var e4 = 0.0
    for t in range(H + 1):
        for b in range(BP):
            var f = min(sf[b] + t, last[sf[b]])
            for k in range(SM_OREF):
                e4 = max(e4, abs(Float64(agent.ib.orefs.data[(t * BP + b) * SM_OREF + k]) - Float64(orefs[f * SM_OREF + k])))
    tl.check(e4 == 0.0, "reference rows of frames min(frame + t, clip end), " + String(H + 1) + " x " + String(BP) + ": max |d| " + String(e4))
    var qbad = 0.0
    var nan4 = 0
    var cbad = 0
    for b in range(BP):
        var qn = 0.0
        for k in range(SM_OBJ):
            if isnan(Float64(agent.ib.obj0.data[b * SM_OBJ + k])):
                nan4 += 1
        for k in range(4):
            qn += Float64(agent.ib.obj0.data[b * SM_OBJ + 3 + k]) * Float64(agent.ib.obj0.data[b * SM_OBJ + 3 + k])
        qbad = max(qbad, abs(sqrt(qn) - 1.0))
        var m = Float64(agent.ib.octx.data[b * SM_OBJ_CTX])
        var mu = Float64(agent.ib.octx.data[b * SM_OBJ_CTX + 1])
        if abs(m - 1.0) > 1e-6 or mu < 0.5 - 1e-6 or mu > 1.3 + 1e-6:
            cbad += 1
    tl.check(nan4 == 0 and qbad <= 1e-4 and cbad == 0, "start objects finite, unit quaternions (off by " + String(qbad) + "), context (1 kg, friction in [0.5, 1.3])")

    # ── [5], [6] ──
    print("[5] after the updates: the object encoder moved, the tracker did not")
    var a_trk2 = agent.trainer.base.act[NL](h0, g0)
    var a_obj2 = agent.trainer.act[NL](h0, g0, s0, o0, r0)
    tl.check(_max_diff(a_trk2, a_trk) == 0.0, "the tracker's action on the saved inputs is bit-identical")
    var moved = _max_diff(a_obj2, a_trk)
    tl.check(moved > 0.0, "the interface's action differs from the tracker's by up to " + String(moved))
    print("[6] gain 0: the tracker again")
    var keep = agent.trainer.gain
    agent.trainer.gain = 0.0
    var a_obj3 = agent.trainer.act[NL](h0, g0, s0, o0, r0)
    agent.trainer.gain = keep
    tl.check(_max_diff(a_obj3, a_trk) == 0.0, "max |d| " + String(_max_diff(a_obj3, a_trk)))

    print("\n=== " + String(tl.ok) + " / " + String(tl.total) + " passed ===")
    if tl.ok != tl.total:
        raise Error("test_supermimic_obj_agent_gpu: " + String(tl.total - tl.ok) + " failed")
