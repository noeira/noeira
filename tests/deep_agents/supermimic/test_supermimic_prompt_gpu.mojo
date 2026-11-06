"""SuperMimic's prompt interface: its device path against its CPU path.

    pixi run -e apple mojo run -I . tests/deep_agents/supermimic/test_supermimic_prompt_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/deep_agents/supermimic/test_supermimic_prompt_gpu.mojo

Same weights, same batch, small networks: the CPU path carries the oracle and
finite-difference gates (`test_supermimic_prompt`, `test_supermimic_prompt_imag`),
this one that the kernels compute the same thing. Bands as `test_supermimic_gpu`:
vectors by cosine ≥ 0.99999 and max error ≤ 1e-4 of the largest; weights after a
step by the elements that moved.

[1] the rollout: states, actions, rewards, survival, J, the alignment's MSE.
[2] −J/H's gradient over the encoder.
[3] one `policy_update`: the encoder and its critic, and everything else unmoved.
[4] a mixed `act`.
[5] the two agent kernels, the only device code the CPU loop does not share:
    a lane's command and a start's inputs, on small fake tables.

⚠ A band written on Metal is a TF32 trap on CUDA: compare bit for bit first if a
5090 lands outside it.
"""

from std.math import sqrt
from std.random import random_float64, seed

from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.deep_agents.supermimic.config import SMConfig
from noeira.deep_agents.supermimic.lanes import talloc, tp
from noeira.deep_agents.supermimic.params import sm_restore, sm_snapshot
from noeira.deep_agents.supermimic.prompt import SMSpec, sm_spec_velocity
from noeira.deep_agents.supermimic.prompt_agent_gpu import _k_prompt_cmd, _k_prompt_start
from noeira.deep_agents.supermimic.prompt_imagination import SMPromptBatch, sm_imagine_prompt
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfig
from noeira.deep_agents.supermimic.prompt_trainer import SMPromptTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, SMDims
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer, _ScaleParam
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl


comptime R = G1MjlabRobot
comptime D = SMDims[R]
comptime N = 16
comptime H = 8
comptime NREF = 8
comptime CMD = 7
comptime CC = SMConfig["cpu", R, 64, 16, 32, 32, 32, 32, 64, 32, 32, 32, 32, 32, 16, 8, N, H]
comptime CG = SMConfig["gpu", R, 64, 16, 32, 32, 32, 32, 64, 32, 32, 32, 32, 32, 16, 8, N, H]
comptime PC = SMPromptConfig[CC, CMD, 32, 32, 32, 16]
comptime PG = SMPromptConfig[CG, CMD, 32, 32, 32, 16]
comptime GAMMA = 0.99


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


def _flat(name: String, a: List[Scalar[DT]], b: List[Scalar[DT]], mut tl: Tally):
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    var big = 0.0
    var err = 0.0
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
        big = max(big, abs(x))
        err = max(err, abs(x - y))
    var cos = dot / sqrt(max(na * nb, 1e-300))
    tl.check(cos >= 0.99999 and err <= 1e-4 * big, name + ": " + String(len(a)) + " values, cosine " + String(cos) + ", max err " + String(err / max(big, 1e-30)) + " of the largest")


def _lists(name: String, a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]], mut tl: Tally):
    var fa = List[Scalar[DT]]()
    var fb = List[Scalar[DT]]()
    for i in range(len(a)):
        for k in range(len(a[i])):
            fa.append(a[i][k])
            fb.append(b[i][k])
    _flat(name, fa, fb, tl)


def _same(a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]]) -> Bool:
    for k in range(len(a)):
        for i in range(len(a[k])):
            if a[k][i] != b[k][i]:
                return False
    return True


def _batch() -> SMPromptBatch[R, CMD]:
    var ib = SMPromptBatch[R, CMD](N, H, NREF)
    for b in range(N):
        var o = b * D.S
        ib.state.data[o + 2] = Scalar[DT](0.74 + 0.01 * Float64(b))
        ib.state.data[o + 3] = 1
        for j in range(D.NDOF):
            ib.state.data[o + 7 + j] = Scalar[DT](R.default_joint_pos(j) + random_float64(-0.1, 0.1))
            ib.state.data[o + D.NQ + 6 + j] = Scalar[DT](random_float64(-0.5, 0.5))
            ib.prev_action.data[b * R.NDOF + j] = Scalar[DT](random_float64(-0.2, 0.2))
        for k in range(6):
            ib.state.data[o + D.NQ + k] = Scalar[DT](random_float64(-0.3, 0.3))
        for k in range(D.X):
            ib.xi.data[b * D.X + k] = Scalar[DT](random_float64(-0.01, 0.01))
        ib.xi.data[b * D.X] = 0.8
        for hh in range(D.HIST):
            for k in range(D.OBS):
                ib.hist0.data[(hh * N + b) * D.OBS + k] = Scalar[DT](random_float64(-0.3, 0.3))
    for t in range(H + 1):
        for b in range(N):
            for k in range(CMD):
                ib.cmds.data[(t * N + b) * CMD + k] = Scalar[DT](random_float64(-1, 1))
    for t in range(H):
        for b in range(N):
            var sp = ib.specs.data.unsafe_ptr().unsafe_offset((t * N + b) * SMSpec.SIZE).as_unsafe_any_origin()
            sm_spec_velocity[DT](sp, Scalar[DT](0.2 * Float64(b)), Scalar[DT](0.1), Scalar[DT](-0.2 + 0.1 * Float64(t)))
            sp[unsafe_offset=SMSpec.W + SMSpec.G_LHAND] = 0.5
            sp[unsafe_offset=SMSpec.T + SMSpec.A_LX + 2] = 0.3
            sp[unsafe_offset=SMSpec.IW + SMSpec.A_LX + 2] = 20
            for k in range(D.LATENT):
                ib.lat.data[(t * N + b) * D.LATENT + k] = Scalar[DT](random_float64(-0.5, 0.5))
    return ib^


def _sync(mut c: SMPromptTrainer[PC], mut g: SMPromptTrainer[PG], ctx: DeviceContext) raises:
    """The GPU trainer gets the CPU trainer's exact weights."""
    sm_restore["gpu"](g.base.wm, sm_snapshot["cpu"](c.base.wm, None), Optional(ctx))
    sm_restore["gpu"](g.base.pol, sm_snapshot["cpu"](c.base.pol, None), Optional(ctx))
    sm_restore["gpu"](g.enc, sm_snapshot["cpu"](c.enc, None), Optional(ctx))
    sm_restore["gpu"](g.critic, sm_snapshot["cpu"](c.critic, None), Optional(ctx))


def _ints(ctx: DeviceContext, v: List[Int32]) raises -> DeviceBuffer[DType.int32]:
    var b = ctx.enqueue_create_buffer[DType.int32](len(v))
    var h = ctx.enqueue_create_host_buffer[DType.int32](len(v))
    ctx.synchronize()
    for i in range(len(v)):
        h[i] = v[i]
    ctx.enqueue_copy(b, h)
    ctx.synchronize()
    return b^


def main() raises:
    var tl = Tally()
    seed(9)
    var ctx = DeviceContext()
    var tc = SMPromptTrainer[PC](SuperMimicTrainer[CC].for_robot(R.SIM.create()))
    var tg = SMPromptTrainer[PG](SuperMimicTrainer[CG].for_robot(R.SIM.create(), Optional(ctx)))
    # make the action's path matter, as `test_supermimic_prompt_imag` does
    var wv = sm_snapshot["cpu"](tc.base.wm, None)
    for k in range(len(wv)):
        for i in range(len(wv[k])):
            wv[k][i] += Scalar[DT](random_float64(-0.05, 0.05))
    sm_restore["cpu"](tc.base.wm, wv^, None)
    var sc100 = _ScaleParam(String("decoder.4.weight"), Scalar[DT](100.0))
    tc.base.pol.for_each_param["cpu"](sc100, None)
    _sync(tc, tg, ctx)
    var ib = _batch()
    var ig = SMPromptBatch[R, CMD](N, H, NREF)
    ig.state.data = ib.state.data.copy()
    ig.prev_action.data = ib.prev_action.data.copy()
    ig.xi.data = ib.xi.data.copy()
    ig.hist0.data = ib.hist0.data.copy()
    ig.cmds.data = ib.cmds.data.copy()
    ig.specs.data = ib.specs.data.copy()
    ig.lat.data = ib.lat.data.copy()
    ig.to_device(ctx)

    print("[1] the rollout")
    tc.enc.zero_grad["cpu"](None)
    tg.enc.zero_grad["gpu"](Optional(ctx))
    var oc = sm_imagine_prompt[PC, N, "cpu"](tc.base.wm, tc.base.pol, tc.enc, tc.critic, tc.base.tables, ib, GAMMA, 0.1, True, 77)
    var og = sm_imagine_prompt[PG, N, "gpu"](tg.base.wm, tg.base.pol, tg.enc, tg.critic, tg.base.tables, ig, GAMMA, 0.1, True, 77, Optional(ctx))
    ctx.synchronize()
    og.to_host(ctx)
    _flat("states", oc.states.data, og.states.data, tl)
    _flat("actions", oc.actions.data, og.actions.data, tl)
    _flat("rewards", oc.rewards.data, og.rewards.data, tl)
    _flat("survival", oc.survival.data, og.survival.data, tl)
    tl.check(abs(oc.j - og.j) <= 1e-4 * abs(oc.j), "J " + String(oc.j) + " vs " + String(og.j))
    tl.check(abs(oc.align_mse - og.align_mse) <= 1e-4 * abs(oc.align_mse), "the alignment's MSE")

    print("[2] the gradient")
    _lists("over the encoder", sm_snapshot["cpu"](tc.enc, None, grads=True), sm_snapshot["gpu"](tg.enc, Optional(ctx), grads=True), tl)

    print("[3] one policy update")
    var p0 = sm_snapshot["gpu"](tg.base.pol, Optional(ctx))
    var w0 = sm_snapshot["gpu"](tg.base.wm, Optional(ctx))
    var e0 = sm_snapshot["gpu"](tg.enc, Optional(ctx))
    var vals = List[Float64]()
    var tgts = List[Float64]()
    var perms = List[Int](length=5 * H * N, fill=0)
    for ep in range(5):
        for i in range(H * N):
            perms[ep * H * N + i] = i
    _ = tc.policy_update[N](ib, perms, vals, tgts, False)
    var up = tg.policy_update[N](ig, perms, vals, tgts, False, resident=True)
    print("  J", up.imag_objective, "|g|", up.policy_grad_norm, "critic", up.critic_loss)
    _lists("the encoder after its step", sm_snapshot["cpu"](tc.enc, None), sm_snapshot["gpu"](tg.enc, Optional(ctx)), tl)
    _lists("its critic after 20 steps", sm_snapshot["cpu"](tc.critic, None), sm_snapshot["gpu"](tg.critic, Optional(ctx)), tl)
    tl.check(not _same(e0, sm_snapshot["gpu"](tg.enc, Optional(ctx))), "the encoder moved on the device")
    tl.check(_same(p0, sm_snapshot["gpu"](tg.base.pol, Optional(ctx))), "the decoder and the motion encoder did not")
    tl.check(_same(w0, sm_snapshot["gpu"](tg.base.wm, Optional(ctx))), "the world model did not")

    print("[4] a mixed act")
    var hist = TensorImpl[DT].alloc(N * D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(N * D.GOAL)
    var cmds = TensorImpl[DT].alloc(N * CMD)
    for i in range(N * D.HIST * D.OBS):
        hist.data[i] = Scalar[DT](random_float64(-0.3, 0.3))
    for i in range(N * D.GOAL):
        goals.data[i] = Scalar[DT](random_float64(-0.3, 0.3))
    for i in range(N * CMD):
        cmds.data[i] = Scalar[DT](random_float64(-1, 1))
    var ac = tc.act[N](hist, goals, cmds, 5)
    var ag = tg.act[N](hist, goals, cmds, 5)
    _flat("lanes split at 5", ac.data, ag.data, tl)

    print("[5] the agent's kernels")
    comptime L = 8
    comptime FR = 6
    comptime KB = 4
    comptime RW = CMD + SMSpec.SIZE
    var cmd_ref = TensorImpl[DT].alloc(FR * CMD)
    var vel_ref = TensorImpl[DT].alloc(FR * 3)
    var bank = TensorImpl[DT].alloc(KB * RW)
    for f in range(FR):
        for k in range(CMD):
            cmd_ref.data[f * CMD + k] = Scalar[DT](100 * f + k)
        vel_ref.data[f * 3] = Scalar[DT](0.1 * Float64(f))
    for b in range(KB):
        for k in range(RW):
            bank.data[b * RW + k] = Scalar[DT](1000 * (b + 1) + k)
    cmd_ref.upload_resident(ctx)
    vel_ref.upload_resident(ctx)
    bank.upload_resident(ctx)
    var frame = List[Int32](length=L, fill=3)
    var ep_len = List[Int32](length=L, fill=0)
    var src_h = List[Int32](length=L, fill=7)
    var hold_h = List[Int32](length=L, fill=0)
    var fb = _ints(ctx, frame)
    var eb = _ints(ctx, ep_len)
    var sb = _ints(ctx, src_h)
    var hb = _ints(ctx, hold_h)
    var ocmd = talloc["gpu"](L * CMD, Optional(ctx))
    var ospec = talloc["gpu"](L * SMSpec.SIZE, Optional(ctx))
    ctx.enqueue_function[_k_prompt_cmd[CMD]](
        Int64(L), Int64(2), Int64(250), Scalar[DT](0.5), UInt32(1234), Int64(KB),
        mptr(fb.unsafe_ptr()), mptr(eb.unsafe_ptr()), mptr(sb.unsafe_ptr()), mptr(hb.unsafe_ptr()),
        tp["gpu"](cmd_ref), tp["gpu"](vel_ref), tp["gpu"](bank), tp["gpu"](ocmd), tp["gpu"](ospec),
        grid_dim=1, block_dim=64,
    )
    ctx.synchronize()
    ocmd.download(ctx)
    ospec.download(ctx)
    var hs = ctx.enqueue_create_host_buffer[DType.int32](L)
    var hh2 = ctx.enqueue_create_host_buffer[DType.int32](L)
    ctx.enqueue_copy(hs, sb)
    ctx.enqueue_copy(hh2, hb)
    ctx.synchronize()
    var below = hs[0] == 7 and hs[1] == 7
    var cmd_ok = True
    var nrefc = 0
    var hold_ok = True
    for l in range(2, L):
        hold_ok = hold_ok and hh2[l] == 249
        var s = Int(hs[l])
        if s < 0:
            nrefc += 1
            cmd_ok = cmd_ok and ocmd.data[l * CMD] == cmd_ref.data[3 * CMD] and abs(Float64(ospec.data[l * SMSpec.SIZE + SMSpec.T + SMSpec.A_VX]) - 0.3) < 1e-6
        else:
            cmd_ok = cmd_ok and s < KB and ocmd.data[l * CMD] == bank.data[s * RW] and ospec.data[l * SMSpec.SIZE] == bank.data[s * RW + CMD]
    print("  sources", hs[2], hs[3], hs[4], hs[5], hs[6], hs[7])
    tl.check(below, "lanes below the split are left alone")
    tl.check(hold_ok, "a drawn lane holds for the configured steps")
    tl.check(cmd_ok, "each lane reads its source's command and spec")

    # a start's inputs: two starts on reference commands, one past its clip's end,
    # two holding a bank row
    comptime NB = 5
    comptime HS = 4
    var lat_t = TensorImpl[DT].alloc(FR * D.LATENT)
    for f in range(FR):
        for k in range(D.LATENT):
            lat_t.data[f * D.LATENT + k] = Scalar[DT](10 * f + k)
    lat_t.upload_resident(ctx)
    var last = _ints(ctx, [5, 5, 5, 5, 5, 5])            # one clip: frames 0 .. 5
    var fr_s = _ints(ctx, [1, 4, 0, 0, 0])
    var hist_s = TensorImpl[DT].alloc(D.HIST * NB * D.S)
    var hist_a = TensorImpl[DT].alloc(D.HIST * NB * R.NDOF)
    var xi_s = TensorImpl[DT].alloc(NB * D.X)
    for i in range(D.HIST * NB * D.S):
        hist_s.data[i] = Scalar[DT](random_float64(-0.5, 0.5))
    for hh in range(D.HIST):                              # unit root quaternions
        for b in range(NB):
            var o = (hh * NB + b) * D.S
            hist_s.data[o + 3] = 1
            hist_s.data[o + 4] = 0
            hist_s.data[o + 5] = 0
            hist_s.data[o + 6] = 0
    hist_s.upload_resident(ctx)
    hist_a.upload_resident(ctx)
    xi_s.upload_resident(ctx)
    var ob0 = talloc["gpu"](D.HIST * NB * D.OBS, Optional(ctx))
    var oc2 = talloc["gpu"]((HS + 1) * NB * CMD, Optional(ctx))
    var os2 = talloc["gpu"](HS * NB * SMSpec.SIZE, Optional(ctx))
    var ol2 = talloc["gpu"](HS * NB * D.LATENT, Optional(ctx))
    ctx.enqueue_function[_k_prompt_start[R, CMD]](
        Int64(NB), Int64(HS), Int64(3), UInt32(55), Int64(0), Int64(KB), tg.base.tables.head_p["gpu"](), mptr(last.unsafe_ptr()),
        mptr(fr_s.unsafe_ptr()), tp["gpu"](hist_s), tp["gpu"](hist_a), tp["gpu"](xi_s),
        tp["gpu"](cmd_ref), tp["gpu"](vel_ref), tp["gpu"](lat_t), tp["gpu"](bank),
        tp["gpu"](ob0), tp["gpu"](oc2), tp["gpu"](os2), tp["gpu"](ol2),
        grid_dim=1, block_dim=64,
    )
    ctx.synchronize()
    ob0.download(ctx)
    oc2.download(ctx)
    os2.download(ctx)
    ol2.download(ctx)
    var ref_ok = True
    var frames0: List[Int] = [1, 4, 0]
    for b in range(3):
        for t in range(HS + 1):
            var f = min(frames0[b] + t, 5)
            ref_ok = ref_ok and oc2.data[(t * NB + b) * CMD + 2] == cmd_ref.data[f * CMD + 2]
            if t < HS:
                ref_ok = ref_ok and ol2.data[(t * NB + b) * D.LATENT + 3] == lat_t.data[f * D.LATENT + 3]
                ref_ok = ref_ok and abs(Float64(os2.data[(t * NB + b) * SMSpec.SIZE + SMSpec.T + SMSpec.A_VX]) - 0.1 * Float64(f)) < 1e-6
    var held_ok = True
    for b in range(3, NB):
        var row = Int(oc2.data[b * CMD] / 1000) - 1
        held_ok = held_ok and row >= 0 and row < KB
        for t in range(HS + 1):
            held_ok = held_ok and oc2.data[(t * NB + b) * CMD + 1] == bank.data[row * RW + 1]
            if t < HS:
                held_ok = held_ok and os2.data[(t * NB + b) * SMSpec.SIZE] == bank.data[row * RW + CMD]
    # the history: slot 7's gravity on an identity root is (0, 0, −1)
    var obs_ok = True
    for b in range(NB):
        obs_ok = obs_ok and abs(Float64(ob0.data[(7 * NB + b) * D.OBS + 2 * D.NDOF + 5]) + 1.0) < 1e-6
    tl.check(ref_ok, "reference starts read their frames, and hold the clip's last one past its end")
    tl.check(held_ok, "held starts read one bank row for the whole rollout")
    tl.check(obs_ok, "the history is rebuilt from the logged states")

    print("\n===", tl.ok, "/", tl.total, "passed ===")
    if tl.ok != tl.total:
        raise Error("test_supermimic_prompt_gpu failed")
