"""SuperMimic's prompt interface: its rollout and gradient (`prompt_imagination`).

    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_prompt_imag.mojo

There is no reference implementation (the reproduction has no downstream
interface), so the two references are built here, independent of the scan:

WHAT EACH SECTION CAN CATCH
===========================
[1] the batched forward against a one-lane rollout composed by hand from the
    primitives, lane by lane: states, actions, rewards, survival, V and J. A
    stride or slot slip in the scan (the history window, the t-major layouts,
    the command at H) shows here. Lanes are unequal on purpose.
[2] −J/H's gradient against central differences of the forward along random
    directions of the encoder's parameters, with and without observation
    noise. Float32 forward: 2 % on the directional derivative, and the section
    checks |ΔJ| clears float32's rounding (at the decoder's and the encoder's
    initial gains it does not, and the comparison is noise: the fixture raises
    both).
[3] the alignment's gradient alone, the same way: a reward of zero and a
    critic of zero leave only it. Its lanes are the first `nref`: the others
    must not contribute (checked by moving their targets).
[4] the stop-gradient. One actor step moves the encoder and nothing else; one
    critic step moves its critic and nothing else; the decoder, the motion
    encoder and the world model stay bit-identical.
[5] one `trainer.act`: lanes below the split are the tracker's actions, the
    others the prompt interface's, each equal to a pure batch of that kind.

Planted mutants this gate was run against are in the plan's progress notes.
"""

from std.math import exp
from std.random import random_float64, seed

from noeira.deep_agents.supermimic.config import SMConfig, SMCriticOf, SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.blocks.prompt_step import PromptActorStep, PromptCriticStep
from noeira.deep_agents.supermimic.heads import sm_observation_lane
from noeira.deep_agents.supermimic.imagination import SMImagOut
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import sm_dec_in, sm_wm_in, talloc, tp
from noeira.deep_agents.supermimic.params import sm_restore, sm_snapshot
from noeira.deep_agents.supermimic.prompt import SMPromptScratch, SMSpec, sm_prompt_fall_logit, sm_prompt_reward_lane, sm_spec_velocity
from noeira.deep_agents.supermimic.prompt_imagination import SMPromptBatch, sm_imagine_prompt
from noeira.deep_agents.supermimic.prompt_lanes import sm_p_critic_in, sm_p_enc_in
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfig, SMPromptCriticOf, SMPromptEncOf
from noeira.deep_agents.supermimic.prompt_trainer import SMPromptTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, SMDims
from noeira.deep_agents.supermimic.state_spec import sm_integrate_lane
from noeira.deep_agents.supermimic.tables import SMTables, sm_tables
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer, _ScaleParam
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


comptime R = G1MjlabRobot
comptime D = SMDims[R]
comptime N = 8
comptime H = 8
comptime NREF = 4
comptime CMD = 7
comptime C = SMConfig["cpu", R, 64, 16, 32, 32, 32, 32, 64, 32, 32, 32, 32, 32, 8, 8, N, H]
comptime P = SMPromptConfig[C, CMD, 32, 32, 32, 16]
comptime GAMMA = 0.99


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


def _batch(mut tb: SMTables[DT]) -> SMPromptBatch[R, CMD]:
    """Lanes unequal in pose, velocity, command and spec."""
    var ib = SMPromptBatch[R, CMD](N, H, NREF)
    var s0 = List[Scalar[DT]](length=D.S, fill=0)
    for b in range(N):
        var o = b * D.S
        ib.state.data[o + 2] = Scalar[DT](0.76 + 0.01 * Float64(b))
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
            sm_spec_velocity[DT](sp, Scalar[DT](0.3 * Float64(b)), Scalar[DT](0.1), Scalar[DT](-0.2 + 0.1 * Float64(t)))
            sp[unsafe_offset=SMSpec.W + SMSpec.G_H] = 1
            sp[unsafe_offset=SMSpec.T + SMSpec.A_H] = 0.7
            sp[unsafe_offset=SMSpec.IW + SMSpec.A_H] = 40
            sp[unsafe_offset=SMSpec.W + SMSpec.G_LHAND] = 0.5
            sp[unsafe_offset=SMSpec.T + SMSpec.A_LX + 2] = 0.3
            sp[unsafe_offset=SMSpec.IW + SMSpec.A_LX + 2] = 20
            for k in range(D.LATENT):
                ib.lat.data[(t * N + b) * D.LATENT + k] = Scalar[DT](random_float64(-0.5, 0.5))
    return ib^


def _nudge(vals: List[List[Scalar[DT]]], dirs: List[List[Scalar[DT]]], e: Float64) -> List[List[Scalar[DT]]]:
    var out = vals.copy()
    for k in range(len(out)):
        for i in range(len(out[k])):
            out[k][i] = vals[k][i] + Scalar[DT](e) * dirs[k][i]
    return out^


def _dirs(vals: List[List[Scalar[DT]]]) -> List[List[Scalar[DT]]]:
    var out = vals.copy()
    for k in range(len(out)):
        for i in range(len(out[k])):
            out[k][i] = Scalar[DT](random_float64(-1, 1))
    return out^


def _dot(a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]]) -> Float64:
    var s = 0.0
    for k in range(len(a)):
        for i in range(len(a[k])):
            s += Float64(a[k][i]) * Float64(b[k][i])
    return s


def _same(a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]]) -> Bool:
    for k in range(len(a)):
        for i in range(len(a[k])):
            if a[k][i] != b[k][i]:
                return False
    return True


def _moved(a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]]) -> Bool:
    return not _same(a, b)


def main() raises:
    var t = Tally()
    seed(5)
    var sim = R.SIM.create()
    var q0 = List[Float64](length=D.NDOF, fill=0.0)
    for j in range(D.NDOF):
        q0[j] = R.default_joint_pos(j)
    var tb = sm_tables[DT, R](sim, q0)
    var trainer = SMPromptTrainer[P](SuperMimicTrainer[C].for_robot(R.SIM.create()))
    # the dynamics' last layer starts at zero, which would hide the action's path
    var wv = sm_snapshot["cpu"](trainer.base.wm, None)
    for k in range(len(wv)):
        for i in range(len(wv[k])):
            wv[k][i] += Scalar[DT](random_float64(-0.05, 0.05))
    sm_restore["cpu"](trainer.base.wm, wv^, None)
    # ⚠ and the decoder's last layer starts at gain 0.01: at that gain J moves by
    # 1e-8 along any direction, float32's rounding, and [2] measures nothing
    var sc100 = _ScaleParam(String("decoder.4.weight"), Scalar[DT](100.0))
    trainer.base.pol.for_each_param["cpu"](sc100, None)
    if sc100.hit != 1:
        raise Error("the decoder's last layer was not found")
    # the same for the encoder's last layer, which the trainer starts at 0.01
    var se100 = _ScaleParam(String("2.weight"), Scalar[DT](100.0))
    trainer.enc.for_each_param["cpu"](se100, None)
    if se100.hit != 1:
        raise Error("the encoder's last layer was not found")
    var ib = _batch(tb)

    print("\n[1] the batched forward against one lane by hand")
    var r = sm_imagine_prompt[P, N, "cpu"](
        trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, 0.1, False,
    )
    var worst_s = 0.0
    var worst_r = 0.0
    var worst_c = 0.0
    var jhand = 0.0
    var sc = List[Scalar[DT]](length=SMPromptScratch[R].SIZE, fill=0)
    var spc = sc.unsafe_ptr().as_unsafe_any_origin()
    for b in range(N):
        # O slots of this lane alone, oldest first
        var O = TensorImpl[DT].alloc((D.HIST + H) * D.OBS)
        for hh in range(D.HIST):
            for k in range(D.OBS):
                O.data[hh * D.OBS + k] = ib.hist0.data[(hh * N + b) * D.OBS + k]
        var s = TensorImpl[DT].alloc((H + 1) * D.S)
        for i in range(D.S):
            s.data[i] = ib.state.data[b * D.S + i]
        var xi = TensorImpl[DT].alloc(D.X)
        for i in range(D.X):
            xi.data[i] = ib.xi.data[b * D.X + i]
        var ap = TensorImpl[DT].alloc(R.NDOF)
        for i in range(R.NDOF):
            ap.data[i] = ib.prev_action.data[b * R.NDOF + i]
        var c = 1.0
        var gp = 1.0
        var j = 0.0
        for st in range(H):
            var cmd = TensorImpl[DT].alloc(CMD)
            for k in range(CMD):
                cmd.data[k] = ib.cmds.data[(st * N + b) * CMD + k]
            var ein = TensorImpl[DT].alloc(CMD + D.HIST_FLAT)
            var op = tp["cpu"](O).unsafe_offset(st * D.OBS)
            sm_p_enc_in["cpu", R, CMD](None, 1, tp["cpu"](cmd), op, D.OBS, D.OBS, tp["cpu"](ein))
            var z = TensorImpl[DT].alloc(D.LATENT)
            trainer.enc.forward["cpu", 1](TensorRefs[1, _, DT](ein), z, None)
            var din = TensorImpl[DT].alloc(D.DEC_IN)
            sm_dec_in["cpu", R](None, 1, tp["cpu"](z), op, D.OBS, D.OBS, tp["cpu"](din))
            var a = TensorImpl[DT].alloc(R.NDOF)
            trainer.base.pol.dec.forward["cpu", 1](TensorRefs[1, _, DT](din), a, None)
            var x = TensorImpl[DT].alloc(D.WM_IN)
            var s_p = tp["cpu"](s).unsafe_offset(st * D.S)
            var s1 = tp["cpu"](s).unsafe_offset((st + 1) * D.S)
            sm_wm_in["cpu", R](None, 1, s_p, tp["cpu"](a), tp["cpu"](xi), tp["cpu"](x))
            var dl = TensorImpl[DT].alloc(D.DELTA)
            trainer.base.wm.dyn.forward["cpu", 1](TensorRefs[1, _, DT](x), dl, None)
            sm_integrate_lane[DT, R](s_p, tp["cpu"](dl), s1)
            var spec = ib.specs.data.unsafe_ptr().unsafe_offset((st * N + b) * SMSpec.SIZE).as_unsafe_any_origin()
            var rew = sm_prompt_reward_lane[DT, R](tb.tree_ptr(), tb.head_ptr(), s1, tp["cpu"](a), tp["cpu"](ap), spec, spc)
            sm_observation_lane[DT, R](tb.head_ptr(), s1, tp["cpu"](a), tp["cpu"](xi), tp["cpu"](O).unsafe_offset((D.HIST + st) * D.OBS))
            var lg = Float64(sm_prompt_fall_logit[DT](s1))
            var d = 1.0 / (1.0 + exp(-lg))
            j += gp * c * Float64(rew)
            worst_r = max(worst_r, abs(Float64(rew) - Float64(r.rewards.data[st * N + b])))
            worst_c = max(worst_c, abs(c - Float64(r.survival.data[st * N + b])))
            for k in range(D.S):
                worst_s = max(worst_s, abs(Float64(s.data[(st + 1) * D.S + k]) - Float64(r.states.data[((st + 1) * N + b) * D.S + k])))
            for k in range(R.NDOF):
                worst_s = max(worst_s, abs(Float64(a.data[k]) - Float64(r.actions.data[(st * N + b) * R.NDOF + k])))
                ap.data[k] = a.data[k]
            c *= 1.0 - d
            gp *= GAMMA
        var cin = TensorImpl[DT].alloc(D.S + CMD + D.X)
        var cmdh = TensorImpl[DT].alloc(CMD)
        for k in range(CMD):
            cmdh.data[k] = ib.cmds.data[(H * N + b) * CMD + k]
        sm_p_critic_in["cpu", R, CMD](None, 1, 1, tp["cpu"](s).unsafe_offset(H * D.S), tp["cpu"](cmdh), tp["cpu"](xi), tp["cpu"](cin))
        var v = TensorImpl[DT].alloc(1)
        trainer.critic.forward["cpu", 1](TensorRefs[1, _, DT](cin), v, None)
        j += gp * c * Float64(v.data[0])
        jhand += j / Float64(N)
    print("  states + actions", worst_s, "rewards", worst_r, "survival", worst_c, "| J", r.j, "by hand", jhand)
    t.truth(worst_s < 2e-4, "states and actions, every step of every lane")
    t.truth(worst_r < 1e-5, "rewards")
    t.truth(worst_c < 1e-5, "survival")
    t.truth(abs(jhand - r.j) < 1e-4 * max(abs(jhand), 1.0), "J")

    print("\n[2] −J/H's gradient against central differences")
    var e = 1e-2
    for noise in range(2):
        var nseed = UInt32(77) if noise == 1 else UInt32(0)
        trainer.enc.zero_grad["cpu"](None)
        _ = sm_imagine_prompt[P, N, "cpu"](
            trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, 0.0, True, nseed,
        )
        var g = sm_snapshot["cpu"](trainer.enc, None, grads=True)
        var th = sm_snapshot["cpu"](trainer.enc, None)
        var worst = 0.0
        var smallest = 1e9
        for _ in range(6):
            var d = _dirs(th)
            sm_restore["cpu"](trainer.enc, _nudge(th, d, e), None)
            var jp = sm_imagine_prompt[P, N, "cpu"](trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, 0.0, False, nseed).j
            sm_restore["cpu"](trainer.enc, _nudge(th, d, -e), None)
            var jm = sm_imagine_prompt[P, N, "cpu"](trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, 0.0, False, nseed).j
            var fd = -(jp - jm) / (2 * e) / Float64(H)
            var an = _dot(g, d)
            smallest = min(smallest, abs(jp - jm))
            worst = max(worst, abs(fd - an) / max(abs(fd), 1e-6))
        sm_restore["cpu"](trainer.enc, th^, None)
        print("  noise", noise, ": worst relative error", worst, "| smallest |ΔJ|", smallest)
        t.truth(smallest > 2e-6, "ΔJ clears float32's rounding (~3e-8) by 60×: a 2 % band reads noise below that")
        t.truth(worst < 0.02, "the directional derivative, noise " + String(noise))

    print("\n[3] the alignment's gradient")
    var w = 0.1
    trainer.enc.zero_grad["cpu"](None)
    _ = sm_imagine_prompt[P, N, "cpu"](trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, w, True)
    var g1 = sm_snapshot["cpu"](trainer.enc, None, grads=True)
    trainer.enc.zero_grad["cpu"](None)
    _ = sm_imagine_prompt[P, N, "cpu"](trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, 0.0, True)
    var g0 = sm_snapshot["cpu"](trainer.enc, None, grads=True)
    var th3 = sm_snapshot["cpu"](trainer.enc, None)
    var ga3 = g1.copy()
    for k in range(len(ga3)):
        for i in range(len(ga3[k])):
            ga3[k][i] = g1[k][i] - g0[k][i]
    var worst3 = 0.0
    var e3 = 1e-3          # a quadratic with plenty of signal: a small step
    for _ in range(6):
        var d = _dirs(th3)
        sm_restore["cpu"](trainer.enc, _nudge(th3, d, e3), None)
        var ap3 = sm_imagine_prompt[P, N, "cpu"](trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, w, False).align_mse
        sm_restore["cpu"](trainer.enc, _nudge(th3, d, -e3), None)
        var am3 = sm_imagine_prompt[P, N, "cpu"](trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, w, False).align_mse
        var fd = w * (ap3 - am3) / (2 * e3)
        worst3 = max(worst3, abs(fd - _dot(ga3, d)) / max(abs(fd), 1e-7))
    sm_restore["cpu"](trainer.enc, th3^, None)
    print("  worst relative error", worst3)
    t.truth(worst3 < 0.03, "the directional derivative of w · MSE")
    var base_mse = sm_imagine_prompt[P, N, "cpu"](trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, w, False).align_mse
    var lat_saved = ib.lat.data.copy()
    for st in range(H):
        for b in range(NREF, N):
            for k in range(D.LATENT):
                ib.lat.data[(st * N + b) * D.LATENT + k] = 9
    var other_mse = sm_imagine_prompt[P, N, "cpu"](trainer.base.wm, trainer.base.pol, trainer.enc, trainer.critic, tb, ib, GAMMA, w, False).align_mse
    ib.lat.data = lat_saved^
    t.truth(base_mse == other_mse, "lanes past `nref` carry no alignment")

    print("\n[4] the stop-gradient")
    var p0 = sm_snapshot["cpu"](trainer.base.pol, None)
    var w0 = sm_snapshot["cpu"](trainer.base.wm, None)
    var c0 = sm_snapshot["cpu"](trainer.base.critic, None)
    var e0 = sm_snapshot["cpu"](trainer.enc, None)
    var v0 = sm_snapshot["cpu"](trainer.critic, None)
    var vals = List[Float64]()
    var tgts = List[Float64]()
    var none = List[Int]()
    var up = trainer.policy_update[N](ib, none, vals, tgts, True)
    print("  J", up.imag_objective, "|g|", up.policy_grad_norm, "align", up.align_mse, "critic", up.critic_loss)
    t.truth(_moved(e0, sm_snapshot["cpu"](trainer.enc, None)), "the prompt encoder moved")
    t.truth(_moved(v0, sm_snapshot["cpu"](trainer.critic, None)), "its critic moved")
    t.truth(_same(p0, sm_snapshot["cpu"](trainer.base.pol, None)), "the decoder and the motion encoder did not")
    t.truth(_same(w0, sm_snapshot["cpu"](trainer.base.wm, None)), "the world model did not")
    t.truth(_same(c0, sm_snapshot["cpu"](trainer.base.critic, None)), "the tracker's critic did not")

    print("\n[5] a mixed batch of lanes")
    var hist = TensorImpl[DT].alloc(N * D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(N * D.GOAL)
    var cmds = TensorImpl[DT].alloc(N * CMD)
    for i in range(N * D.HIST * D.OBS):
        hist.data[i] = Scalar[DT](random_float64(-0.3, 0.3))
    for i in range(N * D.GOAL):
        goals.data[i] = Scalar[DT](random_float64(-0.3, 0.3))
    for i in range(N * CMD):
        cmds.data[i] = Scalar[DT](random_float64(-1, 1))
    var a_t = trainer.act[N](hist, goals, cmds, N)
    var a_p = trainer.act[N](hist, goals, cmds, 0)
    var a_m = trainer.act[N](hist, goals, cmds, 3)
    var ok_t = True
    var ok_p = True
    var differ = False
    for l in range(N):
        for j in range(R.NDOF):
            var v = a_m.data[l * R.NDOF + j]
            if l < 3:
                ok_t = ok_t and v == a_t.data[l * R.NDOF + j]
            else:
                ok_p = ok_p and v == a_p.data[l * R.NDOF + j]
            differ = differ or a_t.data[l * R.NDOF + j] != a_p.data[l * R.NDOF + j]
    t.truth(differ, "the two interfaces act differently here (the gate is not vacuous)")
    t.truth(ok_t, "lanes below the split are the tracker's")
    t.truth(ok_p, "lanes from the split on are the prompt's")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_prompt_imag: " + String(t.fails) + " failed")
