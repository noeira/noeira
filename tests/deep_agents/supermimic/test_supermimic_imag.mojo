"""SuperMimic M5: the H = 32 imagined rollout and its policy gradient.

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/imag_oracle.py --out /tmp/sm_imag
    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_imag.mojo /tmp/sm_imag

The oracle is the reproduction's `imagine` + `ImagStep` + `policy_objective`
(and `Policy`, `Critic`, `WorldModel`) run verbatim in torch float64 from
float32-rounded weights and inputs, on a small random reference library; this
side is float32 (`nn`). Small widths, the paper's H = 32 and γ = 0.99, 8
lanes, one of which reaches its clip's end inside the horizon.

WHAT EACH SECTION CAN CATCH
===========================
[1] all five networks load by NAME in the oracle's order (dynamics,
    termination head, encoder, decoder, critic).
[2] the forward rollout, every step: states, actions, rewards, d̂,
    survival, root-error features, V(s_H), and J to 1e-4 relative (plan M5).
    Float32 over 32 steps through a world model: per-element bounds are
    1e-3 of each quantity's scale.
[4] one `PolicyTrainer.update` (the reproduction's, verbatim, with its
    `randperm` draws recorded and replayed here): V on all (H+1)·N imagined
    states and the TD(λ) targets (1e-5 of scale), the pre-clip policy grad
    norm, the policy after its clipped Adam step (well-conditioned elements
    within 2 fp32 ulps of max(|w|, lr)), the critic after its 20 minibatch
    steps (elements that moved ≥ 10 % of the largest move: to 1e-3 of it),
    the last minibatch loss (1e-4).
[3] the policy gradient (encoder + decoder, every parameter): cosine
    ≥ 0.99999 AND max error ≤ 1e-4 of the largest component, per network.
    Measured agreement is 3e-7 of the largest.

⚠ PLAN M5's "COSINE ≥ 0.999" IS NOT A GATE. Four planted cuts of the scan's
carried gradients, measured (encoder / decoder):
    no gradient into the observation history   cosine 0.99997 / 0.99999,  max err 8e-3 / 4e-3
    no a_prev gradient (action-rate term)      cosine 0.99998 / 0.999998, max err 6e-3 / 9e-4
    no survival-chain gradient                 cosine 0.9986  / 0.9948,   max err 5e-2 / 8e-2
    no critic gradient into s_H                cosine 0.9993  / 0.9990,   max err 6e-2 / 5e-2
The first two pass at 0.999 and at a 1e-2 error bound; the bounds above
fail all four. A swapped λ / (1 − λ) in the TD(λ) recursion fails [4].

NOT COVERED: `clip_grad_value(0.1)` never binds here (the policy's gradient
norm is 4e-3), so a broken value clip would pass.
"""

from std.math import sqrt
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.core.bytes import string_from_bytes
from noeira.deep_agents.supermimic.config import SMConfig, SMCriticOf, SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.imagination import SMImagBatch, sm_imagine
from noeira.deep_agents.supermimic.robot import G1Robot, G1Sim, SMDims
from noeira.deep_agents.supermimic.tables import sm_tables
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer
from noeira.io.fileio import read_file_bytes
from noeira.io.npy import load_npy_2d_f64
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor


comptime N = 8
comptime H = 32
# the oracle's widths (imag_oracle.py): WM 32 / 16, encoder 24-20-16-12, decoder 28-22-18-14, critic 16-12
comptime TC = SMConfig["cpu", G1Robot, 32, 16, 24, 20, 16, 12, 28, 22, 18, 14, 16, 12, N, 64, N, H]
comptime SM_GOAL_DIM = SMDims[G1Robot].GOAL
comptime SM_OBS = SMDims[G1Robot].OBS
comptime SM_REF_DIM = SMDims[G1Robot].REF


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


struct _Load(ParamVisitor):
    var vals: List[Float64]
    var names: List[String]
    var sizes: List[Int]
    var at: Int
    var k: Int
    var bad: Int

    def __init__(out self, var vals: List[Float64], var names: List[String], var sizes: List[Int]):
        self.vals = vals^
        self.names = names^
        self.sizes = sizes^
        self.at = 0
        self.k = 0
        self.bad = 0

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^
        self.names = move.names^
        self.sizes = move.sizes^
        self.at = move.at
        self.k = move.k
        self.bad = move.bad

    def visit[target: StaticString, N_: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        if self.k >= len(self.names) or self.names[self.k] != name or self.sizes[self.k] != N_:
            print("    param", self.k, "is", name, N_, "; the oracle has",
                  self.names[self.k] if self.k < len(self.names) else String("nothing"))
            self.bad += 1
        for i in range(N_):
            param.data[i] = Scalar[DT](self.vals[self.at + i])
        self.at += N_
        self.k += 1


struct _Grads(ParamVisitor):
    var out: List[Float64]

    def __init__(out self):
        self.out = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.out = move.out^

    def visit[target: StaticString, N_: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N_):
            self.out.append(Float64(grad.data[i]))


struct _Params(ParamVisitor):
    var out: List[Float64]

    def __init__(out self):
        self.out = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.out = move.out^

    def visit[target: StaticString, N_: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N_):
            self.out.append(Float64(param.data[i]))


def _fill(mut dst: List[Scalar[DT]], src: List[Float64]) raises:
    if len(dst) != len(src):
        raise Error("size " + String(len(src)) + " for a buffer of " + String(len(dst)))
    for i in range(len(src)):
        dst[i] = Scalar[DT](src[i])


def _err(got: List[Scalar[DT]], want: List[Float64]) -> Tuple[Float64, Float64]:
    """(max |got − want|, max |want|)."""
    var e = 0.0
    var sc = 0.0
    for i in range(len(want)):
        e = max(e, abs(Float64(got[i]) - want[i]))
        sc = max(sc, abs(want[i]))
    return (e, sc)


def main() raises:
    var av = argv()
    var dump = String(av[1]) if len(av) > 1 else String("/tmp/sm_imag")
    var t = Tally()
    var sim = G1Sim.create()

    print("\n[1] the five networks")
    var names = List[String]()
    var sizes = List[Int]()
    for l in string_from_bytes(read_file_bytes(dump + "/i_names.txt")).split("\n"):
        var ls = String(l)
        if ls.byte_length() == 0:
            continue
        var f = ls.split("\t")
        names.append(String(f[0]))
        sizes.append(Int(String(f[1])))
    var total = 0
    for k in range(len(sizes)):
        total += sizes[k]
    var wm = SMWorldModelOf[TC].make[Kaiming]()
    var pol = SMPolicyOf[TC].make[Kaiming]()
    var critic = SMCriticOf[TC].make["cpu", Kaiming](None)
    var ld = _Load(load_npy_2d_f64(dump + "/i_weights.npy", total).values.copy(), names.copy(), sizes.copy())
    wm.for_each_param["cpu"](ld, None)
    pol.for_each_param["cpu"](ld, None)
    critic.for_each_param["cpu"](ld, None, String("critic"))
    t.truth(ld.bad == 0 and ld.k == len(names) and ld.at == total, String(ld.k) + " params by name, " + String(total) + " values")

    var q0 = load_npy_2d_f64(dump + "/i_q0.npy", 29).values.copy()
    var tb = sm_tables[DT, G1Robot](sim, q0)
    var ib = SMImagBatch[G1Robot](N, H)
    _fill(ib.state.data, load_npy_2d_f64(dump + "/i_state.npy", 71).values)
    _fill(ib.prev_action.data, load_npy_2d_f64(dump + "/i_prev_action.npy", 29).values)
    _fill(ib.xi.data, load_npy_2d_f64(dump + "/i_xi.npy", 33).values)
    _fill(ib.hist0.data, load_npy_2d_f64(dump + "/i_hist0.npy", SM_OBS).values)
    _fill(ib.goals.data, load_npy_2d_f64(dump + "/i_goals.npy", SM_GOAL_DIM).values)
    _fill(ib.refs.data, load_npy_2d_f64(dump + "/i_refs.npy", SM_REF_DIM).values)

    print("\n[2] the forward rollout, H =", H)
    pol.zero_grad["cpu"](None)
    var o = sm_imagine[TC, N](wm, pol, critic, tb, ib, 0.99, True)
    var wj = load_npy_2d_f64(dump + "/i_j.npy", 2).values.copy()
    var ej = abs(o.j - wj[0]) / max(abs(wj[0]), 1e-12)
    t.truth(ej <= 1e-4, "J " + String(o.j) + " (oracle " + String(wj[0]) + "), rel err " + String(ej))
    var eu = abs(o.return_undisc - wj[1]) / max(abs(wj[1]), 1e-12)
    t.truth(eu <= 1e-4, "undiscounted return " + String(o.return_undisc) + ", rel err " + String(eu))
    var qs = List[String]()
    qs.append("states")
    qs.append("actions")
    qs.append("rewards")
    qs.append("dhat")
    qs.append("survival")
    qs.append("e_root")
    qs.append("v_h")
    var cols: List[Int] = [71, 29, 1, 1, 1, 4, 1]
    for k in range(len(qs)):
        var want = load_npy_2d_f64(dump + "/i_" + qs[k] + ".npy", cols[k]).values.copy()
        var r: Tuple[Float64, Float64]
        if qs[k] == "states":
            r = _err(o.states.data, want)
        elif qs[k] == "actions":
            r = _err(o.actions.data, want)
        elif qs[k] == "rewards":
            r = _err(o.rewards.data, want)
        elif qs[k] == "dhat":
            r = _err(o.dhat.data, want)
        elif qs[k] == "survival":
            r = _err(o.survival.data, want)
        elif qs[k] == "e_root":
            r = _err(o.e_root.data, want)
        else:
            r = _err(o.value_h.data, want)
        t.truth(r[0] <= 1e-3 * max(r[1], 1e-3), qs[k] + ": max err " + String(r[0]) + " (scale " + String(r[1]) + ")")

    print("\n[3] the policy gradient")
    var g = _Grads()
    pol.for_each_param["cpu"](g, None)
    var wg = load_npy_2d_f64(dump + "/i_grads.npy", total).values.copy()
    # the policy's slice of the oracle's list: encoder then decoder
    var lo = 0
    var hi = 0
    var off = 0
    var nets = List[String]()
    var bounds = List[Int]()
    for k in range(len(names)):
        if names[k].startswith("encoder.") and lo == 0 and hi == 0:
            lo = off
        if names[k].startswith("decoder."):
            hi = off + sizes[k]
        off += sizes[k]
    t.truth(hi - lo == len(g.out), String(len(g.out)) + " policy params (oracle slice " + String(hi - lo) + ")")
    var enc_end = lo
    for k in range(len(names)):
        if names[k].startswith("encoder."):
            enc_end += sizes[k]
    for part in range(2):
        var a0 = lo if part == 0 else enc_end
        var a1 = enc_end if part == 0 else hi
        var ab = 0.0
        var aa = 0.0
        var bb = 0.0
        var mx = 0.0
        var md = 0.0
        for i in range(a0, a1):
            var x = g.out[i - lo]
            var y = wg[i]
            ab += x * y
            aa += x * x
            bb += y * y
            mx = max(mx, abs(y))
            md = max(md, abs(x - y))
        var cs = ab / sqrt(aa * bb)
        var nm = String("encoder") if part == 0 else String("decoder")
        t.truth(cs >= 0.99999 and md <= 1e-4 * mx, nm + ": cosine " + String(cs) + ", max err " + String(md / mx) + " of the largest (|g|max " + String(mx) + ")")
    _ = nets
    _ = bounds


    print("\n[4] one PolicyTrainer.update, replaying the reference's minibatch draws")
    var tr = SuperMimicTrainer[TC](sm_tables[DT, G1Robot](sim, q0))
    var w0 = load_npy_2d_f64(dump + "/i_weights.npy", total).values.copy()
    var ld2 = _Load(w0.copy(), names.copy(), sizes.copy())
    tr.wm.for_each_param["cpu"](ld2, None)
    tr.pol.for_each_param["cpu"](ld2, None)
    tr.critic.for_each_param["cpu"](ld2, None, String("critic"))
    comptime ROWS = H * N
    var pf = load_npy_2d_f64(dump + "/u_perms.npy", 5 * ROWS).values.copy()
    var perms = List[Int](length=len(pf), fill=0)
    for i in range(len(pf)):
        perms[i] = Int(pf[i])
    var vals = List[Float64]()
    var tgts = List[Float64]()
    var met = tr.policy_update[N](ib, perms, vals, tgts)
    var um = load_npy_2d_f64(dump + "/u_metrics.npy", 6).values.copy()
    var wv = load_npy_2d_f64(dump + "/u_values.npy", 1).values.copy()
    var wt = load_npy_2d_f64(dump + "/u_targets.npy", 1).values.copy()
    var ev = 0.0
    var sv = 0.0
    for i in range(len(wv)):
        ev = max(ev, abs(vals[i] - wv[i]))
        sv = max(sv, abs(wv[i]))
    var et = 0.0
    var st = 0.0
    for i in range(len(wt)):
        et = max(et, abs(tgts[i] - wt[i]))
        st = max(st, abs(wt[i]))
    t.truth(len(vals) == len(wv) and ev <= 1e-5 * sv, String(len(vals)) + " values: max err " + String(ev) + " (scale " + String(sv) + ")")
    t.truth(len(tgts) == len(wt) and et <= 1e-5 * st, String(len(tgts)) + " TD(lambda) targets: max err " + String(et) + " (scale " + String(st) + ")")
    t.truth(abs(met.policy_grad_norm - um[2]) <= 1e-4 * um[2], "policy grad norm " + String(met.policy_grad_norm) + " (oracle " + String(um[2]) + ")")
    t.truth(abs(met.imag_objective - um[0]) <= 1e-4 * abs(um[0]), "objective " + String(met.imag_objective) + " (oracle " + String(um[0]) + ")")
    # the policy after one clipped Adam step
    var pg = _Params()
    tr.pol.for_each_param["cpu"](pg, None)
    var wpol = load_npy_2d_f64(dump + "/u_policy.npy", len(pg.out)).values.copy()
    var clip = min(1.0, 1.0 / (um[2] + 1e-6))
    var e_good = 0.0
    var e_rest = 0.0
    for i in range(len(wpol)):
        var d = abs(pg.out[i] - wpol[i])
        var gi = max(min(wg[lo + i] * clip, 0.1), -0.1)
        if abs(gi) > 1e-6:
            # fp32 ulps of max(|w|, lr): near w = 0 the update's own rounding rules
            e_good = max(e_good, d / (max(abs(wpol[i]), 2e-4) * 1.1920929e-7))
        else:
            e_rest = max(e_rest, d)
    t.truth(e_good <= 2.0 and e_rest <= 2e-5, "policy after its step: " + String(e_good) + " fp32 ulps (well-conditioned), " + String(e_rest) + " abs near eps")
    # the critic after 5 x 4 minibatch steps: compare the MOVES
    var cg = _Params()
    tr.critic.for_each_param["cpu"](cg, None)
    var wcr = load_npy_2d_f64(dump + "/u_critic.npy", len(cg.out)).values.copy()
    var c0 = 0
    for k in range(len(names)):
        if names[k].startswith("critic."):
            break
        c0 += sizes[k]
    # After 20 Adam steps an element whose gradient sits near eps moves almost
    # arbitrarily under fp32 noise (Adam normalises it up to ~lr a step). So:
    # elements that MOVED (≥ 10 % of the largest move) tightly, the rest bounded.
    var mmax = 0.0
    for i in range(len(wcr)):
        mmax = max(mmax, abs(wcr[i] - w0[c0 + i]))
    var d_big = 0.0
    var d_small = 0.0
    var n_big = 0
    for i in range(len(wcr)):
        var mv_ref = wcr[i] - w0[c0 + i]
        var d = abs((cg.out[i] - w0[c0 + i]) - mv_ref)
        if abs(mv_ref) >= 0.1 * mmax:
            d_big = max(d_big, d)
            n_big += 1
        else:
            d_small = max(d_small, d)
    t.truth(d_big <= 1e-3 * mmax, "critic after 20 steps, " + String(n_big) + " elements that moved: agree to " + String(d_big / mmax) + " of the largest move (" + String(mmax) + ")")
    t.truth(d_small <= 0.5 * mmax, String(len(wcr) - n_big) + " elements that barely moved: within " + String(d_small / mmax) + " of the largest move")
    t.truth(abs(met.critic_loss - um[3]) <= 1e-4 * um[3], "last critic minibatch loss " + String(met.critic_loss) + " (oracle " + String(um[3]) + ")")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_supermimic_imag: " + String(t.fails) + " failed")
