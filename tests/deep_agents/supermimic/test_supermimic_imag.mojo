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
fail all four.
"""

from std.math import sqrt
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.core.bytes import string_from_bytes
from noeira.deep_agents.supermimic.g1_fk import g1_fk_tree
from noeira.deep_agents.supermimic.heads import SM_REF_DIM, g1_heads_table
from noeira.deep_agents.supermimic.imagination import SMImagBatch, sm_imagine
from noeira.deep_agents.supermimic.policy import SMCritic, SMPolicy, SM_GOAL_DIM, SM_OBS
from noeira.deep_agents.supermimic.world_model import SMWorldModel
from noeira.envs.robots import UnitreeG1
from noeira.io.fileio import read_file_bytes
from noeira.io.npy import load_npy_2d_f64
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor


comptime N = 8
comptime H = 32


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
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()

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
    var wm = SMWorldModel[32, 16].make[Kaiming]()
    var pol = SMPolicy[24, 20, 16, 12, 28, 22, 18, 14].make[Kaiming]()
    var critic = SMCritic[16, 12].make["cpu", Kaiming](None)
    var ld = _Load(load_npy_2d_f64(dump + "/i_weights.npy", total).values.copy(), names.copy(), sizes.copy())
    wm.for_each_param["cpu"](ld, None)
    pol.for_each_param["cpu"](ld, None)
    critic.for_each_param["cpu"](ld, None, String("critic"))
    t.truth(ld.bad == 0 and ld.k == len(names) and ld.at == total, String(ld.k) + " params by name, " + String(total) + " values")

    var q0 = load_npy_2d_f64(dump + "/i_q0.npy", 29).values.copy()
    var tree = g1_fk_tree[DT](env)
    var table = g1_heads_table[DT](env, q0)
    var ib = SMImagBatch(N, H)
    _fill(ib.state, load_npy_2d_f64(dump + "/i_state.npy", 71).values)
    _fill(ib.prev_action, load_npy_2d_f64(dump + "/i_prev_action.npy", 29).values)
    _fill(ib.xi, load_npy_2d_f64(dump + "/i_xi.npy", 33).values)
    _fill(ib.hist0, load_npy_2d_f64(dump + "/i_hist0.npy", SM_OBS).values)
    _fill(ib.goals, load_npy_2d_f64(dump + "/i_goals.npy", SM_GOAL_DIM).values)
    _fill(ib.refs, load_npy_2d_f64(dump + "/i_refs.npy", SM_REF_DIM).values)

    print("\n[2] the forward rollout, H =", H)
    pol.zero_grad["cpu"](None)
    var o = sm_imagine[32, 16, 24, 20, 16, 12, 28, 22, 18, 14, 16, 12, N](wm, pol, critic, tree, table, ib, 0.99, True)
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
            r = _err(o.states, want)
        elif qs[k] == "actions":
            r = _err(o.actions, want)
        elif qs[k] == "rewards":
            r = _err(o.rewards, want)
        elif qs[k] == "dhat":
            r = _err(o.dhat, want)
        elif qs[k] == "survival":
            r = _err(o.survival, want)
        elif qs[k] == "e_root":
            r = _err(o.e_root, want)
        else:
            r = _err(o.value_h, want)
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

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_supermimic_imag: " + String(t.fails) + " failed")
