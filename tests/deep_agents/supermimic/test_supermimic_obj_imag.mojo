"""The object interface's imagined rollout and its gradient, on the CPU.

`object_imagination.sm_imagine_obj`. The tracker's networks and start batch
are the tracker's own oracle dump (`tools/supermimic/imag_oracle.py`, the
reproduction's weights and library); the object's side is made here.

        uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
            python tools/supermimic/imag_oracle.py --out /tmp/sm_imag
    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_obj_imag.mojo /tmp/sm_imag

[1] the zero start. With the object network and the object encoder as made
    (last layers zero), the ROBOT's rollout is the tracker's `sm_imagine`
    exactly: states, actions, d̂, survival, root error. The reward is the
    tracker's plus the object's three terms. The object moves at constant
    velocity for the whole horizon.
[2] off the zero start (every all-zero parameter filled with small numbers),
    the gradient into the object encoder against central differences of
    loss = −J / H along 7 directions: the gradient's own, then the gradient
    restricted to each of the encoder's 6 parameter tensors, so every layer's
    share is checked where it is largest. Float32 rollouts: J's rounding puts
    a floor of about 1e-7 under a difference quotient, so the band is 5 % of
    the directional derivative plus that floor, and each direction's
    derivative must stand well above it for the check to count. (A random
    direction is useless here: its derivative is |g| / √dim, under the floor.)
[3] what must not get a gradient does not: with gain = 0 the object encoder's
    gradient is exactly zero; a reference object moved 10 m away (every
    kernel in its tail) leaves the robot's rollout unchanged at the zero start.

[4] the independent transcription (`tools/supermimic/obj_imag_oracle.py`:
    torch float64, autograd), run when its dump is the second argument: the
    same three new networks, object inputs and gain; the rollout's states,
    object states, actions, rewards, the object's terms, d̂, survival, V and
    J agree to float32 bands, and the object encoder's gradient agrees with
    autograd's (cosine and largest entry).

        uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
            python tools/supermimic/obj_imag_oracle.py --tracker /tmp/sm_imag --out /tmp/sm_obj_imag
    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_obj_imag.mojo /tmp/sm_imag /tmp/sm_obj_imag
"""

from std.math import sqrt
from std.random import random_float64, seed
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.core.bytes import string_from_bytes
from noeira.deep_agents.supermimic.config import SMConfig, SMCriticOf, SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.imagination import SMImagBatch, sm_imagine
from noeira.deep_agents.supermimic.object_imagination import SMObjImagBatch, sm_imagine_obj
from noeira.deep_agents.supermimic.object_policy import SMObjConfig, SMObjCriticOf, SMObjEncOf, SMObjModelOf
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX
from noeira.deep_agents.supermimic.object_task import SM_OBJ_TERMS, SM_OREF, SM_ORW, sm_obj_reward_params
from noeira.deep_agents.supermimic.robot import G1Robot, G1Sim, SMDims
from noeira.deep_agents.supermimic.state_spec import SM_WM_DT
from noeira.deep_agents.supermimic.tables import sm_tables
from noeira.io.fileio import read_file_bytes
from noeira.io.npy import load_npy_2d_f64
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor


comptime N = 8
comptime H = 32
comptime R = G1Robot
comptime D = SMDims[R]
# the tracker oracle's widths (imag_oracle.py)
comptime TC = SMConfig["cpu", R, 32, 16, 24, 20, 16, 12, 28, 22, 18, 14, 16, 12, N, 64, N, H]
comptime OC = SMObjConfig[TC, 24, 20, 16, 16, 12]
comptime GAIN = 0.1
# [2]'s gain: at 0.1 the encoder's gradient on these small networks is 2e-6, ten times the float32 floor of a
# difference quotient; at 1 it is far enough above it for a 5 % band. The gain is data, the code path is the same.
comptime GAIN_FD = 1.0
# [4]'s bands on the gradient. Measured agreement (2026-11-12): cosine 1 − 2e-14, largest error 3e-7 of the largest
# entry. The bands sit about 60 times above that and NOT at a "float32-sized" 1e-3: with 1e-6 / 2e-3, three planted
# bugs on the paths through the object's own state passed (their share of this gradient is small).
comptime GRAD_COS = 1e-10
comptime GRAD_MAX = 2e-5


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
            self.bad += 1
        for i in range(N_):
            param.data[i] = Scalar[DT](self.vals[self.at + i])
        self.at += N_
        self.k += 1


struct _LoadFlat(ParamVisitor):
    """Parameters from a flat list, in walk order; sizes checked."""
    var vals: List[Float64]
    var sizes: List[Int]
    var at: Int
    var k: Int
    var bad: Int

    def __init__(out self, var vals: List[Float64], var sizes: List[Int], at: Int, k: Int):
        self.vals = vals^
        self.sizes = sizes^
        self.at = at
        self.k = k
        self.bad = 0

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^
        self.sizes = move.sizes^
        self.at = move.at
        self.k = move.k
        self.bad = move.bad

    def visit[target: StaticString, N_: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        if self.k >= len(self.sizes) or self.sizes[self.k] != N_:
            self.bad += 1
            return
        for i in range(N_):
            param.data[i] = Scalar[DT](self.vals[self.at + i])
        self.at += N_
        self.k += 1


def _band(got: List[Scalar[DT]], want: List[Float64]) -> Tuple[Float64, Float64]:
    """(max |got − want|, max |want|)."""
    var e = 0.0
    var sc = 0.0
    if len(got) != len(want):
        return (1e30, 0.0)
    for i in range(len(want)):
        e = max(e, abs(Float64(got[i]) - want[i]))
        sc = max(sc, abs(want[i]))
    return (e, sc)


struct _FillZeros(ParamVisitor):
    """Every all-zero parameter (a zero-started layer) gets U(−a, a)."""
    var a: Float64
    var filled: Int

    def __init__(out self, a: Float64):
        self.a = a
        self.filled = 0

    def __init__(out self, *, deinit move: Self):
        self.a = move.a
        self.filled = move.filled

    def visit[target: StaticString, N_: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        var any = False
        for i in range(N_):
            if param.data[i] != 0:
                any = True
        if not any:
            for i in range(N_):
                param.data[i] = Scalar[DT](random_float64(-self.a, self.a))
            self.filled += 1


struct _Grads(ParamVisitor):
    var out: List[Float64]
    var sizes: List[Int]

    def __init__(out self):
        self.out = List[Float64]()
        self.sizes = List[Int]()

    def __init__(out self, *, deinit move: Self):
        self.out = move.out^
        self.sizes = move.sizes^

    def visit[target: StaticString, N_: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N_):
            self.out.append(Float64(grad.data[i]))
        self.sizes.append(N_)


struct _Shift(ParamVisitor):
    """params += c · d (d flat, in walk order)."""
    var d: List[Float64]
    var c: Float64
    var at: Int

    def __init__(out self, var d: List[Float64], c: Float64):
        self.d = d^
        self.c = c
        self.at = 0

    def __init__(out self, *, deinit move: Self):
        self.d = move.d^
        self.c = move.c
        self.at = move.at

    def visit[target: StaticString, N_: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N_):
            param.data[i] = Scalar[DT](Float64(param.data[i]) + self.c * self.d[self.at + i])
        self.at += N_


def _fill(mut dst: List[Scalar[DT]], src: List[Float64]) raises:
    if len(dst) != len(src):
        raise Error("size " + String(len(src)) + " for a buffer of " + String(len(dst)))
    for i in range(len(src)):
        dst[i] = Scalar[DT](src[i])


def _max_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Float64:
    var e = 0.0
    for i in range(len(a)):
        e = max(e, abs(Float64(a[i]) - Float64(b[i])))
    return e


def _tracker_batch(dump: String) raises -> SMImagBatch[R]:
    var ib = SMImagBatch[R](N, H)
    _fill(ib.state.data, load_npy_2d_f64(dump + "/i_state.npy", D.S).values)
    _fill(ib.prev_action.data, load_npy_2d_f64(dump + "/i_prev_action.npy", R.NDOF).values)
    _fill(ib.xi.data, load_npy_2d_f64(dump + "/i_xi.npy", D.X).values)
    _fill(ib.hist0.data, load_npy_2d_f64(dump + "/i_hist0.npy", D.OBS).values)
    _fill(ib.goals.data, load_npy_2d_f64(dump + "/i_goals.npy", D.GOAL).values)
    _fill(ib.refs.data, load_npy_2d_f64(dump + "/i_refs.npy", D.REF).values)
    return ib^


def _unit4(mut v: List[Scalar[DT]], off: Int):
    var n = 0.0
    for k in range(4):
        n += Float64(v[off + k]) * Float64(v[off + k])
    n = sqrt(n)
    for k in range(4):
        v[off + k] = Scalar[DT](Float64(v[off + k]) / n)


def _object_batch(dump: String, far: Float64) raises -> SMObjImagBatch[R]:
    """The tracker's batch, an object 0.4 m in front of each root, moving, and
    references a few cm from it (+ `far` m along x). The reward's kernels are
    wide (0.8 m, 2 rad, 1.2 m), so no lane sits in a tail."""
    var ob = SMObjImagBatch[R](N, H)
    ob.base = _tracker_batch(dump)
    seed(7)
    for b in range(N):
        for k in range(3):
            ob.obj0.data[b * SM_OBJ + k] = ob.base.state.data[b * D.S + k] + Scalar[DT](random_float64(-0.2, 0.2))
        ob.obj0.data[b * SM_OBJ] += Scalar[DT](0.4)
        for k in range(4):
            ob.obj0.data[b * SM_OBJ + 3 + k] = Scalar[DT](random_float64(-1.0, 1.0))
        _unit4(ob.obj0.data, b * SM_OBJ + 3)
        for k in range(6):
            ob.obj0.data[b * SM_OBJ + 7 + k] = Scalar[DT](random_float64(-0.3, 0.3))
        ob.octx.data[b * SM_OBJ_CTX] = Scalar[DT](random_float64(0.2, 1.9))
        ob.octx.data[b * SM_OBJ_CTX + 1] = Scalar[DT](random_float64(0.5, 1.3))
    for t in range(H + 1):
        for b in range(N):
            var r = (t * N + b) * SM_OREF
            for k in range(3):
                ob.orefs.data[r + k] = ob.obj0.data[b * SM_OBJ + k] + Scalar[DT](random_float64(-0.1, 0.1) + 0.004 * Float64(t))
            ob.orefs.data[r] += Scalar[DT](far)
            for k in range(4):
                ob.orefs.data[r + 3 + k] = ob.obj0.data[b * SM_OBJ + 3 + k] + Scalar[DT](random_float64(-0.2, 0.2))
            _unit4(ob.orefs.data, r + 3)
            for k in range(6):
                ob.orefs.data[r + 7 + k] = Scalar[DT](random_float64(-0.3, 0.3))
            for k in range(6):
                ob.orefs.data[r + SM_OBJ + k] = Scalar[DT](random_float64(-0.3, 0.3))
    var rw = sm_obj_reward_params(1.0, 0.8, 0.5, 2.0, 1.0, 1.2)
    for k in range(SM_ORW):
        ob.rw.data[k] = rw[k]
    return ob^


def main() raises:
    var av = argv()
    var dump = String(av[1]) if len(av) > 1 else String("/tmp/sm_imag")
    var t = Tally()
    var sim = G1Sim.create()

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
    var tcritic = SMCriticOf[TC].make["cpu", Kaiming](None)
    var ld = _Load(load_npy_2d_f64(dump + "/i_weights.npy", total).values.copy(), names.copy(), sizes.copy())
    wm.for_each_param["cpu"](ld, None)
    pol.for_each_param["cpu"](ld, None)
    tcritic.for_each_param["cpu"](ld, None, String("critic"))
    if ld.bad != 0 or ld.at != total:
        raise Error("the tracker's dump does not match its networks: regenerate " + dump)
    var q0 = load_npy_2d_f64(dump + "/i_q0.npy", R.NDOF).values.copy()
    var tb = sm_tables[DT, R](sim, q0)
    seed(11)
    var om = SMObjModelOf[OC].make[Kaiming]()
    var enc = SMObjEncOf[OC].make["cpu", Kaiming](None)
    var critic = SMObjCriticOf[OC].make["cpu", Kaiming](None)

    # ── [1] ──
    print("\n[1] the zero start: the tracker's rollout, an object at constant velocity")
    var tib = _tracker_batch(dump)
    var o0 = sm_imagine[TC, N](wm, pol, tcritic, tb, tib, 0.99, False)
    var ob = _object_batch(dump, 0.0)
    var o1 = sm_imagine_obj[OC, N](wm, pol, om, enc, critic, tb, ob, 0.99, GAIN, False)
    var es = max(_max_diff(o1.base.states.data, o0.states.data), _max_diff(o1.base.actions.data, o0.actions.data))
    var ec = max(
        _max_diff(o1.base.dhat.data, o0.dhat.data),
        max(_max_diff(o1.base.survival.data, o0.survival.data), _max_diff(o1.base.e_root.data, o0.e_root.data)),
    )
    t.truth(es == 0.0 and ec == 0.0, "states, actions, d̂, survival, root error are sm_imagine's: max |d| " + String(max(es, ec)))
    var er = 0.0
    var osum = 0.0
    for i in range(H * N):
        var ro = 0.0
        for k in range(SM_OBJ_TERMS):
            ro += Float64(o1.oterms.data[i * SM_OBJ_TERMS + k])
        osum += ro
        er = max(er, abs(Float64(o1.base.rewards.data[i]) - (Float64(o0.rewards.data[i]) + ro)))
    t.truth(er <= 1e-7 and osum > 1e-2, "reward = the tracker's + the object's three terms: max |d| " + String(er) + " (object terms sum " + String(osum) + ")")
    var ev = 0.0
    var ep = 0.0
    for b in range(N):
        for k in range(6):
            ev = max(ev, abs(Float64(o1.objs.data[(H * N + b) * SM_OBJ + 7 + k]) - Float64(ob.obj0.data[b * SM_OBJ + 7 + k])))
        for k in range(3):
            var want = Float64(ob.obj0.data[b * SM_OBJ + k]) + Float64(H) * SM_WM_DT * Float64(ob.obj0.data[b * SM_OBJ + 7 + k])
            ep = max(ep, abs(Float64(o1.objs.data[(H * N + b) * SM_OBJ + k]) - want))
    t.truth(ev == 0.0 and ep <= 1e-5, "the object keeps its velocity (max |d| " + String(ev) + ") and ends at p₀ + H·dt·v₀ (max |d| " + String(ep) + ")")

    # ── [2] ──
    print("\n[2] the object encoder's gradient against central differences of −J / H")
    seed(12)
    # small: a larger object-network output throws the robot (survival → 0) and J stops depending on the policy
    var fz = _FillZeros(0.003)
    om.for_each_param["cpu"](fz, None)
    var fe = _FillZeros(0.3)
    enc.for_each_param["cpu"](fe, None)
    t.truth(fz.filled >= 2 and fe.filled >= 2, "zero parameters filled: " + String(fz.filled) + " of the object network, " + String(fe.filled) + " of the encoder")
    enc.zero_grad["cpu"](None)
    var o2 = sm_imagine_obj[OC, N](wm, pol, om, enc, critic, tb, ob, 0.99, GAIN_FD, True)
    var away = _max_diff(o2.base.actions.data, o0.actions.data)
    var surv = 0.0
    for b in range(N):
        surv += Float64(o2.base.survival.data[H * N + b]) / Float64(N)
    var gv = _Grads()
    enc.for_each_param["cpu"](gv, None)
    var g = gv.out.copy()
    var gn = 0.0
    for i in range(len(g)):
        gn += g[i] * g[i]
    gn = sqrt(gn)
    t.truth(
        away > 1e-3 and gn > 1e-5 and len(gv.sizes) == 6,
        "off the zero start: actions differ from the tracker's by " + String(away) + ", survival at H " + String(surv) + ", |g| = " + String(gn),
    )
    comptime FLOOR = 2e-7
    var worst = 0.0
    var weakest = 1e9
    var lo = 0
    for trial in range(len(gv.sizes) + 1):
        var hi = len(g)
        if trial > 0:
            hi = lo + gv.sizes[trial - 1]
        var d = List[Float64](length=len(g), fill=0.0)
        var dn = 0.0
        for i in range(lo, hi):
            d[i] = g[i]
            dn += g[i] * g[i]
        dn = sqrt(dn)
        for i in range(lo, hi):
            d[i] = d[i] / dn
        var want = dn                     # <g, d> = |g restricted|
        var eps = 0.02
        var up = _Shift(d.copy(), eps)
        enc.for_each_param["cpu"](up, None)
        var jp = sm_imagine_obj[OC, N](wm, pol, om, enc, critic, tb, ob, 0.99, GAIN_FD, False).base.j
        var dn2 = _Shift(d.copy(), -2.0 * eps)
        enc.for_each_param["cpu"](dn2, None)
        var jm = sm_imagine_obj[OC, N](wm, pol, om, enc, critic, tb, ob, 0.99, GAIN_FD, False).base.j
        var back = _Shift(d.copy(), eps)
        enc.for_each_param["cpu"](back, None)
        var fd = -(jp - jm) / (2.0 * eps * Float64(H))
        var excess = max(abs(fd - want) - FLOOR, 0.0) / want
        worst = max(worst, excess)
        weakest = min(weakest, want)
        print("    direction", trial, "(", hi - lo, "params ): FD", fd, "analytic", want, "| excess over the floor, relative:", excess)
        if trial > 0:
            lo = hi
        else:
            lo = 0
    t.truth(
        worst <= 0.05 and weakest > 10.0 * FLOOR,
        "7 directions: worst excess " + String(worst) + " of the derivative; the weakest derivative is " + String(weakest),
    )

    # ── [3] ──
    print("\n[3] what must not get a gradient")
    enc.zero_grad["cpu"](None)
    _ = sm_imagine_obj[OC, N](wm, pol, om, enc, critic, tb, ob, 0.99, 0.0, True)
    var g0 = _Grads()
    enc.for_each_param["cpu"](g0, None)
    var gmax = 0.0
    for i in range(len(g0.out)):
        gmax = max(gmax, abs(g0.out[i]))
    t.truth(gmax == 0.0, "gain 0: the object encoder's gradient is exactly zero (max " + String(gmax) + ")")
    seed(11)
    var om_z = SMObjModelOf[OC].make[Kaiming]()
    var enc_z = SMObjEncOf[OC].make["cpu", Kaiming](None)
    var obf = _object_batch(dump, 10.0)
    var o3 = sm_imagine_obj[OC, N](wm, pol, om_z, enc_z, critic, tb, obf, 0.99, GAIN, False)
    var tail = 0.0
    for i in range(H * N):
        tail = max(tail, abs(Float64(o3.oterms.data[i * SM_OBJ_TERMS])))
    t.truth(
        _max_diff(o3.base.states.data, o0.states.data) == 0.0 and tail < 1e-30,
        "a reference 10 m away at the zero start: the robot's rollout is unchanged, the position term is " + String(tail),
    )

    # ── [4] ──
    if len(av) > 2:
        var od = String(av[2])
        print("\n[4] the torch transcription (", od, ")")
        var osz = List[Int]()
        for l in string_from_bytes(read_file_bytes(od + "/o_sizes.txt")).split("\n"):
            var ls = String(l)
            if ls.byte_length() == 0:
                continue
            osz.append(Int(String(ls.split("\t")[1])))
        var ototal = 0
        for k in range(len(osz)):
            ototal += osz[k]
        var wflat = load_npy_2d_f64(od + "/o_weights.npy", ototal).values.copy()
        var om4 = SMObjModelOf[OC].make[Kaiming]()
        var enc4 = SMObjEncOf[OC].make["cpu", Kaiming](None)
        var critic4 = SMObjCriticOf[OC].make["cpu", Kaiming](None)
        var l1 = _LoadFlat(wflat.copy(), osz.copy(), 0, 0)
        om4.for_each_param["cpu"](l1, None)
        var l2 = _LoadFlat(wflat.copy(), osz.copy(), l1.at, l1.k)
        enc4.for_each_param["cpu"](l2, None)
        var l3 = _LoadFlat(wflat.copy(), osz.copy(), l2.at, l2.k)
        critic4.for_each_param["cpu"](l3, None)
        t.truth(
            l1.bad + l2.bad + l3.bad == 0 and l3.at == ototal and l3.k == len(osz),
            String(l3.k) + " parameter tensors of the three new networks by order and size, " + String(ototal) + " values",
        )
        var ob4 = SMObjImagBatch[R](N, H)
        ob4.base = _tracker_batch(dump)
        _fill(ob4.obj0.data, load_npy_2d_f64(od + "/o_obj0.npy", SM_OBJ).values)
        _fill(ob4.octx.data, load_npy_2d_f64(od + "/o_octx.npy", SM_OBJ_CTX).values)
        _fill(ob4.orefs.data, load_npy_2d_f64(od + "/o_orefs.npy", SM_OREF).values)
        _fill(ob4.rw.data, load_npy_2d_f64(od + "/o_rw.npy", SM_ORW).values)
        var meta = load_npy_2d_f64(od + "/o_meta.npy", 2).values.copy()
        enc4.zero_grad["cpu"](None)
        var o4 = sm_imagine_obj[OC, N](wm, pol, om4, enc4, critic4, tb, ob4, meta[1], meta[0], True)
        var wj = load_npy_2d_f64(od + "/o_j.npy", 2).values.copy()
        var ej = abs(o4.base.j - wj[0]) / max(abs(wj[0]), 1e-12)
        t.truth(ej <= 1e-4, "J " + String(o4.base.j) + " (torch " + String(wj[0]) + "), rel err " + String(ej))
        var r_s = _band(o4.base.states.data, load_npy_2d_f64(od + "/o_states.npy", D.S).values)
        t.truth(r_s[0] <= 1e-3 * max(r_s[1], 1e-3), "states: max err " + String(r_s[0]) + " (scale " + String(r_s[1]) + ")")
        var r_o = _band(o4.objs.data, load_npy_2d_f64(od + "/o_objs.npy", SM_OBJ).values)
        t.truth(r_o[0] <= 1e-3 * max(r_o[1], 1e-3), "object states: max err " + String(r_o[0]) + " (scale " + String(r_o[1]) + ")")
        var r_a = _band(o4.base.actions.data, load_npy_2d_f64(od + "/o_actions.npy", R.NDOF).values)
        t.truth(r_a[0] <= 1e-3 * max(r_a[1], 1e-3), "actions: max err " + String(r_a[0]) + " (scale " + String(r_a[1]) + ")")
        var r_r = _band(o4.base.rewards.data, load_npy_2d_f64(od + "/o_rewards.npy", 1).values)
        t.truth(r_r[0] <= 1e-3 * max(r_r[1], 1e-3), "rewards: max err " + String(r_r[0]) + " (scale " + String(r_r[1]) + ")")
        var r_t = _band(o4.oterms.data, load_npy_2d_f64(od + "/o_oterms.npy", SM_OBJ_TERMS).values)
        t.truth(r_t[0] <= 1e-3 * max(r_t[1], 1e-3), "the object's three terms: max err " + String(r_t[0]) + " (scale " + String(r_t[1]) + ")")
        var r_d = _band(o4.base.dhat.data, load_npy_2d_f64(od + "/o_dhat.npy", 1).values)
        var r_c = _band(o4.base.survival.data, load_npy_2d_f64(od + "/o_survival.npy", 1).values)
        var r_v = _band(o4.base.value_h.data, load_npy_2d_f64(od + "/o_v_h.npy", 1).values)
        t.truth(
            r_d[0] <= 1e-4 and r_c[0] <= 1e-4 and r_v[0] <= 1e-3 * max(r_v[1], 1e-3),
            "d̂ " + String(r_d[0]) + ", survival " + String(r_c[0]) + ", V(s_H, o_H) " + String(r_v[0]) + " (scale " + String(r_v[1]) + ")",
        )
        var g4 = _Grads()
        enc4.for_each_param["cpu"](g4, None)
        var wg = load_npy_2d_f64(od + "/o_grads.npy", len(g4.out)).values.copy()
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        var emax = 0.0
        var gbig = 0.0
        for i in range(len(wg)):
            dot += g4.out[i] * wg[i]
            na += g4.out[i] * g4.out[i]
            nb += wg[i] * wg[i]
            emax = max(emax, abs(g4.out[i] - wg[i]))
            gbig = max(gbig, abs(wg[i]))
        var cosv = dot / max(sqrt(na) * sqrt(nb), 1e-300)
        t.truth(
            cosv >= 1.0 - GRAD_COS and emax <= GRAD_MAX * gbig and sqrt(nb) > 1e-6,
            "the object encoder's gradient vs autograd: " + String(len(wg)) + " values, cosine " + String(cosv) + ", max err " + String(emax / gbig)
            + " of the largest (" + String(gbig) + "), |g| " + String(sqrt(nb)),
        )
        # per parameter tensor: every layer's share on its own
        var lo4 = 0
        var worst_layer = 1.0
        for k in range(len(g4.sizes)):
            var d2 = 0.0
            var a2 = 0.0
            var b2 = 0.0
            for i in range(lo4, lo4 + g4.sizes[k]):
                d2 += g4.out[i] * wg[i]
                a2 += g4.out[i] * g4.out[i]
                b2 += wg[i] * wg[i]
            worst_layer = min(worst_layer, d2 / max(sqrt(a2) * sqrt(b2), 1e-300))
            lo4 += g4.sizes[k]
        t.truth(worst_layer >= 1.0 - 10.0 * GRAD_COS, "per parameter tensor (6): worst cosine " + String(worst_layer))
    else:
        print("\n[4] SKIPPED: no torch dump given (second argument)")

    print("\n=== " + String(t.checks - t.fails) + " / " + String(t.checks) + " passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_obj_imag: " + String(t.fails) + " failed")
