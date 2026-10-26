"""`FBRegHead` smoke gate — FB-MEBE's regularity critic enters where it should
and NOWHERE else. CPU, random data. `noeira-docs/FB_MEBE_G1_PLAN.md` §2 B3.

  [1] a step runs; every reported term is finite; the head's reward mean is
      the mean of the `br` it was handed.
  [2] ⚠ the load-bearing one: with `reg_coeff = 0` the four FB nets (B, F1,
      F2, actor) are BIT-IDENTICAL to a plain `FBTrainer` stepped on the same
      batch — a FB-MEBE arm with λ = 0 IS the FB arm.
  [3] with `reg_coeff > 0` the ACTOR moves differently while B, F1 and F2
      stay bit-identical: the term reaches the actor and only the actor.
  [4] `raw` reaches the trainer: the raw (release) and the |Q_fb|-scaled
      (BFM-Zero) forms move the actor differently from the same start.
  [5] the critic is a critic: on a frozen batch with constant reward r and
      γ = 0.5, hard target updates, `Q → r / (1 − γ)` = −2 within 0.1 — a
      head whose reward never reaches the target, or whose bootstrap reads
      the wrong net, does not converge there.
  [6] the `.reg` sidecar round-trips (Q on a probe equal after load); a
      missing sidecar RAISES.

Run: pixi run mojo run -I . tests/deep_agents/test_fb_reg_head.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true
from std.os import remove
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import Tanh, ReLU
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.reg_head import FBRegHead
from noeira.deep_agents.fb import sample_z_uniform


comptime OBS: Int = 4
comptime ACT: Int = 2
comptime D: Int = 6
comptime BATCH: Int = 16
comptime HID: Int = 32
comptime SEED: Int = 20261012

comptime F_IN = OBS + ACT + D
comptime A_IN = OBS + D

comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]
comptime QNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, 1]]

comptime Plain = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH]
comptime Head = FBRegHead[FNet, BNet, ANet, QNet, OBS, ACT, D, BATCH]


struct _ReadVals(ParamVisitor):
    var vals: List[List[Scalar[DT]]]

    def __init__(out self):
        self.vals = List[List[Scalar[DT]]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var g = List[Scalar[DT]](capacity=N)
        for i in range(N):
            g.append(param.data[i])
        self.vals.append(g^)


def _same(ref a: _ReadVals, ref b: _ReadVals) raises -> Bool:
    assert_true(len(a.vals) == len(b.vals), "param walks differ in length")
    for k in range(len(a.vals)):
        assert_true(len(a.vals[k]) == len(b.vals[k]), "param sizes differ")
        for j in range(len(a.vals[k])):
            if a.vals[k][j] != b.vals[k][j]:
                return False
    return True


def _rand_tensor(n: Int, scale: Float64) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT]((random_float64() * 2.0 - 1.0) * scale)
    return t^


def _clone(ref t: Tensor, n: Int) raises -> Tensor:
    var c = Tensor.alloc(n)
    for i in range(n):
        c.data[i] = t.data[i]
    return c^


def _z_tensor(batch: Int) raises -> Tensor:
    var z = sample_z_uniform[D](batch)
    var t = Tensor.alloc(batch * D)
    for i in range(batch * D):
        t.data[i] = z[i]
    return t^


struct _Batch(Movable & Deinitable):
    var s: Tensor
    var a: Tensor
    var sn: Tensor
    var sp: Tensor
    var z: Tensor
    var r: Tensor

    def __init__(out self) raises:
        self.s = _rand_tensor(BATCH * OBS, 1.0)
        self.a = _rand_tensor(BATCH * ACT, 1.0)
        self.sn = _rand_tensor(BATCH * OBS, 1.0)
        self.sp = _rand_tensor(BATCH * OBS, 1.0)
        self.z = _z_tensor(BATCH)
        self.r = _rand_tensor(BATCH, 0.5)

    def __init__(out self, *, deinit move: Self):
        self.s = move.s^
        self.a = move.a^
        self.sn = move.sn^
        self.sp = move.sp^
        self.z = move.z^
        self.r = move.r^


def _make_t() raises -> Plain:
    seed(SEED)
    var t = Plain.make(
        lr=1e-3, gamma=0.98, tau=0.01, ortho_weight=1.0, ctx=None,
        seed=UInt64(SEED), max_grad_norm=0.0, bc_weight=0.0, lr_b=1e-3,
        act_l2_weight=0.0, act_l2_margin=0.0,
    )
    # ⚠ smoothing OFF for the bit-identity legs (as in test_fb_cpr_smoke):
    # the head draws its own noise; σ = 0 makes every draw multiply to 0
    t.policy_noise = 0.0
    return t^


def _make_head(reg: Float64, raw: Bool) raises -> Head:
    var h = Head.make(
        None, lr_q=1e-3, gamma=0.98, tau_q=0.005, max_grad_norm=0.0,
        reg_coeff=reg, raw=raw, policy_noise=0.0, seed=UInt64(SEED),
    )
    return h^


def _load(mut t: Plain, ref b: _Batch) raises:
    var s = _clone(b.s, BATCH * OBS)
    var a = _clone(b.a, BATCH * ACT)
    var sn = _clone(b.sn, BATCH * OBS)
    var sp = _clone(b.sp, BATCH * OBS)
    var z = _clone(b.z, BATCH * D)
    t.load_batch(s, a, sn, sp, z)


def _load_r(mut h: Head, ref b: _Batch, mut t: Plain) raises:
    h.ensure_sized(t)
    for i in range(BATCH):
        h.br.data[i] = b.r.data[i]


def _read_fb(
    mut t: Plain, mut rb: _ReadVals, mut rf1: _ReadVals, mut rf2: _ReadVals,
    mut ra: _ReadVals,
) raises:
    t.bnet.online.for_each_param["cpu"](rb, None)
    t.f1.online.for_each_param["cpu"](rf1, None)
    t.f2.online.for_each_param["cpu"](rf2, None)
    t.actor.online.for_each_param["cpu"](ra, None)


def test_step_runs() raises:
    print("[1] a step runs; reward mean is the batch's ...")
    seed(SEED + 1)
    var b = _Batch()
    var t = _make_t()
    var h = _make_head(20.0, True)
    _load(t, b)
    _load_r(h, b, t)
    var l = h.step(t)
    var rm = Float64(0)
    var qm = Float64(0)
    var ql = Float64(0)
    var qp = Float64(0)
    h.read_diag(rm, qm, ql, qp)
    var want = Float64(0)
    for i in range(BATCH):
        want += Float64(b.r.data[i])
    want /= Float64(BATCH)
    print("      measure", l.measure, " actor", l.actor, " r", rm, "(want", want,
          ") Q", qm, " Qloss", ql, " Qpi", qp)
    assert_true(abs(rm - want) < 1e-5, "head's reward mean is not the batch's")
    for x in [l.measure, l.actor, qm, ql, qp]:
        assert_true(x == x and abs(x) < 1e12, "non-finite term")


def _run(reg: Float64, raw: Bool, steps: Int, ref b: _Batch, mut out_t: Plain) raises:
    var h = _make_head(reg, raw)
    _load_r(h, b, out_t)
    for _ in range(steps):
        _load(out_t, b)
        _ = h.step(out_t)


def test_reg0_bit_identical() raises:
    print("[2] reg 0 -> B, F1, F2, actor bit-identical to plain FB ...")
    seed(SEED + 2)
    var b = _Batch()
    var tp = _make_t()
    for _ in range(3):
        _load(tp, b)
        _ = tp.train_step()
    var th = _make_t()
    _run(0.0, True, 3, b, th)
    var pb = _ReadVals(); var pf1 = _ReadVals(); var pf2 = _ReadVals(); var pa = _ReadVals()
    var hb = _ReadVals(); var hf1 = _ReadVals(); var hf2 = _ReadVals(); var ha = _ReadVals()
    _read_fb(tp, pb, pf1, pf2, pa)
    _read_fb(th, hb, hf1, hf2, ha)
    var ok = _same(pb, hb) and _same(pf1, hf1) and _same(pf2, hf2) and _same(pa, ha)
    print("      identical:", ok)
    assert_true(ok, "reg 0 changed the FB nets")


def test_reg_moves_actor_only() raises:
    print("[3] reg > 0 -> actor differs; B, F1, F2 identical ...")
    seed(SEED + 3)
    var b = _Batch()
    var t0 = _make_t()
    _run(0.0, True, 3, b, t0)
    var t1 = _make_t()
    _run(20.0, True, 3, b, t1)
    var b0 = _ReadVals(); var f10 = _ReadVals(); var f20 = _ReadVals(); var a0 = _ReadVals()
    var b1 = _ReadVals(); var f11 = _ReadVals(); var f21 = _ReadVals(); var a1 = _ReadVals()
    _read_fb(t0, b0, f10, f20, a0)
    _read_fb(t1, b1, f11, f21, a1)
    # ⚠ B, F1, F2 are stepped BEFORE the actor in each step, but from step 2
    # on they read actions of an actor that has diverged — so only step 1's
    # B/F are comparable. Re-run one step for that half.
    var u0 = _make_t()
    _run(0.0, True, 1, b, u0)
    var u1 = _make_t()
    _run(20.0, True, 1, b, u1)
    var ub0 = _ReadVals(); var uf10 = _ReadVals(); var uf20 = _ReadVals(); var ua0 = _ReadVals()
    var ub1 = _ReadVals(); var uf11 = _ReadVals(); var uf21 = _ReadVals(); var ua1 = _ReadVals()
    _read_fb(u0, ub0, uf10, uf20, ua0)
    _read_fb(u1, ub1, uf11, uf21, ua1)
    var fb_same = _same(ub0, ub1) and _same(uf10, uf11) and _same(uf20, uf21)
    var actor_same = _same(ua0, ua1)
    print("      one step: B/F identical", fb_same, "  actor identical", actor_same)
    assert_true(fb_same, "the reg term reached B or F")
    assert_true(not actor_same, "the reg term did not reach the actor")
    assert_true(not _same(a0, a1), "three steps: actors identical")


def test_raw_flag_reaches_trainer() raises:
    print("[4] raw vs |Q_fb|-scaled move the actor differently ...")
    seed(SEED + 4)
    var b = _Batch()
    var tr = _make_t()
    _run(20.0, True, 1, b, tr)
    var ts = _make_t()
    _run(20.0, False, 1, b, ts)
    var x1 = _ReadVals(); var x2 = _ReadVals(); var x3 = _ReadVals(); var ar = _ReadVals()
    var y1 = _ReadVals(); var y2 = _ReadVals(); var y3 = _ReadVals(); var as_ = _ReadVals()
    _read_fb(tr, x1, x2, x3, ar)
    _read_fb(ts, y1, y2, y3, as_)
    print("      actors identical:", _same(ar, as_))
    assert_true(not _same(ar, as_), "pi_extra_raw had no effect")


def test_critic_converges() raises:
    print("[5] constant r = -1, γ 0.5, hard targets -> Q ≈ -2 ...")
    seed(SEED + 5)
    var b = _Batch()
    for i in range(BATCH):
        b.r.data[i] = Scalar[DT](-1.0)
    var t = _make_t()
    var h = Head.make(
        None, lr_q=3e-3, gamma=0.5, tau_q=1.0, max_grad_norm=0.0,
        reg_coeff=0.0, raw=True, policy_noise=0.0, seed=UInt64(SEED),
    )
    _load_r(h, b, t)
    var qm = Float64(0)
    for k in range(600):
        _load(t, b)
        _ = h.step(t)
        if k % 200 == 199:
            var rm = Float64(0); var ql = Float64(0); var qp = Float64(0)
            h.read_diag(rm, qm, ql, qp)
            print("      step", k + 1, " Q", qm, " Qloss", ql)
    assert_true(abs(qm - (-2.0)) < 0.1, "Q did not converge to r/(1-γ)")


def test_sidecar() raises:
    print("[6] .reg sidecar round trip; missing sidecar raises ...")
    seed(SEED + 6)
    var b = _Batch()
    var t = _make_t()
    var h = _make_head(20.0, True)
    _load_r(h, b, t)
    for _ in range(3):
        _load(t, b)
        _ = h.step(t)
    var path = String("/tmp/_fb_reg_head_test.ckpt")
    h.save_sidecar(path)
    var t2 = _make_t()
    var h2 = _make_head(20.0, True)
    h2.ensure_sized(t2)
    h2.load_sidecar(path)
    var r1 = _ReadVals()
    var r2 = _ReadVals()
    h.q1n.online.for_each_param["cpu"](r1, None)
    h2.q1n.online.for_each_param["cpu"](r2, None)
    var q1 = _ReadVals()
    var q2 = _ReadVals()
    h2.q1n.target_net.for_each_param["cpu"](q2, None)
    print("      Q1 params identical after load:", _same(r1, r2))
    assert_true(_same(r1, r2), "sidecar round trip changed Q1")
    assert_true(_same(r2, q2), "target not synced to online after load")
    remove(path + ".reg")
    var raised = False
    try:
        h2.load_sidecar(path)
    except:
        raised = True
    assert_true(raised, "missing sidecar did not raise")
    _ = q1


def main() raises:
    test_step_runs()
    test_reg0_bit_identical()
    test_reg_moves_actor_only()
    test_raw_flag_reaches_trainer()
    test_critic_converges()
    test_sidecar()
    print("ALL PASS — fb reg head")
