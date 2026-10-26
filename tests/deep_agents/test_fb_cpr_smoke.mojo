"""`FBCPRTrainer` smoke gate — the CPR terms enter where they should and
NOWHERE else.

  [1] a CPR step runs; every reported term is finite; at init the two BCE
      halves sit near log 2 (D has not seen anything yet).
  [2] the expert window encoding: rows of one window are identical, every
      row is on the radius-sqrt(D) sphere, and the value equals the hand
      mean of `B(s'_j)` projected — through `backward_embed`, the other
      producer of `B`.
  [3] ⚠ the load-bearing one: with `reg_coeff = 0` the FB half is
      BIT-IDENTICAL to a plain `FBTrainer` on the same batch and seed —
      every parameter of B, F1, F2 and the actor. The 24-D base's arithmetic
      is untouched when CPR is off, so a CPR arm differs from the base by
      the CPR terms and nothing else.
  [4] with `reg_coeff > 0` the ACTOR moves differently while B, F1, F2 stay
      bit-identical: the style term reaches the actor and only the actor.
  [5] the gradient penalty contributes: `gp_coef` 0 vs 10 move D
      differently; the same seed twice moves it identically.
  [6] checkpoint: FB file + `.cpr` sidecar round-trip (D logits on a probe
      equal after `load_state`); a missing sidecar RAISES.
  [7] `stash_critic_grads` (the actor step's backward through F1, F2, Q_D1
      and Q_D2 with their gradient buffers moved out, so no dW is formed)
      leaves EVERY net bit-identical to the old form-and-discard path over
      three steps: B, F1, F2, actor, D, Q_D1, Q_D2.

⚠ Smoke gate on random data. Whether CPR closes any of the walker gap is
the run (`examples/fb/fb_train_cpr_gpu.mojo`), not this file.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_cpr_smoke.mojo
"""

from std.math import abs, sqrt, log
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import Tanh, ReLU
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.cpr import FBCPRTrainer
from noeira.deep_agents.fb import sample_z_uniform


comptime OBS: Int = 4
comptime ACT: Int = 2
comptime D: Int = 6
comptime SEQ: Int = 4
comptime BATCH: Int = 16
comptime HID: Int = 32
comptime SEED: Int = 20260908

comptime F_IN = OBS + ACT + D
comptime A_IN = OBS + D
comptime D_IN = OBS + D

comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]
# BFM-Zero's discriminator shape at toy width: Linear -> LayerNorm -> Tanh
# -> (Linear -> ReLU) x2 -> Linear(1).
comptime DNet = Sequential[
    Linear[D_IN, HID], LayerNorm[HID], Tanh[HID],
    Linear[HID, HID], ReLU[HID], Linear[HID, HID], ReLU[HID], Linear[HID, 1],
]
comptime QNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, 1]]

comptime Plain = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH]
comptime CPR = FBCPRTrainer[FNet, BNet, ANet, DNet, QNet, OBS, ACT, D, BATCH, SEQ]


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
    var es: Tensor
    var esn: Tensor

    def __init__(out self) raises:
        self.s = _rand_tensor(BATCH * OBS, 1.0)
        self.a = _rand_tensor(BATCH * ACT, 1.0)
        self.sn = _rand_tensor(BATCH * OBS, 1.0)
        self.sp = _rand_tensor(BATCH * OBS, 1.0)
        self.z = _z_tensor(BATCH)
        self.es = _rand_tensor(BATCH * OBS, 1.0)
        self.esn = _rand_tensor(BATCH * OBS, 1.0)

    def __init__(out self, *, deinit move: Self):
        self.s = move.s^
        self.a = move.a^
        self.sn = move.sn^
        self.sp = move.sp^
        self.z = move.z^
        self.es = move.es^
        self.esn = move.esn^


def _make_cpr(reg: Float64, gp: Float64, bc: Float64) raises -> CPR:
    seed(SEED)
    var c = CPR.make(
        None, lr=1e-3, lr_b=1e-3, lr_d=1e-3, lr_q=1e-3, ortho_weight=1.0,
        max_grad_norm=0.0, bc_weight=bc, reg_coeff=reg, gp_coef=gp,
        seed=UInt64(SEED),
    )
    # ⚠ Target smoothing OFF on both trainers for the bit-identity gates:
    # the CPR step draws its own host noise before the inner step, which
    # moves the host RNG, so the inner's noise VALUES differ from a plain
    # trainer's; with sigma = 0 they multiply to exactly 0 on both.
    c.t.policy_noise = 0.0
    c.head.policy_noise = 0.0
    return c^


def _load(mut c: CPR, ref b: _Batch) raises:
    var s = _clone(b.s, BATCH * OBS)
    var a = _clone(b.a, BATCH * ACT)
    var sn = _clone(b.sn, BATCH * OBS)
    var sp = _clone(b.sp, BATCH * OBS)
    var z = _clone(b.z, BATCH * D)
    var es = _clone(b.es, BATCH * OBS)
    var esn = _clone(b.esn, BATCH * OBS)
    c.t.load_batch(s, a, sn, sp, z)
    c.load_expert(es, esn)
    c.encode_expert()


def _read_fb(mut t: Plain, mut rb: _ReadVals, mut rf1: _ReadVals, mut rf2: _ReadVals, mut ra: _ReadVals) raises:
    t.bnet.online.for_each_param["cpu"](rb, None)
    t.f1.online.for_each_param["cpu"](rf1, None)
    t.f2.online.for_each_param["cpu"](rf2, None)
    t.actor.online.for_each_param["cpu"](ra, None)


def test_step_runs() raises -> _Batch:
    print("[1] a CPR step runs; BCE halves near log 2 at init ...")
    seed(SEED + 1)
    var b = _Batch()
    var c = _make_cpr(0.01, 10.0, 1.0)
    _load(c, b)
    var l = c.train_step()
    print(
        "      measure", l.fb.measure, " ortho", l.fb.ortho, " actor", l.fb.actor,
        "\n      d_pos", l.d_pos, " d_neg", l.d_neg, " gp", l.d_gp,
        "\n      r_mean", l.r_mean, " q_mean", l.q_mean, " q_loss", l.q_loss, " q_pi", l.q_pi,
    )
    assert_true(l.d_pos == l.d_pos and l.d_neg == l.d_neg, "BCE is NaN")
    assert_true(l.d_gp == l.d_gp and l.q_loss == l.q_loss, "gp / q_loss is NaN")
    assert_true(l.fb.measure == l.fb.measure, "measure is NaN")
    var ln2 = log(2.0)
    assert_true(abs(l.d_pos - ln2) < 0.6 and abs(l.d_neg - ln2) < 0.6,
                "BCE halves far from log 2 at init")
    assert_true(l.d_gp > 0.0, "gradient penalty reported zero")
    return b^


def test_expert_encoding(ref b: _Batch) raises:
    print("[2] expert window encoding: repeated per window, on the sphere, == mean B(s') ...")
    var c = _make_cpr(0.01, 10.0, 1.0)
    _load(c, b)
    var esn = _clone(b.esn, BATCH * OBS)
    var bb = Tensor()
    c.t.backward_embed[BATCH](esn, bb)
    var radius = sqrt(Float64(D))
    var worst = Float64(0)
    comptime NW = BATCH // SEQ
    for w in range(NW):
        var m = List[Float64](length=D, fill=0.0)
        for j in range(SEQ):
            for k in range(D):
                m[k] += Float64(bb.data[(w * SEQ + j) * D + k]) / Float64(SEQ)
        var n = Float64(0)
        for k in range(D):
            n += m[k] * m[k]
        n = sqrt(n)
        for j in range(SEQ):
            var rn = Float64(0)
            for k in range(D):
                var got = Float64(c.head.ez.data[(w * SEQ + j) * D + k])
                var want = m[k] * radius / n
                var d = abs(got - want)
                if d > worst:
                    worst = d
                rn += got * got
                # identical across the window
                assert_true(c.head.ez.data[(w * SEQ + j) * D + k] == c.head.ez.data[w * SEQ * D + k],
                            "window rows differ")
            assert_true(abs(sqrt(rn) - radius) < 1e-4, "expert z off the sphere")
    print("      worst |ez − project(mean B)| =", worst)
    assert_true(worst < 1e-5, "expert encoding != projected window mean")


def test_reg_zero_is_plain_fb(ref b: _Batch) raises:
    print("[3] reg_coeff = 0: B, F1, F2, actor BIT-IDENTICAL to a plain FBTrainer ...")
    seed(SEED)
    var p = Plain.make(lr=1e-3, ortho_weight=1.0, seed=UInt64(SEED), bc_weight=1.0, lr_b=1e-3)
    p.policy_noise = 0.0
    var s = _clone(b.s, BATCH * OBS)
    var a = _clone(b.a, BATCH * ACT)
    var sn = _clone(b.sn, BATCH * OBS)
    var sp = _clone(b.sp, BATCH * OBS)
    var z = _clone(b.z, BATCH * D)
    p.load_batch(s, a, sn, sp, z)
    _ = p.train_step()
    var pb = _ReadVals()
    var pf1 = _ReadVals()
    var pf2 = _ReadVals()
    var pa = _ReadVals()
    _read_fb(p, pb, pf1, pf2, pa)

    var c = _make_cpr(0.0, 10.0, 1.0)
    _load(c, b)
    _ = c.train_step()
    var cb = _ReadVals()
    var cf1 = _ReadVals()
    var cf2 = _ReadVals()
    var ca = _ReadVals()
    _read_fb(c.t, cb, cf1, cf2, ca)
    assert_true(_same(pb, cb), "B differs from the plain trainer at reg 0")
    assert_true(_same(pf1, cf1), "F1 differs from the plain trainer at reg 0")
    assert_true(_same(pf2, cf2), "F2 differs from the plain trainer at reg 0")
    assert_true(_same(pa, ca), "actor differs from the plain trainer at reg 0")
    print("      4 / 4 nets bit-identical")


def test_reg_moves_actor_only(ref b: _Batch) raises:
    print("[4] reg_coeff > 0 moves the ACTOR; B, F1, F2 unchanged ...")
    var c0 = _make_cpr(0.0, 10.0, 1.0)
    _load(c0, b)
    _ = c0.train_step()
    var b0 = _ReadVals()
    var f10 = _ReadVals()
    var f20 = _ReadVals()
    var a0 = _ReadVals()
    _read_fb(c0.t, b0, f10, f20, a0)

    var c1 = _make_cpr(1.0, 10.0, 1.0)
    _load(c1, b)
    _ = c1.train_step()
    var b1 = _ReadVals()
    var f11 = _ReadVals()
    var f21 = _ReadVals()
    var a1 = _ReadVals()
    _read_fb(c1.t, b1, f11, f21, a1)
    assert_true(_same(b0, b1), "B moved with reg_coeff")
    assert_true(_same(f10, f11), "F1 moved with reg_coeff")
    assert_true(_same(f20, f21), "F2 moved with reg_coeff")
    assert_true(not _same(a0, a1), "actor did NOT move with reg_coeff — the hook is dead")
    # ... and with BC off the scaled path (`axpy_by_mag`) is the one taken.
    var c2 = _make_cpr(0.0, 10.0, 0.0)
    _load(c2, b)
    _ = c2.train_step()
    var a2 = _ReadVals()
    c2.t.actor.online.for_each_param["cpu"](a2, None)
    var c3 = _make_cpr(1.0, 10.0, 0.0)
    _load(c3, b)
    _ = c3.train_step()
    var a3 = _ReadVals()
    c3.t.actor.online.for_each_param["cpu"](a3, None)
    assert_true(not _same(a2, a3), "actor did NOT move with reg_coeff at bc 0")
    print("      actor moved, three other nets bit-identical, both BC settings")


def test_gp_contributes(ref b: _Batch) raises:
    print("[5] gradient penalty: gp 0 vs 10 move D differently; same seed twice identical ...")
    var c0 = _make_cpr(0.01, 0.0, 1.0)
    _load(c0, b)
    _ = c0.train_step()
    var d0 = _ReadVals()
    c0.head.disc.for_each_param["cpu"](d0, None)
    var c1 = _make_cpr(0.01, 10.0, 1.0)
    _load(c1, b)
    _ = c1.train_step()
    var d1 = _ReadVals()
    c1.head.disc.for_each_param["cpu"](d1, None)
    assert_true(not _same(d0, d1), "gp_coef had no effect on D")
    var c2 = _make_cpr(0.01, 10.0, 1.0)
    _load(c2, b)
    _ = c2.train_step()
    var d2 = _ReadVals()
    c2.head.disc.for_each_param["cpu"](d2, None)
    assert_true(_same(d1, d2), "same seed, different D — a hidden RNG")
    print("      ok")


def test_checkpoint(ref b: _Batch) raises:
    print("[6] checkpoint: FB file + .cpr sidecar round trip; missing sidecar raises ...")
    var c = _make_cpr(0.01, 10.0, 1.0)
    _load(c, b)
    for _ in range(3):
        _ = c.train_step()
    var path = String("/tmp/fb_cpr_smoke.ckpt")
    c.save_state(path)
    var s = _clone(b.s, BATCH * OBS)
    var z = _clone(b.z, BATCH * D)
    var l0 = Tensor()
    c.discriminate[BATCH](s, z, l0)

    var c2 = _make_cpr(0.01, 10.0, 1.0)
    c2.load_state(path)
    var l1 = Tensor()
    c2.discriminate[BATCH](s, z, l1)
    # ⚠ The checkpoint is TEXT (`String(value)`), so the round trip is
    # not bit-exact — the FB checkpoint gate bands at 1e-6 for the same reason.
    var wd = Float64(0)
    for i in range(BATCH):
        var e = abs(Float64(l0.data[i]) - Float64(l1.data[i]))
        if e > wd:
            wd = e
    print("      worst |D logit before − after| =", wd)
    assert_true(wd < 1e-5, "D logits differ after load: " + String(wd))
    # the FB half loads into the plain trainer the evals build
    var p = Plain.make(lr=1e-3)
    p.load_state(path)
    var pa = Tensor()
    var ca = Tensor()
    p.act[BATCH](s, z, pa)
    c2.t.act[BATCH](s, z, ca)
    var wa = Float64(0)
    for i in range(BATCH * ACT):
        var e = abs(Float64(pa.data[i]) - Float64(ca.data[i]))
        if e > wa:
            wa = e
    assert_true(wa < 1e-6, "plain trainer reads a different actor: " + String(wa))

    var c3 = _make_cpr(0.01, 10.0, 1.0)
    var raised = False
    try:
        c3.load_state(String("/tmp/fb_cpr_smoke_missing.ckpt"))
    except:
        raised = True
    # (the FB file is missing too, so this raises either way — the sidecar
    # rule is checked on a file whose FB half exists:)
    c.t.save_state(String("/tmp/fb_cpr_smoke_nosidecar.ckpt"))
    var raised2 = False
    try:
        c3.load_state(String("/tmp/fb_cpr_smoke_nosidecar.ckpt"))
    except:
        raised2 = True
    assert_true(raised and raised2, "missing sidecar did not raise")
    print("      ok")


def _read_all(mut c: CPR) raises -> List[_ReadVals]:
    var out = List[_ReadVals]()
    for k in range(7):
        out.append(_ReadVals())
    c.t.bnet.online.for_each_param["cpu"](out[0], None)
    c.t.f1.online.for_each_param["cpu"](out[1], None)
    c.t.f2.online.for_each_param["cpu"](out[2], None)
    c.t.actor.online.for_each_param["cpu"](out[3], None)
    c.head.disc.for_each_param["cpu"](out[4], None)
    c.head.qd1.online.for_each_param["cpu"](out[5], None)
    c.head.qd2.online.for_each_param["cpu"](out[6], None)
    return out^


def test_stash_critic_grads(ref b: _Batch) raises:
    print("[7] stash_critic_grads: every net bit-identical to form-and-discard ...")
    # ⚠ One arm AFTER the other, never interleaved: the CPR step draws from
    # the global host RNG, which `_make_cpr` re-seeds.
    var c0 = _make_cpr(1.0, 10.0, 1.0)
    c0.t.stash_critic_grads = False
    _load(c0, b)
    for _ in range(3):
        _ = c0.train_step()
    var r0 = _read_all(c0)
    var c1 = _make_cpr(1.0, 10.0, 1.0)
    assert_true(c1.t.stash_critic_grads, "the stash is not the default")
    _load(c1, b)
    for _ in range(3):
        _ = c1.train_step()
    var r1 = _read_all(c1)
    var names: List[String] = ["B", "F1", "F2", "actor", "D", "Q_D1", "Q_D2"]
    for k in range(7):
        assert_true(_same(r0[k], r1[k]), names[k] + " differs with the stash")
    print("      7 / 7 nets bit-identical after 3 steps")


def main() raises:
    print("=" * 70)
    print("FBCPRTrainer smoke")
    print("=" * 70)
    var b = test_step_runs()
    test_expert_encoding(b)
    test_reg_zero_is_plain_fb(b)
    test_reg_moves_actor_only(b)
    test_gp_contributes(b)
    test_checkpoint(b)
    test_stash_critic_grads(b)
    print("\n[PASS] FBCPRTrainer smoke")
