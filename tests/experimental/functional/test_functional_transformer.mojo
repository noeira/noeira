"""A small transformer written with the functional tape's ops — the
integration gate of N-d tensors, the new ops and nn's Adam.

Task: reverse a sequence of T tokens from a vocabulary of V (it needs
attention AND positions). The model, every piece a tape op: an embedding
(`F.gather`) plus a positional parameter (broadcast over the batch); two
attention heads (`F.linear` on `[B, T, D]`, `reshape` / `permute` to
`[B, H, T, Dh]`, batched `q @ k.T()`, scaled `softmax`, `@ v`, back);
a residual; a LayerNorm written out (`mean(-1, keepdim)`, `sqrt`); a ReLU
MLP; `F.cross_entropy` over `[B, T, V]` logits.

1. CPU: 400 Adam steps; the loss falls below half its start and held-out
   accuracy is far above chance (1/V).
2. GPU vs CPU: the same five steps from the same init, losses and final
   parameters.
3. NVIDIA: the training step — forward, backward, ADOPTED Adam — captured
   once and replayed on refilled token buffers == eager, bit for bit.

Run: pixi run -e apple mojo run -I . tests/experimental/functional/test_functional_transformer.mojo
"""

from std.math import sqrt as fsqrt
from std.random import seed, random_float64, random_ui64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.experimental.functional import Tape, FTensor
import noeira.experimental.functional.F as F


comptime V = 8
comptime T = 6
comptime D = 32
comptime NH = 2
comptime DH = D // NH
comptime B = 32


def _batch(n_seed: Int) -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    """B random sequences and their reversals, as float token ids."""
    seed(n_seed)
    var tok = List[Scalar[DT]]()
    var tgt = List[Scalar[DT]](length=B * T, fill=0)
    for _ in range(B * T):
        tok.append(Scalar[DT](Int(random_ui64(0, V - 1))))
    for b in range(B):
        for t in range(T):
            tgt[b * T + t] = tok[b * T + (T - 1 - t)]
    return (tok^, tgt^)


struct Model(Copyable):
    var p: List[FTensor]

    def __init__(out self, tape: Tape) raises:
        seed(42)
        self.p = List[FTensor]()
        self.p.append(tape.param[Xavier](V, D, name="emb"))
        var pos = List[Scalar[DT]]()
        for _ in range(T * D):
            pos.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * 0.5))
        self.p.append(tape.param(T, D, pos, name="pos"))
        var names: List[String] = ["wq", "wk", "wv", "wo", "w1", "w2"]
        var bnames: List[String] = ["bq", "bk", "bv", "bo", "b1", "b2"]
        for i in range(len(names)):
            self.p.append(tape.param[Xavier](D, D, name=names[i]))
            self.p.append(tape.zeros(1, D, name=bnames[i]))
        self.p.append(tape.param[Xavier](D, V, name="wout"))
        self.p.append(tape.zeros(1, V, name="bout"))

    def logits(self, tape: Tape, tok: FTensor) raises -> FTensor:
        var emb = self.p[0]
        var x = F.gather(emb, tok) + self.p[1]  # [B, T, D] + [T, D]

        def heads(h: FTensor) raises -> FTensor:
            return h.reshape([B, T, NH, DH]).permute([0, 2, 1, 3])  # [B, NH, T, DH]

        var q = heads(F.linear(x, self.p[2], self.p[3]))
        var k = heads(F.linear(x, self.p[4], self.p[5]))
        var v = heads(F.linear(x, self.p[6], self.p[7]))
        var att = ((q @ k.T()) * Scalar[DT](1.0 / fsqrt(Float64(DH)))).softmax()
        var a = (att @ v).permute([0, 2, 1, 3]).reshape([B, T, D])
        var h = x + F.linear(a, self.p[8], self.p[9])
        var mu = h.mean(-1, keepdim=True)
        var c = h - mu
        var hn = c / ((c * c).mean(-1, keepdim=True) + 1e-5).sqrt()
        var h2 = hn + F.linear(F.linear(hn, self.p[10], self.p[11], "relu"), self.p[12], self.p[13])
        return F.linear(h2, self.p[14], self.p[15])  # [B, T, V]

    def params_flat(self) raises -> List[Scalar[DT]]:
        var out = List[Scalar[DT]]()
        for i in range(len(self.p)):
            out.extend(self.p[i].values())
        return out^


def _step[target: StaticString](tape: Tape, m: Model, mut opt: Adam, tok: FTensor, tgt: FTensor, ctx: Optional[DeviceContext]) raises -> FTensor:
    tape.zero_grad()
    var loss = F.cross_entropy(m.logits(tape, tok), tgt)
    loss.backward()
    comptime if target == "gpu":
        var t = tape
        opt.step["gpu"](t, ctx)
    else:
        var t = tape
        opt.step["cpu"](t, ctx)
    return loss


def _accuracy(tape: Tape, m: Model) raises -> Float64:
    var bt = _batch(9999)
    var correct = 0
    with tape.no_grad():
        tape.reset()
        var lg = m.logits(tape, tape.input([B, T], bt[0].copy())).values()
        for r in range(B * T):
            var best = 0
            for c in range(1, V):
                if lg[r * V + c] > lg[r * V + best]:
                    best = c
            if Scalar[DT](best) == bt[1][r]:
                correct += 1
    return Float64(correct) / Float64(B * T)


def _train_cpu() raises -> Bool:
    var tape = Tape()
    var m = Model(tape)
    var opt = Adam(lr=3e-3)
    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(400):
        var bt = _batch(step)
        tape.reset()
        var loss = _step["cpu"](tape, m, opt, tape.input([B, T], bt[0].copy()),
                                tape.input([B, T], bt[1].copy()), None)
        if step == 0:
            first = loss.item()
        if step % 100 == 0 or step == 399:
            print("    step", step, "loss", loss.item())
        last = loss.item()
    var acc = _accuracy(tape, m)
    var ok = last < first * 0.5 and acc > 0.6
    print("    loss ", first, " -> ", last, ", held-out accuracy ", acc,
          " (chance ", 1.0 / Float64(V), ")  ", "OK" if ok else "FAIL", sep="")
    return ok


def _five[target: StaticString](ctx: Optional[DeviceContext]) raises -> List[Scalar[DT]]:
    var tape = Tape(ctx.value()) if ctx else Tape()
    var m = Model(tape)
    var opt = Adam(lr=3e-3)
    var out = List[Scalar[DT]]()
    for step in range(5):
        var bt = _batch(step)
        tape.reset()
        out.append(_step[target](tape, m, opt, tape.input([B, T], bt[0].copy()),
                                 tape.input([B, T], bt[1].copy()), ctx).item())
    out.extend(m.params_flat())
    return out^


def _captured(ctx: DeviceContext, capture: Bool) raises -> List[Scalar[DT]]:
    """Adopted Adam; tokens through device buffers refilled per step. With
    `capture`, step 1 is captured and steps 2.. replayed."""
    var tape = Tape(ctx)
    var m = Model(tape)
    var opt = Adam(lr=3e-3)
    opt.adopt["gpu"](tape, ctx)
    var tok = Tensor.alloc(B * T)
    var tgt = Tensor.alloc(B * T)
    tok.upload(ctx)
    tgt.upload(ctx)

    def feed(i: Int) raises {mut tok, mut tgt, ctx}:
        var bt = _batch(i)
        for k in range(B * T):
            tok.data[k] = bt[0][k]
            tgt.data[k] = bt[1][k]
        tok.upload_resident(ctx)
        tgt.upload_resident(ctx)

    def step() raises {mut tape, m, mut opt, mut tok, mut tgt, ctx} -> FTensor:
        tape.reset()
        var tk = tape.input_device(B * T, 1, tok.dev.value()).reshape([B, T])
        var tg = tape.input_device(B * T, 1, tgt.dev.value())
        return _step["gpu"](tape, m, opt, tk, tg, ctx)

    var losses = List[Scalar[DT]]()
    feed(0)
    var loss = step()
    losses.append(loss.item())
    if capture:
        feed(1)
        tape.capture_begin()
        loss = step()
        tape.capture_end()  # records, then runs it once
        if not tape.captured():
            return List[Scalar[DT]]()
        losses.append(loss.item())
        for i in range(2, 30):
            feed(i)
            tape.replay()
            losses.append(loss.item())
    else:
        for i in range(1, 30):
            feed(i)
            loss = step()
            losses.append(loss.item())
    losses.extend(m.params_flat())
    return losses^


def _cmp(name: String, got: List[Scalar[DT]], want: List[Scalar[DT]], tol: Float64, exact: Bool) -> Bool:
    var worst: Float64 = 0
    var differ = 0
    for k in range(len(want)):
        if got[k] != want[k]:
            differ += 1
        worst = max(worst, abs(Float64(got[k]) - Float64(want[k])) / max(1.0, abs(Float64(want[k]))))
    var ok = len(got) == len(want) and len(want) > 0 and ((differ == 0) if exact else (worst <= tol))
    print("    ", name, ": ", len(want), " compared, ", differ,
          " not bit-identical, worst rel ", worst, "  ", "OK" if ok else "FAIL", sep="")
    return ok


def main() raises:
    var ok = True
    print("CPU: train to reverse", T, "tokens of", V)
    ok = _train_cpu() and ok
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var g5 = _five["gpu"](ctx)
    var c5 = _five["cpu"](None)
    # Losses tightly; parameters loosely — Adam's first steps move each
    # weight by ~±lr, so a gradient that differs by an ulp NEAR ZERO can
    # flip one step's direction (|Δ| <= 2·lr = 6e-3).
    var gl = List[Scalar[DT]]()
    var cl = List[Scalar[DT]]()
    var gp = List[Scalar[DT]]()
    var cp = List[Scalar[DT]]()
    for k in range(len(c5)):
        if k < 5:
            gl.append(g5[k])
            cl.append(c5[k])
        else:
            gp.append(g5[k])
            cp.append(c5[k])
    ok = _cmp("five Adam steps, GPU vs CPU: losses", gl, cl, 1e-4, False) and ok
    ok = _cmp("five Adam steps, GPU vs CPU: parameters", gp, cp, 1e-2, False) and ok
    var c = _captured(ctx, True)
    if len(c) == 0:
        print("    capture unavailable on this device (NVIDIA only) — not tested")
    else:
        ok = _cmp("captured + replayed training (30 steps) vs eager",
                  c, _captured(ctx, False), 0, True) and ok
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("functional transformer gate failed")
