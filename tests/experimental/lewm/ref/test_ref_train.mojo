"""G2b — the reference-exact LeWM TRAINING step against torch: loss, every
gradient, BN running stats.

docs/LEWM_REOPEN_PLAN.md P2. `ref_model.LeWMLossGraphRef` loaded with the
published checkpoint runs the dump's `train` window (B = 4, T = 4) exactly as
`train.py:lejepa_forward` does in torch (float64): BN in TRAIN mode, dropout
off, SIGReg on torch's own projection matrix (injected via `fixed_a`),
λ = 0.09, no stop-gradient. Compared:

  * the loss and its two terms, the embeddings, the predictions;
  * d loss / d param for every one of the 309 parameters with a torch
    counterpart (the 6 predictor qkv biases have none: the documented
    deviation, ref_model.mojo) — this is the check of every VJP the model uses;
  * BN running stats after the step (the unbiased update);
  * the global gradient norm over the mapped parameters (torch:
    `clip_grad_norm_`'s return, `adamw.grad_norm`).

Why no AdamW-step comparison: on a FIRST step Adam's update is
lr·g/(|g|+eps) ≈ lr·sign(g) — the clip scale cancels and the weight decay
(lr·wd·p ≈ 5e-8·|p|) is below any float32 tolerance, so a one-step delta
checks signs, not the optimizer. P6's multi-step training parity gates it.

Unit: max |ours - torch| / std(torch) per tensor; for GRADIENTS the
denominator is max(std(torch), global grad RMS) — tensors with an exact-zero
gradient (bias / LN beta into a train-mode BN, attention key biases) are
otherwise roundoff over roundoff — and each gradient is held to
max(TOL, 10 x torch's own float32 error on it), measured by the dumper's
`--noise` in the same metric.

Run (after the dump + convert, see test_ref_forward.mojo):
    pixi run -e apple mojo run -I . tests/experimental/lewm/ref/test_ref_train.mojo
"""

from std.sys import argv
from std.math import sqrt, abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_model import LeWMLossGraphRef
from noeira.experimental.lewm.ref_load import LoadRef, ref_input, std_err


comptime TOL = 1e-4
comptime B = 4
comptime T = 4
comptime H = 3
comptime EMB = 192
comptime LAMBDA = 0.09

comptime Graph = LeWMLossGraphRef[
    3, 224, 14, 192, 3, 12, EMB, 2048,
    T, 10, H, 1,
    16, 64, 2048, 6,
    1024, 17,
]


struct _GradCheck(ParamVisitor):
    """Each param's gradient vs `ours_grad.<name>`; also the global norm."""

    var dump: RefDump
    var worst: Float64
    var worst_name: String
    var n_checked: Int
    var n_skipped: Int
    var failed: List[String]
    var sumsq: Float64
    var rms: Float64
    var worst_ratio: Float64

    def __init__(out self, var dump: RefDump) raises:
        self.rms = Float64(dump.get(String("train.grad_rms"))[0])
        self.worst_ratio = 0.0
        self.dump = dump^
        self.worst = 0.0
        self.worst_name = String("")
        self.n_checked = 0
        self.n_skipped = 0
        self.failed = List[String]()
        self.sumsq = 0.0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var key = String("ours_grad.") + name
        if not self.dump.has(key):
            self.n_skipped += 1  # the qkv biases: no torch counterpart
            return
        comptime if target == "gpu":
            grad.download(ctx.value())
        var want = self.dump.get(key)
        var mean = 0.0
        for i in range(N):
            mean += Float64(want[i])
        mean /= Float64(N)
        var var_ = 0.0
        var worst = 0.0
        for i in range(N):
            var d = Float64(want[i]) - mean
            var_ += d * d
            worst = max(worst, abs(Float64(grad.data[i]) - Float64(want[i])))
            self.sumsq += Float64(grad.data[i]) * Float64(grad.data[i])
        # floor at the global gradient RMS: several tensors have an EXACT zero
        # gradient (a bias / LN beta feeding a train-mode BatchNorm, attention
        # key biases), where std(torch) is roundoff (see the dumper)
        var den = max(sqrt(var_ / Float64(N)), self.rms)
        var err = worst / den
        # held to 10x torch's OWN float32 error on this tensor (same metric)
        var noise = Float64(self.dump.get(String("ours_noise_grad.") + name)[0])
        var tol = max(TOL, 10.0 * noise)
        self.n_checked += 1
        if err > self.worst:
            self.worst = err
            self.worst_name = name
        self.worst_ratio = max(self.worst_ratio, err / tol)
        if err > tol:
            self.failed.append(name + " " + String(err) + " (tol " + String(tol) + ")")


def _node_err[NODE: StaticString, target: StaticString](
    mut g: Graph, dump: String, ref_name: String, ctx: Optional[DeviceContext]
) raises -> Float64:
    ref out = g.node_output[NODE]()
    comptime if target == "gpu":
        ctx.value().synchronize()
        out.download(ctx.value())
    var t = Tensor.alloc(out.n)
    for i in range(out.n):
        t.data[i] = out.data[i]
    var e = std_err["cpu"](dump, ref_name, t, ctx)
    var flag = String("  ") if e <= TOL else String("✗ ")
    print("     ", flag, NODE, " ", e, sep="")
    return e


struct _BNCheck(ParamVisitor):
    """BN running stats after the training forward vs `ours_bn_after.*`."""

    var dump: RefDump
    var worst: Float64
    var n_checked: Int

    def __init__(out self, var dump: RefDump):
        self.dump = dump^
        self.worst = 0.0
        self.n_checked = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var key = String("ours_bn_after.") + name
        if not self.dump.has(key):
            return
        comptime if target == "gpu":
            param.download(ctx.value())
        var want = self.dump.get(key)
        var worst = 0.0
        var mean = 0.0
        for i in range(N):
            mean += Float64(want[i])
        mean /= Float64(N)
        var var_ = 0.0
        for i in range(N):
            var_ += (Float64(want[i]) - mean) * (Float64(want[i]) - mean)
            worst = max(worst, abs(Float64(param.data[i]) - Float64(want[i])))
        var sd = sqrt(var_ / Float64(N))
        self.worst = max(self.worst, worst / (sd if sd > 0.0 else 1.0))
        self.n_checked += 1


def _run[target: StaticString](dump: String, ctx: Optional[DeviceContext]) raises -> Int:
    print("  --", target)
    var g = Graph.make[target, Kaiming](ctx)
    var lv = LoadRef(RefDump(dump), String(""))
    g.for_each_param[target](lv, ctx)
    g.for_each_state[target](lv, ctx)
    lv.check()
    print("     loaded", lv.loaded, "tensors")
    g.set_node_attr["sig_s", "multiplier"](Scalar[DT](LAMBDA))
    # torch's own normalised projection matrix, for forward AND vjp
    var a = RefDump(dump).get(String("train.sigreg_A"))
    var c = ctx.value() if ctx else DeviceContext()
    var abuf = c.enqueue_create_buffer[DT](len(a))
    with abuf.map_to_host() as h:
        for i in range(len(a)):
            h[i] = a[i]
    g.set_node_attr_buf["sig", "fixed_a"](abuf)

    var pix = ref_input[target](dump, String("train.pixels"), ctx)
    var act = ref_input[target](dump, String("train.action"), ctx)
    g.zero_grad[target](ctx)
    g.set_input["pixels", B](pix, ctx)
    g.set_input["actions", B](act, ctx)
    var loss = Tensor.alloc(B)
    g.forward[B, target](loss, ctx)
    var seed = Tensor.alloc(B)
    for i in range(B):
        seed.data[i] = Scalar[DT](1.0 / Float64(B))
    comptime if target == "gpu":
        seed.upload(ctx.value())
    g.vjp[B, target](seed, ctx)

    var fails = 0
    # the scalars: batch mean of the per-sample outputs
    comptime if target == "gpu":
        ctx.value().synchronize()
        loss.download(ctx.value())
    var lm = 0.0
    for i in range(B):
        lm += Float64(loss.data[i])
    lm /= Float64(B)
    var want_loss = Float64(RefDump(dump).get(String("train.loss"))[0])
    var rel = abs(lm - want_loss) / abs(want_loss)
    print("     loss  ours", lm, " torch", want_loss, " rel", rel)
    if rel > 1e-5:
        fails += 1
        print("     ✗ loss")

    var e_emb = _node_err["emb", target](g, dump, String("train.emb"), ctx)
    var e_pred = _node_err["pred", target](g, dump, String("train.pred_emb"), ctx)
    for e in [e_emb, e_pred]:
        if e > TOL:
            fails += 1

    var bc = _BNCheck(RefDump(dump))
    g.for_each_state[target](bc, ctx)
    print("     BN running stats after the step:", bc.n_checked, "checked; worst", bc.worst)
    if bc.n_checked != 4 or bc.worst > TOL:
        fails += 1
        print("     ✗ BN running stats")

    var gc = _GradCheck(RefDump(dump))
    g.for_each_param[target](gc, ctx)
    var want_norm = Float64(RefDump(dump).get(String("adamw.grad_norm"))[0])
    var norm = sqrt(gc.sumsq)
    var nrel = abs(norm - want_norm) / want_norm
    print("     grads:", gc.n_checked, "checked,", gc.n_skipped, "skipped (qkv bias); worst",
          gc.worst, "at", gc.worst_name, "; worst err/tol", gc.worst_ratio)
    print("     grad norm ours", norm, " torch", want_norm, " rel", nrel)
    for f in gc.failed:
        print("     ✗ grad", f)
    fails += len(gc.failed)
    if nrel > 1e-5:
        fails += 1
        print("     ✗ grad norm")
    if gc.n_checked != 309:
        fails += 1
        print("     ✗ expected 309 checked gradients")
    return fails


def main() raises:
    var dump = String("/tmp/lewm_ref")
    var args = argv()
    if len(args) > 1:
        dump = String(args[1])
    print("G2b  LeWM reference training step vs torch (TOL", TOL, "std units)")
    var fails = _run["cpu"](dump, None)
    var c = DeviceContext()
    fails += _run["gpu"](dump, Optional(c))
    if fails > 0:
        raise Error("FAIL G2b: " + String(fails) + " check(s) above tolerance")
    print("PASS")
