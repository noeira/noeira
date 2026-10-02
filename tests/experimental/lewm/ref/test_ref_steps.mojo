"""G6a — five training steps of the reference-exact LeWM against torch:
the optimizer, the clip, the decay and the BN statistics, chained.

docs/LEWM_REOPEN_PLAN.md P6. The dump's `steps` section ran `train.py`'s step
five times from the published weights in float64 (each step its own (4, 4)
window and SIGReg matrix, dropout off): `clip_grad_norm_(1.0)` then torch
AdamW (lr 5e-5, wd 1.0 on EVERY parameter — the gate's wd: at the recipe's
1e-3 the decay is below float32). `ref_trainer.LeWMRefTrainer` replays it:

  * per step: loss, pred loss, SIGReg, pre-clip norm (relative) — each step
    runs on OUR weights after our previous steps, so an optimizer error
    shows here as soon as it moves the loss;
  * after five steps: every parameter's total delta, max |ours - torch| /
    std(torch) over the LIVE elements (a gradient above 1e-6 of the RMS at
    some step — below that float32 cannot resolve its sign and Adam turns it
    into a ±lr step either way), held to max(TOL, 10 x torch's own float32
    error on that delta); the predictor qkv bias (no reference counterpart)
    must still be exactly 0;
  * the BN running stats after five steps.

Mutation (`--mutate-decay`): AdamW skipping the `apply_decay=False` params
(`nn.Adam`'s default) must fail.

Run (after `pixi run dump-lewm-ref`):
    pixi run -e apple mojo run -I . tests/experimental/lewm/ref/test_ref_steps.mojo
"""

from std.sys import argv
from std.math import sqrt, abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_load import tf32_gemm
from noeira.experimental.lewm.ref_trainer import LeWMRefTrainer


comptime B = 4
comptime TOL = 1e-3
"""Deltas, std units (torch's own float32 error: median 2.5e-3, max 0.09 —
most tensors are held to 10x their own)."""
comptime TOL_SCALAR = 1e-5
comptime TOL_BN = 1e-4
comptime REL_TF32 = 2e-2
comptime COS_TF32 = 0.99


struct _DeltaCheck(ParamVisitor):
    var dump: RefDump
    var tf32: Bool
    var n_checked: Int
    var worst: Float64
    var worst_name: String
    var worst_ratio: Float64
    var worst_cos: Float64
    var failed: List[String]
    var qkv_bias_zero: Bool
    var dead_tensors: List[String]

    def __init__(out self, var dump: RefDump, tf32: Bool):
        self.dump = dump^
        self.tf32 = tf32
        self.n_checked = 0
        self.worst = 0.0
        self.worst_name = String("")
        self.worst_ratio = 0.0
        self.worst_cos = 1.0
        self.failed = List[String]()
        self.qkv_bias_zero = True
        self.dead_tensors = List[String]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            ctx.value().synchronize()
            param.download(ctx.value())
        var key = String("ours_steps_delta.") + name
        if not self.dump.has(key):
            # the predictor qkv bias: no reference counterpart, must not move
            for i in range(N):
                if param.data[i] != Scalar[DT](0.0):
                    self.qkv_bias_zero = False
            return
        var p0 = self.dump.get(String("ours.") + name)
        var want = self.dump.get(key)
        var dead = self.dump.get(String("ours_steps_dead.") + name)
        var n = 0
        var mean = 0.0
        for i in range(N):
            if dead[i] == Scalar[DT](0.0):
                mean += Float64(want[i])
                n += 1
        if n < 2:
            # wholly dead: a bias / LN beta feeding a train-mode BN (its
            # effect is removed with the batch mean) — nothing to gate
            self.dead_tensors.append(name)
            return
        mean /= Float64(n)
        var var_ = 0.0
        var worst = 0.0
        var dot = 0.0
        var nw = 0.0
        var ng = 0.0
        for i in range(N):
            if dead[i] != Scalar[DT](0.0):
                continue
            var w = Float64(want[i])
            var d = Float64(param.data[i]) - Float64(p0[i])
            var_ += (w - mean) * (w - mean)
            worst = max(worst, abs(d - w))
            dot += d * w
            nw += w * w
            ng += d * d
        var sd = sqrt(var_ / Float64(n))
        var err = worst / (sd if sd > 0.0 else 1.0)
        self.n_checked += 1
        if err > self.worst:
            self.worst = err
            self.worst_name = name
        if self.tf32:
            var cos = dot / max(sqrt(nw) * sqrt(ng), 1e-300)
            self.worst_cos = min(self.worst_cos, cos)
            if cos < COS_TF32:
                self.failed.append(name + " cos " + String(cos))
            return
        var noise = Float64(self.dump.get(String("ours_steps_noise.") + name)[0])
        var tol = max(TOL, 10.0 * noise)
        self.worst_ratio = max(self.worst_ratio, err / tol)
        if err > tol:
            self.failed.append(name + " " + String(err) + " (tol " + String(tol) + ")")


struct _BNCheck(ParamVisitor):
    var dump: RefDump
    var n_checked: Int
    var worst_ratio: Float64
    var failed: List[String]
    var band: Float64

    def __init__(out self, var dump: RefDump, band: Float64):
        self.dump = dump^
        self.n_checked = 0
        self.worst_ratio = 0.0
        self.failed = List[String]()
        self.band = band

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var key = String("ours_steps_bn.") + name
        if not self.dump.has(key):
            return
        comptime if target == "gpu":
            param.download(ctx.value())
        var want = self.dump.get(key)
        var mean = 0.0
        for i in range(N):
            mean += Float64(want[i])
        mean /= Float64(N)
        var var_ = 0.0
        var worst = 0.0
        for i in range(N):
            var_ += (Float64(want[i]) - mean) ** 2
            worst = max(worst, abs(Float64(param.data[i]) - Float64(want[i])))
        var sd = sqrt(var_ / Float64(N))
        var err = worst / (sd if sd > 0.0 else 1.0)
        var noise = Float64(self.dump.get(String("ours_steps_noise_bn.") + name)[0])
        var tol = max(self.band, 10.0 * noise)
        self.n_checked += 1
        self.worst_ratio = max(self.worst_ratio, err / tol)
        if err > tol:
            self.failed.append(name + " " + String(err) + " (tol " + String(tol) + ")")


def _run[target: StaticString](
    dump: String, ctx: Optional[DeviceContext], decay_all: Bool
) raises -> Int:
    var tf32 = tf32_gemm[target]()
    if tf32:
        print("  --", target, "(TF32 GEMMs: scalars rel", REL_TF32, ", deltas by cosine >=", COS_TF32, ")")
    else:
        print("  --", target)
    var rd = RefDump(dump)
    var hp = rd.get(String("steps.hparams"))  # lr, wd, clip, K
    var K = Int(hp[3])
    var tr = LeWMRefTrainer[target, B](
        ctx, lr=Float64(hp[0]), wd=Float64(hp[1]), max_norm=Float64(hp[2]),
        decay_all=decay_all,
    )
    var n = tr.load(dump)
    print("     loaded", n, "tensors; lr", hp[0], " wd", hp[1], " clip", hp[2], " steps", K)
    var want = rd.get(String("steps.scalars"))          # (K, 4)
    var noise = rd.get(String("steps.noise_scalars"))   # (K, 4) rel
    var names: List[String] = ["loss", "pred", "sigreg", "norm"]
    var fails = 0
    for k in range(K):
        var p = String("steps.") + String(k) + "."
        tr.set_sigreg_a(rd.get(p + "sigreg_A"))
        var st = tr.train_step(rd.get(p + "pixels"), rd.get(p + "action"))
        var got: List[Float64] = [st.loss, st.pred_loss, st.sigreg_loss, st.grad_norm]
        var line = String("     step ") + String(k)
        for j in range(4):
            var w = Float64(want[k * 4 + j])
            var rel = abs(got[j] - w) / abs(w)
            var tol = REL_TF32 if tf32 else max(TOL_SCALAR, 10.0 * Float64(noise[k * 4 + j]))
            line += "  " + names[j] + " " + String(Float32(got[j])) + " (rel " + String(Float32(rel)) + ")"
            if rel > tol:
                fails += 1
                line += " ✗"
        print(line)

    var dc = _DeltaCheck(RefDump(dump), tf32)
    tr.graph.for_each_param[target](dc, ctx)
    var dead = String("")
    for d in dc.dead_tensors:
        dead += " " + d
    print("     deltas:", dc.n_checked, "checked +", len(dc.dead_tensors), "wholly dead (" + dead + " ); worst", dc.worst, "at", dc.worst_name,
          "; worst err/tol", dc.worst_ratio, "; worst cosine", dc.worst_cos)
    for f in dc.failed:
        print("     ✗ delta", f)
    fails += len(dc.failed)
    if dc.n_checked != 305 or len(dc.dead_tensors) != 4:
        fails += 1
        print("     ✗ expected 305 checked + 4 wholly dead deltas")
    if not dc.qkv_bias_zero:
        fails += 1
        print("     ✗ the predictor qkv bias moved")
    var bc = _BNCheck(RefDump(dump), 3e-2 if tf32 else TOL_BN)
    tr.graph.for_each_state[target](bc, ctx)
    print("     BN running stats:", bc.n_checked, "checked; worst err/tol", bc.worst_ratio)
    for f in bc.failed:
        print("     ✗ bn", f)
    fails += len(bc.failed)
    if bc.n_checked != 4:
        fails += 1
    return fails


def main() raises:
    var dump = String("/tmp/lewm_ref")
    var decay_all = True
    var args = argv()
    for i in range(1, len(args)):
        if String(args[i]) == "--mutate-decay":
            decay_all = False
        else:
            dump = String(args[i])
    print("G6a  five LeWM training steps vs torch (AdamW, clip, decay on every parameter)")
    if not decay_all:
        print("  MUTATION: decay only on apply_decay params — must FAIL")
    var fails = _run["cpu"](dump, None, decay_all)
    var c = DeviceContext()
    fails += _run["gpu"](dump, Optional(c), decay_all)
    if fails > 0:
        raise Error("FAIL G6a: " + String(fails) + " check(s)")
    print("PASS")
