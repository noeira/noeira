"""P6.1 — short-run training parity: the torch run's steps, taken by ours.

docs/LEWM_REOPEN_PLAN.md P6. `tools/lewm/train_parity_ref.py` trained the
reference LeWM from its random init on the real PushT dataset and recorded the
init, the batches (clip indices), the action normaliser and every step's SIGReg
matrix; `convert_ref_to_ours.py --dump <run>` put the init in our names. This
takes the SAME steps with `ref_trainer.LeWMRefTrainer` — same init, same
windows read natively from the same HDF5 file, same matrices, AdamW lr 5e-5,
wd 1e-3 on every parameter, clip 1.0 — and prints both loss curves.

The two runs differ only by arithmetic (ours: TF32 GEMMs on a 5090; torch:
float32), so they agree closely for the first steps and then drift apart the
way two seeds do: the gate is on the CURVES (windowed means), not per step.
The band is torch's own: the same torch run with TF32 matmuls / convolutions
(`train_parity_ref.py --tf32`) sits up to 0.17 % from torch float32 over the
first 10 steps and up to 2.2 % on a 50-step window (measured 2026-10-02,
batch 32, 1000 steps). GATE: first 10 steps within 0.5 %, every window
within 3 %. Measured: 0.09 % and 1.7 % (mean window 0.59 %, torch-TF32's
0.69 %).

    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_train_parity.mojo \\
        --run /workspace/lewm_parity128 --h5 /workspace/.../pusht_expert_train.h5 [--steps N]
"""

from std.sys import argv
from std.math import abs, isnan
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_trainer import LeWMRefTrainer, REF_T, REF_IMG, REF_ACT_IN
from noeira.experimental.lewm.batch_loader import LewmBatchLoaderThread


comptime B = 128
"""The recipe's batch, and the torch run's (`--batch 128`). Until
`49a56a667` this graph could not take one step at 64 on a 32 GB 5090; with
the ViT blocks checkpointed it peaks at 17.2 GB."""
comptime HW = REF_IMG * REF_IMG
comptime PIX = REF_T * 3 * HW
comptime ACT = REF_T * REF_ACT_IN
comptime WINDOW = 50
comptime GATE_EARLY = 5e-3
comptime GATE_WINDOW = 3e-2


def _mean(v: List[Float64], a: Int, b: Int) -> Float64:
    var s = 0.0
    for i in range(a, b):
        s += v[i]
    return s / Float64(b - a)


def main() raises:
    var run = String("/workspace/lewm_parity128")
    var h5 = String("/workspace/lewm_session_a/stablewm/pusht_expert_train.h5")
    var n_steps = -1
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--run":
            run = String(args[i + 1]); i += 1
        elif a == "--h5":
            h5 = String(args[i + 1]); i += 1
        elif a == "--steps":
            n_steps = Int(String(args[i + 1])); i += 1
        i += 1

    var rd = RefDump(run)
    var hp = rd.get(String("run.hparams"))  # lr, wd, clip, steps, batch
    if Int(hp[4]) != B:
        raise Error("the torch run used batch " + String(Int(hp[4])) + ", this replay is built for " + String(B))
    var total = Int(hp[3])
    n_steps = total if n_steps < 0 else min(n_steps, total)
    var want = rd.get(String("run.scalars"))  # (steps, 5)
    var clip_idx = rd.get(String("run.clip_idx"))
    var a_mean = rd.get(String("run.action_mean"))
    var a_std = rd.get(String("run.action_std"))

    var c = DeviceContext()
    var ctx = Optional(c)
    var tr = LeWMRefTrainer["gpu", B](
        ctx, lr=Float64(hp[0]), wd=Float64(hp[1]), max_norm=Float64(hp[2])
    )
    var n = tr.load(run)
    print("P6.1 parity replay:", n_steps, "steps x batch", B, "; loaded", n, "tensors (torch's init)")

    var stages: List[Pointer[Scalar[DType.uint8], MutAnyOrigin]] = [tr.staging(0), tr.staging(1)]
    var idx = List[Int](capacity=n_steps * B)
    for k in range(n_steps * B):
        idx.append(Int(clip_idx[k]))
    var am: List[Float32] = [Float32(a_mean[0]), Float32(a_mean[1])]
    var asd: List[Float32] = [Float32(a_std[0]), Float32(a_std[1])]
    # the reads run on their own thread (batch_loader.mojo): a step's ~4,000
    # kernel launches hold this one
    var loader = LewmBatchLoaderThread[B](h5, idx, am, asd, stages)

    var ours = List[Float64]()
    var theirs = List[Float64]()
    var t_all = perf_counter_ns()
    loader.request(0)
    for s in range(n_steps):
        var t0 = perf_counter_ns()
        var t_next = loader.wait(s)
        var t_wait = Float64(perf_counter_ns() - t0) / 1e9
        # slot (s+1) % 2 was last read by step s-1, finished: the loader can
        # fill it while this thread is held by step s's kernel launches
        if s + 1 < n_steps:
            loader.request(s + 1)
        tr.set_sigreg_a(rd.get(String("run.A.") + String(s)))
        tr.submit_staged(loader.actions(s), s % 2)
        var st = tr.finish()
        if s == 0:
            # the device normalisation against torch's float32 order on the
            # host (u8 / 255, then (x - mean) / std): must be bit-equal
            var im_mean: List[Float32] = [0.485, 0.456, 0.406]
            var im_std: List[Float32] = [0.229, 0.224, 0.225]
            var stage = stages[0]
            tr.pix.download(c)
            var diff = 0
            for f in range(B * REF_T):
                for p in range(HW):
                    for ch in range(3):
                        var x = Float32(stage[(f * HW + p) * 3 + ch]) / Float32(255.0)
                        var want = (x - im_mean[ch]) / im_std[ch]
                        if rebind[Float32](tr.pix.data[(f * 3 + ch) * HW + p]) != want:
                            diff += 1
            print("  step 0: device pixel normalisation vs host float32:", diff, "of", B * PIX, "differ")
            if diff > 0:
                raise Error("the device pixel normalisation is not torch's")
        ours.append(st.loss)
        theirs.append(Float64(want[s * 5]))
        if s % 25 == 0 or s == n_steps - 1 or s < 5:
            print(
                "  step", s, " loss ours", Float32(st.loss), "torch", Float32(want[s * 5]),
                "  pred", Float32(st.pred_loss), Float32(want[s * 5 + 1]),
                "  sigreg", Float32(st.sigreg_loss), Float32(want[s * 5 + 2]),
                "  norm", Float32(st.grad_norm), Float32(want[s * 5 + 3]),
                "  (wall", Float32(Float64(perf_counter_ns() - t0) / 1e9), "s: waited",
                Float32(t_wait), "s for a batch the loader read in", Float32(t_next), "s)",
            )

    print("  total", Float64(perf_counter_ns() - t_all) / 1e9, "s")
    loader.stop()
    var early = 0.0
    for s in range(min(10, n_steps)):
        early = max(early, abs(ours[s] - theirs[s]) / theirs[s])
    print("  first", min(10, n_steps), "steps: max |Δloss| / loss =", early)
    print("  window means (", WINDOW, "steps): ours / torch / rel")
    var worst = 0.0
    var w = 0
    while (w + 1) * WINDOW <= n_steps:
        var mo = _mean(ours, w * WINDOW, (w + 1) * WINDOW)
        var mt = _mean(theirs, w * WINDOW, (w + 1) * WINDOW)
        var rel = abs(mo - mt) / mt
        worst = max(worst, rel)
        print("    ", w * WINDOW, "-", (w + 1) * WINDOW - 1, " ", Float32(mo), " ", Float32(mt), " ", Float32(rel))
        w += 1
    print("  worst window rel", worst)
    var csv = String("step,ours,torch\n")
    for s in range(n_steps):
        csv += String(s) + "," + String(ours[s]) + "," + String(theirs[s]) + "\n"
    with open(run + "/ours_vs_torch.csv", "w") as f:
        f.write(csv)
    if early > GATE_EARLY or worst > GATE_WINDOW:
        raise Error("FAIL P6.1: first steps " + String(early) + " (gate " + String(GATE_EARLY)
                    + "), worst window " + String(worst) + " (gate " + String(GATE_WINDOW) + ")")
    print("PASS")
