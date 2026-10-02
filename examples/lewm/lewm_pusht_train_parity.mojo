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
        --run /workspace/lewm_parity32 --h5 /workspace/.../pusht_expert_train.h5 [--steps N]
"""

from std.sys import argv
from std.math import abs, isnan
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.datasets.lewm_pusht import LewmPushTExpert
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_trainer import LeWMRefTrainer, REF_T, REF_IMG, REF_ACT_IN


comptime B = 32
"""The torch run's batch (`--batch 32`). Not the recipe's 128: one step of
this graph allocates 31.5 GB at batch 64 (`NOEIRA_ALLOC_TRACE=1`: a persistent
gradient buffer beside every activation, attention scores and FFN hiddens
4.6 GB each, K-padded Linear inputs 3.9 GB) and the 5090 has 32; torch fits
128. The memory is P6.2's question; parity does not depend on the batch."""
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
    var run = String("/workspace/lewm_parity32")
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
    var ds = LewmPushTExpert(frameskip=5, num_steps=REF_T, path=h5)
    print("  dataset:", len(ds), "clips")

    var u8 = List[UInt8](length=REF_T * HW * 3, fill=0)
    var dense = List[UInt8](length=REF_T * 5 * HW * 3, fill=0)
    var araw = List[Float32](length=ACT, fill=0)
    var pix = List[Scalar[DT]](length=B * PIX, fill=0)
    var act = List[Scalar[DT]](length=B * ACT, fill=0)
    var im_mean: List[Float32] = [0.485, 0.456, 0.406]
    var im_std: List[Float32] = [0.229, 0.224, 0.225]

    var ours = List[Float64]()
    var theirs = List[Float64]()
    var t_all = perf_counter_ns()
    for s in range(n_steps):
        var t0 = perf_counter_ns()
        for b in range(B):
            ds.sample_clip_pixels_uint8(
                Int(clip_idx[s * B + b]),
                rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](u8.unsafe_ptr()),
                rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](araw.unsafe_ptr()),
                rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](dense.unsafe_ptr()),
            )
            # torch: u8.float() / 255, permute to CHW, (x - mean) / std — float32
            for t in range(REF_T):
                for p in range(HW):
                    for ch in range(3):
                        var x = Float32(u8[(t * HW + p) * 3 + ch]) / Float32(255.0)
                        pix[b * PIX + (t * 3 + ch) * HW + p] = rebind[Scalar[DT]](
                            (x - im_mean[ch]) / im_std[ch]
                        )
            # (20, 2) dense = (4, 10) row-major; z-score per raw dim, NaN -> 0
            for k in range(ACT):
                var z = (araw[k] - Float32(a_mean[k % 2])) / Float32(a_std[k % 2])
                act[b * ACT + k] = Scalar[DT](0.0) if isnan(z) else rebind[Scalar[DT]](z)
        var t_data = perf_counter_ns()
        tr.set_sigreg_a(rd.get(String("run.A.") + String(s)))
        var st = tr.train_step(pix, act)
        ours.append(st.loss)
        theirs.append(Float64(want[s * 5]))
        if s % 25 == 0 or s == n_steps - 1:
            print(
                "  step", s, " loss ours", Float32(st.loss), "torch", Float32(want[s * 5]),
                "  pred", Float32(st.pred_loss), Float32(want[s * 5 + 1]),
                "  sigreg", Float32(st.sigreg_loss), Float32(want[s * 5 + 2]),
                "  norm", Float32(st.grad_norm), Float32(want[s * 5 + 3]),
                "  (data", Float32(Float64(t_data - t0) / 1e9), "s, step",
                Float32(Float64(perf_counter_ns() - t_data) / 1e9), "s)",
            )

    print("  total", Float64(perf_counter_ns() - t_all) / 1e9, "s")
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
