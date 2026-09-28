"""Is `F` running away? Read it off the checkpoint ladder.

The 484 k-step run left one question open. Its `measure` loss drifted
-129 (2-50 k) -> -148 (102-150 k) -> -305 (366-400 k) -> -296 (452-484 k), and
with `||B||` pinned at sqrt(d) that can only come from the anchor term
`-2*E[F.B]` growing, i.e. from `||F||` growing. Two readings fit:

  * `F` LEARNING to align with `B` — expected, healthy, self-limiting.
  * `F` RUNNING AWAY — §13's defect 3, where `F` is the unconstrained half of
    the pair and the measure loss is unbounded below when it grows.

The log could not separate them, because `FBLosses` computes `f_norm` and the
training script never printed it (now fixed). The checkpoints can, and they do
not need the dataset or a GPU.

## What is measured, and which number actually answers it

  **Parameter norms** (`sqrt(sum over params of ||w||^2)`, per net). This is
  the direct reading: "running away" IS the weights growing without bound. It
  is input-independent, so no dataset is required and there is nothing to
  argue about.

  **Output norms** on a FIXED synthetic batch, comparable in TREND to the
  logged `f_norm`.

  ⚠ The synthetic batch is N(0,1), NOT the walker state distribution, so the
  ABSOLUTE output norms will not match the logged values and are not meant to.
  Only the trend across checkpoints is meaningful, and it is meaningful because
  every checkpoint sees the identical input.

  **`||B(s)||` as a cross-check.** It must read sqrt(128) = 11.3137 for EVERY
  checkpoint. `LayerNormNoAffine` pins it, so any deviation means either the
  load silently failed or the eps floor was being approached at that point in
  the run — and a load that silently produced garbage would otherwise make the
  `F` numbers look like data.

⚠ Only reads `fb_walker_all_d128.ckpt.*`. The older `fb_walker_d128.ckpt.*`
files are from the LEARNABLE-`LayerNorm` run and have a different BNet
parameter set; loading them here would fail or, worse, partially match.

Run:
    pixi run mojo run -I . examples/fb/fb_inspect_ladder.mojo
    pixi run mojo run -I . examples/fb/fb_inspect_ladder.mojo --run <run_id>
"""

from std.math import sqrt
from std.random import random_float64, seed
from std.sys import argv
from std.os.path import exists
from noeira.core.project import projects_root
from noeira.io.proc import quote_arg, run_capture

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.module import Module
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Xavier
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.deep_agents.fb.trainer import FBTrainer

from max.gpu.host import DeviceContext


# ── must match `fb_train_gpu.mojo` exactly ───────────────────────────────
comptime NQ: Int = 9
comptime NV: Int = 9
comptime NACT: Int = 6
comptime OBS: Int = NQ + NV
comptime D: Int = 128
comptime HID: Int = 1024
comptime BATCH: Int = 256  # probe batch; the checkpoint does not encode it

comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D

comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, NACT, D, BATCH, "cpu"]

comptime CKPT_DIR: StaticString = "checkpoints/"
# ⚠⚠ THIS READS THE HISTORICAL LADDER IN `checkpoints/`, WHICH NEW RUNS NO LONGER
# WRITE. Since 2026-09-09 (`docs/PROJECT_LAYER_PLAN.md` P0d) the FB trainers write
# `runs/<id>/checkpoints/step_<n>.ckpt` instead — a unique directory per run, so
# two arms at the same `--tag` cannot overwrite each other. The files this points
# at still exist and still load; a ladder from a NEW run is read with
# `--run <run_id>`.
comptime CKPT_STEM: StaticString = "fb_walker_all_d128.ckpt."
comptime N_CKPT: Int = 10
comptime FIRST_STEP: Int = 50_000
comptime STEP_STRIDE: Int = 50_000
comptime SEED: Int = 424242


struct ParamNorm(ParamVisitor):
    """Accumulates `sum ||w||^2` over every Param of a module."""

    var sq: Float64
    var n: Int

    def __init__(out self):
        self.sq = 0.0
        self.n = 0

    def __init__(out self, *, deinit move: Self):
        self.sq = move.sq
        self.n = move.n

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            var x = Float64(param.data[i])
            self.sq += x * x
        self.n += N


def _wnorm[M: Module](mut net: M) raises -> Float64:
    var pn = ParamNorm()
    net.for_each_param["cpu"](pn, None)
    return sqrt(pn.sq)


def _rms(ref t: Tensor, n: Int, rows: Int) -> Float64:
    """Row-mean L2 norm — the same shape of quantity `FBLosses` reports."""
    var s = Float64(0)
    for i in range(n):
        var x = Float64(t.data[i])
        s += x * x
    return sqrt(s / Float64(rows))


def _run_dir(handle: String) raises -> String:
    """`handle` as a run directory: a path to one, or a run id under `runs/` or
    any `projects/*/runs/`. Raises when none of them holds a `checkpoints/`."""
    if exists(handle + "/checkpoints"):
        return handle
    if exists(String("runs/") + handle + "/checkpoints"):
        return String("runs/") + handle
    var root = projects_root()
    var listing = run_capture(
        String("ls -d ") + quote_arg(root) + "/*/runs/" + quote_arg(handle)
        + " 2>/dev/null; true",
        1 << 16,
    )
    for line in listing.split("\n"):
        var d = String(String(line).strip())
        if d.byte_length() > 0 and exists(d + "/checkpoints"):
            return d
    raise Error(
        "--run '" + handle + "' is neither a run directory nor a run id under"
        " runs/ or " + root + "/*/runs/"
    )


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    seed(SEED)
    print("=" * 78)
    print("FB checkpoint ladder — is F running away?")
    print("=" * 78)

    # `--run <run_id|run_dir>` reads a NEW run's `checkpoints/step_<n>.ckpt`
    # ladder; without it, the historical `CKPT_DIR + CKPT_STEM + <n>` one.
    var run_arg = _flag(String("--run"), String(""))
    var run_dir = _run_dir(run_arg) if run_arg.byte_length() > 0 else String("")
    if run_dir.byte_length() > 0:
        print("  run:", run_dir)

    # One FIXED probe batch, drawn before any load, reused for every rung.
    var ps = Tensor.alloc(BATCH * OBS)
    var pa = Tensor.alloc(BATCH * NACT)
    var pz = Tensor.alloc(BATCH * D)
    for i in range(BATCH * OBS):
        ps.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    for i in range(BATCH * NACT):
        pa.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    # z on the sqrt(d) sphere, as the sampler produces it.
    for b in range(BATCH):
        var s2 = Float64(0)
        for k in range(D):
            var g = random_float64() * 2.0 - 1.0
            pz.data[b * D + k] = Scalar[DT](g)
            s2 += g * g
        var inv = sqrt(Float64(D)) / sqrt(s2) if s2 > 0 else 0.0
        for k in range(D):
            pz.data[b * D + k] = Scalar[DT](
                Float64(pz.data[b * D + k]) * inv
            )

    print("")
    print("  step     |W_F|     |W_B|     |W_A|      |F(x)|     |B(s)|")
    print("  " + "-" * 60)

    var hist_wf = List[Float64]()
    var hist_wb = List[Float64]()
    var hist_wa = List[Float64]()
    var first_fo = Float64(0)
    var last_fo = Float64(0)
    var worst_bdev = Float64(0)
    var loaded = 0

    for k in range(N_CKPT):
        var at = FIRST_STEP + k * STEP_STRIDE
        var path = String(CKPT_DIR) + String(CKPT_STEM) + String(at)
        if run_dir.byte_length() > 0:
            path = run_dir + "/checkpoints/step_" + String(at) + ".ckpt"

        var t = Trainer.make[Xavier](lr=1e-4, max_grad_norm=1.0, bc_weight=1.0)
        try:
            t.load_state(path)
        except e:
            print("  ", at, " MISSING/UNREADABLE:", e)
            continue

        var wf = _wnorm(t.f1.online)
        var wb = _wnorm(t.bnet.online)
        var wa = _wnorm(t.actor.online)

        var b_out = Tensor()
        t.backward_embed[BATCH](ps, b_out)
        var bo = _rms(b_out, BATCH * D, BATCH)

        var f_out = Tensor()
        t.forward_f[BATCH](ps, pa, pz, f_out)
        var fo = _rms(f_out, BATCH * D, BATCH)

        print(
            "  ", at, "  ", wf, "  ", wb, "  ", wa, "   ", fo, "  ", bo,
        )

        hist_wf.append(wf)
        hist_wb.append(wb)
        hist_wa.append(wa)
        if loaded == 0:
            first_fo = fo
        last_fo = fo
        var dev = bo - sqrt(Float64(D))
        if dev < 0:
            dev = -dev
        if dev > worst_bdev:
            worst_bdev = dev
        loaded += 1

    print("")
    print("=" * 78)
    if loaded < 2:
        print("Not enough rungs loaded (", loaded, ") — nothing to compare.")
        return

    print("  |B(s)| worst deviation from sqrt(128) =", worst_bdev)
    if worst_bdev > 0.05:
        print("  ⚠⚠ B is NOT on the sqrt(d) sphere. Either the architecture")
        print("     here does not match the checkpoint's, or the eps floor was")
        print("     hit during the run. Treat the F numbers as unverified.")
        return
    print("     (pinned — the loads are sound and LayerNormNoAffine held)")
    print("")

    var n = len(hist_wf)
    var wf_ratio = hist_wf[n - 1] / hist_wf[0]
    var wb_ratio = hist_wb[n - 1] / hist_wb[0]
    var wa_ratio = hist_wa[n - 1] / hist_wa[0]
    var fo_ratio = last_fo / first_fo if first_fo > 0 else 0.0

    # ⚠⚠ The ENDPOINT RATIO ALONE IS NOT A VERDICT, and an earlier version of
    # this script issued one from it: it read |W_F| 52.5 -> 107.0 = 2.04x,
    # compared that to the measure loss roughly doubling, and declared "F IS
    # RUNNING AWAY". Its own table refuted it two lines above. Runaway
    # ACCELERATES; F's per-interval growth COLLAPSED from +17.4 to +2.3 per
    # 50 k, i.e. 13% of its initial rate. That is convergence.
    #
    # The second thing the ratio hides: B grew MORE (3.44x) and is
    # ACCELERATING (+4.4 -> +6.4), while B's output is provably pinned at
    # sqrt(d). So the fastest-growing net is the one whose growth is
    # definitionally harmless, which is what generic unregularised Adam drift
    # looks like — not an F-specific pathology.
    print("  per-50k growth in |W_F| (runaway ACCELERATES, convergence decays):")
    var first_d = hist_wf[1] - hist_wf[0]
    var last_d = hist_wf[n - 1] - hist_wf[n - 2]
    for i in range(1, n):
        print("     rung", i, " d|W_F| =", hist_wf[i] - hist_wf[i - 1])
    var decay = last_d / first_d if first_d > 0 else 1.0
    print("")
    print("  |W_F| ", hist_wf[0], "->", hist_wf[n - 1], " ratio", wf_ratio)
    print("  |W_B| ", hist_wb[0], "->", hist_wb[n - 1], " ratio", wb_ratio,
          " (output PINNED, so this growth is inert)")
    print("  |W_A| ", hist_wa[0], "->", hist_wa[n - 1], " ratio", wa_ratio)
    print("  |F(x)|", first_fo, "->", last_fo, " ratio", fo_ratio)
    print("  growth-rate decay (last interval / first) =", decay)
    print("")

    var accelerating = decay > 0.8
    var f_is_outlier = wf_ratio > 1.3 * wb_ratio and wf_ratio > 1.3 * wa_ratio
    if accelerating and f_is_outlier:
        print("  VERDICT: F IS RUNNING AWAY — growth is not decaying (",
              decay, ") and F outpaces both B and A.")
        print("  §13 defect 3. MAX_GRAD_NORM bounds the STEP, not the weights;")
        print("  add weight decay on F, or a norm constraint like B's.")
    elif accelerating:
        print("  VERDICT: ALL nets are growing without decay. Not F-specific,")
        print("  but unbounded — add weight decay before a longer run.")
    else:
        print("  VERDICT: F is NOT running away. Its growth rate decayed to",
              decay, "of the initial, and B — whose output is PINNED, so whose")
        print("  growth cannot matter — grew MORE (", wb_ratio,
              "x vs", wf_ratio, "x).")
        print("  The measure drift is F converging toward its target, not")
        print("  diverging. What the table does show is that NO net has weight")
        print("  regularisation, so all three drift up under Adam; benign here,")
        print("  worth revisiting only if a run goes much longer than 2 M.")
    print("=" * 78)
