"""Fit a `BcNet` to a `BcDataset` — normalise, baseline, fit, save, re-score.

⚠⚠ THE BASELINES ARE PRINTED BESIDE THE RESULT. A validation MSE on its own
is unreadable, so two references come with it: predicting ZERO and
predicting the TRAINING MEAN action. A fit that does not beat the mean has
learned nothing, and `fit_bc` raises rather than reporting a number that
looks like progress.

⚠⚠ THE CHECKPOINT IS RE-SCORED ON THE CPU, through the path a driver loads
it by. The fit lives on the device; what a driver runs is the file reloaded
into a CPU network. A save that wrote the wrong copy, a load that filled
nothing, a shape that drifted — each leaves a policy that trains well and
acts nothing like it. The reloaded number is the one that counts.
"""

from std.os import makedirs
from std.os.path import dirname, exists
from std.random import seed as seed_rng, random_ui64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.checkpoint import save_params, load_params
from noeira.nn.core.initializer import Kaiming
from noeira.nn.optimizer.adam import Adam
from .policy import BcNet, BC_HID, write_bc_norm
from .dataset import BcDataset


struct BcFitReport(Copyable, Movable, ImplicitlyCopyable):
    var best_val: Float64
    """Best validation MSE over the epochs, on the device."""
    var reloaded_val: Float64
    """Validation MSE of the saved checkpoint, reloaded on the CPU."""
    var mse_zero: Float64
    var mse_mean: Float64

    def __init__(
        out self, best_val: Float64, reloaded_val: Float64,
        mse_zero: Float64, mse_mean: Float64,
    ):
        self.best_val = best_val
        self.reloaded_val = reloaded_val
        self.mse_zero = mse_zero
        self.mse_mean = mse_mean


def _pad(s: String, n: Int) -> String:
    var out = String(s)
    if out.byte_length() > n:
        return String(out[byte = 0 : n])
    while out.byte_length() < n:
        out += " "
    return out^


def fit_bc[OBS: Int, ACT: Int, BATCH: Int](
    mut data: BcDataset,
    epochs: Int,
    lr: Float64,
    out_path: String,
    who: String = "bc",
    seed: Int = 7,
) raises -> BcFitReport:
    """Standardise the inputs with the TRAINING rows' statistics (applied to
    both halves, in place), fit `BcNet[OBS, ACT]` with Adam on MSE, write the
    checkpoint and its `.norm` sidecar to `out_path`, re-score the reloaded
    checkpoint on the CPU. Raises if the reload does not reproduce the fit or
    the fit does not beat the training-mean baseline.

    ⚠ `BATCH` IS COMPTIME: `forward` / `vjp` take the batch as a parameter,
    so one build has one batch size."""
    if data.obs_dim != OBS or data.act_dim != ACT:
        raise Error(
            who + ": the dataset is " + String(data.obs_dim) + " / "
            + String(data.act_dim) + ", the net " + String(OBS) + " / "
            + String(ACT)
        )
    if data.n_tr == 0 or data.n_va == 0:
        raise Error(who + ": an empty split — nothing to fit or score")
    var n_tr = data.n_tr
    var n_va = data.n_va

    # ── normalisation, from the TRAINING rows only ────────────────────────
    # ⚠ THE VALIDATION ROWS DO NOT CONTRIBUTE. A mean that saw them is a leak,
    # small here and free to avoid.
    var mu = List[Float64](length=OBS, fill=0.0)
    var sd = List[Float64](length=OBS, fill=0.0)
    for r in range(n_tr):
        for j in range(OBS):
            mu[j] += Float64(data.x_tr[r * OBS + j])
    for j in range(OBS):
        mu[j] /= Float64(n_tr)
    for r in range(n_tr):
        for j in range(OBS):
            var dv = Float64(data.x_tr[r * OBS + j]) - mu[j]
            sd[j] += dv * dv
    for j in range(OBS):
        sd[j] = (sd[j] / Float64(n_tr)) ** 0.5
        # ⚠ A CONSTANT COLUMN KEEPS SCALE 1: dividing by its zero spread would
        # write inf into every row (a mask word that is 1 on every task, a
        # fixture's untouched joint).
        if sd[j] < 1.0e-6:
            sd[j] = 1.0
    for r in range(n_tr):
        for j in range(OBS):
            data.x_tr[r * OBS + j] = Scalar[DT](
                (Float64(data.x_tr[r * OBS + j]) - mu[j]) / sd[j]
            )
    for r in range(n_va):
        for j in range(OBS):
            data.x_va[r * OBS + j] = Scalar[DT](
                (Float64(data.x_va[r * OBS + j]) - mu[j]) / sd[j]
            )

    # ── the two baselines every epoch is read against ─────────────────────
    var mean_act = List[Float64](length=ACT, fill=0.0)
    for r in range(n_tr):
        for j in range(ACT):
            mean_act[j] += Float64(data.y_tr[r * ACT + j])
    for j in range(ACT):
        mean_act[j] /= Float64(n_tr)
    var mse_zero = 0.0
    var mse_mean = 0.0
    for r in range(n_va):
        for j in range(ACT):
            var a = Float64(data.y_va[r * ACT + j])
            mse_zero += a * a
            var dm = a - mean_act[j]
            mse_mean += dm * dm
    mse_zero /= Float64(n_va * ACT)
    mse_mean /= Float64(n_va * ACT)
    print("  baselines (val MSE): predict ZERO", mse_zero,
          "| predict the TRAINING MEAN", mse_mean)

    # ── the fit ───────────────────────────────────────────────────────────
    seed_rng(seed)
    var ctx = DeviceContext()
    var net = BcNet[OBS, ACT].make["gpu", Kaiming](Optional(ctx))
    var opt = Adam(lr=Scalar[DT](lr))
    # ⚠ HOST-BACKED WITH A DEVICE CELL: `Tensor.alloc(n)` then `upload`, the
    # shape every nn driver uses — the batch is built on the host and pushed.
    var bx = Tensor.alloc(BATCH * OBS)
    var by = Tensor.alloc(BATCH * ACT)
    var pred = Tensor.alloc(BATCH * ACT)
    var gout = Tensor.alloc(BATCH * ACT)
    var gin = Tensor.alloc(BATCH * OBS)
    bx.upload(ctx)
    by.upload(ctx)
    pred.upload(ctx)
    gout.upload(ctx)
    gin.upload(ctx)
    var n_batches = n_tr // BATCH
    var best_val = 1.0e30
    print("  net   : ", OBS, "->", BC_HID, "->", BC_HID, "->", ACT, "| Adam lr",
          lr, "| batch", BATCH, "|", n_batches, "batches/epoch")

    for ep in range(epochs):
        var tr_loss = 0.0
        for _ in range(n_batches):
            # BC rows are i.i.d. only across demos, so the batch is sampled
            # row-wise with replacement
            for r in range(BATCH):
                var src = Int(random_ui64(0, UInt64(n_tr - 1)))
                for j in range(OBS):
                    bx.data[r * OBS + j] = data.x_tr[src * OBS + j]
                for j in range(ACT):
                    by.data[r * ACT + j] = data.y_tr[src * ACT + j]
            bx.upload(ctx)
            by.upload(ctx)
            net.forward["gpu", BATCH](TensorRefs[1](bx), pred, Optional(ctx))
            pred.download(ctx)
            ctx.synchronize()
            # MSE and its gradient on the host: BATCH x ACT words per step,
            # and it keeps the loss arithmetic in one readable place.
            var loss = 0.0
            for k in range(BATCH * ACT):
                var e2 = Float64(pred.data[k]) - Float64(by.data[k])
                loss += e2 * e2
                gout.data[k] = Scalar[DT](2.0 * e2 / Float64(BATCH * ACT))
            tr_loss += loss / Float64(BATCH * ACT)
            gout.upload(ctx)
            net.zero_grad["gpu"](Optional(ctx))
            net.vjp["gpu", BATCH](
                TensorRefs[1](bx), gout, TensorRefs[1](gin), Optional(ctx)
            )
            # ⚠⚠ `opt.step`, NOT A BARE PARAM WALK. `step` bumps every param
            # value's `version` after updating it, and `Linear`'s GPU forward
            # re-pads its cached weight ONLY when that version moves
            # (`_ensure_w_pad`). Driving the walk directly updates `val` and
            # leaves the pad at the INITIAL weights: the padded input layer
            # trains against a frozen forward while `val` drifts away. Measured
            # on LIBERO: reloaded val MSE 0.305 against the fit's 0.132.
            opt.step["gpu"](net, Optional(ctx))

        # ── validation, in whole batches ──────────────────────────────────
        var va_loss = 0.0
        var va_batches = n_va // BATCH
        for b in range(va_batches):
            for r in range(BATCH):
                var src = b * BATCH + r
                for j in range(OBS):
                    bx.data[r * OBS + j] = data.x_va[src * OBS + j]
                for j in range(ACT):
                    by.data[r * ACT + j] = data.y_va[src * ACT + j]
            bx.upload(ctx)
            net.forward["gpu", BATCH](TensorRefs[1](bx), pred, Optional(ctx))
            pred.download(ctx)
            ctx.synchronize()
            for k in range(BATCH * ACT):
                var e2 = Float64(pred.data[k]) - Float64(by.data[k])
                va_loss += e2 * e2
        va_loss /= Float64(va_batches * BATCH * ACT)
        print("   epoch", _pad(String(ep), 3), " train MSE",
              tr_loss / Float64(n_batches), " val MSE", va_loss,
              " (zero", mse_zero, ", mean", mse_mean, ")")
        if va_loss < best_val:
            best_val = va_loss

    # ── the artefact ──────────────────────────────────────────────────────
    var od = dirname(out_path)
    if od != "" and not exists(od):
        makedirs(od)
    save_params["gpu"](net, out_path, Optional(ctx))
    # ⚠ THE NORMALISATION IS PART OF THE POLICY: a driver reads this sidecar
    # and refuses a checkpoint without it.
    write_bc_norm(out_path + ".norm", mu, sd, ACT)
    print()
    print("  wrote", out_path, "and", out_path + ".norm")

    # ── the checkpoint, reloaded on the CPU and re-scored ─────────────────
    var check = BcNet[OBS, ACT].make["cpu", Kaiming](None)
    _ = load_params["cpu"](check, out_path, None)
    var cx = Tensor.alloc(BATCH * OBS)
    var cy = Tensor.alloc(BATCH * ACT)
    var cy_pred = Tensor.alloc(BATCH * ACT)
    var re_loss = 0.0
    var re_abs = 0.0
    var re_batches = n_va // BATCH
    for b in range(re_batches):
        for r in range(BATCH):
            var src = b * BATCH + r
            for j in range(OBS):
                cx.data[r * OBS + j] = data.x_va[src * OBS + j]
            for j in range(ACT):
                cy.data[r * ACT + j] = data.y_va[src * ACT + j]
        check.forward["cpu", BATCH](TensorRefs[1](cx), cy_pred, None)
        for k in range(BATCH * ACT):
            var e2 = Float64(cy_pred.data[k]) - Float64(cy.data[k])
            re_loss += e2 * e2
            re_abs += abs(Float64(cy_pred.data[k]))
    re_loss /= Float64(re_batches * BATCH * ACT)
    re_abs /= Float64(re_batches * BATCH * ACT)
    var tgt_abs = 0.0
    for r in range(re_batches * BATCH):
        for j in range(ACT):
            tgt_abs += abs(Float64(data.y_va[r * ACT + j]))
    tgt_abs /= Float64(re_batches * BATCH * ACT)
    print("  reloaded on CPU: val MSE", re_loss, "| mean |a| predicted",
          re_abs, "recorded", tgt_abs)

    # ── the verdict ───────────────────────────────────────────────────────
    print()
    if re_loss > best_val * 1.5 + 1.0e-9:
        print("  FAIL: the RELOADED checkpoint scores", re_loss,
              "against the fit's", best_val, "— the file is not the network"
              " that trained")
        raise Error(who + ": the checkpoint does not reproduce the fit")
    if best_val >= mse_mean:
        print("  FAIL: val MSE", best_val, ">= the TRAINING-MEAN baseline",
              mse_mean, "— the fit learned nothing")
        raise Error(who + ": the fit does not beat its baseline")
    return BcFitReport(best_val, re_loss, mse_zero, mse_mean)
