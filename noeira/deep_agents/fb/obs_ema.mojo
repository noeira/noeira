"""`ObsEma[OBS]` — BFM-Zero's observation normaliser: `BatchNorm1d(affine=False,
momentum=0.01)` run the way the reference runs it (G3.2).

`agents/normalizers.py` + `fb_cpr_aux/agent.py::update`: every training
batch's `obs` and `next_obs` UPDATE the running statistics (train mode),
then obs, next_obs, the expert rows and the rollout observation are
normalised with the running statistics in eval mode:

    running_mean ← (1 − m) · running_mean + m · mean_batch
    running_var  ← (1 − m) · running_var  + m · var_batch  (UNBIASED, n/(n−1))
    y = (x − running_mean) / sqrt(running_var + 1e-5)

with m = 0.01 and torch's BatchNorm1d eps 1e-5. Initial state: mean 0,
var 1. The replay ring and the expert table keep RAW rows; the statistics
are applied at gather time, so a moving normaliser never rewrites what was
stored (`online.mojo` header: "a running normaliser under a TD bootstrap is
a moving target that needs its own gate" — this is that gate's subject,
`tests/fb/test_obs_ema_vs_batchnorm.mojo`).

Why an EMA and not `core/obs_norm.mojo`'s count-based (Chan) merge: the
reference's statistics track the RECENT batches with a time constant of
100 updates; a cumulative average converges to the whole-run mean and
lags a distribution that the reference-state init and the lie-down mix
keep moving. Same operation on the two devices, one launch each:

    ema_update_kernel[ROWS, OBS]   one thread per dim, reduces ROWS rows
    ema_apply_kernel[ROWS, OBS]    one thread per element, in place

Both are RNG-free and allocation-free, so they sit inside a captured train
step. The host mirror (`sync_host`) serves the single-row greedy path and
the `.norm` sidecar (`ObsNorm`'s format: `N` then `mu sd var` per line, `sd =
sqrt(var + eps)`, then `n_updates <k>`), which the evals load with
`ObsNorm.try_load` unchanged — it reads the first two columns.
"""

from max.gpu import global_idx
from std.math import sqrt
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.ptr import mptr
from noeira.io.fileio import write_text_atomic

from .kernels import ensure_t, _blocks


comptime OBS_EMA_EPS: Float64 = 1e-5      # torch BatchNorm1d default
comptime OBS_EMA_MOMENTUM: Float64 = 0.01  # `BatchNormNormalizerConfig.momentum`


def ema_update_kernel[ROWS: Int, OBS: Int](
    rows: Pointer[Scalar[DT], MutAnyOrigin],
    mean: Pointer[Scalar[DT], MutAnyOrigin],
    var_: Pointer[Scalar[DT], MutAnyOrigin],
    momentum: Scalar[DT],
):
    """One thread per dim: batch mean and UNBIASED variance over ROWS rows,
    then the EMA step — `BatchNorm1d`'s running-statistics update."""
    var d = Int(global_idx.x)
    if d >= OBS:
        return
    var s = Scalar[DT](0)
    var ss = Scalar[DT](0)
    for r in range(ROWS):
        var x = rows[unsafe_offset=r * OBS + d]
        s += x
        ss += x * x
    var n = Scalar[DT](ROWS)
    var mb = s / n
    var vb = (ss - n * mb * mb) / Scalar[DT](ROWS - 1)
    if vb < Scalar[DT](0):
        vb = Scalar[DT](0)
    var one_m = Scalar[DT](1) - momentum
    mean[unsafe_offset=d] = one_m * mean[unsafe_offset=d] + momentum * mb
    var_[unsafe_offset=d] = one_m * var_[unsafe_offset=d] + momentum * vb


def ema_apply_kernel[ROWS: Int, OBS: Int](
    rows: Pointer[Scalar[DT], MutAnyOrigin],
    mean: Pointer[Scalar[DT], MutAnyOrigin],
    var_: Pointer[Scalar[DT], MutAnyOrigin],
    eps: Scalar[DT],
):
    """In place: `x ← (x − mean) / sqrt(var + eps)`, eval-mode BatchNorm."""
    var i = Int(global_idx.x)
    if i >= ROWS * OBS:
        return
    var d = i % OBS
    rows[unsafe_offset=i] = (rows[unsafe_offset=i] - mean[unsafe_offset=d]) / sqrt(
        var_[unsafe_offset=d] + eps
    )


def ema_copy_apply_kernel[ROWS: Int, OBS: Int](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    mean: Pointer[Scalar[DT], MutAnyOrigin],
    var_: Pointer[Scalar[DT], MutAnyOrigin],
    eps: Scalar[DT],
):
    """`dst ← normalise(src)` — the rollout path, which must not touch the
    env's observation buffer."""
    var i = Int(global_idx.x)
    if i >= ROWS * OBS:
        return
    var d = i % OBS
    dst[unsafe_offset=i] = (src[unsafe_offset=i] - mean[unsafe_offset=d]) / sqrt(
        var_[unsafe_offset=d] + eps
    )


struct ObsEma[OBS: Int](Movable, Deinitable):
    var mean: Tensor       # OBS, device + host mirror
    var var_: Tensor       # OBS
    var enabled: Bool
    var momentum: Float64
    var n_updates: Int
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.mean = Tensor()
        self.var_ = Tensor()
        self.enabled = False
        self.momentum = OBS_EMA_MOMENTUM
        self.n_updates = 0
        self.ctx = None

    def __init__(out self, *, deinit move: Self):
        self.mean = move.mean^
        self.var_ = move.var_^
        self.enabled = move.enabled
        self.momentum = move.momentum
        self.n_updates = move.n_updates
        self.ctx = move.ctx^

    @staticmethod
    def make(
        ctx: DeviceContext, *, momentum: Float64 = OBS_EMA_MOMENTUM
    ) raises -> Self:
        """Mean 0, variance 1 on device (BatchNorm's initial running stats)."""
        var e = Self()
        e.ctx = Optional[DeviceContext](ctx)
        e.enabled = True
        e.momentum = momentum
        ensure_t["gpu"](e.mean, Self.OBS, e.ctx)
        ensure_t["gpu"](e.var_, Self.OBS, e.ctx)
        for d in range(Self.OBS):
            e.mean.data[d] = Scalar[DT](0)
            e.var_.data[d] = Scalar[DT](1)
        e.mean.upload(ctx)
        e.var_.upload(ctx)
        return e^

    # ── device ────────────────────────────────────────────────────────────
    def update[ROWS: Int](mut self, mut rows: Tensor) raises:
        """Merge ROWS raw rows (device) into the running statistics."""
        var c = self.ctx.value()
        c.enqueue_function[ema_update_kernel[ROWS, Self.OBS]](
            mptr(rows.dev.value().unsafe_ptr()),
            mptr(self.mean.dev.value().unsafe_ptr()),
            mptr(self.var_.dev.value().unsafe_ptr()),
            Scalar[DT](self.momentum),
            grid_dim=_blocks(Self.OBS), block_dim=TPB,
        )
        self.n_updates += 1

    def apply[ROWS: Int](self, mut rows: Tensor) raises:
        """Normalise ROWS rows in place (device)."""
        var c = self.ctx.value()
        c.enqueue_function[ema_apply_kernel[ROWS, Self.OBS]](
            mptr(rows.dev.value().unsafe_ptr()),
            mptr(self.mean.dev.value().unsafe_ptr()),
            mptr(self.var_.dev.value().unsafe_ptr()),
            Scalar[DT](OBS_EMA_EPS),
            grid_dim=_blocks(ROWS * Self.OBS), block_dim=TPB,
        )

    def apply_into[ROWS: Int](
        self,
        src: Pointer[Scalar[DT], MutAnyOrigin],
        mut dst: Tensor,
    ) raises:
        """`dst ← normalise(src)` for ROWS rows (device), `src` untouched."""
        var c = self.ctx.value()
        c.enqueue_function[ema_copy_apply_kernel[ROWS, Self.OBS]](
            src,
            mptr(dst.dev.value().unsafe_ptr()),
            mptr(self.mean.dev.value().unsafe_ptr()),
            mptr(self.var_.dev.value().unsafe_ptr()),
            Scalar[DT](OBS_EMA_EPS),
            grid_dim=_blocks(ROWS * Self.OBS), block_dim=TPB,
        )

    # ── host ──────────────────────────────────────────────────────────────
    def sync_host(mut self) raises:
        var c = self.ctx.value()
        self.mean.download(c)
        self.var_.download(c)
        c.synchronize()

    def apply_host(self, mut row: List[Scalar[DT]]):
        """One row on the host, from the last `sync_host`."""
        for d in range(Self.OBS):
            row[d] = (row[d] - self.mean.data[d]) / sqrt(
                self.var_.data[d] + Scalar[DT](OBS_EMA_EPS)
            )

    def apply_host_tensor(self, mut row: Tensor):
        """One row held in a Tensor's host mirror, from the last `sync_host`."""
        for d in range(Self.OBS):
            row.data[d] = (row.data[d] - self.mean.data[d]) / sqrt(
                self.var_.data[d] + Scalar[DT](OBS_EMA_EPS)
            )

    def save(mut self, path: String) raises:
        """`ObsNorm`'s sidecar format: `N`, then `mu sd var` per dimension
        with `sd = sqrt(var + eps)` — the evals apply the first two through
        `ObsNorm.try_load` / `apply_row`, which reads two columns and ignores
        the rest — then a trailing `n_updates <k>` line.

        ⚠ THE THIRD COLUMN IS WHAT MAKES A RESUME EXACT. Rebuilding the
        variance as `sd² − eps` cancels catastrophically for a dimension whose
        variance is near or below eps (1e-5): a near-constant input came back
        with a variance off by orders of magnitude, or negative. Written
        atomically: a crash mid-save must not leave the checkpoint beside it
        with a truncated normaliser."""
        self.sync_host()
        var s = String(Self.OBS) + "\n"
        for d in range(Self.OBS):
            var v = Float64(self.var_.data[d])
            var sd = sqrt(v + OBS_EMA_EPS)
            s += (
                String(Float64(self.mean.data[d])) + " " + String(sd) + " "
                + String(v) + "\n"
            )
        s += "n_updates " + String(self.n_updates) + "\n"
        write_text_atomic(path, s)

    def load(mut self, path: String) raises:
        """Restore from the sidecar (`var = sd² − eps`), upload."""
        var content: String
        with open(path, "r") as f:
            content = f.read()
        var lines = content.split("\n")
        if len(lines) < Self.OBS + 1:
            raise Error("ObsEma.load: sidecar " + path + " is short")
        var n = atol(String(lines[0]).strip())
        if n != Self.OBS:
            raise Error(
                "ObsEma.load: sidecar holds " + String(n) + " dims, expected "
                + String(Self.OBS)
            )
        for d in range(Self.OBS):
            var parts = String(lines[1 + d]).split(" ")
            var mu = atof(String(parts[0]).strip())
            self.mean.data[d] = Scalar[DT](mu)
            if len(parts) >= 3:
                self.var_.data[d] = Scalar[DT](atof(String(parts[2]).strip()))
            else:
                # A sidecar from before the `var` column: lossy near eps.
                var sd = atof(String(parts[1]).strip())
                self.var_.data[d] = Scalar[DT](sd * sd - OBS_EMA_EPS)
        for i in range(Self.OBS + 1, len(lines)):
            var kv = String(lines[i]).strip().split(" ")
            if len(kv) == 2 and String(kv[0]) == "n_updates":
                self.n_updates = atol(String(kv[1]))
        var c = self.ctx.value()
        self.mean.upload(c)
        self.var_.upload(c)
        self.enabled = True
