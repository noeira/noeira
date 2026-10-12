"""Running observation / return statistics and the policy's augmented
observation — host and device, shared by the vectorised PPO drivers
(`ppo_vec_driver`, `tasks/ppo_family_driver` and its device rollout).

  - `RunningMeanStd` — CleanRL's running mean / variance (parallel Welford over
    batches), the host struct; `save` / `load` write the text format the
    family runs' `obs_norm.txt` uses.
  - `update_rms_device` / `normalize_device` — its device twins over `[N, D]`
    batches (one thread per dimension walks the lanes in the host's order;
    rows flagged in `skip` are left out / zeroed).
  - `norm_reward` — CleanRL's `NormalizeReward` scale-and-clip, the ONE copy
    the reward kernels call.
  - `augment_k` — the policy's observation: the env row, the last W action
    words, then (T > 0) a per-joint lead `prev[j] - raw[qa[j]]`.

Held against each other by `tests/tasks/test_ppo_family_device_kernels.mojo`.
"""

from layout import Layout, LayoutTensor
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import sqrt, sqrt as fsqrt

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor


comptime _V[n: Int] = LayoutTensor[DT, Layout.row_major(n), MutAnyOrigin]


struct RunningMeanStd(Movable):
    """CleanRL's running mean / variance (parallel Welford over batches)."""

    var mean: List[Float64]
    var var_: List[Float64]
    var count: Float64

    def __init__(out self, dim: Int):
        self.mean = List[Float64](length=dim, fill=0.0)
        self.var_ = List[Float64](length=dim, fill=1.0)
        self.count = 1e-4

    def update(
        mut self, x: Pointer[Scalar[DT], MutAnyOrigin], n_rows: Int, dim: Int,
        skip: List[Bool] = List[Bool](),
    ):
        """Rows with `skip[i]` set are left out (diverged lanes)."""
        var bm = List[Float64](length=dim, fill=0.0)
        var bv = List[Float64](length=dim, fill=0.0)
        var n = 0
        for i in range(n_rows):
            if len(skip) > 0 and skip[i]:
                continue
            n += 1
            for k in range(dim):
                bm[k] += Float64(x[unsafe_offset = i * dim + k])
        if n == 0:
            return
        for k in range(dim):
            bm[k] /= Float64(n)
        for i in range(n_rows):
            if len(skip) > 0 and skip[i]:
                continue
            for k in range(dim):
                var d = Float64(x[unsafe_offset = i * dim + k]) - bm[k]
                bv[k] += d * d
        for k in range(dim):
            bv[k] /= Float64(n)
        var tot = self.count + Float64(n)
        for k in range(dim):
            var delta = bm[k] - self.mean[k]
            var m_a = self.var_[k] * self.count
            var m_b = bv[k] * Float64(n)
            var m2 = m_a + m_b + delta * delta * self.count * Float64(n) / tot
            self.mean[k] += delta * Float64(n) / tot
            self.var_[k] = m2 / tot
        self.count = tot

    def normalize_into(
        self,
        src: Pointer[Scalar[DT], MutAnyOrigin],
        dst: Pointer[Scalar[DT], MutAnyOrigin],
        n: Int,
        dim: Int,
        clip: Float64,
    ):
        for i in range(n):
            for k in range(dim):
                var v = (Float64(src[unsafe_offset = i * dim + k]) - self.mean[k]) / sqrt(
                    self.var_[k] + 1e-8
                )
                if v > clip:
                    v = clip
                elif v < -clip:
                    v = -clip
                dst[unsafe_offset = i * dim + k] = Scalar[DT](v)

    def load(mut self, path: String) raises:
        var txt = String()
        with open(path, "r") as f:
            txt = f.read()
        var lines = txt.split("\n")
        self.count = Float64(String(lines[0].split(" ")[1]))
        var m = lines[1].split(" ")
        var v = lines[2].split(" ")
        for k in range(len(self.mean)):
            self.mean[k] = Float64(String(m[k + 1]))
            self.var_[k] = Float64(String(v[k + 1]))

    def save(self, path: String) raises:
        var s = String("count ") + String(self.count) + "\n"
        s += "mean"
        for k in range(len(self.mean)):
            s += " " + String(self.mean[k])
        s += "\nvar"
        for k in range(len(self.var_)):
            s += " " + String(self.var_[k])
        s += "\n"
        with open(path, "w") as f:
            f.write(s)



@always_inline
def norm_reward(r: Scalar[DT], ret_var: Scalar[DT], clip: Scalar[DT]) -> Scalar[DT]:
    """CleanRL `NormalizeReward`: r / std(discounted return), clipped."""
    var v = r * (Scalar[DT](1.0) / fsqrt(ret_var + Scalar[DT](1e-8)))
    if v > clip:
        return clip
    if v < -clip:
        return -clip
    return v


def augment_k[N: Int, E_OBS: Int, W: Int, T: Int, ACT: Int](
    raw: _V[N * E_OBS],
    hist: _V[N * W + 1],
    tprev: _V[N * ACT],
    qa: _V[ACT],
    aug: _V[N * (E_OBS + W + T)],
):
    """The env row, the history (W words), then (T = ACT) the target's lead
    over the joints `qa`; T = 0 leaves `tprev` / `qa` unread. `hist` is sized
    `N*W + 1` so W = 0 still builds."""
    comptime assert T == 0 or T == ACT, "augment_k: the lead is ACT words or none"
    comptime A = E_OBS + W + T
    var e = Int(global_idx.x)
    if e >= N:
        return
    for k in range(E_OBS):
        aug[e * A + k] = raw[e * E_OBS + k]
    for k in range(W):
        aug[e * A + E_OBS + k] = hist[e * W + k]
    comptime if T > 0:
        for j in range(ACT):
            var q = rebind[Scalar[DT]](
                raw[e * E_OBS + Int(rebind[Scalar[DT]](qa[j]))]
            )
            aug[e * A + E_OBS + W + j] = rebind[Scalar[DT]](
                tprev[e * ACT + j]
            ) - q



def _rms_update_k[N: Int, D: Int](
    x: _V[N * D],
    skip: _V[N],
    use_skip: Int32,
    mean: _V[D],
    var_: _V[D],
    count: _V[1],
):
    """`RunningMeanStd.update`, one thread per dimension (each walks the
    lanes in the host's order); rows with `skip` set are left out when
    `use_skip`. The shared count is NOT written here (`_rms_count_k`), so
    every dimension merges against the same old count."""
    var k = Int(global_idx.x)
    if k >= D:
        return
    var n = 0
    var bm: Scalar[DT] = 0.0
    for i in range(N):
        if use_skip != 0 and rebind[Scalar[DT]](skip[i]) > Scalar[DT](0.5):
            continue
        n += 1
        bm += rebind[Scalar[DT]](x[i * D + k])
    if n == 0:
        return
    bm /= Scalar[DT](n)
    var bv: Scalar[DT] = 0.0
    for i in range(N):
        if use_skip != 0 and rebind[Scalar[DT]](skip[i]) > Scalar[DT](0.5):
            continue
        var d = rebind[Scalar[DT]](x[i * D + k]) - bm
        bv += d * d
    bv /= Scalar[DT](n)
    var c = rebind[Scalar[DT]](count[0])
    var nf = Scalar[DT](n)
    var tot = c + nf
    var mk = rebind[Scalar[DT]](mean[k])
    var delta = bm - mk
    var m2 = rebind[Scalar[DT]](var_[k]) * c + bv * nf + delta * delta * c * nf / tot
    mean[k] = mk + delta * nf / tot
    var_[k] = m2 / tot


def _rms_count_k[N: Int](
    skip: _V[N], use_skip: Int32, count: _V[1],
):
    """The count's half of `RunningMeanStd.update`, after `_rms_update_k`."""
    if Int(global_idx.x) != 0:
        return
    var n = 0
    for i in range(N):
        if use_skip != 0 and rebind[Scalar[DT]](skip[i]) > Scalar[DT](0.5):
            continue
        n += 1
    if n > 0:
        count[0] = rebind[Scalar[DT]](count[0]) + Scalar[DT](n)


def _rms_normalize_k[N: Int, D: Int](
    x: _V[N * D],
    dst: _V[N * D],
    mean: _V[D],
    var_: _V[D],
    clip: Scalar[DT],
    zero: _V[N],
    use_zero: Int32,
):
    """`RunningMeanStd.normalize_into`, then (`use_zero`) the diverged lanes'
    rows zeroed, as `run_ppo` does to a diverged lane's terminal obs."""
    var i = Int(global_idx.x)
    if i >= N * D:
        return
    var e = i // D
    var k = i % D
    if use_zero != 0 and rebind[Scalar[DT]](zero[e]) > Scalar[DT](0.5):
        dst[i] = Scalar[DT](0.0)
        return
    var v = (rebind[Scalar[DT]](x[i]) - rebind[Scalar[DT]](mean[k])) / fsqrt(
        rebind[Scalar[DT]](var_[k]) + Scalar[DT](1e-8)
    )
    if v > clip:
        v = clip
    elif v < -clip:
        v = -clip
    dst[i] = v



def update_rms_device[N: Int, D: Int](
    ctx: DeviceContext,
    mut x: Tensor,
    mut skip: Tensor,
    mut mean: Tensor,
    mut var_: Tensor,
    mut count: Tensor,
    use_skip: Bool,
) raises:
    """`RunningMeanStd.update` on the device: the per-dimension merge, then
    the shared count."""
    var us = Int32(1) if use_skip else Int32(0)
    ctx.enqueue_function[_rms_update_k[N, D]](
        x.lt["gpu", Layout.row_major(N * D)](),
        skip.lt["gpu", Layout.row_major(N)](),
        us,
        mean.lt["gpu", Layout.row_major(D)](),
        var_.lt["gpu", Layout.row_major(D)](),
        count.lt["gpu", Layout.row_major(1)](),
        grid_dim=(D + TPB - 1) // TPB, block_dim=TPB,
    )
    ctx.enqueue_function[_rms_count_k[N]](
        skip.lt["gpu", Layout.row_major(N)](),
        us,
        count.lt["gpu", Layout.row_major(1)](),
        grid_dim=1, block_dim=1,
    )


def normalize_device[N: Int, D: Int](
    ctx: DeviceContext,
    mut src: Tensor,
    mut dst: Tensor,
    mut mean: Tensor,
    mut var_: Tensor,
    mut zero: Tensor,
    clip: Float64,
    zero_rows: Bool,
) raises:
    """`RunningMeanStd.normalize_into` on the device (`zero_rows`: the rows
    flagged in `zero` written as 0)."""
    ctx.enqueue_function[_rms_normalize_k[N, D]](
        src.lt["gpu", Layout.row_major(N * D)](),
        dst.lt["gpu", Layout.row_major(N * D)](),
        mean.lt["gpu", Layout.row_major(D)](),
        var_.lt["gpu", Layout.row_major(D)](),
        Scalar[DT](clip),
        zero.lt["gpu", Layout.row_major(N)](),
        Int32(1) if zero_rows else Int32(0),
        grid_dim=(N * D + TPB - 1) // TPB, block_dim=TPB,
    )
