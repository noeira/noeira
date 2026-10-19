"""GpuMCTSExampleReplay — fully device-resident AlphaZero example replay.

The GPU sibling of `MCTSExampleReplay`: the big tensors (obs, packed target
``[π | z]``) NEVER leave the device — eliminating the per-step obs/policy D2H and
the per-train-step batch H2D of the host ring. Only tiny per-step control-flow
scalars (``done`` / ``rew`` / per-env trajectory lengths, all O(N_ENVS)) touch
the host. Sampling is a device→device gather with an on-device RNG seed (no
per-step slot upload), so the train step (sample → graph forward/vjp → optimizer)
is a pure device pipeline — CUDA-graph-capturable.

Layout (mirrors the MuZero GPU replay's obs-on-device + host-bookkeeping split):
  * per-env trajectory   `traj_obs [N_ENVS, MAX_TRAJ, OBS]`,
                         `traj_pol [N_ENVS, MAX_TRAJ, ACT]`  (device)
  * the example ring     `obs_ring [CAP, OBS]`, `tgt_ring [CAP, ACT+1]`  (device)
  * host counters        `traj_len[N_ENVS]`, `pos`, `size`  (control flow only)

Flow per self-play step:
  1. `record_step_gpu(obs_dev, pol_dev)` — store this move's root obs + visit
     policy into each env's open trajectory slot (device store kernels).
  2. `flush_finished_gpu(done_dev, rew_dev)` — for each finished env, a flush
     kernel appends its trajectory to the ring with the strict-alternation value
     target ``z`` computed IN-KERNEL (``z_k = +1 if win and (L-1-k) even, -1 if
     win and odd, 0 otherwise`` — the GPU driver's exact convention).
  3. `sample_batch_gpu[B](obs_out, tgt_out)` — device→device gather into the
     ComputeGraph's storage Tensors.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.core.tiles import Tile, tile


@always_inline
def _xs64(s: UInt64) -> UInt64:
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


# ── Kernels (element-parallel; one thread per item × element). ─────────────


def _store_rows_kernel[
    N_ENVS: Int, W: Int, MAX_TRAJ: Int
](
    src: Tile[DT, N_ENVS, W],
    traj: Tile[DT, N_ENVS, MAX_TRAJ, W],
    len_dev: Tile[DType.int32, N_ENVS],
):
    """Append `src[e, :]` into env `e`'s trajectory at slot `len_dev[e]`
    (skipped once the trajectory is full). Thread `t` → env `e=t//W`, elem
    `d=t%W`."""
    var t = Int(global_idx.x)
    if t >= N_ENVS * W:
        return
    var e = t // W
    var d = t % W
    var k = Int(len_dev[e])
    if k < MAX_TRAJ:
        traj[e, k, d] = rebind[Scalar[DT]](src[e, d])


def _flush_obs_kernel[
    N_ENVS: Int, OBS: Int, MAX_TRAJ: Int, CAP: Int
](
    traj_obs: Tile[DT, N_ENVS, MAX_TRAJ, OBS],
    obs_ring: Tile[DT, CAP, OBS],
    e: Int32, L: Int32, start: Int32,
):
    """Copy env `e`'s `L` trajectory obs rows into the ring at
    `(start + k) % CAP`."""
    var t = Int(global_idx.x)
    if t >= Int(L) * OBS:
        return
    var k = t // OBS
    var d = t % OBS
    var slot = (Int(start) + k) % CAP
    obs_ring[slot, d] = rebind[Scalar[DT]](traj_obs[Int(e), k, d])


def _flush_tgt_kernel[
    N_ENVS: Int, ACT: Int, MAX_TRAJ: Int, CAP: Int
](
    traj_pol: Tile[DT, N_ENVS, MAX_TRAJ, ACT],
    tgt_ring: Tile[DT, CAP, ACT + 1],
    e: Int32, L: Int32, start: Int32, win: Scalar[DT],
):
    """Write env `e`'s `L` packed targets `[π(ACT) | z]` into the ring. `z` is
    computed in-kernel from `win` + step parity (last mover gets +1 on a win,
    signs alternate backward; 0 on non-win)."""
    comptime TGT = ACT + 1
    var t = Int(global_idx.x)
    if t >= Int(L) * TGT:
        return
    var k = t // TGT
    var c = t % TGT
    var slot = (Int(start) + k) % CAP
    if c < ACT:
        tgt_ring[slot, c] = rebind[Scalar[DT]](traj_pol[Int(e), k, c])
    else:
        var z = Scalar[DT](0)
        if win > Scalar[DT](0.5):
            z = Scalar[DT](1) if ((Int(L) - 1 - k) % 2 == 0) else Scalar[DT](-1)
        tgt_ring[slot, ACT] = z


def _gather_rows_kernel[
    B: Int, W: Int, CAP: Int
](
    ring: Tile[DT, CAP, W],
    dst: Tile[DT, B, W],
    rng: Tile[DType.uint64, 1],
    size: Int32,
):
    """Gather `B` rows from `ring` into `dst`. The source slot for window `b` is
    `xorshift64(seed ^ (b+1)) % size` (deterministic in `b` given the shared
    device seed `rng[0]`), so obs and tgt gathers with the same seed pick the
    SAME example per window."""
    var t = Int(global_idx.x)
    if t >= B * W:
        return
    var b = t // W
    var d = t % W
    var seed = rebind[Scalar[DType.uint64]](rng[0])
    var slot = Int(_xs64(seed ^ UInt64(b + 1)) % UInt64(size))
    dst[b, d] = rebind[Scalar[DT]](ring[slot, d])


def _advance_rng_kernel(
    rng: Tile[DType.uint64, 1],
):
    """Advance the device sampling seed (1 thread). Run AFTER the gathers so the
    next `sample_batch_gpu` draws a fresh set."""
    if Int(global_idx.x) == 0:
        rng[0] = _xs64(rebind[Scalar[DType.uint64]](rng[0]))


struct GpuMCTSExampleReplay[
    OBS: Int, ACT: Int, CAP: Int, N_ENVS: Int, MAX_TRAJ: Int
](Movable, Deinitable, Sized):
    comptime TGT = Self.ACT + 1

    var ctx: DeviceContext
    # device trajectory + ring (RAII DeviceBuffers)
    var traj_obs: DeviceBuffer[DT]   # [N_ENVS, MAX_TRAJ, OBS]
    var traj_pol: DeviceBuffer[DT]   # [N_ENVS, MAX_TRAJ, ACT]
    var obs_ring: DeviceBuffer[DT]   # [CAP, OBS]
    var tgt_ring: DeviceBuffer[DT]   # [CAP, TGT]
    var len_dev: DeviceBuffer[DType.int32]   # [N_ENVS] open-trajectory lengths
    var rng_dev: DeviceBuffer[DType.uint64]  # [1] device sampling seed
    # host control-flow staging
    var len_host: HostBuffer[DType.int32]
    var done_host: HostBuffer[DT]
    var rew_host: HostBuffer[DT]
    var traj_len: List[Int]
    var pos: Int
    var size: Int

    def __init__(
        out self, ctx: DeviceContext, seed: UInt64 = 0x243F6A8885A308D3
    ) raises:
        self.ctx = ctx
        self.traj_obs = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.MAX_TRAJ * Self.OBS
        )
        self.traj_pol = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.MAX_TRAJ * Self.ACT
        )
        self.obs_ring = ctx.enqueue_create_buffer[DT](Self.CAP * Self.OBS)
        self.tgt_ring = ctx.enqueue_create_buffer[DT](Self.CAP * Self.TGT)
        self.len_dev = ctx.enqueue_create_buffer[DType.int32](Self.N_ENVS)
        self.rng_dev = ctx.enqueue_create_buffer[DType.uint64](1)
        self.len_host = ctx.enqueue_create_host_buffer[DType.int32](Self.N_ENVS)
        self.done_host = ctx.enqueue_create_host_buffer[DT](Self.N_ENVS)
        self.rew_host = ctx.enqueue_create_host_buffer[DT](Self.N_ENVS)
        self.traj_len = List[Int](length=Self.N_ENVS, fill=0)
        self.pos = 0
        self.size = 0
        # seed the device RNG (1 element, host-staged once).
        var sh = ctx.enqueue_create_host_buffer[DType.uint64](1)
        ctx.synchronize()
        sh[0] = seed | 1
        ctx.enqueue_copy(self.rng_dev, sh)
        ctx.synchronize()

    def __len__(self) -> Int:
        return self.size

    def record_step_gpu(
        mut self, mut obs_dev: DeviceBuffer[DT], mut pol_dev: DeviceBuffer[DT]
    ) raises:
        """Append this move's root obs (`obs_dev[N_ENVS, OBS]`) + visit policy
        (`pol_dev[N_ENVS, ACT]`) into each env's open trajectory. Device→device;
        only the O(N_ENVS) length vector is staged to the device."""
        # publish host lengths to the device store-slot vector (tiny H2D).
        for e in range(Self.N_ENVS):
            self.len_host[e] = Int32(self.traj_len[e])
        self.ctx.enqueue_copy(self.len_dev, self.len_host)
        var len_lt = tile[Self.N_ENVS](self.len_dev)

        var src_obs = tile[Self.N_ENVS, Self.OBS](obs_dev)
        var traj_obs_lt = tile[Self.N_ENVS, Self.MAX_TRAJ, Self.OBS](self.traj_obs)
        self.ctx.enqueue_function[
            _store_rows_kernel[Self.N_ENVS, Self.OBS, Self.MAX_TRAJ]
        ](
            src_obs, traj_obs_lt, len_lt,
            grid_dim=(Self.N_ENVS * Self.OBS + TPB - 1) // TPB, block_dim=TPB,
        )

        var src_pol = tile[Self.N_ENVS, Self.ACT](pol_dev)
        var traj_pol_lt = tile[Self.N_ENVS, Self.MAX_TRAJ, Self.ACT](self.traj_pol)
        self.ctx.enqueue_function[
            _store_rows_kernel[Self.N_ENVS, Self.ACT, Self.MAX_TRAJ]
        ](
            src_pol, traj_pol_lt, len_lt,
            grid_dim=(Self.N_ENVS * Self.ACT + TPB - 1) // TPB, block_dim=TPB,
        )

        for e in range(Self.N_ENVS):
            if self.traj_len[e] < Self.MAX_TRAJ:
                self.traj_len[e] += 1

    def flush_finished_gpu(
        mut self, mut done_dev: DeviceBuffer[DT], mut rew_dev: DeviceBuffer[DT]
    ) raises:
        """For each finished env, append its trajectory to the ring with
        in-kernel value targets `z`, then reset its open length. The big copies
        are device→device flush kernels; only the O(N_ENVS) done/rew vectors are
        read back to drive the (sequential) ring-position bookkeeping."""
        self.ctx.enqueue_copy(self.done_host, done_dev)
        self.ctx.enqueue_copy(self.rew_host, rew_dev)
        self.ctx.synchronize()

        var traj_obs_lt = tile[Self.N_ENVS, Self.MAX_TRAJ, Self.OBS](self.traj_obs)
        var traj_pol_lt = tile[Self.N_ENVS, Self.MAX_TRAJ, Self.ACT](self.traj_pol)
        var obs_ring_lt = tile[Self.CAP, Self.OBS](self.obs_ring)
        var tgt_ring_lt = tile[Self.CAP, Self.TGT](self.tgt_ring)

        for e in range(Self.N_ENVS):
            if self.done_host[e] > Scalar[DT](0.5):
                var L = self.traj_len[e]
                if L > 0:
                    var win = (
                        Scalar[DT](1) if self.rew_host[e] > Scalar[DT](0.5)
                        else Scalar[DT](0)
                    )
                    self.ctx.enqueue_function[
                        _flush_obs_kernel[
                            Self.N_ENVS, Self.OBS, Self.MAX_TRAJ, Self.CAP
                        ]
                    ](
                        traj_obs_lt, obs_ring_lt,
                        Int32(e), Int32(L), Int32(self.pos),
                        grid_dim=(L * Self.OBS + TPB - 1) // TPB, block_dim=TPB,
                    )
                    self.ctx.enqueue_function[
                        _flush_tgt_kernel[
                            Self.N_ENVS, Self.ACT, Self.MAX_TRAJ, Self.CAP
                        ]
                    ](
                        traj_pol_lt, tgt_ring_lt,
                        Int32(e), Int32(L), Int32(self.pos), win,
                        grid_dim=(L * Self.TGT + TPB - 1) // TPB, block_dim=TPB,
                    )
                    self.pos = (self.pos + L) % Self.CAP
                    if self.size < Self.CAP:
                        self.size = min(self.size + L, Self.CAP)
                self.traj_len[e] = 0

    def sample_batch_gpu[
        B: Int
    ](mut self, mut obs_out: Tensor, mut tgt_out: Tensor) raises:
        """Device→device gather of `B` uniform examples into the storage Tensors'
        device buffers (`obs_out[B*OBS]` / `tgt_out[B*TGT]`). No host transfer —
        the train step is a pure device pipeline."""
        obs_out.ensure_gpu(self.ctx, B * Self.OBS)
        tgt_out.ensure_gpu(self.ctx, B * Self.TGT)
        var rng_lt = tile[1](self.rng_dev)

        var obs_ring_lt = tile[Self.CAP, Self.OBS](self.obs_ring)
        var obs_out_lt = obs_out.tt["gpu", B, Self.OBS]()
        self.ctx.enqueue_function[_gather_rows_kernel[B, Self.OBS, Self.CAP]](
            obs_ring_lt, obs_out_lt, rng_lt, Int32(self.size),
            grid_dim=(B * Self.OBS + TPB - 1) // TPB, block_dim=TPB,
        )

        var tgt_ring_lt = tile[Self.CAP, Self.TGT](self.tgt_ring)
        var tgt_out_lt = tgt_out.tt["gpu", B, Self.TGT]()
        self.ctx.enqueue_function[_gather_rows_kernel[B, Self.TGT, Self.CAP]](
            tgt_ring_lt, tgt_out_lt, rng_lt, Int32(self.size),
            grid_dim=(B * Self.TGT + TPB - 1) // TPB, block_dim=TPB,
        )

        self.ctx.enqueue_function[_advance_rng_kernel](
            rng_lt, grid_dim=1, block_dim=1
        )
