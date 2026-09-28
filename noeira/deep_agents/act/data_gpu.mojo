# +--------------------------------------------------------------------------+ #
# | ACTDeviceDataset — the whole dataset resident on the GPU, sampled there
# +--------------------------------------------------------------------------+ #
"""Draw, gather and normalize a batch entirely on the device.

The host path (`data.mojo::sample_batch` + `Tensor.upload_resident`) is 16.1 ms
of a 144.8 ms ACT iteration with the GPU idle throughout, plus two 29.5 MB
element-by-element fills into pinned memory and a 29.5 MB H2D, plus 8 of the
~28 device synchronizations a step pays. This replaces all of it with three
kernels over data that never leaves the device.

Shaped after `deep_agents/data/gpu_sequence_replay.mojo`, which does the same
job for SAC's replay: a Philox draw kernel with a DEVICE-RESIDENT offset, then
an element-parallel gather. ACT is easier than SAC in the one way that matters
— the dataset is fixed, so the device copy is uploaded once and never
invalidated.

## Why uint8 on the device

Images are `N_CAM * 3 * H * W` uint8 = 460,800 B per frame. The 50-episode
store is 15,447 rows:

    7.12 GB as uint8        fits a 32 GB card with ~25 GB to spare
    28.47 GB as float32     does NOT fit, today

So the normalize belongs in the gather kernel, not in the storage format. That
also means the upload is 4x smaller, and it is the difference between "this
dataset is resident" and "this dataset is not".

⚠ **This does not scale past about an hour of recording** — 108,000 rows is
49.8 GB. `docs/ACT_GPU_DATA_PATH.md` describes the windowed tier-2 design for
that; nothing here forecloses it, because everything below is addressed through
`_g` (the chosen row per batch slot) and a window would simply change what `_g`
indexes into.

## Parity

The device RNG is Philox and the host RNG is a xorshift, so the two samplers
CANNOT draw the same batches and no seed makes them. What is testable, and what
`tests/deep_agents/act/test_act_dataset_gpu.mojo` tests, is the part that is
supposed to agree: given the SAME `(episode, start_ts)`, the device gather must
produce the same tensors as `ACTDataset.fill_at`. That is why `gather_at`
exists as a public entry point separate from `sample`.
"""

from max.gpu import global_idx, thread_idx
from std.memory import Pointer
from std.random.philox import Random as PhiloxRandom
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.io.hdf5 import H5Dataset

from .data import ACTDataset
from .augment import (
    AUG_SALT,
    AUG_UNIFORMS,
    AUG_WORDS,
    ImageAugConfig,
    aug_param,
    aug_src_index,
    aug_value_u8,
)
from .config import (
    IMAGENET_MEAN_R,
    IMAGENET_MEAN_G,
    IMAGENET_MEAN_B,
    IMAGENET_STD_R,
    IMAGENET_STD_G,
    IMAGENET_STD_B,
)

comptime U8 = DType.uint8
comptime I32 = DType.int32
comptime U64 = DType.uint64

# Rows per HDF5 read during the one-time upload. 64 x 460,800 B = 29.5 MB of
# pinned staging — big enough to amortise the per-read overhead, small enough
# not to be a second copy of the dataset in host RAM.
comptime UPLOAD_CHUNK_ROWS = 64


# ── draw ────────────────────────────────────────────────────────────────


def _act_draw_kernel[
    B: Int, K: Int
](
    g: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    n_real: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    eps: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    ep_start: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    ep_len: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    n_split: Int32,
    seed: UInt64,
    offset_buf: LayoutTensor[U64, Layout.row_major(1), MutAnyOrigin],
):
    """Thread `b` picks an episode from the split, then a start step in it.

    Mirrors the host sampler's two draws exactly in STRUCTURE —
    `eps[rand_below(n_split)]` then `rand_below(ep_len)` — but with Philox, so
    the streams differ. That is expected and is why parity is tested through
    `gather_at`, not through a shared seed.

    ⚠ The offset comes from a DEVICE buffer and is advanced by a device kernel,
    not a host counter. Same reason `Adam._pow_dev` exists: a host counter
    freezes under CUDA-graph replay. Nothing captures this yet (split-K blocks
    it, see `_split_k_cannot_be_cuda_graph_captured`), but building it host-side
    would have to be undone later.
    """
    var b = Int(global_idx.x)
    if b >= B:
        return
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(b), offset=offset_base)
    var u0 = Float32(philox.step_uniform()[0])
    var u1 = Float32(philox.step_uniform()[0])

    var ns = Int(n_split)
    if ns < 1:
        ns = 1
    var ei = Int(u0 * Float32(ns))
    if ei >= ns:
        ei = ns - 1
    if ei < 0:
        ei = 0
    var ep = Int(eps[ei])

    var elen = Int(ep_len[ep])
    if elen < 1:
        elen = 1
    var ts = Int(u1 * Float32(elen))
    if ts >= elen:
        ts = elen - 1
    if ts < 0:
        ts = 0

    g[b] = Int32(Int(ep_start[ep]) + ts)
    var remaining = elen - ts
    n_real[b] = Int32(K if remaining > K else remaining)


def _act_advance_offset_kernel[
    B: Int
](offset: LayoutTensor[U64, Layout.row_major(1), MutAnyOrigin]):
    """Bump the RNG offset by the 2 draws per row this batch consumed."""
    if Int(thread_idx.x) == 0:
        offset[0] = offset[0] + UInt64(B * 2)


# ── gather + normalize ──────────────────────────────────────────────────


def _act_gather_images_kernel[
    B: Int, IMG_ELEMS: Int, CAM_ELEMS: Int, HW: Int
](
    src: Pointer[Scalar[U8], MutAnyOrigin],
    g: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    out_img: LayoutTensor[DT, Layout.row_major(B * IMG_ELEMS), MutAnyOrigin],
    n_rows: Int64,
):
    """`out[b, e] = (src[g[b], e]/255 - mean_ch) * inv_std_ch`.

    One thread per output element. The channel is recovered from the flat
    offset because the layout is `[N_CAM][3][H][W]`: `cam = e / CAM_ELEMS`,
    `ch = (e % CAM_ELEMS) / HW`. Same arithmetic as the host loop in
    `data.mojo::_fill_one`, which is what makes the parity gate meaningful.

    ⚠⚠ `src` is a RAW POINTER, not a `LayoutTensor`, and that is load-bearing.
    Its extent is `n_rows * IMG_ELEMS` = **7.1 BILLION** elements for the
    50-episode store, and `LayoutTensor`'s `linear_idx_type` defaults to
    **int32** — so indexing one wraps at 2.147 G however carefully the offset
    is computed. The first version of this kernel did compute the offset in
    Int64 and still read garbage, because the wrap was downstream of the
    arithmetic, in the indexing.

    The parity gate caught it only because it was run against BOTH stores: at
    1,997 rows (0.92 GB, 920 M elements) it passes, and at 15,447 rows it does
    not. A gate on the small store alone would have called this correct.
    """
    var i = Int(global_idx.x)
    if i >= B * IMG_ELEMS:
        return
    var b = i // IMG_ELEMS
    var e = i % IMG_ELEMS
    var ch = (e % CAM_ELEMS) // HW

    var mean = Scalar[DT](IMAGENET_MEAN_R) if ch == 0 else (
        Scalar[DT](IMAGENET_MEAN_G) if ch == 1 else Scalar[DT](IMAGENET_MEAN_B)
    )
    var std = Scalar[DT](IMAGENET_STD_R) if ch == 0 else (
        Scalar[DT](IMAGENET_STD_G) if ch == 1 else Scalar[DT](IMAGENET_STD_B)
    )

    var row = Int(g[b])
    var flat = row * IMG_ELEMS + e          # Int is 64-bit on device
    var v = Scalar[DT](Int(src[unsafe_offset=flat])) / Scalar[DT](255.0)
    out_img[i] = (v - mean) * (Scalar[DT](1.0) / std)


# ── augmentation (docs/DOMAIN_RANDOMIZATION_PLAN.md, Phase 1) ─────────────


def _act_aug_draw_kernel[
    B: Int, N_CAM: Int, IMG_H: Int, IMG_W: Int
](
    params: LayoutTensor[DT, Layout.row_major(B * N_CAM * AUG_WORDS), MutAnyOrigin],
    seed: UInt64,
    offset_buf: LayoutTensor[U64, Layout.row_major(1), MutAnyOrigin],
    brightness: Float32,
    contrast: Float32,
    gamma: Float32,
    gain: Float32,
    noise_sigma: Float32,
    max_shift: Int32,
    cutout_prob: Float32,
    cutout_max_frac: Float32,
):
    """Thread `(b, cam)` draws its record — `augment.aug_param` from 16
    Philox uniforms.

    ⚠ Reads the SAME device offset as `_act_draw_kernel`, and runs BEFORE the
    advance kernel, so pinning the offset (`set_offset`, validation) pins the
    augmentation too and `offset_host` keeps its `+= 2*B` rule. The streams
    are separated by the SEED (`seed ^ AUG_SALT`), not by the offset."""
    var t = Int(global_idx.x)
    if t >= B * N_CAM:
        return
    var philox = PhiloxRandom(
        seed=(seed ^ AUG_SALT) + UInt64(t),
        offset=rebind[UInt64](offset_buf[0]),
    )
    var u = SIMD[DType.float32, AUG_UNIFORMS](0.0)
    comptime for k in range(AUG_UNIFORMS // 4):
        var r = philox.step_uniform()
        comptime for j in range(4):
            u[4 * k + j] = Float32(r[j])
    var cfg = ImageAugConfig(
        True, brightness, contrast, gamma, gain, noise_sigma, Int(max_shift),
        cutout_prob, cutout_max_frac,
    )
    var p = aug_param(cfg, u, IMG_H, IMG_W)
    comptime for w in range(AUG_WORDS):
        params[t * AUG_WORDS + w] = Scalar[DT](p[w])


def _act_gather_images_aug_kernel[
    B: Int, N_CAM: Int, IMG_H: Int, IMG_W: Int
](
    src: Pointer[Scalar[U8], MutAnyOrigin],
    g: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    params: LayoutTensor[DT, Layout.row_major(B * N_CAM * AUG_WORDS), MutAnyOrigin],
    out_img: LayoutTensor[
        DT, Layout.row_major(B * N_CAM * 3 * IMG_H * IMG_W), MutAnyOrigin
    ],
    n_rows: Int64,
):
    """`_act_gather_images_kernel` with the augmented byte in place of the
    stored one. Everything after the byte — `/255`, ImageNet, `* (1/std)` —
    is the SAME arithmetic, which is what keeps the normalisation the one
    rule the deploy path shares.

    ⚠ `src` stays a raw pointer for the int32-wrap reason documented on the
    un-augmented kernel."""
    comptime HW = IMG_H * IMG_W
    comptime CAM_ELEMS = 3 * HW
    comptime IMG_ELEMS = N_CAM * CAM_ELEMS
    var i = Int(global_idx.x)
    if i >= B * IMG_ELEMS:
        return
    var b = i // IMG_ELEMS
    var e = i % IMG_ELEMS
    var cam = e // CAM_ELEMS
    var within = e % CAM_ELEMS
    var ch = within // HW
    var pl = within % HW
    var y = pl // IMG_W
    var x = pl % IMG_W

    var p = SIMD[DType.float32, AUG_WORDS](0.0)
    var pbase = (b * N_CAM + cam) * AUG_WORDS
    comptime for w in range(AUG_WORDS):
        p[w] = Float32(rebind[Scalar[DT]](params[pbase + w]))

    var row = Int(g[b])
    var flat = row * IMG_ELEMS + cam * CAM_ELEMS + ch * HW + aug_src_index(
        p, y, x, IMG_H, IMG_W
    )
    var byte = aug_value_u8(p, src[unsafe_offset=flat], ch, y, x, within)

    var mean = Scalar[DT](IMAGENET_MEAN_R) if ch == 0 else (
        Scalar[DT](IMAGENET_MEAN_G) if ch == 1 else Scalar[DT](IMAGENET_MEAN_B)
    )
    var std = Scalar[DT](IMAGENET_STD_R) if ch == 0 else (
        Scalar[DT](IMAGENET_STD_G) if ch == 1 else Scalar[DT](IMAGENET_STD_B)
    )
    var v = Scalar[DT](Int(byte)) / Scalar[DT](255.0)
    out_img[i] = (v - mean) * (Scalar[DT](1.0) / std)


def _act_gather_qpos_kernel[
    B: Int, QPOS: Int
](
    src: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    g: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    mean: LayoutTensor[DT, Layout.row_major(QPOS), MutAnyOrigin],
    std: LayoutTensor[DT, Layout.row_major(QPOS), MutAnyOrigin],
    out_q: LayoutTensor[DT, Layout.row_major(B * QPOS), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i >= B * QPOS:
        return
    var b = i // QPOS
    var j = i % QPOS
    var flat = Int(g[b]) * QPOS + j
    out_q[i] = (
        rebind[Scalar[DT]](src[flat]) - rebind[Scalar[DT]](mean[j])
    ) / rebind[Scalar[DT]](std[j])


def _act_gather_actions_kernel[
    B: Int, K: Int, ADIM: Int
](
    src: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    g: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    n_real: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    mean: LayoutTensor[DT, Layout.row_major(ADIM), MutAnyOrigin],
    std: LayoutTensor[DT, Layout.row_major(ADIM), MutAnyOrigin],
    out_a: LayoutTensor[DT, Layout.row_major(B * K * ADIM), MutAnyOrigin],
    out_v: LayoutTensor[DT, Layout.row_major(B * K), MutAnyOrigin],
    n_rows: Int64,
):
    """Actions `[g, g+K)`, padded past the episode end.

    ⚠ The pad value is the NORMALIZED zero, `(0 - mean)/std`, NOT zero — the
    reference pads the raw action and normalizes afterwards
    (`utils.py`), and `data.mojo::_fill_one` reproduces that. Writing a plain
    0.0 here would be a silent distribution shift on every truncated chunk.
    """
    var i = Int(global_idx.x)
    if i >= B * K * ADIM:
        return
    var b = i // (K * ADIM)
    var rem = i % (K * ADIM)
    var t = rem // ADIM
    var j = rem % ADIM

    var valid = t < Int(n_real[b])
    var raw = Scalar[DT](0.0)
    if valid:
        var flat = Int64(Int(g[b]) + t) * Int64(ADIM) + Int64(j)
        raw = rebind[Scalar[DT]](src[Int(flat)])
    out_a[i] = (raw - rebind[Scalar[DT]](mean[j])) / rebind[Scalar[DT]](
        std[j]
    )
    if j == 0:
        out_v[b * K + t] = Scalar[DT](1.0) if valid else Scalar[DT](0.0)


# ── ACTDeviceDataset ────────────────────────────────────────────────────


struct ACTDeviceDataset[
    QPOS: Int, ADIM: Int, N_CAM: Int, IMG_H: Int, IMG_W: Int
](Movable):
    comptime HW = Self.IMG_H * Self.IMG_W
    comptime CAM_ELEMS = 3 * Self.HW
    comptime IMG_ELEMS = Self.N_CAM * Self.CAM_ELEMS

    var images_u8: TensorImpl[U8]
    """[n_rows, IMG_ELEMS] uint8 — the whole image set, uploaded once."""
    var qpos_raw: Tensor
    var action_raw: Tensor
    var qpos_mean: Tensor
    var qpos_std: Tensor
    var action_mean: Tensor
    var action_std: Tensor
    var ep_start: TensorImpl[I32]
    var ep_len: TensorImpl[I32]
    var train_eps: TensorImpl[I32]
    var val_eps: TensorImpl[I32]
    var n_train: Int
    var n_val: Int
    var n_rows: Int

    var g: TensorImpl[I32]
    """[B] chosen flat row per batch slot — the ONE indirection everything
    else reads through, and the seam a windowed tier-2 would replace."""
    var n_real: TensorImpl[I32]
    var rng_offset: TensorImpl[U64]
    var offset_host: UInt64
    """Host MIRROR of `rng_offset`, exact because the advance is deterministic
    (`+= 2*B` per `sample`). It exists so a caller can save and restore the
    stream WITHOUT a D2H — which validation needs: `act_so101_train_gpu`
    pins the sampler around every validation pass so each one scores the same
    batches, and without that `best_val` selects the luckiest draw rather than
    the best model."""
    var seed: UInt64

    var aug: ImageAugConfig
    """Training-batch augmentation. `off` (the default) runs the original
    gather kernel; validation batches are NEVER augmented."""
    var aug_params: Tensor
    """[B, N_CAM, AUG_WORDS] — the records the last augmented gather used."""

    def __init__(out self):
        self.images_u8 = TensorImpl[U8]()
        self.qpos_raw = Tensor()
        self.action_raw = Tensor()
        self.qpos_mean = Tensor()
        self.qpos_std = Tensor()
        self.action_mean = Tensor()
        self.action_std = Tensor()
        self.ep_start = TensorImpl[I32]()
        self.ep_len = TensorImpl[I32]()
        self.train_eps = TensorImpl[I32]()
        self.val_eps = TensorImpl[I32]()
        self.n_train = 0
        self.n_val = 0
        self.n_rows = 0
        self.g = TensorImpl[I32]()
        self.n_real = TensorImpl[I32]()
        self.rng_offset = TensorImpl[U64]()
        self.offset_host = 0
        self.seed = 0
        self.aug = ImageAugConfig.off()
        self.aug_params = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.images_u8 = move.images_u8^
        self.qpos_raw = move.qpos_raw^
        self.action_raw = move.action_raw^
        self.qpos_mean = move.qpos_mean^
        self.qpos_std = move.qpos_std^
        self.action_mean = move.action_mean^
        self.action_std = move.action_std^
        self.ep_start = move.ep_start^
        self.ep_len = move.ep_len^
        self.train_eps = move.train_eps^
        self.val_eps = move.val_eps^
        self.n_train = move.n_train
        self.n_val = move.n_val
        self.n_rows = move.n_rows
        self.g = move.g^
        self.n_real = move.n_real^
        self.rng_offset = move.rng_offset^
        self.offset_host = move.offset_host
        self.seed = move.seed
        self.aug = move.aug
        self.aug_params = move.aug_params^

    @staticmethod
    def upload_from[
        B: Int
    ](
        mut host: ACTDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
        ctx: DeviceContext,
        seed: UInt64 = 0x2545F4914F6CDD1D,
    ) raises -> Self:
        """Upload the whole dataset once. `B` sizes the per-batch scratch.

        ⚠ Images stream through a bounded pinned buffer rather than being
        materialised in host RAM — the host store is deliberately NOT resident
        (7.1 GB of RSS for a 460 KB row was the reason), so reading it all into
        a `List` first would reintroduce exactly the cost that decision
        avoided.
        """
        var d = Self()
        d.seed = seed
        d.n_rows = host.store.n_rows()

        # ── images: chunked HDF5 read -> pinned staging -> device ────────
        var total = d.n_rows * Self.IMG_ELEMS
        d.images_u8.ensure_gpu(ctx, total)
        var chunk = List[Scalar[U8]](
            length=UPLOAD_CHUNK_ROWS * Self.IMG_ELEMS, fill=0
        )
        var staged = ctx.enqueue_create_host_buffer[U8](
            UPLOAD_CHUNK_ROWS * Self.IMG_ELEMS
        )
        ctx.synchronize()
        var r = 0
        while r < d.n_rows:
            var end = r + UPLOAD_CHUNK_ROWS
            if end > d.n_rows:
                end = d.n_rows
            var rows = end - r
            var n = rows * Self.IMG_ELEMS
            host._img_dset.read_range[U8](r, end, mptr(chunk))
            for i in range(n):
                staged[i] = chunk[i]
            var dst = d.images_u8.dev.value().create_sub_buffer[U8](
                r * Self.IMG_ELEMS, n
            )
            var srcw = staged.create_sub_buffer[U8](0, n)
            ctx.enqueue_copy(dst, srcw)
            ctx.synchronize()  # `staged` is reused next chunk
            r = end

        # ── the small tables ────────────────────────────────────────────
        d.qpos_raw = _up_f32(host.qpos_raw, ctx)
        d.action_raw = _up_f32(host.action_raw, ctx)
        d.qpos_mean = _up_f32(host.qpos_mean, ctx)
        d.qpos_std = _up_f32(host.qpos_std, ctx)
        d.action_mean = _up_f32(host.action_mean, ctx)
        d.action_std = _up_f32(host.action_std, ctx)

        var n_eps = host.store.episodes.n_episodes()
        var starts = List[Scalar[I32]](length=n_eps, fill=0)
        var lens = List[Scalar[I32]](length=n_eps, fill=0)
        for e in range(n_eps):
            starts[e] = Int32(host.store.episodes.start_of(e))
            lens[e] = Int32(host.store.episodes.length_of(e))
        d.ep_start = _up_i32(starts, ctx)
        d.ep_len = _up_i32(lens, ctx)

        d.n_train = len(host.train_eps)
        d.n_val = len(host.val_eps)
        var tr = List[Scalar[I32]](length=d.n_train if d.n_train > 0 else 1, fill=0)
        for i in range(d.n_train):
            tr[i] = Int32(host.train_eps[i])
        var va = List[Scalar[I32]](length=d.n_val if d.n_val > 0 else 1, fill=0)
        for i in range(d.n_val):
            va[i] = Int32(host.val_eps[i])
        d.train_eps = _up_i32(tr, ctx)
        d.val_eps = _up_i32(va, ctx)

        d.g = TensorImpl[I32].alloc_gpu(ctx, B)
        d.n_real = TensorImpl[I32].alloc_gpu(ctx, B)
        d.rng_offset = TensorImpl[U64].alloc_gpu(ctx, 1)
        # Allocated even while augmentation is off, so turning it on later
        # allocates nothing (a capture region cannot).
        d.aug_params = Tensor.alloc_gpu(ctx, B * Self.N_CAM * AUG_WORDS)
        ctx.synchronize()
        return d^

    def set_augment(mut self, cfg: ImageAugConfig):
        """Augment TRAINING batches from now on (`off` restores the original
        gather). Validation batches are never augmented: `best_val` must rank
        models, not draws."""
        self.aug = cfg

    def sample[
        B: Int, K: Int
    ](
        mut self,
        val: Bool,
        mut out_qpos: Tensor,
        mut out_images: Tensor,
        mut out_actions: Tensor,
        mut out_valid: Tensor,
        ctx: DeviceContext,
    ) raises:
        """Draw + gather a batch. NO host work, NO H2D, NO synchronization."""
        comptime nb = (B + TPB - 1) // TPB
        ctx.enqueue_function[_act_draw_kernel[B, K]](
            self.g.lt["gpu", Layout.row_major(B)](),
            self.n_real.lt["gpu", Layout.row_major(B)](),
            (self.val_eps if val else self.train_eps).lt[
                "gpu", Layout.row_major(B)
            ](),
            self.ep_start.lt["gpu", Layout.row_major(B)](),
            self.ep_len.lt["gpu", Layout.row_major(B)](),
            Int32(self.n_val if val else self.n_train),
            self.seed,
            self.rng_offset.lt["gpu", Layout.row_major(1)](),
            grid_dim=nb,
            block_dim=TPB,
        )
        var augment = self.aug.enabled and not val
        if augment:
            comptime na = (B * Self.N_CAM + TPB - 1) // TPB
            ctx.enqueue_function[
                _act_aug_draw_kernel[B, Self.N_CAM, Self.IMG_H, Self.IMG_W]
            ](
                self.aug_params.lt[
                    "gpu", Layout.row_major(B * Self.N_CAM * AUG_WORDS)
                ](),
                self.seed,
                self.rng_offset.lt["gpu", Layout.row_major(1)](),
                self.aug.brightness,
                self.aug.contrast,
                self.aug.gamma,
                self.aug.gain,
                self.aug.noise_sigma,
                Int32(self.aug.max_shift),
                self.aug.cutout_prob,
                self.aug.cutout_max_frac,
                grid_dim=na,
                block_dim=TPB,
            )
        ctx.enqueue_function[_act_advance_offset_kernel[B]](
            self.rng_offset.lt["gpu", Layout.row_major(1)](),
            grid_dim=1,
            block_dim=1,
        )
        self.offset_host += UInt64(B * 2)
        self._gather[B, K](
            out_qpos, out_images, out_actions, out_valid, ctx, augment
        )

    def note_replayed_sample[B: Int](mut self):
        """Advance the HOST mirror for a `sample` that ran as a graph REPLAY.

        Under CUDA-graph capture the offset-advance kernel is inside the
        captured region and runs on every replay; the `self.offset_host += 2*B`
        line in `sample` is host code and does not. Left unsynced, the mirror
        drifts behind the device by one draw per replay — and the damage
        surfaces at the NEXT `set_offset` restore, which would rewind the
        device stream to an offset training has already consumed, silently
        re-drawing batches. Nothing about the loss curve would look wrong.

        ⚠ Must be called exactly once per replayed step, and NOT for a step
        that actually ran (`sample` already advanced it then)."""
        self.offset_host += UInt64(B * 2)

    def set_offset(mut self, ctx: DeviceContext, v: UInt64) raises:
        """Pin the RNG stream. Set to a fixed value before a validation pass
        and restore afterwards, so every pass draws the SAME batches."""
        self.rng_offset.dev.value().enqueue_fill(v)
        self.offset_host = v

    def gather_at[
        B: Int, K: Int
    ](
        mut self,
        rows: List[Int],
        n_reals: List[Int],
        mut out_qpos: Tensor,
        mut out_images: Tensor,
        mut out_actions: Tensor,
        mut out_valid: Tensor,
        ctx: DeviceContext,
    ) raises:
        """Gather EXPLICIT rows — the parity entry point.

        The device RNG is Philox and the host's is a xorshift, so no seed makes
        the two samplers agree and a "same batch" gate is impossible. This is
        what IS comparable: the same `(row, n_real)` must produce the same
        tensors as `ACTDataset.fill_at`."""
        self.g.ensure(B)
        self.n_real.ensure(B)
        for b in range(B):
            self.g.data[b] = Int32(rows[b])
            self.n_real.data[b] = Int32(n_reals[b])
        self.g.upload_resident(ctx)
        self.n_real.upload_resident(ctx)
        self._gather[B, K](out_qpos, out_images, out_actions, out_valid, ctx)

    def gather_at_augmented[
        B: Int, K: Int
    ](
        mut self,
        rows: List[Int],
        n_reals: List[Int],
        params: List[Scalar[DT]],
        mut out_qpos: Tensor,
        mut out_images: Tensor,
        mut out_actions: Tensor,
        mut out_valid: Tensor,
        ctx: DeviceContext,
    ) raises:
        """`gather_at` with EXPLICIT augmentation records
        (`[B, N_CAM, AUG_WORDS]`) — the parity entry point for the augmented
        kernel against `augment_camera_u8` + `normalize_camera_chw`."""
        if len(params) != B * Self.N_CAM * AUG_WORDS:
            raise Error("gather_at_augmented: params must be B*N_CAM*AUG_WORDS")
        self.g.ensure(B)
        self.n_real.ensure(B)
        for b in range(B):
            self.g.data[b] = Int32(rows[b])
            self.n_real.data[b] = Int32(n_reals[b])
        self.aug_params.ensure(B * Self.N_CAM * AUG_WORDS)
        for i in range(len(params)):
            self.aug_params.data[i] = params[i]
        self.g.upload_resident(ctx)
        self.n_real.upload_resident(ctx)
        self.aug_params.upload_resident(ctx)
        self._gather[B, K](
            out_qpos, out_images, out_actions, out_valid, ctx, True
        )

    def _gather[
        B: Int, K: Int
    ](
        mut self,
        mut out_qpos: Tensor,
        mut out_images: Tensor,
        mut out_actions: Tensor,
        mut out_valid: Tensor,
        ctx: DeviceContext,
        augment: Bool = False,
    ) raises:
        out_qpos.ensure_gpu(ctx, B * Self.QPOS)
        out_images.ensure_gpu(ctx, B * Self.IMG_ELEMS)
        out_actions.ensure_gpu(ctx, B * K * Self.ADIM)
        out_valid.ensure_gpu(ctx, B * K)

        comptime nimg = (B * Self.IMG_ELEMS + TPB - 1) // TPB
        if augment:
            ctx.enqueue_function[
                _act_gather_images_aug_kernel[
                    B, Self.N_CAM, Self.IMG_H, Self.IMG_W
                ]
            ](
                self.images_u8.dev.value(),
                self.g.lt["gpu", Layout.row_major(B)](),
                self.aug_params.lt[
                    "gpu", Layout.row_major(B * Self.N_CAM * AUG_WORDS)
                ](),
                out_images.lt["gpu", Layout.row_major(B * Self.IMG_ELEMS)](),
                Int64(self.n_rows),
                grid_dim=nimg,
                block_dim=TPB,
            )
        else:
            ctx.enqueue_function[
                _act_gather_images_kernel[
                    B, Self.IMG_ELEMS, Self.CAM_ELEMS, Self.HW
                ]
            ](
                self.images_u8.dev.value(),
                self.g.lt["gpu", Layout.row_major(B)](),
                out_images.lt["gpu", Layout.row_major(B * Self.IMG_ELEMS)](),
                Int64(self.n_rows),
                grid_dim=nimg,
                block_dim=TPB,
            )
        comptime nq = (B * Self.QPOS + TPB - 1) // TPB
        ctx.enqueue_function[_act_gather_qpos_kernel[B, Self.QPOS]](
            self.qpos_raw.lt["gpu", Layout.row_major(1)](),
            self.g.lt["gpu", Layout.row_major(B)](),
            self.qpos_mean.lt["gpu", Layout.row_major(Self.QPOS)](),
            self.qpos_std.lt["gpu", Layout.row_major(Self.QPOS)](),
            out_qpos.lt["gpu", Layout.row_major(B * Self.QPOS)](),
            grid_dim=nq,
            block_dim=TPB,
        )
        comptime na = (B * K * Self.ADIM + TPB - 1) // TPB
        ctx.enqueue_function[_act_gather_actions_kernel[B, K, Self.ADIM]](
            self.action_raw.lt["gpu", Layout.row_major(1)](),
            self.g.lt["gpu", Layout.row_major(B)](),
            self.n_real.lt["gpu", Layout.row_major(B)](),
            self.action_mean.lt["gpu", Layout.row_major(Self.ADIM)](),
            self.action_std.lt["gpu", Layout.row_major(Self.ADIM)](),
            out_actions.lt["gpu", Layout.row_major(B * K * Self.ADIM)](),
            out_valid.lt["gpu", Layout.row_major(B * K)](),
            Int64(self.n_rows),
            grid_dim=na,
            block_dim=TPB,
        )


def _up_f32(ref src: List[Scalar[DT]], ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.alloc(len(src))
    for i in range(len(src)):
        t.data[i] = src[i]
    t.upload_resident(ctx)
    return t^


def _up_i32(
    ref src: List[Scalar[I32]], ctx: DeviceContext
) raises -> TensorImpl[I32]:
    var t = TensorImpl[I32]()
    t.ensure(len(src))
    for i in range(len(src)):
        t.data[i] = src[i]
    t.upload_resident(ctx)
    return t^
