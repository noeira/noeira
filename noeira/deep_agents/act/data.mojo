# +--------------------------------------------------------------------------+ #
# | ACT episodic sampler over a TrajectoryStore
# +--------------------------------------------------------------------------+ #
"""One observation, one chunk of future actions — the ACT training sample.

Mirrors `EpisodicDataset.__getitem__` (`references/act-main/utils.py:24`):
pick an episode, pick a start timestep uniformly inside it, take the
observation AT that step and the next `K` actions FROM it, zero-padding past
the episode end.

Input is a `TrajectoryStore` `.h5` written by `tools/act/lerobot_v3_to_store.py`
with columns `qpos` (S,), `action` (A,), `images` (C,3,H,W) uint8.

## Images are resident only while they fit

`qpos` and `action` are always resident — 15 k rows of 6 floats is 370 KB. The
image column is the one that scales with the recording, and it is the one that
stops fitting: 5 episodes at 240x320 is 878 MB, but 50 episodes is **7.1 GiB**,
which does not belong in RAM on a 16 GB machine alongside the model.

So residency is a decision, taken on the column's byte count against
`max_image_bytes`, and `images_resident` reports which way it went. The
streamed path reads ONE row per sample with `read_range` on a held-open
handle.

`store.mojo`'s residency note quotes scattered per-row HDF5 gathers at ~4500x
the cost of the same gather from RAM — but that ratio was measured on a state
column whose row is a few floats, where the per-row call overhead IS the cost.
An image row here is 460,800 bytes in one chunk, and a random 8-row gather
measures **4 ms** (1.0 GB/s) — bandwidth-bound, the overhead gone. Next to a
ResNet18 forward+backward over the same 8 samples that is free, so streaming
costs the training loop nothing it can measure. Do not carry the 4500x
conclusion across to a fat column; it is a statement about narrow rows.

## Two things the reference does that look like bugs and are not

1. **Padding happens BEFORE normalization.** `utils.py:52` zero-fills
   `padded_action` and only then applies `(a - mean)/std`, so a padded slot
   holds `-mean/std`, not 0. It never reaches the loss (masked) or the CVAE
   encoder (key-padding-masked), so it is unobservable — but reproducing it
   keeps the reference gate exact instead of nearly-exact.

2. **The L1 denominator counts padded slots.** See `l1_per_sample`; this file
   only has to deliver the mask.

## One thing it does that IS specific to ALOHA

`utils.py:45` shifts the action window by one for real-robot data
(`action = root['/action'][max(0, start_ts - 1):]`, commented "hack, to make
timesteps more aligned"). That compensates for a misalignment in ALOHA's own
recorder. LeRobot writes `action[t]` alongside `observation.state[t]` already,
so the shift is NOT applied here. Deviation, deliberate.

## Mask polarity

The buffer carries `valid` — **1.0 = a real action, 0.0 = padding** — which is
the INVERSE of the reference's `is_pad`. One name, one meaning; the loss node
multiplies by it and the attention node turns it into an additive bias.
"""

from std.memory import alloc, dealloc
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.data.store import TrajectoryStore
from noeira.io.hdf5 import H5Dataset

from .config import NORM_STD_FLOOR, TRAIN_SPLIT_RATIO
from .inference import normalize_camera_chw
from .augment import (
    AUG_UNIFORMS, AUG_WORDS, ImageAugConfig, aug_param, augment_camera_u8,
)


# ── residency budget ─────────────────────────────────────────────────────
# 2 GiB. Big enough that every store built so far at 240x320 from a few dozen
# episodes stays resident (5 episodes = 878 MB), small enough that the 50
# episode store (7.1 GiB) streams instead of pushing a 16 GB machine into
# swap. Not a hardware probe: a caller who knows its box can pass its own
# `max_image_bytes`, and the streamed path is correct at any size.
comptime IMAGES_RESIDENT_MAX_BYTES: Int = 2 << 30


# ── deterministic RNG ────────────────────────────────────────────────────
# splitmix64. Local and seeded rather than the global `random_float64()` the
# CPU samplers in `data/sampler.mojo` use, because the dataset gates compare
# against a recorded index sequence and must reproduce it exactly.


@always_inline
def _splitmix64(mut state: UInt64) -> UInt64:
    state += UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def _rand_below(mut state: UInt64, n: Int) -> Int:
    """Uniform in `[0, n)`. `n <= 0` is a caller bug, not a sample."""
    return Int(_splitmix64(state) % UInt64(n))


# ── the dataset ──────────────────────────────────────────────────────────


struct ACTDataset[
    QPOS: Int, ADIM: Int, N_CAM: Int, IMG_H: Int, IMG_W: Int
](Movable & Deinitable):
    """Columns + normalization statistics + the episode split.

    ONE instance serves both splits (`sample_batch[..](val=…)`); holding two
    would duplicate the image column — or its file handle — for no reason.

    `qpos`/`action` are always resident. `images` is resident only if the
    column fits `max_image_bytes`; otherwise it is streamed a row at a time.
    Read `images_resident` to know which happened.
    """

    comptime CAM_ELEMS: Int = 3 * Self.IMG_H * Self.IMG_W
    comptime IMG_ELEMS: Int = Self.N_CAM * Self.CAM_ELEMS

    var store: TrajectoryStore
    var qpos_raw: List[Scalar[DT]]
    """[n_rows, QPOS] — UNNORMALIZED, as recorded (lerobot degrees / 0-100)."""
    var action_raw: List[Scalar[DT]]
    var images_raw: List[Scalar[DType.uint8]]
    """[n_rows, N_CAM, 3, IMG_H, IMG_W] uint8 CHW — EMPTY when streaming."""
    var images_resident: Bool
    """False = one `read_range` per sample from `_img_dset` into `_img_row`."""
    var _img_dset: H5Dataset
    """Held open either way: reopening per row is the one overhead that does
    NOT amortise over a 460 KB read."""
    var _img_row: List[Scalar[DType.uint8]]
    """[IMG_ELEMS] staging for one streamed row — empty when resident."""

    var qpos_mean: List[Scalar[DT]]
    var qpos_std: List[Scalar[DT]]
    var action_mean: List[Scalar[DT]]
    var action_std: List[Scalar[DT]]

    var train_eps: List[Int]
    var val_eps: List[Int]
    var rng: UInt64

    var aug: ImageAugConfig
    """Training-batch augmentation (`set_augment`); `off` by default.
    Validation batches and `fill_at` are never augmented."""
    var aug_rng: UInt64
    """The augmentation's OWN stream, separate from `rng`, so turning
    augmentation on does not move which rows `sample_batch` draws."""
    var last_rows: List[Int]
    """The flat rows the last `sample_batch` drew, slot order — what a gate
    re-derives the batch from."""
    var aug_last: List[SIMD[DType.float32, AUG_WORDS]]
    """`[BATCH * N_CAM]` records of the last augmented batch — what a gate
    compares the images against."""

    # ── where does `sample_batch` actually spend its time? ───────────────
    # It is 16.1 ms of a 144.8 ms ACT iteration with the GPU idle throughout,
    # and it does two very different things: an HDF5 row read (I/O, 7.37 MB of
    # uint8 per batch) and a per-pixel normalize (7.37M elements, scalar,
    # single-threaded, writing 29.5 MB of float32). ONLY THE SECOND CAN MOVE
    # TO THE GPU, and moving it would also cut the H2D transfer 4x by sending
    # uint8 instead of float32 — so the split decides whether that rewrite is
    # worth anything. Accumulated in nanoseconds, never reset; the caller
    # takes differences.
    var ns_img_io: Int
    """Time inside `read_range` (streamed) — the part that CANNOT move."""
    var ns_img_norm: Int
    """Time in the per-pixel uint8 -> normalized float32 loop — the part that
    can, and that a GPU kernel would do in ~0.04 ms."""

    def __init__(
        out self,
        var path: String,
        seed: UInt64 = 0,
        max_image_bytes: Int = IMAGES_RESIDENT_MAX_BYTES,
    ) raises:
        self.ns_img_io = 0
        self.ns_img_norm = 0
        self.store = TrajectoryStore(path^)

        ref m = self.store.manifest
        var q_spec = m.column(String("qpos"))
        var a_spec = m.column(String("action"))
        var i_spec = m.column(String("images"))
        if q_spec.row_dim() != Self.QPOS:
            raise Error(
                "ACTDataset: store 'qpos' is width "
                + String(q_spec.row_dim())
                + " but QPOS=" + String(Self.QPOS)
            )
        if a_spec.row_dim() != Self.ADIM:
            raise Error(
                "ACTDataset: store 'action' is width "
                + String(a_spec.row_dim())
                + " but ADIM=" + String(Self.ADIM)
            )
        if i_spec.row_dim() != Self.IMG_ELEMS:
            raise Error(
                "ACTDataset: store 'images' is "
                + String(i_spec.row_dim())
                + " bytes/row but N_CAM*3*H*W=" + String(Self.IMG_ELEMS)
                + " — the store was built at a different resolution or camera"
                " count. Rebuild with tools/act/lerobot_v3_to_store.py."
            )

        self.qpos_raw = self.store.load_column[DT](String("qpos"))
        self.action_raw = self.store.load_column[DT](String("action"))

        # Residency is decided on the column's actual size, not on the type of
        # column it is. The handle is opened either way: the resident path
        # never touches it, and paying an H5Dopen once costs nothing, while
        # making it conditional would need an Optional field for no gain.
        self._img_dset = self.store.open_column[DType.uint8](String("images"))
        var img_bytes = self.store.n_rows() * Self.IMG_ELEMS
        self.images_resident = img_bytes <= max_image_bytes
        if self.images_resident:
            self.images_raw = self.store.load_column[DType.uint8](
                String("images"), max_bytes=max_image_bytes
            )
            self._img_row = List[Scalar[DType.uint8]]()
        else:
            self.images_raw = List[Scalar[DType.uint8]]()
            self._img_row = List[Scalar[DType.uint8]](
                unsafe_uninit_length=Self.IMG_ELEMS
            )

        # Statistics are RECOMPUTED here rather than read from the converter's
        # `norm_*` datasets. `utils.py:get_norm_stats` is four lines; a second
        # independent implementation is what makes the gate against the
        # converter's numpy version mean something.
        self.qpos_mean = List[Scalar[DT]]()
        self.qpos_std = List[Scalar[DT]]()
        self.action_mean = List[Scalar[DT]]()
        self.action_std = List[Scalar[DT]]()
        self.train_eps = List[Int]()
        self.val_eps = List[Int]()
        self.rng = seed if seed != 0 else UInt64(0x2545F4914F6CDD1D)
        self.aug = ImageAugConfig.off()
        self.aug_rng = self.rng ^ UInt64(0xA06D1A7E5EED0001)
        self.aug_last = List[SIMD[DType.float32, AUG_WORDS]]()
        self.last_rows = List[Int]()

        var n = self.store.n_rows()
        _moments(self.qpos_raw, n, Self.QPOS, self.qpos_mean, self.qpos_std)
        _moments(
            self.action_raw, n, Self.ADIM, self.action_mean, self.action_std
        )
        self._split_episodes()

    def __init__(out self, *, deinit move: Self):
        self.ns_img_io = move.ns_img_io
        self.ns_img_norm = move.ns_img_norm
        self.store = move.store^
        self.qpos_raw = move.qpos_raw^
        self.action_raw = move.action_raw^
        self.images_raw = move.images_raw^
        self.images_resident = move.images_resident
        self._img_dset = move._img_dset^
        self._img_row = move._img_row^
        self.qpos_mean = move.qpos_mean^
        self.qpos_std = move.qpos_std^
        self.action_mean = move.action_mean^
        self.action_std = move.action_std^
        self.train_eps = move.train_eps^
        self.val_eps = move.val_eps^
        self.rng = move.rng
        self.aug = move.aug
        self.aug_rng = move.aug_rng
        self.aug_last = move.aug_last^
        self.last_rows = move.last_rows^

    def _split_episodes(mut self) raises:
        """`utils.py:112` — shuffle episode ids, first 80% train.

        The reference uses `np.random.permutation`; a Fisher-Yates over the
        local splitmix stream is the same construction with a reproducible
        stream. With 5 episodes this is 4 train / 1 validation.
        """
        var n_ep = self.store.n_episodes()
        var perm = List[Int](capacity=n_ep)
        for i in range(n_ep):
            perm.append(i)
        for i in range(n_ep - 1, 0, -1):
            var j = _rand_below(self.rng, i + 1)
            var t = perm[i]
            perm[i] = perm[j]
            perm[j] = t

        var n_train = Int(TRAIN_SPLIT_RATIO * Float64(n_ep))
        if n_train < 1:
            n_train = 1
        if n_train >= n_ep:
            # Everything in train leaves no validation split, and the
            # reference's model selection IS validation loss.
            n_train = n_ep - 1
        for i in range(n_ep):
            if i < n_train:
                self.train_eps.append(perm[i])
            else:
                self.val_eps.append(perm[i])

    def n_rows(self) -> Int:
        return self.store.n_rows()

    def set_augment(mut self, cfg: ImageAugConfig):
        """Augment TRAINING batches from `sample_batch` — the host twin of
        `ACTDeviceDataset.set_augment`: the same `aug_param` draw rule and
        the same `augment_camera_u8` per-pixel rule (gated device == host to
        one byte in `test_act_augment_gpu.mojo`), then the unchanged
        `normalize_camera_chw`. The uniforms come from `aug_rng`, not
        Philox, so a host batch and a device batch are not the same draws —
        only the same distribution."""
        self.aug = cfg
    def n_episodes(self) -> Int:
        return self.store.n_episodes()

    # ── sampling ─────────────────────────────────────────────────────────

    def sample_batch[
        K: Int, BATCH: Int
    ](
        mut self,
        val: Bool,
        mut out_qpos: List[Scalar[DT]],
        mut out_images: List[Scalar[DT]],
        mut out_actions: List[Scalar[DT]],
        mut out_valid: List[Scalar[DT]],
    ) raises:
        """Fill `BATCH` samples. Buffers are sized here if they are empty.

        `out_qpos`    [BATCH, QPOS]              normalized
        `out_images`  [BATCH, N_CAM, 3, H, W]    /255 then ImageNet-normalized
        `out_actions` [BATCH, K, ADIM]           normalized, padded per above
        `out_valid`   [BATCH, K]                 1.0 real / 0.0 padding
        """
        _ensure(out_qpos, BATCH * Self.QPOS)
        _ensure(out_images, BATCH * Self.IMG_ELEMS)
        _ensure(out_actions, BATCH * K * Self.ADIM)
        _ensure(out_valid, BATCH * K)

        ref eps = self.val_eps if val else self.train_eps
        if len(eps) == 0:
            raise Error("ACTDataset.sample_batch: the split is empty")

        # ⚠ AUGMENTATION DRAWS FROM ITS OWN STREAM (`aug_rng`), after the
        # row draws are untouched: the same seed samples the same rows with
        # augmentation on or off, and validation (never augmented) is pinned
        # through `rng` alone.
        var augment = self.aug.enabled and not val
        if augment:
            self.aug_last = List[SIMD[DType.float32, AUG_WORDS]]()
            for _ in range(BATCH * Self.N_CAM):
                var u = SIMD[DType.float32, AUG_UNIFORMS](0.0)
                for j in range(AUG_UNIFORMS):
                    u[j] = Float32(
                        Float64(Int(_splitmix64(self.aug_rng) >> 11))
                        * (1.0 / 9007199254740992.0)
                    )
                self.aug_last.append(
                    aug_param(self.aug, u, Self.IMG_H, Self.IMG_W)
                )
        self.last_rows = List[Int]()
        for b in range(BATCH):
            var ep = eps[_rand_below(self.rng, len(eps))]
            var ep_start = self.store.episodes.start_of(ep)
            var ep_len = self.store.episodes.length_of(ep)
            var start_ts = _rand_below(self.rng, ep_len)
            self.last_rows.append(ep_start + start_ts)
            self._fill_one[K](b, ep_start + start_ts, ep_len - start_ts,
                              out_qpos, out_images, out_actions, out_valid,
                              augment)

    def fill_at[
        K: Int
    ](
        mut self,
        slot: Int,
        ep: Int,
        start_ts: Int,
        mut out_qpos: List[Scalar[DT]],
        mut out_images: List[Scalar[DT]],
        mut out_actions: List[Scalar[DT]],
        mut out_valid: List[Scalar[DT]],
    ) raises:
        """Deterministic single-sample fill at an explicit `(episode, step)`.

        The open-loop evaluation walks an episode step by step, and the gates
        need a sample they can name. Buffers must already be sized.
        """
        var ep_start = self.store.episodes.start_of(ep)
        var ep_len = self.store.episodes.length_of(ep)
        if start_ts < 0 or start_ts >= ep_len:
            raise Error(
                "ACTDataset.fill_at: step " + String(start_ts)
                + " outside episode " + String(ep) + " (length "
                + String(ep_len) + ")"
            )
        self._fill_one[K](slot, ep_start + start_ts, ep_len - start_ts,
                          out_qpos, out_images, out_actions, out_valid)

    def image_row_u8(
        mut self, g: Int, mut out: List[Scalar[DType.uint8]]
    ) raises:
        """Row `g`'s images as the store holds them: `[N_CAM, 3, H, W]` uint8.

        The UNNORMALIZED bytes, which is what makes this useful: it is the
        only way to LOOK at what the policy was trained on.
        `act_so101_deploy_real.mojo --snap` writes these beside the live
        camera's frames so an operator can confirm that physical camera 0 is
        the camera that filled slot 0 — a mix-up nothing else can detect and
        that presents as a policy which simply does not work.
        """
        if g < 0 or g >= self.store.n_rows():
            raise Error(
                "ACTDataset.image_row_u8: row " + String(g) + " of "
                + String(self.store.n_rows())
            )
        if len(out) != Self.IMG_ELEMS:
            out = List[Scalar[DType.uint8]](
                unsafe_uninit_length=Self.IMG_ELEMS
            )
        if self.images_resident:
            var base = g * Self.IMG_ELEMS
            for i in range(Self.IMG_ELEMS):
                out[i] = self.images_raw[base + i]
        else:
            self._img_dset.read_range[DType.uint8](g, g + 1, mptr(out))

    def _fill_one[
        K: Int
    ](
        mut self,
        slot: Int,
        g: Int,
        remaining: Int,
        mut out_qpos: List[Scalar[DT]],
        mut out_images: List[Scalar[DT]],
        mut out_actions: List[Scalar[DT]],
        mut out_valid: List[Scalar[DT]],
        augment: Bool = False,
    ) raises:
        """`g` = flat row of the observation; `remaining` = steps left in its
        episode (so `min(K, remaining)` actions are real). `augment` applies
        `aug_last[slot * N_CAM + cam]` to each camera before the normalise."""

        # qpos at g.
        var qo = slot * Self.QPOS
        for j in range(Self.QPOS):
            out_qpos[qo + j] = (
                self.qpos_raw[g * Self.QPOS + j] - self.qpos_mean[j]
            ) / self.qpos_std[j]

        # Actions [g, g+K), padded. The pad value is the NORMALIZED zero,
        # `(0 - mean)/std`, exactly as `utils.py` produces it by padding first.
        var n_real = K if remaining > K else remaining
        var ao = slot * K * Self.ADIM
        for t in range(K):
            var valid = t < n_real
            for j in range(Self.ADIM):
                var raw = (
                    self.action_raw[(g + t) * Self.ADIM + j]
                    if valid
                    else Scalar[DT](0.0)
                )
                out_actions[ao + t * Self.ADIM + j] = (
                    raw - self.action_mean[j]
                ) / self.action_std[j]
            out_valid[slot * K + t] = Scalar[DT](
                1.0
            ) if valid else Scalar[DT](0.0)

        # Images at g: uint8 CHW -> /255 -> per-channel ImageNet normalize.
        # Resident: read row `g` in place. Streamed: pull that one row (one
        # HDF5 chunk, contiguous) into the staging buffer, which then holds it
        # at offset 0. The `ref` is bound AFTER the read so the read still has
        # `_img_row` mutably; both branches leave one buffer + one base index
        # for the loop below, so the per-pixel work is branch-free.
        var src: Int
        if self.images_resident:
            src = g * Self.IMG_ELEMS
        else:
            var t_io0 = perf_counter_ns()
            self._img_dset.read_range[DType.uint8](
                g, g + 1, mptr(self._img_row)
            )
            self.ns_img_io += perf_counter_ns() - t_io0
            src = 0
        ref img = self.images_raw if self.images_resident else self._img_row

        var io = slot * Self.IMG_ELEMS
        var t_nm0 = perf_counter_ns()
        # ⚠ THE PER-PIXEL RULE LIVES IN `inference.mojo`, NOT HERE. It used to
        # be written out inline, and the same rule is needed by
        # `act_so101_deploy_real.mojo` to prepare a LIVE camera frame. Two
        # copies of "/255 then ImageNet" is the defect shape this repo hits
        # most often, and its failure here is invisible: a policy that behaves
        # on the dataset and worse on the robot, with nothing raising.
        var cam_u8 = List[Scalar[DType.uint8]]()
        if augment:
            cam_u8 = List[Scalar[DType.uint8]](length=Self.CAM_ELEMS, fill=0)
        for c in range(Self.N_CAM):
            var cbase = c * Self.CAM_ELEMS
            if augment:
                # the augmented uint8 frame, THEN the one shared normalise
                augment_camera_u8[Self.IMG_H, Self.IMG_W](
                    img, src + cbase, self.aug_last[slot * Self.N_CAM + c],
                    cam_u8, 0,
                )
                normalize_camera_chw[Self.IMG_H, Self.IMG_W](
                    cam_u8, 0, out_images, io + cbase
                )
            else:
                normalize_camera_chw[Self.IMG_H, Self.IMG_W](
                    img, src + cbase, out_images, io + cbase
                )
        self.ns_img_norm += perf_counter_ns() - t_nm0


# ── helpers ──────────────────────────────────────────────────────────────


def _ensure(mut buf: List[Scalar[DT]], n: Int):
    """Size a caller buffer once; a reused buffer of the right size is left
    alone (the sampler runs every training step)."""
    if len(buf) != n:
        buf = List[Scalar[DT]](unsafe_uninit_length=n)


def _moments(
    ref data: List[Scalar[DT]],
    n_rows: Int,
    dim: Int,
    mut mean: List[Scalar[DT]],
    mut std: List[Scalar[DT]],
) raises:
    """Per-column mean and UNBIASED std, floored at `NORM_STD_FLOOR`.

    ⚠ ddof = 1. `torch.std` is unbiased by default and `utils.py` uses it;
    `np.std` is not. The ratio is only sqrt(N/(N-1)) — 1.00025 at N=1997, far
    below anything a single check would flag — but it is a systematic offset
    that would then sit under every comparison against the reference.
    """
    if n_rows < 2:
        raise Error("ACTDataset: need at least 2 rows to compute statistics")
    mean = List[Scalar[DT]](unsafe_uninit_length=dim)
    std = List[Scalar[DT]](unsafe_uninit_length=dim)
    for j in range(dim):
        var s = Float64(0.0)
        for i in range(n_rows):
            s += Float64(data[i * dim + j])
        var mu = s / Float64(n_rows)
        var ss = Float64(0.0)
        for i in range(n_rows):
            var d = Float64(data[i * dim + j]) - mu
            ss += d * d
        var sd = (ss / Float64(n_rows - 1)) ** 0.5
        if sd < NORM_STD_FLOOR:
            sd = NORM_STD_FLOOR
        mean[j] = Scalar[DT](mu)
        std[j] = Scalar[DT](sd)
