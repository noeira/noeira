# +--------------------------------------------------------------------------+ #
# | LeRobot v3.0 dataset -> TrajectoryStore, entirely in Mojo
# +--------------------------------------------------------------------------+ #
"""Imports a downloaded LeRobot v3.0 dataset into one `.h5` store.

    import_lerobot_v3(root, out_path, height=240, width=320)

This is the Mojo replacement for `tools/act/lerobot_v3_to_store.py`. That
script needed `huggingface_hub`, `pyarrow`, `imageio`, `Pillow`, `numpy` and
`h5py`; this needs the `ffmpeg` binary and nothing else, because the pieces it
used to import now exist natively:

    pyarrow   -> `noeira/io/parquet`   (thrift + snappy + RLE + PLAIN)
    json      -> `noeira/io/json`
    imageio   -> `noeira/io/video`     (a pipe from `ffmpeg`, not a struct FFI)
    Pillow    -> `noeira/io/image`     (bit-exact BILINEAR)
    h5py      -> `noeira/data/store`   (already native)

## What LeRobot v3.0 looks like

    meta/info.json                        features, fps, total_{episodes,frames}
    meta/episodes/chunk-*/file-*.parquet  per-episode index + per-camera video map
    data/chunk-*/file-*.parquet           action / observation.state / timestamps
    videos/<key>/chunk-*/file-*.mp4       MANY episodes concatenated per file

The episode index carries `dataset_from_index` / `dataset_to_index` — the flat
row range, i.e. exactly `ep_offset` / `ep_len` — and, per camera, the
`(chunk_index, file_index, from_timestamp)` locating that episode inside the
packed mp4. `round(from_timestamp * fps)` is the episode's first frame index
within its video file.

## The one structural constraint

⚠ **THE STORE WRITER IS APPEND-ONLY, AND THE VIDEO IS SEQUENTIAL.** The Python
converter sidestepped this by pre-creating the images dataset and writing rows
by index; `TrajectoryStoreWriter` cannot, and buffering the images to reorder
them is exactly the 7.1 GB allocation that converter was rewritten to remove.

Both constraints are satisfiable at once because LeRobot writes each camera's
episodes into its video files IN DATASET ORDER. So one decoder per camera is
opened, all are advanced in lockstep with the episode loop, and rows are
written as they are produced — peak memory is one native frame per camera
(921 KB at 480x640) rather than the whole image column.

That ordering is a property of the writer, not of the format, so it is
CHECKED, not assumed: `_CameraPlan` raises if any camera's episodes are not in
non-decreasing `(file ordinal, first frame)` order. Assuming it and being wrong
would silently pair every episode with another episode's video.
"""

from std.math import sqrt
from std.os import listdir, makedirs
from std.os.path import exists, isdir
from std.pathlib import Path

from noeira.io.fileio import file_size, parent_dir, rename_over
from noeira.io.proc import quote_arg, run_capture
from noeira.io.hf import (
    HF_DATASET,
    hf_client,
    hf_hub_cache,
    hf_tree,
    noeira_cache,
    path_prefix,
    repo_slug,
)
from noeira.io.image import resize_bilinear_pil
from noeira.vision.calib_file import read_calib
from noeira.vision.fisheye import FisheyeLens, Pinhole, UndistortMap
from noeira.io.json import J_ARRAY, JsonDoc, load_json, parse_json
from noeira.io.parquet import ParquetFile
from noeira.io.video import VideoDecoder

from .column import ColumnSpec
from .lerobot_rejected import is_rejected, kept_rows, load_rejected_episodes
from .store import TrajectoryStoreWriter


comptime STD_FLOOR = 1e-2
"""`references/act-main/utils.py:96`, `torch.clip(std, 1e-2, inf)`."""


# ══════════════════════════════════════════════════════════════════════════
# meta/info.json
# ══════════════════════════════════════════════════════════════════════════

struct LeRobotInfo(Movable):
    var codebase_version: String
    var fps: Int
    var total_frames: Int
    var total_episodes: Int
    var cameras: List[String]
    """Video feature keys, sorted by name — the camera slot order in the
    store's `images` column. Sorted rather than file-order so the layout does
    not depend on JSON key ordering."""
    var state_dim: Int
    var action_dim: Int

    def __init__(out self, *, deinit move: Self):
        self.codebase_version = move.codebase_version^
        self.fps = move.fps
        self.total_frames = move.total_frames
        self.total_episodes = move.total_episodes
        self.cameras = move.cameras^
        self.state_dim = move.state_dim
        self.action_dim = move.action_dim

    def __init__(out self, root: String) raises:
        var doc = load_json(root + "/meta/info.json")
        var r = doc.root()

        self.codebase_version = doc.string(
            doc.field(r, String("codebase_version"))
        )
        if not self.codebase_version.startswith("v3"):
            raise Error(
                "lerobot: this importer targets v3.x; meta/info.json says '"
                + self.codebase_version + "'"
            )
        self.fps = doc.integer(doc.field(r, String("fps")))
        self.total_frames = doc.integer(doc.field(r, String("total_frames")))
        self.total_episodes = doc.integer(
            doc.field(r, String("total_episodes"))
        )

        var feats = doc.field(r, String("features"))
        if feats < 0:
            raise Error("lerobot: meta/info.json has no 'features'")
        self.cameras = List[String]()
        self.state_dim = 0
        self.action_dim = 0
        for i in range(doc.size(feats)):
            var key = doc.key_at(feats, i)
            var v = doc.at(feats, i)
            var dt = doc.string(doc.field(v, String("dtype")))
            if dt == "video":
                self.cameras.append(key^)
            elif key == "observation.state":
                self.state_dim = doc.integer(
                    doc.at(doc.field(v, String("shape")), 0)
                )
            elif key == "action":
                self.action_dim = doc.integer(
                    doc.at(doc.field(v, String("shape")), 0)
                )
        _sort_strings(self.cameras)

        if self.state_dim <= 0 or self.action_dim <= 0:
            raise Error(
                "lerobot: meta/info.json declares state_dim="
                + String(self.state_dim) + " action_dim="
                + String(self.action_dim)
            )
        if len(self.cameras) == 0:
            raise Error("lerobot: no features with dtype 'video'")


def _sort_strings(mut xs: List[String]):
    """Insertion sort. The list is a handful of camera names."""
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            xs.swap_elements(j - 1, j)
            j -= 1


# ══════════════════════════════════════════════════════════════════════════
# chunk-XXX/file-YYY.<ext> enumeration
# ══════════════════════════════════════════════════════════════════════════

@fieldwise_init
struct ChunkFile(Copyable, ImplicitlyCopyable, Movable):
    var chunk: Int
    var file: Int
    var path: String


def _parse_numbered(name: String, prefix: String) -> Int:
    """`chunk-007` -> 7, `file-012.parquet` -> 12; -1 if it does not match.

    Digits are read until the first non-digit, so the extension needs no
    special case and a stray `chunk-000.bak` still parses as 0 rather than
    being silently dropped from the file list.
    """
    if not name.startswith(prefix):
        return -1
    var b = name.as_bytes()
    var i = prefix.byte_length()
    var v = -1
    while i < name.byte_length():
        var c = Int(b[i])
        if c < 0x30 or c > 0x39:
            break
        if v < 0:
            v = 0
        v = v * 10 + (c - 0x30)
        i += 1
    return v


def list_chunk_files(base: String, ext: String) raises -> List[ChunkFile]:
    """`<base>/chunk-*/file-*.<ext>`, sorted by (chunk, file)."""
    if not isdir(base):
        raise Error("lerobot: missing directory " + base)
    var out = List[ChunkFile]()
    var chunks = listdir(Path(base))
    for ci in range(len(chunks)):
        var cname = String(chunks[ci])
        var cnum = _parse_numbered(cname, String("chunk-"))
        if cnum < 0:
            continue
        var cdir = base + "/" + cname
        if not isdir(cdir):
            continue
        var files = listdir(Path(cdir))
        for fi in range(len(files)):
            var fname = String(files[fi])
            if not fname.endswith("." + ext):
                continue
            var fnum = _parse_numbered(fname, String("file-"))
            if fnum < 0:
                continue
            out.append(ChunkFile(cnum, fnum, cdir + "/" + fname))
    # Insertion sort by (chunk, file); a dataset has tens of these, not
    # thousands, and the order decides row order for the whole import.
    for i in range(1, len(out)):
        var j = i
        while j > 0 and (
            out[j - 1].chunk > out[j].chunk
            or (out[j - 1].chunk == out[j].chunk and out[j - 1].file > out[j].file)
        ):
            out.swap_elements(j - 1, j)
            j -= 1
    if len(out) == 0:
        raise Error("lerobot: no chunk-*/file-*." + ext + " under " + base)
    return out^


# ══════════════════════════════════════════════════════════════════════════
# meta/episodes/*.parquet
# ══════════════════════════════════════════════════════════════════════════

struct EpisodeIndex(Movable):
    var length: List[Int]
    var from_index: List[Int]
    var to_index: List[Int]
    var vid_chunk: List[List[Int]]
    """`[camera][episode]`."""
    var vid_file: List[List[Int]]
    var vid_from_ts: List[List[Float64]]

    def __init__(out self, *, deinit move: Self):
        self.length = move.length^
        self.from_index = move.from_index^
        self.to_index = move.to_index^
        self.vid_chunk = move.vid_chunk^
        self.vid_file = move.vid_file^
        self.vid_from_ts = move.vid_from_ts^

    def __init__(out self, root: String, cameras: List[String]) raises:
        var files = list_chunk_files(root + "/meta/episodes", String("parquet"))

        var ep_id = List[Int]()
        var raw_len = List[Int]()
        var raw_from = List[Int]()
        var raw_to = List[Int]()
        var raw_chunk = List[List[Int]]()
        var raw_file = List[List[Int]]()
        var raw_ts = List[List[Float64]]()
        for _ in range(len(cameras)):
            raw_chunk.append(List[Int]())
            raw_file.append(List[Int]())
            raw_ts.append(List[Float64]())

        for fi in range(len(files)):
            var pf = ParquetFile(String(files[fi].path))
            var idx = pf.read_i64(String("episode_index"))
            var ln = pf.read_i64(String("length"))
            var f0 = pf.read_i64(String("dataset_from_index"))
            var f1 = pf.read_i64(String("dataset_to_index"))
            if len(ln) != len(idx) or len(f0) != len(idx) or len(f1) != len(idx):
                raise Error(
                    "lerobot: " + files[fi].path + " has ragged episode columns"
                )
            for i in range(len(idx)):
                ep_id.append(Int(idx[i]))
                raw_len.append(Int(ln[i]))
                raw_from.append(Int(f0[i]))
                raw_to.append(Int(f1[i]))
            for c in range(len(cameras)):
                var pre = String("videos/") + cameras[c] + "/"
                var ci = pf.read_i64(pre + "chunk_index")
                var fj = pf.read_i64(pre + "file_index")
                var ts = pf.read_f64(pre + "from_timestamp")
                if len(ci) != len(idx) or len(fj) != len(idx) or len(ts) != len(idx):
                    raise Error(
                        "lerobot: camera '" + cameras[c] + "' has ragged video"
                        " columns in " + files[fi].path
                    )
                for i in range(len(idx)):
                    raw_chunk[c].append(Int(ci[i]))
                    raw_file[c].append(Int(fj[i]))
                    raw_ts[c].append(ts[i])

        # ── permute into episode_index order ──────────────────────────
        # `episode_index` must be a permutation of 0..n-1. Building the inverse
        # permutation both sorts in O(n) and PROVES that, which an argsort
        # would not: a duplicated id sorts fine and then silently drops an
        # episode.
        var n = len(ep_id)
        var order = List[Int](unsafe_uninit_length=n)
        for i in range(n):
            order[i] = -1
        for i in range(n):
            var e = ep_id[i]
            if e < 0 or e >= n:
                raise Error(
                    "lerobot: episode_index " + String(e) + " outside [0, "
                    + String(n) + ")"
                )
            if order[e] != -1:
                raise Error(
                    "lerobot: episode_index " + String(e) + " appears twice in"
                    " the episode index"
                )
            order[e] = i

        self.length = List[Int](unsafe_uninit_length=n)
        self.from_index = List[Int](unsafe_uninit_length=n)
        self.to_index = List[Int](unsafe_uninit_length=n)
        self.vid_chunk = List[List[Int]]()
        self.vid_file = List[List[Int]]()
        self.vid_from_ts = List[List[Float64]]()
        for _ in range(len(cameras)):
            self.vid_chunk.append(List[Int](unsafe_uninit_length=n))
            self.vid_file.append(List[Int](unsafe_uninit_length=n))
            self.vid_from_ts.append(List[Float64](unsafe_uninit_length=n))

        for e in range(n):
            var i = order[e]
            self.length[e] = raw_len[i]
            self.from_index[e] = raw_from[i]
            self.to_index[e] = raw_to[i]
            for c in range(len(cameras)):
                self.vid_chunk[c][e] = raw_chunk[c][i]
                self.vid_file[c][e] = raw_file[c][i]
                self.vid_from_ts[c][e] = raw_ts[c][i]

        # ── the flat row ranges must tile [0, N) exactly ──────────────
        for e in range(n):
            if self.to_index[e] - self.from_index[e] != self.length[e]:
                raise Error(
                    "lerobot: episode " + String(e) + " has length "
                    + String(self.length[e]) + " but rows ["
                    + String(self.from_index[e]) + ", "
                    + String(self.to_index[e]) + ")"
                )
        if n > 0 and self.from_index[0] != 0:
            raise Error("lerobot: episode 0 does not start at row 0")
        for e in range(1, n):
            if self.from_index[e] != self.to_index[e - 1]:
                raise Error(
                    "lerobot: a gap or overlap between episodes "
                    + String(e - 1) + " and " + String(e)
                    + " — flat ep_offset would be wrong"
                )

    def n_episodes(self) -> Int:
        return len(self.length)

    def total_rows(self) -> Int:
        var n = 0
        for i in range(len(self.length)):
            n += self.length[i]
        return n


# ══════════════════════════════════════════════════════════════════════════
# data/*.parquet — the flat per-frame table
# ══════════════════════════════════════════════════════════════════════════

struct FrameTable(Movable):
    var qpos: List[Float32]
    var action: List[Float32]
    var task_index: List[Int32]
    """The per-frame `task_index`, straight from the parquet.

    ⚠⚠ PER FRAME, NOT PER EPISODE. LeRobot stores it per row and a consumer
    doing multi-task work needs it there — a per-episode id would be strictly
    less general and would have to be re-derived. Dropping it, which this
    importer did until now, makes every frame of a multi-task dataset look
    like one task and NOTHING says so: the store loads, the shapes are right,
    and a VLA trains every sample against whichever instruction the consumer
    happened to pick.
    """
    var state_dim: Int
    var action_dim: Int
    var n_rows: Int

    def __init__(out self, *, deinit move: Self):
        self.qpos = move.qpos^
        self.action = move.action^
        self.task_index = move.task_index^
        self.state_dim = move.state_dim
        self.action_dim = move.action_dim
        self.n_rows = move.n_rows

    def __init__(out self, root: String, state_dim: Int, action_dim: Int) raises:
        var files = list_chunk_files(root + "/data", String("parquet"))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qpos = List[Float32]()
        self.action = List[Float32]()
        self.task_index = List[Int32]()
        self.n_rows = 0

        for fi in range(len(files)):
            var pf = ParquetFile(String(files[fi].path))
            var rows = pf.num_rows()
            var q = pf.read_f64(String("observation.state.list.element"))
            var a = pf.read_f64(String("action.list.element"))
            # ⚠ A SCALAR COLUMN, so no `.list.element` suffix — the two above
            # are LIST columns and the reader spells the leaf path. Using the
            # list spelling here returns nothing and the length check below
            # reports it as a row-count mismatch, which points at the file
            # rather than at the path.
            var ti = pf.read_i64(String("task_index"))
            if len(q) != rows * state_dim:
                raise Error(
                    "lerobot: " + files[fi].path + " has " + String(len(q))
                    + " observation.state values for " + String(rows)
                    + " rows x " + String(state_dim)
                )
            if len(a) != rows * action_dim:
                raise Error(
                    "lerobot: " + files[fi].path + " has " + String(len(a))
                    + " action values for " + String(rows) + " rows x "
                    + String(action_dim)
                )
            # Float32 is the source dtype; the widening to Float64 in the
            # parquet reader is exact, so this narrows back losslessly.
            for i in range(len(q)):
                self.qpos.append(Float32(q[i]))
            for i in range(len(a)):
                self.action.append(Float32(a[i]))
            # ⚠ REQUIRED, NOT OPTIONAL. Every LeRobot v3 frame table carries
            # `task_index`; a file without one is not a v3 dataset, and
            # silently defaulting to 0 is exactly the failure this column
            # exists to prevent.
            if len(ti) != rows:
                raise Error(
                    "lerobot: " + files[fi].path + " has " + String(len(ti))
                    + " task_index values for " + String(rows) + " rows."
                    " Every v3 frame table carries one per frame; a store"
                    " written without it makes a multi-task dataset look"
                    " single-task with nothing to say so."
                )
            for i in range(len(ti)):
                self.task_index.append(Int32(ti[i]))
            self.n_rows += rows

    def check_episode_order(self, root: String, index: EpisodeIndex) raises:
        """`episode_index` must run 0,0,…,1,1,… in episode order.

        The flat `ep_offset`/`ep_len` the store writes only mean anything if
        the rows are already grouped that way. This reads the column back out
        of the parquet rather than trusting the layout.
        """
        var files = list_chunk_files(root + "/data", String("parquet"))
        var seen = 0
        var ep = 0
        var left = index.length[0] if index.n_episodes() > 0 else 0
        for fi in range(len(files)):
            var pf = ParquetFile(String(files[fi].path))
            var col = pf.read_i64(String("episode_index"))
            for i in range(len(col)):
                while left == 0 and ep + 1 < index.n_episodes():
                    ep += 1
                    left = index.length[ep]
                if Int(col[i]) != ep:
                    raise Error(
                        "lerobot: flat row " + String(seen) + " belongs to"
                        " episode " + String(Int(col[i])) + ", but the episode"
                        " index puts it in episode " + String(ep)
                        + " — the data rows are not grouped by episode in"
                        " index order"
                    )
                left -= 1
                seen += 1
        if seen != index.total_rows():
            raise Error(
                "lerobot: data/*.parquet holds " + String(seen) + " rows, the"
                " episode index sums to " + String(index.total_rows())
            )


# ══════════════════════════════════════════════════════════════════════════
# Video streaming, one decoder per camera
# ══════════════════════════════════════════════════════════════════════════

struct CameraStream(Movable):
    """One camera's video files, read strictly forward.

    Holds the open `ffmpeg` for the current file plus a scratch native frame.
    `seek_forward` advances by decoding and discarding, which is the only
    thing a pipe can do — and is free here, because the frames it skips are
    almost always none (episodes are contiguous inside a file).
    """

    var camera: String
    var root: String
    var dec: Optional[VideoDecoder]
    var chunk: Int
    var file: Int
    var cursor: Int
    """Next frame index within the open file."""
    var raw: List[UInt8]
    var width: Int
    var height: Int
    var skipped: Int
    var decoded: Int

    def __init__(out self, var camera: String, var root: String):
        self.camera = camera^
        self.root = root^
        self.dec = None
        self.chunk = -1
        self.file = -1
        self.cursor = 0
        self.raw = List[UInt8]()
        self.width = 0
        self.height = 0
        self.skipped = 0
        self.decoded = 0

    def __init__(out self, *, deinit move: Self):
        self.camera = move.camera^
        self.root = move.root^
        self.dec = move.dec^
        self.chunk = move.chunk
        self.file = move.file
        self.cursor = move.cursor
        self.raw = move.raw^
        self.width = move.width
        self.height = move.height
        self.skipped = move.skipped
        self.decoded = move.decoded

    def _path(self, chunk: Int, file: Int) -> String:
        return (
            self.root + "/videos/" + self.camera + "/chunk-" + _pad3(chunk)
            + "/file-" + _pad3(file) + ".mp4"
        )

    def open_at(mut self, chunk: Int, file: Int, frame: Int) raises:
        """Position on `frame` of `(chunk, file)`, opening it if needed."""
        if chunk != self.chunk or file != self.file:
            if self.dec:
                self.dec.value().close()
            var p = self._path(chunk, file)
            if not exists(p):
                raise Error("lerobot: missing video file " + p)
            var d = VideoDecoder(p^)
            self.width = d.width
            self.height = d.height
            self.raw.resize(d.frame_bytes, UInt8(0))
            self.dec = d^
            self.chunk = chunk
            self.file = file
            self.cursor = 0
        if frame < self.cursor:
            raise Error(
                "lerobot: camera '" + self.camera + "' was asked to go back to"
                " frame " + String(frame) + " of chunk-" + _pad3(chunk)
                + "/file-" + _pad3(file) + " after reaching "
                + String(self.cursor) + "; this importer streams forward only"
            )
        while self.cursor < frame:
            if not self.dec.value().next_into(_uptr(self.raw)):
                raise Error(
                    "lerobot: camera '" + self.camera + "' ran out of frames"
                    " seeking to " + String(frame) + " in chunk-"
                    + _pad3(chunk) + "/file-" + _pad3(file)
                )
            self.cursor += 1
            self.skipped += 1

    def next_native(mut self) raises:
        """Decode one frame into `self.raw` (HWC, RGB24, source resolution)."""
        if not self.dec:
            raise Error("lerobot: camera '" + self.camera + "' has no open file")
        if not self.dec.value().next_into(_uptr(self.raw)):
            raise Error(
                "lerobot: camera '" + self.camera + "' ran out of frames in"
                " chunk-" + _pad3(self.chunk) + "/file-" + _pad3(self.file)
                + " after " + String(self.cursor) + " frames — the episode"
                " index and the video disagree"
            )
        self.cursor += 1
        self.decoded += 1

    def close(mut self) raises:
        if self.dec:
            self.dec.value().close()
            self.dec = None


def _pad3(v: Int) -> String:
    var s = String(v)
    while s.byte_length() < 3:
        s = "0" + s
    return s^


@always_inline
def _uptr(
    mut lst: List[UInt8],
) -> Pointer[Scalar[DType.uint8], MutAnyOrigin]:
    return (
        lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
        .as_unsafe_any_origin()
    )


@always_inline
def _fptr(
    mut lst: List[Float32],
) -> Pointer[Scalar[DType.float32], MutAnyOrigin]:
    return (
        lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
        .as_unsafe_any_origin()
    )


@always_inline
def read_task_table(root: String) raises -> List[Tuple[Int, String]]:
    """`meta/tasks.parquet` -> `[(task_index, task text)]`, BYTE-EXACT.

    ⚠⚠ THE TEXT IS PASSED THROUGH UNCHANGED — no trim, no normalisation, no
    re-encode. A consumer tokenises it, and lerobot's own pipeline appends
    "\n" BEFORE tokenising (`NewLineTaskProcessorStep`), so "Grab the green
    cube" is six ids ending in 198, not five. A store that normalised the text
    would let an equality gate against a token table either fail on a cosmetic
    difference or — worse — PASS while the ids came from different bytes than
    the store records.

    ⚠ `task` IS A BYTE_ARRAY COLUMN, so it comes back as one concatenated
    buffer plus offsets rather than a list of strings — see
    `ParquetFile.read_byte_arrays` for why (CIFAR's 60k blobs). The bytes are
    copied out span by span and NUL-terminated, never walked with `chr`, which
    would re-encode every byte above 127.
    """
    var path = root + "/meta/tasks.parquet"
    var pf = ParquetFile(path)
    var idx = pf.read_i64(String("task_index"))
    var vals = List[UInt8]()
    var offs = List[Int]()
    var n = pf.read_byte_arrays(String("task"), vals, offs)
    if n != len(idx):
        raise Error(
            "lerobot: " + path + " has " + String(len(idx))
            + " task_index values and " + String(n) + " task strings"
        )
    var out = List[Tuple[Int, String]]()
    for i in range(n):
        var b = List[UInt8]()
        for k in range(offs[i], offs[i + 1]):
            b.append(vals[k])
        b.append(0)
        out.append(
            (Int(idx[i]), String(unsafe_from_utf8_ptr=b.unsafe_ptr()))
        )
    return out^


@always_inline
def _iptr(
    mut lst: List[Int32],
) -> Pointer[Scalar[DType.int32], MutAnyOrigin]:
    return (
        lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.int32]]()
        .as_unsafe_any_origin()
    )


def _check_forward_only(
    index: EpisodeIndex, cameras: List[String], fps: Int
) raises:
    """Every camera's episodes must sit at non-decreasing (file, frame).

    ⚠ THE WHOLE STREAMING DESIGN RESTS ON THIS and the format does not
    guarantee it — LeRobot's writer happens to append episodes in order. If it
    ever stops doing so, decoding forward pairs each episode with a DIFFERENT
    episode's frames, which produces a perfectly well-formed store of
    mislabelled images. Checked up front, before any decoding, so the failure
    costs nothing.
    """
    for c in range(len(cameras)):
        var last_chunk = -1
        var last_file = -1
        var last_frame = -1
        for e in range(index.n_episodes()):
            var ch = index.vid_chunk[c][e]
            var fl = index.vid_file[c][e]
            var fr = Int(round(index.vid_from_ts[c][e] * Float64(fps)))
            var newer = (
                ch > last_chunk
                or (ch == last_chunk and fl > last_file)
            )
            if not newer and (ch != last_chunk or fl != last_file):
                raise Error(
                    "lerobot: camera '" + cameras[c] + "' episode " + String(e)
                    + " goes back to chunk-" + _pad3(ch) + "/file-"
                    + _pad3(fl) + " after chunk-" + _pad3(last_chunk)
                    + "/file-" + _pad3(last_file)
                )
            if ch == last_chunk and fl == last_file and fr < last_frame:
                raise Error(
                    "lerobot: camera '" + cameras[c] + "' episode " + String(e)
                    + " starts at frame " + String(fr) + ", before the previous"
                    " episode's " + String(last_frame) + " in the same file"
                )
            last_chunk = ch
            last_file = fl
            last_frame = fr + index.length[e]


# ══════════════════════════════════════════════════════════════════════════
# Normalisation statistics
# ══════════════════════════════════════════════════════════════════════════

def norm_stats(
    x: List[Float32], dim: Int, mut mean: List[Float32], mut std: List[Float32]
) raises:
    """`get_norm_stats` (`references/act-main/utils.py:78`), generalized.

    ⚠ **ACCUMULATE IN FLOAT64, AND USE ddof=1.** Both were learned the hard
    way on the Python side: a float32 accumulator drifted 2.3e-3 at 15,447
    rows, and `torch.std` is UNBIASED by default while `np.std` is not. The
    bias gap is only sqrt(N/(N-1)) — 1.00003 here — but it is systematic, and
    a store whose statistics differ from the reference's puts a constant offset
    under every comparison made afterwards.
    """
    var n = len(x) // dim
    if n <= 1:
        raise Error("lerobot: need at least two rows for a std")
    mean.resize(dim, Float32(0))
    std.resize(dim, Float32(0))
    for d in range(dim):
        var s = 0.0
        for i in range(n):
            s += Float64(x[i * dim + d])
        var m = s / Float64(n)
        var acc = 0.0
        for i in range(n):
            var e = Float64(x[i * dim + d]) - m
            acc += e * e
        var sd = sqrt(acc / Float64(n - 1))
        if sd < STD_FLOOR:
            sd = STD_FLOOR
        mean[d] = Float32(m)
        std[d] = Float32(sd)


# ══════════════════════════════════════════════════════════════════════════
# The import
# ══════════════════════════════════════════════════════════════════════════

def import_lerobot_v3(
    root: String,
    out_path: String,
    height: Int = 240,
    width: Int = 320,
    var env_id: String = String(""),
    var source_commit: String = String(""),
    verbose: Bool = True,
    undistort_dir: String = String(""),
    undistort_fovy: Float64 = 73.7398,
) raises:
    """Convert a downloaded LeRobot v3.0 dataset at `root` into `out_path`.

    `undistort_dir` (optional): each camera's native frame is first brought
    to the SIM's pinhole — `vision/fisheye.mojo`, fovy `undistort_fovy`
    (the rig's `overhead_cam` / `wrist_cam`) at the native size — from
    `<undistort_dir>/camera_<name>.txt`, `<name>` being the camera key's
    last dotted part (`observation.images.overhead` -> `overhead`), THEN
    resized as always. A calibration that is missing, not `fisheye`, of
    another size, or whose map leaves the lens, is REFUSED: a store half in
    one camera model and half in another is not a store.

    Writes to `<out_path>.tmp` and renames on success, so an interrupted run
    never leaves a half-built store where the next one would find it.
    """
    if not isdir(root):
        raise Error("lerobot: no dataset directory at " + root)

    var info = LeRobotInfo(root)
    if env_id == "":
        env_id = String("lerobot/") + root

    if verbose:
        var cams = String("")
        for i in range(len(info.cameras)):
            if i > 0:
                cams += ", "
            cams += info.cameras[i]
        print(
            "[1/4] " + info.codebase_version + "  fps=" + String(info.fps)
            + "  state=" + String(info.state_dim)
            + "  action=" + String(info.action_dim)
        )
        print("      cameras: " + cams + "  -> " + String(height) + "x"
              + String(width))

    # ── metadata ──────────────────────────────────────────────────────
    if verbose:
        print("[2/4] reading parquet ...")
    var index = EpisodeIndex(root, info.cameras)
    var frames = FrameTable(root, info.state_dim, info.action_dim)
    frames.check_episode_order(root, index)
    _check_forward_only(index, info.cameras, info.fps)

    var n_ep = index.n_episodes()
    var n_rows = frames.n_rows
    if n_rows != info.total_frames:
        raise Error(
            "lerobot: data holds " + String(n_rows) + " rows, meta/info.json"
            " says " + String(info.total_frames)
        )
    if n_ep != info.total_episodes:
        raise Error(
            "lerobot: " + String(n_ep) + " episodes in the index,"
            " meta/info.json says " + String(info.total_episodes)
        )
    if index.total_rows() != n_rows:
        raise Error(
            "lerobot: episode lengths sum to " + String(index.total_rows())
            + ", the data has " + String(n_rows) + " rows"
        )
    if verbose:
        print(
            "      " + String(n_rows) + " frames over " + String(n_ep)
            + " episodes"
        )

    # ⚠⚠ EPISODES THE OPERATOR REJECTED ARE NOT IMPORTED — neither their
    # frames nor their share of the normalisation statistics. They are still
    # in the videos and the parquet (the writer is append-only); see
    # `lerobot_rejected.mojo`. The consistency checks above ran on the FULL
    # dataset first, so a rejection can never hide a malformed one.
    var rejected = load_rejected_episodes(String(root))
    for r in rejected:
        if r >= n_ep:
            raise Error(
                "lerobot: meta/rejected_episodes.json rejects episode "
                + String(r) + ", but the dataset has " + String(n_ep)
            )
    if len(rejected) == n_ep:
        raise Error("lerobot: every episode is rejected; nothing to import")
    if verbose and len(rejected) > 0:
        var names = String("")
        for r in rejected:
            names += (", " if names.byte_length() > 0 else "") + String(r)
        print(
            "      skipping " + String(len(rejected))
            + " rejected episode(s): " + names
        )

    # ── the fisheye -> sim-pinhole maps, one per camera — checked BEFORE the
    # store exists, so a missing calibration leaves no partial file ──────
    var undistort = undistort_dir.byte_length() > 0
    var maps = List[UndistortMap]()
    var und = List[UInt8]()
    var calib_paths = List[String]()
    if undistort:
        for c in range(len(info.cameras)):
            var key = String(info.cameras[c])
            var parts = key.split(".")
            var short = String(parts[len(parts) - 1])
            var cp = undistort_dir + "/camera_" + short + ".txt"
            if not exists(cp):
                raise Error(
                    "lerobot: --undistort needs " + cp + " for camera '" + key
                    + "' (examples/vision/calibrate_fisheye.mojo --name " + short
                    + ")"
                )
            var cal = read_calib(cp)
            var lens = FisheyeLens.from_calib(cal)
            var um = UndistortMap(
                lens, Pinhole.sim(undistort_fovy, cal.width, cal.height)
            )
            if um.n_outside > 0:
                raise Error(
                    "lerobot: " + cp + ": " + String(um.n_outside) + " pixels of"
                    " the " + String(undistort_fovy) + "-degree pinhole fall"
                    " outside the lens — they would be black in every frame"
                )
            if verbose:
                print("      undistort " + key + ": " + String(lens))
            maps.append(um^)
            calib_paths.append(cp)

    # ── store ─────────────────────────────────────────────────────────
    var n_cam = len(info.cameras)
    var cam_elems = 3 * height * width
    var row_elems = n_cam * cam_elems

    var columns = List[ColumnSpec]()
    columns.append(ColumnSpec(String("qpos"), DType.float32, info.state_dim))
    columns.append(ColumnSpec(String("action"), DType.float32, info.action_dim))
    # ⚠ INT32, NOT INT64. The parquet is i64 and this narrows: LeRobot's task
    # tables are tens of rows, and the column is per FRAME, so the width is
    # paid on every row of the store. A dataset with more than 2^31 tasks is
    # not a thing; a store twice the size for a column of small integers is.
    columns.append(ColumnSpec(String("task_index"), DType.int32, 1))
    var img_shape = List[Int]()
    img_shape.append(n_cam)
    img_shape.append(3)
    img_shape.append(height)
    img_shape.append(width)
    columns.append(ColumnSpec(String("images"), DType.uint8, img_shape^))

    var tmp = out_path + ".tmp"

    # ⚠⚠ THE DISK IS CHECKED BEFORE THE FIRST FRAME IS DECODED. The first
    # 50-episode store (19,365 rows, 8.9 GB) filled a rented box's disk and
    # failed only at the final flush, as "H5Fflush failed: ret=-1", after every
    # video had been decoded. The images column dominates the size, so the
    # estimate is exact to within the small columns and HDF5's overhead; 5% of
    # headroom covers both.
    var rows_to_write = 0
    for e in range(n_ep):
        if not is_rejected(rejected, e):
            rows_to_write += index.length[e]
    var need = Int(Float64(rows_to_write * row_elems) * 1.05) + 64_000_000
    if exists(tmp):
        # A previous failed import's partial store is replaced by this one.
        need -= file_size(tmp)
    var free = free_bytes(parent_dir(out_path))
    if free >= 0 and free < need:
        raise Error(
            "lerobot: the store needs ~" + String(need // 1_000_000_000) + "."
            + String((need // 100_000_000) % 10) + " GB (" + String(rows_to_write)
            + " rows x " + String(n_cam) + " cameras x " + String(height) + "x"
            + String(width) + "x3) but " + parent_dir(out_path) + " has "
            + String(free // 1_000_000_000) + "." + String((free // 100_000_000) % 10)
            + " GB free. Free space, or pass --out on a larger disk."
            + (" A partial store from an earlier attempt is at " + tmp if exists(tmp) else "")
        )
    if verbose:
        print(
            "      store ~" + String(need // 1_000_000_000) + "."
            + String((need // 100_000_000) % 10) + " GB, "
            + (String(free // 1_000_000_000) + " GB free" if free >= 0 else "free space unknown")
        )

    if undistort_dir.byte_length() > 0:
        # ⚠ IN THE MANIFEST, so a store says which camera model its pixels
        # are in — a raw-fisheye store and an undistorted one have the same
        # columns and shapes and nothing else would tell them apart.
        source_commit += (
            " | images undistorted fisheye -> sim pinhole fovy "
            + String(undistort_fovy) + " (vision/fisheye.mojo) from "
            + undistort_dir + "/camera_<name>.txt"
        )
    var w = TrajectoryStoreWriter(
        String(tmp), columns^, env_id^, 0, source_commit^
    )
    # ⚠ THE TASK TABLE, so a `task_index` in the store can be RESOLVED. The
    # column alone says which task each frame is; only this says what those
    # indices mean. Without it a consumer needs a side-car table whose only
    # binding to the store is its filename — and if the two drift, index 0
    # resolves to a different instruction than the one a tokeniser saw, with
    # nothing anywhere to complain.
    var task_rows = read_task_table(String(root))
    for i in range(len(task_rows)):
        w.add_task(task_rows[i][0], String(task_rows[i][1]))
    if verbose:
        print("[2/4] task table: " + String(len(task_rows)) + " task(s)")

    # ── decode + write, one episode at a time ─────────────────────────
    if verbose:
        print("[3/4] decoding video (" + String(n_cam) + " camera(s)) ...")
    var streams = List[CameraStream]()
    for c in range(n_cam):
        streams.append(CameraStream(String(info.cameras[c]), String(root)))

    var row_img = List[UInt8](unsafe_uninit_length=row_elems)
    var hwc = List[UInt8](unsafe_uninit_length=cam_elems)
    var scratch = List[UInt8]()

    var written_eps = 0
    var written_rows = 0
    for e in range(n_ep):
        if is_rejected(rejected, e):
            continue
        var length = index.length[e]
        for c in range(n_cam):
            var first = Int(round(index.vid_from_ts[c][e] * Float64(info.fps)))
            streams[c].open_at(index.vid_chunk[c][e], index.vid_file[c][e], first)

        for t in range(length):
            var g = index.from_index[e] + t
            for c in range(n_cam):
                streams[c].next_native()
                var src_ptr = _uptr(streams[c].raw)
                if undistort:
                    if (
                        streams[c].width != maps[c].src_w
                        or streams[c].height != maps[c].src_h
                    ):
                        raise Error(
                            "lerobot: camera '" + String(info.cameras[c])
                            + "' decodes at " + String(streams[c].width) + "x"
                            + String(streams[c].height) + ", " + calib_paths[c]
                            + " was measured at " + String(maps[c].src_w) + "x"
                            + String(maps[c].src_h)
                        )
                    if len(und) != len(streams[c].raw):
                        und = List[UInt8](length=len(streams[c].raw), fill=0)
                    maps[c].apply_hwc(streams[c].raw, 0, 3, und, 0)
                    src_ptr = _uptr(und)
                resize_bilinear_pil(
                    src_ptr,
                    streams[c].height,
                    streams[c].width,
                    _uptr(hwc),
                    height,
                    width,
                    scratch,
                    3,
                )
                # HWC -> CHW, into this camera's slot of the flat row.
                var base = c * cam_elems
                for ch in range(3):
                    var dst0 = base + ch * height * width
                    for y in range(height):
                        var src0 = (y * width) * 3 + ch
                        var d0 = dst0 + y * width
                        for x in range(width):
                            row_img[d0 + x] = hwc[src0 + x * 3]

            w.append[DType.uint8](String("images"), _uptr(row_img), 1)
            w.append[DType.float32](
                String("qpos"),
                _fptr(frames.qpos).unsafe_offset(g * info.state_dim),
                1,
            )
            w.append[DType.float32](
                String("action"),
                _fptr(frames.action).unsafe_offset(g * info.action_dim),
                1,
            )
            w.append[DType.int32](
                String("task_index"),
                _iptr(frames.task_index).unsafe_offset(g),
                1,
            )
        w.end_episode()
        written_eps += 1
        written_rows += length
        if verbose and (written_eps % 10 == 1 or e == n_ep - 1):
            # ⚠ Counts of what is WRITTEN: with rejected episodes skipped, the
            # source indices ("episode 56/56 rows 21297") overstated the store.
            print(
                "      episode " + String(written_eps) + "/"
                + String(n_ep - len(rejected)) + "  rows " + String(written_rows)
            )

    for c in range(n_cam):
        streams[c].close()

    # ── statistics + finish ───────────────────────────────────────────
    if verbose:
        print("[4/4] norm stats, index and manifest ...")
    var qm = List[Float32]()
    var qs = List[Float32]()
    var am = List[Float32]()
    var as_ = List[Float32]()
    if len(rejected) == 0:
        norm_stats(frames.qpos, info.state_dim, qm, qs)
        norm_stats(frames.action, info.action_dim, am, as_)
    else:
        norm_stats(
            kept_rows(frames.qpos, info.state_dim, index.from_index, index.length, rejected),
            info.state_dim, qm, qs,
        )
        norm_stats(
            kept_rows(frames.action, info.action_dim, index.from_index, index.length, rejected),
            info.action_dim, am, as_,
        )

    w.write_vector[DType.float32](
        String("norm_qpos_mean"), _fptr(qm), info.state_dim
    )
    w.write_vector[DType.float32](
        String("norm_qpos_std"), _fptr(qs), info.state_dim
    )
    w.write_vector[DType.float32](
        String("norm_action_mean"), _fptr(am), info.action_dim
    )
    w.write_vector[DType.float32](
        String("norm_action_std"), _fptr(as_), info.action_dim
    )
    w.close()

    # ⚠ ATOMIC. The store is built at `<out>.tmp` and renamed only once it is
    # complete, so an interrupted import leaves the previous good store rather
    # than a plausible-looking truncated one.
    rename_over(String(tmp), String(out_path))
    if verbose:
        print("      wrote " + out_path)


# ══════════════════════════════════════════════════════════════════════════
# Getting the dataset onto the box, without huggingface_hub
# ══════════════════════════════════════════════════════════════════════════
# `huggingface_hub` is one `pip` dependency and one Python process; libcurl
# plus the Hub's public tree API is neither, and the JSON reader needed to walk
# the response already exists for `meta/info.json`.
#
#     GET https://huggingface.co/api/datasets/<repo>/tree/<rev>?recursive=1
#       -> [ {"type": "file"|"directory", "path": "...", "size": N, ...}, ... ]
#     GET https://huggingface.co/datasets/<repo>/resolve/<rev>/<path>
#       -> the bytes (302 to the CDN, so redirects must be followed)
#
# ⚠ ONE CLIENT FOR THE WHOLE REPO. This used to be one `curl` process per
# file: a fork, an exec and a fresh TLS handshake for each of several hundred
# chunk files. The connection, the TLS session and the DNS answer are now
# reused across the entire download.
#
# ⚠ Files are downloaded to `<dest>/<path>.part` and renamed on success, and an
# existing file whose size already matches the manifest is SKIPPED — so an
# interrupted 700 MB download resumes at file granularity instead of starting
# over. Size is a weak check next to a hash, but it is the one the tree listing
# gives for free on every file, LFS or not.


def hf_snapshot_path(repo: String, revision: String = String("main")) raises -> String:
    """The `huggingface_hub` cache path for `repo`, or "" if it is not there.

    Resolves `refs/<revision>` to the commit sha the way the hub cache does,
    so a dataset already pulled by a Python tool is reused rather than
    re-downloaded.
    """
    var base = (
        hf_hub_cache() + "/datasets--" + repo.replace("/", "--")
    )
    var ref_file = base + "/refs/" + revision
    if not exists(ref_file):
        return String("")
    var f = open(ref_file, "r")
    var sha = f.read()
    f.close()
    var sha_clean = String(sha.strip())
    var snap = base + "/snapshots/" + sha_clean
    if not isdir(snap):
        return String("")
    return snap^


def hf_download_dataset(
    repo: String,
    var dest: String = String(""),
    revision: String = String("main"),
    verbose: Bool = True,
    var token: String = String(""),
) raises -> String:
    """Download every file of a Hub dataset repo into a plain directory.

    ⚠ `token` IS EXPLICIT BECAUSE `.env` IS NOT THE ENVIRONMENT. This repo
    keeps secrets in a `.env` file that `load_dotenv` reads into a Dict — it
    never exports them — so a caller holding a token that way has to hand it
    over. Without it a PRIVATE repo answers 401 on the tree listing, which
    reads as a bad repo name rather than as missing auth.
    """
    if dest == "":
        dest = noeira_cache() + "/lerobot/" + repo_slug(repo)
    makedirs(dest, exist_ok=True)

    var client = hf_client(token.copy())
    if verbose:
        print("  listing " + repo + " @ " + revision + " ...")
    var listing = hf_tree(repo, HF_DATASET, revision, token.copy())
    var lbytes = List[UInt8]()
    for i in range(listing.byte_length()):
        lbytes.append(listing.as_bytes()[i])
    var doc = parse_json(lbytes^)
    var arr = doc.root()
    if doc.kind_of(arr) != J_ARRAY:
        raise Error(
            "lerobot: the Hub tree API did not return a list for '" + repo
            + "' — is the repo name right, and is it public (or HF_TOKEN set)?"
        )

    var n_files = 0
    var n_skipped = 0
    for i in range(doc.size(arr)):
        var ent = doc.at(arr, i)
        if doc.string(doc.field(ent, String("type"))) != "file":
            continue
        var rel = doc.string(doc.field(ent, String("path")))
        var size = -1
        var sn = doc.field(ent, String("size"))
        if sn >= 0:
            size = doc.integer(sn)

        var out = dest + "/" + rel
        var slash = out.rfind("/")
        if slash > 0:
            makedirs(path_prefix(out, slash), exist_ok=True)

        if exists(out) and size >= 0 and file_size(out) == size:
            n_skipped += 1
            continue

        var url = (
            "https://huggingface.co/datasets/" + repo + "/resolve/" + revision
            + "/" + rel
        )
        if verbose:
            print("  " + rel + "  (" + String(size) + " bytes)")
        var r = client.download(
            url, out + ".part", 0, rel if verbose else String("")
        )
        if not r.ok():
            raise Error(
                "lerobot: GET " + url + " -> " + String(r.status) + ": "
                + r.text()
            )
        # ⚠ A CUT CONNECTION ENDS A 200 EARLY and nothing in the status says
        # so. The listing's size is the only check that separates a complete
        # file from a truncated one, and this loop's whole resume story — skip
        # what is already the right size — depends on never renaming a short
        # file into place.
        if size >= 0:
            var got = file_size(out + ".part")
            if got != size:
                raise Error(
                    "lerobot: '" + rel + "' downloaded " + String(got)
                    + " bytes, the Hub listing says " + String(size)
                    + " — the transfer was truncated"
                )
        rename_over(out + ".part", String(out))
        n_files += 1

    if verbose:
        print(
            "  " + String(n_files) + " file(s) downloaded, " + String(n_skipped)
            + " already present -> " + dest
        )
    return dest^


def resolve_dataset_root(
    repo: String,
    revision: String = String("main"),
    download: Bool = True,
    verbose: Bool = True,
) raises -> String:
    """A local directory holding `repo`, downloading it only if necessary.

    Prefers, in order: this repo's own cache, then the `huggingface_hub`
    cache (so a dataset a Python tool already pulled is not fetched twice),
    then the network.
    """
    var mine = noeira_cache() + "/lerobot/" + repo_slug(repo)
    if exists(mine + "/meta/info.json"):
        if verbose:
            print("  using cached dataset " + mine)
        return mine^
    var hf = hf_snapshot_path(repo, revision)
    if hf != "" and exists(hf + "/meta/info.json"):
        if verbose:
            print("  using huggingface_hub cache " + hf)
        return hf^
    if not download:
        raise Error(
            "lerobot: '" + repo + "' is not in " + mine + " or the"
            " huggingface_hub cache, and downloading was disabled"
        )
    return hf_download_dataset(repo, mine^, revision, verbose)


def free_bytes(dir: String) -> Int:
    """Free bytes on the filesystem holding `dir`, or -1 if `df` cannot say.

    `df -Pk` is POSIX (Linux and macOS agree on its columns), and -1 lets a box
    with no `df` import anyway rather than refuse on a check it cannot run.
    """
    try:
        var out = run_capture(
            String("df -Pk ") + quote_arg(dir) + " 2>/dev/null | tail -1", 4096
        )
        var cols = List[String]()
        for c in out.split(" "):
            var t = String(c.strip())
            if t.byte_length() > 0:
                cols.append(t)
        if len(cols) < 4:
            return -1
        return atol(cols[3]) * 1024
    except:
        return -1
