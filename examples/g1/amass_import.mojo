# +--------------------------------------------------------------------------+ #
# | Import the retargeted AMASS dump for the G1 into a TrajectoryStore
# +--------------------------------------------------------------------------+ #
"""`fleaven/Retargeted_AMASS_for_robotics` → the 50 Hz store BFM-Zero reads
— §10.7 of `docs/BFM_ZERO_NEXT_LEVEL.md`.

    # what a slice would cost, downloading nothing:
    pixi run mojo run -I . examples/g1/amass_import.mojo --plan
    pixi run mojo run -I . examples/g1/amass_import.mojo --plan --subset ACCAD

    # build one:
    pixi run mojo run -I . examples/g1/amass_import.mojo \
        --subset ACCAD --out amass_accad_50hz.h5

    # the whole dump, in eight pieces:
    pixi run mojo run -I . examples/g1/amass_import.mojo --shard 0/8 --out amass_50hz.h5

This is `lafan_import.mojo`'s sibling: the same store schema, the same
conversion (`data/lafan.mojo`'s `convert_frames`, which the oracle gate
holds against the reference dump), a different step 1 (`data/amass.mojo`).

THE SIZE, WHICH IS THE WHOLE PROBLEM. 17 717 clips, **59.4 hours**, 6.95 GB
of `.npy`. At 50 Hz that is **10.70 M rows**, and a row of LAFAN's full
schema is 4008 bytes — so the full store is **42.9 GB**, which does not fit
on a laptop with 17 GiB free. Three things follow, and all three are
options rather than assumptions:

  * **`--plan` prints the projection and downloads nothing.** Clip duration
    comes from the listed file size and the fps in the name
    (`amass_frames_of_size`), so the whole budget is known before the first
    byte moves.
  * **The default schema is LEAN.** `qpos` 36 + `qvel` 35 + `state` 64 +
    `privileged` 463 + `motion_id` — 2396 bytes a row, against 4008. The
    four `body_*` columns are 403 floats, **40 % of the row**, and nothing
    on the training or probe path reads them: `bfm_zero_train_gpu`,
    `bfm_zero_bank_build`, `bfm_zero_spec_probe`, `bfm_zero_cmd_probe`,
    `bfm_zero_pose_cmd_probe` and `bfm_zero_reward_joystick` load `state`,
    `privileged`, `qpos` and `qvel` and nothing else (grepped, not assumed).
    `--full` writes LAFAN's schema; the oracle gate needs it, a trainer
    does not. Lean takes the full store to 25.6 GB and roughly doubles what
    a laptop holds.
  * **The source is streamed and deleted** unless `--keep-source`, so the
    peak source footprint is one clip (2.9 MB at most) rather than 6.95 GB.

⚠ **THE IMPORT IS DOWNLOAD-BOUND, NOT COMPUTE-BOUND.** Measured on ACCAD:
230 clips, 72 682 rows, **240 s wall at 4 % CPU** — 9.8 s of CPU and the
rest in 230 serial HTTP round trips, about 1.04 s each. The whole dump is
therefore ~25 min of conversion behind **~5 h of serial requests**, and no
faster machine changes that. `--source-dir DIR` is the way out: bulk-fetch
the repo once with a parallel downloader (`hf download fleaven/
Retargeted_AMASS_for_robotics --repo-type dataset --local-dir DIR`) and the
importer becomes pure compute. It needs 6.95 GB for the copy.

SELECTORS, all composable, applied in this order so that `--plan` and the
build always choose the same clips:

    --subset A,B     only these sub-datasets (25 of them; `--plan` lists all)
    --min-seconds S  drop clips shorter than S        (default 1.0)
    --max-seconds S  drop clips longer than S         (default 0 = no cap)
    --hours H        stop once H hours have been selected (0 = no cap)
    --max-clips N    stop after N clips               (0 = no cap)
    --shard i/n      keep clip k where k % n == i, AFTER the filters above

⚠ **`--hours` AND `--max-clips` TRUNCATE A SORTED LIST, THEY DO NOT
SAMPLE.** The list is sorted by path, so `--hours 2` is the first two hours
alphabetically — ACCAD and part of BMLhandball, not two representative
hours. For a representative slice use `--shard`, which strides the whole
dump: `--shard 0/30` is one thirtieth of EVERY sub-dataset.

⚠ **THE SUB-DATASET LICENCES TRAVEL WITH THE DATA.** CC-BY-4.0 covers the
retargeting, not AMASS's 25 sources. Every `license.txt` and `citation.bib`
of every selected sub-dataset is downloaded next to the store, into
`<out>.licenses/`, and the run refuses to finish if one is missing.

⚠ **A COUNT IS NOT COVERAGE.** The run prints rows AND hours AND the clip
count per sub-dataset at the end, because an import that silently dropped a
sub-dataset still writes a plausible number of rows.

Options
-------
--out PATH         output .h5 (default `amass_g1_50hz.h5`); `--shard i/n`
                   writes `PATH` with `.shard<i>of<n>` before the suffix
--plan             print the selection and the projected sizes; download
                   nothing, write nothing
--full             LAFAN's schema (adds body_pos/quat/vel/ang_vel)
--keep-source      keep the downloaded `.npy` files in the HF cache
--index PATH       a cached blobs listing; written on the first run
--source-dir DIR   look for each clip at `DIR/<repo path>` and use it when
                   its size matches the index; only the misses are
                   downloaded, and a `--source-dir` file is never deleted
--revision REV     dataset revision (default `main`)
--force            rebuild even if the output exists
--no-download      fail rather than fetch anything
--min-free GB      refuse to start below this much free disk after the
                   projection (default 2.0)
"""

from std.os import makedirs
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns

from noeira.core.bytes import string_from_bytes
from noeira.data.amass import (
    AMASS_COLS, AMASS_MIN_SECONDS, AMASS_REPO, AmassClip, amass_frames_of_size,
    amass_fps_of_name, amass_is_clip, amass_subset_of, convert_amass_clip,
    load_amass_clip,
)
from noeira.data.column import ColumnSpec
from noeira.data.lafan import (
    LafanRows, LAFAN_NQ, LAFAN_NV, LAFAN_STATE_DIM, LAFAN_PRIV_DIM,
    LAFAN_N_BODIES, LAFAN_ENV_DT,
)
from noeira.data.store import TrajectoryStoreWriter
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from noeira.io.fileio import (
    file_size, read_file_bytes, remove_file, rename_over, write_text_atomic,
)
from noeira.io.hf import (
    HF_DATASET, hf_download_sized, hf_repo_blobs,
)
from noeira.io.json import J_ARRAY, parse_json
from noeira.io.proc import disk_free_bytes


comptime ENV_ID = "unitree_g1_amass_50hz"
comptime LEAN_ROW_BYTES: Int = (
    (LAFAN_NQ + LAFAN_NV + LAFAN_STATE_DIM + LAFAN_PRIV_DIM) * 4 + 4
)
comptime FULL_ROW_BYTES: Int = LEAN_ROW_BYTES + LAFAN_N_BODIES * 13 * 4


# ── the index ───────────────────────────────────────────────────────────────
struct Entry(Copyable, Movable):
    var path: String
    var subset: String
    var size: Int
    var fps: Int
    var frames: Int

    def __init__(out self, var path: String, var subset: String, size: Int, fps: Int, frames: Int):
        self.path = path^
        self.subset = subset^
        self.size = size
        self.fps = fps
        self.frames = frames

    def seconds(self) -> Float64:
        return Float64(self.frames) / Float64(self.fps)

    def rows(self) -> Int:
        """`ceil(length / 0.02)` — the resampler's own count.

        ⚠ `length` IS ROUNDED TO FLOAT32 FIRST, because `convert_frames`
        does: it computes `length32 = Float32(length64)` and takes the ceil of
        that. Without the cast the projection was 72 677 rows where the build
        wrote 72 682 — five clips whose float32 length rounds up over a
        multiple of 0.02. Five rows in 230 clips is nothing for a disk
        budget and everything for a number that is supposed to be the same
        number.
        """
        var length = Float64(Float32(Float64(self.frames - 1) / Float64(self.fps)))
        var q = length / LAFAN_ENV_DT
        var k = Int(q)
        if Float64(k) < q:
            k += 1
        return k


def _opt(args: List[String], name: String, fallback: String) raises -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _flag(args: List[String], name: String) -> Bool:
    for i in range(len(args)):
        if args[i] == name:
            return True
    return False


def _pad(s: String, w: Int) -> String:
    var o = s
    while o.byte_length() < w:
        o += String(" ")
    return o^


def _lpad(s: String, w: Int) -> String:
    var o = String("")
    var k = w - s.byte_length()
    for _ in range(k):
        o += String(" ")
    return o + s


def _gb(b: Int) -> String:
    return String(Float64(b) / 1e9)[byte=0:5] + " GB"


def _hm(secs: Float64) -> String:
    var h = Int(secs / 3600.0)
    var m = Int((secs - Float64(h) * 3600.0) / 60.0)
    return String(h) + " h " + String(m) + " min"


def load_index(
    index_path: String, revision: String, download: Bool
) raises -> String:
    """The blobs listing, from `index_path` or the Hub (then cached there)."""
    if exists(index_path) and file_size(index_path) > 1024:
        print("  index: " + index_path + " (cached, " + String(file_size(index_path)) + " bytes)")
        return string_from_bytes(read_file_bytes(index_path))
    if not download:
        raise Error("--no-download and no cached index at " + index_path)
    print("  index: fetching the blobs listing for " + String(AMASS_REPO))
    var j = hf_repo_blobs(String(AMASS_REPO), HF_DATASET, revision)
    write_text_atomic(index_path, j)
    print("  index: " + String(j.byte_length()) + " bytes -> " + index_path)
    return j^


def parse_index(j: String) raises -> List[Entry]:
    """`siblings` → one `Entry` per clip, sorted by path.

    ⚠ SORTED, AND BY PATH, because `--shard` and `--hours` both cut a list
    and a listing whose order the Hub chose would make a shard a different
    set of clips on a different day.
    """
    var jb = List[UInt8]()
    var src = j.as_bytes()
    for i in range(j.byte_length()):
        jb.append(src[i])
    var doc = parse_json(jb^)
    var root = doc.root()
    var sib = doc.field(root, String("siblings"))
    if sib < 0 or doc.kind_of(sib) != J_ARRAY:
        raise Error(
            "amass: the blobs listing has no `siblings` array — is the"
            " cached index a truncated download?"
        )
    var out = List[Entry]()
    var missing_size = 0
    for i in range(doc.size(sib)):
        var e = doc.at(sib, i)
        var p = doc.string(doc.field(e, String("rfilename")))
        if not amass_is_clip(p):
            continue
        var sn = doc.field(e, String("size"))
        if sn < 0:
            missing_size += 1
            continue
        var size = doc.integer(sn)
        var fps = amass_fps_of_name(p)
        out.append(Entry(p.copy(), amass_subset_of(p), size, fps, amass_frames_of_size(size)))
    if missing_size > 0:
        raise Error(
            "amass: " + String(missing_size) + " clips in the listing carry no"
            " size — the planner cannot budget without one (was the index"
            " fetched without `?blobs=true`?)"
        )
    # insertion sort over paths: the listing is near-sorted already
    for i in range(1, len(out)):
        var k = out[i].copy()
        var j2 = i - 1
        while j2 >= 0 and out[j2].path > k.path:
            out[j2 + 1] = out[j2].copy()
            j2 -= 1
        out[j2 + 1] = k^
    return out^


def select(
    all: List[Entry], subsets: List[String], min_s: Float64, max_s: Float64,
    hours: Float64, max_clips: Int, shard_i: Int, shard_n: Int,
) raises -> List[Entry]:
    var kept = List[Entry]()
    var secs = 0.0
    var n_short = 0
    var n_long = 0
    var k = 0
    for i in range(len(all)):
        var e = all[i].copy()
        if len(subsets) > 0:
            var hit = False
            for s in range(len(subsets)):
                if subsets[s] == e.subset:
                    hit = True
                    break
            if not hit:
                continue
        var d = e.seconds()
        if d < min_s:
            n_short += 1
            continue
        if max_s > 0.0 and d > max_s:
            n_long += 1
            continue
        if shard_n > 1:
            var mine = k % shard_n == shard_i
            k += 1
            if not mine:
                continue
        else:
            k += 1
        if hours > 0.0 and (secs + d) / 3600.0 > hours:
            break
        if max_clips > 0 and len(kept) >= max_clips:
            break
        secs += d
        kept.append(e.copy())
    if n_short > 0 or n_long > 0:
        print(
            "  dropped " + String(n_short) + " clips under " + String(min_s)
            + " s" + (", " + String(n_long) + " over " + String(max_s) + " s" if max_s > 0.0 else "")
        )
    return kept^


def report(kept: List[Entry], row_bytes: Int, label: String) raises -> Int:
    """Rows, hours and bytes, per sub-dataset and in total. Returns the bytes."""
    var names = List[String]()
    var clips = List[Int]()
    var rows = List[Int]()
    var secs = List[Float64]()
    var src = List[Int]()
    for i in range(len(kept)):
        var e = kept[i].copy()
        var at = -1
        for j in range(len(names)):
            if names[j] == e.subset:
                at = j
                break
        if at < 0:
            names.append(e.subset.copy())
            clips.append(0)
            rows.append(0)
            secs.append(0.0)
            src.append(0)
            at = len(names) - 1
        clips[at] += 1
        rows[at] += e.rows()
        secs[at] += e.seconds()
        src[at] += e.size
    var t_clips = 0
    var t_rows = 0
    var t_secs = 0.0
    var t_src = 0
    print("")
    print(label)
    print(
        "  " + _pad(String("sub-dataset"), 26) + _lpad(String("clips"), 7)
        + _lpad(String("hours"), 9) + _lpad(String("rows"), 12)
        + _lpad(String("src MB"), 10) + _lpad(String("store"), 11)
    )
    for i in range(len(names)):
        print(
            "  " + _pad(names[i], 26) + _lpad(String(clips[i]), 7)
            + _lpad(String(String(secs[i] / 3600.0)[byte=0:5]), 9)
            + _lpad(String(rows[i]), 12)
            + _lpad(String(String(Float64(src[i]) / 1e6)[byte=0:7]), 10)
            + _lpad(_gb(rows[i] * row_bytes), 11)
        )
        t_clips += clips[i]
        t_rows += rows[i]
        t_secs += secs[i]
        t_src += src[i]
    print(
        "  " + _pad(String("TOTAL"), 26) + _lpad(String(t_clips), 7)
        + _lpad(String(String(t_secs / 3600.0)[byte=0:5]), 9)
        + _lpad(String(t_rows), 12)
        + _lpad(String(String(Float64(t_src) / 1e6)[byte=0:7]), 10)
        + _lpad(_gb(t_rows * row_bytes), 11)
    )
    print("  " + _hm(t_secs) + " of motion, " + String(row_bytes) + " bytes a row")
    return t_rows * row_bytes


def _columns(full: Bool) -> List[ColumnSpec]:
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, LAFAN_NQ))
    cols.append(ColumnSpec(String("qvel"), DType.float32, LAFAN_NV))
    cols.append(ColumnSpec(String("state"), DType.float32, LAFAN_STATE_DIM))
    cols.append(ColumnSpec(String("privileged"), DType.float32, LAFAN_PRIV_DIM))
    if full:
        cols.append(ColumnSpec(String("body_pos"), DType.float32, LAFAN_N_BODIES * 3))
        cols.append(ColumnSpec(String("body_quat"), DType.float32, LAFAN_N_BODIES * 4))
        cols.append(ColumnSpec(String("body_vel"), DType.float32, LAFAN_N_BODIES * 3))
        cols.append(ColumnSpec(String("body_ang_vel"), DType.float32, LAFAN_N_BODIES * 3))
    cols.append(ColumnSpec(String("motion_id"), DType.int32, 1))
    return cols^


@always_inline
def _fptr(mut lst: List[Float32]) -> Pointer[Scalar[DType.float32], MutAnyOrigin]:
    return lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]().as_unsafe_any_origin()


@always_inline
def _iptr(mut lst: List[Int32]) -> Pointer[Scalar[DType.int32], MutAnyOrigin]:
    return lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.int32]]().as_unsafe_any_origin()


@always_inline
def _dptr(mut lst: List[Float64]) -> Pointer[Scalar[DType.float64], MutAnyOrigin]:
    return lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.float64]]().as_unsafe_any_origin()


def _write_clip(
    mut w: TrajectoryStoreWriter, mut rows: LafanRows, motion_id: Int, full: Bool
) raises:
    var n = rows.n_rows
    w.append[DType.float32](String("qpos"), _fptr(rows.qpos), n)
    w.append[DType.float32](String("qvel"), _fptr(rows.qvel), n)
    w.append[DType.float32](String("state"), _fptr(rows.state), n)
    w.append[DType.float32](String("privileged"), _fptr(rows.privileged), n)
    if full:
        w.append[DType.float32](String("body_pos"), _fptr(rows.body_pos), n)
        w.append[DType.float32](String("body_quat"), _fptr(rows.body_quat), n)
        w.append[DType.float32](String("body_vel"), _fptr(rows.body_vel), n)
        w.append[DType.float32](String("body_ang_vel"), _fptr(rows.body_ang_vel), n)
    var ids = List[Int32](length=n, fill=Int32(motion_id))
    w.append[DType.int32](String("motion_id"), _iptr(ids), n)
    w.end_episode()


def fetch_licenses(
    kept: List[Entry], out_dir: String, index: String, revision: String
) raises:
    """Every selected sub-dataset's `license.txt` and `citation.bib`.

    ⚠ NOT OPTIONAL, AND NOT BEST-EFFORT. The retargeting is CC-BY-4.0; the
    25 AMASS sources are not, and each states its own terms beside its
    clips. A store without them is a store nobody can redistribute, and the
    omission is invisible.
    """
    var subs = List[String]()
    for i in range(len(kept)):
        var hit = False
        for j in range(len(subs)):
            if subs[j] == kept[i].subset:
                hit = True
                break
        if not hit:
            subs.append(kept[i].subset.copy())
    makedirs(out_dir, exist_ok=True)
    var jb = List[UInt8]()
    var src = index.as_bytes()
    for i in range(index.byte_length()):
        jb.append(src[i])
    var doc = parse_json(jb^)
    var sib = doc.field(doc.root(), String("siblings"))
    var want = List[String]()
    var sizes = List[Int]()
    for i in range(doc.size(sib)):
        var e = doc.at(sib, i)
        var p = doc.string(doc.field(e, String("rfilename")))
        if not (p.endswith("license.txt") or p.endswith("citation.bib")
                or p.endswith("amass.bib")):
            continue
        var s = amass_subset_of(p)
        var mine = s == "" or p.count("/") <= 1
        for j in range(len(subs)):
            if subs[j] == s:
                mine = True
                break
        if not mine:
            continue
        var sn = doc.field(e, String("size"))
        want.append(p.copy())
        sizes.append(doc.integer(sn) if sn >= 0 else -1)
    if len(want) == 0:
        raise Error(
            "amass: no licence file was found in the index for "
            + String(len(subs)) + " sub-datasets — refusing to write a store"
            " whose terms did not travel with it"
        )
    print("  licences: " + String(len(want)) + " files -> " + out_dir)

    # ⚠ WHICH SUB-DATASETS HAVE NO LICENCE TEXT OF THEIR OWN. Measured on
    # the dump: it holds 49 licence files in all, and three sub-datasets are
    # missing half their pair — `MOYO_smplh_gendered` has only a
    # `citation.bib`, `SSM_synced` and `Transitions_mocap` only a
    # `license.txt`. A missing `citation.bib` costs an attribution; a missing
    # `license.txt` means that sub-dataset's TERMS are not in this store at
    # all, and the two must not report the same way. Named, not counted: a
    # tally of 49 looks complete.
    var no_text = List[String]()
    for j in range(len(subs)):
        var found = False
        for i in range(len(want)):
            if (amass_subset_of(want[i]) == subs[j]
                    and want[i].endswith("license.txt")):
                found = True
                break
        if not found:
            no_text.append(subs[j].copy())
    if len(no_text) > 0:
        var names = String("")
        for j in range(len(no_text)):
            names += (String(", ") if j > 0 else String("")) + no_text[j]
        print(
            "  ⚠ no `license.txt` in the dump for: " + names
            + " — their terms are NOT in this store; check the AMASS source"
            " before redistributing it"
        )
    for i in range(len(want)):
        var dest = out_dir + "/" + String(want[i].replace("/", "__"))
        _ = hf_download_sized(
            String(AMASS_REPO), want[i], sizes[i], HF_DATASET, dest, revision
        )


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))

    var out = _opt(args, String("--out"), String("amass_g1_50hz.h5"))
    var revision = _opt(args, String("--revision"), String("main"))
    var index_path = _opt(args, String("--index"), String("amass_index.json"))
    var source_dir = _opt(args, String("--source-dir"), String(""))
    var plan_only = _flag(args, String("--plan"))
    var full = _flag(args, String("--full"))
    var keep_source = _flag(args, String("--keep-source"))
    var force = _flag(args, String("--force"))
    var download = not _flag(args, String("--no-download"))
    var min_s = Float64(_opt(args, String("--min-seconds"), String(AMASS_MIN_SECONDS)))
    var max_s = Float64(_opt(args, String("--max-seconds"), String("0")))
    var hours = Float64(_opt(args, String("--hours"), String("0")))
    var max_clips = Int(_opt(args, String("--max-clips"), String("0")))
    var min_free = Float64(_opt(args, String("--min-free"), String("2.0")))
    var shard = _opt(args, String("--shard"), String(""))
    var subset_arg = _opt(args, String("--subset"), String(""))

    var shard_i = 0
    var shard_n = 1
    if shard != "":
        var sp = shard.split("/")
        if len(sp) != 2:
            raise Error("--shard takes i/n, e.g. --shard 0/8")
        shard_i = Int(String(sp[0]))
        shard_n = Int(String(sp[1]))
        if shard_n < 1 or shard_i < 0 or shard_i >= shard_n:
            raise Error("--shard " + shard + " is out of range")
        var dot = out.rfind(".")
        var stem = String(out[byte=0:dot]) if dot > 0 else out.copy()
        var ext = String(out[byte=dot:]) if dot > 0 else String("")
        out = stem + ".shard" + String(shard_i) + "of" + String(shard_n) + ext

    var subsets = List[String]()
    if subset_arg != "":
        for t in subset_arg.split(","):
            if String(t) != "":
                subsets.append(String(t))

    var row_bytes = FULL_ROW_BYTES if full else LEAN_ROW_BYTES
    print("Retargeted AMASS (G1 29-DoF) -> TrajectoryStore")
    print("  repo: " + String(AMASS_REPO) + " @ " + revision)
    print("  schema: " + ("full (LAFAN's columns)" if full else "lean (no body_*)")
          + ", " + String(row_bytes) + " bytes a row")
    if not plan_only:
        print("  out: " + out)
        if exists(out) and not force:
            print("  already present — pass --force to rebuild")
            return

    var index = load_index(index_path, revision, download)
    var all = parse_index(index)
    print("  " + String(len(all)) + " clips in the index")
    _ = report(all, row_bytes, String("THE WHOLE DUMP:"))

    var kept = select(all, subsets, min_s, max_s, hours, max_clips, shard_i, shard_n)
    if len(kept) == 0:
        raise Error("the selectors kept no clips — nothing to import")
    var store_bytes = report(kept, row_bytes, String("SELECTED:"))

    # ── the disk, before anything is downloaded ──────────────────────────
    var src_bytes = 0
    for i in range(len(kept)):
        src_bytes += kept[i].size
    var peak_src = src_bytes if keep_source else 0
    var need = store_bytes + peak_src
    var free = disk_free_bytes(String("."))
    print("")
    print("  projected store " + _gb(store_bytes)
          + (" + source " + _gb(src_bytes) + " kept" if keep_source else " (source streamed and deleted)"))
    if free < 0:
        print("  ⚠ the free space on this filesystem could not be read")
    else:
        print("  free now " + _gb(free) + ", left after " + _gb(free - need))
        if Float64(free - need) < min_free * 1e9:
            print("")
            print("  REFUSING TO START: this would leave "
                  + _gb(free - need) + ", under the " + String(min_free)
                  + " GB floor.")
            print("  Narrow it (--subset / --shard i/n / --hours), or raise"
                  " --min-free if you mean it.")
            if not plan_only:
                raise Error("amass: not enough disk for the selected slice")
    if plan_only:
        print("")
        print("  --plan: nothing downloaded, nothing written")
        return
    if not download:
        raise Error("--no-download: the clips are not cached")

    # ── convert ──────────────────────────────────────────────────────────
    var t0 = perf_counter_ns()
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var part = out + ".part"
    if exists(part):
        remove_file(part)
    var w = TrajectoryStoreWriter(part, _columns(full), String(ENV_ID), 0, String(""))
    var total_rows = 0
    var written = 0
    var skipped = 0
    var from_dir = 0
    for ci in range(len(kept)):
        var e = kept[ci].copy()
        # ⚠ A `--source-dir` HIT IS CHECKED BY SIZE, and never deleted: the
        # directory is the user's, not a cache this run owns.
        var local = String("")
        var local_is_ours = True
        if source_dir != "":
            var cand = source_dir + "/" + e.path
            if exists(cand) and file_size(cand) == e.size:
                local = cand
                local_is_ours = False
                from_dir += 1
        if local == "":
            local = hf_download_sized(
                String(AMASS_REPO), e.path, e.size, HF_DATASET, String(""), revision
            )
        var clip: AmassClip
        var rows: LafanRows
        try:
            clip = load_amass_clip(local, e.path.copy())
            rows = convert_amass_clip(clip, env)
        except err:
            # A clip the pipeline refuses is named and skipped, never counted.
            print("  [skip] " + e.path + ": " + String(err))
            skipped += 1
            if not keep_source and local_is_ours:
                remove_file(local)
            continue
        _write_clip(w, rows, written, full)
        w.add_task(written, e.path)
        total_rows += rows.n_rows
        written += 1
        if not keep_source and local_is_ours:
            remove_file(local)
        if written % 50 == 0 or ci == len(kept) - 1:
            var el = Float64(perf_counter_ns() - t0) * 1e-9
            print(
                "  [" + String(ci + 1) + "/" + String(len(kept)) + "] "
                + String(total_rows) + " rows, " + String(el)[byte=0:6]
                + " s, " + String(Float64(total_rows) / el)[byte=0:7]
                + " rows/s  (" + e.subset + ")"
            )

    var default = List[Float32](length=G1_N_DOF, fill=Float32(0))
    for j in range(G1_N_DOF):
        default[j] = Float32(g1_default_pos(j))
    w.write_vector[DType.float32](String("default_dof_pos"), _fptr(default), G1_N_DOF)
    var dt = List[Float64](length=1, fill=LAFAN_ENV_DT)
    w.write_vector[DType.float64](String("env_dt"), _dptr(dt), 1)
    w.close()
    rename_over(part, out.copy())

    fetch_licenses(kept, out + ".licenses", index, revision)

    var el = Float64(perf_counter_ns() - t0) * 1e-9
    print("")
    print(
        "wrote " + out + ": " + String(total_rows) + " rows, " + String(written)
        + " episodes, " + String(skipped) + " skipped, "
        + _gb(file_size(out)) + " in " + String(el)[byte=0:7] + " s"
    )
    if source_dir != "":
        print(
            "  " + String(from_dir) + " of " + String(len(kept))
            + " clips came from --source-dir, " + String(len(kept) - from_dir)
            + " were downloaded"
        )
    print("  " + String(Float64(total_rows) / 50.0 / 3600.0)[byte=0:5]
          + " h of motion at 50 Hz")
