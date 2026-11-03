# +--------------------------------------------------------------------------+ #
# | BONES-SEED CSVs → SuperMimic's motion library, as a TrajectoryStore (M1)
# +--------------------------------------------------------------------------+ #
"""SEED CSVs → filtered 50 Hz reference rows in one `TrajectoryStore`, plus a
verdict manifest. This is the reproduction's `convert_csv_to_npz` +
`filter_bones_seed.py` + `build_library.py` in one pass, with no npz in
between.

THE STORE
=========
    ref        float32 (N_rows, 71)  the compact library row, per kept clip
    motion_id  int32   (N_rows, 1)   the clip's episode index
    fps        float64 (1,)          50
and one episode per kept clip, its name in the manifest's task table
(`task_text(motion_id)`). This is the reproduction's library
(`{"ref", "names", "starts", "lengths"}`): `starts`/`lengths` are the
episode index.

⚠ FLOAT32, NOT THE REPRODUCTION'S FLOAT16. `build_library.py` stores the
rows as fp16, and fp16 spacing is 1/1024 at 1 and 1/128 at 10: a pelvis
that walks 10 m from the origin is stored to the nearest 7.8 mm, and its
height to 0.5 mm. We keep float32 (284 B a row, 1.0 GB for 20 h); a parity
run against the reproduction rounds to fp16 at load, not here.

THE MANIFEST is `filter_bones_seed.py`'s `manifest_filtered.csv`, same
header (`name,frames,keep,reason`), one row per input clip, so the two
can be diffed directly. `frames` is the 50 Hz row count; a clip our reader
refuses is listed with `reason=refused` (the reproduction would have
failed earlier, at conversion).

Memory peaks at one clip; the store is streamed.
"""

from noeira.data.bones_seed import (
    MimicMotion, SEED_OUT_FPS, SEED_REF_DIM, load_seed_csv, resample_mjlab, seed_name_filter,
)
from noeira.data.bones_seed_filter import seed_feasibility
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStoreWriter
from noeira.envs.robots import UnitreeG1
from noeira.io.fileio import remove_file, rename_over, write_text_atomic
from std.os.path import exists


comptime SEED_ENV_ID = "unitree_g1_bones_seed_50hz"
comptime SEED_REF_COLUMN = "ref"
comptime SEED_MOTION_ID_COLUMN = "motion_id"


struct SeedImportSummary(Movable):
    var clips: Int
    var kept: Int
    var refused: Int
    var rows_kept: Int
    var rows_all: Int
    var reasons: Dict[String, Int]

    def __init__(out self):
        self.clips = 0
        self.kept = 0
        self.refused = 0
        self.rows_kept = 0
        self.rows_all = 0
        self.reasons = Dict[String, Int]()

    def __init__(out self, *, deinit move: Self):
        self.clips = move.clips
        self.kept = move.kept
        self.refused = move.refused
        self.rows_kept = move.rows_kept
        self.rows_all = move.rows_all
        self.reasons = move.reasons^


def seed_clip_name(path: String) -> String:
    """The clip name the reproduction keys everything on: the file's stem."""
    var slash = path.rfind("/")
    var base = String(path[byte = slash + 1 :]) if slash >= 0 else path.copy()
    if base.endswith(".csv"):
        return String(base[byte = 0 : base.byte_length() - 4])
    return base^


def _columns() -> List[ColumnSpec]:
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String(SEED_REF_COLUMN), DType.float32, SEED_REF_DIM))
    cols.append(ColumnSpec(String(SEED_MOTION_ID_COLUMN), DType.int32, 1))
    return cols^


def import_seed_clips(
    paths: List[String], out_path: String, manifest_path: String, verbose: Bool = True
) raises -> SeedImportSummary:
    """Every CSV of `paths`, in that order: read, resample to 50 Hz, the name
    filter, then the feasibility filter on our G1; kept clips are appended to
    the store at `out_path` (written to `<out_path>.part` and renamed at the
    end, so an interrupted import leaves no plausible-looking store)."""
    var s = SeedImportSummary()
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    var part = out_path + ".part"
    if exists(part):
        remove_file(part)
    var w = TrajectoryStoreWriter(part, _columns(), String(SEED_ENV_ID), 0, String(""))
    var man = String("name,frames,keep,reason\n")
    for ci in range(len(paths)):
        var name = seed_clip_name(paths[ci])
        s.clips += 1
        var reason: String
        var motion: MimicMotion
        try:
            motion = resample_mjlab(load_seed_csv(paths[ci], name.copy()))
        except e:
            if verbose:
                print("  [refused] " + name + ": " + String(e))
            s.refused += 1
            s.reasons["refused"] = s.reasons.get("refused", 0) + 1
            man += name + ",0,0,refused\n"
            continue
        var frames = motion.n
        s.rows_all += frames
        reason = seed_name_filter(name)
        if reason.byte_length() == 0:
            reason = seed_feasibility(motion, env)
        if reason.byte_length() == 0:
            var rows = List[Float32](length=frames * SEED_REF_DIM, fill=Float32(0))
            for i in range(frames * SEED_REF_DIM):
                rows[i] = Float32(motion.frames[i])
            var ids = List[Int32](length=frames, fill=Int32(s.kept))
            w.append[DType.float32](
                String(SEED_REF_COLUMN),
                rows.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]().as_unsafe_any_origin(),
                frames,
            )
            w.append[DType.int32](
                String(SEED_MOTION_ID_COLUMN),
                ids.unsafe_ptr().unsafe_bitcast[Scalar[DType.int32]]().as_unsafe_any_origin(),
                frames,
            )
            w.end_episode()
            w.add_task(s.kept, name.copy())
            s.kept += 1
            s.rows_kept += frames
        s.reasons[reason] = s.reasons.get(reason, 0) + 1
        man += name + "," + String(frames) + "," + ("1" if reason.byte_length() == 0 else "0") + "," + reason + "\n"
        if verbose and ((ci + 1) % 500 == 0 or ci == len(paths) - 1):
            print(
                "  [" + String(ci + 1) + "/" + String(len(paths)) + "] kept " + String(s.kept)
                + ", " + String(Float64(s.rows_kept) / SEED_OUT_FPS / 3600.0) + " h"
            )
    if s.kept == 0:
        raise Error("bones_seed_import: no clip survived the filters; no store written")
    var fps = List[Float64](length=1, fill=SEED_OUT_FPS)
    w.write_vector[DType.float64](
        String("fps"), fps.unsafe_ptr().unsafe_bitcast[Scalar[DType.float64]]().as_unsafe_any_origin(), 1
    )
    w.close()
    rename_over(part, out_path.copy())
    write_text_atomic(manifest_path.copy(), man)
    return s^
