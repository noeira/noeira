# +--------------------------------------------------------------------------+ #
# | hhtools trajectories for Berkeley Humanoid Lite → SuperMimic's library
# +--------------------------------------------------------------------------+ #
"""Retargeted clips (RoboParty's hhtools, robot `berkeley_humanoid_lite`) into a
SuperMimic library store for `BhlRobot`.

    pixi run mojo run -I . examples/bhl/retarget_import.mojo --csv-dir DIR --out bhl_50hz.h5

The CSVs are read by joint name, resampled to 50 Hz, and stored with
`examples/g1/seed_import.mojo`'s layout (`noeira/data/retarget_csv.mojo`).
hhtools' export height shift is taken back out unless `--keep-lift` is given.

⚠ Clips retargeted from BONES-SEED are confidential like SEED itself: private
storage only, never in a repository.

Options
-------
--csv-dir DIR    every `*.csv` under DIR (one level of sub-folders included), sorted
--list FILE      or the CSV paths FILE lists, one per line
--out PATH       the store (default bhl_50hz.h5); its manifest is PATH.manifest.csv
--max-clips N    stop after N files (0 = all)
--keep-lift      keep hhtools' `playback_mesh_z_lift` in the root height
--force          overwrite an existing store
"""

from std.builtin.sort import sort
from std.os import listdir
from std.os.path import exists, isdir
from std.sys import argv
from std.time import perf_counter_ns

from noeira.core.bytes import string_from_bytes
from noeira.data.retarget_csv import import_retarget_clips
from noeira.envs.robots.berkeley_humanoid_lite_pd import BHL_N_DOF, bhl_dof_name
from noeira.io.fileio import read_file_bytes


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _flag(args: List[String], name: String) -> Bool:
    for i in range(len(args)):
        if args[i] == name:
            return True
    return False


def _csvs(d: String) raises -> List[String]:
    var out = List[String]()
    var names = List[String]()
    for e in listdir(d):
        names.append(String(e))
    sort(names)
    for i in range(len(names)):
        var p = d + "/" + names[i]
        if names[i].endswith(".csv"):
            out.append(p)
        elif isdir(p):
            var inner = List[String]()
            for e in listdir(p):
                if String(e).endswith(".csv"):
                    inner.append(String(e))
            sort(inner)
            for k in range(len(inner)):
                out.append(p + "/" + inner[k])
    return out^


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var csv_dir = _opt(args, String("--csv-dir"), String(""))
    var list_path = _opt(args, String("--list"), String(""))
    var out = _opt(args, String("--out"), String("bhl_50hz.h5"))
    var max_clips = Int(_opt(args, String("--max-clips"), String("0")))
    if (csv_dir == "") == (list_path == ""):
        raise Error("give exactly one of --csv-dir DIR or --list FILE")
    if exists(out) and not _flag(args, String("--force")):
        print(out + " already exists; pass --force to rebuild")
        return

    var paths = List[String]()
    if csv_dir != "":
        paths = _csvs(csv_dir)
    else:
        for l in string_from_bytes(read_file_bytes(list_path)).split("\n"):
            var p = String(String(l).strip())
            if p.byte_length() > 0:
                paths.append(p)
    if max_clips > 0 and len(paths) > max_clips:
        paths.resize(max_clips, String(""))
    if len(paths) == 0:
        raise Error("no CSV selected")

    var joints = List[String]()
    for j in range(BHL_N_DOF):
        joints.append(String(bhl_dof_name(j)))
    var keep_lift = _flag(args, String("--keep-lift"))
    print("hhtools (Berkeley Humanoid Lite) -> SuperMimic library: " + String(len(paths)) + " clips -> " + out)
    var t0 = perf_counter_ns()
    var s = import_retarget_clips(
        paths, joints, out, out + ".manifest.csv", String("berkeley_humanoid_lite_retarget_50hz"),
        remove_lift=not keep_lift,
    )
    print(
        "  " + String(s.clips) + " clips: " + String(s.kept) + " kept, " + String(s.too_short) + " too short, "
        + String(s.refused) + " refused by the reader"
    )
    print("  " + String(Float64(s.rows_kept) / 50.0 / 60.0) + " min kept at 50 Hz")
    print(
        "  hhtools' height shift " + ("kept" if keep_lift else "removed") + ": between " + String(s.lift_min * 1000.0)
        + " and " + String(s.lift_max * 1000.0) + " mm over the kept clips"
    )
    print("  wrote " + out + " and its manifest in " + String(Float64(perf_counter_ns() - t0) * 1e-9) + " s")
