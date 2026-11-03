# +--------------------------------------------------------------------------+ #
# | Import BONES-SEED G1 CSVs into SuperMimic's motion library
# +--------------------------------------------------------------------------+ #
"""SEED CSVs → filtered 50 Hz reference rows (`data/bones_seed_import.mojo`).

    pixi run mojo run -I . examples/g1/seed_import.mojo --csv-dir DIR --out seed_g1_50hz.h5
    pixi run mojo run -I . examples/g1/seed_import.mojo --list clips.txt --out seed_g1_50hz.h5

⚠ SEED IS CONFIDENTIAL. Its licence allows private compute only: run this on
the box or a private disk, never inside the public tree, and never commit
the store or its manifest. The public tree is gated on synthetic CSVs
(`tools/supermimic/make_synthetic_seed.py`).

ON THE BOX, without storing the 23.5 GB archive: pick the clip names (the
metadata CSV is enough for the name filter and the durations), then extract
only those while streaming:

    curl -sL <g1.tar.gz URL> | tar -xz -T wanted.txt -C seed_csv/
    find seed_csv -name '*.csv' | sort > clips.txt

The store is ~3x smaller than the CSV text it comes from (284 B a 50 Hz row
against ~40 kB/s of 120 Hz text), so if the CSVs fit, the store does too.

Options
-------
--csv-dir DIR     every `*.csv` directly in DIR (not recursive), sorted
--list FILE       one CSV path per line, in that order
--out PATH        the store (default `seed_g1_50hz.h5`)
--manifest PATH   the verdicts (default `<out>.manifest.csv`)
--max-clips N     stop after N inputs (0 = all)
--shard i/n       keep input k where k % n == i
--force           overwrite an existing store
"""

from std.builtin.sort import sort
from std.os import listdir
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns

from noeira.core.bytes import string_from_bytes
from noeira.data.bones_seed import SEED_OUT_FPS
from noeira.data.bones_seed_import import import_seed_clips
from noeira.io.fileio import file_size, read_file_bytes


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


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var csv_dir = _opt(args, String("--csv-dir"), String(""))
    var list_path = _opt(args, String("--list"), String(""))
    var out = _opt(args, String("--out"), String("seed_g1_50hz.h5"))
    var manifest = _opt(args, String("--manifest"), out + ".manifest.csv")
    var max_clips = Int(_opt(args, String("--max-clips"), String("0")))
    var shard = _opt(args, String("--shard"), String(""))
    if (csv_dir == "") == (list_path == ""):
        raise Error("give exactly one of --csv-dir DIR or --list FILE")
    if exists(out) and not _flag(args, String("--force")):
        print(out + " already exists; pass --force to rebuild")
        return

    var paths = List[String]()
    if csv_dir != "":
        var names = List[String]()
        for e in listdir(csv_dir):
            var n = String(e)
            if n.endswith(".csv"):
                names.append(n)
        sort(names)
        for i in range(len(names)):
            paths.append(csv_dir + "/" + names[i])
    else:
        for l in string_from_bytes(read_file_bytes(list_path)).split("\n"):
            var p = String(String(l).strip())
            if p.byte_length() > 0:
                paths.append(p)

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
    var chosen = List[String]()
    for k in range(len(paths)):
        if k % shard_n != shard_i:
            continue
        if max_clips > 0 and len(chosen) >= max_clips:
            break
        chosen.append(paths[k])
    if len(chosen) == 0:
        raise Error("no CSV selected")

    print("BONES-SEED (G1) -> SuperMimic library: " + String(len(chosen)) + " clips -> " + out)
    var t0 = perf_counter_ns()
    var s = import_seed_clips(chosen, out, manifest)
    var el = Float64(perf_counter_ns() - t0) * 1e-9
    print("")
    print(
        "  " + String(s.clips) + " clips: " + String(s.kept) + " kept, " + String(s.clips - s.kept - s.refused)
        + " filtered, " + String(s.refused) + " refused by the reader"
    )
    print(
        "  " + String(Float64(s.rows_kept) / SEED_OUT_FPS / 3600.0) + " h kept of "
        + String(Float64(s.rows_all) / SEED_OUT_FPS / 3600.0) + " h read"
    )
    print("  by reason:")
    for e in s.reasons.items():
        print("    " + (e.key if e.key.byte_length() > 0 else String("(kept)")) + ": " + String(e.value))
    print("  wrote " + out + " (" + String(file_size(out)) + " bytes) and " + manifest + " in " + String(el) + " s")
