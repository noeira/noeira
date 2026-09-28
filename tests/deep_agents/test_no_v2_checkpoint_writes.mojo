# +--------------------------------------------------------------------------+ #
# | Nothing in the library writes a v2 checkpoint, or writes a file in place
# +--------------------------------------------------------------------------+ #
"""A SOURCE gate over `noeira/`.

    pixi run mojo run -I . tests/deep_agents/test_no_v2_checkpoint_writes.mojo

## ⚠⚠ Why this exists

v3 landed and twelve trainers kept writing v2 for months: the v2 writer was
still called `CheckpointWriter`, the name anyone reaches for, and eleven of
the twelve also wrote with a bare `open(path, "w")` — non-atomic, and cut off
at 2 GiB by a single `write(2)`. Nothing failed; the files were just 3× larger
and one crash away from corrupting the previous good checkpoint. So:

  1. `LegacyV2CheckpointWriter(` is constructed nowhere in `noeira/` outside
     `nn/core/checkpoint.mojo` (tests may still build v2 fixtures with it).
  2. No file under `noeira/deep_agents/` opens a file for writing directly:
     checkpoints, sidecars and data files all go through `io/fileio.mojo`'s
     atomic writers.

## ⚠ The scan is GLOBBED here, and that is deliberate

`test_checkpoints_announce.mojo` writes its file list down because it checks
that a rule is FOLLOWED at known sites. This gate checks that a pattern is
ABSENT everywhere, so a new file must be scanned without anyone remembering
to add it. The scanned-file count is printed and floored, because "0
violations" is also what scanning nothing reports.
"""

from noeira.io.fileio import read_file_bytes
from noeira.io.proc import run_capture


comptime MIN_FILES = 400
"""The tree has well over this many `.mojo` files under `noeira/`; a scan
that reaches fewer did not run where it should (wrong cwd, `find` failed)."""


def _read(path: String) raises -> String:
    var bytes = read_file_bytes(path)
    bytes.append(0)
    return String(unsafe_from_utf8_ptr=bytes.unsafe_ptr())


def _is_code(line: String) -> Bool:
    """Not a comment line. Docstring prose that MENTIONS a pattern is not a
    use of it, and the patterns below are specific enough (a call with its
    parenthesis) that prose rarely matches them anyway."""
    var s = String(line.strip())
    return s.byte_length() > 0 and not s.startswith("#")


def main() raises:
    print("=== no v2 checkpoint writes, no in-place file writes ===")
    var listing = run_capture(
        String("find noeira -name '*.mojo' -type f"), max_bytes=1 << 22
    )
    var files = List[String]()
    for f in listing.split("\n"):
        var s = String(f.strip())
        if s.byte_length() > 0:
            files.append(s)

    var v2 = 0
    var raw = 0
    var n_deep = 0
    for path in files:
        var text = _read(path)
        var is_deep = path.startswith("noeira/deep_agents/")
        if is_deep:
            n_deep += 1
        var n = 0
        for line in text.split("\n"):
            n += 1
            var l = String(line)
            if not _is_code(l):
                continue
            if (
                l.find("LegacyV2CheckpointWriter(") >= 0
                and path != "noeira/nn/core/checkpoint.mojo"
            ):
                v2 += 1
                print("    " + path + ":" + String(n) + " writes a v2 checkpoint")
            if is_deep and l.find("open(") >= 0 and l.find('"w"') >= 0:
                raw += 1
                print(
                    "    " + path + ":" + String(n)
                    + " opens a file for writing in place — use"
                    " write_file_atomic / write_text_atomic"
                )

    print(
        "  " + String(len(files)) + " files scanned (" + String(n_deep)
        + " under deep_agents), " + String(v2) + " v2 writes, "
        + String(raw) + " in-place writes"
    )
    if len(files) < MIN_FILES:
        raise Error(
            "scanned only " + String(len(files)) + " files — expected at least "
            + String(MIN_FILES) + "; run from the repo root"
        )
    if n_deep == 0:
        raise Error("no file under noeira/deep_agents/ was scanned")
    if v2 != 0 or raw != 0:
        raise Error(
            String(v2 + raw) + " violation(s) — see the lines above"
        )
    print("[PASS] no-v2-checkpoint-writes")
