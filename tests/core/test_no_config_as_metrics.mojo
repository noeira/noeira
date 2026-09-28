# +--------------------------------------------------------------------------+ #
# | A run's settings are config, not metrics
# +--------------------------------------------------------------------------+ #
"""A SOURCE gate over `noeira/`, `examples/` and `tools/`.

    pixi run mojo run -I . tests/core/test_no_config_as_metrics.mojo

## ⚠⚠ Why

`CsvLogger.set_config` used to be `pass`, so a run's CSV could not say what
produced it. Three drivers worked around that by logging their settings as
scalars at step 0 — `cfg/lanes`, `cfg/tau`, `cfg/seed`, per-task weights and
margins (the SAC family driver, BFM, HIL-SERL). The CSV became
self-describing, and the dashboard drew every one of them as a one-point
chart among the curves.

The config now has a file of its own (`metrics.config.kv`, written by
`CsvLogger` beside the CSV) and goes to the dashboard through `/runs`. THE
RULE: no code line names a `cfg/` metric. A setting goes through
`set_config`; a number that changes over training is a metric with a plain
name.

## ⚠ Scope

Code lines only: comments and docstring prose about the old blocks are
history, not a use. The scanned-file count is floored, because "0 offenders"
is also what scanning nothing reports.
"""

from noeira.io.fileio import read_file_bytes
from noeira.io.proc import run_capture


comptime MIN_FILES = 900
"""`noeira/` + `examples/` + `tools/` hold well over this many `.mojo` files;
a scan that reaches fewer did not run where it should."""


def _read(path: String) raises -> String:
    var bytes = read_file_bytes(path)
    bytes.append(0)
    return String(unsafe_from_utf8_ptr=bytes.unsafe_ptr())


def main() raises:
    print("=== a run's settings are config, not cfg/* metrics ===")
    var listing = run_capture(
        String("find noeira examples tools -name '*.mojo' -type f"),
        max_bytes=1 << 22,
    )
    var files = List[String]()
    for f in listing.split("\n"):
        var s = String(String(f).strip())
        if s.byte_length() > 0:
            files.append(s)

    var offenders = 0
    for path in files:
        var text = _read(path)
        var n = 0
        for line in text.split("\n"):
            n += 1
            var l = String(String(line).strip())
            if l.byte_length() == 0 or l.startswith("#"):
                continue
            if l.find('"cfg/') >= 0:
                offenders += 1
                print("    " + path + ":" + String(n) + " logs a cfg/ metric")

    print(
        "  " + String(len(files)) + " files scanned, " + String(offenders)
        + " cfg/ metric(s)"
    )
    if len(files) < MIN_FILES:
        raise Error(
            "scanned only " + String(len(files)) + " files — expected at least "
            + String(MIN_FILES) + "; run from the repo root"
        )
    if offenders != 0:
        raise Error(
            String(offenders) + " cfg/ metric(s): use `set_config` — the"
            " setting lands in metrics.config.kv and the dashboard's /runs"
        )
    print("[PASS] no config as metrics")
