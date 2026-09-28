# +--------------------------------------------------------------------------+ #
# | A metric name is flat snake_case, never a `/` namespace
# +--------------------------------------------------------------------------+ #
"""A SOURCE gate over `noeira/`, `examples/` and `tools/`.

    pixi run mojo run -I . tests/core/test_metric_names_are_flat.mojo

## ⚠⚠ Why

The dashboard (`noeira-cloud/client/src/lib/metric-groups.ts`) groups charts
by EXACT name. By 2026-09-22 the older bundles emitted `policy_loss`-style
names it understood, and every newer driver had drifted to TensorBoard /
wandb namespaces: `fb/measure`, `cpr/d_pos`, `loss/actor`, `train/l1`,
`eval/mean_return`, `online/imag_rew_mean`. None of them matched a group, so
each became its own ungrouped chart, and the same quantity carried three
names across three drivers (FB-CPR's `cpr/r_mean` was BFM's `cpr/r_d`).

THE RULE: in a file that logs metrics, a quoted literal on a `log_scalar` or
`.append(` code line must not have the shape `word/word` — the shape of a
namespaced metric name. The naming convention itself (`_loss`, `_mean`,
`eval_`, per-task suffixes) is in `docs-site/.../tooling/logging.mdx`.

## ⚠ Scope, and how the rule was checked

Only files that mention `log_scalar`, and only lines that log or build a
name list, so a file path in an unrelated `.append` elsewhere is not read as
a metric. Run against the tree before the rename, the same rule found 136
namespaced names — it is not vacuous by construction. The scanned-file
count is floored for the same reason.
"""

from noeira.io.fileio import read_file_bytes
from noeira.io.proc import run_capture


comptime MIN_FILES = 900
comptime MIN_LOGGING_FILES = 40
"""Files that mention `log_scalar`: ~60 on 2026-09-22. Fewer means the
`find` or the filter stopped reaching them."""


def _read(path: String) raises -> String:
    var bytes = read_file_bytes(path)
    bytes.append(0)
    return String(unsafe_from_utf8_ptr=bytes.unsafe_ptr())


def _is_ident_byte(byte: UInt8) -> Bool:
    var b = Int(byte)
    return (
        (b >= ord("a") and b <= ord("z"))
        or (b >= ord("A") and b <= ord("Z"))
        or (b >= ord("0") and b <= ord("9"))
        or b == ord("_")
    )


def _is_namespaced(lit: String) -> Bool:
    """`word/word…`: identifier bytes and at least one `/`, not leading.
    A path with a dot, a space or a leading slash is not this shape."""
    var bytes = lit.as_bytes()
    var n = len(bytes)
    if n < 3 or Int(bytes[0]) == ord("/"):
        return False
    var slashes = 0
    for i in range(n):
        var b = bytes[i]
        if Int(b) == ord("/"):
            slashes += 1
        elif not _is_ident_byte(b):
            return False
    return slashes > 0 and Int(bytes[n - 1]) != ord("/")


def _literals(line: String) -> List[String]:
    """Every `"…"` on the line (no escapes in metric names)."""
    var out = List[String]()
    var bytes = line.as_bytes()
    var i = 0
    var start = -1
    while i < len(bytes):
        if Int(bytes[i]) == ord('"'):
            if start < 0:
                start = i + 1
            else:
                out.append(String(line[byte = start : i]))
                start = -1
        i += 1
    return out^


def main() raises:
    print("=== metric names are flat snake_case ===")
    var listing = run_capture(
        String("find noeira examples tools -name '*.mojo' -type f"),
        max_bytes=1 << 22,
    )
    var files = List[String]()
    for f in listing.split("\n"):
        var s = String(String(f).strip())
        if s.byte_length() > 0:
            files.append(s)

    var logging_files = 0
    var offenders = 0
    for path in files:
        var text = _read(path)
        if text.find("log_scalar") < 0:
            continue
        logging_files += 1
        var n = 0
        for line in text.split("\n"):
            n += 1
            var l = String(String(line).strip())
            if l.byte_length() == 0 or l.startswith("#"):
                continue
            if l.find("log_scalar") < 0 and l.find(".append(") < 0:
                continue
            for lit in _literals(l):
                if _is_namespaced(lit):
                    offenders += 1
                    print(
                        "    " + path + ":" + String(n) + " `" + lit
                        + "` is a namespaced metric name"
                    )

    print(
        "  " + String(len(files)) + " files, " + String(logging_files)
        + " log metrics, " + String(offenders) + " namespaced name(s)"
    )
    if len(files) < MIN_FILES:
        raise Error(
            "scanned only " + String(len(files)) + " files; run from the"
            " repo root"
        )
    if logging_files < MIN_LOGGING_FILES:
        raise Error(
            "gate went vacuous: only " + String(logging_files)
            + " files mention log_scalar"
        )
    if offenders != 0:
        raise Error(
            String(offenders) + " namespaced metric name(s): use flat"
            " snake_case (`fb/measure` -> `fb_measure_loss`) so the"
            " dashboard's metric groups match them"
        )
    print("[PASS] metric names are flat")
