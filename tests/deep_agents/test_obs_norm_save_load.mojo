"""`RunningMeanStd.save` / `load`: the obs-norm text file round-trips, and the
write is atomic (no `.tmp` left beside it).

`save` went from an in-place `open(path, "w")` to `write_text_atomic` (the
`test_no_v2_checkpoint_writes` rule), and nothing else reads the file back
before a run resumes from it. The values come back within one ulp, not bit
for bit: Mojo's String -> Float64 parse can be one ulp low
(`_mojos_string_to_float64_is_one_ulp_low`).

Run: pixi run mojo run -I . tests/deep_agents/test_obs_norm_save_load.mojo
"""

from std.math import abs
from std.os import remove
from std.os.path import exists
from noeira.deep_agents.training.obs_norm import RunningMeanStd

comptime DIM = 5


def _close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= 4e-16 * max(abs(a), 1.0)


def main() raises:
    var path = String("/tmp/noeira_test_obs_norm_save_load.txt")
    var s = RunningMeanStd(DIM)
    s.count = 12345.0
    for k in range(DIM):
        s.mean[k] = 0.1 * Float64(k + 1) - 0.25
        s.var_[k] = 1.0 / Float64(3 + k)
    s.save(path)
    if exists(path + ".tmp"):
        raise Error("save left " + path + ".tmp behind: the write was not renamed")

    var r = RunningMeanStd(DIM)
    r.load(path)
    remove(path)
    var fails = 0
    if not _close(r.count, s.count):
        print("  FAIL count", r.count, "want", s.count)
        fails += 1
    for k in range(DIM):
        if not _close(r.mean[k], s.mean[k]):
            print("  FAIL mean[", k, "]", r.mean[k], "want", s.mean[k])
            fails += 1
        if not _close(r.var_[k], s.var_[k]):
            print("  FAIL var[", k, "]", r.var_[k], "want", s.var_[k])
            fails += 1
    if fails > 0:
        raise Error(String(fails, " value(s) did not round-trip"))
    print("test_obs_norm_save_load: ALL PASS (", 2 * DIM + 1, "values round-trip, no .tmp left )")
