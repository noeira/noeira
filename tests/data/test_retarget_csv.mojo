"""The hhtools trajectory reader and its import (`noeira/data/retarget_csv.mojo`).

    pixi run python tools/supermimic/retarget_oracle.py --out /tmp/retarget_synth
    pixi run mojo run -I . tests/data/test_retarget_csv.mojo /tmp/retarget_synth

WHAT EACH TEST CAN CATCH
========================
rows      every fixture against the oracle's own numpy resample: a joint
          column read by position instead of by name (the fixtures' columns are
          shuffled), the quaternion taken as WXYZ, the lift left in or
          subtracted twice, a wrong velocity column offset for a hinge count
          that is not the G1's, a wrong slerp weight (one fixture spins fast
          enough to leave the linear branch; the oracle asserts it).
          Tolerance 1e-10 relative to the row's largest entry: the CSV crosses
          decimal text.
lift      the same file read with `remove_lift=False`: root height differs by
          exactly the header's value, nothing else moves.
refusals  a missing joint, an extra joint column, a ragged row, no sample
          rate, a NaN.
import    the store's episodes and row counts, the manifest's verdicts (the
          20-frame fixture is `too_short` at 50 Hz), and the first stored row
          equal to the oracle's in float32.
"""

from std.math import abs
from std.os.path import exists
from std.sys import argv
from std.testing import assert_true

from noeira.core.bytes import string_from_bytes
from noeira.data.bones_seed import resample_mjlab_rows
from noeira.data.retarget_csv import import_retarget_clips, load_retarget_csv
from noeira.data.store import TrajectoryStore
from noeira.io.fileio import read_file_bytes, remove_file


comptime NDOF = 5
comptime WIDTH = 13 + 2 * NDOF


def _dir() -> String:
    var a = argv()
    if len(a) < 2:
        return String("/tmp/retarget_synth")
    return String(a[len(a) - 1])


def _joints() -> List[String]:
    return [String("ja"), String("jb"), String("jc"), String("jd"), String("je")]


def _names() -> List[String]:
    return [String("walk_a"), String("spin_b"), String("slow_c"), String("short_d")]


def _ref(name: String) raises -> Tuple[Int, Float64, List[Float64]]:
    """(frames, lift, rows) of the oracle's file."""
    var lines = string_from_bytes(read_file_bytes(_dir() + "/ref/" + name + ".txt")).split("\n")
    var head = String(lines[0]).split(" ")
    var n = Int(String(head[0]))
    assert_true(Int(String(head[1])) == WIDTH, "the oracle's row width")
    var rows = List[Float64]()
    for i in range(1, n + 1):
        for f in String(lines[i]).split(" "):
            rows.append(Float64(String(f)))
    assert_true(len(rows) == n * WIDTH, "the oracle's row count")
    return (n, Float64(String(head[2])), rows^)


def test_rows_match_the_oracle() raises:
    var worst = 0.0
    var names = _names()
    for c in range(len(names)):
        var want = _ref(names[c])
        var clip = load_retarget_csv(_dir() + "/clips/" + names[c] + ".csv", names[c].copy(), _joints())
        assert_true(clip.lift_removed and clip.z_lift == want[1], names[c] + ": the header's lift")
        var out = resample_mjlab_rows(clip.raw, clip.n, NDOF, clip.fps)
        assert_true(out[0] == want[0], names[c] + ": " + String(out[0]) + " frames, the oracle has " + String(want[0]))
        for i in range(want[0]):
            var scale = 1.0
            for k in range(WIDTH):
                scale = max(scale, abs(want[2][i * WIDTH + k]))
            for k in range(WIDTH):
                var e = abs(out[1][i * WIDTH + k] - want[2][i * WIDTH + k]) / scale
                worst = max(worst, e)
                assert_true(
                    e < 1e-10,
                    names[c] + " frame " + String(i) + " column " + String(k) + ": " + String(out[1][i * WIDTH + k])
                    + " against " + String(want[2][i * WIDTH + k]),
                )
    print("  4 clips, worst relative difference", worst)


def test_lift_is_the_only_difference() raises:
    var with_lift = load_retarget_csv(_dir() + "/clips/walk_a.csv", String("walk_a"), _joints(), False)
    var without = load_retarget_csv(_dir() + "/clips/walk_a.csv", String("walk_a"), _joints(), True)
    assert_true(without.z_lift < -0.02 and not with_lift.lift_removed, "walk_a carries a lift of about -22 mm")
    comptime RAW = 7 + NDOF
    for i in range(with_lift.n):
        for k in range(RAW):
            var d = with_lift.raw[i * RAW + k] - without.raw[i * RAW + k]
            if k == 2:
                assert_true(abs(d - without.z_lift) < 1e-15, "root z differs by the lift")
            else:
                assert_true(d == 0.0, "only root z moves")
    print("  lift", without.z_lift, "taken out of root z only")


def _refused(name: String) -> Bool:
    try:
        _ = load_retarget_csv(_dir() + "/bad/" + name + ".csv", name.copy(), _joints())
    except e:
        print("    " + name + ": " + String(e))
        return True
    return False


def test_refusals() raises:
    assert_true(_refused(String("missing_joint")), "a file without one of the robot's joints was read")
    assert_true(_refused(String("extra_joint")), "a file with a joint the robot does not have was read")
    assert_true(_refused(String("ragged")), "a ragged row was read")
    assert_true(_refused(String("no_rate")), "a file without a sample rate was read")
    assert_true(_refused(String("nan")), "a NaN was read")


def test_import() raises:
    var out = _dir() + "/lib.h5"
    var man = _dir() + "/lib.manifest.csv"
    if exists(out):
        remove_file(out)
    var paths = List[String]()
    var names = _names()
    for c in range(len(names)):
        paths.append(_dir() + "/clips/" + names[c] + ".csv")
    paths.append(_dir() + "/bad/ragged.csv")
    var s = import_retarget_clips(paths, _joints(), out, man, String("synthetic_retarget_50hz"), verbose=False)
    assert_true(s.clips == 5 and s.kept == 3 and s.refused == 1 and s.too_short == 1, "the import's counts")
    assert_true(s.lift_min < -0.02 and s.lift_max == 0.0, "the kept clips' lifts span -22 mm to 0")
    var st = TrajectoryStore(out)
    assert_true(st.n_episodes() == 3, "three episodes")
    var refc = st.load_column[DType.float32](String("ref"))
    var total = 0
    for e in range(3):
        var want = _ref(names[e])
        assert_true(Int(st.episodes.ep_len[e]) == want[0], names[e] + ": episode length")
        for k in range(WIDTH):
            assert_true(
                Float32(refc[total * WIDTH + k]) == Float32(want[2][k]) or abs(Float64(refc[total * WIDTH + k]) - want[2][k]) < 1e-6,
                names[e] + ": first stored row, column " + String(k),
            )
        total += want[0]
    assert_true(s.rows_kept == total, "rows kept")
    var m = string_from_bytes(read_file_bytes(man))
    assert_true("short_d,8,0,too_short" in m, "the 20-frame clip is 8 frames at 50 Hz and too short")
    assert_true("ragged,0,0,refused" in m, "the ragged file is refused in the manifest")
    assert_true("walk_a," in m and ",1,," in m, "kept clips are listed")
    print("  store:", st.n_episodes(), "episodes,", total, "rows")


def main() raises:
    # Called directly: the fixture directory is a positional argument, which
    # `TestSuite` would take for one of its own. A failed assertion raises,
    # so the exit code is the verdict.
    print("fixtures:", _dir())
    print("[rows]")
    test_rows_match_the_oracle()
    print("[lift]")
    test_lift_is_the_only_difference()
    print("[refusals]")
    test_refusals()
    print("[import]")
    test_import()
    print("ALL PASSED")
