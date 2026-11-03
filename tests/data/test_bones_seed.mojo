"""The BONES-SEED import (M1 of `docs/SUPERMIMIC_PORT_PLAN.md`), against the
SuperMimic reproduction's own code.

    pixi run python tools/supermimic/make_synthetic_seed.py --out /tmp/seed_synth
    pixi run mojo run -I . tests/data/test_bones_seed.mojo /tmp/seed_synth

The same command on a dump made with `--real-dir <SEED CSVs>` gates real
clips (private storage only: SEED is confidential).

WHAT EACH SECTION CAN CATCH
===========================
[1] the Euler convention, on answers known BY CONSTRUCTION (not from scipy):
    90° about one axis at a time, then (90°, 90°, 0) — extrinsic 'xyz' gives
    (½, ½, ½, −½), intrinsic 'XYZ' gives (½, ½, ½, +½). The sign of w on a
    > 180° rotation is checked too, because scipy does not canonicalise it.
[2] the reader's refusals: a permuted joint header (a valid pose with two
    joints exchanged if read), a missing root column, a ragged row; and the
    units on a two-row CSV written here (cm → m, deg → rad).
[3] every clip of the dump against the reproduction's `resample_motion`
    (`tools/supermimic/make_synthetic_seed.py` runs it): the raw rows and the
    71-d reference rows. Tolerance is 1e-11 relative: the CSV crosses decimal
    text, where `Float64(String)` can be one ULP low, and LLVM may fuse a
    blend's `a·b + c` (numpy rounds twice). A wrong convention is ≥ 1e-3.
    The fixtures are checked to EXERCISE the slerp's non-linear branch
    (dot ≤ 0.9995) — without that, a wrong sin-weight passes.
[4] the name filter against the reproduction's `name_filter`, on probes
    shaped as substring traps plus every clip name of the dump.
[5] the BFM-Zero path runs: `convert_seed_clip` on one clip, the store's row
    count, and its first root pose equal to the clip's (float32).
[6] the feasibility filter on OUR G1. First each threshold's edge on rows
    and heights built here (29 vs 30 frames elevated, 80 % vs 81 % below
    −0.08, 24 vs 25 frames floating, a joint speed that opens the "still"
    gate). Then every clip of the dump against the reproduction's
    `feasibility_filter` run on our MjModel: per-frame lowest link origin and
    sole height (1e-12 m: FK of nearly equal poses) and the verdict, exactly.
    The fixtures must reach all four verdicts, and each of the upright,
    standing, still and duration gates is shown stopping one (fixtures
    only: a `--real-dir` dump, marked by a `REAL` file, checks parity).
[7] `import_seed_clips` over every CSV of the dump, refused one included:
    the manifest's verdicts are the name filter's, else the feasibility
    oracle's; the store holds exactly the kept clips, in order, named in the
    task table, their rows the float32 of `resample_mjlab`.
"""

from std.math import ceil, sqrt
from std.os.path import exists
from std.sys import argv

from noeira.core.bytes import string_from_bytes
from noeira.data.bones_seed import (
    SEED_RAW_DIM, SEED_REF_DIM, euler_xyz_deg_to_quat_wxyz, load_seed_csv,
    parse_seed_csv, resample_mjlab, seed_name_filter,
)
from noeira.data.bones_seed import MimicMotion
from noeira.data.bones_seed_filter import (
    SeedBodyHeights, feasibility_reason, longest_run, seed_body_heights,
)
from noeira.data.bones_seed_import import (
    SEED_MOTION_ID_COLUMN, SEED_REF_COLUMN, import_seed_clips,
)
from noeira.data.bones_seed_store import convert_seed_clip
from noeira.data.store import TrajectoryStore
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_dof_name
from noeira.io.fileio import read_file_bytes
from noeira.io.npy import load_npy_2d_f64


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)

    def near(mut self, got: Float64, want: Float64, tol: Float64, msg: String):
        var d = abs(got - want)
        self.truth(d <= tol, msg + " (got " + String(got) + ", want " + String(want) + ")")


def _header(permute: Bool = False, drop: String = "") -> String:
    var cols: List[String] = [
        "Frame", "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
    ]
    var j = List[String]()
    for i in range(G1_N_DOF):
        j.append(String(g1_dof_name(i)) + "_dof")
    if permute:
        var tmp = j[3]
        j[3] = j[9]
        j[9] = tmp
    var h = String("")
    for c in cols:
        if c == drop:
            continue
        h += (c if h.byte_length() == 0 else "," + c)
    for c in j:
        h += "," + c
    return h


def _row(frame: Int, x: Float64, z: Float64, rz: Float64, jdeg: Float64, ncols_drop: Int = 0) -> String:
    var r = String(frame) + "," + String(x) + ",0," + String(z) + ",0,0," + String(rz)
    for _ in range(G1_N_DOF - ncols_drop):
        r += "," + String(jdeg)
    return r


def _raises_parse(text: String) -> Bool:
    try:
        _ = parse_seed_csv(text, String("probe"))
    except:
        return True
    return False


def _lines(path: String) raises -> List[String]:
    var out = List[String]()
    for l in string_from_bytes(read_file_bytes(path)).split("\n"):
        if String(l).strip().byte_length() > 0:
            out.append(String(l))
    return out^


def _mask(n: Int, lo: Int, hi: Int) -> List[Bool]:
    var m = List[Bool](length=n, fill=False)
    for i in range(lo, hi):
        m[i] = True
    return m^


def _fill(mut v: List[Float64], lo: Int, hi: Int, x: Float64):
    for i in range(lo, hi):
        v[i] = x


def _still_motion(n: Int, qd: Float64 = 0.0) -> MimicMotion:
    """`n` 50 Hz rows standing still: pelvis at 0.79 m, upright, every joint
    speed `qd`."""
    var fr = List[Float64](length=n * SEED_REF_DIM, fill=0.0)
    for i in range(n):
        fr[i * SEED_REF_DIM + 2] = 0.79
        fr[i * SEED_REF_DIM + 3] = 1.0
        for j in range(G1_N_DOF):
            fr[i * SEED_REF_DIM + 42 + j] = qd
    return MimicMotion(50.0, n, fr^)


def main() raises:
    var a = argv()
    var dump = String(a[1]) if len(a) > 1 else String("/tmp/seed_synth")
    var t = Tally()
    # A `--real-dir` dump checks parity only; the coverage checks (every
    # verdict reached, a refusal present) are properties of the FIXTURES.
    var synthetic = not exists(dump + "/REAL")
    print("dump:", dump, "(synthetic fixtures)" if synthetic else "(REAL clips: coverage checks skipped)")
    comptime H = 0.7071067811865476

    # ── [1] the Euler convention ────────────────────────────────────────────
    print("\n[1] Euler 'xyz' (extrinsic, degrees) -> WXYZ, known answers")
    var q = euler_xyz_deg_to_quat_wxyz(90.0, 0.0, 0.0)
    t.truth(abs(q[0] - H) < 1e-15 and abs(q[1] - H) < 1e-15 and abs(q[2]) < 1e-15 and abs(q[3]) < 1e-15, "90 deg about x -> (c, s, 0, 0)")
    q = euler_xyz_deg_to_quat_wxyz(0.0, 90.0, 0.0)
    t.truth(abs(q[0] - H) < 1e-15 and abs(q[2] - H) < 1e-15 and abs(q[1]) < 1e-15 and abs(q[3]) < 1e-15, "90 deg about y -> (c, 0, s, 0)")
    q = euler_xyz_deg_to_quat_wxyz(0.0, 0.0, 90.0)
    t.truth(abs(q[0] - H) < 1e-15 and abs(q[3] - H) < 1e-15 and abs(q[1]) < 1e-15 and abs(q[2]) < 1e-15, "90 deg about z -> (c, 0, 0, s)")
    q = euler_xyz_deg_to_quat_wxyz(90.0, 90.0, 0.0)
    t.truth(
        abs(q[0] - 0.5) < 1e-15 and abs(q[1] - 0.5) < 1e-15 and abs(q[2] - 0.5) < 1e-15 and abs(q[3] + 0.5) < 1e-15,
        "(90, 90, 0) -> (1/2, 1/2, 1/2, -1/2): EXTRINSIC (intrinsic XYZ would give +1/2)",
    )
    q = euler_xyz_deg_to_quat_wxyz(0.0, 0.0, 270.0)
    t.truth(q[0] < 0.0 and abs(q[0] + H) < 1e-15, "270 deg about z keeps w = -0.707 (scipy does not canonicalise)")

    # ── [2] the reader ──────────────────────────────────────────────────────
    print("\n[2] the CSV reader: units and refusals")
    var ok_text = _header() + "\n" + _row(0, 150.0, 79.0, 90.0, 45.0) + "\n" + _row(1, 151.0, 79.5, 90.0, -30.0) + "\n"
    var c = parse_seed_csv(ok_text, String("units"))
    t.truth(c.n == 2 and c.fps == 120.0, "2 frames at 120 fps")
    t.near(c.raw[0], 1.5, 1e-15, "root x: 150 cm -> 1.5 m")
    t.near(c.raw[2], 0.79, 1e-15, "root z: 79 cm -> 0.79 m")
    t.near(c.raw[3], H, 1e-15, "root w of 90 deg about z")
    t.near(c.raw[6], H, 1e-15, "root z-component of 90 deg about z (WXYZ: index 3)")
    t.near(c.raw[7], 0.7853981633974483, 1e-15, "joint 0: 45 deg -> pi/4")
    t.near(c.raw[SEED_RAW_DIM + 7 + 28], -0.5235987755982988, 1e-15, "frame 1 joint 28: -30 deg")
    t.truth(_raises_parse(_header(permute=True) + "\n" + _row(0, 0, 79, 0, 1) + "\n"), "a permuted joint header is REFUSED")
    t.truth(_raises_parse(_header(drop="root_rotateY") + "\n" + _row(0, 0, 79, 0, 1) + "\n"), "a missing root column is REFUSED")
    t.truth(_raises_parse(_header() + "\n" + _row(0, 0, 79, 0, 1, ncols_drop=1) + "\n"), "a ragged row is REFUSED")
    t.truth(_raises_parse(String("0,1,2\n")), "a file without the SEED header is REFUSED")

    # ── [3] every clip against the reproduction's resample ──────────────────
    print("\n[3] the dump at", dump, "against `resample_motion`")
    var clips = _lines(dump + "/clips.tsv")
    var n_ok = 0
    var nonlinear = 0
    var worst = 0.0
    var bfm_clip = String("")
    var bfm_path = String("")
    for li in range(len(clips)):
        var f = clips[li].split("\t")
        var stem = String(f[0])
        var status = String(f[1])
        var path = String(f[2])
        if status == "refused":
            var raised = False
            try:
                _ = load_seed_csv(path, stem)
            except:
                raised = True
            t.truth(raised, stem + ": refused by both sides")
            continue
        var clip = load_seed_csv(path, stem)
        var raw = load_npy_2d_f64(dump + "/ref/" + stem + ".raw.npy", SEED_RAW_DIM)
        var oref = load_npy_2d_f64(dump + "/ref/" + stem + ".ref71.npy", SEED_REF_DIM)
        var e_raw = 0.0
        var same_n = raw.rows == clip.n
        if same_n:
            for i in range(clip.n * SEED_RAW_DIM):
                e_raw = max(e_raw, abs(clip.raw[i] - raw.values[i]) / max(1.0, abs(raw.values[i])))
        var m = resample_mjlab(clip)
        var e_ref = 0.0
        var same_out = oref.rows == m.n
        if same_out:
            for i in range(m.n * SEED_REF_DIM):
                e_ref = max(e_ref, abs(m.frames[i] - oref.values[i]) / max(1.0, abs(oref.values[i])))
        worst = max(worst, max(e_raw, e_ref))
        t.truth(
            same_n and same_out and e_raw <= 1e-11 and e_ref <= 1e-11,
            stem + ": " + String(clip.n) + " -> " + String(m.n) + " rows (oracle "
            + String(oref.rows) + "), max rel err raw " + String(e_raw) + ", ref71 " + String(e_ref),
        )
        # coverage of the slerp's two branches, from the raw brackets
        for i in range(clip.n - 1):
            var d = 0.0
            for k in range(4):
                d += clip.raw[i * SEED_RAW_DIM + 3 + k] * clip.raw[(i + 1) * SEED_RAW_DIM + 3 + k]
            if abs(d) <= 0.9995:
                nonlinear += 1
        if clip.n > 300 and bfm_clip.byte_length() == 0:
            bfm_clip = stem
            bfm_path = path
        n_ok += 1
    t.truth(n_ok >= 1, String(n_ok) + " clips compared (worst rel err " + String(worst) + ")")
    t.truth(nonlinear > 0, String(nonlinear) + " consecutive-frame pairs take the slerp's NON-linear branch")

    # ── [4] the name filter ─────────────────────────────────────────────────
    print("\n[4] the name filter against `name_filter`")
    var names = _lines(dump + "/name_filter.tsv")
    var mism = 0
    var dropped = 0
    for li in range(len(names)):
        var f = names[li].split("\t")
        var nm = String(f[0])
        var want = String(f[1]) if len(f) > 1 else String("")
        var got = seed_name_filter(nm)
        if got != want:
            mism += 1
            print("    mismatch:", nm, "mojo '" + got + "' python '" + want + "'")
        if want.byte_length() > 0:
            dropped += 1
    t.truth(mism == 0, String(len(names)) + " names, " + String(dropped) + " dropped, 0 mismatches")
    t.truth(dropped > 0 and dropped < len(names), "the probes exercise both outcomes")

    # ── [5] the BFM-Zero store path ─────────────────────────────────────────
    print("\n[5] BFM-Zero's pipeline on", bfm_clip)
    if bfm_clip.byte_length() == 0:
        t.truth(False, "no clip longer than 300 frames in the dump")
    else:
        var clip = load_seed_csv(bfm_path, bfm_clip)
        var env = UnitreeG1[DType.float64]()
        _ = env.reset()
        var rows = convert_seed_clip(clip, env)
        var want_rows = Int(ceil((Float64(clip.n - 1) / clip.fps) / 0.02))
        t.truth(rows.n_rows == want_rows, String(rows.n_rows) + " rows = ceil(len / 0.02) = " + String(want_rows))
        var e = 0.0
        for k in range(3):
            e = max(e, abs(Float64(rows.qpos[k]) - Float64(Float32(clip.raw[k]))))
        for j in range(G1_N_DOF):
            e = max(e, abs(Float64(rows.qpos[7 + j]) - Float64(Float32(clip.raw[7 + j]))))
        t.truth(e <= 1e-6, "row 0's root position and joints are frame 0's (max err " + String(e) + ")")
        # The root quaternion takes BFM-Zero's float32 matrix round trip and
        # its re-sign rule (largest component positive), and its float32
        # slerp returns a slightly NON-UNIT quaternion even at blend 0
        # (`data/lafan.mojo`: "reproduced, not fixed"; 7e-6 here). So compare
        # the direction: |<q/|q|, q'>|, and report the norm.
        var d = 0.0
        var nq = 0.0
        for k in range(4):
            d += Float64(rows.qpos[3 + k]) * clip.raw[3 + k]
            nq += Float64(rows.qpos[3 + k]) * Float64(rows.qpos[3 + k])
        nq = sqrt(nq)
        t.truth(
            abs(abs(d) / nq - 1.0) <= 1e-6,
            "row 0's root rotation is frame 0's up to sign (|<q/|q|,q'>| = " + String(abs(d) / nq)
            + ", |q| = " + String(nq) + ")",
        )

    # ── [6] the feasibility filter on our G1 ────────────────────────────────
    print("\n[6] the feasibility filter: threshold edges by construction")
    t.truth(longest_run(_mask(10, 3, 7)) == 4 and longest_run(_mask(5, 0, 0)) == 0 and longest_run(_mask(5, 0, 5)) == 5, "longest_run")
    var still = _still_motion(100)
    var h = SeedBodyHeights(100)
    t.truth(feasibility_reason(still, h) == "", "standing still on the floor: kept")
    _fill(h.zmin, 0, 29, 0.25)
    t.truth(feasibility_reason(still, h) == "", "zmin > 0.20 for 29 frames (0.58 s): kept")
    _fill(h.zmin, 0, 30, 0.25)
    t.truth(feasibility_reason(still, h) == "elevated_support", "for 30 frames (0.6 s): elevated_support")
    h = SeedBodyHeights(100)
    _fill(h.zmin, 0, 80, -0.09)
    t.truth(feasibility_reason(still, h) == "", "zmin < -0.08 on 80 % of frames: kept (needs MORE than 80 %)")
    _fill(h.zmin, 0, 81, -0.09)
    t.truth(feasibility_reason(still, h) == "ground_penetration", "on 81 %: ground_penetration")
    h = SeedBodyHeights(100)
    h.zmin[50] = -0.1201
    t.truth(feasibility_reason(still, h) == "ground_penetration", "one frame below -0.12: ground_penetration")
    h = SeedBodyHeights(100)
    _fill(h.foot_z, 10, 34, 0.03)
    t.truth(feasibility_reason(still, h) == "", "soles at 0.03 m for 24 frames (0.48 s): kept")
    _fill(h.foot_z, 10, 35, 0.03)
    t.truth(feasibility_reason(still, h) == "floating_feet", "for 25 frames (0.5 s): floating_feet")
    var spin = _still_motion(100, qd=0.51)
    t.truth(feasibility_reason(spin, h) == "", "the same, joint speed RMS 0.51 rad/s (not still): kept")

    print("\n[6] the feasibility filter on our G1 against `feasibility_filter`")
    var feas = _lines(dump + "/feasibility.tsv")
    var path_of = Dict[String, String]()
    for li in range(len(clips)):
        var f = clips[li].split("\t")
        path_of[String(f[0])] = String(f[2])
    var env6 = UnitreeG1[DType.float64]()
    _ = env6.reset()
    var seen = Dict[String, Int]()
    var worst_h = 0.0
    var n_feas = 0
    for li in range(len(feas)):
        var f = feas[li].split("\t")
        var stem = String(f[0])
        var want = String(f[1]) if len(f) > 1 else String("")
        var m = resample_mjlab(load_seed_csv(path_of[stem], stem))
        var hh = seed_body_heights(m, env6)
        var oh = load_npy_2d_f64(dump + "/ref/" + stem + ".feas.npy", 2)
        var e = 0.0
        if oh.rows == m.n:
            for i in range(m.n):
                e = max(e, max(abs(hh.zmin[i] - oh.values[i * 2]), abs(hh.foot_z[i] - oh.values[i * 2 + 1])))
        else:
            e = 1.0
        worst_h = max(worst_h, e)
        var got = feasibility_reason(m, hh)
        t.truth(
            e <= 1e-12 and got == want,
            stem + ": '" + got + "' (oracle '" + want + "'), heights max err " + String(e),
        )
        seen[want] = seen.get(want, 0) + 1
        n_feas += 1
    var verdicts = String("")
    for e in seen.items():
        verdicts += " " + (e.key if e.key.byte_length() > 0 else String("keep")) + "=" + String(e.value)
    print("    verdicts:" + verdicts + "; worst height err " + String(worst_h) + " m")
    if synthetic:
        t.truth(
            "" in seen and "elevated_support" in seen and "ground_penetration" in seen and "floating_feet" in seen,
            String(n_feas) + " fixture clips reach all four verdicts",
        )

    # ── [7] the import: store + manifest ────────────────────────────────────
    print("\n[7] import_seed_clips over the whole dump, read back")
    var want_reason = Dict[String, String]()
    for li in range(len(names)):
        var f = names[li].split("\t")
        var r = String(f[1]) if len(f) > 1 else String("")
        if r.byte_length() > 0:
            want_reason[String(f[0])] = r
    for li in range(len(feas)):
        var f = feas[li].split("\t")
        var stem = String(f[0])
        if stem not in want_reason:
            want_reason[stem] = String(f[1]) if len(f) > 1 else String("")
    var all_paths = List[String]()
    for li in range(len(clips)):
        var f = clips[li].split("\t")
        all_paths.append(String(f[2]))
        if String(f[1]) == "refused":
            want_reason[String(f[0])] = "refused"
    var lib = dump + "/lib.h5"
    var man_path = dump + "/lib.manifest.csv"
    var summ = import_seed_clips(all_paths, lib, man_path, verbose=False)
    var man = _lines(man_path)
    t.truth(man[0] == "name,frames,keep,reason" and len(man) == len(all_paths) + 1, "manifest: the reproduction's header, one row per input")
    var bad = 0
    var kept_names = List[String]()
    var n_name_drop = 0
    for li in range(1, len(man)):
        var f = man[li].split(",")
        var nm = String(f[0])
        var r = String(f[3]) if len(f) > 3 else String("")
        var keep = String(f[2]) == "1"
        if nm not in want_reason or want_reason[nm] != r or keep != (r.byte_length() == 0):
            bad += 1
            print("    manifest mismatch:", man[li])
        if keep:
            kept_names.append(nm)
        if r.startswith("name:"):
            n_name_drop += 1
    t.truth(bad == 0, String(len(man) - 1) + " verdicts = name filter, then feasibility (refused for the reader's refusals)")
    if synthetic:
        t.truth(n_name_drop > 0 and summ.refused > 0, "the fixtures exercise a name drop and a refusal")
    else:
        print("    real dump: " + String(n_name_drop) + " name drops, " + String(summ.refused) + " refused")
    var st = TrajectoryStore(lib)
    t.truth(st.n_episodes() == summ.kept and summ.kept == len(kept_names), String(st.n_episodes()) + " episodes = kept clips")
    var refc = st.load_column[DType.float32](String(SEED_REF_COLUMN))
    var mid = st.load_column[DType.int32](String(SEED_MOTION_ID_COLUMN))
    var row_bad = 0
    var name_bad = 0
    for e in range(st.n_episodes()):
        var nm = st.manifest.task_text(e)
        if nm != kept_names[e]:
            name_bad += 1
        var m = resample_mjlab(load_seed_csv(path_of[nm], nm))
        var off = Int(st.episodes.ep_offset[e])
        if Int(st.episodes.ep_len[e]) != m.n:
            row_bad += 1
            continue
        for i in range(m.n):
            if Int(mid[off + i]) != e:
                row_bad += 1
            for k in range(SEED_REF_DIM):
                if refc[(off + i) * SEED_REF_DIM + k] != Float32(m.frames[i * SEED_REF_DIM + k]):
                    row_bad += 1
    t.truth(name_bad == 0, "episode e's task text is the e-th kept clip's name")
    t.truth(row_bad == 0, String(st.n_rows()) + " rows = float32 of resample_mjlab, motion_id = episode")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_bones_seed: " + String(t.fails) + " failed")
