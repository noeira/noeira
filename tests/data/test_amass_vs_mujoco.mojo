"""Our reading of the AMASS dump against MuJoCo running the dump's own recipe.

    pixi run python tools/g1/amass_reference_frames.py \
        --clips "$C/A1 - Stand_poses_120_jpos.npy" \
                "$C/A10 - lie to crouch_poses_120_jpos.npy" \
                "$C/A11 - crawl forward_poses_120_jpos.npy" \
                "$C/A12 - crawl backwards_poses_120_jpos.npy" --out /tmp/amass_ref
    pixi run mojo run -I . examples/g1/amass_import.mojo \
        --subset ACCAD --max-clips 4 --full --keep-source --out /tmp/amass_gate.h5
    pixi run mojo run -I . tests/data/test_amass_vs_mujoco.mojo \
        /tmp/amass_gate.h5 /tmp/amass_ref

WHY THIS EXISTS
===============
`test_amass.mojo` gates step 1 against fixtures whose answer is known by
construction. That proves the code does what this repo BELIEVES the dataset
means. It cannot prove the belief — the layout is stated in exactly one
place, the dataset's own `g1/visualize.py`, and a misreading of it produces
a well-formed store of the wrong robot pose.

So this one runs that script's recipe through MuJoCo on real clips and
compares BODY POSITIONS — the thing a wrong root rotation or a wrong DoF
slice moves, and the thing nothing else in our pipeline can be wrong about
without this gate noticing. It compares store row 0 of each episode, the
only row the 50 Hz resampler reproduces exactly (time 0 → phase 0 → index
0, blend 0, with no float32 truncation to argue about).

⚠ ROW 0'S BODY QUATERNION IS NOT NECESSARILY FRAME 0'S, AND WHICH IT IS
CANNOT BE PREDICTED PER BODY. `idx1` at row 0 is frame ONE; the blend weight
is zero, which makes the POSITIONS frame 0 exactly, but the rotations do not
take the blend — they take the reference's `slerp`, and it has two live
branches here (§10 of `noeira/data/lafan.mojo`, reproduced deliberately):

  * `c >= 1` — the float32 dot of two consecutive 120 Hz frames rounds to
    exactly 1 — returns `q0`;
  * `sin(half-angle) < 1e-3`, which two consecutive 120 Hz frames otherwise
    always satisfy, returns the UNNORMALISED MIDPOINT `(q0 + q1) / 2`.

Which branch a body takes depends on how fast THAT BODY is moving at frame
0, so it varies within a clip. Measured: comparing to frame 0 alone fails at
3.9e-4 on the two clips moving at frame 0 and passes at 1.4e-5 on the one
standing still; comparing to the midpoint alone inverts that and fails at
2.9e-2 on the standing clip. Both of those are the artefact, not a layout
error. So this gate asserts each body matches ONE OF THE TWO — a real
constraint, since a mis-ordered quaternion matches neither — and leaves
WHICH to `test_lafan_import_vs_oracle`, whose business step 4 is.

The POSITIONS are frame 0 exactly, and they are what pins the layout.

⚠ THE CONTROL IS THE POINT. The oracle also runs the three plausible
misreadings — the `+ 0.793` dropped, the quaternion read WXYZ, the DoF
slice off by one — and this gate asserts our store does NOT match them. On
`A1 - Stand`, whose frame 0 is nearly the default pose, all four variants
put most bodies within centimetres of each other; without the control a
fixture that cannot discriminate passes exactly as a correct reader does.
`A10 - lie to crouch` and `A11 - crawl forward` start prone, which is why
they are in the clip list.

It needs the `--full` schema: the lean store the importer writes by default
has no `body_*` columns (nothing on the training path reads them).
"""

from std.sys import argv

from noeira.core.bytes import string_from_bytes
from noeira.data.lafan import LAFAN_N_BODIES, LAFAN_N_SKEL
from noeira.data.store import TrajectoryStore
from noeira.io.fileio import read_file_bytes
from noeira.io.json import J_ARRAY, parse_json


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


def main() raises:
    var raw = argv()
    var store_path = String(raw[1]) if len(raw) > 1 else String("/tmp/amass_gate.h5")
    var ref_dir = String(raw[2]) if len(raw) > 2 else String("/tmp/amass_ref")
    var t = Tally()

    var st = TrajectoryStore(store_path)
    var n = st.n_rows()
    var bpos = st.load_column[DType.float32](String("body_pos"))
    var bquat = st.load_column[DType.float32](String("body_quat"))
    print("store:", st.n_episodes(), "episodes,", n, "rows")

    var js = string_from_bytes(read_file_bytes(ref_dir + "/frames.json"))
    var jb = List[UInt8]()
    for i in range(js.byte_length()):
        jb.append(js.as_bytes()[i])
    var doc = parse_json(jb^)
    var arr = doc.root()
    if doc.kind_of(arr) != J_ARRAY:
        raise Error("the oracle dump is not a JSON array")
    var n_clips = doc.size(arr)
    print("oracle:", n_clips, "clips")
    if n_clips != st.n_episodes():
        raise Error(
            "the oracle has " + String(n_clips) + " clips and the store "
            + String(st.n_episodes()) + " episodes — they must be the SAME"
            " clips in the SAME order (both sorted by path)"
        )

    # the row offset of each episode's first row
    var starts = List[Int]()
    for e in range(st.n_episodes()):
        starts.append(st.episodes.start_of(e))

    # ⚠ THE BAND. Our body poses come from the engine's float64 kinematics
    # cast to float32; MuJoCo's from its own float64 chain. §13 measured the
    # two agreeing to 2.4e-6 on LAFAN body positions, so 5e-5 m here is
    # twenty times that and still four orders below the smallest thing a
    # misreading moves (the DoF shift below is the tightest, at 7 mm).
    comptime TOL: Float64 = 5e-5
    comptime QTOL: Float64 = 1e-4
    var worst_pos = 0.0
    var worst_quat = 0.0
    var ctrl_min = List[Float64](length=3, fill=1e9)

    for c in range(n_clips):
        var rec = doc.at(arr, c)
        var name = doc.string(doc.field(rec, String("clip")))
        var vars_ = doc.field(rec, String("variants"))
        var q1n = doc.field(rec, String("quat_frame1"))
        var r0 = starts[c]
        var clip_worst = 0.0
        var clip_qworst = 0.0
        for vi in range(doc.size(vars_)):
            var vnode = doc.at(vars_, vi)
            var which = doc.integer(doc.field(vnode, String("variant")))
            var pos = doc.field(vnode, String("pos"))
            var quat = doc.field(vnode, String("quat"))
            var mx = 0.0
            var mq = 0.0
            for s in range(LAFAN_N_SKEL):
                var pr = doc.at(pos, s)
                var qr = doc.at(quat, s)
                for k in range(3):
                    var got = Float64(bpos[(r0 * LAFAN_N_BODIES + s) * 3 + k])
                    var want = doc.number(doc.at(pr, k))
                    var d = got - want
                    if d < 0:
                        d = -d
                    if d > mx:
                        mx = d
                # The store's row-0 rotation is the slerp MIDPOINT of frames
                # 0 and 1 (see the header), so that is what we compare to —
                # with frame 1 first aligned into frame 0's hemisphere, as the
                # slerp's own `c < 0` branch does. Only variant 0 has frame 1.
                var dq = 0.0
                if which == 0:
                    var dot = 0.0
                    for k in range(4):
                        dot += (
                            doc.number(doc.at(qr, k))
                            * doc.number(doc.at(doc.at(q1n, s), k))
                        )
                    var sgn = 1.0 if dot >= 0.0 else -1.0
                    # Both admissible branches, and the body must match ONE.
                    var e0a = 0.0
                    var e0b = 0.0
                    var e1a = 0.0
                    var e1b = 0.0
                    for k in range(4):
                        var got = Float64(bquat[(r0 * LAFAN_N_BODIES + s) * 4 + k])
                        var f0 = doc.number(doc.at(qr, k))
                        var mid = 0.5 * (
                            f0 + sgn * doc.number(doc.at(doc.at(q1n, s), k))
                        )
                        var d = got - f0
                        if d < 0:
                            d = -d
                        if d > e0a:
                            e0a = d
                        d = got + f0
                        if d < 0:
                            d = -d
                        if d > e0b:
                            e0b = d
                        d = got - mid
                        if d < 0:
                            d = -d
                        if d > e1a:
                            e1a = d
                        d = got + mid
                        if d < 0:
                            d = -d
                        if d > e1b:
                            e1b = d
                    var e0 = e0a if e0a < e0b else e0b
                    var e1 = e1a if e1a < e1b else e1b
                    dq = e0 if e0 < e1 else e1
                if dq > mq:
                    mq = dq
            if which == 0:
                clip_worst = mx
                clip_qworst = mq
                if mx > worst_pos:
                    worst_pos = mx
                if mq > worst_quat:
                    worst_quat = mq
            else:
                if mx < ctrl_min[which - 1]:
                    ctrl_min[which - 1] = mx
        t.truth(
            clip_worst <= TOL,
            name + ": 30 body positions within " + String(TOL) + " m (worst "
            + String(clip_worst) + ")",
        )
        t.truth(
            clip_qworst <= QTOL,
            name + ": 30 body rotations match one of the slerp's two branches"
            " (worst " + String(clip_qworst) + ")",
        )

    print("")
    print("worst body position over all clips:", worst_pos, "m")
    print("worst body rotation over all clips:", worst_quat)

    # ── the control: each misreading must be FAR, on at least one clip ────
    print("")
    print("[control] the three plausible misreadings, closest clip:")
    var labels = List[String]()
    labels.append(String("the `+ 0.793` height offset dropped"))
    labels.append(String("the root quaternion read WXYZ"))
    labels.append(String("the DoF slice shifted by one"))
    for i in range(3):
        print("  " + labels[i] + ": " + String(ctrl_min[i]) + " m away")
        t.truth(
            ctrl_min[i] > 100.0 * TOL,
            labels[i] + " is rejected by this gate (" + String(ctrl_min[i])
            + " m > " + String(100.0 * TOL) + ")",
        )

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_amass_vs_mujoco: " + String(t.fails) + " failed")
