"""G4b — our PushT renderer against the dataset's frames, in pixels and in
the eyes of the published encoder.

docs/LEWM_REOPEN_PLAN.md P4. For frames sampled from box session A's pixel
fixture (`tools/lewm/sample_fixture_frames.py` -> /tmp/lewm_frames), our
renderer draws the frame's recorded state (`sim_frame_chw_norm`, origin pose),
and is compared with the recorded frame:

  * pixels: mean |ours - real| (0..255), and the share of pixels off by > 30
    in some channel;
  * the reference encoder (published weights, BN eval): ‖e(ours) - e(real)‖
    against ‖e(real_next) - e(real)‖ — how far the SAME scene moves in latent
    space in ONE env step. A renderer gap the encoder reads as less than a
    step of motion is harmless to planning; one it reads as many steps is not.

GATE: ours vs the dataset, median pixel MAE < 1.0 and median latent gap
< 1 env step (measured: 0.48 / 0.56 with `render_swm.mojo`; the env's own
`render.mojo` drawing: 2.15 / 1.49). The third column, swm 0.0.6's own
re-render of the bare state (`frames.swm_render`, made on the box), sits at
0.83 step from the dataset: ours is at that floor.

Run (after the frames + `pixi run dump-lewm-ref`):
    pixi run -e apple mojo run -I . tests/experimental/lewm/ref/test_pusht_render.mojo
"""

from std.math import sqrt, abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.ref_rollout import RefEncoder, encode_ref, REF_EMB
from noeira.experimental.lewm.pusht_sim_bridge import sim_frame_chw_norm


comptime IMG = 224
comptime HW = IMG * IMG
comptime NB = 8  # frames per encoder call
comptime MEAN: Array[Float64, 3] = [0.485, 0.456, 0.406]
comptime STD: Array[Float64, 3] = [0.229, 0.224, 0.225]


def _norm_chw01(chw01: List[Scalar[DT]], off: Int, mut out: List[Scalar[DT]]):
    """CHW in [0, 1] -> ImageNet-normalised CHW, appended to `out`."""
    var mean = materialize[MEAN]()
    var std = materialize[STD]()
    for c in range(3):
        for i in range(HW):
            out.append(Scalar[DT]((Float64(chw01[off + c * HW + i]) - mean[c]) / std[c]))


def _hwc255_to_chw01(hwc: List[Scalar[DT]], off: Int) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]](length=3 * HW, fill=Scalar[DT](0))
    for i in range(HW):
        for c in range(3):
            out[c * HW + i] = hwc[off + i * 3 + c] / Scalar[DT](255.0)
    return out^


def _pct(mut v: List[Float64], q: Float64) -> Float64:
    sort(v)
    return v[min(len(v) - 1, Int(q * Float64(len(v))))]


def main() raises:
    var fr = RefDump(String("/tmp/lewm_frames"))
    var real = fr.get(String("frames.pixels"))        # (N, 224, 224, 3) 0..255
    var nxt = fr.get(String("frames.next_pixels"))
    var state = fr.get(String("frames.state"))        # (N, 7)
    # stable-worldmodel 0.0.6's own renderer at the same states (run on the
    # box): the third column — ours vs it is OUR rasteriser's error; the
    # dataset vs it is what the recording itself adds
    # optional: made on a box (`tools/lewm/pusht_replay_oracle.py`'s venv);
    # without it the swm rows read as copies of the dataset
    var has_swm = fr.has(String("frames.swm_render"))
    var swm = fr.get(String("frames.swm_render")) if has_swm else real.copy()
    var N = len(state) // 7
    N = (N // NB) * NB
    print("G4b  PushT renderer vs", N, "dataset frames")

    # pairs: 0 ours vs dataset, 1 ours vs swm, 2 dataset vs swm
    var mae = List[List[Float64]]()
    var off30 = List[List[Float64]]()
    for _ in range(3):
        mae.append(List[Float64]())
        off30.append(List[Float64]())
    var ours01 = List[Scalar[DT]]()   # (N, 3, HW) in [0,1]
    var real01 = List[Scalar[DT]]()
    var next01 = List[Scalar[DT]]()
    var swm01 = List[Scalar[DT]]()
    var buf = List[Scalar[DT]](length=3 * HW, fill=Scalar[DT](0))
    for n in range(N):
        sim_frame_chw_norm[IMG](
            state[n * 7 + 2], state[n * 7 + 3], state[n * 7 + 4],
            state[n * 7 + 0], state[n * 7 + 1],
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr()),
        )
        var r = _hwc255_to_chw01(real, n * HW * 3)
        var x = _hwc255_to_chw01(nxt, n * HW * 3)
        var w = _hwc255_to_chw01(swm, n * HW * 3)
        for pair in range(3):
            var s_ = 0.0
            var bad = 0
            for i in range(HW):
                var worst = 0.0
                for c in range(3):
                    var a_ = Float64(buf[c * HW + i]) if pair < 2 else Float64(r[c * HW + i])
                    var b_ = Float64(r[c * HW + i]) if pair == 0 else Float64(w[c * HW + i])
                    var d = abs(a_ - b_) * 255.0
                    s_ += d
                    worst = max(worst, d)
                if worst > 30.0:
                    bad += 1
            mae[pair].append(s_ / Float64(3 * HW))
            off30[pair].append(Float64(bad) / Float64(HW))
        for i in range(3 * HW):
            ours01.append(buf[i])
            real01.append(r[i])
            next01.append(x[i])
            swm01.append(w[i])

    # frame 0, CHW [0,1], one value per line (ours, real) — for inspection
    # only (tools turn it into a PNG); not part of the comparison
    for which in range(2):
        var txt = String()
        for i in range(3 * HW):
            txt += String(ours01[i] if which == 0 else real01[i]) + "\n"
        with open(String("/tmp/lewm_frames/inspect_") + ("ours" if which == 0 else "real") + ".txt", "w") as f:
            f.write(txt)

    var c = DeviceContext()
    var ctx = Optional(c)
    var enc = RefEncoder.make["gpu", Kaiming](ctx)
    _ = load_ref["gpu"](enc, String("/tmp/lewm_ref"), String("emb.0."), ctx)
    var gap = List[List[Float64]]()    # same 3 pairs
    var ratio = List[List[Float64]]()
    for _ in range(3):
        gap.append(List[Float64]())
        ratio.append(List[Float64]())
    var step = List[Float64]()
    for b in range(N // NB):
        var xo = List[Scalar[DT]]()
        var xr = List[Scalar[DT]]()
        var xn = List[Scalar[DT]]()
        var xw = List[Scalar[DT]]()
        for k in range(NB):
            var off = (b * NB + k) * 3 * HW
            _norm_chw01(ours01, off, xo)
            _norm_chw01(real01, off, xr)
            _norm_chw01(next01, off, xn)
            _norm_chw01(swm01, off, xw)
        var eo = encode_ref["gpu", NB](enc, xo, ctx)
        var er = encode_ref["gpu", NB](enc, xr, ctx)
        var en = encode_ref["gpu", NB](enc, xn, ctx)
        var ew = encode_ref["gpu", NB](enc, xw, ctx)
        for k in range(NB):
            var g = List[Float64](length=3, fill=0.0)
            var st = 0.0
            for d in range(REF_EMB):
                var i = k * REF_EMB + d
                g[0] += (Float64(eo[i]) - Float64(er[i])) ** 2
                g[1] += (Float64(eo[i]) - Float64(ew[i])) ** 2
                g[2] += (Float64(er[i]) - Float64(ew[i])) ** 2
                st += (Float64(en[i]) - Float64(er[i])) ** 2
            step.append(sqrt(st))
            for pair in range(3):
                gap[pair].append(sqrt(g[pair]))
                ratio[pair].append(sqrt(g[pair]) / max(sqrt(st), 1e-9))

    var names: List[String] = ["ours vs dataset", "ours vs swm    ", "dataset vs swm "]
    print("  one env step in latent space ‖e(next) - e(real)‖: median", _pct(step, 0.5))
    print("  pair            | pixel MAE med / p90 (0..255) | off>30 med | ‖Δe‖ med | ‖Δe‖ / one step: med / p90")
    for pair in range(3):
        if pair > 0 and not has_swm:
            continue
        print(
            "  ", names[pair], " | ", _pct(mae[pair], 0.5), " / ", _pct(mae[pair], 0.9),
            " | ", _pct(off30[pair], 0.5), " | ", _pct(gap[pair], 0.5),
            " | ", _pct(ratio[pair], 0.5), " / ", _pct(ratio[pair], 0.9), sep="",
        )
    # GATE on ours vs the dataset (measured 2026-10-02 with render_swm.mojo:
    # pixel MAE 0.48, latent gap 0.56 of one env step; the env's own
    # renderer: 2.15 / 1.49 — it fails both). swm's own re-render from the
    # bare state sits at 0.83 step from the dataset (median): ours is at
    # that floor.
    var e_pix = _pct(mae[0], 0.5)
    var e_lat = _pct(ratio[0], 0.5)
    if e_pix > 1.0 or e_lat > 1.0:
        raise Error("FAIL G4b: renderer off the dataset (pixel MAE " + String(e_pix) + ", latent " + String(e_lat) + " steps)")
    print("PASS")
