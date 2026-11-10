# +--------------------------------------------------------------------------+ #
# | Kokoro ON THE DEVICE, against hexgrad's KModel — stage by stage
# +--------------------------------------------------------------------------+ #
"""Gates `noeira/audio/kokoro/device.mojo` on the reference dumps.

    pixi run -e apple|nvidia mojo build -I . tests/audio/test_kokoro_gpu.mojo -o build/test_kokoro_gpu
    ./build/test_kokoro_gpu [kokoro_dir] [case ...]                 (Orin: pixi run -e jetson build-jetson)

The device path against the REFERENCE again (not against our CPU path), with
the CPU gate's comparisons (`test_kokoro_reference.mojo`): PL-BERT, the
duration encoder and LSTM, the frames per phoneme (EQUAL), the text encoder,
the alignment, pitch and energy, the decoder's blocks; then the generator FROM
THE REFERENCE'S SOURCE SPECTRUM down to `conv_post` and the waveform (the
source itself is the host's, gated by the CPU gate). Errors: max |d| / max
|reference| of the stage.

Falsification (inputs only: the device copy of the weights is not edited): the
decoder's style half given as the prosody style; one frame moved between
phonemes; the source spectrum one frame late.

Then the timings: each stage on its own (synchronised), and `synthesize` end to
end, warm.

`-D KOKORO_TF32=1`: the vocoder's GEMMs in TF32 (NVIDIA), held to `TF32_BOUND`.
"""

from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.npy import load_npy_f64
from noeira.nn.core.tensor import Tensor
from noeira.audio.kokoro.device import KokoroDevice, DSeq, KD_TF32
from noeira.audio.kokoro.decoder import kk_istft, kk_harmonic_source, kk_stft
from noeira.audio.kokoro.tts import KokoroTTS, kokoro_noise
from noeira.audio.kokoro.paths import kokoro_dir


comptime STAGE_BOUND: Float64 = 1.0e-4
comptime GEN_BOUND: Float64 = 1.0e-2 if KD_TF32 else 2.0e-4
"""fp32: the CPU gate's bound. TF32: provisional, set from a measured run."""


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def rel(ref got: Tensor, ref want: List[Float64], n: Int) -> Float64:
    var worst = 0.0
    var scale = 0.0
    for i in range(n):
        var d = _abs(Float64(got.data[i]) - want[i])
        if d > worst:
            worst = d
        if _abs(want[i]) > scale:
            scale = _abs(want[i])
    return worst / scale if scale > 0.0 else 1.0e30


def rel_cf(ref got: Tensor, ref want: List[Float64], rows: Int, cols: Int) -> Float64:
    """`got [rows, cols]` against `want` stored channels-first `[cols, rows]`."""
    var worst = 0.0
    var scale = 0.0
    for r in range(rows):
        for c in range(cols):
            var wv = want[c * rows + r]
            var d = _abs(Float64(got.data[r * cols + c]) - wv)
            if d > worst:
                worst = d
            if _abs(wv) > scale:
                scale = _abs(wv)
    return worst / scale if scale > 0.0 else 1.0e30


def check(what: String, err: Float64, bound: Float64) -> Int:
    var ok = err <= bound
    print("  " + ("PASS" if ok else "FAIL") + "  " + what + "  rel err = " + String(err))
    return 0 if ok else 1


def caught(what: String, err: Float64, bound: Float64) -> Int:
    var ok = err > bound
    print("  " + ("PASS" if ok else "FAIL") + "  defect: " + what + "  rel err = " + String(err)
          + ("  (caught)" if ok else "  NOT CAUGHT"))
    return 0 if ok else 1


def load_ids(path: String) raises -> List[Int]:
    var v = load_npy_f64(path)
    var out = List[Int]()
    for i in range(len(v)):
        out.append(Int(v[i]))
    return out^


def load_f32(path: String) raises -> List[Float32]:
    var v = load_npy_f64(path)
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(Float32(v[i]))
    return out^


def ms(a: Int, b: Int) -> Float64:
    return Float64(b - a) / 1.0e6


def har_rows(ref want: List[Float64], hf: Int, shift: Int) -> List[Float32]:
    """The reference's `gen_har [22, hf]` as `[hf, 22]`, optionally `shift` frames late."""
    var out = List[Float32](length=hf * 22, fill=0.0)
    for r in range(shift, hf):
        for k in range(22):
            out[r * 22 + k] = Float32(want[k * hf + r - shift])
    return out^


def main() raises:
    var args = argv()
    var dir = String(args[1]) if len(args) >= 2 else kokoro_dir()
    var cases = List[String]()
    for i in range(2, len(args)):
        cases.append(String(args[i]))
    if len(cases) == 0:
        cases = [String("en_1"), String("en_2"), String("fr_1"), String("fr_2")]
    print("Kokoro ON THE DEVICE vs hexgrad's KModel (" + dir + ")" + ("  [TF32 vocoder]" if KD_TF32 else ""))
    var t_load = perf_counter_ns()
    var k = KokoroDevice.load(DeviceContext(), dir)
    print("  loaded onto the device in " + String(ms(t_load, perf_counter_ns())) + " ms")
    var fails = 0
    var compared = 0

    for c in range(len(cases)):
        var rd = dir + "/ref/" + cases[c]
        var ids = load_ids(rd + "/input_ids.npy")
        var ref_s = load_f32(rd + "/ref_s.npy")
        var t = len(ids)
        var taps = List[DSeq]()
        var p = k.durations(ids, ref_s, 1.0, taps)
        var want_frames = load_npy_f64(rd + "/pred_dur.npy")
        var fr = 0
        var same = len(want_frames) == t
        for i in range(t):
            fr += p[1][i]
            if same and p[1][i] != Int(want_frames[i]):
                same = False
        print("  case " + cases[c] + ": " + String(t) + " tokens, " + String(fr) + " frames")
        var worst = rel(k.download(taps[0]), load_npy_f64(rd + "/bert.npy"), t * 768)
        var e = rel_cf(k.download(taps[1]), load_npy_f64(rd + "/d_en.npy"), t, 512)
        if e > worst:
            worst = e
        e = rel(k.download(taps[2]), load_npy_f64(rd + "/dur_enc.npy"), t * 640)
        if e > worst:
            worst = e
        e = rel(k.download(taps[3]), load_npy_f64(rd + "/dur_lstm.npy"), t * 512)
        if e > worst:
            worst = e
        fails += check(String("PL-BERT, bert_encoder, duration encoder and LSTM (worst)"), worst, STAGE_BOUND)
        print("  " + ("PASS" if same else "FAIL") + "  frames per phoneme EQUAL (" + String(fr) + " frames)")
        if not same:
            fails += 1

        var a = k.acoustic(ids, ref_s, p[0], p[1], taps)
        worst = rel_cf(k.download(taps[4]), load_npy_f64(rd + "/t_en.npy"), t, 512)
        e = rel_cf(k.download(taps[5]), load_npy_f64(rd + "/en.npy"), fr, 640)
        if e > worst:
            worst = e
        e = rel_cf(k.download(a[0]), load_npy_f64(rd + "/asr.npy"), fr, 512)
        if e > worst:
            worst = e
        fails += check(String("text encoder, alignment"), worst, STAGE_BOUND)
        fails += check(String("pitch curve"), rel(k.download(a[1]), load_npy_f64(rd + "/f0.npy"), 2 * fr), STAGE_BOUND)
        fails += check(String("energy curve"), rel(k.download(a[2]), load_npy_f64(rd + "/n.npy"), 2 * fr), STAGE_BOUND)

        var x = k.decoder_blocks(a[0], a[1], a[2], ref_s, taps)
        worst = rel_cf(k.download(taps[6]), load_npy_f64(rd + "/dec_encode.npy"), fr, 1024)
        for b in range(4):
            var rows = fr if b < 3 else 2 * fr
            var cols = 1024 if b < 3 else 512
            e = rel_cf(k.download(taps[7 + b]), load_npy_f64(rd + "/dec_decode_" + String(b) + ".npy"), rows, cols)
            if e > worst:
                worst = e
        fails += check(String("decoder: the five AdaIN blocks (worst)"), worst, STAGE_BOUND)

        var hf = 120 * fr + 1
        var want_har = load_npy_f64(rd + "/gen_har.npy")
        var har = DSeq(k.upload(har_rows(want_har, hf, 0)), hf, 22)
        var post = k.generate(x^, har, ref_s, taps)
        fails += check(String("generator: upsampling 0"), rel_cf(k.download(taps[11]), load_npy_f64(rd + "/gen_ups_0.npy"), 20 * fr, 256), GEN_BOUND)
        fails += check(String("generator: stage 0"), rel_cf(k.download(taps[12]), load_npy_f64(rd + "/gen_res_0_lrelu.npy"), 20 * fr, 256), GEN_BOUND)
        fails += check(String("generator: upsampling 1"), rel_cf(k.download(taps[13]), load_npy_f64(rd + "/gen_ups_1.npy"), 120 * fr, 128), GEN_BOUND)
        fails += check(String("generator: stage 1"), rel_cf(k.download(taps[14]), load_npy_f64(rd + "/gen_res_1_lrelu.npy"), hf, 128), GEN_BOUND)
        var post_h = k.download(post)
        fails += check(String("conv_post"), rel_cf(post_h, load_npy_f64(rd + "/gen_post.npy"), hf, 22), GEN_BOUND)
        var audio = kk_istft(post_h, hf)
        var want_audio = load_npy_f64(rd + "/audio.npy")
        if len(audio) != len(want_audio):
            print("  FAIL  waveform has " + String(len(audio)) + " samples, reference " + String(len(want_audio)))
            fails += 1
        else:
            var aw = 0.0
            var ascale = 0.0
            for i in range(len(audio)):
                var d = _abs(Float64(audio[i]) - want_audio[i])
                if d > aw:
                    aw = d
                if _abs(want_audio[i]) > ascale:
                    ascale = _abs(want_audio[i])
            fails += check(String("waveform, ") + String(len(audio)) + " samples", aw / ascale, GEN_BOUND)
        compared += t

    if compared == 0:
        raise Error("gate: nothing compared — this run proved nothing")

    # ── falsification on the first case ──────────────────────────────────
    var d0 = dir + "/ref/" + cases[0]
    var ids0 = load_ids(d0 + "/input_ids.npy")
    var s0 = load_f32(d0 + "/ref_s.npy")
    var swapped = List[Float32]()
    for i in range(128):
        swapped.append(s0[i])
    for i in range(128):
        swapped.append(s0[i])
    var tp = List[DSeq]()
    var pa = k.durations(ids0, swapped, 1.0, tp)
    fails += caught(String("the decoder's style half used for prosody"),
                    rel(k.download(tp[2]), load_npy_f64(d0 + "/dur_enc.npy"), len(ids0) * 640), STAGE_BOUND)

    var tq = List[DSeq]()
    var pb = k.durations(ids0, s0, 1.0, tq)
    var moved = pb[1].copy()
    moved[1] += 1
    moved[2] -= 1
    var ab = k.acoustic(ids0, s0, pb[0], moved, tq)
    var fr0 = ab[0].rows
    fails += caught(String("one frame moved to the neighbouring phoneme"),
                    rel_cf(k.download(tq[5]), load_npy_f64(d0 + "/en.npy"), fr0, 640), STAGE_BOUND)

    var tr = List[DSeq]()
    var pc = k.durations(ids0, s0, 1.0, tr)
    var ac = k.acoustic(ids0, s0, pc[0], pc[1], tr)
    var xc = k.decoder_blocks(ac[0], ac[1], ac[2], s0, tr)
    var hf0 = 120 * ac[0].rows + 1
    var late = DSeq(k.upload(har_rows(load_npy_f64(d0 + "/gen_har.npy"), hf0, 1)), hf0, 22)
    _ = k.generate(xc^, late, s0, tr)
    fails += caught(String("the source spectrum one frame late"),
                    rel_cf(k.download(tr[12]), load_npy_f64(d0 + "/gen_res_0_lrelu.npy"), 20 * ac[0].rows, 256), GEN_BOUND)

    # ── timings, first case ───────────────────────────────────────────────
    var tt = List[DSeq]()
    k.ctx.synchronize()
    var q0 = perf_counter_ns()
    var pt = k.durations(ids0, s0, 1.0, tt)
    k.ctx.synchronize()
    var q1 = perf_counter_ns()
    var at = k.acoustic(ids0, s0, pt[0], pt[1], tt)
    k.ctx.synchronize()
    var q2 = perf_counter_ns()
    var xt = k.decoder_blocks(at[0], at[1], at[2], s0, tt)
    k.ctx.synchronize()
    var q3 = perf_counter_ns()
    var hft = 120 * at[0].rows + 1
    var hart = DSeq(k.upload(har_rows(load_npy_f64(d0 + "/gen_har.npy"), hft, 0)), hft, 22)
    var q4 = perf_counter_ns()
    var postt = k.generate(xt^, hart, s0, tt)
    k.ctx.synchronize()
    var q5 = perf_counter_ns()
    _ = k.download(postt)
    print("  stages (" + cases[0] + ", " + String(at[0].rows) + " frames): durations " + String(ms(q0, q1))
          + " ms | acoustic " + String(ms(q1, q2)) + " ms | decoder " + String(ms(q2, q3))
          + " ms | generator " + String(ms(q4, q5)) + " ms")
    # the host's share
    var h0 = perf_counter_ns()
    var f0h = k.download(at[1])
    var h1 = perf_counter_ns()
    var nz = kokoro_noise(at[0].rows * 600, 0)
    var h2 = perf_counter_ns()
    var src = kk_harmonic_source(k.w, f0h, 2 * at[0].rows, nz)
    var h3 = perf_counter_ns()
    var harh = Tensor()
    var hfh = kk_stft(src, harh)
    var h4 = perf_counter_ns()
    _ = k.upload(harh.data)
    var h5 = perf_counter_ns()
    var posth = k.download(postt)
    var h6 = perf_counter_ns()
    _ = kk_istft(posth, hfh)
    var h7 = perf_counter_ns()
    var host = KokoroTTS()
    host.dir = dir
    var h8 = perf_counter_ns()
    _ = host.style_row(String("af_heart"), 25)
    var h9 = perf_counter_ns()
    print("  host: f0 down " + String(ms(h0, h1)) + " | noise " + String(ms(h1, h2)) + " | source "
          + String(ms(h2, h3)) + " | STFT " + String(ms(h3, h4)) + " | har up " + String(ms(h4, h5))
          + " | post down " + String(ms(h5, h6)) + " | iSTFT " + String(ms(h6, h7))
          + " | voice file " + String(ms(h8, h9)) + " ms")
    var phonemes = String("aɪɐm ɡˌoʊɪŋ tə ðə fɹˈɪdʒ.")
    _ = k.synthesize(phonemes, String("af_heart"))
    var e0 = perf_counter_ns()
    var sp = k.synthesize(phonemes, String("af_heart"))
    var e1 = perf_counter_ns()
    print("  synthesize end to end, warm: " + String(ms(e0, e1)) + " ms for "
          + String(Float64(len(sp.audio)) / 24000.0) + " s of speech")

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("kokoro device gate failed")
