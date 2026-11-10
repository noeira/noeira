# +--------------------------------------------------------------------------+ #
# | The Mojo Kokoro, against hexgrad's KModel — stage by stage
# +--------------------------------------------------------------------------+ #
"""Gates `noeira/audio/kokoro/` on the reference dump.

    pixi run -e detect python tools/kokoro/convert_kokoro_checkpoint.py
    pixi run -e detect python tools/kokoro/dump_kokoro_reference.py
    pixi run mojo run -I . tests/audio/test_kokoro_reference.mojo [kokoro_dir] [case ...]

So far (docs/NN_MODEL_PORTS_PLAN.md §11.5, steps 2 to 4): from `input_ids`
and the voice's style row, the PL-BERT output, `bert_encoder`, the duration
encoder, the duration LSTM and the durations; then the text encoder, both
encoders repeated per frame by the alignment, and the pitch and energy curves;
then (step 5) the decoder's five blocks, and the vocoder in three separate gates:

- the HARMONIC SOURCE, fed the reference's recorded noise (`noise_0`,
  `noise_1`), within `SOURCE_BOUND` of the reference's. That bound is the
  REFERENCE's own float32 rounding (a phase of thousands of radians in float32;
  a float64 evaluation of the same formulas differs from it by 6e-3 of its
  peak) — no implementation can be held closer;
- the STFT, applied to the REFERENCE's source: magnitudes, and phases where
  the bin is not empty, tight;
- the GENERATOR from the REFERENCE's source spectrum (`gen_har`): both
  upsampling stages, `conv_post` and the waveform, tight (`GEN_BOUND`). The
  vocoder amplifies a source difference (its output goes through an `exp`), so
  only from the same source is it comparable.

The port's own chain (own source into own generator) is then reported against
the reference waveform as a correlation: a sanity figure, not a gate. Errors are the largest
|ours - theirs| over the largest |theirs| of the stage, bounded by
`STAGE_BOUND` (both sides float32, operations in a different order); the frames
per phoneme must be EQUAL.

Falsification, on the weights and the inputs (no switch in the code under
test): ALBERT's query and key weights exchanged; an LSTM's two directions
exchanged; the decoder's half of the style row given as the prosody style;
the pitch and energy heads' output convolutions exchanged; one frame moved from
a phoneme to its neighbour; a residual block's two Snake parameters exchanged;
the source spectrum delayed by one frame. Each must push its stage over its
bound.
"""

from std.sys import argv

from noeira.io.npy import load_npy_f64
from noeira.nn.core.tensor import Tensor
from noeira.audio.kokoro.paths import kokoro_dir
from std.math import sqrt

from noeira.audio.kokoro.decoder import (
    kk_decode, kk_decoder_blocks, kk_generate, kk_harmonic_source, kk_stft,
    KokoroNoise, KokoroDecoded,
)
from noeira.audio.kokoro.prosody import (
    kk_durations, kk_acoustic, KokoroProsody, KokoroAcoustic,
)
from noeira.audio.kokoro.weights import KokoroWeights


comptime STAGE_BOUND: Float64 = 1.0e-4
comptime SOURCE_BOUND: Float64 = 2.0e-2
"""Of the source's peak. Not ours to tighten: see the module docstring."""
comptime GEN_BOUND: Float64 = 2.0e-4
"""The generator from the reference's source spectrum, ~50 convolutions deep.
Measured: 3.5e-6 on its stages, 2.2e-5 on the waveform (which goes through an
`exp`); the smaller generator defect below is 4.7e-3."""


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def rel_error(ref got: Tensor, ref want: List[Float64], n: Int) -> Float64:
    """Largest |got - want| over the first `n` values, over the largest |want|."""
    var worst = 0.0
    var scale = 0.0
    for i in range(n):
        var d = _abs(Float64(got.data[i]) - want[i])
        if d > worst:
            worst = d
        var a = _abs(want[i])
        if a > scale:
            scale = a
    return worst / scale if scale > 0.0 else 1.0e30


def rel_error_cf(ref got: Tensor, ref want: List[Float64], rows: Int, cols: Int) -> Float64:
    """`got` is `[rows, cols]`; `want` is the same data channels-first, `[cols, rows]`."""
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


def rel_error_part(
    ref got: Tensor, ref want: List[Float64], rows: Int, cols: Int, c0: Int, c1: Int,
) -> Float64:
    """`rel_error_cf` over columns `[c0, c1)` only."""
    var worst = 0.0
    var scale = 0.0
    for r in range(rows):
        for c in range(c0, c1):
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


def caught(what: String, err: Float64) -> Int:
    var ok = err > STAGE_BOUND
    print(
        "  " + ("PASS" if ok else "FAIL") + "  defect: " + what + "  rel err = "
        + String(err) + ("  (caught)" if ok else "  NOT CAUGHT")
    )
    return 0 if ok else 1


def caught_over(what: String, err: Float64, bound: Float64) -> Int:
    var ok = err > bound
    print(
        "  " + ("PASS" if ok else "FAIL") + "  defect: " + what + "  rel err = "
        + String(err) + ("  (caught)" if ok else "  NOT CAUGHT")
    )
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


def swap_tensors(mut w: KokoroWeights, a: String, b: String) raises:
    var i = w.at(a)
    var j = w.at(b)
    for k in range(w.numel(i)):
        var t = w.tensors[i].data[k]
        w.tensors[i].data[k] = w.tensors[j].data[k]
        w.tensors[j].data[k] = t


def main() raises:
    var args = argv()
    var dir = String(args[1]) if len(args) >= 2 else kokoro_dir()
    var cases = List[String]()
    for i in range(2, len(args)):
        cases.append(String(args[i]))
    if len(cases) == 0:
        cases = [String("en_1"), String("en_2"), String("fr_1"), String("fr_2")]
    print("Kokoro vs hexgrad's KModel (" + dir + ")")
    var w = KokoroWeights.load(dir + "/kokoro-v1_0.safetensors")
    var fails = 0
    var compared = 0

    for c in range(len(cases)):
        var ref_dir = dir + "/ref/" + cases[c]
        var ids = load_ids(ref_dir + "/input_ids.npy")
        var ref_s = load_f32(ref_dir + "/ref_s.npy")
        var t = len(ids)
        var p = kk_durations(w, ids, ref_s)
        print("  case " + cases[c] + ": " + String(t) + " tokens, " + String(p.frames) + " frames")
        fails += check(String("PL-BERT"), rel_error(p.bert, load_npy_f64(ref_dir + "/bert.npy"), t * 768), STAGE_BOUND)
        fails += check(String("bert_encoder"), rel_error_cf(p.d_en, load_npy_f64(ref_dir + "/d_en.npy"), t, 512), STAGE_BOUND)
        fails += check(String("duration encoder"), rel_error(p.dur_enc, load_npy_f64(ref_dir + "/dur_enc.npy"), t * 640), STAGE_BOUND)
        fails += check(String("duration LSTM"), rel_error(p.dur_lstm, load_npy_f64(ref_dir + "/dur_lstm.npy"), t * 512), STAGE_BOUND)
        var want_dur = load_npy_f64(ref_dir + "/duration.npy")
        var want_frames = load_npy_f64(ref_dir + "/pred_dur.npy")
        var dur_worst = 0.0
        var same = len(want_frames) == t
        for i in range(t):
            var d = _abs(Float64(p.duration[i]) - want_dur[i])
            if d > dur_worst:
                dur_worst = d
            if same and p.pred_dur[i] != Int(want_frames[i]):
                same = False
        fails += check(String("durations before rounding (max |d|, frames)"), dur_worst, 1.0e-3)
        print("  " + ("PASS" if same else "FAIL") + "  frames per phoneme EQUAL (" + String(p.frames) + " frames)")
        if not same:
            fails += 1
        compared += t

        var a = kk_acoustic(w, ids, ref_s, p)
        var fr = p.frames
        fails += check(String("text encoder"), rel_error_cf(a.t_en, load_npy_f64(ref_dir + "/t_en.npy"), t, 512), STAGE_BOUND)
        fails += check(String("duration encoder per frame"), rel_error_cf(a.en, load_npy_f64(ref_dir + "/en.npy"), fr, 640), STAGE_BOUND)
        fails += check(String("text encoder per frame"), rel_error_cf(a.asr, load_npy_f64(ref_dir + "/asr.npy"), fr, 512), STAGE_BOUND)
        fails += check(String("pitch curve"), rel_error(a.f0, load_npy_f64(ref_dir + "/f0.npy"), 2 * fr), STAGE_BOUND)
        fails += check(String("energy curve"), rel_error(a.n, load_npy_f64(ref_dir + "/n.npy"), 2 * fr), STAGE_BOUND)

        var noise = KokoroNoise()
        noise.phase = load_f32(ref_dir + "/noise_0.npy")
        noise.sines = load_f32(ref_dir + "/noise_1.npy")
        var hf = 120 * fr + 1

        # the decoder's blocks
        var dec = KokoroDecoded()
        var x = kk_decoder_blocks(w, a, ref_s, dec)
        var blocks_worst = rel_error_cf(dec.encode, load_npy_f64(ref_dir + "/dec_encode.npy"), fr, 1024)
        for b in range(4):
            var rows = fr if b < 3 else 2 * fr
            var cols = 1024 if b < 3 else 512
            var e = rel_error_cf(dec.decode[b], load_npy_f64(ref_dir + "/dec_decode_" + String(b) + ".npy"), rows, cols)
            if e > blocks_worst:
                blocks_worst = e
        fails += check(String("decoder: the five AdaIN blocks (worst)"), blocks_worst, STAGE_BOUND)

        # the harmonic source, with the reference's noise
        var want_src = load_npy_f64(ref_dir + "/gen_source.npy")
        var src = kk_harmonic_source(w, a.f0, 2 * fr, noise)
        var sw = 0.0
        var speak = 0.0
        for i in range(len(want_src)):
            var dd = _abs(Float64(src[i]) - want_src[i])
            if dd > sw:
                sw = dd
            if _abs(want_src[i]) > speak:
                speak = _abs(want_src[i])
        fails += check(String("harmonic source (bounded by the reference's own rounding)"), sw / speak, SOURCE_BOUND)

        # the STFT, on the reference's source
        var want_har = load_npy_f64(ref_dir + "/gen_har.npy")
        var ref_src = List[Float32]()
        for i in range(len(want_src)):
            ref_src.append(Float32(want_src[i]))
        var st = Tensor()
        var st_frames = kk_stft(ref_src, st)
        if st_frames != hf:
            print("  FAIL  STFT frames " + String(st_frames) + ", reference " + String(hf))
            fails += 1
        fails += check(String("STFT magnitudes"), rel_error_part(st, want_har, hf, 22, 0, 11), STAGE_BOUND)
        var mag_peak = 0.0
        for i in range(11 * hf):
            if want_har[i] > mag_peak:
                mag_peak = want_har[i]
        var ph_worst = 0.0
        for r in range(hf):
            for k in range(11):
                if want_har[k * hf + r] < 1.0e-2 * mag_peak:
                    continue  # an empty bin has no phase to speak of
                var dp = _abs(Float64(st.data[r * 22 + 11 + k]) - want_har[(11 + k) * hf + r])
                if dp > 3.141592653589793:
                    dp = 6.283185307179586 - dp
                if dp > ph_worst:
                    ph_worst = dp
        fails += check(String("STFT phases where the bin is not empty (max |d|, rad)"), ph_worst, 1.0e-3)

        # the generator, from the REFERENCE's source spectrum
        var har = Tensor.alloc(hf * 22)
        for r in range(hf):
            for k in range(22):
                har.data[r * 22 + k] = Float32(want_har[k * hf + r])
        kk_generate(w, dec, x^, ref_s, har, hf)
        fails += check(String("upsampling 0"), rel_error_cf(dec.ups[0], load_npy_f64(ref_dir + "/gen_ups_0.npy"), 20 * fr, 256), GEN_BOUND)
        fails += check(String("stage 0"), rel_error_cf(dec.res[0], load_npy_f64(ref_dir + "/gen_res_0_lrelu.npy"), 20 * fr, 256), GEN_BOUND)
        fails += check(String("upsampling 1"), rel_error_cf(dec.ups[1], load_npy_f64(ref_dir + "/gen_ups_1.npy"), 120 * fr, 128), GEN_BOUND)
        fails += check(String("stage 1"), rel_error_cf(dec.res[1], load_npy_f64(ref_dir + "/gen_res_1_lrelu.npy"), hf, 128), GEN_BOUND)
        fails += check(String("conv_post"), rel_error_cf(dec.post, load_npy_f64(ref_dir + "/gen_post.npy"), hf, 22), GEN_BOUND)
        var want_audio = load_npy_f64(ref_dir + "/audio.npy")
        var n_audio = len(want_audio)
        if len(dec.audio) != n_audio:
            print("  FAIL  waveform has " + String(len(dec.audio)) + " samples, reference " + String(n_audio))
            fails += 1
        else:
            var aw = 0.0
            var ascale = 0.0
            for i in range(n_audio):
                var dd = _abs(Float64(dec.audio[i]) - want_audio[i])
                if dd > aw:
                    aw = dd
                if _abs(want_audio[i]) > ascale:
                    ascale = _abs(want_audio[i])
            fails += check(String("waveform, ") + String(n_audio) + " samples", aw / ascale, GEN_BOUND)

        # the port's own chain: a sanity figure, not a gate
        var own = kk_decode(w, a, ref_s, noise)
        var sxy = 0.0
        var sxx = 0.0
        var syy = 0.0
        for i in range(n_audio):
            var u = Float64(own.audio[i])
            sxy += u * want_audio[i]
            sxx += u * u
            syy += want_audio[i] * want_audio[i]
        print("        own source into own generator: correlation with the reference waveform "
              + String(sxy / sqrt(sxx * syy)))

    if compared == 0:
        raise Error("gate: nothing compared — this run proved nothing")

    # ── falsification on the first case ──────────────────────────────────
    var d0 = dir + "/ref/" + cases[0]
    var ids0 = load_ids(d0 + "/input_ids.npy")
    var s0 = load_f32(d0 + "/ref_s.npy")
    var t0 = len(ids0)
    var bert0 = load_npy_f64(d0 + "/bert.npy")
    var enc0 = load_npy_f64(d0 + "/dur_enc.npy")
    var layer = String("bert.encoder.albert_layer_groups.0.albert_layers.0.attention.")

    swap_tensors(w, layer + "query.weight", layer + "key.weight")
    var pa = kk_durations(w, ids0, s0)
    fails += caught(String("ALBERT query and key weights exchanged"), rel_error(pa.bert, bert0, t0 * 768))
    swap_tensors(w, layer + "query.weight", layer + "key.weight")

    var l0 = String("predictor.text_encoder.lstms.0.")
    var parts: List[String] = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]
    for k in range(4):
        swap_tensors(w, l0 + parts[k], l0 + parts[k] + "_reverse")
    var pb = kk_durations(w, ids0, s0)
    fails += caught(String("an LSTM's two directions exchanged"), rel_error(pb.dur_enc, enc0, t0 * 640))
    for k in range(4):
        swap_tensors(w, l0 + parts[k], l0 + parts[k] + "_reverse")

    var s_swapped = List[Float32]()
    for i in range(128):
        s_swapped.append(s0[i])
    for i in range(128):
        s_swapped.append(s0[i])
    var pc = kk_durations(w, ids0, s_swapped)
    fails += caught(String("the decoder's style half used for prosody"), rel_error(pc.dur_enc, enc0, t0 * 640))

    var pd = kk_durations(w, ids0, s0)
    fails += check(String("first case after the restores"), rel_error(pd.dur_enc, enc0, t0 * 640), STAGE_BOUND)

    var f0_ref = load_npy_f64(d0 + "/f0.npy")
    var en_ref = load_npy_f64(d0 + "/en.npy")
    swap_tensors(w, String("predictor.F0_proj.weight"), String("predictor.N_proj.weight"))
    var ae = kk_acoustic(w, ids0, s0, pd)
    fails += caught(String("pitch and energy output convolutions exchanged"), rel_error(ae.f0, f0_ref, 2 * pd.frames))
    swap_tensors(w, String("predictor.F0_proj.weight"), String("predictor.N_proj.weight"))

    pd.pred_dur[1] += 1
    pd.pred_dur[2] -= 1
    var af = kk_acoustic(w, ids0, s0, pd)
    fails += caught(String("one frame moved to the neighbouring phoneme"), rel_error_cf(af.en, en_ref, pd.frames, 640))
    pd.pred_dur[1] -= 1
    pd.pred_dur[2] += 1

    var ag = kk_acoustic(w, ids0, s0, pd)
    fails += check(String("pitch curve after the restores"), rel_error(ag.f0, f0_ref, 2 * pd.frames), STAGE_BOUND)

    # ── generator defects, from the reference's source spectrum ──────────
    var fr0 = pd.frames
    var hf0 = 120 * fr0 + 1
    var har_ref = load_npy_f64(d0 + "/gen_har.npy")
    var res0_ref = load_npy_f64(d0 + "/gen_res_0_lrelu.npy")
    var har0 = Tensor.alloc(hf0 * 22)
    for r in range(hf0):
        for k in range(22):
            har0.data[r * 22 + k] = Float32(har_ref[k * hf0 + r])

    var blk = String("decoder.generator.noise_res.0.")
    swap_tensors(w, blk + "alpha1.0", blk + "alpha2.0")
    var dh = KokoroDecoded()
    var xh = kk_decoder_blocks(w, ag, s0, dh)
    kk_generate(w, dh, xh^, s0, har0, hf0)
    fails += caught_over(String("a residual block's two Snake parameters exchanged"),
                         rel_error_cf(dh.res[0], res0_ref, 20 * fr0, 256), GEN_BOUND)
    swap_tensors(w, blk + "alpha1.0", blk + "alpha2.0")

    var har_late = Tensor.alloc(hf0 * 22)
    for r in range(1, hf0):
        for k in range(22):
            har_late.data[r * 22 + k] = har0.data[(r - 1) * 22 + k]
    var di = KokoroDecoded()
    var xi = kk_decoder_blocks(w, ag, s0, di)
    kk_generate(w, di, xi^, s0, har_late, hf0)
    fails += caught_over(String("the source spectrum delayed by one frame"),
                         rel_error_cf(di.res[0], res0_ref, 20 * fr0, 256), GEN_BOUND)

    var dj = KokoroDecoded()
    var xj = kk_decoder_blocks(w, ag, s0, dj)
    kk_generate(w, dj, xj^, s0, har0, hf0)
    fails += check(String("stage 0 after the restores"), rel_error_cf(dj.res[0], res0_ref, 20 * fr0, 256), GEN_BOUND)

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("kokoro gate failed")
