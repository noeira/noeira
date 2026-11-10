# +--------------------------------------------------------------------------+ #
# | The Mojo log-mel front end, against transformers' ParakeetFeatureExtractor
# +--------------------------------------------------------------------------+ #
"""Gates `noeira/audio/mel.mojo` on the reference feature extractor's output.

    pixi run -e detect python tools/parakeet/dump_parakeet_reference.py \
        --wav clip.wav --out ~/.cache/noeira/parakeet/ref/<case> [--pad-s 8]
    pixi run mojo run -I . tests/audio/test_log_mel_reference.mojo [ref_root] [case ...]

Three things are compared, because the features alone cannot see all of them:

1. **The mel filterbank, against librosa's** (`mel_filters.npy`). ⚠ The feature
   comparison is BLIND to a per-filter scale: a scale becomes an additive
   constant after the log and the per-bin mean removes it. Slaney's area
   normalization is exactly such a scale, so only this direct comparison gates it.
   Required IDENTICAL in float32, weight for weight.
2. **The window**, against `torch.hann_window(400, periodic=False)`, within one
   float32 ulp (torch's vectorized cosine; 391 of 400 identical).
3. **The features**, at ONE compiled maximum (`MAX_FRAMES`) for every case: the
   valid frames within the two bounds below, the valid count equal to the
   reference's mask, and every row beyond it exactly zero. A case dumped with
   `--pad-s` is the reference's own padded path, compared over all its rows.

Then the FALSIFICATION: four defects, each of which must push the first case
over BOTH bounds. A gate that passes any of them proves nothing.

## Two bounds, because the reference is float32

The reference computes in float32 and this code in float64, so the two agree to
the REFERENCE's rounding. That rounding is not uniform:

- a bin with signal (mel power >= 1e-4, about a third of the bins of a spoken
  command) agrees to ~2e-6: `LOUD_BOUND`;
- a near-silent bin (mel power around 3e-9, twenty times BELOW the 2^-24 guard)
  holds the float32 STFT's noise floor, the log and the division by a small
  per-bin deviation amplify it, and the same reference disagrees with ITSELF
  by 2.3e-5 between its padded and unpadded paths (`--compare-pad`). Measured
  here up to 1.1e-3: `ALL_BOUND`.

One bound for both would either fail on the reference's noise or be loose
enough to pass a defect confined to the bins that carry the speech. The
smallest defect below moves the loud bins by 2.4e-2, a thousand times
`LOUD_BOUND`.
"""

from std.os import getenv
from std.os.path import exists
from std.sys import argv

from noeira.io.npy import load_npy_f64
from noeira.audio.mel import LogMel, hann_window


comptime MAX_FRAMES: Int = 801
"""8 s at 100 frames/s, plus the frame the STFT adds: the padded dump's shape."""
comptime N_MELS: Int = 128
comptime LOUD_LOG_POWER: Float64 = -9.210340371976182
"""`log(1e-4)`: a bin at or above this mel power carries signal."""
comptime LOUD_BOUND: Float64 = 2.0e-5
comptime ALL_BOUND: Float64 = 5.0e-3
comptime WINDOW_BOUND: Float64 = 6.0e-8
"""One float32 ulp just below 1."""


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def load_f32(path: String) raises -> List[Float32]:
    var v = load_npy_f64(path)
    var out = List[Float32](capacity=len(v))
    for i in range(len(v)):
        out.append(Float32(v[i]))
    return out^


@fieldwise_init
struct FeatureError(Copyable, Movable):
    var loud: Float64
    """Largest |got - want| over the bins with signal."""
    var all: Float64
    """Largest |got - want| over every valid bin."""
    var n_loud: Int

    def within_bounds(self) -> Bool:
        return self.n_loud > 0 and self.loud <= LOUD_BOUND and self.all <= ALL_BOUND

    def beyond_both(self) -> Bool:
        return self.n_loud > 0 and self.loud > LOUD_BOUND and self.all > ALL_BOUND

    def text(self) -> String:
        return (
            "max|d| = " + String(self.loud) + " over " + String(self.n_loud)
            + " bins with signal, " + String(self.all) + " over all"
        )


def features_error(
    ref got: List[Float32],
    ref want: List[Float64],
    ref log_mel: List[Float64],
    frames: Int,
) -> FeatureError:
    """`log_mel` is the un-normalized log-mel of the run that produced `got`."""
    var e = FeatureError(0.0, 0.0, 0)
    for i in range(frames * N_MELS):
        var d = _abs(Float64(got[i]) - want[i])
        if d > e.all:
            e.all = d
        if log_mel[i] >= LOUD_LOG_POWER:
            e.n_loud += 1
            if d > e.loud:
                e.loud = d
    return e^


def main() raises:
    var args = argv()
    var root = getenv("HOME") + "/.cache/noeira/parakeet/ref"
    if len(args) >= 2:
        root = String(args[1])
    var cases = List[String]()
    for i in range(2, len(args)):
        cases.append(String(args[i]))
    if len(cases) == 0:
        cases = [String("fr_10"), String("fr_10_pad8"), String("en_16")]
    if not exists(root + "/" + cases[0] + "/features.npy"):
        raise Error(
            "no reference dump at " + root + "/" + cases[0]
            + " — generate it with tools/parakeet/dump_parakeet_reference.py"
        )

    print("Log-mel front end vs ParakeetFeatureExtractor")
    print("  reference dumps: " + root)
    var fails = 0
    var fe = LogMel[]()

    # ── 1. the filterbank ────────────────────────────────────────────────
    var want_filt = load_npy_f64(root + "/" + cases[0] + "/mel_filters.npy")
    if len(want_filt) != len(fe.filt):
        raise Error("mel_filters.npy is not [128, 257]")
    var filt_worst = 0.0
    var filt_exact = 0
    var filt_nonzero = 0
    for i in range(len(want_filt)):
        var d = _abs(fe.filt[i] - want_filt[i])
        if d > filt_worst:
            filt_worst = d
        if want_filt[i] != 0.0:
            filt_nonzero += 1
            if fe.filt[i] == want_filt[i]:
                filt_exact += 1
    var filt_ok = filt_worst == 0.0 and filt_nonzero > 0 and filt_exact == filt_nonzero
    print(
        "  " + ("PASS" if filt_ok else "FAIL") + "  mel filterbank  max|d| = "
        + String(filt_worst) + "  " + String(filt_exact) + "/"
        + String(filt_nonzero) + " non-zero weights identical in float32"
    )
    if not filt_ok:
        fails += 1

    # ── 2. the window ────────────────────────────────────────────────────
    var want_win = load_npy_f64(root + "/" + cases[0] + "/window.npy")
    var left = (512 - 400) // 2
    var win_worst = 0.0
    var win_exact = 0
    for i in range(len(want_win)):
        var d = _abs(fe.window[left + i] - want_win[i])
        if d > win_worst:
            win_worst = d
        if d == 0.0:
            win_exact += 1
    var win_ok = len(want_win) == 400 and win_worst <= WINDOW_BOUND
    print(
        "  " + ("PASS" if win_ok else "FAIL") + "  hann window     max|d| = "
        + String(win_worst) + "  " + String(win_exact) + "/400 identical in float32"
    )
    if not win_ok:
        fails += 1

    # ── 3. the features, every case at the same compiled maximum ─────────
    var got = List[Float32](length=MAX_FRAMES * N_MELS, fill=0.0)
    var compared = 0
    for c in range(len(cases)):
        var dir = root + "/" + cases[c]
        var audio = load_f32(dir + "/audio.npy")
        var want = load_npy_f64(dir + "/features.npy")
        var want_len = Int(load_npy_f64(dir + "/feature_len.npy")[0])
        var want_rows = len(want) // N_MELS
        if want_rows > MAX_FRAMES:
            raise Error(cases[c] + ": the dump is longer than MAX_FRAMES")
        var frames = fe.extract[MAX_FRAMES](audio, got)
        var err = features_error(
            got, want, fe.log_mel, frames if frames < want_len else want_len
        )
        # Beyond the valid frames: zero in ours (to MAX_FRAMES) and in theirs.
        var tail_nonzero = 0
        for i in range(frames * N_MELS, MAX_FRAMES * N_MELS):
            if got[i] != 0.0:
                tail_nonzero += 1
        for i in range(want_len * N_MELS, want_rows * N_MELS):
            if want[i] != 0.0:
                tail_nonzero += 1
        var ok = frames == want_len and err.within_bounds() and tail_nonzero == 0
        compared += frames * N_MELS
        print(
            "  " + ("PASS" if ok else "FAIL") + "  " + cases[c] + "  "
            + String(frames) + " valid frames (reference " + String(want_len)
            + " of " + String(want_rows) + ")  " + err.text()
            + "  non-zero beyond: " + String(tail_nonzero)
        )
        if not ok:
            fails += 1
    if compared == 0:
        raise Error("gate: no frames compared — this run proved nothing")

    # ── 4. falsification: each defect must fail the first case ───────────
    var dir0 = root + "/" + cases[0]
    var audio0 = load_f32(dir0 + "/audio.npy")
    var want0 = load_npy_f64(dir0 + "/features.npy")
    var len0 = Int(load_npy_f64(dir0 + "/feature_len.npy")[0])

    var d1 = LogMel[]()
    d1.preemphasis = 0.96
    _ = d1.extract[MAX_FRAMES](audio0, got)
    var e1 = features_error(got, want0, d1.log_mel, len0)

    var d2 = LogMel[]()
    var periodic = hann_window(400, True)
    for i in range(400):
        d2.window[left + i] = periodic[i]
    _ = d2.extract[MAX_FRAMES](audio0, got)
    var e2 = features_error(got, want0, d2.log_mel, len0)

    var d3 = LogMel[]()
    for m in range(N_MELS):
        var k = 256
        while k > 0:
            d3.filt[m * 257 + k] = d3.filt[m * 257 + k - 1]
            k -= 1
        d3.filt[m * 257] = 0.0
    d3.refresh_filter_ranges()
    _ = d3.extract[MAX_FRAMES](audio0, got)
    var e3 = features_error(got, want0, d3.log_mel, len0)

    var shifted = List[Float32](capacity=len(audio0))
    for i in range(1, len(audio0)):
        shifted.append(audio0[i])
    shifted.append(0.0)
    _ = fe.extract[MAX_FRAMES](shifted, got)
    var e4 = features_error(got, want0, fe.log_mel, len0)

    var names = [
        String("pre-emphasis 0.96 instead of 0.97"),
        String("periodic hann window"),
        String("filterbank moved up one FFT bin"),
        String("audio advanced by one sample"),
    ]
    var errs = [e1.copy(), e2.copy(), e3.copy(), e4.copy()]
    for i in range(4):
        var caught = errs[i].beyond_both()
        print(
            "  " + ("PASS" if caught else "FAIL") + "  defect: " + names[i]
            + "  " + errs[i].text()
            + ("  (caught)" if caught else "  NOT CAUGHT")
        )
        if not caught:
            fails += 1

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("log-mel gate failed")
