"""The speaker-embedding port against its reference, stage by stage (CPU).

    # the reference (docs/ROBOT_BRAIN_PLAN.md B4): on the machine that has the model file
    python tools/speaker/extract_wespeaker.py --onnx voxceleb_resnet34.onnx --out resnet34.safetensors
    python tools/speaker/dump_wespeaker_reference.py --onnx voxceleb_resnet34.onnx --out ref clip1.wav ...
    pixi run mojo run -I . tests/audio/test_speaker_reference.mojo [dir, default ~/.cache/noeira/local-ai/speaker] [ref folder name]

For every clip of `<dir>/ref/meta.json`:

1. THE FRONT END: `kaldi_fbank` of the clip against Kaldi's own code
   (`kaldi-native-fbank`): the same number of frames, every log energy within
   FBANK_TOL.
2. THE NETWORK, fed the REFERENCE's features (so a front-end difference cannot
   hide in it, nor excuse it): the stem, the four stages, the pooled
   statistics and the embedding against `onnxruntime` on the published file,
   each within STAGE_REL of the reference's largest value.
3. END TO END, from the samples: the embedding's cosine with the reference's
   at least E2E_COS.
Then the cosines between the clips' embeddings against the reference's own
table, and five MUTANTS, each of which must break its bound: a Hann window, no
pre-emphasis, a shortcut at the wrong stride, a deviation divided by n, the
pooled means and deviations swapped. Exits non-zero on a failure.
"""

from std.os import getenv
from std.sys import argv, exit
from std.time import perf_counter_ns

from noeira.audio.speaker import SpeakerNet, kaldi_fbank, subtract_mean, pcm_int16_range, cosine, FB_MELS
from noeira.audio.speaker.model import Plane, SPK_STAGES
from noeira.io.json import load_json
from noeira.io.npy import load_npy_f64
from noeira.io.wav import load_wav


comptime FBANK_TOL: Float64 = 2e-3
"""Log energies: Kaldi computes in float32, this port in float64."""
comptime STAGE_REL: Float64 = 1e-4
comptime E2E_COS: Float64 = 0.99999
comptime COS_TOL: Float64 = 1e-3


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok  :", what)
    else:
        print("  FAIL:", what)
        fails += 1


def rel_err(ref got: List[Float32], ref want: List[Float64]) -> Float64:
    """Largest error over the largest reference value; 1e9 when the sizes differ."""
    if len(got) != len(want):
        return 1e9
    var top = 0.0
    var err = 0.0
    for i in range(len(want)):
        top = max(top, abs(want[i]))
        err = max(err, abs(Float64(got[i]) - want[i]))
    return err / top if top > 0.0 else err


def sci(x: Float64) -> String:
    if x == 0.0:
        return String("0")
    var e = 0
    var m = abs(x)
    while m < 1.0:
        m *= 10.0
        e -= 1
    while m >= 10.0:
        m /= 10.0
        e += 1
    return String(round(m, 1)) + "e" + String(e)


def as_f32(ref x: List[Float64]) -> List[Float32]:
    var out = List[Float32](capacity=len(x))
    for v in x:
        out.append(Float32(v))
    return out^


def main() raises:
    var args = argv()
    var dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/local-ai/speaker"
    var ref_name = String(args[2]) if len(args) > 2 else String("ref")
    var rdir = dir + "/" + ref_name
    var fails = 0
    var net = SpeakerNet.load(dir + "/resnet34.safetensors")
    var meta = load_json(rdir + "/meta.json")
    var clips = meta.field(meta.root(), "clips")
    var n = meta.size(clips)
    print("speaker port vs reference:", n, "clips in", rdir)
    var embs = List[List[Float32]]()
    var first_wav = String("")
    var first_name = String("")
    for c in range(n):
        var clip = meta.at(clips, c)
        var name = meta.string(meta.field(clip, "name"))
        var wav_path = meta.string(meta.field(clip, "wav"))
        if not wav_path.startswith("/"):
            wav_path = rdir + "/" + wav_path
        if c == 0:
            first_wav = wav_path.copy()
            first_name = name.copy()
        var want_frames = meta.integer(meta.field(clip, "frames"))
        print(name + " (" + String(want_frames) + " frames)")
        # 1. the front end
        var wav = load_wav(wav_path)
        var feats = kaldi_fbank(pcm_int16_range(wav.samples))
        var ref_feats = load_npy_f64(rdir + "/" + name + ".feats.npy")
        var worst = 1e9
        if len(feats) == len(ref_feats):
            worst = 0.0
            for i in range(len(feats)):
                worst = max(worst, abs(Float64(feats[i]) - ref_feats[i]))
        check(fails, len(feats) == want_frames * FB_MELS and worst <= FBANK_TOL, "filterbank: " + String(len(feats) // FB_MELS)
              + " frames, largest difference " + sci(worst))
        # 2. the network on the reference's features
        var rf = as_f32(ref_feats)
        subtract_mean(rf)
        var t0 = perf_counter_ns()
        var x = net.stem_out(rf)
        var e_stem = rel_err(x.v, load_npy_f64(rdir + "/" + name + ".stem.npy"))
        var e_stage = List[Float64]()
        for s in range(SPK_STAGES):
            x = net.stage(x^, s)
            e_stage.append(rel_err(x.v, load_npy_f64(rdir + "/" + name + ".s" + String(s + 1) + ".npy")))
        var pooled = net.pool(x)
        var e_pool = rel_err(pooled, load_npy_f64(rdir + "/" + name + ".pooled.npy"))
        var emb = net.seg(pooled)
        var ref_emb = load_npy_f64(rdir + "/" + name + ".emb.npy")
        var e_emb = rel_err(emb, ref_emb)
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        var worst_stage = max(e_stem, max(max(e_stage[0], e_stage[1]), max(e_stage[2], e_stage[3])))
        check(fails, worst_stage <= STAGE_REL and e_pool <= STAGE_REL and e_emb <= STAGE_REL,
              "network on the reference's features: stem " + sci(e_stem) + ", stages " + sci(e_stage[0]) + " " + sci(e_stage[1]) + " "
              + sci(e_stage[2]) + " " + sci(e_stage[3]) + ", pooled " + sci(e_pool) + ", embedding " + sci(e_emb) + "  (" + String(Int(ms)) + " ms)")
        # 3. end to end, from the samples
        subtract_mean(feats)
        var mine = net.embed(feats)
        var ref32 = as_f32(ref_emb)
        var cs = cosine(mine, ref32)
        check(fails, cs >= E2E_COS, "end to end: cosine with the reference's embedding " + String(round(cs, 7)))
        embs.append(mine^)
    # the clips against each other, as the reference has them
    var table = meta.field(meta.root(), "cosine")
    var worst_cos = 0.0
    for i in range(n):
        var row = meta.at(table, i)
        for j in range(n):
            worst_cos = max(worst_cos, abs(cosine(embs[i], embs[j]) - meta.number(meta.at(row, j))))
    check(fails, worst_cos <= COS_TOL, "the clips' cosines against the reference's table: largest difference " + sci(worst_cos))

    print("mutants, on " + first_name + " (each must break its bound)")
    var wav = load_wav(first_wav)
    var x16 = pcm_int16_range(wav.samples)
    var ref_feats = load_npy_f64(rdir + "/" + first_name + ".feats.npy")
    var ref_emb = load_npy_f64(rdir + "/" + first_name + ".emb.npy")
    var hann = kaldi_fbank(x16, 0.97, False)
    var no_pre = kaldi_fbank(x16, 0.0, True)
    var d_hann = 0.0
    var d_pre = 0.0
    for i in range(len(ref_feats)):
        d_hann = max(d_hann, abs(Float64(hann[i]) - ref_feats[i]))
        d_pre = max(d_pre, abs(Float64(no_pre[i]) - ref_feats[i]))
    check(fails, d_hann > FBANK_TOL, "a Hann window: " + sci(d_hann))
    check(fails, d_pre > FBANK_TOL, "no pre-emphasis: " + sci(d_pre))
    var rf = as_f32(ref_feats)
    subtract_mean(rf)
    var bits: List[Int] = [1, 2, 4]
    var names: List[String] = ["a shortcut at the wrong stride", "a deviation divided by n", "the means and deviations swapped"]
    for k in range(3):
        net.debug = bits[k]
        var e = rel_err(net.embed(rf), ref_emb)
        check(fails, e > STAGE_REL, names[k] + ": embedding " + sci(e))
    net.debug = 0

    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)
