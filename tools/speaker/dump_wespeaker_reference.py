"""Reference values for noeira's speaker-embedding port (WeSpeaker ResNet34), stage by stage.

    PYTHONPATH=<deps> python tools/speaker/dump_wespeaker_reference.py \\
        --onnx voxceleb_resnet34.onnx --out ref clip1.wav clip2.wav ...

For each 16 kHz mono 16-bit clip, WeSpeaker's own recipe (`wespeaker/cli/speaker.py`): Kaldi filterbank
features (80 bins, 25 ms frames every 10 ms, a Hamming window, no dither, on samples in the int16 range),
the clip's mean subtracted from every frame, then the network. The features come from
`kaldi-native-fbank` (Kaldi's own code), the network from `onnxruntime` on the published file, with the
graph's inner tensors exposed as extra outputs:

    <clip>.feats.npy    [T, 80]   filterbank features, BEFORE the mean is subtracted
    <clip>.stem.npy     [32, 80, T]        after the first convolution and its ReLU
    <clip>.s1.npy .. .s4.npy                the output of each of the four stages
    <clip>.pooled.npy   [5120]    the mean and the standard deviation over time, as the linear layer takes them
    <clip>.emb.npy      [256]     the embedding
    meta.json           the clips, their frame counts, and every pair's cosine

A stage's output is the ReLU after the last residual addition of that stage, found in the graph by its
structure (the 16 additions that feed a ReLU are the 16 blocks; stages end at blocks 3, 7, 13, 16).
"""

import argparse
import json
import os
import wave

import kaldi_native_fbank as knf
import numpy as np
import onnx
import onnxruntime as ort


def read_wav(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)  # the int16 range


def fbank(samples):
    o = knf.FbankOptions()
    o.frame_opts.samp_freq = 16000
    o.frame_opts.dither = 0.0
    o.frame_opts.window_type = "hamming"
    o.frame_opts.frame_length_ms = 25.0
    o.frame_opts.frame_shift_ms = 10.0
    o.frame_opts.snip_edges = True
    o.mel_opts.num_bins = 80
    f = knf.OnlineFbank(o)
    f.accept_waveform(16000, samples.tolist())
    f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0.0, help="keep only the first seconds of each clip's speech (0 = all)")
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = onnx.load(a.onnx)
    g = m.graph
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    relus = [n for n in g.node if n.op_type == "Relu"]
    stem = relus[0].output[0]
    block_out = []
    for n in g.node:
        if n.op_type == "Add":
            nxt = consumers.get(n.output[0], [])
            if len(nxt) == 1 and nxt[0].op_type == "Relu":
                block_out.append(nxt[0].output[0])
    assert len(block_out) == 16, len(block_out)
    stages = [block_out[2], block_out[6], block_out[12], block_out[15]]
    pooled = [n for n in g.node if n.op_type == "Gemm"][0].input[0]
    extra = [stem] + stages + [pooled]
    for name in extra:
        g.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])

    meta = {"clips": []}
    embs = []
    for path in a.wavs:
        name = os.path.splitext(os.path.basename(path))[0]
        x = read_wav(path)
        if a.seconds > 0:
            # the clip's speech: from the first sample above 2 % of full scale
            loud = np.nonzero(np.abs(x) > 655)[0]
            start = int(loud[0]) if len(loud) else 0
            x = x[start:start + int(a.seconds * 16000)]
            name += "_%gs" % a.seconds
            with wave.open(os.path.join(a.out, name + ".wav"), "w") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.astype("<i2").tobytes())
        feats = fbank(x)
        cmn = feats - feats.mean(axis=0, keepdims=True)
        outs = sess.run(["embs"] + extra, {"feats": cmn[None]})
        emb, st, s1, s2, s3, s4, pl = [o[0] for o in outs]
        np.save(os.path.join(a.out, name + ".feats.npy"), feats)
        np.save(os.path.join(a.out, name + ".stem.npy"), st)
        for k, s in enumerate((s1, s2, s3, s4)):
            np.save(os.path.join(a.out, name + ".s%d.npy" % (k + 1)), s)
        np.save(os.path.join(a.out, name + ".pooled.npy"), pl)
        np.save(os.path.join(a.out, name + ".emb.npy"), emb)
        embs.append(emb)
        meta["clips"].append({"name": name, "samples": int(len(x)), "frames": int(feats.shape[0]),
                              "wav": name + ".wav" if a.seconds > 0 else os.path.abspath(path),
                              "s4_shape": list(s4.shape), "emb_norm": float(np.linalg.norm(emb))})
        print(name, "samples", len(x), "frames", feats.shape[0], "stem", st.shape, "s4", s4.shape, "|emb|", round(float(np.linalg.norm(emb)), 4))
    cos = [[float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))) for q in embs] for p in embs]
    meta["cosine"] = cos
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("cosines:")
    for row in cos:
        print("  ", " ".join("%.3f" % c for c in row))


if __name__ == "__main__":
    main()
