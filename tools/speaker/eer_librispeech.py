"""How well the speaker model separates real voices, by how much speech it is given.

    PYTHONPATH=<deps> python tools/speaker/eer_librispeech.py --onnx voxceleb_resnet34.onnx \\
        --data LibriSpeech/test-clean --out eer.json [--utts 8]

LibriSpeech test-clean (40 speakers, read English, https://www.openslr.org/12, CC BY 4.0). For each
speaker, `--utts` utterances of at least 11 s; from each, the first 1, 2, 3, 5 and 10 seconds after
the speech starts. Every cut goes through WeSpeaker's recipe (`dump_wespeaker_reference.py`: Kaldi
features, mean subtracted, the published ONNX file) with `onnxruntime`: noeira's port equals it to
1e-6 (`tests/audio/test_speaker_reference.mojo`), so these numbers are the port's.

A TRIAL is two cuts from two DIFFERENT utterances: the same speaker (target) or two speakers
(non-target), scored by the cosine of their embeddings. Printed and written per pair of durations
(enrolment, test): the equal error rate, the rate of targets refused when 1 % of non-targets are
accepted, that threshold, and the two score distributions' means.

What it does not say: French, a far-field microphone, a noisy room. It ranks durations and gives a
first threshold; the robot's own microphone and its owner's voice have the last word.
"""

import argparse
import glob
import json
import os
import subprocess

import numpy as np
import onnxruntime as ort

from dump_wespeaker_reference import fbank

DURS = [1, 2, 3, 5, 10]


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32)


def eer(tar, non):
    scores = np.concatenate([tar, non])
    labels = np.concatenate([np.ones(len(tar)), np.zeros(len(non))])
    order = np.argsort(-scores)
    labels = labels[order]
    fa = np.cumsum(1 - labels) / len(non)      # non-targets accepted, threshold going down
    fr = 1 - np.cumsum(labels) / len(tar)      # targets refused
    k = int(np.argmin(np.abs(fa - fr)))
    k1 = int(np.searchsorted(fa, 0.01))
    k1 = min(k1, len(fa) - 1)
    return float((fa[k] + fr[k]) / 2), float(fr[k1]), float(scores[order][k1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--utts", type=int, default=8)
    a = ap.parse_args()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 6
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    speakers = sorted(os.listdir(a.data))
    emb = {d: [] for d in DURS}       # per duration: rows (speaker index, utterance index, vector)
    for si, spk in enumerate(speakers):
        files = sorted(glob.glob(os.path.join(a.data, spk, "*", "*.flac")))
        kept = 0
        for path in files:
            if kept >= a.utts:
                break
            x = decode(path)
            loud = np.nonzero(np.abs(x) > 655)[0]          # 2 % of full scale
            start = int(loud[0]) if len(loud) else 0
            if len(x) - start < 11 * 16000:
                continue
            for d in DURS:
                f = fbank(x[start:start + d * 16000])
                f = f - f.mean(axis=0, keepdims=True)
                e = sess.run(["embs"], {"feats": f[None]})[0][0]
                emb[d].append((si, kept, e / np.linalg.norm(e)))
            kept += 1
        print(spk, kept, "utterances", flush=True)
    result = {"speakers": len(speakers), "utterances_per_speaker": a.utts, "pairs": []}
    pairs = [(d, d) for d in DURS] + [(3, 1), (3, 2), (10, 1), (10, 2), (10, 3)]
    for enrol, test in pairs:
        A, B = emb[enrol], emb[test]
        ea = np.stack([r[2] for r in A])
        eb = np.stack([r[2] for r in B])
        s = ea @ eb.T
        sa = np.array([r[0] for r in A])[:, None]
        ua = np.array([r[1] for r in A])[:, None]
        sb = np.array([r[0] for r in B])[None, :]
        ub = np.array([r[1] for r in B])[None, :]
        tar = s[(sa == sb) & (ua != ub)]
        non = s[sa != sb]
        e, fr1, th1 = eer(tar, non)
        row = {"enrol_s": enrol, "test_s": test, "targets": int(len(tar)), "non_targets": int(len(non)), "eer": e,
               "refused_at_1pct_accepted": fr1, "threshold_at_1pct": th1, "target_mean": float(tar.mean()),
               "non_target_mean": float(non.mean()), "target_p05": float(np.percentile(tar, 5)),
               "non_target_p99": float(np.percentile(non, 99))}
        result["pairs"].append(row)
        print("enrol %2d s, test %2d s: EER %5.2f %%, refused at 1 %% accepted %5.2f %% (threshold %.3f); target mean %.3f, non-target mean %.3f, p99 %.3f"
              % (enrol, test, 100 * e, 100 * fr1, th1, row["target_mean"], row["non_target_mean"], row["non_target_p99"]), flush=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
