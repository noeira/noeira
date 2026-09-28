"""Widen a PPO family run's observation by EXTRA trailing words, keeping its policy.

    pixi run python tools/tasks/pad_ppo_obs.py SRC_RUN_DIR DST_RUN_DIR EXTRA

`--init` reads `checkpoints/last.ckpt` and `obs_norm.txt` from a run dir. A
driver built with `-D TASK_PPO_ACT_HIST=K` appends the last K executed
actions (K x 6 words) to the env's observation, so its first layers are
EXTRA inputs wider than a run trained without it. This writes DST_RUN_DIR
with:

* `actor.0.weight` / `critic.0.weight` ([IN, OUT] row-major, `Linear`'s
  layout) grown by EXTRA ZERO ROWS — the new inputs start with no say, so
  the widened net computes exactly what the source did;
* `obs_norm.txt` grown by EXTRA words of mean 0, var 1 (the running
  statistics take them over from there).

Everything else (the other layers, log-std, the `K` scalars) is copied byte
for byte. ⚠ Gate it: a `--steps 0` greedy eval of DST under the widened
build must reproduce SRC's eval under the plain build.
"""
import os
import shutil
import struct
import sys


def pad_ckpt(src, dst, extra):
    b = open(src, "rb").read()
    i = b.index(b"\n") + 1
    out = bytearray(b[:i])
    widened = []
    while i < len(b):
        start = i
        j = b.index(b"\n", i)
        hdr = b[i:j].decode()
        t = hdr.split()
        if t[0] not in ("P", "S"):
            out += b[i:]
            break
        n = int(t[2])
        reps = 3 if len(t) > 3 and t[3] == "1" else 1
        body = b[j + 1: j + 1 + n * 4 * reps]
        i = j + 1 + n * 4 * reps
        if t[0] == "P" and t[1] in ("actor.0.weight", "critic.0.weight"):
            if reps != 1:
                raise SystemExit(t[1] + ": has optimizer moments; pad them too")
            # the first layer's OUT is its bias's size, the section after it
            k = b.index(b"\n", i)
            bh = b[i:k].decode().split()
            out_dim = int(bh[2])
            if n % out_dim:
                raise SystemExit(t[1] + ": " + str(n) + " not a multiple of " + str(out_dim))
            n2 = n + extra * out_dim
            t[2] = str(n2)
            out += (" ".join(t) + "\n").encode()
            out += body + bytes(extra * out_dim * 4)
            widened.append((t[1], n // out_dim, n2 // out_dim, out_dim))
        else:
            out += b[start:i]
    with open(dst, "wb") as f:
        f.write(out)
    return widened


def pad_norm(src, dst, extra):
    lines = open(src).read().split("\n")
    m = lines[1].split(" ")
    v = lines[2].split(" ")
    m += ["0.0"] * extra
    v += ["1.0"] * extra
    with open(dst, "w") as f:
        f.write(lines[0] + "\n" + " ".join(m) + "\n" + " ".join(v) + "\n")
    return len(m) - 1


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    src, dst, extra = sys.argv[1], sys.argv[2], int(sys.argv[3])
    os.makedirs(dst + "/checkpoints", exist_ok=True)
    for w in pad_ckpt(src + "/checkpoints/last.ckpt", dst + "/checkpoints/last.ckpt", extra):
        print("  %s: [%d, %d] -> [%d, %d]" % (w[0], w[1], w[3], w[2], w[3]))
    print("  obs_norm.txt:", pad_norm(src + "/obs_norm.txt", dst + "/obs_norm.txt", extra), "words")
    for name in ("metrics.config.kv", "run.kv"):
        if os.path.exists(src + "/" + name):
            shutil.copy(src + "/" + name, dst + "/" + name)
    with open(dst + "/PADDED_FROM", "w") as f:
        f.write(src + " +" + str(extra) + " obs words\n")
    print("wrote", dst)


if __name__ == "__main__":
    main()
