"""Synthetic hhtools trajectories and their 50 Hz reference rows, for `tests/data/test_retarget_csv.mojo`.

    pixi run python tools/supermimic/retarget_oracle.py --out /tmp/retarget_synth

Writes, for a made-up 5-joint robot (joints `ja` .. `je`):

    clips/<name>.csv   files in hhtools' CSV layout (`# sample_rate`, `# playback_mesh_z_lift`,
                       root position, root quaternion XYZW, `dof_<joint>` in a SHUFFLED column order)
    ref/<name>.txt     the rows the importer must produce, one frame per line: this file's own numpy
                       implementation of SuperMimic's resample (linear blend, slerp with a linear branch
                       above dot 0.9995, np.gradient, a central SO(3) difference for the root's angular
                       velocity), on the trajectory in the robot's joint order with the lift taken out
    bad/*.csv          files the reader must refuse

No data from any dataset: the motions are sums of sinusoids.
"""

import argparse
import os

import numpy as np

JOINTS = ["ja", "jb", "jc", "jd", "je"]


def quat_from_rotvec(v):
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * v / a])


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def rotvec(q):
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    angle = 2.0 * np.arctan2(s, q[0])
    return q[1:] * (angle / s if s > 1e-9 else 2.0)


def resample(pos, quat, dof, fps_in, fps_out=50.0):
    n_in = len(pos)
    dt_in, dt_out = 1.0 / fps_in, 1.0 / fps_out
    duration = (n_in - 1) * dt_in
    n_out = int(np.floor(duration / dt_out + 1e-6)) + 1
    P, Q, D = np.zeros((n_out, 3)), np.zeros((n_out, 4)), np.zeros((n_out, dof.shape[1]))
    nonlinear = 0
    for i in range(n_out):
        x = (i * dt_out) / duration * (n_in - 1) if duration > 0 else 0.0
        i0 = min(max(int(np.floor(x)), 0), n_in - 1)
        i1 = min(i0 + 1, n_in - 1)
        b = x - i0
        P[i] = pos[i0] * (1 - b) + pos[i1] * b
        D[i] = dof[i0] * (1 - b) + dof[i1] * b
        q0, q1 = quat[i0] / np.linalg.norm(quat[i0]), quat[i1] / np.linalg.norm(quat[i1])
        dot = float(q0 @ q1)
        if dot < 0:
            q1, dot = -q1, -dot
        if dot > 0.9995:
            q = (1 - b) * q0 + b * q1
        else:
            th = np.arccos(np.clip(dot, -1, 1))
            q = (np.sin((1 - b) * th) * q0 + np.sin(b * th) * q1) / np.sin(th)
            nonlinear += 1
        Q[i] = q / np.linalg.norm(q)
    V = np.gradient(P, dt_out, axis=0)
    DD = np.gradient(D, dt_out, axis=0)
    W = np.zeros((n_out, 3))
    for i in range(1, n_out - 1):
        W[i] = rotvec(qmul(Q[i + 1], Q[i - 1] * np.array([1, -1, -1, -1]))) / (2 * dt_out)
    W[0], W[-1] = W[1], W[-2]
    return np.hstack([P, Q, D, V, W, DD]), nonlinear


def motion(rng, n, fps, spin):
    t = np.arange(n) / fps
    pos = np.stack([0.4 * t + 0.1 * np.sin(1.3 * t), 0.2 * np.sin(0.7 * t), -0.04 + 0.03 * np.sin(2.1 * t)], 1)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    quat = np.array([quat_from_rotvec(axis * spin * tt + 0.2 * np.array([np.sin(tt), np.cos(1.7 * tt), 0.0])) for tt in t])
    w = rng.uniform(1.0, 6.0, size=len(JOINTS))
    ph = rng.uniform(0, 6.28, size=len(JOINTS))
    dof = 0.8 * np.sin(np.outer(t, w) + ph)
    return pos, quat, dof


def write_csv(path, pos, quat, dof, fps, lift, order, rng, extra=None, drop=None, ragged=False):
    cols = ["time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]
    names = [JOINTS[k] for k in order if JOINTS[k] != drop]
    cols += ["dof_" + n for n in names] + (["dof_" + extra] if extra else [])
    with open(path, "w") as f:
        f.write("# num_dofs: %d\n# playback_mesh_z_lift: %.6f\n# retarget_backend: newton\n# robot: synthetic\n"
                "# sample_rate: %.6f\n" % (len(names), lift, fps))
        f.write(",".join(cols) + "\n")
        for i in range(len(pos)):
            row = [i / fps, pos[i, 0], pos[i, 1], pos[i, 2] + lift, quat[i, 1], quat[i, 2], quat[i, 3], quat[i, 0]]
            row += [dof[i, JOINTS.index(n)] for n in names] + ([0.0] if extra else [])
            if ragged and i == 3:
                row = row[:-1]
            f.write(",".join(repr(float(v)) for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for d in ("clips", "ref", "bad"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    rng = np.random.default_rng(0)
    # name, frames, fps, lift, root spin (rad/s; the fast one drives the slerp's non-linear branch)
    specs = [("walk_a", 400, 120.0, -0.0221, 0.6), ("spin_b", 300, 120.0048, 0.0, 9.0),
             ("slow_c", 90, 30.0, -0.0105, 3.0), ("short_d", 20, 120.0, -0.003, 0.5)]
    total_nonlinear = 0
    for name, n, fps, lift, spin in specs:
        pos, quat, dof = motion(rng, n, fps, spin)
        order = list(rng.permutation(len(JOINTS)))
        write_csv(os.path.join(a.out, "clips", name + ".csv"), pos, quat, dof, fps, lift, order, rng)
        rows, nl = resample(pos, quat, dof, fps)
        total_nonlinear += nl
        with open(os.path.join(a.out, "ref", name + ".txt"), "w") as f:
            f.write("%d %d %r\n" % (rows.shape[0], rows.shape[1], lift))
            for r in rows:
                f.write(" ".join(repr(float(v)) for v in r) + "\n")
    assert total_nonlinear > 0, "no frame took the slerp's non-linear branch"
    pos, quat, dof = motion(rng, 30, 120.0, 0.5)
    ident = list(range(len(JOINTS)))
    bad = os.path.join(a.out, "bad")
    write_csv(os.path.join(bad, "missing_joint.csv"), pos, quat, dof, 120.0, 0.0, ident, rng, drop="jc")
    write_csv(os.path.join(bad, "extra_joint.csv"), pos, quat, dof, 120.0, 0.0, ident, rng, extra="jz")
    write_csv(os.path.join(bad, "ragged.csv"), pos, quat, dof, 120.0, 0.0, ident, rng, ragged=True)
    text = open(os.path.join(a.out, "clips", "walk_a.csv")).read()
    open(os.path.join(bad, "no_rate.csv"), "w").write("\n".join(l for l in text.split("\n") if "sample_rate" not in l))
    open(os.path.join(bad, "nan.csv"), "w").write(text.replace(text.split("\n")[9].split(",")[2], "nan", 1))
    print("wrote %d clips (%d frames on the slerp's non-linear branch) and 5 bad files to %s"
          % (len(specs), total_nonlinear, a.out))


if __name__ == "__main__":
    main()
