"""Fit the real follower's servo response from a `--sysid` recording.

    pixi run python tools/so101/fit_servo_response.py sysid.csv

`examples/so101/pixel_student_deploy_real.mojo --arm --sysid sysid.csv`
steps each joint +A, back, -A, back (0.8 s each) from the sim's start pose.
For every step this finds the command change, then fits, per joint,

    delay d (ticks) and tau (s):  q(t) = q0 + (q_tgt - q0)(1 - exp(-(t - t0 - d dt)/tau))

by grid search over the ticks after the step, and reports the steady-state
error (gravity sag) too. The output is `delta_action.ServoLag`'s numbers:
pass `--lag-tau lo,hi --lag-delay lo,hi` spanning the joints' values.
"""
import csv, sys
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t_s"]) for r in rows])
    jn = np.array([int(r["joint"]) for r in rows])
    tgt = np.array([float(r["tgt"]) for r in rows])
    q = np.array([[float(r[f"q{i}"]) for i in range(6)] for r in rows])
    dt = float(np.median(np.diff(t)))
    print(f"{len(rows)} ticks, control period {dt * 1000:.1f} ms")
    taus, delays = [], []
    for j in range(6):
        idx = np.nonzero(jn == j)[0]
        if len(idx) < 10:
            continue
        tj, gj, qj = t[idx], tgt[idx], q[idx, j]
        steps = [k for k in range(1, len(idx)) if abs(gj[k] - gj[k - 1]) > 1e-6]
        fits = []
        for k in steps:
            end = min(len(idx), k + int(0.8 / dt))
            q0, qt = qj[k - 1], gj[k]
            seg = qj[k:end]
            if abs(qt - q0) < 1e-3 or len(seg) < 8:
                continue
            frac = (seg - q0) / (qt - q0)
            best = None
            for d in range(0, 6):
                for tau in np.arange(0.02, 0.60, 0.005):
                    n = np.arange(len(seg)) + 1 - d
                    model = np.where(n > 0, 1 - np.exp(-np.maximum(n, 0) * dt / tau), 0.0)
                    err = np.mean((frac - model) ** 2)
                    if best is None or err < best[2]:
                        best = (d, tau, err)
            sse = 1 - frac[-3:].mean()
            fits.append((best[0], best[1], best[2], sse))
        if not fits:
            continue
        d = np.median([f[0] for f in fits])
        tau = np.median([f[1] for f in fits])
        err = np.median([f[2] for f in fits])
        sse = np.median([f[3] for f in fits])
        taus.append(tau)
        delays.append(d)
        print(f"  {JOINTS[j]:14s} delay {d:.0f} ticks  tau {tau * 1000:5.0f} ms  fit mse {err:.4f}  "
              f"steady-state shortfall {sse * 100:5.1f} %  ({len(fits)} steps)")
    if taus:
        print(f"suggested: --lag-tau {min(taus) * 1000 * 0.8:.0f},{max(taus) * 1000 * 1.2:.0f}"
              f" --lag-delay {int(min(delays))},{int(max(delays))}")


if __name__ == "__main__":
    main(sys.argv[1])
