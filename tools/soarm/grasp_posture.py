#!/usr/bin/env python
"""The GRASP POSTURE of a real recording, and of an expert `.demo`, side by side.

    pixi run python tools/soarm/grasp_posture.py
    pixi run python tools/soarm/grasp_posture.py --demo projects/so101-tower/demos/<file>.demo

`examples/so101/tower_expert_record.mojo` grasps with the fixed finger (gripper
-z) pointing DOWN and the pinch axis (gripper +x) RADIAL. A student trained on
those grasps meets a real operator who does neither, so this measures, at each
episode's grasp, the same two quantities in the expert's own terms:

  * tilt   — the angle of the finger axis from straight down (deg), and its
             radial component's sign: + = the finger points AWAY from the base
  * pinch  — the signed angle of the pinch axis's horizontal part from the
             radial direction, folded to (-90, 90] (a jaw is symmetric end
             for end); 0 = the expert's grasp
  * reach  — the grasp centre's horizontal distance from the base and its
             height (m)

plus the six joints (real-arm degrees; the jaw 0..100).

THE GRASP FRAME, real: the first frame, with the arm off its rest (lift above
-90 deg), where the jaw STATE has closed at least 8 below its running maximum
and then holds within 1 for 6 frames — the closed plateau on the brick
(`_grasp_row`; the episode's minimum is not it, the operator may close the
jaw fully on the way back to rest). Expert: the first row whose jaw ACTION falls 0.3 below
its running maximum (the close command). Joints go through the measured zero
(`robot/so101/sim_map.tower_follower_zero_deg`, read from the source).
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/soarm"))
sys.path.insert(0, str(ROOT / "tools/demo"))
from check_joint_zero import zero_from_mojo  # noqa: E402
from demo_stats import read_demo  # noqa: E402

SCENE = ROOT / "noeira/tasks/scenes/so101_tower.xml"
NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "jaw"]


class Fk:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.g = self.m.body("robot_gripper").id
        self.s = self.m.site("robot_grasp_center").id
        self.lo, self.hi = self.m.actuator_ctrlrange[5]

    def features(self, q_model):
        d = self.d
        d.qpos[:] = 0
        d.qpos[:6] = q_model
        mujoco.mj_kinematics(self.m, d)
        R = d.xmat[self.g].reshape(3, 3)
        finger = R @ np.array([0.0, 0.0, -1.0])
        pinch = R @ np.array([1.0, 0.0, 0.0])
        p = d.site_xpos[self.s].copy()
        radial = p[:2] / max(np.linalg.norm(p[:2]), 1e-9)
        tilt = np.degrees(np.arccos(np.clip(-finger[2], -1, 1)))
        tilt *= np.sign(finger[:2] @ radial) or 1.0
        ph = pinch[:2]
        ang = np.degrees(np.arctan2(radial[0] * ph[1] - radial[1] * ph[0], radial @ ph))
        ang = (ang + 90.0) % 180.0 - 90.0
        return tilt, ang, np.linalg.norm(p[:2]), p[2]


def _grasp_row(x):
    """The first row, off rest, where the jaw has closed at least 8 below its
    running maximum and then HOLDS (within 1 for the next 6 rows) — closed on
    the object. -1 if none. (The episode's minimum jaw is not it: an operator
    may close the jaw fully on the way back to rest.)"""
    jaw = x[:, 5]
    away = x[:, 1] > -90.0
    run_max = -1e9
    for i in range(len(x) - 6):
        if not away[i]:
            continue
        run_max = max(run_max, jaw[i])
        if jaw[i] < run_max - 8.0 and np.all(np.abs(jaw[i + 1:i + 7] - jaw[i]) <= 1.0):
            return i
    return -1


def real_grasps(dataset, zero_deg, fk):
    t = pq.read_table(
        dataset / "data/chunk-000/file-000.parquet",
        columns=["observation.state", "episode_index"],
    ).to_pydict()
    s = np.array(t["observation.state"])
    ep = np.array(t["episode_index"])
    rej_path = dataset / "meta/rejected_episodes.json"
    rej = set(json.loads(rej_path.read_text())["rejected_episodes"]) if rej_path.exists() else set()
    joints, feats = [], []
    for e in sorted(set(ep.tolist()) - rej):
        x = s[ep == e]
        g = _grasp_row(x)
        if g < 0:
            continue
        r = x[g]
        q = np.radians(r[:5] + zero_deg)
        g = fk.lo + r[5] / 100.0 * (fk.hi - fk.lo)
        joints.append(r)
        feats.append(fk.features(np.r_[q, g]))
    return np.array(joints), np.array(feats)


def demo_grasps(path, zero_deg, fk):
    d = read_demo(path)
    joints, feats = [], []
    for s0, ln, ok in d["episodes"]:
        act = d["act"][s0:s0 + ln, 5]
        run_max = np.maximum.accumulate(act)
        idx = np.where(act < run_max - 0.3)[0]
        if len(idx) == 0:
            continue
        q = d["obs"][s0 + idx[0], :6].astype(float)
        feats.append(fk.features(q))
        deg = np.degrees(q[:5]) - zero_deg  # model -> real-arm degrees
        jaw = 100.0 * (q[5] - fk.lo) / (fk.hi - fk.lo)
        joints.append(np.r_[deg, jaw])
    return np.array(joints), np.array(feats)


def table(label, joints, feats):
    print(f"\n{label}: {len(joints)} grasps  (p5 / p25 / median / p75 / p95)")
    cols = [(n, joints[:, j]) for j, n in enumerate(NAMES)]
    cols += [("tilt deg", feats[:, 0]), ("pinch deg", feats[:, 1]),
             ("reach m", feats[:, 2]), ("height m", feats[:, 3])]
    for n, v in cols:
        q = np.percentile(v, [5, 25, 50, 75, 95])
        print("  %-11s" % n + "".join("%9.3f" % x if "m" == n[-1] else "%9.1f" % x for x in q))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="projects/so101-tower/datasets/cube-in-bowl")
    ap.add_argument("--demo", default="")
    args = ap.parse_args()
    zero = zero_from_mojo()
    fk = Fk()
    j, f = real_grasps(Path(args.dataset), zero, fk)
    table("REAL " + args.dataset, j, f)
    if args.demo:
        j2, f2 = demo_grasps(args.demo, zero, fk)
        table("EXPERT " + args.demo, j2, f2)


if __name__ == "__main__":
    main()
