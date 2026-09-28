#!/usr/bin/env python3
"""Replay the expert's close + lift in MuJoCo 3.12 from OUR sim's exact state.

    pixi run python tools/soarm/replay_close_mujoco.py DUMP_DIR [--log EXPERT.log]

`tower_expert_record.mojo --dump-close DIR` writes, per episode, the state at
the moment the close starts (qpos, qvel) and then every control step of the
close and the lift: the ctrl our env applied and the qpos it reached. This
loads the SAME scene (`noeira/tasks/scenes/so101_tower.xml`), sets that
state, applies the same ctrl for the same 16 substeps of 2 ms per control
step, and compares:

- the ARM's joints, ours vs MuJoCo's, over the close (a check of the replay
  itself: the ctrl mapping, the frame skip — the arm barely touches anything
  in the first steps);
- the BRICK's height at the end of the lift, ours vs MuJoCo's: lifted (> 20
  mm above its start) or not. A brick MuJoCo lifts and ours does not is a
  contact/physics disagreement on the same state.

With `--log`, the expert's verbose log labels each episode's outcome.
"""

import argparse
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE = "noeira/tasks/scenes/so101_tower.xml"
FRAME_SKIP = 16
NQ = 20
ARM = slice(0, 6)
BRICK_Z = 16  # brick_free: qpos 13..19 = pos(3) quat(4); z at 15
BRICK = slice(13, 16)


def load_dump(p):
    qpos = qvel = None
    steps = []
    for line in Path(p).read_text().splitlines():
        if line.startswith("qpos "):
            qpos = np.array([float(x) for x in line.split()[1:]])
        elif line.startswith("qvel "):
            qvel = np.array([float(x) for x in line.split()[1:]])
        elif line.startswith("step "):
            a, b = line[5:].split("|")
            steps.append((np.array([float(x) for x in a.split()]),
                          np.array([float(x) for x in b.split()])))
    return qpos, qvel, steps


def jaw_brick_contacts(m, d):
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.body(m.geom_bodyid[c.geom1]).name
        b2 = m.body(m.geom_bodyid[c.geom2]).name
        if "brick" in (b1 + b2) and ("gripper" in b1 + b2 or "jaw" in b1 + b2):
            out.append((m.geom(c.geom1).name or b1, m.geom(c.geom2).name or b2, c.dist))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--log", default=None)
    ap.add_argument("--verbose", type=int, default=-1, help="episode to trace step by step")
    a = ap.parse_args()
    outcome = {}
    if a.log:
        for line in open(a.log):
            m_ = re.match(r"\s+ep (\d+) -> (\S+)", line)
            if m_:
                outcome[int(m_.group(1))] = m_.group(2) == "SUCCESS"
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    rows = []
    for p in sorted(Path(a.dump).glob("ep_*.txt"), key=lambda q: int(q.stem[3:])):
        ep = int(p.stem[3:])
        qpos, qvel, steps = load_dump(p)
        if qpos is None or len(steps) == 0:
            continue
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        mujoco.mj_forward(m, d)
        z0 = qpos[BRICK][2]
        arm_err = []
        for k, (ctrl, ours) in enumerate(steps):
            d.ctrl[:] = ctrl
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(m, d)
            if k < 5:
                arm_err.append(np.abs(d.qpos[ARM] - ours[ARM]).max())
            if ep == a.verbose:
                print(f"  step {k:2d} ctrl jaw {ctrl[5]:+.3f} | jaw ours {ours[5]:+.3f} mj {d.qpos[5]:+.3f}"
                      f" | brick z ours {1000*(ours[BRICK][2]-z0):+6.1f} mj {1000*(d.qpos[BRICK][2]-z0):+6.1f} mm"
                      f" | mj jaw-brick contacts {len(jaw_brick_contacts(m, d))}")
        ours_dz = 1000 * (steps[-1][1][BRICK][2] - z0)
        mj_dz = 1000 * (d.qpos[BRICK][2] - z0)
        rows.append((ep, ours_dz, mj_dz, max(arm_err) if arm_err else 0.0,
                     steps[-1][1][5], d.qpos[5], outcome.get(ep)))
    ours_l = np.array([r[1] > 20 for r in rows])
    mj_l = np.array([r[2] > 20 for r in rows])
    print(f"episodes {len(rows)} | lifted: ours {ours_l.sum()}  MuJoCo {mj_l.sum()}"
          f" | both {np.sum(ours_l & mj_l)}  only MuJoCo {np.sum(~ours_l & mj_l)}"
          f"  only ours {np.sum(ours_l & ~mj_l)}")
    ae = np.array([r[3] for r in rows])
    print(f"arm |ours - MuJoCo| over the first 5 steps (rad): median {np.median(ae):.2e}  max {ae.max():.2e}")
    print("episodes lifted by MuJoCo only (ep, ours dz, mj dz, jaw ours, jaw mj):")
    for r in rows:
        if r[2] > 20 and r[1] <= 20:
            print(f"  {r[0]:4d} {r[1]:7.1f} {r[2]:7.1f}  {r[4]:+.3f} {r[5]:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
