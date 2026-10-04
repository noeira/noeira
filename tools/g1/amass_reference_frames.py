"""Oracle for AMASS step 1: the dataset's OWN recipe through MuJoCo.

`examples/g1/amass_import.mojo` reads `fleaven/Retargeted_AMASS_for_robotics`
by one convention, stated only in that dataset's `g1/visualize.py`:

    d.qpos       = data[step, :7 + 29]
    d.qpos[3]    = data[step, 6]        # w first
    d.qpos[4:7]  = data[step, 3:6]
    jpos[:, 2]  += 0.793

Everything downstream of that is already gated (`test_lafan_import_vs_oracle`
holds the conversion against the reference dump, `test_unitree_g1_privileged_obs`
holds the 463-D observation). What is NOT gated by either is whether we READ
THE FILE RIGHT — and a wrong quaternion order gives a valid rotation, a wrong
height offset gives a plausible crouch, and a wrong DoF slice gives a pose.
All three pass every internal consistency check there is.

So this runs the recipe above through MuJoCo on the ONE frame where the 50 Hz
resampler is exact by construction — frame 0, where time = 0 gives phase 0,
index 0 and blend 0, with no float32 subtlety to argue about — and dumps the
world position and quaternion of the 30 skeleton bodies.

⚠ IT ALSO DUMPS THE THREE PLAUSIBLE WRONG READINGS, and the gate asserts our
store does NOT match them: the height offset dropped, the quaternion read
WXYZ, and the DoF slice shifted by one. A gate that only checks agreement
with the right answer cannot tell a correct reader from a fixture too
insensitive to disagree with anything — and on a near-upright frame all four
variants put the pelvis in nearly the same place. Variant 0 is the answer;
1-3 are the control.

Clips are chosen for their FRAME 0, not their name: "A10 - lie to crouch"
starts lying down and "A11 - crawl forward" on all fours, where a rotation
error is metres rather than millimetres.

    pixi run python tools/g1/amass_reference_frames.py \
        --clips <npy> [<npy> ...] --out /tmp/amass_ref
"""
import argparse
import json
import os
import re

import numpy as np
import mujoco

# the reference's `body_names` order -> our model's body ids, from
# `noeira/envs/robots/unitree_g1_priv_obs.mojo:g1_skeleton_body`
def skeleton_body(i):
    if i < 7:
        return 1 + i
    if i < 13:
        return 12 + (i - 7)
    if i < 16:
        return 22 + (i - 13)
    if i < 23:
        return 25 + (i - 16)
    return 33 + (i - 23)


def fps_of(path):
    """The dataset's own parse: `fr = fpath[-12:-9]`."""
    fr = path[-12:-9]
    return int(fr[1:]) if fr[0] == "_" else int(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="noeira/envs/robots/assets/unitree_g1.xml")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/amass_ref")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.xml)
    d = mujoco.MjData(m)
    os.makedirs(a.out, exist_ok=True)
    out = []
    for path in a.clips:
        jpos = np.load(path)
        assert jpos.ndim == 2 and jpos.shape[1] == 36, jpos.shape
        jpos = jpos.copy()
        jpos[:, 2] += 0.793                      # read_rtj
        fr = fps_of(path)
        rec = {
            "clip": os.path.basename(path),
            "fps": fr,
            "frames": int(jpos.shape[0]),
            "variants": [],
        }
        allraw = np.load(path)
        raw = allraw[0]                          # frame 0, UNSHIFTED
        for variant in range(4):
            q = np.zeros(7 + 29)
            if variant == 0:                     # the dataset's own recipe
                q[0:3] = raw[0:3]
                q[2] += 0.793
                q[3] = raw[6]
                q[4:7] = raw[3:6]
                q[7:] = raw[7:36]
            elif variant == 1:                   # the height offset dropped
                q[0:3] = raw[0:3]
                q[3] = raw[6]
                q[4:7] = raw[3:6]
                q[7:] = raw[7:36]
            elif variant == 2:                   # the quaternion read WXYZ
                q[0:3] = raw[0:3]
                q[2] += 0.793
                q[3:7] = raw[3:7]
                q[7:] = raw[7:36]
            else:                                # the DoF slice off by one
                q[0:3] = raw[0:3]
                q[2] += 0.793
                q[3] = raw[6]
                q[4:7] = raw[3:6]
                q[7:35] = raw[6:34]
                q[35] = 0.0
            d.qpos[: 7 + 29] = q
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            pos = []
            quat = []
            for s in range(30):
                b = skeleton_body(s)
                pos.append([float(x) for x in d.xpos[b]])
                # MuJoCo's xquat is WXYZ; the store writes XYZW
                qq = d.xquat[b]
                quat.append([float(qq[1]), float(qq[2]), float(qq[3]), float(qq[0])])
            rec["variants"].append({"variant": variant, "pos": pos, "quat": quat})

        # ⚠ FRAME 1 AS WELL, for variant 0 only. The 50 Hz resampler blends
        # body POSITIONS with weight 0 at row 0, so they are frame 0 exactly;
        # body ROTATIONS go through the reference's `slerp`, which returns the
        # unnormalised MIDPOINT of its two inputs whenever
        # sin(half-angle) < 1e-3 — and two consecutive 120 Hz frames always
        # are. So the store's row-0 quaternion is (q[0] + q[1]) / 2, not q[0],
        # and comparing it to frame 0 can only ever agree to within half a
        # frame of rotation (3.9e-4 on a clip that is moving at frame 0,
        # 1.4e-5 on one that is standing still). Reproduced, not fixed: see
        # §10 of noeira/data/lafan.mojo.
        q1 = allraw[1]
        qq1 = np.zeros(7 + 29)
        qq1[0:3] = q1[0:3]
        qq1[2] += 0.793
        qq1[3] = q1[6]
        qq1[4:7] = q1[3:6]
        qq1[7:] = q1[7:36]
        d.qpos[: 7 + 29] = qq1
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        nxt = []
        for s in range(30):
            b = skeleton_body(s)
            qz = d.xquat[b]
            nxt.append([float(qz[1]), float(qz[2]), float(qz[3]), float(qz[0])])
        rec["quat_frame1"] = nxt
        out.append(rec)
        print(f"{rec['clip']}: {rec['frames']} frames @ {fr} fps, 4 variants")

    dst = os.path.join(a.out, "frames.json")
    with open(dst, "w") as f:
        json.dump(out, f)
    print("wrote", dst)


if __name__ == "__main__":
    main()
