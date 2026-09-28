#!/usr/bin/env python
"""Check (or fit) the follower's joint zero against a RECORDED dataset.

    pixi run python tools/soarm/check_joint_zero.py                  # check
    pixi run python tools/soarm/check_joint_zero.py --fit            # re-measure
    pixi run python tools/soarm/check_joint_zero.py --zero=-10,-3,-7,0,0

`noeira/robot/so101/sim_map.mojo::tower_follower_zero_deg` is the correction
`model_rad = deg2rad(lerobot_deg) + zero` for the so101-tower follower. This
re-runs the measurement that set it, with nothing but the dataset:

  * a frame of the OVERHEAD video, undistorted to the sim's pinhole with the
    fisheye calibration (`vision/fisheye.mojo`'s map, transcribed);
  * the tower scene in MuJoCo posed at that frame's `observation.state`
    (LeRobot units -> radians + zero, the gripper by fraction of ctrlrange),
    seen from the scene's `tower_overhead_cam`;
  * the score: the fraction of the sim arm's DARK pixels (the servos) that
    land on dark real pixels.

A wrong zero puts the sim servos beside the real ones. The check passes when
the zero beats `zero = 0` on at least `--min-win` of the frames. `--fit`
re-measures pan, lift, elbow (coordinate descent, 1-degree grid); the wrist
joints move the score by < 0.01 over +-8 degrees and are NOT measurable here.

⚠ The zero belongs to ONE calibration (`sim_map.tower_follower_calib_mid`):
a dataset recorded after a `lerobot-calibrate` re-run needs its own `--fit`.
⚠ The camera is taken as the asset's. A camera yaw about the base would be
absorbed by the pan zero; the scene's static parts put that yaw within 0..+3
degrees (which would make the pan zero MORE negative, not zero).
Needs `ffmpeg` on PATH, mujoco and pyarrow (the pixi env).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SIM_MAP = ROOT / "noeira/robot/so101/sim_map.mojo"
SCENE = ROOT / "noeira/tasks/scenes/so101_tower.xml"
CAM = "tower_overhead_cam"
FOVY = 73.7398  # the tower's overhead_cam (the tracer's and MuJoCo's)
W, H = 640, 480
DARK = 150  # r + g + b below this is a servo / cable


def zero_from_mojo():
    """`tower_follower_zero_deg`'s list, read from the source (one copy)."""
    src = SIM_MAP.read_text()
    m = re.search(r"def tower_follower_zero_deg.*?=\s*\[([^\]]*)\]", src, re.S)
    if m is None:
        sys.exit(f"cannot find tower_follower_zero_deg in {SIM_MAP}")
    return np.array([float(v) for v in m.group(1).split(",")])[:5]


def read_calib(path):
    kv = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts:
            kv[parts[0]] = parts[1:]
    if kv.get("model", [""])[0] != "fisheye":
        sys.exit(f"{path}: not a fisheye calibration")
    return [float(v) for v in kv["intrinsics"]], [float(v) for v in kv["dist"]]


class Undistort:
    """`vision/fisheye.mojo::UndistortMap` to `Pinhole.sim(FOVY, W, H)`."""

    def __init__(self, calib):
        (fx, fy, cx, cy), k = read_calib(calib)
        f = H / (2 * np.tan(np.radians(FOVY) / 2))
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        a = (x - (W / 2 - 0.5)) / f
        b = (y - (H / 2 - 0.5)) / f
        r = np.hypot(a, b)
        th = np.arctan(r)
        t2 = th * th
        thd = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
        sc = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        self.mx = fx * a * sc + cx
        self.my = fy * b * sc + cy

    def __call__(self, img):
        x0 = np.clip(np.floor(self.mx).astype(int), 0, W - 2)
        y0 = np.clip(np.floor(self.my).astype(int), 0, H - 2)
        ax = (self.mx - x0)[..., None]
        ay = (self.my - y0)[..., None]
        return (
            img[y0, x0] * (1 - ax) * (1 - ay) + img[y0, x0 + 1] * ax * (1 - ay)
            + img[y0 + 1, x0] * (1 - ax) * ay + img[y0 + 1, x0 + 1] * ax * ay
        )


def video_frame(dataset, ep, fi, fps):
    mp4 = dataset / f"videos/observation.images.overhead/chunk-000/file-{ep:03d}.mp4"
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{(fi + 0.5) / fps:.4f}", "-i", str(mp4),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(H, W, 3).astype(float)


class Scene:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        # the props are parked 50 m away, which sets extent (and znear) so
        # large that MuJoCo clips the arm: pin it
        self.m.stat.extent = 1.0
        self.d = mujoco.MjData(self.m)
        self.lo, self.hi = self.m.actuator_ctrlrange[5]
        self.r = mujoco.Renderer(self.m, H, W)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[0] = 1
        self.opt.geomgroup[2] = 1
        self.robot = np.array([
            self.m.body(self.m.geom_bodyid[i]).name.startswith("robot_")
            for i in range(self.m.ngeom)
        ])

    def dark_arm(self, state, zero_deg):
        m, d = self.m, self.d
        d.qpos[:] = 0
        d.qpos[:5] = np.radians(np.asarray(state[:5]) + zero_deg)
        d.qpos[5] = self.lo + state[5] / 100 * (self.hi - self.lo)
        for jn in ("bowl_free", "brick_free"):
            a = m.jnt_qposadr[m.joint(jn).id]
            d.qpos[a:a + 3] = [10, 0, 50]
            d.qpos[a + 3] = 1
        mujoco.mj_forward(m, d)
        self.r.update_scene(d, camera=CAM, scene_option=self.opt)
        rgb = self.r.render().astype(float)
        self.r.enable_segmentation_rendering()
        seg = self.r.render()[:, :, 0]
        self.r.disable_segmentation_rendering()
        arm = (seg >= 0) & self.robot[np.clip(seg, 0, m.ngeom - 1)]
        return (rgb.sum(2) < DARK) & arm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="projects/so101-tower/datasets/cube-in-bowl")
    ap.add_argument("--calib", default="projects/so101-tower/cameras/camera_overhead.txt")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--zero", default="", help="5 comma-separated degrees (default: sim_map.mojo's)")
    ap.add_argument("--min-win", type=float, default=0.9)
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    zero = (np.array([float(v) for v in args.zero.split(",")]) if args.zero
            else zero_from_mojo())
    fps = 30
    t = pq.read_table(
        dataset / "data/chunk-000/file-000.parquet",
        columns=["episode_index", "frame_index", "observation.state"],
    ).to_pydict()
    rows = {}
    for i, (e, fi) in enumerate(zip(t["episode_index"], t["frame_index"])):
        rows[(e, fi)] = i
    n_ep = max(e for e, _ in rows) + 1
    rng = np.random.default_rng(args.seed)
    und = Undistort(args.calib)
    frames = []
    for e in rng.choice(n_ep, min(args.frames, n_ep), replace=False):
        last = max(fi for (ee, fi) in rows if ee == e)
        fi = int(rng.integers(10, last - 10))
        dark = und(video_frame(dataset, int(e), fi, fps)).sum(2) < DARK
        frames.append((t["observation.state"][rows[(int(e), fi)]], dark))
    scene = Scene()

    def scores(z):
        out = []
        for st, dark in frames:
            s = scene.dark_arm(st, z)
            out.append((s & dark).sum() / max(1, s.sum()))
        return np.array(out)

    base = scores(np.zeros(5))
    if args.fit:
        cur = zero.copy()
        for sweep in range(2):
            for j in range(3):
                grid = cur[j] + np.arange(-8, 8.1, 1.0)
                sc = []
                for g in grid:
                    z = cur.copy()
                    z[j] = g
                    sc.append(scores(z).mean())
                cur[j] = grid[int(np.argmax(sc))]
            print(f"  sweep {sweep}: pan {cur[0]:+.1f}  lift {cur[1]:+.1f}  elbow {cur[2]:+.1f}")
        zero = cur
    s = scores(zero)
    wins = (s > base).mean()
    print(f"dataset {dataset}  ({len(frames)} frames, seed {args.seed})")
    print("zero (deg): " + " ".join(f"{v:+.1f}" for v in zero))
    print(f"  servo pixels on real dark pixels: zero=0 {base.mean():.3f}  "
          f"this zero {s.mean():.3f}  (median {np.median(s):.3f})")
    print(f"  this zero beats zero=0 on {int(round(wins * len(s)))}/{len(s)} frames")
    if wins < args.min_win:
        print(f"FAIL: below {args.min_win:.0%} — re-measure with --fit")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
