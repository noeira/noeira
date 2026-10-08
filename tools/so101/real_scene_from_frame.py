"""A real scene's prop positions, from one overhead frame — the `--brick x,y
--bowl x,y` that rebuild it in the sim.

    pixi run python tools/so101/real_scene_from_frame.py runs/_real_so101/px_rec_w15_1
    pixi run python tools/so101/real_scene_from_frame.py overhead_undistorted_16.png [...]

Takes `--record` directories of `examples/so101/pixel_student_deploy_real.mojo`
(their `overhead_undistorted_16.png`) or overhead PNGs. ⚠ The frame must be
UNDISTORTED to the sim's pinhole (the deploy's `_undistorted` frames are; a
raw LeRobot frame is not). ⚠ A recording's frame 0 can be stale — taken
before the ramp to the start pose, with the props where they stood then — so
a directory reads frame 16.

1. The props are found by colour: the blue cube, the yellow bowl. The rows
   below `--bottom` are left out (the arm's base is blue). A prop under the
   arm is not seen: pick a frame where both are in view.
2. Each prop's pixel centroid goes through its own desk-plane homography
   (the two props' centres sit at different heights). They are fitted to
   `tools/so101/sim_prop_pixels.mojo`'s samples — where the SIM's overhead
   camera sees each prop over a 7 x 7 desk grid — at 640 x 480. Fit error:
   brick 1 mm mean, bowl 3 mm mean. After a change of the sim's camera,
   re-run that tool and pass its CSV as `--refit`.
3. `--brick-dx` (default 0.02 m) is added to the brick's x. The homography
   reads the real cube about 2 cm short in x: replays of real runs through the
   student (`pixel_student_replay_real.mojo`) matched the logged actions best
   with the brick there.
"""
import argparse, csv, os, sys

import numpy as np
from PIL import Image

# fitted to tools/so101/sim_prop_pixels.mojo's output (pixels x 2 -> 640 x 480)
H = {
    "brick": np.array([
        [1.86042588509356e-05, -0.0023599301835992596, 0.9889439787424165],
        [-0.0026790873341459233, -0.0002934848169078861, 0.7435716894458202],
        [-1.6044491708172447e-05, 0.0022008182268007707, 1.0]]),
    "bowl": np.array([
        [-4.531752865505868e-06, -0.0023093098760313684, 0.9658813761215654],
        [-0.0025383099505729427, -0.00029263665226921697, 0.7037650573835081],
        [-5.136966190589773e-05, 0.0020955601807708477, 1.0]]),
}


def refit(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for prop in ("brick", "bowl"):
        A = []
        for r in rows:
            if r["prop"] != prop:
                continue
            u, v = 2 * float(r["px"]), 2 * float(r["py"])
            x, y = float(r["x"]), float(r["y"])
            A.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
            A.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
        h = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
        out[prop] = h / h[2, 2]
    return out


def find_props(path, bottom, min_px):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    a = np.asarray(im).astype(int)
    s = 640.0 / w  # the homographies are in 640 x 480 pixels
    a = a[: int(h - bottom / s)]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    masks = {
        "brick": (b > r + 50) & (b > g + 20) & (b > 100),
        "bowl": (r > 150) & (g > 100) & (b < 80) & (r > b + 100),
    }
    found = {}
    for prop, m in masks.items():
        ys, xs = np.nonzero(m)
        if len(xs) >= min_px:
            found[prop] = (xs.mean() * s, ys.mean() * s, len(xs))
    return found


def to_desk(Hs, prop, u, v):
    p = Hs[prop] @ [u, v, 1.0]
    return p[0] / p[2], p[1] / p[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="recording directories or overhead PNGs")
    ap.add_argument("--frame", type=int, default=16, help="the frame a directory reads")
    ap.add_argument("--brick-dx", type=float, default=0.02)
    ap.add_argument("--bottom", type=int, default=80, help="rows left out at the bottom (640 x 480)")
    ap.add_argument("--min-px", type=int, default=20)
    ap.add_argument("--refit", help="a sim_prop_pixels.mojo CSV to refit the homographies from")
    args = ap.parse_args()
    Hs = refit(args.refit) if args.refit else H
    bad = 0
    for inp in args.inputs:
        path = (os.path.join(inp, "overhead_undistorted_%d.png" % args.frame)
                if os.path.isdir(inp) else inp)
        found = find_props(path, args.bottom, args.min_px)
        flags = []
        for prop in ("brick", "bowl"):
            if prop not in found:
                flags.append("(%s not seen)" % prop)
                bad += 1
                continue
            u, v, n = found[prop]
            x, y = to_desk(Hs, prop, u, v)
            if prop == "brick":
                x += args.brick_dx
            flags.append("--%s %.3f,%.3f" % (prop, x, y))
        print("%s: %s" % (inp, " ".join(flags)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
