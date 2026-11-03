#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | Synthetic BONES-SEED G1 CSVs + the SuperMimic reproduction's answer for them
# +--------------------------------------------------------------------------+ #
"""Writes BONES-SEED-format G1 CSVs, then runs the reproduction's own code on them.

    pixi run python tools/supermimic/make_synthetic_seed.py --out /tmp/seed_synth
    pixi run mojo run -I . tests/data/test_bones_seed.mojo /tmp/seed_synth

No SEED byte is involved. The dataset is licensed confidential, so the public
tree can only gate the importer on fixtures. This script writes them in the
layout the reproduction documents (`references/SuperMimic-Reproduced`,
`src/supermimic/data/motion_convert.py` and its `docs/EXPERIMENTS.md`, 21 Oct):

    a header row: Frame, root_translateX/Y/Z (CENTIMETRES),
    root_rotateX/Y/Z (Euler 'xyz' -- scipy's lower case, i.e. EXTRINSIC --
    in DEGREES), then the 29 G1 joints as `<joint>_dof` in DEGREES,
    in the G1 MJCF order; 120 fps.

Once the licence is accepted, a real slice (`--real-dir`, a few hundred CSVs
from the head of `g1.tar.gz`) goes through the same oracle on the box or the
laptop; the Mojo gate does not care where the CSVs came from.

THE ORACLE IS THE REPRODUCTION'S CODE, LOADED FROM `references/`:
  * `resample_motion` (and its `_so3_derivative`) verbatim, from
    `motion_convert.py`, loaded by file path so its package `__init__` (torch)
    is never imported;
  * `name_filter` verbatim from `scripts/data/filter_bones_seed.py`, with a
    stub `tyro` module, which that script imports only for its CLI.
  * `load_csv` is the one piece NOT run as written, because it reads with
    pandas and the pixi env has none. It is re-stated below with `csv` +
    scipy's `Rotation.from_euler("xyz", degrees=True)` -- the same calls on
    the same columns. The Euler convention, the risky part, is scipy's, not
    ours, so the Mojo side cannot agree with it by sharing a mistake.

Every fixture is shaped to break one thing if it is wrong:
  * `turn_walk`: the yaw passes +-180 deg, so the Euler Z column wraps; a
    hemisphere-blind slerp or a sign-sensitive compare shows up here.
  * `fast_spin_lying`: roll ~90 deg (lying) plus a ~1000 deg/s yaw spin, so
    consecutive 120 fps frames are > 3.6 deg apart and the slerp's
    NON-linear branch (dot <= 0.9995) runs; every other clip takes the
    linear branch almost everywhere, which would hide a wrong sin-weight.
  * `big_euler`: all three angles large and unequal, so 'xyz' extrinsic,
    'XYZ' intrinsic and a swapped column order give different quaternions.
  * `exact_grid`: 121 frames = 1.000 s, the end point lands on the 50 Hz grid
    (the `+ 1e-6` in `n_out`); `off_grid` (127 frames) does not.
  * `two_frames` / `one_frame`: `_so3_derivative`'s `n < 3` branch and the
    zero-duration phase.
  * `extra_column`: an unknown column between the root and the joints; the
    reader must select by NAME.
  * `permuted_joints`: two joint columns swapped; both sides must REFUSE it.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "references" / "SuperMimic-Reproduced"
FPS = 120.0


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


def reference_modules():
    if "tyro" not in sys.modules:
        sys.modules["tyro"] = types.ModuleType("tyro")  # only its CLI uses it
    mc = _load_module("sm_motion_convert", REF / "src/supermimic/data/motion_convert.py")
    fb = _load_module("sm_filter_bones_seed", REF / "scripts/data/filter_bones_seed.py")
    return mc, fb


# ── fixtures ────────────────────────────────────────────────────────────────
def _joint_traj(t: np.ndarray, rng: np.random.Generator, amp_deg: float) -> np.ndarray:
    """(n, 29) smooth joint angles in degrees: per-joint sinusoids, distinct phases."""
    n = t.shape[0]
    out = np.empty((n, 29))
    for j in range(29):
        f = 0.3 + 1.7 * rng.random()
        ph = 2 * math.pi * rng.random()
        out[:, j] = (5.0 + amp_deg * rng.random()) * np.sin(2 * math.pi * f * t + ph) + (j - 14) * 0.7
    return out


def fixtures(rng: np.random.Generator) -> dict[str, dict]:
    """name -> {pos_cm (n,3), eul_deg (n,3), joints_deg (n,29), extra (bool), permute (bool)}."""
    out: dict[str, dict] = {}

    def clip(n):
        return np.arange(n) / FPS

    # turn_walk: 4 s, walks 1 m/s in a circle; yaw from 0 to 400 deg, wrapped to (-180, 180]
    t = clip(480)
    yaw = 100.0 * t
    yaw_w = (yaw + 180.0) % 360.0 - 180.0
    pos = np.stack([100 * np.cos(np.deg2rad(yaw)) * 1.5, 100 * np.sin(np.deg2rad(yaw)) * 1.5, 79.0 + 2 * np.sin(2 * math.pi * 2 * t)], 1)
    eul = np.stack([3 * np.sin(2 * math.pi * 2 * t), 2 * np.cos(2 * math.pi * 2 * t), yaw_w], 1)
    out["turn_walk_R_001__A101"] = dict(pos_cm=pos, eul_deg=eul, joints_deg=_joint_traj(t, rng, 30))

    # fast_spin_lying: 1.5 s lying on the side, spinning ~1000 deg/s about world z
    t = clip(180)
    pos = np.stack([10 * t, -5 * t, 15.0 + 0 * t], 1)
    yaw = 1000.0 * t
    eul = np.stack([88.0 + 4 * np.sin(2 * math.pi * t), 10 * np.sin(2 * math.pi * 0.5 * t), (yaw + 180.0) % 360.0 - 180.0], 1)
    out["fast_spin_lying_001__A102_M"] = dict(pos_cm=pos, eul_deg=eul, joints_deg=_joint_traj(t, rng, 60))

    # big_euler: all three Euler angles large and unequal, slowly varying
    t = clip(240)
    eul = np.stack([40 + 30 * np.sin(t), -55 + 20 * np.cos(1.3 * t), 120 + 50 * np.sin(0.7 * t)], 1)
    pos = np.stack([5 * t, 3 * t, 60 + 0 * t], 1)
    out["big_euler_001__A103"] = dict(pos_cm=pos, eul_deg=eul, joints_deg=_joint_traj(t, rng, 45))

    for name, n in (("exact_grid_001__A104", 121), ("off_grid_001__A105", 127), ("two_frames_001__A106", 2), ("one_frame_001__A107", 1)):
        t = clip(n)
        pos = np.stack([20 * t, 0 * t, 78.5 + 0 * t], 1)
        eul = np.stack([1 + 0 * t, -2 + 0 * t, 30 + 40 * t], 1)
        out[name] = dict(pos_cm=pos, eul_deg=eul, joints_deg=_joint_traj(t, rng, 20))

    t = clip(150)
    pos = np.stack([30 * t, 10 * t, 79 + 0 * t], 1)
    eul = np.stack([2 * np.sin(t), 1 + 0 * t, -170 + 60 * t], 1)
    out["extra_column_001__A108"] = dict(pos_cm=pos, eul_deg=eul, joints_deg=_joint_traj(t, rng, 25), extra=True)
    out["permuted_joints_001__A109"] = dict(pos_cm=pos, eul_deg=eul, joints_deg=_joint_traj(t, rng, 25), permute=True)
    return out


def write_csv(path: Path, fx: dict, joint_names: tuple[str, ...]) -> None:
    cols = ["Frame", "root_translateX", "root_translateY", "root_translateZ", "root_rotateX", "root_rotateY", "root_rotateZ"]
    jcols = [f"{n}_dof" for n in joint_names]
    if fx.get("permute"):
        jcols[3], jcols[9] = jcols[9], jcols[3]  # left_knee <-> right_knee, data unchanged
    extra = bool(fx.get("extra"))
    header = cols + (["unknown_channel"] if extra else []) + jcols
    n = fx["pos_cm"].shape[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            row = [str(i)]
            row += [f"{v:.6f}" for v in fx["pos_cm"][i]]
            row += [f"{v:.6f}" for v in fx["eul_deg"][i]]
            if extra:
                row.append(f"{123.456 + i:.6f}")
            row += [f"{v:.6f}" for v in fx["joints_deg"][i]]
            w.writerow(row)


# ── the oracle ──────────────────────────────────────────────────────────────
def load_csv_oracle(path: Path, joint_names: tuple[str, ...]) -> np.ndarray:
    """`motion_convert.load_csv`'s 'bones' branch with `csv` for pandas. (N, 36) float64:
    root pos (m), root quat XYZW, 29 joints (rad)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    cols = rows[0]
    data = np.array([[float(v) for v in r] for r in rows[1:]], dtype=np.float64)
    col = {c: i for i, c in enumerate(cols)}
    joint_cols = [c for c in cols if c.endswith("_joint_dof")]
    expected = [f"{n}_dof" for n in joint_names]
    if joint_cols != expected:
        raise ValueError(f"{path}: joint columns differ from G1 order")
    pos = data[:, [col["root_translateX"], col["root_translateY"], col["root_translateZ"]]] / 100.0
    eul = data[:, [col["root_rotateX"], col["root_rotateY"], col["root_rotateZ"]]]
    quat_xyzw = Rotation.from_euler("xyz", eul, degrees=True).as_quat()
    joints = np.deg2rad(data[:, [col[c] for c in joint_cols]])
    return np.concatenate([pos, quat_xyzw, joints], axis=1)


def dump_clip(mc, csv_path: Path, out_dir: Path, stem: str) -> str:
    """Writes `<stem>.raw.npy` [n_in, 36] (quat WXYZ) and `<stem>.ref71.npy` [n_out, 71]; returns a status."""
    try:
        raw = load_csv_oracle(csv_path, mc.G1_JOINT_NAMES)
    except ValueError:
        return "refused"
    m = mc.resample_motion(raw, FPS, 50.0)
    raw_wxyz = raw.copy()
    raw_wxyz[:, 3:7] = raw[:, [6, 3, 4, 5]]
    # The compact library's 71-d row (build_library.py): the pelvis's pose and
    # velocities are the free joint's, so they are the resampled root's.
    ref = np.concatenate([m.root_pos, m.root_quat, m.joint_pos, m.root_lin_vel_w, m.root_ang_vel_w, m.joint_vel], axis=1)
    np.save(out_dir / f"{stem}.raw.npy", raw_wxyz)
    np.save(out_dir / f"{stem}.ref71.npy", ref)
    return "ok"


NAME_PROBES = [
    # keep
    "walk_forward_R_001__A001", "jog_turn_left_002__A010_M", "deposit_object_001__A002",
    "visitor_wave_001__A003", "chairless_squat_001__A004", "unsitting_001__A005",
    # drop (each by a different keyword; some are substring traps of the list)
    "sit_down_chair_001__A006", "sitting_floor_001__A007", "climb_ladder_001__A008",
    "stair_up_001__A009", "box_jump_001__A010", "cartwheel_left_001__A011",
    "handstand_hold_001__A012", "safety_roll_fwd_001__A013", "jump_off_50cm_001__A014",
    "look_over_fence_001__A015", "on_box_balance_001__A016", "bike_ride_001__A017",
    "Table_Lean_001__A018", "kozak_dance_001__A019", "push_obstacle_001__A020",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--real-dir", type=Path, default=None, help="real SEED CSVs to dump instead of fixtures (private)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    mc, fb = reference_modules()
    csv_dir = a.out / "csv"
    ref_dir = a.out / "ref"
    csv_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    if a.real_dir is None:
        for name, fx in fixtures(np.random.default_rng(a.seed)).items():
            write_csv(csv_dir / f"{name}.csv", fx, mc.G1_JOINT_NAMES)
        src = csv_dir
    else:
        src = a.real_dir
    lines = []
    for p in sorted(src.glob("*.csv")):
        status = dump_clip(mc, p, ref_dir, p.stem)
        lines.append(f"{p.stem}\t{status}\t{p.resolve()}")
    (a.out / "clips.tsv").write_text("\n".join(lines) + "\n")

    names = NAME_PROBES + [p.stem for p in sorted(src.glob("*.csv"))]
    rows = [f"{n}\t{fb.name_filter(n) or ''}" for n in names]
    (a.out / "name_filter.tsv").write_text("\n".join(rows) + "\n")
    print(f"{len(lines)} clips ({sum(1 for l in lines if l.split(chr(9))[1] == 'ok')} ok), {len(names)} names -> {a.out}")


if __name__ == "__main__":
    main()
