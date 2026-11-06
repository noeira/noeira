#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | SuperMimic prompt oracle: the task reward from MuJoCo and the paper's table
# +--------------------------------------------------------------------------+ #
"""Reference values for `tests/deep_agents/supermimic/test_supermimic_prompt.mojo`.

    uv run --no-project --python 3.12 --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/prompt_oracle.py --out /tmp/sm_prompt

An independent second implementation of `deep_agents/supermimic/prompt.mojo`:
numpy for the formulas (transcribed from its docstring and the paper's Table
A.6, not from the Mojo), MuJoCo 3.12.0 on mjlab's G1 for the kinematics
(`mj_kinematics`, `mj_comPos`, `mj_comVel`, `mj_objectVelocity(mjOBJ_XBODY,
flg_local=0)`). No code is shared with the port, so a transcription slip on
either side shows up as a mismatch.

The reproduction has no prompt or velocity interface, so there is no reference
to run verbatim: this oracle pins the DEFINITION. Its gradient is gated in the
test against central differences of the forward this oracle checks.

Written (float64, 2-D `.npy`, B rows):
    p_s [B, 71], p_a [B, 29], p_ap [B, 29], p_xi [B, 33], p_spec [B, 31]
    p_atoms [B, 12], p_r [B, 1], p_logit [B, 1]
    p_hands [1, 2]      the hands' FK rows (body id − 1)
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
XML = REPO / "noeira/envs/robots/assets/unitree_g1_mjlab.xml"

GROUPS = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 12)]  # vel, yaw, height, tilt, left hand, right hand
DT = 0.02


def qmul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


def conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate(q, v):
    return qmul(qmul(q, np.r_[0.0, v]), conj(q))[1:]


def rotate_inv(q, v):
    return rotate(conj(q), v)


def heading(q):
    """The yaw-only quaternion of a unit quaternion."""
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lanes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    hl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    hr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    lo = m.jnt_range[1:, 0]
    hi = m.jnt_range[1:, 1]
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.9
    slo, shi = mid - half, mid + half
    rng = np.random.default_rng(a.seed)
    B = a.lanes
    S = np.zeros((B, 71))
    A = rng.normal(0, 0.5, (B, 29))
    AP = rng.normal(0, 0.5, (B, 29))
    XI = rng.normal(0, 0.05, (B, 33))
    SPEC = np.zeros((B, 31))
    ATOMS = np.zeros((B, 12))
    R = np.zeros((B, 1))
    LG = np.zeros((B, 1))
    for b in range(B):
        # a third upright, a third tilted, a third anywhere
        kind = b % 3
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = [rng.uniform(-0.2, 0.2), rng.uniform(-1.0, 1.0), rng.uniform(-3.1, 3.1)][kind]
        q = np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis]
        yaw = rng.uniform(-3.1, 3.1)
        q = qmul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), q)
        p = np.r_[rng.uniform(-2, 2, 2), rng.uniform(0.2, 1.0)]
        qj = rng.uniform(lo - 0.05, hi + 0.05)          # some lanes past the soft limits
        v = rng.normal(0, 1.0, 3)
        w = rng.normal(0, 1.5, 3)                       # body frame
        qd = rng.normal(0, 2.0, 29)
        d.qpos[:] = np.r_[p, q, qj]
        d.qvel[:] = np.r_[v, w, qd]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        hq = heading(q)
        vh = rotate_inv(hq, v)
        wz = rotate(q, w)[2]
        g = rotate_inv(q, np.array([0.0, 0.0, -1.0]))
        pl = rotate_inv(hq, d.xpos[hl] - p)
        pr = rotate_inv(hq, d.xpos[hr] - p)
        at = np.r_[vh[0], vh[1], wz, p[2], g[0], g[1], pl, pr]
        spec = np.zeros(31)
        spec[:12] = at + rng.normal(0, 0.3, 12)
        spec[12:24] = rng.uniform(0, 6, 12)
        spec[24:30] = rng.uniform(0, 3, 6)
        spec[30] = rng.uniform(0, 4)
        task = 0.0
        for g_i, (i0, i1) in enumerate(GROUPS):
            e = (spec[12 + i0:12 + i1] * (at[i0:i1] - spec[i0:i1]) ** 2).sum()
            task += spec[24 + g_i] * np.exp(-e)
        ar = ((A[b] - AP[b]) ** 2).sum()
        jl = (np.maximum(slo - qj, 0) + np.maximum(qj - shi, 0)).sum()
        uz_world = rotate(d.xquat[tb], np.array([0.0, 0.0, 1.0]))
        lean = rotate_inv(hq, uz_world)[1] ** 2                    # sin²φ, sideways only
        res = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, tb, res, 0)
        wh = rotate_inv(hq, res[:3])
        reward = DT * (task - 0.3 * ar - 2.0 * jl - spec[30] * lean - 0.05 * (wh[0] ** 2 + wh[1] ** 2))
        S[b] = np.r_[p, q, qj, v, w, qd]
        SPEC[b] = spec
        ATOMS[b] = at
        R[b, 0] = reward
        LG[b, 0] = max((0.35 - p[2]) / 0.02, (1 + g[2] - 0.7) / 0.05)
    np.save(out / "p_s.npy", S)
    np.save(out / "p_a.npy", A)
    np.save(out / "p_ap.npy", AP)
    np.save(out / "p_xi.npy", XI)
    np.save(out / "p_spec.npy", SPEC)
    np.save(out / "p_atoms.npy", ATOMS)
    np.save(out / "p_r.npy", R)
    np.save(out / "p_logit.npy", LG)
    np.save(out / "p_hands.npy", np.array([[hl - 1, hr - 1]], dtype=np.float64))
    print("wrote", B, "lanes to", out)


if __name__ == "__main__":
    main()
