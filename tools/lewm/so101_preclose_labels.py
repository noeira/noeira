"""Pre-close grasp labels for SO-101 tower stores (sim renders and real teleop).

noeira-docs/SO101_LEWM_PLAN.md §3c step 2 (B). The label is "the cube is
between the OPEN jaws: closing now would grasp". It is read off the gripper
alone, in LeRobot units (follower q = `qpos[:, 5]`, 0..100), by the SAME
rule in both domains, so a probe trained on sim labels is tested on real
labels of the same definition:

    hold      q in [HOLD_LO, HOLD_HI] for >= HOLD_TICKS ticks: the jaws
              stalled on the cube (the target need not keep squeezing: the
              sim teacher's noisy target often eases off while the jaws stay
              on the cube). Sim only: the brick (`state[:, 15]`, z) must also
              rise LIFT above its resting z within the hold + LIFT_WINDOW
              ticks, or the close is left unlabelled
    miss      q < HOLD_LO for >= EMPTY_TICKS ticks right after the jaws were
              open (q >= JAWS_OPEN within 30 ticks), with a hold LATER in the
              episode: a missed grasp that was retried (a shut after the
              release, over the bowl, is not one)

    label 1   the last POS + 1 ticks before a hold's first tick s with the
              jaws still open (q >= JAWS_OPEN, t >= s - WINDOW): the jaws are
              closing round the cube, so it is between them
    label 0   kind 2, hard negatives: the same ticks before a miss (the jaws
              open and closing, on nothing); kind 0: approach ticks
              t <= s_first - NEG_GAP (the cube still far from the jaws) every
              NEG_STRIDE ticks, from the first tick the jaws are open

⚠ NOT the close ONSET (the first tick the target closes): the real operator
starts closing with the cube 5-10 cm ahead of the jaws and the sim teacher
earlier still, so an onset label carries each policy's timing into the test.
The window before the stall is the same physical scene in both domains.

Real reference values (`lewm_so101_align.mojo`): open ~26-33, shut 0-9, on
the cube ~12-14. The sim teacher opens wider (50-70) and stalls at ~13 on
the brick (sim jaw +0.10 rad on a -0.175..1.745 range).
"""
import numpy as np

HOLD_LO, HOLD_HI, HOLD_TICKS = 8.0, 17.0, 15
LIFT, LIFT_WINDOW = 0.02, 30
BRICK_Z = 15  # the brick's z in the sim `state` column (qpos: arm 6, bowl 7, brick 7)
EMPTY_TICKS = 10
JAWS_OPEN = 18.0
POS = 5
WINDOW = 25
NEG_GAP = 40
NEG_STRIDE = 5


def _segments(mask, min_len):
    """(start, end) of runs of True at least `min_len` long."""
    out, s = [], None
    for t, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = t
        elif not m and s is not None:
            if t - s >= min_len:
                out.append((s, t))
            s = None
    return out


def _closing_window(q, s):
    """The last POS + 1 ticks before s with open jaws, within WINDOW."""
    ts = [t for t in range(max(0, s - WINDOW), s) if q[t] >= JAWS_OPEN]
    return ts[-(POS + 1):] if len(ts) >= POS + 1 else []


def episode_labels(q, brick_z=None):
    """Per-episode rows: list of (tick, label, kind); kind 0 = approach
    negative, 1 = pre-hold positive, 2 = pre-miss hard negative.
    `brick_z` (sim): holds that never lift the brick are left out."""
    q = np.asarray(q, np.float64)
    holds = _segments((q >= HOLD_LO) & (q <= HOLD_HI), HOLD_TICKS)
    if brick_z is not None and holds:
        z = np.asarray(brick_z, np.float64)
        rest = np.median(z[: max(1, holds[0][0])])
        holds = [(s, e) for s, e in holds if z[s : e + LIFT_WINDOW].max() >= rest + LIFT]
    rows, starts = [], []
    for s, _ in holds:
        w = _closing_window(q, s)
        if w:
            starts.append(s)
            rows += [(t, 1, 1) for t in w]
    if not starts:
        return []
    last_hold = max(starts)
    misses = [
        s for s, _ in _segments(q < HOLD_LO, EMPTY_TICKS)
        if s < last_hold and q[max(0, s - 30) : s].max(initial=0.0) >= JAWS_OPEN
    ]
    for s in misses:
        rows += [(t, 0, 2) for t in _closing_window(q, s)]
    opened = np.nonzero(q >= JAWS_OPEN)[0]
    t0 = int(opened[0]) if len(opened) else 0
    first = min(starts + misses)
    rows += [(t, 0, 0) for t in range(t0, first - NEG_GAP + 1, NEG_STRIDE)]
    return rows


def store_labels(f):
    """All labelled rows of an open h5py TrajectoryStore: arrays
    (row, episode, tick, label, kind)."""
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    q = f["qpos"][:, 5]
    z = f["state"][:, BRICK_Z] if "state" in f else None
    out = []
    for e in range(len(off)):
        o, n = int(off[e]), int(ln[e])
        ze = None if z is None else z[o : o + n]
        for t, y, k in episode_labels(q[o : o + n], ze):
            out.append((o + t, e, t, y, k))
    r = np.array(out, np.int64).reshape(-1, 5)
    return r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4]


if __name__ == "__main__":
    import sys
    import h5py

    for p in sys.argv[1:]:
        with h5py.File(p, "r") as f:
            row, ep, t, y, k = store_labels(f)
            n_ep = len(f["ep_offset"])
        print(
            f"{p.split('/')[-1]}: {n_ep} eps | pos {int((k == 1).sum())} in {len(set(ep[k == 1]))} eps"
            f" | approach neg {int((k == 0).sum())} | missed-grasp neg {int((k == 2).sum())}"
            f" in {len(set(ep[k == 2]))} eps"
        )
