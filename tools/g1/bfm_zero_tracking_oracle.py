"""BFM-Zero's tracking-evaluation protocol with the RELEASED actor — the G2 rung.

    # the MuJoCo column (the paper's sim-to-sim row, re-run here):
    pixi run python tools/g1/bfm_zero_tracking_oracle.py --mujoco --clips 7 25
    # the Mojo gate imports this module and drives OUR engine instead:
    pixi run mojo run -I . examples/g1/bfm_zero_tracking_gate.mojo

One module, two simulators. Everything that is NOT physics lives here and is
shared: the segment data, the initial state, the actor-side observation
(state, last action, the 4-step history in the reference's layout), the
ONNX actor call, and the metrics of `_calc_metrics` in
`humanoidverse/agents/evaluations/humanoidverse_isaac.py`. A driver only
has to (1) set `qpos`/`qvel`, (2) step 4 substeps of 1/200 s under the
reference's PD, (3) hand back `qpos`/`qvel`. So a difference between the
MuJoCo column and the Mojo column is physics, and nothing else.

THE PROTOCOL, from `_async_tracking_worker` and the released checkpoint's
`config.json` (`lafan_29dof_10s-clipped.pkl`, 862 ten-second segments):

  - A segment is a contiguous 300-frame chunk of a full LAFAN1 clip
    (verified frame for frame against `lafan_29dof.pkl`): source frames
    [300 k, 300 k + 300). Its motion length is (300 − 1) / 30 s, so it has
    T = ceil(299 / 30 / 0.02) = 499 rows at 50 Hz, which are rows
    [500 k, 500 k + 499) of the full clip in our store — same times, same
    blends up to float32 rounding of the time-to-frame map.
  - z_t = project(B(privileged obs of row t + 1)), t = 0 .. T − 2: the
    target is the NEXT frame. B depends on the frame only, so the released
    per-step z of the full clip (`zs_7.npy`, `zs_25.npy`, norm 16 = sqrt(256))
    sliced at 500 k gives the segment's z exactly.
  - Reset to row 0 of the segment: root pose + root velocities + joint
    angles + joint velocities from the reference (`ref_body_*[0]`,
    `dof_pos[0]`, `ref_dof_vel[0]`); the root angular velocity is WORLD
    frame in the reference and the MuJoCo backend rotates it into the body
    frame (`set_actor_root_state_tensor`). Then T − 1 policy steps with the
    MEAN action, no zero-action warm-up step (`reset_all` sets the state
    and computes the observation, nothing else).
  - The actor input is 465 + 256: `state` 64 = [dof − default (29), dof
    vel (29), projected gravity (3), body-frame root angular velocity × 0.25
    (3)], `last_action` 29 = the CLIPPED, ×5-SCALED action applied in the
    step that produced this state (`raw_obs["actions"]` = `self.actions`
    after `_pre_physics_step`, zero after reset), `history_actor` 372 = the
    same five quantities over the previous four steps, laid out KEY-MAJOR
    in sorted key order — actions (4 × 29), base_ang_vel (4 × 3), dof_pos
    (4 × 29), dof_vel (4 × 29), projected_gravity (4 × 3) — newest first
    within each key, and z. The history buffer is zero at reset, the reset
    observation is NEVER pushed into it, and each step's observation is
    pushed AFTER that step's history was read (`_post_physics_step`: compute
    observations, then `history_handler.add`). So the policy at step t
    sees rows t − 1 .. t − 4 with row 0 replaced by zeros.
  - Observation noise OFF and domain randomisation OFF here. The released
    CSV (`humanoidverse_tracking_eval.csv`, Isaac, 1024 envs) kept both on,
    as the paper's rows did; the paper's own Isaac-vs-MuJoCo spread on the
    same model is 2 % on tracking.
  - Metrics on the 29 joint angles over the T rows (reset row included):
    `distance` = mean_t ‖q_t − q*_t‖₂ (the paper's "tracking"), `mpjpe_l` =
    1000 × distance, `proximity` (bound 2, margin 2), `emd` = optimal
    transport between the two row sets with uniform weights (an assignment
    problem for equal sizes: `linear_sum_assignment` is `ot.emd2` here),
    `obs_state_distance` = the same distance on the first 23 joints (the
    reference kept `QVEL_IDX = 23` from the 23-DoF model), and `vel_dist` /
    `accel_dist` EXACTLY as written there: their `[:, 1:] − [:, :-1]`
    differences run along the JOINT axis of a (T, 29) array, not time.
    Reproduced as data, not corrected.

⚠ THE PD CHAIN IS `tests/robots/g1_bake.py::ReferencePD` — the same object
the G0 gate used against MuJoCo: `a × 5 → clip ±5 → × 0.25 × effort / kp +
default`, torque `kp (target − q) − kd q̇` clipped to the yaml effort, then
the actuator's own ctrlrange in the simulator (88 N m on four hips against
the yaml's 139). Not re-transcribed here.

⚠ THE ONNX INPUT IS BATCH 1 BY EXPORT ([1, 721]); one call per step, ~2 ms
on an M1 — the whole rung is under two minutes per simulator.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RELEASED = REPO / "references" / "BFM-Zero-main" / "released" / "new_model"
REF_SCENE = (
    REPO / "references" / "BFM-Zero-main" / "humanoidverse" / "data" / "robots" / "g1"
    / "scene_29dof_freebase_noadditional_actuators.xml"
)
STORE = REPO / "lafan_g1_50hz.h5"
sys.path.insert(0, str(REPO / "tests" / "robots"))

SEG_FRAMES = 300          # the clipped pickle's segment length at 30 fps
SEG_ROWS = 499            # ceil((300 - 1) / 30 / 0.02)
SEG_STRIDE = 500          # rows between segment starts in the full clip (10 s at 50 Hz)
ENV_DT = 0.02
ANG_VEL_SCALE = 0.25
NORMALIZE_TO = 5.0
ACTION_CLIP = 5.0
N_DOF = 29
Z_DIM = 256
HIST_LEN = 4
HIST_KEYS = ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")  # sorted
HIST_DIMS = {"actions": 29, "base_ang_vel": 3, "dof_pos": 29, "dof_vel": 29, "projected_gravity": 3}
STATE_DIM = 64
ACTOR_OBS_DIM = STATE_DIM + N_DOF + HIST_LEN * sum(HIST_DIMS.values()) + Z_DIM  # 721
QVEL_IDX = 23
RELEASED_ZS = {7: "zs_7.npy", 25: "zs_25.npy"}


def quat_rotate_inverse_xyzw(q, v):
    """The reference's `quat_rotate_inverse` (`a - b + c`, unit quaternion assumed)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qv, w = q[:3], q[3]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * np.dot(qv, v) * 2.0
    return a - b + c


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)



def _unescape_task_text(v: str) -> str:
    """The inverse of `manifest.escape_task_text`: strip the quotes, undo the
    five escapes, byte-wise."""
    b = v.encode("utf-8", errors="surrogateescape")
    assert len(b) >= 2 and b[0] == 0x22 and b[-1] == 0x22, v
    out = bytearray()
    i = 1
    while i < len(b) - 1:
        c = b[i]
        if c == 0x5C:
            i += 1
            n = b[i]
            out.append({0x6E: 0x0A, 0x72: 0x0D, 0x74: 0x09}.get(n, n))
        else:
            out.append(c)
        i += 1
    return out.decode("utf-8", errors="surrogateescape")


def store_clip_names(f):
    """The clip names, in episode order.

    The Python dump wrote them as a `motion_key` dataset; the native importer
    (`examples/g1/lafan_import.mojo`) records them as the store's task table
    in the manifest (`task=<clip>\t"<name>"`, `data/manifest.mojo`), the
    place a `TrajectoryStore` keeps per-index text. Either layout reads."""
    if "motion_key" in f:
        return [k.decode() for k in f["motion_key"][:]]
    text = bytes(np.asarray(f["__manifest__"][...], dtype=np.uint8)).decode("utf-8", errors="surrogateescape")
    tasks = {}
    for line in text.split("\n"):
        if not line.startswith("task="):
            continue
        idx, val = line[len("task="):].split("\t", 1)
        tasks[int(idx)] = _unescape_task_text(val)
    n = int(f["ep_len"].shape[0])
    missing = [i for i in range(n) if i not in tasks]
    assert not missing, f"the store's task table lacks clips {missing[:5]}"
    return [tasks[i] for i in range(n)]


class Protocol:
    """The store, the released z and actor, and the released CSV to compare with."""

    def __init__(self, store: Path = STORE, released: Path | None = RELEASED, with_actor: bool = True):
        """`released=None` (or `with_actor=False`): no ONNX, no released z —
        the segment data, the reset state and the metrics only, for an eval
        that brings its OWN policy and z (`examples/g1/bfm_zero_eval_tracking.mojo`).
        The released CSV is still read when present, as the comparison column."""
        self.f = h5py.File(store, "r")
        self.keys = store_clip_names(self.f)
        self.ep_off = self.f["ep_offset"][:].astype(np.int64)
        self.ep_len = self.f["ep_len"][:].astype(np.int64)
        self.default = np.asarray(self.f["default_dof_pos"][...]).reshape(-1).astype(np.float64)
        assert abs(float(np.asarray(self.f["env_dt"][...]).reshape(-1)[0]) - ENV_DT) < 1e-12
        self.released = Path(released) if released is not None else None
        self.zs = {}
        self.sess = None
        self.csv = {}
        if self.released is not None and with_actor:
            for clip, name in RELEASED_ZS.items():
                z = np.load(self.released / "exported" / name).astype(np.float32)
                assert z.shape == (self.ep_len[clip] - 1, Z_DIM), (clip, z.shape, self.ep_len[clip])
                self.zs[clip] = z
            import onnxruntime as ort

            self.sess = ort.InferenceSession(
                str(self.released / "exported" / "FBcprAuxModel.onnx"), providers=["CPUExecutionProvider"]
            )
            inp = self.sess.get_inputs()[0]
            assert list(inp.shape) == [1, ACTOR_OBS_DIM], inp.shape
        csv_path = (
            self.released / "humanoidverse_tracking_eval.csv"
            if self.released is not None
            else None
        )
        if csv_path is not None and csv_path.exists():
            self.csv = self._read_csv(csv_path)
        else:
            # ⚠ SAY SO. Without this the comparison columns print `nan / nan
            # (0)` for every clip, which reads like a join failure against the
            # released CSV rather than the file being absent — and that is
            # exactly how it was misread once (docs §12.32). `references/` is
            # a SYMLINK excluded from git (`.git/info/exclude`), so a fresh
            # checkout on a rented box has no reference data at all.
            print(
                "  ⚠ released tracking CSV NOT FOUND at "
                + str(csv_path)
                + "\n    the MuJoCo / Isaac comparison columns will be EMPTY"
                " (`nan / nan (0)`) — this is a MISSING FILE, not a"
                " disagreement.\n    `references/` is a symlink excluded from"
                " git; copy it to the box, or score with `--out <file>` and"
                " join against the CSV where it lives."
            )

    @staticmethod
    def _read_csv(path):
        rows = list(csv.DictReader(open(path)))
        tmax = max(int(r["timestep"]) for r in rows)
        return {r["motion_file"]: r for r in rows if int(r["timestep"]) == tmax}

    def clips(self):
        return sorted(self.zs.keys())

    def n_segments(self, clip):
        """Segments of this clip: the CSV's own count when it lists the clip,
        else every 500-row start whose 499 rows fit — the same number for
        every clip the CSV covers (checked on all 40)."""
        key = self.keys[clip]
        n = sum(1 for k in self.csv if k.startswith(key + "_clip"))
        if n == 0:
            n = int((self.ep_len[clip] - SEG_ROWS) // SEG_STRIDE) + 1
        assert n > 0, key
        assert (n - 1) * SEG_STRIDE + SEG_ROWS <= self.ep_len[clip], (key, n, self.ep_len[clip])
        return n

    def all_clips(self):
        return list(range(len(self.keys)))

    def csv_row(self, clip, seg):
        return self.csv.get(f"{self.keys[clip]}_clip{seg}")

    def episode(self, clip, seg):
        return Episode(self, clip, seg)

    def actor(self, obs):
        x = np.ascontiguousarray(obs, dtype=np.float32).reshape(1, ACTOR_OBS_DIM)
        return self.sess.run(None, {"actor_obs": x})[0][0].astype(np.float64)


class Episode:
    """One segment: the initial state, the observation pipeline, the record, the metrics."""

    def __init__(self, proto: Protocol, clip: int, seg: int):
        self.proto = proto
        self.clip, self.seg = clip, seg
        r0 = int(proto.ep_off[clip] + seg * SEG_STRIDE)
        self.T = SEG_ROWS
        self.qpos_ref = proto.f["qpos"][r0 : r0 + self.T].astype(np.float64)
        self.qvel_ref = proto.f["qvel"][r0 : r0 + self.T].astype(np.float64)
        if clip in proto.zs:
            self.z = proto.zs[clip][seg * SEG_STRIDE : seg * SEG_STRIDE + self.T - 1]
            assert self.z.shape[0] == self.T - 1
        else:
            self.z = None  # the caller's (`set_z`) — an eval with its own B
        self.target = self.qpos_ref[:, 7:].copy()  # (T, 29) joint angles
        self.default = proto.default
        self.last_action = np.zeros(N_DOF)
        self.hist = {k: np.zeros((HIST_LEN, HIST_DIMS[k])) for k in HIST_KEYS}
        self.joint_pos = []
        self.t = 0

    def set_z(self, z_flat):
        """z for steps 0 .. T−2 as a flat list of (T−1)·256 floats."""
        z = np.asarray([float(v) for v in z_flat], dtype=np.float32).reshape(self.T - 1, Z_DIM)
        self.z = z

    def first_row(self):
        """The store row index of this segment's first row."""
        return int(self.proto.ep_off[self.clip] + self.seg * SEG_STRIDE)

    # ── state ────────────────────────────────────────────────────────────
    def init_state(self):
        """(qpos 36, qvel 35) of row 0, root angular velocity in the BODY frame
        (the store keeps the reference's world-frame estimate)."""
        qpos = self.qpos_ref[0].copy()
        qvel = self.qvel_ref[0].copy()
        qvel[3:6] = quat_rotate_inverse_xyzw(wxyz_to_xyzw(qpos[3:7]), qvel[3:6])
        return qpos, qvel

    def state_from(self, qpos, qvel):
        """The 64-D proprio `state`, in the reference's order and scale, noise off."""
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        q_xyzw = wxyz_to_xyzw(qpos[3:7])
        parts = {
            "dof_pos": qpos[7:] - self.default,
            "dof_vel": qvel[6:],
            "projected_gravity": quat_rotate_inverse_xyzw(q_xyzw, np.array([0.0, 0.0, -1.0])),
            "base_ang_vel": qvel[3:6] * ANG_VEL_SCALE,
        }
        state = np.concatenate([parts["dof_pos"], parts["dof_vel"], parts["projected_gravity"], parts["base_ang_vel"]])
        return state, parts

    def actor_obs(self, qpos, qvel):
        """The 721-vector the ONNX actor consumes at this step, and the raw parts."""
        assert self.t < self.T - 1, "episode is over"
        state, parts = self.state_from(qpos, qvel)
        hist = np.concatenate([self.hist[k].reshape(-1) for k in HIST_KEYS])
        return np.concatenate([state, self.last_action, hist, self.z[self.t]]), parts

    def act(self, qpos, qvel):
        """Policy step t: returns the RAW actor output (what the driver hands to
        the PD chain) and advances the observation state the way the env does."""
        obs, parts = self.actor_obs(qpos, qvel)
        a = self.proto.actor(obs)
        if self.t >= 1:  # the reset observation is never pushed
            self._push(parts)
        self.last_action = np.clip(a * NORMALIZE_TO, -ACTION_CLIP, ACTION_CLIP)
        return a

    def _push(self, parts):
        for k in HIST_KEYS:
            buf = self.hist[k]
            buf[1:] = buf[:-1]
            buf[0] = self.last_action if k == "actions" else parts[k]

    def record(self, qpos):
        """Call once with the reset state, then after every step."""
        self.joint_pos.append(np.asarray(qpos, dtype=np.float64)[7:].copy())
        if len(self.joint_pos) > 1:
            self.t += 1

    # ── metrics ──────────────────────────────────────────────────────────
    def metrics(self):
        jp = np.stack(self.joint_pos)
        assert jp.shape == self.target.shape, (jp.shape, self.target.shape)
        return compute_metrics(jp, self.target)


def _emd(x, y):
    from scipy.optimize import linear_sum_assignment

    cost = np.sqrt(np.maximum(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1), 0.0))
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def _proximity(x, y, bound=2.0, margin=2.0):
    dist = np.linalg.norm(x - y, axis=-1)
    inb = dist <= bound
    outb = dist > bound + margin
    prox = inb + ((bound + margin - dist) / margin) * (~inb) * (~outb)
    return float(prox.mean()), float(dist.mean())


def compute_metrics(joint_pos, target):
    """`_calc_metrics` + `compute_joint_pos_metrics`, formula for formula."""
    m = {}
    prox, dist = _proximity(joint_pos, target)
    m["distance"] = dist
    m["mpjpe_l"] = dist * 1000.0
    m["proximity"] = prox
    m["emd"] = _emd(joint_pos, target)
    # their velocity / acceleration terms difference along axis 1 of a (T, D) array
    vel_gt = target[:, 1:] - target[:, :-1]
    vel_pred = joint_pos[:, 1:] - joint_pos[:, :-1]
    m["vel_dist"] = float(np.linalg.norm(vel_pred - vel_gt, axis=-1).mean() * 1000.0)
    acc_gt = target[:, :-2] - 2 * target[:, 1:-1] + target[:, 2:]
    acc_pred = joint_pos[:, :-2] - 2 * joint_pos[:, 1:-1] + joint_pos[:, 2:]
    m["accel_dist"] = float(np.linalg.norm(acc_pred - acc_gt, axis=-1).mean() * 100.0)
    # `obs_state_*`: the first 23 entries of `state`, i.e. dof - default; the
    # default cancels in the difference, so on the raw angles directly.
    p23, d23 = _proximity(joint_pos[:, :QVEL_IDX], target[:, :QVEL_IDX])
    m["obs_state_distance"] = d23
    m["obs_state_proximity"] = p23
    m["obs_state_emd"] = _emd(joint_pos[:, :QVEL_IDX], target[:, :QVEL_IDX])
    return m


# ── the MuJoCo driver: the paper's sim-to-sim row, re-run ────────────────────
def run_mujoco(proto: Protocol, clip: int, seg: int, timestep: float = 0.005):
    import mujoco
    from g1_bake import ReferencePD

    m = mujoco.MjModel.from_xml_path(str(REF_SCENE))
    m.opt.timestep = timestep  # the runtime override the reference applies (1 / sim.fps)
    d = mujoco.MjData(m)
    pd = ReferencePD()
    ep = proto.episode(clip, seg)
    qpos, qvel = ep.init_state()
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(m, d)
    ep.record(d.qpos)
    for _ in range(ep.T - 1):
        a = ep.act(d.qpos.copy(), d.qvel.copy())
        pd.control_step(mujoco, m, d, a)
        ep.record(d.qpos)
    return ep.metrics()


class Tally:
    """Per-segment rows from a driver, the released Isaac row beside each, and
    an optional MuJoCo column (the CSV `--out` of this module) — so the Mojo
    gate prints one table with three simulators and gates on the means."""

    def __init__(self, proto: Protocol, mujoco_csv: str | None = None):
        self.proto = proto
        self.rows = []
        self.mujoco = {}
        if mujoco_csv and Path(mujoco_csv).exists():
            for r in csv.DictReader(open(mujoco_csv)):
                self.mujoco[(int(r["clip"]), int(r["seg"]))] = {
                    k[len("mujoco_"):]: float(v) for k, v in r.items() if k.startswith("mujoco_")
                }

    def add(self, clip, seg, m):
        ref = self.proto.csv_row(clip, seg)
        mj = self.mujoco.get((int(clip), int(seg)))
        self.rows.append((int(clip), int(seg), dict(m), ref, mj))
        s = f"  seg {int(seg):2d}  ours distance {m['distance']:.3f} emd {m['emd']:.3f} prox {m['proximity']:.4f}"
        if mj is not None:
            s += f"  | MuJoCo {mj['distance']:.3f} / {mj['emd']:.3f}"
        if ref is not None:
            s += f"  | Isaac (released) {float(ref['distance']):.3f} / {float(ref['emd']):.3f}"
        return s

    def _col(self, c, column):
        """One comparison column, or `absent` when no row was found.

        `nan / nan (0)` is indistinguishable from a broken join; say which it
        is, at the only place a reader looks.
        """
        n = self.count(c, column)
        if n == 0:
            return "absent"
        return (
            f"{self.mean(c, column=column):.3f} /"
            f" {self.mean(c, 'emd', column):.3f} ({n})"
        )

    def mean(self, clip, key="distance", column="ours"):
        vals = []
        for c, s, m, ref, mj in self.rows:
            if c != int(clip):
                continue
            if column == "ours":
                vals.append(m[key])
            elif column == "mujoco" and mj is not None:
                vals.append(mj[key])
            elif column == "isaac" and ref is not None:
                vals.append(float(ref[key]))
        return float(np.mean(vals)) if vals else float("nan")

    def count(self, clip, column="ours"):
        return sum(
            1 for c, s, m, ref, mj in self.rows
            if c == int(clip) and (column == "ours" or (column == "mujoco" and mj is not None) or (column == "isaac" and ref is not None))
        )

    def write_csv(self, path):
        """Per-segment rows: ours, the MuJoCo column and the released Isaac
        number where present."""
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["clip", "key", "seg", "distance", "emd", "proximity", "obs_state_distance",
                        "mujoco_distance", "mujoco_emd", "isaac_distance", "isaac_emd"])
            for c, sg, m, ref, mj in self.rows:
                w.writerow([c, self.proto.keys[c], sg, m["distance"], m["emd"], m["proximity"], m["obs_state_distance"],
                            mj["distance"] if mj else "", mj["emd"] if mj else "",
                            float(ref["distance"]) if ref else "", float(ref["emd"]) if ref else ""])
        return len(self.rows)

    def report(self, clip):
        c = int(clip)
        return (
            f"  MEAN clip {c} {self.proto.keys[c]} over {self.count(c)} segments:"
            f"  ours distance {self.mean(c):.3f} emd {self.mean(c, 'emd'):.3f}"
            f"  | MuJoCo {self._col(c, 'mujoco')}"
            f"  | Isaac (released) {self._col(c, 'isaac')}"
        )


def format_row(label, m, ref):
    s = f"  {label:8s} distance {m['distance']:.3f}  emd {m['emd']:.3f}  prox {m['proximity']:.4f}  obs23 {m['obs_state_distance']:.3f}"
    if ref is not None:
        s += f"   | released Isaac: distance {float(ref['distance']):.3f}  emd {float(ref['emd']):.3f}  prox {float(ref['proximity']):.4f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mujoco", action="store_true", help="run the MuJoCo driver")
    ap.add_argument("--clips", type=int, nargs="*", default=None)
    ap.add_argument("--segments", type=int, nargs="*", default=None, help="restrict to these segment indices")
    ap.add_argument("--out", type=str, default=None, help="write the per-segment table as CSV")
    args = ap.parse_args()
    proto = Protocol()
    clips = args.clips or proto.clips()
    if not args.mujoco:
        for c in clips:
            print(f"clip {c} {proto.keys[c]}: {proto.ep_len[c]} rows, {proto.n_segments(c)} segments, z {proto.zs[c].shape}")
        return
    rows = []
    for c in clips:
        segs = args.segments if args.segments is not None else range(proto.n_segments(c))
        print(f"clip {c} {proto.keys[c]}: {len(list(segs))} segments (MuJoCo, DR off, noise off)")
        for s in segs:
            t0 = time.perf_counter()
            m = run_mujoco(proto, c, s)
            ref = proto.csv_row(c, s)
            print(format_row(f"seg {s:2d}", m, ref) + f"  [{time.perf_counter() - t0:.1f} s]")
            rows.append({"clip": c, "key": proto.keys[c], "seg": s, **{f"mujoco_{k}": v for k, v in m.items()},
                         **({f"isaac_{k}": float(ref[k]) for k in ("distance", "emd", "proximity")} if ref else {})})
        d = np.array([r["mujoco_distance"] for r in rows if r["clip"] == c])
        e = np.array([r["mujoco_emd"] for r in rows if r["clip"] == c])
        di = np.array([r["isaac_distance"] for r in rows if r["clip"] == c and "isaac_distance" in r])
        ei = np.array([r["isaac_emd"] for r in rows if r["clip"] == c and "isaac_emd" in r])
        print(f"  MEAN  mujoco distance {d.mean():.3f} emd {e.mean():.3f}   released Isaac distance {di.mean():.3f} emd {ei.mean():.3f}")
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
