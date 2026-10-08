"""SO-101 world-model training windows from rendered TrajectoryStores.

noeira-docs/SO101_LEWM_PLAN.md S1. Stores come from
`tower_demo_rerender.mojo --resize R` over the teacher rollouts recorded by
`ppo_state_probe_sim.mojo --demos` (S0): per row `images` (2 × 3 × R × R u8,
slot 0 overhead, slot 1 wrist), `state` (the sim's qpos + qvel, f64) and
`action_sim` (the COMMANDED target, normalised onto each actuator's
ctrlrange).

A window is the LeWM one with the rig's timing:

    frames   T = 4 rows, FS = 5 ticks apart (160 ms at 31 Hz)
    pixels   (T, 6, R, R) float — the two cameras stacked as channels,
             ImageNet-normalised per RGB channel (LeWM's ToImage+Normalize)
    actions  (T, FS × 6) float — for frame t, the FS per-tick CHANGES of the
             commanded target (radians) from tick r + t·FS on, z-scored per
             joint with the dataset's statistics

The action is the change of the commanded target, not the policy's raw
[-1, 1] word: it is what the arm was told after target mode's clamps (the
lead and the ctrlrange), and the critic can compute it for any proposal
(`delta_action.target_step`). The first tick of an episode changes the
target from the arm's own pose (target mode starts at q). Windows stay inside
one episode; the split is by episode (`VAL_EVERY`-th episode of each store
held out).

## Real teleop stores (`teleop=True`, S2)

`act_so101_import_dataset.mojo --undistort` stores: `images`
(2 × 3 × 240 × 320, undistorted to the sim pinhole), `qpos` and `action` in
LeRobot units — the follower's joints and the LEADER's. Frames go through the
same `AreaResize` as the sim renders; the commanded target is the leader's
position (the follower tracks it), in model radians through
`So101TowerUnits` with the FOLLOWER joint zero; the first tick's change is
from the follower's pose. ⚠ teleop has no target-mode clamp (lead ≤ 0.3 rad
per joint): its target changes can be larger than any the sim teacher made.
Normalise real windows with the TRAINING run's statistics
(`load_action_stats`), never their own.
"""

from std.math import sqrt
from std.random import random_ui64

from noeira.nn.constants import DT
from noeira.data.store import TrajectoryStore
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.family import scene_path
from noeira.tasks.spec import load_family
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.so101_tower_rig import So101TowerUnits, RIG_JOINT_ZERO_FOLLOWER
from .so101_frames import AreaResize


comptime T = 4
comptime FS = 5
comptime JOINTS = 6
comptime ACT_IN = FS * JOINTS
comptime CAMS = 2
comptime VAL_EVERY = 10
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"


struct So101Rig(Movable):
    """The arm's qpos addresses and actuator ranges, from the family model."""

    var qa: List[Int]
    var lo: List[Float64]
    var hi: List[Float64]

    def __init__(out self) raises:
        var f = load_family(String(FAMILY_PATH))
        var fmd = parse_model_runtime(scene_path(f))
        var jadr = List[Int]()
        var acc = 0
        for i in range(len(fmd.joints)):
            jadr.append(acc)
            acc += fmd.joints[i].nq
        self.qa = List[Int]()
        self.lo = List[Float64]()
        self.hi = List[Float64]()
        for i in range(JOINTS):
            self.qa.append(jadr[fmd.actuators[i].joint_id])
            self.lo.append(fmd.actuators[i].ctrl_min)
            self.hi.append(fmd.actuators[i].ctrl_max)


struct So101WMData[R: Int](Movable):
    """Every row of the given stores, resident, with its window starts."""

    comptime FRAME = CAMS * 3 * Self.R * Self.R

    var images: List[UInt8]
    var dtarget: List[Float32]
    """(rows, 6) per-tick change of the commanded target, radians."""
    var a_mean: List[Float64]
    var a_std: List[Float64]
    var train_starts: List[Int]
    var val_starts: List[Int]
    var n_rows: Int
    var n_episodes: Int

    def __init__(out self, paths: List[String], teleop: Bool = False) raises:
        var rig = So101Rig()
        var units = So101TowerUnits(String(RIG_JOINT_ZERO_FOLLOWER))
        self.images = List[UInt8]()
        self.dtarget = List[Float32]()
        self.train_starts = List[Int]()
        self.val_starts = List[Int]()
        self.n_rows = 0
        self.n_episodes = 0
        var nqv = So101TowerModel.NQ + So101TowerModel.NV
        for p in paths:
            var st = TrajectoryStore(p)
            var spec = st.column(String("images"))
            var n = st.n_rows()
            var base = self.n_rows
            var tgt = List[Float64](length=n * JOINTS, fill=0.0)
            var q0 = List[Float64](length=n * JOINTS, fill=0.0)
            if teleop:
                if spec.row_dim() != CAMS * 3 * 240 * 320:
                    raise Error("So101WMData: " + p + " is not a 2 x 3 x 240 x 320 teleop store")
                # streamed: a teleop store is GBs (7.5 for the printed set)
                var rz = AreaResize(3, 240, 320, Self.R)
                comptime CHUNK = 256
                comptime FULL = CAMS * 3 * 240 * 320
                var buf = List[UInt8](length=CHUNK * FULL, fill=UInt8(0))
                var r0 = 0
                while r0 < n:
                    var r1 = min(n, r0 + CHUNK)
                    st.read_range[DType.uint8](
                        String("images"), r0, r1,
                        rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](buf.unsafe_ptr()),
                    )
                    var small = rz.frames(buf, (r1 - r0) * CAMS)
                    self.images.extend(small^)
                    r0 = r1
                var qp = st.load_column[DType.float32](String("qpos"), max_bytes=1 << 34)
                var ac = st.load_column[DType.float32](String("action"), max_bytes=1 << 34)
                for r in range(n):
                    for j in range(JOINTS):
                        tgt[r * JOINTS + j] = units.lerobot_to_joint(j, Float64(ac[r * JOINTS + j]))
                        q0[r * JOINTS + j] = units.lerobot_to_joint(j, Float64(qp[r * JOINTS + j]))
            else:
                if spec.row_dim() != Self.FRAME:
                    raise Error("So101WMData: " + p + " images row is " + String(spec.row_dim())
                                + " values, expected " + String(Self.FRAME)
                                + " (2 x 3 x R x R; render with --resize R)")
                var im = st.load_column[DType.uint8](String("images"), max_bytes=1 << 40)
                self.images.extend(im^)
                var state = st.load_column[DType.float64](String("state"), max_bytes=1 << 34)
                var asim = st.load_column[DType.float32](String("action_sim"), max_bytes=1 << 34)
                for r in range(n):
                    for j in range(JOINTS):
                        var mid = 0.5 * (rig.lo[j] + rig.hi[j])
                        var half = 0.5 * (rig.hi[j] - rig.lo[j])
                        tgt[r * JOINTS + j] = mid + half * Float64(asim[r * JOINTS + j])
                        q0[r * JOINTS + j] = Float64(state[r * nqv + rig.qa[j]])
            for e in range(st.n_episodes()):
                var off = Int(st.episodes.ep_offset[e])
                var ln = Int(st.episodes.ep_len[e])
                for r in range(off, off + ln):
                    for j in range(JOINTS):
                        var prev = q0[r * JOINTS + j] if r == off else tgt[(r - 1) * JOINTS + j]
                        self.dtarget.append(Float32(tgt[r * JOINTS + j] - prev))
                # windows: T frames FS apart and their T action blocks
                var last = off + ln - T * FS
                var val = (self.n_episodes % VAL_EVERY) == VAL_EVERY - 1
                for r in range(off, last + 1):
                    if val:
                        self.val_starts.append(base + r)
                    else:
                        self.train_starts.append(base + r)
                self.n_episodes += 1
            self.n_rows += n
        # action statistics over every row
        self.a_mean = List[Float64](length=JOINTS, fill=0.0)
        self.a_std = List[Float64](length=JOINTS, fill=0.0)
        for r in range(self.n_rows):
            for j in range(JOINTS):
                self.a_mean[j] += Float64(self.dtarget[r * JOINTS + j])
        for j in range(JOINTS):
            self.a_mean[j] /= Float64(self.n_rows)
        for r in range(self.n_rows):
            for j in range(JOINTS):
                self.a_std[j] += (Float64(self.dtarget[r * JOINTS + j]) - self.a_mean[j]) ** 2
        for j in range(JOINTS):
            self.a_std[j] = max(sqrt(self.a_std[j] / Float64(self.n_rows)), 1e-6)

    def load_action_stats(mut self, path: String) raises:
        """Replace the normaliser with a training run's `action_stats.txt`
        (one `mean std` line per joint after the header)."""
        var j = 0
        with open(path, "r") as f:
            for ln in f.read().split("\n"):
                var s = String(ln.strip())
                if s.byte_length() == 0 or s.startswith("#"):
                    continue
                var w = s.split(" ")
                self.a_mean[j] = Float64(String(w[0]))
                self.a_std[j] = Float64(String(w[1]))
                j += 1
        if j != JOINTS:
            raise Error("load_action_stats: " + path + " has " + String(j) + " joints")

    def fill(
        self, starts: List[Int], mut pix: List[Scalar[DT]], mut act: List[Scalar[DT]]
    ):
        """Windows starting at `starts` (global rows) -> pixels
        (B, T, 6, R, R) and actions (B, T, 30)."""
        comptime HW = Self.R * Self.R
        var mean: List[Float64] = [0.485, 0.456, 0.406]
        var std: List[Float64] = [0.229, 0.224, 0.225]
        var b = 0
        for r0 in starts:
            for t in range(T):
                var row = r0 + t * FS
                var src = row * Self.FRAME
                var dst = ((b * T + t) * CAMS * 3) * HW
                for ch in range(CAMS * 3):
                    var c = ch % 3
                    var mu = mean[c]
                    var sd = std[c]
                    for p in range(HW):
                        var x = Float64(Int(self.images[src + ch * HW + p])) / 255.0
                        pix[dst + ch * HW + p] = Scalar[DT]((x - mu) / sd)
                for k in range(FS):
                    for j in range(JOINTS):
                        var v = (Float64(self.dtarget[(row + k) * JOINTS + j]) - self.a_mean[j]) / self.a_std[j]
                        act[((b * T + t) * FS + k) * JOINTS + j] = Scalar[DT](v)
            b += 1

    def sample(self, val: Bool, n: Int) -> List[Int]:
        var out = List[Int](capacity=n)
        for _ in range(n):
            if val:
                out.append(self.val_starts[Int(random_ui64(0, UInt64(len(self.val_starts) - 1)))])
            else:
                out.append(self.train_starts[Int(random_ui64(0, UInt64(len(self.train_starts) - 1)))])
        return out^
