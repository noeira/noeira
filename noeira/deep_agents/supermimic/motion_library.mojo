# +--------------------------------------------------------------------------+ #
# | SuperMimic M2: the motion library and the return-based reset sampler
# +--------------------------------------------------------------------------+ #
"""The reproduction's compact `MotionLibrary` (`envs/mdp/motion_library.py`)
and the reset sampler of its `MultiClipMotionCommand` (`envs/mdp/commands.py`,
plan D14), on our G1.

THE LIBRARY. Clips are concatenated on one global frame axis; a frame holds
the 71-d reference row `[root pos, root quat WXYZ, q, root lin vel (world),
root ang vel (world), q̇]` (`data/bones_seed_import.mojo`'s store). Clips are
cut into bins of 50 frames (1 s), numbered globally; the last bin of a clip
may be short. Per frame, on the fly:

    goal features (68)  [q − q0, q̇, R⁻¹ g, R⁻¹ ω, R_yaw⁻¹ v, root z]
    goal window (680)   the features of frames f .. f+9, clamped at the clip end
    reference row (189) the 14 tracked bodies' world pos / quat / lin vel /
                        ang vel by OUR FK of the reference state, then the
                        reference root pos and quat (the heads' `ref` layout)

THE SAMPLER. A reset draws a bin from

    p = 0.5 / n_bins + 0.4 · softmax(−25 · G_b / G_max) + 0.1 · memory / Σ memory
    memory ← 0.99 · memory + 0.01 · softmax(…)          (updated at every draw)

then a frame uniformly inside the bin, kept ≥ 2 frames before its clip's end.
G_b is an EMA (α = 0.05) of the discounted (γ = 0.99) return-to-go earned
from that bin, folded in when episodes end; a TRUNCATED episode (timed out,
not terminated, not at its clip's end) contributes only the steps at least
100 steps before the truncation (no critic bootstrap, the reproduction's
default). G starts at G_max = 10 everywhere.

Precision: float64 here; the reproduction accumulates the bin sums and the
EMA in float32 (its `bin_return` is a float32 tensor), ~1e-7 apart.
"""

from std.math import ceil, exp, log

from noeira.deep_agents.supermimic.g1_fk import FP, G1FKTree, Q4, g1_fk_lane, qapply
from noeira.deep_agents.supermimic.heads import (
    G1HeadsTable, HT_TRACKED, SM_N_TRACKED, SM_REF_BODY, SM_REF_DIM, qapply_inv, yaw_quat,
)
from noeira.nn.constants import DT


comptime SM_GOAL_FEAT: Int = 68
comptime SM_GOAL_K: Int = 10
comptime SM_BIN_FRAMES: Int = 50
comptime SM_REF_ROW: Int = 71


struct SMMotionLibrary(Movable):
    var rows: List[Scalar[DT]]         # num_frames × 71
    var q0: List[Scalar[DT]]           # 29
    var clip_start: List[Int]
    var clip_len: List[Int]
    var frame_clip: List[Int]
    var bin_start: List[Int]
    var bin_end: List[Int]             # exclusive
    var bin_clip: List[Int]
    var clip_bin_offset: List[Int]
    var num_frames: Int
    var num_clips: Int
    var num_bins: Int

    def __init__(out self, var rows: List[Scalar[DT]], clip_len: List[Int], q0: List[Float64]) raises:
        """From concatenated reference rows and each clip's frame count."""
        var total = 0
        for c in range(len(clip_len)):
            if clip_len[c] < 1:
                raise Error("SMMotionLibrary: clip " + String(c) + " is empty")
            total += clip_len[c]
        if total * SM_REF_ROW != len(rows):
            raise Error("SMMotionLibrary: " + String(len(rows)) + " values for " + String(total) + " frames")
        if len(q0) != 29:
            raise Error("SMMotionLibrary: q0 has " + String(len(q0)) + " entries")
        self.rows = rows^
        self.q0 = List[Scalar[DT]](length=29, fill=0)
        for j in range(29):
            self.q0[j] = Scalar[DT](q0[j])
        self.num_frames = total
        self.num_clips = len(clip_len)
        self.clip_start = List[Int]()
        self.clip_len = clip_len.copy()
        self.frame_clip = List[Int](length=total, fill=0)
        self.bin_start = List[Int]()
        self.bin_end = List[Int]()
        self.bin_clip = List[Int]()
        self.clip_bin_offset = List[Int]()
        var st = 0
        for c in range(self.num_clips):
            self.clip_start.append(st)
            self.clip_bin_offset.append(len(self.bin_start))
            for f in range(st, st + clip_len[c]):
                self.frame_clip[f] = c
            var nb = (clip_len[c] + SM_BIN_FRAMES - 1) // SM_BIN_FRAMES
            for k in range(nb):
                self.bin_start.append(st + k * SM_BIN_FRAMES)
                self.bin_end.append(min(st + (k + 1) * SM_BIN_FRAMES, st + clip_len[c]))
                self.bin_clip.append(c)
            st += clip_len[c]
        self.num_bins = len(self.bin_start)

    def __init__(out self, *, deinit move: Self):
        self.rows = move.rows^
        self.q0 = move.q0^
        self.clip_start = move.clip_start^
        self.clip_len = move.clip_len^
        self.frame_clip = move.frame_clip^
        self.bin_start = move.bin_start^
        self.bin_end = move.bin_end^
        self.bin_clip = move.bin_clip^
        self.clip_bin_offset = move.clip_bin_offset^
        self.num_frames = move.num_frames
        self.num_clips = move.num_clips
        self.num_bins = move.num_bins

    def last_frame_of(self, f: Int) -> Int:
        var c = self.frame_clip[f]
        return self.clip_start[c] + self.clip_len[c] - 1

    def bin_of_frame(self, f: Int) -> Int:
        var c = self.frame_clip[f]
        return self.clip_bin_offset[c] + (f - self.clip_start[c]) // SM_BIN_FRAMES

    @always_inline
    def _row(mut self, f: Int) -> FP[DT]:
        return self.rows.unsafe_ptr().unsafe_offset(f * SM_REF_ROW).as_unsafe_any_origin()

    def goal_features(mut self, f: Int, dst: FP[DT]):
        """`goal_features(ref(f), q0)` (68) into `dst`."""
        var r = self._row(f)
        var q = Q4[DT](r[unsafe_offset=3], r[unsafe_offset=4], r[unsafe_offset=5], r[unsafe_offset=6])
        var grav = qapply_inv(q, Q4[DT](0, 0, -1, 0))
        var ang_b = qapply_inv(q, Q4[DT](r[unsafe_offset=39], r[unsafe_offset=40], r[unsafe_offset=41], 0))
        var lin_h = qapply_inv(yaw_quat(q), Q4[DT](r[unsafe_offset=36], r[unsafe_offset=37], r[unsafe_offset=38], 0))
        for j in range(29):
            dst[unsafe_offset=j] = r[unsafe_offset=7 + j] - self.q0[j]
            dst[unsafe_offset=29 + j] = r[unsafe_offset=42 + j]
        for c in range(3):
            dst[unsafe_offset=58 + c] = grav[c]
            dst[unsafe_offset=61 + c] = ang_b[c]
            dst[unsafe_offset=64 + c] = lin_h[c]
        dst[unsafe_offset=67] = r[unsafe_offset=2]

    def goal_window(mut self, f: Int, dst: FP[DT]):
        """(680) features of f .. f+9, clamped at the clip's last frame."""
        var last = self.last_frame_of(f)
        for k in range(SM_GOAL_K):
            self.goal_features(min(f + k, last), dst.unsafe_offset(k * SM_GOAL_FEAT))

    def ref_qpos_qvel(mut self, f: Int, qpos: FP[DT], qvel: FP[DT]):
        """MuJoCo layout: qvel's root angular velocity in the BODY frame."""
        var r = self._row(f)
        for k in range(36):
            qpos[unsafe_offset=k] = r[unsafe_offset=k]
        var q = Q4[DT](r[unsafe_offset=3], r[unsafe_offset=4], r[unsafe_offset=5], r[unsafe_offset=6])
        var ang_b = qapply_inv(q, Q4[DT](r[unsafe_offset=39], r[unsafe_offset=40], r[unsafe_offset=41], 0))
        for k in range(3):
            qvel[unsafe_offset=k] = r[unsafe_offset=36 + k]
            qvel[unsafe_offset=3 + k] = ang_b[k]
        for j in range(29):
            qvel[unsafe_offset=6 + j] = r[unsafe_offset=42 + j]

    def reference_row(mut self, f: Int, mut tree: G1FKTree[DT], mut table: G1HeadsTable[DT], dst: FP[DT]):
        """The heads' `ref` (189) at frame f: tracked bodies by our FK of the
        reference state, then the library's root pos and quat."""
        var qpos = List[Scalar[DT]](length=36, fill=0)
        var qvel = List[Scalar[DT]](length=35, fill=0)
        var fk = List[Scalar[DT]](length=40 * 13, fill=0)
        var qp = qpos.unsafe_ptr().as_unsafe_any_origin()
        var qv = qvel.unsafe_ptr().as_unsafe_any_origin()
        self.ref_qpos_qvel(f, qp, qv)
        var base = fk.unsafe_ptr().as_unsafe_any_origin()
        var pos = base
        var quat = base.unsafe_offset(40 * 3)
        var lin = base.unsafe_offset(40 * 7)
        var ang = base.unsafe_offset(40 * 10)
        g1_fk_lane[DT](tree.ptr(), qp, qv, pos, quat, lin, ang)
        for k in range(SM_N_TRACKED):
            var b = Int(table.data[HT_TRACKED + k])
            var o = k * SM_REF_BODY
            for c in range(3):
                dst[unsafe_offset=o + c] = pos[unsafe_offset=b * 3 + c]
                dst[unsafe_offset=o + 7 + c] = lin[unsafe_offset=b * 3 + c]
                dst[unsafe_offset=o + 10 + c] = ang[unsafe_offset=b * 3 + c]
            for c in range(4):
                dst[unsafe_offset=o + 3 + c] = quat[unsafe_offset=b * 4 + c]
        var r = self._row(f)
        var rr = SM_N_TRACKED * SM_REF_BODY
        for c in range(7):
            dst[unsafe_offset=rr + c] = r[unsafe_offset=c]


# ── the reset sampler ───────────────────────────────────────────────────────
comptime SM_UNIFORM_RATIO: Float64 = 0.5
comptime SM_HARD_RATIO: Float64 = 0.4
comptime SM_MEMORY_RATIO: Float64 = 0.1
comptime SM_SOFTMAX_TEMP: Float64 = 25.0
comptime SM_G_MAX: Float64 = 10.0
comptime SM_SAMPLER_GAMMA: Float64 = 0.99
comptime SM_EMA_ALPHA: Float64 = 0.05
comptime SM_MEMORY_ALPHA: Float64 = 0.01
comptime SM_MIN_REMAINING: Int = 100


struct SMEpisodeLog(Copyable, Movable):
    """One episode's simulator rewards and the bin each was earned in."""
    var rewards: List[Float64]
    var bins: List[Int]
    var terminated: Bool
    var clip_end: Bool

    def __init__(out self):
        self.rewards = List[Float64]()
        self.bins = List[Int]()
        self.terminated = False
        self.clip_end = False


struct SMResetSampler(Movable):
    var bin_return: List[Float64]
    var bin_memory: List[Float64]

    def __init__(out self, num_bins: Int):
        self.bin_return = List[Float64](length=num_bins, fill=SM_G_MAX)
        self.bin_memory = List[Float64](length=num_bins, fill=1.0 / Float64(num_bins))

    def __init__(out self, *, deinit move: Self):
        self.bin_return = move.bin_return^
        self.bin_memory = move.bin_memory^

    def probabilities(mut self) -> List[Float64]:
        """The bin distribution of one draw; updates the memory (as every
        `_sample_frames` call does, once per batch of resets)."""
        var nb = len(self.bin_return)
        var hard = List[Float64](length=nb, fill=0.0)
        var mx = -1e300
        for b in range(nb):
            mx = max(mx, -SM_SOFTMAX_TEMP * self.bin_return[b] / SM_G_MAX)
        var z = 0.0
        for b in range(nb):
            hard[b] = exp(-SM_SOFTMAX_TEMP * self.bin_return[b] / SM_G_MAX - mx)
            z += hard[b]
        var msum = 0.0
        for b in range(nb):
            hard[b] /= z
            self.bin_memory[b] = (1.0 - SM_MEMORY_ALPHA) * self.bin_memory[b] + SM_MEMORY_ALPHA * hard[b]
            msum += self.bin_memory[b]
        var p = List[Float64](length=nb, fill=0.0)
        var ps = 0.0
        for b in range(nb):
            p[b] = SM_UNIFORM_RATIO / Float64(nb) + SM_HARD_RATIO * hard[b] + SM_MEMORY_RATIO * self.bin_memory[b] / msum
            ps += p[b]
        for b in range(nb):
            p[b] /= ps
        return p^

    @staticmethod
    def frame_from(lib: SMMotionLibrary, b: Int, u: Float64) -> Int:
        """A uniform draw `u` in [0, 1) inside bin b, kept ≥ 2 frames before
        the clip's end (and inside the clip)."""
        var f = lib.bin_start[b] + Int(u * Float64(lib.bin_end[b] - lib.bin_start[b]))
        f = min(f, lib.last_frame_of(f) - 2)
        return max(f, lib.clip_start[lib.frame_clip[f]])

    def finalize(mut self, episodes: List[SMEpisodeLog]):
        """Fold a batch of finished episodes (all those reset at one step) into
        the per-bin return estimates."""
        var nb = len(self.bin_return)
        var sums = List[Float64](length=nb, fill=0.0)
        var cnts = List[Float64](length=nb, fill=0.0)
        for e in range(len(episodes)):
            var T = len(episodes[e].rewards)
            if T == 0:
                continue
            var truncated = not episodes[e].terminated and not episodes[e].clip_end
            var rtg = 0.0
            var t = T - 1
            while t >= 0:
                rtg = episodes[e].rewards[t] + SM_SAMPLER_GAMMA * rtg
                if not (truncated and T - t < SM_MIN_REMAINING):
                    var b = episodes[e].bins[t]
                    sums[b] += rtg
                    cnts[b] += 1.0
                t -= 1
        for b in range(nb):
            if cnts[b] > 0:
                self.bin_return[b] = (1.0 - SM_EMA_ALPHA) * self.bin_return[b] + SM_EMA_ALPHA * sums[b] / cnts[b]
