# +--------------------------------------------------------------------------+ #
# | SuperMimic M2: the motion library and the return-based reset sampler
# +--------------------------------------------------------------------------+ #
"""The reproduction's compact `MotionLibrary` (`envs/mdp/motion_library.py`)
and the reset sampler of its `MultiClipMotionCommand` (`envs/mdp/commands.py`,
plan D14), for any `TrackingRobot`.

THE LIBRARY. Clips are concatenated on one global frame axis; a frame holds
the S-wide reference row `[root pos, root quat WXYZ, q, root lin vel (world),
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

from std.builtin.sort import sort
from std.math import ceil, exp, log

from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.deep_agents.supermimic.heads import SMHeadLayout, qapply_inv, yaw_quat
from noeira.deep_agents.supermimic.kinematics import FP, Q4, qapply, sm_fk_lane
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.tables import SMTables
from noeira.nn.constants import DT


comptime SM_GOAL_K: Int = 10
comptime SM_BIN_FRAMES: Int = 50


struct SMMotionLibrary[R: TrackingRobot](Movable):
    var rows: List[Scalar[DT]]         # num_frames × S
    var q0: List[Scalar[DT]]           # n
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
    var rows_dev: Optional[DeviceBuffer[DT]]          # after `to_device`
    var last_dev: Optional[DeviceBuffer[DType.int32]]  # per frame: its clip's last frame

    def __init__(out self, var rows: List[Scalar[DT]], clip_len: List[Int], q0: List[Float64]) raises:
        """From concatenated reference rows and each clip's frame count."""
        var total = 0
        for c in range(len(clip_len)):
            if clip_len[c] < 1:
                raise Error("SMMotionLibrary: clip " + String(c) + " is empty")
            total += clip_len[c]
        if total * SMDims[Self.R].S != len(rows):
            raise Error("SMMotionLibrary: " + String(len(rows)) + " values for " + String(total) + " frames")
        if len(q0) != Self.R.NDOF:
            raise Error("SMMotionLibrary: q0 has " + String(len(q0)) + " entries")
        self.rows = rows^
        self.q0 = List[Scalar[DT]](length=Self.R.NDOF, fill=0)
        for j in range(Self.R.NDOF):
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
        self.rows_dev = None
        self.last_dev = None

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
        self.rows_dev = move.rows_dev^
        self.last_dev = move.last_dev^

    def to_device(mut self, ctx: DeviceContext) raises:
        """The rows and each frame's clip end on the device (once), through
        a fixed 64 MB staging buffer: the full SEED library is ~13 GB of
        rows, and one pinned host copy of it would double the host memory
        and take seconds to pin."""
        comptime CH = 16 * 1024 * 1024        # floats per staging copy
        var nr = len(self.rows)
        var r = ctx.enqueue_create_buffer[DT](nr)
        var l = ctx.enqueue_create_buffer[DType.int32](self.num_frames)
        var hr = ctx.enqueue_create_host_buffer[DT](min(nr, CH))
        var hl = ctx.enqueue_create_host_buffer[DType.int32](min(self.num_frames, CH))
        ctx.synchronize()
        var o = 0
        while o < nr:
            var m = min(CH, nr - o)
            for i in range(m):
                hr[i] = self.rows[o + i]
            ctx.enqueue_copy(r.create_sub_buffer[DT](o, m), hr.create_sub_buffer[DT](0, m))
            ctx.synchronize()                 # the staging buffer is refilled next
            o += m
        o = 0
        while o < self.num_frames:
            var m = min(CH, self.num_frames - o)
            for i in range(m):
                hl[i] = Int32(self.last_frame_of(o + i))
            ctx.enqueue_copy(l.create_sub_buffer[DType.int32](o, m), hl.create_sub_buffer[DType.int32](0, m))
            ctx.synchronize()
            o += m
        self.rows_dev = r^
        self.last_dev = l^

    def rows_ptr(mut self) -> FP[DT]:
        return self.rows.unsafe_ptr().as_unsafe_any_origin()

    def q0_ptr(mut self) -> FP[DT]:
        return self.q0.unsafe_ptr().as_unsafe_any_origin()

    def last_frame_of(self, f: Int) -> Int:
        var c = self.frame_clip[f]
        return self.clip_start[c] + self.clip_len[c] - 1

    def bin_of_frame(self, f: Int) -> Int:
        var c = self.frame_clip[f]
        return self.clip_bin_offset[c] + (f - self.clip_start[c]) // SM_BIN_FRAMES

    @always_inline
    def _row(mut self, f: Int) -> FP[DT]:
        return self.rows.unsafe_ptr().unsafe_offset(f * SMDims[Self.R].S).as_unsafe_any_origin()

    def goal_features(mut self, f: Int, dst: FP[DT]):
        """`goal_features(ref(f), q0)` (2n + 10) into `dst`."""
        sm_goal_features_lane[Self.R](self.rows_ptr(), f, self.q0_ptr(), dst)

    def goal_window(mut self, f: Int, dst: FP[DT]):
        """(10 · (2n + 10)) features of f .. f+9, clamped at the clip's last frame."""
        sm_goal_window_lane[Self.R](self.rows_ptr(), f, self.last_frame_of(f), self.q0_ptr(), dst)

    def ref_qpos_qvel(mut self, f: Int, qpos: FP[DT], qvel: FP[DT]):
        """MuJoCo layout: qvel's root angular velocity in the BODY frame."""
        sm_ref_qpos_qvel_lane[Self.R](self.rows_ptr(), f, qpos, qvel)

    def reference_row(mut self, f: Int, mut tb: SMTables[DT], dst: FP[DT]):
        """The heads' `ref` (13·n_tracked + 7) at frame f: the tracked bodies
        by our FK of the reference state, then the library's root pos and quat."""
        var scratch = List[Scalar[DT]](length=SM_REF_SCRATCH[Self.R], fill=0)
        sm_reference_row_lane[Self.R](
            self.rows_ptr(), f, tb.tree_ptr(), tb.head_ptr(), scratch.unsafe_ptr().as_unsafe_any_origin(), dst
        )


# ── the per-frame functions, one body for the host and the device ──────────
comptime SM_REF_SCRATCH[R: TrackingRobot] = R.NQ + R.NV + SMDims[R].FK_BLOCK
"""`sm_reference_row_lane`'s scratch: qpos, qvel, the FK block."""


@always_inline
def sm_goal_features_lane[R: TrackingRobot](rows: FP[DT], f: Int, q0: FP[DT], dst: FP[DT]):
    """`goal_features(ref(f), q0)`: [q − q0, q̇, R⁻¹ g, R⁻¹ ω, R_yaw⁻¹ v, root z]."""
    comptime n = R.NDOF
    comptime NQ = R.NQ
    var r = rows.unsafe_offset(f * SMDims[R].S)
    var q = Q4[DT](r[unsafe_offset=3], r[unsafe_offset=4], r[unsafe_offset=5], r[unsafe_offset=6])
    var grav = qapply_inv(q, Q4[DT](0, 0, -1, 0))
    var ang_b = qapply_inv(q, Q4[DT](r[unsafe_offset=NQ + 3], r[unsafe_offset=NQ + 4], r[unsafe_offset=NQ + 5], 0))
    var lin_h = qapply_inv(yaw_quat(q), Q4[DT](r[unsafe_offset=NQ], r[unsafe_offset=NQ + 1], r[unsafe_offset=NQ + 2], 0))
    for j in range(n):
        dst[unsafe_offset=j] = r[unsafe_offset=7 + j] - q0[unsafe_offset=j]
        dst[unsafe_offset=n + j] = r[unsafe_offset=NQ + 6 + j]
    comptime for c in range(3):
        dst[unsafe_offset=2 * n + c] = grav[c]
        dst[unsafe_offset=2 * n + 3 + c] = ang_b[c]
        dst[unsafe_offset=2 * n + 6 + c] = lin_h[c]
    dst[unsafe_offset=2 * n + 9] = r[unsafe_offset=2]


@always_inline
def sm_goal_window_lane[R: TrackingRobot](rows: FP[DT], f: Int, last: Int, q0: FP[DT], dst: FP[DT]):
    """The features of f .. f+9, clamped at `last` (the clip's last frame)."""
    for k in range(SM_GOAL_K):
        sm_goal_features_lane[R](rows, min(f + k, last), q0, dst.unsafe_offset(k * SMDims[R].GOAL_FRAME))


@always_inline
def sm_ref_qpos_qvel_lane[R: TrackingRobot](rows: FP[DT], f: Int, qpos: FP[DT], qvel: FP[DT]):
    """The reference state at f in MuJoCo layout (root ω into the body frame)."""
    comptime NQ = R.NQ
    var r = rows.unsafe_offset(f * SMDims[R].S)
    for k in range(NQ):
        qpos[unsafe_offset=k] = r[unsafe_offset=k]
    var q = Q4[DT](r[unsafe_offset=3], r[unsafe_offset=4], r[unsafe_offset=5], r[unsafe_offset=6])
    var ang_b = qapply_inv(q, Q4[DT](r[unsafe_offset=NQ + 3], r[unsafe_offset=NQ + 4], r[unsafe_offset=NQ + 5], 0))
    comptime for k in range(3):
        qvel[unsafe_offset=k] = r[unsafe_offset=NQ + k]
        qvel[unsafe_offset=3 + k] = ang_b[k]
    for j in range(R.NDOF):
        qvel[unsafe_offset=6 + j] = r[unsafe_offset=NQ + 6 + j]


@always_inline
def sm_reference_row_lane[R: TrackingRobot](rows: FP[DT], f: Int, tree: FP[DT], table: FP[DT], scratch: FP[DT], dst: FP[DT]):
    """The heads' `ref` at frame f; `scratch` holds `SM_REF_SCRATCH[R]`."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    var qp = scratch
    var qv = scratch.unsafe_offset(D.NQ)
    var base = scratch.unsafe_offset(D.NQ + D.NV)
    sm_ref_qpos_qvel_lane[R](rows, f, qp, qv)
    var pos = base
    var quat = base.unsafe_offset(D.NB * 3)
    var lin = base.unsafe_offset(D.NB * 7)
    var ang = base.unsafe_offset(D.NB * 10)
    sm_fk_lane[DT, D.NB](tree, qp, qv, pos, quat, lin, ang)
    for k in range(D.N_TRACKED):
        var b = Int(table[unsafe_offset=L.TRACKED + k])
        var o = k * D.REF_BODY
        comptime for c in range(3):
            dst[unsafe_offset=o + c] = pos[unsafe_offset=b * 3 + c]
            dst[unsafe_offset=o + 7 + c] = lin[unsafe_offset=b * 3 + c]
            dst[unsafe_offset=o + 10 + c] = ang[unsafe_offset=b * 3 + c]
        comptime for c in range(4):
            dst[unsafe_offset=o + 3 + c] = quat[unsafe_offset=b * 4 + c]
    var r = rows.unsafe_offset(f * D.S)
    var rr = D.N_TRACKED * D.REF_BODY
    comptime for c in range(7):
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

    def log(mut self, r: Float64, b: Int, cap: Int):
        """One step's reward and bin; past `cap` steps the last slot is
        overwritten (the recorder's bounded per-episode buffer)."""
        if len(self.rewards) < cap:
            self.rewards.append(r)
            self.bins.append(b)
        else:
            self.rewards[cap - 1] = r
            self.bins[cap - 1] = b


@fieldwise_init
struct SMSamplerStats(Copyable, Movable):
    """What the resets of one log window drew, in units of SM_G_MAX (1 = a
    bin whose estimate is the maximum return). They make the training
    curve readable: `avg_reward` averages the episodes that FINISHED, and the
    sampler moves resets toward the bins it estimates hard, so a falling
    `avg_reward` with a falling `reset_bin_return` is the curriculum, not
    the policy."""
    var resets: Int
    var reset_bin_return: Float64     # mean G_b of the drawn bins
    var bin_return_mean: Float64      # mean G_b over all bins (the estimate's level)
    var reset_hard_share: Float64     # share of draws below the last window's bottom-decile G_b


struct SMResetSampler(Movable):
    var bin_return: List[Float64]
    var bin_memory: List[Float64]
    var n_draw: Int                   # draws since the last `take_stats`
    var draw_g: Float64
    var draw_hard: Int
    var hard_g: Float64               # the bottom-decile G_b at the last `take_stats`

    def __init__(out self, num_bins: Int):
        self.bin_return = List[Float64](length=num_bins, fill=SM_G_MAX)
        self.bin_memory = List[Float64](length=num_bins, fill=1.0 / Float64(num_bins))
        self.n_draw = 0
        self.draw_g = 0.0
        self.draw_hard = 0
        self.hard_g = -1e300

    def __init__(out self, *, deinit move: Self):
        self.bin_return = move.bin_return^
        self.bin_memory = move.bin_memory^
        self.n_draw = move.n_draw
        self.draw_g = move.draw_g
        self.draw_hard = move.draw_hard
        self.hard_g = move.hard_g

    def note_draw(mut self, b: Int):
        """Count a reset drawn from bin b (the envs call it after each draw)."""
        var g = self.bin_return[b]
        self.n_draw += 1
        self.draw_g += g
        if g < self.hard_g:
            self.draw_hard += 1

    def take_stats(mut self) -> SMSamplerStats:
        """The window's draws (then cleared); refreshes the bottom-decile
        threshold the next window's hard share is counted against. O(bins
        log bins), once per log line."""
        var nb = len(self.bin_return)
        var srt = self.bin_return.copy()
        sort(srt)
        var mean = 0.0
        for b in range(nb):
            mean += srt[b]
        mean /= Float64(max(nb, 1))
        var n = self.n_draw
        var out = SMSamplerStats(
            n,
            (self.draw_g / Float64(n) if n > 0 else mean) / SM_G_MAX,
            mean / SM_G_MAX,
            Float64(self.draw_hard) / Float64(n) if n > 0 else 0.0,
        )
        self.hard_g = srt[nb // 10] if nb > 0 else -1e300
        self.n_draw = 0
        self.draw_g = 0.0
        self.draw_hard = 0
        return out^

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
    def bin_from_cdf(cdf: List[Float64], x: Float64) -> Int:
        """The first bin whose cumulative probability reaches x (the last bin
        if none does): what the linear scan `while b < len − 1 and cdf[b] <
        x: b += 1` returns, in O(log bins) — the full SEED has ~10⁶ bins and
        every resetting lane draws one."""
        var lo = 0
        var hi = len(cdf) - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cdf[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @staticmethod
    def frame_from[R: TrackingRobot](lib: SMMotionLibrary[R], b: Int, u: Float64) -> Int:
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
