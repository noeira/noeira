"""Reward-bearing Pong pixel-obs offline buffer (Dreamer 4 end-to-end run).

The LeWM `PongOfflineBuffer` (`noeira/envs/arcade_games/pong/offline_buffer.mojo`,
format LWMP) stores frames + actions + dones but NO rewards — fine for the JEPA
world model, but the Dreamer 4 agent's reward head (eq. 9) and imagination RL
(eq. 10/11) need the per-step reward signal. This is a parallel, self-contained
buffer that ALSO records the reward, so the whole pipeline (tokenizer → world
model → reward/continue heads → imagination RL) can train on real Pong data.

File format LWMR v1 (little-endian, native fp32 byte order):
    Bytes [0..3]:    "LWMR" magic
    Bytes [4..7]:    uint32 version (1)
    Bytes [8..15]:   uint64 n_frames
    Bytes [16..19]:  uint32 obs_c  (=4)
    Bytes [20..23]:  uint32 obs_h  (=84)
    Bytes [24..27]:  uint32 obs_w  (=84)
    Bytes [28..31]:  uint32 num_actions  (=3)
    Bytes [32..63]:  reserved (zero)

    Then in order:
      n_frames * PONG_FRAME_BYTES bytes  — pixel data, layout (N, C, H, W) uint8.
      n_frames bytes                     — action index in {0, 1, 2}.
      n_frames bytes                     — done flag in {0, 1}.
      n_frames * 4 bytes                 — reward, fp32.

Unlike `WindowSource` (which surfaces only pixels + actions and picks windows
internally), this buffer's own sampler emits everything for the SAME window —
pixels, one-hot actions, rewards AND dones — so reward/continue labels stay
aligned with the frames the agent sees. Pure CPU; uses an internal xorshift RNG
for window selection so it doesn't depend on global random state ordering.

Sampling protocol mirrors LWMP: pick start ∈ [0, n_frames - T]; reject windows
where any of dones[start : start + T - 1] is set (episode boundary mid-window).
The final frame may be done (a valid prediction / termination target).
"""

from noeira.io.fileio import write_file_atomic
from noeira.envs.arcade_games.pong.offline_buffer import (
    PONG_OBS_C,
    PONG_OBS_H,
    PONG_OBS_W,
    PONG_NUM_ACTIONS,
    PONG_FRAME_BYTES,
)

comptime PONG_R_MAGIC: String = "LWMR"
comptime PONG_R_VERSION: UInt32 = 1
comptime PONG_R_HEADER_BYTES: Int = 64


# ============================================================================
# Endianness helpers (little-endian, matching offline_buffer.mojo)
# ============================================================================


@always_inline
def _write_uint32_le(mut buf: List[UInt8], value: UInt32):
    buf.append(UInt8(value & 0xFF))
    buf.append(UInt8((value >> 8) & 0xFF))
    buf.append(UInt8((value >> 16) & 0xFF))
    buf.append(UInt8((value >> 24) & 0xFF))


@always_inline
def _write_uint64_le(mut buf: List[UInt8], value: UInt64):
    for i in range(8):
        buf.append(UInt8((value >> UInt64(8 * i)) & 0xFF))


@always_inline
def _read_uint32_le(data: List[UInt8], pos: Int) -> UInt32:
    return (
        UInt32(data[pos])
        | (UInt32(data[pos + 1]) << 8)
        | (UInt32(data[pos + 2]) << 16)
        | (UInt32(data[pos + 3]) << 24)
    )


@always_inline
def _read_uint64_le(data: List[UInt8], pos: Int) -> UInt64:
    var v: UInt64 = 0
    for i in range(8):
        v |= UInt64(data[pos + i]) << UInt64(8 * i)
    return v


# ============================================================================
# Dreamer4PongRewardBuffer
# ============================================================================


struct Dreamer4PongRewardBuffer(Movable):
    """Flat offline buffer for Pong pixel trajectories WITH per-step rewards.

    Owns four parallel arrays:
      - frames:  capacity * PONG_FRAME_BYTES uint8  (pixels in [0, 255])
      - actions: capacity uint8
      - dones:   capacity uint8
      - rewards: capacity fp32

    Single-producer / single-consumer (collect then train). Not thread-safe.
    """

    var capacity: Int
    var n_frames: Int
    # Owned host buffers (RAII Lists — no manual alloc/free, no MutAnyOrigin).
    var frames: List[UInt8]
    var actions: List[UInt8]
    var dones: List[UInt8]
    var rewards: List[Scalar[DType.float32]]
    var rng: UInt64

    def __init__(out self, capacity: Int, seed: UInt64 = 0x9E3779B97F4A7C15):
        self.capacity = capacity
        self.n_frames = 0
        self.frames = List[UInt8](length=capacity * PONG_FRAME_BYTES, fill=0)
        self.actions = List[UInt8](length=capacity, fill=0)
        self.dones = List[UInt8](length=capacity, fill=0)
        self.rewards = List[Scalar[DType.float32]](
            length=capacity, fill=Scalar[DType.float32](0.0)
        )
        self.rng = seed | 1

    def __init__(out self, *, deinit move: Self):
        self.capacity = move.capacity
        self.n_frames = move.n_frames
        self.frames = move.frames^
        self.actions = move.actions^
        self.dones = move.dones^
        self.rewards = move.rewards^
        self.rng = move.rng

    @always_inline
    def _u64(mut self) -> UInt64:
        var x = self.rng
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.rng = x
        return x * 0x2545F4914F6CDD1D

    # ------------------------------------------------------------------
    # Append a step
    # ------------------------------------------------------------------

    def add_step_fp32_list(
        mut self,
        obs: List[Scalar[DType.float32]],
        action: Int,
        done: Bool,
        reward: Scalar[DType.float32],
    ):
        """Append one step. obs is the 4×84×84 frame stack in [0, 1] fp32;
        quantized to uint8 via clamp(round(x·255), 0, 255). Stores the raw
        scalar reward."""
        if self.n_frames >= self.capacity:
            return  # silently drop — caller should check capacity
        var base = self.n_frames * PONG_FRAME_BYTES
        for i in range(PONG_FRAME_BYTES):
            var v = obs[i] * 255.0 + 0.5
            if v < 0.0:
                v = 0.0
            elif v > 255.0:
                v = 255.0
            self.frames[base + i] = UInt8(Int(v))
        self.actions[self.n_frames] = UInt8(action)
        self.dones[self.n_frames] = UInt8(1) if done else UInt8(0)
        self.rewards[self.n_frames] = reward
        self.n_frames += 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _window_is_valid(self, start: Int, T: Int) -> Bool:
        # The window covers frames [start, start + T - 1]. Reject if any of
        # dones[start .. start + T - 2] is set (episode boundary mid-window).
        # The last frame may be done (valid prediction / termination target).
        for i in range(start, start + T - 1):
            if self.dones[i] != 0:
                return False
        return True

    def sample_reward_window_batch[
        B: Int, T: Int, ACT: Int,
        pix_fp32_o: Origin[mut=True],
        act_onehot_o: Origin[mut=True],
        rew_o: Origin[mut=True],
        done_o: Origin[mut=True],
    ](
        mut self,
        pix_fp32: Pointer[Scalar[DType.float32], pix_fp32_o],  # [B*T*FRAME]
        act_onehot: Pointer[Scalar[DType.float32], act_onehot_o],  # [B*T*ACT]
        rew: Pointer[Scalar[DType.float32], rew_o],         # [B*T]
        done: Pointer[Scalar[DType.float32], done_o],        # [B*T]
    ) raises:
        """Sample B contiguous-T windows. Fills, for each (b, t):
          pix_fp32:    fp32 pixels in [0, 1], CHW per frame.
          act_onehot:  fp32 one-hot of the dataset action.
          rew:         fp32 reward recorded at that step.
          done:        fp32 done flag {0, 1}.
        Internal RNG picks the window starts (no global random state)."""
        if self.n_frames < T:
            raise Error(
                "Dreamer4PongRewardBuffer.sample_reward_window_batch: too small"
            )
        var max_start = self.n_frames - T
        for b in range(B):
            var start: Int = -1
            for _ in range(64):
                var r = Float64(self._u64() >> 11) * (1.0 / 9007199254740992.0)
                var cand = Int(r * Float64(max_start + 1))
                if cand > max_start:
                    cand = max_start
                if self._window_is_valid(cand, T):
                    start = cand
                    break
            if start < 0:
                for s in range(max_start + 1):
                    if self._window_is_valid(s, T):
                        start = s
                        break
            if start < 0:
                raise Error(
                    "Dreamer4PongRewardBuffer: no valid windows"
                )
            for t in range(T):
                var bt = b * T + t
                var src = (start + t) * PONG_FRAME_BYTES
                var dst = bt * PONG_FRAME_BYTES
                for i in range(PONG_FRAME_BYTES):
                    pix_fp32[unsafe_offset=dst + i] = (
                        Scalar[DType.float32](Float64(self.frames[src + i]))
                        * Scalar[DType.float32](1.0 / 255.0)
                    )
                var a = Int(self.actions[start + t])
                for k in range(ACT):
                    act_onehot[unsafe_offset=bt * ACT + k] = Scalar[DType.float32](0.0)
                if a >= 0 and a < ACT:
                    act_onehot[unsafe_offset=bt * ACT + a] = Scalar[DType.float32](1.0)
                rew[unsafe_offset=bt] = self.rewards[start + t]
                done[unsafe_offset=bt] = Scalar[DType.float32](
                    1.0 if self.dones[start + t] != 0 else 0.0
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: String) raises:
        """Write buffer to disk in LWMR v1 format."""
        var total = (
            PONG_R_HEADER_BYTES
            + self.n_frames * PONG_FRAME_BYTES
            + 2 * self.n_frames
            + 4 * self.n_frames
        )
        var data = List[UInt8](capacity=total)

        for i in range(4):
            data.append(UInt8(ord(PONG_R_MAGIC[byte=i])))
        _write_uint32_le(data, PONG_R_VERSION)
        _write_uint64_le(data, UInt64(self.n_frames))
        _write_uint32_le(data, UInt32(PONG_OBS_C))
        _write_uint32_le(data, UInt32(PONG_OBS_H))
        _write_uint32_le(data, UInt32(PONG_OBS_W))
        _write_uint32_le(data, UInt32(PONG_NUM_ACTIONS))
        while len(data) < PONG_R_HEADER_BYTES:
            data.append(UInt8(0))

        var frame_bytes = self.n_frames * PONG_FRAME_BYTES
        for i in range(frame_bytes):
            data.append(self.frames[i])
        for i in range(self.n_frames):
            data.append(self.actions[i])
        for i in range(self.n_frames):
            data.append(self.dones[i])
        # rewards: raw native fp32 bytes
        var rb = self.rewards.unsafe_ptr().unsafe_bitcast[UInt8]()
        for i in range(4 * self.n_frames):
            data.append(rb[unsafe_offset=i])

        write_file_atomic(path, data)

    @staticmethod
    def load(path: String) raises -> Dreamer4PongRewardBuffer:
        """Load buffer from disk. Validates magic + dimensions."""
        var data: List[UInt8]
        with open(path, "r") as f:
            data = f.read_bytes()

        if len(data) < PONG_R_HEADER_BYTES:
            raise Error("Dreamer4PongRewardBuffer.load: file too small")
        for i in range(4):
            if data[i] != UInt8(ord(PONG_R_MAGIC[byte=i])):
                raise Error("Dreamer4PongRewardBuffer.load: bad magic")
        var version = _read_uint32_le(data, 4)
        if version != PONG_R_VERSION:
            raise Error("Dreamer4PongRewardBuffer.load: unsupported version")

        var n_frames = Int(_read_uint64_le(data, 8))
        var obs_c = Int(_read_uint32_le(data, 16))
        var obs_h = Int(_read_uint32_le(data, 20))
        var obs_w = Int(_read_uint32_le(data, 24))
        var num_actions = Int(_read_uint32_le(data, 28))
        if (
            obs_c != PONG_OBS_C
            or obs_h != PONG_OBS_H
            or obs_w != PONG_OBS_W
            or num_actions != PONG_NUM_ACTIONS
        ):
            raise Error(
                "Dreamer4PongRewardBuffer.load: shape mismatch — expected"
                " 4×84×84 with 3 actions"
            )

        var expected = (
            PONG_R_HEADER_BYTES
            + n_frames * PONG_FRAME_BYTES
            + 2 * n_frames
            + 4 * n_frames
        )
        if len(data) < expected:
            raise Error("Dreamer4PongRewardBuffer.load: file truncated")

        var buf = Dreamer4PongRewardBuffer(capacity=n_frames)
        var off = PONG_R_HEADER_BYTES
        for i in range(n_frames * PONG_FRAME_BYTES):
            buf.frames[i] = data[off + i]
        off += n_frames * PONG_FRAME_BYTES
        for i in range(n_frames):
            buf.actions[i] = data[off + i]
        off += n_frames
        for i in range(n_frames):
            buf.dones[i] = data[off + i]
        off += n_frames
        var rb = buf.rewards.unsafe_ptr().unsafe_bitcast[UInt8]()
        for i in range(4 * n_frames):
            rb[unsafe_offset=i] = data[off + i]
        buf.n_frames = n_frames
        return buf^
