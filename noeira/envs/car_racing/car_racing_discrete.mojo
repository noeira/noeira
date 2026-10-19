"""CarRacingDiscrete — GPU-batched, discrete-action CarRacing on multi-body physics.

This is the training-facing env for value-based agents (DQN / Rainbow): it
conforms to `GPUDiscreteEnv`, embeds a per-env procedural track in the state
buffer (so the static reset/step kernels are self-contained), and drives the car
with the Box2D-faithful `CarDynamicsMB` (hull + 4 wheels + 4 revolute joints).

Action space (5 discrete, Gymnasium CarRacing-v3 discrete):
  0 = do nothing, 1 = steer left, 2 = steer right, 3 = gas, 4 = brake.

Reward (Gymnasium): -0.1 per frame + 1000/N per newly-visited tile; -100 and
terminate when the car leaves the playfield; terminate on lap completion.

Observation: 13-D normalized (state prefix, so the default extract_obs works).

State row layout:
  [OBS(13) | BODIES(5*13) | FORCES(5*3) | JOINTS(4*17) | ROLLING(4) |
   CONTROLS(3) | METADATA(6) | TRACK(MAX_TILES*9) | VISITED(MAX_TILES)]

The embedded track generator mirrors the legacy GPU path's simplified
checkpoint-trace (GPU-friendly); physics fidelity (the part that mattered) comes
from CarDynamicsMB. See project_car_racing_audit memory.
"""

from std.math import sqrt, sin, cos, pi
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom

from noeira.core import GPUDiscreteEnv
from noeira.physics2d import dtype, TPB
from noeira.physics2d.constants import (
    IDX_X, IDX_Y, IDX_ANGLE, IDX_VX, IDX_VY, IDX_OMEGA, BODY_STATE_SIZE,
)
from noeira.physics2d.car import CarDynamicsMB, TileCollision
from noeira.physics2d.car.constants import (
    FRICTION_LIMIT, CAR_DT, CTRL_STEERING, CTRL_GAS, CTRL_BRAKE,
    STEERING_LIMIT, TILE_DATA_SIZE,
)
from .constants import CRConstants
from noeira.core.tiles import Tile, ImmTile, tile


struct CarRacingDiscrete[DTYPE: DType](GPUDiscreteEnv, Copyable, Movable):
    """GPU-batched discrete CarRacing on multi-body physics."""

    comptime dtype = Self.DTYPE
    comptime NAME: String = "CarRacingDiscrete"

    # Topology / sizes
    comptime NB: Int = CarDynamicsMB.NUM_BODIES
    comptime NJ: Int = CarDynamicsMB.NUM_JOINTS
    comptime NW: Int = CarDynamicsMB.NUM_WHEELS
    comptime MAX_TILES: Int = 300

    comptime OBS_DIM: Int = 13
    comptime NUM_ACTIONS: Int = 5
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    # Layout offsets
    comptime OBS_OFFSET: Int = 0
    comptime BODIES_OFFSET: Int = Self.OBS_OFFSET + Self.OBS_DIM
    comptime FORCES_OFFSET: Int = Self.BODIES_OFFSET + Self.NB * BODY_STATE_SIZE
    comptime JOINTS_OFFSET: Int = Self.FORCES_OFFSET + Self.NB * 3
    comptime ROLLING_OFFSET: Int = Self.JOINTS_OFFSET + Self.NJ * 17
    comptime CONTROLS_OFFSET: Int = Self.ROLLING_OFFSET + Self.NW
    comptime METADATA_OFFSET: Int = Self.CONTROLS_OFFSET + 3
    comptime TRACK_OFFSET: Int = Self.METADATA_OFFSET + 6
    comptime VISITED_OFFSET: Int = Self.TRACK_OFFSET + Self.MAX_TILES * TILE_DATA_SIZE
    comptime STATE_SIZE: Int = Self.VISITED_OFFSET + Self.MAX_TILES

    # Metadata fields
    comptime META_STEP: Int = 0
    comptime META_TOTAL_REWARD: Int = 1
    comptime META_DONE: Int = 2
    comptime META_TRUNCATED: Int = 3
    comptime META_TILES_VISITED: Int = 4
    comptime META_NUM_TILES: Int = 5

    # Observation normalization
    comptime POS_SCALE: Float64 = CRConstants.PLAYFIELD
    comptime VEL_SCALE: Float64 = 100.0
    comptime OMEGA_SCALE: Float64 = 5.0
    comptime ROLL_SCALE: Float64 = 200.0

    comptime MAX_STEPS: Int = CRConstants.MAX_STEPS
    comptime LAP_PERCENT: Float64 = CRConstants.LAP_COMPLETE_PERCENT

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    # =========================================================================
    # Discrete action decode -> controls (Gymnasium CarRacing-v3 discrete)
    # =========================================================================
    @always_inline
    @staticmethod
    def _decode_action(
        a: Int,
    ) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
        """(steering in [-1,1], gas in [0,1], brake in [0,1]) for action a.
        Turn actions use full lock (steering ±1 -> ±STEERING_LIMIT), matching
        Gymnasium's discrete steer of ±0.6 (which saturates the ±0.4 joint)."""
        var steer = Scalar[dtype](0.0)
        var gas = Scalar[dtype](0.0)
        var brake = Scalar[dtype](0.0)
        if a == 1:
            steer = Scalar[dtype](-1.0)
        elif a == 2:
            steer = Scalar[dtype](1.0)
        elif a == 3:
            gas = Scalar[dtype](0.2)
        elif a == 4:
            brake = Scalar[dtype](0.8)
        return (steer, gas, brake)

    # =========================================================================
    # Observation write (normalized) for one env
    # =========================================================================
    @always_inline
    @staticmethod
    def _write_obs[
        BATCH: Int, STATE_SIZE: Int
    ](
        env: Int,
        states: Tile[dtype, BATCH, STATE_SIZE],
    ):
        var ho = Self.BODIES_OFFSET  # hull = body 0
        var a = rebind[Scalar[dtype]](states[env, ho + IDX_ANGLE])
        var vx = rebind[Scalar[dtype]](states[env, ho + IDX_VX])
        var vy = rebind[Scalar[dtype]](states[env, ho + IDX_VY])
        var om = rebind[Scalar[dtype]](states[env, ho + IDX_OMEGA])
        var x = rebind[Scalar[dtype]](states[env, ho + IDX_X])
        var y = rebind[Scalar[dtype]](states[env, ho + IDX_Y])
        var fl_a = rebind[Scalar[dtype]](
            states[env, Self.BODIES_OFFSET + 1 * BODY_STATE_SIZE + IDX_ANGLE]
        )
        var ps = Scalar[dtype](Self.POS_SCALE)
        var vs = Scalar[dtype](Self.VEL_SCALE)
        var o = Self.OBS_OFFSET
        states[env, o + 0] = x / ps
        states[env, o + 1] = y / ps
        states[env, o + 2] = sin(a)
        states[env, o + 3] = cos(a)
        states[env, o + 4] = vx / vs
        states[env, o + 5] = vy / vs
        states[env, o + 6] = om / Scalar[dtype](Self.OMEGA_SCALE)
        states[env, o + 7] = (fl_a - a) / Scalar[dtype](STEERING_LIMIT)
        for w in range(Self.NW):
            states[env, o + 8 + w] = rebind[Scalar[dtype]](
                states[env, Self.ROLLING_OFFSET + w]
            ) / Scalar[dtype](Self.ROLL_SCALE)
        states[env, o + 12] = sqrt(vx * vx + vy * vy) / vs

    # =========================================================================
    # Embedded procedural track generator (one env). Returns tile count.
    # =========================================================================
    @always_inline
    @staticmethod
    def _gen_track[
        BATCH: Int, STATE_SIZE: Int
    ](
        states: Tile[dtype, BATCH, STATE_SIZE],
        env: Int,
        mut rng: PhiloxRandom,
    ) -> Int:
        """GPU-friendly checkpoint-trace track embedded at TRACK_OFFSET."""
        var track_rad = Scalar[dtype](CRConstants.TRACK_RAD)
        var track_width = Scalar[dtype](CRConstants.TRACK_WIDTH)
        var detail = Scalar[dtype](CRConstants.TRACK_DETAIL_STEP)
        var turn_rate = Scalar[dtype](CRConstants.TRACK_TURN_RATE)
        var two_pi = Scalar[dtype](2.0 * pi)
        comptime NC = 12

        var cx = Array[Scalar[dtype], NC](fill=Scalar[dtype](0))
        var cy = Array[Scalar[dtype], NC](fill=Scalar[dtype](0))
        for c in range(NC):
            var rv = rng.step_uniform()
            var noise = (rv[0] - Scalar[dtype](0.5)) * two_pi / Scalar[dtype](
                NC
            ) * Scalar[dtype](0.5)
            var alpha = two_pi * Scalar[dtype](c) / Scalar[dtype](NC) + noise
            var rad = track_rad * (Scalar[dtype](0.5) + rv[1] * Scalar[dtype](0.5))
            if c == 0:
                alpha = Scalar[dtype](0.0)
                rad = track_rad
            cx[c] = rad * cos(alpha)
            cy[c] = rad * sin(alpha)

        var x = cx[0]
        var y = cy[0]
        var beta = Scalar[dtype](pi / 2.0)
        var num_tiles = 0
        var dest_i = 1
        var prev_x = x
        var prev_y = y

        for _ in range(500):
            if num_tiles >= Self.MAX_TILES - 1:
                break
            var dx = cx[dest_i % NC] - x
            var dy = cy[dest_i % NC] - y
            var dist = sqrt(dx * dx + dy * dy)
            var ang = Self._atan2(dy, dx)
            var adiff = ang - beta
            while adiff > Scalar[dtype](pi):
                adiff = adiff - two_pi
            while adiff < Scalar[dtype](-pi):
                adiff = adiff + two_pi
            var steer = turn_rate
            if adiff < Scalar[dtype](0.0):
                steer = -turn_rate
            if adiff > Scalar[dtype](-0.1) and adiff < Scalar[dtype](0.1):
                steer = adiff
            beta = beta + steer
            x = x + detail * cos(beta)
            y = y + detail * sin(beta)
            if dist < track_rad * Scalar[dtype](0.3):
                dest_i += 1
            var px = -sin(beta) * track_width / Scalar[dtype](2.0)
            var py = cos(beta) * track_width / Scalar[dtype](2.0)
            var to = Self.TRACK_OFFSET + num_tiles * TILE_DATA_SIZE
            states[env, to + 0] = prev_x - px
            states[env, to + 1] = prev_y - py
            states[env, to + 2] = x - px
            states[env, to + 3] = y - py
            states[env, to + 4] = x + px
            states[env, to + 5] = y + py
            states[env, to + 6] = prev_x + px
            states[env, to + 7] = prev_y + py
            states[env, to + 8] = Scalar[dtype](1.0)  # road friction
            num_tiles += 1
            prev_x = x
            prev_y = y
            if dest_i >= NC + 1:
                break
        return num_tiles

    @always_inline
    @staticmethod
    def _atan2(y: Scalar[dtype], x: Scalar[dtype]) -> Scalar[dtype]:
        """Polynomial atan2 (GPU-friendly; matches legacy track-gen)."""
        var ax = x if x >= Scalar[dtype](0.0) else -x
        var ay = y if y >= Scalar[dtype](0.0) else -y
        var r: Scalar[dtype]
        if ax > ay:
            var t = y / x
            var t2 = t * t
            r = t - t * t2 / Scalar[dtype](3.0) + t * t2 * t2 / Scalar[dtype](5.0)
            if r > Scalar[dtype](0.8):
                r = Scalar[dtype](0.8)
            elif r < Scalar[dtype](-0.8):
                r = Scalar[dtype](-0.8)
            if x < Scalar[dtype](0.0):
                if y >= Scalar[dtype](0.0):
                    r = r + Scalar[dtype](pi)
                else:
                    r = r - Scalar[dtype](pi)
        else:
            var t = x / y
            var t2 = t * t
            var at = t - t * t2 / Scalar[dtype](3.0) + t * t2 * t2 / Scalar[dtype](5.0)
            if at > Scalar[dtype](0.8):
                at = Scalar[dtype](0.8)
            elif at < Scalar[dtype](-0.8):
                at = Scalar[dtype](-0.8)
            r = Scalar[dtype](pi / 2.0) - at
            if y < Scalar[dtype](0.0):
                r = -r
        return r

    # =========================================================================
    # Reset one env: clear, gen track, place car at start
    # =========================================================================
    @always_inline
    @staticmethod
    def _reset_env[
        BATCH: Int, STATE_SIZE: Int
    ](
        states: Tile[dtype, BATCH, STATE_SIZE],
        env: Int,
        seed: Int,
    ):
        for i in range(STATE_SIZE):
            states[env, i] = Scalar[dtype](0.0)

        var rng = PhiloxRandom(seed=UInt64(seed), offset=0)
        var num_tiles = Self._gen_track[BATCH, STATE_SIZE](states, env, rng)
        states[env, Self.METADATA_OFFSET + Self.META_NUM_TILES] = Scalar[dtype](
            num_tiles
        )

        # Start = center of tile 0, heading along the tile.
        var to = Self.TRACK_OFFSET
        var v0x = rebind[Scalar[dtype]](states[env, to + 0])
        var v0y = rebind[Scalar[dtype]](states[env, to + 1])
        var v3x = rebind[Scalar[dtype]](states[env, to + 6])
        var v3y = rebind[Scalar[dtype]](states[env, to + 7])
        var v1x = rebind[Scalar[dtype]](states[env, to + 2])
        var v1y = rebind[Scalar[dtype]](states[env, to + 3])
        var sx = (v0x + v3x) / Scalar[dtype](2.0)
        var sy = (v0y + v3y) / Scalar[dtype](2.0)
        var sa = Self._atan2(v1y - v0y, v1x - v0x)

        CarDynamicsMB.init_env[
            BATCH, STATE_SIZE, Self.BODIES_OFFSET, Self.FORCES_OFFSET,
            Self.JOINTS_OFFSET, Self.ROLLING_OFFSET,
        ](env, states, sx, sy, sa)

        Self._write_obs[BATCH, STATE_SIZE](env, states)

    # =========================================================================
    # Step one env (decode action, physics, reward, termination, obs)
    # =========================================================================
    @always_inline
    @staticmethod
    def _step_env[
        BATCH: Int, STATE_SIZE: Int, ACTION_DIM: Int
    ](
        env: Int,
        states: Tile[dtype, BATCH, STATE_SIZE],
        actions: ImmTile[dtype, BATCH, ACTION_DIM],
        rewards: Tile[dtype, BATCH],
        dones: Tile[dtype, BATCH],
    ):
        var mo = Self.METADATA_OFFSET
        var was_done = rebind[Scalar[dtype]](states[env, mo + Self.META_DONE]) > Scalar[dtype](0.5)
        if was_done:
            rewards[env] = Scalar[dtype](0.0)
            dones[env] = Scalar[dtype](1.0)
            return

        var n = Int(rebind[Scalar[dtype]](states[env, mo + Self.META_NUM_TILES]))
        if n <= 0:
            n = 1

        # Decode discrete action -> controls
        var a = Int(rebind[Scalar[dtype]](actions[env, 0]))
        var dec = Self._decode_action(a)
        states[env, Self.CONTROLS_OFFSET + CTRL_STEERING] = dec[0]
        states[env, Self.CONTROLS_OFFSET + CTRL_GAS] = dec[1]
        states[env, Self.CONTROLS_OFFSET + CTRL_BRAKE] = dec[2]

        # Per-wheel friction from each wheel body position (embedded track)
        var fr = Array[Scalar[dtype], 4](fill=Scalar[dtype](0))
        for w in range(Self.NW):
            var wp = CarDynamicsMB.wheel_world_pos[
                BATCH, STATE_SIZE, Self.BODIES_OFFSET
            ](env, states, w)
            fr[w] = TileCollision.get_friction_limit_at_embedded[
                BATCH, STATE_SIZE, Self.TRACK_OFFSET, Self.MAX_TILES
            ](env, wp[0], wp[1], states, n)

        CarDynamicsMB.step_single_env_pw[
            BATCH, STATE_SIZE, Self.BODIES_OFFSET, Self.FORCES_OFFSET,
            Self.JOINTS_OFFSET, Self.ROLLING_OFFSET, Self.CONTROLS_OFFSET,
        ](env, states, fr[0], fr[1], fr[2], fr[3], Scalar[dtype](CAR_DT))

        # Reward: -0.1/frame + 1000/N per new tile
        var hx = rebind[Scalar[dtype]](states[env, Self.BODIES_OFFSET + IDX_X])
        var hy = rebind[Scalar[dtype]](states[env, Self.BODIES_OFFSET + IDX_Y])
        var reward = Scalar[dtype](-0.1)
        var vr = TileCollision.check_and_mark_visited_embedded[
            BATCH, STATE_SIZE, Self.TRACK_OFFSET, Self.VISITED_OFFSET, Self.MAX_TILES
        ](env, hx, hy, states, n)
        var tiles_visited = rebind[Scalar[dtype]](
            states[env, mo + Self.META_TILES_VISITED]
        )
        if vr[1]:
            reward = reward + Scalar[dtype](1000.0) / Scalar[dtype](n)
            tiles_visited = tiles_visited + Scalar[dtype](1.0)
            states[env, mo + Self.META_TILES_VISITED] = tiles_visited

        # Termination
        var done = Scalar[dtype](0.0)
        var pf = Scalar[dtype](CRConstants.PLAYFIELD)
        var ax = hx if hx >= Scalar[dtype](0.0) else -hx
        var ay = hy if hy >= Scalar[dtype](0.0) else -hy
        if ax > pf or ay > pf:
            done = Scalar[dtype](1.0)
            reward = Scalar[dtype](-100.0)
        if tiles_visited / Scalar[dtype](n) >= Scalar[dtype](Self.LAP_PERCENT):
            done = Scalar[dtype](1.0)
        var step = rebind[Scalar[dtype]](states[env, mo + Self.META_STEP]) + Scalar[dtype](1.0)
        states[env, mo + Self.META_STEP] = step
        if step >= Scalar[dtype](Self.MAX_STEPS):
            done = Scalar[dtype](1.0)
            states[env, mo + Self.META_TRUNCATED] = Scalar[dtype](1.0)

        states[env, mo + Self.META_DONE] = done
        Self._write_obs[BATCH, STATE_SIZE](env, states)
        rewards[env] = reward
        dones[env] = done

    # =========================================================================
    # GPUDiscreteEnv: host launchers
    # =========================================================================
    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int
    ](ctx: DeviceContext, mut states: DeviceBuffer[dtype], rng_seed: UInt64 = 0) raises:
        var st = tile[BATCH_SIZE, STATE_SIZE](states)
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def reset_wrap(
            st: Tile[dtype, BATCH_SIZE, STATE_SIZE],
            seed: Scalar[DType.uint64],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            var s = Int(seed) * 2654435761 + env * 40503
            CarRacingDiscrete[Self.dtype]._reset_env[BATCH_SIZE, STATE_SIZE](st, env, s)

        ctx.enqueue_function[reset_wrap](
            st, Scalar[DType.uint64](rng_seed), grid_dim=(BLOCKS,), block_dim=(TPB,)
        )

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        rng_seed: UInt64,
        workspace_ptr: Optional[Pointer[Scalar[dtype], MutAnyOrigin]] = None,
        rng_counter_ptr: Optional[Pointer[Scalar[DType.uint64], MutAnyOrigin]] = None,
    ) raises:
        var st = tile[BATCH_SIZE, STATE_SIZE](states)
        var dn = tile[BATCH_SIZE](dones)
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def sel_wrap(
            st: Tile[dtype, BATCH_SIZE, STATE_SIZE],
            dn: Tile[dtype, BATCH_SIZE],
            seed: Scalar[DType.uint64],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            if rebind[Scalar[dtype]](dn[env]) > Scalar[dtype](0.5):
                var s = Int(seed) * 2654435761 + env * 40503
                CarRacingDiscrete[Self.dtype]._reset_env[BATCH_SIZE, STATE_SIZE](st, env, s)
                dn[env] = Scalar[dtype](0.0)

        ctx.enqueue_function[sel_wrap](
            st, dn, Scalar[DType.uint64](rng_seed), grid_dim=(BLOCKS,), block_dim=(TPB,)
        )

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int, OBS_DIM: Int
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        mut terminated: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
        workspace_ptr: Optional[Pointer[Scalar[dtype], MutAnyOrigin]] = None,
        rng_counter_ptr: Optional[Pointer[Scalar[DType.uint64], MutAnyOrigin]] = None,
    ) raises:
        var st = tile[BATCH_SIZE, STATE_SIZE](states)
        var ac = tile[BATCH_SIZE, 1](actions)
        var rw = tile[BATCH_SIZE](rewards)
        var dn = tile[BATCH_SIZE](dones)
        var tm = tile[BATCH_SIZE](terminated)
        var ob = tile[BATCH_SIZE, OBS_DIM](obs)
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def step_wrap(
            st: Tile[dtype, BATCH_SIZE, STATE_SIZE],
            ac: ImmTile[dtype, BATCH_SIZE, 1],
            rw: Tile[dtype, BATCH_SIZE],
            dn: Tile[dtype, BATCH_SIZE],
            tm: Tile[dtype, BATCH_SIZE],
            ob: Tile[dtype, BATCH_SIZE, OBS_DIM],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            CarRacingDiscrete[Self.dtype]._step_env[BATCH_SIZE, STATE_SIZE, 1](
                env, st, ac, rw, dn
            )
            var is_trunc = rebind[Scalar[dtype]](
                st[env, Self.METADATA_OFFSET + Self.META_TRUNCATED]
            )
            tm[env] = dn[env] * (Scalar[dtype](1.0) - is_trunc)
            for d in range(OBS_DIM):
                ob[env, d] = st[env, Self.OBS_OFFSET + d]

        ctx.enqueue_function[step_wrap](
            st, ac, rw, dn, tm, ob, grid_dim=(BLOCKS,), block_dim=(TPB,)
        )

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int, OBS_DIM: Int
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
    ) raises:
        """Copy the normalized obs prefix (OBS_OFFSET=0) out of each env's state.

        Overrides the trait default, whose kernel declares the read-only
        `states` view as `MutAnyOrigin` and so fails to bind the immutable
        view built from the non-`mut` `states` buffer (mirrors the legacy env).
        """
        var st = tile[BATCH_SIZE, STATE_SIZE](states)
        var ob = tile[BATCH_SIZE, OBS_DIM](obs)
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def extract_wrap(
            st: ImmTile[dtype, BATCH_SIZE, STATE_SIZE],
            ob: Tile[dtype, BATCH_SIZE, OBS_DIM],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            for d in range(OBS_DIM):
                ob[env, d] = st[env, Self.OBS_OFFSET + d]

        ctx.enqueue_function[extract_wrap](
            st, ob, grid_dim=(BLOCKS,), block_dim=(TPB,)
        )

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[dtype]) raises:
        pass

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[dtype],
        curriculum_values: List[Scalar[dtype]],
    ) raises:
        pass
