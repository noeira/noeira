"""CarRacing environment on the Box2D-faithful multi-body physics (CPU).

This is the successor to the legacy single-body `CarRacing`: it drives the car
with `CarDynamicsMB` (hull + 4 wheels + 4 revolute joints, solved with the
physics2d sequential-impulse pipeline), which removes the off-track
spin-in-place pathology of the legacy model. Key differences vs the legacy env:

  - Physics: multi-body (faithful to Gymnasium's Box2D car), not a lumped body.
  - Friction: looked up PER WHEEL from its own body position (road vs grass).
  - Reward: Gymnasium-faithful `-0.1/frame + 1000/N` per visited tile (NOT the
    legacy GPU path's velocity-shaped reward).
  - Observation: 13-D, NORMALIZED for neural nets (legacy obs were raw).

State row layout (all per-env, CPU BATCH=1):
  [OBS | BODIES | FORCES | JOINTS | ROLLING | CONTROLS | METADATA]
The procedural track is kept CPU-side (List[TrackTile] + a tiles buffer), as in
the legacy CPU path; the GPU/embedded-track + discrete-action variants come next.

Reference: gymnasium/envs/box2d/car_racing.py.
"""

from std.math import sqrt, sin, cos
from std.memory import alloc

from noeira.render import (
    Renderer2D,
    RotatingCamera,
    Transform2D,
    SDL_Color,
    Vec2 as RenderVec2,
    car_red,
    black,
)

from noeira.core import BoxDiscreteActionEnv, BoxContinuousActionEnv
from noeira.physics2d import dtype
from noeira.physics2d.car import CarDynamicsMB, TileCollision
from noeira.physics2d.car.constants import (
    FRICTION_LIMIT,
    CAR_DT,
    CTRL_STEERING,
    CTRL_GAS,
    CTRL_BRAKE,
    STEERING_LIMIT,
    TILE_DATA_SIZE,
    MAX_TRACK_TILES,
)
from noeira.physics2d.constants import (
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    BODY_STATE_SIZE,
)

from .track import TrackGenerator
from .constants import CRConstants
from .car_racing_pixel import CarRacingPixel
from .state import CarRacingState
from .action import CarRacingAction
from noeira.core.tiles import Tile, tile


struct CarRacingMB[DTYPE: DType, PIXEL_OBS: Bool = False, PIX_RES: Int = 84](
    BoxDiscreteActionEnv, BoxContinuousActionEnv, Copyable, Movable
):
    """CarRacing on multi-body physics (CPU, single env).

    Conforms to BOTH `BoxDiscreteActionEnv` (Rainbow / discrete hybrid) and
    `BoxContinuousActionEnv` (DreamerV3 / SAC continuous) — the underlying
    multi-body physics is continuous (steer/gas/brake floats); the discrete path
    just decodes 5 actions onto it. Wrappable by `BatchedCpuDiscreteEnv` /
    `BatchedCpuEnv` for hybrid training (CPU env stepped on host + GPU agent),
    guaranteeing CPU↔GPU transfer and faithful (non-cheatable) closed-loop
    tracks. `PIXEL_OBS=True` exposes the 4×PIX_RES×PIX_RES pixel observation
    through the trait (for the CNN agent); `False` exposes the 13-D clean obs.
    `PIX_RES` (default 84) = the square pixel resolution; the DreamerV3 conv
    path uses 96 (16-divisible)."""

    comptime dtype = Self.DTYPE  # BoxDiscreteActionEnv requirement
    comptime StateType = CarRacingState[Self.DTYPE]
    comptime ActionType = CarRacingAction[Self.DTYPE]

    # --- compile-time layout (within one env's state row) ----------------
    comptime OBS_DIM: Int = 13  # clean-obs layout (state prefix)
    comptime ACTION_DIM: Int = 3
    comptime NUM_ACTIONS: Int = 5  # discrete: noop/left/right/gas/brake
    # Observation dim exposed through the trait (pixel stack vs clean prefix).
    comptime PIX_DIM: Int = CarRacingPixel[Self.DTYPE, Self.PIX_RES].OBS_DIM
    comptime EFF_OBS_DIM: Int = Self.PIX_DIM if Self.PIXEL_OBS else Self.OBS_DIM
    comptime NB: Int = CarDynamicsMB.NUM_BODIES
    comptime NJ: Int = CarDynamicsMB.NUM_JOINTS
    comptime NW: Int = CarDynamicsMB.NUM_WHEELS
    comptime MAX_TILES: Int = MAX_TRACK_TILES

    comptime OBS_OFFSET: Int = 0
    comptime BODIES_OFFSET: Int = Self.OBS_OFFSET + Self.OBS_DIM
    comptime FORCES_OFFSET: Int = Self.BODIES_OFFSET + Self.NB * BODY_STATE_SIZE
    comptime JOINTS_OFFSET: Int = Self.FORCES_OFFSET + Self.NB * 3
    comptime ROLLING_OFFSET: Int = Self.JOINTS_OFFSET + Self.NJ * 17
    comptime CONTROLS_OFFSET: Int = Self.ROLLING_OFFSET + Self.NW
    comptime METADATA_OFFSET: Int = Self.CONTROLS_OFFSET + 3
    comptime STATE_SIZE: Int = Self.METADATA_OFFSET + 6

    # --- observation normalization scales --------------------------------
    comptime POS_SCALE: Float64 = CRConstants.PLAYFIELD  # ~333
    comptime VEL_SCALE: Float64 = 100.0  # top speed
    comptime OMEGA_SCALE: Float64 = 5.0
    comptime ROLL_SCALE: Float64 = 200.0

    var state_buffer: List[Scalar[dtype]]
    var tiles_buffer: List[Scalar[dtype]]
    var track: TrackGenerator[DType.float64]

    var step_count: Int
    var total_reward: Float64
    var done: Bool
    var truncated: Bool
    var tiles_visited: Int
    var max_steps: Int
    var lap_complete_percent: Float64
    var reset_seed: UInt64

    # Renderer (RenderableEnv); transient — never copied/moved.
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    # CPU pixel-observation frame stack — lets a pixel-trained CNN agent be
    # eval-rendered with the real SDL color scene (render_frame) while it acts
    # on the SAME 84x84 grayscale view the GPU CarRacingPixel env produces.
    var _pixel_stack: List[Scalar[Self.DTYPE]]
    var _pixel_idx: Int

    def __init__(out self, max_steps: Int = CRConstants.MAX_STEPS):
        self.state_buffer = List[Scalar[dtype]](capacity=Self.STATE_SIZE)
        for _ in range(Self.STATE_SIZE):
            self.state_buffer.append(Scalar[dtype](0.0))
        var tsz = Self.MAX_TILES * TILE_DATA_SIZE
        self.tiles_buffer = List[Scalar[dtype]](capacity=tsz)
        for _ in range(tsz):
            self.tiles_buffer.append(Scalar[dtype](0.0))
        self.track = TrackGenerator[DType.float64]()
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.truncated = False
        self.tiles_visited = 0
        self.max_steps = max_steps
        self.lap_complete_percent = CRConstants.LAP_COMPLETE_PERCENT
        self.reset_seed = 0
        self._renderer = None
        self._renderer_initialized = False
        self._pixel_stack = List[Scalar[Self.DTYPE]](
            capacity=CarRacingPixel[Self.DTYPE, Self.PIX_RES].OBS_DIM
        )
        for _ in range(CarRacingPixel[Self.DTYPE, Self.PIX_RES].OBS_DIM):
            self._pixel_stack.append(Scalar[Self.DTYPE](0.0))
        self._pixel_idx = 0

    def __init__(out self, *, copy: Self):
        self.state_buffer = copy.state_buffer.copy()
        self.tiles_buffer = copy.tiles_buffer.copy()
        self.track = TrackGenerator[DType.float64]()
        self.track.track = copy.track.track.copy()
        self.track.track_length = copy.track.track_length
        self.step_count = copy.step_count
        self.total_reward = copy.total_reward
        self.done = copy.done
        self.truncated = copy.truncated
        self.tiles_visited = copy.tiles_visited
        self.max_steps = copy.max_steps
        self.lap_complete_percent = copy.lap_complete_percent
        self.reset_seed = copy.reset_seed
        self._renderer = None  # do not copy renderer
        self._renderer_initialized = False
        self._pixel_stack = copy._pixel_stack.copy()
        self._pixel_idx = copy._pixel_idx

    def __init__(out self, *, deinit move: Self):
        self.state_buffer = move.state_buffer^
        self.tiles_buffer = move.tiles_buffer^
        self.track = move.track^
        self.step_count = move.step_count
        self.total_reward = move.total_reward
        self.done = move.done
        self.truncated = move.truncated
        self.tiles_visited = move.tiles_visited
        self.max_steps = move.max_steps
        self.lap_complete_percent = move.lap_complete_percent
        self.reset_seed = move.reset_seed
        self._renderer = move._renderer
        self._renderer_initialized = move._renderer_initialized
        self._pixel_stack = move._pixel_stack^
        self._pixel_idx = move._pixel_idx

    # --- internal tensor views -------------------------------------------
    def _state(self) -> Tile[dtype, 1, Self.STATE_SIZE]:
        return tile[1, Self.STATE_SIZE](rebind[Pointer[Scalar[dtype], MutAnyOrigin]]( self.state_buffer.unsafe_ptr() ))

    def _tiles(self) -> Tile[dtype, Self.MAX_TILES, TILE_DATA_SIZE]:
        return tile[Self.MAX_TILES, TILE_DATA_SIZE](rebind[Pointer[Scalar[dtype], MutAnyOrigin]]( self.tiles_buffer.unsafe_ptr() ))

    # --- API --------------------------------------------------------------
    def obs_dim(self) -> Int:
        return Self.EFF_OBS_DIM  # pixel stack or clean prefix per PIXEL_OBS

    def action_dim(self) -> Int:
        return Self.ACTION_DIM

    # --- ContinuousActionEnv bounds (symmetric [-1, 1]) ------------------
    def action_low(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](-1.0)

    def action_high(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](1.0)

    def reset(mut self) -> Self.StateType:
        """Generate a new track, place the car at the start (Env trait).

        Returns the StateType for trait conformance; obs consumers use
        `reset_obs_list` (which calls this and discards the state)."""
        var seed = self.reset_seed + 1
        self.reset_seed = seed
        self.track.generate_random_track(UInt64(seed * 2654435761 + 42), False)
        self.track.to_buffer[Self.MAX_TILES](self._tiles())

        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.truncated = False
        self.tiles_visited = 0

        var start = self.track.get_start_position()
        var sx = Scalar[dtype](Float64(start[0]))
        var sy = Scalar[dtype](Float64(start[1]))
        var sa = Scalar[dtype](Float64(start[2]))

        CarDynamicsMB.init_env[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.FORCES_OFFSET,
            Self.JOINTS_OFFSET, Self.ROLLING_OFFSET,
        ](0, self._state(), sx, sy, sa)

        var st = self._state()
        st[0, Self.CONTROLS_OFFSET + CTRL_STEERING] = Scalar[dtype](0.0)
        st[0, Self.CONTROLS_OFFSET + CTRL_GAS] = Scalar[dtype](0.0)
        st[0, Self.CONTROLS_OFFSET + CTRL_BRAKE] = Scalar[dtype](0.0)

        self._write_obs()
        return Self.StateType()

    def step(
        mut self, steering: Float64, gas: Float64, brake: Float64
    ) -> Tuple[List[Scalar[Self.DTYPE]], Float64, Bool]:
        """Step with raw controls: steering in [-1,1], gas/brake in [0,1]."""
        var st = self._state()
        st[0, Self.CONTROLS_OFFSET + CTRL_STEERING] = Scalar[dtype](
            clamp(steering, -1.0, 1.0)
        )
        st[0, Self.CONTROLS_OFFSET + CTRL_GAS] = Scalar[dtype](
            clamp(gas, 0.0, 1.0)
        )
        st[0, Self.CONTROLS_OFFSET + CTRL_BRAKE] = Scalar[dtype](
            clamp(brake, 0.0, 1.0)
        )

        # Per-wheel friction from each wheel body's world position.
        var tiles = self._tiles()
        var n = self.track.track_length
        var fr = Array[Scalar[dtype], 4](fill=Scalar[dtype](0))
        for w in range(Self.NW):
            var wp = CarDynamicsMB.wheel_world_pos[
                1, Self.STATE_SIZE, Self.BODIES_OFFSET
            ](0, st, w)
            fr[w] = TileCollision.get_friction_limit_at[Self.MAX_TILES](
                wp[0], wp[1], tiles, n
            )

        CarDynamicsMB.step_single_env_pw[
            1, Self.STATE_SIZE, Self.BODIES_OFFSET, Self.FORCES_OFFSET,
            Self.JOINTS_OFFSET, Self.ROLLING_OFFSET, Self.CONTROLS_OFFSET,
        ](0, st, fr[0], fr[1], fr[2], fr[3], Scalar[dtype](CAR_DT))

        self.step_count += 1

        # Reward: -0.1 per frame + 1000/N per newly visited tile (Gymnasium).
        var hx = rebind[Scalar[dtype]](
            st[0, Self.BODIES_OFFSET + IDX_X]
        )
        var hy = rebind[Scalar[dtype]](
            st[0, Self.BODIES_OFFSET + IDX_Y]
        )
        var reward: Float64 = -0.1
        var tile_idx = TileCollision.check_tile_visited[Self.MAX_TILES](
            hx, hy, tiles, n
        )
        if tile_idx >= 0:
            if self.track.mark_tile_visited(tile_idx):
                self.tiles_visited += 1
                reward += 1000.0 / Float64(max(n, 1))

        # Termination.
        var fhx = Float64(hx)
        var fhy = Float64(hy)
        var pf = Float64(CRConstants.PLAYFIELD)
        var ax = fhx if fhx >= 0.0 else -fhx
        var ay = fhy if fhy >= 0.0 else -fhy
        if ax > pf or ay > pf:
            self.done = True
            reward = -100.0
        var progress = Float64(self.tiles_visited) / Float64(max(n, 1))
        if progress >= self.lap_complete_percent:
            self.done = True
        if self.max_steps > 0 and self.step_count >= self.max_steps:
            self.done = True
            self.truncated = True

        self.total_reward += reward
        self._write_obs()
        return (self._obs_list(), reward, self.done)

    # --- observation ------------------------------------------------------
    def _write_obs(mut self):
        var st = self._state()
        var ho = Self.BODIES_OFFSET  # hull is body 0
        var x = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_X]))
        var y = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_Y]))
        var a = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_ANGLE]))
        var vx = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_VX]))
        var vy = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_VY]))
        var om = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_OMEGA]))
        var spd = sqrt(vx * vx + vy * vy)

        # front wheel steering angle (FL relative to hull)
        var fl_a = Float64(
            rebind[Scalar[dtype]](
                st[0, Self.BODIES_OFFSET + 1 * BODY_STATE_SIZE + IDX_ANGLE]
            )
        )
        var steer = fl_a - a

        var o = Self.OBS_OFFSET
        st[0, o + 0] = Scalar[dtype](x / Self.POS_SCALE)
        st[0, o + 1] = Scalar[dtype](y / Self.POS_SCALE)
        st[0, o + 2] = Scalar[dtype](sin(a))
        st[0, o + 3] = Scalar[dtype](cos(a))
        st[0, o + 4] = Scalar[dtype](vx / Self.VEL_SCALE)
        st[0, o + 5] = Scalar[dtype](vy / Self.VEL_SCALE)
        st[0, o + 6] = Scalar[dtype](om / Self.OMEGA_SCALE)
        st[0, o + 7] = Scalar[dtype](steer / STEERING_LIMIT)
        # rolling wheel omegas (4), normalized
        for w in range(Self.NW):
            var rw = Float64(
                rebind[Scalar[dtype]](st[0, Self.ROLLING_OFFSET + w])
            )
            st[0, o + 8 + w] = Scalar[dtype](rw / Self.ROLL_SCALE)
        st[0, o + 12] = Scalar[dtype](spd / Self.VEL_SCALE)

    def _obs_list(self) -> List[Scalar[Self.DTYPE]]:
        var st = self._state()
        var out = List[Scalar[Self.DTYPE]](capacity=Self.OBS_DIM)
        for i in range(Self.OBS_DIM):
            out.append(Scalar[Self.DTYPE](rebind[Scalar[dtype]](st[0, i])))
        return out^

    # --- accessors (for tests / inspection) ------------------------------
    def hull_pos(self) -> Tuple[Float64, Float64]:
        var st = self._state()
        return (
            Float64(rebind[Scalar[dtype]](st[0, Self.BODIES_OFFSET + IDX_X])),
            Float64(rebind[Scalar[dtype]](st[0, Self.BODIES_OFFSET + IDX_Y])),
        )

    def hull_speed(self) -> Float64:
        var st = self._state()
        var vx = Float64(rebind[Scalar[dtype]](st[0, Self.BODIES_OFFSET + IDX_VX]))
        var vy = Float64(rebind[Scalar[dtype]](st[0, Self.BODIES_OFFSET + IDX_VY]))
        return sqrt(vx * vx + vy * vy)

    def track_length(self) -> Int:
        return self.track.track_length

    def on_grass(self) -> Bool:
        """True if the hull center is off all track tiles (on grass).

        Debug/HUD helper — same hull-position tile test `step()` uses for
        the reward, so it reflects the surface the car is actually on.
        """
        var hp = self.hull_pos()
        var tiles = self._tiles()
        var idx = TileCollision.check_tile_visited[Self.MAX_TILES](
            Scalar[dtype](hp[0]),
            Scalar[dtype](hp[1]),
            tiles,
            self.track.track_length,
        )
        return idx < 0

    # --- CPU pixel observation (matches the GPU CarRacingPixel rasterizer) -
    # Same camera + colors as CarRacingPixel._render_pixel (referenced, not
    # re-derived) so a CNN trained on the GPU pixel env sees in-distribution
    # input. The track shape differs (faithful CPU vs simplified GPU track) but
    # the rendered road appearance is identical.
    def _render_pixel_cpu(
        self, dx: Int, dy: Int, vis: List[Int]
    ) -> Scalar[Self.DTYPE]:
        comptime P = CarRacingPixel[Self.DTYPE, Self.PIX_RES]
        var zero = Scalar[dtype](0.0)
        var camx = (Scalar[dtype](dx) - Scalar[dtype](P.CX)) / Scalar[dtype](
            P.ZOOM_PX
        )
        var camy = (Scalar[dtype](P.CY) - Scalar[dtype](dy)) / Scalar[dtype](
            P.ZOOM_PX
        )
        var acx = camx if camx >= zero else -camx
        var acy = camy if camy >= zero else -camy
        if acx < Scalar[dtype](P.CAR_HW) and acy < Scalar[dtype](P.CAR_HL):
            return Scalar[Self.DTYPE](P.C_CAR)

        var st = self._state()
        var ho = Self.BODIES_OFFSET
        var car_x = rebind[Scalar[dtype]](st[0, ho + IDX_X])
        var car_y = rebind[Scalar[dtype]](st[0, ho + IDX_Y])
        var a = rebind[Scalar[dtype]](st[0, ho + IDX_ANGLE])
        var ca = cos(a)
        var sa = sin(a)
        var wx = car_x + camx * ca - camy * sa
        var wy = car_y + camx * sa + camy * ca

        # Only the camera-culled visible tiles (computed once per frame).
        for j in range(len(vis)):
            var t = self.track.track[vis[j]]
            if TileCollision.point_in_quad(
                wx, wy,
                Scalar[dtype](t.v0_x), Scalar[dtype](t.v0_y),
                Scalar[dtype](t.v1_x), Scalar[dtype](t.v1_y),
                Scalar[dtype](t.v2_x), Scalar[dtype](t.v2_y),
                Scalar[dtype](t.v3_x), Scalar[dtype](t.v3_y),
            ):
                return Scalar[Self.DTYPE](P.C_ROAD)
        return Scalar[Self.DTYPE](P.C_GRASS)

    def _push_pixel_frame(mut self):
        comptime P = CarRacingPixel[Self.DTYPE, Self.PIX_RES]
        # Cull tiles to the camera view once per frame (~10x fewer point-in-quad
        # tests), mirroring the GPU rasterizer.
        var st = self._state()
        var ho = Self.BODIES_OFFSET
        var car_x = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_X]))
        var car_y = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_Y]))
        var vis = List[Int]()
        for i in range(self.track.track_length):
            var t = self.track.track[i]
            var ddx = Float64(t.center_x) - car_x
            var ddy = Float64(t.center_y) - car_y
            if ddx * ddx + ddy * ddy < P.CULL_R2:
                vis.append(i)

        var base = self._pixel_idx * P.FRAME_SIZE
        for dy in range(P.OBS_H):
            for dx in range(P.OBS_W):
                self._pixel_stack[base + dy * P.OBS_W + dx] = (
                    self._render_pixel_cpu(dx, dy, vis)
                )
        self._pixel_idx = (self._pixel_idx + 1) % P.FRAME_STACK

    def get_pixel_obs(self) -> List[Scalar[Self.DTYPE]]:
        """Chronological (oldest→newest) 4×84×84 frame stack — matches the GPU
        env's obs ordering."""
        comptime P = CarRacingPixel[Self.DTYPE, Self.PIX_RES]
        var out = List[Scalar[Self.DTYPE]](capacity=P.OBS_DIM)
        for f in range(P.FRAME_STACK):
            var rs = (self._pixel_idx + f) % P.FRAME_STACK
            var base = rs * P.FRAME_SIZE
            for i in range(P.FRAME_SIZE):
                out.append(self._pixel_stack[base + i])
        return out^

    def reset_pixel(mut self) -> List[Scalar[Self.DTYPE]]:
        """Reset + fill the frame stack with the initial view; return pixel obs.
        """
        _ = self.reset()
        self._pixel_idx = 0
        for _ in range(CarRacingPixel[Self.DTYPE, Self.PIX_RES].FRAME_STACK):
            self._push_pixel_frame()
        return self.get_pixel_obs()

    def step_action_pixel(
        mut self, a: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Float64, Bool]:
        """Discrete step returning the PIXEL observation (for SDL color eval)."""
        var r = self.step_action(a)
        self._push_pixel_frame()
        return (self.get_pixel_obs(), r[1], r[2])

    # --- BoxDiscreteActionEnv interface (for BatchedCpuDiscreteEnv / hybrid) -
    # Obs is the pixel stack when PIXEL_OBS else the 13-D clean prefix.
    def num_actions(self) -> Int:
        return Self.NUM_ACTIONS

    def was_terminated(self) -> Bool:
        """True iff the last step ended by natural termination (off-playfield /
        lap), NOT max-steps truncation — so the agent cuts the TD bootstrap."""
        return self.done and not self.truncated

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        comptime if Self.PIXEL_OBS:
            return self.reset_pixel()
        else:
            _ = self.reset()
            return self._obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        comptime if Self.PIXEL_OBS:
            var r = self.step_action_pixel(action)
            return (r[0].copy(), Scalar[Self.dtype](r[1]), r[2])
        else:
            var r = self.step_action(action)
            return (r[0].copy(), Scalar[Self.dtype](r[1]), r[2])

    def step_obs_into(
        mut self,
        action: Int,
        obs_out: Pointer[Scalar[Self.dtype], MutAnyOrigin],
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Allocation-light step: write the observation straight into obs_out."""
        comptime if Self.PIXEL_OBS:
            var r = self.step_action_pixel(action)
            for d in range(Self.PIX_DIM):
                obs_out[unsafe_offset=d] = r[0][d]
            return (Scalar[Self.dtype](r[1]), r[2])
        else:
            var r = self.step_action(action)
            for d in range(Self.OBS_DIM):
                obs_out[unsafe_offset=d] = r[0][d]
            return (Scalar[Self.dtype](r[1]), r[2])

    # --- remaining Env-trait surface (BatchedCpuDiscreteEnv uses the obs
    # methods above; these satisfy conformance) --------------------------
    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        var r = self.step(
            Float64(action.steering), Float64(action.gas), Float64(action.brake)
        )
        return (Self.StateType(), Scalar[Self.dtype](r[1]), r[2])

    def get_state(mut self) -> Self.StateType:
        return Self.StateType()

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        comptime if Self.PIXEL_OBS:
            return self.get_pixel_obs()
        else:
            return self._obs_list()

    def action_from_index(self, action_idx: Int) -> Self.ActionType:
        return CarRacingAction[Self.DTYPE].from_discrete(action_idx)

    def close(mut self):
        pass

    # --- discrete action (decode matches CarRacingDiscrete) --------------
    def step_action(
        mut self, a: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Float64, Bool]:
        """Step with a discrete action index (Gymnasium CarRacing-v3 discrete),
        decoded identically to CarRacingDiscrete:
        0=noop, 1=left, 2=right, 3=gas, 4=brake."""
        var steer = 0.0
        var gas = 0.0
        var brake = 0.0
        if a == 1:
            steer = -1.0
        elif a == 2:
            steer = 1.0
        elif a == 3:
            gas = 0.2
        elif a == 4:
            brake = 0.8
        return self.step(steer, gas, brake)

    # --- BoxContinuousActionEnv: 3-D action [steer, gas, brake] ----------
    # Gymnasium CarRacing-v3 continuous convention: steering in [-1,1]; gas and
    # brake arrive in [-1,1] (Gaussian-policy range) and remap to [0,1] via
    # (a+1)/2. Returns the EFF obs (pixel stack when PIXEL_OBS, else 13-D clean).
    def _step_continuous_eff[
        DTYPE_C: DType
    ](
        mut self, steer: Float64, gas_raw: Float64, brake_raw: Float64
    ) -> Tuple[List[Scalar[DTYPE_C]], Scalar[DTYPE_C], Bool]:
        var gas = (gas_raw + 1.0) * 0.5
        var brake = (brake_raw + 1.0) * 0.5
        var r = self.step(steer, gas, brake)  # clean obs, reward, done
        comptime if Self.PIXEL_OBS:
            self._push_pixel_frame()
            var px = self.get_pixel_obs()
            var obs = List[Scalar[DTYPE_C]](capacity=len(px))
            for i in range(len(px)):
                obs.append(Scalar[DTYPE_C](px[i]))
            return (obs^, Scalar[DTYPE_C](r[1]), r[2])
        else:
            var obs = List[Scalar[DTYPE_C]](capacity=len(r[0]))
            for i in range(len(r[0])):
                obs.append(Scalar[DTYPE_C](r[0][i]))
            return (obs^, Scalar[DTYPE_C](r[1]), r[2])

    def step_continuous[
        DTYPE_SC: DType
    ](mut self, action: Scalar[DTYPE_SC]) -> Tuple[
        List[Scalar[DTYPE_SC]], Scalar[DTYPE_SC], Bool
    ]:
        """1-D action → steering only (gas/brake off)."""
        return self._step_continuous_eff[DTYPE_SC](
            Float64(action), -1.0, -1.0
        )

    def step_continuous_vec[
        DTYPE_VEC: DType
    ](
        mut self, action: List[Scalar[DTYPE_VEC]], verbose: Bool = False
    ) -> Tuple[List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool]:
        """3-D action [steering, gas, brake] (gas/brake in [-1,1] → [0,1])."""
        var steer = Float64(action[0]) if len(action) > 0 else 0.0
        var gas_raw = Float64(action[1]) if len(action) > 1 else 0.0
        var brake_raw = Float64(action[2]) if len(action) > 2 else 0.0
        return self._step_continuous_eff[DTYPE_VEC](steer, gas_raw, brake_raw)

    # --- rendering (RenderableEnv) ---------------------------------------
    def init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(
            Renderer2D(
                CRConstants.WINDOW_W,
                CRConstants.WINDOW_H,
                CRConstants.FPS,
                "CarRacing (multi-body)",
            )
        )
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self.render(self._renderer.value()[])

    def close_renderer(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].close()
        self._renderer.value().unsafe_free()
        self._renderer_initialized = False

    def is_renderer_open(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return not self._renderer.value()[].get_should_quit()

    def check_renderer_quit(mut self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].get_should_quit()

    def renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer_delay(ms)

    def start_recording(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].start_recording(filename, fps, skip)

    def stop_recording(mut self) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].stop_recording()

    def render(mut self, mut renderer: Renderer2D):
        """Top-down rotating-camera view: grass bg, track, car."""
        var st = self._state()
        var ho = Self.BODIES_OFFSET
        var hx = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_X]))
        var hy = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_Y]))
        var ha = Float64(rebind[Scalar[dtype]](st[0, ho + IDX_ANGLE]))

        var bg = SDL_Color(102, 204, 102, 255)  # grass green
        if not renderer.begin_frame_with_color(bg):
            return

        var zoom = CRConstants.ZOOM * CRConstants.SCALE
        var camera = renderer.make_rotating_camera_offset(
            hx, hy, -ha, zoom,
            Float64(CRConstants.WINDOW_W) / 2.0,
            Float64(CRConstants.WINDOW_H) * 3.0 / 4.0,
        )

        # Track tiles (gray; green-tinted once visited).
        var road = SDL_Color(102, 102, 102, 255)
        var road_seen = SDL_Color(72, 132, 72, 255)
        for i in range(self.track.track_length):
            var t = self.track.track[i]
            var dist = sqrt(
                (Float64(t.center_x) - hx) ** 2 + (Float64(t.center_y) - hy) ** 2
            )
            if dist > 500.0:
                continue
            var col = road_seen if t.visited else road
            var verts = List[RenderVec2]()
            verts.append(RenderVec2(Float64(t.v0_x), Float64(t.v0_y)))
            verts.append(RenderVec2(Float64(t.v1_x), Float64(t.v1_y)))
            verts.append(RenderVec2(Float64(t.v2_x), Float64(t.v2_y)))
            verts.append(RenderVec2(Float64(t.v3_x), Float64(t.v3_y)))
            renderer.draw_polygon_rotating(verts, camera, col, filled=True)

        self._draw_car(renderer, camera, hx, hy, ha)
        renderer.flip()

    def _draw_car(
        self,
        mut renderer: Renderer2D,
        camera: RotatingCamera,
        hx: Float64,
        hy: Float64,
        ha: Float64,
    ):
        var st = self._state()
        var sz = CRConstants.SIZE
        var red = car_red()
        var hull_tf = Transform2D(hx, hy, ha)

        # Hull body — four polygons (front spoiler, cabin, body, rear spoiler).
        var p1 = List[RenderVec2]()
        p1.append(RenderVec2(-60.0 * sz, 130.0 * sz))
        p1.append(RenderVec2(60.0 * sz, 130.0 * sz))
        p1.append(RenderVec2(60.0 * sz, 110.0 * sz))
        p1.append(RenderVec2(-60.0 * sz, 110.0 * sz))
        renderer.draw_transformed_polygon_rotating(p1, hull_tf, camera, red, filled=True)

        var p2 = List[RenderVec2]()
        p2.append(RenderVec2(-15.0 * sz, 120.0 * sz))
        p2.append(RenderVec2(15.0 * sz, 120.0 * sz))
        p2.append(RenderVec2(20.0 * sz, 20.0 * sz))
        p2.append(RenderVec2(-20.0 * sz, 20.0 * sz))
        renderer.draw_transformed_polygon_rotating(p2, hull_tf, camera, red, filled=True)

        var p3 = List[RenderVec2]()
        p3.append(RenderVec2(25.0 * sz, 20.0 * sz))
        p3.append(RenderVec2(50.0 * sz, -10.0 * sz))
        p3.append(RenderVec2(50.0 * sz, -40.0 * sz))
        p3.append(RenderVec2(20.0 * sz, -90.0 * sz))
        p3.append(RenderVec2(-20.0 * sz, -90.0 * sz))
        p3.append(RenderVec2(-50.0 * sz, -40.0 * sz))
        p3.append(RenderVec2(-50.0 * sz, -10.0 * sz))
        p3.append(RenderVec2(-25.0 * sz, 20.0 * sz))
        renderer.draw_transformed_polygon_rotating(p3, hull_tf, camera, red, filled=True)

        var p4 = List[RenderVec2]()
        p4.append(RenderVec2(-50.0 * sz, -120.0 * sz))
        p4.append(RenderVec2(50.0 * sz, -120.0 * sz))
        p4.append(RenderVec2(50.0 * sz, -90.0 * sz))
        p4.append(RenderVec2(-50.0 * sz, -90.0 * sz))
        renderer.draw_transformed_polygon_rotating(p4, hull_tf, camera, red, filled=True)

        # Wheels drawn at their actual rigid-body poses.
        var blk = black()
        var hw = 14.0 * sz
        var hr = 27.0 * sz
        for w in range(Self.NW):
            var wbo = Self.BODIES_OFFSET + (w + 1) * BODY_STATE_SIZE
            var wx = Float64(rebind[Scalar[dtype]](st[0, wbo + IDX_X]))
            var wy = Float64(rebind[Scalar[dtype]](st[0, wbo + IDX_Y]))
            var wa = Float64(rebind[Scalar[dtype]](st[0, wbo + IDX_ANGLE]))
            var wtf = Transform2D(wx, wy, wa)
            var wv = List[RenderVec2]()
            wv.append(RenderVec2(-hw, hr))
            wv.append(RenderVec2(hw, hr))
            wv.append(RenderVec2(hw, -hr))
            wv.append(RenderVec2(-hw, -hr))
            renderer.draw_transformed_polygon_rotating(wv, wtf, camera, blk, filled=True)


def clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
