"""CPU PushT environment (PushTEnv).

Single-env CPU implementation of the PushT task. Reuses the GPU-compatible
strided [BATCH=1, STATE_SIZE] state layout so the step machinery is shared
with the upcoming GPU batched env (PushTV2).

Trait conformance:
    BoxContinuousActionEnv (obs as List[Float], 2D continuous action)

Step pipeline (per env step):
    PD-controller → for SUBSTEPS:
        1. Wall + circle-vs-T contacts
        2. solve_velocity × VEL_ITERATIONS (ImpulseSolver)
        3. integrate_positions_single_env (SemiImplicitEuler)
        4. solve_position_single_env × POS_ITERATIONS
    → coverage reward → keypoints obs
"""

from std.math import cos, sin, sqrt, pi
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.memory import alloc
from layout import LayoutTensor, Layout

from noeira.core import (
    BoxContinuousActionEnv,
    ContinuousStateEnv,
    ContinuousActionEnv,
    Env,
    RenderableEnv,
)
from noeira.render import (
    Renderer2D,
    SDL_Color,
    SDL_Point,
    rgb,
    light_gray,
    white,
    black,
)
from noeira.physics2d.constants import (
    dtype,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_MASS,
    IDX_SHAPE,
)
from noeira.physics2d.integrators.euler import SemiImplicitEuler
from noeira.physics2d.solvers.impulse import ImpulseSolver
from noeira.physics2d.collision.circle_polygon import (
    detect_circle_vs_body_pair,
)

from .constants import PConstants, PushTLayout, PushTShapeBuf
from .state import PushTState
from .action import PushTAction
from .geometry import (
    init_pusht_shape_buffer,
    get_t_keypoints_world,
    compute_coverage,
    t_rect_long_vertex,
    t_body_from_origin,
    t_origin_from_body,
    t_rect_stem_vertex,
)
from .physics import (
    pd_update_agent_single_env,
    detect_wall_contacts_single_env,
    pusht_substep_single_env,
)


# Inertia for one rectangle's vertices about the body origin (parallel-axis
# included implicitly through the polygon moment formula). pymunk uses:
#   I = m/(6 * Σ|cross|) * Σ (|cross| * (v_i·v_i + v_i·v_{i+1} + v_{i+1}·v_{i+1}))
# where cross = v_i × v_{i+1}. We replicate it for the long-bar verts and
# (per the reference's bug) use the SAME verts for both rect inertias.
def _compute_t_inertia() -> Scalar[dtype]:
    var num = Scalar[dtype](0.0)
    var den = Scalar[dtype](0.0)
    for i in range(4):
        var p0 = t_rect_long_vertex(i)
        var p1 = t_rect_long_vertex((i + 1) % 4)
        var c = p0[0] * p1[1] - p0[1] * p1[0]
        if c < Scalar[dtype](0.0):
            c = -c
        var term = (
            p0[0] * p0[0]
            + p0[1] * p0[1]
            + p0[0] * p1[0]
            + p0[1] * p1[1]
            + p1[0] * p1[0]
            + p1[1] * p1[1]
        )
        num = num + c * term
        den = den + c
    var I_one = Scalar[dtype](PConstants.T_MASS) * num / (
        Scalar[dtype](6.0) * den
    )
    # pymunk: inertia1 + inertia2 with inertia2 also computed from vertices1
    return Scalar[dtype](2.0) * I_one


struct PushTEnv[DTYPE: DType](
    BoxContinuousActionEnv, Copyable, Movable, RenderableEnv
):
    """CPU PushT environment.

    Observation: 18D = 16 T-keypoints (x0,y0,...,x7,y7) ++ agent_pos (x,y).
    Action: 2D continuous target position in [0, 512]².
    Reward: clip(coverage / SUCCESS_THRESHOLD, 0, 1).
    Terminates when coverage > SUCCESS_THRESHOLD.
    """

    comptime dtype = Self.DTYPE
    comptime StateType = PushTState[Self.DTYPE]
    comptime ActionType = PushTAction[Self.DTYPE]

    comptime BATCH: Int = 1
    comptime STATE_SIZE: Int = PushTLayout.STATE_SIZE
    comptime MAX_CONTACTS: Int = PushTLayout.MAX_CONTACTS
    comptime NUM_SHAPES: Int = PushTShapeBuf.NUM_SHAPES

    # Owned storage (inline; pointers stay valid for the env's lifetime)
    var state_data: Array[Scalar[dtype], PushTLayout.STATE_SIZE]
    var shapes_data: Array[
        Array[Scalar[dtype], SHAPE_MAX_SIZE],
        PushTShapeBuf.NUM_SHAPES,
    ]
    var contacts_data: Array[
        Scalar[dtype], PushTLayout.MAX_CONTACTS * CONTACT_DATA_SIZE
    ]

    var done: Bool
    var rng_seed: UInt64
    var rng_counter: UInt64

    # Most recent action used by render() to draw the target marker.
    var last_target_x: Scalar[dtype]
    var last_target_y: Scalar[dtype]

    # Renderer (RenderableEnv)
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    # =========================================================================
    # Constructors
    # =========================================================================

    def __init__(out self, seed: UInt64 = 0):
        self.state_data = Array[Scalar[dtype], PushTLayout.STATE_SIZE](
            fill=Scalar[dtype](0.0)
        )
        # Construct shapes_data by filling each inner array.
        var row = Array[Scalar[dtype], SHAPE_MAX_SIZE](
            fill=Scalar[dtype](0.0)
        )
        self.shapes_data = Array[
            Array[Scalar[dtype], SHAPE_MAX_SIZE],
            PushTShapeBuf.NUM_SHAPES,
        ](fill=row)
        self.contacts_data = Array[
            Scalar[dtype], PushTLayout.MAX_CONTACTS * CONTACT_DATA_SIZE
        ](fill=Scalar[dtype](0.0))
        init_pusht_shape_buffer[PushTShapeBuf.NUM_SHAPES](self.shapes_data)
        self.done = False
        self.rng_seed = seed
        self.rng_counter = 0
        self.last_target_x = Scalar[dtype](256.0)
        self.last_target_y = Scalar[dtype](256.0)
        self._renderer = None
        self._renderer_initialized = False
        self._reset_internal()

    def __init__(out self, *, copy: Self):
        self.state_data = copy.state_data.copy()
        self.shapes_data = copy.shapes_data.copy()
        self.contacts_data = copy.contacts_data.copy()
        self.done = copy.done
        self.rng_seed = copy.rng_seed
        self.rng_counter = copy.rng_counter
        self.last_target_x = copy.last_target_x
        self.last_target_y = copy.last_target_y
        # Don't transfer renderer; fresh instance starts uninitialized.
        self._renderer = None
        self._renderer_initialized = False

    def __init__(out self, *, deinit move: Self):
        self.state_data = move.state_data.copy()
        self.shapes_data = move.shapes_data.copy()
        self.contacts_data = move.contacts_data.copy()
        self.done = move.done
        self.rng_seed = move.rng_seed
        self.rng_counter = move.rng_counter
        self.last_target_x = move.last_target_x
        self.last_target_y = move.last_target_y
        self._renderer = move._renderer
        self._renderer_initialized = move._renderer_initialized

    # =========================================================================
    # LayoutTensor views (rebuilt per call — pointer chasing is cheap)
    # =========================================================================

    @always_inline
    def _state_view(
        mut self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(1, PushTLayout.STATE_SIZE), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype,
            Layout.row_major(1, PushTLayout.STATE_SIZE),
        ](Span(self.state_data))

    @always_inline
    def _shapes_view(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(PushTShapeBuf.NUM_SHAPES, SHAPE_MAX_SIZE),
        MutAnyOrigin,
    ]:
        # Array of InlineArrays is contiguous in Mojo, so we can treat
        # the outer storage's pointer as a flat NUM_SHAPES * SHAPE_MAX_SIZE
        # buffer.
        return LayoutTensor[
            dtype,
            Layout.row_major(PushTShapeBuf.NUM_SHAPES, SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ](self.shapes_data.unsafe_ptr().unsafe_bitcast[Scalar[dtype]]().as_unsafe_any_origin())

    @always_inline
    def _contacts_view(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(1, PushTLayout.MAX_CONTACTS, CONTACT_DATA_SIZE),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            dtype,
            Layout.row_major(
                1, PushTLayout.MAX_CONTACTS, CONTACT_DATA_SIZE
            ),
        ](Span(self.contacts_data))

    # =========================================================================
    # Initialization helpers
    # =========================================================================

    def _seed_bodies(
        mut self,
        agent_x: Scalar[dtype],
        agent_y: Scalar[dtype],
        block_x: Scalar[dtype],
        block_y: Scalar[dtype],
        block_angle: Scalar[dtype],
    ):
        var state = self._state_view()
        # Zero all body slots first
        for i in range(2 * BODY_STATE_SIZE):
            state[0, PushTLayout.BODIES_OFFSET + i] = Scalar[dtype](0.0)

        # Agent (body 0): kinematic ⇒ inv_mass=0, inv_inertia=0
        var ao = PushTLayout.BODY_AGENT_OFFSET
        state[0, ao + IDX_X] = agent_x
        state[0, ao + IDX_Y] = agent_y
        state[0, ao + IDX_ANGLE] = Scalar[dtype](0.0)
        state[0, ao + IDX_MASS] = Scalar[dtype](0.0)
        state[0, ao + IDX_INV_MASS] = Scalar[dtype](0.0)
        state[0, ao + IDX_INV_INERTIA] = Scalar[dtype](0.0)
        state[0, ao + IDX_SHAPE] = Scalar[dtype](PushTShapeBuf.SHAPE_AGENT)

        # T-block (body 1)
        var to_ = PushTLayout.BODY_T_OFFSET
        var bxy = t_body_from_origin(block_x, block_y, block_angle)
        state[0, to_ + IDX_X] = bxy[0]
        state[0, to_ + IDX_Y] = bxy[1]
        state[0, to_ + IDX_ANGLE] = block_angle
        var m = Scalar[dtype](PConstants.T_MASS)
        var I = _compute_t_inertia()
        state[0, to_ + IDX_MASS] = m
        state[0, to_ + IDX_INV_MASS] = Scalar[dtype](1.0) / m
        state[0, to_ + IDX_INV_INERTIA] = Scalar[dtype](1.0) / I
        state[0, to_ + IDX_SHAPE] = Scalar[dtype](
            PushTShapeBuf.SHAPE_T_COMPOUND
        )

    def _reset_internal(mut self):
        var rng = PhiloxRandom(seed=self.rng_seed, offset=self.rng_counter)
        var r = rng.step_uniform()
        self.rng_counter = self.rng_counter + 1

        var ar_lo = Scalar[dtype](PConstants.AGENT_RESET_LOW)
        var ar_hi = Scalar[dtype](PConstants.AGENT_RESET_HIGH)
        var br_lo = Scalar[dtype](PConstants.BLOCK_RESET_LOW)
        var br_hi = Scalar[dtype](PConstants.BLOCK_RESET_HIGH)
        var ax = ar_lo + Scalar[dtype](r[0]) * (ar_hi - ar_lo)
        var ay = ar_lo + Scalar[dtype](r[1]) * (ar_hi - ar_lo)
        var bx = br_lo + Scalar[dtype](r[2]) * (br_hi - br_lo)
        var by = br_lo + Scalar[dtype](r[3]) * (br_hi - br_lo)
        var pi_s = Scalar[dtype](pi)
        var ba = (Scalar[dtype](r[0]) * Scalar[dtype](2.0) - Scalar[dtype](1.0)) * pi_s
        # NB: pymunk uses 4 different RNG draws; we reuse r[0] here for the
        # angle which is a minor divergence from the reference — acceptable
        # since we don't claim bit-exact reset reproducibility yet.

        self._seed_bodies(ax, ay, bx, by, ba)
        # Reset metadata + contact count
        var s = self._state_view()
        s[0, PushTLayout.CONTACT_COUNT_OFFSET] = Scalar[dtype](0.0)
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_STEP] = Scalar[dtype](
            0.0
        )
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_DONE] = Scalar[
            dtype
        ](0.0)
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_TOTAL_REWARD] = (
            Scalar[dtype](0.0)
        )
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_COVERAGE] = Scalar[
            dtype
        ](0.0)
        self.done = False
        self._write_obs_to_state()

    @always_inline
    def _write_obs_to_state(mut self):
        var s = self._state_view()
        var to_ = PushTLayout.BODY_T_OFFSET
        var bx = rebind[Scalar[dtype]](s[0, to_ + IDX_X])
        var by = rebind[Scalar[dtype]](s[0, to_ + IDX_Y])
        var ba = rebind[Scalar[dtype]](s[0, to_ + IDX_ANGLE])
        # the physics integrates the T about its cog: back to the origin pose
        var _org = t_origin_from_body(bx, by, ba)
        bx = _org[0]
        by = _org[1]
        var kp = Array[Scalar[dtype], PConstants.KEYPOINTS_DIM](
            fill=Scalar[dtype](0.0)
        )
        get_t_keypoints_world(bx, by, ba, kp)
        for i in range(PConstants.KEYPOINTS_DIM):
            s[0, PushTLayout.OBS_OFFSET + i] = kp[i]
        var ao = PushTLayout.BODY_AGENT_OFFSET
        s[0, PushTLayout.OBS_OFFSET + PConstants.KEYPOINTS_DIM] = rebind[
            Scalar[dtype]
        ](s[0, ao + IDX_X])
        s[0, PushTLayout.OBS_OFFSET + PConstants.KEYPOINTS_DIM + 1] = rebind[
            Scalar[dtype]
        ](s[0, ao + IDX_Y])

    # =========================================================================
    # Physics substep
    # =========================================================================

    def _substep(
        mut self, target_x: Scalar[dtype], target_y: Scalar[dtype]
    ):
        var s = self._state_view()
        var shapes = self._shapes_view()
        var contacts = self._contacts_view()
        pusht_substep_single_env[
            1,
            PushTShapeBuf.NUM_SHAPES,
            PushTLayout.MAX_CONTACTS,
            PushTLayout.STATE_SIZE,
            PushTLayout.NUM_BODIES,
            PushTLayout.BODIES_OFFSET,
            PushTLayout.BODY_AGENT_OFFSET,
            PushTLayout.BODY_T_OFFSET,
            PushTLayout.BODY_AGENT,
            PushTLayout.BODY_T,
            PushTLayout.CONTACT_COUNT_OFFSET,
            PConstants.VEL_ITERATIONS,
            PConstants.POS_ITERATIONS,
        ](
            s,
            shapes,
            contacts,
            0,
            target_x,
            target_y,
            Scalar[dtype](PConstants.K_P),
            Scalar[dtype](PConstants.K_V),
            Scalar[dtype](PConstants.DT),
            Scalar[dtype](PConstants.WORLD_MIN),
            Scalar[dtype](PConstants.WORLD_MAX),
            Scalar[dtype](PConstants.FRICTION),
            Scalar[dtype](PConstants.RESTITUTION),
            Scalar[dtype](0.2),
            Scalar[dtype](0.005),
            Scalar[dtype](PConstants.BLOCK_DAMPING),
        )

    # =========================================================================
    # Env trait methods
    # =========================================================================

    def reset(mut self) -> PushTState[Self.dtype]:
        self._reset_internal()
        return self.get_state()

    def set_state(
        mut self,
        agent_x: Scalar[dtype],
        agent_y: Scalar[dtype],
        block_x: Scalar[dtype],
        block_y: Scalar[dtype],
        block_angle: Scalar[dtype],
        *,
        agent_vx: Scalar[dtype] = 0.0,
        agent_vy: Scalar[dtype] = 0.0,
        settle: Bool = False,
    ) -> PushTState[Self.dtype]:
        """Teleport to an exact (agent, block) configuration — the LeWM
        paper-protocol eval starts episodes from DATASET states (swm
        `_set_state` callable). Same body as `_reset_internal` minus the RNG
        draw: seed bodies, clear contact count / step / done / reward /
        coverage metadata, refresh obs.

        Defaults keep the old behaviour (zero velocities, no physics step).
        swm's `PushT._set_state` (stable-worldmodel 0.0.6) differs on two
        points, opt-in here: it keeps the AGENT velocity from the 7-d state
        (`agent_vx/vy`; the block's velocity is not in the state and stays
        at rest), and it advances the physics ONE free substep (`settle`):
        `space.step(dt)` with no PD force. The free substep is our PD substep
        aimed at `pos + (k_v / k_p) * vel`, where the PD acceleration
        `k_p (target - pos) - k_v vel` is exactly zero."""
        self._seed_bodies(agent_x, agent_y, block_x, block_y, block_angle)
        var s = self._state_view()
        var ao = PushTLayout.BODY_AGENT_OFFSET
        s[0, ao + IDX_VX] = agent_vx
        s[0, ao + IDX_VY] = agent_vy
        if settle:
            var r = Scalar[dtype](PConstants.K_V / PConstants.K_P)
            self._substep(agent_x + r * agent_vx, agent_y + r * agent_vy)
            s = self._state_view()
        s[0, PushTLayout.CONTACT_COUNT_OFFSET] = Scalar[dtype](0.0)
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_STEP] = Scalar[
            dtype
        ](0.0)
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_DONE] = Scalar[
            dtype
        ](0.0)
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_TOTAL_REWARD] = (
            Scalar[dtype](0.0)
        )
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_COVERAGE] = Scalar[
            dtype
        ](0.0)
        self.done = False
        self._write_obs_to_state()
        return self.get_state()

    def step(
        mut self,
        action: PushTAction[Self.dtype],
        verbose: Bool = False,
    ) -> Tuple[PushTState[Self.dtype], Scalar[Self.dtype], Bool]:
        var a = action.clamp()
        var tx = rebind[Scalar[dtype]](a.target_x)
        var ty = rebind[Scalar[dtype]](a.target_y)
        self.last_target_x = tx
        self.last_target_y = ty
        for _ in range(PConstants.N_SUBSTEPS):
            self._substep(tx, ty)

        # Coverage reward
        var s = self._state_view()
        var to_ = PushTLayout.BODY_T_OFFSET
        var bx = rebind[Scalar[dtype]](s[0, to_ + IDX_X])
        var by = rebind[Scalar[dtype]](s[0, to_ + IDX_Y])
        var ba = rebind[Scalar[dtype]](s[0, to_ + IDX_ANGLE])
        # the physics integrates the T about its cog: back to the origin pose
        var _org = t_origin_from_body(bx, by, ba)
        bx = _org[0]
        by = _org[1]
        var cov = compute_coverage(
            bx,
            by,
            ba,
            Scalar[dtype](PConstants.GOAL_X),
            Scalar[dtype](PConstants.GOAL_Y),
            Scalar[dtype](PConstants.GOAL_ANGLE),
        )
        var thr = Scalar[dtype](PConstants.SUCCESS_THRESHOLD)
        var reward = cov / thr
        if reward < Scalar[dtype](0.0):
            reward = Scalar[dtype](0.0)
        elif reward > Scalar[dtype](1.0):
            reward = Scalar[dtype](1.0)

        var terminated = cov > thr
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_COVERAGE] = cov

        # Bump step counter
        var step_v = rebind[Scalar[dtype]](
            s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_STEP]
        )
        step_v = step_v + Scalar[dtype](1.0)
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_STEP] = step_v

        var truncated = Int(step_v) >= PConstants.MAX_STEPS
        self.done = terminated or truncated
        if self.done:
            s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_DONE] = Scalar[
                dtype
            ](1.0)

        self._write_obs_to_state()
        return (
            self.get_state(),
            rebind[Scalar[Self.dtype]](reward),
            self.done,
        )

    def get_state(mut self) -> PushTState[Self.dtype]:
        var out = PushTState[Self.dtype]()
        for i in range(PConstants.KEYPOINTS_DIM):
            out.keypoints[i] = rebind[Scalar[Self.dtype]](
                self.state_data[PushTLayout.OBS_OFFSET + i]
            )
        out.agent_pos[0] = rebind[Scalar[Self.dtype]](
            self.state_data[
                PushTLayout.OBS_OFFSET + PConstants.KEYPOINTS_DIM
            ]
        )
        out.agent_pos[1] = rebind[Scalar[Self.dtype]](
            self.state_data[
                PushTLayout.OBS_OFFSET + PConstants.KEYPOINTS_DIM + 1
            ]
        )
        return out^

    def is_done(self) -> Bool:
        return self.done

    def close(mut self):
        try:
            self.close_renderer()
        except:
            pass

    # =========================================================================
    # ContinuousStateEnv trait methods
    # =========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var out = List[Scalar[Self.dtype]](capacity=PConstants.OBS_DIM)
        for i in range(PConstants.OBS_DIM):
            out.append(
                rebind[Scalar[Self.dtype]](
                    self.state_data[PushTLayout.OBS_OFFSET + i]
                )
            )
        return out^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        _ = self.reset()
        return self.get_obs_list()

    def obs_dim(self) -> Int:
        return PConstants.OBS_DIM

    # =========================================================================
    # ContinuousActionEnv trait methods
    # =========================================================================

    def action_dim(self) -> Int:
        return PConstants.ACTION_DIM

    def action_low(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](PConstants.ACTION_LOW)

    def action_high(self) -> Scalar[Self.dtype]:
        return Scalar[Self.dtype](PConstants.ACTION_HIGH)

    def step_continuous[
        DTYPE_SC: DType
    ](
        mut self, action: Scalar[DTYPE_SC]
    ) -> Tuple[List[Scalar[DTYPE_SC]], Scalar[DTYPE_SC], Bool]:
        # PushT actions are 2D; this 1D variant is a degenerate convenience
        # (sets both target_x and target_y to the same scalar).
        var a = PushTAction[Self.dtype](
            target_x=Scalar[Self.dtype](action),
            target_y=Scalar[Self.dtype](action),
        )
        var result = self.step(a)
        var obs_self = self.get_obs_list()
        var obs = List[Scalar[DTYPE_SC]](capacity=len(obs_self))
        for i in range(len(obs_self)):
            obs.append(Scalar[DTYPE_SC](obs_self[i]))
        return (obs^, Scalar[DTYPE_SC](result[1]), result[2])

    def step_continuous_vec[
        DTYPE_VEC: DType
    ](
        mut self,
        action: List[Scalar[DTYPE_VEC]],
        verbose: Bool = False,
    ) -> Tuple[List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool]:
        var tx = Scalar[Self.dtype](256.0)
        var ty = Scalar[Self.dtype](256.0)
        if len(action) > 0:
            tx = Scalar[Self.dtype](action[0])
        if len(action) > 1:
            ty = Scalar[Self.dtype](action[1])
        var a = PushTAction[Self.dtype](target_x=tx, target_y=ty)
        var result = self.step(a)
        var obs_self = self.get_obs_list()
        var obs = List[Scalar[DTYPE_VEC]](capacity=len(obs_self))
        for i in range(len(obs_self)):
            obs.append(Scalar[DTYPE_VEC](obs_self[i]))
        return (obs^, Scalar[DTYPE_VEC](result[1]), result[2])

    # =========================================================================
    # Convenience
    # =========================================================================

    def coverage(mut self) -> Scalar[Self.dtype]:
        var s = self._state_view()
        return rebind[Scalar[Self.dtype]](
            s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_COVERAGE]
        )

    def block_pose(mut self) -> Tuple[
        Scalar[Self.dtype], Scalar[Self.dtype], Scalar[Self.dtype]
    ]:
        """The block's ORIGIN pose (x, y, angle) — the dataset's convention
        (the physics integrates its cog; see PConstants.T_COG_Y)."""
        var s = self._state_view()
        var to_ = PushTLayout.BODY_T_OFFSET
        var ba = rebind[Scalar[dtype]](s[0, to_ + IDX_ANGLE])
        var o = t_origin_from_body(
            rebind[Scalar[dtype]](s[0, to_ + IDX_X]),
            rebind[Scalar[dtype]](s[0, to_ + IDX_Y]),
            ba,
        )
        return (
            rebind[Scalar[Self.dtype]](o[0]),
            rebind[Scalar[Self.dtype]](o[1]),
            rebind[Scalar[Self.dtype]](ba),
        )

    def agent_pos(mut self) -> Tuple[Scalar[Self.dtype], Scalar[Self.dtype]]:
        var s = self._state_view()
        var ao = PushTLayout.BODY_AGENT_OFFSET
        return (
            rebind[Scalar[Self.dtype]](s[0, ao + IDX_X]),
            rebind[Scalar[Self.dtype]](s[0, ao + IDX_Y]),
        )

    # =========================================================================
    # Rendering (RenderableEnv)
    # =========================================================================

    # Window layout:
    #   Game area is 512×512 starting at (0, HEADER_H). Header bar above is
    #   used for status text. World coords map 1:1 to the game area pixels.
    comptime WINDOW_W: Int = 512
    comptime HEADER_H: Int = 50
    comptime WINDOW_H: Int = 512 + Self.HEADER_H

    @always_inline
    def _world_to_screen(self, wx: Float64, wy: Float64) -> Tuple[Int, Int]:
        return (Int(wx), Int(wy) + Self.HEADER_H)

    @always_inline
    def _t_poly_world(
        self, cx: Float64, cy: Float64, angle: Float64, rect_idx: Int
    ) -> List[SDL_Point]:
        """Transform a T-shape sub-rect (long bar or stem) to world-frame
        SDL_Points in screen coords. Math done in Float64 to keep the renderer
        decoupled from the env's generic DTYPE."""
        var ca = cos(angle)
        var sa = sin(angle)
        var out = List[SDL_Point](capacity=4)
        for v in range(4):
            var local = (
                t_rect_long_vertex(v) if rect_idx == 0 else t_rect_stem_vertex(
                    v
                )
            )
            var lx = Float64(local[0])
            var ly = Float64(local[1])
            var wx = cx + lx * ca - ly * sa
            var wy = cy + lx * sa + ly * ca
            var sp = self._world_to_screen(wx, wy)
            out.append(SDL_Point(Int32(sp[0]), Int32(sp[1])))
        return out^

    def render(mut self, mut renderer: Renderer2D) raises:
        """Draw the current state into `renderer`.

        Z-order (back to front): white bg → goal-T → walls → block-T → agent
        → action target marker → header overlay.
        """
        var white_bg = renderer.make_color(255, 255, 255)
        if not renderer.begin_frame_with_color(white_bg):
            return

        var light_green = renderer.make_color(144, 238, 144)
        var slate = renderer.make_color(119, 136, 153)
        var royal_blue = renderer.make_color(65, 105, 225)
        var lg = light_gray()
        var header_bg = renderer.make_color(240, 240, 240)
        var text_color = renderer.make_color(40, 40, 40)
        var marker_color = renderer.make_color(220, 30, 30)

        # Header strip
        renderer.draw_rect(
            0, 0, Self.WINDOW_W, Self.HEADER_H, header_bg
        )

        # Goal-T (drawn first so block-T overlays it where they overlap)
        var goal_cx = Float64(PConstants.GOAL_X)
        var goal_cy = Float64(PConstants.GOAL_Y)
        var goal_a = Float64(PConstants.GOAL_ANGLE)
        var goal_long = self._t_poly_world(goal_cx, goal_cy, goal_a, 0)
        var goal_stem = self._t_poly_world(goal_cx, goal_cy, goal_a, 1)
        renderer.draw_polygon(goal_long, light_green, True)
        renderer.draw_polygon(goal_stem, light_green, True)

        # Walls: 4 thin rectangles around the playing box
        var wm = Int(PConstants.WORLD_MIN)
        var wM = Int(PConstants.WORLD_MAX)
        var wt = Int(PConstants.WALL_RADIUS) * 2 + 2  # rendered thickness
        # Left, right, top, bottom (offset Y by HEADER_H)
        renderer.draw_rect(
            wm - wt, wm - wt + Self.HEADER_H, wt, wM - wm + 2 * wt, lg
        )
        renderer.draw_rect(
            wM, wm - wt + Self.HEADER_H, wt, wM - wm + 2 * wt, lg
        )
        renderer.draw_rect(
            wm - wt, wm - wt + Self.HEADER_H, wM - wm + 2 * wt, wt, lg
        )
        renderer.draw_rect(
            wm - wt, wM + Self.HEADER_H, wM - wm + 2 * wt, wt, lg
        )

        # Block-T
        var bp = self.block_pose()
        var blk_long = self._t_poly_world(
            Float64(bp[0]), Float64(bp[1]), Float64(bp[2]), 0
        )
        var blk_stem = self._t_poly_world(
            Float64(bp[0]), Float64(bp[1]), Float64(bp[2]), 1
        )
        renderer.draw_polygon(blk_long, slate, True)
        renderer.draw_polygon(blk_stem, slate, True)

        # Agent circle
        var ap = self.agent_pos()
        var asp = self._world_to_screen(Float64(ap[0]), Float64(ap[1]))
        renderer.draw_circle(
            asp[0],
            asp[1],
            Int(PConstants.AGENT_RADIUS),
            royal_blue,
            True,
        )

        # Action target marker (small red cross)
        var tsp = self._world_to_screen(
            Float64(self.last_target_x), Float64(self.last_target_y)
        )
        var mh = 6
        renderer.draw_line(
            tsp[0] - mh, tsp[1], tsp[0] + mh, tsp[1], marker_color, 2
        )
        renderer.draw_line(
            tsp[0], tsp[1] - mh, tsp[0], tsp[1] + mh, marker_color, 2
        )

        # Header text overlay
        var step_v = rebind[Scalar[Self.dtype]](
            self.state_data[
                PushTLayout.METADATA_OFFSET + PushTLayout.META_STEP
            ]
        )
        var cov_v = self.coverage()
        renderer.draw_text(
            "PushT  step="
            + String(Int(Float64(step_v)))
            + "  coverage="
            + fit(String(Float64(cov_v)), 5)
            + "  target=("
            + String(Int(Float64(self.last_target_x)))
            + ","
            + String(Int(Float64(self.last_target_y)))
            + ")",
            10,
            20,
            text_color,
        )

        renderer.flip()

    def init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(
            Renderer2D(
                width=Self.WINDOW_W,
                height=Self.WINDOW_H,
                fps=30,
                title=String("PushT"),
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

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

    @always_inline
    def screen_to_world(
        self, sx: Int, sy: Int
    ) -> Tuple[Scalar[Self.dtype], Scalar[Self.dtype]]:
        """Inverse of `_world_to_screen` — used by the playable demo to map
        mouse-cursor coords back to action targets."""
        return (
            Scalar[Self.dtype](sx),
            Scalar[Self.dtype](sy - Self.HEADER_H),
        )

from noeira.core.fmt import fit