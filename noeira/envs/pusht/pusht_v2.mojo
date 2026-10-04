"""PushT V2 GPU-batched environment.

Native Mojo implementation of PushT with GPU-accelerated batched simulation.

This is the batched counterpart of PushTEnv (CPU). Mirrors PendulumV2's
structure: instance fields for CPU single-env mode (delegating to the
shared substep helper in `.physics`) and static GPU kernel methods for
batched training under the `GPUContinuousEnv` trait.

State layout (193 floats per env, see `constants.PushTLayout`):
    [0  .. 18)   observation (16 keypoints + 2 agent_pos)
    [18 .. 44)   2 bodies × 13 = body state (agent + T-block)
    [44 .. 188)  contacts workspace (16 × 9)
    [188]        contact count
    [189 .. 193) metadata (step, done, total_reward, coverage)

Shapes buffer (NUM_SHAPES × SHAPE_MAX_SIZE = 5×20 = 100 floats) lives in
`STEP_WS_SHARED`. The host calls `init_step_workspace_gpu` once at setup
to populate it from CPU shape definitions.
"""

from std.math import cos, sin, sqrt, pi
from layout import Layout, LayoutTensor
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from std.memory import Pointer

from noeira.core import (
    BoxContinuousActionEnv,
    GPUContinuousEnv,
)
from noeira.physics2d import dtype, TPB
from noeira.physics2d.constants import (
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

from .state import PushTState
from .action import PushTAction
from .constants import PConstants, PushTLayout, PushTShapeBuf
from .geometry import (
    init_pusht_shape_buffer,
    get_t_keypoints_world,
    compute_coverage,
    t_rect_long_vertex,
    t_body_from_origin,
    t_origin_from_body,
)
from .physics import pusht_substep_single_env


# =============================================================================
# Helpers for shape buffer + initial state.
# =============================================================================


@always_inline
def _compute_t_inertia_g() -> Scalar[dtype]:
    """Same inertia formula as the CPU env (pymunk-compatible)."""
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
    return Scalar[dtype](2.0) * I_one


@always_inline
def _seed_env_state_gpu[
    BATCH_SIZE: Int, STATE_SIZE: Int
](
    state: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
    ],
    env: Int,
    agent_x: Scalar[dtype],
    agent_y: Scalar[dtype],
    block_x: Scalar[dtype],
    block_y: Scalar[dtype],
    block_angle: Scalar[dtype],
):
    """Initialize one env's state slot: zero everything then fill agent + T."""
    for i in range(STATE_SIZE):
        state[env, i] = Scalar[dtype](0.0)

    # Agent (body 0): kinematic, inv_mass = inv_inertia = 0
    var ao = PushTLayout.BODY_AGENT_OFFSET
    state[env, ao + IDX_X] = agent_x
    state[env, ao + IDX_Y] = agent_y
    state[env, ao + IDX_SHAPE] = Scalar[dtype](PushTShapeBuf.SHAPE_AGENT)

    # T-block (body 1)
    var to_ = PushTLayout.BODY_T_OFFSET
    var bxy = t_body_from_origin(block_x, block_y, block_angle)
    state[env, to_ + IDX_X] = bxy[0]
    state[env, to_ + IDX_Y] = bxy[1]
    state[env, to_ + IDX_ANGLE] = block_angle
    var m = Scalar[dtype](PConstants.T_MASS)
    var I = _compute_t_inertia_g()
    state[env, to_ + IDX_MASS] = m
    state[env, to_ + IDX_INV_MASS] = Scalar[dtype](1.0) / m
    state[env, to_ + IDX_INV_INERTIA] = Scalar[dtype](1.0) / I
    state[env, to_ + IDX_SHAPE] = Scalar[dtype](PushTShapeBuf.SHAPE_T_COMPOUND)


@always_inline
def _write_obs_to_state_only[
    BATCH_SIZE: Int, STATE_SIZE: Int
](
    state: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
    ],
    env: Int,
):
    """Compute keypoints + agent_pos and store in state's OBS slot only.
    Used during reset; extract_obs_kernel_gpu will then copy state[:, :OBS_DIM]
    into the agent-facing obs buffer.
    """
    var to_ = PushTLayout.BODY_T_OFFSET
    var ao = PushTLayout.BODY_AGENT_OFFSET
    var bx = rebind[Scalar[dtype]](state[env, to_ + IDX_X])
    var by = rebind[Scalar[dtype]](state[env, to_ + IDX_Y])
    var ba = rebind[Scalar[dtype]](state[env, to_ + IDX_ANGLE])
    # the physics integrates the T about its cog: back to the origin pose
    var _org = t_origin_from_body(bx, by, ba)
    bx = _org[0]
    by = _org[1]
    var ax = rebind[Scalar[dtype]](state[env, ao + IDX_X])
    var ay = rebind[Scalar[dtype]](state[env, ao + IDX_Y])
    var kp = Array[Scalar[dtype], PConstants.KEYPOINTS_DIM](
        fill=Scalar[dtype](0.0)
    )
    get_t_keypoints_world(bx, by, ba, kp)
    for i in range(PConstants.KEYPOINTS_DIM):
        state[env, PushTLayout.OBS_OFFSET + i] = kp[i]
    state[env, PushTLayout.OBS_OFFSET + PConstants.KEYPOINTS_DIM] = ax
    state[env, PushTLayout.OBS_OFFSET + PConstants.KEYPOINTS_DIM + 1] = ay


@always_inline
def _write_obs_single_env[
    BATCH_SIZE: Int, STATE_SIZE: Int, OBS_DIM: Int
](
    state: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
    ],
    obs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    env: Int,
):
    """Compute keypoints + agent_pos and write to both the state's obs slot
    and the agent-facing obs buffer.
    """
    var to_ = PushTLayout.BODY_T_OFFSET
    var ao = PushTLayout.BODY_AGENT_OFFSET
    var bx = rebind[Scalar[dtype]](state[env, to_ + IDX_X])
    var by = rebind[Scalar[dtype]](state[env, to_ + IDX_Y])
    var ba = rebind[Scalar[dtype]](state[env, to_ + IDX_ANGLE])
    # the physics integrates the T about its cog: back to the origin pose
    var _org = t_origin_from_body(bx, by, ba)
    bx = _org[0]
    by = _org[1]
    var ax = rebind[Scalar[dtype]](state[env, ao + IDX_X])
    var ay = rebind[Scalar[dtype]](state[env, ao + IDX_Y])
    var kp = Array[Scalar[dtype], PConstants.KEYPOINTS_DIM](
        fill=Scalar[dtype](0.0)
    )
    get_t_keypoints_world(bx, by, ba, kp)
    for i in range(PConstants.KEYPOINTS_DIM):
        state[env, PushTLayout.OBS_OFFSET + i] = kp[i]
        obs[env, i] = kp[i]
    state[env, PushTLayout.OBS_OFFSET + PConstants.KEYPOINTS_DIM] = ax
    state[env, PushTLayout.OBS_OFFSET + PConstants.KEYPOINTS_DIM + 1] = ay
    obs[env, PConstants.KEYPOINTS_DIM] = ax
    obs[env, PConstants.KEYPOINTS_DIM + 1] = ay


# =============================================================================
# PushTV2 — main struct
# =============================================================================


struct PushTV2[DTYPE: DType](
    BoxContinuousActionEnv, GPUContinuousEnv, Copyable, Movable
):
    """PushT environment with GPU-accelerated batched simulation.

    Continuous 2D action (target position in [0, 512]²), 18D obs (keypoints +
    agent_pos). Reward = clip(coverage / 0.95, 0, 1). Episode terminates when
    coverage > 0.95 or truncates at MAX_STEPS.
    """

    # =========================================================================
    # Type aliases and constants
    # =========================================================================

    comptime dtype = Self.DTYPE
    comptime StateType = PushTState[Self.DTYPE]
    comptime ActionType = PushTAction[Self.DTYPE]

    comptime STATE_SIZE: Int = PushTLayout.STATE_SIZE
    comptime OBS_DIM: Int = PConstants.OBS_DIM
    comptime ACTION_DIM: Int = PConstants.ACTION_DIM
    # Shared workspace: the 5×20 shapes buffer.
    comptime STEP_WS_SHARED: Int = (
        PushTShapeBuf.NUM_SHAPES * SHAPE_MAX_SIZE
    )
    comptime STEP_WS_PER_ENV: Int = 0
    comptime NAME: String = "PushTV2"

    # =========================================================================
    # CPU instance fields (for single-env mode)
    # =========================================================================

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

    # =========================================================================
    # Constructors
    # =========================================================================

    def __init__(out self, seed: UInt64 = 0):
        self.state_data = Array[
            Scalar[dtype], PushTLayout.STATE_SIZE
        ](fill=Scalar[dtype](0.0))
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
        self._cpu_reset()

    def __init__(out self, *, copy: Self):
        self.state_data = copy.state_data.copy()
        self.shapes_data = copy.shapes_data.copy()
        self.contacts_data = copy.contacts_data.copy()
        self.done = copy.done
        self.rng_seed = copy.rng_seed
        self.rng_counter = copy.rng_counter

    def __init__(out self, *, deinit move: Self):
        self.state_data = move.state_data.copy()
        self.shapes_data = move.shapes_data.copy()
        self.contacts_data = move.contacts_data.copy()
        self.done = move.done
        self.rng_seed = move.rng_seed
        self.rng_counter = move.rng_counter

    # =========================================================================
    # CPU-side helpers (single env, delegates to shared substep)
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

    def _cpu_reset(mut self):
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
        var s = self._state_view()
        _seed_env_state_gpu[1, PushTLayout.STATE_SIZE](s, 0, ax, ay, bx, by, ba)
        var obs_view = LayoutTensor[
            dtype, Layout.row_major(1, PConstants.OBS_DIM)
        ](Span(self.state_data))  # OBS at offset 0
        _write_obs_single_env[1, PushTLayout.STATE_SIZE, PConstants.OBS_DIM](
            s, obs_view, 0
        )
        self.done = False

    # =========================================================================
    # Env trait methods (CPU mode)
    # =========================================================================

    def reset(mut self) -> PushTState[Self.dtype]:
        self._cpu_reset()
        return self.get_state()

    def step(
        mut self,
        action: PushTAction[Self.dtype],
        verbose: Bool = False,
    ) -> Tuple[PushTState[Self.dtype], Scalar[Self.dtype], Bool]:
        var a = action.clamp()
        var tx = rebind[Scalar[dtype]](a.target_x)
        var ty = rebind[Scalar[dtype]](a.target_y)
        var s = self._state_view()
        var shapes = self._shapes_view()
        var contacts = self._contacts_view()
        for _ in range(PConstants.N_SUBSTEPS):
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
                tx,
                ty,
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

        # Reward
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
        var step_v = rebind[Scalar[dtype]](
            s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_STEP]
        ) + Scalar[dtype](1.0)
        s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_STEP] = step_v
        var truncated = Int(step_v) >= PConstants.MAX_STEPS
        self.done = terminated or truncated
        if self.done:
            s[0, PushTLayout.METADATA_OFFSET + PushTLayout.META_DONE] = Scalar[
                dtype
            ](1.0)

        # Refresh obs slot (so get_state sees up-to-date keypoints)
        var obs_view = LayoutTensor[
            dtype, Layout.row_major(1, PConstants.OBS_DIM)
        ](Span(self.state_data))
        _write_obs_single_env[1, PushTLayout.STATE_SIZE, PConstants.OBS_DIM](
            s, obs_view, 0
        )
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

    def close(mut self):
        pass

    def is_done(self) -> Bool:
        return self.done

    # =========================================================================
    # ContinuousStateEnv / ContinuousActionEnv trait methods (CPU view)
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
    # GPU batch operations (GPUContinuousEnv trait)
    # =========================================================================

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        var states_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)
        ](states)
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def reset_wrapper(
            st: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            seed: Scalar[DType.uint64],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            PushTV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                st, env, UInt64(seed) ^ UInt64(env + 1)
            )

        ctx.enqueue_function[reset_wrapper](
            states_tensor,
            Scalar[DType.uint64](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        rng_seed: UInt64,
        workspace_ptr: Optional[
            Pointer[Scalar[dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        var states_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)
        ](states)
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](dones)
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def selreset_wrapper(
            st: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            dn: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            seed: Scalar[DType.uint64],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            if rebind[Scalar[dtype]](dn[env]) > Scalar[dtype](0.5):
                PushTV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                    st, env, UInt64(seed) ^ UInt64(env + 1)
                )
                dn[env] = Scalar[dtype](0.0)

        ctx.enqueue_function[selreset_wrapper](
            states_tensor,
            dones_tensor,
            Scalar[DType.uint64](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int, OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
    ) raises:
        var st_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)
        ](states)
        var ob_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM)
        ](obs)
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def extract(
            st: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), ImmutAnyOrigin
            ],
            ob: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            for d in range(OBS_DIM):
                ob[env, d] = st[env, d]

        ctx.enqueue_function[extract](
            st_t,
            ob_t,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[dtype]) raises:
        """Fill the shared workspace with the shape buffer."""
        # Build shapes on host then upload.
        var row = Array[Scalar[dtype], SHAPE_MAX_SIZE](
            fill=Scalar[dtype](0.0)
        )
        var shapes_host = Array[
            Array[Scalar[dtype], SHAPE_MAX_SIZE],
            PushTShapeBuf.NUM_SHAPES,
        ](fill=row)
        init_pusht_shape_buffer[PushTShapeBuf.NUM_SHAPES](shapes_host)
        # Flatten into a List then copy into the device buffer.
        var flat = List[Scalar[dtype]](
            capacity=PushTShapeBuf.NUM_SHAPES * SHAPE_MAX_SIZE
        )
        for s in range(PushTShapeBuf.NUM_SHAPES):
            for j in range(SHAPE_MAX_SIZE):
                flat.append(shapes_host[s][j])
        ctx.enqueue_copy(workspace_buf, flat.unsafe_ptr())

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[dtype],
        curriculum_values: List[Scalar[dtype]],
    ) raises:
        pass

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int, OBS_DIM: Int, ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        mut terminated: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
        curriculum_values: List[Scalar[dtype]] = [],
        workspace_ptr: Optional[
            Pointer[Scalar[dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        # Must have a shape workspace
        if not Bool(workspace_ptr):
            raise Error(
                "PushTV2.step_kernel_gpu requires workspace_ptr (shape buffer)"
            )

        var states_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)
        ](states)
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM)
        ](actions)
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](rewards)
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](dones)
        var terminated_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](terminated)
        var obs_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM)
        ](obs)
        var shapes_tensor = LayoutTensor[
            dtype,
            Layout.row_major(PushTShapeBuf.NUM_SHAPES, SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ](workspace_ptr.value())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def step_wrapper(
            st: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            ac: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, ACTION_DIM),
                ImmutAnyOrigin,
            ],
            rw: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dn: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            tm: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            ob: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
            sh: LayoutTensor[
                dtype,
                Layout.row_major(
                    PushTShapeBuf.NUM_SHAPES, SHAPE_MAX_SIZE
                ),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            PushTV2[Self.dtype]._step_env_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](st, ac, rw, dn, tm, ob, sh, env)

        ctx.enqueue_function[step_wrapper](
            states_tensor,
            actions_tensor,
            rewards_tensor,
            dones_tensor,
            terminated_tensor,
            obs_tensor,
            shapes_tensor,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Inline GPU helpers
    # =========================================================================

    @always_inline
    @staticmethod
    def _reset_env_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: UInt64,
    ):
        var rng = PhiloxRandom(seed=seed, offset=0)
        var r = rng.step_uniform()
        var ar_lo = Scalar[dtype](PConstants.AGENT_RESET_LOW)
        var ar_hi = Scalar[dtype](PConstants.AGENT_RESET_HIGH)
        var br_lo = Scalar[dtype](PConstants.BLOCK_RESET_LOW)
        var br_hi = Scalar[dtype](PConstants.BLOCK_RESET_HIGH)
        var ax = ar_lo + rebind[Scalar[dtype]](r[0]) * (ar_hi - ar_lo)
        var ay = ar_lo + rebind[Scalar[dtype]](r[1]) * (ar_hi - ar_lo)
        var bx = br_lo + rebind[Scalar[dtype]](r[2]) * (br_hi - br_lo)
        var by = br_lo + rebind[Scalar[dtype]](r[3]) * (br_hi - br_lo)
        var pi_s = Scalar[dtype](pi)
        var ba = (
            rebind[Scalar[dtype]](r[0]) * Scalar[dtype](2.0)
            - Scalar[dtype](1.0)
        ) * pi_s
        _seed_env_state_gpu[BATCH_SIZE, STATE_SIZE](
            state, env, ax, ay, bx, by, ba
        )
        _write_obs_to_state_only[BATCH_SIZE, STATE_SIZE](state, env)

    @always_inline
    @staticmethod
    def _step_env_gpu[
        BATCH_SIZE: Int, STATE_SIZE: Int, OBS_DIM: Int, ACTION_DIM: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), ImmutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        terminated: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        shapes: LayoutTensor[
            dtype,
            Layout.row_major(PushTShapeBuf.NUM_SHAPES, SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ],
        env: Int,
    ):
        # Action: clamp to [0, 512]
        var raw_tx = rebind[Scalar[dtype]](actions[env, 0])
        var raw_ty = rebind[Scalar[dtype]](actions[env, 1])
        var lo = Scalar[dtype](PConstants.ACTION_LOW)
        var hi = Scalar[dtype](PConstants.ACTION_HIGH)
        if raw_tx < lo:
            raw_tx = lo
        elif raw_tx > hi:
            raw_tx = hi
        if raw_ty < lo:
            raw_ty = lo
        elif raw_ty > hi:
            raw_ty = hi

        # Contacts: read-write workspace embedded in state (per-env slot).
        # We rebuild a tensor view over the contacts slice within the state
        # buffer. The contacts slice occupies CONTACT_DATA_SIZE * MAX_CONTACTS
        # floats starting at PushTLayout.CONTACTS_WS_OFFSET in each env row.
        # For solver compatibility we need a [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE]
        # view, but we only ever touch this env's slot. We construct a view
        # over the whole state buffer reinterpreted that way by using a
        # one-batch helper view aliased at the contacts offset.
        #
        # Trick: since contacts use [env, c, field] indexing in the solver
        # and BATCH==BATCH_SIZE here, the stride between envs is STATE_SIZE
        # floats, while the contacts tensor expects stride
        # MAX_CONTACTS*CONTACT_DATA_SIZE between envs. We CANNOT alias.
        # Instead, allocate per-thread local contacts on the stack.
        var contacts_local = Array[
            Scalar[dtype], PushTLayout.MAX_CONTACTS * CONTACT_DATA_SIZE
        ](fill=Scalar[dtype](0.0))
        var contacts_view = LayoutTensor[
            dtype,
            Layout.row_major(1, PushTLayout.MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_local.unsafe_ptr().as_unsafe_any_origin())
        # Local 1-element state view aliased at this env's slot is NOT trivial
        # — the substep helper expects BATCH (here BATCH_SIZE) and indexes
        # with `env`, which is fine. So we can just pass the full state tensor
        # and pass `env`. But contacts is [1, ...] local. The substep helper
        # is generic in BATCH; pass BATCH_SIZE for state and a separate 1-env
        # contacts buffer with env=0 for solver calls.
        #
        # That requires the substep to accept two different BATCH params.
        # Simpler: build the substep INLINE here rather than via the helper,
        # using contacts_view[0, c, ...] directly. To keep the code compact,
        # we just call the helper with BATCH=BATCH_SIZE for state and use a
        # local wrapper that maps env->0 for contacts indexing. But the
        # helper is templated on BATCH only once.
        #
        # We instead specialize the substep call: pass state with BATCH_SIZE
        # and contacts with 1-batch, but the helper expects a single BATCH.
        # Workaround: rebuild the per-env physics inline here using a fresh
        # 1-batch state view aliased at this env's row.
        var state_one = LayoutTensor[
            dtype, Layout.row_major(1, STATE_SIZE), MutAnyOrigin
        ](state.ptr.unsafe_offset(env * STATE_SIZE))

        for _ in range(PConstants.N_SUBSTEPS):
            pusht_substep_single_env[
                1,
                PushTShapeBuf.NUM_SHAPES,
                PushTLayout.MAX_CONTACTS,
                STATE_SIZE,
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
                state_one,
                shapes,
                contacts_view,
                0,
                raw_tx,
                raw_ty,
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

        # Reward + termination
        var to_ = PushTLayout.BODY_T_OFFSET
        var bx = rebind[Scalar[dtype]](state[env, to_ + IDX_X])
        var by = rebind[Scalar[dtype]](state[env, to_ + IDX_Y])
        var ba = rebind[Scalar[dtype]](state[env, to_ + IDX_ANGLE])
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
        var term = Scalar[dtype](1.0) if cov > thr else Scalar[dtype](0.0)
        state[env, PushTLayout.METADATA_OFFSET + PushTLayout.META_COVERAGE] = cov
        var step_v = rebind[Scalar[dtype]](
            state[env, PushTLayout.METADATA_OFFSET + PushTLayout.META_STEP]
        ) + Scalar[dtype](1.0)
        state[env, PushTLayout.METADATA_OFFSET + PushTLayout.META_STEP] = step_v
        var trunc_v = (
            Scalar[dtype](1.0) if Int(step_v) >= PConstants.MAX_STEPS else Scalar[
                dtype
            ](0.0)
        )
        var done_v = (
            Scalar[dtype](1.0) if term > Scalar[dtype](0.0)
            or trunc_v > Scalar[dtype](0.0) else Scalar[dtype](0.0)
        )
        state[env, PushTLayout.METADATA_OFFSET + PushTLayout.META_DONE] = done_v
        rewards[env] = reward
        terminated[env] = term
        dones[env] = done_v

        # Write keypoints + agent_pos to obs (and into state's obs slot)
        _write_obs_single_env[BATCH_SIZE, STATE_SIZE, OBS_DIM](
            state, obs, env
        )
