"""Pendulum V2 GPU environment.

Native Mojo implementation of Pendulum with GPU-accelerated batched simulation.

Physics matched to Gymnasium Pendulum-v1:
https://gymnasium.farama.org/environments/classic_control/pendulum/

A frictionless pendulum starts from a random position and the goal is to
swing it up and keep it balanced upright.

State observation: [cos(θ), sin(θ), θ_dot] (3D)
Action: torque in [-2.0, 2.0] (1D continuous) — interpreted as raw torque
        (matches PendulumEnv V1 and Gymnasium Pendulum-v1). Callers whose
        policy emits a different range (e.g. tanh-squashed [-1, +1]) must
        scale on their side; out-of-range values are clamped.
Reward: -(θ² + 0.1*θ_dot² + 0.001*torque²)

Episode never terminates naturally (always runs for max_steps=200).
"""

from std.math import sqrt, cos, sin, pi
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from std.memory import alloc

from noeira.core import (
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    BoxDiscreteActionEnv,
    DiscreteEnv,
    TileCoding,
    PolynomialFeatures,
    RenderableEnv,
)
from noeira.render import (
    Renderer2D,
    Camera,
    SDL_Color,
    Vec2,
    sky_blue,
    black,
    light_gray,
    rgb,
)

from .state import PendulumV2State
from .action import PendulumV2Action
from .constants import PConstants, PendulumLayout

# Import global GPU constants
from noeira.physics2d import dtype, TPB
from noeira.core.fmt import fit
from noeira.core.tiles import Tile, ImmTile, tile


# =============================================================================
# PendulumV2 Environment
# =============================================================================


struct PendulumV2[DTYPE: DType](
    BoxContinuousActionEnv,
    BoxDiscreteActionEnv,
    Copyable,
    DiscreteEnv,
    GPUContinuousEnv,
    Movable,
    RenderableEnv,
):
    """Pendulum environment with GPU-accelerated batched simulation.

    This environment implements both CPU single-env and GPU batched interfaces.

    Features:
    - Simple pendulum physics (angle + angular velocity)
    - GPU-compatible state layout for batch training
    - Continuous action space: torque in [-2, 2]
    - Also supports discrete actions: 0 (left), 1 (none), 2 (right)

    Physics:
    - θ'' = (3g/2L) * sin(θ) + (3/mL²) * u
    - Euler integration with dt = 0.05

    Reward:
    - r = -(θ² + 0.1*θ_dot² + 0.001*u²)
    - Where θ is normalized to [-π, π]
    """

    # =========================================================================
    # Type Aliases and Constants
    # =========================================================================

    comptime dtype = Self.DTYPE
    comptime StateType = PendulumV2State[Self.DTYPE]
    comptime ActionType = PendulumV2Action[Self.DTYPE]

    # GPUContinuousEnv trait requirements
    comptime STATE_SIZE: Int = PConstants.STATE_SIZE  # 6
    comptime OBS_DIM: Int = PConstants.OBS_DIM  # 3
    comptime ACTION_DIM: Int = PConstants.ACTION_DIM  # 1
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0
    comptime NAME: String = "PendulumV2"

    # DiscreteEnv trait requirement
    comptime NUM_ACTIONS: Int = 3  # left, none, right

    # =========================================================================
    # Instance Variables (for CPU single-env mode)
    # =========================================================================

    # Physics constants
    var max_speed: Scalar[Self.dtype]
    var max_torque: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]
    var g: Scalar[Self.dtype]
    var m: Scalar[Self.dtype]
    var l: Scalar[Self.dtype]

    # Physics state
    var theta: Scalar[Self.dtype]
    var theta_dot: Scalar[Self.dtype]

    # Episode tracking
    var steps: Int
    var max_steps: Int
    var done: Bool
    var total_reward: Scalar[Self.dtype]
    var last_torque: Scalar[Self.dtype]

    # Discretization settings (for DiscreteEnv)
    var num_bins_angle: Int
    var num_bins_velocity: Int

    # Deterministic Philox RNG state (replaces std.random.random_float64 in
    # reset_obs_list). Per-instance so multiple envs can be seeded
    # independently; counter advances on each reset so subsequent resets of
    # the same env produce different but reproducible initial states.
    # Mirrors the V2 GPU reset path (which also uses Philox) so CPU-vs-GPU
    # parity is no longer confounded by RNG-source mismatch.
    var rng_seed: UInt64
    var rng_counter: UInt64

    # Renderer (RenderableEnv)
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    # =========================================================================
    # Constructors
    # =========================================================================

    def __init__(
        out self,
        num_bins_angle: Int = 15,
        num_bins_velocity: Int = 15,
        seed: UInt64 = 0,
    ):
        """Initialize Pendulum with default physics parameters.

        Args:
            num_bins_angle: Number of bins for angle discretization.
            num_bins_velocity: Number of bins for velocity discretization.
            seed: Per-instance Philox seed for `reset_obs_list`. Pass a
                unique value per env when running multi-env CPU baselines so
                each env explores a different initial-state distribution.
        """
        # Physics constants from Gymnasium
        self.max_speed = Scalar[Self.dtype](PConstants.MAX_SPEED)
        self.max_torque = Scalar[Self.dtype](PConstants.MAX_TORQUE)
        self.dt = Scalar[Self.dtype](PConstants.DT)
        self.g = Scalar[Self.dtype](PConstants.G)
        self.m = Scalar[Self.dtype](PConstants.M)
        self.l = Scalar[Self.dtype](PConstants.L)

        # State (θ=0 is pointing up, positive is clockwise)
        self.theta = Scalar[Self.dtype](pi)  # Start pointing down
        self.theta_dot = Scalar[Self.dtype](0.0)

        # Episode
        self.steps = 0
        self.max_steps = PConstants.MAX_STEPS
        self.done = False
        self.total_reward = Scalar[Self.dtype](0.0)
        self.last_torque = Scalar[Self.dtype](0.0)

        # Discretization settings
        self.num_bins_angle = num_bins_angle
        self.num_bins_velocity = num_bins_velocity

        # Philox RNG state (replaces std.random.random_float64).
        self.rng_seed = seed
        self.rng_counter = 0

        # Renderer
        self._renderer = None
        self._renderer_initialized = False

    def __init__(out self, *, copy: Self):
        """Copy constructor."""
        self.max_speed = copy.max_speed
        self.max_torque = copy.max_torque
        self.dt = copy.dt
        self.g = copy.g
        self.m = copy.m
        self.l = copy.l
        self.theta = copy.theta
        self.theta_dot = copy.theta_dot
        self.steps = copy.steps
        self.max_steps = copy.max_steps
        self.done = copy.done
        self.total_reward = copy.total_reward
        self.last_torque = copy.last_torque
        self.num_bins_angle = copy.num_bins_angle
        self.num_bins_velocity = copy.num_bins_velocity
        self.rng_seed = copy.rng_seed
        self.rng_counter = copy.rng_counter
        # Do not copy renderer — reset to null
        self._renderer = None
        self._renderer_initialized = False

    def __init__(out self, *, deinit move: Self):
        """Move constructor."""
        self.max_speed = move.max_speed
        self.max_torque = move.max_torque
        self.dt = move.dt
        self.g = move.g
        self.m = move.m
        self.l = move.l
        self.theta = move.theta
        self.theta_dot = move.theta_dot
        self.steps = move.steps
        self.max_steps = move.max_steps
        self.done = move.done
        self.total_reward = move.total_reward
        self.last_torque = move.last_torque
        self.num_bins_angle = move.num_bins_angle
        self.num_bins_velocity = move.num_bins_velocity
        self.rng_seed = move.rng_seed
        self.rng_counter = move.rng_counter
        # Transfer renderer ownership
        self._renderer = move._renderer
        self._renderer_initialized = move._renderer_initialized

    # =========================================================================
    # GPU Batch Operations (Static Methods) - GPUContinuousEnv Trait
    # =========================================================================

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
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
        """Perform one environment step with continuous actions (GPUContinuousEnv trait).

        Pendulum physics:
        - θ'' = (3g/2L) * sin(θ) + (3/mL²) * u
        - Euler integration
        - Reward = -(θ² + 0.1*θ_dot² + 0.001*u²)

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            actions: Continuous actions buffer [BATCH_SIZE * ACTION_DIM].
            rewards: Rewards buffer (output) [BATCH_SIZE].
            dones: Done flags buffer (output) [BATCH_SIZE].
            terminated: Terminated flags buffer (output) [BATCH_SIZE]. Always 0 for Pendulum (only truncates).
            obs: Observations buffer (output) [BATCH_SIZE * OBS_DIM].
            rng_seed: Optional random seed (unused for deterministic physics).
            curriculum_values: Optional curriculum values (unused for Pendulum).
            workspace_ptr: Optional workspace pointer (unused for Pendulum).
            rng_counter_ptr: Optional GPU counter pointer for deterministic RNG sequencing.
        """
        # Create tensor views (concrete-origin, direct-from-buffer; widen into
        # the MutAnyOrigin/ImmutAnyOrigin kernel params below).
        var states_tensor = tile[BATCH_SIZE, STATE_SIZE](states)

        var actions_tensor = tile[BATCH_SIZE, ACTION_DIM](actions)

        var rewards_tensor = tile[BATCH_SIZE](rewards)

        var dones_tensor = tile[BATCH_SIZE](dones)

        var terminated_tensor = tile[BATCH_SIZE](terminated)

        var obs_tensor = tile[BATCH_SIZE, OBS_DIM](obs)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def step_wrapper(
            states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
            actions: ImmTile[dtype, BATCH_SIZE, ACTION_DIM],
            rewards: Tile[dtype, BATCH_SIZE],
            dones: Tile[dtype, BATCH_SIZE],
            terminated_out: Tile[dtype, BATCH_SIZE],
            obs: Tile[dtype, BATCH_SIZE, OBS_DIM],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            PendulumV2[Self.dtype]._step_env_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](states, actions, rewards, dones, obs, env)
            # Pendulum never terminates, only truncates at max steps
            terminated_out[env] = Scalar[dtype](0.0)

        ctx.enqueue_function[step_wrapper](
            states_tensor,
            actions_tensor,
            rewards_tensor,
            dones_tensor,
            terminated_tensor,
            obs_tensor,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments to random initial values (GPUContinuousEnv trait).

        Initial angle is uniformly random in [-π, π].
        Initial angular velocity is uniformly random in [-1, 1].

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            rng_seed: Random seed for initial state generation. Use different
                     values across calls for varied initial states.
        """
        var states_tensor = tile[BATCH_SIZE, STATE_SIZE](states)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def reset_wrapper(
            states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
            seed: Scalar[dtype],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            # Combine seed with env index using prime multiplier for good distribution
            var combined_seed = Int(seed) * 2654435761 + env * 12345
            PendulumV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                states, env, combined_seed
            )

        ctx.enqueue_function[reset_wrapper](
            states_tensor,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
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
        """Reset only done environments (GPUContinuousEnv trait).

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            dones: Done flags buffer [BATCH_SIZE].
            rng_seed: Random seed for initialization. Should be different each call
                     (e.g., training step counter) for varied initial states.
            workspace_ptr: Optional workspace pointer (unused for Pendulum).
            rng_counter_ptr: Optional GPU counter pointer. When non-null, reads
                     seed from GPU memory instead of rng_seed parameter.
        """
        var states_tensor = tile[BATCH_SIZE, STATE_SIZE](states)

        var dones_tensor = tile[BATCH_SIZE](dones)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        if Bool(rng_counter_ptr):
            var counter_t = tile[1](rng_counter_ptr.value())

            @always_inline
            def selective_reset_counter_wrapper(
                states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
                dones: Tile[dtype, BATCH_SIZE],
                counter: Tile[DType.uint64, 1],
            ):
                var env = Int(block_dim.x * block_idx.x + thread_idx.x)
                if env >= BATCH_SIZE:
                    return
                if rebind[Scalar[dtype]](dones[env]) > Scalar[dtype](0.5):
                    var combined_seed = (
                        Int(rebind[Scalar[DType.uint64]](counter[0]))
                        * 2654435761
                        + env * 12345
                    )
                    PendulumV2[Self.dtype]._reset_env_gpu[
                        BATCH_SIZE, STATE_SIZE
                    ](states, env, combined_seed)
                    dones[env] = Scalar[dtype](0.0)

            ctx.enqueue_function[selective_reset_counter_wrapper](
                states_tensor,
                dones_tensor,
                counter_t,
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )
        else:

            @always_inline
            def selective_reset_wrapper(
                states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
                dones: Tile[dtype, BATCH_SIZE],
                seed: Scalar[dtype],
            ):
                var env = Int(block_dim.x * block_idx.x + thread_idx.x)
                if env >= BATCH_SIZE:
                    return
                # Only reset if done
                if rebind[Scalar[dtype]](dones[env]) > Scalar[dtype](0.5):
                    # Combine seed with env index using prime multiplier for good distribution
                    var combined_seed = Int(seed) * 2654435761 + env * 12345
                    PendulumV2[Self.dtype]._reset_env_gpu[
                        BATCH_SIZE, STATE_SIZE
                    ](states, env, combined_seed)
                    dones[env] = Scalar[dtype](0.0)

            ctx.enqueue_function[selective_reset_wrapper](
                states_tensor,
                dones_tensor,
                Scalar[dtype](rng_seed),
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
    ) raises:
        """Extract observations from state buffer (trivial copy: obs = state[0:OBS_DIM]).
        """
        var states = tile[BATCH_SIZE, STATE_SIZE_VAL](states_buf)
        var obs = tile[BATCH_SIZE, OBS_DIM_VAL](obs_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def extract_obs(
            states: ImmTile[dtype, BATCH_SIZE, STATE_SIZE_VAL],
            obs: Tile[dtype, BATCH_SIZE, OBS_DIM_VAL],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            for d in range(OBS_DIM_VAL):
                obs[i, d] = states[i, d]

        ctx.enqueue_function[extract_obs](
            states,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[dtype]) raises:
        """No-op: Pendulum doesn't need pre-allocated workspace."""
        pass

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[dtype],
        curriculum_values: List[Scalar[dtype]],
    ) raises:
        """No-op: Pendulum doesn't use curriculum."""
        pass

    # =========================================================================
    # GPU Helper Methods (Static, Inline)
    # =========================================================================

    @always_inline
    @staticmethod
    def _step_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
        actions: ImmTile[dtype, BATCH_SIZE, ACTION_DIM],
        rewards: Tile[dtype, BATCH_SIZE],
        dones: Tile[dtype, BATCH_SIZE],
        obs: Tile[dtype, BATCH_SIZE, OBS_DIM],
        env: Int,
    ):
        """Step a single environment (GPU-compatible inline function).

        Pendulum physics with Euler integration.

        State layout (8 floats):
            [0] cos(theta)      - observation
            [1] sin(theta)      - observation
            [2] theta_dot       - observation
            [3] theta           - raw angle for physics
            [4] step_count      - metadata
            [5] done            - metadata
            [6] total_reward    - metadata
            [7] last_torque     - metadata
        """
        # Physics constants (cast Float64 to dtype)
        var MAX_SPEED = Scalar[dtype](PConstants.MAX_SPEED)
        var MAX_TORQUE = Scalar[dtype](PConstants.MAX_TORQUE)
        var DT = Scalar[dtype](PConstants.DT)
        var G = Scalar[dtype](PConstants.G)
        var M = Scalar[dtype](PConstants.M)
        var L = Scalar[dtype](PConstants.L)
        var MAX_STEPS_VAL = Scalar[dtype](PConstants.MAX_STEPS)

        # Layout offsets - NEW LAYOUT with obs at offset 0
        comptime OBS_COS = PendulumLayout.OBS_COS_THETA  # 0
        comptime OBS_SIN = PendulumLayout.OBS_SIN_THETA  # 1
        comptime OBS_THETA_DOT = PendulumLayout.OBS_THETA_DOT  # 2
        comptime THETA_ABS = PendulumLayout.THETA_ABS  # 3
        comptime META_OFF = PendulumLayout.METADATA_OFFSET  # 4
        comptime META_STEP = PendulumLayout.META_STEP_COUNT
        comptime META_DONE = PendulumLayout.META_DONE
        comptime META_TOTAL_REWARD = PendulumLayout.META_TOTAL_REWARD
        comptime META_LAST_TORQUE = PendulumLayout.META_LAST_TORQUE

        # Read current state - theta from offset 3, theta_dot from obs offset 2
        var theta = rebind[Scalar[dtype]](states[env, THETA_ABS])
        var theta_dot = rebind[Scalar[dtype]](states[env, OBS_THETA_DOT])
        var step_count = rebind[Scalar[dtype]](
            states[env, META_OFF + META_STEP]
        )
        var total_reward = rebind[Scalar[dtype]](
            states[env, META_OFF + META_TOTAL_REWARD]
        )

        # Action is interpreted as raw torque, clamped to ±MAX_TORQUE.
        # Matches PendulumEnv (V1) and Gymnasium Pendulum-v1's contract:
        # callers emit actions in the env's natural torque range
        # [-MAX_TORQUE, +MAX_TORQUE]. Agents whose policy outputs a
        # different range (e.g. tanh-squashed [-1, +1]) must scale on
        # their side before calling this kernel.
        var u = rebind[Scalar[dtype]](actions[env, 0])
        if u > MAX_TORQUE:
            u = MAX_TORQUE
        elif u < -MAX_TORQUE:
            u = -MAX_TORQUE

        # Reward computed from PRE-step (θ, θ_dot) — matches Gymnasium
        # Pendulum-v1 reference. Older versions computed reward post-step
        # which diverged subtly and made the env materially harder for
        # value-based agents. See EZ-V2 Pendulum convergence notes
        # 2026-05-10.
        var reward = -(
            theta * theta
            + Scalar[dtype](0.1) * theta_dot * theta_dot
            + Scalar[dtype](0.001) * u * u
        )

        # Physics: θ'' = (3g/2L) * sin(θ) + (3/mL²) * u
        # Use sin directly on dtype to avoid Float64 on GPU
        var sin_theta = sin(theta)
        var theta_acc = (
            Scalar[dtype](3.0) * G / (Scalar[dtype](2.0) * L)
        ) * sin_theta + (Scalar[dtype](3.0) / (M * L * L)) * u

        # Euler integration — Gymnasium clips θ_dot BEFORE the θ update.
        theta_dot = theta_dot + theta_acc * DT
        if theta_dot > MAX_SPEED:
            theta_dot = MAX_SPEED
        elif theta_dot < -MAX_SPEED:
            theta_dot = -MAX_SPEED
        theta = theta + theta_dot * DT

        # Normalize angle to [-π, π] using dtype-native pi
        var PI = Scalar[dtype](3.14159265358979323846)
        var TWO_PI = PI * Scalar[dtype](2.0)
        while theta > PI:
            theta = theta - TWO_PI
        while theta < -PI:
            theta = theta + TWO_PI

        # Increment step
        step_count = step_count + Scalar[dtype](1.0)
        total_reward = total_reward + reward

        # Check if done (pendulum never terminates early, only truncates)
        var is_done = Scalar[dtype](0.0)
        if step_count >= MAX_STEPS_VAL:
            is_done = Scalar[dtype](1.0)

        # Compute observation values
        var cos_theta = cos(theta)
        var sin_theta_val = sin(theta)

        # Write updated state (obs at offset 0, theta at offset 3, metadata at offset 4)
        states[env, OBS_COS] = cos_theta
        states[env, OBS_SIN] = sin_theta_val
        states[env, OBS_THETA_DOT] = theta_dot
        states[env, THETA_ABS] = theta
        states[env, META_OFF + META_STEP] = step_count
        states[env, META_OFF + META_DONE] = is_done
        states[env, META_OFF + META_TOTAL_REWARD] = total_reward
        states[env, META_OFF + META_LAST_TORQUE] = u

        # Write outputs
        rewards[env] = reward
        dones[env] = is_done

        # Write observation to separate obs buffer (for agent)
        obs[env, 0] = cos_theta
        obs[env, 1] = sin_theta_val
        obs[env, 2] = theta_dot

    @always_inline
    @staticmethod
    def _reset_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment to random initial state (GPU-compatible).

        Initial angle: uniform random in [-π, π]
        Initial velocity: uniform random in [-1, 1]

        State layout (8 floats):
            [0] cos(theta)      - observation
            [1] sin(theta)      - observation
            [2] theta_dot       - observation
            [3] theta           - raw angle for physics
            [4] step_count      - metadata
            [5] done            - metadata
            [6] total_reward    - metadata
            [7] last_torque     - metadata
        """
        # Layout offsets from new layout
        comptime OBS_COS = PendulumLayout.OBS_COS_THETA  # 0
        comptime OBS_SIN = PendulumLayout.OBS_SIN_THETA  # 1
        comptime OBS_THETA_DOT = PendulumLayout.OBS_THETA_DOT  # 2
        comptime THETA_ABS = PendulumLayout.THETA_ABS  # 3
        comptime META_OFF = PendulumLayout.METADATA_OFFSET  # 4

        # Generate random initial state using Philox RNG
        var rng = PhiloxRandom(seed=UInt64(seed), offset=0)
        var rand_vals = rng.step_uniform()

        # Random angle in [-π, π] using dtype-native pi to avoid Float64
        var PI = Scalar[dtype](3.14159265358979323846)
        var theta = (
            rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * PI

        # Random angular velocity in [-1, 1]
        var theta_dot = rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)

        # Clear entire state
        for i in range(STATE_SIZE):
            states[env, i] = Scalar[dtype](0.0)

        # Write observation at offset 0 (CRITICAL for GPU training!)
        states[env, OBS_COS] = cos(theta)
        states[env, OBS_SIN] = sin(theta)
        states[env, OBS_THETA_DOT] = theta_dot

        # Write raw theta for physics updates
        states[env, THETA_ABS] = theta

        # Metadata is already zeroed

    # =========================================================================
    # CPU Single-Environment Methods - BoxContinuousActionEnv Trait
    # =========================================================================

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial observation as list.

        Uses the per-instance Philox stream (seed=`self.rng_seed`,
        offset=`self.rng_counter`) to sample (θ, θ_dot) deterministically,
        then advances the counter. This replaces the previous
        `random_float64()` calls so the V2 CPU path no longer depends on
        the global `std.random` state, matching the V2 GPU `_reset_env_gpu`
        path which also uses Philox.

        To get distinct initial states across multiple envs, construct each
        env with a unique `seed=` (e.g. `PendulumV2(seed=2026 + env_id)`).
        """
        var rng = PhiloxRandom(seed=self.rng_seed, offset=self.rng_counter)
        var rand_vals = rng.step_uniform()
        self.rng_counter += 1

        # PhiloxRandom.step_uniform() returns Float32; cast to Self.dtype.
        var u0 = Scalar[Self.dtype](rand_vals[0])
        var u1 = Scalar[Self.dtype](rand_vals[1])

        var PI = Scalar[Self.dtype](pi)
        # Random initial angle in [-π, π]
        self.theta = (u0 * Scalar[Self.dtype](2.0) - Scalar[Self.dtype](1.0)) * PI
        # Random initial angular velocity in [-1, 1]
        self.theta_dot = u1 * Scalar[Self.dtype](2.0) - Scalar[Self.dtype](1.0)

        self.steps = 0
        self.done = False
        self.total_reward = Scalar[Self.dtype](0.0)
        self.last_torque = Scalar[Self.dtype](0.0)

        return self.get_obs_list()

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as list."""
        var obs = List[Scalar[Self.dtype]](capacity=3)
        obs.append(Scalar[Self.dtype](cos(Float64(self.theta))))
        obs.append(Scalar[Self.dtype](sin(Float64(self.theta))))
        obs.append(self.theta_dot)
        return obs^

    def step_continuous[
        DTYPE_SC: DType
    ](mut self, action: Scalar[DTYPE_SC]) -> Tuple[
        List[Scalar[DTYPE_SC]], Scalar[DTYPE_SC], Bool
    ]:
        """Take 1D continuous action (torque) and return (obs, reward, done)."""
        var result = self._step_with_torque(Scalar[Self.dtype](action))
        var obs_self = self.get_obs_list()
        var obs = List[Scalar[DTYPE_SC]](capacity=len(obs_self))
        for i in range(len(obs_self)):
            obs.append(Scalar[DTYPE_SC](obs_self[i]))
        return (obs^, Scalar[DTYPE_SC](result[1]), result[2])

    def step_continuous_vec[
        DTYPE_VEC: DType
    ](
        mut self, action: List[Scalar[DTYPE_VEC]], verbose: Bool = False
    ) -> Tuple[List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool]:
        """Take continuous action and return (obs, reward, done).

        Action is interpreted as raw torque in [-MAX_TORQUE, +MAX_TORQUE],
        matching PendulumEnv (V1) and Gymnasium Pendulum-v1. Out-of-range
        values are clamped inside `_step_with_torque`.
        """
        var torque = Scalar[Self.dtype](action[0]) if len(
            action
        ) > 0 else Scalar[Self.dtype](0.0)
        var result = self._step_with_torque(torque)
        var obs = List[Scalar[DTYPE_VEC]](capacity=3)
        obs.append(Scalar[DTYPE_VEC](cos(Float64(self.theta))))
        obs.append(Scalar[DTYPE_VEC](sin(Float64(self.theta))))
        obs.append(Scalar[DTYPE_VEC](self.theta_dot))
        return (obs^, Scalar[DTYPE_VEC](result[1]), result[2])

    def obs_dim(self) -> Int:
        """Return observation dimension (3)."""
        return 3

    def action_dim(self) -> Int:
        """Return action dimension (1)."""
        return 1

    def action_low(self) -> Scalar[Self.dtype]:
        """Return lower bound for action values."""
        return -self.max_torque

    def action_high(self) -> Scalar[Self.dtype]:
        """Return upper bound for action values."""
        return self.max_torque

    # =========================================================================
    # CPU Single-Environment Methods - BoxDiscreteActionEnv Trait
    # =========================================================================

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take discrete action and return (obs_list, reward, done)."""
        var torque = Scalar[Self.dtype](Float64(action - 1) * 2.0)
        var result = self._step_with_torque(torque)
        return (self.get_obs_list(), result[1], result[2])

    def num_actions(self) -> Int:
        """Return number of discrete actions (3)."""
        return 3

    # =========================================================================
    # CPU Single-Environment Methods - DiscreteEnv Trait
    # =========================================================================

    def reset(mut self) -> PendulumV2State[Self.dtype]:
        """Reset environment and return discretized state."""
        _ = self.reset_obs_list()
        return self.get_state()

    def step(
        mut self, action: PendulumV2Action[Self.dtype], verbose: Bool = False
    ) -> Tuple[PendulumV2State[Self.dtype], Scalar[Self.dtype], Bool]:
        """Take action and return (state, reward, done)."""
        var torque = action.torque
        var result = self._step_with_torque(torque)
        return (self.get_state(), result[1], result[2])

    def get_state(mut self) -> PendulumV2State[Self.dtype]:
        """Return current observation state."""
        return PendulumV2State[Self.dtype].from_theta(
            self.theta, self.theta_dot
        )

    def state_to_index(self, state: PendulumV2State[Self.dtype]) -> Int:
        """Convert state to index for tabular methods."""
        return self._discretize_obs()

    def action_from_index(
        self, action_idx: Int
    ) -> PendulumV2Action[Self.dtype]:
        """Create action from index."""
        return PendulumV2Action[Self.dtype].from_discrete(action_idx)

    def num_states(self) -> Int:
        """Return total number of discrete states."""
        return self.num_bins_angle * self.num_bins_velocity

    # =========================================================================
    # Internal CPU Helpers
    # =========================================================================

    def _step_with_torque(
        mut self, torque: Scalar[Self.dtype]
    ) -> Tuple[PendulumV2State[Self.dtype], Scalar[Self.dtype], Bool]:
        """Internal step function that accepts continuous torque."""
        # Clamp torque
        var u = torque
        if u > self.max_torque:
            u = self.max_torque
        elif u < -self.max_torque:
            u = -self.max_torque

        self.last_torque = u

        # Reward computed from PRE-step state (Gymnasium Pendulum-v1 order).
        # Older V2 CPU computed reward POST-step using the updated θ/θ_dot,
        # which diverged from V1 / V2 GPU / Gymnasium and made the env
        # materially harder for value-based agents. Fixed 2026-05-15.
        var reward = -(
            self.theta * self.theta
            + Scalar[Self.dtype](0.1) * self.theta_dot * self.theta_dot
            + Scalar[Self.dtype](0.001) * u * u
        )

        # Physics: θ'' = (3g/2L) * sin(θ) + (3/mL²) * u
        var sin_theta = Scalar[Self.dtype](sin(Float64(self.theta)))
        var theta_acc = (Scalar[Self.dtype](3.0) * self.g) / (
            Scalar[Self.dtype](2.0) * self.l
        ) * sin_theta + (
            Scalar[Self.dtype](3.0) / (self.m * self.l * self.l)
        ) * u

        # Euler integration — Gymnasium clips θ_dot BEFORE the θ update.
        # Older V2 CPU updated θ first and then clamped θ_dot, which let
        # over-MAX_SPEED velocity propagate into θ for one step.
        # Fixed 2026-05-15.
        self.theta_dot = self.theta_dot + theta_acc * self.dt
        if self.theta_dot > self.max_speed:
            self.theta_dot = self.max_speed
        elif self.theta_dot < -self.max_speed:
            self.theta_dot = -self.max_speed
        self.theta = self.theta + self.theta_dot * self.dt

        # Normalize angle to [-π, π]
        self.theta = self._angle_normalize(self.theta)

        self.steps += 1
        self.total_reward += reward

        # Pendulum never terminates early, only truncates at max_steps
        self.done = self.steps >= self.max_steps

        return (self.get_state(), reward, self.done)

    def _angle_normalize(self, x: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Normalize angle to [-π, π]."""
        var result = x
        var pi_val = Scalar[Self.dtype](pi)
        var two_pi = Scalar[Self.dtype](2.0 * pi)
        while result > pi_val:
            result -= two_pi
        while result < -pi_val:
            result += two_pi
        return result

    def _discretize_obs(self) -> Int:
        """Discretize current continuous observation into a single state index.
        """

        def bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b_angle = bin_value(
            Float64(self.theta), -pi, pi, self.num_bins_angle
        )
        var b_vel = bin_value(
            Float64(self.theta_dot),
            Float64(-self.max_speed),
            Float64(self.max_speed),
            self.num_bins_velocity,
        )

        return b_angle * self.num_bins_velocity + b_vel

    def is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    # =========================================================================
    # Rendering
    # =========================================================================

    def render(mut self, mut renderer: Renderer2D):
        """Render the current state using SDL2.

        Args:
            renderer: External renderer to use for drawing.
        """
        if not renderer.begin_frame():
            return

        # Convert state variables to Float64 for rendering
        var theta_f64 = Float64(self.theta)
        var theta_dot_f64 = Float64(self.theta_dot)
        var last_torque_f64 = Float64(self.last_torque)

        # Colors
        var sky_color = sky_blue()
        var rod_color = rgb(139, 69, 19)  # Saddle brown
        var bob_color = rgb(255, 0, 0)  # Red
        var pivot_color = rgb(50, 50, 50)  # Dark gray
        var torque_color = rgb(0, 200, 0)  # Green

        # Clear screen with sky color
        renderer.clear_with_color(sky_color)

        # Create camera centered on screen (Y-flip for physics coords)
        var zoom = 100.0  # pixels per world unit
        var camera = renderer.make_camera(zoom, True)

        # World coordinates
        var pivot = Vec2(0.0, 0.0)  # Pivot at origin
        var rod_length_world = 1.5  # Rod length in world units
        var bob_radius_world = 0.2

        # Draw reference circle (the trajectory the bob follows)
        renderer.draw_circle_world(
            pivot, rod_length_world, camera, light_gray(), False
        )

        # Draw torque indicator (arc showing applied torque)
        if last_torque_f64 != 0.0:
            var torque_scale = abs(last_torque_f64) * 0.3
            var torque_direction = 1.0 if last_torque_f64 > 0 else -1.0
            var arc_end = Vec2(
                torque_direction * 0.3,
                0.3 + torque_scale,
            )
            renderer.draw_line_world(
                pivot + Vec2(0, 0.3),
                pivot + arc_end,
                camera,
                torque_color,
                4,
            )

        # Draw pendulum using helper
        # Note: theta=0 points up (negative Y in screen coords before flip)
        renderer.draw_pendulum(
            pivot,
            theta_f64 + pi,  # Adjust so 0 = down for the helper
            rod_length_world,
            bob_radius_world,
            camera,
            rod_color,
            bob_color,
            pivot_color,
            8,  # rod width
        )

        # Draw bob border
        var bob_pos = Vec2(
            pivot.x + rod_length_world * sin(theta_f64),
            pivot.y - rod_length_world * cos(theta_f64),
        )
        renderer.draw_circle_world(
            bob_pos, bob_radius_world, camera, black(), False
        )

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(self.steps))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        info_lines.append(
            "Angle: " + fit(String(theta_f64 * 180.0 / pi), 6) + " deg"
        )
        info_lines.append("Vel: " + fit(String(theta_dot_f64), 6))
        info_lines.append("Torque: " + fit(String(last_torque_f64), 5))
        renderer.draw_info_box(info_lines)

        # Update display
        renderer.flip()

    def close(mut self):
        """Clean up resources."""
        if self._renderer_initialized:
            self._renderer.value()[].close()
            self._renderer.value().unsafe_free()
            self._renderer_initialized = False

    # =========================================================================
    # Static Factory Methods
    # =========================================================================

    @staticmethod
    def make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding[Self.dtype]:
        """Create tile coding configured for Pendulum environment.

        Pendulum observation: [cos(θ), sin(θ), θ_dot]

        Args:
            num_tilings: Number of tilings (default 8).
            tiles_per_dim: Tiles per dimension (default 8).

        Returns:
            TileCoding[Self.dtype] configured for Pendulum observation space.
        """
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)  # cos(θ)
        tiles.append(tiles_per_dim)  # sin(θ)
        tiles.append(tiles_per_dim)  # θ_dot

        # Observation bounds
        var state_low = List[Scalar[Self.dtype]]()
        state_low.append(-1.0)  # cos(θ) min
        state_low.append(-1.0)  # sin(θ) min
        state_low.append(-8.0)  # θ_dot min

        var state_high = List[Scalar[Self.dtype]]()
        state_high.append(1.0)  # cos(θ) max
        state_high.append(1.0)  # sin(θ) max
        state_high.append(8.0)  # θ_dot max

        return TileCoding[Self.dtype](
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    @staticmethod
    def make_poly_features(degree: Int = 2) -> PolynomialFeatures[Self.dtype]:
        """Create polynomial features for Pendulum (3D observation).

        Args:
            degree: Maximum polynomial degree.

        Returns:
            PolynomialFeatures[Self.dtype] extractor configured for Pendulum.
        """
        var state_low = List[Scalar[Self.dtype]]()
        state_low.append(-1.0)  # cos(θ)
        state_low.append(-1.0)  # sin(θ)
        state_low.append(-8.0)  # θ_dot

        var state_high = List[Scalar[Self.dtype]]()
        state_high.append(1.0)  # cos(θ)
        state_high.append(1.0)  # sin(θ)
        state_high.append(8.0)  # θ_dot

        return PolynomialFeatures[Self.dtype](
            state_dim=3,
            degree=degree,
            include_bias=True,
            state_low=state_low^,
            state_high=state_high^,
        )

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    def init_renderer(mut self) raises -> Bool:
        """Initialize the SDL2 renderer."""
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(Renderer2D())
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        """Render the current frame using the internal renderer."""
        if not self._renderer_initialized:
            return
        self.render(self._renderer.value()[])

    def close_renderer(mut self) raises -> None:
        """Close and free the SDL2 renderer."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].close()
        self._renderer.value().unsafe_free()
        self._renderer_initialized = False

    def is_renderer_open(self) -> Bool:
        """Return True if the renderer window is open."""
        if not self._renderer_initialized:
            return False
        return not self._renderer.value()[].get_should_quit()

    def check_renderer_quit(mut self) -> Bool:
        """Return True if the renderer has received a quit event."""
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].get_should_quit()

    def renderer_delay(self, ms: Int) -> None:
        """Delay for frame rate control."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

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
