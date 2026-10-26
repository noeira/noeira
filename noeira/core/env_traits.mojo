"""Environment trait hierarchy for different state/action space types.

This module defines traits that categorize environments by their state and action
space types, enabling compile-time checking of algorithm-environment compatibility.

Trait Hierarchy:
    Env (base)
    ├── DiscreteStateEnv   - States can be converted to integer indices
    ├── ContinuousStateEnv - States are continuous observation vectors (List[Float64])
    ├── DiscreteActionEnv  - Actions are discrete integers
    └── ContinuousActionEnv - Actions are continuous vectors

Combined Traits:
    DiscreteEnv           = DiscreteStateEnv & DiscreteActionEnv
    BoxDiscreteActionEnv  = ContinuousStateEnv & DiscreteActionEnv
    BoxContinuousActionEnv = ContinuousStateEnv & ContinuousActionEnv

GPU Environment Traits (Experimental):
    GPUDiscreteEnv    - GPU-compatible discrete action environments (e.g., LunarLander)
    GPUContinuousEnv  - GPU-compatible continuous action environments (e.g., CarRacing)

Environments implement combinations:
    GridWorld:   DiscreteEnv (tabular)
    CartPole:    DiscreteEnv + BoxDiscreteActionEnv (4D obs)
    MountainCar: DiscreteEnv + BoxDiscreteActionEnv (2D obs)
    Acrobot:     DiscreteEnv + BoxDiscreteActionEnv (6D obs)
    Pendulum:    DiscreteEnv + BoxContinuousActionEnv (3D obs, 1D action)
    LunarLander: BoxDiscreteActionEnv + GPUDiscreteEnv (8D obs, 4 actions)
    CarRacing:   BoxContinuousActionEnv + GPUContinuousEnv (13D obs, 3D action)

Algorithms specify requirements:
    Q-Learning (tabular): DiscreteEnv
    Tile-coded Q-Learning: BoxDiscreteActionEnv
    PPO:                   BoxDiscreteActionEnv (or BoxContinuousActionEnv)
    SAC/DDPG/TD3:          BoxContinuousActionEnv
    GPU DQN:               GPUDiscreteEnv
    GPU SAC/DDPG/TD3:      GPUContinuousEnv
"""

from .env import Env
from noeira.nn.constants import DT as dtype
from max.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import Pointer
from noeira.core.tiles import Tile, tile


# ============================================================================
# Renderable Environment Trait
# ============================================================================


trait RenderableEnv:
    """Trait for environments that support visualization.

    This trait enables environment-owned rendering, where the environment
    manages its own renderer internally. This decouples algorithms from
    specific renderer types (2D vs 3D) and allows a unified rendering
    interface across all environments.

    Benefits:
    - Algorithms don't need to know about renderer types
    - Same `render: Bool` parameter works for all environments
    - Rendering details stay encapsulated within the environment
    - Environments can use whatever renderer is appropriate (2D or 3D)

    Usage in algorithms:
        def evaluate[E: BoxContinuousActionEnv & RenderableEnv](
            self, mut env: E, render: Bool = False
        ):
            if render:
                _ = env.init_renderer()

            for episode in range(num_episodes):
                # ... episode loop ...
                if render:
                    env.render_frame()
                    if env.check_renderer_quit():
                        break

            if render:
                env.close_renderer()

    Environments that don't support rendering can implement these as no-ops.
    """

    def init_renderer(mut self) raises -> Bool:
        """Initialize the renderer.

        Creates and initializes the internal renderer. Should be called
        before any render() calls. Multiple calls are safe (no-op if
        already initialized).

        Returns:
            True if initialization succeeded or was already initialized.
        """
        ...

    def init_renderer(mut self, show_velocity: Bool) raises -> Bool:
        """Initialize the renderer with the option to show velocity.

        Returns:
            True if initialization succeeded or was already initialized.
        """
        return self.init_renderer()

    def render_frame(mut self) raises -> None:
        """Render the current environment state.

        No-op if renderer is not initialized. This method handles all
        rendering internally - the environment knows how to visualize itself.
        """
        ...

    def close_renderer(mut self) raises -> None:
        """Close the renderer and release resources.

        Safe to call multiple times or if renderer was never initialized.
        """
        ...

    def is_renderer_open(self) -> Bool:
        """Check if renderer is currently initialized and open.

        Returns:
            True if renderer is initialized and window is open.
        """
        ...

    def check_renderer_quit(mut self) -> Bool:
        """Check if user requested to close the renderer window.

        Returns:
            True if quit was requested (e.g., user closed window).
        """
        ...

    def renderer_delay(self, ms: Int) -> None:
        """Delay for specified milliseconds (for frame rate control).

        No-op if renderer is not initialized.

        Args:
            ms: Milliseconds to delay.
        """
        ...

    def renderer_is_paused(self) -> Bool:
        """Check if the renderer is currently paused (Space key).

        Returns:
            True if simulation is paused.
        """
        ...

    def renderer_step_once(self) -> Bool:
        """Check if the user requested a single step while paused (Right arrow).

        This flag is consumed each frame by check_renderer_quit().

        Returns:
            True if a single step was requested.
        """
        ...

    def start_recording(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        """Start recording rendered frames to a video file.

        Format is determined by file extension (.gif or .mp4).
        No-op if renderer is not initialized.
        """
        pass

    def stop_recording(mut self) raises:
        """Stop recording and finalize the output file.

        No-op if renderer is not initialized or not recording.
        """
        pass


# ============================================================================
# State Space Traits
# ============================================================================


trait DiscreteStateEnv(Env):
    """Environment with discrete states that can be indexed.

    Use this for tabular methods where states map to integer indices.
    The state space must be finite and enumerable.

    Examples: GridWorld, FrozenLake, Taxi, discretized CartPole
    """

    def state_to_index(self, state: Self.StateType) -> Int:
        """Convert a state to an integer index for tabular methods."""
        ...

    def num_states(self) -> Int:
        """Return the total number of discrete states."""
        ...


trait ContinuousStateEnv(Env):
    """Environment with continuous observation/state vectors.

    Use this for function approximation methods (tile coding, neural networks)
    where states are represented as continuous vectors.

    Observations are returned as List[Float64] for flexibility with any
    observation dimension. Environments may also provide SIMD-optimized
    methods internally for performance.

    Examples: CartPole (4D), MountainCar (2D), Acrobot (6D), MuJoCo environments.
    """

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a flexible list."""
        ...

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial continuous observation."""
        ...

    def obs_dim(self) -> Int:
        """Return the dimension of the observation vector."""
        ...


# ============================================================================
# Action Space Traits
# ============================================================================


trait DiscreteActionEnv(Env):
    """Environment with discrete action space.

    Actions are represented as integer indices. Use this for environments
    where the agent chooses from a finite set of actions.

    Examples: CartPole (left/right), GridWorld (up/down/left/right)
    """

    def action_from_index(self, action_idx: Int) -> Self.ActionType:
        """Create an action from an integer index."""
        ...

    def num_actions(self) -> Int:
        """Return the number of discrete actions available."""
        ...


trait ContinuousActionEnv(Env):
    """Environment with continuous action space.

    Actions are continuous vectors. Use this for fine-grained control tasks
    where actions can take any value within bounds.

    Examples: Pendulum (torque), HalfCheetah (joint torques)
    """

    def action_dim(self) -> Int:
        """Return the dimension of the action vector."""
        ...

    def action_low(self) -> Scalar[Self.dtype]:
        """Return the lower bound for action values.

        Note: Assumes symmetric bounds. For asymmetric bounds,
        environments should provide additional methods.
        """
        ...

    def action_high(self) -> Scalar[Self.dtype]:
        """Return the upper bound for action values."""
        ...


# ============================================================================
# Combined Traits (Common Combinations)
# ============================================================================


trait DiscreteEnv(DiscreteActionEnv, DiscreteStateEnv):
    """Environment with discrete states and actions suitable for tabular RL.

    Combines discrete state and discrete action spaces.
    Use with Q-Learning, SARSA, Monte Carlo, etc.

    This is the primary trait for tabular RL methods that require
    integer indices for states and actions.

    Examples: GridWorld, FrozenLake, Taxi, discretized CartPole
    """

    pass


trait BoxDiscreteActionEnv(ContinuousStateEnv, DiscreteActionEnv):
    """Environment with continuous observations (Box space) and discrete actions.

    Use with function approximation algorithms that handle continuous observations
    but discrete action selection:
    - Tile coding / Linear function approximation
    - Policy gradient methods (REINFORCE, Actor-Critic, PPO)
    - DQN

    Examples: CartPole (4D), MountainCar (2D), Acrobot (6D), LunarLander.
    """

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take discrete action and return (continuous_obs, reward, done).

        Convenience method for function approximation algorithms that
        work with raw observations and integer actions.
        """
        ...

    def step_obs_into(
        mut self,
        action: Int,
        obs_out: Pointer[Scalar[Self.dtype], MutAnyOrigin],
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Take discrete action, writing the observation directly into
        `obs_out` (caller-owned, `obs_dim()` scalars). Returns
        (reward, done).

        Allocation-free alternative to `step_obs` for hot batched
        stepping loops (`BatchedCpuDiscreteEnv`): the default delegates
        to `step_obs` and copies, so every env conforms; envs with a
        large observation (AtariEnv pixel mode: 28,224 floats rebuilt
        per step) override it to skip the per-step List allocation and
        write the obs buffer once."""
        var res = self.step_obs(action)
        for d in range(len(res[0])):
            obs_out[unsafe_offset=d] = res[0][d]
        return (res[1], res[2])


trait BoxContinuousActionEnv(ContinuousActionEnv, ContinuousStateEnv):
    """Environment with continuous observations and continuous actions.

    Use with continuous control algorithms:
    - Policy gradient with Gaussian policies
    - DDPG, TD3, SAC

    Examples: Pendulum (3D obs, 1D action), BipedalWalker (24D obs, 4D action).
    """

    def step_continuous[
        DTYPE: DType
    ](mut self, action: Scalar[DTYPE]) -> Tuple[
        List[Scalar[DTYPE]], Scalar[DTYPE], Bool
    ]:
        """Take 1D continuous action and return (continuous_obs, reward, done).

        Convenience method for environments with single-dimensional actions.
        For multi-dimensional actions, use step_continuous_vec instead.
        """
        ...

    def step_continuous_vec[
        DTYPE: DType
    ](mut self, action: List[Scalar[DTYPE]], verbose: Bool = False) -> Tuple[
        List[Scalar[DTYPE]], Scalar[DTYPE], Bool
    ]:
        """Take multi-dimensional continuous action and return (obs, reward, done).

        This is the primary method for continuous control algorithms (SAC, DDPG, TD3)
        that work with multi-dimensional action spaces.

        Args:
            action: List of action values, length should match action_dim().
            verbose: Whether to print debug information.

        Returns:
            Tuple of (observation_list, reward, done).
        """
        ...


# ============================================================================
# Termination-aware variant
# ============================================================================


trait TerminationAwareEnv(BoxContinuousActionEnv):
    """Deprecated alias kept for legacy `deep_agents` callers. `was_terminated`
    now lives on the base `Env` trait (single home, inherited by every env), so
    this trait adds nothing beyond `BoxContinuousActionEnv` — it remains only
    so existing `[E: TerminationAwareEnv]` bounds keep resolving."""

    pass


# ============================================================================
# GPU Environment Trait for Composable GPU RL. (Experimental)
# ============================================================================


trait GPUDiscreteEnv:
    """Trait for GPU-compatible discrete action environments.

    Environments must define compile-time constants and inline methods
    for use in fused GPU kernels.
    """

    # Compile-time constants for environment dimensions
    comptime STATE_SIZE: Int
    comptime OBS_DIM: Int
    comptime NUM_ACTIONS: Int

    # Pre-allocated workspace sizes (0 = no pre-allocation needed)
    comptime STEP_WS_SHARED: Int  # Shared buffer (e.g. model) — same across envs
    comptime STEP_WS_PER_ENV: Int  # Per-env buffer (e.g. physics workspace)

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        mut terminated: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
        workspace_ptr: Optional[
            Pointer[Scalar[dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        """Perform one environment step and extract observations.

        Args:
            ctx: GPU device context.
            states: State buffer on GPU.
            actions: Actions buffer on GPU.
            rewards: Rewards buffer on GPU (output).
            dones: Done flags buffer on GPU (output). 1.0 if terminated OR truncated.
            terminated: Terminated flags buffer on GPU (output). 1.0 only if truly terminated (not truncated).
            obs: Observations buffer on GPU (output).
            rng_seed: Optional random seed for physics (e.g., engine dispersion).
            workspace_ptr: Optional pre-allocated workspace pointer.
                          When non-null, avoids per-step GPU buffer allocation.
                          Layout: [shared: STEP_WS_SHARED | per_env: BATCH * STEP_WS_PER_ENV].
                          When null, allocates internally (backward compatible).
            rng_counter_ptr: Optional GPU-side RNG counter pointer.
                            When non-null, kernel reads seed from GPU memory
                            (CUDA graph compatible). When null, uses rng_seed scalar.
        """
        ...

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset state to random initial values.

        Args:
            ctx: GPU device context.
            states: State buffer on GPU.
            rng_seed: Random seed for terrain/initial state generation.
                     Use different values across calls to get varied environments.
        """
        ...

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
        """Reset only done environments to random initial values.

        This enables efficient vectorized training where only completed
        episodes are reset while others continue running.

        Args:
            ctx: GPU device context.
            states: State buffer on GPU.
            dones: Done flags buffer on GPU.
            rng_seed: Random seed for terrain/initial state generation.
                     Should be different each call (e.g., training step counter).
                     Ignored when rng_counter_ptr is non-null.
            workspace_ptr: Optional pre-allocated workspace pointer.
                          When non-null, reuses model data from workspace
                          instead of allocating a new buffer each call.
            rng_counter_ptr: Optional GPU-side RNG counter pointer.
                            When non-null, kernel reads seed from GPU memory
                            (CUDA graph compatible). When null, uses rng_seed scalar.
        """
        ...

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[dtype],) raises:
        """Initialize pre-allocated step workspace (call once at setup).

        No-op for environments with STEP_WS_SHARED == 0.
        """
        ...

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[dtype],
        curriculum_values: List[Scalar[dtype]],
    ) raises:
        """Update curriculum parameters in pre-allocated workspace.

        No-op for environments with STEP_WS_SHARED == 0.
        """
        ...

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
    ) raises:
        """Extract observations from state buffer for all environments.

        Default: copy `obs[e] = state[e][0:OBS_DIM]` for each env.
        Most discrete-action envs (CartPole, MountainCar, Acrobot,
        LunarLander) have observations that are a prefix of the state,
        so the default works out of the box. Override when the obs
        derives from state in a non-trivial way.

        Used after `reset_kernel_gpu` / `selective_reset_kernel_gpu` to
        seed obs (those kernels write state but not obs); also used by
        AlphaZero's single-player `train_gpu` to fetch the current obs
        for the MCTS root prediction before choosing an action.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            obs: Observations output [BATCH_SIZE * OBS_DIM].
        """
        var states_t = tile[BATCH_SIZE * STATE_SIZE](states)
        var obs_t = tile[BATCH_SIZE * OBS_DIM](obs)

        comptime TPB = 256
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def extract_wrapper(
            s: Tile[dtype, BATCH_SIZE * STATE_SIZE],
            o: Tile[dtype, BATCH_SIZE * OBS_DIM],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return
            for d in range(OBS_DIM):
                o[e * OBS_DIM + d] = s[e * STATE_SIZE + d]

        ctx.enqueue_function[extract_wrapper](
            states_t,
            obs_t,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


trait GPUContinuousEnv:
    """Trait for GPU-compatible continuous action environments.

    Environments must define compile-time constants and inline methods
    for use in fused GPU kernels. Unlike GPUDiscreteEnv, actions are
    continuous vectors (e.g., [steering, gas, brake] for CarRacing).

    Examples: CarRacing (3D actions), Pendulum (1D action), BipedalWalker (4D actions).
    """

    # Compile-time constants for environment dimensions
    comptime STATE_SIZE: Int
    comptime OBS_DIM: Int
    comptime ACTION_DIM: Int

    # Pre-allocated workspace sizes (0 = no pre-allocation needed)
    comptime STEP_WS_SHARED: Int  # Shared buffer (e.g. model) — same across envs
    comptime STEP_WS_PER_ENV: Int  # Per-env buffer (e.g. physics workspace)

    comptime NAME: String

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
        """Perform one environment step with continuous actions.

        Args:
            ctx: GPU device context.
            states: State buffer on GPU [BATCH_SIZE * STATE_SIZE].
            actions: Continuous actions buffer on GPU [BATCH_SIZE * ACTION_DIM].
            rewards: Rewards buffer on GPU (output) [BATCH_SIZE].
            dones: Done flags buffer on GPU (output) [BATCH_SIZE]. 1.0 if terminated OR truncated.
            terminated: Terminated flags buffer on GPU (output) [BATCH_SIZE]. 1.0 only if truly terminated (not truncated).
            obs: Observations buffer on GPU (output) [BATCH_SIZE * OBS_DIM].
            rng_seed: Optional random seed for physics. Ignored when rng_counter_ptr is non-null.
            curriculum_values: Environment-specific curriculum parameters.
                              Empty list uses default (strict) bounds.
            workspace_ptr: Optional pre-allocated workspace pointer.
                          When non-null, avoids per-step GPU buffer allocation.
                          Layout: [shared: STEP_WS_SHARED | per_env: BATCH * STEP_WS_PER_ENV].
                          When null, allocates internally (backward compatible).
            rng_counter_ptr: Optional GPU-side RNG counter pointer.
                            When non-null, kernel reads seed from GPU memory
                            (CUDA graph compatible). When null, uses rng_seed scalar.
        """
        ...

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments to random initial values.

        Args:
            ctx: GPU device context.
            states: State buffer on GPU.
            rng_seed: Random seed for terrain/initial state generation.
                     Use different values across calls to get varied environments.
        """
        ...

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
        """Reset only done environments to random initial values.

        This enables efficient vectorized training where only completed
        episodes are reset while others continue running.

        Args:
            ctx: GPU device context.
            states: State buffer on GPU.
            dones: Done flags buffer on GPU.
            rng_seed: Random seed for terrain/initial state generation.
                     Should be different each call (e.g., training step counter).
                     Ignored when rng_counter_ptr is non-null.
            workspace_ptr: Optional pre-allocated workspace pointer.
                          When non-null, reuses model data from workspace
                          instead of allocating a new buffer each call.
            rng_counter_ptr: Optional GPU-side RNG counter pointer.
                            When non-null, kernel reads seed from GPU memory
                            (CUDA graph compatible). When null, uses rng_seed scalar.
        """
        ...

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
    ) raises:
        """Extract observations from state buffer for all environments.

        Each environment implements this to correctly map its internal state
        representation to the observation vector expected by the neural network.
        For environments where obs = state[0:OBS_DIM], this is a simple copy.
        For GC environments, this extracts the correct qpos/qvel subset.

        This is called after reset (initial obs) and after selective_reset
        (to update obs for reset environments). step_kernel_gpu handles its
        own obs extraction internally.

        Args:
            ctx: GPU device context.
            states: State buffer on GPU [BATCH_SIZE * STATE_SIZE].
            obs: Observations buffer on GPU (output) [BATCH_SIZE * OBS_DIM].
        """
        ...

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[dtype],) raises:
        """Initialize pre-allocated step workspace (call once at setup).

        For environments with STEP_WS_SHARED > 0, this initializes the shared
        portion (e.g. physics model) of the workspace buffer. The per-env
        portion doesn't need initialization.

        No-op for environments with STEP_WS_SHARED == 0.

        Args:
            ctx: GPU device context.
            workspace_buf: Buffer of size STEP_WS_SHARED + BATCH_SIZE * STEP_WS_PER_ENV.
        """
        ...

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[dtype],
        curriculum_values: List[Scalar[dtype]],
    ) raises:
        """Update curriculum parameters in pre-allocated workspace.

        Much cheaper than full model initialization — patches only curriculum
        floats instead of rebuilding the entire model buffer.

        No-op for environments with STEP_WS_SHARED == 0.

        Args:
            ctx: GPU device context.
            workspace_buf: Pre-allocated workspace (model at offset 0).
            curriculum_values: Environment-specific curriculum parameters.
        """
        ...

    @staticmethod
    def is_terminal_obs_gpu[
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        obs: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
    ) raises:
        """Check termination from observations for a batch (GPU).

        Used by model-based agents (MBPO) to detect terminal states in
        synthetic rollouts where only predicted observations are available.

        Default: no termination (unsafe_memset dones to 0). Override for envs
        with observation-based termination conditions (Hopper, Ant, etc.).

        Args:
            ctx: GPU device context.
            obs: Observations buffer [BATCH_SIZE * OBS_DIM].
            dones: Output done flags [BATCH_SIZE]. 1.0 = terminal.
        """
        ctx.enqueue_memset(dones, 0)


trait CurriculumScheduler(Copyable, Movable):
    """Trait for environments that support curriculum scheduling.

    Environments must define a method to get the curriculum values.
    """

    @staticmethod
    def get_params[
        DTYPE: DType
    ](progress: Scalar[DTYPE]) -> List[Scalar[DTYPE]]:
        """Get curriculum parameters for given training progress.

        Uses linear interpolation from initial to final values.

        Args:
            progress: Training progress from 0.0 (start) to 1.0 (end).
                     Values outside [0, 1] are clamped.

        Returns:
            List of curriculum parameter values.
        """
        ...

    @staticmethod
    def get_stage_name[DTYPE: DType](progress: Scalar[DTYPE]) -> String:
        """Get human-readable curriculum stage name.

        Used for logging stage transitions during training.

        Args:
            progress: Training progress from 0.0 to 1.0.

        Returns:
            Stage name string, or empty string if no curriculum stages.
        """
        ...


struct NoCurriculumScheduler(CurriculumScheduler):
    """No curriculum scheduler.

    This is a placeholder for environments that do not support curriculum scheduling.
    """

    @staticmethod
    def get_params[
        DTYPE: DType
    ](progress: Scalar[DTYPE]) -> List[Scalar[DTYPE]]:
        return []

    @staticmethod
    def get_stage_name[DTYPE: DType](progress: Scalar[DTYPE]) -> String:
        return ""


# ============================================================================
# Two-Player Turn-Based Environment Traits
# ============================================================================


trait TwoPlayerDiscreteEnv(BoxDiscreteActionEnv):
    """Environment with two players, turn-based, discrete actions.

    Extends BoxDiscreteActionEnv for backward compatibility — board game
    environments can be used with existing single-agent training loops
    (with an internal opponent) or with self-play training loops.

    Observations are always CANONICAL: from the perspective of the
    player about to move. "My pieces" = plane 0, "opponent pieces" = plane 1.
    This enables single-network self-play.

    step() advances one turn. The returned reward is from the perspective
    of the player who just moved.

    Examples: TicTacToe, ConnectFour, Go, Chess.
    """

    def current_player(self) -> Int:
        """Return which player is about to move (0 or 1)."""
        ...

    def legal_action_mask(self) -> List[Bool]:
        """Return mask of legal actions (length = num_actions()).

        True = legal, False = illegal. Used for action masking in
        policy networks and MCTS.
        """
        ...

    def game_result(self) -> Int:
        """Return game outcome.

        Returns:
            0 = ongoing, 1 = player 0 wins, 2 = player 1 wins, 3 = draw.
        """
        ...


trait GPUTwoPlayerDiscreteEnv:
    """Trait for GPU-compatible two-player discrete action environments.

    Similar to GPUDiscreteEnv but with an additional legal_masks output
    from step_kernel_gpu. The legal mask for the NEXT state is computed
    during the step to avoid a separate kernel launch.

    The step kernel also handles canonical observation extraction —
    observations are always from the perspective of the next player to move.
    """

    # Compile-time constants for environment dimensions
    comptime STATE_SIZE: Int
    comptime OBS_DIM: Int
    comptime NUM_ACTIONS: Int

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        mut terminated: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
        mut legal_masks: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        """Perform one environment step for all games in batch.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            actions: Actions buffer [BATCH_SIZE].
            rewards: Rewards output [BATCH_SIZE]. From perspective of player who moved.
            dones: Done flags output [BATCH_SIZE]. 1.0 if game ended.
            terminated: Terminated flags output [BATCH_SIZE].
            obs: Canonical observations output [BATCH_SIZE * OBS_DIM].
            legal_masks: Legal action mask for NEXT state [BATCH_SIZE * NUM_ACTIONS].
            rng_seed: Random seed. Ignored when rng_counter_ptr is non-null.
            rng_counter_ptr: Optional GPU-side RNG counter (CUDA graph compatible).
        """
        ...

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all games to initial state.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            rng_seed: Random seed.
        """
        ...

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        rng_seed: UInt64,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        """Reset only finished games.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            dones: Done flags buffer [BATCH_SIZE].
            rng_seed: Random seed. Ignored when rng_counter_ptr is non-null.
            rng_counter_ptr: Optional GPU-side RNG counter (CUDA graph compatible).
        """
        ...

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
        mut legal_masks: DeviceBuffer[dtype],
    ) raises:
        """Extract canonical observations and legal masks from state.

        Called after reset to get initial obs + masks.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            obs: Observations output [BATCH_SIZE * OBS_DIM].
            legal_masks: Legal mask output [BATCH_SIZE * NUM_ACTIONS].
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Saveable Trait — save/restore environment state
# ═══════════════════════════════════════════════════════════════════════════


trait Saveable:
    """Environments that support saving and restoring their full internal state.

    Required for CPU MCTS with true game rules — each simulation needs to
    save the current position, explore a tree path, then restore.

    The state is stored as a flat array of Scalar[dtype] values plus a done flag.
    This matches the GPU state layout used by GPUTwoPlayerDiscreteEnv.
    """

    comptime SAVE_SIZE: Int
    """Size of the state array to save/restore."""

    def save_env_state(self, mut dst: List[Scalar[dtype]]):
        """Copy current state into `dst[0:SAVE_SIZE]` (caller pre-sizes `dst`
        to >= SAVE_SIZE). Owned `List` — no raw pointer, no manual free."""
        ...

    def load_env_state(mut self, src: List[Scalar[dtype]]):
        """Restore state from `src[0:SAVE_SIZE]`."""
        ...
