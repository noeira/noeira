"""CPU vs GPU Comparison Script for LunarLander Environment.

This script compares the behavior of LunarLander between CPU and GPU modes to
identify any discrepancies in physics, observations, or rewards.

Usage:
    pixi run -e apple mojo run -I . examples/lunar_lander_cpu_gpu_compare.mojo
"""

from std.math import sqrt
from max.gpu.host import DeviceContext, DeviceBuffer
from noeira.core.tiles import Tile, ImmTile, tile
from std.random import seed as set_seed

from noeira.envs.lunar_lander import LunarLander
from noeira.envs.lunar_lander.constants import LLConstants
from noeira.physics2d import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
)


# =============================================================================
# Comparison Utilities
# =============================================================================


def print_separator():
    print("=" * 70)


def print_header(title: String):
    print_separator()
    print(title)
    print_separator()


def abs_f64(x: Float64) -> Float64:
    """Helper for absolute value."""
    if x < 0:
        return -x
    return x


def compare_scalar(
    name: String,
    cpu_val: Scalar[dtype],
    gpu_val: Scalar[dtype],
    tolerance: Float64 = 1e-4,
) -> Bool:
    """Compare two scalar values and print result."""
    var diff = abs_f64(Float64(cpu_val - gpu_val))
    var is_match = diff < tolerance
    var status = "OK" if is_match else "MISMATCH"
    print(
        "  ",
        name,
        "| CPU:",
        fit(String(Float64(cpu_val)), 10),
        "| GPU:",
        fit(String(Float64(gpu_val)), 10),
        "| Diff:",
        fit(String(diff), 10),
        "|",
        status,
    )
    return is_match


def compare_observation(
    cpu_obs: Array[Scalar[dtype], 8],
    gpu_obs: Array[Scalar[dtype], 8],
    tolerance: Float64 = 1e-4,
) -> Bool:
    """Compare CPU and GPU observations."""
    var obs_names = List[String]()
    obs_names.append("x_norm")
    obs_names.append("y_norm")
    obs_names.append("vx_norm")
    obs_names.append("vy_norm")
    obs_names.append("angle")
    obs_names.append("omega_norm")
    obs_names.append("left_contact")
    obs_names.append("right_contact")

    var all_match = True
    for i in range(8):
        var is_match = compare_scalar(
            obs_names[i], cpu_obs[i], gpu_obs[i], tolerance
        )
        if not is_match:
            all_match = False

    return all_match


# =============================================================================
# GPU Extraction Helpers
# =============================================================================


def extract_gpu_observation[
    BATCH: Int, STATE_SIZE: Int
](
    states_buf: DeviceBuffer[dtype],
    ctx: DeviceContext,
    env_idx: Int,
) raises -> Array[Scalar[dtype], 8]:
    """Extract observation from std.gpu state buffer."""
    var obs = Array[Scalar[dtype], 8](fill=Scalar[dtype](0))

    # Copy state to host
    var state_host = Array[Scalar[dtype], LLConstants.STATE_SIZE_VAL](
        fill=Scalar[dtype](0)
    )
    var src_offset = env_idx * STATE_SIZE

    # Create a small buffer for the observation portion
    var obs_host = ctx.enqueue_create_host_buffer[dtype](8)
    var obs_buf = ctx.enqueue_create_buffer[dtype](8)

    # Copy observation portion from states buffer
    @always_inline
    def copy_obs_kernel(
        dst: Tile[dtype, 8],
        src: ImmTile[dtype, BATCH, STATE_SIZE],
        env_arg: Int64,
    ):
        # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
        # a fixed-width `Int64` and re-binds the original name here.
        var env = Int(env_arg)
        for i in range(8):
            dst[i] = src[env, LLConstants.OBS_OFFSET + i]

    var dst_tensor = tile[8](obs_buf.unsafe_ptr())
    var src_tensor = tile[BATCH, STATE_SIZE](states_buf.unsafe_ptr())

    ctx.enqueue_function[copy_obs_kernel, copy_obs_kernel](
        dst_tensor, src_tensor, Int64(env_idx), grid_dim=(1,), block_dim=(1,)
    )
    ctx.enqueue_copy(obs_host, obs_buf)
    ctx.synchronize()

    for i in range(8):
        obs[i] = obs_host[i]

    return obs^


def extract_gpu_body_state[
    BATCH: Int, STATE_SIZE: Int
](
    states_buf: DeviceBuffer[dtype],
    ctx: DeviceContext,
    env_idx: Int,
    body_idx: Int,
) raises -> Array[Scalar[dtype], 6]:
    """Extract body state (x, y, angle, vx, vy, omega) from std.gpu buffer."""
    var state = Array[Scalar[dtype], 6](fill=Scalar[dtype](0))

    var state_host = ctx.enqueue_create_host_buffer[dtype](6)
    var state_buf = ctx.enqueue_create_buffer[dtype](6)

    @always_inline
    def copy_body_kernel(
        dst: Tile[dtype, 6],
        src: ImmTile[dtype, BATCH, STATE_SIZE],
        env_arg: Int64,
        body_arg: Int64,
    ):
        # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
        # a fixed-width `Int64` and re-binds the original name here.
        var env = Int(env_arg)
        var body = Int(body_arg)
        var body_off = LLConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
        dst[0] = src[env, body_off + IDX_X]
        dst[1] = src[env, body_off + IDX_Y]
        dst[2] = src[env, body_off + IDX_ANGLE]
        dst[3] = src[env, body_off + IDX_VX]
        dst[4] = src[env, body_off + IDX_VY]
        dst[5] = src[env, body_off + IDX_OMEGA]

    var dst_tensor = tile[6](state_buf.unsafe_ptr())
    var src_tensor = tile[BATCH, STATE_SIZE](states_buf.unsafe_ptr())

    ctx.enqueue_function[copy_body_kernel, copy_body_kernel](
        dst_tensor,
        src_tensor,
        Int64(env_idx),
        Int64(body_idx),
        grid_dim=(1,),
        block_dim=(1,),
    )
    ctx.enqueue_copy(state_host, state_buf)
    ctx.synchronize()

    for i in range(6):
        state[i] = state_host[i]

    return state^


def extract_gpu_metadata[
    BATCH: Int, STATE_SIZE: Int
](
    states_buf: DeviceBuffer[dtype],
    ctx: DeviceContext,
    env_idx: Int,
) raises -> Array[Scalar[dtype], 4]:
    """Extract metadata (step_count, total_reward, prev_shaping, done) from std.gpu.
    """
    var meta = Array[Scalar[dtype], 4](fill=Scalar[dtype](0))

    var meta_host = ctx.enqueue_create_host_buffer[dtype](4)
    var meta_buf = ctx.enqueue_create_buffer[dtype](4)

    @always_inline
    def copy_meta_kernel(
        dst: Tile[dtype, 4],
        src: ImmTile[dtype, BATCH, STATE_SIZE],
        env_arg: Int64,
    ):
        # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
        # a fixed-width `Int64` and re-binds the original name here.
        var env = Int(env_arg)
        for i in range(4):
            dst[i] = src[env, LLConstants.METADATA_OFFSET + i]

    var dst_tensor = tile[4](meta_buf.unsafe_ptr())
    var src_tensor = tile[BATCH, STATE_SIZE](states_buf.unsafe_ptr())

    ctx.enqueue_function[copy_meta_kernel, copy_meta_kernel](
        dst_tensor, src_tensor, Int64(env_idx), grid_dim=(1,), block_dim=(1,)
    )
    ctx.enqueue_copy(meta_host, meta_buf)
    ctx.synchronize()

    for i in range(4):
        meta[i] = meta_host[i]

    return meta^


# =============================================================================
# Main Comparison Tests
# =============================================================================


def test_reset_comparison(ctx: DeviceContext) raises -> Bool:
    """Test that CPU and GPU reset produce identical initial states."""
    print_header("TEST 1: Reset Comparison (Same Seed)")

    comptime N_ENVS = 1
    comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM = LLConstants.OBS_DIM_VAL

    # Use same seed for both
    var test_seed: UInt64 = 12345

    # CPU environment
    var cpu_env = LunarLander[dtype](seed=test_seed)

    # GPU state buffer
    var gpu_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)

    # Reset GPU with same seed
    LunarLander[dtype].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, gpu_states, test_seed
    )
    ctx.synchronize()

    # Compare observations
    print("\n[Observation Comparison]")
    var cpu_obs = cpu_env.get_observation(0)
    var gpu_obs = extract_gpu_observation[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )
    var obs_match = compare_observation(cpu_obs, gpu_obs, tolerance=1e-3)

    # Compare body states
    print("\n[Lander Body State]")
    var cpu_x = Scalar[dtype](cpu_env.physics.get_body_x(0, 0))
    var cpu_y = Scalar[dtype](cpu_env.physics.get_body_y(0, 0))
    var cpu_angle = Scalar[dtype](cpu_env.physics.get_body_angle(0, 0))
    var cpu_vx = Scalar[dtype](cpu_env.physics.get_body_vx(0, 0))
    var cpu_vy = Scalar[dtype](cpu_env.physics.get_body_vy(0, 0))
    var cpu_omega = Scalar[dtype](cpu_env.physics.get_body_omega(0, 0))

    var gpu_body = extract_gpu_body_state[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0, 0
    )

    var body_match = True
    body_match = compare_scalar("x", cpu_x, gpu_body[0], 1e-3) and body_match
    body_match = compare_scalar("y", cpu_y, gpu_body[1], 1e-3) and body_match
    body_match = (
        compare_scalar("angle", cpu_angle, gpu_body[2], 1e-3) and body_match
    )
    body_match = compare_scalar("vx", cpu_vx, gpu_body[3], 1e-3) and body_match
    body_match = compare_scalar("vy", cpu_vy, gpu_body[4], 1e-3) and body_match
    body_match = (
        compare_scalar("omega", cpu_omega, gpu_body[5], 1e-3) and body_match
    )

    # Compare metadata (prev_shaping especially)
    print("\n[Metadata]")
    var gpu_meta = extract_gpu_metadata[N_ENVS, STATE_SIZE](gpu_states, ctx, 0)
    _ = compare_scalar(
        "step_count", Scalar[dtype](0), gpu_meta[LLConstants.META_STEP_COUNT]
    )
    _ = compare_scalar(
        "total_reward",
        Scalar[dtype](0),
        gpu_meta[LLConstants.META_TOTAL_REWARD],
    )
    var cpu_shaping = cpu_env.prev_shaping
    _ = compare_scalar(
        "prev_shaping",
        cpu_shaping,
        gpu_meta[LLConstants.META_PREV_SHAPING],
        1e-2,
    )
    _ = compare_scalar(
        "done", Scalar[dtype](0), gpu_meta[LLConstants.META_DONE]
    )

    var all_match = obs_match and body_match
    print("\n[RESET TEST RESULT]:", "PASS" if all_match else "FAIL")
    return all_match


def test_step_comparison(ctx: DeviceContext) raises -> Bool:
    """Test that CPU and GPU step produce identical results."""
    print_header("TEST 2: Step Comparison (Action Sequence)")

    comptime N_ENVS = 1
    comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM = LLConstants.OBS_DIM_VAL
    comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL

    var test_seed: UInt64 = 42

    # CPU environment
    var cpu_env = LunarLander[dtype](seed=test_seed)

    # GPU buffers
    var gpu_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var gpu_actions = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var gpu_rewards = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_dones = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_terminated = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_obs = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)

    # Host buffers for actions
    var actions_host = ctx.enqueue_create_host_buffer[dtype](ACTION_DIM)
    var rewards_host = ctx.enqueue_create_host_buffer[dtype](1)
    var dones_host = ctx.enqueue_create_host_buffer[dtype](1)

    # Reset both
    LunarLander[dtype].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, gpu_states, test_seed
    )
    ctx.synchronize()

    # Test action sequence: noop, main engine, left, right, main engine
    var test_actions = List[List[Scalar[dtype]]]()

    # Action 0: noop (main=0, side=0)
    var a0 = List[Scalar[dtype]]()
    a0.append(Scalar[dtype](-1.0))  # main: -1 -> 0 after remap
    a0.append(Scalar[dtype](0.0))  # side: no thrust
    test_actions.append(a0^)

    # Action 1: main engine full (main=1, side=0)
    var a1 = List[Scalar[dtype]]()
    a1.append(Scalar[dtype](1.0))  # main: 1 -> 1 after remap
    a1.append(Scalar[dtype](0.0))  # side: no thrust
    test_actions.append(a1^)

    # Action 2: left thruster (main=0, side=-1)
    var a2 = List[Scalar[dtype]]()
    a2.append(Scalar[dtype](-1.0))  # main off
    a2.append(Scalar[dtype](-1.0))  # side: left engine
    test_actions.append(a2^)

    # Action 3: right thruster (main=0, side=1)
    var a3 = List[Scalar[dtype]]()
    a3.append(Scalar[dtype](-1.0))  # main off
    a3.append(Scalar[dtype](1.0))  # side: right engine
    test_actions.append(a3^)

    # Action 4: main + right (main=1, side=0.8)
    var a4 = List[Scalar[dtype]]()
    a4.append(Scalar[dtype](0.5))  # main: 0.75
    a4.append(Scalar[dtype](0.8))  # side: right engine
    test_actions.append(a4^)

    var all_match = True
    var tolerance: Float64 = (
        1e-2  # Slightly larger tolerance for accumulated errors
    )

    for step in range(test_actions.byte_length()):
        print("\n--- Step", step + 1, "---")

        var action = test_actions[step].copy()
        print("  Action: main=", action[0], ", side=", action[1])

        # CPU step
        var cpu_result = cpu_env.step_continuous_vec[dtype](action)
        var cpu_obs_list = cpu_result[0].copy()
        var cpu_reward = cpu_result[1]
        var cpu_done = cpu_result[2]

        # GPU step - copy action to device
        for i in range(ACTION_DIM):
            actions_host[i] = test_actions[step].copy()[i]
        ctx.enqueue_copy(gpu_actions, actions_host)

        LunarLander[dtype].step_kernel_gpu[
            N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            gpu_states,
            gpu_actions,
            gpu_rewards,
            gpu_dones,
            gpu_terminated,
            gpu_obs,
            UInt64(step),
        )
        ctx.synchronize()

        # Extract GPU results
        ctx.enqueue_copy(rewards_host, gpu_rewards)
        ctx.enqueue_copy(dones_host, gpu_dones)
        ctx.synchronize()

        var gpu_reward = rewards_host[0]
        var gpu_done = dones_host[0]

        # Compare reward and done
        print("\n  [Reward & Done]")
        var reward_match = compare_scalar(
            "reward", cpu_reward, gpu_reward, tolerance
        )
        var done_match = compare_scalar(
            "done",
            Scalar[dtype](1.0) if cpu_done else Scalar[dtype](0.0),
            gpu_done,
        )

        # Compare observations
        print("\n  [Observation]")
        var cpu_obs = Array[Scalar[dtype], 8](fill=Scalar[dtype](0))
        for i in range(8):
            cpu_obs[i] = Scalar[dtype](cpu_obs_list[i])
        var gpu_obs_arr = extract_gpu_observation[N_ENVS, STATE_SIZE](
            gpu_states, ctx, 0
        )
        var obs_match = compare_observation(cpu_obs, gpu_obs_arr, tolerance)

        # Compare body state
        print("\n  [Lander Body]")
        var cpu_body_x = Scalar[dtype](cpu_env.physics.get_body_x(0, 0))
        var cpu_body_y = Scalar[dtype](cpu_env.physics.get_body_y(0, 0))
        var cpu_body_vx = Scalar[dtype](cpu_env.physics.get_body_vx(0, 0))
        var cpu_body_vy = Scalar[dtype](cpu_env.physics.get_body_vy(0, 0))

        var gpu_body = extract_gpu_body_state[N_ENVS, STATE_SIZE](
            gpu_states, ctx, 0, 0
        )
        var body_match = True
        body_match = (
            compare_scalar("x", cpu_body_x, gpu_body[0], tolerance)
            and body_match
        )
        body_match = (
            compare_scalar("y", cpu_body_y, gpu_body[1], tolerance)
            and body_match
        )
        body_match = (
            compare_scalar("vx", cpu_body_vx, gpu_body[3], tolerance)
            and body_match
        )
        body_match = (
            compare_scalar("vy", cpu_body_vy, gpu_body[4], tolerance)
            and body_match
        )

        var step_match = (
            reward_match and done_match and obs_match and body_match
        )
        print(
            "\n  [Step",
            step + 1,
            "Result]:",
            "PASS" if step_match else "FAIL",
        )

        if not step_match:
            all_match = False

        # Early exit on done
        if cpu_done or gpu_done > 0.5:
            print("  Episode terminated, stopping comparison.")
            break

    print("\n[STEP TEST RESULT]:", "PASS" if all_match else "FAIL")
    return all_match


def test_flat_terrain_comparison(ctx: DeviceContext) raises -> Bool:
    """Test with forced flat terrain to isolate physics differences."""
    print_header("TEST 3: Flat Terrain Comparison")

    print("Note: This test uses the same seed which generates similar terrain.")
    print("The helipad region (center 5 chunks) is always flat at HELIPAD_Y.")
    print("Testing lander starting above flat helipad region...\n")

    comptime N_ENVS = 1
    comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM = LLConstants.OBS_DIM_VAL
    comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL

    # Use a specific seed
    var test_seed: UInt64 = 999

    # CPU environment
    var cpu_env = LunarLander[dtype](seed=test_seed)

    # GPU buffers
    var gpu_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var gpu_actions = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var gpu_rewards = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_dones = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_terminated = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_obs = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)

    var actions_host = ctx.enqueue_create_host_buffer[dtype](ACTION_DIM)

    # Reset both
    LunarLander[dtype].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, gpu_states, test_seed
    )
    ctx.synchronize()

    # Test a longer sequence of steps with constant action (free fall then main engine)
    var all_match = True
    var tolerance: Float64 = 5e-2  # Larger tolerance for longer sequences

    # Phase 1: Free fall (5 steps)
    print("[Phase 1: Free Fall - 5 steps]")
    for step in range(5):
        # Noop action
        actions_host[0] = Scalar[dtype](-1.0)  # main off
        actions_host[1] = Scalar[dtype](0.0)  # side off
        ctx.enqueue_copy(gpu_actions, actions_host)

        var action = List[Scalar[dtype]]()
        action.append(Scalar[dtype](-1.0))
        action.append(Scalar[dtype](0.0))

        # CPU step
        var cpu_result = cpu_env.step_continuous_vec[dtype](action)
        var cpu_reward = cpu_result[1]
        var cpu_done = cpu_result[2]

        # GPU step
        LunarLander[dtype].step_kernel_gpu[
            N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            gpu_states,
            gpu_actions,
            gpu_rewards,
            gpu_dones,
            gpu_terminated,
            gpu_obs,
            UInt64(step),
        )
        ctx.synchronize()

        # Compare key metrics
        var rewards_host_buf = ctx.enqueue_create_host_buffer[dtype](1)
        ctx.enqueue_copy(rewards_host_buf, gpu_rewards)
        ctx.synchronize()

        var gpu_reward = rewards_host_buf[0]
        var reward_diff = abs(Float64(cpu_reward - gpu_reward))

        if step == 0 or step == 4:  # Print first and last
            print(
                "  Step",
                step + 1,
                "| CPU reward:",
                fit(String(Float64(cpu_reward)), 8),
                "| GPU reward:",
                fit(String(Float64(gpu_reward)), 8),
                "| Diff:",
                fit(String(reward_diff), 8),
            )

        if cpu_done:
            print("  CPU episode terminated early")
            break

    # Phase 2: Main engine (10 steps)
    print("\n[Phase 2: Main Engine - 10 steps]")
    for step in range(10):
        # Main engine full
        actions_host[0] = Scalar[dtype](1.0)  # main full
        actions_host[1] = Scalar[dtype](0.0)  # side off
        ctx.enqueue_copy(gpu_actions, actions_host)

        var action = List[Scalar[dtype]]()
        action.append(Scalar[dtype](1.0))
        action.append(Scalar[dtype](0.0))

        # CPU step
        var cpu_result = cpu_env.step_continuous_vec[dtype](action)
        var cpu_done = cpu_result[2]

        # GPU step
        LunarLander[dtype].step_kernel_gpu[
            N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            gpu_states,
            gpu_actions,
            gpu_rewards,
            gpu_dones,
            gpu_terminated,
            gpu_obs,
            UInt64(step + 5),
        )
        ctx.synchronize()

        if cpu_done:
            print("  CPU episode terminated at step", step + 1)
            break

    # Final comparison
    print("\n[Final State Comparison]")
    var cpu_obs = cpu_env.get_observation(0)
    var gpu_obs_arr = extract_gpu_observation[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )
    var obs_match = compare_observation(cpu_obs, gpu_obs_arr, tolerance)

    print("\n[FLAT TERRAIN TEST RESULT]:", "PASS" if obs_match else "FAIL")
    return obs_match


def test_reward_accumulation(ctx: DeviceContext) raises -> Bool:
    """Test that reward calculation matches between CPU and GPU."""
    print_header("TEST 4: Reward Calculation Deep Dive")

    comptime N_ENVS = 1
    comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM = LLConstants.OBS_DIM_VAL
    comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL

    var test_seed: UInt64 = 77777

    # CPU environment
    var cpu_env = LunarLander[dtype](seed=test_seed)

    # GPU buffers
    var gpu_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var gpu_actions = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var gpu_rewards = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_dones = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_terminated = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_obs = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)

    var actions_host = ctx.enqueue_create_host_buffer[dtype](ACTION_DIM)
    var rewards_host = ctx.enqueue_create_host_buffer[dtype](1)

    # Reset both
    LunarLander[dtype].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, gpu_states, test_seed
    )
    ctx.synchronize()

    print("[Testing reward calculation over 20 steps]")
    print("Using alternating actions: main engine, noop, left, noop, right...")

    var cpu_total_reward: Float64 = 0.0
    var gpu_total_reward: Float64 = 0.0
    var max_reward_diff: Float64 = 0.0
    var all_match = True

    # Action sequence
    var action_sequence = List[Int]()
    for _ in range(4):
        action_sequence.append(0)  # main
        action_sequence.append(1)  # noop
        action_sequence.append(2)  # left
        action_sequence.append(1)  # noop
        action_sequence.append(3)  # right

    for step in range(20):
        var action_type = action_sequence[step]

        # Set action based on type
        var action = List[Scalar[dtype]]()
        if action_type == 0:  # main engine
            actions_host[0] = Scalar[dtype](1.0)
            actions_host[1] = Scalar[dtype](0.0)
            action.append(Scalar[dtype](1.0))
            action.append(Scalar[dtype](0.0))
        elif action_type == 1:  # noop
            actions_host[0] = Scalar[dtype](-1.0)
            actions_host[1] = Scalar[dtype](0.0)
            action.append(Scalar[dtype](-1.0))
            action.append(Scalar[dtype](0.0))
        elif action_type == 2:  # left
            actions_host[0] = Scalar[dtype](-1.0)
            actions_host[1] = Scalar[dtype](-1.0)
            action.append(Scalar[dtype](-1.0))
            action.append(Scalar[dtype](-1.0))
        else:  # right
            actions_host[0] = Scalar[dtype](-1.0)
            actions_host[1] = Scalar[dtype](1.0)
            action.append(Scalar[dtype](-1.0))
            action.append(Scalar[dtype](1.0))

        ctx.enqueue_copy(gpu_actions, actions_host)

        # CPU step
        var cpu_result = cpu_env.step_continuous_vec[dtype](action)
        var cpu_reward = Float64(cpu_result[1])
        var cpu_done = cpu_result[2]

        # GPU step
        LunarLander[dtype].step_kernel_gpu[
            N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            gpu_states,
            gpu_actions,
            gpu_rewards,
            gpu_dones,
            gpu_terminated,
            gpu_obs,
            UInt64(step),
        )
        ctx.synchronize()

        ctx.enqueue_copy(rewards_host, gpu_rewards)
        ctx.synchronize()
        var gpu_reward = Float64(rewards_host[0])

        cpu_total_reward += cpu_reward
        gpu_total_reward += gpu_reward

        var reward_diff = abs(cpu_reward - gpu_reward)
        if reward_diff > max_reward_diff:
            max_reward_diff = reward_diff

        # Print every 5 steps
        if (step + 1) % 5 == 0:
            print(
                "  Steps 1-",
                step + 1,
                "| CPU total:",
                fit(String(cpu_total_reward), 8),
                "| GPU total:",
                fit(String(gpu_total_reward), 8),
                "| Max diff:",
                fit(String(max_reward_diff), 8),
            )

        if cpu_done:
            print("  Episode terminated at step", step + 1)
            break

    var total_diff = abs(cpu_total_reward - gpu_total_reward)
    print("\n[Final Reward Summary]")
    print("  CPU total reward:", cpu_total_reward)
    print("  GPU total reward:", gpu_total_reward)
    print("  Total difference:", total_diff)
    print("  Max per-step diff:", max_reward_diff)

    # Allow some tolerance due to floating point and physics differences
    var acceptable_diff = abs(cpu_total_reward) * 0.1  # 10% tolerance
    if acceptable_diff < 1.0:
        acceptable_diff = 1.0
    all_match = total_diff < acceptable_diff

    print("\n[REWARD TEST RESULT]:", "PASS" if all_match else "FAIL")
    return all_match


def test_contact_detection(ctx: DeviceContext) raises -> Bool:
    """Test leg contact detection over a long trajectory.

    NOTE: This test may fail due to precision drift between CPU (float64 internal)
    and GPU (float32). Small differences accumulate over 80+ steps, causing
    trajectories to diverge. This is expected behavior, not a bug.
    The critical tests (Reset, Step, Gravity-Only) verify physics is identical
    when states match.
    """
    print_header("TEST 5: Contact Detection (Precision Drift Test)")

    comptime N_ENVS = 1
    comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM = LLConstants.OBS_DIM_VAL
    comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL

    var test_seed: UInt64 = 54321

    # CPU environment
    var cpu_env = LunarLander[dtype](seed=test_seed)

    # GPU buffers
    var gpu_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var gpu_actions = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var gpu_rewards = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_dones = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_terminated = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_obs = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)

    var actions_host = ctx.enqueue_create_host_buffer[dtype](ACTION_DIM)
    var dones_host = ctx.enqueue_create_host_buffer[dtype](1)

    # Reset both
    LunarLander[dtype].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, gpu_states, test_seed
    )
    ctx.synchronize()

    # Verify initial state match
    print("[Verifying initial state after reset]")
    var cpu_init_obs = cpu_env.get_observation(0)
    var gpu_init_obs = extract_gpu_observation[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )
    var init_match = True
    for i in range(8):
        var diff = abs_f64(Float64(cpu_init_obs[i] - gpu_init_obs[i]))
        if diff > 1e-4:
            init_match = False
    print("  Initial states match:", init_match)
    if not init_match:
        print(
            "  CPU obs:",
            cpu_init_obs[0],
            cpu_init_obs[1],
            cpu_init_obs[2],
            cpu_init_obs[3],
        )
        print(
            "  GPU obs:",
            gpu_init_obs[0],
            gpu_init_obs[1],
            gpu_init_obs[2],
            gpu_init_obs[3],
        )

    print("\n[Running until contact or timeout (100 steps)]")
    print("Using mostly noop to let gravity bring lander down...\n")

    var all_match = True

    for step in range(100):
        # Mostly noop, occasional main engine to control descent
        if step % 10 < 3:  # Main engine 30% of time
            actions_host[0] = Scalar[dtype](0.5)
        else:
            actions_host[0] = Scalar[dtype](-1.0)
        actions_host[1] = Scalar[dtype](0.0)
        ctx.enqueue_copy(gpu_actions, actions_host)

        var action = List[Scalar[dtype]]()
        action.append(actions_host[0])
        action.append(Scalar[dtype](0.0))

        # CPU step
        var cpu_result = cpu_env.step_continuous_vec[dtype](action)
        var cpu_obs = cpu_result[0].copy()
        var cpu_done = cpu_result[2]

        # GPU step
        LunarLander[dtype].step_kernel_gpu[
            N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            gpu_states,
            gpu_actions,
            gpu_rewards,
            gpu_dones,
            gpu_terminated,
            gpu_obs,
            UInt64(step),
        )
        ctx.synchronize()

        ctx.enqueue_copy(dones_host, gpu_dones)
        ctx.synchronize()

        # Extract observations to check contact
        var gpu_obs_arr = extract_gpu_observation[N_ENVS, STATE_SIZE](
            gpu_states, ctx, 0
        )

        var cpu_left_contact = cpu_obs[6] > 0.5
        var cpu_right_contact = cpu_obs[7] > 0.5
        var gpu_left_contact = gpu_obs_arr[6] > 0.5
        var gpu_right_contact = gpu_obs_arr[7] > 0.5

        # Show early steps to track divergence
        if step < 5:
            print(
                "  Step",
                step + 1,
                "| CPU vx:",
                cpu_obs[2],
                "GPU vx:",
                gpu_obs_arr[2],
                "| diff:",
                abs_f64(Float64(cpu_obs[2] - gpu_obs_arr[2])),
            )

        # Report contact events
        if (
            cpu_left_contact
            or gpu_left_contact
            or cpu_right_contact
            or gpu_right_contact
        ):
            print(
                "  Step",
                step + 1,
                "| Left: CPU=",
                cpu_left_contact,
                "GPU=",
                gpu_left_contact,
                "| Right: CPU=",
                cpu_right_contact,
                "GPU=",
                gpu_right_contact,
            )

            # Check if contact detection matches
            if cpu_left_contact != gpu_left_contact:
                print("    WARNING: Left contact mismatch!")
                all_match = False
            if cpu_right_contact != gpu_right_contact:
                print("    WARNING: Right contact mismatch!")
                all_match = False

        var gpu_done = dones_host[0] > 0.5
        if cpu_done or gpu_done:
            print("\n  Episode terminated at step", step + 1)
            print("    CPU done:", cpu_done, "| GPU done:", gpu_done)
            if cpu_done != gpu_done:
                print("    WARNING: Done flag mismatch!")
                # Debug: print velocities and termination conditions
                print("\n    [DEBUG - Termination Analysis]")
                var cpu_vx = cpu_obs[2]  # vx_norm
                var cpu_vy = cpu_obs[3]  # vy_norm
                var cpu_omega = cpu_obs[5]  # omega_norm
                var gpu_vx = gpu_obs_arr[2]
                var gpu_vy = gpu_obs_arr[3]
                var gpu_omega = gpu_obs_arr[5]
                print(
                    "    CPU: vx_norm=",
                    cpu_vx,
                    "vy_norm=",
                    cpu_vy,
                    "omega_norm=",
                    cpu_omega,
                )
                print(
                    "    GPU: vx_norm=",
                    gpu_vx,
                    "vy_norm=",
                    gpu_vy,
                    "omega_norm=",
                    gpu_omega,
                )

                # Compute actual speed (vx and vy are normalized by 5.0 in observation)
                var cpu_speed = (
                    sqrt(Float64(cpu_vx * cpu_vx + cpu_vy * cpu_vy)) * 5.0
                )
                var gpu_speed = (
                    sqrt(Float64(gpu_vx * gpu_vx + gpu_vy * gpu_vy)) * 5.0
                )
                # omega_norm is omega / 5.0, so actual omega = omega_norm * 5.0
                var cpu_omega_actual = abs_f64(Float64(cpu_omega)) * 5.0
                var gpu_omega_actual = abs_f64(Float64(gpu_omega)) * 5.0
                print(
                    "    CPU actual: speed=",
                    cpu_speed,
                    "omega=",
                    cpu_omega_actual,
                )
                print(
                    "    GPU actual: speed=",
                    gpu_speed,
                    "omega=",
                    gpu_omega_actual,
                )
                print("    is_at_rest threshold: speed < 0.01, omega < 0.01")
                print(
                    "    CPU is_at_rest:",
                    cpu_speed < 0.01 and cpu_omega_actual < 0.01,
                )
                print(
                    "    GPU is_at_rest:",
                    gpu_speed < 0.01 and gpu_omega_actual < 0.01,
                )

                # Check x_norm bounds
                var cpu_x_norm = cpu_obs[0]
                var gpu_x_norm = gpu_obs_arr[0]
                print(
                    "    CPU x_norm=",
                    cpu_x_norm,
                    "(out of bounds if >= 1 or <= -1)",
                )
                print("    GPU x_norm=", gpu_x_norm)

                # Check lander body contact (crash condition)
                print("\n    [Crash Detection Analysis]")
                print("    If neither is_at_rest nor x_norm out of bounds,")
                print(
                    "    then CPU must be detecting LANDER BODY CONTACT (crash)"
                )
                print(
                    "    This means lander body (not legs) touched ground on"
                    " CPU but not GPU"
                )

                # Get lander body position
                var cpu_body = cpu_env.physics.get_body_y(0, 0)  # lander y
                var cpu_angle = cpu_env.physics.get_body_angle(0, 0)
                print("    CPU lander y=", cpu_body, "angle=", cpu_angle)

                all_match = False
            break

    print("\n[CONTACT TEST RESULT]:", "PASS" if all_match else "FAIL")
    return all_match


def test_deterministic_physics(ctx: DeviceContext) raises -> Bool:
    """Test physics with manually set identical initial conditions.

    This test bypasses RNG differences by:
    1. Resetting both CPU and GPU normally
    2. Manually overwriting GPU state to match CPU exactly
    3. Stepping with identical actions
    4. Comparing physics results
    """
    print_header("TEST 6: Deterministic Physics (Forced Identical State)")

    comptime N_ENVS = 1
    comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM = LLConstants.OBS_DIM_VAL
    comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL

    var test_seed: UInt64 = 42

    # CPU environment - this will be our "source of truth"
    var cpu_env = LunarLander[dtype](seed=test_seed)

    # GPU buffers
    var gpu_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var gpu_actions = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var gpu_rewards = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_dones = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_terminated = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_obs = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)

    var actions_host = ctx.enqueue_create_host_buffer[dtype](ACTION_DIM)
    var rewards_host = ctx.enqueue_create_host_buffer[dtype](1)
    var dones_host = ctx.enqueue_create_host_buffer[dtype](1)

    # Reset GPU first (we'll overwrite it)
    LunarLander[dtype].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, gpu_states, test_seed
    )
    ctx.synchronize()

    # Now manually copy CPU state to GPU to ensure identical starting point
    print("[Copying CPU state to GPU for identical starting conditions]")

    # Create a host buffer to build the GPU state
    var state_host = ctx.enqueue_create_host_buffer[dtype](STATE_SIZE)

    # Copy body states from CPU to host buffer
    for body in range(3):
        var body_off = LLConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
        state_host[body_off + IDX_X] = Scalar[dtype](
            cpu_env.physics.get_body_x(0, body)
        )
        state_host[body_off + IDX_Y] = Scalar[dtype](
            cpu_env.physics.get_body_y(0, body)
        )
        state_host[body_off + IDX_ANGLE] = Scalar[dtype](
            cpu_env.physics.get_body_angle(0, body)
        )
        state_host[body_off + IDX_VX] = Scalar[dtype](
            cpu_env.physics.get_body_vx(0, body)
        )
        state_host[body_off + IDX_VY] = Scalar[dtype](
            cpu_env.physics.get_body_vy(0, body)
        )
        state_host[body_off + IDX_OMEGA] = Scalar[dtype](
            cpu_env.physics.get_body_omega(0, body)
        )
        # Mass properties
        if body == 0:
            state_host[body_off + 9] = Scalar[dtype](LLConstants.LANDER_MASS)
            state_host[body_off + 10] = Scalar[dtype](
                1.0 / LLConstants.LANDER_MASS
            )
            state_host[body_off + 11] = Scalar[dtype](
                1.0 / LLConstants.LANDER_INERTIA
            )
        else:
            state_host[body_off + 9] = Scalar[dtype](LLConstants.LEG_MASS)
            state_host[body_off + 10] = Scalar[dtype](
                1.0 / LLConstants.LEG_MASS
            )
            state_host[body_off + 11] = Scalar[dtype](
                1.0 / LLConstants.LEG_INERTIA
            )
        state_host[body_off + 12] = Scalar[dtype](body)  # shape index

    # Copy observations
    var cpu_obs = cpu_env.get_observation(0)
    for i in range(8):
        state_host[LLConstants.OBS_OFFSET + i] = cpu_obs[i]

    # Copy metadata
    state_host[
        LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
    ] = Scalar[dtype](0)
    state_host[
        LLConstants.METADATA_OFFSET + LLConstants.META_TOTAL_REWARD
    ] = Scalar[dtype](0)
    state_host[
        LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
    ] = cpu_env.prev_shaping
    state_host[LLConstants.METADATA_OFFSET + LLConstants.META_DONE] = Scalar[
        dtype
    ](0)

    # Copy terrain edges from CPU
    var n_edges = LLConstants.TERRAIN_CHUNKS - 1
    state_host[LLConstants.EDGE_COUNT_OFFSET] = Scalar[dtype](n_edges)

    var x_spacing = LLConstants.W_UNITS / Float64(
        LLConstants.TERRAIN_CHUNKS - 1
    )
    for edge in range(n_edges):
        var x0 = Float64(edge) * x_spacing
        var x1 = Float64(edge + 1) * x_spacing
        var y0 = Float64(cpu_env.terrain_heights[edge])
        var y1 = Float64(cpu_env.terrain_heights[edge + 1])

        # Compute edge normal
        var dx = x1 - x0
        var dy = y1 - y0
        var length = sqrt(dx * dx + dy * dy)
        var nx = -dy / length
        var ny = dx / length
        if ny < 0:
            nx = -nx
            ny = -ny

        var edge_off = LLConstants.EDGES_OFFSET + edge * 6
        state_host[edge_off + 0] = Scalar[dtype](x0)
        state_host[edge_off + 1] = Scalar[dtype](y0)
        state_host[edge_off + 2] = Scalar[dtype](x1)
        state_host[edge_off + 3] = Scalar[dtype](y1)
        state_host[edge_off + 4] = Scalar[dtype](nx)
        state_host[edge_off + 5] = Scalar[dtype](ny)

    # Copy joints (2 revolute joints)
    state_host[LLConstants.JOINT_COUNT_OFFSET] = Scalar[dtype](2)
    # Joint data would need to match - for now we rely on GPU reset having set this up

    # Clear forces
    for body in range(3):
        var force_off = LLConstants.FORCES_OFFSET + body * 3
        state_host[force_off + 0] = Scalar[dtype](0)
        state_host[force_off + 1] = Scalar[dtype](0)
        state_host[force_off + 2] = Scalar[dtype](0)

    # Copy host buffer to GPU (only the portions we set)
    # We need a kernel to selectively copy
    @always_inline
    def copy_state_kernel(
        dst: Tile[dtype, N_ENVS, STATE_SIZE],
        src_bodies: ImmTile[dtype, 3, BODY_STATE_SIZE],
        src_obs: ImmTile[dtype, 8],
        src_meta: ImmTile[dtype, 4],
        prev_shaping: Scalar[dtype],
    ):
        # Copy bodies
        for body in range(3):
            var body_off = LLConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
            for i in range(BODY_STATE_SIZE):
                dst[0, body_off + i] = src_bodies[body, i]

        # Copy observations
        for i in range(8):
            dst[0, LLConstants.OBS_OFFSET + i] = src_obs[i]

        # Copy metadata
        dst[0, LLConstants.METADATA_OFFSET + 0] = Scalar[dtype](0)  # step_count
        dst[0, LLConstants.METADATA_OFFSET + 1] = Scalar[dtype](
            0
        )  # total_reward
        dst[0, LLConstants.METADATA_OFFSET + 2] = prev_shaping
        dst[0, LLConstants.METADATA_OFFSET + 3] = Scalar[dtype](0)  # done

    # Prepare source tensors
    var bodies_host = ctx.enqueue_create_host_buffer[dtype](3 * BODY_STATE_SIZE)
    var obs_host_buf = ctx.enqueue_create_host_buffer[dtype](8)
    var meta_host = ctx.enqueue_create_host_buffer[dtype](4)

    for body in range(3):
        for i in range(BODY_STATE_SIZE):
            var body_off = LLConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
            bodies_host[body * BODY_STATE_SIZE + i] = state_host[body_off + i]

    for i in range(8):
        obs_host_buf[i] = state_host[LLConstants.OBS_OFFSET + i]

    meta_host[0] = Scalar[dtype](0)
    meta_host[1] = Scalar[dtype](0)
    meta_host[2] = cpu_env.prev_shaping
    meta_host[3] = Scalar[dtype](0)

    var bodies_buf = ctx.enqueue_create_buffer[dtype](3 * BODY_STATE_SIZE)
    var obs_buf_small = ctx.enqueue_create_buffer[dtype](8)
    var meta_buf = ctx.enqueue_create_buffer[dtype](4)

    ctx.enqueue_copy(bodies_buf, bodies_host)
    ctx.enqueue_copy(obs_buf_small, obs_host_buf)
    ctx.enqueue_copy(meta_buf, meta_host)

    var dst_tensor = tile[N_ENVS, STATE_SIZE](gpu_states.unsafe_ptr())
    var src_bodies_tensor = tile[3, BODY_STATE_SIZE](bodies_buf.unsafe_ptr())
    var src_obs_tensor = tile[8](obs_buf_small.unsafe_ptr())
    var src_meta_tensor = tile[4](meta_buf.unsafe_ptr())

    ctx.enqueue_function[copy_state_kernel, copy_state_kernel](
        dst_tensor,
        src_bodies_tensor,
        src_obs_tensor,
        src_meta_tensor,
        cpu_env.prev_shaping,
        grid_dim=(1,),
        block_dim=(1,),
    )
    ctx.synchronize()

    # Verify initial state matches
    print("\n[Verifying initial state after copy]")
    var gpu_obs_init = extract_gpu_observation[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )
    var init_match = compare_observation(cpu_obs, gpu_obs_init, tolerance=1e-5)

    if not init_match:
        print("  WARNING: Initial state copy failed!")
        return False

    print("  Initial states match!")

    # Now step both with identical actions and compare
    print("\n[Stepping with identical actions - 10 steps]")

    var all_match = True
    var tolerance: Float64 = 1e-3  # Tight tolerance for physics consistency

    # Test sequence: noop, main, noop, left, noop, right, main, noop, main, noop
    var action_types = Array[Int, 10](0, 1, 0, 2, 0, 3, 1, 0, 1, 0)

    for step in range(10):
        var action_type = action_types[step]

        # Set action
        var action = List[Scalar[dtype]]()
        if action_type == 0:  # noop
            actions_host[0] = Scalar[dtype](-1.0)
            actions_host[1] = Scalar[dtype](0.0)
            action.append(Scalar[dtype](-1.0))
            action.append(Scalar[dtype](0.0))
        elif action_type == 1:  # main engine
            actions_host[0] = Scalar[dtype](1.0)
            actions_host[1] = Scalar[dtype](0.0)
            action.append(Scalar[dtype](1.0))
            action.append(Scalar[dtype](0.0))
        elif action_type == 2:  # left
            actions_host[0] = Scalar[dtype](-1.0)
            actions_host[1] = Scalar[dtype](-1.0)
            action.append(Scalar[dtype](-1.0))
            action.append(Scalar[dtype](-1.0))
        else:  # right
            actions_host[0] = Scalar[dtype](-1.0)
            actions_host[1] = Scalar[dtype](1.0)
            action.append(Scalar[dtype](-1.0))
            action.append(Scalar[dtype](1.0))

        ctx.enqueue_copy(gpu_actions, actions_host)

        # CPU step
        var cpu_result = cpu_env.step_continuous_vec[dtype](action)
        var cpu_obs_step = cpu_result[0].copy()
        var cpu_reward = cpu_result[1]
        var cpu_done = cpu_result[2]

        # GPU step
        LunarLander[dtype].step_kernel_gpu[
            N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            gpu_states,
            gpu_actions,
            gpu_rewards,
            gpu_dones,
            gpu_terminated,
            gpu_obs,
            UInt64(step),
        )
        ctx.synchronize()

        ctx.enqueue_copy(rewards_host, gpu_rewards)
        ctx.enqueue_copy(dones_host, gpu_dones)
        ctx.synchronize()

        var gpu_reward = rewards_host[0]
        var gpu_done = dones_host[0] > 0.5

        # Extract GPU observation
        var gpu_obs_step = extract_gpu_observation[N_ENVS, STATE_SIZE](
            gpu_states, ctx, 0
        )

        # Compare
        var reward_diff = abs_f64(Float64(cpu_reward - gpu_reward))

        # Get body states for detailed comparison
        var cpu_x = Scalar[dtype](cpu_env.physics.get_body_x(0, 0))
        var cpu_y = Scalar[dtype](cpu_env.physics.get_body_y(0, 0))
        var cpu_vx = Scalar[dtype](cpu_env.physics.get_body_vx(0, 0))
        var cpu_vy = Scalar[dtype](cpu_env.physics.get_body_vy(0, 0))

        var gpu_body = extract_gpu_body_state[N_ENVS, STATE_SIZE](
            gpu_states, ctx, 0, 0
        )

        var pos_diff = sqrt(
            Float64(
                (cpu_x - gpu_body[0]) * (cpu_x - gpu_body[0])
                + (cpu_y - gpu_body[1]) * (cpu_y - gpu_body[1])
            )
        )
        var vel_diff = sqrt(
            Float64(
                (cpu_vx - gpu_body[3]) * (cpu_vx - gpu_body[3])
                + (cpu_vy - gpu_body[4]) * (cpu_vy - gpu_body[4])
            )
        )

        var step_match = (
            reward_diff < tolerance
            and pos_diff < tolerance
            and vel_diff < tolerance
        )

        var action_name: String
        if action_type == 0:
            action_name = "noop"
        elif action_type == 1:
            action_name = "main"
        elif action_type == 2:
            action_name = "left"
        else:
            action_name = "right"

        print(
            "  Step",
            step + 1,
            "(",
            action_name,
            ")",
            "| Reward diff:",
            fit(String(reward_diff), 8),
            "| Pos diff:",
            fit(String(pos_diff), 8),
            "| Vel diff:",
            fit(String(vel_diff), 8),
            "|",
            "OK" if step_match else "MISMATCH",
        )

        if not step_match:
            all_match = False
            # Print detailed comparison
            print(
                "    CPU: x=", cpu_x, "y=", cpu_y, "vx=", cpu_vx, "vy=", cpu_vy
            )
            print(
                "    GPU: x=",
                gpu_body[0],
                "y=",
                gpu_body[1],
                "vx=",
                gpu_body[3],
                "vy=",
                gpu_body[4],
            )

        if cpu_done or gpu_done:
            print("  Episode terminated at step", step + 1)
            if cpu_done != gpu_done:
                print(
                    "    WARNING: Done flag mismatch! CPU:",
                    cpu_done,
                    "GPU:",
                    gpu_done,
                )
                all_match = False
            break

    print(
        "\n[DETERMINISTIC PHYSICS TEST RESULT]:",
        "PASS" if all_match else "FAIL",
    )
    return all_match


def test_gravity_only(ctx: DeviceContext) raises -> Bool:
    """Test pure physics (gravity only, no engines) with identical states.

    This test isolates the physics integration from engine dispersion RNG.
    Uses only noop actions so no engine forces are applied.
    """
    print_header("TEST 7: Gravity-Only Physics (No Engine RNG)")

    comptime N_ENVS = 1
    comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM = LLConstants.OBS_DIM_VAL
    comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL

    var test_seed: UInt64 = 42

    # CPU environment
    var cpu_env = LunarLander[dtype](seed=test_seed)

    # GPU buffers
    var gpu_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var gpu_actions = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var gpu_rewards = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_dones = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_terminated = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_obs = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)

    var actions_host = ctx.enqueue_create_host_buffer[dtype](ACTION_DIM)
    var rewards_host = ctx.enqueue_create_host_buffer[dtype](1)

    # Reset GPU
    LunarLander[dtype].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, gpu_states, test_seed
    )
    ctx.synchronize()

    # Copy CPU state to GPU (same as deterministic test)
    var bodies_host = ctx.enqueue_create_host_buffer[dtype](3 * BODY_STATE_SIZE)
    var obs_host_buf = ctx.enqueue_create_host_buffer[dtype](8)

    for body in range(3):
        var body_off = body * BODY_STATE_SIZE
        bodies_host[body_off + IDX_X] = Scalar[dtype](
            cpu_env.physics.get_body_x(0, body)
        )
        bodies_host[body_off + IDX_Y] = Scalar[dtype](
            cpu_env.physics.get_body_y(0, body)
        )
        bodies_host[body_off + IDX_ANGLE] = Scalar[dtype](
            cpu_env.physics.get_body_angle(0, body)
        )
        bodies_host[body_off + IDX_VX] = Scalar[dtype](
            cpu_env.physics.get_body_vx(0, body)
        )
        bodies_host[body_off + IDX_VY] = Scalar[dtype](
            cpu_env.physics.get_body_vy(0, body)
        )
        bodies_host[body_off + IDX_OMEGA] = Scalar[dtype](
            cpu_env.physics.get_body_omega(0, body)
        )
        if body == 0:
            bodies_host[body_off + 9] = Scalar[dtype](LLConstants.LANDER_MASS)
            bodies_host[body_off + 10] = Scalar[dtype](
                1.0 / LLConstants.LANDER_MASS
            )
            bodies_host[body_off + 11] = Scalar[dtype](
                1.0 / LLConstants.LANDER_INERTIA
            )
        else:
            bodies_host[body_off + 9] = Scalar[dtype](LLConstants.LEG_MASS)
            bodies_host[body_off + 10] = Scalar[dtype](
                1.0 / LLConstants.LEG_MASS
            )
            bodies_host[body_off + 11] = Scalar[dtype](
                1.0 / LLConstants.LEG_INERTIA
            )
        bodies_host[body_off + 12] = Scalar[dtype](body)

    var cpu_obs = cpu_env.get_observation(0)
    for i in range(8):
        obs_host_buf[i] = cpu_obs[i]

    var bodies_buf = ctx.enqueue_create_buffer[dtype](3 * BODY_STATE_SIZE)
    var obs_buf_small = ctx.enqueue_create_buffer[dtype](8)
    ctx.enqueue_copy(bodies_buf, bodies_host)
    ctx.enqueue_copy(obs_buf_small, obs_host_buf)

    @always_inline
    def copy_state_kernel2(
        dst: Tile[dtype, N_ENVS, STATE_SIZE],
        src_bodies: ImmTile[dtype, 3, BODY_STATE_SIZE],
        src_obs: ImmTile[dtype, 8],
        prev_shaping: Scalar[dtype],
    ):
        for body in range(3):
            var body_off = LLConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
            for i in range(BODY_STATE_SIZE):
                dst[0, body_off + i] = src_bodies[body, i]
        for i in range(8):
            dst[0, LLConstants.OBS_OFFSET + i] = src_obs[i]
        dst[0, LLConstants.METADATA_OFFSET + 0] = Scalar[dtype](0)
        dst[0, LLConstants.METADATA_OFFSET + 1] = Scalar[dtype](0)
        dst[0, LLConstants.METADATA_OFFSET + 2] = prev_shaping
        dst[0, LLConstants.METADATA_OFFSET + 3] = Scalar[dtype](0)

    var dst_tensor = tile[N_ENVS, STATE_SIZE](gpu_states.unsafe_ptr())
    var src_bodies_tensor = tile[3, BODY_STATE_SIZE](bodies_buf.unsafe_ptr())
    var src_obs_tensor = tile[8](obs_buf_small.unsafe_ptr())

    ctx.enqueue_function[copy_state_kernel2, copy_state_kernel2](
        dst_tensor,
        src_bodies_tensor,
        src_obs_tensor,
        cpu_env.prev_shaping,
        grid_dim=(1,),
        block_dim=(1,),
    )
    ctx.synchronize()

    # Verify initial state
    var gpu_obs_init = extract_gpu_observation[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )
    print("[Initial state verification]")
    var init_match = True
    for i in range(8):
        var diff = abs_f64(Float64(cpu_obs[i] - gpu_obs_init[i]))
        if diff > 1e-5:
            init_match = False
    print("  Initial states match:", init_match)

    if not init_match:
        return False

    # Step with noop only (no engines = no RNG used)
    print("\n[Stepping with noop actions only - 20 steps]")
    print("  (No engines = no dispersion RNG, pure physics test)")

    # Set noop action
    actions_host[0] = Scalar[dtype](-1.0)  # main off
    actions_host[1] = Scalar[dtype](0.0)  # side off
    ctx.enqueue_copy(gpu_actions, actions_host)

    var all_match = True
    var tolerance: Float64 = 1e-4  # Very tight tolerance

    for step in range(20):
        # CPU step with noop
        var action = List[Scalar[dtype]]()
        action.append(Scalar[dtype](-1.0))
        action.append(Scalar[dtype](0.0))

        var cpu_result = cpu_env.step_continuous_vec[dtype](action)
        var cpu_reward = cpu_result[1]
        var cpu_done = cpu_result[2]

        # GPU step
        LunarLander[dtype].step_kernel_gpu[
            N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            gpu_states,
            gpu_actions,
            gpu_rewards,
            gpu_dones,
            gpu_terminated,
            gpu_obs,
            UInt64(step),
        )
        ctx.synchronize()

        ctx.enqueue_copy(rewards_host, gpu_rewards)
        ctx.synchronize()
        var gpu_reward = rewards_host[0]

        # Compare body states
        var cpu_x = Scalar[dtype](cpu_env.physics.get_body_x(0, 0))
        var cpu_y = Scalar[dtype](cpu_env.physics.get_body_y(0, 0))
        var cpu_vx = Scalar[dtype](cpu_env.physics.get_body_vx(0, 0))
        var cpu_vy = Scalar[dtype](cpu_env.physics.get_body_vy(0, 0))

        var gpu_body = extract_gpu_body_state[N_ENVS, STATE_SIZE](
            gpu_states, ctx, 0, 0
        )

        var pos_diff = sqrt(
            Float64(
                (cpu_x - gpu_body[0]) * (cpu_x - gpu_body[0])
                + (cpu_y - gpu_body[1]) * (cpu_y - gpu_body[1])
            )
        )
        var vel_diff = sqrt(
            Float64(
                (cpu_vx - gpu_body[3]) * (cpu_vx - gpu_body[3])
                + (cpu_vy - gpu_body[4]) * (cpu_vy - gpu_body[4])
            )
        )
        var reward_diff = abs_f64(Float64(cpu_reward - gpu_reward))

        var step_match = pos_diff < tolerance and vel_diff < tolerance

        if (step + 1) % 5 == 0 or not step_match:
            print(
                "  Step",
                step + 1,
                "| Pos diff:",
                fit(String(pos_diff), 10),
                "| Vel diff:",
                fit(String(vel_diff), 10),
                "| Reward diff:",
                fit(String(reward_diff), 10),
                "|",
                "OK" if step_match else "MISMATCH",
            )

        if not step_match:
            all_match = False
            print(
                "    CPU: x=", cpu_x, "y=", cpu_y, "vx=", cpu_vx, "vy=", cpu_vy
            )
            print(
                "    GPU: x=",
                gpu_body[0],
                "y=",
                gpu_body[1],
                "vx=",
                gpu_body[3],
                "vy=",
                gpu_body[4],
            )

        if cpu_done:
            print("  Episode terminated at step", step + 1)
            break

    print("\n[GRAVITY-ONLY TEST RESULT]:", "PASS" if all_match else "FAIL")

    if all_match:
        print(
            "\n  CONCLUSION: Pure physics (gravity, integration) is CONSISTENT!"
        )
        print(
            "  The divergence in other tests is due to ENGINE DISPERSION RNG."
        )
    else:
        print(
            "\n  CONCLUSION: Physics integration differs between CPU and GPU."
        )

    return all_match


def test_reward_deep_dive(ctx: DeviceContext) raises -> Bool:
    """Deep dive into reward calculation differences.

    This test examines:
    1. Shaping calculation (should be identical if observations match)
    2. Observation values used for shaping
    3. Prev_shaping storage and retrieval
    """
    print_header("TEST 8: Reward Calculation Deep Dive")

    comptime N_ENVS = 1
    comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM = LLConstants.OBS_DIM_VAL
    comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL

    var test_seed: UInt64 = 42

    # CPU environment
    var cpu_env = LunarLander[dtype](seed=test_seed)

    # GPU buffers
    var gpu_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var gpu_actions = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var gpu_rewards = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_dones = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_terminated = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_obs = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)

    var actions_host = ctx.enqueue_create_host_buffer[dtype](ACTION_DIM)
    var rewards_host = ctx.enqueue_create_host_buffer[dtype](1)

    # Reset GPU
    LunarLander[dtype].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, gpu_states, test_seed
    )
    ctx.synchronize()

    # Get initial observations and shaping from both
    var cpu_obs_init = cpu_env.get_observation(0)
    var gpu_obs_init = extract_gpu_observation[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )

    print("[Initial State Analysis]")
    print("\n  CPU Observation:")
    for i in range(8):
        print("    obs[", i, "] =", cpu_obs_init[i])

    print("\n  GPU Observation:")
    for i in range(8):
        print("    obs[", i, "] =", gpu_obs_init[i])

    # Compute shaping manually for both
    # Shaping formula from helpers.mojo:
    # shaping = -100*sqrt(x^2 + y^2) - 100*sqrt(vx^2 + vy^2) - 100*abs(angle) + 10*left + 10*right

    def compute_shaping_manual(obs: Array[Scalar[dtype], 8]) -> Float64:
        var x = Float64(obs[0])
        var y = Float64(obs[1])
        var vx = Float64(obs[2])
        var vy = Float64(obs[3])
        var angle = Float64(obs[4])
        var left = Float64(obs[6])
        var right = Float64(obs[7])

        var dist = sqrt(x * x + y * y)
        var speed = sqrt(vx * vx + vy * vy)
        var angle_abs = angle if angle >= 0 else -angle

        return (
            -100.0 * dist
            - 100.0 * speed
            - 100.0 * angle_abs
            + 10.0 * left
            + 10.0 * right
        )

    var cpu_shaping = compute_shaping_manual(cpu_obs_init)
    var gpu_shaping = compute_shaping_manual(gpu_obs_init)

    print("\n  CPU initial shaping (computed):", cpu_shaping)
    print("  CPU initial shaping (stored):", Float64(cpu_env.prev_shaping))
    print("  GPU initial shaping (computed):", gpu_shaping)

    # Get GPU stored shaping
    var gpu_meta = extract_gpu_metadata[N_ENVS, STATE_SIZE](gpu_states, ctx, 0)
    print(
        "  GPU initial shaping (stored):",
        Float64(gpu_meta[LLConstants.META_PREV_SHAPING]),
    )

    print("\n  Shaping diff (computed):", abs_f64(cpu_shaping - gpu_shaping))
    print(
        "  This diff comes from observation differences (different RNG at"
        " reset)"
    )

    # Now copy CPU state to GPU for identical starting point
    print("\n[Copying CPU state to GPU...]")

    var bodies_host = ctx.enqueue_create_host_buffer[dtype](3 * BODY_STATE_SIZE)
    var obs_host_buf = ctx.enqueue_create_host_buffer[dtype](8)

    for body in range(3):
        var body_off = body * BODY_STATE_SIZE
        bodies_host[body_off + IDX_X] = Scalar[dtype](
            cpu_env.physics.get_body_x(0, body)
        )
        bodies_host[body_off + IDX_Y] = Scalar[dtype](
            cpu_env.physics.get_body_y(0, body)
        )
        bodies_host[body_off + IDX_ANGLE] = Scalar[dtype](
            cpu_env.physics.get_body_angle(0, body)
        )
        bodies_host[body_off + IDX_VX] = Scalar[dtype](
            cpu_env.physics.get_body_vx(0, body)
        )
        bodies_host[body_off + IDX_VY] = Scalar[dtype](
            cpu_env.physics.get_body_vy(0, body)
        )
        bodies_host[body_off + IDX_OMEGA] = Scalar[dtype](
            cpu_env.physics.get_body_omega(0, body)
        )
        if body == 0:
            bodies_host[body_off + 9] = Scalar[dtype](LLConstants.LANDER_MASS)
            bodies_host[body_off + 10] = Scalar[dtype](
                1.0 / LLConstants.LANDER_MASS
            )
            bodies_host[body_off + 11] = Scalar[dtype](
                1.0 / LLConstants.LANDER_INERTIA
            )
        else:
            bodies_host[body_off + 9] = Scalar[dtype](LLConstants.LEG_MASS)
            bodies_host[body_off + 10] = Scalar[dtype](
                1.0 / LLConstants.LEG_MASS
            )
            bodies_host[body_off + 11] = Scalar[dtype](
                1.0 / LLConstants.LEG_INERTIA
            )
        bodies_host[body_off + 12] = Scalar[dtype](body)

    for i in range(8):
        obs_host_buf[i] = cpu_obs_init[i]

    var bodies_buf = ctx.enqueue_create_buffer[dtype](3 * BODY_STATE_SIZE)
    var obs_buf_small = ctx.enqueue_create_buffer[dtype](8)
    ctx.enqueue_copy(bodies_buf, bodies_host)
    ctx.enqueue_copy(obs_buf_small, obs_host_buf)

    @always_inline
    def copy_full_state_kernel(
        dst: Tile[dtype, N_ENVS, STATE_SIZE],
        src_bodies: ImmTile[dtype, 3, BODY_STATE_SIZE],
        src_obs: ImmTile[dtype, 8],
        prev_shaping: Scalar[dtype],
    ):
        for body in range(3):
            var body_off = LLConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
            for i in range(BODY_STATE_SIZE):
                dst[0, body_off + i] = src_bodies[body, i]
        for i in range(8):
            dst[0, LLConstants.OBS_OFFSET + i] = src_obs[i]
        dst[0, LLConstants.METADATA_OFFSET + 0] = Scalar[dtype](0)
        dst[0, LLConstants.METADATA_OFFSET + 1] = Scalar[dtype](0)
        dst[0, LLConstants.METADATA_OFFSET + 2] = prev_shaping
        dst[0, LLConstants.METADATA_OFFSET + 3] = Scalar[dtype](0)

    var dst_tensor = tile[N_ENVS, STATE_SIZE](gpu_states.unsafe_ptr())
    var src_bodies_tensor = tile[3, BODY_STATE_SIZE](bodies_buf.unsafe_ptr())
    var src_obs_tensor = tile[8](obs_buf_small.unsafe_ptr())

    ctx.enqueue_function[copy_full_state_kernel, copy_full_state_kernel](
        dst_tensor,
        src_bodies_tensor,
        src_obs_tensor,
        cpu_env.prev_shaping,
        grid_dim=(1,),
        block_dim=(1,),
    )
    ctx.synchronize()

    # Verify copy
    var gpu_obs_after_copy = extract_gpu_observation[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )
    var gpu_meta_after_copy = extract_gpu_metadata[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )

    print(
        "  After copy - GPU prev_shaping:",
        Float64(gpu_meta_after_copy[LLConstants.META_PREV_SHAPING]),
    )
    print("  After copy - CPU prev_shaping:", Float64(cpu_env.prev_shaping))

    # Now step ONCE with noop and compare rewards in detail
    print("\n[Single Noop Step Analysis]")

    actions_host[0] = Scalar[dtype](-1.0)
    actions_host[1] = Scalar[dtype](0.0)
    ctx.enqueue_copy(gpu_actions, actions_host)

    # Get pre-step states
    var cpu_pre_x = Scalar[dtype](cpu_env.physics.get_body_x(0, 0))
    var cpu_pre_y = Scalar[dtype](cpu_env.physics.get_body_y(0, 0))
    var cpu_pre_vx = Scalar[dtype](cpu_env.physics.get_body_vx(0, 0))
    var cpu_pre_vy = Scalar[dtype](cpu_env.physics.get_body_vy(0, 0))

    print(
        "  Pre-step CPU: x=",
        cpu_pre_x,
        "y=",
        cpu_pre_y,
        "vx=",
        cpu_pre_vx,
        "vy=",
        cpu_pre_vy,
    )

    # CPU step
    var action = List[Scalar[dtype]]()
    action.append(Scalar[dtype](-1.0))
    action.append(Scalar[dtype](0.0))

    var cpu_result = cpu_env.step_continuous_vec[dtype](action)
    var cpu_obs_after = cpu_result[0].copy()
    var cpu_reward = cpu_result[1]

    # GPU step
    LunarLander[dtype].step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM](
        ctx,
        gpu_states,
        gpu_actions,
        gpu_rewards,
        gpu_dones,
        gpu_terminated,
        gpu_obs,
        UInt64(0),
    )
    ctx.synchronize()

    ctx.enqueue_copy(rewards_host, gpu_rewards)
    ctx.synchronize()
    var gpu_reward = rewards_host[0]

    # Get post-step states
    var cpu_post_x = Scalar[dtype](cpu_env.physics.get_body_x(0, 0))
    var cpu_post_y = Scalar[dtype](cpu_env.physics.get_body_y(0, 0))
    var cpu_post_vx = Scalar[dtype](cpu_env.physics.get_body_vx(0, 0))
    var cpu_post_vy = Scalar[dtype](cpu_env.physics.get_body_vy(0, 0))

    var gpu_body_after = extract_gpu_body_state[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0, 0
    )
    var gpu_obs_after = extract_gpu_observation[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )

    print(
        "  Post-step CPU: x=",
        cpu_post_x,
        "y=",
        cpu_post_y,
        "vx=",
        cpu_post_vx,
        "vy=",
        cpu_post_vy,
    )
    print(
        "  Post-step GPU: x=",
        gpu_body_after[0],
        "y=",
        gpu_body_after[1],
        "vx=",
        gpu_body_after[3],
        "vy=",
        gpu_body_after[4],
    )

    # Compute shaping for post-step observations
    var cpu_obs_arr = Array[Scalar[dtype], 8](fill=Scalar[dtype](0))
    for i in range(8):
        cpu_obs_arr[i] = Scalar[dtype](cpu_obs_after[i])

    var cpu_new_shaping = compute_shaping_manual(cpu_obs_arr)
    var gpu_new_shaping = compute_shaping_manual(gpu_obs_after)

    print("\n  CPU new_shaping:", cpu_new_shaping)
    print("  GPU new_shaping:", gpu_new_shaping)
    print("  Shaping diff:", abs_f64(cpu_new_shaping - gpu_new_shaping))

    print("\n  CPU reward:", Float64(cpu_reward))
    print("  GPU reward:", Float64(gpu_reward))
    print("  Reward diff:", abs_f64(Float64(cpu_reward - gpu_reward)))

    # Break down reward components
    var cpu_prev_shaping = Float64(
        cpu_env.prev_shaping
    )  # This was updated after step!
    # We need to get it from BEFORE the step - but we didn't save it
    # Actually the test above shows the issue already

    print("\n  ANALYSIS:")
    print("  If physics matches and observations match, reward should match.")
    print("  Any reward difference indicates:")
    print("  1. prev_shaping was stored differently")
    print("  2. new_shaping computed differently")
    print("  3. Fuel cost computed differently (0 for noop)")

    var all_match = abs_f64(Float64(cpu_reward - gpu_reward)) < 0.01
    print("\n[REWARD DEEP DIVE RESULT]:", "PASS" if all_match else "FAIL")
    return all_match


def test_synchronized_contact_detection(ctx: DeviceContext) raises -> Bool:
    """Test contact detection with synchronized states (no precision drift).

    This test:
    1. Resets CPU environment
    2. Copies CPU state to GPU for identical starting conditions
    3. Runs with NOOP actions only (no engine RNG) until contact
    4. Compares contact detection and done flags at each step
    """
    print_header("TEST 9: Synchronized Contact Detection (CRITICAL)")

    comptime N_ENVS = 1
    comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM = LLConstants.OBS_DIM_VAL
    comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL

    var test_seed: UInt64 = 99999

    # CPU environment
    var cpu_env = LunarLander[dtype](seed=test_seed)

    # GPU buffers
    var gpu_states = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var gpu_actions = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var gpu_rewards = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_dones = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_terminated = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var gpu_obs = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)

    var actions_host = ctx.enqueue_create_host_buffer[dtype](ACTION_DIM)
    var dones_host = ctx.enqueue_create_host_buffer[dtype](1)
    var rewards_host = ctx.enqueue_create_host_buffer[dtype](1)

    # Reset GPU first
    LunarLander[dtype].reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, gpu_states, test_seed
    )
    ctx.synchronize()

    # Copy CPU state to GPU for identical starting conditions
    print("[Copying CPU state to GPU for identical starting conditions]")

    var state_host = ctx.enqueue_create_host_buffer[dtype](STATE_SIZE)

    # Copy body states
    for body in range(3):
        var body_off = LLConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
        state_host[body_off + IDX_X] = Scalar[dtype](
            cpu_env.physics.get_body_x(0, body)
        )
        state_host[body_off + IDX_Y] = Scalar[dtype](
            cpu_env.physics.get_body_y(0, body)
        )
        state_host[body_off + IDX_ANGLE] = Scalar[dtype](
            cpu_env.physics.get_body_angle(0, body)
        )
        state_host[body_off + IDX_VX] = Scalar[dtype](
            cpu_env.physics.get_body_vx(0, body)
        )
        state_host[body_off + IDX_VY] = Scalar[dtype](
            cpu_env.physics.get_body_vy(0, body)
        )
        state_host[body_off + IDX_OMEGA] = Scalar[dtype](
            cpu_env.physics.get_body_omega(0, body)
        )
        if body == 0:
            state_host[body_off + 9] = Scalar[dtype](LLConstants.LANDER_MASS)
            state_host[body_off + 10] = Scalar[dtype](
                1.0 / LLConstants.LANDER_MASS
            )
            state_host[body_off + 11] = Scalar[dtype](
                1.0 / LLConstants.LANDER_INERTIA
            )
        else:
            state_host[body_off + 9] = Scalar[dtype](LLConstants.LEG_MASS)
            state_host[body_off + 10] = Scalar[dtype](
                1.0 / LLConstants.LEG_MASS
            )
            state_host[body_off + 11] = Scalar[dtype](
                1.0 / LLConstants.LEG_INERTIA
            )
        state_host[body_off + 12] = Scalar[dtype](body)

    # Copy observations
    var cpu_obs_init = cpu_env.get_observation(0)
    for i in range(8):
        state_host[LLConstants.OBS_OFFSET + i] = cpu_obs_init[i]

    # Copy metadata
    state_host[
        LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
    ] = Scalar[dtype](0)
    state_host[
        LLConstants.METADATA_OFFSET + LLConstants.META_TOTAL_REWARD
    ] = Scalar[dtype](0)
    state_host[
        LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
    ] = cpu_env.prev_shaping
    state_host[LLConstants.METADATA_OFFSET + LLConstants.META_DONE] = Scalar[
        dtype
    ](0)

    # Copy terrain edges
    var n_edges = LLConstants.TERRAIN_CHUNKS - 1
    state_host[LLConstants.EDGE_COUNT_OFFSET] = Scalar[dtype](n_edges)

    var chunk_width = LLConstants.W_UNITS / Float64(
        LLConstants.TERRAIN_CHUNKS - 1
    )
    for edge in range(n_edges):
        var x0 = Float64(edge) * chunk_width
        var x1 = Float64(edge + 1) * chunk_width
        var y0 = Float64(cpu_env.terrain_heights[edge])
        var y1 = Float64(cpu_env.terrain_heights[edge + 1])
        var dx = x1 - x0
        var dy = y1 - y0
        var length = sqrt(dx * dx + dy * dy)
        var nx = -dy / length
        var ny = dx / length
        if ny < 0:
            nx = -nx
            ny = -ny
        var edge_off = LLConstants.EDGES_OFFSET + edge * 6
        state_host[edge_off + 0] = Scalar[dtype](x0)
        state_host[edge_off + 1] = Scalar[dtype](y0)
        state_host[edge_off + 2] = Scalar[dtype](x1)
        state_host[edge_off + 3] = Scalar[dtype](y1)
        state_host[edge_off + 4] = Scalar[dtype](nx)
        state_host[edge_off + 5] = Scalar[dtype](ny)

    # Copy joint data from CPU physics state
    var cpu_joints = cpu_env.physics.get_joints_tensor()
    var n_joints = cpu_env.physics.get_joint_count(0)
    state_host[LLConstants.JOINT_COUNT_OFFSET] = Scalar[dtype](n_joints)

    for j in range(n_joints):
        var joint_off = LLConstants.JOINTS_OFFSET + j * JOINT_DATA_SIZE
        for k in range(JOINT_DATA_SIZE):
            state_host[joint_off + k] = rebind[Scalar[dtype]](
                cpu_joints[0, j, k]
            )

    # Upload to GPU
    ctx.enqueue_copy(gpu_states, state_host)
    ctx.synchronize()

    # Verify initial state match
    var gpu_obs_init = extract_gpu_observation[N_ENVS, STATE_SIZE](
        gpu_states, ctx, 0
    )
    var init_match = True
    for i in range(8):
        if abs_f64(Float64(cpu_obs_init[i] - gpu_obs_init[i])) > 1e-4:
            init_match = False
    print("  Initial states match:", init_match)

    print("\n[Running with NOOP actions until contact (max 150 steps)]")
    print("  (No engines = no dispersion RNG, pure physics)")

    var all_match = True

    # Set noop action
    actions_host[0] = Scalar[dtype](-1.0)  # main engine off
    actions_host[1] = Scalar[dtype](0.0)  # side engine off
    ctx.enqueue_copy(gpu_actions, actions_host)

    var action = List[Scalar[dtype]]()
    action.append(Scalar[dtype](-1.0))
    action.append(Scalar[dtype](0.0))

    var cpu_total_reward: Float64 = 0.0
    var gpu_total_reward: Float64 = 0.0

    for step in range(150):
        # CPU step
        var cpu_result = cpu_env.step_continuous_vec[dtype](action)
        var cpu_obs = cpu_result[0].copy()
        var cpu_reward = cpu_result[1]
        var cpu_done = cpu_result[2]
        cpu_total_reward += Float64(cpu_reward)

        # GPU step
        LunarLander[dtype].step_kernel_gpu[
            N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            gpu_states,
            gpu_actions,
            gpu_rewards,
            gpu_dones,
            gpu_terminated,
            gpu_obs,
            UInt64(step),
        )
        ctx.synchronize()

        ctx.enqueue_copy(dones_host, gpu_dones)
        ctx.enqueue_copy(rewards_host, gpu_rewards)
        ctx.synchronize()

        var gpu_reward = rewards_host[0]
        gpu_total_reward += Float64(gpu_reward)

        var gpu_obs_arr = extract_gpu_observation[N_ENVS, STATE_SIZE](
            gpu_states, ctx, 0
        )

        # Track physics state divergence (omega is critical for shaping reward)
        var cpu_omega = Float64(cpu_env.physics.get_body_omega(0, 0))
        var gpu_lander_state = extract_gpu_body_state[N_ENVS, STATE_SIZE](
            gpu_states, ctx, 0, 0
        )
        var gpu_omega = Float64(gpu_lander_state[5])
        var omega_diff = abs_f64(cpu_omega - gpu_omega)

        # Report every 10 steps to track divergence growth
        if step % 10 == 0 or omega_diff > 0.1:
            print(
                "  Step",
                step + 1,
                "| CPU omega:",
                cpu_omega,
                "| GPU omega:",
                gpu_omega,
                "| diff:",
                omega_diff,
            )

        # Compare rewards at each step
        var reward_diff = abs_f64(Float64(cpu_reward) - Float64(gpu_reward))
        if reward_diff > 0.01:
            print("  Step", step + 1, "REWARD MISMATCH!")
            print("    CPU reward:", cpu_reward, "| GPU reward:", gpu_reward)
            print("    Diff:", reward_diff)
            all_match = False

        var cpu_left = cpu_obs[6] > 0.5
        var cpu_right = cpu_obs[7] > 0.5
        var gpu_left = gpu_obs_arr[6] > 0.5
        var gpu_right = gpu_obs_arr[7] > 0.5
        var gpu_done = dones_host[0] > 0.5

        # Report contact events
        if cpu_left or gpu_left or cpu_right or gpu_right:
            print(
                "  Step",
                step + 1,
                "| Left: CPU=",
                cpu_left,
                "GPU=",
                gpu_left,
                "| Right: CPU=",
                cpu_right,
                "GPU=",
                gpu_right,
            )
            if cpu_left != gpu_left or cpu_right != gpu_right:
                # Debug: print contact values and leg positions
                print("    CPU obs[6]=", cpu_obs[6], "obs[7]=", cpu_obs[7])
                print(
                    "    GPU obs[6]=", gpu_obs_arr[6], "obs[7]=", gpu_obs_arr[7]
                )

                # Get leg positions from CPU
                var cpu_left_leg_y = cpu_env.physics.get_body_y(0, 1)
                var cpu_right_leg_y = cpu_env.physics.get_body_y(0, 2)
                var cpu_left_leg_x = cpu_env.physics.get_body_x(0, 1)
                var cpu_right_leg_x = cpu_env.physics.get_body_x(0, 2)
                print(
                    "    CPU left_leg: x=", cpu_left_leg_x, "y=", cpu_left_leg_y
                )
                print(
                    "    CPU right_leg: x=",
                    cpu_right_leg_x,
                    "y=",
                    cpu_right_leg_y,
                )

                # Get leg positions from std.gpu
                var gpu_body_left = extract_gpu_body_state[N_ENVS, STATE_SIZE](
                    gpu_states, ctx, 0, 1
                )
                var gpu_body_right = extract_gpu_body_state[N_ENVS, STATE_SIZE](
                    gpu_states, ctx, 0, 2
                )
                print(
                    "    GPU left_leg: x=",
                    gpu_body_left[0],
                    "y=",
                    gpu_body_left[1],
                )
                print(
                    "    GPU right_leg: x=",
                    gpu_body_right[0],
                    "y=",
                    gpu_body_right[1],
                )
                print("    LEG_H=", LLConstants.LEG_H, "tolerance=0.01")

                # Also check main lander body
                var cpu_lander_y = cpu_env.physics.get_body_y(0, 0)
                var cpu_lander_x = cpu_env.physics.get_body_x(0, 0)
                var gpu_lander = extract_gpu_body_state[N_ENVS, STATE_SIZE](
                    gpu_states, ctx, 0, 0
                )
                print("    CPU lander: x=", cpu_lander_x, "y=", cpu_lander_y)
                print("    GPU lander: x=", gpu_lander[0], "y=", gpu_lander[1])
                print(
                    "    Lander diff: x=",
                    abs_f64(Float64(cpu_lander_x) - Float64(gpu_lander[0])),
                    "y=",
                    abs_f64(Float64(cpu_lander_y) - Float64(gpu_lander[1])),
                )

            if cpu_left != gpu_left:
                print("    MISMATCH: Left contact differs!")
                all_match = False
            if cpu_right != gpu_right:
                print("    MISMATCH: Right contact differs!")
                all_match = False

        # Check done flag
        if cpu_done or gpu_done:
            print("\n  Episode terminated at step", step + 1)
            print("    CPU done:", cpu_done, "| GPU done:", gpu_done)

            # Check termination reason
            var cpu_lander_contact = cpu_env._has_lander_body_contact()
            var cpu_vx = Float64(cpu_obs[2])
            var cpu_vy = Float64(cpu_obs[3])
            var cpu_speed = sqrt(cpu_vx * cpu_vx + cpu_vy * cpu_vy)
            var cpu_omega = Float64(cpu_env.physics.get_body_omega(0, 0))
            var cpu_both_legs = cpu_obs[6] > 0.5 and cpu_obs[7] > 0.5
            var cpu_is_at_rest = (
                cpu_speed < 0.01 and abs_f64(cpu_omega) < 0.01 and cpu_both_legs
            )

            print("    CPU termination:")
            print("      lander_contact (crash):", cpu_lander_contact)
            print("      speed:", cpu_speed, "omega:", cpu_omega)
            print(
                "      both_legs:",
                cpu_both_legs,
                "is_at_rest (landing):",
                cpu_is_at_rest,
            )
            print("      final reward:", cpu_reward)

            # GPU termination info
            var gpu_vx = Float64(gpu_obs_arr[2])
            var gpu_vy = Float64(gpu_obs_arr[3])
            var gpu_speed = sqrt(gpu_vx * gpu_vx + gpu_vy * gpu_vy)
            var gpu_lander = extract_gpu_body_state[N_ENVS, STATE_SIZE](
                gpu_states, ctx, 0, 0
            )
            var gpu_omega = Float64(gpu_lander[5])  # omega is at index 5
            var gpu_both_legs = gpu_obs_arr[6] > 0.5 and gpu_obs_arr[7] > 0.5
            var gpu_is_at_rest = (
                gpu_speed < 0.01 and abs_f64(gpu_omega) < 0.01 and gpu_both_legs
            )

            print("    GPU termination:")
            print("      speed:", gpu_speed, "omega:", gpu_omega)
            print(
                "      both_legs:",
                gpu_both_legs,
                "is_at_rest (landing):",
                gpu_is_at_rest,
            )
            print("      final reward:", gpu_reward)

            print("\n    TOTAL EPISODE REWARDS:")
            print("      CPU:", cpu_total_reward)
            print("      GPU:", gpu_total_reward)
            print("      Diff:", abs_f64(cpu_total_reward - gpu_total_reward))

            if cpu_done != gpu_done:
                print("    MISMATCH: Done flag differs!")
                all_match = False
            break

    print(
        "\n[SYNCHRONIZED CONTACT TEST RESULT]:", "PASS" if all_match else "FAIL"
    )
    return all_match


# =============================================================================
# Main Entry Point
# =============================================================================


def main() raises:
    print_header("LUNAR LANDER CPU vs GPU COMPARISON TEST SUITE")
    print("Testing LunarLander implementation consistency...")
    print("Using dtype:", dtype)
    print()

    var ctx = DeviceContext()

    var test_results = List[Bool]()

    # Run all tests
    test_results.append(test_reset_comparison(ctx))
    print()
    test_results.append(test_step_comparison(ctx))
    print()
    test_results.append(test_flat_terrain_comparison(ctx))
    print()
    test_results.append(test_reward_accumulation(ctx))
    print()
    test_results.append(test_contact_detection(ctx))
    print()
    test_results.append(test_deterministic_physics(ctx))
    print()
    test_results.append(test_gravity_only(ctx))
    print()
    test_results.append(test_reward_deep_dive(ctx))
    print()
    test_results.append(test_synchronized_contact_detection(ctx))

    # Summary
    print_header("TEST SUMMARY")

    var test_names = List[String]()
    test_names.append("Reset Comparison (CRITICAL)")
    test_names.append("Step Comparison (CRITICAL)")
    test_names.append("Flat Terrain (CRITICAL)")
    test_names.append("Reward Accumulation (CRITICAL)")
    test_names.append("Contact Detection (precision drift)")
    test_names.append("Deterministic Physics (CRITICAL)")
    test_names.append("Gravity-Only Physics (CRITICAL)")
    test_names.append("Reward Deep Dive (CRITICAL)")
    test_names.append("Synchronized Contact (CRITICAL)")

    var passed = 0
    var failed = 0

    for i in range(test_results.byte_length()):
        var status = "PASS" if test_results[i] else "FAIL"
        print("  ", test_names[i], ":", status)
        if test_results[i]:
            passed += 1
        else:
            failed += 1

    print()
    print("Passed:", passed, "/", test_results.byte_length())
    print("Failed:", failed, "/", test_results.byte_length())

    if failed > 0:
        print(
            "\nNote: 'precision drift' tests may fail due to float32/float64"
            " differences"
        )
        print("accumulating over many steps. This is expected behavior.")
        print(
            "CRITICAL tests verify that physics is identical when states match."
        )
    else:
        print("\nAll tests passed - CPU and GPU behavior appear consistent!")

from noeira.core.fmt import fit