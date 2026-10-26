"""Test GPU CartPole vectorized environment."""

from max.gpu.host import DeviceContext
from layout import LayoutTensor

from noeira.envs.cartpole_gpu import (
    NUM_ENVS,
    STATE_SIZE,
    BLOCKS_PER_GRID,
    THREADS_PER_BLOCK,
    dtype,
    state_layout,
    action_layout,
    reward_layout,
    done_layout,
    rng_layout,
    step_kernel,
    reset_kernel,
    reset_where_done_kernel,
)


def main() raises:
    print("Testing GPU CartPole with", NUM_ENVS, "parallel environments")
    print("Grid:", BLOCKS_PER_GRID, "blocks x", THREADS_PER_BLOCK, "threads")

    with DeviceContext() as ctx:
        # Allocate GPU buffers
        states_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * STATE_SIZE)
        actions_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
        rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
        dones_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
        rng_buf = ctx.enqueue_create_buffer[DType.uint32](NUM_ENVS)

        # Initialize buffers
        states_buf.enqueue_fill(0)
        actions_buf.enqueue_fill(0)
        rewards_buf.enqueue_fill(0)
        dones_buf.enqueue_fill(0)

        # Initialize RNG with unique seeds per environment
        with rng_buf.map_to_host() as rng_host:
            for i in range(NUM_ENVS):
                rng_host[i] = UInt32(i + 1)

        # Create LayoutTensors
        states = LayoutTensor[dtype, state_layout, MutAnyOrigin](states_buf)
        actions = LayoutTensor[DType.int32, action_layout, MutAnyOrigin](
            actions_buf
        )
        rewards = LayoutTensor[dtype, reward_layout, MutAnyOrigin](rewards_buf)
        dones = LayoutTensor[DType.int32, done_layout, MutAnyOrigin](dones_buf)
        rng_states = LayoutTensor[DType.uint32, rng_layout, MutAnyOrigin](
            rng_buf
        )

        # Reset all environments
        print("\n1. Resetting all environments...")
        ctx.enqueue_function[reset_kernel, reset_kernel](
            states,
            rng_states,
            grid_dim=(BLOCKS_PER_GRID,),
            block_dim=(THREADS_PER_BLOCK,),
        )
        ctx.synchronize()

        # Check initial states
        with states_buf.map_to_host() as states_host:
            print("First 5 environment states after reset:")
            for env in range(5):
                x = states_host[env * STATE_SIZE + 0]
                x_dot = states_host[env * STATE_SIZE + 1]
                theta = states_host[env * STATE_SIZE + 2]
                theta_dot = states_host[env * STATE_SIZE + 3]
                print(
                    "  Env",
                    env,
                    ": x=",
                    x,
                    ", x_dot=",
                    x_dot,
                    ", theta=",
                    theta,
                    ", theta_dot=",
                    theta_dot,
                )

        # Set random actions (alternating 0 and 1)
        with actions_buf.map_to_host() as actions_host:
            for i in range(NUM_ENVS):
                actions_host[i] = Int32(i % 2)

        # Run 100 steps
        print("\n2. Running 100 steps...")
        num_steps = 100
        total_reward = Float32(0)
        total_dones = 0

        for step in range(num_steps):
            # Step all environments
            ctx.enqueue_function[step_kernel, step_kernel](
                states,
                actions,
                rewards,
                dones,
                grid_dim=(BLOCKS_PER_GRID,),
                block_dim=(THREADS_PER_BLOCK,),
            )

            # Reset terminated environments
            ctx.enqueue_function[
                reset_where_done_kernel, reset_where_done_kernel
            ](
                states,
                dones,
                rng_states,
                grid_dim=(BLOCKS_PER_GRID,),
                block_dim=(THREADS_PER_BLOCK,),
            )

            ctx.synchronize()

            # Accumulate rewards and count dones
            with rewards_buf.map_to_host() as rewards_host:
                for i in range(NUM_ENVS):
                    total_reward += rewards_host[i]

            with dones_buf.map_to_host() as dones_host:
                for i in range(NUM_ENVS):
                    if dones_host[i] == 1:
                        total_dones += 1

        print("Total reward across all envs:", total_reward)
        print("Total episode terminations:", total_dones)
        print(
            "Average reward per env per step:",
            total_reward / (num_steps * NUM_ENVS),
        )

        # Check final states
        with states_buf.map_to_host() as states_host:
            print("\nFinal states (first 5 environments):")
            for env in range(5):
                x = states_host[env * STATE_SIZE + 0]
                x_dot = states_host[env * STATE_SIZE + 1]
                theta = states_host[env * STATE_SIZE + 2]
                theta_dot = states_host[env * STATE_SIZE + 3]
                print(
                    "  Env",
                    env,
                    ": x=",
                    x,
                    ", x_dot=",
                    x_dot,
                    ", theta=",
                    theta,
                    ", theta_dot=",
                    theta_dot,
                )

        print("\nGPU CartPole test completed successfully!")
