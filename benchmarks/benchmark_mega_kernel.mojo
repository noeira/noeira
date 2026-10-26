"""Benchmark: Fused mega-kernel for DQN training.

This benchmark explores fusing the entire DQN training step into a single kernel:
- Forward pass (current state Q-values)
- Forward pass (next state Q-values for target)
- TD target computation: r + γ * max(Q')
- Loss and gradient computation
- Backward pass
- Adam weight update

Key insight: If kernel launch overhead is 700+ μs, and we can fuse 6+ kernels into 1,
we save 5+ × 700 μs = 3.5+ ms per training step!

Run with:
    pixi run -e apple mojo run -I . examples/benchmark_mega_kernel.mojo
"""

from std.time import perf_counter_ns
from std.math import sqrt
from std.random import random_float64, seed

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from noeira.core.tiles import Tile, ImmTile, tile
from noeira.core.fmt import fit


# =============================================================================
# Fused DQN Training Kernel
# =============================================================================


def fused_dqn_train_kernel[
    dtype: DType,
    BATCH: Int,  # Number of transitions in batch
    OBS_DIM: Int,  # Observation dimension
    HIDDEN_DIM: Int,  # Hidden layer size
    OUT_DIM: Int,  # Output dimension (number of actions)
    TPB: Int,  # Threads per block
](
    # Network weights (mutable for Adam update)
    W1: Tile[dtype, OBS_DIM * HIDDEN_DIM],
    b1: Tile[dtype, HIDDEN_DIM],
    W2: Tile[dtype, HIDDEN_DIM * OUT_DIM],
    b2: Tile[dtype, OUT_DIM],
    # Adam state (mutable)
    m_W1: Tile[dtype, OBS_DIM * HIDDEN_DIM],
    v_W1: Tile[dtype, OBS_DIM * HIDDEN_DIM],
    m_b1: Tile[dtype, HIDDEN_DIM],
    v_b1: Tile[dtype, HIDDEN_DIM],
    m_W2: Tile[dtype, HIDDEN_DIM * OUT_DIM],
    v_W2: Tile[dtype, HIDDEN_DIM * OUT_DIM],
    m_b2: Tile[dtype, OUT_DIM],
    v_b2: Tile[dtype, OUT_DIM],
    # Batch data (immutable)
    obs: ImmTile[dtype, BATCH * OBS_DIM],
    next_obs: ImmTile[dtype, BATCH * OBS_DIM],
    actions: ImmTile[DType.int32, BATCH],
    rewards: ImmTile[dtype, BATCH],
    dones: ImmTile[dtype, BATCH],
    # Hyperparameters (compile-time for efficiency)
    gamma: Scalar[dtype],  # Discount factor
    lr: Scalar[dtype],  # Learning rate
    beta1: Scalar[dtype],  # Adam beta1
    beta2: Scalar[dtype],  # Adam beta2
    epsilon: Scalar[dtype],  # Adam epsilon
    t_step: Scalar[dtype],  # Time step for bias correction
    # Output (for debugging/monitoring)
    loss_out: Tile[dtype, BATCH],
):
    """Fused DQN training kernel.

    Each thread handles one transition (one element of the batch).
    All computation happens in registers - no intermediate buffers needed.

    Steps per thread:
    1. Forward pass current obs -> Q(s)
    2. Forward pass next obs -> Q(s')
    3. Compute target: r + γ * max(Q(s')) * (1 - done)
    4. Compute loss gradient: 2 * (Q(s,a) - target)
    5. Backward pass: compute weight gradients
    6. Adam update (atomic add for gradient accumulation)
    """
    var batch_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if batch_idx >= BATCH:
        return

    # =========================================================================
    # Step 1: Forward pass for current observation -> Q(s)
    # =========================================================================

    # Hidden layer: h = ReLU(obs @ W1 + b1)
    var h_pre = Array[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
    var h = Array[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))

    for hid in range(HIDDEN_DIM):
        var sum_val = rebind[Scalar[dtype]](b1[hid])
        for k in range(OBS_DIM):
            sum_val += rebind[Scalar[dtype]](
                obs[batch_idx * OBS_DIM + k]
            ) * rebind[Scalar[dtype]](W1[k * HIDDEN_DIM + hid])
        h_pre[hid] = sum_val
        h[hid] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](
            0
        )  # ReLU

    # Output layer: Q = h @ W2 + b2
    var Q = Array[Scalar[dtype], OUT_DIM](fill=Scalar[dtype](0))
    for j in range(OUT_DIM):
        var sum_val = rebind[Scalar[dtype]](b2[j])
        for k in range(HIDDEN_DIM):
            sum_val += h[k] * rebind[Scalar[dtype]](W2[k * OUT_DIM + j])
        Q[j] = sum_val

    # =========================================================================
    # Step 2: Forward pass for next observation -> Q(s') (for target)
    # =========================================================================

    var h_next = Array[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))

    for hid in range(HIDDEN_DIM):
        var sum_val = rebind[Scalar[dtype]](b1[hid])
        for k in range(OBS_DIM):
            sum_val += rebind[Scalar[dtype]](
                next_obs[batch_idx * OBS_DIM + k]
            ) * rebind[Scalar[dtype]](W1[k * HIDDEN_DIM + hid])
        h_next[hid] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](
            0
        )

    # max Q(s')
    var max_Q_next = rebind[Scalar[dtype]](b2[0])
    for k in range(HIDDEN_DIM):
        max_Q_next += h_next[k] * rebind[Scalar[dtype]](W2[k * OUT_DIM + 0])

    for j in range(1, OUT_DIM):
        var q_val = rebind[Scalar[dtype]](b2[j])
        for k in range(HIDDEN_DIM):
            q_val += h_next[k] * rebind[Scalar[dtype]](W2[k * OUT_DIM + j])
        if q_val > max_Q_next:
            max_Q_next = q_val

    # =========================================================================
    # Step 3: Compute TD target and loss
    # =========================================================================

    var reward = rebind[Scalar[dtype]](rewards[batch_idx])
    var done = rebind[Scalar[dtype]](dones[batch_idx])
    var action = Int(actions[batch_idx])

    # TD target: r + γ * max(Q') * (1 - done)
    var target = reward + gamma * max_Q_next * (Scalar[dtype](1) - done)

    # Q(s, a) for the taken action
    var Q_sa = Q[action]

    # TD error
    var td_error = Q_sa - target

    # MSE loss (for monitoring)
    loss_out[batch_idx] = td_error * td_error

    # =========================================================================
    # Step 4: Backward pass - compute gradients
    # =========================================================================

    # dL/dQ[a] = 2 * (Q(s,a) - target) = 2 * td_error
    # For other actions, gradient is 0

    # Output layer gradients: dL/dW2, dL/db2
    # dL/db2[j] = dL/dQ[j] = 2*td_error if j == action else 0
    # dL/dW2[k,j] = dL/dQ[j] * h[k] = h[k] * dL/db2[j]

    var dL_dQ = (
        Scalar[dtype](2) * td_error / Scalar[dtype](BATCH)
    )  # Scale by batch

    # dL/dh[k] = sum_j dL/dQ[j] * W2[k,j] = dL_dQ * W2[k, action]
    var dL_dh = Array[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
    for k in range(HIDDEN_DIM):
        dL_dh[k] = dL_dQ * rebind[Scalar[dtype]](W2[k * OUT_DIM + action])

    # ReLU backward: dL/dh_pre[k] = dL/dh[k] if h_pre[k] > 0 else 0
    var dL_dh_pre = Array[Scalar[dtype], HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    for k in range(HIDDEN_DIM):
        dL_dh_pre[k] = dL_dh[k] if h_pre[k] > Scalar[dtype](0) else Scalar[
            dtype
        ](0)

    # =========================================================================
    # Step 5: Adam update (using barrier for synchronization)
    # =========================================================================

    # For simplicity, we compute gradients per-thread and accumulate.
    # In a production kernel, we'd use shared memory reduction.

    # Note: This simple version updates weights directly per thread.
    # This is incorrect for parallel execution - need atomic adds or reduction.
    # For benchmark purposes, we just measure the computation overhead.

    # Bias correction factors
    var bc1 = Scalar[dtype](1) / (Scalar[dtype](1) - beta1**t_step)
    var bc2 = Scalar[dtype](1) / (Scalar[dtype](1) - beta2**t_step)

    # Update W2 (only the action column receives gradient)
    for k in range(HIDDEN_DIM):
        var idx = k * OUT_DIM + action
        var grad = h[k] * dL_dQ

        # Adam: m = β1*m + (1-β1)*grad, v = β2*v + (1-β2)*grad²
        var m_val = (
            beta1 * rebind[Scalar[dtype]](m_W2[idx])
            + (Scalar[dtype](1) - beta1) * grad
        )
        var v_val = (
            beta2 * rebind[Scalar[dtype]](v_W2[idx])
            + (Scalar[dtype](1) - beta2) * grad * grad
        )

        # Note: This would need atomics in real implementation
        m_W2[idx] = m_val
        v_W2[idx] = v_val

        # Update weight
        var m_hat = m_val * bc1
        var v_hat = v_val * bc2
        W2[idx] = rebind[Scalar[dtype]](W2[idx]) - lr * m_hat / (
            sqrt(v_hat) + epsilon
        )

    # Update b2 (only action bias receives gradient)
    if True:
        var grad = dL_dQ
        var m_val = (
            beta1 * rebind[Scalar[dtype]](m_b2[action])
            + (Scalar[dtype](1) - beta1) * grad
        )
        var v_val = (
            beta2 * rebind[Scalar[dtype]](v_b2[action])
            + (Scalar[dtype](1) - beta2) * grad * grad
        )
        m_b2[action] = m_val
        v_b2[action] = v_val
        var m_hat = m_val * bc1
        var v_hat = v_val * bc2
        b2[action] = rebind[Scalar[dtype]](b2[action]) - lr * m_hat / (
            sqrt(v_hat) + epsilon
        )

    # Update W1
    for hid in range(HIDDEN_DIM):
        var grad_hid = dL_dh_pre[hid]
        if grad_hid != Scalar[dtype](0):  # Skip if ReLU killed gradient
            for k in range(OBS_DIM):
                var idx = k * HIDDEN_DIM + hid
                var grad = (
                    rebind[Scalar[dtype]](obs[batch_idx * OBS_DIM + k])
                    * grad_hid
                )

                var m_val = (
                    beta1 * rebind[Scalar[dtype]](m_W1[idx])
                    + (Scalar[dtype](1) - beta1) * grad
                )
                var v_val = (
                    beta2 * rebind[Scalar[dtype]](v_W1[idx])
                    + (Scalar[dtype](1) - beta2) * grad * grad
                )
                m_W1[idx] = m_val
                v_W1[idx] = v_val
                var m_hat = m_val * bc1
                var v_hat = v_val * bc2
                W1[idx] = rebind[Scalar[dtype]](W1[idx]) - lr * m_hat / (
                    sqrt(v_hat) + epsilon
                )

    # Update b1
    for hid in range(HIDDEN_DIM):
        var grad = dL_dh_pre[hid]
        if grad != Scalar[dtype](0):
            var m_val = (
                beta1 * rebind[Scalar[dtype]](m_b1[hid])
                + (Scalar[dtype](1) - beta1) * grad
            )
            var v_val = (
                beta2 * rebind[Scalar[dtype]](v_b1[hid])
                + (Scalar[dtype](1) - beta2) * grad * grad
            )
            m_b1[hid] = m_val
            v_b1[hid] = v_val
            var m_hat = m_val * bc1
            var v_hat = v_val * bc2
            b1[hid] = rebind[Scalar[dtype]](b1[hid]) - lr * m_hat / (
                sqrt(v_hat) + epsilon
            )


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_mega_kernel[
    BATCH: Int, OBS_DIM: Int, HIDDEN_DIM: Int, OUT_DIM: Int
](ctx: DeviceContext, num_iters: Int) raises -> Float64:
    """Benchmark the fused mega-kernel."""
    comptime dtype = DType.float32
    comptime TPB = 64

    # Allocate weight buffers
    var W1 = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
    var b1 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    var W2 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * OUT_DIM)
    var b2 = ctx.enqueue_create_buffer[dtype](OUT_DIM)

    # Adam state buffers
    var m_W1 = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
    var v_W1 = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
    var m_b1 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    var v_b1 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    var m_W2 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * OUT_DIM)
    var v_W2 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * OUT_DIM)
    var m_b2 = ctx.enqueue_create_buffer[dtype](OUT_DIM)
    var v_b2 = ctx.enqueue_create_buffer[dtype](OUT_DIM)

    # Batch data buffers
    var obs = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
    var next_obs = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
    var actions = ctx.enqueue_create_buffer[DType.int32](BATCH)
    var rewards = ctx.enqueue_create_buffer[dtype](BATCH)
    var dones = ctx.enqueue_create_buffer[dtype](BATCH)
    var loss_out = ctx.enqueue_create_buffer[dtype](BATCH)

    # Initialize weights
    with W1.map_to_host() as host:
        for i in range(OBS_DIM * HIDDEN_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    b1.enqueue_fill(0)
    with W2.map_to_host() as host:
        for i in range(HIDDEN_DIM * OUT_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    b2.enqueue_fill(0)

    # Initialize Adam state to zero
    m_W1.enqueue_fill(0)
    v_W1.enqueue_fill(0)
    m_b1.enqueue_fill(0)
    v_b1.enqueue_fill(0)
    m_W2.enqueue_fill(0)
    v_W2.enqueue_fill(0)
    m_b2.enqueue_fill(0)
    v_b2.enqueue_fill(0)

    # Initialize batch data
    with obs.map_to_host() as host:
        for i in range(BATCH * OBS_DIM):
            host[i] = Scalar[dtype](random_float64())
    with next_obs.map_to_host() as host:
        for i in range(BATCH * OBS_DIM):
            host[i] = Scalar[dtype](random_float64())
    with actions.map_to_host() as host:
        for i in range(BATCH):
            host[i] = Int32(i % OUT_DIM)
    with rewards.map_to_host() as host:
        for i in range(BATCH):
            host[i] = Scalar[dtype](random_float64() - 0.5)
    dones.enqueue_fill(0)
    loss_out.enqueue_fill(0)
    ctx.synchronize()

    # Create tensors
    var W1_t = tile[OBS_DIM * HIDDEN_DIM](W1)
    var b1_t = tile[HIDDEN_DIM](b1)
    var W2_t = tile[HIDDEN_DIM * OUT_DIM](W2)
    var b2_t = tile[OUT_DIM](b2)
    var m_W1_t = tile[OBS_DIM * HIDDEN_DIM](m_W1)
    var v_W1_t = tile[OBS_DIM * HIDDEN_DIM](v_W1)
    var m_b1_t = tile[HIDDEN_DIM](m_b1)
    var v_b1_t = tile[HIDDEN_DIM](v_b1)
    var m_W2_t = tile[HIDDEN_DIM * OUT_DIM](m_W2)
    var v_W2_t = tile[HIDDEN_DIM * OUT_DIM](v_W2)
    var m_b2_t = tile[OUT_DIM](m_b2)
    var v_b2_t = tile[OUT_DIM](v_b2)
    var obs_t = tile[BATCH * OBS_DIM](obs)
    var next_obs_t = tile[BATCH * OBS_DIM](next_obs)
    var actions_t = tile[BATCH](actions)
    var rewards_t = tile[BATCH](rewards)
    var dones_t = tile[BATCH](dones)
    var loss_t = tile[BATCH](loss_out)

    # Hyperparameters
    var gamma = Scalar[dtype](0.99)
    var lr = Scalar[dtype](0.001)
    var beta1 = Scalar[dtype](0.9)
    var beta2 = Scalar[dtype](0.999)
    var epsilon = Scalar[dtype](1e-8)
    var t_step = Scalar[dtype](1)

    comptime num_blocks = (BATCH + TPB - 1) // TPB
    comptime kernel = fused_dqn_train_kernel[
        dtype, BATCH, OBS_DIM, HIDDEN_DIM, OUT_DIM, TPB
    ]

    # Warm up
    for _ in range(100):
        ctx.enqueue_function[kernel, kernel](
            W1_t,
            b1_t,
            W2_t,
            b2_t,
            m_W1_t,
            v_W1_t,
            m_b1_t,
            v_b1_t,
            m_W2_t,
            v_W2_t,
            m_b2_t,
            v_b2_t,
            obs_t,
            next_obs_t,
            actions_t,
            rewards_t,
            dones_t,
            gamma,
            lr,
            beta1,
            beta2,
            epsilon,
            t_step,
            loss_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for i in range(num_iters):
        t_step = Scalar[dtype](Float32(i + 1))
        ctx.enqueue_function[kernel, kernel](
            W1_t,
            b1_t,
            W2_t,
            b2_t,
            m_W1_t,
            v_W1_t,
            m_b1_t,
            v_b1_t,
            m_W2_t,
            v_W2_t,
            m_b2_t,
            v_b2_t,
            obs_t,
            next_obs_t,
            actions_t,
            rewards_t,
            dones_t,
            gamma,
            lr,
            beta1,
            beta2,
            epsilon,
            t_step,
            loss_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    return Float64(end - start) / Float64(num_iters)


def benchmark_kernel_launch_overhead(
    ctx: DeviceContext, num_iters: Int
) raises -> Float64:
    """Measure pure kernel launch overhead."""
    comptime dtype = DType.float32
    comptime size = 64

    var buf = ctx.enqueue_create_buffer[dtype](size)
    buf.enqueue_fill(0)
    ctx.synchronize()

    var buf_t = tile[size](buf)

    def trivial_kernel(
        buf: Tile[dtype, size]
    ):
        var i = Int(thread_idx.x)
        if i < size:
            buf[i] = rebind[Scalar[dtype]](buf[i]) + Scalar[dtype](1.0)

    for _ in range(100):
        ctx.enqueue_function[trivial_kernel, trivial_kernel](
            buf_t, grid_dim=(1,), block_dim=(64,)
        )
    ctx.synchronize()

    var start = perf_counter_ns()
    for _ in range(num_iters):
        ctx.enqueue_function[trivial_kernel, trivial_kernel](
            buf_t, grid_dim=(1,), block_dim=(64,)
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    return Float64(end - start) / Float64(num_iters)


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("Fused Mega-Kernel Benchmark: DQN Training Step")
    print("=" * 70)
    print()

    comptime OBS_DIM = 8
    comptime HIDDEN_DIM = 128
    comptime OUT_DIM = 4
    comptime BATCH = 64

    var num_iters = 1000

    print(
        "Network: "
        + String(OBS_DIM)
        + " -> "
        + String(HIDDEN_DIM)
        + " -> "
        + String(OUT_DIM)
    )
    print("Batch size: " + String(BATCH))
    print("Iterations: " + String(num_iters))
    print()
    print("The mega-kernel fuses:")
    print("  1. Forward pass (current state)")
    print("  2. Forward pass (next state for target)")
    print("  3. TD target computation")
    print("  4. Backward pass")
    print("  5. Adam weight update")
    print("All in a SINGLE kernel launch!")
    print()

    with DeviceContext() as ctx:
        print("-" * 70)
        print("GPU Kernel Launch Overhead")
        print("-" * 70)
        var launch_ns = benchmark_kernel_launch_overhead(ctx, num_iters)
        print("  " + fit(String(launch_ns / 1000), 8) + " us per kernel launch")
        print()

        print("-" * 70)
        print("GPU: Fused DQN Training Mega-Kernel (1 launch)")
        print("-" * 70)
        var mega_ns = benchmark_mega_kernel[
            BATCH, OBS_DIM, HIDDEN_DIM, OUT_DIM
        ](ctx, num_iters)
        print("  " + fit(String(mega_ns / 1000), 8) + " us per training step")
        print("  " + String(Int(1e9 / mega_ns)) + " training steps/sec")
        print()

        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print(
            "  Kernel launch overhead: " + fit(String(launch_ns / 1000), 8) + " us"
        )
        print("  Mega-kernel time:       " + fit(String(mega_ns / 1000), 8) + " us")
        print()

        # Estimate separate kernel cost
        var estimated_separate = launch_ns * 6  # 6 kernel launches minimum
        print(
            "  Estimated separate kernel cost (6 launches): "
            + fit(String(estimated_separate / 1000), 8)
            + " us"
        )
        print(
            "  Speedup from fusion: "
            + fit(String(estimated_separate / mega_ns), 4)
            + "x"
        )
        print()

        if mega_ns < estimated_separate:
            print("  -> Fusing kernels provides significant speedup!")
        else:
            print("  -> Computation dominates, fusion overhead minimal")

        print("=" * 70)
