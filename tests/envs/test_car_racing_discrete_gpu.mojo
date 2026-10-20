"""GPU smoke test for CarRacingDiscrete (multi-body, GPUDiscreteEnv).

Resets a batch, drives every car with the discrete "gas" action, and checks the
pipeline end-to-end: cars accelerate, visit tiles (positive Gymnasium reward),
produce finite normalized observations, and episodes terminate.

Run: pixi run -e apple mojo run -I . tests/envs/test_car_racing_discrete_gpu.mojo
"""

from max.gpu.host import DeviceContext
from noeira.physics2d import dtype
from noeira.envs.car_racing import CarRacingDiscrete


comptime B = 16
comptime K = CarRacingDiscrete[DType.float32].MAX_STEPS + 20
"""Past MAX_STEPS so truncation must fire at least once. Derived, not written:
a literal 520 went stale when MAX_STEPS moved 500 -> 1000 (0b985874e) and the
gate then failed with "no episodes terminated"."""


def main() raises:
    print("=== CarRacingDiscrete GPU smoke test ===")
    comptime E = CarRacingDiscrete[DType.float32]
    comptime SSZ = E.STATE_SIZE
    comptime OBS = E.OBS_DIM
    print("STATE_SIZE =", SSZ, " OBS_DIM =", OBS, " NUM_ACTIONS =", E.NUM_ACTIONS)

    var ctx = DeviceContext()
    var states = ctx.enqueue_create_buffer[dtype](B * SSZ)
    var actions = ctx.enqueue_create_buffer[dtype](B * 1)
    var rewards = ctx.enqueue_create_buffer[dtype](B)
    var dones = ctx.enqueue_create_buffer[dtype](B)
    var term = ctx.enqueue_create_buffer[dtype](B)
    var obs = ctx.enqueue_create_buffer[dtype](B * OBS)

    # All cars take action 3 = gas.
    var ahost = ctx.enqueue_create_host_buffer[dtype](B * 1)
    ctx.synchronize()
    for i in range(B):
        ahost[i] = Scalar[dtype](3.0)
    ctx.enqueue_copy(actions, ahost)

    E.reset_kernel_gpu[B, SSZ](ctx, states, rng_seed=7)
    ctx.synchronize()

    var rhost = ctx.enqueue_create_host_buffer[dtype](B)
    var dhost = ctx.enqueue_create_host_buffer[dtype](B)
    var ohost = ctx.enqueue_create_host_buffer[dtype](B * OBS)
    ctx.synchronize()

    var pos_reward_events = 0
    var done_events = 0
    var offfield_events = 0
    var max_speed: Float64 = 0.0
    var nan_seen = False

    for _ in range(K):
        E.step_kernel_gpu[B, SSZ, OBS](
            ctx, states, actions, rewards, dones, term, obs
        )
        # Capture step outputs BEFORE selective_reset clears the done flags.
        ctx.enqueue_copy(rhost, rewards)
        ctx.enqueue_copy(dhost, dones)
        ctx.enqueue_copy(ohost, obs)
        ctx.synchronize()
        E.selective_reset_kernel_gpu[B, SSZ](ctx, states, dones, rng_seed=123)
        ctx.synchronize()
        for e in range(B):
            var r = Float64(rhost[e])
            if r > 0.0:
                pos_reward_events += 1
            if r < -99.0:
                offfield_events += 1
            if Float64(dhost[e]) > 0.5:
                done_events += 1
            var spd = Float64(ohost[e * OBS + 12])  # normalized speed obs
            if spd > max_speed:
                max_speed = spd
            for d in range(OBS):
                var v = Float64(ohost[e * OBS + d])
                if v != v:  # NaN
                    nan_seen = True

    print("pos_reward_events =", pos_reward_events)
    print("done_events =", done_events)
    print("offfield(-100) events =", offfield_events)
    print("max normalized speed obs =", max_speed)

    if nan_seen:
        raise Error("NaN in observations")
    if max_speed <= 0.0:
        raise Error("cars never accelerated under gas")
    if pos_reward_events <= 0:
        raise Error("no tiles visited (no positive rewards) — track/reward broken")
    if done_events <= 0:
        raise Error("no episodes terminated over the run")

    print("=== PASS: GPU discrete env steps, rewards, and terminates ===")
