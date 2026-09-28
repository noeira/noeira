"""Island sleep — Box2D 2.3's b2Island::Solve sleep test.

Reference: `references/pybox2d-2.3.10/Box2D/Dynamics/b2Island.cpp`.

Box2D puts an island (bodies connected by joints / contacts between dynamic
bodies) to sleep once every body in it has stayed below the linear and
angular sleep tolerances for `B2_TIME_TO_SLEEP` seconds; any body above
them resets the island's clock. Gymnasium's LunarLander ends an episode
with +100 exactly when the lander island falls asleep (`not lander.awake`).

The envs here are one island each (the ground is static), so one clock per
env stored in the env's own state is Box2D's per-body `m_sleepTime` minimum.
"""

from layout import LayoutTensor, Layout

from .constants import (
    dtype,
    BODY_STATE_SIZE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    B2_TIME_TO_SLEEP,
    B2_LINEAR_SLEEP_TOLERANCE,
    B2_ANGULAR_SLEEP_TOLERANCE,
)


@always_inline
def island_sleep_time_single_env[
    BATCH: Int,
    NUM_BODIES: Int,
    STATE_SIZE: Int,
    BODIES_OFFSET: Int,
](
    env: Int,
    state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    sleep_time: Scalar[dtype],
    dt: Scalar[dtype],
) -> Scalar[dtype]:
    """Advance the island's sleep clock after a step: 0 if any dynamic body
    moves faster than the tolerances, else `sleep_time + dt`."""
    var lin_tol = Scalar[dtype](B2_LINEAR_SLEEP_TOLERANCE)
    var ang_tol = Scalar[dtype](B2_ANGULAR_SLEEP_TOLERANCE)
    comptime for body in range(NUM_BODIES):
        var off = BODIES_OFFSET + body * BODY_STATE_SIZE
        if rebind[Scalar[dtype]](state[env, off + IDX_INV_MASS]) != Scalar[
            dtype
        ](0):
            var vx = rebind[Scalar[dtype]](state[env, off + IDX_VX])
            var vy = rebind[Scalar[dtype]](state[env, off + IDX_VY])
            var w = rebind[Scalar[dtype]](state[env, off + IDX_OMEGA])
            if w * w > ang_tol * ang_tol or vx * vx + vy * vy > lin_tol * lin_tol:
                return Scalar[dtype](0)
    return sleep_time + dt


@always_inline
def island_is_asleep(sleep_time: Scalar[dtype]) -> Bool:
    """True once the island's clock has reached `B2_TIME_TO_SLEEP`."""
    return sleep_time >= Scalar[dtype](B2_TIME_TO_SLEEP)
