# =============================================================================
# Helper Functions - Observation Normalization & Shaping
# =============================================================================

from .constants import LLConstants
from std.math import sqrt


@always_inline
def normalize_position[
    T: DType
](x: Scalar[T], y: Scalar[T]) -> Tuple[Scalar[T], Scalar[T]]:
    """Normalize position relative to helipad center.

    Args:
        x: Raw x position in world units.
        y: Raw y position in world units.

    Returns:
        Tuple of (x_norm, y_norm) in range approximately [-1, 1].
    """
    var x_norm = (x - Scalar[T](LLConstants.HELIPAD_X)) / Scalar[T](
        LLConstants.W_UNITS / 2.0
    )
    var y_norm = (
        y
        - Scalar[T](LLConstants.HELIPAD_Y + LLConstants.LEG_DOWN)
    ) / Scalar[T](LLConstants.H_UNITS / 2.0)
    return (x_norm, y_norm)


@always_inline
def normalize_velocity[
    T: DType
](vx: Scalar[T], vy: Scalar[T]) -> Tuple[Scalar[T], Scalar[T]]:
    """Normalize velocity for observation.

    Args:
        vx: Raw x velocity.
        vy: Raw y velocity.

    Returns:
        Tuple of (vx_norm, vy_norm) scaled by viewport and FPS.
    """
    var vx_norm = (
        vx * Scalar[T](LLConstants.W_UNITS / 2.0) / Scalar[T](LLConstants.FPS)
    )
    var vy_norm = (
        vy * Scalar[T](LLConstants.H_UNITS / 2.0) / Scalar[T](LLConstants.FPS)
    )
    return (vx_norm, vy_norm)


@always_inline
def normalize_angular_velocity[T: DType](omega: Scalar[T]) -> Scalar[T]:
    """Normalize angular velocity for observation.

    Args:
        omega: Raw angular velocity in rad/s.

    Returns:
        Normalized angular velocity.
    """
    return Scalar[T](20.0) * omega / Scalar[T](LLConstants.FPS)


@always_inline
def get_terrain_height_at_x[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Get terrain height at a given x position using simplified chunk lookup.

    This matches the CPU version's _get_terrain_height() behavior, which returns
    the height at the start of the chunk containing x (no interpolation).

    Note: For GPU, the actual terrain edges need to be passed in. This function
    returns helipad_y for the helipad region and a basic approximation otherwise.
    Use get_terrain_height_from_edges for actual terrain lookup in GPU kernels.

    Args:
        x: X position in world units.

    Returns:
        Approximate terrain height (helipad_y for simplicity).
    """
    # For the helipad region, terrain is always at HELIPAD_Y
    # For non-helipad regions, this is an approximation
    return Scalar[T](LLConstants.HELIPAD_Y)


@always_inline
def compute_shaping[
    T: DType
](
    x_norm: Scalar[T],
    y_norm: Scalar[T],
    vx_norm: Scalar[T],
    vy_norm: Scalar[T],
    angle: Scalar[T],
    left_contact: Scalar[T],
    right_contact: Scalar[T],
) -> Scalar[T]:
    """Compute shaping potential for reward calculation.

    The shaping reward encourages:
    - Being close to landing pad (low distance)
    - Moving slowly (low speed)
    - Being upright (low angle)
    - Having legs in contact with ground

    Args:
        x_norm: Normalized x position.
        y_norm: Normalized y position.
        vx_norm: Normalized x velocity.
        vy_norm: Normalized y velocity.
        angle: Angle in radians.
        left_contact: 1.0 if left leg touching, 0.0 otherwise.
        right_contact: 1.0 if right leg touching, 0.0 otherwise.

    Returns:
        Shaping potential value.
    """
    var dist = sqrt(x_norm * x_norm + y_norm * y_norm)
    var speed = sqrt(vx_norm * vx_norm + vy_norm * vy_norm)
    var abs_angle = angle if angle >= Scalar[T](0.0) else -angle

    return (
        Scalar[T](-100.0) * dist
        - Scalar[T](100.0) * speed
        - Scalar[T](100.0) * abs_angle
        + Scalar[T](10.0) * left_contact
        + Scalar[T](10.0) * right_contact
    )


@always_inline
def lunar_terminal_and_reward[
    T: DType
](
    x_norm: Scalar[T],
    lander_contact: Bool,
    asleep: Bool,
    reward_in: Scalar[T],
) -> Tuple[Scalar[T], Bool]:
    """SINGLE SOURCE of the natural-termination predicate + terminal reward
    overrides — shared by the CPU `_compute_step_result` and BOTH GPU step
    kernels (discrete + continuous).

    Gymnasium's LunarLander, in its order:
      1. crash (lander body touching ground) or out of bounds (|x_norm| >= 1)
         -> terminated, reward = CRASH_PENALTY
      2. `not lander.awake` (the Box2D island fell asleep, `asleep` here)
         -> terminated, reward = LAND_REWARD (SET, not added)

    NOT here, by design (stay at the call sites): the step-cap TRUNCATION
    (not a natural terminal) and the GPU-only too-high check (a documented
    GPU-only deviation, applied before this call so the crash override
    order is unchanged).

    Returns (reward, terminated).
    """
    var reward = reward_in
    var terminated = False
    if lander_contact or x_norm >= Scalar[T](1.0) or x_norm <= Scalar[T](-1.0):
        terminated = True
        reward = Scalar[T](LLConstants.CRASH_PENALTY)
    if asleep:
        terminated = True
        reward = Scalar[T](LLConstants.LAND_REWARD)
    return (reward, terminated)
