"""CarRacing  state struct.

This module defines the state representation for the GPU-enabled CarRacing environment.
"""

from std.math import sqrt, pi
from noeira.core.tiles import Tile
from noeira.core import State


@fieldwise_init
struct CarRacingState[DTYPE: DType](
    Copyable, ImplicitlyCopyable, Movable, State
):
    """State for CarRacing  environment.

    Observation format (13 values, normalized for neural networks):
    - [0-1]: Position (x, y) normalized by PLAYFIELD
    - [2]: Angle normalized by pi
    - [3-4]: Velocity (vx, vy) normalized by max velocity
    - [5]: Angular velocity normalized
    - [6-7]: Front wheel steering angles (FL, FR)
    - [8-11]: Wheel angular velocities (FL, FR, RL, RR)
    - [12]: Speed indicator (magnitude of velocity)
    """

    # Hull state
    var x: Scalar[Self.DTYPE]
    var y: Scalar[Self.DTYPE]
    var angle: Scalar[Self.DTYPE]
    var vx: Scalar[Self.DTYPE]
    var vy: Scalar[Self.DTYPE]
    var angular_velocity: Scalar[Self.DTYPE]

    # Wheel states
    var wheel_angle_fl: Scalar[Self.DTYPE]
    var wheel_angle_fr: Scalar[Self.DTYPE]
    var wheel_omega_fl: Scalar[Self.DTYPE]
    var wheel_omega_fr: Scalar[Self.DTYPE]
    var wheel_omega_rl: Scalar[Self.DTYPE]
    var wheel_omega_rr: Scalar[Self.DTYPE]

    # Speed indicator
    var speed: Scalar[Self.DTYPE]

    def __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.angular_velocity = 0.0
        self.wheel_angle_fl = 0.0
        self.wheel_angle_fr = 0.0
        self.wheel_omega_fl = 0.0
        self.wheel_omega_fr = 0.0
        self.wheel_omega_rl = 0.0
        self.wheel_omega_rr = 0.0
        self.speed = 0.0

    def __init__(out self, *, copy: Self):
        self.x = copy.x
        self.y = copy.y
        self.angle = copy.angle
        self.vx = copy.vx
        self.vy = copy.vy
        self.angular_velocity = copy.angular_velocity
        self.wheel_angle_fl = copy.wheel_angle_fl
        self.wheel_angle_fr = copy.wheel_angle_fr
        self.wheel_omega_fl = copy.wheel_omega_fl
        self.wheel_omega_fr = copy.wheel_omega_fr
        self.wheel_omega_rl = copy.wheel_omega_rl
        self.wheel_omega_rr = copy.wheel_omega_rr
        self.speed = copy.speed

    def __init__(out self, *, deinit move: Self):
        self.x = move.x
        self.y = move.y
        self.angle = move.angle
        self.vx = move.vx
        self.vy = move.vy
        self.angular_velocity = move.angular_velocity
        self.wheel_angle_fl = move.wheel_angle_fl
        self.wheel_angle_fr = move.wheel_angle_fr
        self.wheel_omega_fl = move.wheel_omega_fl
        self.wheel_omega_fr = move.wheel_omega_fr
        self.wheel_omega_rl = move.wheel_omega_rl
        self.wheel_omega_rr = move.wheel_omega_rr
        self.speed = move.speed

    def __eq__(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y

    def to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to list for agent interface."""
        var result = List[Scalar[Self.DTYPE]]()
        result.append(self.x)
        result.append(self.y)
        result.append(self.angle)
        result.append(self.vx)
        result.append(self.vy)
        result.append(self.angular_velocity)
        result.append(self.wheel_angle_fl)
        result.append(self.wheel_angle_fr)
        result.append(self.wheel_omega_fl)
        result.append(self.wheel_omega_fr)
        result.append(self.wheel_omega_rl)
        result.append(self.wheel_omega_rr)
        result.append(self.speed)
        return result^

    def to_list_typed[dtype: DType](self) -> List[Scalar[dtype]]:
        """Convert to list with specified dtype."""
        var result = List[Scalar[dtype]]()
        result.append(Scalar[dtype](self.x))
        result.append(Scalar[dtype](self.y))
        result.append(Scalar[dtype](self.angle))
        result.append(Scalar[dtype](self.vx))
        result.append(Scalar[dtype](self.vy))
        result.append(Scalar[dtype](self.angular_velocity))
        result.append(Scalar[dtype](self.wheel_angle_fl))
        result.append(Scalar[dtype](self.wheel_angle_fr))
        result.append(Scalar[dtype](self.wheel_omega_fl))
        result.append(Scalar[dtype](self.wheel_omega_fr))
        result.append(Scalar[dtype](self.wheel_omega_rl))
        result.append(Scalar[dtype](self.wheel_omega_rr))
        result.append(Scalar[dtype](self.speed))
        return result^

    @staticmethod
    def from_buffer[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
    ](
        state: Tile[DType.float32, BATCH, STATE_SIZE],
        env: Int,
    ) -> CarRacingState[Self.DTYPE]:
        """Create state from std.gpu buffer."""

        var result = CarRacingState[Self.DTYPE]()
        result.x = rebind[Scalar[Self.DTYPE]](state[env, OBS_OFFSET + 0])
        result.y = rebind[Scalar[Self.DTYPE]](state[env, OBS_OFFSET + 1])
        result.angle = rebind[Scalar[Self.DTYPE]](state[env, OBS_OFFSET + 2])
        result.vx = rebind[Scalar[Self.DTYPE]](state[env, OBS_OFFSET + 3])
        result.vy = rebind[Scalar[Self.DTYPE]](state[env, OBS_OFFSET + 4])
        result.angular_velocity = rebind[Scalar[Self.DTYPE]](
            state[env, OBS_OFFSET + 5]
        )
        result.wheel_angle_fl = rebind[Scalar[Self.DTYPE]](
            state[env, OBS_OFFSET + 6]
        )
        result.wheel_angle_fr = rebind[Scalar[Self.DTYPE]](
            state[env, OBS_OFFSET + 7]
        )
        result.wheel_omega_fl = rebind[Scalar[Self.DTYPE]](
            state[env, OBS_OFFSET + 8]
        )
        result.wheel_omega_fr = rebind[Scalar[Self.DTYPE]](
            state[env, OBS_OFFSET + 9]
        )
        result.wheel_omega_rl = rebind[Scalar[Self.DTYPE]](
            state[env, OBS_OFFSET + 10]
        )
        result.wheel_omega_rr = rebind[Scalar[Self.DTYPE]](
            state[env, OBS_OFFSET + 11]
        )
        result.speed = rebind[Scalar[Self.DTYPE]](state[env, OBS_OFFSET + 12])
        return result^
