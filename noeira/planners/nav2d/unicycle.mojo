"""A ground robot as a lagged unicycle, and an MPPI navigator over a NavGrid.

    var nav = UnicycleNavigator[](grid^, response^, params)
    nav.set_goal(gx, gy)
    var cmd = nav.plan(x, y, yaw, v, w)     # (v_cmd, w_cmd) to send now

THE MODEL (state x, y, yaw, v, w; action a in [-1, 1]^2):

    v_cmd = v_min + (a0 + 1)/2 (v_max - v_min)      w_cmd = a1 w_max
    (v*, w*) = response.achieved(v_cmd, w_cmd)       # what the robot DOES
    v += (v* - v)(1 - exp(-dt/tau_v))                # first-order lags
    w += (w* - w)(1 - exp(-dt/tau_w))
    yaw += w dt;  x += v cos(yaw) dt;  y += v sin(yaw) dt

⚠ `ResponseMap` IS THE POINT. A legged robot driven by a learned policy does
not do what it is told: the G1 under BFM-Zero overshoots straight speed by
up to 60 % and turns 1.2–1.5x faster than asked (BFM_ZERO_ROOM_PLAN, layer 3
probe). A planner that assumes identity plans to brake where the robot will
not stop. The map is a measured table, bilinear in the command; `identity()`
is the honest default for a robot that does track.

THE COST (MPPI maximises the sum of rewards + terminal value):

    reward  = - w_c2g     cost_to_go(x', y')                 progress, AROUND obstacles
              - w_clear   max(0, 1 - (clearance - inflate)/clear_band)^2
              - w_collide [clearance < inflate]               the inflated footprint
              - w_effort  |a|^2
    terminal = - w_term cost_to_go - w_stop v^2 exp(-cost_to_go / stop_band)

The last term asks the robot to be SLOW when it is close: a lagged robot
that arrives at speed overshoots by v tau, and the planner can only plan the
brake if being fast near the goal costs something.

⚠ AND A HARD APPROACH LIMIT (`slow_radius` > 0): the commanded speed is
capped at `max(v_slow_min, v_max * cost_to_go / slow_radius)` — in every
rollout AND on the command actually sent (`command_at`), so the planner can
never plan past it. A soft cost alone lost to the progress term on the G1:
it reached 1.36 m/s, cut the command 0.7 m out, and coasted through the goal.

`policy_action_cpu` follows the steepest descent of the cost-to-go: MPPI's
policy-seeded samples (`NUM_PI_TRAJS`) then already go around the obstacle,
so a small sample budget does not have to discover the detour by chance.
"""

from std.math import atan2, cos, exp, sin, sqrt

from noeira.planners.trajectory import MPPICPU, RolloutCallbackCPU
from .costmap import NavGrid


def _wrap(a: Float64) -> Float64:
    var x = a
    while x > 3.141592653589793:
        x -= 6.283185307179586
    while x < -3.141592653589793:
        x += 6.283185307179586
    return x


struct ResponseMap(Copyable, Movable):
    """Commanded (v, w) on a grid -> achieved (v, w), bilinear in between,
    clamped at the edges. Row-major: index = iv * len(ws) + iw."""

    var vs: List[Float64]
    var ws: List[Float64]
    var av: List[Float64]
    var aw: List[Float64]

    def __init__(
        out self, vs: List[Float64], ws: List[Float64], av: List[Float64],
        aw: List[Float64],
    ) raises:
        if len(vs) < 2 or len(ws) < 2 or len(av) != len(vs) * len(ws) or len(aw) != len(av):
            raise Error("ResponseMap: need >= 2 x 2 commands and one achieved pair per cell")
        for i in range(1, len(vs)):
            if vs[i] <= vs[i - 1]:
                raise Error("ResponseMap: commanded v must increase")
        for i in range(1, len(ws)):
            if ws[i] <= ws[i - 1]:
                raise Error("ResponseMap: commanded w must increase")
        self.vs = vs.copy()
        self.ws = ws.copy()
        self.av = av.copy()
        self.aw = aw.copy()

    @staticmethod
    def identity(v_min: Float64, v_max: Float64, w_max: Float64) raises -> ResponseMap:
        var vs: List[Float64] = [v_min, v_max]
        var ws: List[Float64] = [-w_max, w_max]
        var av: List[Float64] = [v_min, v_min, v_max, v_max]
        var aw: List[Float64] = [-w_max, w_max, -w_max, w_max]
        return ResponseMap(vs, ws, av, aw)

    @staticmethod
    def _locate(xs: List[Float64], x: Float64) -> Tuple[Int, Float64]:
        if x <= xs[0]:
            return (0, 0.0)
        var n = len(xs)
        if x >= xs[n - 1]:
            return (n - 2, 1.0)
        for i in range(n - 1):
            if x <= xs[i + 1]:
                return (i, (x - xs[i]) / (xs[i + 1] - xs[i]))
        return (n - 2, 1.0)

    def achieved(self, v: Float64, w: Float64) -> Tuple[Float64, Float64]:
        var a = Self._locate(self.vs, v)
        var b = Self._locate(self.ws, w)
        var nw = len(self.ws)
        var i = a[0]
        var j = b[0]
        var tv = a[1]
        var tw = b[1]
        var k00 = i * nw + j
        var k01 = i * nw + j + 1
        var k10 = (i + 1) * nw + j
        var k11 = (i + 1) * nw + j + 1
        var rv = ((1.0 - tv) * ((1.0 - tw) * self.av[k00] + tw * self.av[k01])
                  + tv * ((1.0 - tw) * self.av[k10] + tw * self.av[k11]))
        var rw = ((1.0 - tv) * ((1.0 - tw) * self.aw[k00] + tw * self.aw[k01])
                  + tv * ((1.0 - tw) * self.aw[k10] + tw * self.aw[k11]))
        return (rv, rw)


@fieldwise_init
struct NavParams(Copyable, ImplicitlyCopyable, Movable):
    var dt: Float64
    var tau_v: Float64
    var tau_w: Float64
    var v_min: Float64
    var v_max: Float64
    var w_max: Float64
    var w_c2g: Float64
    var w_clear: Float64
    var clear_band: Float64
    var w_collide: Float64
    var w_effort: Float64
    var w_term: Float64
    var w_stop: Float64
    var stop_band: Float64
    var slow_radius: Float64   # 0 = no approach limit
    var v_slow_min: Float64


def nav_params_default() -> NavParams:
    return NavParams(
        dt=0.1, tau_v=0.3, tau_w=0.3, v_min=0.0, v_max=1.0, w_max=1.2,
        w_c2g=1.0, w_clear=4.0, clear_band=0.4, w_collide=100.0,
        w_effort=0.01, w_term=5.0, w_stop=20.0, stop_band=0.5,
        slow_radius=0.0, v_slow_min=0.25,
    )


struct UnicycleNavCallback(Movable, Deinitable, RolloutCallbackCPU):
    """The model + cost above, as the trajectory optimisers' callback."""

    comptime LATENT_DIM: Int = 5
    comptime ACTION_DIM: Int = 2

    var grid: NavGrid
    var response: ResponseMap
    var p: NavParams

    def __init__(out self, var grid: NavGrid, var response: ResponseMap, p: NavParams):
        self.grid = grid^
        self.response = response^
        self.p = p

    def command_of(self, a0: Float64, a1: Float64) -> Tuple[Float64, Float64]:
        var c0 = a0 if a0 < 1.0 else 1.0
        c0 = c0 if c0 > -1.0 else -1.0
        var c1 = a1 if a1 < 1.0 else 1.0
        c1 = c1 if c1 > -1.0 else -1.0
        return (
            self.p.v_min + 0.5 * (c0 + 1.0) * (self.p.v_max - self.p.v_min),
            c1 * self.p.w_max,
        )

    def command_at(self, x: Float64, y: Float64, a0: Float64, a1: Float64) -> Tuple[Float64, Float64]:
        """`command_of` with the approach limit at position (x, y)."""
        var c = self.command_of(a0, a1)
        if self.p.slow_radius <= 0.0:
            return c
        var lim = self.p.v_max * self.grid.cost_to_go(x, y) / self.p.slow_radius
        if lim < self.p.v_slow_min:
            lim = self.p.v_slow_min
        return (c[0] if c[0] < lim else lim, c[1])

    def policy_action_cpu(mut self, z: List[Float64], mut action_out: List[Float64]) raises:
        # follow the cost-to-go's steepest descent: turn toward it, and
        # walk only when roughly facing it
        var head = self.grid.descent(z[0], z[1])
        var err = _wrap(head - z[2])
        var aw = 2.0 * err / self.p.w_max
        if aw > 1.0:
            aw = 1.0
        if aw < -1.0:
            aw = -1.0
        action_out[1] = aw
        var ae = err if err > 0.0 else -err
        action_out[0] = 0.4 if ae < 0.5 else -1.0

    def rollout_step_cpu(
        mut self, z: List[Float64], a: List[Float64], mut z_next_out: List[Float64],
    ) raises -> Float64:
        var cmd = self.command_at(z[0], z[1], a[0], a[1])
        var ach = self.response.achieved(cmd[0], cmd[1])
        var kv = 1.0 - exp(-self.p.dt / self.p.tau_v)
        var kw = 1.0 - exp(-self.p.dt / self.p.tau_w)
        var v = z[3] + (ach[0] - z[3]) * kv
        var w = z[4] + (ach[1] - z[4]) * kw
        var yaw = z[2] + w * self.p.dt
        var x = z[0] + v * cos(yaw) * self.p.dt
        var y = z[1] + v * sin(yaw) * self.p.dt
        z_next_out[0] = x
        z_next_out[1] = y
        z_next_out[2] = yaw
        z_next_out[3] = v
        z_next_out[4] = w
        var c = self.grid.clearance(x, y)
        var r = -self.p.w_c2g * self.grid.cost_to_go(x, y)
        var m = 1.0 - (c - self.grid.inflate) / self.p.clear_band
        if m > 0.0:
            r -= self.p.w_clear * m * m
        if c < self.grid.inflate:
            r -= self.p.w_collide
        r -= self.p.w_effort * (a[0] * a[0] + a[1] * a[1])
        return r

    def terminal_value_cpu(mut self, z: List[Float64]) raises -> Float64:
        var c2g = self.grid.cost_to_go(z[0], z[1])
        return -self.p.w_term * c2g - self.p.w_stop * z[3] * z[3] * exp(-c2g / self.p.stop_band)


struct UnicycleNavigator[
    HORIZON: Int = 25,
    NUM_SAMPLES: Int = 256,
    NUM_PI_TRAJS: Int = 16,
    NUM_ITERATIONS: Int = 6,
    NUM_ELITES: Int = 32,
](Movable):
    """MPPI over `UnicycleNavCallback`, replanned every call (receding horizon)."""

    var cb: UnicycleNavCallback
    var mppi: MPPICPU[5, 2, Self.HORIZON, Self.NUM_SAMPLES, Self.NUM_PI_TRAJS, Self.NUM_ITERATIONS, Self.NUM_ELITES]
    var temperature: Float64

    def __init__(out self, var grid: NavGrid, var response: ResponseMap, p: NavParams) raises:
        self.cb = UnicycleNavCallback(grid^, response^, p)
        self.mppi = MPPICPU[5, 2, Self.HORIZON, Self.NUM_SAMPLES, Self.NUM_PI_TRAJS, Self.NUM_ITERATIONS, Self.NUM_ELITES]()
        self.temperature = 0.5

    def set_goal(mut self, gx: Float64, gy: Float64) raises:
        self.cb.grid.set_goal(gx, gy)
        self.mppi.start_episode()

    def plan(
        mut self, x: Float64, y: Float64, yaw: Float64, v: Float64, w: Float64,
    ) raises -> Tuple[Float64, Float64]:
        """The (v_cmd, w_cmd) to send now, in physical units."""
        var z0: List[Float64] = [x, y, yaw, v, w]
        var a = self.mppi.plan(
            self.cb, z0, gamma=1.0, temperature=self.temperature,
            action_scale=1.0, deterministic=True,
        )
        return self.cb.command_at(x, y, a[0], a[1])

    def cost_to_go(self, x: Float64, y: Float64) -> Float64:
        return self.cb.grid.cost_to_go(x, y)

    def clearance(self, x: Float64, y: Float64) -> Float64:
        return self.cb.grid.clearance(x, y)
