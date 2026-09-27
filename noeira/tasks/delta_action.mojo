"""The SO-101 families' DELTA action — written once, for every place that
turns a policy's output into a joint target.

    target_j = clamp(q_j + a_j * scale_j, lo_j, hi_j)    a in [-1, 1]
    scale_j  = DELTA_ARM (rad per control step) for the five body joints,
               DELTA_GRIPPER for the gripper (the LAST action word)

`q` is the joint's MEASURED position at the step (not the previous target),
`lo`/`hi` the actuator's ctrlrange — the model's joint limits, in model
radians. so101-nexus's `pd_joint_delta_pos` (0.05 / 0.2).

⚠⚠ THREE READERS, ONE RULE. The PPO driver trains with it
(`ppo_family_driver._delta_to_env`), the pixel DAgger student is labelled
and evaluated with it, and the REAL-ARM deploy
(`examples/so101/pixel_student_deploy_real.mojo`) commands the follower with
it. A scale changed in one of them and not the others would send a policy's
learned step size to a real arm at the wrong magnitude — silently, with the
policy looking merely clumsy (`_a_rule_written_inline_twice_drifts`).
"""

from std.math import exp

comptime DELTA_ARM: Float64 = 0.05
comptime DELTA_GRIPPER: Float64 = 0.2
comptime DELTA_ACT: Int = 6
"""Five body joints then the gripper."""


@always_inline
def delta_scale(
    j: Int, arm: Float64 = DELTA_ARM, gripper: Float64 = DELTA_GRIPPER
) -> Float64:
    """Radians per control step at a = 1 for action word `j`."""
    return gripper if j == DELTA_ACT - 1 else arm


@always_inline
def delta_target(
    q: Float64, a: Float64, j: Int, lo: Float64, hi: Float64,
    arm: Float64 = DELTA_ARM, gripper: Float64 = DELTA_GRIPPER,
) -> Float64:
    """The joint target (model radians) for action word `a` at measured `q`,
    clamped to [lo, hi]. `arm` / `gripper` are the per-step scales — the
    defaults are so101-nexus's (a stiff sim at 50 Hz); a policy trained under
    `ServoLag` needs larger ones (Squint drives the real SO-101 at 0.15), and
    whatever it was trained with travels in its manifest."""
    var t = q + a * delta_scale(j, arm, gripper)
    if t < lo:
        return lo
    if t > hi:
        return hi
    return t


comptime LAG_MAX_DELAY: Int = 4
"""Control ticks of command delay the servo model can hold."""


struct ServoLag(Movable):
    """The REAL follower's servos, as seen from the control loop: each joint
    target reaches the joint `delay` ticks late and through a first-order lag
    of time constant `tau` —

        u_t  = the commanded target (`delta_target`)
        y_t  = y_{t-1} + alpha (u_{t-d} - y_{t-1})     alpha = 1 - exp(-dt / tau)

    and `y_t`, not `u_t`, is what the sim's stiff position actuator is given.

    ⚠⚠ WHY. The sim's STS3215 is a position actuator with kp ~1000 N m/rad
    against a 2.94 N m force range: any error past ~3 mrad is full torque, so
    a 0.05 rad delta is reached within one control tick. The real servos,
    recorded under the cube-in-bowl pixel student on 27 Sep (a558f193f's
    --record), cover ~20 % of the commanded step per tick (tau ~140 ms) and
    start 2-3 ticks late. A reactive policy trained on the stiff sim flips its
    corrections every tick and the sim arm follows; the real arm averages them
    out and drifts. The scripted expert never showed it: it waits for the arm
    to SETTLE after every leg, so only the pose it reaches matters, not when.

    `tau` and `delay` are drawn PER EPISODE, per lane, uniformly in the
    configured ranges (randomised dynamics: the policy must not depend on the
    exact numbers). Off (`tau_hi_ms <= 0`) it is the identity — every run
    before it is unchanged.
    """

    var n: Int
    var on: Bool
    var tau_lo: Float64
    var tau_hi: Float64
    var d_lo: Int
    var d_hi: Int
    var dt: Float64
    var alpha: List[Float64]
    var delay: List[Int]
    var y: List[Float64]
    """[n * DELTA_ACT]: the target the stiff actuator is handed."""
    var hist: List[Float64]
    """[n * LAG_MAX_DELAY * DELTA_ACT]: the last commands, a ring by tick."""
    var tick: Int

    def __init__(
        out self, n: Int, tau_lo_ms: Float64, tau_hi_ms: Float64, d_lo: Int,
        d_hi: Int, dt: Float64,
    ):
        self.n = n
        self.on = tau_hi_ms > 0.0 or d_hi > 0
        self.tau_lo = tau_lo_ms / 1000.0
        self.tau_hi = tau_hi_ms / 1000.0
        self.d_lo = max(0, min(d_lo, LAG_MAX_DELAY - 1))
        self.d_hi = max(self.d_lo, min(d_hi, LAG_MAX_DELAY - 1))
        self.dt = dt
        self.alpha = List[Float64](length=n, fill=1.0)
        self.delay = List[Int](length=n, fill=0)
        self.y = List[Float64](length=n * DELTA_ACT, fill=0.0)
        self.hist = List[Float64](length=n * LAG_MAX_DELAY * DELTA_ACT, fill=0.0)
        self.tick = 0

    @staticmethod
    def parse(n: Int, tau: String, delay: String, dt: Float64) raises -> Self:
        """`tau` "lo,hi" ms (or "" for off), `delay` "lo,hi" ticks."""
        var tlo = 0.0
        var thi = 0.0
        var dlo = 0
        var dhi = 0
        if tau.byte_length() > 0:
            var p = tau.split(",")
            tlo = Float64(String(p[0]))
            thi = Float64(String(p[len(p) - 1]))
        if delay.byte_length() > 0:
            var p = delay.split(",")
            dlo = Int(String(p[0]))
            dhi = Int(String(p[len(p) - 1]))
        return Self(n, tlo, thi, dlo, dhi, dt)

    def reset_lane(mut self, e: Int, q: List[Float64], q_off: Int, u01: Float64, u02: Float64):
        """A new episode on lane `e`: draw its tau and delay (from the two
        uniforms), and settle the lag on the lane's joints `q[q_off..+6]` — no
        stale target from the last episode survives into this one."""
        if not self.on:
            return
        var tau = self.tau_lo + (self.tau_hi - self.tau_lo) * u01
        self.alpha[e] = 1.0 - exp(-self.dt / tau) if tau > 1e-6 else 1.0
        self.delay[e] = self.d_lo + Int(u02 * Float64(self.d_hi - self.d_lo + 1))
        if self.delay[e] > self.d_hi:
            self.delay[e] = self.d_hi
        for j in range(DELTA_ACT):
            self.y[e * DELTA_ACT + j] = q[q_off + j]
            for k in range(LAG_MAX_DELAY):
                self.hist[(e * LAG_MAX_DELAY + k) * DELTA_ACT + j] = q[q_off + j]

    @always_inline
    def apply(mut self, e: Int, j: Int, u: Float64) -> Float64:
        """The target lane `e`'s joint `j` actuator gets this tick, for the
        command `u`. Call once per (lane, joint) per tick, then `advance`."""
        if not self.on:
            return u
        var w = self.tick % LAG_MAX_DELAY
        self.hist[(e * LAG_MAX_DELAY + w) * DELTA_ACT + j] = u
        var r = (self.tick - self.delay[e] + LAG_MAX_DELAY * 8) % LAG_MAX_DELAY
        var ud = self.hist[(e * LAG_MAX_DELAY + r) * DELTA_ACT + j]
        var i = e * DELTA_ACT + j
        self.y[i] = self.y[i] + self.alpha[e] * (ud - self.y[i])
        return self.y[i]

    def advance(mut self):
        self.tick += 1
