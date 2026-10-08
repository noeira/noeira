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

THE TARGET ANCHOR (`--action target`, `target_step`) is SimToolReal's arm
rule (`env.py` pre_physics_step, `useRelativeControl: False`): the same step,
added to the PREVIOUS TARGET instead of the measured joint —

    target_j = clamp(prev_j + a_j * scale_j, lo_j, hi_j)

⚠ WHY. Anchored on `q`, a command reaches only one step past wherever the
lagging servo happens to be, so the servo's delay and lag sit INSIDE the
policy's loop: the lagged teachers plateaued at 30-44 % and one became
lag-dependent (29 % with the lag, 1.4 % without). Anchored on the previous
target, the target is a path the policy integrates itself, independent of the
plant; a slow servo follows it behind. A sign flip undoes the last step
instead of jumping the target 2 x scale around `q`. The target is then hidden
state, so the policy must see it: `TARGET_OBS`.
(SimToolReal's arm EMA of 0.1 on a previous-target anchor is algebraically a
step gain of 0.1 — `prev + 0.1 (prev + d - prev)` — so it is not taken; the
step scale is the knob. Their arm: 1.5 rad/s x dt x 0.1 = 0.15 rad/s max.)
Only the PPO driver speaks it so far; the DAgger student and the deploys
refuse a target-mode teacher until they do.
"""

from std.math import exp
from std.random import random_float64
from std.sys import is_defined
from std.sys.defines import get_defined_int

comptime DELTA_ARM: Float64 = 0.05
comptime DELTA_GRIPPER: Float64 = 0.2
comptime DELTA_ACT: Int = 6
"""Five body joints then the gripper."""
comptime ACT_HIST: Int = get_defined_int["TASK_PPO_ACT_HIST", 0]()
"""`-D TASK_PPO_ACT_HIST=K`: a policy also sees the last K EXECUTED actions
(K x 6 words, the most recent first, clipped to [-1, 1], zero at an episode's
start) — the PPO teacher after the env's observation, the pixel student as
K x 6 more planes. ⚠ WHY: under `ServoLag` the servos run 1-3 ticks behind
the commands, so `q` and `qd` do not say what is already on its way — two
states alike in `q` / `qd` with different commands in flight need different
actions, and the lagged teachers without it plateaued at 30-36 % greedy
(1af8c0ed, 51a2e3bc) where the stiff sim reached 79.5 %. The real deploy
knows what it sent. A BUILD CHOICE the checkpoints depend on."""
comptime TARGET_OBS: Int = DELTA_ACT if is_defined["TASK_PPO_TARGET_OBS"]() else 0
"""`-D TASK_PPO_TARGET_OBS`: a policy also sees its target's LEAD over the
measured joints, `target_j - q_j` (6 words, model rad, after the action
history; 0 at an episode's start). Required by `--action target`, whose
target is state the policy integrates (SimToolReal's actor reads its
`prev_action_targets`); the lead rather than the raw target because it is
what the servo still has to travel — the commands in flight — and it
normalises on its own scale instead of riding on `q`'s. A BUILD CHOICE the
checkpoints depend on."""


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


@always_inline
def target_step(
    prev: Float64, q: Float64, a: Float64, j: Int, lo: Float64, hi: Float64,
    arm: Float64 = DELTA_ARM, gripper: Float64 = DELTA_GRIPPER,
    lead: Float64 = 0.0,
) -> Float64:
    """`--action target`: the joint target for action word `a` from the
    PREVIOUS target `prev` (see the module header), clamped to [lo, hi].

    `lead` > 0 also keeps the target within `lead` rad of the measured `q`.
    ⚠ WHY it exists: an integrated target is not tied to the arm, so an arm
    held by the desk or the brick lets it run on, and a position servo then
    pushes at full torque towards a target far past the obstacle — absorbed
    by the sim's rigid desk, an overload on the real STS3215. 0 (the default,
    SimToolReal's rule) leaves it unbounded; with it, `q` re-enters the rule
    only at that bound."""
    var t = delta_target(prev, a, j, lo, hi, arm, gripper)
    if lead > 0.0:
        if t > q + lead:
            t = q + lead
        elif t < q - lead:
            t = q - lead
        if t < lo:
            t = lo
        elif t > hi:
            t = hi
    return t


comptime JAW_OPEN: Float64 = 0.20
comptime JAW_SHUT_EMPTY: Float64 = -0.10
"""An EMPTY CLOSE, in model radians of the gripper joint: the jaw opened past
`JAW_OPEN`, then shut past `JAW_SHUT_EMPTY` — on the brick it stalls near
+0.10 (sim) / +0.12-0.15 (real), so only a close on nothing gets below
-0.10 (the commanded floor is -0.17). One event per open -> shut. Read by
the PPO driver's `--empty-close-penalty` and its greedy eval, and by
`examples/so101/ppo_state_probe_sim.mojo` — one rule. ⚠ WHY: fd368031 (76.8 %)
closes empty before its first lift in 25 % of random placements and 22/30
of the shifted real scenes, then regrasps — PPO's success (goal held at any
step) never charged the miss, and on the real arm the miss knocks or
drops the cube."""


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

    ⚠ `set_limits`: a SPEED CAP on the arm joints (the gripper's is left
    alone) drawn per episode in `vmax` "lo,hi" rad/s, and the elbow's REAL
    upper stop. The real follower tops out at ~1.0-1.25 rad/s per arm joint
    under the pixel students (p95 0.8-1.1; the sim arm 2-3 rad/s), so the
    policy's 0.1 rad/tick asks ~3 rad/s and the real arm falls behind — 0.4
    rad of shoulder_lift after 1.5 s on 29 Sep, where the sim policy was
    already turning to the cube. The 0.1 rad sysid steps never reached the
    cap. The real elbow folds no further than ~1.56 rad (sysid and the runs;
    the model's range reaches 1.69).

    ⚠ PER-JOINT DYNAMICS, OFFSETS AND THE PERIOD (`set_per_joint`,
    `set_offset`, `set_period`; all off by default — a run without them draws
    exactly as before). From noeira-7d's OPEN-LOOP test (1 Oct): the real arm
    played a sim rollout's commanded targets (scene 5, tau 50 / delay 2 /
    vmax 1.1) at a steady 32 ms; real minus sim rms pan 0.007, lift 0.035,
    elbow 0.032 (a steady +0.03), wrist 0.006-0.010 rad — the arm joints are
    IN the shared model's range (delay 1 fits pan / lift / wflex marginally
    better, < 0.01 rad), but the gripper runs ~2x slower than the uncapped
    sim jaw (peak ~1.0-1.2 rad/s, a poor fit: the cube stalls the jaw), and
    lift / elbow hold a small POSITIVE offset (+0.02 / +0.02-0.05 rad). The
    real loop also ran at ~36 ms in closed loop.
    ⚠ An earlier per-joint fit from CLOSED-LOOP runs (wrist flex delay 2-3,
    tau 80-120 ms, a lift sagging below its target) was confounded by noisy
    policy targets and the 36 ms ticks and was DISCARDED — a policy probed
    under it (e330e6a0, 2/18 on the rebuilt scenes) was being tested against
    dynamics the arm does not have.
    - `set_per_joint`: tau / delay / speed-cap ranges PER JOINT, each drawn
      independently per (episode, lane, joint) — the gripper cap lives here;
    - `set_offset`: a signed per-joint offset (drawn per episode) added to
      what the actuator is handed: the joint settles at `y + off`;
    - `set_period`: the control period the servo model integrates over is
      drawn per episode (`alpha` and the speed cap use it) — the arm covers
      what a longer real tick would let it cover. The sim's physics tick
      stays its own; the stiff actuator follows `y` within it.
    """

    var n: Int
    var on: Bool
    var tau_lo: Float64
    var tau_hi: Float64
    var d_lo: Int
    var d_hi: Int
    var dt: Float64
    var alpha: List[Float64]
    """[n * DELTA_ACT]: this episode's lag factor per (lane, joint)."""
    var delay: List[Int]
    """[n * DELTA_ACT]: this episode's delay in ticks per (lane, joint)."""
    var y: List[Float64]
    """[n * DELTA_ACT]: the lagged target (the actuator gets it + `off`)."""
    var hist: List[Float64]
    """[n * LAG_MAX_DELAY * DELTA_ACT]: the last commands, a ring by tick."""
    var tick: Int
    var vmax_lo: Float64
    var vmax_hi: Float64
    var vcap: List[Float64]
    """[n * DELTA_ACT]: this episode's speed cap, rad per tick (0: none)."""
    var elbow_max: Float64
    """The elbow's upper stop, rad (0: none)."""
    var per_joint: Bool
    var jt_lo: List[Float64]
    var jt_hi: List[Float64]
    """[DELTA_ACT]: per-joint tau range, s (both 0: the shared range)."""
    var jd_lo: List[Int]
    var jd_hi: List[Int]
    """[DELTA_ACT]: per-joint delay range, ticks (-1: the shared range)."""
    var jv_lo: List[Float64]
    var jv_hi: List[Float64]
    """[DELTA_ACT]: per-joint speed cap range, rad/s (both 0: the shared
    arm cap; the gripper has none unless given here)."""
    var off_lo: List[Float64]
    var off_hi: List[Float64]
    """[DELTA_ACT]: per-joint offset range, rad (both 0: none)."""
    var off: List[Float64]
    """[n * DELTA_ACT]: this episode's offset per (lane, joint), rad."""
    var has_off: Bool
    var dt_lo: Float64
    var dt_hi: Float64
    """The drawn control period's range, s (both 0: `dt`)."""

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
        self.alpha = List[Float64](length=n * DELTA_ACT, fill=1.0)
        self.delay = List[Int](length=n * DELTA_ACT, fill=0)
        self.y = List[Float64](length=n * DELTA_ACT, fill=0.0)
        self.hist = List[Float64](length=n * LAG_MAX_DELAY * DELTA_ACT, fill=0.0)
        self.tick = 0
        self.vmax_lo = 0.0
        self.vmax_hi = 0.0
        self.vcap = List[Float64](length=n * DELTA_ACT, fill=0.0)
        self.elbow_max = 0.0
        self.per_joint = False
        self.jt_lo = List[Float64](length=DELTA_ACT, fill=0.0)
        self.jt_hi = List[Float64](length=DELTA_ACT, fill=0.0)
        self.jd_lo = List[Int](length=DELTA_ACT, fill=-1)
        self.jd_hi = List[Int](length=DELTA_ACT, fill=-1)
        self.jv_lo = List[Float64](length=DELTA_ACT, fill=0.0)
        self.jv_hi = List[Float64](length=DELTA_ACT, fill=0.0)
        self.off_lo = List[Float64](length=DELTA_ACT, fill=0.0)
        self.off_hi = List[Float64](length=DELTA_ACT, fill=0.0)
        self.off = List[Float64](length=n * DELTA_ACT, fill=0.0)
        self.has_off = False
        self.dt_lo = 0.0
        self.dt_hi = 0.0

    def set_limits(mut self, vmax: String, elbow_max: Float64) raises:
        """`vmax` "lo,hi" rad/s (or "" for none), `elbow_max` rad (0: none).
        Either turns the model on (the lag itself stays the identity if its
        ranges are empty)."""
        if vmax.byte_length() > 0:
            var p = vmax.split(",")
            self.vmax_lo = Float64(String(p[0]))
            self.vmax_hi = Float64(String(p[len(p) - 1]))
        self.elbow_max = elbow_max
        if self.vmax_hi > 0.0 or self.elbow_max > 0.0:
            self.on = True

    @staticmethod
    def _ranges(spec: String, what: String) raises -> List[Float64]:
        """"lo,hi;lo,hi;..." for the six joints (a bare "v" is v,v) -> 12
        words; an empty spec -> empty."""
        var out = List[Float64]()
        if spec.byte_length() == 0:
            return out^
        var parts = spec.split(";")
        if len(parts) != DELTA_ACT:
            raise Error("ServoLag: " + what + " needs " + String(DELTA_ACT)
                        + " ';'-separated joint ranges, got " + String(len(parts)))
        for p in parts:
            var q = String(p).split(",")
            out.append(Float64(String(q[0])))
            out.append(Float64(String(q[len(q) - 1])))
        return out^

    def set_per_joint(mut self, tau_ms: String, delay: String, vmax: String) raises:
        """Per-joint ranges, "lo,hi;..." six times (pan, lift, elbow, wrist
        flex, wrist roll, gripper): `tau_ms` ms, `delay` ticks, `vmax` rad/s.
        Any non-empty one switches every joint to INDEPENDENT draws; a
        quantity left empty keeps the shared range."""
        var t = Self._ranges(tau_ms, "tau")
        var d = Self._ranges(delay, "delay")
        var v = Self._ranges(vmax, "vmax")
        for j in range(DELTA_ACT):
            if len(t) > 0:
                self.jt_lo[j] = t[2 * j] / 1000.0
                self.jt_hi[j] = t[2 * j + 1] / 1000.0
            if len(d) > 0:
                self.jd_lo[j] = max(0, min(Int(d[2 * j]), LAG_MAX_DELAY - 1))
                self.jd_hi[j] = max(self.jd_lo[j], min(Int(d[2 * j + 1]), LAG_MAX_DELAY - 1))
            if len(v) > 0:
                self.jv_lo[j] = v[2 * j]
                self.jv_hi[j] = v[2 * j + 1]
        if len(t) > 0 or len(d) > 0 or len(v) > 0:
            self.per_joint = True
            self.on = True

    def set_offset(mut self, off: String) raises:
        """Signed per-joint offsets, "lo,hi;..." six times, rad (or "" for
        none): the joint settles at its lagged target + the offset."""
        var o = Self._ranges(off, "offset")
        for j in range(len(o) // 2):
            self.off_lo[j] = o[2 * j]
            self.off_hi[j] = o[2 * j + 1]
            if self.off_lo[j] != 0.0 or self.off_hi[j] != 0.0:
                self.has_off = True
        if self.has_off:
            self.on = True

    def set_period(mut self, period_ms: String) raises:
        """The control period the servo model integrates over, "lo,hi" ms
        drawn per episode (or "" for the sim's own `dt`)."""
        if period_ms.byte_length() > 0:
            var p = period_ms.split(",")
            self.dt_lo = Float64(String(p[0])) / 1000.0
            self.dt_hi = Float64(String(p[len(p) - 1])) / 1000.0
            if self.dt_hi > 0.0:
                self.on = True

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

    def reset_lane(
        mut self, e: Int, q: List[Float64], q_off: Int, u01: Float64,
        u02: Float64, u03: Float64 = 0.5,
    ):
        """A new episode on lane `e`: draw its dynamics and settle the lag on
        the lane's joints `q[q_off..+6]` — no stale target from the last
        episode survives into this one. Shared mode draws one tau, delay and
        cap for all joints from `u01`, `u02`, `u03` (as before); per-joint
        mode, the offsets and the period draw from the host RNG."""
        if not self.on:
            return
        var dt = self.dt
        if self.dt_hi > 0.0:
            dt = self.dt_lo + (self.dt_hi - self.dt_lo) * random_float64()
        for j in range(DELTA_ACT):
            var i = e * DELTA_ACT + j
            if self.has_off:
                self.off[i] = self.off_lo[j] + (self.off_hi[j] - self.off_lo[j]) * random_float64()
            var ut = u01
            var ud = u02
            var uv = u03
            if self.per_joint:
                ut = random_float64()
                ud = random_float64()
                uv = random_float64()
            var tlo = self.tau_lo
            var thi = self.tau_hi
            if self.per_joint and self.jt_hi[j] > 0.0:
                tlo = self.jt_lo[j]
                thi = self.jt_hi[j]
            var tau = tlo + (thi - tlo) * ut
            self.alpha[i] = 1.0 - exp(-dt / tau) if tau > 1e-6 else 1.0
            var dlo = self.d_lo
            var dhi = self.d_hi
            if self.per_joint and self.jd_lo[j] >= 0:
                dlo = self.jd_lo[j]
                dhi = self.jd_hi[j]
            self.delay[i] = dlo + Int(ud * Float64(dhi - dlo + 1))
            if self.delay[i] > dhi:
                self.delay[i] = dhi
            var vmax = 0.0
            if self.per_joint and self.jv_hi[j] > 0.0:
                vmax = self.jv_lo[j] + (self.jv_hi[j] - self.jv_lo[j]) * uv
            elif j < DELTA_ACT - 1:
                vmax = self.vmax_lo + (self.vmax_hi - self.vmax_lo) * uv
            self.vcap[i] = vmax * dt
            self.y[i] = q[q_off + j]
            for k in range(LAG_MAX_DELAY):
                self.hist[(e * LAG_MAX_DELAY + k) * DELTA_ACT + j] = q[q_off + j]

    @always_inline
    def apply(mut self, e: Int, j: Int, u: Float64) -> Float64:
        """The target lane `e`'s joint `j` actuator gets this tick, for the
        command `u`. Call once per (lane, joint) per tick, then `advance`."""
        if not self.on:
            return u
        var i = e * DELTA_ACT + j
        var w = self.tick % LAG_MAX_DELAY
        self.hist[(e * LAG_MAX_DELAY + w) * DELTA_ACT + j] = u
        var r = (self.tick - self.delay[i] + LAG_MAX_DELAY * 8) % LAG_MAX_DELAY
        var ud = self.hist[(e * LAG_MAX_DELAY + r) * DELTA_ACT + j]
        var dy = self.alpha[i] * (ud - self.y[i])
        var cap = self.vcap[i]
        if cap > 0.0:
            dy = cap if dy > cap else (-cap if dy < -cap else dy)
        self.y[i] = self.y[i] + dy
        if self.elbow_max > 0.0 and j == 2 and self.y[i] > self.elbow_max:
            self.y[i] = self.elbow_max
        return self.y[i] + self.off[i]

    def advance(mut self):
        self.tick += 1
