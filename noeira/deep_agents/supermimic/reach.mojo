"""A hand held at a point of the world through the prompt interface's z, the tracker frozen.

The prompt interface turns a reward into z by retrieval over a pool. For "this
hand at this point" the retrieved z is only a starting guess: the tracker may
or may not put the hand there under it. This module holds what replaces the
retrieval for that one command, and it knows neither a simulator nor a policy:

    SMReachInterface    target (3) -> z, fitted on z's the tracker itself was
                        seen to succeed with (a kernel ridge regression), and
                        the density of those successes around a target, which
                        says how far the answer can be trusted there
    SMReachLoop         the command over time: a ramp of z from the one in
                        force, then a slow integral on the hand's measured
                        error in the world, kept inside the interface's domain
    SMZRamp             the way back: z ramped to another z (the stand's)

A caller steps its own simulator. Each control step, BEFORE the simulator's,
it gives the loop the pelvis pose and the hand's place and sets the z the loop
hands back when it says z changed:

    var g = loop.start(f, target, pelvis, z_now)     # SM_REACH_REFUSED: nothing started
    ...
    if loop.step(f, pelvis, hand):
        set_z(loop.z)

FRAMES. A target and a hand's place are relative to the pelvis in the heading
frame (x forward, y left, z up from the pelvis, metres): the frame of the
prompt interface's hand atoms. `pelvis` is (x, y, z, yaw) in the world. The
world point of a reach is fixed at `start` from the pelvis pose then; the body
may turn or drift under it afterwards, the loop holds the point.

THE INTERFACE'S FILE (text, whitespace-separated):

    smreach 1 <n> <width> <ell> <loop_threshold> <cut>
    <width numbers>                       m, the mean of z in units of s
    <width numbers>                       s, the unit of each z component
    n rows of 4 + width numbers           x y z | counts | alpha

z(t) = (m + sum_i exp(-|t - x_i|^2 / 2 ell^2) alpha_i) * s. `counts` is 1 for a
pair the tracker succeeded with (it counts in the density) and 0 for a
starting pair (a retrieved z at a small weight, there so that z is defined
away from the successes). The density at t is the sum of the kernel over the
pairs that count.

TWO THRESHOLDS on that density: `loop_threshold` bounds the region the loop's
command may move in, `cut` (not under it) is what a target needs to be taken
as SURE. Between the two a target is LIKELY: the loop has little room around
it and gives up more often. Under the loop's threshold it is REFUSED. Both
are calibrated on runs of the whole command, not derived here.

⚠ z is not portable: a file belongs to the tracker, the prompt interface and
the pool its pairs were collected with.
"""

from std.math import cos, exp, sin, sqrt

from noeira.core.bytes import string_from_bytes
from noeira.io.fileio import read_file_bytes


comptime SM_REACH_REFUSED: Int = 0
comptime SM_REACH_LIKELY: Int = 1
comptime SM_REACH_SURE: Int = 2


def sm_reach_grade_name(g: Int) -> String:
    if g == SM_REACH_SURE:
        return String("sure")
    if g == SM_REACH_LIKELY:
        return String("likely")
    return String("refused")


def sm_min_jerk(x: Float64) -> Float64:
    """0 at 0, 1 at 1, no speed and no acceleration at either end."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def sm_reach_world(pelvis: List[Float64], rel: List[Float64]) -> List[Float64]:
    """A place given from the pelvis (heading frame), in the world."""
    var c = cos(pelvis[3])
    var s = sin(pelvis[3])
    return [pelvis[0] + c * rel[0] - s * rel[1], pelvis[1] + s * rel[0] + c * rel[1], pelvis[2] + rel[2]]


def sm_reach_body(pelvis: List[Float64], world: List[Float64]) -> List[Float64]:
    """A point of the world, from the pelvis (heading frame)."""
    var c = cos(pelvis[3])
    var s = sin(pelvis[3])
    var dx = world[0] - pelvis[0]
    var dy = world[1] - pelvis[1]
    return [c * dx + s * dy, -s * dx + c * dy, world[2] - pelvis[2]]


struct SMReachInterface(Movable):
    """target -> z and the density of the successes it was fitted on. Empty (n = 0) until loaded: it refuses everything."""

    var n: Int
    var width: Int
    var ell: Float64
    var loop_threshold: Float64
    var cut: Float64
    var m: List[Float64]
    var s: List[Float64]
    var x: List[Float64]
    var counts: List[Float64]
    var alpha: List[Float64]

    def __init__(out self):
        self.n = 0
        self.width = 0
        self.ell = 1.0
        self.loop_threshold = 0.0
        self.cut = 0.0
        self.m = List[Float64]()
        self.s = List[Float64]()
        self.x = List[Float64]()
        self.counts = List[Float64]()
        self.alpha = List[Float64]()

    @staticmethod
    def load(path: String) raises -> Self:
        var tok = List[String]()
        var txt = string_from_bytes(read_file_bytes(path))
        for l in txt.split("\n"):
            for p in String(String(l).strip()).split(" "):
                var w = String(p)
                if w.byte_length() > 0:
                    tok.append(w)
        if len(tok) < 7 or tok[0] != "smreach":
            raise Error(path + ": not a reach interface file (no `smreach` header)")
        if tok[1] != "1":
            raise Error(path + ": reach interface file version " + tok[1] + ", this loader reads 1")
        var f = Self()
        f.n = Int(Float64(tok[2]))
        f.width = Int(Float64(tok[3]))
        f.ell = Float64(tok[4])
        f.loop_threshold = Float64(tok[5])
        f.cut = Float64(tok[6])
        if f.n <= 0 or f.width <= 0 or f.ell <= 0.0:
            raise Error(path + ": n, width and ell must be positive")
        if f.cut < f.loop_threshold:
            raise Error(path + ": the cut is under the loop's threshold")
        var want = 7 + 2 * f.width + f.n * (4 + f.width)
        if len(tok) != want:
            raise Error(path + ": " + String(len(tok)) + " numbers, " + String(want) + " for " + String(f.n) + " pairs of width " + String(f.width))
        for j in range(f.width):
            f.m.append(Float64(tok[7 + j]))
        for j in range(f.width):
            f.s.append(Float64(tok[7 + f.width + j]))
        var o = 7 + 2 * f.width
        for i in range(f.n):
            var r = o + i * (4 + f.width)
            for j in range(3):
                f.x.append(Float64(tok[r + j]))
            f.counts.append(Float64(tok[r + 3]))
            for j in range(f.width):
                f.alpha.append(Float64(tok[r + 4 + j]))
        return f^

    def loaded(self) -> Bool:
        return self.n > 0

    def density(self, t: List[Float64]) -> Float64:
        """How many successes the fit has around `t`, each weighted by the kernel."""
        var acc = 0.0
        var inv = 1.0 / (2.0 * self.ell * self.ell)
        for i in range(self.n):
            if self.counts[i] > 0.5:
                var dx = t[0] - self.x[3 * i]
                var dy = t[1] - self.x[3 * i + 1]
                var dz = t[2] - self.x[3 * i + 2]
                acc += exp(-(dx * dx + dy * dy + dz * dz) * inv)
        return acc

    def eval(self, t: List[Float64], mut z: List[Float64]):
        """z for the target `t`, into `z` (at least `width` long)."""
        var acc = List[Float64](length=self.width, fill=0.0)
        var inv = 1.0 / (2.0 * self.ell * self.ell)
        for i in range(self.n):
            var dx = t[0] - self.x[3 * i]
            var dy = t[1] - self.x[3 * i + 1]
            var dz = t[2] - self.x[3 * i + 2]
            var k = exp(-(dx * dx + dy * dy + dz * dz) * inv)
            for j in range(self.width):
                acc[j] += k * self.alpha[self.width * i + j]
        for j in range(self.width):
            z[j] = (self.m[j] + acc[j]) * self.s[j]

    def grade(self, t: List[Float64]) -> Int:
        if self.n == 0:
            return SM_REACH_REFUSED
        var r = self.density(t)
        if r >= self.cut:
            return SM_REACH_SURE
        if r >= self.loop_threshold:
            return SM_REACH_LIKELY
        return SM_REACH_REFUSED


struct SMReachLoop(Movable):
    """One reach over time. `start`, then `step` once a control step before the simulator's; `z` is the command to set
    whenever `step` returns True.

    step 0 .. ramp_steps        z goes from the z in force to the interface's, a value every `z_every` steps (min-jerk)
    from `first`, every `period` the hand's error in the world, turned into the present heading frame, is added with `gain`
                                to a correction of the commanded target (each axis within `corr_max`), and z is the
                                interface's at target + correction. A correction that takes the command out of the domain
                                is scaled back to its edge; `stuck_max` such updates in a row and the loop GIVES UP: the
                                correction is dropped, z is the interface's at the target, and it stays so (`gave`).

    The defaults are the ones the skill was measured with. Changing one is a new skill to measure."""

    var ramp_steps: Int
    var z_every: Int
    var first: Int
    var period: Int
    var gain: Float64
    var corr_max: Float64
    var stuck_max: Int

    var on: Bool
    var t: Int
    var grade: Int
    var gave: Bool
    var stuck: Int
    var target: List[Float64]
    var world: List[Float64]
    var corr: List[Float64]
    var zfrom: List[Float64]
    var zgoal: List[Float64]
    var z: List[Float64]

    def __init__(out self):
        self.ramp_steps = 50
        self.z_every = 5
        self.first = 100
        self.period = 50
        self.gain = 0.30
        self.corr_max = 0.15
        self.stuck_max = 5
        self.on = False
        self.t = 0
        self.grade = SM_REACH_REFUSED
        self.gave = False
        self.stuck = 0
        self.target = [0.0, 0.0, 0.0]
        self.world = [0.0, 0.0, 0.0]
        self.corr = [0.0, 0.0, 0.0]
        self.zfrom = List[Float64]()
        self.zgoal = List[Float64]()
        self.z = List[Float64]()

    def start(mut self, f: SMReachInterface, target: List[Float64], pelvis: List[Float64], z_now: List[Float64]) -> Int:
        """Begin a reach of `target` (from the pelvis, heading frame) from the z in force. Returns the target's grade;
        SM_REACH_REFUSED starts nothing and leaves the loop as it was."""
        var g = f.grade(target)
        if g == SM_REACH_REFUSED:
            return g
        self.on = True
        self.t = 0
        self.grade = g
        self.gave = False
        self.stuck = 0
        self.target = [target[0], target[1], target[2]]
        self.world = sm_reach_world(pelvis, target)
        self.corr = [0.0, 0.0, 0.0]
        self.zfrom = List[Float64](length=f.width, fill=0.0)
        self.zgoal = List[Float64](length=f.width, fill=0.0)
        self.z = List[Float64](length=f.width, fill=0.0)
        for k in range(f.width):
            self.zfrom[k] = z_now[k]
            self.z[k] = z_now[k]
        f.eval(self.target, self.zgoal)
        return g

    def stop(mut self):
        self.on = False

    def error(self, pelvis: List[Float64], hand: List[Float64]) -> Float64:
        """The hand's distance to the reach's world point, metres."""
        var h = sm_reach_world(pelvis, hand)
        return sqrt((h[0] - self.world[0]) ** 2 + (h[1] - self.world[1]) ** 2 + (h[2] - self.world[2]) ** 2)

    def correction(self) -> Float64:
        """The largest component of the correction in force, metres."""
        var a = 0.0
        for k in range(3):
            var c = self.corr[k] if self.corr[k] >= 0.0 else -self.corr[k]
            if c > a:
                a = c
        return a

    def step(mut self, f: SMReachInterface, pelvis: List[Float64], hand: List[Float64]) -> Bool:
        """One control step, before the simulator's. True when `z` is new and must be set."""
        if not self.on:
            return False
        var t = self.t
        var changed = False
        if t < self.ramp_steps:
            if t % self.z_every == 0:
                var u = sm_min_jerk(Float64(t + self.z_every) / Float64(self.ramp_steps))
                for k in range(f.width):
                    self.z[k] = self.zfrom[k] + u * (self.zgoal[k] - self.zfrom[k])
                changed = True
        elif self.period > 0 and not self.gave and t >= self.first and (t - self.first) % self.period == 0:
            var c = cos(pelvis[3])
            var s = sin(pelvis[3])
            var h = sm_reach_world(pelvis, hand)
            var ex = self.world[0] - h[0]
            var ey = self.world[1] - h[1]
            var eb: List[Float64] = [c * ex + s * ey, -s * ex + c * ey, self.world[2] - h[2]]
            for k in range(3):
                self.corr[k] += self.gain * eb[k]
                if self.corr[k] > self.corr_max:
                    self.corr[k] = self.corr_max
                if self.corr[k] < -self.corr_max:
                    self.corr[k] = -self.corr_max
            var tq: List[Float64] = [self.target[0] + self.corr[0], self.target[1] + self.corr[1], self.target[2] + self.corr[2]]
            if f.density(tq) < f.loop_threshold:
                # scale the correction down until the command is back inside the domain
                var lo = 0.0
                var hi = 1.0
                for _ in range(10):
                    var mid = 0.5 * (lo + hi)
                    tq = [self.target[0] + mid * self.corr[0], self.target[1] + mid * self.corr[1], self.target[2] + mid * self.corr[2]]
                    if f.density(tq) >= f.loop_threshold:
                        lo = mid
                    else:
                        hi = mid
                for k in range(3):
                    self.corr[k] = lo * self.corr[k]
                self.stuck += 1
            else:
                self.stuck = 0
            if self.stuck >= self.stuck_max:
                self.gave = True
                for k in range(3):
                    self.corr[k] = 0.0
            var tg: List[Float64] = [self.target[0] + self.corr[0], self.target[1] + self.corr[1], self.target[2] + self.corr[2]]
            f.eval(tg, self.z)
            changed = True
        self.t += 1
        return changed


struct SMZRamp(Movable):
    """z ramped from one value to another (min-jerk, a value every `z_every` steps): the way out of a reach, back to the
    stand's z. `step` once a control step before the simulator's; True when `z` is new. `on` falls when the ramp is over."""

    var steps: Int
    var z_every: Int
    var on: Bool
    var t: Int
    var zfrom: List[Float64]
    var zto: List[Float64]
    var z: List[Float64]

    def __init__(out self):
        self.steps = 50
        self.z_every = 5
        self.on = False
        self.t = 0
        self.zfrom = List[Float64]()
        self.zto = List[Float64]()
        self.z = List[Float64]()

    def start(mut self, z_now: List[Float64], z_to: List[Float64]):
        self.on = True
        self.t = 0
        self.zfrom = z_now.copy()
        self.zto = z_to.copy()
        self.z = z_now.copy()

    def step(mut self) -> Bool:
        if not self.on:
            return False
        var changed = False
        if self.t % self.z_every == 0:
            var u = sm_min_jerk(Float64(self.t + self.z_every) / Float64(self.steps))
            for k in range(len(self.z)):
                self.z[k] = self.zfrom[k] + u * (self.zto[k] - self.zfrom[k])
            changed = True
        self.t += 1
        if self.t >= self.steps:
            self.on = False
        return changed
