"""The G1 walker's L0 gates, on the CPU (G1_WALKER_PLAN §6).

    pixi run mojo build -I . examples/g1/g1_walk_eval.mojo -o build/g1_walk_eval
    ./build/g1_walk_eval --run runs/g1_walk_s1 [--recipe mix|rp] [--episodes 20] [--starts 5]

Loads `<run>/ckpt` and `<run>/obs_norm.txt` (`g1_walk_policy.WalkPolicy`),
runs the CPU env in eval mode (no noise, no pushes, stand reset, the
command written by this tool) and prints one line per gate:

  falls     random command sequences (the training distribution, redrawn
            every 2-4 s, 20 % zeros), `--episodes` x 20 s; falls counted
  tracking  (vx, wz) over the room's range, 6 s each from a stand; the
            steady-state error is the mean over the last 3 s of
            |v_body - v_cmd| (x and y) and |w - w_cmd|
  lag       0 -> 0.8 m/s: the 63 % rise time of the body-frame speed
  stop      1.0 m/s for 4 s, then 0: travel after the switch, and the time
            until the speed stays under 0.05 m/s
  hold      after that stop, 10 s more at 0 (from 1.5 s after the switch):
            drift of the pelvis in xy, and the lowest pelvis height

and the mean of every reward term over the falls rollouts (the trained
policy's per-term picture; the driver logs only the total).

⚠ GREEDY. The actor's mean, clamped +-4 — what the room will execute; the
training return was measured on Gaussian samples.
"""

from std.math import abs, sqrt, cos, sin
from std.random import seed as seed_rng, random_float64
from std.sys import argv

from noeira.core.cont_action import ContAction
from noeira.envs.robots.unitree_g1_walk import (
    UnitreeG1Walk,
    UnitreeG1WalkModel,
    g1_walk_host_reset,
    g1_walk_host_terms,
)
from noeira.envs.robots.unitree_g1_walk_config import (
    G1_WALK_N_TERMS,
    G1_WALK_OBS_DIM,
    G1W_CMD_VX,
    G1W_CMD_VY,
    G1W_CMD_WZ,
    G1W_CMD_TIMER,
    g1_walk_command,
    g1_walk_term_name,
    g1_walk_weight,
    g1_rotate_inverse,
)

from noeira.envs.robots.unitree_g1_walk_rp import (
    UnitreeG1WalkRP, g1r_host_reset, g1r_host_terms,
)
from noeira.envs.robots.unitree_g1_walk_rp_config import (
    G1R_N_TERMS, G1R_OBS_DIM, G1R_O_CMD, G1R_CMD_VX, G1R_CMD_VY, G1R_CMD_WZ,
    G1R_CMD_TIMER, g1r_term_name, g1r_weight,
)

from g1_walk_policy import WalkPolicy
from g1_walk_rp_policy import WalkRPPolicy

comptime NQ = UnitreeG1WalkModel.NQ
comptime NV = UnitreeG1WalkModel.NV
comptime ACT = UnitreeG1WalkModel.ACTION_DIM
comptime E = UnitreeG1Walk[False]
comptime ER = UnitreeG1WalkRP[False]
comptime DT_CTRL = 0.02


def _arg(name: String, default: String) raises -> String:
    var a = argv()
    for i in range(len(a) - 1):
        if String(a[i]) == name:
            return String(a[i + 1])
    return default


trait WalkSim(Movable):
    """A recipe's CPU env + its trained policy, stepped together — what the
    gates below drive."""

    def reset(mut self, seed: Int) raises:
        ...

    def command(mut self, vx: Float64, vy: Float64, wz: Float64):
        ...

    def step(mut self, log_terms: Bool) raises -> Bool:
        """One control step; True = the episode ended (fell)."""
        ...

    def body_vel(self) -> Tuple[Float64, Float64, Float64]:
        ...

    def xy(self) -> Tuple[Float64, Float64]:
        ...

    def z(self) -> Float64:
        ...

    def print_terms(self):
        ...


def _random_heading_stand(mut qpos: List[Float64]):
    var yaw = (random_float64() * 2.0 - 1.0) * 3.141592653589793
    qpos[3] = cos(yaw / 2)
    qpos[6] = sin(yaw / 2)


struct MixSim(WalkSim):
    """Runs s1-s6: the Playground / RoboParty mix (`unitree_g1_walk.mojo`)."""

    var env: E
    var pol: WalkPolicy
    var obs: List[Float64]
    var terms: Array[Float64, G1_WALK_N_TERMS]
    var term_sum: List[Float64]
    var term_n: Int

    def __init__(out self, run_dir: String) raises:
        self.env = E()
        self.pol = WalkPolicy(run_dir)
        self.obs = List[Float64](length=G1_WALK_OBS_DIM, fill=0.0)
        self.terms = Array[Float64, G1_WALK_N_TERMS](fill=0.0)
        self.term_sum = List[Float64](length=G1_WALK_N_TERMS, fill=0.0)
        self.term_n = 0

    def reset(mut self, seed: Int) raises:
        var o = self.env.reset()
        # the stand pose, with a random heading so no gate is axis-aligned
        g1_walk_host_reset(self.env.d, seed, False)
        var q = List[Float64]()
        for i in range(NQ):
            q.append(Float64(self.env.d.qpos.data[i]))
        _random_heading_stand(q)
        var v = List[Float64](length=NV, fill=0.0)
        self.env.set_state(q, v)
        self.env.current_step = 0
        _ = o
        self.pol.reset()
        self.command(0, 0, 0)
        self._read_obs()

    def command(mut self, vx: Float64, vy: Float64, wz: Float64):
        self.env.d.meta.data[G1W_CMD_VX] = vx
        self.env.d.meta.data[G1W_CMD_VY] = vy
        self.env.d.meta.data[G1W_CMD_WZ] = wz
        self.env.d.meta.data[G1W_CMD_TIMER] = -1.0
        # the observation carries the command: refresh it now
        self.obs[9] = vx
        self.obs[10] = vy
        self.obs[11] = wz

    def _read_obs(mut self):
        var o = self.env._get_obs()
        for k in range(G1_WALK_OBS_DIM):
            self.obs[k] = Float64(o.data[k])

    def step(mut self, log_terms: Bool) raises -> Bool:
        """One control step; True = fell."""
        var a = self.pol.act(self.obs)
        var act = ContAction[ACT]()
        for j in range(ACT):
            act[j] = a[j]
        var r = self.env.step(act)
        var fell = g1_walk_host_terms(self.env.d, a, self.terms)
        if log_terms:
            for t in range(G1_WALK_N_TERMS):
                self.term_sum[t] += self.terms[t]
            self.term_n += 1
        for k in range(G1_WALK_OBS_DIM):
            self.obs[k] = Float64(r[0].data[k])
        return fell

    def body_vel(self) -> Tuple[Float64, Float64, Float64]:
        """(vx, vy) in the pelvis frame and the yaw rate."""
        var lv = g1_rotate_inverse(
            Float64(self.env.d.qpos.data[3]), Float64(self.env.d.qpos.data[4]),
            Float64(self.env.d.qpos.data[5]), Float64(self.env.d.qpos.data[6]),
            Float64(self.env.d.qvel.data[0]), Float64(self.env.d.qvel.data[1]),
            Float64(self.env.d.qvel.data[2]),
        )
        return (lv[0], lv[1], Float64(self.env.d.qvel.data[5]))

    def xy(self) -> Tuple[Float64, Float64]:
        return (Float64(self.env.d.qpos.data[0]), Float64(self.env.d.qpos.data[1]))

    def z(self) -> Float64:
        return Float64(self.env.d.qpos.data[2])

    def print_terms(self):
        for t in range(G1_WALK_N_TERMS):
            var m = self.term_sum[t] / Float64(max(self.term_n, 1))
            print("    ", g1_walk_term_name(t), m, " -> ", g1_walk_weight(t) * m)


struct RPSim(WalkSim):
    """RoboParty's recipe (`unitree_g1_walk_rp.mojo`), 10-frame policy."""

    var env: ER
    var pol: WalkRPPolicy
    var obs: List[Float64]
    var terms: Array[Float64, G1R_N_TERMS]
    var term_sum: List[Float64]
    var term_n: Int

    def __init__(out self, run_dir: String) raises:
        self.env = ER()
        self.pol = WalkRPPolicy(run_dir)
        self.obs = List[Float64](length=G1R_OBS_DIM, fill=0.0)
        self.terms = Array[Float64, G1R_N_TERMS](fill=0.0)
        self.term_sum = List[Float64](length=G1R_N_TERMS, fill=0.0)
        self.term_n = 0

    def reset(mut self, seed: Int) raises:
        _ = self.env.reset()
        g1r_host_reset(self.env.d, seed, False)
        var q = List[Float64]()
        for i in range(NQ):
            q.append(Float64(self.env.d.qpos.data[i]))
        _random_heading_stand(q)
        var v = List[Float64](length=NV, fill=0.0)
        self.env.set_state(q, v)
        self.env.current_step = 0
        self.pol.reset()
        self.command(0, 0, 0)
        var o = self.env._get_obs()
        for k in range(G1R_OBS_DIM):
            self.obs[k] = Float64(o.data[k])
        self.obs[G1R_O_CMD] = 0
        self.obs[G1R_O_CMD + 1] = 0
        self.obs[G1R_O_CMD + 2] = 0

    def command(mut self, vx: Float64, vy: Float64, wz: Float64):
        self.env.d.meta.data[G1R_CMD_VX] = vx
        self.env.d.meta.data[G1R_CMD_VY] = vy
        self.env.d.meta.data[G1R_CMD_WZ] = wz
        self.env.d.meta.data[G1R_CMD_TIMER] = -1.0
        self.obs[G1R_O_CMD] = vx
        self.obs[G1R_O_CMD + 1] = vy
        self.obs[G1R_O_CMD + 2] = wz

    def step(mut self, log_terms: Bool) raises -> Bool:
        var a = self.pol.act(self.obs)
        var act = ContAction[ACT]()
        for j in range(ACT):
            act[j] = a[j]
        var r = self.env.step(act)
        var fell = g1r_host_terms(self.env.d, a, self.terms)
        if log_terms:
            for t in range(G1R_N_TERMS):
                self.term_sum[t] += self.terms[t]
            self.term_n += 1
        for k in range(G1R_OBS_DIM):
            self.obs[k] = Float64(r[0].data[k])
        return fell

    def body_vel(self) -> Tuple[Float64, Float64, Float64]:
        var lv = g1_rotate_inverse(
            Float64(self.env.d.qpos.data[3]), Float64(self.env.d.qpos.data[4]),
            Float64(self.env.d.qpos.data[5]), Float64(self.env.d.qpos.data[6]),
            Float64(self.env.d.qvel.data[0]), Float64(self.env.d.qvel.data[1]),
            Float64(self.env.d.qvel.data[2]),
        )
        return (lv[0], lv[1], Float64(self.env.d.qvel.data[5]))

    def xy(self) -> Tuple[Float64, Float64]:
        return (Float64(self.env.d.qpos.data[0]), Float64(self.env.d.qpos.data[1]))

    def z(self) -> Float64:
        return Float64(self.env.d.qpos.data[2])

    def print_terms(self):
        for t in range(G1R_N_TERMS):
            var m = self.term_sum[t] / Float64(max(self.term_n, 1))
            print("    ", g1r_term_name(t), m, " -> ", g1r_weight(t) * m)


def gate_falls[S: WalkSim](mut sim: S, episodes: Int) raises -> Int:
    var falls = 0
    for ep in range(episodes):
        sim.reset(1000 + ep)
        var left = 0
        for k in range(1000):
            if left <= 0:
                var c = g1_walk_command(random_float64(), random_float64(),
                                        random_float64(), random_float64())
                sim.command(c[0], c[1], c[2])
                left = 100 + Int(random_float64() * 100.0)
            left -= 1
            if sim.step(True):
                falls += 1
                print("    fall: episode", ep, "step", k)
                break
    return falls


def gate_tracking[S: WalkSim](mut sim: S, starts: Int) raises:
    var vxs: List[Float64] = [0.0, 0.3, 0.6, 1.0]
    var wzs: List[Float64] = [-1.0, 0.0, 1.0]
    var worst_v = 0.0
    var worst_w = 0.0
    var sum_v = 0.0
    var sum_w = 0.0
    var n = 0
    var falls = 0
    print("    vx_cmd wz_cmd |  vx   vy   wz (steady mean)  | err_v err_w")
    for vx in vxs:
        for wz in wzs:
            var ev = 0.0
            var ew = 0.0
            var mvx = 0.0
            var mvy = 0.0
            var mwz = 0.0
            var cnt = 0
            for s in range(starts):
                sim.reset(2000 + s)
                sim.command(vx, 0.0, wz)
                for k in range(300):
                    if sim.step(False):
                        falls += 1
                        break
                    if k >= 150:
                        var b = sim.body_vel()
                        ev += sqrt((b[0] - vx) ** 2 + b[1] ** 2)
                        ew += abs(b[2] - wz)
                        mvx += b[0]
                        mvy += b[1]
                        mwz += b[2]
                        cnt += 1
            if cnt == 0:
                continue
            var c = Float64(cnt)
            print("    ", vx, " ", wz, " | ", mvx / c, " ", mvy / c, " ", mwz / c,
                  " | ", ev / c, " ", ew / c)
            worst_v = max(worst_v, ev / c)
            worst_w = max(worst_w, ew / c)
            sum_v += ev / c
            sum_w += ew / c
            n += 1
    print("  TRACKING mean err_v", sum_v / Float64(max(n, 1)), "worst", worst_v,
          "| mean err_w", sum_w / Float64(max(n, 1)), "worst", worst_w,
          "| falls", falls, "  (bars: v < 0.1, w < 0.15)")


def gate_lag_stop_hold[S: WalkSim](mut sim: S, starts: Int) raises:
    var rise_sum = 0.0
    var travel_sum = 0.0
    var settle_sum = 0.0
    var drift_sum = 0.0
    var drift_worst = 0.0
    var zmin_worst = 10.0
    var falls = 0
    var n = 0
    for s in range(starts):
        sim.reset(3000 + s)
        # lag: 2 s stand, then 0.8 m/s
        var fell = False
        for _ in range(100):
            if sim.step(False):
                fell = True
                break
        sim.command(0.8, 0.0, 0.0)
        var rise = -1.0
        for k in range(150):
            if fell or sim.step(False):
                fell = True
                break
            var b = sim.body_vel()
            if rise < 0 and b[0] >= 0.63 * 0.8:
                rise = Float64(k + 1) * DT_CTRL
        # stop: 1.0 m/s for 4 s, then 0
        sim.command(1.0, 0.0, 0.0)
        for _ in range(200):
            if fell or sim.step(False):
                fell = True
                break
        var p0 = sim.xy()
        sim.command(0.0, 0.0, 0.0)
        var settle = -1.0
        var p_hold = p0
        var zmin = 10.0
        for k in range(575):            # 1.5 s + 10 s
            if fell or sim.step(False):
                fell = True
                break
            var b = sim.body_vel()
            var sp = sqrt(b[0] ** 2 + b[1] ** 2)
            if sp >= 0.05:
                settle = -1.0
            elif settle < 0:
                settle = Float64(k + 1) * DT_CTRL
            if k == 74:
                p_hold = sim.xy()
            if k >= 75:
                zmin = min(zmin, sim.z())
        if fell:
            falls += 1
            print("    start", s, "fell")
            continue
        var p1 = sim.xy()
        # travel after the switch: up to the start of the hold window
        var travel = sqrt((p_hold[0] - p0[0]) ** 2 + (p_hold[1] - p0[1]) ** 2)
        var drift = sqrt((p1[0] - p_hold[0]) ** 2 + (p1[1] - p_hold[1]) ** 2)
        print("    start", s, "| rise", rise, "s | stop travel", travel,
              "m, settled at", settle, "s | hold drift", drift, "m, min pelvis z", zmin)
        rise_sum += rise
        travel_sum += travel
        settle_sum += settle
        drift_sum += drift
        drift_worst = max(drift_worst, drift)
        zmin_worst = min(zmin_worst, zmin)
        n += 1
    var c = Float64(max(n, 1))
    print("  LAG rise (63 %)", rise_sum / c, "s  (bar < 0.4)")
    print("  STOP travel", travel_sum / c, "m, settle", settle_sum / c,
          "s  (bars < 0.4 m, < 1.5 s)")
    print("  HOLD drift mean", drift_sum / c, "worst", drift_worst,
          "m, min pelvis z", zmin_worst, "  (bars < 0.05 m, > 0.7 m)")
    print("  falls in these sequences", falls, "of", starts)


def run_gates[S: WalkSim](mut sim: S, episodes: Int, starts: Int) raises:
    print("falls (random command sequences, 20 s)")
    var falls = gate_falls(sim, episodes)
    print("  FALLS", falls, "of", episodes, "  (bar 0)")
    print("  reward terms, mean per step over those rollouts (weight x mean):")
    sim.print_terms()
    print("tracking")
    gate_tracking(sim, starts)
    print("lag / stop / hold")
    gate_lag_stop_hold(sim, starts)


def main() raises:
    var run = _arg("--run", "runs/g1_walk_s1")
    var recipe = _arg("--recipe", "mix")
    var episodes = Int(_arg("--episodes", "20"))
    var starts = Int(_arg("--starts", "5"))
    seed_rng(Int(_arg("--seed", "1")))
    print("G1 walker eval | run", run, "| recipe", recipe, "| episodes", episodes,
          "| starts", starts)
    if recipe == "rp":
        var sim = RPSim(run)
        run_gates(sim, episodes, starts)
    elif recipe == "mix":
        var sim = MixSim(run)
        run_gates(sim, episodes, starts)
    else:
        raise Error("--recipe must be mix or rp")
