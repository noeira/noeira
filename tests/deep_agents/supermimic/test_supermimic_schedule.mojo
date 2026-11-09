"""SuperMimic's training schedule: the lr and a resumed run (`agent.mojo`).

    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_schedule.mojo

Answers known by construction, no reference implementation.

WHAT EACH SECTION CAN CATCH
===========================
[1] the cosine (`schedule.cosine_lr`): base at 0, 3 % of it from the end of
    its decay (250 / 320 of the run) on, the midpoint between.
[2] realmimic's warm restart (`training/schedule.py` `scheduled_lr`): 0.3 x
    base at the restart, 0.03 x base after its decay, the midpoint 0.165 x.
[3] a resumed agent: it counts on from the checkpoint's iteration to the
    run's total (not by it), keeps the checkpoint's update count while the
    warm-up runs again (its replay is empty), and takes the restarted lr;
    without a restart it stays on the first cosine.
"""

from std.math import abs

from noeira.deep_agents.supermimic.agent import sm_agent, sm_cosine_lr, sm_restart_lr
from noeira.deep_agents.supermimic.config import SMConfig
from noeira.deep_agents.supermimic.prompt_eval import sm_stand_library
from noeira.deep_agents.supermimic.robot import G1MjlabRobot


comptime R = G1MjlabRobot
comptime C = SMConfig["cpu", R, 32, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 8, 8, 8, 8]


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _near(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= 1e-12 * max(abs(b), 1e-30)


def main() raises:
    var t = Tally()
    comptime BASE = 2e-4

    print("\n[1] the cosine")
    t.truth(_near(sm_cosine_lr(0, 320000, BASE), BASE), "base at step 0")
    t.truth(_near(sm_cosine_lr(125000, 320000, BASE), BASE * (0.03 + 0.97 * 0.5)), "the midpoint of its 250k decay")
    t.truth(_near(sm_cosine_lr(250000, 320000, BASE), 0.03 * BASE) and _near(sm_cosine_lr(319999, 320000, BASE), 0.03 * BASE), "3 % from 250k on")

    print("\n[2] the warm restart")
    t.truth(_near(sm_restart_lr(0, 30000, BASE, 0.3), 0.3 * BASE), "0.3 x base at the restart")
    t.truth(_near(sm_restart_lr(15000, 30000, BASE, 0.3), 0.165 * BASE), "0.165 x base halfway")
    t.truth(_near(sm_restart_lr(30000, 30000, BASE, 0.3), 0.03 * BASE) and _near(sm_restart_lr(90000, 30000, BASE, 0.3), 0.03 * BASE), "0.03 x base after its decay")

    print("\n[3] a resumed agent")
    var sim0 = R.SIM.create()
    var agent = sm_agent[C](sm_stand_library[R](2, 200, 0.05, sim0.qpos(2)), 2000, False, 0)
    agent.resume(1000, 400, 0.3, 500)
    agent.train(1004, 0)
    t.truth(agent.iteration == 1004, "it counts on from 1000 to the total 1004, not by it (" + String(agent.iteration) + ")")
    t.truth(agent.updates == 400, "and keeps 400 updates while the warm-up runs again (" + String(agent.updates) + ")")
    t.truth(_near(agent._lr(), sm_restart_lr(4, 500, agent.sched.lr, 0.3)), "its lr is the restart's, 4 steps in")
    var plain = sm_agent[C](sm_stand_library[R](2, 200, 0.05, sim0.qpos(2)), 2000, False, 0)
    plain.resume(1000, 400)
    t.truth(_near(plain._lr(), sm_cosine_lr(1000, 2000, plain.sched.lr)), "without a restart it stays on the first cosine")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_schedule: " + String(t.fails) + " failed")
