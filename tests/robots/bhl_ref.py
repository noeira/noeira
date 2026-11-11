"""Berkeley Humanoid Lite — the reference controller, and what our 2 ms step costs.

    pixi run python tests/robots/bhl_ref.py

`ReferencePD` is a SEPARATE transcription of BHL's actuator law (from its
`policy_humanoid.yaml`: arms first, then legs; not from the bake's parse of
the Isaac config), for `test_berkeley_humanoid_lite_vs_mujoco`.

Run as a script, it measures in MuJoCo, on our baked file, the distance
between a rollout at our 2 ms and the same rollout at BHL's 0.5 ms.
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OURS = os.path.join(REPO, "noeira", "envs", "robots", "assets", "berkeley_humanoid_lite.xml")

# policy_humanoid.yaml: 10 arm joints, then 12 leg joints.
KP = [10.0] * 10 + [20.0] * 12
KD = [2.0] * 22
EFFORT = [4.0] * 10 + [6.0] * 12
SPEED = [10.0] * 22
DEFAULT = [0.0] * 10 + [0.0, 0.0, -0.2, 0.4, -0.3, 0.0] * 2
SCALE = 0.25


class ReferencePD:
    """target = default + 0.25 · a (unclipped); torque = kp (target − q) −
    kd q̇ clipped to ± effort, and 0 where it would push a joint already
    beyond its speed limit; re-evaluated every substep."""

    def __init__(self):
        import numpy as np
        self.kp, self.kd = np.array(KP), np.array(KD)
        self.effort, self.speed, self.default = np.array(EFFORT), np.array(SPEED), np.array(DEFAULT)

    def stand_state(self, mujoco, m, d):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)

    def torque(self, target, q, qd):
        import numpy as np
        tau = np.clip(self.kp * (target - q) - self.kd * qd, -self.effort, self.effort)
        tau[(qd > self.speed) & (tau > 0)] = 0.0
        tau[(qd < -self.speed) & (tau < 0)] = 0.0
        return tau

    def control_step(self, mujoco, m, d, action, decimation=10):
        import numpy as np
        target = self.default + SCALE * np.asarray(action, dtype=np.float64)
        worst = 0
        for _ in range(decimation):
            d.ctrl[:] = self.torque(target, d.qpos[7:], d.qvel[6:])
            mujoco.mj_step(m, d)
            worst = max(worst, int(d.ncon))
        return worst


def main():
    import mujoco
    import numpy as np

    pd = ReferencePD()
    for label, amp in (("stand", 0.0), ("driven 0.3", 0.3), ("driven 1.0", 1.0)):
        runs = {}
        for dt, dec in ((0.002, 10), (0.0005, 40)):
            m = mujoco.MjModel.from_xml_path(OURS)
            m.opt.timestep = dt
            d = mujoco.MjData(m)
            pd.stand_state(mujoco, m, d)
            traj = []
            for t in range(100):
                a = [amp * np.sin(t * 0.23 + j * 0.61) for j in range(22)]
                pd.control_step(mujoco, m, d, a, dec)
                traj.append(d.qpos.copy())
            runs[dt] = np.array(traj)
        diff = np.abs(runs[0.002] - runs[0.0005])
        print("%-11s 2 ms against 0.5 ms, worst |dqpos|: after 0.2 s %.2e, 0.5 s %.2e, 1 s %.2e, 2 s %.2e; "
              "root z at 2 s %.3f / %.3f; finite %s"
              % (label, diff[:10].max(), diff[:25].max(), diff[:50].max(), diff.max(),
                 runs[0.002][-1, 2], runs[0.0005][-1, 2], bool(np.isfinite(runs[0.002]).all())))


if __name__ == "__main__":
    main()
