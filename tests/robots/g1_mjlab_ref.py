"""Layer-1 gate for the mjlab G1: our baked XML vs mjlab's model as mjlab edits it, both via MuJoCo.

    pixi run python tests/robots/g1_mjlab_ref.py

The reference side is mjlab's `xmls/g1.xml` compiled through `MjSpec` with
the edits mjlab makes at RUNTIME re-done here from `g1_constants.py` — a
SECOND transcription, structured as mjlab structures it (one entry per
`BuiltinPositionActuatorCfg`, `target_names_expr` tuples, the
`FULL_COLLISION` regex dicts), sharing no table with `g1_mjlab_bake.py`:

  * `FULL_COLLISION`: `.*_collision` geoms contype 1 / conaffinity 1, condim
    {feet: 3, rest: 1}, priority {feet: 1}, friction {feet: (0.6,)};
  * every hinge's `armature` = its actuator's reflected inertia;
  * the terrain: a plane, MuJoCo defaults, size (0, 0, 0.01);
  * `MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)` (the
    reproduction's), integrator EULER — the bake's one deviation (mjlab
    integrates `implicitfast`), applied to the reference so the rest of
    `opt` is compared;
  * the `stand` key from `KNEES_BENT_KEYFRAME`;
  * a torque motor per hinge, ctrlrange ± its actuator's effort limit (mjlab
    has `<position>` servos; ours are motors driven by the env's PD — the
    same deviation, so the motor tables are compared against the effort
    limits rather than skipped).

Every `mjModel` table is then diffed at tolerance 0.0 except the sensor
tables (the bake drops the block; nothing reads it). Neither our parser nor
our engine is in the loop: a mismatch isolates the XML TEXT the bake wrote.

⚠ `references/` IS GITIGNORED. Absent the reference tree this SKIPS.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tests", "dm_control"))

import mjmodel_diff  # noqa: E402

REF_XML = os.path.join(REPO, "references", "mjlab-8bb2bc9", "src", "mjlab", "asset_zoo", "robots", "unitree_g1", "xmls", "g1.xml")
OURS = os.path.join(REPO, "noeira", "envs", "robots", "assets", "unitree_g1_mjlab.xml")
SENSOR_TABLES = ("sensor_type", "sensor_objid", "sensor_adr", "sensor_dim")


def _two_stage(rotor, gear):
    # `reflected_inertia_from_two_stage_planetary`
    return rotor[0] * (gear[1] * gear[2]) ** 2 + rotor[1] * gear[2] ** 2 + rotor[2]


_A = {
    "5020": (_two_stage((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + (46 / 18), 1 + (56 / 16))), 25.0),
    "7520_14": (_two_stage((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + (48 / 22))), 88.0),
    "7520_22": (_two_stage((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)), 139.0),
    "4010": (_two_stage((0.068e-4, 0.0, 0.0), (1, 5, 5)), 5.0),
}
# (target_names_expr, armature, effort_limit) per `G1_ARTICULATION` actuator
ACTUATORS = [
    ((".*_elbow_joint", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_wrist_roll_joint"),
     _A["5020"][0], _A["5020"][1]),
    ((".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"), _A["7520_14"][0], _A["7520_14"][1]),
    ((".*_hip_roll_joint", ".*_knee_joint"), _A["7520_22"][0], _A["7520_22"][1]),
    ((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), _A["4010"][0], _A["4010"][1]),
    (("waist_pitch_joint", "waist_roll_joint"), _A["5020"][0] * 2, _A["5020"][1] * 2),
    ((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), _A["5020"][0] * 2, _A["5020"][1] * 2),
]
KNEES_BENT = {
    ".*_hip_pitch_joint": -0.312, ".*_knee_joint": 0.669, ".*_ankle_pitch_joint": -0.363, ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2,
}
CONDIM = {r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1}
PRIORITY = {r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0}
FRICTION = {r"^(left|right)_foot[1-7]_collision$": (0.6,)}


def _first(d, name, default=None):
    for pat, v in d.items():
        if re.fullmatch(pat, name) or re.match(pat, name):
            return v
    return default


def _actuator(name):
    hits = [(a, e) for exprs, a, e in ACTUATORS if any(re.fullmatch(x, name) for x in exprs)]
    assert len(hits) == 1, (name, hits)
    return hits[0]


class ReferencePD:
    """mjlab's `BuiltinPositionActuator` and `JointPositionAction`, from THIS
    file's transcription (not the bake's): target = default + 0.25 · effort /
    kp · a (unclipped), torque = kp (target − q) − kd q̇ clipped to ± effort,
    re-evaluated every substep. kp = armature · ω², kd = 2 · 2 · armature · ω,
    ω = 2π · 10 Hz. For `test_unitree_g1_mjlab_vs_mujoco`."""

    def __init__(self):
        import mujoco
        import numpy as np
        m = mujoco.MjModel.from_xml_path(OURS)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
        w = 10 * 2.0 * 3.1415926535
        arm = np.array([_actuator(n)[0] for n in names])
        self.kp = arm * w ** 2
        self.kd = 2.0 * 2.0 * arm * w
        self.effort = np.array([_actuator(n)[1] for n in names])
        self.default = np.array([next((v for pat, v in KNEES_BENT.items() if re.fullmatch(pat, n)), 0.0) for n in names])
        self.scale = 0.25 * self.effort / self.kp

    def stand_state(self, mujoco, m, d):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)

    def control_step(self, mujoco, m, d, action, decimation=4):
        import numpy as np
        target = self.default + self.scale * np.asarray(action, dtype=np.float64)
        worst = 0
        for _ in range(decimation):
            d.ctrl[:] = np.clip(self.kp * (target - d.qpos[7:]) - self.kd * d.qvel[6:], -self.effort, self.effort)
            mujoco.mj_step(m, d)
            worst = max(worst, int(d.ncon))
        return worst


def reference():
    import mujoco
    sp = mujoco.MjSpec.from_file(REF_XML)
    for g in sp.geoms:
        if not g.name or not re.fullmatch(".*_collision", g.name):
            continue
        g.contype = 1
        g.conaffinity = 1
        g.condim = _first(CONDIM, g.name)
        g.priority = _first(PRIORITY, g.name, 0)
        f = _first(FRICTION, g.name)
        if f is not None:
            g.friction = [f[0], g.friction[1], g.friction[2]]
    hinges = [j for j in sp.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    for j in hinges:
        j.armature = _actuator(j.name)[0]
    # the terrain first in the worldbody, as the bake writes it
    sp.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
    sp.option.timestep = 0.005
    sp.option.iterations = 10
    sp.option.ls_iterations = 20
    sp.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    q = [0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0]
    for j in hinges:
        q.append(next((v for pat, v in KNEES_BENT.items() if re.fullmatch(pat, j.name)), 0.0))
    sp.add_key(name="stand", qpos=q)
    for j in hinges:
        e = _actuator(j.name)[1]
        a = sp.add_actuator(name=j.name[: -len("_joint")], target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.ctrllimited = 1
        a.ctrlrange = [-e, e]
    for s in list(sp.sensors):
        sp.delete(s)
    return sp.compile()


def main():
    os.chdir(REPO)
    if not os.path.exists(REF_XML):
        print("SKIP: reference tree absent —", REF_XML)
        return 0
    import mujoco
    ref = reference()
    ours = mujoco.MjModel.from_xml_path(OURS)
    bad = mjmodel_diff.diff_models(ref, ours, skip_tables=SENSOR_TABLES)
    n = mjmodel_diff.n_tables() - len(SENSOR_TABLES)
    print("unitree_g1_mjlab: {} tables compared, {} counts, {} option fields".format(
        n, len(mjmodel_diff._COUNTS), len(mjmodel_diff._OPTS)))
    print("  mass ours {:.4f} ref {:.4f}; ls_iterations ours {} ref {}".format(
        ours.body_mass.sum(), ref.body_mass.sum(), ours.opt.ls_iterations, ref.opt.ls_iterations))
    if ours.opt.ls_iterations != ref.opt.ls_iterations:
        bad.append("ls_iterations")
    if bad:
        print("FAIL — {} mismatch(es):".format(len(bad)))
        for b in bad[:40]:
            print("  ", b)
        return 1
    print("PASS — unitree_g1_mjlab.xml compiles to mjlab's runtime model exactly"
          " (sensors excluded by design; integrator Euler on both by the stated deviation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
