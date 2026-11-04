"""Bake the Unitree G1 as mjlab models it (SuperMimic's reproduction trains on it).

    pixi run python tests/robots/g1_mjlab_bake.py            # write the assets
    pixi run python tests/robots/g1_mjlab_bake.py --check    # CI: fail on any diff

WHAT IT EMITS, and from where (mjlab at commit 8bb2bc9,
`references/mjlab-8bb2bc9/src/mjlab/asset_zoo/robots/unitree_g1/`):

    noeira/envs/robots/assets/unitree_g1_mjlab.xml
        `xmls/g1.xml` (the G1 rev 1.0 body: torso and waist meshes
        `*_rev_1_0.STL`, 33.34 kg) with what mjlab adds at runtime written
        into the file:
          * the collision set `FULL_COLLISION` (`g1_constants.py`): every
            `*_collision` geom contype 1 / conaffinity 1 (self-collisions
            on), condim 1, except the 14 foot capsules: condim 3, priority 1,
            friction 0.6;
          * each hinge's armature, the reflected rotor inertia of its motor
            (`reflected_inertia_from_two_stage_planetary`);
          * the ground: a plane with MuJoCo's default geom properties
            (`TerrainEntityCfg(terrain_type="plane")`, size 0 0 0.01);
          * the reproduction's solver settings (`SuperMimic-Reproduced/src/
            supermimic/envs/tracking_env_cfg.py`): timestep 0.005, Newton,
            10 iterations, 20 line-search iterations (mjlab's `MujocoCfg`
            defaults for the rest: pyramidal cone, impratio 1);
          * the `stand` keyframe, mjlab's `KNEES_BENT_KEYFRAME` (the
            entity's `init_state`, hence the action offset);
          * 29 torque `<motor>`s, gear 1, ctrlrange ± the effort limit.
        The `<sensor>` block is dropped (nothing reads it).
    noeira/envs/robots/assets/unitree_g1/*_rev_1_0.STL
        The three rev-1.0 meshes our BFM-Zero G1 does not have; every other
        mesh mjlab names is byte-identical to ours (checked here).
    noeira/envs/robots/unitree_g1_mjlab_pd.mojo
        mjlab's actuator model per DoF in the model's joint order: kp =
        armature · ω², kd = 2 · ζ · armature · ω (ω = 2π · 10 Hz, ζ = 2),
        effort limit, armature, the default angle (the keyframe), and the
        action scale 0.25 · effort / kp (`G1_ACTION_SCALE`).

⚠ TWO DEVIATIONS FROM mjlab, both measured (docs/SUPERMIMIC_PORT_PLAN.md,
2026-11-04):
  1. mjlab's actuators are `<position kp kv forcerange>` servos integrated
     with `implicitfast`; ours are motors whose torque the ENV computes every
     substep (`kp (target − q) − kd q̇`, clipped to ± effort) under Euler,
     the integrator physics3d's batched path runs. In MuJoCo on this model,
     from 48 SEED clip starts tracking the reference: worst-body distance
     0.3 mm at 0.1 s, 1.8 mm at 0.5 s (median), 7 mm (p90).
  2. none: everything else is the file mjlab simulates.

⚠ `references/` IS GITIGNORED. This is a local tool; the generated files are
what ships.
"""

import math
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(REPO, "references", "mjlab-8bb2bc9", "src", "mjlab", "asset_zoo", "robots", "unitree_g1")
REF_XML = os.path.join(REF, "xmls", "g1.xml")
REF_MESH = os.path.join(REF, "xmls", "assets")

ASSET_DIR = os.path.join(REPO, "noeira", "envs", "robots", "assets")
ASSET_XML = os.path.join(ASSET_DIR, "unitree_g1_mjlab.xml")
MESH_DIR = os.path.join(ASSET_DIR, "unitree_g1")
PD_MOJO = os.path.join(REPO, "noeira", "envs", "robots", "unitree_g1_mjlab_pd.mojo")

TIMESTEP = 0.005
ITERATIONS = 10
LS_ITERATIONS = 20
CONTROL_DECIMATION = 4

# ── mjlab's actuator model (`g1_constants.py`, `utils/actuator.py`) ──


def reflected_inertia(rotor, gears):
    assert gears[0] == 1
    return rotor[0] * (gears[1] * gears[2]) ** 2 + rotor[1] * gears[2] ** 2 + rotor[2]


ARM_5020 = reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
ARM_7520_14 = reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
ARM_7520_22 = reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
ARM_4010 = reflected_inertia((0.068e-4, 0.0, 0.0), (1, 5, 5))
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

# (regex over joint names, armature, effort limit) — the six actuator groups
GROUPS = [
    (r".*_elbow_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint|.*_shoulder_yaw_joint|.*_wrist_roll_joint", ARM_5020, 25.0),
    (r".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint", ARM_7520_14, 88.0),
    (r".*_hip_roll_joint|.*_knee_joint", ARM_7520_22, 139.0),
    (r".*_wrist_pitch_joint|.*_wrist_yaw_joint", ARM_4010, 5.0),
    # waist pitch/roll and ankles: two 5020s through a linkage, nominal 1:1
    (r"waist_pitch_joint|waist_roll_joint", 2 * ARM_5020, 2 * 25.0),
    (r".*_ankle_pitch_joint|.*_ankle_roll_joint", 2 * ARM_5020, 2 * 25.0),
]

# `KNEES_BENT_KEYFRAME`
INIT_POS = (0.0, 0.0, 0.76)
INIT_JOINTS = [
    (r".*_hip_pitch_joint", -0.312),
    (r".*_knee_joint", 0.669),
    (r".*_ankle_pitch_joint", -0.363),
    (r".*_elbow_joint", 0.6),
    (r"left_shoulder_roll_joint", 0.2),
    (r"left_shoulder_pitch_joint", 0.2),
    (r"right_shoulder_roll_joint", -0.2),
    (r"right_shoulder_pitch_joint", 0.2),
]

FOOT_RE = r"^(left|right)_foot[1-7]_collision$"
FOOT_FRICTION = 0.6
REV_MESHES = ["torso_link_rev_1_0.STL", "waist_roll_link_rev_1_0.STL", "waist_yaw_link_rev_1_0.STL"]


def actuator(name):
    hits = [(a, e) for pat, a, e in GROUPS if re.fullmatch(pat, name)]
    if len(hits) != 1:
        raise SystemExit("joint %s matches %d actuator groups" % (name, len(hits)))
    a, e = hits[0]
    return a, a * NATURAL_FREQ ** 2, 2.0 * DAMPING_RATIO * a * NATURAL_FREQ, e


def default_pos(name):
    hits = [v for pat, v in INIT_JOINTS if re.fullmatch(pat, name)]
    return hits[0] if hits else 0.0


def _need(path):
    if not os.path.exists(path):
        raise SystemExit("missing reference file: %s (references/ is gitignored, fetch it locally)" % path)


def joint_names(xml):
    return re.findall(r'<joint name="([a-z_]+_joint)"', xml)


def bake_xml():
    _need(REF_XML)
    src = open(REF_XML).read()
    out = src
    # the meshes: our shared mesh directory
    out = out.replace('meshdir="assets"', 'meshdir="unitree_g1"', 1)
    # the collision set: condim 1 on every collision geom, the feet condim 3 / priority 1 / friction 0.6
    out = out.replace(
        '<geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1"/>',
        '<geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1" condim="1"/>', 1)
    out = out.replace(
        '<geom type="capsule" size="0.01"/>',
        '<geom type="capsule" size="0.01" condim="3" priority="1" friction="%g 0.005 0.0001"/>' % FOOT_FRICTION, 1)
    # every foot capsule must be in that class
    for m in re.finditer(r'<geom name="([a-z0-9_]+)"([^>]*)>', out):
        name, attrs = m.group(1), m.group(2)
        if re.match(FOOT_RE, name) and 'class="foot_capsule"' not in attrs:
            raise SystemExit("foot geom %s is not in class foot_capsule" % name)
        if name.endswith("_collision") and "condim" in attrs:
            raise SystemExit("collision geom %s overrides condim" % name)
    # armature per hinge
    names = joint_names(out)
    if len(names) != 29:
        raise SystemExit("expected 29 hinges, found %d" % len(names))
    for n in names:
        a = actuator(n)[0]
        pat = '<joint name="%s"' % n
        if out.count(pat) != 1:
            raise SystemExit("joint %s appears %d times" % (n, out.count(pat)))
        out = out.replace(pat, '<joint name="%s" armature="%.17g"' % (n, a))
    # no sensors
    out = re.sub(r"\n  <sensor>.*?</sensor>", "", out, flags=re.S)
    if "<sensor" in out:
        raise SystemExit("sensor block not removed")
    # option, ground, keyframe, motors
    head = (
        "<!-- GENERATED by tests/robots/g1_mjlab_bake.py from mjlab 8bb2bc9 — do not edit. -->\n"
    )
    out = head + out
    opt = (
        '\n  <!-- the reproduction\'s MujocoCfg; Euler, not mjlab\'s implicitfast: see the bake script -->\n'
        '  <option timestep="%g" iterations="%d" ls_iterations="%d"/>\n' % (TIMESTEP, ITERATIONS, LS_ITERATIONS)
    )
    out = out.replace("autolimits=\"true\"/>\n", "autolimits=\"true\"/>\n" + opt, 1)
    ground = '  <worldbody>\n    <geom name="floor" type="plane" size="0 0 0.01"/>\n'
    assert out.count("  <worldbody>\n") == 1
    out = out.replace("  <worldbody>\n", ground, 1)
    q = []
    for n in names:
        q.append("%.17g" % default_pos(n))
    key = '\n  <keyframe>\n    <key name="stand" qpos="%g %g %g 1 0 0 0 %s"/>\n  </keyframe>\n' % (
        INIT_POS[0], INIT_POS[1], INIT_POS[2], " ".join(q))
    motors = ["\n  <actuator>"]
    for n in names:
        e = actuator(n)[3]
        motors.append('    <motor name="%s" joint="%s" ctrlrange="%g %g"/>' % (n[: -len("_joint")], n, -e, e))
    motors.append("  </actuator>\n")
    assert out.rstrip().endswith("</mujoco>")
    out = out.rstrip()[: -len("</mujoco>")] + key + "\n".join(motors) + "</mujoco>\n"
    return out, names


def bake_pd(names):
    rows = [actuator(n) + (default_pos(n),) for n in names]
    L = []
    L.append('"""Unitree G1 as mjlab models it — PD tables, GENERATED, DO NOT EDIT.\n')
    L.append("Regenerate with:  pixi run python tests/robots/g1_mjlab_bake.py")
    L.append("CI checks it with: pixi run python tests/robots/g1_mjlab_bake.py --check\n")
    L.append("Source of truth is mjlab 8bb2bc9 `asset_zoo/robots/unitree_g1/g1_constants.py`:")
    L.append("kp = armature · (2π · 10 Hz)², kd = 2 · 2 · armature · (2π · 10 Hz), the")
    L.append("effort limit of the joint's motor group, the reflected rotor inertia as the")
    L.append("armature (also written into the XML), and `KNEES_BENT_KEYFRAME` as the default")
    L.append("angle. The torque law (mjlab's `BuiltinPositionActuator`, computed in the env):\n")
    L.append("    target = q_default + G1M_ACTION_SCALE(i) · a          (no clip)")
    L.append("    tau    = kp (target − q) − kd q̇, clipped to ± effort, EVERY substep")
    L.append('"""\n')
    L.append("comptime G1M_N_DOF: Int = %d" % len(names))
    L.append("comptime G1M_SIM_TIMESTEP: Float64 = %g" % TIMESTEP)
    L.append("comptime G1M_CONTROL_DECIMATION: Int = %d" % CONTROL_DECIMATION)
    L.append("comptime G1M_INIT_ROOT_Z: Float64 = %g\n" % INIT_POS[2])

    def table(fn, doc, col):
        L.append("\n@always_inline")
        L.append("def %s(i: Int) -> Float64:" % fn)
        L.append('    """%s"""' % doc)
        for i, (n, r) in enumerate(zip(names, rows)):
            kw = "if" if i == 0 else "elif"
            L.append("    %s i == %d:  # %s" % (kw, i, n))
            L.append("        return %r" % float(col(r)))
        L.append("    return 0.0")

    L.append("\n@always_inline")
    L.append("def g1m_dof_name(i: Int) -> StaticString:")
    for i, n in enumerate(names):
        kw = "if" if i == 0 else "elif"
        L.append("    %s i == %d:" % (kw, i))
        L.append('        return "%s"' % n)
    L.append('    return ""')
    table("g1m_kp", "Stiffness, N·m/rad.", lambda r: r[1])
    table("g1m_kd", "Damping, N·m·s/rad.", lambda r: r[2])
    table("g1m_effort", "Effort limit, N·m (the torque clip and the motor's ctrlrange).", lambda r: r[3])
    table("g1m_armature", "Reflected rotor inertia, kg·m² (as written into the XML).", lambda r: r[0])
    table("g1m_default_pos", "`KNEES_BENT_KEYFRAME`, rad: the action offset and the reset pose.", lambda r: r[4])
    table("g1m_action_scale", "`G1_ACTION_SCALE` = 0.25 · effort / kp, rad per action unit.", lambda r: 0.25 * r[3] / r[1])
    return "\n".join(L) + "\n"


def main():
    check = "--check" in sys.argv[1:]
    xml, names = bake_xml()
    pd = bake_pd(names)
    stale = []
    for path, text in [(ASSET_XML, xml), (PD_MOJO, pd)]:
        old = open(path).read() if os.path.exists(path) else None
        if old != text:
            if check:
                stale.append(path)
            else:
                open(path, "w").write(text)
                print("wrote", os.path.relpath(path, REPO))
    for f in sorted(os.listdir(REF_MESH)):
        src = os.path.join(REF_MESH, f)
        dst = os.path.join(MESH_DIR, f)
        same = os.path.exists(dst) and open(src, "rb").read() == open(dst, "rb").read()
        if same:
            continue
        if f not in REV_MESHES:
            raise SystemExit("mesh %s differs from ours and is not a rev-1.0 mesh" % f)
        if check:
            stale.append(dst)
        else:
            shutil.copyfile(src, dst)
            print("wrote", os.path.relpath(dst, REPO))
    if check:
        if stale:
            print("STALE:", *[os.path.relpath(p, REPO) for p in stale])
            return 1
        print("up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
