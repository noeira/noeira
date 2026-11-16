#!/usr/bin/env python3
"""Bake `assets/unitree_g1_mjlab_box.xml`: mjlab's G1 plus one free box to carry.

    pixi run python tests/robots/g1_mjlab_box_bake.py            # write the asset
    pixi run python tests/robots/g1_mjlab_box_bake.py --check    # fail if the asset is stale

The source is our own baked `assets/unitree_g1_mjlab.xml` (see `g1_mjlab_bake.py`), edited as TEXT in three places, so
every body, joint, geom and actuator of the robot is that file's, byte for byte:

  * one body `box` appended at the END of the worldbody, with a free joint and one box geom. Appended last, it takes
    the last body, joint, geom, qpos (36..42) and qvel (35..40) slots: every index of the robot is unchanged, and a
    state row or a checkpoint written for `unitree_g1_mjlab` addresses the same robot here;
  * the `stand` key gets the box's rest pose appended;
  * `<option>` gains `cone="elliptic" impratio="10"`, and the solver's two caps double: `iterations` 10 to 20,
    `ls_iterations` 20 to 40 (see THE HOLD below). This one is GLOBAL: the feet's contacts with the floor are solved under it too, so this model's robot does not step
    exactly as `unitree_g1_mjlab`'s does.

THE BOX. A cuboid of 0.327 x 0.345 x 0.361 m, 1 kg, centred on its body frame. The size is the bounding cuboid of the
`largebox` scan of the OMOMO dataset as OmniRetarget's G1 carry clips use it (fitted on its resting face,
`docs/OBJECT_INTERACTION_PLAN.md` O1.0 / O1.2): a G1 holds it between its two hands, against the chest.

ITS CONTACTS. The robot's collision geoms are `condim 1` (frictionless) except the feet, as mjlab has them; a box held
by squeezing needs friction. The box geom is therefore `priority 1` with its own condim: MuJoCo takes condim and
friction from the higher-priority geom of a pair, so every contact of the box (hands, forearms, torso, floor) is a
frictional one with the BOX's friction, and the robot's geoms are left as mjlab has them. Against the feet (also
priority 1) MuJoCo takes the larger friction. `BOX_FRICTION` is then the grip's friction, and the one to randomise.

THE HOLD (2026-11-16, `docs/OBJECT_INTERACTION_PLAN.md` §6, "Two diagnoses"). Until then the box was `condim 3` under
MuJoCo's default option (pyramidal cone, `impratio 1`), and a carried box swung 50 degrees about the line between the
two hands and hung 7 cm low:

  * the hands are capsules, and a capsule that is not parallel to a face touches it at ONE point. A frictional point
    (`condim 3`) resists sliding and not twisting about its own normal, so two of them, one a side, are a hinge: on a
    bench (a 1 kg box gripped 6 cm ahead of its centre) the box swings 100 to 135 degrees at any squeeze force. The
    box is therefore `condim 4`: torsional friction, the standard model of a contact PATCH. Its coefficient is a
    length: for a patch of radius a under uniform pressure the limit torque is (2/3)·mu·a·N, and 0.02 m is a patch
    of 3.3 cm, the hand capsule's radius (3.5 cm). On the bench the box then holds once 2·0.02·N exceeds gravity's
    torque about the grip, as the physics says, and not below;
  * with `impratio 1` friction is as soft as the normal force, and a squeezed box creeps (1 cm in 3 s on a centred
    bench grip, and the robot pinned at 15 of its own carry poses kept the box in 5). `cone="elliptic" impratio="10"`
    is what mjlab sets for its own manipulation task (`tasks/manipulation/lift_cube_env_cfg.py`, with this same
    timestep); with it 11 of the 15 keep the box. The pyramidal cone with `condim 4` drops the box at a 10 N squeeze
    on the bench, so the cone changes with the rest;
  * `iterations="20" ls_iterations="40"`, where the robot's own model and mjlab's task have 10 and 20. Both are CAPS:
    the solver stops when it has converged, so they cost only where they were cutting it short. Under the elliptic
    cone at `impratio 10` they were, measured in MuJoCo against its own converged solve over 40 control steps:
      - at `ls_iterations 20` the standing robot is 8e-4 away (the LINE SEARCH stops short; Newton takes 1 to 5
        iterations there), at 25 and above 6e-7 or less on the five scenarios of `test_unitree_g1_mjlab_box_vs_mujoco`;
      - at `iterations 10` the driven catch ends some substeps at the cap, unconverged, and two solvers that agree to
        1e-7 until then come out 6e-5 apart (ours against MuJoCo, the same step counts on both sides until the first
        capped substep). At 20 neither is capped and that scenario is 3e-7 from the converged solve.

WHAT THIS DID NOT FIX. With the same policy replayed (28 clips), the stiff friction is what buys holds (carry steps
within 10 cm 52.9 % to 58 %, with or without the torsional row); the torsional row takes the box's turn from 65 to 50
degrees and no further, at four times the coefficient either. The clips' own hands turn around the box as the body
straightens up (forearms from vertical to horizontal, 53 degrees of pitch in the median clip): a pivot grasp, with the
box leaning on the chest. No friction model makes a box follow hands that are meant to turn on it.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "noeira", "envs", "robots", "assets", "unitree_g1_mjlab.xml")
OUT = os.path.join(REPO, "noeira", "envs", "robots", "assets", "unitree_g1_mjlab_box.xml")

BOX_SIZE = (0.327, 0.345, 0.361)        # full extents, m
BOX_MASS = 1.0                          # kg (OmniRetarget randomises 0.1 - 2)
BOX_CONDIM = 4                          # sliding + torsional: a contact patch, not a point (THE HOLD above)
BOX_FRICTION = (0.9, 0.02, 0.0001)      # lateral 0.9 from the scan's URDF; torsional 0.02 m = (2/3)·0.9·(3.3 cm patch)
OPTION_SRC = '<option timestep="0.005" iterations="10" ls_iterations="20"/>'
OPTION_BOX = '<option timestep="0.005" iterations="20" ls_iterations="40" cone="elliptic" impratio="10"/>'
BOX_REST_XY = (0.6, 0.0)                # in front of the standing robot, out of reach of its feet


def bake() -> str:
    src = open(SRC).read()
    half = [s / 2 for s in BOX_SIZE]
    body = (
        '    <body name="box" pos="%g %g %.17g">\n'
        '      <freejoint name="box_joint"/>\n'
        '      <geom name="box_collision" type="box" size="%.17g %.17g %.17g" mass="%g" contype="1" conaffinity="1"\n'
        '            condim="%d" priority="1" friction="%g %g %g" group="3" rgba=".2 .5 .9 .6"/>\n'
        '    </body>\n'
    ) % (BOX_REST_XY[0], BOX_REST_XY[1], half[2], half[0], half[1], half[2], BOX_MASS, BOX_CONDIM, *BOX_FRICTION)
    end = "  </worldbody>\n"
    assert src.count(end) == 1, "expected one </worldbody>"
    out = src.replace(end, body + end)
    assert out.count(OPTION_SRC) == 1, "expected the source asset's <option> line"
    out = out.replace(OPTION_SRC, OPTION_BOX)

    key_open = '<key name="stand" qpos="'
    assert out.count(key_open) == 1, "expected one stand key"
    i = out.index(key_open) + len(key_open)
    j = out.index('"', i)
    assert len(out[i:j].split()) == 36, "the stand key is not the robot's 36 qpos"
    box_q = "%g %g %.17g 1 0 0 0" % (BOX_REST_XY[0], BOX_REST_XY[1], half[2])
    out = out[:j] + " " + box_q + out[j:]

    head = "<!-- GENERATED by tests/robots/g1_mjlab_bake.py from mjlab 8bb2bc9 — do not edit. -->\n"
    assert out.startswith(head), "unexpected header in the source asset"
    out = (
        "<!-- GENERATED by tests/robots/g1_mjlab_box_bake.py from unitree_g1_mjlab.xml (itself baked from mjlab 8bb2bc9)"
        " — do not edit. -->\n" + out[len(head):]
    )
    return out.replace('<mujoco model="g1_29dof_rev_1_0">', '<mujoco model="g1_29dof_rev_1_0_box">')


def main() -> int:
    text = bake()
    if "--check" in sys.argv:
        if not os.path.exists(OUT) or open(OUT).read() != text:
            print("STALE: %s differs from the bake; run tests/robots/g1_mjlab_box_bake.py" % os.path.relpath(OUT, REPO))
            return 1
        print("up to date:", os.path.relpath(OUT, REPO))
        return 0
    open(OUT, "w").write(text)
    import mujoco

    m = mujoco.MjModel.from_xml_path(OUT)
    r = mujoco.MjModel.from_xml_path(SRC)
    assert (m.nbody, m.njnt, m.nq, m.nv, m.ngeom) == (r.nbody + 1, r.njnt + 1, r.nq + 7, r.nv + 6, r.ngeom + 1)
    # the robot's slots are unchanged
    for name in ("body_parentid", "body_pos", "body_mass", "jnt_qposadr", "jnt_dofadr", "jnt_range", "geom_bodyid", "geom_size", "geom_condim"):
        a, b = getattr(m, name), getattr(r, name)
        assert (a[: len(b)] == b).all(), name
    b = m.nbody - 1
    print("wrote %s: nbody %d njnt %d nq %d nv %d ngeom %d; box = body %d, qpos %d.., mass %g, inertia %s" % (
        os.path.relpath(OUT, REPO), m.nbody, m.njnt, m.nq, m.nv, m.ngeom, b, m.jnt_qposadr[m.njnt - 1], m.body_mass[b],
        [round(float(x), 5) for x in m.body_inertia[b]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
