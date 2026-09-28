#!/usr/bin/env python3
"""The `so101-tower` props — ONE set of numbers, a print file AND a sim asset each.

    pixi run python tools/soarm/gen_tower_props.py            # write
    pixi run python tools/soarm/gen_tower_props.py --check    # CI

The bowl and the brick the rig's task uses were sized off the recorded frames
(`props/bowl.xml`, `props/duplo_brick.xml` as first written). For sim-to-real
the object should be the SAME object on both sides, so this file is the
source of truth for both: from the constants below it writes

    projects/so101-tower/assets/hardware/bowl_octagon.stl    (mm, to print)
    projects/so101-tower/assets/hardware/brick_25mm.stl      (mm, to print)
    noeira/tasks/assets/props/bowl.xml                      (m, the sim asset)
    noeira/tasks/assets/props/brick.xml                     (m, the sim asset)
    noeira/tasks/assets/props/so101_tower_props/bowl_octagon.stl   (the visual)

Change a number here, run it, print the part, regenerate the family. A number
edited in the `.xml` by hand is what `--check` catches.

## THE BOWL

An OCTAGONAL bowl, not a round one, on purpose: the collision path hulls every
mesh (`collision/hull_cache.mojo`), and a hull of a bowl is a solid puck.
Eight wall boxes and a floor box keep the cavity, and an octagon is what eight
boxes ARE — the print and the collision model differ only where two adjacent
wall boxes overlap at a corner (inside the wall) and at the floor's four
corners (the floor is one square box of half-width = the inradius; its corners
poke 0.41 × inradius past the walls, INSIDE the wall boxes' volume, so no
brick can ever fall into a slit and nothing visible changes). The printed
shell is the VISUAL geom (`contype="0"`, `density="0"`); the boxes are the
collision geoms (group 3, not drawn). What you see is what you print; what
collides is what the solver can hold.

Inradius 50 mm (a 32 mm brick has room to be dropped, not placed), wall 6 mm
(prints without supports, no warping at 45 mm tall), floor 4 mm.

## THE BRICK

A 25 mm cube. `examples/tasks/task_grasp_feasibility.mojo` measured the stock
sim jaw holding a 24 mm cube and NOT a 30 mm one (`props/cube.xml`), and the
real jaw holds the 32 mm Duplo the recordings used. 25 keeps the printed
object inside the band both sides hold, so a flat lift curve is the policy
and not the object (`_the_task_was_not_feasible_and_the_curve_could_not_say_so`).
Change `BRICK_MM` if the feasibility probe on THIS family says otherwise; the
family's `slot_geom=` line is printed at the end and must follow.

## THE MASSES ARE MEASURED, THE DENSITIES DERIVED

The printed parts were weighed (2026-09-20): bowl 88 g (PLA basic, #FEC600),
cube 10 g (PLA matte, #0078BF). The sim carries each as a `density` on its
COLLISION boxes — MuJoCo sums geom masses, and the eight wall boxes overlap
at the corners — so the density is `mass / (sum of the box volumes)`, computed
below from the same numbers that size the boxes, and the total body mass
comes out at the scale's reading. The colours are the CAMERAS' view of the filaments (`BOWL_RGBA`).

## THE MATERIAL IS MATTE, AND THAT IS A RENDERING FIX

MuJoCo's default material is `specular="0.5" shininess="0.5"` (exponent 64)
and the family scene lights the desk from straight above. A horizontal face
seen from above then reflects the light's AND the headlight's specular into
the camera over its whole area — the bowl's cavity floor rendered WHITE, and
turned yellow only where the arm's shadow removed the light's term. That is
the model's lighting, not the viewer's (MuJoCo draws it the same). PLA is
matte: `PLA_SPECULAR` / `PLA_SHININESS` go on a `<material>` per prop, which
both the viewer (`model_def_from_xml` reads `mat_specular`) and the tracer
(`raytrace/visual` writes `MAT_IDX_SPECULAR`) honour. Geom-level `specular=`
would NOT be read here (only the material's is).
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HW = "projects/so101-tower/assets/hardware"
PROPS = "noeira/tasks/assets/props"
VIS_DIR = "so101_tower_props"

# ── the numbers (mm) ────────────────────────────────────────────────────
BOWL_INRADIUS_MM = 50.0     # inner flat-to-centre
BOWL_WALL_MM = 6.0
BOWL_HEIGHT_MM = 45.0
BOWL_FLOOR_MM = 4.0
BOWL_MASS_G = 88.0          # weighed, PLA basic 20 % infill
BOWL_FILAMENT_RGBA = "0.996 0.776 0.0 1"   # #FEC600, the filament's sRGB
BOWL_RGBA = "1.0 0.66 0.09 1"
"""The bowl's RENDER albedo: what the rig's cameras record it as under the
family's calibrated lights (2026-09-24, cube-in-bowl-printed: real median RGB
208,140,21 overhead — boxed at the bowl's estimated pose, a whole-frame mask
reads the wooden floor — and 176,114,13 wrist, both hue ~36; albedo = real /
the sim's light level on the bowl, 0.76 overhead / 0.61 wrist). Red wants
more than 1, so green is set for the hue: renders hue 36 on both cameras. The
filament's own colour rendered clipped yellow (hue 60)."""

BRICK_MM = 25.0
BRICK_MASS_G = 10.0         # weighed, PLA matte
BRICK_FILAMENT_RGBA = "0.0 0.471 0.749 1"  # #0078BF, the filament's sRGB
BRICK_RGBA = "0.17 0.474 0.662 1"
"""The brick's RENDER albedo (as `BOWL_RGBA`): real 46,100,135 overhead /
20,72,104 wrist (hue 204 / 203), the two cameras' geometric mean — darker and
less saturated than the filament, which rendered clipped cyan on the wrist."""

CONTACT = 'solref="0.01 1" solimp="0.998 0.998 0.001"'
"""The contact softness on every prop geom and the desk — robosuite's and
LIBERO's object recipe, 2x stiffer than MuJoCo's default (timeconst 0.02) and
an impedance of 0.998 instead of 0.9..0.95. Measured in MuJoCo on this scene
(2026-09-20): a 5 N press sinks the brick 2.36 mm into the desk at the
defaults and 0.80 mm here. Under teleop the arm presses harder than that
while lining up, and the brick was visibly INSIDE the desk. `solref`/`solimp`
are AVERAGED over a contact pair, so the desk carries the same values."""
BRICK_FRICTION = "1.5 0.01 0.0001"
"""Sliding / torsional / rolling. MuJoCo takes the MAX of the pair, so this
is the jaw-brick friction too (the jaw hulls keep the default 1). The
printed cube slipped out of the sim jaw under teleop where the real one
does not; the real pinch has compliance and edges the convex hulls lack,
and 1.5 is the fingertip value manipulation sims run with."""

PLA_SPECULAR = 0.1          # matte print; MuJoCo's default is 0.5
PLA_SHININESS = 0.1         # exponent 12.8; the default 0.5 is 64

N_SIDES = 8
HALF_ANGLE = math.pi / N_SIDES          # 22.5°
COS_H = math.cos(HALF_ANGLE)


def _bowl_box_volume_mm3():
    """The sum of the collision boxes' volumes (overlaps counted, as MuJoCo
    counts them when it sums geom masses)."""
    a = BOWL_INRADIUS_MM
    half_len = (a + BOWL_WALL_MM) * math.tan(HALF_ANGLE)
    half_h = (BOWL_HEIGHT_MM - BOWL_FLOOR_MM) / 2.0
    floor = (2 * a) * (2 * a) * BOWL_FLOOR_MM
    walls = N_SIDES * BOWL_WALL_MM * (2 * half_len) * (2 * half_h)
    return floor + walls


# kg/m^3 = g / mm^3 * 1e6
BOWL_DENSITY = round(BOWL_MASS_G / _bowl_box_volume_mm3() * 1e6, 1)
BRICK_DENSITY = round(BRICK_MASS_G / BRICK_MM ** 3 * 1e6, 1)


def _octagon(circumradius, z):
    """Vertices of a regular octagon with a FLAT facing +x, CCW seen from +z."""
    return np.array(
        [
            [circumradius * math.cos(HALF_ANGLE + 2 * HALF_ANGLE * k),
             circumradius * math.sin(HALF_ANGLE + 2 * HALF_ANGLE * k), z]
            for k in range(N_SIDES)
        ]
    )


def bowl_triangles():
    """Outward-facing triangles of the shell, mm."""
    a = BOWL_INRADIUS_MM
    ro = (a + BOWL_WALL_MM) / COS_H
    ri = a / COS_H
    H = BOWL_HEIGHT_MM
    F = BOWL_FLOOR_MM
    ob = _octagon(ro, 0.0)
    ot = _octagon(ro, H)
    it = _octagon(ri, H)
    if_ = _octagon(ri, F)
    tris = []
    n = N_SIDES
    for k in range(n):
        j = (k + 1) % n
        # outer wall, normal outward: (ob[k], ob[j], ot[j]), (ob[k], ot[j], ot[k])
        tris += [[ob[k], ob[j], ot[j]], [ob[k], ot[j], ot[k]]]
        # top rim (normal +z): outer k -> outer j -> inner j -> inner k
        tris += [[ot[k], ot[j], it[j]], [ot[k], it[j], it[k]]]
        # inner wall, normal INWARD (towards the axis): reverse winding
        tris += [[it[k], it[j], if_[j]], [it[k], if_[j], if_[k]]]
        # cavity floor (normal +z): fan from the centre
        c = np.array([0.0, 0.0, F])
        tris += [[c, if_[k], if_[j]]]
        # bottom (normal -z): fan, reversed
        c0 = np.array([0.0, 0.0, 0.0])
        tris += [[c0, ob[j], ob[k]]]
    return np.array(tris)


def brick_triangles():
    h = BRICK_MM / 2.0
    v = np.array([[sx * h, sy * h, sz * h + h]
                  for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    # indices: bit0 x, bit1 y, bit2 z
    faces = [
        (0, 2, 3, 1),  # -z
        (4, 5, 7, 6),  # +z
        (0, 1, 5, 4),  # -y
        (2, 6, 7, 3),  # +y
        (0, 4, 6, 2),  # -x
        (1, 3, 7, 5),  # +x
    ]
    tris = []
    for a, b, c, d in faces:
        tris += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
    return np.array(tris)


def write_stl(path, tris, name):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    rec["n"] = n
    rec["v"] = tris.reshape(-1, 9)
    hdr = ("binary STL: %s — tools/soarm/gen_tower_props.py, mm" % name).encode()[:80].ljust(80, b"\0")
    return hdr + struct.pack("<I", len(tris)) + rec.tobytes()


def _m(mm):
    return "%.6g" % (mm / 1000.0)


def bowl_xml():
    a = BOWL_INRADIUS_MM
    rc = a + BOWL_WALL_MM / 2.0                 # wall box centre radius
    half_len = (a + BOWL_WALL_MM) * math.tan(HALF_ANGLE)  # outer half-edge: overlaps at corners
    half_h = (BOWL_HEIGHT_MM - BOWL_FLOOR_MM) / 2.0
    zc = BOWL_FLOOR_MM + half_h
    ro = (a + BOWL_WALL_MM) / COS_H
    lines = [
        '<mujoco model="bowl">',
        "  <!-- GENERATED by tools/soarm/gen_tower_props.py — DO NOT EDIT. The",
        "       octagonal bowl of the so101-tower task: the printed shell",
        "       (bowl_octagon.stl, %g mm inradius, %g mm wall, %g mm tall, %g mm"
        % (a, BOWL_WALL_MM, BOWL_HEIGHT_MM, BOWL_FLOOR_MM),
        "       floor) as the VISUAL geom, eight wall boxes and a floor box as",
        "       the COLLISION geoms. See the generator's header for why.",
        "",
        "       slot_geom=bowl:0.0,%s,%s   (bottom_z, top_z, h_radius = outer"
        % (_m(BOWL_HEIGHT_MM), _m(ro)),
        "       circumradius) — the family must say the same. -->",
        '  <compiler angle="radian" meshdir="%s"/>' % VIS_DIR,
        "  <asset>",
        '    <mesh name="bowl_octagon" file="bowl_octagon.stl" scale="0.001 0.001 0.001"/>',
        '    <material name="bowl_pla" rgba="%s" specular="%g" shininess="%g"/>'
        % (BOWL_RGBA, PLA_SPECULAR, PLA_SHININESS),
        "  </asset>",
        "  <worldbody>",
        '    <body name="bowl">',
        '      <freejoint name="free"/>',
        '      <geom name="shell" type="mesh" mesh="bowl_octagon" contype="0" conaffinity="0"',
        '            density="0" group="0" material="bowl_pla"/>',
        '      <geom name="floor" type="box" pos="0 0 %s" size="%s %s %s"'
        % (_m(BOWL_FLOOR_MM / 2), _m(a), _m(a), _m(BOWL_FLOOR_MM / 2)),
        '            density="%g" group="3" material="bowl_pla" %s/>' % (BOWL_DENSITY, CONTACT),
    ]
    for k in range(N_SIDES):
        th = 2 * HALF_ANGLE * k
        lines.append(
            '      <geom name="wall%d" type="box" pos="%s %s %s" euler="0 0 %.7f"'
            % (k, _m(rc * math.cos(th)), _m(rc * math.sin(th)), _m(zc), th)
        )
        lines.append(
            '            size="%s %s %s" density="%g" group="3" material="bowl_pla" %s/>'
            % (_m(BOWL_WALL_MM / 2), _m(half_len), _m(half_h), BOWL_DENSITY, CONTACT)
        )
    lines += ["    </body>", "  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines), ro


def brick_xml():
    h = BRICK_MM / 2.0
    return "\n".join([
        '<mujoco model="brick">',
        "  <!-- GENERATED by tools/soarm/gen_tower_props.py — DO NOT EDIT. The",
        "       printed %g mm cube of the so101-tower task (brick_%dmm.stl)."
        % (BRICK_MM, int(BRICK_MM)),
        "       A box is its own visual; no mesh. See the generator's header for",
        "       why %g mm, and `props/duplo_brick.xml` for the recorded object." % BRICK_MM,
        "",
        "       ⚠ THE ORIGIN IS THE CUBE'S CENTRE (a box geom's), so slot_geom's",
        "       bottom_z is negative: slot_geom=brick:%s,%s,%s" % (_m(-h), _m(h), _m(h * math.sqrt(2))),
        "       (h_radius = the half-diagonal). The print's origin is its base. -->",
        '  <compiler angle="radian"/>',
        "  <asset>",
        '    <material name="brick_pla" rgba="%s" specular="%g" shininess="%g"/>'
        % (BRICK_RGBA, PLA_SPECULAR, PLA_SHININESS),
        "  </asset>",
        "  <worldbody>",
        '    <body name="brick">',
        '      <freejoint name="free"/>',
        '      <geom name="geom" type="box" size="%s %s %s" density="%g"' % (_m(h), _m(h), _m(h), BRICK_DENSITY),
        '            material="brick_pla" friction="%s" %s/>' % (BRICK_FRICTION, CONTACT),
        "    </body>",
        "  </worldbody>",
        "</mujoco>",
        "",
    ]), h


def outputs():
    bt = bowl_triangles()
    kt = brick_triangles()
    bx, ro = bowl_xml()
    kx, h = brick_xml()
    stl_bowl = write_stl(None, bt, "bowl_octagon")
    stl_brick = write_stl(None, kt, "brick_%dmm" % int(BRICK_MM))
    return {
        os.path.join(HW, "bowl_octagon.stl"): stl_bowl,
        os.path.join(HW, "brick_%dmm.stl" % int(BRICK_MM)): stl_brick,
        os.path.join(PROPS, VIS_DIR, "bowl_octagon.stl"): stl_bowl,
        os.path.join(PROPS, "bowl.xml"): bx.encode(),
        os.path.join(PROPS, "brick.xml"): kx.encode(),
    }, ro, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    out, ro, h = outputs()
    stale = []
    for path, data in out.items():
        cur = open(path, "rb").read() if os.path.exists(path) else None
        if cur != data:
            stale.append(path)
            if not args.check:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "wb").write(data)
                print("wrote", path, len(data), "bytes")
    if args.check:
        if stale:
            print("STALE:", *stale)
            return 1
        print("ok: %d files current" % len(out))
        return 0
    if not stale:
        print("all %d files already current" % len(out))
    print("family lines:")
    print("  slot_geom=bowl:0.0,%s,%s" % (_m(BOWL_HEIGHT_MM), _m(ro)))
    print("  slot_geom=brick:%s,%s,%s" % (_m(-h), _m(h), _m(h * math.sqrt(2))))
    bowl_vol = 0.0  # informational: the shell's print volume
    a = BOWL_INRADIUS_MM
    area_o = N_SIDES * (a + BOWL_WALL_MM) ** 2 * math.tan(HALF_ANGLE)
    area_i = N_SIDES * a ** 2 * math.tan(HALF_ANGLE)
    bowl_vol = area_o * BOWL_FLOOR_MM + (area_o - area_i) * (BOWL_HEIGHT_MM - BOWL_FLOOR_MM)
    print("  bowl print volume %.0f cm^3 (solid), brick %.1f cm^3" % (bowl_vol / 1000, BRICK_MM ** 3 / 1000))
    print("  masses: bowl %g g -> density %g on %.1f cm^3 of boxes; brick %g g -> density %g"
          % (BOWL_MASS_G, BOWL_DENSITY, _bowl_box_volume_mm3() / 1000, BRICK_MASS_G, BRICK_DENSITY))
    return 0


if __name__ == "__main__":
    sys.exit(main())
