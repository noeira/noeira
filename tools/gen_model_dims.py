#!/usr/bin/env python3
"""Generate the comptime model dimensions from MuJoCo — phase 1b.

WHY THIS EXISTS
---------------
Every model's dimensions (NBODY, NQ, NV, ...) are compile-time constants,
because they parameterise the physics types. Today they come from
`parse_xml()` running in the comptime interpreter over an MJCF string embedded
in the Mojo source — and that embedding is the only reason the MJCF cannot be
a file on disk: the interpreter cannot `open()` (docs §10.2, probe
`design_spikes/probe_comptime_file_read.mojo`). Generating the dims ahead of
time removes the last comptime consumer of the XML, after which the model is
read at runtime like MuJoCo does.

GENERATED FROM MuJoCo, NOT FROM OUR PARSER
------------------------------------------
`mujoco.MjModel` is the authority on what a model's counts ARE — the standing
rule in `feedback_count_model_elements_with_mujoco_not_grep`, where the dog's
`njnt` is 75 and no amount of tag-counting says so. Generating from our own
`parse_xml` would instead freeze whatever it currently believes, and a
generator that agrees with the thing it replaces proves nothing.

The two already agree: `tests/physics3d/test_model_dims_vs_mujoco.mojo` checks
57 models × 20 fields against `mjModel` on every run. This generator and that
gate are deliberately independent — the gate compares `parse_xml` against
MuJoCo, so it keeps testing the parser that still ships even after the
generated dims are what the models are built from.

⚠ TWO FIELDS ARE NOT RAW `mjModel` MEMBERS
------------------------------------------
* NEQ is the weld/connect/joint equality SLAB. `<equality><tendon>`
  (mjEQ_TENDON) is deliberately excluded — it rides on the tendon record via
  TENDON_IDX_IS_EQUALITY, not in that slab. Emitting raw `m.neq` would
  over-size quadruped, manipulator and stacker and mean something different
  from what the field names.
* MAX_CONDIM is ours file-wide; MuJoCo keeps condim per-geom and per-pair, so
  it is the max over both arrays.

`from_xml_path` — AND THAT IS THE POINT
---------------------------------------
This used to have to read the file and pass the STRING, because the extracted
assets carried project-root-relative `meshdir`/`file=` while MuJoCo resolves
those against THE DIRECTORY OF THE XML — so `from_xml_path` could not find
so_arm100's or sawyer's meshes and raised.

§10.5 decision 1 settled that in MuJoCo's favour and the asset paths are now
model-file-relative, so `from_xml_path` simply works. The generator reading
our own assets exactly the way any other MuJoCo tool would is the check that
the decision actually landed: revert the paths and this file stops loading.

USAGE
    pixi run python tools/gen_model_dims.py            # write the files
    pixi run python tools/gen_model_dims.py --check    # CI: fail on any diff

Run from the project root.
"""
from __future__ import annotations

import argparse
import os
import sys

# (module holding the composed MJCF, [symbols], ...) — the same table the
# extraction used. Each symbol has an extracted .xml asset; the generated dims
# module sits beside the *_xml.mojo file that declares the symbol.
MODELS = [
    ("noeira/envs/ant/ant_xml.mojo", ["ant_xml"]),
    ("noeira/envs/half_cheetah/half_cheetah_xml.mojo", ["half_cheetah_xml"]),
    ("noeira/envs/hopper/hopper_xml.mojo", ["hopper_xml"]),
    ("noeira/envs/humanoid/humanoid_xml.mojo", ["humanoid_xml"]),
    ("noeira/envs/humanoid_standup/humanoid_standup_xml.mojo",
     ["humanoid_standup_xml"]),
    ("noeira/envs/inverted_double_pendulum/inverted_double_pendulum_xml.mojo",
     ["inverted_double_pendulum_xml"]),
    ("noeira/envs/inverted_pendulum/inverted_pendulum_xml.mojo",
     ["inverted_pendulum_xml"]),
    ("noeira/envs/pusher/pusher_xml.mojo", ["pusher_xml"]),
    ("noeira/envs/reacher/reacher_xml.mojo", ["reacher_xml"]),
    ("noeira/envs/swimmer/swimmer_xml.mojo", ["swimmer_xml"]),
    ("noeira/envs/walker2d/walker2d_xml.mojo", ["walker2d_xml"]),
    ("noeira/envs/metaworld/sawyer_reach_xml.mojo", ["sawyer_reach_xml"]),
    ("noeira/envs/robots/so_arm100_xml.mojo", ["SO_ARM100_XML"]),
    ("noeira/envs/robots/so_arm101_xml.mojo", ["SO_ARM101_XML"]),
    # Unitree G1 29-DoF, baked from the BFM-Zero sim-to-sim model by
    # `tests/robots/g1_bake.py` (meshes + PD tables come with it).
    ("noeira/envs/robots/unitree_g1_xml.mojo", ["UNITREE_G1_XML"]),
    # ⚠ THE P0 SCENE-BUDGET PROBE MODELS, and they are here on purpose.
    # `assets/so101_park_k*.xml` are emitted by
    # `tools/tasks/gen_park_scenes.py`; listing them here makes their
    # dimensions a CI ASSERTION AGAINST MuJoCo rather than a comment. That is
    # the mechanism a task FAMILY will use to keep its fixed scene budget
    # honest — see `docs/TASK_LAYER_IMPLEMENTATION.md` Gap B — so it is worth
    # having the first user of it be something that already needs it.
    ("noeira/envs/robots/so101_park_xml.mojo",
     ["SO101_PARK_K0_XML", "SO101_PARK_K3_XML",
      "SO101_PARK_K6_XML", "SO101_PARK_K9_XML",
      # ⚠ 12 and 13 exist only because P4 made them COMPILE. Before it,
      # `je_spills` budgeted `Je` alone and the blocked Newton kernel asked
      # ptxas for 136,212 B against a 101,376 B limit.
      "SO101_PARK_K12_XML", "SO101_PARK_K13_XML"]),
    # The task-layer families. ⚠ Regenerate the SCENE first
    # (`pixi run gen-family-scenes`) — this reads the composed .xml, not the
    # .family, so a stale scene would freeze stale dimensions.
    ("noeira/tasks/so101_tabletop_xml.mojo", ["SO101_TABLETOP_XML"]),
    # The so101-tower rig (`families/so101_tower.family`): the follower with
    # its wrist camera on the printed mount, the overhead camera stand, a bowl
    # and a brick — `docs/camera-rig.md`.
    ("noeira/tasks/so101_tower_xml.mojo", ["SO101_TOWER_XML"]),
    ("noeira/envs/libero/models/libero_goal_xml.mojo", ["LIBERO_GOAL_XML"]),
    ("noeira/envs/libero/models/libero_spatial_xml.mojo", ["LIBERO_SPATIAL_XML"]),
    ("noeira/envs/libero/models/libero_object_xml.mojo", ["LIBERO_OBJECT_XML"]),
    # ⚠ THE TWENTY LIBERO-10/90 SCENE FAMILIES — model defs GENERATED by
    # `tools/libero/gen_libero_envs.mojo` into `noeira/envs/libero/models/`.
    ("noeira/envs/libero/models/libero_kitchen_scene1_xml.mojo", ["LIBERO_KITCHEN_SCENE1_XML"]),
    ("noeira/envs/libero/models/libero_kitchen_scene10_xml.mojo", ["LIBERO_KITCHEN_SCENE10_XML"]),
    ("noeira/envs/libero/models/libero_kitchen_scene2_xml.mojo", ["LIBERO_KITCHEN_SCENE2_XML"]),
    ("noeira/envs/libero/models/libero_kitchen_scene3_xml.mojo", ["LIBERO_KITCHEN_SCENE3_XML"]),
    ("noeira/envs/libero/models/libero_kitchen_scene4_xml.mojo", ["LIBERO_KITCHEN_SCENE4_XML"]),
    ("noeira/envs/libero/models/libero_kitchen_scene5_xml.mojo", ["LIBERO_KITCHEN_SCENE5_XML"]),
    ("noeira/envs/libero/models/libero_kitchen_scene6_xml.mojo", ["LIBERO_KITCHEN_SCENE6_XML"]),
    ("noeira/envs/libero/models/libero_kitchen_scene7_xml.mojo", ["LIBERO_KITCHEN_SCENE7_XML"]),
    ("noeira/envs/libero/models/libero_kitchen_scene8_xml.mojo", ["LIBERO_KITCHEN_SCENE8_XML"]),
    ("noeira/envs/libero/models/libero_kitchen_scene9_xml.mojo", ["LIBERO_KITCHEN_SCENE9_XML"]),
    ("noeira/envs/libero/models/libero_living_room_scene1_xml.mojo", ["LIBERO_LIVING_ROOM_SCENE1_XML"]),
    ("noeira/envs/libero/models/libero_living_room_scene2_xml.mojo", ["LIBERO_LIVING_ROOM_SCENE2_XML"]),
    ("noeira/envs/libero/models/libero_living_room_scene3_xml.mojo", ["LIBERO_LIVING_ROOM_SCENE3_XML"]),
    ("noeira/envs/libero/models/libero_living_room_scene4_xml.mojo", ["LIBERO_LIVING_ROOM_SCENE4_XML"]),
    ("noeira/envs/libero/models/libero_living_room_scene5_xml.mojo", ["LIBERO_LIVING_ROOM_SCENE5_XML"]),
    ("noeira/envs/libero/models/libero_living_room_scene6_xml.mojo", ["LIBERO_LIVING_ROOM_SCENE6_XML"]),
    ("noeira/envs/libero/models/libero_study_scene1_xml.mojo", ["LIBERO_STUDY_SCENE1_XML"]),
    ("noeira/envs/libero/models/libero_study_scene2_xml.mojo", ["LIBERO_STUDY_SCENE2_XML"]),
    ("noeira/envs/libero/models/libero_study_scene3_xml.mojo", ["LIBERO_STUDY_SCENE3_XML"]),
    ("noeira/envs/libero/models/libero_study_scene4_xml.mojo", ["LIBERO_STUDY_SCENE4_XML"]),
    ("noeira/envs/dm_control/acrobot/acrobot_xml.mojo", ["dm_acrobot_xml"]),
    ("noeira/envs/dm_control/ball_in_cup/ball_in_cup_xml.mojo",
     ["dm_ball_in_cup_xml"]),
    ("noeira/envs/dm_control/cartpole/cartpole_xml.mojo",
     ["dm_cartpole1_xml", "dm_cartpole2_xml", "dm_cartpole3_xml"]),
    ("noeira/envs/dm_control/cheetah/cheetah_xml.mojo", ["dm_cheetah_xml"]),
    ("noeira/envs/dm_control/finger/finger_xml.mojo",
     ["dm_finger_xml", "dm_finger_spin_xml"]),
    ("noeira/envs/dm_control/fish/fish_xml.mojo", ["dm_fish_xml"]),
    ("noeira/envs/dm_control/hopper/hopper_xml.mojo", ["dm_hopper_xml"]),
    ("noeira/envs/dm_control/humanoid/humanoid_xml.mojo",
     ["dm_humanoid_xml"]),
    ("noeira/envs/dm_control/humanoid_cmu/humanoid_cmu_xml.mojo",
     ["dm_humanoid_cmu_xml"]),
    ("noeira/envs/dm_control/manipulator/manipulator_xml.mojo",
     ["dm_manipulator_bring_ball_xml", "dm_manipulator_bring_peg_xml",
      "dm_manipulator_insert_ball_xml", "dm_manipulator_insert_peg_xml"]),
    ("noeira/envs/dm_control/pendulum/pendulum_xml.mojo", ["dm_pendulum_xml"]),
    ("noeira/envs/dm_control/point_mass/point_mass_xml.mojo",
     ["dm_point_mass_xml"]),
    ("noeira/envs/dm_control/quadruped/quadruped_xml.mojo",
     ["dm_quadruped_walk_xml", "dm_quadruped_run_xml",
      "dm_quadruped_fetch_xml", "dm_quadruped_escape_xml"]),
    # ⚠ BOTH REACHERS. `dm_reacher_hard_xml` differs from the easy one in a
    # single attribute VALUE (the target's radius) and used to borrow the
    # easy model's `pmr` dims outright, on the stated grounds that every count
    # is "identical by construction". That is true — checked against
    # `mjModel`, all 15 counts and the timestep agree — but it made the hard
    # model invisible to a corpus defined by "who calls parse_xml", which is
    # how it was missed by the 1b.1 extraction. It carries its own generated
    # dims now, so the claim is gated instead of asserted in a comment.
    ("noeira/envs/dm_control/reacher/reacher_xml.mojo",
     ["dm_reacher_xml", "dm_reacher_hard_xml"]),
    ("noeira/envs/dm_control/stacker/stacker_xml.mojo",
     ["dm_stacker_2_xml", "dm_stacker_4_xml"]),
    ("noeira/envs/dm_control/swimmer/swimmer_xml.mojo",
     ["dm_swimmer6_xml", "dm_swimmer15_xml"]),
    ("noeira/envs/dm_control/walker/walker_xml.mojo", ["dm_walker_xml"]),
    ("noeira/envs/dm_control/manipulation_lift_box_xml.mojo",
     ["lift_large_box_xml"]),
    ("noeira/envs/dm_control/manipulation_place_cradle_xml.mojo",
     ["place_cradle_xml"]),
    ("noeira/envs/dm_control/manipulation_place_brick_xml.mojo",
     ["place_brick_xml"]),
    ("noeira/envs/dm_control/manipulation_lift_brick_xml.mojo",
     ["lift_brick_xml"]),
    ("noeira/envs/dm_control/manipulation_reassemble5_xml.mojo",
     ["reassemble5_xml"]),
    ("noeira/envs/dm_control/manipulation_reach_xml.mojo",
     ["reach_site_features_xml"]),
    ("noeira/envs/dm_control/manipulation_reach_duplo_xml.mojo",
     ["reach_duplo_xml"]),
    ("noeira/envs/dm_control/manipulation_stack_3_bricks_xml.mojo",
     ["stack_3_bricks_xml"]),
    ("noeira/envs/dm_control/manipulation_stack3r_xml.mojo",
     ["stack_3_random_xml"]),
    ("noeira/envs/dm_control/manipulation_stack_2_bricks_moveable_base_xml"
     ".mojo", ["stack_2_bricks_moveable_base_xml"]),
    ("noeira/envs/dm_control/manipulation_stack2_xml.mojo",
     ["stack_2_bricks_xml"]),
    ("noeira/envs/dm_control/dog/dog_xml.mojo",
     ["dm_dog_stand_walk_xml", "dm_dog_run_xml", "dm_dog_trot_xml"]),
    ("noeira/envs/dm_control/dog/dog_fetch_xml.mojo", ["dm_dog_fetch_xml"]),
]

MJEQ_TENDON = 3


def asset_path(module: str, sym: str) -> str:
    """Where the extracted `.xml` for this symbol lives."""
    parts = module.split("/")
    base = sym.lower()
    if base.endswith("_xml"):
        base = base[:-4]
    if base.startswith("dm_"):
        base = base[3:]
    # ⚠ A TASK FAMILY's scene is COMPOSED, not authored, and lives outside
    # `envs/<pkg>/assets/`. It is regenerated by `pixi run gen-family-scenes`
    # from the `.family` slot table; listing it here is what makes the
    # family's fixed scene BUDGET a CI assertion against MuJoCo rather than a
    # comment (`TASK_LAYER_IMPLEMENTATION.md` Gap B). Adding a slot changes
    # nq/nv, and `--check` says so.
    if parts[1] == "tasks":
        return "noeira/tasks/scenes/%s.xml" % base
    # LIBERO is a task root of its own: its scenes are composed too, into
    # `noeira/envs/libero/scenes/` — NOT the pack under `envs/libero/assets/`.
    if parts[2] == "libero":
        return "noeira/envs/libero/scenes/%s.xml" % base
    if parts[2] == "dm_control":
        if parts[-1].startswith("manipulation_"):
            return "noeira/envs/dm_control/assets/manipulation/%s.xml" % base
        return "noeira/envs/dm_control/assets/%s.xml" % base
    return "noeira/envs/%s/assets/%s.xml" % (parts[2], base)


def dims_module(module: str) -> str:
    """The generated `*_dims.mojo` beside the `*_xml.mojo` that declares it."""
    assert module.endswith("_xml.mojo"), module
    return module[: -len("_xml.mojo")] + "_dims.mojo"


def const_name(sym: str) -> str:
    n = sym.upper()
    if n.endswith("_XML"):
        n = n[: -len("_XML")]
    return n + "_DIMS"


def dims_from_mujoco(xml_path: str) -> dict:
    import mujoco

    # See the module docstring: `from_xml_path`, because the assets are
    # model-file-relative and that is MuJoCo's own resolution rule.
    m = mujoco.MjModel.from_xml_path(xml_path)

    neq_slab = sum(
        1 for i in range(m.neq) if int(m.eq_type[i]) != MJEQ_TENDON
    )
    max_condim = 3
    for i in range(m.ngeom):
        max_condim = max(max_condim, int(m.geom_condim[i]))
    for i in range(m.npair):
        max_condim = max(max_condim, int(m.pair_dim[i]))

    return dict(
        nbody=m.nbody, njoint=m.njnt, nq=m.nq, nv=m.nv, ngeom=m.ngeom,
        nact=m.nu, ntex=m.ntex, nmat=m.nmat, nlight=m.nlight, ncam=m.ncam,
        nsite=m.nsite, neq=neq_slab, nexclude=m.nexclude, npair=m.npair,
        ntendon=m.ntendon, nsensor=m.nsensor, nsensordata=m.nsensordata,
        timestep=m.opt.timestep, max_condim=max_condim,
        noslip_iter=m.opt.noslip_iterations, ccd_tol=m.opt.ccd_tolerance,
        ccd_iter=m.opt.ccd_iterations,
    )


def fmt_float(x: float) -> str:
    """Round-trippable, and never bare `1` for a Float64 field."""
    s = repr(float(x))
    return s if ("." in s or "e" in s or "E" in s) else s + ".0"


def _header(regen: str, check: str) -> list:
    return [
        '"""Model dimensions — GENERATED, DO NOT EDIT.\n',
        "\n",
        "Regenerate with:  %s\n" % regen,
        "CI checks it with: %s\n" % check,
        "\n",
        "Source of truth is the `.xml` asset, read through `mujoco.MjModel`.\n",
        "Editing a VALUE in the asset (a mass, a size, a colour) needs no\n",
        "regeneration — only adding or removing an element does, because only\n",
        "that changes a count. `--check` fails the build if you forget.\n",
        '"""\n',
        "\n",
        "from noeira.physics3d.parser.xml_parser import ParsedModel\n",
    ]


def _dims_block(xml_path: str, const: str) -> list:
    """One `comptime <const> = ParsedModel(...)` — the ONE spelling of it,
    shared by the `MODELS` table and `--xml`."""
    d = dims_from_mujoco(xml_path)
    out = ["\n\n# %s\n" % xml_path, "comptime %s = ParsedModel(\n" % const]
    for k in ("nbody", "njoint", "nq", "nv", "ngeom", "nact", "ntex",
              "nmat", "nlight", "ncam", "nsite", "neq", "nexclude",
              "npair", "ntendon", "nsensor", "nsensordata"):
        out.append("    %s=%d,\n" % (k, int(d[k])))
    out.append("    timestep=%s,\n" % fmt_float(d["timestep"]))
    out.append("    max_condim=%d,\n" % int(d["max_condim"]))
    out.append("    noslip_iter=%d,\n" % int(d["noslip_iter"]))
    out.append("    ccd_tol=%s,\n" % fmt_float(d["ccd_tol"]))
    out.append("    ccd_iter=%d,\n" % int(d["ccd_iter"]))
    out.append(")\n")
    return out


def render(module: str, syms: list) -> str:
    out = _header("pixi run python tools/gen_model_dims.py",
                  "pixi run python tools/gen_model_dims.py --check")
    for sym in syms:
        out += _dims_block(asset_path(module, sym), const_name(sym))
    return "".join(out)


def render_xml(xml_path: str, emit: str, name: str) -> str:
    """`--xml`: the dims of ANY scene, for a model that is not the library's.

    ⚠ A PROJECT's scene (`projects/<p>/...`) is not in `MODELS` and must not
    be: the library does not list what a private project builds. The file is
    written where the caller says, and its header records the exact command,
    so the project regenerates and checks it the same way CI checks ours.
    """
    args = "--xml %s --emit %s --name %s" % (xml_path, emit, name)
    out = _header("pixi run python tools/gen_model_dims.py " + args,
                  "pixi run python tools/gen_model_dims.py " + args + " --check")
    out += _dims_block(xml_path, name.upper() + "_DIMS")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if any file would change")
    ap.add_argument("--xml", help="one scene NOT in MODELS (e.g. a project's);"
                    " needs --emit and --name")
    ap.add_argument("--emit", help="with --xml: the dims module to write")
    ap.add_argument("--name", help="with --xml: the constant is <NAME>_DIMS")
    args = ap.parse_args()

    if not os.path.isdir("noeira/envs"):
        print("run me from the project root", file=sys.stderr)
        return 2

    if args.xml or args.emit or args.name:
        if not (args.xml and args.emit and args.name):
            print("--xml needs --emit and --name", file=sys.stderr)
            return 2
        text = render_xml(args.xml, args.emit, args.name)
        old = None
        if os.path.exists(args.emit):
            with open(args.emit, "r") as f:
                old = f.read()
        if old == text:
            print("up to date —", args.emit)
            return 0
        if args.check:
            print("STALE —", args.emit, "differs from", args.xml)
            return 1
        with open(args.emit, "w") as f:
            f.write(text)
        print("wrote", args.emit)
        return 0

    stale, written = [], 0
    for module, syms in MODELS:
        target = dims_module(module)
        text = render(module, syms)
        old = None
        if os.path.exists(target):
            with open(target, "r") as f:
                old = f.read()
        if old == text:
            continue
        if args.check:
            stale.append(target)
        else:
            with open(target, "w") as f:
                f.write(text)
            written += 1

    n = sum(len(s) for _, s in MODELS)
    if args.check:
        if stale:
            print("STALE — %d generated file(s) differ from the assets:"
                  % len(stale))
            for p in stale:
                print("   ", p)
            print("\nrun: pixi run python tools/gen_model_dims.py")
            return 1
        print("up to date — %d models across %d files" % (n, len(MODELS)))
        return 0

    print("wrote %d file(s); %d models across %d files"
          % (written, n, len(MODELS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
