"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from noeira.physics3d.parser.xml_parser import ParsedModel


# noeira/tasks/scenes/so101_tower.xml
comptime SO101_TOWER_DIMS = ParsedModel(
    nbody=13,
    njoint=8,
    nq=20,
    nv=18,
    ngeom=68,
    nact=6,
    ntex=0,
    nmat=23,
    nlight=1,
    ncam=2,
    nsite=4,
    neq=0,
    nexclude=0,
    npair=0,
    ntendon=0,
    nsensor=0,
    nsensordata=0,
    timestep=0.002,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
