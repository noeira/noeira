"""Model dimensions — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tools/gen_model_dims.py
CI checks it with: pixi run python tools/gen_model_dims.py --check

Source of truth is the `.xml` asset, read through `mujoco.MjModel`.
Editing a VALUE in the asset (a mass, a size, a colour) needs no
regeneration — only adding or removing an element does, because only
that changes a count. `--check` fails the build if you forget.
"""

from noeira.physics3d.parser.xml_parser import ParsedModel


# noeira/envs/robots/assets/unitree_g1_mjlab_box.xml
comptime UNITREE_G1_MJLAB_BOX_DIMS = ParsedModel(
    nbody=32,
    njoint=31,
    nq=43,
    nv=41,
    ngeom=70,
    nact=29,
    ntex=0,
    nmat=2,
    nlight=1,
    ncam=1,
    nsite=6,
    neq=0,
    nexclude=4,
    npair=0,
    ntendon=0,
    nsensor=0,
    nsensordata=0,
    timestep=0.005,
    max_condim=3,
    noslip_iter=0,
    ccd_tol=1e-06,
    ccd_iter=35,
)
