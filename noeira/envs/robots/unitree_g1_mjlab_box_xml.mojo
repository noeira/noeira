"""Unitree G1 (mjlab's) plus one free box to carry — model definition.

The asset is `assets/unitree_g1_mjlab_box.xml`, baked by
`tests/robots/g1_mjlab_box_bake.py` from `assets/unitree_g1_mjlab.xml`: the
same robot, byte for byte, and one body `box` appended at the END of the
worldbody (free joint, one box geom of 0.327 x 0.345 x 0.361 m, 1 kg).

Measured against MuJoCo 3.12.0 (`tools/gen_model_dims.py`):
    nbody 32   njnt 31 (2 free + 29 hinge)   nq 43   nv 41   nu 29
    ngeom 70 (the robot's 69 + the box)

THE BOX TAKES THE LAST SLOTS: body 31, joint 30, geom 69, qpos 36..42
(position, quaternion WXYZ), qvel 35..40 (linear in the world, angular in the
box's frame). Every index of the robot is `unitree_g1_mjlab`'s, so a state row,
a PD table or a checkpoint written for that model addresses the same robot
here (`G1M_BOX_QPOS` / `G1M_BOX_QVEL` below are where the box starts).

ITS CONTACTS are frictional with the BOX's friction, whatever touches it
(`condim 3`, `priority 1` on the box geom; the robot's geoms stay `condim 1`
as mjlab has them): see the bake script.
"""

from noeira.physics3d.parser import ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.envs.robots.unitree_g1_mjlab_box_dims import UNITREE_G1_MJLAB_BOX_DIMS
from noeira.envs.robots.unitree_g1_mjlab_xml import UNITREE_G1_MJLAB_ACTION_DIM, UNITREE_G1_MJLAB_OBS_DIM


comptime _pm = UNITREE_G1_MJLAB_BOX_DIMS

comptime G1M_BOX_QPOS: Int = 36          # the box's 7 qpos start here
comptime G1M_BOX_QVEL: Int = 35          # its 6 qvel
comptime G1M_BOX_BODY: Int = 31
comptime G1M_BOX_GEOM: Int = 69
# the bake's rest pose (the `stand` key): on the floor in front of the robot
comptime G1M_BOX_REST_X: Float64 = 0.6
comptime G1M_BOX_REST_Z: Float64 = 0.1805

comptime UnitreeG1MjlabBoxModel = ModelDefFromXML[
    xml_path="noeira/envs/robots/assets/unitree_g1_mjlab_box.xml",
    nbody=_pm.NBODY,
    njoint=_pm.NJOINT,
    nq=_pm.NQ,
    nv=_pm.NV,
    ngeom=_pm.NGEOM,
    nact=_pm.NACT,
    ntex=_pm.NTEX,
    nmat=_pm.NMAT,
    nlight=_pm.NLIGHT,
    ncam=_pm.NCAM,
    nsite=_pm.NSITE,
    nsensor=_pm.NSENSOR, nsensordata=_pm.NSENSORDATA,
    neq=_pm.NEQ,
    nexclude=_pm.NEXCLUDE,
    npair=_pm.NPAIR,
    timestep=_pm.TIMESTEP,
    cone_type=ConeType.PYRAMIDAL,
    # the robot's 64 (see `unitree_g1_mjlab_xml`) plus the box on the floor
    # (4) and held (hands, forearms, torso)
    max_contacts=80,
    obs_dim_override=UNITREE_G1_MJLAB_OBS_DIM,
    action_dim_override=UNITREE_G1_MJLAB_ACTION_DIM,
    nkey=1,
]
