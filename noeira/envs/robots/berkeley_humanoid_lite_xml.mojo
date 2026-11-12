"""Berkeley Humanoid Lite (22 DoF) — model definition.

Reference: HybridRobotics/Berkeley-Humanoid-Lite-Assets fc90fedd008b (CC BY-SA
4.0), baked by `tests/robots/bhl_bake.py`: BHL's own MJCF without its visual
meshes, on a ground plane, with the training config's effort limits as the
motors' ctrlrange, self-collisions off as in that config, and a 2 ms step.
`tests/robots/bhl_ref.py` states what each of those changes costs in MuJoCo.

Measured against MuJoCo 3.12.0 (`tools/gen_model_dims.py`):
    nbody 27   njnt 23 (1 free + 22 hinge)   nq 29   nv 28   nu 22
    ngeom 12 (1 plane, 7 box, 4 cylinder)   nkey 1   cone pyramidal

⚠ THE ROOT FRAME IS NOT BHL's. BHL's `base` frame is its CAD origin, at floor
level; the bake moves the frame of that body midway between the hip roll
joints (`BHL_ROOT_IN_CAD_BASE_*`), so a standing robot has root z ≈ 0.50 and
a fall lowers it by 0.46 m, as on the G1. A root position exported for BHL's
own URDF or MJCF has to be moved by R · that offset before it is used here.

⚠ THE ACTUATORS ARE TORQUE MOTORS; the env computes BHL's PD law every
substep (`berkeley_humanoid_lite_config`).

⚠ NO MESH AT ALL, so the hull budget is 0.
"""

from noeira.physics3d.parser import ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.envs.robots.berkeley_humanoid_lite_dims import BERKELEY_HUMANOID_LITE_DIMS


comptime _pm = BERKELEY_HUMANOID_LITE_DIMS

# [q - q_default (22), qdot (22), projected gravity (3), root angular velocity (3)]
comptime BHL_OBS_DIM: Int = 50
comptime BHL_ACTION_DIM: Int = 22

comptime BerkeleyHumanoidLiteModel = ModelDefFromXML[
    xml_path="noeira/envs/robots/assets/berkeley_humanoid_lite.xml",
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
    # 11 robot geoms against one plane: a box gives at most 4 points, so a
    # robot lying flat stays far under this.
    max_contacts=32,
    obs_dim_override=BHL_OBS_DIM,
    action_dim_override=BHL_ACTION_DIM,
    nkey=1,
]

comptime ROOT_QPOS_SIZE: Int = 7
comptime ROOT_QVEL_SIZE: Int = 6
