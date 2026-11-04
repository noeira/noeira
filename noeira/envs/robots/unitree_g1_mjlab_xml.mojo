"""Unitree G1 (29 DoF) — model definition, mjlab's G1 (SuperMimic's reproduction).

Reference: mjlab 8bb2bc9 `asset_zoo/robots/unitree_g1/xmls/g1.xml` (the G1
rev 1.0 body), with what mjlab applies at runtime baked in by
`tests/robots/g1_mjlab_bake.py` (collision set, armature, ground, the
reproduction's solver settings, the knees-bent keyframe, torque motors). The
layer-1 gate `tests/robots/g1_mjlab_ref.py` proves the file IS mjlab's
runtime model (104 `mjModel` tables at 0.0, integrator aside).

Measured against MuJoCo 3.12.0 (`tools/gen_model_dims.py`):
    nbody 31   njnt 30 (1 free + 29 hinge)   nq 36   nv 35   nu 29
    ngeom 69 (1 plane, 2 sphere, 31 capsule, 35 visual-only mesh)
    nexclude 4   nkey 1   cone pyramidal

HOW IT DIFFERS FROM `unitree_g1_xml` (BFM-Zero's sim-to-sim G1): 1.78 kg
lighter torso (rev 1.0), capsule / sphere collision geoms (condim 1, the 14
foot capsules condim 3 / priority 1 / friction 0.6, self-collision on),
per-motor armature instead of 0.01, no joint damping or friction loss, and
mjlab's actuator gains (`unitree_g1_mjlab_pd.mojo`). The joint order and the
29 hinges' names are the same, so a SEED reference row is valid on both.

⚠ THE ACTUATORS ARE TORQUE MOTORS. mjlab's are `<position>` servos
integrated `implicitfast`; physics3d's batched path runs Euler, so the env
computes the same PD torque every substep (`unitree_g1_mjlab_config`). The
bake script states the measured cost of that deviation.

⚠ NO COLLIDABLE MESH: every mesh geom is visual (contype 0), so the hull
budget is 0 and the narrow phase's mesh branch compiles out.
"""

from noeira.physics3d.parser import ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.envs.robots.unitree_g1_mjlab_dims import UNITREE_G1_MJLAB_DIMS


comptime _pm = UNITREE_G1_MJLAB_DIMS

# [q - q_default (29), qdot (29), projected gravity (3), root angular velocity (3)]
comptime UNITREE_G1_MJLAB_OBS_DIM: Int = 64
comptime UNITREE_G1_MJLAB_ACTION_DIM: Int = 29

comptime UnitreeG1MjlabModel = ModelDefFromXML[
    xml_path="noeira/envs/robots/assets/unitree_g1_mjlab.xml",
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
    # The reproduction's `nconmax=35` per env; 64 leaves room for a body on
    # the ground with self-contacts (the parity test prints MuJoCo's max).
    max_contacts=64,
    obs_dim_override=UNITREE_G1_MJLAB_OBS_DIM,
    action_dim_override=UNITREE_G1_MJLAB_ACTION_DIM,
    nkey=1,
]

comptime ROOT_QPOS_SIZE: Int = 7
comptime ROOT_QVEL_SIZE: Int = 6
