"""Unitree G1 (29 DoF) — model definition, BFM-Zero's sim-to-sim variant.

Reference: `references/BFM-Zero-main/humanoidverse/data/robots/g1/
scene_29dof_freebase_noadditional_actuators.xml`, the MuJoCo model
BFM-Zero's own `--simulator mujoco` path loads and the one its "MuJoCo (DR)"
table row (tracking 1.079 / reward 207.3 / pose 1.104) was produced with.
Baked by `tests/robots/g1_bake.py` into `assets/unitree_g1.xml` with the
`<include>` inlined, the sensors dropped and the reference's runtime
timestep (1/200 s) written into the file; see that script's docstring for
the deviation list, and `tests/robots/g1_ref.py` for the layer-1 gate that
proves the baked file is the reference model table-for-table.

Measured against MuJoCo 3.12.0 (`tools/gen_model_dims.py`):
    nbody 41   njnt 30 (1 free + 29 hinge)   nq 36   nv 35   nu 29
    ngeom 74 (1 plane, 8 sphere, 4 cylinder, 61 mesh)   nmesh 36
    nsite 14   ncam 2   nkey 1   nexclude 0   npair 0   cone pyramidal

⚠ THE ACTUATORS ARE TORQUE MOTORS, NOT SERVOS. Every one of the 29 is a plain
`<motor joint= ctrlrange=+-effort>` with gear 1. The PD controller the paper
describes (`kp (q* - q) - kd qdot`, clipped to the effort limit, evaluated
every 200 Hz substep) lives in the ENV — `unitree_g1_config.mojo` — exactly
where the reference keeps it (`legged_robot_base._compute_torques`), with
the gains in the generated `unitree_g1_pd.mojo`. The Menagerie `g1.xml`
uses `<position kp="500" dampratio="1">` instead; that is a DIFFERENT
controller and is not what BFM-Zero trained against.

⚠ BODY INDICES are worldbody DFS order with world at 0, identical to
MuJoCo's, and pinned by name in `test_unitree_g1_vs_mujoco`. Ten of the 41
bodies are massless `dummy_lf_*` / `dummy_rf_*` foot-contact frames the
reference's sensors hang off; they carry no geom and cost nothing.

⚠ THE FIRST 29-DoF HUMANOID ON THE BATCHED GPU PATH, at nv 35. The SO-101
family broke Metal's per-thread stack at nv 24, so this env is NVIDIA-only
on the GPU; the CPU `Phyics3dEnv` runs anywhere.
"""

from noeira.physics3d.parser import ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.envs.robots.unitree_g1_dims import UNITREE_G1_DIMS


comptime _pm = UNITREE_G1_DIMS

# BFM-Zero's proprioceptive `state`: `[q - q_default (29), qdot (29),
# projected gravity (3), root angular velocity / 4 (3)]`
# (`humanoidverse_isaac.py:404-450`; the ONNX exporter pins `state_end = 64`),
# followed since G3.0 by the 463-D privileged `max_local_self`
# (`unitree_g1_priv_obs.mojo`): `[state 64 | privileged 463]` = 527, the
# observation `b` and `discriminator` consume — their reference key lists are
# exactly `state + privileged_state` (§12.34). Consumers of the proprio part
# alone read the first `UNITREE_G1_STATE_DIM` entries.
#
# ⚠ `UNITREE_G1_OBS_DIM` is NOT what `f`, `critic` or the actor see any more.
# The packed row carries `last_action 29 | history 372` after this prefix —
# `UNITREE_G1_FULL_OBS_DIM` in `unitree_g1_history.mojo`. This constant is now
# specifically "the b/discriminator prefix", and the ENV still produces exactly
# this much: the 401 is assembled by the driver (rollout) or derived from the
# ring (training), never by the physics hook.
comptime UNITREE_G1_STATE_DIM: Int = 64
comptime UNITREE_G1_PRIV_DIM: Int = 463
comptime UNITREE_G1_OBS_DIM: Int = UNITREE_G1_STATE_DIM + UNITREE_G1_PRIV_DIM
comptime UNITREE_G1_ACTION_DIM: Int = 29

comptime UnitreeG1Model = ModelDefFromXML[
    xml_path="noeira/envs/robots/assets/unitree_g1.xml",
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
    # ⚠ Both 0 here and measured so — and both default to 0 when omitted, so
    # a future `<contact>` section would be dropped silently. Passed through
    # from the generated dims for that reason.
    nexclude=_pm.NEXCLUDE,
    npair=_pm.NPAIR,
    timestep=_pm.TIMESTEP,
    # No `<option cone>` upstream — MuJoCo's default, gated at layer 1.
    cone_type=ConeType.PYRAMIDAL,
    # ⚠ MEASURED CEILING, NOT A GUESS. MuJoCo reports ncon <= 21 on the
    # standing keyframe and the board's random-control rollouts, with the two
    # feet each carrying a multicontact manifold against the plane. A fall
    # (30 % of the reference's episodes start prone) brings the hands, knees
    # and torso meshes down too, so the budget is sized for a body on the
    # ground, not a body standing. `test_unitree_g1_vs_mujoco` prints
    # MuJoCo's max `ncon` over its rollouts so the number stays evidence.
    max_contacts=64,
    obs_dim_override=UNITREE_G1_OBS_DIM,
    action_dim_override=UNITREE_G1_ACTION_DIM,
    # The `stand` keyframe — asserted by the loader, which refuses a model
    # def whose `nkey` disagrees with the XML.
    nkey=1,
]

# Body indices, worldbody DFS order with world at 0 (pinned by name in the
# parity test).
comptime PELVIS_BODY_IDX: Int = 1
comptime LEFT_ANKLE_ROLL_BODY_IDX: Int = 7
comptime RIGHT_ANKLE_ROLL_BODY_IDX: Int = 17
comptime TORSO_BODY_IDX: Int = 24
comptime LEFT_HAND_BODY_IDX: Int = 32
comptime RIGHT_HAND_BODY_IDX: Int = 40

# qpos / qvel layout. One free joint then 29 hinges in the reference's
# `dof_names` order (`unitree_g1_pd.g1_dof_name`).
comptime ROOT_QPOS_SIZE: Int = 7
comptime ROOT_QVEL_SIZE: Int = 6

# ⚠⚠ SIZED FROM **OUR** HULL, NOT MUJOCO'S — see `so_arm100_xml.mojo` for
# the method: set it low, read the raise, which states the exact requirement
# (`load_mesh_hull` refuses to truncate). MuJoCo's `mesh_graph` is the wrong
# source because our hull keeps more vertices than qhull does.
#
# Measured 2026-09-08: `fields_build` needs **34 643** vertices over the 40
# collidable mesh geoms (`unitree_g1.xml`, 61 mesh geoms of which 21 are
# visual-only, `contype="0"`). 35 328 is that rounded up to a multiple of
# 512. For scale, SO-101 needs 32 934 for TEN collidable meshes — the G1's
# collision hulls are far leaner per part.
comptime UNITREE_G1_NMESH_VERTS: Int = 35328
