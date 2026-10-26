"""The per-step task hooks, once, for any family with a `PlacementTable`.

    pre_step_full_gpu        -> repark_inactive_slots[T, ...](qpos, qvel, meta, env)
    custom_extract_obs_gpu   -> write_task_obs[T, ...](qpos, qvel, xpos, site_xpos,
                                                      meta, obs, env)
    custom_extract_obs_cpu   -> write_task_obs_host[T, ...](d, obs)

## ⚠⚠ THEY WERE WRITTEN FOR ONE FAMILY, LIKE THE RESET WAS

`So101TabletopConfig` carried all three inline over its own constants — three
free slots unrolled with `comptime for`, ONE region site for every goal
(`REGION_SITE_ID`, because every `so101_tabletop` region hangs off
`table_surface`), its gripper at site 1. A LIBERO config copying that shape
would have pointed every `In`/`On` goal word at whichever site the copy named,
while LIBERO's regions hang off a dozen sites. The rule is here once; a family
supplies DATA through `PlacementTable` (`region_site(r)`, `GRIPPER_SITE`, the
park poses), and `tasks/placement/check.placement_table_drift` diffs it.

## THE OBSERVATION LAYOUT — `NQ + NV + N_FREE + TASK_GOAL_WORDS`

    [0, NQ)                     qpos, in full
    [NQ, NQ + NV)               qvel
    [NQ + NV, + N_FREE)         one ACTIVE word per free slot (1.0 / 0.0)
    [.., + 9)                   gripper xyz, subject - gripper, target - subject

An inactive slot's 7 pose and 6 velocity words are ZEROED as well as flagged —
`obs.write_free_slot_obs` says why either alone is a bug.

⚠ THE TWO OBSERVATION WRITERS ARE PINNED TO EACH OTHER. One takes
`LayoutTensor`s and one a `Data`; there is no type that is both, so the reads
are written twice and the RULES (`goal_frame`, the layout) once.
`tests/tasks/test_active_mask.mojo` (SO-101) and `tests/tasks/
test_libero_task_hooks.mojo` (every LIBERO family) demand identical vectors.
"""


from noeira.physics3d.fields import Data, DimsLike
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_ACTIVE, META_IDX_TASK_PARAM_0,
)
from .obs import (
    slot_active, write_free_slot_obs, write_free_slot_obs_host, FREE_JOINT_NV,
)
from .gpu_eval import goal_frame_ids
from .predicates import OP_IN, OP_ON, OP_AT_REGION
from .placement.table import PlacementTable
from noeira.core.tiles import Tile


comptime TASK_GOAL_WORDS: Int = 9
"""Gripper(3), subject - gripper(3), target - subject(3)."""


@always_inline
def goal_frame[T: PlacementTable](
    op: Int, a: Int, b: Int
) -> Tuple[Int, Int, Int, Int]:
    """`gpu_eval.goal_frame_ids` with THIS family's region site.

    ⚠⚠ `b` IS A REGION INDEX ONLY FOR `In`/`On`/`AtRegion`. For every other op
    it is a BODY id, and `T.region_site(b)` would be a real, wrong site; so the
    site is looked up only for the region ops, and `goal_frame_ids` ignores the
    argument for the rest."""
    var rs = -1
    if op == OP_IN or op == OP_ON or op == OP_AT_REGION:
        rs = T.region_site(b)
    return goal_frame_ids(op, a, b, rs)


@always_inline
def repark_inactive_slots[
    T: PlacementTable,
    DTYPE: DType,
    BATCH_SIZE: Int,
    NQ_F: Int,
    NV_F: Int,
](
    qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
    qvel: Tile[DTYPE, BATCH_SIZE, NV_F],
    meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
    env: Int,
):
    """Pin every INACTIVE free slot at its park pose, pose AND velocity.

    `TASK_LAYER_IMPLEMENTATION.md` Gap D: gravity is shared by the batch, so a
    parked body falls unless it is pinned every step; pinning the pose alone
    lets `qvel` grow without bound (11.8 m/s over a 300-step horizon).

    ⚠ AN ACTIVE SLOT IS NOT TOUCHED, AND NEITHER IS `meta`. This runs before
    physics every step and at the end of `_reset_env_lane`, after
    `init_qpos_gpu` — a write to an active slot would freeze the props the task
    is about, and a write to `meta` would land on the tape."""
    var mask = rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_ACTIVE])
    for j in range(T.N_FREE):
        if slot_active[DTYPE](mask, T.free_slot(j)):
            continue
        var qa = T.free_qadr(j)
        var da = T.free_dadr(j)
        # ⚠ W-FIRST IN `qpos`, and the identity is (1, 0, 0, 0).
        qpos[env, qa + 0] = T.free_park_x[DTYPE](j)
        qpos[env, qa + 1] = T.free_park_y[DTYPE](j)
        qpos[env, qa + 2] = T.free_park_z[DTYPE](j)
        qpos[env, qa + 3] = Scalar[DTYPE](1)
        qpos[env, qa + 4] = Scalar[DTYPE](0)
        qpos[env, qa + 5] = Scalar[DTYPE](0)
        qpos[env, qa + 6] = Scalar[DTYPE](0)
        for k in range(FREE_JOINT_NV):
            qvel[env, da + k] = Scalar[DTYPE](0)


@always_inline
def write_task_obs[
    T: PlacementTable,
    DTYPE: DType,
    BATCH_SIZE: Int,
    NQ_F: Int,
    NV_F: Int,
    NBODY_F: Int,
    SITE_DIM: Int,
    OBS_DIM: Int,
](
    qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
    qvel: Tile[DTYPE, BATCH_SIZE, NV_F],
    xpos: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
    site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
    meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
    obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
    env: Int,
):
    """The observation of lane `env`, laid out as the module header says.

    ⚠ THE MASK AND THE TAPE ARE READ, NEVER WRITTEN. The host writes both
    once per episode; an observation hook that computed either would be
    deciding what the task is while reporting what the state is."""
    comptime MASK_BASE = NQ_F + NV_F
    comptime GOAL_BASE = NQ_F + NV_F + T.N_FREE
    for i in range(NQ_F):
        obs[env, i] = qpos[env, i]
    for i in range(NV_F):
        obs[env, NQ_F + i] = qvel[env, i]

    var mask = rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_ACTIVE])
    for j in range(T.N_FREE):
        write_free_slot_obs[DTYPE, BATCH_SIZE, OBS_DIM](
            obs, env,
            slot_active[DTYPE](mask, T.free_slot(j)),
            T.free_qadr(j),
            NQ_F + T.free_dadr(j),
            MASK_BASE + j,
        )

    comptime GS = T.GRIPPER_SITE
    var gx = rebind[Scalar[DTYPE]](site_xpos[env, GS * 3])
    var gy = rebind[Scalar[DTYPE]](site_xpos[env, GS * 3 + 1])
    var gz = rebind[Scalar[DTYPE]](site_xpos[env, GS * 3 + 2])
    var sx = Scalar[DTYPE](0)
    var sy = Scalar[DTYPE](0)
    var sz = Scalar[DTYPE](0)
    var tx = Scalar[DTYPE](0)
    var ty = Scalar[DTYPE](0)
    var tz = Scalar[DTYPE](0)
    # ⚠ `op < 0` IS THE EMPTY TAPE — a lane whose goal was never written. Its
    # goal words stay zero rather than reading term 0's garbage as an id.
    var g_op = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0]))
    if g_op >= 0:
        var ga = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0 + 1]))
        var gb = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_TASK_PARAM_0 + 2]))
        var ids = goal_frame[T](g_op, ga, gb)
        if ids[0] == 1:
            sx = rebind[Scalar[DTYPE]](site_xpos[env, ids[1] * 3])
            sy = rebind[Scalar[DTYPE]](site_xpos[env, ids[1] * 3 + 1])
            sz = rebind[Scalar[DTYPE]](site_xpos[env, ids[1] * 3 + 2])
        else:
            sx = rebind[Scalar[DTYPE]](xpos[env, ids[1] * 3])
            sy = rebind[Scalar[DTYPE]](xpos[env, ids[1] * 3 + 1])
            sz = rebind[Scalar[DTYPE]](xpos[env, ids[1] * 3 + 2])
        if ids[2] == 1:
            tx = rebind[Scalar[DTYPE]](site_xpos[env, ids[3] * 3])
            ty = rebind[Scalar[DTYPE]](site_xpos[env, ids[3] * 3 + 1])
            tz = rebind[Scalar[DTYPE]](site_xpos[env, ids[3] * 3 + 2])
        else:
            tx = rebind[Scalar[DTYPE]](xpos[env, ids[3] * 3])
            ty = rebind[Scalar[DTYPE]](xpos[env, ids[3] * 3 + 1])
            tz = rebind[Scalar[DTYPE]](xpos[env, ids[3] * 3 + 2])
    obs[env, GOAL_BASE + 0] = gx
    obs[env, GOAL_BASE + 1] = gy
    obs[env, GOAL_BASE + 2] = gz
    # ⚠ RELATIVE, NOT ABSOLUTE: the reward is a function of the differences.
    obs[env, GOAL_BASE + 3] = sx - gx
    obs[env, GOAL_BASE + 4] = sy - gy
    obs[env, GOAL_BASE + 5] = sz - gz
    obs[env, GOAL_BASE + 6] = tx - sx
    obs[env, GOAL_BASE + 7] = ty - sy
    obs[env, GOAL_BASE + 8] = tz - sz


def write_task_obs_host[T: PlacementTable, DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1], mut obs: List[Scalar[DTYPE]]
):
    """`write_task_obs` on a single-env `Data`, APPENDING to `obs`.

    ⚠ `d.dims.get_nq()`, NOT `D.NQ` — the comptime members are poison on the
    dynamic provider (`DynDims.NQ` is a negative sentinel), and a
    `range(D.NQ)` there copies nothing without an error."""
    var nq = d.dims.get_nq()
    var nv = d.dims.get_nv()
    for i in range(nq):
        obs.append(d.qpos.data[i])
    for i in range(nv):
        obs.append(d.qvel.data[i])
    var mask_base = len(obs)
    for _ in range(T.N_FREE):
        obs.append(Scalar[DTYPE](0))
    var mask = d.meta.data[META_IDX_TASK_ACTIVE]
    for j in range(T.N_FREE):
        write_free_slot_obs_host[DTYPE](
            obs,
            slot_active[DTYPE](mask, T.free_slot(j)),
            T.free_qadr(j),
            nq + T.free_dadr(j),
            mask_base + j,
        )

    comptime GS = T.GRIPPER_SITE
    var gx = d.site_xpos.data[GS * 3]
    var gy = d.site_xpos.data[GS * 3 + 1]
    var gz = d.site_xpos.data[GS * 3 + 2]
    var sx = Scalar[DTYPE](0)
    var sy = Scalar[DTYPE](0)
    var sz = Scalar[DTYPE](0)
    var tx = Scalar[DTYPE](0)
    var ty = Scalar[DTYPE](0)
    var tz = Scalar[DTYPE](0)
    var g_op = Int(d.meta.data[META_IDX_TASK_PARAM_0])
    if g_op >= 0:
        var ga = Int(d.meta.data[META_IDX_TASK_PARAM_0 + 1])
        var gb = Int(d.meta.data[META_IDX_TASK_PARAM_0 + 2])
        var ids = goal_frame[T](g_op, ga, gb)
        if ids[0] == 1:
            sx = d.site_xpos.data[ids[1] * 3]
            sy = d.site_xpos.data[ids[1] * 3 + 1]
            sz = d.site_xpos.data[ids[1] * 3 + 2]
        else:
            sx = d.xpos.data[ids[1] * 3]
            sy = d.xpos.data[ids[1] * 3 + 1]
            sz = d.xpos.data[ids[1] * 3 + 2]
        if ids[2] == 1:
            tx = d.site_xpos.data[ids[3] * 3]
            ty = d.site_xpos.data[ids[3] * 3 + 1]
            tz = d.site_xpos.data[ids[3] * 3 + 2]
        else:
            tx = d.xpos.data[ids[3] * 3]
            ty = d.xpos.data[ids[3] * 3 + 1]
            tz = d.xpos.data[ids[3] * 3 + 2]
    obs.append(gx)
    obs.append(gy)
    obs.append(gz)
    obs.append(sx - gx)
    obs.append(sy - gy)
    obs.append(sz - gz)
    obs.append(tx - sx)
    obs.append(ty - sy)
    obs.append(tz - sz)
