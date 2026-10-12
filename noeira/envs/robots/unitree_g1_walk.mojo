"""Unitree G1 walker — the env aliases and the host evaluation of its hooks.

    from noeira.envs.robots.unitree_g1_walk import (
        UnitreeG1Walk, UnitreeG1WalkBatched, g1_walk_host_reset,
        g1_walk_host_pre_step, g1_walk_host_terms,
    )
    var env = UnitreeG1Walk[TRAIN=False]()          # CPU, any platform
    var batch = UnitreeG1WalkBatched[1024](ctx)     # GPU, NVIDIA only

    # host loop, the batched env's order: pre-step, physics, terms
    g1_walk_host_reset(env.d, seed, random=True); env.refresh()
    g1_walk_host_pre_step[PUSHES=False](env.d)
    _ = env.step(action)
    var done = g1_walk_host_terms(env.d, action_list, terms)

The config, the observation and the reward are in
`unitree_g1_walk_config.mojo`; the plan is `docs/G1_WALKER_PLAN.md`.

⚠ THE HOST HELPERS CALL THE GPU HOOKS' OWN FUNCTIONS over `[1, N]` views of
a batch-1 CPU `Data` (the pattern of `tasks/host_reward.mojo`): one
implementation, so a check run on the Mac checks the arithmetic the 5090
trains on. What differs is dtype (float64 here, float32 on the device).

⚠ THE CPU ENV DOES NOT CALL THE PRE-STEP. `Phyics3dEnv.step` calls
`pre_step_cpu` (the trait's no-op), so a host loop that wants command draws
or pushes calls `g1_walk_host_pre_step` itself, BEFORE `step` — where the
batched env calls `pre_step_full_gpu`.
"""

from layout import Layout

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.parser import ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.physics3d.fields import Data, DimsLike
from noeira.physics3d.gpu.constants import CONTACT_SIZE, METADATA_SIZE
from ..phyics3d_env import Phyics3dEnv
from ..phyics3d_batched_env import Phyics3dBatchedEnv
from .unitree_g1_dims import UNITREE_G1_DIMS
from .unitree_g1_xml import UNITREE_G1_ACTION_DIM
from .unitree_g1_walk_config import (
    UnitreeG1WalkConfig,
    G1_WALK_OBS_DIM,
    G1_WALK_N_TERMS,
    g1_walk_init,
    g1_walk_pre_step,
    g1_walk_terms,
)

comptime _pm = UNITREE_G1_DIMS

comptime UnitreeG1WalkModel = ModelDefFromXML[
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
    nexclude=_pm.NEXCLUDE,
    npair=_pm.NPAIR,
    timestep=_pm.TIMESTEP,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=64,
    obs_dim_override=G1_WALK_OBS_DIM,
    action_dim_override=UNITREE_G1_ACTION_DIM,
    nkey=1,
]
"""`UnitreeG1Model` (`unitree_g1_xml.mojo`, where every argument is
explained) with the walker's 70-D observation. Same XML, same dims."""

comptime UnitreeG1Walk[TRAIN: Bool = False, DTYPE: DType = DType.float64] = Phyics3dEnv[
    UnitreeG1WalkModel, UnitreeG1WalkConfig[TRAIN], DTYPE, True
]
"""CPU, with early termination on (an eval wants falls to end episodes)."""

comptime UnitreeG1WalkBatched[N_ENVS: Int, TRAIN: Bool = True] = Phyics3dBatchedEnv[
    UnitreeG1WalkModel, UnitreeG1WalkConfig[TRAIN], N_ENVS,
    TERMINATE_ON_UNHEALTHY=True,
]


def g1_walk_host_reset[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1], seed: Int, random: Bool
):
    """`init_qpos_gpu` for lane 0 with an explicit seed. The caller refreshes
    the FK and body velocities afterwards (`Phyics3dEnv.set_state` with the
    lane's own qpos / qvel does both)."""
    comptime L_Q = Layout.row_major(1, D.NQ)
    comptime L_V = Layout.row_major(1, D.NV)
    comptime L_M = Layout.row_major(1, METADATA_SIZE)
    if random:
        g1_walk_init[DTYPE, 1, D.NQ, D.NV, True](
            d.qpos.lt["cpu", L_Q](), d.qvel.lt["cpu", L_V](),
            d.meta.lt["cpu", L_M](), 0, seed,
        )
    else:
        g1_walk_init[DTYPE, 1, D.NQ, D.NV, False](
            d.qpos.lt["cpu", L_Q](), d.qvel.lt["cpu", L_V](),
            d.meta.lt["cpu", L_M](), 0, seed,
        )


def g1_walk_host_pre_step[PUSHES: Bool, DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1]
):
    """`pre_step_full_gpu` for lane 0: command countdown / draw, pushes."""
    g1_walk_pre_step[DTYPE, 1, D.NQ, D.NV, PUSHES](
        d.qpos.lt["cpu", Layout.row_major(1, D.NQ)](),
        d.qvel.lt["cpu", Layout.row_major(1, D.NV)](),
        d.meta.lt["cpu", Layout.row_major(1, METADATA_SIZE)](),
        0,
    )


def g1_walk_host_terms[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1],
    action: List[Float64],
    mut terms: Array[Float64, G1_WALK_N_TERMS],
) raises -> Bool:
    """The reward hook's terms for lane 0 after a step; advances the lane's
    feet state exactly as the device does. Returns the termination."""
    if len(action) != UNITREE_G1_ACTION_DIM:
        raise Error(
            "g1_walk_host_terms: action has " + String(len(action))
            + " words, expected " + String(UNITREE_G1_ACTION_DIM)
        )
    var a = TensorImpl[DTYPE].alloc(UNITREE_G1_ACTION_DIM)
    for j in range(UNITREE_G1_ACTION_DIM):
        a.data[j] = Scalar[DTYPE](action[j])
    return g1_walk_terms[
        DTYPE, 1, D.NQ, D.NV, D.NBODY, UNITREE_G1_ACTION_DIM, D.MAX_CONTACTS
    ](
        d.qpos.lt["cpu", Layout.row_major(1, D.NQ)](),
        d.qvel.lt["cpu", Layout.row_major(1, D.NV)](),
        d.xangvel.lt["cpu", Layout.row_major(1, D.NBODY * 3)](),
        d.contacts.lt["cpu", Layout.row_major(1, D.MAX_CONTACTS * CONTACT_SIZE)](),
        d.meta.lt["cpu", Layout.row_major(1, METADATA_SIZE)](),
        a.lt["cpu", Layout.row_major(1, UNITREE_G1_ACTION_DIM)](),
        0,
        terms,
    )
