"""Unitree G1 walker, RoboParty's recipe — env aliases and host evaluation.

    from noeira.envs.robots.unitree_g1_walk_rp import (
        UnitreeG1WalkRP, UnitreeG1WalkRPBatched, g1r_host_reset,
        g1r_host_pre_step, g1r_host_terms,
    )

The recipe is `unitree_g1_walk_rp_config.mojo`; `unitree_g1_walk.mojo` is
the earlier Playground / RoboParty mix (runs s1-s6), kept so their
checkpoints still load. Same host-evaluation contract: the helpers call the
GPU hooks' own functions over `[1, N]` views of a batch-1 CPU `Data`, and
the CPU env does not call the pre-step — a host loop calls
`g1r_host_pre_step` before `step`.
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
from .unitree_g1_walk_rp_config import (
    UnitreeG1WalkRPConfig,
    G1R_OBS_DIM,
    G1R_N_TERMS,
    g1r_init,
    g1r_pre_step,
    g1r_terms,
)

comptime _pm = UNITREE_G1_DIMS

comptime UnitreeG1WalkRPModel = ModelDefFromXML[
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
    obs_dim_override=G1R_OBS_DIM,
    action_dim_override=UNITREE_G1_ACTION_DIM,
    nkey=1,
]
"""`UnitreeG1Model` (`unitree_g1_xml.mojo`) with the recipe's 82-D obs."""

comptime UnitreeG1WalkRP[TRAIN: Bool = False, DTYPE: DType = DType.float64] = Phyics3dEnv[
    UnitreeG1WalkRPModel, UnitreeG1WalkRPConfig[TRAIN], DTYPE, True
]

comptime UnitreeG1WalkRPBatched[N_ENVS: Int, TRAIN: Bool = True] = Phyics3dBatchedEnv[
    UnitreeG1WalkRPModel, UnitreeG1WalkRPConfig[TRAIN], N_ENVS,
    TERMINATE_ON_UNHEALTHY=True,
]


def g1r_host_reset[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1], seed: Int, random: Bool
):
    comptime L_Q = Layout.row_major(1, D.NQ)
    comptime L_V = Layout.row_major(1, D.NV)
    comptime L_M = Layout.row_major(1, METADATA_SIZE)
    if random:
        g1r_init[DTYPE, 1, D.NQ, D.NV, True](
            d.qpos.lt["cpu", L_Q](), d.qvel.lt["cpu", L_V](),
            d.meta.lt["cpu", L_M](), 0, seed,
        )
    else:
        g1r_init[DTYPE, 1, D.NQ, D.NV, False](
            d.qpos.lt["cpu", L_Q](), d.qvel.lt["cpu", L_V](),
            d.meta.lt["cpu", L_M](), 0, seed,
        )


def g1r_host_pre_step[PUSHES: Bool, DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1]
):
    g1r_pre_step[DTYPE, 1, D.NQ, D.NV, PUSHES](
        d.qpos.lt["cpu", Layout.row_major(1, D.NQ)](),
        d.qvel.lt["cpu", Layout.row_major(1, D.NV)](),
        d.meta.lt["cpu", Layout.row_major(1, METADATA_SIZE)](),
        0,
    )


def g1r_host_terms[DTYPE: DType, D: DimsLike](
    mut d: Data[DTYPE, D, 1],
    action: List[Float64],
    mut terms: Array[Float64, G1R_N_TERMS],
) raises -> Bool:
    if len(action) != UNITREE_G1_ACTION_DIM:
        raise Error("g1r_host_terms: action has " + String(len(action)) + " words")
    var a = TensorImpl[DTYPE].alloc(UNITREE_G1_ACTION_DIM)
    for j in range(UNITREE_G1_ACTION_DIM):
        a.data[j] = Scalar[DTYPE](action[j])
    return g1r_terms[
        DTYPE, 1, D.NQ, D.NV, D.NBODY, UNITREE_G1_ACTION_DIM, D.MAX_CONTACTS
    ](
        d.qpos.lt["cpu", Layout.row_major(1, D.NQ)](),
        d.qvel.lt["cpu", Layout.row_major(1, D.NV)](),
        d.xpos.lt["cpu", Layout.row_major(1, D.NBODY * 3)](),
        d.xquat.lt["cpu", Layout.row_major(1, D.NBODY * 4)](),
        d.xvel.lt["cpu", Layout.row_major(1, D.NBODY * 3)](),
        d.contacts.lt["cpu", Layout.row_major(1, D.MAX_CONTACTS * CONTACT_SIZE)](),
        d.meta.lt["cpu", Layout.row_major(1, METADATA_SIZE)](),
        a.lt["cpu", Layout.row_major(1, UNITREE_G1_ACTION_DIM)](),
        0,
        terms,
    )
