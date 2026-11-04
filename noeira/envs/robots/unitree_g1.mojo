"""Unitree G1 — `Phyics3dEnv[UnitreeG1Model, UnitreeG1Config]` and its batch.

    from noeira.envs.robots import UnitreeG1, UnitreeG1Batched
    var env = UnitreeG1[DType.float64]()            # CPU, any platform
    var batch = UnitreeG1Batched[N_ENVS=256](ctx)   # GPU, NVIDIA only

BFM-Zero's body under BFM-Zero's controller: 29 torque-level PD targets at
50 Hz over 200 Hz physics, the 64-D proprioceptive state, no reward, no
termination, 500-step episodes. See `unitree_g1_config.mojo` for what is
and is not the reference's yet, and `docs/BFM_ZERO_G1_REPRODUCTION.md` for
the rung this is (G0).
"""

from .unitree_g1_xml import UnitreeG1Model
from .unitree_g1_config import UnitreeG1Config
from ..phyics3d_env import Phyics3dEnv
from ..phyics3d_batched_env import Phyics3dBatchedEnv


comptime UnitreeG1[DTYPE: DType = DType.float64] = Phyics3dEnv[
    UnitreeG1Model, UnitreeG1Config, DTYPE, False
]

comptime UnitreeG1Batched[N_ENVS: Int] = Phyics3dBatchedEnv[
    UnitreeG1Model, UnitreeG1Config, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime UnitreeG1BatchedDR[N_ENVS: Int] = Phyics3dBatchedEnv[
    UnitreeG1Model, UnitreeG1Config, N_ENVS, TERMINATE_ON_UNHEALTHY=False, LANE_DR=True
]
"""The same batch with per-lane overrides of contact friction and CoM
(`physics3d/constraints/lane_dr.mojo`): SuperMimic's domain randomisation."""
