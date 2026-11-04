"""Unitree G1 as mjlab models it — `Phyics3dEnv[UnitreeG1MjlabModel, UnitreeG1MjlabConfig]` and its batches.

    from noeira.envs.robots import UnitreeG1Mjlab, UnitreeG1MjlabBatchedDR
    var env = UnitreeG1Mjlab[DType.float64]()

The body and actuator model SuperMimic's reproduction trains on (mjlab's
`get_g1_robot_cfg`): see `unitree_g1_mjlab_xml.mojo` for how it differs from
BFM-Zero's `UnitreeG1`.
"""

from .unitree_g1_mjlab_xml import UnitreeG1MjlabModel
from .unitree_g1_mjlab_config import UnitreeG1MjlabConfig
from ..phyics3d_env import Phyics3dEnv
from ..phyics3d_batched_env import Phyics3dBatchedEnv


comptime UnitreeG1Mjlab[DTYPE: DType = DType.float64] = Phyics3dEnv[
    UnitreeG1MjlabModel, UnitreeG1MjlabConfig, DTYPE, False
]

comptime UnitreeG1MjlabBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    UnitreeG1MjlabModel, UnitreeG1MjlabConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime UnitreeG1MjlabBatchedDR[N_ENVS: Int] = Phyics3dBatchedEnv[
    UnitreeG1MjlabModel, UnitreeG1MjlabConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False, LANE_DR=True
]
"""With per-lane contact friction and CoM overrides (SuperMimic's DR)."""
