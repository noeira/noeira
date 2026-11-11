"""Berkeley Humanoid Lite — `Phyics3dEnv[BerkeleyHumanoidLiteModel, BerkeleyHumanoidLiteConfig]` and its batches.

    from noeira.envs.robots import BerkeleyHumanoidLite, BerkeleyHumanoidLiteBatchedDR
    var env = BerkeleyHumanoidLite[DType.float64]()

The open-source 22-DoF humanoid of Berkeley's Hybrid Robotics lab, with the
actuator limits its authors train under: see `berkeley_humanoid_lite_xml.mojo`.
"""

from .berkeley_humanoid_lite_xml import BerkeleyHumanoidLiteModel
from .berkeley_humanoid_lite_config import BerkeleyHumanoidLiteConfig
from ..phyics3d_env import Phyics3dEnv
from ..phyics3d_batched_env import Phyics3dBatchedEnv


comptime BerkeleyHumanoidLite[DTYPE: DType = DType.float64] = Phyics3dEnv[
    BerkeleyHumanoidLiteModel, BerkeleyHumanoidLiteConfig, DTYPE, False
]

comptime BerkeleyHumanoidLiteBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    BerkeleyHumanoidLiteModel, BerkeleyHumanoidLiteConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime BerkeleyHumanoidLiteBatchedDR[N_ENVS: Int] = Phyics3dBatchedEnv[
    BerkeleyHumanoidLiteModel, BerkeleyHumanoidLiteConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False, LANE_DR=True
]
"""With per-lane contact friction and CoM overrides (SuperMimic's DR)."""
