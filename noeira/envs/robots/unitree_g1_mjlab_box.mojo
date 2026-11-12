"""Unitree G1 (mjlab's) plus one free box to carry — the env and its batches.

    from noeira.envs.robots import UnitreeG1MjlabBox
    var env = UnitreeG1MjlabBox[DType.float64]()

`UnitreeG1Mjlab`'s config drives it unchanged (the same PD on the robot's 29
hinges; the box is passive). Its reset puts the box at rest in front of the
robot: see `unitree_g1_mjlab_box_xml.mojo` for where the box lives in the
state.
"""

from .unitree_g1_mjlab_box_xml import UnitreeG1MjlabBoxModel
from .unitree_g1_mjlab_config import UnitreeG1MjlabConfig
from ..phyics3d_env import Phyics3dEnv
from ..phyics3d_batched_env import Phyics3dBatchedEnv


comptime UnitreeG1MjlabBox[DTYPE: DType = DType.float64] = Phyics3dEnv[
    UnitreeG1MjlabBoxModel, UnitreeG1MjlabConfig, DTYPE, False
]

comptime UnitreeG1MjlabBoxBatched[N_ENVS: Int] = Phyics3dBatchedEnv[
    UnitreeG1MjlabBoxModel, UnitreeG1MjlabConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime UnitreeG1MjlabBoxBatchedDR[N_ENVS: Int] = Phyics3dBatchedEnv[
    UnitreeG1MjlabBoxModel, UnitreeG1MjlabConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False, LANE_DR=True
]
