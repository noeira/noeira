"""Real-robot ports — physical hardware you can buy, build and transfer to.

Distinct from `envs/dm_control/` (a benchmark suite) and `envs/metaworld/`
(a manipulation benchmark): everything here corresponds to a robot that
exists, and the point of the port is that a policy trained on it can be run on
the thing itself. `docs/SIM_TO_REAL_PLATFORM_PLAN.md` is the north star.

    SO-ARM100  Menagerie `trs_so_arm100` — 5 DOF + gripper, hand-authored
               collision, elliptic cone. The better ENGINEERING target: it
               needs no engine feature we do not already have.
    SO-ARM101  TheRobotStudio `Simulation/SO101` — the same skeleton with
               different inertials and 10x the collision-mesh cost. The
               better HARDWARE target, since it is the arm being bought.

⚠⚠ THE TWO ARE NOT INTERCHANGEABLE. Same topology, and the two long links are
bit-identical, but moving mass is 25% apart and `qpos` does not transfer
between them (different joint-axis conventions). `docs/SO_ARM101_PORT_ASSESSMENT.md`
§3 has the measured comparison; §5 says to pick which one the POLICY trains on
before training, not after.

    Unitree G1 BFM-Zero's sim-to-sim `g1_29dof` model — 29 DoF humanoid under
               the reference's torque-level PD. The baseline body for the
               G1 reproduction (`docs/BFM_ZERO_G1_REPRODUCTION.md`); GPU
               path is NVIDIA-only at nv 35.

⚠ ToddlerBot is NOT here yet — see `docs/TODDLERBOT_PORT_PLAN.md`. It is
blocked on `<inertial fullinertia>` at parse time, which these two arms sidestep
with a labelled bake (`tests/robots/so_arm_bake.py`).
"""

from .so_arm100_xml import SoArm100Model
from .so_arm101_xml import SoArm101Model
from .so_arm_reach_config import SoArmReachConfig
from .so_arm100 import SoArm100Reach, SoArm100ReachConfig
from .so_arm101 import SoArm101Reach, SoArm101ReachConfig
from .unitree_g1_xml import UnitreeG1Model
from .unitree_g1_config import UnitreeG1Config
from .unitree_g1 import UnitreeG1, UnitreeG1Batched, UnitreeG1BatchedDR
from .unitree_g1_mjlab_xml import UnitreeG1MjlabModel
from .unitree_g1_mjlab_config import UnitreeG1MjlabConfig
from .unitree_g1_mjlab import UnitreeG1Mjlab, UnitreeG1MjlabBatched, UnitreeG1MjlabBatchedDR
