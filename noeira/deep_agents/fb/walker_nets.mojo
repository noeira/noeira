"""The dm_control walker's online-FB networks, shared by the FB-MEBE driver
and the walker evals so the two cannot drift (`noeira-docs/FB_MEBE_G1_PLAN.md`
§4, rung W).

The F / B / actor shapes are the walker-era online family
(`examples/fb/fb_online_walker_gpu.mojo`, the A3 / CPR runs whose numbers
rung W is read against): one hidden layer of 1024 for F and the actor, B a
256-wide MLP with a no-affine LayerNorm (norm √D). Rung W varies ONE axis
against those runs — what B reads — so everything else is the same family.

    WalkerBNetFull   B on dm_control's whole 24-D observation (A3 / W0)
    WalkerBNetProj   B on ϑ_w = torso (xx, xz) ++ height ++ (ż, ẋ, θ̇_y) of
                     the root (W1-W4): the quantities the three walker rewards
                     are functions of (`walker_config.mojo`: torso height,
                     torso uprightness, horizontal velocity). FB-MEBE's
                     appendix "Note on B mapping": B may be trained on a
                     projection of the state with the same guarantees for
                     rewards that are functions of it (Touati & Ollivier 2021,
                     Theorem 4).

dm_control walker observation (24): `orientations` = `xmat[1:, [xx, xz]]`
for the 7 bodies after the world (torso first → cols 0, 1), `height` = torso
z (col 14), `velocity` = qvel (cols 15-23: rootz, rootx, rooty, then the six
leg joints → cols 18-23).
"""

from noeira.nn.combinators.parallel import Parallel
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.nn.primitives.slice import Slice


comptime WALKER_OBS: Int = 24
comptime WALKER_ACT: Int = 6
comptime WALKER_D: Int = 128
comptime WALKER_HID: Int = 1024
comptime WALKER_BHID: Int = 256

# observation columns (see the module header)
comptime WALKER_COL_TORSO_XX: Int = 0
comptime WALKER_COL_HEIGHT: Int = 14
comptime WALKER_COL_ROOTX_VEL: Int = 16
comptime WALKER_COL_JOINT_VEL: Int = 18   # 6 leg-joint velocities from here
comptime WALKER_N_JOINT: Int = 6

comptime _F_IN = WALKER_OBS + WALKER_ACT + WALKER_D
comptime _A_IN = WALKER_OBS + WALKER_D
comptime WALKER_PROJ: Int = 6   # 2 torso + 1 height + 3 root velocities

comptime WalkerFNet = Sequential[
    Linear[_F_IN, WALKER_HID], ReLU[WALKER_HID], Linear[WALKER_HID, WALKER_D]
]
comptime WalkerANet = Sequential[
    Linear[_A_IN, WALKER_HID], ReLU[WALKER_HID],
    Linear[WALKER_HID, WALKER_ACT], Tanh[WALKER_ACT],
]
# Q_reg on F's own [s | a | z] row
comptime WalkerQNet = Sequential[
    Linear[_F_IN, WALKER_HID], ReLU[WALKER_HID], Linear[WALKER_HID, 1]
]
comptime WalkerBNetFull = Sequential[
    Linear[WALKER_OBS, WALKER_BHID], ReLU[WALKER_BHID],
    Linear[WALKER_BHID, WALKER_D], LayerNormNoAffine[WALKER_D],
]
comptime WalkerBNetProj = Sequential[
    Parallel[
        Slice[WALKER_OBS, 0, 2],      # torso xx, xz
        Slice[WALKER_OBS, 14, 18],    # height, rootz vel, rootx vel, rooty rate
    ],
    Linear[WALKER_PROJ, WALKER_BHID], ReLU[WALKER_BHID],
    Linear[WALKER_BHID, WALKER_D], LayerNormNoAffine[WALKER_D],
]
