"""SuperMimic's object interface: its config and its two networks.

`SMObjConfigT` is the interface's identity on top of a tracker's `SMConfigT`,
as `SMPromptConfigT` is the prompt interface's
(`docs/OBJECT_INTERACTION_PLAN.md` §4.4, step O1.4):

    l = tanh(enc_motion(goal ⊙ gscale)) + gain · enc_obj([features, hist ⊙ oscale])
    a = decoder([l, hist ⊙ oscale])
    V = critic([the tracker's critic row, features, normalise(ctx)])

    features (33) = `object_task.sm_oe_feat_lane`: the object, and where it
                    should be, seen from the robot's heading frame

The object encoder's last layer starts at ZERO, so the first policy is the
tracker's, exactly. The motion encoder and the decoder never move; neither do
the tracker's dynamics. What trains: the object encoder, the interface's
critic, and the object network of the world model (`object_model.mojo`), which
keeps fitting on the interface's own replay.
"""

from noeira.deep_agents.supermimic.config import SMConfigT
from noeira.deep_agents.supermimic.object_imag_lanes import SMObjImagDims
from noeira.deep_agents.supermimic.object_model import SMObjectModel
from noeira.deep_agents.supermimic.policy import SMValueNet
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear_swish import LinearSwish
from noeira.nn.primitives.zero_linear import ZeroLinear


comptime SMObjEncoder[R: TrackingRobot, E1: Int, E2: Int] = Sequential[
    LinearSwish[SMObjImagDims[R].EO_IN, E1], LinearSwish[E1, E2], ZeroLinear[E2, SMDims[R].LATENT],
]


trait SMObjConfigT:
    comptime BASE: SMConfigT         # the tracker it sits on
    comptime OBJ_H: Int              # the world model's object network
    comptime ENC1: Int
    comptime ENC2: Int
    comptime CRIT1: Int
    comptime CRIT2: Int
    comptime DEF_GAIN: Float64       # the object encoder's share of the latent
    comptime DEF_LR: Float64


comptime SMObjEncOf[OC: SMObjConfigT] = SMObjEncoder[OC.BASE.ROBOT, OC.ENC1, OC.ENC2]
comptime SMObjCriticOf[OC: SMObjConfigT] = SMValueNet[SMObjImagDims[OC.BASE.ROBOT].OC_IN, OC.CRIT1, OC.CRIT2]
comptime SMObjModelOf[OC: SMObjConfigT] = SMObjectModel[OC.BASE.ROBOT, OC.OBJ_H]


struct SMObjConfig[base: SMConfigT, obj_h: Int, enc1: Int, enc2: Int, crit1: Int, crit2: Int](SMObjConfigT):
    comptime BASE = Self.base
    comptime OBJ_H = Self.obj_h
    comptime ENC1 = Self.enc1
    comptime ENC2 = Self.enc2
    comptime CRIT1 = Self.crit1
    comptime CRIT2 = Self.crit2
    comptime DEF_GAIN = 0.1
    comptime DEF_LR = 2e-4
