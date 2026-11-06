"""SuperMimic's prompt interface: its config and its two networks.

`SMPromptConfigT` is the interface's identity on top of a tracker's
`SMConfigT`: the command's width and the widths of the two networks that train
(the paper's Table A.4 has the velocity encoder at MLP (256, 256), the critic
at MLP (256, 128), "one critic per interface").

    l = enc_prompt([cmd, hist ⊙ oscale])                      (tanh, LATENT)
    V = critic([canonical(s), cmd, normalise(ξ)])

The paper's encoder reads "a feature of the last 10 frames of o" it does not
define: here, the decoder's own flattened, scaled history. `CMD` is 3 for the
velocity control and LATENT for a reward-derived z; everything else about the
interface is data (`prompt_bank.mojo`).
"""

from noeira.deep_agents.supermimic.config import SMConfigT
from noeira.deep_agents.supermimic.policy import SMValueNet
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.activations import Tanh
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_swish import LinearSwish


comptime SMPromptEncoder[R: TrackingRobot, CMD: Int, E1: Int, E2: Int] = Sequential[
    LinearSwish[CMD + SMDims[R].HIST_FLAT, E1], LinearSwish[E1, E2], Linear[E2, SMDims[R].LATENT], Tanh[SMDims[R].LATENT],
]


trait SMPromptConfigT:
    comptime BASE: SMConfigT         # the tracker it sits on
    comptime CMD: Int                # command width
    comptime ENC1: Int
    comptime ENC2: Int
    comptime CRIT1: Int
    comptime CRIT2: Int
    comptime DEF_ALIGN: Float64      # the alignment's weight (Appendix G)
    comptime DEF_LR: Float64
    comptime DEF_REF_SHARE: Float64  # the share of reference commands (Appendix G: 0.5)
    comptime DEF_HOLD: Int           # steps between two command draws in the simulator (250 = 5 s)


comptime SMPromptEncOf[P: SMPromptConfigT] = SMPromptEncoder[P.BASE.ROBOT, P.CMD, P.ENC1, P.ENC2]
comptime SMPromptCriticOf[P: SMPromptConfigT] = SMValueNet[SMDims[P.BASE.ROBOT].S + P.CMD + SMDims[P.BASE.ROBOT].X, P.CRIT1, P.CRIT2]


struct SMPromptConfig[base: SMConfigT, cmd: Int, enc1: Int, enc2: Int, crit1: Int, crit2: Int](SMPromptConfigT):
    comptime BASE = Self.base
    comptime CMD = Self.cmd
    comptime ENC1 = Self.enc1
    comptime ENC2 = Self.enc2
    comptime CRIT1 = Self.crit1
    comptime CRIT2 = Self.crit2
    comptime DEF_ALIGN = 0.1
    comptime DEF_LR = 2e-4
    comptime DEF_REF_SHARE = 0.5
    comptime DEF_HOLD = 250
