"""SuperMimic config — the full compile-time identity of one setup.

Same shape as `sac/config.mojo`: a trait bundling the deployment `TARGET`,
the robot, every network width and the batch shapes, plus zero-field conformer
structs. Every SuperMimic type and block takes ONE `C: SMConfigT` instead of a
dozen widths, and the network types are derived from it:

    SMWorldModelOf[C]   dynamics (WM_H³) + termination head (TERM_H²)
    SMPolicyOf[C]       motion encoder (ENC1-4) + decoder (DEC1-4)
    SMCriticOf[C]       critic (CRIT1, CRIT2)

`SMPaperG1` is the paper's setup on our G1 (Table A.4 widths, 4096 lanes,
batch 2048, H = 32); `SMSmokeG1` is a smoke scale.

The tuned scalars ride on the config too (`DEF_*`, Table A.3), as SAC's do:
the agents' factories (`sm_agent`, `sm_agent_gpu`) default every knob to
them, and every knob stays overridable at the call.
"""

from noeira.deep_agents.supermimic.policy import SMCritic, SMPolicy
from noeira.deep_agents.supermimic.robot import G1Robot, TrackingRobot
from noeira.deep_agents.supermimic.world_model import SMWorldModel


trait SMConfigT:
    comptime TARGET: StaticString
    comptime ROBOT: TrackingRobot
    # networks
    comptime WM_H: Int
    comptime TERM_H: Int
    comptime ENC1: Int
    comptime ENC2: Int
    comptime ENC3: Int
    comptime ENC4: Int
    comptime DEC1: Int
    comptime DEC2: Int
    comptime DEC3: Int
    comptime DEC4: Int
    comptime CRIT1: Int
    comptime CRIT2: Int
    # shapes
    comptime LANES: Int          # simulator lanes
    comptime WM_BATCH: Int       # k-step windows per world-model batch
    comptime PI_BATCH: Int       # imagined starts per policy update
    comptime HORIZON: Int        # imagined steps
    # tuned scalars (defaults of the factories)
    comptime DEF_LR: Float64         # all three networks, cosine-decayed
    comptime DEF_GAMMA: Float64      # imagination and TD(λ)
    comptime DEF_UTD: Float64        # updates per env step
    comptime DEF_WARMUP: Int         # env steps before the first update
    comptime DEF_REPLAY_CAP: Int     # replay rows per lane
    comptime DEF_WM_K: Int           # world-model window length
    comptime DEF_WM_BATCHES: Int     # world-model batches per update


comptime SMWorldModelOf[C: SMConfigT] = SMWorldModel[C.ROBOT, C.WM_H, C.TERM_H]
comptime SMPolicyOf[C: SMConfigT] = SMPolicy[
    C.ROBOT, C.ENC1, C.ENC2, C.ENC3, C.ENC4, C.DEC1, C.DEC2, C.DEC3, C.DEC4,
]
comptime SMCriticOf[C: SMConfigT] = SMCritic[C.ROBOT, C.CRIT1, C.CRIT2]


struct SMConfig[
    target: StaticString, R: TrackingRobot,
    wm_h: Int, term_h: Int, enc1: Int, enc2: Int, enc3: Int, enc4: Int,
    dec1: Int, dec2: Int, dec3: Int, dec4: Int, crit1: Int, crit2: Int,
    lanes: Int, wm_batch: Int, pi_batch: Int, horizon: Int,
](SMConfigT):
    """A config from explicit widths (tests, sweeps)."""
    comptime TARGET = Self.target
    comptime ROBOT = Self.R
    comptime WM_H = Self.wm_h
    comptime TERM_H = Self.term_h
    comptime ENC1 = Self.enc1
    comptime ENC2 = Self.enc2
    comptime ENC3 = Self.enc3
    comptime ENC4 = Self.enc4
    comptime DEC1 = Self.dec1
    comptime DEC2 = Self.dec2
    comptime DEC3 = Self.dec3
    comptime DEC4 = Self.dec4
    comptime CRIT1 = Self.crit1
    comptime CRIT2 = Self.crit2
    comptime LANES = Self.lanes
    comptime WM_BATCH = Self.wm_batch
    comptime PI_BATCH = Self.pi_batch
    comptime HORIZON = Self.horizon
    comptime DEF_LR = 2e-4
    comptime DEF_GAMMA = 0.99
    comptime DEF_UTD = 0.4
    comptime DEF_WARMUP = 40
    comptime DEF_REPLAY_CAP = 1000
    comptime DEF_WM_K = 8
    comptime DEF_WM_BATCHES = 10


comptime SMPaperG1[target: StaticString] = SMConfig[
    target, G1Robot, 2048, 64, 2048, 1024, 512, 256, 2048, 1024, 512, 256, 256, 128,
    4096, 2048, 4096, 32,
]
"""The paper's widths (Table A.4) and shapes on our G1."""

comptime SMSmokeG1[target: StaticString = "cpu"] = SMConfig[
    target, G1Robot, 128, 64, 128, 64, 64, 32, 128, 64, 64, 32, 64, 32,
    8, 64, 16, 32,
]
"""Smoke scale: wiring, not results."""
