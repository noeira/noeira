"""SuperMimic — `docs/SUPERMIMIC_PORT_PLAN.md`.

A world model fit on replay from a simulator that is never differentiated,
and a policy trained by BPTT through it (H = 32) with ANALYTIC reward and
observation heads, which run through a differentiable forward kinematics.

Ported from the MIT reproduction in `references/SuperMimic-Reproduced`
(Copyright (c) 2026 Yuxi Liu). Generic over the robot (`robot.mojo`:
`TrackingRobot` + `TrackingSim`; `G1Robot` is noeira's BFM-Zero G1, plan D-1).

Layout (SAC's pattern):
    config.mojo         SMConfigT: target, robot, widths, shapes; presets
    agent.mojo          SuperMimicAgent: env + replay + trainer + schedule
    trainer.mojo        SuperMimicTrainer: networks, tables, the blocks
    blocks/             ActStep, WorldModelUpdateStep, ActorStep, CriticFitStep
                        — each `step[target]`, the GPU branches still raise
    robot.mojo          TrackingRobot / TrackingSim, SMDims, G1Robot / G1Sim
    tables.mojo         a robot's runtime tables (FK tree, heads, feet)
    kinematics.mojo     per-lane FK + VJP          ┐ pointer-in / pointer-out,
    heads.mojo          reward, termination, obs   │ no allocation: the bodies
    state_spec.mojo     features, integration, err ┘ of the GPU kernels
    world_model.mojo    dynamics + termination head, Eq. 1's k-step loss
    policy.mojo         encoder / decoder / critic, history and goal layouts
    imagination.mojo    the H-step rollout and −J/H's gradient
    motion_library.mojo the reference library and the reset sampler
    env.mojo            the tracking env, one `R.SIM` lane per env
The replay is the shared `deep_agents.data.EpisodicReplay`.
"""

from .config import SMConfigT, SMConfig, SMPaperG1, SMSmokeG1, SMWorldModelOf, SMPolicyOf, SMCriticOf
from .robot import TrackingRobot, TrackingSim, SMDims, G1Robot, G1Sim
from .agent import SuperMimicAgent, SMSchedule
from .trainer import SuperMimicTrainer, SMUpdateMetrics
