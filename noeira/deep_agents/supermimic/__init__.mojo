"""SuperMimic — `docs/SUPERMIMIC_PORT_PLAN.md`.

A world model fit on replay from a simulator that is never differentiated,
and a policy trained by BPTT through it (H = 32) with ANALYTIC reward and
observation heads, which run through a differentiable forward kinematics.

Ported from the MIT reproduction in `references/SuperMimic-Reproduced`
(Copyright (c) 2026 Yuxi Liu). Generic over the robot (`robot.mojo`:
`TrackingRobot` + `TrackingSim` / `TrackingBatchedSim`; `G1Robot` is noeira's
BFM-Zero G1, plan D-1).

Two targets, one body of code: every per-lane loop is a phase in `lanes.mojo`
that runs as a CPU loop or as a kernel, so `config.TARGET` picks where the
networks and updates run. Three ways to run it:
    SuperMimicAgent[C["cpu"]]       everything on the CPU
    SuperMimicAgent[C["gpu"]]       env + replay on the CPU, the networks and
                                    the four blocks on the device (any GPU)
    SuperMimicAgentGpu[C["gpu"], BS] everything on the device (a batched
                                    simulator: the G1's is NVIDIA only)

Layout (SAC's pattern):
    config.mojo         SMConfigT: target, robot, widths, shapes; presets
    agent.mojo          SuperMimicAgent: env + replay + trainer + schedule
    agent_gpu.mojo      SuperMimicAgentGpu: the same loop, device env + replay
    trainer.mojo        SuperMimicTrainer: networks, tables, the blocks, checkpoint
    metrics.mojo        SMMetrics (dashboard names), the log window, sm_tick
    blocks/             ActStep, WorldModelUpdateStep, ActorStep, CriticFitStep
                        — each `step[target]`
    lanes.mojo          the scans' per-lane phases, CPU loop or kernel
    params.mojo         param snapshots across targets (the parity gates)
    robot.mojo          TrackingRobot / TrackingSim / TrackingBatchedSim,
                        SMDims, G1Robot / G1Sim / G1BatchedSim
    tables.mojo         a robot's runtime tables (FK tree, heads, feet)
    kinematics.mojo     per-lane FK + VJP          ┐ pointer-in / pointer-out,
    heads.mojo          reward, termination, obs   │ no allocation: the bodies
    state_spec.mojo     features, integration, err ┘ of the GPU kernels
    world_model.mojo    dynamics + termination head, Eq. 1's k-step loss
    policy.mojo         encoder / decoder / critic, history and goal layouts
    imagination.mojo    the H-step rollout and −J/H's gradient
    motion_library.mojo the reference library and the reset sampler
    env.mojo            the tracking env, one `R.SIM` lane per env
    env_gpu.mojo        the tracking env on a batched device simulator
The replay is the shared `deep_agents.data.EpisodicReplay` (host) or
`EpisodicReplayGpu` (device).
"""

from .config import SMConfigT, SMConfig, SMPaperG1, SMSmokeG1, SMWorldModelOf, SMPolicyOf, SMCriticOf
from .robot import TrackingRobot, TrackingSim, TrackingBatchedSim, SMDims, G1Robot, G1Sim, G1BatchedSim, G1MjlabRobot, G1MjlabSim, G1MjlabBatchedSim
from .agent import SuperMimicAgent, SMSchedule, sm_agent, sm_schedule
from .agent_gpu import SuperMimicAgentGpu, sm_agent_gpu
from .metrics import SMMetrics
from .trainer import SuperMimicTrainer, SMUpdateMetrics
