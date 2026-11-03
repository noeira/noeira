"""SuperMimic on noeira's G1 — `docs/SUPERMIMIC_PORT_PLAN.md`.

A world model fit on replay from a simulator that is never differentiated,
and a policy trained by BPTT through it (H = 32) with ANALYTIC reward and
observation heads, which run through a differentiable forward kinematics.

Ported from the MIT reproduction in `references/SuperMimic-Reproduced`
(Copyright (c) 2026 Yuxi Liu), on the BFM-Zero G1 body (plan D-1).

M3: `g1_fk` (per-lane FK of every body's pose and world velocity at its
origin) and `heads` (reward terms, termination features, observation), each
with its hand-written VJP.
M4: `state_spec` (canonical features, explicit integration, the training
error, with VJPs) and `world_model` (dynamics MLP + termination head on
`nn`, the k-step loss of Eq. 1 and its reverse scan).
"""
