"""SuperMimic on noeira's G1 — `docs/SUPERMIMIC_PORT_PLAN.md`.

A world model fit on replay from a simulator that is never differentiated,
and a policy trained by BPTT through it (H = 32) with ANALYTIC reward and
observation heads, which run through a differentiable forward kinematics.

Ported from the MIT reproduction in `references/SuperMimic-Reproduced`
(Copyright (c) 2026 Yuxi Liu), on the BFM-Zero G1 body (plan D-1).

M3 so far: `g1_fk` — per-lane FK of every body's pose and world velocity
at its origin, with its hand-written VJP.
"""
