# Planners package — reusable planning/search components.
#
# Sub-packages:
#   common      — shared value encoding, noise sampling, action bounds,
#                 MinMaxStats Q-range tracker
#   tree_search — MCTS strategy traits + Representation/Dynamics/Prediction
#                 model contract + GenericCPUMCTS (Phase 3 CPU half).
#                 GPU variants and EZv2-specific (sampled, gumbel) land
#                 in subsequent slices.
#   trajectory  — CategoricalCEMOptimizer + CategoricalRandomShooter
#                 (Phase 1), MPPICPU / MPPIGPUBatched (Phase 2),
#                 iLQR (Phase 4).
#   nav2d       — ground navigation: footprints from a scene -> NavGrid
#                 (exact clearance + Dijkstra navigation function) -> MPPI
#                 over a lagged unicycle with a measured ResponseMap.
#   testing     — stub world models + parity harness for isolated tests.
#
# See `docs/PLANNERS_PACKAGE.md` for the full plan.
