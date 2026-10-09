"""Data-parallel training over the parameter arena (prototype).

`ProcessGroup` (N ranks, one host thread, MAX `comm` collectives or a
shared-context simulator) and `DataParallel` (DDP over `ParamArena`). See
`docs/DISTRIBUTED_TRAINING_PLAN.md` in noeira-docs for the plan.
"""
