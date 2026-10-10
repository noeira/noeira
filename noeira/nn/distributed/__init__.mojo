"""Data-parallel training over the parameter arena (prototype).

`ProcessGroup` (N ranks, one host thread, MAX `comm` collectives or a
shared-context simulator), `DataParallel` (DDP over `ParamArena`, optionally
overlapped with the backward), `Zero1` (sharded optimizer state) and
`ZeroSharded` (ZeRO-2 / ZeRO-3 over per-block units, its step on
`RankFibers`). See `docs/DISTRIBUTED_TRAINING_PLAN.md` and
`docs/DISTRIBUTED_TRAINING_RESULTS.md`. The simulator gates run with
`pixi run -e apple test-nn-distributed` (or `-e nvidia`).
"""
