"""Parameter-efficient fine-tuning: frozen parameters (and, next, adapters).

`freeze_params` / `unfreeze_params` freeze or unfreeze every parameter of a
model; `FreezeParams` / `UnfreezeParams` are the visitors behind them, for a
struct that walks its parameters without being a `ParamWalkable`. A frozen
parameter is one whose gradient buffer is empty (`Param.freeze`).
"""
