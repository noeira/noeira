"""Parameter-efficient fine-tuning: frozen parameters and LoRA adapters.

`freeze_params` / `unfreeze_params` freeze or unfreeze every parameter of a
model; `FreezeParams` / `UnfreezeParams` are the visitors behind them, for a
struct that walks its parameters without being a `ParamWalkable`. A frozen
parameter is one whose gradient buffer is empty (`Param.freeze`).

`LoRAAdapter` (`lora.mojo`) is a low-rank adapter beside a linear, driven by
hand-written layer loops; `LoRALinear` is a frozen `Linear` and its adapter
as a `Module`. `peft_io.mojo` reads and writes PEFT's adapter files
(`adapter_model.safetensors` + `adapter_config.json`).
"""
