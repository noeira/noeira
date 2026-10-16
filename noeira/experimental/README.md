# `noeira/experimental/`

Research code that isn't on the project's main path. Different stability bar than `noeira/nn/`, `noeira/deep_agents/`, etc.:

- **APIs may break** between sessions without deprecation notices.
- **Not used by the main agents/training infra.** Modules here can depend on `noeira.nn` etc., but nothing in the rest of the repo should import from here.
- **Each subdir is self-contained** with its own README explaining the experiment, its current status, and what was learned.

If something here graduates — converges on real results, becomes stable, or gets absorbed into a larger system — it moves out of `experimental/` into the appropriate package.

## Currently here

| Module | Status | Topic |
|---|---|---|
| [`functional/`](functional/) | Active | An eager mode for `nn`: a runtime-shaped tape with operators (`x @ w + b`), CPU and GPU from one binary; comptime nn modules as tape ops, nn's Adam over the tape, CUDA-graph capture of a training step. See `functional/README.md`. |
| [`pcn/`](pcn/) | Active | Predictive Coding Networks (Bogacz canonical). See `pcn/README.md` and `docs/PCN_REDESIGN.md`. |
| [`swm/`](swm/) | Active (Phase 1/6) | Sheaf World Models: orthogonal transports on a place graph, holonomy read as an observable. See `swm/README.md`, `docs/SHEAF_WORLD_MODELS_V2.md` and `docs/SWM_IMPLEMENTATION_PLAN.md`. |
