"""LeWM (LeWorldModel JEPA) — the reference-exact port (experimental).

docs/LEWM_REOPEN_PLAN.md. The published PushT model, layer for layer, gated
against torch, trained and planned with natively:

  * `ref_model` / `ref_load`: the architecture and the checkpoint loader;
  * `ref_rollout`: the planner's rollout and CEM (stable-worldmodel's);
  * `ref_trainer` / `batch_loader`: `train.py`'s training step and the
    threaded HDF5 batches;
  * `paper_pairs`: the 50-pair paper protocol (column M, AdaJEPA);
  * `decoder` / `decoder_trainer`: the visualisation-only reconstruction
    probe;
  * `pusht_sim_bridge`, `pong_data`, `pixel_convert`: frames and windows.

The first port (its own encoder, trainer, MPC and closed loop) was removed
2026-10-03 once this one had replaced it.
"""
