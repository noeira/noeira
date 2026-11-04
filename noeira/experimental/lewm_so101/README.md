# `experimental/lewm_so101/` — LeWM world models for the SO-101 (closed)

**Status: closed, 2 Nov 2026.** A world model trained in sim did not predict
the real arm, with three encoders (ViT-tiny, ViT-tiny plus an
inverse-dynamics term, frozen V-JEPA 2.1). The only model under the
real / sim gate learnt from real teleop data and beat copying the last frame
by 8.5 %. The full record, the open checks and the lessons are in the closing
note of `noeira-docs/SO101_LEWM_PLAN.md`.

It builds on the reference LeWM in `noeira/deep_agents/lewm/`:

| file | what |
|---|---|
| `graphs.mojo` | `LeWMLossGraphRef` with proprioception in the latent (v2) and with an inverse-dynamics term (IDM) |
| `trainer.mojo` | the reference training step over those graphs |
| `data.mojo` | training windows from rendered and real `TrajectoryStore`s (T = 4 frames, 5 ticks apart) |

The drivers are `examples/lewm/lewm_so101_*.mojo`. The frozen-V-JEPA
predictor is Python, in `tools/lewm/so101_vjepa_wm.py`. The image resampler
shared by sim and real is `noeira/vision/area_resize.mojo`.
