"""LeWM (LeWorldModel JEPA) — the reference-exact port.

docs/LEWM_REOPEN_PLAN.md. The published PushT model (`quentinll/lewm-pusht`,
stable-worldmodel 0.1.1), layer for layer, gated against torch forward and
backward (`tests/deep_agents/lewm/`), trained and planned with natively; the
paper's PushT results are reproduced with it:

  * `model` / `load`: the architecture and the checkpoint loader;
  * `rollout`: the planner's rollout, its costs and CEM (stable-worldmodel's);
  * `trainer`: `train.py`'s training step, and the snapshot / restore AdaJEPA
    adapts from;
  * `adaptation`: AdaJEPA's test-time adaptation windows;
  * `intact`: INTACT's intent-to-action actor, Direct planning;
  * `decoder` / `decoder_trainer`: the visualisation-only reconstruction
    probe.

Its data and evaluation live with their subjects: the dataset in
`nn/datasets/lewm_pusht`, its threaded batches in `data/lewm_pusht_loader`,
the PushT frames and the 50-pair protocol in `envs/pusht/encoder_frame` and
`envs/pusht/eval_pairs`, generic windows in `data/window_source`. The SO-101
variants (proprioception, inverse dynamics) are research code in
`experimental/lewm_so101/`.
"""
