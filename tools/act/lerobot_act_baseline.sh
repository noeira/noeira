#!/usr/bin/env bash
# +--------------------------------------------------------------------------+ #
# | LeRobot's own ACT on our 50-episode dataset — the baseline our port needs
# +--------------------------------------------------------------------------+ #
#
# WHY. Our ACT plateaus at validation L1 ~0.394 (pretrained backbone, epoch 7.7)
# and no amount of code reading says whether that is the data's ceiling or our
# bug. One number decides where every remaining hour goes. LeRobot is the right
# reference because it reads this dataset NATIVELY — no converter, no
# TrajectoryStore — so the entire data path is out of the comparison, and it is
# what recorded the dataset in the first place.
#
# Run on the NVIDIA box, in a LeRobot checkout, NOT in noeira:
#
#     pip install -e '.[feetech]'      # or the project's own install
#     bash /path/to/noeira/tools/act/lerobot_act_baseline.sh defaults
#
# ─────────────────────────────────────────────────────────────────────────────
# THE SPLIT IS THE SAME ONE
#
# `--dataset.eval_split=0.2` holds out "the last ceil(n*0.2) episodes per task"
# **in the order `--dataset.episodes` gives them** (`datasets/factory.py:211`;
# `LeRobotDataset` stores the list as given, it does not sort). So the ordering
# below — our 40 training episodes first, our 10 validation episodes last —
# reproduces our holdout exactly.
#
# Our split is a Fisher-Yates over a splitmix64 stream seeded 7
# (`deep_agents/act/data.mojo:_split_episodes`); these ids were printed by the
# Mojo code itself, not re-derived:
#
#     val = 4 18 30 31 34 37 41 43 45 46
#
# ⚠ If you change `--dataset.episodes`, the holdout changes with it and the
# comparison below is void.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT TO COMPARE, AND THE THREE CAVEATS
#
# Ours (`val_l1_loss` in the dashboard):   best 0.3941 @ epoch 7.7, pretrained
#                                     best 0.4076 @ epoch 15.5, random backbone
# LeRobot logs `eval_loss` every `--eval_steps`.
#
#   1. `eval_loss` is the TOTAL loss (l1 + kl_weight*kld), not l1 alone
#      (`scripts/lerobot_train.py:795`). Once the KL collapses — and it will,
#      see below — `kl_weight*kld` is ~0.01, so `eval_loss` and l1 converge.
#      Early in the run they do not; compare the plateau, not epoch 1.
#
#   2. ⚠ THE L1 DENOMINATORS DIFFER, so the two numbers are not the same
#      statistic. `act-main` (what we ported) does `(all_l1 * ~is_pad).mean()`
#      — padded slots contribute 0 to the numerator and STILL COUNT in the
#      denominator. LeRobot divides by the valid count only
#      (`modeling_act.py:150`). With K=60 over episodes of 240-528 frames,
#      roughly 19% of start positions carry padding, so **ours reads ~5% LOWER
#      than LeRobot's for identical predictions.** A 5% gap is not a finding; a
#      2x gap is.
#
#   3. Resolution. LeRobot trains at the dataset's native 480x640; our store is
#      240x320. That is a real advantage for LeRobot and part of what
#      "defaults" is measuring.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS ALREADY SETTLED, so nobody re-derives it from this run
#
#   * The KL collapse is FAITHFUL. Both implementations sum the KL over the 32
#     latent dims and mean over the batch (`policy.py:80 total_kld=klds.sum(1)
#     .mean(0)`; `modeling_act.py:159 (...).sum(-1).mean()` — LeRobot's variable
#     is named `mean_kld` but it is the same quantity, which is worth knowing
#     before reading a 32x discrepancy into the name). At kl_weight=10 one nat
#     per dim costs 320 in loss units against an L1 of ~0.28. Expect LeRobot's
#     kld to collapse too. If it does NOT, that is the finding.
#   * `n_decoder_layers=1` — LeRobot documents the `hs[0]` bug in the original
#     and defaults to 1, which is what we do.
#   * Image normalization matches: LeRobot's `use_imagenet_stats=True` applies
#     0.485/0.456/0.406 and 0.229/0.224/0.225, bit-identical to our constants.
#     (`act-main` uses `/255` and NOTHING else — it is the odd one out.)
#
# Known deviations still on our side, which this run is meant to price:
# trainable BatchNorm where both references freeze it, and lr 1e-4 where both
# use 1e-5.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ID="${REPO_ID:-DenisLabs/record-test_20260828_092736}"
MODE="${1:-defaults}"

# Our 40 training episodes, then our 10 validation episodes. Order matters.
TRAIN_EPS="0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,32,33,35,36,38,39,40,42,44,47,48,49"
VAL_EPS="4,18,30,31,34,37,41,43,45,46"
EPISODES="[${TRAIN_EPS},${VAL_EPS}]"

# ⚠ `--policy.push_to_hub=false`. It defaults to TRUE, and `validate()` then
# refuses to start without a `--policy.repo_id`
# (`configs/train.py:329`) — a fail-fast that fires AFTER the dataset is
# resolved, so it costs a download before it tells you.
#
# ⚠ The output directory must NOT already exist (`configs/train.py:334`,
# FileExistsError, no --force). Timestamped so a second run of the same mode
# does not fail on the first one's directory.
STAMP="$(date +%Y%m%d_%H%M%S)"

COMMON=(
  # `--policy.type` FIRST: draccus resolves the policy config class from it,
  # and the other `--policy.*` flags have nowhere to land until it has.
  --policy.type=act
  --policy.device=cuda
  --policy.push_to_hub=false
  --dataset.repo_id="${REPO_ID}"
  --dataset.episodes="${EPISODES}"
  --dataset.eval_split=0.2
  --steps=40000          # ~26 epochs at batch 8; ours plateaus by epoch 8-15
  --eval_steps=1000      # matches our VAL_EVERY
  --log_freq=200
  --save_freq=10000
  --wandb.enable=false
  --seed=7
)

case "${MODE}" in
  defaults)
    # ACT as LeRobot ships it: dim 512, ff 3200, chunk 100, lr 1e-5,
    # ImageNet backbone, FrozenBatchNorm, native 480x640.
    #
    # ANSWERS: "what does ACT achieve on these 50 episodes?" If this also lands
    # near 0.39, our implementation is fine and the ceiling is the data. If it
    # reaches 0.20, we have a real defect and `matched` becomes the bisect.
    echo "=== LeRobot ACT, stock defaults ==="
    lerobot-train "${COMMON[@]}" \
      --output_dir="outputs/act_so101_lerobot_defaults_${STAMP}" \
      --job_name=act_so101_lerobot_defaults
    ;;

  matched)
    # Our geometry, so a gap cannot be explained by capacity or horizon.
    # ⚠ Only worth running if `defaults` showed a gap — otherwise it costs
    # hours to confirm something already known.
    #
    # K=60 is 2.0 s at 30 fps, the paper's horizon in seconds rather than its
    # frame count (ALOHA is 50 Hz x 100 steps). `n_action_steps` must equal
    # `chunk_size` when temporal ensembling is off, which is LeRobot's default.
    echo "=== LeRobot ACT, matched to our configuration ==="
    lerobot-train "${COMMON[@]}" \
      --policy.dim_model=256 \
      --policy.dim_feedforward=1024 \
      --policy.chunk_size=60 \
      --policy.n_action_steps=60 \
      --batch_size=16 \
      --output_dir="outputs/act_so101_lerobot_matched_${STAMP}" \
      --job_name=act_so101_lerobot_matched
    ;;

  *)
    echo "usage: $0 [defaults|matched]" >&2
    exit 2
    ;;
esac

cat <<'EOF'

Done. What to read out of it:

  grep -E "eval_loss|l1_loss|kld_loss" outputs/act_so101_lerobot_*/*.log

  * the eval_loss PLATEAU, against our 0.3941 (allowing the ~5% denominator
    difference in caveat 2 above)
  * whether kld_loss collapses toward 0 as ours does. If it holds up, the
    difference is ours and worth hunting.
  * the epoch at which eval_loss turns, against our 7.7.
EOF
