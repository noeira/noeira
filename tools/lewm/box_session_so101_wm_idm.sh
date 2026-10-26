#!/usr/bin/env bash
# +--------------------------------------------------------------------------+ #
# | SO-101 world models v1 + v2 with inverse dynamics, on per-episode looks
# +--------------------------------------------------------------------------+ #
# noeira-docs/SO101_LEWM_PLAN.md, "Retrained on per-episode looks (23 Oct)":
# both models collapsed onto the episode's look once each episode was
# rendered under one draw. This retrains them with H-JEPA's inverse-dynamics
# term (`-D SO101_WM_IDM=1`, weight IDM_W) on the same stores, the two in
# parallel on one GPU.
#
#   D=/workspace/so101_ep setsid nohup bash tools/lewm/box_session_so101_wm_idm.sh \
#       > /dev/null 2>&1 < /dev/null &
#
# The stores: $D/teacher_n{10,20,00}.rendered.h5 from
# `box_session_so101_wm_v2.sh` with DR_PER=episode. Missing, this runs its
# build, record and render stages first (TEACHER run dir needed, see there;
# ~50 min). Then:
#   build   the two IDM trainers                                ~15 min
#   train   EPOCHS (8) each, batch 128, both at once            ~2 h 30
# Writes $D/wm_v1_idm, $D/wm_v2_idm (epoch_<k>/, action_stats.txt,
# prop_stats.txt for v2, log.csv) and $D/summary.txt. Fetch epoch_0, epoch_7,
# the stats and log.csv for the laptop's checks (`lewm_so101_real_check`,
# `lewm_so101_slow_features`: the dumps load into the plain v1 / v2 graphs).
# IDM_W (100, H-JEPA's DROID weight), SIGREG (0.09; theirs 0.08), MODELS
# ("v1 v2").
set -u
D=${D:-/workspace/so101_ep}
EPOCHS=${EPOCHS:-8}
IDM_W=${IDM_W:-100}
SIGREG=${SIGREG:-0.09}
MODELS=${MODELS:-v1 v2}
cd "$(dirname "$0")/../.."
mkdir -p "$D" build
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$D/summary.txt"; }
STORES="$D/teacher_n10.rendered.h5,$D/teacher_n20.rendered.h5,$D/teacher_n00.rendered.h5"

if [ ! -f "$D/teacher_n00.rendered.h5" ] || [ ! -f "$D/render.done" ]; then
  say "stores missing: S0 data, one look per episode"
  D=$D DR_PER=episode STAGES="build record render" bash tools/lewm/box_session_so101_wm_v2.sh \
    || { say "DATA-FAILED"; exit 1; }
fi

command -v pixi > /dev/null || export PATH=/workspace/.pixi-bin/bin:$HOME/.pixi/bin:$PATH
PIXI_ENV=${PIXI_ENV:-$([ -d .pixi/envs/nvidia ] && echo nvidia || echo default)}
set +u  # the shell-hook reads NVCC_PREPEND_FLAGS unset
eval "$(pixi shell-hook -e "$PIXI_ENV")"
set -u

if [ ! -f "$D/build_idm.done" ]; then
  say "build IDM trainers"
  for m in $MODELS; do
    p=$([ "$m" = v2 ] && echo 1 || echo 0)
    mojo build -I . -D SO101_WM_IDM=1 -D SO101_WM_PROPRIO=$p examples/lewm/lewm_so101_train.mojo \
      -o build/so101_train_${m}_idm > "$D/build_train_${m}_idm.log" 2>&1 \
      || { say "BUILD-FAILED $m idm"; exit 1; }
  done
  touch "$D/build_idm.done"
fi

say "train $MODELS with IDM $IDM_W, sigreg $SIGREG, $EPOCHS epochs"
for m in $MODELS; do
  build/so101_train_${m}_idm --stores "$STORES" --epochs "$EPOCHS" --log-every 100 \
    --idm "$IDM_W" --sigreg "$SIGREG" --out "$D/wm_${m}_idm" > "$D/train_${m}_idm.log" 2>&1 &
  eval "PID_$m=$!"
done
for m in $MODELS; do
  eval "wait \$PID_$m"; rc=$?
  say "  $m exit $rc | $(grep -h 'val pred [0-9]' "$D/train_${m}_idm.log" | tail -1 | cut -c1-160)"
done
say "IDM-END"
