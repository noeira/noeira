#!/usr/bin/env bash
# +--------------------------------------------------------------------------+ #
# | LeWM training under a supervisor: restarted from the last resume state
# +--------------------------------------------------------------------------+ #
# `examples/lewm/lewm_pusht_train_ref.mojo --resume` continues from
# `<out>/latest` (or starts fresh when there is none), so a crashed run is
# simply relaunched. Up to MAX_ATTEMPTS, a minute apart; every attempt and its
# exit code go to `<out>/supervisor.log`, the run's output to `<out>/run.log`.
#
#   BIN=/workspace/train_bin OUT=/workspace/lewm_train EPOCHS=10 \
#       setsid nohup bash tools/lewm/train_supervisor.sh > /dev/null 2>&1 < /dev/null &
#
# BIN is a `mojo build` of the driver (built once: a supervisor that rebuilt
# would pick up whatever the tree holds at the time of the crash). Extra
# driver flags go in ARGS (e.g. ARGS="--h5 ... --split ... --init ...").
set -u
BIN=${BIN:-/workspace/train_bin}
OUT=${OUT:-/workspace/lewm_train}
EPOCHS=${EPOCHS:-10}
SAVE_EVERY=${SAVE_EVERY:-2000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
ARGS=${ARGS:-}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== attempt $attempt $(date -u +%FT%TZ)" >> "$OUT/supervisor.log"
  pixi run -e nvidia "$BIN" --out "$OUT" --epochs "$EPOCHS" --save-every "$SAVE_EVERY" \
     --log-every 100 --resume $ARGS >> "$OUT/run.log" 2>&1
  rc=$?
  echo "=== attempt $attempt exited rc=$rc $(date -u +%FT%TZ)" >> "$OUT/supervisor.log"
  if [ $rc -eq 0 ]; then echo TRAIN-DONE >> "$OUT/supervisor.log"; exit 0; fi
  df -h "$OUT" | tail -1 >> "$OUT/supervisor.log"
  sleep 60
done
echo TRAIN-GAVE-UP >> "$OUT/supervisor.log"
exit 1
