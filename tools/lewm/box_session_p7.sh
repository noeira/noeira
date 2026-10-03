#!/usr/bin/env bash
# +--------------------------------------------------------------------------+ #
# | LeWM box session P7: multi-seed boards, then AdaJEPA E1 / E2 / E3
# +--------------------------------------------------------------------------+ #
# docs/LEWM_REOPEN_PLAN.md P6.2 (open measurement) and P7. NO DATASET NEEDED:
# every run plays box session A's 50 fixture pairs. Bring to the box:
#   RUN  the P6.2 checkpoints (laptop: ~/.cache/noeira/lewm_pusht/runs/
#        p6_2026-10-02/lewm_train — epoch_0..7, ~560 MB)
#   FIX  the session-A fixture (laptop: ~/.cache/noeira/lewm_pusht/session_a/
#        out/fixture, ~195 MB)
#
#   RUN=/workspace/lewm_train FIX=/workspace/fixture OUT=/workspace/p7 \
#       setsid nohup bash tools/lewm/box_session_p7.sh > /dev/null 2>&1 < /dev/null &
#
# Then read $OUT/summary.txt. ~4 h on an RTX 5090 (a 50-pair board ~100 s;
# an AdaJEPA config at --receding 1 ~20 min for both arms). SKIP_BOARDS=1
# runs the AdaJEPA part only (the boards do not depend on it).
set -u
RUN=${RUN:-/workspace/lewm_train}
FIX=${FIX:-/workspace/fixture}
OUT=${OUT:-/workspace/p7}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }

log "build"
pixi run -e nvidia mojo build -I . examples/lewm/lewm_pusht_column_m.mojo -o "$OUT/colm_bin" 2>&1 | grep error
pixi run -e nvidia mojo build -I . examples/lewm/lewm_pusht_adajepa.mojo -o "$OUT/adajepa_bin" 2>&1 | grep error

# ── 1. multi-seed boards (seed 0 is in the P6.2 table) ─────────────────────
for E in $([ -n "${SKIP_BOARDS:-}" ] || echo 4 7); do
  for SEED in 1 2 3 4; do
    pixi run -e nvidia "$OUT/colm_bin" --dump "$RUN/epoch_$E" --fixture "$FIX" \
      --episodes 50 --seed $SEED > "$OUT/board_e${E}_s${SEED}.log" 2>&1
    log "board epoch_$E seed $SEED: $(grep '^Column M:' "$OUT/board_e${E}_s${SEED}.log" | cut -c1-40)"
  done
done

ada() {  # name, then driver flags
  local name=$1; shift
  pixi run -e nvidia "$OUT/adajepa_bin" --fixture "$FIX" --episodes 50 "$@" \
    > "$OUT/ada_$name.log" 2>&1
  log "adajepa $name: $(grep '^frozen:' "$OUT/ada_$name.log")"
}

# ── 2. E1: in distribution, the data-limited checkpoints and the best ─────
ada e1_epoch0 --dump "$RUN/epoch_0" --receding 1
ada e1_epoch1 --dump "$RUN/epoch_1" --receding 1
ada e1_epoch7 --dump "$RUN/epoch_7" --receding 1

# ── 3. E2: visual shift (every observed frame, start and goal included) ───
ada e2_dark   --dump "$RUN/epoch_7" --receding 1 --shift dark:0.5
ada e2_noise  --dump "$RUN/epoch_7" --receding 1 --shift noise:0.1
ada e2_swap   --dump "$RUN/epoch_7" --receding 1 --shift swap

# ── 4. E3: ablations on the data-limited epoch_1 ──────────────────────────
ada e3_predlast_enclast --dump "$RUN/epoch_1" --receding 1 --subset predlast_enclast
ada e3_bn_train         --dump "$RUN/epoch_1" --receding 1 --tta-bn train
ada e3_lambda0          --dump "$RUN/epoch_1" --receding 1 --lambda 0
ada e3_steps2           --dump "$RUN/epoch_1" --receding 1 --tta-steps 2
ada e3_lr5x             --dump "$RUN/epoch_1" --receding 1 --tta-lr 2.5e-4

log "P7-SESSION-DONE"
