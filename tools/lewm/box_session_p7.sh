#!/usr/bin/env bash
# +--------------------------------------------------------------------------+ #
# | LeWM box session P7: multi-seed boards, then AdaJEPA E1 / E2 / E3
# +--------------------------------------------------------------------------+ #
# docs/LEWM_REOPEN_PLAN.md P6.2 (open measurement) and P7. NO DATASET NEEDED:
# every run plays box session A's 50 fixture pairs. Bring to the box:
#   RUN  the P6.2 checkpoints (laptop: ~/.cache/noeira/lewm_pusht/runs/
#        p6_2026-10-02/lewm_train — epoch_0, epoch_1, epoch_7 for the
#        AdaJEPA part, epoch_4 too for the boards; ~70 MB each)
#   FIX  the session-A fixture (laptop: ~/.cache/noeira/lewm_pusht/session_a/
#        out/fixture, ~195 MB)
#
#   SKIP_BOARDS=1 RUN=/workspace/lewm_train FIX=/workspace/fixture \
#       OUT=/workspace/p7 setsid nohup bash tools/lewm/box_session_p7.sh \
#       > /dev/null 2>&1 < /dev/null &
#
# Then read $OUT/summary.txt. The driver's defaults are the official AdaJEPA
# protocol (references/adajepa-main): one block per replan, warm start,
# budget 100, predictor last block + final norm + encoder projector, a fresh
# Adam per adaptation. COST picks the planning cost (`PlanCost`), default
# AdaJEPA's `staged`: LeWM's last-step cost PROCRASTINATES when one block is
# executed per replan (laptop 2026-10-04, epoch_7 frozen, 9 pairs, budget 50:
# last 3/9 margin 1.73, all:2 8/9 0.50, staged 9/9 0.43; receding 5 9/9 0.38).
# Each config logs success, the mean best margin (< 1 =
# success; far less noisy per pair) and the mean prediction loss per arm.
# On an RTX 5090: a 50-pair board ~2 min; an AdaJEPA config ~45 min for both
# arms at budget 100 — E1 ~2.3 h, E2 ~2.3 h, E3 ~4 h, ordered so the session
# can be stopped after any block. SKIP_BOARDS=1 skips the boards (done
# 2026-10-03: epoch_4 95.6 %, epoch_7 97.2 % over 5 seeds). BLOCKS picks the
# AdaJEPA blocks (default "e1 e2 e3"; E1/E2 done 2026-10-04: no success gain,
# yet the prediction loss drops 27 % on epoch_0 and 71 % under dark). E4
# probes those two cases: stop-gradient target, 10x LR, λ = 0, the whole model.
set -u
RUN=${RUN:-/workspace/lewm_train}
FIX=${FIX:-/workspace/fixture}
OUT=${OUT:-/workspace/p7}
COST=${COST:-staged}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }

log "build (cost $COST)"
[ -n "${SKIP_BOARDS:-}" ] || \
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
  pixi run -e nvidia "$OUT/adajepa_bin" --fixture "$FIX" --episodes 50 --cost "$COST" "$@" \
    > "$OUT/ada_$name.log" 2>&1
  log "adajepa $name: $(grep '^frozen:' "$OUT/ada_$name.log")"
  log "    $(grep '^mean best margin' "$OUT/ada_$name.log")"
  log "    $(grep '^mean pred loss' "$OUT/ada_$name.log")"
}

# ── 2. E1: in distribution, the data-limited checkpoints and the best ─────
e1() {
  ada e1_epoch0 --dump "$RUN/epoch_0"
  ada e1_epoch1 --dump "$RUN/epoch_1"
  ada e1_epoch7 --dump "$RUN/epoch_7"
}

# ── 3. E2: visual shift (every observed frame, start and goal included;
#       the default subset adapts the encoder's projector) ─────────────────
e2() {
  ada e2_dark   --dump "$RUN/epoch_7" --shift dark:0.5
  ada e2_noise  --dump "$RUN/epoch_7" --shift noise:0.1
  ada e2_swap   --dump "$RUN/epoch_7" --shift swap
}

# ── 4. E3: ablations on the data-limited epoch_1 ──────────────────────────
e3() {
  ada e3_pred      --dump "$RUN/epoch_1" --subset pred
  ada e3_adam_ep   --dump "$RUN/epoch_1" --adam per-episode
  ada e3_lambda0   --dump "$RUN/epoch_1" --lambda 0
  ada e3_steps2    --dump "$RUN/epoch_1" --tta-steps 2
  ada e3_lr10x     --dump "$RUN/epoch_1" --tta-lr 5e-4
}

# ── 5. E4: where adapting moves the prediction loss — dark (epoch_7) and the
#       data-limited epoch_0 — does a detached target, a bigger step, no
#       SIGReg or the whole model turn it into planning? ──────────────────
e4() {
  ada e4_dark_sg      --dump "$RUN/epoch_7" --shift dark:0.5 --stop-grad-target
  ada e4_dark_lr10x   --dump "$RUN/epoch_7" --shift dark:0.5 --tta-lr 5e-4
  ada e4_dark_all_sg  --dump "$RUN/epoch_7" --shift dark:0.5 --subset all --stop-grad-target
  # noise: the encoder's cost landscape survives it (lewm_pusht_cost_landscape:
  # Spearman 0.76 vs 0.78 clean), the predictor does not (pred loss 13x)
  ada e4_noise_lr10x  --dump "$RUN/epoch_7" --shift noise:0.1 --tta-lr 5e-4
  ada e4_e0_sg        --dump "$RUN/epoch_0" --stop-grad-target
  ada e4_e0_lr10x     --dump "$RUN/epoch_0" --tta-lr 5e-4
  ada e4_e0_lambda0   --dump "$RUN/epoch_0" --lambda 0
}

for B in ${BLOCKS:-e1 e2 e3}; do $B; done

log "P7-SESSION-DONE"
