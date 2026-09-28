#!/usr/bin/env bash
# Sweep ONE comptime knob in newton_solve.mojo across values, one build per
# value, and collect a full attribution leg for each.
#
#   # WHICH serial term costs what (bit-exact at every value):
#   KNOB=NEWTON_SERIAL_PROBE VALUES="0 1 2 3 4" \
#       pixi run -e nvidia bash scripts/p0_knob_sweep.sh
#
#   # Is the cost per-ITERATION or per-SOLVE? (NOT bit-exact — timing only):
#   KNOB=NEWTON_MIN_ITER VALUES="0 20 100 180" \
#       pixi run -e nvidia bash scripts/p0_knob_sweep.sh
#
#   # F3 step 2 — OCCUPANCY. Threads per block = max(MAX_CONTACTS, FLOOR),
#   # where FLOOR is this knob only for blocks over NEWTON_SMALL_BLOCK_MAX_BYTES
#   # (smaller ones take NEWTON_THREADS_FLOOR_SMALL_BLOCK — sweep that knob for
#   # the park k <= 6 legs). Bit-exact at every value; lives in je_budget.mojo:
#   KNOB_SRC=noeira/physics3d/solver/je_budget.mojo \
#   KNOB=NEWTON_THREADS_FLOOR VALUES="16 32 64 128" \
#       pixi run -e nvidia bash scripts/p0_knob_sweep.sh
#
# This is one script rather than one per knob on purpose: the sed/restore/
# fingerprint logic is the part that must not drift between sweeps, and it was
# already copied once.
#
#   # Where in the SETUP is the time? (WRONG ANSWER — timing only):
#   KNOB=NEWTON_STOP_AFTER VALUES="0 1 2 3 4" \
#       pixi run -e nvidia bash scripts/p0_knob_sweep.sh
#
# ⚠⚠ THE MAGNITUDE IS THE NON-VACUITY CHECK, and it differs per knob. For
# SERIAL_PROBE at REPEAT=10 a term worth 5% of newton must move newton ~45%; a
# term inside the noise floor (~1.7% here for kernels over 1 ms/step) has either
# been optimised away or is genuinely free, and THOSE ARE NOT THE SAME ANSWER —
# re-run it at REPEAT=100 before believing it is free. For MIN_ITER, `t(N)` must
# be FLAT then LINEAR; if it is linear from the very first value the knee is
# below the smallest N and the sweep must be redone lower.
#
# ⚠ Restores the knob to ITS ORIGINAL VALUE on ANY exit, including Ctrl-C. Check `git diff`
# after a crash anyway: a probe build left in the tree is a wrong production
# binary, and `NEWTON_MIN_ITER` is NOT bit-exact.
set -uo pipefail

# ⚠⚠ `KNOB_SRC`, NOT `SRC`. The pixi nvidia environment's ACTIVATION exports
# `SRC` (it points at the CUDA interceptor's C source), so a `SRC=... pixi run`
# prefix is overwritten before this script ever sees it and the sweep tries to
# sed a knob into `cuda_intercept.c`. Anything read from the ambient
# environment needs a name that is not a three-letter English word.
KNOB_SRC=${KNOB_SRC:-noeira/physics3d/solver/newton_solve.mojo}
KNOB=${KNOB:?set KNOB, e.g. KNOB=NEWTON_MIN_ITER}
VALUES=${VALUES:?set VALUES, e.g. VALUES="0 8 16 32"}
KEY="comptime ${KNOB}: Int"
export KS=${KS:-"0 13"}
TAG=${TAG:-$(echo "$KNOB" | tr 'A-Z_' 'a-z-')}

command -v mojo >/dev/null || {
  echo "!! run inside the pixi env: pixi run -e nvidia bash $0"; exit 1; }
grep -q "^${KEY} = " "$KNOB_SRC" || { echo "!! cannot find '$KEY' in $SRC"; exit 1; }

# ⚠ RESTORE WHAT WAS THERE, NOT A HARD-CODED 0. Production is 0 for the probe
# knobs and 1 for `NEWTON_THREADS_MULT` (a 0 there would be clamped by
# `_max_one`, so it would not break — it would just leave the tree quietly
# differing from the commit, which is worse).
ORIG=$(sed -n "s/^${KEY} = \(.*\)$/\1/p" "$KNOB_SRC")
[ -n "$ORIG" ] || { echo "!! could not read the current value of $KNOB"; exit 1; }
echo "-- ${KNOB} is currently $ORIG; that is what gets restored"

_set() { sed -i "s/^${KEY} = .*/${KEY} = $1/" "$KNOB_SRC"; }
_restore() { _set "$ORIG"; echo "-- restored ${KNOB} = $ORIG"; }
trap _restore EXIT INT TERM

DIRS=()
for V in $VALUES; do
  echo "############## ${KNOB} = $V"
  _set "$V"
  grep -n "^${KEY} = " "$KNOB_SRC"
  D="p0_${TAG}_${V}"
  OUT="$D" bash scripts/p0_attrib.sh || { echo "!! leg $V failed"; exit 1; }
  DIRS+=("$D")
done

echo
echo "=== build fingerprints — EVERY ONE MUST DIFFER ==="
echo "    (equal hashes mean two arms are the same program; that voided a whole"
echo "     sweep once, and it looked like a clean negative result)"
for D in "${DIRS[@]}"; do
  printf "  %-28s " "$D"
  sed -n 's/^binary_md5 *//p' "$D/BUILD.txt" 2>/dev/null || echo "(missing)"
done
