#!/usr/bin/env bash
# probe_build.sh FILE.mojo [extra mojo build flags]
#
# Compile one file the way a compile-MEMORY bisection needs it: a cold cache,
# `--mlir-timing --mlir-timing-display list`, the compiler's own peak RSS
# sampled every second (a time series too), and a WATCHDOG that kills the
# compile when the whole container's anonymous memory passes LIMIT_GB — so a
# training or render running beside it is never the kernel's OOM victim.
#
#   LIMIT_GB=50 bash tools/compile_timing/probe_build.sh build/probe/env_only.mojo
#
# Output in build/compile_probe/out_<name>/: timing.txt (the report AND the
# compiler's stderr, errors included), rss_series ("epoch kb"), peak_kb,
# watchdog (only when it fired). One summary line on stdout.
#
# Env: LIMIT_GB (default 50), JOBS (-j, default 4), ENV (pixi env, default
# nvidia). Linux only (/proc, cgroup v2 memory.stat).
#
# Used 2026-09-23 to find why tower_act_eval needed > 57 GB: its parts
# compiled in 1-6 GB each, and the culprit was `Optional[ACTTrainer[...]]`
# (docs/COMPILE_TIME_PROFILING.md §3.10). Probe parts, then PAIRS, and look
# for a pass that appears only in the combination.
set -uo pipefail
SRC=$1; shift
N=$(basename "$SRC" .mojo)
LIMIT_GB=${LIMIT_GB:-50}
O=build/compile_probe/out_$N${TAG:-}; rm -rf "$O"; mkdir -p "$O"
CACHE=$(mktemp -d)
( peak=0; while sleep 1; do
    pids=$(pgrep -f "bin/mojo build .*$N.mojo" | tr '\n' ' ')
    [[ -z $pids ]] && continue
    rss=0; for p in $pids; do r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); rss=$((rss + ${r:-0})); done
    (( rss > peak )) && peak=$rss && echo $peak > "$O/peak_kb"
    echo "$(date +%s) $rss" >> "$O/rss_series"
    anon=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || echo 0)
    if (( anon > LIMIT_GB * 1073741824 )); then
        echo "WATCHDOG: container anon $((anon/1073741824)) GB > $LIMIT_GB, killing" > "$O/watchdog"; kill $pids
    fi
  done ) & W=$!
t0=$(date +%s)
MODULAR_CACHE_DIR=$CACHE nice -n 10 pixi run -e "${ENV:-nvidia}" mojo build -j "${JOBS:-4}" \
    --mlir-timing --mlir-timing-display list "$@" -I . -o "$O/bin" "$SRC" \
    > "$O/build.log" 2> "$O/timing.txt"
rc=$?
t1=$(date +%s)
kill $W 2>/dev/null
rm -rf "$CACHE"
echo "$N exit $rc wall $((t1-t0)) s peak_rss_gb $(( $(cat "$O/peak_kb" 2>/dev/null || echo 0) / 1048576 )) $(cat "$O/watchdog" 2>/dev/null)"
