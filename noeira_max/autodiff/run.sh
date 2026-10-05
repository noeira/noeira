#!/usr/bin/env bash
# Run Python for the autodiff prototype through the MAIN checkout's pixi env,
# so a worktree shares its MAX compile cache, with this checkout first on
# PYTHONPATH. Works from the main checkout too.
#
#   noeira_max/autodiff/run.sh -m unittest discover -s noeira_max/autodiff/tests -t .
#   noeira_max/autodiff/run.sh noeira_max/autodiff/m0/probe_transform.py
#   AUTODIFF_ENV=act-ref noeira_max/autodiff/run.sh ...   # another pixi env
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
cd "$ROOT"
exec pixi run --manifest-path "$MAIN/pixi.toml" -e "${AUTODIFF_ENV:-default}" \
    env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python "$@"
