#!/bin/bash
# P3 step 2: the detector evaluation, ONE GPU model process at a time (two at
# once crashed the 16 GB laptop, 2026-11-10). Same seeded image set for all.
set -u
cd "$(dirname "$0")/../.."
M=~/.cache/noeira/datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat
O=runs/p3/detector_eval
mkdir -p $O
pixi run -e detect python tools/perception/detector_eval.py --mat $M --out $O --per-word 120 --negatives 120 --seed 0 --detectors owlv2,gdino --tag open > $O/run_open.log 2>&1
pixi run -e detect python tools/perception/detector_eval.py --mat $M --out $O --per-word 120 --negatives 120 --seed 0 --detectors sam3 --tag sam3 > $O/run_sam3.log 2>&1
