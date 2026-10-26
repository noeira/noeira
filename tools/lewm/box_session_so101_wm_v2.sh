#!/usr/bin/env bash
# +--------------------------------------------------------------------------+ #
# | SO-101 world model v2 box session: S0 data, then the proprio model
# +--------------------------------------------------------------------------+ #
# noeira-docs/SO101_LEWM_PLAN.md S1 v2. Rebuilds the S0 sim data exactly as
# v1 had it (seeded, same flags), then trains the model with proprioception
# in the latent (`-D SO101_WM_PROPRIO=1`). Bring to the box:
#   TEACHER  the teacher run dir (laptop: runs/2026-10-08_ppo-so101_tower_
#            cube_in_bowl_cc10f7bb, 2.6 MB) — copied to $REPO/runs/ if absent
#
#   D=/workspace/so101_v2 setsid nohup bash tools/lewm/box_session_so101_wm_v2.sh \
#       > /dev/null 2>&1 < /dev/null &
#
# Stages (each writes $D/<stage>.done; a rerun skips finished ones):
#   build   the probe, the rerenderer, the v2 trainer           ~15 min
#   record  2,000 teacher episodes under the real-fitted lag     ~1 min
#           model: 1,200 at exec noise 0.1, 400 at 0.2, 400 at 0
#   render  both cameras, room DR, 112 x 112, 3 stores, 5.5 GB   ~35 min
#   train   8 epochs, batch 128 (~18,400 steps at ~0.42 s)      ~2 h 15
# Then read $D/summary.txt; fetch $D/wm_v2/{epoch_7,action_stats.txt,
# prop_stats.txt,log.csv} for the laptop's offline gates
# (`lewm_so101_real_check -D SO101_WM_PROPRIO=1`).
# STAGES picks stages (default "build record render train"); EPOCHS (8).
# DR_PER (launch): `episode` renders each episode under ONE look instead of a
# redraw every 32 rows (plan §3c step 1: the in-episode redraws were most of
# v1's copy error); `launch` is how S0 was rendered.
set -u
D=${D:-/workspace/so101_v2}
STAGES=${STAGES:-build record render train}
EPOCHS=${EPOCHS:-8}
DR_PER=${DR_PER:-launch}
cd "$(dirname "$0")/../.."
REPO=$(pwd)
mkdir -p "$D" build
command -v pixi > /dev/null || export PATH=/workspace/.pixi-bin/bin:$HOME/.pixi/bin:$PATH
# the shell-hook before `set -u`'s reach: it reads NVCC_PREPEND_FLAGS unset.
# PIXI_ENV: `nvidia` where installed, else `default` (= the nvidia feature
# set; a box that has only it would pay ~11 GB of disk for a second copy)
PIXI_ENV=${PIXI_ENV:-$([ -d .pixi/envs/nvidia ] && echo nvidia || echo default)}
set +u
eval "$(pixi shell-hook -e "$PIXI_ENV")"
set -u
RUN=runs/2026-10-08_ppo-so101_tower_cube_in_bowl_cc10f7bb
LAG=(--lag-tau 30,120 --lag-delay 1,3
     --lag-tau-j "30,120;30,120;30,120;30,120;30,120;100,160"
     --lag-vmax-j "1.0,1.25;1.0,1.25;1.0,1.25;1.0,1.25;0.6,1.25;0.8,2.5"
     --lag-offset-j "0;0,0.04;0.02,0.05;0;0;0" --lag-period 32,38 --elbow-max 1.56)

say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$D/summary.txt"; }
want() { [[ " $STAGES " == *" $1 "* ]] && [ ! -f "$D/$1.done" ]; }

if want build; then
  say "build"
  # the rerenderer's PNG check writes through the HTTP shim's library; a
  # fresh worktree has none (git-ignored build product)
  [ -f noeira/io/libnra_http.so ] || pixi run -e "$PIXI_ENV" build-http > "$D/build_http.log" 2>&1 \
    || { say "BUILD-FAILED http shim"; exit 1; }
  mojo build -I . -D TASK_PPO_ACT_HIST=3 -D TASK_PPO_TARGET_OBS \
    examples/so101/ppo_state_probe_sim.mojo -o build/ppo_state_probe > "$D/build_probe.log" 2>&1 \
    || { say "BUILD-FAILED probe"; exit 1; }
  mojo build -I . examples/so101/tower_demo_rerender.mojo -o build/rerender \
    > "$D/build_rerender.log" 2>&1 || { say "BUILD-FAILED rerender"; exit 1; }
  mojo build -I . -D SO101_WM_PROPRIO=1 examples/lewm/lewm_so101_train.mojo \
    -o build/so101_train_v2 > "$D/build_train.log" 2>&1 || { say "BUILD-FAILED train"; exit 1; }
  touch "$D/build.done"
fi

if want record; then
  [ -d "$RUN" ] || { say "MISSING $RUN (copy the teacher run dir)"; exit 1; }
  say "record"
  build/ppo_state_probe --run $RUN --episodes 1200 --seed 10000 --quiet 1 "${LAG[@]}" \
    --demos "$D/teacher_n10.demo" --exec-noise 0.1 > "$D/rec_n10.log" 2>&1 &
  build/ppo_state_probe --run $RUN --episodes 400 --seed 20000 --quiet 1 "${LAG[@]}" \
    --demos "$D/teacher_n20.demo" --exec-noise 0.2 > "$D/rec_n20.log" 2>&1 &
  build/ppo_state_probe --run $RUN --episodes 400 --seed 30000 --quiet 1 "${LAG[@]}" \
    --demos "$D/teacher_n00.demo" > "$D/rec_n00.log" 2>&1 &
  wait
  for n in n10 n20 n00; do say "  $(grep -h 'episodes reached' "$D/rec_$n.log")"; done
  touch "$D/record.done"
fi

if want render; then
  say "render, dr per $DR_PER"
  for n in n10 n20 n00; do
    s=$([ $n = n10 ] && echo 11 || ([ $n = n20 ] && echo 22 || echo 33))
    build/rerender --demos "$D/teacher_$n.demo" --all-episodes --episodes 100000 \
      --out "$D/teacher_$n.rendered.h5" --dr room --dr-seed $s --dr-per "$DR_PER" \
      --joint-zero follower --resize 112 > "$D/render_$n.log" 2>&1 &
  done
  wait
  for n in n10 n20 n00; do
    grep -q "=== OK ===" "$D/render_$n.log" || { say "RENDER-FAILED $n"; exit 1; }
  done
  say "  $(du -sh "$D"/*.rendered.h5 | tr '\n' ' ')"
  touch "$D/render.done"
fi

if want train; then
  say "train v2, $EPOCHS epochs"
  build/so101_train_v2 \
    --stores "$D/teacher_n10.rendered.h5,$D/teacher_n20.rendered.h5,$D/teacher_n00.rendered.h5" \
    --epochs "$EPOCHS" --log-every 100 --out "$D/wm_v2" > "$D/train_v2.log" 2>&1
  rc=$?
  say "  train exit $rc | $(grep -h 'val pred [0-9]' "$D/train_v2.log" | tail -1 | cut -c1-120)"
  [ $rc -eq 0 ] && touch "$D/train.done"
fi
say "SESSION-END"
