#!/usr/bin/env bash
# +--------------------------------------------------------------------------+ #
# | SO-101 world model on frozen V-JEPA 2.1 features: data, encode, train
# +--------------------------------------------------------------------------+ #
# noeira-docs/SO101_LEWM_PLAN.md, "Frozen V-JEPA world model". Bring to $D:
#   h40.demo     the 40 held-out teacher episodes the ViT-tiny checks used
#   real_vj.h5   `so101_vjepa_wm.py export-real` of the printed teleop store
# and the teacher run dir to $REPO/runs/ (see box_session_so101_wm_v2.sh).
#
#   D=/workspace/so101_vj setsid nohup bash tools/lewm/box_session_so101_vjepa_wm.sh \
#       > /dev/null 2>&1 < /dev/null &
#
# Stages (each writes $D/<stage>.done; a rerun skips finished ones):
#   setup   torch venv, vjepa2 clone, the 7.4 GB encoder pull   ~25 min, in
#           the background of build + record
#   build   the probe and the rerenderer                        ~10 min
#   record  the S0 teacher episodes (same seeds as the ViT-tiny stores)
#   heldout h40 at 240 x 320, per-episode DR (look seed 77) and plain;
#           encoded with real_vj.h5
#   chunks  the teacher episodes in CHUNK-episode .demo files, rendered at
#           240 x 320 under per-episode room DR (RENDERS at a time), each
#           encoded then deleted: a full-res row is 460,800 bytes and the
#           box has ~28 GB
#   train   the predictor (STEPS), then eval on held-out sim and real
# Then read $D/summary.txt; fetch $D/wm/ and $D/eval_*.txt.
set -u
D=${D:-/workspace/so101_vj}
E=${E:-/workspace/enc}
STAGES=${STAGES:-setup build record heldout chunks train}
CHUNK=${CHUNK:-150}
RENDERS=${RENDERS:-2}
STEPS=${STEPS:-20000}
cd "$(dirname "$0")/../.."
REPO=$(pwd)
PY=$REPO/tools/lewm/so101_vjepa_wm.py
mkdir -p "$D" build "$D/feat"
command -v pixi > /dev/null || export PATH=/workspace/.pixi-bin/bin:$HOME/.pixi/bin:$PATH
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
disk() { df -BG --output=avail /workspace | tail -1 | tr -d ' G'; }
venv() { set +u; . $E/venv/bin/activate; set -u; }

if want setup; then
  say "setup (background)"
  (
    mkdir -p $E
    if [ ! -d $E/venv ]; then
      command -v uv > /dev/null || pip install -q uv
      uv venv -q --python 3.11 $E/venv
      venv
      uv pip install -q --index-url https://download.pytorch.org/whl/cu128 torch torchvision
    fi
    venv
    uv pip install -q timm einops numpy h5py scikit-learn pillow requests
    cd $E
    [ -d vjepa2 ] || git clone -q --depth 1 https://github.com/facebookresearch/vjepa2.git
    [ -f $E/vjepa21_target_encoder_bf16.pt ] || python $REPO/tools/lewm/extract_torch_subdict_http.py \
      https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt target_encoder \
      $E/vjepa21_target_encoder_bf16.pt --bf16
    python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
  ) > "$D/setup.log" 2>&1 && touch "$D/setup.done" &
  SETUP_PID=$!
fi

if want build; then
  say "build"
  [ -f noeira/io/libnra_http.so ] || pixi run -e "$PIXI_ENV" build-http > "$D/build_http.log" 2>&1 \
    || { say "BUILD-FAILED http shim"; exit 1; }
  mojo build -I . -D TASK_PPO_ACT_HIST=3 -D TASK_PPO_TARGET_OBS \
    examples/so101/ppo_state_probe_sim.mojo -o build/ppo_state_probe > "$D/build_probe.log" 2>&1 \
    || { say "BUILD-FAILED probe"; exit 1; }
  mojo build -I . examples/so101/tower_demo_rerender.mojo -o build/rerender \
    > "$D/build_rerender.log" 2>&1 || { say "BUILD-FAILED rerender"; exit 1; }
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
  wait $(jobs -p | grep -v "^${SETUP_PID:-x}$")
  for n in n10 n20 n00; do say "  $(grep -h 'episodes reached' "$D/rec_$n.log")"; done
  touch "$D/record.done"
fi

render() {  # demo out [dr seed]
  local dr=()
  [ $# -ge 3 ] && dr=(--dr room --dr-seed "$3" --dr-per episode)
  build/rerender --demos "$1" --all-episodes --episodes 100000 --out "$2" "${dr[@]}" \
    --joint-zero follower > "$2.log" 2>&1 && grep -q "=== OK ===" "$2.log"
}

wait_setup() {
  [ -n "${SETUP_PID:-}" ] && wait "$SETUP_PID"
  [ -f "$D/setup.done" ] || { say "SETUP-FAILED"; tail -5 "$D/setup.log" >> "$D/summary.txt"; exit 1; }
}

encode() {  # store out.npz
  python $PY encode --store "$1" --out "$2" --hub $E/vjepa2 --weights $E/vjepa21_target_encoder_bf16.pt \
    > "$2.log" 2>&1
}

if want heldout; then
  say "heldout render"
  render "$D/h40.demo" "$D/h40_dr.rendered.h5" 77 &
  p1=$!
  render "$D/h40.demo" "$D/h40_plain.rendered.h5" &
  p2=$!
  wait $p1 && wait $p2 || { say "RENDER-FAILED heldout"; exit 1; }
  wait_setup
  venv
  for s in h40_dr h40_plain; do
    encode "$D/$s.rendered.h5" "$D/feat/$s.npz" || { say "ENCODE-FAILED $s"; exit 1; }
  done
  # the upload marks itself complete with real_vj.h5.ok
  while [ ! -f "$D/real_vj.h5.ok" ]; do sleep 30; done
  encode "$D/real_vj.h5" "$D/feat/real.npz" || { say "ENCODE-FAILED real"; exit 1; }
  say "  $(tail -1 "$D/feat/real.npz.log") | disk $(disk) GB"
  rm -f "$D"/h40_*.rendered.h5
  touch "$D/heldout.done"
fi

if want chunks; then
  wait_setup
  venv
  if [ ! -f "$D/split.done" ]; then
    for n in n10 n20 n00; do python $PY split "$D/teacher_$n.demo" "$D/split" "$CHUNK" >> "$D/split.log"; done
    touch "$D/split.done"
  fi
  say "chunks: $(ls "$D"/split/*.demo | wc -l) of $CHUNK episodes, $RENDERS renders at a time"
  # renderers in the background, RENDERS at a time; the encoder follows in order
  (
    i=0
    for f in "$D"/split/*.demo; do
      c=$(basename "$f" .demo)
      [ -f "$D/feat/$c.npz" ] && continue
      while [ "$(jobs -rp | wc -l)" -ge "$RENDERS" ]; do sleep 5; done
      while [ "$(disk)" -lt 8 ]; do sleep 20; done
      seed=$((1000 + i)); i=$((i + 1))
      ( render "$f" "$D/$c.rendered.h5" $seed && touch "$D/$c.rendered.ok" || touch "$D/$c.rendered.bad" ) &
    done
    wait
  ) &
  for f in "$D"/split/*.demo; do
    c=$(basename "$f" .demo)
    [ -f "$D/feat/$c.npz" ] && continue
    while [ ! -f "$D/$c.rendered.ok" ] && [ ! -f "$D/$c.rendered.bad" ]; do sleep 10; done
    [ -f "$D/$c.rendered.bad" ] && { say "RENDER-FAILED $c"; exit 1; }
    encode "$D/$c.rendered.h5" "$D/feat/$c.npz" || { say "ENCODE-FAILED $c"; exit 1; }
    say "  $c: $(tail -1 "$D/feat/$c.npz.log" | cut -c1-80) | disk $(disk) GB"
    rm -f "$D/$c.rendered.h5" "$D/$c.rendered.ok"
  done
  wait
  touch "$D/chunks.done"
fi

if want train; then
  venv
  say "train, $STEPS steps"
  feats=$(ls "$D"/feat/teacher_*.npz | paste -sd,)
  python $PY train --feats "$feats" --out "$D/wm" --steps "$STEPS" > "$D/train.log" 2>&1 \
    || { say "TRAIN-FAILED"; tail -5 "$D/train.log" >> "$D/summary.txt"; exit 1; }
  say "  $(tail -1 "$D/train.log")"
  python $PY eval --model "$D/wm" --set sim_dr:"$D/feat/h40_dr.npz" --set sim_plain:"$D/feat/h40_plain.npz" \
    --set real:"$D/feat/real.npz" > "$D/eval_final.txt" 2>&1 || say "EVAL-FAILED"
  cat "$D/eval_final.txt" >> "$D/summary.txt"
  touch "$D/train.done"
fi
say "VJ-END"
