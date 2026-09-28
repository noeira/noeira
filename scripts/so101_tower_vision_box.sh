#!/usr/bin/env bash
# so101_tower vision student on a CUDA box: expert demos -> rendered store -> ACT.
#
#   bash scripts/so101_tower_vision_box.sh            # all three stages
#   STAGES="render" bash scripts/so101_tower_vision_box.sh
#   ACT_STEPS=20000 bash scripts/so101_tower_vision_box.sh
#   DR=full DR_SEED=1 bash scripts/so101_tower_vision_box.sh   # a DR store
#     (its own file name; the preview PNGs land beside it — look first)
#
# Run it from the repo root, inside tmux: the render and the training outlive
# an SSH drop. Every stage skips work whose output already exists, so a rerun
# after a failure resumes where it stopped.
#
# Stage `demos`: the two cube-in-bowl expert files. Copy them from the laptop
# if they exist there (41 MB, seconds):
#     scp projects/so101-tower/demos/expert_cube_in_bowl_*_300.demo \
#         <box>:<repo>/projects/so101-tower/demos/
# otherwise they are regenerated here on the CPU from the same seeds (the
# success set can differ by a few episodes across CPUs; it is training data,
# not a gate).
#
# Stage `render`: `tower_demo_rerender` on both files -> one store (~96k rows,
# ~3 GB with deflate). It checks itself on its first launch (device vs host
# tracer + a negative control) and refuses on a mismatch.
#
# Stage `act`: `act_so101_train_gpu` with ACT_STORE on that store. The
# ImageNet ResNet18 comes from the Hub on the first run (no PyTorch).
#
# ⚠ COMPILES ARE THE SLOW PART ON A RENTED BOX: the expert ~3 min, the
# rerender (tracer kernel) and the ACT graph ~6 min each. The binaries land in
# build/so101_vision/ and are reused when their source has not changed.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGES="${STAGES:-demos render act}"
D=projects/so101-tower/demos
B=build/so101_vision
DR="${DR:-off}"          # off | light | full — render-time domain randomization
DR_SEED="${DR_SEED:-0}"
STORE=$D/expert_cube_in_bowl_418.rendered.h5
[[ $DR == off ]] || STORE=$D/expert_cube_in_bowl_418_dr${DR}_s${DR_SEED}.rendered.h5
CLEAN=$D/expert_cube_in_bowl_clean_300.demo
NOISY=$D/expert_cube_in_bowl_flat02_300.demo
mkdir -p "$D" "$B"

log() { printf '\n=== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }

# build SRC -> OUT, only when OUT is missing or older than SRC
build() {
    local src=$1 out=$2
    if [[ ! -x $out || $src -nt $out ]]; then
        log "build $src"
        pixi run -e nvidia mojo build -I . -o "$out" "$src"
    fi
}

has() { [[ " $STAGES " == *" $1 "* ]]; }

pixi install -e nvidia >/dev/null

if has demos; then
    if [[ -s $CLEAN && -s $NOISY ]]; then
        log "demos: present, skipped"
    else
        build examples/so101/tower_expert_record.mojo "$B/tower_expert"
        log "demos: regenerating on the CPU (two in parallel)"
        [[ -s $CLEAN ]] || "$B/tower_expert" so101_tower_cube_in_bowl \
            --episodes 300 --seed 11000 --quiet --out "$CLEAN" \
            > "$B/gen_clean.log" 2>&1 &
        [[ -s $NOISY ]] || "$B/tower_expert" so101_tower_cube_in_bowl \
            --episodes 300 --seed 14000 --noise 0.02 --flat-noise --quiet \
            --out "$NOISY" > "$B/gen_noisy.log" 2>&1 &
        wait
        tail -n 2 "$B"/gen_*.log
    fi
fi

if has render; then
    if [[ -s $STORE ]]; then
        log "render: $STORE present, skipped (delete it to re-render)"
    else
        build examples/so101/tower_demo_rerender.mojo "$B/tower_rerender"
        log "render: $CLEAN + $NOISY -> $STORE"
        "$B/tower_rerender" --demos "$CLEAN,$NOISY" --out "$STORE.partial" \
            --dr "$DR" --dr-seed "$DR_SEED" --dr-preview 8 \
            2>&1 | grep -v '^Warning: attached model' | tee "$B/render.log"
        mv "$STORE.partial" "$STORE"
        mv "$STORE.partial.overhead.row0.png" "$B/overhead.row0.png" 2>/dev/null || true
        mv "$STORE.partial.wrist.row0.png" "$B/wrist.row0.png" 2>/dev/null || true
    fi
fi

if has act; then
    build examples/so101/act_so101_train_gpu.mojo "$B/act_train_gpu"
    if [[ -d projects/so101-tower ]] && [[ -f projects/so101-tower/project.kv ]]; then
        export ACT_PROJECT="${ACT_PROJECT:-so101-tower}"
    fi
    log "act: ACT_STORE=$STORE ACT_PROJECT=${ACT_PROJECT:-so101} ACT_STEPS=${ACT_STEPS:-default}"
    ACT_STORE="$STORE" "$B/act_train_gpu" 2>&1 | tee "$B/act_train.log"
fi
log "done"
