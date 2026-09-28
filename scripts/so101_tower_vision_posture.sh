#!/usr/bin/env bash
# so101_tower vision student, series 2: does the HUMAN-POSTURE expert make a
# student that uses REAL frames — and which training closes the APPEARANCE gap?
#
#   bash scripts/so101_tower_vision_posture.sh                  # arms H HA HD, seed 1
#   ARMS="V H" SEEDS="1 2 3" bash scripts/so101_tower_vision_posture.sh   # the full series
#   STAGES="build" bash scripts/so101_tower_vision_posture.sh    # compile + the yaw GPU test
#   PAR=1 bash scripts/so101_tower_vision_posture.sh             # one training at a time
#
# THE SETUP (all of it new since series 1, and shared by every arm): episodes
# start at the real folded rest (9d4429420), the brick's yaw is drawn
# (ce11b131f), and every store and eval uses the REAL follower's joint zero
# (`--joint-zero follower`, 05a6298d9) — every student here is meant for the
# real arm. Every demo is recorded with `--return-rest` (dc25c8b38).
#
# THE ARMS
#   H   200 episodes of `--posture human` (b971ad6fc, 4635d325a), plain render
#   HA  H's store, trained with image augmentation (`ACT_AUGMENT=default`)
#   HD  the same 200 episodes rendered with `--dr full` (the DR plan's L2),
#       trained with `ACT_AUGMENT=default`
#   V   200 episodes of the DEFAULT (vertical) expert — OFF by default
#   G   400 episodes of `--posture human` — OFF by default: series 1 measured
#       400 vs 200 expert episodes at 49.2 +- 10.2 vs 52.3 +- 9.2 % (3 seeds)
# (120 clean + 80 noisy per 200, as series 1.)
#
# WHY H / HA / HD AT ONE SEED (24 Sep): the H s1 student gets the wrist roll
# right on its own sim val (5.3 deg) and wrong on the real frames (21-22):
# the gap is APPEARANCE (VISION_STUDENT_PLAN.md §4). The arms are ranked on
# the REAL `--moving-only` error, not on the sim rate: sim success spreads
# >= 20 points across seeds, the real error of the nine series-1 students
# spread 40-47 deg, so a >= 10 deg real difference is readable from one seed.
# The sim rate is still measured — it catches an arm that breaks the grasp.
#
# THE REAL CHECK runs here only if the real store is at $REAL (8.5 GB, scp
# from the laptop's ~/.cache/noeira/act_so101/so101-tower__cube-in-bowl_240x320_undist.h5).
# Without it the box does train + sim only, and the checkpoints are scored on
# the laptop after the box is released:
#   tower_real_check --ckpt <run>/checkpoints --student-zero follower \
#       --store <undist.h5> --store-zero follower --episodes 24 --moving-only
#
# PARALLELISM: `PAR` trainings at once (default: as many as the box's RAM
# holds). Each one keeps its store's whole image column resident in host RAM
# (`ACT_RESIDENT_GB`, sized from the render log — a column that does not fit
# STREAMS at ~0.6 s/step instead of 0.13, and nothing says so).
#
# OUTPUT: build/so101_vision/posture/results.tsv, one row per run. Resumable:
# every stage skips finished work. Run inside tmux, from the repo root.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PATH=/workspace/.pixi-bin/bin:$PATH:$HOME/.pixi/bin

ARMS="${ARMS:-H HA HD}"
SEEDS="${SEEDS:-1}"
STAGES="${STAGES:-build demos render train}"
REAL="${REAL:-$HOME/.cache/noeira/act_so101/so101-tower__cube-in-bowl_240x320_undist.h5}"
D=projects/so101-tower/demos
B=build/so101_vision/posture
BIN=build/so101_vision/bin
mkdir -p "$B" "$BIN"
RES=$B/results.tsv
[[ -s $RES ]] || printf 'arm\tseed\trun\tbest_val_l1\tsim_success\tsim_no_grasp\treal_mean_err\treal_hold_err\n' > "$RES"
log() { printf '\n=== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
arm_on() { [[ " $ARMS " == *" $1 "* ]]; }
MEM_GB=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 64)
# ⚠ A container's limit is its cgroup's, not the host's /proc/meminfo: on 24 Sep
# a vast box showed 188 GB in `free` and killed the third 30 GB training at 79.
CG_MAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
if [[ $CG_MAX =~ ^[0-9]+$ ]] && (( CG_MAX / 1073741824 < MEM_GB )); then
    MEM_GB=$(( CG_MAX / 1073741824 ))
fi

grep -q ':yaw' noeira/tasks/tasks/so101_tower_cube_in_bowl.task || {
    echo "cube_in_bowl does not draw the brick's yaw: this checkout predates ce11b131f"; exit 1; }
grep -q 'base_qpos_jitter' noeira/tasks/families/so101_tower.family || {
    echo "the family does not start folded: this checkout predates 9d4429420"; exit 1; }
[[ -d projects/so101-tower && -f projects/so101-tower/project.kv ]] && export ACT_PROJECT=so101-tower

build() {  # src out [flags...]
    local src=$1 out=$2; shift 2
    if [[ ! -x $out || $src -nt $out ]]; then
        pixi run -e nvidia mojo build -j 8 "$@" -I . -o "$out" "$src" > "$out.build.log" 2>&1 \
            || { echo "BUILD FAILED: $src"; grep -A6 'error:' "$out.build.log" | head -30; return 1; }
    fi
}
if has build; then
    log "build (one at a time: ${MEM_GB} GB RAM)"
    pixi install -e nvidia > /dev/null
    build examples/so101/tower_expert_record.mojo $BIN/tower_expert
    build examples/so101/tower_demo_rerender.mojo $BIN/tower_rerender
    build examples/so101/act_so101_train_gpu.mojo $BIN/act_train_host -D ACT_HOST_DATA
    build examples/so101/tower_act_eval.mojo $BIN/tower_act_eval
    [[ -s $REAL ]] && build examples/so101/tower_real_check.mojo $BIN/tower_real_check
    log "the brick-yaw reset on CUDA (test_device_placement_gpu)"
    pixi run -e nvidia mojo run -I . tests/tasks/test_device_placement_gpu.mojo 2>&1 \
        | grep -v 'intercept\]' | tail -8
    log "build ok"
fi

# `_rr`: recorded with --return-rest (dc25c8b38) — the episodes end folded, as the real ones do
VC=$D/p2_vertical_clean_rr.demo;  VN=$D/p2_vertical_noisy_rr.demo
HC=$D/p2_human_clean_rr.demo;     HN=$D/p2_human_noisy_rr.demo
if has demos; then
    log "demos (CPU, in parallel)"
    if arm_on V; then
        [[ -s $VC ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 300 --seed 51000 \
            --return-rest --quiet --out $VC > $B/gen_vc.log 2>&1 &
        [[ -s $VN ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 300 --seed 54000 \
            --noise 0.02 --flat-noise --return-rest --quiet --out $VN > $B/gen_vn.log 2>&1 &
    fi
    # human: 62 % clean, 33 % with noise succeed (23 Sep, without --return-rest).
    # H/HA/HD need 120 + 80 kept; G needs 240 + 160.
    if arm_on G; then NHC=600; NHN=700; else NHC=260; NHN=340; fi
    [[ -s $HC ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes $NHC --seed 41000 \
        --posture human --return-rest --quiet --out $HC > $B/gen_hc.log 2>&1 &
    [[ -s $HN ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes $NHN --seed 44000 \
        --posture human --noise 0.02 --flat-noise --return-rest --quiet --out $HN > $B/gen_hn.log 2>&1 &
    wait
    for f in $B/gen_*.log; do echo "$(basename $f): $(tail -n 2 $f | head -1)"; done
fi

# the store an arm trains on: HA shares H's
store_arm() { case $1 in HA) echo H ;; *) echo "$1" ;; esac; }
augment_of() { case $1 in HA|HD) echo default ;; *) echo off ;; esac; }
store_of() { echo "$D/p2rr_$(store_arm "$1").rendered.h5"; }
render_arm() {
    local s; s=$(store_arm "$1")
    local out; out=$(store_of "$s")
    [[ -s $out ]] && return 0
    local demos caps dr=(--dr off)
    case $s in
        V) demos=$VC,$VN; caps=120,80 ;;
        H) demos=$HC,$HN; caps=120,80 ;;
        HD) demos=$HC,$HN; caps=120,80; dr=(--dr full --dr-seed 1 --dr-preview 4) ;;
        G) demos=$HC,$HN; caps=240,160 ;;
    esac
    $BIN/tower_rerender --demos "$demos" --per-file "$caps" --joint-zero follower "${dr[@]}" \
        --out "$out.partial" 2>&1 | grep -v '^Warning: attached\|intercept\]' \
        | grep -v '^  *[0-9]* episodes |' > "$B/render_$s.log"
    tail -5 "$B/render_$s.log"
    mv "$out.partial" "$out"
}
STORES=$(for a in $ARMS; do store_arm "$a"; done | sort -u | tr '\n' ' ')
if has render; then
    log "render stores: $STORES (follower joint zero)"
    for s in $STORES; do render_arm "$s" & done
    wait
    for s in $STORES; do [[ -s $(store_of "$s") ]] || { echo "render of store $s failed"; exit 1; }; done
fi

# The image column's GiB, from the render log's row count, + 1 GiB of slack:
# the whole column must be resident or the host path streams (0.6 s/step).
resident_gb() {
    local rows; rows=$(grep -o '| [0-9]* rows |' "$B/render_$(store_arm "$1").log" 2>/dev/null \
        | tail -1 | tr -dc '0-9')
    [[ -n $rows ]] || { echo 40; return; }
    echo $(( rows * 460800 / 1073741824 + 2 ))
}

train_one() {
    local a=$1 s=$2 tag="$1_s$2"
    grep -q "^$a	$s	" "$RES" && return 0
    grep -q 'best validation l1' "$B/act_$tag.log" 2>/dev/null && return 0
    ACT_STORE=$(store_of "$a") ACT_SEED=$s ACT_NO_MONITOR=1 ACT_RESIDENT_GB=$(resident_gb "$a") \
        ACT_AUGMENT=$(augment_of "$a") \
        $BIN/act_train_host > "$B/act_$tag.log" 2>&1 || { echo "train $tag FAILED"; tail -20 "$B/act_$tag.log"; return 1; }
    log "trained $tag"
}
score_one() {
    local a=$1 s=$2 tag="$1_s$2"
    grep -q "^$a	$s	" "$RES" && return 0
    local run; run=$(tr '\r' '\n' < "$B/act_$tag.log" | grep -m1 '^  run ' | awk '{print $2}')
    local val; val=$(tr '\r' '\n' < "$B/act_$tag.log" | grep -m1 'best validation l1' | awk '{print $4}')
    # the sim eval and the real check side by side (a GPU each is plenty)
    $BIN/tower_act_eval --ckpt "$run/checkpoints" --episodes 128 --joint-zero follower \
        > "$B/eval_$tag.log" 2>&1 &
    local pe=$!
    if [[ -s $REAL && -x $BIN/tower_real_check ]]; then
        $BIN/tower_real_check --ckpt "$run/checkpoints" --student-zero follower \
            --store "$REAL" --store-zero follower --episodes 24 --moving-only > "$B/real_$tag.log" 2>&1 \
            || { echo "real check $tag FAILED"; tail -20 "$B/real_$tag.log"; }
    fi
    wait $pe || { echo "eval $tag FAILED"; tail -20 "$B/eval_$tag.log"; return 1; }
    local ok ng; ok=$(grep -m1 'SUCCESS' "$B/eval_$tag.log" | awk '{print $2}')
    ng=$(grep -m1 'no grasp' "$B/eval_$tag.log" | awk '{print $3}')
    local rm="-" rh="-"
    if [[ -s $B/real_$tag.log ]]; then
        rm=$(grep -m1 '^RESULT' "$B/real_$tag.log" | grep -o 'l1_all=[^ ]*' | cut -d= -f2 || echo "-")
        rh=$(grep -m1 '^RESULT' "$B/real_$tag.log" | grep -o 'hold_ens=[^ ]*' | cut -d= -f2 || echo "-")
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$a" "$s" "$run" "$val" "$ok" "$ng" "$rm" "$rh" >> "$RES"
    log "$tag: sim $ok/128 | real err $rm (hold $rh) | $run"
}
run_one() { train_one "$1" "$2" && score_one "$1" "$2"; }
if has train; then
    # RAM per training: its resident column + ~10 GB (process, batches, eval)
    need=10
    for a in $ARMS; do g=$(( $(resident_gb "$a") + 10 )); (( g > need )) && need=$g; done
    PAR=${PAR:-$(( MEM_GB / need ))}
    (( PAR >= 1 )) || PAR=1
    log "train arms: $ARMS | seeds: $SEEDS | $PAR at once (${MEM_GB} GB RAM, ~${need} GB each)"
    for s in $SEEDS; do
        for a in $ARMS; do
            while (( $(jobs -rp | wc -l) >= PAR )); do wait -n || true; done
            log "train $a seed $s (augment $(augment_of "$a"), resident $(resident_gb "$a") GiB)"
            run_one "$a" "$s" &
            sleep 60   # staggered: the ImageNet fetch and the store load do not all land at once
        done
    done
    wait
fi

log "results"
column -t -s $'\t' "$RES" || cat "$RES"
[[ -s $REAL ]] || echo "(no real store here: score the runs above on the laptop with --moving-only)"
log "done"
