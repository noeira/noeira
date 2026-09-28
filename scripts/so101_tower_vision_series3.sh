#!/usr/bin/env bash
# so101_tower vision student, series 3: the settled scene and the stronger
# teacher — does the CALIBRATED look (and DR centred on it) move the student
# on real frames?
#
#   bash scripts/so101_tower_vision_series3.sh                  # arms C L CD, seed 1
#   ARMS="C CD" SEEDS="1 2" bash scripts/so101_tower_vision_series3.sh
#   STAGES="build real" bash scripts/so101_tower_vision_series3.sh   # compile + the real stores
#   PAR=1 bash scripts/so101_tower_vision_series3.sh             # one training at a time
#
# THE SETUP (every arm): the real-layout placements with the bowl's 135 mm
# separation (8082616e5, 657f1e9cb), the gripper's measured map and the
# follower's roll zero (`--joint-zero follower`, dbd873e15 / 2ecedf871), desk
# friction 0.5 (b8f668315), the calibrated cameras. Demos: `--posture human
# --clear-plan --return-rest` — the planner library's clean grasp (12 mm aim,
# the via-point, tilt 20..65) and the rig's LOW release in the bowl
# (d0c119794); ~72 % of draws succeed in sim.
#
# THE ARMS (the SAME demos, rendered three ways; 120 clean + 80 noisy)
#   C   the calibrated look (the default since 26a862992; marker, backdrop)
#   L   `--look legacy` (series 2's look) — the calibration's control
#   CD  the calibrated look + `--dr full` (re-centred on it, b7bad4d0f) +
#       ACT_AUGMENT=default
# Each arm x seed: ACT on the host data path, then
#   sim   tower_act_eval, 128 held-out placements, the arm's own look
#   real  tower_real_check --moving-only on the printed-props recording
#         (cube-in-bowl-printed: the blue cube and the yellow octagonal bowl,
#         the sim's own props), and on it with BOTH images blanked — the
#         student's real skill without vision. The lime-Duplo set is no longer
#         scored (26 Sep, the user: the printed set is the test).
#
# NEEDS on the box: the printed recording's raw dataset under
# projects/so101-tower/datasets/cube-in-bowl-printed (rsync from the laptop,
# ~140 MB) and the camera calibration (project-pull); the `real` stage imports
# it undistorted (7.2 GB). The demos are generated on the
# laptop (CPU) and rsync'ed to projects/so101-tower/demos/; the `demos` stage
# regenerates any that are missing.
#
# OUTPUT: build/so101_vision/series3/results.tsv. Resumable. Run inside tmux,
# from the repo root.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PATH=/workspace/.pixi-bin/bin:$PATH:$HOME/.pixi/bin

ARMS="${ARMS:-C L CD}"
SEEDS="${SEEDS:-1}"
STAGES="${STAGES:-build real demos render train}"
CACHE=$HOME/.cache/noeira/act_so101
REAL_P=$CACHE/so101-tower__cube-in-bowl-printed_240x320_undist.h5
D=projects/so101-tower/demos
B=build/so101_vision/series3
BIN=build/so101_vision/bin
mkdir -p "$B" "$BIN"
RES=$B/results.tsv
[[ -s $RES ]] || printf 'arm\tseed\trun\tbest_val_l1\tsim_success\tsim_no_grasp\tprinted_all\tprinted_roll\tprinted_hold\tprinted_blind\n' > "$RES"
log() { printf '\n=== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
# ⚠ A CONTAINER'S LIMIT IS ITS CGROUP'S, not /proc/meminfo (24 Sep: `free`
# said 188 GB, the cgroup 79, and the third training was OOM-killed)
MEM_GB=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 64)
CG_MAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
if [[ $CG_MAX =~ ^[0-9]+$ ]] && (( CG_MAX / 1073741824 < MEM_GB )); then
    MEM_GB=$(( CG_MAX / 1073741824 ))
fi

grep -q 'sep=0.135' noeira/tasks/tasks/so101_tower_cube_in_bowl.task || {
    echo "cube_in_bowl lacks the bowl's :sep=0.135: this checkout predates 657f1e9cb"; exit 1; }
grep -q 'place_mode\|--place' examples/so101/tower_expert_record.mojo || {
    echo "the expert has no --place: this checkout predates d0c119794"; exit 1; }
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
    build examples/so101/tower_real_check.mojo $BIN/tower_real_check
    build examples/so101/act_so101_import_dataset.mojo $BIN/act_import
    log "build ok"
fi

if has real; then
    log "the real store (undistorted, 240x320)"
    for ds in cube-in-bowl-printed; do
        out=$CACHE/so101-tower__${ds}_240x320_undist.h5
        [[ -s $out ]] && continue
        [[ -d projects/so101-tower/datasets/$ds ]] || { echo "missing projects/so101-tower/datasets/$ds (rsync it from the laptop)"; exit 1; }
        $BIN/act_import --project so101-tower --dataset $ds --undistort projects/so101-tower/cameras \
            > "$B/import_$ds.log" 2>&1 || { echo "import $ds FAILED"; tail -5 "$B/import_$ds.log"; exit 1; }
        tail -2 "$B/import_$ds.log"
    done
fi

HC=$D/s3_human_clean.demo; HN=$D/s3_human_noisy.demo
if has demos; then
    log "demos (CPU; skipped if rsync'ed from the laptop)"
    [[ -s $HC ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 220 --seed 71000 \
        --posture human --return-rest --clear-plan --quiet --out $HC > $B/gen_hc.log 2>&1 &
    # ⚠ TAPERED noise (no --flat-noise), unlike series 1-2: --clear-plan
    # presses the fingers into the desk, and flat noise at the close kept
    # 35/220 (16 %); tapered 49/150 (33 %); clean 159/220 (72 %)
    [[ -s $HN ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 300 --seed 74000 \
        --posture human --return-rest --clear-plan --noise 0.02 --quiet --out $HN > $B/gen_hn.log 2>&1 &
    wait
fi

store_arm() { case $1 in *) echo "$1" ;; esac; }
store_of() { echo "$D/s3_$1.rendered.h5"; }
look_of() { case $1 in L) echo legacy ;; *) echo calibrated ;; esac; }
augment_of() { case $1 in CD) echo default ;; *) echo off ;; esac; }
render_arm() {
    local a=$1 out; out=$(store_of "$1")
    [[ -s $out ]] && return 0
    local dr=(--dr off)
    [[ $a == CD ]] && dr=(--dr full --dr-seed 1 --dr-preview 4)
    $BIN/tower_rerender --demos "$HC,$HN" --per-file 120,80 --joint-zero follower \
        --look "$(look_of "$a")" "${dr[@]}" --out "$out.partial" 2>&1 \
        | grep -v '^Warning: attached\|intercept\]' | grep -v '^  *[0-9]* episodes |' > "$B/render_$a.log"
    tail -5 "$B/render_$a.log"
    mv "$out.partial" "$out"
}
if has render; then
    log "render arms: $ARMS"
    for a in $ARMS; do render_arm "$a" & done
    wait
    for a in $ARMS; do [[ -s $(store_of "$a") ]] || { echo "render of arm $a failed"; exit 1; }; done
fi

# the image column's GiB from the render log's rows, + 2: resident or it
# streams at 0.6 s/step, silently
resident_gb() {
    local rows; rows=$(grep -o '| [0-9]* rows |' "$B/render_$1.log" 2>/dev/null | tail -1 | tr -dc '0-9')
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
real_one() {  # ckpt store out [extra...]
    local ck=$1 st=$2 out=$3; shift 3
    $BIN/tower_real_check --ckpt "$ck" --student-zero follower --store "$st" --store-zero follower \
        --episodes 24 --moving-only "$@" > "$out" 2>&1 || { echo "real check FAILED: $out"; tail -8 "$out"; }
}
field() { grep -m1 '^RESULT' "$1" 2>/dev/null | grep -o "$2=[^ ]*" | cut -d= -f2 || echo "-"; }
roll() { grep -m1 '^ *wrist_roll' "$1" 2>/dev/null | awk '{print $2}' || echo "-"; }
score_one() {
    local a=$1 s=$2 tag="$1_s$2"
    grep -q "^$a	$s	" "$RES" && return 0
    local run; run=$(tr '\r' '\n' < "$B/act_$tag.log" | grep -m1 '^  run ' | awk '{print $2}')
    local val; val=$(tr '\r' '\n' < "$B/act_$tag.log" | grep -m1 'best validation l1' | awk '{print $4}')
    local ck="$run/checkpoints"
    $BIN/tower_act_eval --ckpt "$ck" --episodes 128 --joint-zero follower --look "$(look_of "$a")" \
        > "$B/eval_$tag.log" 2>&1 &
    local pe=$!
    real_one "$ck" "$REAL_P" "$B/real_p_$tag.log"
    real_one "$ck" "$REAL_P" "$B/real_pblind_$tag.log" --mask-overhead 0,0,320,240 --mask-wrist 0,0,320,240
    wait $pe || { echo "eval $tag FAILED"; tail -20 "$B/eval_$tag.log"; return 1; }
    local ok ng; ok=$(grep -m1 'SUCCESS' "$B/eval_$tag.log" | awk '{print $2}')
    ng=$(grep -m1 'no grasp' "$B/eval_$tag.log" | awk '{print $3}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$a" "$s" "$run" "$val" "$ok" "$ng" \
        "$(field "$B/real_p_$tag.log" l1_all)" "$(roll "$B/real_p_$tag.log")" \
        "$(field "$B/real_p_$tag.log" hold_ens)" "$(field "$B/real_pblind_$tag.log" l1_all)" >> "$RES"
    log "$tag: sim $ok/128 | printed $(field "$B/real_p_$tag.log" l1_all) (hold $(field "$B/real_p_$tag.log" hold_ens), blind $(field "$B/real_pblind_$tag.log" l1_all)) | $run"
}
run_one() { train_one "$1" "$2" && score_one "$1" "$2"; }
if has train; then
    # RAM per training: its resident column + ~10 GB
    need=10
    for a in $ARMS; do g=$(( $(resident_gb "$a") + 10 )); (( g > need )) && need=$g; done
    PAR=${PAR:-$(( MEM_GB / need ))}
    (( PAR >= 1 )) || PAR=1
    log "train arms: $ARMS | seeds: $SEEDS | $PAR at once (${MEM_GB} GB RAM, ~${need} GB each)"
    for s in $SEEDS; do
        for a in $ARMS; do
            while (( $(jobs -rp | wc -l) >= PAR )); do wait -n || true; done
            log "train $a seed $s (look $(look_of "$a"), augment $(augment_of "$a"), resident $(resident_gb "$a") GiB)"
            run_one "$a" "$s" &
            sleep 60
        done
    done
    wait
fi

log "results"
column -t -s $'\t' "$RES" || cat "$RES"
log "done"
