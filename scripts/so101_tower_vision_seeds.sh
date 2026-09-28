#!/usr/bin/env bash
# so101_tower vision student: a SEED MATRIX, so data recipes are compared as
# distributions and not as single runs.
#
#   bash scripts/so101_tower_vision_seeds.sh                 # arms A B D, seeds 1 2 3
#   ARMS="A B C D" SEEDS="1 2 3 4" bash scripts/so101_tower_vision_seeds.sh
#   STAGES="build" bash scripts/so101_tower_vision_seeds.sh  # compile only
#
# WHY. Four ACT students on overlapping data (22 Sep, old headlight) scored
# 32 / 16 / 56 / 41% closed loop; the 56% one trained on a SUBSET of the 32%
# one's data with the same recipe. The run-to-run spread is >= 20 points, as
# large as every effect being measured, so each arm needs several seeds.
#
# THE ARMS (every store rendered under the scene's CURRENT look — the
# headlight of d1b7adf7d; the script refuses a scene without it):
#   A  200 expert episodes (120 clean + 80 noisy)          — the baseline
#   B  A + the 88 DAgger relabels (first row < 36 mm)     — DAgger round 1
#   C  A + the 31 DAgger relabels at SETTLED arrivals      — optional, small
#   D  400 expert episodes (240 clean + 160 noisy)         — more expert data
# Each arm x seed: ACT on the HOST data path (image column resident in RAM,
# ~10 GB of GPU, so three runs share the 5090), then the closed-loop eval on
# the same 128 held-out placements (seeds 30000+). One seed of every arm runs
# at a time, so the arms of a batch see the same GPU contention.
#
# NEEDS on the box, from the laptop (small files, seconds):
#     projects/so101-tower/demos/expert_cube_in_bowl_{clean,flat02}_300.demo
#     projects/so101-tower/demos/dagger_vision_r1{,_settled}.demo
# Arm D's second 200 expert episodes are generated here on the CPU.
#
# OUTPUT: build/so101_vision/seeds/results.tsv (arm, seed, run dir, best val
# L1, success / 128, no grasp, dropped) and a per-arm mean +- spread at the
# end. Every stage skips work whose output exists: rerun after a drop.
# Run it inside tmux, from the repo root.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PATH=/workspace/.pixi-bin/bin:$PATH:$HOME/.pixi/bin

# PAR: ACT runs at once. Each holds its store's image column in RAM (A 21 GB,
# B 28, D 42), so PAR=3 needs ~100 GB; the default is chosen from the box's
# RAM. With PAR=1 each run's eval overlaps the next run's training.
# BUILD_PAR: builds at once (a Mojo build of the ACT graph takes GBs).
MEM_GB=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 64)
PAR="${PAR:-$([[ $MEM_GB -ge 160 ]] && echo 3 || echo 1)}"
BUILD_PAR="${BUILD_PAR:-$([[ $MEM_GB -ge 160 ]] && echo 4 || echo 1)}"
ARMS="${ARMS:-A B D}"
SEEDS="${SEEDS:-1 2 3}"
STAGES="${STAGES:-build demos render train}"
D=projects/so101-tower/demos
B=build/so101_vision/seeds
BIN=build/so101_vision/bin
mkdir -p "$B" "$BIN"
RES=$B/results.tsv
[[ -s $RES ]] || printf 'arm\tseed\trun\tbest_val_l1\tsuccess\tno_grasp\tdropped\n' > "$RES"
log() { printf '\n=== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

grep -q '<headlight' noeira/tasks/scenes/so101_tower.xml || {
    echo "the tower scene has no <headlight>: pull d1b7adf7d or later first"; exit 1; }
[[ -d projects/so101-tower && -f projects/so101-tower/project.kv ]] && export ACT_PROJECT=so101-tower

# ── build: four binaries, in parallel (~25 min on the box's CPU) ──────────
build() {  # src out [defines...]
    local src=$1 out=$2; shift 2
    if [[ ! -x $out || $src -nt $out ]]; then
        pixi run -e nvidia mojo build "$@" -I . -o "$out" "$src" > "$out.build.log" 2>&1 \
            || { echo "BUILD FAILED: $src"; grep -A6 'error:' "$out.build.log" | head -30; return 1; }
    fi
}
if has build; then
    log "build"
    pixi install -e nvidia > /dev/null
    if [[ $BUILD_PAR -ge 4 ]]; then
        build examples/so101/tower_expert_record.mojo $BIN/tower_expert &
        P1=$!
        build examples/so101/tower_demo_rerender.mojo $BIN/tower_rerender &
        P2=$!
        build examples/so101/tower_act_eval.mojo $BIN/tower_act_eval &
        P3=$!
        build examples/so101/act_so101_train_gpu.mojo $BIN/act_train_host -D ACT_HOST_DATA &
        P4=$!
        wait $P1 && wait $P2 && wait $P3 && wait $P4
    else
        build examples/so101/tower_expert_record.mojo $BIN/tower_expert
        build examples/so101/tower_demo_rerender.mojo $BIN/tower_rerender
        build examples/so101/act_so101_train_gpu.mojo $BIN/act_train_host -D ACT_HOST_DATA
        build examples/so101/tower_act_eval.mojo $BIN/tower_act_eval
    fi
    log "build ok (PAR=$PAR BUILD_PAR=$BUILD_PAR, ${MEM_GB} GB RAM)"
fi

C1=$D/expert_cube_in_bowl_clean_300.demo
N1=$D/expert_cube_in_bowl_flat02_300.demo
C2=$D/expert_cube_in_bowl_clean_300b.demo
N2=$D/expert_cube_in_bowl_flat02_300b.demo
G1=$D/dagger_vision_r1.demo
G2=$D/dagger_vision_r1_settled.demo

if has demos && [[ " $ARMS " == *" D "* ]]; then
    for f in $C1 $N1; do [[ -s $f ]] || { echo "missing $f (copy it from the laptop)"; exit 1; }; done
    log "demos for arm D (CPU)"
    [[ -s $C2 ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 300 --seed 17000 \
        --quiet --out $C2 > $B/gen_clean_b.log 2>&1 &
    [[ -s $N2 ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 300 --seed 20000 \
        --noise 0.02 --flat-noise --quiet --out $N2 > $B/gen_noisy_b.log 2>&1 &
    wait
fi

store_of() { echo "$D/vision_seeds_$1.rendered.h5"; }
render_arm() {  # arm
    local out; out=$(store_of "$1")
    [[ -s $out ]] && return 0
    local demos caps
    case $1 in
        A) demos=$C1,$N1;           caps=120,80 ;;
        B) demos=$C1,$N1,$G1;       caps=120,80,100000 ;;
        C) demos=$C1,$N1,$G2;       caps=120,80,100000 ;;
        D) demos=$C1,$N1,$C2,$N2;   caps=120,80,120,80 ;;
    esac
    $BIN/tower_rerender --demos "$demos" --per-file "$caps" --out "$out.partial" 2>&1 \
        | grep -v '^Warning: attached\|intercept\]' | grep -v '^  *[0-9]* episodes |' \
        > "$B/render_$1.log"
    tail -6 "$B/render_$1.log"
    mv "$out.partial" "$out"
}
if has render; then
    log "render arms: $ARMS"
    for a in $ARMS; do render_arm "$a" & done
    wait
    for a in $ARMS; do [[ -s $(store_of "$a") ]] || { echo "render of arm $a failed"; exit 1; }; done
fi

train_one() {  # arm seed — skips a run whose training log says it finished
    local a=$1 s=$2 tag="$1_s$2"
    grep -q "^$a	$s	" "$RES" && return 0
    grep -q 'best validation l1' "$B/act_$tag.log" 2>/dev/null && return 0
    ACT_STORE=$(store_of "$a") ACT_SEED=$s ACT_NO_MONITOR=1 ACT_RESIDENT_GB=60 \
        $BIN/act_train_host > "$B/act_$tag.log" 2>&1 || { echo "train $tag FAILED"; tail -20 "$B/act_$tag.log"; return 1; }
    log "trained $tag"
}
eval_one() {  # arm seed — deferred (not failed) when the eval binary is missing
    local a=$1 s=$2 tag="$1_s$2"
    grep -q "^$a	$s	" "$RES" && return 0
    if [[ ! -x $BIN/tower_act_eval ]]; then
        log "no eval binary: $tag's eval deferred (STAGES=eval later)"; return 0
    fi
    local run; run=$(tr '\r' '\n' < "$B/act_$tag.log" | grep -m1 '^  run ' | awk '{print $2}')
    local val; val=$(tr '\r' '\n' < "$B/act_$tag.log" | grep -m1 'best validation l1' | awk '{print $4}')
    $BIN/tower_act_eval --ckpt "$run/checkpoints" --episodes 128 > "$B/eval_$tag.log" 2>&1 \
        || { echo "eval $tag FAILED"; tail -20 "$B/eval_$tag.log"; return 1; }
    local ok ng dr
    ok=$(grep -m1 'SUCCESS' "$B/eval_$tag.log" | awk '{print $2}')
    ng=$(grep -m1 'no grasp' "$B/eval_$tag.log" | awk '{print $3}')
    dr=$(grep -m1 'dropped' "$B/eval_$tag.log" | awk '{print $2}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$a" "$s" "$run" "$val" "$ok" "$ng" "$dr" >> "$RES"
    log "$tag: $ok/128 (val $val) $run"
}
one_run() { train_one "$1" "$2" && eval_one "$1" "$2"; }
if has train; then
    if [[ $PAR -ge 3 ]]; then
        for s in $SEEDS; do
            log "seed $s: arms $ARMS in parallel"
            for a in $ARMS; do one_run "$a" "$s" & done
            wait
        done
    else
        # one training at a time; its eval runs beside the NEXT training
        for s in $SEEDS; do
            for a in $ARMS; do
                log "train $a seed $s"
                train_one "$a" "$s" || continue
                eval_one "$a" "$s" &
            done
        done
        wait
    fi
fi

# STAGES="eval": evaluate every trained run that has no result yet (after the
# eval binary exists — it needs > 62 GB of RAM to compile at 6208ae25e)
if has eval; then
    for s in $SEEDS; do for a in $ARMS; do
        [[ -f $B/act_${a}_s$s.log ]] && eval_one "$a" "$s"
    done; done
fi

log "results"
column -t -s $'\t' "$RES" || cat "$RES"
python3 - "$RES" <<'EOF'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
arms = sorted({r['arm'] for r in rows})
print("\narm  n  mean%  min%  max%  sd%")
for a in arms:
    v = [100 * int(r['success']) / 128 for r in rows if r['arm'] == a and r['success']]
    if v:
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f"{a:3}  {len(v)}  {st.mean(v):5.1f}  {min(v):4.1f}  {max(v):4.1f}  {sd:4.1f}")
EOF
log "done"
