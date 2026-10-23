#!/usr/bin/env bash
# +--------------------------------------------------------------------------+ #
# | SO-101 LeWM plan step 2 on a CUDA box: frozen V-JEPA 2.1 / DINOv2 probes
# +--------------------------------------------------------------------------+ #
# noeira-docs/SO101_LEWM_PLAN.md §3c step 2. Bring to $W (/workspace/step2):
#   real_step2.h5 (so101_step2_export_real.py), twin.demo + .map.txt
#   (lewm_so101_twin.mojo), and this directory's three .py files. Needs
#   build/rerender (box_session_so101_wm_v2.sh build) and the per-episode-DR
#   teacher demo at /workspace/so101_ep/teacher_n10.demo. ~2 h, most of it
#   the 7.4 GB range pull of the encoder (~7 MB/s from dl.fbaipublicfiles).
# ⚠ The report's sklearn fits take every core: a trainer sharing the box
#   slows ~5x while it runs.
W=${W:-/workspace/step2}
set -u
cd /workspace/noeira
export PATH=/workspace/.pixi-bin/bin:$PATH
set +u; eval "$(pixi shell-hook -e default)"; set -u

build/rerender --demos $W/twin.demo --all-episodes --episodes 100000 --joint-zero follower \
  --out $W/twin.rendered.h5 > $W/render_twin.log 2>&1
build/rerender --demos /workspace/so101_ep/teacher_n10.demo --all-episodes --episodes 150 \
  --dr room --dr-seed 11 --dr-per episode --joint-zero follower \
  --out $W/sim150.rendered.h5 > $W/render_sim150.log 2>&1
echo "RENDER-END $(grep -c '=== OK ===' $W/render_twin.log $W/render_sim150.log | tr '\n' ' ')" >> $W/render.done
E=/workspace/enc
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/step2.log; }
mkdir -p $E
if [ ! -d $E/venv ]; then
  # torch cu128: the 5090 (sm_120) needs it; RoboJEPA pins 2.7.1+cu126
  command -v uv > /dev/null || pip install -q uv
  uv venv -q --python 3.11 $E/venv
  . $E/venv/bin/activate
  uv pip install -q --index-url https://download.pytorch.org/whl/cu128 torch torchvision
fi
. $E/venv/bin/activate
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" >> $W/step2.log 2>&1 \
  || { say "TORCH-FAILED"; exit 1; }
uv pip install -q timm einops numpy h5py scikit-learn pillow requests >> $W/step2.log 2>&1 || { say "PIP-FAILED"; exit 1; }
cd $E
[ -d vjepa2 ] || git clone -q --depth 1 https://github.com/facebookresearch/vjepa2.git
[ -d dinov2 ] || git clone -q --depth 1 https://github.com/facebookresearch/dinov2.git
say "vjepa2 licence: $(head -2 vjepa2/LICENSE | tr '\n' ' ')"
if [ ! -f $E/vjepa21_target_encoder_bf16.pt ]; then
  say "extract V-JEPA 2.1 target_encoder"
  python $W/extract_torch_subdict_http.py https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt \
    target_encoder $E/vjepa21_target_encoder_bf16.pt --bf16 >> $W/extract.log 2>&1 || { say "EXTRACT-FAILED"; exit 1; }
  say "  $(tail -1 $W/extract.log)"
fi

say "render: $(cat $W/render.done)"
cd $W
for enc in vjepa21 dinov2; do
  say "features $enc"
  python $W/so101_frozen_encoder_step2.py features --enc $enc --dir $W --hub $E/$([ $enc = vjepa21 ] && echo vjepa2 || echo dinov2) \
    --weights $E/vjepa21_target_encoder_bf16.pt > $W/feat_$enc.log 2>&1 || { say "FEATURES-FAILED $enc"; tail -5 $W/feat_$enc.log >> $W/step2.log; }
done
say "report"
python $W/so101_frozen_encoder_step2.py report --dir $W --enc vjepa21,dinov2 > $W/report.txt 2>&1
say "STEP2-END"
