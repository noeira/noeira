#!/usr/bin/env bash
# LeWM box session A (docs/LEWM_REOPEN_PLAN.md §5) — on a rented CUDA box.
#
#   bash tools/lewm/box_session_a.sh            # from the noeira repo root
#
# 1. a throwaway Python 3.10 venv with stable-worldmodel 0.1.1 (PINNED: its
#    `rollout` is le-wm-main's; GitHub main has since changed it);
# 2. the HF weights converted to the `_object.ckpt` eval.py loads (README),
#    and an `import eval` smoke — a version mismatch fails BEFORE step 3;
# 3. the dataset, streamed through zstd (~13 GB in, 47 GB on disk);
# 4. column R: the reference eval, 50 episodes, seed 42, budget 50;
# 5. tools/lewm/session_a_extract.py: the same 50 pairs (cross-checked), the
#    stats and the fixture -> one tarball to copy home.
#
# Needs: nvidia driver + CUDA, curl, zstd, git. Writes under $WORK only.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/workspace/lewm_session_a}"
export STABLEWM_HOME="${STABLEWM_HOME:-$WORK/stablewm}"
OUT="$WORK/out"
LEWM_REF="$WORK/le-wm-main"
mkdir -p "$WORK" "$STABLEWM_HOME" "$OUT"
log() { echo "[session-a $(date +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }

# ---- 1. the reference environment ----------------------------------------
if [[ ! -x "$WORK/venv/bin/python" ]]; then
    log "venv: python 3.10 + stable-worldmodel[train,env]==0.1.1"
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python=3.10 "$WORK/venv"
    VIRTUAL_ENV="$WORK/venv" uv pip install "stable-worldmodel[train,env]==0.1.1" h5py
fi
PY="$WORK/venv/bin/python"
"$PY" - <<'PY' | tee -a "$OUT/session.log"
import importlib.metadata as m
for p in ("stable-worldmodel", "stable-pretraining", "torch", "transformers", "pymunk"):
    try: print(f"  {p} {m.version(p)}")
    except m.PackageNotFoundError: print(f"  {p} MISSING")
import torch; print("  cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
# the reference repo: copied from references/ if present, else cloned
if [[ ! -d "$LEWM_REF" ]]; then
    if [[ -d "$REPO/references/le-wm-main" ]]; then cp -r "$REPO/references/le-wm-main" "$LEWM_REF"
    else git clone --depth 1 https://github.com/lucas-maes/le-wm "$LEWM_REF"; fi
fi

# ---- 2. weights -> _object.ckpt (le-wm-main README, verbatim) -------------
CKPT="$STABLEWM_HOME/pusht/lewm_object.ckpt"
if [[ ! -s "$CKPT" ]]; then
    log "weights: HF quentinll/lewm-pusht -> $CKPT"
    mkdir -p "$STABLEWM_HOME/hf_pusht"
    for f in weights.pt config.json; do
        curl -fsL -o "$STABLEWM_HOME/hf_pusht/$f" "https://huggingface.co/quentinll/lewm-pusht/resolve/main/$f"
    done
    (cd "$LEWM_REF" && "$PY" - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")
cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"], patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"], pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
pred = {k: v for k, v in cfg["predictor"].items() if k != "_target_"}
act = {k: v for k, v in cfg["action_encoder"].items() if k != "_target_"}
model = JEPA(encoder=encoder, predictor=ARPredictor(**pred),
             action_encoder=Embedder(**act), projector=mlp("projector"),
             pred_proj=mlp("pred_proj"))
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
print("  converted ->", out)
PY
    ) | tee -a "$OUT/session.log"
fi

# fail BEFORE the 13 GB download if eval.py does not import against 0.1.1
log "smoke: import le-wm-main eval.py under stable-worldmodel 0.1.1"
(cd "$LEWM_REF" && "$PY" -c "import eval; import stable_worldmodel as swm; print('  eval.py imports; swm', swm.__file__)") \
    2>&1 | tee -a "$OUT/session.log"

# ---- 3. the dataset --------------------------------------------------------
H5="$STABLEWM_HOME/pusht_expert_train.h5"
if [[ ! -s "$H5" ]]; then
    log "dataset: streaming pusht_expert_train.h5.zst through zstd"
    curl -fL "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst" \
        | zstd -d -o "$H5.part"
    mv "$H5.part" "$H5"
fi
log "dataset: $(du -h "$H5" | cut -f1)"

# ---- 4. column R ------------------------------------------------------------
log "R: eval.py policy=pusht/lewm (50 episodes, seed 42, budget 50)"
# `++cache_dir`: eval.py reads cfg.cache_dir, which pusht.yaml does not
# define (Hydra struct mode raises on a missing key); `++` adds-or-overrides.
(cd "$LEWM_REF" && "$PY" eval.py --config-name=pusht.yaml policy=pusht/lewm \
    "++cache_dir=$STABLEWM_HOME") 2>&1 \
    | tee "$OUT/eval_R.log"
cp "$STABLEWM_HOME/pusht/pusht_results.txt" "$OUT/" 2>/dev/null || true

# ---- 5. pairs, stats, fixture --------------------------------------------
log "extract: pairs + stats + fixture"
"$PY" "$REPO/tools/lewm/session_a_extract.py" --h5 "$H5" --out "$OUT/fixture" \
    --eval-log "$OUT/eval_R.log" 2>&1 | tee -a "$OUT/session.log"

tar -C "$WORK" -czf "$WORK/lewm_session_a.tgz" out
log "DONE: $(du -h "$WORK/lewm_session_a.tgz" | cut -f1) -> scp it home, then unpack under ~/.cache/noeira/lewm_pusht/"
