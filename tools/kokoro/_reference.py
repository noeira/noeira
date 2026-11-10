"""Load hexgrad's own Kokoro model code for the reference tools.

The `kokoro` package's `__init__` imports its text front end (misaki, spaCy,
espeak), which the MODEL does not need. The tools here drive `KModel`
(phonemes -> audio) alone, so the package is entered through a stub that
exposes its modules without running that `__init__`.

Setup (once; the `detect` pixi environment supplies torch and transformers):

    uv pip install --python .pixi/envs/detect/bin/python --no-deps \
        --target ~/.cache/noeira/kokoro/ref-site kokoro misaki loguru addict attrs
    pixi run -e detect python -c "from huggingface_hub import snapshot_download as d; \
        d('hexgrad/Kokoro-82M', allow_patterns=['config.json', 'kokoro-v1_0.pth', 'voices/*.pt'])"
"""

import glob
import os
import sys
import types

REPO = "hexgrad/Kokoro-82M"
SITE = os.path.expanduser("~/.cache/noeira/kokoro/ref-site")


def snapshot_dir():
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    found = glob.glob(os.path.join(hub, "models--hexgrad--Kokoro-82M", "snapshots", "*", ""))
    if not found:
        raise SystemExit("no hexgrad/Kokoro-82M in the Hub cache — see tools/kokoro/_reference.py")
    return found[0]


def load_kmodel():
    """`(KModel in eval mode, snapshot dir)`, fp32, CPU."""
    if not os.path.isdir(os.path.join(SITE, "kokoro")):
        raise SystemExit(f"no kokoro package in {SITE} — see tools/kokoro/_reference.py")
    sys.path.append(SITE)
    pkg = types.ModuleType("kokoro")
    pkg.__path__ = [os.path.join(SITE, "kokoro")]
    sys.modules["kokoro"] = pkg
    from loguru import logger

    logger.remove()
    from kokoro.model import KModel

    snap = snapshot_dir()
    model = KModel(repo_id=REPO, config=snap + "config.json", model=snap + "kokoro-v1_0.pth").eval()
    return model, snap
