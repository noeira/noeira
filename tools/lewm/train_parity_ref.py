#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | LeWM short-run training parity — the torch run (P6.1, box session B)
# +--------------------------------------------------------------------------+ #
"""Train the reference LeWM from its own random init for N steps on the real
PushT dataset, and record everything a Mojo replay needs to take the SAME
steps (docs/LEWM_REOPEN_PLAN.md P6):

    python tools/lewm/train_parity_ref.py --h5 .../pusht_expert_train.h5 \\
        --out /workspace/lewm_parity32 --steps 1000 --batch 32

then `convert_ref_to_ours.py --dump <out>` (the init in our names) and
`examples/lewm/lewm_pusht_train_parity.mojo --run <out>`.

What is shared, so the two runs differ only by arithmetic:

* the init: the reference's (`build_model` of `dump_lewm_reference.py` —
  HF ViT init, torch defaults, AdaLN zero) under `--seed`, dumped as
  `param.<checkpoint name>` (BN running stats included);
* the batches: `run.clip_idx` (steps x B) — indices into the dataset's clip
  list, the order `LewmPushTExpert` and stable-worldmodel both build
  (episode-major, every start with a full 20-frame span);
* the action normaliser: `train.py`'s `get_column_normalizer` (float32 torch
  mean / unbiased std over the non-NaN rows), `run.action_mean|std`;
* every step's SIGReg matrix, column-normalised: `run.A.<k>`.

What differs from the recipe, on purpose: dropout off (Mojo's graph has
none), constant lr (no warmup / cosine), float32 (no bf16 autocast). The
optimizer is the recipe's: AdamW lr 5e-5, wd 1e-3 on every parameter, clip
1.0.

Recorded per step: `run.scalars` (steps x 5): loss, pred loss, SIGReg,
pre-clip grad norm, wall seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401  (the dataset's pixels may be Blosc-filtered)
except ImportError:
    pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.act.dump_act_reference import Dump  # noqa: E402
from tools.lewm.dump_lewm_reference import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, _FixedRandn, build_model, ckpt_name,
    dropout_off, lejepa_forward, load_model, swm_loss,
)

FRAMESKIP = 5
NUM_STEPS = 4  # history 3 + 1 prediction
SPAN = FRAMESKIP * NUM_STEPS


def clip_list(ep_len: np.ndarray):
    """(episode, start) for every window with a full span — episode-major,
    the order of `LewmPushTExpert.clip_*` and swm's `clip_indices`."""
    eps, starts = [], []
    for ep, n in enumerate(ep_len):
        if n >= SPAN:
            k = int(n) - SPAN + 1
            eps.append(np.full(k, ep, dtype=np.int64))
            starts.append(np.arange(k, dtype=np.int64))
    return np.concatenate(eps), np.concatenate(starts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(Path.home() / ".cache/noeira/lewm_pusht/hf/weights.pt"),
                    help="only for the checkpoint's parameter NAMES (the init is random)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tf32", action="store_true",
                    help="TF32 matmuls and convolutions: torch's run in OUR arithmetic class, the "
                         "third column that says how far arithmetic alone moves the curve")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    dev = torch.device(args.device)
    out = Path(args.out)
    dump = Dump(out)

    # names: load once so `ckpt_name` maps model keys to the checkpoint's
    config = json.loads((Path(args.weights).parent / "config.json").read_text())
    load_model(Path(args.weights), torch.float32)
    torch.manual_seed(args.seed)
    model = build_model(config).float()
    for k, v in model.state_dict().items():
        if v.dtype != torch.int64:
            dump.add(f"param.{ckpt_name(k)}", v)
    model = model.to(dev)
    model.train()
    dropout_off(model)
    sigreg = swm_loss.SIGReg(knots=17, num_proj=1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    f = h5py.File(args.h5, "r")
    ep_len = f["ep_len"][:]
    ep_off = f["ep_offset"][:]
    clip_ep, clip_start = clip_list(ep_len)
    act_all = torch.from_numpy(f["action"][:])
    ok = ~torch.isnan(act_all).any(dim=1)
    a_mean = act_all[ok].mean(0, keepdim=True)
    a_std = act_all[ok].std(0, keepdim=True)
    dump.add("run.action_mean", a_mean.reshape(-1))
    dump.add("run.action_std", a_std.reshape(-1))
    print(f"dataset: {len(ep_len)} episodes, {len(clip_ep)} clips; "
          f"action mean {a_mean.tolist()} std {a_std.tolist()}")

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(clip_ep), size=(args.steps, args.batch))
    dump.add("run.clip_idx", idx.astype(np.float64))
    dump.add("run.hparams", np.array([args.lr, args.wd, args.clip, args.steps, args.batch]))
    gen = torch.Generator().manual_seed(args.seed + 1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    pix_ds = f["pixels"]

    rows = []
    t_all = time.time()
    for s in range(args.steps):
        t0 = time.time()
        pix = np.empty((args.batch, NUM_STEPS, 224, 224, 3), dtype=np.uint8)
        act = np.empty((args.batch, SPAN, 2), dtype=np.float32)
        for b, ci in enumerate(idx[s]):
            g = int(ep_off[clip_ep[ci]] + clip_start[ci])
            pix[b] = pix_ds[g:g + SPAN][::FRAMESKIP]
            act[b] = act_all[g:g + SPAN].numpy()
        x = torch.from_numpy(pix).float().div_(255.0).permute(0, 1, 4, 2, 3)
        x = ((x - mean) / std).to(dev)
        a = ((torch.from_numpy(act) - a_mean) / a_std).float()
        a = a.reshape(args.batch, NUM_STEPS, FRAMESKIP * 2).to(dev)
        a_raw = torch.randn(192, 1024, generator=gen)
        dump.add(f"run.A.{s}", a_raw / a_raw.norm(p=2, dim=0))

        opt.zero_grad(set_to_none=True)
        with _FixedRandn(a_raw.to(dev)):
            o = lejepa_forward(model, sigreg, {"pixels": x, "action": a})
        o["loss"].backward()
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        torch.cuda.synchronize() if dev.type == "cuda" else None
        r = [o["loss"].item(), o["pred_loss"].item(), o["sigreg_loss"].item(),
             total.item(), time.time() - t0]
        rows.append(r)
        if s % 25 == 0 or s == args.steps - 1:
            print(f"step {s:5d}  loss {r[0]:.5f}  pred {r[1]:.5f}  sigreg {r[2]:.4f}  "
                  f"norm {r[3]:.3f}  {r[4]:.2f} s", flush=True)
    dump.add("run.scalars", np.array(rows))
    dump.close()
    print(f"done: {args.steps} steps in {time.time() - t_all:.0f} s -> {out}")


if __name__ == "__main__":
    main()
