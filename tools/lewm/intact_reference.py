#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | INTACT (history-free PushT) -> our dump, plus Direct-plan references
# +--------------------------------------------------------------------------+ #
"""Convert INTACT's history-free PushT checkpoint to THIS framework's names
and dump the Direct plans the Mojo port must reproduce (LEWM_REOPEN_PLAN P8).

    pixi run mojo run -I . tools/lewm/list_ref_params.mojo > <out>/ours_names.tsv
    pixi run -e act-ref python tools/lewm/intact_reference.py --out <out> \\
        --ckpt ~/.cache/noeira/lewm_pusht/intact/checkpoints/\\
intact_goal_history_free_pusht_s0_p77_v1/weights_epoch_1.pt

⚠ act-ref pixi environment only (torch); nothing under `noeira/` imports it.

## The checkpoint

`INTACT-JEPA/INTACT` on Hugging Face, folder `INTACT-no-previous-action`
(manifest: `feature_layout four_slot`, `actor_input [z_t, m_t, z_t * m_t]`,
576 features, `uses_previous_action false`). Every LeWM module is LeWM's
(`references/INTACT-JEPA-main/module.py` diffed against le-wm): the
encoder is the same `vit_hf` tiny/14 saved under transformers 5.x names, the
predictor's `adaLN_modulation` is renamed `modulation`. So the model is
rebuilt with `dump_lewm_reference.build_model` (the published LeWM's
`config.json` — identical widths) and loaded STRICT; the actor is rebuilt
from `module.IntentActionActor` with no action-history slot.

## Written into --out

- `param.<name>` every LeWM tensor under the published checkpoint's names,
  then `ours.<name>` through `convert_ref_to_ours`'s rules (the same MAP the
  LeWM gates use) and `ours.actor.<i>.*` for the actor (our Sequential
  `Linear LN GELU x3, Linear`: torch `net.{0,1,4,5,8,9,11}` -> ours
  `{0,1,3,4,6,7,9}`).
- `intact.*` Direct references on the first `--pairs` fixture pairs, float64:
  `z0` / `zg` (encoded start and goal frames), `mean0` / `log_std0` (the
  actor on (z0, zg − z0)), and the 5-block plans in both context modes:

  * `plan_intact` — `jepa.get_action` verbatim: the action history starts as
    [a_{-1}] beside [z0] and grows by appending; each step REPLACES the last
    action of the context with the new one. From step 1 on, the older context
    positions pair z_j with a_{j-1} — one block off the training pairing
    (`train.py: predict_adjacent_latents` pairs frame t with the action taken
    FROM t). Their published numbers come from this.
  * `plan_aligned` — the training pairing: z_j with a_j at every position.

  a_{-1} is the episode-start convention (raw zero, normalised): the fixture
  has no dataset history; the history-free actor never reads it, only the
  `intact` rollout context does.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.act.dump_act_reference import Dump  # noqa: E402
from tools.lewm import dump_lewm_reference as R  # noqa: E402
from tools.lewm.convert_ref_to_ours import _rules, NO_REF, read_dump  # noqa: E402

HORIZON = 5
CTX = 3
ACT = 10
# torch Sequential index -> ours (no Dropout children in ours)
ACTOR_IDX = {0: 0, 1: 1, 4: 3, 5: 4, 8: 6, 9: 7, 11: 9}
# INTACT saved its ViT under transformers 5.x names; the converter's rules and
# the published LeWM dump use 4.x (`dump_lewm_reference._VIT_4_TO_5`, inverted)
_VIT_5_TO_4 = [
    (r"^encoder\.layers\.(\d+)\.attention\.q_proj\.", r"encoder.encoder.layer.\1.attention.attention.query."),
    (r"^encoder\.layers\.(\d+)\.attention\.k_proj\.", r"encoder.encoder.layer.\1.attention.attention.key."),
    (r"^encoder\.layers\.(\d+)\.attention\.v_proj\.", r"encoder.encoder.layer.\1.attention.attention.value."),
    (r"^encoder\.layers\.(\d+)\.attention\.o_proj\.", r"encoder.encoder.layer.\1.attention.output.dense."),
    (r"^encoder\.layers\.(\d+)\.mlp\.fc1\.", r"encoder.encoder.layer.\1.intermediate.dense."),
    (r"^encoder\.layers\.(\d+)\.mlp\.fc2\.", r"encoder.encoder.layer.\1.output.dense."),
    (r"^encoder\.layers\.(\d+)\.layernorm_", r"encoder.encoder.layer.\1.layernorm_"),
]


class Actor(torch.nn.Module):
    """`module.IntentActionActor` with no action-history slot (576 inputs)."""

    def __init__(self, emb=192, hidden=1024, depth=3, act=ACT):
        super().__init__()
        layers = [torch.nn.Linear(3 * emb, hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU()]
        for _ in range(depth - 1):
            layers += [torch.nn.Dropout(0.0), torch.nn.Linear(hidden, hidden),
                       torch.nn.LayerNorm(hidden), torch.nn.GELU()]
        layers.append(torch.nn.Linear(hidden, 2 * act))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z, m):
        p = self.net(torch.cat([z, m, z * m], dim=-1))
        mean, log_std = p.chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)


def load(ckpt: Path, dtype):
    config = R.json.loads((Path.home() / ".cache/noeira/lewm_pusht/hf/config.json").read_text())
    model = R.build_model(config)
    want = set(model.state_dict())
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    mapped, actor_sd, ckpt4 = {}, {}, {}
    for k, v in sd.items():
        if k.startswith("intent_actor."):
            actor_sd[k[len("intent_actor."):]] = v
            continue
        mk = k.replace(".modulation.", ".adaLN_modulation.")
        k4 = mk  # the published checkpoint's (transformers 4.x) name
        for pat, rep in _VIT_5_TO_4:
            k4 = re.sub(pat, rep, k4)
        model_key = mk if mk in want else k4
        if model_key not in want:
            sys.exit(f"{k}: no model parameter {mk} / {k4}")
        mapped[model_key] = v
        ckpt4[model_key] = k4
    model.load_state_dict(mapped, strict=True)
    actor = Actor()
    actor.load_state_dict(actor_sd, strict=True)
    return model.to(dtype).eval(), actor.to(dtype).eval(), ckpt4


def convert(out: Path, model, actor, ckpt4):
    dump = Dump(out)
    for k, v in model.state_dict().items():
        if v.dtype == torch.int64:
            continue
        dump.add(f"param.{ckpt4.get(k, k)}", v)
    dump.close()
    ref = read_dump(out)
    names = [line.split("\t") for line in (out / "ours_names.tsv").read_text().splitlines()]
    rules = _rules()
    dump = Dump(out)
    for kind, name, size in names:
        hit = [(m, f) for rx, f in rules if (m := rx.match(name))]
        if len(hit) != 1:
            sys.exit(f"{name}: {len(hit)} mapping rules match")
        m, f = hit[0]
        keys, tf = f(m)
        if keys == [NO_REF]:
            dump.add(f"ours.{name}", np.zeros(int(size), dtype=np.float32))
            continue
        v = np.ascontiguousarray(tf([ref[f"param.{k}"] for k in keys])).reshape(-1)
        if v.size != int(size):
            sys.exit(f"{name}: {v.size} values, ours holds {size}")
        dump.add(f"ours.{name}", v)
    n = 0
    for k, v in actor.state_dict().items():
        i, leaf = k.split(".")[1], k.split(".")[2]
        ours_i = ACTOR_IDX[int(i)]
        if isinstance(actor.net[int(i)], torch.nn.Linear):
            arr = v.T if leaf == "weight" else v  # ours [in, out]
            dump.add(f"ours.actor.{ours_i}.{leaf}", arr.contiguous())
        else:
            dump.add(f"ours.actor.{ours_i}.{'gamma' if leaf == 'weight' else 'beta'}", v)
        n += 1
    dump.close()
    print(f"  converted: {len(names)} LeWM tensors, {n} actor tensors")


def frame(u8_hwc: np.ndarray, dtype) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(u8_hwc)).to(dtype) / 255.0
    x = x.movedim(-1, -3)
    mean = torch.tensor(R.IMAGENET_MEAN, dtype=dtype).view(3, 1, 1)
    std = torch.tensor(R.IMAGENET_STD, dtype=dtype).view(3, 1, 1)
    return (x - mean) / std


def direct(model, actor, z0, zg, a_hist, aligned: bool):
    """`jepa.get_action` (history-free actor) for one pair; returns (H, ACT)."""
    embs = [z0]
    acts = [] if aligned else [a_hist]
    plan = []
    for k in range(HORIZON):
        cur = embs[-1]
        a, _ = actor(cur[None], (zg - cur)[None])
        a = a[0]
        plan.append(a)
        if aligned:
            ctx_a = (acts + [a])[-min(CTX, len(embs)):]
        else:
            size = min(CTX, len(embs), len(acts))
            ctx_a = acts[-size:]
            ctx_a = ctx_a[:-1] + [a]
        ctx_e = embs[-len(ctx_a):]
        pred = model.predict(torch.stack(ctx_e)[None],
                             model.action_encoder(torch.stack(ctx_a)[None]))[0, -1]
        embs.append(pred)
        acts.append(a)
    return torch.stack(plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixture", default=str(Path.home() / ".cache/noeira/lewm_pusht/session_a/out/fixture"))
    ap.add_argument("--pairs", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    model, actor, ckpt4 = load(Path(args.ckpt), torch.float64)
    convert(out, model, actor, ckpt4)

    fx = read_dump(Path(args.fixture))
    sp = fx["pairs.start_pixels"].reshape(-1, R.IMG, R.IMG, 3)
    gp = fx["pairs.goal_pixels"].reshape(-1, R.IMG, R.IMG, 3)
    mean = np.tile(fx["stats.action_mean"].astype(np.float64), 5)
    std = np.tile(fx["stats.action_std_train"].astype(np.float64), 5)
    a_hist = torch.from_numpy((0.0 - mean) / std)
    dump = Dump(out)
    rows = {k: [] for k in ("z0", "zg", "mean0", "log_std0", "plan_intact", "plan_aligned")}
    with torch.no_grad():
        for e in range(args.pairs):
            pix = torch.stack([frame(sp[e], torch.float64), frame(gp[e], torch.float64)])
            z = model.encode({"pixels": pix[:, None]})["emb"][:, 0]
            z0, zg = z[0], z[1]
            m0, s0 = actor(z0[None], (zg - z0)[None])
            rows["z0"].append(z0)
            rows["zg"].append(zg)
            rows["mean0"].append(m0[0])
            rows["log_std0"].append(s0[0])
            rows["plan_intact"].append(direct(model, actor, z0, zg, a_hist, aligned=False))
            rows["plan_aligned"].append(direct(model, actor, z0, zg, a_hist, aligned=True))
    for k, v in rows.items():
        dump.add(f"intact.{k}", torch.stack(v))
    dump.add("intact.a_hist", a_hist)
    dump.close()
    d = (torch.stack(rows["plan_intact"]) - torch.stack(rows["plan_aligned"])).abs()
    print(f"  Direct on {args.pairs} pairs: first actions equal in both modes "
          f"{bool(d[:, 0].max() == 0)}; later steps differ by up to {float(d[:, 1:].max()):.4f}")


if __name__ == "__main__":
    main()
