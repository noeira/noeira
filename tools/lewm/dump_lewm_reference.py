#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | LeWorldModel reference dumps — the numbers the Mojo port is gated against
# +--------------------------------------------------------------------------+ #
"""Run the PUBLISHED LeWM PushT model under PyTorch and dump every quantity
the Mojo port must reproduce (docs/LEWM_REOPEN_PLAN.md, P1/P2).

    pixi run -e act-ref python tools/lewm/dump_lewm_reference.py --out /tmp/lewm_ref

⚠ Runs in the `act-ref` pixi environment ONLY. Nothing under `noeira/` imports
torch; this exists so the port is checked against the reference rather than
against itself.

## What "the reference" is

- Weights: `quentinll/lewm-pusht` `weights.pt` (MIT), expected at
  `~/.cache/noeira/lewm_pusht/hf/weights.pt` (`--weights` to override).
- Model code: `stable_worldmodel` **0.1.1** — the PyPI wheel, vendored at
  `references/stable-worldmodel-0.1.1/` — because the checkpoint's
  `config.json` names `stable_worldmodel.wm.lewm.LeWM`. 0.1.1's `rollout`
  keeps `references/le-wm-main`'s semantics; GitHub main has since changed
  it. `lewm.py`, `module.py` and `wm/loss.py` are loaded BY FILE PATH: the
  package `__init__` imports every environment (gymnasium, mujoco, ...).
- Encoder: `stable_pretraining.backbone.utils.vit_hf("tiny", patch_size=14,
  image_size=224, pretrained=False, use_mask_token=False)`, rebuilt here from
  its source (stable-pretraining ab836bf699a2): a HuggingFace `ViTModel` with
  `ViTConfig` defaults (exact GELU, LayerNorm eps 1e-12, qkv bias, no
  dropout) and no pooling layer. Eager attention, so no fused kernel.
- Pixels: `ToImage(ImageNet)` = uint8 -> float / 255 -> (x - mean) / std.
- The training step: `lejepa_forward` of `references/le-wm-main/train.py`
  (copied below — train.py itself imports hydra/lightning), AdamW lr 5e-5,
  wd 1e-3 on EVERY parameter (stable-pretraining's `exclude_bias_norm`
  defaults to False), gradient-norm clip 1.0.

## Precision

Everything runs in **float64** by default: the dump is then the exact answer
to within float64 rounding, and a Mojo float32 gate measures ITS OWN error
against it. `--noise` re-runs the forward sections in float32 and writes
`noise.<name>` = max |f32 - f64| / std(f64) — the error torch itself makes in
float32, i.e. the scale a Mojo float32 gate should be held to.

## Output format

`tools/act/dump_act_reference.py`'s: one `<name>.bin` per array (raw
little-endian float32, C order) + `manifest.txt` (`name<TAB>d0,d1,...`),
read on the Mojo side by `noeira.deep_agents.act.refload.RefDump`.

⚠ Parameters are dumped in TORCH layout under their torch names
(`param.<state-dict key>`): `nn.Linear.weight` is `[out, in]`, the patch
Conv2d `[192, 3, 14, 14]`, the action Conv1d `[10, 10, 1]`. The Mojo loader
owns every transpose, in ONE place (the ACT lesson: a transposed square
`[192, 192]` projection is a plausible wrong number, not a shape error).

## Sections (`--only` selects one; default all)

* `params`    every tensor of `weights.pt` (incl. BN running stats).
* `encoder`   N frames -> pixels u8 / normalised -> ViT embeddings, each of
              the 12 layer outputs, final LayerNorm, CLS, projector stages
              (Linear, BN eval, GELU, Linear) -> `emb`.
* `action`    z-scored actions (B, T, 10) -> Conv1d -> Linear -> SiLU ->
              Linear -> `act_emb`.
* `predictor` a 3-frame context -> + pos embedding -> each of the 6
              AdaLN-zero blocks (block 0 also internally) -> final LN ->
              `pred_proj` (BN eval) -> `pred`.
* `train`     the training loss on a (B=4, T=4) window, BN in TRAIN mode,
              dropout OFF, SIGReg with an INJECTED projection matrix; the
              gradient of every parameter; BN running stats after the step.
* `adamw`     one AdamW step on those gradients: the clip's total norm and
              each parameter's delta (`adam_delta.<name>` = new - old).
* `steps`     P6: five AdamW steps (lr 5e-5, gate wd 1.0) on fresh windows
              and SIGReg matrices: per-step loss and pre-clip norm, every
              parameter's total delta, BN stats after (`section_steps`).
* `rollout`   `get_cost` for S candidate action sequences (horizon 5,
              history 1 growing to 3): `predicted_emb`, `goal_emb`, `cost`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SWM = REPO / "references" / "stable-worldmodel-0.1.1" / "stable_worldmodel"
sys.path.insert(0, str(REPO))
from tools.act.dump_act_reference import Dump  # noqa: E402

IMG = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ACT_DIM = 10  # frameskip 5 x 2


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


swm_module = _load_file_module("swm_lewm_module", SWM / "wm" / "lewm" / "module.py")
swm_lewm = _load_file_module("swm_lewm", SWM / "wm" / "lewm" / "lewm.py")
swm_loss = _load_file_module("swm_loss", SWM / "wm" / "loss.py")


# ⚠ THE ONE PATCH TO THE REFERENCE. 0.1.1's `Embedder.forward` starts with
# `x = x.float()`, which breaks a float64 run; stable-worldmodel main replaced
# it with a cast to the weights' dtype. In float32 the two are the same op,
# and `_check_embedder_patch` asserts it bit for bit on every run.
_embedder_forward_011 = swm_module.Embedder.forward


def _embedder_forward(self, x):
    x = x.to(self.patch_embed.weight.dtype)
    x = x.permute(0, 2, 1)
    x = self.patch_embed(x)
    x = x.permute(0, 2, 1)
    x = self.embed(x)
    return x


swm_module.Embedder.forward = _embedder_forward


def _check_embedder_patch(model32):
    x = torch.randn(2, 4, 10, dtype=torch.float32)
    ae = model32.action_encoder
    assert torch.equal(_embedder_forward(ae, x), _embedder_forward_011(ae, x)), \
        "the Embedder dtype patch changed the float32 result"


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


def build_model(config: dict):
    from transformers import ViTConfig, ViTModel

    e = config["encoder"]
    assert e["size"] == "tiny" and not e["pretrained"] and not e["use_mask_token"]
    vc = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=192 * 4,
        image_size=e["image_size"],
        patch_size=e["patch_size"],
    )
    vc._attn_implementation = "eager"
    encoder = ViTModel(vc, add_pooling_layer=False, use_mask_token=False)
    encoder.config.interpolate_pos_encoding = True

    p = {k: v for k, v in config["predictor"].items() if k != "_target_"}
    predictor = swm_module.Predictor(**p)
    a = {k: v for k, v in config["action_encoder"].items() if k != "_target_"}
    action_encoder = swm_module.Embedder(**a)

    def mlp(c):
        return swm_module.MLP(
            input_dim=c["input_dim"],
            hidden_dim=c["hidden_dim"],
            output_dim=c["output_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    return swm_lewm.LeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=mlp(config["projector"]),
        pred_proj=mlp(config["pred_proj"]),
    )


# The checkpoint was saved under transformers 4.x ViT names; 5.x renamed the
# same parameters (identical pre-LN block, same defaults). Map 4.x -> 5.x
# when the installed library wants the new names. The DUMP always uses the
# checkpoint's 4.x names (`ckpt_name`), so the Mojo side does not depend on
# the transformers version this ran under.
_VIT_4_TO_5 = [
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
    (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
    (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_", r"encoder.layers.\1.layernorm_"),
]
_CKPT_NAME: dict[str, str] = {}  # model key -> checkpoint key


def ckpt_name(model_key: str) -> str:
    return _CKPT_NAME.get(model_key, model_key)


def load_model(weights: Path, dtype) -> torch.nn.Module:
    import re

    config = json.loads((weights.parent / "config.json").read_text())
    model = build_model(config)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    want = set(model.state_dict())
    mapped = {}
    for k, v in sd.items():
        mk = k
        if mk not in want:
            for pat, rep in _VIT_4_TO_5:
                mk = re.sub(pat, rep, mk)
        _CKPT_NAME[mk] = k
        mapped[mk] = v
    # strict: a key or shape the rebuilt model does not have is an
    # architecture misreading, not something to skip
    model.load_state_dict(mapped, strict=True)
    return model.to(dtype)


def dropout_off(model: torch.nn.Module):
    """Train mode for BatchNorm, but deterministic: every dropout to 0
    (the predictor's is 0.1; `Attention` reads its own float attribute)."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if isinstance(m, swm_module.Attention):
            m.dropout = 0.0


# --------------------------------------------------------------------------- #
# Inputs: PushT-like frames drawn the way gym-pusht draws them
# --------------------------------------------------------------------------- #


def _tee_polys(x, y, theta, scale=30.0, length=4.0):
    """gym_pusht `add_tee`: two rectangles in the body frame, rotated."""
    r1 = [(-length * scale / 2, scale), (length * scale / 2, scale),
          (length * scale / 2, 0), (-length * scale / 2, 0)]
    r2 = [(-scale / 2, scale), (-scale / 2, length * scale),
          (scale / 2, length * scale), (scale / 2, scale)]
    c, s = math.cos(theta), math.sin(theta)
    return [[(x + c * px - s * py, y + s * px + c * py) for px, py in r]
            for r in (r1, r2)]


def draw_frame(agent, block, goal=(256.0, 256.0, math.pi / 4)) -> np.ndarray:
    """(224, 224, 3) uint8. White field, LightGreen goal tee, LightSlateGray
    tee, RoyalBlue agent (r 15), LightGray walls — the gym-pusht palette.
    Drawn at 4x and box-downsampled. Realistic enough for LAYER parity (P2);
    the renderer itself is gated against real dataset frames in P4."""
    from PIL import Image, ImageDraw

    ss = 4
    w = 512
    im = Image.new("RGB", (w * ss, w * ss), (255, 255, 255))
    d = ImageDraw.Draw(im)
    S = lambda pts: [(px * ss, py * ss) for px, py in pts]  # noqa: E731
    for poly in _tee_polys(*goal):
        d.polygon(S(poly), fill=(144, 238, 144))
    for poly in _tee_polys(*block):
        d.polygon(S(poly), fill=(119, 136, 153))
    ax, ay = agent
    d.ellipse([(ax - 15) * ss, (ay - 15) * ss, (ax + 15) * ss, (ay + 15) * ss],
              fill=(65, 105, 225))
    for a, b in [((5, 5), (5, 506)), ((5, 506), (506, 506)),
                 ((506, 506), (506, 5)), ((5, 5), (506, 5))]:
        d.line(S([a, b]), fill=(211, 211, 211), width=4 * ss)
    im = im.resize((IMG, IMG), Image.BOX)
    return np.asarray(im, dtype=np.uint8)


def frames_sequence(rng: np.random.Generator, n_seq: int, t_len: int) -> np.ndarray:
    """(n_seq, t_len, 224, 224, 3) uint8: each sequence a small coherent
    motion of agent + block from a random start."""
    out = np.zeros((n_seq, t_len, IMG, IMG, 3), dtype=np.uint8)
    for i in range(n_seq):
        ag = rng.uniform(80, 432, size=2)
        bl = np.array([*rng.uniform(140, 372, size=2), rng.uniform(-math.pi, math.pi)])
        dag = rng.normal(0, 12, size=2)
        dbl = np.array([*rng.normal(0, 4, size=2), rng.normal(0, 0.05)])
        for t in range(t_len):
            out[i, t] = draw_frame(tuple(ag + t * dag), tuple(bl + t * dbl))
    return out


def preprocess(u8: np.ndarray, dtype) -> torch.Tensor:
    """(..., H, W, 3) uint8 -> (..., 3, H, W): ToImage(ImageNet)."""
    x = torch.from_numpy(u8).to(dtype) / 255.0
    x = x.movedim(-1, -3)
    mean = torch.tensor(IMAGENET_MEAN, dtype=dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=dtype).view(3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def section_params(dump: Dump, model):
    for k, v in model.state_dict().items():
        if v.dtype == torch.int64:  # num_batches_tracked
            continue
        dump.add(f"param.{ckpt_name(k)}", v)


def run_encoder(model, pix):
    """`LeWM.encode`'s image path, stage by stage. Returns a dict."""
    out = {}
    enc = model.encoder
    hs = enc(pix, interpolate_pos_encoding=True, output_hidden_states=True)
    out["vit_embeddings"] = hs.hidden_states[0]
    for i in range(1, len(hs.hidden_states)):
        out[f"vit_layer{i - 1}"] = hs.hidden_states[i]
    out["vit_final_ln"] = hs.last_hidden_state
    cls = hs.last_hidden_state[:, 0]
    out["cls"] = cls
    net = model.projector.net
    h = net[0](cls)
    out["proj_lin1"] = h
    h = net[1](h)
    out["proj_bn"] = h
    h = net[2](h)
    out["proj_gelu"] = h
    out["emb"] = net[3](h)
    # the same through the model's own `encode`, which must agree exactly
    full = model.encode({"pixels": pix.unsqueeze(1)})["emb"][:, 0]
    assert torch.equal(full, out["emb"]), "stage-by-stage encode != LeWM.encode"
    return out


def section_encoder(dump: Dump, model, dtype, rng, n: int, f32_model=None):
    model.eval()
    u8 = frames_sequence(rng, n, 1)[:, 0]
    pix = preprocess(u8, dtype)
    dump.add("enc.pixels_u8", u8.astype(np.float32))
    dump.add("enc.pixels", pix)
    with torch.no_grad():
        out = run_encoder(model, pix)
    for k, v in out.items():
        dump.add(f"enc.{k}", v)
    if f32_model is not None:
        f32_model.eval()
        with torch.no_grad():
            o32 = run_encoder(f32_model, preprocess(u8, torch.float32))
        for k in ("vit_layer0", "vit_layer11", "cls", "emb"):
            _noise(dump, f"enc.{k}", out[k], o32[k])


def run_action(model, act):
    ae = model.action_encoder
    out = {}
    x = act.permute(0, 2, 1)
    x = ae.patch_embed(x).permute(0, 2, 1)
    out["conv"] = x
    h = ae.embed[0](x)
    out["lin1"] = h
    h = ae.embed[1](h)
    out["silu"] = h
    out["act_emb"] = ae.embed[2](h)
    assert torch.equal(ae(act), out["act_emb"]), "stage-by-stage != Embedder"
    return out


def section_action(dump: Dump, model, dtype, rng, b: int, t: int):
    act = torch.from_numpy(rng.normal(size=(b, t, ACT_DIM))).to(dtype)
    dump.add("act.in", act)
    with torch.no_grad():
        out = run_action(model, act)
    for k, v in out.items():
        dump.add(f"act.{k}", v)


def run_predictor(model, emb, act_emb):
    """`Predictor.forward` + `pred_proj`, block by block; block 0 inside."""
    pr = model.predictor
    tr = pr.transformer
    out = {}
    T = emb.size(1)
    x = emb + pr.pos_embedding[:, :T]
    out["x_pos"] = x
    x = tr.input_proj(x)
    c = tr.cond_proj(act_emb)
    for i, blk in enumerate(tr.layers):
        if i == 0:
            mod = blk.adaLN_modulation(c)
            out["blk0.mod"] = mod  # (B, T, 6*D): shift/scale/gate msa, mlp
            sh1, sc1, g1, sh2, sc2, g2 = mod.chunk(6, dim=-1)
            a_in = swm_module.modulate(blk.norm1(x), sh1, sc1)
            out["blk0.attn_in"] = a_in
            a = blk.attn(a_in)
            out["blk0.attn_out"] = a
            x1 = x + g1 * a
            out["blk0.x_mid"] = x1
            m_in = swm_module.modulate(blk.norm2(x1), sh2, sc2)
            m = blk.mlp(m_in)
            out["blk0.mlp_out"] = m
            x_chk = x1 + g2 * m
            x = blk(x, c)
            assert torch.allclose(x, x_chk, rtol=0, atol=1e-12 if x.dtype == torch.float64 else 1e-5)
        else:
            x = blk(x, c)
        out[f"blk{i}"] = x
    x = tr.norm(x)
    out["final_ln"] = x
    x = tr.output_proj(x)
    B = emb.size(0)
    h = x.reshape(B * T, -1)
    net = model.pred_proj.net
    h = net[0](h)
    out["pp_lin1"] = h.reshape(B, T, -1)
    h = net[1](h)
    out["pp_bn"] = h.reshape(B, T, -1)
    h = net[2](h)
    out["pred"] = net[3](h).reshape(B, T, -1)
    assert torch.allclose(
        model.predict(emb, act_emb), out["pred"], rtol=0,
        atol=1e-12 if emb.dtype == torch.float64 else 1e-5,
    ), "stage-by-stage != LeWM.predict"
    return out


def section_predictor(dump: Dump, model, dtype, rng, b: int, f32_model=None):
    """Context = encoded frames of real-looking sequences (not noise), so the
    activations sit in the distribution the weights were trained on."""
    model.eval()
    u8 = frames_sequence(rng, b, 3)
    act = torch.from_numpy(rng.normal(size=(b, 3, ACT_DIM))).to(dtype)
    with torch.no_grad():
        info = model.encode({"pixels": preprocess(u8, dtype), "action": act})
        emb, act_emb = info["emb"], info["act_emb"]
        out = run_predictor(model, emb, act_emb)
    dump.add("pred.emb", emb)
    dump.add("pred.act_emb", act_emb)
    for k, v in out.items():
        dump.add(f"pred.{k}", v)
    if f32_model is not None:
        f32_model.eval()
        with torch.no_grad():
            o32 = run_predictor(f32_model, emb.float(), act_emb.float())
        for k in ("blk0", "blk5", "pred"):
            _noise(dump, f"pred.{k}", out[k], o32[k])


def lejepa_forward(model, sigreg, batch, ctx_len=3, n_preds=1, lambd=0.09):
    """`references/le-wm-main/train.py:lejepa_forward`, minus the logging."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = model.predict(ctx_emb, ctx_act)  # pred
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    output["pred_emb"] = pred_emb
    return output


class _FixedRandn:
    """Replaces `torch.randn` for the duration of ONE SIGReg forward, so the
    reference's own code (`A = randn(D, P); A /= ||A||`) runs verbatim on a
    projection matrix the dump records."""

    def __init__(self, a_raw):
        self.a_raw = a_raw
        self.orig = torch.randn

    def __enter__(self):
        torch.randn = lambda *a, **k: self.a_raw.clone()
        return self

    def __exit__(self, *exc):
        torch.randn = self.orig


def section_train(dump: Dump, model, dtype, rng, b: int = 4, t: int = 4,
                  do_adamw: bool = True, f32_model=None):
    model.train()
    dropout_off(model)
    sigreg = swm_loss.SIGReg(knots=17, num_proj=1024).to(dtype)
    u8 = frames_sequence(rng, b, t)
    pix = preprocess(u8, dtype)
    act = torch.from_numpy(rng.normal(size=(b, t, ACT_DIM))).to(dtype)
    a_raw = torch.from_numpy(rng.normal(size=(192, 1024))).to(dtype)
    dump.add("train.pixels", pix)
    dump.add("train.action", act)
    dump.add("train.sigreg_A_raw", a_raw)
    dump.add("train.sigreg_A", a_raw / a_raw.norm(p=2, dim=0))

    model.zero_grad(set_to_none=True)
    with _FixedRandn(a_raw):
        out = lejepa_forward(model, sigreg, {"pixels": pix, "action": act})
    out["loss"].backward()
    for k in ("emb", "act_emb", "pred_emb"):
        dump.add(f"train.{k}", out[k])
    for k in ("pred_loss", "sigreg_loss", "loss"):
        dump.add(f"train.{k}", out[k].reshape(1))
    print(f"  train: pred_loss {out['pred_loss'].item():.6g}  "
          f"sigreg {out['sigreg_loss'].item():.6g}  loss {out['loss'].item():.6g}")
    n_grad = 0
    for k, p in model.named_parameters():
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        dump.add(f"grad.{ckpt_name(k)}", g)
        n_grad += p.grad is not None
    print(f"  train: {n_grad} parameters received a gradient")
    for k, v in model.state_dict().items():
        if "running_" in k:
            dump.add(f"bn_after.{ckpt_name(k)}", v)

    if f32_model is not None:
        # torch's OWN float32 error on this step, per gradient tensor:
        # max |g32 - g64| / std(g64). The scale a float32 port's gradient
        # gate has to be held to (it is not the forward's).
        f32_model.train()
        dropout_off(f32_model)
        sig32 = swm_loss.SIGReg(knots=17, num_proj=1024)
        f32_model.zero_grad(set_to_none=True)
        with _FixedRandn(a_raw.float()):
            o32 = lejepa_forward(f32_model, sig32, {"pixels": pix.float(), "action": act.float()})
        o32["loss"].backward()
        g64 = dict(model.named_parameters())
        # floor: the global gradient RMS. Several tensors have an EXACT zero
        # gradient (a bias or LN beta feeding a train-mode BatchNorm — BN
        # removes the batch mean; attention key biases — softmax ignores a
        # per-query constant), so std(g64) is roundoff and max|d|/std is
        # meaningless; against the RMS they are judged in absolute terms.
        allg = torch.cat([p.grad.double().reshape(-1) for p in model.parameters()
                          if p.grad is not None])
        rms = allg.square().mean().sqrt().item()
        dump.add("train.grad_rms", np.array([rms]))
        worst = []
        for k, p32 in f32_model.named_parameters():
            r = g64[k].grad.double()
            if p32.grad is None:
                continue
            den = max(r.std().item(), rms)
            v = (p32.grad.double() - r).abs().max().item() / den
            dump.add(f"noise_grad.{ckpt_name(k)}", np.array([v]))
            worst.append((v, ckpt_name(k)))
        worst.sort(reverse=True)
        print(f"  noise grads (max|g32-g64| / max(std, rms={rms:.3g})): max {worst[0][0]:.3g} ({worst[0][1]}); "
              f"median {worst[len(worst) // 2][0]:.3g}")
        for v, k in worst[:6]:
            print(f"     {v:.3g}  {k}")

    if not do_adamw:
        return
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    dump.add("adamw.grad_norm", total.reshape(1))
    opt.step()
    for k, p in model.named_parameters():
        dump.add(f"adam_delta.{ckpt_name(k)}", p.detach() - before[k])
    print(f"  adamw: pre-clip grad norm {total.item():.6g}")


def _adamw_run(model, batches, a_raws, lr, wd, clip):
    """K training steps of `train.py` on the given batches: BN train mode,
    dropout off, SIGReg on the injected matrices, torch AdamW (decay on every
    parameter) after `clip_grad_norm_`. Returns the per-step scalars and,
    per parameter, the max |grad| seen over the K steps."""
    model.train()
    dropout_off(model)
    dtype = next(model.parameters()).dtype
    sigreg = swm_loss.SIGReg(knots=17, num_proj=1024).to(dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rows = []
    gmax = {k: torch.zeros_like(p, dtype=torch.float64) for k, p in model.named_parameters()}
    for (pix, act), a_raw in zip(batches, a_raws):
        opt.zero_grad(set_to_none=True)
        with _FixedRandn(a_raw.to(dtype)):
            out = lejepa_forward(model, sigreg, {"pixels": pix.to(dtype), "action": act.to(dtype)})
        out["loss"].backward()
        for k, p in model.named_parameters():
            if p.grad is not None:
                gmax[k] = torch.maximum(gmax[k], p.grad.detach().double().abs())
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        rows.append([out["loss"].item(), out["pred_loss"].item(),
                     out["sigreg_loss"].item(), total.item()])
    return np.array(rows), gmax


def section_steps(dump: Dump, weights: Path, rng, k: int = 5, b: int = 4, t: int = 4,
                  lr: float = 5e-5, wd: float = 1.0, clip: float = 1.0, noise: bool = True):
    """P6 G6a: K AdamW steps from the published weights, each on its own
    (B, T) window and its own SIGReg matrix. lr is the recipe's (5e-5); wd is
    the GATE's (1.0, not 1e-3): at the recipe's, decoupled decay moves a
    weight by 5e-8 of itself per step, below float32, and would go unchecked;
    at 1.0 it is lr x p, the size of Adam's own step. (lr 1e-3 knocks the
    published model off its minimum — loss 0.18 -> 0.82 in 2 steps — and
    torch float32 then drifts 2.4 % from float64: a chaotic run gates nothing.)

    Dumps the inputs (`steps.<k>.pixels|action|sigreg_A`), the per-step
    scalars (`steps.scalars`: loss, pred_loss, sigreg_loss, pre-clip norm),
    each parameter's total delta (`steps_delta.<name>`) and BN running stats
    after (`steps_bn.<name>`).

    `steps_dead.<name>` marks the elements whose gradient was ZERO at every
    step (|g| < 1e-8 x the gradient RMS: biases / LN betas feeding a
    train-mode BN, the ViT key biases). Adam turns their roundoff gradient
    into a full ±lr step of random sign — in torch float32 as in ours — and
    they cannot move the training loss, so the delta gate skips them; the
    BN running stats downstream DO see them, which `--noise` measures.
    `steps_noise.<name>` = max |d32 - d64| / std(d64) over the live elements:
    torch's own float32 error on the K-step delta, the scale ours is held to.
    """
    model = load_model(weights, torch.float64)
    before = {kk: p.detach().clone() for kk, p in model.named_parameters()}
    batches, a_raws = [], []
    for s in range(k):
        u8 = frames_sequence(rng, b, t)
        pix = preprocess(u8, torch.float64)
        act = torch.from_numpy(rng.normal(size=(b, t, ACT_DIM)))
        a_raw = torch.from_numpy(rng.normal(size=(192, 1024)))
        dump.add(f"steps.{s}.pixels", pix)
        dump.add(f"steps.{s}.action", act)
        dump.add(f"steps.{s}.sigreg_A", a_raw / a_raw.norm(p=2, dim=0))
        batches.append((pix, act))
        a_raws.append(a_raw)
    rows, gmax = _adamw_run(model, batches, a_raws, lr, wd, clip)
    dump.add("steps.scalars", rows)
    dump.add("steps.hparams", np.array([lr, wd, clip, float(k)]))
    rms = math.sqrt(sum(float(g.square().sum()) for g in gmax.values())
                    / sum(g.numel() for g in gmax.values()))
    delta = {}
    n_dead = 0
    for kk, p in model.named_parameters():
        delta[kk] = (p.detach() - before[kk]).double()
        dead = (gmax[kk] < 1e-6 * rms).double()
        n_dead += int(dead.sum())
        dump.add(f"steps_delta.{ckpt_name(kk)}", delta[kk])
        dump.add(f"steps_dead.{ckpt_name(kk)}", dead)
    for kk, v in model.state_dict().items():
        if "running_" in kk:
            dump.add(f"steps_bn.{ckpt_name(kk)}", v)
    for s, r in enumerate(rows):
        print(f"  steps {s}: loss {r[0]:.6g}  pred {r[1]:.6g}  sigreg {r[2]:.6g}  "
              f"pre-clip norm {r[3]:.4g}{'  (clipped)' if r[3] > clip else ''}")
    print(f"  steps: {n_dead} dead gradient elements (skipped by the delta gate)")

    if not noise:
        return
    m32 = load_model(weights, torch.float32)
    rows32, _ = _adamw_run(m32, [(p.float(), a.float()) for p, a in batches],
                           [a.float() for a in a_raws], lr, wd, clip)
    dump.add("steps.noise_scalars", np.abs(rows32 - rows) / np.abs(rows))
    p64 = dict(model.named_parameters())
    worst = []
    for kk, p in m32.named_parameters():
        d32 = p.detach().double() - before[kk].double()
        live = ~(gmax[kk] < 1e-6 * rms)
        d64 = delta[kk]
        if live.sum() < 2:
            v = 0.0
        else:
            sd = d64[live].std().item() or 1.0
            v = (d32 - d64)[live].abs().max().item() / sd
        dump.add(f"steps_noise.{ckpt_name(kk)}", np.array([v]))
        worst.append((v, ckpt_name(kk)))
    sd64 = model.state_dict()
    for kk, v32 in m32.state_dict().items():
        if "running_" in kk:
            r = sd64[kk].double()
            v = (v32.double() - r).abs().max().item() / (r.std().item() or 1.0)
            dump.add(f"steps_noise_bn.{ckpt_name(kk)}", np.array([v]))
            worst.append((v, "bn " + ckpt_name(kk)))
    worst.sort(reverse=True)
    print(f"  steps noise (max|d32-d64|/std(d64), live elements): max {worst[0][0]:.3g} "
          f"({worst[0][1]}); median {worst[len(worst) // 2][0]:.3g}; "
          f"scalars {np.abs(rows32 - rows).max() / np.abs(rows).min():.3g} rel")
    for v, kk in worst[:6]:
        print(f"     {v:.3g}  {kk}")


def section_rollout(dump: Dump, model, dtype, rng, s: int = 8, horizon: int = 5):
    """`LeWM.get_cost` as the CEM solver calls it: one env (B=1), S candidate
    sequences of `horizon` z-scored action blocks, ONE history frame (eval
    `history_size: 1`) growing to the predictor's 3."""
    model.eval()
    u8 = frames_sequence(rng, 2, 1)[:, 0]  # [start, goal]
    pix = preprocess(u8, dtype)
    start = pix[0].expand(1, s, 1, *pix.shape[1:])
    goal = pix[1].expand(1, s, 1, *pix.shape[1:])
    cand = torch.from_numpy(rng.normal(size=(1, s, horizon, ACT_DIM))).to(dtype)
    info = {"pixels": start, "goal": goal, "action": cand[:, :, :1]}
    with torch.no_grad():
        cost = model.get_cost(info, cand)
    dump.add("roll.start_pixels", pix[0])
    dump.add("roll.goal_pixels", pix[1])
    dump.add("roll.candidates", cand)
    dump.add("roll.predicted_emb", info["predicted_emb"])  # (1, S, 1+horizon, D)
    dump.add("roll.goal_emb", info["goal_emb"])
    dump.add("roll.cost", cost)
    print(f"  rollout: predicted_emb {tuple(info['predicted_emb'].shape)}  "
          f"cost {cost.flatten()[:4].tolist()} ...")


def section_cem(dump: Dump, model, dtype, rng, num_samples: int = 64,
                n_steps: int = 4, topk: int = 8, horizon: int = 5, seed: int = 1234):
    """`stable_worldmodel` 0.1.1 `CEMSolver.solve` for ONE env, replicated
    line for line (solver/cem.py imports gymnasium/loguru) around the model's
    own `get_cost`. Per iteration k the dump holds what the gate needs to
    re-run that iteration from torch's state: the raw noise, the incoming
    mean / std, the costs, the elite indices, the updated mean / std.
    Smaller than the eval budget (300 x 30, top 30) — the semantics are the
    same; only a float64 run's cost differs."""
    model.eval()
    u8 = frames_sequence(rng, 2, 1)[:, 0]
    pix = preprocess(u8, dtype)
    info = {
        "pixels": pix[0].expand(1, num_samples, 1, *pix.shape[1:]),
        "goal": pix[1].expand(1, num_samples, 1, *pix.shape[1:]),
        "action": torch.zeros(1, num_samples, 1, ACT_DIM, dtype=dtype),
    }
    dump.add("cem.start_pixels", pix[0])
    dump.add("cem.goal_pixels", pix[1])
    gen = torch.Generator().manual_seed(seed)
    # init_action_distrib: var = var_scale * ones, mean = zeros (no warm start)
    mean = torch.zeros(1, horizon, ACT_DIM, dtype=dtype)
    var = 1.0 * torch.ones(1, horizon, ACT_DIM, dtype=dtype)
    with torch.no_grad():
        for k in range(n_steps):
            noise = torch.randn(1, num_samples, horizon, ACT_DIM, generator=gen, dtype=dtype)
            candidates = noise * var.unsqueeze(1) + mean.unsqueeze(1)
            candidates[:, 0] = mean
            costs = model.get_cost(info, candidates)
            topk_vals, topk_inds = torch.topk(costs, k=topk, dim=1, largest=False)
            topk_candidates = candidates[torch.zeros(1, topk, dtype=torch.long), topk_inds]
            dump.add(f"cem.{k}.noise", noise)
            dump.add(f"cem.{k}.mean_in", mean)
            dump.add(f"cem.{k}.var_in", var)
            dump.add(f"cem.{k}.costs", costs)
            dump.add(f"cem.{k}.topk_inds", topk_inds.double())
            mean = topk_candidates.mean(dim=1)
            var = topk_candidates.std(dim=1)
            dump.add(f"cem.{k}.mean_out", mean)
            dump.add(f"cem.{k}.var_out", var)
    dump.add("cem.actions", mean)
    print(f"  cem: {n_steps} iterations x {num_samples} samples, top {topk}; "
          f"final elite cost {topk_vals.mean().item():.4g}")


def _noise(dump: Dump, name: str, ref64, got32):
    """max |f32 - f64| / std(f64): torch's own float32 error at this stage."""
    r = ref64.double()
    g = got32.double()
    sd = r.std().item() or 1.0
    v = (g - r).abs().max().item() / sd
    dump.add(f"noise.{name}", np.array([v]))
    print(f"  noise {name:22s} max|f32-f64|/std = {v:.3g}")


# --------------------------------------------------------------------------- #


SECTIONS = ("params", "encoder", "action", "predictor", "train", "rollout", "cem", "steps")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="/tmp/lewm_ref")
    ap.add_argument("--weights",
                    default=str(Path.home() / ".cache/noeira/lewm_pusht/hf/weights.pt"))
    ap.add_argument("--only", choices=SECTIONS)
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--noise", action="store_true",
                    help="also measure torch's own float32 error per stage")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    dump = Dump(Path(args.out))
    weights = Path(args.weights)
    model = load_model(weights, dtype)
    f32 = load_model(weights, torch.float32) if args.noise else None
    _check_embedder_patch(f32 if f32 is not None else load_model(weights, torch.float32))
    print(f"reference model: {sum(p.numel() for p in model.parameters())} params, "
          f"{args.dtype}, weights {weights}")

    run = (lambda s: args.only in (None, s))  # noqa: E731
    # each section draws from its own generator so --only reproduces it
    sub = lambda k: np.random.default_rng([args.seed, k])  # noqa: E731
    if run("params"):
        section_params(dump, model)
    if run("encoder"):
        section_encoder(dump, model, dtype, sub(1), n=8, f32_model=f32)
    if run("action"):
        section_action(dump, model, dtype, sub(2), b=4, t=4)
    if run("predictor"):
        section_predictor(dump, model, dtype, sub(3), b=4, f32_model=f32)
    if run("rollout"):  # before `train`: the optimizer step mutates the model
        section_rollout(dump, model, dtype, sub(5))
    if run("cem"):
        section_cem(dump, model, dtype, sub(6))
    if run("train"):
        section_train(dump, model, dtype, sub(4), f32_model=f32)
    if run("steps"):  # its own float64 model from the weights: `train` stepped this one
        section_steps(dump, weights, sub(7), noise=args.noise)
    dump.close()


if __name__ == "__main__":
    main()
