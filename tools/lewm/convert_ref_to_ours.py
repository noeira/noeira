#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | LeWM reference dump -> our parameter names and layouts
# +--------------------------------------------------------------------------+ #
"""Re-express a `dump_lewm_reference.py` dump in THIS framework's names and
layouts, so the Mojo gates read it with `refload.LoadRefParams` unchanged.

    pixi run mojo run -I . tools/lewm/list_ref_params.mojo > /tmp/lewm_ref/ours_names.tsv
    pixi run -e act-ref python tools/lewm/convert_ref_to_ours.py --dump /tmp/lewm_ref

Writes, into the same dump directory, for every Param / State name of
`ref_model.LeWMLossGraphRef` (as listed by `list_ref_params.mojo`):

    ours.<name>          the checkpoint value           (P and S)
    ours_grad.<name>     d loss / d param of `train`     (P)
    ours_adam.<name>     the AdamW step's delta          (P)
    ours_bn_after.<name> running stats after `train`     (S)

## The ONE place layouts are decided

`MAP` below: our name -> (reference tensor(s), transform). Every transform is
linear and applies the same way to weights, gradients and Adam deltas.

- `Linear`: ours `[in, out]` row-major (y = x @ W + b); torch `[out, in]` -> T.
- ViT q/k/v: three torch Linears -> our fused `[q|k|v]` (`QKVToMajor`'s order).
- predictor AdaLN: one torch Linear(192 -> 1152), chunked in the reference's
  order (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) ->
  our `sh1 sc1 g1 sh2 sc2 g2` ZeroLinears.
- the predictor's `to_qkv` has NO bias; ours does -> zeros, no grad/adam
  counterpart (the documented deviation, ref_model.mojo).
- patch Conv2D: ours `[OC, IC*K*K]` = torch `[OC, IC, K, K]` flattened.

Coverage is asserted both ways: every one of our names is produced, and every
reference Param is consumed exactly once. A misread architecture fails here,
before any Mojo runs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.act.dump_act_reference import Dump  # noqa: E402

NO_REF = "NO_REF"  # our tensor with no reference counterpart (zeros)


def _rules():
    """(regex on our name) -> f(match) -> (list of ref keys, transform)."""
    T = lambda xs: xs[0].T  # noqa: E731
    same = lambda xs: xs[0]  # noqa: E731
    cat_t = lambda xs: np.concatenate(xs, axis=0).T  # noqa: E731  (q|k|v weights)
    cat = lambda xs: np.concatenate(xs, axis=0)  # noqa: E731
    flat = lambda xs: xs[0].reshape(-1)  # noqa: E731
    E = "encoder.encoder.layer"
    ada = ["sh1", "sc1", "g1", "sh2", "sc2", "g2"]

    def chunk(j, transpose):
        def f(xs):
            c = np.split(xs[0], 6, axis=0)[j]
            return c.T if transpose else c
        return f

    def conv1d_t(xs):  # (out, in, 1) -> [in, out]
        return xs[0][:, :, 0].T

    rules = [
        (r"emb\.0\.0\.0\.weight", lambda m: (["encoder.embeddings.patch_embeddings.projection.weight"], flat)),
        (r"emb\.0\.0\.0\.bias", lambda m: (["encoder.embeddings.patch_embeddings.projection.bias"], same)),
        (r"emb\.0\.1\.tokens", lambda m: (["encoder.embeddings.cls_token"], flat)),
        (r"emb\.0\.2\.bias", lambda m: (["encoder.embeddings.position_embeddings"], flat)),
        (r"emb\.0\.3\.(\d+)\.0\.0\.0\.0\.(gamma|beta)", lambda m: (
            [f"{E}.{m[1]}.layernorm_before.{'weight' if m[2] == 'gamma' else 'bias'}"], same)),
        (r"emb\.0\.3\.(\d+)\.0\.0\.1\.0\.0\.weight", lambda m: (
            [f"{E}.{m[1]}.attention.attention.{q}.weight" for q in ("query", "key", "value")], cat_t)),
        (r"emb\.0\.3\.(\d+)\.0\.0\.1\.0\.0\.bias", lambda m: (
            [f"{E}.{m[1]}.attention.attention.{q}.bias" for q in ("query", "key", "value")], cat)),
        (r"emb\.0\.3\.(\d+)\.0\.0\.1\.3\.0\.(weight|bias)", lambda m: (
            [f"{E}.{m[1]}.attention.output.dense.{m[2]}"], T if m[2] == "weight" else same)),
        (r"emb\.0\.3\.(\d+)\.1\.0\.0\.0\.(gamma|beta)", lambda m: (
            [f"{E}.{m[1]}.layernorm_after.{'weight' if m[2] == 'gamma' else 'bias'}"], same)),
        (r"emb\.0\.3\.(\d+)\.1\.0\.1\.0\.0\.(weight|bias)", lambda m: (
            [f"{E}.{m[1]}.intermediate.dense.{m[2]}"], T if m[2] == "weight" else same)),
        (r"emb\.0\.3\.(\d+)\.1\.0\.1\.2\.0\.(weight|bias)", lambda m: (
            [f"{E}.{m[1]}.output.dense.{m[2]}"], T if m[2] == "weight" else same)),
        (r"emb\.0\.4\.0\.(gamma|beta)", lambda m: (
            [f"encoder.layernorm.{'weight' if m[1] == 'gamma' else 'bias'}"], same)),
        # projectors: ours Sequential[Linear(0), BN(1), GELU(2), Linear(3)]
        (r"(emb\.0\.6|pred\.0)\.(0|3)\.(weight|bias)", lambda m: (
            [f"{'projector' if m[1] == 'emb.0.6' else 'pred_proj'}.net.{m[2]}.{m[3]}"],
            T if m[3] == "weight" else same)),
        (r"(emb\.0\.6|pred\.0)\.1\.(gamma|beta|running_mean|running_var)", lambda m: (
            [f"{'projector' if m[1] == 'emb.0.6' else 'pred_proj'}.net.1."
             f"{ {'gamma': 'weight', 'beta': 'bias'}.get(m[2], m[2]) }"], same)),
        (r"act_emb\.0\.0\.weight", lambda m: (["action_encoder.patch_embed.weight"], conv1d_t)),
        (r"act_emb\.0\.0\.bias", lambda m: (["action_encoder.patch_embed.bias"], same)),
        (r"act_emb\.1\.0\.(weight|bias)", lambda m: (
            [f"action_encoder.embed.0.{m[1]}"], T if m[1] == "weight" else same)),
        (r"act_emb\.2\.0\.(weight|bias)", lambda m: (
            [f"action_encoder.embed.2.{m[1]}"], T if m[1] == "weight" else same)),
        (r"x_pe\.bias", lambda m: (["predictor.pos_embedding"], flat)),
        (r"pred_raw\.(\d+)\.(sh1|sc1|g1|sh2|sc2|g2)\.0\.(weight|bias)", lambda m: (
            [f"predictor.transformer.layers.{m[1]}.adaLN_modulation.1.{m[3]}"],
            chunk(ada.index(m[2]), m[3] == "weight"))),
        (r"pred_raw\.(\d+)\.ln_a\.0\.(gamma|beta)", lambda m: (
            [f"predictor.transformer.layers.{m[1]}.attn.norm.{'weight' if m[2] == 'gamma' else 'bias'}"], same)),
        (r"pred_raw\.(\d+)\.attn\.0\.0\.weight", lambda m: (
            [f"predictor.transformer.layers.{m[1]}.attn.to_qkv.weight"], T)),
        (r"pred_raw\.(\d+)\.attn\.0\.0\.bias", lambda m: ([NO_REF], None)),
        (r"pred_raw\.(\d+)\.attn\.3\.0\.(weight|bias)", lambda m: (
            [f"predictor.transformer.layers.{m[1]}.attn.to_out.0.{m[2]}"], T if m[2] == "weight" else same)),
        (r"pred_raw\.(\d+)\.ln_f\.0\.(gamma|beta)", lambda m: (
            [f"predictor.transformer.layers.{m[1]}.mlp.net.0.{'weight' if m[2] == 'gamma' else 'bias'}"], same)),
        (r"pred_raw\.(\d+)\.mlp\.0\.0\.(weight|bias)", lambda m: (
            [f"predictor.transformer.layers.{m[1]}.mlp.net.1.{m[2]}"], T if m[2] == "weight" else same)),
        (r"pred_raw\.(\d+)\.mlp\.2\.0\.(weight|bias)", lambda m: (
            [f"predictor.transformer.layers.{m[1]}.mlp.net.4.{m[2]}"], T if m[2] == "weight" else same)),
        (r"pred_ln\.0\.(gamma|beta)", lambda m: (
            [f"predictor.transformer.norm.{'weight' if m[1] == 'gamma' else 'bias'}"], same)),
    ]
    return [(re.compile(p + r"$"), f) for p, f in rules]


def read_dump(root: Path) -> dict[str, np.ndarray]:
    arrays = {}
    for line in (root / "manifest.txt").read_text().splitlines():
        name, shape = line.split("\t")
        shp = tuple(int(x) for x in shape.split(",")) if shape else ()
        arrays[name] = np.fromfile(root / f"{name}.bin", dtype="<f4").reshape(shp)
    return arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/lewm_ref")
    ap.add_argument("--names", help="list_ref_params.mojo output (default <dump>/ours_names.tsv)")
    args = ap.parse_args()
    root = Path(args.dump)
    names = Path(args.names) if args.names else root / "ours_names.tsv"
    ref = read_dump(root)
    rules = _rules()

    ours = []
    for line in names.read_text().splitlines():
        kind, name, size = line.split("\t")
        ours.append((kind, name, int(size)))

    out = Dump(root)
    used: dict[str, int] = {}
    n_grad = n_adam = n_bn = 0
    for kind, name, size in ours:
        hit = [(m, f) for rx, f in rules if (m := rx.match(name))]
        if len(hit) != 1:
            sys.exit(f"{name}: {len(hit)} mapping rules match (need exactly 1)")
        m, f = hit[0]
        keys, tf = f(m)
        if keys == [NO_REF]:
            out.add(f"ours.{name}", np.zeros(size, dtype=np.float32))
            continue
        for k in keys:
            used[k] = used.get(k, 0) + 1

        def conv(prefix):
            xs = [ref[f"{prefix}.{k}"] for k in keys]
            v = np.ascontiguousarray(tf(xs)).reshape(-1)
            if v.size != size:
                sys.exit(f"{name}: mapped {prefix} has {v.size} values, ours holds {size}")
            return v

        out.add(f"ours.{name}", conv("param"))
        if kind == "P":
            if all(f"grad.{k}" in ref for k in keys):
                out.add(f"ours_grad.{name}", conv("grad")); n_grad += 1
            if all(f"noise_grad.{k}" in ref for k in keys):
                # torch's own float32 gradient error; a fused tensor takes
                # the worst of its parts
                out.add(f"ours_noise_grad.{name}",
                        np.array([max(float(ref[f"noise_grad.{k}"][0]) for k in keys)]))
            if all(f"adam_delta.{k}" in ref for k in keys):
                out.add(f"ours_adam.{name}", conv("adam_delta")); n_adam += 1
        elif all(f"bn_after.{k}" in ref for k in keys):
            out.add(f"ours_bn_after.{name}", conv("bn_after")); n_bn += 1

    ref_params = {k[len("param."):] for k in ref if k.startswith("param.")}
    unused = sorted(ref_params - set(used))
    # the AdaLN Linear is split six ways (one chunk per ZeroLinear): six uses
    want_uses = lambda k: 6 if ".adaLN_modulation." in k else 1  # noqa: E731
    twice = sorted(k for k, c in used.items() if c != want_uses(k))
    if unused or twice:
        sys.exit(f"reference tensors unused: {unused[:8]}  used the wrong number of times: {twice[:8]}")
    out.close()
    print(f"  {len(ours)} of our tensors mapped from {len(ref_params)} reference "
          f"tensors (each used once; AdaLN six ways); grads {n_grad}, adam deltas {n_adam}, bn_after {n_bn}")


if __name__ == "__main__":
    main()
