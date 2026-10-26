"""SO-101 world model plan, step 2: do FROZEN pretrained encoders carry a
"cube between the open jaws" signal from sim renders to real frames?

noeira-docs/SO101_LEWM_PLAN.md §3c step 2. Two subcommands, run on a CUDA
box in a venv with torch, h5py, scikit-learn (+ the hub clones below):

    python so101_frozen_encoder_step2.py features --enc vjepa21 --dir W \\
        --hub W/vjepa2 --weights W/vjepa2_1_vitG_384.pt
    python so101_frozen_encoder_step2.py features --enc dinov2 --dir W --hub W/dinov2
    python so101_frozen_encoder_step2.py report --dir W --enc vjepa21,dinov2

`W` holds three stores (2 cameras x 3 x 240 x 320 u8, LeRobot units):
    real_step2.h5          the real teleop rows used here: `rows` (row ids
                           in the full store), `images`, `qpos`, the twin
                           index (`twin_rows`, `twin_ep`, `twin_tick`) and the
                           labels (`lab_*`, `so101_preclose_labels`)
    twin.rendered.h5       the sim twins (`lewm_so101_twin.mojo`, plain
                           look): twin episode e, tick t = row ep_offset[e] + t
    sim150.rendered.h5     per-episode-DR teacher episodes; labelled here

Encoders (each camera encoded on its own, ImageNet normalisation):
    vjepa21   V-JEPA 2.1 ViT-G/384, RoboJEPA's way (`world_model.encode`):
              one frame (T = 1) at 240 x 320, a 15 x 20 x 1664 token grid,
              parameter-free layer norm. ⚠ research-only weights
    dinov2    DINOv2 ViT-L/14 at 238 x 322 (bilinear), 17 x 23 x 1024
              normed patch tokens
Per frame and camera: the token MEAN and a 3 x 4 average-pooled GRID.

`report`:
  A. twin alignment (`lewm_so101_align.mojo`'s protocol on the mean
     features, both cameras concatenated): d_step (consecutive sim twins),
     d_rand (random sim pairs), d_pair (real vs its twin after subtracting
     the episode's tick-0 offset, ticks >= 30) and top-1 (the nearest sim
     twin is the same episode within 5 ticks). Scale-free: ÷ d_rand.
  B. the pre-close probe: logistic regression (standardised features, C
     picked by episode-grouped CV on sim) fit on sim, AUC on real
     positives vs approach negatives. Ceiling: the same probe
     cross-validated on real, grouped by episode (C picked on the test
     folds: an optimistic ceiling). Baseline: the jaw angle q alone.
     Gate (plan): real AUC >= 0.85 and >= 90 % of the ceiling.
"""
import argparse
import os
import sys

import h5py
import numpy as np

H, W = 240, 320
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def _frames(f, rows):
    """(n, 2, 3, H, W) uint8 of an open store's `images` at `rows`."""
    u, inv = np.unique(np.asarray(rows), return_inverse=True)  # h5py: sorted, unique
    return f["images"][u].reshape(-1, 2, 3, H, W)[inv]


# ── encoders ───────────────────────────────────────────────────────────────


class _Enc:
    def __init__(self, name, hub, weights):
        import torch

        self.torch = torch
        self.name = name
        if name == "vjepa21":
            enc, _ = torch.hub.load(hub, "vjepa2_1_vit_gigantic_384", source="local", pretrained=False)
            sd = torch.load(weights, map_location="cpu", mmap=True)["target_encoder"]
            enc.load_state_dict({k.replace("module.backbone.", ""): v for k, v in sd.items()})
            self.size, self.patch = (H, W), 16
        elif name == "dinov2":
            enc = torch.hub.load(hub, "dinov2_vitl14", source="local", pretrained=True)
            self.size, self.patch = (238, 322), 14
        else:
            raise ValueError(name)
        self.enc = enc.cuda().eval()

    def tokens(self, x):
        """x (b, 3, H, W) float normalised -> (b, gh, gw, d) float32."""
        torch = self.torch
        F = torch.nn.functional
        if self.size != (H, W):
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        gh, gw = self.size[0] // self.patch, self.size[1] // self.patch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.name == "vjepa21":
                h = self.enc(x[:, :, None])  # (b, 3, T=1, H, W)
            else:
                h = self.enc.forward_features(x)["x_norm_patchtokens"]
        h = h.float().reshape(x.shape[0], gh, gw, -1)
        if self.name == "vjepa21":
            h = F.layer_norm(h, (h.shape[-1],))
        return h

    def features(self, frames, batch=32):
        """frames (n, 2, 3, H, W) uint8 -> mean (n, 2, d), grid (n, 2, 12, d) f16."""
        torch = self.torch
        means, grids = [], []
        for i in range(0, len(frames), batch):
            x = frames[i : i + batch].reshape(-1, 3, H, W).astype(np.float32) / 255.0
            x = torch.from_numpy((x - MEAN) / STD).cuda()
            h = self.tokens(x)  # (b*2, gh, gw, d)
            g = torch.nn.functional.adaptive_avg_pool2d(h.permute(0, 3, 1, 2), (3, 4))
            b = h.shape[0] // 2
            means.append(h.mean((1, 2)).reshape(b, 2, -1).half().cpu().numpy())
            grids.append(g.flatten(2).transpose(1, 2).reshape(b, 2, 12, -1).half().cpu().numpy())
        return np.concatenate(means), np.concatenate(grids)


def cmd_features(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from so101_preclose_labels import store_labels

    enc = _Enc(a.enc, a.hub, a.weights)
    out = {}
    with h5py.File(os.path.join(a.dir, "real_step2.h5"), "r") as f:
        fr = f["images"][:].reshape(-1, 2, 3, H, W)
        out["real_mean"], out["real_grid"] = enc.features(fr)
        print(a.enc, "real", fr.shape[0], "frames", flush=True)
    with h5py.File(os.path.join(a.dir, "twin.rendered.h5"), "r") as f:
        off = f["ep_offset"][:]
    with h5py.File(os.path.join(a.dir, "real_step2.h5"), "r") as f:
        te, tt = f["twin_ep"][:], f["twin_tick"][:]
    with h5py.File(os.path.join(a.dir, "twin.rendered.h5"), "r") as f:
        out["twin_mean"], _ = enc.features(_frames(f, off[te] + tt))
        print(a.enc, "twin", len(te), "frames", flush=True)
    with h5py.File(os.path.join(a.dir, "sim150.rendered.h5"), "r") as f:
        row, ep, tick, y, kind = store_labels(f)
        out["sim_mean"], out["sim_grid"] = enc.features(_frames(f, row))
        out["sim_q"] = f["qpos"][:, 5][row]
        out["sim_ep"], out["sim_y"], out["sim_kind"] = ep, y, kind
        print(a.enc, "sim", len(row), "frames", flush=True)
    np.savez(os.path.join(a.dir, f"feat_{a.enc}.npz"), **out)


# ── report ─────────────────────────────────────────────────────────────────


def _twin(fe, real):
    rows = real["rows"][:]
    ri = np.searchsorted(rows, real["twin_rows"][:])
    zr = fe["real_mean"][ri].astype(np.float64).reshape(len(ri), -1)
    zs = fe["twin_mean"].astype(np.float64).reshape(len(ri), -1)
    ep, tk = real["twin_ep"][:], real["twin_tick"][:]
    rng = np.random.default_rng(0)
    nxt = [(i, j) for i in range(len(ep)) for j in [i + 1] if j < len(ep) and ep[j] == ep[i]]
    d_step = np.mean([np.linalg.norm(zs[i] - zs[j]) for i, j in nxt])
    pr = rng.integers(0, len(ep), (2000, 2))
    d_rand = np.mean([np.linalg.norm(zs[i] - zs[j]) for i, j in pr if i != j])
    zc = zr.copy()
    for e in np.unique(ep):
        m = ep == e
        i0 = np.nonzero(m & (tk == 0))[0][0]
        zc[m] -= zr[i0] - zs[i0]
    late = tk >= 30
    d_pair = np.mean(np.linalg.norm(zc[late] - zs[late], axis=1))
    d_raw = np.mean(np.linalg.norm(zr[late] - zs[late], axis=1))
    hit = 0
    for i in np.nonzero(late)[0]:
        j = np.argmin(np.linalg.norm(zs - zc[i], axis=1))
        hit += ep[j] == ep[i] and abs(tk[j] - tk[i]) <= 5
    return d_step, d_rand, d_raw, d_pair, hit / late.sum()


def _probe_sets(fe, real, wrist_only):
    rows = real["rows"][:]
    li = np.searchsorted(rows, real["lab_rows"][:])
    cam = slice(1, 2) if wrist_only else slice(0, 2)
    sets = {}
    for kind in ("mean", "grid"):
        s = fe[f"sim_{kind}"][:, cam].astype(np.float32).reshape(len(fe["sim_y"]), -1)
        r = fe[f"real_{kind}"][li][:, cam].astype(np.float32).reshape(len(li), -1)
        sets[kind] = (s, r)
    return sets, li


def _fit_auc(xs, ys, gs, xt, yt, Cs, sim_cv):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def model(C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))

    best, best_c = -1.0, Cs[0]
    if sim_cv:
        for C in Cs:
            p = np.zeros(len(ys))
            for tr, te in GroupKFold(5).split(xs, ys, gs):
                p[te] = model(C).fit(xs[tr], ys[tr]).predict_proba(xs[te])[:, 1]
            auc = roc_auc_score(ys, p)
            if auc > best:
                best, best_c = auc, C
    m = model(best_c).fit(xs, ys)
    return roc_auc_score(yt, m.predict_proba(xt)[:, 1]), best, best_c


def _cv_auc(x, y, g, Cs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best = -1.0
    for C in Cs:
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(x, y, g):
            p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(
                x[tr], y[tr]).predict_proba(x[te])[:, 1]
        best = max(best, roc_auc_score(y, p))
    return best


def cmd_report(a):
    from sklearn.metrics import roc_auc_score

    real = h5py.File(os.path.join(a.dir, "real_step2.h5"), "r")
    ry, rk, rep = real["lab_y"][:], real["lab_kind"][:], real["lab_ep"][:]
    li = np.searchsorted(real["rows"][:], real["lab_rows"][:])
    rq = real["qpos"][:, 5][li]
    Cs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    print(f"real labelled: {int((ry == 1).sum())} pos / {int((ry == 0).sum())} neg in {len(set(rep))} episodes")
    for name in a.enc.split(","):
        fe = np.load(os.path.join(a.dir, f"feat_{name}.npz"))
        print(f"\n== {name} ==")
        d_step, d_rand, d_raw, d_pair, top1 = _twin(fe, real)
        print(f"A. twins: d_step {d_step:.3f} d_rand {d_rand:.3f} | real-sim raw {d_raw:.3f}, offset-corrected"
              f" {d_pair:.3f} = {d_pair / d_rand:.2f} x d_rand (ViT-tiny v1: 0.67) | step {d_step / d_rand:.2f}"
              f" x d_rand (ViT-tiny 0.14) | top-1 {100 * top1:.1f} % (ViT-tiny 27.5 %)")
        sy, sk, sep = fe["sim_y"], fe["sim_kind"], fe["sim_ep"]
        print(f"   sim labelled: {int((sy == 1).sum())} pos / {int((sk == 0).sum())} approach /"
              f" {int((sk == 2).sum())} missed-grasp in {len(set(sep))} episodes")
        for wrist_only in (False, True):
            sets, _ = _probe_sets(fe, real, wrist_only)
            for kind, (xs, xr) in sets.items():
                auc, sim_auc, c = _fit_auc(xs, sy, sep, xr, ry, Cs, True)
                ceil = _cv_auc(xr, ry, rep, Cs)
                hard = sk != 0
                sim_hard = _cv_auc(xs[hard], sy[hard], sep[hard], [c]) if len(set(sy[hard])) == 2 else float("nan")
                tag = ("wrist " if wrist_only else "both  ") + kind
                gate = "PASS" if auc >= 0.85 and auc >= 0.9 * ceil else "fail"
                print(f"B. {tag}: sim->real AUC {auc:.3f} | real ceiling {ceil:.3f} ({auc / ceil:.2f})"
                      f" | sim CV {sim_auc:.3f}, pos vs missed {sim_hard:.3f} | C {c:g} | gate {gate}")
    # the jaw angle alone: what the pictures must beat
    fe = np.load(os.path.join(a.dir, f"feat_{a.enc.split(',')[0]}.npz"))
    sq = fe["sim_q"].astype(np.float64)[:, None]
    auc_q, _, _ = _fit_auc(sq, fe["sim_y"], fe["sim_ep"], rq[:, None], ry, [1.0], False)
    print(f"\nbaseline jaw q only: sim->real AUC {auc_q:.3f} | real ceiling {_cv_auc(rq[:, None], ry, rep, [1.0]):.3f}"
          f" | raw q AUC on real {roc_auc_score(ry, -rq):.3f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--enc", required=True, choices=["vjepa21", "dinov2"])
    f.add_argument("--dir", required=True)
    f.add_argument("--hub", required=True)
    f.add_argument("--weights", default="")
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--enc", default="vjepa21,dinov2")
    a = p.parse_args()
    cmd_features(a) if a.cmd == "features" else cmd_report(a)


if __name__ == "__main__":
    main()
