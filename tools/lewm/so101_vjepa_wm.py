"""SO-101 world model on FROZEN V-JEPA 2.1 features: does a predictor
trained on sim features transfer to real?

noeira-docs/SO101_LEWM_PLAN.md, "Frozen V-JEPA world model". The ViT-tiny
world models (v1 / v2, + inverse dynamics) learn a look-invariant arm state
inside sim but fail S3 on real. This trains only a predictor, on the frozen
features of V-JEPA 2.1 ViT-G (RoboJEPA's encoder), and scores it with the
ViT-tiny checks' numbers: pred / copy-last on held-out sim and on real, real /
sim pred, the actions-shuffled ratio, and a joint-angle probe fitted on sim and
applied to real.

    python so101_vjepa_wm.py split DEMO OUTDIR EPISODES_PER_CHUNK
    python so101_vjepa_wm.py export-real STORE OUT.h5         # laptop, no torch
    python so101_vjepa_wm.py encode --store S --out F.npz --hub H --weights W
    python so101_vjepa_wm.py train --feats a.npz,b.npz --out DIR
    python so101_vjepa_wm.py eval --model DIR --set sim_dr:F.npz --set real:G.npz

## Frames and actions

The ViT-tiny windows (`experimental/lewm/so101_data.mojo`): T = 4 frames
FS = 5 ticks apart; for frame t, the FS per-tick changes of the commanded
target from that tick on, z-scored per joint with the training set's
statistics. Here only the ticks 0, 5, 10, ... of each episode are encoded
(one window phase out of five; the frames 1 tick apart are near-duplicates),
so a window is 4 consecutive encoded frames of one episode.

The target changes are taken in LEROBOT UNITS (`action` and `qpos`, both
stores): the sim store is rendered with `--joint-zero follower`, so the two
domains share the unit map, and LeRobot units are a per-joint affine map of
the model radians, which the per-joint z-score removes. The first tick of an
episode changes the target from the arm's own pose, as in the Mojo loader.
Teleop's `action` is the leader's position (no target-mode clamp).

## Features

`so101_frozen_encoder_step2._Enc`: each camera alone, one frame (T = 1) at
240 x 320, a 15 x 20 x 1664 token grid, layer-normed, average-pooled to a
3 x 4 grid. A frame is 24 tokens (2 cameras x 12 cells) of 1664, kept fp16.

## Predictor

Tokens -> Linear(1664, d), plus a learnt (time, token) position, plus an
action embedding (MLP 30 -> d) added to every token of its frame; a
pre-norm transformer with a frame-causal mask; Linear(d, 1664) back,
RESIDUAL on the input frame (predicts the change). Teacher-forced: frames
0..2 and actions 0..2 predict frames 1..3, MSE. The target is frozen, so
there is no collapse to guard against: no SIGReg, no inverse dynamics.
"""
import argparse
import json
import os
import struct
import sys

import h5py
import numpy as np

H, W = 240, 320
FRAME = 2 * 3 * H * W
FS, T, J = 5, 4, 6
VAL_EVERY = 10


# ── .demo split (deep_agents/demos/file.mojo, version 1) ──────────────────


def cmd_split(a):
    """Cut a `.demo` into files of `n` episodes each, so a 240 x 320 render
    of one fits the box's disk (a full-res row is 460,800 bytes)."""
    b = open(a.demo, "rb").read()
    assert b[:8] == b"MRLDEMO1", a.demo
    ver, od, ad = struct.unpack_from("<III", b, 8)
    n, ne = struct.unpack_from("<QI", b, 20)
    assert ver == 1
    hdr = 32
    rb = 4 * (2 * od + ad + 2) + 4
    tab = hdr + n * rb
    assert len(b) == tab + 20 * ne, "unexpected .demo size"
    eps = [struct.unpack_from("<QQI", b, tab + 20 * e) for e in range(ne)]
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.basename(a.demo).removesuffix(".demo")
    for c, e0 in enumerate(range(0, ne, a.n)):
        sel = eps[e0 : e0 + a.n]
        rows = b"".join(b[hdr + s * rb : hdr + (s + ln) * rb] for s, ln, _ in sel)
        m = sum(ln for _, ln, _ in sel)
        out = bytearray(b[:8]) + struct.pack("<IIIQI", ver, od, ad, m, len(sel)) + rows
        start = 0
        for _, ln, ok in sel:
            out += struct.pack("<QQI", start, ln, ok)
            start += ln
        p = os.path.join(a.outdir, f"{stem}_c{c}.demo")
        open(p, "wb").write(out)
        print(p, len(sel), "episodes", m, "rows")


# ── what gets encoded ──────────────────────────────────────────────────────


def _frames_of(ep_off, ep_len):
    """Rows encoded: ticks 0, FS, 2 FS, ... of each episode, with (ep, tick)."""
    rows, ep, tick = [], [], []
    for e, (o, ln) in enumerate(zip(ep_off, ep_len)):
        for t in range(0, ln, FS):
            rows.append(o + t)
            ep.append(e)
            tick.append(t)
    return np.array(rows), np.array(ep), np.array(tick)


def _dtarget(qpos, action, ep_off, ep_len):
    """Per tick: commanded target minus the previous one (the arm's pose on
    an episode's first tick). LeRobot units, (n, 6) float64."""
    tgt = action.astype(np.float64)
    prev = np.empty_like(tgt)
    prev[1:] = tgt[:-1]
    prev[ep_off] = qpos[ep_off]
    return tgt - prev


def _blocks(dt, rows, ep_off, ep_len, ep):
    """Each encoded frame's FS per-tick changes from its tick on, (k, 30);
    `ok` where all FS ticks exist."""
    end = (np.asarray(ep_off) + np.asarray(ep_len))[ep]
    ok = rows + FS <= end
    idx = np.minimum(rows[:, None] + np.arange(FS)[None], len(dt) - 1)
    return dt[idx].reshape(len(rows), FS * J).astype(np.float32), ok


def cmd_export_real(a):
    """The real teleop store's encoded rows only (ticks 0, 5, ... per
    episode) plus every row's qpos / action: ~1/5 of the 7 GB to upload."""
    with h5py.File(a.store, "r") as f, h5py.File(a.out, "w") as g:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        rows, _, _ = _frames_of(off, ln)
        for k in ("qpos", "action", "ep_offset", "ep_len"):
            g[k] = f[k][:]
        g["frame_rows"] = rows
        d = g.create_dataset("frame_images", (len(rows), FRAME), np.uint8, chunks=(1, FRAME),
                             compression="gzip", compression_opts=4)
        for i in range(0, len(rows), 256):
            d[i : i + 256] = f["images"][rows[i : i + 256]]
        print(a.out, len(rows), "frames of", f["images"].shape[0], "rows,", len(off), "episodes")


def cmd_encode(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from so101_frozen_encoder_step2 import _Enc

    enc = _Enc("vjepa21", a.hub, a.weights)
    with h5py.File(a.store, "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        qpos, action = f["qpos"][:], f["action"][:]
        rows, ep, tick = _frames_of(off, ln)
        if "frame_images" in f:  # export-real's file
            assert np.array_equal(f["frame_rows"][:], rows)
            get = lambda i0, i1: f["frame_images"][i0:i1]
        else:  # a rendered store: read row spans, keep the encoded rows
            assert f["images"].shape[1] == FRAME, "render without --resize"

            def get(i0, i1):
                r0, r1 = rows[i0], rows[i1 - 1] + 1
                return f["images"][r0:r1][rows[i0:i1] - r0]

        grids = []
        for i in range(0, len(rows), 256):
            fr = get(i, min(i + 256, len(rows))).reshape(-1, 2, 3, H, W)
            grids.append(enc.features(fr, batch=a.batch)[1])
            print(f"  {i + len(fr)} / {len(rows)}", flush=True)
    dt = _dtarget(qpos, action, off, ln)
    blk, ok = _blocks(dt, rows, off, ln, ep)
    np.savez(a.out, grid=np.concatenate(grids), ep=ep, tick=tick, q=qpos[rows].astype(np.float32),
             act=blk, act_ok=ok, ep_len=ln)
    print(a.out, len(rows), "frames", len(ln), "episodes")


# ── data ───────────────────────────────────────────────────────────────────


class Feats:
    """Encoded frames of one or more npz files and their windows."""

    def __init__(self, paths, n_episodes=0):
        g, ep, lep, tick, q, act, ok = [], [], [], [], [], [], []
        base = 0
        for p in paths:
            z = np.load(p)
            e = z["ep"]
            keep = e < n_episodes if n_episodes else np.ones(len(e), bool)
            g.append(z["grid"][keep])
            ep.append(e[keep] + base)
            lep.append(e[keep])
            tick.append(z["tick"][keep])
            q.append(z["q"][keep])
            act.append(z["act"][keep])
            ok.append(z["act_ok"][keep])
            base += int(e.max()) + 1
        self.grid = np.concatenate(g)  # (n, 2, 12, D) f16
        self.ep, self.tick = np.concatenate(ep), np.concatenate(tick)
        lep = np.concatenate(lep)
        self.q, self.act, self.ok = np.concatenate(q), np.concatenate(act), np.concatenate(ok)
        n = len(self.ep)
        # window start i: frames i..i+3 in one episode, consecutive ticks, all
        # four action blocks complete
        s = np.arange(max(n - (T - 1), 0))
        good = np.ones(len(s), bool)
        for t in range(1, T):
            good &= (self.ep[s + t] == self.ep[s]) & (self.tick[s + t] == self.tick[s] + t * FS)
        for t in range(T):
            good &= self.ok[s + t]
        self.starts = s[good]
        # by each file's own episode number, so a file keeps its split when
        # it is mixed with others (or listed twice to weight it)
        self.val = (lep[self.starts] % VAL_EVERY) == VAL_EVERY - 1

    def act_stats(self):
        a = self.act[self.ok].reshape(-1, FS, J).reshape(-1, J).astype(np.float64)
        return a.mean(0), np.maximum(a.std(0), 1e-6)


# ── model ──────────────────────────────────────────────────────────────────


def _model(D, d, layers, heads):
    import torch
    from torch import nn

    class Pred(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(D, d)
            self.out = nn.Linear(d, D)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.act = nn.Sequential(nn.Linear(FS * J, d), nn.GELU(), nn.Linear(d, d))
            self.pos = nn.Parameter(torch.randn(1, T - 1, 24, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            m = torch.arange(T - 1).repeat_interleave(24)
            self.register_buffer("mask", m[None, :] > m[:, None])  # True = blocked

        def forward(self, x, a):
            """x (B, T-1, 24, D), a (B, T-1, 30) -> frames 1..T-1 (B, T-1, 24, D)."""
            B, t = x.shape[:2]
            h = self.inp(x) + self.pos[:, :t] + self.act(a)[:, :, None]
            h = self.blocks(h.reshape(B, t * 24, d), mask=self.mask[: t * 24, : t * 24])
            return x + self.out(self.norm(h)).reshape(x.shape)

    return Pred()


def _window(torch, grid, act, starts, dev):
    """GPU tensors: grid (n, 24, D) f16 on dev, act (n, 30) normalised."""
    idx = torch.as_tensor(starts, device=dev)[:, None] + torch.arange(T, device=dev)[None]
    return grid[idx].float(), act[idx]


def cmd_train(a):
    import torch

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    fe = Feats(a.feats.split(","))
    mu, sd = fe.act_stats()
    os.makedirs(a.out, exist_ok=True)
    np.savetxt(os.path.join(a.out, "action_stats.txt"), np.stack([mu, sd], 1),
               header="per-joint target change (LeRobot units): mean std")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = fe.grid.shape[-1]
    grid = torch.from_numpy(fe.grid.reshape(len(fe.grid), 24, D)).to(dev)
    act = torch.from_numpy(((fe.act.reshape(-1, FS, J) - mu) / sd).reshape(-1, FS * J).astype(np.float32)).to(dev)
    tr, va = fe.starts[~fe.val], fe.starts[fe.val]
    print(f"frames {len(fe.ep)} | windows train {len(tr)} val {len(va)} | D {D}", flush=True)
    net = _model(D, a.d, a.layers, a.heads).to(dev)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    cfg = dict(D=D, d=a.d, layers=a.layers, heads=a.heads)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"))
    log = open(os.path.join(a.out, "log.csv"), "w")
    log.write("step,loss,copy,val_loss,val_copy\n")

    def losses(st):
        x, ac = _window(torch, grid, act, st, dev)
        with torch.autocast(dev, dtype=torch.bfloat16):
            p = net(x[:, :-1], ac[:, :-1])
        l = ((p.float() - x[:, 1:]) ** 2).mean()
        c = ((x[:, :-1] - x[:, 1:]) ** 2).mean()
        return l, c

    rng = np.random.default_rng(a.seed)
    for step in range(1, a.steps + 1):
        net.train()
        l, c = losses(tr[rng.integers(0, len(tr), a.batch)])
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == a.steps:
            net.eval()
            with torch.no_grad():
                vl, vc = zip(*[losses(va[i : i + 512]) for i in range(0, len(va), 512)])
            vl, vc = float(torch.stack(vl).mean()), float(torch.stack(vc).mean())
            log.write(f"{step},{l.item()},{c.item()},{vl},{vc}\n")
            log.flush()
            print(f"step {step} | loss {l.item():.5f} copy {c.item():.5f} | val {vl:.5f} copy {vc:.5f}"
                  f" (pred/copy {vl / vc:.3f})", flush=True)
    torch.save(net.state_dict(), os.path.join(a.out, "pred.pt"))


# ── eval ───────────────────────────────────────────────────────────────────


def _ridge_q(xs, ys, Cs=(1e1, 1e2, 1e3, 1e4, 1e5)):
    """Standardised ridge, alpha by 5-fold CV on xs; returns a predictor."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), RidgeCV(alphas=Cs)).fit(xs, ys)


def _r2(y, p):
    return 1.0 - ((y - p) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-12)


def cmd_eval(a):
    import torch

    cfg = json.load(open(os.path.join(a.model, "config.json")))
    st = np.loadtxt(os.path.join(a.model, "action_stats.txt"))
    mu, sd = st[:, 0], st[:, 1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = _model(cfg["D"], cfg["d"], cfg["layers"], cfg["heads"]).to(dev).eval()
    net.load_state_dict(torch.load(os.path.join(a.model, "pred.pt"), map_location=dev))
    sets = dict(s.split(":", 1) for s in a.set)
    res, probe_x = {}, {}
    g = torch.Generator(device=dev).manual_seed(0)
    for name, path in sets.items():
        fe = Feats(path.split(","), a.episodes)
        if a.val_only:
            fe.starts = fe.starts[fe.val]
        D = fe.grid.shape[-1]
        grid = torch.from_numpy(fe.grid.reshape(len(fe.grid), 24, D)).to(dev)
        act = torch.from_numpy(((fe.act.reshape(-1, FS, J) - mu) / sd).reshape(-1, FS * J).astype(np.float32)).to(dev)
        tot = dict(pred=0.0, copy=0.0, shuf=0.0, n=0)
        with torch.no_grad():
            for i in range(0, len(fe.starts), 256):
                s = fe.starts[i : i + 256]
                x, ac = _window(torch, grid, act, s, dev)
                with torch.autocast(dev, dtype=torch.bfloat16):
                    p = net(x[:, :-1], ac[:, :-1])
                    ps = net(x[:, :-1], ac[torch.randperm(len(s), generator=g, device=dev)][:, :-1])
                tot["pred"] += float(((p.float() - x[:, 1:]) ** 2).mean()) * len(s)
                tot["shuf"] += float(((ps.float() - x[:, 1:]) ** 2).mean()) * len(s)
                tot["copy"] += float(((x[:, :-1] - x[:, 1:]) ** 2).mean()) * len(s)
                tot["n"] += len(s)
        n = tot.pop("n")
        r = {k: v / n for k, v in tot.items()}
        r["windows"], r["episodes"] = n, int(len(set(fe.ep)))
        res[name] = r
        probe_x[name] = (fe.grid.astype(np.float32).reshape(len(fe.grid), -1), fe.q.astype(np.float64), fe.ep)
        print(f"[{name}] {r['episodes']} episodes, {n} windows | pred {r['pred']:.5f} | copy-last {r['copy']:.5f}"
              f" (pred / copy {r['pred'] / r['copy']:.3f}) | actions shuffled x{r['shuf'] / r['pred']:.3f}", flush=True)
    if "real" in res and a.sim in res:
        print(f"real / {a.sim} pred: {res['real']['pred'] / res[a.sim]['pred']:.3f} (gate: <= ~2)")
    # joint-angle probe: fit on the even episodes of the sim set, test on its
    # odd episodes and on every frame of the other sets
    xs, ys, es = probe_x[a.sim]
    fit = es % 2 == 0
    m = _ridge_q(xs[fit], ys[fit])
    fmt = lambda v: " ".join(f"{x:.3f}" for x in v)
    print(f"q probe R2, {a.sim} fit, its odd episodes: {fmt(_r2(ys[~fit], m.predict(xs[~fit])))}")
    for name, (x, y, e) in probe_x.items():
        if name != a.sim:
            print(f"q probe R2, {a.sim} fit -> {name}: {fmt(_r2(y, m.predict(x)))}")
            if name == "real":
                fr = e % 2 == 0
                mr = _ridge_q(x[fr], y[fr])
                print(f"q probe R2, real own fit (odd episodes): {fmt(_r2(y[~fr], mr.predict(x[~fr])))}")
    json.dump(res, open(os.path.join(a.model, f"eval_{a.tag}.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("demo")
    s.add_argument("outdir")
    s.add_argument("n", type=int)
    s = sub.add_parser("export-real")
    s.add_argument("store")
    s.add_argument("out")
    s = sub.add_parser("encode")
    s.add_argument("--store", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--hub", required=True)
    s.add_argument("--weights", required=True)
    s.add_argument("--batch", type=int, default=32)
    s = sub.add_parser("train")
    s.add_argument("--feats", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=20000)
    s.add_argument("--batch", type=int, default=128)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--wd", type=float, default=0.05)
    s.add_argument("--d", type=int, default=384)
    s.add_argument("--layers", type=int, default=6)
    s.add_argument("--heads", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--log-every", type=int, default=500)
    s = sub.add_parser("eval")
    s.add_argument("--model", required=True)
    s.add_argument("--set", action="append", required=True, help="name:feats.npz[,more.npz]")
    s.add_argument("--sim", default="sim_dr", help="the set the q probe is fitted on")
    s.add_argument("--episodes", type=int, default=0, help="first N episodes of each set (0 = all)")
    s.add_argument("--val-only", action="store_true",
                   help="the predictor on the validation episodes only (the probe still uses all)")
    s.add_argument("--tag", default="final")
    a = p.parse_args()
    {"split": cmd_split, "export-real": cmd_export_real, "encode": cmd_encode,
     "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
