"""Slow-feature diagnostics of the SO-101 world models (offline, no arm).

noeira-docs/SO101_LEWM_PLAN.md §3c step 1. H-JEPA (arXiv 2610.06805, App. L)
shows that when the scene changes every episode — our room randomisation —
an encoder can keep only the background: an identity predictor then scores
zero prediction loss and SIGReg, which only sees the spread of single
embeddings, cannot object ("slow-feature collapse"). Per data set, over the
first `--episodes` episodes, every frame at the window stride (FS rows):

    chi        within-episode share of the variance, E_e[tr Var_t(x | e)] /
               tr Var(x) (their Eq. 10): 0 when x is constant inside every
               episode; H-JEPA: <= 0.016 collapsed, ~0.72 Cube, 0.12 DROID
               with inverse dynamics. For the pixels themselves (the share
               the encoder COULD keep), the image latent, and for a v2 model
               the predictor's state z = image + proprio embedding and the
               proprio embedding alone
    sens       ||pred(a) − pred(a')|| / ||pred(a)|| with a' another
               episode's actions (their "action sensitivity": < 0.05
               collapsed, ~0.28 healthy), and the same change against the
               true latent step ||z_{t+1} − z_t||
    pred/copy  one-step error against the identity predictor's (their θ̂)
    look cut   the same split by whether a frame pair crosses a DR redraw:
               `tower_demo_rerender --dr` redraws the look (lights, colours,
               camera pose and fovy, background) every `LANES` = 32 rows of
               an episode, so ~15/32 of the windows hold a jump no action
               explains. Per class: the image latent's step and pred/copy
               (a set without DR has no cut: its "cut" class is the same
               rows, a control)
    q probe    ridge regression image latent -> the six joint angles, fit
               on the first set's even episodes: R² on its odd episodes, and
               on every frame of each other set (does the arm survive in the
               picture's latent, and does the readout transfer?)

    pixi run -e apple mojo run -I . examples/lewm/lewm_so101_slow_features.mojo \\
        --dump <run>/epoch_7 --stats <run>/action_stats.txt \\
        --set sim_dr:<a.rendered.h5> --set sim_plain:<b.rendered.h5> \\
        --set real:<teleop_undist.h5> [--episodes 40]

A set whose label starts with `real` is read as a teleop store. A v2
(proprio) dump is recognised by its `prop_emb` tensors, so one binary serves
both (`mojo run`: the laptop's linker rejects this driver's symbol lengths).
"""

from std.sys import argv
from std.sys.defines import get_defined_int
from std.os.path import exists
from std.math import sqrt
from std.random import seed
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.experimental.lewm.ref_trainer import LeWMRefTrainer, WMWindowLoss
from noeira.experimental.lewm.so101_trainer import LeWMProprioTrainer
from noeira.experimental.lewm.so101_data import So101WMData, ACT_IN, CAMS, JOINTS, PROP, T, FS


comptime R = 112
comptime B = get_defined_int["SO101_WM_B", 16]()
comptime TR = LeWMRefTrainer["gpu", B, CAMS * 3, R, ACT_IN]
comptime TRP = LeWMProprioTrainer["gpu", B, CAMS * 3, R, ACT_IN, PROP]
comptime FRAME_PIX = CAMS * 3 * R * R
comptime PIX = T * FRAME_PIX
comptime ACT = T * ACT_IN
comptime PRO = T * PROP
comptime EMB = 192
comptime PD = EMB + 1
"""Probe features: the latent and a bias."""
comptime DR_LANES = 32
"""`tower_demo_rerender`'s rows per launch (one DR draw each)."""


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


struct _VarSplit(Movable):
    """Streaming within / total variance of D-dim vectors tagged by episode."""

    var d: Int
    var sums: List[Float64]
    """(episodes, d)"""
    var counts: List[Int]
    var sq: Float64

    def __init__(out self, d: Int, episodes: Int):
        self.d = d
        self.sums = List[Float64](length=d * episodes, fill=0.0)
        self.counts = List[Int](length=episodes, fill=0)
        self.sq = 0.0

    def add[dt: DType](mut self, e: Int, x: List[Scalar[dt]], off: Int):
        for k in range(self.d):
            var v = Float64(x[off + k])
            self.sums[e * self.d + k] += v
            self.sq += v * v
        self.counts[e] += 1

    def chi(self) -> Tuple[Float64, Float64]:
        """(within-episode share, total variance trace per sample)."""
        var n = 0
        var within = self.sq
        var tot = List[Float64](length=self.d, fill=0.0)
        for e in range(len(self.counts)):
            var c = self.counts[e]
            if c == 0:
                continue
            n += c
            for k in range(self.d):
                var s = self.sums[e * self.d + k]
                within -= s * s / Float64(c)
                tot[k] += s
        var total = self.sq
        for k in range(self.d):
            total -= tot[k] * tot[k] / Float64(n)
        return (within / total, total / Float64(n))


struct _Frames(Movable):
    """Every visited frame's latents, episode and joint angles."""

    var img: List[Float32]
    var z: List[Float32]
    var q: List[Float32]
    var ep: List[Int]

    def __init__(out self):
        self.img = List[Float32]()
        self.z = List[Float32]()
        self.q = List[Float32]()
        self.ep = List[Int]()

    def n(self) -> Int:
        return len(self.ep)


def _chi_of(x: List[Float32], ep: List[Int], n_ep: Int) -> Tuple[Float64, Float64]:
    var v = _VarSplit(EMB, n_ep)
    for i in range(len(ep)):
        v.add(ep[i], x, i * EMB)
    return v.chi()


struct _Ridge(Movable):
    var xtx: List[Float64]
    var xty: List[Float64]

    def __init__(out self):
        self.xtx = List[Float64](length=PD * PD, fill=0.0)
        self.xty = List[Float64](length=PD * JOINTS, fill=0.0)

    def add(mut self, f: _Frames, i: Int):
        var x = List[Float64](length=PD, fill=1.0)
        for k in range(EMB):
            x[k] = Float64(f.img[i * EMB + k])
        for a in range(PD):
            for b in range(PD):
                self.xtx[a * PD + b] += x[a] * x[b]
            for j in range(JOINTS):
                self.xty[a * JOINTS + j] += x[a] * Float64(f.q[i * JOINTS + j])

    def solve(self, lam: Float64) raises -> List[Float64]:
        """(PD, JOINTS) weights of (XᵀX + λI) W = XᵀY, by Gauss-Jordan."""
        var m = self.xtx.copy()
        var w = self.xty.copy()
        for a in range(EMB):
            m[a * PD + a] += lam
        for c in range(PD):
            var piv = c
            for r in range(c + 1, PD):
                if abs(m[r * PD + c]) > abs(m[piv * PD + c]):
                    piv = r
            if abs(m[piv * PD + c]) < 1e-12:
                raise Error("_Ridge: singular system")
            if piv != c:
                for k in range(PD):
                    var t = m[c * PD + k]
                    m[c * PD + k] = m[piv * PD + k]
                    m[piv * PD + k] = t
                for j in range(JOINTS):
                    var t = w[c * JOINTS + j]
                    w[c * JOINTS + j] = w[piv * JOINTS + j]
                    w[piv * JOINTS + j] = t
            var d = m[c * PD + c]
            for r in range(PD):
                if r == c:
                    continue
                var f = m[r * PD + c] / d
                if f == 0.0:
                    continue
                for k in range(PD):
                    m[r * PD + k] -= f * m[c * PD + k]
                for j in range(JOINTS):
                    w[r * JOINTS + j] -= f * w[c * JOINTS + j]
        for c in range(PD):
            for j in range(JOINTS):
                w[c * JOINTS + j] /= m[c * PD + c]
        return w^


def _r2(f: _Frames, w: List[Float64], parity: Int) -> List[Float64]:
    """Per-joint R² of the probe on the frames whose episode has the given
    parity (−1: every frame)."""
    var n = 0
    var mean = List[Float64](length=JOINTS, fill=0.0)
    for i in range(f.n()):
        if parity >= 0 and f.ep[i] % 2 != parity:
            continue
        n += 1
        for j in range(JOINTS):
            mean[j] += Float64(f.q[i * JOINTS + j])
    for j in range(JOINTS):
        mean[j] /= Float64(max(n, 1))
    var sse = List[Float64](length=JOINTS, fill=0.0)
    var sst = List[Float64](length=JOINTS, fill=0.0)
    for i in range(f.n()):
        if parity >= 0 and f.ep[i] % 2 != parity:
            continue
        for j in range(JOINTS):
            var p = w[EMB * JOINTS + j]
            for k in range(EMB):
                p += Float64(f.img[i * EMB + k]) * w[k * JOINTS + j]
            var y = Float64(f.q[i * JOINTS + j])
            sse[j] += (y - p) ** 2
            sst[j] += (y - mean[j]) ** 2
    var out = List[Float64]()
    for j in range(JOINTS):
        out.append(1.0 - sse[j] / max(sst[j], 1e-12))
    return out^


def _fmt(xs: List[Float64]) -> String:
    var s = String("")
    for x in xs:
        s += " " + String(Float32(x))
    return s^


struct _Sens(Movable):
    var dp: Float64
    var p: Float64
    var step: Float64
    var err2: Float64
    var copy2: Float64
    var cut_n: List[Int]
    """Per class (0 inside one look, 1 across a redraw): pairs, the image
    latent's step norm, one-step error² and copy error²."""
    var cut_img: List[Float64]
    var cut_err2: List[Float64]
    var cut_copy2: List[Float64]

    def __init__(out self):
        self.dp = 0.0
        self.p = 0.0
        self.step = 0.0
        self.err2 = 0.0
        self.copy2 = 0.0
        self.cut_n = [0, 0]
        self.cut_img = [0.0, 0.0]
        self.cut_err2 = [0.0, 0.0]
        self.cut_copy2 = [0.0, 0.0]


def _visit[TRT: WMWindowLoss](
    mut tr: TRT, data: So101WMData[R], episodes: Int, mut fr: _Frames, mut se: _Sens,
) raises -> Tuple[Float64, Float64]:
    """Non-overlapping windows covering each of the first `episodes`
    episodes: their latents into `fr`, the action-sensitivity sums into
    `se`; returns the pixels' variance split."""
    var n_ep = min(episodes, len(data.ep_off))
    var starts = List[Int]()
    var eps = List[Int]()
    for e in range(n_ep):
        var r = data.ep_off[e]
        while r + T * FS <= data.ep_off[e] + data.ep_len[e]:
            starts.append(r)
            eps.append(e)
            r += T * FS
    var nw = len(starts)
    # another episode's window for each window's actions
    var other = List[Int]()
    for i in range(nw):
        var k = (i + nw // 2) % nw
        while eps[k] == eps[i]:
            k = (k + 1) % nw
        other.append(starts[k])
    var pix = List[Scalar[DT]](length=B * PIX, fill=Scalar[DT](0))
    var junk = List[Scalar[DT]](length=B * PIX, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=B * ACT, fill=Scalar[DT](0))
    var act2 = List[Scalar[DT]](length=B * ACT, fill=Scalar[DT](0))
    var pro = List[Scalar[DT]](length=B * PRO, fill=Scalar[DT](0))
    var px = _VarSplit(FRAME_PIX, n_ep)
    var i0 = 0
    while i0 < nw:
        var bs = List[Int]()
        var bo = List[Int]()
        for b in range(B):
            var i = min(i0 + b, nw - 1)
            bs.append(starts[i])
            bo.append(other[i])
        data.fill(bs, pix, act)
        data.fill(bo, junk, act2)
        data.fill_prop(bs, pro)
        var la = tr.window_latents(pix, act, pro)
        var lb = tr.window_latents(pix, act2, pro)
        var valid = min(B, nw - i0)
        for b in range(valid):
            var e = eps[i0 + b]
            for t in range(T):
                var row = bs[b] + t * FS
                px.add(e, pix, (b * T + t) * FRAME_PIX)
                for k in range(EMB):
                    fr.img.append(la.img[(b * T + t) * EMB + k])
                    fr.z.append(la.z[(b * T + t) * EMB + k])
                for j in range(JOINTS):
                    fr.q.append(data.prop[row * PROP + j])
                fr.ep.append(e)
            for t in range(T - 1):
                var rel = bs[b] + t * FS - data.ep_off[e]
                var cut = 1 if rel // DR_LANES != (rel + FS) // DR_LANES else 0
                var di = 0.0
                for k in range(EMB):
                    di += (Float64(la.img[(b * T + t + 1) * EMB + k]) - Float64(la.img[(b * T + t) * EMB + k])) ** 2
                var dp = 0.0
                var pp = 0.0
                var st = 0.0
                var er = 0.0
                for k in range(EMB):
                    var pa = Float64(la.pred[(b * (T - 1) + t) * EMB + k])
                    var pb = Float64(lb.pred[(b * (T - 1) + t) * EMB + k])
                    var z0 = Float64(la.z[(b * T + t) * EMB + k])
                    var z1 = Float64(la.z[(b * T + t + 1) * EMB + k])
                    dp += (pa - pb) ** 2
                    pp += pa * pa
                    st += (z1 - z0) ** 2
                    er += (pa - z1) ** 2
                se.dp += sqrt(dp)
                se.p += sqrt(pp)
                se.step += sqrt(st)
                se.err2 += er
                se.copy2 += st
                se.cut_n[cut] += 1
                se.cut_img[cut] += sqrt(di)
                se.cut_err2[cut] += er
                se.cut_copy2[cut] += st
        i0 += B
    return px.chi()


def _run[TRT: WMWindowLoss](
    mut tr: TRT, labels: List[String], paths: List[String],
    stats: String, pstats: String, episodes: Int, proprio: Bool,
) raises:
    var w = List[Float64]()
    for s in range(len(labels)):
        var ps = List[String]()
        for p in paths[s].split(","):
            ps.append(String(p))
        var data = So101WMData[R](ps, teleop=labels[s].startswith("real"))
        data.load_action_stats(stats)
        if proprio:
            data.load_prop_stats(pstats)
        var n_ep = min(episodes, len(data.ep_off))
        var fr = _Frames()
        var se = _Sens()
        var pxc = _visit(tr, data, episodes, fr, se)
        print("[" + labels[s] + "]", n_ep, "episodes,", fr.n(), "frames")
        var ci = _chi_of(fr.img, fr.ep, n_ep)
        var line = String("  chi: pixels ") + String(Float32(pxc[0]))
        line += " | image latent " + String(Float32(ci[0])) + " (tr var " + String(Float32(ci[1])) + ")"
        if proprio:
            var cz = _chi_of(fr.z, fr.ep, n_ep)
            var pe = List[Float32](length=len(fr.z), fill=0.0)
            for i in range(len(fr.z)):
                pe[i] = fr.z[i] - fr.img[i]
            var cp = _chi_of(pe, fr.ep, n_ep)
            line += " | z " + String(Float32(cz[0])) + " (tr var " + String(Float32(cz[1])) + ")"
            line += " | proprio emb " + String(Float32(cp[0])) + " (tr var " + String(Float32(cp[1])) + ")"
        print(line)
        print("  action sensitivity: /|pred|", Float32(se.dp / se.p),
              "| /|true step|", Float32(se.dp / se.step),
              "| pred/copy", Float32(se.err2 / se.copy2))
        var cl = String("  look cut (pairs, |image step|, pred/copy):")
        var names: List[String] = [" inside", " | across"]
        for k in range(2):
            var n = Float64(max(se.cut_n[k], 1))
            cl += names[k] + " " + String(se.cut_n[k]) + ", " + String(Float32(se.cut_img[k] / n))
            cl += ", " + String(Float32(se.cut_err2[k] / max(se.cut_copy2[k], 1e-12)))
        print(cl)
        if s == 0:
            var rg = _Ridge()
            for i in range(fr.n()):
                if fr.ep[i] % 2 == 0:
                    rg.add(fr, i)
            w = rg.solve(1.0)
            print("  q probe R2 (held-out odd episodes):" + _fmt(_r2(fr, w, 1)))
        else:
            var rg = _Ridge()
            for i in range(fr.n()):
                if fr.ep[i] % 2 == 0:
                    rg.add(fr, i)
            var ws = rg.solve(1.0)
            print("  q probe R2, own fit (odd episodes):" + _fmt(_r2(fr, ws, 1)))
            print("  q probe R2, " + labels[0] + "'s fit (all frames):" + _fmt(_r2(fr, w, -1)))


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var dump = _arg(args, "--dump", "")
    var stats = _arg(args, "--stats", "")
    var episodes = Int(_arg(args, "--episodes", "40"))
    var labels = List[String]()
    var paths = List[String]()
    for i in range(len(args) - 1):
        if args[i] == "--set":
            var kv = args[i + 1].split(":")
            labels.append(String(kv[0]))
            paths.append(String(kv[1]))
    if dump.byte_length() == 0 or stats.byte_length() == 0 or len(labels) == 0:
        raise Error("--dump, --stats and at least one --set label:path are required")
    var pstats = _arg(args, "--pstats", String(stats[byte = 0 : stats.rfind("/") + 1]) + "prop_stats.txt")
    seed(7)
    var c = DeviceContext()
    var ctx = Optional(c)
    if exists(dump + "/ours.prop_emb.0.0.weight.bin"):
        var tr = TRP(ctx, dropout=False)
        var n = tr.load(dump)
        tr.set_eval()
        print("slow-feature check | v2 (proprio) model", dump, "(", n, "tensors ) | batch", B)
        _run(tr, labels, paths, stats, pstats, episodes, True)
    else:
        var tr = TR(ctx, dropout=False)
        var n = tr.load(dump)
        tr.set_eval()
        print("slow-feature check | v1 model", dump, "(", n, "tensors ) | batch", B)
        _run(tr, labels, paths, stats, pstats, episodes, False)
