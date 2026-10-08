"""S2 (1) — does the encoder put a real frame where it puts its sim twin?

noeira-docs/SO101_LEWM_PLAN.md S2. Real teleop frames (the printed set,
undistorted, `So101WMData(teleop=True)`) against their sim twins
(`lewm_so101_twin.mojo` -> `tower_demo_rerender --resize 112`: the same arm
pose, the props where the real overhead frame shows them), both through the
SAME `AreaResize` and the trained encoder (`emb.0.*` of a
`lewm_so101_train.mojo` dump):

    d_pair    mean ‖z_real(t) − z_sim(t)‖      the domain gap at one pose
    d_step    mean ‖z_sim(t) − z_sim(t + 5)‖    one model step of motion
    d_rand    mean ‖z_sim(i) − z_sim(j)‖       two random frames
    top-1     the real frame's nearest sim frame (over EVERY twin frame) is
              its own twin: same episode, within 5 ticks; chance ~ 3 / N at stride 5

d_pair ≪ d_step and a high top-1 = the encoder sees the real scene as the
sim scene; d_pair ~ d_rand = it does not. The real latents' own d_step /
d_rand are printed too: a real cluster much tighter than the sim one makes a
low real PREDICTION loss (`lewm_so101_real_check`) meaningless.

The REAL cost landscape: along each whole real episode (every `--stride`
ticks), the latent distance to the episode's LAST frame against the ticks
still to go — Spearman per episode, averaged. A domain OFFSET (real cloud
displaced from the sim one) leaves it intact; a critic comparing predictions
with a real goal frame needs exactly this to be high.

    build/so101_align --dump <run>/epoch_7 --real <teleop store> \\
        --twin twin.rendered.h5 --map twin.demo.map.txt
"""

from std.sys import argv
from std.math import sqrt
from std.random import seed, random_ui64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.experimental.lewm.ref_model import LeWMEncoderRef
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.so101_data import So101WMData, CAMS


comptime R = 112
comptime EMB = 192
comptime NB = 16
comptime Enc = LeWMEncoderRef[CAMS * 3, R, 14, 192, 3, 12, EMB, 2048]
comptime FR = CAMS * 3 * R * R


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def _encode(
    mut enc: Enc, data: So101WMData[R], rows: List[Int], ctx: Optional[DeviceContext]
) raises -> List[List[Float64]]:
    """The listed rows -> latents, NB frames per forward (eval-mode BN)."""
    var mean: List[Float64] = [0.485, 0.456, 0.406]
    var std: List[Float64] = [0.229, 0.224, 0.225]
    enc.set_attr["training"](Scalar[DT](0.0))
    var out = List[List[Float64]]()
    var i = 0
    while i < len(rows):
        var x = Tensor.alloc(NB * FR)
        for b in range(NB):
            var r = rows[min(i + b, len(rows) - 1)]
            for ch in range(CAMS * 3):
                var c = ch % 3
                for p in range(R * R):
                    var v = Float64(Int(data.images[r * FR + ch * R * R + p])) / 255.0
                    x.data[b * FR + ch * R * R + p] = Scalar[DT]((v - mean[c]) / std[c])
        x.upload(ctx.value())
        var y = Tensor.alloc(NB * EMB)
        enc.forward["gpu", NB](TensorRefs[1](x), y, ctx)
        ctx.value().synchronize()
        y.download(ctx.value())
        for b in range(NB):
            if i + b >= len(rows):
                break
            var z = List[Float64](capacity=EMB)
            for d in range(EMB):
                z.append(Float64(y.data[b * EMB + d]))
            out.append(z^)
        i += NB
    return out^


def _ranks(v: List[Float64]) -> List[Float64]:
    var r = List[Float64](length=len(v), fill=0.0)
    for i in range(len(v)):
        var below = 0
        for j in range(len(v)):
            if v[j] < v[i] or (v[j] == v[i] and j < i):
                below += 1
        r[i] = Float64(below)
    return r^


def _spearman(a: List[Float64], b: List[Float64]) -> Float64:
    var ra = _ranks(a)
    var rb = _ranks(b)
    var n = Float64(len(a))
    var ma = 0.0
    var mb = 0.0
    for i in range(len(a)):
        ma += ra[i]
        mb += rb[i]
    ma /= n
    mb /= n
    var sab = 0.0
    var saa = 0.0
    var sbb = 0.0
    for i in range(len(a)):
        sab += (ra[i] - ma) * (rb[i] - mb)
        saa += (ra[i] - ma) ** 2
        sbb += (rb[i] - mb) ** 2
    return sab / max((saa * sbb) ** 0.5, 1e-12)


def _dist(a: List[Float64], b: List[Float64]) -> Float64:
    var s = 0.0
    for d in range(len(a)):
        s += (a[d] - b[d]) ** 2
    return sqrt(s)


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var dump = _arg(args, "--dump", "")
    var real = _arg(args, "--real", "")
    var twin = _arg(args, "--twin", "")
    var mapf = _arg(args, "--map", "")
    var stride = Int(_arg(args, "--stride", "5"))
    seed(11)
    var c = DeviceContext()
    var ctx = Optional(c)
    var enc = Enc.make["gpu", Kaiming](ctx)
    var n_loaded = load_ref["gpu"](enc, dump, String("emb.0."), ctx)
    var rp = List[String]()
    rp.append(real)
    var rd = So101WMData[R](rp, teleop=True)
    var tp = List[String]()
    tp.append(twin)
    var td = So101WMData[R](tp)
    # pairs (real row, twin row), every `stride`-th tick of each mapped episode
    var real_rows = List[Int]()
    var twin_rows = List[Int]()
    var ep_of = List[Int]()
    var t_of = List[Int]()
    var tw_base = 0
    var k = 0
    with open(mapf, "r") as f:
        for ln in f.read().split("\n"):
            var s = String(ln.strip())
            if s.byte_length() == 0 or s.startswith("#"):
                continue
            var w = s.split(" ")
            var off = Int(String(w[2]))
            var n = Int(String(w[3]))
            var t = 0
            while t < n:
                real_rows.append(off + t)
                twin_rows.append(tw_base + t)
                ep_of.append(k)
                t_of.append(t)
                t += stride
            tw_base += n
            k += 1
    print("S2 alignment | encoder", dump, "(", n_loaded, "tensors ) |", k, "episodes,",
          len(real_rows), "pairs (every", stride, "ticks)")
    var zr = _encode(enc, rd, real_rows, ctx)
    var zs = _encode(enc, td, twin_rows, ctx)
    var n = len(zr)
    var d_pair = 0.0
    var d_step = 0.0
    var d_step_r = 0.0
    var n_step = 0
    for i in range(n):
        d_pair += _dist(zr[i], zs[i])
        if i + 1 < n and ep_of[i + 1] == ep_of[i]:
            d_step += _dist(zs[i], zs[i + 1])
            d_step_r += _dist(zr[i], zr[i + 1])
            n_step += 1
    d_pair /= Float64(n)
    d_step /= Float64(max(n_step, 1))
    d_step_r /= Float64(max(n_step, 1))
    var d_rand = 0.0
    var d_rand_r = 0.0
    for _ in range(2000):
        var i = Int(random_ui64(0, UInt64(n - 1)))
        var j = Int(random_ui64(0, UInt64(n - 1)))
        d_rand += _dist(zs[i], zs[j])
        d_rand_r += _dist(zr[i], zr[j])
    d_rand /= 2000.0
    d_rand_r /= 2000.0
    var hit = 0
    for i in range(n):
        var best = 0
        var bd = 1e30
        for j in range(n):
            var d = _dist(zr[i], zs[j])
            if d < bd:
                bd = d
                best = j
        if ep_of[best] == ep_of[i] and abs(t_of[best] - t_of[i]) <= 5:
            hit += 1
    print("  d_pair (real vs its twin)", Float32(d_pair), "| d_step (5 ticks of motion)",
          Float32(d_step), "| d_rand", Float32(d_rand))
    print("  real latents among themselves: d_step", Float32(d_step_r), "| d_rand", Float32(d_rand_r),
          "(a collapsed real cluster makes real prediction trivially easy)")
    print("  d_pair / d_step", Float32(d_pair / d_step), "| d_pair / d_rand", Float32(d_pair / d_rand))
    # the real cost landscape along whole real episodes
    var rho_sum = 0.0
    var n_ep = 0
    var rows_all = List[Int]()
    var ep_start = List[Int]()
    var real_starts = List[Int]()
    var real_lens = List[Int]()
    var st_off = 0
    # episode bounds of the real store, from the window starts' gaps is not
    # enough: re-read them from the twin map's real offsets and the next one
    with open(mapf, "r") as f:
        for ln in f.read().split("\n"):
            var s = String(ln.strip())
            if s.byte_length() == 0 or s.startswith("#"):
                continue
            var w = s.split(" ")
            real_starts.append(Int(String(w[2])))
    for e in range(len(real_starts)):
        var a = real_starts[e]
        var b = real_starts[e + 1] if e + 1 < len(real_starts) else rd.n_rows
        ep_start.append(len(rows_all))
        var t = a
        while t < b:
            rows_all.append(t)
            t += stride
        real_lens.append(len(rows_all) - ep_start[e])
    var zall = _encode(enc, rd, rows_all, ctx)
    for e in range(len(real_starts)):
        var s0 = ep_start[e]
        var m = real_lens[e]
        if m < 4:
            continue
        var cost = List[Float64]()
        var togo = List[Float64]()
        for i in range(m):
            cost.append(_dist(zall[s0 + i], zall[s0 + m - 1]))
            togo.append(Float64(m - 1 - i))
        rho_sum += _spearman(cost, togo)
        n_ep += 1
    print("  REAL cost landscape: Spearman(dist to the episode's last frame, ticks to go)",
          Float32(rho_sum / Float64(max(n_ep, 1))), "over", n_ep, "episodes (1 = monotone)")
    print("  top-1 real -> own twin (same episode, within 5 ticks):", hit, "/", n,
          "=", Float32(100.0 * Float64(hit) / Float64(n)), "% | chance ~",
          Float32(100.0 * 3.0 / Float64(n)), "%")
