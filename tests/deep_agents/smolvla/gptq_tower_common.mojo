"""Sequential GPTQ calibration of a 2-layer SmolVLA text tower, for
`test_gptq_tower[_gpu].mojo`.

    [1] the taps: layer 0's attention Hessian against Σ n(x) n(x)ᵀ computed
        here from the inputs (n = the layer's RMSNorm), and every Hessian
        symmetric, of the right width, over N_CAL · P rows
    [2] every projection's GPTQ objective below its RTN objective
    [3] on held-out inputs, the GPTQ tower's cache K / V and final norm are
        closer to the fp32 tower's than the RTN tower's are
    [4] RTN calibration's blocks are `rtn_q4_0_blocks` of the original
        weights bit for bit (the write-back does not touch what it reads)
    [5] the blocks file: `save_q4_blocks`, then an int4 tower loaded with
        `load_q4` from it (and a base file under the same names) runs the
        same function as the calibrated fp32 tower holding deq(Ŵ) (1e-5 on
        the CPU and Metal; MAX's int4 GEMM's 3e-2 band on NVIDIA)

The inputs' channels are scaled over 100x, so H is far from isotropic and
GPTQ has something to exploit, as with a transformer's outlier channels.
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.quant.q4_0 import rtn_q4_0_blocks
from noeira.nn.quant.qlinear import Q4_INT4_GEMM
from noeira.io.safetensors import SafeTensors, SafeTensorsWriter
from noeira.deep_agents.smolvla.text import SmolVLMTextLayers, SMOLLM_THETA
from noeira.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from noeira.deep_agents.smolvla.fused import SmolVLAPrefill
from noeira.deep_agents.smolvla.attn_mask import att_2d_mask_square, smolvla_ar
from noeira.deep_agents.smolvla.names import text_name_map
from noeira.deep_agents.smolvla.layer_weights import (
    HESS_ATTN, HESS_O, HESS_MLP, HESS_DOWN, SMOLLM_RMS_EPS, q4_proj_name,
)
from noeira.deep_agents.smolvla.gptq_calib import (
    calibrate_tower_q4, save_q4_blocks, Q4_GPTQ, Q4_RTN,
)

comptime P = 6
comptime S = 3
comptime B = 1
comptime L = 2
comptime W = 64
comptime FF = 96
comptime HEADS = 2
comptime NKV = 1
comptime HD = 32
comptime KVW = NKV * HD
comptime XN = B * P * W
comptime CACHE_N = L * B * P * KVW
comptime N_CAL = 512
comptime N_VAL = 8

comptime Tower = SmolVLMTextLayers[L, W, FF, KVW]
comptime TowerQ = SmolVLMTextLayers[L, W, FF, KVW, 0, True]
comptime Cache = SmolVLAKVCache[L, P, S, NKV, HD, B]
comptime Pre = SmolVLAPrefill[P, S, B, L, W, FF, HEADS, NKV, HD, SMOLLM_THETA]
comptime PreQ = SmolVLAPrefill[
    P, S, B, L, W, FF, HEADS, NKV, HD, SMOLLM_THETA, False, 0, True
]


def _seed_weights[target: StaticString](mut t: Tower, ctx: Optional[DeviceContext]) raises:
    """Seeded weights per projection and zero biases (SmolLM2 has none)."""
    for i in range(L):
        for which in range(7):
            seed(1000 * (i + 1) + which)
            var dims = t.layers[i].proj_dims(which)
            var w = List[Float32](capacity=dims[0] * dims[1])
            for _ in range(dims[0] * dims[1]):
                w.append(Float32((random_float64() - 0.5) * 0.2))
            t.layers[i].set_base_weight[target](which, w, ctx)
        _zero_bias[target](t.layers[i].q.bias.val, ctx)
        _zero_bias[target](t.layers[i].k.bias.val, ctx)
        _zero_bias[target](t.layers[i].v.bias.val, ctx)
        _zero_bias[target](t.layers[i].o.bias.val, ctx)
        _zero_bias[target](t.layers[i].mlp.gate.bias.val, ctx)
        _zero_bias[target](t.layers[i].mlp.up.bias.val, ctx)
        _zero_bias[target](t.layers[i].mlp.down.bias.val, ctx)


def _zero_bias[target: StaticString](mut b: Tensor, ctx: Optional[DeviceContext]) raises:
    for i in range(len(b.data)):
        b.data[i] = 0
    comptime if target == "gpu":
        b.upload_resident(ctx.value())
    b.version += 1


def _inputs(n: Int, salt: Int) -> List[List[Float32]]:
    seed(77)
    var scale = List[Float64]()
    for c in range(W):
        scale.append(0.1 * (100.0 ** ((Float64(c * 37 % W)) / Float64(W - 1))))
    seed(salt)
    var xs = List[List[Float32]]()
    for _ in range(n):
        var x = List[Float32](capacity=XN)
        for t in range(B * P):
            for c in range(W):
                _ = t
                x.append(Float32((random_float64() - 0.5) * scale[c]))
        xs.append(x^)
    return xs^


def _host[target: StaticString](mut t: Tensor, n: Int, ctx: Optional[DeviceContext]) raises -> List[Float64]:
    comptime if target == "gpu":
        t.download(ctx.value())
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(Float64(t.data[i]))
    return out^


def _run[
    target: StaticString
](
    mut pre: Pre, mut tower: Tower, mut cache: Cache, ref x: List[Float32],
    ctx: Optional[DeviceContext],
) raises -> List[Float64]:
    """K, V and the final norm, concatenated."""
    var xt = Tensor.alloc(XN)
    var out = Tensor.alloc(XN)
    for i in range(XN):
        xt.data[i] = Scalar[DT](x[i])
    comptime if target == "gpu":
        xt.upload(ctx.value())
        out = Tensor.alloc_gpu(ctx.value(), XN)
    cache.reset()
    pre.run[target](tower, cache, xt, out, ctx)
    var r = _host[target](cache.k, CACHE_N, ctx)
    r.extend(_host[target](cache.v, CACHE_N, ctx))
    r.extend(_host[target](out, XN, ctx))
    return r^


def _run_q[
    target: StaticString
](
    mut pre: PreQ, mut tower: TowerQ, mut cache: Cache, ref x: List[Float32],
    ctx: Optional[DeviceContext],
) raises -> List[Float64]:
    var xt = Tensor.alloc(XN)
    var out = Tensor.alloc(XN)
    for i in range(XN):
        xt.data[i] = Scalar[DT](x[i])
    comptime if target == "gpu":
        xt.upload(ctx.value())
        out = Tensor.alloc_gpu(ctx.value(), XN)
    cache.reset()
    pre.run[target](tower, cache, xt, out, ctx)
    var r = _host[target](cache.k, CACHE_N, ctx)
    r.extend(_host[target](cache.v, CACHE_N, ctx))
    r.extend(_host[target](out, XN, ctx))
    return r^


def _sqdist(ref a: List[Float64], ref b: List[Float64]) -> Float64:
    var s = Float64(0)
    for i in range(len(a)):
        s += (a[i] - b[i]) * (a[i] - b[i])
    return s


def _maxrel(ref a: List[Float64], ref b: List[Float64]) -> Float64:
    var d = Float64(0)
    var m = Float64(0)
    for i in range(len(b)):
        d = max(d, abs(a[i] - b[i]))
        m = max(m, abs(b[i]))
    return d / max(m, 1e-30)


def run_gptq_tower[target: StaticString](ctx: Optional[DeviceContext]) raises:
    var mask = att_2d_mask_square(smolvla_ar(3, 2, 1, 0))
    var tf = Tower.make[target, Deterministic](ctx)
    var tg = Tower.make[target, Deterministic](ctx)
    var tr = Tower.make[target, Deterministic](ctx)
    _seed_weights[target](tf, ctx)
    _seed_weights[target](tg, ctx)
    _seed_weights[target](tr, ctx)
    var pre = Pre.make[target](mask, ctx)
    var cache = Cache.make[target](ctx)
    var xs = _inputs(N_CAL, 5)
    var xv = _inputs(N_VAL, 6)

    print("[1] the taps")
    tf.layers[0].hessian_start()
    for j in range(N_CAL):
        _ = _run[target](pre, tf, cache, xs[j], ctx)
    var hs = tf.layers[0].hessian_take()
    # Σ n(x) n(x)ᵀ from the inputs: layer 0's attention input is the
    # input_layernorm of the prefill's input.
    var gam = List[Float64]()
    for c in range(W):
        gam.append(Float64(tf.layers[0].input_layernorm.gamma.val.data[c]))
    var want = List[Float64](length=W * W, fill=0)
    for j in range(N_CAL):
        for t in range(B * P):
            var ms = Float64(0)
            for c in range(W):
                var v = Float64(xs[j][t * W + c])
                ms += v * v
            var inv = 1.0 / sqrt(ms / Float64(W) + Float64(SMOLLM_RMS_EPS))
            var nx = List[Float64](capacity=W)
            for c in range(W):
                nx.append(Float64(xs[j][t * W + c]) * inv * gam[c])
            for a in range(W):
                for b in range(W):
                    want[a * W + b] += nx[a] * nx[b]
    var e1 = _maxrel(hs[HESS_ATTN].h, want)
    var widths_ok = hs[HESS_ATTN].n == W and hs[HESS_O].n == HEADS * HD and hs[HESS_MLP].n == W and hs[HESS_DOWN].n == FF
    var rows_ok = True
    var sym_ok = True
    for s in range(4):
        rows_ok = rows_ok and hs[s].rows == N_CAL * B * P
        var n = hs[s].n
        for a in range(n):
            sym_ok = sym_ok and hs[s].h[a * n + a] >= 0
            for b in range(a):
                sym_ok = sym_ok and hs[s].h[a * n + b] == hs[s].h[b * n + a]
    print("    layer 0 attention H vs sum n(x) n(x)^T: max rel diff", e1,
          "| widths", widths_ok, "| rows", hs[0].rows, "| symmetric, diag >= 0", sym_ok)
    assert_true(e1 < 1e-5, "the attention tap is not the normed input")
    assert_true(widths_ok and rows_ok and sym_ok, "a Hessian has the wrong width, rows or symmetry")
    assert_true(len(tf.layers[0].hess) == 0, "hessian_take left the taps on")

    print("[2] GPTQ vs RTN, per projection, on its Hessian")
    var bg = List[List[UInt8]]()
    var stg = calibrate_tower_q4[target](pre, tg, cache, xs, Q4_GPTQ, bg, ctx)
    var all_better = True
    for st in stg:
        print("    layer", st.layer, q4_proj_name(st.which), " GPTQ / RTN", st.obj_gptq / st.obj_rtn,
              " (rows", st.rows, ", dead", st.dead, ")")
        all_better = all_better and st.obj_gptq < st.obj_rtn
    assert_true(len(stg) == 7 * L and len(bg) == 7 * L, "not every linear was calibrated")
    assert_true(all_better, "GPTQ's objective is not below RTN's everywhere")

    var br = List[List[UInt8]]()
    _ = calibrate_tower_q4[target](pre, tr, cache, xs, Q4_RTN, br, ctx)

    print("[3] held-out inputs: distance to the fp32 tower (K, V, final norm)")
    var dg = Float64(0)
    var dr = Float64(0)
    for j in range(N_VAL):
        var f = _run[target](pre, tf, cache, xv[j], ctx)
        dg += _sqdist(_run[target](pre, tg, cache, xv[j], ctx), f)
        dr += _sqdist(_run[target](pre, tr, cache, xv[j], ctx), f)
    var cg = Float64(0)
    var cr = Float64(0)
    for j in range(N_VAL):
        var f = _run[target](pre, tf, cache, xs[j], ctx)
        cg += _sqdist(_run[target](pre, tg, cache, xs[j], ctx), f)
        cr += _sqdist(_run[target](pre, tr, cache, xs[j], ctx), f)
    print("    sum of squared differences, held out: GPTQ", dg, " RTN", dr, " GPTQ / RTN", dg / dr,
          "| on", N_VAL, "calibration inputs: GPTQ / RTN", cg / cr)
    assert_true(dg < dr, "the GPTQ tower is not closer to fp32 than the RTN tower")

    print("[4] RTN calibration = rtn_q4_0_blocks of the original weights")
    var tf0 = Tower.make[target, Deterministic](ctx)
    _seed_weights[target](tf0, ctx)
    var bad = 0
    for i in range(L):
        for which in range(7):
            var dims = tf0.layers[i].proj_dims(which)
            var want_b = rtn_q4_0_blocks(tf0.layers[i].base_weight(which), dims[1], dims[0])
            ref got = br[i * 7 + which]
            for e in range(len(want_b)):
                bad += Int(got[e] != want_b[e])
    print("    bytes differing:", bad)
    assert_true(bad == 0, "RTN calibration's blocks differ from RTN")

    print("[5] the blocks file into an int4 tower")
    var names = text_name_map[W, FF, L, KVW]()
    var dims = List[Tuple[Int, Int]]()
    for which in range(7):
        dims.append(tg.layers[0].proj_dims(which))
    var bpath = String("/tmp/noeira_test_gptq_blocks.safetensors")
    var fpath = String("/tmp/noeira_test_gptq_base.safetensors")
    save_q4_blocks(bpath, names, bg, L, dims, Q4_GPTQ, String("test: N_CAL seeded inputs"))
    var wr = SafeTensorsWriter()
    for i in range(L):
        for which in range(7):
            var ours = String("layers.") + String(i) + "." + q4_proj_name(which) + ".weight"
            var wi = names.index_of_ours(ours)
            var shape: List[Int] = [dims[which][1], dims[which][0]]
            var w = tf0.layers[i].base_weight(which)
            wr.add_f32(names.theirs[wi], shape, w, len(w))
    wr.save(fpath)
    var tq = TowerQ.make[target, Deterministic](ctx)
    var claimed = tq.load_q4[target](SafeTensors(fpath), names, ctx, bpath)
    comptime if target == "gpu":
        tq.ensure_q4_scratch(ctx.value())
    var preq = PreQ.make[target](mask, ctx)
    var cq = Cache.make[target](ctx)
    comptime INT4 = target == "gpu" and Q4_INT4_GEMM
    comptime TOL = 3e-2 if INT4 else 1e-5
    var e5 = Float64(0)
    for j in range(N_VAL):
        var a = _run_q[target](preq, tq, cq, xv[j], ctx)
        var b = _run[target](pre, tg, cache, xv[j], ctx)
        e5 = max(e5, _maxrel(a, b))
    print("    claimed", claimed, "map entries | int4 tower vs calibrated fp32 tower: max rel diff", e5,
          "(band", TOL, ", int4 GEMM", INT4, ")")
    assert_true(claimed == 2 * 7 * L, "load_q4 did not claim every linear")
    assert_true(e5 < TOL, "the int4 tower loaded from the blocks file differs")
