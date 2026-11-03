"""SuperMimic M4: the world model's state spec and its k-step loss + gradient.

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/wm_oracle.py --out /tmp/sm_wm
    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_wm.mojo /tmp/sm_wm

The oracle is the reproduction's `StateSpec` and `WorldModel.window_loss`
run verbatim in torch float64 (`tools/supermimic/wm_oracle.py`).

WHAT EACH SECTION CAN CATCH
===========================
[1] float64 `canonical`, `integrate`, `error_vector` against the oracle on
    48 lanes (a third inside the small-angle series), 1e-12; their VJPs
    against autograd, 1e-10 relative.
[2] the same VJPs against central differences of our own forwards (no
    torch), 16 directions each, 1e-6 relative.
[3] the k = 8 window loss in float32 (our `nn`) from the oracle's weights
    and batch: loss, state loss, BCE and both open-loop errors, 1e-5
    relative (plan M4's gate).
[4] its gradient, every parameter of both networks: cosine ≥ 0.9999 (plan
    M4) and max error ≤ 1e-4 of the largest component, per layer.
[5] `clip_grad_norm(1.0)` + one `Adam(2e-4)` step against torch's: the
    pre-clip norm (1e-5); stepped weights whose clipped gradient is > 1e-6,
    within two float32 ulps of the weight (the rest, near Adam's eps, only bounded by 10 % of lr).
[6] the zero-initialised dynamics: `make` gives delta = 0, so one step is
    exactly constant-velocity integration.

⚠ COSINE ALONE IS NOT A GRADIENT GATE HERE. Cutting the BCE's path into the
dynamics (planted) leaves the cosine at 0.99999999, above plan M4's 0.9999:
the BCE gradient is small next to the state loss's. The max-error bound in
[4] (1e-4 of the largest component) and the Adam step in [5] catch it.
Two other planted VJP errors (the canonical sign flip, a lost dt) fail [1].
"""

from std.math import sqrt
from std.random import random_float64, seed
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.core.bytes import string_from_bytes
from noeira.deep_agents.supermimic.g1_fk import FP
from noeira.deep_agents.supermimic.state_spec import (
    SM_DELTA_DIM, SM_ERR_DIM, SM_FEAT_DIM, SM_WM_DT, sm_canonical_lane, sm_canonical_vjp_lane,
    sm_error_lane, sm_error_vjp_lane, sm_integrate_lane, sm_integrate_vjp_lane,
)
from noeira.deep_agents.supermimic.world_model import (
    SMWindowBatch, SMWorldModel, SM_WM_IN, sm_wm_window_loss,
)
from noeira.io.fileio import read_file_bytes
from noeira.io.npy import load_npy_2d_f64
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm


comptime H = 32
comptime TH = 16
comptime K = 8
comptime B = 16


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


@always_inline
def _p(mut l: List[Float64], off: Int = 0) -> FP[DType.float64]:
    return l.unsafe_ptr().unsafe_offset(off).as_unsafe_any_origin()


def _rel(got: Float64, want: Float64) -> Float64:
    return abs(got - want) / max(1.0, abs(want))


struct _Load(ParamVisitor):
    """Write `vals` into the params in walk order, checking names and sizes."""
    var vals: List[Float64]
    var names: List[String]
    var sizes: List[Int]
    var at: Int
    var k: Int
    var bad: Int

    def __init__(out self, var vals: List[Float64], var names: List[String], var sizes: List[Int]):
        self.vals = vals^
        self.names = names^
        self.sizes = sizes^
        self.at = 0
        self.k = 0
        self.bad = 0

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^
        self.names = move.names^
        self.sizes = move.sizes^
        self.at = move.at
        self.k = move.k
        self.bad = move.bad

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        if self.k >= len(self.names) or self.names[self.k] != name or self.sizes[self.k] != N:
            print("    param", self.k, "is", name, N, "; the oracle has",
                  self.names[self.k] if self.k < len(self.names) else String("nothing"))
            self.bad += 1
        for i in range(N):
            param.data[i] = Scalar[DT](self.vals[self.at + i])
        self.at += N
        self.k += 1


struct _Collect(ParamVisitor):
    var out: List[Float64]
    var grads: Bool

    def __init__(out self, grads: Bool):
        self.out = List[Float64]()
        self.grads = grads

    def __init__(out self, *, deinit move: Self):
        self.out = move.out^
        self.grads = move.grads

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            self.out.append(Float64(grad.data[i]) if self.grads else Float64(param.data[i]))


def _cos(a: List[Float64], b: List[Float64], lo: Int, hi: Int) -> Tuple[Float64, Float64]:
    """(cosine, max |a − b| / max |b|) over [lo, hi)."""
    var ab = 0.0
    var aa = 0.0
    var bb = 0.0
    var mx = 0.0
    var md = 0.0
    for i in range(lo, hi):
        ab += a[i] * b[i]
        aa += a[i] * a[i]
        bb += b[i] * b[i]
        mx = max(mx, abs(b[i]))
        md = max(md, abs(a[i] - b[i]))
    return (ab / sqrt(aa * bb), md / max(mx, 1e-30))


def _lane_err(got: List[Float64], want: List[Float64]) -> Float64:
    var e = 0.0
    for i in range(len(got)):
        e = max(e, _rel(got[i], want[i]))
    return e


def main() raises:
    var av = argv()
    var dump = String(av[1]) if len(av) > 1 else String("/tmp/sm_wm")
    var t = Tally()

    # ── [1] state spec vs the oracle ──
    print("\n[1] state spec (float64) against the reproduction's StateSpec")
    var s = load_npy_2d_f64(dump + "/a_s.npy", 71)
    var n = s.rows
    var dlt = load_npy_2d_f64(dump + "/a_delta.npy", SM_DELTA_DIM).values.copy()
    var st = load_npy_2d_f64(dump + "/a_s_true.npy", 71).values.copy()
    var w_feat = load_npy_2d_f64(dump + "/a_feat.npy", SM_FEAT_DIM).values.copy()
    var w_next = load_npy_2d_f64(dump + "/a_next.npy", 71).values.copy()
    var w_err = load_npy_2d_f64(dump + "/a_err.npy", SM_ERR_DIM).values.copy()
    var g_feat = load_npy_2d_f64(dump + "/a_g_feat.npy", SM_FEAT_DIM).values.copy()
    var g_next = load_npy_2d_f64(dump + "/a_g_next.npy", 71).values.copy()
    var g_err = load_npy_2d_f64(dump + "/a_g_err.npy", SM_ERR_DIM).values.copy()
    var w_gsc = load_npy_2d_f64(dump + "/a_gs_canon.npy", 71).values.copy()
    var w_gsi = load_npy_2d_f64(dump + "/a_gs_integ.npy", 71).values.copy()
    var w_gdi = load_npy_2d_f64(dump + "/a_gd_integ.npy", SM_DELTA_DIM).values.copy()
    var w_gse = load_npy_2d_f64(dump + "/a_gs_err.npy", 71).values.copy()
    var sv = s.values.copy()
    var feat = List[Float64](length=n * SM_FEAT_DIM, fill=0.0)
    var nxt = List[Float64](length=n * 71, fill=0.0)
    var err = List[Float64](length=n * SM_ERR_DIM, fill=0.0)
    var gsc = List[Float64](length=n * 71, fill=0.0)
    var gsi = List[Float64](length=n * 71, fill=0.0)
    var gdi = List[Float64](length=n * SM_DELTA_DIM, fill=0.0)
    var gse = List[Float64](length=n * 71, fill=0.0)
    for b in range(n):
        sm_canonical_lane[DType.float64](_p(sv, b * 71), _p(feat, b * SM_FEAT_DIM))
        sm_integrate_lane[DType.float64](_p(sv, b * 71), _p(dlt, b * SM_DELTA_DIM), _p(nxt, b * 71))
        sm_error_lane[DType.float64](_p(sv, b * 71), _p(st, b * 71), _p(err, b * SM_ERR_DIM))
        sm_canonical_vjp_lane[DType.float64](_p(sv, b * 71), _p(g_feat, b * SM_FEAT_DIM), _p(gsc, b * 71))
        sm_integrate_vjp_lane[DType.float64](_p(sv, b * 71), _p(dlt, b * SM_DELTA_DIM), _p(g_next, b * 71), _p(gsi, b * 71), _p(gdi, b * SM_DELTA_DIM))
        sm_error_vjp_lane[DType.float64](_p(sv, b * 71), _p(st, b * 71), _p(g_err, b * SM_ERR_DIM), _p(gse, b * 71))
    t.truth(_lane_err(feat, w_feat) <= 1e-12, "canonical: max err " + String(_lane_err(feat, w_feat)))
    t.truth(_lane_err(nxt, w_next) <= 1e-12, "integrate: max err " + String(_lane_err(nxt, w_next)))
    t.truth(_lane_err(err, w_err) <= 1e-12, "error_vector: max err " + String(_lane_err(err, w_err)))
    t.truth(_lane_err(gsc, w_gsc) <= 1e-10, "canonical VJP: max rel err " + String(_lane_err(gsc, w_gsc)))
    var e_int = max(_lane_err(gsi, w_gsi), _lane_err(gdi, w_gdi))
    t.truth(e_int <= 1e-10, "integrate VJP (state and delta): max rel err " + String(e_int))
    t.truth(_lane_err(gse, w_gse) <= 1e-10, "error_vector VJP: max rel err " + String(_lane_err(gse, w_gse)))

    # ── [2] against central differences ──
    print("\n[2] the same VJPs against central differences of our forwards")
    seed(3)
    var h = 1e-6
    var e_fd = List[Float64](length=3, fill=0.0)
    for it in range(16):
        var b = (it * 5) % n
        var ds = List[Float64](length=71, fill=0.0)
        var dd = List[Float64](length=SM_DELTA_DIM, fill=0.0)
        for k in range(71):
            ds[k] = random_float64(-1.0, 1.0)
        for k in range(SM_DELTA_DIM):
            dd[k] = random_float64(-1.0, 1.0)
        var v = List[Float64](length=6, fill=0.0)
        for sg in range(2):
            var sc = h if sg == 0 else -h
            var sx = List[Float64](length=71, fill=0.0)
            var dx = List[Float64](length=SM_DELTA_DIM, fill=0.0)
            for k in range(71):
                sx[k] = sv[b * 71 + k] + sc * ds[k]
            for k in range(SM_DELTA_DIM):
                dx[k] = dlt[b * SM_DELTA_DIM + k] + sc * dd[k]
            var f = List[Float64](length=SM_FEAT_DIM, fill=0.0)
            var o = List[Float64](length=71, fill=0.0)
            var e = List[Float64](length=SM_ERR_DIM, fill=0.0)
            sm_canonical_lane[DType.float64](_p(sx), _p(f))
            sm_integrate_lane[DType.float64](_p(sx), _p(dx), _p(o))
            sm_error_lane[DType.float64](_p(sx), _p(st, b * 71), _p(e))
            for k in range(SM_FEAT_DIM):
                v[sg * 3 + 0] += g_feat[b * SM_FEAT_DIM + k] * f[k]
            for k in range(71):
                v[sg * 3 + 1] += g_next[b * 71 + k] * o[k]
            for k in range(SM_ERR_DIM):
                v[sg * 3 + 2] += g_err[b * SM_ERR_DIM + k] * e[k]
        var an = List[Float64](length=3, fill=0.0)
        for k in range(71):
            an[0] += gsc[b * 71 + k] * ds[k]
            an[1] += gsi[b * 71 + k] * ds[k]
            an[2] += gse[b * 71 + k] * ds[k]
        for k in range(SM_DELTA_DIM):
            an[1] += gdi[b * SM_DELTA_DIM + k] * dd[k]
        for c in range(3):
            var fd = (v[c] - v[3 + c]) / (2.0 * h)
            e_fd[c] = max(e_fd[c], abs(fd - an[c]) / max(1.0, abs(an[c])))
    t.truth(e_fd[0] <= 1e-6, "canonical: max rel err " + String(e_fd[0]))
    t.truth(e_fd[1] <= 1e-6, "integrate: max rel err " + String(e_fd[1]))
    t.truth(e_fd[2] <= 1e-6, "error_vector: max rel err " + String(e_fd[2]))

    # ── [3] the window loss ──
    print("\n[3] the k = 8 window loss (float32 nn) from the oracle's weights")
    var names = List[String]()
    var sizes = List[Int]()
    for l in string_from_bytes(read_file_bytes(dump + "/b_names.txt")).split("\n"):
        var ls = String(l)
        if ls.byte_length() == 0:
            continue
        var f = ls.split("\t")
        names.append(String(f[0]))
        sizes.append(Int(String(f[1])))
    var total = 0
    for k in range(len(sizes)):
        total += sizes[k]
    var wv = load_npy_2d_f64(dump + "/b_weights.npy", total).values.copy()
    var wm = SMWorldModel[H, TH].make[Kaiming]()
    var ld = _Load(wv.copy(), names.copy(), sizes.copy())
    wm.for_each_param["cpu"](ld, None)
    t.truth(ld.bad == 0 and ld.k == len(names) and ld.at == len(wv), String(ld.k) + " params loaded by name, " + String(ld.at) + " values")
    var batch = SMWindowBatch(K, B)
    var bs = load_npy_2d_f64(dump + "/b_states.npy", 71).values.copy()
    var ba = load_npy_2d_f64(dump + "/b_a_exec.npy", 29).values.copy()
    var bx = load_npy_2d_f64(dump + "/b_xi.npy", 33).values.copy()
    var bd = load_npy_2d_f64(dump + "/b_done.npy", 1).values.copy()
    var br = load_npy_2d_f64(dump + "/b_ref_root.npy", 7).values.copy()
    for i in range(len(bs)):
        batch.states[i] = Scalar[DT](bs[i])
    for i in range(len(ba)):
        batch.a_exec[i] = Scalar[DT](ba[i])
    for i in range(len(bx)):
        batch.xi[i] = Scalar[DT](bx[i])
    for i in range(len(bd)):
        batch.done[i] = Scalar[DT](bd[i])
    for i in range(len(br)):
        batch.ref_root[i] = Scalar[DT](br[i])
    var want = load_npy_2d_f64(dump + "/b_loss.npy", 6).values.copy()
    wm.zero_grad["cpu"](None)
    var lo = sm_wm_window_loss[H, TH, B](wm, batch, True)
    t.truth(_rel(lo.loss, want[0]) <= 1e-5, "loss " + String(lo.loss) + " (oracle " + String(want[0]) + ")")
    t.truth(_rel(lo.state_loss, want[1]) <= 1e-5 and _rel(lo.term_bce, want[2]) <= 1e-5,
        "state loss " + String(lo.state_loss) + ", BCE " + String(lo.term_bce))
    t.truth(_rel(lo.nse_1step, want[3]) <= 1e-5 and _rel(lo.nse_kstep, want[4]) <= 1e-5,
        "open-loop error 1-step " + String(lo.nse_1step) + ", 8-step " + String(lo.nse_kstep))

    # ── [4] its gradient ──
    print("\n[4] the gradient, every parameter")
    var gc = _Collect(True)
    wm.for_each_param["cpu"](gc, None)
    var wg = load_npy_2d_f64(dump + "/b_grads.npy", len(gc.out)).values.copy()
    var all = _cos(gc.out, wg, 0, len(wg))
    t.truth(all[0] >= 0.9999 and all[1] <= 1e-4, "all " + String(len(wg)) + " params: cosine " + String(all[0]) + ", max err " + String(all[1]) + " of the largest")
    var off = 0
    var worst_cos = 1.0
    var worst_err = 0.0
    var worst_name = String("")
    for k in range(len(names)):
        var c = _cos(gc.out, wg, off, off + sizes[k])
        if c[0] < worst_cos or c[1] > worst_err:
            worst_name = names[k]
        worst_cos = min(worst_cos, c[0])
        worst_err = max(worst_err, c[1])
        off += sizes[k]
    t.truth(worst_cos >= 0.9999 and worst_err <= 1e-4, "worst layer (" + worst_name + "): cosine " + String(worst_cos) + ", max err " + String(worst_err))

    # ── [5] clip + Adam ──
    print("\n[5] clip_grad_norm(1.0) + one Adam(2e-4) step")
    var gn = clip_grad_norm["cpu"](wm, Scalar[DT](1.0), None)
    t.truth(_rel(Float64(gn), want[5]) <= 1e-5, "pre-clip norm " + String(gn) + " (oracle " + String(want[5]) + ")")
    var opt = Adam(lr=Scalar[DT](2e-4))
    opt.step["cpu"](wm, None)
    var pc = _Collect(False)
    wm.for_each_param["cpu"](pc, None)
    var ws = load_npy_2d_f64(dump + "/b_stepped.npy", len(pc.out)).values.copy()
    # Adam's first step is lr·g/(|g| + 1e-8): where the CLIPPED |g| is near
    # eps, fp32 noise in g (4e-7 of the largest component, [4]) moves the
    # update by percents of lr. So gate the well-conditioned elements tightly
    # and only bound the rest.
    var scale = min(1.0, 1.0 / (want[5] + 1e-6))
    var e_good = 0.0
    var e_rest = 0.0
    var n_rest = 0
    for i in range(len(ws)):
        var d = abs(pc.out[i] - ws[i])
        if abs(wg[i] * scale) > 1e-6:
            # in float32 ulps of the weight (a stepped weight is rounded to fp32)
            e_good = max(e_good, d / (abs(ws[i]) * 1.1920929e-7 + 1e-12))
        else:
            e_rest = max(e_rest, d)
            n_rest += 1
    # 0.5 ulp is the final rounding; the fp32 gradient's own error reaches
    # the update through eps/|g| when |g| is ~1e-6 (measured 1.5 ulp worst)
    t.truth(e_good <= 2.0, "stepped weights with |clipped g| > 1e-6: max err " + String(e_good) + " float32 ulps of the weight")
    t.truth(e_rest <= 2e-5, String(n_rest) + " near-eps elements: max abs err " + String(e_rest) + " (lr 2e-4)")

    # ── [6] zero init ──
    print("\n[6] make: the dynamics start at constant velocity")
    var wz = SMWorldModel[H, TH].make[Kaiming]()
    var x = TensorImpl[DT].alloc(B * SM_WM_IN)
    for i in range(B * SM_WM_IN):
        x.data[i] = Scalar[DT](random_float64(-1.0, 1.0))
    var dl = TensorImpl[DT].alloc(B * SM_DELTA_DIM)
    wz.dyn.forward["cpu", B](TensorRefs[1, _, DT](x), dl, None)
    var mx = 0.0
    for i in range(B * SM_DELTA_DIM):
        mx = max(mx, abs(Float64(dl.data[i])))
    var zero = List[Float64](length=SM_DELTA_DIM, fill=0.0)
    var o = List[Float64](length=71, fill=0.0)
    sm_integrate_lane[DType.float64](_p(sv, 0), _p(zero), _p(o))
    var ce = 0.0
    for k in range(3):
        ce = max(ce, abs(o[k] - (sv[k] + SM_WM_DT * sv[36 + k])))
    for j in range(29):
        ce = max(ce, abs(o[7 + j] - (sv[7 + j] + SM_WM_DT * sv[42 + j])))
    t.truth(mx == 0.0 and ce <= 1e-15, "delta = 0 (max " + String(mx) + "); one step = p + dt v, q + dt qd (err " + String(ce) + ")")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_supermimic_wm: " + String(t.fails) + " failed")
