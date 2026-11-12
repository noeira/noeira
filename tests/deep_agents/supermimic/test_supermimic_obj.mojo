"""SuperMimic with an object: the object's state spec and the joint k-step loss + gradient.

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/obj_oracle.py --out /tmp/sm_obj
    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_obj.mojo /tmp/sm_obj

The oracle is a second transcription of the object block in torch float64
with autograd (`tools/supermimic/obj_oracle.py`); the robot's side of it is
the reproduction's `StateSpec` and `WorldModel`, verbatim.

WHAT EACH SECTION CAN CATCH
===========================
[1] float64 `sm_obj_feat_lane`, `sm_obj_integrate_lane`, `sm_obj_error_lane`
    against the oracle on 48 lanes (a third inside the small-angle series),
    1e-12; their VJPs against autograd, 1e-10 relative — including the
    cotangent of the ROBOT's root pose (the heading), which a robot-only
    test never exercises.
[2] the same VJPs against central differences of our own forwards (no
    torch), 16 directions each, 1e-6 relative.
[3] the joint k = 8 window loss in float32 (our `nn`) from the oracle's
    weights and batch: loss, robot loss, BCE, object loss and the four
    open-loop errors, 1e-5 relative.
[4] its gradient, every parameter of the three networks (the tracker's
    dynamics and termination head, the object network): cosine ≥ 0.9999 and
    max error ≤ 1e-4 of the largest component, overall and per layer.
[5] a zero-initialised object network (`make`): the robot's loss terms and
    the world model's gradient are `sm_wm_window_loss`'s own, and the object
    moves at constant velocity. This is the claim that the block starts as
    the tracker's world model.
[6] heading invariance: turning robot and object together about the
    vertical leaves the features unchanged and turns the integrated object
    with them.

⚠ COSINE ALONE IS NOT A GRADIENT GATE (`test_supermimic_wm`'s note): the
per-layer max-error bound in [4] is what catches a small missing path.
"""

from std.math import cos, sin, sqrt
from std.random import random_float64, seed
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.core.bytes import string_from_bytes
from noeira.deep_agents.supermimic.kinematics import FP, Q4, qapply, qmul
from noeira.deep_agents.supermimic.object_lanes import SMObjDims
from noeira.deep_agents.supermimic.object_model import SMObjectModel, SMObjWindowBatch, sm_obj_window_loss
from noeira.deep_agents.supermimic.object_spec import (
    SM_OBJ, SM_OBJ_DELTA, SM_OBJ_FEAT, sm_obj_error_lane, sm_obj_error_vjp_lane, sm_obj_feat_lane, sm_obj_feat_vjp_lane,
    sm_obj_integrate_lane, sm_obj_integrate_vjp_lane,
)
from noeira.deep_agents.supermimic.robot import G1Robot, SMDims
from noeira.deep_agents.supermimic.state_spec import SM_WM_DT
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWorldModel, sm_wm_window_loss
from noeira.io.fileio import read_file_bytes
from noeira.io.npy import load_npy_2d_f64
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


comptime R = G1Robot
comptime S = SMDims[R].S                # 71
comptime O = SM_OBJ                     # 13
comptime OF = SM_OBJ_FEAT               # 15
comptime OD = SM_OBJ_DELTA              # 12
comptime H = 32
comptime TH = 16
comptime HO = 24
comptime K = 8
comptime B = 16
comptime F64 = DType.float64


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


def _lane_err(got: List[Float64], want: List[Float64]) -> Float64:
    var e = 0.0
    for i in range(len(got)):
        e = max(e, _rel(got[i], want[i]))
    return e


def _max_abs(a: List[Float64], lo: Int, hi: Int, stride: Int) -> Float64:
    """max |a[r·stride + lo : r·stride + hi]| over the rows."""
    var m = 0.0
    for r in range(len(a) // stride):
        for i in range(lo, hi):
            m = max(m, abs(a[r * stride + i]))
    return m


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


def _fill(mut dst: TensorImpl[DT], src: List[Float64]) raises:
    if len(src) != len(dst.data):
        raise Error("_fill: " + String(len(src)) + " values for a tensor of " + String(len(dst.data)))
    for i in range(len(src)):
        dst.data[i] = Scalar[DT](src[i])


def main() raises:
    var av = argv()
    var dump = String(av[1]) if len(av) > 1 else String("/tmp/sm_obj")
    var t = Tally()

    # ── [1] the object's state spec vs the oracle ──
    print("\n[1] object state spec (float64) against the torch transcription")
    var s = load_npy_2d_f64(dump + "/a_s.npy", S)
    var n = s.rows
    var sv = s.values.copy()
    var ov = load_npy_2d_f64(dump + "/a_o.npy", O).values.copy()
    var dlt = load_npy_2d_f64(dump + "/a_delta.npy", OD).values.copy()
    var ot = load_npy_2d_f64(dump + "/a_o_true.npy", O).values.copy()
    var w_feat = load_npy_2d_f64(dump + "/a_feat.npy", OF).values.copy()
    var w_next = load_npy_2d_f64(dump + "/a_next.npy", O).values.copy()
    var w_err = load_npy_2d_f64(dump + "/a_err.npy", OD).values.copy()
    var g_feat = load_npy_2d_f64(dump + "/a_g_feat.npy", OF).values.copy()
    var g_next = load_npy_2d_f64(dump + "/a_g_next.npy", O).values.copy()
    var g_err = load_npy_2d_f64(dump + "/a_g_err.npy", OD).values.copy()
    var w_gsf = load_npy_2d_f64(dump + "/a_gs_feat.npy", S).values.copy()
    var w_gof = load_npy_2d_f64(dump + "/a_go_feat.npy", O).values.copy()
    var w_gsi = load_npy_2d_f64(dump + "/a_gs_integ.npy", S).values.copy()
    var w_goi = load_npy_2d_f64(dump + "/a_go_integ.npy", O).values.copy()
    var w_gdi = load_npy_2d_f64(dump + "/a_gd_integ.npy", OD).values.copy()
    var w_goe = load_npy_2d_f64(dump + "/a_go_err.npy", O).values.copy()
    var feat = List[Float64](length=n * OF, fill=0.0)
    var nxt = List[Float64](length=n * O, fill=0.0)
    var err = List[Float64](length=n * OD, fill=0.0)
    var gsf = List[Float64](length=n * S, fill=0.0)
    var gof = List[Float64](length=n * O, fill=0.0)
    var gsi = List[Float64](length=n * S, fill=0.0)
    var goi = List[Float64](length=n * O, fill=0.0)
    var gdi = List[Float64](length=n * OD, fill=0.0)
    var goe = List[Float64](length=n * O, fill=0.0)
    for b in range(n):
        sm_obj_feat_lane[F64, R](_p(sv, b * S), _p(ov, b * O), _p(feat, b * OF))
        sm_obj_integrate_lane[F64, R](_p(sv, b * S), _p(ov, b * O), _p(dlt, b * OD), _p(nxt, b * O))
        sm_obj_error_lane[F64](_p(ov, b * O), _p(ot, b * O), _p(err, b * OD))
        sm_obj_feat_vjp_lane[F64, R](_p(sv, b * S), _p(ov, b * O), _p(g_feat, b * OF), _p(gsf, b * S), _p(gof, b * O))
        sm_obj_integrate_vjp_lane[F64, R](
            _p(sv, b * S), _p(ov, b * O), _p(dlt, b * OD), _p(g_next, b * O), _p(gsi, b * S), _p(goi, b * O), _p(gdi, b * OD),
        )
        sm_obj_error_vjp_lane[F64](_p(ov, b * O), _p(ot, b * O), _p(g_err, b * OD), _p(goe, b * O))
    t.truth(_lane_err(feat, w_feat) <= 1e-12, "features: max err " + String(_lane_err(feat, w_feat)))
    t.truth(_lane_err(nxt, w_next) <= 1e-12, "integrate: max err " + String(_lane_err(nxt, w_next)))
    t.truth(_lane_err(err, w_err) <= 1e-12, "error: max err " + String(_lane_err(err, w_err)))
    var e_f = max(_lane_err(gsf, w_gsf), _lane_err(gof, w_gof))
    t.truth(e_f <= 1e-10, "features VJP (robot state and object): max rel err " + String(e_f))
    var e_i = max(max(_lane_err(gsi, w_gsi), _lane_err(goi, w_goi)), _lane_err(gdi, w_gdi))
    t.truth(e_i <= 1e-10, "integrate VJP (robot state, object, delta): max rel err " + String(e_i))
    t.truth(_lane_err(goe, w_goe) <= 1e-10, "error VJP: max rel err " + String(_lane_err(goe, w_goe)))
    # the heading's share is there to be gated, and nothing else of the robot's state gets a cotangent
    var hq_f = _max_abs(w_gsf, 3, 7, S)
    var hq_i = _max_abs(w_gsi, 3, 7, S)
    var rest = max(max(_max_abs(gsf, 7, S, S), _max_abs(gsi, 0, 3, S)), _max_abs(gsi, 7, S, S))
    t.truth(hq_f > 1.0 and hq_i > 0.1 and rest == 0.0,
        "the robot's root quaternion carries a cotangent (features " + String(hq_f) + ", integrate " + String(hq_i)
        + "); its joints and velocities none (" + String(rest) + ")")

    # ── [2] against central differences ──
    print("\n[2] the same VJPs against central differences of our forwards")
    seed(5)
    var h = 1e-6
    var e_fd = List[Float64](length=3, fill=0.0)
    for it in range(16):
        var b = (it * 5) % n
        var ds = List[Float64](length=S, fill=0.0)
        var do = List[Float64](length=O, fill=0.0)
        var dd = List[Float64](length=OD, fill=0.0)
        for k in range(S):
            ds[k] = random_float64(-1.0, 1.0)
        for k in range(O):
            do[k] = random_float64(-1.0, 1.0)
        for k in range(OD):
            dd[k] = random_float64(-1.0, 1.0)
        var v = List[Float64](length=6, fill=0.0)
        for sg in range(2):
            var sc = h if sg == 0 else -h
            var sx = List[Float64](length=S, fill=0.0)
            var ox = List[Float64](length=O, fill=0.0)
            var dx = List[Float64](length=OD, fill=0.0)
            for k in range(S):
                sx[k] = sv[b * S + k] + sc * ds[k]
            for k in range(O):
                ox[k] = ov[b * O + k] + sc * do[k]
            for k in range(OD):
                dx[k] = dlt[b * OD + k] + sc * dd[k]
            var f = List[Float64](length=OF, fill=0.0)
            var o1 = List[Float64](length=O, fill=0.0)
            var e = List[Float64](length=OD, fill=0.0)
            sm_obj_feat_lane[F64, R](_p(sx), _p(ox), _p(f))
            sm_obj_integrate_lane[F64, R](_p(sx), _p(ox), _p(dx), _p(o1))
            sm_obj_error_lane[F64](_p(ox), _p(ot, b * O), _p(e))
            for k in range(OF):
                v[sg * 3 + 0] += g_feat[b * OF + k] * f[k]
            for k in range(O):
                v[sg * 3 + 1] += g_next[b * O + k] * o1[k]
            for k in range(OD):
                v[sg * 3 + 2] += g_err[b * OD + k] * e[k]
        var an = List[Float64](length=3, fill=0.0)
        for k in range(S):
            an[0] += gsf[b * S + k] * ds[k]
            an[1] += gsi[b * S + k] * ds[k]
        for k in range(O):
            an[0] += gof[b * O + k] * do[k]
            an[1] += goi[b * O + k] * do[k]
            an[2] += goe[b * O + k] * do[k]
        for k in range(OD):
            an[1] += gdi[b * OD + k] * dd[k]
        for c in range(3):
            var fd = (v[c] - v[3 + c]) / (2.0 * h)
            e_fd[c] = max(e_fd[c], abs(fd - an[c]) / max(1.0, abs(an[c])))
    t.truth(e_fd[0] <= 1e-6, "features: max rel err " + String(e_fd[0]))
    t.truth(e_fd[1] <= 1e-6, "integrate: max rel err " + String(e_fd[1]))
    t.truth(e_fd[2] <= 1e-6, "error: max rel err " + String(e_fd[2]))

    # ── [3] the joint window loss ──
    print("\n[3] the joint k = 8 window loss (float32 nn) from the oracle's weights")
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
    var wm = SMWorldModel[R, H, TH].make[Kaiming]()
    var om = SMObjectModel[R, HO].make[Kaiming]()
    var ld = _Load(wv.copy(), names.copy(), sizes.copy())
    wm.for_each_param["cpu"](ld, None)
    var n_wm = ld.at
    om.for_each_param["cpu"](ld, None)
    t.truth(ld.bad == 0 and ld.k == len(names) and ld.at == len(wv),
        String(ld.k) + " params loaded by name, " + String(ld.at) + " values (" + String(n_wm) + " the world model's)")
    var batch = SMWindowBatch[R](K, B)
    var obatch = SMObjWindowBatch(K, B)
    _fill(batch.states, load_npy_2d_f64(dump + "/b_states.npy", S).values)
    _fill(batch.a_exec, load_npy_2d_f64(dump + "/b_a_exec.npy", 29).values)
    _fill(batch.xi, load_npy_2d_f64(dump + "/b_xi.npy", 33).values)
    _fill(batch.done, load_npy_2d_f64(dump + "/b_done.npy", 1).values)
    _fill(batch.ref_root, load_npy_2d_f64(dump + "/b_ref_root.npy", 7).values)
    _fill(obatch.states, load_npy_2d_f64(dump + "/b_objs.npy", O).values)
    _fill(obatch.ctx, load_npy_2d_f64(dump + "/b_octx.npy", 2).values)
    var want = load_npy_2d_f64(dump + "/b_loss.npy", 9).values.copy()
    var w_obj = want[8]
    wm.zero_grad["cpu"](None)
    om.zero_grad["cpu"](None)
    var lo = sm_obj_window_loss[R, H, TH, HO, B](wm, om, batch, obatch, True, w_obj=w_obj)
    t.truth(_rel(lo.loss, want[0]) <= 1e-5, "loss " + String(lo.loss) + " (oracle " + String(want[0]) + ")")
    t.truth(_rel(lo.state_loss, want[1]) <= 1e-5 and _rel(lo.term_bce, want[2]) <= 1e-5,
        "robot loss " + String(lo.state_loss) + ", BCE " + String(lo.term_bce))
    t.truth(_rel(lo.obj_loss, want[3]) <= 1e-5, "object loss " + String(lo.obj_loss) + " (oracle " + String(want[3]) + ")")
    t.truth(_rel(lo.nse_1step, want[4]) <= 1e-5 and _rel(lo.nse_kstep, want[5]) <= 1e-5
        and _rel(lo.obj_nse_1step, want[6]) <= 1e-5 and _rel(lo.obj_nse_kstep, want[7]) <= 1e-5,
        "open-loop errors: robot " + String(lo.nse_1step) + " / " + String(lo.nse_kstep) + ", object "
        + String(lo.obj_nse_1step) + " / " + String(lo.obj_nse_kstep))

    # ── [4] its gradient ──
    print("\n[4] the gradient, every parameter of the three networks")
    var gc = _Collect(True)
    wm.for_each_param["cpu"](gc, None)
    om.for_each_param["cpu"](gc, None)
    var wg = load_npy_2d_f64(dump + "/b_grads.npy", len(gc.out)).values.copy()
    var all = _cos(gc.out, wg, 0, len(wg))
    t.truth(all[0] >= 0.9999 and all[1] <= 1e-4,
        "all " + String(len(wg)) + " params: cosine " + String(all[0]) + ", max err " + String(all[1]) + " of the largest")
    var c_wm = _cos(gc.out, wg, 0, n_wm)
    var c_om = _cos(gc.out, wg, n_wm, len(wg))
    t.truth(c_wm[0] >= 0.9999 and c_wm[1] <= 1e-4, "the world model's " + String(n_wm) + ": cosine " + String(c_wm[0]) + ", max err " + String(c_wm[1]))
    t.truth(c_om[0] >= 0.9999 and c_om[1] <= 1e-4,
        "the object network's " + String(len(wg) - n_wm) + ": cosine " + String(c_om[0]) + ", max err " + String(c_om[1]))
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
    t.truth(worst_cos >= 0.9999 and worst_err <= 1e-4,
        "worst layer (" + worst_name + "): cosine " + String(worst_cos) + ", max err " + String(worst_err))

    # ── [5] the zero-initialised object network ──
    print("\n[5] make: the block starts as the tracker's world model")
    var om0 = SMObjectModel[R, HO].make[Kaiming]()
    wm.zero_grad["cpu"](None)
    var plain = sm_wm_window_loss[R, H, TH, B](wm, batch, True)
    var gp = _Collect(True)
    wm.for_each_param["cpu"](gp, None)
    wm.zero_grad["cpu"](None)
    om0.zero_grad["cpu"](None)
    var l0 = sm_obj_window_loss[R, H, TH, HO, B](wm, om0, batch, obatch, True, w_obj=w_obj)
    var gj = _Collect(True)
    wm.for_each_param["cpu"](gj, None)
    t.truth(l0.state_loss == plain.state_loss and l0.term_bce == plain.term_bce and l0.nse_kstep == plain.nse_kstep,
        "robot loss " + String(l0.state_loss) + " and BCE " + String(l0.term_bce) + " equal sm_wm_window_loss's exactly")
    var cz = _cos(gj.out, gp.out, 0, len(gp.out))
    t.truth(cz[1] <= 1e-6, "the world model's gradient is sm_wm_window_loss's: max diff " + String(cz[1]) + " of the largest")
    # constant velocity: the object's k-step chain from o_0 is p + j dt v, and the loss is that chain's error
    var cv = 0.0
    var ob0 = load_npy_2d_f64(dump + "/b_objs.npy", O).values.copy()
    for step in range(K):
        for b in range(B):
            var pred = List[Float64](length=O, fill=0.0)
            var zero = List[Float64](length=OD, fill=0.0)
            var cur = List[Float64](length=O, fill=0.0)
            for i in range(O):
                cur[i] = ob0[b * O + i]
            for _ in range(step + 1):
                sm_obj_integrate_lane[F64, R](_p(sv, 0), _p(cur), _p(zero), _p(pred))
                for i in range(O):
                    cur[i] = pred[i]
            var e = List[Float64](length=OD, fill=0.0)
            sm_obj_error_lane[F64](_p(cur), _p(ob0, ((step + 1) * B + b) * O), _p(e))
            for i in range(OD):
                cv += e[i] * e[i] / Float64(B * K)
            if step == 0:
                var ce = 0.0
                for i in range(3):
                    ce = max(ce, abs(cur[i] - (ob0[b * O + i] + SM_WM_DT * ob0[b * O + 7 + i])))
                    ce = max(ce, abs(cur[7 + i] - ob0[b * O + 7 + i]))
                    ce = max(ce, abs(cur[10 + i] - ob0[b * O + 10 + i]))
                if ce > 1e-15:
                    cv = 1e30
    t.truth(_rel(l0.obj_loss, cv) <= 1e-5, "object loss " + String(l0.obj_loss) + " = the constant-velocity chain's (" + String(cv) + ")")
    var g0 = _Collect(True)
    om0.for_each_param["cpu"](g0, None)
    var n_last = HO * SMObjDims[R].OUT + SMObjDims[R].OUT
    var up = 0.0
    var last = 0.0
    for i in range(len(g0.out)):
        if i < len(g0.out) - n_last:
            up = max(up, abs(g0.out[i]))
        else:
            last = max(last, abs(g0.out[i]))
    t.truth(up == 0.0 and last > 1e-3, "only the object network's last layer has a gradient at the start (last " + String(last) + ", before it " + String(up) + ")")

    # ── [6] heading invariance ──
    print("\n[6] turning robot and object together about the vertical")
    var psi = 0.7
    var qz = Q4[F64](cos(psi / 2), 0, 0, sin(psi / 2))
    var e_feat = 0.0
    var e_int = 0.0
    for b in range(n):
        var s2 = List[Float64](length=S, fill=0.0)
        var o2 = List[Float64](length=O, fill=0.0)
        for i in range(S):
            s2[i] = sv[b * S + i]
        for i in range(O):
            o2[i] = ov[b * O + i]
        var rp = qapply(qz, Q4[F64](sv[b * S], sv[b * S + 1], sv[b * S + 2], 0))
        var rq = qmul(qz, Q4[F64](sv[b * S + 3], sv[b * S + 4], sv[b * S + 5], sv[b * S + 6]))
        var op2 = qapply(qz, Q4[F64](ov[b * O], ov[b * O + 1], ov[b * O + 2], 0))
        var oq2 = qmul(qz, Q4[F64](ov[b * O + 3], ov[b * O + 4], ov[b * O + 5], ov[b * O + 6]))
        var ovv = qapply(qz, Q4[F64](ov[b * O + 7], ov[b * O + 8], ov[b * O + 9], 0))
        for i in range(3):
            s2[i] = rp[i]
            o2[i] = op2[i]
            o2[7 + i] = ovv[i]
        for i in range(4):
            s2[3 + i] = rq[i]
            o2[3 + i] = oq2[i]
        var f2 = List[Float64](length=OF, fill=0.0)
        sm_obj_feat_lane[F64, R](_p(s2), _p(o2), _p(f2))
        for i in range(OF):
            e_feat = max(e_feat, abs(f2[i] - feat[b * OF + i]))
        var n2 = List[Float64](length=O, fill=0.0)
        sm_obj_integrate_lane[F64, R](_p(s2), _p(o2), _p(dlt, b * OD), _p(n2))
        var np1 = qapply(qz, Q4[F64](nxt[b * O], nxt[b * O + 1], nxt[b * O + 2], 0))
        var nv1 = qapply(qz, Q4[F64](nxt[b * O + 7], nxt[b * O + 8], nxt[b * O + 9], 0))
        var nq1 = qmul(qz, Q4[F64](nxt[b * O + 3], nxt[b * O + 4], nxt[b * O + 5], nxt[b * O + 6]))
        for i in range(3):
            e_int = max(e_int, abs(n2[i] - np1[i]))
            e_int = max(e_int, abs(n2[7 + i] - nv1[i]))
            e_int = max(e_int, abs(n2[10 + i] - nxt[b * O + 10 + i]))
        for i in range(4):
            e_int = max(e_int, abs(n2[3 + i] - nq1[i]))
    t.truth(e_feat <= 1e-12, "features unchanged: max diff " + String(e_feat))
    t.truth(e_int <= 1e-12, "the integrated object turns with them: max diff " + String(e_int))

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_supermimic_obj: " + String(t.fails) + " failed")
