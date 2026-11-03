# +--------------------------------------------------------------------------+ #
# | SuperMimic M4: the world model — dynamics MLP, termination head, Eq. 1
# +--------------------------------------------------------------------------+ #
"""The reproduction's `WorldModel` (`world_model/world_model.py`,
`dynamics.py`) on noeira's `nn`, with the k-step autoregressive loss of the
paper's Eq. 1 and its gradient by a hand-written reverse scan.

    one step:  feat = canonical(s);  x = [feat, a, normalise(ξ)]  (WM_IN; 133 on the G1)
               delta = dynamics(x)   (MLP WM_IN → H → H → H → DELTA, SiLU, last
                                      layer zero-initialised)
               s' = integrate(s, delta)
    the head:  logit = term_head([f/thr, |f/thr|])   (MLP 4 → TH → TH → 1,
               SiLU), f = the analytic termination features of s',
               thr = (0.25 m, 0.8)

    loss = (1/k) Σ_j [ mean_b Σ_d e_j² ] + w_term (1/k) Σ_j mean_b BCE(logit_j, done_j)

with s_0 the logged state, s_{j+1} the model's own prediction (open loop),
e_j = `error_vector(s_{j+1}, logged s_{j+1})`. The gradient flows through
the whole chain and, as in the reproduction, from the BCE into the dynamics
through the predicted root pose.

Paper sizes are H = 2048, TH = 64; the tests run small H.
"""

from std.math import exp, log

from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.heads import sm_term_features, sm_term_features_vjp
from noeira.deep_agents.supermimic.state_spec import (
    sm_canonical_lane, sm_canonical_vjp_lane, sm_error_lane, sm_error_vjp_lane, sm_integrate_lane,
    sm_integrate_vjp_lane, sm_xi_normalize,
)
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.param import ParamVisitor, ParamWalkable
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.walkers import join_name
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_swish import LinearSwish
from noeira.nn.primitives.zero_linear import ZeroLinear

from max.gpu.host import DeviceContext


comptime SM_TERM_IN: Int = 4
comptime SM_HEIGHT_THR: Float64 = 0.25
comptime SM_TILT_THR: Float64 = 0.8

comptime SMDynamics[R: TrackingRobot, H: Int] = Sequential[
    LinearSwish[SMDims[R].WM_IN, H], LinearSwish[H, H], LinearSwish[H, H], ZeroLinear[H, SMDims[R].DELTA],
]
comptime SMTermHead[TH: Int] = Sequential[
    LinearSwish[SM_TERM_IN, TH], LinearSwish[TH, TH], Linear[TH, 1],
]


struct SMWorldModel[R: TrackingRobot, H: Int, TH: Int](ParamWalkable):
    """The dynamics MLP and the termination head, walked as ONE model so a
    global-norm clip spans both (`clip_grad_norm_(wm.parameters())`).
    Param names: `dynamics.<i>.weight|bias`, `term_head.<i>.weight|bias`;
    weights are `[IN, OUT]` row-major (torch's transposed)."""
    var dyn: SMDynamics[Self.R, Self.H]
    var term: SMTermHead[Self.TH]

    def __init__(out self, var dyn: SMDynamics[Self.R, Self.H], var term: SMTermHead[Self.TH]):
        self.dyn = dyn^
        self.term = term^

    def __init__(out self, *, deinit move: Self):
        self.dyn = move.dyn^
        self.term = move.term^

    @staticmethod
    def make[INIT: Initializer]() raises -> Self:
        return Self(SMDynamics[Self.R, Self.H].make["cpu", INIT](None), SMTermHead[Self.TH].make["cpu", INIT](None))

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext], prefix: String = String("")
    ) raises:
        self.dyn.for_each_param[target](visitor, ctx, join_name(prefix, String("dynamics")))
        self.term.for_each_param[target](visitor, ctx, join_name(prefix, String("term_head")))

    def zero_grad[target: StaticString](mut self, ctx: Optional[DeviceContext]) raises:
        self.dyn.zero_grad[target](ctx)
        self.term.zero_grad[target](ctx)

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext], prefix: String = String("")
    ) raises:
        self.dyn.for_each_state[target](visitor, ctx, join_name(prefix, String("dynamics")))
        self.term.for_each_state[target](visitor, ctx, join_name(prefix, String("term_head")))


struct SMWindowBatch[R: TrackingRobot](Movable):
    """k-step replay windows for B lanes, t-major:
    states [(k+1)·B·S] logged, a_exec [k·B·n] executed actions, xi [B·X],
    done [k·B] termination label of state j+1, ref_root [k·B·7] the reference
    root (pos 3, quat 4) of transition j."""
    var k: Int
    var b: Int
    var states: TensorImpl[DT]
    var a_exec: TensorImpl[DT]
    var xi: TensorImpl[DT]
    var done: TensorImpl[DT]
    var ref_root: TensorImpl[DT]

    def __init__(out self, k: Int, b: Int):
        self.k = k
        self.b = b
        self.states = TensorImpl[DT].alloc((k + 1) * b * SMDims[Self.R].S)
        self.a_exec = TensorImpl[DT].alloc(k * b * Self.R.NDOF)
        self.xi = TensorImpl[DT].alloc(b * SMDims[Self.R].X)
        self.done = TensorImpl[DT].alloc(k * b)
        self.ref_root = TensorImpl[DT].alloc(k * b * 7)

    def __init__(out self, *, deinit move: Self):
        self.k = move.k
        self.b = move.b
        self.states = move.states^
        self.a_exec = move.a_exec^
        self.xi = move.xi^
        self.done = move.done^
        self.ref_root = move.ref_root^


@fieldwise_init
struct SMWMLoss(Copyable, Movable):
    var loss: Float64
    var state_loss: Float64
    var term_bce: Float64
    var nse_1step: Float64
    var nse_kstep: Float64


@always_inline
def _row(mut l: List[Scalar[DT]], off: Int) -> FP[DT]:
    return l.unsafe_ptr().unsafe_offset(off).as_unsafe_any_origin()


@always_inline
def _trow(mut t: TensorImpl[DT], off: Int) -> FP[DT]:
    return t.data.unsafe_ptr().unsafe_offset(off).as_unsafe_any_origin()


@always_inline
def _sigmoid(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x)) if x >= 0 else exp(x) / (1.0 + exp(x))


def sm_wm_window_loss[R: TrackingRobot, H: Int, TH: Int, B: Int](
    mut wm: SMWorldModel[R, H, TH], mut w: SMWindowBatch[R], grad: Bool, w_term: Float64 = 1.0
) raises -> SMWMLoss:
    """Eq. 1 on one batch of windows. With `grad`, ADDS d loss / d params into
    the model's grads (call `wm.zero_grad` first)."""
    comptime D = SMDims[R]
    comptime SM_S = D.S
    comptime SM_A = R.NDOF
    comptime SM_XI = D.X
    comptime SM_WM_IN = D.WM_IN
    comptime SM_FEAT_DIM = D.S
    comptime SM_DELTA_DIM = D.DELTA
    comptime SM_ERR_DIM = D.DELTA
    if w.b != B:
        raise Error("sm_wm_window_loss: batch of " + String(w.b) + " lanes, compiled for " + String(B))
    var K = w.k
    var sp = List[Scalar[DT]](length=(K + 1) * B * SM_S, fill=0)   # the predicted chain
    for i in range(B * SM_S):
        sp[i] = w.states.data[i]
    var xin = List[Scalar[DT]](length=B * SM_XI, fill=0)
    for i in range(B * SM_XI):
        xin[i] = Scalar[DT](sm_xi_normalize[R.NDOF](i % SM_XI, Float64(w.xi.data[i])))
    var xs = List[TensorImpl[DT]]()
    var ds = List[TensorImpl[DT]]()
    var tins = List[TensorImpl[DT]]()
    var lgs = List[TensorImpl[DT]]()
    var errs = List[Scalar[DT]](length=K * B * SM_ERR_DIM, fill=0)
    var state_tot = 0.0
    var bce_tot = 0.0
    var nse1 = 0.0
    var nsek = 0.0
    for j in range(K):
        var x = TensorImpl[DT].alloc(B * SM_WM_IN)
        for b in range(B):
            sm_canonical_lane[DT, R](_row(sp, (j * B + b) * SM_S), _trow(x, b * SM_WM_IN))
            for k in range(SM_A):
                x.data[b * SM_WM_IN + SM_FEAT_DIM + k] = w.a_exec.data[(j * B + b) * SM_A + k]
            for k in range(SM_XI):
                x.data[b * SM_WM_IN + SM_FEAT_DIM + SM_A + k] = xin[b * SM_XI + k]
        var dl = TensorImpl[DT].alloc(B * SM_DELTA_DIM)
        wm.dyn.forward["cpu", B](TensorRefs[1, _, DT](x), dl, None)
        var tin = TensorImpl[DT].alloc(B * SM_TERM_IN)
        var nse = 0.0
        for b in range(B):
            var so = ((j + 1) * B + b) * SM_S
            sm_integrate_lane[DT, R](_row(sp, (j * B + b) * SM_S), _trow(dl, b * SM_DELTA_DIM), _row(sp, so))
            var eo = (j * B + b) * SM_ERR_DIM
            sm_error_lane[DT, R](_row(sp, so), _row(w.states.data, so), _row(errs, eo))
            var sq = 0.0
            for d in range(SM_ERR_DIM):
                sq += Float64(errs[eo + d]) * Float64(errs[eo + d])
            state_tot += sq / Float64(B)
            nse += sq / Float64(SM_ERR_DIM * B)
            var r = (j * B + b) * 7
            var q = SIMD[DT, 4](w.ref_root.data[r + 3], w.ref_root.data[r + 4], w.ref_root.data[r + 5], w.ref_root.data[r + 6])
            var f = sm_term_features[DT](_row(sp, so), w.ref_root.data[r + 2], q)
            var x0 = f[0] / Scalar[DT](SM_HEIGHT_THR)
            var x1 = f[1] / Scalar[DT](SM_TILT_THR)
            tin.data[b * 4 + 0] = x0
            tin.data[b * 4 + 1] = x1
            tin.data[b * 4 + 2] = abs(x0)
            tin.data[b * 4 + 3] = abs(x1)
        var lg = TensorImpl[DT].alloc(B)
        wm.term.forward["cpu", B](TensorRefs[1, _, DT](tin), lg, None)
        for b in range(B):
            var l = Float64(lg.data[b])
            var y = Float64(w.done.data[j * B + b])
            bce_tot += (max(l, 0.0) - l * y + log(1.0 + exp(-abs(l)))) / Float64(B)
        if j == 0:
            nse1 = nse
        nsek = nse
        xs.append(x^)
        ds.append(dl^)
        tins.append(tin^)
        lgs.append(lg^)
    var out = SMWMLoss(state_tot / Float64(K) + w_term * bce_tot / Float64(K), state_tot / Float64(K), bce_tot / Float64(K), nse1, nsek)
    if not grad:
        return out^

    # ── the reverse scan ──
    var gs = List[Scalar[DT]](length=B * SM_S, fill=0)          # d loss / d s_{j+1}
    var cs = Scalar[DT](2.0 / Float64(K * B))
    var cb = w_term / Float64(K * B)
    var j = K - 1
    while j >= 0:
        var g_err = List[Scalar[DT]](length=SM_ERR_DIM, fill=0)
        var glg = TensorImpl[DT].alloc(B)
        for b in range(B):
            var eo = (j * B + b) * SM_ERR_DIM
            for d in range(SM_ERR_DIM):
                g_err[d] = errs[eo + d] * cs
            var so = ((j + 1) * B + b) * SM_S
            sm_error_vjp_lane[DT, R](_row(sp, so), _row(w.states.data, so), _row(g_err, 0), _row(gs, b * SM_S))
            glg.data[b] = Scalar[DT]((_sigmoid(Float64(lgs[j].data[b])) - Float64(w.done.data[j * B + b])) * cb)
        var gtin = TensorImpl[DT].alloc(B * SM_TERM_IN)
        # ⚠ RE-RUN THE STEP'S FORWARD FIRST: the fused Linear+SiLU leaves keep
        # the activations of their LAST forward call, which is step k − 1's.
        # A `vjp` at step j against them is the wrong gradient (measured: term
        # head cosine 0.98 against torch before this line).
        var relg = TensorImpl[DT].alloc(B)
        wm.term.forward["cpu", B](TensorRefs[1, _, DT](tins[j]), relg, None)
        wm.term.vjp["cpu", B](TensorRefs[1, _, DT](tins[j]), glg, TensorRefs[1, _, DT](gtin), None)
        for b in range(B):
            var x0 = tins[j].data[b * 4 + 0]
            var x1 = tins[j].data[b * 4 + 1]
            var s0 = Scalar[DT](1) if x0 > 0 else (Scalar[DT](-1) if x0 < 0 else Scalar[DT](0))
            var s1 = Scalar[DT](1) if x1 > 0 else (Scalar[DT](-1) if x1 < 0 else Scalar[DT](0))
            var g0 = (gtin.data[b * 4 + 0] + s0 * gtin.data[b * 4 + 2]) / Scalar[DT](SM_HEIGHT_THR)
            var g1 = (gtin.data[b * 4 + 1] + s1 * gtin.data[b * 4 + 3]) / Scalar[DT](SM_TILT_THR)
            sm_term_features_vjp[DT](_row(sp, ((j + 1) * B + b) * SM_S), g0, g1, _row(gs, b * SM_S))
        # through s_{j+1} = integrate(s_j, dynamics([canonical(s_j), a_j, ξ]))
        var gprev = List[Scalar[DT]](length=B * SM_S, fill=0)
        var gdl = TensorImpl[DT].alloc(B * SM_DELTA_DIM)
        for b in range(B):
            sm_integrate_vjp_lane[DT, R](
                _row(sp, (j * B + b) * SM_S), _trow(ds[j], b * SM_DELTA_DIM), _row(gs, b * SM_S),
                _row(gprev, b * SM_S), _trow(gdl, b * SM_DELTA_DIM),
            )
        var gx = TensorImpl[DT].alloc(B * SM_WM_IN)
        var redl = TensorImpl[DT].alloc(B * SM_DELTA_DIM)
        wm.dyn.forward["cpu", B](TensorRefs[1, _, DT](xs[j]), redl, None)   # see the term head's note
        wm.dyn.vjp["cpu", B](TensorRefs[1, _, DT](xs[j]), gdl, TensorRefs[1, _, DT](gx), None)
        if j > 0:   # s_0 is data
            for b in range(B):
                sm_canonical_vjp_lane[DT, R](_row(sp, (j * B + b) * SM_S), _trow(gx, b * SM_WM_IN), _row(gprev, b * SM_S))
        gs = gprev^
        j -= 1
    return out^
