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
from noeira.deep_agents.supermimic.lanes import (
    SM_HEIGHT_THR, SM_TERM_IN, SM_TILT_THR, sm_canon_back, sm_wm_dyn_back, sm_wm_err_back, sm_wm_in, sm_wm_step,
    talloc, tp,
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
    def make[INIT: Initializer, target: StaticString = "cpu"](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self(SMDynamics[Self.R, Self.H].make[target, INIT](ctx), SMTermHead[Self.TH].make[target, INIT](ctx))

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

    def to_device(mut self, ctx: DeviceContext) raises:
        """Copy the host fields into their device buffers (kept across calls)."""
        self.states.upload_resident(ctx)
        self.a_exec.upload_resident(ctx)
        self.xi.upload_resident(ctx)
        self.done.upload_resident(ctx)
        self.ref_root.upload_resident(ctx)


@fieldwise_init
struct SMWMLoss(Copyable, Movable):
    var loss: Float64
    var state_loss: Float64
    var term_bce: Float64
    var nse_1step: Float64
    var nse_kstep: Float64

    def add_scaled(mut self, l: SMWMLoss, w: Float64):
        """self += w · l (the mean over an update's batches)."""
        self.loss += w * l.loss
        self.state_loss += w * l.state_loss
        self.term_bce += w * l.term_bce
        self.nse_1step += w * l.nse_1step
        self.nse_kstep += w * l.nse_kstep


@always_inline
def _sigmoid(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x)) if x >= 0 else exp(x) / (1.0 + exp(x))


def sm_wm_window_loss[R: TrackingRobot, H: Int, TH: Int, B: Int, target: StaticString = "cpu"](
    mut wm: SMWorldModel[R, H, TH], mut w: SMWindowBatch[R], grad: Bool, w_term: Float64 = 1.0,
    ctx: Optional[DeviceContext] = None,
) raises -> SMWMLoss:
    """Eq. 1 on one batch of windows. With `grad`, ADDS d loss / d params into
    the model's grads (call `wm.zero_grad` first). On "gpu" the batch must be
    on the device (`w.to_device`) and the model made there; the per-lane
    phases are `lanes.mojo`'s, the reductions to the scalars below run on the
    host over per-lane values (K·B square errors and logits)."""
    comptime D = SMDims[R]
    comptime S = D.S
    comptime n = R.NDOF
    if w.b != B:
        raise Error("sm_wm_window_loss: batch of " + String(w.b) + " lanes, compiled for " + String(B))
    var K = w.k
    var sp = talloc[target]((K + 1) * B * S, ctx)                  # the predicted chain
    var errs = talloc[target](K * B * D.DELTA, ctx)
    var sq = talloc[target](K * B, ctx)
    var lgall = talloc[target](K * B, ctx)
    comptime if target == "gpu":
        ctx.value().enqueue_copy(sp.dev.value().create_sub_buffer[DT](0, B * S), w.states.dev.value().create_sub_buffer[DT](0, B * S))
    else:
        for i in range(B * S):
            sp.data[i] = w.states.data[i]
    var xs = List[TensorImpl[DT]]()
    var ds = List[TensorImpl[DT]]()
    var tins = List[TensorImpl[DT]]()
    for j in range(K):
        var x = talloc[target](B * D.WM_IN, ctx)
        sm_wm_in[target, R](ctx, B, tp[target](sp).unsafe_offset(j * B * S), tp[target](w.a_exec).unsafe_offset(j * B * n), tp[target](w.xi), tp[target](x))
        var dl = talloc[target](B * D.DELTA, ctx)
        wm.dyn.forward[target, B](TensorRefs[1, _, DT](x), dl, ctx)
        var tin = talloc[target](B * SM_TERM_IN, ctx)
        sm_wm_step[target, R](
            ctx, B, tp[target](sp).unsafe_offset(j * B * S), tp[target](dl), tp[target](sp).unsafe_offset((j + 1) * B * S),
            tp[target](w.states).unsafe_offset((j + 1) * B * S), tp[target](w.ref_root).unsafe_offset(j * B * 7),
            tp[target](errs).unsafe_offset(j * B * D.DELTA), tp[target](tin), tp[target](sq).unsafe_offset(j * B),
        )
        var lg = talloc[target](B, ctx)
        wm.term.forward[target, B](TensorRefs[1, _, DT](tin), lg, ctx)
        comptime if target == "gpu":
            ctx.value().enqueue_copy(lgall.dev.value().create_sub_buffer[DT](j * B, B), lg.dev.value())
        else:
            for b in range(B):
                lgall.data[j * B + b] = lg.data[b]
        xs.append(x^)
        ds.append(dl^)
        tins.append(tin^)
    # the scalars, on the host
    comptime if target == "gpu":
        sq.download_enqueue(ctx.value())
        lgall.download_enqueue(ctx.value())
        w.done.download_enqueue(ctx.value())
        ctx.value().synchronize()
        sq.download_finalize()
        lgall.download_finalize()
        w.done.download_finalize()
    var state_tot = 0.0
    var bce_tot = 0.0
    var nse1 = 0.0
    var nsek = 0.0
    for j in range(K):
        var nse = 0.0
        for b in range(B):
            var v = Float64(sq.data[j * B + b])
            state_tot += v / Float64(B)
            nse += v / Float64(D.DELTA * B)
            var l = Float64(lgall.data[j * B + b])
            var y = Float64(w.done.data[j * B + b])
            bce_tot += (max(l, 0.0) - l * y + log(1.0 + exp(-abs(l)))) / Float64(B)
        if j == 0:
            nse1 = nse
        nsek = nse
    var out = SMWMLoss(state_tot / Float64(K) + w_term * bce_tot / Float64(K), state_tot / Float64(K), bce_tot / Float64(K), nse1, nsek)
    if not grad:
        return out^

    # ── the reverse scan ──
    var gs = talloc[target](B * S, ctx)                 # d loss / d s_{j+1}
    var gprev = talloc[target](B * S, ctx)
    var glg = talloc[target](B, ctx)
    var gtin = talloc[target](B * SM_TERM_IN, ctx)
    var gdl = talloc[target](B * D.DELTA, ctx)
    var gx = talloc[target](B * D.WM_IN, ctx)
    var relg = talloc[target](B, ctx)
    var redl = talloc[target](B * D.DELTA, ctx)
    var cs = Scalar[DT](2.0 / Float64(K * B))
    var cb = Scalar[DT](w_term / Float64(K * B))
    var j = K - 1
    while j >= 0:
        var s1 = tp[target](sp).unsafe_offset((j + 1) * B * S)
        sm_wm_err_back[target, R](
            ctx, B, s1, tp[target](w.states).unsafe_offset((j + 1) * B * S), tp[target](errs).unsafe_offset(j * B * D.DELTA),
            tp[target](lgall).unsafe_offset(j * B), tp[target](w.done).unsafe_offset(j * B), cs, cb, tp[target](gs), tp[target](glg),
        )
        # ⚠ RE-RUN THE STEP'S FORWARD FIRST: the fused Linear+SiLU leaves keep
        # the activations of their LAST forward call, which is step k − 1's.
        # A `vjp` at step j against them is the wrong gradient (measured: term
        # head cosine 0.98 against torch before this line).
        wm.term.forward[target, B](TensorRefs[1, _, DT](tins[j]), relg, ctx)
        wm.term.vjp[target, B](TensorRefs[1, _, DT](tins[j]), glg, TensorRefs[1, _, DT](gtin), ctx)
        # through s_{j+1} = integrate(s_j, dynamics([canonical(s_j), a_j, ξ]))
        sm_wm_dyn_back[target, R](
            ctx, B, tp[target](tins[j]), tp[target](gtin), tp[target](sp).unsafe_offset(j * B * S), s1,
            tp[target](ds[j]), tp[target](gs), tp[target](gprev), tp[target](gdl),
        )
        wm.dyn.forward[target, B](TensorRefs[1, _, DT](xs[j]), redl, ctx)   # see the term head's note
        wm.dyn.vjp[target, B](TensorRefs[1, _, DT](xs[j]), gdl, TensorRefs[1, _, DT](gx), ctx)
        # s_0 is data: no canonical VJP at j = 0
        sm_canon_back[target, R](ctx, B, D.WM_IN, tp[target](sp).unsafe_offset(j * B * S), tp[target](gx), tp[target](gprev), tp[target](gs), j > 0)
        j -= 1
    return out^
