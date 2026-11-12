# +--------------------------------------------------------------------------+ #
# | SuperMimic with an object: the per-lane phases of the object's k-step loss
# +--------------------------------------------------------------------------+ #
"""The object block's phases, in `lanes.mojo`'s form: one `_*_lane` body per
phase and a runner `phase[target]` that loops on "cpu" and launches one kernel
on "gpu". They sit BETWEEN the robot's own phases in
`object_model.sm_obj_window_loss`; the widths are `SMObjDims[R]`.

    forward, step j     sm_obj_in     xo = [x (the robot's input), feat(s, o), ctx]
                        sm_obj_step   dl += dlo[:DELTA]  (the robot's delta gets the
                                      object network's correction: `dl` is the TOTAL
                                      from here on);  o1 = integrate(s, o, dlo[DELTA:]);
                                      err, sq against the logged o1
    reverse, step j     sm_obj_err_back   go += the error's VJP at o1
                        sm_obj_dyn_back   through o1 = integrate(s, o, ·): goprev
                                          (written), gprev += (the heading), and the
                                          object network's output cotangent
                                          gdlo = [gdl (the robot's), ∂/∂ its own 12]
                        sm_obj_feat_back  gx += gxo[:WM_IN]; the features' VJP into
                                          gprev and goprev; go = goprev

Same kernel-safety rules as `lanes.mojo`.
"""

from std.math import ceildiv

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import SM_TPB
from noeira.deep_agents.supermimic.object_spec import (
    SM_OBJ, SM_OBJ_CTX, SM_OBJ_DELTA, SM_OBJ_FEAT, sm_obj_ctx_normalize, sm_obj_error_lane, sm_obj_error_vjp_lane,
    sm_obj_feat_lane, sm_obj_feat_vjp_lane, sm_obj_integrate_lane, sm_obj_integrate_vjp_lane,
)
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.nn.constants import DT


struct SMObjDims[R: TrackingRobot]:
    """The object network's widths on robot R."""
    comptime IN = SMDims[Self.R].WM_IN + SM_OBJ_FEAT + SM_OBJ_CTX      # [the robot's input | feat | ctx]
    comptime OUT = SMDims[Self.R].DELTA + SM_OBJ_DELTA                  # [a correction to the robot's delta | the object's]


@always_inline
def _grid(n: Int) -> Int:
    return ceildiv(n, SM_TPB)


# ── forward ─────────────────────────────────────────────────────────────────
@always_inline
def _obj_in_lane[R: TrackingRobot](b: Int, x: FP[DT], s: FP[DT], o: FP[DT], octx: FP[DT], xo: FP[DT]):
    """xo[b] = [x[b], feat(s[b], o[b]), normalise(ctx[b])]."""
    comptime D = SMDims[R]
    comptime IN = SMObjDims[R].IN
    var xr = xo.unsafe_offset(b * IN)
    for i in range(D.WM_IN):
        xr[unsafe_offset=i] = x[unsafe_offset=b * D.WM_IN + i]
    sm_obj_feat_lane[DT, R](s.unsafe_offset(b * D.S), o.unsafe_offset(b * SM_OBJ), xr.unsafe_offset(D.WM_IN))
    comptime for k in range(SM_OBJ_CTX):
        xr[unsafe_offset=D.WM_IN + SM_OBJ_FEAT + k] = sm_obj_ctx_normalize[DT](k, octx[unsafe_offset=b * SM_OBJ_CTX + k])


def _k_obj_in[R: TrackingRobot](nb: Int64, x: FP[DT], s: FP[DT], o: FP[DT], octx: FP[DT], xo: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _obj_in_lane[R](b, x, s, o, octx, xo)


def sm_obj_in[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int, x: FP[DT], s: FP[DT], o: FP[DT], octx: FP[DT], xo: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_obj_in[R]](Int64(nb), x, s, o, octx, xo, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _obj_in_lane[R](b, x, s, o, octx, xo)


@always_inline
def _obj_step_lane[R: TrackingRobot](
    b: Int, s: FP[DT], o: FP[DT], dlo: FP[DT], dl: FP[DT], o1: FP[DT], ot1: FP[DT], err: FP[DT], sq: FP[DT]
):
    """dl[b] += dlo[b][:DELTA]; o1[b] = integrate(s[b], o[b], dlo[b][DELTA:]);
    err[b] = error(o1[b], ot1[b]) and its square sum sq[b]."""
    comptime D = SMDims[R]
    comptime OUT = SMObjDims[R].OUT
    var row = dlo.unsafe_offset(b * OUT)
    for d in range(D.DELTA):
        dl[unsafe_offset=b * D.DELTA + d] += row[unsafe_offset=d]
    var o1b = o1.unsafe_offset(b * SM_OBJ)
    sm_obj_integrate_lane[DT, R](s.unsafe_offset(b * D.S), o.unsafe_offset(b * SM_OBJ), row.unsafe_offset(D.DELTA), o1b)
    var eb = err.unsafe_offset(b * SM_OBJ_DELTA)
    sm_obj_error_lane[DT](o1b, ot1.unsafe_offset(b * SM_OBJ), eb)
    var acc = Scalar[DT](0)
    comptime for d in range(SM_OBJ_DELTA):
        acc += eb[unsafe_offset=d] * eb[unsafe_offset=d]
    sq[unsafe_offset=b] = acc


def _k_obj_step[R: TrackingRobot](
    nb: Int64, s: FP[DT], o: FP[DT], dlo: FP[DT], dl: FP[DT], o1: FP[DT], ot1: FP[DT], err: FP[DT], sq: FP[DT]
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _obj_step_lane[R](b, s, o, dlo, dl, o1, ot1, err, sq)


def sm_obj_step[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int,
    s: FP[DT], o: FP[DT], dlo: FP[DT], dl: FP[DT], o1: FP[DT], ot1: FP[DT], err: FP[DT], sq: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_obj_step[R]](
            Int64(nb), s, o, dlo, dl, o1, ot1, err, sq, grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _obj_step_lane[R](b, s, o, dlo, dl, o1, ot1, err, sq)


# ── reverse ─────────────────────────────────────────────────────────────────
@always_inline
def _obj_err_back_lane(b: Int, o1: FP[DT], ot1: FP[DT], err: FP[DT], c: Scalar[DT], go: FP[DT]):
    """go[b] += ∂(c·|err|²/2)/∂o1: `c` holds the loss's 2·w_obj/(K·B)."""
    var eb = err.unsafe_offset(b * SM_OBJ_DELTA)
    comptime for d in range(SM_OBJ_DELTA):
        eb[unsafe_offset=d] = eb[unsafe_offset=d] * c         # err becomes its own cotangent
    sm_obj_error_vjp_lane[DT](o1.unsafe_offset(b * SM_OBJ), ot1.unsafe_offset(b * SM_OBJ), eb, go.unsafe_offset(b * SM_OBJ))


def _k_obj_err_back(nb: Int64, o1: FP[DT], ot1: FP[DT], err: FP[DT], c: Scalar[DT], go: FP[DT]):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _obj_err_back_lane(b, o1, ot1, err, c, go)


def sm_obj_err_back[target: StaticString](
    ctx: Optional[DeviceContext], nb: Int, o1: FP[DT], ot1: FP[DT], err: FP[DT], c: Scalar[DT], go: FP[DT]
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_obj_err_back](Int64(nb), o1, ot1, err, c, go, grid_dim=_grid(nb), block_dim=SM_TPB)
    else:
        for b in range(nb):
            _obj_err_back_lane(b, o1, ot1, err, c, go)


@always_inline
def _obj_dyn_back_lane[R: TrackingRobot](
    b: Int, s: FP[DT], o: FP[DT], dlo: FP[DT], go: FP[DT], gprev: FP[DT], goprev: FP[DT], gdl: FP[DT], gdlo: FP[DT]
):
    """Through o1 = integrate(s, o, dlo[DELTA:]) with cotangent go[b]:
    goprev[b] = ∂/∂o (written), gprev[b] += ∂/∂s, gdlo[b][DELTA:] = ∂/∂ the
    object's delta; and gdlo[b][:DELTA] = gdl[b], the cotangent of the robot's
    TOTAL delta, which the object network's correction shares."""
    comptime D = SMDims[R]
    comptime OUT = SMObjDims[R].OUT
    var gop = goprev.unsafe_offset(b * SM_OBJ)
    comptime for i in range(SM_OBJ):
        gop[unsafe_offset=i] = 0
    var grow = gdlo.unsafe_offset(b * OUT)
    sm_obj_integrate_vjp_lane[DT, R](
        s.unsafe_offset(b * D.S), o.unsafe_offset(b * SM_OBJ), dlo.unsafe_offset(b * OUT + D.DELTA),
        go.unsafe_offset(b * SM_OBJ), gprev.unsafe_offset(b * D.S), gop, grow.unsafe_offset(D.DELTA),
    )
    for d in range(D.DELTA):
        grow[unsafe_offset=d] = gdl[unsafe_offset=b * D.DELTA + d]


def _k_obj_dyn_back[R: TrackingRobot](
    nb: Int64, s: FP[DT], o: FP[DT], dlo: FP[DT], go: FP[DT], gprev: FP[DT], goprev: FP[DT], gdl: FP[DT], gdlo: FP[DT]
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _obj_dyn_back_lane[R](b, s, o, dlo, go, gprev, goprev, gdl, gdlo)


def sm_obj_dyn_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int,
    s: FP[DT], o: FP[DT], dlo: FP[DT], go: FP[DT], gprev: FP[DT], goprev: FP[DT], gdl: FP[DT], gdlo: FP[DT],
) raises:
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_obj_dyn_back[R]](
            Int64(nb), s, o, dlo, go, gprev, goprev, gdl, gdlo, grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _obj_dyn_back_lane[R](b, s, o, dlo, go, gprev, goprev, gdl, gdlo)


@always_inline
def _obj_feat_back_lane[R: TrackingRobot](
    b: Int, s: FP[DT], o: FP[DT], gxo: FP[DT], gx: FP[DT], gprev: FP[DT], goprev: FP[DT], go: FP[DT], use: Int
):
    """gx[b] += gxo[b][:WM_IN] (the object network read the robot's input
    too); when `use`, the features' VJP of gxo[b][WM_IN : WM_IN + 15] into
    gprev[b] and goprev[b]; then go[b] = goprev[b]."""
    comptime D = SMDims[R]
    comptime IN = SMObjDims[R].IN
    var grow = gxo.unsafe_offset(b * IN)
    for i in range(D.WM_IN):
        gx[unsafe_offset=b * D.WM_IN + i] += grow[unsafe_offset=i]
    var gop = goprev.unsafe_offset(b * SM_OBJ)
    if use != 0:
        sm_obj_feat_vjp_lane[DT, R](
            s.unsafe_offset(b * D.S), o.unsafe_offset(b * SM_OBJ), grow.unsafe_offset(D.WM_IN), gprev.unsafe_offset(b * D.S), gop,
        )
    var gob = go.unsafe_offset(b * SM_OBJ)
    comptime for i in range(SM_OBJ):
        gob[unsafe_offset=i] = gop[unsafe_offset=i]


def _k_obj_feat_back[R: TrackingRobot](
    nb: Int64, s: FP[DT], o: FP[DT], gxo: FP[DT], gx: FP[DT], gprev: FP[DT], goprev: FP[DT], go: FP[DT], use: Int64
):
    var b = Int(global_idx.x)
    if b < Int(nb):
        _obj_feat_back_lane[R](b, s, o, gxo, gx, gprev, goprev, go, Int(use))


def sm_obj_feat_back[target: StaticString, R: TrackingRobot](
    ctx: Optional[DeviceContext], nb: Int,
    s: FP[DT], o: FP[DT], gxo: FP[DT], gx: FP[DT], gprev: FP[DT], goprev: FP[DT], go: FP[DT], use: Bool,
) raises:
    var u = 1 if use else 0
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_obj_feat_back[R]](
            Int64(nb), s, o, gxo, gx, gprev, goprev, go, Int64(u), grid_dim=_grid(nb), block_dim=SM_TPB
        )
    else:
        for b in range(nb):
            _obj_feat_back_lane[R](b, s, o, gxo, gx, gprev, goprev, go, u)
