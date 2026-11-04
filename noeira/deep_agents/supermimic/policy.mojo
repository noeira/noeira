# +--------------------------------------------------------------------------+ #
# | SuperMimic M5: policy (goal encoder + decoder), critic, their inputs
# +--------------------------------------------------------------------------+ #
"""The reproduction's `policy/nets.py` on `nn`, and the per-lane input
plumbing around it (`world_model/heads.py`'s history flattening, the critic's
root-error features).

    a = decoder([tanh(encoder(goal ⊙ gscale)),  hist ⊙ oscale])
    V = critic([canonical(s), goal ⊙ gscale, normalise(ξ), e_root(s, ref)])

    goal (G)    = 10 future frames × [q n, q̇ n, grav 3, ω_b 3, v_h 3, h 1],
                  divided per term by (1, 5, 1, 2, 1, 1)          (680 on the G1)
    hist        = the last 10 observations, TERM-MAJOR (mjlab's layout):  (930)
                  [q of obs 0..9, q̇ of obs 0..9, ω 0..9, grav 0..9, a 0..9],
                  divided per term by (1, 5, 2, 1, 1)
    e_root (4)  = [R_yaw(ref)⁻¹ (p − p_ref) (3), wrap(yaw(q) − yaw(q_ref))]

Paper sizes: encoder 680 → 2048-1024-512-256 → 32 (tanh), decoder
(32 + 930) → 2048-1024-512-256 → 29 (last layer orthogonal, gain 0.01),
critic (71 + 680 + 33 + 4) → 256-128 → 1. All SiLU.
"""

from std.math import atan2, floor, pi

from noeira.deep_agents.supermimic.kinematics import FP, Q4, dot3, qapply_vjp, qconj
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.heads import qapply_inv, yaw_quat, yaw_quat_vjp
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.param import ParamVisitor, ParamWalkable
from noeira.nn.core.walkers import join_name
from noeira.nn.primitives.activations import Tanh
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_swish import LinearSwish

from max.gpu.host import DeviceContext


comptime SMEncoder[R: TrackingRobot, H1: Int, H2: Int, H3: Int, H4: Int] = Sequential[
    LinearSwish[SMDims[R].GOAL, H1], LinearSwish[H1, H2], LinearSwish[H2, H3], LinearSwish[H3, H4],
    Linear[H4, SMDims[R].LATENT], Tanh[SMDims[R].LATENT],
]
comptime SMDecoder[R: TrackingRobot, H1: Int, H2: Int, H3: Int, H4: Int] = Sequential[
    LinearSwish[SMDims[R].DEC_IN, H1], LinearSwish[H1, H2], LinearSwish[H2, H3], LinearSwish[H3, H4],
    Linear[H4, R.NDOF],
]
comptime SMCritic[R: TrackingRobot, C1: Int, C2: Int] = Sequential[
    LinearSwish[SMDims[R].CRITIC_IN, C1], LinearSwish[C1, C2], Linear[C2, 1],
]


struct SMPolicy[R: TrackingRobot, E1: Int, E2: Int, E3: Int, E4: Int, D1: Int, D2: Int, D3: Int, D4: Int](ParamWalkable):
    """The motion encoder and the decoder, walked as one model (the policy's
    clip spans both). Params `encoder.<i>.*`, `decoder.<i>.*`."""
    var enc: SMEncoder[Self.R, Self.E1, Self.E2, Self.E3, Self.E4]
    var dec: SMDecoder[Self.R, Self.D1, Self.D2, Self.D3, Self.D4]

    def __init__(out self, var enc: SMEncoder[Self.R, Self.E1, Self.E2, Self.E3, Self.E4], var dec: SMDecoder[Self.R, Self.D1, Self.D2, Self.D3, Self.D4]):
        self.enc = enc^
        self.dec = dec^

    def __init__(out self, *, deinit move: Self):
        self.enc = move.enc^
        self.dec = move.dec^

    @staticmethod
    def make[INIT: Initializer, target: StaticString = "cpu"](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self(
            SMEncoder[Self.R, Self.E1, Self.E2, Self.E3, Self.E4].make[target, INIT](ctx),
            SMDecoder[Self.R, Self.D1, Self.D2, Self.D3, Self.D4].make[target, INIT](ctx),
        )

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext], prefix: String = String("")
    ) raises:
        self.enc.for_each_param[target](visitor, ctx, join_name(prefix, String("encoder")))
        self.dec.for_each_param[target](visitor, ctx, join_name(prefix, String("decoder")))

    def zero_grad[target: StaticString](mut self, ctx: Optional[DeviceContext]) raises:
        self.enc.zero_grad[target](ctx)
        self.dec.zero_grad[target](ctx)

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext], prefix: String = String("")
    ) raises:
        self.enc.for_each_state[target](visitor, ctx, join_name(prefix, String("encoder")))
        self.dec.for_each_state[target](visitor, ctx, join_name(prefix, String("decoder")))


@always_inline
def sm_goal_scale[n: Int, dt: DType = DType.float64](i: Int) -> Scalar[dt]:
    """1 / scale of goal entry `i` (per frame of 2n + 10: q 1, q̇ 5, grav 1,
    ω_b 2, v_h 1, h 1)."""
    var k = i % (2 * n + 10)
    if k < n:
        return 1.0
    if k < 2 * n:
        return 0.2
    if k < 2 * n + 3:
        return 1.0
    if k < 2 * n + 6:
        return 0.5
    return 1.0


@always_inline
def sm_hist_flat_index[n: Int](h: Int, o: Int) -> Int:
    """Where entry `o` (0 .. 3n+5) of history slot `h` (0 oldest .. 9) sits in
    the term-major flat history: each term's 10 slots are contiguous, terms in
    obs order (q n, q̇ n, ω 3, grav 3, a n)."""
    var start: Int
    var d: Int
    if o < n:
        start = 0
        d = n
    elif o < 2 * n:
        start = n
        d = n
    elif o < 2 * n + 3:
        start = 2 * n
        d = 3
    elif o < 2 * n + 6:
        start = 2 * n + 3
        d = 3
    else:
        start = 2 * n + 6
        d = n
    return 10 * start + h * d + (o - start)


@always_inline
def sm_obs_scale[n: Int, dt: DType = DType.float64](o: Int) -> Scalar[dt]:
    """1 / scale of observation entry `o`: q 1, q̇ 5, ω 2, grav 1, a 1."""
    if o < n:
        return 1.0
    if o < 2 * n:
        return 0.2
    if o < 2 * n + 3:
        return 0.5
    return 1.0


@always_inline
def sm_obs_noise_half[n: Int, dt: DType = DType.float64](o: Int) -> Scalar[dt]:
    """Table A.1's uniform noise half-width of observation entry `o`:
    q 0.01, q̇ 0.5, ω 0.2, grav 0.05, a 0."""
    if o < n:
        return 0.01
    if o < 2 * n:
        return 0.5
    if o < 2 * n + 3:
        return 0.2
    if o < 2 * n + 6:
        return 0.05
    return 0.0


# ── the critic's privileged root-error features ─────────────────────────────
@always_inline
def _heading_yaw[dt: DType](q: Q4[dt]) -> Scalar[dt] where dt.is_floating_point():
    return atan2((q[0] * q[3] + q[1] * q[2]) * 2, Scalar[dt](1) - (q[2] * q[2] + q[3] * q[3]) * 2)


@always_inline
def _heading_yaw_vjp[dt: DType](q: Q4[dt], g: Scalar[dt]) -> Q4[dt] where dt.is_floating_point():
    var Y = (q[0] * q[3] + q[1] * q[2]) * 2
    var X = Scalar[dt](1) - (q[2] * q[2] + q[3] * q[3]) * 2
    var r2 = X * X + Y * Y
    var gY = g * X / r2
    var gX = -g * Y / r2
    return Q4[dt](gY * 2 * q[3], gY * 2 * q[2], gY * 2 * q[1] - gX * 4 * q[2], gY * 2 * q[0] - gX * 4 * q[3])


@always_inline
def _wrap_to_pi[dt: DType](a: Scalar[dt]) -> Scalar[dt] where dt.is_floating_point():
    """torch's `(a + π) % (2π) − π` (the remainder takes the divisor's sign)."""
    var two = Scalar[dt](2 * pi)
    var x = a + Scalar[dt](pi)
    var r = x - two * floor(x / two)
    return r - Scalar[dt](pi)


def sm_root_error_lane[dt: DType](s: FP[dt], rf_root_p: Q4[dt], rf_root_q: Q4[dt], e_out: FP[dt]) where dt.is_floating_point():
    """`AnalyticHeads.root_error_features`: [R_yaw(ref)⁻¹ (p − p_ref), wrap(yaw(q) − yaw(q_ref))]
    (yaw of the RAW state quaternion)."""
    var dp = qapply_inv(yaw_quat(rf_root_q), Q4[dt](s[unsafe_offset=0], s[unsafe_offset=1], s[unsafe_offset=2], 0) - rf_root_p)
    var q = Q4[dt](s[unsafe_offset=3], s[unsafe_offset=4], s[unsafe_offset=5], s[unsafe_offset=6])
    e_out[unsafe_offset=0] = dp[0]
    e_out[unsafe_offset=1] = dp[1]
    e_out[unsafe_offset=2] = dp[2]
    e_out[unsafe_offset=3] = _wrap_to_pi(_heading_yaw(q) - _heading_yaw(rf_root_q))


def sm_root_error_vjp_lane[dt: DType](s: FP[dt], rf_root_q: Q4[dt], g: FP[dt], g_s: FP[dt]) where dt.is_floating_point():
    """`g_s += ∂/∂s <g, root_error(s, ·)>` (ADDS; the wrap is piecewise identity)."""
    var yq = yaw_quat(rf_root_q)
    # ∂/∂v of qapply_inv(yq, v) = qapply_vjp(conj(yq), v, g)[1]
    var gv = qapply_vjp(qconj(yq), Q4[dt](0), Q4[dt](g[unsafe_offset=0], g[unsafe_offset=1], g[unsafe_offset=2], 0))[1]
    g_s[unsafe_offset=0] += gv[0]
    g_s[unsafe_offset=1] += gv[1]
    g_s[unsafe_offset=2] += gv[2]
    var q = Q4[dt](s[unsafe_offset=3], s[unsafe_offset=4], s[unsafe_offset=5], s[unsafe_offset=6])
    var gq = _heading_yaw_vjp(q, g[unsafe_offset=3])
    comptime for k in range(4):
        g_s[unsafe_offset=3 + k] += gq[k]
