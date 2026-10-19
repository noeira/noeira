"""RSample[ACT] — squashed-Gaussian reparameterized sample (storage surface).

The heart of SAC's stochastic policy. Transformed from legacy
`deep_agents.primitives.RSample` + the canonical squashed-Gaussian math
(`deep_agents.loss.squashed_gaussian`), inlined here so `nn/storage` is
self-contained.

  input  (actor_output)  [BATCH, 2*ACT]   packed [mu | log_std]
  output                 [BATCH, ACT+1]   packed [action | log_prob]

  action[b,j] = action_scale · tanh(mu_j + exp(clamp(log_std_j)) · z_j)
  log_prob[b] = Σ_j ( -½ z_j² - log_std_j - ½ log(2π) - log(scale·(1-y²)+ε) )

Fresh reparam noise z ~ N(0,1) is drawn each forward (CPU host Box-Muller /
shared GPU Philox `_box_muller_kernel_dev` via `lt_gpu` — no raw ptr) and cached
(z is random, not recomputable); the storage `vjp` gets the actor_output as
`forward_input`, so nothing else is cached. No params.

`action_scale` is a public field (env action bound); `noise_scale`-style eval is
N/A (SAC eval uses the mean action — a separate path).
"""

from std.math import exp, log, tanh as ftanh
from std.random import random_float64
from std.math import sqrt as fsqrt, log as flog, cos as fcos, pi
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.random.box_muller import (
    _box_muller_kernel_dev,
    advance_rng_offset_kernel,
)
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] = 2.0
comptime EPS_TANH_CORR: Scalar[DT] = 1e-6
comptime LOG_2PI: Scalar[DT] = 1.8378770664093453


def _clamp_ls(ls: Scalar[DT]) -> Scalar[DT]:
    if ls < LOG_STD_MIN:
        return LOG_STD_MIN
    elif ls > LOG_STD_MAX:
        return LOG_STD_MAX
    return ls


def _draw_normal_cpu(mut buf: List[Scalar[DT]], n: Int):
    """Fill buf[0:n] with iid N(0,1) via Box-Muller (both branches)."""
    var k = 0
    while k + 1 < n:
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var r = fsqrt(Float64(-2.0) * flog(u1))
        var theta = Float64(2.0) * pi * u2
        buf[k] = Scalar[DT](r * fcos(theta))
        buf[k + 1] = Scalar[DT](r * fcos(theta + 0.5 * pi))
        k += 2
    if k < n:
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var r = fsqrt(Float64(-2.0) * flog(u1))
        var theta = Float64(2.0) * pi * u2
        buf[k] = Scalar[DT](r * fcos(theta))


# ── GPU kernels (packed-direct: action/log_prob in the [BATCH, ACT+1] out) ──
def _rsample_fwd_kernel[
    ACT: Int, BATCH: Int, OUT: Int
](
    actor_output: Tile[DT, BATCH, 2 * ACT],
    z: Tile[DT, BATCH, ACT],
    output: Tile[DT, BATCH, OUT],
    action_scale: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var lp: Scalar[DT] = 0.0
    for j in range(ACT):
        var mu = rebind[Scalar[DT]](actor_output[b, j])
        var ls = _clamp_ls(rebind[Scalar[DT]](actor_output[b, ACT + j]))
        var std = exp(ls)
        var zj = rebind[Scalar[DT]](z[b, j])
        var y = ftanh(mu + std * zj)
        output[b, j] = action_scale * y
        var corr = action_scale * (Scalar[DT](1.0) - y * y) + EPS_TANH_CORR
        lp += (
            Scalar[DT](-0.5) * zj * zj
            - ls
            - Scalar[DT](0.5) * LOG_2PI
            - log(corr)
        )
    output[b, ACT] = lp


def _rsample_bwd_kernel[
    ACT: Int, BATCH: Int, OUT: Int
](
    actor_output: Tile[DT, BATCH, 2 * ACT],
    z: Tile[DT, BATCH, ACT],
    grad_out: Tile[DT, BATCH, OUT],
    grad_in: Tile[DT, BATCH, 2 * ACT],
    action_scale: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * ACT:
        return
    var b = idx // ACT
    var j = idx % ACT
    var mu = rebind[Scalar[DT]](actor_output[b, j])
    var ls_raw = rebind[Scalar[DT]](actor_output[b, ACT + j])
    var ls = _clamp_ls(ls_raw)
    var clamped = (ls_raw < LOG_STD_MIN) or (ls_raw > LOG_STD_MAX)
    var std = exp(ls)
    var zj = rebind[Scalar[DT]](z[b, j])
    var y = ftanh(mu + std * zj)
    var c_om = action_scale * (Scalar[DT](1.0) - y * y)
    var corr = c_om + EPS_TANH_CORR
    var da_dmu = c_om
    var da_dls = c_om * zj * std
    var dlp_dmu = (Scalar[DT](2.0) * y * c_om) / corr
    var dlp_dls = (
        Scalar[DT](-1.0) + (Scalar[DT](2.0) * y * c_om * zj * std) / corr
    )
    var ga = rebind[Scalar[DT]](grad_out[b, j])
    var glp = rebind[Scalar[DT]](grad_out[b, ACT])
    grad_in[b, j] = ga * da_dmu + glp * dlp_dmu
    if clamped:
        grad_in[b, ACT + j] = Scalar[DT](0.0)
    else:
        grad_in[b, ACT + j] = ga * da_dls + glp * dlp_dls


struct RSample[ACT_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=2 * Self.ACT_)
    comptime OUT_DIM = Self.ACT_ + 1  # packed [action | log_prob]

    @staticmethod
    def display_label() -> String:
        return String("RSample")

    var action_scale: Scalar[DT]
    var noise_seed: UInt64
    var noise_offset: TensorImpl[DType.uint64]  # GPU Philox offset
    var z: Tensor  # [BATCH, ACT] cached noise

    def __init__(out self):
        self.action_scale = Scalar[DT](1.0)
        self.noise_seed = UInt64(1)
        self.noise_offset = TensorImpl[DType.uint64]()
        self.z = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var r = Self()
        comptime if target == "gpu":
            var c = ctx.value()
            r.noise_offset.ensure_gpu(c, 1)
            r.noise_offset.dev.value().enqueue_fill(UInt64(0))
        return r^

    def set_noise_seed(mut self, seed: UInt64) raises:
        self.noise_seed = seed
        if self.noise_offset.dev:
            self.noise_offset.dev.value().enqueue_fill(UInt64(0))

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "action_scale":
            self.action_scale = value

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime AO = 2 * Self.ACT_
        comptime OUT = Self.ACT_ + 1
        ref ao = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * OUT)
            self.z.ensure(B * Self.ACT_)
            _draw_normal_cpu(self.z.data, B * Self.ACT_)
            for b in range(B):
                var lp: Scalar[DT] = 0.0
                for j in range(Self.ACT_):
                    var mu = ao.data[b * AO + j]
                    var ls = _clamp_ls(ao.data[b * AO + Self.ACT_ + j])
                    var std = exp(ls)
                    var zj = self.z.data[b * Self.ACT_ + j]
                    var y = ftanh(mu + std * zj)
                    out.data[b * OUT + j] = self.action_scale * y
                    var corr = (
                        self.action_scale * (Scalar[DT](1.0) - y * y)
                        + EPS_TANH_CORR
                    )
                    lp += (
                        Scalar[DT](-0.5) * zj * zj
                        - ls
                        - Scalar[DT](0.5) * LOG_2PI
                        - log(corr)
                    )
                out.data[b * OUT + Self.ACT_] = lp
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * OUT)
            self.z.ensure_gpu(c, B * Self.ACT_)
            comptime NZ = B * Self.ACT_
            comptime nb_z = (NZ + TPB - 1) // TPB
            c.enqueue_function[_box_muller_kernel_dev[NZ]](
                self.z.tt["gpu", NZ](),
                self.noise_seed,
                self.noise_offset.tt["gpu", 1](),
                grid_dim=nb_z,
                block_dim=TPB,
            )
            c.enqueue_function[advance_rng_offset_kernel[((NZ + 1) // 2) * 2]](
                self.noise_offset.tt["gpu", 1](),
                grid_dim=1,
                block_dim=1,
            )
            comptime nb_b = (B + TPB - 1) // TPB
            c.enqueue_function[_rsample_fwd_kernel[Self.ACT_, B, OUT]](
                ao.tt["gpu", B, AO](),
                self.z.tt["gpu", B, Self.ACT_](),
                out.tt["gpu", B, OUT](),
                self.action_scale,
                grid_dim=nb_b,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime AO = 2 * Self.ACT_
        comptime OUT = Self.ACT_ + 1
        ref ao = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * AO)
            for b in range(B):
                var glp = grad_output.data[b * OUT + Self.ACT_]
                for j in range(Self.ACT_):
                    var mu = ao.data[b * AO + j]
                    var ls_raw = ao.data[b * AO + Self.ACT_ + j]
                    var ls = _clamp_ls(ls_raw)
                    var clamped = (ls_raw < LOG_STD_MIN) or (
                        ls_raw > LOG_STD_MAX
                    )
                    var std = exp(ls)
                    var zj = self.z.data[b * Self.ACT_ + j]
                    var y = ftanh(mu + std * zj)
                    var c_om = self.action_scale * (Scalar[DT](1.0) - y * y)
                    var corr = c_om + EPS_TANH_CORR
                    var da_dmu = c_om
                    var da_dls = c_om * zj * std
                    var dlp_dmu = (Scalar[DT](2.0) * y * c_om) / corr
                    var dlp_dls = (
                        Scalar[DT](-1.0)
                        + (Scalar[DT](2.0) * y * c_om * zj * std) / corr
                    )
                    var ga = grad_output.data[b * OUT + j]
                    gin.data[b * AO + j] = ga * da_dmu + glp * dlp_dmu
                    if clamped:
                        gin.data[b * AO + Self.ACT_ + j] = Scalar[DT](0.0)
                    else:
                        gin.data[b * AO + Self.ACT_ + j] = (
                            ga * da_dls + glp * dlp_dls
                        )
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * AO)
            comptime nb = (B * Self.ACT_ + TPB - 1) // TPB
            c.enqueue_function[_rsample_bwd_kernel[Self.ACT_, B, OUT]](
                ao.tt["gpu", B, AO](),
                self.z.tt["gpu", B, Self.ACT_](),
                grad_output.tt["gpu", B, OUT](),
                gin.tt["gpu", B, AO](),
                self.action_scale,
                grid_dim=nb,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
