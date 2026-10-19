"""PPOObjective[ACT_] — quaternary Module for the PPO clipped-surrogate loss.

Inputs (ARITY=4):
  - actor_output  [BATCH, 2*ACT]  from GaussianHead: [mu | log_std].
  - action        [BATCH, ACT]    unbounded sample stored at rollout time.
  - old_log_prob  [BATCH, 1]      log p_θ_old(action | s) at rollout.
  - advantage     [BATCH, 1]      GAE advantage, normalised per minibatch.

Output:
  - loss_per_b    [BATCH, 1]      per-sample (un-averaged) PPO loss.

The 1/BATCH mean factor lives in the seed gradient, not inside this kernel.

Math (per sample b):
    new_log_prob = Σ_j  -0.5 * (LOG_2PI + 2·ls_j + ((a_j-mu_j)/std_j)²)
    diff         = clamp(new_log_prob - old_log_prob, ±20)
    ratio        = exp(diff)
    unclipped    = ratio * adv ;  clipped = clip(ratio, 1±ε) * adv
    entropy      = Σ_j  0.5 * (LOG_2PI + 1 + 2·ls_j)
    loss_per_b   = -min(unclipped, clipped) - entropy_coef * entropy

grad_action / grad_old_log_prob / grad_advantage are identically zero
(non-differentiable rollout-time inputs); per-element grad clip ±10.

STORAGE migration: standalone arity-4 leaf (NOT in a ComputeGraph — the graph
dispatch tops out at arity 3). `vjp` reads the four inputs from `forward_input`
(no cached pointers). The CPU loops + the two GPU kernels are unchanged from the
legacy leaf; only the storage surface (`TensorRefs[4, o]` + owned `Tensor` out +
`.lt` views) differs.
"""

from std.math import exp
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] = 2.0
comptime LOG_PROB_DIFF_MAX: Scalar[DT] = 20.0
comptime GRAD_CLIP: Scalar[DT] = 10.0
comptime EPS_STD: Scalar[DT] = 1e-6
comptime LOG_2PI: Scalar[DT] = 1.8378770664093453


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one thread per batch row (per-row math is purely local).
# ──────────────────────────────────────────────────────────────────────


def _ppo_forward_kernel[ACT: Int, BATCH: Int](
    ao: Tile[DT, BATCH, 2 * ACT],
    act: Tile[DT, BATCH, ACT],
    olp: Tile[DT, BATCH, 1],
    adv: Tile[DT, BATCH, 1],
    loss_out: Tile[DT, BATCH, 1],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var new_log_prob: Scalar[DT] = 0.0
    var entropy: Scalar[DT] = 0.0
    for j in range(ACT):
        var mu = rebind[Scalar[DT]](ao[b, j])
        var ls = rebind[Scalar[DT]](ao[b, ACT + j])
        if ls < LOG_STD_MIN:
            ls = LOG_STD_MIN
        elif ls > LOG_STD_MAX:
            ls = LOG_STD_MAX
        var std = exp(ls)
        var a = rebind[Scalar[DT]](act[b, j])
        var z = (a - mu) / (std + EPS_STD)
        new_log_prob += Scalar[DT](-0.5) * (LOG_2PI + Scalar[DT](2.0) * ls + z * z)
        entropy += Scalar[DT](0.5) * (LOG_2PI + Scalar[DT](1.0) + Scalar[DT](2.0) * ls)
    var olp_b = rebind[Scalar[DT]](olp[b, 0])
    var adv_b = rebind[Scalar[DT]](adv[b, 0])
    var diff = new_log_prob - olp_b
    if diff > LOG_PROB_DIFF_MAX:
        diff = LOG_PROB_DIFF_MAX
    elif diff < -LOG_PROB_DIFF_MAX:
        diff = -LOG_PROB_DIFF_MAX
    var ratio = exp(diff)
    var clipped_ratio = ratio
    if clipped_ratio < Scalar[DT](1.0) - clip_eps:
        clipped_ratio = Scalar[DT](1.0) - clip_eps
    elif clipped_ratio > Scalar[DT](1.0) + clip_eps:
        clipped_ratio = Scalar[DT](1.0) + clip_eps
    var unclipped_obj = ratio * adv_b
    var clipped_obj = clipped_ratio * adv_b
    var min_obj: Scalar[DT] = unclipped_obj
    if clipped_obj < unclipped_obj:
        min_obj = clipped_obj
    loss_out[b, 0] = -min_obj - entropy_coef * entropy


def _ppo_vjp_kernel[ACT: Int, BATCH: Int](
    ao: Tile[DT, BATCH, 2 * ACT],
    act: Tile[DT, BATCH, ACT],
    olp: Tile[DT, BATCH, 1],
    adv: Tile[DT, BATCH, 1],
    go: Tile[DT, BATCH, 1],
    gi0: Tile[DT, BATCH, 2 * ACT],
    gi1: Tile[DT, BATCH, ACT],
    gi2: Tile[DT, BATCH, 1],
    gi3: Tile[DT, BATCH, 1],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    for j in range(ACT):
        gi1[b, j] = Scalar[DT](0.0)
    gi2[b, 0] = Scalar[DT](0.0)
    gi3[b, 0] = Scalar[DT](0.0)
    var new_log_prob: Scalar[DT] = 0.0
    for j in range(ACT):
        var mu = rebind[Scalar[DT]](ao[b, j])
        var ls = rebind[Scalar[DT]](ao[b, ACT + j])
        if ls < LOG_STD_MIN:
            ls = LOG_STD_MIN
        elif ls > LOG_STD_MAX:
            ls = LOG_STD_MAX
        var std = exp(ls)
        var a = rebind[Scalar[DT]](act[b, j])
        var z = (a - mu) / (std + EPS_STD)
        new_log_prob += Scalar[DT](-0.5) * (LOG_2PI + Scalar[DT](2.0) * ls + z * z)
    var olp_b = rebind[Scalar[DT]](olp[b, 0])
    var adv_b = rebind[Scalar[DT]](adv[b, 0])
    var diff = new_log_prob - olp_b
    if diff > LOG_PROB_DIFF_MAX:
        diff = LOG_PROB_DIFF_MAX
    elif diff < -LOG_PROB_DIFF_MAX:
        diff = -LOG_PROB_DIFF_MAX
    var ratio = exp(diff)
    var clipped_ratio = ratio
    if clipped_ratio < Scalar[DT](1.0) - clip_eps:
        clipped_ratio = Scalar[DT](1.0) - clip_eps
    elif clipped_ratio > Scalar[DT](1.0) + clip_eps:
        clipped_ratio = Scalar[DT](1.0) + clip_eps
    var unclipped_obj = ratio * adv_b
    var clipped_obj = clipped_ratio * adv_b
    var is_clipped = clipped_obj < unclipped_obj
    var go_b = rebind[Scalar[DT]](go[b, 0])
    for j in range(ACT):
        if is_clipped:
            gi0[b, j] = Scalar[DT](0.0)
            gi0[b, ACT + j] = -entropy_coef * Scalar[DT](1.0) * go_b
        else:
            var mu = rebind[Scalar[DT]](ao[b, j])
            var ls = rebind[Scalar[DT]](ao[b, ACT + j])
            if ls < LOG_STD_MIN:
                ls = LOG_STD_MIN
            elif ls > LOG_STD_MAX:
                ls = LOG_STD_MAX
            var std = exp(ls)
            var a = rebind[Scalar[DT]](act[b, j])
            var z = (a - mu) / (std + EPS_STD)
            var d_lp_d_mu = z / (std + EPS_STD)
            var d_lp_d_ls = z * z - Scalar[DT](1.0)
            var gmu = -adv_b * ratio * d_lp_d_mu * go_b
            var gls = (-adv_b * ratio * d_lp_d_ls - entropy_coef) * go_b
            if gmu > GRAD_CLIP:
                gmu = GRAD_CLIP
            elif gmu < -GRAD_CLIP:
                gmu = -GRAD_CLIP
            if gls > GRAD_CLIP:
                gls = GRAD_CLIP
            elif gls < -GRAD_CLIP:
                gls = -GRAD_CLIP
            gi0[b, j] = gmu
            gi0[b, ACT + j] = gls


struct PPOObjective[ACT_: Int](Module):
    """PPO clipped-surrogate + entropy bonus as a quaternary Module."""

    comptime ARITY: Int = 4
    comptime IN_DIMS = Self._build_in_dims()
    comptime OUT_DIM: Int = 1

    @staticmethod
    def _build_in_dims() -> Array[Int, 4]:
        var d = Array[Int, 4](fill=1)
        d[0] = 2 * Self.ACT_
        d[1] = Self.ACT_
        return d^

    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]

    def __init__(out self):
        self.clip_eps = Scalar[DT](0.2)
        self.entropy_coef = Scalar[DT](0.0)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOObjective: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "clip_eps":
            self.clip_eps = value
        elif ATTR == "entropy_coef":
            self.entropy_coef = value

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime ACT = Self.ACT_
        ref ao = inputs[0]
        ref act = inputs[1]
        ref olp = inputs[2]
        ref adv = inputs[3]
        comptime if target == "cpu":
            out.ensure(B * 1)
            var ao_v = TileTensor(ao.data, row_major[B, 2 * ACT]())
            var act_v = TileTensor(act.data, row_major[B, ACT]())
            var olp_v = TileTensor(olp.data, row_major[B, 1]())
            var adv_v = TileTensor(adv.data, row_major[B, 1]())
            var out_v = TileTensor(out.data, row_major[B, 1]())
            for b in range(B):
                var new_log_prob: Scalar[DT] = 0.0
                var entropy: Scalar[DT] = 0.0
                for j in range(ACT):
                    var mu = ao_v[b, j]
                    var ls = ao_v[b, ACT + j]
                    if ls < LOG_STD_MIN:
                        ls = LOG_STD_MIN
                    elif ls > LOG_STD_MAX:
                        ls = LOG_STD_MAX
                    var std = exp(ls)
                    var a = act_v[b, j]
                    var z = (a - mu) / (std + EPS_STD)
                    new_log_prob += Scalar[DT](-0.5) * (
                        LOG_2PI + Scalar[DT](2.0) * ls + z * z
                    )
                    entropy += Scalar[DT](0.5) * (
                        LOG_2PI + Scalar[DT](1.0) + Scalar[DT](2.0) * ls
                    )
                var olp_b = olp_v[b, 0]
                var adv_b = adv_v[b, 0]
                var diff = new_log_prob - olp_b
                if diff > LOG_PROB_DIFF_MAX:
                    diff = LOG_PROB_DIFF_MAX
                elif diff < -LOG_PROB_DIFF_MAX:
                    diff = -LOG_PROB_DIFF_MAX
                var ratio = exp(diff)
                var clipped_ratio = ratio
                if clipped_ratio < Scalar[DT](1.0) - self.clip_eps:
                    clipped_ratio = Scalar[DT](1.0) - self.clip_eps
                elif clipped_ratio > Scalar[DT](1.0) + self.clip_eps:
                    clipped_ratio = Scalar[DT](1.0) + self.clip_eps
                var unclipped_obj = ratio * adv_b
                var clipped_obj = clipped_ratio * adv_b
                var min_obj: Scalar[DT] = unclipped_obj
                if clipped_obj < unclipped_obj:
                    min_obj = clipped_obj
                out_v[b, 0] = -min_obj - self.entropy_coef * entropy
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * 1)
            comptime n_blocks = (B + TPB - 1) // TPB
            c.enqueue_function[_ppo_forward_kernel[ACT, B]](
                ao.tt["gpu", B, 2 * ACT](),
                act.tt["gpu", B, ACT](),
                olp.tt["gpu", B, 1](),
                adv.tt["gpu", B, 1](),
                out.tt["gpu", B, 1](),
                self.clip_eps, self.entropy_coef,
                grid_dim=n_blocks, block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime ACT = Self.ACT_
        ref ao = forward_input[0]
        ref act = forward_input[1]
        ref olp = forward_input[2]
        ref adv = forward_input[3]
        ref gi0 = grad_inputs[0]
        ref gi1 = grad_inputs[1]
        ref gi2 = grad_inputs[2]
        ref gi3 = grad_inputs[3]
        comptime if target == "cpu":
            gi0.ensure(B * 2 * ACT)
            gi1.ensure(B * ACT)
            gi2.ensure(B * 1)
            gi3.ensure(B * 1)
            var ao_v = TileTensor(ao.data, row_major[B, 2 * ACT]())
            var act_v = TileTensor(act.data, row_major[B, ACT]())
            var olp_v = TileTensor(olp.data, row_major[B, 1]())
            var adv_v = TileTensor(adv.data, row_major[B, 1]())
            var go_v = TileTensor(grad_output.data, row_major[B, 1]())
            var gi0_v = TileTensor(gi0.data, row_major[B, 2 * ACT]())
            var gi1_v = TileTensor(gi1.data, row_major[B, ACT]())
            var gi2_v = TileTensor(gi2.data, row_major[B, 1]())
            var gi3_v = TileTensor(gi3.data, row_major[B, 1]())
            for b in range(B):
                for j in range(ACT):
                    gi1_v[b, j] = Scalar[DT](0.0)
                gi2_v[b, 0] = Scalar[DT](0.0)
                gi3_v[b, 0] = Scalar[DT](0.0)
                var new_log_prob: Scalar[DT] = 0.0
                for j in range(ACT):
                    var mu = ao_v[b, j]
                    var ls = ao_v[b, ACT + j]
                    if ls < LOG_STD_MIN:
                        ls = LOG_STD_MIN
                    elif ls > LOG_STD_MAX:
                        ls = LOG_STD_MAX
                    var std = exp(ls)
                    var a = act_v[b, j]
                    var z = (a - mu) / (std + EPS_STD)
                    new_log_prob += Scalar[DT](-0.5) * (
                        LOG_2PI + Scalar[DT](2.0) * ls + z * z
                    )
                var olp_b = olp_v[b, 0]
                var adv_b = adv_v[b, 0]
                var diff = new_log_prob - olp_b
                if diff > LOG_PROB_DIFF_MAX:
                    diff = LOG_PROB_DIFF_MAX
                elif diff < -LOG_PROB_DIFF_MAX:
                    diff = -LOG_PROB_DIFF_MAX
                var ratio = exp(diff)
                var clipped_ratio = ratio
                if clipped_ratio < Scalar[DT](1.0) - self.clip_eps:
                    clipped_ratio = Scalar[DT](1.0) - self.clip_eps
                elif clipped_ratio > Scalar[DT](1.0) + self.clip_eps:
                    clipped_ratio = Scalar[DT](1.0) + self.clip_eps
                var unclipped_obj = ratio * adv_b
                var clipped_obj = clipped_ratio * adv_b
                var is_clipped = clipped_obj < unclipped_obj
                var go_b = go_v[b, 0]
                for j in range(ACT):
                    if is_clipped:
                        gi0_v[b, j] = Scalar[DT](0.0)
                        gi0_v[b, ACT + j] = -self.entropy_coef * Scalar[DT](1.0) * go_b
                    else:
                        var mu = ao_v[b, j]
                        var ls = ao_v[b, ACT + j]
                        if ls < LOG_STD_MIN:
                            ls = LOG_STD_MIN
                        elif ls > LOG_STD_MAX:
                            ls = LOG_STD_MAX
                        var std = exp(ls)
                        var a = act_v[b, j]
                        var z = (a - mu) / (std + EPS_STD)
                        var d_lp_d_mu = z / (std + EPS_STD)
                        var d_lp_d_ls = z * z - Scalar[DT](1.0)
                        var gmu = -adv_b * ratio * d_lp_d_mu * go_b
                        var gls = (
                            -adv_b * ratio * d_lp_d_ls - self.entropy_coef
                        ) * go_b
                        if gmu > GRAD_CLIP:
                            gmu = GRAD_CLIP
                        elif gmu < -GRAD_CLIP:
                            gmu = -GRAD_CLIP
                        if gls > GRAD_CLIP:
                            gls = GRAD_CLIP
                        elif gls < -GRAD_CLIP:
                            gls = -GRAD_CLIP
                        gi0_v[b, j] = gmu
                        gi0_v[b, ACT + j] = gls
        else:
            var c = ctx.value()
            gi0.ensure_gpu(c, B * 2 * ACT)
            gi1.ensure_gpu(c, B * ACT)
            gi2.ensure_gpu(c, B * 1)
            gi3.ensure_gpu(c, B * 1)
            comptime n_blocks = (B + TPB - 1) // TPB
            c.enqueue_function[_ppo_vjp_kernel[ACT, B]](
                ao.tt["gpu", B, 2 * ACT](),
                act.tt["gpu", B, ACT](),
                olp.tt["gpu", B, 1](),
                adv.tt["gpu", B, 1](),
                grad_output.tt["gpu", B, 1](),
                gi0.tt["gpu", B, 2 * ACT](),
                gi1.tt["gpu", B, ACT](),
                gi2.tt["gpu", B, 1](),
                gi3.tt["gpu", B, 1](),
                self.clip_eps, self.entropy_coef,
                grid_dim=n_blocks, block_dim=TPB,
            )
