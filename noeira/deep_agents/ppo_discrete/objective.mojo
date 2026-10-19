"""PPODiscreteObjective[N_] — quaternary Module for the categorical PPO loss.

The discrete (categorical) sibling of `ppo/objective.mojo`'s
`PPOObjective`. Same quaternary storage Module shape — (actor_out, action,
old_log_prob, advantage) — but the actor output is a vector of `N_` logits
(not a [mu | log_std] pair) and the action is a single discrete index.

Inputs (ARITY=4):

  - actor_output     [BATCH, N_]      logits over N_ discrete actions.
  - action           [BATCH, 1]       sampled action INDEX (stored as a float).
  - old_log_prob     [BATCH, 1]       log p_θ_old(a | s) at rollout time.
  - advantage        [BATCH, 1]       GAE advantage, normalised per minibatch.

Output:

  - loss_per_b       [BATCH, 1]       per-sample (un-averaged) PPO loss.

The 1/BATCH mean factor lives in the seed gradient, not inside this kernel —
identical to the continuous objective.

Math (per sample b, with a = action index):
    max_l        = max_j logits_j
    sum_exp      = Σ_j exp(logits_j - max_l)
    log_sum      = log(sum_exp)
    p_j          = exp(logits_j - max_l) / sum_exp
    log_p_j      = (logits_j - max_l) - log_sum
    new_log_prob = log_p_a
    entropy      = -Σ_j p_j · log_p_j
    diff         = clamp(new_log_prob - old_log_prob, ±20)
    ratio        = exp(diff)
    unclipped    = ratio · adv
    clipped      = clip(ratio, 1-ε, 1+ε) · adv
    loss_per_b   = -min(unclipped, clipped) - entropy_coef · entropy

Backward (per sample b, with go = grad_loss_per_b[b]):
    is_clipped = clipped < unclipped
    For each action j:
        d_nlp_j   = (1 if j == a else 0) - p_j
        g_ratio_j = 0                          if is_clipped
                    -adv · ratio · d_nlp_j     otherwise
        g_ent_j   = entropy_coef · p_j · (log_p_j + entropy)
        grad_j    = (g_ratio_j + g_ent_j) · go        (clip ±10)

grad_action / grad_old_log_prob / grad_advantage are identically zero
(non-differentiable rollout-time inputs).

STORAGE migration: standalone arity-4 leaf (NOT in a ComputeGraph — the graph
dispatch tops out at arity 3). `vjp` reads the four inputs from `forward_input`
(no cached pointers). The CPU loops + the two GPU kernels are unchanged from the
legacy leaf; only the storage surface (`TensorRefs[4, o]` + owned `Tensor` out +
`.lt` views) differs.
"""

from std.math import exp, log as flog
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


comptime LOG_PROB_DIFF_MAX: Scalar[DT] = 20.0
comptime GRAD_CLIP: Scalar[DT] = 10.0


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one thread per batch row. Per-row math is purely local
# (softmax over N_ + a single selected index), so this maps 1:1 to a
# `BATCH`-wide grid.
# ──────────────────────────────────────────────────────────────────────


def _ppo_disc_forward_kernel[N: Int, BATCH: Int](
    ao: Tile[DT, BATCH, N],
    act: Tile[DT, BATCH, 1],
    olp: Tile[DT, BATCH, 1],
    adv: Tile[DT, BATCH, 1],
    loss_out: Tile[DT, BATCH, 1],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var a_idx = Int(rebind[Scalar[DT]](act[b, 0]))
    var max_l = rebind[Scalar[DT]](ao[b, 0])
    for j in range(1, N):
        var lj = rebind[Scalar[DT]](ao[b, j])
        if lj > max_l:
            max_l = lj
    var sum_exp: Scalar[DT] = 0.0
    for j in range(N):
        sum_exp += exp(rebind[Scalar[DT]](ao[b, j]) - max_l)
    var log_sum = flog(sum_exp)
    var entropy: Scalar[DT] = 0.0
    for j in range(N):
        var lp_j = (rebind[Scalar[DT]](ao[b, j]) - max_l) - log_sum
        var p_j = exp(lp_j)
        entropy += -p_j * lp_j
    var new_log_prob = (rebind[Scalar[DT]](ao[b, a_idx]) - max_l) - log_sum
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


def _ppo_disc_vjp_kernel[N: Int, BATCH: Int](
    ao: Tile[DT, BATCH, N],
    act: Tile[DT, BATCH, 1],
    olp: Tile[DT, BATCH, 1],
    adv: Tile[DT, BATCH, 1],
    go: Tile[DT, BATCH, 1],
    gi0: Tile[DT, BATCH, N],
    gi1: Tile[DT, BATCH, 1],
    gi2: Tile[DT, BATCH, 1],
    gi3: Tile[DT, BATCH, 1],
    clip_eps: Scalar[DT],
    entropy_coef: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    # Non-differentiable rollout inputs — zero their grad slots.
    gi1[b, 0] = Scalar[DT](0.0)
    gi2[b, 0] = Scalar[DT](0.0)
    gi3[b, 0] = Scalar[DT](0.0)
    var a_idx = Int(rebind[Scalar[DT]](act[b, 0]))
    var max_l = rebind[Scalar[DT]](ao[b, 0])
    for j in range(1, N):
        var lj = rebind[Scalar[DT]](ao[b, j])
        if lj > max_l:
            max_l = lj
    var sum_exp: Scalar[DT] = 0.0
    for j in range(N):
        sum_exp += exp(rebind[Scalar[DT]](ao[b, j]) - max_l)
    var log_sum = flog(sum_exp)
    var entropy: Scalar[DT] = 0.0
    for j in range(N):
        var lp_j = (rebind[Scalar[DT]](ao[b, j]) - max_l) - log_sum
        var p_j = exp(lp_j)
        entropy += -p_j * lp_j
    var new_log_prob = (rebind[Scalar[DT]](ao[b, a_idx]) - max_l) - log_sum
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
    for j in range(N):
        var lp_j = (rebind[Scalar[DT]](ao[b, j]) - max_l) - log_sum
        var p_j = exp(lp_j)
        var d_nlp_j = (Scalar[DT](1.0) if j == a_idx else Scalar[DT](0.0)) - p_j
        var g_ratio_j: Scalar[DT] = 0.0
        if not is_clipped:
            g_ratio_j = -adv_b * ratio * d_nlp_j
        var g_ent_j = entropy_coef * p_j * (lp_j + entropy)
        var g = (g_ratio_j + g_ent_j) * go_b
        if g > GRAD_CLIP:
            g = GRAD_CLIP
        elif g < -GRAD_CLIP:
            g = -GRAD_CLIP
        gi0[b, j] = g


struct PPODiscreteObjective[N_: Int](Module):
    """Categorical PPO clipped-surrogate + entropy bonus as a quaternary
    Module. `N_` is the number of discrete actions (actor logit width)."""

    comptime ARITY: Int = 4
    # [logits (N_) | action_idx (1) | old_log_prob (1) | advantage (1)].
    comptime IN_DIMS = Self._build_in_dims()
    comptime OUT_DIM: Int = 1                  # per-sample loss

    @staticmethod
    def _build_in_dims() -> Array[Int, 4]:
        var d = Array[Int, 4](fill=1)
        d[0] = Self.N_
        # d[1] = d[2] = d[3] = 1 already from fill=1
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
            "PPODiscreteObjective: target must be 'cpu' or 'gpu'"
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
        comptime N = Self.N_
        ref ao = inputs[0]
        ref act = inputs[1]
        ref olp = inputs[2]
        ref adv = inputs[3]
        comptime if target == "cpu":
            out.ensure(B * 1)
            var ao_v = TileTensor(ao.data, row_major[B, N]())
            var act_v = TileTensor(act.data, row_major[B, 1]())
            var olp_v = TileTensor(olp.data, row_major[B, 1]())
            var adv_v = TileTensor(adv.data, row_major[B, 1]())
            var out_v = TileTensor(out.data, row_major[B, 1]())
            for b in range(B):
                var a_idx = Int(act_v[b, 0])
                var max_l = ao_v[b, 0]
                for j in range(1, N):
                    var lj = ao_v[b, j]
                    if lj > max_l:
                        max_l = lj
                var sum_exp: Scalar[DT] = 0.0
                for j in range(N):
                    sum_exp += exp(ao_v[b, j] - max_l)
                var log_sum = flog(sum_exp)
                var entropy: Scalar[DT] = 0.0
                for j in range(N):
                    var lp_j = (ao_v[b, j] - max_l) - log_sum
                    var p_j = exp(lp_j)
                    entropy += -p_j * lp_j
                var new_log_prob = (ao_v[b, a_idx] - max_l) - log_sum
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
            c.enqueue_function[_ppo_disc_forward_kernel[N, B]](
                ao.tt["gpu", B, N](),
                act.tt["gpu", B, 1](),
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
        comptime N = Self.N_
        ref ao = forward_input[0]
        ref act = forward_input[1]
        ref olp = forward_input[2]
        ref adv = forward_input[3]
        ref gi0 = grad_inputs[0]
        ref gi1 = grad_inputs[1]
        ref gi2 = grad_inputs[2]
        ref gi3 = grad_inputs[3]
        comptime if target == "cpu":
            gi0.ensure(B * N)
            gi1.ensure(B * 1)
            gi2.ensure(B * 1)
            gi3.ensure(B * 1)
            var ao_v = TileTensor(ao.data, row_major[B, N]())
            var act_v = TileTensor(act.data, row_major[B, 1]())
            var olp_v = TileTensor(olp.data, row_major[B, 1]())
            var adv_v = TileTensor(adv.data, row_major[B, 1]())
            var go_v = TileTensor(grad_output.data, row_major[B, 1]())
            var gi0_v = TileTensor(gi0.data, row_major[B, N]())
            var gi1_v = TileTensor(gi1.data, row_major[B, 1]())
            var gi2_v = TileTensor(gi2.data, row_major[B, 1]())
            var gi3_v = TileTensor(gi3.data, row_major[B, 1]())
            for b in range(B):
                # Zero non-differentiable rollout grad slots.
                gi1_v[b, 0] = Scalar[DT](0.0)
                gi2_v[b, 0] = Scalar[DT](0.0)
                gi3_v[b, 0] = Scalar[DT](0.0)
                var a_idx = Int(act_v[b, 0])
                var max_l = ao_v[b, 0]
                for j in range(1, N):
                    var lj = ao_v[b, j]
                    if lj > max_l:
                        max_l = lj
                var sum_exp: Scalar[DT] = 0.0
                for j in range(N):
                    sum_exp += exp(ao_v[b, j] - max_l)
                var log_sum = flog(sum_exp)
                var entropy: Scalar[DT] = 0.0
                for j in range(N):
                    var lp_j = (ao_v[b, j] - max_l) - log_sum
                    var p_j = exp(lp_j)
                    entropy += -p_j * lp_j
                var new_log_prob = (ao_v[b, a_idx] - max_l) - log_sum
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
                for j in range(N):
                    var lp_j = (ao_v[b, j] - max_l) - log_sum
                    var p_j = exp(lp_j)
                    var d_nlp_j = (
                        Scalar[DT](1.0) if j == a_idx else Scalar[DT](0.0)
                    ) - p_j
                    var g_ratio_j: Scalar[DT] = 0.0
                    if not is_clipped:
                        g_ratio_j = -adv_b * ratio * d_nlp_j
                    var g_ent_j = self.entropy_coef * p_j * (lp_j + entropy)
                    var g = (g_ratio_j + g_ent_j) * go_b
                    if g > GRAD_CLIP:
                        g = GRAD_CLIP
                    elif g < -GRAD_CLIP:
                        g = -GRAD_CLIP
                    gi0_v[b, j] = g
        else:
            var c = ctx.value()
            gi0.ensure_gpu(c, B * N)
            gi1.ensure_gpu(c, B * 1)
            gi2.ensure_gpu(c, B * 1)
            gi3.ensure_gpu(c, B * 1)
            comptime n_blocks = (B + TPB - 1) // TPB
            c.enqueue_function[_ppo_disc_vjp_kernel[N, B]](
                ao.tt["gpu", B, N](),
                act.tt["gpu", B, 1](),
                olp.tt["gpu", B, 1](),
                adv.tt["gpu", B, 1](),
                grad_output.tt["gpu", B, 1](),
                gi0.tt["gpu", B, N](),
                gi1.tt["gpu", B, 1](),
                gi2.tt["gpu", B, 1](),
                gi3.tt["gpu", B, 1](),
                self.clip_eps, self.entropy_coef,
                grid_dim=n_blocks, block_dim=TPB,
            )
