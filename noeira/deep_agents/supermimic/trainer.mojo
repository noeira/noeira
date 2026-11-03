# +--------------------------------------------------------------------------+ #
# | SuperMimic M5: one policy update — actor step, TD(λ) targets, critic fit
# +--------------------------------------------------------------------------+ #
"""The reproduction's `PolicyTrainer.update` (`imagination/policy_trainer.py`)
after the start batch is drawn:

    actor   imagine (H steps, frozen world model and critic), loss = −J / H,
            clip_grad_norm(1.0) then clip_grad_value(0.1), Adam(2e-4, β = (0.9, 0.95))
    critic  V on every imagined state, TD(λ) targets (`gae_targets`, λ = 0.95):
                R_t = r_{t+1} + γ (1 − d̂_{t+1}) [(1 − λ) V_{t+1} + λ R_{t+1}],  R_H = V_H
            then 5 epochs × 4 minibatches of MSE on (s_t, R_t), t < H, each
            clipped at norm 1.0, Adam(2e-4)

The critic's inputs are `[canonical(s_t), goal_t ⊙ gscale, normalise(ξ),
e_root_t]`, all detached. Minibatch order: a permutation per epoch, drawn here
or passed in (`perms`, to replay another run's draws exactly).
"""

from std.random import random_ui64

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.g1_fk import FP, G1FKTree
from noeira.deep_agents.supermimic.heads import G1HeadsTable
from noeira.deep_agents.supermimic.imagination import SMImagBatch, SMImagOut, sm_imagine
from noeira.deep_agents.supermimic.policy import (
    SMCritic, SMPolicy, SM_CRITIC_IN, SM_EROOT, SM_GOAL_DIM, sm_goal_scale,
)
from noeira.deep_agents.supermimic.state_spec import SM_FEAT_DIM, sm_canonical_lane, sm_xi_normalize
from noeira.deep_agents.supermimic.world_model import SMWorldModel
from noeira.nn.constants import DT
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm


comptime SM_GAMMA: Float64 = 0.99
comptime SM_GAE_LAMBDA: Float64 = 0.95
comptime SM_LR: Float64 = 2e-4
comptime SM_CLIP_NORM: Float64 = 1.0
comptime SM_CLIP_VALUE: Float64 = 0.1
comptime SM_CRITIC_EPOCHS: Int = 5
comptime SM_CRITIC_MB: Int = 4


struct _ClipValueCPU(ParamVisitor):
    """torch's `clip_grad_value_`: every gradient element into [−c, c]."""
    var c: Scalar[DT]

    def __init__(out self, c: Scalar[DT]):
        self.c = c

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            grad.data[i] = min(max(grad.data[i], -self.c), self.c)


def sm_gae_targets(o: SMImagOut, values: List[Float64], h: Int, n: Int, gamma: Float64, lam: Float64) -> List[Float64]:
    """`gae_targets(rewards, dhat[1:], values)`: [h·n], t-major. `values` is [(h+1)·n]."""
    var out = List[Float64](length=h * n, fill=0.0)
    for b in range(n):
        var R = values[h * n + b]
        var t = h - 1
        while t >= 0:
            R = Float64(o.rewards[t * n + b]) + gamma * (1.0 - Float64(o.dhat[(t + 1) * n + b])) * (
                (1.0 - lam) * values[(t + 1) * n + b] + lam * R
            )
            out[t * n + b] = R
            t -= 1
    return out^


def sm_random_perm(n: Int) -> List[Int]:
    var p = List[Int](length=n, fill=0)
    for i in range(n):
        p[i] = i
    var i = n - 1
    while i > 0:
        var j = Int(random_ui64(0, UInt64(i)))
        var tmp = p[i]
        p[i] = p[j]
        p[j] = tmp
        i -= 1
    return p^


@fieldwise_init
struct SMUpdateMetrics(Copyable, Movable):
    var imag_objective: Float64
    var imag_return_undisc: Float64
    var policy_grad_norm: Float64
    var critic_loss: Float64
    var critic_value_mean: Float64
    var critic_target_mean: Float64


struct SMPolicyTrainer(Movable):
    """The two optimisers (state across updates)."""
    var pol_opt: Adam
    var critic_opt: Adam

    def __init__(out self):
        self.pol_opt = Adam(lr=Scalar[DT](SM_LR), beta1=Scalar[DT](0.9), beta2=Scalar[DT](0.95))
        self.critic_opt = Adam(lr=Scalar[DT](SM_LR))

    def __init__(out self, *, deinit move: Self):
        self.pol_opt = move.pol_opt^
        self.critic_opt = move.critic_opt^

    def update[
        HW: Int, TH: Int, E1: Int, E2: Int, E3: Int, E4: Int, D1: Int, D2: Int, D3: Int, D4: Int,
        C1: Int, C2: Int, N: Int, H: Int,
    ](
        mut self,
        mut wm: SMWorldModel[HW, TH],
        mut pol: SMPolicy[E1, E2, E3, E4, D1, D2, D3, D4],
        mut critic: SMCritic[C1, C2],
        mut tree: G1FKTree[DT],
        mut table: G1HeadsTable[DT],
        mut ib: SMImagBatch,
        perms: List[Int],
        mut values_out: List[Float64],
        mut targets_out: List[Float64],
    ) raises -> SMUpdateMetrics:
        """One update. `perms` is either empty (draw here) or
        `SM_CRITIC_EPOCHS × (H·N)` indices, epoch-major. `values_out` /
        `targets_out` receive V on the (H+1)·N states and the H·N targets."""
        comptime ROWS = H * N
        comptime MB = ROWS // SM_CRITIC_MB
        comptime assert ROWS % SM_CRITIC_MB == 0, "H·N must split into the critic's 4 minibatches"
        if ib.h != H:
            raise Error("SMPolicyTrainer.update: a batch of horizon " + String(ib.h) + ", compiled for " + String(H))
        # ── actor ──
        pol.zero_grad["cpu"](None)
        var o = sm_imagine[HW, TH, E1, E2, E3, E4, D1, D2, D3, D4, C1, C2, N](
            wm, pol, critic, tree, table, ib, SM_GAMMA, True
        )
        var gn = clip_grad_norm["cpu"](pol, Scalar[DT](SM_CLIP_NORM), None)
        var cv = _ClipValueCPU(Scalar[DT](SM_CLIP_VALUE))
        pol.for_each_param["cpu"](cv, None)
        self.pol_opt.step["cpu"](pol, None)

        # ── critic inputs, detached: [canonical(s_t), goal_t·scale, ξ_n, e_root_t] ──
        var X = List[Scalar[DT]](length=(H + 1) * N * SM_CRITIC_IN, fill=0)
        for t in range(H + 1):
            for b in range(N):
                var r = (t * N + b) * SM_CRITIC_IN
                sm_canonical_lane[DT](
                    o.states.unsafe_ptr().unsafe_offset((t * N + b) * 71).as_unsafe_any_origin(),
                    X.unsafe_ptr().unsafe_offset(r).as_unsafe_any_origin(),
                )
                for k in range(SM_GOAL_DIM):
                    X[r + SM_FEAT_DIM + k] = ib.goals[(t * N + b) * SM_GOAL_DIM + k] * Scalar[DT](sm_goal_scale(k))
                for k in range(33):
                    X[r + SM_FEAT_DIM + SM_GOAL_DIM + k] = Scalar[DT](sm_xi_normalize(k, Float64(ib.xi[b * 33 + k])))
                for k in range(SM_EROOT):
                    X[r + SM_FEAT_DIM + SM_GOAL_DIM + 33 + k] = o.e_root[(t * N + b) * SM_EROOT + k]
        values_out = List[Float64](length=(H + 1) * N, fill=0.0)
        for t in range(H + 1):
            var xin = TensorImpl[DT].alloc(N * SM_CRITIC_IN)
            for i in range(N * SM_CRITIC_IN):
                xin.data[i] = X[t * N * SM_CRITIC_IN + i]
            var v = TensorImpl[DT].alloc(N)
            critic.forward["cpu", N](TensorRefs[1, _, DT](xin), v, None)
            for b in range(N):
                values_out[t * N + b] = Float64(v.data[b])
        targets_out = sm_gae_targets(o, values_out, H, N, SM_GAMMA, SM_GAE_LAMBDA)

        # ── critic fit: rows t < H ──
        var last = 0.0
        for ep in range(SM_CRITIC_EPOCHS):
            var perm: List[Int]
            if len(perms) == 0:
                perm = sm_random_perm(ROWS)
            else:
                perm = List[Int](length=ROWS, fill=0)
                for i in range(ROWS):
                    perm[i] = perms[ep * ROWS + i]
            for mbi in range(SM_CRITIC_MB):
                var xin = TensorImpl[DT].alloc(MB * SM_CRITIC_IN)
                var y = List[Float64](length=MB, fill=0.0)
                for i in range(MB):
                    var row = perm[mbi * MB + i]
                    for k in range(SM_CRITIC_IN):
                        xin.data[i * SM_CRITIC_IN + k] = X[row * SM_CRITIC_IN + k]
                    y[i] = targets_out[row]
                var pred = TensorImpl[DT].alloc(MB)
                critic.forward["cpu", MB](TensorRefs[1, _, DT](xin), pred, None)
                var g = TensorImpl[DT].alloc(MB)
                var l = 0.0
                for i in range(MB):
                    var d = Float64(pred.data[i]) - y[i]
                    l += d * d / Float64(MB)
                    g.data[i] = Scalar[DT](2.0 * d / Float64(MB))
                critic.zero_grad["cpu"](None)
                var gin = TensorImpl[DT].alloc(MB * SM_CRITIC_IN)
                critic.vjp["cpu", MB](TensorRefs[1, _, DT](xin), g, TensorRefs[1, _, DT](gin), None)
                _ = clip_grad_norm["cpu"](critic, Scalar[DT](SM_CLIP_NORM), None)
                self.critic_opt.step["cpu"](critic, None)
                last = l
        var vm = 0.0
        for i in range(len(values_out)):
            vm += values_out[i]
        var tm = 0.0
        for i in range(len(targets_out)):
            tm += targets_out[i]
        return SMUpdateMetrics(o.j, o.return_undisc, Float64(gn), last, vm / Float64(len(values_out)), tm / Float64(len(targets_out)))
