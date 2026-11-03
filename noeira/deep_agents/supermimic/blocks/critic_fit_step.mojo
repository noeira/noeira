"""CriticFitStep — TD(λ) targets on an imagined rollout and the critic's fit.

`PolicyTrainer.update`'s critic half: V on every imagined state (inputs
`[canonical(s_t), goal_t ⊙ gscale, normalise(ξ), e_root_t]`, detached), the
`gae_targets` recursion

    R_t = r_{t+1} + γ (1 − d̂_{t+1}) [(1 − λ) V_{t+1} + λ R_{t+1}],  R_H = V_H,

then 5 epochs × 4 minibatches of MSE on (s_t, R_t), t < H, each clipped at
norm 1.0, Adam(lr). Minibatch order: a permutation per epoch, drawn here or
passed in (`perms`, epoch-major) to replay another run's draws.
"""

from std.random import random_ui64

from noeira.deep_agents.supermimic.config import SMConfigT, SMCriticOf
from noeira.deep_agents.supermimic.imagination import SMImagBatch, SMImagOut
from noeira.deep_agents.supermimic.policy import sm_goal_scale
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.deep_agents.supermimic.state_spec import sm_canonical_lane, sm_xi_normalize
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm


comptime SM_GAE_LAMBDA: Float64 = 0.95
comptime SM_CRITIC_EPOCHS: Int = 5
comptime SM_CRITIC_MB: Int = 4
comptime SM_CRITIC_LR: Float64 = 2e-4
comptime SM_CRITIC_CLIP: Float64 = 1.0


def sm_gae_targets[R: TrackingRobot](
    o: SMImagOut[R], values: List[Float64], h: Int, n: Int, gamma: Float64, lam: Float64
) -> List[Float64]:
    """`gae_targets(rewards, dhat[1:], values)`: [h·n], t-major. `values` is [(h+1)·n]."""
    var out = List[Float64](length=h * n, fill=0.0)
    for b in range(n):
        var R_ = values[h * n + b]
        var t = h - 1
        while t >= 0:
            R_ = Float64(o.rewards.data[t * n + b]) + gamma * (1.0 - Float64(o.dhat.data[(t + 1) * n + b])) * (
                (1.0 - lam) * values[(t + 1) * n + b] + lam * R_
            )
            out[t * n + b] = R_
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
struct SMCriticFit(Copyable, Movable):
    var last_loss: Float64
    var value_mean: Float64
    var target_mean: Float64


struct CriticFitStep[C: SMConfigT](Movable):
    var opt: Adam

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](SM_CRITIC_LR))

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^

    def step[target: StaticString, N: Int, H: Int](
        mut self,
        mut critic: SMCriticOf[Self.C],
        mut o: SMImagOut[Self.C.ROBOT],
        mut ib: SMImagBatch[Self.C.ROBOT],
        gamma: Float64,
        perms: List[Int],
        mut values_out: List[Float64],
        mut targets_out: List[Float64],
    ) raises -> SMCriticFit:
        """`values_out` / `targets_out` receive V on the (H+1)·N states and the
        H·N targets."""
        comptime if target == "gpu":
            raise Error("SuperMimic CriticFitStep: the GPU path is not built yet")
        comptime R = Self.C.ROBOT
        comptime D = SMDims[R]
        comptime CIN = D.CRITIC_IN
        comptime ROWS = H * N
        comptime MB = ROWS // SM_CRITIC_MB
        comptime assert ROWS % SM_CRITIC_MB == 0, "H·N must split into the critic's 4 minibatches"
        # inputs, detached
        var X = List[Scalar[DT]](length=(H + 1) * N * CIN, fill=0)
        for t in range(H + 1):
            for b in range(N):
                var r = (t * N + b) * CIN
                sm_canonical_lane[DT, R](
                    o.states.data.unsafe_ptr().unsafe_offset((t * N + b) * D.S).as_unsafe_any_origin(),
                    X.unsafe_ptr().unsafe_offset(r).as_unsafe_any_origin(),
                )
                for k in range(D.GOAL):
                    X[r + D.S + k] = ib.goals.data[(t * N + b) * D.GOAL + k] * Scalar[DT](sm_goal_scale[R.NDOF](k))
                for k in range(D.X):
                    X[r + D.S + D.GOAL + k] = Scalar[DT](sm_xi_normalize[R.NDOF](k, Float64(ib.xi.data[b * D.X + k])))
                for k in range(D.EROOT):
                    X[r + D.S + D.GOAL + D.X + k] = o.e_root.data[(t * N + b) * D.EROOT + k]
        values_out = List[Float64](length=(H + 1) * N, fill=0.0)
        for t in range(H + 1):
            var xin = TensorImpl[DT].alloc(N * CIN)
            for i in range(N * CIN):
                xin.data[i] = X[t * N * CIN + i]
            var v = TensorImpl[DT].alloc(N)
            critic.forward["cpu", N](TensorRefs[1, _, DT](xin), v, None)
            for b in range(N):
                values_out[t * N + b] = Float64(v.data[b])
        targets_out = sm_gae_targets[R](o, values_out, H, N, gamma, SM_GAE_LAMBDA)
        # the fit, rows t < H
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
                var xin = TensorImpl[DT].alloc(MB * CIN)
                var y = List[Float64](length=MB, fill=0.0)
                for i in range(MB):
                    var row = perm[mbi * MB + i]
                    for k in range(CIN):
                        xin.data[i * CIN + k] = X[row * CIN + k]
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
                var gin = TensorImpl[DT].alloc(MB * CIN)
                critic.vjp["cpu", MB](TensorRefs[1, _, DT](xin), g, TensorRefs[1, _, DT](gin), None)
                _ = clip_grad_norm["cpu"](critic, Scalar[DT](SM_CRITIC_CLIP), None)
                self.opt.step["cpu"](critic, None)
                last = l
        var vm = 0.0
        for i in range(len(values_out)):
            vm += values_out[i]
        var tm = 0.0
        for i in range(len(targets_out)):
            tm += targets_out[i]
        return SMCriticFit(last, vm / Float64(len(values_out)), tm / Float64(len(targets_out)))
