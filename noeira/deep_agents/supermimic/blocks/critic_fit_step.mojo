"""CriticFitStep — TD(λ) targets on an imagined rollout and the critic's fit.

`PolicyTrainer.update`'s critic half: V on every imagined state (inputs
`[canonical(s_t), goal_t ⊙ gscale, normalise(ξ), e_root_t]`, detached), the
`gae_targets` recursion

    R_t = r_{t+1} + γ (1 − d̂_{t+1}) [(1 − λ) V_{t+1} + λ R_{t+1}],  R_H = V_H,

then 5 epochs × 4 minibatches of MSE on (s_t, R_t), t < H, each clipped at
norm 1.0, Adam(lr). Minibatch order: a permutation per epoch, drawn here or
passed in (`perms`, epoch-major) to replay another run's draws. On "gpu" the
rollout and the batch are on the device; the values, targets and the last
minibatch's loss come back to the host.
"""

from max.gpu.host import DeviceContext

from std.random import random_ui64

from noeira.deep_agents.supermimic.config import SMConfigT, SMCriticOf
from noeira.deep_agents.supermimic.imagination import SMImagBatch, SMImagOut
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import sm_critic_in, sm_gae, sm_mb_gather, sm_mse, talloc, tp
from noeira.deep_agents.supermimic.policy import SMValueNet
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm


comptime SM_GAE_LAMBDA: Float64 = 0.95
comptime SM_CRITIC_EPOCHS: Int = 5
comptime SM_CRITIC_MB: Int = 4
comptime SM_CRITIC_CLIP: Float64 = 1.0


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


def sm_value_fit[target: StaticString, CIN: Int, C1: Int, C2: Int, N: Int, H: Int](
    mut critic: SMValueNet[CIN, C1, C2],
    mut opt: Adam,
    mut X: TensorImpl[DT],
    rewards: FP[DT],
    dhat: FP[DT],
    gamma: Float64,
    perms: List[Int],
    mut values_out: List[Float64],
    mut targets_out: List[Float64],
    ctx: Optional[DeviceContext] = None,
) raises -> SMCriticFit:
    """The fit on prebuilt inputs `X` [(H+1)·N·CIN]: V on every imagined state,
    the TD(λ) targets of `rewards` [H·N] and `dhat` [(H+1)·N], 5 epochs × 4
    minibatches. `values_out` / `targets_out` receive the (H+1)·N values and
    H·N targets."""
    comptime ROWS = H * N
    comptime MB = ROWS // SM_CRITIC_MB
    comptime assert ROWS % SM_CRITIC_MB == 0, "H·N must split into the critic's 4 minibatches"
    var values = talloc[target]((H + 1) * N, ctx)
    critic.forward[target, (H + 1) * N](TensorRefs[1, _, DT](X), values, ctx)
    var targets = talloc[target](ROWS, ctx)
    sm_gae[target](ctx, N, H, Scalar[DT](gamma), Scalar[DT](SM_GAE_LAMBDA), tp[target](values), rewards, dhat, tp[target](targets))
    # the fit, rows t < H
    var idx = TensorImpl[DT].alloc(ROWS)
    var xin = talloc[target](MB * CIN, ctx)
    var y = talloc[target](MB, ctx)
    var pred = talloc[target](MB, ctx)
    var g = talloc[target](MB, ctx)
    var sq = talloc[target](MB, ctx)
    var gin = talloc[target](MB * CIN, ctx)
    for ep in range(SM_CRITIC_EPOCHS):
        var perm: List[Int]
        if len(perms) == 0:
            perm = sm_random_perm(ROWS)
        else:
            perm = List[Int](length=ROWS, fill=0)
            for i in range(ROWS):
                perm[i] = perms[ep * ROWS + i]
        for i in range(ROWS):
            idx.data[i] = Scalar[DT](perm[i])          # exact below 2^24 rows
        comptime if target == "gpu":
            idx.upload_resident(ctx.value())
        for mbi in range(SM_CRITIC_MB):
            sm_mb_gather[target](ctx, MB, CIN, tp[target](idx).unsafe_offset(mbi * MB), tp[target](X), tp[target](targets), tp[target](xin), tp[target](y))
            critic.forward[target, MB](TensorRefs[1, _, DT](xin), pred, ctx)
            sm_mse[target](ctx, MB, tp[target](pred), tp[target](y), tp[target](g), tp[target](sq))
            critic.zero_grad[target](ctx)
            critic.vjp[target, MB](TensorRefs[1, _, DT](xin), g, TensorRefs[1, _, DT](gin), ctx)
            comptime if target == "gpu":
                opt.clip_grads_device[target](critic, Scalar[DT](SM_CRITIC_CLIP), ctx)
            else:
                _ = clip_grad_norm[target](critic, Scalar[DT](SM_CRITIC_CLIP), ctx)
            opt.step[target](critic, ctx)
    comptime if target == "gpu":
        values.download_enqueue(ctx.value())
        targets.download_enqueue(ctx.value())
        sq.download_enqueue(ctx.value())
        ctx.value().synchronize()
        values.download_finalize()
        targets.download_finalize()
        sq.download_finalize()
    var last = 0.0
    for i in range(MB):
        last += Float64(sq.data[i]) / Float64(MB)
    values_out = List[Float64](length=(H + 1) * N, fill=0.0)
    for i in range((H + 1) * N):
        values_out[i] = Float64(values.data[i])
    targets_out = List[Float64](length=ROWS, fill=0.0)
    for i in range(ROWS):
        targets_out[i] = Float64(targets.data[i])
    var vm = 0.0
    for i in range(len(values_out)):
        vm += values_out[i]
    var tm = 0.0
    for i in range(len(targets_out)):
        tm += targets_out[i]
    return SMCriticFit(last, vm / Float64(len(values_out)), tm / Float64(len(targets_out)))


struct CriticFitStep[C: SMConfigT](Movable):
    """On "gpu" the optimiser adopts the critic into its arena at the first
    step: its 20 clips per update stay on the device."""
    var opt: Adam
    var adopted: Bool

    def __init__(out self):
        self.opt = Adam(lr=Scalar[DT](Self.C.DEF_LR))
        self.adopted = False

    def __init__(out self, *, deinit move: Self):
        self.opt = move.opt^
        self.adopted = move.adopted

    def step[target: StaticString, N: Int, H: Int](
        mut self,
        mut critic: SMCriticOf[Self.C],
        mut o: SMImagOut[Self.C.ROBOT],
        mut ib: SMImagBatch[Self.C.ROBOT],
        gamma: Float64,
        perms: List[Int],
        mut values_out: List[Float64],
        mut targets_out: List[Float64],
        ctx: Optional[DeviceContext] = None,
    ) raises -> SMCriticFit:
        """`values_out` / `targets_out` receive V on the (H+1)·N states and the
        H·N targets."""
        comptime if target == "gpu":
            if not self.adopted:
                self.opt.adopt[target](critic, ctx)
                self.adopted = True
        comptime R = Self.C.ROBOT
        comptime CIN = SMDims[R].CRITIC_IN
        # inputs, detached
        var X = talloc[target]((H + 1) * N * CIN, ctx)
        sm_critic_in[target, R](ctx, (H + 1) * N, N, tp[target](o.states), tp[target](ib.goals), tp[target](ib.xi), tp[target](o.e_root), tp[target](X))
        return sm_value_fit[target, CIN, Self.C.CRIT1, Self.C.CRIT2, N, H](
            critic, self.opt, X, tp[target](o.rewards), tp[target](o.dhat), gamma, perms, values_out, targets_out, ctx
        )
