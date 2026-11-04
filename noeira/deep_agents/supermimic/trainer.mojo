"""SuperMimicTrainer — the algorithm: networks, the robot's tables, the blocks.

Owns the world model, the policy and the critic of config `C` (their types are
`SMWorldModelOf[C]` / `SMPolicyOf[C]` / `SMCriticOf[C]`), the robot's runtime
tables, and the four blocks (`blocks/`). It knows nothing about the
environment or the replay; the agent (`agent.mojo`) feeds it batches:

    act[L](hist, goals)            → actions, the policy over L lanes
    wm_update[B](windows)          → one Eq. 1 step (WorldModelUpdateStep)
    policy_update[N](starts, …)    → ActorStep then CriticFitStep on its rollout

Every block call is `step[C.TARGET]`: the config's target picks where the
networks live and where the blocks run. Batches arrive as host tensors (the
CPU env and replay); on "gpu" the trainer copies them to the device before
the block and copies the actions back, so the same agent runs either way.
A device env and replay will hand device tensors over directly.
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.blocks import ActStep, ActorStep, CriticFitStep, WorldModelUpdateStep
from noeira.deep_agents.supermimic.config import SMConfigT, SMCriticOf, SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.imagination import SMImagBatch
from noeira.deep_agents.supermimic.tables import SMTables, sm_tables
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss
from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import CheckpointScalars, load_params_multi, save_params_multi
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor, TensorImpl


@fieldwise_init
struct SMUpdateMetrics(Copyable, Movable):
    var imag_objective: Float64
    var imag_return_undisc: Float64
    var policy_grad_norm: Float64
    var critic_loss: Float64
    var critic_value_mean: Float64
    var critic_target_mean: Float64


struct _ScaleParam(ParamVisitor):
    """Multiply one named param by `k`."""
    var name: String
    var k: Scalar[DT]
    var hit: Int

    def __init__(out self, var name: String, k: Scalar[DT]):
        self.name = name^
        self.k = k
        self.hit = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        if name == self.name:
            for i in range(N):
                param.data[i] *= self.k
            comptime if target == "gpu":
                param.upload_resident(ctx.value())
            param.version += 1
            self.hit += 1


struct SuperMimicTrainer[C: SMConfigT](Movable):
    var wm: SMWorldModelOf[Self.C]
    var pol: SMPolicyOf[Self.C]
    var critic: SMCriticOf[Self.C]
    var tables: SMTables[DT]
    var act_blk: ActStep[Self.C]
    var wm_blk: WorldModelUpdateStep[Self.C]
    var actor_blk: ActorStep[Self.C]
    var critic_blk: CriticFitStep[Self.C]
    var ctx: Optional[DeviceContext]

    def __init__(out self, var tables: SMTables[DT], ctx: Optional[DeviceContext] = None) raises:
        """Fresh networks (Kaiming; the dynamics' last layer zero, the
        decoder's last layer scaled to gain 0.01) over the robot's tables,
        on the config's target (`ctx` required for "gpu")."""
        comptime T = Self.C.TARGET
        comptime if T == "gpu":
            if not ctx:
                raise Error("SuperMimicTrainer: a GPU config needs a DeviceContext")
        self.ctx = ctx
        self.wm = SMWorldModelOf[Self.C].make[Kaiming, T](ctx)
        self.pol = SMPolicyOf[Self.C].make[Kaiming, T](ctx)
        self.critic = SMCriticOf[Self.C].make[T, Kaiming](ctx)
        var sc = _ScaleParam(String("decoder.4.weight"), Scalar[DT](0.01))
        self.pol.for_each_param[T](sc, ctx)
        if sc.hit != 1:
            raise Error("SuperMimicTrainer: the decoder's last layer was not found to rescale")
        self.tables = tables^
        comptime if T == "gpu":
            self.tables.to_device(ctx.value())
        self.act_blk = ActStep[Self.C]()
        self.wm_blk = WorldModelUpdateStep[Self.C]()
        self.actor_blk = ActorStep[Self.C]()
        self.critic_blk = CriticFitStep[Self.C]()

    def __init__(out self, *, deinit move: Self):
        self.wm = move.wm^
        self.pol = move.pol^
        self.critic = move.critic^
        self.tables = move.tables^
        self.act_blk = move.act_blk^
        self.wm_blk = move.wm_blk^
        self.actor_blk = move.actor_blk^
        self.critic_blk = move.critic_blk^
        self.ctx = move.ctx^

    @staticmethod
    def for_robot(sim: Self.C.ROBOT.SIM, ctx: Optional[DeviceContext] = None) raises -> Self:
        """Tables from the robot's names, its default pose and `sim`'s records."""
        var q0 = List[Float64](length=Self.C.ROBOT.NDOF, fill=0.0)
        for j in range(Self.C.ROBOT.NDOF):
            q0[j] = Self.C.ROBOT.default_joint_pos(j)
        return Self(sm_tables[DT, Self.C.ROBOT](sim, q0), ctx)

    # ── checkpoint: ONE v3 file, world model + policy + critic ──
    def save_state(mut self, path: String, iteration: Int, updates: Int) raises:
        """The three networks' weights and the loop's counters (`iteration`,
        `updates`, as `K` scalars) into one v3 binary checkpoint (atomic
        write). Optimiser moments are NOT saved (SAC's choice: a resume
        re-warms Adam; the paper-width moments would triple the file)."""
        var sc = CheckpointScalars()
        sc.set_int("iteration", iteration)
        sc.set_int("updates", updates)
        save_params_multi[Self.C.TARGET](path, self.ctx, False, self.wm, self.pol, self.critic, scalars=sc)

    def load_state(mut self, path: String) raises -> Tuple[Int, Int]:
        """Restore the three networks; returns (iteration, updates)."""
        var sc = load_params_multi[Self.C.TARGET](path, self.ctx, self.wm, self.pol, self.critic)
        return (sc.get_int("iteration", 0), sc.get_int("updates", 0))

    def set_lr(mut self, lr: Float64):
        """The same lr for world model, policy and critic (`Trainer.run`)."""
        self.wm_blk.opt.lr = Scalar[DT](lr)
        self.actor_blk.opt.lr = Scalar[DT](lr)
        self.critic_blk.opt.lr = Scalar[DT](lr)

    def act[L: Int](mut self, mut hist: TensorImpl[DT], mut goals: TensorImpl[DT], resident: Bool = False) raises -> TensorImpl[DT]:
        """Host `hist` / `goals` in, host actions out (on "gpu" via the
        device); `resident`: inputs and actions stay on the device."""
        comptime T = Self.C.TARGET
        comptime if T == "gpu":
            if not resident:
                hist.upload_resident(self.ctx.value())
                goals.upload_resident(self.ctx.value())
        var a = self.act_blk.step[T, L](self.pol, hist, goals, self.ctx)
        comptime if T == "gpu":
            if not resident:
                a.download(self.ctx.value())
        return a^

    def wm_update[B: Int](mut self, mut w: SMWindowBatch[Self.C.ROBOT], resident: Bool = False) raises -> SMWMLoss:
        """`resident`: the windows are already on the device."""
        comptime if Self.C.TARGET == "gpu":
            if not resident:
                w.to_device(self.ctx.value())
        var r = self.wm_blk.step[Self.C.TARGET, B](self.wm, w, self.ctx)
        return r[0].copy()

    def policy_update[N: Int](
        mut self,
        mut ib: SMImagBatch[Self.C.ROBOT],
        perms: List[Int],
        mut values_out: List[Float64],
        mut targets_out: List[Float64],
        obs_noise: Bool = False,
        resident: Bool = False,
    ) raises -> SMUpdateMetrics:
        """`PolicyTrainer.update` after the start batch: the actor step, then
        the critic's fit on that rollout. `perms` as in `CriticFitStep`."""
        if ib.h != Self.C.HORIZON:
            raise Error("policy_update: a batch of horizon " + String(ib.h) + ", the config's is " + String(Self.C.HORIZON))
        comptime if Self.C.TARGET == "gpu":
            if not resident:
                ib.to_device(self.ctx.value())
        var ro = self.actor_blk.step[Self.C.TARGET, N](self.wm, self.pol, self.critic, self.tables, ib, obs_noise, self.ctx)
        var cf = self.critic_blk.step[Self.C.TARGET, N, Self.C.HORIZON](
            self.critic, ro[0], ib, Self.C.DEF_GAMMA, perms, values_out, targets_out, self.ctx
        )
        return SMUpdateMetrics(ro[0].j, ro[0].return_undisc, ro[1], cf.last_loss, cf.value_mean, cf.target_mean)
