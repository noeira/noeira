"""SuperMimicTrainer — the algorithm: networks, the robot's tables, the blocks.

Owns the world model, the policy and the critic of config `C` (their types are
`SMWorldModelOf[C]` / `SMPolicyOf[C]` / `SMCriticOf[C]`), the robot's runtime
tables, and the four blocks (`blocks/`). It knows nothing about the
environment or the replay; the agent (`agent.mojo`) feeds it batches:

    act[L](hist, goals)            → actions, the policy over L lanes
    wm_update[B](windows)          → one Eq. 1 step (WorldModelUpdateStep)
    policy_update[N](starts, …)    → ActorStep then CriticFitStep on its rollout

Every block call is `step[C.TARGET]`: switching the config's target is what
selects the GPU path once the blocks have one.
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.blocks import ActStep, ActorStep, CriticFitStep, WorldModelUpdateStep
from noeira.deep_agents.supermimic.blocks.actor_step import SM_GAMMA
from noeira.deep_agents.supermimic.config import SMConfigT, SMCriticOf, SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.imagination import SMImagBatch
from noeira.deep_agents.supermimic.tables import SMTables, sm_tables
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss
from noeira.nn.constants import DT
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

    def __init__(out self, var tables: SMTables[DT]) raises:
        """Fresh networks (Kaiming; the dynamics' last layer zero, the
        decoder's last layer scaled to gain 0.01) over the robot's tables."""
        self.wm = SMWorldModelOf[Self.C].make[Kaiming]()
        self.pol = SMPolicyOf[Self.C].make[Kaiming]()
        self.critic = SMCriticOf[Self.C].make["cpu", Kaiming](None)
        var sc = _ScaleParam(String("decoder.4.weight"), Scalar[DT](0.01))
        self.pol.for_each_param["cpu"](sc, None)
        if sc.hit != 1:
            raise Error("SuperMimicTrainer: the decoder's last layer was not found to rescale")
        self.tables = tables^
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

    @staticmethod
    def for_robot(sim: Self.C.ROBOT.SIM) raises -> Self:
        """Tables from the robot's names, its default pose and `sim`'s records."""
        var q0 = List[Float64](length=Self.C.ROBOT.NDOF, fill=0.0)
        for j in range(Self.C.ROBOT.NDOF):
            q0[j] = Self.C.ROBOT.default_joint_pos(j)
        return Self(sm_tables[DT, Self.C.ROBOT](sim, q0))

    def set_lr(mut self, lr: Float64):
        """The same lr for world model, policy and critic (`Trainer.run`)."""
        self.wm_blk.opt.lr = Scalar[DT](lr)
        self.actor_blk.opt.lr = Scalar[DT](lr)
        self.critic_blk.opt.lr = Scalar[DT](lr)

    def act[L: Int](mut self, hist: TensorImpl[DT], goals: TensorImpl[DT]) raises -> TensorImpl[DT]:
        return self.act_blk.step[Self.C.TARGET, L](self.pol, hist, goals)

    def wm_update[B: Int](mut self, mut w: SMWindowBatch[Self.C.ROBOT]) raises -> SMWMLoss:
        var r = self.wm_blk.step[Self.C.TARGET, B](self.wm, w)
        return r[0].copy()

    def policy_update[N: Int](
        mut self,
        mut ib: SMImagBatch[Self.C.ROBOT],
        perms: List[Int],
        mut values_out: List[Float64],
        mut targets_out: List[Float64],
        obs_noise: Bool = False,
    ) raises -> SMUpdateMetrics:
        """`PolicyTrainer.update` after the start batch: the actor step, then
        the critic's fit on that rollout. `perms` as in `CriticFitStep`."""
        if ib.h != Self.C.HORIZON:
            raise Error("policy_update: a batch of horizon " + String(ib.h) + ", the config's is " + String(Self.C.HORIZON))
        var ro = self.actor_blk.step[Self.C.TARGET, N](self.wm, self.pol, self.critic, self.tables, ib, obs_noise)
        var cf = self.critic_blk.step[Self.C.TARGET, N, Self.C.HORIZON](
            self.critic, ro[0], ib, SM_GAMMA, perms, values_out, targets_out
        )
        return SMUpdateMetrics(ro[0].j, ro[0].return_undisc, ro[1], cf.last_loss, cf.value_mean, cf.target_mean)
