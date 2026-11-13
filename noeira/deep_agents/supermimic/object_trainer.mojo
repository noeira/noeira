"""SMObjTrainer — a tracker, frozen, with the object interface on top.

Owns the tracker's trainer (`base`: its world model and policy, neither of
which moves here) and the three networks that train
(`docs/OBJECT_INTERACTION_PLAN.md` §4, step O1.4): the world model's object
network, the object encoder and the interface's critic. The agent feeds it
batches:

    act[L](hist, goals, s, o, oref)    a = dec([enc_motion(goal) + gain · enc_obj(·), hist])
    om_update[B](windows, obj windows) the joint k-step loss, the object network's step
    policy_update[N](batch)            the object encoder, then its critic

⚠ The tracker's checkpoint is never written: with the object encoder at its
zero start the policy IS the tracker's. The interface's checkpoint holds the
object network, the object encoder and the critic, and needs the tracker's
checkpoint beside it. `load_object` reads the object network alone, as
`examples/g1/supermimic_object_fit.mojo` saves it (O1.3's fit).
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.blocks.obj_policy_step import ObjActorStep, ObjCriticStep
from noeira.deep_agents.supermimic.blocks.obj_update_step import ObjectModelUpdateStep
from noeira.deep_agents.supermimic.lanes import sm_dec_in, sm_enc_in, talloc, tp
from noeira.deep_agents.supermimic.object_imag_lanes import SMObjImagDims, sm_oe_in, sm_oe_mix
from noeira.deep_agents.supermimic.object_imagination import SMObjImagBatch
from noeira.deep_agents.supermimic.object_model import SMObjLoss, SMObjWindowBatch
from noeira.deep_agents.supermimic.object_policy import SMObjConfigT, SMObjCriticOf, SMObjEncOf, SMObjModelOf
from noeira.deep_agents.supermimic.object_task import SM_OBJ_TERMS
from noeira.deep_agents.supermimic.robot import SMDims
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer
from noeira.deep_agents.supermimic.world_model import SMWindowBatch
from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import CheckpointScalars, load_params_multi, save_params_multi
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


@fieldwise_init
struct SMObjUpdate(Copyable, Movable):
    var imag_objective: Float64
    var imag_return_undisc: Float64
    var policy_grad_norm: Float64
    var critic_loss: Float64
    var critic_value_mean: Float64
    var critic_target_mean: Float64
    # the object's three reward terms, mean per imagined step (position,
    # orientation, hands); −1 unless `policy_update` was asked for them
    var r_obj_pos: Float64
    var r_obj_ori: Float64
    var r_obj_hands: Float64


struct SMObjTrainer[OC: SMObjConfigT](Movable):
    var base: SuperMimicTrainer[Self.OC.BASE]
    var om: SMObjModelOf[Self.OC]
    var enc: SMObjEncOf[Self.OC]
    var critic: SMObjCriticOf[Self.OC]
    var om_blk: ObjectModelUpdateStep[Self.OC.BASE, Self.OC.OBJ_H]
    var actor_blk: ObjActorStep[Self.OC]
    var critic_blk: ObjCriticStep[Self.OC]
    var gain: Float64

    def __init__(out self, var base: SuperMimicTrainer[Self.OC.BASE]) raises:
        """Fresh object network (zero-started), object encoder (zero-started)
        and critic over `base`, on its target."""
        comptime T = Self.OC.BASE.TARGET
        var ctx = base.ctx
        self.om = SMObjModelOf[Self.OC].make[Kaiming, T](ctx)
        self.enc = SMObjEncOf[Self.OC].make[T, Kaiming](ctx)
        self.critic = SMObjCriticOf[Self.OC].make[T, Kaiming](ctx)
        self.base = base^
        self.om_blk = ObjectModelUpdateStep[Self.OC.BASE, Self.OC.OBJ_H]()
        self.actor_blk = ObjActorStep[Self.OC]()
        self.critic_blk = ObjCriticStep[Self.OC]()
        self.gain = Self.OC.DEF_GAIN

    def __init__(out self, *, deinit move: Self):
        self.base = move.base^
        self.om = move.om^
        self.enc = move.enc^
        self.critic = move.critic^
        self.om_blk = move.om_blk^
        self.actor_blk = move.actor_blk^
        self.critic_blk = move.critic_blk^
        self.gain = move.gain

    # ── checkpoints ──
    def load_object(mut self, path: String) raises:
        """The object network alone (O1.3's fit)."""
        _ = load_params_multi[Self.OC.BASE.TARGET](path, self.base.ctx, self.om)

    def save_state(mut self, path: String, iteration: Int, updates: Int) raises:
        var sc = CheckpointScalars()
        sc.set_int("iteration", iteration)
        sc.set_int("updates", updates)
        save_params_multi[Self.OC.BASE.TARGET](path, self.base.ctx, False, self.om, self.enc, self.critic, scalars=sc)

    def load_state(mut self, path: String) raises -> Tuple[Int, Int]:
        var sc = load_params_multi[Self.OC.BASE.TARGET](path, self.base.ctx, self.om, self.enc, self.critic)
        return (sc.get_int("iteration", 0), sc.get_int("updates", 0))

    def set_lr(mut self, lr: Float64, om_lr: Float64):
        """`lr` for the object encoder and its critic; the object network
        starts fitted and keeps fitting at its own `om_lr`."""
        self.actor_blk.opt.lr = Scalar[DT](lr)
        self.critic_blk.opt.lr = Scalar[DT](lr)
        self.om_blk.opt.lr = Scalar[DT](om_lr)

    def act[L: Int](
        mut self, mut hist: TensorImpl[DT], mut goals: TensorImpl[DT], mut s: TensorImpl[DT], mut o: TensorImpl[DT],
        mut oref: TensorImpl[DT], resident: Bool = False,
    ) raises -> TensorImpl[DT]:
        """`hist` [L · 10 · OBS], `goals` [L · GOAL], the robot's states `s`
        [L · S], the object's `o` [L · 13] and its reference rows `oref`
        [L · 19] → actions [L · n]. `resident`: inputs and actions stay on the
        device."""
        comptime T = Self.OC.BASE.TARGET
        comptime R = Self.OC.BASE.ROBOT
        comptime D = SMDims[R]
        var ctx = self.base.ctx
        comptime if T == "gpu":
            if not resident:
                hist.upload_resident(ctx.value())
                goals.upload_resident(ctx.value())
                s.upload_resident(ctx.value())
                o.upload_resident(ctx.value())
                oref.upload_resident(ctx.value())
        var ein = talloc[T](L * D.GOAL, ctx)
        sm_enc_in[T, R](ctx, L, tp[T](goals), tp[T](ein))
        var z = talloc[T](L * D.LATENT, ctx)
        self.base.pol.enc.forward[T, L](TensorRefs[1, _, DT](ein), z, ctx)
        var eoin = talloc[T](L * SMObjImagDims[R].EO_IN, ctx)
        sm_oe_in[T, R](ctx, L, tp[T](s), tp[T](o), tp[T](oref), tp[T](hist), D.OBS, D.HIST * D.OBS, tp[T](eoin))
        var e = talloc[T](L * D.LATENT, ctx)
        self.enc.forward[T, L](TensorRefs[1, _, DT](eoin), e, ctx)
        sm_oe_mix[T](ctx, L * D.LATENT, Scalar[DT](self.gain), tp[T](e), tp[T](z))
        var din = talloc[T](L * D.DEC_IN, ctx)
        sm_dec_in[T, R](ctx, L, tp[T](z), tp[T](hist), D.OBS, D.HIST * D.OBS, tp[T](din))
        var a = talloc[T](L * R.NDOF, ctx)
        self.base.pol.dec.forward[T, L](TensorRefs[1, _, DT](din), a, ctx)
        comptime if T == "gpu":
            if not resident:
                a.download(ctx.value())
        return a^

    def om_update[B: Int](
        mut self, mut w: SMWindowBatch[Self.OC.BASE.ROBOT], mut wo: SMObjWindowBatch, resident: Bool = False
    ) raises -> SMObjLoss:
        """One step of the object network on a batch of windows, the tracker's
        world model frozen beside it."""
        comptime T = Self.OC.BASE.TARGET
        var ctx = self.base.ctx
        comptime if T == "gpu":
            if not resident:
                w.to_device(ctx.value())
                wo.to_device(ctx.value())
        var r = self.om_blk.step[T, B](self.base.wm, self.om, w, wo, ctx)
        return r[0].copy()

    def policy_update[N: Int](
        mut self,
        mut ib: SMObjImagBatch[Self.OC.BASE.ROBOT],
        perms: List[Int],
        mut values_out: List[Float64],
        mut targets_out: List[Float64],
        obs_noise: Bool = False,
        resident: Bool = False,
        terms: Bool = False,
    ) raises -> SMObjUpdate:
        """The object encoder's step on one rollout, then the critic's fit on
        it. `terms`: also return the object's three reward terms (a download)."""
        comptime T = Self.OC.BASE.TARGET
        comptime H = Self.OC.BASE.HORIZON
        if ib.base.h != H:
            raise Error("policy_update: a batch of horizon " + String(ib.base.h) + ", the config's is " + String(H))
        var ctx = self.base.ctx
        comptime if T == "gpu":
            if not resident:
                ib.to_device(ctx.value())
        var ro = self.actor_blk.step[T, N](
            self.base.wm, self.base.pol, self.om, self.enc, self.critic, self.base.tables, ib, self.gain, obs_noise, ctx
        )
        var cf = self.critic_blk.step[T, N, H](self.critic, ro[0], ib, Self.OC.BASE.DEF_GAMMA, perms, values_out, targets_out, ctx)
        var tm: List[Float64] = [-1.0, -1.0, -1.0]
        if terms:
            comptime if T == "gpu":
                ro[0].oterms.download(ctx.value())
            for k in range(SM_OBJ_TERMS):
                var acc = 0.0
                for i in range(H * N):
                    acc += Float64(ro[0].oterms.data[i * SM_OBJ_TERMS + k])
                tm[k] = acc / Float64(H * N)
        return SMObjUpdate(
            ro[0].base.j, ro[0].base.return_undisc, ro[1], cf.last_loss, cf.value_mean, cf.target_mean, tm[0], tm[1], tm[2],
        )
