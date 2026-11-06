"""SMPromptTrainer — a tracker, frozen, with a prompt interface on top.

Owns the tracker's trainer (`base`: the world model, which keeps fitting, and
the policy, which does not move) and the two networks that train: the prompt
encoder and its critic. The agent feeds it batches:

    act[L](hist, goals, cmds, split)   lanes below `split` run the tracker,
                                       the others the prompt interface
    wm_update[B](windows)              Eq. 1, the tracker's own step
    policy_update[N](batch)            the encoder, then its critic

⚠ `final.ckpt`'s motion encoder and decoder are never written: tracking through
them cannot change, which is the design's point. The prompt's checkpoint holds
the world model, the prompt encoder and its critic, and needs the tracker's
checkpoint beside it.
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.blocks.prompt_step import PromptActorStep, PromptCriticStep
from noeira.deep_agents.supermimic.config import SMConfigT
from noeira.deep_agents.supermimic.lanes import sm_dec_in, sm_enc_in, talloc, tp
from noeira.deep_agents.supermimic.prompt_imagination import SMPromptBatch
from noeira.deep_agents.supermimic.prompt_lanes import sm_p_enc_in, sm_p_merge
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfigT, SMPromptCriticOf, SMPromptEncOf
from noeira.deep_agents.supermimic.robot import SMDims
from noeira.deep_agents.supermimic.trainer import SMUpdateMetrics, SuperMimicTrainer, _ScaleParam
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss
from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import CheckpointScalars, load_params_multi, save_params_multi
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs


@fieldwise_init
struct SMPromptUpdate(Copyable, Movable):
    var imag_objective: Float64
    var imag_return_undisc: Float64
    var align_mse: Float64
    var policy_grad_norm: Float64
    var critic_loss: Float64
    var critic_value_mean: Float64
    var critic_target_mean: Float64


struct SMPromptTrainer[P: SMPromptConfigT](Movable):
    var base: SuperMimicTrainer[Self.P.BASE]
    var enc: SMPromptEncOf[Self.P]
    var critic: SMPromptCriticOf[Self.P]
    var actor_blk: PromptActorStep[Self.P]
    var critic_blk: PromptCriticStep[Self.P]
    var w_align: Float64

    def __init__(out self, var base: SuperMimicTrainer[Self.P.BASE]) raises:
        """Fresh prompt networks over `base`, on its target."""
        comptime T = Self.P.BASE.TARGET
        var ctx = base.ctx
        self.enc = SMPromptEncOf[Self.P].make[T, Kaiming](ctx)
        # the tracker's latents sit near 0.24 per entry. Fresh, this layer would
        # saturate its tanh, and the decoder answers such a latent with violent
        # actions (measured: −1 per step against a +0.08 maximum, all of it the
        # action-rate price): start at the centre of what the decoder knows
        var sc = _ScaleParam(String("2.weight"), Scalar[DT](0.01))
        self.enc.for_each_param[T](sc, ctx)
        if sc.hit != 1:
            raise Error("SMPromptTrainer: the encoder's last layer was not found to rescale")
        self.critic = SMPromptCriticOf[Self.P].make[T, Kaiming](ctx)
        self.base = base^
        self.actor_blk = PromptActorStep[Self.P]()
        self.critic_blk = PromptCriticStep[Self.P]()
        self.w_align = Self.P.DEF_ALIGN
        # the tracker's termination labels mean nothing on prompt lanes, and
        # its head is not read here: no BCE reaches the dynamics
        self.base.wm_blk.w_term = 0.0

    def __init__(out self, *, deinit move: Self):
        self.base = move.base^
        self.enc = move.enc^
        self.critic = move.critic^
        self.actor_blk = move.actor_blk^
        self.critic_blk = move.critic_blk^
        self.w_align = move.w_align

    # ── checkpoint: the world model and the two prompt networks ──
    def save_state(mut self, path: String, iteration: Int, updates: Int) raises:
        var sc = CheckpointScalars()
        sc.set_int("iteration", iteration)
        sc.set_int("updates", updates)
        save_params_multi[Self.P.BASE.TARGET](path, self.base.ctx, False, self.base.wm, self.enc, self.critic, scalars=sc)

    def load_state(mut self, path: String) raises -> Tuple[Int, Int]:
        var sc = load_params_multi[Self.P.BASE.TARGET](path, self.base.ctx, self.base.wm, self.enc, self.critic)
        return (sc.get_int("iteration", 0), sc.get_int("updates", 0))

    def set_lr(mut self, lr: Float64, wm_lr: Float64):
        """`lr` for the encoder and its critic. The world model starts trained:
        it keeps fitting at its own lower `wm_lr`."""
        self.actor_blk.opt.lr = Scalar[DT](lr)
        self.critic_blk.opt.lr = Scalar[DT](lr)
        self.base.wm_blk.opt.lr = Scalar[DT](wm_lr)

    def act[L: Int](
        mut self, mut hist: TensorImpl[DT], mut goals: TensorImpl[DT], mut cmds: TensorImpl[DT], split: Int, resident: Bool = False
    ) raises -> TensorImpl[DT]:
        """`hist` [L · 10 · OBS], `goals` [L · GOAL] (read below `split`), `cmds`
        [L · CMD] (read from `split` on) → actions [L · n]. `resident`: inputs
        and actions stay on the device."""
        comptime T = Self.P.BASE.TARGET
        comptime R = Self.P.BASE.ROBOT
        comptime D = SMDims[R]
        comptime CMD = Self.P.CMD
        var ctx = self.base.ctx
        comptime if T == "gpu":
            if not resident:
                hist.upload_resident(ctx.value())
                goals.upload_resident(ctx.value())
                cmds.upload_resident(ctx.value())
        var z = talloc[T](L * D.LATENT, ctx)
        if split > 0:
            var ein = talloc[T](L * D.GOAL, ctx)
            sm_enc_in[T, R](ctx, L, tp[T](goals), tp[T](ein))
            self.base.pol.enc.forward[T, L](TensorRefs[1, _, DT](ein), z, ctx)
        if split < L:
            var pin = talloc[T](L * (CMD + D.HIST_FLAT), ctx)
            sm_p_enc_in[T, R, CMD](ctx, L, tp[T](cmds), tp[T](hist), D.OBS, D.HIST * D.OBS, tp[T](pin))
            var zp = talloc[T](L * D.LATENT, ctx)
            self.enc.forward[T, L](TensorRefs[1, _, DT](pin), zp, ctx)
            sm_p_merge[T, D.LATENT](ctx, L, split, tp[T](zp), tp[T](z))
        var din = talloc[T](L * D.DEC_IN, ctx)
        sm_dec_in[T, R](ctx, L, tp[T](z), tp[T](hist), D.OBS, D.HIST * D.OBS, tp[T](din))
        var a = talloc[T](L * R.NDOF, ctx)
        self.base.pol.dec.forward[T, L](TensorRefs[1, _, DT](din), a, ctx)
        comptime if T == "gpu":
            if not resident:
                a.download(ctx.value())
        return a^

    def wm_update[B: Int](mut self, mut w: SMWindowBatch[Self.P.BASE.ROBOT], resident: Bool = False) raises -> SMWMLoss:
        return self.base.wm_update[B](w, resident)

    def policy_update[N: Int](
        mut self,
        mut ib: SMPromptBatch[Self.P.BASE.ROBOT, Self.P.CMD],
        perms: List[Int],
        mut values_out: List[Float64],
        mut targets_out: List[Float64],
        obs_noise: Bool = False,
        resident: Bool = False,
    ) raises -> SMPromptUpdate:
        """The encoder's step on one rollout, then the critic's fit on it."""
        comptime T = Self.P.BASE.TARGET
        comptime H = Self.P.BASE.HORIZON
        if ib.h != H:
            raise Error("policy_update: a batch of horizon " + String(ib.h) + ", the config's is " + String(H))
        var ctx = self.base.ctx
        comptime if T == "gpu":
            if not resident:
                ib.to_device(ctx.value())
        var ro = self.actor_blk.step[T, N](
            self.base.wm, self.base.pol, self.enc, self.critic, self.base.tables, ib, self.w_align, obs_noise, ctx
        )
        var cf = self.critic_blk.step[T, N, H](self.critic, ro[0], ib, Self.P.BASE.DEF_GAMMA, perms, values_out, targets_out, ctx)
        return SMPromptUpdate(ro[0].j, ro[0].return_undisc, ro[0].align_mse, ro[1], cf.last_loss, cf.value_mean, cf.target_mean)
