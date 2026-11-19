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

from std.math import abs
from std.os.path import exists

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.blocks import ActStep, ActorStep, CriticFitStep, WorldModelUpdateStep
from noeira.deep_agents.supermimic.config import SMConfigT, SMCriticOf, SMPolicyOf, SMWorldModelOf
from noeira.deep_agents.supermimic.imagination import SMImagBatch
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import sm_mirror_windows, tp
from noeira.deep_agents.supermimic.state_spec import SM_WM_DRIFT_MILLI
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
    # The fast-training options (`docs/SUPERMIMIC_FAST_TRAINING_PLAN.md`), all
    # off unless set: the trainer is then the reproduction's.
    var pol_avg: SMPolicyOf[Self.C]     # the DEPLOYED policy when `ema` > 0: a running average of `pol`
    var ema: Float64                    # its decay per policy update (0 = off: `pol` is deployed)
    var ema_n: Int                      # updates averaged so far (the first one copies `pol`)
    var ref_skip: Bool                  # actions = decoder output + the action commanding the target pose
    var skip_k: TensorImpl[DT]          # 1 / action scale per joint, on the config's target
    var analytic_term: Bool             # imagination terminates on the real thresholds, not on the learned head
    var v_max: Float64                  # the value ceiling in the bootstrap and the critic's targets (0 = none)
    var mirror_share: Float64           # share of each world-model batch replaced by its left-right mirror (0 = none)
    var mirror_tab: TensorImpl[DT]      # [partner n, sign n] per joint, host and (on "gpu") device

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
        self.pol_avg = SMPolicyOf[Self.C].make[Kaiming, T](ctx)
        self.ema = 0.0
        self.ema_n = 0
        self.ref_skip = False
        self.skip_k = TensorImpl[DT].alloc(Self.C.ROBOT.NDOF)
        self.analytic_term = False
        self.v_max = 0.0
        self.mirror_share = 0.0
        self.mirror_tab = TensorImpl[DT].alloc(2 * Self.C.ROBOT.NDOF)

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
        self.pol_avg = move.pol_avg^
        self.ema = move.ema
        self.ema_n = move.ema_n
        self.ref_skip = move.ref_skip
        self.skip_k = move.skip_k^
        self.analytic_term = move.analytic_term
        self.v_max = move.v_max
        self.mirror_share = move.mirror_share
        self.mirror_tab = move.mirror_tab^

    # ── the fast-training options ──
    def set_ema(mut self, decay: Float64):
        """Deploy a running average of the policy (InstantTrack's `--pi_ema`):
        after every policy update `pol_avg ← decay·pol_avg + (1 − decay)·pol`,
        the first update copying `pol`. Checkpoints, the benchmark and
        `act(deployed=True)` then use `pol_avg`; collection and the update
        keep `pol`. 0 turns it off."""
        self.ema = decay
        self.ema_n = 0

    def set_ref_skip(mut self, on: Bool) raises:
        """The reference skip (BlinkMimic's `--pi_ref_skip`): every action is
        the decoder's output plus `(q_ref − q0) / scale` of the step's target
        frame, so an untrained policy already commands the reference pose.
        Stored in the checkpoint; a policy trained with it is meaningless
        without it."""
        if on:
            for j in range(Self.C.ROBOT.NDOF):
                var s = Self.C.ROBOT.action_scale(j)
                if s <= 0.0:
                    raise Error("SuperMimicTrainer.set_ref_skip: this robot declares no action scale (`TrackingRobot.action_scale`)")
                self.skip_k.data[j] = Scalar[DT](1.0 / s)
            comptime if Self.C.TARGET == "gpu":
                self.skip_k.upload_resident(self.ctx.value())
        self.ref_skip = on

    def require_no_ref_skip(self, who: String) raises:
        """For a reader that runs this tracker's decoder itself, outside `act`
        and the tracker's own imagined rollout (the prompt interface, the
        object encoder): it does not add the reference skip, so a tracker
        trained with it would be run without it and nothing would say so.
        Raises instead."""
        if self.ref_skip:
            raise Error(
                who + ": this tracker was trained with the reference skip (`--ref-skip`), which " + who
                + " does not add to its actions; give it a tracker trained without the skip"
            )

    def set_mirror(mut self, share: Float64) raises:
        """Replace this share of every world-model batch by its left-right
        mirror image (BlinkMimic's `--wm_mirror 0.5`): the robot is
        symmetric, so the reflection of a logged transition is another valid
        one, and the world model sees twice the states for the same
        simulation. The joint table comes from the robot's joint names:
        `left_` pairs with `right_`, `_roll_` and `_yaw_` joints change sign.
        It refuses a robot whose names, default pose or action scales are
        not mirror-consistent."""
        comptime n = Self.C.ROBOT.NDOF
        if share < 0.0 or share > 1.0:
            raise Error("SuperMimicTrainer.set_mirror: a share in [0, 1], not " + String(share))
        if share > 0.0:
            var names = List[String]()
            for j in range(n):
                var nm = Self.C.ROBOT.joint_name(j)
                if nm.byte_length() == 0:
                    raise Error("SuperMimicTrainer.set_mirror: this robot declares no joint names (`TrackingRobot.joint_name`)")
                names.append(nm^)
            for j in range(n):
                var partner = names[j].copy()
                if names[j].startswith("left_"):
                    partner = String("right_") + String(names[j].removeprefix("left_"))
                elif names[j].startswith("right_"):
                    partner = String("left_") + String(names[j].removeprefix("right_"))
                var pj = -1
                for i in range(n):
                    if names[i] == partner:
                        pj = i
                if pj < 0:
                    raise Error("SuperMimicTrainer.set_mirror: " + names[j] + " has no partner " + partner)
                var flip = names[j].find("_roll_") >= 0 or names[j].find("_yaw_") >= 0
                self.mirror_tab.data[j] = Scalar[DT](pj)
                self.mirror_tab.data[n + j] = Scalar[DT](-1.0) if flip else Scalar[DT](1.0)
            for j in range(n):
                var pj = Int(self.mirror_tab.data[j])
                var sg = Float64(self.mirror_tab.data[n + j])
                if Int(self.mirror_tab.data[pj]) != j or Float64(self.mirror_tab.data[n + pj]) != sg:
                    raise Error("SuperMimicTrainer.set_mirror: the joint pairs are not symmetric at " + names[j])
                # the action a commands q0 + scale·a: its mirror is only the mirrored action if both are symmetric
                if abs(sg * Self.C.ROBOT.default_joint_pos(pj) - Self.C.ROBOT.default_joint_pos(j)) > 1e-9:
                    raise Error("SuperMimicTrainer.set_mirror: the default pose is not mirror-symmetric at " + names[j])
                if abs(Self.C.ROBOT.action_scale(pj) - Self.C.ROBOT.action_scale(j)) > 1e-12 or Self.C.ROBOT.action_scale(j) <= 0.0:
                    raise Error("SuperMimicTrainer.set_mirror: the action scale is not mirror-symmetric (or not declared) at " + names[j])
            comptime if Self.C.TARGET == "gpu":
                self.mirror_tab.upload_resident(self.ctx.value())
        self.mirror_share = share

    def set_muon(mut self, on: Bool, wd: Float64 = 0.0) raises:
        """Muon (`nn/optimizer/muon.mojo`) on the hidden weight matrices of
        the dynamics, the encoder and the decoder: every Linear between the
        first and the last of each MLP, as in BlinkMimic's `MuonAdam` and
        the official code. Adam keeps the first and last layers, the biases,
        the termination head and the critic. `wd`: Muon's decoupled weight
        decay (BlinkMimic 0, torch's default and the official code 0.1).
        Turn it on BEFORE the first update: Muon hands its matrices to Adam
        with a zero gradient, which leaves them alone only while Adam's
        moments for them are zero. After an Adam step they are not, and Adam
        would keep moving those matrices beside Muon. (A resumed run starts
        with fresh moments, so it qualifies.)"""
        if on and not self.wm_blk.muon_on and (self.wm_blk.opt.t > 0 or self.actor_blk.opt.t > 0):
            raise Error("SuperMimicTrainer.set_muon: Adam has already stepped; Muon must be on from the first update")
        if on and len(self.wm_blk.muon.names) == 0:
            comptime W = Self.C.WM_H
            self.wm_blk.muon.add(String("dynamics.1.weight"), W, W)
            self.wm_blk.muon.add(String("dynamics.2.weight"), W, W)
            self.actor_blk.muon.add(String("encoder.1.weight"), Self.C.ENC1, Self.C.ENC2)
            self.actor_blk.muon.add(String("encoder.2.weight"), Self.C.ENC2, Self.C.ENC3)
            self.actor_blk.muon.add(String("encoder.3.weight"), Self.C.ENC3, Self.C.ENC4)
            self.actor_blk.muon.add(String("decoder.1.weight"), Self.C.DEC1, Self.C.DEC2)
            self.actor_blk.muon.add(String("decoder.2.weight"), Self.C.DEC2, Self.C.DEC3)
            self.actor_blk.muon.add(String("decoder.3.weight"), Self.C.DEC3, Self.C.DEC4)
        self.wm_blk.muon.wd = wd
        self.actor_blk.muon.wd = wd
        self.wm_blk.muon_on = on
        self.actor_blk.muon_on = on

    def _skip(mut self) -> Optional[FP[DT]]:
        if self.ref_skip:
            return Optional(tp[Self.C.TARGET](self.skip_k))
        return None

    def _average_policy(mut self) raises:
        """One step of the deployed average, after a policy update."""
        comptime T = Self.C.TARGET
        var tau = Scalar[DT](1.0) if self.ema_n == 0 else Scalar[DT](1.0 - self.ema)
        self.pol_avg.enc.polyak_from[T](self.pol.enc, tau, self.ctx)
        self.pol_avg.dec.polyak_from[T](self.pol.dec, tau, self.ctx)
        self.ema_n += 1

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
        re-warms Adam; the paper-width moments would triple the file).
        With an averaged policy (`set_ema`), `path` holds the AVERAGE in the
        policy's place (what every reader of a checkpoint deploys) and
        `path + ".live"` the live policy, which only a resume reads. The
        `ref_skip` scalar says whether the policy needs the reference skip."""
        var sc = CheckpointScalars()
        sc.set_int("iteration", iteration)
        sc.set_int("updates", updates)
        sc.set_int("ref_skip", 1 if self.ref_skip else 0)
        sc.set_int("wm_drift_milli", SM_WM_DRIFT_MILLI)
        if self.ema > 0.0 and self.ema_n > 0:
            save_params_multi[Self.C.TARGET](path, self.ctx, False, self.wm, self.pol_avg, self.critic, scalars=sc)
            save_params_multi[Self.C.TARGET](path + ".live", self.ctx, False, self.pol, scalars=sc)
        else:
            save_params_multi[Self.C.TARGET](path, self.ctx, False, self.wm, self.pol, self.critic, scalars=sc)

    def load_state(mut self, path: String) raises -> Tuple[Int, Int]:
        """Restore the three networks; returns (iteration, updates). The
        reference skip follows the checkpoint. With an averaged policy set
        BEFORE the load (a resume), the average restarts from the file's
        policy and the live policy comes from `path + ".live"` when it exists."""
        var sc = load_params_multi[Self.C.TARGET](path, self.ctx, self.wm, self.pol, self.critic)
        var drift = sc.get_int("wm_drift_milli", 1000)
        if drift != SM_WM_DRIFT_MILLI:
            raise Error(
                "SuperMimicTrainer.load_state: " + path + " holds a world model trained with the drift weight " + String(drift)
                + "/1000, this binary integrates with " + String(SM_WM_DRIFT_MILLI) + "/1000 (build with -D SM_WM_DRIFT_MILLI=" + String(drift) + ")"
            )
        self.set_ref_skip(sc.get_int("ref_skip", 0) != 0)
        if self.ema > 0.0:
            self.ema_n = 0
            self._average_policy()
            if exists(path + ".live"):
                _ = load_params_multi[Self.C.TARGET](path + ".live", self.ctx, self.pol)
        return (sc.get_int("iteration", 0), sc.get_int("updates", 0))

    def set_lr(mut self, lr: Float64):
        """The same lr for world model, policy and critic (`Trainer.run`)."""
        self.set_lrs(lr, lr, lr)

    def set_lrs(mut self, lr_wm: Float64, lr_pi: Float64, lr_v: Float64):
        """One lr per network (BlinkMimic's `--pi_lr_mult` raises the policy's)."""
        self.wm_blk.opt.lr = Scalar[DT](lr_wm)
        self.actor_blk.opt.lr = Scalar[DT](lr_pi)
        self.critic_blk.opt.lr = Scalar[DT](lr_v)

    def act[L: Int](
        mut self, mut hist: TensorImpl[DT], mut goals: TensorImpl[DT], resident: Bool = False, deployed: Bool = False,
    ) raises -> TensorImpl[DT]:
        """Host `hist` / `goals` in, host actions out (on "gpu" via the
        device); `resident`: inputs and actions stay on the device.
        `deployed`: the averaged policy when there is one (`set_ema`), the
        policy a checkpoint of this trainer would hold."""
        comptime T = Self.C.TARGET
        comptime if T == "gpu":
            if not resident:
                hist.upload_resident(self.ctx.value())
                goals.upload_resident(self.ctx.value())
        var sk = self._skip()
        var a: TensorImpl[DT]
        if deployed and self.ema > 0.0 and self.ema_n > 0:
            a = self.act_blk.step[T, L](self.pol_avg, hist, goals, self.ctx, sk)
        else:
            a = self.act_blk.step[T, L](self.pol, hist, goals, self.ctx, sk)
        comptime if T == "gpu":
            if not resident:
                a.download(self.ctx.value())
        return a^

    def wm_update[B: Int](mut self, mut w: SMWindowBatch[Self.C.ROBOT], resident: Bool = False) raises -> SMWMLoss:
        """`resident`: the windows are already on the device. With a mirror
        share (`set_mirror`), the first share·B windows are reflected IN
        PLACE first, where the batch lives."""
        var nm = Int(self.mirror_share * Float64(B))
        comptime if Self.C.TARGET == "gpu":
            if resident and nm > 0:
                sm_mirror_windows["gpu", Self.C.ROBOT](
                    self.ctx, nm, w.k, B, tp["gpu"](w.states), tp["gpu"](w.a_exec), tp["gpu"](w.xi), tp["gpu"](w.ref_root), tp["gpu"](self.mirror_tab),
                )
        if nm > 0 and not (resident and Self.C.TARGET == "gpu"):
            sm_mirror_windows["cpu", Self.C.ROBOT](
                None, nm, w.k, B, tp["cpu"](w.states), tp["cpu"](w.a_exec), tp["cpu"](w.xi), tp["cpu"](w.ref_root), tp["cpu"](self.mirror_tab),
            )
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
        var ro = self.actor_blk.step[Self.C.TARGET, N](
            self.wm, self.pol, self.critic, self.tables, ib, obs_noise, self.ctx, self._skip(), self.analytic_term, self.v_max,
        )
        if self.ema > 0.0:
            self._average_policy()
        var cf = self.critic_blk.step[Self.C.TARGET, N, Self.C.HORIZON](
            self.critic, ro[0], ib, Self.C.DEF_GAMMA, perms, values_out, targets_out, self.ctx, self.v_max
        )
        return SMUpdateMetrics(ro[0].j, ro[0].return_undisc, ro[1], cf.last_loss, cf.value_mean, cf.target_mean)
