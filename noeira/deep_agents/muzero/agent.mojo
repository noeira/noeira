"""MuZeroAgent — facade over the self-play driver + greedy eval + checkpointing.

Mirrors `AlphaZeroAgent`, but MuZero carries a *learned model* — three nets
(h/g/f = representation / dynamics / prediction) with three optimizers — instead
of AlphaZero's single net + true game rules. Parameterized on `TARGET`
("cpu" / "gpu") like the SAC dual-path. The GPU path is **fully on-device
Gumbel MuZero** (`selfplay_gpu_device.mojo`): `GumbelGPUMCTS` searches over the
resident device nets and the K-step BPTT unroll trains them in place — the
validated GPU configuration (CartPole greedy 500 by 4k). ``MAX_K`` is the
Gumbel root-candidate count (default = ``ACT``). The older CPU-search /
GPU-train hybrid (`selfplay_gpu.mojo`) remains available as a standalone
driver. `save` / `load` are device-aware.

  * `__init__(ctx, lr, gamma, v_min, v_max, value_coef)` — build fresh Kaiming
    rep/dyn/pred nets on `TARGET`. `v_min`/`v_max` are the **h-space** value/reward
    support shared with the targets and the planner.
  * `train(env, iterations, ...)` — single-player self-play (env → CPU MCTS over
    the learned model → sequence replay → K-step BPTT unroll). Optimizers are
    session-local (recreated per call), nets persist across calls.
  * `eval_greedy(env, episodes)` — the deployed-agent metric: noise-off,
    argmax-visit rollout. ALWAYS judge the policy by this, not the exploratory
    training return (∝-visit sampling + root Dirichlet noise badly understates it).
  * `save` / `load` — one-file v3 `storage-ckpt` file packing all three nets
    (sections `rep` / `dyn` / `pred`).

The three nets share `LATENT` + `BINS`; reward (`dyn`) and value (`pred`) heads
are categorical over `[v_min, v_max]` — keep those in sync with the config.
"""

from noeira.nn.core.ptr import untracked
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Kaiming
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.checkpoint import save_params, load_params
from noeira.core.env_traits import BoxDiscreteActionEnv
from noeira.planners.tree_search import (
    GenericCPUMCTS,
    GumbelGPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    SinglePlayer,
)

from noeira.core.logger import Logger, NoOpLogger
from .selfplay_cpu import run_muzero_selfplay_cpu
from .selfplay_gpu import run_muzero_selfplay_gpu, mz_sync_gpu_to_cpu
from .selfplay_gpu_device import run_muzero_gumbel_selfplay_gpu
from ..zero.mcts_adapters_mz_cpu import MZRepCPU, MZDynCPU, MZPredCPU
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU


@fieldwise_init
struct MuZeroAgent[
    TARGET: StaticString,
    ENV: BoxDiscreteActionEnv & Deinitable,
    REP: Module,
    DYN: Module,
    PRED: Module,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
    MAX_K: Int = ACT,
](Deinitable, Movable):
    var ctx: Optional[DeviceContext]
    var rep: Self.REP
    var dyn: Self.DYN
    var pred: Self.PRED
    var lr: Scalar[DT]
    var gamma: Scalar[DT]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]
    var value_coef: Scalar[DT]
    var max_grad_norm: Scalar[DT]

    def __init__(
        out self,
        ctx: Optional[DeviceContext],
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.997),
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
        value_coef: Scalar[DT] = Scalar[DT](0.25),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
    ) raises:
        self.ctx = ctx
        self.rep = Self.REP.make[Self.TARGET, Kaiming](ctx)
        self.dyn = Self.DYN.make[Self.TARGET, Kaiming](ctx)
        self.pred = Self.PRED.make[Self.TARGET, Kaiming](ctx)
        self.lr = lr
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.value_coef = value_coef
        # Global grad-norm clip applied to all three session-local optimizers in
        # `train` (0.0 = off, Adam's default). The CartPole v2 convergence runs
        # need 10.0 — keep it on the agent so the facade reproduces them.
        self.max_grad_norm = max_grad_norm

    def train[
        L: Logger = NoOpLogger,
    ](
        mut self,
        mut env: Self.ENV,
        iterations: Int,
        learning_starts: Int = 500,
        train_per_iter: Int = 1,
        seed: UInt64 = 0,
        max_ep_steps: Int = 500,
        temperature_decay_steps: Int = 0,
        reanalyze_every: Int = 0,
        eval_every: Int = 0,
        eval_episodes: Int = 5,
        diag_every: Int = 0,
        report_every: Int = 0,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        verbose: Bool = False,
    ) raises -> Float64:
        """Single-player self-play training over the learned model. Returns the
        last training loss. Optimizers are recreated here (session-local); the
        nets keep their weights across `train` calls.

        ``temperature_decay_steps`` schedules the visit-sampling temperature
        (1.0 → 0.5 → 0.25; 0 = always 1.0). ``reanalyze_every`` refreshes one
        stored (policy, root value) per that many iterations with a fresh
        search — both were required for CartPole to reach sustained greedy 500
        (see the convergence example)."""
        comptime if Self.TARGET == "cpu":
            var orep = Adam(lr=self.lr)
            var odyn = Adam(lr=self.lr)
            var opred = Adam(lr=self.lr)
            return run_muzero_selfplay_cpu[
                Self.ENV, Self.REP, Self.DYN, Self.PRED,
                Self.OBS, Self.ACT, Self.LATENT, Self.BINS,
                Self.NUM_SIMS, Self.MAX_NODES, Self.CAP,
                Self.B, Self.K, Self.N,
                L=L,
            ](
                env, self.rep, self.dyn, self.pred, orep, odyn, opred,
                iterations=iterations,
                learning_starts=learning_starts,
                train_per_iter=train_per_iter,
                gamma=self.gamma,
                v_min=self.v_min,
                v_max=self.v_max,
                seed=seed,
                max_ep_steps=max_ep_steps,
                value_coef=self.value_coef,
                max_grad_norm=Float64(self.max_grad_norm),
                temperature_decay_steps=temperature_decay_steps,
                reanalyze_every=reanalyze_every,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                diag_every=diag_every,
                report_every=report_every,
                logger=logger,
                verbose=verbose,
            )
        else:
            var c = self.ctx.value()
            var orep = Adam(lr=self.lr)
            var odyn = Adam(lr=self.lr)
            var opred = Adam(lr=self.lr)
            # Fully on-device Gumbel MuZero — the validated GPU path (greedy
            # 500 by 4k on the CartPole lighthouse, vs the hybrid's mirror-sync
            # overhead and the vanilla device search's NoNoise eval anomaly).
            return run_muzero_gumbel_selfplay_gpu[
                Self.ENV, Self.REP, Self.DYN, Self.PRED,
                Self.OBS, Self.ACT, Self.LATENT, Self.BINS,
                Self.NUM_SIMS, Self.MAX_NODES, Self.MAX_K, Self.CAP,
                Self.B, Self.K, Self.N,
                L=L,
            ](
                c, env, self.rep, self.dyn, self.pred, orep, odyn, opred,
                iterations=iterations,
                learning_starts=learning_starts,
                train_per_iter=train_per_iter,
                gamma=self.gamma,
                v_min=self.v_min,
                v_max=self.v_max,
                seed=seed,
                max_ep_steps=max_ep_steps,
                value_coef=self.value_coef,
                max_grad_norm=Float64(self.max_grad_norm),
                temperature_decay_steps=temperature_decay_steps,
                reanalyze_every=reanalyze_every,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                diag_every=diag_every,
                report_every=report_every,
                logger=logger,
                verbose=verbose,
            )

    def eval_greedy(
        mut self,
        mut env: Self.ENV,
        episodes: Int = 10,
        max_ep_steps: Int = 500,
    ) raises -> Float64:
        """Deployed-agent metric: greedy (noise-off, argmax-visit) rollout mean
        return. This is the honest measure of the policy — the training return is
        exploratory and understates it."""
        comptime if Self.TARGET == "cpu":
            var mcts = GenericCPUMCTS[
                Self.ACT, Self.LATENT, Self.NUM_SIMS, Self.MAX_NODES,
                MuZeroPUCT[19652.0, 1.25],
                DirichletNoise[0.25, 0.25], SinglePlayer, 8, 3,
            ](gamma=Float64(self.gamma))
            var rep_a = MZRepCPU[Self.OBS, Self.LATENT, Self.REP](
                net=untracked(Pointer(to=self.rep))
            )
            var dyn_a = MZDynCPU[Self.LATENT, Self.ACT, Self.BINS, Self.DYN](
                net=untracked(Pointer(to=self.dyn)),
                v_min=self.v_min, v_max=self.v_max,
            )
            var pred_a = MZPredCPU[
                Self.LATENT, Self.ACT, Self.BINS, Self.PRED
            ](
                net=untracked(Pointer(to=self.pred)),
                v_min=self.v_min, v_max=self.v_max,
            )
            var total = 0.0
            for _ in range(episodes):
                var eo = env.reset_obs_list()
                var eo_f = List[Float64]()
                for j in range(Self.OBS):
                    eo_f.append(Float64(eo[j]))
                var eret = 0.0
                for _step in range(max_ep_steps):
                    var ep = mcts.search[
                        type_of(rep_a), type_of(dyn_a), type_of(pred_a)
                    ](rep_a, dyn_a, pred_a, eo_f, add_noise=False)
                    var best = 0
                    for a in range(1, Self.ACT):
                        if ep[a] > ep[best]:
                            best = a
                    var es = env.step_obs(best)
                    eret += Float64(es[1])
                    eo_f = List[Float64]()
                    for j in range(Self.OBS):
                        eo_f.append(Float64(es[0][j]))
                    if es[2]:
                        break
                total += eret
            return total / Float64(episodes)
        else:
            # Fully on-device: Gumbel search over the resident GPU nets, greedy
            # action = argmax of the improved policy (matches the GPU train
            # path — Gumbel root sampling IS the exploration; argmax is greedy).
            var c = self.ctx.value()
            var planner = GumbelGPUMCTS[
                1, Self.ACT, Self.LATENT, Self.BINS, Self.MAX_NODES,
                Self.MAX_K, Self.NUM_SIMS, SinglePlayer,
            ](
                c, gamma=Float64(self.gamma),
                v_min=Float64(self.v_min), v_max=Float64(self.v_max),
                # Tree-GLOBAL sigma(Q) norm — small-ACT single-player (see
                # GumbelGPUMCTS.qnorm_per_node; per-node degenerate at ACT=2).
                qnorm_per_node=False,
            )
            var rep_a = MZRepGPU[Self.OBS, Self.LATENT, Self.REP].make(
                self.rep
            )
            var dyn_a = MZDynGPU[
                Self.LATENT, Self.ACT, Self.BINS, Self.DYN
            ].make(self.dyn)
            var pred_a = MZPredGPU[
                Self.LATENT, Self.ACT, Self.BINS, Self.PRED
            ].make(self.pred)
            var d_obs = c.enqueue_create_buffer[DT](Self.OBS)
            var h_obs = List[Scalar[DT]](length=Self.OBS, fill=0)
            var h_pol = List[Scalar[DT]](length=Self.ACT, fill=0)
            var mseed = UInt32(0)
            var total = 0.0
            for _ in range(episodes):
                var eo = env.reset_obs_list()
                var eo_f = List[Float64]()
                for j in range(Self.OBS):
                    eo_f.append(Float64(eo[j]))
                var eret = 0.0
                for _step in range(max_ep_steps):
                    for j in range(Self.OBS):
                        h_obs[j] = Scalar[DT](eo_f[j])
                    c.enqueue_copy(d_obs, h_obs.unsafe_ptr())  # H2D (sanctioned)
                    var obs_t = LayoutTensor[
                        DT, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](d_obs)
                    planner.search_gpu[
                        type_of(rep_a), type_of(dyn_a), type_of(pred_a)
                    ](c, rep_a, dyn_a, pred_a, obs_t,
                      apply_legal=False, k_actual=Self.MAX_K, rng_seed=mseed)
                    mseed += UInt32(1)
                    c.enqueue_copy(h_pol.unsafe_ptr(), planner.policies_view())
                    c.synchronize()
                    var best = 0
                    for a in range(1, Self.ACT):
                        if Float64(h_pol[a]) > Float64(h_pol[best]):
                            best = a
                    var es = env.step_obs(best)
                    eret += Float64(es[1])
                    eo_f = List[Float64]()
                    for j in range(Self.OBS):
                        eo_f.append(Float64(es[0][j]))
                    if es[2]:
                        break
                total += eret
            return total / Float64(episodes)

    def save(mut self, path: String) raises:
        """Weights-only snapshot of the three learned-model nets (rep / dyn
        / pred) in the v3 `storage-ckpt` file. Uses the `save`/`load`
        surface shared by every agent facade. NOTE: optimizers are
        session-local — rebuilt per `train_*` call — so only weights
        persist; this is the inference / self-play artifact, not a
        training-resume checkpoint. The three nets go to ``path`` + ``.rep`` /
        ``.dyn`` / ``.pred`` via the storage checkpoint (walks for_each_param then
        for_each_state, owns file I/O)."""
        save_params[Self.TARGET, Self.REP](
            self.rep, path + ".rep", self.ctx, save_moments=False
        )
        save_params[Self.TARGET, Self.DYN](
            self.dyn, path + ".dyn", self.ctx, save_moments=False
        )
        save_params[Self.TARGET, Self.PRED](
            self.pred, path + ".pred", self.ctx, save_moments=False
        )

    def load(mut self, path: String) raises:
        """Inverse of `save` — restores rep / dyn / pred weights from the three
        per-net checkpoints. Optimizer state is not checkpointed (see `save`)."""
        load_params[Self.TARGET, Self.REP](self.rep, path + ".rep", self.ctx)
        load_params[Self.TARGET, Self.DYN](self.dyn, path + ".dyn", self.ctx)
        load_params[Self.TARGET, Self.PRED](self.pred, path + ".pred", self.ctx)
