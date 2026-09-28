"""EZv2DiscreteAgent — facade over the discrete self-play driver + greedy eval +
checkpointing (CPU).

Mirrors `MuZeroAgent`, but EZv2 carries two extra SimSiam nets (projector +
predictor) and trains with the temporal-consistency objective. This is the
**CPU** discrete agent (the lighthouse path, decision D1: CPU uses vanilla PUCT
MCTS). The GPU Gumbel path is a separate build.

  * `__init__(lr, gamma, v_min, v_max, value_coef, consistency_coef)` — build
    fresh Kaiming rep/dyn/pred/proj/predh nets on CPU.
  * `train(env, iterations, ...)` — single-player self-play (env → CPU MCTS over
    the learned model → sequence replay → EZv2 BPTT+consistency unroll).
    Optimizers are session-local (recreated per call); nets persist across calls.
  * `eval_greedy(env, episodes)` — the deployed-agent metric: noise-off,
    argmax-visit rollout (judge the policy by this, not the exploratory training
    return). The projector/predictor are consistency-only, unused here.
  * `save` / `load` — one-file v3 `storage-ckpt` file packing all five nets
    (sections `rep` / `dyn` / `pred` / `proj` / `predh`).
"""

from noeira.nn.core.ptr import untracked
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Kaiming
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.checkpoint import save_params, load_params
from noeira.core.env_traits import BoxDiscreteActionEnv
from noeira.planners.tree_search import (
    GenericCPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    SinglePlayer,
)

from noeira.core.logger import Logger, NoOpLogger
from .selfplay_cpu import run_ezv2_selfplay_cpu
from ..zero.mcts_adapters_mz_cpu import MZRepCPU, MZDynCPU, MZPredCPU


@fieldwise_init
struct EZv2DiscreteAgent[
    ENV: BoxDiscreteActionEnv & Deinitable,
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
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
](Deinitable, Movable):
    var rep: Self.REP
    var dyn: Self.DYN
    var pred: Self.PRED
    var proj: Self.PROJM
    var predh: Self.PREDH
    var lr: Scalar[DT]
    var gamma: Scalar[DT]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]
    var value_coef: Scalar[DT]
    var consistency_coef: Scalar[DT]

    def __init__(
        out self,
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.997),
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
        value_coef: Scalar[DT] = Scalar[DT](0.25),
        consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    ) raises:
        self.rep = Self.REP.make["cpu", Kaiming]()
        self.dyn = Self.DYN.make["cpu", Kaiming]()
        self.pred = Self.PRED.make["cpu", Kaiming]()
        self.proj = Self.PROJM.make["cpu", Kaiming]()
        self.predh = Self.PREDH.make["cpu", Kaiming]()
        self.lr = lr
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.value_coef = value_coef
        self.consistency_coef = consistency_coef

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
        reanalyze_every: Int = 0,
        reanalyze_batch: Int = 1,
        eval_every: Int = 0,
        eval_episodes: Int = 5,
        diag_every: Int = 0,
        report_every: Int = 0,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        verbose: Bool = False,
    ) raises -> Float64:
        """Single-player self-play training (MuZero BPTT + SimSiam consistency).
        Returns the last training loss. Optimizers recreated here; nets persist."""
        var orep = Adam(lr=self.lr)
        var odyn = Adam(lr=self.lr)
        var opred = Adam(lr=self.lr)
        var oproj = Adam(lr=self.lr)
        var opredh = Adam(lr=self.lr)
        return run_ezv2_selfplay_cpu[
            Self.ENV, Self.REP, Self.DYN, Self.PRED, Self.PROJM, Self.PREDH,
            Self.OBS, Self.ACT, Self.LATENT, Self.BINS,
            Self.NUM_SIMS, Self.MAX_NODES, Self.CAP,
            Self.B, Self.K, Self.N,
            L=L,
        ](
            env, self.rep, self.dyn, self.pred, self.proj, self.predh,
            orep, odyn, opred, oproj, opredh,
            iterations=iterations,
            learning_starts=learning_starts,
            train_per_iter=train_per_iter,
            gamma=self.gamma,
            v_min=self.v_min,
            v_max=self.v_max,
            seed=seed,
            max_ep_steps=max_ep_steps,
            value_coef=self.value_coef,
            consistency_coef=self.consistency_coef,
            reanalyze_every=reanalyze_every,
            reanalyze_batch=reanalyze_batch,
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
        return — the honest measure of the policy."""
        var mcts = GenericCPUMCTS[
            Self.ACT, Self.LATENT, Self.NUM_SIMS, Self.MAX_NODES,
            MuZeroPUCT[19652.0, 1.25],
            DirichletNoise[0.25, 0.25], SinglePlayer, 8, 3,
        ](gamma=Float64(self.gamma))
        var rep_a = MZRepCPU[Self.OBS, Self.LATENT, Self.REP](
            net=untracked(Pointer(to=self.rep)),
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

    def save(mut self, path: String) raises:
        """Weights-only snapshot of the five EZv2 nets (rep / dyn / pred /
        proj / predh) in the v3 `storage-ckpt` file. Uses the `save`/`load`
        surface shared by every agent facade. NOTE: optimizers are
        session-local — rebuilt per `train_*` call — so only weights
        persist; this is the inference / self-play artifact, not a
        training-resume checkpoint. The five nets go to ``path`` + .rep/.dyn/
        .pred/.proj/.predh via the storage checkpoint (per-net file)."""
        save_params["cpu", Self.REP](self.rep, path + ".rep", None, save_moments=False)
        save_params["cpu", Self.DYN](self.dyn, path + ".dyn", None, save_moments=False)
        save_params["cpu", Self.PRED](self.pred, path + ".pred", None, save_moments=False)
        save_params["cpu", Self.PROJM](self.proj, path + ".proj", None, save_moments=False)
        save_params["cpu", Self.PREDH](self.predh, path + ".predh", None, save_moments=False)

    def load(mut self, path: String) raises:
        """Inverse of `save` — restores all five net weights from the per-net
        sidecars. Optimizer state is not checkpointed (session-local)."""
        load_params["cpu", Self.REP](self.rep, path + ".rep", None)
        load_params["cpu", Self.DYN](self.dyn, path + ".dyn", None)
        load_params["cpu", Self.PRED](self.pred, path + ".pred", None)
        load_params["cpu", Self.PROJM](self.proj, path + ".proj", None)
        load_params["cpu", Self.PREDH](self.predh, path + ".predh", None)
