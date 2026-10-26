"""DDPGTargetYBlock — deterministic target-value compute (storage ComputeGraph).

Computes  y[b] = r[b] + γ·(1−term[b])·Q_t(s', μ_t(s'))  with the deterministic
TARGET actor μ_t and the TARGET critic. Sibling of SAC's `TargetYBlock` and TD3's
`TD3TargetYBlock`, DDPG-specific shape: deterministic actor, single critic, no
log-prob / min reduction.

STORAGE migration: name-wired ComputeGraph (`InputSlot`/`Node`/`ExternalNode`),
the target actor + target critic threaded as tracked `mut` ExternalNodes. The
graph computes only the BOOTSTRAP `γ·Q'` (Scale node); `step` then folds in the
reward + terminal mask via the shared `apply_terminal_mask`
(`y = r + (1−term)·γ·Q'`). Forward-only — `y` is a target, no grad flows.

  graph: s' → actor → action ; (s', action) → concat → q' ; γ·q' (output)

The actor is Tanh-bounded (range [-1,1]), so the legacy `clamp(±action_scale)`
on the target action is a no-op and is omitted — the critic sees the raw Tanh
action, bit-consistent with `DDPGActorLoss` (which also feeds the raw action to
the critic). `action_scale` is applied at env-interaction time, not inside the
critic/target.

Surface:
    DDPGTargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]
        make[target](action_scale, gamma, ctx) -> Self
        step[target, POLICY](mut state, mut actor_t, mut critic_t)
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Zero
from noeira.nn.primitives.concat import Concat2
from noeira.nn.primitives.scale import Scale
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import InputSlot, Node, ExternalNode
from ..loss.loss_block import LossBlock
from ..training.terminal_mask import apply_terminal_mask
from ..training.trainer_block import TrainerState


struct DDPGTargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT

    # Graph computes the BOOTSTRAP `gamma_q = γ·Q'`; the reward add + terminal
    # mask `y = r + (1−term)·gamma_q` happen in `step` (per-sample data).
    comptime Graph = ComputeGraph[
        InputSlot["sp", Self.OBS],                             # s'
        ExternalNode["a_sp", Self.ACTOR, "sp"],                # → action [B,ACT]
        Node["sa", Concat2[Self.OBS, Self.ACT], "sp", "a_sp"],
        ExternalNode["q", Self.CRITIC, "sa"],                  # → [B, 1]
        Node["gamma_q", Scale[1], "q"],                        # γ·Q' (output)
    ]

    var graph: Self.Graph
    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]

    def __init__(out self):
        self.graph = Self.Graph()
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)

    @staticmethod
    def make[
        target: StaticString
    ](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "DDPGTargetYBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACTOR.IN_DIMS[0] == Self.OBS
        ), "DDPGTargetYBlock: ACTOR.IN_DIM must equal OBS"
        comptime assert (
            Self.ACTOR.OUT_DIM == Self.ACT
        ), "DDPGTargetYBlock: ACTOR.OUT_DIM must equal ACT"
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "DDPGTargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "DDPGTargetYBlock: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        blk.graph = Self.Graph.make[target, Zero](ctx)
        blk.graph.set_node_attr["gamma_q", "multiplier"](gamma)  # γ (constant)
        blk.action_scale = action_scale
        blk.gamma = gamma
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_t: Self.ACTOR,
        mut critic_t: Self.CRITIC,
    ) raises:
        """Write `state.mb_y[b] = r[b] + (1−term[b])·γ·Q_t(s', μ_t(s'))`."""
        var ctx = state.ctx
        # Seed s' (a COPY into the graph pool), then forward → γ·Q' into mb_y.
        self.graph.set_input["sp", Self.BATCH](state.mb_sp, ctx)
        self.graph.forward[Self.BATCH, target, POLICY](
            state.mb_y, ctx, actor_t, critic_t
        )
        # mb_y now holds the bootstrap γ·Q'; fold in reward + terminal mask.
        apply_terminal_mask[target, Self.BATCH](
            ctx,
            state.mb_r.tt[target, Self.BATCH](),
            state.mb_d.tt[target, Self.BATCH](),
            state.mb_y.tt[target, Self.BATCH, 1](),
        )
