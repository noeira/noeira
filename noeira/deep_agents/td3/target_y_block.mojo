"""TD3TargetYBlock — twin-critic target-y with clipped noise (storage ComputeGraph).

STORAGE migration. Sibling of the storage `DDPGTargetYBlock` (single critic, no
smoothing) and the storage SAC `TargetYBlock` (twin-critic min + α·logp). TD3's
target value adds target-policy smoothing (clipped Gaussian noise on the target
action) and takes the min over the two TARGET critics.

Graph topology (computes only the BOOTSTRAP `γ·min(Q1',Q2')`; the reward add and
terminal mask are applied in `step` via `apply_terminal_mask`):

    InputSlot     ["sp",          OBS]                          # s'
    InputSlot     ["noise",       ACT]                          # σ-scaled, host/device-side
    ExternalNode  ["a_sp",        ACTOR,            "sp"]       # → μ_t(s')  [B,ACT]
    Node          ["noise_clip",  Clamp[ACT],       "noise"]    # clamp(ε, ±c·scale)
    Node          ["a_plus_n",    BinaryElemAdd[ACT],"a_sp","noise_clip"]
    Node          ["a_smoothed",  Clamp[ACT],       "a_plus_n"] # clamp(., ±scale)
    Node          ["sa",          Concat2[OBS,ACT], "sp","a_smoothed"]
    ExternalNode  ["q1",          CRITIC,           "sa"]       # → [B,1]
    ExternalNode  ["q2",          CRITIC,           "sa"]       # → [B,1]
    Node          ["min_q",       BinaryElemMin[1], "q1","q2"]
    Node          ["gamma_q",     Scale[1],         "min_q"]    # γ·min (output)

`step` then writes `y[b] = r[b] + (1 − term[b])·gamma_q[b]`.

TD3 target-policy smoothing (Fujimoto et al., 2018):
    a'    = clamp(μ_t(s') + clamp(ε, -c·scale, c·scale), -scale, scale)
            with ε ~ N(0, (σ·scale)^2)
    y[b]  = r[b] + (1 − term[b])·γ·min(Q1_t(s',a'), Q2_t(s',a'))

The two critics are `ExternalNode`s (forward-only here — target-y is a target,
not a loss, so no gradient flows through them on this path; the graph's vjp is
never called for target-y). CPU samples the smoothing noise via `box_muller_normal`
(host list); GPU samples on-device via Philox box-muller + a σ-scale kernel.

The TD bootstrap is masked per-sample by the natural-termination flag (`term`):
dropped on termination, kept on time-limit truncation. For truncation-only envs
(`term ≡ 0`) this is exactly `r + γ·min(Q1,Q2)`.

Surface:
    TD3TargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]
        make[target](action_scale, gamma, noise_std, noise_clip, ctx) -> Self
        step[target, POLICY](mut state, mut actor_t, mut critic1_t, mut critic2_t)
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Zero
from noeira.nn.primitives.clamp import Clamp
from noeira.nn.primitives.scale import Scale
from noeira.nn.primitives.concat import Concat2
from noeira.nn.primitives.binary_elementwise import (
    BinaryElemMin, BinaryElemAdd,
)
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import InputSlot, Node, ExternalNode
from noeira.nn.random.box_muller import box_muller_normal, box_muller_normal_gpu
from ..loss.loss_block import LossBlock
from ..training.terminal_mask import apply_terminal_mask
from ..training.trainer_block import TrainerState
from noeira.core.tiles import Tile


def _scale_inplace_kernel[N: Int](
    buf: Tile[DT, N],
    s: Scalar[DT],
):
    """`buf[idx] *= s` — σ-scale the device standard-normal noise buffer
    (TD3 target-policy smoothing). One thread per element."""
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = rebind[Scalar[DT]](buf[idx]) * s


struct TD3TargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT

    # Graph computes only the BOOTSTRAP `gamma_q = γ·min(Q1',Q2')`; the reward
    # add and terminal mask `y = r + (1−term)·gamma_q` happen in `step`.
    comptime Graph = ComputeGraph[
        InputSlot["sp", Self.OBS],                             # s'
        InputSlot["noise", Self.ACT],                          # σ-scaled noise
        ExternalNode["a_sp", Self.ACTOR, "sp"],                # → μ_t(s') [B,ACT]
        Node["noise_clip", Clamp[Self.ACT], "noise"],          # clamp(ε, ±c·scale)
        Node["a_plus_n", BinaryElemAdd[Self.ACT], "a_sp", "noise_clip"],
        Node["a_smoothed", Clamp[Self.ACT], "a_plus_n"],       # clamp(., ±scale)
        Node["sa", Concat2[Self.OBS, Self.ACT], "sp", "a_smoothed"],
        ExternalNode["q1", Self.CRITIC, "sa"],                 # → [B,1]
        ExternalNode["q2", Self.CRITIC, "sa"],                 # → [B,1]
        Node["min_q", BinaryElemMin[1], "q1", "q2"],
        Node["gamma_q", Scale[1], "min_q"],                    # γ·min (output)
    ]

    var graph: Self.Graph
    var noise: Tensor  # [BATCH * ACT] σ-scaled smoothing noise (owned scratch)

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var noise_std: Scalar[DT]   # σ for target-policy smoothing
    var noise_clip: Scalar[DT]  # c — noise clamped to ±c·action_scale
    # Philox state for the GPU target-smoothing noise (gpu path only).
    var _noise_rng_seed: UInt64
    var _noise_rng_offset: UInt64

    def __init__(out self):
        self.graph = Self.Graph()
        self.noise = Tensor()
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self.noise_std = Scalar[DT](0.2)
        self.noise_clip = Scalar[DT](0.5)
        self._noise_rng_seed = UInt64(0x7D3_5EED_C0DE)
        self._noise_rng_offset = UInt64(0)

    @staticmethod
    def make[
        target: StaticString
    ](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        noise_std: Scalar[DT] = Scalar[DT](0.2),
        noise_clip: Scalar[DT] = Scalar[DT](0.5),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "TD3TargetYBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACTOR.IN_DIMS[0] == Self.OBS
        ), "TD3TargetYBlock: ACTOR.IN_DIM must equal OBS"
        comptime assert (
            Self.ACTOR.OUT_DIM == Self.ACT
        ), "TD3TargetYBlock: ACTOR.OUT_DIM must equal ACT"
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "TD3TargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "TD3TargetYBlock: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        blk.graph = Self.Graph.make[target, Zero](ctx)
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk.noise_std = noise_std
        blk.noise_clip = noise_clip
        # Bake noise-clip + action-clamp + γ into the graph (constant across
        # calls).
        var clip_lim = noise_clip * action_scale
        blk.graph.set_node_attr["noise_clip", "min_val"](-clip_lim)
        blk.graph.set_node_attr["noise_clip", "max_val"](clip_lim)
        blk.graph.set_node_attr["a_smoothed", "min_val"](-action_scale)
        blk.graph.set_node_attr["a_smoothed", "max_val"](action_scale)
        blk.graph.set_node_attr["gamma_q", "multiplier"](gamma)
        comptime N = Self.BATCH * Self.ACT
        comptime if target == "cpu":
            blk.noise = Tensor.alloc(N)
        else:
            blk.noise = Tensor.alloc_gpu(ctx.value(), N)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_t: Self.ACTOR,
        mut critic1_t: Self.CRITIC,
        mut critic2_t: Self.CRITIC,
    ) raises:
        """Write `state.mb_y[b] = r[b] + (1−term[b])·γ·min(Q1,Q2)(s', smoothed a')`.
        """
        var ctx = state.ctx
        comptime N = Self.BATCH * Self.ACT
        var sigma = self.noise_std * self.action_scale

        # Sample standard-normal noise, then σ-scale in place. The graph's
        # `noise_clip` node clamps to ±(noise_clip · action_scale). CPU uses
        # std.random box-muller (host list); GPU uses Philox box-muller + a
        # σ-scale kernel (separate baseline, same math).
        comptime if target == "cpu":
            box_muller_normal(self.noise.data.unsafe_ptr(), N)
            for k in range(N):
                self.noise.data[k] = self.noise.data[k] * sigma
        else:
            var c = ctx.value()
            var noise_flat = self.noise.tt["gpu", N]()
            box_muller_normal_gpu[N](
                c, noise_flat.ptr, self._noise_rng_seed, self._noise_rng_offset,
            )
            self._noise_rng_offset += UInt64(((N + 1) // 2) * 2)
            comptime n_blocks = (N + TPB - 1) // TPB
            c.enqueue_function[_scale_inplace_kernel[N]](
                self.noise.tt["gpu", N](),
                sigma,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

        # Seed the two named inputs (a COPY into the graph pool), then forward
        # → γ·min(Q1',Q2') into mb_y (actor + 2 target critics threaded refs).
        self.graph.set_input["sp", Self.BATCH](state.mb_sp, ctx)
        self.graph.set_input["noise", Self.BATCH](self.noise, ctx)
        self.graph.forward[Self.BATCH, target, POLICY](
            state.mb_y, ctx, actor_t, critic1_t, critic2_t
        )
        # mb_y now holds the bootstrap γ·min; fold in reward + terminal mask.
        apply_terminal_mask[target, Self.BATCH](
            ctx,
            state.mb_r.tt[target, Self.BATCH](),
            state.mb_d.tt[target, Self.BATCH](),
            state.mb_y.tt[target, Self.BATCH, 1](),
        )
