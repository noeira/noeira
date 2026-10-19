"""Zero-series MCTS adapters — bridge nn ``Module`` networks into the
``planners.tree_search`` model-trait surface.

These are the zero-series analogue of ``tdmpc2/callback.mojo``: thin structs
holding a pointer to a trainer-owned nn ``Module`` and forwarding through the
planner's compile-time trait methods. Unlike the legacy
``deep_agents/alphazero/gpu_trait_adapters.mojo`` adapters (which wrapped the
old ``nn.training.Network`` and threaded params/state/workspace by hand), nn
modules are self-contained — params live inside the module — so the adapter is
one ``forward`` call deep.

Note on dtype: the legacy planner trait signatures speak
``noeira.nn.constants.dtype`` while nn forwards speak
``noeira.nn.constants.DT``. Both alias the identical ``DType.float32`` comptime
value, so the LayoutTensor↔TileTensor bridge is a plain pointer reinterpret and
the adapter — written in terms of ``DT`` — conforms to the trait's ``dtype``
surface with no conversion.

Phase A scope: ``AZPredGPU`` (AlphaZero prediction: obs → ``[policy | value]``).
The ``EnvStepGPU`` board adapter is env-agnostic and reused from the legacy
module; MuZero's representation/dynamics adapters land in Phase B.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.core.env_traits import GPUTwoPlayerDiscreteEnv
from noeira.planners.tree_search import PredictionGPU, EnvStepGPU
from noeira.core.tiles import Tile


def _copy2d_kernel[B: Int, D: Int](
    src: Tile[DT, B, D],
    dst: Tile[DT, B, D],
):
    """Element copy between two device LayoutTensors (the planner-owned buffer
    and an owned scratch). Both are real allocations, so a 2-operand copy kernel
    is safe — unlike two non-owning `view_gpu` operands. No raw pointer."""
    var i = Int(global_idx.x)
    if i < B * D:
        dst[i // D, i % D] = rebind[Scalar[DT]](src[i // D, i % D])


@fieldwise_init
struct AZPredGPU[OBS: Int, ACT: Int, NET: Module, o: Origin[mut=True]](
    Deinitable, Movable, PredictionGPU
):
    """AlphaZero prediction adapter: ``obs → policy_logits ⊕ raw_value``.

    For AlphaZero there is no learned representation, so the planner's
    ``LATENT_DIM`` *is* the obs dim — the prediction net is fed the raw board
    observation directly. ``PRED_OUT_DIM = ACT + 1`` (scalar value head;
    tanh-squashed downstream by the expand kernel's ``VALUE_SQUASH=True``).

    Holds a non-owning pointer to a trainer-owned ``NET`` (GPU-resident params).
    Construct via ``AZPredGPU[...].make(net)`` while the net is live; the caller
    must keep the net alive for the adapter's lifetime (same contract as the
    TD-MPC2 callback — see ``feedback_mojo_set_external_lifetime``).

    Storage bridge: the planner owns the ``hidden`` / ``pred_out`` device
    buffers. ``copy_from_device`` stages ``hidden`` into an owned input scratch,
    the net runs entirely on owned storage, then ``copy_to_device`` writes the
    owned output scratch back to the planner's ``pred_out``. (Two non-owning
    ``view_gpu`` operands of one kernel miscompile on Metal — so the boundary
    copies, see ``Tensor.view_gpu``'s warning. The scratch Tensors are reused
    across calls, lazily grown.)
    """

    comptime LATENT_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = Self.ACT
    comptime PRED_OUT_DIM: Int = Self.ACT + 1

    var net: Pointer[Self.NET, Self.o]
    var sc_in: Tensor   # owned input scratch (reused across predict calls)
    var sc_out: Tensor  # owned output scratch

    @staticmethod
    def make(ref[Self.o] net: Self.NET) -> Self:
        return Self(net=Pointer(to=net), sc_in=Tensor(), sc_out=Tensor())

    def predict_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        hidden: Tile[DT, B, Self.LATENT_DIM],
        pred_out: Tile[DT, B, Self.PRED_OUT_DIM],
    ) raises:
        comptime LD = Self.LATENT_DIM
        comptime PD = Self.PRED_OUT_DIM
        self.sc_in.ensure_gpu(ctx, B * LD)
        self.sc_out.ensure_gpu(ctx, B * PD)
        # Stage the planner's hidden buffer into the owned input scratch
        # (LayoutTensor-space copy — no raw pointer).
        var sin = self.sc_in.tt["gpu", B, LD]()
        ctx.enqueue_function[_copy2d_kernel[B, LD]](
            hidden, sin, grid_dim=(B * LD + TPB - 1) // TPB, block_dim=TPB
        )
        # The net runs entirely on owned storage.
        call_forward["gpu", B](
            self.net[], TensorRefs[Self.NET.ARITY](self.sc_in), self.sc_out, Optional(ctx)
        )
        # Write the owned output scratch back to the planner's pred_out buffer.
        var sout = self.sc_out.tt["gpu", B, PD]()
        ctx.enqueue_function[_copy2d_kernel[B, PD]](
            sout, pred_out, grid_dim=(B * PD + TPB - 1) // TPB, block_dim=TPB
        )


@fieldwise_init
struct AZEnvGPU[
    E: GPUTwoPlayerDiscreteEnv,
    STATE: Int,
    OBSD: Int,
    A: Int,
](EnvStepGPU, Deinitable, Movable):
    """AlphaZero env-step adapter — true game rules as MCTS leaf expansion.

    Stateless wrapper over the env's static ``step_kernel_gpu``. The planner
    (``search_gpu_alphazero``) stages each pending (parent_state, action) pair,
    calls ``step_gpu`` to play it, and feeds the post-step obs into the
    prediction net. ``board_dtype == DT == float32`` so the
    ``DeviceBuffer[DT]`` surface bridges the env's ``board_dtype`` buffers
    with no conversion. Env-agnostic — reused unchanged for Connect4 / Go.
    """

    comptime STATE_SIZE: Int = Self.STATE
    comptime OBS_DIM: Int = Self.OBSD
    comptime ACTION_DIM: Int = Self.A

    def step_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut states: DeviceBuffer[DT],
        actions: DeviceBuffer[DT],
        mut rewards_out: DeviceBuffer[DT],
        mut dones_out: DeviceBuffer[DT],
        mut terminated_out: DeviceBuffer[DT],
        mut obs_out: DeviceBuffer[DT],
        mut legal_masks_out: DeviceBuffer[DT],
        rng_seed: UInt64,
    ) raises:
        Self.E.step_kernel_gpu[B, Self.STATE_SIZE, Self.OBS_DIM](
            ctx,
            states,
            actions,
            rewards_out,
            dones_out,
            terminated_out,
            obs_out,
            legal_masks_out,
            rng_seed=rng_seed,
        )
