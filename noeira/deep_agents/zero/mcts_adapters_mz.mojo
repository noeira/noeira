"""MuZero MCTS adapters — bridge the nn h / g / f networks into the
``planners.tree_search`` learned-dynamics model traits.

The MuZero counterpart of `mcts_adapters.mojo`'s AlphaZero ``AZPredGPU``. Where
AlphaZero learns only the prediction net, MuZero learns all three, so the GPU
search needs one adapter per net:

  * ``MZRepGPU``  — h: obs → hidden        (RepresentationGPU)
  * ``MZDynGPU``  — g: [hidden ⊕ a] → [hidden' | reward_logits]  (DynamicsGPU)
  * ``MZPredGPU`` — f: hidden → [policy_logits | value_logits]   (PredictionGPU)
  * ``MZContPredGPU`` — continuous-EZv2 f: hidden → [μ|σ|value]  (PredictionGPU)

Each holds a non-owning ``Pointer`` to a trainer-owned storage ``Module`` (origin
threaded via ``o``, the ``feedback_mojo_set_external_lifetime`` contract) and is
one ``forward`` deep, returning raw logits (the MCTS kernels own all decoding).

Storage bridge (same as ``AZPredGPU``): the planner owns the in/out device
buffers; a LayoutTensor-space copy kernel stages the input into an owned scratch
``Tensor``, the net runs entirely on owned storage, then the owned output scratch
is copied back to the planner's buffer — pointer-free (no Pointer/rebind).
The scratch Tensors are reused across calls (lazily grown).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.planners.tree_search import (
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
)
from noeira.core.tiles import Tile


def _copy2d_kernel[B: Int, D: Int](
    src: Tile[DT, B, D],
    dst: Tile[DT, B, D],
):
    """Element copy between two device LayoutTensors (planner buffer ↔ owned
    scratch). Both are real allocations, so a 2-operand copy kernel is safe."""
    var i = Int(global_idx.x)
    if i < B * D:
        dst[i // D, i % D] = rebind[Scalar[DT]](src[i // D, i % D])


@fieldwise_init
struct MZRepGPU[OBS: Int, LATENT: Int, NET: Module, o: Origin[mut=True]](
    Movable, Deinitable, RepresentationGPU
):
    """H adapter: ``obs (B, OBS) → hidden (B, LATENT)``."""

    comptime OBS_DIM: Int = Self.OBS
    comptime LATENT_DIM: Int = Self.LATENT

    var net: Pointer[Self.NET, Self.o]
    var sc_in: Tensor
    var sc_out: Tensor

    @staticmethod
    def make(ref[Self.o] net: Self.NET) -> Self:
        return Self(net=Pointer(to=net), sc_in=Tensor(), sc_out=Tensor())

    def encode_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        obs: Tile[DT, B, Self.OBS_DIM],
        hidden_out: Tile[DT, B, Self.LATENT_DIM],
    ) raises:
        comptime ID = Self.OBS_DIM
        comptime OD = Self.LATENT_DIM
        self.sc_in.ensure_gpu(ctx, B * ID)
        self.sc_out.ensure_gpu(ctx, B * OD)
        var sin = self.sc_in.tt["gpu", B, ID]()
        ctx.enqueue_function[_copy2d_kernel[B, ID]](
            obs, sin, grid_dim=(B * ID + TPB - 1) // TPB, block_dim=TPB
        )
        call_forward["gpu", B](
            self.net[],
            TensorRefs[Self.NET.ARITY](self.sc_in), self.sc_out, Optional(ctx)
        )
        var sout = self.sc_out.tt["gpu", B, OD]()
        ctx.enqueue_function[_copy2d_kernel[B, OD]](
            sout, hidden_out, grid_dim=(B * OD + TPB - 1) // TPB, block_dim=TPB
        )


@fieldwise_init
struct MZDynGPU[
    LATENT: Int, ACT: Int, BINS: Int, NET: Module, o: Origin[mut=True]
](Movable, Deinitable, DynamicsGPU):
    """G adapter: ``[hidden ⊕ onehot(a)] → [hidden' | reward_logits]``."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime DYN_IN_DIM: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT_DIM: Int = Self.LATENT + Self.BINS

    var net: Pointer[Self.NET, Self.o]
    var sc_in: Tensor
    var sc_out: Tensor

    @staticmethod
    def make(ref[Self.o] net: Self.NET) -> Self:
        return Self(net=Pointer(to=net), sc_in=Tensor(), sc_out=Tensor())

    def step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        dyn_in: Tile[DT, B, Self.DYN_IN_DIM],
        dyn_out: Tile[DT, B, Self.DYN_OUT_DIM],
    ) raises:
        comptime ID = Self.DYN_IN_DIM
        comptime OD = Self.DYN_OUT_DIM
        self.sc_in.ensure_gpu(ctx, B * ID)
        self.sc_out.ensure_gpu(ctx, B * OD)
        var sin = self.sc_in.tt["gpu", B, ID]()
        ctx.enqueue_function[_copy2d_kernel[B, ID]](
            dyn_in, sin, grid_dim=(B * ID + TPB - 1) // TPB, block_dim=TPB
        )
        call_forward["gpu", B](
            self.net[],
            TensorRefs[Self.NET.ARITY](self.sc_in), self.sc_out, Optional(ctx)
        )
        var sout = self.sc_out.tt["gpu", B, OD]()
        ctx.enqueue_function[_copy2d_kernel[B, OD]](
            sout, dyn_out, grid_dim=(B * OD + TPB - 1) // TPB, block_dim=TPB
        )


@fieldwise_init
struct MZPredGPU[
    LATENT: Int, ACT: Int, BINS: Int, NET: Module, o: Origin[mut=True]
](Movable, Deinitable, PredictionGPU):
    """F adapter: ``hidden (B, LATENT) → [policy_logits | value_logits]
    (B, ACT+BINS)``. Value is categorical (``BINS`` bins)."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime PRED_OUT_DIM: Int = Self.ACT + Self.BINS

    var net: Pointer[Self.NET, Self.o]
    var sc_in: Tensor
    var sc_out: Tensor

    @staticmethod
    def make(ref[Self.o] net: Self.NET) -> Self:
        return Self(net=Pointer(to=net), sc_in=Tensor(), sc_out=Tensor())

    def predict_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        hidden: Tile[DT, B, Self.LATENT_DIM],
        pred_out: Tile[DT, B, Self.PRED_OUT_DIM],
    ) raises:
        comptime ID = Self.LATENT_DIM
        comptime OD = Self.PRED_OUT_DIM
        self.sc_in.ensure_gpu(ctx, B * ID)
        self.sc_out.ensure_gpu(ctx, B * OD)
        var sin = self.sc_in.tt["gpu", B, ID]()
        ctx.enqueue_function[_copy2d_kernel[B, ID]](
            hidden, sin, grid_dim=(B * ID + TPB - 1) // TPB, block_dim=TPB
        )
        call_forward["gpu", B](
            self.net[],
            TensorRefs[Self.NET.ARITY](self.sc_in), self.sc_out, Optional(ctx)
        )
        var sout = self.sc_out.tt["gpu", B, OD]()
        ctx.enqueue_function[_copy2d_kernel[B, OD]](
            sout, pred_out, grid_dim=(B * OD + TPB - 1) // TPB, block_dim=TPB
        )


@fieldwise_init
struct MZContPredGPU[
    LATENT: Int, ACT_DIM: Int, BINS: Int, NET: Module, o: Origin[mut=True]
](Movable, Deinitable, PredictionGPU):
    """Continuous-EZv2 f adapter: ``hidden (B, LATENT) → [μ_raw | σ_raw | value]
    (B, 2·ACT_DIM + BINS)``."""

    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT_DIM
    comptime PRED_OUT_DIM: Int = 2 * Self.ACT_DIM + Self.BINS

    var net: Pointer[Self.NET, Self.o]
    var sc_in: Tensor
    var sc_out: Tensor

    @staticmethod
    def make(ref[Self.o] net: Self.NET) -> Self:
        return Self(net=Pointer(to=net), sc_in=Tensor(), sc_out=Tensor())

    def predict_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        hidden: Tile[DT, B, Self.LATENT_DIM],
        pred_out: Tile[DT, B, Self.PRED_OUT_DIM],
    ) raises:
        comptime ID = Self.LATENT_DIM
        comptime OD = Self.PRED_OUT_DIM
        self.sc_in.ensure_gpu(ctx, B * ID)
        self.sc_out.ensure_gpu(ctx, B * OD)
        var sin = self.sc_in.tt["gpu", B, ID]()
        ctx.enqueue_function[_copy2d_kernel[B, ID]](
            hidden, sin, grid_dim=(B * ID + TPB - 1) // TPB, block_dim=TPB
        )
        call_forward["gpu", B](
            self.net[],
            TensorRefs[Self.NET.ARITY](self.sc_in), self.sc_out, Optional(ctx)
        )
        var sout = self.sc_out.tt["gpu", B, OD]()
        ctx.enqueue_function[_copy2d_kernel[B, OD]](
            sout, pred_out, grid_dim=(B * OD + TPB - 1) // TPB, block_dim=TPB
        )
