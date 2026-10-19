"""PCModule — PCN composite on nn's storage layer (Phase A spike).

Re-architecture of the PCN composite onto nn's Tensor core. The weight
slab — previously a caller-owned raw `params` buffer juggled by
`PCTrainer` — now lives in a single owned `Param`, which makes the network
walkable by nn's `Adam`/`AdamW` (optimizer) and, later, the v2 checkpoint
envelope. **PCN keeps its own settling loop** (the local error-minimization
that IS the method); only where the weights live changes.

The struct is a *minimal* `Module` purely so it satisfies the `M: Module`
bound on `Adam.make` / `Adam.step`:
  - `forward` / `vjp` raise — PCN does not do backprop; learning runs
    through `pc_module_train_one_batch` (settling + `weight_grad`).
  - `for_each_param` / `zero_grad` / `for_each_state` are INHERITED from
    the trait defaults (reflection over `IsParam` fields), which discover
    the `weights` `Param` automatically.

Storage mapping (this phase): weights → `Param`. Working buffers (latents,
μ/ε, a_below, z_below, dx) stay caller-allocated in the trainer fn — moving
them onto the holder as `Scratch`/`Cache` is Phase B's storage propagation.

Sign convention: `PCBlock.weight_grad` already bakes the −sign so `grads`
holds the standard +∂E/∂W (descent gradient) consumed directly by
`params -= lr·grad`. That is exactly Adam's convention, so the grad buffer
feeds Adam with **no negation** — confirmed in `pc_block.mojo` (the "−sign
expected by Optimizer.step").
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.param import Param
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP

from .predictive_model import PCBlockTrait
from .pc_sequential import PCSequential
from .pc_initializer import PCInitializer, PCXavier
from noeira.core.tiles import tile


struct PCModule[*BLOCKS: PCBlockTrait](Module):
    """Stateful PCN composite — owns the weight slab as one `Param`.

    Compose exactly like `PCSequential`:
        comptime Net = PCModule[
            PCBlock[4, 8, PCIdentity],
            PCBlock[8, 2, PCIdentity],
        ]
        var net = Net.make_pcn[PCXavier]()
        var opt = Adam(lr=Scalar[DT](1e-2))   # nn.storage Adam/AdamW
    """

    comptime NET = PCSequential[*Self.BLOCKS]

    # ── Module trait surface ─────────────────────────────────────────────
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.NET.IN_DIM)
    comptime OUT_DIM: Int = Self.NET.OUT_DIM

    # The whole concatenated parameter slab (per-block W|b, same layout as
    # the legacy `params` buffer). `APPLY_DECAY=False`: PC weight decay, if
    # any, is handled in the PC energy, not via the optimizer.
    var weights: Param["pc_params", False, Self.NET.PARAM_SIZE]

    def __init__(out self):
        self.weights = Param["pc_params", False, Self.NET.PARAM_SIZE]()

    # ── Real PCN constructor (CPU) ───────────────────────────────────────

    @staticmethod
    def make_pcn[INIT: PCInitializer = PCXavier]() raises -> Self:
        """Allocate CPU weight storage and run per-block PCN init via the
        composite's `init_params_pc` (each block type's own layout: linear /
        conv W via INIT, norm γ=1, etc.). Legacy-`nn`-free. The
        nn-`Initializer`-based `make` below is unusable for PCN (different
        init contract), so this is the constructor real code calls."""
        var net = Self()
        net.weights = Param["pc_params", False, Self.NET.PARAM_SIZE].make[
            "cpu"
        ]()

        var params = tile[Self.NET.PARAM_SIZE](net.weights.val.data)
        Self.NET.pc_init_params[INIT, DT](params)
        return net^

    @staticmethod
    def make_pcn_gpu[
        INIT: PCInitializer = PCXavier
    ](ctx: DeviceContext) raises -> Self:
        """GPU counterpart of `make_pcn`: allocate device weight storage,
        run the (host) per-block init, and upload. Weights live in
        `weights.val.dev`; nn `Adam.make['gpu']` and `compute_grads_only_gpu`
        read them on-device."""
        var net = Self()
        net.weights = Param["pc_params", False, Self.NET.PARAM_SIZE].make[
            "gpu"
        ](ctx)
        # storage `Param.make["gpu"]` allocates grd.dev but leaves val on the
        # host (a leaf's INIT.upload normally fills val.dev). PCN inits per
        # block on the host, so allocate val.dev here and upload into it below.
        net.weights.val.ensure_gpu(ctx, Self.NET.PARAM_SIZE)
        # Init on host, then upload to the device value buffer.
        var host = List[Scalar[DT]](
            length=Self.NET.PARAM_SIZE, fill=Scalar[DT](0)
        )
        var host_view = tile[Self.NET.PARAM_SIZE](host)
        Self.NET.pc_init_params[INIT, DT](host_view)
        ctx.enqueue_copy(net.weights.val.dev.value(), host)
        ctx.synchronize()
        _ = host^
        return net^

    # ── Module factory — required by the trait; unusable for PCN ─────────

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        raise Error(
            "PCModule.make: PCN initializes per-block (fan_in/fan_out), not"
            " via the nn Initializer contract. Use make_pcn[PCInitializer]()."
        )

    # ── forward / vjp — PCN does not backprop; these only satisfy Module ─

    def forward[
        target: StaticString,
        BATCH: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "PCModule.forward: PCN uses the settling loop"
            " (pc_module_train_one_batch / forward_eval), not Module.forward."
        )

    def vjp[
        target: StaticString,
        BATCH: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "PCModule.vjp: PCN uses local error-minimization (weight_grad),"
            " not vector-Jacobian backprop."
        )

    @staticmethod
    def display_label() -> String:
        return String("PCModule")
