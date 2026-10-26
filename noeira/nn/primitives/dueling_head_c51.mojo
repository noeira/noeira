"""DuelingHeadC51[NA, N_ATOMS] — Distributional Dueling DQN aggregation
(storage surface).

The Rainbow-style fusion of `DuelingHead` and `C51`. Input layout:

    [B, (1 + NA) · N_ATOMS]
        |  V (atoms)         |  A_0 (atoms)  |  A_1  | ... | A_{NA-1}  |
        |   N_ATOMS columns  |  N_ATOMS cols | ...                     |

Output: `[B, NA · N_ATOMS]`, with per-atom Wang et al. 2016 dueling
aggregation:

    Q[b, a, k] = V[b, k] + A[b, a, k] − (1/NA) · Σ_a' A[b, a', k]

Each atom k gets its own V-stream and its own mean-subtracted A-stream.
After atom-wise softmax (done by the C51 loss/target paths), the
resulting per-action distribution implicitly fuses the value baseline
with the advantage shape, which Rainbow reports as a strong gain over
plain Dueling at the same parameter count.

Backward:
    grad_in[b, V_k]     = Σ_a grad_out[b, a, k]
    grad_in[b, A_a_k]   = grad_out[b, a, k] − (1/NA) · Σ_a grad_out[b, a, k]

Use as: `Sequential[backbone..., Linear[H, (1 + NA)·N_ATOMS], DuelingHeadC51[NA, N_ATOMS]]`.
Pure architectural swap — no params, no cache. CPU + GPU.

Transformed from legacy `nn.primitives.DuelingHeadC51` (surface-only change).
The CPU aggregation loops and the two GPU kernels (combine / grad) are carried
over verbatim.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _dueling_c51_combine_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int,
](
    raw_in: Tile[DT, BATCH, (1 + NA) * N_ATOMS],
    q_out: Tile[DT, BATCH, NA * N_ATOMS],
):
    """One thread per (batch, atom). For each atom k:
        mean_A_k = (1/NA) · Σ_a A[b, a, k]
        Q[b, a, k] = V[b, k] + A[b, a, k] − mean_A_k  ∀ a
    """
    var lin = Int(global_idx.x)
    var total = BATCH * N_ATOMS
    if lin < total:
        var b = lin // N_ATOMS
        var k = lin % N_ATOMS
        var v_k = rebind[Scalar[DT]](raw_in[b, k])
        var sum_a: Scalar[DT] = 0.0
        for a in range(NA):
            sum_a = sum_a + rebind[Scalar[DT]](
                raw_in[b, N_ATOMS + a * N_ATOMS + k]
            )
        var mean_a = sum_a * (Scalar[DT](1.0) / Scalar[DT](NA))
        for a in range(NA):
            var adv = rebind[Scalar[DT]](
                raw_in[b, N_ATOMS + a * N_ATOMS + k]
            )
            q_out[b, a * N_ATOMS + k] = v_k + (adv - mean_a)


def _dueling_c51_grad_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int,
](
    grad_out: Tile[DT, BATCH, NA * N_ATOMS],
    grad_in: Tile[DT, BATCH, (1 + NA) * N_ATOMS],
):
    """One thread per (batch, atom). For each atom k:
        dV[b, k]      = Σ_a grad_out[b, a, k]
        dA[b, a, k]   = grad_out[b, a, k] − (1/NA) · dV[b, k]
    """
    var lin = Int(global_idx.x)
    var total = BATCH * N_ATOMS
    if lin < total:
        var b = lin // N_ATOMS
        var k = lin % N_ATOMS
        var sum_dq: Scalar[DT] = 0.0
        for a in range(NA):
            sum_dq = sum_dq + rebind[Scalar[DT]](
                grad_out[b, a * N_ATOMS + k]
            )
        grad_in[b, k] = sum_dq
        var inv = Scalar[DT](1.0) / Scalar[DT](NA)
        for a in range(NA):
            grad_in[b, N_ATOMS + a * N_ATOMS + k] = (
                rebind[Scalar[DT]](grad_out[b, a * N_ATOMS + k])
                - inv * sum_dq
            )


struct DuelingHeadC51[NA: Int, N_ATOMS: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=(1 + Self.NA) * Self.N_ATOMS)
    comptime OUT_DIM: Int = Self.NA * Self.N_ATOMS

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "DuelingHeadC51: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.NA > 0, "DuelingHeadC51: NA must be > 0"
        comptime assert Self.N_ATOMS > 0, "DuelingHeadC51: N_ATOMS must be > 0"
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            var input = TileTensor(
                in0.data, row_major[B, (1 + Self.NA) * Self.N_ATOMS]()
            )
            var output_v = TileTensor(
                out.data, row_major[B, Self.NA * Self.N_ATOMS]()
            )
            var inv = Scalar[DT](1.0) / Scalar[DT](Self.NA)
            for b in range(B):
                for k in range(Self.N_ATOMS):
                    var v_k = input[b, k]
                    var sum_a: Scalar[DT] = 0.0
                    for a in range(Self.NA):
                        sum_a = sum_a + input[
                            b, Self.N_ATOMS + a * Self.N_ATOMS + k
                        ]
                    var mean_a = sum_a * inv
                    for a in range(Self.NA):
                        var adv = input[
                            b, Self.N_ATOMS + a * Self.N_ATOMS + k
                        ]
                        output_v[b, a * Self.N_ATOMS + k] = v_k + (adv - mean_a)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            comptime total = B * Self.N_ATOMS
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _dueling_c51_combine_kernel[
                B, Self.NA, Self.N_ATOMS,
            ]
            c.enqueue_function[kernel](
                in0.tt["gpu", B, (1 + Self.NA) * Self.N_ATOMS](),
                out.tt["gpu", B, Self.NA * Self.N_ATOMS](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * (1 + Self.NA) * Self.N_ATOMS)
            var grad_output_v = TileTensor(
                grad_output.data, row_major[B, Self.NA * Self.N_ATOMS]()
            )
            var grad_input_v = TileTensor(
                gin.data, row_major[B, (1 + Self.NA) * Self.N_ATOMS]()
            )
            var inv = Scalar[DT](1.0) / Scalar[DT](Self.NA)
            for b in range(B):
                for k in range(Self.N_ATOMS):
                    var sum_dq: Scalar[DT] = 0.0
                    for a in range(Self.NA):
                        sum_dq = sum_dq + grad_output_v[
                            b, a * Self.N_ATOMS + k
                        ]
                    grad_input_v[b, k] = sum_dq
                    for a in range(Self.NA):
                        grad_input_v[
                            b, Self.N_ATOMS + a * Self.N_ATOMS + k
                        ] = (
                            grad_output_v[b, a * Self.N_ATOMS + k]
                            - inv * sum_dq
                        )
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * (1 + Self.NA) * Self.N_ATOMS)
            comptime total = B * Self.N_ATOMS
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _dueling_c51_grad_kernel[
                B, Self.NA, Self.N_ATOMS,
            ]
            c.enqueue_function[kernel](
                grad_output.tt["gpu", B, Self.NA * Self.N_ATOMS](),
                gin.tt["gpu", B, (1 + Self.NA) * Self.N_ATOMS](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
