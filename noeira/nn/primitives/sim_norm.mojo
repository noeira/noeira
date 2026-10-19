"""SimNorm[DIM, GROUPS] — simplicial normalisation / per-group softmax (storage).

Transformed from legacy `nn.primitives.SimNorm` (surface-only change). Param-less;
the softmax output `y` is cached in a leaf-owned `Tensor` for backward. CPU loops
+ the two GPU kernels (one thread per (batch, group); group sizes ≤32 so no in-
block reduction) are carried over verbatim. Used by TDMPC2 dynamics/encoder heads.

Math, with `G = DIM/GROUPS` per group:
    sub_g(x) = x[g·G : (g+1)·G]
    y[g·G + k] = exp(sub_g[k] - max(sub_g)) / Σ_j exp(sub_g[j] - max(sub_g))

Backward (per group, standard softmax Jacobian):
    dot_g = Σ_k grad_y[g·G+k] · y[g·G+k]
    grad_x[g·G+k] = y[g·G+k] · (grad_y[g·G+k] - dot_g)
"""

from std.math import exp
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) —
# one thread per (batch, group). The serial inner loop over GROUP_SIZE is
# the canonical TDMPC2 layout (GROUP_SIZE ≤ 32) so no in-block reduction.
# ──────────────────────────────────────────────────────────────────────


def _sim_norm_forward_kernel[
    BATCH: Int, DIM: Int, GROUPS: Int, GROUP_SIZE: Int,
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    cache: Tile[DT, BATCH, DIM],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * GROUPS:
        return
    var b = idx // GROUPS
    var g = idx % GROUPS
    var base = g * GROUP_SIZE

    # Register-cache the group: read the input once, then reuse exp(x-max) for
    # both the normaliser sum and the write (the legacy kernel read input 3×
    # and recomputed exp twice). Capped so the local array stays in registers.
    comptime if GROUP_SIZE <= 32:
        var grp = Array[Scalar[DT], GROUP_SIZE](fill=Scalar[DT](0))
        var max_val = rebind[Scalar[DT]](input[b, base])
        grp[0] = max_val

        comptime for k in range(1, GROUP_SIZE):
            var v = rebind[Scalar[DT]](input[b, base + k])
            grp[k] = v
            if v > max_val:
                max_val = v
        var sum_exp: Scalar[DT] = 0.0

        comptime for k in range(GROUP_SIZE):
            var ek = exp(grp[k] - max_val)
            grp[k] = ek
            sum_exp += ek
        var inv_sum = Scalar[DT](1.0) / sum_exp

        comptime for k in range(GROUP_SIZE):
            var y = grp[k] * inv_sum
            output[b, base + k] = y
            cache[b, base + k] = y
    else:
        var max_val = rebind[Scalar[DT]](input[b, base])
        for k in range(1, GROUP_SIZE):
            var v = rebind[Scalar[DT]](input[b, base + k])
            if v > max_val:
                max_val = v
        var sum_exp: Scalar[DT] = 0.0
        for k in range(GROUP_SIZE):
            var v = rebind[Scalar[DT]](input[b, base + k])
            sum_exp += exp(v - max_val)
        var inv_sum = Scalar[DT](1.0) / sum_exp
        for k in range(GROUP_SIZE):
            var v = rebind[Scalar[DT]](input[b, base + k])
            var y = exp(v - max_val) * inv_sum
            output[b, base + k] = y
            cache[b, base + k] = y


def _sim_norm_backward_kernel[
    BATCH: Int, DIM: Int, GROUPS: Int, GROUP_SIZE: Int,
](
    grad_output: Tile[DT, BATCH, DIM],
    cache: Tile[DT, BATCH, DIM],
    grad_input: Tile[DT, BATCH, DIM],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * GROUPS:
        return
    var b = idx // GROUPS
    var g = idx % GROUPS
    var base = g * GROUP_SIZE

    var dot: Scalar[DT] = 0.0
    for k in range(GROUP_SIZE):
        var dy = rebind[Scalar[DT]](grad_output[b, base + k])
        var y = rebind[Scalar[DT]](cache[b, base + k])
        dot += dy * y

    for k in range(GROUP_SIZE):
        var dy = rebind[Scalar[DT]](grad_output[b, base + k])
        var y = rebind[Scalar[DT]](cache[b, base + k])
        grad_input[b, base + k] = y * (dy - dot)


struct SimNorm[DIM_: Int, GROUPS_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_
    comptime GROUP_SIZE: Int = Self.DIM_ // Self.GROUPS_

    # Cache holds softmax outputs `[BATCH, DIM]` for backward.
    var cache_y: Tensor  # [BATCH, DIM]

    def __init__(out self):
        self.cache_y = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert Self.GROUPS_ > 0, "SimNorm: GROUPS must be > 0"
        comptime assert Self.DIM_ % Self.GROUPS_ == 0, (
            "SimNorm: DIM must be divisible by GROUPS"
        )
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
            out.ensure(B * Self.DIM_)
            self.cache_y.ensure(B * Self.DIM_)
            # ⚠ TWO fixes over the obvious form, both worth a lot here.
            # (1) `exp` was evaluated TWICE per element — once to build
            #     `sum_exp`, once to write the output. `exp` is the whole cost
            #     of this kernel, and SimNorm runs on every Encoder/Dynamics
            #     trunk, so that alone was ~137k redundant transcendentals per
            #     call at B=268, DIM=512.
            # (2) A group is `DIM // GROUPS` wide (8 for TD-MPC2's SN=8) and
            #     independent of every other group, so a power-of-two group is
            #     exactly one SIMD vector: load, reduce_max, one vector `exp`,
            #     reduce_add, one multiply. `simd_softmax_group` below.
            comptime GS = Self.GROUP_SIZE
            comptime VEC_OK = GS > 1 and (GS & (GS - 1)) == 0
            var ip = in0.data.unsafe_ptr()
            var op = out.data.unsafe_ptr()
            var cp = self.cache_y.data.unsafe_ptr()
            for b in range(B):
                var row = b * Self.DIM_
                for g in range(Self.GROUPS_):
                    var base = row + g * GS
                    comptime if VEC_OK:
                        var v = ip.unsafe_load[width=GS](base)
                        var e = exp(v - v.reduce_max())
                        var y = e * (Scalar[DT](1.0) / e.reduce_add())
                        op.unsafe_store(base, y)
                        cp.unsafe_store(base, y)
                    else:
                        var max_val: Scalar[DT] = ip[unsafe_offset=base]
                        for k in range(1, GS):
                            var v2 = ip[unsafe_offset=base + k]
                            if v2 > max_val:
                                max_val = v2
                        var es = Array[Scalar[DT], GS](
                            fill=Scalar[DT](0)
                        )
                        var sum_exp: Scalar[DT] = 0.0
                        for k in range(GS):
                            var ev = exp(ip[unsafe_offset=base + k] - max_val)
                            es[k] = ev
                            sum_exp += ev
                        var inv_sum = Scalar[DT](1.0) / sum_exp
                        for k in range(GS):
                            var y2 = es[k] * inv_sum
                            op[unsafe_offset=base + k] = y2
                            cp[unsafe_offset=base + k] = y2
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            self.cache_y.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime total = B * Self.GROUPS_
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _sim_norm_forward_kernel[
                    B, Self.DIM_, Self.GROUPS_, Self.GROUP_SIZE
                ]
            ](
                in0.tt["gpu", B, Self.DIM_](),
                out.tt["gpu", B, Self.DIM_](),
                self.cache_y.tt["gpu", B, Self.DIM_](),
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
            gin.ensure(B * Self.DIM_)
            var cache_t = TileTensor(
                self.cache_y.data, row_major[B, Self.DIM_]()
            )
            var go_t = TileTensor(grad_output.data, row_major[B, Self.DIM_]())
            var gi_t = TileTensor(gin.data, row_major[B, Self.DIM_]())
            for b in range(B):
                for g in range(Self.GROUPS_):
                    var base = g * Self.GROUP_SIZE
                    var dot: Scalar[DT] = 0.0
                    for k in range(Self.GROUP_SIZE):
                        dot += go_t[b, base + k] * cache_t[b, base + k]
                    for k in range(Self.GROUP_SIZE):
                        var y = cache_t[b, base + k]
                        gi_t[b, base + k] = y * (go_t[b, base + k] - dot)
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime total = B * Self.GROUPS_
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _sim_norm_backward_kernel[
                    B, Self.DIM_, Self.GROUPS_, Self.GROUP_SIZE
                ]
            ](
                grad_output.tt["gpu", B, Self.DIM_](),
                self.cache_y.tt["gpu", B, Self.DIM_](),
                gin.tt["gpu", B, Self.DIM_](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (param-less leaf → no-op). No polyak_from (no Params).
