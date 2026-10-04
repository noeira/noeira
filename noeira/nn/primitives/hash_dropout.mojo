"""HashDropout[DIM, P] — inverted dropout with a counter-hash mask.

    on:   y = x · keep / (1 - P),   keep = hash_keep(seed, ctr, b·DIM + i)
    off:  y = x

`nn.random.hash_mask`: the mask is a function of (this instance's seed, a
counter bumped once per training forward, the element index), so the
backward REDRAWS it from the counter the forward used — no mask cache — and
CPU and GPU draw identical masks.

Why not `Dropout`: its seed is a comptime parameter, so the copies of a block
inside `Repeat` / `RepeatConditional` would all draw the SAME mask; and it is
switched by `"training"`, the attribute BatchNorm uses, so a BN in train mode
could not run beside a dropout that is off. Here the seed is drawn per
instance at construction (`std.random`) and the switch is `"dropout"` (0/1),
ON by default. (The counter lives on the host: not CUDA-graph capturable.)
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.random.hash_mask import hash_keep, new_dropout_seed
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


def _hash_dropout_kernel(
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    n: Int64,
    seed: UInt64,
    ctr: UInt64,
    p: Float32,
    scale: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var v = src[unsafe_offset=i] * scale
    if not hash_keep(seed, ctr, UInt64(i), p):
        v = Scalar[DT](0)
    dst[unsafe_offset=i] = v


def _copy_kernel(
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    n: Int64,
):
    var i = Int(global_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


struct HashDropout[DIM_: Int, P: Float64](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    var on: Bool
    var seed: UInt64
    var ctr: UInt64
    """Bumped by every forward that drew a mask."""
    var ctr_fwd: UInt64
    """The counter the last forward drew with — the vjp redraws from it."""
    var drew: Bool
    """Whether the last forward drew a mask (the vjp must match it)."""

    def __init__(out self):
        comptime assert Self.P >= 0.0 and Self.P < 1.0, "HashDropout: P in [0, 1)"
        self.on = True
        self.seed = new_dropout_seed()
        self.ctr = 0
        self.ctr_fwd = 0
        self.drew = False

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """`dropout` (0/1)."""
        comptime if ATTR == "dropout":
            self.on = value != Scalar[DT](0)

    def _apply[target: StaticString, B: Int](
        mut self, mut src: Tensor, mut dst: Tensor, ctx: Optional[DeviceContext],
        draw: Bool, ctr: UInt64,
    ) raises:
        comptime N = B * Self.DIM_
        var scale = Scalar[DT](1.0 / (1.0 - Self.P))
        comptime if target == "cpu":
            dst.ensure(N)
            for i in range(N):
                if not draw:
                    dst.data[i] = src.data[i]
                elif hash_keep(self.seed, ctr, UInt64(i), Float32(Self.P)):
                    dst.data[i] = src.data[i] * scale
                else:
                    dst.data[i] = Scalar[DT](0)
        else:
            var c = ctx.value()
            dst.ensure_gpu(c, N)
            if draw:
                c.enqueue_function[_hash_dropout_kernel](
                    src.dev.value(), dst.dev.value(), Int64(N), self.seed, ctr,
                    Float32(Self.P), scale,
                    grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
                )
            else:
                c.enqueue_function[_copy_kernel](
                    src.dev.value(), dst.dev.value(), Int64(N),
                    grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
                )

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.drew = self.on and Self.P > 0.0
        if self.drew:
            self.ctr_fwd = self.ctr
            self.ctr += 1
        ref x = inputs[0]
        self._apply[target, B](x, out, ctx, self.drew, self.ctr_fwd)

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gx = grad_inputs[0]
        self._apply[target, B](grad_output, gx, ctx, self.drew, self.ctr_fwd)
