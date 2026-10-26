"""RepeatConditional[N, Inner] — chain N copies of a 2-input conditional block
(storage surface).

Stacks `Inner` (ARITY=2, `forward(x, c)`, dim-preserving) N times: the main
stream `x` chains block→block while the conditioning `c` is **broadcast** to
every block (same input). On backward, grad_x chains in reverse and grad_c is
**accumulated** across all N blocks (c fans out to every layer, so its gradient
is the sum of each layer's contribution):

    x_0 = x;  x_{i+1} = Inner_i(x_i, c);  out = x_N
    grad_c = Σ_i (∂Inner_i/∂c)·grad_{x_{i+1}}

The LeWM AR-predictor stack: `RepeatConditional[DEPTH,
ConditionalTransformerBlock[...]]`. Each block owns its own params
(shared=False, like `Repeat`).

Storage wiring (§B0): a block's two inputs must share ONE origin, so they are
read from a single owning pool — `pool` holds the x-copy (slot 0), the mid
activations (slots 1..N-1) and the broadcast c-copy (slot N), all `MutAnyOrigin`
via `TensorPack.__getitem__`. The backward grad pool `gpool` holds the per-block
grad_x slots (0..N-1) + a reused grad_c temp (slot N); each block's grad_c temp
is ACCUMULATED into the caller's grad_inputs[1] (elementwise, origin-free), and
block 0's grad_x is copied out to grad_inputs[0]. Mirrors `Repeat`'s mid-slab
reuse + `ComputeGraph`'s fan-out accumulation kernel.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.fill import fill_dev
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


def _rc_accum_kernel[
    ADT: DType, N: Int
](
    dst: Tile[ADT, N],
    src: Tile[ADT, N],
):
    """dst[i] += src[i] — grad_c fan-out accumulation on device (at the
    activation dtype `ADT`)."""
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[ADT]](dst[i]) + rebind[Scalar[ADT]](src[i])


struct RepeatConditional[N: Int, Inner: Module](Module):
    comptime ARITY = 2
    comptime DX = Self.Inner.OUT_DIM
    """Width of the residual stream `x` — dim-preserving, so also OUT_DIM."""
    comptime DC = Self.Inner.IN_DIMS[1]
    """Width of the broadcast conditioning `c`. NEED NOT equal `DX`: the DETR
    decoder stack packs `[query_pos | memory+pos | memory]` into one `c` that is
    far wider than the query stream. The original code assumed `DX == DC`
    because LeWM's AdaLN conditioning happened to match its stream width."""
    comptime IN_DIMS = _rc_in_dims[Self.DX, Self.DC]()
    comptime OUT_DIM = Self.Inner.OUT_DIM
    # Both stream `x` and conditioning `c` flow at the block's activation dtype;
    # the pools (and the seeded x/c copies) are stored here.
    comptime ACT_DT = Self.Inner.ACT_DT

    var children: List[Self.Inner]
    # Forward pool: slot 0 = x copy, slots 1..N-1 = block(0..N-2) outputs,
    # slot N = broadcast c copy. (§B0: each block reads two slots of ONE pool.)
    var pool: TensorPack[Self.N + 1, Self.ACT_DT]
    # Backward pool: slots 0..N-1 = per-block grad_x, slot N = grad_c temp.
    var gpool: TensorPack[Self.N + 1, Self.ACT_DT]

    def __init__(out self):
        comptime assert Self.N >= 1, "RepeatConditional requires N >= 1"
        comptime assert (
            Self.Inner.ARITY == 2
        ), "RepeatConditional: Inner must be ARITY=2 (forward(x, c))"
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
        ), (
            "RepeatConditional: Inner must be dim-preserving on the stream"
            " (IN0 == OUT). The conditioning IN1 is unconstrained."
        )
        self.children = List[Self.Inner]()
        self.pool = TensorPack[Self.N + 1, Self.ACT_DT]()
        self.gpool = TensorPack[Self.N + 1, Self.ACT_DT]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var r = Self()
        for _ in range(Self.N):
            r.children.append(Self.Inner.make[target, INIT](ctx))
        return r^

    @staticmethod
    def _copy_into[
        target: StaticString
    ](
        mut dst: TensorImpl[Self.ACT_DT], mut src: TensorImpl[Self.ACT_DT],
        n: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        """dst[0:n] = src[0:n] (target-aware copy into a pool slot)."""
        comptime if target == "cpu":
            dst.ensure(n)
            for q in range(n):
                dst.data[q] = src.data[q]
        else:
            var c = ctx.value()
            dst.ensure_gpu(c, n)
            # Size-exact sub-buffer copy: either side may be larger than `n`
            # (monotone ensure_gpu / reused pool slots); a whole-buffer copy
            # errors on the size mismatch. Mirrors compute_graph's fix.
            var src_sub = src.dev.value().create_sub_buffer[Self.ACT_DT](0, n)
            var dst_sub = dst.dev.value().create_sub_buffer[Self.ACT_DT](0, n)
            c.enqueue_copy(dst_sub, src_sub)

    @staticmethod
    def _accum_into[
        target: StaticString, NN: Int
    ](
        mut dst: TensorImpl[Self.ACT_DT], mut src: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """dst[0:NN] += src[0:NN] (target-aware grad_c fan-out)."""
        comptime if target == "cpu":
            for q in range(NN):
                dst.data[q] += src.data[q]
        else:
            ctx.value().enqueue_function[_rc_accum_kernel[Self.ACT_DT, NN]](
                dst.tt["gpu", NN](),
                src.tt["gpu", NN](),
                grid_dim=(NN + TPB - 1) // TPB,
                block_dim=TPB,
            )

    @staticmethod
    def _zero[
        target: StaticString
    ](
        mut dst: TensorImpl[Self.ACT_DT], n: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "cpu":
            dst.ensure(n)
            for q in range(n):
                dst.data[q] = Scalar[Self.ACT_DT](0)
        else:
            dst.ensure_gpu(ctx.value(), n)
            # ⚠ NOT `enqueue_fill`: on Metal that is a synchronize (`nn/core/fill.mojo`).
            fill_dev(dst.dev.value(), n, Scalar[Self.ACT_DT](0), ctx.value())

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime n = B * Self.DX
        comptime nc = B * Self.DC
        # Child-edge dtypes (== Self.ACT_DT / ARITY, asserted, but distinct to
        # the checker). The 2-ary block input pack is built inline over two
        # pool slots (both `MutAnyOrigin` via `TensorPack.__getitem__`, §B0) and
        # `rebind`-ed to the child's `(cn, ci)`; the mut output buffer goes
        # through `rebind[TensorImpl[ci]]`.
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        # Seed pool: slot 0 = x copy, slot N = broadcast c copy.
        Self._copy_into[target](self.pool[0], inputs[0], n, ctx)
        Self._copy_into[target](self.pool[Self.N], inputs[1], nc, ctx)
        comptime for i in range(Self.N):
            comptime if i == Self.N - 1:
                self.children[i].forward[target, B, POLICY=POLICY](
                    rebind[TensorRefs[cn, MutAnyOrigin, ci]](
                        TensorRefs[Self.Inner.ARITY, MutAnyOrigin, Self.ACT_DT](
                            self.pool[i], self.pool[Self.N]
                        )
                    ),
                    rebind[TensorImpl[ci]](out),
                    ctx,
                )
            else:
                self.children[i].forward[target, B, POLICY=POLICY](
                    rebind[TensorRefs[cn, MutAnyOrigin, ci]](
                        TensorRefs[Self.Inner.ARITY, MutAnyOrigin, Self.ACT_DT](
                            self.pool[i], self.pool[Self.N]
                        )
                    ),
                    rebind[TensorImpl[ci]](self.pool[i + 1]),
                    ctx,
                )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        forward_input: TensorRefs[2, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[2, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime n = B * Self.DX
        comptime nc = B * Self.DC
        # Child-edge dtypes (see forward): the 2-ary forward-input pack (pool)
        # and grad_input scatter pack (gpool) are built inline + `rebind`-ed to
        # `(cn, ci)`; mut grad buffers via `rebind[TensorImpl[ci]]`.
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        # grad_c accumulator (caller slot) starts at zero; each block adds its
        # grad_c temp (gpool[N]) into it.
        Self._zero[target](grad_inputs[1], nc, ctx)
        # Reverse-topo: block i's output-grad is grad_output (last) or block
        # i+1's already-computed grad_x (gpool[i+1]). Its grad_inputs scatter
        # into (gpool[i], gpool[N]=grad_c temp) — both from gpool (§B0).
        comptime for j in range(Self.N):
            comptime i = Self.N - 1 - j
            comptime if i == Self.N - 1:
                self.children[i].vjp[target, B, POLICY=POLICY](
                    rebind[TensorRefs[cn, MutAnyOrigin, ci]](
                        TensorRefs[Self.Inner.ARITY, MutAnyOrigin, Self.ACT_DT](
                            self.pool[i], self.pool[Self.N]
                        )
                    ),
                    rebind[TensorImpl[ci]](grad_output),
                    rebind[TensorRefs[cn, MutAnyOrigin, ci]](
                        TensorRefs[Self.Inner.ARITY, MutAnyOrigin, Self.ACT_DT](
                            self.gpool[i], self.gpool[Self.N]
                        )
                    ),
                    ctx,
                )
            else:
                self.children[i].vjp[target, B, POLICY=POLICY](
                    rebind[TensorRefs[cn, MutAnyOrigin, ci]](
                        TensorRefs[Self.Inner.ARITY, MutAnyOrigin, Self.ACT_DT](
                            self.pool[i], self.pool[Self.N]
                        )
                    ),
                    rebind[TensorImpl[ci]](self.gpool[i + 1]),
                    rebind[TensorRefs[cn, MutAnyOrigin, ci]](
                        TensorRefs[Self.Inner.ARITY, MutAnyOrigin, Self.ACT_DT](
                            self.gpool[i], self.gpool[Self.N]
                        )
                    ),
                    ctx,
                )
            Self._accum_into[target, B * Self.DC](
                grad_inputs[1], self.gpool[Self.N], ctx
            )
        # Block 0's grad_x (gpool[0]) is the gradient w.r.t. the x input.
        Self._copy_into[target](grad_inputs[0], self.gpool[0], n, ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Forward a named runtime attribute to every stacked block.

        ⚠ Without this, `Module.set_attr`'s `pass` default applied and the call
        silently did nothing — so a `Dropout` or `BatchNorm` inside a
        `RepeatConditional` could never be switched to eval, no matter what the
        caller asked for. `Repeat`, `Sequential`, `Tokenwise`, `Residual` and
        `Parallel` all had this; `RepeatConditional` was the one that did not.
        Found by the ACT DETR-encoder-stack gate, which read a 0.6 disagreement
        against a reference where the bare layer matched at 5e-7.
        """
        for i in range(Self.N):
            self.children[i].set_attr[ATTR](value)

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        for i in range(Self.N):
            self.children[i].for_each_param[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        for i in range(Self.N):
            self.children[i].for_each_state[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        for i in range(Self.N):
            self.children[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        for i in range(Self.N):
            self.children[i].polyak_from[target](src.children[i], tau, ctx)


def _rc_in_dims[DX: Int, DC: Int]() -> Array[Int, 2]:
    """`Array` has no variadic-element literal in Mojo 1.0 (`Array` is not
    `ImplicitlyCopyable`); mirrors `concat.mojo`'s comptime helpers."""
    var a = Array[Int, 2](fill=DX)
    a[1] = DC
    return a^
