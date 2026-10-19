"""Parallel[*BRANCHES] — N-ary fan-out then column-concat (storage, CPU + GPU).

Runs each branch on the SAME input and concatenates the per-branch outputs along
columns: `out = [B0(x) | B1(x) | … ]`. Backward splits grad_output into per-branch
column slices, runs each branch's vjp into a shared gi_temp, and accumulates the
per-branch grad-inputs.

Variadic from the start — the storage `TensorPack[N]` (whose `__getitem__`
MutAnyOrigin pin makes each slot a safe branch output) + comptime cumulative
offsets make the N-ary form as clean as the old hardcoded 2-branch one. The
2-branch `Parallel[A, B]` (e.g. the actor's `[mu | log_std]` heads) is just N=2.
`BranchConcat` is a parametric alias of this.

  ARITY  = 1            (one input, N branches)
  OUT_DIM = Σ BRANCHES[i].OUT_DIM

All branches share ONE activation dtype (asserted in __init__); the concat
changes OUT_DIM (Σ branch dims) but NOT the dtype, so every activation buffer
(slabs, out, grads, gi_temp) is `TensorImpl[Self.ACT_DT]` and the concat/split
kernels are parametrized by `ADT` (defaulting to DT → NoAMP unchanged).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.fill import fill_dev
from ..core.tensor_refs import TensorRefs, child_refs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


# ── grad-input accumulation (dst[i] += src[i]) ──────────────────────────
# Local ADT-parametrized twin of linear's `_accum_kernel` (which is still
# DT-hardcoded): the per-branch grad-inputs sum on the combinator's ACT_DT
# buffers, so this must run at bf16 too. `ADT` defaults to DT → NoAMP unchanged.
def _par_accum_kernel[
    N: Int, ADT: DType = DT
](
    dst: Tile[ADT, N],
    src: Tile[ADT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[ADT]](dst[i]) + rebind[Scalar[ADT]](src[i])


# ── comptime variadic helpers ──────────────────────────────────────────
def _total_out_dim[*BRANCHES: Module]() -> Int:
    var s = 0
    comptime for i in range(BRANCHES.length):
        s += BRANCHES[i].OUT_DIM
    return s


def _cumulative_offset[index: Int, *BRANCHES: Module]() -> Int:
    var s = 0
    comptime for j in range(index):
        s += BRANCHES[j].OUT_DIM
    return s


# ── per-branch offset write/read into the packed buffer ─────────────────
# `ADT` is the combinator's activation dtype (DT by default → NoAMP unchanged);
# the concat moves columns, never recasts, so all operands share it.
def _par_write_kernel[
    B: Int, OI: Int, OD: Int, ADT: DType = DT
](
    slab: Tile[ADT, B, OI],
    packed: Tile[ADT, B, OD],
    off_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var off = Int(off_arg)
    var idx = Int(global_idx.x)
    if idx < B * OI:
        var bi = idx // OI
        var ji = idx % OI
        packed[bi, off + ji] = rebind[Scalar[ADT]](slab[bi, ji])


def _par_read_kernel[
    B: Int, OI: Int, OD: Int, ADT: DType = DT
](
    packed: Tile[ADT, B, OD],
    slab: Tile[ADT, B, OI],
    off_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var off = Int(off_arg)
    var idx = Int(global_idx.x)
    if idx < B * OI:
        var bi = idx // OI
        var ji = idx % OI
        slab[bi, ji] = rebind[Scalar[ADT]](packed[bi, off + ji])


# ── 2-branch concat/split kernels — kept for SkipConcat's input|inner merge.
def _par_concat_kernel[
    B: Int, OA: Int, OB: Int, ADT: DType = DT
](
    a: Tile[ADT, B, OA],
    bb: Tile[ADT, B, OB],
    packed: Tile[ADT, B, OA + OB],
):
    var idx = Int(global_idx.x)
    var total = B * (OA + OB)
    if idx < total:
        var bi = idx // (OA + OB)
        var ji = idx % (OA + OB)
        if ji < OA:
            packed[bi, ji] = rebind[Scalar[ADT]](a[bi, ji])
        else:
            packed[bi, ji] = rebind[Scalar[ADT]](bb[bi, ji - OA])


def _par_split_kernel[
    B: Int, OA: Int, OB: Int, ADT: DType = DT
](
    packed: Tile[ADT, B, OA + OB],
    a: Tile[ADT, B, OA],
    bb: Tile[ADT, B, OB],
):
    var idx = Int(global_idx.x)
    var total = B * (OA + OB)
    if idx < total:
        var bi = idx // (OA + OB)
        var ji = idx % (OA + OB)
        if ji < OA:
            a[bi, ji] = rebind[Scalar[ADT]](packed[bi, ji])
        else:
            bb[bi, ji - OA] = rebind[Scalar[ADT]](packed[bi, ji])


struct Parallel[*BRANCHES: Module](Module):
    comptime ARITY = 1
    comptime N = Self.BRANCHES.length
    comptime IN = Self.BRANCHES[0].IN_DIMS[0]
    comptime IN_DIMS = Array[Int, 1](fill=Self.BRANCHES[0].IN_DIMS[0])
    comptime OUT_DIM = _total_out_dim[*Self.BRANCHES]()
    # All branches share one activation dtype (asserted in __init__); concat
    # changes OUT_DIM but not the dtype, so this is the slab/out/grad dtype.
    comptime ACT_DT = Self.BRANCHES[0].ACT_DT

    var branches: Tuple[*Self.BRANCHES]
    var slabs: TensorPack[Self.N, Self.ACT_DT]
    var gi_temp: TensorImpl[Self.ACT_DT]

    def __init__(out self):
        comptime assert Self.N >= 1, "Parallel requires >= 1 branch"
        comptime for i in range(Self.N):
            comptime assert (
                Self.BRANCHES[i].IN_DIMS[0] == Self.BRANCHES[0].IN_DIMS[0]
            ), "Parallel: all BRANCHES must share IN_DIM"
        comptime for i in range(Self.N):
            comptime assert (
                Self.BRANCHES[i].ACT_DT == Self.ACT_DT
            ), "Parallel: all BRANCHES must share ACT_DT (one activation dtype)"
        self.branches = Tuple[*Self.BRANCHES]()
        self.slabs = TensorPack[Self.N, Self.ACT_DT]()
        self.gi_temp = TensorImpl[Self.ACT_DT]()


    def __init__[
        target: StaticString, INIT: Initializer
    ](out self, *, ctx: Optional[DeviceContext]) raises:
        """Build the children IN PLACE, straight from their `make` — same
        idiom and same reason as `Sequential.__init__[target, INIT]`
        (docs/COMPILE_TIME_PROFILING.md §3.2).
        """
        comptime assert Self.N >= 1, "Parallel requires >= 1 branch"
        comptime for i in range(Self.N):
            comptime assert (
                Self.BRANCHES[i].IN_DIMS[0] == Self.BRANCHES[0].IN_DIMS[0]
            ), "Parallel: all BRANCHES must share IN_DIM"
        comptime for i in range(Self.N):
            comptime assert (
                Self.BRANCHES[i].ACT_DT == Self.ACT_DT
            ), "Parallel: all BRANCHES must share ACT_DT (one activation dtype)"
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.branches)
        )
        comptime for i in range(Self.N):
            Pointer(to=self.branches[i]).unsafe_write(
                Self.BRANCHES[i].make[target, INIT](ctx)
            )
        self.slabs = TensorPack[Self.N, Self.ACT_DT]()
        self.gi_temp = TensorImpl[Self.ACT_DT]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self.__init__[target, INIT](ctx=ctx)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Buffers are typed at Self.ACT_DT; each branch's forward wants the
        # BRANCH's (ARITY, ACT_DT) (== Self.ACT_DT, asserted, but distinct to the
        # checker). Bridge: `child_refs[cn, ci]` for the input, `rebind` for the
        # mut slab. The concat (slabs → out) runs entirely at Self.ACT_DT.
        ref in0 = inputs[0]
        comptime for i in range(Self.N):
            comptime ci = Self.BRANCHES[i].ACT_DT
            comptime cn = Self.BRANCHES[i].ARITY
            self.branches[i].forward[target, B, POLICY=POLICY](
                child_refs[cn, ci](in0),
                rebind[TensorImpl[ci]](self.slabs[i]),
                ctx,
            )
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            comptime for i in range(Self.N):
                comptime off = _cumulative_offset[i, *Self.BRANCHES]()
                comptime oi = Self.BRANCHES[i].OUT_DIM
                for b in range(B):
                    for j in range(oi):
                        out.data[b * Self.OUT_DIM + off + j] = self.slabs[
                            i
                        ].data[b * oi + j]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            comptime for i in range(Self.N):
                comptime off = _cumulative_offset[i, *Self.BRANCHES]()
                comptime oi = Self.BRANCHES[i].OUT_DIM
                c.enqueue_function[
                    _par_write_kernel[B, oi, Self.OUT_DIM, Self.ACT_DT]
                ](
                    self.slabs[i].tt["gpu", B, oi](),
                    out.tt["gpu", B, Self.OUT_DIM](),
                    Int64(off),
                    grid_dim=(B * oi + TPB - 1) // TPB,
                    block_dim=TPB,
                )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime NIN = B * Self.IN
        comptime if target == "cpu":
            gin.ensure(NIN)
            for k in range(NIN):
                gin.data[k] = Scalar[Self.ACT_DT](0)
        else:
            var c0 = ctx.value()
            gin.ensure_gpu(c0, NIN)
            # ⚠ NOT `enqueue_fill`: on Metal that is a synchronize (`nn/core/fill.mojo`).
            fill_dev(gin.dev.value(), NIN, Scalar[Self.ACT_DT](0), c0)

        comptime for i in range(Self.N):
            comptime off = _cumulative_offset[i, *Self.BRANCHES]()
            comptime oi = Self.BRANCHES[i].OUT_DIM
            comptime ci = Self.BRANCHES[i].ACT_DT
            comptime cn = Self.BRANCHES[i].ARITY
            comptime if target == "cpu":
                for b in range(B):
                    for j in range(oi):
                        self.slabs[i].data[b * oi + j] = grad_output.data[
                            b * Self.OUT_DIM + off + j
                        ]
            else:
                var c = ctx.value()
                c.enqueue_function[
                    _par_read_kernel[B, oi, Self.OUT_DIM, Self.ACT_DT]
                ](
                    grad_output.tt["gpu", B, Self.OUT_DIM](),
                    self.slabs[i].tt["gpu", B, oi](),
                    Int64(off),
                    grid_dim=(B * oi + TPB - 1) // TPB,
                    block_dim=TPB,
                )
            # Branch vjp: gradient flows slabs[i] → gi_temp at the BRANCH's
            # (ARITY, ACT_DT) — bridge fin/slab/gi_temp like forward.
            self.branches[i].vjp[target, B, POLICY=POLICY](
                child_refs[cn, ci](fin),
                rebind[TensorImpl[ci]](self.slabs[i]),
                child_refs[cn, ci](self.gi_temp),
                ctx,
            )
            comptime if target == "cpu":
                var gp = gin.data.unsafe_ptr()
                var ap = self.gi_temp.data.unsafe_ptr()
                comptime W = CPU_SIMD_W
                var k = 0
                while k + W <= NIN:
                    gp.unsafe_store(k, gp.unsafe_load[width=W](k) + ap.unsafe_load[width=W](k))
                    k += W
                while k < NIN:
                    gp[unsafe_offset=k] = gp[unsafe_offset=k] + ap[unsafe_offset=k]
                    k += 1
            else:
                var c = ctx.value()
                c.enqueue_function[_par_accum_kernel[NIN, Self.ACT_DT]](
                    gin.tt["gpu", NIN](),
                    self.gi_temp.tt["gpu", NIN](),
                    grid_dim=(NIN + TPB - 1) // TPB,
                    block_dim=TPB,
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        comptime for i in range(Self.N):
            self.branches[i].for_each_param[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        comptime for i in range(Self.N):
            self.branches[i].for_each_state[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime for i in range(Self.N):
            self.branches[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        comptime for i in range(Self.N):
            self.branches[i].polyak_from[target](src.branches[i], tau, ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime for i in range(Self.N):
            self.branches[i].set_attr[ATTR](value)
