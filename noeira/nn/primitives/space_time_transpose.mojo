"""SpaceTimeTranspose[T, S, D] — swap time/space axes of a token grid (storage).

Transformed from legacy `nn.primitives.space_time_transpose` (surface-only
change; the CPU permutation loops and the GPU `_stt_kernel` are carried over
VERBATIM).

Dreamer 4 factorizes attention into space layers (attend over S tokens per
frame) and time layers (attend over T frames per token). Between them the
(T, S) grid of D-vectors is transposed. This leaf reinterprets each sample's
`T*S*D` slab as a row-major `(T, S, D)` tensor and writes `(S, T, D)`:

    out[b, (s*T + t)*D + d] = in[b, (t*S + s)*D + d]

IN_DIM == OUT_DIM == T*S*D; param-free, cache-free. Backward is the inverse
permutation (the same map with T↔S swapped). Conforms to `Module`.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor

comptime STT_VEC = 4   # 128-bit run-copy width (B2; used when D % STT_VEC == 0)
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import RowMajor, Tile


# ── GPU kernel (verbatim from legacy; args MutAnyOrigin = GPU ABI) ────────
def _stt_kernel[
    BATCH: Int, T: Int, S: Int, D: Int, INVERSE: Bool
](
    src: Tile[DT, BATCH, T * S * D],
    dst: Tile[DT, BATCH, T * S * D],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * T * S * D
    if gid >= total:
        return
    var b = gid // (T * S * D)
    var rem = gid % (T * S * D)
    var d = rem % D
    var pos = rem // D                # destination grid position
    comptime if INVERSE:
        # dst grid is (T, S): pos = t*S + s ; read from src (S, T) at s*T + t
        var t = pos // S
        var s = pos % S
        dst[b, gid % (T * S * D)] = rebind[Scalar[DT]](
            src[b, (s * T + t) * D + d]
        )
    else:
        # dst grid is (S, T): pos = s*T + t ; read from src (T, S) at t*S + s
        var s = pos // T
        var t = pos % T
        dst[b, gid % (T * S * D)] = rebind[Scalar[DT]](
            src[b, (t * S + s) * D + d]
        )


# ── vectorized run-copy (B2): D % STT_VEC == 0. The innermost D dim is
# contiguous in both layouts, so each thread copies a STT_VEC-wide chunk of
# one d-run with a single 128-bit load/store, computing the (t,s) permutation
# once per chunk instead of per element. NVIDIA: 2.4× in the L2-resident
# regime, ~1.0× (no worse) when HBM-bound. ────────────────────────────────
def _stt_vec_kernel[
    BATCH: Int, T: Int, S: Int, D: Int, INVERSE: Bool
](
    src: Tile[DT, BATCH, T * S * D],
    dst: Tile[DT, BATCH, T * S * D],
):
    var gid = Int(global_idx.x)
    comptime TSD = T * S * D
    comptime DV = D // STT_VEC
    comptime total = BATCH * T * S * DV
    if gid >= total:
        return
    var b = gid // (T * S * DV)
    var rem = gid % (T * S * DV)
    var dv = rem % DV
    var pos = rem // DV               # destination grid position
    comptime if INVERSE:
        var t = pos // S
        var s = pos % S
        var sb = b * TSD + (s * T + t) * D + dv * STT_VEC
        var db = b * TSD + pos * D + dv * STT_VEC
        dst.ptr.unsafe_store(db, src.ptr.unsafe_load[width=STT_VEC](sb))
    else:
        var s = pos // T
        var t = pos % T
        var sb = b * TSD + (t * S + s) * D + dv * STT_VEC
        var db = b * TSD + pos * D + dv * STT_VEC
        dst.ptr.unsafe_store(db, src.ptr.unsafe_load[width=STT_VEC](sb))


struct SpaceTimeTranspose[T: Int, S: Int, D: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.T * Self.S * Self.D)
    # `Array` is not `ImplicitlyCopyable` (Mojo 1.0): indexing the comptime
    # `IN_DIMS` from a runtime context would materialize the whole array.
    comptime IN_DIM0 = Self.T * Self.S * Self.D
    comptime OUT_DIM = Self.T * Self.S * Self.D

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert target == "cpu" or target == "gpu", (
            "SpaceTimeTranspose: target must be 'cpu' or 'gpu'"
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
        comptime N = B * Self.OUT_DIM
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(N)
            ref ip = in0.data
            ref op = out.data
            for b in range(B):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            op[
                                b * Self.OUT_DIM + (s * Self.T + t) * Self.D + d
                            ] = ip[
                                b * Self.IN_DIM0 + (t * Self.S + s) * Self.D + d
                            ]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            self._run_gpu[B, False](in0, out, c)

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
        comptime N = B * Self.IN_DIM0
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            ref gop = grad_output.data
            ref gip = gin.data
            # inverse permutation: grad_in[(t,s)] = grad_out[(s,t)]
            for b in range(B):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            gip[
                                b * Self.IN_DIM0 + (t * Self.S + s) * Self.D + d
                            ] = gop[
                                b * Self.OUT_DIM + (s * Self.T + t) * Self.D + d
                            ]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            self._run_gpu[B, True](grad_output, gin, c)

    def _run_gpu[
        B: Int, INVERSE: Bool
    ](mut self, mut src: Tensor, mut dst: Tensor, c: DeviceContext) raises:
        comptime lay = RowMajor[B, Self.T * Self.S * Self.D]
        comptime total = B * Self.T * Self.S * Self.D
        comptime if Self.D % STT_VEC == 0:
            # vectorized run-copy: STT_VEC elements per thread (B2)
            comptime n_blocks = (total // STT_VEC + TPB - 1) // TPB
            comptime kernel = _stt_vec_kernel[
                B, Self.T, Self.S, Self.D, INVERSE
            ]
            c.enqueue_function[kernel](
                src.tt["gpu", B, Self.T * Self.S * Self.D](),
                dst.tt["gpu", B, Self.T * Self.S * Self.D](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )
        else:
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _stt_kernel[B, Self.T, Self.S, Self.D, INVERSE]
            c.enqueue_function[kernel](
                src.tt["gpu", B, Self.T * Self.S * Self.D](),
                dst.tt["gpu", B, Self.T * Self.S * Self.D](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults.
