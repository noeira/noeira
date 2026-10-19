"""RoPE[SEQ, N_HEADS, HEAD_DIM, THETA] — rotary position embedding (storage).

Rotates consecutive channel PAIRS of every attention head by an angle that
grows with the token's position, so a dot product between two rotated vectors
depends on their RELATIVE offset. Applied to q and k, never to v.

    theta_j = THETA^(-2j/HEAD_DIM)        j in [0, HEAD_DIM/2)
    a       = pos * theta_j
    y[j]      = x[j]*cos(a) - x[j+H]*sin(a)      H = HEAD_DIM/2
    y[j+H]    = x[j]*sin(a) + x[j+H]*cos(a)

## ⚠ Which two channels are a pair

The pairing is a FILE FORMAT convention and the classic way a RoPE port loads
successfully and computes gibberish. Two live conventions:

    safetensors / HF  x = [re_0 … re_{H-1}, im_0 … im_{H-1}]   pair (j, j+H)
    GGUF              x = [re_0, im_0, re_1, im_1, …]          pair (2j, 2j+1)

**This leaf implements the SPLIT-HALVES (safetensors) form**, which is what
HuggingFace checkpoints — SmolLM2 included — store. It is the same rule MAX's
`nn/rope.mojo:get_safetensors_idx` encodes as `(i//2, i//2 + head_size//2)`,
verified against that kernel on-device before this was written.

Getting it wrong is shape-legal and silent: the model still runs, still produces
finite numbers, and is simply a different function.

## Backward

A rotation is orthogonal, so the VJP is the transpose — the same rotation by
`-a`, needing no cache beyond the tables:

    gx[j]     =  go[j]*cos(a) + go[j+H]*sin(a)
    gx[j+H]   = -go[j]*sin(a) + go[j+H]*cos(a)

## Layout

Input and output are `[BATCH, SEQ * N_HEADS * HEAD_DIM]`, token-major with the
heads packed inside each token — the layout a `Linear[DIM, N_HEADS*HEAD_DIM]`
already produces. Param-free; the cos/sin tables are constants (`Tensor`, not
`Param`), built once in `make` and uploaded with the model.

`POS_OFFSET` shifts every position, for the decode case where the current token
sits after a cached prefix.
"""

from std.math import cos, sin, exp, log
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


def build_rope_tables[
    SEQ: Int, HEAD_DIM: Int, THETA: Float64, POS_OFFSET: Int
]() -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    """`cos`/`sin` of shape `[SEQ, HEAD_DIM/2]`, row-major by position."""
    comptime H = HEAD_DIM // 2
    var cs = List[Scalar[DT]]()
    var sn = List[Scalar[DT]]()
    for t in range(SEQ):
        var pos = Float64(t + POS_OFFSET)
        for j in range(H):
            # theta^(-2j/HEAD_DIM), via exp/log so THETA stays a Float64
            var inv = exp(-(2.0 * Float64(j)) / Float64(HEAD_DIM) * log(THETA))
            var a = pos * inv
            cs.append(Scalar[DT](cos(a)))
            sn.append(Scalar[DT](sin(a)))
    return (cs^, sn^)


# ── GPU kernels: one thread per (batch, token, head, pair) ──────────────
# Consecutive threads walk consecutive `j` inside one head, so both the
# `base+j` read and the `base+j` write are coalesced; the `base+H+j` partner
# is a second coalesced run one half-head away.
def _rope_fwd_kernel[
    BATCH: Int, SEQ: Int, N_HEADS: Int, HEAD_DIM: Int, N: Int
](
    x: Tile[DT, BATCH, N],
    cs: Tile[DT, SEQ * (HEAD_DIM // 2)],
    sn: Tile[DT, SEQ * (HEAD_DIM // 2)],
    dst: Tile[DT, BATCH, N],
):
    comptime H = HEAD_DIM // 2
    comptime PAIRS = BATCH * SEQ * N_HEADS * H
    var idx = Int(global_idx.x)
    if idx >= PAIRS:
        return
    var j = idx % H
    var r1 = idx // H
    var h = r1 % N_HEADS
    var r2 = r1 // N_HEADS
    var t = r2 % SEQ
    var b = r2 // SEQ
    var base = t * (N_HEADS * HEAD_DIM) + h * HEAD_DIM
    var c = rebind[Scalar[DT]](cs.ptr[unsafe_offset = t * H + j])
    var s = rebind[Scalar[DT]](sn.ptr[unsafe_offset = t * H + j])
    var i0 = b * N + base + j
    var i1 = i0 + H
    var x0 = rebind[Scalar[DT]](x.ptr[unsafe_offset=i0])
    var x1 = rebind[Scalar[DT]](x.ptr[unsafe_offset=i1])
    dst.ptr[unsafe_offset=i0] = x0 * c - x1 * s
    dst.ptr[unsafe_offset=i1] = x0 * s + x1 * c


def _rope_bwd_kernel[
    BATCH: Int, SEQ: Int, N_HEADS: Int, HEAD_DIM: Int, N: Int
](
    go: Tile[DT, BATCH, N],
    cs: Tile[DT, SEQ * (HEAD_DIM // 2)],
    sn: Tile[DT, SEQ * (HEAD_DIM // 2)],
    gi: Tile[DT, BATCH, N],
):
    comptime H = HEAD_DIM // 2
    comptime PAIRS = BATCH * SEQ * N_HEADS * H
    var idx = Int(global_idx.x)
    if idx >= PAIRS:
        return
    var j = idx % H
    var r1 = idx // H
    var h = r1 % N_HEADS
    var r2 = r1 // N_HEADS
    var t = r2 % SEQ
    var b = r2 // SEQ
    var base = t * (N_HEADS * HEAD_DIM) + h * HEAD_DIM
    var c = rebind[Scalar[DT]](cs.ptr[unsafe_offset = t * H + j])
    var s = rebind[Scalar[DT]](sn.ptr[unsafe_offset = t * H + j])
    var i0 = b * N + base + j
    var i1 = i0 + H
    var g0 = rebind[Scalar[DT]](go.ptr[unsafe_offset=i0])
    var g1 = rebind[Scalar[DT]](go.ptr[unsafe_offset=i1])
    gi.ptr[unsafe_offset=i0] = g0 * c + g1 * s
    gi.ptr[unsafe_offset=i1] = -g0 * s + g1 * c


struct RoPE[
    SEQ: Int,
    N_HEADS: Int,
    HEAD_DIM: Int,
    THETA: Float64 = 10000.0,
    POS_OFFSET: Int = 0,
](Module):
    comptime ARITY: Int = 1
    comptime H: Int = Self.HEAD_DIM // 2
    comptime WIDTH: Int = Self.N_HEADS * Self.HEAD_DIM
    comptime N: Int = Self.SEQ * Self.WIDTH
    comptime IN_DIMS = Array[Int, 1](fill=Self.N)
    comptime OUT_DIM: Int = Self.N
    comptime TABLE: Int = Self.SEQ * Self.H

    var cs: Tensor
    var sn: Tensor

    def __init__(out self):
        comptime assert Self.HEAD_DIM % 2 == 0, (
            "RoPE: HEAD_DIM must be even — it rotates channel pairs"
        )
        self.cs = Tensor()
        self.sn = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "RoPE: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        var tab = build_rope_tables[
            Self.SEQ, Self.HEAD_DIM, Self.THETA, Self.POS_OFFSET
        ]()
        m.cs = Tensor.alloc(Self.TABLE)
        m.sn = Tensor.alloc(Self.TABLE)
        for i in range(Self.TABLE):
            m.cs.data[i] = tab[0][i]
            m.sn.data[i] = tab[1][i]
        comptime if target != "cpu":
            m.cs.upload(ctx.value())
            m.sn.upload(ctx.value())
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.N)
            var xp = x.data.unsafe_ptr()
            var op = out.data.unsafe_ptr()
            var cp = self.cs.data.unsafe_ptr()
            var sp = self.sn.data.unsafe_ptr()
            for b in range(B):
                for t in range(Self.SEQ):
                    for h in range(Self.N_HEADS):
                        var base = (
                            b * Self.N + t * Self.WIDTH + h * Self.HEAD_DIM
                        )
                        for j in range(Self.H):
                            var c = cp[unsafe_offset = t * Self.H + j]
                            var s = sp[unsafe_offset = t * Self.H + j]
                            var x0 = xp[unsafe_offset = base + j]
                            var x1 = xp[unsafe_offset = base + Self.H + j]
                            op[unsafe_offset = base + j] = x0 * c - x1 * s
                            op[unsafe_offset = base + Self.H + j] = (
                                x0 * s + x1 * c
                            )
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.N)
            comptime PAIRS = B * Self.SEQ * Self.N_HEADS * Self.H
            comptime n_blocks = (PAIRS + TPB - 1) // TPB
            c.enqueue_function[
                _rope_fwd_kernel[
                    B, Self.SEQ, Self.N_HEADS, Self.HEAD_DIM, Self.N
                ]
            ](
                x.tt["gpu", B, Self.N](),
                self.cs.tt["gpu", Self.TABLE](),
                self.sn.tt["gpu", Self.TABLE](),
                out.tt["gpu", B, Self.N](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

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
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.N)
            var gp = grad_output.data.unsafe_ptr()
            var ip = gin.data.unsafe_ptr()
            var cp = self.cs.data.unsafe_ptr()
            var sp = self.sn.data.unsafe_ptr()
            for b in range(B):
                for t in range(Self.SEQ):
                    for h in range(Self.N_HEADS):
                        var base = (
                            b * Self.N + t * Self.WIDTH + h * Self.HEAD_DIM
                        )
                        for j in range(Self.H):
                            var c = cp[unsafe_offset = t * Self.H + j]
                            var s = sp[unsafe_offset = t * Self.H + j]
                            var g0 = gp[unsafe_offset = base + j]
                            var g1 = gp[unsafe_offset = base + Self.H + j]
                            ip[unsafe_offset = base + j] = g0 * c + g1 * s
                            ip[unsafe_offset = base + Self.H + j] = (
                                -g0 * s + g1 * c
                            )
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.N)
            comptime PAIRS = B * Self.SEQ * Self.N_HEADS * Self.H
            comptime n_blocks = (PAIRS + TPB - 1) // TPB
            c.enqueue_function[
                _rope_bwd_kernel[
                    B, Self.SEQ, Self.N_HEADS, Self.HEAD_DIM, Self.N
                ]
            ](
                grad_output.tt["gpu", B, Self.N](),
                self.cs.tt["gpu", Self.TABLE](),
                self.sn.tt["gpu", Self.TABLE](),
                gin.tt["gpu", B, Self.N](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        pass

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        pass
