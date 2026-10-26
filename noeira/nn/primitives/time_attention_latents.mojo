"""TimeAttentionLatents[D, N_HEADS, T, S, N_LATENTS] — causal time attention
over the latent tokens of a (T, S, D) grid. Storage-surface port of
`nn.primitives.time_attention_latents` (surface-only change; the gather/scatter
GPU kernels + the CPU gather/scatter loops are carried over VERBATIM).

Dreamer 4's block-causal transformer factorizes attention into space layers
(over S tokens per frame) and a time layer every few blocks (causal over the
T frames). The space layers run at effective batch B·T (one frame per sample,
sequence S) via plain `Sequential` leaves. The time layer needs the *other*
grouping — causal attention over T per token — which no per-sample leaf can
express on the B·T layout. This leaf bridges that: it reads the full B·T
batch, regroups the **latent** tokens to (B·L, T) and runs a causal
`MultiHeadAttention` over T, then scatters back.

I/O (per the B·T layout): IN_DIM == OUT_DIM == S·D. Sample `bt = b*T + t`,
token `s`, channel `d` lives at `((b*T+t)*S + s)*D + d`. Only the first
`N_LATENTS` tokens are time-attended; non-latent outputs are **0** so the
enclosing `Residual` leaves them unchanged.

Internally wraps `MultiHeadAttention[D, N_HEADS, T, causal=True]` driven at
batch B·L. Params (qkv/out projections) are owned by that inner module;
`for_each_param` / `for_each_state` / `zero_grad` / `polyak_from` delegate to it
(it's a Module-typed child, not an `IsParam` field, so the reflection default
can't see it — we override, mirroring `Tokenwise`).

Unlike legacy, the gather/scatter scratch is SEPARATE owned `Tensor` fields
(one buffer per slab, `ensure`/`ensure_gpu`) instead of the single `mptr`-sliced
device `Cache` — no `mptr`, no `Cache`.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT, TPB
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP
from ..models.transformer import MultiHeadAttention


# Gather full (B·T, S·D) latent tokens → packed (B·L, T·D); same index map is
# reused for the forward input and the backward grad_output. Carried VERBATIM.
def _tal_gather_kernel[
    B: Int, T: Int, S: Int, L: Int, D: Int
](
    src: Tile[DT, B * T * S * D],
    dst: Tile[DT, B * L * T * D],
):
    var idx = Int(global_idx.x)
    if idx >= B * L * T * D:
        return
    var d = idx % D
    var rem = idx // D
    var t = rem % T
    var rem2 = rem // T
    var l = rem2 % L
    var b = rem2 // L
    dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](src.ptr[unsafe_offset=((b * T + t) * S + l) * D + d])


# Scatter packed (B·L, T·D) → full (B·T, S·D), zeroing non-latent (s≥L)
# positions. Reused for the forward output and the backward grad_input.
# Carried VERBATIM.
def _tal_scatter_kernel[
    B: Int, T: Int, S: Int, L: Int, D: Int
](
    packed: Tile[DT, B * L * T * D],
    dst: Tile[DT, B * T * S * D],
):
    var idx = Int(global_idx.x)
    if idx >= B * T * S * D:
        return
    var d = idx % D
    var rem = idx // D
    var s = rem % S
    var rem2 = rem // S
    var t = rem2 % T
    var b = rem2 // T
    if s < L:
        dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
            packed.ptr[unsafe_offset=(b * L + s) * T * D + t * D + d]
        )
    else:
        dst.ptr[unsafe_offset=idx] = Scalar[DT](0.0)


struct TimeAttentionLatents[
    D: Int, N_HEADS: Int, T: Int, S: Int, N_LATENTS: Int
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.S * Self.D)
    comptime OUT_DIM = Self.S * Self.D
    comptime MHA = MultiHeadAttention[Self.D, Self.N_HEADS, Self.T, True]

    var mha: Self.MHA
    # Separate owned scratch slabs (one buffer per slab, vs the legacy single
    # `mptr`-sliced device Cache). Lazily sized via ensure / ensure_gpu.
    var packed_in: Tensor    # forward: gather → MHA input  [B*L, T*D]
    var packed_out: Tensor   # forward: MHA output → scatter
    var grad_pout: Tensor    # backward: gather grad_output → MHA grad_output
    var grad_pin: Tensor     # backward: MHA grad_input → scatter

    def __init__(out self):
        self.mha = Self.MHA()
        self.packed_in = Tensor()
        self.packed_out = Tensor()
        self.grad_pout = Tensor()
        self.grad_pin = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TimeAttentionLatents: target must be 'cpu' or 'gpu'"
        )
        comptime if target != "cpu":
            if not ctx:
                raise Error("TimeAttentionLatents.make[gpu]: ctx required")
        var m = Self()
        m.mha = Self.MHA.make[target=target, INIT=INIT](ctx)
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime assert B % Self.T == 0, (
            "TimeAttentionLatents: B (=Bn*T) must be divisible by T"
        )
        comptime Bn = B // Self.T
        comptime BL = Bn * Self.N_LATENTS
        comptime TD = Self.T * Self.D
        comptime PACKED = BL * TD
        comptime FULL = B * Self.S * Self.D
        ref in0 = inputs[0]

        comptime if target == "cpu":
            self.packed_in.ensure(PACKED)
            self.packed_out.ensure(PACKED)
            out.ensure(FULL)
            ref ip = in0.data
            ref pin = self.packed_in.data
            # gather latents: packed_in[b,l,t,d] = input[b,t,s=l,d]
            for b in range(Bn):
                for l in range(Self.N_LATENTS):
                    for t in range(Self.T):
                        for d in range(Self.D):
                            pin[
                                (b * Self.N_LATENTS + l) * TD + t * Self.D + d
                            ] = ip[
                                ((b * Self.T + t) * Self.S + l) * Self.D + d
                            ]
            self.mha.forward[target, BL, POLICY=POLICY](
                TensorRefs[1](self.packed_in), self.packed_out, ctx
            )
            ref pout = self.packed_out.data
            ref op = out.data
            # scatter: out[b,t,s,d] = packed_out[b,s,t,d] if s<L else 0
            for b in range(Bn):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            var v = Scalar[DT](0.0)
                            if s < Self.N_LATENTS:
                                v = pout[
                                    (b * Self.N_LATENTS + s) * TD
                                    + t * Self.D + d
                                ]
                            op[
                                ((b * Self.T + t) * Self.S + s) * Self.D + d
                            ] = v
        else:
            var c = ctx.value()
            self.packed_in.ensure_gpu(c, PACKED)
            self.packed_out.ensure_gpu(c, PACKED)
            out.ensure_gpu(c, FULL)
            comptime lay_full = RowMajor[FULL]
            comptime lay_pack = RowMajor[PACKED]
            comptime gk = _tal_gather_kernel[
                Bn, Self.T, Self.S, Self.N_LATENTS, Self.D
            ]
            c.enqueue_function[gk](
                in0.tt["gpu", FULL](),
                self.packed_in.tt["gpu", PACKED](),
                grid_dim=(PACKED + TPB - 1) // TPB, block_dim=TPB,
            )
            self.mha.forward[target, BL, POLICY=POLICY](
                TensorRefs[1](self.packed_in), self.packed_out, ctx
            )
            comptime sk = _tal_scatter_kernel[
                Bn, Self.T, Self.S, Self.N_LATENTS, Self.D
            ]
            c.enqueue_function[sk](
                self.packed_out.tt["gpu", PACKED](),
                out.tt["gpu", FULL](),
                grid_dim=(FULL + TPB - 1) // TPB, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime Bn = B // Self.T
        comptime BL = Bn * Self.N_LATENTS
        comptime TD = Self.T * Self.D
        comptime PACKED = BL * TD
        comptime FULL = B * Self.S * Self.D
        ref gin = grad_inputs[0]

        comptime if target == "cpu":
            self.grad_pout.ensure(PACKED)
            self.grad_pin.ensure(PACKED)
            gin.ensure(FULL)
            ref gop = grad_output.data
            ref gpout = self.grad_pout.data
            # gather grad_output latents (non-latent outputs were 0 → no grad)
            for b in range(Bn):
                for l in range(Self.N_LATENTS):
                    for t in range(Self.T):
                        for d in range(Self.D):
                            gpout[
                                (b * Self.N_LATENTS + l) * TD + t * Self.D + d
                            ] = gop[
                                ((b * Self.T + t) * Self.S + l) * Self.D + d
                            ]
            self.mha.vjp[target, BL, POLICY=POLICY](
                TensorRefs[1](self.packed_in),
                self.grad_pout,
                TensorRefs[1](self.grad_pin),
                ctx,
            )
            ref gpin = self.grad_pin.data
            ref gip = gin.data
            # scatter grad to latent input positions; non-latents get 0
            for b in range(Bn):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            var v = Scalar[DT](0.0)
                            if s < Self.N_LATENTS:
                                v = gpin[
                                    (b * Self.N_LATENTS + s) * TD
                                    + t * Self.D + d
                                ]
                            gip[
                                ((b * Self.T + t) * Self.S + s) * Self.D + d
                            ] = v
        else:
            var c = ctx.value()
            self.grad_pout.ensure_gpu(c, PACKED)
            self.grad_pin.ensure_gpu(c, PACKED)
            gin.ensure_gpu(c, FULL)
            comptime lay_full = RowMajor[FULL]
            comptime lay_pack = RowMajor[PACKED]
            comptime gk = _tal_gather_kernel[
                Bn, Self.T, Self.S, Self.N_LATENTS, Self.D
            ]
            c.enqueue_function[gk](
                grad_output.tt["gpu", FULL](),
                self.grad_pout.tt["gpu", PACKED](),
                grid_dim=(PACKED + TPB - 1) // TPB, block_dim=TPB,
            )
            self.mha.vjp[target, BL, POLICY=POLICY](
                TensorRefs[1](self.packed_in),
                self.grad_pout,
                TensorRefs[1](self.grad_pin),
                ctx,
            )
            comptime sk = _tal_scatter_kernel[
                Bn, Self.T, Self.S, Self.N_LATENTS, Self.D
            ]
            c.enqueue_function[sk](
                self.grad_pin.tt["gpu", PACKED](),
                gin.tt["gpu", FULL](),
                grid_dim=(FULL + TPB - 1) // TPB, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.mha.for_each_param[target](
            visitor, ctx, join_name(prefix, String(0))
        )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.mha.for_each_state[target](
            visitor, ctx, join_name(prefix, String(0))
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.mha.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.mha.polyak_from[target](src.mha, tau, ctx)
