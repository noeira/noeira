"""Dropout[DIM] — inverted-dropout regularisation (storage surface).

Transformed from legacy `nn.primitives.Dropout` (surface-only change). The
inverted-dropout math, the per-element PhiloxRandom uniform draw, and the cached
scaled-mask backward are all carried over VERBATIM. The ONLY change is the
surface (TensorRefs input pack + `Tensor` storage + storage RNG idiom).

Math (inverted dropout, identical to PyTorch):
    training:  mask ~ Bernoulli(1 - p), y = x · mask / (1 - p)
    eval:      y = x  (identity)

Backward (mask cached from forward — the scaled 0-or-1/(1-p) mask):
    training:  grad_x = grad_y · mask
    eval:      grad_x = grad_y

RNG idiom (storage): the CPU path keeps the legacy host PhiloxRandom counter
(`call_counter`, host-bumped). The GPU path holds the Philox offset as a 1-elem
device `TensorImpl[uint64]` (`noise_offset`) — mirroring `noisy_linear` /
`rsample` — and the train kernel reads the offset FROM that device buffer (a
LayoutTensor passed via `lt_gpu`, NO raw ptr); after the draw the offset is
advanced with the shared `advance_rng_offset_kernel`. That keeps the device
offset capture-friendly and the surface unsafe_ptr-free.

Streams (`set_attr["rng_stream"](k)`, default 0): every draw's Philox offset
is moved by `k << 48`, so stream k's masks are independent of stream 0's.
The seed is a type parameter, so data-parallel replicas of one model would
otherwise draw the same masks on different rows; a distributed trainer puts
rank r on stream r. Stream 0 is the unchanged sequence.

Copies (`set_attr["rng_copy"](i)`, sent by `Repeat.make` to its copy i):
every draw uses Philox subsequence i. The copies of a repeated block share
one type, hence one SEED, and with the same counter they drew the SAME mask
at every forward (every GPT block dropped the same entries until
2026-10-12). Subsequence 0 (copy 0, and every dropout outside a `Repeat`)
is the unchanged sequence. The attr appends 16 bits, so the copies of a
nested `Repeat` stay distinct (the outer index lands in the low bits: the
outer `make` runs after the inner one).

Math note (carried verbatim): mask = scale if u >= p else 0, with
scale = 1/(1-p), threshold = p, u ~ U(0,1) from PhiloxRandom.step_uniform().

Cache: leaf-owned `[BATCH, DIM]` slab — the SCALED mask (0 or 1/(1-p)) so
backward is one elementwise multiply.

bf16-FLOW (AMP "Step B"): `Dropout[DIM]` is fp32 (unchanged), while
`Dropout[DIM, p, SEED, DType.bfloat16]` flows ACTIVATIONS at bf16 (`ACT_DT ==
bfloat16`). The cached mask is an ACTIVATION → stored at the flow dtype `ADT`.
The mask VALUES (0 or 1/(1-p)) are representable in bf16; the RNG draw/comparison
stays fp32, the mask is cast to ADT on write. Forward = bf16 in × bf16 mask → bf16
out (train) / bf16 passthrough (eval). Backward = bf16 grad × bf16 mask. The
mask-apply kernels are dtype-parametric (`ADT`); the `set_training` toggle still
works. The fp32 (ACT_DT == DT) path is byte-for-byte the legacy NoAMP path; the
bf16 path is GPU-only.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor, TileTensor, row_major

from noeira.nn.constants import DT, TPB, CPU_SIMD_W
from noeira.nn.random.box_muller import advance_rng_offset_kernel
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — flat 1-D launch over BATCH·DIM. PhiloxRandom keyed by the
# type's SEED, on the instance's subsequence (its `Repeat` copy), at offset
# (stream_base + base_offset + idx), so each (forward, lane) pair draws a
# fresh number. Carried VERBATIM from legacy, except the train kernel reads
# `base_offset` from a 1-elem device buffer (storage RNG idiom) instead of a
# baked scalar arg, and the stream and subsequence arguments.
#
# Dtype-parametric (`ADT`): the fp32 path runs the mask-apply at DT, the
# bf16-flow path at bfloat16 (input/output/mask all bf16). The RNG draw +
# threshold comparison run at DT (the mask VALUE is then cast to ADT on
# write — the values 0 / 1/(1-p) are representable in bf16).
# ──────────────────────────────────────────────────────────────────────


def _dropout_train_forward_kernel[
    N: Int, SEED: UInt64, ADT: DType = DT
](
    input: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
    cache: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
    threshold: Scalar[DT],
    scale: Scalar[DT],
    stream_base: UInt64,
    subsequence: UInt64,
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N:
        return
    var base_offset = rebind[UInt64](offset_buf[0])
    var rng = PhiloxRandom(
        seed=SEED,
        subsequence=subsequence,
        offset=stream_base + base_offset + UInt64(idx),
    )
    var rand = Scalar[DT](rng.step_uniform()[0])
    var mask: Scalar[DT] = scale if rand >= threshold else Scalar[DT](0.0)
    cache[idx] = mask.cast[ADT]()
    output[idx] = rebind[Scalar[ADT]](input[idx]) * mask.cast[ADT]()


def _dropout_eval_kernel[N: Int, ADT: DType = DT](
    src: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N:
        return
    dst[idx] = rebind[Scalar[ADT]](src[idx])


def _dropout_train_backward_kernel[N: Int, ADT: DType = DT](
    grad_output: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
    cache: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
    grad_input: LayoutTensor[ADT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N:
        return
    grad_input[idx] = (
        rebind[Scalar[ADT]](grad_output[idx])
        * rebind[Scalar[ADT]](cache[idx])
    )


struct Dropout[
    DIM_: Int, p: Float64 = 0.5, SEED: UInt64 = 1, ADT: DType = DT
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_
    # Activation-flow dtype. `Dropout[DIM]` = fp32 (ACT_DT == DT, the legacy
    # path); `Dropout[DIM, p, SEED, bfloat16]` flows activations at bf16.
    comptime ACT_DT = Self.ADT

    # ⚠ p == 0 IS IDENTITY, AND THE TRAIN PATH STILL RAN.
    # `NormedLinearDropout` is structurally always present (Mojo cannot
    # conditionally alias two type shapes — see nets.mojo), and TD-MPC2's
    # default `QP = 0.0` makes this layer mathematically a no-op: mask =
    # 1/(1-0) = 1 and threshold = 0, so every element passes scaled by exactly
    # 1.0. But `training` defaults to True, so each call still launched the
    # masked-forward kernel AND `advance_rng_offset_kernel` — the latter a
    # grid_dim=1, one-thread launch to increment a counter nothing reads.
    # In a TD-MPC2 MPC plan that is 8 Q-head forwards x 2 launches.
    # The eval kernel is already an exact copy, so p==0 routes there: same
    # values (x * 1.0 is exact), one launch instead of two, and no mask buffer.
    comptime IS_IDENTITY = Self.p == 0.0

    # Runtime state.
    var training: Bool
    var call_counter: UInt64  # host CPU Philox counter (legacy slot 0)
    var stream_base: UInt64
    """Added to every draw's Philox offset: `rng_stream << 48` (0 = default)."""
    var subsequence: UInt64
    """Every draw's Philox subsequence: the `Repeat` copy this dropout sits in
    (`rng_copy`; 0 = the first copy, or not repeated)."""
    # [BATCH, DIM] scaled mask (0 or 1/(1-p)) — an ACTIVATION → stored at the
    # flow dtype `Self.ADT` (bf16 in the bf16-flow path).
    var cache_mask: TensorImpl[Self.ADT]
    var noise_offset: TensorImpl[DType.uint64]  # GPU Philox offset (1 elem)

    def __init__(out self):
        self.training = True
        self.call_counter = 0
        self.stream_base = 0
        self.subsequence = 0
        self.cache_mask = TensorImpl[Self.ADT]()
        self.noise_offset = TensorImpl[DType.uint64]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params)."""
        comptime assert target == "cpu" or target == "gpu", (
            "Dropout: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.p >= 0.0 and Self.p < 1.0, (
            "Dropout: p must be in [0, 1)"
        )
        var d = Self()
        comptime if target == "gpu":
            var c = ctx.value()
            d.noise_offset.ensure_gpu(c, 1)
            d.noise_offset.dev.value().enqueue_fill(UInt64(0))
        return d^

    def set_training(mut self, v: Bool):
        self.training = v

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Named-attr hook so a parent `Sequential` / `Repeat` / `ComputeGraph`
        can switch this dropout to eval via `net.set_attr["training"](0.0)`.
        `value != 0` -> training (sample a mask); else eval (identity).

        ⚠ Without this, `set_training` was reachable only on a Dropout the
        caller held DIRECTLY. Any dropout nested inside a container was stuck
        in training mode for its whole life, and `set_attr["training"](0)` on
        the container returned quietly — `Module.set_attr`'s default is `pass`,
        so the call site looked correct and did nothing. `BatchNorm2D` has had
        this override since the AlphaZero BN switch; dropout was missed.
        """
        comptime if ATTR == "training":
            self.training = value != Scalar[DT](0.0)
        elif ATTR == "rng_stream":
            # A rank index: an integer far below fp32's 2^24.
            self.stream_base = UInt64(Int(value)) << 48
        elif ATTR == "rng_copy":
            # A `Repeat` copy index (< 2^16), appended below the copy
            # indices of any `Repeat` nested inside that copy.
            self.subsequence = (self.subsequence << 16) | UInt64(Int(value))

    def set_seed(mut self) raises:
        """Reset the per-instance RNG counters to the start of the stream."""
        self.call_counter = 0
        if self.noise_offset.dev:
            self.noise_offset.dev.value().enqueue_fill(UInt64(0))

    def forward[
        target: StaticString,
        B: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime N = B * Self.DIM_
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT here — rebind the activation refs (sound; dtypes equal).
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            ref maskd = rebind[Tensor](self.cache_mask)
            comptime if target == "cpu":
                outd.ensure(N)
                if Self.IS_IDENTITY or not self.training:
                    # SIMD copy — the scalar loop this replaces cost 247 us at
                    # B=268, DIM=512 (1.8 ns/element) to move 137k floats.
                    var sp = in0d.data.unsafe_ptr()
                    var dp = outd.data.unsafe_ptr()
                    var i = 0
                    while i + CPU_SIMD_W <= N:
                        dp.unsafe_store(i, sp.unsafe_load[width=CPU_SIMD_W](i))
                        i += CPU_SIMD_W
                    while i < N:
                        dp[unsafe_offset=i] = sp[unsafe_offset=i]
                        i += 1
                    # Eval pass doesn't bump the counter — keeps training
                    # determinism cleanly separated from eval calls.
                    return
                maskd.ensure(N)
                var scale = Scalar[DT](1.0 / (1.0 - Self.p))
                var threshold = Scalar[DT](Self.p)
                var zero = Scalar[DT](0.0)
                var base_offset = (
                    self.stream_base + self.call_counter * UInt64(N)
                )
                for b in range(B):
                    for i in range(Self.DIM_):
                        var idx = b * Self.DIM_ + i
                        var rng = PhiloxRandom(
                            seed=Self.SEED,
                            subsequence=self.subsequence,
                            offset=base_offset + UInt64(idx),
                        )
                        var rand = Scalar[DT](rng.step_uniform()[0])
                        var mask: Scalar[DT] = (
                            scale if rand >= threshold else zero
                        )
                        maskd.data[idx] = mask
                        outd.data[idx] = in0d.data[idx] * mask
                self.call_counter += 1
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, N)
                comptime n_blocks = (N + TPB - 1) // TPB
                comptime lN = Layout.row_major(N)
                comptime loff = Layout.row_major(1)
                if Self.IS_IDENTITY or not self.training:
                    comptime eval_kernel = _dropout_eval_kernel[N]
                    c.enqueue_function[eval_kernel](
                        in0d.lt["gpu", lN](),
                        outd.lt["gpu", lN](),
                        grid_dim=n_blocks,
                        block_dim=TPB,
                    )
                    return
                maskd.ensure_gpu(c, N)
                var scale = Scalar[DT](1.0 / (1.0 - Self.p))
                var threshold = Scalar[DT](Self.p)
                comptime train_kernel = _dropout_train_forward_kernel[
                    N, Self.SEED
                ]
                c.enqueue_function[train_kernel](
                    in0d.lt["gpu", lN](),
                    outd.lt["gpu", lN](),
                    maskd.lt["gpu", lN](),
                    self.noise_offset.lt["gpu", loff](),
                    threshold,
                    scale,
                    self.stream_base,
                    self.subsequence,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
                c.enqueue_function[advance_rng_offset_kernel[N]](
                    self.noise_offset.lt["gpu", loff](),
                    grid_dim=1,
                    block_dim=1,
                )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow Dropout is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime lN = Layout.row_major(N)
            comptime loff = Layout.row_major(1)
            if Self.IS_IDENTITY or not self.training:
                comptime eval_kernel = _dropout_eval_kernel[N, Self.ADT]
                c.enqueue_function[eval_kernel](
                    in0.lt["gpu", lN](),
                    out.lt["gpu", lN](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
                return
            self.cache_mask.ensure_gpu(c, N)
            var scale = Scalar[DT](1.0 / (1.0 - Self.p))
            var threshold = Scalar[DT](Self.p)
            comptime train_kernel = _dropout_train_forward_kernel[
                N, Self.SEED, Self.ADT
            ]
            c.enqueue_function[train_kernel](
                in0.lt["gpu", lN](),
                out.lt["gpu", lN](),
                self.cache_mask.lt["gpu", lN](),
                self.noise_offset.lt["gpu", loff](),
                threshold,
                scale,
                self.stream_base,
                self.subsequence,
                grid_dim=n_blocks,
                block_dim=TPB,
            )
            c.enqueue_function[advance_rng_offset_kernel[N]](
                self.noise_offset.lt["gpu", loff](),
                grid_dim=1,
                block_dim=1,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime N = B * Self.DIM_
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            ref gind = rebind[Tensor](gin)
            ref god = rebind[Tensor](grad_output)
            ref maskd = rebind[Tensor](self.cache_mask)
            comptime if target == "cpu":
                gind.ensure(N)
                if Self.IS_IDENTITY or not self.training:
                    for i in range(N):
                        gind.data[i] = god.data[i]
                    return
                for i in range(N):
                    gind.data[i] = god.data[i] * maskd.data[i]
            else:
                var c = ctx.value()
                gind.ensure_gpu(c, N)
                comptime n_blocks = (N + TPB - 1) // TPB
                comptime lN = Layout.row_major(N)
                if Self.IS_IDENTITY or not self.training:
                    comptime eval_kernel = _dropout_eval_kernel[N]
                    c.enqueue_function[eval_kernel](
                        god.lt["gpu", lN](),
                        gind.lt["gpu", lN](),
                        grid_dim=n_blocks,
                        block_dim=TPB,
                    )
                    return
                comptime back_kernel = _dropout_train_backward_kernel[N]
                c.enqueue_function[back_kernel](
                    god.lt["gpu", lN](),
                    maskd.lt["gpu", lN](),
                    gind.lt["gpu", lN](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow Dropout is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime lN = Layout.row_major(N)
            if Self.IS_IDENTITY or not self.training:
                comptime eval_kernel = _dropout_eval_kernel[N, Self.ADT]
                c.enqueue_function[eval_kernel](
                    grad_output.lt["gpu", lN](),
                    gin.lt["gpu", lN](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
                return
            comptime back_kernel = _dropout_train_backward_kernel[N, Self.ADT]
            c.enqueue_function[back_kernel](
                grad_output.lt["gpu", lN](),
                self.cache_mask.lt["gpu", lN](),
                gin.lt["gpu", lN](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad / polyak_from inherit the Module reflection
    # defaults (param-less: reflection finds no IsParam fields).
