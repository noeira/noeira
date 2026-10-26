"""ConvTransposePCBlock — one transposed-convolution PCN level (Bogacz, bottom-up).

The generative dual of `ConvPCBlock`. Where ConvPCBlock.predict is a forward
(downsampling) conv, this block's predict is a TRANSPOSED (upsampling) conv:
a small below latent predicts a larger above latent. A transposed conv is the
adjoint of a forward conv, so the three ops mirror ConvPCBlock:

    a_below = ACT(x_below)                                  # [B, C_in·H·W]   (small)
    μ       = ConvT(a_below; W, b)                          # [B, C_out·H'·W'](big)   ← predict (scatter)
    ε       = x_above − μ
    z_below = Conv(ε; W)                                    # [B, C_in·H·W]          ← pull_back (im2col+GEMM)
    z_gated = z_below ⊙ ACT'(x_below)
    dW      = −corr(a_below, im2col(ε)) ; db = −Σ ε         #                        ← weight_grad

Output spatial (PyTorch ConvTranspose2d convention):
    out_h = (in_h − 1)·stride − 2·padding + kernel_size

Layout:
  - x_below idx:  ic·in_h·in_w + ih·in_w + iw            (C_in·in_h·in_w = IN_DIM, small)
  - μ idx:        oc·big_spatial + oh·out_w + ow         (C_out·out_h·out_w = OUT_DIM, big)
  - small↔big:    oh = ih·stride − padding + kh,  ow = iw·stride − padding + kw
  - W:            params[ic·col_size + (oc·k² + kh·k + kw)],  col_size = C_out·k²
  - bias:         params[W_SIZE + oc]   (per BIG output channel)
  - PARAM_SIZE =  C_in·col_size + C_out

CPU-only (naive scalar fallback + Apple Accelerate fast path). No GPU / no
PCBlockTrait conformance — callers invoke the static methods directly (the
bidirectional generative path does not go through PCSequential).
"""

from std.sys import CompilationTarget

from .pc_initializer import PCInitializer
from .pc_apple_cblas import apple_sgemm_accum

from .predictive_model import PCActivation, PCReLU
from noeira.core.tiles import Tile, tile


struct ConvTransposePCBlock[
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    in_h: Int,
    in_w: Int,
    ACT: PCActivation = PCReLU,
](ImplicitlyCopyable):
    comptime out_h: Int = (
        (Self.in_h - 1) * Self.stride - 2 * Self.padding + Self.kernel_size
    )
    comptime out_w: Int = (
        (Self.in_w - 1) * Self.stride - 2 * Self.padding + Self.kernel_size
    )
    comptime small_spatial: Int = Self.in_h * Self.in_w
    comptime big_spatial: Int = Self.out_h * Self.out_w
    comptime col_size: Int = (
        Self.out_channels * Self.kernel_size * Self.kernel_size
    )
    comptime CACHE: Int = Self.small_spatial * Self.col_size

    comptime IN_DIM: Int = Self.in_channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.out_channels * Self.out_h * Self.out_w
    comptime W_SIZE: Int = Self.in_channels * Self.col_size
    comptime PARAM_SIZE: Int = Self.W_SIZE + Self.out_channels

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    # =========================================================================
    # Init — transposed-conv fans (fan_in = C_in·k², fan_out = C_out·k²)
    # =========================================================================

    @staticmethod
    def pc_init_params[
        INIT: PCInitializer, dtype: DType = DType.float32
    ](
        params: Tile[dtype, Self.PARAM_SIZE],
    ) raises:
        """Init: conv-transpose W via INIT.fill(fan_in/out); zero bias."""
        var W_view = tile[Self.W_SIZE](params.ptr)
        INIT.fill[
            Self.W_SIZE,
            Self.in_channels * Self.kernel_size * Self.kernel_size,
            Self.out_channels * Self.kernel_size * Self.kernel_size,
            dtype,
        ](W_view)
        for j in range(Self.out_channels):
            params.ptr[unsafe_offset=Self.W_SIZE + j] = Scalar[dtype](0)

    # =========================================================================
    # im2col of the BIG tensor (ε or x_above) → ecol[b, p_small·col_size + ocol]
    #   ecol[p_small, oc·k²+kh·k+kw] = big[oc, oh, ow]  (oh = ih·s − p + kh) or 0
    # =========================================================================

    @staticmethod
    def _im2col_big[
        BATCH: Int, dtype: DType
    ](
        big: Tile[dtype, BATCH, Self.OUT_DIM],
        mut ecol: List[Scalar[dtype]],
    ):
        for b in range(BATCH):
            for ih in range(Self.in_h):
                for iw in range(Self.in_w):
                    var p_small = ih * Self.in_w + iw
                    for oc in range(Self.out_channels):
                        for kh in range(Self.kernel_size):
                            for kw in range(Self.kernel_size):
                                var oh = ih * Self.stride - Self.padding + kh
                                var ow = iw * Self.stride - Self.padding + kw
                                var ocol = (
                                    oc * Self.kernel_size * Self.kernel_size
                                    + kh * Self.kernel_size
                                    + kw
                                )
                                var ci = (
                                    b * Self.CACHE
                                    + p_small * Self.col_size
                                    + ocol
                                )
                                if (
                                    oh >= 0
                                    and oh < Self.out_h
                                    and ow >= 0
                                    and ow < Self.out_w
                                ):
                                    ecol[ci] = rebind[Scalar[dtype]](
                                        big[
                                            b,
                                            oc * Self.big_spatial
                                            + oh * Self.out_w
                                            + ow,
                                        ]
                                    )
                                else:
                                    ecol[ci] = Scalar[dtype](0)

    # =========================================================================
    # predict:  a_below = ACT(x_below);  μ = ConvT(a_below; W) + b   (upsample)
    # =========================================================================

    @staticmethod
    def predict[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        mu: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        Self.ACT.apply[BATCH, Self.IN_DIM, dtype](x_below, a_below)
        comptime use_apple = CompilationTarget.is_macos() and (
            dtype == DType.float32
        )
        # Full param-slab view (no rebind): weight head W[0:W_SIZE] + bias tail
        # W[W_SIZE + oc]. Matches the original whole-slab pointer `wp`.
        var W = tile[Self.PARAM_SIZE](params.ptr)

        for b in range(BATCH):
            for i in range(Self.OUT_DIM):
                mu[b, i] = Scalar[dtype](0)

        comptime if use_apple:
            # Step 1: dcol[b, p_small·col_size + ocol] = Σ_ic a_below[ic,p_small]·W[ic,ocol]
            #         (transpose_a GEMM: a_belowᵀ @ W per batch). dcol is an owned
            #         List (no raw alloc); its pointer feeds the cblas FFI.
            var dcol = List[Scalar[dtype]](
                length=BATCH * Self.CACHE, fill=Scalar[dtype](0)
            )
            # cblas FFI boundary (Scalar→Float32 + origin), as nn.storage's Linear.
            var Wp = rebind[Pointer[Float32, ImmutAnyOrigin]](params.ptr)
            for b in range(BATCH):
                var a_b = rebind[Pointer[Float32, ImmutAnyOrigin]](
                    a_below.ptr.unsafe_offset(b * Self.IN_DIM)
                )
                var dcol_b = rebind[Pointer[Float32, MutAnyOrigin]](
                    dcol.unsafe_ptr().unsafe_offset(b * Self.CACHE)
                )
                try:
                    apple_sgemm_accum[transpose_a=True, transpose_b=False](
                        Self.small_spatial,
                        Self.col_size,
                        Self.in_channels,
                        Float32(1.0),
                        a_b,
                        Self.small_spatial,
                        Wp,
                        Self.col_size,
                        Float32(0.0),
                        dcol_b,
                        Self.col_size,
                    )
                except:
                    pass
            # Step 2: col2im scatter dcol → μ (accumulate over overlapping taps)
            for b in range(BATCH):
                for ih in range(Self.in_h):
                    for iw in range(Self.in_w):
                        var p_small = ih * Self.in_w + iw
                        for oc in range(Self.out_channels):
                            for kh in range(Self.kernel_size):
                                for kw in range(Self.kernel_size):
                                    var oh = (
                                        ih * Self.stride - Self.padding + kh
                                    )
                                    var ow = (
                                        iw * Self.stride - Self.padding + kw
                                    )
                                    if (
                                        oh >= 0
                                        and oh < Self.out_h
                                        and ow >= 0
                                        and ow < Self.out_w
                                    ):
                                        var ocol = (
                                            oc
                                            * Self.kernel_size
                                            * Self.kernel_size
                                            + kh * Self.kernel_size
                                            + kw
                                        )
                                        var mcol = (
                                            oc * Self.big_spatial
                                            + oh * Self.out_w
                                            + ow
                                        )
                                        mu[b, mcol] = (
                                            mu[b, mcol]
                                            + dcol[
                                                b * Self.CACHE
                                                + p_small * Self.col_size
                                                + ocol
                                            ]
                                        )
            _ = dcol^
        else:
            for b in range(BATCH):
                for ih in range(Self.in_h):
                    for iw in range(Self.in_w):
                        for ic in range(Self.in_channels):
                            var av = a_below[
                                b,
                                ic * Self.small_spatial + ih * Self.in_w + iw,
                            ]
                            for oc in range(Self.out_channels):
                                for kh in range(Self.kernel_size):
                                    for kw in range(Self.kernel_size):
                                        var oh = (
                                            ih * Self.stride - Self.padding + kh
                                        )
                                        var ow = (
                                            iw * Self.stride - Self.padding + kw
                                        )
                                        if (
                                            oh >= 0
                                            and oh < Self.out_h
                                            and ow >= 0
                                            and ow < Self.out_w
                                        ):
                                            var widx = (
                                                ic * Self.col_size
                                                + oc
                                                * Self.kernel_size
                                                * Self.kernel_size
                                                + kh * Self.kernel_size
                                                + kw
                                            )
                                            var mcol = (
                                                oc * Self.big_spatial
                                                + oh * Self.out_w
                                                + ow
                                            )
                                            mu[b, mcol] = mu[b, mcol] + (
                                                W[widx] * av
                                            )

        # + bias per BIG output channel
        for b in range(BATCH):
            for oc in range(Self.out_channels):
                var bias_val = W[Self.W_SIZE + oc]
                var off = oc * Self.big_spatial
                for s in range(Self.big_spatial):
                    mu[b, off + s] = mu[b, off + s] + bias_val

    # =========================================================================
    # eps_compute:  ε = x_above − μ   (elementwise)
    # =========================================================================

    @staticmethod
    def eps_compute[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_above: Tile[dtype, BATCH, Self.OUT_DIM],
        mu: Tile[dtype, BATCH, Self.OUT_DIM],
        eps: Tile[dtype, BATCH, Self.OUT_DIM],
    ):
        for b in range(BATCH):
            for j in range(Self.OUT_DIM):
                eps[b, j] = x_above[b, j] - mu[b, j]

    # =========================================================================
    # pull_back:  z_below = Conv(ε; W)   (forward conv = adjoint of predict)
    # =========================================================================

    @staticmethod
    def pull_back[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        z_below: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        comptime use_apple = CompilationTarget.is_macos() and (
            dtype == DType.float32
        )
        # W sub-view over the param slab (no rebind).
        var W = tile[Self.W_SIZE](params.ptr)

        comptime if use_apple:
            # im2col(ε_big) → ecol, then z_small[ic,p] = Σ_ocol W[ic,ocol]·ecol[p,ocol]
            # (W @ ecolᵀ per batch). ecol is an owned List (no raw alloc).
            var ecol = List[Scalar[dtype]](
                length=BATCH * Self.CACHE, fill=Scalar[dtype](0)
            )
            Self._im2col_big[BATCH, dtype](eps_above, ecol)
            # cblas FFI boundary (Scalar→Float32 + origin), as nn.storage's Linear.
            var Wp = rebind[Pointer[Float32, ImmutAnyOrigin]](params.ptr)
            for b in range(BATCH):
                var ecol_b = rebind[Pointer[Float32, ImmutAnyOrigin]](
                    ecol.unsafe_ptr().unsafe_offset(b * Self.CACHE)
                )
                var z_b = rebind[Pointer[Float32, MutAnyOrigin]](
                    z_below.ptr.unsafe_offset(b * Self.IN_DIM)
                )
                try:
                    apple_sgemm_accum[transpose_a=False, transpose_b=True](
                        Self.in_channels,
                        Self.small_spatial,
                        Self.col_size,
                        Float32(1.0),
                        Wp,
                        Self.col_size,
                        ecol_b,
                        Self.col_size,
                        Float32(0.0),
                        z_b,
                        Self.small_spatial,
                    )
                except:
                    pass
            _ = ecol^
        else:
            for b in range(BATCH):
                for i in range(Self.IN_DIM):
                    z_below[b, i] = Scalar[dtype](0)
            for b in range(BATCH):
                for ih in range(Self.in_h):
                    for iw in range(Self.in_w):
                        for ic in range(Self.in_channels):
                            var acc = eps_above[b, 0] - eps_above[b, 0]
                            for oc in range(Self.out_channels):
                                for kh in range(Self.kernel_size):
                                    for kw in range(Self.kernel_size):
                                        var oh = (
                                            ih * Self.stride - Self.padding + kh
                                        )
                                        var ow = (
                                            iw * Self.stride - Self.padding + kw
                                        )
                                        if (
                                            oh >= 0
                                            and oh < Self.out_h
                                            and ow >= 0
                                            and ow < Self.out_w
                                        ):
                                            var widx = (
                                                ic * Self.col_size
                                                + oc
                                                * Self.kernel_size
                                                * Self.kernel_size
                                                + kh * Self.kernel_size
                                                + kw
                                            )
                                            acc = acc + (
                                                W[widx]
                                                * eps_above[
                                                    b,
                                                    oc * Self.big_spatial
                                                    + oh * Self.out_w
                                                    + ow,
                                                ]
                                            )
                            z_below[
                                b,
                                ic * Self.small_spatial + ih * Self.in_w + iw,
                            ] = acc

    # =========================================================================
    # act_derivative_mul:  z_out = z_in ⊙ ACT'(x_below)
    # =========================================================================

    @staticmethod
    def act_derivative_mul[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        z_in: Tile[dtype, BATCH, Self.IN_DIM],
        z_out: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        Self.ACT.apply_derivative_mul[BATCH, Self.IN_DIM, dtype](
            x_below, z_in, z_out
        )

    # =========================================================================
    # weight_grad:  dW = −Σ_b a_below @ im2col(ε) ;  db = −Σ ε   (−sign baked)
    # =========================================================================

    @staticmethod
    def weight_grad[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
        grads: Tile[dtype, Self.PARAM_SIZE],
    ):
        comptime use_apple = CompilationTarget.is_macos() and (
            dtype == DType.float32
        )
        # Grad slab as a flat view + W head sub-view (no rebind).
        var grads_flat = tile[Self.PARAM_SIZE](grads.ptr)
        var W_grad = tile[Self.W_SIZE](grads.ptr)
        for i in range(Self.PARAM_SIZE):
            grads_flat[i] = Scalar[dtype](0)

        comptime if use_apple:
            # ecol = im2col(ε_big); dW[ic,ocol] += (−1)·Σ_p a_below[ic,p]·ecol[p,ocol]
            # (alpha=−1, beta=1 accumulate over batch). ecol = owned List.
            var ecol = List[Scalar[dtype]](
                length=BATCH * Self.CACHE, fill=Scalar[dtype](0)
            )
            Self._im2col_big[BATCH, dtype](eps_above, ecol)
            # cblas FFI boundary (Scalar→Float32 + origin), as nn.storage's Linear.
            var Cw = rebind[Pointer[Float32, MutAnyOrigin]](grads.ptr)
            for b in range(BATCH):
                var a_b = rebind[Pointer[Float32, ImmutAnyOrigin]](
                    a_below.ptr.unsafe_offset(b * Self.IN_DIM)
                )
                var ecol_b = rebind[Pointer[Float32, ImmutAnyOrigin]](
                    ecol.unsafe_ptr().unsafe_offset(b * Self.CACHE)
                )
                try:
                    apple_sgemm_accum[transpose_a=False, transpose_b=False](
                        Self.in_channels,
                        Self.col_size,
                        Self.small_spatial,
                        Float32(-1.0),
                        a_b,
                        Self.small_spatial,
                        ecol_b,
                        Self.col_size,
                        Float32(1.0),
                        Cw,
                        Self.col_size,
                    )
                except:
                    pass
            _ = ecol^
        else:
            for b in range(BATCH):
                for ih in range(Self.in_h):
                    for iw in range(Self.in_w):
                        var p_small = ih * Self.in_w + iw
                        _ = p_small
                        for ic in range(Self.in_channels):
                            var av = a_below[
                                b,
                                ic * Self.small_spatial + ih * Self.in_w + iw,
                            ]
                            for oc in range(Self.out_channels):
                                for kh in range(Self.kernel_size):
                                    for kw in range(Self.kernel_size):
                                        var oh = (
                                            ih * Self.stride - Self.padding + kh
                                        )
                                        var ow = (
                                            iw * Self.stride - Self.padding + kw
                                        )
                                        if (
                                            oh >= 0
                                            and oh < Self.out_h
                                            and ow >= 0
                                            and ow < Self.out_w
                                        ):
                                            var widx = (
                                                ic * Self.col_size
                                                + oc
                                                * Self.kernel_size
                                                * Self.kernel_size
                                                + kh * Self.kernel_size
                                                + kw
                                            )
                                            W_grad[widx] = W_grad[widx] + (
                                                -av
                                                * eps_above[
                                                    b,
                                                    oc * Self.big_spatial
                                                    + oh * Self.out_w
                                                    + ow,
                                                ]
                                            )

        # db[oc] = −Σ_{b, big spatial} ε  (accumulate in the element type)
        for oc in range(Self.out_channels):
            var acc = eps_above[0, 0] - eps_above[0, 0]
            for b in range(BATCH):
                var off = oc * Self.big_spatial
                for s in range(Self.big_spatial):
                    acc = acc + eps_above[b, off + s]
            grads_flat[Self.W_SIZE + oc] = -acc
