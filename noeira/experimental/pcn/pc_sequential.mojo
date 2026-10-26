"""PCSequential — variadic composition of PCBlocks (Bogacz canonical).

Architecture (3-block example):

    input ──[block_0]── x_1 ──[block_1]── x_2 ──[block_2]── output ↔ target
            (Identity)         (ReLU)              (ReLU)            (loss)

Block 0 typically uses ACT=PCIdentity (input is clamped data, no activation
needed). Interior + readout blocks usually use PCReLU. The very last block
is the readout — its OUT_DIM is the network's output dim, and there's no
latent above it: the supervised target plays the role of x_above for the
readout.

Latent storage: only INTERIOR latents are stored. For N blocks, there are
N-1 interior latents (x_1 .. x_{N-1}). The bottom is clamped to input,
the top is clamped to target via the output loss.

Layouts (per sample, for one inference step):
    params:    [block_0 (W + b) | block_1 (W + b) | ... | block_{N-1} (W + b)]
    latents:   [x_1 | x_2 | ... | x_{N-1}]    interior only
    mu_buf:    [μ_1 | μ_2 | ... | μ_N]        per-block, OUT_DIM each
    eps_buf:   [ε_1 | ε_2 | ... | ε_N]        per-block, OUT_DIM each
    a_below:   [a_0 | a_1 | ... | a_{N-1}]    per-block, IN_DIM each (cached ACT(x_below))
    z_below:   [z_0 | z_1 | ... | z_{N-1}]    per-block, IN_DIM each (W^T·ε_above)

For Phase 1 (CPU), no alignment padding between blocks. Add for GPU later.
"""

from max.gpu.host import DeviceContext


from .predictive_model import PCBlockTrait
from .pc_initializer import PCInitializer
from noeira.core.tiles import Tile, tile


@fieldwise_init
struct PCSequential[*BLOCKS: PCBlockTrait]:
    """Variadic chain of N PCBlocks.

    Compose like:
        alias NET = PCSequential[
            PCBlock[784, 256, PCIdentity],   # block_0: input → h_1
            PCBlock[256, 256, PCReLU],       # block_1: h_1   → h_2
            PCBlock[256, 10,  PCReLU],       # block_2: h_2   → output (readout)
        ]
    """

    comptime block_types = Self.BLOCKS
    comptime N = Self.block_types.length
    comptime N_LATENTS = Self.N - 1

    comptime IN_DIM: Int = Self.block_types[0].IN_DIM
    comptime OUT_DIM: Int = Self.block_types[Self.N - 1].OUT_DIM

    # =========================================================================
    # Sum helpers
    # =========================================================================

    @staticmethod
    def _sum_param_size() -> Int:
        var total = 0
        comptime for i in range(Self.N):
            total += Self.block_types[i].PARAM_SIZE
        return total

    @staticmethod
    def _sum_latent_dim() -> Int:
        """Sum of OUT_DIM for blocks 0..N-2 (interior latents only)."""
        var total = 0
        comptime for i in range(Self.N - 1):
            total += Self.block_types[i].OUT_DIM
        return total

    @staticmethod
    def _sum_out_dim() -> Int:
        """Sum of OUT_DIM across ALL blocks (for μ, ε scratch)."""
        var total = 0
        comptime for i in range(Self.N):
            total += Self.block_types[i].OUT_DIM
        return total

    @staticmethod
    def _sum_in_dim() -> Int:
        """Sum of IN_DIM across ALL blocks (for a_below, z_below scratch)."""
        var total = 0
        comptime for i in range(Self.N):
            total += Self.block_types[i].IN_DIM
        return total

    comptime PARAM_SIZE: Int = Self._sum_param_size()
    comptime LATENT_DIM: Int = Self._sum_latent_dim()
    comptime SCRATCH_OUT_DIM: Int = Self._sum_out_dim()
    comptime SCRATCH_IN_DIM: Int = Self._sum_in_dim()

    # =========================================================================
    # Offset helpers
    # =========================================================================

    @staticmethod
    def _param_offset[idx: Int]() -> Int:
        var total = 0
        comptime for j in range(idx):
            total += Self.block_types[j].PARAM_SIZE
        return total

    @staticmethod
    def _latent_offset[idx: Int]() -> Int:
        """Offset (per sample) of interior latent x_{idx+1} in the latents buffer.
        Valid for idx in 0..N_LATENTS-1.
        """
        var total = 0
        comptime for j in range(idx):
            total += Self.block_types[j].OUT_DIM
        return total

    @staticmethod
    def _out_offset[idx: Int]() -> Int:
        """Offset (per sample) of block idx's μ_idx / ε_idx slot in the out buffer.
        """
        var total = 0
        comptime for j in range(idx):
            total += Self.block_types[j].OUT_DIM
        return total

    @staticmethod
    def _in_offset[idx: Int]() -> Int:
        """Offset (per sample) of block idx's a_below / z_below slot."""
        var total = 0
        comptime for j in range(idx):
            total += Self.block_types[j].IN_DIM
        return total

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def pc_init_params[
        INIT: PCInitializer, dtype: DType = DType.float32
    ](
        params: Tile[dtype, Self.PARAM_SIZE],
    ) raises:
        """Init: per-block `pc_init_params` (vendored
        `PCInitializer`, legacy-`nn`-free). Mirror of `initialize_params`."""
        for i in range(Self.PARAM_SIZE):
            params.ptr[unsafe_offset=i] = Scalar[dtype](0)

        comptime for i in range(Self.N):
            var li_p = tile[Self.block_types[i].PARAM_SIZE](params.ptr.unsafe_offset(Self._param_offset[i]()))
            Self.block_types[i].pc_init_params[INIT, dtype](li_p)

    # =========================================================================
    # Forward eval — plain feedforward bottom-up, NO latent allocation,
    # NO inference loop. This is the test-time / classification path.
    # Mirrors a plain MLP forward through `μ_above = W·act(x_below) + b`.
    # =========================================================================

    @staticmethod
    def forward_eval[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_in: Tile[dtype, BATCH, Self.IN_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        output: Tile[dtype, BATCH, Self.OUT_DIM],
    ):
        """Feedforward: x_in → block_0.predict → ... → block_{N-1}.predict → output.

        Each block applies its own ACT to its input then computes
        `μ = ACT(input) @ W + b`. Block 0 typically has PCIdentity so input
        is consumed unchanged; subsequent blocks apply their ACT to the
        previous block's μ output (acting as the latent in eval mode).

        This routine ALLOCATES intermediate buffers (`a_below_l` and
        `μ_l`) internally — fine for CPU eval where we don't need to be
        zero-alloc. GPU version will use a workspace.
        """
        # Scratch: a_below_l (per block, IN_DIM_l) and μ_l (per block, OUT_DIM_l).
        # Allocate one flat buffer per kind, with comptime offsets.
        var a_storage = List[Scalar[dtype]](
            capacity=BATCH * Self.SCRATCH_IN_DIM
        )
        for _ in range(BATCH * Self.SCRATCH_IN_DIM):
            a_storage.append(0)
        var mu_storage = List[Scalar[dtype]](
            capacity=BATCH * Self.SCRATCH_OUT_DIM
        )
        for _ in range(BATCH * Self.SCRATCH_OUT_DIM):
            mu_storage.append(0)

        var a_full = tile[BATCH * Self.SCRATCH_IN_DIM](a_storage)
        var a_ptr = a_full.ptr
        var mu_full = tile[BATCH * Self.SCRATCH_OUT_DIM](mu_storage)
        var mu_ptr = mu_full.ptr

        comptime for i in range(Self.N):
            var li_p = tile[Self.block_types[i].PARAM_SIZE](params.ptr.unsafe_offset(Self._param_offset[i]()))
            var li_a = tile[BATCH, Self.block_types[i].IN_DIM](a_ptr.unsafe_offset(BATCH * Self._in_offset[i]()))

            comptime if i == 0:
                # Input: external x_in
                var li_x = tile[BATCH, Self.block_types[i].IN_DIM](x_in.ptr)
                comptime if Self.N == 1:
                    # Single-block net: write μ directly to external output
                    var li_mu_out = tile[BATCH, Self.block_types[i].OUT_DIM](output.ptr)
                    Self.block_types[i].predict[BATCH, dtype](
                        li_x, li_p, li_mu_out, li_a
                    )
                else:
                    var li_mu = tile[BATCH, Self.block_types[i].OUT_DIM](mu_ptr.unsafe_offset(BATCH * Self._out_offset[i]()))
                    Self.block_types[i].predict[BATCH, dtype](
                        li_x, li_p, li_mu, li_a
                    )
            elif i == Self.N - 1:
                # Last block: input is previous block's μ; write to external output
                var li_x_below = tile[BATCH, Self.block_types[i].IN_DIM](mu_ptr.unsafe_offset(BATCH * Self._out_offset[i - 1]()))
                var li_mu_out = tile[BATCH, Self.block_types[i].OUT_DIM](output.ptr)
                Self.block_types[i].predict[BATCH, dtype](
                    li_x_below, li_p, li_mu_out, li_a
                )
            else:
                # Middle: input is previous block's μ; output to next slot
                var li_x_below = tile[BATCH, Self.block_types[i].IN_DIM](mu_ptr.unsafe_offset(BATCH * Self._out_offset[i - 1]()))
                var li_mu = tile[BATCH, Self.block_types[i].OUT_DIM](mu_ptr.unsafe_offset(BATCH * Self._out_offset[i]()))
                Self.block_types[i].predict[BATCH, dtype](
                    li_x_below, li_p, li_mu, li_a
                )

    # =========================================================================
    # Initialize latents via forward sweep: x_l ← μ_l for l=1..N-1
    # =========================================================================

    @staticmethod
    def init_latents[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_in: Tile[dtype, BATCH, Self.IN_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        latents: Tile[dtype, BATCH, Self.LATENT_DIM],
    ):
        """Forward sweep: x_l ← μ_l for l = 1..N-1 (interior latents).

        After this call, latents[block_l_offset .. block_l_offset+OUT_DIM]
        holds μ_l = the prediction of the l-th interior latent computed
        from x_{l-1} via block_{l-1}.predict. The forward sweep is the
        Bogacz-recommended init for inference.
        """
        # Use latents itself as the rolling x_below buffer for blocks 1..N-2.
        # For block 0, x_below = x_in (external).
        # For block i >= 1, x_below = latents[block_{i-1}_offset...] (= μ_{i-1}).
        # The OUTPUT of block i, for i in 0..N-2, is written into
        # latents[block_i_offset...] as the latent x_i.
        # The LAST block (readout) is NOT written to latents — its output
        # (= μ_N) is computed during eps_compute against the supervised target,
        # so it doesn't need persistent storage. (Trainer handles that.)
        #
        # We need a transient `a_below` per block for the predict call. Allocate
        # one shared scratch buffer per kind.

        var a_storage = List[Scalar[dtype]](
            capacity=BATCH * Self.SCRATCH_IN_DIM
        )
        for _ in range(BATCH * Self.SCRATCH_IN_DIM):
            a_storage.append(0)
        var a_full = tile[BATCH * Self.SCRATCH_IN_DIM](a_storage)
        var a_ptr = a_full.ptr

        comptime for i in range(Self.N - 1):  # only fill interior latents
            var li_p = tile[Self.block_types[i].PARAM_SIZE](params.ptr.unsafe_offset(Self._param_offset[i]()))
            var li_a = tile[BATCH, Self.block_types[i].IN_DIM](a_ptr.unsafe_offset(BATCH * Self._in_offset[i]()))

            # Output: latents[..., latent_offset[i]..+OUT_DIM]  (= x_{i+1})
            var li_mu = tile[BATCH, Self.block_types[i].OUT_DIM](latents.ptr.unsafe_offset(BATCH * Self._latent_offset[i]()))

            comptime if i == 0:
                var li_x = tile[BATCH, Self.block_types[i].IN_DIM](x_in.ptr)
                Self.block_types[i].predict[BATCH, dtype](
                    li_x, li_p, li_mu, li_a
                )
            else:
                # x_below = previous latent: latents[..., latent_offset[i-1]...]
                var li_x_below = tile[BATCH, Self.block_types[i].IN_DIM](latents.ptr.unsafe_offset(BATCH * Self._latent_offset[i - 1]()))
                Self.block_types[i].predict[BATCH, dtype](
                    li_x_below, li_p, li_mu, li_a
                )

    # =========================================================================
    # GPU paths
    # =========================================================================

    @staticmethod
    def forward_eval_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_in: Tile[dtype, BATCH, Self.IN_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        output: Tile[dtype, BATCH, Self.OUT_DIM],
        mu_buf: Tile[dtype, BATCH, Self.SCRATCH_OUT_DIM],
        a_buf: Tile[dtype, BATCH, Self.SCRATCH_IN_DIM],
    ) raises:
        """GPU feedforward through all blocks. Caller-owned scratch buffers."""
        comptime for i in range(Self.N):
            var li_p = tile[Self.block_types[i].PARAM_SIZE](params.ptr.unsafe_offset(Self._param_offset[i]()))
            var li_a = tile[BATCH, Self.block_types[i].IN_DIM](a_buf.ptr.unsafe_offset(BATCH * Self._in_offset[i]()))

            comptime if i == 0:
                var li_x = tile[BATCH, Self.block_types[i].IN_DIM](x_in.ptr)
                comptime if Self.N == 1:
                    var li_mu_out = tile[BATCH, Self.block_types[i].OUT_DIM](output.ptr)
                    Self.block_types[i].predict_gpu[BATCH, dtype](
                        ctx, li_x, li_p, li_mu_out, li_a
                    )
                else:
                    var li_mu = tile[BATCH, Self.block_types[i].OUT_DIM](mu_buf.ptr.unsafe_offset(BATCH * Self._out_offset[i]()))
                    Self.block_types[i].predict_gpu[BATCH, dtype](
                        ctx, li_x, li_p, li_mu, li_a
                    )
            elif i == Self.N - 1:
                var li_x_below = tile[BATCH, Self.block_types[i].IN_DIM](mu_buf.ptr.unsafe_offset(BATCH * Self._out_offset[i - 1]()))
                var li_mu_out = tile[BATCH, Self.block_types[i].OUT_DIM](output.ptr)
                Self.block_types[i].predict_gpu[BATCH, dtype](
                    ctx, li_x_below, li_p, li_mu_out, li_a
                )
            else:
                var li_x_below = tile[BATCH, Self.block_types[i].IN_DIM](mu_buf.ptr.unsafe_offset(BATCH * Self._out_offset[i - 1]()))
                var li_mu = tile[BATCH, Self.block_types[i].OUT_DIM](mu_buf.ptr.unsafe_offset(BATCH * Self._out_offset[i]()))
                Self.block_types[i].predict_gpu[BATCH, dtype](
                    ctx, li_x_below, li_p, li_mu, li_a
                )

    @staticmethod
    def init_latents_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_in: Tile[dtype, BATCH, Self.IN_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        latents: Tile[dtype, BATCH, Self.LATENT_DIM],
        a_buf: Tile[dtype, BATCH, Self.SCRATCH_IN_DIM],
    ) raises:
        """Forward sweep on GPU: x_l ← μ_l for l = 1..N-1."""
        comptime for i in range(Self.N - 1):
            var li_p = tile[Self.block_types[i].PARAM_SIZE](params.ptr.unsafe_offset(Self._param_offset[i]()))
            var li_a = tile[BATCH, Self.block_types[i].IN_DIM](a_buf.ptr.unsafe_offset(BATCH * Self._in_offset[i]()))
            var li_mu = tile[BATCH, Self.block_types[i].OUT_DIM](latents.ptr.unsafe_offset(BATCH * Self._latent_offset[i]()))

            comptime if i == 0:
                var li_x = tile[BATCH, Self.block_types[i].IN_DIM](x_in.ptr)
                Self.block_types[i].predict_gpu[BATCH, dtype](
                    ctx, li_x, li_p, li_mu, li_a
                )
            else:
                var li_x_below = tile[BATCH, Self.block_types[i].IN_DIM](latents.ptr.unsafe_offset(BATCH * Self._latent_offset[i - 1]()))
                Self.block_types[i].predict_gpu[BATCH, dtype](
                    ctx, li_x_below, li_p, li_mu, li_a
                )
