"""Split-K GEMM on a caller-owned workspace.

⚠ LINE NUMBERS ARE PINNED TO MAX v26.6.0, the release pixi ships, and the
tree checked out at `references/modular-max-v26.6.0/max/kernels/src/`. Do not
look them up on `main`: the file moves every week (the split-K allocation was
at :2714 there on 2026-10-09). Main was re-read on that date and nothing below
had changed except the line numbers — still one allocation per call on every
NVIDIA path, and `select_config` byte-identical. When pixi moves to a new
release, re-check the claims against that release's tree and bump this pin.

WHY THIS EXISTS
---------------
`linalg.matmul` allocates its split-K reduction workspace on EVERY call
(`linalg/matmul/gpu/__init__.mojo:1890`, freed at `:1960`). That costs a
`cuMemAlloc_v2`/`cuMemFree_v2` pair per GEMM, and — because a synchronous
driver allocation is illegal inside a capture region — it makes any training
step containing a split-K GEMM impossible to put in a CUDA graph.
`docs/MODULAR_MATMUL_ALLOC_REPORT.md` Measurement 5 concluded that was a hard
block. It is not: every symbol `multistage_gemm`'s split-K branch uses is
exported from the shipped `linalg` package, so we can run that branch against
a workspace we own.

Measured, RTX 5090, `[256 x 2592] @ [2592 x 256]` (an ACT dW shape):

    linalg.matmul                     83.30 us    allocates
    ours, select_config's P=2         60.62 us    1.37x, bit-identical
    ours, autotuned P=33              11.11 us    7.50x, rel err 1.7e-06
    inside a CUDA graph               captures (2 nodes), replays, exact

The partition count turns out to matter more than the allocation: across
ACT's dW shapes the slice goes 3.05 -> 1.58 ms/step by owning the workspace,
and 3.05 -> 0.35 ms/step once P is tuned. See
`benchmarks/bench_splitk_act_dw_sweep.mojo` and
`docs/MODULAR_SOURCE_DEEP_DIVE.md`.

WHICH HARDWARE THIS IS FOR
--------------------------
`MatmulKernels.ampere_*` are CONFIG NAMES (a block tile shape and a stage
count), not architecture-locked kernels — `multistage_gemm_split_k_kernel` is
generic and MAX itself runs it on everything from sm_80 up. What varies is
whether `_matmul_gpu` REACHES that path at all:

    A100 (sm_80), L4/A10 (sm_89)   multistage       <- ours applies
    RTX 50xx (sm_120/121)          multistage       <- ours applies
    H100 (sm_90)                   matmul_dispatch_sm90 first, and only falls
                                   through when it returns a False status,
                                   which we cannot know from the outside
    B200/GB200 (sm_100/101/103)    matmul_dispatch_sm100, returns
                                   UNCONDITIONALLY — multistage is never reached
    AMD, Apple                     their own branches; no split-K here

`_has_blackwell_tcgen05()` is the sm_100/101/103 test, and notably does NOT
include sm_120 — consumer Blackwell has no tcgen05, which is exactly why an
RTX 5090 lands on the multistage path and why this work applies to it.

So `splitk_path_applies()` excludes H100 and datacenter Blackwell, and we fall
back to `linalg.matmul` unchanged there. This is deliberately conservative: we
intervene only where we can show MAX would itself have chosen
`multistage_gemm` with a partitioned K.

⚠ THAT EXCLUSION IS NOT BECAUSE THOSE PATHS ARE IMMUNE. They are not — the
allocate-per-call pattern is MAX-wide, and on H100 and B200 it is WORSE than
on the multistage path, two buffers and a memset instead of one buffer:

    sm_80/89/120   multistage_gemm                     `work_space_data`
                   (matmul/gpu/__init__.mojo:1890)     freed at :1960

    sm_90 (H100)   warp_specialize_gemm_with_          `workspace_data` +
                   multicasting_splitk                 `locks_ptr`, plus an
                   (sm90/matmul.mojo:707, :870, :885)  enqueue_memset; both
                                                       freed with `_ = x^`

    sm_100 (B200)  _blackwell_matmul_tma_umma_         `reduction_workspace` +
                   warp_specialized_split_k            `locks_buffer`, plus an
                   (sm100_structured/default/          enqueue_memset; both
                    matmul.mojo:500, :687, :703)       freed with `_ = x^`

So both consequences carry to every NVIDIA architecture: the per-call
allocator tax, and — since a synchronous driver allocation is illegal inside a
capture region on any part — the CUDA-graph block. An H100 or B200 training
step containing a split-K GEMM is as uncapturable as ours was.

We exclude them for a different reason: on those parts `_matmul_gpu` reaches a
DIFFERENT and better kernel (warp-specialised, TMA/UMMA, its own scheduler),
and substituting the generic multistage one would be a regression. Fixing them
the same way is possible — `warp_specialize_gemm_with_multicasting_splitk` and
`SplitKTileScheduler` are exported from the shipped package too, verified — but
it is a much larger port (TMA descriptors, a locks buffer, the scheduler's
state) and we have no H100 or B200 to validate it on. Documented rather than
attempted.
"""

from std.math import ceildiv, align_up
from std.os import getenv
from std.sys import has_nvidia_gpu_accelerator
from std.sys.info import _has_blackwell_tcgen05

from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import H100, GPUInfo
from layout import Layout, LayoutTensor, TileTensor, RuntimeLayout, UNKNOWN_VALUE
from layout import row_major, Coord
from std.utils.index import Index

from linalg.matmul import matmul as max_matmul
from linalg.utils_gpu import MatmulConfig, MatmulKernels, select_config
from linalg.matmul.gpu import multistage_gemm_split_k_kernel, split_k_reduce

from noeira.nn.constants import DT
from .tensor import Tensor, TensorImpl


@always_inline
def splitk_path_applies[info: GPUInfo]() -> Bool:
    """True on the parts where MAX's own dispatch reaches `multistage_gemm`.

    Excludes H100 (tries `matmul_dispatch_sm90` first) and sm_100/101/103
    datacenter Blackwell (`matmul_dispatch_sm100` returns unconditionally).
    See the module docstring.
    """
    comptime not_h100 = info != H100
    return (
        has_nvidia_gpu_accelerator()
        and not _has_blackwell_tcgen05()
        and not_h100
    )


@always_inline
def multistage_shape_ok(m: Int, n: Int, k: Int) -> Bool:
    """Would MAX's own dispatch hand THIS shape to `multistage_gemm`?

    This is `multi_gemm_cond` from `matmul/gpu/__init__.mojo:608` (v26.6.0),
    reduced to its generic-NVIDIA form (both `h100_matmul_cond` and
    `amdgpu_matmul_cond` are False on the parts `splitk_path_applies` admits):

        m > 1  and  n % 128 == 0  and  k % 32 == 0  and  k >= 128

    MAX also requires N and K to be STATIC (`has_static_NK`, `:618`) before
    it considers multistage at all; a dynamic one goes to the vendor
    fallback. That is not checked here because it cannot fail at this call:
    `splitk_gemm` already needs a static N (`layout.shape[1].value()`), and
    every Module that routes through it has compile-time dims.

    ⚠ CHECKING THIS IS NOT OPTIONAL, and `select_config` is not a substitute.
    `select_config` is a CONFIG CHOOSER that MAX only reaches AFTER
    `multi_gemm_cond` passes; it will happily return `num_k_partitions > 1` for
    a shape the multistage kernel is never given. A shape that fails the
    condition goes to the VENDOR fallback (cuBLASLt), and substituting the
    multistage split-K kernel for it produces a WRONG ANSWER, not a slow one.

    Measured: `Conv2D`'s ResNet18 stem dW is `[64 x 307200] @ [307200 x 160]`.
    `CPAD` pads the im2col column count to a multiple of 32 (the FORWARD's
    contraction alignment), and 160 % 128 = 32 — so MAX takes cuBLAS. Routing
    it through split-K anyway gave rel err 1.02e-3 against the plain GEMM,
    which is what `tests/nn/test_conv2d_splitk_dw_gpu.mojo` caught.

    `Linear` satisfies it by construction (`N_PAD_TO = 128`), which is the only
    reason that integration was correct before this check existed.
    """
    return m > 1 and n % 128 == 0 and k % 32 == 0 and k >= 128


@always_inline
def partitions_legal(K: Int, P: Int, BK: Int) -> Bool:
    """Whether `P` is a usable partition count for this K.

    `multistage_gemm_split_k_kernel` carves K with
    `LayoutTensor.split[axis, split_alignment=BK]`, whose body is
    (`layout/layout_tensor.mojo:3864`, v26.6.0):

        part   = align_up(K // P, BK)          <- FLOOR divide, then align UP
        size_i = min(part, K - i * part)
        ptr_i  = base + i * part * stride

    There are TWO ways that goes wrong and only one of them is loud:

        OVERRUN        (P-1)*part >= K   the last block starts past the end of
                                         A and B with a NEGATIVE size
                                         -> CUDA_ERROR_ILLEGAL_ADDRESS
        UNDERCOVERAGE  P*part < K        the partitions fail to SPAN K, the
                                         tail is never accumulated, and the
                                         GEMM SILENTLY RETURNS A WRONG ANSWER

    Both measured on a 5090 at K=2592, BK=16: P=16 and 20 overrun, P=23 and 40
    undercover. P=40 drops 32 of 2592 contraction elements and the observed
    relative error was 0.0123436 against 32/2592 = 0.0123457 — the error IS
    the fraction of K dropped.

    ⚠ Legality is NOT MONOTONE in P. At that K the legal set is 1..15, 17, 18,
    21, 24, 27, 33, 41. Anything scanning for a good P must CONTINUE past a
    failure, not break: stopping at the first one returns 15 and misses the
    optimum by a wide margin.

    ⚠ MAX checks neither rule. `select_config` breaks on `K < P * bk` and is
    otherwise saved only by its separate `min_k_partition = 1024` test keeping
    P low for unrelated reasons. Small P is safe on the multistage path only
    because the dispatch gate already forces `k % 32 == 0`; nothing verifies
    it. MAX's own `TUNE_NUM_K_PARTITIONS` autotune define can reach both.
    """
    if P < 1 or K < 1 or BK < 1:
        return False
    if P == 1:
        return True
    var part = align_up(K // P, BK)
    if part < 1:
        return False
    return (P - 1) * part < K and P * part >= K


def choose_partitions(
    M: Int, N: Int, K: Int, BM: Int, BN: Int, BK: Int, sm_count: Int,
    max_p: Int = 48,
) -> Int:
    """Pick P without measuring: fill the machine once, and no more.

    Blocks launched is `tiles * P` with `tiles = ceildiv(M,BM)*ceildiv(N,BN)`,
    and the sweep puts the knee exactly at the wave boundary — on a 170-SM
    5090, 128 blocks are 38% FASTER than 192 for identical work.

    ⚠ MEASURED ACCURACY: exact on the 16-tile ACT shapes (picks P=10, the
    measured optimum) and ~7% slow on the 4-tile one (picks P=41 / 11.76 us
    where P=33 / 11.11 us wins). It models only the GEMM side; the reduce
    reads `P * M * N` and so grows LINEARLY in P, which is ~5.8 us of an
    11.11 us total at P=33. Modelling that second term from three shapes would
    be curve-fitting.

    Prefer `autotune_partitions` wherever a measurement is possible — the dW
    shapes are compile-time constants, so one sweep at init settles it. This
    is the fallback, and it is never WRONG, only up to ~7% slow.

    Returns 1 when nothing better is available, meaning "do not split".
    """
    var tiles = ceildiv(M, BM) * ceildiv(N, BN)
    var best = 1
    for P in range(2, max_p + 1):
        if tiles * P > sm_count:
            continue
        if not partitions_legal(K, P, BK):
            continue          # overruns, or silently undercovers
        if P > best:
            best = P
    return best


def splitk_gemm[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    ws_type: DType, //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    num_partitions: Int,
    mut ws: TensorImpl[ws_type],
    ctx: DeviceContext,
) raises:
    """`multistage_gemm`'s split-K branch, on `ws` instead of a fresh buffer.

    Mirrors `matmul/gpu/__init__.mojo:1884-1960` (v26.6.0) with its
    `enqueue_create_buffer` / `_ = work_space_data^` pair removed. `ws` must
    already hold at least `num_partitions * M * N` elements — size it with
    `ensure_gpu` on an eager step, never inside a capture region.

    ⚠ `ws`, and every operand, must outlive the LAST REPLAY of any graph this
    is captured into: a CUDA graph holds RAW POINTERS to all of them. Owning
    the workspace as a field of the Module (which outlives the trainer's
    graph) is what makes that true by construction; a scope-local workspace
    captures fine and then faults on replay.

    The comptime `config` supplies the TILE SHAPE only. On NVIDIA the
    partition count is a runtime kernel argument — the kernel slices its
    workspace by `block_idx.z * M * N` — which is why MAX passes a static
    `kernels.ampere_*` beside a runtime config whose `num_k_partitions`
    differs, and why `MatmulConfig.__eq__` compares only `block_tile_shape`
    and `num_pipeline_stages`. One instantiation covers every P.
    """
    comptime assert ws_type == config.split_k_reduction_type, (
        "workspace dtype must equal config.split_k_reduction_type"
    )
    var tensor_c = c.to_layout_tensor()
    var tensor_a = a.to_layout_tensor()
    var tensor_b = b.to_layout_tensor()
    var M = tensor_c.dim[0]()
    var N = tensor_c.dim[1]()
    var K_dim = Int(tensor_a.dim[1]())
    comptime BK = config.block_tile_shape[2]

    if not multistage_shape_ok(Int(M), Int(N), K_dim):
        raise Error(
            "this shape fails multi_gemm_cond, so MAX would route it to the"
            " VENDOR fallback, not to multistage — the split-K kernel would"
            " return a wrong answer here. See multistage_shape_ok"
        )
    if not partitions_legal(K_dim, num_partitions, BK):
        raise Error(
            "illegal num_k_partitions for this K: it either overruns the"
            " operands (crash) or fails to span K (silent wrong answer) —"
            " see partitions_legal"
        )
    if num_partitions * M * N > ws.n:
        raise Error(
            "split-K workspace too small; call ensure_gpu before capture"
        )

    comptime static_N = tensor_c.layout.shape[1].value()
    comptime ws_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, static_N)
    var ws_rt_layout = RuntimeLayout[ws_layout].row_major(
        Index(num_partitions, M, N)
    )
    var ws_lt = LayoutTensor[ws_type, ws_layout, MutAnyOrigin](
        ws.dev.value(), ws_rt_layout
    )

    comptime kern = multistage_gemm_split_k_kernel[
        c_type, tensor_c.layout,
        a_type, tensor_a.layout,
        b_type, tensor_b.layout,
        ws_type, ws_lt.layout,
        transpose_b, config, None,
    ]

    ctx.enqueue_function[kern](
        tensor_c, tensor_a, tensor_b, ws_lt, Int32(num_partitions),
        grid_dim=(
            ceildiv(N, config.block_tile_shape[1]),
            ceildiv(M, config.block_tile_shape[0]),
            num_partitions,
        ),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
        ),
    )

    var ws_tt = TileTensor(
        ws.dev.value(), row_major(Coord(num_partitions, M, N))
    )
    split_k_reduce(c, ws_tt, ctx)


# ── shared split-K routing ─────────────────────────────────────────────────
# Every Module that routes a weight-gradient GEMM through split-K needs the
# same two decisions: WHICH tile config to instantiate, and HOW MANY
# partitions to use. Both were written inline in `Linear` and then copied
# verbatim into `Conv2D`, and the next four sites would have made six copies
# of a rule whose failure mode is a SILENT WRONG GRADIENT. They live here
# once instead; a Module supplies only its own shape.


def dispatch_splitk_gemm[transpose_b: Bool = False](
    c: TileTensor[mut=True, DT, ...],
    a: TileTensor[mut=False, DT, ...],
    b: TileTensor[mut=False, DT, ...],
    M: Int,
    N: Int,
    K: Int,
    num_partitions: Int,
    mut ws: Tensor,
    ctx: DeviceContext,
) raises:
    """`c = a @ b` (or `a @ bᵀ`) through the caller-owned workspace `ws`.

    ⚠ NOT ONLY FOR dW. The first version of this was, and that assumption cost
    a CUDA-graph capture: `Conv2D`'s FORWARD is `col[BS, CPAD] @ wᵀ[OC, CPAD]`,
    whose contraction is `CPAD = IC * K * K` — 2304 for a 3x3 at IC=256 — and
    once that clears `select_config`'s `K >= 2048` floor MAX partitions it and
    allocates, exactly like a dW. Any `max_matmul` inside a captured region can
    do this; the shape is what decides, not which gradient it belongs to.

    The comptime config must be the tile `select_config` chose — it sets the
    grid and the shared-memory request, so a mismatch is wrong launch geometry
    rather than a slowdown. MAX picks between three; on anything but an A100
    `ampere_256x128_3` is excluded by its own shared-memory check, but all
    three are handled so an A100 does not fall off the end.

    fp32 only. A bf16-flow dW (`LinearAct`/`Embedding`'s AMP path) is a
    DIFFERENT instantiation — `MatmulKernels[bfloat16, bfloat16, float32]`,
    and `_bk_base` returns a different BK for it, which changes both the tile
    config and `partitions_legal`'s alignment. Those sites stay on
    `max_matmul` until there is a bf16 gate to validate them against.
    """
    comptime kernels = MatmulKernels[DT, DT, DT, transpose_b]()
    var picked = select_config[DT, DT, DT, transpose_b](M, N, K, ctx)
    if picked == kernels.ampere_256x64_4:
        splitk_gemm[transpose_b=transpose_b, config = kernels.ampere_256x64_4](
            c, a, b, num_partitions, ws, ctx
        )
    elif picked == kernels.ampere_256x128_3:
        splitk_gemm[
            transpose_b=transpose_b, config = kernels.ampere_256x128_3
        ](c, a, b, num_partitions, ws, ctx)
    else:
        splitk_gemm[
            transpose_b=transpose_b, config = kernels.ampere_128x128_4
        ](c, a, b, num_partitions, ws, ctx)


def decide_partitions[transpose_b: Bool = False](
    M: Int, N: Int, K: Int, ctx: DeviceContext
) raises -> Int:
    """Partition count for a `[M, K] @ [K, N]` dW GEMM. 1 = do not split.

    Call this ONCE, on an eager step, and cache the answer: `grid_dim` derives
    from P and is baked into a captured graph, so a P that drifted between
    replays would be a wrong launch, not a slow one. Size the workspace to
    `P * M * N` in the same eager step — it can never grow inside a capture
    region.

    `NOEIRA_SPLITK=0` pins it to 1 (plain `max_matmul`), for bisecting a
    suspected split-K problem against a known-good run without a rebuild.

    ⚠ Note what this does NOT mean when it returns 1. It is deliberately
    conservative: we intervene only where MAX would ITSELF have partitioned K,
    so a shape it would have run as a plain multistage GEMM stays exactly
    that. `select_config` needs `K // P >= 1024` for P=2, i.e. **K >= 2048**,
    on top of `multistage_shape_ok`. Most RL trunks run a dW whose K is the
    minibatch (256-1024), so on those this returns 1 and the Module's split-K
    plumbing is inert — correct, and worth nothing until the batch grows.
    """
    if getenv("NOEIRA_SPLITK", "1") == "0":
        return 1
    # ⚠ multi_gemm_cond FIRST. `select_config` is only reached after it in
    # MAX's dispatch and will otherwise hand back a partition count for a
    # shape the multistage kernel never sees — a wrong answer, not a
    # slowdown. See multistage_shape_ok.
    if not multistage_shape_ok(M, N, K):
        return 1
    # `multi_gemm_cond` does not depend on `transpose_b` (source: it tests
    # only m/n/k), and neither does `select_config`'s loop — but the CONFIG it
    # returns is typed on it, so thread it rather than rebind.
    var picked = select_config[DT, DT, DT, transpose_b](M, N, K, ctx)
    if picked.num_k_partitions <= 1:
        return 1

    comptime sm_count = ctx.default_device_info.sm_count
    var bt = picked.block_tile_shape
    # `choose_partitions` fills one wave, which measured exact on the 16-tile
    # ACT shapes and ~7% slow on the 4-tile one.
    var want = choose_partitions(M, N, K, bt[0], bt[1], bt[2], sm_count)
    # Fall back to MAX's own P if the wave rule found nothing better; it is
    # legal by construction here, but check anyway — undercoverage is silent,
    # and a silent wrong gradient is the worst outcome available.
    if want <= 1:
        want = picked.num_k_partitions
    if not partitions_legal(K, want, bt[2]):
        return 1
    return want
