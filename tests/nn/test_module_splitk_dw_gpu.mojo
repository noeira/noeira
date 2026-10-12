"""Does routing the OTHER dW GEMMs through our own split-K workspace change
the gradient?

`Linear` and `Conv2D` got this first (see their two gates). The same
`grad_w = Xᵀ @ go` shape — long K, small M and N — occurs in four more
Modules, and each was still on `linalg.matmul`'s allocate-per-call split-K:

    LinearAct         [IN, B] @ [B, OUT]      backs LinearReLU / LinearTanh /
                                              LinearMish / LinearSwish /
                                              LinearSigmoid, i.e. every RL trunk
    NoisyLinear       [IN, B] @ [B, OUT]      DQN / C51 / Rainbow heads
    Embedding         [VOCAB, B] @ [B, ED]    GPT — the longest K in the repo,
                                              `Tokenwise` flattens BATCH*SEQ_LEN
    Conv2DTranspose   [IC, BS] @ [BS, COLT]   DreamerV3 decoder, BS = B * H * W

This is the in-process A/B of exactly that change: two identical Modules, the
same weights and the same input, one pinned to `_sk_p = 1` so it takes plain
`max_matmul`. Their weight gradient must agree.

⚠ VACUITY IS THE FAILURE MODE HERE, and the shape gate is the sharp edge.
`multi_gemm_cond` needs `N % 128 == 0`, and unlike `Linear` (whose `N_PAD_TO`
is 128 by construction) NONE of these four pads N — `LinearAct`'s N is `OUT_`
as written, `Embedding`'s is `EMBED_DIM_`, `Conv2DTranspose`'s is `OC * K²`.
A shape that fails it must NOT split: MAX routes it to cuBLAS and the
multistage kernel would return a wrong answer, which is how the Conv2D
integration was caught at rel 1.02e-3. So every shape below asserts its
EXPECTED routing decision, split and no-split alike, and the run fails if
nothing split at all.

`LinearAct`'s dW runs on `cublas_gemm` on NVIDIA by default (`CUBLAS_BWD`,
β = 1 straight into the master grad: no split-K workspace, `_sk_p` never
decided), so its section runs only under `-D NN_GEMM_PATH=max` (or `bwdmax`
/ `linmax`) and is skipped, said so, otherwise.

    pixi run -e nvidia mojo run -I . tests/nn/test_module_splitk_dw_gpu.mojo
    pixi run -e nvidia mojo run -I . -D NN_GEMM_PATH=max tests/nn/test_module_splitk_dw_gpu.mojo
    NOEIRA_SPLITK=0 pixi run -e nvidia mojo run -I . tests/nn/...   # both arms plain
"""

from std.math import abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.splitk_gemm import splitk_path_applies
from noeira.nn.core.cublas_gemm import CUBLAS_BWD
from noeira.nn.primitives.linear_act import LinearAct
from noeira.nn.primitives.ops.relu_op import ReLUOp
from noeira.nn.primitives.noisy_linear import NoisyLinear
from noeira.nn.primitives.embedding import Embedding
from noeira.nn.primitives.conv2d_transpose import Conv2DTranspose


# ── shared helpers ─────────────────────────────────────────────────────────


def fill_pattern(mut t: Tensor, n: Int, scale: Scalar[DT], period: Int):
    """Deterministic sign-changing fill. Sign changes matter: a split-K bug
    that drops part of the contraction shows up as a *relative* error equal to
    the fraction of K lost, and that only reads correctly if the summands do
    not all share a sign."""
    for i in range(n):
        t.data[i] = scale * Scalar[DT]((i % period) - (period // 2))


def compare(
    mut a: Tensor, mut b: Tensor, n: Int, ctx: DeviceContext
) raises -> Tuple[Float64, Float64]:
    """(max abs difference, relative to the plain arm's largest magnitude)."""
    a.download(ctx)
    b.download(ctx)
    ctx.synchronize()
    var max_abs = Float64(0)
    var mag = Float64(0)
    for i in range(n):
        var av = Float64(a.data[i])
        var bv = Float64(b.data[i])
        if abs(bv) > mag:
            mag = abs(bv)
        var d = abs(av - bv)
        if d > max_abs:
            max_abs = d
    return (max_abs, (max_abs / mag) if mag > 0.0 else 0.0)


def report(
    label: String, p: Int, expect: Bool, max_abs: Float64, rel: Float64,
    mut n_split: Int,
) raises:
    if p > 1:
        n_split += 1
    # Assert the ROUTING DECISION per shape, not just that something split.
    # Counting alone is what let the Linear integration ship with one of its
    # two dW sites unrouted: every shape took the other one and printed 0.0.
    if (p > 1) != expect:
        raise Error(
            label + ": routing decision does not match the shape's arithmetic:"
            " expected split=" + String(expect) + " got P=" + String(p)
        )
    print(
        "  ", label, "   P=", p, "   max|dW_split - dW_plain|=", max_abs,
        "  rel=", rel, sep="",
    )
    # The reduce sums `P` separate fp32 accumulations, so this is not expected
    # to be bit-identical once P > 1 — but it must stay at fp32 rounding. A
    # dropped K tail (see `partitions_legal`) shows up here as a relative error
    # equal to the fraction of K lost, i.e. 1e-3 or worse.
    if rel > 1e-4:
        raise Error(label + ": split-K dW disagrees beyond fp32 noise")


# ── LinearAct (and therefore LinearReLU / Tanh / Mish / Swish / Sigmoid) ────


def check_linear_act[IN: Int, OUT: Int, B: Int, EXPECT: Bool](
    ctx: DeviceContext, mut n_split: Int
) raises:
    comptime L = LinearAct[IN, OUT, ReLUOp]
    var la = L.make["gpu", INIT=Kaiming](ctx=ctx)
    var lb = L.make["gpu", INIT=Kaiming](ctx=ctx)
    lb._sk_p = 1

    la.weight.val.ensure_host(ctx, L.W_SIZE)
    lb.weight.val.ensure_host(ctx, L.W_SIZE)
    la.bias.val.ensure_host(ctx, L.B_SIZE)
    lb.bias.val.ensure_host(ctx, L.B_SIZE)
    for i in range(L.W_SIZE):
        lb.weight.val.data[i] = la.weight.val.data[i]
    for i in range(L.B_SIZE):
        lb.bias.val.data[i] = la.bias.val.data[i]
    la.weight.val.upload_resident(ctx)
    lb.weight.val.upload_resident(ctx)
    la.bias.val.upload_resident(ctx)
    lb.bias.val.upload_resident(ctx)

    var xa = Tensor.alloc(B * IN)
    var xb = Tensor.alloc(B * IN)
    fill_pattern(xa, B * IN, 0.01, 37)
    fill_pattern(xb, B * IN, 0.01, 37)
    xa.upload(ctx)
    xb.upload(ctx)

    var ya = Tensor.alloc_gpu(ctx, B * OUT)
    var yb = Tensor.alloc_gpu(ctx, B * OUT)
    la.forward["gpu", B](TensorRefs[1](xa), ya, Optional(ctx))
    lb.forward["gpu", B](TensorRefs[1](xb), yb, Optional(ctx))

    var goa = Tensor.alloc(B * OUT)
    var gob = Tensor.alloc(B * OUT)
    fill_pattern(goa, B * OUT, 0.003, 23)
    fill_pattern(gob, B * OUT, 0.003, 23)
    goa.upload(ctx)
    gob.upload(ctx)

    var gia = Tensor.alloc_gpu(ctx, B * IN)
    var gib = Tensor.alloc_gpu(ctx, B * IN)
    la.vjp["gpu", B](TensorRefs[1](xa), goa, TensorRefs[1](gia), Optional(ctx))
    lb.vjp["gpu", B](TensorRefs[1](xb), gob, TensorRefs[1](gib), Optional(ctx))

    var d = compare(la.weight.grd, lb.weight.grd, L.W_SIZE, ctx)
    report(
        "LinearAct IN=" + String(IN) + " OUT=" + String(OUT) + " B=" + String(B),
        la._sk_p, EXPECT, d[0], d[1], n_split,
    )


def check_linear_act_all(ctx: DeviceContext, mut n_split: Int) raises:
    print("== LinearAct: dW = [IN, B] @ [B, OUT] ==")
    # `select_config` needs K // P >= 1024 for even P=2, i.e. K >= 2048.
    check_linear_act[256, 256, 2592, True](ctx, n_split)
    check_linear_act[256, 512, 4096, True](ctx, n_split)
    # ⚠ THIS SHAPE FLIPPED IN a769999b AND THE ASSERTION IS WHAT CAUGHT IT.
    # It was written as a no-split control because `LinearAct` did not pad
    # N, so its dW ran at N = OUT_ = 100 and failed `n % 128`. LinearAct
    # now pads N to 128 (it was on the cuBLAS vendor path twice over), so
    # the dW is [K_PAD=256, N_PAD=128] @ K=2592, which is eligible — P>1 is
    # the CORRECT answer here now, not a regression. The routing assert
    # turned an invisible behaviour change into a loud one; a test that
    # only compared gradients would have stayed green and said nothing.
    check_linear_act[256, 100, 2592, True](ctx, n_split)
    # Below min_k_partition: an ordinary RL minibatch. This is what every
    # SAC / TD3 / DQN trunk actually sees, and it must stay on max_matmul.
    check_linear_act[256, 256, 256, False](ctx, n_split)
    # A `multi_gemm_cond` failure that padding CANNOT fix, replacing the
    # control the line above lost: M and N are now always 128-multiples by
    # construction, so B is the only axis left that can fail the gate.
    # 2590 % 32 = 30, so `k % 32 == 0` fails and this must not split.
    check_linear_act[256, 256, 2590, False](ctx, n_split)


# ── NoisyLinear ────────────────────────────────────────────────────────────


def check_noisy[IN: Int, OUT: Int, B: Int, EXPECT: Bool](
    ctx: DeviceContext, mut n_split: Int
) raises:
    """⚠ Compares `mu_w.grd`, not `sigma_w.grd`.

    `NoisyLinear` resamples its factorized noise on every forward, so the two
    arms hold DIFFERENT noise and their `sigma_w` gradients (which are
    `dW * ε_out ⊗ ε_in`) legitimately differ. `mu_w.grd` is the raw
    `dW = cacheTᵀ @ go` with no noise term in it — that is the GEMM under test,
    and it is comparable across arms.
    """
    comptime L = NoisyLinear[IN, OUT]
    var la = L.make["gpu", INIT=Kaiming](ctx=ctx)
    var lb = L.make["gpu", INIT=Kaiming](ctx=ctx)
    lb._sk_p = 1

    la.mu_w.val.ensure_host(ctx, L.W_SIZE)
    lb.mu_w.val.ensure_host(ctx, L.W_SIZE)
    for i in range(L.W_SIZE):
        lb.mu_w.val.data[i] = la.mu_w.val.data[i]
    la.mu_w.val.upload_resident(ctx)
    lb.mu_w.val.upload_resident(ctx)

    var xa = Tensor.alloc(B * IN)
    var xb = Tensor.alloc(B * IN)
    fill_pattern(xa, B * IN, 0.01, 37)
    fill_pattern(xb, B * IN, 0.01, 37)
    xa.upload(ctx)
    xb.upload(ctx)

    var ya = Tensor.alloc_gpu(ctx, B * OUT)
    var yb = Tensor.alloc_gpu(ctx, B * OUT)
    la.forward["gpu", B](TensorRefs[1](xa), ya, Optional(ctx))
    lb.forward["gpu", B](TensorRefs[1](xb), yb, Optional(ctx))

    var goa = Tensor.alloc(B * OUT)
    var gob = Tensor.alloc(B * OUT)
    fill_pattern(goa, B * OUT, 0.003, 23)
    fill_pattern(gob, B * OUT, 0.003, 23)
    goa.upload(ctx)
    gob.upload(ctx)

    var gia = Tensor.alloc_gpu(ctx, B * IN)
    var gib = Tensor.alloc_gpu(ctx, B * IN)
    la.vjp["gpu", B](TensorRefs[1](xa), goa, TensorRefs[1](gia), Optional(ctx))
    lb.vjp["gpu", B](TensorRefs[1](xb), gob, TensorRefs[1](gib), Optional(ctx))

    var d = compare(la.mu_w.grd, lb.mu_w.grd, L.W_SIZE, ctx)
    report(
        "NoisyLinear IN=" + String(IN) + " OUT=" + String(OUT)
        + " B=" + String(B),
        la._sk_p, EXPECT, d[0], d[1], n_split,
    )


# ── Embedding ──────────────────────────────────────────────────────────────


def check_embedding[VOCAB: Int, ED: Int, B: Int, EXPECT: Bool](
    ctx: DeviceContext, mut n_split: Int
) raises:
    comptime L = Embedding[VOCAB, ED]
    var la = L.make["gpu", INIT=Kaiming](ctx=ctx)
    var lb = L.make["gpu", INIT=Kaiming](ctx=ctx)
    lb._sk_p = 1

    la.weight.val.ensure_host(ctx, L.W_SIZE)
    lb.weight.val.ensure_host(ctx, L.W_SIZE)
    for i in range(L.W_SIZE):
        lb.weight.val.data[i] = la.weight.val.data[i]
    la.weight.val.upload_resident(ctx)
    lb.weight.val.upload_resident(ctx)

    # Embedding's input is a one-hot [B, VOCAB] row per token.
    var xa = Tensor.alloc(B * VOCAB)
    var xb = Tensor.alloc(B * VOCAB)
    for i in range(B * VOCAB):
        xa.data[i] = 0.0
        xb.data[i] = 0.0
    for b in range(B):
        var tok = (b * 7 + 3) % VOCAB
        xa.data[b * VOCAB + tok] = 1.0
        xb.data[b * VOCAB + tok] = 1.0
    xa.upload(ctx)
    xb.upload(ctx)

    var ya = Tensor.alloc_gpu(ctx, B * ED)
    var yb = Tensor.alloc_gpu(ctx, B * ED)
    la.forward["gpu", B](TensorRefs[1](xa), ya, Optional(ctx))
    lb.forward["gpu", B](TensorRefs[1](xb), yb, Optional(ctx))

    var goa = Tensor.alloc(B * ED)
    var gob = Tensor.alloc(B * ED)
    fill_pattern(goa, B * ED, 0.003, 23)
    fill_pattern(gob, B * ED, 0.003, 23)
    goa.upload(ctx)
    gob.upload(ctx)

    var gia = Tensor.alloc_gpu(ctx, B * VOCAB)
    var gib = Tensor.alloc_gpu(ctx, B * VOCAB)
    la.vjp["gpu", B](TensorRefs[1](xa), goa, TensorRefs[1](gia), Optional(ctx))
    lb.vjp["gpu", B](TensorRefs[1](xb), gob, TensorRefs[1](gib), Optional(ctx))

    var d = compare(la.weight.grd, lb.weight.grd, L.W_SIZE, ctx)
    report(
        "Embedding VOCAB=" + String(VOCAB) + " ED=" + String(ED)
        + " B=" + String(B),
        la._sk_p, EXPECT, d[0], d[1], n_split,
    )


# ── Conv2DTranspose ────────────────────────────────────────────────────────


def check_conv_t[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, B: Int,
    EXPECT: Bool,
](ctx: DeviceContext, mut n_split: Int) raises:
    comptime L = Conv2DTranspose[IC, OC, K, S, P, H, W]
    var la = L.make["gpu", INIT=Kaiming](ctx=ctx)
    var lb = L.make["gpu", INIT=Kaiming](ctx=ctx)
    lb._sk_p = 1

    la.weight.val.ensure_host(ctx, L.W_SIZE)
    lb.weight.val.ensure_host(ctx, L.W_SIZE)
    la.bias.val.ensure_host(ctx, L.B_SIZE)
    lb.bias.val.ensure_host(ctx, L.B_SIZE)
    for i in range(L.W_SIZE):
        lb.weight.val.data[i] = la.weight.val.data[i]
    for i in range(L.B_SIZE):
        lb.bias.val.data[i] = la.bias.val.data[i]
    la.weight.val.upload_resident(ctx)
    lb.weight.val.upload_resident(ctx)
    la.bias.val.upload_resident(ctx)
    lb.bias.val.upload_resident(ctx)

    var xa = Tensor.alloc(B * L.IN_FLAT)
    var xb = Tensor.alloc(B * L.IN_FLAT)
    fill_pattern(xa, B * L.IN_FLAT, 0.01, 37)
    fill_pattern(xb, B * L.IN_FLAT, 0.01, 37)
    xa.upload(ctx)
    xb.upload(ctx)

    var ya = Tensor.alloc_gpu(ctx, B * L.OUT_FLAT)
    var yb = Tensor.alloc_gpu(ctx, B * L.OUT_FLAT)
    la.forward["gpu", B](TensorRefs[1](xa), ya, Optional(ctx))
    lb.forward["gpu", B](TensorRefs[1](xb), yb, Optional(ctx))

    var goa = Tensor.alloc(B * L.OUT_FLAT)
    var gob = Tensor.alloc(B * L.OUT_FLAT)
    fill_pattern(goa, B * L.OUT_FLAT, 0.003, 23)
    fill_pattern(gob, B * L.OUT_FLAT, 0.003, 23)
    goa.upload(ctx)
    gob.upload(ctx)

    var gia = Tensor.alloc_gpu(ctx, B * L.IN_FLAT)
    var gib = Tensor.alloc_gpu(ctx, B * L.IN_FLAT)
    la.vjp["gpu", B](TensorRefs[1](xa), goa, TensorRefs[1](gia), Optional(ctx))
    lb.vjp["gpu", B](TensorRefs[1](xb), gob, TensorRefs[1](gib), Optional(ctx))

    var d = compare(la.weight.grd, lb.weight.grd, L.W_SIZE, ctx)
    report(
        "Conv2DTranspose IC=" + String(IC) + " OC=" + String(OC)
        + " COLT=" + String(L.COLT) + " BS=" + String(B * L.SI),
        la._sk_p, EXPECT, d[0], d[1], n_split,
    )


def main() raises:
    comptime if not splitk_path_applies[DeviceContext.default_device_info]():
        print(
            "split-K path does not apply on this device (Apple / AMD / H100 /"
            " sm_100 Blackwell) — these Modules are unchanged here, nothing to"
            " test."
        )
        return

    with DeviceContext() as ctx:
        var n_split = 0

        # Shapes expected to split, per Module run: LinearAct 3, the others 1.
        comptime EXPECT_SPLIT = 3 if CUBLAS_BWD else 6
        comptime N_SHAPES = 6 if CUBLAS_BWD else 11
        comptime if CUBLAS_BWD:
            print(
                "== LinearAct: SKIPPED — its dW runs on cuBLAS here"
                " (CUBLAS_BWD); -D NN_GEMM_PATH=max runs it on split-K =="
            )
        else:
            check_linear_act_all(ctx, n_split)

        print("== NoisyLinear: dW = [IN, B] @ [B, OUT] ==")
        check_noisy[256, 256, 2592, True](ctx, n_split)
        check_noisy[256, 256, 512, False](ctx, n_split)

        print("== Embedding: dW = [VOCAB, B] @ [B, ED] ==")
        # TinyShakespeare char-level GPT: Tokenwise flattens BATCH*SEQ_LEN.
        check_embedding[65, 384, 4096, True](ctx, n_split)
        # ED = 192 fails N % 128 → cuBLAS → must not split.
        check_embedding[65, 192, 4096, False](ctx, n_split)

        print("== Conv2DTranspose: dW = [IC, BS] @ [BS, COLT] ==")
        # COLT = OC * K² = 512, BS = B * H * W = 4096.
        check_conv_t[64, 32, 4, 2, 1, 8, 8, 64, True](ctx, n_split)
        # COLT = 30 * 16 = 480, and 480 % 128 = 96 → cuBLAS → must not split.
        check_conv_t[64, 30, 4, 2, 1, 8, 8, 64, False](ctx, n_split)

        print()
        print("split shapes:", n_split, "of", N_SHAPES)
        if n_split == 0:
            raise Error(
                "NO shape took the split-K path — this run tested nothing."
                " Check select_config's min_k_partition and the device gate"
                " before reading the zeros above as a pass."
            )
        if n_split < EXPECT_SPLIT:
            raise Error(
                "fewer split shapes than expected: all four Modules should"
                " split at their long-K shape, so a shortfall means one of the"
                " dW sites is not routed"
            )
        print("all good")
